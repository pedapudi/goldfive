"""End-to-end test: ``adk web`` driving goldfive over the presentation_agent tree.

Why this test exists
--------------------

Before the ``feat/registry-dispatch-model`` refactor, ``goldfive.wrap(...)``
over a coordinator + AgentTool tree hung indefinitely under ``adk web`` with
a real LLM: the coordinator's LLM looped on AgentTool calls, ``adapter.invoke``
never returned, and the UI saw zero events. PRs #47 and #119 claimed
"orchestration works under adk-web" and shipped with only mock-mode
programmatic tests — the specific adk-web scenario was never validated.

This test closes that gap. It spins up the ADK FastAPI app in-process,
loads ``examples/presentation_agent`` in mock mode, POSTs a prompt via
``/run_sse``, and asserts:

* a single ``plan_submitted`` with the expected four-task plan,
* one ``task_started`` + one terminal (``task_completed``) per task,
* at least one ``task_completed`` fires (proving dispatch progressed —
  the specific pathology we're guarding against was "plan submitted,
  but no task ever completes"),
* per-task dispatch hits the right assignee — ``AgentInvocationStarted``
  for task ``research`` carries ``agent_name == "research_agent"``, not
  the coordinator (the "coordinator-always-runs" regression the Phase 1
  refactor was supposed to kill),
* the ``"Plugin 'goldfive_adk_plugin' registered."`` log count stays
  bounded (the earlier-session bug reported 8+ registrations on a
  single task; under registry-dispatch it should be exactly one per
  reachable agent, i.e. 5 for the presentation tree),
* ``run_completed`` fires,
* the whole test finishes in < 30 seconds wall-clock.

Approach: in-process FastAPI (not subprocess)
---------------------------------------------

The test builds the ADK FastAPI app directly via
``google.adk.cli.fast_api.get_fast_api_app`` with a tmp ``agents_dir``
that contains ``presentation_agent/``, and drives it with
``fastapi.testclient.TestClient``. The subprocess-launched
``adk web --port <ephemeral>`` fallback documented in the test spec is
NOT used because the in-process path turned out to be trivially reliable
(~30 ms for a full SSE round-trip on a developer laptop) and because the
in-process path lets us install the recording sink on the inner goldfive
Runner before the first dispatch.

Mock-mode wiring
----------------

``OPENAI_API_KEY`` and ``HARMONOGRAF_SERVER`` are scrubbed from the env
BEFORE the ``app`` attribute is first resolved; the presentation_agent
module's PEP 562 ``__getattr__`` then picks the mock-mode branch and
builds the ``App`` with ``_MockLlm`` subagents, ``_mock_planner_call_llm``
for the planner, and ``_mock_goal_call_llm`` for the goal deriver. No
network, no LLM, deterministic output.

Known Phase 1 wiring gap: the adapter's ``bind_steerer`` is never
invoked by the executor, so ``SessionContext.steerer`` is ``None`` in
the plugin and ``_emit_observability`` short-circuits before emitting
``AgentInvocationStarted``. The test calls ``adapter.bind_steerer(...)``
explicitly during setup to exercise the observability path — when the
wiring is fixed in the Runner / executor, this workaround can be
deleted without changing the assertions. Noted here so the gap is
visible on every test run.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("fastapi")

# Goldfive's proto extra supplies the pb modules that InMemorySink emits.
# Without it the recording path is unusable. Skip cleanly so the test
# doesn't fail on a bare dev install.
pytest.importorskip("google.protobuf")


# Hard wall-clock budget for the full test body. The earlier-session bug
# made ``adapter.invoke`` hang indefinitely; a timeout is the bluntest
# way to make that failure visible to CI.
WALL_CLOCK_BUDGET_SECONDS = 30.0

# Expected plugin-registration count. The presentation_agent tree has
# 5 reachable agents (coordinator + 4 specialists), so the adapter
# constructs 5 per-agent ``InMemoryRunner`` s and registers the goldfive
# plugin once on each. AgentTool-spawned sub-Runners inherit the parent
# runner's plugin_manager, so they do NOT re-register. A hard cap of 10
# leaves headroom for a small ADK-internal re-registration (e.g. if the
# fast_api server's own Runner ever installed the plugin a second time)
# without admitting the 8+-per-task pathology.
PLUGIN_REGISTRATIONS_MAX = 10


@pytest.fixture
def presentation_agent_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Stage ``examples/presentation_agent`` under a tmp agents_dir.

    Copies the example tree into a temp dir so the ADK ``AgentLoader``
    picks it up by name without polluting the repo. Scrubs env vars
    that would flip ``_build_app`` into live mode (OPENAI_API_KEY) or
    trigger harmonograf connection attempts (HARMONOGRAF_SERVER).

    Yields the temp agents_dir path. Cleanup is handled by the
    ``TemporaryDirectory`` context manager.
    """
    # Mock-mode env: no real LLM, no telemetry server.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HARMONOGRAF_SERVER", raising=False)
    # Pin the topic so plan / goal IDs are stable across reruns. The
    # presentation_agent module reads this at _build_app time.
    monkeypatch.setenv("GOLDFIVE_EXAMPLE_TOPIC", "solar-panels")

    # Reset any cached ``_APP`` so a prior test in the same process
    # doesn't hand back a stale App built under different env vars. The
    # module's ``__getattr__`` re-builds on first access when ``_APP``
    # is ``None``.
    from examples.presentation_agent import agent as agent_mod

    agent_mod._APP = None

    # The ADK AgentLoader discovers agents by iterating ``agents_dir``;
    # stage a copy of the example under a fresh temp root so the
    # loader's ``list_agents`` finds exactly one app.
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "examples",
        "presentation_agent",
    )
    with tempfile.TemporaryDirectory() as agents_dir:
        dst = os.path.join(agents_dir, "presentation_agent")
        shutil.copytree(src, dst, symlinks=True)
        yield agents_dir


def _consume_sse(client: Any, body: dict[str, Any]) -> tuple[int, float]:
    """POST ``/run_sse`` and drain the stream. Returns ``(lines, elapsed)``.

    We don't parse individual SSE events here — the FastAPI stream is
    ADK's view of the run. Goldfive's own event stream is captured via
    the recording sink the test installs on the inner Runner. This
    function exists to drive the outer /run_sse to completion and
    surface wall-clock cost.
    """
    start = time.monotonic()
    lines = 0
    with client.stream("POST", "/run_sse", json=body) as s:
        assert s.status_code == 200, (
            f"/run_sse returned {s.status_code} — expected 200. "
            f"Body preview: {s.read()[:400]!r}"
        )
        for _line in s.iter_lines():
            lines += 1
            # Defensive: if ADK hangs on an LLM loop the executor
            # timeout wouldn't trip until the outer wall-clock budget;
            # bail early if the stream balloons past what mock mode
            # could plausibly produce.
            if lines > 200:
                break
            if time.monotonic() - start > WALL_CLOCK_BUDGET_SECONDS:
                raise AssertionError(
                    "SSE consumption exceeded 30 s — the exact hang "
                    "this test exists to guard against. Check "
                    "adapter.invoke for AgentTool-loop regressions."
                )
    return lines, time.monotonic() - start


def _event_kind(event: Any) -> str:
    """Return the oneof payload kind on a goldfive pb ``Event``."""
    # proto events expose WhichOneof; defensively handle dict events
    # (make_event fallback) too.
    which = getattr(event, "WhichOneof", None)
    if callable(which):
        return str(which("payload") or "")
    if isinstance(event, dict):
        return str(event.get("kind", ""))
    return ""


def test_presentation_agent_e2e_under_adk_web(
    presentation_agent_env: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Full round-trip: ADK FastAPI → /run_sse → goldfive dispatch → events."""
    # Capture ADK's plugin_manager log so we can bound plugin registrations.
    # ADK's loggers live under the ``google_adk.`` prefix (see ADK source:
    # ``logger = logging.getLogger("google_adk." + __name__)`` in plugin_manager).
    caplog.set_level(logging.INFO, logger="google_adk.google.adk.plugins.plugin_manager")

    # Tag the run start BEFORE we touch any ADK import so the budget
    # covers every observable step the CI user cares about.
    wall_clock_start = time.monotonic()

    from fastapi.testclient import TestClient
    from google.adk.cli.fast_api import get_fast_api_app
    from google.adk.cli.utils.agent_loader import AgentLoader

    from goldfive.sinks import InMemorySink

    # Pre-load the agent so we can attach a recording sink to the inner
    # goldfive Runner BEFORE /run_sse first dispatches. Using the same
    # AgentLoader instance for get_fast_api_app ensures the FastAPI server's
    # ``get_runner_async`` hits the cache instead of re-loading (and
    # therefore sees the same wrapped root_agent with our sink attached).
    agents_dir = presentation_agent_env
    agent_loader = AgentLoader(agents_dir=agents_dir)
    agent_or_app = agent_loader.load_agent("presentation_agent")

    # Sanity: the module returned an ADK ``App`` (the mock-mode lazy
    # build path). If this trips it means ``_build_app`` took the live
    # branch because OPENAI_API_KEY leaked — fail loudly rather than
    # fall through to a network call.
    from google.adk.apps.app import App

    assert isinstance(agent_or_app, App), (
        f"expected an adk.App from mock-mode lazy build, got "
        f"{type(agent_or_app).__name__}"
    )
    root_agent = agent_or_app.root_agent

    # Attach the recording sink before any dispatch runs.
    recording_sink = InMemorySink()
    root_agent.add_sink(recording_sink)

    # KNOWN PHASE 1 GAP — bind the adapter's steerer manually.
    #
    # ``ADKAdapter.invoke`` constructs ``SessionContext(steerer=self._steerer, ...)``
    # and the plugin's ``_emit_observability`` short-circuits when
    # ``ctx.steerer is None``. In production the executor / runner never
    # call ``adapter.bind_steerer``, so ``_steerer`` stays None and
    # ``AgentInvocationStarted`` / ``AgentInvocationCompleted`` /
    # ``DelegationObserved`` events never emit. Binding it here
    # exercises the observability path so the per-task-dispatch
    # assertion below can actually fire. When the executor is fixed to
    # call ``adapter.bind_steerer`` during its setup phase (same shape
    # as ``steerer.bind``) this workaround becomes a no-op and should
    # be deleted.
    inner_runner = root_agent.runner
    inner_runner.agent.bind_steerer(inner_runner.steerer)
    # The steerer's ``_sinks`` is populated by ``Runner.run`` via
    # ``steerer.bind(sinks=..., planner=...)`` at the top of each turn,
    # and our sink was appended to ``runner.sinks`` before this point,
    # so there's nothing else to wire — the upcoming ``/run_sse`` call
    # flows through ``run_async_impl`` → ``self._runner.run(...)`` and
    # re-binds the steerer with the full sink list including ours.

    # Build the FastAPI app using the pre-loaded AgentLoader. ``web=False``
    # skips the Angular asset mount — not needed for programmatic testing
    # and avoids a filesystem dependency on the ADK wheel's /browser dir.
    fapp = get_fast_api_app(
        agents_dir=agents_dir,
        agent_loader=agent_loader,
        web=False,
    )
    client = TestClient(fapp)

    # Create a session the exact same way adk web's UI does.
    create_resp = client.post(
        "/apps/presentation_agent/users/user/sessions",
        json={},
    )
    assert create_resp.status_code == 200, (
        f"session create failed: {create_resp.status_code} {create_resp.text}"
    )
    session_id = create_resp.json()["id"]
    assert session_id, "session service returned empty session id"

    # Drive the turn exactly as adk web's frontend drives it.
    run_body: dict[str, Any] = {
        "app_name": "presentation_agent",
        "user_id": "user",
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": "make a presentation about solar panels"}],
        },
        "streaming": False,
    }
    lines, sse_elapsed = _consume_sse(client, run_body)
    assert lines > 0, "/run_sse produced zero SSE events — the exact hang bug"

    # -----------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------
    events = list(recording_sink.events)
    assert events, "recording sink collected zero events from the run"

    kinds = [_event_kind(e) for e in events]

    # 1. Exactly one plan_submitted, with all four expected tasks.
    plan_events = [e for e, k in zip(events, kinds, strict=True) if k == "plan_submitted"]
    assert len(plan_events) == 1, (
        f"expected exactly one plan_submitted; got {len(plan_events)} "
        f"(kinds: {kinds})"
    )
    plan_pb = plan_events[0].plan_submitted.plan
    planned_tasks = {
        t.id: t.assignee_agent_id for t in plan_pb.tasks
    }
    assert planned_tasks == {
        "research": "research_agent",
        "build": "web_developer_agent",
        "review": "reviewer_agent",
        "debug": "debugger_agent",
    }, f"unexpected plan shape: {planned_tasks}"

    # 2. Every planned task must reach a terminal state. Goldfive's
    # terminal task events are task_completed / task_failed / task_cancelled.
    started_ids = {
        e.task_started.task_id for e, k in zip(events, kinds, strict=True) if k == "task_started"
    }
    completed_ids = {
        e.task_completed.task_id
        for e, k in zip(events, kinds, strict=True)
        if k == "task_completed"
    }
    failed_ids = {
        e.task_failed.task_id for e, k in zip(events, kinds, strict=True) if k == "task_failed"
    }
    cancelled_ids = {
        e.task_cancelled.task_id
        for e, k in zip(events, kinds, strict=True)
        if k == "task_cancelled"
    }
    terminal_ids = completed_ids | failed_ids | cancelled_ids

    assert started_ids == set(planned_tasks), (
        f"task_started missing for some planned tasks. "
        f"planned={set(planned_tasks)} started={started_ids}"
    )
    assert set(planned_tasks) <= terminal_ids, (
        f"some planned tasks never reached terminal state. "
        f"planned={set(planned_tasks)} terminal={terminal_ids}"
    )

    # 3. At least one task_completed. If every task failed/cancelled
    # that would satisfy "reached terminal state" but not "dispatch
    # actually progressed" — which is the specific regression we're
    # guarding against.
    assert completed_ids, (
        "no task_completed event fired — plan was submitted but no "
        "task progressed. This is the coordinator-loop pathology the "
        "registry-dispatch refactor was supposed to eliminate."
    )

    # 4. Per-task dispatch hits the right assignee. The Phase 1 refactor's
    # core promise: ``adapter.invoke(task)`` routes to
    # ``registry[task.assignee_agent_id]``, NOT to the coordinator. The
    # ``AgentInvocationStarted`` emitted from the plugin's
    # before_run_callback carries the running agent's name.
    inv_events_by_task: dict[str, str] = {}
    for event, kind in zip(events, kinds, strict=True):
        if kind != "agent_invocation_started":
            continue
        inv = event.agent_invocation_started
        # Only record top-level invocations (parent empty) — nested
        # AgentTool sub-Runner invocations also fire this event but
        # carry a populated parent_invocation_id.
        if inv.parent_invocation_id:
            continue
        if inv.task_id and inv.task_id not in inv_events_by_task:
            inv_events_by_task[inv.task_id] = inv.agent_name
    expected_dispatch = {
        "research": "research_agent",
        "build": "web_developer_agent",
        "review": "reviewer_agent",
        "debug": "debugger_agent",
    }
    for task_id, expected_agent in expected_dispatch.items():
        actual = inv_events_by_task.get(task_id)
        assert actual == expected_agent, (
            f"task {task_id!r} dispatched to {actual!r} — expected "
            f"{expected_agent!r}. Coordinator-routing regression? "
            f"invocations: {inv_events_by_task}"
        )

    # 5. Bounded plugin registrations. The pre-refactor bug surfaced as
    # 8+ registrations for a single task; under registry-dispatch there
    # should be exactly one per reachable agent (5 for this tree).
    registrations = [
        rec
        for rec in caplog.records
        if "Plugin 'goldfive_adk_plugin' registered." in rec.getMessage()
    ]
    assert len(registrations) <= PLUGIN_REGISTRATIONS_MAX, (
        f"goldfive plugin was registered {len(registrations)} times "
        f"(cap {PLUGIN_REGISTRATIONS_MAX}). Earlier-session bug saw "
        f"8+ per-task registrations; this cap catches a resurgence."
    )
    # Positive lower-bound: if we see ZERO the plugin didn't install
    # anywhere and the test is accidentally green.
    assert registrations, (
        "goldfive plugin registered zero times — adapter wiring is "
        "silently broken; every callback would be inactive."
    )

    # 6. run_completed must fire — proves the outer pipeline terminated
    # cleanly rather than hanging.
    assert "run_completed" in kinds, (
        f"run_completed not emitted; kinds={kinds}"
    )

    # 7. Bounded wall-clock for the whole test body.
    total_elapsed = time.monotonic() - wall_clock_start
    assert total_elapsed < WALL_CLOCK_BUDGET_SECONDS, (
        f"test body took {total_elapsed:.1f}s (cap "
        f"{WALL_CLOCK_BUDGET_SECONDS}s). SSE drain was "
        f"{sse_elapsed:.2f}s — the rest is import / app build cost."
    )
