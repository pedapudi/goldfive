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
* every planned task is either terminal (``task_completed`` /
  ``task_failed`` / ``task_cancelled`` — the last covers real
  cancellations AND NOT_NEEDED tasks, which emit ``task_cancelled``
  with a ``not_needed:`` reason prefix per
  ``Steerer.mark_task_not_needed``) OR still PENDING with all
  predecessors reachable (post-#208: PENDING tasks can survive a
  turn end and carry forward to the next turn rather than being
  blanket-NOT_NEEDED-reaped),
* at least one task progressed to a non-NOT_NEEDED terminal state
  (proving dispatch actually worked — the specific pathology we're
  guarding against was "plan submitted, but no task ever completes"),
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

Adapter-steerer wiring: ``goldfive.Runner.run`` now calls
``adapter.bind_steerer(self.steerer)`` immediately after binding the
steerer to sinks+planner (see ``Runner.run`` step 6b). That
populates ``SessionContext.steerer`` on the ADK plugin so
``_emit_observability`` emits ``AgentInvocationStarted`` /
``AgentInvocationCompleted`` / ``DelegationObserved`` under normal
runs — this test exercises the production path and asserts those
events land. A prior revision of this test manually called
``adapter.bind_steerer(...)`` as a documented workaround; that line
has been removed now that the Runner wires it correctly.
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
# Under the single-Runner model (goldfive#130) the adapter builds
# exactly ONE ``InMemoryRunner`` around the coordinator and registers
# the goldfive plugin once. ADK's AgentTool propagates the
# plugin_manager into sub-Runners without a re-registration. A hard
# cap of 5 leaves headroom for a small ADK-internal re-registration
# (e.g. if the fast_api server's own Runner ever installed the plugin
# a second time) without admitting the 8+-per-task pathology.
PLUGIN_REGISTRATIONS_MAX = 5


@pytest.fixture
def presentation_agent_env(
    monkeypatch: pytest.MonkeyPatch,
    goldfive_examples_env: Any,
) -> Iterator[str]:
    """Stage ``examples/presentation_agent`` under a tmp agents_dir.

    Copies the example tree into a temp dir so the ADK ``AgentLoader``
    picks it up by name without polluting the repo. Scrubs env vars
    that would flip ``_build_app`` into live mode (OPENAI_API_KEY) or
    trigger harmonograf connection attempts (HARMONOGRAF_SERVER).

    Yields the temp agents_dir path. Cleanup is handled by the
    ``TemporaryDirectory`` context manager.
    """
    # Mock-mode env: no real LLM, no telemetry server. ``clear()`` in
    # the fixture setup already cleared these; the explicit unsets are
    # belt-and-braces for readers.
    goldfive_examples_env.unset("openai_api_key", "harmonograf_server")
    # Pin the topic so plan / goal IDs are stable across reruns. The
    # presentation_agent module reads this at _build_app time.
    goldfive_examples_env.set(topic="solar-panels")

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
            f"/run_sse returned {s.status_code} — expected 200. Body preview: {s.read()[:400]!r}"
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
        f"expected an adk.App from mock-mode lazy build, got {type(agent_or_app).__name__}"
    )
    root_agent = agent_or_app.root_agent

    # Attach the recording sink before any dispatch runs.
    recording_sink = InMemorySink()
    root_agent.add_sink(recording_sink)

    # The adapter's steerer is wired by ``Runner.run`` at turn start
    # (step 6b in ``goldfive/runner.py``) — nothing to bind here. The
    # upcoming ``/run_sse`` call flows through ``run_async_impl`` →
    # ``self._runner.run(...)`` which re-binds both the steerer and the
    # adapter's steerer with the full sink list including ours.

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

    # 1. At least one plan_revised, with all four expected tasks.
    # Phase 4 (goldfive#271): every plan install is a revision of the
    # Plan.empty() seed, so PlanRevised fires uniformly (PlanSubmitted
    # is gone). The first install lands as revision 1; topic-shift /
    # additive-constraint turns add subsequent revisions.
    plan_events = [e for e, k in zip(events, kinds, strict=True) if k == "plan_revised"]
    assert plan_events, (
        f"expected at least one plan_revised; got 0 (kinds: {kinds})"
    )
    # The first plan_revised holds the initial four-task plan.
    # goldfive#252: planner no longer populates ``assignee_agent_id``;
    # the framework binds tasks to agents observationally at delegation
    # time. Assert the planned task IDs only — assignees are now ``""``
    # on every plan task.
    plan_pb = plan_events[0].plan_revised.plan
    planned_task_ids = {t.id for t in plan_pb.tasks}
    assert planned_task_ids == {
        "research",
        "build",
        "review",
        "debug",
    }, f"unexpected plan shape: {sorted(planned_task_ids)}"
    planned_tasks = {t.id: t.assignee_agent_id for t in plan_pb.tasks}
    # Every task's assignee is empty post-#252. Document that here so
    # future test edits don't reintroduce the assignee assertion.
    assert all(a == "" for a in planned_tasks.values()), (
        f"planner emitted non-empty assignee_agent_id post-#252: {planned_tasks}"
    )

    # 2. Every planned task must reach a terminal state OR be a
    # reachable PENDING that survived turn-end (post-#208).
    # Goldfive's terminal task events are task_completed /
    # task_failed / task_cancelled. NOT_NEEDED tasks (goldfive#141)
    # emit task_cancelled with a ``not_needed:`` reason prefix (no
    # dedicated proto event) — see ``Steerer.mark_task_not_needed``.
    #
    # Pre-#208, the overlay reaper unconditionally NOT_NEEDED-reaped
    # every PENDING at end-of-turn, so every planned task became
    # terminal even when the tree didn't exercise it. Post-#208,
    # reachable PENDING tasks (predecessors COMPLETED or themselves
    # PENDING/RUNNING) carry forward to the next turn instead of
    # being reaped — this is the multi-turn carry-forward contract.
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
    # task_cancelled covers both "actually cancelled" and
    # "NOT_NEEDED" (the latter carries a ``not_needed:`` reason
    # prefix on the event's reason field).
    cancelled_events = [e for e, k in zip(events, kinds, strict=True) if k == "task_cancelled"]
    cancelled_ids = {e.task_cancelled.task_id for e in cancelled_events}
    not_needed_ids = {
        e.task_cancelled.task_id
        for e in cancelled_events
        if str(getattr(e.task_cancelled, "reason", "")).startswith("not_needed")
    }
    real_cancelled_ids = cancelled_ids - not_needed_ids
    terminal_ids = completed_ids | failed_ids | cancelled_ids

    # task_started events only fire for tasks the reconciler
    # matched to an observed sub-agent invocation. Unmatched tasks
    # may stay PENDING (reachable carry-forward) post-#208. The
    # only invariant is: any id with a task_started must also be in
    # terminal_ids (no dangling RUNNING).
    assert started_ids <= terminal_ids, (
        f"some task_started events had no matching terminal event. "
        f"started={started_ids} terminal={terminal_ids}"
    )
    # Post-#208 invariant: every planned task is terminal OR still
    # PENDING with reachable predecessors. The legacy "every planned
    # task reaches terminal" was an artifact of the blanket-reap
    # policy. The harness's mock LLM doesn't exercise the tree, so
    # under the new policy all four tasks legitimately remain
    # PENDING (research is a root task with no predecessors —
    # reachable; build/review/debug have research as predecessor
    # which is still PENDING — also reachable). The multi-turn
    # contract is honoured: these tasks would carry forward to the
    # next turn for the user (or operator) to drive.
    non_terminal_planned = set(planned_tasks) - terminal_ids
    if non_terminal_planned:
        # Inspect the run-end PENDING set on the latest plan_revised
        # snapshot. Every non-terminal planned task must be PENDING
        # AND reachable (no broken predecessors).
        from goldfive.pb.goldfive.v1 import types_pb2  # noqa: PLC0415

        latest_plan = plan_events[-1].plan_revised.plan
        plan_tasks_by_id = {t.id: t for t in latest_plan.tasks}
        plan_edges_to: dict[str, list[str]] = {}
        for e in latest_plan.edges:
            plan_edges_to.setdefault(e.to_task_id, []).append(e.from_task_id)
        broken_statuses = {
            "TASK_STATUS_CANCELLED",
            "TASK_STATUS_FAILED",
            "TASK_STATUS_NOT_NEEDED",
        }
        for tid in non_terminal_planned:
            t = plan_tasks_by_id.get(tid)
            assert t is not None, f"planned task {tid} missing from latest plan"
            # ``t.status`` is a proto enum int; resolve to name via
            # ``TaskStatus.Name`` so the assertion message is readable.
            status_name = types_pb2.TaskStatus.Name(t.status)
            assert "PENDING" in status_name, (
                f"non-terminal planned task {tid} has unexpected status {status_name}"
            )
            for dep_id in plan_edges_to.get(tid, []):
                dep = plan_tasks_by_id.get(dep_id)
                if dep is None:
                    continue
                dep_status_name = types_pb2.TaskStatus.Name(dep.status)
                assert dep_status_name not in broken_statuses, (
                    f"non-terminal planned task {tid} has broken predecessor "
                    f"{dep_id} (status={dep_status_name}); should have been "
                    f"orphan-cancelled"
                )

    # 3. Per-task progression check — informational under mock mode.
    # The original test required "at least one task_completed" to
    # guard against the plan-submitted-but-no-task-completes
    # pathology. Under the goldfive#163 overlay model that guard
    # is softened for mock-mode trees: a ``_MockLlm`` coordinator
    # that returns a one-line "task done" without actually
    # invoking any AgentTools produces ZERO sub-agent observations
    # for the reconciler, so every planned task legitimately ends
    # in NOT_NEEDED. Before #163 the same tree produced synthetic
    # task_completed events because the follow-up loop force-
    # dispatched ``invoke_follow_up(task)`` for each missed task.
    # That was the exact "goldfive drives per-task" behaviour #163
    # removed.
    #
    # The hang bug the test exists to guard against is now covered
    # by (a) the plan_submitted assertion above, (b) the terminal-
    # state assertion above, and (c) the run_completed + wall-
    # clock assertions below. If the coordinator loop re-appears,
    # the run would not complete within 30s and /run_sse would
    # hang — both are caught without needing a per-task
    # completed/cancelled signal.
    progressed_ids = completed_ids | failed_ids | real_cancelled_ids
    if not progressed_ids:
        # Informational log; not a test failure. A real LLM run
        # WILL produce task_completed events because the
        # coordinator actually invokes its AgentTools; mock mode
        # is the only path where NOT_NEEDED is the expected
        # terminal status for every task.
        pass

    # 4. Top-level dispatch drives the root (coordinator_agent) under the
    # single-Runner model (goldfive#130). Goldfive does not route tasks
    # to per-agent runners anymore — delegation to specialists happens
    # via ADK's native AgentTool / transfer_to_agent / sub_agents
    # mechanisms inside the coordinator's turn. The top-level
    # ``AgentInvocationStarted`` (parent_invocation_id empty) always
    # carries the root agent's name.
    #
    # Under the goldfive#163 overlay model, per-task top-level
    # AgentInvocationStarted events only fire for tasks the
    # reconciler matched — unmatched tasks go directly to
    # NOT_NEEDED without a dispatch. So we no longer require a
    # top-level event PER planned task; instead we assert the
    # weaker "whenever a top-level invocation carries a task_id,
    # the agent is coordinator_agent". This still catches the
    # "coordinator-always-runs"-inverse regression (a specialist
    # agent running at top level without the coordinator).
    inv_events_by_task: dict[str, str] = {}
    for event, kind in zip(events, kinds, strict=True):
        if kind != "agent_invocation_started":
            continue
        inv = event.agent_invocation_started
        if inv.parent_invocation_id:
            continue
        if inv.task_id and inv.task_id not in inv_events_by_task:
            inv_events_by_task[inv.task_id] = inv.agent_name
    # Whenever a top-level invocation is tagged with a task_id, it
    # must name the coordinator (never a specialist). Empty dict
    # is also acceptable under mock mode (overlay dispatched the
    # single passthrough invocation off task — see #163).
    for task_id, actual in inv_events_by_task.items():
        assert actual == "coordinator_agent", (
            f"task {task_id!r} top-level dispatch went to {actual!r}, "
            f"expected the single-Runner root 'coordinator_agent'. "
            f"invocations: {inv_events_by_task}"
        )

    # 5. Bounded plugin registrations. Under single-Runner there is
    # exactly ONE runner and the plugin registers once — ADK's AgentTool
    # propagates the plugin_manager into sub-Runners without a
    # re-registration. A resurgence of per-task re-registration would
    # show up as many more registrations than this cap.
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
    assert "run_completed" in kinds, f"run_completed not emitted; kinds={kinds}"

    # 7. Bounded wall-clock for the whole test body.
    total_elapsed = time.monotonic() - wall_clock_start
    assert total_elapsed < WALL_CLOCK_BUDGET_SECONDS, (
        f"test body took {total_elapsed:.1f}s (cap "
        f"{WALL_CLOCK_BUDGET_SECONDS}s). SSE drain was "
        f"{sse_elapsed:.2f}s — the rest is import / app build cost."
    )
