"""Regression tests for the v22 GOAL_DRIFT judge cancellation bug.

V22 trace ``49b0eb10-5636-465d-b96b-9e9d03d91e81`` — at 19:17:19,
``research_panels`` transitioned RUNNING -> COMPLETED, then the
``judge_goal_drift`` span opened and immediately failed with
``CancelledError`` (empty stack, no LLM-call duration recorded). No
``invocation_cancelled`` event fired at that timestamp — the cancel
came from a different propagation path.

Root cause: :meth:`DefaultSteerer._maybe_run_goal_drift_on_task_boundary`
and :meth:`DefaultSteerer.note_agent_turn` ``await``-ed the LLM judge
inline on the agent's ADK invocation task. That task is registered
with the ADK plugin's ``_invocation_tasks`` for cooperative
cancellation, so any sibling drift firing
``request_invocation_cancel(cancel_inflight_task=True)`` could
``task.cancel()`` the agent's task and surface a ``CancelledError``
inside the judge's own ``await call_llm(...)``. PR #320 had already
removed the per-turn drain that was killing reasoning judges; this
sibling family of cancels needed an analogous fix.

Fix: spawn the goal-drift judge as a fire-and-forget task on
:attr:`DefaultSteerer._background_judges` (same pattern as the
reasoning judge in :meth:`_run_judge_background`). asyncio Tasks do
not form a parent-child cancel tree, so a cancel on the agent's
invocation task does NOT propagate to the spawned judge child. The
judge runs to completion regardless; :meth:`shutdown` (driven by
``Runner.close``) drains it at teardown.

These tests are scoped to the regression FAMILY, not the specific v22
scenario:

* The judge survives a ``task.cancel()`` on its spawning task,
  regardless of whether the cancel is fired from a sibling drift's
  ``request_invocation_cancel`` path or from upstream cancel propagation
  (adk-web stream close, generator ``aclose()``).
* The fix preserves PR #320's contract: the judge is still drainable
  by ``steerer.shutdown()``.
* Both spawn sites — task-boundary (``mark_task_*``) and turn-counter
  (``note_agent_turn``) — are covered.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Stubs (kept local so this file is self-contained as a regression bundle)
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class StubPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, *, goals, available_agents, context=None):
        return None

    async def refine(self, *, plan, drift, goals):
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        return None


def _make_session(task_id: str = "t1") -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=task_id, title="Research solar panels", description="Find specs")],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Cancellation-source coverage: a cancel on the spawning agent's task must
# NOT propagate to the goal-drift judge.
# ---------------------------------------------------------------------------


async def test_judge_survives_cancel_of_spawning_task_at_task_boundary() -> None:
    """Cancel the task that hosted ``mark_task_completed`` — judge runs.

    Reproduces the v22 mechanism: a sibling drift fires
    ``request_invocation_cancel(cancel_inflight_task=True)`` on the
    agent's invocation task while the report-tool handler is still
    executing the inline goal-drift judge. Pre-fix, the judge died
    with ``CancelledError`` and an empty stack. Post-fix, the spawn-
    and-detach pattern keeps it alive.
    """
    judge_started = asyncio.Event()
    judge_complete = asyncio.Event()

    async def slow_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        judge_started.set()
        # Sleep long enough to make sure the cancel below lands while
        # the judge is still mid-call. 0.3s is well within CI tolerance.
        try:
            await asyncio.sleep(0.3)
        finally:
            judge_complete.set()
        return json.dumps({"progressing": True})

    steerer = DefaultSteerer(
        goal_drift_check_interval=100,
        goal_drift_call_llm=slow_call_llm,
    )
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    async def host_agent_invocation() -> None:
        # This task models the ADK agent invocation task that hosts
        # the report_task_completed tool handler. mark_task_completed
        # spawns the goal-drift judge as a background task; we then
        # cancel ourselves to simulate a sibling cancel landing on
        # this task.
        await steerer.mark_task_completed(
            "t1", session=session, summary="research_panels done"
        )
        # Wait for the judge to have started before we yield to the
        # cancel — ensures we are reproducing "cancel lands while
        # judge is running" rather than "cancel before judge spawned".
        await judge_started.wait()
        # Now sleep so the cancel below has time to fire. The judge
        # runs on a separate task and is unaffected.
        await asyncio.sleep(1.0)

    host = asyncio.create_task(host_agent_invocation(), name="host-invoke")
    # Wait for judge to start.
    await asyncio.wait_for(judge_started.wait(), timeout=2.0)
    # Cancel the host task — same mechanism as
    # ``_GoldfiveADKPlugin.request_invocation_cancel(cancel_inflight_task=True)``.
    host.cancel()
    try:
        await host
    except asyncio.CancelledError:
        pass

    # Drain the spawned background judge. With the spawn-and-detach
    # fix, this task is independent of ``host`` and runs to completion.
    pending = list(steerer._background_judges)
    assert pending, (
        "task-boundary trigger must have spawned a background judge "
        "before the host was cancelled"
    )
    results = await asyncio.gather(*pending, return_exceptions=True)
    for r in results:
        assert not isinstance(r, BaseException), (
            f"background judge raised {r!r}; the spawn-and-detach fix "
            "should have isolated it from the host's cancel"
        )

    assert judge_complete.is_set(), (
        "goal-drift judge LLM call must have completed despite the "
        "host invocation task being cancelled mid-call"
    )


async def test_judge_survives_cancel_of_spawning_task_via_note_agent_turn() -> None:
    """Same property for the ``note_agent_turn`` (after_run_callback) path.

    The turn-counter trigger fires from the ADK plugin's
    ``after_run_callback`` which also runs on the agent's invocation
    task. The spawn site is symmetric with the task-boundary one and
    must benefit from the same isolation.
    """
    judge_started = asyncio.Event()
    judge_complete = asyncio.Event()

    async def slow_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        judge_started.set()
        try:
            await asyncio.sleep(0.3)
        finally:
            judge_complete.set()
        return json.dumps({"progressing": True})

    steerer = DefaultSteerer(
        goal_drift_check_interval=1,  # fire on every turn
        goal_drift_call_llm=slow_call_llm,
    )
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    async def host_after_run() -> None:
        await steerer.note_agent_turn(session)
        await judge_started.wait()
        await asyncio.sleep(1.0)

    host = asyncio.create_task(host_after_run(), name="host-after-run")
    await asyncio.wait_for(judge_started.wait(), timeout=2.0)
    host.cancel()
    try:
        await host
    except asyncio.CancelledError:
        pass

    pending = list(steerer._background_judges)
    assert pending, "note_agent_turn must spawn the judge as a background task"
    results = await asyncio.gather(*pending, return_exceptions=True)
    for r in results:
        assert not isinstance(r, BaseException), (
            f"background judge raised {r!r}; expected clean completion"
        )
    assert judge_complete.is_set()


async def test_two_back_to_back_task_transitions_do_not_clobber_running_judge() -> None:
    """A second task transition during the rate-limit window does not
    cancel the in-flight judge from the first one.

    Generalises the v22 scenario: even if the second mark_task_*
    happens to share a task with the first (cascading cancel-cancel),
    the first judge — already running on the background task — must
    not die when the second call's host returns / is cancelled.
    """
    judge_completions: list[bool] = []

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        await asyncio.sleep(0.2)
        judge_completions.append(True)
        return json.dumps({"progressing": True})

    steerer = DefaultSteerer(
        goal_drift_check_interval=100,
        goal_drift_call_llm=call_llm,
    )
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()

    # First transition spawns the judge, primes the rate-limit
    # timestamp. Second transition rate-limits to no-op (timestamp guard).
    await steerer.mark_task_completed("t1", session=session, summary="done")
    # Sanity: spawn happened.
    assert len(steerer._background_judges) == 1
    # Second transition immediately after — within rate-limit window —
    # should NOT spawn a second judge AND must not cancel the first.
    plan = session.plan
    assert plan is not None
    # goldfive#247: Plan is frozen — extend via add_tasks.
    from goldfive.types import (
        add_tasks,
        channel_processor_active,
        set_session_plan,
    )
    with channel_processor_active():
        set_session_plan(
            session,
            add_tasks(plan, [Task(id="t2", title="t2", description="")]),
        )
    await steerer.mark_task_completed("t2", session=session, summary="also done")
    # First judge still pending.
    assert len(steerer._background_judges) == 1

    pending = list(steerer._background_judges)
    results = await asyncio.gather(*pending, return_exceptions=True)
    for r in results:
        assert not isinstance(r, BaseException), (
            f"first judge must complete cleanly; got {r!r}"
        )
    assert judge_completions == [True], (
        "first judge must run to completion exactly once; second "
        "transition was rate-limited"
    )


# ---------------------------------------------------------------------------
# Drainability: PR #320 contract preservation.
# ---------------------------------------------------------------------------


async def test_goal_drift_judge_drainable_at_steerer_shutdown() -> None:
    """``steerer.shutdown()`` drains a still-running goal-drift judge.

    PR #320's contract — judges live for the lifetime of the Runner,
    drained at ``Runner.close()`` — must hold for goal-drift judges
    too. The shutdown path uses :meth:`DefaultSteerer.shutdown` which
    walks ``_background_judges``; the goal-drift spawn registers
    there, so this is a structural test.
    """

    async def slow_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        await asyncio.sleep(2.0)
        return json.dumps({"progressing": True})

    steerer = DefaultSteerer(
        goal_drift_check_interval=100,
        goal_drift_call_llm=slow_call_llm,
    )
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()

    await steerer.mark_task_completed("t1", session=session, summary="done")
    assert len(steerer._background_judges) == 1

    # shutdown with a tight timeout — the judge is sleeping 2s, so it
    # will be cancelled by the shutdown path.
    await steerer.shutdown(timeout=0.1)
    # Yield once so done-callbacks fire.
    await asyncio.sleep(0)
    assert steerer._background_judges == set(), (
        "shutdown must clear _background_judges; got "
        f"{len(steerer._background_judges)} stragglers"
    )


async def test_goal_drift_judge_completes_naturally_when_left_alone() -> None:
    """Bare-baseline: the new spawn path doesn't change happy-path semantics.

    A judge that runs to completion must still emit its drift event
    on the sink and (if off-track) reach the planner. This guards
    against the spawn refactor accidentally breaking the wire-up.
    """

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps({"progressing": False, "reason": "off the rails"})

    steerer = DefaultSteerer(
        goal_drift_check_interval=100,
        goal_drift_call_llm=call_llm,
    )
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    await steerer.mark_task_completed("t1", session=session, summary="done")
    # Drain.
    pending = list(steerer._background_judges)
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)

    # GOAL_DRIFT drift was emitted.
    drifts = [
        e for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]
    goal_drifts = [d for d in drifts if "off the rails" in d.drift_detected.detail]
    assert goal_drifts, (
        "judge that returns progressing=false must still produce a "
        "GOAL_DRIFT drift via the spawn-and-detach path; got "
        f"{[d.drift_detected.detail for d in drifts]}"
    )


# ---------------------------------------------------------------------------
# Spawn-time invariants
# ---------------------------------------------------------------------------


async def test_spawn_helper_is_noop_when_no_judge_is_wired() -> None:
    """``_spawn_goal_drift_judge_background`` no-ops without a wired call_llm.

    The caller short-circuit at the top of
    ``_maybe_run_goal_drift_on_task_boundary`` already guards this,
    but the spawn helper itself must be safe to invoke directly so
    operator-driven one-shot triggers (or future internal callers)
    can rely on it.
    """
    steerer = DefaultSteerer()  # no goal_drift_call_llm
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()
    steerer._spawn_goal_drift_judge_background(session)
    assert steerer._background_judges == set(), (
        "spawn helper must not register a task when no judge is wired"
    )


def test_spawn_helper_is_noop_outside_event_loop() -> None:
    """The helper degrades to a silent no-op when no event loop is running.

    Synchronous test harnesses / module-import-time wiring must not
    crash if a call site fires through. Real production paths always
    run inside an async context (the spawn site is reached via an
    ``await mark_task_*`` call). This test is intentionally
    synchronous so :func:`asyncio.get_running_loop` raises
    ``RuntimeError`` and the spawn helper exercises its degraded path.
    """

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps({"progressing": True})

    steerer = DefaultSteerer(goal_drift_call_llm=call_llm)
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()

    # No running loop — helper must return cleanly without scheduling.
    steerer._spawn_goal_drift_judge_background(session)
    assert steerer._background_judges == set()
