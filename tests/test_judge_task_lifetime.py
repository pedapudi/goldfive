"""Regression tests for goldfive#319: background reasoning-judge tasks
must not be cancelled at the per-turn boundary.

The fire-and-forget reasoning judge (goldfive#251) runs off the critical
path so the adapter's model-response callback can return before the
slow LLM judge completes. Pre-fix, ``GoldfiveADKAgent._run_async_impl``
called ``steerer.shutdown(timeout=0.5)`` in its per-turn ``finally``
block, which fired ``task.cancel()`` on every still-running judge — the
verdict was dropped on the floor whenever the LLM was slower than 0.5s,
even when the same Runner had many more turns ahead of it.

The fix removes the per-turn drain so background judges live for the
lifetime of the :class:`~goldfive.runner.Runner`. The canonical drain
runs only at :meth:`Runner.close`. A late verdict that arrives after
its target invocation has already terminated is still recorded on the
sink (observability preserved); the steerer skips the cancel + ladder
dispatch via :meth:`DefaultSteerer._is_late_drift_for_terminated_invocation`
inside :meth:`DefaultSteerer._run_judge_background`.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Test stubs (mirrors of test_reasoning_judge.py — kept local so this
# file is self-contained as a regression bundle for #319).
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class NullPlanner:
    """Non-None planner whose ``refine`` records every call. The
    presence-check in :meth:`DefaultSteerer._handle_drift` requires a
    non-None planner before refine is reached, so the late-drift
    tolerance test wires this in to confirm refine is *not* invoked.
    """

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return None


def _session(run_id: str = "r1", task_id: str = "t1") -> Session:
    task = Task(
        id=task_id, title="Research solar panels", description="Find specs"
    )
    plan = Plan(
        id="p1",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
    )
    return Session(
        run_id=run_id,
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id=task_id,
    )


def _task_status(session: Session, task_id: str) -> TaskStatus:
    plan = session.plan
    assert plan is not None
    for t in plan.tasks:
        if t.id == task_id:
            return t.status
    raise AssertionError(f"task {task_id!r} not found on plan")


# ---------------------------------------------------------------------------
# Slow-judge survival across turn boundaries (goldfive#319 core regression)
# ---------------------------------------------------------------------------


async def test_slow_judge_completes_when_per_turn_drain_is_removed() -> None:
    """A judge slower than the deleted 0.5s drain budget completes cleanly.

    The pre-fix ``_drain_steerer_background_judges`` cancelled judges
    whose LLM round-trip exceeded 0.5s; this test simulates that exact
    scenario. With the fix, the judge runs to completion and emits its
    drift verdict to the sink.
    """

    async def slow_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        # Well past the 0.5s the previous per-turn drain allowed. We
        # keep it modest so the test is fast on CI.
        await asyncio.sleep(0.7)
        return json.dumps(
            {"on_task": False, "severity": "info", "reason": "slightly off"}
        )

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=slow_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    await steerer.observe_reasoning(
        "raccoons are nocturnal", session=session
    )
    assert len(steerer._background_judges) == 1, (
        "observe_reasoning must schedule the judge as a background task"
    )

    # Wait for completion — no cancel was applied.
    pending = list(steerer._background_judges)
    results = await asyncio.gather(*pending, return_exceptions=True)
    for r in results:
        assert not isinstance(r, BaseException), (
            f"background judge raised {r!r}; expected clean completion"
        )

    drifts = [
        e for e in sink.events if e.WhichOneof("payload") == "drift_detected"
    ]
    assert len(drifts) == 1, (
        f"slow judge must produce its drift verdict; got {len(drifts)} drift "
        "events on the sink"
    )


async def test_done_callback_removes_from_background_judges() -> None:
    """The ``add_done_callback`` registered at spawn must clear the set."""

    async def fast_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps({"on_task": True})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=fast_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    await steerer.observe_reasoning("clean reasoning", session=session)
    assert len(steerer._background_judges) == 1

    # Drain the task; the done_callback fires synchronously when the
    # task completes.
    pending = list(steerer._background_judges)
    await asyncio.gather(*pending, return_exceptions=True)
    # Yield once so the done_callback is invoked.
    await asyncio.sleep(0)

    assert steerer._background_judges == set(), (
        "done_callback must remove the completed task from "
        "_background_judges; got "
        f"{len(steerer._background_judges)} stragglers"
    )


# ---------------------------------------------------------------------------
# Late-drift tolerance (goldfive#319): a verdict produced by the
# fire-and-forget background judge whose target invocation has already
# terminated must record the drift but skip refine.
# ---------------------------------------------------------------------------
#
# Scoped to ``_run_judge_background`` — only the background judge path
# can emit verdicts that outlive their originating invocation.
# Synchronous detectors fire inline on the model-response callback and
# always see a live invocation, so they never need this guard.


async def test_late_judge_verdict_records_drift_but_skips_refine(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A judge verdict produced when no invocations are active is
    recorded on the sink but does NOT trigger planner.refine.
    """

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps(
            {
                "on_task": False,
                "severity": "warning",
                "reason": "drifted to raccoons",
            }
        )

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # No invocations registered on the OrchestrationStore for this
    # session — the agent has moved on. The fire-and-forget judge
    # treats this as a stale verdict.
    with caplog.at_level(logging.INFO, logger="goldfive.steerer"):
        await steerer.observe_reasoning(
            "raccoons are nocturnal", session=session
        )
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)

    # The drift was emitted on the sink (observability preserved).
    drift_events = [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]
    assert len(drift_events) == 1, (
        "late judge verdict must still be recorded on the sink for "
        f"observability; got {len(drift_events)} drift event(s)"
    )

    # planner.refine was NOT called.
    assert planner.refine_calls == [], (
        "late judge verdict whose target invocation is gone must NOT "
        f"trigger planner.refine; got {len(planner.refine_calls)} call(s)"
    )

    # The structural log line was emitted at INFO so operators can see
    # late verdicts in the run log.
    info_records = [
        r
        for r in caplog.records
        if r.name == "goldfive.steerer"
        and r.levelno == logging.INFO
        and "stale judge verdict" in r.getMessage()
    ]
    assert len(info_records) == 1, (
        f"stale-judge-verdict log line must fire at INFO; got "
        f"{len(info_records)} matching record(s)"
    )


async def test_judge_verdict_with_live_invocation_proceeds_normally() -> None:
    """When an invocation is registered, the judge's verdict reaches
    the refine path normally.

    Confirms the late-drift guard is not over-broad — a WARNING drift
    on a live agent must still reach planner.refine (Level 1 ABSORB).
    """
    from goldfive.orchestration_store import OrchestrationStore

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps(
            {
                "on_task": False,
                "severity": "warning",
                "reason": "drifted to raccoons",
            }
        )

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # Register a fake live invocation. The task itself is irrelevant —
    # only the registry's non-emptiness gates the late-drift guard.
    store = OrchestrationStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(0.5)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-live", fake_task)
    try:
        await steerer.observe_reasoning(
            "raccoons are nocturnal", session=session
        )
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)

        # planner.refine WAS called (Level 1 ABSORB on first WARNING).
        assert len(planner.refine_calls) == 1, (
            "WARNING drift on a live invocation must reach planner.refine; "
            f"got {len(planner.refine_calls)} call(s)"
        )
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)


async def test_synchronous_tool_flow_unaffected_by_late_drift_guard() -> None:
    """Synchronous tool-flow drifts (mark_task_failed -> TASK_FAILED_*)
    fire inline through ``_handle_drift`` and must NOT be gated by the
    late-drift guard.

    Regression bound: the guard is scoped to ``_run_judge_background``
    only; broadening it to every ``_handle_drift`` call would skip
    refine for tool-callback flows whose invocation registry isn't
    necessarily populated in test setups (and shouldn't be: those
    callers ARE the live agent reporting status).
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # Drive a tool-flow drift (mark_task_failed -> TASK_FAILED_RECOVERABLE
    # -> Level 1 ABSORB) without any registered invocation. The guard
    # must not fire because this isn't the background-judge path.
    await steerer.mark_task_failed(
        "t1", session=session, reason="boom", recoverable=True
    )

    assert _task_status(session, "t1") is TaskStatus.FAILED
    # planner.refine WAS called: synchronous tool-flow drifts route
    # through ``_handle_drift`` directly and are not gated.
    assert len(planner.refine_calls) >= 1, (
        "synchronous tool-flow drift on a non-registered session must "
        "still reach planner.refine; got "
        f"{len(planner.refine_calls)} call(s)"
    )
