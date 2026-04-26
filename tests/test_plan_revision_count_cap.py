"""Plan-revision count cap (goldfive#271 follow-up).

Pre-fix evidence (demo-v8.log): 3 successive *successful* plan revisions
on the same ``(off_topic, research_solar)`` drift signature in 30
minutes — each refine succeeded, which reset the failure counter to 0,
and the loop was unbounded.

The fix adds :attr:`Session.plan_revision_counts` and a parallel cap
(:attr:`DefaultSteerer.PLAN_REVISION_COUNT_LIMIT`, default 3). Once the
cap is hit, further drifts of the same ``(kind, task_id)`` are routed
to ``HUMAN_INTERVENTION_REQUIRED`` and the runner pauses.

These tests pin the cap behaviour without a real LLM.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)

    async def close(self) -> None:
        return None


class _LoopingPlanner:
    """Planner that always returns a successful, slightly-different plan
    revision so the cap (not the failure counter) is what gates."""

    def __init__(self) -> None:
        self.refine_calls: list[DriftEvent] = []
        self._counter = 0

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan:
        self.refine_calls.append(drift)
        self._counter += 1
        # Bump revision_index to satisfy the validator's "for_revision=True"
        # rule (the revised plan must have a higher revision_index than
        # the prior plan).
        return Plan(
            id=plan.id,
            run_id=plan.run_id,
            goal_ids=list(plan.goal_ids),
            tasks=[Task(id=t.id, title=t.title, status=t.status) for t in plan.tasks],
            edges=list(plan.edges),
            revision_index=plan.revision_index + 1,
        )


def _make_drift(kind: DriftKind, task_id: str) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=DriftSeverity.WARNING,
        detail=f"{kind.value} drift on {task_id}",
        current_task_id=task_id,
    )


def _make_session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1")],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan,
    )


@pytest.mark.asyncio
async def test_revision_count_default_limit_is_3():
    """The class default is 3 successful revisions per (kind, task)."""
    assert DefaultSteerer.PLAN_REVISION_COUNT_LIMIT == 3


@pytest.mark.asyncio
async def test_revision_count_cap_escalates_after_threshold():
    """After ``PLAN_REVISION_COUNT_LIMIT`` successful revisions on the
    same (kind, task), the next drift escalates to
    HUMAN_INTERVENTION_REQUIRED and pauses the runner."""
    steerer = DefaultSteerer()
    session = _make_session()
    sink = _ListSink()
    planner = _LoopingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # Force the cap low so the test stays fast.
    steerer.PLAN_REVISION_COUNT_LIMIT = 2

    # Drive 2 successful refines — both should land normally.
    for _ in range(2):
        await steerer._handle_drift(
            _make_drift(DriftKind.OFF_TOPIC, "t1"),
            session,
        )
    assert len(planner.refine_calls) == 2
    assert session.plan_revision_counts[("t1", DriftKind.OFF_TOPIC.value)] == 2
    assert session.paused_for_human_intervention is False

    # The next drift on the same (kind, task) is gated → no refine,
    # session paused, HUMAN_INTERVENTION_REQUIRED emitted.
    await steerer._handle_drift(
        _make_drift(DriftKind.OFF_TOPIC, "t1"),
        session,
    )
    assert len(planner.refine_calls) == 2  # NOT bumped
    assert session.paused_for_human_intervention is True


@pytest.mark.asyncio
async def test_revision_count_isolates_per_task():
    """The cap is keyed on (kind, task_id) — a different task starts
    a fresh budget."""
    steerer = DefaultSteerer()
    session = _make_session()
    # Add a second task so the second drift's task_id is valid.
    assert session.plan is not None
    session.plan.tasks.append(Task(id="t2", title="T2"))
    sink = _ListSink()
    planner = _LoopingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.PLAN_REVISION_COUNT_LIMIT = 1

    await steerer._handle_drift(_make_drift(DriftKind.OFF_TOPIC, "t1"), session)
    assert len(planner.refine_calls) == 1
    # t1 cap consumed; another t1 drift would be gated.
    await steerer._handle_drift(_make_drift(DriftKind.OFF_TOPIC, "t1"), session)
    assert len(planner.refine_calls) == 1  # gated

    # But a t2 drift still gets a refine — independent (kind, task) key.
    await steerer._handle_drift(_make_drift(DriftKind.OFF_TOPIC, "t2"), session)
    assert len(planner.refine_calls) == 2


@pytest.mark.asyncio
async def test_revision_count_isolates_per_kind():
    """The cap is keyed on (kind, task_id) — a different drift kind on
    the same task starts a fresh budget."""
    steerer = DefaultSteerer()
    session = _make_session()
    sink = _ListSink()
    planner = _LoopingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.PLAN_REVISION_COUNT_LIMIT = 1

    await steerer._handle_drift(_make_drift(DriftKind.OFF_TOPIC, "t1"), session)
    assert len(planner.refine_calls) == 1
    # OFF_TOPIC budget on t1 consumed.
    await steerer._handle_drift(_make_drift(DriftKind.OFF_TOPIC, "t1"), session)
    assert len(planner.refine_calls) == 1  # gated

    # But a CONFUSION drift on the same task still gets through.
    await steerer._handle_drift(_make_drift(DriftKind.CONFUSION, "t1"), session)
    assert len(planner.refine_calls) == 2


@pytest.mark.asyncio
async def test_user_steer_bypasses_revision_count_cap():
    """USER_STEER / USER_CANCEL / GOAL_DRIFT bypass the cap — user
    actions are always honoured."""
    steerer = DefaultSteerer()
    session = _make_session()
    sink = _ListSink()
    planner = _LoopingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.PLAN_REVISION_COUNT_LIMIT = 1

    # Drive an OFF_TOPIC into the cap.
    await steerer._handle_drift(_make_drift(DriftKind.OFF_TOPIC, "t1"), session)
    await steerer._handle_drift(_make_drift(DriftKind.OFF_TOPIC, "t1"), session)
    assert len(planner.refine_calls) == 1  # second was gated

    # USER_STEER on the same task is exempt — increments
    # plan_revision_counts but is NOT gated.
    user_drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="user steered",
        current_task_id="t1",
    )
    await steerer._handle_drift(user_drift, session)
    # USER_STEER bypasses the gate; the planner sees it.
    assert len(planner.refine_calls) >= 2
