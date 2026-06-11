"""Guard test: #469 (ledger refine retirement) × #470 (pacing) interleaving.

AGENCY-PRESERVATION.md §5.3 / §5.5. PR 8 (#470) and PR 12 (#469) both
modify the goldfive-drift PROMOTION dispatch branch in
``DriftObserver._handle_drift`` (``if promote_to_steer:``); the rebase that
landed #470 onto #469 resolved the overlap **pacing-first**:

    if promote_to_steer:
        if await self._apply_signal_pacing(...):   # PR 8 — gates FIRST
            return
        if self._ledger_mode():                     # PR 12 — ledger/forecast fork
            await self._ledger_retire_refine(...)
        else:
            await self._promote_drift_to_steer(...)
        return

Each PR's own suite exercises only its slice (PR 8's pacing tests run in
forecast/non-ledger; PR 12's ledger-retire tests run with the grace window
off), so NEITHER pins the COMBINED behaviour: in ledger + request_context
mode, a promotion drift that is inside its grace window must be FULLY
suppressed — it must NOT fall through to ``_ledger_retire_refine``'s
force-FAIL / pause / note (re-failing or re-pausing a key the agent was
just signalled about defeats the grace window), and a past-window one MUST
reach the ledger rung. A future refactor that reordered the two (ledger
fork before pacing) would pass every existing test while silently
re-acting inside the window; this test is the guard against that.

Routing note (verified): OFF_TOPIC + WARNING under ``threshold="warning"``
reaches the promotion branch and, in ledger mode, routes
``_signal_pacing_decision`` → (proceed) → ``_ledger_retire_refine`` →
``_dispatch_nudge`` (the non-looping/non-hard-safety else rung).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.observer_note_queue import ObserverNoteQueue  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskKind,
    TaskStatus,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _RecordingPlanner:
    """Records any forecast-repair refine so the test can assert NONE ran."""

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_steer_calls: list[dict[str, Any]] = []

    async def generate(self, *, goals: Any, available_agents: Any, context: Any = None) -> Any:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return kwargs.get("plan")

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return kwargs.get("plan")


def _make_steerer(*, grace: int = 3) -> tuple[DefaultSteerer, _RecordingPlanner]:
    """Ledger + request_context steerer with the grace window armed."""
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            plan_mode="ledger",
            signal_channel="request_context",
            grace_window_turns=grace,
            observation_only=False,
            threshold="warning",
        )
    )
    planner = _RecordingPlanner()
    steerer.bind(sinks=[_ListSink()], planner=planner)
    return steerer, planner


def _ledger_session(*, turn: int) -> Session:
    plan = Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(
            Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),
            Task(
                id="d1",
                title="writer: drafting",
                discovered=True,
                kind=TaskKind.DISCOVERED,
                status=TaskStatus.RUNNING,
            ),
        ),
        edges=(),
        revision_index=1,
    )
    session = Session(run_id="r", goals=[Goal(id="g", summary="summarise the deck")], plan=plan)
    session._reasoning_turn = turn
    return session


def _promotion_drift() -> DriftEvent:
    # OFF_TOPIC + WARNING under threshold="warning" reaches the promotion
    # branch; bound to the DISCOVERED task d1.
    return DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent wandered off the goal",
        current_task_id="d1",
        current_agent_id="writer",
        authored_by="goldfive",
    )


def _render_prior_signal(session: Session, *, turn: int) -> None:
    """Enqueue + RENDER a prior signal note for (off_topic, d1) at ``turn``.

    Render (``mark_delivered``) is what starts the grace window — the gate
    keys on render-visibility, not enqueue.
    """
    q = ObserverNoteQueue.for_session(session)
    q.enqueue(
        body="Observation: wandered",
        observation="wandered",
        severity="warning",
        drift_id="prior-signal-uuid",
        kind=DriftKind.OFF_TOPIC.value,
        task_id="d1",
        turn=turn,
    )
    q.mark_delivered("prior-signal-uuid", channel="request_context", turn=turn)


def _spy_ledger_retire(steerer: DefaultSteerer) -> list[Any]:
    """Wrap ``_ledger_retire_refine`` to record calls; returns the call list."""
    calls: list[Any] = []
    orig = steerer.drift._ledger_retire_refine

    async def _spy(drift: DriftEvent, session: Session) -> None:
        calls.append(drift.kind)
        await orig(drift, session)

    steerer.drift._ledger_retire_refine = _spy  # type: ignore[method-assign]
    return calls


async def test_ledger_promotion_within_grace_window_is_fully_suppressed() -> None:
    """A within-window promotion re-fire short-circuits BEFORE the ledger fork.

    Pins pacing-FIRST: ``_apply_signal_pacing`` suppresses and returns, so
    ``_ledger_retire_refine`` never runs — no force-FAIL, no pause, no new
    note. A reorder (ledger fork before pacing) would re-fire here.
    """
    steerer, planner = _make_steerer(grace=3)
    session = _ledger_session(turn=6)
    # A prior signal for (off_topic, d1) was RENDERED at turn 5 → age 1 < 3.
    _render_prior_signal(session, turn=5)
    retire_calls = _spy_ledger_retire(steerer)
    q = ObserverNoteQueue.for_session(session)
    count_before = q.signal_count("off_topic", "d1")  # 1 (the prior signal)

    await steerer.drift.handle_drift(_promotion_drift(), session)

    # The ledger-retire fork was NOT reached (the invariant).
    assert retire_calls == [], "pacing must short-circuit BEFORE _ledger_retire_refine"
    # No forecast-repair refine of either flavour either.
    assert planner.refine_calls == [] and planner.refine_steer_calls == []
    # FULLY suppressed: no new signal note enqueued...
    assert q.signal_count("off_topic", "d1") == count_before
    # ...and the bound DISCOVERED task was NOT force-FAILed.
    d1 = next(t for t in session.plan.tasks if t.id == "d1")
    assert d1.status is TaskStatus.RUNNING


async def test_ledger_promotion_past_grace_window_reaches_ledger_retire() -> None:
    """Past the window, the promotion drift reaches the PR-12 ledger rung.

    The control half of the invariant: with no prior render the pacing gate
    returns ``proceed`` and the ledger/forecast fork runs — for a
    non-looping/non-hard-safety drift that is the advisory-note rung.
    """
    steerer, planner = _make_steerer(grace=3)
    session = _ledger_session(turn=0)  # no prior render → not in any window
    retire_calls = _spy_ledger_retire(steerer)
    q = ObserverNoteQueue.for_session(session)

    await steerer.drift.handle_drift(_promotion_drift(), session)

    # The ledger-retire fork fired (proceed → PR-12 rung).
    assert retire_calls == [DriftKind.OFF_TOPIC], "past the window the ledger rung must fire"
    # It retired the forecast refine_steer (no refine ran)...
    assert planner.refine_calls == [] and planner.refine_steer_calls == []
    # ...and the else rung enqueued the advisory note.
    assert q.pending(), "the ledger else-rung should enqueue an advisory note"
    # The bound task is untouched (a note, not a force-FAIL, for OFF_TOPIC).
    d1 = next(t for t in session.plan.tasks if t.id == "d1")
    assert d1.status is TaskStatus.RUNNING


async def test_ledger_promotion_escalate_also_short_circuits_the_fork() -> None:
    """A 3rd-occurrence escalate is handled by pacing, never by the fork.

    Past the window with ``signal_count >= REFINE_FAILURE_THRESHOLD``,
    ``_signal_pacing_decision`` returns ``escalate`` → pacing dispatches the
    pause and returns; ``_ledger_retire_refine`` must NOT also run (no
    double-disposition).
    """
    steerer, planner = _make_steerer(grace=3)
    session = _ledger_session(turn=50)
    threshold = steerer.REFINE_FAILURE_THRESHOLD
    # Enqueue >= threshold prior signals for the key, all rendered long ago
    # (turn 0) so they are PAST the window (age 50) — escalation, not suppress.
    q = ObserverNoteQueue.for_session(session)
    for i in range(threshold):
        did = f"prior-{i}"
        q.enqueue(
            body="o", observation="o", severity="warning",
            drift_id=did, kind=DriftKind.OFF_TOPIC.value, task_id="d1", turn=0,
        )
        q.mark_delivered(did, channel="request_context", turn=0)
    retire_calls = _spy_ledger_retire(steerer)

    await steerer.drift.handle_drift(_promotion_drift(), session)

    # Escalation is a pacing decision — the ledger fork must not also fire.
    assert retire_calls == [], "escalate must short-circuit before _ledger_retire_refine"
    assert planner.refine_calls == [] and planner.refine_steer_calls == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
