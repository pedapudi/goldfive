"""Minimum-intervention pacing — grace windows + escalation (PR 8).

AGENCY-PRESERVATION.md Stage 2 PR 8. After a note for a ``(kind, task)`` key is
RENDERED, that key cannot re-signal/escalate for ``grace_window_turns`` logical
turns; the 2nd signal is re-authored quoting the first; the 3rd occurrence
escalates to a pause. Two BINDING requirements from the #462 review are pinned
here:

1. the grace window keys on VISIBILITY — the ObserverNoteQueue's
   ``delivered_turn`` (stamped at render), NOT the SignalLedger's dispatch turn;
2. ``self_corrected_after_signal`` attribution uses visibility — a note
   enqueued-but-never-rendered records ``self_corrected_unaided``.

The hypothesis interleaving tests (§5.5) hammer the queue + ledger pacing
bookkeeping under concurrent fires, late drifts, user steers, and restarts.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.control import ControlChannel, ControlKind  # noqa: E402
from goldfive.events import (  # noqa: E402
    SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL,
    SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED,
)
from goldfive.observer_note_queue import ObserverNoteQueue  # noqa: E402
from goldfive.signal_ledger import SignalLedger  # noqa: E402
from goldfive.state_store import set_active_steer  # noqa: E402
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(turn: int = 0) -> Session:
    s = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship a memo")],
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[Task(id="t1", title="research")],
            edges=[],
        ),
        current_task_id="t1",
    )
    s._reasoning_turn = turn
    return s


def _steerer(*, grace: int = 3, channel: str = "request_context") -> DefaultSteerer:
    return DefaultSteerer(
        steering_config=SteeringConfig(
            signal_channel=channel, grace_window_turns=grace
        )
    )


def _enqueue(
    session: Session,
    *,
    drift_id: str,
    kind: str = "looping_tool_call",
    task: str = "t1",
    turn: int = 0,
) -> None:
    ObserverNoteQueue.for_session(session).enqueue(
        body="Observation: x", observation="x", severity="warning",
        drift_id=drift_id, kind=kind, task_id=task, turn=turn,
    )


def _drift(
    kind: DriftKind = DriftKind.LOOPING_TOOL_CALL,
    severity: DriftSeverity = DriftSeverity.CRITICAL,
) -> DriftEvent:
    return DriftEvent(
        kind=kind, severity=severity, detail="x",
        current_task_id="t1", current_agent_id="agent", authored_by="goldfive",
    )


# ---------------------------------------------------------------------------
# Queue pacing reads (visibility source of truth)
# ---------------------------------------------------------------------------


def test_last_rendered_turn_is_render_not_enqueue() -> None:
    session = _session()
    q = ObserverNoteQueue.for_session(session)
    _enqueue(session, drift_id="d1", turn=5)
    # Enqueued but NOT rendered → -1 (no grace window started).
    assert q.last_rendered_turn("looping_tool_call", "t1") == -1
    q.mark_delivered("d1", channel="request_context", turn=7, surface="before_model")
    # Rendered at turn 7 → that is the grace anchor (not the enqueue turn 5).
    assert q.last_rendered_turn("looping_tool_call", "t1") == 7


def test_last_rendered_turn_takes_max() -> None:
    session = _session()
    q = ObserverNoteQueue.for_session(session)
    _enqueue(session, drift_id="d1")
    _enqueue(session, drift_id="d2")
    q.mark_delivered("d1", channel="request_context", turn=3)
    q.mark_delivered("d2", channel="request_context", turn=9)
    assert q.last_rendered_turn("looping_tool_call", "t1") == 9


def test_signal_count_counts_enqueued_notes() -> None:
    session = _session()
    q = ObserverNoteQueue.for_session(session)
    assert q.signal_count("looping_tool_call", "t1") == 0
    _enqueue(session, drift_id="d1")
    _enqueue(session, drift_id="d2")
    assert q.signal_count("looping_tool_call", "t1") == 2
    # Distinct key is independent.
    assert q.signal_count("off_topic", "t1") == 0


def test_rendered_keys_only_includes_rendered() -> None:
    session = _session()
    q = ObserverNoteQueue.for_session(session)
    _enqueue(session, drift_id="d1", kind="looping_tool_call")
    _enqueue(session, drift_id="d2", kind="off_topic")
    q.mark_delivered("d1", channel="request_context", turn=1)
    # d2 enqueued but never rendered → not in rendered_keys.
    assert q.rendered_keys() == {("looping_tool_call", "t1")}


def test_pacing_reads_exclude_correction_notes() -> None:
    """Task-#11 correction notes are not drift signals — excluded from the
    grace window / escalation / attribution reads (composition with #468)."""
    from goldfive.observer_note_queue import CORRECTION_DRIFT_ID_PREFIX

    session = _session()
    q = ObserverNoteQueue.for_session(session)
    # A RENDERED correction note for (off_topic, t1).
    cid = f"{CORRECTION_DRIFT_ID_PREFIX}writer:t1:1"
    q.enqueue(
        body="b", observation="o", severity="warning", drift_id=cid,
        kind="off_topic", task_id="t1", agent_id="writer", turn=2,
    )
    q.mark_delivered(cid, channel="request_context", turn=2)
    # The correction does NOT count as a signal render / count / rendered key.
    assert q.last_rendered_turn("off_topic", "t1") == -1
    assert q.signal_count("off_topic", "t1") == 0
    assert ("off_topic", "t1") not in q.rendered_keys()
    # A genuine drift-signal note for the SAME key IS counted.
    q.enqueue(
        body="b", observation="o", severity="warning", drift_id="real-uuid",
        kind="off_topic", task_id="t1", turn=3,
    )
    q.mark_delivered("real-uuid", channel="request_context", turn=3)
    assert q.last_rendered_turn("off_topic", "t1") == 3
    assert q.signal_count("off_topic", "t1") == 1
    assert ("off_topic", "t1") in q.rendered_keys()


# ---------------------------------------------------------------------------
# Ordered gates — _signal_pacing_decision
# ---------------------------------------------------------------------------


def test_pacing_proceed_on_first_signal() -> None:
    steerer, session = _steerer(), _session(turn=0)
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "proceed"


def test_pacing_suppresses_inside_grace_window() -> None:
    steerer = _steerer(grace=3)
    session = _session(turn=6)
    _enqueue(session, drift_id="d1")
    ObserverNoteQueue.for_session(session).mark_delivered(
        "d1", channel="request_context", turn=5
    )
    # Rendered at 5, now turn 6 (age 1 < 3) → suppress.
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "suppress"


def test_pacing_proceeds_after_grace_window_expires() -> None:
    steerer = _steerer(grace=3)
    session = _session(turn=8)
    _enqueue(session, drift_id="d1")
    ObserverNoteQueue.for_session(session).mark_delivered(
        "d1", channel="request_context", turn=5
    )
    # Rendered at 5, now turn 8 (age 3 >= 3) → window expired; 1 prior signal
    # (< threshold) → proceed (this is the 2nd signal).
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "proceed"


def test_pacing_escalates_on_third_occurrence() -> None:
    steerer = _steerer(grace=3)
    threshold = steerer.REFINE_FAILURE_THRESHOLD  # 2
    session = _session(turn=20)
    q = ObserverNoteQueue.for_session(session)
    for i in range(threshold):
        _enqueue(session, drift_id=f"d{i}")
        q.mark_delivered(f"d{i}", channel="request_context", turn=i)
    # Past the window (turn 20 >> last render), signal_count == threshold →
    # escalate.
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "escalate"


def test_pacing_unrendered_note_does_not_start_grace() -> None:
    steerer = _steerer(grace=3)
    session = _session(turn=1)
    _enqueue(session, drift_id="d1", turn=0)  # enqueued, never rendered
    # No render → no grace window; 1 prior note (< threshold) → proceed.
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "proceed"


def test_pacing_gate1_fresh_user_steer_suppresses() -> None:
    steerer = _steerer(grace=3)
    session = _session(turn=2)
    # No queue note at all — but a fresh user steer is active → gate 1 suppress.
    set_active_steer(
        session.state, body="user says stop", at_turn=1, author="op", source="user"
    )
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "suppress"


def test_pacing_noop_in_legacy_channel() -> None:
    steerer = _steerer(channel="legacy_user_message", grace=3)
    session = _session(turn=6)
    _enqueue(session, drift_id="d1")
    ObserverNoteQueue.for_session(session).mark_delivered(
        "d1", channel="request_context", turn=5
    )
    # Legacy regime: PR 8 is a no-op → always proceed.
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "proceed"


def test_pacing_grace_disabled_still_applies_user_steer_gate() -> None:
    steerer = _steerer(grace=0)  # grace disabled
    session = _session(turn=6)
    _enqueue(session, drift_id="d1")
    ObserverNoteQueue.for_session(session).mark_delivered(
        "d1", channel="request_context", turn=5
    )
    # Grace disabled → no grace suppress (proceed despite recent render)...
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "proceed"
    # ...but a fresh user steer still suppresses (gate 1 is independent).
    set_active_steer(
        session.state, body="stop", at_turn=6, author="op", source="user"
    )
    assert steerer.drift._signal_pacing_decision(session, _drift()) == "suppress"


# ---------------------------------------------------------------------------
# 2nd-signal re-authoring (quoting the first)
# ---------------------------------------------------------------------------


async def test_second_signal_quotes_the_first() -> None:
    steerer = _steerer()
    session = _session(turn=5)
    # Enqueue a first note manually so _route_corrective_note sees one prior,
    # and RENDER it — the "This repeats an earlier observer note" claim is
    # only truthful when the agent actually saw the prior.
    q = ObserverNoteQueue.for_session(session)
    q.enqueue(
        body="Observation: search_web looped",
        observation="search_web was called 5 times with identical args",
        severity="warning", drift_id="d1", kind="looping_tool_call",
        task_id="t1", turn=0,
    )
    q.mark_delivered("d1", channel="request_context", turn=1)
    await steerer.drift._route_corrective_note(
        session, _drift(), "Observation: still looping", ladder_level="signal"
    )
    notes = ObserverNoteQueue.for_session(session).notes()
    second = [n for n in notes if n.drift_id != "d1"]
    assert len(second) == 1
    # The 2nd note's body quotes the first note's observation.
    assert "repeats an earlier observer note" in second[0].body
    assert "search_web was called 5 times" in second[0].body


async def test_second_signal_unrendered_prior_is_not_quoted() -> None:
    """Truthfulness: a prior the agent never SAW must not be claimed as
    "an earlier observer note" — enqueued-but-never-rendered priors compose
    the 2nd note WITHOUT the repeat claim."""
    steerer = _steerer()
    session = _session(turn=5)
    ObserverNoteQueue.for_session(session).enqueue(
        body="Observation: search_web looped",
        observation="search_web was called 5 times with identical args",
        severity="warning", drift_id="d1", kind="looping_tool_call",
        task_id="t1", turn=0,
    )  # NOT marked delivered — never rendered to the agent
    await steerer.drift._route_corrective_note(
        session, _drift(), "Observation: still looping", ladder_level="signal"
    )
    second = [
        n
        for n in ObserverNoteQueue.for_session(session).notes()
        if n.drift_id != "d1"
    ]
    assert len(second) == 1
    assert second[0].body == "Observation: still looping"
    assert "repeats an earlier observer note" not in second[0].body


async def test_second_signal_dry_run_prior_is_not_quoted() -> None:
    """A dry-run consume (observation_only shadow) never reached the agent,
    so it must not be quoted as a repeat either."""
    steerer = _steerer()
    session = _session(turn=5)
    q = ObserverNoteQueue.for_session(session)
    q.enqueue(
        body="Observation: search_web looped",
        observation="search_web was called 5 times with identical args",
        severity="warning", drift_id="d1", kind="looping_tool_call",
        task_id="t1", turn=0,
    )
    q.mark_delivered("d1", channel="request_context", turn=1, dry_run=True)
    await steerer.drift._route_corrective_note(
        session, _drift(), "Observation: still looping", ladder_level="signal"
    )
    second = [n for n in q.notes() if n.drift_id != "d1"]
    assert len(second) == 1
    assert "repeats an earlier observer note" not in second[0].body


async def test_second_signal_correction_prior_is_not_counted_or_quoted() -> None:
    """A task-#11 correction note is a plan-revision notice, not a drift
    signal — it must never be quoted as "an earlier observer note" (a false
    claim), nor count as the single prior that triggers the quote."""
    from goldfive.observer_note_queue import CORRECTION_DRIFT_ID_PREFIX

    steerer = _steerer()
    session = _session(turn=5)
    q = ObserverNoteQueue.for_session(session)
    # A RENDERED correction note for the SAME (kind, task) key.
    cid = f"{CORRECTION_DRIFT_ID_PREFIX}writer:t1:1"
    q.enqueue(
        body="The plan changed", observation="the plan was revised",
        severity="warning", drift_id=cid, kind="looping_tool_call",
        task_id="t1", agent_id="writer", turn=0,
    )
    q.mark_delivered(cid, channel="request_context", turn=1)
    await steerer.drift._route_corrective_note(
        session, _drift(), "Observation: still looping", ladder_level="signal"
    )
    new = [n for n in q.notes() if n.drift_id != cid]
    assert len(new) == 1
    # Treated as the FIRST signal: no repeat claim, and never quoting the
    # correction's observation.
    assert "repeats an earlier observer note" not in new[0].body
    assert "the plan was revised" not in new[0].body


def test_signal_notes_excludes_corrections_and_keeps_order() -> None:
    from goldfive.observer_note_queue import CORRECTION_DRIFT_ID_PREFIX

    session = _session()
    q = ObserverNoteQueue.for_session(session)
    _enqueue(session, drift_id="d1", turn=1)
    q.enqueue(
        body="b", observation="o", severity="warning",
        drift_id=f"{CORRECTION_DRIFT_ID_PREFIX}writer:t1:1",
        kind="looping_tool_call", task_id="t1", turn=2,
    )
    _enqueue(session, drift_id="d2", turn=3)
    got = q.signal_notes("looping_tool_call", "t1")
    assert [n.drift_id for n in got] == ["d1", "d2"]
    assert q.signal_notes("off_topic", "t1") == []


# ---------------------------------------------------------------------------
# Visibility attribution (binding requirement #2)
# ---------------------------------------------------------------------------


def test_resolve_task_after_signal_only_when_rendered() -> None:
    state: dict[str, Any] = {}
    ledger = SignalLedger(state)
    # A real (non-dry-run) delivery is recorded at dispatch for (looping, t1).
    ledger.record_delivery(
        drift_kind="looping_tool_call", task_id="t1", drift_id="d1",
        channel="request_context", turn=1, dry_run=False,
    )
    # ...but the note was NEVER rendered → rendered_keys empty → unaided.
    resolved = ledger.resolve_task(task_id="t1", turn=5, rendered_keys=set())
    assert len(resolved) == 1
    assert resolved[0].outcome == SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED


def test_resolve_task_after_signal_when_rendered() -> None:
    state: dict[str, Any] = {}
    ledger = SignalLedger(state)
    ledger.record_delivery(
        drift_kind="looping_tool_call", task_id="t1", drift_id="d1",
        channel="request_context", turn=1, dry_run=False,
    )
    resolved = ledger.resolve_task(
        task_id="t1", turn=5, rendered_keys={("looping_tool_call", "t1")}
    )
    assert resolved[0].outcome == SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL


def test_resolve_task_legacy_falls_back_to_has_real_delivery() -> None:
    state: dict[str, Any] = {}
    ledger = SignalLedger(state)
    ledger.record_delivery(
        drift_kind="looping_tool_call", task_id="t1", drift_id="d1",
        channel="nudge_replay", turn=1, dry_run=False,
    )
    # rendered_keys=None (legacy regime) → attribution by has_real_delivery.
    resolved = ledger.resolve_task(task_id="t1", turn=5, rendered_keys=None)
    assert resolved[0].outcome == SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL


# ---------------------------------------------------------------------------
# Composition with #469 (PR 12) — promotion-path pacing gates BEFORE the
# ledger/forecast fork.
#
# A promotion-eligible goldfive drift in ledger + request_context mode now hits
# BOTH PR 8's pacing gate AND #469's `_ledger_retire_refine` routing at the
# `if promote_to_steer:` site. Pacing MUST run first: a within-window re-fire
# must be a no-op (NOT a ledger force-FAIL), and a pacing-escalate must dispatch
# exactly one pause (NOT also #469's force-FAIL). LOOPING_TOOL_CALL is the probe
# kind — it is promotion-eligible AND a ledger force-FAIL kind, so a missed
# short-circuit is observable as a FAILED bound task.
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _RecordingPlanner:
    """Records refine / refine_steer calls so 'no refine ran' is assertable."""

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

    @property
    def refined(self) -> bool:
        return bool(self.refine_calls or self.refine_steer_calls)


def _ledger_session(turn: int) -> Session:
    s = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship a memo")],
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id="t1",
                    title="writer: drafting",
                    discovered=True,
                    kind=TaskKind.DISCOVERED,
                    status=TaskStatus.RUNNING,
                )
            ],
            edges=[],
        ),
        current_task_id="t1",
    )
    s._reasoning_turn = turn
    return s


def _ledger_steerer(*, grace: int = 3) -> tuple[DefaultSteerer, _RecordingPlanner]:
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            signal_channel="request_context",
            grace_window_turns=grace,
            plan_mode="ledger",
            threshold="warning",
            observation_only=False,
        )
    )
    planner = _RecordingPlanner()
    steerer.bind(sinks=[_ListSink()], planner=planner)
    return steerer, planner


def _promotion_drift() -> DriftEvent:
    # WARNING (clears threshold="warning" without tripping the CRITICAL cancel)
    # + LOOPING_TOOL_CALL (promotion-eligible AND ledger force-FAIL kind).
    return DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="search_web looped",
        current_task_id="t1",
        current_agent_id="agent",
        authored_by="goldfive",
    )


def _t1(session: Session) -> Task:
    return next(t for t in session.plan.tasks if t.id == "t1")


async def test_promotion_suppress_does_not_force_fail_in_ledger_mode() -> None:
    """Within-window promotion re-fire → suppressed BEFORE #469's ledger fork.

    If pacing did not gate first, the looping kind would force-FAIL the bound
    task; the grace window exists precisely to give the agent room to
    self-correct after it saw the prior note, so a suppressed re-fire must be a
    no-op.
    """
    steerer, planner = _ledger_steerer(grace=3)
    session = _ledger_session(turn=6)
    _enqueue(session, drift_id="d1", turn=5)
    ObserverNoteQueue.for_session(session).mark_delivered(
        "d1", channel="request_context", turn=5
    )
    await steerer.drift.handle_drift(_promotion_drift(), session)
    assert _t1(session).status is not TaskStatus.FAILED
    assert not planner.refined


async def test_promotion_escalate_pauses_once_no_force_fail() -> None:
    """3rd-occurrence promotion re-fire → exactly ONE pause, no force-FAIL.

    The pacing-escalate dispatches PAUSE_ESCALATE itself and returns 'stop'; it
    must NOT fall through to #469's `_ledger_retire_refine`, which for a looping
    kind would force-FAIL the task (and a hard-safety kind would dispatch a
    SECOND pause). Composition bug = a FAILED task or two pauses.
    """
    steerer, planner = _ledger_steerer(grace=3)
    channel = ControlChannel()
    steerer.bind_control_channel(channel)
    session = _ledger_session(turn=20)
    q = ObserverNoteQueue.for_session(session)
    for i in range(steerer.REFINE_FAILURE_THRESHOLD):
        _enqueue(session, drift_id=f"d{i}", turn=i)
        q.mark_delivered(f"d{i}", channel="request_context", turn=i)
    await steerer.drift.handle_drift(_promotion_drift(), session)
    assert _t1(session).status is not TaskStatus.FAILED
    assert not planner.refined
    drained: list[Any] = []
    inbox = channel._inbox  # noqa: SLF001 — test inspection
    while not inbox.empty():
        drained.append(inbox.get_nowait())
    pauses = [
        m
        for m in drained
        if getattr(m, "kind", None) is ControlKind.GOLDFIVE_PAUSE_ESCALATE
    ]
    assert len(pauses) == 1, (
        f"expected exactly one pause; got {[getattr(m, 'kind', None) for m in drained]}"
    )


async def test_promotion_proceed_runs_ledger_retire_in_ledger_mode() -> None:
    """Past the window, < threshold prior signals → proceed → #469's ledger
    fork fires (force-FAIL the looping bound task). The 'else' arm of the
    composition: when pacing proceeds, the ledger retirement still happens."""
    steerer, planner = _ledger_steerer(grace=3)
    session = _ledger_session(turn=8)
    _enqueue(session, drift_id="d1", turn=4)
    ObserverNoteQueue.for_session(session).mark_delivered(
        "d1", channel="request_context", turn=4
    )
    await steerer.drift.handle_drift(_promotion_drift(), session)
    assert _t1(session).status is TaskStatus.FAILED
    # Ledger mode retires the forecast-repair refine — no planner refine ran.
    assert not planner.refined


# ---------------------------------------------------------------------------
# §5.5 — property-based interleaving over queue + ledger pacing bookkeeping
# ---------------------------------------------------------------------------


_OPS = st.lists(
    st.tuples(
        st.sampled_from(["enqueue", "render", "fire", "resolve", "user_steer"]),
        st.sampled_from(["a", "b"]),  # drift ids
        st.sampled_from(["looping_tool_call", "off_topic"]),  # kinds
        st.integers(min_value=0, max_value=6),  # turn
    ),
    min_size=0,
    max_size=40,
)


@settings(max_examples=150)
@given(ops=_OPS)
def test_pacing_interleaving_invariants(
    ops: list[tuple[str, str, str, int]],
) -> None:
    session = _session()
    q = ObserverNoteQueue.for_session(session)
    ledger = SignalLedger.for_session(session)
    for op, did, kind, turn in ops:
        session._reasoning_turn = turn
        if op == "enqueue":
            q.enqueue(
                body="b", observation="o", severity="warning",
                drift_id=did, kind=kind, task_id="t1", turn=turn,
            )
        elif op == "render":
            q.mark_delivered(did, channel="request_context", turn=turn)
        elif op == "fire":
            ledger.record_fire(drift_kind=kind, task_id="t1", turn=turn, drift_id=did)
        elif op == "resolve":
            ledger.resolve_task(
                task_id="t1", turn=turn, rendered_keys=q.rendered_keys()
            )
        elif op == "user_steer":
            set_active_steer(
                session.state, body="x", at_turn=turn, author="op", source="user"
            )
        # Invariant: queue + ledger state always re-parse without error.
        for n in q.notes():
            assert n.note_id
        for entry in ledger.entries():
            assert entry.drift_kind is not None

    # Invariant: rendered_keys ⊆ all enqueued keys; last_rendered_turn never
    # exceeds the max turn we ever stamped (no time travel).
    all_keys = {(n.kind, n.task_id) for n in q.notes()}
    assert q.rendered_keys() <= all_keys
    for kind in ("looping_tool_call", "off_topic"):
        lr = q.last_rendered_turn(kind, "t1")
        assert lr == -1 or 0 <= lr <= 6
        # signal_count is the number of distinct enqueued notes for the key.
        assert q.signal_count(kind, "t1") == sum(
            1 for n in q.notes() if n.kind == kind and n.task_id == "t1"
        )
    # Invariant: every resolved ledger entry has a terminal outcome in the
    # visibility-attributed set (never after_signal for an unrendered key).
    rendered = q.rendered_keys()
    for entry in ledger.entries():
        if entry.outcome == SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL:
            assert (entry.drift_kind, entry.task_id) in rendered
