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
    # Enqueue a first note manually so _route_corrective_note sees one prior.
    ObserverNoteQueue.for_session(session).enqueue(
        body="Observation: search_web looped",
        observation="search_web was called 5 times with identical args",
        severity="warning", drift_id="d1", kind="looping_tool_call",
        task_id="t1", turn=0,
    )
    await steerer.drift._route_corrective_note(
        session, _drift(), "Observation: still looping", ladder_level="signal"
    )
    notes = ObserverNoteQueue.for_session(session).notes()
    second = [n for n in notes if n.drift_id != "d1"]
    assert len(second) == 1
    # The 2nd note's body quotes the first note's observation.
    assert "repeats an earlier observer note" in second[0].body
    assert "search_web was called 5 times" in second[0].body


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
