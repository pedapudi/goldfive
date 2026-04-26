"""Tests asserting globally-unique ``event_id`` across simulated outer-session pin.

goldfive#271 Phase 3 Addition B: ``event_id`` was introduced because the
existing ``(session_id, run_id, sequence)`` PK breaks when an outer system
collapses multiple Sessions onto the same outer-session id (harmonograf#61).
Each turn restarts ``Session._next_sequence`` at 0, so two turns sharing
the outer ``session_id`` would produce ``(outer-session, run-A, 0)`` and
``(outer-session, run-B, 0)`` — distinct PK rows, fine. But once the
outer pin collapses ``run-A`` and ``run-B`` onto the same ``run_id``
(harmonograf#61's "stamp client run_id" path), the two seq-0 events
collide on the composite PK and the SQLite sink's ``INSERT OR IGNORE``
silently drops the second.

The new ``event_id`` field includes a uuid4 suffix so the field is
globally unique regardless of outer-session pin or run_id collapse.
"""

from __future__ import annotations

from goldfive.events import (
    new_event,
    run_started_event,
    task_completed_event,
    task_started_event,
)
from goldfive.types import Session


def test_two_turns_same_run_id_no_collisions() -> None:
    """Two Sessions sharing a (collapsed) run_id produce distinct event_ids.

    Simulates harmonograf#61's outer-session-pin failure mode: two
    turns collapse onto the same outer ``run_id``. Without the
    Phase 3 event_id, sequence-0 events from both turns would collide
    on the (session_id, run_id, sequence) PK; with event_id they don't.
    """
    # Two distinct goldfive Sessions that — in the harmonograf outer-pin
    # world — share the same effective run_id at the sink.
    turn1 = Session(run_id="outer-pin-run")
    turn2 = Session(run_id="outer-pin-run")

    seq1, eid1 = turn1.next_sequence_and_event_id()
    seq2, eid2 = turn2.next_sequence_and_event_id()

    # Both turns minted sequence=0 (counters restart per Session) — that's
    # the failure shape pre-#271.
    assert seq1 == seq2 == 0
    # But event_ids are globally unique because of the uuid4 suffix.
    assert eid1 != eid2

    # Stamp them on real Event envelopes; assert the stamping survives.
    e1 = new_event(turn1.run_id, seq1, event_id=eid1)
    e2 = new_event(turn2.run_id, seq2, event_id=eid2)
    assert e1.event_id != e2.event_id
    # Sequence and run_id remain identical — only event_id differentiates.
    assert e1.run_id == e2.run_id == "outer-pin-run"
    assert e1.sequence == e2.sequence == 0


def test_typed_factory_preserves_event_id_when_passed() -> None:
    """Typed factories thread ``event_id`` through ``new_event`` correctly."""
    s = Session(run_id="run-typed")
    seq, eid = s.next_sequence_and_event_id()
    evt = task_started_event(s.run_id, seq, "task-1", "starting", event_id=eid)
    assert evt.event_id == eid


def test_typed_factory_synthesizes_event_id_when_omitted() -> None:
    """Typed factories without explicit event_id still get a unique stamp.

    Out-of-band emit sites that don't have a Session in scope (test
    scaffolding, legacy producers) still get a valid uuid-suffixed
    event_id from the ``new_event`` fallback path.
    """
    a = task_started_event("run-x", 0, "task-a", "")
    b = task_started_event("run-x", 0, "task-b", "")
    assert a.event_id != b.event_id


def test_run_started_event_has_event_id() -> None:
    s = Session(run_id="run-started")
    seq, eid = s.next_sequence_and_event_id()
    evt = run_started_event(s.run_id, seq, "demo goal", event_id=eid)
    assert evt.event_id == eid


def test_task_completed_event_has_event_id() -> None:
    s = Session(run_id="run-completed")
    seq, eid = s.next_sequence_and_event_id()
    evt = task_completed_event(s.run_id, seq, "task-c", "done", artifacts={"k": "v"}, event_id=eid)
    assert evt.event_id == eid


def test_three_turns_collapsed_no_pk_collisions() -> None:
    """Realistic 3-turn scenario: all events from all turns have unique event_ids.

    Stand-in for the SQLite sink round-trip — the per-turn seq-0
    events would collide on the composite PK; with event_id they
    don't.
    """
    # Three turns "collapsed" onto the same outer run_id by harmonograf.
    turns = [Session(run_id="collapsed") for _ in range(3)]

    all_event_ids = []
    for turn in turns:
        # Each turn emits 5 events; under outer-pin collapse the
        # sequences are 0..4, 0..4, 0..4 — exactly the failure shape.
        for _ in range(5):
            seq, eid = turn.next_sequence_and_event_id()
            evt = new_event(turn.run_id, seq, event_id=eid)
            all_event_ids.append(evt.event_id)

    # 15 events, all distinct event_ids (no PK collisions).
    assert len(all_event_ids) == 15
    assert len(set(all_event_ids)) == 15


def test_event_id_round_trip_through_proto_serialization() -> None:
    """``event_id`` survives proto serialize/deserialize."""
    from goldfive.pb.goldfive.v1 import events_pb2

    s = Session(run_id="run-rt")
    seq, eid = s.next_sequence_and_event_id()
    evt = new_event(s.run_id, seq, session_id=s.id, event_id=eid)
    evt.task_started.task_id = "t1"

    # Round-trip through serialize/parse.
    raw = evt.SerializeToString()
    decoded = events_pb2.Event()
    decoded.ParseFromString(raw)

    assert decoded.event_id == eid
    assert decoded.run_id == s.run_id
    assert decoded.sequence == seq
    assert decoded.session_id == s.id
