"""Tests for the ``event_id`` format minted by ``Session.next_event_id``.

goldfive#271 Phase 3 Addition B: every emitted ``Event`` envelope carries a
globally-unique ``event_id`` of the form ``{run_id}:{sequence}:{uuid4_short}``.
The ``(run_id, sequence)`` prefix preserves chronological sortability and
per-run debuggability; the uuid4 suffix guarantees PK uniqueness even when
an outer system collapses multiple Sessions onto the same outer-session id
(harmonograf#61).

These tests pin the format contract so downstream consumers (harmonograf's
PK migration, sink-side parsers, observability tooling) can rely on it.
"""

from __future__ import annotations

import re

from goldfive.events import make_event, new_event
from goldfive.types import Session


def test_next_event_id_format() -> None:
    """``Session.next_event_id`` returns ``{run_id}:{sequence}:{hex8}``."""
    s = Session(run_id="run-123")
    eid = s.next_event_id()
    parts = eid.split(":")
    assert len(parts) == 3, f"expected 3 colon-separated parts, got {eid!r}"
    run_id, seq_str, suffix = parts
    assert run_id == "run-123"
    assert seq_str.isdigit()
    assert int(seq_str) == 0  # first call returns sequence 0
    assert len(suffix) == 8
    # Suffix is hex (uuid4 first 8 chars).
    assert re.fullmatch(r"[0-9a-f]{8}", suffix), f"suffix {suffix!r} is not 8 hex chars"


def test_next_event_id_increments_sequence() -> None:
    """Each call advances the per-Session sequence counter."""
    s = Session(run_id="run-x")
    eids = [s.next_event_id() for _ in range(3)]
    seqs = [int(eid.split(":")[1]) for eid in eids]
    assert seqs == [0, 1, 2]


def test_next_event_id_with_explicit_sequence() -> None:
    """Passing ``sequence`` reuses the int without re-incrementing."""
    s = Session(run_id="run-y")
    seq = s.next_sequence()
    assert seq == 0
    eid = s.next_event_id(seq)
    assert eid.startswith("run-y:0:")
    # The counter was advanced exactly once (by next_sequence).
    next_seq = s.next_sequence()
    assert next_seq == 1


def test_next_sequence_and_event_id_atomic_pair() -> None:
    """``next_sequence_and_event_id`` increments once and returns the pair."""
    s = Session(run_id="run-z")
    seq1, eid1 = s.next_sequence_and_event_id()
    seq2, eid2 = s.next_sequence_and_event_id()
    assert seq1 == 0
    assert seq2 == 1
    assert eid1.startswith("run-z:0:")
    assert eid2.startswith("run-z:1:")
    assert eid1 != eid2  # uuid suffix differs


def test_next_event_id_uniqueness() -> None:
    """1000 successive calls produce 1000 distinct ids."""
    s = Session(run_id="run-bulk")
    ids = {s.next_event_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_next_event_id_distinct_across_sessions_with_same_run_id() -> None:
    """Two Sessions sharing a run_id still produce distinct ids per call.

    Real goldfive never reuses run_id across Sessions — but the uuid4
    suffix means even pathological reuse can't collide on the wire.
    """
    s1 = Session(run_id="dup-run")
    s2 = Session(run_id="dup-run")
    eid1 = s1.next_event_id()  # dup-run:0:<rand1>
    eid2 = s2.next_event_id()  # dup-run:0:<rand2>
    # Same prefix.
    assert eid1.split(":")[:2] == eid2.split(":")[:2] == ["dup-run", "0"]
    # Different suffix.
    assert eid1.split(":")[2] != eid2.split(":")[2]
    assert eid1 != eid2


# ---------------------------------------------------------------------------
# Envelope-level guarantees: new_event always stamps event_id
# ---------------------------------------------------------------------------


def test_new_event_synthesizes_event_id_when_omitted() -> None:
    """Calling ``new_event`` without ``event_id`` still produces a unique id."""
    a = new_event("run-a", 0)
    b = new_event("run-a", 0)
    # Both got synthesised event_ids — uuid suffix prevents collision.
    assert a.event_id
    assert b.event_id
    assert a.event_id != b.event_id
    # Format: run_id:sequence:hex8.
    parts = a.event_id.split(":")
    assert parts[0] == "run-a"
    assert parts[1] == "0"
    assert len(parts[2]) == 8


def test_new_event_honors_supplied_event_id() -> None:
    """Caller-supplied ``event_id`` is preserved verbatim."""
    evt = new_event("run-x", 5, event_id="explicit-id")
    assert evt.event_id == "explicit-id"


def test_make_event_dict_envelope_has_event_id() -> None:
    """The dict envelope path also stamps a unique ``event_id``."""
    a = make_event("run-d", 0, "task_started")
    b = make_event("run-d", 0, "task_started")
    assert a["event_id"] != b["event_id"]
    parts = a["event_id"].split(":")
    assert parts[0] == "run-d"
    assert parts[1] == "0"
    assert len(parts[2]) == 8


def test_make_event_honors_supplied_event_id() -> None:
    """Caller-supplied ``event_id`` is preserved verbatim on dict envelope."""
    d = make_event("run-y", 7, "task_completed", event_id="explicit-dict-id")
    assert d["event_id"] == "explicit-dict-id"
