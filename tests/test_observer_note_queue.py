"""Unit + property-based tests for :class:`ObserverNoteQueue` (PR 6).

AGENCY-PRESERVATION.md PR 6, §5.2 + §5.5. The queue is the StateStore-backed
substrate the four observer-note delivery surfaces share. Its load-bearing
invariants — exactly-once delivery via the ``delivered`` flag, per-request
coalescing (≤1 block, most-severe wins), stable goldfive-minted keys (§5.6),
and a state that is always parseable — are hammered here directly so the
interleaving tests can stress the bookkeeping without standing up a dispatch
path. Concurrency is where this codebase's bugs live (§5.5), so the mutators
are idempotent folds and the hypothesis tests assert that property.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from goldfive.observer_note_queue import (
    KEY_OBSERVER_NOTE_QUEUE,
    OBSERVER_NOTE_BLOCK_BEGIN,
    OBSERVER_NOTE_BLOCK_END,
    OBSERVER_NOTE_MARKER_PREFIX,
    ObserverNote,
    ObserverNoteQueue,
    render_block,
    render_tool_annotation,
    strip_prior_block,
)


def _enqueue(
    q: ObserverNoteQueue,
    *,
    severity: str = "warning",
    drift_id: str = "d1",
    kind: str = "looping_tool_call",
    task_id: str = "t1",
    turn: int = 0,
    body: str | None = None,
    observation: str = "obs",
) -> ObserverNote:
    return q.enqueue(
        body=body if body is not None else f"Observation: {observation}",
        observation=observation,
        severity=severity,
        drift_id=drift_id,
        kind=kind,
        task_id=task_id,
        agent_id="agent",
        turn=turn,
        ladder_level="nudge",
    )


# ---------------------------------------------------------------------------
# enqueue / dedup
# ---------------------------------------------------------------------------


def test_enqueue_then_pending() -> None:
    state: dict[str, Any] = {}
    q = ObserverNoteQueue(state)
    note = _enqueue(q)
    assert note.note_id == "d1"  # minted from drift_id
    assert [n.note_id for n in q.pending()] == ["d1"]
    # Persisted under the goldfive-prefixed slot, parseable shape.
    assert KEY_OBSERVER_NOTE_QUEUE in state
    assert isinstance(state[KEY_OBSERVER_NOTE_QUEUE]["notes"], list)


def test_enqueue_dedup_by_drift_id() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="d1", severity="info")
    _enqueue(q, drift_id="d1", severity="critical")  # same id -> coalesce
    pend = q.pending()
    assert len(pend) == 1
    # Re-enqueue refreshed the still-pending note's severity (latest wins).
    assert pend[0].severity == "critical"


def test_distinct_drift_ids_are_distinct_notes() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="d1")
    _enqueue(q, drift_id="d2")
    assert len(q.pending()) == 2


def test_note_id_minted_from_content_when_drift_id_empty() -> None:
    q = ObserverNoteQueue({})
    n = _enqueue(q, drift_id="")
    assert n.note_id.startswith("n_")  # content hash, not LLM-minted
    # Stable: same content re-mints the same id (coalesces).
    _enqueue(q, drift_id="")
    assert len(q.pending()) == 1


# ---------------------------------------------------------------------------
# coalescing — most-severe wins
# ---------------------------------------------------------------------------


def test_peek_picks_most_severe() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="a", severity="info")
    _enqueue(q, drift_id="b", severity="critical")
    _enqueue(q, drift_id="c", severity="warning")
    assert q.peek_for_render().note_id == "b"


def test_peek_tie_breaks_toward_newest() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="a", severity="warning", turn=1)
    _enqueue(q, drift_id="b", severity="warning", turn=3)  # newer turn wins
    _enqueue(q, drift_id="c", severity="warning", turn=2)
    assert q.peek_for_render().note_id == "b"


def test_peek_kinds_filter() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="a", kind="goal_drift", severity="critical")
    _enqueue(q, drift_id="b", kind="looping_tool_call", severity="warning")
    loop = frozenset({"looping_tool_call", "looping_reasoning"})
    # The critical goal_drift is most-severe overall, but the loop filter
    # selects only the loop-shaped note (surface 4 discipline).
    assert q.peek_for_render().note_id == "a"
    assert q.peek_for_render(kinds=loop).note_id == "b"


def test_peek_none_when_empty() -> None:
    assert ObserverNoteQueue({}).peek_for_render() is None


# ---------------------------------------------------------------------------
# mark_delivered — exactly-once flag
# ---------------------------------------------------------------------------


def test_mark_delivered_idempotent() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="d1")
    assert (
        q.mark_delivered("d1", channel="request_context", turn=1, surface="before_model")
        is True
    )
    # Second mark is a no-op — exactly-once.
    assert (
        q.mark_delivered("d1", channel="request_context", turn=2, surface="boundary_replay")
        is False
    )


def test_delivered_note_skipped_by_peek() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="d1")
    q.mark_delivered("d1", channel="request_context", turn=1)
    assert q.peek_for_render() is None  # not pending anymore
    assert q.pending() == []


def test_delivered_note_not_resurrected_by_reenqueue() -> None:
    q = ObserverNoteQueue({})
    _enqueue(q, drift_id="d1")
    q.mark_delivered("d1", channel="request_context", turn=1)
    # Re-enqueuing the same drift must NOT re-open the delivered note.
    again = _enqueue(q, drift_id="d1", severity="critical")
    assert again.delivered is True
    assert q.peek_for_render() is None


def test_mark_unknown_id_is_false() -> None:
    q = ObserverNoteQueue({})
    assert q.mark_delivered("nope", channel="request_context", turn=0) is False


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_render_block_shape() -> None:
    note = ObserverNote(
        note_id="d1",
        body="Observation: x\nThe user's goal: g\nThis note is advisory.",
        observation="x",
        severity="warning",
    )
    block = render_block(note)
    assert block.startswith(OBSERVER_NOTE_BLOCK_BEGIN)
    assert block.rstrip().endswith(OBSERVER_NOTE_BLOCK_END)
    assert OBSERVER_NOTE_MARKER_PREFIX in block
    assert "Observation: x" in block
    # Exactly one marker pair.
    assert block.count(OBSERVER_NOTE_MARKER_PREFIX) == 1
    assert block.count(OBSERVER_NOTE_BLOCK_END) == 1


def test_render_tool_annotation_compact_and_attributed() -> None:
    note = ObserverNote(
        note_id="d1",
        body="Observation: `search_web` was invoked 5 times ...",
        observation="`search_web` was invoked 5 times with identical arguments",
        severity="warning",
    )
    ann = render_tool_annotation(note)
    assert ann == "[goldfive observer: `search_web` was invoked 5 times with identical arguments]"
    # Compact — a single line, no block markers (it rides the tool result).
    assert OBSERVER_NOTE_MARKER_PREFIX not in ann
    assert "\n" not in ann


def test_strip_prior_block_round_trip() -> None:
    base = "You are a helpful agent."
    note = ObserverNote(note_id="d1", body="Observation: x", observation="x", severity="info")
    combined = base + "\n\n" + render_block(note)
    stripped = strip_prior_block(combined)
    assert OBSERVER_NOTE_MARKER_PREFIX not in stripped
    assert "helpful agent" in stripped
    # Idempotent: stripping again is a no-op.
    assert strip_prior_block(stripped) == stripped


def test_strip_prior_block_no_marker_is_identity() -> None:
    assert strip_prior_block("no markers here") == "no markers here"


# ---------------------------------------------------------------------------
# cap / prune
# ---------------------------------------------------------------------------


def test_prune_evicts_oldest_delivered_first() -> None:
    from goldfive.observer_note_queue import _NOTES_CAP

    q = ObserverNoteQueue({})
    # Fill to the cap, all delivered.
    for i in range(_NOTES_CAP):
        _enqueue(q, drift_id=f"d{i}")
        q.mark_delivered(f"d{i}", channel="request_context", turn=i)
    # One more pending note over the cap evicts the oldest delivered.
    _enqueue(q, drift_id="fresh", severity="critical")
    ids = {n.note_id for n in q.notes()}
    assert "fresh" in ids  # pending note retained
    assert "d0" not in ids  # oldest delivered evicted
    assert len(q.notes()) <= _NOTES_CAP


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def test_note_dict_round_trip() -> None:
    note = ObserverNote(
        note_id="d1",
        body="b",
        observation="o",
        severity="critical",
        drift_id="d1",
        kind="looping_tool_call",
        task_id="t1",
        agent_id="a",
        turn=3,
        ladder_level="nudge",
        enqueued_seq=2,
        delivered=True,
        delivered_channel="request_context",
        delivered_surface="before_model",
        delivered_turn=4,
    )
    assert ObserverNote.from_dict(note.to_dict()) == note


# ---------------------------------------------------------------------------
# property-based interleaving (§5.5)
# ---------------------------------------------------------------------------


_OPS = st.lists(
    st.tuples(
        st.sampled_from(["enqueue", "deliver"]),
        st.sampled_from(["a", "b", "c", "d"]),  # drift ids
        st.sampled_from(["info", "warning", "critical"]),
        st.integers(min_value=0, max_value=5),  # turn
    ),
    min_size=0,
    max_size=40,
)


@settings(max_examples=200)
@given(ops=_OPS)
def test_interleaving_invariants(ops: list[tuple[str, str, str, int]]) -> None:
    state: dict[str, Any] = {}
    q = ObserverNoteQueue(state)
    delivered_true_count: dict[str, int] = {}
    for op, did, sev, turn in ops:
        if op == "enqueue":
            _enqueue(q, drift_id=did, severity=sev, turn=turn)
        else:
            if q.mark_delivered(did, channel="request_context", turn=turn):
                delivered_true_count[did] = delivered_true_count.get(did, 0) + 1
        # Invariant: state is always parseable into ObserverNote objects.
        for raw in state.get(KEY_OBSERVER_NOTE_QUEUE, {}).get("notes", []):
            ObserverNote.from_dict(raw)
    # Invariant: mark_delivered returns True at most once per note_id
    # (exactly-once chokepoint).
    assert all(c <= 1 for c in delivered_true_count.values())
    # Invariant: pending() and the delivered set partition the notes.
    all_notes = q.notes()
    pend = {n.note_id for n in q.pending()}
    deliv = {n.note_id for n in all_notes if n.delivered}
    assert pend.isdisjoint(deliv)
    assert pend | deliv == {n.note_id for n in all_notes}
    # Invariant: no pending note is ever selected after being delivered.
    peek = q.peek_for_render()
    if peek is not None:
        assert peek.is_pending
