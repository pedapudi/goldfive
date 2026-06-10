"""SignalLedger unit + property-based interleaving tests.

AGENCY-PRESERVATION.md PR 5 (#449/#452), §5.5. The ledger is observe-only
bookkeeping keyed ``(drift_kind, task_id)`` (the
:meth:`~goldfive.drift_observer.DefaultSteerer._record_refine_outcome` key
discipline — stable goldfive-minted task ids, never LLM-minted) that records
signal deliveries, drift re-fires, and resolution outcomes. It gates nothing.

The hypothesis tests hammer the four invariants the design names for new
concurrent state (the codebase's race history says concurrency is where its
bugs live): no double-count per ``drift_id``, monotone turn stamps, exactly
one terminal outcome per delivered key, and a ledger state that is always
parseable.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from goldfive.events import (
    SIGNAL_OUTCOME_ESCALATED,
    SIGNAL_OUTCOME_INVOCATION_ENDED,
    SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL,
    SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED,
    SIGNAL_OUTCOME_USER_INTERVENED,
    SIGNAL_OUTCOMES,
)
from goldfive.signal_ledger import (
    KEY_SIGNAL_LEDGER,
    DeliveryRecord,
    LedgerEntry,
    SignalLedger,
    compose_key,
)

# ---------------------------------------------------------------------------
# Focused unit tests
# ---------------------------------------------------------------------------


def test_compose_key_roundtrip_stable() -> None:
    assert compose_key("LOOPING_TOOL_CALL", "t1") == compose_key("LOOPING_TOOL_CALL", "t1")
    assert compose_key("A", "t1") != compose_key("B", "t1")
    assert compose_key("A", "t1") != compose_key("A", "t2")


def test_record_fire_creates_entry_and_dedups_by_drift_id() -> None:
    led = SignalLedger({})
    led.record_fire(drift_kind="K", task_id="t1", turn=1, drift_id="d1")
    led.record_fire(drift_kind="K", task_id="t1", turn=3, drift_id="d1")  # dup id
    led.record_fire(drift_kind="K", task_id="t1", turn=5, drift_id="d2")
    e = led.entry("K", "t1")
    assert e is not None
    assert e.fire_count == 2  # d1 counted once, d2 once
    assert e.fired_drift_ids == ["d1", "d2"]
    assert e.first_fire_turn == 1
    assert e.last_fire_turn == 5
    assert e.is_open
    assert not e.has_delivery


def test_record_delivery_dedups_by_drift_id_and_channel() -> None:
    led = SignalLedger({})
    _, rec1 = led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=2, dry_run=True, note_text="n",
    )
    _, rec2 = led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=2, dry_run=True,
    )
    # Same drift on a DIFFERENT channel is a distinct delivery.
    _, rec3 = led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="steer_control",
        turn=2, dry_run=True,
    )
    assert rec1 is True
    assert rec2 is False  # deduped
    assert rec3 is True
    e = led.entry("K", "t1")
    assert e is not None
    assert len(e.deliveries) == 2
    assert e.fire_count == 1  # one drift_id, folded once
    assert e.first_delivery_turn == 2


def test_refire_count_counts_fires_after_first_delivery() -> None:
    led = SignalLedger({})
    led.record_fire(drift_kind="K", task_id="t1", turn=1, drift_id="d1")
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=1, dry_run=True,
    )
    # Two distinct re-fires after the delivery.
    led.record_fire(drift_kind="K", task_id="t1", turn=2, drift_id="d2")
    led.record_fire(drift_kind="K", task_id="t1", turn=3, drift_id="d3")
    e = led.entry("K", "t1")
    assert e is not None
    assert e.fire_count == 3
    assert e.refire_count == 2


def test_resolve_task_after_signal_when_real_delivery() -> None:
    led = SignalLedger({})
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=1, dry_run=False,  # REAL delivery
    )
    resolved = led.resolve_task(task_id="t1", turn=4)
    assert [e.outcome for e in resolved] == [SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL]
    assert resolved[0].turns_to_resolution() == 3
    assert resolved[0].has_real_delivery is True


def test_resolve_task_unaided_when_only_dry_run() -> None:
    led = SignalLedger({})
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=1, dry_run=True,  # dry-run only
    )
    resolved = led.resolve_task(task_id="t1", turn=2)
    assert [e.outcome for e in resolved] == [SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED]


def test_resolve_task_ignores_undelivered_keys() -> None:
    """A key with fires but NO delivery emits no outcome (outcomes pair 1:1
    with SignalDelivered)."""
    led = SignalLedger({})
    led.record_fire(drift_kind="K", task_id="t1", turn=1, drift_id="d1")
    resolved = led.resolve_task(task_id="t1", turn=2)
    assert resolved == []
    assert led.entry("K", "t1").is_open  # still open


def test_resolve_is_idempotent_one_terminal_outcome() -> None:
    led = SignalLedger({})
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=1, dry_run=True,
    )
    first = led.resolve_task(task_id="t1", turn=2)
    assert len(first) == 1
    # A second resolution attempt (any kind) is a no-op — exactly one outcome.
    assert led.resolve_task(task_id="t1", turn=3) == []
    assert led.resolve_user_intervened(turn=4) == []
    assert led.finalize_open(turn=5) == []
    e = led.entry("K", "t1")
    assert e.outcome == SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED
    assert e.outcome_turn == 2


def test_resolve_escalated() -> None:
    led = SignalLedger({})
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="pause_control",
        turn=1, dry_run=False,
    )
    e = led.resolve_escalated(drift_kind="K", task_id="t1", turn=2)
    assert e is not None and e.outcome == SIGNAL_OUTCOME_ESCALATED
    # Escalating an unknown / undelivered key is a no-op.
    assert led.resolve_escalated(drift_kind="K", task_id="other", turn=2) is None


def test_resolve_user_intervened_only_delivered_open_keys() -> None:
    led = SignalLedger({})
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=1, dry_run=True,
    )
    led.record_fire(drift_kind="K", task_id="t2", turn=1, drift_id="d2")  # no delivery
    resolved = led.resolve_user_intervened(turn=3)
    assert [(e.drift_kind, e.task_id, e.outcome) for e in resolved] == [
        ("K", "t1", SIGNAL_OUTCOME_USER_INTERVENED)
    ]
    # The undelivered key stays open (not over-attributed to the user).
    assert led.entry("K", "t2").is_open


def test_finalize_open_invocation_ended_and_idempotent() -> None:
    led = SignalLedger({})
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="steer_control",
        turn=1, dry_run=True,
    )
    resolved = led.finalize_open(turn=9)
    assert [e.outcome for e in resolved] == [SIGNAL_OUTCOME_INVOCATION_ENDED]
    assert led.finalize_open(turn=10) == []  # idempotent


def test_entry_roundtrip_and_state_parseable() -> None:
    state: dict = {}
    led = SignalLedger(state)
    led.record_delivery(
        drift_kind="K", task_id="t1", drift_id="d1", channel="nudge_replay",
        turn=1, dry_run=True, severity="WARNING", ladder_level="nudge", note_text="obs",
    )
    led.resolve_task(task_id="t1", turn=2)
    # The whole state blob is JSON-serialisable (the JSONL sink requirement).
    blob = json.dumps(state)
    reloaded = json.loads(blob)
    e = LedgerEntry.from_dict(reloaded[KEY_SIGNAL_LEDGER][compose_key("K", "t1")])
    assert e.outcome == SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED
    assert e.deliveries[0].note_text == "obs"
    assert e.deliveries[0].severity == "WARNING"


def test_from_dict_tolerates_malformed() -> None:
    # Malformed / partial blobs never raise — they degrade to defaults.
    e = LedgerEntry.from_dict({"drift_kind": "K", "fire_count": "nan", "deliveries": "x"})
    assert e.drift_kind == "K"
    assert e.fire_count == 0
    assert e.deliveries == []
    d = DeliveryRecord.from_dict({"turn": None})
    assert d.turn == 0


def test_ledger_tolerates_malformed_state_blob() -> None:
    # A non-dict ledger value reads as empty rather than raising.
    led = SignalLedger({KEY_SIGNAL_LEDGER: "not a dict"})
    assert led.entries() == []
    # And a fresh write repairs it.
    led.record_fire(drift_kind="K", task_id="t1", turn=0, drift_id="d1")
    assert led.entry("K", "t1") is not None


# ---------------------------------------------------------------------------
# Property-based interleaving tests (§5.5)
# ---------------------------------------------------------------------------

_KINDS = ["K0", "K1"]
_TASKS = ["t0", "t1"]
_DRIFT_IDS = ["d0", "d1", "d2", "d3", "d4"]
_CHANNELS = ["nudge_replay", "steer_control", "pause_control", "promotion"]

_fire_op = st.tuples(
    st.just("fire"),
    st.sampled_from(_KINDS),
    st.sampled_from(_TASKS),
    st.integers(min_value=0, max_value=20),
    st.sampled_from(_DRIFT_IDS),
)
_deliver_op = st.tuples(
    st.just("deliver"),
    st.sampled_from(_KINDS),
    st.sampled_from(_TASKS),
    st.sampled_from(_DRIFT_IDS),
    st.sampled_from(_CHANNELS),
    st.integers(min_value=0, max_value=20),
    st.booleans(),
)
_resolve_task_op = st.tuples(
    st.just("resolve_task"), st.sampled_from(_TASKS), st.integers(min_value=0, max_value=20)
)
_escalate_op = st.tuples(
    st.just("escalate"),
    st.sampled_from(_KINDS),
    st.sampled_from(_TASKS),
    st.integers(min_value=0, max_value=20),
)
_user_op = st.tuples(st.just("user"), st.integers(min_value=0, max_value=20))
_finalize_op = st.tuples(st.just("finalize"), st.integers(min_value=0, max_value=20))

_op = st.one_of(
    _fire_op, _deliver_op, _resolve_task_op, _escalate_op, _user_op, _finalize_op
)


def _apply(led: SignalLedger, op: tuple) -> list[LedgerEntry]:
    """Apply one generated op; return any newly-resolved entries."""
    kind = op[0]
    if kind == "fire":
        _, k, t, turn, did = op
        led.record_fire(drift_kind=k, task_id=t, turn=turn, drift_id=did)
        return []
    if kind == "deliver":
        _, k, t, did, chan, turn, dry = op
        led.record_delivery(
            drift_kind=k, task_id=t, drift_id=did, channel=chan, turn=turn, dry_run=dry
        )
        return []
    if kind == "resolve_task":
        _, t, turn = op
        return led.resolve_task(task_id=t, turn=turn)
    if kind == "escalate":
        _, k, t, turn = op
        e = led.resolve_escalated(drift_kind=k, task_id=t, turn=turn)
        return [e] if e is not None else []
    if kind == "user":
        _, turn = op
        return led.resolve_user_intervened(turn=turn)
    if kind == "finalize":
        _, turn = op
        return led.finalize_open(turn=turn)
    raise AssertionError(f"unknown op {kind!r}")  # pragma: no cover


@settings(max_examples=300, deadline=None)
@given(ops=st.lists(_op, max_size=60))
def test_ledger_invariants_under_interleaved_ops(ops: list[tuple]) -> None:
    state: dict = {}
    led = SignalLedger(state)
    resolved_keys: list[tuple[str, str]] = []
    for op in ops:
        for entry in _apply(led, op):
            resolved_keys.append((entry.drift_kind, entry.task_id))

    # Invariant 1 — no double-count per drift_id, no duplicate delivery.
    for e in led.entries():
        assert len(e.fired_drift_ids) == len(set(e.fired_drift_ids)), (
            "a drift_id was counted as a fire more than once"
        )
        assert e.fire_count == len(e.fired_drift_ids)
        seen_deliveries = [(d.drift_id, d.channel) for d in e.deliveries]
        assert len(seen_deliveries) == len(set(seen_deliveries)), (
            "a (drift_id, channel) delivery was recorded more than once"
        )

    # Invariant 2 — monotone / sane turn stamps.
    for e in led.entries():
        if e.first_fire_turn >= 0:
            assert e.first_fire_turn <= e.last_fire_turn
        assert e.turns_to_resolution() >= 0
        for d in e.deliveries:
            assert d.turn >= 0

    # Invariant 3 — exactly one terminal outcome per delivered key.
    assert len(resolved_keys) == len(set(resolved_keys)), (
        "a key was resolved to a terminal outcome more than once"
    )
    for e in led.entries():
        if not e.is_open:
            assert e.outcome in SIGNAL_OUTCOMES
            assert e.has_delivery, "an undelivered key must never be resolved"

    # Invariant 4 — ledger state is always parseable + round-trips.
    blob = json.dumps(state)
    reloaded = SignalLedger(json.loads(blob))
    assert {(e.drift_kind, e.task_id, e.outcome) for e in reloaded.entries()} == {
        (e.drift_kind, e.task_id, e.outcome) for e in led.entries()
    }


@settings(max_examples=200, deadline=None)
@given(
    ids=st.lists(st.sampled_from(_DRIFT_IDS), max_size=30),
    turns=st.lists(st.integers(min_value=0, max_value=15), max_size=30),
)
def test_fire_dedup_idempotent_regardless_of_interleaving(
    ids: list[str], turns: list[int]
) -> None:
    """Firing the same id repeatedly at arbitrary turns never inflates the
    count past the number of distinct ids."""
    led = SignalLedger({})
    n = min(len(ids), len(turns))
    for i in range(n):
        led.record_fire(drift_kind="K", task_id="t", turn=turns[i], drift_id=ids[i])
    if n == 0:
        assert led.entry("K", "t") is None
        return
    e = led.entry("K", "t")
    assert e.fire_count == len(set(ids[:n]))
    assert e.first_fire_turn == min(turns[:n])
    assert e.last_fire_turn == max(turns[:n])


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
