"""Determinism tests for :mod:`goldfive.runtime`.

The contract :func:`goldfive.runtime.set_seed` advertises: two
goldfive runs with the same seed and the same input produce
byte-identical event streams (the ``events.jsonl`` files compare
equal).

This test pins:

* :func:`seeded_uuid4` produces deterministic v4-shaped UUIDs across
  two ``set_seed(N)`` resets.
* :meth:`Session.next_event_id` returns deterministic suffixes under
  the seed.
* :class:`DriftEvent` default ids are deterministic under the seed.
* Building proto ``Event`` envelopes (via :func:`goldfive.events.new_event`)
  produces byte-identical wire output across two seeded runs when the
  envelope's ``emitted_at`` timestamp is held constant (the only
  remaining source of non-determinism is wall clock, which the test
  zeroes out).
"""

from __future__ import annotations

import json

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.events import new_event  # noqa: E402
from goldfive.runtime import (  # noqa: E402
    clear_seed,
    is_seeded,
    seeded_uuid4,
    set_seed,
)
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Session,
)


@pytest.fixture(autouse=True)
def _clear_seed_around_each_test():
    """Make sure each test sees a clean seed state."""
    clear_seed()
    yield
    clear_seed()


def test_seeded_uuid4_is_deterministic() -> None:
    set_seed(123)
    a = [seeded_uuid4().hex for _ in range(5)]
    set_seed(123)
    b = [seeded_uuid4().hex for _ in range(5)]
    assert a == b


def test_seeded_uuid4_respects_version_and_variant_bits() -> None:
    set_seed(99)
    u = seeded_uuid4()
    # Version 4 — high nibble of byte 6 is 0b0100.
    assert (u.bytes[6] & 0xF0) >> 4 == 4
    # RFC 4122 variant — top two bits of byte 8 are 0b10.
    assert (u.bytes[8] & 0xC0) == 0x80


def test_seeded_uuid4_differs_from_clear_state() -> None:
    """Without a seed, two calls return non-identical UUIDs."""
    clear_seed()
    a = seeded_uuid4()
    b = seeded_uuid4()
    assert a != b


def test_session_next_event_id_is_deterministic_under_seed() -> None:
    set_seed(7)
    s1 = Session(run_id="run-x")
    ids_a = [s1.next_event_id() for _ in range(3)]
    set_seed(7)
    s2 = Session(run_id="run-x")
    ids_b = [s2.next_event_id() for _ in range(3)]
    assert ids_a == ids_b


def test_drift_event_default_id_is_deterministic_under_seed() -> None:
    set_seed(42)
    a = DriftEvent(
        kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING, detail=""
    ).id
    b = DriftEvent(
        kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING, detail=""
    ).id
    set_seed(42)
    a2 = DriftEvent(
        kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING, detail=""
    ).id
    b2 = DriftEvent(
        kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING, detail=""
    ).id
    assert (a, b) == (a2, b2)


def test_proto_event_envelope_byte_identical_when_timestamp_held() -> None:
    """Two seeded runs produce byte-identical ``Event`` envelopes.

    The ``emitted_at`` timestamp is wall-clock and not seeded; we
    overwrite it to a fixed value on both runs to factor it out (it is
    explicitly documented as advisory). Every other field on the
    envelope — ``event_id``, ``run_id``, ``sequence``, ``session_id`` —
    is deterministic.
    """
    from google.protobuf.timestamp_pb2 import Timestamp

    fixed_ts = Timestamp(seconds=1_700_000_000, nanos=0)

    def _build(seed: int) -> list[bytes]:
        set_seed(seed)
        out: list[bytes] = []
        for sequence in range(3):
            evt = new_event("run-det", sequence, session_id="session-det")
            evt.emitted_at.CopyFrom(fixed_ts)
            out.append(evt.SerializeToString())
        return out

    a = _build(2026)
    b = _build(2026)
    assert a == b


def test_two_seeded_runs_produce_identical_events_jsonl_stream() -> None:
    """End-to-end determinism: a small drift trace serialises identically.

    Simulates the events.jsonl stream by emitting a fixed sequence of
    events (run_started → drift_detected → run_completed), serialising
    each as JSON, and comparing the two seeded outputs.
    """
    from google.protobuf.timestamp_pb2 import Timestamp

    fixed_ts = Timestamp(seconds=1_700_000_000, nanos=0)

    def _trace(seed: int) -> str:
        set_seed(seed)
        session = Session(run_id="run-det")
        lines: list[str] = []
        for _ in range(3):
            seq = session.next_sequence()
            evt_id = session.next_event_id(seq)
            drift = DriftEvent(
                kind=DriftKind.OFF_TOPIC,
                severity=DriftSeverity.WARNING,
                detail="x",
                current_task_id="t1",
            )
            row = {
                "event_id": evt_id,
                "sequence": seq,
                "drift_id": drift.id,
                "emitted_at": fixed_ts.seconds,
            }
            lines.append(json.dumps(row, sort_keys=True))
        return "\n".join(lines)

    assert _trace(11) == _trace(11)
    # Different seeds → different outputs (sanity).
    assert _trace(11) != _trace(22)


def test_clear_seed_restores_non_deterministic_behaviour() -> None:
    set_seed(7)
    assert is_seeded()
    a = seeded_uuid4()
    clear_seed()
    assert not is_seeded()
    b = seeded_uuid4()
    c = seeded_uuid4()
    # Without a seed, consecutive UUIDs differ.
    assert b != c
    # Re-seeding produces the same first UUID as before.
    set_seed(7)
    a2 = seeded_uuid4()
    assert a == a2
