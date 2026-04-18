"""Event sequence numbers must be strictly monotonic per run.

These tests verify the ``Session.next_sequence()`` contract pinned in
INTERFACE_SPEC.md and that executors/steerers that emit events through a
sink preserve strict ordering.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

types = pytest.importorskip("goldfive.types")


def test_next_sequence_is_monotonic_from_zero() -> None:
    session = types.Session(run_id="run-seq-1")
    seqs = [session.next_sequence() for _ in range(10)]
    assert seqs == list(range(10))


def test_next_sequence_is_independent_across_sessions() -> None:
    a = types.Session(run_id="a")
    b = types.Session(run_id="b")
    for _ in range(3):
        a.next_sequence()
    assert b.next_sequence() == 0
    assert a.next_sequence() == 3


def test_next_sequence_is_strictly_increasing_under_many_calls() -> None:
    session = types.Session(run_id="long")
    last = -1
    for _ in range(1000):
        s = session.next_sequence()
        assert s > last
        last = s


def test_next_sequence_is_not_reset_by_field_mutation() -> None:
    session = types.Session(run_id="run")
    session.next_sequence()
    session.next_sequence()
    # Mutating an unrelated field must not rewind the counter.
    session.current_task_id = "t1"
    assert session.next_sequence() == 2


class _CollectingSink:
    """Minimal in-memory sink for capturing emitted events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:  # pragma: no cover - trivial
        return None


async def test_sink_receives_events_in_strict_monotonic_order() -> None:
    """If sinks get events in emission order, the recorded sequences
    must be strictly increasing. We emulate an executor's emit loop."""

    session = types.Session(run_id="order-test")
    sink = _CollectingSink()

    class _Evt:
        def __init__(self, seq: int, run_id: str) -> None:
            self.sequence = seq
            self.run_id = run_id

    for _ in range(25):
        seq = session.next_sequence()
        await sink.emit(_Evt(seq=seq, run_id=session.run_id))

    sequences = [e.sequence for e in sink.events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))
    assert all(e.run_id == "order-test" for e in sink.events)


async def test_concurrent_callers_do_not_duplicate_sequence_numbers() -> None:
    """``next_sequence`` does not need to be thread-safe per the spec, but
    single-threaded async callers (the executor's invariant) must still
    observe unique values when interleaved via ``asyncio.gather``."""

    session = types.Session(run_id="concurrent")

    async def _take() -> int:
        # Yield control once to give the scheduler a chance to interleave.
        await asyncio.sleep(0)
        return session.next_sequence()

    values = await asyncio.gather(*[_take() for _ in range(50)])
    assert sorted(values) == list(range(50))
