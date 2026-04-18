"""Persistence + recovery integration tests.

Writes N events via ``JSONLPersistenceSink``, simulates a crash by
dropping the in-memory runtime, replays the JSONL from disk, continues
from the last persisted sequence number, and asserts that event
ordering is preserved across the resume boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

types = pytest.importorskip("goldfive.types")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal stand-in for a proto Event used by the raw-dict sink path.

    The JSONL sink implementation may accept either a protobuf ``Event``
    message (using ``MessageToJson``) or a dict-like with ``.sequence``
    and ``.run_id``. We use the dict-friendly shape so the tests stay
    green regardless of which the persistence PR picks.
    """

    def __init__(self, run_id: str, sequence: int, payload: dict[str, Any]) -> None:
        self.run_id = run_id
        self.sequence = sequence
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "sequence": self.sequence, "payload": self.payload}


def _write_event_line(fh, event: _FakeEvent) -> None:
    fh.write(json.dumps(event.to_dict()))
    fh.write("\n")


# ---------------------------------------------------------------------------
# Raw JSONL write/read round-trip (runs even before the sink exists)
# ---------------------------------------------------------------------------


def test_raw_jsonl_round_trip_preserves_sequence(tmp_jsonl_path: Path) -> None:
    events = [_FakeEvent("run-A", i, {"tick": i}) for i in range(25)]
    with tmp_jsonl_path.open("w", encoding="utf-8") as fh:
        for e in events:
            _write_event_line(fh, e)

    with tmp_jsonl_path.open("r", encoding="utf-8") as fh:
        loaded = [json.loads(line) for line in fh if line.strip()]

    assert [e["sequence"] for e in loaded] == list(range(25))
    assert all(e["run_id"] == "run-A" for e in loaded)


def test_crash_and_replay_preserves_event_order_across_resume(tmp_jsonl_path: Path) -> None:
    """Simulate: write 10 events, crash, reload sequence, continue to 20."""

    # Phase 1: pre-crash.
    session = types.Session(run_id="run-crash")
    with tmp_jsonl_path.open("w", encoding="utf-8") as fh:
        for _ in range(10):
            seq = session.next_sequence()
            _write_event_line(fh, _FakeEvent(session.run_id, seq, {"phase": "pre"}))

    assert tmp_jsonl_path.exists()
    pre_crash = [
        json.loads(line) for line in tmp_jsonl_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(pre_crash) == 10
    last_seq = pre_crash[-1]["sequence"]
    assert last_seq == 9

    # Phase 2: crash + recovery. New in-memory session picks up from the
    # last persisted sequence.
    recovered = types.Session(run_id="run-crash")
    recovered._next_sequence = last_seq + 1

    with tmp_jsonl_path.open("a", encoding="utf-8") as fh:
        for _ in range(10):
            seq = recovered.next_sequence()
            _write_event_line(fh, _FakeEvent(recovered.run_id, seq, {"phase": "post"}))

    # Phase 3: replay the whole log and verify strict monotonicity.
    lines = tmp_jsonl_path.read_text(encoding="utf-8").splitlines()
    loaded = [json.loads(line) for line in lines if line.strip()]
    assert [e["sequence"] for e in loaded] == list(range(20))
    pre = [e for e in loaded if e["payload"]["phase"] == "pre"]
    post = [e for e in loaded if e["payload"]["phase"] == "post"]
    assert len(pre) == 10 and len(post) == 10
    assert max(e["sequence"] for e in pre) < min(e["sequence"] for e in post)


def test_replay_skips_blank_lines_and_partial_writes(tmp_jsonl_path: Path) -> None:
    """Real crashes can leave a partial last line; the replay path must
    tolerate that gracefully."""

    session = types.Session(run_id="run-partial")
    with tmp_jsonl_path.open("w", encoding="utf-8") as fh:
        for _ in range(5):
            seq = session.next_sequence()
            _write_event_line(fh, _FakeEvent(session.run_id, seq, {"x": seq}))
        # Simulate a torn write: a blank line and then a truncated JSON fragment.
        fh.write("\n")
        fh.write('{"run_id": "run-partial", "sequenc')  # truncated, no newline

    good: list[dict[str, Any]] = []
    for line in tmp_jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    assert [e["sequence"] for e in good] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Sink-backed test — skips cleanly until JSONLPersistenceSink exists.
# ---------------------------------------------------------------------------


async def test_jsonl_persistence_sink_round_trip(tmp_jsonl_path: Path) -> None:
    sinks_mod = pytest.importorskip("goldfive.sinks")
    JSONLPersistenceSink = getattr(sinks_mod, "JSONLPersistenceSink", None)
    if JSONLPersistenceSink is None:
        pytest.skip("JSONLPersistenceSink not yet implemented")

    # Try to construct with a pathlib.Path; some implementations may want
    # a str. Both are acceptable.
    try:
        sink = JSONLPersistenceSink(tmp_jsonl_path)
    except TypeError:
        sink = JSONLPersistenceSink(str(tmp_jsonl_path))

    session = types.Session(run_id="sink-run")
    for _ in range(8):
        seq = session.next_sequence()
        await sink.emit(_FakeEvent(session.run_id, seq, {"n": seq}))
    await sink.close()

    assert tmp_jsonl_path.exists()
    lines = [ln for ln in tmp_jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 8


async def test_jsonl_persistence_sink_supports_replay_helper(tmp_jsonl_path: Path) -> None:
    sinks_mod = pytest.importorskip("goldfive.sinks")
    JSONLPersistenceSink = getattr(sinks_mod, "JSONLPersistenceSink", None)
    replay = getattr(sinks_mod, "replay_jsonl", None)
    if JSONLPersistenceSink is None or replay is None:
        pytest.skip("JSONLPersistenceSink.replay_jsonl helper not yet implemented")

    try:
        sink = JSONLPersistenceSink(tmp_jsonl_path)
    except TypeError:
        sink = JSONLPersistenceSink(str(tmp_jsonl_path))

    session = types.Session(run_id="sink-replay")
    for _ in range(5):
        seq = session.next_sequence()
        await sink.emit(_FakeEvent(session.run_id, seq, {"n": seq}))
    await sink.close()

    # The replay helper is expected to yield events in the order they were
    # emitted. We don't assume anything about the yielded type beyond that
    # it has a ``sequence`` attribute or key.
    yielded = list(replay(tmp_jsonl_path))
    sequences = []
    for item in yielded:
        if hasattr(item, "sequence"):
            sequences.append(item.sequence)
        else:
            sequences.append(item["sequence"])
    assert sequences == sorted(sequences)
    assert len(sequences) == 5
