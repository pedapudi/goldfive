"""Unit tests for :class:`SQLitePersistenceSink`.

Covers:
* proto round-trip via emit/close + replay_from_sqlite
* concurrent emits do not corrupt the primary key or drop rows
* list_runs / replay_from_sqlite across multiple run_ids in one DB
* non-proto (dict-shaped) events still persist
* helpers tolerate a missing table cleanly
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from goldfive.pb.goldfive.v1 import events_pb2
from goldfive.sinks import (
    SQLitePersistenceSink,
    list_runs,
    replay_from_sqlite,
)


def _make_event(seq: int, run_id: str = "run-1") -> events_pb2.Event:
    """Build a minimal valid Event with a TaskStarted payload."""
    evt = events_pb2.Event(event_id=f"evt-{run_id}-{seq}", run_id=run_id, sequence=seq)
    evt.emitted_at.seconds = 1_700_000_000 + seq
    evt.task_started.task_id = f"task-{seq}"
    evt.task_started.detail = f"starting {seq}"
    return evt


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


async def test_sqlite_sink_roundtrip(tmp_path) -> None:
    db = tmp_path / "events.db"
    sink = SQLitePersistenceSink(db)
    events = [_make_event(i) for i in range(5)]
    for e in events:
        await sink.emit(e)
    await sink.close()

    replayed = replay_from_sqlite(db, run_id="run-1")
    assert len(replayed) == 5
    for original, round_tripped in zip(events, replayed, strict=True):
        assert original == round_tripped


async def test_sqlite_sink_creates_parent_dirs(tmp_path) -> None:
    db = tmp_path / "nested" / "a" / "events.db"
    sink = SQLitePersistenceSink(db)
    await sink.emit(_make_event(0))
    await sink.close()
    assert db.exists()


async def test_sqlite_sink_preserves_schema_columns(tmp_path) -> None:
    db = tmp_path / "schema.db"
    sink = SQLitePersistenceSink(db)
    await sink.emit(_make_event(0, run_id="schema-run"))
    await sink.close()

    conn = sqlite3.connect(db)
    try:
        cursor = conn.execute(
            "SELECT run_id, sequence, emitted_at, kind FROM goldfive_events"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()
    assert rows == [("schema-run", 0, 1_700_000_000_000, "task_started")]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_sqlite_sink_concurrent_emit_persists_every_row(tmp_path) -> None:
    db = tmp_path / "concurrent.db"
    sink = SQLitePersistenceSink(db)
    n = 50
    events = [_make_event(i) for i in range(n)]
    await asyncio.gather(*(sink.emit(e) for e in events))
    await sink.close()

    replayed = replay_from_sqlite(db, run_id="run-1")
    assert len(replayed) == n
    replayed_ids = sorted(e.event_id for e in replayed)
    expected_ids = sorted(e.event_id for e in events)
    assert replayed_ids == expected_ids
    # Rows come back ordered by sequence even under concurrent writes.
    assert [e.sequence for e in replayed] == list(range(n))


# ---------------------------------------------------------------------------
# Multi-run queries
# ---------------------------------------------------------------------------


async def test_list_runs_and_replay_isolate_by_run_id(tmp_path) -> None:
    db = tmp_path / "multi.db"
    sink = SQLitePersistenceSink(db)
    for seq in range(3):
        await sink.emit(_make_event(seq, run_id="run-A"))
    for seq in range(2):
        await sink.emit(_make_event(seq, run_id="run-B"))
    await sink.close()

    runs = list_runs(db)
    assert set(runs) == {"run-A", "run-B"}

    events_a = replay_from_sqlite(db, run_id="run-A")
    events_b = replay_from_sqlite(db, run_id="run-B")
    assert [e.sequence for e in events_a] == [0, 1, 2]
    assert [e.sequence for e in events_b] == [0, 1]
    assert all(e.run_id == "run-A" for e in events_a)
    assert all(e.run_id == "run-B" for e in events_b)


async def test_list_runs_orders_by_first_emit(tmp_path) -> None:
    db = tmp_path / "ordering.db"
    sink = SQLitePersistenceSink(db)
    # run-B fires first, run-A second. list_runs should reflect that.
    await sink.emit(_make_event(0, run_id="run-B"))
    await sink.emit(_make_event(0, run_id="run-A"))
    await sink.emit(_make_event(1, run_id="run-B"))
    await sink.close()
    assert list_runs(db) == ["run-B", "run-A"]


def test_list_runs_on_missing_table_returns_empty(tmp_path) -> None:
    db = tmp_path / "empty.db"
    # Create an empty SQLite file with no tables.
    sqlite3.connect(db).close()
    assert list_runs(db) == []


def test_list_runs_on_missing_file_returns_empty(tmp_path) -> None:
    # sqlite3.connect creates the file on demand, so this should not raise;
    # the table check still reports empty.
    db = tmp_path / "nope.db"
    assert list_runs(db) == []


# ---------------------------------------------------------------------------
# Restart / append semantics
# ---------------------------------------------------------------------------


async def test_sqlite_sink_reopen_appends(tmp_path) -> None:
    db = tmp_path / "append.db"
    sink1 = SQLitePersistenceSink(db)
    await sink1.emit(_make_event(0))
    await sink1.emit(_make_event(1))
    await sink1.close()

    sink2 = SQLitePersistenceSink(db)
    await sink2.emit(_make_event(2))
    await sink2.emit(_make_event(3))
    await sink2.close()

    replayed = replay_from_sqlite(db, run_id="run-1")
    assert [e.sequence for e in replayed] == [0, 1, 2, 3]


async def test_sqlite_sink_duplicate_primary_key_raises(tmp_path) -> None:
    db = tmp_path / "dup.db"
    sink = SQLitePersistenceSink(db)
    await sink.emit(_make_event(0))
    with pytest.raises(sqlite3.IntegrityError):
        await sink.emit(_make_event(0))
    await sink.close()


# ---------------------------------------------------------------------------
# Custom table name
# ---------------------------------------------------------------------------


async def test_sqlite_sink_custom_table(tmp_path) -> None:
    db = tmp_path / "custom.db"
    sink = SQLitePersistenceSink(db, table="harmonograf_events")
    await sink.emit(_make_event(0, run_id="run-X"))
    await sink.close()

    replayed = replay_from_sqlite(db, run_id="run-X", table="harmonograf_events")
    assert len(replayed) == 1
    assert list_runs(db, table="harmonograf_events") == ["run-X"]
    # Default table does not carry the row.
    assert list_runs(db) == []


def test_sqlite_sink_rejects_injection_table_name(tmp_path) -> None:
    with pytest.raises(ValueError):
        SQLitePersistenceSink(tmp_path / "x.db", table="events; DROP TABLE")


# ---------------------------------------------------------------------------
# Non-proto events (dicts)
# ---------------------------------------------------------------------------


class _DictEvent:
    """Dict-shaped event exposing ``to_dict`` — matches the JSONL sink's
    duck-typed fallback path so both sinks accept the same shapes."""

    def __init__(self, run_id: str, sequence: int, kind: str, payload: dict) -> None:
        self.run_id = run_id
        self.sequence = sequence
        self.kind = kind
        self.payload = payload

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "emitted_at_ms": 42,
            "payload": self.payload,
        }


async def test_sqlite_sink_accepts_dict_events(tmp_path) -> None:
    db = tmp_path / "dicts.db"
    sink = SQLitePersistenceSink(db)
    await sink.emit(_DictEvent("run-dict", 0, "custom", {"x": 1}))
    await sink.emit(_DictEvent("run-dict", 1, "custom", {"x": 2}))
    await sink.close()

    assert list_runs(db) == ["run-dict"]

    # The row is there even if it isn't a proto-parseable Event: inspect
    # directly to avoid the Parse step in replay_from_sqlite.
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT run_id, sequence, kind, emitted_at FROM goldfive_events "
            "WHERE run_id = ? ORDER BY sequence",
            ("run-dict",),
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("run-dict", 0, "custom", 42),
        ("run-dict", 1, "custom", 42),
    ]
