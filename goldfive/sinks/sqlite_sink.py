"""SQLite persistence sink and replay helpers.

:class:`SQLitePersistenceSink` writes each event as a single row in a
SQLite table. The table carries the minimum needed for cross-run query
and replay::

    CREATE TABLE goldfive_events (
        run_id       TEXT    NOT NULL,
        sequence     INTEGER NOT NULL,
        emitted_at   INTEGER NOT NULL,
        kind         TEXT    NOT NULL,
        payload_json TEXT    NOT NULL,
        PRIMARY KEY (run_id, sequence)
    )

The ``(run_id, sequence)`` primary key is what makes per-run replay an
indexed lookup, and unique-per-run sequence numbers fall out for free.

The sink is async-safe: concurrent ``emit`` coroutines are serialised by
an :class:`asyncio.Lock` so the shared connection is only used by one
writer at a time. SQLite writes inside the lock are synchronous — that's
fine for the event volumes goldfive runs produce and keeps the
dependency surface to stdlib only.

JSONL's sibling module (:mod:`goldfive.sinks.persistence`) is the richer
format: per-event JSON lines with full proto round-trip. SQLite trades
a little replay fidelity for cross-run queryability. The two sinks can
coexist on the same run.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.protobuf.json_format import MessageToJson, Parse

if TYPE_CHECKING:
    from goldfive.pb.goldfive.v1 import events_pb2 as _events_pb2  # noqa: F401


_DEFAULT_TABLE = "goldfive_events"


def _events_module() -> Any:
    """Lazy-import the generated ``events_pb2`` module."""
    try:
        from goldfive.pb.goldfive.v1 import events_pb2
    except ModuleNotFoundError as exc:  # pragma: no cover — exercised by tests
        raise ModuleNotFoundError(
            "goldfive protobuf stubs not available; generate them via "
            "`make proto` (requires the `proto` optional-dependency group) "
            "or install the package with the `proto` extra. See issue #3."
        ) from exc
    return events_pb2


def _validate_table(name: str) -> str:
    """Return ``name`` if it is a safe SQL identifier, else raise.

    SQLite parameter binding does not cover table names so we template
    the table name into the DDL/DML ourselves. Accept the conservative
    ``[A-Za-z_][A-Za-z0-9_]*`` shape to keep the surface injection-free.
    """
    if not name:
        raise ValueError("table name must be non-empty")
    if not (name[0].isalpha() or name[0] == "_"):
        raise ValueError(f"invalid table name {name!r}")
    if not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(f"invalid table name {name!r}")
    return name


def _event_fields(event: Any) -> tuple[str, int, int, str, str]:
    """Extract ``(run_id, sequence, emitted_at_ms, kind, payload_json)``.

    Proto ``Event`` messages are serialised via ``MessageToJson`` with
    sorted keys. Dict-shaped events fall through to ``json.dumps`` so
    tests that pre-date the proto wiring still round-trip.
    """
    if hasattr(event, "DESCRIPTOR"):
        run_id = str(getattr(event, "run_id", "") or "")
        sequence = int(getattr(event, "sequence", 0) or 0)
        emitted_at_ms = 0
        if event.HasField("emitted_at"):
            ts = event.emitted_at
            emitted_at_ms = int(ts.seconds) * 1000 + int(ts.nanos) // 1_000_000
        kind = event.WhichOneof("payload") or ""
        payload_json = MessageToJson(event, sort_keys=True, indent=None)
        return run_id, sequence, emitted_at_ms, kind, payload_json

    if hasattr(event, "to_dict"):
        data = event.to_dict()
    elif isinstance(event, dict):
        data = dict(event)
    else:
        data = {k: v for k, v in vars(event).items() if not k.startswith("_")}

    run_id = str(data.get("run_id", "") or "")
    sequence = int(data.get("sequence", 0) or 0)
    emitted_at_ms = int(data.get("emitted_at_ms", data.get("emitted_at", 0)) or 0)
    kind = str(data.get("kind", "") or "")
    payload_json = json.dumps(data, sort_keys=True, default=str)
    return run_id, sequence, emitted_at_ms, kind, payload_json


class SQLitePersistenceSink:
    """EventSink that persists events into a SQLite table.

    Parameters
    ----------
    path:
        Filesystem path to the SQLite database. Parent directories are
        created on first emit if they do not already exist. Pass
        ``":memory:"`` for an in-process database — useful in tests.
    table:
        Name of the table to write into. Defaults to ``goldfive_events``.
        Must match ``[A-Za-z_][A-Za-z0-9_]*``; see :func:`_validate_table`.

    Notes
    -----
    The connection is opened lazily on the first ``emit`` so building
    the sink is side-effect-free and can happen off-loop. ``close``
    closes the connection; subsequent emits reopen it.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        table: str = _DEFAULT_TABLE,
    ) -> None:
        self._path = path if path == ":memory:" else Path(path)
        self._table = _validate_table(table)
        self._conn: sqlite3.Connection | None = None
        # Created lazily so the sink can be built off-loop and attached
        # to an event loop later (asyncio.Lock binds to the running loop
        # at first await).
        self._lock = asyncio.Lock()

    def _open(self) -> None:
        if self._conn is not None:
            return
        if isinstance(self._path, Path):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            target: str = str(self._path)
        else:
            target = self._path
        # ``isolation_level=None`` puts sqlite3 in autocommit mode so
        # each INSERT lands immediately — matches the write-through
        # semantics of the JSONL sink.
        self._conn = sqlite3.connect(target, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {self._table} (
                run_id       TEXT    NOT NULL,
                sequence     INTEGER NOT NULL,
                emitted_at   INTEGER NOT NULL,
                kind         TEXT    NOT NULL,
                payload_json TEXT    NOT NULL,
                PRIMARY KEY (run_id, sequence)
            )"""
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._table}_run_id "
            f"ON {self._table}(run_id)"
        )

    async def emit(self, event: Any) -> None:
        """Insert ``event`` into the events table.

        Proto messages are serialised with ``MessageToJson(sort_keys=True)``
        so :func:`replay_from_sqlite` can round-trip via ``Parse``.
        Dict-shaped events store the dict JSON directly. Concurrent
        callers are serialised by an :class:`asyncio.Lock`.
        """
        row = _event_fields(event)
        async with self._lock:
            self._open()
            assert self._conn is not None
            self._conn.execute(
                f"INSERT INTO {self._table} "
                "(run_id, sequence, emitted_at, kind, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                row,
            )

    async def close(self) -> None:
        """Close the underlying connection. Safe to call repeatedly."""
        async with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def replay_from_sqlite(
    path: str | Path,
    run_id: str,
    *,
    table: str = _DEFAULT_TABLE,
) -> list[Any]:
    """Return all events for ``run_id`` as parsed ``Event`` messages.

    Rows are fetched ordered by ``sequence`` so the return value is the
    emit-order timeline for the run. Each ``payload_json`` is parsed via
    ``google.protobuf.json_format.Parse`` into a fresh ``Event``; rows
    whose JSON is not a valid ``Event`` propagate the parser's exception.
    """
    pb = _events_module()
    table = _validate_table(table)
    target = str(path) if path == ":memory:" else str(Path(path))
    events: list[Any] = []
    conn = sqlite3.connect(target)
    try:
        cursor = conn.execute(
            f"SELECT payload_json FROM {table} WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        )
        for (payload_json,) in cursor:
            msg = pb.Event()
            Parse(payload_json, msg)
            events.append(msg)
    finally:
        conn.close()
    return events


def list_runs(
    path: str | Path,
    *,
    table: str = _DEFAULT_TABLE,
) -> list[str]:
    """Return the distinct ``run_id`` values present in the database.

    Ordered by the earliest ``sequence`` seen for each run so the
    return value roughly follows emit order. Returns an empty list if
    the database or table does not yet exist.
    """
    table = _validate_table(table)
    target = str(path) if path == ":memory:" else str(Path(path))
    conn = sqlite3.connect(target)
    try:
        # If the table isn't there yet treat the database as empty.
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if row is None:
            return []
        # Order by ROWID rather than emitted_at so ``list_runs`` reflects
        # the actual insertion order of the first row per run. emitted_at
        # can collide across runs when clocks are low-resolution or when
        # tests use synthetic timestamps; rowid never does.
        cursor = conn.execute(
            f"SELECT run_id FROM {table} GROUP BY run_id ORDER BY MIN(rowid)"
        )
        return [run_id for (run_id,) in cursor]
    finally:
        conn.close()
