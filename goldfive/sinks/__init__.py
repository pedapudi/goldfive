"""Built-in EventSinks for goldfive.

Five implementations ship in-box:

* :class:`InMemorySink` — collects events in a list; used by tests and
  ephemeral callers that want to inspect the stream after the run.
* :class:`LoggingSink` — renders each event as a single JSON line onto a
  :class:`logging.Logger`; drop-in for stdout/journald style observation.
* :class:`JSONLPersistenceSink` — appends events to a newline-delimited
  JSON file suitable for crash recovery. Paired with
  :func:`replay_from_jsonl` and :func:`reconstruct_session` for resume.
* :class:`SQLitePersistenceSink` — writes events into a SQLite table so
  dashboards and shared-DB integrations can query across runs. Paired
  with :func:`replay_from_sqlite` and :func:`list_runs`.
* :class:`GRPCSink` — forwards proto events over a client-streaming
  gRPC RPC to a ``GoldfiveIngress`` server (see ``goldfive.server``).

``InMemorySink`` is always importable. The others depend on the optional
``proto`` extra (``google.protobuf``, plus ``grpcio`` for :class:`GRPCSink`)
— importing them when the extra is missing raises :class:`ImportError`
with a friendly message. The top-level ``goldfive`` module handles that
import gracefully so the bulk of the public API stays usable without
``proto``.

All sinks conform to the ``EventSink`` protocol pinned in
``goldfive.protocols`` (async ``emit`` / async ``close``).
"""

from __future__ import annotations

from goldfive.sinks.memory import InMemorySink

try:
    from goldfive.sinks.logging_sink import LoggingSink
except ImportError:  # pragma: no cover — proto extra not installed
    LoggingSink = None  # type: ignore[assignment]

try:
    from goldfive.sinks.persistence import (
        JSONLPersistenceSink,
        reconstruct_session,
        replay_from_jsonl,
    )
except ImportError:  # pragma: no cover — proto extra not installed
    JSONLPersistenceSink = None  # type: ignore[assignment]
    reconstruct_session = None  # type: ignore[assignment]
    replay_from_jsonl = None  # type: ignore[assignment]

try:
    from goldfive.sinks.sqlite_sink import (
        SQLitePersistenceSink,
        list_runs,
        replay_from_sqlite,
    )
except ImportError:  # pragma: no cover — proto extra not installed
    SQLitePersistenceSink = None  # type: ignore[assignment]
    list_runs = None  # type: ignore[assignment]
    replay_from_sqlite = None  # type: ignore[assignment]

try:
    from goldfive.sinks.grpc_sink import GRPCSink
except ImportError:  # pragma: no cover — grpcio not installed
    GRPCSink = None  # type: ignore[assignment]

__all__ = [
    "GRPCSink",
    "InMemorySink",
    "JSONLPersistenceSink",
    "LoggingSink",
    "SQLitePersistenceSink",
    "list_runs",
    "reconstruct_session",
    "replay_from_jsonl",
    "replay_from_sqlite",
]
