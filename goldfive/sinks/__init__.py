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

``InMemorySink`` and the protobuf-backed logging and persistence sinks are
always importable. The gRPC sink depends on ``proto``, which carries the gRPC
and code-generation tools. Importing that sink without its extra raises
:class:`ImportError` with a friendly message. The top-level ``goldfive``
module handles that import gracefully so the rest of the public API remains
usable without the gRPC tooling.

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
