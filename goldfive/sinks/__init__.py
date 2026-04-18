"""Built-in EventSinks for goldfive.

Three implementations ship in-box:

* :class:`InMemorySink` — collects events in a list; used by tests and
  ephemeral callers that want to inspect the stream after the run.
* :class:`LoggingSink` — renders each event as a single JSON line onto a
  :class:`logging.Logger`; drop-in for stdout/journald style observation.
* :class:`JSONLPersistenceSink` — appends events to a newline-delimited
  JSON file suitable for crash recovery. Paired with
  :func:`replay_from_jsonl` and :func:`reconstruct_session` for resume.

``InMemorySink`` is always importable. ``LoggingSink`` and
``JSONLPersistenceSink`` depend on the optional ``proto`` extra
(``google.protobuf``) — importing them when the extra is missing
raises :class:`ImportError` with a friendly message. The top-level
``goldfive`` module handles that import gracefully so the bulk of the
public API stays usable without ``proto``.

All sinks conform to the ``EventSink`` protocol pinned in
``INTERFACE_SPEC.md`` (async ``emit`` / async ``close``).
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

__all__ = [
    "InMemorySink",
    "JSONLPersistenceSink",
    "LoggingSink",
    "reconstruct_session",
    "replay_from_jsonl",
]
