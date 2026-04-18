"""Built-in EventSinks for goldfive.

Three implementations ship in-box:

* :class:`InMemorySink` — collects events in a list; used by tests and
  ephemeral callers that want to inspect the stream after the run.
* :class:`LoggingSink` — renders each event as a single JSON line onto a
  :class:`logging.Logger`; drop-in for stdout/journald style observation.
* :class:`JSONLPersistenceSink` — appends events to a newline-delimited
  JSON file suitable for crash recovery. Paired with
  :func:`replay_from_jsonl` and :func:`reconstruct_session` for resume.

All sinks conform to the ``EventSink`` protocol pinned in
``INTERFACE_SPEC.md`` (async ``emit`` / async ``close``).
"""

from __future__ import annotations

from goldfive.sinks.logging_sink import LoggingSink
from goldfive.sinks.memory import InMemorySink
from goldfive.sinks.persistence import (
    JSONLPersistenceSink,
    reconstruct_session,
    replay_from_jsonl,
)

__all__ = [
    "InMemorySink",
    "JSONLPersistenceSink",
    "LoggingSink",
    "reconstruct_session",
    "replay_from_jsonl",
]
