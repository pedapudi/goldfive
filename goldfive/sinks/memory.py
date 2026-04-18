"""In-memory EventSink — collects every emitted event in a list.

Primarily a test fixture: the list is the canonical ground-truth of what
a run emitted, and tests assert against it directly. Cheap enough to use
in production callers that want a recent tail of events, but the whole
stream is kept so long-running runs will grow unbounded.
"""

from __future__ import annotations

from typing import Any


class InMemorySink:
    """EventSink that appends every event to a Python list.

    The ``events`` attribute is a live list — emits mutate it in place and
    readers see updates immediately. There is no thread/task safety beyond
    what ``list.append`` already provides (which is atomic under CPython's
    GIL); concurrent emits from multiple tasks are fine.
    """

    def __init__(self) -> None:
        self._events: list[Any] = []

    @property
    def events(self) -> list[Any]:
        """The collected events, in emit order. Mutating the list is the
        caller's prerogative — the sink does not defend against it."""
        return self._events

    async def emit(self, event: Any) -> None:
        """Append ``event`` to the internal list. Never raises."""
        self._events.append(event)

    async def close(self) -> None:
        """No-op. The list stays populated after close so tests can
        inspect it post-run."""
        return None
