"""Logging-backed EventSink — renders each event as a JSON log line.

One ``Logger.log`` call per event, JSON-serialised via proto's
``MessageToJson`` with proto field names preserved (so log lines match
the JSONL persistence format). Callers supply their own logger so
formatting, handlers, and filtering stay under their control; when no
logger is provided the sink uses its own module logger.
"""

from __future__ import annotations

import logging
from typing import Any

from google.protobuf.json_format import MessageToJson

_default_logger = logging.getLogger("goldfive.sinks.logging_sink")


class LoggingSink:
    """EventSink that logs each event as a one-line JSON string.

    Parameters
    ----------
    logger:
        Optional :class:`logging.Logger`. Defaults to this module's
        logger so callers can configure it by name.
    level:
        Level passed to ``logger.log``. Defaults to ``logging.INFO``.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        self._logger = logger if logger is not None else _default_logger
        self._level = level

    async def emit(self, event: Any) -> None:
        """Log a single JSON line for ``event``.

        Uses ``MessageToJson(..., preserving_proto_field_name=True)`` and
        strips newlines so each event produces exactly one log record.
        Serialisation failures fall back to ``repr`` so a malformed event
        never silently drops.
        """
        try:
            payload = MessageToJson(
                event,
                preserving_proto_field_name=True,
                indent=None,
            )
        except Exception:  # pragma: no cover - defensive
            payload = repr(event)
        self._logger.log(self._level, payload)

    async def close(self) -> None:
        """No-op. Loggers manage their own handler lifecycle."""
        return None
