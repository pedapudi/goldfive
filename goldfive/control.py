"""Bidirectional async control channel between a Runner and external controllers.

The :class:`ControlChannel` is the goldfive-side primitive that lets external
processes (the harmonograf UI, a CLI, tests) steer a running ``Runner``: pause,
resume, cancel, steer, rewind. Messages are queued asynchronously; runners poll
:meth:`ControlChannel.receive` and publish acknowledgements via
:meth:`ControlChannel.ack`. External bridges drain those acks via
:meth:`ControlChannel.acks`.

This module is intentionally dependency-light — only ``asyncio`` — so it can be
imported by adapters and bridges without dragging in protobuf or grpc.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ControlKind(StrEnum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    STEER = "STEER"            # payload: {"note": "...", "suggested_action": "..."}
    REWIND_TO = "REWIND_TO"    # payload: {"task_id": "..."}
    APPROVE = "APPROVE"        # payload: {"target_id": "...", "detail": "..."}
    REJECT = "REJECT"          # payload: {"target_id": "...", "detail": "..."}


class AckResult(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ControlMessage:
    kind: ControlKind
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: dict[str, Any] = field(default_factory=dict)
    issued_at_ms: int = 0


@dataclass
class ControlAck:
    control_id: str
    result: AckResult
    detail: str = ""
    acked_at_ms: int = 0


class ControlChannel:
    """Bidirectional async control channel between a Runner and external
    controllers (harmonograf UI, CLI, tests).

    External -> runner: ControlMessages pushed via :meth:`send`, consumed
    by the runner via :meth:`receive`.

    Runner -> external: ControlAcks emitted via :meth:`ack`, consumed by
    the external bridge via :meth:`acks` async iterator.
    """

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[ControlMessage] = asyncio.Queue()
        self._outbox: asyncio.Queue[ControlAck] = asyncio.Queue()
        self._closed = False

    async def send(self, msg: ControlMessage) -> None:
        """External caller pushes a control message to the runner."""
        await self._inbox.put(msg)

    async def receive(self, timeout: float | None = None) -> ControlMessage | None:
        """Runner polls for the next control message.

        Returns ``None`` on timeout or when the channel is closed.
        """
        if self._closed:
            return None
        try:
            if timeout is None:
                return await self._inbox.get()
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except TimeoutError:
            return None

    async def ack(self, ack: ControlAck) -> None:
        """Runner publishes an ack for a received message."""
        await self._outbox.put(ack)

    async def acks(self) -> AsyncIterator[ControlAck]:
        """External bridge iterates ack output.

        Yields acks until the channel is closed. Once :meth:`close` is
        called, a sentinel is queued so a pending consumer wakes up and
        the loop terminates without leaking the task.
        """
        while not self._closed:
            ack = await self._outbox.get()
            if ack is _CLOSE_SENTINEL:
                break
            yield ack

    def close(self) -> None:
        """Mark the channel closed.

        After ``close()``: :meth:`receive` returns ``None`` immediately,
        and any consumer blocked on :meth:`acks` is woken up via a
        sentinel so it exits the iterator cleanly.
        """
        if self._closed:
            return
        self._closed = True
        # Wake any pending acks() consumer with a sentinel.
        self._outbox.put_nowait(_CLOSE_SENTINEL)


# A private sentinel pushed onto _outbox by close() so that a coroutine
# blocked on acks() wakes up and exits the iterator. Comparing by identity
# avoids accidental matches against real ControlAck values.
_CLOSE_SENTINEL: Any = object()
