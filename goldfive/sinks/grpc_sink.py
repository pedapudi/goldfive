"""gRPC EventSink — ships proto ``Event`` messages to a GoldfiveIngress server.

:class:`GRPCSink` is the network-facing twin of :class:`InMemorySink` /
:class:`JSONLPersistenceSink`: it implements the ``EventSink`` protocol
(``async emit`` / ``async close``) and forwards each proto ``Event`` over
a client-streaming gRPC RPC to a ``GoldfiveIngress`` server (see
``goldfive.server``).

The sink is async-native:

* ``emit`` enqueues the proto message on an ``asyncio.Queue`` and returns
  immediately, so it does not block the executor's event loop on network
  latency.
* A single background drain task opens a ``StreamEvents`` RPC on first
  emit and reads events off the queue as an async iterator. The server
  half-closes once the client signals completion (via a sentinel) and
  returns a ``StreamEventsResponse`` summary.
* ``close`` pushes a sentinel so the drain task finishes cleanly,
  awaits its completion, and closes the channel.

Only proto ``Event`` messages (anything with a ``DESCRIPTOR`` attribute)
are forwarded. Non-proto events — e.g. the dict envelopes some executors
emit via :func:`goldfive.events.make_event` — are silently dropped with a
debug log, because the wire format is strictly proto. Callers that mix
dict and proto events should pair ``GRPCSink`` with a local sink that
accepts both (``JSONLPersistenceSink``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger("goldfive.sinks.grpc")


class GRPCSink:
    """EventSink that forwards proto ``Event`` messages over a gRPC stream.

    Parameters
    ----------
    endpoint:
        ``host:port`` address of the ``GoldfiveIngress`` server. Passed
        verbatim to ``grpc.aio.insecure_channel`` (or
        ``grpc.aio.secure_channel`` when ``credentials`` is set).
    credentials:
        Optional ``grpc.ChannelCredentials``. If ``None`` the sink uses an
        insecure channel, suitable for localhost / trusted-network
        deployments. For TLS, pass the result of
        ``grpc.ssl_channel_credentials(...)``.
    reconnect:
        When ``True`` (the default) the drain task retries the stream on
        transient RPC errors with a short backoff. When ``False`` the
        first error ends the drain task and subsequent ``emit`` calls
        still enqueue events but they will not be delivered.
    max_queue:
        Maximum number of pending events buffered in memory. ``0`` (the
        default) means unbounded — appropriate for low-volume runs.
        Non-zero values cause ``emit`` to back-pressure when the queue is
        full.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        credentials: Any = None,
        reconnect: bool = True,
        max_queue: int = 0,
    ) -> None:
        self._endpoint = endpoint
        self._credentials = credentials
        self._reconnect = reconnect
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_queue)
        self._channel: Any = None
        self._stub: Any = None
        self._drain_task: asyncio.Task[Any] | None = None
        self._closed = False
        self._sentinel: object = object()
        self._start_lock = asyncio.Lock()
        self._last_response: Any = None

    # ------------------------------------------------------------------
    # EventSink protocol
    # ------------------------------------------------------------------

    async def emit(self, event: Any) -> None:
        """Enqueue ``event`` for delivery. Returns immediately."""
        if self._closed:
            log.debug("GRPCSink: emit after close; dropping event")
            return
        if not hasattr(event, "DESCRIPTOR"):
            log.debug(
                "GRPCSink: dropping non-proto event of type %r",
                type(event).__name__,
            )
            return
        await self._ensure_started()
        await self._queue.put(event)

    async def close(self) -> None:
        """Flush the queue, finish the stream, and close the channel."""
        if self._closed:
            return
        self._closed = True
        if self._drain_task is None:
            # emit was never called; channel was never opened.
            return
        await self._queue.put(self._sentinel)
        try:
            await self._drain_task
        except Exception as exc:  # noqa: BLE001
            log.warning("GRPCSink drain task ended with error: %s", exc)
        if self._channel is not None:
            try:
                await self._channel.close()
            finally:
                self._channel = None
                self._stub = None

    # ------------------------------------------------------------------
    # Read-only helpers (useful for tests + observability)
    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def last_response(self) -> Any:
        """The most recent ``StreamEventsResponse`` seen by the drain task.

        ``None`` until the server has acknowledged the stream (i.e. until
        after ``close`` completes on a healthy connection).
        """
        return self._last_response

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._drain_task is not None:
            return
        async with self._start_lock:
            if self._drain_task is not None:
                return
            import grpc.aio  # local import — keeps module import cheap

            from goldfive.pb.goldfive.v1 import service_pb2_grpc

            if self._credentials is None:
                self._channel = grpc.aio.insecure_channel(self._endpoint)
            else:
                self._channel = grpc.aio.secure_channel(self._endpoint, self._credentials)
            self._stub = service_pb2_grpc.GoldfiveIngressStub(self._channel)
            self._drain_task = asyncio.create_task(self._drain())

    async def _iter_until_sentinel(self) -> AsyncIterator[Any]:
        """Yield queued events until the sentinel is seen, then stop."""
        while True:
            item = await self._queue.get()
            if item is self._sentinel:
                return
            yield item

    async def _drain(self) -> None:
        """Background task that pumps the queue into the gRPC stream."""
        import grpc

        backoff = 0.5
        while True:
            try:
                response = await self._stub.StreamEvents(self._iter_until_sentinel())
                self._last_response = response
                # Sentinel reached; clean end of stream.
                return
            except grpc.aio.AioRpcError as exc:
                if self._closed or not self._reconnect:
                    log.warning("GRPCSink stream ended with error: %s", exc)
                    return
                log.warning("GRPCSink stream failed (%s); reconnecting", exc)
                await asyncio.sleep(backoff)
                # Bounded exponential backoff — cap at 5s so a long
                # outage doesn't let the queue grow forever.
                backoff = min(backoff * 2, 5.0)
            except Exception as exc:  # noqa: BLE001
                log.warning("GRPCSink drain raised unexpectedly: %s", exc)
                return


__all__ = ["GRPCSink"]
