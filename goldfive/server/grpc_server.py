"""Reference gRPC server for the ``GoldfiveIngress`` service.

:class:`GoldfiveIngressServer` is the server-side twin of
:class:`goldfive.sinks.GRPCSink`. It implements the single RPC defined
in ``proto/goldfive/v1/service.proto``:

    rpc StreamEvents(stream Event) returns (StreamEventsResponse);

For each incoming stream the servicer fans every received ``Event``
message out to a local list of :class:`~goldfive.protocols.EventSink`
instances — typically a persistence sink (JSONL / SQLite) plus an
``InMemorySink`` for tests. Sink exceptions are caught and logged so a
single misbehaving sink cannot abort the stream.

The server is async-native (``grpc.aio``) and exposes two ways to run:

* :meth:`GoldfiveIngressServer.start` / :meth:`GoldfiveIngressServer.stop`
  — fine-grained lifecycle, suitable for tests and composable services.
* :meth:`GoldfiveIngressServer.run` — convenience wrapper that starts
  the server and blocks until it is terminated.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import grpc.aio

from goldfive.pb.goldfive.v1 import service_pb2, service_pb2_grpc

if TYPE_CHECKING:
    from goldfive.protocols import EventSink

log = logging.getLogger("goldfive.server.grpc")


class GoldfiveIngressServicer(service_pb2_grpc.GoldfiveIngressServicer):
    """Servicer that fans received ``Event`` messages to local sinks.

    The servicer can be reused across many ``StreamEvents`` calls — each
    call gets its own request iterator and its own response envelope, but
    the underlying sink list is shared. ``received_total`` is a
    cumulative counter across all streams the servicer has handled.
    """

    def __init__(self, sinks: list[EventSink]) -> None:
        self._sinks: list[EventSink] = list(sinks)
        self._received_total: int = 0

    @property
    def sinks(self) -> list[EventSink]:
        return self._sinks

    @property
    def received_total(self) -> int:
        """Total number of events processed across all streams."""
        return self._received_total

    async def StreamEvents(  # noqa: N802 — gRPC spec case
        self,
        request_iterator: Any,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.StreamEventsResponse:
        """Consume the client stream, fan-out to sinks, return a summary.

        The method reads proto ``Event`` messages off the client stream,
        awaits ``sink.emit(event)`` on every configured sink (concurrently,
        via :func:`asyncio.gather`), and returns a
        ``StreamEventsResponse`` counting the successfully-processed
        events. Sink exceptions are logged and swallowed so one bad sink
        cannot stop the stream.
        """
        count = 0
        error_text = ""
        try:
            async for event in request_iterator:
                await _fanout(self._sinks, event)
                count += 1
            self._received_total += count
        except grpc.aio.AioRpcError as exc:  # pragma: no cover — rare in-process
            error_text = f"rpc error: {exc.code().name}"
            log.warning("StreamEvents rpc error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            log.exception("StreamEvents failed")
        return service_pb2.StreamEventsResponse(received=count, error=error_text)


async def _fanout(sinks: list[EventSink], event: Any) -> None:
    """Await every sink's ``emit`` concurrently; log-and-swallow failures."""
    if not sinks:
        return
    results = await asyncio.gather(
        *(sink.emit(event) for sink in sinks),
        return_exceptions=True,
    )
    for sink, result in zip(sinks, results, strict=True):
        if isinstance(result, BaseException):
            log.warning(
                "sink %s.emit raised: %s",
                type(sink).__name__,
                result,
            )


class GoldfiveIngressServer:
    """Server wrapper around a :class:`GoldfiveIngressServicer`.

    Parameters
    ----------
    sinks:
        Local :class:`~goldfive.protocols.EventSink` instances. Every
        event received over the wire is fanned out to each of them in
        turn. Pass an empty list to run a validation-only endpoint.
    credentials:
        Optional ``grpc.ServerCredentials``. If ``None`` the server binds
        insecurely (plain TCP), which is fine for localhost / trusted-
        network deployments and for tests.
    server_options:
        Optional sequence of ``(key, value)`` tuples passed straight
        through to ``grpc.aio.server`` for callers that need to tune
        channel options (keepalive, message size caps, etc.).
    """

    def __init__(
        self,
        sinks: list[EventSink],
        *,
        credentials: Any = None,
        server_options: list[tuple[str, Any]] | None = None,
    ) -> None:
        self._sinks: list[EventSink] = list(sinks)
        self._credentials = credentials
        self._server_options = list(server_options) if server_options else None
        self._server: grpc.aio.Server | None = None
        self._servicer = GoldfiveIngressServicer(self._sinks)
        self._bound_port: int | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def servicer(self) -> GoldfiveIngressServicer:
        return self._servicer

    @property
    def sinks(self) -> list[EventSink]:
        return self._sinks

    @property
    def bound_port(self) -> int | None:
        """The port the server is bound to, or ``None`` before ``start``."""
        return self._bound_port

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, host: str = "127.0.0.1", port: int = 50051) -> int:
        """Bind the server and begin serving. Returns the bound port.

        Pass ``port=0`` to request an ephemeral port from the OS; the
        returned integer tells callers (in particular tests) which port
        was actually bound.
        """
        if self._server is not None:
            raise RuntimeError("GoldfiveIngressServer already started")
        server = grpc.aio.server(options=self._server_options)
        service_pb2_grpc.add_GoldfiveIngressServicer_to_server(self._servicer, server)
        address = f"{host}:{port}"
        if self._credentials is None:
            actual_port = server.add_insecure_port(address)
        else:
            actual_port = server.add_secure_port(address, self._credentials)
        await server.start()
        self._server = server
        self._bound_port = actual_port
        log.info("GoldfiveIngressServer listening on %s:%d", host, actual_port)
        return actual_port

    async def stop(self, grace: float | None = 1.0) -> None:
        """Stop the server and close local sinks. Safe to call repeatedly."""
        if self._server is None:
            return
        try:
            await self._server.stop(grace)
        finally:
            self._server = None
            self._bound_port = None
            for sink in self._sinks:
                try:
                    await sink.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("sink %s.close raised: %s", type(sink).__name__, exc)

    async def wait_for_termination(self) -> None:
        """Block until the server is terminated (e.g. by an external signal)."""
        if self._server is not None:
            await self._server.wait_for_termination()

    async def run(self, host: str = "127.0.0.1", port: int = 50051) -> None:
        """Start the server and block until it terminates.

        Convenience wrapper: equivalent to calling :meth:`start` then
        :meth:`wait_for_termination`. Callers that need finer control
        (e.g. to stop the server from another coroutine) should use the
        lower-level primitives directly.
        """
        await self.start(host, port)
        try:
            await self.wait_for_termination()
        finally:
            await self.stop(grace=None)


__all__ = [
    "GoldfiveIngressServer",
    "GoldfiveIngressServicer",
]
