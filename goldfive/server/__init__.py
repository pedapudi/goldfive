"""Server-side transports for goldfive.

The ``goldfive.server`` subpackage hosts reference implementations of
network-facing event ingress services. Clients (for example the
``GRPCSink`` in ``goldfive.sinks.grpc_sink``) connect to one of these
servers and stream proto ``Event`` messages; the server fans each event
out to a locally-configured list of :class:`~goldfive.protocols.EventSink`
instances.

Currently one transport ships in-box:

* :class:`GoldfiveIngressServer` — a gRPC server implementing the
  ``goldfive.v1.GoldfiveIngress`` service (client-streaming
  ``StreamEvents``).
"""

from __future__ import annotations

try:
    from goldfive.server.grpc_server import (
        GoldfiveIngressServer,
        GoldfiveIngressServicer,
    )
except ImportError:  # pragma: no cover — grpcio / proto extra not installed
    GoldfiveIngressServer = None  # type: ignore[assignment]
    GoldfiveIngressServicer = None  # type: ignore[assignment]

__all__ = [
    "GoldfiveIngressServer",
    "GoldfiveIngressServicer",
]
