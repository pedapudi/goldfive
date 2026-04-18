"""Round-trip tests for GRPCSink + GoldfiveIngressServer.

Exercises the gRPC transport end-to-end: spin up a
``GoldfiveIngressServer`` bound to an ephemeral localhost port, connect
a ``GRPCSink``, emit a batch of proto events, close the sink (which
half-closes the stream), stop the server, and assert the server-side
``InMemorySink`` saw every event in order.
"""

from __future__ import annotations

import asyncio

import pytest

from goldfive.pb.goldfive.v1 import events_pb2
from goldfive.server import GoldfiveIngressServer
from goldfive.sinks import GRPCSink, InMemorySink

if GRPCSink is None or GoldfiveIngressServer is None:  # pragma: no cover
    pytest.skip("gRPC transport requires the proto extra", allow_module_level=True)


def _make_event(seq: int, run_id: str = "run-grpc") -> events_pb2.Event:
    evt = events_pb2.Event(event_id=f"evt-{seq}", run_id=run_id, sequence=seq)
    evt.task_started.task_id = f"task-{seq}"
    evt.task_started.detail = f"hello {seq}"
    return evt


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


async def test_grpc_sink_round_trip_to_server() -> None:
    server_sink = InMemorySink()
    server = GoldfiveIngressServer(sinks=[server_sink])
    port = await server.start(host="127.0.0.1", port=0)
    assert port > 0
    assert server.bound_port == port

    client = GRPCSink(f"127.0.0.1:{port}", reconnect=False)
    events = [_make_event(i) for i in range(5)]
    for e in events:
        await client.emit(e)
    await client.close()

    # Allow the server to drain — close() waits for the stream response,
    # but the servicer awaits sink.emit inside the stream, so once the
    # client half-closes the server has already populated server_sink.
    assert [e.event_id for e in server_sink.events] == [e.event_id for e in events]
    assert [e.sequence for e in server_sink.events] == list(range(5))
    assert server.servicer.received_total == 5

    # Server-side response was delivered back to the client.
    assert client.last_response is not None
    assert client.last_response.received == 5
    assert client.last_response.error == ""

    await server.stop(grace=None)


async def test_grpc_sink_fans_out_to_multiple_local_sinks() -> None:
    sink_a = InMemorySink()
    sink_b = InMemorySink()
    server = GoldfiveIngressServer(sinks=[sink_a, sink_b])
    port = await server.start(host="127.0.0.1", port=0)

    client = GRPCSink(f"127.0.0.1:{port}", reconnect=False)
    await client.emit(_make_event(0))
    await client.emit(_make_event(1))
    await client.close()

    assert len(sink_a.events) == 2
    assert len(sink_b.events) == 2
    assert sink_a.events[0].event_id == "evt-0"
    assert sink_b.events[1].event_id == "evt-1"

    await server.stop(grace=None)


async def test_grpc_sink_close_without_emit_is_safe() -> None:
    """A sink that never saw an emit should close cleanly without opening a
    channel — no background task, no connection attempt."""
    client = GRPCSink("127.0.0.1:1", reconnect=False)
    await client.close()
    # last_response stays None — drain task never ran.
    assert client.last_response is None


async def test_grpc_sink_drops_non_proto_events() -> None:
    """Dict envelopes (from make_event) are silently dropped by GRPCSink —
    the wire format is strictly proto."""
    server_sink = InMemorySink()
    server = GoldfiveIngressServer(sinks=[server_sink])
    port = await server.start(host="127.0.0.1", port=0)

    client = GRPCSink(f"127.0.0.1:{port}", reconnect=False)
    # Mix of dict (dropped) and proto (forwarded) events.
    await client.emit({"kind": "RunStarted", "payload": {}})
    await client.emit(_make_event(0))
    await client.emit({"kind": "RunCompleted"})
    await client.emit(_make_event(1))
    await client.close()

    assert [e.sequence for e in server_sink.events] == [0, 1]

    await server.stop(grace=None)


async def test_grpc_sink_survives_concurrent_emits() -> None:
    server_sink = InMemorySink()
    server = GoldfiveIngressServer(sinks=[server_sink])
    port = await server.start(host="127.0.0.1", port=0)

    client = GRPCSink(f"127.0.0.1:{port}", reconnect=False)
    n = 50
    events = [_make_event(i) for i in range(n)]
    await asyncio.gather(*(client.emit(e) for e in events))
    await client.close()

    # Order on the wire follows queue order; queue respects insertion
    # order across concurrent puts for a single event loop.
    assert len(server_sink.events) == n
    assert sorted(e.event_id for e in server_sink.events) == sorted(
        e.event_id for e in events
    )

    await server.stop(grace=None)


async def test_grpc_sink_endpoint_property() -> None:
    client = GRPCSink("example.test:1234", reconnect=False)
    assert client.endpoint == "example.test:1234"
    await client.close()


# ---------------------------------------------------------------------------
# servicer fanout error isolation
# ---------------------------------------------------------------------------


class _RaisingSink:
    async def emit(self, event):  # noqa: ANN001, ANN201
        raise RuntimeError("nope")

    async def close(self) -> None:
        return None


async def test_server_isolates_bad_sink_from_good_sinks() -> None:
    good_sink = InMemorySink()
    server = GoldfiveIngressServer(sinks=[_RaisingSink(), good_sink])
    port = await server.start(host="127.0.0.1", port=0)

    client = GRPCSink(f"127.0.0.1:{port}", reconnect=False)
    await client.emit(_make_event(0))
    await client.emit(_make_event(1))
    await client.close()

    assert [e.sequence for e in good_sink.events] == [0, 1]
    # Bad sink raised, but the stream still reports both events received.
    assert client.last_response is not None
    assert client.last_response.received == 2
    assert client.last_response.error == ""

    await server.stop(grace=None)
