"""grpc_ingress — round-trip demo of the goldfive gRPC transport.

Starts a :class:`goldfive.server.GoldfiveIngressServer` with an
:class:`InMemorySink` on an ephemeral localhost port, connects a
:class:`goldfive.sinks.GRPCSink` to it, emits a handful of proto
``Event`` messages, and prints what the server received.

The goal is to show the moving parts in a single file:

* How to assemble the server with a local fan-out sink list.
* How to wire :class:`GRPCSink` into a ``sinks=[...]`` list that a
  :class:`goldfive.Runner` could consume.
* How the stream lifecycle lines up with ``close()`` on the sink.

Run with::

    uv run --extra proto python examples/grpc_ingress.py
"""

from __future__ import annotations

import asyncio

from goldfive.pb.goldfive.v1 import events_pb2
from goldfive.server import GoldfiveIngressServer
from goldfive.sinks import GRPCSink, InMemorySink


def _task_started(run_id: str, seq: int, task_id: str, detail: str) -> events_pb2.Event:
    evt = events_pb2.Event(
        event_id=f"evt-{seq}",
        run_id=run_id,
        sequence=seq,
    )
    evt.task_started.task_id = task_id
    evt.task_started.detail = detail
    return evt


def _task_completed(run_id: str, seq: int, task_id: str, summary: str) -> events_pb2.Event:
    evt = events_pb2.Event(
        event_id=f"evt-{seq}",
        run_id=run_id,
        sequence=seq,
    )
    evt.task_completed.task_id = task_id
    evt.task_completed.summary = summary
    return evt


async def main() -> None:
    # 1) Start an ingress server with an InMemorySink so we can inspect
    #    what the server received at the end of the demo. In production
    #    the server would be backed by a JSONLPersistenceSink /
    #    SQLitePersistenceSink (or both).
    server_sink = InMemorySink()
    server = GoldfiveIngressServer(sinks=[server_sink])
    port = await server.start(host="127.0.0.1", port=0)
    print(f"Server listening on 127.0.0.1:{port}")

    # 2) The client is a GRPCSink — drop this into a Runner's sinks list
    #    and every proto event that reaches the sink gets streamed to
    #    the server.
    client = GRPCSink(f"127.0.0.1:{port}", reconnect=False)

    # 3) Emit a hand-rolled mini event stream so this file does not need
    #    the executor / adapter machinery just to demonstrate the wire.
    run_id = "grpc-demo"
    events = [
        _task_started(run_id, 0, "t1", "loading inputs"),
        _task_completed(run_id, 1, "t1", "inputs loaded"),
        _task_started(run_id, 2, "t2", "running analysis"),
        _task_completed(run_id, 3, "t2", "analysis complete"),
    ]
    for evt in events:
        await client.emit(evt)

    # 4) close() flushes the queue, half-closes the stream, awaits the
    #    server's StreamEventsResponse, and tears down the channel.
    await client.close()

    print(f"Server received {server.servicer.received_total} events")
    print(f"Server response: received={client.last_response.received}")
    for evt in server_sink.events:
        kind = evt.WhichOneof("payload")
        tid = getattr(getattr(evt, kind), "task_id", "?") if kind else "?"
        print(f"  seq={evt.sequence:>2}  {kind:<16}  task={tid}")

    await server.stop(grace=None)


if __name__ == "__main__":
    asyncio.run(main())
