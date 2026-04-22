# gRPC transport

goldfive `EventSink`s are local-process: the `Runner` awaits
`sink.emit(event_pb)` inside the orchestrator's own loop. For
out-of-process observers — a separate persistence daemon, a harmonograf
aggregator, a dashboard service — you need something that crosses a
network boundary. The gRPC transport is that something.

Related:
[writing-an-event-sink.md](writing-an-event-sink.md),
[persistence-and-recovery.md](persistence-and-recovery.md),
[harmonograf-integration.md](harmonograf-integration.md).

## The shape

```
Runner process                                 observer process
──────────────                                 ────────────────
┌─────────────┐                                ┌──────────────┐
│ SequentialExecutor / ParallelDAGExecutor      │
│                                               │
│   emit(sink_list, event_pb)                   │
│     ├─► InMemorySink (optional)               │
│     └─► GRPCSink ────► asyncio.Queue ────┐    │
└────────────────────────────────────────────┼────┘
                                             │
                       ┌─────────────────────▼────────────────┐
                       │ gRPC: StreamEvents(stream Event)     │
                       │       returns StreamEventsResponse   │
                       └─────────────────────┬────────────────┘
                                             │
                          ┌──────────────────▼──────────────────┐
                          │ GoldfiveIngressServer              │
                          │   servicer receives each Event     │
                          │   fans out to local sinks:         │
                          │     ├─► JSONLPersistenceSink       │
                          │     ├─► SQLitePersistenceSink      │
                          │     └─► InMemorySink (tests)       │
                          └────────────────────────────────────┘
```

The service is defined in `proto/goldfive/v1/service.proto`:

```proto
service GoldfiveIngress {
  rpc StreamEvents(stream Event) returns (StreamEventsResponse);
}

message StreamEventsResponse {
  uint64 received = 1;
  string error = 2;
}
```

One RPC, client-streaming. The client sends a sequence of proto `Event`s
and half-closes; the server responds with a summary after it has
consumed the stream.

## Client: `GRPCSink`

```python
from goldfive import Runner, SequentialExecutor, CallableAdapter
from goldfive.sinks import GRPCSink, JSONLPersistenceSink

# Keep a local JSONL log for crash recovery *and* stream to the
# observer. Pair them; do not replace the local log.
grpc_sink = GRPCSink("observer.internal:50051")
local_log = JSONLPersistenceSink("./runs/current.jsonl")

runner = Runner(
    agent=CallableAdapter(my_agent, available_agents=["worker"]),
    planner=my_planner,
    executor=SequentialExecutor(),
    sinks=[local_log, grpc_sink],
)
outcome = await runner.run("do the thing")
await runner.close()  # flushes both sinks; closes the gRPC channel
```

What `GRPCSink` does, mechanically:

1. On the first `emit`, it lazily opens a `grpc.aio.insecure_channel`
   (or a secure channel if you pass `credentials=`) and starts a
   background drain task.
2. Each `emit(event_pb)` enqueues the proto message on an internal
   `asyncio.Queue` and returns immediately — no network I/O on the
   executor's loop.
3. The drain task opens a `StreamEvents` RPC whose request iterator
   reads off the queue until it sees a sentinel.
4. `close()` pushes the sentinel, awaits the drain task (which in
   turn awaits the server's `StreamEventsResponse`), and closes the
   channel.

### Construction

```python
GRPCSink(
    endpoint,                # "host:port", e.g. "observer.local:50051"
    *,
    credentials=None,        # grpc.ChannelCredentials or None
    reconnect=True,          # retry on transient RPC errors
    max_queue=0,             # 0 = unbounded; non-zero applies back-pressure
)
```

For TLS, construct credentials as usual:

```python
import grpc
creds = grpc.ssl_channel_credentials(root_certificates=open("ca.pem", "rb").read())
sink = GRPCSink("observer.internal:443", credentials=creds)
```

### Proto only

`GRPCSink` forwards **only** proto `Event` messages. Since #55 the
Runner, executor, and steerer all emit proto, so the full lifecycle
crosses the wire. If a caller (or a custom component) hands a dict
envelope to the sink, it is silently dropped with a debug log —
pair with `JSONLPersistenceSink` locally if you need that path.

### Session id stamping

Every `Event` on the wire carries `session_id` (tag 5, goldfive#155).
Server-side routing uses it as the multiplex key when a single stream
carries events from multiple Sessions — which matters for consumers
like `HarmonografSink` where the client-side buffer is process-wide
but runs are session-scoped.

Under the adk-web integration the outer adk-web session id is pinned
onto `Session.id` before dispatch (goldfive#161), so the goldfive
Session, ADKAdapter's internal session, and adk-web's URL session id
all match. One session row per run; no reconciliation on the
server side.

### Lazy Hello (harmonograf-specific, harmonograf#85)

The harmonograf client's `Client(name=..., server_addr=...)`
constructor used to issue a Hello RPC synchronously, which meant the
session was minted server-side at *client construction* time — before
the goldfive Session existed. That led to a race where the
server-side session id and the goldfive-side session id disagreed.

Current harmonograf (≥ #85) defers the Hello until the first event
emission. The client observes the first goldfive event's `session_id`
and uses it as the Hello session id. Result: server-side session =
goldfive Session = adk-web session. No reconciliation needed; no
duplicate rows in the UI.

If you're building a server that consumes the goldfive gRPC stream,
mirror the same lazy pattern — defer session creation until the first
Event's `session_id` is known.

### Reconnect semantics

With `reconnect=True` (the default) the drain task retries `StreamEvents`
on `AioRpcError` with exponential backoff capped at 5 s. Events that
were enqueued but not yet sent are delivered on the next attempt; events
that were sent but not acknowledged (gRPC has no per-message ack in
client-streaming mode) may be re-sent after reconnect, so treat the
transport as **at-least-once** for sinks that aggregate across
reconnects. `reconnect=False` makes it **at-most-once**: the first RPC
error ends the drain task and subsequent `emit`s accumulate in the queue
without delivery.

## Server: `GoldfiveIngressServer`

```python
from goldfive.server import GoldfiveIngressServer
from goldfive.sinks import JSONLPersistenceSink, InMemorySink

server = GoldfiveIngressServer(
    sinks=[
        JSONLPersistenceSink("./observer.jsonl"),
        InMemorySink(),  # optional — handy for tests / debug endpoints
    ],
)
await server.run(host="0.0.0.0", port=50051)  # blocks until terminated
```

Or, if you want fine-grained control:

```python
port = await server.start(host="127.0.0.1", port=0)  # ephemeral bind
# ... connect clients, run assertions, etc.
await server.stop(grace=None)
```

The server fans each received `Event` out to every local sink
concurrently (`asyncio.gather` with `return_exceptions=True`). A
misbehaving sink that raises from its `emit` is logged and skipped; the
stream keeps going and the remaining sinks still see the event. The
`StreamEventsResponse` returned to the client reports how many events
the server read off the stream; `error` is non-empty only if the
stream itself faulted (not if an individual sink raised).

### Authentication / authorization

The transport ships insecure by default, suitable for localhost and
trusted-network deployments. For production, wire a
`grpc.ServerCredentials` into the server and a
`grpc.ChannelCredentials` into the client; both accept standard gRPC
TLS material. goldfive itself does not prescribe an authz scheme —
wrap the server in your own interceptor if you need one.

## End-to-end: run the example

```
uv run --extra proto python examples/grpc_ingress.py
```

The example starts a server on an ephemeral port, connects a
`GRPCSink`, emits four proto events, and prints what the server
received.

## Regenerating stubs

The proto file lives at `proto/goldfive/v1/service.proto`. To
regenerate `goldfive/pb/goldfive/v1/service_pb2*.py` (including the
`service_pb2_grpc.py` stubs) after editing it:

```
make proto
```

The `make proto` target now invokes `grpc_tools.protoc` with
`--python_out`, `--pyi_out`, and `--grpc_python_out`, so message types
and gRPC service stubs regenerate together.
