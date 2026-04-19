# Choosing a sink

Your agent's event stream is a multi-consumer firehose. Pick the sinks
that match how you'll consume it.

Six implementations ship: `InMemorySink` from the stock install, four
more (`LoggingSink`, `JSONLPersistenceSink`, `SQLitePersistenceSink`,
`GRPCSink`) under the `proto` extra (`uv add 'goldfive[proto]'`), and
`HarmonografSink` from the `harmonograf_client` package.

## Quick decision matrix

| Sink | In-process? | Persistent? | Cross-process? | Proto+dict? | Install |
| --- | --- | --- | --- | --- | --- |
| `InMemorySink` | yes | no | no | both | stdlib |
| `LoggingSink` | yes | via logging config | no | proto cleanly, else `repr` | `proto` extra |
| `JSONLPersistenceSink` | yes | yes (file) | no | both | `proto` extra |
| `SQLitePersistenceSink` | yes | yes (DB) | via DB reads | both | `proto` extra |
| `GRPCSink` | no (streams out) | no (the server persists) | yes | proto only | `proto` extra + `grpcio` |
| `HarmonografSink` | no (streams out) | no (harmonograf persists) | yes | proto only | `harmonograf_client` |

"Proto+dict" means the sink accepts both the Runner's dict lifecycle
envelopes and the executor's proto `Event` messages. `GRPCSink` and
`HarmonografSink` drop non-proto events with a debug log — the wire
formats are strictly proto.

## Comparison

| Sink | What it does | Best for | Tradeoffs | Install |
| --- | --- | --- | --- | --- |
| `InMemorySink` | appends events to a list | tests, post-run inspection | unbounded memory on long runs | stdlib |
| `LoggingSink` | logs each event as one JSON line | stdout / journald observation | dict events render as `repr` | `proto` extra |
| `JSONLPersistenceSink` | appends JSONL to a file | crash recovery, stable diffs | single-file; no cross-run query | `proto` extra |
| `SQLitePersistenceSink` | inserts rows keyed by `(run_id, sequence)` | dashboards, cross-run queries | one writer at a time per file | `proto` extra |
| `GRPCSink` | streams proto over client-streaming RPC | out-of-process observers | proto-only; at-least-once on reconnect | `proto` extra + `grpcio` |
| `HarmonografSink` | pushes proto into a harmonograf `Client` | live Gantt + drawer UI | proto-only; needs a harmonograf server | `harmonograf_client` |

## Per-sink sections

### `InMemorySink`

Appends every event to a Python list. Zero dependencies, zero IO.

Use when:

- Writing tests that assert on the event stream.
- Running a short-lived tool that inspects events after the run.
- Pairing with another sink as a debug tap.

Don't use for: long-lived or high-volume runs — the list grows without bound.

```python
from goldfive import Runner, InMemorySink

sink = InMemorySink()
runner = Runner(agent=..., planner=..., executor=..., sinks=[sink])
await runner.run("do the thing")
for ev in sink.events:
    print(ev)
```

### `LoggingSink`

One `Logger.log` call per event, JSON-serialised via proto's
`MessageToJson` with field names preserved. You control logger and level.

Use when:

- You want events on stdout / journald without managing a file.
- You already centralise logs via `logging` handlers.
- You want a low-overhead tap for human-readable debugging.

Don't use for: durable persistence — rotation and retention are your
logging stack's concern, and dict lifecycle envelopes serialise as
`repr` not JSON.

```python
import logging
from goldfive import Runner
from goldfive.sinks import LoggingSink

logging.basicConfig(level=logging.INFO)
sink = LoggingSink(logger=logging.getLogger("agent.events"))
runner = Runner(agent=..., planner=..., executor=..., sinks=[sink])
```

Requires the `proto` extra because `MessageToJson` is imported at module load.

### `JSONLPersistenceSink`

Appends each event as one JSON line. Proto events go through
`MessageToJson(sort_keys=True)` for byte-stable output; dict events go
through `json.dumps(sort_keys=True)`. Pairs with `replay_from_jsonl`
and `reconstruct_session` for crash recovery.

Use when:

- You want crash-safe on-disk persistence for a single run.
- You want stable diffs across reruns (sorted keys).
- You plan to replay the run into `reconstruct_session`.

Don't use for: cross-run queries — one file per run, no indexes.

```python
from goldfive import Runner
from goldfive.sinks import JSONLPersistenceSink

sink = JSONLPersistenceSink("./runs/current.jsonl")  # mode="append" default
runner = Runner(agent=..., planner=..., executor=..., sinks=[sink])
await runner.run("do the thing")
await runner.close()  # flushes and closes the file handle
```

Pass `mode="write"` to truncate on open. Parent directories are created
on first emit.

### `SQLitePersistenceSink`

Inserts each event as a row in a SQLite table keyed by
`(run_id, sequence)`. Connection is opened lazily on first emit, runs
in WAL + autocommit. Pairs with `replay_from_sqlite` and `list_runs`.

Use when:

- You want a single shared store across many runs.
- A dashboard or analytics job needs to query events via SQL.
- You want cheap per-run replay via indexed primary key.

Don't use for: high-frequency writes across processes — SQLite
serialises writers per file. Use the gRPC transport with a single
server-side writer.

```python
from goldfive import Runner
from goldfive.sinks import SQLitePersistenceSink

sink = SQLitePersistenceSink("./events.db")  # or ":memory:" for tests
runner = Runner(agent=..., planner=..., executor=..., sinks=[sink])
await runner.run("do the thing")
await runner.close()
```

Override the table via `SQLitePersistenceSink(path, table="my_events")`
— names must match `[A-Za-z_][A-Za-z0-9_]*`.

### `GRPCSink`

Forwards proto `Event` messages over a client-streaming RPC to a
`GoldfiveIngress` server. `emit` enqueues on an `asyncio.Queue` and
returns; a background drain task pumps the stream. Reconnects on
transient RPC errors by default.

Use when:

- A separate process / host aggregates events from many runners.
- You have a harmonograf-style dashboard on the other side.
- You want at-least-once delivery without writing a custom transport.

Don't use for: durable local persistence (pair with
`JSONLPersistenceSink` in case the server is down at run start), or
dict lifecycle envelopes (silently dropped).

```python
from goldfive import Runner
from goldfive.sinks import GRPCSink

sink = GRPCSink("observer.internal:50051")  # insecure channel
# For TLS: GRPCSink(endpoint, credentials=grpc.ssl_channel_credentials(...))
runner = Runner(agent=..., planner=..., executor=..., sinks=[sink])
await runner.run("do the thing")
await runner.close()  # flushes the queue, closes the channel
```

Kwargs: `credentials=None`, `reconnect=True`, `max_queue=0` (unbounded).
See [grpc-transport.md](grpc-transport.md) for wire shape and TLS.

### `HarmonografSink`

Adapter that pushes each proto `Event` onto a harmonograf `Client`'s
span-transport buffer, wrapped server-side as
`TelemetryUp(goldfive_event=...)`. The sink does not own the client
lifecycle — you construct, pass in, and `shutdown()` the client.

Use when:

- You want the goldfive run to drive harmonograf's live timeline.
- You already run a harmonograf server for span observability.
- You want reconnect / backpressure / heartbeat for free (reused from
  the span transport).

Don't use for: standalone deployments without a harmonograf server —
use `GRPCSink` against a `GoldfiveIngressServer` instead.

```python
from goldfive import Runner
from harmonograf_client import Client, HarmonografSink

client = Client(name="research", server_addr="127.0.0.1:7531")
sink = HarmonografSink(client)
runner = Runner(agent=..., planner=..., executor=..., sinks=[sink])
await runner.run("do the thing")
await runner.close()     # closes the sink
client.shutdown()        # flushes and joins harmonograf transport
```

See [harmonograf-integration.md](harmonograf-integration.md) for the
proto alignment story.

## Common combinations

### Crash-safe production: `JSONLPersistenceSink + GRPCSink`

Local JSONL is the source of truth for recovery; gRPC ships live events
to a central observer. If the observer is down the JSONL still lands on
disk, and the gRPC queue drains when the stream reconnects.

```python
from goldfive import Runner
from goldfive.sinks import JSONLPersistenceSink, GRPCSink

runner = Runner(
    agent=..., planner=..., executor=...,
    sinks=[
        JSONLPersistenceSink(f"./runs/{run_id}.jsonl"),
        GRPCSink("observer.internal:50051"),
    ],
)
```

### Live observability: `LoggingSink + HarmonografSink`

Human-readable JSON on stdout for the terminal tail, live Gantt in
harmonograf for the dashboard. Neither is durable — pair with a
persistence sink if you need recovery.

```python
import logging
from goldfive import Runner
from goldfive.sinks import LoggingSink
from harmonograf_client import Client, HarmonografSink

logging.basicConfig(level=logging.INFO)
client = Client(name="agent")
runner = Runner(
    agent=..., planner=..., executor=...,
    sinks=[LoggingSink(), HarmonografSink(client)],
)
```

### Tests: `InMemorySink` alone

No IO, no external server, no cleanup. The list is the canonical
ground truth of what the run emitted.

```python
from goldfive import Runner, InMemorySink

sink = InMemorySink()
runner = Runner(agent=..., planner=..., executor=..., sinks=[sink])
await runner.run("exercise the plan")
kinds = [getattr(ev, "WhichOneof", lambda _: None)("payload") for ev in sink.events]
assert "run_completed" in kinds
```

### Shared analytics DB: `SQLitePersistenceSink + HarmonografSink`

SQLite is the queryable history across runs; harmonograf is the live
surface. Analytics jobs read SQLite; operators watch harmonograf.

```python
from goldfive import Runner
from goldfive.sinks import SQLitePersistenceSink
from harmonograf_client import Client, HarmonografSink

client = Client(name="agent")
runner = Runner(
    agent=..., planner=..., executor=...,
    sinks=[
        SQLitePersistenceSink("./analytics.db"),
        HarmonografSink(client),
    ],
)
```

## FAQ

**Can I use multiple sinks?** Yes. `Runner(..., sinks=[a, b, c])` fans
each event out to every sink in order; per-emit ordering is preserved.

**What happens if a sink fails mid-run?** The Runner catches and logs
the exception; the remaining sinks still see the event; the run keeps
going. Design your sink for graceful degradation — see
[writing-an-event-sink.md](writing-an-event-sink.md).

**Which sinks accept dict events?** All local sinks accept both shapes.
The Runner emits dict envelopes for lifecycle markers; executors emit
proto `Event` messages for per-task state. `GRPCSink` and
`HarmonografSink` are proto-only and drop non-proto events with a
debug log.

**I want a new sink — where do I start?**
[writing-an-event-sink.md](writing-an-event-sink.md). The protocol is
two async methods (`emit`, `close`); the guide walks through a minimal
stdout sink, a worker-pattern HTTP sink, and testing patterns.

## Related docs

- [getting-started.md](getting-started.md) — Runner wiring if the
  snippets above assume too much.
- [grpc-transport.md](grpc-transport.md) — `GRPCSink` wire shape,
  `GoldfiveIngressServer`, TLS, reconnect semantics.
- [persistence-and-recovery.md](persistence-and-recovery.md) —
  `JSONLPersistenceSink`, `replay_from_jsonl`, `reconstruct_session`.
- [harmonograf-integration.md](harmonograf-integration.md) — why
  goldfive and harmonograf share a proto.
- [writing-an-event-sink.md](writing-an-event-sink.md) — `EventSink`
  protocol, backpressure patterns, custom-sink recipes.
