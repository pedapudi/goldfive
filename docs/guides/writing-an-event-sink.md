# Writing an event sink

`EventSink` is how goldfive delivers observability to the outside
world. Every state change — run started, task completed, drift
detected, plan revised — produces a proto-encoded `Event` that is
fanned out to every configured sink.

This guide covers writing a custom sink. Read
[EVENT-MODEL.md](../design/EVENT-MODEL.md) first if you haven't —
it's the contract this guide assumes.

## When to write a custom sink

Write one when you want to:

- Forward events to your own observability system (OpenTelemetry,
  Datadog, a custom collector).
- Push events to a live dashboard over WebSockets.
- Fan events out to a message bus (Kafka, NATS, Redis Streams).
- Drive a UI like harmonograf (see the
  [harmonograf-integration guide](harmonograf-integration.md)).
- Record runs for later replay or analysis.

You don't need to write one for:

- **Recording events for tests** — use `InMemorySink`.
- **Logging to stdout** — use `LoggingSink`.
- **Disk persistence for crash recovery** — use
  `JSONLPersistenceSink`.

## The protocol

```python
@runtime_checkable
class EventSink(Protocol):
    async def emit(self, event_pb: Any) -> None: ...
    async def close(self) -> None: ...
```

Two methods. `emit(event)` is called once per event, in per-run
sequence order. `close()` is called once when the run ends. That's
the entire contract.

Since #55 the Runner, executor, and steerer all emit proto `Event`
envelopes — a sink only needs to handle that one shape. Dispatch on
`event.WhichOneof("payload")` to branch on kind.
`goldfive.events.make_event` still returns a dict envelope for
callers who want the simpler shape without proto, but no shipped
goldfive component feeds dicts to sinks today.

## A minimal stdout sink

```python
from __future__ import annotations

from typing import Any


class StdoutSink:
    async def emit(self, event: Any) -> None:
        seq = int(getattr(event, "sequence", 0))
        kind = event.WhichOneof("payload") or "?"
        print(f"[{seq:04d}] {kind} (run={event.run_id})")

    async def close(self) -> None:
        pass
```

Plug it in:

```python
from goldfive import Runner
runner = Runner(agent=..., planner=..., executor=..., sinks=[StdoutSink()])
```

## A richer example — forwarding to an HTTP backend

Here's a sink that POSTs each event as JSON to an observability
backend, with retries and a bounded in-process queue so network
slowness doesn't stall the run.

```python
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from google.protobuf.json_format import MessageToJson


class HttpBackendSink:
    """Forwards events to an HTTP backend. Drops on overflow."""

    def __init__(
        self,
        endpoint: str,
        *,
        max_queue: int = 1024,
        timeout_s: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_queue)
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._worker = asyncio.create_task(self._drain())
        self._closed = False

    async def emit(self, event_pb: Any) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(event_pb)
        except asyncio.QueueFull:
            # drop on overflow — prefer ongoing execution over full fidelity
            pass

    async def close(self) -> None:
        self._closed = True
        await self._queue.put(None)  # poison pill
        await self._worker
        await self._client.aclose()

    async def _drain(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            payload = json.loads(
                MessageToJson(item, preserving_proto_field_name=True)
            )
            try:
                await self._client.post(self._endpoint, json=payload)
            except httpx.HTTPError:
                # real-world: structured logging + retry with backoff
                pass
```

Key design points:

- **The worker pattern.** `emit` enqueues; a background worker drains.
  Under the hood, `emit` now returns almost immediately even if the
  backend is slow.
- **Bounded queue with drop-on-overflow.** Prevents unbounded memory
  growth if the backend is persistently slow. A production sink
  should log the drop and expose a metric.
- **Poison pill on close.** Sends a sentinel to unblock the worker so
  it can exit cleanly.
- **Serialization via `MessageToJson`.** The canonical way to get a
  wire-friendly JSON representation of a proto message.

## Design considerations

### Back-pressure

goldfive awaits every `emit` call. A sink that blocks blocks the
run. Always use one of:

- A bounded internal queue with drop-on-overflow (the HTTP sink above).
- A bounded internal queue with a short timeout (log and proceed on
  timeout).
- Synchronous, fast writes (the `LoggingSink` pattern — writes go to
  the standard library's thread-safe `logging` layer).

**Never** use an unbounded queue. Unbounded queues turn a slow
backend into an OOM.

### Error handling

`emit` can raise. goldfive catches and logs; your run keeps going.
But emit-time errors are costly — they indicate the sink is broken.
Design for graceful degradation:

- Log once, suppress subsequent errors for a cooldown window.
- Expose a health signal so ops can notice.
- If the sink is critical (e.g. it's the only durable store), crash
  the process rather than silently dropping.

### Ordering

Events arrive in per-run sequence order. If your backend cares about
order (most do), forward them in the same order. The HTTP sink above
uses a single worker coroutine to preserve order; a concurrent worker
pool would lose it.

### Durability

`emit` returns before the event has hit the wire in the queued
design. If the process crashes after `emit` returns but before the
worker drains, you lose events. If your sink needs strict durability
(e.g. for audit logs), use a synchronous write (like
`JSONLPersistenceSink`) or a queue backed by a durable store.

### Multiple sinks at once

goldfive passes the **same** event object to every sink. Treat it as
immutable. If your sink needs to mutate (e.g. strip fields, transform
types), clone first:

```python
event_copy = type(event_pb)()
event_copy.CopyFrom(event_pb)
# mutate event_copy freely
```

## Useful proto helpers

```python
from google.protobuf.json_format import MessageToDict, MessageToJson, ParseDict

# event -> Python dict
as_dict = MessageToDict(event_pb, preserving_proto_field_name=True)

# event -> JSON string
as_json = MessageToJson(event_pb, preserving_proto_field_name=True)

# dict -> event (for replay)
from goldfive.pb.goldfive.v1.events_pb2 import Event
ev = Event()
ParseDict(some_dict, ev)
```

These are the same helpers `JSONLPersistenceSink` uses for its
round-trip. If you build a sink that wraps a transport, you'll
probably re-use them.

## Pattern: filtering sinks

A common pattern is a sink that only forwards certain kinds. Compose
rather than configure:

```python
from typing import Iterable


class FilteringSink:
    def __init__(self, inner, *, kinds: Iterable[str]) -> None:
        self._inner = inner
        self._kinds = set(kinds)

    async def emit(self, event) -> None:
        kind = event.WhichOneof("payload") or ""
        if kind in self._kinds:
            await self._inner.emit(event)

    async def close(self) -> None:
        await self._inner.close()
```

Used:

```python
sink = FilteringSink(
    HttpBackendSink("https://obs.example.com/events"),
    kinds={"drift_detected", "plan_revised", "run_aborted"},
)
```

## Pattern: replay from `JSONLPersistenceSink`

The persistence sink stores every event, so you can reconstruct any
run with the module-level ``replay_from_jsonl`` helper:

```python
from goldfive.sinks import replay_from_jsonl

events = replay_from_jsonl("./runs/abc123.jsonl")
for ev in events:
    kind = ev.WhichOneof("payload")
    print(ev.sequence, kind)
```

You can feed those events into a custom sink after the fact to
rebuild external state (e.g. backfill a newly-added observability
system from historical JSONL files).

## Testing a custom sink

Two patterns.

**In isolation** — instantiate your sink, call `emit` with hand-built
events, assert on side effects:

```python
import pytest
from goldfive.pb.goldfive.v1.events_pb2 import Event


@pytest.mark.asyncio
async def test_http_sink_posts_events(httpx_mock):
    httpx_mock.add_response(status_code=200)
    sink = HttpBackendSink("http://mock.local/events")

    ev = Event(run_id="r1", sequence=0)
    ev.run_started.run_id = "r1"
    ev.run_started.goal_summary = "hello"

    await sink.emit(ev)
    await sink.close()

    assert httpx_mock.get_requests()  # at least one POST
```

**End-to-end** — wire it into a `Runner` run with a `CallableAdapter`
and a known plan, then assert your sink received the expected events
in order.

## Upstream contribution

If you write a sink that generalizes (popular framework, commonly-used
observability product), goldfive is receptive to upstreaming. Open an
issue with a spec and a test plan. Candidate "blessed" sinks:

- OpenTelemetry (OTLP) forwarder.
- Grafana / Tempo / Loki forwarders.
- harmonograf integration (forthcoming as a separate package; see
  [harmonograf-integration.md](harmonograf-integration.md)).
