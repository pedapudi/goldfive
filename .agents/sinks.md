---
name: sinks
description: The EventSink protocol — shipped implementations, how to pick one, how to write your own.
applies-when: ["write a sink", "send events somewhere", "EventSink contract", "persist events"]
---

# Sinks

An `EventSink` receives proto `Event` messages (and a small number of
dict envelopes from the Runner) as a run unfolds. Sinks are the
integration surface for logging, persistence, dashboards, and
downstream services.

## Contract

From `goldfive/protocols.py`:

```python
from typing import Protocol, runtime_checkable
from typing import Any

@runtime_checkable
class EventSink(Protocol):
    async def emit(self, event_pb: Any) -> None: ...
    async def close(self) -> None: ...
```

Two methods. `emit` is called once per event. `close` is called once
from `Runner.close` — buffered sinks flush there.

**`emit` may receive two shapes.** Current `main` emits every event
as a proto `Event` (PR #55). A dict fallback from
`goldfive.events.make_event` is still supported for callers that don't
have the `proto` extra. Sinks that only accept proto should duck-type
on `hasattr(event, "DESCRIPTOR")` and skip dicts cleanly.

## Shipped implementations

| Sink | Module | Extra | Use |
|---|---|---|---|
| `InMemorySink` | `goldfive.sinks.memory` | (core) | Tests, ephemeral inspection. |
| `LoggingSink` | `goldfive.sinks.logging_sink` | `proto` | stdout / journald-style logs. |
| `JSONLPersistenceSink` | `goldfive.sinks.persistence` | `proto` | Crash recovery via `replay_from_jsonl`. |
| `SQLitePersistenceSink` | `goldfive.sinks.sqlite_sink` | `proto` | Cross-run dashboards, shared DB. |
| `GRPCSink` | `goldfive.sinks.grpc_sink` | `proto` | Stream to a `GoldfiveIngress` server (harmonograf). |

`InMemorySink` is always importable. The others depend on the `proto`
extra. When it is missing, the import guard in
`goldfive/sinks/__init__.py` sets the class attribute to `None` so the
top-level package stays importable.

## Picking one

Decision tree in [docs/guides/choosing-a-sink.md](../docs/guides/choosing-a-sink.md).
Quick version:

- Just want to inspect post-run → `InMemorySink`.
- Want logs → `LoggingSink`.
- Need crash recovery or replay → `JSONLPersistenceSink`.
- Need cross-run dashboards in-process → `SQLitePersistenceSink`.
- Streaming to harmonograf or a server → `GRPCSink` (+ a local JSONL
  sink if you care about the Runner lifecycle dict envelopes).

Multiple sinks compose — `Runner(sinks=[a, b, c])` fans each emit out
concurrently. See `examples/multi_sink_fanout.py`.

## Writing a new sink

Skeleton:

```python
from __future__ import annotations

from typing import Any


class MySink:
    def __init__(self, client) -> None:
        self._client = client
        self._buffer: list[Any] = []

    async def emit(self, event: Any) -> None:
        # Handle both shapes.
        if hasattr(event, "DESCRIPTOR"):
            self._buffer.append(event)
        else:
            # dict envelope from the Runner; translate or skip.
            return
        if len(self._buffer) >= 64:
            await self._flush()

    async def close(self) -> None:
        if self._buffer:
            await self._flush()
        await self._client.shutdown()

    async def _flush(self) -> None:
        batch, self._buffer = self._buffer, []
        await self._client.send_many(batch)
```

## Key design constraints

- **`emit` must not raise on the happy path.** Exceptions bubble up
  through `goldfive.events.emit`'s `asyncio.gather(return_exceptions=True)`
  and are re-raised after all sinks have been awaited — one faulty sink
  shouldn't prevent the others from seeing the event, but it will
  still propagate. Catch and log if you want best-effort.
- **Flush in `close`.** `runner.close()` calls `sink.close()` in turn.
  Anything buffered in-process is lost if you don't.
- **Be prepared for dict envelopes.** At minimum, skip them cleanly.
- **Ordering.** Events are emitted in sequence order per run. Don't
  reorder within a sink.

## Quick reference

```python
from goldfive import InMemorySink, Runner
from goldfive.sinks import JSONLPersistenceSink, LoggingSink

# multi-sink fanout
runner = Runner(
    ...,
    sinks=[
        InMemorySink(),                            # post-run assertions
        LoggingSink(),                             # live stdout tail
        JSONLPersistenceSink("./runs/run.jsonl"),  # crash recovery
    ],
)
```

## Common pitfalls

- Forgot `await runner.close()` → buffered sinks drop data.
- Sink passed somewhere other than `Runner(sinks=[...])` → not plumbed
  to executor / steerer.
- Sink class is `None` because the `proto` extra isn't installed →
  `Runner(sinks=[None])` silently accepts it. Assert not-None.
- `GRPCSink` missing early events — it filters to proto-only. Pair
  with a local JSONL sink or upgrade the server to accept dicts.
- Emitting blocking I/O from `emit` without awaiting → starves the
  event loop. Use `asyncio.to_thread` or an async client.

## Related

- [events.md](events.md) — what events exist and who emits them.
- [docs/guides/choosing-a-sink.md](../docs/guides/choosing-a-sink.md) — which shipped sink to use.
- [docs/guides/writing-an-event-sink.md](../docs/guides/writing-an-event-sink.md) — full prose guide.
- [docs/design/EVENT-MODEL.md](../docs/design/EVENT-MODEL.md) — sink contract and sequence semantics.
