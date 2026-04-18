# Event model

goldfive emits a proto-encoded event stream during every run. Events
are the only observability surface: nothing about run state is
reconstructable without them, and nothing about run state is
published except through them.

This document covers:

- The event taxonomy (what events exist, what each carries).
- Sequence semantics (ordering and uniqueness guarantees).
- The `EventSink` contract (what sinks may and may not do).
- How to build a custom sink.

Related: [ARCHITECTURE.md](ARCHITECTURE.md), [PROTOCOLS.md](PROTOCOLS.md#eventsink),
and the [writing-an-event-sink guide](../guides/writing-an-event-sink.md).

## Event envelope

Every event, regardless of kind, carries a common envelope:

| Field | Type | Meaning |
|---|---|---|
| `event_id` | `string` | UUIDv7, unique per event. |
| `run_id` | `string` | The owning `Session.run_id`. |
| `emitted_at` | `google.protobuf.Timestamp` | Wall-clock time of emission. |
| `sequence` | `uint64` | Monotonic per-run counter from `Session.next_sequence()`. |
| `payload` | `oneof` | The event-specific body; see the taxonomy. |

The envelope is defined in `proto/goldfive/v1/events.proto` (see
[issue #3](https://github.com/pedapudi/goldfive/issues/3)). Generated
Python lives under `goldfive/pb/goldfive/v1/events_pb2.py`.

Helpers in `goldfive/events.py` produce envelopes:

```python
from goldfive.events import new_event, emit

event = new_event(run_id=session.run_id, sequence=session.next_sequence())
event.task_started.task_id = task.id
event.task_started.detail = "beginning research phase"
await emit(sinks, event)
```

## Event taxonomy

Ten event kinds are defined in v0.1, grouped by the phase they fire in.

### Run lifecycle

| Event | Fired by | When |
|---|---|---|
| `RunStarted` | `Runner` | At the very top of `run()` before goal derivation. Carries `user_input` (truncated if long) and `started_at`. |
| `RunCompleted` | `Runner` | When the executor returns `success=True`. Carries `outcome_summary` and `completed_task_ids`. |
| `RunAborted` | `Runner`, `Executor` | When the executor surrenders or the runner catches an exception. Carries `reason` and `drift` (optional) explaining why. |

### Goal and plan

| Event | Fired by | When |
|---|---|---|
| `GoalDerived` | `Runner` | After `goal_deriver.derive()` returns. Carries the `list[Goal]`. Fired once per run. |
| `PlanSubmitted` | `Runner` | After initial `planner.generate()` succeeds. Carries the full `Plan`. |
| `PlanRevised` | `Executor` | After a successful `planner.refine()` swap. Carries the revised `Plan`, the `revision_reason`, the triggering `DriftKind`, severity, and a monotonically increasing `revision_index`. |

### Task lifecycle

One event per task state transition. Emitted by the `Steerer` when it
applies the transition to the session.

| Event | State transition |
|---|---|
| `TaskStarted` | PENDING → RUNNING |
| `TaskProgress` | RUNNING → RUNNING (reports fraction 0.0–1.0; not a transition) |
| `TaskCompleted` | RUNNING → COMPLETED |
| `TaskFailed` | RUNNING → FAILED |
| `TaskBlocked` | RUNNING → BLOCKED (or stays RUNNING with a blocker note) |
| `TaskCancelled` | PENDING → CANCELLED (executor-driven, e.g. after an unrecoverable drift cascade) |

Every task-lifecycle event carries `task_id`, a human-readable `detail`
string, and an optional `artifacts` map for `TaskCompleted`.

### Drift

| Event | Fired by | When |
|---|---|---|
| `DriftDetected` | `Steerer` | Whenever `detect_drift()` returns a non-None `DriftEvent`. Always fired before the corresponding `PlanRevised` (if refine runs). Carries `kind`, `severity`, `detail`, `current_task_id`, and a summarized `raw` trigger. |

See [DRIFT.md](DRIFT.md) for the full drift-kind taxonomy.

## Sequence semantics

`sequence` is assigned exactly once per event, via
`Session.next_sequence()`:

```python
def next_sequence(self) -> int:
    s = self._next_sequence
    self._next_sequence = s + 1
    return s
```

`next_sequence` is not thread-safe by itself; the `ParallelDAGExecutor`
is careful to only call it from the executor's coroutine, not from
inside `asyncio.gather`'d worker tasks. Steerer transitions happen
serially.

Sequence provides three guarantees:

1. **Uniqueness.** No two events in one run share a sequence.
2. **Monotonicity.** An event's sequence is always greater than
   every prior event's sequence in the same run.
3. **Total ordering.** Every event in a run has a sequence, and the
   sequences form a total order that reflects the order events were
   emitted.

Sinks may rely on these to reconstruct state. In particular,
`JSONLPersistenceSink` round-trips through sequence order when
replaying a crashed run.

### Happens-before relationships

Beyond the per-run total order, goldfive guarantees:

- `RunStarted` is the first event; `RunCompleted` or `RunAborted` is
  the last.
- `GoalDerived` precedes `PlanSubmitted`.
- `PlanSubmitted` precedes every task-lifecycle event.
- `TaskStarted(t)` precedes `TaskCompleted(t)`, `TaskFailed(t)`, or
  `TaskBlocked(t)` for the same task `t`.
- `DriftDetected` precedes `PlanRevised` when the drift triggered a
  successful refine.

## The EventSink contract

```python
@runtime_checkable
class EventSink(Protocol):
    async def emit(self, event_pb: Any) -> None: ...
    async def close(self) -> None: ...
```

### What sinks may do

- Serialize events to disk, database, gRPC, HTTP, stdout, anywhere.
- Filter, transform, or batch internally.
- Raise exceptions. goldfive catches and logs but does not re-raise
  to the caller — one failing sink cannot take down the run.

### What sinks must not do

- **Block indefinitely.** `emit()` is on the critical path. A sink
  that blocks stalls the executor. Use a bounded internal queue if
  the downstream is slow.
- **Read back.** goldfive never reads from a sink. If a sink wants to
  expose a read interface (e.g. `InMemorySink.events`), that is
  orthogonal to the goldfive protocol.
- **Mutate the event.** The same event object is passed to every
  sink. Treat it as immutable.
- **Depend on other sinks.** Each sink receives the event
  independently; order between sinks is unspecified.

### Lifecycle

`sink.emit()` is called once per event, in sequence order, from
whichever coroutine is doing the emission. The calling coroutine
awaits the returned coroutine, so slow sinks slow down the run.

`sink.close()` is called once per run, after the terminal event
(`RunCompleted` or `RunAborted`) has been emitted. Sinks must flush
any buffered writes and release resources (file handles, sockets)
before returning.

## Built-in sinks

### `InMemorySink`

```python
from goldfive.sinks import InMemorySink

sink = InMemorySink()
# ... run ...
for event in sink.events:
    print(event.WhichOneof("payload"), event.sequence)
```

Keeps every event in a list. For tests. Never use in production.

### `LoggingSink`

```python
import logging
from goldfive.sinks import LoggingSink

logging.basicConfig(level=logging.INFO)
sink = LoggingSink(logger=logging.getLogger("goldfive"), level=logging.INFO)
```

Logs a one-line summary per event via the standard `logging` module.
Useful for local dev.

### `JSONLPersistenceSink`

```python
from goldfive.sinks import JSONLPersistenceSink

sink = JSONLPersistenceSink(path="./runs/{run_id}.jsonl")
```

Writes one JSON-encoded proto message per line. Paired with
`from_jsonl(path)` for crash recovery. See the full
[persistence guide](../guides/persistence-and-recovery.md).

## Building a custom sink

The shape is minimal:

```python
from __future__ import annotations

from typing import Any

from goldfive.protocols import EventSink


class PrintingSink:
    """Prints one line per event to stdout."""

    async def emit(self, event_pb: Any) -> None:
        kind = event_pb.WhichOneof("payload")
        print(f"[{event_pb.sequence:4d}] {kind} (run={event_pb.run_id})")

    async def close(self) -> None:
        pass


assert isinstance(PrintingSink(), EventSink)  # runtime-checkable
```

Plug into a `Runner`:

```python
runner = Runner(
    agent=CallableAdapter(my_callable),
    planner=PassthroughPlanner(plan),
    executor=SequentialExecutor(),
    sinks=[PrintingSink(), JSONLPersistenceSink("./runs/example.jsonl")],
)
```

Every sink receives every event. Order between sinks is unspecified —
do not design a sink that depends on another sink having already seen
an event.

For a more substantial example, see the
[writing-an-event-sink guide](../guides/writing-an-event-sink.md) —
it walks through a sink that forwards events to a custom observability
backend.

## Forward compatibility

The proto event schema adheres to proto3 compatibility rules:

- Never remove or renumber a field.
- Add new event kinds by extending the `oneof payload`; old sinks
  ignore unknown kinds.
- Add new fields as `optional`; old sinks ignore them.

The practical implication: **an event-sink written against v0.1 will
continue to receive a valid event stream from later goldfive versions,
with new fields and kinds appearing as unknowns.**
