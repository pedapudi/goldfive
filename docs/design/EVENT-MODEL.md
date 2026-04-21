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
[VOCABULARY.md §6 — Event payload kinds](VOCABULARY.md#6-event-payload-kinds)
(exhaustive factory-to-emitter reference for every `oneof` variant),
[RATIONALE.md §"Why `EventSink` protocol is proto-Event-shaped, not
dict-shaped"](RATIONALE.md#why-eventsink-protocol-is-proto-event-shaped-not-dict-shaped),
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
# pseudo-code: `session`, `task`, and `sinks` are supplied by the caller;
# the `await` runs inside an async function.
from goldfive.events import new_event, emit

event = new_event(run_id=session.run_id, sequence=session.next_sequence())
event.task_started.task_id = task.id
event.task_started.detail = "beginning research phase"
await emit(sinks, event)
```

## Event taxonomy

Event kinds are grouped by the phase they fire in. Since #55 every
event on the stream is a proto `Event` envelope — there is no
mixed-shape stream on the sink side.

### Run lifecycle

| Event | Fired by | When |
|---|---|---|
| `RunStarted` | `Runner` | At the very top of `run()` before goal derivation. Carries `goal_summary` (the incoming `user_input` or the first goal's summary) and `started_at`. |
| `RunCompleted` | `Executor` | When every task has reached a terminal state and the plan is fully realized. Carries `outcome_summary`. |
| `RunAborted` | `Runner`, `Executor` | The `Runner` emits it when setup fails (goal derivation, plan generation, tool registration, steerer bind); the executor emits it when the plan cannot be driven to completion (e.g. `max_task_invocations` exhausted, unrecoverable drift). Carries `reason`. |

### Goal and plan

| Event | Fired by | When |
|---|---|---|
| `GoalDerived` | `Runner` | After `goal_deriver.derive()` returns. Carries the `list[Goal]`. Fired once per run. |
| `PlanSubmitted` | `Runner` | After initial `planner.generate()` succeeds. Carries the full `Plan`. |
| `PlanRevised` | `Executor` | After a successful `planner.refine()` swap. Carries the revised `Plan`, the `revision_reason`, the triggering `DriftKind`, severity, a monotonically increasing `revision_index`, and a `PlanRevisionDiff` sidecar (`added_task_ids`, `removed_task_ids`, `modified_task_ids`, `added_edges`, `removed_edges`) so sinks can render "what changed" without re-fetching the prior plan. See PLAN-LIFECYCLE.md §2.1. |

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

### Agent-invocation events

Three observability-only events describe the **dispatch and
delegation** shape of a run under the ADK registry-dispatch model
(see [ARCHITECTURE.md §"Registry dispatch"](ARCHITECTURE.md#registry-dispatch-goldfive-drives-adk-executes)).
They do not change task state and the framework does not interpret
them; sinks (harmonograf in particular) surface them to make the
"who actually ran what" story visible on a Gantt, especially when a
coordinator invokes `AgentTool`-wrapped specialists.

| Event | Fired by | When |
|---|---|---|
| `AgentInvocationStarted` | ADK plugin `before_run_callback` | At the top of every runner invocation — both the top-level goldfive dispatch and any nested AgentTool-spawned sub-Runner. |
| `AgentInvocationCompleted` | ADK plugin `after_run_callback` | When a runner invocation finishes. One per `AgentInvocationStarted`. |
| `DelegationObserved` | ADK plugin `before_tool_callback` | When the host agent is about to invoke a tool that wraps another agent (ADK's `AgentTool`). |

#### `AgentInvocationStarted`

| Field | Type | Meaning |
|---|---|---|
| `agent_name` | `string` | The dispatched agent — `task.assignee_agent_id` for top-level, the wrapped agent's name for AgentTool-spawned sub-invocations. |
| `task_id` | `string` | The goldfive-dispatched task id. Propagates unchanged into nested invocations — see "Nested ordering" below. |
| `invocation_id` | `string` | ADK's per-run invocation id. Unique per runner invocation. |
| `parent_invocation_id` | `string` | Empty for the top-level dispatch; set to the outer `invocation_id` when the plugin fires on an AgentTool-spawned sub-Runner. |
| `started_at` | `Timestamp` | Emission wall-clock. |

#### `AgentInvocationCompleted`

| Field | Type | Meaning |
|---|---|---|
| `agent_name` | `string` | Matches the corresponding `AgentInvocationStarted`. |
| `task_id` | `string` | Same as the matching Started event. |
| `invocation_id` | `string` | Matches the Started event's `invocation_id`. |
| `summary` | `string` | Optional short description of the outcome (final assistant text, for sinks that render a timeline). |
| `completed_at` | `Timestamp` | Emission wall-clock. |

#### `DelegationObserved`

| Field | Type | Meaning |
|---|---|---|
| `from_agent` | `string` | The host agent whose `before_tool_callback` fired — the one about to call the AgentTool. |
| `to_agent` | `string` | The wrapped agent the AgentTool will invoke. |
| `task_id` | `string` | The goldfive-dispatched task id. |
| `invocation_id` | `string` | The host agent's invocation id at the moment the AgentTool was called. |
| `observed_at` | `Timestamp` | Emission wall-clock. |

#### Nested ordering and the shared `task_id`

In a delegation chain, the events arrive in nested parentheses-like
order and all carry the **same `task_id`** — the goldfive-dispatched
task. This is how harmonograf reconstructs delegation chains on the
Gantt:

```
                time ──────────────────────────▶
  outer:  started ─────────────────────────────── completed
  inner:          started ─── completed
  delegation:     observed (from=outer, to=inner)
```

For a coordinator with an `AgentTool(specialist)` called once,
given a task assigned to the coordinator:

1. `AgentInvocationStarted(agent_name="coordinator",
   task_id="t_42", invocation_id="inv_A", parent_invocation_id="")`
2. `DelegationObserved(from_agent="coordinator",
   to_agent="specialist", task_id="t_42", invocation_id="inv_A")`
3. `AgentInvocationStarted(agent_name="specialist",
   task_id="t_42", invocation_id="inv_B",
   parent_invocation_id="inv_A")`
4. `AgentInvocationCompleted(agent_name="specialist",
   task_id="t_42", invocation_id="inv_B")`
5. `AgentInvocationCompleted(agent_name="coordinator",
   task_id="t_42", invocation_id="inv_A")`

#### What consumers should do

- **Timeline renderers** (harmonograf Gantt) should treat
  `AgentInvocationStarted` / `AgentInvocationCompleted` as a pair
  of brackets defining a span, use `parent_invocation_id` to nest
  spans, and draw a delegation edge from `from_agent`'s active
  span to `to_agent`'s span when a `DelegationObserved` arrives
  between a Started + its Completed.
- **Log-only sinks** can treat the events as informational lines;
  they carry no state-affecting semantics.
- **Steerers and drift classifiers** should ignore these events —
  they are emitted by the plugin and never round-trip through the
  steerer.
- **Correlation with other events.** All events emitted during a
  dispatch share the envelope's `run_id`, so pair them with
  `TaskStarted` / `TaskCompleted` for the same `task_id` to get
  the full per-task story.

Emission is best-effort: a failure in the plugin's observability
path is swallowed and logged at DEBUG — the run never blocks on
observability.

## Sequence semantics

`sequence` is assigned exactly once per event, via
`Session.next_sequence()`:

```python
# pseudo-code: reproduces the body of ``Session.next_sequence`` in
# ``goldfive/types.py`` for reference.
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
# pseudo-code: signature only. The live definition is in
# ``goldfive/protocols.py``.
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

Five sinks ship in `goldfive.sinks`. Pick per use-case; they compose
freely. See [choosing-a-sink.md](../guides/choosing-a-sink.md) for the
full decision matrix.

### `InMemorySink`

```python
# pseudo-code: the ``# ... run ...`` placeholder is where the caller's
# Runner invocation populates the sink.
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

# ``path`` is a literal file path; there is no ``{run_id}`` templating
# built in — callers that want one file per run format the path
# themselves and construct a new sink each run.
sink = JSONLPersistenceSink(path="./runs/example-run.jsonl")
```

Writes one proto-canonical JSON line per event. Paired with
`replay_from_jsonl(path)` for crash recovery. See the full
[persistence guide](../guides/persistence-and-recovery.md).

### `SQLitePersistenceSink`

```python
from goldfive.sinks import SQLitePersistenceSink

sink = SQLitePersistenceSink("./runs/goldfive.db")
```

Inserts each event into a `goldfive_events` table keyed by
`(run_id, sequence)`. Pairs with `replay_from_sqlite(path, run_id)`
and `list_runs(path)`. Useful when you want cross-run SQL queries or
a single shared store for many runs.

### `GRPCSink`

```python
from goldfive.sinks import GRPCSink

sink = GRPCSink("observer.internal:50051")
```

Streams proto `Event` messages to a `GoldfiveIngressServer` over a
client-streaming RPC. Enqueues on an `asyncio.Queue`, drains in the
background, reconnects on transient RPC errors by default. See
[grpc-transport.md](../guides/grpc-transport.md) for TLS, reconnect
semantics, and the server side.

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
# pseudo-code: `my_callable` and `plan` are caller-supplied.
runner = Runner(
    agent=CallableAdapter(my_callable),
    planner=StaticPlanner(plan),
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
