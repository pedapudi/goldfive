# goldfive architecture

goldfive is a framework-agnostic harness that wraps an agent and
injects **planning, task tracking, drift analysis, and steering**. The
agent does the thinking; goldfive decides what the agent thinks about
and whether it is still on track.

This document is the top-level architectural reference. It covers the
six primitives, how they compose into a `Runner`, and the full lifecycle
of one `Runner.run()` invocation. Downstream documents go deep on
individual pieces:

- [PROTOCOLS.md](PROTOCOLS.md) — the protocol contracts in detail.
- [EVENT-MODEL.md](EVENT-MODEL.md) — the proto event taxonomy and the
  EventSink contract.
- [DRIFT.md](DRIFT.md) — the drift-kind taxonomy and refine policy.
- [STATE-MACHINE.md](STATE-MACHINE.md) — the task lifecycle state
  diagram.

## The vision

> "Stay on target."

Agents wander. Left alone, a capable LLM agent will often:

- Merge, split, or reorder work in ways the caller did not intend.
- Declare a task "done" when it has only narrated an approach.
- Blow through a context budget and truncate mid-turn.
- Hallucinate artifacts that do not exist.
- Silently abandon the user's real request.

goldfive treats each of these as a **drift** — a structured observation
that execution has diverged from an explicit plan derived from an
explicit goal. Drift is first-class: it is classified, emitted as an
event, and optionally drives a replan.

## The six primitives

| Primitive | Role | Default implementation |
|---|---|---|
| `GoalDeriver` | Converts user input into an explicit list of `Goal`s. | `PassthroughGoalDeriver`, `LLMGoalDeriver` |
| `Planner` | Produces and refines a `Plan` from goals. | `PassthroughPlanner`, `LLMPlanner` |
| `Executor` | Drives plan execution task-by-task. | `SequentialExecutor`, `ParallelDAGExecutor` |
| `Steerer` | Owns the task state machine and classifies drift. | `DefaultSteerer` |
| `AgentAdapter` | Wraps a specific agent framework. | `CallableAdapter`, `ADKAdapter`, `ClaudeAgentSDKAdapter` |
| `EventSink` | Receives proto-encoded orchestration events. | `InMemorySink`, `LoggingSink`, `JSONLPersistenceSink` |

Each primitive is a `Protocol` (see [PROTOCOLS.md](PROTOCOLS.md)).
goldfive ships default implementations but any conforming class can be
swapped in.

### Primitive responsibilities

**GoalDeriver** — one job: take whatever the caller handed in
(`user_input: str` or a pre-built `list[Goal]`) and return an explicit
list of `Goal`s. Each goal has a `summary` and, optionally, a
`success_predicate: Callable[[Session], bool]` that the framework can
consult to decide "are we done yet?".

**Planner** — produces a `Plan` (DAG of `Task`s) from goals, and
revises it on drift via `refine(plan, drift, goals)`. The planner is
the only component that ever decides *what* tasks exist. Executors and
steerers just mechanically walk whatever plan the planner hands them.

**Executor** — walks the plan. Two executors ship in v0.1:
`SequentialExecutor` runs tasks one at a time in topological order;
`ParallelDAGExecutor` batches independent tasks per topological stage
and runs the stage with `asyncio.gather`. The executor is the only
component that calls `adapter.invoke(task, session)`.

**Steerer** — owns the task state machine. It reacts to reporting tool
calls (from the adapter) and to adapter-observed events, transitions
task status monotonically, and runs drift classification. When drift
crosses a severity threshold it invokes `planner.refine(...)`.

**AgentAdapter** — the only primitive that knows about a specific
agent framework (ADK, Claude Agent SDK, a bare callable, ...). It
registers reporting tools, renders current-task context into whatever
form the framework expects (shared session state, a system prompt,
whatever), invokes the agent, and routes intercepted tool calls and
observed events back to the steerer.

**EventSink** — consumes the proto-encoded event stream. goldfive
ships three sinks (in-memory for tests, logging for dev,
JSONL-persistence for crash recovery) and is designed to hand off to
external observability systems. harmonograf is the canonical external
sink (see [harmonograf-integration guide](../guides/harmonograf-integration.md)).

## How they compose

```
                                ┌────────────────────────────────────────┐
                                │                Runner                  │
                                │                                        │
          user_input    ─────▶  │  GoalDeriver ─▶ Planner ─▶ Executor   │  ─────▶  ExecutionOutcome
          (str or                │                  ▲            │        │
           list[Goal])           │                  │            ▼        │
                                │       refine(drift, goals)  adapter.   │
                                │                  │          invoke()   │
                                │                  │            │        │
                                │              Steerer ◀────────┘        │
                                │             (drift kinds)              │
                                │                  │                      │
                                │                  ▼                      │
                                │             EventSinks                  │
                                │        (InMemory, Logging,              │
                                │         JSONL, harmonograf, …)         │
                                └────────────────────────────────────────┘
                                                   │
                                                   ▼
                                    proto Event stream (RunStarted,
                                    PlanSubmitted, TaskStarted, …)
```

**Data direction:**

- `user_input → goals → plan → tasks → invocations`. The "forward"
  pipeline.
- `agent output → steerer observations → drift events → refine(plan)`.
  The "reverse" feedback loop.
- Every state-affecting decision emits an `Event` to every `EventSink`
  in `sinks`. Event emission is fan-out: sinks never see each other.

**Control direction:**

- The `Executor` owns the top-level loop. It calls `adapter.invoke()`
  per task, then asks the `Steerer` whether drift occurred, then — if
  so — calls `Planner.refine()` and swaps in the new plan before the
  next iteration.
- The `Steerer` is passive with respect to control; it classifies and
  transitions, it does not drive execution.

## The `Runner`

`Runner` is the single public entry point. It composes the six
primitives, owns a `Session`, and runs one invocation end-to-end. See
[api.md](../reference/api.md#runner) for the full signature.

Minimal construction:

```python
from goldfive import Runner
from goldfive.adapters.callable import CallableAdapter
from goldfive.planner import PassthroughPlanner
from goldfive.executors.sequential import SequentialExecutor

runner = Runner(
    agent=CallableAdapter(my_async_callable),
    planner=PassthroughPlanner(plan=precomputed_plan),
    executor=SequentialExecutor(),
)
outcome = await runner.run("build me a slide deck about Python")
```

When the caller omits `goal_deriver`, `steerer`, or `sinks`, the
`Runner` substitutes sane defaults:

- `goal_deriver` → `PassthroughGoalDeriver()` (wraps the string in a
  single `Goal`).
- `steerer` → `DefaultSteerer()`.
- `sinks` → `[]` (events go nowhere; recommended to supply at least
  `InMemorySink` in dev and `JSONLPersistenceSink` in prod).

## The full lifecycle

One call to `runner.run(user_input)` walks the following steps. Every
step emits one or more events (see [EVENT-MODEL.md](EVENT-MODEL.md) for
the full taxonomy).

### 1. Setup

- Generate a fresh `run_id` (UUIDv7-style).
- Construct a `Session(run_id=...)`. `Session` holds all live state for
  this invocation.
- Emit `RunStarted(run_id, user_input, started_at)`.

### 2. Goal derivation

- Call `goal_deriver.derive(user_input, context=...)`.
- Store the returned `list[Goal]` in `session.goals`.
- Emit `GoalDerived(goals=...)`.

If the caller passed a `list[Goal]` directly to `runner.run(...)`, the
`PassthroughGoalDeriver` returns it verbatim and goal derivation is a
no-op.

### 3. Plan generation

- Collect `available_agents` from `adapter.available_agents`.
- Call `planner.generate(goals=session.goals, available_agents=..., context=...)`.
- If `None`: abort with `RunAborted(reason="planner returned no plan")`.
- Otherwise: store the `Plan` on `session.plan` and emit
  `PlanSubmitted(plan=...)`.

### 4. Execution loop

Delegated to `executor.run(plan=..., session=..., adapter=..., steerer=..., planner=..., sinks=...)`.

The executor loop (simplified Sequential flow):

```
while session.plan has unfinished tasks:
    for task in topological_order(session.plan):
        if task.status != PENDING: continue
        if any dep of task is not COMPLETED: skip
        steerer.transition(task.id, RUNNING, session=session)
        result = await adapter.invoke(task, session)    # agent runs here
        # adapter has been forwarding reporting-tool calls
        # to steerer throughout the invocation
        drift = steerer.detect_drift(result, session)
        if drift and drift.severity >= WARNING:
            revised = await planner.refine(plan=session.plan, drift=drift, goals=session.goals)
            if revised:
                session.plan = revised
                emit PlanRevised(plan=revised, revision_metadata=...)
                break to outer loop  # restart walk with revised plan
```

ParallelDAGExecutor differs only in that each topological *stage* is
an `asyncio.gather` over the stage's tasks; refine runs between
stages, not mid-stage. See [PROTOCOLS.md](PROTOCOLS.md#executor) for
the full contract.

### 5. Termination

One of three terminal events is emitted:

- `RunCompleted(run_id, outcome_summary)` — all tasks reached a
  terminal state and the plan is fully realized.
- `RunAborted(run_id, reason)` — executor surrendered (e.g. hit
  `max_plan_reinvocations` with tasks still PENDING, planner returned
  `None` during refine, unrecoverable drift).
- Exception propagation — uncaught exceptions surface to
  `runner.run()`'s caller. Sinks still receive a `RunAborted` event
  before the exception re-raises.

The `ExecutionOutcome(success, session, reason)` returned to the
caller captures the same information.

### 6. Sink teardown

- `await sink.close()` is called on every sink.
- `JSONLPersistenceSink.close()` is the one that matters most: it
  flushes buffered writes and releases the file lock. See the
  [persistence guide](../guides/persistence-and-recovery.md).

## Invariants

Four invariants hold across every `Runner.run(...)` invocation:

1. **Monotonic task state** — once a task reaches `COMPLETED`,
   `FAILED`, or `CANCELLED`, it cannot transition out. The steerer
   enforces this. See [STATE-MACHINE.md](STATE-MACHINE.md).
2. **Plan revisions preserve history** — `planner.refine()` may add,
   remove, or re-order tasks, but the executor only re-runs tasks that
   have not yet reached a terminal state. Completed work is never
   re-done.
3. **Events are monotonically sequenced** — every event carries
   `sequence = session.next_sequence()`. Sinks can rely on sequence
   order to reconstruct state.
4. **Sinks are side-effect-only** — goldfive never reads from a sink.
   Sinks may be slow, lossy, or remote without affecting correctness.

## What's out of scope for v0.1

- **CLI.** goldfive is a library. Callers build CLIs on top.
- **Multi-run coordination.** One `Runner.run()` = one `run_id`. For
  resuming a crashed run, see [persistence-and-recovery.md](../guides/persistence-and-recovery.md).
- **Human-in-the-loop mid-run steering.** Control actions (pause,
  resume, external steer) are on the roadmap but not implemented in
  v0.1. External systems can emit drift by synthesizing a
  `DriftEvent(kind=USER_STEER)` and handing it to the steerer, but
  there is no in-box protocol yet.
- **Streaming to the caller.** `Runner.run()` is a batch call.
  Streaming live progress is expected to happen through a caller-
  supplied `EventSink`, not through an async generator.

## Reading order

If you're new, read this doc in order with:

1. [PROTOCOLS.md](PROTOCOLS.md) — the shape contracts.
2. [STATE-MACHINE.md](STATE-MACHINE.md) — how one task moves through
   the lifecycle.
3. [DRIFT.md](DRIFT.md) — what counts as drift and what to do about it.
4. [EVENT-MODEL.md](EVENT-MODEL.md) — how observability flows out.

For hands-on, jump to [getting-started.md](../guides/getting-started.md).
