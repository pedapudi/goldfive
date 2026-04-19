# Protocol contracts

goldfive is assembled from six composable `Protocol`s. Each protocol
is `@runtime_checkable`, so duck-typed implementations pass
`isinstance(x, Protocol)` at construction time. This doc enumerates
the contract of each protocol — the shape, the semantics, and the
invariants a correct implementation must uphold — plus minimal
working implementations.

Related: [ARCHITECTURE.md](ARCHITECTURE.md), [api.md](../reference/api.md),
[VOCABULARY.md](VOCABULARY.md) for the enums and dataclasses these
signatures reference, [RATIONALE.md](RATIONALE.md) for the design
decisions behind each protocol.

All method signatures in this doc track `goldfive/protocols.py`. When
they disagree, the source wins.

## GoalDeriver

```python
# pseudo-code: protocol signature — live definition lives in
# ``goldfive/protocols.py``.
@runtime_checkable
class GoalDeriver(Protocol):
    async def derive(
        self,
        user_input: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> list[Goal]: ...
```

### Contract

- Takes a single string (the user's raw request) and an optional
  context map.
- Returns a list of one or more `Goal` objects. Each `Goal` has a
  unique `id`, a `summary` string, and an optional `success_predicate`.
- Called exactly once per `Runner.run()`.
- Must be pure with respect to the run state: no writes to the
  session, no side effects on sinks.

### Invariants

1. Non-empty return. A goal-deriver that genuinely cannot derive any
   goal must raise, not return `[]`. The `Runner` aborts on empty.
2. Unique `Goal.id` within the returned list.
3. No mutation of `user_input` or `context`.

### Minimal implementation

```python
from __future__ import annotations

from typing import Any, Mapping, Optional

from goldfive.types import Goal


class PassthroughGoalDeriver:
    """Wraps the raw user input in a single Goal."""

    async def derive(
        self,
        user_input: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> list[Goal]:
        return [Goal(id="g1", summary=user_input)]
```

See the [goals-and-plans guide](../guides/goals-and-plans.md) for when
to write an `LLMGoalDeriver` instead.

## Planner

```python
# pseudo-code: protocol signature — live definition lives in
# ``goldfive/protocols.py``.
@runtime_checkable
class Planner(Protocol):
    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Plan]: ...

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Optional[Plan]: ...
```

### Contract — `generate`

- Called once per run, between goal derivation and executor dispatch.
- Receives the full list of goals and the adapter's
  `available_agents`.
- Returns a `Plan` (DAG of `Task`s) whose tasks' `assignee_agent_id`
  values are drawn from `available_agents`. Returning `None` aborts
  the run with `goal_unreachable`.

### Contract — `refine`

- Called when the steerer classifies a drift of severity ≥ warning
  and the current plan has unfinished tasks.
- Receives the current (possibly already-revised) plan, the drift
  event, and the original goals.
- Returns a revised plan that **preserves every completed task's
  identity and outcome**, or `None` to signal unrecoverable. Returning
  the same plan object unchanged is legal but pointless; the executor
  re-runs the outer loop regardless.

### Invariants

1. Generated plans are acyclic. `Plan.topological_stages()` raises on
   a cycle; planners are responsible for producing valid DAGs.
2. `plan.goal_ids` lists every goal the plan intends to satisfy.
3. `refine()` preserves `Task.id` for completed tasks; it may add,
   remove, or edit `PENDING` tasks freely.
4. `refine()` monotonically increments `plan.revision_index`.

### Minimal implementation

```python
from __future__ import annotations

from typing import Any, Mapping, Optional

from goldfive.types import DriftEvent, Goal, Plan


class PassthroughPlanner:
    """No-op planner — ``generate`` and ``refine`` both return ``None``.

    Makes it safe to wire a ``planner=`` kwarg everywhere without
    forcing callers to opt in to planning on day one. Mirrors the
    live implementation in ``goldfive/planner.py``.
    """

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Plan]:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Optional[Plan]:
        return None
```

Callers who already have a pre-built plan use ``StaticPlanner(plan)``
instead — it returns the supplied plan verbatim and also declines to
refine. For the full `LLMPlanner` that parses JSON plans out of an LLM
response, see `goldfive/planner.py` or
[goals-and-plans.md](../guides/goals-and-plans.md#writing-a-custom-planner).

## Steerer

```python
# pseudo-code: protocol signature — live definition lives in
# ``goldfive/protocols.py``.
@runtime_checkable
class Steerer(Protocol):
    async def observe(self, event: Any, session: Session) -> None: ...

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None: ...

    def detect_drift(self, event: Any, session: Session) -> Optional[DriftEvent]: ...

    def bind(
        self,
        *,
        sinks: list["EventSink"],
        planner: Planner,
    ) -> None: ...
```

### Contract

- `observe(event, session)` — the adapter streams raw framework
  events (LLM text, tool calls, stream chunks) here. The steerer may
  classify drift, update per-task progress, or no-op. Must not mutate
  `event`.
- `transition(task_id, to, session)` — the single source-of-truth
  state-mutating method. Enforces monotonicity (see
  [STATE-MACHINE.md](STATE-MACHINE.md)) and emits one event per
  successful transition.
- `detect_drift(event, session)` — pure classifier. Returns a
  `DriftEvent` or `None`. No state mutation, no event emission.
- `bind(sinks, planner)` — called once by the executor before the
  first `observe`. Wires the steerer to its downstream dependencies.

### Invariants

1. Terminal states absorb. `transition(t, to, ...)` is a no-op if `t`
   is already in `{COMPLETED, FAILED, CANCELLED}`.
2. Exactly one event emitted per successful transition.
3. `detect_drift()` is side-effect-free.
4. Steerer operations are serialized per-run (callers guarantee this).

### Minimal implementation

```python
from __future__ import annotations

from typing import Any, Optional

from goldfive.types import DriftEvent, Session, TaskStatus
from goldfive.protocols import EventSink, Planner


_TERMINAL = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class MinimalSteerer:
    """Minimal but compliant Steerer. No drift detection."""

    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Optional[Planner] = None

    def bind(self, *, sinks: list[EventSink], planner: Planner) -> None:
        self._sinks = sinks
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:
        # no-op
        return

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None:
        assert session.plan is not None
        task = next(t for t in session.plan.tasks if t.id == task_id)
        if task.status in _TERMINAL:
            return
        task.status = to
        # (skip event emission for brevity — see DefaultSteerer)

    def detect_drift(self, event: Any, session: Session) -> Optional[DriftEvent]:
        return None
```

The production `DefaultSteerer` in `goldfive/steerer.py` adds the full
drift classifier and event emission.

> See [RATIONALE.md §"Why `Steerer` is a protocol, and what
> `DefaultSteerer` does"](RATIONALE.md#why-steerer-is-a-protocol-and-what-defaultsteerer-does)
> for the state-vs-observation-vs-drift split rationale, and
> [RATIONALE.md §"Why the Steerer invokes `planner.refine` and not the
> Executor"](RATIONALE.md#why-the-steerer-invokes-plannerrefine-and-not-the-executor)
> for why refine lives here.

## AgentAdapter

```python
# pseudo-code: protocol signature — live definition lives in
# ``goldfive/protocols.py``.
@runtime_checkable
class AgentAdapter(Protocol):
    async def register_reporting_tools(
        self,
        tools: list[ReportingToolSpec],
    ) -> None: ...

    async def invoke(
        self,
        task: Task,
        session: Session,
    ) -> InvocationResult: ...

    @property
    def available_agents(self) -> list[str]: ...
```

### Contract

- `register_reporting_tools(tools)` — called once before the first
  invoke. Adapter translates `ReportingToolSpec` into whatever form
  its framework uses (ADK `FunctionTool`, Claude SDK inline tool,
  etc.) and wires a hook that intercepts calls and routes them
  through the steerer.
- `invoke(task, session)` — runs the wrapped agent for one task.
  Must render current-task context (task id, description, completed
  results) into the agent's input channel. Returns an
  `InvocationResult` with the final text and stop reason.
- `available_agents` — names of agents the adapter can dispatch to.
  Consumed by the planner.

### Invariants

1. `invoke` must not return until the agent has finished its turn.
2. Reporting tool interceptions run through the steerer before the
   tool body returns its `{"acknowledged": True}` ack.
3. Observed events (tool calls, text chunks, stream blocks) are fed
   to `steerer.observe()` in order. Drift classification happens in
   the steerer, not the adapter.
4. Exceptions raised by the wrapped agent propagate out of `invoke`.
   The executor catches and classifies.

### Minimal implementation (CallableAdapter)

```python
from __future__ import annotations

from typing import Awaitable, Callable

from goldfive.protocols import AgentAdapter
from goldfive.reporting import ReportingToolSpec
from goldfive.results import InvocationResult
from goldfive.types import Session, Task


AgentFn = Callable[
    [Task, Session, list[ReportingToolSpec]],
    Awaitable[InvocationResult],
]


class CallableAdapter:
    """Wraps an async callable as an AgentAdapter."""

    def __init__(
        self,
        agent: AgentFn,
        *,
        available_agents: list[str] | None = None,
    ) -> None:
        self._agent = agent
        self._tools: list[ReportingToolSpec] = []
        self._available_agents = available_agents or ["default"]

    async def register_reporting_tools(
        self,
        tools: list[ReportingToolSpec],
    ) -> None:
        self._tools = tools

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        return await self._agent(task, session, self._tools)

    @property
    def available_agents(self) -> list[str]:
        return self._available_agents
```

See [writing-an-agent-adapter.md](../guides/writing-an-agent-adapter.md)
for a full worked example that wraps a hypothetical new framework.

> See [RATIONALE.md §"Why `AgentAdapter` exists and isn't the agent
> itself"](RATIONALE.md#why-agentadapter-exists-and-isnt-the-agent-itself)
> for the "how to invoke" vs "how to be invoked" split.

## Executor

```python
# pseudo-code: protocol signature — live definition lives in
# ``goldfive/protocols.py``.
@runtime_checkable
class Executor(Protocol):
    async def run(
        self,
        *,
        plan: Plan,
        session: Session,
        adapter: AgentAdapter,
        steerer: Steerer,
        planner: Planner,
        sinks: list["EventSink"],
    ) -> ExecutionOutcome: ...
```

### Contract

- Walks the plan in topological order.
- For each `PENDING` task whose dependencies are `COMPLETED`:
  - Calls `steerer.transition(task.id, RUNNING, session=session)`.
  - Calls `adapter.invoke(task, session)` and awaits the result.
  - Routes observed events / results to `steerer.observe()`.
  - Calls `steerer.detect_drift(result, session)` after the task
    returns.
- On drift ≥ warning: calls `planner.refine(...)`, swaps the plan,
  emits `PlanRevised`, and restarts the loop.
- Enforces `max_plan_reinvocations` (typically 3): if the loop runs
  that many times without net progress, aborts with `RunAborted`.
- Emits `RunStarted` at entry and `RunCompleted` / `RunAborted` at
  exit.

### Invariants

1. **No mid-stage replan (parallel).** Parallel-DAG finishes the
   current stage before refining. Sequential can refine after any
   task.
2. **Termination.** Every `Executor.run()` terminates: either all
   tasks reach a terminal state (success) or the reinvocation limit
   is hit (abort).
3. **Single-pass per task.** A task cannot transition out of `RUNNING`
   without the executor having invoked the adapter for it.

### Minimal implementation

```python
from __future__ import annotations

from goldfive.protocols import AgentAdapter, EventSink, Executor, Planner, Steerer
from goldfive.results import ExecutionOutcome
from goldfive.types import Plan, Session, TaskStatus


class LinearSequentialExecutor:
    """Minimal executor — assumes a linear plan (no branching), no refine."""

    async def run(
        self,
        *,
        plan: Plan,
        session: Session,
        adapter: AgentAdapter,
        steerer: Steerer,
        planner: Planner,
        sinks: list[EventSink],
    ) -> ExecutionOutcome:
        steerer.bind(sinks=sinks, planner=planner)
        for task in plan.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            await steerer.transition(task.id, TaskStatus.RUNNING, session=session)
            result = await adapter.invoke(task, session)
            # adapter has been calling steerer.observe() throughout invoke()
            final_status = (
                TaskStatus.FAILED if result.error else TaskStatus.COMPLETED
            )
            await steerer.transition(task.id, final_status, session=session)
        return ExecutionOutcome(success=True, session=session)
```

The production executors (`SequentialExecutor`, `ParallelDAGExecutor`)
add topological scheduling, drift handling, refine, and the
reinvocation loop.

> See [RATIONALE.md §"Why `Executor` is a protocol instead of a
> function"](RATIONALE.md#why-executor-is-a-protocol-instead-of-a-function)
> for why this shape was chosen.

## EventSink

```python
# pseudo-code: protocol signature — live definition lives in
# ``goldfive/protocols.py``.
@runtime_checkable
class EventSink(Protocol):
    async def emit(self, event_pb: Any) -> None: ...
    async def close(self) -> None: ...
```

### Contract

- `emit(event_pb)` is called once per event, in sequence order, from
  whichever coroutine is emitting.
- `close()` is called once per run after the terminal event.
- Sinks may raise; goldfive catches and logs. One failing sink cannot
  take down the run.

See [EVENT-MODEL.md](EVENT-MODEL.md#the-eventsink-contract) for the
full semantics and [writing-an-event-sink.md](../guides/writing-an-event-sink.md)
for a walkthrough.

> See [RATIONALE.md §"Why `EventSink` protocol is proto-Event-shaped,
> not dict-shaped"](RATIONALE.md#why-eventsink-protocol-is-proto-event-shaped-not-dict-shaped)
> for why the contract is proto, and
> [RATIONALE.md §"Why `make_event` (dict) coexists with typed
> factories"](RATIONALE.md#why-make_event-dict-coexists-with-typed-factories-proto)
> for why the dict escape hatch stays.

### Minimal implementation

```python
class NullSink:
    async def emit(self, event_pb):
        return

    async def close(self):
        return
```

## Composition example

See [getting-started.md](../guides/getting-started.md) for a full
runnable `Runner` composed from these minimal implementations plus
a 3-task plan.
