# Public API surface

Hand-maintained reference for goldfive's public API as of v0.1. When
this doc disagrees with the code, trust the code and file a patch to
fix the doc.

Related: [PROTOCOLS.md](../design/PROTOCOLS.md),
[ARCHITECTURE.md](../design/ARCHITECTURE.md).

## Top-level imports

Everything documented here is re-exported from `goldfive.__init__`:

```python
from goldfive import (
    Runner,
    # types
    Goal, Plan, Task, TaskEdge, TaskStatus, DriftKind, DriftSeverity,
    DriftEvent, Session,
    # protocols
    GoalDeriver, Planner, Executor, Steerer, AgentAdapter, EventSink,
    # results
    InvocationResult, ExecutionOutcome,
    # reporting
    ReportingToolSpec,
)
```

Subpackages for implementations:

```python
from goldfive.adapters.callable import CallableAdapter
from goldfive.adapters.adk import ADKAdapter          # extra: adk
from goldfive.adapters.claude import ClaudeAgentSDKAdapter  # extra: claude

from goldfive.executors.sequential import SequentialExecutor
from goldfive.executors.parallel import ParallelDAGExecutor

from goldfive.planner import PassthroughPlanner, LLMPlanner
from goldfive.goal_deriver import PassthroughGoalDeriver, LLMGoalDeriver
from goldfive.steerer import DefaultSteerer

from goldfive.sinks import InMemorySink, LoggingSink, JSONLPersistenceSink
```

## `Runner`

The single entry point.

```python
class Runner:
    def __init__(
        self,
        *,
        agent: AgentAdapter,
        planner: Planner,
        executor: Executor,
        goal_deriver: Optional[GoalDeriver] = None,
        steerer: Optional[Steerer] = None,
        sinks: Optional[list[EventSink]] = None,
        max_plan_reinvocations: int = 3,
    ) -> None: ...

    async def run(
        self,
        user_input: str | list[Goal],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> ExecutionOutcome: ...

    async def resume(
        self,
        session: Session,
    ) -> ExecutionOutcome: ...  # pairs with JSONLPersistenceSink
```

| Parameter | Default | Notes |
|---|---|---|
| `agent` | required | Any `AgentAdapter`. |
| `planner` | required | Any `Planner`. |
| `executor` | required | Any `Executor`. Typically `SequentialExecutor()` or `ParallelDAGExecutor()`. |
| `goal_deriver` | `PassthroughGoalDeriver()` | Optional; defaults substitute when `None`. |
| `steerer` | `DefaultSteerer()` | Optional. |
| `sinks` | `[]` | Optional. Recommended: at least `InMemorySink` in tests and `JSONLPersistenceSink` in prod. |
| `max_plan_reinvocations` | `3` | Executor aborts after this many consecutive refine loops without net progress. |

## Data types (`goldfive.types`)

### `TaskStatus`

```python
class TaskStatus(StrEnum):
    PENDING
    RUNNING
    COMPLETED
    FAILED
    CANCELLED
    BLOCKED
```

### `DriftSeverity`

```python
class DriftSeverity(StrEnum):
    INFO
    WARNING
    CRITICAL
```

### `DriftKind`

25+ values. See [DRIFT.md](../design/DRIFT.md) for the full
taxonomy. Examples:

```python
class DriftKind(StrEnum):
    TOOL_ERROR, AGENT_REFUSAL, NEW_WORK_DISCOVERED, PLAN_DIVERGENCE,
    USER_STEER, USER_CANCEL, TASK_FAILED_RECOVERABLE,
    TASK_FAILED_FATAL, CONTEXT_PRESSURE, BLOCKED, WRONG_AGENT, ...
    CUSTOM
```

### `Task`

```python
@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    assignee_agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    predicted_start_ms: int = 0
    predicted_duration_ms: int = 0
    bound_span_id: str = ""
```

### `TaskEdge`

```python
@dataclass
class TaskEdge:
    from_task_id: str
    to_task_id: str
```

### `Plan`

```python
@dataclass
class Plan:
    id: str
    run_id: str
    goal_ids: list[str]
    tasks: list[Task]
    edges: list[TaskEdge]
    summary: str = ""
    revision_reason: str = ""
    revision_kind: str = ""        # DriftKind value or ""
    revision_severity: str = ""    # DriftSeverity value or ""
    revision_index: int = 0

    def topological_stages(self) -> list[list[Task]]:
        """Kahn's algorithm — returns tasks grouped by DAG depth."""
```

### `Goal`

```python
@dataclass
class Goal:
    id: str
    summary: str
    success_predicate: Optional[Callable[[Session], bool]] = None
    metadata: dict[str, str] = field(default_factory=dict)
```

### `DriftEvent`

```python
@dataclass
class DriftEvent:
    kind: DriftKind
    severity: DriftSeverity
    detail: str = ""
    current_task_id: str = ""
    current_agent_id: str = ""
    raw: Any = None
```

### `Session`

```python
@dataclass
class Session:
    run_id: str
    goals: list[Goal] = field(default_factory=list)
    plan: Optional[Plan] = None
    current_task_id: str = ""
    completed_results: dict[str, str] = field(default_factory=dict)
    task_progress: dict[str, float] = field(default_factory=dict)
    agent_notes: dict[str, str] = field(default_factory=dict)
    divergence_flag: bool = False
    history: list[Any] = field(default_factory=list)
    started_at_ms: int = 0

    def next_sequence(self) -> int:
        """Monotonic event sequence counter."""
```

## Results (`goldfive.results`)

### `InvocationResult`

Returned by `AgentAdapter.invoke()`.

```python
@dataclass
class InvocationResult:
    task_id: str
    text: str = ""
    stop_reason: str = ""
    error: Optional[Exception] = None
    raw: Any = None
```

### `ExecutionOutcome`

Returned by `Executor.run()` and `Runner.run()`.

```python
@dataclass
class ExecutionOutcome:
    success: bool
    session: Session
    reason: str = ""
```

## Protocols (`goldfive.protocols`)

All are `@runtime_checkable`. Full contracts in
[PROTOCOLS.md](../design/PROTOCOLS.md).

| Protocol | Methods |
|---|---|
| `GoalDeriver` | `derive(user_input, *, context=None) -> list[Goal]` |
| `Planner` | `generate(*, goals, available_agents, context=None) -> Optional[Plan]` · `refine(*, plan, drift, goals) -> Optional[Plan]` |
| `Steerer` | `observe(event, session)` · `transition(task_id, to, *, detail="", session)` · `detect_drift(event, session) -> Optional[DriftEvent]` · `bind(*, sinks, planner)` |
| `AgentAdapter` | `register_reporting_tools(tools)` · `invoke(task, session) -> InvocationResult` · property `available_agents: list[str]` |
| `Executor` | `run(*, plan, session, adapter, steerer, planner, sinks) -> ExecutionOutcome` |
| `EventSink` | `emit(event_pb)` · `close()` |

## Default implementations

### GoalDerivers (`goldfive.goal_deriver`)

```python
class PassthroughGoalDeriver(GoalDeriver):
    async def derive(self, user_input, *, context=None): ...

class LLMGoalDeriver(GoalDeriver):
    def __init__(
        self,
        *,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str,
    ): ...
```

### Planners (`goldfive.planner`)

```python
class PassthroughPlanner(Planner):
    def __init__(self, *, plan: Plan): ...

class LLMPlanner(Planner):
    def __init__(
        self,
        *,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str,
        system_prompt_override: Optional[str] = None,
        refine_system_prompt_override: Optional[str] = None,
    ): ...
```

### Steerer (`goldfive.steerer`)

```python
class DefaultSteerer(Steerer):
    # Implements the full state machine + drift classifier from
    # DRIFT.md and STATE-MACHINE.md. No required constructor args.
```

### Executors (`goldfive.executors`)

```python
class SequentialExecutor(Executor):
    def __init__(
        self,
        *,
        nudge_timeout_s: float = 60.0,
    ): ...

class ParallelDAGExecutor(Executor):
    def __init__(
        self,
        *,
        max_concurrency: Optional[int] = None,
        drift_policy: Literal["cancel_stage", "finish_stage"] = "finish_stage",
    ): ...
```

### Adapters (`goldfive.adapters`)

```python
# Always available
class CallableAdapter(AgentAdapter):
    def __init__(
        self,
        agent: Callable[
            [Task, Session, list[ReportingToolSpec]],
            Awaitable[InvocationResult],
        ],
        *,
        available_agents: Optional[list[str]] = None,
    ): ...

# Requires `goldfive[adk]`
class ADKAdapter(AgentAdapter):
    def __init__(
        self,
        root_agent: Any,  # google.adk.BaseAgent
        *,
        runner: Optional[Any] = None,
    ): ...

# Requires `goldfive[claude]`
class ClaudeAgentSDKAdapter(AgentAdapter):
    def __init__(
        self,
        *,
        system_prompt: str,
        model: str,
        client: Optional[Any] = None,  # claude_agent_sdk.ClaudeSDKClient
    ): ...
```

### Sinks (`goldfive.sinks`)

```python
class InMemorySink(EventSink):
    events: list[Event]  # attribute, inspect after the run

class LoggingSink(EventSink):
    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
    ): ...

class JSONLPersistenceSink(EventSink):
    def __init__(
        self,
        path: str,  # may contain {run_id} placeholder
    ): ...

    @classmethod
    def from_jsonl(cls, path: str) -> list[Event]:
        """Load a run's events for replay / recovery."""
```

## Reporting (`goldfive.reporting`)

```python
@dataclass
class ReportingToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[
        [dict[str, Any], Session, Steerer],
        Awaitable[dict[str, Any]],
    ]


# The seven canonical tool names (stable contract).
REPORTING_TOOL_NAMES: frozenset[str] = frozenset({
    "report_task_started",
    "report_task_progress",
    "report_task_completed",
    "report_task_failed",
    "report_task_blocked",
    "report_new_work_discovered",
    "report_plan_divergence",
})


def build_default_reporting_tools(steerer: Steerer) -> list[ReportingToolSpec]:
    """The seven canonical tool specs, wired to `steerer`."""
```

See [tool-protocol.md](tool-protocol.md) for full semantics.

## Events (`goldfive.events`)

```python
def new_event(run_id: str, sequence: int) -> Event:
    """Fresh envelope with run_id, sequence, emitted_at populated."""

def now_ts() -> google.protobuf.Timestamp:
    """Current wall-clock time as a proto Timestamp."""

async def emit(sinks: list[EventSink], event_pb: Event) -> None:
    """Fan out to every sink; swallow per-sink exceptions."""
```

## Proto (`goldfive.pb.goldfive.v1`)

Generated stubs committed to git. Key messages:

```python
# types_pb2
Goal, Plan, Task, TaskEdge, TaskStatus, DriftKind, PlanRevision

# events_pb2
Event, RunStarted, GoalDerived, PlanSubmitted, PlanRevised,
TaskStarted, TaskProgress, TaskCompleted, TaskFailed,
TaskBlocked, TaskCancelled, DriftDetected, RunCompleted, RunAborted
```

The `Event` envelope has a `oneof payload` with one field per
per-event message. See [EVENT-MODEL.md](../design/EVENT-MODEL.md) for
the full catalog.

## Convenience helpers (`goldfive.conv`)

Proto ↔ dataclass round-trip:

```python
def to_pb_plan(plan: Plan) -> pb.Plan: ...
def from_pb_plan(msg: pb.Plan) -> Plan: ...

def to_pb_goal(goal: Goal) -> pb.Goal: ...
def from_pb_goal(msg: pb.Goal) -> Goal: ...

# ... similar for Task, TaskEdge, DriftEvent
```

## Recovery (`goldfive.recovery`)

```python
def reconstruct_session(events: list[Event]) -> Session:
    """Replay events to rebuild a Session from scratch."""
```

Used by `Runner.resume()`. See
[persistence-and-recovery.md](../guides/persistence-and-recovery.md).

## Drift classifiers (`goldfive.drift`)

For custom steerer subclasses that want to compose the standard
classifiers:

```python
def classify_tool_error(err: Exception, tool_name: str) -> DriftEvent: ...
def classify_refusal(text: str) -> Optional[DriftEvent]: ...
def classify_stop_reason(reason: str) -> Optional[DriftEvent]: ...
def classify_schema_violation(payload: Any, schema: Any) -> DriftEvent: ...
```

## Version compatibility

- **v0.1** — the shapes in this document. Stable within v0.1.x patch
  releases.
- **v0.2+** — shapes may be extended (new fields, new enum values,
  new methods on protocols with defaults). Anything in this doc is
  expected to stay backwards-compatible through v1.0.
- **Proto compatibility** — proto3 rules. Never remove or renumber a
  field; add new events by extending the `oneof payload`.

Anything not documented here is internal and subject to change
without notice. Use the public surface.
