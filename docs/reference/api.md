# Public API surface

Hand-maintained reference for goldfive's public API as of v0.1. When
this doc disagrees with the code, trust the code and file a patch to
fix the doc.

Related: [PROTOCOLS.md](../design/PROTOCOLS.md),
[ARCHITECTURE.md](../design/ARCHITECTURE.md).

## Top-level imports

Everything documented here is re-exported from `goldfive.__init__`
(see `goldfive.__all__` for the canonical list):

```python
from goldfive import (
    Runner, wrap, run, quickstart,
    # types
    Goal, Plan, Task, TaskEdge, TaskStatus, DriftKind, DriftSeverity,
    DriftEvent, Session,
    # protocols
    GoalDeriver, Planner, Executor, Steerer, AgentAdapter, EventSink,
    # results
    InvocationResult, ExecutionOutcome,
    # reporting
    ReportingToolSpec, BUILTIN_REPORTING_TOOLS,
    # live steering
    ControlChannel, ControlMessage, ControlAck, ControlKind, AckResult,
    # default implementations re-exported at the package root
    CallableAdapter,
    SequentialExecutor, ParallelDAGExecutor,
    PassthroughPlanner, StaticPlanner, LLMPlanner,
    PassthroughGoalDeriver, LiteralGoalDeriver, LLMGoalDeriver,
    DefaultSteerer,
    InMemorySink, LoggingSink,
    JSONLPersistenceSink, SQLitePersistenceSink, GRPCSink,
    # drift helpers
    classify_tool_error, classify_refusal, classify_stop_reason,
)
```

Subpackages for framework adapters gated behind optional extras:

```python
from goldfive.adapters.callable import CallableAdapter
from goldfive.adapters.adk import ADKAdapter          # extra: adk
from goldfive.adapters.claude import ClaudeAgentSDKAdapter  # extra: claude
from goldfive.adapters.auto import auto_adapter
```

`LoggingSink`, `JSONLPersistenceSink`, `SQLitePersistenceSink`, and
`GRPCSink` require the `proto` extra at runtime (they import
`google.protobuf` eagerly; `GRPCSink` also needs `grpcio`). They
appear in `goldfive.__all__` unconditionally; when the extra is
missing the module attribute resolves to `None` at import time,
surfacing the missing dependency at construction rather than import.

## `goldfive.wrap` / `goldfive.run`

One-line convenience wrappers over `Runner`. `wrap` returns a
`Runner` pre-wired with sensible defaults and an auto-detected
`AgentAdapter`; `run` is `wrap(...).run(user_input)` in a single
call.

```python
def wrap(
    agent: Any,
    *,
    planner: Optional[Planner] = None,
    goal_deriver: Optional[GoalDeriver] = None,
    executor: Optional[Executor] = None,
    steerer: Optional[Steerer] = None,
    sinks: Optional[list[EventSink]] = None,
    call_llm: Optional[Callable[[str, str, str], Awaitable[str]]] = None,
    model: Optional[str] = None,
    max_plan_reinvocations: int = 32,
) -> Runner: ...


async def run(
    agent: Any,
    user_input: str | list[Goal],
    *,
    context: Optional[Mapping[str, Any]] = None,
    **wrap_kwargs: Any,
) -> ExecutionOutcome: ...
```

`agent` can be any of:

- an object implementing `goldfive.AgentAdapter` (used verbatim),
- a `google.adk.agents.BaseAgent` or ADK `Runner` (requires `goldfive[adk]`),
- a zero-arg callable returning `claude_agent_sdk.ClaudeSDKClient` (requires `goldfive[claude]`),
- an async `(task, session, tools) -> InvocationResult` callable.

Adapter dispatch favours ADK over the async-callable path so ADK
agents are not misrouted to `CallableAdapter`. Unknown shapes raise
`TypeError` with a pointer to the supported options.

When no `call_llm` is supplied and the agent does not expose an LLM
surface `wrap` can detect (currently only ADK), `wrap` falls back to
`PassthroughPlanner` + `LiteralGoalDeriver` and emits a `DEBUG`
log line on the `goldfive.wrap` logger.

Direct `auto_adapter` access is available for callers who want the
dispatch logic without the Runner defaults:

```python
from goldfive.adapters.auto import auto_adapter

adapter: AgentAdapter = auto_adapter(agent)
```

## `quickstart`

One-call `Runner` factory for the common case. Complements `wrap` —
`quickstart` always uses a static one-task-per-goal plan rather than
an LLM-driven planner.

```python
def quickstart(
    agent: Any,
    goals: str | Goal | list[str | Goal],
    *,
    planner: Planner | None = None,
    sinks: list[EventSink] | None = None,
) -> Runner: ...
```

- `agent` — either an existing `AgentAdapter` (used verbatim) or a
  `CallableAdapter`-compatible async callable (wrapped).
- `goals` — a single string, a single `Goal`, or a list of either.
  Each entry becomes one task in the default plan.
- `planner` / `sinks` — optional overrides. Defaults to a
  `StaticPlanner` whose plan has one task per goal and
  `[InMemorySink()]` respectively.

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
        control: Optional[ControlChannel] = None,
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
        persistence_path: str,
    ) -> ExecutionOutcome: ...  # best-effort JSONL replay

    async def close(self) -> None: ...  # close every sink
```

`Runner.resume(persistence_path)` loads a JSONL log written by
`JSONLPersistenceSink`, replays it through
`goldfive.sinks.reconstruct_session`, and returns an
`ExecutionOutcome` whose `session` reflects the latest state seen in
the log. v0.1 does not continue execution from the cursor — callers
who want live continuation should build a fresh `Runner` from the
recovered goals. Tracked in issue #15.

| Parameter | Default | Notes |
|---|---|---|
| `agent` | required | Any `AgentAdapter`. |
| `planner` | required | Any `Planner`. |
| `executor` | required | Any `Executor`. Typically `SequentialExecutor()` or `ParallelDAGExecutor()`. |
| `goal_deriver` | `PassthroughGoalDeriver("run")` | Optional. When `None`, the Runner substitutes a passthrough deriver that emits a single `Goal(id="g1", summary="run")`. |
| `steerer` | `DefaultSteerer()` | Optional. |
| `sinks` | `[]` | Optional. Recommended: at least `InMemorySink` in tests and `JSONLPersistenceSink` in prod. |
| `control` | `None` | Optional `ControlChannel` for live pause / cancel / steer / rewind / approve / reject. When `None`, the run has no live-steering surface. See [../design/CONTROL.md](../design/CONTROL.md). |
| `max_plan_reinvocations` | `3` | Stamped onto the planner context so executors that honour it can cap refine loops. |

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

25 values. See [DRIFT.md](../design/DRIFT.md) for the full
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

All are `@runtime_checkable`. Every method is `async` unless explicitly
marked otherwise below. Full contracts in
[PROTOCOLS.md](../design/PROTOCOLS.md).

| Protocol | Methods |
|---|---|
| `GoalDeriver` | `derive(user_input, *, context=None) -> list[Goal]` |
| `Planner` | `generate(*, goals, available_agents, context=None) -> Optional[Plan]` · `refine(*, plan, drift, goals) -> Optional[Plan]` |
| `Steerer` | `observe(event, session)` · `transition(task_id, to, *, detail="", session)` · `detect_drift(event, session) -> Optional[DriftEvent]` (sync) · `bind(*, sinks, planner)` (sync) |
| `AgentAdapter` | `register_reporting_tools(tools)` · `invoke(task, session) -> InvocationResult` · property `available_agents: list[str]` (sync) |
| `Executor` | `run(*, plan, session, adapter, steerer, planner, sinks) -> ExecutionOutcome` |
| `EventSink` | `emit(event_pb)` · `close()` |

## Default implementations

### GoalDerivers (`goldfive.goal_deriver`)

```python
class PassthroughGoalDeriver(GoalDeriver):
    def __init__(self, goals: str | list[str] | list[Goal]) -> None: ...
    # ``derive(user_input, ...)`` ignores ``user_input`` and returns
    # the pre-configured goals verbatim.

class LiteralGoalDeriver(GoalDeriver):
    def __init__(self) -> None: ...
    # ``derive(user_input, ...)`` wraps a non-empty string as a single
    # ``Goal(id="g1", summary=user_input)``.

class LLMGoalDeriver(GoalDeriver):
    def __init__(
        self,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str = "",
        *,
        system_prompt: Optional[str] = None,
    ) -> None: ...
```

### Planners (`goldfive.planner`)

```python
class PassthroughPlanner(Planner):
    def __init__(self) -> None: ...
    # generate() and refine() always return None.

class StaticPlanner(Planner):
    def __init__(self, plan: Plan) -> None: ...
    # generate() returns a fresh copy of ``plan`` with run_id and
    # goal_ids rewritten to match the current session. refine()
    # always returns None.

class LLMPlanner(Planner):
    def __init__(
        self,
        *,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str = "",
        system_prompt: Optional[str] = None,
        refine_system_prompt: Optional[str] = None,
    ) -> None: ...
```

### Steerer (`goldfive.steerer`)

```python
class DefaultSteerer(Steerer):
    # Implements the full state machine + drift classifier from
    # DRIFT.md and STATE-MACHINE.md. No required constructor args.
```

In addition to the `Steerer` protocol methods, `DefaultSteerer`
exposes `mark_task_running`, `mark_task_progress`,
`mark_task_completed`, `mark_task_failed`, `mark_task_blocked`,
`mark_task_cancelled`, `report_new_work_discovered`, and
`report_plan_divergence` for the canonical reporting-tool handlers to
call into.

### Executors (`goldfive.executors`)

```python
class SequentialExecutor(Executor):
    def __init__(
        self,
        *,
        max_plan_reinvocations: int = 32,
        fail_fast: bool = True,
    ) -> None: ...

class ParallelDAGExecutor(Executor):
    def __init__(
        self,
        max_concurrency: int = 0,  # 0 = unbounded fan-out
        drift_policy: Literal["cancel_stage", "finish_stage"] = "finish_stage",
        max_plan_reinvocations: int = 3,
    ) -> None: ...
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
    ) -> None: ...

# Requires `goldfive[adk]`
class ADKAdapter(AgentAdapter):
    def __init__(
        self,
        agent_or_runner: Any,  # google.adk.BaseAgent OR an existing Runner
        *,
        user_id: str = "goldfive_user",
        session_id: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> None: ...

# Requires `goldfive[claude]`
class ClaudeAgentSDKAdapter(AgentAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[], Any],  # () -> claude_agent_sdk.ClaudeSDKClient
        steerer: Optional[Steerer] = None,
        system_prompt_template: Optional[str] = None,
        model: Optional[str] = None,
        available_agents: Optional[list[str]] = None,
    ) -> None: ...

    def bind_steerer(self, steerer: Steerer) -> None:
        """Wire a :class:`Steerer` in after construction."""
```

### Sinks (`goldfive.sinks`)

```python
class InMemorySink(EventSink):
    events: list[Any]  # property — the live event list

class LoggingSink(EventSink):
    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
    ) -> None: ...

class JSONLPersistenceSink(EventSink):
    def __init__(
        self,
        path: str | Path,
        mode: Literal["append", "write"] = "append",
    ) -> None: ...

class SQLitePersistenceSink(EventSink):
    def __init__(
        self,
        path: str | Path,
        *,
        table: str = "goldfive_events",
    ) -> None: ...

class GRPCSink(EventSink):
    def __init__(
        self,
        endpoint: str,
        *,
        credentials: Optional[grpc.ChannelCredentials] = None,
        reconnect: bool = True,
        max_queue: int = 0,  # 0 = unbounded
    ) -> None: ...


# Module-level replay / reconstruction helpers.
def replay_from_jsonl(path: str | Path) -> list[Any]:
    """Parse a JSONL log written by ``JSONLPersistenceSink`` into Event
    proto messages, in emit order."""

def replay_from_sqlite(
    path: str | Path,
    run_id: str,
    *,
    table: str = "goldfive_events",
) -> list[Any]:
    """Read a single run's events from a SQLite database, ordered by
    sequence."""

def list_runs(
    path: str | Path,
    *,
    table: str = "goldfive_events",
) -> list[str]:
    """Return the distinct run_ids present in a SQLite database."""

def reconstruct_session(events: list[Any]) -> Session:
    """Replay events to rebuild a best-effort :class:`Session`."""
```

`LoggingSink`, `JSONLPersistenceSink`, `SQLitePersistenceSink`,
`GRPCSink`, and the replay helpers require the `proto` extra (they
lean on `google.protobuf`; `GRPCSink` also requires `grpcio`). When
the extra is missing the symbols resolve to `None` at import time
so the rest of the package stays usable.

### Ingress server (`goldfive.server`)

```python
class GoldfiveIngressServer:
    def __init__(
        self,
        sinks: list[EventSink],
        *,
        credentials: Any = None,
        server_options: list[tuple[str, Any]] | None = None,
    ) -> None: ...
    async def start(self, host: str = "127.0.0.1", port: int = 50051) -> int: ...
    async def stop(self, grace: float | None = 1.0) -> None: ...
    async def run(self, host: str = "127.0.0.1", port: int = 50051) -> None: ...
```

The companion server for `GRPCSink`. Receives proto `Event` messages
over the `GoldfiveIngress.StreamEvents` RPC (defined in
`proto/goldfive/v1/service.proto`) and fans them out to local
sinks. See [grpc-transport.md](../guides/grpc-transport.md).

## Live steering (`goldfive.control`)

Primitive for external pause / resume / cancel / steer / rewind /
approve / reject. Full design in
[../design/CONTROL.md](../design/CONTROL.md).

### `ControlKind`

```python
class ControlKind(StrEnum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    STEER = "STEER"          # payload: {"note": "...", "suggested_action": "..."}
    REWIND_TO = "REWIND_TO"  # payload: {"task_id": "..."}
    APPROVE = "APPROVE"      # payload: {"target_id": "...", "detail": "..."}
    REJECT = "REJECT"        # payload: {"target_id": "...", "detail": "..."}
```

`STATUS_QUERY` and `INTERCEPT_TRANSFER` are not in the Phase-1 enum
but are accepted as raw strings by the executor's dispatcher.

### `AckResult`

```python
class AckResult(StrEnum):
    SUCCESS
    FAILURE
    UNSUPPORTED
```

### `ControlMessage`

```python
@dataclass
class ControlMessage:
    kind: ControlKind
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: dict[str, Any] = field(default_factory=dict)
    issued_at_ms: int = 0
```

### `ControlAck`

```python
@dataclass
class ControlAck:
    control_id: str
    result: AckResult
    detail: str = ""
    acked_at_ms: int = 0
```

### `ControlChannel`

```python
class ControlChannel:
    def __init__(self) -> None: ...

    async def send(self, msg: ControlMessage) -> None: ...
    async def receive(self, timeout: float | None = None) -> ControlMessage | None: ...
    async def ack(self, ack: ControlAck) -> None: ...
    def acks(self) -> AsyncIterator[ControlAck]: ...
    def close(self) -> None: ...
```

Bidirectional: external callers push via `send` and iterate acks via
`acks()`; the runner consumes via `receive()` and publishes via
`ack()`. Dependency-light (`asyncio.Queue` under the hood) — tests
instantiate directly.

## Reporting (`goldfive.reporting`)

```python
@dataclass
class ReportingToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON schema
    handler: Callable[
        [dict[str, Any], Session, Steerer],
        Awaitable[dict[str, Any]],
    ]


# The seven canonical tool names (stable contract).
REPORTING_TOOL_NAMES: tuple[str, ...] = (
    "report_task_started",
    "report_task_progress",
    "report_task_completed",
    "report_task_failed",
    "report_task_blocked",
    "report_new_work_discovered",
    "report_plan_divergence",
)


# Pre-built list of the seven canonical specs. ``Runner`` registers
# these with the adapter automatically; custom adapters can consume
# the list directly.
BUILTIN_REPORTING_TOOLS: list[ReportingToolSpec]
```

See [tool-protocol.md](tool-protocol.md) for full semantics.

## Events (`goldfive.events`)

```python
def new_event(run_id: str, sequence: int) -> Any:
    """Fresh envelope with run_id, sequence, emitted_at populated."""

def make_event(
    *,
    run_id: str,
    sequence: int,
    kind: str,
    payload: dict[str, Any],
) -> Any:
    """Build an envelope from a ``kind`` + ``payload`` dict. Falls back
    to a plain dict when the ``proto`` extra is not installed."""

def now_ts() -> Any:
    """Current wall-clock time as a proto ``Timestamp``."""

async def emit(sinks: list[EventSink], event_pb: Any) -> None:
    """Fan out to every sink; swallow per-sink exceptions."""
```

Per-event builders (`run_started_event`, `run_completed_event`,
`run_aborted_event`, `goal_derived_event`, `plan_submitted_event`,
`plan_revised_event`, `task_started_event`, `task_progress_event`,
`task_completed_event`, `task_failed_event`, `task_blocked_event`,
`task_cancelled_event`, `drift_detected_event`) are also available in
`goldfive.events` as convenience constructors for sink implementations.

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

def to_pb_task(task: Task) -> pb.Task: ...
def from_pb_task(msg: pb.Task) -> Task: ...

def to_pb_task_edge(edge: TaskEdge) -> pb.TaskEdge: ...
def from_pb_task_edge(msg: pb.TaskEdge) -> TaskEdge: ...

def to_pb_drift_event(evt: DriftEvent) -> pb.DriftEvent: ...
def from_pb_drift_event(msg: pb.DriftEvent) -> DriftEvent: ...
```

## Recovery (`goldfive.sinks`)

```python
def replay_from_jsonl(path: str | Path) -> list[Any]:
    """Parse a JSONL log into ``Event`` proto messages, in emit order."""

def replay_from_sqlite(path: str | Path, run_id: str) -> list[Any]:
    """Read a single run's events from a SQLite database."""

def reconstruct_session(events: list[Any]) -> Session:
    """Replay events to rebuild a best-effort :class:`Session`."""
```

Used by `Runner.resume()`. See
[persistence-and-recovery.md](../guides/persistence-and-recovery.md).

## Drift classifiers (`goldfive.drift`)

For custom steerer subclasses that want to compose the standard
classifiers:

```python
def classify_tool_error(event: Any) -> Optional[DriftEvent]: ...
def classify_refusal(text: Any) -> Optional[DriftEvent]: ...
def classify_stop_reason(reason: Any) -> Optional[DriftEvent]: ...
```

Each returns `None` when the input does not match the classifier's
signal; callers compose them into a steerer's `detect_drift`
pipeline. The module also exposes the marker tables
`LLM_REFUSAL_MARKERS: tuple[str, ...]` and
`CONTEXT_PRESSURE_STOP_REASONS: tuple[str, ...]` for downstream reuse.

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
