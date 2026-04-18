# goldfive interface spec (v0.1)

This file pins the type and protocol contracts that parallel work-in-progress PRs must target. It is the single source of truth for inter-module shapes while PRs are being written in parallel. Once all foundational PRs land, this file may be deleted or moved into `docs/design/`.

## Dataclasses (from `goldfive/types.py` — issue #4)

```python
from __future__ import annotations
import dataclasses
from enum import StrEnum
from typing import Any, Callable, Mapping, Optional


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


class DriftKind(StrEnum):
    TOOL_ERROR = "tool_error"
    AGENT_REFUSAL = "agent_refusal"
    NEW_WORK_DISCOVERED = "new_work_discovered"
    PLAN_DIVERGENCE = "plan_divergence"
    USER_STEER = "user_steer"
    USER_CANCEL = "user_cancel"
    TASK_FAILED_RECOVERABLE = "task_failed_recoverable"
    TASK_FAILED_FATAL = "task_failed_fatal"
    CONTEXT_PRESSURE = "context_pressure"
    BLOCKED = "blocked"
    WRONG_AGENT = "wrong_agent"
    AGENT_TRANSFER = "agent_transfer"
    MODEL_REFUSAL = "model_refusal"
    STOPPED_EARLY = "stopped_early"
    TOO_MANY_STEPS = "too_many_steps"
    GOAL_UNREACHABLE = "goal_unreachable"
    TASK_TIMEOUT = "task_timeout"
    REPEATED_FAILURE = "repeated_failure"
    UNEXPECTED_OUTPUT = "unexpected_output"
    SCHEMA_VIOLATION = "schema_violation"
    HALLUCINATION_SUSPECTED = "hallucination_suspected"
    SAFETY_CONCERN = "safety_concern"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    CUSTOM = "custom"


class DriftSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclasses.dataclass
class Task:
    id: str
    title: str
    description: str = ""
    assignee_agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    predicted_start_ms: int = 0
    predicted_duration_ms: int = 0
    bound_span_id: str = ""


@dataclasses.dataclass
class TaskEdge:
    from_task_id: str
    to_task_id: str


@dataclasses.dataclass
class Plan:
    id: str
    run_id: str
    goal_ids: list[str]
    tasks: list[Task]
    edges: list[TaskEdge]
    summary: str = ""
    revision_reason: str = ""
    revision_kind: str = ""           # DriftKind value (str) or ""
    revision_severity: str = ""       # DriftSeverity value (str) or ""
    revision_index: int = 0

    def topological_stages(self) -> list[list[Task]]: ...


@dataclasses.dataclass
class Goal:
    id: str
    summary: str
    success_predicate: Optional[Callable[["Session"], bool]] = None
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DriftEvent:
    kind: DriftKind
    severity: DriftSeverity
    detail: str = ""
    current_task_id: str = ""
    current_agent_id: str = ""
    raw: Any = None   # original event that triggered detection


@dataclasses.dataclass
class Session:
    """Live state for one Runner.run() invocation."""
    run_id: str
    goals: list[Goal] = dataclasses.field(default_factory=list)
    plan: Optional[Plan] = None
    current_task_id: str = ""
    completed_results: dict[str, str] = dataclasses.field(default_factory=dict)
    # task_id -> progress fraction 0-1
    task_progress: dict[str, float] = dataclasses.field(default_factory=dict)
    # task_id -> last agent note
    agent_notes: dict[str, str] = dataclasses.field(default_factory=dict)
    divergence_flag: bool = False
    history: list[Any] = dataclasses.field(default_factory=list)
    started_at_ms: int = 0
    # monotonic event sequence counter for sinks
    _next_sequence: int = 0

    def next_sequence(self) -> int:
        s = self._next_sequence
        self._next_sequence = s + 1
        return s
```

## Reporting tools (from `goldfive/reporting.py` — issue #5 / #6)

```python
@dataclasses.dataclass
class ReportingToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON schema for parameters
    # handler invoked by the adapter when the agent calls the tool
    handler: Callable[[dict[str, Any], "Session", "Steerer"], Awaitable[dict[str, Any]]]
```

The seven canonical tools (names are stable contract — do not rename):

1. `report_task_started(task_id: str, detail: str = "")`
2. `report_task_progress(task_id: str, fraction: float = 0.0, detail: str = "")`
3. `report_task_completed(task_id: str, summary: str, artifacts: dict[str,str] | None = None)`
4. `report_task_failed(task_id: str, reason: str, recoverable: bool = True)`
5. `report_task_blocked(task_id: str, blocker: str, needed: str = "")`
6. `report_new_work_discovered(parent_task_id: str, title: str, description: str, assignee: str = "")`
7. `report_plan_divergence(note: str, suggested_action: str = "")`

## Results (from `goldfive/results.py` — issue #5)

```python
@dataclasses.dataclass
class InvocationResult:
    task_id: str
    text: str = ""             # final assistant text
    stop_reason: str = ""      # adapter-specific
    error: Optional[Exception] = None
    raw: Any = None


@dataclasses.dataclass
class ExecutionOutcome:
    success: bool
    session: Session
    reason: str = ""           # populated when success=False
```

## Protocols (from `goldfive/protocols.py` — issue #5)

All async. All Protocols use `@runtime_checkable`.

```python
from typing import Any, Awaitable, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class GoalDeriver(Protocol):
    async def derive(
        self,
        user_input: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> list[Goal]: ...


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

    # Called by executors to wire sinks/planner into the steerer.
    def bind(
        self,
        *,
        sinks: list["EventSink"],
        planner: Planner,
    ) -> None: ...


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

    # Optional: adapters may emit raw events (e.g., streamed message blocks)
    # into the executor-provided observation sink. Default returns None.
    @property
    def available_agents(self) -> list[str]: ...


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


@runtime_checkable
class EventSink(Protocol):
    async def emit(self, event_pb: Any) -> None: ...   # pb Event message
    async def close(self) -> None: ...
```

## Runner (from `goldfive/runner.py` — issue #15)

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
```

## Event emission convention

- Executors and Steerers emit proto `Event` messages via `sinks`.
- `Event.sequence` is assigned via `session.next_sequence()`.
- `Event.run_id` = `session.run_id`.
- `Event.emitted_at` = current `google.protobuf.Timestamp` (helper: `goldfive.events.now_ts()` — implement wherever convenient; document in issue #4 or #5).

## Event helpers (from `goldfive/events.py` — any PR may create; first creator owns it)

```python
def new_event(run_id: str, sequence: int) -> "EventPB": ...  # fresh envelope
def emit(sinks: list[EventSink], event_pb: "EventPB") -> Awaitable[None]: ...  # fan-out
```

## Conventions

- All public-facing API is async.
- No module in `goldfive/` may import `google.adk` or `claude_agent_sdk` except under `goldfive.adapters.adk` / `goldfive.adapters.claude`.
- Optional deps: `adk`, `claude`, `dev`, `examples`, `proto`.
- Python 3.11+.
- Line length 100.
- `from __future__ import annotations` at top of every file.

## Deleting this file

Once #2–#5 have merged and the contracts are live in code, delete this file. Until then, it's authoritative over any spec drift in individual PRs.
