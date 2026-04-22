# Public API surface

Hand-maintained reference for goldfive's public API. When
this doc disagrees with the code, trust the code and file a patch to
fix the doc.

Related: [PROTOCOLS.md](../design/PROTOCOLS.md),
[ARCHITECTURE.md](../design/ARCHITECTURE.md),
[DRIFT.md](../design/DRIFT.md).

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
    planner: Planner | None = None,
    goal_deriver: GoalDeriver | None = None,
    executor: Executor | None = None,
    steerer: Steerer | None = None,
    sinks: list[EventSink] | None = None,
    control: ControlChannel | None = None,
    call_llm: Callable[[str, str, str], Awaitable[str]] | None = None,
    model: str | None = None,
    max_task_invocations: int | None = None,
    plugins: list[Any] | None = None,
) -> Runner: ...


async def run(
    agent: Any,
    user_input: str | list[Goal],
    *,
    context: Mapping[str, Any] | None = None,
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

Default `executor` is `SequentialExecutor(overlay_mode=True,
max_task_invocations=max_task_invocations)`; callers who supply their
own `executor=` retain full control of the execution model.

When no `call_llm` is supplied and the agent does not expose an LLM
surface `wrap` can detect (currently only ADK), `wrap` falls back to
`PassthroughPlanner` + `LiteralGoalDeriver` and emits a `DEBUG`
log line on the `goldfive.wrap` logger.

`plugins=` is forwarded to `ADKAdapter(plugins=...)` and installed on
the one runner. ADK propagates the plugin manager into any
`AgentTool`-spawned sub-Runner so delegation inherits the same plugin
surface. Duplicate plugin instances (same `plugin.name`) are
silently deduped (#166/#169).

When the wrap target is an ADK `BaseAgent`, the returned object is a
`GoldfiveADKAgent` — a `BaseAgent` subclass that *also* exposes the
`Runner` surface, so the same call site works programmatically and as
the `root_agent` of an `adk web` app. `GoldfiveADKAgent` pins the
outer adk-web session id onto the inner `ADKAdapter` so all three
session layers align (#161/#164).

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
        max_task_invocations: Optional[int] = None,
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

    async def close(self) -> None: ...  # close every sink, then run hooks

    # Post-construction extension API. See "Extension API" below.
    def add_sink(self, sink: EventSink) -> None: ...
    def add_close_hook(
        self, hook: Callable[[], Awaitable[None]]
    ) -> None: ...

    @property
    def control(self) -> ControlChannel | None: ...
    @control.setter
    def control(self, value: ControlChannel) -> None: ...
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
| `control` | `None` | Optional `ControlChannel` for live pause / cancel / steer / rewind / approve / reject. When `None`, the run has no live-steering surface. May also be attached post-construction via the `control` setter; see "Extension API" below. See [../design/CONTROL.md](../design/CONTROL.md). |
| `max_task_invocations` | `None` (unbounded) | Optional cap on adapter invocations per run, stamped onto the planner context so executors that honour it can cap refine / task-invocation loops. Accepts the deprecated `max_plan_reinvocations` kwarg for one release with a `DeprecationWarning`. |

### Extension API

Three post-construction hooks let external integrations attach
additional behaviour without subclassing or attribute mutation:

- **`add_sink(sink)`** — append an `EventSink` after construction.
  Takes effect for events emitted by subsequent `run()` calls;
  in-flight runs keep the sink list they started with.
- **`add_close_hook(hook)`** — register an async callable invoked by
  `close()` *after* sinks are closed. Hooks fire in registration
  order; an exception in one hook is logged and does not prevent the
  rest from running. `close()` is idempotent — a second call is a
  no-op and hooks fire exactly once.
- **`control` setter** — attach a `ControlChannel` after construction.
  Idempotent on identity (`is`) re-attach; raises `RuntimeError` if a
  different channel is already attached. The constructor kwarg
  `control=...` is unchanged; the setter is the post-construction
  path.

`GoldfiveADKAgent` (returned by `goldfive.wrap(adk_agent)`) mirrors
the same contract and delegates to the inner `Runner`, so callers
can use the extension API uniformly whether they hold a plain
`Runner` or an ADK-wrapped one.

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
    NOT_NEEDED   # terminal; overlay-mode sweep at invocation end (#141/#163)
```

`TERMINAL_TASK_STATUSES = {COMPLETED, FAILED, CANCELLED, NOT_NEEDED}`.

### `DriftSeverity`

```python
class DriftSeverity(StrEnum):
    INFO
    WARNING
    CRITICAL
```

### `DriftKind`

Full taxonomy. Proto numbers are the authoritative wire values from
`proto/goldfive/v1/types.proto::DriftKind`. Python `StrEnum` values
use snake_case for forward compatibility with sinks.

| Proto # | Name | Python value | Meaning |
|---|---|---|---|
| 1 | TOOL_ERROR | `"tool_error"` | A tool invocation raised or returned a structured error. |
| 2 | AGENT_REFUSAL | `"agent_refusal"` | Agent declined to perform; graduated severity via `classify_refusal`. |
| 3 | NEW_WORK_DISCOVERED | `"new_work_discovered"` | `report_new_work_discovered` called; refine expected. |
| 4 | PLAN_DIVERGENCE | `"plan_divergence"` | `report_plan_divergence`; cross-layer `AgentTool` call from `GoldfivePlanner`. |
| 5 | USER_STEER | `"user_steer"` | `ControlKind.STEER` arrived; cascade-cancel + refine. |
| 6 | USER_CANCEL | `"user_cancel"` | `ControlKind.CANCEL` arrived. |
| 7 | TASK_FAILED_RECOVERABLE | `"task_failed_recoverable"` | `report_task_failed(recoverable=True)`; WARNING. |
| 8 | TASK_FAILED_FATAL | `"task_failed_fatal"` | `report_task_failed(recoverable=False)`; CRITICAL. |
| 9 | CONTEXT_PRESSURE | `"context_pressure"` | Stop-reason implied context / token pressure. |
| 10 | BLOCKED | `"blocked"` | `report_task_blocked`; awaits external input. |
| 11 | WRONG_AGENT | `"wrong_agent"` | Agent-transfer landed at the wrong target. |
| 12 | AGENT_TRANSFER | `"agent_transfer"` | Transfer observed; informational. |
| 13 | MODEL_REFUSAL | `"model_refusal"` | Provider-level refusal. |
| 14 | STOPPED_EARLY | `"stopped_early"` | Response truncated before the task completed. |
| 15 | TOO_MANY_STEPS | `"too_many_steps"` | Per-task / per-lineage step cap exceeded. |
| 16 | GOAL_UNREACHABLE | `"goal_unreachable"` | Planner concluded no plan can satisfy the goal. |
| 17 | TASK_TIMEOUT | `"task_timeout"` | Task exceeded `predicted_duration_ms` by threshold. |
| 18 | REPEATED_FAILURE | `"repeated_failure"` | N consecutive refine failures for the same `(kind, task)` pair. |
| 19 | UNEXPECTED_OUTPUT | `"unexpected_output"` | Output shape violates declared schema. |
| 20 | SCHEMA_VIOLATION | `"schema_violation"` | Hard JSON / schema parse failure. |
| 21 | HALLUCINATION_SUSPECTED | `"hallucination_suspected"` | Content inconsistent with the session's facts. |
| 22 | SAFETY_CONCERN | `"safety_concern"` | Policy / safety signal. |
| 23 | RESOURCE_EXHAUSTED | `"resource_exhausted"` | Rate limits, quota, etc. |
| 24 | AMBIGUOUS_INTENT | `"ambiguous_intent"` | Signals the planner needs clarification. |
| 25 | CUSTOM | `"custom"` | Escape hatch paired with `DriftDetected.detail`. |
| 26 | LOOPING_TOOL_CALL | `"looping_tool_call"` | Reporting-tool loop guard tripped. |
| 27 | LOOPING_REASONING | `"looping_reasoning"` | Reasoning-content loop (hash/embedding) **or** tool-loop detector (#181). |
| 28 | CONFUSION | `"confusion"` | Reasoning expresses uncertainty; INFO. |
| 29 | OFF_TOPIC | `"off_topic"` | Reasoning topic distant from task description (embedding). |
| 30 | INTENT_DIVERGENCE | `"intent_divergence"` | Reasoning mentions a non-session goal; graduated severity. |
| 31 | UNCERTAIN_PROGRESS | `"uncertain_progress"` | Opt-in reflective check: yes-but-low-confidence. |
| 32 | SELF_REPORTED_STUCK | `"self_reported_stuck"` | Opt-in reflective check: agent says no progress. |
| 33 | REASONING_CLUSTER_TIGHTENING | `"reasoning_cluster_tightening"` | Embedding-only early-warning below the LOOPING_REASONING cliff. |
| 34 | CONFABULATION_RISK | `"confabulation_risk"` | Task implies external data but no tool was called; hallucinated `function_call` name (from `GoldfivePlanner`). |
| 35 | RUNAWAY_DELEGATION | `"runaway_delegation"` | `ADKAdapter(agent_tool_cap=N)` exceeded; CRITICAL. |
| 36 | REFINE_VALIDATION_FAILED | `"refine_validation_failed"` | `LLMPlanner.refine` exhausted retries; CRITICAL terminal signal. |
| 37 | HUMAN_INTERVENTION_REQUIRED | `"human_intervention_required"` | Ladder Level 4 escalation; CRITICAL. |
| 38 | GOAL_DRIFT | `"goal_drift"` | Periodic trajectory-level goal-alignment check; CRITICAL. |

`DriftKind.USER_PAUSE` also exists on the Python side (no proto
member) for in-process PAUSE bookkeeping.

See [DRIFT.md](../design/DRIFT.md) for severity bands, classifier
pipelines, and the intervention ladder's level-mapping table.

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

Abbreviated shape (see `goldfive.types.Session` for the full
dataclass including reflective-check counters and ladder-handoff
slots):

```python
@dataclass
class Session:
    run_id: str
    conversation_id: str = ""
    goals: list[Goal] = field(default_factory=list)
    plan: Plan | None = None
    current_task_id: str = ""
    completed_results: dict[str, str] = field(default_factory=dict)
    task_progress: dict[str, float] = field(default_factory=dict)
    agent_notes: dict[str, str] = field(default_factory=dict)
    divergence_flag: bool = False
    history: list[Any] = field(default_factory=list)
    started_at_ms: int = 0

    # Reasoning-drift pipeline (goldfive#96)
    reasoning_history: list[str] = field(default_factory=list)
    reasoning_history_max: int = 20

    # Intervention-ladder handoffs (goldfive#142)
    paused_for_human_intervention: bool = False
    pending_nudges: list[str] = field(default_factory=list)
    pending_corrective_message: str | None = None

    # Goldfive-orchestration session state (goldfive#152). Owned key
    # names live in ``goldfive.orchestration_state``; see below.
    state: dict[str, Any] = field(default_factory=dict)

    def next_sequence(self) -> int:
        """Monotonic event sequence counter."""

    @property
    def id(self) -> str:
        """Alias for ``run_id``; used as ``Event.session_id`` (goldfive#155)."""
```

`goldfive.orchestration_state` owns these keys under `goldfive.*`:

| Key | Written by | Read by |
|---|---|---|
| `goldfive.current_plan_id` | plan-submitted / plan-revised paths | planners, sinks |
| `goldfive.current_task_id` | `PlanReconciler` on RUNNING | `GoldfivePlanner`, sinks |
| `goldfive.current_task_title` | same | same |
| `goldfive.goals_summary` | Runner (on goals change / USER_STEER) | `GoldfivePlanner` |
| `goldfive.active_steer.body` | `DefaultSteerer` on USER_STEER | `GoldfivePlanner` |
| `goldfive.active_steer.at_turn` | same | refine / drift |
| `goldfive.active_steer.author` | same | sinks |
| `goldfive.processed_steer_ids` | `DefaultSteerer.observe` dedupe | self |
| `goldfive.cancelled_function_call_ids` | adapter `_heal_pending_tool_calls` | `GoldfivePlanner` response filter |

`_GoldfiveADKPlugin.before_run_callback` bridges a subset of these
from `goldfive.Session.state` onto the live ADK `session.state` so
`GoldfivePlanner` sees them on its request-side read
(goldfive#170/#173).

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
| `Planner` | `generate(*, goals, available_agents, context=None) -> Plan \| None` · `refine(*, plan, drift, goals, observed_actions=None, available_agents=None) -> Plan \| None` |
| `Steerer` | `observe(event, session)` · `transition(task_id, to, *, detail="", session)` · `detect_drift(event, session) -> DriftEvent \| None` (sync) · `bind(*, sinks, planner)` (sync) |
| `AgentAdapter` | `register_reporting_tools(tools)` · `invoke(task, session) -> InvocationResult` · `emit_reasoning(text, *, task=None, session, provider="", call_id="")` · property `available_agents: list[str]` (sync) |
| `Executor` | `run(*, plan, session, adapter, steerer, planner, sinks, control=None, user_input="")` — overlay-mode executors honour `user_input`; legacy per-task executors ignore it. |
| `EventSink` | `emit(event_pb)` · `close()` |

Overlay-specific adapter methods are **duck-typed**, not in the
`AgentAdapter` protocol: `invoke_passthrough(user_message, *, session,
reconciler=None, ctx=None)` and `invoke_follow_up(task, session)` are
defined on `ADKAdapter`. Custom adapters that want to participate in
the overlay execution path implement them; callers look them up via
`getattr` and fall back to `invoke` when absent. Similarly,
`available_agents_tree` (goldfive#151) is a duck-typed property
shipped on `ADKAdapter` / `CallableAdapter` / `ClaudeAgentSDKAdapter`
but not part of the Protocol contract.

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
    DEFAULT_MAX_REFINE_ATTEMPTS: int = 2

    def __init__(
        self,
        *,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str = "",
        system_prompt: str | None = None,
        refine_system_prompt: str | None = None,
        user_steer_system_prompt: str | None = None,
        looping_tool_call_system_prompt: str | None = None,
        plan_divergence_system_prompt: str | None = None,
        max_refine_attempts: int | None = None,
    ) -> None: ...

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None: ...

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: list[ObservedAction] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None: ...
```

`available_agents` may be a plain `list[str]` (legacy callers) or the
structured walker produced by
`ADKAdapter.available_agents_tree` (goldfive#151). When the tree form
is supplied the prompt renders an `AGENT TREE` section and the
validator rejects any task whose `assignee_agent_id` is not in the
registry, feeding the validator message back into a
retry-with-correction loop. An empty / `None` registry skips the
assignee check for back-compat.

`refine` behaviour by drift kind:

| `drift.kind` | Prompt used | Uses `observed_actions`? | Uses `available_agents`? |
|---|---|---|---|
| `USER_STEER` | user-steer system prompt | no | yes |
| `LOOPING_TOOL_CALL` / `LOOPING_REASONING` | looping-tool-call prompt | no | yes |
| `PLAN_DIVERGENCE` with `observed_actions` | divergence / reconciler prompt | yes — ABSORB or `{"reject": true, ...}` | yes |
| `REFINE_VALIDATION_FAILED` | — | — | — (returns `None`; terminal) |
| everything else | generic refine prompt | no | yes |

Goal-aware refine (#154): `goals` are included in the divergence
prompt; USER_STEER-sourced goals render with `[STICKY]`, and the
validator rejects revisions that silently drop them.

### Steerer (`goldfive.steerer`)

```python
class DefaultSteerer(Steerer):
    # Implements the full state machine + drift classifier from
    # DRIFT.md and STATE-MACHINE.md plus the six-level intervention
    # ladder. No required constructor args.
```

In addition to the `Steerer` protocol methods, `DefaultSteerer`
exposes `mark_task_running`, `mark_task_progress`,
`mark_task_completed`, `mark_task_failed`, `mark_task_blocked`,
`mark_task_cancelled`, `report_new_work_discovered`, and
`report_plan_divergence` for the canonical reporting-tool handlers to
call into.

STEER idempotency (goldfive#171): `observe(event, session)` dedupes
`ControlMessage`s of kind `STEER` by source annotation id. The
dedupe key is taken from the `ControlMessage.payload["annotation_id"]`
when the bridge supplied one; otherwise the `ControlMessage.id` is
used as the fallback. Processed ids are stored in
`session.state[goldfive.processed_steer_ids]` with FIFO eviction at
`DefaultSteerer.PROCESSED_STEER_IDS_CAP`. Content-based drifts
(`LOOPING_REASONING`, tool errors, etc.) are **not** deduped — they
are heuristic signals, not user actions.

Intervention ladder (goldfive#142):

| Level | Name | Action |
|---|---|---|
| 0 | OBSERVE | Emit `DriftDetected`; no further action. |
| 1 | ABSORB | Call `planner.refine`; continue. |
| 2 | NUDGE | Queue a soft follow-up on `session.pending_nudges`; overlay loop picks it up at the next invocation boundary. |
| 3 | CANCEL_REINVOKE | Cancel in-flight, refine, stash a corrective message on `session.pending_corrective_message`. |
| 4 | PAUSE_ESCALATE | Emit `HUMAN_INTERVENTION_REQUIRED`; set `session.paused_for_human_intervention`. |
| 5 | TERMINATE | Run-level abort (reserved for unhandled Level 4 timeouts). |

Mapping from `(drift_kind, severity, occurrence_count)` to level lives
in `DefaultSteerer._ladder_level_for`.

### `GoldfivePlanner` (`goldfive.planners.goldfive_planner`)

ADK `BasePlanner` subclass auto-attached by `goldfive.wrap` to every
`LlmAgent` in the tree (goldfive#153/#156). Two jobs:

1. **Request side** — `build_planning_instruction` returns a
   tree-agnostic `[GOLDFIVE ORCHESTRATION CONTEXT]` block assembled
   from `session.state[goldfive.*]` keys. Request-side injection is
   performed by `_GoldfiveADKPlugin.before_model_callback`
   (workaround for ADK's `isinstance(planner, PlanReActPlanner)` gate
   in `_nl_planning.py`).
2. **Response side** — `process_planning_response` filters response
   parts:
   - strips `function_call` parts whose id is in
     `session.state[goldfive.cancelled_function_call_ids]`
   - classifies each remaining `function_call` via a three-stage
     gate (goldfive#178/#184): own-tool → skip; cross-layer agent
     name in the tree registry → `PLAN_DIVERGENCE` (WARNING);
     hallucinated name → `CONFABULATION_RISK` (WARNING). Calls are
     **never blocked**.

```python
class GoldfivePlanner(BasePlanner):
    def __init__(
        self,
        *,
        user_planner: BasePlanner | None = None,
        agent_registry: Iterable[str] | None = None,
        steerer: Any = None,
        session: Any = None,
    ) -> None: ...

    def bind(
        self,
        *,
        agent_registry: Iterable[str] | None = None,
        steerer: Any = None,
        session: Any = None,
    ) -> None: ...

    def build_planning_instruction(
        self, readonly_context, llm_request
    ) -> str | None: ...

    def process_planning_response(
        self, callback_context, response_parts
    ) -> list[Part] | None: ...
```

Composes with a user-supplied `BasePlanner` via `user_planner=`: the
user's `build_planning_instruction` is called first and **prepended**
ahead of goldfive's block; on the response side goldfive's filters
run first and the cleaned parts flow through the user planner's
`process_planning_response`.

### `ToolLoopTracker` (`goldfive.drift.tool_loops`)

Tool-call loop detector (goldfive#181/#186) plumbed on every
`after_tool_callback` dispatch in `_GoldfiveADKPlugin`. Per-
`(invocation_id, agent_name)` ring buffer; emits
`DriftEvent(kind=LOOPING_REASONING, ...)` on three patterns.

```python
class ToolLoopTracker:
    def __init__(
        self,
        *,
        window: int = 7,                   # DEFAULT_WINDOW
        exact_threshold: int = 3,          # DEFAULT_EXACT_THRESHOLD
        name_threshold: int = 5,           # DEFAULT_NAME_THRESHOLD
        alternating_threshold: int = 5,    # DEFAULT_ALTERNATING_THRESHOLD
    ) -> None: ...

    def observe_tool_call(
        self,
        *,
        invocation_id: str,
        agent_name: str,
        tool_name: str,
        args: Any,
        task_id: str = "",
    ) -> list[DriftEvent]: ...

    def on_task_progress(
        self, *, invocation_id: str, agent_name: str
    ) -> None: ...

    def clear(self) -> None: ...

    def buffer_size(self, *, invocation_id: str, agent_name: str) -> int: ...


def args_hash(args: Any) -> str: ...          # 8-char md5 hex of sorted-keys JSON
def load_thresholds_from_env() -> dict[str, int]: ...
```

Detection modes:

| Mode | Pattern | Default | Severity |
|---|---|---|---|
| Exact | same `(tool_name, args_hash)` ≥ threshold in last `window` | 3 / 7 | WARNING |
| Name | same `tool_name` ≥ threshold in last `window`, no task progress | 5 / 7 | WARNING |
| Alternating | A,B,A,B,A pattern in last `alternating_threshold` | 5 | INFO |

Progress-reporting tools (`report_task_*`) call
`on_task_progress(...)` which clears the per-(invocation, agent)
window, so legitimate scripted sequences aren't flagged.

Env-var overrides:
`GOLDFIVE_TOOL_LOOP_WINDOW`,
`GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD`,
`GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD`,
`GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD`.

Follow-ups tracked: #179 (umbrella), #182 (args-quality),
#183 (silent-success), #185 (wrong-tool).

### Executors (`goldfive.executors`)

```python
class SequentialExecutor(Executor):
    def __init__(
        self,
        *,
        max_task_invocations: int | None = None,       # None = unbounded
        max_retries_per_task_lineage: int = 3,
        fail_fast: bool = True,
        overlay_mode: bool = False,                     # goldfive#141
    ) -> None: ...

class ParallelDAGExecutor(Executor):
    def __init__(
        self,
        max_concurrency: int = 0,  # 0 = unbounded fan-out
        drift_policy: Literal["cancel_stage", "finish_stage"] = "finish_stage",
        max_task_invocations: int | None = None,  # None = unbounded
    ) -> None: ...
```

`overlay_mode=True` (set by `goldfive.wrap` by default) swaps the
executor's per-task loop for a single
`adapter.invoke_passthrough(user_input)` invocation. Observation
happens via the plugin callback surface and `PlanReconciler` maps
observed agent turns to plan-task transitions. STEER control
messages cancel the in-flight invocation and restart `invoke_passthrough`
with the composed steer-restart message as the new user input. At
invocation end any task still PENDING lands in
`TaskStatus.NOT_NEEDED` (no soft follow-up, goldfive#163).

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
        session_id: str | None = None,
        app_name: str | None = None,
        plugins: list[Any] | None = None,
        agent_tool_cap: int | None = None,  # default 16; 0 disables
    ) -> None: ...

    # Overlay-model entry points (goldfive#141). `goldfive.wrap` uses
    # ``invoke_passthrough`` exclusively; ``invoke`` and
    # ``invoke_follow_up`` are kept for external callers.

    async def invoke_passthrough(
        self,
        user_message: str,
        *,
        session: Session,
        reconciler: Any = None,
        ctx: Any = None,
    ) -> InvocationResult:
        """Drive ONE ADK turn with the user's original request verbatim."""

    async def invoke_follow_up(
        self, task: Task, session: Session
    ) -> InvocationResult:
        """Gentle ``Also, please: {title}.`` for a missed task.

        Not called by the overlay executor since goldfive#163 —
        PENDING tasks land in ``TaskStatus.NOT_NEEDED`` at invocation
        end. Retained for external callers.
        """

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        """DEPRECATED — per-task drive. Uses ``invoke_follow_up`` phrasing."""

    def add_plugin(self, plugin: Any) -> None: ...

    @property
    def available_agents(self) -> list[str]:
        """Sorted names of every agent reachable from the wrap target.

        Walks ``sub_agents`` / ``inner_agent`` / ``AgentTool.agent``
        edges at ``__init__`` time. Advisory for the planner, which
        populates ``task.assignee_agent_id`` as a delegation hint;
        goldfive does not route on the assignee under the single-
        Runner model (goldfive#130).
        """

    @property
    def available_agents_tree(self) -> list[dict[str, Any]]:
        """Structured walker of the tree: one dict per reachable agent
        with ``name`` / ``depth`` / ``parent`` / ``role`` / ``kind``.
        Passed to ``LLMPlanner.generate(available_agents=...)`` so the
        prompt and validator can enforce on-registry assignees
        (goldfive#151).
        """

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

`ADKAdapter.invoke_passthrough(user_message, session=..., reconciler=...)`
drives the one runner regardless of `task.assignee_agent_id` — the
assignee is a planner hint, not a routing key. The plugin enforces a
per-invocation cap on AgentTool spawns (default 16,
`agent_tool_cap=0` disables); on exceed the plugin emits a
`RUNAWAY_DELEGATION` drift at CRITICAL severity and cancels the
invocation.

When the caller or the executor cancels an in-flight invocation,
`ADKAdapter` invokes `plugin.on_cancellation(invocation_id)` on every
plugin that defines the method (goldfive#167/#168). Observability
plugins like `HarmonografTelemetryPlugin` use this to close open
spans with `status=CANCELLED` before the `CancelledError` is
re-raised. Exceptions in the hook are swallowed — cancel semantics
take precedence.

See [ARCHITECTURE.md §"Single-Runner dispatch"](../design/ARCHITECTURE.md#single-runner-dispatch-goldfive-drives-the-root-adk-delegates-within)
for the model and [adk-web-integration.md §"Pre-built Runner degrade mode"](../guides/adk-web-integration.md#pre-built-runner-degrade-mode)
for the degraded-mode contract when the caller passes a
pre-built `Runner` instead of a `BaseAgent`.

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

### Agent-invocation events (ADK single-Runner + AgentTool delegation)

Three observability-only events emitted by the goldfive ADK plugin
on every dispatch. They describe the "who actually ran what"
shape of a run — both goldfive's top-level dispatch and any
AgentTool-spawned sub-Runner invocations. See
[EVENT-MODEL.md §"Agent-invocation events"](../design/EVENT-MODEL.md#agent-invocation-events)
for the nesting rules and consumer guidance.

```python
def agent_invocation_started_event(
    run_id: str,
    sequence: int,
    *,
    agent_name: str,
    task_id: str = "",
    invocation_id: str = "",
    parent_invocation_id: str = "",
    started_at: Any | None = None,
) -> Any: ...

def agent_invocation_completed_event(
    run_id: str,
    sequence: int,
    *,
    agent_name: str,
    task_id: str = "",
    invocation_id: str = "",
    summary: str = "",
    completed_at: Any | None = None,
) -> Any: ...

def delegation_observed_event(
    run_id: str,
    sequence: int,
    *,
    from_agent: str,
    to_agent: str,
    task_id: str = "",
    invocation_id: str = "",
    observed_at: Any | None = None,
) -> Any: ...
```

`AgentInvocationStarted` payload fields:

| Field | Type | Meaning |
|---|---|---|
| `agent_name` | `string` | Dispatched agent. `task.assignee_agent_id` for top-level; wrapped agent's name for AgentTool sub-Runners. |
| `task_id` | `string` | Goldfive-dispatched task id. Propagates into nested invocations. |
| `invocation_id` | `string` | ADK's per-run invocation id. |
| `parent_invocation_id` | `string` | Empty on top-level; set to outer `invocation_id` on AgentTool sub-Runners. |
| `started_at` | `Timestamp` | Emission time. |

`AgentInvocationCompleted` payload fields:

| Field | Type | Meaning |
|---|---|---|
| `agent_name` | `string` | Matches the corresponding Started event. |
| `task_id` | `string` | Same as the Started event. |
| `invocation_id` | `string` | Matches the Started event. |
| `summary` | `string` | Optional outcome summary (final assistant text). |
| `completed_at` | `Timestamp` | Emission time. |

`DelegationObserved` payload fields:

| Field | Type | Meaning |
|---|---|---|
| `from_agent` | `string` | Host agent about to call the AgentTool. |
| `to_agent` | `string` | Wrapped agent the AgentTool will invoke. |
| `task_id` | `string` | Goldfive-dispatched task id. |
| `invocation_id` | `string` | Host agent's invocation id. |
| `observed_at` | `Timestamp` | Emission time. |

## Proto (`goldfive.pb.goldfive.v1`)

Generated stubs committed to git. Key messages:

```python
# types_pb2
Goal, Plan, Task, TaskEdge, TaskStatus, DriftKind, PlanRevision

# events_pb2
Event, RunStarted, GoalDerived, PlanSubmitted, PlanRevised,
TaskStarted, TaskProgress, TaskCompleted, TaskFailed,
TaskBlocked, TaskCancelled, DriftDetected, RunCompleted, RunAborted,
AgentInvocationStarted, AgentInvocationCompleted, DelegationObserved

# control_pb2
ControlEvent, ControlAck, ControlKind, ControlAckResult, ControlTarget,
SteerPayload, RewindPayload, ApprovePayload, RejectPayload,
InjectMessagePayload
```

The `Event` envelope has a `oneof payload` with one field per
per-event message. See [EVENT-MODEL.md](../design/EVENT-MODEL.md) for
the full catalog.

Envelope fields worth knowing about:

| # | Field | Added | Purpose |
|---|---|---|---|
| 1 | `event_id` | v0.1 | UUIDv7 recommended for sink dedupe. |
| 2 | `run_id` | v0.1 | Stable across every event in a run. |
| 3 | `sequence` | v0.1 | Per-run monotonic, gap-free starting at 0. |
| 4 | `emitted_at` | v0.1 | Wall clock; advisory only. |
| 5 | `session_id` | goldfive#155/#157 | Per-event `Session.id` for stream-multiplexed consumers. Empty means "route via stream Hello". |

Drift-side augmentations:

- `DriftDetected.annotation_id` (field 6, goldfive#176/#177) — source
  annotation id for USER_STEER / USER_CANCEL drifts minted from a
  `ControlMessage`; empty for goldfive-minted drifts. Sinks dedupe
  against the annotation card.

Control-side augmentations (`SteerPayload`, goldfive#171/#175):

| # | Field | Purpose |
|---|---|---|
| 1 | `note` | Steer body text. |
| 2 | `suggested_action` | Optional hint to the planner. |
| 3 | `author` | Operator identity from the originating annotation; empty when the bridge doesn't source annotations. |
| 4 | `annotation_id` | Source annotation id used for idempotent delivery in `DefaultSteerer.observe`. |

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
pipeline.

`classify_refusal` grades severity by matching tier:

| Tier       | Tuple                            | Severity    | Example marker          |
| ---------- | -------------------------------- | ----------- | ----------------------- |
| Policy     | `LLM_REFUSAL_MARKERS_CRITICAL`   | `CRITICAL`  | `"i must decline"`      |
| Capability | `LLM_REFUSAL_MARKERS_WARNING`    | `WARNING`   | `"i cannot"`            |
| Hedging    | `LLM_REFUSAL_MARKERS_INFO`       | `INFO`      | `"i'm not confident"`   |

The scan order is `CRITICAL -> WARNING -> INFO`; first match wins, so
a policy/safety refusal is never downgraded when the same text also
contains a capability or hedging marker. `DefaultSteerer` only
triggers `planner.refine` for `WARNING`/`CRITICAL` drift; INFO
matches are emitted to sinks for observability only.

The module also exposes the context-pressure table
`CONTEXT_PRESSURE_STOP_REASONS: tuple[str, ...]` for downstream
reuse. The flat `LLM_REFUSAL_MARKERS` tuple is retained as the
concatenation of the three tiered tuples for back-compat; new code
should import the tiered names directly.

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
