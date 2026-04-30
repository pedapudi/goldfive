# The Runner and the Session

`Runner` and `Session` are the two objects every goldfive run touches. Everything else — planners, steerers, sinks, adapters, reconcilers — is composed into the Runner; everything stateful observed during the run lives on the Session. This guide is the narrative tour; see `docs/design/ARCHITECTURE.md` for the rationale and `docs/reference/api.md` for the formal signatures.

## Runner: six pluggable components, one loop

```
┌─────────────────────── Runner ──────────────────────────────┐
│                                                              │
│   GoalDeriver  ──▶  goals:list[Goal]                         │
│       │                                                      │
│       ▼                                                      │
│   Planner     ──▶  plan:Plan   (tasks + edges + revision)    │
│       │                                                      │
│       ▼                                                      │
│   Executor   ◀──▶  AgentAdapter ◀──▶  agent framework        │
│       │               (ADK / Claude / callable)              │
│       ▼                                                      │
│   Steerer   ──▶  DriftDetected + intervention-ladder action  │
│       │                                                      │
│       ▼                                                      │
│   EventSink[] ──▶  harmonograf / JSONL / custom              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

The Runner itself is small — it sequences the six primitives and owns a Session. All the per-framework code lives in the `AgentAdapter`; all the drift policy lives in the `Steerer`; all the persistence lives in the `EventSink`.

### Constructing one

For a custom tree you usually don't hand-wire this — use `goldfive.wrap(tree)` for ADK or `goldfive.quickstart(call_llm=..., goal="...")` for a plain callable. But the hand-wired form makes the composition explicit:

```python
from goldfive import Runner
from goldfive.adapters.adk import ADKAdapter
from goldfive.planner import LLMPlanner
from goldfive.executors.sequential import SequentialExecutor
from goldfive.steerer import DefaultSteerer
from goldfive.sinks.harmonograf import HarmonografSink

runner = Runner(
    agent=ADKAdapter(root_agent=tree),
    planner=LLMPlanner(call_llm=call_llm),
    executor=SequentialExecutor(),
    steerer=DefaultSteerer(),
    sinks=[HarmonografSink(client=harmo_client)],
)
```

Defaults cover the common cases:

- `goal_deriver` defaults to `PassthroughGoalDeriver("run")` (a single `Goal(id="g1", summary="run")`).
- `steerer` defaults to `DefaultSteerer()` with the Level 0-5 intervention ladder.
- `sinks` defaults to `[]`.
- `control` is optional — attach a `ControlChannel` when you want live STEER / CANCEL / PAUSE from an external controller (harmonograf UI, CLI, tests).

### `wrap()` returns a Runner

```python
wrapped = goldfive.wrap(root_agent, sinks=[HarmonografSink(...)])
```

`wrap` is a one-call factory that auto-detects the framework, picks an adapter, registers the canonical reporting tools on sub-agents, auto-attaches a `GoldfivePlanner(BasePlanner)` for per-turn structural steering, installs the harmonograf telemetry plugin once (deduping by name), and returns a Runner. When the returned object is a `BaseAgent` (ADK mode), that agent is actually a `GoldfiveADKAgent` that wraps the Runner — every adk-web invocation goes through `Runner.run` transparently.

## Runner.run lifecycle

One call to `Runner.run(user_input, session_id=...)` does this, in order, emitting events as it goes:

1. **ConversationStarted** (once per Conversation; piggy-backs on the first Session).
2. **RunStarted** (owned by Runner).
3. **GoalDerived** (owned by Runner) — `goal_deriver.derive(user_input)` produces `list[Goal]`.
4. **PlanSubmitted** (owned by Runner) — `planner.generate(goals, available_agents=...)` produces a `Plan`. The `available_agents` structured walker comes from `ADKAdapter.available_agents_tree` and lets the validator reject off-registry assignees.
5. **Reporting-tool registration** — the Runner stamps the eight `report_task_*` tools onto every sub-agent in the tree so they can self-report progress. Tool-loop detection exempts these.
6. **Executor.run** takes over — it walks the plan, delegates to the adapter, observes events, and forwards drift observations to the steerer. For ADK trees under the overlay model, this is a single invocation that lets the tree run naturally; the reconciler maps observations back onto plan tasks.
7. During execution: **Task\***, **PlanRevised**, **DriftDetected** (owned by Executor / Steerer).
8. Terminal: **RunCompleted** or **RunAborted** (owned by Executor).

The Runner owns `Run*` lifecycle events; the Executor owns task events, plan revisions, and the terminal event; the Steerer owns drift events. Ownership matters for replay — when reading a JSONL persistence log you can reconstruct what each component was told.

The call returns an `ExecutionOutcome`:

```python
@dataclass
class ExecutionOutcome:
    success: bool
    session: Session       # the final live Session — inspect after run
    reason: str | None     # set when success is False
```

## Session: live state for one Runner.run

The Session is created at the top of `Runner.run` and returned in the outcome. It carries every piece of mutable state the run touches. Key fields:

```python
@dataclass
class Session:
    run_id: str                          # matches the adk-web session id post-unification
    conversation_id: str = ""            # ties turns in a multi-turn Conversation
    goals: list[Goal]
    plan: Plan | None
    current_task_id: str
    completed_results: dict[str, str]
    task_progress: dict[str, float]
    agent_notes: dict[str, str]
    history: list[Any]
    started_at_ms: int
    pending_approvals: dict[str, asyncio.Event]
    pending_approvals_meta: dict[str, dict[str, Any]]
    reasoning_history: list[str]              # bounded ring for reasoning-drift detectors
    refine_outcomes: dict[tuple[str, str], RefineOutcome]   # per-(kind, task) refine outcome (#215 P2)
    # Intervention-ladder handoffs (goldfive#142):
    paused_for_human_intervention: bool       # Level 4 handoff
    pending_nudges: list[str]                 # Level 2 soft-follow-up queue
    pending_corrective_message: str | None    # Level 3 cancel-reinvoke slot
    # Orchestration-level state dict (see below):
    state: dict[str, Any]
```

### `Session.id` property

Post session unification (PR #164), `Session.id` is a property that returns `run_id`. Three session layers collapse to one id when wrapped in adk-web:

1. `adk_web.Session.id` — adk-web's session id (the URL fragment).
2. `goldfive.Session.id` — pinned to the adk-web id by `GoldfiveADKAgent`.
3. `harmonograf.Session.id` — pinned by lazy Hello (harmonograf#85) so first emit stamps the same id server-side.

All goldfive events stamp `Event.session_id` (proto field 5) with this id, and HarmonografSink routes them accordingly.

### `Session.state`: the orchestration-level dict

`Session.state` is goldfive's private dict for coordination between its own components. It is NOT the same surface as the ADK `session.state` dict that agents read through their system-instruction context block.

Goldfive owns keys under the `goldfive.*` namespace. See `goldfive.orchestration_state` for the documented key names and helper functions. A selection:

| Key | Written by | Read by |
|---|---|---|
| `goldfive.current_task_id` | PlanReconciler (on observation) | planners / tools / docs |
| `goldfive.current_task_title` | PlanReconciler | GoldfivePlanner (renders into orchestration block) |
| `goldfive.goals_summary` | Runner (on goal derivation / USER_STEER) | GoldfivePlanner |
| `goldfive.active_steer.body` | DefaultSteerer (on STEER observe) | GoldfivePlanner |
| `goldfive.active_steer.author` | DefaultSteerer | GoldfivePlanner |
| `goldfive.active_steer.at_turn` | DefaultSteerer | GoldfivePlanner |
| `goldfive.cancelled_function_call_ids` | ADKAdapter (on cancel) | GoldfivePlanner (strips cancelled calls from next response) |
| `goldfive.processed_steer_ids` | DefaultSteerer (idempotency bookkeeping) | DefaultSteerer |

The ADK side of the bridge is implemented in `goldfive.adapters._adk_state_protocol`. On every ADK `before_run_callback`, the `_GoldfiveADKPlugin` mirrors the relevant `Session.state` keys onto the live ADK `InvocationContext.session.state` so agents see them through the `GoldfivePlanner`-built orchestration block. Without this bridge, the block would always render `(none)` even when a steer is active (fixed in PR #173).

### Intervention-ladder handoff slots

The Steerer doesn't directly reach into the executor's control loop; it writes to three Session fields and the executor reads them:

- `pending_nudges: list[str]` — Level 2 (NUDGE). The overlay loop pops from the front on each invocation end and dispatches as a soft follow-up user message.
- `pending_corrective_message: str | None` — Level 3 (CANCEL + re-invoke). A single slot; a second Level 3 overwrites the first.
- `paused_for_human_intervention: bool` — Level 4 (pause + escalate). The executor blocks in the pre-task loop until a CONTROL_RESUME or CONTROL_STEER arrives.

Level 0 (observe) and Level 1 (absorb into plan revision) don't need Session handoff — the steerer just emits events or calls `planner.refine` directly. Level 5 (terminate) trips a `run_aborted_event`.

## Sub-Runners, AgentTool, and session propagation

When an ADK tree uses `AgentTool(sub_agent)`, each sub-agent delegation spawns its own `InMemoryRunner` with its own `InvocationContext.session`. Three things matter:

1. **Plugin propagation** (goldfive#122). `_GoldfiveADKPlugin` and `HarmonografTelemetryPlugin` are registered on the sub-Runner via `_register_plugin_on_runner`, so every delegated agent emits spans and events through the same sinks.

2. **Session id propagation** (goldfive#164). The sub-Runner's session gets a fresh ADK id by default, but goldfive's plugin stamps the outer `Session.run_id` on every event's `session_id` field 5. HarmonografSink routes by that field — so cross-runner spans roll up onto the one canonical session.

3. **Per-agent id attribution** (harmonograf#80). The client's telemetry plugin stacks `per_agent_id = f"{client_id}:{ctx.agent.name}"` via before/after_agent callbacks. Every span emitted during that agent's execution carries its per-agent id regardless of which sub-Runner is running. Harmonograf's `_ensure_route` auto-registers the agent on first span and the Gantt renders one row per agent.

## Multi-turn and Conversations

A `Runner` can be called multiple times for a multi-turn Conversation. Each `run()` mints a new Session (so per-turn state doesn't leak) but keeps the stable `conversation_id`. The Conversation is responsible for minting the per-turn Session; `session_id` can be overridden by the caller to adopt an external id — this is how `GoldfiveADKAgent` inherits the adk-web session id.

`ConversationStarted` fires once per Conversation (first turn only); `ConversationEnded` fires when the Runner is closed. Multi-turn runs in adk-web share the plan across turns — the planner's refine accumulates task history across turns.

## Closing the Runner

```python
await runner.close()
```

Shuts down any `control` channel, flushes sinks, and runs registered close hooks. `wrap`-returned runners auto-register a close hook for the telemetry plugin.

## What to read next

- `docs/design/ARCHITECTURE.md` §"The Runner" — formal composition and event-ownership ADR.
- `docs/reference/api.md` §"Runner", §"Session", §"Orchestration state keys" — full API surface.
- `docs/design/PLAN-LIFECYCLE.md` — planner / refine / revision metadata.
- `docs/design/STATE-MACHINE.md` — task state transitions (PENDING → RUNNING → COMPLETED / FAILED / CANCELLED / NOT_NEEDED).
- `docs/guides/goals-and-plans.md` — GoalDeriver and Plan shape.
- `docs/guides/writing-an-agent-adapter.md` — authoring a custom AgentAdapter.
