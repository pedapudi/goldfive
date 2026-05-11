# goldfive

**Stay on target.**

goldfive is a small, framework-agnostic Python library that wraps an agent
tree with the orchestration scaffolding most agents quietly need: an
explicit **goal**, a **plan** broken into tasks, per-turn **drift analysis**,
and a **steering** loop that nudges the agent back on course when it wanders
— without driving the tree per-task.

Execution model (since goldfive#141): **overlay, not controller.**
`goldfive.wrap(tree)` hands the caller's original request verbatim to the
tree exactly once, observes via ADK callbacks, and intervenes structurally
through an intervention ladder (Levels 0-5) rather than rewriting every
turn. The tree runs its natural flow; goldfive watches, steers on drift,
and reconciles a plan view alongside.

It does not ship an LLM client, a prompt DSL, or a tool registry. It wraps
whatever agent runtime you already use (Google ADK, the Anthropic SDK, a
plain callable, ...) behind a narrow `AgentAdapter` protocol and gives you:

- a `Runner` (or one-line `goldfive.wrap` / `goldfive.run`) that overlays
  goldfive's goal / plan / drift machinery on top of a live tree
- pluggable `GoalDeriver`, `Planner`, `Executor`, `Steerer` components
  plus an ADK `BasePlanner` subclass (`GoldfivePlanner`) auto-attached per
  `LlmAgent` for per-turn structural steering
- an observation-driven `PlanReconciler` that maps before/after-agent
  callbacks to plan-task transitions
- an `EventSink` stream of proto-encoded events you can log, persist, or
  ship to an observability console

goldfive is the orchestration half of
[harmonograf](https://github.com/pedapudi/harmonograf), extracted so you
can use the control loop without the console.

## Architecture at a glance

```
           caller's agent tree (any shape)
           ┌──────────────────────────────┐
           │  LlmAgent / Coordinator      │
           │    ├─ AgentTool(Specialist)  │
           │    └─ sub_agents=[...]       │
           └──────────────────────────────┘
                         │
                         ▼
            goldfive.wrap(tree, ...)
                         │
                         ▼
     ┌─────────────────────────────────────────────┐
     │         GoldfiveADKAgent (BaseAgent)        │
     │  (adk-web sees a root_agent; Runner inside) │
     └─────────────────────────────────────────────┘
                         │
                         ▼
     ┌─────────────────────────────────────────────┐
     │   Runner                                    │
     │     SequentialExecutor(overlay_mode=True)   │
     │       ──► one adapter.invoke_passthrough    │
     │                                             │
     │   auto-attached per LlmAgent:               │
     │     GoldfivePlanner (BasePlanner subclass)  │
     │                                             │
     │   observation:                              │
     │     PlanReconciler (before/after_agent)     │
     │     ToolLoopTracker (after_tool)            │
     │                                             │
     │   control:                                  │
     │     DefaultSteerer + intervention ladder    │
     │     LLMPlanner.{plan,refine}                │
     └─────────────────────────────────────────────┘
                         │
                         ▼
        EventSink stream (proto goldfive.v1.Event)
            InMemory / Logging / JSONL / SQLite /
                   GRPC / HarmonografSink
                         │
                         ▼
              harmonograf server + UI
```

Key properties:

| Property | Shape |
|---|---|
| Tree shape | any — single `LlmAgent`, coordinator + `AgentTool` specialists, deep `sub_agents` nesting |
| Tree rewriting | none; `goldfive.wrap` walks once to build a `name → BaseAgent` registry |
| Per-task driving | no (since #141) — one invocation, natural flow |
| Planning | LLM-driven by default (detects ADK's LLM); falls back to `PassthroughPlanner` when no LLM |
| Drift signals | tool errors, refusals, tool-loops, reasoning similarity, goal drift (opt-in), cross-layer delegation, hallucinated tools |
| Intervention | six-level ladder (OBSERVE / ABSORB / NUDGE / CANCEL_REINVOKE / PAUSE_ESCALATE / TERMINATE) in `DefaultSteerer` |

## Get running in 10 minutes

The fastest way to see goldfive work end-to-end is with its observability
console, [harmonograf](https://github.com/pedapudi/harmonograf). The
walkthrough installs both, boots a local stack, runs the
`examples/harmonograf_observed/` agent, and shows every event flowing
into the UI — no LLM credentials required.

1. `uv sync` in this repo (Python 3.11+, `uv` on your PATH).
2. Clone and `make demo` in harmonograf (server + UI on :7531 and :5173).
3. `uv run python examples/harmonograf_observed/agent.py`.

Full walkthrough: **[observability-with-harmonograf.md](docs/guides/observability-with-harmonograf.md)**.

## Install

```bash
uv add goldfive           # recommended
# or
pip install goldfive
```

Optional extras:

- `goldfive[adk]` — Google ADK adapter (`google-adk`).
- `goldfive[claude]` — Claude Agent SDK adapter (`anthropic`).
- `goldfive[examples]` — runtime deps for the scripts in [`examples/`](examples/) (`rich`).
- `goldfive[proto]` — regenerate proto stubs with `make proto` (`grpcio`, `grpcio-tools`, `mypy-protobuf`).
- `goldfive[dev]` — test + lint tooling used by the repo itself (`pytest`, `ruff`, `mypy`, ...).

## Hello goldfive

The fastest path to a goldfive-wrapped agent is a single call to
`goldfive.run`. It picks the right adapter for your agent, reuses
the agent's LLM when it can detect one, and returns an
`ExecutionOutcome`:

```python
import asyncio
import goldfive

# `agent` is any of: an ADK BaseAgent (or pre-built Runner), a Claude
# SDK client factory, an async (task, session, tools) -> InvocationResult
# callable, or anything implementing goldfive.AgentAdapter.
outcome = await goldfive.run(agent, "make a presentation about waffles")
```

Prefer to keep the runner around (for `.resume()`, custom sinks, or
multiple runs)? Use `goldfive.wrap`:

```python
runner = goldfive.wrap(
    agent,
    sinks=[my_sink],
    # Common overrides (all optional):
    # planner=LLMPlanner(call_llm=..., model=...),
    # goal_deriver=LLMGoalDeriver(call_llm=..., model=...),
    # steerer=DefaultSteerer(),
    # plugins=[HarmonografTelemetryPlugin(...)],
    # control=ControlChannel(),
)
outcome = await runner.run("make a presentation about waffles")
```

Every default component is overridable. Keyword arguments accepted by
both `wrap` and `run`:

| Keyword | Default | Notes |
|---|---|---|
| `planner=` | `LLMPlanner` when an LLM is detectable, else `PassthroughPlanner` | Any `Planner` |
| `goal_deriver=` | `LLMGoalDeriver` / `LiteralGoalDeriver` by the same rule | Any `GoalDeriver` |
| `executor=` | `SequentialExecutor(overlay_mode=True)` | Any `Executor` |
| `steerer=` | `DefaultSteerer()` | |
| `sinks=` | `[LoggingSink()]` | Pass `[]` to suppress |
| `call_llm=` | auto-detected from ADK trees; none otherwise | Async `(system, user, model) -> str` |
| `model=` | auto-detected from ADK; else empty string | |
| `max_task_invocations=` | `None` (unbounded) | Cap on adapter invocations per run |
| `plugins=` | `None` | List of ADK `BasePlugin` instances installed on the runner |
| `control=` | `None` | `ControlChannel` for live PAUSE / STEER / CANCEL / etc. |
| `observation_only=` | `True` (#254) | Detection + refine still run + sinks see `PlanRevised` with `dry_run=true`, but goldfive does NOT mutate `session.plan`, dispatch `GOLDFIVE_STEER`, or cancel the in-flight invocation. Pass `False` to opt into active steering — see [`docs/design/OBSERVATION-ONLY.md`](docs/design/OBSERVATION-ONLY.md). |

`goldfive.wrap(any_adk_tree)` works regardless of tree shape — single
agent, coordinator with `AgentTool`-wrapped specialists, deep
`sub_agents` nesting, `inner_agent` wrappers. Under the **single-Runner,
overlay-mode** model (since goldfive#141):

- goldfive walks the tree once at wrap time, builds a
  `name → BaseAgent` registry, and **attaches `GoldfivePlanner`** to
  every `LlmAgent` so per-turn structural context is injected.
- `goldfive.wrap` builds **one** `InMemoryRunner` around the root; the
  tree runs its natural flow (coordinator delegates, specialists report
  back, etc.).
- `adapter.invoke_passthrough(user_input)` sends the caller's request
  verbatim — no `"Task: X"` framing, no goldfive jargon — and the
  `PlanReconciler` maps observed agent turns back to plan-task
  transitions via ADK callbacks.
- The tree is **respected, never rewritten or flattened.**

See
[`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md)
for the full model and
[`docs/guides/adk-web-integration.md`](docs/guides/adk-web-integration.md)
for a coordinator+AgentTool example under `adk web`.

A runnable demo lives in
[`examples/hello_callable.py`](examples/hello_callable.py).

## What's new in this version

Arc since the last stable doc refresh, in rough order of impact:

- **Overlay execution model** (#141-#148) — one invocation per run, observation-driven reconciliation, per-task driving is retired.
- **Intervention ladder** (#142/#147) — Levels 0-5 uniformly map `(drift_kind, severity, occurrence_count)` to the right response.
- **GoldfivePlanner** (#153/#156) — `BasePlanner` subclass auto-attached per `LlmAgent`, injects tree-agnostic orchestration context + structural drift gate.
- **Tree-aware planner** (#151/#160) — `LLMPlanner` plans + refines against a structured agent registry.
- **`goldfive.*` session-state namespace** (#152/#159) — documented keys
  on ADK session state bridged from `goldfive.orchestration_state`.
- **STEER idempotency + author propagation** (#171/#175) — `DefaultSteerer.observe` dedupes by source `annotation_id`.
- **Tool-loop detector** (#181/#186) — three-mode args-aware detector on every tool call, not just reporting tools.
- **Per-LLM-call instrumentation** (#172/#174) — structured request/response logs with `chars`/`messages_count`/`duration`/`usage`.

Full list: [CHANGELOG.md](CHANGELOG.md).

## Docs

**Start with [`docs/guides/getting-started.md`](docs/guides/getting-started.md)** —
install, run your first goldfive-wrapped agent in about ten minutes,
inspect the event stream. Concrete and runnable.

### Design

- [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md) — overview of the six primitives, how they compose, full lifecycle.
- [`docs/design/PROTOCOLS.md`](docs/design/PROTOCOLS.md) — the six protocol contracts with minimal implementations.
- [`docs/design/STATE-MACHINE.md`](docs/design/STATE-MACHINE.md) — task lifecycle state diagram, transition rules, invariants.
- [`docs/design/TASK-LIFECYCLE.md`](docs/design/TASK-LIFECYCLE.md) — per-task lifecycle, reporting-tool dispatch layering, cancellation protocol.
- [`docs/design/PLAN-LIFECYCLE.md`](docs/design/PLAN-LIFECYCLE.md) — plan-level state machine: revision modes, run-termination predicate, cascade semantics.
- [`docs/design/DRIFT.md`](docs/design/DRIFT.md) — full drift-kind taxonomy (25+), classification rules, refine policy.
- [`docs/design/EVENT-MODEL.md`](docs/design/EVENT-MODEL.md) — proto event taxonomy, sequence semantics, `EventSink` contract.
- [`docs/design/CONTROL.md`](docs/design/CONTROL.md) — live-steering control channel protocol (PAUSE / RESUME / CANCEL / STEER / REWIND_TO / APPROVE / REJECT).
- [`docs/design/APPROVAL.md`](docs/design/APPROVAL.md) — human-in-the-loop approval flows (Flow A: goldfive-native; Flow B: ADK tool confirmation).

### Further reading — the "why" docs

- [`docs/design/VOCABULARY.md`](docs/design/VOCABULARY.md) — exhaustive type-system reference. Every enum value, every bridge between types, side-by-side. Start here if `ControlKind.STEER` vs `DriftKind.USER_STEER` ever confuses you.
- [`docs/design/RATIONALE.md`](docs/design/RATIONALE.md) — design-rationale "why is it this way?" for each major abstraction. Read when a choice feels arbitrary.

### Guides

- [`docs/guides/getting-started.md`](docs/guides/getting-started.md) — install + first agent.
- [`docs/guides/observability-with-harmonograf.md`](docs/guides/observability-with-harmonograf.md) — ten-minute end-to-end with the harmonograf UI.
- [`docs/guides/telemetry-with-harmonograf.md`](docs/guides/telemetry-with-harmonograf.md) — reading the UI: Gantt, span popovers, Inspector Drawer, live steering, plan revisions.
- [`docs/guides/insight-from-logs.md`](docs/guides/insight-from-logs.md) — operators without the UI: raw event stream, session state after a run, post-mortem from JSONL / SQLite.
- [`docs/guides/common-failure-modes.md`](docs/guides/common-failure-modes.md) — catalog of observed failure shapes, each with its signature and recovery path.
- [`docs/guides/writing-an-agent-adapter.md`](docs/guides/writing-an-agent-adapter.md) — wrap a new framework.
- [`docs/guides/writing-an-event-sink.md`](docs/guides/writing-an-event-sink.md) — build a custom sink.
- [`docs/guides/choosing-a-sink.md`](docs/guides/choosing-a-sink.md) — decision matrix across the five shipped sinks.
- [`docs/guides/goals-and-plans.md`](docs/guides/goals-and-plans.md) — authoring custom `GoalDeriver` / `Planner`.
- [`docs/guides/persistence-and-recovery.md`](docs/guides/persistence-and-recovery.md) — JSONL + SQLite persistence, `Runner.resume()`.
- [`docs/guides/grpc-transport.md`](docs/guides/grpc-transport.md) — `GRPCSink` + `GoldfiveIngressServer` for out-of-process observers.
- [`docs/guides/harmonograf-integration.md`](docs/guides/harmonograf-integration.md) — plugging harmonograf in as a sink.
- [`docs/guides/troubleshooting.md`](docs/guides/troubleshooting.md) — common setup / runtime failures.

### Reference

- [`docs/reference/api.md`](docs/reference/api.md) — public API surface.
- [`docs/reference/tool-protocol.md`](docs/reference/tool-protocol.md) — the eight reporting tools.
- [`docs/performance.md`](docs/performance.md) — orchestration-overhead baseline.

## License

Apache-2.0.
