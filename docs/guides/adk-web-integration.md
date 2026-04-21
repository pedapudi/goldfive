# Using goldfive with `adk web`

`adk web` is Google ADK's local web UI for driving an agent
interactively. It expects the module it loads to expose an
`App(root_agent=...)` whose `root_agent` is a
`google.adk.agents.BaseAgent`. Before goldfive Phase 2,
`goldfive.wrap(agent)` always returned a `goldfive.Runner` — great for
programmatic use, but *not* a `BaseAgent`, so the ADK UI couldn't load
it.

As of issue [#77], `goldfive.wrap(adk_agent)` returns a polymorphic
[`GoldfiveADKAgent`](../../goldfive/adapters/adk_wrap.py). The same
object now works in both contexts:

| Call site | Needs | What goldfive returns |
|---|---|---|
| `adk web` | `BaseAgent` with `run_async(ctx)` | `GoldfiveADKAgent` — it IS a `BaseAgent`. |
| Programmatic | `await runner.run(user_input)` | `GoldfiveADKAgent` — same object, same method. |

No new entry points, no second wrap variant — the existing call site
just works.

## The any-tree guarantee

`goldfive.wrap(any_adk_tree)` works regardless of tree shape.
Goldfive builds one `InMemoryRunner` around the tree root; ADK's
native mechanisms (`AgentTool`, `transfer_to_agent`, `sub_agents`)
resolve delegation inside the tree. The tree is **respected, never
rewritten or flattened**:

| Tree shape | What wrap does |
|---|---|
| Single agent — `Agent(name="worker", ...)` | One runner around `worker`; every invoke drives it. |
| Coordinator with `sub_agents` | One runner around the coordinator. ADK's `transfer_to_agent` handles delegation within a turn. |
| Coordinator with `AgentTool`-wrapped specialists | One runner around the coordinator. AgentTool calls spawn sub-Runners that inherit the goldfive plugin. |
| Deep nesting (`inner_agent` wrappers, `AgentTool` inside an agent whose parent is itself an `AgentTool`, …) | One runner at the root; ADK handles arbitrary-depth delegation. `available_agents` walks the tree to expose every reachable name for the planner. |

A writer that has an `editor` sub-agent still has that editor when
the writer's invocation drives it. A coordinator that has three
`AgentTool`s as tools still has those tools. `available_agents` is
an advisory list the planner uses to populate
`task.assignee_agent_id` as a delegation hint; goldfive does not
route on it. See
[ARCHITECTURE.md §"Single-Runner dispatch"](../design/ARCHITECTURE.md#single-runner-dispatch-goldfive-drives-the-root-adk-delegates-within)
for the underlying model.

## The recipe

Swap the line that constructs the `root_agent` for `goldfive.wrap(...)`
and leave the rest of your ADK code untouched:

```python
# agent.py — what `adk web` loads
from google.adk.agents import Agent
from google.adk.apps.app import App
import goldfive

real_agent = Agent(
    name="coordinator",
    model="gpt-4o-mini",
    sub_agents=[...],
)

# Before: root_agent = real_agent
root_agent = goldfive.wrap(real_agent)

app = App(name="my-demo", root_agent=root_agent)
```

Run it:

```bash
uv pip install -e '.[adk]'
adk web agent.py
```

Every user turn the UI submits now flows through goldfive's pipeline —
goal-derive → plan → execute → emit events — and the ADK UI renders a
short stream of `Event` objects summarising the plan + each completed
task.

A complete runnable file lives at
[`examples/adk_web_wrapped.py`](../../examples/adk_web_wrapped.py).

## What the UI sees each turn

When ADK invokes `root_agent.run_async(ctx)`, goldfive:

1. Extracts the latest user text from `ctx.user_content` (falling back
   to the session's event history).
2. Runs one `Runner.run(user_input, context={"adk_ctx": ctx})` pass.
3. Synthesises an `Event` stream from the resulting
   [`ExecutionOutcome`](../../goldfive/results.py):
   - A **plan summary** event (the first message).
   - One event per completed task, keyed by `Task.title`.
   - One line per drift event observed during the turn.
   - A terminal `turn_complete=True` event closing the turn.

This is deliberately minimal — enough for `adk web` to render a coherent
turn without duplicating what goldfive emits into its own event sinks.
Richer views come from attaching a sink to the wrapped runner (see
below).

## Programmatic use still works

```python
root_agent = goldfive.wrap(real_agent)

# Same object. Different call site.
outcome = await root_agent.run("plan a presentation about waffles")
```

`GoldfiveADKAgent.run(user_input, **kwargs)` delegates straight to the
inner `Runner.run(...)`, so every Runner knob — `context=`, cancellation
via `ControlChannel`, drift handling — behaves as before.

## Harmonograf observability composes cleanly

The wrapper exposes the inner `Runner`'s sink list as a property, so
`harmonograf_client.observe()` (which appends a `HarmonografSink`) works
unchanged:

```python
import goldfive
import harmonograf_client

root_agent = harmonograf_client.observe(goldfive.wrap(real_agent))
# observe() appended a sink to root_agent.sinks — the returned object is
# still the same GoldfiveADKAgent. adk web will load it just the same.

app = App(name="observed-demo", root_agent=root_agent)
```

## What is and is not shared with the adk-web session

The goldfive pipeline that runs for each turn uses its own internal
`InMemoryRunner` (via `ADKAdapter` — see
[ARCHITECTURE.md §"Single-Runner dispatch"](../design/ARCHITECTURE.md#single-runner-dispatch-goldfive-drives-the-root-adk-delegates-within)).
That runner is independent of the ADK session that `adk web`
hosts. In practice this means:

- Per-turn goldfive state (plan, drift history, reporting tool calls)
  lives in goldfive's `Session` and on goldfive's sinks.
- The ADK UI sees the synthesised `Event` stream from step 3 above.
- Cross-turn memory / state shared at the *adk web session* level is on
  the Phase 3 roadmap (the sibling agent currently wires
  conversation-level continuity through `run()`; see issue #71).

## Coordinator + AgentTool: delegation happens inside the turn

A common shape: a coordinator whose `tools` list contains
`AgentTool(specialist_a)`, `AgentTool(specialist_b)`, etc. The
coordinator's instruction text typically tells it to call the
appropriate specialist as a tool based on the user request.

**What goldfive does with this tree.** Builds one `InMemoryRunner`
around the coordinator with the goldfive plugin installed. Every
`invoke(task, session)` drives that one runner. The coordinator's
LLM decides whether to answer directly or delegate via
`AgentTool(specialist_a)`; ADK spawns a sub-Runner for the
specialist and propagates the plugin manager into it automatically.

**What the planner sees.** `available_agents` lists every reachable
agent name — the planner may populate `task.assignee_agent_id` with
a specific agent name to hint the coordinator toward a specialist.
Under the single-Runner model this is advisory only: the hint rides
in the task context the coordinator reads but does not force
routing.

**The runaway-delegation cap.** A coordinator whose prompt
describes a pipeline ("first research, then build, then review…")
can enter a self-delegating loop. Goldfive cannot require prompt
cooperation (users bring their own trees), so the plugin enforces
a per-invocation cap on AgentTool spawns — default 16, configurable
via `ADKAdapter(agent_tool_cap=N)` (set to 0 to disable). When the
cap trips, the plugin emits a `RUNAWAY_DELEGATION` drift and
cancels the invocation; the Steerer's refine hook gets a chance to
salvage the run. See
[common-failure-modes §"coordinator+AgentTool loop under real LLM"](common-failure-modes.md#8-coordinatoragenttool-loop-under-real-llm)
for the detailed signature and
[RATIONALE.md §"Why single-Runner, not registry-dispatch"](../design/RATIONALE.md#why-single-runner-not-registry-dispatch)
for the design history.

### End-to-end example: coordinator + AgentTool under `adk web`

```python
# agent.py — what `adk web` loads
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.tools.agent_tool import AgentTool
import goldfive

researcher = Agent(
    name="researcher",
    model="gpt-4o-mini",
    instruction=(
        "You are a researcher. Each message you receive is a single "
        "task — research the topic and call report_task_completed "
        "with a concise summary. Do not delegate."
    ),
)

writer = Agent(
    name="writer",
    model="gpt-4o-mini",
    instruction=(
        "You are a writer. Each message is a single task — produce "
        "the requested draft and call report_task_completed."
    ),
)

coordinator = Agent(
    name="coordinator",
    model="gpt-4o-mini",
    tools=[AgentTool(researcher), AgentTool(writer)],
    instruction=(
        "You coordinate a research + writing pipeline. When asked to "
        "do multi-step work, call the researcher AgentTool first, "
        "then the writer AgentTool with the research results."
    ),
)

# Wrap the coordinator — goldfive builds one runner around it.
root_agent = goldfive.wrap(coordinator)

# goldfive.adapters.adk.ADKAdapter.available_agents is
# ['coordinator', 'researcher', 'writer'] — advisory for the
# planner. Every task drives the coordinator; delegation to
# researcher / writer happens via the AgentTool calls the
# coordinator makes inside its turn.

app = App(name="multi-agent-demo", root_agent=root_agent)
```

Run:

```bash
uv pip install -e '.[adk]'
adk web agent.py
```

Each user turn, the UI submits a request; goldfive's planner
produces a two-task plan (research → write) and the executor drives
the coordinator for each task. The coordinator's AgentTool calls
delegate to the specialists; the per-invocation cap catches any
runaway loop.

## Pre-built Runner degrade mode

If you construct your own `InMemoryRunner` and hand it to
`goldfive.wrap(...)` (or to `ADKAdapter(...)` directly), goldfive
uses the runner verbatim. `available_agents` reports just the
runner's root agent name; the wrap-time plugin-installed integrity
check is skipped because the caller may have passed a runner shape
we don't fully control.

Every `invoke(task, session)` drives that one runner — which is
the same behaviour as the non-degraded path under the single-Runner
model, the only difference being that goldfive didn't construct the
runner itself.

## Limitations

- `ControlChannel` steering from a harmonograf UI does not yet reach the
  adk-web-driven pipeline — Phase 4 follow-up.
- Only ADK-shaped agents get the polymorphic return. Callable agents and
  Claude SDK factories still return a plain `Runner`; that's intentional
  (they can't satisfy `BaseAgent` anyway).

[#77]: https://github.com/pedapudi/goldfive/issues/77
