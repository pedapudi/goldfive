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

`goldfive.wrap(any_adk_tree)` works regardless of tree shape. The
adapter walks the wrap target once and builds a `name -> BaseAgent`
registry covering every reachable agent via `sub_agents`,
`inner_agent`, and `AgentTool.agent` edges. The registry is the
dispatch map the executor uses for every task:

| Tree shape | What wrap does |
|---|---|
| Single agent — `Agent(name="worker", ...)` | Single-entry registry; every dispatch goes to `worker`. |
| Coordinator with `sub_agents` | Registry lists coordinator and every sub-agent; planner may assign tasks to any of them. |
| Coordinator with `AgentTool`-wrapped specialists | Registry lists coordinator and every wrapped specialist; planner may route to a specialist directly, or route to the coordinator to let it compose specialists within its own turn. |
| Deep nesting (`inner_agent` wrappers, `AgentTool` inside an agent whose parent is itself an `AgentTool`, ...) | Every reachable agent is in the registry; cycles are skipped by id. |

The tree is **respected, never rewritten or flattened**. A writer
that has an `editor` sub-agent still has that editor when goldfive
dispatches a task to the writer. A coordinator that has three
`AgentTool`s as tools still has those tools when goldfive dispatches
a task to the coordinator. See
[ARCHITECTURE.md §"Registry dispatch"](../design/ARCHITECTURE.md#registry-dispatch-goldfive-drives-adk-executes)
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
per-agent `InMemoryRunner`s (via `ADKAdapter` — see
[ARCHITECTURE.md §"Registry dispatch"](../design/ARCHITECTURE.md#registry-dispatch-goldfive-drives-adk-executes)).
Those runners are independent of the ADK session that `adk web`
hosts. In practice this means:

- Per-turn goldfive state (plan, drift history, reporting tool calls)
  lives in goldfive's `Session` and on goldfive's sinks.
- The ADK UI sees the synthesised `Event` stream from step 3 above.
- Cross-turn memory / state shared at the *adk web session* level is on
  the Phase 3 roadmap (the sibling agent currently wires
  conversation-level continuity through `run()`; see issue #71).

## Coordinator + AgentTool: how dispatch interacts with the tree

A common shape: a coordinator whose `tools` list contains
`AgentTool(specialist_a)`, `AgentTool(specialist_b)`, etc. The
coordinator's instruction text will typically tell it to call the
appropriate specialist as a tool based on the user request.

**What goldfive does with this tree.**

- Registers every reachable agent (coordinator, specialist_a,
  specialist_b, plus any sub-agents of those) in the registry.
- Builds one `InMemoryRunner` per registered agent, with the
  goldfive plugin installed on each.
- The LLM planner sees every name in `available_agents` and may
  route tasks directly to a specialist.

**What changes for users wrapping a coordinator+AgentTool tree:**
when goldfive's planner assigns a task to `specialist_a`,
`ADKAdapter.invoke` dispatches directly to `specialist_a`'s runner.
The coordinator's instruction text that says "call the right
specialist via AgentTool" is effectively **superseded** for that
task, *but is not modified* — goldfive never edits the tree. If a
later task is assigned to the coordinator, the coordinator will
run with its instruction text unchanged and its AgentTools still
bound.

**What doesn't change.** When an agent is dispatched, its **full
subtree is available to it**. An AgentTool the dispatched agent
owns still fires when the agent calls it. A `sub_agent` still
resolves. ADK's plugin-inheritance carries the goldfive plugin
into the AgentTool-spawned sub-Runner so the nested invocation
still sees the state-protocol keys and emits observability events
([EVENT-MODEL.md §"Agent-invocation events"](../design/EVENT-MODEL.md#agent-invocation-events)).

This rules out a class of failure mode that plagued earlier
versions — the coordinator-loop-under-real-LLM that burned through
ADK's 500-call ceiling because every task re-entered the
coordinator's routing LLM turn. See
[RATIONALE.md §"Why per-agent runners, not always-root-dispatch"](../design/RATIONALE.md#why-per-agent-runners-not-always-root-dispatch)
for the detailed "why" and
[common-failure-modes §"coordinator+AgentTool loop under real LLM"](common-failure-modes.md#8-coordinatoragenttool-loop-under-real-llm-fixed-by-registry-dispatch)
for the signature and recovery path.

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

# Wrap the coordinator — goldfive discovers all three agents.
root_agent = goldfive.wrap(coordinator)

# goldfive.adapters.adk.ADKAdapter.available_agents is
# ['coordinator', 'researcher', 'writer']. An LLMPlanner with these
# three names may assign one task to 'researcher' and a dependent
# task to 'writer' — each goes directly to the assigned agent's
# per-agent runner. The coordinator is still available for tasks
# that legitimately require composition.

app = App(name="multi-agent-demo", root_agent=root_agent)
```

Run:

```bash
uv pip install -e '.[adk]'
adk web agent.py
```

Each user turn, the UI submits a request; goldfive's planner
produces a two-task plan (research → write) and the executor
dispatches each task to its assignee directly. No coordinator
routing loop, no 500-call ceiling, no wasted inference.

## Pre-built Runner degrade mode

If you construct your own `InMemoryRunner` and hand it to
`goldfive.wrap(...)` (or to `ADKAdapter(...)` directly), goldfive
cannot walk a tree it never saw. It falls back to a **single-entry
registry** pointing at the runner's root agent and emits a WARNING
log at wrap time:

```text
ADKAdapter: caller passed a pre-built Runner; per-task-assignee
dispatch is unavailable, all tasks will invoke the runner's
root_agent
```

Every `invoke(task, session)` drives that one runner regardless of
`task.assignee_agent_id`. If the planner assigns to an agent name
different from the root's name, the mismatch is logged at DEBUG
and the dispatch proceeds to the root anyway.

This mode is supported for backwards compatibility. The full
registry model requires passing a `BaseAgent` (the tree root) so
goldfive can discover the tree itself.

## Limitations

- `ControlChannel` steering from a harmonograf UI does not yet reach the
  adk-web-driven pipeline — Phase 4 follow-up.
- Only ADK-shaped agents get the polymorphic return. Callable agents and
  Claude SDK factories still return a plain `Runner`; that's intentional
  (they can't satisfy `BaseAgent` anyway).

[#77]: https://github.com/pedapudi/goldfive/issues/77
