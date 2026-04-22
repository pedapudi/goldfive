# Using goldfive with `adk web`

`adk web` is Google ADK's local web UI for driving an agent
interactively. It loads a Python module that exposes an
`App(root_agent=...)` whose `root_agent` is a
`google.adk.agents.BaseAgent`. `goldfive.wrap(adk_agent)` returns a
polymorphic
[`GoldfiveADKAgent`](../../goldfive/adapters/adk_wrap.py) — a
`BaseAgent` subclass that also exposes the Runner surface, so one
object works in both contexts.

| Call site | Needs | What goldfive returns |
|---|---|---|
| `adk web` | `BaseAgent` with `run_async(ctx)` | `GoldfiveADKAgent` — it IS a `BaseAgent`. |
| Programmatic | `await runner.run(user_input)` | `GoldfiveADKAgent` — same object, same method. |

## The any-tree guarantee

`goldfive.wrap(any_adk_tree)` works regardless of tree shape. One
`InMemoryRunner` around the tree root (single-Runner model,
goldfive#130); ADK's native mechanisms (`AgentTool`,
`transfer_to_agent`, `sub_agents`) resolve delegation inside the tree.
The tree is **respected, never rewritten or flattened**.

| Tree shape | What wrap does |
|---|---|
| Single `LlmAgent` | One runner around the agent; every turn drives it. |
| Flat list of specialists under a coordinator via `sub_agents` | One runner around the coordinator. ADK's `transfer_to_agent` handles delegation within a turn. |
| Coordinator with `AgentTool`-wrapped specialists | One runner around the coordinator. AgentTool calls spawn sub-Runners; ADK propagates the goldfive plugin manager into them automatically. |
| Deep nesting / wrappers / arbitrary composition | One runner at the root; ADK handles arbitrary-depth delegation. `available_agents_tree` walks the tree once to give the planner a structured map (name / depth / parent / role / kind). |

`available_agents` (sorted list of reachable agent names) and the
richer `available_agents_tree` (structured tree metadata,
goldfive#151) are advisory inputs to the planner. The planner may
populate `task.assignee_agent_id` as a delegation hint; goldfive does
not route on it under the single-Runner model.

See
[ARCHITECTURE.md §"Single-Runner dispatch"](../design/ARCHITECTURE.md)
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

root_agent = goldfive.wrap(real_agent)

app = App(name="my-demo", root_agent=root_agent)
```

Run it:

```bash
uv pip install -e '.[adk]'
adk web agent.py
```

Every user turn the UI submits now flows through goldfive's pipeline —
goal-derive → plan → execute (overlay) → emit events — and the ADK UI
renders a short stream of `Event` objects summarising the plan + each
completed task.

A complete runnable file lives at
[`examples/adk_web_wrapped.py`](../../examples/adk_web_wrapped.py).

## What wrap installs on the tree

`goldfive.wrap(adk_agent)` performs three tree-wide passes at
construction time:

1. **Reporting-tool augmentation.** Walks `sub_agents` / `inner_agent`
   / `tool.agent` edges and appends the seven `report_task_*` tools
   to every agent that carries a mutable `tools` list. Idempotent —
   agents that already have them are skipped.
2. **`GoldfivePlanner` attachment.** Every `LlmAgent` gets a
   `GoldfivePlanner(BasePlanner)` attached as its `planner=` field
   (goldfive#153). The planner composes with a user-supplied planner
   if one was already set. Per-agent opt-out via
   `agent._goldfive_planner_opt_out = True`.
3. **Plugin install.** Builds one `InMemoryRunner` around the root
   and installs `_GoldfiveADKPlugin` on it. The plugin is where every
   structural observation lives — `before_tool_callback`,
   `after_tool_callback`, `before_model_callback`, `before_agent_callback`,
   `after_agent_callback`, and the tool-loop detector (goldfive#181).

`goldfive.wrap(agent, plugins=[HarmonografTelemetryPlugin(client)])`
forwards additional plugins into the same runner, deduped by plugin
`name` (goldfive#166) so caller-supplied plugins and goldfive's own
plugin never double-install.

## The per-turn orchestration context block

On every LLM call inside the wrapped tree, the goldfive plugin's
`before_model_callback` detects that the running agent carries a
`GoldfivePlanner` and appends a short structural context block to
`llm_request.config.system_instruction`. The block is tree-agnostic —
no presentation / research / coordinator vocabulary — and reads
from `session.state['goldfive.*']` keys the reconciler + steerer
populate per turn:

- Current task id, title, description, status.
- Plan summary + every completed task's summary.
- Goal summaries.
- Active USER_STEER (when one is in flight) + its author.
- Cancelled function-call ids to avoid (goldfive#147).

The injection path subclasses `BasePlanner` directly rather than
`PlanReActPlanner` so goldfive does not inherit ReAct response
filtering (which would constrain agent output). See
`goldfive/planners/goldfive_planner.py` for the full contract.

## What the UI sees each turn

When ADK invokes `root_agent.run_async(ctx)`, `GoldfiveADKAgent`:

1. Pins `ctx.session.id` (adk-web's outer session id) onto the
   goldfive `Session.id` and the `ADKAdapter`'s internal session
   bookkeeping (goldfive#161 + #164). Result: one session in
   harmonograf per run, and per-event `session_id` stamping
   (goldfive#155) routes correctly.
2. Extracts the latest user text from `ctx.user_content` (falling
   back to `ctx.session.events` for frameworks whose context shape
   varies).
3. Runs one `Runner.run(user_input, context={"adk_ctx": ctx},
   session_id=outer_sid)` pass.
4. Synthesises an `Event` stream from the resulting
   `ExecutionOutcome`:
   - A **plan summary** event.
   - One event per completed task, keyed by `Task.title`.
   - One line per drift observed during the turn.
   - A terminal `turn_complete=True` event closing the turn.

Deliberately minimal — enough for `adk web` to render a coherent turn
without duplicating goldfive's own sink output. Richer views come from
attaching a sink to the wrapped runner (see the harmonograf section
below) or attaching `HarmonografTelemetryPlugin` to the `App`.

## Model setup via `LiteLlm`

ADK sub-agents typically use a LiteLLM model string (`openai/gpt-4o-mini`,
`anthropic/claude-sonnet-4-5`, etc.) or a Gemini model id. A common
pattern in the examples:

```python
from google.adk.agents import Agent

MODEL_NAME = os.environ.get("USER_MODEL_NAME", "openai/gpt-4o-mini")
agent = Agent(name="writer", model=MODEL_NAME, instruction="...")
```

Pinning the default matters because of the pitfall below.

### Pitfall: Gemini default without `GOOGLE_API_KEY`

harmonograf's `tests/reference_agents/presentation_agent_orchestrated/agent.py`
defaults `USER_MODEL_NAME` to `gemini-2.5-flash`. Running that
module without `export USER_MODEL_NAME=openai/gpt-4o-mini` (or
setting `GOOGLE_API_KEY`) gets you a coordinator that silently
instantiates a Gemini client with no credentials. The first LLM call
raises after goldfive has already run through plan / goal-derive /
initial dispatch; the stream ends with a bare
`AttributeError: '_async_httpx_client'` at teardown and the UI shows
"goldfive run complete." with nothing actually done.

**Signature.** Run terminates almost immediately with a single
"goldfive run complete." event. Traceback mentions
`_async_httpx_client`. Coordinator's instruction looked fine. Works
when you set `USER_MODEL_NAME` to a non-Gemini model.

**Fix.**

```bash
export USER_MODEL_NAME=openai/gpt-4o-mini
# or, for the kikuchi local LLM stack:
export USER_MODEL_NAME=openai/qwen3-coder-30b
export OPENAI_BASE_URL=http://kikuchi:8000/v1
export OPENAI_API_KEY=sk-anything   # not validated by kikuchi
```

Then re-run `adk web ...`. The same env vars feed
`LLMPlanner.call_llm` when an `OPENAI_API_KEY` is set, so planner +
subagents land on the same backend.

## Programmatic use still works

```python
root_agent = goldfive.wrap(real_agent)

# Same object, different call site.
outcome = await root_agent.run("plan a presentation about waffles")
```

`GoldfiveADKAgent.run(user_input, **kwargs)` delegates straight to the
inner `Runner.run(...)`, so every Runner knob — `context=`,
cancellation via `ControlChannel`, drift handling — behaves as before.

## Attaching harmonograf

Two independent wiring points:

1. **`HarmonografSink`** on the goldfive runner — receives every
   `Event` (RunStarted / GoalDerived / PlanSubmitted / TaskStarted /
   TaskCompleted / DriftDetected / PlanRevised / …).
2. **`HarmonografTelemetryPlugin`** on the ADK `App` — captures
   per-agent spans (INVOCATION / LLM_CALL / TOOL_CALL), which the UI
   renders on the Agents timeline with per-agent Gantt rows
   (goldfive#170 + harmonograf#80).

```python
import goldfive
from google.adk.apps.app import App
from harmonograf_client import Client, HarmonografSink, HarmonografTelemetryPlugin

client = Client(name="my-agent", server_addr="127.0.0.1:7531")
wrapped = goldfive.wrap(root_agent, sinks=[HarmonografSink(client)])

app = App(
    name="my-demo",
    root_agent=wrapped,
    plugins=[HarmonografTelemetryPlugin(client)],
)
```

Both hooks deduplicate by plugin name (goldfive#166), so if you also
pass the plugin to `goldfive.wrap(plugins=[...])` only one instance
ends up installed on the runner.

`harmonograf_client.observe(wrapped)` is the legacy entry point —
still works, still appends a `HarmonografSink`. The newer path is
the explicit `App(plugins=[...])` wire above, because adk-web needs
the plugin on the App-level runner to see ADK-native spans.

## Coordinator + AgentTool: delegation happens inside the turn

A common shape: a coordinator whose `tools` list contains
`AgentTool(specialist_a)`, `AgentTool(specialist_b)`, etc. The
coordinator's instruction typically tells it to call the appropriate
specialist as a tool based on the user request.

**What goldfive does with this tree.** Builds one `InMemoryRunner`
around the coordinator with the goldfive plugin installed. Every
per-turn invocation drives that one runner. The coordinator's LLM
decides whether to answer directly or delegate via
`AgentTool(specialist_a)`; ADK spawns a sub-Runner for the specialist
and propagates the plugin manager into it automatically.

**What the overlay reconciler sees.** `before_agent_callback` fires
for every agent the tree runs, including sub-Runners. The reconciler
maps each `(name, invocation_id)` onto the first PENDING plan task
whose `assignee_agent_id == name` (with a contextual match fallback
walking the parent chain, goldfive#151). `after_agent_callback`
moves it to COMPLETED (or FAILED on error). PENDING tasks the tree
never visited are transitioned to `NOT_NEEDED` at end-of-invocation
(goldfive#163) — no soft follow-up dispatch, which prevented a class
of flow-prompted coordinator re-running its whole pipeline on every
follow-up user message.

**The runaway-delegation cap.** A coordinator whose prompt describes
a pipeline can enter a self-delegating loop. The plugin enforces a
per-invocation cap on AgentTool spawns (default 16, configurable via
`ADKAdapter(agent_tool_cap=N)`; `0` disables). On exceed the plugin
emits `RUNAWAY_DELEGATION` at CRITICAL severity and short-circuits
further AgentTool dispatches. The steerer's intervention ladder then
decides what to do (ABSORB via refine, PAUSE_ESCALATE, etc.).

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
        "You are a researcher. Each turn, goldfive gives you one "
        "current task via the orchestration context block at the "
        "top of your system instruction. Do that task; call "
        "report_task_completed when done; stop. Do not delegate."
    ),
)

writer = Agent(
    name="writer",
    model="gpt-4o-mini",
    instruction=(
        "You are a writer. Same contract as researcher — one task "
        "per turn from the orchestration context; call "
        "report_task_completed; stop."
    ),
)

coordinator = Agent(
    name="coordinator",
    model="gpt-4o-mini",
    tools=[AgentTool(researcher), AgentTool(writer)],
    instruction=(
        "You coordinate research + writing. Read the current task "
        "from the orchestration context and call the appropriate "
        "AgentTool — researcher for research tasks, writer for "
        "drafting."
    ),
)

root_agent = goldfive.wrap(coordinator)

app = App(name="multi-agent-demo", root_agent=root_agent)
```

Run:

```bash
uv pip install -e '.[adk]'
adk web agent.py
```

Each turn, goldfive's planner produces a multi-task plan; the
coordinator reads the current task from the context block and
delegates via AgentTool. The reconciler attributes each sub-Runner's
`before_agent` / `after_agent` pair to the right plan task.

For the full multi-agent example — coordinator + 4 specialists +
real file-writing tools — see
[`examples/presentation_agent/`](../../examples/presentation_agent/).

## Pre-built Runner degrade mode

If you construct your own `InMemoryRunner` and hand it to
`goldfive.wrap(...)` (or to `ADKAdapter(...)` directly), goldfive uses
the runner verbatim. `available_agents` reports just the runner's
root agent name; the wrap-time plugin-installed integrity check is
skipped because the caller may have passed a runner shape we don't
fully control.

Every `invoke(task, session)` drives that one runner — same behaviour
as the non-degraded path, the only difference being that goldfive
didn't construct the runner itself.

## Live steering from harmonograf

The harmonograf UI's Steer / Pause / Cancel / Approve buttons drive
the goldfive `ControlChannel`. Wire it through:

```python
from goldfive import ControlChannel
from harmonograf_client import Client, HarmonografSink

channel = ControlChannel()
client = Client(name="my-agent", server_addr="127.0.0.1:7531")

wrapped = goldfive.wrap(
    root_agent,
    control=channel,
    sinks=[HarmonografSink(client)],
)
# ... drain client.observe() into channel.send(bridge(event)) in a
#     companion task; forward channel.acks() back to the client.
```

STEER annotations carry an `annotation_id` (goldfive#171) that
propagates into the drift detail, the refine reason on the revised
plan, and the resulting `DriftDetected.annotation_id` field. This
lets harmonograf deduplicate redundant STEER clicks and show the
author of each refine.

See [../design/CONTROL.md](../design/CONTROL.md) for the full
protocol.

[#77]: https://github.com/pedapudi/goldfive/issues/77
