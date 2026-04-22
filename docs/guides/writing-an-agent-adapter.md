# Writing an agent adapter

goldfive is agent-framework-agnostic. The adapter is the one piece of
the system that knows about a specific framework — ADK, Claude Agent
SDK, LangGraph, an MCP server, a bare async callable, anything.

This guide walks through writing a new `AgentAdapter` for a framework
goldfive doesn't ship. Reference adapters:

- `goldfive.adapters.callable.CallableAdapter` — the simplest form.
  Start here when prototyping a new adapter.
- `goldfive.adapters.adk.ADKAdapter` — the overlay-model flagship.
  Implements `invoke_passthrough` for the single-Runner overlay
  execution path (goldfive#141) plus the per-task `invoke` fallback.
  Installs the goldfive ADK plugin once per runner
  (`_register_plugin_on_runner` is idempotent by plugin name,
  goldfive#166). Exposes `available_agents_tree` for structural
  steering (goldfive#151).
- `goldfive.adapters.claude.ClaudeAgentSDKAdapter` — wraps Anthropic's
  Claude Agent SDK. Uses the per-task `invoke` path.

Related: [PROTOCOLS.md](../design/PROTOCOLS.md#agentadapter),
[tool-protocol.md](../reference/tool-protocol.md),
[ARCHITECTURE.md](../design/ARCHITECTURE.md).

## The protocol

```python
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

    # Optional (overlay model, goldfive#141):
    async def invoke_passthrough(
        self,
        user_message: str,
        session: Session,
        *,
        reconciler: PlanReconciler,
    ) -> InvocationResult: ...

    # Optional (structural steering, goldfive#151):
    @property
    def available_agents_tree(self) -> list[dict[str, Any]]: ...
```

Three required members plus two overlay extensions.

Responsibilities:

1. **Register reporting tools.** The steerer hands you a list of
   `ReportingToolSpec`s (see [tool-protocol.md](../reference/tool-protocol.md)).
   Translate each spec into the tool shape your framework expects and
   install a hook that routes calls to the spec's handler.
2. **Invoke the agent.** Two shapes:
   - **Per-task `invoke(task, session)`** (classic path) — render
     task context, run the agent, feed observed events to the
     steerer, return an `InvocationResult`.
   - **`invoke_passthrough(user_message, session, *, reconciler)`**
     (overlay model) — the executor passes the full user message
     once and observes the tree running naturally. The adapter feeds
     `before_agent` / `after_agent` / delegation observations into
     the `PlanReconciler`; the reconciler maps them to plan tasks.
     This is the path `SequentialExecutor(overlay_mode=True)` uses
     for adapters that expose the method (`ADKAdapter` does,
     `ClaudeAgentSDKAdapter` does not). `goldfive.wrap(adk_agent)`
     defaults to overlay mode.
3. **Expose `available_agents`.** The names the planner may populate
   `task.assignee_agent_id` with — advisory hints for delegation,
   not routing keys under the single-Runner model. Must be sorted
   and unique. Optionally also expose `available_agents_tree` —
   structured metadata (name / depth / parent / role / kind) per
   agent — which the planner uses to constrain `assignee_agent_id`
   selection for deep hierarchies (goldfive#151).

### The `available_agents` contract

`available_agents` is the advisory list the planner uses to
populate `task.assignee_agent_id`. The contract is:

- **Return a sorted list** so planner context is deterministic.
- **No duplicate names** — the planner cannot disambiguate.
- **Names should be reachable in the tree** so the hint is
  meaningful (e.g. the coordinator can recognize "research_agent"
  when it reads the current task's assignee from state).
- **Empty assignee is legal.** `task.assignee_agent_id == ""` is
  the single-agent default.
- **Unknown assignees do not raise.** Under the single-Runner
  model the adapter drives one runner; the planner's hint rides
  in task context but doesn't force dispatch. An unknown name is
  a no-op hint.

`ADKAdapter` exposes every agent reachable via `sub_agents` /
`inner_agent` / `AgentTool.agent` edges (see
`goldfive/adapters/adk.py::_collect_reachable_agent_names`).
`CallableAdapter` takes `available_agents` as a constructor
argument.

## The four things every adapter must do

### Render task context

The agent needs to know what task it's working on. Different
frameworks expose this differently:

- **ADK** — write `session.state["goldfive.current_task_id"]` and
  friends in the plugin's `before_run_callback` (against the live
  invocation session, not a `get_session` copy — see
  [TASK-LIFECYCLE.md §2.5](../design/TASK-LIFECYCLE.md#25-state-protocol-writes-live-in-the-plugins-before_run_callback));
  the agent reads them.
- **Claude Agent SDK** — inject task context into the system prompt
  or the first user message.
- **Custom callable** — pass it as a function argument.

All three approaches are equivalent. Pick whichever matches your
framework's idiom. The context the agent needs:

- `task.id`, `task.title`, `task.description`
- The summary of every completed task (from `session.completed_results`)
- The plan summary (`session.plan.summary`)
- Optionally: the goal summaries

### Intercept reporting tool calls

When the agent calls `report_task_completed("t3", summary="...")`,
your adapter must:

1. Recognize the tool name (it's in `REPORTING_TOOL_NAMES`).
2. Parse the arguments.
3. Route through
   `goldfive.adapters._tool_invocation.invoke_tool(specs, name, args, session, steerer)`
   — **not** `spec.handler` directly. `invoke_tool` runs four guard
   layers (schema / terminal-task / per-task loop / session-wide
   volume cap) before dispatching the spec's handler. Skipping this
   is the root cause of the filler-loop class of bugs (#108): the
   handler runs, the state updates, but the adapter keeps
   re-invoking the agent past a task that already completed.
4. Return the helper's return value as the tool result.

The handler is where the steerer applies the state transition.
`{"acknowledged": True}` is the conventional response body; the
guard layers may instead return `{"acknowledged": False, "error":
"task_already_terminal" | "loop_detected" | "missing_task_id" |
"unknown_task_id", ...}` — return those payloads verbatim to the
agent as the tool response. They are designed to be model-readable
so the agent can course-correct.

### Forward observed events to the steerer

Events that aren't reporting tool calls — LLM text output, non-report
tool calls, framework status events — should be fed to
`steerer.observe(event, session)`. This is how drift detection sees
refusals, context pressure, unexpected transfers, etc.

The adapter does not classify drift itself; it just forwards raw
observations. `DefaultSteerer.detect_drift()` is the classifier.

### Return an `InvocationResult`

```python
@dataclasses.dataclass
class InvocationResult:
    task_id: str
    text: str = ""
    stop_reason: str = ""
    error: Optional[Exception] = None
    raw: Any = None
```

After the agent finishes its turn, return:

- `task_id` — the id of the task that was invoked.
- `text` — the final assistant text (the "answer"), for callers that
  want to surface it.
- `stop_reason` — framework-specific stop reason (e.g. `"end_turn"`,
  `"max_tokens"`, `"tool_use"`). The steerer uses this for
  `CONTEXT_PRESSURE` and `TOO_MANY_STEPS` classification.
- `error` — an `Exception` if the framework raised, otherwise `None`.
- `raw` — the original framework response object, preserved for
  advanced sinks that want to introspect.

## Worked example: wrapping a hypothetical `awesome_agent_sdk`

Suppose you're using a framework called `awesome_agent_sdk` with this
shape:

```python
# what awesome_agent_sdk looks like (for the purposes of this example):

class AwesomeClient:
    def __init__(self, system_prompt: str, tools: list[ToolDef]): ...
    async def run(self, user_message: str) -> AsyncIterator[Block]: ...

class ToolDef:
    name: str
    parameters_schema: dict
    handler: Callable[[dict], Awaitable[dict]]

class Block:
    type: str   # "text" | "tool_use" | "stop"
    text: str = ""
    tool_name: str = ""
    tool_args: dict = {}
    stop_reason: str = ""
```

A full adapter for this fictional SDK is about 50 lines:

```python
from __future__ import annotations

from typing import Any

from awesome_agent_sdk import AwesomeClient, ToolDef  # fictional

from goldfive.adapters._tool_invocation import invoke_tool
from goldfive.protocols import AgentAdapter
from goldfive.reporting import ReportingToolSpec
from goldfive.results import InvocationResult
from goldfive.types import Session, Task


class AwesomeAdapter:
    """Adapter for awesome_agent_sdk."""

    def __init__(
        self,
        *,
        system_prompt: str,
        available_agents: list[str] | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._tools: list[ReportingToolSpec] = []
        self._available_agents = available_agents or ["default"]
        self._steerer = None  # injected lazily by the executor

    async def register_reporting_tools(
        self,
        tools: list[ReportingToolSpec],
    ) -> None:
        self._tools = tools

    @property
    def available_agents(self) -> list[str]:
        return self._available_agents

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        client = AwesomeClient(
            system_prompt=self._render_prompt(task, session),
            tools=[self._wrap_tool(spec, session) for spec in self._tools],
        )

        final_text, final_stop = "", ""
        async for block in client.run(user_message=task.description):
            if block.type == "text":
                final_text += block.text
                # feed to steerer for drift classification
                if self._steerer is not None:
                    await self._steerer.observe(block, session)
            elif block.type == "tool_use":
                # reporting tools are handled by the tool's handler;
                # nothing to do here. Non-reporting tool calls would
                # flow through the framework's own tool-call path.
                if self._steerer is not None:
                    await self._steerer.observe(block, session)
            elif block.type == "stop":
                final_stop = block.stop_reason

        return InvocationResult(
            task_id=task.id,
            text=final_text,
            stop_reason=final_stop,
        )

    def _render_prompt(self, task: Task, session: Session) -> str:
        completed = "\n".join(
            f"- {tid}: {summary}"
            for tid, summary in session.completed_results.items()
        )
        return (
            f"{self._system_prompt}\n\n"
            f"Current task: {task.title}\n"
            f"Description: {task.description}\n"
            f"Task id: {task.id}\n"
            f"Completed so far:\n{completed or '(nothing yet)'}\n"
        )

    def _wrap_tool(self, spec: ReportingToolSpec, session: Session):
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            # Route through ``invoke_tool`` — NOT ``spec.handler`` direct —
            # so the schema / terminal / loop / volume-cap guards fire.
            # Skipping this is the pre-#108 bug class.
            return await invoke_tool(
                self._tools, spec.name, args, session, self._steerer
            )

        return ToolDef(
            name=spec.name,
            parameters_schema=spec.parameters,
            handler=handler,
        )
```

Use it:

```python
from goldfive import Runner
from goldfive.executors.sequential import SequentialExecutor
from goldfive.planner import StaticPlanner

runner = Runner(
    agent=AwesomeAdapter(system_prompt="You are a helpful assistant."),
    planner=StaticPlanner(my_plan),
    executor=SequentialExecutor(),
)
outcome = await runner.run("do the thing")
```

That's it. ~50 lines, one new framework wrapped, full goldfive
semantics on top.

## Just wrap the root agent (multi-agent trees)

When a framework supports nested agents — ADK's `sub_agents`, agent-as-tool
wrappers, or custom "inner agent" composites — the adapter must coordinate
**the entire tree**, not just the root. A sub-agent that is missing the
reporting tools cannot report task outcomes, and the steerer will never see
its state transitions.

The rule for goldfive adapters: **the caller wraps the root agent; the
adapter handles subtree propagation itself.** Users should never have to
attach reporting tools by hand on every node.

`ADKAdapter` walks the agent graph at wrap time to **register
reporting tools** on every reachable agent, and separately to
**collect agent names** for `available_agents`. Both walks follow
three edges:

- `agent.sub_agents` — native ADK child agents.
- `agent.inner_agent` — wrapper agents that compose a single child.
- `tool.agent` for each tool in `agent.tools` — agents exposed to a parent
  via `AgentTool` (agent-as-tool).

Every node that carries a mutable `tools` list gets the reporting tools
appended. The walk is idempotent: agents that already carry the canonical
reporting tool names are skipped, so double-registration is a no-op.

Because ADK's plugin callbacks are **runner-scoped**, not agent-scoped,
the `before_tool_callback` / `before_model_callback` installed once on a
`Runner` fires for tool calls from every sub-agent of that runner's root.
Under the single-Runner model (goldfive#130) `ADKAdapter` constructs
**one runner** around the tree root and installs the goldfive
plugin on it. ADK propagates the plugin manager into any
`AgentTool`-spawned sub-Runners automatically, so the plugin's
callbacks fire across the full delegation tree without per-agent
re-registration.

```python
from goldfive.adapters.adk import ADKAdapter

# Tree: root -> child -> grandchild, plus a sibling agent-as-tool.
adapter = ADKAdapter(root_agent)
# adapter.available_agents is sorted and includes every reachable
# agent name. The planner populates task.assignee_agent_id as a
# delegation hint; invoke() drives the one runner regardless.
```

If you are writing an adapter for a different framework with its own
nested-agent shape, mirror this pattern:

1. Walk the agent graph from the root you were handed to attach
   reporting tools (dedupe by name) and collect advisory names.
2. Install a single shared intercept (plugin / middleware / callback) on
   the framework's outermost runtime so every sub-agent's tool calls route
   through one handler map.
3. In `invoke(task, session)`, drive the root / entry-point once;
   delegation within the tree is the framework's job.

See `goldfive/adapters/adk.py::_augment_subtree_with_reporting` and
`::_collect_reachable_agent_names` for the reference implementation.

## The steerer reference

In the example above, `self._steerer` is accessed but never set. The
`AgentAdapter` protocol deliberately does not mandate how the steerer
reaches the adapter — different frameworks need different wiring:

- **CallableAdapter** — the shipped reference adapter does not bind a
  steerer at all. Reporting-tool handlers close over the steerer
  (the `Runner` stamps them in via `register_reporting_tools`) and
  the callable forwards the tool list to user code.
- **ClaudeAgentSDKAdapter** — exposes `bind_steerer(steerer)`
  explicitly so the adapter can install `before_model` / observer
  hooks against it.
- **ADKAdapter** — the executor / plugin injects the steerer through
  ADK's own plugin callbacks; the adapter doesn't need a setter.

If your framework wants per-turn drift observation, expose a public
`bind_steerer` (or accept the `Steerer` at construction) and have
the executor hand in the `Steerer` before the first `invoke`.
Otherwise, rely on the closure that `register_reporting_tools`
establishes: the reporting-tool handlers already carry a reference
to the steerer wired up by the `Runner`.

## Mapping drift-relevant signals

Your adapter should forward the following signals to
`steerer.observe()`:

| Framework signal | Why it matters |
|---|---|
| LLM text chunk / block | Refusal phrases detected by `classify_refusal`. |
| Non-reporting tool call with error | `TOOL_ERROR` drift. |
| Agent transfer / delegation event | `AGENT_TRANSFER`, potentially `WRONG_AGENT`. |
| Final stop reason | `CONTEXT_PRESSURE`, `TOO_MANY_STEPS`, `STOPPED_EARLY`. |
| Unhandled exception in agent invocation | `TASK_FAILED_FATAL` if unrecoverable. |

The [DRIFT.md](../design/DRIFT.md) taxonomy lists every kind and
what the steerer looks for. You do not need to classify — just forward.

## Optional: streaming observations

If your framework supports streaming, you can call
`steerer.observe()` on every chunk for real-time drift detection. If
it doesn't, call `observe()` once with the final response. Both are
legal; streaming gives the steerer an earlier chance to detect issues
like context pressure.

## Testing your adapter

The `CallableAdapter`'s tests (in `tests/test_callable_adapter.py`)
are the template. The pattern:

1. Construct a fake framework response (a canned iterator, a mock
   client).
2. Plug it into your adapter.
3. Run a single `invoke()` against a one-task plan.
4. Assert the steerer saw the expected transitions and drift.
5. Assert the `InvocationResult` carries the expected fields.

For integration tests, use the adapter with a `DefaultSteerer` and
`InMemorySink`, run a full plan, and inspect the event stream.

## Publishing your adapter

Adapters can live outside the goldfive repo. Publish as a separate
package that depends on `goldfive` and whatever framework it wraps.
The `@runtime_checkable` `AgentAdapter` protocol means no inheritance
or registration is required — any class with the right methods works.

If you want to contribute an adapter upstream, open an issue
describing the framework and its shape; goldfive is receptive to
merging adapters when the framework has momentum and the adapter is
tested.
