# Writing an agent adapter

goldfive is agent-framework-agnostic. The adapter is the one piece of
the system that knows about a specific framework — ADK, Claude Agent
SDK, LangGraph, an MCP server, a bare async callable, anything.

This guide walks through writing a new `AgentAdapter` for a framework
goldfive doesn't ship. Reference v0.1 adapters:

- `goldfive.adapters.callable.CallableAdapter` — the simplest form.
  Start here when prototyping a new adapter.
- `goldfive.adapters.adk.ADKAdapter` — ports harmonograf's ADK plugin.
- `goldfive.adapters.claude.ClaudeAgentSDKAdapter` — wraps Anthropic's
  Claude Agent SDK.

Related: [PROTOCOLS.md](../design/PROTOCOLS.md#agentadapter),
[tool-protocol.md](../reference/tool-protocol.md).

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
```

Three responsibilities:

1. **Register reporting tools.** The steerer hands you a list of
   `ReportingToolSpec`s (see [tool-protocol.md](../reference/tool-protocol.md)).
   Translate each spec into the tool shape your framework expects and
   install a hook that routes calls to the spec's handler.
2. **Invoke the agent for one task.** Render current-task context,
   run the agent, stream observed events to the steerer, return an
   `InvocationResult`.
3. **Expose `available_agents`.** The names the planner can use as
   `task.assignee_agent_id`.

## The four things every adapter must do

### Render task context

The agent needs to know what task it's working on. Different
frameworks expose this differently:

- **ADK** — write `session.state["goldfive.current_task_id"]` and
  friends in `before_model_callback`; the agent reads them.
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
3. Invoke the spec's `handler(args, session, steerer)`.
4. Return the handler's return value as the tool result.

The handler is where the steerer applies the state transition.
`{"acknowledged": True}` is the conventional response body.

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
            return await spec.handler(args, session, self._steerer)

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
from goldfive.planner import PassthroughPlanner

runner = Runner(
    agent=AwesomeAdapter(system_prompt="You are a helpful assistant."),
    planner=PassthroughPlanner(plan=my_plan),
    executor=SequentialExecutor(),
)
outcome = await runner.run("do the thing")
```

That's it. ~50 lines, one new framework wrapped, full goldfive
semantics on top.

## The steerer reference

In the example above, `self._steerer` is accessed but never set. In
the real adapters, the executor injects the steerer via a protocol
method or a framework-specific hook. For v0.1, the shape is under
active iteration; check `goldfive.adapters.callable.CallableAdapter`
for the current idiom and mirror it.

A robust pattern:

```python
class AwesomeAdapter:
    def bind_steerer(self, steerer):
        self._steerer = steerer
```

The executor calls `adapter.bind_steerer(steerer)` before the first
`invoke`. The `CallableAdapter` in `goldfive/adapters/callable.py` is
the reference implementation.

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
