---
name: how-to-add-a-new-adapter
description: Step-by-step for wrapping a new agent framework as a goldfive AgentAdapter — the protocol, the invoke_tool dispatch contract, and the architectural invariants that must hold.
applies-when: ["add adapter", "wrap framework", "new agent runtime", "AgentAdapter"]
---

# Add a new `AgentAdapter`

goldfive is framework-agnostic. The adapter is the one place that
knows about a specific framework (ADK, Claude SDK, a bare callable,
LangGraph, MCP, anything). This skill is the checklist for writing
a new one. See the longer-form guide at
[docs/guides/writing-an-agent-adapter.md](../docs/guides/writing-an-agent-adapter.md)
for the full worked example; this file is the terse checklist.

## The protocol contract

Live definition: `goldfive/protocols.py::AgentAdapter`.

```python
@runtime_checkable
class AgentAdapter(Protocol):
    async def register_reporting_tools(
        self, tools: list[ReportingToolSpec]
    ) -> None: ...
    async def invoke(
        self, task: Task, session: Session
    ) -> InvocationResult: ...
    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",
    ) -> None: ...
    @property
    def available_agents(self) -> list[str]: ...
```

The protocol is `@runtime_checkable`; a duck-typed implementation
passes `isinstance(x, AgentAdapter)`.

## Checklist

### 1. Register reporting tools

Store the full list of `ReportingToolSpec` on the adapter instance.
**Do not** translate to just a name-to-handler map — you need the
full specs to feed `invoke_tool`, which runs the guard layers.

```python
async def register_reporting_tools(self, tools):
    self._tools: list[ReportingToolSpec] = list(tools)
    # Translate each spec into the framework's tool shape (ADK
    # FunctionTool, Claude SDK inline tool, MCP tool, …).
    ...
```

Reference impls:

- ADK: `goldfive/adapters/adk.py::ADKAdapter.register_reporting_tools` — plus `_augment_subtree_with_reporting` walks `sub_agents` / `inner_agent` / `tool.agent` so every descendant is wired.
- Claude SDK: `goldfive/adapters/claude.py::ClaudeAgentSDKAdapter.register_reporting_tools`.
- Callable: `goldfive/adapters/callable.py::CallableAdapter.register_reporting_tools`.

### 2. Route every tool call through `invoke_tool`

This is the non-negotiable. The adapter's tool-invocation hook
(ADK `before_tool_callback`, Claude SDK handler wrapper, your
framework's equivalent) MUST route through:

```python
from goldfive.adapters._tool_invocation import invoke_tool

ack = await invoke_tool(
    self._tools, name, args, session, self._steerer
)
```

`invoke_tool` runs four guard layers before calling
`spec.handler`:

1. **Schema rejection** — missing / unknown `task_id` for a
   task-scoped tool is rejected with a structured error before
   any counter update.
2. **Terminal-task rejection** — calls on COMPLETED / FAILED /
   CANCELLED tasks return `task_already_terminal`.
3. **Per-task loop guard** — duplicate-args returns `duplicate`;
   sustained bursts or volume caps flip the `(task, tool)` bucket
   into a hard-reject state.
4. **Session-wide volume cap** — > 50 calls of the same tool name
   across all tasks in a session flags the tool session-wide.

Calling `spec.handler` directly bypasses all four. Pre-#108 that
was the root cause of the filler-loop class of bugs: the state
machine advanced to COMPLETED, but the adapter kept re-invoking
the agent until the framework's own ceiling tripped.

### 3. Hook drift observation

Non-reporting-tool events — LLM text, transfer events, stop
reasons, tool errors — are fed to the steerer as raw observations:

```python
if self._steerer is not None:
    await self._steerer.observe(event, session)
```

The steerer classifies; you forward. `DefaultSteerer.detect_drift`
is the classifier.

For chain-of-thought surfaces (OpenAI `reasoning_content`,
Anthropic `thinking` blocks, Google thought parts), implement
`emit_reasoning` and call the steerer:

```python
async def emit_reasoning(
    self, text, *, task=None, session, provider="", call_id=""
):
    if self._steerer is None or not text.strip():
        return
    await self._steerer.observe_reasoning(
        text, task=task, session=session, provider=provider
    )
```

Adapters that cannot surface reasoning simply never call this; the
reasoning-drift pipeline degrades to pattern-only detectors.

### 4. Render current-task context for the agent

Frameworks differ:

- **ADK** — write `session.state["goldfive.current_task_id"]`
  etc. in `before_model_callback`; the agent reads them.
- **Claude SDK** — inject task context into the first user
  message or the system prompt. See
  `goldfive/adapters/_claude_prompt.py`.
- **Custom callable** — pass as a function argument.

Either approach works; pick whichever matches the framework's idiom.
Required bits: `task.id`, `task.title`, `task.description`, the plan
summary, and the summary of every completed task
(`session.completed_results`).

### 5. Return an `InvocationResult`

```python
return InvocationResult(
    task_id=task.id,
    text=final_assistant_text,
    stop_reason=framework_stop_reason,  # "end_turn", "max_tokens", …
    error=exc_if_the_framework_raised_else_None,
    raw=the_raw_framework_response_object,
)
```

The executor uses `stop_reason` for `CONTEXT_PRESSURE` /
`TOO_MANY_STEPS` classification. It uses `error` to route through
`steerer.mark_task_failed` when the agent didn't report a terminal
state itself.

### 6. Expose `available_agents`

A property that lists every agent the adapter can dispatch to.
Consumed by the planner to populate `task.assignee_agent_id`. For
nested-agent trees (ADK `sub_agents`, agent-as-tool wrappers), walk
the tree and return every name.

## The architectural invariant

**Framework-to-framework handoff must not cross an SDK-managed
state boundary without the SDK's own copy semantics being taken
into account.** The concrete failures this rule prevents:

- Storing `SessionContext` on `session.state` in ADK and expecting
  it to survive the turn. ADK deep-copies its session state between
  turns; mutations are silently dropped. The fix is to bind state
  to the adapter/plugin instance (a Python reference that survives),
  not to `session.state` (an SDK-owned copy).
- Relying on a `ContextVar` set in one coroutine being visible in
  another coroutine that the SDK dispatched via
  `loop.run_in_executor` / `asyncio.to_thread`. Every framework
  handles this differently; don't assume.
- Assuming the handler registered with the SDK at construction
  time is the handler that runs when the tool is called. Some
  frameworks bind by name, not by reference.

If you are writing an adapter for a framework that offers more
than one way to install a tool-dispatch hook, pick the one that
(a) survives the SDK's own state-copy boundary and (b) exposes the
raw tool name + args payload, not a pre-parsed surface.

See the "structural vs symptomatic" heuristic in
[debug-goldfive.md §"Structural vs symptomatic debugging"](debug-goldfive.md)
for the postmortem that extracted this invariant.

## `InvocationStateStore` (tracked)

Issue #117 proposes a shared `InvocationStateStore` primitive on
the adapter base — a small object owned by the adapter that
bundles (session, steerer, task, tool_specs, pending call-ids) for
one `invoke()` call and is explicitly thread-agnostic. When that
lands, new adapters will inherit the store rather than re-rolling
the ContextVar / plugin-instance dance each time. Until then,
every adapter handles its own wiring; follow the existing adapters'
pattern of keeping a reference on `self` that the hook closes over.

## Multi-agent trees

When the framework supports nested agents, wrap the root and walk
the tree yourself — do not require callers to attach reporting
tools to every child. Reference: `ADKAdapter` walks
`agent.sub_agents`, `agent.inner_agent`, and every `tool.agent`
for `tool in agent.tools`. Idempotent: agents that already carry
the canonical reporting-tool names are skipped.

## Testing

Reference tests: `tests/test_callable_adapter.py`,
`tests/test_claude_adapter.py`, `tests/test_adk_adapter.py`. At
minimum assert:

- `isinstance(adapter, AgentAdapter)` — protocol-compliant.
- `register_reporting_tools` accepts an empty list without crashing.
- `invoke` calls the spec's `handler` via `invoke_tool` (a
  terminal-status task must receive `task_already_terminal`).
- Exceptions from the wrapped agent surface as
  `InvocationResult.error`, not as raises out of `invoke`.

## Related

- [docs/guides/writing-an-agent-adapter.md](../docs/guides/writing-an-agent-adapter.md) — long-form walkthrough with worked 50-line example.
- [docs/design/PROTOCOLS.md §AgentAdapter](../docs/design/PROTOCOLS.md#agentadapter) — contract.
- [docs/reference/tool-protocol.md](../docs/reference/tool-protocol.md) — the seven reporting tools.
- [adapters.md](adapters.md) — the existing adapters skill.
- [debug-goldfive.md](debug-goldfive.md) — triage when an adapter misbehaves.
