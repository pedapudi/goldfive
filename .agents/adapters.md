---
name: adapters
description: The AgentAdapter protocol — what it is, how to implement one, and what ships in-box.
applies-when: ["write an adapter", "wrap a framework", "AgentAdapter contract"]
---

# Adapters

An `AgentAdapter` is the seam between goldfive's executor and whatever
agent runtime actually does the work. It is how goldfive stays
framework-agnostic.

## Contract

From `goldfive/protocols.py`:

```python
from typing import Protocol, runtime_checkable

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

Three members. That's it.

- **`register_reporting_tools`** — called once per run, between
  planning and execution. You get the seven canonical reporting tool
  specs (see [events.md](events.md)) and must surface them to the
  underlying framework so the agent can call them during `invoke`.
- **`invoke`** — called per task. Run the agent against `task`, using
  the registered tools to report state changes. Return an
  `InvocationResult` with the final text and/or artifacts.
- **`available_agents`** — stable list of agent identifiers this
  adapter can dispatch to. Planners consult this so they don't generate
  tasks for non-existent agents.

## Shipped implementations

| Adapter | Module | Extra |
|---|---|---|
| `CallableAdapter` | `goldfive.adapters.callable` | (core) |
| `ADKAdapter` | `goldfive.adapters.adk` | `goldfive[adk]` |
| `ClaudeAgentSDKAdapter` | `goldfive.adapters.claude` | `goldfive[claude]` |

`CallableAdapter` is the reference implementation and the preferred
vehicle for deterministic tests — it wraps an async callable of
shape `(task, session, tools) -> InvocationResult` and forwards tools
verbatim.

## Writing a new adapter

Skeleton:

```python
from __future__ import annotations

from goldfive.reporting import ReportingToolSpec
from goldfive.results import InvocationResult
from goldfive.types import Session, Task


class MyFrameworkAdapter:
    def __init__(self, underlying_agent) -> None:
        self._agent = underlying_agent
        self._tools: list[ReportingToolSpec] = []

    async def register_reporting_tools(
        self, tools: list[ReportingToolSpec]
    ) -> None:
        self._tools = list(tools)
        # Translate each ReportingToolSpec into whatever your framework
        # wants. For an SDK with an `add_tool(name, schema, fn)` API:
        # for spec in tools:
        #     self._agent.add_tool(spec.name, spec.parameters, spec.handler)

    async def invoke(
        self, task: Task, session: Session
    ) -> InvocationResult:
        # Dispatch to the framework. The agent's generated tool calls
        # should land on the handlers you registered above.
        text = await self._agent.run(task.description, tools=self._tools)
        return InvocationResult(task_id=task.id, text=text)

    @property
    def available_agents(self) -> list[str]:
        return ["my-framework-agent"]
```

## Key design constraints

- **Tool handlers are `async`** and have signature
  `(args: dict, session: Session, steerer: Steerer) -> dict`. When the
  framework calls `handler`, it must pass the *live* `Session` and the
  steerer bound by the Runner — `CallableAdapter` short-circuits this
  by exposing the raw specs and letting the callable drive them; ADK
  and Claude adapters do the wiring in native tool-call form.
- **Return cleanly OR call a terminal tool. Don't half-do both.**
  `SequentialExecutor` auto-completes a task that's still `PENDING` or
  `RUNNING` when `invoke` returns. If the agent already called
  `report_task_completed` / `_failed` / `_blocked` / `_cancelled`, the
  auto-complete is a no-op. But if the agent transitioned the task
  mid-flight to a non-terminal state without a terminal call, the
  executor keeps re-invoking until `max_task_invocations` (if set) or
  the per-task-lineage cap trips.
- **`invoke` must not crash on well-formed input.** If the underlying
  framework raises, catch and return `InvocationResult(..., error=...)`
  so the executor can surface it as a `TaskFailed`.
- **Thread an `available_agents` list that matches what planners will
  emit.** If your adapter wraps a sub-agent tree, enumerate every
  routable leaf.

## Quick reference

Minimal test adapter for integration tests:

```python
from goldfive import CallableAdapter, InvocationResult

async def agent(task, session, tools):
    return InvocationResult(task_id=task.id, text=f"did {task.title}")

adapter = CallableAdapter(agent, available_agents=["worker"])
```

## Common pitfalls

- `register_reporting_tools` is a no-op → the agent never sees the
  reporting tools → no state transitions → every task auto-completes
  but nothing meaningful happens.
- Framework's tool calls land on a handler you didn't wire to the
  goldfive handler → state stays `PENDING` forever.
- `available_agents` is empty → planners assume no routable agents
  and either emit tasks with a default assignee or refuse to plan.
- Swallowing framework exceptions inside `invoke` → task stays
  `RUNNING`; either return `error=...` or let the executor catch it.

## Related

- [events.md](events.md) — the seven reporting tools the adapter surfaces.
- [docs/guides/writing-an-agent-adapter.md](../docs/guides/writing-an-agent-adapter.md) — the full prose guide.
- [docs/design/PROTOCOLS.md](../docs/design/PROTOCOLS.md) — protocol contracts with minimal implementations.
- [docs/reference/tool-protocol.md](../docs/reference/tool-protocol.md) — canonical tool schemas.
