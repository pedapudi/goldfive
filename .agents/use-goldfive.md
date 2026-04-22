---
name: use-goldfive
description: Wrap an existing agent with goldfive's planning, drift, and event stream — minimal install-to-first-run path.
applies-when: ["wrap my agent", "use goldfive", "add orchestration", "goldfive quickstart"]
---

# Use goldfive to wrap an agent

You have an agent (ADK, Claude SDK, or a plain async callable) and want
goldfive to plan, drive, observe, and steer it. This skill is the
shortest correct path.

## Install

```bash
uv add goldfive           # recommended
# or
pip install goldfive
```

Optional extras, install only what you need:

- `goldfive[adk]` — Google ADK adapter.
- `goldfive[claude]` — Claude Agent SDK adapter.
- `goldfive[proto]` — enables `LoggingSink`, `JSONLPersistenceSink`,
  `SQLitePersistenceSink`, `GRPCSink`. Required any time you want
  events on the wire or on disk.
- `goldfive[examples]` — runtime deps for scripts in `examples/`.

## The one-liner (`goldfive.run` / `goldfive.wrap`)

For ADK agents and bare callables alike, two convenience helpers:

```python
import goldfive

# One line: wrap + run, get an ExecutionOutcome back.
outcome = await goldfive.run(my_agent, "make a presentation about waffles")

# Or, keep the Runner around:
runner = goldfive.wrap(my_agent, sinks=[my_sink])
outcome = await runner.run("make a presentation about waffles")
```

`wrap` auto-detects the adapter from the agent's shape (ADK
BaseAgent, Claude SDK client factory, async callable, or an existing
`AgentAdapter` instance); tries to reuse the agent's LLM for the
planner + goal-deriver; and defaults the executor to
`SequentialExecutor(overlay_mode=True)` for ADK wrap targets
(goldfive#141).

For the tightest possible Runner construction with an explicit
`StaticPlanner`, `goldfive.quickstart` is still available:

```python
import asyncio
from goldfive import InvocationResult, quickstart


async def agent(task, session, tools):
    return InvocationResult(task_id=task.id, text=f"did {task.title}")


async def main() -> None:
    runner = quickstart(agent, "Say hello, ask a name, thank them")
    outcome = await runner.run("go")
    await runner.close()
    print(f"success={outcome.success}")


asyncio.run(main())
```

`quickstart` wires a `CallableAdapter`, `StaticPlanner` (one task per
goal), `SequentialExecutor`, and a single `InMemorySink`. Pass
`planner=` or `sinks=` to override.

## The explicit form (`Runner`)

Reach for this when you want a custom plan, a specific executor, or
multiple sinks.

```python
import asyncio

from goldfive import (
    CallableAdapter, InMemorySink, InvocationResult, Plan, ReportingToolSpec,
    Runner, SequentialExecutor, Session, StaticPlanner, Task, TaskEdge,
)


async def worker(
    task: Task, session: Session, tools: list[ReportingToolSpec]
) -> InvocationResult:
    return InvocationResult(task_id=task.id, text=f"did {task.title}")


async def main() -> None:
    plan = Plan(
        id="demo", run_id="", goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="gather", assignee_agent_id="worker"),
            Task(id="t2", title="draft",  assignee_agent_id="worker"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        summary="linear demo",
    )
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(worker, available_agents=["worker"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        sinks=[sink],
    )
    outcome = await runner.run("go")
    await runner.close()
    print(f"success={outcome.success} events={len(sink.events)}")


asyncio.run(main())
```

## Swapping in a real framework

For ADK, use `goldfive.wrap(adk_agent)` directly — it returns a
`GoldfiveADKAgent` that both satisfies the `BaseAgent` contract (so
`adk web` loads it) and exposes `Runner.run`:

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
# `adk web agent.py` now loads wrapped; programmatic use:
#   outcome = await wrapped.run("make a presentation about waffles")
```

For the Claude Agent SDK:

```python
# Claude Agent SDK (requires goldfive[claude])
from goldfive.adapters.claude import ClaudeAgentSDKAdapter
agent = ClaudeAgentSDKAdapter(
    system_prompt="You are a helpful assistant.",
    model="claude-sonnet-4-5",
)
```

See [adapters.md](adapters.md) for the contract and
[docs/guides/writing-an-agent-adapter.md](../docs/guides/writing-an-agent-adapter.md)
for wrapping a new framework.

## Observability

To see every event live, pair goldfive with harmonograf. The full
ten-minute walkthrough (install both, boot the stack, stream events
to the console) is in
[docs/guides/observability-with-harmonograf.md](../docs/guides/observability-with-harmonograf.md).

To log events from a test or script without harmonograf, use
`LoggingSink` or `InMemorySink` — see [sinks.md](sinks.md).

## Quick reference

```python
from goldfive import (
    # entry points
    Runner, quickstart,
    # adapters
    CallableAdapter,
    # planners
    StaticPlanner, PassthroughPlanner, LLMPlanner,
    # executors
    SequentialExecutor, ParallelDAGExecutor,
    # sinks
    InMemorySink, LoggingSink, JSONLPersistenceSink, SQLitePersistenceSink, GRPCSink,
    # results / types
    InvocationResult, ExecutionOutcome, Plan, Task, TaskEdge, Goal,
)
```

Always call `await runner.close()`. Buffered sinks (`GRPCSink`,
harmonograf) flush there.

## Common pitfalls

- Forgot `await runner.close()` — buffered sinks drop events on exit.
- `StaticPlanner(Plan(tasks=[]))` → outcome has `reason="no plan generated"`.
- `Task.assignee_agent_id` doesn't match anything in
  `adapter.available_agents` → the executor invokes the adapter anyway
  and the agent gets an unroutable task.
- Using a `proto`-gated sink without the extra — the class is `None`,
  not raising. Guard with `assert LoggingSink is not None`.

## Related

- [adapters.md](adapters.md) — adapter contract.
- [sinks.md](sinks.md) — sink contract and shipped implementations.
- [debug-goldfive.md](debug-goldfive.md) — when something breaks.
- [docs/guides/getting-started.md](../docs/guides/getting-started.md) — full prose walkthrough.
