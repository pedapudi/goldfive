# goldfive

**Stay on target.**

goldfive is a small, framework-agnostic Python library that wraps an agent
with the orchestration scaffolding most agents quietly need: an explicit
**goal**, a **plan** broken into tasks, per-turn **drift analysis**, and a
**steering** loop that nudges the agent back on course when it wanders.

It does not ship an LLM client, a prompt DSL, or a tool registry. It wraps
whatever agent runtime you already use (Google ADK, the Anthropic SDK, a
plain callable, ...) behind a narrow `AgentAdapter` protocol and gives you:

- a `Runner` that drives the agent turn by turn against a `Goal`
- pluggable `Planner`, `DriftAnalyzer`, and `Steerer` components
- a `TelemetrySink` stream of structured events you can log, render, or
  ship to an observability console

goldfive is the orchestration half of
[harmonograf](https://github.com/pedapudi/harmonograf), extracted so you
can use the control loop without the console.

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

```python
import asyncio

from goldfive import (
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
)


async def greeter(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    text = {
        "greet": "Hello!",
        "ask": "What's your name?",
        "thank": "Thanks for chatting!",
    }.get(task.id, "")
    return InvocationResult(task_id=task.id, text=text)


async def main() -> None:
    plan = Plan(
        id="hello",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="greet", title="Greet", assignee_agent_id="greeter"),
            Task(id="ask", title="Ask for a name", assignee_agent_id="greeter"),
            Task(id="thank", title="Thank the user", assignee_agent_id="greeter"),
        ],
        edges=[
            TaskEdge(from_task_id="greet", to_task_id="ask"),
            TaskEdge(from_task_id="ask", to_task_id="thank"),
        ],
        summary="Greet, ask, thank.",
    )

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(greeter, available_agents=["greeter"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        sinks=[sink],
    )

    outcome = await runner.run("run the greeter workflow")
    await runner.close()

    print(f"success={outcome.success} events={len(sink.events)}")


asyncio.run(main())
```

A runnable copy of this example lives in
[`examples/hello_callable.py`](examples/hello_callable.py).

## Docs

**Start with [`docs/guides/getting-started.md`](docs/guides/getting-started.md)** —
install, run your first goldfive-wrapped agent in about ten minutes,
inspect the event stream. Concrete and runnable.

### Design

- [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md) — overview of the six primitives, how they compose, full lifecycle.
- [`docs/design/PROTOCOLS.md`](docs/design/PROTOCOLS.md) — the six protocol contracts with minimal implementations.
- [`docs/design/STATE-MACHINE.md`](docs/design/STATE-MACHINE.md) — task lifecycle state diagram, transition rules, invariants.
- [`docs/design/DRIFT.md`](docs/design/DRIFT.md) — full drift-kind taxonomy (25+), classification rules, refine policy.
- [`docs/design/EVENT-MODEL.md`](docs/design/EVENT-MODEL.md) — proto event taxonomy, sequence semantics, `EventSink` contract.

### Guides

- [`docs/guides/getting-started.md`](docs/guides/getting-started.md) — install + first agent.
- [`docs/guides/writing-an-agent-adapter.md`](docs/guides/writing-an-agent-adapter.md) — wrap a new framework.
- [`docs/guides/writing-an-event-sink.md`](docs/guides/writing-an-event-sink.md) — build a custom sink.
- [`docs/guides/goals-and-plans.md`](docs/guides/goals-and-plans.md) — authoring custom `GoalDeriver` / `Planner`.
- [`docs/guides/persistence-and-recovery.md`](docs/guides/persistence-and-recovery.md) — JSONL persistence + `Runner.resume()`.
- [`docs/guides/harmonograf-integration.md`](docs/guides/harmonograf-integration.md) — plugging harmonograf in as a sink.

### Reference

- [`docs/reference/api.md`](docs/reference/api.md) — public API surface.
- [`docs/reference/tool-protocol.md`](docs/reference/tool-protocol.md) — the seven reporting tools.

## License

Apache-2.0.
