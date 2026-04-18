# Getting started with goldfive

This is the hands-on entry point. Clone, install, and run your first
goldfive-wrapped agent in about ten minutes. Every snippet is
self-contained and runs against v0.1.

If you want the motivation and design context first, start with
[ARCHITECTURE.md](../design/ARCHITECTURE.md).

## Prerequisites

| Tool | Minimum | Notes |
|---|---|---|
| Python | 3.11 | `StrEnum` is required. |
| [`uv`](https://github.com/astral-sh/uv) | recent | Primary package manager. `pip` works but is untested. |
| `git` | any | For cloning. |

No LLM credentials are required for this guide. The "hello-callable"
walkthrough uses a scripted Python callable as the agent.

## Install

```bash
git clone https://github.com/pedapudi/goldfive.git
cd goldfive
uv sync
```

Verify the import surface:

```bash
uv run python -c "from goldfive import Runner; print(Runner)"
```

You should see `<class 'goldfive.runner.Runner'>`.

Optional extras (only install what you need):

```bash
uv sync --extra adk       # google-adk integration
uv sync --extra claude    # claude-agent-sdk integration
uv sync --extra dev       # dev tools (pytest, ruff, mypy)
uv sync --extra examples  # example-specific deps
```

## Mental model in five sentences

goldfive wraps an agent and runs it against an explicit plan.

1. Your user input becomes one or more **Goals**.
2. A **Planner** produces a **Plan** (a DAG of **Tasks**) that, if
   executed, satisfies the goals.
3. An **Executor** walks the plan. For each task, it calls your
   **AgentAdapter**, which drives whatever underlying framework
   (Claude, ADK, a bare callable) actually runs the agent.
4. A **Steerer** watches the execution and classifies **drift** —
   structured "we're off-track" signals.
5. On drift, the planner revises the plan; every state change emits a
   proto **Event** to every configured **EventSink**.

Six primitives, one `Runner` object, one `await runner.run(...)` call.

## Hello-callable: your first agent in 10 minutes

We'll build the simplest possible goldfive run: a three-task plan,
driven by an async Python callable pretending to be an agent, with an
in-memory event sink you can inspect at the end.

### Step 1 — Write the agent callable

Create `hello.py`:

```python
from __future__ import annotations

import asyncio

from goldfive import Runner
from goldfive.adapters.callable import CallableAdapter
from goldfive.executors.sequential import SequentialExecutor
from goldfive.planner import StaticPlanner
from goldfive.results import InvocationResult
from goldfive.sinks import InMemorySink
from goldfive.steerer import DefaultSteerer
from goldfive.types import Plan, Session, Task, TaskEdge


# -- the "agent" --------------------------------------------------------
#
# A CallableAdapter expects an async callable with this shape:
#   (task, session, tools) -> InvocationResult
#
# `tools` is the list of reporting-tool specs the adapter registered.
# In a real agent, this is where the LLM would be invoked and its
# `report_task_*` tool calls intercepted.  The SequentialExecutor
# auto-announces RUNNING before `invoke` and auto-completes the task
# after a clean return, so a toy agent can just return a result.

async def my_agent(task: Task, session: Session, tools) -> InvocationResult:
    _ = tools  # unused in this toy agent
    return InvocationResult(task_id=task.id, text=f"did {task.title}")


# -- the plan -----------------------------------------------------------
#
# A linear 3-task plan: t1 → t2 → t3.

plan = Plan(
    id="p1",
    run_id="",  # Runner stamps this at dispatch time
    goal_ids=["g1"],
    tasks=[
        Task(id="t1", title="gather context", assignee_agent_id="default"),
        Task(id="t2", title="draft response", assignee_agent_id="default"),
        Task(id="t3", title="polish output", assignee_agent_id="default"),
    ],
    edges=[
        TaskEdge(from_task_id="t1", to_task_id="t2"),
        TaskEdge(from_task_id="t2", to_task_id="t3"),
    ],
    summary="3-step linear demo",
)


# -- the Runner ---------------------------------------------------------

def event_kind(event) -> str:
    # Runner emits dict events; executors/steerers emit proto Event
    # messages whose payload is a oneof. Handle both shapes.
    if isinstance(event, dict):
        return event.get("kind", "?")
    return event.WhichOneof("payload") or "?"


def event_sequence(event) -> int:
    if isinstance(event, dict):
        return int(event.get("sequence", 0))
    return int(getattr(event, "sequence", 0))


async def main() -> None:
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(my_agent),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        steerer=DefaultSteerer(),
        sinks=[sink],
    )
    outcome = await runner.run("demo run")

    print(f"success={outcome.success}, tasks={len(outcome.session.plan.tasks)}")
    for event in sink.events:
        print(f"  [{event_sequence(event):2d}] {event_kind(event)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2 — Run it

```bash
uv run python hello.py
```

Expected output:

```
success=True, tasks=3
  [ 0] RunStarted
  [ 1] GoalDerived
  [ 2] PlanSubmitted
  [ 3] run_started
  [ 4] task_started
  [ 5] task_completed
  [ 6] task_started
  [ 7] task_completed
  [ 8] task_started
  [ 9] task_completed
  [10] run_completed
```

Sequences 0–2 come from the Runner (it emits dict envelopes with a
``kind`` field) and the rest come from the executor and steerer (proto
``Event`` messages whose oneof payload is matched by ``WhichOneof``).
You'll see a second ``run_started`` at sequence 3 because the
executor also announces its start — treat the two together as the
run's start marker.

That's it. You have a full goldfive run, observable via the in-memory
sink, walking a 3-task plan end-to-end.

### Step 3 — Inspect what happened

Everything you'd want to know is in `sink.events`. The executor and
steerer emit proto ``Event`` messages — use ``WhichOneof`` to pick out
the payload, skipping the dict envelopes the Runner emits:

```python
for event in sink.events:
    if isinstance(event, dict):
        continue  # Runner envelope: RunStarted / GoalDerived / PlanSubmitted
    kind = event.WhichOneof("payload")
    if kind == "task_completed":
        tc = event.task_completed
        print(f"task {tc.task_id}: {tc.summary}")
```

For a run that was persisted to disk, use `JSONLPersistenceSink` —
see [persistence-and-recovery.md](persistence-and-recovery.md).

## Wiring a real agent

The `CallableAdapter` is the simplest `AgentAdapter`: it just hands
your callable the reporting tools and lets the callable do whatever it
wants. For real agents, use:

- **`CallableAdapter`** — you want full control (most custom harnesses).
- **`ClaudeAgentSDKAdapter`** — you're using Anthropic's Claude Agent
  SDK. Install with `uv sync --extra claude`.
- **`ADKAdapter`** — you're using Google's Agent Development Kit.
  Install with `uv sync --extra adk`.

Swapping adapters is a one-line change:

```python
# was:
# agent=CallableAdapter(my_agent),

# now:
from goldfive.adapters.claude import ClaudeAgentSDKAdapter
agent=ClaudeAgentSDKAdapter(
    system_prompt="You are a helpful assistant.",
    model="claude-opus-4-5-20251101",
)
```

The rest of the `Runner` construction is identical. That is the core
payoff of goldfive: framework-agnostic orchestration over whatever
agent primitive you have.

## Using an LLM planner instead

`PassthroughPlanner` is great for demos because you hand it a plan
up-front. For real workloads, let the LLM decompose:

```python
from goldfive.planner import LLMPlanner

async def call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    # wire this to your LLM of choice.
    ...

runner = Runner(
    agent=CallableAdapter(my_agent),
    planner=LLMPlanner(call_llm=call_llm, model="claude-opus-4-5-20251101"),
    executor=SequentialExecutor(),
)
outcome = await runner.run("build me a slide deck about Python")
```

See [goals-and-plans.md](goals-and-plans.md) for the planner prompt
format and how to customize it.

## Adding a persistence sink

One line to make runs crash-recoverable:

```python
from goldfive.sinks import JSONLPersistenceSink

runner = Runner(
    agent=CallableAdapter(my_agent),
    planner=PassthroughPlanner(plan),
    executor=SequentialExecutor(),
    sinks=[JSONLPersistenceSink(path=f"./runs/{run_id}.jsonl")],
)
```

Full failure-mode walkthrough in
[persistence-and-recovery.md](persistence-and-recovery.md).

## Running in parallel

Swap `SequentialExecutor` for `ParallelDAGExecutor`:

```python
from goldfive.executors.parallel import ParallelDAGExecutor

runner = Runner(
    agent=CallableAdapter(my_agent),
    planner=PassthroughPlanner(plan),
    executor=ParallelDAGExecutor(max_concurrency=4),
)
```

Tasks whose dependencies are satisfied run concurrently via
`asyncio.gather`. Refine is deferred to stage boundaries.

## What's next

- [writing-an-agent-adapter.md](writing-an-agent-adapter.md) — wrap a
  new framework.
- [writing-an-event-sink.md](writing-an-event-sink.md) — send events
  to your own observability backend.
- [goals-and-plans.md](goals-and-plans.md) — author custom
  GoalDerivers and Planners.
- [persistence-and-recovery.md](persistence-and-recovery.md) — crash
  recovery with `JSONLPersistenceSink`.
- [tool-protocol.md](../reference/tool-protocol.md) — the seven
  reporting tools that drive task state.
- [ARCHITECTURE.md](../design/ARCHITECTURE.md) — the full design
  reference.

## Troubleshooting

**`ModuleNotFoundError: goldfive.adapters.claude`.** The Claude SDK
adapter is gated behind the `claude` extra. `uv sync --extra claude`.

**`ImportError: google.adk`.** Same idea for ADK:
`uv sync --extra adk`. Plus you need Google's `adk-python` checked
out somewhere your Python can find it.

**Run hangs indefinitely.** Most common cause: your agent callable
never calls `report_task_completed`. The executor waits for every
task to reach a terminal state. Either fix the agent or add a
timeout and use a drift-driven refine.

**Every sink saw an event with sequence 0 twice.** Almost certainly a
bug in a custom executor that called `Session.next_sequence()` from
inside `asyncio.gather` without serialization. Serialize the call.
See [EVENT-MODEL.md](../design/EVENT-MODEL.md#sequence-semantics).
