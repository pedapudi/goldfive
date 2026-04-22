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

## One-line wrapping

For the common case — "I already have an agent, wrap it with goldfive
planning" — two helpers skip the hand-wired `Runner` construction:

```python
import goldfive

# One line: wrap + run, get an ExecutionOutcome back.
outcome = await goldfive.run(my_agent, "make a presentation about waffles")

# Or, keep the Runner around:
runner = goldfive.wrap(my_agent, sinks=[my_sink])
outcome = await runner.run("make a presentation about waffles")
```

`wrap` auto-detects the adapter from the agent's shape:

| Agent shape | Adapter chosen |
|---|---|
| Implements `goldfive.AgentAdapter` | passed through as-is |
| `google.adk.agents.BaseAgent` (or an ADK `Runner`) | `ADKAdapter` (requires `goldfive[adk]`) |
| Zero-arg factory returning `claude_agent_sdk.ClaudeSDKClient` | `ClaudeAgentSDKAdapter` (requires `goldfive[claude]`) |
| `async (task, session, tools) -> InvocationResult` | `CallableAdapter` |

It also tries to reuse the agent's LLM. For ADK agents, `wrap` walks
the agent tree, finds the first `.model`, and wires a matching
`LLMPlanner` + `LLMGoalDeriver`. For non-ADK agents you can supply
`call_llm=` / `model=` yourself; if neither is provided, `wrap`
degrades to `PassthroughPlanner` + `LiteralGoalDeriver` and logs a
DEBUG line explaining the drop.

Every default is overridable via keyword argument — `planner=`,
`goal_deriver=`, `executor=`, `steerer=`, `sinks=`, and
`max_task_invocations=` all win over the auto-wiring when passed.

## Mental model in five sentences

goldfive wraps an agent and **observes** it running against an explicit plan.

1. Your user input becomes one or more **Goals**.
2. A **Planner** produces a **Plan** (a DAG of **Tasks**) that, if
   executed, satisfies the goals.
3. For ADK trees the **Executor** issues ONE invocation with the user
   request and lets the tree run naturally; a **PlanReconciler** maps
   the tree's observed `before_agent` / `after_agent` callbacks onto
   plan tasks (the "overlay" model, goldfive#141). For non-ADK adapters
   it falls back to the per-task invoke loop.
4. A **Steerer** watches execution and classifies **drift** — structured
   "we're off-track" signals — then routes through a six-level
   intervention ladder (OBSERVE → ABSORB → NUDGE → CANCEL_REINVOKE →
   PAUSE_ESCALATE → TERMINATE, goldfive#142).
5. On drift at Level 1+ the planner refines the plan; every state change
   emits a proto **Event** to every configured **EventSink**.

Six primitives, one `Runner` object, one `await runner.run(...)` call.

## Using goldfive with an ADK agent (the common path)

If you already have a Google ADK agent tree — single `Agent`, flat
specialists, coordinator+`AgentTool`, or deep hierarchies — the whole
integration is:

```python
import goldfive
from google.adk.apps.app import App

wrapped = goldfive.wrap(root_agent)            # ADK BaseAgent subclass
app = App(name="my-demo", root_agent=wrapped)  # works in adk web
```

`goldfive.wrap` on an ADK agent:

1. Builds one `InMemoryRunner` around the tree root (single-Runner
   model, goldfive#130). ADK handles delegation within the tree via
   `AgentTool` / `sub_agents` / `transfer_to_agent`.
2. Walks the tree and attaches the goldfive reporting-tools to every
   reachable agent (`sub_agents` / `inner_agent` / `tool.agent` edges).
3. Attaches a `GoldfivePlanner(BasePlanner)` to every `LlmAgent` in
   the tree (goldfive#153). This injects a per-turn orchestration
   context block into the LLM's system instruction (current task id /
   title, plan summary, active user steer, goal summaries) via the
   goldfive ADK plugin's `before_model_callback`.
4. Installs the goldfive ADK plugin on the runner. It observes tool
   calls, function responses, agent transitions, and LLM request /
   response pairs; routes drift to the steerer; heals orphan tool-call
   ids on USER_STEER cancellation.

The pipeline runs naturally inside `adk web` — the wrapped object IS
a `BaseAgent` (`GoldfiveADKAgent`, see
[adk-web-integration.md](adk-web-integration.md)) — and the same
object still works programmatically:

```python
outcome = await wrapped.run("make a presentation about waffles")
```

See [adk-web-integration.md](adk-web-integration.md) for end-to-end
setup including environment variables and the `HarmonografTelemetryPlugin`
pairing.

## Hello-callable: your first agent in 10 minutes

We'll build the simplest possible goldfive run: a three-task plan,
driven by an async Python callable pretending to be an agent, with an
in-memory event sink you can inspect at the end.

### Step 1 — Write the agent callable

Create `hello.py`:

```python
from __future__ import annotations

import asyncio

from goldfive import (
    CallableAdapter,
    DefaultSteerer,
    InMemorySink,
    InvocationResult,
    Plan,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
)


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

async def main() -> None:
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(my_agent, available_agents=["default"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        steerer=DefaultSteerer(),
        sinks=[sink],
    )
    outcome = await runner.run("demo run")
    await runner.close()

    print(f"success={outcome.success}, tasks={len(outcome.session.plan.tasks)}")
    for event in sink.events:
        kind = event.WhichOneof("payload")
        print(f"  [{event.sequence:2d}] {kind}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2 — Run it

```bash
uv sync --extra proto
uv run python hello.py
```

Expected output:

```
success=True, tasks=3
  [ 0] run_started
  [ 1] goal_derived
  [ 2] plan_submitted
  [ 3] task_started
  [ 4] task_completed
  [ 5] task_started
  [ 6] task_completed
  [ 7] task_started
  [ 8] task_completed
  [ 9] run_completed
```

Every event is a proto `Event` with a `oneof payload` you extract via
`WhichOneof("payload")`. Sequences are a monotonic per-run counter
produced by `Session.next_sequence()`.

That's it. You have a full goldfive run, observable via the in-memory
sink, walking a 3-task plan end-to-end.

### Step 3 — Inspect what happened

Everything you'd want to know is in `sink.events`. Every entry is a
proto `Event`; dispatch on the `oneof payload`:

```python
for event in sink.events:
    kind = event.WhichOneof("payload")
    if kind == "task_completed":
        tc = event.task_completed
        print(f"task {tc.task_id}: {tc.summary}")
```

For a run that was persisted to disk, use `JSONLPersistenceSink` or
`SQLitePersistenceSink` — see
[persistence-and-recovery.md](persistence-and-recovery.md) and
[choosing-a-sink.md](choosing-a-sink.md).

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

def make_client():
    from claude_agent_sdk import ClaudeSDKClient
    return ClaudeSDKClient(...)  # your client config

agent = ClaudeAgentSDKAdapter(
    client_factory=make_client,
    system_prompt_template="You are a helpful assistant.",
    model="claude-opus-4-5-20251101",
)
```

The rest of the `Runner` construction is identical. That is the core
payoff of goldfive: framework-agnostic orchestration over whatever
agent primitive you have.

## Using an LLM planner instead

`PassthroughPlanner` is great for demos because you hand it a plan
up-front. For real workloads, let the LLM decompose. If you already
have a `call_llm` async binding, hand it to `goldfive.wrap` — it
builds the `LLMPlanner` + `LLMGoalDeriver` pair for you:

```python
import goldfive

async def call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    # wire this to your LLM of choice.
    ...

runner = goldfive.wrap(
    my_agent,
    call_llm=call_llm,
    model="claude-opus-4-5-20251101",
)
outcome = await runner.run("build me a slide deck about Python")
```

Or the explicit form with `Runner(...)`:

```python
from goldfive import CallableAdapter, LLMPlanner, Runner, SequentialExecutor

runner = Runner(
    agent=CallableAdapter(my_agent, available_agents=["default"]),
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
from goldfive import JSONLPersistenceSink

runner = Runner(
    agent=CallableAdapter(my_agent, available_agents=["default"]),
    planner=StaticPlanner(plan),
    executor=SequentialExecutor(),
    sinks=[JSONLPersistenceSink(path=f"./runs/{run_id}.jsonl")],
)
```

For cross-run queries, swap in `SQLitePersistenceSink` (or pair
both). Full failure-mode walkthrough in
[persistence-and-recovery.md](persistence-and-recovery.md); full sink
matrix in [choosing-a-sink.md](choosing-a-sink.md).

## Live steering — pause, cancel, redirect

Long runs benefit from an operator who can reach in and steer
mid-flight. goldfive ships a single primitive for this: a
`ControlChannel` passed into `Runner(control=...)`. Here is the
smallest possible PAUSE / RESUME demo, still against the three-task
`hello.py` plan:

```python
import asyncio

from goldfive import ControlChannel, ControlKind, ControlMessage


async def drive(channel: ControlChannel) -> None:
    # Give the run a moment, then PAUSE, then RESUME.
    await asyncio.sleep(0.05)
    await channel.send(ControlMessage(kind=ControlKind.PAUSE))
    await asyncio.sleep(0.1)
    await channel.send(ControlMessage(kind=ControlKind.RESUME))


async def main() -> None:
    channel = ControlChannel()
    runner = Runner(
        agent=CallableAdapter(my_agent, available_agents=["default"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        control=channel,
    )
    # Run the agent and the controller concurrently.
    _, outcome = await asyncio.gather(
        drive(channel),
        runner.run("demo run"),
    )
    await runner.close()
    channel.close()
    print(f"success={outcome.success}")
```

Beyond PAUSE / RESUME, `ControlKind` supports `CANCEL` (abort the run),
`STEER` (redirect via planner refine), `REWIND_TO` (reset a task and
its downstream), and `APPROVE` / `REJECT` (resolve pending approvals).
Full semantics and the end-to-end UI path are in
[../design/CONTROL.md](../design/CONTROL.md). A runnable demo covering
PAUSE, STEER, and CANCEL against an offline canned-LLM planner is at
[`examples/live_steering.py`](../../examples/live_steering.py).

## Running in parallel

Swap `SequentialExecutor` for `ParallelDAGExecutor`:

```python
from goldfive import ParallelDAGExecutor

runner = Runner(
    agent=CallableAdapter(my_agent, available_agents=["default"]),
    planner=StaticPlanner(plan),
    executor=ParallelDAGExecutor(max_concurrency=4),
)
```

Tasks whose dependencies are satisfied run concurrently via
`asyncio.gather`. Refine is deferred to stage boundaries.

## What's next

- [observability-with-harmonograf.md](observability-with-harmonograf.md) —
  end-to-end: boot the harmonograf UI and watch this same run animate.
- [writing-an-agent-adapter.md](writing-an-agent-adapter.md) — wrap a
  new framework.
- [writing-an-event-sink.md](writing-an-event-sink.md) — send events
  to your own observability backend.
- [choosing-a-sink.md](choosing-a-sink.md) — picking between the five
  shipped sinks.
- [goals-and-plans.md](goals-and-plans.md) — author custom
  GoalDerivers and Planners.
- [persistence-and-recovery.md](persistence-and-recovery.md) — crash
  recovery with `JSONLPersistenceSink` / `SQLitePersistenceSink`.
- [grpc-transport.md](grpc-transport.md) — streaming events to an
  out-of-process observer.
- [../design/CONTROL.md](../design/CONTROL.md) — live steering and the
  control channel.
- [troubleshooting.md](troubleshooting.md) — common install / run-time
  failures.
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
