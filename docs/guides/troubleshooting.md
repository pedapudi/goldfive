# Troubleshooting

Things that break on first install or first run. If you hit something
that isn't here, file an issue or check
[getting-started.md](getting-started.md).

## Contents

- [Install / setup](#install--setup)
- [Running your first agent](#running-your-first-agent)
- [Events and sinks](#events-and-sinks)
- [harmonograf integration](#harmonograf-integration)
- [Proto and types](#proto-and-types)
- [See also](#see-also)

## Install / setup

### Symptom: `ImportError: cannot import name 'LoggingSink'`

**What you see.**

```
ImportError: cannot import name 'LoggingSink' from 'goldfive.sinks'
```

Same shape for `JSONLPersistenceSink`, `SQLitePersistenceSink`, `GRPCSink`.

**Why it happens.** These sinks depend on the optional `proto` extra
(`grpcio`, `grpcio-tools`, `mypy-protobuf`) and `google.protobuf`.
`goldfive/sinks/__init__.py` wraps each optional import in `try/except
ImportError` and sets the name to `None` when the extra is missing.

**Fix.**

```bash
uv sync --extra proto
# or
pip install 'goldfive[proto]'
```

### Symptom: `ModuleNotFoundError: goldfive.pb.goldfive.v1.events_pb2`

**What you see.**

```
ModuleNotFoundError: goldfive protobuf stubs not available;
generate them via `make proto` ...
```

**Why it happens.** The generated proto modules live under
`goldfive/pb/goldfive/v1/` and are required by any sink that serialises
events. A fresh clone includes them; `make clean` or a slim install
removes them.

**Fix.**

```bash
uv sync --extra proto && make proto
```

### Symptom: `TypeError: 'NoneType' object is not callable` on a sink class

**What you see.**

```python
from goldfive.sinks import JSONLPersistenceSink
sink = JSONLPersistenceSink("./log.jsonl")
# TypeError: 'NoneType' object is not callable
```

**Why it happens.** Same root cause as the first entry — the `proto`
extra is missing, so the module attribute is `None` rather than the
class. The top-level package stays importable so the rest of the API
keeps working.

**Fix.** Install the extra, or guard explicitly:

```python
from goldfive.sinks import JSONLPersistenceSink
if JSONLPersistenceSink is None:
    raise RuntimeError("install goldfive[proto] to enable JSONL persistence")
```

### Symptom: `ImportError: cannot import name 'StrEnum' from 'enum'`

**What you see.** An `ImportError` or `SyntaxError` during
`import goldfive`.

**Why it happens.** goldfive targets Python 3.11+. `StrEnum` landed in
the stdlib in 3.11.

**Fix.**

```bash
python --version   # must be 3.11 or newer
uv python install 3.12 && uv sync --python 3.12
```

## Running your first agent

### Symptom: `outcome.success == False` with `reason == "no plan generated"`

**What you see.** The Runner returns immediately with that exact
`reason` before any task fires.

**Why it happens.** The planner returned `None`. Typical causes:
`PassthroughPlanner` (no-op by design), `StaticPlanner` built with
`Plan(tasks=[])`, or `LLMPlanner` whose `call_llm` raised or returned
non-JSON (the error is caught and swallowed into `None`).
`LLMPlanner.generate` also short-circuits to `None` when `goals` is
empty.

**Fix.** Use a planner that produces tasks:

```python
from goldfive import Runner, SequentialExecutor, CallableAdapter, StaticPlanner
from goldfive.types import Plan, Task

plan = Plan(
    id="demo", run_id="", goal_ids=["g1"],
    tasks=[Task(id="t1", title="do the thing", assignee_agent_id="worker")],
    edges=[],
)
runner = Runner(
    agent=CallableAdapter(my_agent, available_agents=["worker"]),
    planner=StaticPlanner(plan),
    executor=SequentialExecutor(),
)
```

For `LLMPlanner`, enable `logging.DEBUG` on `goldfive.planner` to see
the raw response before it is discarded.

### Symptom: `outcome.reason` starts with `"planner.generate raised:"`

**Why it happens.** `planner.generate` raised. The Runner catches it,
emits `RunAborted`, and surfaces the exception text. `LLMPlanner`
itself catches `call_llm` exceptions internally and returns `None`
(which gives you `"no plan generated"` instead), so this reason
almost always points at a custom planner.

**Fix.**

```python
import logging
logging.getLogger("goldfive.runner").setLevel(logging.DEBUG)
```

The full trace is logged at `ERROR` on `goldfive.runner`.

### Symptom: plan tasks never start

**What you see.** `RunStarted` and `PlanSubmitted` fire, but no
`TaskStarted` ever does. The run terminates with reason
`"exhausted max_task_invocations=... with pending task <id>"` or
completes with zero tasks run.

**Why it happens.** The executor only picks tasks whose predecessors
are all `COMPLETED`. Two common shapes:

1. Dependency cycle — `Plan.topological_stages()` tolerates cycles but
   leaves the cyclic tasks un-ready, so `_pick_next_task` never
   returns them.
2. `assignee_agent_id` on every task sits outside
   `adapter.available_agents`. The adapter still receives the invoke
   call; it is the adapter that then fails or no-ops.

**Fix.** Print the DAG and cross-check against the adapter:

```python
for stage in plan.topological_stages():
    print([t.id for t in stage])

missing = {t.assignee_agent_id for t in plan.tasks} - set(adapter.available_agents)
assert not missing, f"plan references unknown agents: {missing}"
```

### Symptom: adapter returned but the task is still `RUNNING`

**What you see.** The adapter calls `report_task_started`, does work,
returns — but the task never reaches `COMPLETED`. The executor loops
until `max_task_invocations` trips.

**Why it happens.** `SequentialExecutor` auto-transitions a task to
`COMPLETED` (or `FAILED` if `InvocationResult.error` is set) only when
the task is in `PENDING` or `RUNNING` on return. If the adapter
transitioned the task to `BLOCKED` or `CANCELLED` mid-invocation, or
it silently returned after only calling `report_task_started`, the
auto-complete gate still applies — but if some other branch moved the
task to a non-terminal state the executor keeps re-invoking. The
contract is: return cleanly and let the executor finish the task, OR
call a terminal reporting tool yourself. Don't half-do both.

**Fix.** Either rely on auto-complete (return, don't call reporting
tools) or ensure every code path ends with `report_task_completed` /
`report_task_failed`.

### Symptom: adapter never sees the reporting tools

**What you see.** Inside the adapter, the `tools` list is empty — or
the underlying agent has no `report_task_*` functions exposed.

**Why it happens.** `Runner.run` calls
`agent.register_reporting_tools(BUILTIN_REPORTING_TOOLS)` after
planning and before execution. Your adapter must store the list and
forward it to the underlying framework. `CallableAdapter` forwards
them as the `tools=` argument on every `invoke`; the bundled ADK and
Claude adapters wire them into their native tool registration step. A
custom adapter with a no-op `register_reporting_tools` never exposes
them.

**Fix.** Implement `register_reporting_tools` and forward the specs.
See [writing-an-agent-adapter.md](writing-an-agent-adapter.md).

## Events and sinks

### Symptom: sink registered but never receives events

**What you see.** An `InMemorySink` is on `Runner(sinks=[...])`, but
`sink.events` is empty after the run.

**Why it happens.**

1. Sink passed somewhere other than `Runner(sinks=[...])` — only that
   list is plumbed to the executor and steerer.
2. The process exited before `await runner.close()` ran. Buffered
   sinks (`GRPCSink`, `HarmonografSink`) flush during `close`.
3. The run aborted before any task emissions; you should still see
   `RunStarted` and the terminal `RunAborted` on the sink.

**Fix.**

```python
runner = Runner(..., sinks=[my_sink])
try:
    outcome = await runner.run(user_input)
finally:
    await runner.close()
```

### Symptom: `GRPCSink` appears to drop events

**What you see.** The upstream server sees fewer events than expected.

**Why it happens.** `GRPCSink` only forwards objects with a proto
`DESCRIPTOR` attribute. Since #55 every goldfive component emits
proto, so the full lifecycle crosses the wire — but a custom sink
or caller that hands a dict (e.g. built via
`goldfive.events.make_event`) to `GRPCSink.emit` has it silently
dropped with a debug log. See
[grpc-transport.md](grpc-transport.md#proto-only).

**Fix.** Either ensure the upstream component emits proto, or pair
`GRPCSink` with a local `JSONLPersistenceSink` as a durable
fallback:

```python
runner = Runner(..., sinks=[
    JSONLPersistenceSink("./runs/current.jsonl"),
    GRPCSink("observer.internal:50051"),
])
```

### Symptom: `JSONLPersistenceSink` writes zero lines

**Why it happens.** Either `JSONLPersistenceSink` is `None` because
the `proto` extra is missing (and `Runner(sinks=[None])` tolerates it
silently), or the process exited before `await runner.close()` ran.
The sink is line-buffered and flushes per-write, so partial writes
are rare; lost output is almost always a missed `close`.

**Fix.**

```python
from goldfive.sinks import JSONLPersistenceSink
assert JSONLPersistenceSink is not None, "install goldfive[proto]"
sink = JSONLPersistenceSink("./runs/current.jsonl")
runner = Runner(..., sinks=[sink])
try:
    outcome = await runner.run(user_input)
finally:
    await runner.close()
```

## adk-web integration

### Symptom: Run terminates immediately with "goldfive run complete."

**What you see.** You open adk-web, submit a prompt, and the stream
shows a plan summary followed instantly by "goldfive run complete."
— nothing actually ran. Sometimes a bare `AttributeError:
'_async_httpx_client'` shows up in the server log at teardown.

**Why it happens.** Almost always the model is misconfigured. Two
common shapes:

1. **Gemini default with no `GOOGLE_API_KEY`.** `presentation_agent_orchestrated`
   and similar examples default `USER_MODEL_NAME` to `gemini-2.5-flash`;
   without credentials the first LLM call raises after goldfive has
   already emitted RunStarted / PlanSubmitted, so the stream looks
   short but ended.
2. **Custom planner returned `None` from `generate`** — e.g.
   `LLMPlanner` with a `call_llm` that raised on first use. The
   Runner emits `RunAborted` with `reason="no plan generated"` and
   the UI renders it as "goldfive run complete." with no tasks.

**Fix.**

```bash
export USER_MODEL_NAME=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...
# or for local LLMs:
export USER_MODEL_NAME=openai/qwen3-coder-30b
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=sk-anything
```

Then inspect the underlying issue:

```python
import logging
logging.getLogger("goldfive.runner").setLevel(logging.DEBUG)
logging.getLogger("goldfive.planner").setLevel(logging.DEBUG)
```

The full exception trace is logged at `ERROR` on `goldfive.runner`.

### Symptom: Spans don't appear per-agent in the harmonograf Gantt

**What you see.** The Gantt timeline shows one big bar per turn
instead of a row per sub-agent.

**Why it happens.** You're on a goldfive / harmonograf combination
that predates per-agent span stamping (goldfive#170 +
harmonograf#80). Or the `HarmonografTelemetryPlugin` didn't install
on the App-level runner — goldfive's in-process plugin sees its own
observations but not the ADK-native spans the telemetry plugin adds.

**Fix.**

```python
from google.adk.apps.app import App
from harmonograf_client import Client, HarmonografTelemetryPlugin

client = Client(name="my-agent", server_addr="127.0.0.1:7531")
app = App(
    name="my-demo",
    root_agent=goldfive.wrap(root_agent),
    plugins=[HarmonografTelemetryPlugin(client)],  # <— this line
)
```

The plugin must be on the `App`; putting it on `goldfive.wrap(plugins=...)`
also works but the App-level path is idempotent and doesn't require
that goldfive construct the runner.

### Symptom: Steer doesn't take effect

**What you see.** You click Steer in the harmonograf UI, the
`DriftDetected{kind=user_steer}` event appears, but no `PlanRevised`
follows and the run keeps running against the old plan.

**Possible causes.**

1. **`planner.refine` silently failed.** Look for
   `DriftDetected{kind=refine_validation_failed, severity=critical}`
   — that's the terminal signal from `LLMPlanner` when it exhausts
   its retry budget. See [common-failure-modes.md §3](common-failure-modes.md).
2. **Annotation not propagated.** STEER annotations are deduped by
   `annotation_id` (goldfive#171). If two clicks have the same id
   the second is a no-op. Check the harmonograf client's bridge is
   forwarding a fresh id per click.
3. **Author missing.** The refine's revised plan should carry the
   steering author in `revision_reason`. If it's empty the bridge
   didn't forward `ControlMessage.steer.author`.

**Fix.** Enable DEBUG on `goldfive.planner` and re-steer; the
`LLMPlanner._refine_user_steer: attempt X/2: <error>` log line tells
you exactly which attempt failed and why.

### Symptom: Multiple session rows in harmonograf per run

**What you see.** Every goldfive run produces two (or three) rows in
the harmonograf Sessions picker — one has the plan, another has the
spans, and they refer to the same underlying run.

**Why it happens.** Pre-goldfive#161 / #164 + pre-harmonograf#85
(lazy Hello). Goldfive's `Session.id` and the ADKAdapter's internal
session id disagreed with adk-web's outer `ctx.session.id`; per-event
routing split the stream across multiple sessions.

**Fix.** Upgrade both goldfive (≥ #164) and harmonograf (≥ #85). In
current `main` `GoldfiveADKAgent._run_async_impl` pins the outer
session id onto both goldfive and the adapter before sub-agent
dispatch runs, and harmonograf's client defers the Hello RPC until
the first event — so every span + event carries the same session id.

If you're still seeing duplicates on current code, check for a
pre-built `Runner` constructed outside `goldfive.wrap` (degrade mode)
— the pin only works when the wrap path built the runner.

## harmonograf integration

### Symptom: `HarmonografSink` — connection refused

**What you see.**

```
grpc._channel._InactiveRpcError: ... Connection refused (127.0.0.1:7531)
```

**Why it happens.** The harmonograf server is not running. The default
demo address is `127.0.0.1:7531`.

**Fix.**

```bash
cd ~/git/harmonograf
make demo          # server + frontend + adk web
# or just the server
make server-run
```

### Symptom: harmonograf frontend shows no sessions

**Why it happens.** The client buffered events never drained.
`HarmonografSink.close` only marks the sink closed — it does not flush
the wire. The underlying `Client` owns the transport and must be shut
down separately. Kill the process between the last emit and
`client.shutdown()` and everything in the buffer is lost.

**Fix.** Two-step shutdown, in order:

```python
runner = Runner(..., sinks=[HarmonografSink(client)])
try:
    outcome = await runner.run(user_input)
finally:
    await runner.close()                 # 1. flush goldfive side
    client.shutdown(flush_timeout=5.0)   # 2. flush wire buffer
```

### Symptom: `ModuleNotFoundError: harmonograf_client`

**Why it happens.** The client library ships from the harmonograf
repo, not from goldfive.

**Fix.**

```bash
uv pip install harmonograf-client
# or
git clone https://github.com/pedapudi/harmonograf.git
cd harmonograf && make install
```

### Symptom: events arrive but the plan diff doesn't render

**What you see.** `PlanRevised` is in the timeline but the UI shows no
before/after diff banner.

**Why it happens.** The UI only renders a diff when both the old and
new plan snapshots are present. A `PlanRevised` that re-uses the prior
plan's `revision_index` leaves the UI with nothing to compare.
This is a harmonograf UI limitation, not a goldfive bug.

**Fix.** Ensure `refine` increments `Plan.revision_index` on every
revision — the bundled `LLMPlanner.refine` does this automatically.
Custom planners should stamp
`revised.revision_index = plan.revision_index + 1`.

## Proto and types

### Symptom: `isinstance(evt, Event)` is `False` on a proto event

**Why it happens.** Two `Event` classes are in scope — usually one
from `goldfive.pb.goldfive.v1.events_pb2` and a duplicate copy a
downstream project generated into its own package. Python sees them as
distinct types even though the wire format is identical.

**Fix.** Follow the harmonograf pattern: do not re-emit goldfive's
proto messages into your package. Depend on `goldfive[proto]` and
import `goldfive.pb.goldfive.v1.events_pb2.Event` directly. If you
must re-run `protoc`, pass goldfive's proto directory via
`--proto_path=` and write stubs under `goldfive/pb/` so the
namespace-package graft in `goldfive/pb/__init__.py` collapses both
sources. See harmonograf's `Makefile` (`proto-python` target) for the
reference recipe.

### Symptom: `TypeError` when building a `Task` or `Plan`

**What you see.**

```
TypeError: __init__() got an unexpected keyword argument '<proto field>'
TypeError: Plan.__init__() missing 1 required positional argument: 'run_id'
```

**Why it happens.** `goldfive.Task`, `Plan`, `Goal`, `Session`,
`DriftEvent`, and `TaskEdge` are `@dataclasses.dataclass` types, not
proto messages. Their fields do not line up with the proto wire format
verbatim.

**Fix.** Use the dataclass kwargs:

```python
from goldfive.types import Plan, Task, TaskEdge

plan = Plan(
    id="demo", run_id="", goal_ids=["g1"],
    tasks=[
        Task(id="research", title="research"),
        Task(id="draft", title="draft", assignee_agent_id="writer"),
    ],
    edges=[TaskEdge(from_task_id="research", to_task_id="draft")],
)
```

Convert between dataclass and proto via `goldfive.conv.to_pb_plan` /
`goldfive.conv.from_pb_plan` and siblings.

## See also

- [getting-started.md](getting-started.md) — clone, install, first run.
- [choosing-a-sink.md](choosing-a-sink.md) — picking between
  `InMemorySink`, `JSONLPersistenceSink`, `SQLitePersistenceSink`,
  `GRPCSink`, and `HarmonografSink`.
- [grpc-transport.md](grpc-transport.md) — wire format, reconnect
  semantics, proto-only behaviour.
- [harmonograf-integration.md](harmonograf-integration.md) — pairing
  goldfive with the harmonograf console.
- [writing-an-event-sink.md](writing-an-event-sink.md) — implementing
  a custom sink.
- [persistence-and-recovery.md](persistence-and-recovery.md) — JSONL
  format, replay, and `Runner.resume`.
