# Observability with harmonograf

End-to-end walkthrough: install goldfive, boot a local harmonograf
server and frontend, run the bundled
[`examples/harmonograf_observed`](../../examples/harmonograf_observed/)
agent, and watch its events animate the Gantt UI in real time. Target
wall-clock: about ten minutes on a warm machine.

This guide picks up where [getting-started.md](getting-started.md)
leaves off. That guide gets you a working goldfive install with an
`InMemorySink`. This one adds the console. For the design rationale
and the wire-format contract, read
[harmonograf-integration.md](harmonograf-integration.md).

No LLM credentials are required. The example uses a scripted
`CallableAdapter` as the agent.

## Architecture at a glance

```
┌─────────────────────────────┐          ┌──────────────────────────────┐
│       your agent process    │          │        harmonograf           │
│                             │          │                              │
│   goldfive.Runner           │          │   server (gRPC :7531)        │
│     ├─► InMemorySink        │          │   frontend (Vite :5173)      │
│     └─► HarmonografSink ────┼──────────┼─► gRPC StreamEvents          │
│           │                 │          │       │                      │
│           └─ harmonograf_   │          │       └─► SQLite + pub/sub   │
│              client.Client  │          │           └─► frontend bus   │
│                             │          │                              │
└─────────────────────────────┘          └──────────────────────────────┘
```

The wire surface is goldfive's `goldfive.v1.Event` proto. Every
orchestration state change on the left becomes one proto message
shipped to harmonograf, which persists it and pushes a delta into the
browser. For the per-event UI mapping see
[harmonograf-integration.md](harmonograf-integration.md#what-harmonograf-renders-per-event).

## Prerequisites

| Tool | Minimum | Notes |
|---|---|---|
| Python | 3.11 | `StrEnum` is required. |
| [`uv`](https://github.com/astral-sh/uv) | recent | Primary package manager for both repos. |
| `git` | any | For cloning goldfive and harmonograf. |
| Node | 20 | Required by the harmonograf Vite frontend. |
| `pnpm` | recent | The frontend uses `pnpm` with a frozen lockfile. |

No LLM credentials are required. No cloud accounts. Everything runs
on localhost.

## Step 1 — Install goldfive

```bash
git clone https://github.com/pedapudi/goldfive.git
cd goldfive
uv sync --extra proto
```

The `proto` extra pulls in `grpcio` and `protobuf`. Since #55 every
goldfive event is a proto `Event`, so the extra is required for the
`LoggingSink` / `JSONLPersistenceSink` / `SQLitePersistenceSink` /
`GRPCSink` / `HarmonografSink` paths.

Verify the import surface:

```bash
uv run python -c "from goldfive import Runner, quickstart; print(Runner, quickstart)"
```

For the zero-observability walkthrough, see
[getting-started.md](getting-started.md). The rest of this guide
assumes you have goldfive installed here.

## Step 2 — Install and start harmonograf

Harmonograf lives in a separate repo. Clone it next to goldfive (any
path works; `~/git/harmonograf` is a common choice).

```bash
cd ..
git clone https://github.com/pedapudi/harmonograf.git
cd harmonograf
```

Harmonograf's `make install` target installs three components — the
Python server, the Python client library, and the Vite frontend. It
also expects Google's `adk-python` checked out at
`third_party/adk-python/` as an editable path dep (only required for
the ADK-driven demo; still needed for `make install` to succeed).

```bash
git clone https://github.com/google/adk-python.git third_party/adk-python
make install
```

Two options for starting the server:

- `make server-run` — boots only the Python gRPC server (`127.0.0.1:7531`).
  The frontend is not started.
- `make demo` — boots the server, the frontend (`127.0.0.1:5173`), and
  an `adk web` process hosting the bundled `presentation_agent`
  (`127.0.0.1:8080`). Useful for the full ADK rollout, not required
  for this guide.

For the minimum stack the goldfive example needs, open one terminal
and start the server plus the frontend:

```bash
# terminal 1
make server-run

# terminal 2
cd frontend && pnpm dev --port 5173 --strictPort
```

Ports in use:

- `127.0.0.1:7531` — server gRPC listener. Goldfive's
  `HarmonografSink` connects here.
- `127.0.0.1:7532` — server gRPC-Web listener. The frontend connects
  here.
- `127.0.0.1:5173` — the frontend itself.

For the full harmonograf install walkthrough, including the ADK-driven
`make demo` path, see
[harmonograf/docs/quickstart.md](https://github.com/pedapudi/harmonograf/blob/main/docs/quickstart.md).

## Step 3 — Install the harmonograf client into your goldfive env

The example imports `harmonograf_client.Client` and
`harmonograf_client.HarmonografSink`. `harmonograf-client` is a
workspace member of the harmonograf repo, not a PyPI package — install
it editable from your local clone.

```bash
cd ../goldfive
uv pip install -e ../harmonograf/client
```

Verify:

```bash
uv run python -c "from harmonograf_client import Client, HarmonografSink; print(Client, HarmonografSink)"
```

If the import fails with `ModuleNotFoundError: harmonograf_client`,
the install above pointed at the wrong path — the package lives under
`harmonograf/client/`, not `harmonograf/`.

For the sink matrix (which sinks solve which problem), see
[choosing-a-sink.md](choosing-a-sink.md).

## Step 4 — Run the observed example

The repo ships a ready-to-run example at
[`examples/harmonograf_observed/agent.py`](../../examples/harmonograf_observed/agent.py).
It builds a four-task `StaticPlanner` (`research` → `draft` →
`review` → `publish`), wires the goldfive `Runner` to both an
`InMemorySink` and a `HarmonografSink`, and tears both down cleanly
at the end.

With the server running at `127.0.0.1:7531` (from step 2):

```bash
uv run python examples/harmonograf_observed/agent.py
```

Expected output:

```text
success=True, reason=''
run_id=<32-hex-char uuid>
sinks=[InMemorySink, HarmonografSink -> 127.0.0.1:7531]
InMemorySink captured 12 events.
Open the harmonograf UI (server at 127.0.0.1:7531) to inspect the plan, task timeline, and drift markers for this run.
```

Twelve events is correct: `RunStarted`, `GoalDerived`, `PlanSubmitted`,
four `TaskStarted` / `TaskCompleted` pairs (one per task), and
`RunCompleted`. Runner lifecycle envelopes and executor proto events
both count.

An ADK-driven run emits additional dispatch-level events —
`AgentInvocationStarted`, `AgentInvocationCompleted`, and (when a
coordinator invokes an `AgentTool`) `DelegationObserved`. They are
emitted by the goldfive ADK plugin's `before_run` / `after_run` /
`before_tool` callbacks and surface on every sink. Harmonograf
renders them as nested bars with delegation edges on the Agents
timeline (see [telemetry-with-harmonograf.md](telemetry-with-harmonograf.md)).
See [EVENT-MODEL.md §"Agent-invocation events"](../design/EVENT-MODEL.md#agent-invocation-events)
for the schemas.

The two sinks:

- **`InMemorySink`** — local-process list. Cheap sanity check; the
  count matches what the server persists.
- **`HarmonografSink`** — forwards each proto `Event` to the
  harmonograf `Client`, which enqueues it on a ring buffer. A daemon
  thread ships the buffer to the server over gRPC.

If `harmonograf_client` is not installed, the example falls back to
`InMemorySink` + `LoggingSink` and prints a pointer back to this
guide. You can still see the event stream; the UI just stays empty.

To point at a server on another host or port:

```bash
HARMONOGRAF_SERVER=10.0.0.5:7531 \
  uv run python examples/harmonograf_observed/agent.py
```

## Step 5 — Watch the UI

Open `http://127.0.0.1:5173` in a browser. The session picker
(top-left) auto-selects the newest session. For the run you just
triggered you should see:

- Four Gantt bars, one per task, labelled `research`, `draft`,
  `review`, and `publish`. Each bar transitions from `PENDING` to
  `RUNNING` to `COMPLETED` as the executor walks the DAG.
- Three edges between the bars reflecting the sequential dependency
  chain.
- An event-count chip in the session header showing `12`, matching
  the `InMemorySink captured 12 events.` line the example printed.
- A `RunCompleted` marker at the right edge of the timeline when the
  run finishes.
- A drawer (click any bar) with the task's `title`, `summary`, and
  the raw proto payload.

Run the example a second time — a new session appears in the picker
with its own row of bars. Every invocation produces an independent
`run_id`; harmonograf keys sessions on that.

Optional: if you also ran `make demo` (step 2), switch to the ADK tab
at `http://127.0.0.1:8080`, type a prompt into the bundled
`presentation_agent`, and watch a richer rollout materialise in the
harmonograf UI — multiple agent rows, live plan diffs, drift
markers. For the full ADK walkthrough, see harmonograf's
[goldfive-integration.md](https://github.com/pedapudi/harmonograf/blob/main/docs/goldfive-integration.md).

## Understanding the flow

What travels on the wire when you hit run:

1. `Runner.run(user_input)` starts a new session and emits
   `RunStarted` (a proto `Event`) to every sink.
2. `PassthroughGoalDeriver` returns its pre-configured `Goal`; the
   Runner emits `GoalDerived`.
3. `StaticPlanner.generate` returns the four-task plan; the Runner
   emits `PlanSubmitted`.
4. `SequentialExecutor` walks the DAG. For each task it emits
   `TaskStarted` before invoking the adapter and `TaskCompleted`
   (or `TaskFailed`) after.
5. On run termination the executor emits `RunCompleted` (success) or
   `RunAborted` (surrender); setup failures surface a `RunAborted`
   emitted by the Runner itself.

Inside `HarmonografSink.emit` the work is non-blocking: each proto
`Event` is pushed onto the `Client`'s ring buffer and returns
immediately. A daemon thread drains the buffer into a gRPC
client-streaming RPC (`StreamEvents`) on the harmonograf server. The
server's ingest dispatches on the `Event.payload` oneof, writes the
row to SQLite, and broadcasts a delta on the frontend bus. The
browser re-renders.

Because `HarmonografSink.emit` never awaits the network, a slow or
down server never stalls the goldfive executor. The tradeoff is the
shutdown dance: a buffer that hasn't drained before the process exits
loses its contents. The example handles it correctly:

```python
outcome = await runner.run("run the observed workflow")
await runner.close()              # 1. flush every sink's emit path
if client is not None:
    client.shutdown(flush_timeout=5.0)  # 2. drain the wire buffer
```

The order matters. `runner.close()` flushes the goldfive side
(`HarmonografSink.close` marks the sink closed but does not drain the
buffer). `client.shutdown(flush_timeout=5.0)` then waits up to five
seconds for the daemon thread to ship remaining events and joins.
Kill the process between those two calls and the in-flight buffer is
gone. See
[troubleshooting.md](troubleshooting.md#symptom-harmonograf-frontend-shows-no-sessions).

For what harmonograf renders per event kind, see the table in
[harmonograf-integration.md](harmonograf-integration.md#what-harmonograf-renders-per-event).

## Next steps

### Use your own agent

`CallableAdapter` is the simplest `AgentAdapter`. Swap it for
`ADKAdapter` (`uv sync --extra adk`) or `ClaudeAgentSDKAdapter`
(`uv sync --extra claude`) — the rest of the Runner construction,
including the sinks list, is unchanged. See
[writing-an-agent-adapter.md](writing-an-agent-adapter.md) for how to
wrap a new framework.

If you just want the shortest possible Runner construction, the
one-line `goldfive.wrap` / `goldfive.run` helpers auto-detect the
adapter and wire every default for you:

```python
import goldfive

runner = goldfive.wrap(root_agent, sinks=[HarmonografSink(client), LoggingSink()])
outcome = await runner.run("make a presentation about waffles")
```

`goldfive.wrap` picks `ADKAdapter`, `ClaudeAgentSDKAdapter`, or
`CallableAdapter` automatically based on `root_agent`'s shape, and
reuses the ADK agent's `.model` to configure `LLMPlanner` and
`LLMGoalDeriver`. Override any default (`planner=`, `executor=`,
`call_llm=`, `model=`, ...) by passing it as a keyword argument.

For the full ADK + `adk web` pairing — including the
`HarmonografTelemetryPlugin` that captures per-agent spans so the
Gantt shows one row per sub-agent (goldfive#170 + harmonograf#80) —
see [adk-web-integration.md](adk-web-integration.md) and
[harmonograf-integration.md §"The two harmonograf hooks"](harmonograf-integration.md#the-two-harmonograf-hooks-sink--telemetry-plugin).

The lower-level `goldfive.quickstart()` factory is still available
for callers who want a ready-to-run `StaticPlanner` with one task per
goal — see the source of `goldfive/quickstart.py`.

### Add persistence

Pair `HarmonografSink` with `JSONLPersistenceSink` so a crashing
process leaves a durable log the UI can replay from:

```python
from goldfive.sinks import JSONLPersistenceSink

sinks = [
    HarmonografSink(client),
    JSONLPersistenceSink(path=f"./runs/{run_id}.jsonl"),
]
```

`JSONLPersistenceSink` writes every proto `Event` to disk;
`HarmonografSink` streams the same events to the live UI. They
complement cleanly. See
[persistence-and-recovery.md](persistence-and-recovery.md) for the
recovery protocol and [choosing-a-sink.md](choosing-a-sink.md) for
the full sink matrix.

### Live steering from the harmonograf UI

Observability goes two ways. Once the harmonograf UI is rendering
your run, the same connection carries control *back* — pause,
cancel, steer, rewind, approve/reject buttons on the frontend
become `ControlMessage` values on a goldfive `ControlChannel`.

The bridge wiring:

```python
from goldfive import ControlChannel, Runner

channel = ControlChannel()
runner = Runner(
    agent=...,
    planner=...,
    executor=SequentialExecutor(),
    control=channel,
    sinks=[HarmonografSink(client)],
)

# In a companion task, drain client.observe() and translate harmonograf
# ControlEvents to goldfive ControlMessages:
#
#   async for control_event in client.observe():
#       await channel.send(bridge_to_goldfive(control_event))
#
# The ack path is the mirror — iterate channel.acks() and forward
# them back to harmonograf.
```

End-to-end, a single STEER click in the browser travels:

```
UI click → harmonograf server (:7531) → harmonograf_client.observe →
  goldfive ControlChannel → executor.dispatch_control →
    steerer.observe → DriftEvent(USER_STEER) → planner.refine →
      PlanRevised event on sinks → UI re-renders
```

Round-trip is typically sub-100 ms on the transport; total latency is
dominated by the LLM refine (seconds). Full protocol walkthrough,
per-kind behaviour, and approval flows are in
[../design/CONTROL.md](../design/CONTROL.md).

### Read the architectural depth

- [harmonograf-integration.md](harmonograf-integration.md) — why the
  two projects split, what the wire contract is, how the proto
  alignment works.
- harmonograf-side
  [goldfive-integration.md](https://github.com/pedapudi/harmonograf/blob/main/docs/goldfive-integration.md) —
  the Phase D integration design, including the full ADK-driven
  example with `HarmonografTelemetryPlugin`.
- [EVENT-MODEL.md](../design/EVENT-MODEL.md) — goldfive's event
  sequencing semantics.

## If something goes wrong

**`grpc._channel._InactiveRpcError: Connection refused (127.0.0.1:7531)`.**
The harmonograf server is not running. Go back to step 2 and start
`make server-run`. Full entry:
[troubleshooting.md](troubleshooting.md#symptom-harmonografsink--connection-refused).

**`ModuleNotFoundError: harmonograf_client`.**
Step 3 was skipped or installed from the wrong path. The package
lives under `harmonograf/client/`, not `harmonograf/`:
`uv pip install -e ../harmonograf/client`. Full entry:
[troubleshooting.md](troubleshooting.md#symptom-modulenotfounderror-harmonograf_client).

**Example runs, server is up, but the frontend shows no sessions.**
The buffer did not drain before exit. Verify the teardown order —
`await runner.close()` must precede `client.shutdown(flush_timeout=5.0)`,
and the process must not exit between them. Full entry:
[troubleshooting.md](troubleshooting.md#symptom-harmonograf-frontend-shows-no-sessions).

**`ModuleNotFoundError: goldfive protobuf stubs not available`.**
The `proto` extra was not installed. `uv sync --extra proto` and
re-run. Full entry:
[troubleshooting.md](troubleshooting.md#symptom-modulenotfounderror-goldfivepbgoldfivev1events_pb2).

**Vite dev server exits with `address already in use`.** Another
process holds `:5173`. Stop it, or pass a different port
(`pnpm dev --port 5273 --strictPort`) and open that URL instead.
The server's gRPC and gRPC-Web ports are independent.
