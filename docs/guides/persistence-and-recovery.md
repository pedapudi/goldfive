# Persistence and recovery

goldfive runs can crash. Processes get SIGKILL'd, hosts reboot,
network partitions drop the LLM connection mid-invocation. In every
case you'd like to pick up where you left off without redoing completed
work.

`JSONLPersistenceSink` + `Runner.resume()` is the answer.

Related: [EVENT-MODEL.md](../design/EVENT-MODEL.md),
[writing-an-event-sink.md](writing-an-event-sink.md).

## The idea

Every state-affecting event is already emitted to every sink. If one
of those sinks is durable, the event stream is a log of everything
that happened. Replaying the log reconstructs the session, and a new
`Runner` can continue from the reconstructed state.

```
     first run                  crash!           resume run
     ─────────                  ──────           ──────────
┌──────────────┐           ┌──────────┐     ┌──────────────┐
│   Runner.    │           │ process  │     │   Runner.    │
│   run(...)   │           │  dies    │     │   resume(    │
│              │           │          │     │     events   │
│ emits events │           │          │     │   )          │
│ to:          │           │          │     │              │
│              │           │          │     │ reads events │
└──────┬───────┘           └──────────┘     │ → rebuilds   │
       │                                     │   session    │
       ▼                                     │ → continues  │
┌──────────────┐                             │   from the   │
│  JSONL file  │────────────────────────────▶│   crash point│
│  on disk     │   file persists             │              │
└──────────────┘                             └──────────────┘
```

## The sink

```python
from goldfive.sinks import JSONLPersistenceSink

sink = JSONLPersistenceSink(path="./runs/{run_id}.jsonl")

runner = Runner(
    agent=...,
    planner=...,
    executor=...,
    sinks=[sink],
)
outcome = await runner.run("do the thing")
```

The sink writes one JSON-encoded proto `Event` per line. `{run_id}`
in the path is substituted at run start.

On-disk shape:

```
{"eventId":"01H...","runId":"abc","sequence":"0","emittedAt":"...","runStarted":{...}}
{"eventId":"01H...","runId":"abc","sequence":"1","emittedAt":"...","goalDerived":{...}}
{"eventId":"01H...","runId":"abc","sequence":"2","emittedAt":"...","planSubmitted":{...}}
{"eventId":"01H...","runId":"abc","sequence":"3","emittedAt":"...","taskStarted":{...}}
...
```

Every line is valid JSON on its own. You can `jq` the file, split it
across workers, tail it live, or feed it to anything that speaks
line-delimited JSON.

### Atomicity

`JSONLPersistenceSink.emit()` uses an `fcntl.flock(LOCK_EX)` around
each line append. Concurrent writers (shouldn't happen in one run,
but in case of shared files) serialize via the OS. On POSIX systems,
a single `write()` of a line < `PIPE_BUF` (4096 bytes on Linux) is
atomic — full events rarely exceed this, so partial writes are
exceptional.

On crash, the worst case is one truncated line at the tail. The
recovery path tolerates this.

### Close

`await sink.close()` flushes and releases the file lock. Call this
via the normal `Runner.run()` exit path; it happens automatically.

## The recovery path

```python
from goldfive import Runner
from goldfive.sinks import JSONLPersistenceSink


# Step 1: load the events from the crashed run.
events = JSONLPersistenceSink.from_jsonl("./runs/abc.jsonl")

# Step 2: rebuild a Session from the events.
from goldfive.recovery import reconstruct_session  # helper from issue #15
session = reconstruct_session(events)

# Step 3: construct a new Runner with the same components.
runner = Runner(
    agent=my_agent_adapter,   # same adapter configuration
    planner=my_planner,
    executor=my_executor,
    sinks=[JSONLPersistenceSink(path="./runs/abc.jsonl")],  # same file, append
)

# Step 4: resume.
outcome = await runner.resume(session=session)
```

`Runner.resume()` picks up from the reconstructed session. The
reconstructed plan has tasks in whatever terminal states the log
records (COMPLETED, FAILED, CANCELLED), RUNNING tasks are rolled
back to PENDING (with a warning log — see below), and PENDING tasks
are executed as normal.

### What `reconstruct_session` does

Given the ordered event list:

1. Replays `RunStarted`, `GoalDerived`, and `PlanSubmitted` to
   initialize `session.run_id`, `session.goals`, and `session.plan`.
2. Replays `PlanRevised` in order to reach the latest revision.
3. For every `TaskStarted` / `TaskCompleted` / `TaskFailed` /
   `TaskBlocked` / `TaskCancelled`, applies the corresponding
   status to the task in `session.plan`.
4. Rebuilds `session.completed_results` from `TaskCompleted.summary`.
5. Rebuilds `session.task_progress` from `TaskProgress.fraction` (last
   write wins).
6. Re-seeds `session._next_sequence` to one past the highest observed
   sequence.

If the log contains a `TaskStarted(t)` but no terminal event for `t`,
that task is left in `RUNNING` in the raw replay but the resume path
converts it to `PENDING` so the executor re-invokes the agent. This
is the one place state is "rolled back" — the agent might have been
mid-work when the process died.

### Idempotency: will tasks re-run?

The executor's walk is idempotent over completed tasks: it skips any
task in a terminal state. So:

- `COMPLETED` tasks — not re-run.
- `FAILED` tasks — not re-run. (A refine might add a replacement
  task; that one runs.)
- `CANCELLED` tasks — not re-run.
- `RUNNING` tasks in the log — reset to `PENDING` on resume and
  re-run. This is the "we might have done some work already" case —
  your agent needs to tolerate a retry.
- `PENDING` tasks — run as normal.

## Failure-mode walkthrough

Four concrete failures and how they recover.

### 1. Process killed mid-task

```
   emit RunStarted
   emit GoalDerived
   emit PlanSubmitted
   emit TaskStarted(t1)
   emit TaskCompleted(t1)
   emit TaskStarted(t2)
   <SIGKILL>
```

Resume:

- Replay gives a session with `t1 = COMPLETED` and `t2 = RUNNING`.
- Resume converts `t2 = RUNNING` back to `PENDING`.
- Executor re-invokes `t2`.
- `t3` runs as normal afterward.

Your `t2` agent ran once and produced some output, but
`report_task_completed` never fired. On resume, the agent runs again.
If it's idempotent (most tool-using agents are with respect to
reads), you get a clean completion. If it has side effects
(file writes, API calls that mutate state), you either:

- Tag side effects with the run-id so repeats are detected.
- Add a check at the top of the agent's logic that notices the task
  is already half-done.
- Accept the repeat and make the final step idempotent.

### 2. Agent raised mid-invocation

```
   emit RunStarted / GoalDerived / PlanSubmitted
   emit TaskStarted(t1)
   adapter.invoke(t1) raises ConnectionError
   executor catches → emit TaskFailed(t1, reason="ConnectionError")
   executor classifies as TASK_FAILED_RECOVERABLE
   planner.refine(...) → returns revised plan with t1 replaced by t1'
   emit PlanRevised(revision_index=1)
   emit TaskStarted(t1')
   ...
```

Here there's no crash. The run continues via the refine path. The log
shows the failure and the revision; a resume-from-log would pick up
in the middle of the revised plan.

### 3. Sink failure doesn't crash the run

```
   emit TaskStarted(t1)     # to [JSONLSink, HttpSink]
     → JSONL writes OK
     → Http raises timeout
   goldfive catches HttpSink error, logs, continues
   emit TaskCompleted(t1)   # to [JSONLSink, HttpSink]
     → JSONL writes OK
     → Http raises again; logged, suppressed
   ...
```

JSONL has the full log. Your HTTP backend missed some events. This is
why `JSONLPersistenceSink` is the canonical "durable" sink — it does
local disk writes with explicit locking and should only fail on disk
exhaustion, which is a stop-the-world condition anyway.

### 4. The log is itself truncated

Your tail line is `{"eventId":"01H...","runId":"abc","sequenc...`
(cut off).

`JSONLPersistenceSink.from_jsonl()` skips unparseable lines with a
warning log. The reconstructed session will be slightly behind the
true crash point, but every event up to the last complete line is
replayed. On resume, the executor reconstructs from that state and
carries on. You may re-run a task that was actually completed, which
loops back to the idempotency argument from case (1).

## Operational guidance

**Path layout.** One file per run is the default. For high volume,
shard by date: `./runs/2026/04/18/{run_id}.jsonl`.

**Retention.** Logs grow; rotate or delete completed runs after a
window (hours for ephemeral work, weeks for audit).

**Compression.** JSONL compresses well (typical event line is
~200 bytes, ratio 5–10x with zstd). If you need compressed-at-rest,
wrap the sink in a `zstd.ZstdCompressor` stream.

**Multiple sinks.** Persistence is orthogonal to everything else. A
typical prod config:

```python
sinks = [
    JSONLPersistenceSink(path=f"./runs/{run_id}.jsonl"),
    HttpBackendSink(endpoint="https://obs.example.com/events"),
    LoggingSink(level=logging.INFO),
]
```

JSONL is the durable log; the HTTP sink feeds your observability
platform; logging gives you stdout visibility. Each sink is
independent.

**Monitoring.** Track three things per run:

- JSONL file size as a proxy for "run complexity".
- Time from `RunStarted` to terminal (`RunCompleted` / `RunAborted`)
  — the run's wall-clock duration.
- Count of `DriftDetected` / `PlanRevised` — the run's "turbulence".

These three, dashboarded, give you an operational feel for goldfive
without instrumenting anything beyond the sink.

## What's explicitly not in v0.1

- **Live replay** — feeding JSONL back through sinks in real time.
  You can do this manually (read the file, emit to sinks), but there's
  no `Runner.replay()` yet.
- **Partial resume** — resuming in the middle of a refine call, or
  mid-turn within a task. The resume boundary is between-tasks only.
- **Concurrent resume** — two processes resuming the same run-id is
  undefined. Use `flock` or a DB-backed coordinator if you need to
  prevent it.

## Verifying recovery works

A quick smoke test:

```python
import asyncio, os
from goldfive import Runner
from goldfive.sinks import JSONLPersistenceSink

# run 1 — interrupted after task 1
path = "./runs/smoke.jsonl"
if os.path.exists(path):
    os.remove(path)

runner_a = Runner(..., sinks=[JSONLPersistenceSink(path)])
try:
    # custom executor that raises after the first task, for the test
    await asyncio.wait_for(runner_a.run("demo"), timeout=1.0)
except Exception:
    pass

# run 2 — resume
events = JSONLPersistenceSink.from_jsonl(path)
from goldfive.recovery import reconstruct_session
session = reconstruct_session(events)

runner_b = Runner(..., sinks=[JSONLPersistenceSink(path)])
outcome = await runner_b.resume(session=session)
assert outcome.success
```

`tests/test_persistence_recovery.py` (in [issue #16](https://github.com/pedapudi/goldfive/issues/16))
expands this into a full matrix across crash points.
