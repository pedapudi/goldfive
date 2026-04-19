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
import uuid
from goldfive.sinks import JSONLPersistenceSink

run_id = uuid.uuid4().hex
sink = JSONLPersistenceSink(f"./runs/{run_id}.jsonl")

runner = Runner(
    agent=...,
    planner=...,
    executor=...,
    sinks=[sink],
)
outcome = await runner.run("do the thing")
```

The sink writes one JSON-encoded event per line via
`google.protobuf.json_format.MessageToJson(..., sort_keys=True)`
(since #55 every event on the stream is a proto `Event`). The
`path` is a literal string — pick a filename yourself (for
example, by generating a `run_id` up-front and formatting it in)
since the sink does not substitute placeholders. Pass `mode="write"`
to truncate on open; the default is append.

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
from goldfive.sinks import (
    JSONLPersistenceSink,
    reconstruct_session,
    replay_from_jsonl,
)


# Step 1: load the events from the crashed run.
events = replay_from_jsonl("./runs/abc.jsonl")

# Step 2: rebuild a Session from the events.
session = reconstruct_session(events)

# Step 3: construct a new Runner with the same components.
runner = Runner(
    agent=my_agent_adapter,   # same adapter configuration
    planner=my_planner,
    executor=my_executor,
    sinks=[JSONLPersistenceSink("./runs/abc.jsonl")],  # same file, append
)

# Step 4: resume from the persisted log. ``resume`` replays the
# events internally, reports the last terminal marker as the
# outcome, and returns the reconstructed session on
# ``outcome.session``. v0.1 does not yet continue execution from the
# last un-finished task — see "What's explicitly not in v0.1" below.
outcome = await runner.resume("./runs/abc.jsonl")
```

`Runner.resume()` replays the JSONL log and returns an
`ExecutionOutcome` carrying the reconstructed session. The
reconstructed plan has tasks in whatever terminal states the log
records (COMPLETED, FAILED, CANCELLED); tasks that were mid-flight
when the log ended stay RUNNING in ``reconstruct_session``'s output
but the resume path will be converted to PENDING once live
continuation lands (TODO in ``Runner.resume``). PENDING tasks are
executed as normal when continuation is implemented.

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

`replay_from_jsonl()` currently raises on unparseable lines. Callers
that want best-effort tail-tolerant replay should catch
``google.protobuf.json_format.ParseError`` and stop at the last
parsed event. The reconstructed session then reflects every event up
to the last complete line. On resume, the executor reconstructs from
that state and carries on. You may re-run a task that was actually
completed, which loops back to the idempotency argument from
case (1).

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

## SQLite: cross-run queryable persistence

JSONL is perfect when you want the full event stream for a single run
in one file you can `jq` and tail. It is less useful when you want to
ask questions like "how many drift events across all runs this week?"
or "what's the latest plan the harmonograf dashboard should render?"
For that, goldfive ships `SQLitePersistenceSink`:

```python
from goldfive.sinks import (
    SQLitePersistenceSink,
    list_runs,
    replay_from_sqlite,
)

sink = SQLitePersistenceSink("./runs/goldfive.db")
runner = Runner(..., sinks=[sink])
await runner.run("do the thing")
```

Each event is written to a single table (`goldfive_events` by default)
with schema:

```sql
CREATE TABLE goldfive_events (
    run_id       TEXT    NOT NULL,
    sequence     INTEGER NOT NULL,
    emitted_at   INTEGER NOT NULL,   -- milliseconds since epoch
    kind         TEXT    NOT NULL,   -- e.g. "task_started", "drift_detected"
    payload_json TEXT    NOT NULL,   -- full proto Event as JSON
    PRIMARY KEY (run_id, sequence)
);
```

The `(run_id, sequence)` primary key enforces at-most-one row per
emitted event and makes per-run replay an indexed lookup. Concurrent
`emit` coroutines are serialised with `asyncio.Lock`, same as the JSONL
sink, so writes never interleave.

### Queries

```python
# What runs are in the database?
list_runs("./runs/goldfive.db")          # ['run-A', 'run-B', ...]

# Replay one run as parsed proto Events.
events = replay_from_sqlite("./runs/goldfive.db", run_id="run-A")

# Pass to the same reconstructor used by JSONL recovery.
from goldfive.sinks import reconstruct_session
session = reconstruct_session(events)
```

Because `payload_json` is the full proto-encoded Event, raw SQL works
too:

```sql
-- drift events across the database
SELECT run_id, sequence, json_extract(payload_json, '$.driftDetected.kind')
FROM goldfive_events
WHERE kind = 'drift_detected';

-- which runs actually completed
SELECT DISTINCT run_id
FROM goldfive_events
WHERE kind = 'run_completed';
```

### When to pick which

- Single run, durable audit, happy with files: **JSONL**.
- Cross-run dashboards, shared integrations (e.g. a harmonograf server
  reading goldfive events from the same file as observability data),
  ad-hoc SQL: **SQLite**.
- Both at once is fine — the two sinks are independent:
  ```python
  sinks = [
      JSONLPersistenceSink(path=f"./runs/{run_id}.jsonl"),
      SQLitePersistenceSink("./runs/goldfive.db"),
  ]
  ```

### Custom tables

Pass `table=` to co-locate goldfive events alongside existing schemas:

```python
SQLitePersistenceSink("./shared.db", table="harmonograf_events")
```

`replay_from_sqlite` and `list_runs` take the same `table=` argument.

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
import asyncio
import os

from goldfive import Runner
from goldfive.sinks import (
    JSONLPersistenceSink,
    reconstruct_session,
    replay_from_jsonl,
)

path = "./runs/smoke.jsonl"
if os.path.exists(path):
    os.remove(path)

# run 1 — write events to disk.
runner_a = Runner(..., sinks=[JSONLPersistenceSink(path, mode="write")])
try:
    await asyncio.wait_for(runner_a.run("demo"), timeout=1.0)
except Exception:
    pass
await runner_a.close()

# Inspect what made it to the log (optional).
events = replay_from_jsonl(path)
session = reconstruct_session(events)
print(
    f"recovered run_id={session.run_id}, "
    f"statuses={[t.status.value for t in (session.plan.tasks if session.plan else [])]}"
)

# run 2 — resume from the log. v0.1 replays the events and
# surfaces the last terminal marker via outcome.success; live
# continuation of pending work is tracked as future work on
# Runner.resume.
runner_b = Runner(...)
outcome = await runner_b.resume(path)
print(f"resume outcome: success={outcome.success}")
```

`tests/test_persistence_recovery.py` (in [issue #16](https://github.com/pedapudi/goldfive/issues/16))
expands this into a full matrix across crash points.
