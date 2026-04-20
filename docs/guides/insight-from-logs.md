# Insight from logs

Guide for operators who don't have (or don't want) the harmonograf
UI but need to read goldfive's raw event stream to understand a
run. Covers what `LoggingSink` emits, how to inspect
`outcome.session.plan` after a run, early-warning signals for the
common failure modes, and post-mortem patterns from raw SQLite
dumps.

If you want the UI view of the same data, see
[telemetry-with-harmonograf.md](telemetry-with-harmonograf.md).

## `LoggingSink` output

`goldfive.sinks.LoggingSink` logs every event as one JSON line via
`google.protobuf.json_format.MessageToJson(..., preserving_proto_field_name=True)`.
The log record's message is the JSON; the logger is configurable
so you can route to a file, stdout, or a structured-logging
backend.

Minimal wiring:

```python
import logging
from goldfive import wrap
from goldfive.sinks import LoggingSink

logging.basicConfig(level=logging.INFO)
runner = wrap(my_agent, sinks=[LoggingSink()])
outcome = await runner.run("make a presentation about waffles")
```

### What each event kind looks like

Twelve event payloads ship in v0.1. The envelope is common —
`event_id`, `run_id`, `emitted_at`, `sequence`, plus one
`oneof payload`. The following shows just the payload
(abbreviated) for each kind.

```jsonc
// RunStarted — fires once; sequence=0.
{"run_started": {
  "goal_summary": "make a presentation about waffles",
  "started_at_ms": 1713600000000
}}

// GoalDerived — the goal_deriver's output.
{"goal_derived": {
  "goals": [{"id": "g1", "summary": "presentation about waffles"}]
}}

// PlanSubmitted — the first plan.
{"plan_submitted": {
  "plan": {"id": "p1", "tasks": [...], "edges": [...]}
}}

// TaskStarted — PENDING → RUNNING transition.
{"task_started": {"task_id": "research", "detail": ""}}

// TaskProgress — RUNNING → RUNNING; agent reporting forward motion.
{"task_progress": {"task_id": "research", "fraction": 0.6, "detail": "found 3 sources"}}

// TaskCompleted — RUNNING → COMPLETED.
{"task_completed": {"task_id": "research", "summary": "...", "artifacts": {}}}

// TaskFailed — RUNNING → FAILED.
{"task_failed": {"task_id": "research", "reason": "LLM timeout", "recoverable": true}}

// TaskBlocked — RUNNING → BLOCKED.
{"task_blocked": {"task_id": "publish", "blocker": "awaiting approval"}}

// TaskCancelled — PENDING/RUNNING → CANCELLED.
{"task_cancelled": {"task_id": "review", "reason": "cascade from research"}}

// DriftDetected — fires whenever detect_drift returns non-None.
{"drift_detected": {
  "kind": "DRIFT_KIND_LOOPING_TOOL_CALL",
  "severity": "DRIFT_SEVERITY_WARNING",
  "detail": "report_task_completed called 7 times with identical args",
  "current_task_id": "draft"
}}

// PlanRevised — fires after a successful planner.refine.
// Includes the PlanRevisionDiff sidecar (goldfive #106).
{"plan_revised": {
  "plan": {...},
  "revision_reason": "redirect: focus on vegan waffles",
  "revision_kind": "user_steer",
  "revision_severity": "warning",
  "revision_index": 1,
  "diff": {
    "added_task_ids": ["research_vegan", "draft_vegan"],
    "removed_task_ids": ["research"],
    "modified_task_ids": [],
    "added_edges": [{"from_task_id": "research_vegan", "to_task_id": "draft_vegan"}],
    "removed_edges": []
  }
}}

// RunCompleted — last event on a successful run.
{"run_completed": {"outcome_summary": "4/4 tasks completed"}}

// RunAborted — last event on a failed run.
{"run_aborted": {"reason": "orphaned pending tasks after run"}}
```

### The monotone rules

- `sequence` increases by one per event within a run. Gaps mean
  events were lost between emission and ingestion; this should
  never happen for in-process sinks. Gaps in a JSONL file mean
  the sink wasn't flushed — see
  [persistence-and-recovery.md](persistence-and-recovery.md).
- `RunStarted` has the lowest sequence; `RunCompleted` /
  `RunAborted` has the highest. Anything after a terminal event
  is a bug.
- `DriftDetected(kind=X)` that produces a refine is followed by
  `PlanRevised(revision_kind=X)` before the next `TaskStarted`.
  If they don't pair, refine returned `None` (or raised).

## Inspecting the plan state after a run

`Runner.run` returns an `ExecutionOutcome` with the full
`session` attached. Five fields to read:

```python
outcome = await runner.run("...")
await runner.close()

s = outcome.session
print(f"success={outcome.success}  reason={outcome.reason!r}")
print(f"goals:         {[g.summary for g in s.goals]}")
print(f"current_task:  {s.current_task_id!r}")

if s.plan is not None:
    print(f"plan.id={s.plan.id}  revision={s.plan.revision_index}")
    for t in s.plan.tasks:
        print(f"  {t.status.value:<10} {t.id:>20}  {t.title}")

# Outcome summaries per-task.
for task_id, summary in s.completed_results.items():
    print(f"  {task_id}: {summary[:80]}")

# Approvals still open (non-empty is a red flag after a run ends).
print("pending_approvals:", list(s.pending_approvals))

# Per-(drift_kind, task_id) refine failure counters.
# Non-zero means refine couldn't recover from that drift.
print("refine_failure_counts:", dict(s.refine_failure_counts))

# Last 20 reasoning blocks. Useful for post-hoc reasoning-drift.
print(f"reasoning_history: {len(s.reasoning_history)} blocks")
```

## Early warning for the filler-loop class

goldfive has a set of structural guards against filler loops
(TASK-LIFECYCLE.md §5), but the **earliest** signal from the
logs alone is the tool-call name distribution within one
`invoke()`:

```python
from collections import Counter

sink = InMemorySink()
# ... run ...

tool_calls = Counter()
for e in sink.events:
    # Reporting-tool dispatch shows up as before_tool_callback in
    # the underlying framework, not as a dedicated event. Look at
    # the DriftDetected events of kind LOOPING_TOOL_CALL instead.
    if e.WhichOneof("payload") == "drift_detected":
        d = e.drift_detected
        if d.kind == "DRIFT_KIND_LOOPING_TOOL_CALL":
            tool_calls[d.detail] += 1

print(tool_calls.most_common(5))
```

If you see the same reporting-tool name dominating the count
(> 30 calls with identical args, or > 50 calls across varied
args on the same `task_id`), the guard detected a filler loop
and cut it off. Cross-reference with
[how-to-debug-a-filler-loop.md](../../.agents/how-to-debug-a-filler-loop.md)
for the postmortem playbook.

For the pre-guard signal — counting raw tool invocations before
any goldfive guard would fire — instrument the adapter's
before-tool hook directly or read the harmonograf DB (next
section).

## Capturing planner reasoning

When you're running LLM-backed planning, the planner's own
reasoning can reveal drift before the first task executes. A
pattern used by `/tmp/e2e-raccoons.py` and similar reproducers:
wrap the `call_llm` callable and log both the prompt and the raw
response:

```python
import json

async def logging_call_llm(system, prompt, model):
    resp = await real_call_llm(system, prompt, model)
    logging.getLogger("goldfive.planner").debug(
        "call_llm response model=%s len=%d\nfirst 500 chars: %s",
        model, len(resp), resp[:500]
    )
    # For OpenAI / Claude models that return structured reasoning,
    # pull the reasoning_content / thinking field out here too.
    return resp

runner = goldfive.wrap(
    my_agent,
    call_llm=logging_call_llm,
    model="gpt-4o-mini",
)
```

What to look for in the response body:

- Truncated JSON (missing closing `}`) — the planner LLM hit a
  token limit. Often a silent cause of refine failures. Manifests
  as `outcome.reason` containing "failed to parse LLM output" or
  the `DriftDetected.detail` containing `_refine_user_steer:
  empty/non-string`.
- An empty `tasks` array — the planner decided no further work
  is needed. Legitimate for some refine cases, disastrous when
  you expected replanning.
- Reasoning that doesn't match the plan — the LLM's
  chain-of-thought says it will do A, but the emitted JSON says
  B. The `INTENT_DIVERGENCE` detector catches the agent-side
  version of this; for planner-side, read the logs.

## Post-mortem from a harmonograf SQLite dump

When a run goes wrong in production, the JSONL file or harmonograf
SQLite DB is the ground truth. The DB schema (harmonograf's
`server/db.sql`) has one row per event:

```sql
CREATE TABLE events (
  run_id      TEXT NOT NULL,
  sequence    INTEGER NOT NULL,
  event_type  TEXT NOT NULL,
  payload     TEXT NOT NULL,   -- JSON
  ts          INTEGER NOT NULL,
  PRIMARY KEY (run_id, sequence)
);
```

Useful queries:

```sql
-- List runs with their start time and end state.
SELECT run_id,
       MIN(ts) AS started_at,
       MAX(CASE WHEN event_type IN ('RunCompleted', 'RunAborted')
                THEN event_type ELSE NULL END) AS ended_as
  FROM events GROUP BY run_id ORDER BY started_at DESC LIMIT 20;

-- Every DriftDetected in one run.
SELECT sequence, json_extract(payload, '$.kind') AS kind,
       json_extract(payload, '$.severity') AS sev,
       json_extract(payload, '$.detail') AS detail
  FROM events
 WHERE run_id = '<run_id>' AND event_type = 'DriftDetected'
 ORDER BY sequence;

-- Final task statuses.
SELECT json_extract(payload, '$.plan.tasks') AS tasks
  FROM events
 WHERE run_id = '<run_id>' AND event_type IN ('PlanSubmitted', 'PlanRevised')
 ORDER BY sequence DESC LIMIT 1;

-- Tool call frequency by name — the filler-loop signal.
SELECT json_extract(payload, '$.detail'), COUNT(*)
  FROM events
 WHERE run_id = '<run_id>'
   AND event_type = 'DriftDetected'
   AND json_extract(payload, '$.kind') = 'DRIFT_KIND_LOOPING_TOOL_CALL'
 GROUP BY 1 ORDER BY 2 DESC;
```

This is the pattern the week's postmortem used when the UI
truncated long reasoning blocks — direct SQL on the JSON payload
surfaced the full detail strings that the UI was eliding.

## Replay from JSONL

If you persisted with `JSONLPersistenceSink`, the full event
stream is replayable:

```python
from goldfive.sinks.persistence import replay_from_jsonl

for event in replay_from_jsonl("./runs/my-run.jsonl"):
    kind = event.WhichOneof("payload")
    # ... same inspection as sink.events ...
```

`JSONLPersistenceSink.close()` flushes before returning, so a
cleanly-terminated run produces a complete file. A crash mid-run
leaves a truncated last line; the replay skips it and logs a
warning.

For cross-run SQL queries, the `SQLitePersistenceSink` produces
the same schema as the harmonograf DB (minus the harmonograf
server-side augmentation). See
[choosing-a-sink.md](choosing-a-sink.md) for the matrix.

## Related

- [telemetry-with-harmonograf.md](telemetry-with-harmonograf.md) — UI-side view of the same data.
- [common-failure-modes.md](common-failure-modes.md) — reason → root cause.
- [choosing-a-sink.md](choosing-a-sink.md) — JSONL vs SQLite vs Logging vs GRPC.
- [persistence-and-recovery.md](persistence-and-recovery.md) — durable event storage.
- [../design/EVENT-MODEL.md](../design/EVENT-MODEL.md) — the proto event taxonomy.
- [troubleshooting.md](troubleshooting.md) — symptom → fix catalogue for the common problems.
