---
name: events
description: The goldfive event taxonomy and reporting tools — what exists, who emits it, how to emit a new one.
applies-when: ["what events exist", "emit an event", "reporting tool", "plan revised"]
---

# Events

goldfive speaks in events. Every state change is a proto `Event`
message fanned out to every configured sink. This skill is the
one-page map.

## Event taxonomy

Lifecycle (run-scoped):

- `RunStarted` — emitted by `Runner` once at the top of `run`.
- `GoalDerived` — goals resolved or passed through. Emitted by `Runner`.
- `PlanSubmitted` — initial plan accepted. Emitted by `Runner`.
- `PlanRevised` — plan refined by the planner in response to drift.
  Emitted by `Executor`.
- `RunCompleted` / `RunAborted` — terminal. Emitted by `Executor` in
  the happy path and at executor-level aborts. `Runner` emits
  `RunAborted` for pre-executor failures (goal derivation, planning,
  reporting-tool registration, steerer bind).

Task-scoped (all emitted by the `Steerer` via `mark_task_*`):

- `TaskStarted` — task transitioned to `RUNNING`.
- `TaskProgress` — optional mid-task progress hint (0.0–1.0 fraction).
- `TaskCompleted` — terminal success.
- `TaskFailed` — terminal failure; `recoverable` toggles whether the
  planner may route around it.
- `TaskBlocked` — non-terminal stall; needs an external unblock.
- `TaskCancelled` — terminal cancellation.

Drift:

- `DriftDetected` — emitted by the `Steerer` when `detect_drift`
  returns a non-`None` `DriftEvent`.

## Envelope shapes

Current `main` emits every event (Runner, Executor, Steerer) as a proto
`Event` message — see PR #55. The dict envelope
`{run_id, sequence, emitted_at, kind, payload}` from
`goldfive.events.make_event` is still exported as a fallback for
callers that can't use the `proto` extra; sinks should tolerate both.

Common idiom for dispatch:

```python
if hasattr(event, "DESCRIPTOR"):
    kind = event.WhichOneof("payload")
    payload = getattr(event, kind) if kind else None
else:
    kind, payload = event["kind"], event["payload"]
```

## The seven reporting tools

Surfaced to every adapter via `Runner.register_reporting_tools`. Names
are a stable contract — don't rename.

| Tool | Purpose |
|---|---|
| `report_task_started` | Agent is beginning a task. |
| `report_task_progress` | Mid-task progress hint. |
| `report_task_completed` | Task done; includes `summary` and optional `artifacts`. |
| `report_task_failed` | Task failed; `recoverable` flag. |
| `report_task_blocked` | Need external unblock; describes `blocker` and `needed`. |
| `report_new_work_discovered` | Unplanned work; planner may add it as a child of `parent_task_id`. |
| `report_plan_divergence` | Current plan no longer matches reality; triggers replan. |

Full schemas: `goldfive.reporting.BUILTIN_REPORTING_TOOLS` and
[docs/reference/tool-protocol.md](../docs/reference/tool-protocol.md).

## Emitting an event

Typed factories live in `goldfive.events` — one per event kind.

```python
from goldfive.events import emit, task_completed_event

evt = task_completed_event(
    run_id=session.run_id,
    sequence=session.next_sequence(),
    task_id="t1",
    summary="did the thing",
    artifacts={"draft": "s3://..."},
)
await emit(sinks, evt)
```

`session.next_sequence()` is the canonical monotonic counter. Serialise
your call to it — two concurrent `asyncio` tasks calling it inside
`gather` without a lock race and can emit duplicate sequence numbers
(see `docs/design/EVENT-MODEL.md` § sequence semantics).

## Adding a new event

Rare — the taxonomy is a contract. If you genuinely need a new kind:

1. Add it to `proto/goldfive/v1/events.proto` as a new oneof case on
   `Event.payload`.
2. `make proto` to regenerate stubs.
3. Add a typed factory in `goldfive/events.py`.
4. Decide who emits it (Runner, Executor, or Steerer) and wire it in.
5. Update `docs/design/EVENT-MODEL.md` with the new kind and its
   ownership.
6. Update any sinks that do per-kind dispatch.

Prefer piggybacking on an existing kind (e.g. encoding a custom
signal as a `DriftKind.CUSTOM` `DriftDetected`) over adding a new
payload type.

## Quick reference

```python
# Every lifecycle event factory
from goldfive.events import (
    run_started_event, run_completed_event, run_aborted_event,
    goal_derived_event, plan_submitted_event, plan_revised_event,
    task_started_event, task_progress_event, task_completed_event,
    task_failed_event, task_blocked_event, task_cancelled_event,
    drift_detected_event,
    emit,            # fan-out helper
    make_event,      # dict envelope fallback
)
```

## Common pitfalls

- Calling `session.next_sequence()` from inside `asyncio.gather`
  without a lock → duplicate sequences. Serialise.
- Using `make_event` (dict) when proto stubs are available — mix and
  match confuses sinks that only accept one shape.
- Agent skips `report_task_started` and jumps to `report_task_completed`
  → the steerer rejects the transition (task isn't `RUNNING`) and the
  executor eventually auto-completes.
- Emitting the same `Event` to multiple sinks *sequentially* with
  awaits between — use `goldfive.events.emit`, which fans out via
  `gather`.

## Related

- [adapters.md](adapters.md) — how the reporting tools reach the agent.
- [sinks.md](sinks.md) — where events go.
- [docs/design/EVENT-MODEL.md](../docs/design/EVENT-MODEL.md) — sequence semantics, ownership rules.
- [docs/reference/tool-protocol.md](../docs/reference/tool-protocol.md) — tool schemas.
- [docs/design/DRIFT.md](../docs/design/DRIFT.md) — drift taxonomy for `DriftDetected`.
