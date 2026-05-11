# Observation-only mode

goldfive#254 introduces a configurable "observation-only" mode that
runs every detector and every `planner.refine` LLM call exactly as
active steering does, but suppresses the three points where goldfive
injects into the coordinator's runtime. The mode is the production
default on `goldfive.wrap()`, `Runner.__init__`, and
`DefaultSteerer.__init__`; existing programmatic callers opt back into
active steering with `observation_only=False`.

## Motivation

Pre-#254, goldfive's defaults actively cancelled the in-flight LLM
call, dispatched a corrective `GOLDFIVE_STEER` ControlMessage, and
overwrote `session.plan` with the revised plan — all on the same
detection tick. That contract is correct, but several operator
profiles want to evaluate goldfive on their stack BEFORE giving it the
keys to the coordinator:

* Operators who want a confidence period: see *what* goldfive flags
  and *what* it would have steered, but leave the coordinator's
  behaviour untouched.
* Harmonograf integrators who want to render "would have steered"
  annotations on the Gantt without affecting the run.
* Anyone running goldfive against a paid model where a misfire is
  expensive.

Observation-only mode flips the default so the first run produces
**signals only**; operators graduate to `observation_only=False` once
they trust what they see.

## Contract

When `observation_only=True`:

1. **`session.plan` is NOT mutated** —
   `DefaultSteerer._apply_revision` builds and returns the stamped
   revised plan (so the dry-run sink event can carry it) but does not
   call `set_session_plan`. The per-(kind, target)
   `last_addressed_revision_by_drift_key` watermark is NOT stamped
   either: a dry-run must not dampen subsequent real detection on the
   same condition.
2. **No `GOLDFIVE_STEER` ControlMessage is dispatched** — the
   corrective body is composed and logged at INFO so operators see
   what would have been sent, but `channel.send` is skipped. The
   `GOLDFIVE_PAUSE_ESCALATE` peer (the human-intervention escalation
   on the same channel) is suppressed under the same passive-observer
   semantics.
3. **`request_invocation_cancel` is a no-op** — returns `[]` without
   writing to the cancel-pending registry or calling the plugin's
   cancel hook. The downstream guard at
   `_is_late_drift_for_terminated_invocation` consults the cancel
   registry, so leaving it empty is the correct semantics for a
   passive observer.

**Still happens** in observation-only mode:

* Every detector runs (judges, drift classifiers, loop detectors,
  CAPABILITY_MISMATCH, GOAL_DRIFT, reflective self-progress, …).
* `planner.refine` / `planner.refine_steer` is called exactly as in
  active steering.
* `DriftDetected` events fire on the sink bus.
* `PlanRevised` events fire on the sink bus carrying the
  would-have-applied plan + the new `dry_run=true` proto flag.

## Sink-event shape

A new boolean field is added to the `PlanRevised` proto:

```
message PlanRevised {
  // ... existing fields ...
  bool dry_run = 11;
}
```

`dry_run=false` (the proto default) is byte-identical to pre-#254
producers — active steering keeps its existing wire shape. `dry_run`
is set to `true` only when the steerer that emitted the event is in
observation-only mode. Sinks (harmonograf) MAY render dry-run rows
differently (greyed-out, marked "would have applied", etc.); the
payload is otherwise the full revised plan + diff + metadata.

`DriftDetected` carries no new field — the dry-run framing applies to
the **response**, not the **observation**. Operators must always see
that goldfive saw the drift.

## Configuration entry points

All three default to `True` (the user-facing behaviour change):

* `goldfive.wrap(agent, observation_only=True)` — convenience entry
  point.
* `Runner(observation_only=True, ...)` — programmatic entry point.
* `DefaultSteerer(observation_only=True, ...)` — direct steerer
  construction.

When the caller supplies their own `steerer=...` to the Runner, the
Runner's `observation_only` parameter is **ignored** and the
steerer's pre-baked flag wins. An INFO log line records the mismatch
so it is visible from logs. Operators who pre-build their own steerer
must set its `observation_only` flag directly.

## Test-suite override

The existing test suite was written against the pre-#254 active-
steering default. Rather than touching every test, `tests/conftest.py`
ships an autouse fixture `_goldfive_active_steering` that flips the
module-level `_test_default_observation_only` to `False` for the
duration of every test. Tests that want to exercise the
observation-only path explicitly pass `observation_only=True` to their
construction sites; the override only governs the **unspecified**
default, so explicit-True from a test is always honoured.

See `tests/test_observation_only.py` for the full set of cases:

* Reasoning-judge drift end-to-end with all three suppressions.
* Positive-control twin with `observation_only=False`.
* Pre-built steerer's flag wins over the Runner parameter.
* `goldfive.wrap()` forwards the flag into the default steerer.
* On-task verdict is a no-op in both modes (negative regression).
* Proto round-trip preserves `dry_run`.
* Production default resolution returns `True`.

## Migration

Existing callers fall into two buckets:

* **Programmatic callers that explicitly pass `observation_only=...`** —
  no change. The explicit value always wins over the resolved default.
* **Programmatic callers that don't pass the kwarg** — these inherit
  the new default and become observation-only. The fix is a one-line
  addition: `observation_only=False`.

CI / integration tests for downstream packages that depended on the
pre-#254 active-steering behaviour should add `observation_only=False`
explicitly to their `wrap()` / `Runner()` calls. The autouse fixture
in this repo's `tests/conftest.py` is the canonical pattern.
