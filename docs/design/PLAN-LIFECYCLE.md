# Plan lifecycle & refinement contract

This document specifies goldfive's **plan-level** state machine — how a
Plan evolves across refinements, what a refined plan must preserve
relative to the plan it replaces, and what "run complete" actually
means at the plan layer.

It is the plan-layer counterpart to
[STATE-MACHINE.md](STATE-MACHINE.md) (task-layer state transitions)
and [TASK-LIFECYCLE.md](TASK-LIFECYCLE.md) (cross-protocol task
choreography). A task is a vertex; a plan is the DAG those vertices
live in. The graph itself has a lifecycle, and most steering /
refinement bugs surface at the graph layer, not the vertex layer.

Audiences:
- reviewers validating that a proposed change to `LLMPlanner.refine`
  or `DefaultSteerer._apply_revision` doesn't silently break a
  cross-revision invariant,
- implementers writing a new planner or executor who need to know
  what a correct refinement produces,
- operators triaging "run says success but no output was produced"
  (§6 covers the termination predicate).

All citations are `file:line` into `goldfive/` unless noted.

---

## 1. Plan states

```
[none]
  │
  ▼
CREATED ────────► EXECUTING ────────► COMPLETE
                      ▲ │
                      │ ▼
                  REVISING                   ABORTED
                      ▲ │
                      └─┘   (refine→refine is illegal; see §4.4)
                                        ▲
                                        │
EXECUTING ──────────────────────────────┘
  (unrecoverable drift cascade — see §6.2)
```

| State | Meaning |
|---|---|
| `CREATED` | `Planner.generate` returned a `Plan`; it has passed structural validation (§5) and been installed on `Session.plan`. No task has been invoked yet. |
| `EXECUTING` | The executor is driving tasks. One or more tasks may be in `RUNNING`, others `PENDING`. The plan's `revision_index` is the current index. |
| `REVISING` | `Steerer._handle_drift` has called `Planner.refine(...)`. The *executor* may still be mid-invocation on the currently-running task; the *planner* is generating the next plan. This is a transient state — the existing plan remains installed on `session.plan` for the duration. |
| `COMPLETE` | Every task in the currently-installed plan is in a terminal status (COMPLETED, FAILED, CANCELLED), no orphan PENDINGs remain (§6.1), and the executor has returned an `Outcome` with a computed `success` verdict. |
| `ABORTED` | An unrecoverable drift (`recoverable=False`) or a `USER_CANCEL` control message ended the run mid-flight. The cascade semantics in §6.2 apply. |

**Invariant.** A plan never leaves `COMPLETE` or `ABORTED`. These are absorbing states at the plan layer, just as `COMPLETED` / `FAILED` / `CANCELLED` are at the task layer.

**Invariant.** Only one plan revision is installed on a `Session` at a time. `session.plan` is a pointer to the current revision; prior revisions are not retained in-memory by the steerer (sinks may retain them via emitted `PlanRevised` events; the in-memory object is replaced atomically by `_apply_revision`, `steerer.py:~579`).

---

## 2. Plan metadata carried across revisions

A `Plan` (see `types.py::Plan`) carries revision metadata:

- `id: str` — unique plan id; **stays the same across revisions** so sinks can stitch a revision history.
- `revision_index: int` — monotonically increasing; `rev=0` is the output of `Planner.generate`, refines produce `rev=N+1`.
- `revision_kind: str` — the `DriftKind` value that triggered this refinement (`""` at rev 0).
- `revision_severity: str` — the severity of the triggering drift.
- `revision_reason: str` — free-text detail (the drift's `detail`).

`_apply_revision` (`steerer.py:~579–599`) enforces:

- `new.revision_index ≥ old.revision_index + 1` (bumps if the planner returned a smaller number).
- `revision_kind` / `revision_severity` / `revision_reason` default to the triggering drift's values if the planner didn't set them.

**Observability.** Every revision emits one `PlanRevised(old_id=..., new_id=..., revision_index=..., kind=..., severity=..., reason=...)` event on the sink channel. Sinks that cache plan structure MUST re-read `session.plan` on every `PlanRevised`.

---

## 3. The refinement contract

A refined plan is not a free-form rewrite. It must satisfy five invariants relative to the plan it replaces. The steerer enforces these in `_apply_revision` (structural) and the planner is expected to uphold them (semantic).

### 3.0 Task identity

A task's identity is its `id`, nothing else. The framework has two identity rules, and every refinement invariant below is phrased in terms of them:

1. **Same id, same task.** A task with id `X` in revision N and a task with id `X` in revision N+1 are the *same task* — and §3.1 says they must agree on status and most metadata. This is how "task history" is stitched across revisions.

2. **Different id, different task.** A task in revision N+1 whose id does not appear in revision N is a *new task*, full stop. It starts PENDING with empty progress / notes / retry-lineage counters, regardless of whether its title, description, or intent resembles an earlier task. The planner is free to generate tasks like `draft_slides_kid_friendly` after a USER_STEER cancels `draft_slides` — the new task is not a "retry" of the old one unless it explicitly uses the `retry_` / `retry<N>_` naming convention that the per-task retry cap (#102) watches.

Title overlap, assignee overlap, intent overlap — none of these create cross-revision identity. Only id does.

**Corollary:** a planner that wants a "fresh attempt at the same work" should *emit a new id* (and optionally a `retry_` prefix to opt into the retry-lineage cap). Reusing an old id to "reset" a task is illegal — §3.1 will reject it as a status regression.

### 3.1 Terminal tasks are frozen

Every task whose status in the outgoing plan is terminal
(`COMPLETED`, `FAILED`, `CANCELLED`) **must** appear in the new plan
with:
- the same `id`,
- the same `status` (monotonic — a terminal task cannot be un-failed
  or un-completed),
- the same `title`, `description`, `assignee_agent_id`, and
  `bound_span_id` (so harmonograf's cross-revision span lineage
  stays intact).

If a refined plan is missing a terminal task from the outgoing plan,
or its status has regressed, the steerer rejects the revision and
emits a CRITICAL `SCHEMA_VIOLATION` drift (see #100 /
`_apply_revision`).

### 3.2 Terminal→terminal edges are frozen

If both endpoints of an edge in the outgoing plan were terminal,
that edge **must** appear in the new plan. This preserves the
historical topology so a later forensic replay is faithful.

### 3.3 Terminal→mutable edges may be re-drawn

An edge from a terminal task to a non-terminal task represents "the
prior work feeds the next step." During refinement, the planner may:
- drop the edge (the downstream step no longer consumes that work),
- re-target it to a newly-added task,
- keep it unchanged.

The planner may NOT target a terminal edge at a brand-new terminal
task (terminals cannot have mutable sources). Use a new PENDING
task instead and let the refined plan execute it.

### 3.4 Mutable edges are freely mutable

Edges whose tail is non-terminal in the outgoing plan (i.e., between
PENDING / RUNNING / BLOCKED tasks) may be added, removed, or
rewritten arbitrarily by refine. This is where delete-and-replan
(§4.2) operates.

### 3.5 No cycles, no dangling edges

`Plan.validate(for_revision=True)` (from #100) rejects any revision
whose structure contains:
- duplicate task IDs,
- edges referencing an unknown task,
- cycles (detected via `topological_stages`: any leftover task is a
  cycle member).

On validation failure, the revision is discarded, a CRITICAL
`SCHEMA_VIOLATION` drift is emitted, and `session.plan` is
unchanged. The refine-failure counter (§4.5) is also incremented.

---

## 4. Refinement modes

Different drifts demand different rewrites of the DAG. The current
planner implements three modes, each with its own preservation
contract layered on top of the §3 invariants.

### 4.1 Inline-edit (default)

**Triggers:** `TOOL_ERROR`, `AGENT_REFUSAL`, `PLAN_DIVERGENCE`,
`NEW_WORK_DISCOVERED`, `TASK_FAILED_RECOVERABLE`, `BLOCKED`,
most WARNING-severity drifts.

**Semantic:** "patch the plan around the failing task." The planner
receives the full plan + drift context, emits a JSON plan that
typically:
- leaves all non-affected tasks untouched,
- marks the offender FAILED or leaves it PENDING with updated
  metadata,
- inserts 1-3 new pending tasks to remediate (retry with different
  assignee, add a prerequisite, add a fallback).

**Preservation:** §3.1-3.5. In practice the planner preserves the
bulk of the DAG.

### 4.2 Delete-and-replan (USER_STEER)

**Trigger:** `USER_STEER` drift (from a `ControlKind.STEER`
message).

**Semantic:** "the user has changed direction; discard everything
pending and regenerate around the new intent." The planner:
- preserves every terminal task verbatim (§3.1),
- preserves edges between terminal tasks (§3.2),
- **drops** every PENDING / RUNNING / BLOCKED task,
- regenerates a fresh sub-DAG of PENDING tasks whose roots attach to
  the terminal boundary.

**Partial-state atomicity (the gap this doc formalises).** Before
refine runs, `SequentialExecutor._apply_steer`
(`executors/sequential.py:~247`) has already cancelled the currently
running task. If the refine then **fails** (LLM returns `None`,
raises, or produces an unparseable plan), the "delete" half of
delete-and-replan has happened but the "replan" half has not. The
outgoing plan is now inconsistent: it still lists the dropped
PENDING tasks, but their upstream has been CANCELLED, orphaning
them.

The correct behaviour is not to roll back the cancel (the user
asked for a steer; ignoring it violates the control contract). It
is to roll **forward** the delete: cascade-CANCEL every task in the
dropped set, so the plan lands in a consistent terminal-only shape
with `success=False` and a clear `revision_reason`. This is
implemented by the cascade semantics in §6.3.

### 4.3 Fail-and-route (LOOPING_TOOL_CALL / LOOPING_REASONING)

**Triggers:** `LOOPING_TOOL_CALL`, `LOOPING_REASONING`.

**Semantic:** "the offender is stuck; mark it FAILED and generate
alternative paths." Framework-level guarantee: even if the LLM
refuses to mark the offender FAILED, the planner's deterministic
fallback (`planner.py::_refine_looping_tool_call:~948`) forces the
offender to FAILED in the returned plan. This is the one mode that
cannot soft-fail into "plan unchanged" — the looping task always
terminates.

**Preservation:** §3.1-3.5 plus the offender's transition to FAILED
is a hard constraint.

### 4.4 Refine-within-refine is illegal

While a refine is in flight (`REVISING` state), another drift MAY
fire — but the steerer serializes its drift handling, so the
in-flight refine either (a) completes and installs before the next
drift is processed, or (b) completes and is discarded because the
new drift's refine supersedes it. The steerer never runs two
refines in parallel on the same session.

Concrete enforcement: `DefaultSteerer._handle_drift` is an `async
def` that awaits the planner's `refine` call sequentially.
Concurrent dispatch into the same Session is not supported by the
protocol (see [PROTOCOLS.md](PROTOCOLS.md#steerer)).

### 4.5 Refine failure lifecycle

A refine that raises, returns `None`, or produces a revision that
fails `Plan.validate(for_revision=True)` is a **refine failure**.
The steerer handles it in three layers (see TASK-LIFECYCLE.md §4):

1. Visibility: emit a follow-up CRITICAL `DriftDetected` with
   `detail="refine failed (<kind>): <reason>"`.
2. Backoff: increment `session.refine_failure_counts[(kind,
   task_id)]`; at `REFINE_FAILURE_THRESHOLD=2` consecutive failures,
   mark the originating task FAILED and emit a CRITICAL
   `REPEATED_FAILURE` drift (instead of attempting refine a third
   time).
3. Partial-state atomicity (§4.2 / §6.3): if the triggering drift
   was `USER_STEER`, the delete has already happened — cascade-
   CANCEL dependents so the plan lands in a consistent shape.

Successful refine resets the counter for that `(kind, task_id)`.

---

## 5. Plan validation on creation and revision

Called by `LLMPlanner.generate` (creation) and
`DefaultSteerer._apply_revision` (revision), from #100.
`Plan.validate()` enforces:

- Every task has a non-empty `id`.
- Task IDs are unique within the plan.
- Every edge references existing task IDs.
- No cycles (verified via `topological_stages`'s residual check).
- **At creation only** (`for_revision=False`): every task is
  `PENDING`. At revision (`for_revision=True`), terminal tasks are
  allowed (expected, per §3.1).

On failure: `ValueError` with a descriptive message. `generate`
returns `None`; `_apply_revision` rejects the revision and emits
`SCHEMA_VIOLATION`.

---

## 6. Run termination

This is the section whose absence from prior docs allowed the
"success=True with orphan PENDINGs" bug to ship.

### 6.1 Successful completion

A run completes successfully when **all three** hold:

- **Every task is in a terminal status.** No `PENDING`, no
  `RUNNING`, no `BLOCKED`.
- **No cascade orphans remain** (§6.3). Every CANCELLED /
  FAILED task's downstream PENDINGs have themselves been
  cascade-terminated.
- **Every goal's `success_predicate` returns True** (or is
  `None`, treated as vacuously true).

If all three hold: `Outcome.success = True`.

If only the first two hold (everything is terminal, no orphans, but
some goal predicates returned False or raised): `Outcome.success =
False` with `reason="goal '<summary>' unmet"` (or `"goal '<summary>'
predicate raised: <exc>"` for an exception). Implemented by
`goldfive.results.evaluate_goal_predicates` (`results.py:42`), called
from `sequential.py:490` and `parallel.py:327` before each executor's
`run_completed_event` emission. Predicates are evaluated in
`session.goals` order and short-circuit on the first unmet goal; a
predicate that raises is logged at WARNING and treated as unmet.

### 6.2 Unrecoverable cascade (ABORTED)

When a drift fires with `recoverable=False` — today only
`TASK_FAILED_FATAL`, `USER_CANCEL`, and `INTENT_DIVERGENCE` — the
executor runs the unrecoverable cascade (STATE-MACHINE.md
§"Cascade semantics"):

1. Mark the current task FAILED (if not already terminal).
2. Mark every RUNNING task FAILED.
3. BFS downstream from each just-FAILED task; every reachable
   PENDING → CANCELLED.
4. Clear adapter-bound task context.
5. Emit `RunAborted(reason=drift.kind, drift=drift)`.

Outcome: `success=False`, reason carries the triggering drift.

### 6.3 Cancellation cascade (recoverable path)

Previously undocumented; implemented by the cascade worker
(fix/cancel-cascade PR).

When a task transitions to CANCELLED by any means —
`mark_task_cancelled` from the executor, `mark_task_cancelled` from
a reporting tool, or the implicit cancel in `_apply_steer` — the
steerer walks the plan's edges forward from the cancelled task and
cancels every reachable PENDING task with
`reason="cascade from <original_task_id>"`. Already-terminal
downstream tasks are left alone.

This makes the CANCEL the primitive that delete-and-replan relies
on (§4.2): the cancel alone gets the plan to a terminal-only shape;
the replan adds fresh pending work that will run on top. If the
replan fails, the cascaded CANCELs remain and the run ends cleanly
as "incomplete but not broken."

### 6.4 Reachability audit on executor exit

Belt-and-suspenders for any cancellation path we missed: when the
executor's `_pick_next_task` returns `None` while PENDING tasks
remain (which should be impossible after §6.3 fires correctly), the
executor:

- Emits a CRITICAL `PLAN_DIVERGENCE` drift with detail listing the
  orphan task IDs.
- Cancels each orphan with `reason="reachability audit"`.
- Records `Outcome.success = False` with reason
  `"orphaned pending tasks after run"`.

This catches regressions where a future refactor breaks the
cascade primitive, without letting a silent "success with orphans"
slip through.

---

## 7. Lifecycle invariants, stated

These are the plan-level invariants the framework maintains.
Violating any of them is a bug in goldfive.

1. **Monotonic revision index.** `session.plan.revision_index`
   strictly increases across plan swaps.
2. **Stable plan id.** `session.plan.id` does not change across
   revisions.
3. **Terminal task preservation.** Every terminal task in revision
   N appears with the same status in revision N+1.
4. **Terminal edge preservation.** Every terminal→terminal edge in
   revision N appears in revision N+1.
5. **Plan validation on install.** Every plan installed on
   `session.plan` passes `Plan.validate()`.
6. **Refinement serialization.** Only one `planner.refine` call is
   in flight per session at any time.
7. **Atomic partial-state.** Cancel → cancel dependents; if refine
   fails after a cancel, the cancel cascade has already fired.
   There is no state "cancelled upstream, live downstream pending."
8. **No orphan at COMPLETE.** A run in `COMPLETE` state has no
   PENDING, RUNNING, or BLOCKED tasks remaining.
9. **Success is not the default verdict.** `Outcome.success = True`
   is computed from §6.1's conjunction, not "did anything fail?"

---

## 8. Known gaps (open)

- **Terminal→terminal edge preservation is not currently enforced.** `_apply_revision` does not yet check edge preservation; a buggy planner could silently drop terminal edges. Add to `Plan.validate(for_revision=True)`.
- **Unrecoverable cascade (§6.2) does not currently use the same cascade primitive as §6.3** — they're two different code paths that happen to do similar work. Unify once both are implemented.
- **No cross-revision lineage view.** Sinks receive `PlanRevised` events but the proto doesn't carry the full diff. Harmonograf's UI currently has to stitch revisions by plan_id + revision_index. A dedicated `revision_diff` sidecar would simplify the UI.
