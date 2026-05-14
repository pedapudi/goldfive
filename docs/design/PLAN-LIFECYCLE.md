# Plan lifecycle & refinement contract

See [CONTROL-CHANNEL.md](CONTROL-CHANNEL.md) for the actor-model
context: every revision documented below is applied inside the
channel processor (the actor's serialization point), and `Plan` /
`Task` are immutable so each revision is an atomic epoch swap.

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

**Observability.** Every revision emits one `PlanRevised(plan=..., revision_index=..., drift_kind=..., severity=..., reason=..., diff=...)` event on the sink channel. Sinks that cache plan structure MUST re-read `session.plan` on every `PlanRevised`.

### 2.1 Cross-revision diff sidecar (`PlanRevisionDiff`)

`PlanRevised.diff` (proto `goldfive.v1.PlanRevisionDiff`) is a minimal
change-set attached to every `PlanRevised` event so sinks that want to
render "what changed" don't have to re-fetch and diff the two plans
client-side. It carries:

- `added_task_ids: list[str]` — tasks present in the new plan but not
  the old plan, in new-plan order.
- `removed_task_ids: list[str]` — tasks present in the old plan but
  not the new plan, in old-plan order.
- `modified_task_ids: list[str]` — tasks present in both plans where
  at least one of `title`, `description`, `assignee_agent_id`, or
  `status` changed. (Pure status-only transitions are *also* covered by
  the `Task*` events on the stream; `modified_task_ids` is the
  authoritative per-revision set keyed by id.)
- `added_edges: list[TaskEdge]` / `removed_edges: list[TaskEdge]` —
  edges keyed by the `(from_task_id, to_task_id)` pair. An edge
  re-target shows up as one removed + one added entry.

Identity rules match §3.0: tasks are keyed by `id`, edges by the pair
of endpoints. The diff is built by `goldfive.events.build_plan_revision_diff(old, new)`
and populated in `DefaultSteerer._emit_plan_revised` before the event is
fanned out. `diff` is optional at the proto layer (proto3 default) so
older sinks that haven't updated yet continue to ignore it.

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
`_apply_revision`). Enforced by `Plan.validate(for_revision=True,
prior=session.plan)` in `types.py`, called from
`DefaultSteerer._apply_revision`; id / status preservation is the
machine-checked half of this invariant. Title / description /
assignee / `bound_span_id` preservation remains a planner-semantic
contract.

### 3.2 Terminal→terminal edges are frozen

If both endpoints of an edge in the outgoing plan were terminal,
that edge **must** appear in the new plan. This preserves the
historical topology so a later forensic replay is faithful.
Enforced by `Plan.validate(for_revision=True, prior=session.plan)`
in `types.py`: a revision that drops a terminal→terminal edge is
rejected with a `ValueError`, which the steerer surfaces as a
CRITICAL `SCHEMA_VIOLATION` drift while leaving the prior plan
installed.

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

### 3.6 Corrective predecessors via `supersedes` (non-terminal)

**Motivation (goldfive#248 — tomato e2e false-positive).** When the
planner-LLM produces a corrective revision (e.g. `fix_research_X`
correcting a hallucinated research task), it sometimes attaches the
new task as an *independent root* rather than wiring it in as a
predecessor of the existing PENDING tasks that depended on the
contaminated work. The pre-#248 validator only enforced terminal
preservation (§3.1, §3.2), so this shape passed validation. The
empirical consequence: with REV 2 carrying both `fix_research_X`
(corrective) and `draft_slides` (PENDING, still reachable from
COMPLETED `research_X`), the reconciler claimed `draft_slides` for
the in-flight coordinator's drafting work while `fix_research_X`
sat unrun — out-of-order plan execution, with the slides being
drafted from the very output the correction was meant to fix.

**Contract.** Whenever a revised task `X` declares
`X.supersedes == Y` and `Y` exists in `prior` AND `Y.status` is
*non-terminal* (PENDING / RUNNING / BLOCKED), the revision must
take exactly one of these two shapes:

- **Shape A — keep Y, prepend X.** `Y` stays in the revision with
  status PENDING and the revision contains an edge `X -> Y`. The
  corrective `X` runs first; `Y`'s eventual execution is gated on
  `X`'s output. Y's prior downstream edges (`Y -> Z`) carry forward
  unchanged: the ordering chain becomes `X -> Y -> Z`.
- **Shape B — re-edge consumers through X.** Every prior edge
  `Y -> Z` (where `Z` is non-terminal in the revision) is replaced
  by `X -> Z`. `Y` may stay or be dropped; if it stays, its
  remaining downstream edges to terminal tasks are preserved per
  §3.2. The corrective `X` becomes the new gating predecessor for
  every consumer that previously waited on `Y`.

Inserting `X` as an independent root while leaving `Y` (and its
PENDING downstreams) reachable without going through `X` first is
REJECTED with:

```
task 'X' supersedes 'Y' but downstream consumers of 'Y' not
re-edged through 'X': missing edges [...]. Either add an edge
'X' -> Z for every prior consumer Z of 'Y', OR re-mark 'Y'
PENDING in the revision and add an edge 'X' -> 'Y'.
```

**Interaction with #214 REPLACE / CORRECT.** When `Y` is *terminal*
in the prior plan (COMPLETED / FAILED / CANCELLED / NOT_NEEDED),
this §3.6 check is a no-op — the existing #214 REPLACE path
(failed/cancelled task replaced by a new PENDING successor) and
CORRECT path (completed-but-drift-contaminated task corrected by
an inserted child) already handle the topology correctly via §3.1
/ §3.2's terminal-preservation rules and the planner's
`_emit_plan_revised` re-pinning logic. §3.6 specifically addresses
the gap: superseding work that has not yet finished.

**Self-supersedes and mutual supersedes are rejected.** A task
cannot be its own predecessor (`X.supersedes == X.id`) and a pair
cannot mutually supersede each other (`X.supersedes == Y.id` and
`Y.supersedes == X.id`) — both shapes are structurally meaningless
for a corrective predecessor and surface a confused emit at the
LLM layer (the safety-net `_normalize_supersession_kinds` pass
also clears self-references at parse time, but the validator
catches the pair-cycle case the parser cannot reason about).

**Field shape.** `Task.supersedes` is a single string id (the
existing goldfive#237 field). The §3.6 invariant treats one
target per task; planner LLM prompts ask for one supersedes id
per X. (A future evolution to `tuple[str, ...]` for multi-target
supersession was scoped against the existing single-string proto
schema and the broad reach of the field across reporting,
executor causal-tier matching, and steerer routing — out of scope
for #248.)

---

## 4. Refinement modes

Different drifts demand different rewrites of the DAG. The current
planner implements three modes, each with its own preservation
contract layered on top of the §3 invariants.

### 4.0 Refine inputs and goal-awareness

`Planner.refine(*, plan, drift, goals, observed_actions=None, available_agents=None) -> Plan | None`
is the contract. Beyond `drift` + `plan` the refine receives:

- **`goals`** (`list[Goal]`) — the run's active goals. Goal-aware
  refine (goldfive#154) surfaces them in the LLM prompt AND enforces
  a "every sticky USER_STEER goal must have at least one advancing
  task" validator check. A USER_STEER synthesizes a new `Goal` via
  `planner.synthesize_goal_from_steer(body)` which returns
  `(Goal, mode)` where mode is `"append"` or `"replace"` (see
  `goldfive/steerer.py::DefaultSteerer._apply_user_steer_state`).
  Sticky goals are marked via `Goal.metadata["source"] == "user_steer"`
  and cannot be silently dropped by subsequent refines.
- **`observed_actions`** (`list[ObservedAction] | None`, goldfive#144)
  — only consulted on `drift.kind is DriftKind.PLAN_DIVERGENCE`. The
  reconciler (`PlanReconciler`) builds the list from observed tool
  calls / agent transitions so the planner can either ABSORB the
  observed activity into a revised plan or REJECT by emitting
  `{"reject": true, "reason": "..."}`. A reject collapses to
  `None` — the steerer then escalates via the intervention ladder.
- **`available_agents`** — either a flat `list[str]` (legacy) or a
  structured tree (`list[dict]` with `{name, depth, parent, role,
  kind}` per entry, goldfive#151). On the tree form the planner
  renders an "AGENT TREE" section in its prompt and the validator
  rejects off-registry assignees.

The drift's `kind` + `severity` plus the `DriftEvent.detail` string
is how refine derives its "reason". Conventional reasons:

| Reason origin | Typical `drift.kind` |
|---|---|
| `USER_STEER` | `ControlKind.STEER` → `DriftKind.USER_STEER` |
| `PLAN_DIVERGENCE` | Three-stage gate cross-layer delegation, reconciler off-plan agent, or `report_plan_divergence` |
| `DRIFT_ESCALATION` (umbrella term; no literal enum) | Any other drift that reached Level 1 (ABSORB) or Level 3 (CANCEL_REINVOKE) — `TOOL_ERROR`, `LOOPING_REASONING`, `LOOPING_TOOL_CALL`, `AGENT_REFUSAL`, `INTENT_DIVERGENCE`, `RUNAWAY_DELEGATION`, etc. Generic refine path. |

`REFINE_VALIDATION_FAILED` is **not** a refine input — it's the
terminal output when the planner's own retry budget is exhausted.
The steerer deliberately does NOT feed it back into another refine
(infinite-loop risk); it routes to Level 4 (PAUSE_ESCALATE).

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

### 4.2.1 User-steer rejection is structurally impossible

A USER_STEER drift NEVER produces a rejected revision. The
`install_user_steer` API (`steerer.py`) returns `Plan` (not `Plan | None`)
and never raises `ValueError`. When the LLM-produced revision fails
`Plan.validate(for_revision=True, prior=...)`, the steerer falls back
to a deterministic minimum evolution: preserve every terminal task
verbatim (§3.1), cancel every PENDING/RUNNING/BLOCKED task, drop every
edge incident to a cancelled task. The minimum is provably valid by
construction.

This is a contract, not a heuristic. The principle "user-provided
steering is never rejected by the validator" is enforced at the type
level (return type `Plan`), at the API level (no failure path), at the
implementation level (deterministic fallback always validates), at the
test level (property-based invariant in
`tests/test_user_steer_invariant.py`), and at the doc level (here).

Any change to `Plan.validate`, `_apply_revision`, or `install_user_steer`
that could cause a user steer to abort the turn is a contract violation
and MUST be reverted.

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

**Early-warning ladder (`REASONING_CLUSTER_TIGHTENING`).** The
`LOOPING_REASONING` cliff at cosine similarity `>= 0.9` is paired
with a graduated INFO-severity tier `REASONING_CLUSTER_TIGHTENING`
that fires in the `[0.75, 0.9)` band. The INFO tier is **not** a
refinement mode — it does not mark tasks FAILED, does not reach
`planner.refine`, and does not move the plan out of `EXECUTING`.
Sinks surface it as a "may be looping soon" hint so operators can
intervene before the cliff fires. See
[DRIFT.md](DRIFT.md#reasoning-category--the-models-chain-of-thought-exposes-drift-before-the-tool-calls-do)
for the ladder table.

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
2. Backoff: bump the `fail_count` on
   `session.refine_outcomes[(kind, task_id)]` (goldfive#215 P2);
   at `REFINE_FAILURE_THRESHOLD=2` consecutive failures, mark the
   originating task FAILED and emit a CRITICAL `REPEATED_FAILURE`
   drift (instead of attempting refine a third time).
3. Partial-state atomicity (§4.2 / §6.3): if the triggering drift
   was `USER_STEER`, the delete has already happened — cascade-
   CANCEL dependents so the plan lands in a consistent shape.

Successful refine writes `state="succeeded"` for that `(kind,
task_id)` (which short-circuits a follow-up same-turn drift of the
same key — the prior refine already handled it).

### 4.5.1 Goldfive-authored revision rejection is non-fatal by default

When `Plan.validate(for_revision=True)` rejects a goldfive-authored
revision (autonomous refine output), the run does NOT abort by default.
The existing plan is retained on `session.plan`, a
HUMAN_INTERVENTION_REQUIRED INFO drift is emitted for observability,
and execution continues. The next refine cycle (if the underlying drift
persists) gets another attempt; the existing REFINE_FAILURE_THRESHOLD=2
escalation (§4.5) still fires after two consecutive failures of the
same (kind, task_id).

Operators wanting strict abort-on-rejection — useful for CI, regression
testing, or debugging refine logic — opt in via:

- `Runner(fail_fast_on_revision_rejection=True)`
- env var `GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1`

Default is non-fatal because plan-revision rejection is recoverable: the
existing plan is still installed and valid, agents can still make
progress, and aborting on transient LLM output unreliability is
user-hostile. Loud aborts are a debugging affordance, not a default
user contract.

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
- **At revision with a `prior` plan supplied**
  (`for_revision=True, prior=old_plan`):
    - Every terminal task in `prior` appears in the revision with
      the same id and the same terminal status (no regression, no
      drop) — §3.1.
    - Every terminal→terminal edge in `prior` appears in the
      revision — §3.2.
    - **Re-animation rejected (goldfive#138):** a task that was
      terminal (`COMPLETED` / `FAILED` / `CANCELLED` / `NOT_NEEDED`)
      in the prior plan cannot be re-emitted as `PENDING` in the
      revision. This prevents the planner from "retrying" a
      terminal task by resetting its status — the planner must
      emit a NEW task id (optionally with a `retry_` prefix) if it
      wants a fresh attempt.
- **Off-registry assignees rejected (goldfive#151):** when
  `available_agents` is supplied as a tree, every task's
  `assignee_agent_id` must be a reachable name in the registry.
  An unknown assignee is rejected and fed back to the LLM as a
  validator-correction message for the retry loop.
- **Sticky USER_STEER goal coverage (goldfive#154):** when any
  goal on the session carries
  `Goal.metadata["source"] == "user_steer"`, the revision must
  include at least one PENDING task advancing it. Planner-side
  escape: emit `{"reject": true, "reason": "..."}` instead of
  silently dropping the goal.

`LLMPlanner.refine` / `_refine_user_steer` / `_refine_looping_tool_call`
thread `prior=plan` so the planner catches its own violations before
the steerer does. The retry loop feeds the validator's error message
back to the LLM on attempt N+1 (up to `max_refine_attempts`, default
2). If both attempts fail the planner emits a CRITICAL
`REFINE_VALIDATION_FAILED` drift and returns `None` — see §4.0.

On failure: `ValueError` with a descriptive message. `generate`
returns `None`; `_apply_revision` rejects the revision and emits
`SCHEMA_VIOLATION`.

### 5.1 Revision metadata

Every revision carries, besides the plan structure itself:

| Field | Source | Meaning |
|---|---|---|
| `revision_index` | Monotonic from `prior.revision_index + 1` | Ordering key across revisions; stamped by `_apply_revision` if the planner didn't. |
| `revision_kind` | `drift.kind` | Which drift triggered the refine. |
| `revision_severity` | `drift.severity` | Drift severity at the time. |
| `revision_reason` | `drift.detail` | Human-readable reason. |

All four are surfaced on the `PlanRevised` event so sinks can render
revision timelines without walking back through the drift stream.
The accompanying `PlanRevisionDiff` sidecar (§2.1) tells the sink
what specifically changed.

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

When a drift fires with `recoverable=False` — today
`TASK_FAILED_FATAL`, `USER_CANCEL`, and `INTENT_DIVERGENCE` **at
`CRITICAL` severity** (see DRIFT.md for the graduated
`INTENT_DIVERGENCE` bands; `INFO` / `WARNING` intent-divergence
drifts are recoverable and refine normally) — the unrecoverable
cascade runs (STATE-MACHINE.md §"Cascade semantics"):

1. Mark the current task FAILED (if not already terminal). Owned by
   `Steerer.mark_task_failed(..., recoverable=False)`.
2. Mark every RUNNING task FAILED. Sequential executors have at most
   one RUNNING task so this step is vacuous for them; parallel
   executors own this step explicitly.
3. BFS downstream from each just-FAILED task; every reachable
   non-terminal task → CANCELLED. **This step funnels through the
   same shared primitive as §6.3** —
   `Steerer.cascade_cancel_downstream(session, task_id)` — so both
   cascades emit the same `TaskCancelled` event stream and share the
   terminal-skip / diamond-dedup guards. `mark_task_failed` calls the
   primitive itself when `recoverable=False`.
4. Clear adapter-bound task context.
5. Emit `RunAborted(reason=drift.kind, drift=drift)`.

Outcome: `success=False`, reason carries the triggering drift.

### 6.3 Cancellation cascade (recoverable path)

When a task transitions to CANCELLED by any means —
`mark_task_cancelled` from the executor, `mark_task_cancelled` from
a reporting tool, or the implicit cancel in `_apply_steer` — the
steerer walks the plan's edges forward from the cancelled task and
cancels every reachable PENDING task with
`reason="cascade from <original_task_id>"`. Already-terminal
downstream tasks are left alone.

The downstream walk is the shared primitive
`Steerer.cascade_cancel_downstream(session, task_id)`: the same
method the unrecoverable path (§6.2) invokes from
`mark_task_failed(..., recoverable=False)`. Both cascades therefore
produce the same `TaskCancelled` event stream for the downstream
set, share the terminal-skip / diamond-dedup guards, and share
observability.

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
10. **Plan and Task are immutable; the channel processor is the
    sole writer onto `session.plan` (goldfive#247).** See §7.1
    below.

### 7.1 Immutable Plan/Task and the single-writer rule (goldfive#247)

`Plan` and `Task` are `@dataclasses.dataclass(frozen=True)` (see
`goldfive/types.py`). Every "mutation" — flipping a task's status,
appending tasks, replacing the edge list, bumping `revision_index` —
constructs a NEW object via the helpers in `goldfive.types`:

| helper | purpose |
|---|---|
| `replace_task(plan, task_id, **changes)` | Return a new Plan with one task replaced via `dataclasses.replace`. |
| `with_task_status(plan, task_id, status)` | Sugar over `replace_task` for the common status-transition path. |
| `add_tasks(plan, new_tasks)` | Return a new Plan with tasks appended; preserves order. |
| `replace_edges(plan, edges)` | Return a new Plan with the edge list replaced. |
| `bump_revision(plan, *, revision_index=..., revision_kind=..., ...)` | Return a new Plan with revision metadata updated. |

These helpers NEVER mutate the input. They are pure functions over
`Plan`/`Task`.

**Why the freeze fixes a bug class.** Pre-#247, judges and sinks
read `session.plan` / `task.status`, awaited an LLM, and produced a
verdict against a snapshot the live state had mutated past. The
brussels-sprouts and tomato e2e sessions surfaced this as the
"torn-read" symptom: the reconciler's `mark_task_completed` flipped
`task.status` in place, `_apply_revision` swapped `session.plan`
whole-cloth, and both happened DURING a judge's LLM round-trip. With
frozen types the bug class is impossible: anyone who captured a
`Plan` reference keeps operating on that snapshot for the duration
of their work; the live `session.plan` may move, but their snapshot
doesn't.

**Single-writer enforcement.** The pointer swap onto `session.plan`
is owned by the steerer's channel processor — `mark_task_*`,
`_apply_revision`, `cascade_cancel_downstream`, and the executor
install paths. Every blessed swap site goes through:

```python
from goldfive.types import channel_processor_active, set_session_plan, with_task_status

with channel_processor_active():
    set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.RUNNING))
```

`set_session_plan` checks the `_CHANNEL_PROCESSOR_ACTIVE` contextvar
on every call. Outside the contextvar:

* **Default (production):** logs a `WARNING` and proceeds. The
  runtime stays defensive — a sink that accidentally writes to
  `session.plan` does not crash the run, but the smell is recorded.
* **`GOLDFIVE_STRICT_STATE_OWNERSHIP=1` (CI / dev):** raises
  `goldfive.types.PlanOwnershipViolation`. Pytest auto-enables this
  unless explicitly disabled, mirroring the existing
  `goldfive._state_audit` opt-in tripwire.

This is structural enforcement of "single writer onto session.plan"
without forcing every reader to defensively copy. Sinks and judges
read freely; the steerer's channel processor is the only path that
swaps the pointer.

**Cross-references.**

* Phase 1 (#245) adds `observed_revision_index` to `DriftEvent` so
  drift handlers can detect when a drift was minted against an
  earlier plan. The frozen Plan refactor makes this signal honest:
  the drift's `observed_revision_index` is forever associated with
  the plan revision the drift was minted from, even if `session.plan`
  has moved on.
* Phase 2 (#246) routes drift handling through `ControlMessage`. The
  ControlMessage routing site enters `channel_processor_active`
  before delegating into the steerer's mutation helpers.
* Phase 4 (#248) adds `Task.supersedes` validation. The
  `supersedes: tuple[str, ...] = ()`-style default is in place from
  Phase 3 so Phase 4 only has to wire the validator + prompt + tests.
  (Note: in this revision the field stays `str = ""` for back-compat
  with the existing supersedes-by-id design; Phase 4 may evolve it.)
* Phase 5 (#249) is the top-level design doc; orthogonal to the
  type-level fix here.

---

## 8. Known gaps (open)

(All previously-listed gaps have been closed: §7.1 terminal-status unification by #98, §7.2 plan validation by #100, §7.3 refine-failure backoff by #99, §7.4 ADK session heal by #101, §7.7 per-task retry cap by #102, cascade cascade by #103, goal predicates by #104, terminal edge preservation by #105, revision-diff sidecar by #106, and cascade codepath unification by #107. Any new gaps discovered during maintenance should be added to this list.)

### 8.1 Reflective self-progress check — graduated-risk tool (opt-in)

The observation-based drift pipeline (tool-loop guard, reasoning
hash / cosine loops, refusal markers, stop-reason classifiers) cannot
catch the subtle failure mode where an agent is varying its tool args
but not actually advancing on the task. For that class of drift,
`DefaultSteerer` ships an **opt-in** reflective self-progress check:
every `reflective_check_interval` LLM turns (default 15) the steerer
asks the model "are you making forward progress on task X?" and
classifies the JSON reply into one of three outcomes:

- `{"making_progress": true, "confidence": >= 0.5}` → no drift.
- `{"making_progress": true, "confidence": < 0.5}` →
  `UNCERTAIN_PROGRESS` (INFO — observational only).
- `{"making_progress": false}` → `SELF_REPORTED_STUCK` (WARNING;
  flows through the normal refine pipeline).

The whole feature is off by default. It is enabled by constructing
the steerer with a `reflective_call_llm` callable:

```python
DefaultSteerer(reflective_check_interval=15, reflective_call_llm=my_llm)
```

Operators who don't configure it never trigger the extra LLM call.
The counter (`Session._llm_calls_since_check`) is incremented by
`DefaultSteerer.note_llm_call(session)`, which adapters invoke once
per LLM invocation (the ADK adapter does so from
`after_model_callback`). The counter resets on task transitions so
each task gets a fresh assessment window.

This is **graduated-risk**: the cost of an extra LLM call per check
is non-trivial, and the model's self-assessment can itself be wrong
— so the feature is offered as a tool operators can opt into when
their workload justifies it, not as a default. See
[DRIFT.md §"Reflective self-progress category"](DRIFT.md#reflective-self-progress-category--the-agent-assessing-itself)
for the full taxonomy and failure-mode handling.

---

## 9. Plan-descriptive growth (Phase 1)

Phase 1 of [goldfive#423](https://github.com/pedapudi/goldfive/issues/423)
extends the plan-lifecycle contract to cover one new write path:
**reactive growth at delegation-observation time** when a coordinator
delegates to an agent the planner did not forecast.

When `_maybe_pin_delegation_task` cannot match an observed delegation
to an existing PENDING task via the tier-1 / tier-2 / tier-3 disambiguation
ladder, the steerer synthesises a `Task(discovered=True, ...)`, installs
it onto `session.plan` via `PlanReviser.install_descriptive_growth`, and
re-pins `session.current_task_id` onto the new task. The revision is
emitted as a `NEW_WORK_DISCOVERED` drift (INFO severity, framework-
authored) and lands in both steering and observation modes (the
`_apply_revision` discovery carve-out from goldfive#258).

**Feature flag.** Gated on `SteeringConfig.descriptive_growth_enabled`
(env `GOLDFIVE_STEER_DESCRIPTIVE_GROWTH`, default OFF). When the flag
is off, `_maybe_pin_delegation_task` retains the pre-#423 fallback
behaviour and `CAPABILITY_MISMATCH` Rule C fires as today. PR 4 of
#423 (deferred pending live validation) flips the default.

**Dedup.** Repeated delegations to the same `(agent_name,
args-token-set)` re-pin to the existing discovered task rather than
growing the plan. The key is `Task.discovery_identity_hash`, a
stable SHA-256 prefix computed from the observed
`DelegationObserved.tool_args_json` payload (NOT a goldfive-side
intercept of args at pin time — see PLAN-DESCRIPTIVE-GROWTH.md §13
"adaptive, not predictive").

**Write path.** `install_descriptive_growth` acquires the per-session
plan lock (`_get_plan_lock(session)`) for the swap window, re-reads
`session.plan` inside the lock as the linearisation point against
concurrent refines and concurrent discoveries, runs the dedup check,
and only then builds the revised plan via `add_tasks` + `bump_revision`
and swaps via `set_session_plan`. This is the lock-acquiring synchronous
growth path (Option D from the design doc) — single writer, inside the
lock, full stop. The race contract is identical to goldfive#403's
partial-apply window fix.

**Topology.** Discovered tasks land as independent sub-DAG roots: no
predecessor edges, no supersedes link. Rule 7 of `Plan.validate`
allows this because the predecessor set is empty. A subsequent
planner-authored refine may supersede a discovered task with a
`SupersessionKind.REPLACE` / `CORRECT` link if the planner decides to
consolidate the discovered work into its forecast.

See [PLAN-DESCRIPTIVE-GROWTH.md](PLAN-DESCRIPTIVE-GROWTH.md) for the
full design rationale, the dedup hash function, the refine-cascade
semantics, and the §11.6 race-test contract.
