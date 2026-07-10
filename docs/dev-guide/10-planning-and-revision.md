# 10. Planning and Revision

## Read this chapter when...

- You are touching how goldfive **produces a plan** (initial plan, per-turn
  replan) or **revises one** in response to drift.
- You are editing `LLMPlanner` (`goldfive/planner.py`) — the prompt assembly,
  the JSON parser, the retry-with-correction loop, or one of its four entry
  points (`generate` / `refine` / `refine_steer` / `handle_turn`).
- You are editing how a revised plan is **installed** onto the session
  (`goldfive/plan_reviser.py`: `PlanReviser`, `_apply_revision`, the
  observation-only carve-outs, `_emit_plan_revised`, `_cancel_inflight_for_revision`).
- You are editing how **observed agent runs are reconciled** against the plan
  (`goldfive/reconciler.py`: `PlanReconciler`) or how tasks are marked
  `NOT_NEEDED` after an invocation.
- You are editing the **task state machine** (`goldfive/task_state_machine.py`)
  or the frozen `Plan` / `Task` model + validator (`goldfive/types.py`).
- You are editing the ADK request/response `BasePlanner`
  (`goldfive/planners/goldfive_planner.py`) or the CORRECT-kind correction
  write-side (`goldfive/_correction_injection.py`).

If you are here for the *decision* about whether a drift produces a nudge, a
cancel, or a refine at all, that is the steering ladder — see
`09-steering-ladder-and-gates.md`. This chapter is about what happens once the
ladder has decided "refine": how the new plan is built, validated, installed,
and reconciled.

## Files covered

| File | What it owns |
|------|--------------|
| `goldfive/types.py` | `Plan`, `Task`, `TaskEdge`, `Goal`, `ObservedAction`, `TaskStatus`, `TERMINAL_TASK_STATUSES`, `SupersessionKind`; `Plan.validate` (8 rules), `Plan.topological_stages`; the frozen-plan derivation helpers (`replace_task`, `with_task_status`, `add_tasks`, `replace_edges`, `bump_revision`); `set_session_plan` + `channel_processor_active` single-writer guard. |
| `goldfive/planner.py` | `PassthroughPlanner`, `StaticPlanner`, `LLMPlanner`; the `generate` / `refine` / `refine_steer` / `handle_turn` entry points; prompt assembly; `_plan_from_json`; the retry-with-correction loop; `RefineExhausted` failure arm. |
| `goldfive/task_state_machine.py` | `TaskStateMachine` — every sanctioned `Task` status transition; the frozen-plan swap pattern; the `#486` drift-condition resolution hook; `cascade_cancel_downstream`. |
| `goldfive/plan_reviser.py` | `PlanReviser` — `install_initial_plan`, `install_revision_for_drift`, `install_revision_for_user_steer`, `_install_with_drift`, `_apply_revision` (+ observation-only carve-outs), `observe_refine`, `_emit_plan_revised`, `install_descriptive_growth`. |
| `goldfive/reconciler.py` | `PlanReconciler` — observation-driven task reconciliation, delegation observation, `get_missed_tasks` (KEEP), `reset_for_new_plan`, contextual parent-chain matching. |
| `goldfive/planners/goldfive_planner.py` | `GoldfivePlanner(BasePlanner)` — request-side orchestration-context injection, response-side structural filtering, compose-with-user-planner. |
| `goldfive/_correction_injection.py` | Write-side for CORRECT-kind supersedes — `queue_corrections_for_revision`, `pending_correction_key` (full agent path), the `pending_corrections` state protocol + GC. |

Cross-referenced neighbours: `03-runner-and-conversation.md` (who calls
`handle_turn`), `04-executors-and-control.md` (who calls `get_missed_tasks` and
consumes the supersede flag), `09-steering-ladder-and-gates.md` (who decides to
refine), `11-state-ownership.md` (the single-writer plan-swap contract),
`12-events-sinks-telemetry.md` (`PlanRevised`, `TaskTransitioned`,
`refine_attempted/failed`), `05-adk-plugin.md` (how `GoldfivePlanner` is
injected).

## Invariants that bind you here

These are the CANON hard invariants as they bite in *this* subsystem. Violating
any of them is a review-blocking regression.

1. **No prompt-cooperation contracts.** The planner and reconciler must work
   even if the wrapped agent never calls a `report_*` tool. Task transitions are
   driven by *observed* `before_agent` / `after_agent` callbacks in
   `PlanReconciler`, not by the agent volunteering status. Never add a code path
   that only advances the plan when the agent cooperates.
2. **No regex/keyword NL classification.** `handle_turn` collapsed the old
   regex `planner_gate` (retired with `#166`/`#167`) into one LLM call. Do not
   reintroduce a regex that decides "is this a steer?" / "is this factual?".
   Exact-equality / hash matching of *structured* data is fine (e.g.
   `discovery_identity_hash`, task-id set comparison in
   `_plans_structurally_identical`).
3. **Any ADK tree shape.** The reconciler is tree-agnostic: flat single-agent,
   flat specialists, deep coordinator+AgentTool hierarchies all reconcile
   through the same `before/after_agent` + parent-chain logic. Never special-case
   a tree shape.
4. **Adaptive, not predictive.** `install_descriptive_growth` mints a
   `discovered=True` task from the *observed* `DelegationObserved.tool_args_json`
   (the event the agent authored), not from a goldfive-side intercept of agent
   state at pin/dispatch time.
5. **`observation_only=True` is the production default and STRICTLY passive.**
   The plan-install gate in `_apply_revision` reads the kill-switch through
   **exactly one** predicate: `self._steerer.is_active_steering()` (module helper
   `steering_is_active(steerer)` for maybe-steerer callers). Missing / `None` /
   raising ⇒ passive. Never read `_observation_only` directly, and never add a
   second predicate.
6. **Lifecycle gates need stable identity keys.** Correction keys are
   `goldfive.pending_corrections.<full_agent_path>.<task_id>` — never key on an
   LLM-minted / churning id. The `#479` change made the agent segment the
   **verbatim full path** (`team_a.researcher`), not the last dotted segment,
   precisely so two same-named agents in different subtrees do not collide.
7. **Single-writer plan swaps.** The only sanctioned way to change
   `session.plan` is `set_session_plan(session, plan)` inside a
   `channel_processor_active()` region. The steerer's channel processor is the
   sole owner.

---

## 1. The subsystem at a glance

Planning in goldfive is an **overlay** model (`#141`), not a driving model.
goldfive does **not** call `adapter.invoke(task)` in a loop. Instead:

1. The planner produces a `Plan` (a frozen DAG of `Task` vertices).
2. The executor issues **one** invocation of the user's agent tree with the
   user's original request.
3. As the tree runs naturally, the ADK plugin fires `before_agent` /
   `after_agent` / `delegation_observed` callbacks. `PlanReconciler` maps those
   observations onto plan-task status transitions.
4. When drift is detected, the steering ladder may call `planner.refine(...)`
   (or `refine_steer` / a user-steer path). The result is installed as a new
   *revision* of the plan via `PlanReviser`.
5. After the invocation ends, the executor marks any never-exercised PENDING
   tasks `NOT_NEEDED` (`#163`), reads `get_missed_tasks` for observability.

The plan is therefore a **forecast that reality reconciles against**, not a
script the framework executes. This is why `Task.assignee_agent_id` is
"observational, not declarative" (`#252`) — see §3.4.

### Who calls the planner

| Caller | Entry point | When |
|--------|-------------|------|
| `Runner._handle_turn_via_planner` (`runner.py`) | `planner.handle_turn(...)` | Every conversational turn — the single per-turn decision point (`#271` Phase 4). |
| `Runner` first-turn path | `planner.generate(...)` | Legacy / goal-list entry — produces the initial plan when the caller passes `list[Goal]` and `handle_turn` did not produce one. |
| `DriftObserver._handle_drift_dispatch` (`drift_observer.py`, reached via `steerer.drift`) | `planner.refine(...)` | Autonomous / user drift the ladder routed to ABSORB / CANCEL_REINVOKE. |
| `DefaultSteerer._promote_drift_to_steer` | `planner.refine_steer(...)` | A goldfive-detected drift cleared the steer threshold + suppression window. |
| `ParallelDAGExecutor._refine` (`executors/parallel.py`) | `planner.refine(...)` | The parallel executor's own drift-driven refine fallback. |

### Who installs the result

Every produced plan flows into `PlanReviser` (held at `steerer.plans`):

| Situation | Install API |
|-----------|-------------|
| Turn-1 first plan (empty seed) or a PIVOT turn | `install_initial_plan(session=, plan=, is_pivot=)` |
| Turn N+1 replan from a fresh user message | `install_revision_for_drift(session=, drift=NEW_WORK_DISCOVERED, revised_plan=)` |
| Autonomous / goldfive drift refine | `install_revision_for_drift(...)` |
| Genuine operator `STEER` ControlMessage | `install_revision_for_user_steer(session=, raw=, revised_plan=)` |
| Descriptive growth (unmatched delegation) | `install_descriptive_growth(session, agent_name=, tool_args_json=)` |

---

## 2. The frozen Plan/Task data model (`types.py`)

### 2.1 `Task` — an immutable record

`Task` is `@dataclasses.dataclass(frozen=True)` (`#247`). Freezing is
*structural enforcement* of the no-torn-read invariant: a judge or sink that
captured a `Task` reference cannot have it mutated underneath it. Every status
transition builds a **new** `Task` (via `dataclasses.replace` or a helper) and
swaps a new `Plan` onto `session.plan` atomically.

Fields you will actually touch:

```python
# goldfive/types.py — class Task (frozen)
id: str
title: str
description: str = ""
assignee_agent_id: str = ""            # observational, not declarative (#252)
status: TaskStatus = TaskStatus.PENDING
predicted_start_ms: int = 0
predicted_duration_ms: int = 0
bound_span_id: str = ""
cancel_reason: str = ""                # sink-schema slot; goldfive core never writes it
supersedes: str = ""                   # id of the task this one replaces (#237)
supersedes_kind: SupersessionKind = SupersessionKind.UNSPECIFIED   # #251
required_tools: tuple[str, ...] = ()   # advisory; capability_check reads it (#253)
discovered: bool = False               # minted reactively at delegation time (#423)
discovery_identity_hash: str = ""      # stable dedup hash for discovered tasks (#423)
```

Notes for editors:

- `required_tools` is a **tuple** (not list) so `Task` stays hashable and
  shareable under the frozen-`Plan` invariant. Keep it a tuple.
- `cancel_reason` is a **downstream sink schema slot**. goldfive's own planner /
  steerer does **not** mutate it. Do not start writing it from core — the
  colon-prefixed reason strings (`upstream_failed:<id>`, `user_cancel:<id>`,
  `superseded_by_revision:<id>`) are populated by harmonograf-side ingest.
- `discovered` / `discovery_identity_hash` are opaque metadata at validate time
  except for two carve-outs (see §2.5, rule notes).

### 2.2 `TaskStatus` and `TERMINAL_TASK_STATUSES`

```python
class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    NOT_NEEDED = "NOT_NEEDED"          # overlay-model skip (#141)

TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.NOT_NEEDED}
)
```

`TERMINAL_TASK_STATUSES` is the **single source of truth** (`#485`) used by the
steerer's transition guards, the tool-dispatch terminal-task rejection, the ADK
adapter's invoke-loop early break, the parallel scheduler's "skip terminal
tasks" pass, and `TaskStateMachine`. **Import it from `goldfive.types`; never
inline the set.** If `TaskStatus` gains a new terminal member, add it here AND to
`TaskStatus.is_terminal` (both sources must stay in lock-step — the docstring
says so explicitly).

`NOT_NEEDED` is deliberately distinct from `CANCELLED`: `CANCELLED` means
"user/system cancelled"; `NOT_NEEDED` means "the tree chose not to run this
because the plan-to-execution mapping made it redundant" (the reconciler's
post-invocation sweep, §7.3). Sinks distinguish them by `task.status`.

### 2.3 `Plan` — an immutable DAG

`Plan` is also `frozen=True`. Every "edit" — adding tasks, marking one
completed, replacing edges, bumping the revision — constructs a **new** `Plan`.
The live reference is `Session.plan`; swap it only via `set_session_plan`
(§2.6).

```python
# goldfive/types.py — class Plan (frozen)
id: str
run_id: str
goal_ids: tuple[str, ...]
tasks: tuple[Task, ...]
edges: tuple[TaskEdge, ...]
summary: str = ""
revision_reason: str = ""
revision_kind: str = ""                # DriftKind value (str) or ""
revision_severity: str = ""            # DriftSeverity value (str) or ""
revision_index: int = 0
revision_trigger_event_id: str = ""    # mirrors PlanRevised.trigger_event_id (#199)
_goldfive_pivot: bool = False          # F5/#322 pivot marker (declared field, not attr)
```

`goal_ids` / `tasks` / `edges` are **tuples**. `__post_init__` coerces
list-typed inputs to tuples so legacy call sites (tests, the JSON parser,
sink round-trips) that pass lists keep working — but do not rely on that; prefer
passing tuples.

`_goldfive_pivot` used to be a dynamically-attached attribute
(`plan._goldfive_pivot`); with `frozen=True` it is now a declared dataclass
field. `LLMPlanner.handle_turn` sets it via the parser when the turn was a
pivot; `Runner._install_revision` reads it to route through
`install_initial_plan` (no Rule 6 binding — see §2.5 and §6.1).

`Plan.empty(*, run_id="")` is the classmethod that mints the seed plan the
Runner installs on turn 1 (no tasks, no edges); a `Plan` with no tasks is the
unambiguous "first install" signal (see §6.1).

### 2.4 `TaskEdge`, `Goal`, `ObservedAction`

```python
@dataclasses.dataclass(frozen=True)
class TaskEdge:
    from_task_id: str
    to_task_id: str
```

`Goal` (mutable dataclass) carries `id`, `summary`, an optional
`success_predicate`, `metadata`, and a `source` string. The one value you must
know: `GOAL_SOURCE_USER_STEER = "USER_STEER"` (`types.py`). Goals carrying that
source are **sticky** — `LLMPlanner.refine` rejects any revision whose tasks
silently drop them (`#154`). See §4.6.

`ObservedAction` is a snapshot of one observed agent invocation
(`agent_name`, `invocation_id`, `parent_invocation_id`, `started_at`,
`completed_at`). It exists so the planner could compare the planned dispatch
against the observed dispatch on the `PLAN_DIVERGENCE` refine path (`#144`).
**Important:** in production this channel is *dead* — no production caller passes
`observed_actions` into `planner.refine`. See §4.4 and the Common Mistakes
table.

### 2.5 `Plan.validate` — the 8 structural rules

`Plan.validate(for_revision=False, *, prior=None)` raises `ValueError` on
failure. It is a **pure-data** validator (it does not mutate the plan) and is
called at creation (`generate`), at revision (`refine`, `_install_with_drift`,
`_apply_revision` callers), and in `handle_turn`.

| # | Rule | Applies when |
|---|------|--------------|
| 1 | Every task has a non-empty `id`. | Always |
| 2 | Task ids are unique within `tasks`. | Always |
| 3 | Every edge endpoint references a task that exists. | Always |
| 4 | The graph is acyclic (every task placeable by `topological_stages`; leftover = cycle member). | Always |
| 5 | Every task's status is `PENDING`. | Only when `for_revision=False` (creation path). Revisions legitimately carry preserved terminal tasks, so this is skipped when `for_revision=True`. |
| 6 | **Terminal-task preservation** (§3.1) + **terminal→terminal edge preservation** (§3.2): every prior terminal task must reappear with the same id AND same terminal status; every prior edge whose both endpoints were terminal must survive. | `for_revision=True` AND `prior` supplied |
| 7 | **No CANCELLED/FAILED→PENDING edges** (`#137` reachability). New PENDING work must form an independent sub-DAG root or chain off a COMPLETED predecessor. | Always |
| 8 | **Corrective-predecessor topology** (§3.6, `#248`): when a revised task `X` declares `X.supersedes == Y` and `Y` is *non-terminal* in `prior`, the revision must re-route `Y`'s downstreams through `X` (or re-mark `Y` PENDING with edge `X→Y`). Self-supersedes and mutual-supersedes are rejected. When `Y` is *terminal*, the `#214` REPLACE/CORRECT path governs and this rule is a no-op. | `for_revision=True` |

The `discovered=True` carve-outs to these rules: a discovered task is always a
valid sub-DAG root (rule 7 is trivially satisfied — no predecessors by
construction), and it is protected from terminal-drop once terminal (rule 6
already covers this). A discovered task MAY carry a `supersedes` reference but is
not required to.

**Why Rule 6 matters to you:** the single most common validation failure a weak
model produces on a refine is a Rule 6 regression — the LLM "loses" a COMPLETED
task or regresses it to PENDING. The retry-with-correction loop (§4.3) feeds the
exact validator message back to the LLM on the next attempt. If you are adding a
new install path, you MUST call `validate(for_revision=True, prior=session.plan)`
before installing, or a bad revision silently corrupts the DAG.

### 2.6 `topological_stages` — Kahn's algorithm, cycle-tolerant

```python
def topological_stages(self) -> list[list[Task]]:
    # stage 0 = tasks with no deps; each later stage's deps satisfied by earlier
    # stages. Cycles / unknown-id edges tolerated: any unplaceable task is
    # appended to a trailing stage so the full set is always returned.
```

It is used by the parallel executor for stage scheduling and by
`Plan.validate` rule 4 (acyclicity is "did every task get placed before the
leftover fallback fired"). It never raises. If you need "is task X ready", use
the module-level `task_upstream_ready(plan, task_id)` helper instead — it is
supersession-aware (redirects through a replacement when the edge still names the
superseded id) and treats a dangling edge as *not ready* (conservative). That
helper is the DAG-readiness primitive the ADK adapter uses to gate
`goldfive.current_task_id` pinning (`#242`).

### 2.7 Frozen-plan derivation helpers + the single-writer guard

Because `Plan` and `Task` are frozen, framework code derives new plans through a
small named vocabulary. **Use these; do not hand-roll `dataclasses.replace` at
call sites.**

| Helper | Effect |
|--------|--------|
| `replace_task(plan, task_id, **changes)` | New plan with one task replaced. |
| `with_task_status(plan, task_id, status)` | Sugar over `replace_task` for the status-transition path. |
| `add_tasks(plan, new_tasks)` | New plan with `new_tasks` appended (no dedup — validate afterwards). |
| `replace_edges(plan, edges)` | New plan with the edge list replaced (accepts `TaskEdge` or `(from,to)` tuples). |
| `bump_revision(plan, *, revision_index=None, revision_kind=None, ...)` | New plan with revision metadata stamped; `revision_index=None` ⇒ `plan.revision_index + 1`. |

None of these swap the plan onto a session — that is deliberately the caller's
job, done through the single-writer guard:

```python
# goldfive/types.py
@contextlib.contextmanager
def channel_processor_active() -> Iterator[None]: ...
def set_session_plan(session: Session, plan: Plan | None) -> None: ...
```

`set_session_plan` is the **only** blessed mutation primitive for
`Session.plan`. It checks the `_CHANNEL_PROCESSOR_ACTIVE` contextvar: a swap
outside a `channel_processor_active()` region logs a WARNING (production) or
raises `PlanOwnershipViolation` under `GOLDFIVE_STRICT_STATE_OWNERSHIP=1`
(default-on under pytest). The channel processor — the steerer's
`_handle_drift_dispatch` / `_apply_revision` / the executor install paths — is the sole
owner. Sinks and judges **read** `session.plan` and never write. See
`11-state-ownership.md` for the full ownership map.

**When you write a new plan-swap site:** wrap it in `with channel_processor_active():`
and call `set_session_plan`. `TaskStateMachine.mark_task_*` already does this
(§5); copy that pattern.

---

## 3. Supersession — the vocabulary that drives revision topology

Supersession is how a revision expresses "this new task replaces that old one".
It appears everywhere in the revision code, so understand it before editing.

### 3.1 `SupersessionKind`

```python
class SupersessionKind(StrEnum):
    UNSPECIFIED = "UNSPECIFIED"   # default / no link / legacy plan
    REPLACE = "REPLACE"           # old task was PENDING/RUNNING; new takes its DAG slot
    CORRECT = "CORRECT"           # old task had COMPLETED but output was drift-contaminated
```

The kind is **authoritative on the `Task`** (not inferred from the old task's
status at read time) because status is mutable — a CORRECT link must stay
CORRECT even after the plan progresses.

- **REPLACE**: old task was non-terminal; the new task takes its slot. Reporting
  tool calls on the old id are rerouted to the new one
  (`goldfive.reporting._resolve_effective_task_id`).
- **CORRECT**: old task had already COMPLETED but the refiner judged its output
  drift-contaminated. The old task **stays** in the plan as a historical
  COMPLETED node; an edge `old → new` is added; the new task is a correction
  child. Reporting calls on the old id are **not** rerouted (the old work's
  completion is historical fact).

### 3.2 How the kind is set and normalized

The LLM emits a raw `supersedes_kind` in its refine JSON. `_plan_from_json`
parses it raw (`_coerce_supersession_kind`, defaults to `UNSPECIFIED`). Then the
post-parse validator `_normalize_supersession_kinds(revised, prior=...)` in
`planner.py` coerces it based on the old task's *actual* status: if the LLM set a
kind that disagrees with the status, the validator coerces (COMPLETED old ⇒
CORRECT, non-terminal old ⇒ REPLACE) and logs a WARNING.
`_backfill_retry_supersedes` additionally infers a `supersedes` link for
retry-style task ids (`retry_X` → `X`) that the LLM forgot to declare, and
`_check_supersedes_coverage` reports links whose target is missing.

### 3.3 Where the topology is applied

`PlanReviser._integrate_correction_supersedes(revised)` rewires the DAG for
CORRECT-kind links (keeps old, inserts new as child).
`_repin_current_task_on_supersedes` re-pins `session.current_task_id` onto the
replacement. `PlanReviser._emit_plan_revised` calls both, then calls the
correction write-side (§9).

### 3.4 Assignees are observational (`#252`)

`_plan_from_json` **drops** any LLM-supplied `assignee_agent_id`:

```python
# goldfive/planner.py — _plan_from_json
# goldfive#252: assignee is observational, not declarative. Drop
# any LLM-supplied value.
tasks.append(Task(id=tid, title=title, description=..., assignee_agent_id="", ...))
```

This is a load-bearing design decision: goldfive no longer *predicts* which agent
runs a task. The reconciler *observes* which agent ran and credits the matching
task by title/order/parent-chain, not by a declared assignee. Consequence: the
assignee-registry validator `_validate_plan_assignees` (§4.5) is effectively a
**no-op for LLM-produced plans** (every assignee is `""`, and it skips empty
assignees). It still fires for `StaticPlanner` (which preserves baked assignees)
and for any custom planner that populates the field. Do **not** "fix" this by
re-populating assignees from the LLM output — that reintroduces the predictive
model `#252` removed.

---

## 4. `LLMPlanner` — the four entry points (`planner.py`)

`LLMPlanner` delegates to a caller-supplied async LLM callable
`call_llm(system_prompt, user_prompt, model) -> str`. The returned string must
be JSON conforming to the plan schema (it may be triple-backtick fenced;
`_strip_code_fences` handles that). **On any parse error or `call_llm`
exception, every entry point logs a warning and returns `None`** — the host
continues without a plan update. The planner must never break the run.

Two sibling planners exist for non-LLM use: `PassthroughPlanner` (all three of
`generate`/`refine`/`handle_turn` return `None` — safe default so `planner=`
can be wired everywhere) and `StaticPlanner` (returns a baked `Plan` verbatim on
`generate`, `None` on `refine`/`handle_turn`).

### 4.1 Construction

```python
LLMPlanner(
    *, call_llm, model="",
    system_prompt=None, refine_system_prompt=None,
    user_steer_system_prompt=None, looping_tool_call_system_prompt=None,
    plan_divergence_system_prompt=None,
    max_refine_attempts=None,                 # default DEFAULT_MAX_REFINE_ATTEMPTS = 2
    user_steer_one_attempt=None,              # test seam
)
```

Two class constants you may tune:

- `DEFAULT_MAX_REFINE_ATTEMPTS = 2` — validator-error-feedback round-trips
  before falling back. Two covers the typical "the correction fixed it" case.
- `MAX_OUTPUT_TOKENS = 16384` — per-call `max_output_tokens` budget, read once
  per call into `goldfive._llm.call_llm_budget`. Every planner LLM call returns
  structured JSON, so thinking is disabled (`call_llm_thinking_disabled`) and
  the budget is bounded. This is the fix for the unbounded `max_tokens` /
  9.6-minute calls (`demo-v8.log` evidence). The wall-clock backstop is separate
  (`DEFAULT_LLM_CALL_TIMEOUT_MS` in the adapter — see `05-adk-plugin.md`).

The steerer wires three optional emitters/providers via `bind()`:
`set_drift_emitter` (for `REFINE_VALIDATION_FAILED` drifts),
`set_retry_budget_emitter` (per-attempt retry-budget telemetry), and
`set_span_context_provider` (for `GoldfiveLLMCallStart/End` spans). All degrade
to no-ops when unset (standalone / test use).

### 4.2 `generate` — the initial plan

```python
async def generate(self, *, goals, available_agents, context=None) -> Plan | None
```

Flow (attempt loop, `attempts = max(1, max_refine_attempts)`):

1. `if not goals: return None`.
2. Build the base prompt via `_build_generate_prompt(goals, available_agents, context)`.
3. Open a `plan_generate` span (trajectory-level: no target task/agent).
4. Call `call_llm` under `call_llm_budget(MAX_OUTPUT_TOKENS)` +
   `call_llm_thinking_disabled()`.
5. Strip fences → `json.loads` → `_plan_from_json(parsed, run_id=, goal_ids=)`.
6. `plan.validate(for_revision=False)` (creation path — every task must be
   PENDING).
7. `_validate_plan_assignees(plan, available_agents)` — registry check (`#151`),
   a no-op for empty assignees.
8. Return the plan.

**Failure arms** (each appends a correction to the prompt and retries when
`attempt < attempts`): `call_llm` raised; JSON parse failed; `_plan_from_json`
returned `None`; validator rejected; off-registry assignee. **Empty / non-string
response is terminal, not retried** (`#182`): it logs INFO and breaks the loop
with `_EMPTY_RESPONSE_ERROR` — a small-model "no answer" is a model-quality
issue, not a planner failure, and retrying doubles cost without changing the
outcome. On exhaustion, `generate` returns `None` (WARNING for genuine failures;
no redundant WARNING for the empty-response case).

### 4.3 `refine` — revise in response to a drift

```python
async def refine(self, *, plan, drift, goals,
                 observed_actions=None, available_agents=None) -> Plan | None
```

`refine` is a **router** by `drift.kind`:

| `drift.kind` | Routed to | Prompt |
|--------------|-----------|--------|
| `USER_STEER` | `_refine_steer(..., source="user")` | delete-and-replan (§4.4) |
| `LOOPING_TOOL_CALL`, `LOOPING_REASONING` | `_refine_looping_tool_call` | fail-the-looper-and-regenerate (§4.7) |
| `REFINE_VALIDATION_FAILED` | returns `None` immediately | belt-and-braces (refining on our own terminal signal risks an infinite loop) |
| `PLAN_DIVERGENCE`, `OFF_TOPIC`, `JUSTIFIED_DEVIATION` | goal-aware ABSORB/REJECT prompt (`_plan_divergence_system_prompt`) | §4.4 |
| everything else | generic refine prompt (`_refine_system_prompt`) | §4.4 |

Prompt selection is **by drift kind alone**, not gated on
`observed_actions is not None` (`iter-12` / `#220`). Before `#220` the gate was
`observed_actions is not None`, which meant production `PLAN_DIVERGENCE` drifts
(whose callers pass no `observed_actions`) silently fell through to the generic
prompt. Selecting by kind makes routing emission-site-independent. The OBSERVED
ACTIVITY block is still rendered only when `observed_actions` is supplied;
without it the goal-aware prompt receives goals + plan + drift detail, which
`#345` demonstrated is sufficient for the OFF_TOPIC path.

`allow_reject` is enabled on every plan-context path (`PLAN_DIVERGENCE` /
`OFF_TOPIC` / `JUSTIFIED_DEVIATION`) and whenever a sticky `USER_STEER` goal is
present. When the LLM emits the reject sentinel (`{"reject": true, "reason":
...}`), `refine` returns `None` and the steerer escalates via the intervention
ladder — a reject is a *successful decision*, not a validation failure, so it
does **not** emit `REFINE_VALIDATION_FAILED`.

On success, `refine` stamps revision metadata via `bump_revision` (index =
`plan.revision_index + 1`, reason = `drift.detail`, kind/severity from the
drift) and returns the new plan.

On exhausted retries (`revised is None`, not a reject, not empty-response), it
emits `REFINE_VALIDATION_FAILED` via `_emit_refine_validation_failed` and
returns `None`. **It never synthesises a clone of the prior plan with a bumped
index** — that would masquerade a failed refine as a successful no-op revision,
exactly the silent-fallback behaviour `#133` eliminated.

The shared LLM-call-and-validate core is `_call_and_validate_refine(...)`; it
runs the same attempt loop as `generate` but validates with
`for_revision=True, prior=plan` and honours `allow_reject`.

### 4.4 `_refine_steer` — the delete-and-replan path

Shared by `refine` (USER_STEER, `source="user"`) and `refine_steer`
(goldfive-promoted drift, `source="goldfive"`):

```python
async def refine_steer(self, *, plan, drift, goals, available_agents=None) -> Plan | None
```

`refine_steer` is the public entry called by
`DefaultSteerer._promote_drift_to_steer`. Both sources share `_refine_steer` to
guarantee identical merge/validation semantics; only the prompt framing differs
(operator directive vs "goldfive detected agent drift — discard prior work").

Mechanics:

- Terminal (COMPLETED/FAILED/CANCELLED/NOT_NEEDED) tasks are **preserved
  verbatim** (same ids, titles, assignees, statuses).
- PENDING/RUNNING/BLOCKED tasks are dropped; the LLM produces a fresh PENDING
  set that honours the steer.
- The returned plan reuses `plan.id` and `plan.run_id` (lineage intact).
- Each attempt routes through `_user_steer_one_attempt` (overridable via the
  `user_steer_one_attempt=` ctor kwarg / `set_user_steer_one_attempt`, a test
  seam that scripts `(merged_plan, error)` tuples without monkeypatching the
  bound method).
- `iter-11C` short-circuit: if two consecutive attempts produce the **same
  structural-class** validator rejection (`_extract_rejection_kind`), stop
  retrying — feeding Qwen its own error a second time on the same invariant is
  empirically unproductive (~10s burned per attempt). Emit
  `REFINE_VALIDATION_FAILED` and return `None` so the steerer falls through to
  the supersede path immediately.
- Empty-response is terminal (INFO, no escalation) per `#182`.

### 4.5 `available_agents` resolution (the tree registry, `#151`)

`available_agents` may be:

- a plain `list[str]` (legacy flat registry), or
- a **structured tree** as produced by `ADKAdapter.available_agents_tree`.

`_is_tree_entry_list` detects the tree shape; `_render_agents_block` renders an
"AGENT TREE" section for the prompt; `_flatten_agent_names` flattens the tree to
the set the registry check uses. Callers resolve it as:

```python
tree = getattr(adapter, "available_agents_tree", None)
if isinstance(tree, list) and tree:
    available_agents = list(tree)
else:
    flat = getattr(adapter, "available_agents", None)   # list[str] fallback
    available_agents = list(flat) if flat else None
```

An empty/None registry skips the assignee check (back-compat). The steerer's
`_planner_refine_accepts_available_agents(planner)` (see §4.9) probes whether a
given planner's `refine` even accepts the kwarg before threading it.

### 4.6 Sticky USER_STEER goals

`_user_steer_goals(goals)` returns goals whose `source == GOAL_SOURCE_USER_STEER`.
`_check_user_steer_goals_preserved(...)` (using `_goal_summary_tokens` for a
token-overlap check) rejects a revision that silently drops a sticky goal so a
later autonomous drift cannot unwind an operator steer by refining around it
(`#154`). `_render_sticky_goals_block` renders those goals with a `[STICKY]`
annotation in the prompt so the LLM sees which goals it must not drop.

### 4.7 `_refine_looping_tool_call` — fail-and-regenerate

For `LOOPING_TOOL_CALL` / `LOOPING_REASONING`: the looping task is forced to
`FAILED` (via a `_force_looper_failed` pre-validation stamp) so the rest of the
plan can route around it; non-looping completed tasks are preserved verbatim.
Retry loop as in §4.3. On exhaustion it emits `REFINE_VALIDATION_FAILED` and
falls back to the deterministic `_fallback_fail_loop_plan` — losing the looper's
slot is still better than re-looping. `_build_looping_tool_call_prompt`
explicitly enumerates the structural invariants the validator enforces so the
LLM has no reason to silently lose a terminal task (`#133`).

> **PROTECTED KEEP.** `LOOPING_TOOL_CALL` deliberately emits `LOOPING_REASONING`
> with NUDGE-first CRITICAL routing (`#204`/`#206`). Do not "simplify" the two
> kinds into one or delete the looping-tool-call planner surface without explicit
> human sign-off. See `17-invariants-hazards-history.md`.

### 4.8 `handle_turn` — the single per-turn decision (`#271` Phase 4)

```python
async def handle_turn(self, *, user_input, session,
                      conversation_history=None, available_agents=None, context=None) -> Plan | None
```

`handle_turn` collapsed the old **five-stage** per-turn pipeline into **one** LLM
call:

1. ~~`planner_gate` regex short-circuits (factual-question + steer-language)~~
2. ~~`planner_gate.classify_turn` LLM gate~~
3. ~~`synthesize_goal_from_steer` LLM call~~
4. ~~regex-based qualification merge~~
5. ~~`planner.refine` / `planner.generate`~~

All collapsed into one call that produces both the decision AND the next plan.
"Classification" is now an emergent property of "did the LLM produce a plan or
not" rather than a synthetic categorical label — this is how goldfive stayed
compliant with the no-regex-NL-classification invariant after retiring the
regex gates (`#166`/`#167`).

Contract:

- Returns `None` when the input is purely conversational and the current
  revision still describes the right work → the Runner reuses `session.plan`.
- Returns the next `Plan` revision when a plan change is warranted.
- Reads the prior plan + goals off `session.plan` / `session.goals` (the Runner
  guarantees `session.plan` is non-None on every turn — it seeds `Plan.empty` on
  turn 1, so the planner produces revision 1 against an empty prior).
- Any LLM/parse failure → WARNING → `None`. Never breaks the run.

Retry: `max_attempts = 2` (one retry) with validator-feedback appended (F7,
`#322`). A first-attempt validation failure — almost always a Rule 6
terminal-task / terminal→terminal-edge regression — gets a second chance with an
explicit error appended. A pivot install (`_goldfive_pivot`) validates against
`prior=None` (no Rule 6); otherwise against `prior=prior_plan`. On a final-attempt
failure it returns the candidate anyway (the Runner's install path surfaces the
error as a `SCHEMA_VIOLATION` drift).

`_parse_handle_turn_response(raw, prior_plan, context)` parses the
`{"reasoning": ..., "plan": ... | null}` envelope (see the prompt tail in
`_build_handle_turn_prompt`). A `null` plan ⇒ `None` (conversational). A produced
plan is minted with a fresh id and the pivot flag set when the response
classified the turn as a pivot (`replaces_prior`).

The Runner then installs the result via `Runner._install_revision` (§6.1).

### 4.9 The signature-probing call convention

Not every planner's `refine` accepts the `#151` `available_agents=` kwarg
(user-supplied / pre-`#151` test stubs predate it). The steerer probes the
signature **once per drift** rather than blindly passing the kwarg and catching
`TypeError`:

```python
# goldfive/steerer.py
def _planner_refine_accepts_available_agents(planner: Any) -> bool:
    refine = getattr(planner, "refine", None)
    if refine is None:
        return False
    try:
        sig = inspect.signature(refine)
    except (TypeError, ValueError):
        return False                         # unintrospectable → don't pass the kwarg
    params = sig.parameters
    if "available_agents" in params:
        return True
    for p in params.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:   # **kwargs → passes through
            return True
    return False
```

`DriftObserver._handle_drift_dispatch` calls this and branches: when `True` it calls
`refine(plan=, drift=, goals=, available_agents=)`, otherwise the legacy
`refine(plan=, drift=, goals=)`. The same helper gates the parallel-executor
refine path. **Do not remove the probe** — it is what keeps the `#151` kwarg
additive for custom planners without forcing every third-party planner to update
its signature.

---

## 5. `TaskStateMachine` — every sanctioned transition (`task_state_machine.py`)

`TaskStateMachine` (held as part of the steerer surface) is the **only**
sanctioned way to transition a `Task`. Its methods:

| Method | Target status | Cascades? |
|--------|---------------|-----------|
| `mark_task_running` | RUNNING | no |
| `mark_task_progress` | *(no transition — liveness ping only)* | no |
| `mark_task_completed` | COMPLETED | no |
| `mark_task_failed` | FAILED (+ `TASK_FAILED_*` drift) | via caller |
| `mark_task_blocked` | BLOCKED | no |
| `mark_task_cancelled` | CANCELLED | yes (cascade to downstream) |
| `mark_task_not_needed` | NOT_NEEDED | **no** (per-task observation, `#141`) |
| `cascade_cancel_downstream` | CANCELLED (BFS over non-terminal downstream) | — |

Every mutating method follows the same frozen-plan pattern (do not deviate):

```python
# goldfive/task_state_machine.py — mark_task_running (representative)
task = self._find_task(session, task_id)
if task is None:
    return
if task.status in _TERMINAL_TASK_STATUSES:      # idempotent guard — never re-transition terminal
    return
from_status = task.status
with channel_processor_active():                # single-writer envelope
    assert session.plan is not None
    set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.RUNNING))
task = self._find_task(session, task_id) or task  # refresh — the old ref is now stale
session.current_task_id = task_id
_ostate.sync_current_task_from_transition(session.state, task, TaskStatus.RUNNING)  # #152
await self._emit_task_started(session, task_id, detail)
await self._emit_task_transitioned(session, task, from_status=..., to_status=..., source=source)
```

(`_TERMINAL_TASK_STATUSES` in this module is a local alias for the canonical
`goldfive.types.TERMINAL_TASK_STATUSES` — same frozenset.)

Three things a weak model routinely gets wrong here:

1. **Refresh the local `task` after the swap.** The pre-swap `task` reference is
   frozen and stale — reading `.status` off it after `set_session_plan` gives the
   old value. Always re-`_find_task`.
2. **The terminal guard is not optional.** `if task.status in
   _TERMINAL_TASK_STATUSES: return` makes every transition idempotent on terminal
   tasks. Removing it lets a late callback resurrect a COMPLETED task.
3. **`source` is threaded to `TaskTransitioned`.** The live LLM path passes
   `"llm_report"` / `"handler_default"` / `"supersedes_reroute"`; the reconciler
   passes observation strings; plan-revision transitions pass `"plan_revision"`.
   Defaulting to `"other"` is fine for back-compat but prefer threading the real
   source (`#251` R4). See `12-events-sinks-telemetry.md`.

### 5.1 The `#486` drift-condition resolution hook

`_emit_task_transitioned` is the funnel every transition passes through
(`mark_task_*`, cascade, plan-revision transitions). It carries the `#486`
lifecycle-resolution hook:

```python
# goldfive/task_state_machine.py — _emit_task_transitioned
if task_id_for_progress and to_status in _TERMINAL_TASK_STATUSES:
    await self._steerer.drift.resolve_conditions_for_terminal_task(
        session, task_id=task_id_for_progress, to_status=to_status
    )
```

A terminal task **moots** every open drift-condition pinned to it: no further
observation on that task can escalate or recover them, so leaving them in
`KEY_ACTIVE_DRIFTS` makes the active set grow monotonically and consumers never
see an intervention succeed. `resolve_conditions_for_terminal_task` calls
`_ostate.resolve_drifts_matching(session.state, task_id=...)` and emits one
`DriftDetected(lifecycle=DRIFT_LIFECYCLE_RESOLVED)` per resolved condition
(INFO severity, carrying `prev_severity`). This is **pure lifecycle telemetry**
— no intervention decision reads the result, so behaviour is identical under
`observation_only` True and False.

The hook runs **before** the sink check (it must land even when sinks are
missing) and is wrapped so a bookkeeping failure never breaks a live transition.

The complementary resolution path is `_resolve_conditions_on_on_task_verdict`
(the reasoning judge returning a clean ON-TASK verdict resolves only the
`_REASONING_PIPELINE_DRIFT_KINDS`: `LOOPING_REASONING`,
`REASONING_CLUSTER_TIGHTENING`, `OFF_TOPIC`, `JUSTIFIED_DEVIATION`,
`INTENT_DIVERGENCE`). **`GOAL_DRIFT` is deliberately absent** from that set — a
reasoning-scoped verdict carries no evidence about a trajectory-level goal
question, so **`GOAL_DRIFT` conditions resolve only at task-terminal**. Do not
add `GOAL_DRIFT` to `_REASONING_PIPELINE_DRIFT_KINDS`.

### 5.2 `cascade_cancel_downstream`

BFS over `plan.edges` cancelling every non-terminal downstream of `cancelled_id`
(skips already-terminal tasks). Shared by the unrecoverable-cascade
(PLAN-LIFECYCLE §6.2) and cancellation-cascade (§6.3) paths.
`mark_task_not_needed` deliberately does **not** cascade — a NOT_NEEDED task is
an observation about one plan entry, not a signal that downstream work is
invalid. `mark_task_not_needed` emits `TaskCancelled` at the proto level with a
`not_needed:` reason prefix (there is no dedicated `TaskNotNeeded` message; the
live `task.status` on the plan is the authoritative signal).

---

## 6. Installing revisions — `PlanReviser` (`plan_reviser.py`)

`PlanReviser` is `steerer.plans`. It owns the install pipeline: validate →
apply → cancel-in-flight → emit `PlanRevised`. The install APIs were listed in
§1; this section covers the shared internals.

### 6.1 The install APIs and how the Runner routes

`Runner._install_revision` (`runner.py`) branches:

```python
first_turn = session.plan is None or not session.plan.tasks
is_pivot = bool(getattr(revised_plan, "_goldfive_pivot", False))
if first_turn or is_pivot:
    installed = await self.steerer.plans.install_initial_plan(
        session=session, plan=revised_plan, is_pivot=is_pivot)
else:
    drift = DriftEvent(kind=DriftKind.NEW_WORK_DISCOVERED, severity=DriftSeverity.INFO,
                       detail=user_text, authored_by="goldfive")
    installed = await self.steerer.plans.install_revision_for_drift(
        session=session, drift=drift, revised_plan=revised_plan)
```

- **Turn-1 first plan** (empty seed) → `install_initial_plan`. **No
  `DriftDetected` is emitted** — installing the first plan is structural, not an
  intervention. This eliminated the synthetic `USER_STEER` drift the pre-Option-A
  path fabricated (`#271` follow-up). The internal placeholder drift it threads
  is `NEW_WORK_DISCOVERED` (INFO) purely so `PlanRevised.drift_kind` has a
  coherent value; it is **never** emitted as a `DriftDetected`.
- **PIVOT turn** → `install_initial_plan(is_pivot=True)`. The plan already
  carries a fresh id (minted in the parser); routing through `install_initial_plan`
  means Rule 6 doesn't reject a legitimate pivot for "dropping" the prior's
  terminal tasks (the validator runs with `prior=None`).
- **Turn N+1 replan** → `install_revision_for_drift` with a `NEW_WORK_DISCOVERED`
  drift at INFO. This is the honest classification: not an intervention, not a
  `USER_STEER` (no operator ControlMessage exists), just additional work the
  planner integrated.

Genuine operator `STEER` ControlMessages do **not** flow through
`_install_revision` — they take the executor's steer loop straight to
`install_revision_for_user_steer`. `install_revision_for_drift` actively
**refuses** `USER_STEER` drifts (raises `ValueError`) so callers can't fabricate
a USER_STEER from plumbing (the category error `#199`/`#302` papered over).
`install_revision_for_user_steer` builds the `USER_STEER` drift internally from
the `raw` ControlMessage (with `authored_by="user"`) so callers cannot fabricate
a user steer from plumbing either.

### 6.2 `_install_with_drift` — the shared pipeline

Both `install_revision_for_drift` and `install_revision_for_user_steer` delegate
to `_install_with_drift(session=, drift=, revised_plan=, apply_user_steer_state=)`:

1. If `apply_user_steer_state`: `_apply_user_steer_state(drift, session)` (active-steer
   bookkeeping + dedup — only for genuine operator STEERs).
2. `_emit_drift_detected(session, drift)` — the real `DriftDetected`.
3. `_fold_runtime_terminal_statuses(revised_plan, session.plan)` — fold runtime
   terminal statuses from the prior plan onto the candidate **before validation**
   (the `I4` fix; this is where the `v24` phantom-state regression lived).
4. `revised_plan.validate(for_revision=True, prior=session.plan)`; on failure emit
   `SCHEMA_VIOLATION` (CRITICAL) and return `False`.
5. **No-op revision rejection** (`#271`): if `_plans_structurally_identical(session.plan,
   revised_plan)` (same task ids, edges, assignees, statuses), skip the install
   entirely (INFO log, return `False`) — bumping the index for an unchanged plan
   would emit a misleading `PlanRevised` with no diff.
6. Capture `prev_plan` **before** apply; mint `attempt_id`; emit `refine_attempted`.
7. `revised_plan, was_installed = self._apply_revision(session, revised_plan, drift)`.
8. `await self._cancel_inflight_for_revision(drift, session)` (§6.5).
9. `await self._emit_plan_revised(..., dry_run=not was_installed)`.

### 6.3 `_apply_revision` — stamp + the observation-only gate

This is the single most important method in the file. Signature:

```python
def _apply_revision(self, session, revised, drift) -> tuple[Plan, bool]:
    # returns (stamped_revised, was_installed)
```

**It no longer mutates session state** (`#403`). It computes the stamped plan and
decides *whether* it should install; the actual `set_session_plan` write,
`last_addressed_revision_by_drift_key` stamp, and orchestration-state pointer
push all moved **into** `_emit_plan_revised`'s lock-protected region. Pre-`#403`
those three mutations ran *outside* the plan lock, then the caller awaited
`_cancel_inflight_for_revision` (which yields the loop) before
`_emit_plan_revised` acquired the lock — leaving a partial-apply window where a
`_wait_plan_stable` reader saw a bumped `revision_index` with the un-rewired
(pre-supersedes) edge DAG. The contract is now **"compute, don't install."**

The gate:

```python
# goldfive/plan_reviser.py — _apply_revision
is_bootstrap = prev is None
is_user_authored = (drift.authored_by or "").lower() == "user"
is_discovery = drift.kind is DriftKind.NEW_WORK_DISCOVERED
gate_active = (
    (not is_bootstrap)
    and (not is_user_authored)
    and (not is_discovery)
    and (not self._steerer.is_active_steering())   # THE kill-switch read (canon #5)
)
if gate_active:
    # observation_only: do NOT swap session.plan, do NOT stamp the addressed
    # watermark, do NOT push the orchestration-state pointer. Still return the
    # stamped `revised` so _emit_plan_revised can render the dry-run preview.
    return revised, False
return revised, True
```

The revision metadata is **always** stamped (via `bump_revision`) regardless of
the gate — so a would-have-applied preview renders faithfully. Only the actual
install is skipped when the gate fires.

**The three carve-outs that install even under `observation_only=True`:**

| Carve-out | Predicate | Why it is not "corrective" |
|-----------|-----------|----------------------------|
| **bootstrap** | `prev is None` | Cold session — a first plan is structural, not a correction. |
| **user-authored** | `drift.authored_by == "user"` | The operator has the authority to override observation mode. |
| **discovery** | `drift.kind is DriftKind.NEW_WORK_DISCOVERED` (`#258`) | A description of work the planner / a sub-agent reported — not a framework-driven correction. Covers turn-1 installs through `install_initial_plan` (where `prev` is the `Plan.empty` seed, **non-None**, so the bootstrap predicate misses it) and turn N+1 replans. |

The mental model: **observation-only suppresses framework-driven
*corrections*, not framework-driven *observability* of what the planner/agents
are doing.** Everything else — detection, `planner.refine_steer`, `PlanRevised`
emission — still runs; the emitted `PlanRevised` just carries `dry_run=True`.

**Critical constraint (canon #5):** the kill-switch read is
`self._steerer.is_active_steering()`. Do not add a second read of
`_observation_only`, do not inline the check, and do not read the config flag
directly. A missing / raising predicate must resolve to passive. See
`09-steering-ladder-and-gates.md` for the module-helper `steering_is_active`
variant used by maybe-steerer callers.

### 6.4 `observe_refine` and `_emit_plan_revised`

`observe_refine(session, drift)` is an async context manager wrapping a
`planner.refine` call with observability + ContextVar setup:

- On enter: mint `attempt_id`, set `_active_session_var` (so the planner's span
  provider resolves the right session under concurrency), emit `refine_attempted`.
- On exception (`Exception`): emit `refine_failed(failure_kind="llm_error")` and
  re-raise.
- On `BaseException` (i.e. `CancelledError`): emit
  `refine_failed(failure_kind="cancelled")`, mark the cancellation stash
  complete, and re-raise (asyncio contract). This pairs a
  `refine_attempted` with a `refine_failed` even when the refine is cancelled
  mid-flight — sinks never see an unmatched `refine_attempted`.
- On clean exit: reset `_active_session_var`. The caller emits `plan_revised`
  (success) or `refine_failed` (None / rejected).

Used by both the steerer's own refine sites and `ParallelDAGExecutor._refine`
(without it, the parallel path's refines would emit no
`refine_attempted`/`_failed`/`_orphaned_tasks` events).

`_emit_plan_revised(session, revised, drift, *, prev_plan=, attempt_id=,
dry_run=)` is where the actual install happens (post-`#403`), under the
per-session plan lock: `_integrate_correction_supersedes` →
`_repin_current_task_on_supersedes` → `set_session_plan` (when not dry-run) →
watermark stamp → orchestration-state pointer → the correction write-side (§9) →
the `PlanRevised` envelope + `PlanRevisionDiff` sidecar +
`_emit_plan_revision_transitions` (one `TaskTransitioned` per changed task,
`source="plan_revision"`).

### 6.5 `_cancel_inflight_for_revision`

Lives on the `DriftObserver` (`drift_observer.py`) but is called from every
drift-driven install path **after** the revised plan is applied and **before**
`PlanRevised` is emitted. It:

1. Stamps `session._supersede_pending = True` (best-effort) so the executor's
   overlay loop can distinguish this **internal** cancel (switch the in-flight
   agent onto the new plan, restart the passthrough) from an **external** cancel
   (USER_CANCEL / asyncio cancellation, which abort the turn).
2. Short-circuits if this drift already had a pre-refine flag-only cancel
   (`_cancelled_drift_ids`) — firing a second cancel could land on a different
   invocation than the first targeted (`#405` MEDIUM #6).
3. Stamps the per-invocation supersede registry on the `StateStore` (`#405` LOW #7).
4. Calls `request_invocation_cancel(drift=, session=, cancel_inflight_task=True)`,
   which writes the sticky cancel-state flag AND fires `task.cancel()` so the
   coordinator's in-flight LLM call observes `CancelledError` within ~one loop
   tick instead of the full LLM-call duration (the `v15` concurrent-invocation
   bug fix).

Best-effort throughout: an unbound / non-ADK adapter or an empty resolved-id
list is a no-op (the refined plan still lands; the in-flight invocation runs to
completion under the older contract). Turn-1 installs skip this path entirely
(nothing is running).

**Observation-only interaction:** `request_invocation_cancel` itself gates the
flag write on `is_active_steering()`, so under `observation_only=True` no cancel
flag is written — the supersede marker is still stamped (cheap, idempotent) but
the live agent is never touched.

### 6.6 `install_descriptive_growth` — reactive plan growth (`#423`)

The descriptive-growth fallback for an unmatched delegation. Synthesises a
`discovered=True` task and installs it under the per-session plan lock:

```python
async def install_descriptive_growth(self, session, *, agent_name,
                                      tool_args_json, delegation_event_id="") -> Task
```

- `identity_hash = discovery_identity_hash(agent_name, tool_args_json or None)`.
- Idempotent by hash: inside the lock it re-reads `session.plan` (the lock is the
  linearisation point) and returns the existing task if one already carries the
  same `discovery_identity_hash` — two simultaneous delegations of the same
  `(agent, args-token-set)` produce **one** task (`§11.6` dedup linearisability).
- The new task lands as an independent sub-DAG root (no predecessor edges, no
  supersedes) — Rule 7 allows this because the predecessor set is empty. Its id
  is `discovered-<uuid[:12]>` (belt-and-braces against a same-hash id collision).
- Synthesises a `NEW_WORK_DISCOVERED` INFO drift so the `#258` discovery
  carve-out in `_apply_revision` lets the revision land in **both** steering and
  observation mode; then hands off to `install_revision_for_drift`.
- Never raises: on rejection it returns a would-have-been-discovery `Task` so the
  caller can still pin observationally.

**Adaptive-not-predictive constraint (canon #4):** `tool_args_json` MUST come
from the observed `DelegationObserved.tool_args_json` proto field (the
agent-authored event), NOT from a goldfive-side intercept of agent state at pin
time. `discovery_identity_hash(agent_id, tool_args)` (`types.py`) is a
deterministic `sha256[:16]` over `(agent_id, normalized-args-token-set)` (via
`_normalize_args_tokens`: stringify → lowercase → `\w+` tokenise → drop
`_DISCOVERY_STOP_TOKENS`) so a task minted in one process and replayed from a
sink dedups consistently.

---

## 7. `PlanReconciler` — observation → transition (`reconciler.py`)

The reconciler is the heart of the overlay model. One instance per invocation
(one per `ADKAdapter.invoke_passthrough` call); it outlives the invocation so the
runner can call `get_missed_tasks` after the invocation generator ends. It is
deliberately small, framework-agnostic, holds **no sink access**, and calls back
into the steerer via `steerer.transition(...)` for state changes and
`steerer.observe(...)`/`handle_drift` for divergence signals.

### 7.1 Signal streams

The ADK plugin forwards three hooks:

```python
async def on_before_agent(self, *, agent_name, invocation_id="", parent_invocation_id="")
async def on_after_agent(self, *, agent_name, invocation_id="", error=None, summary="", parent_invocation_id="")
async def on_delegation_observed(self, *, from_agent, to_agent, invocation_id="")
```

`before/after_agent` **drive task-state transitions**; `delegation_observed` is
**observability-only** (a no-op beyond the signal that a delegation happened —
the delegated sub-agent fires its own `before/after_agent` pair, which is what
the reconciler picks up). Never move task-state work into
`on_delegation_observed`; it would double-count against the before/after pair.

### 7.2 `on_before_agent` matching logic (tree-agnostic)

1. Record `invocation_id → agent_name` and `invocation_id → parent_invocation_id`
   into `_invocation_agent` / `_invocation_parent` (the maps that power
   parent-chain walks; the reconciler stores what the plugin tells it and has no
   notion of depth).
2. If `_is_host_agent_turn(agent_name, invocation_id)` — the outermost host
   agent's before/after wraps the whole dispatch — return **without** claiming a
   task (the coordinator's own turn must not steal a task meant for a sub-agent).
   Exception: a plan that explicitly assigns a task to the host still matches via
   the normal rule (`_is_host_agent_turn` returns `False` when a task is assigned
   to it).
3. `task = _pick_pending_for_agent(agent_name)` — first PENDING task whose
   `assignee_agent_id == agent_name`. If found: mark RUNNING, record in
   `_running_by_agent[agent_name]` and `_observed_task_ids`, stamp
   orchestration-state current-task keys via
   `_ostate.sync_current_task_from_transition`.
4. If no direct match: **contextual fallback** (`#151`) —
   `_pick_pending_via_parent_chain(invocation_id)` walks the parent chain and
   claims the first ancestor that has a pending plan task (handles plans that
   assigned work to a coordinator the tree routes through via
   `transfer_to_agent` / `AgentTool`).
5. If still no match: emit `PLAN_DIVERGENCE` **once** per off-plan agent
   (`_off_plan_seen` dedup), but only after four suppression checks:
   - already seen (`_off_plan_seen`)? skip.
   - `_agent_has_any_plan_task(agent_name)` (matches ANY task incl. terminal)? →
     a re-visit, not divergence (a coordinator delegating to research twice is
     normal).
   - `_invocation_chain_contains_plan_attached_descendant`? → intermediate
     plumbing, not divergence.
   - any ancestor in `_parent_chain` has a plan task? → leaf-side of a
     coordinator delegation; the ancestor already claimed its task.

The "1-to-many across invocations, 1-to-1 within an invocation" rule: an agent
that re-fires matches either the still-RUNNING task it opened OR (if that task
finished in between) the next PENDING task with the same assignee.

### 7.3 `on_after_agent`, missed tasks, and NOT_NEEDED

`on_after_agent` pops `_running_by_agent[agent_name]`, finds the task, and (if
non-terminal) transitions it: `FAILED` when `error is not None`, else
`COMPLETED` (with `summary` as the detail). It clears the orchestration-state
current-task stamp.

After the invocation ends, the executor's overlay loop calls `get_missed_tasks`:

```python
def get_missed_tasks(self, plan=None) -> list[Task]:
    target = plan if plan is not None else self._session.plan
    # PENDING tasks NOT in self._observed_task_ids (a task seen RUNNING but not
    # completed — e.g. cancelled invocation — is NOT counted as "never exercised")
```

As of `#163`, the overlay executor transitions the missed tasks to
`TaskStatus.NOT_NEEDED` and does **not** dispatch soft follow-ups
(flow-prompted coordinators were re-running their full pipeline on every
follow-up user message). `get_missed_tasks` reads the session's **live** plan
when `plan is None` — correct, because the steerer may have swapped in a revision
mid-run, and missed-task detection should always operate on the latest shape.

> **PROTECTED KEEP.** `get_missed_tasks` (`#163`) is on the never-delete list.
> Even though the overlay executor now marks tasks NOT_NEEDED instead of
> re-dispatching, the method is retained for external callers (custom executors,
> telemetry) that want to surface the coverage gap their own way. Do not delete
> it as "dead code" — see `17-invariants-hazards-history.md` and `#490`'s
> archaeology discipline.

### 7.4 `reset_for_new_plan`

When the steerer installs a revised plan mid-run, the reconciler's
agent→task mapping points at task ids that may no longer exist. The reviser
calls `reset_for_new_plan(new_plan)`, which clears `_observed_task_ids`,
`_running_by_agent`, `_off_plan_seen` (per-plan claim state) but **preserves**
cumulative `observed_agents` / `divergence_events` (historical records for
replay/introspection). The `new_plan` arg is accepted for call-site clarity but
not read — `get_missed_tasks` reads the session's live plan.

---

## 8. `GoldfivePlanner` — the ADK `BasePlanner` (`planners/goldfive_planner.py`)

**Do not confuse this with `LLMPlanner`.** `LLMPlanner` (`planner.py`) is
goldfive's *plan producer*. `GoldfivePlanner` (`planners/goldfive_planner.py`) is
an **ADK `BasePlanner` subclass** that `goldfive.wrap` auto-attaches to every
`LlmAgent` in the tree to do two structural jobs on every LLM call.

### 8.1 Why it exists — the ADK `isinstance` gate

ADK's `flows/llm_flows/_nl_planning.py` gates request-side instruction injection
on `isinstance(planner, PlanReActPlanner)` (see
`_NlPlanningRequestProcessor.run_async`). goldfive wants to inject an
orchestration-context block but **not** inherit `PlanReActPlanner`'s ReAct
response filtering (which would constrain agent output to a tag-based shape). So:

- `GoldfivePlanner` subclasses `BasePlanner` **directly** — NOT
  `PlanReActPlanner`.
- Because `BasePlanner` fails the `isinstance` gate, request-side injection is
  taken over by the plugin: `_GoldfiveADKPlugin.before_model_callback` detects
  the `GoldfivePlanner`, calls `build_planning_instruction`, and appends the
  result to `llm_request.config.system_instruction` via `append_instructions`.
  See `05-adk-plugin.md`.
- The **response-side** gate in `_nl_planning.py` is permissive (fires for any
  `BasePlanner` other than `BuiltInPlanner`), so `process_planning_response` runs
  natively without a workaround.

Do not change `GoldfivePlanner`'s base class to `PlanReActPlanner` to "simplify"
the injection — you would inherit ReAct filtering and break arbitrary agent
output shapes.

### 8.2 Request side — `build_planning_instruction`

Builds a tree-agnostic `[GOLDFIVE ORCHESTRATION CONTEXT]` block from the goldfive
`Session` (reached via the `goldfive._session_context` stash on ADK state; falls
back to reading ADK state directly only for legacy tests). It reads through the
`StateStore`: `pin_current_task()` (id), `_task_title_from_plan` (title from the
typed `Session.plan.tasks`), `goals_summary()`, `get_active_steer()`. The
active-steer line is source-attributed: `Active steer (user): ...` /
`Active steer (goldfive): ...` / `(none)`.

Phase 2.0 of `#271`: the planner reads the goldfive `Session` **directly** —
the old callback-time write to ADK `session.state` (which raced ADK's
optimistic-concurrency contract, `#275`) is gone. It returns `None` only on
internal error, **never** an empty string (empty would make ADK skip the append,
hiding the bug).

Compose-with-user-planner: if a `user_planner` is set (rebound per-invocation via
`bind(...)`), its `build_planning_instruction` runs first and is **prepended**
(user meta-framing above goldfive's per-turn ambient state — LLMs parse the
general-then-specific order more reliably).

### 8.3 Response side — `process_planning_response`

Two filters run in order:

1. **Cancelled-id filter.** Strip `function_call` parts whose `id` is in the
   `cancelled_function_call_ids` set (read via `StateStore`). This drops LLM
   retries of calls goldfive already cancelled (e.g. on USER_STEER).
2. **Three-stage tool-call drift classification** (per retained `function_call`):
   - **Stage 1 — own tool**: name in the running agent's `tools`
     (`_extract_own_tool_names`), or a `report_*` protocol call → legitimate, no
     drift.
   - **Stage 2 — cross-layer agent**: name in `_agent_registry` but not in the
     agent's tools → the LLM tried to delegate past its layer. **The
     `PLAN_DIVERGENCE` drift here is disabled** (`#252`): the detection still runs
     (logs at DEBUG, sets `divergence_fired`) but no `DriftEvent` is constructed —
     `CAPABILITY_MISMATCH` (`#253`) replaced this signal with one grounded in
     actual agent tools rather than declared assignees.
   - **Stage 3 — nowhere**: neither a tool nor a known agent → hallucination →
     emit `CONFABULATION_RISK` (WARNING) via `_emit_tool_call_drift` →
     `steerer.drift.handle_drift`.

**Calls are NEVER blocked** — this is signal-only; the steerer decides
escalation. On a healthy turn (nothing stripped, no divergence fired),
`process_planning_response` returns `None` — ADK's "leave response untouched"
signal. It only returns the `kept` list when it actually modified something
(`stripped_count or divergence_fired`). Preserve this no-op-on-healthy-turn
behaviour; returning a fresh list on every turn forces ADK to rebuild parts
needlessly.

Compose-with-user-planner: goldfive's structural filters run **first** (the user
planner sees structurally-clean parts); then `user_planner.process_planning_response`
runs on `kept`.

---

## 9. Corrections — CORRECT-kind write-side (`_correction_injection.py`)

This module owns the **write-side** of the CORRECT-kind supersedes prompt
injection (`#251` Stream D). When a refine lands a new task with
`supersedes_kind == CORRECT`, the agent that owns the new task should be told —
on its next turn — to do the corrected work, without re-running the old
(historically-COMPLETED) task.

### 9.1 The three streams (context)

- **Stream A** (`types.py`): `SupersessionKind` + the CORRECT topology (old stays
  COMPLETED, new attached as a correction child).
- **Stream B** (`adapters/adk_llm_instrumentation.py`): the read-side — a dynamic
  instruction resolver that appends the `(agent_name, current_task_id)`-keyed
  pending correction onto the agent's system prompt every turn (via
  `StateStore.get_correction`).
- **Stream C** (`adapters/_adk_plugin.py` + `_adk_state_protocol.py`): cooperative
  cancel of the offending in-flight invocation when the drift is CRITICAL.

### 9.2 The state key — full agent path (`#479`)

```python
def pending_correction_key(agent_name: str, task_id: str) -> str:
    return f"{_sp.KEY_PENDING_CORRECTIONS}.{agent_name}.{task_id}"
    # KEY_PENDING_CORRECTIONS == "goldfive.pending_corrections"
```

`_normalize_agent_name` keeps the id **verbatim** (whitespace-stripped only): a
fully-qualified path like `team_a.researcher` stays intact. This is the `#479`
fix — collapsing to the last dotted segment made `team_a.researcher` and
`team_b.researcher` **collide** on one key (canon invariant #6: lifecycle gates
need stable identity keys, and the key must be as specific as the thing it
gates). Do not "simplify" the key back to a bare agent name.

The same key formula is re-exported from
`adapters.adk_llm_instrumentation.pending_correction_key` (the read-side spelling)
so the write-side does not have to import the ADK adapter module; both spellings
produce the same key.

### 9.3 `queue_corrections_for_revision`

Called from `PlanReviser._emit_plan_revised` right after
`_integrate_correction_supersedes` rewired the DAG. For every new task with
`supersedes_kind == CORRECT`:

1. Look up the superseded task (in `revised` first — the CORRECT topology keeps
   it — then `prev_plan` as a defensive fallback).
2. `build_correction_payload(new_task=, old_task=, drift=, revision_number=)` —
   the payload carries `agent_name`, `task_id`, `superseded_task_id`,
   `superseded_task_title`, `drift_kind` (lower-case wire form), `drift_reason`,
   `revision_number`, `issued_at_ms`.
3. `write_correction(session, payload)` stamps a plain `dict` under
   `goldfive.pending_corrections.<agent>.<task_id>` (plain dict so sinks/persistence
   round-trip it without knowing this module's contract).

Skips (with a DEBUG log) a CORRECT task with no assignee (no `(agent, task)` key
to write) or a superseded id absent from both plans (the validator would have
rejected it — defensive).

**Correction text is directive, not diagnostic** (`#250`/`#252`/`#253`/`#259`):
the *prompt* rendered by the read-side (`format_correction_block`) tells the LLM
what to do on the new task, **not** what went wrong with the old task —
problem-naming language ("failed", "broken") makes LLMs invent workarounds and
apologies instead of proceeding. The diagnostic fields stay in the dict for
programmatic consumers (sinks/observability) but are not rendered into prompt
text.

### 9.4 The `pending_corrections` GC protocol

| Function | Trigger | Effect |
|----------|---------|--------|
| `clear_correction(session, agent_name=, task_id=)` | `report_task_started` on the correction task (`goldfive.reporting`) | The agent acknowledged the new task → stop re-injecting the correction block. |
| `clear_corrections_for_task(session, task_id)` | plan-revision GC in `_emit_plan_revised` | A new revision superseded a correction task → drop every correction keyed on that task id (task-scoped, matches across all agents). |
| `clear_obsolete_corrections_on_revision(session, revised)` | paired with `queue_corrections_for_revision` | Walk `revised` for any task whose `supersedes` points at an earlier id; drop corrections keyed on that earlier id (CORRECT or REPLACE — both make the prior correction obsolete). Idempotent. |

`is_pending_correction_key(key)` gates the sweeps to the pending-corrections
family (prefix `goldfive.pending_corrections.`).

---

## 10. Known future work (deferred — do NOT present as current)

These are real, planned, and **not on main**. If you find yourself reaching for
one of them, stop — the design was deliberately deferred, and the reason
matters.

| Deferred item | Rationale for deferral |
|---------------|------------------------|
| **twin-refine-pipeline extraction** (splitting the refine prompt/validate core into a reusable twin) | Blocked on the agency-preservation branch-merge decision. Main-side code must not copy from that branch. |
| **evidence-ledger replacement of the ~7 stacked `handle_drift` suppression gates** | Same block. The current stacked gates (stale-index, in-flight, addressed-watermark, suppression-window, etc.) stay until the ledger lands. |
| **`plan_mode=forecast` planner flag** | Lives on the agency-preservation branch, default-OFF. **It does not exist in `goldfive/config.py` on main** — do not reference it in main-side code or docs as if it does. |
| **judge windowing / cadence expansion** and **judge-facade dispatch authority** | Blocked on a judge regression harness. Not a planning concern directly, but it feeds the refine trigger rate. |
| **checkpoint-rollback / tool-gating hold / fork-and-judge** | Stage-4 ambition, bench-gated. |

The agency-preservation branch (`#453`–`#474`, unmerged) holds Stages 1–3 behind
default-OFF flags; step 13b (three-arm bench + measurement-gated default flips +
hard deletions) is LOCKED on explicit user sign-off. **Do not copy code from that
branch into main, and do not write doc text claiming its features exist on
main.**

---

## 11. Common mistakes (concrete wrong edits and the correct alternative)

| Wrong edit a weak model plausibly makes | Why it is wrong | Correct alternative |
|------------------------------------------|-----------------|---------------------|
| Passing `observed_actions=...` into a **new** `planner.refine` caller to "give the LLM more context". | `observed_actions` is a **known dead channel** in production — no production caller passes it, and prompt selection is by drift kind alone since `#220`. Adding it to one new caller creates an inconsistent, half-wired channel. | Do **not** wire `observed_actions` into new callers. If you genuinely need observed activity in the prompt, put it in `drift.detail` / `drift.trigger_input` (which the goal-aware prompt already surfaces). |
| Mutating `task.status` or `plan.tasks` in place. | `Task` and `Plan` are `frozen=True` (`#247`). Attribute assignment raises `FrozenInstanceError`. | Derive a new plan via `with_task_status` / `replace_task` / `add_tasks` / `replace_edges` / `bump_revision`, then `set_session_plan` inside `channel_processor_active()`. |
| Calling `set_session_plan` outside `channel_processor_active()`. | Trips the single-writer guard — WARNING in prod, `PlanOwnershipViolation` under pytest / strict mode. | Wrap in `with channel_processor_active():`. Only the steerer's channel processor owns `session.plan` swaps. |
| Transitioning a task directly (`session.plan = with_task_status(...)`) instead of through `TaskStateMachine`. | Bypasses the terminal-idempotency guard, the `#152` orchestration-state sync, the `#486` drift-condition resolution hook, the lineage bookkeeping, and the `TaskTransitioned` / `TaskStarted` emissions. | Call the appropriate `mark_task_*` method on the state machine. |
| Reading `_observation_only` (or the config flag) directly to gate a plan install. | Canon #5: the ONLY sanctioned read is `is_active_steering()` / `steering_is_active(steerer)`. A second predicate drifts out of sync and can make observation-only leak. | Call `self._steerer.is_active_steering()`. Missing/None/raising ⇒ passive. |
| Adding a fourth carve-out to the `_apply_revision` gate (e.g. "always install CRITICAL"). | The three carve-outs (bootstrap / user-authored / discovery) are the audited set. A CRITICAL carve-out would make observation-only *correct* the agent — the exact thing it must not do. | Leave the gate at three carve-outs. If you need a new "always-install" class, it must be reviewed against the observation-only contract. |
| Re-populating `Task.assignee_agent_id` from the LLM's refine output. | `#252`: assignees are observational, not declarative. `_plan_from_json` drops them on purpose; the reconciler observes the real assignee. Re-populating reintroduces the predictive model. | Leave assignee `""` for LLM plans. If you need "who should run this", that is a reconciler-observed fact, not a plan field. |
| Inlining `{COMPLETED, FAILED, CANCELLED}` (forgetting `NOT_NEEDED`). | Drops the overlay-model `NOT_NEEDED` terminal state (`#141`/`#485`); a NOT_NEEDED task would be treated as still-runnable. | `from goldfive.types import TERMINAL_TASK_STATUSES` and use it. |
| Deleting `get_missed_tasks` / `PLAN_DIVERGENCE` machinery / `LOOPING_*` planner surfaces as "dead code". | These are PROTECTED KEEP decisions (`#163`, `#252`-disabled-but-KEEP, `#204`/`#206`). `#490`'s archaeology discipline: verify against history before deleting. | Do not delete without explicit human sign-off. See `17-invariants-hazards-history.md`. |
| Synthesising a clone of the prior plan with `revision_index + 1` when a refine fails. | That masquerades a failed refine as a successful no-op revision — the silent-fallback `#133` eliminated. | Return `None` on exhaustion; emit `REFINE_VALIDATION_FAILED` (unless the failure is `_EMPTY_RESPONSE_ERROR`, `#182`) and let the steerer's backoff take over. |
| Retrying an empty/non-string LLM response. | `#182`: a small-model "no answer" is a model-quality issue, not a planner failure; retrying doubles cost and escalating via `REFINE_VALIDATION_FAILED` adds noise. | Treat `_EMPTY_RESPONSE_ERROR` as terminal: log INFO, return `None`, no escalation emit. |
| Collapsing `team_a.researcher` correction keys to a bare `researcher`. | `#479`: same-named agents in different subtrees collide, so a correction for one agent leaks to the other (canon #6). | Keep the full path in the key via `_normalize_agent_name` (verbatim, whitespace-stripped only). |
| Changing `GoldfivePlanner`'s base class to `PlanReActPlanner` to make request injection work "natively". | You inherit ReAct response filtering, constraining arbitrary agent output. The plugin's `before_model_callback` already handles injection precisely to avoid this. | Keep the `BasePlanner` base + the plugin injection path (`05-adk-plugin.md`). |
| Making `process_planning_response` return the parts list on every turn. | Forces ADK to rebuild parts even when goldfive changed nothing; the `None` return is ADK's "leave untouched" fast path. | Return `None` unless `stripped_count or divergence_fired`. |
| Adding `GOAL_DRIFT` to `_REASONING_PIPELINE_DRIFT_KINDS` so an on-task verdict resolves it. | A reasoning-scoped verdict carries no evidence about a trajectory-level goal question; `GOAL_DRIFT` resolves only at task-terminal (`#486`). | Leave `GOAL_DRIFT` out of that set. It resolves through `resolve_conditions_for_terminal_task`. |
| Skipping `validate(for_revision=True, prior=session.plan)` in a new install path. | A Rule 6 regression (dropped/regressed terminal task) silently corrupts the DAG; downstream readiness checks misbehave. | Always validate against the prior before install; on failure emit `SCHEMA_VIOLATION` and return `False`. |
| Deleting the `_planner_refine_accepts_available_agents` signature probe. | Custom / pre-`#151` planners whose `refine` lacks `available_agents=` would raise `TypeError` on every drift. | Keep the probe; it is what keeps the `#151` kwarg additive. |
| Routing a genuine operator STEER through `install_revision_for_drift`. | It **raises `ValueError`** on `USER_STEER` by design — the active-steer bookkeeping + dedup only fire on the user-steer path. | Route operator STEERs through `install_revision_for_user_steer`. |

---

## 12. Verification checklist

Run these after touching this subsystem. Commands assume repo root
`/home/sunil/git/goldfive` with the dev+adk extras synced:

```bash
uv sync --extra dev --extra adk
```

### 12.1 Targeted test files

```bash
# Plan/Task model + frozen-plan invariant + single-writer guard
uv run pytest -q tests/test_immutable_plan.py

# Planner surface: generate/refine/handle_turn/observed-actions/correction-prompt
uv run pytest -q \
  tests/test_planner.py \
  tests/test_plan_refinement.py \
  tests/test_planner_goal_aware_refine.py \
  tests/test_planner_handle_turn.py \
  tests/test_planner_observed_actions.py \
  tests/test_planner_correction_prompt.py \
  tests/test_planner_refine_guidance.py \
  tests/test_planner_close.py \
  tests/test_refine_steer_retry_feedback.py

# Supersession + causal replacement + validator + task binding follows revision
uv run pytest -q \
  tests/test_supersede_causal_replacement.py \
  tests/test_task_supersedes_validator.py \
  tests/test_task_binding_follows_revision.py

# Plan reviser + install pipeline + observation-only carve-out + atomicity
uv run pytest -q \
  tests/test_plan_reviser.py \
  tests/test_observation_only_emit_supersedes_carveout.py \
  tests/test_refine_atomicity_events.py \
  tests/test_refine_emit_parity.py \
  tests/test_plan_revised_trigger_id.py \
  tests/test_cancel_inflight_on_refine.py

# Reconciler + task state machine + intra-session carry-forward
uv run pytest -q \
  tests/test_plan_reconciler.py \
  tests/test_task_state_machine.py \
  tests/test_intra_session_plan_carry_forward.py

# GoldfivePlanner (ADK BasePlanner) + corrections
uv run pytest -q \
  tests/test_goldfive_planner.py \
  tests/test_correction_injection.py \
  tests/test_e2e_plan_causal_correction.py

# Runner install-revision policy + descriptive-growth race
uv run pytest -q \
  tests/test_runner_install_revision_stall.py \
  tests/test_runner_revision_rejection_policy.py \
  tests/test_plan_descriptive_growth_race.py
```

### 12.2 Full suite + lint (always before you push)

```bash
uv run pytest -q           # ~30s, expect ~2912 passed / ~61 skipped
ruff check .               # must stay clean; do NOT ruff-format (repo is not format-clean)
```

### 12.3 Grep guards for the invariants

```bash
# (canon #5) There must be exactly ONE kill-switch read in the install gate.
# Expect the hit to be `self._steerer.is_active_steering()` and NOTHING reading
# `_observation_only` directly in plan_reviser.py:
grep -n "is_active_steering\|_observation_only" goldfive/plan_reviser.py
#   -> _apply_revision should show is_active_steering(); no direct _observation_only reads.

# (canon #7) Every session.plan swap must go through set_session_plan.
# A raw `session.plan =` outside types.py/set_session_plan is a smell:
grep -rn "\.plan = " goldfive/ | grep -v "set_session_plan\|# " | grep -v "types.py"

# (canon #2) No regex-based NL classification reintroduced in the planning modules:
grep -rn "re\.compile\|re\.match\|re\.search" goldfive/planner.py goldfive/reconciler.py goldfive/plan_reviser.py
#   -> the only legitimate regex is the discovery-args tokeniser in types.py;
#      the planning modules above should return no NL-classification regex.

# (canon #6) Correction keys must use the full agent path (verbatim):
grep -n "_normalize_agent_name\|pending_correction_key" goldfive/_correction_injection.py

# (#485) TERMINAL_TASK_STATUSES must be imported, never re-inlined:
grep -rn "COMPLETED.*FAILED.*CANCELLED" goldfive/ | grep -v "TERMINAL_TASK_STATUSES\|is_terminal\|types.py"
#   -> ideally no hits; any hit is a candidate re-inlined terminal set to fix.

# (#252) Assignees must stay dropped in the JSON parser:
grep -n "assignee_agent_id=\"\"" goldfive/planner.py
#   -> _plan_from_json must show assignee_agent_id="".
```

### 12.4 Behavioural spot-check (when you changed a hot path)

If you changed `_apply_revision`, the reconciler transition logic, or a
`TaskStateMachine` method, drive a real run and confirm the plan actually
installs / reconciles — a passing unit test can still miss a broken flow (see the
`v18 harness PASSED but flow was broken` lesson). Use the project's run/verify
skills rather than trusting DB-only checks; confirm on the wire that a
`PlanRevised` event carries `dry_run=False` under active steering and
`dry_run=True` under the default `observation_only=True`.
