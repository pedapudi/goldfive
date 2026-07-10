# 04. Executors and the Control Channel

> **⚠ Predates the agency-preservation merge.** This chapter describes the
> pre-merge mechanics; the merge (PRs #453–#504) renamed `NUDGE`→`SIGNAL`,
> replaced corrective templates with advisory observer notes, and added the
> default-OFF ledger/signal regimes. Default-flag behavior described here is
> still accurate; for the merged as-built state read
> `docs/design/AGENCY-PRESERVATION.md` §6 first — it wins on any conflict.

**Read this chapter when...**

- You need to change how a `Plan` is walked to completion: the sequential legacy loop,
  the sequential overlay loop, or the parallel DAG walker (`goldfive/executors/sequential.py`,
  `goldfive/executors/parallel.py`).
- You are adding, removing, or changing the semantics of a control message —
  `PAUSE`, `RESUME`, `CANCEL`, `STEER`, `REWIND_TO`, `STATUS_QUERY`, `INTERCEPT_TRANSFER`,
  `APPROVE`/`REJECT`, `GOLDFIVE_STEER`, `GOLDFIVE_PAUSE_ESCALATE`
  (`goldfive/control.py`, `goldfive/executors/_control.py`).
- You are debugging: a run that hangs on a pause, a `RunAborted` you did not expect, a
  STEER that "does nothing", a nudge that never replays (or replays forever), a task left
  stuck `PENDING` at run end, a supersede-cancel that terminated a turn it was meant to
  continue, or a control message that one executor honours and the other ignores.
- You were told to "add an intervention" or "make goldfive do X mid-run" — read the
  parity table (**Sequential vs parallel: the honest parity table**) first; an
  intervention wired into only one executor and only one of its two loops is the single
  most common defect in this subsystem.
- You are touching the pause deadline / TERMINATE path (`abort_expired_pause`,
  `pause_deadline_s`, #482).

**Files covered**

| File | What lives there |
|---|---|
| `goldfive/executors/sequential.py` | `SequentialExecutor` — the default executor. Two run modes: the legacy per-task loop (`run`) and the overlay loop (`_run_overlay` + its post-#489 stage methods). Plus the module-level plan helpers (`_pick_next_task`, `_has_live_replacement`, `_unreachable_pending_task_ids`, `_lineage_root`, …) and `build_task_nudge`. |
| `goldfive/executors/parallel.py` | `ParallelDAGExecutor` — the topological-stage walker. `run`, `_run_stage`, `_refine`, `_apply_pre_stage_controls`. **No overlay mode.** |
| `goldfive/executors/_control.py` | Control-message *dispatch*: `dispatch_control`, `drain_controls`, `ControlOutcome`, `_ControlCancelled`, `build_status_snapshot`, `pause_deadline_s`, `abort_expired_pause`, and the private `_rewind_plan` / `_resolve_approval`. |
| `goldfive/executors/_shared.py` | Non-dispatch glue shared verbatim by both executors: `apply_steer`, `mark_cancelled_if_live`, `tag_adapter_cancel_user_steer`, `emit_drift_event`, `emit_pipeline_failure_drift`, `CANCEL_REASON_USER_STEER`. |
| `goldfive/control.py` | The `ControlChannel` primitive, the `ControlKind` / `AckResult` enums, and the `ControlMessage` / `ControlAck` dataclasses. Dependency-light (asyncio only). |
| `goldfive/executors/__init__.py` | Re-exports `SequentialExecutor`, `ParallelDAGExecutor`, `build_task_nudge`. |

Related chapters: the `Runner` that *calls* `executor.run()` and owns `RunStarted` is
03-runner-and-conversation.md; the `Steerer` that `dispatch_control` and the executors
call into (`steerer.transition`, `steerer.drift.observe`) is 09-steering-ladder-and-gates.md;
`PlanReconciler` (the overlay's coverage tracker) is 05-adk-plugin.md and
10-planning-and-revision.md; `planner.refine` is 10-planning-and-revision.md; the
`observation_only` kill-switch and `steering_is_active` are 09-steering-ladder-and-gates.md;
`Session` field ownership (`session.plan`, `session.pending_nudges`,
`session._supersede_pending`, `completed_outputs`) is 11-state-ownership.md; the events these
paths emit (`RunAborted`, `PlanRevised`, `TaskCancelled`) are 12-events-sinks-telemetry.md;
`report_awaiting_approval` and the `APPROVE`/`REJECT` waiters are
13-reporting-tools-and-approval.md.

**Invariants that bind you here**

1. **No prompt-cooperation contracts.** Termination, cancellation, and control must work
   even if the wrapped agent never calls a goldfive reporting tool and never follows an
   instruction. Concretely: `CANCEL` cancels the in-flight `asyncio.Task` directly
   (`_cancel_invoke_task`); the executor auto-transitions a task to `COMPLETED`/`FAILED`
   on a clean/errored return even when the agent emitted no `report_task_*` call; the
   overlay's `_sweep_unreachable_pending` disposes of tasks the tree never touched. Never
   add a code path that *only* works when the agent cooperates.
2. **`observation_only=True` is the production default and is STRICTLY passive.** The
   only sanctioned read of the kill-switch is `steering_is_active(steerer)` (module helper
   in `goldfive/steerer.py`, delegating to `DefaultSteerer.is_active_steering`;
   missing/None/raising → PASSIVE). Every place the executor would *enforce* — abort on a
   failed-task-without-replacement, replay a queued nudge, honour a `GOLDFIVE_PAUSE_ESCALATE`,
   cancel the in-flight invoke for a goldfive pause — is guarded by
   `not steering_is_active(steerer)` and becomes a log-and-continue. These are
   defense-in-depth carve-outs; the *primary* gate is on the steerer's dispatch side. Do
   not read `session._observation_only`, `config.observation_only`, or any raw flag here.
3. **Any ADK tree shape must work, including coordinator + `AgentTool`.** The overlay
   drives the tree through `adapter.invoke_passthrough(user_input, …)` exactly once per
   iteration and observes coverage via `PlanReconciler`; it does not assume the tree has a
   task-per-agent shape or that it will visit tasks in plan order.
4. **Adaptive over predictive.** The executors react to observed plan state
   (`session.plan` task statuses after each invocation), never to a prediction of what the
   agent will do. `_pick_next_task` / `topological_stages()` read live `Task.status`;
   drift enters via `steerer.drift.observe(result, session)` on the raw invocation
   envelope, not by intercepting at dispatch time.
5. **Lifecycle gates need stable identity keys.** The per-lineage retry cap keys on
   `_lineage_root(task.id)` (retry-prefix-stripped), the refine back-off keys on
   `(drift.kind.value, task_id)`, the supersede-cancel discriminator keys on the
   `StateStore` per-`invocation_id` registry plus the legacy `session._supersede_pending`
   bool. Do not re-key any of these on an LLM-minted id that churns per turn.
6. **Frozen plan.** `Plan` and `Task` are frozen dataclasses (goldfive#247). Every status
   change goes through `steerer.transition(...)` or a `with_task_status`/`set_session_plan`
   swap under `channel_processor_active()`. Never mutate `task.status` in place in an
   executor.

---

## The two executors at a glance

goldfive ships two `Executor` implementations. Both satisfy the same protocol
(`goldfive/protocols.py`, `Executor`) and both are driven by the `Runner`
(03-runner-and-conversation.md), which passes `plan`, `session`, `adapter`, `steerer`,
`planner`, `sinks`, and an optional `control` channel into `executor.run(...)`.

| | `SequentialExecutor` | `ParallelDAGExecutor` |
|---|---|---|
| File | `goldfive/executors/sequential.py` | `goldfive/executors/parallel.py` |
| Unit of work | one task at a time (legacy) or one whole-tree passthrough (overlay) | one topological *stage* (a batch of tasks whose predecessors are all `COMPLETED`) run under `asyncio.gather` |
| Default via `goldfive.wrap` | **yes** — `SequentialExecutor(overlay_mode=True)` (`goldfive/convenience.py`, `resolved_executor`) | no — opt in by passing `executor=ParallelDAGExecutor(...)` |
| Overlay mode | yes (`overlay_mode=True`) | **no such thing** |
| Refine trigger | via the steerer's reporting-tool handlers + `steerer.drift.observe`; the executor itself does not call `planner.refine` | inline in `run` after each stage (`self._refine(...)`) |
| Concurrency | none (one invoke in flight) | `max_concurrency` semaphore per stage (0 = unbounded fan-out) |

**Both executors are provider-agnostic.** They talk to the agent only through
`AgentAdapter.invoke(task, session)` (per-task) and, for the overlay,
`AgentAdapter.invoke_passthrough(user_input, session=…, reconciler=…)`. The ADK-specific
plumbing lives in the adapter (06-adapters-and-instrumentation.md). Neither executor
imports the ADK adapter module — that is why `_shared.CANCEL_REASON_USER_STEER` is a plain
string duplicated from `goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER` (see the comment
at `_shared.py`, `CANCEL_REASON_USER_STEER`).

**Lifecycle ownership.** Neither executor emits `RunStarted` — the `Runner` owns that (see
`Runner._emit_run_started`, and the `NOTE:` comments in both `run` bodies). The executor
owns the terminal `RunCompleted` / `RunAborted`, the `PlanRevised` it detects/generates,
and (via the steerer) the per-task `Task*` events.

---

## The executor invocation contract (how `run` is called)

`SequentialExecutor.run` signature (`goldfive/executors/sequential.py`):

```python
async def run(
    self,
    *,
    plan: Plan,
    session: Session,
    adapter: AgentAdapter,
    steerer: Steerer,
    planner: Planner,
    sinks: list[EventSink],
    control: ControlChannel | None = None,
    user_input: str = "",
) -> ExecutionOutcome:
```

`ParallelDAGExecutor.run` is identical **except it has no `user_input` parameter** — the
parallel executor has no overlay mode and never re-invokes with a composed user message.
This is the first honest asymmetry: if you add a parameter to one `run`, decide
deliberately whether the other needs it.

Both `run` bodies start with the same two steps, in this order:

1. **Pin the plan onto the session** under the single-writer guard:

   ```python
   with channel_processor_active():
       set_session_plan(session, plan)
   ```

   `set_session_plan` warns (or, in strict mode, raises) when called outside
   `channel_processor_active()` — that is the structural enforcement of "single writer onto
   `session.plan`" (goldfive#247). The initial pin is a legitimate channel-processor write,
   hence the wrapper. See 11-state-ownership.md.

2. **Bind sinks + planner into the steerer**: `steerer.bind(sinks=sinks, planner=planner)`
   so reporting-tool handlers can emit events and trigger `planner.refine` on drift.
   (The parallel executor wraps this in a `try/except` and logs at debug; the sequential
   one does not — a stub steerer without `bind` is tolerated in the parallel tests.)

Then the sequential executor branches on `overlay_mode`:

```python
if self.overlay_mode and callable(getattr(adapter, "invoke_passthrough", None)):
    return await self._run_overlay(...)
# else: fall through to the legacy per-task loop
```

The `callable(getattr(adapter, "invoke_passthrough", None))` guard is deliberate:
third-party `AgentAdapter` implementations that predate the overlay refactor and do not
expose `invoke_passthrough` fall back to the legacy loop even if the caller passed
`overlay_mode=True`. Do not remove this duck-typed guard.

`ExecutionOutcome` (`goldfive/results.py`) carries `success: bool`, `session: Session`, and
`reason: str`. Every `run` return path builds one; the `Runner` reads it to decide the turn
outcome.

### Construction knobs (`SequentialExecutor.__init__`)

| kwarg | Default | Meaning |
|---|---|---|
| `max_task_invocations` | `None` (unbounded) | Ceiling on total adapter `invoke()` calls per `run`. Each eligible task burns one; mid-run revisions do not refund. |
| `max_retries_per_task_lineage` | `3` | Per-lineage invocation cap (see `_lineage_root`). Bounds a runaway refine loop to a small constant multiple of plan size. `TASK-LIFECYCLE.md §7.7`. |
| `fail_fast` | `True` | First `FAILED`-without-live-replacement task aborts the run. |
| `overlay_mode` | `False` (but `goldfive.wrap` passes `True`) | Switch to `_run_overlay`. |
| `fail_fast_on_invoke_cancel` | `None` → env `GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL` | Governs **only** the overlay's goldfive-internal supersede-cancel branch. Explicit `True`/`False` wins over env. External cancels always abort. |

`_MAX_NUDGE_REPLAYS = 3` is a **class attribute** (not a constructor kwarg) so subclasses can
tune it without expanding the public constructor (goldfive#202). Two legacy kwargs are
accepted-and-deprecated: `max_plan_reinvocations` (aliases to `max_task_invocations`) and
`max_follow_up_rounds` (no effect; the overlay's soft follow-up loop was removed in #163).

---

## Sequential legacy run loop (`run`, non-overlay)

The legacy loop is the older, per-task model. It is **not** the default under
`goldfive.wrap` (overlay is), but it is the default when you construct
`SequentialExecutor()` directly with no `overlay_mode`, and it is what the vast majority of
`tests/test_sequential_executor.py` exercises. Understand it first: the overlay reuses many
of its helpers and its abort semantics.

### The loop skeleton

The outer `while` is bounded by `max_task_invocations` (default `None` = unbounded):

```python
while self.max_task_invocations is None or invocations < self.max_task_invocations:
```

Each iteration, in order:

1. **Drain pre-task controls** via `_apply_pre_task_controls(...)` (covered below). This
   returns `(stop, steer_msg)`; a `CANCEL` raises `_ControlCancelled` (caught → `run_failed`),
   a `PAUSE` blocks inside the helper until `RESUME`/`CANCEL`/`STEER`. A returned
   `steer_msg` is fed through `apply_steer(...)` so the steerer swaps `session.plan`.
2. **Detect an out-of-band plan revision.** Compare `current_plan.id` /
   `current_plan.revision_index` against `last_plan_id` / `last_revision_index`; if they
   changed (the steerer swapped the plan on the session), emit a `plan_revised_event`. Two
   emitters exist for the same swap (the steerer's `_emit_plan_revised` and this
   swap-detector); the log line at `SequentialExecutor: plan-swap detected` disambiguates
   who emitted it (Phase 2.X / goldfive#271 Gap 2).
3. **Pick the next task**: `task = _pick_next_task(current_plan)`. `None` → break (plan done
   or everything blocked).
4. **Per-lineage cap check.** Before spending an invocation, look up
   `lineage_invocations[_lineage_root(task.id)]`; if it is already at
   `max_retries_per_task_lineage` (default 3), transition the task to `FAILED` *without
   invoking the adapter* and either abort (`fail_fast`) or `continue`. This bounds a runaway
   refine loop (`t0 → retry_t0 → retry2_retry_t0 → …`) to a small constant, independent of
   how many plan revisions happen. See the `max_retries_per_task_lineage` docstring and
   `TASK-LIFECYCLE.md §7.7`.
5. **Auto-announce RUNNING.** If `task.status == PENDING`, `steerer.transition(task.id,
   RUNNING, …)` so agents that never call `report_task_started` still produce a
   `TaskStarted` (invariant 1). Idempotent once the task leaves `PENDING`.
6. **Invoke with control**: `_invoke_with_control(...)` races `adapter.invoke(task, session)`
   against the control channel and returns `(kind, payload)`:
   - `("result", InvocationResult | None)` — normal completion.
   - `("adapter_error", BaseException)` — `invoke` raised.
   - `("cancelled", reason_str)` — a `CANCEL` interrupted it.
   - `("steer", ControlMessage)` — a `STEER` interrupted it.
7. **Fold the outcome** (see below).

### Folding the invoke outcome (legacy)

| outcome_kind | What the loop does |
|---|---|
| `cancelled` | `mark_cancelled_if_live(task, cancel_reason=session._last_cancel_reason_prefix)`; set `run_failed`; break. The prefix (`user_cancel:<annotation_id>`) was stashed by dispatch (#205). |
| `adapter_error` | `log.exception`; set `failure_reason`; `run_failed`; break. |
| `steer` | Stamp a `user_steer:<annotation_id>` cancel reason via `_steer_cancel_reason_prefix`, `mark_cancelled_if_live`, `apply_steer(...)`, then `continue` (loop picks up the steerer-swapped plan). |
| `result` | Continue below. |

For `result`:

- If `result.error is not None` (adapter reported an error on the envelope but did not
  raise), record it; the auto-transition routes it to `FAILED`.
- **Route the raw envelope through drift detection**: `await steerer.drift.observe(result,
  session)`. This is what makes drift enter on the sequential path via the raw invocation
  envelope, not only via reporting-tool handlers — it mirrors the parallel executor. A
  raise here is surfaced as an INFO `CUSTOM` `emit_pipeline_failure_drift` (goldfive#134),
  never swallowed.
- **Record the full output** (zicato#12 mechanism 1): `session.completed_outputs[task.id] =
  result.full_text or result.text` on a clean return, *independent* of whether the agent
  self-reported. This is the canonical gradeable artifact; the self-reported summary in
  `completed_results` is separate metadata and never shadows it.
- **Auto-transition** if the task is still `PENDING`/`RUNNING`: `FAILED` on
  `invocation_error`, else `COMPLETED` with `detail=result.text`. `mark_task_*` is
  idempotent on terminal states, so agents that *did* self-report are unaffected
  (invariant 1).
- **`fail_fast` handling on `FAILED`** — this is subtle and load-bearing:
  - If the failed task has a **live replacement** in the current plan
    (`_has_live_replacement`), the failure is **not** fatal (`failure_reason = ""`); the
    replacement is the forward-progress path (goldfive#202).
  - Else if **`not steering_is_active(steerer)`** (observation_only), **do not abort** — the
    replacement-producing refine is dry-run under observation_only so there is no
    replacement to install; aborting would defeat the passive contract. Log and fall
    through (goldfive#260, invariant 2).
  - Else `run_failed = True; break`.

### The legacy terminal cascade (post-loop)

After the loop, `run` runs a sequence of gates, each of which drains background steerer
tasks (`_drain_steerer_at_run_boundary`) then emits its terminal event. In order:

1. `run_failed` → `RunAborted(reason=failure_reason)`.
2. **Invocation-budget exhaustion**: if a finite `max_task_invocations` was hit *and* a
   task is still pickable → `RunAborted`.
3. **`fail_fast=False` residual failures**: `_fatally_failed_task_ids(...)` non-empty →
   `RunAborted`, *unless* `not steering_is_active(steerer)` (observation_only carve-out).
4. **Reachability audit** (belt-and-suspenders): `_pending_task_ids(...)` non-empty after
   `_pick_next_task` declined all of them → those tasks are orphaned. Transition each to
   `CANCELLED` with `run_aborted:orphaned by plan revision failure`, emit a **CRITICAL
   `PLAN_DIVERGENCE`** drift (`_plan_divergence_drift_event`), then `RunAborted`.
5. **Goal predicates**: `evaluate_goal_predicates(session)` returns a non-`None` reason →
   `RunAborted` (PLAN-LIFECYCLE.md §6.1).
6. Otherwise `RunCompleted(outcome_summary=_outcome_summary(session))` and
   `ExecutionOutcome(success=True)`.

`_drain_steerer_at_run_boundary` (goldfive#243) drains the steerer's per-session background
drift/judge tasks *before* the terminal emission, so a drift cascade dispatched at end of
turn N (e.g. a `report_*`-triggered refine) cannot outlive the turn and run against an
abandoned session. It is duck-typed on
`steerer.drift.drain_session_background_tasks(session_id=…)`; custom steerers without it
fall through cleanly.

---

## Sequential overlay mode — why it exists

The overlay is the default for real ADK coordinator trees (`goldfive.wrap`). Understanding
*why* it exists prevents the most damaging class of "simplification" a weak model attempts
here.

### The regression history

The legacy loop drives **per task**: it picks a task, invokes the adapter with that task,
folds the result, picks the next. For a coordinator + `AgentTool` tree (a host agent that
delegates to sub-agents), this is wrong: the tree runs its *own* natural flow the moment
you invoke it. Driving it task-by-task means goldfive keeps re-invoking the whole
coordinator pipeline once per plan task — the **coordinator-flow-looping regression**.

The overlay model (goldfive#141, refined by #163) fixes this:

1. Call `adapter.invoke_passthrough(user_input, …)` **once** with the operator's original
   request and a `PlanReconciler` plugged into the plugin, watching the tree run its
   natural flow. The reconciler transitions plan tasks as it observes sub-agent activity.
2. When the invocation ends, decide what to do with tasks the tree did not exercise.
3. Emit terminal `RunCompleted`/`RunAborted`.

**goldfive#163** is the critical follow-up. The original overlay (#141) had a *soft
follow-up loop*: at invocation end, any still-`PENDING` task was re-dispatched as a new
user message. Flow-prompted coordinators re-ran their **entire pipeline** on every such
message, turning a ~10-minute run into 40+ minutes. #163 deleted that loop wholesale.
The `max_follow_up_rounds` constructor kwarg is accepted for back-compat and does nothing
(it raises a `DeprecationWarning`; see `__init__`). Since #163, PENDING tasks at
invocation end are dispositioned by `_sweep_unreachable_pending`, and STEER is the
user-driven path for exercising uncovered work.

**goldfive#202** re-introduced — in a *narrowly scoped* form — a post-invocation re-invoke:
the nudge-replay path (`_drain_nudges`). It fires **only** when the steerer explicitly
queued a nudge in `session.pending_nudges` in response to a tracked drift + plan revision
(e.g. `LOOPING_REASONING` → refine spawned `<task>_v2`), AND there is still live work, AND
it is capped at `_MAX_NUDGE_REPLAYS` (3). This is the carefully-narrowed successor to the
blanket #163 loop. Do not widen it back into "re-invoke on every PENDING at invocation
end" — that is the exact amplification #163 removed.

### DO / DON'T for the overlay

| DON'T | DO |
|---|---|
| Re-add a loop that re-invokes on every PENDING task at invocation end. | Leave reachable PENDING alone (`_sweep_unreachable_pending`) and let STEER / the next turn pick it up. |
| Drive the tree per-task in overlay (defeats the point). | Drive it once per iteration via `invoke_passthrough`; observe via `PlanReconciler`. |
| Uncap the nudge replay. | Keep `state.nudge_replays < self._MAX_NUDGE_REPLAYS`. |
| NOT_NEEDED-reap every PENDING at overlay exit (the #163 policy #208 replaced). | Cancel only *structurally unreachable* PENDING; leave reachable PENDING live for the next turn. |

---

## The overlay loop skeleton (`_run_overlay`)

Post-#489, `_run_overlay` is a thin dispatcher: it owns the `while True:` loop and threads a
single mutable `_OverlayTurnState` through named stage methods. The loop body reads (verbatim
from `goldfive/executors/sequential.py`, `_run_overlay`):

```python
state = _OverlayTurnState(current_user_input=user_input)
while True:
    self._clear_stale_supersede(session)
    kind, payload = await self._race_control(...)
    if kind == "cancelled":
        outcome = await self._handle_invoke_cancelled(...)
        if outcome is not None:
            return outcome
        continue
    if kind == "adapter_error":
        ...
        return await self._abort_overlay(..., reason=failure_reason)
    if kind == "steer":
        await self._restart_after_user_steer(...)
        continue
    if kind == "goldfive_steer":
        self._restart_after_goldfive_steer(...)
        continue
    if kind == "goldfive_pause":
        outcome = await self._handle_goldfive_pause(...)
        if outcome is not None:
            return outcome
        continue
    # kind == "result"
    if self._drain_nudges(...):
        continue
    break

await self._sweep_unreachable_pending(plan=plan, session=session, steerer=steerer)
reason = self._classify_fatal_failure(plan=plan, session=session, steerer=steerer)
if reason is not None:
    return await self._abort_overlay(..., reason=reason)
unmet = evaluate_goal_predicates(session)
if unmet is not None:
    return await self._abort_overlay(..., reason=unmet)
await _drain_steerer_at_run_boundary(steerer, session)
await emit(sinks, run_completed_event(...))
return ExecutionOutcome(success=True, session=session)
```

`_OverlayTurnState` (`@dataclasses.dataclass`, `goldfive/executors/sequential.py`) is the
per-turn scratchpad:

```python
@dataclasses.dataclass
class _OverlayTurnState:
    current_user_input: str
    next_reentry_kind: ReentryKind | None = None
    nudge_replays: int = 0
```

- `current_user_input` — what the next `invoke_passthrough` iteration is fed. Rewritten by
  the steer/goldfive-steer/nudge branches to a goldfive-composed framed message.
- `next_reentry_kind` — a **one-shot** re-entry pin (harmonograf#234). ContextVars snapshot
  at `asyncio.create_task` time, so the `reentry()` context manager must wrap the call site
  in `_race_control`, not the create-task inside `_invoke_passthrough_with_control`. Cleared
  after consumption. Values come from `ReentryKind` (`goldfive/adapters/adk_reentry.py`):
  `STEER_REPLAY`, `GOLDFIVE_STEER_REPLAY`, `NUDGE_REPLAY` (the default `None` means "no
  executor-level pin"; `ADKAdapter.invoke_passthrough` pins `OVERLAY_REPLAY` itself).
- `nudge_replays` — the `_MAX_NUDGE_REPLAYS` counter.

The `kind` strings returned into the loop map to the branches: `"result"`, `"adapter_error"`,
`"cancelled"`, `"steer"`, `"goldfive_steer"`, `"goldfive_pause"`. These come out of
`_invoke_passthrough_with_control` (below), *not* directly from `dispatch_control`.

The `planner` parameter to `_run_overlay` is currently unused (`# noqa: ARG002 -- reserved
for future refine hooks`) — the overlay's refines all run inside the steerer's reporting-tool
handlers, not in the executor. Do not wire `planner.refine` directly into the overlay loop
without reading 10-planning-and-revision.md first.

---

## Overlay stage methods, one by one

Each subsection: **trigger** (what causes the loop to enter this method), **state
read/written**, **events emitted**, **issue lineage**. All live in
`goldfive/executors/sequential.py` on `SequentialExecutor`.

### `_clear_stale_supersede(session)` — top of every iteration

**Trigger.** Runs unconditionally at the top of the `while True:` loop, before
`_race_control`.

**What it does.** Wipes two supersede markers so a stale flag from a prior iteration cannot
misclassify a genuine external cancel on the *next* iteration as a goldfive-internal
supersede:

```python
session._supersede_pending = False
StateStore.for_session(session).clear_all_supersede_pending()
```

The flag is set by `DefaultSteerer._cancel_inflight_for_revision` immediately before the
cancel that `_handle_invoke_cancelled` consumes. Branches that don't visit the cancelled
branch (e.g. STEER, which calls `_cancel_invoke_task` directly) can still trigger the
flag-set as a side effect of `steerer.drift.observe → install_user_steer →
_cancel_inflight_for_revision`; clearing here prevents that leak. Issue #405 LOW #7 also
wipes the per-invocation `StateStore` registry (each `invocation_id` is unique, so it can't
be clobbered cross-invocation, but an unconsumed entry would confuse the cancelled branch).
Both writes are wrapped in bare `try/except` — this is best-effort hygiene, not a hard
dependency.

**Events.** None.

### `_race_control(...)` — the one passthrough invocation

**Trigger.** Every iteration, after `_clear_stale_supersede`.

**What it does.** Wraps `_invoke_passthrough_with_control` in the re-entry pin. If
`state.next_reentry_kind is None`, calls it directly; otherwise wraps the call in
`with reentry(state.next_reentry_kind):` and clears the kind afterward (one-shot). The
`reentry()` wrapper *must* be here (not deeper) because ContextVars snapshot at
`create_task` time inside `_invoke_passthrough_with_control`.

**State.** Reads `state.next_reentry_kind` and `state.current_user_input`; writes
`state.next_reentry_kind = None` after a pinned call.

**Returns.** The `(kind, payload)` tuple from `_invoke_passthrough_with_control`.

### `_handle_invoke_cancelled(...)` — cancelled: supersede vs external abort

**Trigger.** `kind == "cancelled"`.

**The problem it solves (v22 validation Bug A).** A goldfive-internal *supersede-cancel* —
the steerer's `_cancel_inflight_for_revision` cancelling the in-flight invocation so a newly
installed revised plan can be exercised — used to fall through and emit `RunAborted`,
terminating the user's turn even though the supersede was meant to **continue** the turn
against the revised plan. The fix mirrors the STEER branch: if the cancel was internal,
reset the reconciler and restart the loop.

**Discrimination (invariant 5 — stable keys).** Union-of-signals read (#405 LOW #7):

```python
supersede_bool = bool(getattr(session, "_supersede_pending", False))
supersede_registry = StateStore.for_session(session).has_any_supersede_pending()
supersede_pending = supersede_bool or supersede_registry
```

The bool is the primary signal (all `tests/test_executor_supersede_cancel_nonfatal.py`
exercise it); the per-`invocation_id` registry is the defensive backstop that survives
concurrent overlay iterations. The dual read is transitional — issue #430 tracks retiring
the bool.

**Branches:**

- `supersede_pending and not self._fail_fast_on_invoke_cancel` → **restart, non-fatal.**
  Clear both markers, `reconciler.reset_for_new_plan(session.plan)`, return `None` (loop
  `continue`s against the swapped plan). The user's input is unchanged (supersede swaps the
  plan, not the request); no reentry kind is pinned (this is autonomous-drift-driven, not a
  STEER).
- **External cancel OR `_fail_fast_on_invoke_cancel=True`** → **abort.** External cancels
  (USER_CANCEL via the channel, `asyncio.CancelledError` from the caller) never set the
  supersede marker and **always** abort regardless of the flag. Clear any stale supersede
  flag, clear `session._last_cancel_reason_prefix`, and `return
  self._abort_overlay(reason=failure_reason)`.

**The flag.** `fail_fast_on_invoke_cancel` (constructor kwarg; `None` → consult
`GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL=1`) preserves the pre-fix abort behaviour for
CI/regression/debugging. Explicit `True`/`False` from the kwarg wins over the env. It
governs **only** the goldfive-internal supersede branch. See `__init__` and the class
docstring; mirrors PR #332's `fail_fast_on_revision_rejection` principle.

**Returns.** `None` (restart loop) or an aborted `ExecutionOutcome`.

**Events.** None directly; `_abort_overlay` emits `RunAborted` on the abort path.

### `_restart_after_user_steer(...)` — compose the next iteration after a user STEER

**Trigger.** `kind == "steer"`.

**What it does**, in order:

1. `await apply_steer(payload, steerer=steerer, session=session)` — feeds the STEER
   `ControlMessage` through `steerer.drift.observe` so `USER_STEER` drift fires →
   cascade-cancel + `planner.refine` runs → `session.plan` is replaced. Without this, the
   overlay would mark the *pre-steer* plan's tasks NOT_NEEDED and miss the steer entirely
   (the goldfive#149 regression, preserved here post-#163).
2. `state.current_user_input = self._compose_steer_restart_message(payload,
   fallback=state.current_user_input)` — wraps the steer body in a `[USER STEERING CONTROL
   — supersedes prior task context]` header (goldfive#152) so the LLM sees an operator
   override, not a fresh user turn.
3. `reconciler.reset_for_new_plan(session.plan)` — stale `task_id → agent` claims from the
   pre-steer plan must not leak into the replay.
4. `state.next_reentry_kind = ReentryKind.STEER_REPLAY` — plugins observing the inner
   runner's user-message hook see `STEER_REPLAY`, not `USER_TURN`, and suppress duplicate
   emission (harmonograf#234).

**State.** Writes `state.current_user_input`, `state.next_reentry_kind`; mutates
`session.plan` transitively via the steerer.

**Events.** Whatever `steerer.drift.observe` emits (a `USER_STEER` `DriftDetected`, a
`PlanRevised` from the refine).

### `_restart_after_goldfive_steer(...)` — compose the next iteration after a GOLDFIVE_STEER

**Trigger.** `kind == "goldfive_steer"`. This is a **synchronous** method (no `await`) — it
only composes the next input; the steerer already did the plan work before dispatching.

**What it does.** Reads the control message payload:

- `body` — the corrective body.
- `superseded_task_ids` — task ids the LLM should not resume.
- `replacement_task_ids` — task ids that supersede them.

Then:

1. `state.current_user_input = self._compose_steer_restart_message(control_msg,
   fallback=body_text or state.current_user_input, source="goldfive",
   superseded_task_ids=…, replacement_task_ids=…)` — `[GOLDFIVE STEERING CONTROL …]`
   framing with an explicit superseded/replacement task-id block.
2. `reconciler.reset_for_new_plan(session.plan)`.
3. `state.next_reentry_kind = ReentryKind.GOLDFIVE_STEER_REPLAY`.

**Crucial difference from the user-STEER branch:** there is **no** `apply_steer` /
`steerer.drift.observe` call. The steerer *originated* this message and already swapped
`session.plan`; observing again would loop. This is Phase 2 of the path-duality fix — the
steerer mints `GOLDFIVE_STEER` on the same channel as user STEER so the executor has a
single cancel-and-restart junction, but the provenance is kept straight via the distinct
`ControlOutcome.goldfive_steer_message` field.

**Events.** None directly.

### `_handle_goldfive_pause(...)` — GOLDFIVE_PAUSE_ESCALATE in the overlay

**Trigger.** `kind == "goldfive_pause"`.

**Background.** This replaces the deleted `session.paused_for_human_intervention` flag-set
(Phase 2 of the path-duality fix). The steerer's intervention-ladder Level-4 pause
dispatches a `GOLDFIVE_PAUSE_ESCALATE` message; `_invoke_passthrough_with_control` cancelled
the in-flight invoke and returned this kind.

**observation_only carve-out (invariant 2, goldfive#264, defense-in-depth).** The *primary*
gate is at `DefaultSteerer._dispatch_goldfive_pause_control`: under `observation_only=True`
the channel send is skipped and this branch is never reached via the supported steerer. But
a custom steerer subclass or future path that bypasses the dispatcher would otherwise drive
an overlay-terminating pause. So:

```python
if not steering_is_active(steerer):
    log.info("... observation_only=True — would have paused ... continuing overlay ...")
    return None
```

`return None` continues the overlay loop without ending the turn or cancelling the upstream
invoke. The originating `HUMAN_INTERVENTION_REQUIRED` drift on the sink stream remains the
durable signal.

**Active path.** Log, `await _drain_steerer_at_run_boundary(steerer, session)`, and return
`ExecutionOutcome(success=True, reason="goldfive_pause_escalate: …")`. Ending the overlay
turn here hands control back to the `Runner`; the *next* `run` cycle's
`_apply_pre_task_controls` blocks on the channel for an operator `RESUME`/`CANCEL`/`STEER`.

**Returns.** `None` (continue) or the terminal outcome.

### `_drain_nudges(...)` — the scoped nudge-replay path (#202)

**Trigger.** `kind == "result"` (invocation ended normally), evaluated before the loop
`break`.

**Gate.** All three must hold to replay:

```python
pending = list(session.pending_nudges)
if (
    pending
    and state.nudge_replays < self._MAX_NUDGE_REPLAYS
    and _has_live_pending_or_running(session.plan or plan)
):
```

- Nudges were queued by the steerer during this invocation.
- The replay budget isn't exhausted.
- There is still live (`PENDING`/`RUNNING`) work — never replay against a terminated plan.

**observation_only carve-out (invariant 2, defense-in-depth).** Inside the gate, if
`not steering_is_active(steerer)`: **discard** the queue
(`session.pending_nudges.clear()`, `pending_nudges_revision_installed = False`) and
`return False` (end the turn). The queue must never be injected later.

**Active replay.** Read `plan_revised = session.pending_nudges_revision_installed`, clear
the queue, `state.nudge_replays += 1`, compose the framed body via
`_compose_nudge_replay_message(pending, plan_revised=plan_revised)`,
`reconciler.reset_for_new_plan(session.plan)`, pin
`state.next_reentry_kind = ReentryKind.NUDGE_REPLAY`, `return True` (loop re-invokes).

`plan_revised` selects the framing (`_compose_nudge_replay_message`): only when the steerer
recorded that `_apply_revision` actually installed a revision does the header claim a plan
revision (`[GOLDFIVE PLAN REVISION — replace superseded task(s)]`); otherwise a
course-correction header that asserts nothing about the plan
(`[GOLDFIVE COURSE CORRECTION]`).

**Returns.** `True` (re-invoke) or `False` (fall through to the sweep).

### `_sweep_unreachable_pending(...)` — end-of-overlay PENDING disposition (#163 → #208)

**Trigger.** Once, after the loop `break`, before the terminal gates.

**Policy (structural reachability, goldfive#208 replacing #163's blanket NOT_NEEDED reap).**
The tree finished its natural flow; decide what to do with tasks it never exercised:

- **Reachable PENDING** — every predecessor either `COMPLETED` or itself a still-live
  `PENDING`/`RUNNING` task that can reach `COMPLETED`. **Leave PENDING.** The Conversation
  carry-forward (`stash_plan` / `prior_plan_for`, 03-runner-and-conversation.md) seeds the
  next turn with this task still live.
- **Unreachable PENDING** — at least one transitive predecessor reached a
  terminal-non-COMPLETED status (`CANCELLED`/`FAILED`/`NOT_NEEDED`) with no live replacement.
  `steerer.transition(tid, CANCELLED, cancel_reason="run_aborted:orphaned by plan revision
  failure")` — the same reason the legacy reachability audit uses.

The computation is `_unreachable_pending_task_ids(live_plan)` (forward-propagation from
seed-broken tasks to fixed point, AND-join semantics; see the helper below). The #163
blanket reap silently destroyed cross-turn user intent (a user-pivoted plan whose downstream
stages hadn't dispatched yet); #208's structural policy is why you must never reintroduce
"NOT_NEEDED every PENDING at overlay exit".

**Events.** `TaskCancelled` per unreachable task (via the steerer).

### `_classify_fatal_failure(...)` — the overlay's fail_fast gate

**Trigger.** After the sweep.

**Returns** the abort reason string, or `None`. `fatally_failed =
_fatally_failed_task_ids(session.plan or plan)`. If non-empty and `self.fail_fast`:

- **observation_only carve-out** (invariant 2, goldfive#260): if `not
  steering_is_active(steerer)`, log and `return None` — the replacement-producing refine is
  dry-run under observation_only, so the executor has no replacement to install; the ADK
  coordinator's autonomous flow may still recover.
- Else return `"one or more tasks failed without a live replacement: …"`.

A `FAILED` task with a live replacement (`_has_live_replacement`) is filtered out of
`_fatally_failed_task_ids` and is never fatal (goldfive#202).

### `_abort_overlay(...)` — the shared overlay abort tail

**Trigger.** Called by every overlay failure path (`adapter_error`, external cancel,
fatal-failure, unmet goal predicates).

**What it does.** `await _drain_steerer_at_run_boundary(steerer, session)`, emit
`run_aborted_event(reason=reason)`, return `ExecutionOutcome(success=False, reason=reason)`.
This is the overlay analogue of the legacy loop's per-gate drain+emit — extracted in #489 so
the six failure paths share one tail. If you add a new overlay failure path, route it
through `_abort_overlay`; do not hand-roll the drain + `RunAborted`.

---

## The passthrough control race (`_invoke_passthrough_with_control`)

This is where the overlay's `(kind, payload)` tuples are actually minted. It mirrors the
legacy `_invoke_with_control` but drives `invoke_passthrough` and understands the
goldfive-internal control kinds.

**No-channel fast path.** When `control is None`, just `await invoke_task`; map
`CancelledError` → re-raise (external cancellation propagates), any other exception →
`("adapter_error", exc)`, else `("result", result)`.

**With a channel**, it loops: create a `control.receive()` task, `asyncio.wait({invoke_task,
recv_task}, FIRST_COMPLETED)`.

- **Invoke finished first** → cancel the recv task, return `("result", …)` /
  `("adapter_error", …)` / `("cancelled", "invoke cancelled")` (if the invoke was itself
  cancelled).
- **A control message arrived** (`msg = recv_task.result()`; `None` means the channel
  closed — fall back to awaiting the invoke) → `dispatch_control(msg, …)`, ack, then act on
  the `ControlOutcome`:

| `ControlOutcome` field set | Action in `_invoke_passthrough_with_control` |
|---|---|
| `cancel_run` | `_cancel_invoke_task(invoke_task)`; stash `cancel_reason_prefix` on `session._last_cancel_reason_prefix`; return `("cancelled", …)`. |
| `steer_message` | `_cancel_invoke_task`; return `("steer", msg)`. **Overlay does not loop on steer here** — it returns so the loop's `_restart_after_user_steer` runs. |
| `goldfive_steer_message` | `_cancel_invoke_task`; return `("goldfive_steer", msg)`. |
| `goldfive_pause_message` | **observation_only carve-out**: if `not steering_is_active(steerer)`, log, drop the message, `continue` (keep waiting) — do NOT cancel the invoke. Else `_cancel_invoke_task`; return `("goldfive_pause", msg)`. |
| none of the above | Non-cancelling control (PAUSE/RESUME/REWIND_TO/STATUS_QUERY/INTERCEPT_TRANSFER/APPROVE/REJECT) — already applied inside `dispatch_control`; keep waiting. |

Note the third observation_only carve-out lives here (in addition to `_handle_goldfive_pause`
and `_drain_nudges`) — this one prevents even *cancelling the in-flight invoke* under
observation_only, so a bypass path cannot interrupt the tree.

`_cancel_invoke_task` (static) cancels with a **5-second grace window**: it polls
`invoke_task.done()` for 5s; if the adapter ignores `task.cancel()`, it logs a warning and
abandons the orphaned task for the event loop to reap. Adapters that ignore cancellation
must not wedge the run (invariant 1).

### `_compose_steer_restart_message` / `_compose_nudge_replay_message`

Both are static methods that wrap a body in a goldfive-authored header so the LLM sees an
override, not a fresh user turn. `_compose_steer_restart_message(msg, *, fallback,
source="user", superseded_task_ids=None, replacement_task_ids=None)`:

- Reads `payload["note"]` (or `payload["body"]` as a courtesy) — the harmonograf server maps
  `PostAnnotation(body=…)` onto `ControlEvent.steer.note`, and the control bridge rehydrates
  that into `payload["note"]`.
- `source="user"` → `[USER STEERING CONTROL — supersedes prior task context]`.
- `source="goldfive"` → `[GOLDFIVE STEERING CONTROL — supersedes prior task context]`, with
  an appended task-id block when `superseded_task_ids` / `replacement_task_ids` are provided.
- Falls back to `fallback` when the extracted body is empty, then wraps *that* in the same
  header so the re-invocation always shows override semantics.

Keep the header prefixes exact — prompt templates and tests match on
`"[USER STEERING CONTROL"` / `"[GOLDFIVE STEERING CONTROL"`.

---

## Pre-task / pre-stage control draining and the pause loop

Both executors have a "before the next unit of work" control gate that (a) drains queued
messages non-blockingly, then (b) if a PAUSE is pending, **blocks** on the channel until it
unwinds. They are near-identical; the differences are called out in the parity table.

- Sequential: `_apply_pre_task_controls(control, session, steerer, sinks) -> (cancel_run,
  steer_message)`.
- Parallel: `_apply_pre_stage_controls(...)` — same signature and shape.

### The drain phase

`outcomes = await drain_controls(control, …)` collects one `ControlOutcome` per queued
message (see `_control.py`). The executor folds them:

- First `cancel_run` wins → stash `cancel_reason_prefix`, `raise _ControlCancelled(reason)`.
- Any `steer_message` is remembered (returned so the caller runs `apply_steer`).
- `request_pause` sets `paused = True`; if the outcome also carries `goldfive_pause_message`,
  remember it (it may carry a `deadline_s`).
- `request_resume` clears `paused` and the pause message.

### The blocking pause loop (with the #482 deadline)

If `paused`, the executor blocks:

```python
deadline_s = pause_deadline_s(pause_msg)
deadline_at = time.monotonic() + deadline_s if deadline_s is not None else None
while paused:
    if deadline_at is None:
        msg = await control.receive()             # unbounded — operator PAUSE
    else:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            await abort_expired_pause(...)          # NoReturn: raises _ControlCancelled
        try:
            msg = await asyncio.wait_for(control.receive(), timeout=remaining)
        except TimeoutError:
            await abort_expired_pause(...)
    if msg is None:                                 # channel closed → treat as resume
        paused = False
        break
    outcome = await dispatch_control(msg, ...)
    await control.ack(outcome.ack)
    # unwind conditions:
    if outcome.cancel_run: raise _ControlCancelled(...)
    if outcome.request_resume: paused = False
    if outcome.steer_message is not None: steer_msg = ...; paused = False
    if outcome.goldfive_steer_message is not None: steer_msg = ...; paused = False   # sequential only (#404)
    if outcome.goldfive_pause_message is not None: adopt tighter deadline (only)
    if outcome.rewind_task_id: paused = False
```

Key behaviours:

- **Operator PAUSE blocks unbounded** (no `deadline_s`). Only a `GOLDFIVE_PAUSE_ESCALATE`
  whose payload carries `deadline_s` bounds the wait.
- **A repeat `GOLDFIVE_PAUSE_ESCALATE` while already paused does NOT re-enter the pause** —
  it only *adopts a tighter deadline* (`if new_deadline_at < deadline_at`). This is how the
  ladder's TERMINATE row (#482) lands on an unbounded Level-4 pause: the repeat escalation's
  deadline converts the wait into a bounded one. See the `goldfive#404` comment.
- **`GOLDFIVE_STEER` while paused unwinds the pause — sequential only.** The sequential
  `_apply_pre_task_controls` treats `outcome.goldfive_steer_message` as an unblock
  (goldfive#404) so a goldfive-authored drift produced while paused isn't silently dropped
  (which would wedge the run). **The parallel `_apply_pre_stage_controls` does NOT have this
  line** — see the parity table.

`abort_expired_pause` is `NoReturn`: it transitions every non-terminal task to `CANCELLED`
with `run_aborted:pause_escalate_deadline:<drift_kind>` then raises `_ControlCancelled`,
which the executor's outer `try/except _ControlCancelled` catches and turns into `RunAborted`
carrying the escalation lineage (`drift_kind` + `ladder_level` from the pause payload).

**Mid-task PAUSE (both executors) lets the current unit finish.** PAUSE only blocks *between*
tasks/stages. `_invoke_with_control` / `_run_stage` treat a mid-invoke PAUSE as a
non-cancelling control (it is applied but the task/stage runs to completion). This matches
spec; do not change it to a mid-task interrupt without a very good reason.

**No channel → the ladder pause has nothing to wait on.** When `control is None`, both
pre-* helpers `return False, None` immediately. The steerer's `GOLDFIVE_PAUSE_ESCALATE`
dispatch is best-effort and dropped at the source when no channel is attached; the
`HUMAN_INTERVENTION_REQUIRED` drift on the sink stream remains the durable signal.

---

## The control channel (`goldfive/control.py`)

`ControlChannel` is the bidirectional async primitive between a `Runner` and an external
controller (harmonograf UI, CLI, tests). It is intentionally dependency-light (asyncio
only) so adapters and bridges can import it without dragging in protobuf/grpc.

Two `asyncio.Queue`s:

- `_inbox` — external → runner. `send(msg)` enqueues; `receive(timeout=None)` dequeues.
- `_outbox` — runner → external. `ack(ack)` enqueues; `acks()` async-iterates.

### Semantics you must respect

| Method | Contract |
|---|---|
| `await send(msg)` | External caller pushes a `ControlMessage`. Never blocks meaningfully (unbounded queue). |
| `await receive(timeout=None)` | Runner polls. Returns the next `ControlMessage`, or **`None`** on timeout **or when the channel is closed**. A `timeout` uses `asyncio.wait_for` and returns `None` on `TimeoutError`. |
| `await ack(ack)` | Runner publishes a `ControlAck`. |
| `async for ack in acks()` | External bridge drains acks until `close()`. |
| `close()` | Idempotent. Sets `_closed`; after this `receive()` returns `None` immediately, and any consumer blocked on `acks()` is woken via a private `_CLOSE_SENTINEL` so it exits cleanly (no leaked task). |

**The dropped-message caveat.** `receive()` returns `None` on *both* timeout and close, and
the executors treat a `None` from `receive()` inside the pause loop as "channel closed →
resume so we don't wedge". If you add a `timeout` to a `receive()` in an executor pause path,
a benign timeout would look identical to a close and would unwedge the pause — which is
usually wrong. The pause loops use `asyncio.wait_for(control.receive(), timeout=remaining)`
precisely so a timeout raises `TimeoutError` (→ `abort_expired_pause`) rather than returning
`None`. Do not "simplify" that to `receive(timeout=remaining)`; you would lose the ability to
distinguish deadline-expiry from channel-close.

**`drain_controls` reaches into `channel._inbox` directly** (`inbox.get_nowait()` in a
`while not inbox.empty()` loop). This is a deliberate non-blocking drain of everything
*currently* queued. It is coupled to `ControlChannel`'s internals — if you rename `_inbox`,
update `drain_controls`.

### `ControlKind` — the message vocabulary

`ControlKind` is a `StrEnum`, so members compare equal to their raw strings. **It must stay
in lockstep with `proto/goldfive/v1/control.proto`** — a drift guard in
`tests/test_control_proto.py` enforces both directions. Members:

| Kind | Payload | Origin |
|---|---|---|
| `PAUSE` / `RESUME` | none | external operator |
| `CANCEL` | `{reason?, annotation_id?}` | external operator |
| `STEER` | `{note?, suggested_action?}` (`body` also accepted) | external operator |
| `REWIND_TO` | `{task_id}` | external operator |
| `APPROVE` / `REJECT` | `{target_id, detail?}` | external operator |
| `STATUS_QUERY` | none | external operator (read-only) |
| `INTERCEPT_TRANSFER` | `{enabled: bool}` | external operator |
| `INJECT_MESSAGE` | `{role, text}` | external operator |
| `GOLDFIVE_STEER` | `{drift_kind, drift_id, body, superseded_task_ids[], replacement_task_ids[]}` | **goldfive-internal** (steerer mints) |
| `GOLDFIVE_PAUSE_ESCALATE` | `{reason, drift_id, deadline_s?, drift_kind?, ladder_level?}` | **goldfive-internal** (steerer mints) |

The two `GOLDFIVE_*` kinds are Phase 2 of the path-duality fix: the steerer routes
goldfive-authored drift through the *same* cancel-and-restart junction as user STEER by
minting these on the channel. **External bridges must not originate them** — they encode a
goldfive-side decision, not an operator directive.

`ControlMessage.id` defaults to a `seeded_uuid4().hex` (deterministic under the seeded
runtime — see `goldfive/runtime.py`). `AckResult` is `SUCCESS` / `FAILURE` / `UNSUPPORTED`.
`INJECT_MESSAGE` is defined in the enum but has **no `dispatch_control` branch** — it falls
through to the `UNSUPPORTED` ack today. Do not assume every enum member is wired.

---

## Control-message dispatch (`goldfive/executors/_control.py`)

`dispatch_control(msg, *, session, steerer, sinks) -> ControlOutcome` is the single
interpreter both executors share. It matches on `_kind_value(msg)` (uppercased string, so
enum members and raw strings both match) and returns a `ControlOutcome` with a pre-built
`ack`. **The caller publishes the ack** (`channel.ack(outcome.ack)`); `dispatch_control`
does not touch the channel except transitively.

### `ControlOutcome` fields

```python
@dataclasses.dataclass
class ControlOutcome:
    ack: ControlAck
    cancel_run: bool = False
    steer_message: ControlMessage | None = None
    goldfive_steer_message: ControlMessage | None = None
    goldfive_pause_message: ControlMessage | None = None
    request_pause: bool = False
    request_resume: bool = False
    rewind_task_id: str = ""
    cancel_reason: str = ""
    cancel_reason_prefix: str = ""
```

The three `*_message` fields keep provenance straight: `steer_message` (operator),
`goldfive_steer_message` (goldfive drift), `goldfive_pause_message` (goldfive Level-4/5). A
weak model will be tempted to collapse `goldfive_steer_message` into `steer_message` — **do
not**: the overlay branches compose different framing (`[USER STEERING CONTROL]` vs
`[GOLDFIVE STEERING CONTROL]`) and the goldfive branch must **not** call `apply_steer`
(re-observing would loop).

### Per-kind behaviour

| Kind | `ControlOutcome` produced | Side effects / notes |
|---|---|---|
| `CANCEL` | `cancel_run=True`, `cancel_reason`, `cancel_reason_prefix` | Prefix `user_cancel:<annotation_id>` (falls back to `user_cancel:<msg.id>`) for harmonograf trajectory provenance (#205/#176). No state mutation here. |
| `PAUSE` | `request_pause=True` | — |
| `RESUME` | `request_resume=True` | Also unwinds a goldfive-initiated pause — Phase 2 means both drive the same pause state. |
| `STEER` | `steer_message=msg` | The executor feeds it to `steerer.drift.observe`; a STEER also acts as an implicit RESUME. |
| `GOLDFIVE_STEER` | `goldfive_steer_message=msg` | Steerer already swapped `session.plan`; message carries the corrective body. |
| `GOLDFIVE_PAUSE_ESCALATE` | `goldfive_pause_message=msg`, `request_pause=True` | Replaces the deleted `session.paused_for_human_intervention` flag. |
| `REWIND_TO` | `rewind_task_id=target` (or FAILURE ack) | `_rewind_plan` resets target + downstream to PENDING (frozen-plan swap under `channel_processor_active`); emits one `TaskTransitioned(source="control_rewind")` per affected task via the duck-typed `steerer.tasks._emit_task_transitioned` (F10/#251 R4). Also pops `completed_results`/`completed_outputs`/`task_progress` for reset tasks. |
| `STATUS_QUERY` | SUCCESS ack with `detail=<snapshot>` | **Read-only. Emits NO events.** `build_status_snapshot` returns a compact string via the ack detail. A prior version synthesised a `DriftDetected` per poll → 33k bogus `kind=0` drift events in a 5-min run; that is the bug this fixed. |
| `INTERCEPT_TRANSFER` | SUCCESS ack | Sets `session._intercept_transfer` (adapters that honour it refuse transfers). |
| `APPROVE` / `REJECT` | SUCCESS/FAILURE ack | `_resolve_approval` sets the `session.pending_approvals[target_id]` waiter and emits `ApprovalGranted`/`ApprovalRejected` *before* releasing the waiter (ordering: resolution event visible → waiter releases → tool returns). FAILURE when no waiter is registered (13-reporting-tools-and-approval.md). |
| anything else (incl. `INJECT_MESSAGE`) | UNSUPPORTED ack | — |

### `drain_controls`

Non-blocking drain of every currently-queued message: `while not inbox.empty():
inbox.get_nowait() → dispatch_control → channel.ack`. Returns the list of outcomes for the
caller to fold. `None` channel or empty inbox → `[]`. The ack is published *before* return.

### `build_status_snapshot`

Compact string for the STATUS_QUERY ack detail:
`status_query control_id=<id> current_task=<id> completed=<n>/<total>
pending=<comma-ids>`. Read-only; touches no sink. This is the *only* way a status poll
surfaces plan state — it must never grow a `DriftDetected` emission.

---

## The pause deadline and `abort_expired_pause` (#482)

Before #482 the escalation ladder's terminus was unbounded: a Level-4 pause with no operator
watching blocked forever. #482 added `pause_escalate_deadline_s` (`SteeringConfig`) and a
real TERMINATE with a 600s built-in deadline fallback.

`pause_deadline_s(msg) -> float | None`:

```python
payload = getattr(msg, "payload", None) or {}
try:
    value = float(payload.get("deadline_s"))
except (TypeError, ValueError):
    return None
return value if value > 0 else None
```

`None` when absent / non-numeric / non-positive → the pause blocks unbounded (historical
behaviour). Operator PAUSE never carries `deadline_s`.

`abort_expired_pause(*, session, steerer, pause_msg, deadline_s) -> NoReturn`:

1. Reads `drift_kind` / `ladder_level` from the pause payload.
2. Builds `reason = "pause escalation deadline expired after {deadline_s:g}s with no
   operator action (drift_kind=…, ladder_level=…)"`.
3. Transitions every non-terminal task to `CANCELLED` with
   `cancel_reason="run_aborted:pause_escalate_deadline:<drift_kind>"` (mirrors the
   operator-CANCEL cascade; no-ops on already-terminal tasks).
4. `raise _ControlCancelled(reason)`.

The `_ControlCancelled` propagates out of the pause loop; the executor's outer handler emits
`RunAborted(reason)` so the escalation lineage is durable on the sink stream. This is why
`abort_expired_pause` is imported by **both** executors and by the `Runner`
(03-runner-and-conversation.md lists it under "Files covered (partial)").

**When you add a new pause path, it MUST carry a deadline** (or a deliberate `None` with a
comment saying why unbounded is correct). An unbounded pause that no operator is watching is
a wedged run. This is one of the three named Common Mistakes below.

---

## `_ControlCancelled` flow

`_ControlCancelled(BaseException)` (`_control.py`) is the internal short-circuit for CANCEL
and deadline-expiry. It subclasses **`BaseException`, not `Exception`**, on purpose: stray
`except Exception` handlers inside adapter/agent code must not swallow it. Carries a
`.detail` string.

Flow:

1. A `CANCEL` arrives during pre-task/pre-stage drain → `_apply_pre_*_controls` raises it.
2. Or `abort_expired_pause` raises it on deadline expiry.
3. The executor's `run` catches it:
   - Sequential legacy: `except _ControlCancelled as cancelled: failure_reason =
     cancelled.detail; run_failed = True; break` → the terminal cascade emits `RunAborted`.
   - Parallel: `except _ControlCancelled as cancelled: abort_reason = cancelled.detail;
     break` → the post-loop `if abort_reason:` emits `RunAborted`.
   - Overlay: a CANCEL *mid-invoke* comes back as `("cancelled", …)` and goes through
     `_handle_invoke_cancelled` (external cancel → `_abort_overlay`), not `_ControlCancelled`.
     `_ControlCancelled` in the overlay is reached only via the pre-task pause loop on a
     *subsequent* `run` cycle.

Never catch `_ControlCancelled` with a broad handler that would suppress it. If you write
new adapter code that wraps `invoke`, use `except Exception` (not `except BaseException`) or
explicitly re-raise `_ControlCancelled` and `asyncio.CancelledError`.

---

## Shared helpers (`goldfive/executors/_shared.py`)

The non-dispatch glue both executors need. #485 consolidated the terminal-status handling;
these helpers are the "cannot drift apart" surface between the two executors.

| Helper | Purpose |
|---|---|
| `CANCEL_REASON_USER_STEER = "user_steer"` | Plain-string mirror of `goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER`, duplicated so the provider-agnostic executors don't import the optional ADK module. **Keep in sync** (goldfive#139). |
| `tag_adapter_cancel_user_steer(adapter, session=None)` | Tags the adapter's *next* mid-invocation cancel with the USER_STEER reason so the adapter appends an LLM-actionable synthetic `function_response` instead of generic jargon. Routes through `adapter.set_next_cancel_reason(session, …)` when available (keyed by `session.id` so it can't bleed across concurrent sessions sharing one adapter, #294/#271); falls back to the bare `adapter._next_cancel_reason` attribute. |
| `mark_cancelled_if_live(*, task_id, steerer, session, cancel_reason="")` | Transitions a not-yet-terminal task to `CANCELLED` via `steerer.transition`. No-ops if the task is already terminal or the plan is `None`. Defaults `cancel_reason` to `user_cancel:cancelled_by_control` (#205). |
| `apply_steer(message, *, steerer, session)` | Feeds a STEER `ControlMessage` to `steerer.drift.observe(message, session)`. Swallows+warns on raise. |
| `emit_drift_event(*, session, sinks, drift)` | Builds a `DriftDetected` envelope with correct proto enum mapping (`DRIFT_KIND_<NAME>` / `DRIFT_SEVERITY_<NAME>`) and emits. The generic `drift_detected_event` helper silently fails the name lookup for StrEnum lowercase names and would leave `kind=0`; this helper is the fix. |
| `emit_pipeline_failure_drift(*, session, sinks, task_id, reason)` | Emits an INFO `CUSTOM` drift when the drift pipeline itself raised (`steerer.drift.observe`/`detect_drift`), so plumbing failures surface instead of being swallowed. INFO so it's record-only and doesn't trigger another refine. Filter on the `drift_pipeline_failed:` prefix (goldfive#134). |

`TERMINAL_TASK_STATUSES` itself lives in `goldfive/types.py` (canonicalised in #485):

```python
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.NOT_NEEDED}
)
```

`NOT_NEEDED` **is** terminal — that is why the parallel scheduler's `_pending` predicate
skips it and the sequential overlay's sweep treats a `NOT_NEEDED` predecessor as a broken
link. Both executors import the single set; there is no local re-definition anymore. If you
add a new terminal status, add it here and audit `_pick_next_task`,
`_unreachable_pending_task_ids`, and the parallel `_pending` filter.

---

## The parallel DAG executor (`goldfive/executors/parallel.py`)

`ParallelDAGExecutor` walks `Plan.topological_stages()`. Each stage is a set of tasks whose
predecessors already finished; within a stage it `asyncio.gather`s `adapter.invoke(task,
session)` under an optional `max_concurrency` semaphore. **Plan refinement never happens
mid-stage** — it runs between stages, which keeps the DAG walk deterministic (mirrors the
harmonograf walker it was ported from).

### Construction knobs

| kwarg | Meaning |
|---|---|
| `max_concurrency` (default 0) | Semaphore bound within a stage. `0` = unbounded fan-out; `1` = forced-sequential. |
| `drift_policy` (`"finish_stage"` default / `"cancel_stage"`) | On the first drift `>= WARNING`: `cancel_stage` cancels every other in-flight task in the stage; `finish_stage` lets siblings finish. Refinement runs before the next stage either way. |
| `max_task_invocations` (default `None`) | **Here this counts plan refinements, not task invocations** — deliberately unified name, different meaning from the sequential executor. Read the `__init__` note before touching it. |

### The `run` loop

1. Pin plan + bind steerer (as above).
2. `while True:`
   a. `_apply_pre_stage_controls(...)` — drain + pause (same shape as the sequential
      pre-task helper).
   b. Recompute `stages = current_plan.topological_stages()` from the *current* plan every
      iteration. **`topological_stages()` is purely structural — it does NOT filter by
      status** — so terminal tasks (a reconciler-stamped `NOT_NEEDED`, or tasks completed
      before a refinement) MUST be dropped here or they get re-invoked. That is the #485
      terminal-task filter:

      ```python
      def _pending(t: Task) -> bool:
          return t.id not in completed_stage_ids and t.status not in TERMINAL_TASK_STATUSES
      pending_stages = [stage for stage in stages if any(_pending(t) for t in stage)]
      if not pending_stages: break
      stage_tasks = [t for t in pending_stages[0] if _pending(t)]
      ```

      `completed_stage_ids` is a local set of "already run this call" ids; the status filter
      catches terminal-by-reporting-tool. Both are needed.
   c. `_run_stage(stage_tasks, …)` → `(stage_results, first_drift, control_outcome)`.
   d. Fold the stage: a mid-stage CANCEL cancels every task and breaks; otherwise
      auto-transition each task from the **live** plan status (not the stale captured `task`
      snapshot — #247: the auto-RUNNING swap produced a new `Plan`), recording
      `completed_outputs` (zicato#12) and `completed_results`.
   e. If a mid-stage STEER arrived, `apply_steer` and drop the stage-level drift (avoid
      double-refine).
   f. If `first_drift` is non-`None`, `_refine(...)`; install the revised plan under
      `channel_processor_active`, emit `PlanRevised`, and `continue`. A `None` refine bumps
      the per-`(kind, task)` failure counter and aborts at `REFINE_FAILURE_THRESHOLD` (2).
3. Post-loop: `abort_reason` → `RunAborted`; else goal predicates → `RunAborted`; else
   `RunCompleted`. An outer `except BaseException` folds the reason, emits `RunAborted`, and
   re-raises `CancelledError`.

### `_run_stage` internals worth knowing

- Each `run_one(task)` emits `TaskTransitioned(source="executor_dispatch")` on the
  framework auto-start (PENDING→RUNNING) via the duck-typed
  `steerer.tasks._emit_task_transitioned` (F10/#251 R4), swapping the plan under
  `channel_processor_active` first.
- Drift enters via `steerer.drift.observe(inv, session)` + `steerer.drift.detect_drift(inv,
  session)`; a pipeline raise → `emit_pipeline_failure_drift`, `drift = None`.
- The stage races a persistent `control.receive()` (`_ensure_recv_task`) against the
  in-flight tasks. A mid-stage CANCEL/STEER folds any already-done tasks, sets
  `stage_control_outcome`, `tag_adapter_cancel_user_steer` (on STEER), cancels the stage,
  and breaks. **Only `cancel_run` and `steer_message` interrupt the stage** — a mid-stage
  `GOLDFIVE_STEER` / `GOLDFIVE_PAUSE_ESCALATE` is a non-interrupting control here (parity
  gap, below).
- Outer `asyncio.CancelledError` reaps every inner task and the recv task, then re-raises.

### `_refine`

Asks `planner.refine(plan, drift, goals)` for a revision, validates it
(`refined.validate(for_revision=True, prior=plan)`), and on **every** failure mode (raises,
returns `None`, fails validation) emits a CRITICAL follow-up `DriftDetected`
(`_escalate_refine_failure_as_critical_drift`) so sinks see "refine failed" instead of
silently re-entering the same stage (goldfive#134). When a steerer with `observe_refine` is
bound, the refine is wrapped so it produces paired `refine_attempted` + (`refine_failed` |
`plan_revised`) events with a shared `attempt_id` (#263/#264); otherwise a legacy direct-call
path preserves only the CRITICAL mirror. A cancelled refine emits the mirror then re-raises
(CANCELLATION-CONTRACT.md §C2; `CancelledError` is a `BaseException` and bypasses `except
Exception`). The per-`(kind, task_id)` failure counter lives on `session.refine_outcomes`
(`RefineOutcome`, goldfive#215 iter-8 P2) — shared with the steerer, not a separate int
counter — and `REFINE_FAILURE_THRESHOLD = 2`.

---

## Sequential vs parallel: the honest parity table

This is the single most important table in the chapter. The two executors interpret the
*same* control vocabulary but do **not** honour it identically. An intervention that works
in one executor and one of its loops is a real, shipped asymmetry — do not assume symmetry.

| Capability | Sequential legacy loop | Sequential overlay loop | Parallel |
|---|---|---|---|
| Overlay / single-passthrough | n/a | **yes** | **no** — no `invoke_passthrough`, no `_run_overlay` |
| `user_input` re-invocation (STEER/nudge restart with composed message) | no | **yes** | **no** (`run` has no `user_input` param) |
| `CANCEL` mid-work | yes (`_invoke_with_control`) | yes (`_invoke_passthrough_with_control`) | yes (`_run_stage` recv race) |
| `STEER` mid-work | yes → `("steer")` → `apply_steer` + `continue` | yes → `_restart_after_user_steer` | yes → `stage_control_outcome.steer_message` → `apply_steer` |
| `GOLDFIVE_STEER` mid-work | **no** — `_invoke_with_control` only checks `cancel_run`/`steer_message` | **yes** → `_restart_after_goldfive_steer` | **no** — `_run_stage` only checks `cancel_run`/`steer_message` |
| `GOLDFIVE_PAUSE_ESCALATE` mid-work | **no** (mid-task) | yes → `_handle_goldfive_pause` | **no** (mid-stage) |
| `GOLDFIVE_PAUSE_ESCALATE` **pre-work** (via `request_pause`) | yes (`_apply_pre_task_controls`) | yes (next `run` cycle) | yes (`_apply_pre_stage_controls`) |
| Pause deadline (#482) honoured in pre-work loop | yes | yes | yes |
| `GOLDFIVE_STEER` unwinds an active pause (#404) | **yes** (`_apply_pre_task_controls`) | **yes** | **NO** — `_apply_pre_stage_controls` has no `goldfive_steer_message` unwind line |
| Deadline-tightening from repeat `GOLDFIVE_PAUSE_ESCALATE` while paused | yes | yes | yes |
| Refine driver | steerer reporting-tool handlers | steerer reporting-tool handlers | **inline in `run`** (`self._refine`) |
| Per-lineage retry cap | yes (`max_retries_per_task_lineage`) | no (overlay drives the tree, not per-task) | no |
| Reachability / orphan sweep | yes (post-loop audit) | yes (`_sweep_unreachable_pending`) | via `topological_stages` naturally blocking |

**Reading of the table for a change:** if you are asked to make goldfive honour a new
control mid-work everywhere, you must touch **five** consumption sites: `_invoke_with_control`
(legacy), `_invoke_passthrough_with_control` (overlay), `_run_stage` (parallel), and both
pre-* pause loops — plus `dispatch_control` and a `ControlOutcome` field. Miss one and you
ship the exact asymmetry the `GOLDFIVE_STEER` rows document.

---

## Message-shape reference

Quick lookup for the payload dicts the executors read. All are `dict[str, Any]` on
`ControlMessage.payload`.

| Kind | Keys read by goldfive | Read at |
|---|---|---|
| `CANCEL` | `reason`, `annotation_id` | `dispatch_control` |
| `STEER` | `note` (or `body`), `suggested_action`, `annotation_id` | `_compose_steer_restart_message`, `_steer_cancel_reason_prefix` |
| `REWIND_TO` | `task_id` | `dispatch_control` → `_rewind_plan` |
| `APPROVE`/`REJECT` | `target_id`, `detail` | `dispatch_control` → `_resolve_approval` |
| `INTERCEPT_TRANSFER` | `enabled` | `dispatch_control` |
| `GOLDFIVE_STEER` | `body`, `superseded_task_ids[]`, `replacement_task_ids[]`, `drift_kind` | `_restart_after_goldfive_steer` |
| `GOLDFIVE_PAUSE_ESCALATE` | `reason`, `deadline_s`, `drift_kind`, `ladder_level` | `_handle_goldfive_pause`, `pause_deadline_s`, `abort_expired_pause` |

Composed restart-message headers (exact prefixes, matched by tests / prompt templates):

- `[USER STEERING CONTROL — supersedes prior task context]` — user STEER.
- `[GOLDFIVE STEERING CONTROL — supersedes prior task context]` — GOLDFIVE_STEER.
- `[GOLDFIVE COURSE CORRECTION]` — nudge replay, no plan revision.
- `[GOLDFIVE PLAN REVISION — replace superseded task(s)]` — nudge replay with a revision
  installed.

---

## Plan helper reference (module-level, `sequential.py`)

These pure functions encode the executor's scheduling and reachability logic. They are used
by both the legacy loop and the overlay sweep.

| Function | Contract |
|---|---|
| `_pick_next_task(plan) -> Task \| None` | First `PENDING` task (in `topological_stages()` order) whose every predecessor is `COMPLETED`. Status authority is `Task.status`. |
| `_lineage_root(task_id) -> str` | Strips a chain of `retry_` / `retry<N>_` prefixes (bounded 16 iterations) so `t0`, `retry_t0`, `retry2_retry_t0` collapse to `t0`. The stable key for the per-lineage cap (invariant 5). Uses `_RETRY_PREFIX_RE = re.compile(r"^retry(?:\d+)?_")` — this is **structural id matching, not NL classification**, so it does not violate the no-regex-heuristics invariant (#166/#167). |
| `_has_live_replacement(plan, failed) -> bool` | Two-tier (#213): Tier 1 causal (`Task.supersedes` chain, 16-hop bounded, any non-FAILED/CANCELLED status counts); Tier 2 name-pattern fallback (`<id>_v2` versioned at any live status; `retry_<id>` shared-lineage only at PENDING/RUNNING). Assignee-scoped when both ids carry `assignee_agent_id`. |
| `_fatally_failed_task_ids(plan) -> list[str]` | FAILED tasks with **no** live replacement. Drives both fail_fast gates. |
| `_pending_task_ids(plan) -> list[str]` | Every still-PENDING task id. Used by the legacy reachability audit. |
| `_unreachable_pending_task_ids(plan) -> set[str]` | PENDING tasks with a transitively-broken predecessor (seed = terminal-non-COMPLETED with no live replacement; forward-propagate to fixed point; AND-join = any broken predecessor is fatal). Drives `_sweep_unreachable_pending` (#208). |
| `_has_live_pending_or_running(plan) -> bool` | Any PENDING/RUNNING task remains. Gates the nudge replay (#202). |
| `_plan_divergence_drift_event(session, detail)` | Builds a CRITICAL `PLAN_DIVERGENCE` `DriftDetected` with real proto enum values (not `UNSPECIFIED`). |
| `build_task_nudge(task) -> str` | The canonical next-task nudge string (ported from harmonograf). Adapters that want harmonograf-identical prompting feed this to the agent. Re-exported from `goldfive.executors`. |

---

## Control-message data flow (end to end)

Trace one operator message all the way through, so you know which layer owns which step.
Take a `STEER` arriving mid-overlay-invocation as the canonical case:

1. **External send.** harmonograf (or a CLI/test) calls `channel.send(ControlMessage(
   kind=ControlKind.STEER, payload={"note": "focus on the auth flow instead"}))`. The
   message lands on `ControlChannel._inbox`. (harmonograf builds this from a
   `PostAnnotation`; the server maps `body` → `ControlEvent.steer.note`; the control bridge
   rehydrates it into `payload["note"]` — see `_compose_steer_restart_message`'s docstring.)
2. **Runner poll.** The executor is inside `_invoke_passthrough_with_control`, blocked on
   `asyncio.wait({invoke_task, recv_task})` where `recv_task = control.receive()`. The
   send unblocks `recv_task`; `msg = recv_task.result()` is the STEER.
3. **Dispatch.** `outcome = await dispatch_control(msg, session, steerer, sinks)`.
   `_kind_value(msg) == "STEER"` → `ControlOutcome(steer_message=msg, ack=SUCCESS("steer
   queued"))`. **No state mutated yet** — dispatch for STEER is pure classification.
4. **Ack.** `await control.ack(outcome.ack)` — the ack lands on `_outbox`; harmonograf's
   `acks()` iterator surfaces `AckResult.SUCCESS` so the UI knows the annotation landed.
5. **Executor action.** `outcome.steer_message is not None` → `_cancel_invoke_task(
   invoke_task)` (5s grace) → return `("steer", msg)` up into `_run_overlay`.
6. **Loop branch.** `_run_overlay` sees `kind == "steer"` → `_restart_after_user_steer(
   payload=msg, …)`.
7. **Steer applied.** `apply_steer(msg, …)` → `steerer.drift.observe(msg, session)` → the
   steerer classifies it as `USER_STEER` drift, runs the cascade-cancel of contaminated
   tasks, calls `planner.refine`, and swaps `session.plan` in place (11-state-ownership.md).
   Whatever events that emits (`DriftDetected(USER_STEER)`, `PlanRevised`) go to the sinks.
8. **Restart composed.** `state.current_user_input = _compose_steer_restart_message(msg,
   …)` (the `[USER STEERING CONTROL …]` framed body); `reconciler.reset_for_new_plan(
   session.plan)`; `state.next_reentry_kind = ReentryKind.STEER_REPLAY`.
9. **Re-invoke.** The loop `continue`s; `_race_control` wraps the next
   `invoke_passthrough(state.current_user_input, …)` in `reentry(STEER_REPLAY)` so plugins
   watching the inner runner attribute the replay correctly.

The key ownership boundaries: **`dispatch_control` classifies and acks; the executor acts;
the steerer mutates the plan.** A `STEER` never mutates `session.plan` inside
`dispatch_control` — that happens in step 7, inside the steerer, driven by the executor.
This separation is what lets both executors share one dispatcher while keeping their own
action logic.

Contrast with a `REWIND_TO`, which *does* mutate inside `dispatch_control` (via
`_rewind_plan`, under `channel_processor_active()`), because the rewrite is purely
structural (reset statuses to PENDING) and needs no steerer/planner involvement. And a
`STATUS_QUERY`, which mutates nothing and emits nothing — it only fills the ack detail.

The three classes of dispatch side effect:

| Class | Kinds | Where the effect happens |
|---|---|---|
| Pure classification (ack only) | `PAUSE`, `RESUME`, `STEER`, `GOLDFIVE_STEER`, `GOLDFIVE_PAUSE_ESCALATE`, `CANCEL` | Effect deferred to the executor via `ControlOutcome` fields |
| Mutate-in-dispatch | `REWIND_TO` (plan swap), `INTERCEPT_TRANSFER` (flag), `APPROVE`/`REJECT` (waiter + event) | Inside `dispatch_control` / its helpers |
| Read-only | `STATUS_QUERY` | Snapshot string into the ack detail |

---

## Cancel-reason taxonomy

Every `TaskCancelled` / `RunAborted` carries a structured `cancel_reason` (goldfive#205) so
harmonograf's Trajectory view can answer "why was this cancelled?". Getting the prefix right
matters — the frontend dedupes annotation rows against drift rows by it (#176). The prefixes
the executors and dispatch produce:

| Prefix | Produced at | Meaning |
|---|---|---|
| `user_cancel:<annotation_id>` | `dispatch_control` (CANCEL, annotation present) | Operator hit Cancel; the id ties the row to the annotation. |
| `user_cancel:<msg.id>` | `dispatch_control` (CANCEL, no annotation) | Direct SDK CANCEL with no annotation id. |
| `user_cancel:cancelled_by_control` | `mark_cancelled_if_live` default | Fallback when no more specific reason was threaded. |
| `user_steer:<annotation_id>` | `_steer_cancel_reason_prefix` (STEER, annotation present) | Task superseded by an operator steer. |
| `user_steer:<control_id>` | `_steer_cancel_reason_prefix` (STEER, no annotation) | Direct SDK STEER. |
| `user_steer:steer` | `_steer_cancel_reason_prefix` final fallback | Ensures the reason is never empty on a steer-driven cancel. |
| `run_aborted:orphaned by plan revision failure` | legacy reachability audit / `_sweep_unreachable_pending` | PENDING task whose predecessor chain is broken with no live replacement. |
| `run_aborted:pause_escalate_deadline:<drift_kind>` | `abort_expired_pause` | Non-terminal task cancelled because a pause deadline expired with no operator action. |
| `adk_cancellation:stage_task` | parallel `_run_stage` fold | A stage task cancelled by `asyncio.CancelledError` (routed through the steerer so a `TaskCancelled` still reaches sinks — previously a silent status mutation). |

`_steer_cancel_reason_prefix(steer_msg)` (`sequential.py`) is the helper that builds the
`user_steer:*` prefix; it reads `payload["annotation_id"]` first, then `msg.id`, then the
`user_steer:steer` constant. When you add a new cancel path, mint a prefix in the same
`<class>:<detail>` shape and thread it through `steerer.transition(..., cancel_reason=…)` —
never emit a bare, prefixless reason.

---

## Worked walkthroughs

### A goldfive supersede-cancel (non-fatal, the common autonomous case)

The steerer detects `LOOPING_REASONING` mid-invocation, refines the plan (spawns
`analyze_v2` superseding `analyze`), and needs the in-flight invocation to stop so the tree
picks up the new plan:

1. `DefaultSteerer._cancel_inflight_for_revision` sets `session._supersede_pending = True`
   (and stamps the per-`invocation_id` `StateStore` registry), then cancels the invoke task.
2. `_invoke_passthrough_with_control`'s `invoke_task` raises `CancelledError` →
   returns `("cancelled", "invoke cancelled")`.
3. `_run_overlay` → `_handle_invoke_cancelled`. Union read: `supersede_pending == True`.
   `_fail_fast_on_invoke_cancel` is `False` (default). → **non-fatal branch**: clear both
   markers, `reconciler.reset_for_new_plan(session.plan)`, `return None`.
4. Loop `continue`s. `_clear_stale_supersede` wipes the markers again (idempotent).
   `_race_control` re-invokes `invoke_passthrough` with the *unchanged* `user_input` (a
   supersede swaps the plan, not the request) and **no** reentry pin.
5. The tree runs the revised plan; `analyze_v2` executes; the turn continues to
   `RunCompleted`.

Had `_fail_fast_on_invoke_cancel` been `True` (CI/regression), step 3 would have taken the
abort branch → `RunAborted`. Had the cancel been an **external** USER_CANCEL (no supersede
marker), step 3 always aborts regardless of the flag.

### An operator CANCEL during a pre-task pause (legacy loop)

1. Operator sends `PAUSE`. `_apply_pre_task_controls` drains it → `request_pause=True` →
   the pause `while` loop blocks on `control.receive()` (unbounded — operator PAUSE carries
   no `deadline_s`).
2. Operator sends `CANCEL` with `{annotation_id: "a17"}`. `receive()` returns it;
   `dispatch_control` → `ControlOutcome(cancel_run=True,
   cancel_reason_prefix="user_cancel:a17")`; ack published.
3. The pause loop's `if outcome.cancel_run:` stashes the prefix on
   `session._last_cancel_reason_prefix` and `raise _ControlCancelled(reason)`.
4. `run`'s `except _ControlCancelled` sets `run_failed = True; break`.
5. Terminal cascade: `_drain_steerer_at_run_boundary` → `RunAborted(reason)`. Any
   still-live task the loop was about to pick is left for the reachability audit / the
   caller's teardown; the CANCEL prefix rides on `_last_cancel_reason_prefix` for the next
   per-task cancel.

### A pause-escalation TERMINATE landing on an unbounded Level-4 pause (#404 + #482)

1. Steerer hits Level 4 → dispatches `GOLDFIVE_PAUSE_ESCALATE` with **no** `deadline_s`.
   The pre-* loop enters an unbounded pause (`deadline_at is None`).
2. The condition persists; steerer's ladder reaches TERMINATE → dispatches a second
   `GOLDFIVE_PAUSE_ESCALATE` carrying `deadline_s=600`.
3. In the pause loop, `outcome.goldfive_pause_message is not None` →
   `pause_deadline_s(...) == 600` → since `deadline_at is None`, adopt it:
   `deadline_at = monotonic() + 600`. The loop keeps blocking, now **bounded**.
4. If no operator `RESUME`/`CANCEL`/`STEER` arrives within 600s, `asyncio.wait_for` raises
   `TimeoutError` → `abort_expired_pause` cancels every non-terminal task with
   `run_aborted:pause_escalate_deadline:<drift_kind>` and raises `_ControlCancelled` →
   `RunAborted` with the escalation lineage.

Note this whole path only works if a `control` channel is attached. Headless runs
(`control is None`) drop the ladder pause at the source; the `HUMAN_INTERVENTION_REQUIRED`
drift on the sink stream is the only signal there.

---

## Common mistakes

Concrete wrong edits a weaker model plausibly makes here, each with the correct alternative.

### 1. Adding an intervention consumed at only one executor / only one loop

**Wrong.** "Make goldfive able to hard-redirect the agent mid-run." You add a new
`ControlKind`, a `ControlOutcome` field, and handle it in
`_invoke_passthrough_with_control`. Tests pass under `goldfive.wrap` (overlay). You ship.
The parallel executor and the legacy sequential loop silently ignore it (they only check
`cancel_run` / `steer_message`), and any operator using `ParallelDAGExecutor` gets nothing.

**Right.** Enumerate the five consumption sites from the parity table and touch every one
that should honour the new control: `_invoke_with_control`, `_invoke_passthrough_with_control`,
`_run_stage`, `_apply_pre_task_controls`, `_apply_pre_stage_controls` — plus `dispatch_control`
and the `ControlOutcome` field. If a site *should not* honour it, add a comment saying so.
Add a test per executor. The already-shipped `GOLDFIVE_STEER` asymmetry (honoured in the
sequential pause loop via #404 but **not** the parallel one) is the cautionary precedent.

### 2. Forgetting the parallel executor exists

**Wrong.** You change the abort semantics, the terminal-status handling, or the
cancel-reason prefixing in `sequential.py` and consider the task done.

**Right.** `ParallelDAGExecutor` is a first-class, shipped executor with its own `run`,
`_run_stage`, `_refine`, and `_apply_pre_stage_controls`. Terminal-status logic
(`TERMINAL_TASK_STATUSES`, #485), cancel-reason prefixing (#205), and drift-pipeline failure
emission (`emit_pipeline_failure_drift`, #134) are duplicated by design in both. Grep both
files: `grep -n "TERMINAL_TASK_STATUSES\|_last_cancel_reason_prefix\|emit_pipeline_failure_drift"
goldfive/executors/*.py`. When you change shared behaviour, change `_shared.py` /
`_control.py` (the "cannot drift apart" modules) rather than one executor, or change both and
add a test to each.

### 3. New pause path without a deadline

**Wrong.** You add a code path that sends a `GOLDFIVE_PAUSE_ESCALATE` (or any new pause) and
omit `deadline_s`. Under a headless run with no operator, the pre-* pause loop blocks on
`control.receive()` forever — a wedged run that never emits a terminal event.

**Right.** Every goldfive-authored pause must carry a `deadline_s` (from
`SteeringConfig.pause_escalate_deadline_s`, or the #482 TERMINATE built-in 600s fallback), or
a deliberate `None` with a comment explaining why unbounded is correct (operator-issued PAUSE
is the only sanctioned unbounded pause, because a human is watching). The deadline is what
`pause_deadline_s` extracts and `abort_expired_pause` enforces. Test with
`tests/test_pause_deadline.py` / `tests/test_pause_escalate.py`.

### 4. Collapsing `goldfive_steer_message` into `steer_message`

**Wrong.** "These are both steers, dedupe the field." You route `GOLDFIVE_STEER` through the
user-STEER branch. Now the overlay calls `apply_steer` → `steerer.drift.observe` on a message
the steerer *originated* → the steerer re-observes its own drift → refine loop; and the LLM
gets `[USER STEERING CONTROL]` framing for a goldfive decision.

**Right.** Keep the three distinct `ControlOutcome` message fields. The goldfive-steer branch
(`_restart_after_goldfive_steer`) must **not** call `apply_steer` (the plan swap already
happened) and must use `source="goldfive"` framing with the superseded/replacement task-id
block.

### 5. Reintroducing the #163 amplification

**Wrong.** "Some PENDING tasks never ran; re-invoke to cover them." You add a loop that
re-invokes `invoke_passthrough` for every PENDING task at overlay exit, or you widen
`_drain_nudges` to fire whenever any nudge-eligible condition holds.

**Right.** The only sanctioned post-invocation re-invoke is `_drain_nudges`, gated on (a) the
steerer *explicitly* queued a nudge in `session.pending_nudges`, (b)
`state.nudge_replays < _MAX_NUDGE_REPLAYS`, (c) live work remains, and (d)
`steering_is_active(steerer)`. Uncovered PENDING is handled by `_sweep_unreachable_pending`
(leave reachable live, cancel unreachable) — not by re-invocation. STEER is the user path for
exercising uncovered work.

### 6. Reading the kill-switch directly / enforcing under observation_only

**Wrong.** `if session._observation_only: ...` or `if config.observation_only: ...`, or
adding an enforcement (abort, cancel, nudge replay) with no observation_only guard.

**Right.** The only sanctioned read is `steering_is_active(steerer)` (imported from
`goldfive.steerer`). Every enforcement in these executors is guarded by
`not steering_is_active(steerer)` → log-and-continue. There are three carve-out sites in the
overlay alone (`_handle_goldfive_pause`, `_drain_nudges`,
`_invoke_passthrough_with_control`'s goldfive-pause branch) plus the fail_fast gates in
`_classify_fatal_failure` and the legacy `run`. A missing/None/raising predicate must resolve
to PASSIVE (invariant 2).

### 7. Mutating `task.status` in place

**Wrong.** `task.status = TaskStatus.CANCELLED` anywhere in an executor.

**Right.** `Plan`/`Task` are frozen (#247). Route status changes through
`steerer.transition(task_id, status, …)` (emits the `Task*` event with a structured
`cancel_reason`) or, for control-plane rewrites, `with_task_status` + `set_session_plan` under
`channel_processor_active()` (as `_rewind_plan` does). A silent in-place mutation both trips
the frozen-dataclass error and skips the observable transition event.

### 8. Catching `_ControlCancelled` (or `CancelledError`) with `except Exception`

**Wrong.** Wrapping an `invoke` call in `try: ... except Exception: log; continue`.

**Right.** `_ControlCancelled` is a `BaseException` specifically so `except Exception` does
not swallow it; `asyncio.CancelledError` is likewise a `BaseException`. New wrapping code
must either use `except Exception` (which correctly lets both propagate) or, if it uses
`except BaseException`, explicitly re-raise `asyncio.CancelledError` and `_ControlCancelled`.
The `_refine` cancellation branches (`except BaseException … mark_stash_completed(); raise`)
are the pattern to copy.

### 9. `receive(timeout=…)` in a deadline-bearing pause loop

**Wrong.** Replacing `asyncio.wait_for(control.receive(), timeout=remaining)` with
`control.receive(timeout=remaining)` to "simplify".

**Right.** `receive(timeout=…)` returns `None` on both timeout and channel-close, which the
loop treats as "resume"; a benign timeout would then unwedge a pause that should have aborted.
`asyncio.wait_for(...)` raises `TimeoutError` on expiry, which routes to `abort_expired_pause`.
The distinction is load-bearing — keep the `wait_for` form.

### 10. Re-keying the supersede discriminator on a churning id

**Wrong.** "Track supersede per task-id" or per LLM-minted plan-revision id.

**Right.** The supersede-cancel discriminator is the `session._supersede_pending` bool plus
the `StateStore.for_session(session)` per-`invocation_id` registry (`invocation_id` is unique
and cannot be clobbered cross-invocation). Both are cleared at the top of every overlay
iteration by `_clear_stale_supersede`. Do not key it on anything the LLM mints per turn
(invariant 5). Issue #430 tracks retiring the bool in favour of the registry alone.

---

## Verification checklist

Run these after touching anything in this subsystem. Commands assume repo root
`/home/sunil/git/goldfive` and the dev+adk extras installed (`uv sync --extra dev --extra
adk`).

**1. The executor + control test suites (fast, run these first):**

```bash
uv run pytest -q \
  tests/test_sequential_executor.py \
  tests/test_sequential_executor_overlay.py \
  tests/test_parallel_executor.py \
  tests/test_executor_control.py \
  tests/test_control_primitive.py \
  tests/test_control_proto.py
```

**2. Overlay-specific behaviour (steer, forward-progress, supersede, reentry):**

```bash
uv run pytest -q \
  tests/test_overlay_steer.py \
  tests/test_overlay_forward_progress.py \
  tests/test_executor_supersede_cancel_nonfatal.py \
  tests/test_goldfive_steer_request_cancel.py \
  tests/test_adk_reentry.py \
  tests/test_steer_unification.py
```

**3. Pause deadline / TERMINATE (#482) and the observation_only carve-outs:**

```bash
uv run pytest -q \
  tests/test_pause_deadline.py \
  tests/test_pause_escalate.py \
  tests/test_observation_only_pause_escalate_carveout.py \
  tests/test_observation_only_nudge_gate.py \
  tests/test_observation_only_abort_carveout.py
```

**4. Terminal-status single-source (#485), lineage cap, reconciler, run-boundary drain:**

```bash
uv run pytest -q \
  tests/test_terminal_statuses_single_source.py \
  tests/test_task_lineage.py \
  tests/test_plan_reconciler.py \
  tests/test_drift_drain_at_run_boundary.py \
  tests/test_runner_close_orphan_audit.py
```

**5. Parity grep — before you claim a control is honoured "everywhere", confirm all five
consumption sites reference it:**

```bash
# Every mid-work + pre-work consumer of a given ControlOutcome field:
grep -n "goldfive_steer_message\|goldfive_pause_message\|steer_message\|cancel_run" \
  goldfive/executors/sequential.py goldfive/executors/parallel.py

# Terminal-status handling must reference the single frozenset, not a local set:
grep -n "TERMINAL_TASK_STATUSES" goldfive/executors/*.py goldfive/types.py

# No direct kill-switch reads in the executors (should show only steering_is_active):
grep -n "observation_only\|steering_is_active\|is_active_steering" goldfive/executors/*.py
```

**6. No in-place status mutation (should return nothing):**

```bash
grep -nE "\.status\s*=\s*TaskStatus\." goldfive/executors/*.py
```

**7. Full suite + lint (the merge gate — must be green, ~30s):**

```bash
uv run pytest -q          # ~2912 passed / ~61 skipped
ruff check .              # must stay clean; do NOT ruff-format (repo is not format-clean)
```

**8. If you changed `ControlKind` / `ControlMessage` shape**, the proto lockstep guard is
non-optional:

```bash
uv run pytest -q tests/test_control_proto.py
```

and update `proto/goldfive/v1/control.proto` in the same change — the `StrEnum` members and
the proto enum must match in both directions.

**9. If you touched the overlay's control race or pause loop**, drive an end-to-end
steering run rather than trusting unit tests alone (per the layered-validation discipline —
DB-only checks miss flow breakage): `tests/test_live_steering_e2e.py` is the closest
in-repo harness.

```bash
uv run pytest -q tests/test_live_steering_e2e.py
```
