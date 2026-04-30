# Task lifecycle & protocol choreography

This document reviews how the goldfive protocols (Planner, Executor,
Adapter, Steerer, reporting-tool handlers) interact across a task's
end-to-end lifecycle. It is the companion to
[STATE-MACHINE.md](STATE-MACHINE.md), which focuses on *what* the
states are; this doc focuses on *how* the protocols move a task
through them and where the contracts between them sit.

Audiences:
- implementers writing a new adapter / executor / sink who need to
  know what a correct implementation looks like,
- operators triaging a stuck run who need to know which layer owns
  which behaviour,
- reviewers validating that a proposed change doesn't break an
  invariant another layer depends on.

Citations are file:line into `goldfive/` unless otherwise noted.

---

## 1. Creation protocol

A Plan is the unit of work and a Task is an entry in a Plan. Two
paths produce one.

### 1.1 Planner.generate — initial plan

`LLMPlanner.generate` (`planner.py:397–455`) takes `Goal[]` +
available agent list, calls the configured LLM, parses the JSON
response, and returns a `Plan`. Defaults:

- Every task starts at `TaskStatus.PENDING` (`types.py`, `Task`
  dataclass default).
- `Plan.revision_index = 0`, `revision_reason = ""`.
- `Plan.edges` are `TaskEdge(from_task_id, to_task_id)`. Topological
  order is computed on demand by
  `Plan.topological_stages()` (`types.py`), not cached.

### 1.2 Planner.refine — revised plan

`LLMPlanner.refine` (`planner.py:743–850`) branches on `drift.kind`
and dispatches to one of three paths:

- `USER_STEER` → `_refine_user_steer` (`planner.py:1010–1115`):
  delete-and-replan. Completed / failed / cancelled tasks are
  preserved verbatim by id; PENDING / RUNNING / BLOCKED tasks are
  dropped. LLM generates fresh PENDING work. Final plan =
  `done_tasks + fresh_pending`.
- `LOOPING_TOOL_CALL` / `LOOPING_REASONING` →
  `_refine_looping_tool_call` (`planner.py:849–945`): the looping
  task is forced to FAILED (framework overrides the LLM if it
  forgets). There is a **deterministic fallback** (`planner.py:948–1015`)
  that marks the looper FAILED locally if the LLM is unreachable
  entirely — refine cannot soft-fail into "no change."
- All other drift kinds → generic refine prompt. The LLM may rewrite
  arbitrarily, subject to the invariants enforced by
  `_apply_revision` below.

### 1.3 Invariants the steerer enforces on any refined plan

`DefaultSteerer._apply_revision` (`steerer.py:512–531`):

- `revision_index` monotonically increases: `new_index ≥
  old_index + 1`.
- `revision_kind` / `revision_severity` / `revision_reason` are
  stamped from the triggering drift if the planner didn't set them.
- `session.plan` is replaced as a single atomic assignment — readers
  in the adapter / executor see either the old plan or the new plan,
  never a half-merged state.

**Soundness gap (open):** `Plan` has no structural validation at
creation or revision — duplicate task IDs, edges referencing missing
tasks, or cycles are silently accepted. `topological_stages()`
tolerates cycles by appending un-placeable tasks to a final stage.
See §7.

---

## 2. Assignment protocol

Once a Plan exists, tasks move from `PENDING` to one of two active
states (`RUNNING` under the overlay model, via a
`PlanReconciler` observation; or `PENDING → NOT_NEEDED` at invocation
end if the tree never exercised it). Under the legacy per-task loop
the executor drives `PENDING → RUNNING` directly.

### Overlay-model assignment

Under the default overlay mode (goldfive#141, refined by #163 /
#149 / #152) the executor issues ONE
`adapter.invoke_passthrough(user_input)` per run / turn. While the
tree runs:

1. ADK plugin's `before_agent_callback` fires → the plugin calls
   `PlanReconciler.on_before_agent(name, invocation_id, parent_invocation_id)`
   → the reconciler picks the first PENDING task assigned to that
   agent (or a contextual match via the invocation-parent chain,
   goldfive#151) and transitions it `PENDING → RUNNING` via
   `steerer.transition`.
2. `after_agent_callback` fires with an optional error → the
   reconciler transitions `RUNNING → COMPLETED` (no error) or
   `RUNNING → FAILED` (error).
3. Drift signals (PLAN_DIVERGENCE from three-stage gate,
   LOOPING_REASONING from tool-loop and reasoning detectors, ...)
   ride the ADK plugin → steerer.observe / steerer._handle_drift
   → intervention ladder.

### Invocation-end `PENDING → NOT_NEEDED` sweep (goldfive#163)

When the passthrough invocation generator ends cleanly (tree
finished its natural flow), `SequentialExecutor._run_overlay`
walks the live plan and transitions every still-PENDING task to
`NOT_NEEDED`. The previous behaviour (goldfive#141 pre-#163)
dispatched a soft follow-up per missed task; flow-prompted
coordinators re-ran their full pipeline on every follow-up,
amplifying a ~10 min run into 40+ min. STEER remains the user-
driven path for exercising uncovered work.

### Legacy per-task assignment (overlay_mode=False)

### 2.1 Who picks the next task (legacy per-task mode)

`SequentialExecutor._pick_next_task` (`executors/sequential.py:~624`)
walks topological stages and returns the first task that is `PENDING`
**and** whose incoming-edge predecessors are all `COMPLETED`.
`BLOCKED` tasks are never picked — they can only return to `RUNNING`
through a refine-driven transition (see STATE-MACHINE.md §"Blocked
vs non-blocked resume").

`ParallelDAGExecutor` (`executors/parallel_dag.py`) picks all
ready-to-run tasks within a stage and invokes them concurrently; the
Steerer's state mutations are serialised through an asyncio lock so
concurrent `report_task_*` calls don't interleave state writes.

### 2.2 Who drives PENDING → RUNNING

Two entry points, reconciled by the terminal-status guard:

1. **Executor pre-invoke transition.** Before calling
   `adapter.invoke(...)`, the executor calls
   `steerer.transition(task_id, RUNNING, ...)` so observability sinks
   see a `TaskStarted` event even for adapters whose agent never
   calls `report_task_started`.
2. **Agent-driven transition.** When the agent calls
   `report_task_started(task_id)`, the reporting-tool handler routes
   through `DefaultSteerer.mark_task_running`
   (`steerer.py:142–159`). If the task is already `RUNNING`, the
   `_TERMINAL_TASK_STATUSES` check falls through harmlessly (RUNNING
   is not terminal) and the second call is a no-op.

These two entries converge on `DefaultSteerer.mark_task_running`,
which is idempotent by design (`if task.status in _TERMINAL_…:
return` + re-emit-suppression on same-status writes).

### 2.3 The adapter contract for `invoke`

`adapter.invoke(task, session) -> InvocationResult` is the single
boundary between goldfive and the agent framework.

- **Input:** exactly one task, plus the live session (so the adapter
  can read the current plan / completed results / agent notes).
- **Output:** `InvocationResult(task_id, text, stop_reason, error,
  raw)`. `stop_reason` is a short string — see §3 for the set of
  values the ADK adapter returns.
- **During the call:** the agent may issue any number of LLM turns
  and tool calls. Reporting-tool calls route through
  `goldfive.adapters._tool_invocation.invoke_tool` (`§5`) which
  mutates `session.plan` in place.
- **Lifetime:** the adapter must return when the task is done.
  "Done" means one of: the agent produced a final response, the
  agent reported the task terminal (COMPLETED / FAILED / CANCELLED),
  an exception propagated, or the executor cancelled the invoke
  coroutine.

### 2.4 Single-Runner dispatch (ADK adapter)

`ADKAdapter.invoke` drives exactly one runner — the one built
around the wrap-target root — for every task. Delegation within
the tree happens via ADK's native `AgentTool`, `transfer_to_agent`,
and `sub_agents` mechanisms (see [ARCHITECTURE.md §"Single-Runner dispatch"](ARCHITECTURE.md#single-runner-dispatch-goldfive-drives-the-root-adk-delegates-within)).
`task.assignee_agent_id` rides on the task for observability and
as a delegation hint the agent's prompt can read via the state
protocol, but goldfive does not route on it.

ADK's plugin manager propagates into any AgentTool-spawned
sub-Runner, so nested invocations still see the state-protocol
keys and emit `AgentInvocationStarted` / `AgentInvocationCompleted`
/ `DelegationObserved` events ([EVENT-MODEL.md §"Agent-invocation events"](EVENT-MODEL.md#agent-invocation-events)).

A per-invocation AgentTool-spawn cap (default 16, configurable via
`ADKAdapter(agent_tool_cap=N)`) catches runaway coordinators whose
prompts describe a pipeline. On exceed the plugin emits a
`RUNAWAY_DELEGATION` drift at CRITICAL severity and cancels the
invocation; the current task is marked FAILED (via the Steerer's
refine path) and the planner's `refine` hook gets a chance to
salvage the run.

### 2.5 State-protocol writes live in the plugin's `before_run_callback`

The authoritative write of the `goldfive.*` session-state keys
(`run_id`, `plan_context`, `current_task`, `tools_available`)
happens inside the goldfive plugin's `before_run_callback`, *not*
in `ADKAdapter.invoke`. This is the reliability-critical path:
`invocation_context.session` inside the callback is the **live**
session ADK is running against, so writes are visible to every
subsequent callback and tool on the same session — including
AgentTool-spawned sub-Runners, whose own `before_run_callback`
fires with their own live session and therefore gets its own
authoritative seed.

Previously the adapter wrote these keys against a session fetched
via `session_service.get_session`, which ADK's `InMemorySessionService`
returns as a shallow copy. The writes landed on a stranded dict
the runner never saw. That made state-protocol keys unreliable
(the user-memory note ["Verify plugin callback state handoff is
read-readable"](..) covers the class of bug). The relocation to
`before_run_callback` closes the gap structurally — the only
session state write that matters is the one on the live session,
so that is where goldfive writes it.

`adapter.invoke` still mirrors the SessionContext into ADK state
as a best-effort fallback for legacy unit-test harnesses that
drive callbacks with a hand-built state dict; the live-run path
does not depend on that write succeeding.

---

## 3. Quiescence protocol

When does an invocation end? This is where symptomatic bugs have
historically crept in — "the agent keeps running after it reports
the task done" is a quiescence failure, not a reporting-tool bug.

### 3.1 The ADK adapter's loop exits on four conditions

`ADKAdapter.invoke` (`adapters/adk.py:463–537`) iterates events from
`self._runner.run_async(...)` and stops when:

| Exit | `stop_reason` | Cause |
|---|---|---|
| Final event | `final_response` | `_is_final_event(event)` returns True (the agent emitted a final response). |
| Terminal task | `task_terminal` | `_task_is_terminal(task, session)` returns True — the agent reported the task COMPLETED/FAILED/CANCELLED via a reporting tool. *This is the structural fix for the call-budget-burn pattern — without it, the adapter would keep feeding events and let the agent spam reporting tools until ADK's 500-call ceiling triggers an error.* See `adapters/adk.py:~540–560`. |
| Exception | `error:<ExcName>` | The generator raised; caught, logged, and stored in `InvocationResult.error`. |
| Cancellation | (executor-driven) | The executor cancelled the invoke coroutine via `_cancel_invoke_task` (see §6). |

### 3.2 Post-invoke reconciliation

After `adapter.invoke` returns, the executor reads the *live* task
status from `session.plan` (not from the `task` object passed in —
it may be stale) and:

- If status is already terminal (reporting-tool drove it), the
  executor only emits a `TaskEnded`-style event for observability and
  moves on.
- If status is still `PENDING` or `RUNNING`, the executor
  auto-transitions:
  - `COMPLETED` if `result.error is None`,
  - `FAILED` if `result.error is not None`.

This means a correctly-written adapter never has to actively mark
its task done — the executor will do it if the agent forgets.

### 3.3 No goldfive-level timeout

There is no goldfive-level wall-clock timeout on `invoke`. The ADK
framework has its own per-invocation max-LLM-call ceiling (default
500). If the agent hangs or stalls silently, the run relies on ADK's
timeout, or the `ToolLoopTracker`'s graduated-severity drift path
(§5.3), or a CANCEL on the control channel.

**Soundness gap (open):** there is no cap on the number of
invocations per task by default. A task can be re-invoked indirectly
through a refine that produces a `retry_<task>` task; if the retry
also loops, a new refine is triggered, and so on. Blast radius is
bounded primarily by `max_retries_per_task_lineage` (default 3,
see §7.7) and by the per-tool / per-ADK turn caps inside the
adapter; `SequentialExecutor.max_task_invocations` (default
`None` == unbounded) can be set to a finite integer as a
belt-and-suspenders run-wide cap when you want a hard ceiling on
total adapter invocations. See §7.

---

## 4. Revision protocol

A drift event with `severity ≥ WARNING` triggers refine via
`DefaultSteerer._handle_drift` (`steerer.py:492–555`). Drifts come
from two places:

- **Observed drift:** `Steerer.observe(event, session)` runs the
  drift classifiers on raw adapter output (tool errors, refusals,
  stop reasons).
- **Reporting-tool drift:** the tool handlers call
  `_handle_drift` directly — `mark_task_failed`,
  `mark_task_blocked`, `report_new_work_discovered`,
  `report_plan_divergence`, `report_awaiting_approval` (on timeout).

### 4.1 What refine preserves

All done tasks (`COMPLETED`, `FAILED`, `CANCELLED`) are preserved
verbatim — same id, title, assignee, status, bound_span_id. This is
enforced by:

- The prompt to the LLM (don't rewrite done tasks).
- `_refine_user_steer`'s explicit merge step
  (`planner.py:~1080–1100`).
- `_refine_looping_tool_call`'s deterministic fallback plan
  (`planner.py:948–1015`).

### 4.2 In-flight invocation during refine

A refine that fires inside a reporting-tool handler does not cancel
the ADK generator. The in-flight invocation keeps running but:

- Reads from `session.plan` see the new plan (atomic swap).
- The early-break quiescence check (§3.1) catches the case where
  the refine marked the currently-running task terminal — the
  adapter loop exits on the next event.
- Subsequent tool calls from the still-running agent are checked
  against the new plan by the terminal-task rejection layer (§5.1).

### 4.3 Refine-failure surfacing

If `planner.refine` raises **or** returns `None`,
`_handle_drift` no longer silently swallows the failure. It logs a
WARNING and emits a follow-up `DriftDetected` with `severity =
CRITICAL` and `detail = "refine failed (<kind>): <reason>"`. This
lets sinks (and the harmonograf UI) render the refine failure
rather than hiding it behind a stale plan.

**Soundness gap (open):** refine failures do not themselves
terminate the run. If the LLM is down entirely, the same drift can
retrigger → refine fails → drift emitted again. The per-lineage
cap (`max_retries_per_task_lineage`) and the optional
`max_task_invocations` ceiling eventually stop the loop, but there
is no per-drift-kind backoff or dedup. See §7.

---

## 5. Reporting-tool dispatch

**Invariant — all adapters route reporting-tool calls through
`invoke_tool`.** The dispatcher owns schema validation; every
subsequent decision (idempotency, invalid-transition, state rotation)
lives inside the reporting handlers themselves. Adapters MUST NOT
short-circuit by calling `spec.handler(...)` directly — doing so
silently bypasses schema validation. The regression guards for this
invariant are:

- ADK adapter: `tests/test_adk_adapter.py::test_reporting_tool_dispatch_routes_through_invoke_tool`
  (asserts an unknown `task_id` gets `unknown_task_id` instead of
  reaching the handler) and the companion tests
  (`test_reporting_tool_on_terminal_task_returns_structured_rejection`,
  `test_reporting_tool_duplicate_returns_idempotent_ack`).
- Claude adapter: `tests/test_claude_adapter.py::test_pretooluse_hook_on_terminal_task_returns_structured_rejection`.
- Callable adapter: the user callable is expected to call
  `invoke_tool(tools, name, args, session, steerer)` directly
  (see `tests/test_callable_adapter.py::test_tool_routing_drives_session_transitions`).
  The adapter itself does not dispatch — it just forwards the spec
  list to the user's callable.

The current dispatch protocol has a single dispatcher-owned layer
(schema validation) plus two handler-owned layers (idempotency +
invalid-transition). Tool-loop detection is handled out-of-band by
`ToolLoopTracker` at the ADK plugin's `after_tool_callback`, not
inline on the dispatch path.

> **Historical note (goldfive#206).** Earlier versions of this
> document described a four-layer pipeline with a per-task +
> session-wide `ToolLoopGuard` that lived in
> `goldfive/adapters/_tool_loop_guard.py`. The guard was retired
> in goldfive#206: it pre-dated both the handler-owned idempotency
> matrix (#201, #203) and the graduated-severity
> `ToolLoopTracker` (#181, #194, #204), so benign idempotent
> retries were firing CRITICAL `LOOPING_TOOL_CALL` drifts and
> aborting runs. The newer stack covers every protection the old
> guard provided.

### 5.1 Layer 1 — schema + terminal-task rejection (prevention)

**Schema rejections.** For task-scoped tools (every canonical tool
except `report_plan_divergence`), the dispatcher first validates
the call carries a usable `task_id`:

- **Missing task_id** — if `task_id` is absent / empty / whitespace:
  ```python
  {"acknowledged": False, "error": "missing_task_id", "tool": "<name>",
   "message": "Tool '<name>' requires a task_id; ..."}
  ```
- **Unknown task_id** — if the id doesn't appear in the current
  `session.plan.tasks`:
  ```python
  {"acknowledged": False, "error": "unknown_task_id", "tool": "<name>",
   "task_id": "<id>",
   "message": "Task with id '<id>' does not exist in the current plan ..."}
  ```

These rejections fire **before** the call reaches the handler, so a
malformed-call flood never drives a steerer transition.

**Terminal-task handling (goldfive#201).** Terminal-task retries are
owned by the handler, not the dispatcher. The handler's per-tool
idempotency matrix splits the terminal case two ways:

* **Same-transition retry** (e.g. `report_task_completed` on a
  `COMPLETED` task) returns an idempotent ACK. No steerer call, no
  state mutation, no drift:

  ```python
  {
      "acknowledged": True,
      "idempotent": True,
      "current_status": "COMPLETED",
  }
  ```

* **Cross-transition** (e.g. `report_task_started` on a `COMPLETED`
  task, or `report_task_progress` on a `FAILED` task) returns a
  structured invalid-transition error — a real "agent is confused
  about state" signal:

  ```python
  {
      "acknowledged": False,
      "error": "invalid_transition",
      "tool": "<name>",
      "task_id": "<id>",
      "current_status": "FAILED",
      "attempted": "RUNNING",
      "message": "Cannot 'report_task_started' task '<id>' from FAILED "
                 "to RUNNING. The task is already in a terminal or "
                 "otherwise-incompatible state; do not retry.",
  }
  ```

Pre-goldfive#201 the dispatcher short-circuited both cases with a
single `task_already_terminal` error. That shape conflated a
confused-model retry (benign) with a structural invalid transition
(a confusion signal) — benign retries tripped the tool-loop detector
and triggered spurious plan revisions. The matrix separates the two.

**Why a structured error and not a silent ACK on the invalid case.**
`{"acknowledged": True}` cannot be read by the model as "stop." The
structured error gives the agent clear, model-readable feedback that
loops can respect. The idempotent ACK is readable as "already done,
move on"; the invalid-transition error is readable as "you're
confused, do not retry this specific transition".

### 5.2 Layer 2 — handler-owned idempotency + invalid-transition

The reporting handlers (in `goldfive.reporting`) own the status-
machine semantics:

- **Same-transition retries** (e.g. a second `report_task_completed`
  on a `COMPLETED` task) return
  `{"acknowledged": True, "idempotent": True, "current_status": "..."}`
  without re-entering the steerer. See §5.1 for the full response
  shape.
- **Cross-transitions** on terminal tasks return
  `{"acknowledged": False, "error": "invalid_transition", ...}`.
- **Legal transitions** drive the steerer normally, and terminal
  transitions rotate `goldfive.current_task_id` via
  `orchestration_state.rotate_current_task_id` so the next
  sub-agent turn sees the next assigned task (goldfive#201 Bug B).

`report_awaiting_approval` is non-idempotent by design — polling the
same approval arguments is expected, and the handler blocks on the
approval waiter rather than short-circuiting.

### 5.3 Tool-loop detection (out-of-band)

Tool-loop detection is handled by
`goldfive.drift.tool_loops.ToolLoopTracker`, wired into the ADK
plugin's `after_tool_callback` (goldfive#181). It observes **every**
tool call the plugin sees — not just the reporting namespace — and
classifies runs of identical or same-name calls into INFO / WARNING
/ CRITICAL drifts keyed by category (meta vs work, goldfive#204).
Acknowledged-success reporting responses reset the per-(invocation,
agent) window via `on_task_progress`, so benign idempotent retries
never accumulate (goldfive#192, #206).

See `docs/design/DRIFT.md` for the full classifier contract and the
intervention-ladder routing.

---

## 6. Cancellation protocol

Cancellation arrives on the control channel as
`ControlKind.CANCEL` (or is induced by `USER_STEER` that the refine
translates). The executor handles it at two points:

- **Between tasks** (`sequential.py:137–155`): the executor drains
  the control inbox before picking the next task. A CANCEL at this
  point aborts the run cleanly.
- **Mid-task** (`sequential.py:468–575`): the executor races
  `adapter.invoke(...)` against the control channel via
  `asyncio.wait(FIRST_COMPLETED)`. On CANCEL, it calls
  `_cancel_invoke_task(invoke_task)` — `task.cancel()` + up to 5s
  grace, then warning-log + move on.

### 6.1 Cancellation cascade

Cancelling a task is **not** a leaf operation: the task's downstream
dependents have to end up cancelled too, otherwise the executor's
dependency check (`_pick_next_task`: "all predecessors COMPLETED")
silently orphans them.

`DefaultSteerer.mark_task_cancelled` (`steerer.py`) therefore does two
things on every legal cancel:

1. Transition ``task_id`` to `CANCELLED` and emit `TaskCancelled`.
2. BFS forward through ``session.plan.edges``; every reachable
   non-terminal task (PENDING / RUNNING / BLOCKED) is transitioned to
   `CANCELLED` with reason `"cascade from <task_id>"`, one
   `TaskCancelled` event per task. Terminal tasks (`COMPLETED` /
   `FAILED` / already-`CANCELLED`) are preserved verbatim and not
   traversed past — a diamond DAG does not double-cancel.

Both this cascade and the "unrecoverable drift" cascade share the
same downstream-CANCEL primitive —
`Steerer.cascade_cancel_downstream(session, task_id)` — so they
emit identical `TaskCancelled` event streams for the same downstream
set. The unrecoverable path differs only in the *initiator*
transition (FAILED instead of CANCELLED); the fan-out is identical.
See [STATE-MACHINE.md §"Cascade semantics on unrecoverable drift"](STATE-MACHINE.md#cascade-semantics-on-unrecoverable-drift)
and [STATE-MACHINE.md §"Shared downstream-CANCEL primitive"](STATE-MACHINE.md#shared-downstream-cancel-primitive).
Whether the cancel arrives via (a) a control-channel `CANCEL`,
(b) a `USER_STEER` whose refine returns no new plan (so the executor
preserves the CANCELLED current task without replacing it),
(c) an explicit `mark_task_cancelled` call, or
(d) a `mark_task_failed(..., recoverable=False)` fatal failure,
the cascade fans out through the same steerer primitive.

The `SequentialExecutor` additionally runs a **reachability audit**
at loop exit: if `_pick_next_task` returns None but some tasks are
still `PENDING`, each is transitioned to `CANCELLED` with reason
`"orphaned by plan revision failure"`, a CRITICAL `PLAN_DIVERGENCE`
drift is emitted, and the run ends with `success=False`. This is a
belt-and-suspenders catch for any cancellation path that fails to
cascade (e.g. a future control-channel variant that bypasses the
steerer).

### 6.2 Known cleanup gap: orphaned tool_call_ids

ADK's `run_async` may be mid-turn with an assistant message carrying
`tool_calls` when the cancel raises `CancelledError`. The matching
`tool_result` messages never get appended. The ADK session's
conversation history is now malformed. On the next invocation, ADK's
engine flags this as `"Missing tool results for tool_call_id(s): [...]"`
(observable in driver logs) and/or LiteLLM's
`heal_missing_tool_results` auto-inserts synthetic placeholders.

This is the structural reason a USER_STEER refine on a still-running
task can produce a truncated/malformed JSON response — the refine
LLM call inherits a poisoned session history. The early-break
quiescence fix (§3.1) addresses the happy path (agent reports
terminal cleanly); it does not address the ADK session repair
needed after a mid-tool-call cancel. See §7.

---

## 7. Known soundness gaps

Open issues against the design, ranked by impact. The current
implementation is sound for the common happy path; these are edge
cases or failure-mode ergonomics.

### 7.1 Multiple sources of truth for `_TERMINAL_TASK_STATUSES`

Three modules each define the same frozenset:

- `steerer.py:47–55`
- `adapters/_tool_invocation.py:30–37`
- `adapters/adk.py:239–244`

Each carries a comment warning maintainers to update all three in
lockstep. No automated check enforces it.

**Fix direction:** extract to a single module (e.g.
`goldfive.types._constants`) and import from it. Low effort, high
impact.

### 7.2 Plan validation at creation and revision (closed)

Closed by #100 and #105. `Plan.validate(for_revision=..., prior=...)`
runs at plan creation (`LLMPlanner.generate`) and on every plan
revision (`DefaultSteerer._apply_revision` and `LLMPlanner.refine`).
Checks duplicate task ids, edge referential integrity, acyclicity,
PENDING-only tasks on creation, and — when a `prior` plan is
supplied — terminal-task and terminal→terminal-edge preservation
(PLAN-LIFECYCLE §3.1–§3.2). Malformed plans are rejected with
`ValueError` before they are installed on the session.

### 7.3 Refine failure retry backoff (closed)

Closed by #99. `session.refine_outcomes: dict[(kind, task),
RefineOutcome]` tracks the per-drift-key outcome of the last
refine attempt this turn (goldfive#215 P2 — replaces the older
`refine_failure_counts` int counter). When `fail_count` crosses
`DefaultSteerer.REFINE_FAILURE_THRESHOLD` (default `2`), the
steerer marks the task FAILED and emits a CRITICAL
`REPEATED_FAILURE` drift. A successful refine writes
`state="succeeded"`, which short-circuits a follow-up same-(kind,
task) drift on the same turn. The whole table resets every
`run_started` boundary (`DefaultSteerer.reset_for_turn`).

### 7.4 Orphaned tool_call_ids on mid-invocation cancel (closed)

Closed by #101. `ADKAdapter` tracks pending `function_call_id`s
observed in the current `invoke()`'s event stream in
`self._pending_tool_call_ids` / `self._pending_tool_call_names`.
On mid-invocation cancel (or unexpected exception), the adapter
synthesises `{"cancelled": true}` function-response messages for
every pending id so the ADK session history stays well-formed and
the next turn does not see an orphaned `function_call`.

### 7.5 Cross-task ADK session history pollution

The ADK adapter reuses one `session_id` for the entire goldfive
run. Agent conversation history (including turns from failed/
cancelled tasks) carries forward into every subsequent invocation.
Without truncation, the agent's context window grows linearly with
the run and can see references to tasks that no longer exist in the
plan (after a USER_STEER delete-and-replan).

**Fix direction:** optional `truncate_conversation_on_revision`
flag on the executor/adapter that trims the ADK session history at
revision boundaries. Medium effort, medium impact.

### 7.6 Soft reject vs hard stop on terminal-task reporting (narrowed)

Layer 1 (§5.1) returns a structured error, but does not actively
prevent the agent from calling again. A misbehaving agent can
ignore the error and call `report_task_failed` 100 times; each call
is cheap (no handler, no drift) but still consumes an ADK turn. The
per-task cap (§5.3, 15 calls) bounds this for single-task spam; the
session-wide cap (§5.4, 50 calls) bounds the adversarial
varying-`task_id` pattern that defeats the per-task cap. Worst-case
wasted turns is now 49 across the whole session — acceptable.

**Fix direction:** none required; bounded by Layer 3 + Layer 4.

### 7.7 No per-task invocation cap

If a refine produces a `retry_<task>` task and the retry also
loops, another refine fires. The primary bound on blast radius is
now the per-lineage cap `max_retries_per_task_lineage` (default 3),
which refuses to invoke the adapter on the N+1st clone of any
lineage root. `max_task_invocations` (default `None` == unbounded)
can be set to a finite integer as an additional run-wide ceiling on
adapter invocations when a hard budget is desired.

**Fix direction:** implemented — `max_retries_per_task_lineage`
applies the per-lineage cap. After N invocations on the same
lineage, the next clone is refused and transitioned to FAILED in
place; downstream tasks then block via dependency cascade.

### 7.8 CANCELLED cascade to downstream PENDING tasks (closed)

Historically, `mark_task_cancelled` transitioned only the target
task and emitted a single `TaskCancelled` event. This implicitly
assumed CANCELLED predecessors were non-terminal — but
`_pick_next_task` only picks tasks whose predecessors are all
`COMPLETED`, so a CANCELLED predecessor silently orphaned every
descendant. Combined with a `USER_STEER` refine that returned no new
plan, the sequential executor would run only the independent
branches and exit with `success=True`, leaving the dependent branch
stuck in `PENDING`. The "make a presentation about waffles" regression
run (`research_history` cancelled by STEER; refine produced empty
JSON; 7 downstream tasks left PENDING; executor reported success)
was the in-the-wild trigger.

**Fix (see §6.1):** `mark_task_cancelled` now BFS-cancels all
downstream non-terminal tasks, and `SequentialExecutor` runs a
reachability audit at loop exit that flips any remaining PENDING
tasks to CANCELLED, emits a CRITICAL `PLAN_DIVERGENCE` drift, and
fails the run. Closed by `fix/cancel-cascade`.

---

## Appendix: event emission guarantees

Every terminal transition in the state machine emits exactly one
event. See [STATE-MACHINE.md §"Invariant 3"](STATE-MACHINE.md#invariant-3--transitions-emit-exactly-one-event)
for the full table.

Every drift triggers exactly one `DriftDetected` event, regardless
of severity. A refine-failed follow-up (§4.3) emits a *second*
`DriftDetected` of severity CRITICAL — sinks should treat this as
"the prior drift was not handled" rather than a new distinct drift.

A plan revision emits `PlanRevised(old_plan_id, new_plan_id,
revision_index, kind, severity, reason)` after
`_apply_revision` completes. Sinks that cache plan state should
re-read `session.plan` on every `PlanRevised`.

---

## Appendix: orchestration state namespace (goldfive#152)

`Session.state` is a dict goldfive stamps with keys under the
`goldfive.*` prefix so downstream consumers (prompt templates,
refine paths, `GoldfivePlanner`, drift detectors) can read
orchestration context without walking `session.plan` + the drift
stream. This is **not** the same surface as the ADK adapter's
`session.state` (which is the ADK runtime's own dict; see
[`goldfive/adapters/_adk_state_protocol.py`](../../goldfive/adapters/_adk_state_protocol.py)).
The orchestration namespace is framework-agnostic — every adapter
goes through `Session.state`, not just ADK.

Keys:

| Key | Owner | Lifecycle |
|---|---|---|
| `goldfive.current_plan_id` | `Runner` on plan submit, `DefaultSteerer._apply_revision` on revision | Set when a plan is installed; persists across the run |
| `goldfive.current_task_id`, `goldfive.current_task_title` | `PlanReconciler` on before/after_agent; `DefaultSteerer.mark_task_*` on legacy transitions | Set on RUNNING; cleared on terminal transition of the same task and at run end |
| `goldfive.goals_summary` | `Runner` after goal derivation; `DefaultSteerer._apply_user_steer_state` after USER_STEER | Formatted `"- [id] summary\n..."` block; refreshed whenever `session.goals` changes |
| `goldfive.active_steer.body`, `goldfive.active_steer.at_turn` | `DefaultSteerer._apply_user_steer_state` on USER_STEER drift | `body` is the raw steer text; `at_turn` is the session sequence value at fire time. Cleared at run end |
| `goldfive.cancelled_function_call_ids` | `ADKAdapter._heal_pending_tool_calls` | Append-only list of function_call ids that were healed mid-invocation. De-duplicated within the run |

Ownership rules:

1. **Only goldfive writes.** Adapters and agents do not write into
   this namespace — it's orchestration-owned. Agent-side state
   lives on the ADK `session.state` dict (see `_adk_state_protocol`)
   or equivalent per-adapter surface.
2. **Writers go through `goldfive.orchestration_state`.** The
   helper module enforces the `goldfive.*` prefix and exposes
   typed writers / readers so consumers don't hand-roll key
   strings.
3. **Steering is always active (no cooldown).** `active_steer.*`
   is a durable read-back of the most recent USER_STEER, not a
   cooldown window. The `at_turn` field lets consumers reason
   about "is this a fresh steer or a stale one?" against the
   session's monotonic sequence counter; they do the comparison
   themselves.

USER_STEER flow (`DefaultSteerer._apply_user_steer_state`):

1. Stamp `active_steer.body` / `active_steer.at_turn`.
2. Call `planner.synthesize_goal_from_steer(body)`. The method
   returns `(Goal, mode)` where `mode` is `"append"` or
   `"replace"`. Falls back to a passthrough `Goal(id="steer",
   summary=body)` when the planner doesn't implement the hook.
3. Mutate `session.goals` accordingly — `append` adds, `replace`
   clears and sets the sole goal.
4. Refresh `goldfive.goals_summary`.
5. The steerer's normal drift flow (`_handle_drift` continuation)
   then calls `planner.refine` with `list(session.goals)` — which
   now contains the synthesized steer goal — so the refined plan
   sees the pivot as a first-class goal, not just a drift detail
   string.

Steer restart framing (`SequentialExecutor._compose_steer_restart_message`):

When USER_STEER fires mid-invocation under overlay mode, the
executor cancels the in-flight invocation (goldfive#149) and
restarts with the steer body as user input. The body is wrapped in
a goldfive-authored override header:

```
[USER STEERING CONTROL — supersedes prior task context]

{steer_body}

Notes:
- Prior research, partial work, or planned tasks from the pre-steer
  conversation are superseded unless this message explicitly
  references them.
- Proceed with the new direction. Do not continue prior work unless
  doing so directly serves this steer.
```

Clean framing without wiping conversation history — the LLM can
still see earlier turns, but the framing makes it unambiguous that
the steer is the new north star.
