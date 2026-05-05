# The control channel as actor mailbox

**Status.** Top-level design doc for the actor-model architecture
established by goldfive#245-#248 (Phases 1-4 of the structural fix).
Written as Phase 5 of that fix; merges before Phases 1-3 so the
architectural commitment is visible early. Phases 1-3's own design
docs (DRIFT.md, PLAN-LIFECYCLE.md, CANCELLATION-CONTRACT.md) cross-
link back to this one.

This document describes the **target architecture** — the shape the
codebase has after Phases 1-4 land. Where a phase has not yet
shipped, the prose names the phase that delivers the change and the
issue number that tracks it. A reader on `main` after all four phases
have merged will find every claim grounded in code; a reader before
that should treat the unlanded sections as a contract the implementer
phases must honor.

Related design docs (each updated by one or more of Phases 1-4):

- [PLAN-LIFECYCLE.md](PLAN-LIFECYCLE.md) — immutable `Plan` /
  `Task`, `supersedes` invariant, plan validation. Updated by
  Phases 3 and 4.
- [DRIFT.md](DRIFT.md) — verdict freshness, dispatch routing,
  drift event shape (`observed_revision_index`). Updated by
  Phases 1 and 2.
- [CANCELLATION-CONTRACT.md](CANCELLATION-CONTRACT.md) — channel-
  mediated cancel, stash-bypass audit. Updated by Phase 2.
- [CONTROL.md](CONTROL.md) — the `ControlChannel` primitive
  (sender / receiver semantics, ack contract, lifecycle). Phase 2
  re-uses this primitive as the actor mailbox; the mechanics in
  CONTROL.md are unchanged.

---

## §1. Motivation — two false-positive cascades

Two end-to-end runs in 2026-04 surfaced a pattern that prior fixes
kept failing to fully close. Their symptom was the same: a
**recoverable** drift fired correctly the first time, the planner
revised the plan, work resumed on the new revision — and then a
second drift fired against **state from the prior revision** and
cascaded the run into a cancellation. The cascade looked like a
chain of judgment errors but was structural: the second drift was a
true observation about a state that no longer existed.

### §1.1 The brussels-sprouts cascade

Earlier of the two. A multi-task culinary plan was steered by the
operator at `t=0:34` ("brussels sprouts, not asparagus"). The
planner ran `_refine_user_steer`, dropped pending tasks, emitted a
new plan with `revision_index=2`, and the executor restarted the
overlay invocation with a `[USER STEERING CONTROL — supersedes
prior task context]` framed message. So far, correct.

What followed exposed the gap. The reasoning judge had been
launched at `t=0:30` against the pre-revision state to classify
whether the agent had hit `LOOPING_REASONING`. By the time it
returned a verdict at `t=0:36`, `session.plan` was already the
post-revision plan. The judge's verdict carried no version pin, so
it landed in `_handle_drift` as a fresh observation, fired a
`LOOPING_REASONING/WARNING` drift against a `current_task_id` that
had been dropped by the steer, refine ran a second time, and the
cascade unwound from there.

### §1.2 The tomato cascade — fully traced

Days later. A two-stage research-then-draft plan asked an HTML-
drafting coordinator about tomato cultivation. Timeline:

```
t=0:12  OFF_TOPIC/WARNING fires  — TRUE positive
        the agent had wandered into companion-planting trivia.
        observed: revision_index=1 (no pin recorded)
        Steerer: refine → revision_index=2 (drop wandering branch,
                 add focused-research task).

t=0:48  Coordinator begins drafting HTML for the focused page.
        Output buffer fills with valid markup; the agent is
        making progress.

t=1:01  OFF_TOPIC/WARNING fires  — FALSE positive
        the OFF_TOPIC judge had been launched at t=0:55 against
        rev=1 state, returned a verdict at t=1:01 with no pin.
        Symptom in the detail: complaint about "wandering
        coordinator" — but the wandering had been corrected at
        revision_index=2 already.
        Steerer: refine → revision_index=3 (cancels the HTML
                 drafting mid-render, replaces with "redo from
                 a different angle").

t=1:13  GOAL_DRIFT/WARNING fires — FALSE positive
        the goal-drift judge had snapshotted the *intermediate
        HTML* during the t=1:01 cancellation, then the planner's
        revision_index=3 plan has different goal coverage. The
        snapshot is now of a phantom state — neither rev=2 (which
        produced it) nor rev=3 (which discarded it).
        Steerer: refine → REFINE_VALIDATION_FAILED (the LLM
                 cannot produce a coherent revision from an
                 incoherent snapshot).

t=1:20  Cascade. PAUSE_ESCALATE → operator unblock → CANCEL.
        Run terminated at outcome.success=False.
```

The coordinator was producing correct output throughout the run.
None of the false-positive verdicts were wrong about what they
observed; they were wrong because **what they observed had been
superseded by the time the verdict landed**.

### §1.3 Why these were structural, not isolated bugs

A failure where stale verdicts cascade across plan revisions cannot
be patched per-judge. The detector is correct: the agent *was*
off-topic in the buffer the judge inspected. The patch we kept
reaching for — "tighten the prompt", "add a heuristic", "change the
threshold" — addresses the wrong layer. The right layer is the one
where the verdict meets `session.plan`: a fresh verdict against a
stale snapshot of state must be rejected before refine runs, and
the only mechanism that knows whether state has moved is the one
that owns `session.plan`'s lifecycle.

The five issues that conspired to permit these cascades are
structural in the same way. Each is a place where the system
*pretends* to have a coherent state when in fact two parties hold
divergent views of it. They are enumerated in §2; the architectural
fix is in §3 / §4.

---

## §2. The five structural issues

Each issue below is a place where a goldfive-internal write happens
outside the channel, or a read happens against a snapshot that does
not match the writer's view. The phase number on each row is the
phase that closes that issue.

### §2.1 `pending_corrective_message` is write-only for goldfive-authored drifts

`session.pending_corrective_message` (`goldfive/types.py:593`) was
introduced for the Level 3 (`CANCEL_REINVOKE`) intervention path.
The steerer composes a corrective user message and stashes it on
the session (`goldfive/steerer.py:1643`). The intent: the Runner's
overlay loop reads it, cancels the in-flight invocation, and re-
invokes with the composed text as the new prompt.

The slot is a write-only slot for **goldfive-authored** drifts. The
comment in `steerer.py:1639-1641` is explicit: *"Until #141 lands,
this slot is inert -- nobody reads it."* `USER_STEER`, by contrast,
takes a different path: the executor's `_apply_steer` cancels the
invocation directly and the steer body lands in the agent prompt
through `_compose_steer_restart_message` (`sequential.py:~1243`).

The result is **path duality**. User-initiated steers and goldfive-
authored corrective steers travel different routes, with different
ordering relative to plan revision and different cancellation
mechanics. A drift detector that wants to "cancel the in-flight
work and apply a corrective" must hope that whoever wires up the
read side has not changed since the detector was written.

**Closed by Phase 2 (#246):** plan revisions and pause-escalate are
routed through the `ControlChannel` as goldfive-internal control
messages. A `goldfive_steer` `ControlMessage` carries the corrective
body in its payload, the channel processor cancels the in-flight
invocation as part of its dispatch, and the corrective lands in the
agent prompt the same way `USER_STEER` does. Path duality
collapses.

### §2.2 Drift judges fire against post-revision state without version pinning

`DriftEvent` (`goldfive/types.py::DriftEvent`) carries `kind`,
`severity`, `detail`, `current_task_id`, `current_agent_id`, and
`raw`. It does **not** carry a revision pin. A judge launched
against `revision_index=N` that returns a verdict after
`session.plan.revision_index` has advanced to N+1 cannot be
distinguished from a judge that observed N+1 directly. The
`_handle_drift` pipeline routes both as fresh observations.

This is the direct mechanism behind the §1.1 / §1.2 cascades. It is
worse than it looks because judges run asynchronously: the
reasoning judge, the goal-drift judge, the off-topic judge, and the
loop classifier are all dispatched as background tasks, and any of
them may land arbitrarily late relative to plan revisions.

Where the gap lived pre-fix:

- `goldfive/types.py::DriftEvent` (~line 24-32) — no
  `observed_revision_index` field.
- `goldfive/steerer.py::_handle_drift` (~line 1500+) — no
  freshness check before refine.
- `goldfive/steerer.py::_run_judge_background` — emits drift
  events without recording the revision they observed.

**Closed by Phase 1 (#245):** `DriftEvent.observed_revision_index`
is added; judges record `session.plan.revision_index` at dispatch
time; `_handle_drift` re-reads `session.plan.revision_index` after
the LLM call and rejects the verdict if it advanced. Stale verdicts
are emitted as `INFO`-severity *observational* drifts (so sinks see
the judge fired) but do not route into refine.

### §2.3 Plan validator and refine prompt accept ill-shaped corrective insertions

When `Planner.refine` inserts a corrective task — for example, in
response to a recoverable drift on `t_research`, the planner adds
`t_research_retry` and edges from `t_research` (now FAILED) to it —
nothing in the validator enforces that the new task **declares
which task it supersedes**. The planner prompt says "consider"
adding a `supersedes` field; the validator never checks it; the
refine retry loop can produce a plan where two tasks run in
parallel claim to address the same drift.

The §1.2 cascade hit this at `revision_index=3`: the LLM, faced
with a corrective on already-corrective work, produced a task that
claimed to be a new attempt at the original goal but did not name
the task it replaced. The plan was structurally valid (no cycles,
all edges point at known tasks) yet semantically incoherent.

Pre-fix locations:

- `goldfive/types.py::Task` — no `supersedes` field; planner-
  semantic only, never machine-checked.
- `goldfive/types.py::Plan.validate` — no supersedes invariant in
  the `for_revision=True` branch.
- `goldfive/planner.py` refine prompts (~`_refine_user_steer`,
  `_refine_looping_tool_call`) — do not require supersedes.

**Closed by Phase 4 (#248):** `Task.supersedes: tuple[str, ...]`
is added to the frozen `Task` dataclass (Phase 3 made it
immutable). `Plan.validate(for_revision=True)` rejects revisions
where any non-terminal task references a `supersedes` id that does
not appear in the plan, and rejects revisions where two non-
terminal tasks claim to supersede the same id. The refine prompt
is updated to require `supersedes` on every PENDING task that
replaces a FAILED predecessor.

### §2.4 `paused_for_human_intervention` is unread by the overlay executor

The Level 4 (`PAUSE_ESCALATE`) intervention sets
`session.paused_for_human_intervention = True`
(`goldfive/steerer.py:1870`) and emits
`HUMAN_INTERVENTION_REQUIRED`. The Runner's outer loop is supposed
to honor the flag — but the overlay executor (`SequentialExecutor`
with `overlay_mode=True`, the default since #148) has its own loop
that drains controls and races the adapter. Pre-fix, that loop
read the flag in `executors/sequential.py:984-991` and `:1019` —
but the *overlay* path and the *per-task* path read it at
different times relative to the in-flight invocation, and the
overlay path re-cleared the flag in
`executors/_control.py:224,236` on RESUME without coordinating
with the steerer.

The result is a subtle leak: a Level 4 escalation produced by the
steerer between iterations of the overlay loop could be lost if a
RESUME landed before the overlay's next drain. The §1.2 t=1:20
cascade rode this gap — the operator's first PAUSE-clearing
intervention was discarded because the steerer's
`paused_for_human_intervention=True` had not yet reached the
overlay's read site.

Pre-fix:

- Read sites: `goldfive/executors/sequential.py:984, 1019` (per-
  task and overlay-restart points).
- Write sites: `goldfive/steerer.py:1870`,
  `goldfive/executors/_control.py:224, 236`,
  `goldfive/executors/parallel.py:800`.
- No serialization between writers and readers.

**Closed by Phase 2 (#246):** the pause-escalate transition
becomes a `ControlMessage` (kind `goldfive_pause_escalate`) on the
same channel as user-initiated PAUSE. The channel processor is the
single reader; the steerer no longer writes the flag directly.
RESUME on the same channel clears it. Reader / writer race is
gone.

### §2.5 No plan-revision generation barrier

Even with §2.1-§2.4 closed, a window remains where the writer side
of `session.plan = revised_plan` (`steerer.py:1633`) and a reader
on the executor / adapter side could observe the assignment in
inconsistent order relative to other state. Python guarantees the
attribute write is atomic (the GIL), but `session.plan.tasks`,
`session.completed_results`, and `session.task_progress` are not
guaranteed to be updated atomically with respect to each other.

Pre-fix, this manifested as transient inconsistencies in
observability: a `PlanRevised` event could land on the sink
stream before the per-task status updates implied by the
revision had been applied to `session.task_progress`.

**Closed by Phases 2-3 jointly (#246, #247):** Phase 3 makes
`Plan` and `Task` immutable; mutation is replacement of the whole
object. Phase 2 routes the `session.plan` swap *only* through the
channel processor, which holds the mailbox lock for the duration
of the swap and any companion updates. The result is an **epoch
boundary**: every reader either sees revision N's full state or
revision N+1's full state, never a mix.

---

## §3. The actor-model commitment

After Phases 1-4 land, goldfive's session is an actor in the
classical sense: a single mailbox in front of a single state
container, processed serially by one consumer.

The four invariants below are what "actor model" means for
goldfive specifically. They are stronger claims than "we use
async / await": they are about who is allowed to observe state
and who is allowed to mutate it.

### §3.1 Invariant 1 — the channel is the only mailbox

The session is an actor; `ControlChannel` is its only mailbox.

Every state-mutating event — user-initiated PAUSE / RESUME /
CANCEL / STEER / REWIND_TO / APPROVE / REJECT, **and** every
goldfive-internal drift response that mutates plan or task status
— enters the actor through `channel.send(ControlMessage(...))`.
There is no other entry point. Direct calls into
`steerer.observe`, `steerer._handle_drift`, or
`session.plan = ...` from outside the channel processor are
prohibited.

The channel itself has not changed. CONTROL.md §2 is still the
authoritative reference for `ControlChannel.send` / `receive` /
`ack`. What changed is who calls `send`: pre-fix, only the
external bridge did; post-fix, goldfive-internal drift response
also calls `send` to deliver `goldfive_steer` /
`goldfive_pause_escalate` / `goldfive_install_revision` messages
to its own actor.

### §3.2 Invariant 2 — state mutations only inside the channel processor

State mutations to `session.plan` and `task.status` happen only
inside the channel processor.

Pre-fix, `session.plan` could be assigned from any of:
`steerer._apply_revision`, the executor's pre-task install path,
or `_apply_steer`. Post-fix, all three route through a single
processor — `executors/_control.py::dispatch_control` — which
serializes incoming messages and applies their effects.

`task.status` is mediated through helper functions
(`with_task_status(plan, task_id, status)` returning a new `Plan`)
because `Task` is now frozen (Invariant 3); the helper is only
*called* from inside the channel processor.

The practical consequence for code reviewers: a diff that contains
`task.status = ...` outside `goldfive/executors/_control.py` is
suspect. Phase 4's `STATE-OWNERSHIP-CONTRACT.md` audit catalog
will be extended to include channel-only mutation sites; new
violations are caught by the same `_state_audit` runtime tripwire
as the ADK-state-ownership rules.

### §3.3 Invariant 3 — `Plan` and `Task` are immutable

`Plan` and `Task` become frozen dataclasses. Mutations construct
new instances; `dataclasses.replace` is the canonical operation.

This is what makes the channel processor's serial discipline
actually buy something. If `Task` were mutable, two parties
holding references to "the same task" could disagree about its
status — even if both observed the assignment correctly, they
could observe its *side effects* in different orders. With frozen
`Task`, the only way to "change" a task is to construct a new
`Plan` with a new `Task`; the swap is the channel processor's
atomic step (Invariant 2); every reader either sees the old `Plan`
or the new one, never a mix.

This invariant has migration cost. Pre-fix code in `steerer.py`,
`executors/sequential.py`, and `reporting.py` calls `task.status =
TaskStatus.RUNNING` and similar in a dozen places. Phase 3
migrates each call site to `with_task_status(...)`; Phase 4 makes
`Task.__setattr__` raise.

### §3.4 Invariant 4 — every drift verdict carries `observed_revision_index`

Every `DriftEvent` carries `observed_revision_index`. Stale
verdicts are rejected at dispatch.

The mechanism: `_handle_drift` reads
`session.plan.revision_index` *after* the LLM call returns, and
compares against the verdict's `observed_revision_index`. If the
plan has advanced, the verdict is logged as observational (an
`INFO`-severity drift carrying both indices in `detail` so sinks
see the freshness gap) and dropped.

This is what closes the §1.1 / §1.2 cascades: a judge that
launches against revision N and returns its verdict after
revision N+1 has been installed cannot route into refine. The
verdict was correct about what it saw; what it saw is no longer
the system state; the next round of judges (launched against N+1)
will fire again if the post-revision state is also off-topic.

---

## §4. The fix architecture — cancel-and-restart through the channel

The pattern that USER_STEER already had — and that goldfive-
authored corrections did not — is the actor-model fix. It is a
single primitive applied uniformly: **the channel processor is the
epoch boundary; every plan revision is a cancel-and-restart
through it**.

### §4.1 The diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Drift detected                                                   │
│   judge.run(snapshot, observed_revision_index=N)                 │
│   → DriftEvent(kind=..., severity=...,                           │
│                observed_revision_index=N)                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼  channel.send(...)
┌─────────────────────────────────────────────────────────────────┐
│ ControlMessage(kind="goldfive_steer",                            │
│                payload={"drift": <event>,                        │
│                         "corrective_body": "..."})               │
│   onto session._channel._inbox                                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Channel processor (executors/_control.py::dispatch_control)      │
│                                                                  │
│  1. Re-read session.plan.revision_index (call it M).             │
│     If M > N (drift's observed_revision_index):                  │
│       emit observational DriftDetected(INFO,                      │
│         detail="stale verdict: observed=N current=M")            │
│       ack SUCCESS, return — the actor mailbox moves on.          │
│                                                                  │
│  2. Cancel in-flight invocation:                                  │
│     adapter.cancel_current_invocation(grace_seconds=5)            │
│                                                                  │
│  3. Run planner.refine(plan, drift) → revised Plan.               │
│     If validation fails (Phase 4 supersedes invariant included): │
│       register refine failure, route to ladder Level 4.           │
│                                                                  │
│  4. Apply revision (atomic, inside the processor):                │
│     session.plan = revised        # frozen Plan, full replace    │
│     emit PlanRevised(M+1)                                         │
│                                                                  │
│  5. Restart: invoke_passthrough with corrective body in the       │
│     agent prompt (the same path USER_STEER already uses).        │
└─────────────────────────────────────────────────────────────────┘
```

### §4.2 Why this is a fix, not a workaround

The fix is not "add another check at the dispatch site." Every
prior attempt has done that and the cascades kept reappearing. The
fix is to commit to a discipline: the only way for any party
(judge, detector, user, executor) to change `session.plan` is to
send a message to the actor and let the actor decide, with full
knowledge of the current epoch, whether the message is still live.

USER_STEER worked because external clicks naturally route through
the channel — the bridge constructs a `ControlMessage` and calls
`channel.send`. Goldfive-authored drift responses have always had
the option of routing through the channel; Phase 2 makes that the
*only* option for state-mutating responses.

### §4.3 What "epoch boundary" means in code

An epoch boundary is the moment in
`executors/_control.py::dispatch_control` between step 4 (apply
revision) and step 5 (restart invocation). Before that moment,
every reader of `session.plan` sees the old plan; after that
moment, every reader sees the new plan. Because the processor is
single-consumer and `Plan` is frozen, no reader can observe a
partially-applied transition.

The freshness check in step 1 is the epoch-boundary check: a
verdict carrying `observed_revision_index=N` against
`session.plan.revision_index=M` where `M > N` is a verdict from a
prior epoch; the actor honors it as observational and discards it.

### §4.4 Tracing USER_STEER as the model

USER_STEER's existing path (CONTROL.md §3) is the worked example.
Walking through it with §4.1's diagram:

1. **Drift detected.** Operator clicks [Steer] in the harmonograf
   UI; bridge constructs `ControlMessage(kind=STEER, payload=...)`.
   The operator does not carry an `observed_revision_index` —
   they observed visually, on whatever revision was rendered. The
   bridge stamps no version pin.

2. **Onto the channel.** `channel.send(msg)` lands the message
   on `_inbox`.

3. **Channel processor.** `dispatch_control` receives. There is
   no freshness check for USER_STEER (operator intent is sticky;
   we honor it regardless of plan-revision drift); the processor
   proceeds to step 2.

4. **Cancel in-flight.** `adapter.cancel_current_invocation` with
   a 5-second grace.

5. **Refine.** `_refine_user_steer` drops pending tasks,
   preserves terminals, regenerates a sub-DAG.

6. **Apply revision.** `session.plan = revised_plan`.

7. **Restart.** `invoke_passthrough` with
   `_compose_steer_restart_message(...)` framing the corrective
   body.

Goldfive-authored drift response post-Phase-2 follows steps 1, 2,
3 (with the freshness check active), 4, 5, 6, 7. The only
difference is who constructed the `ControlMessage`: external
bridge vs. internal drift judge. The mailbox does not care.

### §4.5 Cancellation mechanics — see CANCELLATION-CONTRACT.md

Step 4's `adapter.cancel_current_invocation` has its own contract:
the adapter's coroutine must propagate `asyncio.CancelledError`
without swallowing it through `except Exception:`, and any owner
of post-cancel state-stash duties must use `try / finally` or the
`except BaseException: stash; raise` pattern. The full audit and
the runtime tripwire that prevents regressions live in
[CANCELLATION-CONTRACT.md](CANCELLATION-CONTRACT.md). Phase 2
extends that catalog with the channel-processor sites added by
this refactor.

---

## §5. What the actor model does NOT do

The actor model is a structural fix. It is precise about *which*
class of failures it eliminates; it does not eliminate the
underlying difficulty of building an LLM-driven framework. Being
honest about boundaries makes it easier to spot the next class of
bug.

### §5.1 It does not fix planner LLM hallucinations

If `_refine_user_steer` returns a plan that hallucinates an agent
name not in the registry, the actor model does not catch that.
Plan validation does (via `available_agents` enforcement,
`PLAN-LIFECYCLE.md` §5). Plan validation is upstream of the
channel processor's "apply revision" step; a validation failure
short-circuits to ladder Level 4, not to step 6 of §4.1.

The actor model ensures that *whatever* the planner produces, that
plan applies to a coherent epoch. It does not improve the
planner.

### §5.2 It does not improve verdict quality

If the off-topic judge has a 5% false-positive rate, the actor
model does not reduce it to 0%. It reduces the **cascade
amplification** of false positives by ensuring that a stale
false-positive verdict is dropped at the dispatch boundary.

A fresh false positive — a judge that observes the post-revision
state correctly but classifies it incorrectly — still routes
through `_handle_drift` and refine. The fix for *that* is judge
prompt strengthening, dedicated classifiers, or eliminating the
need for the judge entirely (see the
`feedback_no_regex_heuristics` lesson). The actor model is
orthogonal.

### §5.3 It does not eliminate judges

Judges are still useful. The §1.1 / §1.2 cascades had **true
positive** verdicts at their roots — the agent really did go
off-topic at `t=0:12` in the tomato run, and the steer that fired
was correct. The structural fix is about *false positives chained
to true positives via stale state*, not about removing the
detection layer.

A run with no judges and no drift detection is a run that cannot
self-correct. The actor model lets self-correction be reliable
where it was previously fragile.

### §5.4 What it does

It ensures that whatever the system decides, that decision applies
to a coherent state. The actor's mailbox is the only entry point;
the actor's processor is the only mutator; the actor's frozen
state objects make every read consistent with some specific epoch.

The cascades from §1 cannot recur in this architecture — not
because every component became smarter, but because the structure
that allowed stale verdicts to cascade is gone.

---

## §6. References

### §6.1 Sibling design docs

- [PLAN-LIFECYCLE.md](PLAN-LIFECYCLE.md) — immutable `Plan` and
  `Task`, `supersedes` invariant, the six structural validation
  rules. Updated by **Phases 3 and 4** to add the immutability
  contract and the supersedes invariant. The §3 refinement
  contract there is the per-revision view of what the actor model
  enforces per epoch.

- [DRIFT.md](DRIFT.md) — the `DriftEvent` shape including
  `observed_revision_index`, the freshness rule at the dispatch
  site, the intervention ladder. Updated by **Phases 1 and 2** to
  document the freshness check and the goldfive-authored drift
  routing.

- [CANCELLATION-CONTRACT.md](CANCELLATION-CONTRACT.md) — channel-
  mediated cancel mechanics and the audit catalog. Updated by
  **Phase 2** to add the channel-processor cancellation sites.

- [CONTROL.md](CONTROL.md) — the underlying `ControlChannel`
  primitive. **Not changed by Phases 1-4** (the primitive itself
  is stable); the only delta is who calls `channel.send`. CONTROL.md
  §3's diagram now also models internal callers, but the contract
  is identical.

- [STATE-OWNERSHIP-CONTRACT.md](STATE-OWNERSHIP-CONTRACT.md) —
  parent of the channel-only mutation rule. The actor-model
  invariants in §3 are an extension of the state-ownership rules
  to goldfive-internal state (this doc) layered on top of the
  ADK-state rules already in place there.

### §6.2 The five sibling tasks

Tasks #242-#244 (precursor narrow fixes) addressed individual
symptoms before the structural fix was scoped:

- **#242** — close iter-11D race between cancel-requested and
  terminated mark in the steerer. Narrow per-symptom fix.
- **#243** — drain background drift / judge tasks at run-boundary,
  not only `Runner.close`. Narrow per-symptom fix.
- **#244** — thread `available_agents` into the reasoning-judge
  prompt so it recognizes sub-agent delegation. Reduces a class
  of false-positive judgments but does not address the cascade.

The structural phases:

- **#245 — Phase 1.** `DriftEvent.observed_revision_index` +
  dispatch-time freshness gate + post-LLM re-read in judges.
  Closes §2.2.
- **#246 — Phase 2.** Route plan revisions and pause-escalate
  through `ControlChannel`. Collapse path duality with
  USER_STEER. Closes §2.1, §2.4, and the writer half of §2.5.
- **#247 — Phase 3.** `Plan` and `Task` become frozen
  dataclasses; mutations via `dataclasses.replace`; `session.plan`
  swap only inside the channel processor. Closes the reader half
  of §2.5 and is a prerequisite for §3.3.
- **#248 — Phase 4.** `Task.supersedes: tuple[str, ...]` +
  `Plan.validate` invariant + refine prompt rule. Closes §2.3.

This document — **#249 — Phase 5** — is the architectural
commitment that ties the four implementation phases together.

### §6.3 Code references (post-Phases-1-4)

For readers who want to walk the code after all four phases have
shipped:

- `goldfive/control.py::ControlChannel` — the mailbox primitive
  (unchanged from CONTROL.md §2).
- `goldfive/executors/_control.py::dispatch_control` — the
  channel processor; the actor's serialization point.
- `goldfive/types.py::DriftEvent.observed_revision_index` —
  Phase 1 freshness pin.
- `goldfive/types.py::Plan, Task` — frozen dataclasses,
  Phase 3.
- `goldfive/types.py::Task.supersedes` — Phase 4 supersedes
  field.
- `goldfive/types.py::Plan.validate(for_revision=True, prior=...)` —
  Phase 4 supersedes invariant; Phases 1-3 do not change this
  function, but the invariants it enforces grow.
- `goldfive/steerer.py::DefaultSteerer._handle_drift` — sender
  side after Phase 2; constructs `ControlMessage(kind=
  "goldfive_steer")` and `channel.send`s it instead of mutating
  `session.plan` directly.
- `goldfive/steerer.py::DefaultSteerer._apply_revision` —
  *deprecated as a public mutation site*. Post-Phase-2, called
  only from inside `dispatch_control`.

---

## §7. Migration story

For code reviewers and future contributors. This section is the
short version of "how to write code that does not break the actor
model."

### §7.1 If you are tempted to write `task.status = ...`

You are outside the channel and frozen-`Task` will refuse. The
correct pattern is:

```python
from goldfive.types import with_task_status, TaskStatus

# WRONG (post-Phase-3): raises FrozenInstanceError
task.status = TaskStatus.COMPLETED

# RIGHT: returns a new Plan with the task replaced
new_plan = with_task_status(plan, task_id, TaskStatus.COMPLETED)

# Then dispatch through the channel:
await channel.send(ControlMessage(
    kind="goldfive_install_revision",
    payload={"plan": new_plan, "reason": "task completed"},
))
```

The `with_task_status` helper lives in `goldfive/types.py` and is
the only blessed way to construct a derived `Plan`. It validates
that the target task exists and that the target status is a legal
transition from the current one.

### §7.2 If you are tempted to write `session.plan = ...`

Same answer: outside the channel, this will not work post-Phase-2.
The channel processor is the only writer. From inside a steerer or
executor method, send a message:

```python
await session._channel.send(ControlMessage(
    kind="goldfive_install_revision",
    payload={"plan": revised_plan, "drift": drift},
))
```

The processor's handler for `goldfive_install_revision` performs
the validation, the `session.plan = ...` assignment, the
`PlanRevised` emission, and the corrective restart in one
atomic step.

### §7.3 If you are writing a new drift judge

Stamp `observed_revision_index` at dispatch time:

```python
async def my_judge_background(session, snapshot):
    pinned = session.plan.revision_index   # capture early
    verdict = await self._call_llm(...)
    if verdict.indicates_drift:
        await session._channel.send(ControlMessage(
            kind="goldfive_steer",
            payload={
                "drift": DriftEvent(
                    kind=DriftKind.OFF_TOPIC,
                    severity=DriftSeverity.WARNING,
                    detail=verdict.detail,
                    observed_revision_index=pinned,    # required
                ),
                "corrective_body": verdict.corrective,
            },
        ))
```

The freshness check at the dispatch site does the rest. A judge
that forgets to stamp the pin will see its drifts treated as
freshly observed against the current revision — which is exactly
the bug the actor model is structured to prevent.

### §7.4 If you are writing a new control kind

CONTROL.md §7 is still the authoritative reference. The actor
model does not change how custom kinds are added; it only adds a
discipline that goldfive-internal kinds use the same mailbox.

Reserve the `goldfive_` prefix for internal kinds (e.g.
`goldfive_steer`, `goldfive_pause_escalate`,
`goldfive_install_revision`). External / user-facing kinds use
the existing `ControlKind` enum (`STEER`, `PAUSE`, etc.). The
prefix discipline lets a sink filter "operator-initiated"
control traffic from "self-correction" control traffic without
introspecting payloads.

### §7.5 The runtime tripwire

`STATE-OWNERSHIP-CONTRACT.md` ships a runtime tripwire
(`goldfive/_state_audit.py`) that catches direct ADK-session
mutations. Phase 2 extends the same module with a
`PlanMutationViolation` and a `TaskStatusMutationViolation` that
fire when `session.plan` or `task.status` is assigned outside
`executors/_control.py::dispatch_control`. Default-off, gated on
`GOLDFIVE_STRICT_STATE_OWNERSHIP=1`; tests run with strict mode
on by default, same as the existing audits.

A diff that fails the tripwire in CI is a sign the contributor
did not read this section. Reviewers: when you see a
`PlanMutationViolation` test failure, point the contributor at
§7.1 / §7.2.

---

## §8. Forward look

The actor model is a foundation, not a finish line. Some directions
that become tractable once Phases 1-4 ship:

- **Sub-second refresh of the harmonograf timeline.** Sinks
  observe `PlanRevised` events; with epoch boundaries guaranteed,
  a sink can render "what changed at revision N+1" without
  defensively re-fetching state.
- **Deterministic replay.** With every state mutation routed
  through the channel as a typed message, the message sequence is
  the run's reduced form. Replay = re-apply messages to a fresh
  session with a recorded LLM oracle.
- **Multi-actor topologies.** A future feature like "two
  independent steerers running concurrently against different
  goal subsets" becomes a question of mailbox routing, not state-
  ownership negotiation.

These are not promised. They are listed to convey that the
discipline of §3 / §4 is not a tax — it is an enabler. The cost is
a one-time migration of in-tree mutation sites; the benefit is
that future structural work has a foundation to stand on.

---

## §9. See also

- [CONTROL.md](CONTROL.md) — `ControlChannel` mechanics, every
  kind's effect, custom-kind extensibility.
- [PLAN-LIFECYCLE.md](PLAN-LIFECYCLE.md) — per-revision invariants,
  refinement modes, plan validation.
- [DRIFT.md](DRIFT.md) — drift taxonomy, severity, ladder mapping.
- [CANCELLATION-CONTRACT.md](CANCELLATION-CONTRACT.md) — cancel
  propagation rules and the audit catalog.
- [STATE-OWNERSHIP-CONTRACT.md](STATE-OWNERSHIP-CONTRACT.md) —
  ADK-side ownership rules; this doc layers the goldfive-internal
  rules on top.
- [ARCHITECTURE.md](ARCHITECTURE.md) — high-level component map.
- [RATIONALE.md](RATIONALE.md) — design choices explained for
  novel readers.
- [../reference/api.md](../reference/api.md) — public API
  surface; Phase 2 adds the `goldfive_*` internal kinds in a
  framework-internal note (§"Live steering").
