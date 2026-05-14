# Plan-descriptive growth: separating goals / plan / DAG / drift

## 1. Status

**Proposal — not yet implemented.** Filed for tracking; ships behind a feature
flag (`GOLDFIVE_PLAN_DESCRIPTIVE_GROWTH=1`) if accepted. No production code
in this PR. A separate implementation PR (with the 5-PR sequence in §9)
follows after design review.

Related: [PLAN-LIFECYCLE.md](PLAN-LIFECYCLE.md) (the contract this proposal
extends), [DRIFT.md](DRIFT.md) (the `CAPABILITY_MISMATCH` Rule C retirement
case), [STATE-OWNERSHIP-CONTRACT.md](STATE-OWNERSHIP-CONTRACT.md) (the
write-path discipline that constrains §5).

## 2. Motivation

Today goldfive conflates two concepts under one `Plan` object:

1. **Plan as intent (declarative).** What the planner forecast the run
   would do; updated only through `Planner.refine` and installed via
   `PlanReviser.install_revision_for_drift`.
2. **Plan as execution record (descriptive).** What the run actually did;
   the DAG of agent invocations + reporting-tool transitions the executor
   stamps onto the same `Plan` instance.

When reality matches the planner's forecast the conflation is invisible.
When reality grows past the forecast — a sub-agent decides to call a
debugger the planner did not anticipate — goldfive's only response is
either to **pin** the delegation to some pre-existing PENDING task via
`_maybe_pin_delegation_task` (often by misattribution) or to **fire
`CAPABILITY_MISMATCH`** when the pin fails (or fires onto the wrong task).

Specifically:

- In **observation mode** (`observation_only=True`), the plan view in
  harmonograf does not reflect what actually happened. The 20+ debugger
  delegations are visible only as drift events, not as nodes in the plan.
  Reviewers reading the post-run UI see a 3-task plan that "succeeded"
  alongside a stream of CAPABILITY_MISMATCH events nobody can explain.
- In **steering mode**, every legitimate delegation off the planner's
  forecast fires CAPABILITY_MISMATCH Rule C (`#268`) and triggers a
  refine. The refine almost always lands a near-identical plan because
  the LLM has no clean way to say "the existing plan is fine, the
  coordinator just discovered some incidental work."

The clean separation:

- **`Goals`.** What the user wants. Frozen at goal-derivation time. The
  intent. Already a separate field on `Session`.
- **`Plan` (immutable, `revision_index N`).** The planner's forecast of
  what should happen. Updates only via refines authored by the planner or
  by a USER_STEER. Already immutable (`#247`).
- **DAG / execution graph.** What actually happens, including everything
  the agents did. Always grows monotonically with observed work.
- **Drift events.** The comparison signal between DAG (reality) and
  Plan (forecast) or Goals (intent). Already first-class.

The proposal keeps the existing `Plan` type but adds a `discovered: bool`
marker on `Task` so the same data structure carries both the forecast
nodes (`discovered=False`) and the observed-execution nodes
(`discovered=True`). The execution-graph layer is **not** a new object —
it is the existing `Plan` with the descriptive growth contract overlaid.

### 2.1 Motivating evidence

E2E session `2d27ff4a` (cherry-tree run, 2026-05-13):

- Planner produced 3 tasks: `find_presentation_files`,
  `read_presentation`, `summarise_presentation`.
- Coordinator delegated to `debugger_agent` 20+ times to locate the file
  on disk before delegating to the planned agents.
- `_maybe_pin_delegation_task` pinned every `debugger_agent` delegation
  to `find_presentation_files` (the only PENDING task whose stem matched
  any token in the args).
- `detect_capability_mismatch` Rule C fired on each: `debugger_agent`'s
  stem `debugger` was absent from `find_presentation_files` and present
  in no other PENDING task — so Rule C technically should NOT have
  fired, but Rule A did (`debugger_agent` had only `AgentTool` wrappers
  and `find_presentation_files` reads as a leaf task).
- Net effect: 20+ spurious refines, each producing a near-identical
  plan, each adding ~5s to the run.

The session is the canonical reproducer for the rest of this doc.

## 3. ADK + goldfive surface today

The relevant surface area:

### 3.1 `_maybe_pin_delegation_task` (`goldfive/adapters/_adk_plugin.py:3825`)

Fires synchronously inside ADK's `before_tool_callback` when the
coordinator dispatches an `AgentTool`. Walks the live plan, builds the
eligible PENDING set (every upstream predecessor COMPLETED), and runs
three disambiguation tiers:

1. **Tier 1 — required-tools cover.** Pick the unique candidate whose
   `Task.required_tools` is fully covered by the invoked agent's tool
   names.
2. **Tier 2 — agent-name semantic match.** Pick the unique candidate
   whose `title + description` contains a stem from
   `invoked_agent_name` (e.g. `reviewer_agent` → `reviewer`).
3. **Tier 3 — topic-args scorer + topo-order fallback.** Tokenise
   `tool_args` against each candidate; first non-zero score wins; tie
   or zero overlap → first eligible by plan order.

On miss (zero eligible OR all tiers ambiguous AND fallback misfires):
the pin is best-effort — it always picks **something** from the eligible
set unless the set is empty. The structural mismatch is caught later by
`_maybe_emit_capability_mismatch` (Rule C).

Side effects of a successful pin:

- `replace_task(plan, chosen_id, assignee_agent_id=invoked_agent_name)`
  inside a `channel_processor_active()` envelope.
- `set_session_plan(session, new_plan)` — observational, not a revision,
  no `PlanRevised`.
- `StateStore.set_pin_current_task(...)` — pin the chosen task as
  `session.current_task_id`.

### 3.2 `_maybe_emit_capability_mismatch` (`_adk_plugin.py:4033`)

Fires from the same callback path AFTER the pin lands. Reads the bound
task off `session.current_task_id`, feeds the invoked agent's live tool
list to `goldfive.drift.capability_check.detect_capability_mismatch`,
and routes any returned `DriftEvent` through `steerer._handle_drift`
(intervention ladder → cancel + refine).

### 3.3 `Plan` and `Task` (`goldfive/types.py`)

`Plan` is frozen (`#247`). Every "edit" is a new instance built via
`replace_task` / `add_tasks` / `replace_edges` / `bump_revision`. The
live ref lives on `Session.plan` and is swapped via `set_session_plan`,
which enforces a single-writer invariant via the
`_CHANNEL_PROCESSOR_ACTIVE` `ContextVar`.

`Task` fields (current):

```
id, title, description, assignee_agent_id, status,
predicted_start_ms, predicted_duration_ms, bound_span_id,
cancel_reason, supersedes, supersedes_kind, required_tools
```

### 3.4 `PlanReviser` (`goldfive/plan_reviser.py`)

`PlanReviser._emit_plan_revised` is the canonical install path. Holds
the **per-session plan lock** (`_get_plan_lock`) across:

- supersedes integration,
- `session.plan` swap,
- watermark stamp,
- orchestration-state pointer update,
- pending-corrections GC + queue,
- `current_task_id` repin on supersedes,
- `PlanRevised` envelope emit + paired `plan_revised` correlation
  envelope.

`PlanReviser._wait_plan_stable(session, timeout=1.0)` is the
**read-side barrier**: callers acquire-then-release the lock so they
observe either pre- or post-revision plan state, never partial. Used by
report_task_* handlers and by `_resolve_effective_task_id`.

The lock is per-session, keyed on `session.id`. Concurrent sessions
sharing one steerer get independent locks.

Crucially: **`_maybe_pin_delegation_task` does NOT acquire this lock
today.** It performs its `set_session_plan` swap inside a
`channel_processor_active()` envelope but NOT inside the plan lock.
Pin writes race with refine writes; today the race is benign because
the pin only mutates `assignee_agent_id` (which the validator does not
re-check) on an existing task id (which always exists across revisions
the same way per `#247`). The proposal in §5 carefully preserves this
"pin write is not under the plan lock" property.

## 4. Proposed design

### 4.1 New `Task` field

```python
@dataclasses.dataclass(frozen=True)
class Task:
    ...
    #: When True, this task was added reactively at delegation time
    #: (or by an equivalent observation hook), not by initial planning
    #: or by a planner-authored refine. Default False so legacy plans
    #: and the validator's existing rules are unaffected.
    discovered: bool = False
```

Plain bool, default `False`. The validator (`Plan.validate`) treats
`discovered` as opaque metadata — it does not change rule 5 (creation:
all-PENDING), rule 6 (terminal preservation), rule 7 (no
absorbing→PENDING edges), or rule 8 (corrective-predecessor topology).

### 4.2 New event kind

Reuse the existing `DriftKind.NEW_WORK_DISCOVERED` machinery rather
than introducing `plan_descriptive_growth`. Justification:

- `NEW_WORK_DISCOVERED` is already a first-class drift kind with the
  `observation_only` carve-out wired (§4.3 of PLAN-LIFECYCLE.md,
  `_apply_revision` discovery branch in `plan_reviser.py:1510`).
- It already routes through `install_revision_for_drift` and emits
  `DriftDetected` + `PlanRevised` correlated envelopes.
- Harmonograf already renders `NEW_WORK_DISCOVERED` revisions as
  "discovery" rather than "correction."

The new task is identified by `task.discovered == True`. The
`PlanRevisionDiff.added_task_ids` list already exists; sinks consume
the bool by re-reading the new task off the `PlanRevised.plan` proto.

(Sub-alternative: add a `discovered_task_ids` repeated field on
`PlanRevisionDiff`. Worth doing in PR 3 if the harmonograf side wants
to filter without re-reading the full plan.)

### 4.3 New flow in `_maybe_pin_delegation_task`

```
on delegation_observed (before_tool_callback):
    eligible = build_eligible_pending(plan)
    if eligible:
        chosen = run_tier_1_2_3(eligible, invoked_agent, args)
        if chosen has stem match with invoked_agent_name:
            stamp_assignee_and_pin(chosen)        # today's path
            return

    # All tiers missed (zero eligible OR no stem match).
    # NEW: synthesise a discovered task.
    title = derive_title(invoked_agent_name, tool_args)
    discovered_task = Task(
        id=mint_id(invoked_agent_name),
        title=title,
        description=truncate(tool_args, 256),
        assignee_agent_id=invoked_agent_name,
        status=TaskStatus.PENDING,
        discovered=True,
    )

    # Schedule an async revision adding the discovered task.
    # See §5 for the async-ordering contract.
    schedule_discovery_revision(session, discovered_task)

    # Pin the agent to the discovered task id immediately so the
    # synchronous emit of delegation_observed carries the correct
    # task_id without waiting for the revision to land.
    pin_discovered_task(session, discovered_task)
```

#### 4.3.1 Title derivation

In priority order:

1. The `request` / `task` / `goal` argument off `tool_args` (the conventional
   `AgentTool` payload key), truncated to 80 chars.
2. The first 80 chars of `invoked_agent_name + ": " +
   first_reasoning_trace` (collected via the `Steerer.observe_reasoning`
   side channel once the sub-invocation starts).
3. Fallback: `f"{invoked_agent_name}: discovered work"`.

The title is stable for the lifetime of the discovered task — a refine
may supersede it (§4.5) but the original discovered task is not
retitled.

#### 4.3.2 Replacing Rule C, not Rule A/B

The discovery path runs **only when the pin would miss**. Today's pin
already prefers Tier 1 / Tier 2 matches before falling back. We
strengthen the fallback gate so Tier 3 (topic-args scorer / topo-order)
no longer runs — if neither Tier 1 nor Tier 2 produces a stem match,
the discovery branch fires instead.

Rule A (coordinator-style leaf-assignment) and Rule B (required-tools
cover) remain valid (§7) and run after the pin lands on the discovered
task. The discovered task has empty `required_tools` so Rule B is silent;
Rule A may still fire if the planner's title-derivation produces a
delegation-shaped title for a leaf-only agent, but that is the genuine
structural anti-pattern Rule A exists to catch and not a side effect of
discovery.

### 4.4 Observability

Every discovered task generates:

1. `PlanRevised` with `drift_kind=NEW_WORK_DISCOVERED`, `revision_index`
   bumped by 1, `diff.added_task_ids=[discovered.id]`, and the new task
   carrying `discovered=True` on its proto representation.
2. `DriftDetected` (`NEW_WORK_DISCOVERED`, INFO severity — see §4.6)
   stamped with `current_task_id=discovered.id` and
   `current_agent_id=invoked_agent_name`.
3. The standard `delegation_observed` event with `task_id=discovered.id`
   (see §5 for how this stays consistent under the async-ordering
   constraint).

### 4.5 Refine cascade

A subsequent planner-authored refine may:

- **Modify** a discovered task (any field except `discovered`). The
  `discovered=True` marker is preserved across revisions — the validator
  treats it like any other immutable metadata.
- **Supersede** a discovered task with a `SupersessionKind.REPLACE`
  link to a non-discovered task. Useful when the planner decides the
  discovered work was a precursor to a planned task and wants to merge
  the two — e.g. discovered `debugger: locate files` becomes
  `find_presentation_files` in the next refine.
- **Remove** a discovered task is allowed only via the
  REPLACE/CORRECT supersedes path (PLAN-LIFECYCLE.md §3.5). Direct
  removal violates terminal-preservation (§3.1) if the task is
  terminal, and corrective-predecessor topology (§3.6) if non-terminal
  with downstreams.

### 4.6 Severity

`NEW_WORK_DISCOVERED` is currently WARNING (DRIFT.md §"Discovery
category"). For descriptive-growth synthesis we downgrade to INFO when
the drift was synthesised by the framework (not by a sub-agent calling
`report_new_work_discovered`). This keeps the existing reporting-tool
path at WARNING (the planner SHOULD adapt when a sub-agent flags new
work explicitly) while marking framework-synthesised discoveries as
observational (the work has already started — refining the plan is
optional).

Encoding: stamp `drift.authored_by="goldfive"` on the synthesised drift
(matching the existing convention) and have `DefaultSteerer.should_refine`
opt out of refine for `(kind=NEW_WORK_DISCOVERED, authored_by=goldfive,
severity=INFO)` triples.

## 5. Async ordering invariants (the tricky part)

`_maybe_pin_delegation_task` runs synchronously inside ADK's
`before_tool_callback`. The synchronous contract: by the time the
callback returns, the `delegation_observed` event must carry a stable
`task_id`. ADK does not give us a deferred-emit primitive at this layer.

`PlanReviser._emit_plan_revised` holds the per-session plan lock across
~200 lines of state-mutation work. The lock is per-session, async, and
re-entrant only via the same task (asyncio's `Lock` is not reentrant —
attempting to acquire from within a holder deadlocks).

We considered three options for the install path.

### Option A — `task_id=DISCOVERY_PENDING` sentinel + back-fill

Emit `delegation_observed` with `task_id="discovered:pending"`; schedule
the revision via `asyncio.create_task`; downstream events back-fill the
task_id on the report_task_* handlers.

**Why we rejected.** Back-fill across event boundaries is the same
shape of bug `_wait_plan_stable` exists to prevent. Sinks (harmonograf)
that join the delegation event against the task_plans table on a
sentinel id observe a row with no foreign key for the half-second the
revision is in flight. The fix-up requires every sink to special-case
the sentinel.

### Option B — pre-allocate task_id at pin-time, revision lands async

Mint the discovered task's id synchronously, emit `delegation_observed`
with the real id, schedule the revision via `asyncio.create_task`. The
new task is in the plan by the next read.

**Why we rejected for the install path.** The synchronous emit of the
delegation event carries a `task_id` that is NOT (yet) in
`session.plan.tasks`. Any reader that consults `session.plan` between
the pin and the revision lands sees a dangling reference. The
`_wait_plan_stable` barrier in report_task_* handlers protects them
from the racing revision write but does NOT teach them that the task
is about to appear — they observe the pre-revision plan and treat the
delegation as belonging to no plan task.

### Option C — synchronous plan growth via a lighter-weight code path *(chosen)*

`_maybe_pin_delegation_task` performs the plan swap synchronously
inside the existing `channel_processor_active()` envelope, the same
way the assignee-stamp swap does today. The discovery write does NOT
acquire the per-session plan lock and does NOT route through
`PlanReviser._emit_plan_revised`'s full mutation pipeline. Instead:

1. Synchronously, inside `_maybe_pin_delegation_task` (still in the
   ADK callback):
   - Build the new `Plan` via `add_tasks(plan, [discovered_task])` +
     `bump_revision(...)`. No edges added (the discovered task is an
     independent sub-DAG root by construction — it has no DAG
     predecessors; rule 7 allows this because the predecessor set is
     empty).
   - Swap the live ref via `set_session_plan(session, new_plan)` inside
     `channel_processor_active()`.
   - Stamp `session.current_task_id = discovered.id` and the
     `StateStore` pin.
2. After the callback returns, schedule a fire-and-forget task that
   emits the paired `PlanRevised` + `DriftDetected` correlation envelopes
   via `PlanReviser._emit_plan_revised_correlation` (which does NOT
   acquire the lock and does NOT re-swap `session.plan` — it is purely
   the wire-emit half of the pipeline).

Why this is safe:

- **No lock contention.** The discovery write does not contend with
  refines because refines that land on the same revision_index are
  rejected by the validator (rule 5/6) and refines that land at N+1
  see the discovery-augmented plan as their `prior` — they preserve
  the discovered task via the terminal-preservation rule once it
  reaches a terminal status, and they may freely modify or supersede
  it while it is non-terminal.
- **No partial apply visible to `_wait_plan_stable` callers.** The
  swap is atomic at the `ContextVar`-protected `set_session_plan`
  call. A reader either sees revision N (no discovered task) or
  revision N+1 (discovered task installed) — never a half-swapped
  state.
- **The supersedes-integration / current_task_id repin / pending-
  corrections sites are not needed.** A freshly-minted PENDING task
  has no supersedes link, no prior pin to migrate from (the pin moves
  TO it), and no pending-corrections entries. The skipped pipeline
  steps are no-ops on a discovery shape by construction.
- **Idempotency.** A discovery write is keyed by the mint function's
  output; the mint function uses `uuid4` so two concurrent delegations
  from the same agent never collide. The pin write itself is
  idempotent on `current_task_id` (set_pin_current_task no-ops on
  equality).

What we lose by skipping `_emit_plan_revised`:

- Supersedes-integration runs (no-op).
- `clear_obsolete_corrections_on_revision` runs (no-op — no corrections
  reference the new id).
- `queue_corrections_for_revision` runs (no-op — discovered tasks have
  no supersedes_kind CORRECT links).
- `_repin_current_task_on_supersedes` runs (no-op — no supersedes
  links on the discovered task).
- Cross-revision diff is computed by the correlation envelope, not the
  primary envelope. Sinks consuming `PlanRevised.diff` see the discovered
  task in `added_task_ids` regardless of which emit path it came from.

The correlation-envelope emit must NOT acquire the plan lock either —
this is a hard contract. It runs after the in-line swap; if a refine
fires concurrently it lands at revision_index N+2 and the correlation
envelope emit at N+1 stays consistent with the snapshot it captured.

### 5.1 The new write helper

```python
# goldfive/plan_reviser.py
async def install_descriptive_growth(
    self,
    *,
    session: Session,
    new_task: Task,
) -> bool:
    """Install a discovered task synchronously.

    Bypasses the full _emit_plan_revised pipeline (supersedes
    integration, pending-corrections GC, current_task_id repin) which
    are no-ops on the discovery shape by construction. The plan swap
    is performed under channel_processor_active() but NOT under the
    plan lock — see PLAN-DESCRIPTIVE-GROWTH.md §5.

    Schedules the paired PlanRevised + DriftDetected emit as a
    fire-and-forget task so the caller (the ADK before_tool_callback)
    can return synchronously.
    """
```

The helper is called directly from `_maybe_pin_delegation_task`. The
synchronous-emit decision is encoded by the helper, not the caller.

### 5.2 Test plan for the async contract

Required tests (impl PR 2):

- Concurrent discovery + refine: a `NEW_WORK_DISCOVERED`-synthesised
  growth racing with a planner-authored refine targeting the same
  prior revision_index lands one or the other deterministically (the
  loser observes the winner's plan as `prior` and re-validates).
- `_wait_plan_stable` callers crossing a discovery write observe
  either the pre-discovery or post-discovery plan — never a torn read.
- Pin idempotency under multiple deliveries of the same delegation.

## 6. Steering mode vs observation mode

### 6.1 Steering mode (`observation_only=False`)

- Discovered task is in `session.plan`.
- Reasoning content from the discovered task's agent still feeds the
  `Steerer.observe_reasoning` pipeline → goal-judge runs against
  `session.goals`. Off-topic reasoning fires `OFF_TOPIC` /
  `INTENT_DIVERGENCE` drift on a task that is genuinely in the plan,
  so refine has a clean place to land (it can supersede or cancel the
  discovered task).
- The intervention ladder operates normally on the discovered task.

### 6.2 Observation mode (`observation_only=True`)

- Discovered task is installed identically (the `NEW_WORK_DISCOVERED`
  carve-out in `_apply_revision` — see `plan_reviser.py:1510` —
  already exempts discovery from the observation-only gate; the
  descriptive-growth proposal piggybacks on this exemption).
- Drift signals stay observational — no refine, no cancel, no STEER.
- The plan view in harmonograf reflects what actually happened, which
  is the primary user-visible win.

## 7. CAPABILITY_MISMATCH after this

| Rule | Current behaviour | Post-proposal |
|---|---|---|
| **A** — coordinator-style leaf-assignment | Fires CRITICAL when the invoked agent has only AgentTool wrappers AND the bound task reads as a leaf | **Unchanged.** Still valid — structural anti-pattern, unrelated to plan-matching. |
| **B** — required-tools cover | Fires CRITICAL when `task.required_tools` is non-empty and the invoked agent's tool surface does not cover it | **Unchanged.** Still valid — skill-gap signal. |
| **C** — out-of-DAG-order delegation (`#268`) | Fires CRITICAL when the invoked agent's role stem is absent from the bound task AND present in another PENDING task | **Becomes inert.** Plan grows to match — the discovered task carries the agent's role stem in its title, so the stem-present-in-another-task condition never trips after the growth. Recommend retiring Rule C entirely (PR 4). |

Rule C's failure mode was specifically the symptom of the
forecast-vs-reality conflation §2 describes: the pin landed on the
wrong task because there was no right task. Growing the plan to add the
right task addresses the cause, not the symptom.

### 7.1 Retiring Rule C

Two PRs (4a, 4b):

1. **Soft retirement.** Disable `_rule_c_dag_order` behind a feature
   flag (`GOLDFIVE_CAPABILITY_RULE_C=0` by default after descriptive
   growth is on). Ship for one release cycle.
2. **Hard retirement.** Delete the rule, the `all_pending_tasks`
   parameter, and the `_task_text_contains_stem` helper. Update
   `detect_capability_mismatch` to two rules.

## 8. What changes about the pothos / cherry-tree run

Walk-through of session `2d27ff4a` under the new model.

Initial plan (rev 1, planner-authored):

```
T1: find_presentation_files       (PENDING)
T2: read_presentation             (PENDING, depends on T1)
T3: summarise_presentation        (PENDING, depends on T2)
```

Coordinator delegates to `debugger_agent` with
`request="locate cherry tree files"`. Today's flow: pin lands on T1,
Rule A fires (debugger_agent has only AgentTool wrappers, T1 reads as
leaf), refine runs, plan barely changes. Repeats 20x.

Proposed flow:

1. `_maybe_pin_delegation_task` runs Tier 1 (`debugger_agent` has no
   `required_tools` cover for T1 — skipped). Runs Tier 2: stem
   `debugger` is absent from T1's title+description. Skipped. Runs
   Tier 3: zero token overlap; today would fall back to first eligible
   (T1); under proposal, the discovery branch fires instead.
2. Synthesise discovered task `T1d`:
   ```
   id: discovered-<uuid4>
   title: "debugger_agent: locate cherry tree files"
   description: "request='locate cherry tree files'"
   assignee_agent_id: debugger_agent
   status: PENDING
   discovered: True
   ```
3. `install_descriptive_growth` lands `T1d`. Plan is now rev 2:
   ```
   T1, T2, T3 (unchanged), T1d (discovered, PENDING)
   ```
4. Pin: `session.current_task_id = T1d.id`. `delegation_observed`
   emitted with `task_id=T1d.id`.
5. `_maybe_emit_capability_mismatch` runs: bound task is T1d whose
   title contains the stem `debugger`. Rule A fires? T1d's title
   starts with `debugger_agent:` — substring check for delegation
   markers passes (`"delegate"` not in title) so Rule A judges T1d
   leaf-shaped. **This is a problem.** See §11 for the open question.
6. Debugger searches the filesystem (calls `find_files` etc.). Each
   reasoning block runs through `observe_reasoning`. Cherry-tree
   queries are off-topic relative to `session.goals` (`"summarise the
   pothos presentation"`) — `OFF_TOPIC` fires. Preserved by the
   proposal — `OFF_TOPIC` is content-based, not plan-structure-based.
7. `LOOPING_REASONING` tunes itself to fire on N+ consecutive off-goal
   calls (cross-link: tracking issue for the looping detector's
   off-goal-aware tuning).
8. Eventually the coordinator delegates to the planned agents. Plan
   view at end: 3 original + 1 discovered task, with drift markers on
   T1d.

Harmonograf UI: 4 tasks, T1d clearly badged "discovered" (separate
visualisation, PR 3), drift event timeline on T1d. The 20+ spurious
refines do not happen.

## 9. Implementation plan

| PR | Scope | Behind flag |
|---|---|---|
| 1 | `Task.discovered: bool` field + `Plan.validate` update (no rule change; just opaque metadata) + state-store migration (proto field + harmonograf-side schema bump) | n/a — additive |
| 2 | `_maybe_pin_delegation_task` fallback to discovery + `PlanReviser.install_descriptive_growth` helper + the §5.2 test suite | `GOLDFIVE_PLAN_DESCRIPTIVE_GROWTH=1` |
| 3 | Harmonograf-side rendering of discovered tasks (badge, separate visualisation lane, drift filter) | flag-gated |
| 4 | Retire `CAPABILITY_MISMATCH` Rule C (4a soft, 4b hard) | flag → default-on |
| 5 | Docs: update PLAN-LIFECYCLE.md (revision-cascade §4.5 covers discovery), DRIFT.md (Rule C retirement), this design doc (mark Implemented) | n/a |

Each PR is independently merge-able. PR 2 ships behind the flag so
production traffic stays on the old path until PR 3 lands the UI
support.

## 10. Alternatives considered

### 10.1 Plan auto-growth from `delegation_observed` directly

Skip `_maybe_pin_delegation_task` entirely; grow the plan every time a
delegation is observed.

**Why rejected.** Doesn't reuse the pin machinery, which is the right
behaviour when the planner DID forecast the delegation correctly.
Tier 1 (required-tools cover) and Tier 2 (stem match) succeed in the
common case where the planner got the assignee right; only the fallback
needs descriptive growth. Always growing the plan would produce a
duplicate-PENDING-task shape (planned T1 + discovered T1d for the same
agent) that the validator would reject as a same-stem ambiguity and
that the executor would race on.

### 10.2 Adding a separate `ExecutionGraph` object beside the `Plan`

Cleaner separation — Plan stays declarative, ExecutionGraph is
strictly descriptive — but a much bigger refactor:

- New proto type, new sink contract, new harmonograf table.
- `current_task_id` needs to disambiguate between Plan task ids and
  ExecutionGraph node ids.
- Reporting-tool calls need to target one or the other.
- Drift detectors need to know which graph to consult.

The `Task.discovered` overlay achieves 80% of the semantic benefit at
20% of the code cost. The full separation is a future-Phase-7 move if
the overlay proves insufficient.

### 10.3 Leaving the plan static + improving the UI

Render the unplanned delegations purely in the drift timeline; don't
touch the plan structure. Smaller change but doesn't address the
semantic confusion — `_maybe_pin_delegation_task` still pins to
arbitrary tasks, Rule C still fires spuriously, the planner still
gets refine requests it cannot productively answer.

## 11. Open questions

### 11.1 Granularity: per-delegation vs per-agent-task pair

If the coordinator delegates to `debugger_agent` 20 times in one run,
do we synthesise 20 discovered tasks or 1?

- **Per-delegation (1-to-1).** Conceptually clean — the DAG node count
  matches the delegation count. Plan grows large on chatty
  coordinators.
- **Per-(agent, task-prefix) (deduplicated).** First delegation to
  `debugger_agent` mints `T1d`; subsequent delegations with similar
  `tool_args` re-pin to `T1d` instead of growing the plan. Match
  predicate is the same Tier 2 stem + Tier 3 args-overlap pair.

Tentative resolution: **per-(agent, args-token-set)** with a TTL of
"until the discovered task reaches terminal" so the plan does not
grow unboundedly. Final answer in impl PR 2 after we measure on the
cherry-tree session.

### 11.2 Plan-validate invariants: do discovered tasks have predecessor edges?

Today's proposal lands every discovered task as an independent sub-DAG
root (no upstream edges). Pro: bypasses rule 7 (no
absorbing→PENDING) cleanly. Con: the executor's DAG scheduler treats
every discovered task as immediately eligible, which is true at
discovery time (the delegation has already fired) but loses the causal
ordering info if the discovered task was discovered while a planned
task was running.

Tentative resolution: **no predecessor edges** for the initial
implementation. A discovered task discovered while `T_running` is
RUNNING can carry an optional `discovered_during: str = ""` field
(separate from `supersedes`) for sink-side causal display, but it is
not a DAG edge. If we later want execution ordering, the planner can
refine and add the edge.

### 11.3 Refine cascade: can refine subsequently modify or remove a discovered task?

§4.5 says yes via the supersedes path. The validator already permits
this — the `discovered` field is opaque metadata, and supersedes
mechanics work the same way regardless of the marker.

Open: should the planner's prompt mention discovered tasks
differently? E.g. "you may consolidate discovered tasks into your
existing forecast." This is a planner-prompt tuning question, not a
core-types question. Defer to impl PR 2.

### 11.4 Rule A on discovered tasks

§8 step 5 noted Rule A could spuriously fire on a discovered task
whose title is the auto-derived `agent_name: request_summary` shape.
Three resolutions:

- **(a)** Skip Rule A entirely on `task.discovered=True` tasks. Loses
  the legitimate signal when a coordinator delegates to a coordinator
  for a leaf task.
- **(b)** Adjust the title-derivation function so titles read as
  delegation-shaped when the invoked agent is a coordinator (use the
  agent's tools list at synthesis time). More complex; depends on
  knowing the invoked agent's tools at pin time, which we already
  do for Rule B.
- **(c)** Special-case Rule A: when bound task is discovered, suppress
  the leaf-task heuristic but keep the AgentTool-only check. Fires
  only when the discovered task has been there long enough for the
  agent to have produced output — i.e., delays the fire to
  `after_tool_callback` or later.

Tentative resolution: **(a)** for impl PR 2; revisit if real runs
surface false negatives.

### 11.5 Interaction with `RUNAWAY_DELEGATION`

`RUNAWAY_DELEGATION` fires when the per-invocation AgentTool spawn
count exceeds `agent_tool_cap`. Descriptive growth does NOT bypass
this — discovered-task delegations count against the same cap. A
chatty coordinator hitting the cap still triggers the structural
break-glass.

## 12. References

### Issues

- `#410` — Facade cleanup (the immediate downstream cleanup once the
  descriptive-growth path lands).
- `#253` — CAPABILITY_MISMATCH replacing PLAN_DIVERGENCE (Rules A + B
  context).
- `#268` — CAPABILITY_MISMATCH Rule C (out-of-DAG-order delegation,
  the rule this proposal retires).
- `#258` — `NEW_WORK_DISCOVERED` carve-out in `_apply_revision`'s
  observation-only gate (the mechanism the descriptive-growth path
  reuses).
- `#247` — Immutable Plan (the dataclass-frozen contract every plan
  edit goes through).
- `#248` — Corrective-predecessor topology (PLAN-LIFECYCLE.md §3.6,
  the validation rule preserved by the proposal).
- `#249` — Actor-model channel processor (the single-writer envelope
  `set_session_plan` enforces).
- `#403` — Partial-apply window in `_emit_plan_revised` (the lock
  contract §5 carefully does NOT contend with).

### Sessions

- `2d27ff4a` — cherry-tree E2E (2026-05-13). The motivating example
  for §2 and §8.

### Adjacent design docs

- [PLAN-LIFECYCLE.md](PLAN-LIFECYCLE.md) — the contract this proposal
  extends.
- [DRIFT.md](DRIFT.md) — taxonomy + the `CAPABILITY_MISMATCH` Rule C
  retirement.
- [STATE-OWNERSHIP-CONTRACT.md](STATE-OWNERSHIP-CONTRACT.md) — the
  write-discipline §5 honours.
- [CONTROL-CHANNEL.md](CONTROL-CHANNEL.md) — the actor-model context
  inside which the new write path lives.
