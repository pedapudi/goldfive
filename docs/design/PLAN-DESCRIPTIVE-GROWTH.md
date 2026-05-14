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
the same way per `#247`). §5 of this proposal **abandons** that
"pin write is not under the plan lock" property for the descriptive-
growth write — see §5 for why, and §3.5 for the conceptual framing
that makes the cost worth paying.

### 3.5 The two entities

The proposal treats **intent** and **record** as conceptually separate
entities, even though they are physically packaged in one `Plan` data
structure via `Task.discovered: bool`. This is the load-bearing
framing for the rest of the doc.

- **Plan-as-intent.** The planner's forecast: what the run is
  supposed to do. Mutated only by refines — corrective insertions,
  supersedes topology, USER_STEER-authored adjustments. Tasks with
  `discovered=False`. Lifecycle owner: `Planner.refine` →
  `PlanReviser.install_revision_for_drift`.
- **Plan-as-record.** What actually executed: the DAG of agent
  invocations the executor witnessed. Grown by
  `_maybe_pin_delegation_task` falling through to discovery. Tasks
  with `discovered=True`. Lifecycle owner: the descriptive-growth
  write helper introduced in §5.
- **Drift signal.** The continuous diff between intent and record.
  Today drift detectors fire on individual content checks (OFF_TOPIC,
  LOOPING_REASONING, CAPABILITY_MISMATCH); after this proposal the
  structural mismatch — "the executor went somewhere the planner did
  not forecast" — becomes first-class as a `NEW_WORK_DISCOVERED`
  revision on the same Plan object.

The two-entity model is what justifies the bool-overlay design choice:
intent and record share a data structure today because the operations
on them (validate, render, query) are the same. The proposal does
**not** invent a new container; it formalises the distinction that
already exists implicitly when `_maybe_pin_delegation_task` and
`Planner.refine` both touch `session.plan`.

**Forward-compatibility note.** The overlay can later promote to a
separate `ExecutionGraph` object beside the `Plan` if the conceptual
distinction proves to need divergent operations (e.g., the record needs
sub-second granularity that the intent does not, or sinks want to
filter without re-reading the `discovered` flag per task). §10.2
sketches what that move looks like; it is **not** required by this
proposal. The bool-overlay form is intentionally the smallest viable
shape that makes the two entities nameable.

## 4. Proposed design

### 4.1 New `Task` fields

```python
@dataclasses.dataclass(frozen=True)
class Task:
    ...
    #: When True, this task was added reactively at delegation time
    #: (or by an equivalent observation hook), not by initial planning
    #: or by a planner-authored refine. Default False so legacy plans
    #: and the validator's existing rules are unaffected.
    discovered: bool = False
    #: Stable hash of (agent_name, args-token-set) at discovery time.
    #: Used by _maybe_pin_delegation_task to dedup repeated delegations
    #: to the same agent for the same logical args — see §4.3.0 and
    #: §11.1. Empty string for forecast tasks. Preserved across
    #: refines (§4.3.0 "Cross-refine survival"). Opaque to the
    #: validator.
    discovery_identity_hash: str = ""
```

Both fields are opaque metadata. The validator (`Plan.validate`) does
not change rule 5 (creation: all-PENDING), rule 6 (terminal
preservation), rule 7 (no absorbing→PENDING edges), or rule 8
(corrective-predecessor topology).

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
        chosen = run_tier_1_2(eligible, invoked_agent, args)
        if chosen has stem match with invoked_agent_name:
            stamp_assignee_and_pin(chosen)        # today's path
            return

    # Tier 1/2 missed.
    # NEW: dedup against prior discovered tasks, then grow if novel.
    identity_hash = discovery_identity_hash(invoked_agent_name, tool_args)
    existing = find_discovered_task_by_hash(plan, identity_hash)
    if existing:
        # A prior delegation in this run already discovered this
        # (agent, args-token-set). Pin to it; no plan growth.
        pin_to_existing(session, existing)
        return

    # Novel (agent, args-token-set). Synthesise a discovered task.
    title = derive_title(invoked_agent_name, tool_args)
    discovered_task = Task(
        id=mint_id(invoked_agent_name),
        title=title,
        description=truncate(tool_args, 256),
        assignee_agent_id=invoked_agent_name,
        status=TaskStatus.PENDING,
        discovered=True,
        discovery_identity_hash=identity_hash,  # stamped on the task
    )

    # Install the discovered task under the per-session plan lock.
    # The lock acquisition is the linearisation point against
    # concurrent refines and concurrent discoveries — see §5 Option D.
    install_descriptive_growth(session, discovered_task)
```

#### 4.3.0 Stable identity hash for dedup

The dedup key collapses (agent, args-token-set) pairs to a stable
hash so repeated delegations from the same coordinator to the same
agent for the same logical task re-pin instead of growing the plan.

**Input source.** `tool_args` is read from the
`DelegationObserved` proto field added in PR 1 (§9) — i.e., from
the observed event the agent itself authored, not from a
goldfive-side intercept of agent state at pin time. The
normalisation below (lowercase, whitespace-strip, drop stop-tokens)
is computed on the OBSERVED args. See §13 for the underlying
"adaptive, not predictive" principle.

```python
def discovery_identity_hash(
    agent_name: str,
    tool_args: Mapping[str, Any],
) -> str:
    """Stable hash for descriptive-growth dedup.

    Two delegations to the same agent_name with the same
    args-token-set hash to the same value, regardless of:
      - whitespace differences in args
      - capitalization differences in args
      - arg-key ordering
      - cosmetic re-orderings of tokens within a single arg value

    The hash is the §6 dedup key: a new delegation whose hash matches
    an existing discovered task pins to that task; a new hash grows
    the plan.
    """
    tokens = _normalize_args_tokens(tool_args)
    payload = f"{agent_name}\0" + ",".join(sorted(tokens))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_args_tokens(tool_args: Mapping[str, Any]) -> frozenset[str]:
    """Lowercase, strip, tokenise on whitespace + punctuation.

    Trivial whitespace and capitalization variants must map to the
    same token set so 'Cherry Trees' and 'cherry trees' dedup.
    """
    text = " ".join(str(v) for v in tool_args.values())
    tokens = re.findall(r"\w+", text.lower())
    # Drop stop-tokens that add no signal.
    return frozenset(tokens) - _DISCOVERY_STOP_TOKENS
```

`Task.discovery_identity_hash: str` is a new field on `Task` (default
empty string for forecast tasks) added in PR 1 alongside
`discovered: bool`. The validator treats it as opaque metadata.

**Cross-refine survival.** When a planner refine modifies a
discovered task (§4.5), the hash is preserved. When a refine
supersedes a discovered task with a non-discovered task, the new
task inherits the hash so a later delegation to the same
`(agent, args-token-set)` pins to the now-non-discovered task rather
than re-growing.

**TTL.** The dedup window is "until the discovered task reaches a
terminal status" — see §11.1.

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

## 5. Install path — lock-acquiring synchronous growth

`_maybe_pin_delegation_task` runs synchronously inside ADK's
`before_tool_callback`. The synchronous contract: by the time the
callback returns, the `delegation_observed` event must carry a stable
`task_id`. ADK does not give us a deferred-emit primitive at this layer.

`PlanReviser._emit_plan_revised` holds the per-session plan lock across
~200 lines of state-mutation work. The lock is per-session, async, and
re-entrant only via the same task (asyncio's `Lock` is not reentrant —
attempting to acquire from within a holder deadlocks).

We considered four options for the install path.

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

### Option C — lock-free synchronous plan growth (REJECTED)

An earlier draft of this doc proposed performing the plan swap
synchronously inside `channel_processor_active()` but **without**
acquiring `_get_plan_lock`, on the theory that the discovery write
mutates only an additive shape (new task, no supersedes, no
current_task_id migration) and so cannot tear an existing reader.

**Why we rejected it.** A concurrent refine on the same session can
be mid-flight inside `Planner.refine` reading `session.plan` to decide
its revision shape — even in observation mode, where the refine still
runs and produces a `dry_run=True` revision so the drift dataset is
complete. The race is exactly the partial-apply window fixed in
**#403**:

1. Refine A reads `session.plan` at revision N (no discovered task).
2. Descriptive-growth write B swaps `session.plan` to revision N+1
   (adds discovered task `T_d`).
3. Refine A finishes computing its revision against the prior it
   sampled in step 1, calls `install_revision_for_drift` with
   `revision_index=N+1`, validation passes against the now-stale
   prior, and the write lands — **clobbering `T_d`.**

#403's lesson was crisp: **single writer, inside the lock, full stop.**
The descriptive-growth write is a writer; it MUST acquire the lock.
The argument that the discovery shape is "additive only" is the same
shape of argument that almost shipped #403 unfixed — the
"additive-only" reasoning ignores the read-side races on the prior
revision_index that lock acquisition exists to serialise.

The "additive only" claim is also wrong on closer inspection: the
discovery write bumps `revision_index`, and any concurrent refine that
samples the pre-bump index will validate against a stale prior. The
clobber risk is identical to #403's partial-apply window.

### Option D — lock-acquiring synchronous plan growth *(chosen)*

`_maybe_pin_delegation_task` performs the discovery write
synchronously inside the ADK callback, but **acquires the per-session
plan lock** for the swap window. This is the same contract
`PlanReviser._emit_plan_revised` honours (post-#403): single writer,
inside the lock, full stop.

1. Synchronously, inside `_maybe_pin_delegation_task` (still in the
   ADK callback), acquire `_get_plan_lock(session.id)` for the
   duration of the swap. Inside the lock:
   - Re-read `session.plan` (the lock acquisition is the linearisation
     point — any concurrent refine has either completed before this
     read or is queued behind it).
   - Run the dedup check from §6 against the post-lock plan; if a
     prior discovery write or refine already added the matching
     `(agent_name, args-token-set)` task, pin to it and skip growth.
   - Build the new `Plan` via `add_tasks(plan, [discovered_task])` +
     `bump_revision(...)`. No edges added (the discovered task is an
     independent sub-DAG root by construction — it has no DAG
     predecessors; rule 7 allows this because the predecessor set is
     empty).
   - Swap the live ref via `set_session_plan(session, new_plan)` inside
     `channel_processor_active()` (the channel-processor envelope is
     orthogonal to the plan lock; both apply).
   - Stamp `session.current_task_id = discovered.id` and the
     `StateStore` pin.
2. Release the lock.
3. Emit the paired `PlanRevised` + `DriftDetected` envelopes. The
   emit can be fire-and-forget (off-lock) because the snapshot it
   carries was captured inside the lock and cannot tear.

Both **steering mode and observation mode** acquire the lock. Refines
run in observation mode too — they emit `dry_run=True` revisions for
the drift dataset — so the read-then-write race exists identically in
both modes.

Why this is safe (and why the §6 dedup design needs the lock):

- **No clobber by concurrent refine.** A refine cannot complete its
  install against revision N while the discovery write holds the lock
  on revision N. The refine either lands first (the discovery write
  then re-reads inside the lock, dedups or grows against the refined
  plan) or queues behind (the refine then sees the post-discovery plan
  as its prior, preserves `T_d` via the immutable-on-supersedes
  contract, and validates cleanly).
- **No partial apply visible to `_wait_plan_stable` callers.** The
  swap is atomic at the `ContextVar`-protected `set_session_plan`
  call AND serialised behind the lock the read-side barrier
  acquire-then-releases. A reader either sees revision N (no
  discovered task) or revision N+1 (discovered task installed) —
  never a half-swapped state.
- **Dedup is consistent.** The §6 dedup check needs to read
  `session.plan.tasks` to find a prior matching discovered task; that
  read must happen inside the lock to be linearisable against
  concurrent discoveries (two delegations of the same
  `(agent, args-token-set)` arriving simultaneously would otherwise
  each mint a fresh discovered task).
- **The supersedes-integration / current_task_id repin / pending-
  corrections sites are still no-ops on the discovery shape.** The
  helper inlines the small slice of `_emit_plan_revised` that applies
  (validate, swap, watermark, pin) and skips the rest. The lock
  acquisition is what gives us the same correctness guarantee — not
  the full pipeline.

#### 5.0.1 Synchronous-emit budget

Lock acquisition adds latency to the ADK callback. The lock is
contended only with refines on the same session; in the cherry-tree
session (§2.1) refines fire on the order of seconds, not milliseconds,
and the discovery write inside the lock is dominated by the
`add_tasks` + `bump_revision` allocations (sub-millisecond). The
post-lock emit is fire-and-forget. Net expected callback cost in the
common case: <5ms; worst case (lock contended behind a refine):
<100ms (one refine cycle), still bounded by the refine's own timeout.

The synchronous-emit cost is what we pay to honour #403. It is worth
it.

### 5.1 The new write helper

```python
# goldfive/plan_reviser.py
async def install_descriptive_growth(
    self,
    *,
    session: Session,
    new_task: Task,
    identity_hash: str,
) -> tuple[Plan, bool]:
    """Install a discovered task synchronously, under the plan lock.

    `identity_hash` is computed by the caller from the observed
    `DelegationObserved.tool_args` payload (PR 1 proto extension —
    see §9, §13) — i.e., from the agent-authored args carried on
    the event, NOT from a goldfive-side state intercept at
    pin/dispatch time. The helper takes the hash as an input rather
    than recomputing it so the lock window stays minimal.

    Acquires _get_plan_lock(session.id) for the swap window. Inside
    the lock:
      1. Re-reads session.plan (linearisation point).
      2. Runs the §6 dedup check against identity_hash. If a prior
         discovered task with the same hash exists, returns
         (existing_plan, False) and the caller pins to the existing
         task.
      3. Otherwise builds the new Plan via add_tasks + bump_revision,
         swaps under channel_processor_active(), stamps the pin, and
         returns (new_plan, True).

    The paired PlanRevised + DriftDetected emit is scheduled
    off-lock as fire-and-forget; the snapshot it carries is captured
    inside the lock and cannot tear.

    Lock contract: single writer inside the lock, post-#403. See
    PLAN-DESCRIPTIVE-GROWTH.md §5 Option D for rationale.
    """
```

The helper is called directly from `_maybe_pin_delegation_task`. The
lock-acquisition decision is encoded by the helper, not the caller.

### 5.2 Test plan for the lock contract

Required tests (impl PR 2):

- **Concurrent refine + discovery race test** (analogous to #413's
  partial-apply window test): a `NEW_WORK_DISCOVERED`-synthesised
  growth racing with a planner-authored refine targeting the same
  prior revision_index lands one or the other deterministically. The
  loser observes the winner's plan as `prior` and re-validates.
  Assert: no torn read of `session.plan`, no clobbered discovery
  (the discovered task survives to the final plan), no half-revision
  visible to `_wait_plan_stable` callers (every observed plan has
  either revision N or N+1 — never an in-between).
- `_wait_plan_stable` callers crossing a discovery write observe
  either the pre-discovery or post-discovery plan — never a torn read.
- Dedup under concurrent discoveries: two delegations of the same
  `(agent, args-token-set)` arriving simultaneously produce one
  discovered task (the second pins to the first; no duplicate growth).
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
| 1 | `Task.discovered: bool` + `Task.discovery_identity_hash: str` fields + `Plan.validate` update (no rule change; just opaque metadata) + **`DelegationObserved.tool_args` proto extension** (canonical args representation on `goldfive.v1`, so PR 2's dedup hash is computed from observed-fact data, not a pin-time intercept — see §13) + state-store migration (proto regen + harmonograf-side ingestion of the new field + schema bump). Forward-compat: old events without `tool_args` default-empty; PR 2's dedup must tolerate `tool_args=None` and fall back to per-`(agent, task_id)` granularity as a coarser-but-still-useful dedup. | n/a — additive |
| 2 | `_maybe_pin_delegation_task` fallback to discovery + `PlanReviser.install_descriptive_growth` helper (§5 Option D, lock-acquiring) + §5.2 test plan + **§11.6 regression race test** (mandatory acceptance criterion) + tracking issue cross-link to #413 for the test template | `GOLDFIVE_PLAN_DESCRIPTIVE_GROWTH=1` |
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

### 11.1 Granularity: per-delegation vs per-(agent, args-token-set) — RESOLVED

**Resolution: per-(agent, args-token-set).**

If the coordinator delegates to `debugger_agent` 20 times in one run,
we synthesise **one** discovered task, not 20. The dedup key is the
`discovery_identity_hash(agent_name, tool_args)` defined in §4.3.0:
the first delegation mints the task; subsequent delegations whose
args-token-set hashes to the same value re-pin to the existing task.

**TTL.** The hash entry survives "until the discovered task reaches
a terminal status" so the plan does not grow unboundedly. Once
terminal, a fresh delegation with the same hash is a genuinely new
unit of work and grows the plan again.

**Rationale.**

- **UX.** Per-delegation would have grown the cherry-tree plan (§2.1)
  to >20 discovered tasks dominated by the debugger's retry storm,
  which is closer to noise than signal. Per-(agent, args-token-set)
  collapses to a small constant per behavior — measured on session
  `2d27ff4a`, the 10 observed delegations dedup to 5 unique
  `(agent, args-token-set)` tuples (research=1, web_developer=2,
  reviewer=1, debugger=1 — see the §2.1 validation note).
- **Drift signal preserved.** Per-call content-divergence drift
  (OFF_TOPIC, LOOPING_REASONING) still fires per-delegation against
  the reasoning content of each call; the dedup only collapses the
  structural growth event, not the observational stream. Operators
  still see every off-goal call in the drift timeline.
- **Consistency with refine cascade.** The hash carries across
  refines (§4.3.0 "Cross-refine survival"), so a planner that
  consolidates a discovered task into the forecast does not break
  dedup for subsequent delegations.

The implementation note for PR 2: the dedup check runs inside the
plan lock (§5 Option D) so two simultaneous delegations of the same
`(agent, args-token-set)` cannot both grow the plan — the second
reads the post-lock plan and finds the first's discovered task.

#### 11.1.1 Validation against real data

The dedup identity hash is computed from the `tool_args` carried on
the `DelegationObserved` proto — i.e., from observed-fact data the
agent itself authored — not from a goldfive-side intercept of agent
state at pin time. The PR 1 proto extension (§9) adds the
`tool_args` field to `DelegationObserved` so PR 2 can compute the
hash off the observed event rather than a captured snapshot. See
§13 for the underlying principle ("adaptive, not predictive") that
makes this the only correct choice.

The 2026-05-13 validation against session `2d27ff4a` measured 5
unique `(agent_name, args-token-set)` tuples across 10 delegations
— well within the small-constant target §11.1 calls for. The
`_normalize_args_tokens` design (§4.3.0) is good to lock in.

If a future investigation finds >5 unique tuples per agent on a
single run (i.e., the dedup is too fine), revisit the
`_normalize_args_tokens` design — broader normalisation (stemming,
synonym collapse) is the lever, not a coarser granularity choice.

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

### 11.6 PR 2 acceptance: regression race test for the lock contract

PR 2 MUST land a regression test analogous to **#413's** partial-apply
race test. The test asserts the §5 Option D lock contract under
concurrent refine + discovery pressure:

- **Setup.** Synthesise a session with a planner-authored plan at
  revision N. Launch two concurrent coroutines:
  1. A `Planner.refine` call that reads `session.plan`, computes a
     `CORRECT`-shaped revision, and calls `install_revision_for_drift`.
  2. A `_maybe_pin_delegation_task` call that misses tier 1/2 and
     falls through to `install_descriptive_growth`.
- **Assertions.**
  - No torn read: every observation of `session.plan` (via a third
    observer coroutine sampling at a high rate) carries either
    revision N or revision N+1 or revision N+2 — never an in-between
    state with a partially-applied diff.
  - No clobbered discovery: if the discovery write lands first, the
    refine's prior is the post-discovery plan, the discovered task
    survives into N+2.
  - No clobbered refine: if the refine lands first, the discovery
    write re-reads inside the lock, dedups if applicable, and grows
    against the refined plan; the refine's tasks survive into N+2.
  - `_wait_plan_stable` callers crossing either write observe a
    consistent plan (no half-revision).
  - Dedup linearisability: two concurrent discoveries with the same
    `discovery_identity_hash` produce one discovered task, not two.

The test is a hard acceptance criterion for PR 2 — without it, the
lock-contract claim in §5 is unenforced. Use #413's test fixture as
the template.

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
  contract §5 Option D explicitly aligns with: single writer,
  inside the lock, full stop).
- `#413` — Concurrent-refine regression test fixture (the template
  PR 2's §11.6 race test is built from).

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

## 13. Design principle: adaptive, not predictive

This proposal is one instance of a broader rule that constrains every
goldfive primitive that depends on an agent-derived value. Stated by
the maintainer during design review:

> "you cannot pin the args. generally, any design that tries to
> predict the agent behavior will be flawed because agents have
> agency, so anything predictive must be adaptive as well (or just
> adaptive). in this case, extend the proto."

**Goldfive observes; agents decide.** Any goldfive primitive that
depends on an agent-derived value — tool args, output shape,
delegation target, reasoning content — MUST source that value from
an observed event (proto-carried, span attribute, or session-state
write the agent itself authored), NOT from a goldfive interception
of agent state at pin / dispatch / wrap time. Predictive snapshots
are stale by construction under model nondeterminism: the agent may
revise the value between the snapshot and the dispatch, and the
goldfive primitive then operates on an obsolete copy. Predictive
must be adaptive (or just adaptive).

Concretely for this proposal: the dedup hash (§4.3.0) is computed
from `DelegationObserved.tool_args` carried on the observed event
(PR 1 proto extension, §9), not from a goldfive-side intercept of
`tool_args` at `_maybe_pin_delegation_task` entry. The agent
authored the event; goldfive reads it.

### 13.1 Precedents in goldfive

The principle is not new — it codifies a discipline already
visible elsewhere in the codebase:

- **`observed_revision_index` re-read on apply (#245).** The
  planner samples `session.plan.revision_index` at refine-decision
  time, but `install_revision_for_drift` re-reads the live
  `revision_index` inside the plan lock at apply time. The
  apply-time read is authoritative; the decision-time sample is
  advisory. A concurrent refine that landed first does not get
  clobbered by a stale predictive snapshot.
- **`_apply_revision` → `_emit_plan_revised` lock contract (#403
  / PR #413).** The partial-apply window was closed by moving the
  `set_session_plan` swap inside the plan lock, so readers
  observe either the pre- or post-revision plan but never a
  predicted-but-not-yet-installed shape. Same lesson: the
  authoritative state is what's been observed to land, not what
  was predicted to land.
- **`DelegationObserved` itself (#249 actor-model channel
  processor).** The event is named `Observed` deliberately —
  goldfive emits it from what ADK delivered, not from what
  goldfive predicted ADK would deliver.

These precedents demonstrate the rule works: every time goldfive
has been bitten by a state-correctness bug at scale, the fix has
been to read the agent's observed output instead of the
goldfive-side prediction. The proto extension in PR 1 is the
specific application of the rule to descriptive growth.
