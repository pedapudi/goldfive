# 11. State Ownership

## Read this chapter when...

- You are about to add a new piece of state to goldfive and do not know **where it belongs** (a `Session` field, a `goldfive.*` key on `Session.state`, an ADK `session.state` key, a `StateStore` registry, or a ContextVar). Read [Where to add new state](#where-to-add-new-state-decision-tree) first — putting it in the wrong place is the single most common state bug.
- You touched `goldfive/types.py`'s `Session` dataclass, `goldfive/state_store.py`, `goldfive/adapters/_adk_state_protocol.py`, or `goldfive/_state_audit.py`.
- A value you wrote from an ADK callback is invisible to a later read (the classic "I set it, why is it empty?" bug — see [The shallow-copy handoff hazard](#the-shallow-copy-handoff-hazard-the-8-hour-lesson)).
- A `StateOwnershipViolation` or `PlanOwnershipViolation` fired in the test suite and you need to understand what the tripwire is protecting.
- You are debugging a lifecycle gate (drift condition, steer freshness, retry budget) that "never engages" or "engages twice" — the root cause is almost always a key-identity or reset-point problem covered here.

## Files covered

| File | What it owns |
|---|---|
| `goldfive/types.py` | The `Session` dataclass (every typed field), the `set_session_plan` single-writer guard, the `_CHANNEL_PROCESSOR_ACTIVE` ContextVar. |
| `goldfive/state_store.py` | The `goldfive.*` key namespace on **goldfive** `Session.state`; the `StateStore` typed handle; the `Drift` condition dataclass + lifecycle helpers; the process-wide invocation/cancel/supersede registries. |
| `goldfive/adapters/_adk_state_protocol.py` | The `goldfive.*` key namespace on **ADK** `session.state` (the LLM-facing bridge); the cooperative-cancellation and invocation-parent helpers. |
| `goldfive/_state_audit.py` | The runtime tripwire that forbids new callback-time writes to ADK `session.state`. |
| `goldfive/_llm.py` | The per-call `LlmCallDiagnostics` ContextVar (`#491`) plus the `MAX_OUTPUT_TOKENS_VAR` / `THINKING_DISABLED_VAR` scoping vars. |
| `goldfive/adapters/_adk_plugin.py` | The `SessionContext` handoff (`_active_ctx`) and the plugin-instance dicts (`_cancel_state`, `_active_ctx`). |

Design doc: `docs/design/STATE-OWNERSHIP-CONTRACT.md`. **The doc is partly stale on module names** — it names `goldfive/orchestration_state.py` and `goldfive.orchestration_store`, which were **merged into `goldfive/state_store.py`** (imported everywhere as `_ostate`). Where the doc and the code disagree, the code wins; this chapter reconciles them.

## Invariants that bind you here

These are the state-ownership-specific projections of the global CANON invariants. Violating any of them is a defect, not a style choice.

1. **One writer per datum.** Every piece of state has exactly one component that may write it. Readers are unrestricted; writers are named. If you find yourself writing a field from two places, you have a design bug — consolidate the writer or split the datum.
2. **goldfive callbacks MUST NOT write ADK `session.state`.** From inside any ADK `before_*` / `after_*` callback, goldfive code may **read** `callback_context.state` / `tool_context.state` / `invocation_context.session.state` freely but must **not** mutate them by subscript / `pop` / `update` / `setdefault`. The `_state_audit.py` tripwire enforces this in tests and CI.
3. **`Session.plan` has exactly one writer: the steerer's channel processor.** Swap it only through `set_session_plan` (`goldfive/types.py`) inside a `channel_processor_active()` region. See [09-steering-ladder-and-gates.md](09-steering-ladder-and-gates.md) and [10-planning-and-revision.md](10-planning-and-revision.md).
4. **`observation_only=True` is strictly passive.** The only sanctioned read of the kill-switch is `DefaultSteerer.is_active_steering()` / the module helper `steering_is_active(steerer)` (`goldfive/steerer.py`). Never read `_observation_only` directly and never gate a *write* to shared state on your own re-derivation of the flag. See [09-steering-ladder-and-gates.md](09-steering-ladder-and-gates.md).
5. **Lifecycle gates need stable identity keys.** A gate keyed on a churning id (an LLM-minted id, a per-observation counter) opens a fresh entry per observation and never engages. Key on the claim's identity (kind + task + agent + turn), never on a value that changes every write. This is why `Drift.condition_id` is a hash of a stable tuple, not a fresh uuid.
6. **Values under `goldfive.*` must round-trip through a JSON sink.** `state_store.write` refuses non-`goldfive.*` keys, and every value stored under the prefix must be JSON-serialisable so sinks can persist the state dict. Non-serialisable objects (an `asyncio.Task`, a live `asyncio.Event`) go on a `Session` field or a module registry, never under the prefix.

---

## The state surfaces at a glance

`docs/design/STATE-OWNERSHIP-CONTRACT.md` §2 catalogues "four surfaces". That count is from Phase 0 and is now an undercount — the live code has **seven** physically-distinct places state can live. You must be able to name all seven and pick the right one. They are, from most-preferred to most-hazardous:

| # | Surface | Backing store | Owner / single writer | Lifetime | Serialisable? |
|---|---|---|---|---|---|
| 1 | **`Session` typed fields** | attributes on the `Session` dataclass instance | the component named in each field's docstring | one `Runner.run()` turn | field-by-field |
| 2 | **goldfive `Session.state['goldfive.*']`** | a plain `dict` on `Session.state` | `goldfive/state_store.py` helpers | one turn (fresh dict per `Session`) | **required** |
| 3 | **StateStore module registries** | `_ACTIVE_INVOCATION_TASKS` etc., keyed by `session.id` | `StateStore` methods only | one turn, wiped by `clear_active_invocations` | **no** (holds live tasks) |
| 4 | **ADK `session.state['goldfive.*']`** | ADK's `SessionService`-managed dict | **ADK** (via `append_event(state_delta=...)`) | across turns, persisted by ADK | ADK's problem |
| 5 | **Plugin-instance dicts** | `self._cancel_state`, `self._active_ctx` on `_GoldfiveADKPlugin` | plugin internals | one plugin instance (≈ one Runner) | no |
| 6 | **ContextVars** | `contextvars.ContextVar` objects | the `@contextmanager` that sets them | one dynamic call scope | no |
| 7 | **Dynamically-stamped private attrs** | `session._supersede_pending` etc., *not* declared fields | ad-hoc | one turn | no — **HAZARD** |

Surfaces 1–3 are goldfive's own and are where **almost all** new state belongs. Surface 4 is ADK's and is **read-only from goldfive callbacks**. Surface 7 is a legacy wart you must not extend.

The two surfaces that look identical at a call site — both end in `.state` — are #2 (goldfive's dict) and #4 (ADK's dict). Confusing them is the root cause of goldfive#275. The mnemonic:

- `session.state[...]` where `session` is a `goldfive.types.Session` → **surface #2, goldfive-owned, write freely.**
- `callback_context.state[...]` / `tool_context.state[...]` / `invocation_context.session.state[...]` inside an ADK callback → **surface #4, ADK-owned, READ ONLY.**

### Who writes what (component → surface matrix)

The single-writer invariant means every component has a small, fixed set of surfaces it may write. Use this as a quick sanity check: if your edit has a component writing a surface it does not own here, stop and reconsider.

| Component | Writes | Reads |
|---|---|---|
| `Runner` (`runner.py`) | surface #1 (plan seed), #2 (`set_current_plan`, `refresh_goals_summary`, `clear_active_steer`) | all |
| `DefaultSteerer` (`steerer.py`) | #1 (`refine_outcomes`, `last_addressed_revision_by_drift_key`, `pending_nudges`), #2 (active steer, drift conditions, reasoning bindings), #3 (cancel/supersede marks), #1 `plan` via `set_session_plan` | all |
| `DriftObserver` (`drift_observer.py`) | #1 (`drift_events`, `last_observed_event_at`, counters), #2 (drift conditions) | all |
| `PlanReconciler` / task state machine | #1 (`current_task_id`, progress, notes), #2 (`sync_current_task_from_transition`) | plan, state |
| ADK plugin (`_adk_plugin.py`) | #1 (`current_agent_id`), #3 (invocation task register/deregister), #5 (`_active_ctx`, `_cancel_state`) | all |
| Reporting handlers (`reporting/`) | #1 (`completed_results`, `task_progress`, `agent_notes`, `pending_approvals`), #2 (`rotate_current_task_id`) | state |
| `_correction_injection.py` | #2 (`goldfive.pending_corrections.*`) | state |
| Executors (`executors/`) | #1 (`pending_nudges_revision_installed` drain, surface-#7 legacy flags), #3 (supersede clear) | all |
| Sinks / judges | **nothing** (read-only by contract) | plan, drift stream, state |

The last row is load-bearing: **sinks and judges never write shared state.** A judge that needs to remember something across calls uses a ContextVar (surface #6) scoped to its own dispatch, or returns a verdict the steerer records — it does not stamp `Session.state`.

---

## Surface 1: the `Session` dataclass

`Session` (`goldfive/types.py`, `@dataclasses.dataclass class Session`) is the live state for **one `Runner.run()` turn**. A fresh `Session` is minted per turn by `Conversation.next_turn_session` (`goldfive/conversation.py`); a handful of fields are *copied* forward from the `Conversation` (goals, completed results, the sequence cursor) and everything else starts at its field default. That construction site **is the reset point** for every non-copied field: if a field is not one of the four seeded in `next_turn_session`, it resets to its `default` / `default_factory` at the start of every turn.

```python
# goldfive/conversation.py — Conversation.next_turn_session
return Session(
    run_id=uuid.uuid4().hex,
    conversation_id=self.id,
    started_at_ms=_now_ms(),
    goals=list(self.goals),                     # copied forward
    completed_results=dict(self.completed_results),   # copied forward
    completed_outputs=dict(self.completed_outputs),   # copied forward
    _next_sequence=self._next_sequence,         # seeded from conversation cursor
)
```

### Field-by-field table

Every field on `Session`, its writer(s), its readers, and where it resets. "Writer" is the component that owns the write per invariant #1. Bare `Session(run_id=...)` construction (tests, programmatic use) starts every field at its default; the "Reset point" column is where the field returns to default *within a live Conversation*.

| Field | Type / default | Writer (owner) | Primary readers | Reset point |
|---|---|---|---|---|
| `run_id` | `str` (required) | constructor | everyone; aliased by `Session.id` property | never (identity) |
| `conversation_id` | `str = ""` | constructor | sinks, harmonograf routing | never |
| `goals` | `list[Goal]` | `goal_deriver`, USER_STEER path | planner, goal-drift judge | copied-forward each turn |
| `plan` | `Plan \| None = None` | **steerer channel processor only**, via `set_session_plan` | executors, reconciler, planner, sinks | seeded from prior turn's stash (`Conversation.prior_plan_for`) |
| `current_task_id` | `str = ""` | reconciler / task state machine | executors, drift attribution | fresh Session |
| `completed_results` | `dict[str,str]` | reporting handlers (agent summary) | planner, results assembly | copied-forward |
| `completed_outputs` | `dict[str,str]` | adapter (full assistant text per task) | evaluators, graders (zicato#12) | copied-forward |
| `task_progress` | `dict[str,float]` | reporting handlers, task machine | drift stall gate | fresh Session |
| `agent_notes` | `dict[str,str]` | reporting handlers | planner, sinks | fresh Session |
| `divergence_flag` | `bool = False` | agent-side write via state protocol | steerer | fresh Session |
| `started_at_ms` | `int = 0` | constructor | duration metrics | fresh Session |
| `pending_approvals` | `dict[str, asyncio.Event]` | reporting/approval registration | control dispatcher | fresh Session |
| `pending_approvals_meta` | `dict[str, dict]` | approval registration + dispatcher | approval waiters | fresh Session |
| `reasoning_history` | `list[str]` (bounded by `reasoning_history_max`) | adapter `emit_reasoning` hook | reasoning-drift detectors | fresh Session |
| `reasoning_history_max` | `int = 20` | config | trim logic | fresh Session |
| `reasoning_cluster_flagged` | `set[str]` | reasoning-similarity ladder | same (one-shot dedup) | fresh Session |
| `reasoning_loop_flagged` | `set[str]` | reserved (cliff detector pair) | — | fresh Session |
| `refine_outcomes` | `dict[tuple[str,str], RefineOutcome]` | `DriftObserver._handle_drift_dispatch` / `_apply_revision` | intervention ladder, REPEATED_FAILURE | Fresh `Session` per turn (`Conversation.next_turn_session` does not copy it forward); `DriftObserver.reset_for_turn` is a no-op on the default path |
| `last_addressed_revision_by_drift_key` | `dict[tuple[str,str], int]` | `_apply_revision` (non-user drifts) | verdict-freshness gate in `_handle_drift` | fresh Session |
| `task_last_progress_at` | `dict[str,float]` (`time.monotonic()`) | `mark_task_running` / `mark_task_progress` / `_emit_task_transitioned` | steerer stall-escalation gate | fresh Session |
| `last_observed_event_at` | `float = 0.0` (`time.monotonic()`) | `DriftObserver._touch_liveness` (every observe path) | stall watchdog `#487` | fresh Session |
| `_llm_calls_since_check` | `int = 0` | `note_llm_call` | reflective self-progress check | check-run or task transition |
| `_reflective_check_task_id` | `str = ""` | reflective-check path | counter-reset guard | task transition |
| `_agent_turns_since_goal_check` | `int = 0` | `note_agent_turn` | goal-drift trajectory judge | check-fire (not task transition) |
| `recent_events` | `list[dict]` (per-kind trimmed) | `note_agent_activity` / `note_tool_observation` | goal-drift + reasoning judges | fresh Session |
| `drift_events` | `list[DriftEvent]` (unbounded) | `DriftObserver._emit_drift_detected` | `drift_summary` property, zicato harnesses | fresh Session |
| `_last_goal_drift_check_ts` | `float = 0.0` (`time.time()`) | task-boundary goal-drift trigger | rate limit | fresh Session |
| `_reasoning_judge_counters` | `dict[tuple[str,str], int]` | reasoning-judge scheduler | judge cadence (`#226`) | fresh Session |
| `pending_nudges` | `list[str]` | `DefaultSteerer` (Level 2 NUDGE) | Runner overlay drain | consumed by drain |
| `pending_nudges_revision_installed` | `bool = False` `#475` | `DriftObserver` (post-ABSORB `was_installed=True`) | overlay replay framing header | consumed by drain (`sequential.py`) |
| `state` | `dict[str, Any]` | `state_store.py` helpers (surface #2) | any goldfive component | fresh dict each Session |
| `current_agent_id` | `str = ""` | ADK plugin `before_agent_callback` (last-writer-wins) | reasoning judge attribution | fresh Session |
| `task_lineage` | `dict[str, set[str]]` | task machine RUNNING + delegation observed | reasoning judge (child-of-delegation) | terminal transition clears per-task |
| `recent_tool_observations_max` | `int = 16` | config | per-kind trim | fresh Session |
| `_next_sequence` | `int = 0` | `next_sequence()` | sinks (`Event.sequence`) | seeded from Conversation cursor |
| `_reasoning_turn` | `int = 0` | `mark_reasoning_turn()` | steer-freshness window (`#441`) | fresh Session (0) |

### Key methods on `Session`

These live on the dataclass and are the **only** blessed way to advance the two counters and mint event ids. Do not increment `_next_sequence` / `_reasoning_turn` by hand.

```python
# goldfive/types.py — Session
def next_sequence(self) -> int:
    s = self._next_sequence
    self._next_sequence = s + 1
    return s

def mark_reasoning_turn(self) -> int:
    self._reasoning_turn += 1
    return self._reasoning_turn

def next_sequence_and_event_id(self) -> tuple[int, str]:
    seq = self.next_sequence()
    return seq, self.next_event_id(seq)

def next_event_id(self, sequence: int | None = None) -> str:
    # format: "{run_id}:{sequence}:{uuid4_short}"
    ...
```

- `next_sequence() -> int` — **post-increment** monotonic wire-event counter. Use `next_sequence_and_event_id()` at new emit sites so `Event.sequence` and `Event.event_id` agree without double-incrementing (goldfive#271 Phase 3 Addition B).
- `mark_reasoning_turn() -> int` — **pre-increment** logical-turn counter; called **once per reasoning observation** by `DriftObserver.observe_reasoning`. Deliberately distinct from `_next_sequence` (which counts *every* emitted event, inflated by decision-telemetry). The steer-freshness window measures in `_reasoning_turn`, not `_next_sequence` (goldfive#441).
- `next_event_id(sequence=None) -> str` — mints `{run_id}:{sequence}:{uuid4_short}`. The `(run_id, sequence)` prefix preserves chronological sortability; the uuid4 suffix guarantees PK uniqueness even when an outer system collapses multiple Sessions onto the same outer-session id (harmonograf#61's outer-session pin restarts `_next_sequence` at 0 per turn). Passing an already-pulled `sequence` avoids double-incrementing.
- `id` (property) → aliases `run_id`. A goldfive `Session` maps 1:1 to a run/turn; `run_id` is its identity and the `Event.session_id` stamp.
- `drift_summary` (property) → O(N) snapshot of `drift_events` at call time; frozen `DriftSummary` (severity weights 1/3/10 for INFO/WARNING/CRITICAL); does not update in place — re-read to see new drifts.

### The `RefineOutcome` / verdict-freshness pair

Two fields work together to prevent (a) infinite refine loops and (b) over-rejection of orthogonal judge verdicts. Both are on `Session`; both are the subject of stable-key discipline.

- `refine_outcomes: dict[tuple[str, str], RefineOutcome]` — keyed by `(drift.kind.value, task_id)`. Each `RefineOutcome` (`goldfive/types.py`) carries `state` (`"succeeded"` / `"failed"`) and `fail_count`. The intervention ladder consumes it via `DriftObserver._occurrence_count_for_ladder`, which maps `"succeeded"` back to `0` (a fresh same-(kind, task) drift is treated as first-occurrence) and `"failed"` to the accumulated `fail_count`. **Reset every turn** by the fresh `Session` that `Conversation.next_turn_session` mints — it does **not** copy `refine_outcomes` forward (`default_factory=dict`), so a wedged drift from a prior turn cannot carry its failure count into a fresh refine attempt. `Runner.run` also *attempts* `DriftObserver.reset_for_turn` via `getattr(self.steerer, "reset_for_turn", None)` right after `run_started`, but that resolves to `None` for the default steerer (the method lives on `DriftObserver`, reached as `steerer.drift.reset_for_turn`), so it only matters when a `Session` is reused across `Runner.run` calls. The field is lock-free single-writer-per-session; ADK's per-session callback serialisation means concurrent writes for the same Session do not arise.
- `last_addressed_revision_by_drift_key: dict[tuple[str, str], int]` — keyed by `(drift.kind.value, drift.current_task_id or "")`. The per-`(kind, target)` watermark of the most recent `revision_index` that successfully addressed a drift of that shape. The verdict-freshness gate in `DriftObserver.handle_drift` compares an observed verdict's `observed_revision_index` against this watermark to detect **redundant** verdicts only. A naive `observed_revision_index < live_revision_index` would over-reject parallel judges firing on orthogonal concerns (a GOAL_DRIFT verdict observed at revision N is not invalidated by a later OFF_TOPIC refine that produced N+1 — distinct claims). Stamped in `_apply_revision` for **non-user drifts only** (user-authored drifts bypass the gate). This is the `#245` fix.

### The `recent_events` ring buffer (`#239`)

`recent_events: list[dict]` is a **unified** ring buffer that replaced two earlier split fields (`recent_agent_activity` fed to the goal-drift judge, `recent_tool_observations` fed to the reasoning judge). Each entry is a small framework-neutral dict with a `kind` discriminator — currently `agent_invocation_started` / `agent_invocation_completed` (old agent-activity entries) and `tool_observed` (old tool-observation entries). Two subtleties matter for anyone touching it:

- **Per-kind trimming.** The writers (`DriftObserver.note_agent_activity`, `note_tool_observation`) trim **per kind-class**, so a flood of tool observations cannot evict agent-activity entries and vice versa. The tool-observed cap is `Session.recent_tool_observations_max` (default 16, `#239`). If you add a new `kind`, add its own cap — do not let it share another kind's budget.
- **Read by filtering, not by slot.** Consumers recover the equivalent of the old buffers via `filter_recent_events_by_kind(session.recent_events, kind)` (`goldfive/types.py:1612`). Read this helper, never index the list positionally.

This merge is a state-consolidation model: two parallel buffers with the same lifetime and near-identical shape collapse onto one list with a discriminator, cutting the number of fields (and thus reset points) a reader must track.

### The approval-waiter state (`pending_approvals` / `pending_approvals_meta`)

Human-in-the-loop approval is a two-field pair on `Session`, and a good example of when a live `asyncio.Event` legitimately lives on a `Session` field rather than a registry (it is small and per-turn):

- `pending_approvals: dict[str, asyncio.Event]` — keyed by `target_id`: the **task_id** for Flow A (`report_awaiting_approval`) and the **ADK `function_call_id`** for Flow B (ADK `require_confirmation`). The waiter registers the `Event`; the control dispatcher sets it when `APPROVE`/`REJECT` arrives.
- `pending_approvals_meta: dict[str, dict]` — per-approval metadata; the dispatcher adds `decision` (`"approve"`/`"reject"`) and optional `detail` **before** setting the event, so the woken waiter reads the decision without a second round-trip.

`#478` hardened the Flow-A path so `report_awaiting_approval` **never hangs**: with no control channel wired it acknowledges immediately as `unavailable`; with a channel it uses a finite default timeout (600s); on expiry it emits `HUMAN_INTERVENTION_REQUIRED`. Under `observation_only=True` the ack strips `plan_state`. The state *shape* is unchanged by `#478` — only the timeout/no-channel behaviour around it. See [13-reporting-tools-and-approval.md](13-reporting-tools-and-approval.md) for the dispatch detail.

Because the `Event` is non-serialisable, this pair is exempt from invariant #6 by living on a `Session` field (surface #1), not under `goldfive.*` (surface #2) — the decision tree's step-1 exception.

### Why `_reasoning_turn` and `_next_sequence` are two counters

This is a canonical example of invariant #5 (stable identity) and worth internalising. `_next_sequence` ticks on *every emitted event*. The decision-telemetry events added in `#436`/`#440`/`#480` emit several events per turn, so a freshness window measured in `_next_sequence` would shrink unpredictably as telemetry volume changed. `_reasoning_turn` ticks once per reasoning observation — a stable, semantically-meaningful "turn" — so the USER_STEER suppression window is invariant to observability volume. **When you add a gate that means "N turns ago", key it on `_reasoning_turn`, never `_next_sequence`.**

### The dynamically-stamped private attrs (surface #7 — HAZARD)

Four attributes are stamped onto `Session` at runtime but are **not declared dataclass fields**. Grep `goldfive/types.py` for them and you find nothing but comments. They are:

| Attr | Stamped in | Read in | What it is |
|---|---|---|---|
| `session._supersede_pending` | `drift_observer.py:5293`, `executors/sequential.py` | `executors/sequential.py:1258` | legacy bool marking an in-flight internal supersede-cancel |
| `session._last_cancel_reason_prefix` | `drift_observer.py:5514`, `executors/{parallel,sequential}.py` | executor cancel-branch framing | last cancel reason prefix for overlay replay |
| `session._intercept_transfer` | `executors/_control.py:385` | control path | flag toggling transfer interception |
| `session._conversational_turn` | `runner.py:950` | runner overlay | marks a conversational (non-plan) turn |

Every write carries `# type: ignore[attr-defined]` or relies on Python's permissive attribute assignment. These exist for historical reasons and are actively being migrated off:

- `_supersede_pending` is the **legacy** signal. Its replacement is the per-invocation `_SUPERSEDE_PENDING_INVOCATIONS` registry on `StateStore` (`mark_supersede_pending` / `is_supersede_pending` / `has_any_supersede_pending`). The bool is kept **only** for back-compat with `tests/test_executor_supersede_cancel_nonfatal.py` and the empty-resolver fallback in `DriftObserver._cancel_inflight_for_revision` (no invocation id to anchor a registry entry). Issue **#430** tracks retiring the bool entirely. Readers should **prefer the registry when they hold an `invocation_id`** and fall back to the bool only otherwise.

**Rule: do not add a fifth dynamically-stamped attr.** If you need per-turn scratch state, add a declared field with a default (so its type, reset point, and default are documented in one place) or, for non-serialisable per-session data keyed by invocation, a `StateStore` registry. A dynamically-stamped attr has no default, no documented reset point, no type, and is invisible to anyone reading the dataclass — it is exactly the kind of "parallel-tracked indirection" that caused the brussels-sprouts/tomato false-positive cascades called out in the `pending_nudges` field comment.

### The per-turn state lifecycle (a timeline)

Understanding **when** each surface resets is half of understanding state ownership. Here is the order of operations for one turn inside a live `Conversation` (see [03-runner-and-conversation.md](03-runner-and-conversation.md) for the Runner detail):

1. **`Conversation.next_turn_session()`** mints a fresh `Session`. Surface #1 fields reset to defaults *except* `goals`/`completed_results`/`completed_outputs`/`_next_sequence` (copied forward). Surface #2 (`Session.state`) is a **brand-new empty dict** — every `goldfive.*` key starts absent.
2. **`Runner.run`** seeds `session.plan` from the prior turn's stash (`Conversation.prior_plan_for`) when carry-forward applies, then stamps orchestration state: `_ostate.set_current_plan(session.state, session.plan)` (`runner.py:666`) and `_ostate.refresh_goals_summary(session.state, session.goals)` (`runner.py:722`).
3. **`Runner.run` fires `run_started`**, then attempts a per-turn refine-outcome reset via `getattr(self.steerer, "reset_for_turn", None)` (runner.py). Note: `reset_for_turn` is a method on **`DriftObserver`** (reachable as `steerer.drift.reset_for_turn`), *not* on `DefaultSteerer`, so this getattr resolves to `None` and is a no-op for the default steerer. In a live `Conversation` this does not matter: `refine_outcomes` starts empty every turn regardless, because `Conversation.next_turn_session` mints a fresh `Session` and does **not** copy `refine_outcomes` forward (`default_factory=dict`). The fresh Session is the real reset point; `DriftObserver.reset_for_turn` only matters when the same `Session` object is reused across `Runner.run` calls, and today it is invoked explicitly only in tests (`steerer.drift.reset_for_turn(session)`).
4. **`Runner.run` clears the active steer**: `_ostate.clear_active_steer(session.state)` (`runner.py:1163`) so a steer from a prior turn does not leak into this one's suppression logic.
5. **The adapter dispatch runs.** `set_active_context` sets `plugin._active_ctx`; the boundary wrapper registers the invocation task on surface #3; callbacks read goldfive state via `session_context_from_invocation`; the reconciler / task machine stamp `current_task_id` and drift conditions accumulate on `KEY_ACTIVE_DRIFTS`.
6. **Teardown.** The adapter calls `StateStore.clear_active_invocations()` — wiping all three surface-#3 registries for the session. The Runner's `finally` stashes the plan for the next turn (`Conversation.stash_plan`).

The single most important consequence: **surface #2 does not persist across turns.** If you need a `goldfive.*` value to survive into the next turn, either promote it to a copied-forward `Session` field or re-seed it at step 2, exactly as `set_current_plan` / `refresh_goals_summary` do. Silently relying on a `goldfive.*` key still being present next turn is a bug.

---

## Surface 2: goldfive `Session.state['goldfive.*']` (`state_store.py`)

`Session.state` is a plain `dict[str, Any]`, **owned by goldfive**, that any goldfive component may read and write. `goldfive/state_store.py` is the **single source of truth** for the `goldfive.*` key namespace on it. The module offers two co-equal APIs over the same dict:

1. **Module-level free functions** — `state_store.set_current_plan(session.state, plan)`, `state_store.read(...)`, etc. Many call sites use these directly (imported as `_ostate`). They are the primitives.
2. **The `StateStore` typed handle** — `StateStore.for_session(session)` returns an object that groups every read/write so the call site doesn't string-fish. The handle's methods route through the same free functions, so the `goldfive.*`-prefix assertion still fires.

Both hit the identical dict. Use whichever is ergonomic; new code tends to prefer the handle for reads and the free functions for the occasional write.

> **This is NOT the ADK `session.state`.** The module docstring says so in bold. Surface #2 is a goldfive-internal, framework-agnostic orchestration convention. Surface #4 (`_adk_state_protocol.py`) is the LLM-facing bridge. They share *string values* for a few keys (e.g. `"goldfive.active_steer.body"`) deliberately — same logical field, two readers — but they are two physically-distinct dicts.

### The write/clear/read primitives

```python
# goldfive/state_store.py
GOLDFIVE_PREFIX = "goldfive."

def write(state, key, value):   # refuses non-goldfive.* keys (ValueError)
    _assert_goldfive_key(key)
    state[key] = value

def clear(state, key):          # state.pop(key, None), also prefix-guarded
    _assert_goldfive_key(key)
    state.pop(key, None)

def read(state, key, default=None):   # tolerant: non-Mapping / missing / None → default
    ...
```

`write` and `clear` **raise `ValueError` on a non-`goldfive.*` key**. This is deliberate (invariant #6): a mis-namespaced write is almost always a typo or a leaky abstraction reaching into application state. `read` is maximally tolerant — a non-Mapping `state`, a missing key, or a `None` value all return the `default` without raising.

### Full key inventory

Every key constant lives in `state_store.py`. `ALL_KEYS` lists the stable core; a few keys (delegation pins, reasoning bindings) are defined as constants but intentionally excluded from `ALL_KEYS` because they are compound-shaped maps rather than scalar slots.

| Constant | Key string | Value shape | Writer | Readers |
|---|---|---|---|---|
| `KEY_CURRENT_PLAN_ID` | `goldfive.current_plan_id` | `str` | `set_current_plan` | plan-lifecycle consumers |
| `KEY_CURRENT_TASK_ID` | `goldfive.current_task_id` | `str` | `set_current_task` / `rotate_current_task_id` / `StateStore.set_pin_current_task` | pin ladder, reporting, resolver |
| `KEY_CURRENT_TASK_TITLE` | `goldfive.current_task_title` | `str` | `set_current_task` | prompt templates |
| `KEY_CURRENT_TASK_REVISION` | `goldfive.current_task_revision` | `int` | `stamp_current_task_revision` | report-time stale-pin classifier (`#266`) |
| `KEY_GOALS_SUMMARY` | `goldfive.goals_summary` | `str` (pre-formatted) | `refresh_goals_summary` | planner instruction block |
| `KEY_ACTIVE_STEER_BODY` | `goldfive.active_steer.body` | `str` | `set_active_steer` | drift suppression, prompt shaping |
| `KEY_ACTIVE_STEER_AT_TURN` | `goldfive.active_steer.at_turn` | `int` (a `_reasoning_turn` value) | `set_active_steer` | steer-freshness compare |
| `KEY_ACTIVE_STEER_AUTHOR` | `goldfive.active_steer.author` | `str` | `set_active_steer` | annotation attribution (`#171`) |
| `KEY_ACTIVE_STEER_SOURCE` | `goldfive.active_steer.source` | `"user"`\|`"goldfive"`\|`""` | `set_active_steer` | suppression policy |
| `KEY_PROCESSED_STEER_IDS` | `goldfive.processed_steer_ids` | `list[str]` (FIFO, cap 256) | `record_processed_steer_id` | USER_STEER idempotency (`#171`) |
| `KEY_CANCELLED_FUNCTION_CALL_IDS` | `goldfive.cancelled_function_call_ids` | `list[str]` (append-only, deduped) | `append_cancelled_function_call_ids` | prompt templates, refine paths |
| `KEY_ACTIVE_DRIFTS` | `goldfive.active_drifts` | `dict[condition_id, Drift-as-dict]` | drift lifecycle helpers | steerer wire emit, other components |
| `REASONING_BINDINGS_KEY` | `goldfive.reasoning_extracted_bindings` | `dict[agent_name, ReasoningBinding-as-dict]` | `record_reasoning_extracted_binding` | pin ladder signal 6 |
| `PENDING_DELEGATIONS_KEY` | `goldfive.pending_delegations` | `dict[function_call_id, {task_id, revision, tool_args?}]` | `set_pending_delegation` | pin ladder signals 1/3, reporting |

The `KEY_ACTIVE_STEER_AT_TURN` value is a **`_reasoning_turn`** snapshot, not a wall-clock or a `_next_sequence`. That is the stable-key discipline (invariant #5) in action: consumers ask "is this steer fresh?" by comparing the stored `at_turn` against the current `_reasoning_turn`.

### Plan / task helpers (module level)

- `set_current_plan(state, plan)` — stamps `KEY_CURRENT_PLAN_ID` or clears when `plan is None`.
- `set_current_task(state, task)` / `clear_current_task(state)` — stamp/clear the id+title (+ revision on clear).
- `stamp_current_task_revision(state, revision)` — companion to `set_current_task` for pin-versioning (`#266`); clamps negatives to 0, coerces via `int()`.
- `read_current_task_revision(state) -> int` — tolerant read, default 0 (missing → treated as "matches initial `revision_index=0`").
- `rotate_current_task_id(state, plan, agent_name) -> str | None` — after a **terminal** transition, advances the pin to the *single* remaining PENDING/RUNNING task for the agent, or clears it (zero or ambiguous-multiple candidates). Keeps the pin pointed at work still to do so subsequent reporting-tool calls can fall back to it instead of failing `missing_task_id`. **Never raises** — degenerate input clears and returns `None`.
- `sync_current_task_from_transition(state, task, to)` — called by the task state machine on **every** transition: RUNNING stamps id+title; terminal statuses (COMPLETED/FAILED/CANCELLED/NOT_NEEDED) clear **only if `task` is the currently-pinned one** (another task may have opened before this terminal write — don't steal its stamp).

### Goals summary helpers

- `format_goals_summary(goals) -> str` — one-per-line `- [gid] summary`; empty/None → the literal `"(no goals)"` so prompt templates interpolate unconditionally.
- `refresh_goals_summary(state, goals)` — recompute + stamp `KEY_GOALS_SUMMARY`. Called whenever goals change (USER_STEER path, goal-aware refine).

### Processed-steer-id idempotency (`#171`)

- `has_processed_steer_id(state, steer_id) -> bool` and `record_processed_steer_id(state, steer_id)` — a bounded FIFO dedup set (`PROCESSED_STEER_IDS_CAP = 256`) so a delivery retry or UI double-fire of the same STEER annotation doesn't cascade-cancel + refine twice. `record_` is safe to call unconditionally after a `has_` check (duplicates are dropped).

### Cancelled-function-call ids (heal path)

- `append_cancelled_function_call_ids(state, call_ids)` / `read_cancelled_function_call_ids(state)` — append-only, order-preserving, deduped list. Written by the adapter's `_heal_pending_tool_calls` when an invocation cancels mid-flight with pending tool calls; downstream prompt templates reference "the cancelled tool call" without poking adapter internals.

### Pending corrections (the full-agent-path key, `#479`)

Correction injection (`goldfive/_correction_injection.py`, "Stream D") writes per-`(agent, task)` correction bodies under a compound key so a correction targeted at one agent/task pair does not leak into another agent's prompt:

```python
# goldfive/_correction_injection.py — pending_correction_key
def pending_correction_key(agent_name: str, task_id: str) -> str:
    return f"{_sp.KEY_PENDING_CORRECTIONS}.{agent_name}.{task_id}"
    # e.g. "goldfive.pending_corrections.researcher.task_3"
```

`#479` changed the key to use the **full agent path** (not a bare leaf name) so two agents with the same leaf name under different coordinators do not collide. Stream D owns the writer (`write_correction`, which stamps `state[key] = dict(correction)`); the reader side is `StateStore.get_correction(agent, task)` / `has_correction(...)` / `iter_corrections_for_agent(agent)`, all of which strip a compound `client:agent` prefix so a compound-named caller still finds a writer's bare-form key. This is one of the §5.5 "already-compliant" writes: it targets goldfive `Session.state`, never ADK state.

---

## The `StateStore` typed handle

`StateStore` (`state_store.py`, `class StateStore`) wraps the same dict with typed accessors so call sites read `store.pin_current_task()` instead of `state.get("goldfive.current_task_id", "")`. It is a **view**, not an owner — it does not own the dict, and concurrent writers can mutate it between calls. Callers needing a snapshot must call once and cache.

### Construction

```python
StateStore.for_session(session)   # backed by session.state; keys the registries by session.id
StateStore.for_state(state, session_id="")   # backed by an arbitrary dict (tests, callbacks)
```

`for_session(None)` yields an empty-state store whose **writes are silently dropped** — so defensive paths inside ADK callbacks never raise. The `session_id` matters only for the module-level registries (see below); every read/write path over the dict works with an empty `session_id`.

`__init__` coerces a non-`Mapping` argument to `{}`. Write methods additionally guard `isinstance(self._state, dict)` and **silently no-op** on a read-only `MappingProxyType` — production `Session.state` is always a mutable `dict`; tests sometimes pass a proxy snapshot.

### Read accessors (typed views)

- `pin_current_task() -> str`, `pin_current_task_title() -> str`, `pin_current_task_revision() -> int`.
- `get_active_steer() -> ActiveSteer | None` — returns `None` when the body is empty (the canonical "no steer" signal), so `if store.get_active_steer():` works without re-checking `.body`. `ActiveSteer` is a frozen dataclass with `body`/`at_turn`/`author`/`source` and `is_active()`.
- `goals_summary() -> str`, `cancelled_function_call_ids() -> list[str]`.
- `get_correction(agent, task)` / `has_correction(...)` / `iter_corrections_for_agent(agent) -> list[str]` — the pending-correction slot written by `goldfive/_correction_injection.py`. `iter_corrections_for_agent` strips a compound `client:agent` prefix so a compound-named caller finds the bare-form keys the writer stamped.
- `get_pending_delegation(fc_id) -> DelegationPin | None` / `iter_pending_delegations() -> Mapping` — normalises both the legacy bare-string and the versioned `{task_id, revision, tool_args}` shape into a frozen `DelegationPin`.
- `get_reasoning_extracted_binding(agent) -> ReasoningBinding | None` — same compound→bare fallback as corrections.

### Write methods

- `set_pin_current_task(task_id, *, source=BindingSource.UNKNOWN, revision=None, title="")` — stamps the pin; no-op on empty `task_id` (use `clear_pin_current_task()` to clear). `source` (a `BindingSource` enum: `DELEGATION_PIN`/`AGENT_CALLBACK`/`REASONING`/`CORRECTION_TARGET`/`STEERER_ROTATION`/`LOW_CONFIDENCE`/`UNKNOWN`) is currently observability-only and **not stored on the pin slot** — the pin slot pre-dates the source vocabulary.
- `set_pending_delegation(fc_id, *, task_id, revision=0, tool_args=None)` — V4 of the audit; every delegation-site pin lands here on **goldfive** `Session.state`, never on ADK state. No-op on empty `fc_id`/`task_id` or read-only state; drops empty/non-mapping `tool_args`.
- `record_reasoning_extracted_binding(*, agent_name, task_id, confidence, recorded_at_turn=0, run_id="", session_id="")` / `clear_reasoning_extracted_binding(agent_name)` — the reasoning-judge stated-intent attribution; confidence clamped to `[0,1]`; read-modify-write preserves unrelated agents' bindings; `clear_` tries both the exact and the bare-form agent name.

### Typed result objects

The store returns frozen dataclasses instead of raw dict fragments so callers get typed fields and shape-normalisation for free. All four live in `state_store.py`.

```python
@dataclasses.dataclass(frozen=True)
class ActiveSteer:
    body: str
    at_turn: int
    author: str
    source: str  # "user" | "goldfive" | ""
    def is_active(self) -> bool:
        return bool(self.body)

@dataclasses.dataclass(frozen=True)
class DelegationPin:
    task_id: str
    revision: int = 0
    tool_args: Mapping[str, Any] | None = None
    def is_set(self) -> bool:
        return bool(self.task_id)

@dataclasses.dataclass(frozen=True)
class ReasoningBinding:
    agent_name: str
    task_id: str
    confidence: float
    recorded_at_turn: int = 0
    run_id: str = ""
    session_id: str = ""
```

- `ActiveSteer` — wraps the four `KEY_ACTIVE_STEER_*` slots. `get_active_steer()` returns `None` (not an empty `ActiveSteer`) when the body is empty, so `if store.get_active_steer():` is the "is a steer set?" idiom. `source` defaults are back-compat: an empty `source` is treated as `"user"` by readers deciding on suppression ("we have an active steer; if we can't prove it's goldfive-authored, treat it as user-authoritative").
- `DelegationPin` — normalises the two on-disk shapes: legacy bare-string entries and the versioned `{task_id, revision, tool_args}` dict (`#266`/F7). `get_pending_delegation` produces one; `iter_pending_delegations` returns the raw live map (no copy) so the pin ladder can walk every entry cheaply.
- `ReasoningBinding` — `recorded_at_turn` lets consumers dismiss bindings older than N turns; `from_dict` is tolerant of partial dicts (a missing `run_id`/`session_id` doesn't drop a legitimate binding) and returns `None` on a missing `task_id`.
- `BindingSource` (a `StrEnum`) — documents which ladder-rung/callback wrote a pin: `DELEGATION_PIN`, `AGENT_CALLBACK`, `REASONING`, `CORRECTION_TARGET`, `STEERER_ROTATION`, `LOW_CONFIDENCE`, `UNKNOWN`. Currently observability-only; **not stored on the pin slot itself** (the slot pre-dates the vocabulary), so passing `BindingSource.UNKNOWN` is always safe.

---

## Drift conditions on `Session.state` (`open`/`escalate`/`resolve`)

The `goldfive.active_drifts` key (`KEY_ACTIVE_DRIFTS`) holds the in-flight drift **conditions** for the current turn — the state that lets a repeated drift emit `opened → escalating → resolved` lifecycle events instead of N indistinguishable `DriftDetected`s. This is the load-bearing example of invariant #5.

### The `Drift` dataclass and `condition_id`

```python
# goldfive/state_store.py
@dataclasses.dataclass
class Drift:
    condition_id: str          # sha1(kind|task|agent|turn)[:16] — STABLE
    kind: DriftKind | None
    task_id: str
    agent_id: str
    turn_id: str
    severity: DriftSeverity | None
    prev_severity: DriftSeverity | None = None
    lifecycle: str = LIFECYCLE_OPENED
    occurrences: int = 1
```

A *condition* is a logical occurrence of drift keyed by **kind + task + agent within a turn**. `compute_condition_id` hashes exactly that tuple:

```python
def compute_condition_id(*, kind, task_id, agent_id, turn_id) -> str:
    payload = f"{kind.value}|{task_id}|{agent_id}|{turn_id}".encode()
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]
```

The key is a **hash of a stable tuple**, never a fresh uuid. At the call site (`DriftObserver`, `drift_observer.py:1020`), `turn_id` is bound to **`session.run_id`** — which is stable for the whole `Runner.run` turn and changes only when a fresh `Session` is minted next turn. So same kind+task+agent within the same turn always collapses onto one condition; the next turn (new Session → new `run_id`) opens a fresh one. If you keyed conditions on an LLM-minted id or a per-emit counter, every emit would open a new condition and the escalation ladder would never engage — the exact lifecycle-gate failure mode invariant #5 warns about.

Two ids are deliberately distinct here and must not be conflated: the drift's **intrinsic per-event `id`** (`#199`, unique per emit) and the **`condition_id`** (`#271`, a logical group the same kind+task+agent re-opens within a turn). The condition key is tolerant of partial state — a drift with empty `current_task_id`/`current_agent_id` still hashes to a stable id (the sha1 hashes the empty strings), so user-control drifts without a pinned task collapse onto one condition per turn per kind.

`Drift.to_dict()` / `from_dict()` round-trip through the state dict as plain JSON (enums become lowercase strings via `_kind_to_str` / `_severity_to_str`) so sinks can persist and replay. `from_dict` is tolerant of unknown enum values (falls back to `None`).

### The lifecycle helpers (module level, mirrored on `StateStore`)

| Helper | Effect | Returns |
|---|---|---|
| `open_or_escalate_drift(state, *, kind, task_id, agent_id, turn_id, severity)` | first emit → `LIFECYCLE_OPENED`, `occurrences=1`; repeat → `LIFECYCLE_ESCALATING`, monotonic severity bump, `occurrences+1`, `prev_severity` set | the resulting `Drift` |
| `resolve_drift(state, condition_id)` | mark `LIFECYCLE_RESOLVED`, remove from active set; idempotent (unknown id → `None`) | final `Drift` or `None` |
| `resolve_drifts_matching(state, *, task_id=None, agent_ids=None, turn_id=None, kinds=None)` | **batch** conjunctive resolve — single read + single write over the active set | list of resolved `Drift`s |
| `escalate_drift_to_human_intervention(state, condition_id)` | force `severity=CRITICAL`, `LIFECYCLE_HUMAN_INTERVENTION_REQUIRED`, remove from active set | final `Drift` or `None` |

The steerer's wire emit reads the returned `Drift`'s `condition_id` / `lifecycle` / `prev_severity` to stamp the matching fields on `DriftDetected` (see [12-events-sinks-telemetry.md](12-events-sinks-telemetry.md)). Severity bumps are **monotonic** in `INFO < WARNING < CRITICAL` — a re-emit at lower severity preserves the higher recorded value.

`resolve_drifts_matching` is the `#486` mechanism: a task-terminal transition or a staleness-guarded on-task verdict moots several conditions at once and emits `DRIFT_LIFECYCLE_RESOLVED`. Filters are **conjunctive** and a `None` filter matches everything, so a caller must supply at least one filter (calling it with all-`None` would resolve *every* active condition). `agent_ids`/`kinds` are membership filters — pass `{"agent", ""}` to include conditions a detector opened without agent attribution. GOAL_DRIFT resolves **only at task-terminal** per `#486`; see [07-deterministic-drift-detection.md](07-deterministic-drift-detection.md) and [09-steering-ladder-and-gates.md](09-steering-ladder-and-gates.md).

The `StateStore` wrappers (`open_or_escalate_drift`, `resolve_drift`, `resolve_drifts_matching`, `escalate_to_human_intervention`) add a fallback: on a non-`MutableMapping` state they return a synthetic single-shot condition (for `open_or_escalate`) or `None`/`[]` (for the resolvers), so the steerer's wire path always has a stable `condition_id`/`lifecycle` to stamp even in degenerate test scaffolding.

### A worked condition lifecycle trace

To make the state transitions concrete, here is one condition's life across a turn for a `GOAL_DRIFT` on `(task_3, agent_x)`. `state` is the goldfive `Session.state`; `turn_id` is `session.run_id`:

1. **First INFO emit.** `open_or_escalate_drift(state, kind=GOAL_DRIFT, task_id="task_3", agent_id="agent_x", turn_id=run_id, severity=INFO)` finds no existing entry → creates `Drift(condition_id=<hash>, lifecycle="opened", severity=INFO, occurrences=1)`, writes it under `KEY_ACTIVE_DRIFTS[<hash>]`, returns it. The steerer stamps `DriftDetected{condition_id=<hash>, lifecycle=DRIFT_LIFECYCLE_OPENED}`.
2. **Second emit, now WARNING.** Same tuple → hashes to the **same** `<hash>`. `open_or_escalate` finds the entry, computes `new_severity = max(INFO, WARNING) = WARNING`, sets `prev_severity=INFO`, `lifecycle="escalating"`, `occurrences=2`, rewrites the slot. Wire: `DriftDetected{lifecycle=DRIFT_LIFECYCLE_ESCALATING, prev_severity=INFO, severity=WARNING}`. Sinks render "INFO → WARNING" without remembering the prior emit.
3. **Third emit at INFO (a lower severity).** Severity bumping is monotonic, so `max(WARNING, INFO)` stays `WARNING` — the recorded value does not regress; `occurrences=3`.
4. **`task_3` transitions to COMPLETED.** The reconciler calls `resolve_drifts_matching(state, task_id="task_3", turn_id=run_id)` — a **single** read+write that resolves this condition (and any other on `task_3`), each finalised with `lifecycle="resolved"` and removed from `KEY_ACTIVE_DRIFTS`. Wire: `DRIFT_LIFECYCLE_RESOLVED` per resolved condition (`#486`). GOAL_DRIFT resolves **only** at this task-terminal point.
5. **Alternative branch — escalation to a human.** Had the ladder exhausted instead, `escalate_to_human_intervention(state, <hash>)` would force `severity=CRITICAL`, `lifecycle="human_intervention_required"`, and remove the entry (further auto-escalation on a human-owned condition would be misleading).

Every transition is a read-modify-write of the one `KEY_ACTIVE_DRIFTS` dict; the `condition_id` is stable throughout because every component of its hash tuple is stable for the turn. That stability is the whole point — it is what lets steps 1–4 be recognised as **one** condition rather than four unrelated drifts.

---

## Surface 3: StateStore module-level registries

Three process-wide dicts live at module scope in `state_store.py`, keyed by `session.id`. They hold data that **cannot go under `goldfive.*`** because the values are not JSON-serialisable (live `asyncio.Task`s) or because they need cross-callback isolation. They are owned exclusively by `StateStore` methods; nothing else may touch them.

```python
# goldfive/state_store.py (module scope)
_ACTIVE_INVOCATION_TASKS: dict[str, dict[str, asyncio.Task[Any]]] = {}
_CANCEL_REQUESTED_INVOCATIONS: dict[str, set[str]] = {}
_SUPERSEDE_PENDING_INVOCATIONS: dict[str, set[str]] = {}
_ACTIVE_INVOCATION_LOCK = threading.Lock()   # guards structural mutations of the outer dicts
```

**Why module-level and not a `Session.state` slot:** an `asyncio.Task` holds a live coroutine + running loop reference and cannot round-trip through a sink (invariant #6). Storing it under `goldfive.*` would break the serialisability contract or force every sink to special-case the slot. Instead the tasks live in this registry and the outer dict is keyed by `Session.id`, so the "StateStore-as-view" contract still holds: each store instance only sees its own session's tasks.

**Why on the store and not the plugin instance:** these were originally on `_GoldfiveADKPlugin._invocation_tasks` (PR #303). Phase 3.5 (`#305`) relocated them to the store because **per-session orchestration state belongs on the per-session surface, not on an adapter plugin instance** — that is the Phase 0 state-ownership contract. The plugin keeps a backwards-compat `_InvocationTaskRegistryView` (`_adk_plugin.py`) that forwards `dict`-shaped operations onto the store for legacy callers.

### The three registries

1. **`_ACTIVE_INVOCATION_TASKS`** (`invocation_id -> asyncio.Task`, per session). The goldfive boundary wrapper registers the running task on entry (`register_invocation_task`) and deregisters in a `finally` (`deregister_invocation_task`). The steerer's `request_invocation_cancel` looks up the task (`get_invocation_task`) to fire `task.cancel()`. `active_invocation_ids()` lists them (diagnostic).

2. **`_CANCEL_REQUESTED_INVOCATIONS`** (`#242`, per-session set of invocation ids). Stamped **synchronously** at the top of `DefaultSteerer.request_invocation_cancel` (`mark_invocation_cancel_requested`) *before* any async work. This closes the iter-11D race: `active_invocation_ids()` only transitions to empty **after** ADK winds down each cancelled invocation (~4–8s later), so the late-drift gate consults `cancel_requested_invocation_ids()` / `is_invocation_cancel_requested()` to treat a drift firing in that window as late.

3. **`_SUPERSEDE_PENDING_INVOCATIONS`** (`#405` LOW #7, per-session set). Per-invocation isolation for the goldfive-internal supersede marker (see the legacy `_supersede_pending` bool above). Stamped by `mark_supersede_pending` before the cancel lands on the task; consumed by the executor's cancelled branch (`is_supersede_pending` / `has_any_supersede_pending`) to distinguish an internal supersede from an external cancel. `clear_supersede_pending` / `clear_all_supersede_pending` drop entries after the signal is consumed.

`clear_active_invocations()` wipes **all three** registries for the session in one shot — called from the adapter's dispatch teardown so a stale handle can't target the next invocation. This is the reset point for surface #3.

**Locking discipline:** `_ACTIVE_INVOCATION_LOCK` (a process-wide `threading.Lock`) guards only *structural* mutations of the outer dicts (sub-dict insert/pop). Inner-dict reads/writes are single-threaded per session (ADK callbacks run on one event loop), but a concurrent `request_invocation_cancel` from a *different* session could race the structural `setdefault`, hence the lock. When you add a registry method, take the lock around the outer-dict mutation and release it before touching the inner bucket, matching the existing pattern.

### The dual-signal supersede design (transitional)

The supersede marker exists in **two** places at once, which is a deliberate transitional state you must not "clean up" without reading `#430`:

- The **legacy bool** `session._supersede_pending` (surface #7). Set to `True` in `DriftObserver` before an internal supersede-cancel (`drift_observer.py:5293`); cleared to `False` at several executor sites (`sequential.py:1132`, `:1271`, `:1304`). It is a *session-scope* flip.
- The **per-invocation set** `_SUPERSEDE_PENDING_INVOCATIONS` (surface #3). Stamped by `mark_supersede_pending(inv_id)` when there **is** an invocation id to anchor.

The executor's cancelled branch (`sequential.py:1258`) reads the **union**: it checks `getattr(session, "_supersede_pending", False)` OR `StateStore.for_session(session).has_any_supersede_pending()`. Why keep both? The empty-resolver fallback in `DriftObserver._cancel_inflight_for_revision` has no invocation id to anchor a registry entry, so the bool is the only signal it can raise. And the concurrent-overlay isolation the audit called for is only provided by the per-invocation set (a session-global bool clear could mask a true supersede from another invocation). **Readers prefer the set when they hold an `invocation_id`, fall back to the bool otherwise.** Issue `#430` retires the bool in favour of a sentinel-id registry stamp.

---

## Surface 5: plugin-instance dicts (`_adk_plugin.py`)

`_GoldfiveADKPlugin` (`goldfive/adapters/_adk_plugin.py`) holds a few dicts/fields that are **plugin-local** — they live and die with one plugin instance (≈ one Runner) and are **not part of any session contract**. The state-ownership tripwire ignores them; reads/writes are unrestricted. They are:

| Field | Set in `__init__` | Purpose |
|---|---|---|
| `self._active_ctx: SessionContext \| None` | `= None` | the currently-driving `SessionContext`; set by `set_active_context` before `run_async`, cleared in the adapter's `finally`. The tree-walk `session_context_from_invocation` reads it. |
| `self._cancel_state: dict[str, Any]` | `= {}` | the plugin's own backing dict for cooperative-cancellation entries (`KEY_CANCEL_REQUESTED`). goldfive routes the `_adk_state_protocol` cancel helpers against **this dict**, not ADK `session.state`. |

`self._active_ctx` is the linchpin of the shallow-copy workaround: because ADK deep-copies `session.state` across `get_session`, the plugin **cannot** hand the `SessionContext` through ADK state on the live path, so it stashes it on itself and callbacks tree-walk to find it. See [The shallow-copy handoff hazard](#the-shallow-copy-handoff-hazard-the-8-hour-lesson) and [05-adk-plugin.md](05-adk-plugin.md).

`_cancel_state` is why the `_adk_state_protocol.write_cancel_request` / `consume_cancel_request` helpers take an **arbitrary `MutableMapping`** rather than hard-coding a surface: goldfive's plugin passes `self._cancel_state`; a harmonograf-side bridge passes its own dict. The `CANCELLATION-STASH` tripwire (`CancellationStashViolation`, a `BaseException`) guards against a `CancelledError` being stashed and swallowed instead of re-raised.

**Rule:** if state must outlive a single invocation but not a whole plugin, and is not per-session-serialisable, a plugin-instance field is acceptable — but prefer a `StateStore` registry keyed by `session.id` for anything that is logically *per session*, because a plugin can drive multiple sessions and per-session data on the plugin instance is a cross-session-leak hazard (the exact reason PR #303's `_invocation_tasks` was relocated to the store in Phase 3.5).

---

## State and concurrency

Goldfive's state contract is **lock-free single-writer-per-session** for the two per-session surfaces (#1 and #2), and **explicitly locked** only for the process-wide registries (#3). Understanding why lets you avoid both over-locking (adding pointless locks to `Session` fields) and under-locking (touching a registry's outer dict without the lock).

- **Per-`Session` surfaces need no lock.** Every `Session` owns its own `state` dict and its own field instances (fresh per turn). ADK's adapter callback contract **serialises drift delivery per session** — callbacks for one session run on one event loop, one at a time — so concurrent writes for the same `Session` do not arise. Cross-session writes target distinct `Session` objects (distinct dicts). This is why `refine_outcomes`, `_reasoning_judge_counters`, and the drift-condition dict are all mutated without locks and it is not a bug. The field comment on `refine_outcomes` documents this explicitly, and notes its predecessor `refine_failure_counts` held the same property — the lock-free pattern is the established contract, not a regression.
- **Sub-Runners are isolated by construction.** `AgentTool` spawns a sub-Runner with its **own `Session`** — a different dict from the parent. ContextVars (surface #6) are the mechanism that keeps concurrent sub-Runners from cross-talking on scoped state: `_CHANNEL_PROCESSOR_ACTIVE` and `LLM_CALL_DIAGNOSTICS_VAR` are per-context, so parallel sub-invocations each maintain independent flags.
- **The registries need a lock because they are process-wide.** `_ACTIVE_INVOCATION_TASKS` et al. are single dicts shared across all sessions in the process. A `request_invocation_cancel` from session A can race a `register_invocation_task` from session B on the **outer** dict's structural mutation (`setdefault` / `pop`), so `_ACTIVE_INVOCATION_LOCK` guards exactly that. Inner-bucket access is still single-threaded per session and needs no lock.

The practical rule: **do not add a lock to a `Session` field or a `goldfive.*` key.** If you think you need one, you have either mis-scoped the datum (it should be a registry) or misdiagnosed a bug that is actually a stale-key or reset-point problem.

### Sub-Runner state propagation (the `AgentTool` case)

CANON invariant #3 requires any tree shape to work, including coordinator + `AgentTool`. State ownership must hold across the sub-Runner boundary, and it does, by construction:

- **Each sub-Runner gets its own `Session`.** `AgentTool` spawns a sub-Runner whose `before_run_callback` fires against a **different** `Session.state` dict from the parent's. Surfaces #1 and #2 are therefore automatically isolated — the child cannot clobber the parent's pin or drift conditions because they live on different dicts.
- **The plugin `_active_ctx` (surface #5) tracks the active sub-invocation.** Because callbacks reach the goldfive session by tree-walking to the plugin's `_active_ctx` (not by reading ADK state), a sub-invocation's callbacks resolve to the sub-Runner's `SessionContext` that `set_active_context` installed for it.
- **The invocation-parent map (surface #4 read) stitches the tree for cancellation only.** `KEY_INVOCATION_PARENTS` records `invocation_id → parent_invocation_id` so a cancel propagates down the sub-tree; it carries **no** orchestration state and **no** notion of "coordinator" or "root". Every agent's children are handled identically — that tree-agnosticism is what lets any tree shape work.
- **The `current_agent_id` field (surface #1) reflects the actual reasoner.** When a coordinator (the task assignee) delegates to a child via `AgentTool`, the child reasons under the parent's task pin, but `current_agent_id` is stamped to the child (`before_agent_callback`, last-writer-wins) so the reasoning judge attributes drift to the agent that actually produced the tokens, not the static plan assignee. This is why the reasoning judge reads `current_agent_id` and falls back to `task.assignee_agent_id` only when it is empty.

The takeaway: you never have to special-case `AgentTool` in state code. Put the datum on the right per-session surface and the sub-Runner isolation is free. If you find yourself writing "if this is a sub-invocation" logic around a state write, you have almost certainly picked the wrong surface.

---

## Surface 4: ADK `session.state['goldfive.*']` (`_adk_state_protocol.py`)

`goldfive/adapters/_adk_state_protocol.py` owns the `goldfive.*` key names goldfive writes into the **ADK** `session.state` dict so the LLM can read active-task/plan context during its turn, and so goldfive can read back agent-side writes after the turn. This is the LLM-facing bridge — a per-adapter concern, not the cross-cutting orchestration surface #2.

**Ownership rule (invariant #2): from inside a goldfive callback, this dict is READ-ONLY.** ADK considers `session.state` writes exclusive to its own `session_service.append_event(EventActions(state_delta=...))` machinery. A direct `dict.__setitem__` from a callback races ADK's optimistic-concurrency model — the symptom in production (goldfive#275) was a stale-session `ValueError` → steerer torn down → **0 observability events emitted**.

### Two directions

- **Goldfive → Agents** (advertised to the LLM in `before_model_callback`): `KEY_CURRENT_TASK_ID`, `_TITLE`, `_DESCRIPTION`, `_ASSIGNEE`, `_REVISION`, `KEY_PLAN_ID`, `KEY_PLAN_SUMMARY`, `KEY_RUN_ID`, `KEY_COMPLETED_TASK_RESULTS`, `KEY_AVAILABLE_TASKS`, `KEY_TOOLS_AVAILABLE`. Also bridged from surface #2 for the `GoldfivePlanner` request-side injection: `KEY_ACTIVE_STEER_BODY`, `KEY_ACTIVE_STEER_AT_TURN`, `KEY_GOALS_SUMMARY`, `KEY_CANCELLED_FUNCTION_CALL_IDS`. The string values are **intentionally identical** to surface #2's keys — same logical field, two readers.
- **Agents → Goldfive** (written by the agent as `state_delta` events, or intercepted by `before_tool_callback`): `KEY_TASK_PROGRESS`, `KEY_TASK_OUTCOME`, `KEY_AGENT_NOTE`, `KEY_DIVERGENCE_FLAG`.

### Full key inventory (surface #4)

Every ADK-side key constant lives in `_adk_state_protocol.py` and is in its `ALL_KEYS`. Note these are a **different `ALL_KEYS`** from `state_store.py`'s — the two modules each own their own tuple even where the string values coincide.

| Constant | Key string | Direction | Value shape |
|---|---|---|---|
| `KEY_CURRENT_TASK_ID` / `_TITLE` / `_DESCRIPTION` / `_ASSIGNEE` | `goldfive.current_task_*` | G→A | `str` |
| `KEY_CURRENT_TASK_REVISION` | `goldfive.current_task_revision` | G→A | `int` |
| `KEY_PLAN_ID` / `KEY_PLAN_SUMMARY` / `KEY_RUN_ID` | `goldfive.plan_id` / `plan_summary` / `run_id` | G→A | `str` |
| `KEY_COMPLETED_TASK_RESULTS` | `goldfive.completed_task_results` | G→A | `dict[str,str]` |
| `KEY_AVAILABLE_TASKS` | `goldfive.available_tasks` | G→A | `list[dict]` |
| `KEY_TOOLS_AVAILABLE` | `goldfive.tools_available` | G→A | `list[str]` |
| `KEY_TASK_PROGRESS` | `goldfive.task_progress` | A→G | `dict[task_id, float]` |
| `KEY_TASK_OUTCOME` | `goldfive.task_outcome` | A→G | `dict[task_id, str]` |
| `KEY_AGENT_NOTE` | `goldfive.agent_note` | A→G | `str` |
| `KEY_DIVERGENCE_FLAG` | `goldfive.divergence_flag` | A→G | `bool` |
| `KEY_PENDING_CORRECTIONS` | `goldfive.pending_corrections` | G→A (prefix) | `{prefix}.{agent}.{task}` bodies |
| `KEY_ACTIVE_STEER_BODY` / `_AT_TURN` / `KEY_GOALS_SUMMARY` / `KEY_CANCELLED_FUNCTION_CALL_IDS` | (mirrors #2) | G→A | see surface #2 |
| `KEY_CANCEL_REQUESTED` | `goldfive.cancel_requested` | internal | `dict[invocation_id, CancellationRequest]` |
| `KEY_INVOCATION_PARENTS` | `goldfive.invocation_parents` | internal | `dict[invocation_id, parent_invocation_id]` |

"G→A" = Goldfive→Agents (advertised to the LLM); "A→G" = Agents→Goldfive (agent-written, read back after the turn).

### Reader functions

All readers are total and tolerant (`read_current_task`, `read_run_id`, `read_plan_id`, `read_plan_summary`, `read_completed_results`, `read_available_tasks`, `read_tools_available`, `read_agent_outcome`, `read_agent_progress`, `read_agent_note`, `read_divergence_flag`). Non-Mapping state, missing keys, and malformed values all return typed defaults; none raise. They share the module-private `_safe_get` / `_safe_str` helpers, which mirror `state_store.read`'s tolerance.

### The writers that were deleted (Phase 2.1)

The adapter-side pin writers (`write_current_task` / `write_current_task_id` / `write_current_task_revision` / `clear_current_task`) **were removed** in Phase 2.1 of `#271`. The pin now lives on **goldfive** `Session.state` exclusively (surface #2). The key *constants* remain (read-side contract for custom adapters that consult the live ADK session), but there is no goldfive writer for them — see the module comment near `KEY_CURRENT_TASK_ID`. If you are looking for where the current-task pin is written, it is `StateStore.set_pin_current_task`, **not** here.

### The writers that survive (blessed exceptions)

Two families of writers remain in `_adk_state_protocol.py`, both writing to state dicts that are **not** the live ADK session inside a racing callback:

- **Cooperative cancellation** (`#251` Stream C): `write_cancel_request` / `read_cancel_request` / `consume_cancel_request` manage `KEY_CANCEL_REQUESTED` (`dict[invocation_id, CancellationRequest]`). `consume_` reads-and-clears so re-entry into the same callback doesn't re-cancel. In goldfive's own plugin these ride the **plugin-local `_cancel_state` dict** (surface #5), not ADK state; the helpers take an arbitrary `MutableMapping` so harmonograf-side bridges can target their own dict.
- **Invocation-parent bookkeeping**: `register_invocation_parent` / `children_of_invocation` / `descendants_of_invocation` manage `KEY_INVOCATION_PARENTS` (`dict[invocation_id, parent_invocation_id]`) for cancel propagation. Tree-agnostic (CANON invariant #3) — no notion of "coordinator" or "root"; every agent's children are handled identically.

`extract_agent_writes(before, after)` diffs two state snapshots and returns the `goldfive.*` keys the agent added/changed/removed (a removed key is `{key: None}`). This is how goldfive reads agent-side writes without mutating ADK state.

### Cooperative-cancellation state flow (worked example)

Cooperative cancellation touches three surfaces in sequence and is a good model for how state moves without ever writing ADK `session.state` from a callback:

1. **Producer (steerer, surface #3).** `DefaultSteerer.request_invocation_cancel(inv_id)` first calls `store.mark_invocation_cancel_requested(inv_id)` **synchronously** (stamps `_CANCEL_REQUESTED_INVOCATIONS`), then looks up the live task via `store.get_invocation_task(inv_id)` (`_ACTIVE_INVOCATION_TASKS`) and fires `task.cancel()`.
2. **Request stash (plugin, surface #5).** For the cooperative-consume path, `write_cancel_request(plugin._cancel_state, invocation_id=inv_id, request=...)` stamps a `CancellationRequest` under `KEY_CANCEL_REQUESTED` on the **plugin-local** dict — not ADK state.
3. **Consumer (callback, surface #5 read).** Every adapter callback that can short-circuit a dispatch calls `consume_cancel_request(plugin._cancel_state, inv_id)` at its top. `consume_` reads-and-**clears** the entry so a re-entry into the same callback (a lingering tool call after the LLM call was cancelled) does not re-emit the cancelled response.
4. **Propagation (surface #4 read).** `descendants_of_invocation(state, inv_id)` walks `KEY_INVOCATION_PARENTS` breadth-first so cancelling a parent flags its whole sub-tree — tree-agnostic, no coordinator/root concept.
5. **Late-drift gate (surface #3 read).** A drift firing in the 4–8s window before ADK winds the invocation down sees `store.cancel_requested_invocation_ids()` non-empty and is treated as late (see [09-steering-ladder-and-gates.md](09-steering-ladder-and-gates.md)).

At no point does goldfive write ADK `session.state` from a callback. The request rides the plugin-local dict; the identity/liveness bookkeeping rides the store registries; only the parent-map *read* touches ADK state. See [04-executors-and-control.md](04-executors-and-control.md) and `docs/design/CANCELLATION-CONTRACT.md`.

### Keys that are mirrored on both surface #2 and surface #4

A handful of key *strings* are identical across the goldfive dict (#2) and the ADK dict (#4) **on purpose** — same logical field, two readers, two physical dicts. The writer is always on the goldfive side (#2); the value reaches #4 only via the ADK-blessed bridge for the LLM's request-side injection.

| Key string | #2 writer (`state_store.py`) | #4 reader | Why mirrored |
|---|---|---|---|
| `goldfive.active_steer.body` | `set_active_steer` | `GoldfivePlanner` request injection | planner surfaces the steer to the model |
| `goldfive.active_steer.at_turn` | `set_active_steer` | `GoldfivePlanner` | freshness context for the model |
| `goldfive.goals_summary` | `refresh_goals_summary` | `GoldfivePlanner` | goals block in the planning instruction |
| `goldfive.cancelled_function_call_ids` | `append_cancelled_function_call_ids` | prompt templates | "the cancelled tool call" reference |
| `goldfive.current_task_*` | `StateStore.set_pin_current_task` | custom adapters (read-only) | active-task advertisement |

**Do not treat the #4 copy as authoritative.** The goldfive-side dict (#2) is the source of truth; the #4 copy is a per-turn advertisement produced by the bridge. When you need the *current* value in goldfive code, read surface #2 via `StateStore`, never the ADK dict.

---

## The shallow-copy handoff hazard (the 8-hour lesson)

This is the most expensive state bug in goldfive's history and the reason surface #4 exists as a read-only contract. **Read this before you ever write to, or read back from, ADK `session.state`.**

### The hazard

ADK's `InMemorySessionService.get_session` returns a **shallow copy** of the stored session on every call — internally `_light_copy` / `copy.copy(session.state)`. So (quoting the plugin source, `_adk_plugin.py`):

> A `SessionContext` (or any value) written into the adapter's own `get_session` copy of `session.state` **never reaches** the fresh copy that `runner.run_async` materialises for the invocation. The callback then sees an empty state and silently falls through to the ACK shim.

Restated as a rule: **a write to ADK `session.state` from one callback frame is not guaranteed to be visible to another callback frame**, because they may be reading different shallow copies. This is a general property of SDKs that hand shallow copies across callbacks (documented in auto-memory `feedback_callback_context_handoff.md` — it cost ~8h of wrong fixes). The failure is silent: no exception, just an empty read and a fall-through.

### The fix goldfive actually uses

Goldfive does **not** hand off the `SessionContext` through ADK state on the live path. Instead:

1. The adapter sets `plugin._active_ctx` (a plugin-instance field, surface #5) via `set_active_context` **before** `runner.run_async`.
2. Callbacks reach the goldfive session by walking the plugin tree: `session_context_from_invocation(invocation_context)` (`_adk_plugin.py`) finds the goldfive plugin on `invocation_context.plugin_manager.plugins` (marked by `__goldfive_adk_plugin__`) and reads its `_active_ctx`.

The `SESSION_CONTEXT_STATE_KEY = "goldfive._session_context"` stash on ADK state is kept **only** for out-of-band test scaffolding that hand-builds a `tool_context`; the docstring explicitly says it is "NOT used on the live-run path" precisely because of the deep-copy. This is the concrete embodiment of "route around ADK state; use the plugin reference". This same route replaced the V7 state-stash in Phase 2.0 of `#271` and is what closes goldfive#275.

### The read-back-verification rule

When you have no choice but to hand a value through a shared-but-shallow-copied dict:

- **Never assume a write is visible synchronously.** After writing, if a later read on a *different* frame depends on it, you must verify the read actually observes the value — do not trust the write landed.
- **Prefer a channel the SDK does not copy:** a plugin-instance field reachable by a tree-walk (`_active_ctx`), a module-level registry keyed by a stable id (`_ACTIVE_INVOCATION_TASKS`), or goldfive's own `Session.state` (surface #2, which goldfive owns end-to-end and never round-trips through ADK's copy machinery).
- **When bridging goldfive→ADK for the LLM, do it through `append_event(state_delta=...)`**, the ADK-blessed mechanism (the one `_heal_pending_tool_calls` uses), not a direct `state[k] = v`.

---

## ContextVars in use

Five `contextvars.ContextVar` objects carry per-dynamic-scope state that must not leak across concurrent tasks. Each is set/reset by a context manager (exception-safe) so concurrent Sessions/sub-Runners never observe each other's values. **A ContextVar is the right tool when the datum is scoped to a call region and must be isolated across concurrent async tasks — never a module global you set and clear by hand.**

| ContextVar | Module | Set by | Purpose |
|---|---|---|---|
| `LLM_CALL_DIAGNOSTICS_VAR` | `goldfive/_llm.py` | `llm_call_diagnostics()` cm | per-call `LlmCallDiagnostics` (thought/answer part counts) `#491` |
| `MAX_OUTPUT_TOKENS_VAR` | `goldfive/_llm.py` | `call_llm_budget(...)` cm | per-callsite output-token cap (`None` = no cap) |
| `THINKING_DISABLED_VAR` | `goldfive/_llm.py` | `call_llm_thinking_disabled()` cm | per-callsite `/no_think` (Qwen/litellm-family only) `#491` |
| `_CHANNEL_PROCESSOR_ACTIVE` | `goldfive/types.py` | `channel_processor_active()` cm | marks the steerer's exclusive `Session.plan`-writer region |
| `_active_callback` | `goldfive/_state_audit.py` | `goldfive_callback(name)` cm | tracks which ADK callback is active, for the tripwire |

### `LlmCallDiagnostics` (`#491`)

Before `#491`, per-call diagnostics were smuggled as **attributes mutated on the shared `call_llm` closure** (`call_llm.last_thought_count`) — last-writer-wins once concurrent background judges dispatched through the same closure. `#491` replaced that side channel with a ContextVar-bound per-call object:

```python
# goldfive/_llm.py
@dataclass
class LlmCallDiagnostics:
    thought_count: int = 0
    answer_count: int = 0

LLM_CALL_DIAGNOSTICS_VAR: contextvars.ContextVar[LlmCallDiagnostics | None] = ...

@contextmanager
def llm_call_diagnostics() -> Iterator[LlmCallDiagnostics]:
    diag = LlmCallDiagnostics()
    token = LLM_CALL_DIAGNOSTICS_VAR.set(diag)
    try:
        yield diag          # caller reads diag.* AFTER await call_llm(...)
    finally:
        LLM_CALL_DIAGNOSTICS_VAR.reset(token)
```

Each consumer installs a fresh object around its own `await call_llm(...)`, so concurrent judges cannot observe each other's counts. `record_llm_call_diagnostics(thought_count=, answer_count=)` writes into the current object and is a **no-op** when no consumer installed one — user-supplied `call_llm` callables are not expected to call it. See [08-llm-judges.md](08-llm-judges.md) for the judge dispatch that consumes these.

### The two `_llm.py` scoping vars (`MAX_OUTPUT_TOKENS_VAR`, `THINKING_DISABLED_VAR`)

`goldfive/_llm.py` is the one internal LLM-call module (`#491`). Besides diagnostics it carries two per-callsite scoping vars, each set by a `@contextmanager` and reset via a token in a `finally`:

- `MAX_OUTPUT_TOKENS_VAR: ContextVar[int | None]`, set by `call_llm_budget(n)` — a per-callsite output-token cap. `None` means "no cap". A goldfive judge or refine call that must bound its output installs a cap around its own `await call_llm(...)`; the builder reads the var and applies it. Because it is a ContextVar, a bounded judge call cannot leak its cap onto a concurrent unbounded call in the same process.
- `THINKING_DISABLED_VAR: ContextVar[bool | None]`, set by `call_llm_thinking_disabled()` — a per-callsite "disable thinking" flag. When set, goldfive's default builders apply the model-family-appropriate disable: `/no_think` prompt prefix and/or `extra_body={"enable_thinking": False}`. `#491` narrowed this to the **Qwen/litellm family only** via the `THINKING_DISABLE_CAPABILITIES` table (a `tuple[(marker, ThinkingDisableCaps)]` matched by lowercase substring on the model id), so the disable is not blindly sent to models that reject it. `None`/unset means "leave thinking as the model default".

The pattern to copy for any new per-call LLM knob: **a ContextVar with a context-manager setter**, read by the default builder, no-op when unset — never a mutable attribute on the shared `call_llm` closure (the pre-`#491` anti-pattern that caused last-writer-wins under concurrency).

### `_CHANNEL_PROCESSOR_ACTIVE` — the plan-write envelope

`Session.plan` has exactly one writer (invariant #3). The enforcement is a ContextVar, not a type: `set_session_plan(session, plan)` (`goldfive/types.py`) checks `_CHANNEL_PROCESSOR_ACTIVE`. Outside the region it logs a WARNING with a stack hint; under `GOLDFIVE_STRICT_STATE_OWNERSHIP=1` it **raises `PlanOwnershipViolation`**. The steerer/executor plan-install paths wrap their region in `channel_processor_active()`; sinks and judges read `session.plan` and never write it. The single writer is the steerer's channel processor (`DefaultSteerer._invoke_passthrough_with_control` / `_handle_drift` / `_apply_revision` and the executor install paths they delegate to). See [10-planning-and-revision.md](10-planning-and-revision.md).

---

## The state-ownership contract and tripwire (reconciled to code)

`docs/design/STATE-OWNERSHIP-CONTRACT.md` is the normative doc for the migration. Two reconciliations you must hold in mind:

1. **Module names are stale.** The doc names `goldfive/orchestration_state.py` (write helpers) and `goldfive.orchestration_store` (the `OrchestrationStore` class). Both were **merged into `goldfive/state_store.py`**; `OrchestrationStore` became `StateStore`. The audit catalog's §5.5 "already-compliant" table still cites `goldfive/orchestration_state.py:179` — read that as `goldfive/state_store.py`'s `write()`. The code is ground truth.
2. **The migration is well past Phase 0.** The doc's roadmap (§4) lists Phases 0–3; the code has completed Phases 0, 1, 2.0, and 2.1 — the callback-time ADK-state writes (V1–V5) are **gone**, the pin lives on goldfive `Session.state` exclusively, and the adapter-side pin writers in `_adk_state_protocol.py` are deleted (see the module comment there). Phase 2.x (SessionContext-stash cleanup, V7/V8) and Phase 3 (tripwire on-by-default in production) remain.

### The audit catalog, reconciled to code

`docs/design/STATE-OWNERSHIP-CONTRACT.md` §5 catalogues nine violations (V1–V9). Here is each one's **current** status against main so you can trust or discard a catalog entry at a glance:

| ID | What it was | Status on main |
|---|---|---|
| V1 | `before_run_callback` initial seed of 10 `goldfive.*` keys onto ADK state | **migrated (2.0)** — resolver/planner read goldfive `Session.state` directly |
| V2 | `before_run_callback` orchestration-state bridge (`_bridge_orchestration_state`) — the literal site of #275 | **migrated (2.0)** — bridge function deleted |
| V3 | `_stamp_current_task_id` writing the pin onto both surfaces | **migrated (2.1)** — pin lands on goldfive state via `set_pin_current_task`; protocol writers deleted |
| V4 | `_pin_delegation_task_id` writing `goldfive.pending_delegations` onto both surfaces | **migrated (2.1)** — lands on goldfive state via `set_pending_delegation` |
| V5 | `before_model_callback` defensive duplicate seed | **migrated (2.0)** — falls out with V1 |
| V6 | `_inject_task_id_from_state` mutating `tool_args["task_id"]` | **kept (by design)** — `tool_args` is not `session.state`; ADK hands it to the callback to fill in; the tripwire does not flag it |
| V7 | `ADKAdapter.invoke` `SessionContext` stash on ADK state | **superseded** — live path uses `plugin._active_ctx` tree-walk; stash kept only for out-of-band tests |
| V8 | `ADKAdapter.invoke` `SessionContext` cleanup (`state.pop`) | **superseded** — companion to V7 |
| V9 | `_heal_pending_tool_calls` cancelled-id stamping | **compliant** — writes goldfive's dict; the bridge that propagated it (V2) is gone |

The only intentional survivor is **V6**, the `tool_args["task_id"]` injection — and it is not a `session.state` write at all. When you read the catalog, treat "blocker/mitigated/cosmetic" severities as historical; the "Status on main" column above is ground truth.

### The tripwire (`goldfive/_state_audit.py`)

The tripwire enforces invariant #2 at runtime by patching the single write funnel and walking the call stack:

1. **Active-callback ContextVar.** `_active_callback` (a `ContextVar[_CallbackFrame | None]`) is set/cleared at the entry/exit of every goldfive plugin callback. `wrap_plugin_callbacks(plugin)` wraps all eight callback methods so the bookkeeping is automatic.
2. **Funnel patch.** `_install_protocol_patch` rewrites `_adk_state_protocol._set` — the single funnel every protocol-module writer goes through — to call `_check_caller(...)` before the underlying mutation.
3. **Stack walk against catalog.** `_check_caller` walks the live stack for a frame whose `(filename suffix, qualname suffix)` matches `_KNOWN_CALLERS` (each entry = one catalogued violation). Match → allow; no match → raise `StateOwnershipViolation`. `known_callers_count()` returns the catalog size.

`StateOwnershipViolation` (and its sibling `CancellationStashViolation`) inherit from **`BaseException`, not `Exception`**. This is deliberate: many catalogued sites sit inside broad `try/except Exception` blocks (state writes are best-effort). An `Exception` subclass would be silently swallowed by those blocks — the exact failure mode `#275` suffers in production, where ADK's stale-session `ValueError` is caught and dropped. `BaseException` propagates through `except Exception` and surfaces loudly.

**Defaults** (resolved by `is_enabled()`): `GOLDFIVE_STRICT_STATE_OWNERSHIP=1` force-on, `=0` force-off, **unset → on inside pytest** (`"pytest" in sys.modules`), off in production. CI runs with it enabled. `enable()` / `disable()` flip the env var; `expect_violation(reason)` is a test cm that asserts a violation fires.

**The catalog is the allowlist.** `_KNOWN_CALLERS` is a `frozenset[tuple[str, str]]` of `(filename_suffix, qualname_suffix)` pairs — one per surviving catalogued write. `_check_caller` → `_frame_matches` walks the stack for a matching frame; a match allows the write, no match raises. This is why the catalog and the allowlist must shrink in lockstep: when both are empty, the migration is complete and any callback-time ADK-state write raises. `known_callers_count()` exposes the current size for a regression assertion. When you legitimately need to add a new catalogued write (rare — you almost never should), you add the `(file, function)` pair here *and* an entry to §5 of the design doc; otherwise the tripwire is telling you your write is on the wrong surface.

### The plan-ownership sibling (`PlanOwnershipViolation`)

`set_session_plan` (`goldfive/types.py`) enforces invariant #3 with the **same env gate** but a distinct exception. Outside a `channel_processor_active()` region it logs a WARNING with a one-line stack hint; under strict mode it raises `PlanOwnershipViolation`. The resolution helper `_strict_state_ownership_enabled()` deliberately **duplicates** the env logic from `_state_audit.is_enabled()` (a few lines) rather than importing `_state_audit`, to avoid a typing-time circular import (state-audit imports `Session`). That duplication is bracketed by a unit test in `tests/test_immutable_plan.py`; if you change the env-resolution rule, change it in both places. The plan check is a **runtime smell-test, not a type guarantee** — the `frozen=True` dataclasses already prevent `task.status = X` mutations; this catches the remaining axis: who owns the *pointer* swap.

### Editing the tripwire safely

The public API of `goldfive/_state_audit.py` (for the rare case you must touch it):

- `wrap_plugin_callbacks(plugin)` — called once from `make_adk_plugin`; wraps the eight callback methods so `_active_callback` is set/cleared automatically. If you **add a new plugin callback**, ensure it is wrapped here or its writes will not be tracked.
- `goldfive_callback(name)` — the context manager that stamps `_active_callback`; use it if you invoke a callback out of band in a test.
- `assert_can_write(state, key)` — the check `_set` calls; raises `StateOwnershipViolation` for un-catalogued writes while a callback is active.
- `is_enabled()` / `enable()` / `disable()` / `expect_violation(reason)` — the env gate and test helpers.
- `known_callers_count()` — for a regression test asserting the catalog is not silently growing.

**Do not** add a `(file, function)` pair to `_KNOWN_CALLERS` to silence a violation the tripwire raised on your new code — that is the tripwire correctly telling you the write is on the wrong surface. Move the write to goldfive `Session.state` (surface #2) or bridge it through `append_event`; catalog entries are for the *historical* migration, and they should only ever shrink.

---

## Where to add new state (decision tree)

Follow this in order. Stop at the first match.

```
1. Is the value a live, non-serialisable object (asyncio.Task, asyncio.Event,
   a lock) that must survive one turn and be keyed by session/invocation?
      → StateStore module registry (surface #3). Add register/deregister/clear
        methods on StateStore; take _ACTIVE_INVOCATION_LOCK around outer-dict
        mutation; wire cleanup into clear_active_invocations().
      → EXCEPTION: asyncio.Event waiters keyed by target_id are already a
        typed Session field (pending_approvals). Prefer a field when the shape
        is per-turn and small.

2. Does the LLM need to READ it during its turn, or does the agent WRITE it
   back to you?
      → ADK session.state via _adk_state_protocol.py (surface #4). Add a KEY_*
        constant + a tolerant reader. Do NOT add a callback-time writer — bridge
        goldfive→ADK through append_event(state_delta=...), and read agent→goldfive
        via extract_agent_writes / the read_* helpers.

3. Is it orchestration state that multiple goldfive components must read
   (planner, drift detectors, steerer, reconciler) and that must round-trip
   through a sink (JSON-serialisable)?
      → goldfive Session.state via state_store.py (surface #2). Add a KEY_*
        constant under GOLDFIVE_PREFIX, a module-level writer + tolerant reader,
        and a StateStore method pair. Add to ALL_KEYS if it is a scalar slot.

4. Is it per-turn state read/written by only ONE or TWO tightly-coupled
   goldfive components (a counter, a timestamp, a small dict), with a typed
   default and a clear reset point?
      → A typed Session field (surface #1). Declare it with a default and a
        docstring naming the writer, readers, and reset point. If it must
        persist across turns, seed it in Conversation.next_turn_session.

5. Is it scoped to one dynamic call region and must be isolated across
   concurrent async tasks (a per-call cap, a per-call diagnostics object,
   a "we are inside the plan-writer" flag)?
      → A ContextVar with a @contextmanager setter (surface #6). Reset the
        token in a finally.

6. None of the above / "it's just a quick flag on the session"?
      → STOP. Do NOT stamp a dynamically-typed private attr on Session
        (surface #7). Go back to step 4 and declare a field, or step 3 and
        add a state key. Surface #7 is legacy debt being retired (issue #430),
        not a pattern to copy.
```

---

## Stable-key discipline

A lifecycle gate is only as good as its key (auto-memory `feedback_stable_keys_for_lifecycle_gates.md`). When you key a dict/set that gates behaviour across observations:

- **Key on the claim's identity, never on a churning value.** `Drift.condition_id` = `sha1(kind|task|agent|turn)[:16]`. `refine_outcomes` keys on `(drift.kind.value, task_id)`. `_reasoning_judge_counters` keys on `(agent_name, task_id)`. None of these change between the observations they must correlate.
- **Never key on an LLM-minted id.** LLMs mint fresh `function_call_id`s / correlation ids per response; a gate keyed on one opens a fresh entry every time and never engages.
- **Prefer the reasoning-turn axis for "N turns ago".** Use `_reasoning_turn` (stable, one tick per reasoning observation), not `_next_sequence` (inflated by telemetry volume — the `#441` bug).
- **If the id you have is churning, fix the churn upstream** — do not coarsen the key to compensate. A coarser key over-collapses distinct claims (the `#245` over-rejection lesson: naive `revision < live_revision` merged orthogonal judge verdicts; the fix was a per-`(kind, target)` watermark — `last_addressed_revision_by_drift_key` — not a global counter).
- **The `_reasoning_judge_counters` cautionary tale:** the pre-fix key was a single string keyed on `current_task_id or ""`. Every unpinned turn from every agent collapsed onto the `""` bucket and legitimate first-block judge firings were skipped. The fix (`#226`) was to key on `(agent_name, task_id)`. That is the pattern: **when a gate mis-fires, widen the key to the full identity tuple, don't narrow it.**

---

## Recipes: adding state to each surface

Concrete, copy-adaptable procedures. Pick the surface via the [decision tree](#where-to-add-new-state-decision-tree) first, then follow the matching recipe exactly.

### Recipe A — add a `goldfive.*` orchestration key (surface #2)

Use when: multiple goldfive components must read a JSON-serialisable orchestration fact.

1. In `goldfive/state_store.py`, add a key constant under the prefix:
   ```python
   KEY_MY_THING = "goldfive.my_thing"
   ```
2. If it is a scalar slot other components enumerate, add it to `ALL_KEYS`.
3. Add a **module-level writer** that routes through `write` (so the prefix assertion and tripwire caller-check fire) and a **tolerant reader** that routes through `read`:
   ```python
   def set_my_thing(state, value: str) -> None:
       write(state, KEY_MY_THING, str(value or ""))

   def read_my_thing(state) -> str:
       v = read(state, KEY_MY_THING, "")
       return v if isinstance(v, str) else ""
   ```
4. Add a `StateStore` method pair (`my_thing()` reader, `set_my_thing(...)` writer) that delegates to the module functions, guarding `isinstance(self._state, dict)` on the write path.
5. Export the new public symbols in `__all__`.
6. If the value must survive into the next turn, re-seed it in `Runner.run` alongside `set_current_plan` — surface #2 resets every turn.
7. Confirm the value is JSON-serialisable. If not, this is the wrong surface (use Recipe C).

### Recipe B — add a `Session` field (surface #1)

Use when: one or two tightly-coupled components need per-turn scratch state with a typed default.

1. In `goldfive/types.py`, add the field to the `Session` dataclass **with a default** and a docstring naming the writer, readers, and reset point:
   ```python
   # Written by DefaultSteerer.<method>; read by <reader>. Resets to 0
   # each turn (fresh Session) — carries no cross-turn meaning.
   my_counter: int = 0
   ```
2. Use `dataclasses.field(default_factory=...)` for mutable defaults (`list`/`dict`/`set`) — never a bare `[]`/`{}`.
3. If it must persist across turns, add a copy to `Conversation.next_turn_session` in `goldfive/conversation.py` (like `goals`/`completed_results`).
4. If it needs an explicit per-turn reset beyond the fresh-`Session` default, note that `DriftObserver.reset_for_turn` is invoked by `Runner.run` only via `getattr(self.steerer, "reset_for_turn", None)`, which is a **no-op for the default steerer** (the method lives on `DriftObserver`) — a clear wired only there fires only when the same `Session` is reused across `Runner.run` calls. The reliable per-turn reset is the fresh `Session` from `Conversation.next_turn_session`.
5. Grep to confirm exactly one writer family: `grep -rn "my_counter" goldfive/`.
6. **Do not** stamp it as a dynamic attr — declare the field.

### Recipe C — add a `StateStore` registry (surface #3)

Use when: the value is a live, non-serialisable object (task, event, lock) that must be per-session and survive one turn.

1. In `goldfive/state_store.py`, add the module-level dict keyed by session id and reuse the existing lock:
   ```python
   _MY_REGISTRY: dict[str, dict[str, MyObj]] = {}
   ```
2. Add `StateStore` methods for register/get/deregister that:
   - no-op when `self._session_id` is empty,
   - take `_ACTIVE_INVOCATION_LOCK` **only** around structural mutation of the outer dict,
   - key the inner bucket by the stable id (invocation id / task id), never a churning id.
3. Wire teardown into `clear_active_invocations()` so a session end pops `_MY_REGISTRY[self._session_id]` alongside the existing three registries.
4. Confirm the teardown call site still fires: `grep -rn "clear_active_invocations" goldfive/`.

### Recipe D — advertise a fact to the LLM (surface #4)

Use when: the model must read a fact during its turn, or the agent writes a fact back.

1. In `goldfive/adapters/_adk_state_protocol.py`, add a `KEY_*` constant and add it to `ALL_KEYS`.
2. Add a **tolerant reader** (`read_my_advert`) mirroring the existing `read_*` helpers.
3. For goldfive→agent, **do not** add a callback-time writer. Produce the value on surface #2 and let the existing bridge (or an `append_event(state_delta=...)` you add on the adapter's non-callback path) publish it. For agent→goldfive, read it back via `extract_agent_writes` or your `read_*` helper after the turn.
4. Verify the tripwire stays green: `uv run pytest -q tests/test_state_audit.py`.

### Recipe E — scope a value to one call region (surface #6)

Use when: a per-call cap, per-call diagnostics object, or a "we are inside the writer" flag must be isolated across concurrent async tasks.

1. Declare the `ContextVar` at module scope with a sensible `default`.
2. Provide a `@contextmanager` that `set`s a token on entry and `reset`s it in a `finally` (exception-safe), modelled on `llm_call_diagnostics()` / `channel_processor_active()`.
3. Readers call `.get(default)`; never read the var without a default.
4. Never substitute a module global you set/clear by hand — that leaks across concurrent Sessions.

---

## Common mistakes

Concrete wrong edits a weaker model would plausibly make here, each with the correct alternative.

### 1. Writing ADK `session.state` from a callback

```python
# WRONG — inside before_tool_callback / before_model_callback / etc.
tool_context.state["goldfive.current_task_id"] = resolved_id
callback_context.state.update({"goldfive.plan_summary": summary})
```

This races ADK's optimistic-concurrency model (goldfive#275) and the tripwire raises `StateOwnershipViolation` in tests. **Correct:** write to goldfive's own surface — `StateStore.for_session(ctx.session).set_pin_current_task(resolved_id, ...)` — and let the LLM read it via the bridge, or bridge goldfive→ADK through `append_event(state_delta=...)`. The one carve-out is `tool_args["task_id"]` injection (V6), which is a separate per-dispatch dict ADK hands you to fill in, not `session.state`.

### 2. Stamping an ad-hoc private attr on `Session`

```python
# WRONG
session._my_new_flag = True   # type: ignore[attr-defined]
```

No type, no default, no documented reset point, invisible to anyone reading the dataclass. **Correct:** declare a field on `Session` with a default and a docstring naming writer/readers/reset (surface #1), or add a `goldfive.*` state key (surface #2). The four existing `_supersede_pending`/`_last_cancel_reason_prefix`/`_intercept_transfer`/`_conversational_turn` attrs are legacy debt (`#430`), not precedent.

### 3. Assuming a callback-context write is visible synchronously

```python
# WRONG — write in before_run, read in before_model, expect it to be there
callback_context.state[KEY] = value          # in before_run_callback
...
v = other_callback_context.state.get(KEY)    # in before_model_callback → may be a DIFFERENT shallow copy
```

ADK's `InMemorySessionService` shallow-copies state per `get_session`; the two frames may read different copies and `v` silently comes back empty. **Correct:** reach the goldfive session via `session_context_from_invocation(invocation_context)` (the plugin `_active_ctx` tree-walk) and use goldfive `Session.state`, which goldfive owns end-to-end. If you genuinely must hand a value through ADK state, verify the read-back rather than trusting the write.

### 4. Putting a non-serialisable value under `goldfive.*`

```python
# WRONG
state_store.write(session.state, "goldfive.my_task", some_asyncio_task)
```

`write` accepts it (it only checks the prefix), but the value cannot round-trip through a JSON sink and violates invariant #6. **Correct:** a `StateStore` module registry keyed by `session.id` (surface #3), following the `_ACTIVE_INVOCATION_TASKS` pattern with `_ACTIVE_INVOCATION_LOCK` and `clear_active_invocations` cleanup.

### 5. Writing a non-`goldfive.*` key through the store

```python
# WRONG — raises ValueError
state_store.write(session.state, "my_app.scratch", 1)
```

`_assert_goldfive_key` refuses it by design (catches typos and leaky abstractions). **Correct:** namespace it (`"goldfive.my_scratch"`) if it is genuinely orchestration state, or keep app-level scratch off goldfive's surfaces entirely.

### 6. Keying a lifecycle gate on a fresh/churning id

```python
# WRONG
gate[uuid.uuid4().hex] = ...              # new entry every call → never engages
gate[function_call_id] = ...              # LLM-minted → churns per response
```

**Correct:** key on a stable identity tuple hashed deterministically, like `compute_condition_id(kind=, task_id=, agent_id=, turn_id=)`. See [Stable-key discipline](#stable-key-discipline).

### 7. Swapping `Session.plan` directly

```python
# WRONG
session.plan = revised_plan
```

Bypasses the single-writer guard; logs a WARNING and raises `PlanOwnershipViolation` under strict mode. **Correct:** `set_session_plan(session, revised_plan)` inside a `channel_processor_active()` region (steerer/executor paths only). See [10-planning-and-revision.md](10-planning-and-revision.md).

### 8. Reading the kill-switch by reaching for `_observation_only`

```python
# WRONG
if not steerer._observation_only:
    session.state["goldfive.something"] = ...   # "active-mode-only" write
```

`observation_only=True` is strictly passive (invariant #4); reaching for the private flag re-derives the kill-switch and can start intervening because a stub steerer's flag is set. **Correct:** `if steering_is_active(steerer):` (module helper in `goldfive/steerer.py`) — missing/None/raising → passive. See [09-steering-ladder-and-gates.md](09-steering-ladder-and-gates.md).

### 9. Calling `resolve_drifts_matching` with no filters

```python
# WRONG — every filter None matches every condition → resolves the whole active set
store.resolve_drifts_matching()
```

The filters are conjunctive and a `None` filter matches everything. **Correct:** always supply at least one filter (`task_id=`, `turn_id=`, `agent_ids=`, or `kinds=`) that scopes the batch to the conditions you actually mean to moot.

### 10. Re-introducing a keyword/regex classifier to route state

Do not classify natural-language content with a regex to decide which state key to write (retired `#166`/`#167`). Exact-equality / hash matching of **structured** data (e.g. `condition_id`, `(name, args_hash)` tool-loop corroboration) is allowed and encouraged; NL classification is not. See [07-deterministic-drift-detection.md](07-deterministic-drift-detection.md) and [08-llm-judges.md](08-llm-judges.md).

### 11. Storing a live `asyncio.Task` or `Event` on a plugin instance keyed by session

```python
# WRONG — a plugin can drive multiple sessions; this leaks across them
self._invocation_tasks[inv_id] = task
```

Per-session data on a plugin instance is a cross-session-leak hazard (the exact reason PR #303's `_invocation_tasks` was relocated in Phase 3.5). **Correct:** `StateStore.for_session(ctx.session).register_invocation_task(inv_id, task)` — a registry keyed by `session.id` so each store view sees only its own session's tasks. The plugin's `_InvocationTaskRegistryView` forwards legacy `dict`-shaped access onto the store for you.

### 12. Reading a `goldfive.*` key and expecting it to persist across turns

```python
# WRONG — surface #2 is a fresh empty dict every turn
if session.state.get("goldfive.my_thing"):   # was set last turn → gone now
    ...
```

`Conversation.next_turn_session` builds a brand-new `Session` with an empty `state` dict. **Correct:** promote the datum to a copied-forward `Session` field, or re-seed the key at the top of `Runner.run` the way `set_current_plan` / `refresh_goals_summary` do. See [The per-turn state lifecycle](#the-per-turn-state-lifecycle-a-timeline).

### 13. Trusting the ADK-side (#4) copy of a mirrored key as authoritative

```python
# WRONG — reads the per-turn advertisement, not the source of truth
body = _sp._safe_get(callback_context.state, "goldfive.active_steer.body", "")
```

The #4 copy is a bridge advertisement produced for the LLM; the goldfive-side #2 dict is authoritative. **Correct:** `StateStore.for_session(ctx.session).get_active_steer()`. Reading #4 in goldfive logic also risks the shallow-copy staleness of mistake #3.

### 14. Incrementing `_next_sequence` / `_reasoning_turn` by hand

```python
# WRONG
session._next_sequence += 1
evt_id = f"{session.run_id}:{session._next_sequence}:..."
```

Hand-incrementing skews the counter and breaks the `Event.sequence`/`Event.event_id` agreement. **Correct:** `seq, evt_id = session.next_sequence_and_event_id()` at emit sites; `session.mark_reasoning_turn()` once per reasoning observation. See [12-events-sinks-telemetry.md](12-events-sinks-telemetry.md).

---

## Debugging state problems (symptom → cause)

When a state bug surfaces, this table maps the symptom to the most likely cause and where to look. Work top-to-bottom; the earlier rows are the more common root causes.

| Symptom | Likely cause | Where to look |
|---|---|---|
| "I wrote it, the read comes back empty" (across callbacks) | shallow-copy handoff — two frames read different ADK-state copies | use `session_context_from_invocation`; [The shallow-copy handoff hazard](#the-shallow-copy-handoff-hazard-the-8-hour-lesson) |
| `StateOwnershipViolation` in tests | a callback-time write to ADK `session.state` at an un-catalogued site | move the write to goldfive `Session.state`; `grep` guard #4 |
| `PlanOwnershipViolation` in tests | `set_session_plan` called outside `channel_processor_active()` | wrap the swap; only steerer/executor may write `plan` |
| `ValueError: refuses to write non-goldfive key` | `state_store.write`/`clear` with a key missing the `goldfive.` prefix | namespace the key, or use a non-goldfive surface |
| Lifecycle gate "never engages" | gate keyed on a churning id (uuid / LLM-minted / `_next_sequence`) | key on a stable tuple; [Stable-key discipline](#stable-key-discipline) |
| Gate "engages twice" / duplicate escalation | two condition_ids for one logical drift because a key component churned | confirm `turn_id`=`run_id`, task/agent ids are stable this turn |
| A `goldfive.*` value "disappears" between turns | surface #2 resets to an empty dict each turn | re-seed in `Runner.run`, or promote to a copied-forward field; [per-turn lifecycle](#the-per-turn-state-lifecycle-a-timeline) |
| Retry budget carries stale failures into a new turn | `refine_outcomes` not reset | confirm the fresh `Session` starts with empty `refine_outcomes` (`Conversation.next_turn_session`); `DriftObserver.reset_for_turn` is a no-op on the default path |
| Judge/sink appears to mutate shared state | a read-only component writing state — contract violation | judges/sinks read only; return a verdict for the steerer to record |
| Cross-session data leak (session A sees B's task) | per-session data on a plugin instance instead of a store registry | move to a `StateStore` registry keyed by `session.id` |
| Concurrent judges report each other's token/thought counts | per-call state on the shared closure instead of a ContextVar | use `llm_call_diagnostics()` / a scoping ContextVar (`#491`) |
| Supersede-cancel treated as external abort | reading only the bool or only the set, not the union | check both `_supersede_pending` and `has_any_supersede_pending()` (`#430`) |

## Verification checklist

Run these after touching any state surface. Commands assume repo root `/home/sunil/git/goldfive` and `uv sync --extra dev --extra adk` already run.

### 1. The state-ownership test suite

```bash
uv run pytest -q \
  tests/test_state_store.py \
  tests/test_orchestration_state.py \
  tests/test_orchestration_store.py \
  tests/test_state_audit.py \
  tests/test_state_protocol_propagation.py \
  tests/test_state_rotation.py \
  tests/test_task_state_machine.py
```

`test_state_audit.py` runs with the tripwire enabled — a new callback-time ADK-state write **will** fail here. `test_state_protocol_propagation.py` covers the shallow-copy handoff.

### 2. The full suite (the tripwire is on by default in pytest)

```bash
uv run pytest -q          # ~30s, ~2912 passed / 61 skipped
ruff check .              # must stay clean; do NOT ruff-format (repo is not format-clean)
```

### 3. Grep guards — no new surface-#7 attrs

```bash
# Any NEW dynamically-stamped private attr on session (expect only the 4 known ones):
grep -rn "session\._[a-z]" goldfive/ --include="*.py" | grep "= " | grep -v "self\._"
# Known-good hits: _supersede_pending, _last_cancel_reason_prefix,
#                  _intercept_transfer, _conversational_turn. Anything else is a defect.
```

### 4. Grep guards — no direct ADK-state writes from goldfive

```bash
# Callback-time writes to the ADK session.state (should be none on the pin path):
grep -rn "callback_context\.state\[\|tool_context\.state\[\|\.session\.state\[" \
  goldfive/adapters/_adk_plugin.py | grep -vE "\.get\(|# "
# Any hit that is a subscript-assignment (state[k] = v) inside a callback is a violation.
```

### 5. Grep guards — no direct `Session.plan` swap

```bash
grep -rn "\.plan = " goldfive/ --include="*.py" | grep -v "def \|#\|set_session_plan"
# Every real plan swap must go through set_session_plan(); bare assignments are defects.
```

### 6. If you added a `goldfive.*` key

- Confirm the value is JSON-serialisable (invariant #6). If it is not, it belongs on a `Session` field or a `StateStore` registry, not the key namespace.
- Confirm you added a **tolerant reader** (non-Mapping/missing/None → typed default) and, if it is a scalar slot, an entry in `ALL_KEYS` and `__all__`.
- Confirm the writer routes through `state_store.write` (so the prefix assertion and — under the tripwire — the caller check fire).

### 7. If you added a `Session` field

- It has a `default` or `default_factory`, a docstring naming writer/readers/reset point, and — if it must persist across turns — a copy in `Conversation.next_turn_session` (`goldfive/conversation.py`).
- Grep that its reset point actually fires: `grep -rn "<field_name>" goldfive/ --include="*.py"` and confirm exactly one writer family.

### 8. If you added a `StateStore` registry or method

- Structural mutations of the outer dict take `_ACTIVE_INVOCATION_LOCK`.
- Teardown is wired into `clear_active_invocations()` so a session end wipes it.
- `grep -rn "clear_active_invocations" goldfive/` confirms the adapter dispatch teardown still calls it.

---

## Appendix: state helper quick reference

Every read/write helper, grouped by surface, for lookup. Prefer the `StateStore` method when you hold a `Session`; the module functions are the primitives.

### Surface #2 — goldfive `Session.state` (`goldfive/state_store.py`)

| Datum | Read | Write / clear |
|---|---|---|
| current plan id | — | `set_current_plan(state, plan)` |
| current task pin | `StateStore.pin_current_task()` / `pin_current_task_title()` | `set_current_task(state, task)` / `StateStore.set_pin_current_task(id, ...)` / `clear_current_task(state)` |
| task revision stamp | `read_current_task_revision(state)` / `StateStore.pin_current_task_revision()` | `stamp_current_task_revision(state, rev)` |
| rotate pin after terminal | — | `rotate_current_task_id(state, plan, agent)` |
| transition-driven sync | — | `sync_current_task_from_transition(state, task, to)` |
| goals summary | `StateStore.goals_summary()` | `refresh_goals_summary(state, goals)` (+ `format_goals_summary`) |
| active steer | `StateStore.get_active_steer()` → `ActiveSteer\|None` | `set_active_steer(state, body=, at_turn=, author=, source=)` / `clear_active_steer(state)` |
| processed steer ids | `has_processed_steer_id(state, id)` | `record_processed_steer_id(state, id)` |
| cancelled fc ids | `read_cancelled_function_call_ids(state)` / `StateStore.cancelled_function_call_ids()` | `append_cancelled_function_call_ids(state, ids)` |
| pending correction | `StateStore.get_correction(agent, task)` / `has_correction` / `iter_corrections_for_agent` | `_correction_injection.write_correction(...)` |
| pending delegation | `StateStore.get_pending_delegation(fc_id)` → `DelegationPin\|None` / `iter_pending_delegations()` | `StateStore.set_pending_delegation(fc_id, task_id=, ...)` |
| reasoning binding | `StateStore.get_reasoning_extracted_binding(agent)` → `ReasoningBinding\|None` | `StateStore.record_reasoning_extracted_binding(...)` / `clear_reasoning_extracted_binding(agent)` |
| drift conditions | `StateStore.active_drifts()` / `get_active_drift(cid)` | `open_or_escalate_drift(...)` / `resolve_drift(cid)` / `resolve_drifts_matching(...)` / `escalate_to_human_intervention(cid)` |
| generic | `read(state, key, default)` | `write(state, key, value)` / `clear(state, key)` |

### Surface #3 — StateStore registries (`goldfive/state_store.py`)

| Datum | Read | Write / clear |
|---|---|---|
| invocation task | `get_invocation_task(id)` / `active_invocation_ids()` | `register_invocation_task(id, task)` / `deregister_invocation_task(id)` |
| cancel-requested | `is_invocation_cancel_requested(id)` / `cancel_requested_invocation_ids()` | `mark_invocation_cancel_requested(id)` |
| supersede-pending | `is_supersede_pending(id)` / `has_any_supersede_pending()` / `supersede_pending_invocation_ids()` | `mark_supersede_pending(id)` / `clear_supersede_pending(id)` / `clear_all_supersede_pending()` |
| all-registry teardown | — | `clear_active_invocations()` |

### Surface #4 — ADK `session.state` (`goldfive/adapters/_adk_state_protocol.py`)

| Datum | Read | Write (adapter non-callback / bridge only) |
|---|---|---|
| current task / plan / run | `read_current_task` / `read_plan_id` / `read_plan_summary` / `read_run_id` | (pin writers deleted in Phase 2.1) |
| completed / available / tools | `read_completed_results` / `read_available_tasks` / `read_tools_available` | (bridge) |
| agent-side writes | `read_agent_outcome(state, task)` / `read_agent_progress(state, task)` / `read_agent_note` / `read_divergence_flag` | agent writes via `state_delta` |
| agent-write diff | `extract_agent_writes(before, after)` | — |
| cancel request | `read_cancel_request(state, id)` / `consume_cancel_request(state, id)` | `write_cancel_request(state, invocation_id=, request=)` |
| invocation parents | `children_of_invocation(state, id)` / `descendants_of_invocation(state, id)` | `register_invocation_parent(state, invocation_id=, parent_invocation_id=)` |

### Surfaces #1 / #6 (`goldfive/types.py`, `goldfive/_llm.py`)

| Datum | Access |
|---|---|
| wire sequence + event id | `session.next_sequence()` / `session.next_sequence_and_event_id()` / `session.next_event_id(seq)` |
| logical turn | `session.mark_reasoning_turn()` |
| plan swap (guarded) | `set_session_plan(session, plan)` inside `channel_processor_active()` |
| recent events | `filter_recent_events_by_kind(session.recent_events, kind)` |
| LLM diagnostics | `with llm_call_diagnostics() as diag:` then read `diag.thought_count` / `diag.answer_count` |
| per-call token cap | `with call_llm_budget(n): await call_llm(...)` |
| per-call thinking-disable | `with call_llm_thinking_disabled(): await call_llm(...)` |

---

## Source-of-truth precedence and further reading

When these disagree, resolve in this order — **higher wins**:

1. **The code on `main`** (`goldfive/types.py`, `state_store.py`, `_adk_state_protocol.py`, `_state_audit.py`, `_llm.py`). This is ground truth. Every claim in this chapter was verified against it.
2. **This chapter and the sibling dev-guide chapters.** Written against the same `main`; cross-referenced by filename throughout.
3. **`docs/design/*.md`.** Normative *intent*, but the design docs lag the code in two known ways covered above: the `orchestration_state.py` / `orchestration_store` module names (now `state_store.py` / `StateStore`) and the migration-phase status (the callback-time ADK writes V1–V5 are already gone).

Relevant design docs, in rough order of usefulness for state work:

- `docs/design/STATE-OWNERSHIP-CONTRACT.md` — the normative contract, audit catalog, and tripwire spec (reconcile module names per this chapter).
- `docs/design/STATE-MACHINE.md` / `docs/design/TASK-LIFECYCLE.md` — how the task state machine drives the current-task pin.
- `docs/design/PLAN-LIFECYCLE.md` — the `Session.plan` single-writer lifecycle behind `set_session_plan`.
- `docs/design/CANCELLATION-CONTRACT.md` — the cancel/supersede state flow across the registries and plugin dict.
- `docs/design/CONTROL-CHANNEL.md` / `docs/design/APPROVAL.md` — the control-message and approval-waiter state.

Sibling chapters: [03-runner-and-conversation.md](03-runner-and-conversation.md) (per-turn `Session` construction), [05-adk-plugin.md](05-adk-plugin.md) (`_active_ctx`, callbacks), [09-steering-ladder-and-gates.md](09-steering-ladder-and-gates.md) (kill-switch, drift-condition consumers), [10-planning-and-revision.md](10-planning-and-revision.md) (plan-write envelope), [12-events-sinks-telemetry.md](12-events-sinks-telemetry.md) (sequence/event-id counters, `DriftDetected` stamping), [13-reporting-tools-and-approval.md](13-reporting-tools-and-approval.md) (approval flow), [14-config-reference.md](14-config-reference.md) (`observation_only`, `stall_watchdog_enabled`), [17-invariants-hazards-history.md](17-invariants-hazards-history.md) (the full invariant set).
