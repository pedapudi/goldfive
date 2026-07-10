# 09. Steering: the Ladder, the Gates, and the observation_only Contract

> **⚠ Predates the agency-preservation merge.** This chapter describes the
> pre-merge mechanics; the merge (PRs #453–#504) renamed `NUDGE`→`SIGNAL`,
> replaced corrective templates with advisory observer notes, and added the
> default-OFF ledger/signal regimes. Default-flag behavior described here is
> still accurate; for the merged as-built state read
> `docs/design/AGENCY-PRESERVATION.md` §6 first — it wins on any conflict.

## Read this chapter when...

- You are about to change **what goldfive does in response to a drift** — nudge, refine, cancel, pause, terminate — or the mapping from a `(kind, severity, occurrence)` triple to one of those actions.
- You are adding a **new intervention surface** (any code path that mutates ADK/agent state, injects a message, cancels an invocation, or refuses a dispatch). If you take one thing from this chapter: **every new intervention surface needs the `observation_only` predicate gate AND a both-modes test.** Skipping this has caused four separate production leaks (see "Common mistakes").
- You are touching `handle_drift` / `_handle_drift_dispatch` — the most-fixed methods in the codebase — or any of the gates that sit in front of the refine machinery (freshness watermark, in-flight-refine keys, refine-outcome gate, progress-stall gate, late-drift gate).
- You are editing the refine flow and need to know that there are **two** near-identical refine pipelines (`_handle_drift_dispatch` and `_promote_drift_to_steer`) that must be kept in sync.
- You need to understand the `DRIFT_LIFECYCLE_*` condition model (OPENED / ESCALATING / RESOLVED / HUMAN_INTERVENTION_REQUIRED) and when conditions resolve (#486).
- You are debugging why a drift fired on the wire but nothing happened to the live agent (almost always `observation_only=True` — the production default — or one of the drop gates).

This is the most safety-critical chapter in the guide. The `observation_only` contract is a hard invariant; a single ungated write path violates it for every operator running the default configuration.

## Files covered

| File | What lives here |
| --- | --- |
| `goldfive/steerer.py` | `DefaultSteerer` — the **router**. Owns shared mutable state (sinks, planner, adapter, control channel, `_observation_only` flag, plan locks, background-task sets, the `_active_session_var` ContextVar). Defines `InterventionLevel`, `steering_is_active()`, `compose_corrective_user_message()`, `RefineExhausted`, `is_active_steering()`, `_should_inject()`, and the `REFINE_FAILURE_THRESHOLD` / `PROGRESS_STALL_THRESHOLD_SECONDS` constants. |
| `goldfive/drift_observer.py` | `DriftObserver` (constructed by `DefaultSteerer.__init__` as `steerer.drift`). Owns the drift-routing surface: `observe`, `handle_drift`, `_handle_drift_dispatch`, the `_LADDER` table + `_ladder_level_for`, all the `_dispatch_*` methods, the promotion policy, `request_invocation_cancel`, `_cancel_inflight_for_revision`, the drift-condition lifecycle helpers, and the decision-telemetry emitters. |
| `goldfive/plan_reviser.py` | `PlanReviser` (`steerer.plans`). Holds the plan-mutation gate — `_apply_revision` (one of the four `observation_only` injection points) and `_emit_plan_revised` (the lock-protected install + `dry_run` stamping). |
| `goldfive/executors/sequential.py` | The overlay loop that *drains* `session.pending_nudges` and *consumes* `GOLDFIVE_STEER` / `GOLDFIVE_PAUSE_ESCALATE` control messages. Carries defense-in-depth `observation_only` gates for steerer subclasses that bypass the dispatcher. |
| `goldfive/adapters/_adk_plugin.py` | The plugin-side intervention surfaces: `request_invocation_cancel` (plugin flag write), the LLM-call-timeout watcher cancel (#476), the F3 pre-dispatch redirect (#481), and the `before_tool_callback` approval gate. `_is_observation_only(ctx)` is the plugin's local wrapper around `steering_is_active`. |
| `goldfive/reporting/rendering.py` | F1 directive acks (`_directive_ack` / `_idempotent_response`) — `plan_state` is a goldfive-authored directive gated on `observation_only` (#478). |
| `goldfive/prompt_shaper.py` | Four prompt-shape injection sites, all gated through `PromptShaper.should_inject()` → `steering_is_active`. |
| `goldfive/context_editor.py` | `ContextEditor.apply` — hard-gated on `observation_only` (Invariant 1 there). |
| `goldfive/config.py` | `SteeringConfig` — the `observation_only`, `threshold`, `suppression_window_turns`, `pause_escalate_deadline_s`, `stall_watchdog_enabled`, `stall_timeout_s`, `name_axis_max_severity` knobs. |

## Invariants that bind you here

These are the CANON invariants, specialised to the steering subsystem. Violating any of them is a release blocker.

1. **No prompt-cooperation contracts.** Termination, cancel, pause, and refine must work even if the agent never calls a goldfive tool and never reads an injected message. The ladder drives the *executor* and the *plugin* (cancel flags, control messages, plan swaps) — never a "please stop" instruction the agent is free to ignore. Nudges and corrective messages are best-effort *additions*; they are never the load-bearing enforcement mechanism.
2. **No regex/keyword heuristics for NL classification.** The ladder consumes a typed `DriftEvent` produced by an LLM judge or a deterministic detector. Inside the steering code, you may hash/compare structured data (drift ids, `(kind, task_id)` tuples, revision indices) but you may **not** parse agent free-text to decide an intervention. The one legacy `_unpack_steer_context` "by X: Y" detail parse is a back-compat fallback for tests that build a `USER_STEER` drift without a `ControlMessage`; do not extend that pattern to drift classification.
3. **Any ADK tree shape must work.** Cancel resolves invocation ids through the plugin registry, not through a fixed tree shape. `request_invocation_cancel` no-ops cleanly on an unbound adapter, a non-ADK adapter, or an empty invocation-id list. coordinator+AgentTool must route the same as a flat agent.
4. **Adaptive over predictive.** The ladder reads *observed* state at drift-fire time (`occurrence_count` from `session.refine_outcomes`, `observed_revision_index` from the plan snapshot). It never predicts what the agent will do next. Gates key on facts already recorded, not forecasts.
5. **`observation_only=True` is the production default and is STRICTLY passive.** The ONLY sanctioned read of the kill-switch is `DefaultSteerer.is_active_steering()` (router-internal) or the module helper `steering_is_active(steerer)` (external consumers). A missing steerer, a missing predicate, or a predicate that raises → **PASSIVE** (fail-safe). No code may read `_observation_only` directly except `is_active_steering()` itself.
6. **Lifecycle gates need stable identity keys.** Every gate keys on a stable tuple: freshness on `(kind.value, current_task_id)`, in-flight on `(session_id, kind.value, current_task_id)`, refine-outcome on `(kind.value, current_task_id)`, drift-condition on `sha1(kind, task_id, agent_id, turn_id)`. **Never** key a gate on the per-event `drift.id` (a fresh UUID4 per emit) or on an LLM-minted id — the gate would open a new entry per observation and never engage.

---

## 1. The shape of the subsystem: router + observer

`goldfive.wrap(...)` builds one `DefaultSteerer` per run configuration. `DefaultSteerer.__init__` (in `goldfive/steerer.py`) constructs three components, each holding a back-reference to the router:

```python
# goldfive/steerer.py — DefaultSteerer.__init__ (tail)
self.tasks: TaskStateMachine = TaskStateMachine(self)
self.plans: PlanReviser = PlanReviser(self)
self.drift: DriftObserver = DriftObserver(self)
```

The **router** (`DefaultSteerer`) owns shared mutable state and the `observation_only` gate. The **observer** (`DriftObserver`, `steerer.drift`) owns everything about drift routing. When you see `self._steerer.X` inside `drift_observer.py`, that is the observer reaching back to router-owned state (`_planner`, `_adapter`, `_control_channel`, `_active_session_var`, `is_active_steering()`, `REFINE_FAILURE_THRESHOLD`). When you see bare `self.X`, that is observer-local state (the ladder table, the emit helpers, the gate registries).

> History note: the whole drift-routing surface used to live on `DefaultSteerer` as `_handle_drift`, `_dispatch_nudge`, etc. The "bucket-3c" split (referenced in the big comment above `_LADDER` in `drift_observer.py`) moved them **verbatim** onto `DriftObserver`. The docstrings still say "verbatim move of `DefaultSteerer._handle_drift`" — that is why. `handle_drift` is the ex-`_handle_drift`; behaviour is unchanged. Code on **main** is ground truth; where a design doc still says the method lives on `DefaultSteerer`, the code wins — it is on `DriftObserver`.

### Entry point: `DriftObserver.observe`

Every observed agent event flows into `DriftObserver.observe(event, session)`:

```python
# goldfive/drift_observer.py — observe (abridged)
self._stamp_last_observed(session)               # #487 liveness stamp
await self._maybe_emit_dispatch_snapshot(session)  # one DetectorDispatchOrdered per session
if self._is_duplicate_steer(event, session):     # dupe-STEER dedupe (goldfive#171)
    return
drift = self._drift_from_control(event, session)  # ControlMessage -> USER_* drift
if drift is None:
    drift = self.detect_drift(event, session)     # deterministic classifiers
if drift is None:
    return
await self.handle_drift(drift, session)
```

Two production paths reach `handle_drift`:

1. **Synchronous** (`observe` above): control messages and inline deterministic detectors. These always see a live invocation.
2. **Background** (`_run_drift_handler_background` / `_run_judge_background`): fire-and-forget LLM-judge verdicts spawned off the critical path so the ADK callback can return before a minute-long local-LLM judge finishes. These may land *after* the originating invocation terminated — hence the **late-drift gate** (§8) sits in front of `handle_drift` on the judge path (`observe_reasoning`, at the `_is_late_drift_for_terminated_invocation` call), NOT inside `handle_drift`.

`reporting`-tool-triggered drifts (`report_task_failed`, etc.) reach `handle_drift` through `_spawn_drift_handler_background` so the tool returns immediately.

The dupe-STEER dedupe (`_is_duplicate_steer` → `_steer_dedupe_id`) keys on the source `annotation_id` (bridge-forwarded, #171) or falls back to `ControlMessage.id`; the id lives in `state.processed_steer_ids` with FIFO eviction. Content-based drifts (LOOPING_REASONING, tool errors) are NOT deduped — they are heuristic signals, not user actions.

---

## 2. `InterventionLevel` — the ladder rungs

Defined in `goldfive/steerer.py`:

```python
# goldfive/steerer.py
class InterventionLevel(enum.IntEnum):
    OBSERVE = 0
    ABSORB = 1
    NUDGE = 2
    CANCEL_REINVOKE = 3
    PAUSE_ESCALATE = 4
    TERMINATE = 5
```

`IntEnum`, ordered by intrusiveness. Meanings:

| Level | Value | What the steerer does | Touches the live agent? |
| --- | --- | --- | --- |
| `OBSERVE` | 0 | Emit `DriftDetected` + `SteeringDecisionMade`, then return. No refine, no cancel, no message. | No |
| `ABSORB` | 1 | Call `planner.refine`, validate, install the revised plan (gated). The agent picks up the new plan at the next task boundary. For a scoped set of "coordinator-stuck" kinds, ALSO queue a Level-2 nudge (`_ABSORB_NUDGE_KINDS`). | Only via plan swap (gated) + optional nudge (gated) |
| `NUDGE` | 2 | Enqueue a corrective user message onto `session.pending_nudges`. The overlay drains it into a synthetic user turn and re-invokes the tree. | Yes — injection (gated) |
| `CANCEL_REINVOKE` | 3 | Same refine as ABSORB, then dispatch a `GOLDFIVE_STEER` control message so the executor cancels in-flight work and restarts with the corrective body. | Yes — cancel + restart (gated) |
| `PAUSE_ESCALATE` | 4 | Emit CRITICAL `HUMAN_INTERVENTION_REQUIRED`, dispatch `GOLDFIVE_PAUSE_ESCALATE` so the executor's pre-task loop blocks for an operator. Does NOT refine (Level 4 means the planner cannot recover). | Yes — pause (gated) |
| `TERMINATE` | 5 | Same channel dispatch as Level 4 but the pause payload always carries a `deadline_s` so the executor aborts the run when no operator intervenes. Real terminus since #482. | Yes — pause-then-abort (gated) |

Key distinction: **ABSORB refines but does not preempt** (new plan lands at the next boundary); **CANCEL_REINVOKE refines AND preempts** (cancels the in-flight invocation and restarts it on the new plan). NUDGE injects a message without refining.

---

## 3. The `_LADDER` table (verbatim) and how to read it

The table lives on `DriftObserver._LADDER` and is populated lazily by `_load_ladder_tables` (lazy to avoid an import cycle with `goldfive.steerer`, which defines `InterventionLevel`). Each entry is a 3-tuple:

```
DriftKind: (info_level, warning_level, (critical_first, critical_repeat))
```

- `info_level` — level for a `DriftSeverity.INFO` drift. `None` → falls back to `OBSERVE`.
- `warning_level` — level for a `WARNING` drift. `None` → falls back to `OBSERVE`.
- `(critical_first, critical_repeat)` — a pair for `CRITICAL`: the first element on first occurrence, the second on repeat. "Repeat" means `occurrence_count >= REFINE_FAILURE_THRESHOLD` (currently `2`).

Here is the table verbatim from `_load_ladder_tables` in `goldfive/drift_observer.py` (import alias `_IL = InterventionLevel`):

```python
# goldfive/drift_observer.py — _load_ladder_tables
cls._LADDER = {
    DriftKind.CONFABULATION_RISK: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.AGENT_REFUSAL: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.MODEL_REFUSAL: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.LOOPING_REASONING: (
        None, _IL.ABSORB, (_IL.NUDGE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.LOOPING_TOOL_CALL: (
        None, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.REASONING_CLUSTER_TIGHTENING: (
        _IL.OBSERVE, None, (_IL.OBSERVE, _IL.OBSERVE),
    ),
    DriftKind.PLAN_DIVERGENCE: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.OFF_TOPIC: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.JUSTIFIED_DEVIATION: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.ABSORB, _IL.ABSORB),
    ),
    DriftKind.INTENT_DIVERGENCE: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.PAUSE_ESCALATE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.TOOL_ERROR: (
        _IL.OBSERVE, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.RUNAWAY_DELEGATION: (
        None, None, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.REFINE_VALIDATION_FAILED: (
        None, None, (_IL.PAUSE_ESCALATE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.HUMAN_INTERVENTION_REQUIRED: (
        None, None, (_IL.PAUSE_ESCALATE, _IL.TERMINATE),
    ),
    DriftKind.GOAL_DRIFT: (
        None, _IL.NUDGE, (_IL.NUDGE, _IL.CANCEL_REINVOKE),
    ),
    DriftKind.SELF_REPORTED_STUCK: (
        None, _IL.ABSORB, (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
    ),
    DriftKind.TASK_TIMEOUT: (
        _IL.OBSERVE, _IL.NUDGE, (_IL.PAUSE_ESCALATE, _IL.PAUSE_ESCALATE),
    ),
}
```

### Reading notes (load-bearing)

- **`LOOPING_REASONING` CRITICAL-first is `NUDGE`, not a cancel.** This is a deliberate PROTECTED KEEP decision (#204/#206 history): tool loops emit `LOOPING_REASONING` and route NUDGE-first even at CRITICAL. Do NOT "fix" this to CANCEL_REINVOKE to make it consistent with `LOOPING_TOOL_CALL`. See 17-invariants-hazards-history.md.
- **`REASONING_CLUSTER_TIGHTENING` is `OBSERVE` at every rung.** It is a soft signal that clustering is tightening; the ladder deliberately never acts on it. `warning_level=None` still resolves to `OBSERVE`.
- **`JUSTIFIED_DEVIATION` never escalates past `ABSORB`.** Its CRITICAL pair is `(ABSORB, ABSORB)` — a justified deviation is, by definition, not a defect worth cancelling.
- **`INTENT_DIVERGENCE` CRITICAL goes straight to `PAUSE_ESCALATE`.** No cancel-reinvoke rung: an agent that has diverged from stated intent at CRITICAL is escalated for a human, not silently re-driven.
- **`HUMAN_INTERVENTION_REQUIRED` CRITICAL repeat is the ONLY `TERMINATE` in the table.** `(PAUSE_ESCALATE, TERMINATE)`: the first CRITICAL pauses; a *repeat* CRITICAL (the operator never intervened and the condition re-fired past the failure threshold) terminates with a deadline.
- **`GOAL_DRIFT` lives on the NUDGE path.** WARNING → `NUDGE`, CRITICAL-first → `NUDGE`, CRITICAL-repeat → `CANCEL_REINVOKE`. The judge's signal is "agent stuck on completed work" — a corrective user message fixes it without refining the (correct) plan. `GOAL_DRIFT` is also in `_ABSORB_NUDGE_KINDS` so a WARNING routed through ABSORB *also* queues a nudge.
- **`TASK_TIMEOUT` (#487).** The wall-clock stall watchdog is the sole producer of this kind, and it is flag-gated (`stall_watchdog_enabled`, default `False`). Row is conservative: INFO → `OBSERVE`, WARNING → `NUDGE`, CRITICAL pair `(PAUSE_ESCALATE, PAUSE_ESCALATE)`. Rationale (from the source comment): a stall is a *liveness* signal, not a plan defect, so it nudges rather than refining (ABSORB would loop the planner against a plan that isn't wrong); CRITICAL pauses at both positions because the watchdog only emits CRITICAL on *continued* silence after its WARNING, so a CRITICAL is by construction already a repeat, and the refine-outcome-based occurrence counter never advances on the NUDGE path — the pair's repeat slot alone would be unreachable.

### The fully-resolved table (every kind × every severity)

This is `_LADDER` expanded through `_ladder_level_for` (`None` → `OBSERVE`, CRITICAL split into first/repeat). Use it as a lookup when you need to know exactly what a `(kind, severity, occurrence)` produces without re-deriving it. "repeat" = `occurrence_count >= 2`.

| DriftKind | INFO | WARNING | CRITICAL (first) | CRITICAL (repeat) |
| --- | --- | --- | --- | --- |
| `CONFABULATION_RISK` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `AGENT_REFUSAL` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `MODEL_REFUSAL` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `LOOPING_REASONING` | OBSERVE | ABSORB | **NUDGE** | PAUSE_ESCALATE |
| `LOOPING_TOOL_CALL` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `REASONING_CLUSTER_TIGHTENING` | OBSERVE | OBSERVE | OBSERVE | OBSERVE |
| `PLAN_DIVERGENCE` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `OFF_TOPIC` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `JUSTIFIED_DEVIATION` | OBSERVE | ABSORB | ABSORB | ABSORB |
| `INTENT_DIVERGENCE` | OBSERVE | ABSORB | PAUSE_ESCALATE | PAUSE_ESCALATE |
| `TOOL_ERROR` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `RUNAWAY_DELEGATION` | OBSERVE | OBSERVE | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `REFINE_VALIDATION_FAILED` | OBSERVE | OBSERVE | PAUSE_ESCALATE | PAUSE_ESCALATE |
| `HUMAN_INTERVENTION_REQUIRED` | OBSERVE | OBSERVE | PAUSE_ESCALATE | **TERMINATE** |
| `GOAL_DRIFT` | OBSERVE | NUDGE | NUDGE | CANCEL_REINVOKE |
| `SELF_REPORTED_STUCK` | OBSERVE | ABSORB | CANCEL_REINVOKE | PAUSE_ESCALATE |
| `TASK_TIMEOUT` | OBSERVE | NUDGE | PAUSE_ESCALATE | PAUSE_ESCALATE |
| *(any unmapped kind)* | OBSERVE | ABSORB | ABSORB | PAUSE_ESCALATE |

Note `PLAN_DIVERGENCE` has a full row but its dispatch is disabled at Gate 0 (§4) — the row is a branch KEEP, not live routing. The two bolded cells (`LOOPING_REASONING` CRITICAL-first `NUDGE`, `HUMAN_INTERVENTION_REQUIRED` CRITICAL-repeat `TERMINATE`) are the two non-obvious cells you are most likely to "fix" by mistake — don't (CM5, §9).

### Kinds NOT in the table

Any `DriftKind` absent from `_LADDER` uses the generic fallback in `_ladder_level_for`:

```python
# goldfive/drift_observer.py — _ladder_level_for (fallback branch)
if severity is DriftSeverity.INFO:
    return _IL.OBSERVE
if severity is DriftSeverity.WARNING:
    return _IL.ABSORB
return _IL.PAUSE_ESCALATE if is_repeat else _IL.ABSORB
```

So an unmapped CRITICAL first-occurrence → `ABSORB`, repeat → `PAUSE_ESCALATE`. This is why adding a *new* `DriftKind` without a table entry still gets sane routing.

### `_ladder_level_for` in full

```python
# goldfive/drift_observer.py
def _ladder_level_for(self, kind, severity, occurrence_count):
    from goldfive.steerer import InterventionLevel as _IL
    self._load_ladder_tables()
    entry = self._LADDER.get(kind)
    is_repeat = occurrence_count >= self._steerer.REFINE_FAILURE_THRESHOLD
    if entry is not None:
        info_level, warning_level, critical_pair = entry
        if severity is DriftSeverity.INFO:
            return info_level or _IL.OBSERVE
        if severity is DriftSeverity.WARNING:
            return warning_level or _IL.OBSERVE
        return critical_pair[1] if is_repeat else critical_pair[0]
    # ... generic fallback above
```

`occurrence_count` comes from `_occurrence_count_for_ladder`, which maps the `(kind, task)` entry in `session.refine_outcomes` back to an int: a `"succeeded"` outcome → `0` (treated as first occurrence); a `"failed"` outcome → its `fail_count`. **The ladder is stateless per call** — it reads the count recorded before this dispatch; it does not increment anything.

---

## 4. `handle_drift` — the entry gates IN ORDER

`handle_drift` (in `drift_observer.py`) is a thin wrapper that runs the *entry guards* inline, then delegates to `_handle_drift_dispatch` inside a `try/finally` that clears the in-flight and cancelled-drift registries. Run them in this exact order — the ordering is load-bearing.

### Gate 0 — `PLAN_DIVERGENCE` hard drop

```python
if drift.kind is DriftKind.PLAN_DIVERGENCE:
    log.debug(...)
    return
```

`PLAN_DIVERGENCE` handling was disabled in #252 (replaced by `CAPABILITY_MISMATCH`). The guard is at the very top so no external producer (legacy caller, replay, sink) can revive it. The *machinery* is a PROTECTED KEEP (do not delete the `_LADDER` row or the reconciler surface); only the dispatch is disabled here.

### Gate 1 — `authored_by` normalisation

```python
if not drift.authored_by:
    drift.authored_by = self._resolve_authored_by(drift)
```

`USER_STEER` / `USER_CANCEL` / `USER_PAUSE` → `"user"`; everything else → `"goldfive"`. Every downstream gate that special-cases user intent reads this normalised value. Do this **before** the freshness gate — the freshness gate exempts user-authored drifts.

### Gate 2 — Verdict-freshness watermark (goldfive#245)

Only applies when the drift is goldfive-authored AND stamped (`drift.observed_revision_index != 0`). User-authored drifts bypass unconditionally (an operator directive must be honoured regardless of the plan cursor).

```python
key = (drift.kind.value, drift.current_task_id or "")
last_addressed = int(session.last_addressed_revision_by_drift_key.get(key, 0))
if last_addressed and drift.observed_revision_index < last_addressed:
    self._verdict_ledger(session)["emitted_redundant"] += 1
    await self._emit_drift_detected(
        session, drift,
        decision_outcome="drift_dropped_stale",
        decision_reason="stale verdict: observed revision N but same (kind, target) addressed at M",
    )
    return
```

Every detector stamps `observed_revision_index` from `session.plan.revision_index` **before** its LLM await (`detect_drift` does the stamping on the positive side of the funnel). The reconciler may transition tasks during that round-trip, so a verdict that arrives after the framework already addressed the *same `(kind, target)`* at a later revision is redundant. The gate is **per-`(kind, target)`**, NOT a naive `observed < live_revision` — that would over-reject a `GOAL_DRIFT` verdict observed at N just because an unrelated `OFF_TOPIC` refine bumped the plan to N+1. The watermark `last_addressed_revision_by_drift_key` is stamped by `PlanReviser._apply_revision`'s install path (under the plan lock, inside `_emit_plan_revised`) after every successful goldfive-authored refine.

**Drop outcome (#480):** `decision_outcome="drift_dropped_stale"` distinguishes this from a real fire (`"drift_emitted"`) in the optimizer's training set. The empty-target `""` coalesces trajectory-level drifts onto one key.

### Gate 3 — In-flight-refine keys (goldfive#405 MEDIUM #4)

Also only for goldfive-authored, stamped drifts. Closes the concurrent-refine race the watermark cannot: two judges that observed the same `(kind, current_task_id)` at the same revision both read `last_addressed == 0` and both pass Gate 2. The in-flight set closes that window synchronously.

```python
inflight_key = (str(session.id or session.run_id or ""), drift.kind.value, drift.current_task_id or "")
if inflight_key in self._inflight_refine_keys:
    self._verdict_ledger(session)["emitted_redundant"] += 1
    await self._emit_drift_detected(
        session, drift,
        decision_outcome="drift_dropped_inflight",
        decision_reason="concurrent refine already in-flight for (kind, target)",
    )
    return
self._inflight_refine_keys.add(inflight_key)
```

The key is `(session.id, kind.value, current_task_id)` — session-scoped because one steerer is shared across sessions in multi-runner processes. Cleared in the `finally` (success, exception, or cancel) so a single crash can't wedge a key permanently:

```python
try:
    await self._handle_drift_dispatch(drift, session)
finally:
    if inflight_key is not None:
        self._inflight_refine_keys.discard(inflight_key)
    drift_id = str(getattr(drift, "id", "") or "")
    if drift_id:
        self._cancelled_drift_ids.discard(drift_id)   # #405 MEDIUM #6
```

**Do not** widen the plan lock to cover `planner.refine` instead of using this set — refine is a multi-second LLM round-trip and the lock would serialise unrelated drift handling on the same session. That trade-off is deliberate; the in-flight set is the intended lighter-weight close.

---

## 5. `_handle_drift_dispatch` — the dispatch gates IN ORDER

Once the entry gates pass, `_handle_drift_dispatch` runs the actual cancel + ladder. Order again matters.

### Step A — Tag adapter cancel reason

```python
self._tag_adapter_cancel_reason(drift, session=session)
```

Writes a symbolic reason (`user_steer`, `goldfive_<kind>`, or `drift`) onto the bound adapter's `_next_cancel_reason` so the synthetic `function_response` the adapter appends on cancel carries LLM-actionable content. Done **before** the drift is emitted so a sink that reacts by cancelling sees the tag. Duck-typed; harmless if no adapter is bound.

### Step B — `USER_STEER` side effects

```python
if drift.kind is DriftKind.USER_STEER:
    _token = self._steerer._active_session_var.set(session)
    try:
        await self._apply_user_steer_state(drift, session)
    finally:
        self._steerer._active_session_var.reset(_token)
```

`_apply_user_steer_state` stamps `goldfive.active_steer.*` onto `session.state` (body, `at_turn = session._reasoning_turn`, author, `source="user"`) and records the source annotation id in `processed_steer_ids` for dedupe. Done **before** the drift emit and **before** refine so the refine sees the new goal shape in the same dispatch. Since Phase 4 (#271), goal synthesis is `Planner.handle_turn`'s job — this method retains only bookkeeping. The ContextVar plumbing exposes `session` to the planner's span-context provider for the duration.

### Step C — Promotion decision + drift emit

```python
promote_to_steer = self._should_promote_to_steer(drift, session)
await self._emit_drift_detected(session, drift)
if drift.suppressed_by_user_steer:
    return   # user-steer suppression window
```

`_should_promote_to_steer` is consulted **before** the drift emit so a suppressed goldfive steer carries `suppressed_by_user_steer=True` on the wire (§6). If suppression won, we emit the `DriftDetected` (observability) and return — no cancel, no refine, no passive-ladder dispatch (a user steer is already active and its refine already happened; running another refine here would race it). See §6 for the promotion policy in full.

### Step D — Cooperative cancel (CRITICAL only)

```python
if self._should_request_cancel_for_drift(drift):
    drift_id = str(getattr(drift, "id", "") or "")
    if drift_id:
        self._cancelled_drift_ids.add(drift_id)   # #405 MEDIUM #6: dedup the post-refine cancel
    try:
        await self.request_invocation_cancel(drift=drift, session=session)
    except Exception:
        ...  # best-effort
```

`_should_request_cancel_for_drift` returns `True` iff the drift is `USER_STEER`/`USER_CANCEL`/`USER_PAUSE` (operator bypass) OR `severity is CRITICAL`. INFO/WARNING never cancel — they flow through the observe/absorb/nudge channels. The `_cancelled_drift_ids` stamp (keyed on `drift.id` — unique per dispatch, a legitimate per-dispatch use, added-and-discarded within one `handle_drift`) lets the later `_cancel_inflight_for_revision` short-circuit its second hard cancel for the same drift. `request_invocation_cancel` itself is `observation_only`-gated (§8). The stamp is written *before* the await so the dedup engages even if the cancel call yields the loop.

### Step E — Promote branch

```python
if promote_to_steer:
    await self._promote_drift_to_steer(drift, session)
    return
```

### Step F — Ladder level + transition telemetry

```python
occurrence_count = self._occurrence_count_for_ladder(session, drift)
level = self._ladder_level_for(drift.kind, drift.severity, occurrence_count)
ladder_reason = "first occurrence" if occurrence_count == 0 else f"repeat (count={occurrence_count})"
await self._emit_ladder_transition(session=session, from_level="", to_level=level.name.lower(), reason=ladder_reason, drift=drift)
```

`from_level` is always empty — the ladder is stateless per call; consumers reconstruct transitions by joining `LadderTransitionDecided` events on `(drift_kind, task_id)` ordered by `sequence`.

### Step G — Terminal-level dispatch

```python
if level is InterventionLevel.OBSERVE:
    return
if level is InterventionLevel.NUDGE:
    await self._dispatch_nudge(drift, session); return
if level is InterventionLevel.PAUSE_ESCALATE:
    await self._dispatch_pause_escalate(drift, session); return
if level is InterventionLevel.TERMINATE:
    await self._dispatch_pause_escalate(drift, session, terminate=True); return
```

Note `TERMINATE` calls the same `_dispatch_pause_escalate` with `terminate=True`. Pre-#482 this branch silently fell through to another `PAUSE_ESCALATE`, making the `(PAUSE_ESCALATE, TERMINATE)` ladder rows identical. See §9.

### Step H — Refine path (ABSORB and CANCEL_REINVOKE)

Both refine; CANCEL_REINVOKE additionally dispatches a `GOLDFIVE_STEER` control message afterward. The refine machinery has its own gate stack (still inside `_handle_drift_dispatch`), in order:

**H1 — planner/plan presence:**
```python
if self._steerer._planner is None or session.plan is None:
    return
```

**H2 — `REFINE_VALIDATION_FAILED` belt-and-braces:**
```python
if drift.kind is DriftKind.REFINE_VALIDATION_FAILED:
    return   # terminal planner signal; ladder already routes it to Level 4
```

**H3 — Refine-outcome gate (goldfive#215 iter-8 P2).** Skip refine when `(kind, task)` already has a terminal outcome on this turn. `USER_STEER`/`USER_CANCEL`/`GOAL_DRIFT` bypass (they are in `_USER_OR_TRAJECTORY_DRIFT_KINDS`):

```python
if drift.kind not in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
    outcome_key = (drift.kind.value, drift.current_task_id or "")
    outcome = session.refine_outcomes.get(outcome_key)
    if outcome is not None:
        if outcome.state == "succeeded":
            await self._emit_policy_applied(
                policy_name="refine_outcome_succeeded_skip", outcome="skipped",
                reason="prior_succeeded_same_turn", ...)
            return
        if outcome.fail_count >= self._steerer.REFINE_FAILURE_THRESHOLD:
            await self._emit_policy_applied(
                policy_name="refine_failure_threshold", outcome="suppressed",
                reason="threshold_reached", ...)
            return
```

This is the "throttle/backoff" surface — goldfive has **no time-based cooldown** (deliberately, per the structural-steering directive; there is no wall-clock gate). The rate limiting is *outcome*-based: a `(kind, task)` that already succeeded this turn is a no-op replay; one that failed `REFINE_FAILURE_THRESHOLD` (2) times already tripped the threshold in `_record_refine_outcome` (which marked the task FAILED and emitted `REPEATED_FAILURE`), so a third tick must not retry. `PolicyApplied` records both drops with distinct `policy_name` / `outcome` so an optimizer can count them.

**H4 — Progress-stall escalation (goldfive#271).** Orthogonal to H3:

```python
if self._is_task_progress_stalled(drift, session):
    await self._emit_progress_stalled_escalation(drift, session)
    return
```

`_is_task_progress_stalled` returns `True` when the task's last progress stamp is older than `PROGRESS_STALL_THRESHOLD_SECONDS` (default `600.0`). A productively-iterating task emits continuous progress events; a stuck one does not. `0` disables the gate (tests). `USER_OR_TRAJECTORY` kinds bypass (trajectory drifts have no single task whose progress could be measured). This escalates to `HUMAN_INTERVENTION_REQUIRED` instead of looping the planner.

**H5 — The refine call itself.** Wrapped in `_state_audit.cancellation_stash_audited(...)` (the Phase-3.5 tripwire; see CANCELLATION-CONTRACT.md). Threads the adapter's `available_agents_tree` (#151) through when `planner.refine` accepts it (probed by `_planner_refine_accepts_available_agents`; falls back to the kwarg-less call for pre-#151 stubs). The exception arms:

| Exception | Handling | Escalation |
| --- | --- | --- |
| `RefineExhausted` | planner explicitly signalled it cannot produce a meaningful change | `_emit_refine_failed(refine_exhausted)` + `_emit_handler_exhausted_escalation` → HUMAN_INTERVENTION |
| `Exception` (LLM error, malformed JSON) | log + `_emit_refine_failed(llm_error)` | `_escalate_refine_failure_as_critical_drift` + `_record_refine_outcome(succeeded=False)` |
| `BaseException` (incl. `CancelledError`) | emit `_emit_refine_failed(cancelled)`, `mark_stash_completed()`, **re-raise** | cancellation propagates per asyncio contract |

**H6 — `revised is None`:** planner exhausted its internal retry budget (#204). Treat as handler exhaustion → `_emit_refine_failed(parse_error)` + `_record_refine_outcome(succeeded=False)` + `_emit_handler_exhausted_escalation`. Do NOT emit a fresh CRITICAL that would recurse through `handle_drift` and eventually abort the run.

**H7 — Fold runtime terminal statuses:** `_fold_runtime_terminal_statuses(revised, session.plan)` folds out-of-band terminal transitions (overlay reap → NOT_NEEDED, reachability audit → CANCELLED, reporting tool → COMPLETED) onto the revised plan **before** validation. `Plan` is frozen (#247) so this returns a new instance.

**H8 — Validation:** `revised.validate(for_revision=True, prior=session.plan)`. On `ValueError`: emit `_emit_refine_failed(validator_rejected)`, emit an **INFO** `SCHEMA_VIOLATION` for observability (does NOT recurse through `handle_drift`), `_record_refine_outcome(succeeded=False)`, `_emit_handler_exhausted_escalation`. Passing `prior=session.plan` enables terminal-task and terminal→terminal edge preservation (PLAN-LIFECYCLE.md §3.1/§3.2).

**H9 — No-op revision rejection (goldfive#271):** if `_plans_structurally_identical(session.plan, revised)`, the "refine" produced no real change. Escalate to HUMAN_INTERVENTION (`_emit_refine_failed(no_op_revision)`) rather than bumping the revision index for a no-op (which would loop forever on a judge that keeps re-firing on a corrected task).

**H10 — Success path:**
```python
await self._record_refine_outcome(session, drift, succeeded=True)
prev_plan = session.plan
revised, was_installed = self._steerer.plans._apply_revision(session, revised, drift)
await self._cancel_inflight_for_revision(drift, session)
await self._steerer.plans._emit_plan_revised(session, revised, drift, prev_plan=prev_plan, attempt_id=attempt_id, dry_run=not was_installed)
if level is InterventionLevel.CANCEL_REINVOKE:
    await self._dispatch_goldfive_steer_control(drift, session)
if level is InterventionLevel.ABSORB and drift.kind in _ABSORB_NUDGE_KINDS:
    # post-ABSORB nudge (gated) — see §5.1
```

The order — `_apply_revision` (which returns `(revised, was_installed)`; the actual `session.plan` swap is *deferred into* `_emit_plan_revised`'s lock per #403), then `_cancel_inflight_for_revision`, then `_emit_plan_revised` — is the canonical pattern. Cancel-before-emit so the synthetic `InvocationCancelled` sink event lands adjacent to the revision on the Gantt.

### 5.1 The post-ABSORB nudge (goldfive#202)

For a scoped set of "coordinator-stuck" kinds, a *successful ABSORB* also queues a Level-2 nudge:

```python
# goldfive/steerer.py
_ABSORB_NUDGE_KINDS = frozenset({
    DriftKind.LOOPING_REASONING,
    DriftKind.LOOPING_TOOL_CALL,
    DriftKind.SELF_REPORTED_STUCK,
    DriftKind.GOAL_DRIFT,
})
```

The rationale: these detectors fire while the coordinator is still mid-invocation, retrying the superseded task. The only way for it to learn its plan changed is a synthetic user message at invocation end. Other ABSORB kinds recover at the next task boundary. **This enqueue is `observation_only`-gated** (it becomes an injected user turn):

```python
if level is InterventionLevel.ABSORB and drift.kind in _ABSORB_NUDGE_KINDS:
    nudge_msg = compose_corrective_user_message(drift=drift, refined_plan=session.plan)
    if not self._steerer.is_active_steering():
        log.info("observation_only=True — SKIPPING post-ABSORB nudge enqueue ...")
        await self._emit_policy_applied(
            policy_name="observation_only_gate", outcome="suppressed",
            reason="observation_only=True", detail="intervention=post_absorb_nudge ...")
        return
    session.pending_nudges.append(nudge_msg)
    if was_installed:
        session.pending_nudges_revision_installed = True
```

---

## 6. The promotion path (goldfive-steer unification)

Some goldfive-detected drifts should ride the **same** cancel-and-restart machinery as an operator `STEER` rather than the passive ladder. `_should_promote_to_steer` decides; `_promote_drift_to_steer` executes.

### `_should_promote_to_steer`

Returns `True` iff, in order:

1. Not user-authored (`USER_STEER`/`USER_CANCEL`/`USER_PAUSE` keep their pre-unification handling).
2. `_resolve_authored_by(drift) == "goldfive"`.
3. `drift.kind in _GOLDFIVE_STEER_ELIGIBLE_KINDS`:
   ```python
   _GOLDFIVE_STEER_ELIGIBLE_KINDS = frozenset({
       DriftKind.OFF_TOPIC, DriftKind.INTENT_DIVERGENCE, DriftKind.UNEXPECTED_OUTPUT,
       DriftKind.CONFABULATION_RISK, DriftKind.LOOPING_REASONING,
       DriftKind.LOOPING_TOOL_CALL, DriftKind.PLAN_DIVERGENCE,
   })
   ```
   Other kinds (SCHEMA_VIOLATION, REFINE_VALIDATION_FAILED, REPEATED_FAILURE, GOAL_DRIFT, …) keep their legacy ladder mapping so escalation / repeated-failure semantics are not rerouted into the cancel-in-flight path.
4. Severity clears the configured threshold (`_severity_meets_promotion_threshold`): `"off"` → never; `"critical"` → only CRITICAL; `"warning"` → WARNING and CRITICAL. Config `SteeringConfig.threshold`, default `"warning"`.
5. No fresh user steer is blocking it. If `StateStore.for_session(session).get_active_steer()` returns a `source == "user"` steer whose age in **logical turns** is within the window, stamp `drift.suppressed_by_user_steer = True` and return `False`.

The freshness window is measured in **logical turns** (`session._reasoning_turn`, one tick per reasoning observation), NOT `_next_sequence` — goldfive#441 fixed a bug where decision-telemetry event volume (from #436/#440) inflated the per-event counter and shrank the effective window:

```python
current_turn = int(getattr(session, "_reasoning_turn", 0) or 0)
age = current_turn - active.at_turn
if 0 <= age < window:
    drift.suppressed_by_user_steer = True
    return False
```

Window is `SteeringConfig.suppression_window_turns`, default `3`. `0` disables suppression.

### `_promote_drift_to_steer`

Mirrors the USER_STEER path. Ordered side effects:

1. Tag adapter cancel reason `goldfive_<kind>` (via `_tag_adapter_cancel_reason_for_promotion`), stamp `session._last_cancel_reason_prefix`.
2. Stamp `goldfive.active_steer.*` (`author="goldfive"`, `source="goldfive"`, `at_turn=session._reasoning_turn`, body from `_compose_goldfive_steer_body(drift)`).
3. Record `drift.id` in `processed_steer_ids` so a redelivery cannot re-promote.
4. Refine via `planner.refine_steer(...)` (falls back to `planner.refine` when the planner lacks the goldfive-specific entry point). Same gate stack as H3/H4 (outcome gate, progress-stall) and same exception arms as H5.
5. Fold/validate/no-op-reject (same as H7-H9).
6. `_apply_revision` → `_cancel_inflight_for_revision` → `_emit_plan_revised` → `_dispatch_goldfive_steer_control(drift, session, body_override=body)`.

**Audit #402 (preserved-then-fixed):** the `GOLDFIVE_STEER` dispatch fires **after** `_emit_plan_revised` swaps `session.plan` to the revised version, so the payload's `replacement_task_ids` read the NEW plan's PENDING tasks. Pre-fix the dispatch fired before refine and pointed the executor's restart at tasks the imminent revision was about to remove. Do not move it back.

**#241 emergency-revert scar:** `_promote_drift_to_steer` does NOT call `adapter.request_cancel(reason)` to kill the in-flight LLM call immediately. That `task.cancel()` used to propagate a `CancelledError` past the executor's invocation-scope catch and killed the entire run (observed as `run_aborted` right after a goldfive-detected drift). The deferred-cancel semantics (tag `_next_cancel_reason` + queue a restart message; the executor picks it up at the next boundary, and `_cancel_inflight_for_revision` scopes a `task.cancel()` only AFTER the superseding plan is installed) are intentional. Letting the in-flight call run to completion is the lesser evil vs. aborting the run.

---

## 7. The twin refine pipelines — WARN: keep them in sync

There are **two** near-identical refine implementations:

| | `_handle_drift_dispatch` (Step H) | `_promote_drift_to_steer` (§6) |
| --- | --- | --- |
| Entry | ABSORB / CANCEL_REINVOKE ladder levels | promotion policy (`_should_promote_to_steer`) |
| Refine call | `planner.refine(...)` | `planner.refine_steer(...)` (falls back to `refine`) |
| Outcome gate (H3) | yes (`PolicyApplied` on drop) | yes (same logic, but returns silently — **no `PolicyApplied` emit on the drop**) |
| Progress-stall (H4) | yes | yes |
| Exception arms | RefineExhausted / Exception / BaseException | identical |
| `revised is None` | exhaustion escalation | identical |
| Fold / validate / no-op | identical | identical (validate's `SCHEMA_VIOLATION` fallback stamps `authored_by="goldfive"` here) |
| Install order | `_apply_revision` → `_cancel_inflight_for_revision` → `_emit_plan_revised` | identical |
| Follow-up dispatch | `GOLDFIVE_STEER` only if `level is CANCEL_REINVOKE` | always `GOLDFIVE_STEER` with `body_override=body` |

**Known deltas** (do not "fix" without understanding them):
- The promotion path's outcome-gate drop is silent (no `PolicyApplied`); the ladder path emits `refine_outcome_succeeded_skip` / `refine_failure_threshold`.
- The promotion path stamps active-steer state and a cancel-reason prefix before refining; the ladder path does not (only USER_STEER does, in Step B).
- The promotion path's `SCHEMA_VIOLATION` fallback stamps `authored_by="goldfive"` on the synthetic INFO drift; the ladder path leaves it default.

> **DEFERRED WORK — do not treat as done.** A "twin-refine-pipeline extraction" that would collapse these two into one shared helper is planned but **blocked on the agency-preservation branch-merge decision** (see 17-invariants-hazards-history.md / MEMORY). Until that lands: **any edit to the refine flow must be applied to BOTH methods.** A fix that only touches `_handle_drift_dispatch` and not `_promote_drift_to_steer` (or vice versa) is a bug. Grep both before you commit.

Likewise, the ~7 stacked suppression gates in `handle_drift`/`_handle_drift_dispatch` (freshness watermark, in-flight keys, suppression window, outcome-succeeded, outcome-failure-threshold, progress-stall, no-op-revision) are slated for replacement by a single **evidence-ledger** — also blocked on the same branch decision. Present them as the current design; do not build a competing consolidation on main.

---

## 8. The observation_only contract — the spec

`observation_only=True` is the **production default** (`SteeringConfig.observation_only: bool = True`). Detection runs in full — every judge, every embedding detector, goal-drift, looping detectors, `CAPABILITY_MISMATCH` — and `planner.refine`/`refine_steer` still run so operators can see the *would-have-applied* plan via `PlanRevised(dry_run=True)`. What is suppressed is every write that touches the live agent.

### The one predicate

```python
# goldfive/steerer.py — DefaultSteerer
def is_active_steering(self) -> bool:
    """True iff interventions may mutate state or inject."""
    return not self._observation_only
```

```python
# goldfive/steerer.py — module helper for consumers holding a maybe-steerer
def steering_is_active(steerer: Any) -> bool:
    predicate = getattr(steerer, "is_active_steering", None)
    if not callable(predicate):
        return False
    try:
        return bool(predicate())
    except Exception:
        return False
```

**Rules (#488):**
- `is_active_steering()` is the ONLY method that reads `self._observation_only`. Everything router-internal calls `is_active_steering()` (or its alias `_should_inject()`, kept because "should inject" reads naturally at the dispatch gates).
- Everything *external* (executors, the ADK plugin, prompt shaper, reporting acks, context editor) calls `steering_is_active(steerer)`.
- The plugin has a thin local wrapper `_is_observation_only(ctx)` = `not steering_is_active(_safe_attr(ctx, "steerer", None))`.
- **PASSIVE is the fail-safe.** A `None` steerer, a steerer missing the predicate, or a predicate that raises → `steering_is_active` returns `False` → surface behaves passively. A surface whose whole purpose is to NOT intervene must not start intervening because a stub steerer forgot a method.
- **`_observation_only` is private.** No code reads it directly except `is_active_steering()`. #488 deleted the old module-global test hook and the autouse fixture that flipped the default; the suite now runs the shipped `observation_only=True` default and ~90 tests opt into active mode explicitly (via the `active_steering_config` / `make_active_steerer` conftest fixtures or an inline `SteeringConfig(observation_only=False)`).

`DefaultSteerer.__init__` sets the flag: `self._observation_only = bool(steering_config.observation_only) if steering_config is not None else True`. A bare `DefaultSteerer()` with no config defaults to passive — matching the production default and avoiding surprising third-party callers who construct it directly.

### The FOUR core gated injection points on the steerer

These are the load-bearing writes. Each checks `is_active_steering()` and, on the passive branch, logs at INFO with a `would_have_*` message + (where an emit path exists) stamps `PolicyApplied(policy_name="observation_only_gate", outcome="suppressed")` and returns without the write.

| # | Injection point | Method | The suppressed write |
| --- | --- | --- | --- |
| 1 | Plan mutation | `PlanReviser._apply_revision` (`goldfive/plan_reviser.py`) | `set_session_plan` + `last_addressed_revision_by_drift_key` stamp |
| 2 | Steer control enqueue | `DriftObserver._dispatch_goldfive_steer_control` | `GOLDFIVE_STEER` ControlMessage onto the channel |
| 3 | Invocation cancel | `DriftObserver.request_invocation_cancel` | plugin cancel-flag write + `task.cancel()` |
| 4 | Nudge enqueue | `DriftObserver._dispatch_nudge` **and** the post-ABSORB handoff | `session.pending_nudges.append(...)` |

**Gate 1 — plan mutation** has a nuance: it suppresses only goldfive-authored **corrective** revisions. Three carve-outs land as REAL revisions even under `observation_only=True`:
```python
# goldfive/plan_reviser.py — _apply_revision
is_bootstrap = prev is None
is_user_authored = (drift.authored_by or "").lower() == "user"
is_discovery = drift.kind is DriftKind.NEW_WORK_DISCOVERED
gate_active = (not is_bootstrap) and (not is_user_authored) and (not is_discovery) and (not self._steerer.is_active_steering())
```
- **bootstrap** (`prev is None`): a cold start has no prior plan; structural, not corrective.
- **user-authored** (`drift.authored_by == "user"`): operator STEER deliveries always land — the operator overrides observation mode.
- **discovery** (`DriftKind.NEW_WORK_DISCOVERED`, #258): the planner/a sub-agent *describing* observed work, not a framework correction. Covers the turn-1 install (where `session.plan` was seeded with `Plan.empty()` so `prev is None` no longer holds) and turn N+1 replans through `install_revision_for_drift`.

When the gate fires, the revision metadata is still stamped and returned so `_emit_plan_revised` can render a faithful `dry_run=True` preview — but `session.plan` is not swapped and the watermark is not stamped. `_emit_plan_revised` re-derives `effective_dry_run = dry_run if dry_run is not None else (not is_active_steering())` and, under dry-run, suppresses the supersedes-integration swap, correction GC/queue, and repin — the whole apply — while still emitting the `PlanRevised` wire event. **dry_run is observability, not silence.**

Gates 2/3/4 are unconditional passive skips (no carve-outs).

### The `_dispatch_goldfive_pause_control` gate (goldfive#264)

`GOLDFIVE_PAUSE_ESCALATE` is *also* gated — a fifth steerer-side write that belongs to the pause path (§9):
```python
# goldfive/drift_observer.py — _dispatch_goldfive_pause_control
if not self._steerer.is_active_steering():
    log.info("observation_only=True — SKIPPING GOLDFIVE_PAUSE_ESCALATE dispatch ...")
    return False
```
Dispatching this message sets `goldfive_pause_message` on the executor's `ControlOutcome`, which drives `_cancel_invoke_task` and ends the overlay turn — killing the live invocation. The originating `HUMAN_INTERVENTION_REQUIRED` drift (emitted by the caller, outside this dispatch) still fires, so observers see the escalation; goldfive just does not cancel. There is a documented live reproduction in the docstring (2026-05-11, session `4538863f-...`) where an OFF_TOPIC drift under `observation_only=True` reached refine exhaustion (#271 no-op path), called this method, dispatched the channel message, and cancelled the in-flight invoke — the carve-out stops that chain.

### The ungated-but-passive-safe surfaces

Not every steering-adjacent read of the predicate is a "core" injection point; several surfaces gate defensively or gate a softer intervention:

| Surface | File | What it gates |
| --- | --- | --- |
| Prompt-shape injections (4 sites) | `prompt_shaper.py` — `should_inject()` → `steering_is_active` | `wrap_conversational_input`, GoldfivePlanner injection, runtime tool-surface injection, goldfive directive injection |
| Executor pause consume | `executors/sequential.py` (`_run_overlay`, GOLDFIVE_PAUSE_ESCALATE branch) | ends the turn / blocks pre-task loop — defense-in-depth for subclasses that bypass the dispatcher |
| Executor nudge drain (`_drain_nudges`) | `executors/sequential.py` | consuming `pending_nudges` and re-invoking — defense-in-depth (clears the queue, ends the turn) |
| Plugin LLM-timeout cancel (#476) | `_adk_plugin.py` (`_run_llm_call_timeout_watcher`) | the cancel-flag write on `LLM_CALL_TIMEOUT`; the drift emit is telemetry and still fires |
| Plugin F3 pre-dispatch redirect (#481) | `_adk_plugin.py` (`before_tool_callback`) | refusing an AgentTool dispatch to a completed-work agent; suppressed → telemetry-only + `PolicyApplied`, dispatch proceeds |
| Plugin tool-approval gate | `_adk_plugin.py` (`before_tool_callback`) | `PolicyApplied(observation_only_gate)` |
| F1 directive acks (#478) | `reporting/rendering.py` (`_directive_ack`, `_idempotent_response`) | the `plan_state` field on the ack — a goldfive-authored directive; under passive the ack keeps only the factual echo of the transition the agent reported |
| Context editor (`ContextEditor.apply`) | `context_editor.py` | the whole context-edit pipeline is hard-skipped under `observation_only` |

The **defense-in-depth** gates in the executor (`_run_overlay` pause branch, `_drain_nudges`) exist because the primary gates are at the steerer's enqueue sites: under `observation_only=True` nothing is queued and these branches never see a nudge/pause. But a *custom steerer subclass* or a direct `session.pending_nudges` writer could drive a goldfive-authored re-invocation there — so the executor gates discard the queue and end the turn rather than injecting. When you add a new consumer of steerer-produced state on the executor side, add the matching defense-in-depth gate.

### The `PolicyApplied` telemetry pattern

Every passive skip that has a session available emits:
```python
await self._emit_policy_applied(
    session=session,
    policy_name="observation_only_gate",
    outcome="suppressed",
    reason="observation_only=True",
    detail="intervention=<name> kind=<kind> task_id=<id>",
)
```
`outcome` is `"suppressed"` for gate skips (`"skipped"` for the refine-outcome-succeeded gate). The drift *drop* labels live on `SteeringDecisionMade` not `PolicyApplied` (`drift_dropped_stale` / `drift_dropped_inflight`). When you add a new gated surface, follow the pattern: log at INFO with a `would_have_*` message AND emit `PolicyApplied(observation_only_gate, suppressed)` so zicato can count would-have-dispatched events. See 12-events-sinks-telemetry.md for the wire schema.

### The late-drift staleness gate (goldfive#319)

Sits in front of `handle_drift` on the background-judge path only (`observe_reasoning`), NOT inside `handle_drift`:

```python
# goldfive/drift_observer.py — observe_reasoning (drift branch)
if self._is_late_drift_for_terminated_invocation(drift, session):
    self._verdict_ledger(session)["emitted_late"] += 1
    if not drift.authored_by:
        drift.authored_by = self._resolve_authored_by(drift)
    await self._emit_drift_detected(session, drift)   # observability only
    return
self._verdict_ledger(session)["acted_on"] += 1
await self.handle_drift(drift, session)
```

`_is_late_drift_for_terminated_invocation` → `_invocation_target_gone(session)` (unless user-authored, which always passes through). "Late" = **no active invocations** OR **any cancel-pending on the session**. The cancel-pending branch closes the iter-11D race (#242): the active-task registry takes 4-8s to drain while ADK winds cancelled invocations down; the cancel-pending flag is stamped synchronously at `request_invocation_cancel` time so a drift handled during that window short-circuits instead of refining against an effectively-dead session. The same `_invocation_target_gone` predicate also gates on-task verdict resolution (§10) so a stale verdict can neither dispatch against nor resolve a fresh condition. The guard is scoped to the background-judge path because only that path produces verdicts that may outlive the originating invocation — synchronous detectors run inline and always see a live invocation.

This is why the conftest `_isolate_orchestration_store_registries` autouse fixture exists — the state-store registries (`_ACTIVE_INVOCATION_TASKS`, `_CANCEL_REQUESTED_INVOCATIONS`) are module-level dicts keyed by `session.id` (aliased to `run_id`); tests sharing `run_id="r1"` would leak a stale cancel-pending entry and silently flip this gate (a CI-only flake).

---

## 9. Pause escalation, the terminus (#482), and TERMINATE

`_dispatch_pause_escalate(drift, session, *, terminate=False)`:

1. `_dispatch_goldfive_pause_control(drift, session, reason=..., terminate=terminate)` — the gated channel write (§8).
2. If the drift IS already `HUMAN_INTERVENTION_REQUIRED`, return (it was emitted at the top of `handle_drift`; just pause).
3. Otherwise: close the originating condition's lifecycle to `human_intervention_required` (`_ostate.escalate_drift_to_human_intervention` on the origin `condition_id`), then emit a **synthetic** CRITICAL `HUMAN_INTERVENTION_REQUIRED` drift directly via `_emit_drift_detected` (NOT through `handle_drift` — that would infinite-loop at CRITICAL).

### The deadline (#482)

`_pause_escalate_deadline_s()` reads `SteeringConfig.pause_escalate_deadline_s` (default `None` = wait forever, the historical Level-4 behaviour). Non-positive is treated as unset. `_dispatch_goldfive_pause_control` builds the payload:

```python
deadline_s = self._pause_escalate_deadline_s()
if terminate and deadline_s is None:
    deadline_s = DEFAULT_TERMINATE_PAUSE_DEADLINE_S   # 600.0
payload = {"reason": reason, "drift_id": ..., "drift_kind": drift.kind.value,
           "ladder_level": "terminate" if terminate else "pause_escalate"}
if deadline_s is not None:
    payload["deadline_s"] = deadline_s
```

- **Level 4 (`PAUSE_ESCALATE`)**: deadline only if the operator configured `pause_escalate_deadline_s`. `None` → the executor's pause wait blocks forever (an operator must resume).
- **Level 5 (`TERMINATE`)**: `terminate=True` ALWAYS attaches a deadline — the configured value, or `DEFAULT_TERMINATE_PAUSE_DEADLINE_S = 600.0`. When it expires, the executor aborts the run (`RunAborted` with escalation lineage). Pre-#482 the `TERMINATE` branch silently degraded to another `PAUSE_ESCALATE`, so the `(PAUSE_ESCALATE, TERMINATE)` rows in the `HUMAN_INTERVENTION_REQUIRED` ladder entry were behaviourally identical — a real terminus never happened.

The only ladder path that reaches `TERMINATE` is a **repeat CRITICAL `HUMAN_INTERVENTION_REQUIRED`** (`_LADDER[HUMAN_INTERVENTION_REQUIRED] = (None, None, (PAUSE_ESCALATE, TERMINATE))`). Executor-side pause/abort semantics are in 04-executors-and-control.md.

---

## 10. Drift-condition lifecycle (#486)

A **drift condition** is a logical group keyed on `sha1(kind, task_id, agent_id, turn_id)` — NOT on the per-event `drift.id`. Multiple emits for the same `(kind, task, agent, turn)` collapse onto one `condition_id`. This is the stable-key invariant in action.

### Stamping (`_stamp_drift_lifecycle`)

Every `_emit_drift_detected` routes through `_stamp_drift_lifecycle`, which calls `_ostate.open_or_escalate_drift(session.state, kind=, task_id=, agent_id=, turn_id=, severity=)`:
- First emit for the tuple in a turn → `DRIFT_LIFECYCLE_OPENED`.
- Subsequent emits → `DRIFT_LIFECYCLE_ESCALATING`, carrying `prev_severity`.

The proto enum mapping is `_drift_lifecycle_pb_value` → `DRIFT_LIFECYCLE_{OPENED,ESCALATING,RESOLVED,HUMAN_INTERVENTION_REQUIRED,UNSPECIFIED}`. Lifecycle stamping is observability-only and wrapped in a broad `except` — a bookkeeping bug must never break the wire emit (it falls through to UNSPECIFIED lifecycle + empty condition_id). A drift with empty `current_task_id`/`current_agent_id` still produces a stable id (the sha1 hashes empty strings), so user-control drifts collapse onto one condition per turn per kind.

### Resolution rules (#486)

Two resolution triggers, both **pure telemetry** — no intervention decision reads them, so behaviour is identical under `observation_only` True/False:

**A. Task-terminal transitions** (`resolve_conditions_for_terminal_task`). Called from `TaskStateMachine.mark_task_*` (see `goldfive/task_state_machine.py`, the `resolve_conditions_for_terminal_task` call) when a task goes COMPLETED / FAILED / CANCELLED / **NOT_NEEDED**. A terminal task moots every condition still open against it; leaving them open makes the active set grow monotonically per run so downstream consumers never see an intervention succeed:
```python
resolved = _ostate.resolve_drifts_matching(session.state, task_id=task_id)
# -> one DriftDetected(lifecycle=RESOLVED) per condition, via _emit_resolved_conditions
```

**B. On-task reasoning verdicts** (`_resolve_conditions_on_on_task_verdict`). When the reasoning judge returns an ON-TASK verdict, it is the clean bill for the current `(task, agent, run)`. Only the kinds *that pipeline* can open resolve:
```python
_REASONING_PIPELINE_DRIFT_KINDS = frozenset({
    DriftKind.LOOPING_REASONING, DriftKind.REASONING_CLUSTER_TIGHTENING,
    DriftKind.OFF_TOPIC, DriftKind.JUSTIFIED_DEVIATION, DriftKind.INTENT_DIVERGENCE,
})
```
Gated on the same staleness predicate as the drift branch (`_invocation_target_gone`) so a verdict landing after its invocation terminated cannot resolve a fresh condition opened by a newer turn. The empty agent_id is accepted (embedding-side detectors open conditions without agent attribution).

**`GOAL_DRIFT` is deliberately ABSENT from the reasoning-pipeline set.** It is opened by the *goal-drift judge*, which answers a trajectory-level question that a reasoning-scoped on-task verdict carries no evidence about. **`GOAL_DRIFT` conditions resolve ONLY at task-terminal** (trigger A), never on an on-task verdict. If you add a kind to `_REASONING_PIPELINE_DRIFT_KINDS`, make sure the reasoning judge is genuinely authoritative over it — otherwise you will resolve conditions the judge has no evidence to close.

### The RESOLVED emit (`_emit_resolved_conditions`)

One `DriftDetected(lifecycle=RESOLVED, severity=INFO)` per resolved condition, with `prev_severity` carrying the condition's last recorded severity (so sinks render "recovered from WARNING"). Deliberately does NOT route through `_emit_drift_detected`: resolution is not a detector decision, so there is no paired `SteeringDecisionMade`/`JudgementEmitted` and no `session.drift_events` append. The state mutation already happened in the caller (`resolve_drifts_matching`), so a missing sink list or proto stub leaves lifecycle truth intact.

### Terminal-drift span cleanup

Separately, when a **terminal drift** fires (`_TERMINAL_DRIFT_KINDS = {HUMAN_INTERVENTION_REQUIRED, REPEATED_FAILURE}`), `_emit_drift_detected` calls `_close_open_boundaries_for_terminal_drift` — the plugin walks its still-open boundaries and emits paired `InvocationBoundaryExited(reason=terminal_drift:<kind>)` so harmonograf's Gantt doesn't render permanently-open spans. Note `LOOPING_REASONING` is intentionally NOT terminal (its CRITICAL-first tier is still recoverable via NUDGE; the eventual escalation to `HUMAN_INTERVENTION_REQUIRED` triggers the close).

---

## 11. Decision telemetry (the #480 label surface)

Every steering decision emits paired events so zicato's optimizer sees both the detector's decision and the steerer's outcome. The emitters live on `DriftObserver`:

| Emitter | Event | Key fields |
| --- | --- | --- |
| `_emit_drift_detected` | `DriftDetected` + paired `SteeringDecisionMade` + `JudgementEmitted` | `detector_name` (#480: a stamped `DriftEvent.detector_name` wins over the kind), `outcome`, `condition_id`, `lifecycle`, `prev_severity`, `observed_revision_index`, `suppressed_by_user_steer` |
| `emit_no_drift_decision` | `SteeringDecisionMade(outcome="no_drift")` | the silent path — a detector ran and decided not to fire |
| `_emit_ladder_transition` | `LadderTransitionDecided` | `from_level` (always empty), `to_level`, `drift_kind` (symbolic `DRIFT_KIND_*` name so `DriftKind.Value` can parse it), `severity` |
| `_emit_policy_applied` | `PolicyApplied` | `policy_name`, `outcome` (`suppressed`/`skipped`), `reason`, `detail` |
| `_emit_detector_dispatch_ordered` | `DetectorDispatchOrdered` | emitted at most once per session |

`SteeringDecisionMade.outcome` values: `"drift_emitted"`, `"drift_suppressed"`, `"no_drift"`, `"drift_dropped_stale"`, `"drift_dropped_inflight"`. The last two are the #480 drop labels — without them, a gate drop is indistinguishable from a real fire in the optimizer's training set. `chosen_severity` stays empty whenever the drift was not actually applied (suppression AND gate drops); only `"drift_emitted"` populates it.

`_detector_name_for_drift`: a `detector_name` stamped on the drift itself wins (the tool-loop tracker emits `LOOPING_REASONING`, same kind as the embedding detector, so the kind alone can't disambiguate the source); falls back to `_DETECTOR_NAME_BY_KIND`, then the bare lowercase kind value.

Other #480 fixes you may encounter: `DriftEvent.detector_name` (the field name for `DriftEvent`), the `drift_dropped_stale`/`drift_dropped_inflight` outcomes, the `capability_check` negative class, and the `ReasoningJudgeInvoked` proto fields 12-15 (`focused_task_id` / `focus_confidence` / `stated_intent` / `provenance`). The last two rows are detector/judge concerns; see 07-deterministic-drift-detection.md and 08-llm-judges.md.

All emitters are **best-effort**: a `ModuleNotFoundError` (proto-less smoke-test env) or any emit exception is swallowed at DEBUG so the routing keeps working. #479 additionally guarantees sink exceptions never abort runs.

The **verdict-utility ledger** (#483) counts `{acted_on, emitted_late, emitted_redundant, parse_fail}` per session (plus a bounded `elapsed_ms` sample list), incremented at the gate sites you saw (`_verdict_ledger(session)["emitted_redundant"] += 1`, etc.), and flushed as a `reasoning_judge_utility_summary` event at run boundary (`drain_session_background_tasks`) with a `shutdown` teardown fallback. #483 also adds the per-steerer judge semaphore (default 3, `ReasoningDriftConfig.max_concurrent_judges`), queued-window coalescing, and an endpoint-contention warning — those are judge-scheduling concerns; see 08-llm-judges.md.

---

## 12. Config knobs (subsystem-scoped)

All on `SteeringConfig` in `goldfive/config.py`. See 14-config-reference.md for the full table; these are the steering-relevant ones:

| Field | Default | Effect |
| --- | --- | --- |
| `observation_only` | `True` | Master kill-switch. `True` = strict-passive (production default). Env `GOLDFIVE_STEER_OBSERVATION_ONLY` (`0`/`false`/`no` → active). |
| `threshold` | `"warning"` | Promotion severity threshold (`off`/`warning`/`critical`) consumed by `_severity_meets_promotion_threshold`. Unknown value falls back to `"warning"` with a warning log. |
| `suppression_window_turns` | `3` | Logical-turn freshness window for user-steer suppression of goldfive promotions. `0` disables. |
| `pause_escalate_deadline_s` | `None` | Level-4 pause deadline. `None`/non-positive = wait forever. |
| `stall_watchdog_enabled` | `False` | Flag-gates the wall-clock stall watchdog (#487) — the sole `TASK_TIMEOUT` producer. |
| `stall_timeout_s` | `600.0` | Silence threshold before the watchdog emits. |
| `name_axis_max_severity` | `"info"` (on `ToolLoopConfig`, not `SteeringConfig`) | #484: caps same-name-varied-args tool-loop severity without `>=2` identical `(name, args_hash)` corroboration. |

Router constants (class attributes on `DefaultSteerer`, tunable by subclasses/tests):
- `REFINE_FAILURE_THRESHOLD = 2` — the "repeat" boundary for the ladder AND the refine-failure-threshold gate.
- `PROGRESS_STALL_THRESHOLD_SECONDS = 600.0` — the progress-stall escalation threshold (`0` disables).

Executor constant: `_MAX_NUDGE_REPLAYS = 3` (`_DEFAULT_MAX_NUDGE_REPLAYS`) in `executors/sequential.py` — bounds nudge re-invocation amplification (#163-style).

Constructor precedence (goldfive#225 pattern, applied to `goldfive_steer_threshold` / `goldfive_steer_suppression_window_turns`): an explicit individual kwarg wins over the `SteeringConfig` dataclass, which wins over the built-in default.

---

## 13. The full gate sequence at a glance

A single ordered checklist of every decision point from `observe` to a landed intervention. Use this as the map when you are lost in `handle_drift`. Each row names the method, the stable key it uses (Invariant 6), and the drop/telemetry it emits.

| # | Where | Check | On fail | Telemetry |
| --- | --- | --- | --- | --- |
| 0 | `observe` | dupe-STEER (`_is_duplicate_steer`) | return | (debug log only) |
| 1 | `observe_reasoning` (judge path only) | late-drift (`_is_late_drift_for_terminated_invocation`) | emit `DriftDetected`, `emitted_late++`, return | `SteeringDecisionMade(drift_emitted)` |
| 2 | `handle_drift` | `PLAN_DIVERGENCE` disabled (Gate 0) | return | (debug log only) |
| 3 | `handle_drift` | freshness watermark (`last_addressed_revision_by_drift_key`, key `(kind, task)`) | emit `DriftDetected`, `emitted_redundant++`, return | `SteeringDecisionMade(drift_dropped_stale)` |
| 4 | `handle_drift` | in-flight-refine (`_inflight_refine_keys`, key `(session_id, kind, task)`) | emit `DriftDetected`, `emitted_redundant++`, return | `SteeringDecisionMade(drift_dropped_inflight)` |
| 5 | `_handle_drift_dispatch` | promotion policy + user-steer suppression window (`_should_promote_to_steer`, key `active_steer.at_turn` vs `_reasoning_turn`) | emit `DriftDetected(suppressed_by_user_steer)`, return | `SteeringDecisionMade(drift_suppressed)` |
| 6 | `_handle_drift_dispatch` | cancel decision (`_should_request_cancel_for_drift`: CRITICAL or USER_*) | (no cancel; continue) | — |
| 7 | `request_invocation_cancel` | `observation_only` gate | return `[]` | INFO log |
| 8 | `_handle_drift_dispatch` | ladder level (`_ladder_level_for`) | OBSERVE → return | `LadderTransitionDecided` |
| 9 | refine (H3) | refine-outcome gate (key `(kind, task)`: succeeded / fail>=threshold) | return | `PolicyApplied(refine_outcome_succeeded_skip / refine_failure_threshold)` |
| 10 | refine (H4) | progress-stall (`task_last_progress_at` vs `PROGRESS_STALL_THRESHOLD_SECONDS`) | escalate HUMAN_INTERVENTION, return | `DriftDetected(HUMAN_INTERVENTION_REQUIRED)` |
| 11 | refine (H6-H9) | `revised is None` / validate / no-op-identical | escalate HUMAN_INTERVENTION, return | `refine_failed(...)` + INFO `SCHEMA_VIOLATION` |
| 12 | `_apply_revision` | `observation_only` gate (unless bootstrap/user/discovery) | skip swap, return `(revised, False)` | `PlanRevised(dry_run=True)` |
| 13 | `_dispatch_goldfive_steer_control` | `observation_only` gate | return `False` | `PolicyApplied(observation_only_gate)` |
| 14 | `_dispatch_nudge` / post-ABSORB | `observation_only` gate | return | `PolicyApplied(observation_only_gate)` |
| 15 | `_drain_nudges` (executor) | `observation_only` defense-in-depth | clear queue, end turn | INFO log |

Rows 3, 4, 5 short-circuit BEFORE any cancel/refine — they are the cheap, common-case drops. Rows 7, 12, 13, 14, 15 are the `observation_only` gates. Rows 9-11 are the refine-side throttle/escalation. Note there is no time-based cooldown anywhere (CM8).

---

## 14. Corrective messages and the nudge drain

The NUDGE / CANCEL_REINVOKE / post-ABSORB paths all build their user-facing message via `compose_corrective_user_message` in `goldfive/steerer.py`. This is the only place goldfive writes text that reaches the agent's LLM, so it is deliberately short, action-focused, and free of goldfive jargon.

### `compose_corrective_user_message`

```python
# goldfive/steerer.py
def compose_corrective_user_message(*, drift, refined_plan):
    current = drift.current_task_id or "the current task"
    next_title = _next_pending_task_title(refined_plan) or "the next planned step"
    next_agent = _next_pending_task_agent(refined_plan) or "the next assigned agent"
    template = _CORRECTIVE_TEMPLATES.get(drift.kind)
    if drift.kind is DriftKind.GOAL_DRIFT and not _task_is_completed(refined_plan, drift.current_task_id):
        template = _GOAL_DRIFT_NOT_COMPLETE_TEMPLATE
    if template is None:
        template = ("The prior attempt on {current_task_id} did not complete "
                    "successfully. Refined plan: proceed with {next_task_title}.")
    return template.format(current_task_id=current, next_task_title=next_title, next_task_agent=next_agent)
```

The `_CORRECTIVE_TEMPLATES` map has one short string per kind (LOOPING_REASONING, LOOPING_TOOL_CALL, PLAN_DIVERGENCE, AGENT_REFUSAL, MODEL_REFUSAL, INTENT_DIVERGENCE, TOOL_ERROR, RUNAWAY_DELEGATION, SELF_REPORTED_STUCK, CONFABULATION_RISK, GOAL_DRIFT). `next_title` / `next_agent` are derived from the first PENDING task on the *refined* plan (`_next_pending_task_title` / `_next_pending_task_agent`); `_next_pending_task_agent` returns the last dot-segment of `assignee_agent_id` so the coordinator gets a name it can pass back as an AgentTool target.

**Truthfulness (the #475 discipline).** The default `GOAL_DRIFT` template asserts "Task '…' is already complete." — but the goal-drift judge can fire while the task is still PENDING/RUNNING or after it FAILED. `compose_corrective_user_message` guards this: it only uses the "already complete" template when `_task_is_completed(refined_plan, drift.current_task_id)` returns `True` (task resolves COMPLETED on the plan); otherwise it falls back to `_GOAL_DRIFT_NOT_COMPLETE_TEMPLATE` ("Set task '…' aside for now. Please proceed to '…' via …") — a directive rather than a status assertion, so the message never claims a completion the plan does not show. When you add a template that asserts a fact about plan state, add the same guard: **a corrective message must not lie about the plan.**

### The nudge drain (`_drain_nudges` in `executors/sequential.py`)

`_dispatch_nudge` and the post-ABSORB handoff only *enqueue* onto `session.pending_nudges` (under the `observation_only` gate). The overlay loop's `_drain_nudges` is what actually re-invokes:

```python
# goldfive/executors/sequential.py — _drain_nudges (abridged)
pending = list(session.pending_nudges)
if (pending and state.nudge_replays < self._MAX_NUDGE_REPLAYS
        and _has_live_pending_or_running(session.plan or plan)):
    if not steering_is_active(steerer):                 # defense-in-depth (#264 pattern)
        session.pending_nudges.clear()
        session.pending_nudges_revision_installed = False
        return False                                    # end turn, never inject
    plan_revised = session.pending_nudges_revision_installed
    session.pending_nudges.clear()
    session.pending_nudges_revision_installed = False
    state.nudge_replays += 1
    state.current_user_input = self._compose_nudge_replay_message(pending, plan_revised=plan_revised)
    # ... re-invoke the passthrough with the composed message
```

Three conditions gate the drain: (1) there ARE pending nudges; (2) `nudge_replays < _MAX_NUDGE_REPLAYS` (3) — the #163-amplification bound: a coordinator whose tree keeps producing nudge-eligible drift on every turn must eventually stop triggering re-invokes; (3) there is still live PENDING/RUNNING work for the tree to do. `pending_nudges_revision_installed` (threaded from the enqueue site's `was_installed`) makes the replay header only claim a plan revision when `_apply_revision` actually installed one — another truthfulness guard. The `steering_is_active` check is defense-in-depth: under the default the enqueue never happened, so this branch never sees a nudge; it exists for custom steerer subclasses or direct `session.pending_nudges` writers.

---

## 15. Cancel internals: `request_invocation_cancel` and `_cancel_inflight_for_revision`

Two cancel entry points, both `observation_only`-gated, both best-effort, both no-op on any adapter/plugin gap (Invariant 3).

### `request_invocation_cancel` (the flag write)

Called from Step D (`handle_drift`, flag-only) and from `_cancel_inflight_for_revision` (post-refine, `cancel_inflight_task=True`). Order of guards:

1. **`observation_only` gate** — `if not self._steerer.is_active_steering(): return []`. Logged at INFO with the drift kind/severity/agent/task so an operator sees WHAT would have been cancelled. This is the first thing the method does — nothing is consulted before it.
2. **Adapter/plugin presence** — no bound adapter → `[]`; no `_plugin` → `[]`; no callable `request_invocation_cancel` on the plugin → `[]`.
3. **Resolve invocation ids** (`_resolve_active_invocation_ids`). Then **stamp cancel-pending synchronously** on the `StateStore` for every id (`mark_invocation_cancel_requested`) BEFORE any async work — this closes the iter-11D race (#242) that the late-drift gate reads.
4. **Empty-id guard** — no resolved ids → `[]` (do not fabricate a cancel on a blank id).
5. Build one `CancellationRequest` (fields: `invocation_id`, `reason` from `_cancel_reason_for_drift`, `severity`, `drift_id`, `drift_kind`, `requested_at_ms`, truncated `detail`) and reuse it for every id.
6. Call `plugin.request_invocation_cancel(invocation_id=, request=, cancel_inflight_task=)`. On `TypeError` (older plugin without the kwarg) fall back to the kwarg-less signature — the task-cancel step is silently skipped but the flag-only contract holds. Aggregate flagged ids (including children the plugin propagates to).

`_cancel_reason_for_drift`: `USER_STEER`→`"user_steer"`, `USER_CANCEL`→`"user_cancel"`, `USER_PAUSE`→`"user_pause"`, else `"drift"`. The reason is operator-visible only (lives on the `InvocationCancelled` sink event), so it carries no prompt-injection risk.

### `_cancel_inflight_for_revision` (the post-refine hard cancel)

Called from H10 / the promotion path right after `_apply_revision` and before `_emit_plan_revised`. It is a goldfive-INTERNAL cancel: the revised plan has just been installed and the cancel is the mechanism that switches the in-flight agent onto it. Contract:

1. **Stamp `session._supersede_pending = True`** BEFORE the cancel so the executor's overlay-loop cancelled branch can distinguish this internal supersede from an external cancel (USER_CANCEL, `asyncio.CancelledError`) and restart the passthrough loop with the new plan instead of aborting the turn. Best-effort; the flag is idempotent and harmless if unconsumed.
2. **Double-cancel dedup (#405 MEDIUM #6)** — if `drift.id in self._cancelled_drift_ids` (stamped at the top of Step D for the same CRITICAL drift), short-circuit: the flag write the executor consumes is idempotent, and firing a SECOND cancel could land on a *different* invocation than the first if the executor already restarted in response to the first flag's channel-message restart. Discards the id and returns `[]`.
3. **Per-invocation supersede registry (#405 LOW #7)** — `store.mark_supersede_pending(inv_id)` for each active id, so a concurrent overlay iteration's defensive `_supersede_pending = False` clear cannot drop the signal for an unrelated invocation.
4. Delegate to `request_invocation_cancel(..., cancel_inflight_task=True)` — the `task.cancel()` fires (deferred via `loop.call_soon`) so the coordinator's in-flight LLM call observes `CancelledError` within ~one event-loop tick rather than the full LLM-call duration.

`install_initial_plan` (turn-1 first-plan install) skips this path directly — there is no in-flight invocation to cancel on a fresh session.

### `GOLDFIVE_STEER` payload (`_dispatch_goldfive_steer_control`)

After a CANCEL_REINVOKE refine (or in the promotion path), the executor gets a `GOLDFIVE_STEER` `ControlMessage`:

```python
payload = {
    "drift_kind": drift.kind.value,
    "drift_id": str(drift.id or ""),
    "body": body,                       # compose_corrective_user_message or body_override
    "superseded_task_ids": [drift.current_task_id] if drift.current_task_id else [],
    "replacement_task_ids": [first PENDING task id on the REVISED plan],
}
```

`replacement_task_ids` is derived by walking `session.plan` (the just-installed revision) for the first PENDING task — the executor renders an explicit "pick these up instead" block in the restart message. This is why the dispatch must fire AFTER `_emit_plan_revised` (audit #402 / CM12): before the swap, `session.plan` is still the prior plan and the ids would point at tasks the revision removes. The whole method is `observation_only`-gated (returns `False`, logs the would-be payload, stamps `PolicyApplied`).

`USER_*` drift severities (from `_drift_from_control`): `STEER`→WARNING, `CANCEL`→CRITICAL, `PAUSE`→INFO. All carry `authored_by="user"` and the raw `ControlMessage` on `drift.raw`.

---

## 16. `_record_refine_outcome` — the failure counter and REPEATED_FAILURE

`_record_refine_outcome(session, drift, *, succeeded)` is the per-turn `(kind, task)` bookkeeping that both the ladder occurrence-count and the H3 refine-outcome gate read. It is the only writer of `session.refine_outcomes` within the steerer/DriftObserver subsystem. (The alternate `ParallelDAGExecutor` in `executors/parallel.py` has its own refine machinery that writes the same table directly — but without the threshold-crossing REPEATED_FAILURE side effect — so it is not part of this subsystem's counter contract.)

- `USER_STEER` / `USER_CANCEL` / `GOAL_DRIFT` (`_USER_OR_TRAJECTORY_DRIFT_KINDS`) bypass the write entirely — operator intent always honoured; trajectory drifts have their own rate limiters.
- `succeeded=True` → `RefineOutcome(state="succeeded", fail_count=0)`. The "succeeded" state still encodes "attempted" so a follow-up same-`(kind, task)` drift on the same turn skips refine (H3).
- `succeeded=False` → increment `fail_count` (init to 1). If `new_count < REFINE_FAILURE_THRESHOLD` (2), return. Otherwise **cross the threshold**: `mark_task_failed(task_id, reason=..., recoverable=False)` (routes through `handle_drift` on a `TASK_FAILED_FATAL` key — a DIFFERENT `(kind, task)` tuple, so no recursion into this counter) AND emit a CRITICAL `REPEATED_FAILURE` drift directly via `_emit_drift_detected` (NOT `handle_drift` — that would refine again on the fresh drift). This is the CM7 pattern in action.

`reset_for_turn(session)` clears `session.refine_outcomes` at the top of every `Runner.run` (right after `run_started`) so the retry budget is per-turn — a wedged drift from a prior turn must not carry its failure count into a fresh refine attempt.

---

## 17. Recipes

### Recipe A — Route a new `DriftKind` through the ladder

1. Add the kind to `DriftKind` in `goldfive/types.py` (see 07/17 for taxonomy rules).
2. Add a row to `_LADDER` in `_load_ladder_tables` (`drift_observer.py`): `(info_level, warning_level, (critical_first, critical_repeat))`. Use `None` for INFO/WARNING to fall back to `OBSERVE`. If you skip this, the generic fallback (INFO→OBSERVE, WARNING→ABSORB, CRITICAL→ABSORB/PAUSE_ESCALATE) applies — fine for many kinds.
3. If the kind should ride the promotion (cancel-and-restart) path, add it to `_GOLDFIVE_STEER_ELIGIBLE_KINDS`.
4. If a successful WARNING ABSORB should also queue a mid-invocation nudge, add it to `_ABSORB_NUDGE_KINDS` (`steerer.py`) AND add a template to `_CORRECTIVE_TEMPLATES`.
5. If an on-task reasoning verdict should resolve conditions of this kind, add it to `_REASONING_PIPELINE_DRIFT_KINDS` — but ONLY if the reasoning judge is authoritative over it (CM10).
6. Add a detector-name entry to `_DETECTOR_NAME_BY_KIND` if the kind has a stable single source.
7. Test: add a row to `tests/test_intervention_ladder.py` asserting the `(severity, occurrence) → level` mapping.

### Recipe B — Add a new gated intervention surface (the CM1 checklist)

Any new code that cancels, injects a message, refuses a dispatch, or mutates agent/ADK state:

1. Resolve the predicate: router-internal → `self._steerer.is_active_steering()`; external → `steering_is_active(steerer)`; plugin → `_is_observation_only(ctx)`.
2. Gate: `if not <active>: log.info("... observation_only=True — SKIPPING <surface>. would_have_...") ; <emit PolicyApplied(observation_only_gate, suppressed) if a session is in scope>; return <passive value>`.
3. If the surface is downstream of steerer-produced state (a queue, a flag), add a defense-in-depth gate at the *consumer* too (like `_drain_nudges`).
4. Write TWO tests: `observation_only=False` asserts the write happens; `observation_only=True` asserts it is suppressed AND the telemetry fired. Model them on `tests/test_observation_only_nudge_gate.py` / `tests/test_observation_only_strict_passive.py`.
5. Run the full `tests/test_observation_only_*.py` set and `grep -rn '_observation_only' goldfive/` to confirm you did not introduce a direct read.

### Recipe C — Add a new suppression gate with a stable key

1. Choose a **stable** key (Invariant 6): a tuple of `(kind.value, current_task_id)` or `(session_id, kind.value, current_task_id)` — NOT `drift.id`, NOT an LLM-minted id.
2. Store the gate state on the `DriftObserver` instance (per-session scope) or on `session.state` via `_ostate` (if it must survive across the run). If it is a per-dispatch set, clear it in the `handle_drift` `finally`.
3. Place the check in `handle_drift` (entry) or the H-stack (refine), before the expensive refine await.
4. Emit a distinct `SteeringDecisionMade` outcome or `PolicyApplied` policy_name so the drop is countable (do NOT reuse `drift_emitted`).
5. If the key can churn (task ids re-minted, etc.), fix the churn upstream — do NOT coarsen the key (see the "stable identity keys" MEMORY entry).

---

## 18. Two worked traces

### Trace 1 — A CRITICAL `OFF_TOPIC` drift under `observation_only=True` (production default)

1. Reasoning judge (background) returns an OFF_TOPIC verdict → `observe_reasoning`.
2. Late-drift gate: invocation still live → pass. `acted_on++`. → `handle_drift`.
3. Gate 0 (not PLAN_DIVERGENCE), Gate 1 (`authored_by="goldfive"`), Gate 2 (fresh), Gate 3 (not in-flight) → pass. `_inflight_refine_keys` stamped.
4. `_handle_drift_dispatch`: `_should_promote_to_steer` → `True` (OFF_TOPIC eligible, CRITICAL ≥ "warning" threshold, no fresh user steer). `_emit_drift_detected` fires (`drift_emitted`). Not suppressed.
5. `_should_request_cancel_for_drift` → CRITICAL → `True`. Stamp `_cancelled_drift_ids`. `request_invocation_cancel`: **`observation_only` gate → returns `[]`, logs "SKIPPING cancel".** No flag written.
6. `promote_to_steer` → `_promote_drift_to_steer`. Refine runs (refine still runs in observation mode). `_apply_revision`: gate_active (goldfive-authored, corrective, not bootstrap/user/discovery) → **skip swap, return `(revised, False)`.** `_emit_plan_revised` emits `PlanRevised(dry_run=True)`. `_cancel_inflight_for_revision`: `request_invocation_cancel` gated → `[]`. `_dispatch_goldfive_steer_control`: **`observation_only` gate → returns `False`, `PolicyApplied(observation_only_gate, suppressed)`.**
7. Net effect: `DriftDetected` + `SteeringDecisionMade(drift_emitted)` + `JudgementEmitted` + `PlanRevised(dry_run=True)` + `PolicyApplied(observation_only_gate)` on the wire. The live agent is **untouched** — it keeps reasoning against the prior plan. Operators see exactly what would have happened.

### Trace 2 — The same drift under `observation_only=False`

Steps 1-4 identical. Then:
5. `request_invocation_cancel`: gate passes → resolve invocation ids → stamp cancel-pending → write the `CancellationRequest` flag on the plugin (+ children). Returns flagged ids.
6. `_promote_drift_to_steer`: refine → `_apply_revision` installs the revised plan (gate inactive) → `_emit_plan_revised(dry_run=False)` swaps `session.plan` under the lock, stamps the `last_addressed_revision_by_drift_key` watermark → `_cancel_inflight_for_revision` fires `task.cancel()` (deferred) so the in-flight LLM call observes `CancelledError` within a tick → `_dispatch_goldfive_steer_control` enqueues the `GOLDFIVE_STEER` message with `replacement_task_ids` from the new plan.
7. The executor's overlay loop sees the cancel (classified as internal supersede via `_supersede_pending`), restarts the passthrough with the `[GOLDFIVE STEERING CONTROL …]` body, and the coordinator picks up the revised plan. Net effect: real intervention.

The ONLY behavioural difference between the two traces is the five gates (rows 7, 12, 13, 14 of §13, plus the `task.cancel()`). Everything else — detection, refine, `PlanRevised` emission — is identical. That symmetry is the whole point of the `observation_only` design: detection and observability are independent of injection.

---

## Common mistakes

Each entry is a wrong edit a weaker model would plausibly make, with the correct alternative.

### CM1 — Adding an intervention surface without the `observation_only` gate

**This is the single highest-severity mistake in this subsystem.** Every historical leak followed the same pattern: a new surface that writes to the live agent shipped WITHOUT the predicate gate, silently making `observation_only=True` runs (the production default) intervene.

| PR | The leak | The fix |
| --- | --- | --- |
| #475 | `_dispatch_nudge` + post-ABSORB nudge enqueued onto `pending_nudges` unconditionally; the overlay drained them into a synthetic user turn | gate both on `is_active_steering()`, emit `PolicyApplied(observation_only_gate)`, keep the nudge text truthful |
| #476 | the LLM-call-timeout watcher cancelled the invocation on `LLM_CALL_TIMEOUT` even under `observation_only` | gate the cancel-flag write on `_is_observation_only(ctx)`; the drift emit is telemetry and still fires |
| #481 | the F3 pre-dispatch redirect refused an AgentTool dispatch under `observation_only` | gate the refusal; under passive it is telemetry-only + `PolicyApplied` and the dispatch proceeds |
| #478 | `report_awaiting_approval` acks carried a goldfive-authored `plan_state` directive under `observation_only`, and could hang with no channel | gate `plan_state` on `steering_is_active`; no-channel → immediate `unavailable` ack; finite 600s default timeout |

**The pattern to internalise: EVERY new surface needs the gate + a both-modes test.**

**DON'T:** add a code path that cancels, injects, refuses, or mutates and assume detection-mode covers it.
**DO:** (1) add `if not <steerer>.is_active_steering(): log + PolicyApplied(observation_only_gate, suppressed); return`, using `is_active_steering()` router-internal or `steering_is_active(steerer)` / `_is_observation_only(ctx)` externally; (2) add a **both-modes test** — one asserting the write happens under `observation_only=False`, one asserting it is suppressed (and telemetry emitted) under `observation_only=True`. Grep the existing `tests/test_observation_only_*.py` for the pattern.

### CM2 — Reading `_observation_only` directly

**DON'T:** `if steerer._observation_only:` or `if not ctx.steerer._observation_only:`.
**DO:** `if not steerer.is_active_steering():` (router-internal) or `if not steering_is_active(steerer):` (external). Direct reads bypass the fail-safe (`None`/missing/raising → passive) and break the #488 single-predicate contract. The only legal direct read is inside `is_active_steering()` itself.

### CM3 — Editing one refine pipeline and not the other

**DON'T:** fix a refine bug in `_handle_drift_dispatch` (Step H) and stop.
**DO:** apply the same fix to `_promote_drift_to_steer` (§6/§7). They are twins pending the deferred extraction. Grep both for the surrounding code (e.g. `_fold_runtime_terminal_statuses`, `_plans_structurally_identical`, `_emit_refine_failed`) and confirm parity. Run `tests/test_promote_drift_to_steer.py` AND `tests/test_intervention_ladder.py`.

### CM4 — Keying a new gate on `drift.id`

**DON'T:** `if drift.id in self._some_new_gate_set:`. `drift.id` is a fresh UUID4 per emit — the gate would open a new entry every observation and never engage (Invariant 6). The one legitimate `drift.id` use is `_cancelled_drift_ids`, a per-*dispatch* dedup that is added and discarded within a single `handle_drift` call, not a cross-observation gate.
**DO:** key on `(kind.value, current_task_id)` (freshness, refine-outcome), `(session_id, kind.value, current_task_id)` (in-flight), or the condition `sha1(kind, task_id, agent_id, turn_id)`. If your gate must survive across observations of the "same logical thing", use one of these stable tuples.

### CM5 — "Fixing" the `LOOPING_REASONING` NUDGE-first ladder row

**DON'T:** change `_LADDER[LOOPING_REASONING]` CRITICAL-first from `NUDGE` to `CANCEL_REINVOKE` "for consistency" with `LOOPING_TOOL_CALL`.
**DO:** leave it. It is a PROTECTED KEEP decision (#204/#206): tool loops deliberately emit `LOOPING_REASONING` and route NUDGE-first at CRITICAL. Similarly, do not delete the `PLAN_DIVERGENCE` `_LADDER` row (disabled at dispatch, machinery is KEEP) or `reconciler.get_missed_tasks` (#163). See 17-invariants-hazards-history.md.

### CM6 — Widening the plan lock around `planner.refine`

**DON'T:** hold `_get_plan_lock(session)` across the `planner.refine(...)` await to close the concurrent-refine race.
**DO:** rely on the `_inflight_refine_keys` set (Gate 3). `refine` is a multi-second LLM round-trip; holding the lock across it serialises unrelated drift handling on the same session and defeats the fire-and-forget judge path (#254). The lock is held only across the consistency-critical region of `_emit_plan_revised` (index bump + supersedes + repin + emit).

### CM7 — Emitting a synthetic escalation drift back through `handle_drift`

**DON'T:** route a `HUMAN_INTERVENTION_REQUIRED` / `REPEATED_FAILURE` / `SCHEMA_VIOLATION` escalation via `handle_drift`.
**DO:** emit it directly via `_emit_drift_detected`. `_dispatch_pause_escalate`, `_record_refine_outcome`, and the validation/no-op fallbacks all do this. Routing a CRITICAL escalation through `handle_drift` would re-enter the ladder at CRITICAL and infinite-loop. The escalation drift must key on a DIFFERENT `(kind, task)` tuple than the source (e.g. `REPEATED_FAILURE` vs the source kind) so it does not feed back into the refine-outcome counter.

### CM8 — Adding a time-based cooldown

**DON'T:** add a wall-clock throttle keyed on "last refine for this kind was N seconds ago".
**DO:** use the outcome-based gate (`session.refine_outcomes` + `REFINE_FAILURE_THRESHOLD`) and the progress-stall escalation. Per the structural-steering directive there is deliberately NO cooldown; `_last_refine_kind` exists but is purely advisory and drives nothing on main. A cooldown reintroduces predictive behaviour (Invariant 4) and starves legitimate multi-step refines.

### CM9 — Forgetting the RESOLVED path is not a detector decision

**DON'T:** make `_emit_resolved_conditions` route through `_emit_drift_detected` or append to `session.drift_events`.
**DO:** keep it a bare wire mirror. Resolution is not a firing; a paired `SteeringDecisionMade`/`JudgementEmitted` would corrupt the optimizer's fire/no-fire training set. Severity is INFO; `prev_severity` carries the recovered-from level.

### CM10 — Adding a kind to `_REASONING_PIPELINE_DRIFT_KINDS` without judge authority

**DON'T:** add `GOAL_DRIFT` (or any trajectory-level kind) to `_REASONING_PIPELINE_DRIFT_KINDS` so on-task verdicts resolve it.
**DO:** only add kinds the reasoning judge is genuinely authoritative over. `GOAL_DRIFT` resolves at task-terminal only (#486) because the reasoning-scoped on-task verdict carries no evidence about a trajectory-level question.

### CM11 — Assuming `NUDGE` or a corrective message is enforcement

**DON'T:** rely on a queued nudge / `GOLDFIVE_STEER` body to stop a runaway agent — those are best-effort messages the agent's LLM may ignore.
**DO:** ensure the enforcement rides the executor/plugin channel (cancel flag, `GOLDFIVE_PAUSE_ESCALATE`, plan swap). Nudges are additions, not the load-bearing mechanism (Invariant 1). Termination must work with an agent that never reads a message.

### CM12 — Moving the `GOLDFIVE_STEER` dispatch before the plan swap

**DON'T:** dispatch `_dispatch_goldfive_steer_control` before `_emit_plan_revised` "to cancel sooner".
**DO:** keep it after `_apply_revision` → `_cancel_inflight_for_revision` → `_emit_plan_revised` (audit #402). The dispatch re-reads `session.plan` to derive `replacement_task_ids`; firing before the swap points the executor's restart at tasks the revision is about to remove.

### CM13 — Reviving `PLAN_DIVERGENCE` dispatch or removing the Gate-0 guard

**DON'T:** delete the top-of-`handle_drift` `PLAN_DIVERGENCE` early-return "because the ladder has a row for it".
**DO:** leave the guard. #252 disabled dispatch (replaced by `CAPABILITY_MISMATCH`/#253) but kept the machinery as a branch KEEP. The guard stops any external producer (replay, sink, legacy caller) from reviving handling.

### CM14 — Double-emitting `JudgementEmitted` for a custom drift-flavoured judge

**DON'T:** in `evaluate_judges`, forward a drift-flavoured verdict to `handle_drift` without setting `drift._judge_emitted_judgement = True`.
**DO:** set the marker. `evaluate_judges` already emitted a `JudgementEmitted` keyed on the judge's real `name`; `_emit_drift_detected` would emit a SECOND one keyed on the drift kind, breaking the "join on judge_name" telemetry contract zicato relies on. The marker is a non-wire runtime attribute the paired-emission path checks.

### CM15 — Blocking the run in a judge or a drift handler

**DON'T:** `await` a slow LLM judge or a refine cascade inline from an ADK callback / reporting-tool handler.
**DO:** spawn it fire-and-forget (`_spawn_drift_handler_background` / the background-judge path) so the callback returns and ADK can dispatch the next turn. This is why the late-drift gate (§8) exists — background verdicts may outlive their invocation. `evaluate_judges` bounds each judge at `JUDGE_EVALUATE_TIMEOUT_S`; a judge that overruns is cancelled and treated as "no signal". A judge that raises is swallowed at WARNING — a misbehaving judge must never break the run or suppress other judges.

### CM16 — Treating `dry_run=True` as "nothing was emitted"

**DON'T:** assume a `PlanRevised(dry_run=True)` means the refine did not run or the operator sees nothing.
**DO:** understand that under `observation_only` the refine DID run (`planner.refine`/`refine_steer` executes in full), the plan was NOT swapped, and `PlanRevised(dry_run=True)` carries the would-have-applied preview so operators can evaluate the intervention. `dry_run` is observability, not silence. Detection and observability are independent of injection.

### CM17 — Adding a knob to `DefaultSteerer.__init__` instead of `SteeringConfig`

**DON'T:** add a new steering behaviour as a bare constructor parameter on `DefaultSteerer`.
**DO:** add it to `SteeringConfig` in `goldfive/config.py` (with an env-var read in `SteeringConfig.from_env` if operators need it) and thread it through `steering_config`. `observation_only` deliberately lives on `SteeringConfig`, not as a constructor param, so operators set it via `RuntimeConfig(steering=SteeringConfig(...))` at `goldfive.wrap` time. Follow the #225 precedence pattern (explicit kwarg > config dataclass > built-in default) only where an existing kwarg already exists.

---

## Verification checklist

Run these after touching anything in this chapter. From the repo root, with the env set up (`uv sync --extra dev --extra adk`).

### 1. Targeted test suites

```bash
# Ladder + level mapping
uv run pytest -q tests/test_intervention_ladder.py

# handle_drift routing, gates, refine flow
uv run pytest -q tests/test_goldfive_drift_routing.py tests/test_steerer.py tests/test_drift_outcomes.py

# observation_only contract — run ALL of these after any gate change
uv run pytest -q tests/test_observation_only.py tests/test_observation_only_strict_passive.py \
  tests/test_observation_only_nudge_gate.py tests/test_observation_only_acks.py \
  tests/test_observation_only_abort_carveout.py tests/test_observation_only_pause_escalate_carveout.py \
  tests/test_observation_only_emit_supersedes_carveout.py

# Promotion path (twin pipeline)
uv run pytest -q tests/test_promote_drift_to_steer.py tests/test_steer_unification.py tests/test_tier2_steer_pipeline.py

# Pause / terminate / deadline
uv run pytest -q tests/test_pause_escalate.py tests/test_pause_deadline.py

# Drift-condition lifecycle + resolution (#486)
uv run pytest -q tests/test_drift_lifecycle.py tests/test_drift_resolution_wiring.py tests/test_terminal_drift_closes_spans.py

# Decision telemetry (#480 labels)
uv run pytest -q tests/test_steering_decision_made.py

# User-steer path + dedupe
uv run pytest -q tests/test_steerer_usersteer.py tests/test_user_steer_invariant.py

# Timeout-cancel gate (#476)
uv run pytest -q tests/test_llm_call_timeout_watcher.py

# Full suite (~30s, ~2912 passed / 61 skipped)
uv run pytest -q
```

### 2. Grep invariants (run before committing)

```bash
# No direct _observation_only reads outside is_active_steering() (Invariant 5 / CM2)
grep -rn '_observation_only' goldfive/ --include=*.py | grep -v 'is_active_steering\|steering_is_active\|def __init__\|self._observation_only: bool\|self._observation_only ='
#   Expected: only the assignment in DefaultSteerer.__init__ and the read in is_active_steering.
#   ANY other hit in goldfive/ (not tests/) is a violation.

# Every gated surface pairs the predicate with a return/skip (audit new surfaces)
grep -rn 'is_active_steering\|steering_is_active\|_is_observation_only' goldfive/ --include=*.py

# Twin-pipeline parity: both refine methods present and both call the shared helpers (CM3)
grep -n '_fold_runtime_terminal_statuses\|_plans_structurally_identical\|_cancel_inflight_for_revision\|_dispatch_goldfive_steer_control' goldfive/drift_observer.py

# No gate keyed on drift.id as a cross-observation key (CM4) — inspect each hit
grep -n 'drift.id\|getattr(drift, "id"' goldfive/drift_observer.py

# Ladder table PROTECTED KEEP rows unchanged (CM5) — eyeball LOOPING_REASONING / PLAN_DIVERGENCE
grep -n 'DriftKind.LOOPING_REASONING\|DriftKind.PLAN_DIVERGENCE' goldfive/drift_observer.py
```

### 3. Lint (must stay clean; do NOT mass-reformat)

```bash
uv run ruff check .
# The repo is intentionally NOT ruff-format-clean — never run `ruff format .`.
```

### 4. Behavioural spot-check (when you touched a dispatch path)

Use the `verify` skill on the affected flow, or drive a minimal run under both modes and diff the `PolicyApplied` / `SteeringDecisionMade` stream: under `observation_only=True` a corrective drift must produce `DriftDetected` + `PlanRevised(dry_run=True)` + `PolicyApplied(observation_only_gate, suppressed)` and NO plan swap / cancel / nudge; under `observation_only=False` the same drift must produce the real install/cancel/nudge. See 15-testing-guide.md for the harness and `tests/test_live_steering_e2e.py` for an end-to-end template.

---

## Cross-references

- **07-deterministic-drift-detection.md** — where `DriftEvent`s come from (tool-error, refusal, stop-reason, tool-loop `name_axis_max_severity` #484). The ladder consumes what these produce.
- **08-llm-judges.md** — the reasoning judge / goal-drift judge, the background-judge scheduling guards (#483), `observe_reasoning`, and the on-task verdict that drives condition resolution.
- **10-planning-and-revision.md** — `PlanReviser._apply_revision` / `_emit_plan_revised` internals, `dry_run`, plan locks, supersedes integration.
- **04-executors-and-control.md** — the overlay loop that drains `pending_nudges` and consumes `GOLDFIVE_STEER` / `GOLDFIVE_PAUSE_ESCALATE`, `_abort_turn` (#489), and `RunAborted`.
- **05-adk-plugin.md** — `request_invocation_cancel` on the plugin side, the LLM-timeout watcher, F3 redirect, `_is_observation_only(ctx)`.
- **11-state-ownership.md** — `StateStore`, `_ostate` condition helpers, the state-audit tripwire, and why cancel/late-drift gates read the invocation registries.
- **12-events-sinks-telemetry.md** — the wire schema for `DriftDetected`, `SteeringDecisionMade`, `LadderTransitionDecided`, `PolicyApplied`, `JudgementEmitted`.
- **13-reporting-tools-and-approval.md** — F1 directive acks (#478) and `report_awaiting_approval`.
- **14-config-reference.md** — full `SteeringConfig` field table and env-var mapping.
- **17-invariants-hazards-history.md** — the PROTECTED KEEP decisions (`LOOPING_TOOL_CALL`/`LOOPING_REASONING` ladder, `PLAN_DIVERGENCE` machinery, `reconciler.get_missed_tasks`), the agency-preservation branch, and the deferred twin-refine / evidence-ledger consolidations.

---

## Appendix A — The escalation emitters

Three places short-circuit the refine path and escalate to a human instead of looping the planner. All three emit a synthetic CRITICAL `HUMAN_INTERVENTION_REQUIRED` drift **directly** via `_emit_drift_detected` (never through `handle_drift` — CM7) and dispatch a `GOLDFIVE_PAUSE_ESCALATE` (which is itself `observation_only`-gated, §8).

| Emitter | Triggered from | Reason string | When |
| --- | --- | --- | --- |
| `_emit_progress_stalled_escalation` | H4 (`_is_task_progress_stalled`) | `"task progress stalled for <kind> on task <id>: <age>s since last progress, threshold <T>s"` | `session.task_last_progress_at[task]` older than `PROGRESS_STALL_THRESHOLD_SECONDS` |
| `_emit_handler_exhausted_escalation` | H6/H8/H9 + `RefineExhausted` | `"refine handler exhausted for <kind> on task <id>: planner cannot produce a meaningful change"` | `revised is None`, validator-rejected, structurally-identical no-op, or explicit `RefineExhausted` |
| `_dispatch_pause_escalate` (Level 4/5) | ladder `PAUSE_ESCALATE` / `TERMINATE` | `"<pause_escalate|terminate> from <kind>: <detail>"` | ladder routed the drift to Level 4 or 5 |

`_emit_progress_stalled_escalation` computes the age from `time.monotonic() - session.task_last_progress_at.get(task_id)`. All three attach `current_task_id` / `current_agent_id` from the source drift so the escalation is correlatable back to the originating condition. `_dispatch_pause_escalate` additionally swaps the *originating* condition's lifecycle to `human_intervention_required` (`_ostate.escalate_drift_to_human_intervention`) so a later `get_active_drift` returns the terminal state on the original condition, not just on the synthesized HUMAN_INTERVENTION row.

Because `HUMAN_INTERVENTION_REQUIRED` is in `_TERMINAL_DRIFT_KINDS`, every one of these emits also triggers `_close_open_boundaries_for_terminal_drift` (§10) — the plugin closes still-open spans so the Gantt doesn't show permanently-open LLM_CALLs after the pause.

---

## Appendix B — Symbol index (the two files at a glance)

A one-line map of the load-bearing symbols, so a weak model can jump to the right method without re-reading 6000 lines. `S` = `goldfive/steerer.py`, `D` = `goldfive/drift_observer.py`, `P` = `goldfive/plan_reviser.py`.

### Router surface (`goldfive/steerer.py`)

| Symbol | Role |
| --- | --- |
| `InterventionLevel` (S) | the ladder enum (OBSERVE..TERMINATE) |
| `steering_is_active(steerer)` (S) | external predicate; missing/None/raising → `False` (passive) |
| `DefaultSteerer.is_active_steering()` (S) | router-internal predicate; the ONLY reader of `_observation_only` |
| `DefaultSteerer._should_inject()` (S) | alias of `is_active_steering()` |
| `compose_corrective_user_message()` (S) | build the short corrective user message (truthful, jargon-free) |
| `_CORRECTIVE_TEMPLATES` / `_GOAL_DRIFT_NOT_COMPLETE_TEMPLATE` (S) | per-kind message templates |
| `_ABSORB_NUDGE_KINDS` (S) | kinds where a successful ABSORB also queues a nudge |
| `RefineExhausted` (S) | planner sentinel → handler exhaustion → HUMAN_INTERVENTION |
| `DefaultSteerer.__init__` (S) | wires config, constructs `tasks`/`plans`/`drift`, sets `_observation_only` |
| `DefaultSteerer.bind` / `bind_adapter` / `bind_control_channel` (S) | run-boundary wiring (sinks, planner, adapter, control channel) |
| `DefaultSteerer._dispatch_goldfive_control(msg)` (S) | send a goldfive ControlMessage on the bound channel (best-effort) |
| `DefaultSteerer.transition(...)` (S) | generic task transition entry (routes to `mark_task_*`) |
| `REFINE_FAILURE_THRESHOLD` = 2 (S) | ladder "repeat" boundary + refine-failure gate |
| `PROGRESS_STALL_THRESHOLD_SECONDS` = 600 (S) | progress-stall escalation threshold |

### Drift-routing surface (`goldfive/drift_observer.py`)

| Symbol | Role |
| --- | --- |
| `DriftObserver.observe(event, session)` (D) | entry: dedupe → control drift → detect → `handle_drift` |
| `DriftObserver.observe_reasoning(...)` (D) | reasoning-judge entry; late-drift gate + on-task resolution live here |
| `detect_drift` (D) | deterministic classifiers (tool-error, refusal, stop-reason); stamps `observed_revision_index` |
| `_drift_from_control` (D) | ControlMessage → `USER_*` drift (STEER=WARNING, CANCEL=CRITICAL, PAUSE=INFO) |
| `_is_duplicate_steer` / `_steer_dedupe_id` (D) | STEER retry dedupe on annotation_id / message id |
| `handle_drift(drift, session)` (D) | entry gates (Gate 0-3) + `try/finally` around dispatch |
| `_handle_drift_dispatch` (D) | Steps A-H: tag, user-steer, promote, emit, cancel, ladder, refine |
| `_LADDER` / `_load_ladder_tables` / `_ladder_level_for` (D) | the ladder table + level resolution |
| `_occurrence_count_for_ladder` (D) | maps `refine_outcomes` → the ladder's `is_repeat` int |
| `_should_promote_to_steer` / `_promote_drift_to_steer` (D) | promotion policy + handler (twin of Step H) |
| `_GOLDFIVE_STEER_ELIGIBLE_KINDS` (D) | kinds eligible for cancel-and-restart promotion |
| `_dispatch_nudge` (D) | Level 2: enqueue `pending_nudges` (gated) |
| `_dispatch_goldfive_steer_control` (D) | Level 3 / promotion: `GOLDFIVE_STEER` enqueue (gated) |
| `_dispatch_pause_escalate` / `_dispatch_goldfive_pause_control` (D) | Level 4/5: pause escalation (gated); deadline logic |
| `_pause_escalate_deadline_s` (D) | reads `pause_escalate_deadline_s`; `None` = wait forever |
| `request_invocation_cancel` (D) | flag write on the plugin (gated); stamps cancel-pending synchronously |
| `_cancel_inflight_for_revision` (D) | post-refine hard cancel; supersede flag + double-cancel dedup |
| `_should_request_cancel_for_drift` / `_cancel_reason_for_drift` (D) | CRITICAL-or-USER cancel decision + reason string |
| `_is_late_drift_for_terminated_invocation` / `_invocation_target_gone` (D) | late-verdict staleness gate |
| `_record_refine_outcome` / `reset_for_turn` (D) | per-turn `(kind, task)` failure counter + REPEATED_FAILURE |
| `_apply_user_steer_state` (D) | USER_STEER bookkeeping (active_steer state + dedupe id) |
| `_emit_drift_detected` (D) | `DriftDetected` + paired `SteeringDecisionMade` + `JudgementEmitted` + lifecycle stamp |
| `emit_no_drift_decision` (D) | the silent path (`SteeringDecisionMade(no_drift)`) |
| `_emit_ladder_transition` / `_emit_policy_applied` / `_emit_detector_dispatch_ordered` (D) | decision telemetry |
| `_stamp_drift_lifecycle` / `_drift_lifecycle_pb_value` (D) | condition OPENED/ESCALATING stamping |
| `resolve_conditions_for_terminal_task` / `_resolve_conditions_on_on_task_verdict` / `_emit_resolved_conditions` (D) | #486 condition RESOLVED |
| `_TERMINAL_DRIFT_KINDS` / `_close_open_boundaries_for_terminal_drift` (D) | terminal-drift span cleanup |
| `_REASONING_PIPELINE_DRIFT_KINDS` (D) | kinds an on-task verdict may resolve (GOAL_DRIFT deliberately absent) |
| `_USER_AUTHORED_DRIFT_KINDS` / `_USER_OR_TRAJECTORY_DRIFT_KINDS` (D) | bypass sets for user/trajectory drifts |
| `_emit_progress_stalled_escalation` / `_emit_handler_exhausted_escalation` (D) | the escalation emitters (Appendix A) |
| `_spawn_drift_handler_background` / `_run_drift_handler_background` (D) | fire-and-forget drift cascade |
| `_verdict_ledger` / `_emit_verdict_utility_summary` (D) | #483 verdict-utility counters |

### Plan-mutation surface (`goldfive/plan_reviser.py`)

| Symbol | Role |
| --- | --- |
| `_apply_revision` (P) | injection point #1: gated `session.plan` swap; returns `(revised, was_installed)` |
| `_emit_plan_revised` (P) | lock-protected install + `PlanRevised(dry_run)` emit; stamps the freshness watermark |
| `_fold_runtime_terminal_statuses` (P) | fold out-of-band terminal statuses onto the revised plan |
| `_plans_structurally_identical` (P) | no-op-revision detector → handler exhaustion |
| `_get_plan_lock` / `_wait_plan_stable` (P) | per-session plan lock (NOT held across `planner.refine`) |

---

## Appendix C — Where drifts come from (severity + the judge bridge)

The ladder is severity-driven, so knowing how a `DriftEvent` acquires its `severity` matters when you debug a routing surprise.

- **`DriftSeverity`** has three levels: `INFO` (0), `WARNING` (1), `CRITICAL` (2) — ordered so `_severity_ge` can compare with `>=`. Detectors and judges assign severity; the ladder consumes it. There is no way for the ladder to *raise* a severity — a WARNING drift can only reach a CANCEL_REINVOKE rung if the ladder table maps WARNING there (none do today; WARNING tops out at ABSORB/NUDGE). CRITICAL is the only tier that reaches cancel (`_should_request_cancel_for_drift`).
- **Deterministic detectors** (`detect_drift` → tool-error / refusal / stop-reason classifiers) assign severity structurally.
- **Reasoning / goal-drift judges** return a verdict the pipeline maps to a `DriftEvent` with a judge-chosen severity.
- **Pluggable judges** (`DefaultSteerer.evaluate_judges`, goldfive#437): operator-supplied `Judge` instances registered via `goldfive.wrap(judges=[...])`. A drift-flavoured verdict is bridged to a `DriftEvent` by `_drift_from_judge_verdict` and forwarded to `handle_drift` — so custom judges ride the SAME ladder + gates as everything else. `evaluate_judges` emits a `JudgementEmitted` keyed on the judge's `name` and sets `drift._judge_emitted_judgement = True` so `_emit_drift_detected` does not double-emit (CM14). Built-in judges' drift verdicts ride the legacy detector path's paired emission instead.
- **Control messages** map to fixed severities in `_drift_from_control`: STEER→WARNING, CANCEL→CRITICAL, PAUSE→INFO. These bypass the freshness/in-flight/cancel-severity gates because they are user-authored (Invariant on operator intent).

---

## Appendix D — Debugging: "a drift fired on the wire but the agent kept going"

Walk this decision tree top to bottom. The FIRST match explains it. (Every one of these is a `grep`-able log line or wire event.)

1. **Is `observation_only=True`?** (The production default.) Then this is expected — you will see `PlanRevised(dry_run=True)` and `PolicyApplied(observation_only_gate, suppressed)` and NO plan swap / cancel / nudge. To see real intervention, run with `SteeringConfig(observation_only=False)` or `GOLDFIVE_STEER_OBSERVATION_ONLY=0`. This explains ~90% of "nothing happened" reports.
2. **`SteeringDecisionMade.outcome == "drift_dropped_stale"`?** Gate 2 (freshness watermark) dropped it — a later revision already addressed the same `(kind, target)`. Redundant by design.
3. **`SteeringDecisionMade.outcome == "drift_dropped_inflight"`?** Gate 3 — a concurrent refine for the same `(kind, target)` is in flight. Redundant by design.
4. **`DriftDetected.suppressed_by_user_steer == true`?** Step C — a fresh user steer is within the suppression window (`suppression_window_turns` logical turns). The user's directive dominates; the goldfive steer is suppressed.
5. **`emitted_late` incremented / log "stale judge verdict"?** The late-drift gate (§8) — the invocation that produced the reasoning already terminated (no active invocation OR cancel-pending). The drift is recorded for observability only.
6. **`PolicyApplied(refine_outcome_succeeded_skip)` or `(refine_failure_threshold)`?** H3 — the `(kind, task)` already succeeded this turn (no-op replay) or already failed `REFINE_FAILURE_THRESHOLD` times (a `REPEATED_FAILURE` was already emitted and the task marked FAILED).
7. **`LadderTransitionDecided.to_level == "observe"`?** The ladder mapped this `(kind, severity, occurrence)` to OBSERVE — expected for INFO drifts and for `REASONING_CLUSTER_TIGHTENING` at every rung.
8. **Level is NUDGE / ABSORB but the agent is at a task boundary?** NUDGE only takes effect when the overlay's `_drain_nudges` re-invokes (needs live PENDING/RUNNING work and `nudge_replays < _MAX_NUDGE_REPLAYS`); ABSORB's revised plan lands at the *next* task boundary, not mid-invocation, unless the kind is in `_ABSORB_NUDGE_KINDS`.
9. **`level == PAUSE_ESCALATE` but no pause?** Under `observation_only` the `GOLDFIVE_PAUSE_ESCALATE` dispatch is gated (§8) — the `HUMAN_INTERVENTION_REQUIRED` drift still fires but goldfive does not cancel. Also check the executor bound a control channel (`bind_control_channel`); an unbound channel makes the dispatch a best-effort no-op.
10. **Adapter/plugin gap?** `request_invocation_cancel` no-ops on an unbound adapter, a non-ADK adapter, a missing `_plugin`, or an empty resolved invocation-id list (Invariant 3). Check the log for "no active invocation for drift".

If none of these match, the intervention DID fire — check the executor side (04-executors-and-control.md) for why the restart/cancel did not reach the agent.

---

## Appendix E — The concurrency model

The steerer is shared across concurrent `runner.run(...)` calls (multi-runner processes), and the LLM-judge path is fire-and-forget. Several mechanisms keep that safe; you must preserve all of them.

### Per-async-task session isolation (`_active_session_var`)

`DefaultSteerer._active_session_var` is a **per-instance** `contextvars.ContextVar` (named `goldfive_active_session_{id(self)}`), NOT a plain attribute and NOT module-global. The steerer sets it just before calling `planner.refine` / `refine_steer` / `synthesize_goal_from_steer` and resets it in a `finally`. It plumbs the active session into the planner's drift-emitter and span-context callbacks.

Why a ContextVar: two concurrent `runner.run(...)` calls share one Steerer (and therefore one Planner). Without isolation, session A can refine while session B is mid-refine, B's value overwrites A's, and A's planner-side span/drift callbacks resolve to B's `run_id`. Per-*instance* (not module-level) so parallel test cases instantiating their own Steerer never collide. **If you add a new planner-callback plumb, thread it through this ContextVar with a `finally` reset — never a plain attribute.**

### The plan lock (`_plan_locks`)

Per-`session.id` `asyncio.Lock`, held ONLY across the consistency-critical region of `_emit_plan_revised` (revision-index bump + supersedes integration + correction GC/queue + repin + `PlanRevised` emit). It is deliberately **NOT** held across `planner.refine` — that would serialise concurrent refines on the same session and defeat the fire-and-forget judge path (#254). Readers that must observe a consistent plan call `_wait_plan_stable` (acquire + immediately release). This is what makes `_wait_plan_stable` callers see either pre- or post-revision state, never a partial apply (goldfive#403 moved every session-mutation site into this lock).

### Race 1 — concurrent refine (freshness watermark + in-flight set)

Two background judges observe the same `(kind, current_task_id)` at the same `observed_revision_index`:

```
Judge A: Gate 2 reads last_addressed==0 -> pass; Gate 3 stamps inflight_key; begins refine
Judge B: Gate 2 reads last_addressed==0 -> pass (A hasn't installed yet);
         Gate 3 sees inflight_key present -> DROP (drift_dropped_inflight)
Judge A: refine completes; _emit_plan_revised stamps last_addressed=N under the lock
Judge C (later): Gate 2 reads last_addressed==N > observed -> DROP (drift_dropped_stale)
```

The watermark (Gate 2) alone cannot close the A/B window because the watermark is written at the END of A's refine (under the lock, after the multi-second LLM round-trip). The in-flight set (Gate 3) closes it synchronously at dispatch entry. The two gates are complementary — do not remove either.

### Race 2 — double cancel (`_cancelled_drift_ids`)

A CRITICAL promote-eligible drift fires two cancels: the flag-only one at Step D (writes the cancel-state flag the executor consumes, driving the channel-message restart), then `_cancel_inflight_for_revision` post-refine. Between them the executor may have restarted the invocation, so the second hard cancel could land on a fresh invocation. `_cancelled_drift_ids` (stamped at Step D, keyed on `drift.id` — a legitimate per-dispatch use, added-and-discarded within one `handle_drift`) lets the post-refine cancel short-circuit. The set is cleared in `handle_drift`'s `finally` so it stays bounded by active dispatch depth.

### Background task sets and draining

- `_background_judges: set[asyncio.Task]` — fire-and-forget reasoning-judge LLM calls (#251). Tasks named `goldfive-...:{session.id}` so `drain_session_background_tasks` can filter by run boundary. Auto-discard via `add_done_callback`.
- `_background_drifts: set[asyncio.Task]` — fire-and-forget drift cascades from `mark_task_failed` / `mark_task_blocked` / reporting tools (iter-11A). Awaiting these inline from a reporting-tool call site blocked the tool for minutes on slow local LLMs; the background set fixes that.
- `_judge_semaphore` — per-steerer `asyncio.Semaphore` (default 3, `ReasoningDriftConfig.max_concurrent_judges`) bounding concurrently-RUNNING judge LLM calls (#483).
- `_queued_judge_windows` — QUEUED windows keyed `(session_id, agent_name, task_id)`; newer observations for the same key coalesce onto the entry instead of scheduling another task (#483).

`shutdown(timeout=5.0)` and `drain_session_background_tasks` drain both sets symmetrically and flush the verdict-utility ledger. `_run_drift_handler_background` / `_run_goal_drift_judge_background` swallow every exception (a flaky cascade must not crash the run) but re-raise `CancelledError` cleanly so teardown can cancel still-running cascades without a stray WARNING.

### Cross-session registry isolation (state store)

`StateStore`'s `_ACTIVE_INVOCATION_TASKS` and `_CANCEL_REQUESTED_INVOCATIONS` are module-level dicts keyed by `session.id` (aliased to `run_id`). The late-drift gate (§8) and `request_invocation_cancel` read/write them. In tests, the `_isolate_orchestration_store_registries` autouse fixture clears them between tests — many tests share `run_id="r1"`, so a leaked cancel-pending entry from one test silently flips the late-drift gate in the next (a CI-only flake). If you write a new test that calls `request_invocation_cancel` or `register_invocation_task`, rely on that fixture rather than hand-rolling cleanup.

---

## Appendix F — The USER_STEER lifecycle end-to-end

An operator STEER is the one intervention that bypasses almost every gate (it is user-authored). Tracing it clarifies why those bypasses exist and how user intent stays dominant over goldfive's own steers.

1. **Ingress.** A `STEER` `ControlMessage` (payload `{note, author, annotation_id}`) reaches `DriftObserver.observe`.
2. **Dedupe.** `_is_duplicate_steer` computes `_steer_dedupe_id` (the `annotation_id`, or the message id) and checks `state.processed_steer_ids`. A UI double-fire or delivery retry of the same STEER lands twice but is dropped the second time. (Content drifts are NOT deduped — they are heuristic signals.)
3. **Coercion.** `_drift_from_control` mints a `DriftEvent(kind=USER_STEER, severity=WARNING, authored_by="user", raw=<ControlMessage>)`. The operator `author` is prefixed onto `detail` as `"by {author}: {note}"` for audit trails; the raw body survives on `drift.raw`.
4. **Entry gates.** Gate 0 (not PLAN_DIVERGENCE), Gate 1 (`authored_by` already `"user"`). Gate 2 (freshness) and Gate 3 (in-flight) both **bypass** — user drifts are exempt (`(drift.authored_by or "").lower() != "user"` guards the stamped-drift branch). An operator directive must be honoured regardless of the plan cursor.
5. **Step B side effects** (`_apply_user_steer_state`): `_unpack_steer_context` recovers `(body, author, dedupe_id)` from `drift.raw`. `_ostate.set_active_steer(state, body=, at_turn=session._reasoning_turn, author=, source="user")` stamps the active-steer slot. `record_processed_steer_id(state, dedupe_id)` records the dedupe id AFTER the active-steer stamp so a mid-dispatch reader always sees the latest steer.
6. **Promotion policy** (`_should_promote_to_steer`): returns `False` for `USER_STEER` (it is in `_USER_AUTHORED_DRIFT_KINDS`) — user steers keep their pre-unification handling. `_emit_drift_detected` fires.
7. **Cancel** (`_should_request_cancel_for_drift`): `USER_STEER` bypasses the severity gate → `True` even at WARNING. `request_invocation_cancel` fires (gated by `observation_only`, but note: the plan-mutation gate has a `user-authored` carve-out so the steer's refine LANDS even under `observation_only` — see §8 Gate 1).
8. **Refine.** The ladder maps `USER_STEER` through the generic path; the refine reads `session.goals` (which `Planner.handle_turn` reshaped from the steer). `USER_STEER` is in `_USER_OR_TRAJECTORY_DRIFT_KINDS` so it bypasses the H3 refine-outcome gate and the H4 progress-stall gate — operator intent is never rate-limited.
9. **Dominance window.** The `active_steer` slot (`source="user"`, `at_turn=N`) now suppresses any goldfive-authored promotion for `suppression_window_turns` logical turns (§6): `_should_promote_to_steer` stamps `suppressed_by_user_steer=True` on competing goldfive steers and returns `False`. This is how a live operator override stays dominant across a few agent turns.

The symmetric goldfive-authored path (`_promote_drift_to_steer`) reuses steps 5-8 with `source="goldfive"`, `author="goldfive"`, and records `drift.id` in `processed_steer_ids` (so a redelivery can't re-promote) — but it is subject to all the gates a user steer bypasses.

---

## Appendix G — What fires on the wire, mode by mode

For a single corrective goldfive-authored drift that the ladder routes to CANCEL_REINVOKE (or promotion), here is the exact wire/state delta in each mode. Use it as the assertion set for a both-modes test.

| Effect | `observation_only=True` (default) | `observation_only=False` |
| --- | --- | --- |
| `DriftDetected` | ✅ fires | ✅ fires |
| `SteeringDecisionMade` | ✅ (`drift_emitted`) | ✅ (`drift_emitted`) |
| `JudgementEmitted` | ✅ | ✅ |
| `LadderTransitionDecided` | ✅ | ✅ |
| `planner.refine` / `refine_steer` runs | ✅ (still runs) | ✅ |
| `session.plan` swapped | ❌ (gate; `dry_run`) | ✅ |
| `last_addressed_revision_by_drift_key` stamped | ❌ | ✅ |
| `PlanRevised` | ✅ (`dry_run=True`) | ✅ (`dry_run=False`) |
| `PolicyApplied(observation_only_gate, suppressed)` | ✅ (one per gated surface) | ❌ |
| plugin cancel flag / `task.cancel()` | ❌ | ✅ |
| `GOLDFIVE_STEER` ControlMessage enqueued | ❌ | ✅ |
| `session.pending_nudges` appended (if ABSORB+nudge kind) | ❌ | ✅ |
| live agent behaviour | unchanged | preempted + restarted on revised plan |

The invariant this encodes: **the top block (detection + observability) is identical in both modes; only the bottom block (writes to the live agent) differs.** A both-modes test asserts the top block is equal and the bottom block flips. If a new surface makes any top-block row differ between modes, you have coupled detection to injection — a bug. If a new surface makes a bottom-block row fire under `observation_only=True`, you have leaked an intervention (CM1).

---

## Appendix H — The observation surface: how events reach the ladder

`handle_drift` is the sink; several entry points on `DriftObserver` are the sources. You will rarely edit these for a steering change, but you need to know where a drift *originated* when debugging. All of them ultimately produce a `DriftEvent` and route it (directly or via a background task) to `handle_drift`.

| Entry point | Producer | Detail |
| --- | --- | --- |
| `observe(event, session)` | ADK model/tool events | dedupe → control-drift → `detect_drift` (deterministic) → `handle_drift` |
| `observe_reasoning(text, session, agent_name, ...)` | thinking-token stream | runs the reasoning judge / embedding detectors (mode-selected); background-judge path; late-drift gate; on-task condition resolution |
| `note_llm_call(session)` | every LLM invocation | increments the reflective-check counter; fires `maybe_run_reflective_check` at the configured interval |
| `note_agent_turn(session)` | agent-invocation turns | drives the turn-based GOAL_DRIFT schedule (`maybe_run_goal_drift_check`) |
| `note_agent_activity(...)` / `note_tool_observation(...)` | agent activity + tool calls | append to the bounded `recent_events` buffer the goal-drift judge reads |
| `maybe_run_goal_drift_check(session)` | trajectory-level | LLM judge over `session.goals` vs recent activity → GOAL_DRIFT drift |
| `_maybe_run_goal_drift_on_task_boundary(session)` | task transitions | task-boundary GOAL_DRIFT trigger, throttled by `_GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` (10s) |
| `maybe_run_reflective_check(session)` | periodic self-check | asks the agent "are you making progress?" → SELF_REPORTED_STUCK / UNCERTAIN_PROGRESS (opt-in; inert without `reflective_call_llm`) |
| `report_new_work_discovered(...)` | reporting tool | mints a `NEW_WORK_DISCOVERED` drift (a discovery carve-out in `_apply_revision`, §8) |
| `report_plan_divergence(...)` | reporting tool | disabled at Gate 0 (#252); observability only |
| the stall watchdog (#487) | wall-clock timer | emits `TASK_TIMEOUT` when `Session.last_observed_event_at` goes silent past `stall_timeout_s` (flag-gated, default OFF) |

The reflective and goal-drift checks are **opt-in**: they are inert unless the operator wires a `reflective_call_llm` / `goal_drift_call_llm` callable (typically `goldfive.wrap` threads the planner LLM in when the feature flag is on). The detector/judge internals live in 07-deterministic-drift-detection.md and 08-llm-judges.md; this chapter only owns what happens once a `DriftEvent` exists.

Two design rules bind every one of these producers (from the CANON invariants): (1) they capture **observed facts** and never predict agent behaviour (Invariant 4) — e.g. the goal-drift judge reads `recent_events`, it does not forecast the next tool call; (2) none of them require the agent to call a goldfive tool — `note_*` and `observe_*` are driven by the plugin observing the tree, so termination/observability work even if the agent never cooperates (Invariant 1). The `report_*` hooks are the exception (they ARE agent-driven), which is exactly why they are additive discovery signals, not load-bearing enforcement.

---

## Appendix I — Glossary

Terms used throughout this chapter, defined against the code so a weaker model does not conflate them.

- **Drift** — a `DriftEvent` (typed `kind` + `severity` + `current_task_id` + `current_agent_id` + `id` + `observed_revision_index` + `authored_by`). A single observed signal. Produced by a detector or judge; consumed by `handle_drift`. Has a fresh UUID4 `id` per emit.
- **Condition** — a *logical group* of drifts keyed on `sha1(kind, task_id, agent_id, turn_id)`, tracked in `state.KEY_ACTIVE_DRIFTS`. Multiple `DriftEvent`s for the same tuple in a turn collapse onto ONE `condition_id`. The condition carries lifecycle (OPENED → ESCALATING → RESOLVED / HUMAN_INTERVENTION_REQUIRED). Do not confuse the per-event `drift.id` with the `condition_id` (§10).
- **Intervention level** — one of six `InterventionLevel` rungs the ladder picks (`OBSERVE`..`TERMINATE`). Distinct from `DriftSeverity` (INFO/WARNING/CRITICAL), which is an INPUT to the ladder.
- **Occurrence count** — the per-`(kind, task)` failure count from `session.refine_outcomes`, mapped by `_occurrence_count_for_ladder`. `>= REFINE_FAILURE_THRESHOLD` (2) means "repeat" and selects the second element of a ladder CRITICAL pair.
- **Refine vs refine_steer** — `planner.refine(...)` is the ladder path's revision call (ABSORB/CANCEL_REINVOKE). `planner.refine_steer(...)` is the promotion path's call, framing the pivot as a correction (falls back to `refine` when the planner lacks it). Both produce a new `Plan`.
- **Promotion** — routing a goldfive-authored drift through the USER_STEER-style cancel-and-restart machinery (`_promote_drift_to_steer`) instead of the passive ladder. Gated by `_should_promote_to_steer` (eligible kind + severity threshold + no fresh user steer).
- **Active steer** — the `goldfive.active_steer.*` slot on `session.state` (`body`, `at_turn`, `author`, `source`). Written by USER_STEER (`source="user"`) or a promotion (`source="goldfive"`). A fresh `source="user"` steer suppresses goldfive promotions for `suppression_window_turns` logical turns.
- **Freshness watermark** — `session.last_addressed_revision_by_drift_key[(kind, task)]`, the revision index at which a `(kind, target)` was last successfully refined. Gate 2 drops verdicts observed at an earlier revision.
- **In-flight-refine key** — `(session_id, kind, task)` in `DriftObserver._inflight_refine_keys`, stamped at dispatch entry and cleared in `finally`. Gate 3 drops a concurrent same-key refine.
- **Supersede** — a goldfive-internal cancel that switches the in-flight agent onto a just-installed revised plan. `session._supersede_pending = True` tells the executor's cancelled branch to restart rather than abort. Set by `_cancel_inflight_for_revision`.
- **dry_run** — the `PlanRevised.dry_run` marker. `True` under `observation_only` (the revision was computed but NOT installed) — a would-have-applied preview. `dry_run` is observability, not silence.
- **Observation-only / strict-passive** — `SteeringConfig.observation_only=True` (production default). Detection + refine + `PlanRevised` still happen; the four gated injection points (§8) are skipped. Read ONLY through `is_active_steering()` / `steering_is_active()`.
- **Late drift** — a background-judge verdict that landed after its originating invocation terminated (no active invocation OR cancel-pending). Recorded for observability, not dispatched (§8).
- **Handler exhaustion** — the planner cannot produce a meaningful change for a drift (`RefineExhausted`, `revised is None`, validator rejection, or a structurally-identical no-op). Escalates to `HUMAN_INTERVENTION_REQUIRED` rather than looping the planner.
- **Cooperative cancel** — flagging an invocation for cancel via a `CancellationRequest` on the plugin state (the plugin short-circuits its next callback); optionally `task.cancel()` when `cancel_inflight_task=True`. "Cooperative" because it does not force-kill the run — the executor decides whether to restart.
- **Terminal drift** — a `DriftKind` in `_TERMINAL_DRIFT_KINDS` (`HUMAN_INTERVENTION_REQUIRED`, `REPEATED_FAILURE`). Emitting one triggers open-boundary cleanup so the Gantt has no permanently-open spans.

---

## Appendix J — Refine observability: attempt_id and the failure_kind vocabulary

Every refine (ladder or promotion) is bracketed by paired observability events so a sink can match an attempt to its outcome. `plans._new_attempt_id()` mints an id; `plans._emit_refine_attempted(session, drift, attempt_id=)` fires before the refine; exactly one of `_emit_refine_failed(...)` or `_emit_plan_revised(...)` fires after. Match on `attempt_id`.

The `failure_kind` on `_emit_refine_failed` is a closed vocabulary — a weak model editing an error arm must reuse the right one, not invent a new string:

| `failure_kind` | Emitted when | Escalation that follows |
| --- | --- | --- |
| `refine_exhausted` | planner raised `RefineExhausted` | `_emit_handler_exhausted_escalation` → HUMAN_INTERVENTION |
| `llm_error` | `planner.refine` raised a generic `Exception` (bad JSON, LLM error) | `_escalate_refine_failure_as_critical_drift` + `_record_refine_outcome(succeeded=False)` |
| `cancelled` | `BaseException`/`CancelledError` during refine | re-raise (cancellation propagates); paired event closes the `refine_attempted` |
| `parse_error` | `refine` returned `None` (internal retry budget spent) | `_record_refine_outcome(False)` + handler-exhausted escalation |
| `validator_rejected` | `revised.validate(for_revision=True)` raised `ValueError` | INFO `SCHEMA_VIOLATION` (observability) + `_record_refine_outcome(False)` + handler-exhausted escalation |
| `no_op_revision` | `_plans_structurally_identical(prior, revised)` | handler-exhausted escalation |

The `cancelled` arm is the one that MUST re-raise — it exists to emit the paired event before cancellation continues, so a mid-flight cancelled refine does not leave a sink with an unmatched `refine_attempted`. Both refine pipelines (§7) have identical arms; keep them in sync.

---

## Appendix K — The cancel_reason vocabulary (`transition`)

When the steerer/executor transitions a task to CANCELLED or FAILED, the `cancel_reason` string on the emitted `TaskCancelled` / `TaskFailed` envelope follows a conventional format harmonograf's Trajectory view parses (from `DefaultSteerer.transition`, `goldfive/steerer.py`). Sinks that do not recognise the format surface the raw string, so the vocabulary can evolve without a proto change — but reuse an existing shape where one fits:

| Format | Meaning |
| --- | --- |
| `upstream_failed:<task_id>` | cascade from a failed/cancelled ancestor |
| `superseded_by_revision:<task_id>` | a refine replaced this task with a new one |
| `run_aborted:<reason>` | fail_fast / validation / budget abort |
| `user_cancel:<annotation_id>` | user-initiated CANCEL control |
| `adk_cancellation:<invocation_id>` | ADK mid-invocation cancel |
| `steerer_policy:<drift_kind>` | steerer-imposed cancel via the intervention ladder |

`cancel_reason` takes precedence over `detail` for the reason field on terminal transitions; passing it for a non-terminal transition (`RUNNING`, `COMPLETED`, `BLOCKED`) is a no-op. The symbolic *adapter* cancel reasons (`user_steer`, `goldfive_<kind>`, `drift`) written by `_tag_adapter_cancel_reason` are a separate, plugin-facing vocabulary — do not conflate the two.

---

## Appendix L — Test-file map for this subsystem

Which test file owns which behaviour, so you extend the right one when you change a surface. All under `tests/`.

| File | Owns |
| --- | --- |
| `test_intervention_ladder.py` | the `_LADDER` table + `_ladder_level_for` `(kind, severity, occurrence) → level` mapping |
| `test_goldfive_drift_routing.py` | `observe` → `handle_drift` routing, control-drift coercion |
| `test_steerer.py` | core `DefaultSteerer` surface, wiring, `transition` |
| `test_drift_outcomes.py` | `_record_refine_outcome`, the refine-outcome gate, `REPEATED_FAILURE` |
| `test_observation_only.py` | the master `observation_only` contract |
| `test_observation_only_strict_passive.py` | the strict-passive predicate (#488): missing/None/raising → passive |
| `test_observation_only_nudge_gate.py` | the nudge enqueue gate (#475) |
| `test_observation_only_acks.py` | F1 directive-ack `plan_state` gating (#478) |
| `test_observation_only_abort_carveout.py` | the abort/pause carve-outs under passive |
| `test_observation_only_pause_escalate_carveout.py` | `GOLDFIVE_PAUSE_ESCALATE` gate (#264) |
| `test_observation_only_emit_supersedes_carveout.py` | supersedes emission under dry-run |
| `test_promote_drift_to_steer.py` | the promotion path (twin pipeline) |
| `test_steer_unification.py` / `test_tier2_steer_pipeline.py` | goldfive-steer unification + Tier-2 pipeline |
| `test_pause_escalate.py` / `test_pause_deadline.py` | Level 4/5 dispatch + the #482 deadline |
| `test_drift_lifecycle.py` | condition OPENED/ESCALATING stamping |
| `test_drift_resolution_wiring.py` | #486 RESOLVED resolution (terminal task + on-task verdict) |
| `test_terminal_drift_closes_spans.py` | terminal-drift boundary cleanup |
| `test_steering_decision_made.py` | `SteeringDecisionMade` + #480 drop labels |
| `test_steerer_usersteer.py` / `test_user_steer_invariant.py` | USER_STEER lifecycle + dedupe + operator dominance |
| `test_llm_call_timeout_watcher.py` | the LLM-timeout cancel gate (#476) |
| `test_steerer_concurrent_sessions.py` | the `_active_session_var` ContextVar isolation (Appendix E) |
| `test_goldfive_steer_request_cancel.py` | `request_invocation_cancel` flag write + plugin fallback |
| `test_steerer_fold_runtime_terminal.py` | `_fold_runtime_terminal_statuses` |
| `test_refine_steer_retry_feedback.py` | refine-steer retry / feedback loop |
| `test_overlay_steer.py` | the executor overlay's steer consumption + nudge drain |
| `test_live_steering_e2e.py` | end-to-end steering, both modes — the template for a both-modes assertion |

When you add a new gated surface (CM1), the fastest correct move is to copy the both-modes structure from `test_observation_only_nudge_gate.py` and adapt the assertions to your surface, then run the whole `tests/test_observation_only_*.py` set plus `test_intervention_ladder.py`.

---

## Appendix M — Constants quick-reference

Every tunable/threshold this chapter touches, with its default and home. Change a threshold in exactly one place — these are the canonical definitions; other components read them by reference (`self._steerer.X`).

| Constant | Value | Location | Meaning |
| --- | --- | --- | --- |
| `REFINE_FAILURE_THRESHOLD` | `2` | `DefaultSteerer` (S) | ladder "repeat" boundary + refine-failure gate trip |
| `PROGRESS_STALL_THRESHOLD_SECONDS` | `600.0` | `DefaultSteerer` (S) | H4 progress-stall escalation; `0` disables |
| `DEFAULT_TERMINATE_PAUSE_DEADLINE_S` | `600.0` | module (D) | Level-5 default pause deadline when unset |
| `_MAX_NUDGE_REPLAYS` | `3` | `SequentialExecutor` (executors) | nudge re-invocation cap (#163) |
| `_GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` | `10.0` | `DriftObserver` (D) | min spacing between task-boundary GOAL_DRIFT calls |
| `_LEDGER_ELAPSED_SAMPLES_CAP` | `1024` | `DriftObserver` (D) | verdict-latency sample cap |
| `REFLECTIVE_MAX_OUTPUT_TOKENS` | `16384` | `DriftObserver` (D) | reflective-check per-call token budget |
| `JUDGE_EVALUATE_TIMEOUT_S` | `30.0` | `DefaultSteerer` (S) | per-pluggable-judge evaluate budget; overrun → cancelled, no signal |
| `observation_only` | `True` | `SteeringConfig` (config) | master kill-switch |
| `threshold` | `"warning"` | `SteeringConfig` (config) | promotion severity threshold |
| `suppression_window_turns` | `3` | `SteeringConfig` (config) | user-steer dominance window (logical turns) |
| `pause_escalate_deadline_s` | `None` | `SteeringConfig` (config) | Level-4 deadline; `None` = forever |
| `stall_watchdog_enabled` | `False` | `SteeringConfig` (config) | flag-gates the `TASK_TIMEOUT` producer (#487) |
| `stall_timeout_s` | `600.0` | `SteeringConfig` (config) | stall-watchdog silence threshold |
| `max_concurrent_judges` | `3` | `ReasoningDriftConfig` (config) | `_judge_semaphore` size (#483) |
| `name_axis_max_severity` | `"info"` | `ToolLoopConfig` (config) | uncorroborated tool-loop name-axis cap (#484) |

`REFINE_FAILURE_THRESHOLD` and `PROGRESS_STALL_THRESHOLD_SECONDS` are class attributes so subclasses/tests can tune them without instance surgery; do not hard-code their values at call sites — read `self._steerer.REFINE_FAILURE_THRESHOLD`.

Env-var overrides (read by `SteeringConfig.from_env`, all `GOLDFIVE_STEER_*`; see 14-config-reference.md for the full list and coercion rules):

| Env var | Field |
| --- | --- |
| `GOLDFIVE_STEER_OBSERVATION_ONLY` | `observation_only` (`0`/`false`/`no` → active) |
| `GOLDFIVE_STEER_THRESHOLD` | `threshold` (`off`/`warning`/`critical`) |
| `GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS` | `suppression_window_turns` |
| `GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S` | `pause_escalate_deadline_s` |
| `GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED` | `stall_watchdog_enabled` |
| `GOLDFIVE_STEER_STALL_TIMEOUT_S` | `stall_timeout_s` |
| `GOLDFIVE_DRIFT_MAX_CONCURRENT_JUDGES` | `ReasoningDriftConfig.max_concurrent_judges` |

An operator flipping to active steering in production does `GOLDFIVE_STEER_OBSERVATION_ONLY=0` or `RuntimeConfig(steering=SteeringConfig(observation_only=False))` at `goldfive.wrap` time — never by editing the `_observation_only` attribute (CM2).
