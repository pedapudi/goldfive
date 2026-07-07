# 03. Runner and Conversation State

## Read this chapter when...

- You are touching `goldfive/runner.py` — the single public entrypoint (`Runner`) — or need to understand the exact order in which a run is set up before the executor is handed control.
- You are adding a new early-exit / abort path to a run and need to know why you must funnel it through `Runner._abort_turn` (post-#489) instead of hand-rolling emit + `absorb_turn`.
- You are working on cross-turn continuity: goals accumulating, prior plans carrying forward, wire-sequence cursors staying collision-free, or "why did session B inherit session A's plan?" bugs.
- You need to know what `Runner.run_streamed` actually streams, and what `Runner.resume` does NOT do (it is replay-only; it does not continue execution — TODO #15).
- You are debugging lifecycle events: which of `RunStarted` / `GoalDerived` / `RunCompleted` / `RunAborted` / `ConversationStarted` / `ConversationEnded` fires, and which component owns each emission.
- You saw `conv.py` in a stack trace and assumed it was conversation state. It is **not** — see the "conv.py is not conversation state" warning below.

## Files covered

| File | Role |
| --- | --- |
| `goldfive/runner.py` | The `Runner` class: construction knobs, `run` → `_run_locked` phase pipeline, `_abort_turn`, `run_streamed`, `resume`, `close`, `new_conversation`, per-key lock discipline. |
| `goldfive/conversation.py` | `Conversation` + `TurnRecord`: cross-turn state, the wire-sequence cursor, `next_turn_session`, `stash_plan` / `prior_plan_for`, `absorb_turn`, `prior_turn_context`. |
| `goldfive/conv.py` | **NOT conversation state.** Proto round-trip converters (`to_pb_*` / `from_pb_*`). Covered briefly only to disambiguate the near-identical filename. |

Sibling chapters you will cross into constantly:

- **04-executors-and-control.md** — where the plan is walked, `RunCompleted` / terminal `RunAborted` are emitted, and where the `_run_overlay` stage methods (the other half of #489) live.
- **09-steering-ladder-and-gates.md** — `DefaultSteerer`, `bind`, `install_initial_plan` / `install_revision_for_drift`, and the `observation_only` kill-switch predicate.
- **10-planning-and-revision.md** — `Planner.handle_turn`, `Planner.generate`, and `PlanReviser` (which emits `PlanRevised`, not the Runner).
- **11-state-ownership.md** — `Session`, `state_store` (`_ostate`), `channel_processor_active`, the single-writer plan discipline.
- **12-events-sinks-telemetry.md** — the `goldfive.events` factories and `emit`.
- **13-reporting-tools-and-approval.md** — `select_reporting_tools`, `report_awaiting_approval`.

## Invariants that bind you here

These are the CANON invariants that actually get violated by edits in this subsystem. Keep them in view:

1. **No prompt-cooperation contracts.** The Runner's F6 conversational-turn directive (the "answer briefly from history, don't re-delegate" wrapping) rides in the *message body*, never the system prompt, and is gated OFF under `observation_only=True`. Termination, control, and observability must all work even if the agent never reads that directive. See `_run_locked` step 7 and `_prompt_shaper.wrap_conversational_input`.
2. **`observation_only=True` is the production default and is strictly passive.** The Runner never reads the kill-switch directly; the single sanctioned read lives in `DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)` (`goldfive/steerer.py`). The only place the Runner is *affected* by it is the F6 wrap, and that gate lives inside `PromptShaper`, which delegates to `steering_is_active`. Do not add a second `observation_only` read in `runner.py`.
3. **Lifecycle gates need stable identity keys.** The per-session `Conversation` map and the per-key `asyncio.Lock` are keyed on the *outer-session id* (`ctx.session.id`), never on an LLM-minted plan id or goal id. Do not re-key any of this bookkeeping on a churning id.
4. **Adaptive over predictive.** The Runner records observed facts (goals derived, plan installed, turn absorbed) and never predicts what the agent will do. `resume` reflects what the log *recorded*; it does not simulate forward.
5. **Any ADK tree shape must work.** The Runner reads `agent.available_agents_tree` when present, else `agent.available_agents`. Never assume a flat single-agent shape.

---

## 1. What a Runner is, and the six components it composes

`Runner` is goldfive's single public entrypoint. It owns nothing about *how* an agent framework works — no ADK or Claude-SDK imports live in `runner.py`. It composes six pluggable collaborators and drives one turn end-to-end:

```python
# goldfive/runner.py — module docstring
* GoalDeriver — turns user_input into list[Goal].
* Planner     — turns goals into a Plan.
* Executor    — walks the plan, dispatches to the adapter.
* AgentAdapter— talks to the underlying agent framework.
* Steerer     — runs the state machine, detects drift.
* EventSink   — persists / observes the event stream.
```

The mental model: **`Runner` is the setup-and-teardown coordinator for a single turn.** It builds a `Session`, derives goals, decides on a plan, wires the steerer into the adapter, then hands everything to the `Executor`, which does the actual agent-driving work. When the executor returns, the Runner folds the turn's result back into the cross-turn `Conversation`.

The component-wiring the Runner performs each turn (all in `_run_locked` phase 6), so you can see who points at whom:

```
                 Runner._run_locked
                        │
    ┌───────────────────┼─────────────────────┬──────────────────┐
    │ 6  steerer.bind   │ 6b agent            │ 6c steerer       │ 6d steerer
    │  (sinks, planner) │  .bind_steerer(st)  │  .bind_adapter   │  .bind_control_channel
    ▼                   ▼                     ▼                  ▼
 DefaultSteerer  ◄── AgentAdapter  ──►  (steerer holds     ControlChannel
 (has sinks +        (callbacks now      adapter ref for    (goldfive-authored
  planner)           see steerer)        cancel-reason      STEER/PAUSE ride here)
                                         tagging)
```

6 and 6b are load-bearing (abort on failure); 6c and 6d are best-effort (log-and-continue). See §4 Phase 6.

### 1.1 Event lifecycle ownership (memorize this table)

A weak model's most common runner bug is emitting a lifecycle event from the wrong component. The ownership split is deliberate and load-bearing:

| Event | Owner | Where it fires |
| --- | --- | --- |
| `RunStarted` | Runner | `_emit_run_started`, called from `_run_locked` step 3 (before goals). |
| `GoalDerived` | Runner | `_emit_goal_derived`, `_run_locked` step 4. |
| `ConversationStarted` | Runner | `_emit_conversation_started`, first turn per key (step 2b). |
| `ConversationEnded` | Runner | `_emit_conversation_ended`, from `new_conversation` and `close`. |
| `RunAborted` (pre-executor) | Runner | `_emit_run_aborted`, via `_abort_turn` only. |
| `PlanRevised` (incl. the initial plan install) | `PlanReviser` (`steerer.plans`) | `_emit_plan_revised`; see §5b. |
| `RunCompleted` / terminal `RunAborted` | Executor | `SequentialExecutor` / `ParallelDAGExecutor` at their own state-machine end. |
| `DriftDetected`, `Task*` transitions | Steerer | see 07/08/09. |

> **The `PlanSubmitted` gotcha.** The `runner.py` module docstring lists `PlanSubmitted` as a Runner-owned event. **The code disagrees, and the code wins.** `plan_submitted_event` (in `goldfive/events.py`) is *not emitted by any live dispatch path* on main — the factory symbol appears only in its own definition in `events.py`. The `plan_submitted` *payload* is still handled on the reconstruction side (`sinks/persistence.py`, referenced in `state_store.py`), but nothing on main calls the factory to emit it. The **initial** plan install emits `PlanRevised` with `revision_index = 1` via `PlanReviser.install_initial_plan` (`goldfive/plan_reviser.py`), *not* `PlanSubmitted`. Treat `PlanSubmitted` as a legacy/reconstruction-only envelope. If you are looking for "where the first plan is announced," it is `PlanRevised` rev 1. (Caveat logged.)

### 1.2 conv.py is NOT conversation state

`goldfive/conv.py` is a proto round-trip converter module: `to_pb_task` / `from_pb_task`, `to_pb_plan`, `to_pb_control_event`, etc. It has **nothing** to do with `Conversation` cross-turn state. The cross-turn state lives entirely in `goldfive/conversation.py`. The two filenames differ by four characters; a weak model searching for "conversation logic" will open `conv.py` and be misled.

- Want to change how goals accumulate across turns, how prior plans carry forward, or the wire-sequence cursor? → `conversation.py`.
- Want to change how a `Task` / `Plan` / `DriftEvent` serializes to protobuf? → `conv.py`.

If you ever need to import the cross-turn class, it is `from goldfive.conversation import Conversation`. `runner.py` does exactly that at the top of the file. One live coupling exists between the two: `goldfive/events.py`'s `goal_derived_event` / `plan_submitted_event` factories import `to_pb_goal` / `to_pb_plan` from `conv.py` — but that is the *event* layer using the converter, not the Conversation class.

---

## 2. Runner construction: every knob and what it does

The constructor is keyword-only (`def __init__(self, *, ...)`) plus a `**legacy_kwargs` trap. Here is the signature and each knob's contract, grounded in the docstring and body.

```python
# goldfive/runner.py — Runner.__init__ (signature)
def __init__(
    self, *,
    agent: AgentAdapter,
    planner: Planner,
    executor: Executor,
    goal_deriver: GoalDeriver | None = None,
    steerer: Steerer | None = None,
    sinks: list[EventSink] | None = None,
    control: ControlChannel | None = None,
    max_task_invocations: int | None = None,
    conversation: Conversation | None = None,
    goal_drift_enabled: bool = True,
    planner_gate: Any = "auto",
    drift_self_reporting: bool | list[str] = False,
    fail_fast_on_revision_rejection: bool | None = None,
    **legacy_kwargs: Any,
) -> None:
```

### 2.1 Required components

| Knob | Default | Notes |
| --- | --- | --- |
| `agent` | (required) | An `AgentAdapter`. The Runner reads `agent.available_agents_tree` (preferred, #151 tree shape) or `agent.available_agents` (flat fallback). It probes `agent.bind_steerer`, `agent.register_reporting_tools`, `agent.subscribe_adk_events` by `getattr` — third-party adapters that predate these hooks still work. |
| `planner` | (required) | The Runner calls `planner.handle_turn` (Phase 4, #271) when present, else falls back to `planner.generate`. See §4. |
| `executor` | (required) | `SequentialExecutor` or `ParallelDAGExecutor`. The Runner hands it the plan; the executor emits the terminal `RunCompleted`/`RunAborted`. |

### 2.2 Optional collaborators

- **`goal_deriver`** — defaults to `PassthroughGoalDeriver("run")`. Bypassed entirely when `user_input` is already a `list[Goal]`.
- **`steerer`** — defaults to `DefaultSteerer()`.
- **`sinks`** — defaults to `[]`. Materialized into `self.sinks = list(sinks) if sinks else []`. Every turn passes `list(self.sinks)` (a snapshot) to the executor, so `add_sink` mid-run affects only *subsequent* runs.
- **`control`** — an optional `ControlChannel` for live pause/resume/cancel/steer/rewind. Forwarded into the executor and into the steerer (step 6d) via `bind_control_channel`.
- **`conversation`** — a seed `Conversation` for the default (`""`) key. If omitted, `Conversation.new()` is used.

### 2.3 Behavior switches

**`max_task_invocations`** — a safety cap on adapter invocations per run. Stamped onto the planner context (`planner_context["max_task_invocations"]`) so executors that honor it can enforce it. `None` = unbounded. Per-task/per-tool caps are the primary runaway guard; this is a coarse backstop.

**`goal_drift_enabled`** (default `True`) — opt-in gate for the trajectory-level `GOAL_DRIFT` periodic check (#143). This knob only *detaches* wiring; it never attaches:

```python
# goldfive/runner.py — Runner.__init__
self.goal_drift_enabled: bool = goal_drift_enabled
if not goal_drift_enabled and hasattr(self.steerer, "_goal_drift_call_llm"):
    self.steerer._goal_drift_call_llm = None
elif (goal_drift_enabled and hasattr(self.steerer, "_goal_drift_call_llm")
      and self.steerer._goal_drift_call_llm is None):
    log.warning("goal_drift_enabled=True but no call_llm wired on steerer; "
                "goal-drift judge disabled")
```

- `False` forcibly sets `steerer._goal_drift_call_llm = None`, which mock-only unit tests want so they never see spurious `GOAL_DRIFT` firings.
- `True` is a soft no-op: the steerer's own `goal_drift_call_llm` wiring governs whether the check fires. If `True` but no callable is wired, the Runner logs a **warning** (not an error, #217) and continues — mock/degraded-LLM runners must still construct.
- Guarded with `hasattr` so custom `Steerer` implementations lacking the attribute construct cleanly.

**`planner_gate`** (default `"auto"`) — per-turn planning behaviour. Post-#271 Phase 4 there is no separate "gate then refine" pipeline; `planner.handle_turn` is a single LLM call that both classifies and produces the merged plan.

- `"auto"` — call `planner.handle_turn` every turn (after the seed). The planner returns the next `Plan` (change warranted) or `None` (purely conversational — reuse `session.plan`). Recommended production setting.
- `None` — disable `handle_turn` entirely; every turn falls through to `planner.generate` (pre-#271 behaviour, useful for deterministic replay).
- Always skipped when `user_input` is a `list[Goal]` (caller opted out of NL derivation).

**`drift_self_reporting`** (default `False`) — opt-in for the drift-related self-reporting tools (#196). Default OFF registers only the lifecycle subset; the framework's observation paths (`classify_goal_drift`, `PlanReconciler`, the steerer's refine machinery) are the canonical drift detectors, so the drift-*opinion* tools are redundant surface that inflates the prompt (~200–400 tokens each) and expands hallucination surface. Accepted shapes:

| Value | Effect |
| --- | --- |
| `False` | Lifecycle subset only: `report_task_started/_progress/_completed/_failed/_blocked/_awaiting_approval` + `report_new_work_discovered`. |
| `True` | Full canonical set (pre-#196). |
| `list[str]` | Lifecycle subset **plus** the named drift tools (e.g. `["report_plan_divergence"]`). Names not in `reporting.DRIFT_SELF_REPORTING_TOOL_NAMES` are silently ignored. Materialized eagerly (`[str(n) for n in ...]`) so a generator/mutable list is stable across turns. |

`report_new_work_discovered` is intentionally **not** a drift tool (no observation analog for an agent surfacing genuinely new work), so it stays default-on. The stored value is passed to `select_reporting_tools(self.drift_self_reporting)` on *every* turn in `_run_locked` step 5.

**`fail_fast_on_revision_rejection`** (default `None`) — strict-abort policy for **goldfive-authored** revisions that fail `Plan.validate(for_revision=True, prior=...)`. See PLAN-LIFECYCLE.md §4.5.1.

- `None` (default) → consult env `GOLDFIVE_FAIL_FAST_REVISION_REJECTION` (`"1"` = strict). Explicit `True`/`False` from the kwarg always wins over the env.
- Default behaviour (`False`): non-fatal. Keep the existing `session.plan`, emit a `HUMAN_INTERVENTION_REQUIRED` INFO `DriftDetected` for observability, continue the turn. The `REFINE_FAILURE_THRESHOLD=2` escalation still fires after two consecutive failures of the same `(kind, task_id)`.
- Strict (`True`): emit `RunAborted` via `_abort_turn`.
- **User-authored** drifts (`USER_STEER` from a `ControlMessage`) are **never** gated by this flag. `DefaultSteerer.install_user_steer` guarantees a valid `Plan` returns even when the LLM revision fails validation (PLAN-LIFECYCLE.md §4.2.1) — user-steer rejection is structurally impossible.

### 2.4 The `legacy_kwargs` trap and build-identity stamping

Two construction-time behaviours worth knowing:

1. **Build identity.** The very first thing `__init__` does is call `_detect_build_identity()` and `log.info("goldfive runner starting: version=%s sha=%s", ...)`. This is deliberate (`feedback_verify_running_build.md` — a 30-minute "is the change actually deployed?" diagnosis trap). `_detect_build_identity()` never raises: it tries `importlib.metadata.version("goldfive")`, then `goldfive.__version__`, then `git rev-parse --short HEAD` (only when a `.git` dir exists, 2s timeout, all exceptions swallowed). Do not make this function able to raise.

2. **Legacy kwarg handling.** `max_plan_reinvocations` is accepted with a `DeprecationWarning` and mapped to `max_task_invocations` (only if the latter is `None`). Any *other* unexpected kwarg raises `TypeError`. Do not silently swallow unknown kwargs — the `raise TypeError(f"Runner got unexpected keyword argument(s): {unexpected}")` is intentional.

---

## 3. Per-session Conversation map and lock discipline

A single `Runner` may be shared across many outer ADK sessions (adk-web reuses one Runner). Cross-turn state must **not** leak between distinct outer sessions. This is enforced by keying everything on the outer-session id.

### 3.1 The four per-key dicts

```python
# goldfive/runner.py — Runner.__init__
seed_conv = conversation or Conversation.new()
self._conversations: dict[str, Conversation] = {"": seed_conv}
self._conversation_announced: dict[str, bool] = {"": False}
self._last_session_by_key: dict[str, Session] = {}
self._convo_locks: dict[str, asyncio.Lock] = {}
```

- **`_conversations`** — one `Conversation` per outer-session id. The **empty-string key (`""`)** holds the conversation for *unpinned* (programmatic) callers; this preserves pre-#161 single-Conversation continuity for one-shot scripts. Each pinned caller (adk-web via `GoldfiveADKAgent`) gets its own `Conversation` keyed by `ctx.session.id`.
- **`_conversation_announced`** — flips `True` after a key's `ConversationStarted` fires (so it fires exactly once per key).
- **`_last_session_by_key`** — the most recent turn's `Session` per key, so `ConversationEnded` can piggy-back on its `next_sequence()` cursor (the terminal marker must share its run_id's sequence keyspace).
- **`_convo_locks`** — one `asyncio.Lock` per key (see §3.3).

There is also a plain `self._last_session: Session | None` (any key, updated every run) kept for back-compat with tests/inspectors that read `runner._last_session` directly.

The public `runner.conversation` / `runner.conversation_id` properties return the `""`-keyed slot — a read-only inspection handle preserving the pre-#293 single-Conversation surface. Pinned slots are read via `runner._conversations[session_id]`.

### 3.2 Why per-session isolation exists (the leak it fixed)

Pre-fix, the Runner held a single `self._conversation`. PR #293 keyed only the *prior-plan stash* on session id, but every other field (`goals`, `completed_results`, `turns`, `_next_sequence`) still leaked. The visible regression (validation v4 Class 1): a session asking "Provide the correct answer to 2+2" inherited leaked goals from a prior unrelated session that had run on the same Runner. Keying the *entire* `Conversation` by outer-session id isolates per-session state in full (#271 follow-up to PR #293).

Lifetime note: Conversations live for the Runner's lifetime. The dict is small (one entry per distinct outer session ever seen) and each Conversation is bounded (`recent_turns` cap on prior-turn context). If a future use case introduces churn (many short-lived outer sessions), add a cleanup hook on session end — do not shorten the key.

### 3.3 The per-key run lock

`run` acquires a per-key `asyncio.Lock` **before** doing any work, then delegates to `_run_locked`:

```python
# goldfive/runner.py — Runner.run
async def run(self, user_input, *, context=None, session_id=None):
    convo_key = self._conversation_key(session_id)
    async with self._lock_for(convo_key):
        return await self._run_locked(
            user_input, context=context, session_id=session_id, convo_key=convo_key,
        )
```

Why: adk-web fires a second `/run_sse` while the first is still in flight. Without the lock, turn 2 enters `run` **before** turn 1's `finally`-block `Conversation.stash_plan` lands, so turn 2's `prior_plan_for` returns `None` and seeds `session.plan = Plan.empty(...)`. Turn 2's `handle_turn` then sees an empty seed, the produced plan inherits the empty seed's id, and the "plan_id stable across turns" invariant breaks (the v7class1-1 forensic timeline in `tests/test_intra_session_plan_carry_forward.py`).

- The lock wraps the **entire** lifecycle: `next_turn_session` → `handle_turn` → `executor.run` → `finally`-block stash → `absorb_turn`. Turn 2 waits for all of turn 1 (including the `finally` stash).
- **Concurrent runs on DIFFERENT keys still proceed in parallel** — distinct outer ADK sessions are independent by design.
- `_lock_for` uses `setdefault` (atomic under asyncio's single-thread model) so two concurrent first-time lookups land on the same `Lock`.
- `_conversation_key(session_id)` maps `None`/`""` → `""` (unpinned slot); any non-empty pin gets its own key.

### 3.4 Key helpers

- `_conversation_for(key)` — returns the `Conversation` for `key`, creating (`Conversation.new()`) on miss and initializing `_conversation_announced[key] = False`. A created Conversation gets a fresh id, empty goals/completed_results/turns, and a zero `_next_sequence` cursor — exactly a brand-new outer session's state.
- `_lock_for(key)` — returns the per-key lock, creating on miss.

---

## 4. `_run_locked`: the phase pipeline

`run` is a thin lock wrapper; **all the work is in `_run_locked`**. Read this section as the canonical ordering — a weak model that reorders these phases will break subtle invariants. The phases are numbered in the code's own comments.

Phase map (all under the per-key lock, in this exact order):

| Phase | Does | Emits | On failure |
| --- | --- | --- | --- |
| 1 | resolve `Conversation` for key | — | — |
| 2 | build `Session`, apply outer-session pin | — | — |
| 2b | announce conversation (once per key) | `ConversationStarted` | — |
| 3 | run-started boundary, `reset_for_turn` | `RunStarted` | — |
| 3a | seed `session.plan` (prior or empty) | — | — |
| 4 | derive/accept + merge goals | `GoalDerived` | `_abort_turn` (site 1) |
| 4a | `handle_turn` / `generate` decision | — | `_abort_turn` (site 2) |
| 4b | install / reuse / abort on the plan | `PlanRevised` (via reviser) / `DriftDetected` | `_abort_turn` (sites 3, 4) |
| 5 | register reporting tools | — | `_abort_turn` (site 5) |
| 6/6b/6c/6d | bind steerer ↔ adapter ↔ control | — | `_abort_turn` (sites 6, 7) / log |
| 7 | executor handoff (+ `finally` stash) | executor's `Task*` / `RunCompleted` | `_abort_turn` (site 8) |
| 8 | clear stamps, `absorb_turn` | — | — |

### Phase 1 — resolve the Conversation

```python
convo = self._conversation_for(convo_key)
```

Already under the per-key lock (acquired in `run`). The comment stresses the lock is acquired *before* `_conversation_for` so two concurrent first-time lookups cannot both `Conversation.new()` into the same slot.

### Phase 2 — build the Session, apply the outer-session pin

```python
session = convo.next_turn_session()
pinned = bool(session_id)
if pinned:
    session.run_id = session_id  # type: ignore[assignment]
self._last_session = session
self._last_session_by_key[convo_key] = session
```

`next_turn_session()` mints a fresh `run_id`, copies (not aliases) the accumulated `goals` / `completed_results` / `completed_outputs`, and seeds `_next_sequence` from the Conversation cursor (§6). Then, if the caller pinned a `session_id` (adk-web passing `ctx.session.id`, #161), the Runner **overrides** the freshly-minted `run_id`. Because sinks stamp `Event.session_id` from `Session.id` (which aliases `run_id`), this aligns every goldfive event this turn with the ADK session harmonograf spans already target — fixing the "plan view has empty Gantt" regression.

`pinned` is threaded through everything downstream: it drives `prior_plan_for`, `stash_plan`, and `absorb_turn`'s carry-forward matrix (§5b, §6).

### Phase 2b — announce the Conversation (first turn per key)

```python
if not self._conversation_announced.get(convo_key, False):
    await self._emit_conversation_started(session, conversation=convo)
    self._conversation_announced[convo_key] = True
```

### Phase 3 — emit `RunStarted`, reset per-turn steerer bookkeeping

```python
await self._emit_run_started(session, user_input)
_reset_for_turn = getattr(self.steerer, "reset_for_turn", None)
if callable(_reset_for_turn):
    _reset_for_turn(session)
```

`RunStarted` is emitted **before anything else for this turn** (before goals, before planning). The `reset_for_turn` hook (#215 iter-8 P2) clears the per-turn refine-outcome table; `getattr` so custom steerers degrade gracefully.

### Phase 3a — seed `session.plan` (prior plan or empty)

```python
prior_plan = convo.prior_plan_for(session.id, pinned=pinned)
with channel_processor_active():
    if prior_plan is not None:
        set_session_plan(session, dataclasses.replace(prior_plan, run_id=session.run_id))
    else:
        set_session_plan(session, Plan.empty(run_id=session.run_id))
_ostate.set_current_plan(session.state, session.plan)
```

Critical details:

- `session.plan` is **always non-None** after this phase, so `handle_turn` always sees a real prior (the empty seed on turn 1). The absence of `.tasks` is the unambiguous "first install" signal used later.
- `Plan` is **frozen** (#247). You do not mutate it in place; you derive a stamped variant via `dataclasses.replace(prior_plan, run_id=session.run_id)`.
- The pin onto `session.plan` is a plan mutation, so it is wrapped in `channel_processor_active()` to satisfy the runtime single-writer check (see 11-state-ownership.md). **Any code you add here that installs a plan must be inside a `channel_processor_active()` context.**
- `prior_plan_for(session.id, pinned=pinned)` applies the carry-forward matrix (§6.3).

### Phase 4 — derive/accept goals, merge, refresh summary

```python
try:
    new_goals = await self._resolve_goals(user_input, context, session=session)
except Exception as exc:  # noqa: BLE001
    log.exception("goal derivation failed")
    return await self._abort_turn(session=session, convo=convo, user_input=user_input,
                                  pinned=pinned, reason=f"goal derivation failed: {exc}")
```

`_resolve_goals` (verbatim, `goldfive/runner.py`):

```python
async def _resolve_goals(self, user_input, context, session=None) -> list[Goal]:
    if isinstance(user_input, list):
        if not user_input:
            raise ValueError("Runner.run: empty goal list")
        if not all(isinstance(g, Goal) for g in user_input):
            raise TypeError("Runner.run: list input must be list[Goal]")
        return list(user_input)
    if not isinstance(user_input, str):
        raise TypeError(f"Runner.run: user_input must be str or list[Goal], "
                        f"got {type(user_input).__name__}")
    span_ctx: dict[str, Any] = dict(context or {})
    if session is not None:
        span_ctx.setdefault("run_id", session.run_id)
        span_ctx.setdefault("session_id", session.id)
        span_ctx.setdefault("next_sequence", session.next_sequence)
    if self.sinks:
        span_ctx.setdefault("sinks", list(self.sinks))
    goals = await self.goal_deriver.derive(user_input, context=span_ctx)
    if not goals:
        raise ValueError("GoalDeriver returned an empty goals list")
    return list(goals)
```

- `list[Goal]` input → validated (non-empty, all `Goal`) and returned as-is (deriver bypassed).
- `str` input → `goal_deriver.derive(user_input, context=span_ctx)`. The Runner injects sink/session correlation into `span_ctx` (`run_id`, `session_id`, `next_sequence`, `sinks`) so an `LLMGoalDeriver` can emit `GoldfiveLLMCallStart/End` spans around its internal call. Note `next_sequence` is passed as the **bound method** `session.next_sequence`, not a value — the deriver calls it to allocate a sequence for its own span events. These `setdefault` overrides are deliberate — the Runner owns the sink list and session id, but uses `setdefault` so a caller who deliberately supplied one is respected.
- Empty result → `ValueError`.

**Goal-id collision handling (F9, #322 Layer 4):** the deriver's prompt tends to emit `g1` every turn. The Runner mints a fresh id when a new goal's id collides with an existing session goal:

```python
# goldfive/runner.py — _run_locked, goal merge
existing_ids = {g.id for g in session.goals if g.id}
next_seq = len(session.goals) + 1
for g in new_goals:
    if g.id and g.id in existing_ids:
        while True:
            candidate = f"g{next_seq}"
            next_seq += 1
            if candidate not in existing_ids:
                break
        g = dataclasses.replace(g, id=candidate)   # never mutate in place
    session.goals.append(g)
    if g.id:
        existing_ids.add(g.id)
        next_seq = max(next_seq, len(session.goals) + 1)
```

The `dataclasses.replace` (never in-place mutation) matters because some derivers — e.g. `PassthroughGoalDeriver` — return shallow copies that share `Goal` objects with an internal cache; an in-place `g.id = candidate` would silently rewrite the deriver's stored state for subsequent turns. Without collision renumbering, the older dedup-on-collision path silently *dropped* legitimate new goals, so multi-turn sessions accumulated only turn 1's goals.

Then `_ostate.refresh_goals_summary(session.state, session.goals)` (#152) and `await self._emit_goal_derived(session)`.

### Phase 4a — the per-turn planner decision

This is the heart of Phase 4 (#271). Two paths converge on `next_plan: Plan | None`:

```python
next_plan: Plan | None = None
decided = False
if (self._planner_gate is not None
        and isinstance(user_input, str)
        and hasattr(self.planner, "handle_turn")):
    try:
        next_plan = await self._invoke_handle_turn(...)
        decided = True
    except Exception as exc:  # noqa: BLE001
        log.warning("planner.handle_turn raised; falling through to generate: %s", exc)
        decided = False
```

`handle_turn` is a single LLM call that decides whether the new `user_input` warrants a plan change and, when it does, produces the next revision in one shot. It returns `None` for a purely conversational turn (reuse `session.plan`). `_invoke_handle_turn` threads `available_agents` (tree-preferred), `conversation_history` (`list(conversation.turns)`), and per-turn context (`run_id`, `max_task_invocations`, `conversation.prior_turn_context()`, then caller `context`, then `run_id` re-stamped last so it always wins).

**The generate fallback:**

```python
first_turn_seed = not session.plan.tasks
needs_generate_fallback = (not decided) or (decided and next_plan is None and first_turn_seed)
if needs_generate_fallback:
    ... next_plan = await self.planner.generate(goals=..., available_agents=..., context=...)
```

`planner.generate` runs when: `handle_turn` was skipped (`planner_gate=None`, list-goal input, or no `handle_turn` method), OR it raised, OR it returned `None` on a first-turn empty seed (true for non-LLM planners like `PassthroughPlanner`/`StaticPlanner`). Both `handle_turn` and `generate` failures route to `_abort_turn` with a descriptive reason.

### Phase 4b — install / reuse / abort on the plan decision

Three branches on `next_plan`:

1. **`next_plan is not None`** → `installed = await self._install_revision(...)` (§5b). On success, continue. On failure (`not installed`): this is always the goldfive-authored path. Emit the `HUMAN_INTERVENTION_REQUIRED` INFO drift (routed through `steerer.drift._emit_drift_detected` so lifecycle/condition_id stamping fires consistently), log a warning, and either `_abort_turn` (strict mode) or keep the existing plan and continue (default). `session.plan` is unchanged on rejection because `_install_with_drift` validates *before* applying.

2. **`next_plan is None` and `not session.plan.tasks`** → first turn AND purely conversational on an empty seed. No plan to drive → `_abort_turn(reason="no plan generated")`.

3. **`next_plan is None` and a real prior exists** → conversational follow-up. Reuse `session.plan` unchanged, no `PlanRevised`. Set `session._conversational_turn = True` (F6, #277) — this flag lets the executor handoff (Phase 7) optionally wrap the input, and lets a parallel adapter-plugin layer tighten the tool surface without coordinating through the message body.

### Phase 5 — register reporting tools

```python
await self.agent.register_reporting_tools(select_reporting_tools(self.drift_self_reporting))
```

Runs every turn. Failure → `_abort_turn`. See §2.3 for `drift_self_reporting` shapes and 13-reporting-tools-and-approval.md.

### Phase 6 — bind the steerer (four sub-steps)

The binding is intricate because the steerer, adapter, and control channel must all know about each other. Every step is `getattr`/`callable`-probed so third-party components that predate a hook still work.

| Step | Call | Purpose | Failure |
| --- | --- | --- | --- |
| 6 | `self.steerer.bind(sinks=list(self.sinks), planner=self.planner)` | Give the steerer sinks + planner. Idempotent (executor re-binds). | `_abort_turn` |
| 6b | `agent.bind_steerer(self.steerer)` | Wire steerer INTO the adapter so plugin callbacks (`AgentInvocationStarted`, `DelegationObserved`, …) fire — they short-circuit when `SessionContext.steerer is None`. | `_abort_turn` |
| 6c | `steerer.bind_adapter(self.agent)` | Wire the adapter BACK into the steerer (#139) so mid-invocation cancels carry a symbolic reason. | `log.debug`, continue (best-effort) |
| 6d | `steerer.bind_control_channel(self._control)` | Give the steerer the control channel so goldfive-authored drift mints `GOLDFIVE_STEER` / `GOLDFIVE_PAUSE_ESCALATE` onto the same cancel-and-restart junction as user `STEER`/`PAUSE`. | `log.debug`, continue |

Note the asymmetry: 6 and 6b abort on failure (they are load-bearing for observability); 6c and 6d are best-effort (custom steerers may not implement them).

### Phase 7 — hand off to the executor

```python
with _state_audit.cancellation_stash_audited("Runner.run.executor_drive"):
    try:
        executor_kwargs = dict(plan=session.plan, session=session, adapter=self.agent,
                               steerer=self.steerer, planner=self.planner, sinks=list(self.sinks))
        if self.control is not None:
            executor_kwargs["control"] = self.control
        if isinstance(user_input, str):
            run_sig = inspect.signature(self.executor.run)
            if "user_input" in run_sig.parameters:
                executor_user_input = user_input
                if getattr(session, "_conversational_turn", False) and user_input.strip():
                    executor_user_input = self._prompt_shaper.wrap_conversational_input(
                        user_input=user_input, session=session, steerer=self.steerer)
                executor_kwargs["user_input"] = executor_user_input
        outcome = await self.executor.run(**executor_kwargs)
    except Exception as exc:  # noqa: BLE001
        log.exception("executor.run raised")
        return await self._abort_turn(..., reason=f"executor.run raised: {exc}")
    finally:
        # STASH — see §6.3
        ...
```

The `finally` block, verbatim (this is the load-bearing `CancelledError`-safe stash):

```python
finally:
    if session.plan is not None and session.plan.tasks:
        convo.stash_plan(session, pinned=pinned)
        log.info("Runner.run: stashed prior plan for next turn's handle_turn "
                 "(plan_id=%s revision_index=%d session_id=%s)", ...)
    _state_audit.mark_stash_completed()
```

Key points:

- **`user_input=` is passed by inspection.** The Runner only passes `user_input` if the executor's `run` signature accepts it (overlay-capable executors do). Legacy executors keep working with the base kwargs. Do not assume every executor takes `user_input`.
- **The stash is in `finally`, not after `await`.** #271 Gap 1's forensic finding: the prior post-success stash (PR #282) was bypassed when ADK cancelled the executor coroutine — `CancelledError` is a `BaseException`, so it flowed past the `except Exception` handler and out of `run` entirely, skipping the stash. The ADK-web user-steer flow hit this on validation v2 (zero stash log lines across 4 turns). `finally` runs the stash on every exit path (success, `Exception`, `BaseException`) without swallowing the exception.
- **F6 conversational wrapping is gated.** On a conversational turn, the input is passed through `PromptShaper.wrap_conversational_input`. Under `observation_only=True` (the strictly-passive default) this returns the raw input **unchanged**; only in active mode does it return the composed F6 directive. The gate lives in `PromptShaper` (which delegates to `steerer.steering_is_active`), NOT in `runner.py` — respecting invariant #2. The wrap lives at the executor handoff (not earlier) so `absorb_turn` / event summaries still see the user's *actual* question.
- **The whole call is inside `_state_audit.cancellation_stash_audited(...)`** (#271 Phase 3.5), whose tripwire verifies the prior-plan stash fired before a cancel propagated past the Runner. The `finally` block satisfies it via `mark_stash_completed()`.

The full revision-rejection observability emission (Phase 4b branch 1, `not installed`) — worth seeing verbatim because a weak model tends to delete it as "dead" or route it wrongly:

```python
obs_drift = DriftEvent(
    kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
    severity=DriftSeverity.INFO,
    detail="autonomous refine produced an invalid plan revision; "
           "existing plan retained; next refine cycle may try again",
    authored_by="goldfive")
emit_helper = getattr(getattr(self.steerer, "drift", None), "_emit_drift_detected", None)
if callable(emit_helper):
    try:
        await emit_helper(session, obs_drift)
    except Exception as exc:
        log.warning("Runner.run: emitting HUMAN_INTERVENTION_REQUIRED "
                    "observability drift raised: %s", exc)
if self._fail_fast_on_revision_rejection:
    return await self._abort_turn(..., reason="plan revision rejected by validator")
# else: keep session.plan (unchanged — validate raised before _apply_revision), continue
```

It is routed through `steerer.drift._emit_drift_detected` (not a raw `emit`) so the lifecycle/`condition_id` stamping fires consistently and harmonograf renders it alongside the surrounding refine activity. The `getattr` chain degrades gracefully for custom steerers without a `drift` sub-object.

### Phase 8 — post-executor cleanup and absorb

```python
_ostate.clear_current_task(session.state)
_ostate.clear_active_steer(session.state)
convo.absorb_turn(outcome,
                  user_input_summary=_initial_goal_summary(user_input),
                  pinned=pinned)
return outcome
```

`clear_current_task` / `clear_active_steer` wipe per-turn stamps (#152); the plan id + goals summary stay (meaningful cross-turn). `absorb_turn` folds the turn back into the Conversation (§6.2). This runs only on the **success/handled path** — the `finally`-block stash (§6.3) covers the `BaseException` path that bypasses this line.

---

## 5. `_abort_turn`: the one abort path (post-#489)

Before #489, `_run_locked` repeated the same three-line block (`_emit_run_aborted` → build failed `ExecutionOutcome` → `absorb_turn`) at **eight** sites. #489 extracted the shared tail into `_abort_turn`; the genuine per-site deltas (the reason string, whether `log.exception` fires) stay at the call sites. (The other half of #489 — decomposing `SequentialExecutor._run_overlay` into named stage methods — is 04-executors-and-control.md's concern.)

```python
# goldfive/runner.py — Runner._abort_turn
async def _abort_turn(self, *, session, convo, user_input, pinned, reason) -> ExecutionOutcome:
    await self._emit_run_aborted(session, reason)
    outcome = ExecutionOutcome(success=False, session=session, reason=reason)
    convo.absorb_turn(outcome, user_input_summary=_initial_goal_summary(user_input), pinned=pinned)
    return outcome
```

### 5.1 The eight call sites (each is a distinct pre-/mid-turn failure)

| # | Phase | Reason string |
| --- | --- | --- |
| 1 | 4 — goal derivation | `f"goal derivation failed: {exc}"` |
| 2 | 4a — generate fallback | `f"planner.generate raised: {exc}"` |
| 3 | 4b — revision rejected, strict mode | `"plan revision rejected by validator"` |
| 4 | 4b — first-turn conversational, no plan | `"no plan generated"` |
| 5 | 5 — register reporting tools | `f"register_reporting_tools raised: {exc}"` |
| 6 | 6 — steerer.bind | `f"steerer.bind raised: {exc}"` |
| 7 | 6b — adapter.bind_steerer | `f"adapter.bind_steerer raised: {exc}"` |
| 8 | 7 — executor.run | `f"executor.run raised: {exc}"` |

Why `absorb_turn` is called even on abort: the next turn's carry-forward must see a consistent stash. An aborted turn still records a `TurnRecord` (with `outcome_success=False`) so the planner on turn N+1 sees the failure. **If you add a new early-exit, you MUST go through `_abort_turn`** — otherwise the Conversation's turn log and prior-plan stash desync and the next turn misbehaves. See Common Mistakes §8.1.

Note the distinction between these Runner-owned **pre-executor** `RunAborted` emissions and the **terminal** `RunAborted` the *executor* emits at its own state-machine end (04-executors-and-control.md). `_abort_turn` is only for failures that happen *before or at* the executor handoff.

---

## 5b. `_install_revision`: routing the produced plan

`_install_revision` (called from Phase 4b) dispatches across steerer APIs based on what is actually happening (#271 Option A):

```python
# goldfive/runner.py — Runner._install_revision (branch skeleton)
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
return bool(installed)
```

- **Turn 1 install** (`session.plan` is the `Plan.empty` seed → no tasks) → `install_initial_plan`. **No `DriftDetected`** — installing the first plan is structural, not an intervention. (This eliminates the synthetic `USER_STEER` drift the pre-Option-A path fabricated.) It emits `PlanRevised` with `revision_index = 1` via `PlanReviser._emit_plan_revised`.
- **Pivot turn** (F5, #322 Layer 2 / #204) → `handle_turn` set `_goldfive_pivot` on the plan; route through `install_initial_plan(is_pivot=True)` so Rule 6 (terminal-task preservation) does not reject a legitimate pivot for "dropping" the prior's terminal tasks. `is_pivot=True` validates structurally only (no `prior`, no runtime-terminal fold).
- **Turn N+1 replan** → `install_revision_for_drift` with a `NEW_WORK_DISCOVERED` INFO drift — the honest classification (the user surfaced new work, not an intervention, not a `USER_STEER`).
- **Genuine operator `STEER` `ControlMessage`** does **not** flow through here — it takes the executor's steer loop straight to `install_revision_for_user_steer`.
- `_install_revision` first re-binds steerer+adapter (idempotent), stamps `run_id` on the frozen plan via `dataclasses.replace` if missing, and returns `True`/`False`. It never raises — bind/install exceptions return `False`.

Remember the ownership: the `PlanRevised` envelope (including rev 1) is emitted by `PlanReviser._emit_plan_revised`, **not** the Runner.

---

## 6. Conversation: cross-turn state

`Conversation` (in `conversation.py`) persists state across successive `runner.run()` calls on the same Runner. Each `run()` is still a distinct goldfive *run* (own `run_id`, own `RunStarted`/`RunCompleted`), but the Runner seeds each turn's `Session` from the Conversation and folds the result back. The UX win: a user says "make it funnier" on turn 2 and the planner sees turn 1's output.

### 6.1 Fields and `TurnRecord`

```python
@dataclasses.dataclass
class Conversation:
    id: str
    started_at_ms: int
    goals: list[Goal]                     # append-only, deduped by Goal.id
    completed_results: dict[str, str]     # task_id -> self-reported summary
    completed_outputs: dict[str, str]     # task_id -> full actual output (zicato#12)
    turns: list[TurnRecord]
    _next_sequence: int = 0               # the wire-sequence cursor (see 6.4)
    _last_plan: Plan | None = None        # prior-plan stash
    _last_plan_session_id: str = ""       # session id that produced _last_plan
    _last_plan_pinned: bool = False       # was that turn pinned?
```

`TurnRecord` is a compact per-turn summary (`run_id`, `user_input_summary`, `plan_summary`, `outcome_success`, `outcome_reason`, `completed_task_ids`, `started_at_ms`, `ended_at_ms`) — kept compact because it is serialized into planner prompts.

`Conversation.new()` builds a fresh empty Conversation with a `uuid4().hex` id.

### 6.2 `next_turn_session` and `absorb_turn`

**`next_turn_session()`** builds the turn's `Session`:
- fresh `run_id` (`uuid4().hex`),
- `conversation_id = self.id`,
- **copies** (not aliases) `goals`, `completed_results`, `completed_outputs` so the executor's in-turn mutations don't retroactively rewrite the Conversation's record,
- seeds `_next_sequence` from the Conversation cursor (§6.4).

**`absorb_turn(outcome, *, user_input_summary, pinned)`** folds the turn back, in this exact order:
1. **Goals merge** — append `session.goals` not already present by id (restated goals don't duplicate).
2. **Results merge** — `completed_results.update(...)` and `completed_outputs.update(...)`. **Later turns win** on id collisions, so a revised result on a follow-up turn is visible to the turn after it.
3. **Sequence lift** — `self._next_sequence = max(self._next_sequence, int(session._next_sequence))` (§6.4). `max` is defensive against a turn that aborted before advancing past the seed.
4. **Plan stash** — `self.stash_plan(session, pinned=pinned)` (§6.3).
5. Append a `TurnRecord` (reading `plan_summary` and the COMPLETED task ids off `session.plan` when present).

The body, verbatim (`goldfive/conversation.py`):

```python
def absorb_turn(self, outcome, *, user_input_summary="", pinned=False) -> TurnRecord:
    session = outcome.session
    seen_ids = {g.id for g in self.goals if g.id}
    for g in session.goals:
        if g.id and g.id in seen_ids:
            continue
        self.goals.append(g)
        if g.id:
            seen_ids.add(g.id)
    self.completed_results.update(session.completed_results)   # later turns win
    self.completed_outputs.update(session.completed_outputs)   # zicato#12
    self._next_sequence = max(self._next_sequence, int(session._next_sequence))
    self.stash_plan(session, pinned=pinned)
    plan_summary = ""
    completed_ids: list[str] = []
    if session.plan is not None:
        plan_summary = session.plan.summary or ""
        completed_ids = [t.id for t in session.plan.tasks
                         if t.status.value == "COMPLETED" and t.id]
    record = TurnRecord(
        run_id=session.run_id, user_input_summary=user_input_summary,
        plan_summary=plan_summary, outcome_success=bool(outcome.success),
        outcome_reason=outcome.reason or "", completed_task_ids=completed_ids,
        started_at_ms=session.started_at_ms, ended_at_ms=_now_ms())
    self.turns.append(record)
    return record
```

`absorb_turn` is called from **both** the success path (Phase 8) and `_abort_turn`. **Do not touch its ordering** — see Common Mistakes §8.2.

### 6.3 The prior-plan stash and the carry-forward matrix

The stash lets turn N+1 seed `session.plan` with turn N's plan (so `handle_turn` sees a real prior). It is scoped by session id so a shared Runner does not leak plans across outer sessions.

**`stash_plan(session, *, pinned)`** records `(_last_plan, _last_plan_session_id, _last_plan_pinned)`, but only when `session.plan is not None and session.plan.tasks` (an empty plan is not a meaningful prior).

**`prior_plan_for(session_id, *, pinned)`** returns the stashed plan iff carry-forward applies:

```python
# goldfive/conversation.py — Conversation.prior_plan_for
if self._last_plan is None:
    return None
if self._last_plan_pinned or pinned:
    if self._last_plan_session_id != session_id:
        return None
return self._last_plan
```

The full matrix (both from the docstring):

| prior turn | new turn | carry forward? | why |
| --- | --- | --- | --- |
| unpinned | unpinned | **YES** | Conversation-level continuity for programmatic callers. |
| unpinned | pinned | NO | pin signals a switch to an externally-owned identity; boundary. |
| pinned | unpinned | NO | symmetric: leaving an externally-owned identity is also a boundary. |
| pinned | pinned, ids match | **YES** | same outer ADK session across turns; intra-session continuity. |
| pinned | pinned, ids differ | NO | the regression case — two distinct outer sessions sharing one Runner (v4 Class 1). |

**Where `stash_plan` is called from — two sites, idempotent:**
1. The Runner's Phase-7 `finally` block: `if session.plan is not None and session.plan.tasks: convo.stash_plan(session, pinned=pinned)`. This exists to cover the **`BaseException` (`CancelledError`)** path — when ADK closes the runner mid-stream, `CancelledError` propagates out of `await self.executor.run(...)`, and because it is a `BaseException` (not `Exception`) the `except Exception` handler does NOT catch it. Control flows out of `run` entirely, bypassing `absorb_turn`. Putting the stash in `finally` runs it regardless of exit path (#271 Gap 1). The exception still propagates after the stash — the block does not swallow it.
2. `absorb_turn` (the normal path). Idempotent: a subsequent `absorb_turn` re-stashes the same `(plan, session_id, pinned)` tuple, so both-paths callers produce identical state.

Pre-fix, the stash lived on `Runner._last_plan` — a process-scoped attribute that leaked across outer sessions (v4 Class 1). Moving it onto `Conversation` (keyed by session id) fixed the leak.

### 6.4 The wire-sequence cursor (#271 Gap 2)

`_next_sequence` is a Conversation-level cursor that keeps `Event.sequence` globally unique **within a conversation, across turns**. Why it matters: #161's outer-session pin makes `Session.run_id` **repeat** across turns. harmonograf's `goldfive_events` PK is `(session_id, run_id, sequence)`. Without a carried cursor, turn 2's early events (sequence 0, 1, 2, …) collide with turn 1's already-persisted rows, and the storage layer's `INSERT OR IGNORE` **silently drops** turn 2's `plan_submitted`, `agent_invocation_started`, etc.

Flow:
- `next_turn_session()` seeds `Session._next_sequence` from `Conversation._next_sequence`.
- Each event calls `session.next_sequence()`, advancing the private counter.
- `absorb_turn` lifts the high-water mark back: `self._next_sequence = max(self._next_sequence, int(session._next_sequence))`.

Single-turn callers see no change (cursor starts at 0). **If you emit events outside the normal turn flow, you must use `session.next_sequence()`** so the cursor stays monotonic — do not hand-pick a sequence number.

### 6.5 `prior_turn_context`

`prior_turn_context(*, recent_turns=3)` returns the cross-turn dict merged into the planner context: `conversation_id`, `prior_completed_results` (a copy), a capped `prior_turns` window (each a compact dict with `run_id` / `user_input_summary` / `plan_summary` / `outcome_success` / `outcome_reason` / `completed_task_ids`), and `turn_index = len(self.turns)`. `recent_turns` caps the window so prompts don't grow unbounded on long conversations; `recent_turns < 0` clamps to 0.

---

## 7. `run_streamed`, `resume`, `new_conversation`, `close`

### 7.1 `run_streamed` — yield inner-adapter events live

`run_streamed` is an async generator that yields, in order, every framework-native event the adapter observes mid-invocation (ADK `Event` objects: `transfer_to_agent`, model text parts, function calls/responses), then **exactly one trailing `ExecutionOutcome`** as the final element. Consumers distinguish the two shapes via `isinstance(item, ExecutionOutcome)`.

Mechanics:
- It does **not** call `run` recursively. It subscribes a sync listener to `agent.subscribe_adk_events` (when present), which `put_nowait`s each event into an unbounded `asyncio.Queue`. `run` is driven in a background `asyncio.Task` (`_drive`). This decouples consumer backpressure from the inner Runner — a slow consumer cannot stall agent progress.
- A `_DONE` sentinel is enqueued in `_drive`'s `finally` so the consumer loop drains buffered events then stops.
- On `CancelledError` / `GeneratorExit` (adk-web disconnect or early `aclose()`), it cancels `run_task`, awaits it (so the driver's `try/finally` teardown — including the plan stash — runs), suppresses the resulting exception, and re-raises.
- Always unsubscribes the listener in `finally`.
- **Non-ADK adapters (callable, Claude SDK)** have no streamable events; `run_streamed` still works — it yields no mid-run events and produces the outcome at the end, exactly as `run` would. Callers do not switch on adapter type.

The driver + consumer loop, verbatim-ish (`goldfive/runner.py`):

```python
queue: asyncio.Queue[Any] = asyncio.Queue()          # unbounded
subscribe = getattr(self.agent, "subscribe_adk_events", None)
unsubscribe = getattr(self.agent, "unsubscribe_adk_events", None)

def _listener(event):        # sync — must not block the adapter loop
    try: queue.put_nowait(event)
    except Exception: log.debug("run_streamed: queue.put_nowait unexpectedly raised")

if callable(subscribe): subscribe(_listener)
_DONE = object()

async def _drive() -> ExecutionOutcome:
    try:
        return await self.run(user_input, context=context, session_id=session_id)
    finally:
        queue.put_nowait(_DONE)                       # end-of-stream sentinel

run_task = asyncio.create_task(_drive())
try:
    while True:
        item = await queue.get()
        if item is _DONE: break
        yield item
    outcome = await run_task
    yield outcome
except (asyncio.CancelledError, GeneratorExit):
    run_task.cancel()
    try: await run_task
    except (asyncio.CancelledError, Exception): pass
    raise
finally:
    if callable(unsubscribe):
        try: unsubscribe(_listener)
        except Exception as exc: log.debug("run_streamed: unsubscribe raised: %s", exc)
```

Why `put_nowait` on an *unbounded* queue: it never raises (so the adapter's event loop is never blocked on a slow consumer), and backpressure is absorbed by memory rather than stalling agent progress. Why cancel-then-await `run_task` on `CancelledError`: to let `_drive`'s `finally` (and inside it, `run`'s Phase-7 `finally` stash) actually run before the generator dies — otherwise a mid-stream adk-web disconnect would lose the turn's plan stash.

This is the primary path for `GoldfiveADKAgent` so `adk web` sees per-agent activity while the goldfive pipeline runs.

### 7.2 `resume` — replay-only (do NOT assume it resumes)

**This is the single most dangerous method to misread in this chapter.** `resume(persistence_path)` does **not** continue execution. It:

```python
# goldfive/runner.py — Runner.resume (behaviour)
events = replay_from_jsonl(persistence_path)
session = reconstruct_session(events)
# scan events for the last run_completed / run_aborted payload:
#   run_completed → success=True, reason=""
#   run_aborted   → success=False, reason=evt.run_aborted.reason
return ExecutionOutcome(success=success, session=session, reason=reason)
```

It reconstructs a `Session` from a JSONL log and reports the latest terminal marker as an `ExecutionOutcome`. It does **not** re-drive the planner or executor from the latest cursor.

- The docstring is explicit: *"We do **not** continue execution from the latest cursor — full resume semantics require planner/executor co-operation that is out-of-scope for this PR. Callers who need a live continuation should construct a new Runner with the goals recovered from the log."*
- `TODO(#15)`: once executors grow a `resume_from` hook, continue from the latest un-finished task.
- Requires the `proto` extra (`replay_from_jsonl` / `reconstruct_session`); raises a clear `RuntimeError` if the stubs are absent.

If someone files "resume doesn't continue my run," the answer is: **that is by design (TODO #15)**, not a bug.

What the two helpers do (both in `goldfive.sinks`, see 12-events-sinks-telemetry.md):
- `replay_from_jsonl(path)` — reads the JSONL persistence log line-by-line and decodes each proto `Event` envelope, returning them in order.
- `reconstruct_session(events)` — folds that event stream into a `Session`: it re-applies `PlanRevised` to rebuild `session.plan`, `Task*` transitions to set each task's terminal status, and goal/result events to repopulate `session.goals` / `completed_results`. The result is a faithful snapshot of the *final observed state*, not a live executor.

The `ExecutionOutcome` `resume` returns has `success` / `reason` derived by scanning for the **last** terminal payload (`run_completed` → `success=True`; `run_aborted` → `success=False, reason=...`). If the log ends mid-run with no terminal marker, `success=False, reason="run did not complete before persistence ended"`.

### 7.2b The `ExecutionOutcome` return shape

Every `run` / `_run_locked` / `_abort_turn` path returns an `ExecutionOutcome` (`goldfive/results.py`):

```python
ExecutionOutcome(success=<bool>, session=<the live Session>, reason=<str>)
```

- `success` — `True` only on a completed executor run; `False` on any abort.
- `session` — the **live** `Session` object (not a copy). Callers inspect `outcome.session.plan.tasks`, `outcome.session.completed_results`, `outcome.session.goals`. Because it is live, the values reflect all in-turn mutations.
- `reason` — empty on success; the abort reason string otherwise (one of the eight in §5.1, or an executor-terminal reason).

`_initial_goal_summary(user_input)` builds the one-liner used in `RunStarted` and `TurnRecord.user_input_summary`: the raw string for `str` input, or the first goal's `.summary` for `list[Goal]` input, else `""`. It is called *before* goals derive, so it must not depend on `session.goals`.

### 7.3 `new_conversation` — reset all keyed state

`new_conversation(*, reason="")` emits `ConversationEnded` for every announced outgoing Conversation (piggy-backing on its anchor session's sequence), then reinstalls fresh state: `_conversations = {"": Conversation.new()}`, `_conversation_announced = {"": False}`, `_last_session_by_key = {}`, `_last_session = None`. The default `""` slot is restored eagerly so the public `conversation` / `conversation_id` properties keep returning a real handle; pinned slots are recreated lazily by `_conversation_for` on the next run. No Runner-side plan reset is needed — the fresh Conversation already starts with `_last_plan = None`.

### 7.4 `close` — idempotent teardown

`close()` (gated on `self._closed`) does, in order:
1. For every announced per-session Conversation: run the **orphan-PENDING audit** (`_audit_conversation_pending_at_close`, #212) — at conversation end there is no next turn, so any still-PENDING task is by definition orphaned; it transitions each to `CANCELLED` with `cancel_reason="conversation_ended:no_engaging_turn"` (idempotent — `mark_task_cancelled` no-ops on terminal tasks). Then emit `ConversationEnded`.
2. `getattr(self.steerer, "shutdown", None)` (duck-typed) — called only if the steerer exposes a top-level `shutdown`. **`DefaultSteerer` does not** (the drain lives on `DriftObserver`, reached as `steerer.drift.shutdown()`), so for the default steerer this step is a **no-op**; it is a cleanup hook for custom steerers. The background reasoning-judge tasks (#251) are actually drained at every run boundary by the executor's `_drain_steerer_at_run_boundary` (`goldfive/executors/sequential.py`), not at `close`.
3. `sink.close()` for every sink.
4. `maybe_close_call_llm` on `planner._call_llm` and `goal_deriver._call_llm` — standard SDK clients own aiohttp sessions that leak unless closed.
5. Registered close hooks (via `add_close_hook`), in registration order, AFTER sinks. A raising hook is logged and does not block subsequent hooks.

Every step swallows-and-logs exceptions — failing cleanup must not hang a process. A second `close()` is a no-op.

### 7.5 Extension API

- `add_sink(sink)` — appends to `self.sinks`; takes effect for subsequent `run`s only (in-flight runs use their kickoff snapshot).
- `add_close_hook(hook)` — async callable run by `close` after sinks.
- `control` property + setter — attach a `ControlChannel` post-construction:

```python
@control.setter
def control(self, value: ControlChannel) -> None:
    if self._control is value:           # same identity → no-op
        return
    if self._control is not None:        # different channel already attached → refuse
        raise RuntimeError("Runner already has a control channel attached; "
                           "detach it first or construct the runner with a specific one.")
    self._control = value
```

Idempotent on same-identity re-attach; raises `RuntimeError` if a *different* channel is already attached (construct a fresh Runner instead — §8.14b).

---

## 7b. Worked trace: a two-turn pinned session

To cement the phase pipeline and the cross-turn bookkeeping, here is the exact sequence for a two-turn adk-web session (`session_id="S1"` pinned on both turns). "→" is a call; indentation is nesting.

**Turn 1 — user says "Write a haiku about rivers":**

1. `run("Write a haiku…", session_id="S1")` → `convo_key = "S1"` → acquire `_lock_for("S1")`.
2. `_run_locked`: `_conversation_for("S1")` → miss → `Conversation.new()` (fresh id `C1`, empty everything, cursor 0). `_conversations["S1"] = C1`.
3. `session = C1.next_turn_session()` → `run_id = <uuid A>`, `conversation_id = C1.id`, `_next_sequence = 0`. `pinned = True` → `session.run_id = "S1"`. `_last_session_by_key["S1"] = session`.
4. Phase 2b: not announced → emit **`ConversationStarted`** (seq 0), mark announced.
5. Phase 3: emit **`RunStarted`** (seq 1). `reset_for_turn`.
6. Phase 3a: `C1.prior_plan_for("S1", pinned=True)` → `_last_plan is None` → returns `None` → `set_session_plan(session, Plan.empty(run_id="S1"))`. `session.plan.tasks == []`.
7. Phase 4: `_resolve_goals` derives `[g1: "haiku about rivers"]`. No collision (empty session goals). `session.goals = [g1]`. Emit **`GoalDerived`** (seq 2).
8. Phase 4a: `handle_turn` returns a real `Plan P1` (one task `t1`). `decided = True`. Not a first-turn None, so no generate fallback.
9. Phase 4b branch 1: `_install_revision(P1)` → `first_turn = True` (no tasks on the empty seed) → `install_initial_plan` → emits **`PlanRevised`** rev 1 (seq 3), no `DriftDetected`. `installed = True`.
10. Phase 5: `register_reporting_tools`. Phase 6: bind steerer/adapter/control.
11. Phase 7: `executor.run(...)`. Executor drives the agent, emits `Task*` events (seq 4…N), and at its end emits **`RunCompleted`** (seq N+1). `finally`: `session.plan.tasks` non-empty → `C1.stash_plan(session, pinned=True)` → `_last_plan = P1`, `_last_plan_session_id = "S1"`, `_last_plan_pinned = True`.
12. Phase 8: clear per-turn stamps. `C1.absorb_turn(outcome, pinned=True)` → merges goal `g1`, lifts cursor to N+2, re-stashes P1 (idempotent), appends `TurnRecord[0]`.
13. Lock released.

**Turn 2 — user says "make it about oceans instead":**

1. `run("make it about oceans…", session_id="S1")` → `convo_key = "S1"` → acquire the **same** `_lock_for("S1")` (turn 1 already released it).
2. `_conversation_for("S1")` → hit → returns `C1` (with `g1`, cursor N+2, `_last_plan = P1`).
3. `session = C1.next_turn_session()` → **new** `run_id = <uuid B>`, `_next_sequence = N+2` (carried!). `pinned = True` → `session.run_id = "S1"`.
4. Phase 2b: already announced → **no** `ConversationStarted`.
5. Phase 3: emit **`RunStarted`** at seq N+2 (not 0 — the carried cursor keeps the PK unique even though `run_id`/`session_id` is again `"S1"`).
6. Phase 3a: `C1.prior_plan_for("S1", pinned=True)` → `_last_plan_pinned` is `True` **and** `_last_plan_session_id ("S1") == session_id ("S1")` → **carry forward** → `set_session_plan(session, dataclasses.replace(P1, run_id="S1"))`. `session.plan` now has `t1`.
7. Phase 4a: `handle_turn` sees the real prior plan P1 and the new message, returns `P2` (pivot: rivers→oceans, `_goldfive_pivot=True`). Because `session.plan.tasks` is non-empty and it is a pivot, `_install_revision` routes through `install_initial_plan(is_pivot=True)` — Rule 6 is skipped so dropping `t1` is legal.
8. …executor drives P2, `RunCompleted`, `finally` stash `_last_plan = P2`, `absorb_turn` appends `TurnRecord[1]`.

The two things that would have broken pre-fix: (a) if the cursor were not carried, turn 2's `RunStarted` at seq 0 would collide with turn 1's persisted seq-0 row under the pinned `run_id` and be silently dropped by `INSERT OR IGNORE`; (b) if the stash were process-scoped on the Runner, a *different* session `S2` sharing this Runner would have inherited P2.

---

## 7c. Provenance: the PRs/issues that shaped this subsystem

When you see a `#NNN` comment in `runner.py` / `conversation.py` and need to know *why* the code is shaped that way, this table is the index. Read the corresponding design doc or `git show` before "simplifying" any of it — most of these encode a fixed regression.

| Ref | What it introduced / fixed | Where in this chapter |
| --- | --- | --- |
| #78 | Phase-3 cross-turn state: goals accumulate, `completed_results` carry over, `TurnRecord` history. | §6 |
| #143 | Trajectory-level `GOAL_DRIFT` periodic check + the `goal_drift_enabled` detach gate. | §2.3 |
| #141 | Overlay model: `user_input` threaded to the executor, `NOT_NEEDED` task status, F6 flag. | §4 Phase 7 |
| #151 | Tree-shaped `available_agents_tree` (coordinator+AgentTool). | §2.1, §4 Phase 4a |
| #152 | Orchestration-state stamps: `refresh_goals_summary`, `set_current_plan`, `clear_current_task`. | §4 Phases 3a/4/8 |
| #155 | `Session.id` aliases `run_id`; sinks stamp `Event.session_id` from it. | §4 Phase 2 |
| #161 | Outer-session pin: `Runner.run(session_id=...)` overrides `run_id` to align with adk-web/harmonograf. | §4 Phase 2 |
| #196 | `drift_self_reporting` gate; default registers only the lifecycle subset. | §2.3, §4 Phase 5 |
| #204 / #322 | F5 pivot detection (`_goldfive_pivot`) routing pivots through `install_initial_plan`. | §5b |
| #212 | Close-time orphan-PENDING audit (`conversation_ended:no_engaging_turn`). | §7.4 |
| #217 | `goal_drift_enabled=True` with no callable → warn, don't raise (mock runners construct). | §2.3 |
| #247 | `Plan` frozen; all installs via `dataclasses.replace` + `channel_processor_active`. | §4 Phase 3a, §5b |
| #251 | Background reasoning-judge tasks drained at run boundaries via `_drain_steerer_at_run_boundary` (executor); `close`'s duck-typed `getattr(steerer,"shutdown")` is a no-op for `DefaultSteerer`. | §7.4 |
| #271 Phase 4 | `planner.handle_turn` collapses the gate-then-refine pipeline into one LLM call. | §4 Phase 4a |
| #271 Gap 1 | Prior-plan stash moved into the Phase-7 `finally` to survive `CancelledError`. | §4 Phase 7, §6.3 |
| #271 Gap 2 | Conversation-level `_next_sequence` cursor keeps wire sequence collision-free under the pin. | §6.4 |
| #271 Phase 3.5 | `cancellation_stash_audited` tripwire + `mark_stash_completed`. | §4 Phase 7 |
| #271 Option A | `_install_revision` splits initial-install (no drift) from replan (NEW_WORK_DISCOVERED). | §5b |
| #293 / v4 Class 1 | Per-session `Conversation` **map** (keyed by outer-session id) to stop cross-session leak. | §3 |
| #294 / v7class1-1 | Per-key `asyncio.Lock` so concurrent `/run_sse` turns on one session serialise. | §3.3 |
| #322 Layer 4 (F9) | Goal-id collision renumbering so multi-turn goals don't get dropped. | §4 Phase 4 |
| #15 (open TODO) | `resume` is replay-only until executors grow a `resume_from` hook. | §7.2 |
| #489 | `_abort_turn` extracted from 8 copy-paste sites; `_run_overlay` stage methods (executor). | §5 |
| #488 | Single `observation_only` predicate `is_active_steering` / `steering_is_active`. | §2 (invariants), §8.6 |
| zicato#12 | `completed_outputs` (full actual output) carried alongside `completed_results`. | §6.1 |

None of these are on the agency-preservation branch — every ref above is **on main**. (The agency-preservation Stages 1–3, #453–#474, live on an unmerged branch and must not be copied into `runner.py`.)

---

## 8. Common mistakes

### 8.1 Adding an abort path without `_abort_turn`

**Wrong:**
```python
# new early-exit you're tempted to write inside _run_locked
await self._emit_run_aborted(session, "my new failure")
return ExecutionOutcome(success=False, session=session, reason="my new failure")
```
This skips `convo.absorb_turn`. The Conversation's `turns` log loses this turn, the prior-plan stash is not updated, the wire-sequence cursor is not lifted, and the **next** turn's `handle_turn` seeds against stale state — producing a plan-id churn or a leaked/empty prior plan.

**Correct:**
```python
return await self._abort_turn(
    session=session, convo=convo, user_input=user_input, pinned=pinned,
    reason="my new failure")
```
Always funnel pre-executor aborts through `_abort_turn`. It is the single sanctioned tail (#489).

### 8.2 Touching `absorb_turn` ordering

The five steps in `absorb_turn` (goals merge → results merge → sequence lift → stash → append `TurnRecord`) have data dependencies:
- The **sequence lift** must use `max(...)` — a naive assignment regresses to dropping turn N+1 events under the #161 outer-session pin (Gap 2).
- The **results merge** must be later-turns-win (`dict.update`), not first-wins — reversing it hides revised outputs from downstream turns.
- The **stash** must happen inside `absorb_turn` (and idempotently mirror the Runner's `finally`-block stash). If you move the stash out, the `CancelledError` path (which bypasses `absorb_turn`) and the normal path diverge.

Do not reorder or "simplify" these. If you must add a step, append it; do not interleave.

### 8.3 Assuming `resume()` resumes

Covered in §7.2. `resume` is replay-only (TODO #15). Do not build a "recover and continue" feature on top of it expecting live continuation. Construct a fresh Runner with goals recovered from the log.

### 8.4 Re-keying the Conversation map or lock on a churning id

The `_conversations` / `_convo_locks` / `_conversation_announced` dicts are keyed on the **outer-session id** (`""` for unpinned). Do **not** re-key them on `plan.id`, `goal.id`, or any LLM-minted/churning id — that violates the "lifecycle gates need stable identity keys" invariant and reopens the cross-session leak. The session id is stable for a session's lifetime; that is the whole point.

### 8.5 Mutating a `Plan` in place in Phase 3a / `_install_revision`

`Plan` is frozen (#247). Writing `session.plan.run_id = ...` or `session.plan.tasks.append(...)` will raise or corrupt the single-writer discipline. Always `dataclasses.replace(...)` to derive a stamped variant, and wrap plan installs in `channel_processor_active()` (Phase 3a does both). See 11-state-ownership.md.

### 8.6 Adding a second `observation_only` read to `runner.py`

The Runner is affected by the kill-switch in exactly one place: the F6 conversational wrap, whose gate lives inside `PromptShaper.wrap_conversational_input` (which calls `steering_is_active(steerer)`). Do **not** add `if observation_only:` checks in `runner.py`. The single sanctioned read is `DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)` in `goldfive/steerer.py` (#488). Missing/None/raising → PASSIVE.

### 8.7 Emitting `RunCompleted` from the Runner

`RunCompleted` (and terminal `RunAborted`) are **executor-owned** (`SequentialExecutor`/`ParallelDAGExecutor`). The Runner only emits pre-executor `RunAborted` (via `_abort_turn`) and the front-of-turn lifecycle events. If you emit `RunCompleted` from the Runner you get a duplicate terminal marker and break harmonograf's run-state machine.

### 8.8 Assuming a flat single agent

`_invoke_handle_turn` and the generate-fallback both prefer `agent.available_agents_tree` (a non-empty list, #151) and fall back to `agent.available_agents`. Do not hard-code a single-agent assumption — coordinator+AgentTool trees are a first-class supported shape (CANON invariant #3).

### 8.9 Passing `user_input` to an executor that doesn't accept it

Phase 7 inspects `self.executor.run`'s signature before passing `user_input=`. If you add code that unconditionally passes `user_input`, legacy/third-party executors whose `run` lacks that parameter will raise `TypeError`. Keep the `inspect.signature(...).parameters` guard.

### 8.10 Editing `conv.py` looking for conversation state

`conv.py` is proto converters. Conversation cross-turn state is in `conversation.py`. See §1.2.

### 8.11 Making `_detect_build_identity` able to raise

`__init__` calls it *before* any component is wired. It must never raise — the whole point is a best-effort "is this build deployed?" log line (`feedback_verify_running_build.md`). If you add a detection source, wrap it in `try/except` and default to `"unknown"`. A raising build-identity probe would make `Runner(...)` construction itself fail — catastrophic and unrelated to the caller's actual code.

### 8.12 Aliasing instead of copying in `next_turn_session`

`next_turn_session` does `goals=list(self.goals)`, `completed_results=dict(self.completed_results)` — **copies**. If you change these to pass the Conversation's own containers by reference, the executor's in-turn mutations (marking tasks complete, appending results) will retroactively rewrite the Conversation's record *before* `absorb_turn` runs, double-counting on the merge and corrupting `prior_turn_context`. Always copy when seeding a per-turn `Session`.

### 8.13 Calling `run` re-entrantly on the same key

`run` holds a per-key `asyncio.Lock` for the entire turn. Do **not** call `runner.run(session_id=X)` from *within* a component invoked by a turn already running on key `X` (e.g. a planner or tool that recursively drives the Runner) — it will deadlock waiting for the lock the current turn holds. Concurrent runs on *different* keys are fine. If you need nested driving, use a distinct `session_id` or a separate Runner.

### 8.14b Swapping the `control` channel mid-lifetime

The `control` setter raises `RuntimeError` if a *different* channel is already attached; same-identity re-attach is a no-op. Do not "fix" this to allow swapping — a control channel carries in-flight `ControlMessage` state and per-run wiring (the executor and steerer already hold references). Swapping it mid-run would strand pending controls. Construct a fresh Runner for a new channel.

### 8.14 Forgetting `pinned` on a new abort/stash path

`pinned` (`= bool(session_id)`) is threaded through `_abort_turn`, `stash_plan`, `absorb_turn`, and `prior_plan_for`. If you add a code path that stashes or absorbs and passes `pinned=False` unconditionally, you break the carry-forward matrix (§6.3): a pinned turn's plan will be treated as unpinned and may carry forward into an unrelated session, or fail to carry forward within the same session. Always thread the turn's real `pinned` value.

---

## 9. Recipes (safe edits to this subsystem)

### 9.1 Add a new construction knob

1. Add the keyword to `Runner.__init__(self, *, ..., my_knob: T = default)` — **keyword-only**, with a default. Never a positional.
2. Document it in the class docstring's Parameters block (the docstring is the primary API reference; keep it accurate).
3. Store it on `self._my_knob` (private) unless callers must read it back.
4. If it should be env-overridable, follow the `fail_fast_on_revision_rejection` pattern: `None` kwarg → consult env; explicit `True`/`False` wins over env.
5. If it toggles a steerer/adapter capability, probe with `hasattr`/`getattr` so custom components without the hook still construct.
6. Do NOT add it to `**legacy_kwargs` handling unless it is a *rename* of an old knob (then follow the `max_plan_reinvocations` deprecation-warning pattern).
7. Add a test in `tests/test_runner_extension.py` (construction) or a dedicated `tests/test_runner_<knob>.py`.

### 9.2 Add a new front-of-turn lifecycle event

1. Add a factory in `goldfive/events.py` (see 12-events-sinks-telemetry.md) taking `run_id`, `sequence`, `session_id`.
2. Add a private `_emit_<name>` on the Runner that calls `session.next_sequence()` for the sequence and `session.id` for `session_id`, then `await emit(self.sinks, evt)`. **Always allocate the sequence via `session.next_sequence()`** — never a literal — so the cursor stays monotonic (§6.4).
3. Call it from the correct phase in `_run_locked`. Front-of-turn events (before the executor handoff) are Runner-owned; terminal events are executor-owned (do not add `RunCompleted`-like events to the Runner — §8.7).
4. If it is a per-conversation event (like `ConversationStarted`), gate it on the per-key `_conversation_announced` flag so it fires once per key.

### 9.3 Add a cross-turn Conversation field

1. Add the field to the `Conversation` dataclass in `conversation.py` with a `dataclasses.field(default_factory=...)` for mutable defaults.
2. If it must be visible to the *next* turn's `Session`, copy it (not alias) in `next_turn_session()` — e.g. `dict(self.my_map)`.
3. Fold it back in `absorb_turn` with explicit merge semantics (later-turns-win = `dict.update`; append-dedup = the goal pattern). **Append your merge; do not interleave it into the existing five ordered steps** (§8.2).
4. If the planner should see it, surface it in `prior_turn_context`.
5. Add a round-trip test in `tests/test_conversation.py`.

### 9.4 Add a new pre-executor abort condition

1. Detect the failure in the appropriate phase.
2. `return await self._abort_turn(session=session, convo=convo, user_input=user_input, pinned=pinned, reason="<specific reason>")`. Nothing else — no hand-rolled emit/outcome/absorb (§8.1).
3. If the failure is an exception you caught, include `{exc}` in the reason and `log.exception(...)` at the call site (the two per-site deltas #489 kept local).
4. Update the "eight call sites" table in §5.1 and the grep-guard count in §10.3 if you genuinely add a *new* site.

### 9.5 Make a custom Executor overlay-aware

If your executor wants the raw user request (for `invoke_passthrough`), accept `user_input` in its `run` signature. The Runner passes it only when `inspect.signature(self.executor.run)` has a `user_input` parameter (§8.9). No registration needed — the inspection is automatic.

---

## 10. Verification checklist

Run these after touching `runner.py` or `conversation.py`. All commands assume the repo root and the dev+adk env (`uv sync --extra dev --extra adk`).

### 10.1 Targeted test files

```bash
# Runner core, extension API, integration, multi-turn:
uv run pytest -q tests/test_runner.py tests/test_runner_extension.py \
  tests/test_runner_integration.py tests/test_runner_multi_turn.py

# Cross-turn Conversation semantics + the carry-forward matrix + isolation:
uv run pytest -q tests/test_conversation.py \
  tests/test_intra_session_plan_carry_forward.py \
  tests/test_runner_conversation_isolation.py \
  tests/test_runner_cross_session_isolation.py

# Streaming, revision-rejection policy, close-time orphan audit, build log, degrade:
uv run pytest -q tests/test_runner_streamed.py \
  tests/test_runner_revision_rejection_policy.py \
  tests/test_runner_close_orphan_audit.py \
  tests/test_runner_install_revision_stall.py \
  tests/test_runner_build_identity_log.py \
  tests/test_degrade_prebuilt_runner.py
```

### 10.2 Full suite + lint

```bash
uv run pytest -q          # ~30s, expect ~2912 passed / 61 skipped
ruff check .              # MUST stay clean; do NOT ruff-format (repo is not format-clean)
```

### 10.3 Grep guards for the invariants

```bash
# There must be exactly 8 _abort_turn call sites + 1 definition (= 9 matches):
grep -n "_abort_turn" goldfive/runner.py | wc -l    # expect 9

# The Runner must NOT *read* the kill-switch directly. The only matches must be
# comments/docstrings — there must be NO code that reads it:
grep -nE "self\._observation_only|steering_config\.observation_only|if .*\bobservation_only\b" goldfive/runner.py   # expect NOTHING
# (a plain `grep observation_only goldfive/runner.py` returns 3 comment-only lines — expected, not a violation)

# Every plan install in runner.py must be under channel_processor_active:
grep -n "set_session_plan\|channel_processor_active" goldfive/runner.py

# The stash appears at exactly two sanctioned sites (finally block + absorb_turn):
grep -n "stash_plan" goldfive/runner.py goldfive/conversation.py

# Confirm RunCompleted is emitted only by executors, never runner.py:
grep -rn "run_completed_event" goldfive/runner.py    # expect NOTHING
grep -rn "run_completed_event" goldfive/executors/   # expect the terminal emitters
```

### 10.4 Behavioural spot-checks after a change

- **Multi-turn goal accumulation**: `test_runner_multi_turn.py` — a second turn's goals must append, not replace, and not leak across a `new_conversation`.
- **Cross-session isolation**: `test_runner_cross_session_isolation.py` — two `run(session_id=...)` calls with different ids must not share plan/goals.
- **Concurrent same-session serialization**: `test_intra_session_plan_carry_forward.py` — the v7class1-1 timeline; turn 2 must see turn 1's post-install plan.
- **Abort tail correctness**: `test_runner_revision_rejection_policy.py` — default keeps plan + emits `HUMAN_INTERVENTION_REQUIRED`; strict aborts.
- **Streaming teardown**: `test_runner_streamed.py` — early `aclose()` must still run the driver's `finally` (stash lands).

If your change is behavioural (not docs/tests-only), also drive an end-to-end run — see 15-testing-guide.md for the layered-validation protocol; a green `pytest` on narrow criteria has historically passed on broken flows (`feedback_validation_criteria_too_narrow.md`).

---

## 11. Where to go next

- The plan the Runner installs is walked by the executor: **04-executors-and-control.md** (terminal `RunCompleted`/`RunAborted`, the `_run_overlay` stage methods — the other half of #489, the control loop that consumes the `ControlChannel` bound in Phase 6d).
- `handle_turn` / `generate` internals and `PlanReviser.install_initial_plan` / `install_revision_for_drift`: **10-planning-and-revision.md**.
- The `observation_only` kill-switch, `DefaultSteerer.bind`, and the `is_active_steering` / `steering_is_active` predicate the F6 wrap gates on: **09-steering-ladder-and-gates.md**.
- `Session`, `_ostate` (`state_store`), `channel_processor_active`, and the frozen-`Plan` single-writer discipline: **11-state-ownership.md**.
- The `goldfive.events` factories (`run_started_event`, `goal_derived_event`, `conversation_started_event`, …) and `emit`: **12-events-sinks-telemetry.md**.
- `select_reporting_tools` and the `report_awaiting_approval` lifecycle: **13-reporting-tools-and-approval.md**.

If in doubt about any claim in this chapter, the **code on main wins over any doc** — including the `runner.py` module docstring's stale `PlanSubmitted` line (§1.1) and any design doc under `docs/design/`. Re-verify citations against the live repo before editing.
