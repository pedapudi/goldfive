# 05. The ADK Plugin

## Read this chapter when...

- You are editing `goldfive/adapters/_adk_plugin.py` — the single largest and most hazardous file in the repo (~7020 lines).
- You need to add, remove, or change an ADK callback (`before_run`, `before_agent`, `before_model`, `before_tool`, `after_model`, `after_agent`, `after_run`, `on_event`, `on_tool_error`, `after_tool`).
- You are touching the delegation-pin machinery (how a delegated sub-agent learns which plan task it is enacting) or the reporting-tool `task_id` injection path.
- You are working on cooperative cancellation, the per-LLM-call timeout watcher, or the stall watchdog — anything that spawns an `asyncio.Task`.
- You need to understand why the plugin duck-types nearly every ADK attribute, and what rules govern touching those reaches.
- You saw a symptom (reporting tools no-op, drift misattributed to the coordinator, a watcher firing twice, state "not propagating") and suspect the plugin.

This chapter is the deep reference for the plugin. It assumes you have read `02-architecture-map.md` (where the plugin sits) and pairs tightly with `06-adapters-and-instrumentation.md` (the `ADKAdapter` that owns and drives the plugin, plus the `adk_llm_instrumentation` helpers), `09-steering-ladder-and-gates.md` (the steerer surfaces the plugin feeds and gates against), and `11-state-ownership.md` (the `StateStore` the pins land on).

## Files covered

| File | Role |
|------|------|
| `goldfive/adapters/_adk_plugin.py` | **This chapter.** The `BasePlugin` subclass and all its module-level helpers. |
| `goldfive/adapters/adk_llm_instrumentation.py` | Request-side `LlmRequest` mutation + measurement helpers (`_measure_request_chars`, `_apply_agent_max_output_tokens_cap`, `_build_runtime_tools_hint`). Re-imported into `_adk_plugin` at module load so historical `from goldfive.adapters._adk_plugin import _measure_request_chars` callsites keep resolving. |
| `goldfive/adapters/_adk_state_protocol.py` | The `goldfive.*` state-key names + `descendants_of_invocation` walker used by cancel propagation. |
| `goldfive/adapters/_tool_invocation.py` | `invoke_tool` — the dispatch path reporting-tool calls route through. |
| `goldfive/prompt_shaper.py` | `PromptShaper` — owns the `before_model_callback` prompt injections and their `observation_only` gate. |
| `goldfive/state_store.py` | `StateStore` — owns the pin storage, pending-delegation map, and invocation-task registry the plugin writes through. |
| `goldfive/steerer.py` | `steering_is_active` / `DefaultSteerer.is_active_steering` — the one kill-switch predicate `_is_observation_only` delegates to. |

The **code on main is the ground truth.** Where a design doc (`docs/design/SHARED-RUNNER-REFACTOR.md`, `TASK-LIFECYCLE.md`, `APPROVAL.md`) disagrees with the file, the file wins — several of those docs describe deferred alternatives that are NOT on main (see "The dual-Runner split" below).

## Invariants that bind you here

These are the CANON hard invariants, specialized to this file. Every edit you make must preserve all six.

1. **No prompt-cooperation contracts.** Every observation, pin, cancellation, and drift emission must work even if the wrapped agent never calls a goldfive tool and never follows an instruction. The plugin observes ADK callbacks; it never *requires* the agent to cooperate. The reporting-tool interception is an *opportunistic* enrichment, not a dependency.
2. **No regex/keyword NL classification.** The delegation-pin tier-2 disambiguator uses `stem_token_match` (bi-directional substring against a small fixed role-suffix list), NOT regex. Do not reintroduce `_GENERIC_VERB_PREFIX_RE`-style heuristics. Exact-equality and hash matching of *structured* data (task ids, `function_call_id`, `(name, args_hash)` tuples) IS allowed and is used throughout.
3. **Any ADK tree shape must work,** including coordinator + `AgentTool`. The plugin has no notion of "coordinator" vs "sub-agent"; every invocation level is treated the same. `_is_agent_tool_dispatch` is the only place tree shape is inspected, and it degrades safely.
4. **Adaptive over predictive.** Capture observed facts (the `tool_args` the agent actually authored, the delegation the agent actually made) via extended protos / events; do not intercept at pin time to predict behavior. `DelegationObserved` carries `tool_args_json` off the observed event (`_safe_jsonify_tool_args`), NOT a goldfive-side guess. See `docs/design` §13 "adaptive, not predictive".
5. **`observation_only=True` is STRICTLY passive** (production default). The ONLY sanctioned read of the kill-switch inside this file is `_is_observation_only(ctx)`, which delegates to `steering_is_active(ctx.steerer)`. Missing / `None` / raising steerer → PASSIVE. Every intervention surface (cancel-flag write, F3 redirect, ContextEditor) MUST be a complete no-op under passive mode; telemetry still fires.
6. **Lifecycle gates need stable identity keys.** Never key a gate, dedup, or pin on an LLM-minted or churning id. The plugin keys on ADK `invocation_id`, `function_call_id`, `session.run_id`, and plan task ids — all structurally stable. The pin ladder deliberately does NOT key on anything the model invented.

---

## 1. What the plugin is, and how it is built

`goldfive.wrap(adk_tree)` ultimately constructs an `ADKAdapter` (see `06-adapters-and-instrumentation.md`), and that adapter's `__init__` calls `make_adk_plugin(...)` to build exactly one plugin instance, which it registers on ADK's `plugin_manager`. The plugin is the routing layer between ADK's callback lifecycle and goldfive's `Steerer`.

The module docstring names its four jobs verbatim:

```python:goldfive/adapters/_adk_plugin.py
1. **State protocol** — ``before_model_callback`` writes the current
   task and plan context into the ADK session state ...
2. **Reporting-tool interception** — ``before_tool_callback`` watches
   for the eight canonical reporting tools ...
3. **Tool confirmation bridge** — the same ``before_tool_callback``
   intercepts any ADK tool flagged ``require_confirmation=True`` ...
4. **Drift observation** — ``after_model_callback``,
   ``on_event_callback`` ... and ``on_tool_error_callback`` feed raw
   signals into ``steerer.drift.observe(...)`` ...
```

Plus a fifth, flag-gated job the docstring adds: the **wall-clock stall watchdog** (§7.2).

### 1.0 What the plugin deliberately does NOT do

Knowing the plugin's boundaries prevents whole classes of misplaced edits:

- **It does not classify natural language.** All NL judgment lives in the steerer's LLM judges (`08-llm-judges.md`). The plugin feeds raw observations (`steerer.drift.observe`) and structural drifts (`handle_drift`); it never decides "is this off-topic" itself.
- **It does not revise plans.** Plan revision is the `PlanReviser` / steerer's job. The plugin's `_maybe_pin_delegation_task` assignee stamp is *observational* (no `PlanRevised`), and descriptive growth calls out to `PlanReviser.install_descriptive_growth` — it does not build the revision inline.
- **It does not run the intervention ladder.** It flags cooperative cancel and routes drifts through `handle_drift`; the steerer decides nudge/steer/cancel/pause/terminate (`09-steering-ladder-and-gates.md`).
- **It does not call LLM judges directly.** The one exception is spawning the idle goal-drift judge via the stall watchdog's `_spawn_goal_drift_judge_background` hook — and that is a spawn-and-detach through the steerer's own tracked machinery, not a direct judge call.
- **It does not own state.** Pins, cancel flags, and the task registry live on `StateStore` / the plugin instance as a *cache/bridge*; the goldfive `Session` is the typed source of truth (`11-state-ownership.md`).

If your edit wants to do one of these things *inside* a callback, it almost certainly belongs in the steerer or `PlanReviser` instead — the plugin's job is to observe ADK faithfully and route.

### 1.1 Lazy ADK import — the factory pattern

`_adk_plugin.py` **never imports `google.adk` at module load.** It imports `BasePlugin` lazily inside `make_adk_plugin`, and imports concrete ADK types (`AgentTool`, `LlmResponse`, `genai.types`) inside the functions that need them, each wrapped in `try/except`. This is deliberate and load-bearing:

- The module must be importable from non-ADK code for type-checking and for unit tests that patch the base class with a stub when ADK is not installed.
- `make_adk_plugin` raises a clean `ImportError("goldfive.adapters.adk requires 'pip install goldfive[adk]'")` when ADK is absent (see the top of the factory).

```python:goldfive/adapters/_adk_plugin.py
def make_adk_plugin(
    *,
    name: str = "goldfive_adk_plugin",
    host_agent_name: str = "",
    agent_tool_cap: int = 16,
    llm_call_timeout_ms: int = DEFAULT_LLM_CALL_TIMEOUT_MS,
    agent_max_output_tokens: int = DEFAULT_AGENT_MAX_OUTPUT_TOKENS,
    context_editor: Any = None,
) -> Any:
```

**DON'T** add a top-level `import google.adk` or `from google.adk... import` anywhere in this module. It breaks every non-ADK import path and ~2900 tests that import the module without the `adk` extra installed. If you need an ADK type, import it lazily inside the function under `try/except`, exactly like `_is_agent_tool_dispatch` and `_make_cancelled_llm_response` do.

### 1.2 The class is closure-scoped

`make_adk_plugin` returns `_state_audit.wrap_plugin_callbacks(_GoldfiveADKPlugin())` — the class `_GoldfiveADKPlugin` is defined *inside* the factory so it can close over `name`, `host_agent_name`, `agent_tool_cap`, `llm_call_timeout_ms`, `agent_max_output_tokens`, and `context_editor`. These are per-wrap configuration; they are captured once at construction and read from the closure, not re-read per callback.

The final line wraps the instance in `_state_audit.wrap_plugin_callbacks(...)` (goldfive#271 Phase 0). This structurally sets one `ContextVar` per callback entry/exit so the state-ownership tripwire can recognize "writes from inside a goldfive callback". The wrap runs unconditionally; the *check* only fires under `GOLDFIVE_STRICT_STATE_OWNERSHIP` / the test fixture. See `11-state-ownership.md` and `docs/design/STATE-OWNERSHIP-CONTRACT.md` §7.

**Consequence for you:** you cannot `isinstance`-check against `_GoldfiveADKPlugin` from outside the factory (it is closure-local). The plugin instead carries a **class-level marker** `__goldfive_adk_plugin__ = True` so external walkers can find it. This marker is the discriminator `session_context_from_invocation` uses (§2.2). Do not remove it.

### 1.3 The factory kwargs (and their kill-switches)

| Kwarg | Default | Meaning | Disable |
|-------|---------|---------|---------|
| `host_agent_name` | `""` | Fallback agent name rendered into observations + `available_tasks` when a task has no explicit assignee. Usually the wrapped root agent's name. | n/a |
| `agent_tool_cap` | `16` | Max `AgentTool` spawns per top-level invocation before `RUNAWAY_DELEGATION` fires (goldfive#130). | set `0` or negative |
| `llm_call_timeout_ms` | `DEFAULT_LLM_CALL_TIMEOUT_MS` = `1_800_000` (30 min) | Per-LLM-call wall-clock budget; on exceed emits CRITICAL `LLM_CALL_TIMEOUT` + flags cancel (§7.1). `goldfive.wrap` threads a tighter 120s via `AgentConfig`. | set `0` or negative |
| `agent_max_output_tokens` | `DEFAULT_AGENT_MAX_OUTPUT_TOKENS` = `16384` | Structural `max_output_tokens` ceiling; ratchets DOWN only (smaller-wins) in `before_model_callback` (goldfive#256). | set `0` or negative |
| `context_editor` | `None` | Request-side `ContextEditor` (goldfive#397). `None` when no rules configured → the `before_model` path short-circuits on `is None` for zero overhead. | pass `None` |

Every one of these has a documented off switch. When you add a new intervention knob, follow this pattern: a numeric-or-`None` default, a cheap short-circuit when disabled, and a docstring paragraph in `make_adk_plugin`.

---

## 2. SessionContext and how callbacks reach goldfive state

Every callback needs three things: the goldfive `Session`, the `Steerer`, and the reporting-tool spec list. All three live on a `SessionContext`.

```python:goldfive/adapters/_adk_plugin.py
class SessionContext:
    __slots__ = (
        "session",
        "steerer",
        "task",
        "tool_handlers",
        "tools",
        "host_agent_name",
    )
```

- `session` — the goldfive `Session` (typed orchestration state; NOT the ADK session).
- `steerer` — the `Steerer` (may be `None` for judge-only / passive stubs). Every kill-switch read goes through this.
- `task` — the task the adapter was invoked with. **`None` on the typical orchestration-only coordinator turn** (`invoke_passthrough`), which is why so many callbacks fall back from `ctx.task.id` to `session.current_task_id`.
- `tools` — the authoritative `ReportingToolSpec` list `before_tool_callback` routes through `invoke_tool`. `tool_handlers` is a legacy name→handler map kept for external callers; when only handlers are supplied, `_tools_from_handlers` synthesizes minimal specs so the dispatch still flows through `invoke_tool` and picks up the terminal-task rejection / idempotency / loop-guard layers.

### 2.1 The dual-Runner split — WHY the resolution is a tree-walk, not a state read

This is the single most important architectural fact about the plugin. **Read it before you "simplify" any context-resolution code.**

`goldfive.wrap(adk_tree)` returns a `GoldfiveADKAgent` whose `_run_async_impl` spins up goldfive's own `Runner`, which holds an `ADKAdapter` that owns a **separate** `InMemoryRunner` around the inner tree. Under `adk web`, two ADK runners exist at once, each with its own `InvocationContext` / `Session`. (This is documented in `docs/design/SHARED-RUNNER-REFACTOR.md` §1 — but note that doc describes a *proposed, unshipped* alternative "7(c)" to *eliminate* the split; the split itself is the current reality on main. Do not treat SHARED-RUNNER-REFACTOR.md as describing current behavior.)

The consequence that bites: ADK's `InMemorySessionService.get_session` returns a **shallow copy** of the stored session (`copy.copy(session.state)`) on every call. So a `SessionContext` written into the adapter's own `get_session` copy **never reaches** the fresh copy that `runner.run_async` materializes for the invocation. A callback reading ADK `session.state` for goldfive keys would see an empty dict and silently fall through to the ACK shim.

Everything that looks like goldfive scaffolding here — the `_active_ctx` field, the plugin-manager tree-walk, the `StateStore`-backed pin storage — exists to repair this split. **The fix is: do not rely on ADK `session.state` as a cross-callback channel.** Two resolution paths exist:

1. **Live-run path (authoritative):** the adapter calls `plugin.set_active_context(ctx)` before `runner.run_async`, stashing `ctx` on the plugin instance field `_active_ctx`. Callbacks read it via `self._resolve_ctx(adk_ctx)`, which prefers `_active_ctx`.
2. **Unit-test path:** tests that drive callbacks directly with a hand-built state dict stash a `SessionContext` under `SESSION_CONTEXT_STATE_KEY = "goldfive._session_context"`; `_session_context_from_callback` reads it back. This path is authoritative *only* for those synthetic harnesses.

```python:goldfive/adapters/_adk_plugin.py
        def _resolve_ctx(self, adk_ctx: Any) -> SessionContext | None:
            if self._active_ctx is not None:
                return self._active_ctx
            return _session_context_from_callback(adk_ctx)
```

**The two paths are never both populated in production.** Do not "merge" them or read state first — the state read is the unreliable channel the whole design routes around.

The `_active_ctx` field comment in `__init__` states the rationale precisely — worth reading verbatim because it is the load-bearing "why":

```python:goldfive/adapters/_adk_plugin.py
            # Callbacks prefer this field over the ADK-state lookup because ADK's
            # InMemorySessionService returns a **shallow copy** of the stored
            # session on every ``get_session`` call (see ``_light_copy`` /
            # ``copy.copy(session.state)``) — so a SessionContext written into the
            # adapter's own ``get_session`` copy never reaches the fresh copy that
            # ``runner.run_async`` materialises for the invocation, and the
            # callbacks would see an empty state and silently fall through to the
            # ACK shim.
            self._active_ctx: SessionContext | None = None
```

The same shallow-copy fact drives THREE separate decisions in this file: `_active_ctx` (context), `_cancel_state` (cancel flags), and the `StateStore`-backed pins/registry. All three are on the plugin instance or `StateStore`, never on ADK `session.state`, for exactly this reason. When you find yourself wanting to stash something on ADK state to read it back in a later callback, stop — use the plugin instance or `StateStore`.

### 2.2 session_context_from_invocation — the tree-walk

Code that has an ADK `InvocationContext` but not a `callback_context` (the dynamic-instruction resolver, the planner's per-turn injection, `_goldfive_session_from_tool_context`) reaches the live session via a plugin-manager tree-walk:

```python:goldfive/adapters/_adk_plugin.py
def session_context_from_invocation(invocation_context: Any) -> SessionContext | None:
    ...
    plugins = getattr(plugin_manager, "plugins", None) or ()
    for plugin in plugins:
        if not getattr(plugin, "__goldfive_adk_plugin__", False):
            continue
        ctx = getattr(plugin, "_active_ctx", None)
        if ctx is not None:
            return ctx
    return None
```

This is why the `__goldfive_adk_plugin__` marker must stay on the class and `_active_ctx` must be set before any callback fires. It returns `None` cleanly for out-of-band invocations and unit tests that drive callbacks directly — every caller degrades gracefully on `None`.

---

## 3. The plugin instance state map

`_GoldfiveADKPlugin.__init__` sets ~25 instance fields. You will touch these constantly, so here is the full map with lifetime and consumer. **Lifetime matters most:** a field cleared in `clear_active_context` is per-dispatch; a field NOT cleared there is plugin-lifetime (survives across sequential invocations on the same wrapped tree).

| Field | Type | Lifetime | Purpose / consumer |
|-------|------|----------|--------------------|
| `_active_ctx` | `SessionContext \| None` | per-dispatch (set by `set_active_context`, cleared last in `clear_active_context`) | The live context all callbacks resolve through. |
| `_reconciler` | `PlanReconciler \| None` | per-dispatch (overlay path only) | Attached by `invoke_passthrough`; forwarded before/after-agent + delegation observations. `None` on the per-task `invoke(task)` path. |
| `_top_invocation_id` | `str` | per-dispatch | Pinned on the first `before_run`; lets nested AgentTool sub-Runners attribute themselves a `parent_invocation_id`. |
| `_invocation_tool_calls` | `dict[str,int]` | per-dispatch | Cumulative tool-call count per invocation → CONFABULATION_RISK. |
| `_invocation_last_text` | `dict[str,str]` | per-dispatch | Last non-empty text per invocation → CONFABULATION_RISK + after-agent summary. |
| `_no_reasoning_streak` | `dict[str,int]` | **plugin-lifetime** | Consecutive empty-reasoning turns per agent (§8). |
| `_no_reasoning_warned` | `set[str]` | **plugin-lifetime** | Agents already warned — keeps the disarm WARNING one-shot across dispatches. |
| `_agent_tool_spawn_count` | `int` | per-dispatch (also reset in `set_active_context`) | Runaway-delegation counter (goldfive#130). |
| `runaway_delegation_tripped` | `bool` | per-dispatch | One-shot cap-tripped flag; the adapter's invoke loop reads it to break out. |
| `_invocation_llm_pending` | `dict[str,dict]` | per-dispatch | Per-LLM-call start time + chars + `watcher` task handle. |
| `_stall_watchdog_task` | `asyncio.Task \| None` | per-dispatch (spawned in `set_active_context`, cancelled in `clear`) | The flag-gated stall watchdog (§7.2). |
| `_tool_loop_tracker` | `ToolLoopTracker` | plugin-lifetime object, `.clear()`ed per-dispatch | Deterministic tool-loop detection (goldfive#181). |
| `_tool_loop_invocation_stats` | `dict[str,dict]` | per-dispatch | Negative-class aggregation for tool-loop `no_drift` decisions. |
| `_progress_reporting_tools` | `frozenset[str]` | constant | The six reporting tools whose acknowledged-success resets the loop window. |
| `_cancel_state` | `dict[str,CancellationRequest]` | per-dispatch | Authoritative cancel-requested flags keyed by `invocation_id` (§6). |
| `_cancelled_invocations` | `set[str]` | per-dispatch | Sticky-cancelled set — makes cancellation idempotent across follow-up callbacks. |
| `_invocation_parents` | `dict[str,str]` | per-dispatch | `invocation_id → parent_invocation_id`, for cancel propagation. |
| `_invocation_pinned_task_id` | `dict[str,str]` | per-dispatch | Per-invocation pin so a child reads its parent's pin (pin signal 5). |
| `_invocation_tasks` | `_InvocationTaskRegistryView` | view over `StateStore` | Legacy `dict`-shaped view; storage lives on `StateStore` (§6.4). |
| `_boundary_entered_invocations` | `set[str]` | per-dispatch | Tracks invocations that emitted `InvocationBoundaryEntered` and owe an `Exited`. |
| `_prompt_shaper` | `PromptShaper` | plugin-lifetime | Owns the `before_model` injections + their observation-only gate. |
| `_context_editor` | closure kwarg | plugin-lifetime | Request-side `ContextEditor` (goldfive#397). |

**When you add a per-invocation field, you MUST clear it in `clear_active_context`.** Every dict/set above that is per-dispatch is explicitly cleared there. Forgetting means state from a finished dispatch leaks into the next dispatch on the same reused adapter (sequential `invoke` calls reuse the plugin instance). The `__init__` comments repeatedly flag this ("reset per invocation so nested AgentTool sub-Runners get their own counters").

**When you deliberately want plugin-lifetime state** (like `_no_reasoning_warned`), leave it out of `clear_active_context` and document why in the `__init__` comment, exactly as that field does.

---

## 4. Lifecycle: set_active_context / clear_active_context

These are the plugin's *own* lifecycle hooks (not ADK callbacks). The adapter calls them around every `runner.run_async`.

### 4.1 set_active_context

```python:goldfive/adapters/_adk_plugin.py
        def set_active_context(self, ctx: SessionContext) -> None:
            self._active_ctx = ctx
            self._agent_tool_spawn_count = 0
            self.runaway_delegation_tripped = False
            self._maybe_start_stall_watchdog(ctx)
```

Three jobs: attach the context, reset the runaway-delegation bookkeeping (so a prior trip doesn't leak), and spawn the flag-gated stall watchdog. Overwriting a non-`None` value is fine — sequential invocations reuse the adapter.

### 4.2 clear_active_context — teardown order is load-bearing

Called from the adapter's `finally`. The ordering here is subtle and you must not reorder it:

1. **First:** `self._invocation_tasks.clear()` — this must run BEFORE `_active_ctx = None`, because the registry view resolves the `StateStore` *through* `_active_ctx.session`. Once `_active_ctx` is `None`, the view no-ops and the StateStore-side bucket would leak across dispatches.
2. **Then:** `_cancel_stall_watchdog()` — cancel BEFORE dropping the context so a watchdog waking mid-teardown cannot observe a half-cleared plugin.
3. **Then:** `_active_ctx = None`, and clear every per-dispatch dict/set (see §3 table), plus cancel any straggling per-LLM-call watchers in `_invocation_llm_pending`.

```python:goldfive/adapters/_adk_plugin.py
            try:
                self._invocation_tasks.clear()
            except Exception as exc:  # noqa: BLE001
                log.debug("clear_active_context: registry clear raised: %s", exc)
            self._cancel_stall_watchdog()
            self._active_ctx = None
            ...
            for pending in self._invocation_llm_pending.values():
                watcher = pending.get("watcher") if isinstance(pending, dict) else None
                if watcher is not None and not watcher.done():
                    watcher.cancel()
            self._invocation_llm_pending.clear()
```

**DON'T** move the `_invocation_tasks.clear()` below `_active_ctx = None`. **DON'T** skip the watcher-cancel loop — an un-cancelled per-call watcher becomes an orphan task that fires an `LLM_CALL_TIMEOUT` against a dead dispatch.

The complete per-dispatch reset checklist `clear_active_context` performs (in order): `_invocation_tasks.clear()` → `_cancel_stall_watchdog()` → `_active_ctx = None` → `_top_invocation_id = ""` → `_agent_tool_spawn_count = 0` → `runaway_delegation_tripped = False` → `_reconciler = None` → cancel + clear `_invocation_llm_pending` → `_tool_loop_tracker.clear()` → `_tool_loop_invocation_stats.clear()` → `_cancel_state.clear()` → `_cancelled_invocations.clear()` → `_invocation_parents.clear()` → `_invocation_pinned_task_id.clear()` → `_boundary_entered_invocations.clear()`. Note what is NOT cleared: `_no_reasoning_streak`, `_no_reasoning_warned` (plugin-lifetime, §3). Anything left in `_boundary_entered_invocations` at this point means a boundary's exit never fired (a bug or test scaffolding); the plugin does NOT emit a synthetic exit here (sinks may be unreachable mid-clear) — the operator-visible signal is the missing pair on the wire.

---

## 5. The callbacks, one at a time

ADK invokes each `*_callback` at a defined lifecycle point. Below, "observes" = reads and feeds the steerer; "mutates" = writes state / short-circuits / injects. Every callback resolves `ctx` first and returns `None` if unbound.

Two return-value contracts you must respect:
- `before_model_callback` returning **non-`None`** short-circuits the LLM dispatch (ADK propagates the return as the response). Returning `None` lets the request proceed. This is why cancellation returns a synthetic `_make_cancelled_llm_response()`, NOT `None` (§6.2).
- `before_tool_callback` returning a **dict** short-circuits the tool dispatch (ADK treats the dict as the tool's response to the model). Returning `None` lets the tool run.

### 5.1 before_run_callback — state seed + invocation registration

Fires once per runner invocation (top-level and per-AgentTool sub-Runner, each with its own session). This is the **reliability-critical** write path: `invocation_context.session` is the session ADK actually streams against, so writes here are visible to every subsequent callback.

What it does, in order:
1. Resolve `ctx`; bail on `None`.
2. Compute `inv_id`. If `_top_invocation_id` is already set, this is a nested sub-Runner and `parent_inv_id = _top_invocation_id`; otherwise pin `_top_invocation_id = inv_id`.
3. Record `_invocation_parents[inv_id] = parent_inv_id` for cancel propagation.
4. **Register the driving asyncio.Task:** `self._invocation_tasks[inv_id] = asyncio.current_task()`. ADK runs `before_run_callback` in the same task as the dispatch, so this IS the task whose cancellation raises `CancelledError` inside the adapter's `async for event in runner.run_async(...)`. `request_invocation_cancel` consults this to fire `task.cancel()` on a supersede.
5. **Cooperative-cancel short-circuit:** if `is_invocation_cancelled(inv_id)`, consume the request, emit `InvocationCancelled`, and return without seeding state or emitting `AgentInvocationStarted`.
6. Reset `_invocation_tool_calls[inv_id] = 0` and `_invocation_last_text[inv_id] = ""`.
7. Emit `AgentInvocationStarted` (best-effort) and call `steerer.drift.note_agent_activity` (duck-typed; feeds the GOAL_DRIFT judge trajectory buffer).

Note the docstring: "V1 (initial seed) and V2 (orchestration-state bridge) both deleted" — the plugin no longer writes goldfive keys onto ADK state here; the planner reads the goldfive `Session` directly via the stash (Phase 2.0 of goldfive#271).

The task-registration + parent-mapping core, verbatim:

```python:goldfive/adapters/_adk_plugin.py
            if inv_id and parent_inv_id:
                self._invocation_parents[inv_id] = parent_inv_id
            if inv_id:
                current = asyncio.current_task()
                if current is not None:
                    self._invocation_tasks[inv_id] = current
            if inv_id and self.is_invocation_cancelled(inv_id):
                pending = self._cancel_state.get(inv_id)
                if pending is not None:
                    request = self.consume_cancel_for_invocation(inv_id)
                    self._cancelled_invocations.add(inv_id)
                    await self._emit_invocation_cancelled(
                        invocation_id=inv_id, agent_name="", request=request,
                    )
                return None
```

`asyncio.current_task()` here IS the task the adapter's `async for event in runner.run_async(...)` iterates — ADK runs `before_run_callback` in that same task, so cancelling it raises `CancelledError` inside the adapter loop. This registration is what makes `cancel_inflight_task=True` (§6.3) able to abort a wedged LLM stream.

### 5.2 before_agent_callback — task-id pinning + boundary entry

Fires once per agent invocation (including sub-agents inside AgentTool sub-Runners). Jobs, in strict order:

1. Compute `agent_name`, `inv_id`, `parent_inv_id`.
2. **Boundary entry:** `_emit_boundary_entered(...)`. Done BEFORE the cancel short-circuit so a cancel flagged before the boundary still produces the entry/exit pair.
3. **Cooperative-cancel checkpoint:** if cancelled, consume, emit `InvocationCancelled`, emit `_emit_boundary_exited(reason="cancelled")`, and return — leaving no side-effects on orchestration state.
4. **Pin the reasoning agent:** `gf_session.current_agent_id = agent_name` (last-writer-wins; the reasoning-drift judge attributes reasoning to the agent that produced it, not the static plan assignee — a coordinator delegating via AgentTool would otherwise mis-attribute every child drift to the coordinator).
5. **Layer-1 task-id pin:** `await self._pin_current_task_id_for_agent(...)` (the 8-signal ladder, §6pin). Wrapped so a raise never breaks the invocation.
6. **Overlay forward:** if a reconciler is attached, `await reconciler.on_before_agent(...)` with a `TypeError` fallback to the pre-#151 signature.

The ordering is load-bearing: boundary-entry BEFORE the cancel checkpoint (so a cancel flagged before the boundary still produces an entry/exit pair), and the cancel checkpoint BEFORE pinning + reconciler work (so a cancelled turn leaves NO side-effects on orchestration state). The cancel checkpoint's exit emit:

```python:goldfive/adapters/_adk_plugin.py
            if inv_id and self.is_invocation_cancelled(inv_id):
                pending = self._cancel_state.get(inv_id)
                if pending is not None:
                    request = self.consume_cancel_for_invocation(inv_id)
                    self._cancelled_invocations.add(inv_id)
                    await self._emit_invocation_cancelled(
                        invocation_id=inv_id, agent_name=agent_name, request=request,
                    )
                if inv_id:
                    await self._emit_boundary_exited(
                        invocation_id=inv_id, agent_name=agent_name,
                        task_id=ctx_task_id, reason="cancelled",
                    )
                return None
```

The `TypeError`-fallback pattern around `reconciler.on_before_agent` is how the plugin stays compatible with custom reconcilers that predate the #151 `parent_invocation_id` kwarg — try the new signature, catch `TypeError`, retry the old one. This pattern recurs at every reconciler + `observe_reasoning` callsite; preserve it when you add a kwarg to any duck-typed steerer/reconciler hook.

### 5.3 before_model_callback — injection + instrumentation + cancel gate

The heaviest request-side callback. Order (all best-effort, none may raise into ADK):

1. Resolve `ctx`; bail on `None`.
2. **Cancel gate:** if `is_invocation_cancelled`, consume + emit `InvocationCancelled`, then **return `_make_cancelled_llm_response()`** (non-`None` → ADK short-circuits the LLM call). This is the checkpoint that matters most — a mid-flight LLM call is the expensive work whose output would contaminate the parent transcript.
3. **Planner injection:** `self._prompt_shaper.inject_goldfive_planner_instruction(...)`. The observation-only gate lives INSIDE the shaper — it short-circuits unless `steering_is_active(ctx.steerer)`.
4. **max_output_tokens cap:** `_apply_agent_max_output_tokens_cap(...)` when `_agent_max_output_tokens > 0`. Smaller-wins ratchet-down (goldfive#256).
5. **Runtime tools hint:** `self._prompt_shaper.inject_runtime_tools_hint(...)` — the pre-emptive "which agents still have PENDING work" hint. Also gated inside the shaper.
6. **ContextEditor:** if `_context_editor is not None`, `await self._context_editor.apply(..., observation_only=observation_only)`. `observation_only` is a hard gate inside `apply()`.
7. **Instrumentation:** `_measure_request_chars(llm_request)` → stash `{start_mono, chars, messages_count}` on `_invocation_llm_pending[inv_id]`, spawn the per-LLM-call watcher (§7.1), and log `goldfive.llm.request` at INFO.
8. Return `None`.

The prompt injections are measured AFTER they run, so the reported `chars` reflect what the model actually sees. This is why instrumentation is last.

The watcher-spawn block, verbatim — note the triple guard (`> 0`, `ctx is not None`, not-already-cancelled) and the stored handle:

```python:goldfive/adapters/_adk_plugin.py
                    if (
                        self._llm_call_timeout_ms > 0
                        and ctx is not None
                        and not self.is_invocation_cancelled(inv_id)
                    ):
                        timeout_s = self._llm_call_timeout_ms / 1000.0
                        try:
                            watcher = asyncio.create_task(
                                self._run_llm_call_timeout_watcher(
                                    invocation_id=inv_id, timeout_s=timeout_s, ctx=ctx,
                                ),
                                name=f"goldfive_llm_watcher_{inv_id}",
                            )
                            pending["watcher"] = watcher
                        except RuntimeError as exc:
                            log.debug("... cannot schedule LLM-timeout watcher: %s", exc)
                    self._invocation_llm_pending[inv_id] = pending
```

### 5.4 before_tool_callback — the reporting/approval/delegation nexus

The most branch-heavy callback. It handles reporting-tool interception, task-id injection, the AgentTool cancel short-circuit, delegation observation + pinning, the runaway cap, the F3 redirect, and the approval bridge. Full flow (§6 and §6pin cover the pieces):

1. Resolve `ctx` + `tool_name` (fall back to `tool.func.__name__`).
2. **AgentTool cancel short-circuit:** if cancelled AND `_is_agent_tool_dispatch(tool)`, consume + emit, return the **minimal** `{"status": "cancelled"}`. FunctionTools are NOT short-circuited here (Bug C, v23) — their side-effect work is already committed and the next `before_model` short-circuit ends the dispatch cleanly anyway.
3. **task-id injection:** `_inject_task_id_from_state(...)` populates `tool_args["task_id"]` for reporting tools from the delegation-site pin then the agent-turn pin. It ONLY rewrites when the existing arg is missing or an obvious placeholder (`_is_placeholder_task_id`: `""`, `placeholder`, `unknown`, `todo`, `none`, `null`, `n/a`, or a non-string). A real-looking id — even one for the WRONG task — is left alone so the handler surfaces it as a proper terminal-task / not-found failure rather than silently re-targeting the call. Wrong ids are better surfaced as failures than masked. Returns `True` if `tool_args` now carries a usable id, `False` if no pin resolved (→ the unresolved-pin branch).
4. **Unresolved-pin branch:** if a reporting tool and no pin resolved, return a bare `{"acknowledged": True}` — no `error`/`detail`/`reason` keys (those become prompt-injection vectors, observed live). If the agent HAS pending candidates, additionally WARN + emit `_emit_pin_unresolved_drift` (OFF_TOPIC with a `pin_unresolved:` prefix) for operator visibility.
5. **Registered reporting tool:** route through `invoke_tool(ctx.tools, tool_name, args_map, ctx.session, ctx.steerer)` and return its dict verbatim (terminal-rejection / idempotent-ACK payloads reach the agent unchanged).
6. **AgentTool path:** `_maybe_pin_delegation_task` (observational assignee re-population), emit `DelegationObserved` (with `tool_args_json`), extend `task_lineage`, forward to reconciler, `_maybe_emit_capability_mismatch`, `_pin_delegation_task_id` (per-`function_call_id` pin), then the runaway cap, then the F3 redirect.
7. **Approval bridge:** if `_tool_requires_confirmation(tool, tool_args)`, `await _await_tool_approval(...)` (Flow B — see `13-reporting-tools-and-approval.md`).
8. Return `None` (let the tool run) if nothing short-circuited.

The `before_model` prompt-shaping order is precise and each stage has a reason it sits where it does:

| Order | Stage | Why here |
|-------|-------|----------|
| 1 | Cancel gate | Cheapest short-circuit; must precede all work. |
| 2 | `inject_goldfive_planner_instruction` | The orchestration context block the model reasons over. Gated inside `PromptShaper` on `steering_is_active`. |
| 3 | `_apply_agent_max_output_tokens_cap` | After the system instruction is in place, before anything reads `config`. |
| 4 | `inject_runtime_tools_hint` | The pre-emptive "which agents still have PENDING work" block. Gated inside `PromptShaper`. |
| 5 | `ContextEditor.apply` | AFTER all additive shaping so edits see the final contents; `observation_only` hard-gate inside `apply()`. |
| 6 | `_measure_request_chars` + watcher spawn + INFO log | LAST, so measured `chars` reflect exactly what the model sees. |

Stages 2 and 4 gate INSIDE `PromptShaper` (not via `_is_observation_only` here) — the shaper short-circuits unless `steering_is_active(ctx.steerer)`, so under passive mode the injections are byte-identical no-ops and the model sees its unmodified prompt. Stage 5 gates inside `ContextEditor.apply(observation_only=...)`. This is why `before_model_callback` itself has only ONE `_is_observation_only` read (for the ContextEditor's `observation_only` argument), not four.

The AgentTool cancel short-circuit, verbatim — note the `_is_agent_tool_dispatch(tool)` conjunct (FunctionTools are excluded) and the minimal response:

```python:goldfive/adapters/_adk_plugin.py
            if (
                inv_id_check
                and self.is_invocation_cancelled(inv_id_check)
                and _is_agent_tool_dispatch(tool)
            ):
                pending = self._cancel_state.get(inv_id_check)
                if pending is not None:
                    request = self.consume_cancel_for_invocation(inv_id_check)
                    self._cancelled_invocations.add(inv_id_check)
                    await self._emit_invocation_cancelled(
                        invocation_id=inv_id_check, agent_name="",
                        request=request, tool_name=tool_name,
                    )
                return {"status": "cancelled"}
```

The unresolved-pin branch, verbatim — the `has_candidates` split (WARN + drift vs. silent) and the bare ack:

```python:goldfive/adapters/_adk_plugin.py
            if _is_reporting_tool_name(tool_name) and not pinned:
                ...
                has_candidates = _agent_has_pending_candidates(ctx, agent_name)
                if has_candidates:
                    log.warning("before_tool_callback: pin_unresolved for %s ...", tool_name, ...)
                    await self._emit_pin_unresolved_drift(
                        ctx=ctx, agent_name=agent_name,
                        tool_name=tool_name, candidate_ids=candidate_ids,
                    )
                    return {"acknowledged": True}
                log.info("before_tool_callback: no task pinned for %s; ... (orchestration-only turn)", tool_name)
                return {"acknowledged": True}
```

Both branches return the SAME bare `{"acknowledged": True}` — the difference is operator visibility (WARN + drift), never the LLM-visible payload. This is the pin-leak lesson from goldfive#252: `research_agent` read an `error: pin_unresolved` payload as a reasoning cue and bypassed the reporting contract.

### 5.5 after_model_callback — the drift-observation heart

Fires after every model turn. Order:
1. Resolve `ctx`; bail if `ctx.steerer is None`.
2. Extract `texts`, `function_calls`, and `(reasoning, reasoning_source)` via `_choose_reasoning_text` (real chain-of-thought wins; content-fallback only on a genuine empty AND the opt-in `ReasoningDriftConfig.fallback_to_content_when_no_reasoning` flag).
3. Feed per-invocation counters (`_invocation_tool_calls`, `_invocation_last_text`) for CONFABULATION_RISK.
4. `_note_reasoning_channel_signal(...)` — the one-shot disarm WARNING (§8).
5. **Instrumentation pairing:** pop `_invocation_llm_pending[inv_id]`, compute `llm.call.duration_ms`, **cancel the per-call watcher** (`watcher.cancel()`), extract usage metadata, log `goldfive.llm.response`.
6. Build the observation `raw` dict and `await ctx.steerer.drift.observe(observation, ctx.session)`.
7. **Reasoning observation (gated on cancel):** if `reasoning` and NOT `is_invocation_cancelled`, call `observe_reasoning(reasoning, task=..., session=..., provider=..., agent_name=...)` with a `TypeError` back-compat fallback. Skipping cancelled invocations avoids judging zombie reasoning.
8. **Reflective counter (gated on cancel):** `note_llm_call(ctx.session)` — also skipped on cancelled invocations because when its counter hits `reflective_check_interval` it fires a fresh reflective LLM call.

The reasoning-observe cancel gate, verbatim — this is the "don't judge zombie reasoning" guard reused across 6+ callbacks:

```python:goldfive/adapters/_adk_plugin.py
            if reasoning:
                if inv_id and self.is_invocation_cancelled(inv_id):
                    log.debug("... skipping observe_reasoning for cancelled invocation ...")
                else:
                    observe_reasoning = getattr(
                        getattr(ctx.steerer, "drift", None), "observe_reasoning", None
                    )
                    if observe_reasoning is not None:
                        ...
                        try:
                            await observe_reasoning(
                                reasoning, task=ctx.task, session=ctx.session,
                                provider=_infer_provider(llm_response),
                                agent_name=reasoning_agent_name,
                            )
                        except TypeError:
                            # Custom steerer without the agent_name kwarg — fall back.
                            await observe_reasoning(reasoning, task=..., session=..., provider=...)
```

`reasoning_agent_name` is the live invocation's running agent (falls back to `host_agent_name`) so the steerer's per-`(agent, task)` reasoning-judge rate-limit bucket isolates agents (goldfive#252 follow-up).

The instrumentation-pairing block, verbatim — pop the pending slot, compute duration, cancel the watcher:

```python:goldfive/adapters/_adk_plugin.py
                pending = self._invocation_llm_pending.pop(inv_id, None) if inv_id else None
                if pending is not None:
                    duration_ms = int((time.monotonic() - pending["start_mono"]) * 1000)
                    metrics["llm.call.duration_ms"] = duration_ms
                    metrics["llm.request.chars"] = int(pending.get("chars", 0))
                    metrics["llm.request.messages_count"] = int(pending.get("messages_count", 0))
                    watcher = pending.get("watcher")
                    if watcher is not None and not watcher.done():
                        watcher.cancel()
                usage = _extract_usage_metadata(llm_response)
                for key, value in usage.items():
                    metrics[f"llm.usage.{key}"] = value
```

The `raw` observation dict handed to `observe` carries `texts`, `function_calls`, `reasoning`, `reasoning_source`, `finish_reason`, and (when present) `metrics` — so a custom steerer sink can surface the LLM metrics inline with each turn. The `metrics` are captured, never used for a control decision (§11).

### 5.6 after_agent_callback / after_run_callback

- **after_agent_callback:** forwards the turn-end to the reconciler (with `TypeError` fallback) inside a `try`, and emits `_emit_boundary_exited(reason="completed")` in the `finally` — so a reconciler raise cannot prevent the boundary from closing. The boundary is the canonical exit-point contract; observability must not depend on third-party hooks.
- **after_run_callback:** emits `AgentInvocationCompleted`; runs `_maybe_emit_confabulation_risk` (issue #128 — external-data-shaped task + non-empty text + zero tool calls → INFO drift, gated on the finishing agent matching the task assignee so nested sub-Runners don't misattribute) and `_maybe_emit_tool_loop_no_drift` (one aggregated `no_drift` decision per clean invocation); then clears per-top-level-invocation bookkeeping.

### 5.7 on_event_callback / on_tool_error_callback / after_tool_callback

- **on_event_callback:** observes ADK `event.actions.transfer_to_agent` / `escalate` and feeds `agent_transfer` / `agent_escalation` observations. Returns early if neither is set:

```python:goldfive/adapters/_adk_plugin.py
            actions = _safe_attr(event, "actions", None)
            transfer_to = _safe_attr(actions, "transfer_to_agent", "") or ""
            escalate = bool(_safe_attr(actions, "escalate", False))
            if not transfer_to and not escalate:
                return None
            kind = "agent_transfer" if transfer_to else "agent_escalation"
```

  This is the sole callback that surfaces ADK's own transfer/escalation machinery to the steerer — with zero prompt cooperation, invariant 1: a coordinator that `transfer_to_agent`s off-plan produces an observation regardless of whether it called any goldfive tool.
- **on_tool_error_callback:** feeds a `tool_error` observation AND records the error on `session.recent_tool_observations` via `note_tool_observation` (so the three-state reasoning judge can recognize a *provoked* deviation rooted in a hard tool exception). The `note_tool_observation` call prefers the live pin (`session.current_agent_id` / `current_task_id`) over the ADK-resolved names, falling back to `host_agent_name` / `ctx.task.id`.
- **after_tool_callback:** the tool-loop detector. Observes EVERY tool call through `_tool_loop_tracker.observe_tool_call(...)` keyed by `(invocation_id, agent_name)` with `session_run_id` (goldfive#420 — buckets accumulate across re-invocations in one run) and `observed_revision_index` (goldfive#245 — the dispatch-time staleness gate). Resets the window ONLY on `_is_progress_report_success(result)` for a progress-reporting tool (goldfive#192 acknowledged-success gate). Routes drifts through `handle_drift` (so the intervention ladder sees them) with an `observe` fallback for stubs. Also records the call on `recent_tool_observations`.

The `observe_tool_call` invocation, verbatim — note the FOUR keys that make the tracker bucket correctly across a run:

```python:goldfive/adapters/_adk_plugin.py
                drifts = self._tool_loop_tracker.observe_tool_call(
                    invocation_id=inv_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    args=dict(args_payload),
                    task_id=task_id,
                    observed_revision_index=_observed_rev,   # goldfive#245 staleness gate
                    session_run_id=_session_run_id,          # goldfive#420 cross-invocation bucket
                )
```

`session_run_id` (goldfive#420) is why a coordinator re-delegating to `debugger_agent` 11 times accumulates one bucket across all 11 sub-invocations instead of 11 fresh 2-entry windows that never trip the 7-call CRITICAL name-tier. The acknowledged-success reset uses the SAME run-scoped key:

```python:goldfive/adapters/_adk_plugin.py
            if tool_name in self._progress_reporting_tools:
                if _is_progress_report_success(result):
                    self._tool_loop_tracker.on_task_progress(
                        invocation_id=inv_id, agent_name=agent_name,
                        session_run_id=_session_run_id,
                    )
```

The tool-loop name-axis is capped at INFO without exact-repeat corroboration (goldfive#484 — `>=2` identical `(name, args_hash)`); see `07-deterministic-drift-detection.md` for the tracker internals and the `name_axis_max_severity` knob. Do NOT lower that cap from inside the plugin — the plugin only feeds the tracker.

### 5.8 after_run_callback cleanup — what gets dropped, and why

`after_run_callback` is where per-invocation state is *reaped*. This runs after the confabulation + no-drift emits and before the callback returns. Getting this wrong is how state leaks across dispatches on a reused plugin. The exact cleanup:

```python:goldfive/adapters/_adk_plugin.py
            if self._top_invocation_id and self._top_invocation_id == inv_id:
                self._top_invocation_id = ""
            if inv_id:
                self._invocation_tool_calls.pop(inv_id, None)
                self._invocation_last_text.pop(inv_id, None)
                self._cancelled_invocations.discard(inv_id)
                self._invocation_tasks.pop(inv_id, None)
```

- Releasing `_top_invocation_id` when the finishing invocation IS the top-level one lets a subsequent `invoke()` on the same plugin get a fresh dispatch.
- Dropping `_cancelled_invocations` for the finishing id prevents a future `invocation_id` collision (test-harness reuse) from inheriting a stale cancel bit.
- Dropping the registered task handle prevents a late-firing `request_invocation_cancel` from targeting a future invocation that reuses the id.

Note that `_cancel_state`, `_invocation_parents`, `_invocation_pinned_task_id`, and the tool-loop tracker are cleared wholesale in `clear_active_context` (dispatch teardown), NOT here — `after_run_callback` fires once *per invocation* (including each sub-Runner), while `clear_active_context` fires once *per dispatch*.

### 5.9 The negative-class emitters (why "nothing happened" is on the wire)

Two detectors emit an explicit "no drift" record so the offline optimizer (zicato) can distinguish "detector ran and passed" from "detector never ran":

- **`_maybe_emit_confabulation_risk`** (from `after_run_callback`): runs `goldfive.drift.classify_confabulation_risk(task, tool_call_count, output_text)`. Gated on a live steerer + task, a resolvable `task_id`, and the finishing agent matching the task assignee (so a nested AgentTool sub-Runner does not misattribute its inner text to the outer task). Routes the positive drift through `handle_drift` (fallback `observe`). The nested-sub-Runner guard, verbatim:

  ```python:goldfive/adapters/_adk_plugin.py
            assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
            if assignee and finishing_agent_name and assignee != finishing_agent_name:
                # Nested AgentTool sub-Runner whose agent is not the task's
                # owner — let the outer runner's after_run fire the check.
                return
            tool_calls = self._invocation_tool_calls.get(inv_id, 0)
            final_text = self._invocation_last_text.get(inv_id, "")
            drift = classify_confabulation_risk(
                task=task, tool_call_count=tool_calls, output_text=final_text,
            )
  ```
- **`_maybe_emit_tool_loop_no_drift`** (from `after_run_callback`): consumes the per-invocation `_tool_loop_invocation_stats` entry. When the invocation ended with `>= 1` observed tool call and zero tool-loop drifts, it emits ONE `emit_no_drift_decision(detector_name="tool_loops", ...)`. Aggregated (not per-call) so the wire isn't flooded. The `capability_check` detector emits its own negative class at delegation cadence inside `_maybe_emit_capability_mismatch` (§6cap).

The invariant: negative-class emits fire ONLY when the detector actually ran. The early-returns (no plan / no pin / import failure) deliberately do NOT emit — "no_drift" there would be a false-negative record. When you add a detector, mirror this: emit the negative class at the point the detector *ran and passed*, never at an early bail.

### 5.10 The emit helpers and the boundary pairing

Every sink emission from the plugin goes through a small family of helpers that share one pattern: resolve `_active_ctx` → `steerer._sinks`; bail if no sinks; compute `run_id` / `session_id` / `seq = session.next_sequence()`; lazily import the typed event factory from `goldfive.events`; `await emit(sinks, evt)`; swallow every failure at DEBUG. **Observability must never block a callback** — this is why every one of these is best-effort.

| Helper | Emits | Notes |
|--------|-------|-------|
| `_emit_observability(kind, **fields)` | `AgentInvocationStarted` / `AgentInvocationCompleted` / `DelegationObserved` | Dispatches on `kind`; `setdefault("session_id", ...)` so callers can override. |
| `_emit_invocation_cancelled(...)` | `InvocationCancelled` | Operator-visible only; extracts `reason`/`severity`/`drift_id`/`drift_kind`/`detail` from the `CancellationRequest` (duck-typed). Typed proto path only — the dict-envelope from PR #259 was removed. |
| `_emit_boundary_entered(...)` | `InvocationBoundaryEntered` | Idempotent: adds `invocation_id` to `_boundary_entered_invocations` and no-ops on a second call for the same id (transfer-to-agent inside one invocation). |
| `_emit_boundary_exited(..., reason=)` | `InvocationBoundaryExited` | No-op unless the id was marked entered; discards it (exit-once). |
| `close_open_boundaries(reason=)` | `InvocationBoundaryExited` for every still-open boundary | Called from the canonical `except CancelledError` / `except Exception` in `ADKAdapter._invoke_internal` so a CancelledError tearing through ADK (which skips `after_agent_callback`) still produces the paired exit. Iterates a *snapshot* so per-item discards don't mutate during iteration. |
| `_emit_approval_requested_from_plugin(...)` | `ApprovalRequested` | Module-level; used by the Flow-B approval bridge (§5.4 step 7). |
| `_emit_policy_applied_from_plugin(...)` | `PolicyApplied` | Module-level; used by the observation-only gate on the F3 redirect (§9). |

**The boundary invariant (goldfive#271 Phase 3.5):** every `InvocationBoundaryEntered` MUST be paired with exactly one `Exited`. `_boundary_entered_invocations` is the pin that guarantees this even when ADK skips `after_agent_callback` on cancel — the canonical CancelledError catch in the adapter calls `close_open_boundaries` to emit the missing exits. When you add a code path that can bypass `after_agent_callback`, verify the boundary still closes (grep `close_open_boundaries` and confirm the catch site covers your path).

### 5.11 A worked trace: one delegation, callback by callback

To ground the callback ordering, here is a single coordinator-delegates-to-`research_agent` turn under the overlay path (`invoke_passthrough`), with the plugin's action at each ADK callback. Read this once; it makes the rest of the file legible.

Setup: a two-agent tree (`coordinator` + `research_agent` behind an `AgentTool`), a plan with one PENDING task `t1` (title "research solar costs"), `observation_only=True` (production default).

1. **`set_active_context(ctx)`** (adapter, not ADK) — `_active_ctx` set, runaway counters reset, stall watchdog NOT spawned (flag off by default).
2. **`before_run_callback`** (top-level) — `_top_invocation_id = inv_A`; `_invocation_tasks[inv_A] = current_task()`; not cancelled; counters `[inv_A]` reset; emits `AgentInvocationStarted`; `note_agent_activity`.
3. **`before_agent_callback`** (agent=`coordinator`) — emits `InvocationBoundaryEntered(inv_A)`; not cancelled; `current_agent_id = "coordinator"`; the 8-signal pin ladder runs but the coordinator has no assignee-matching task, so it likely lands via a relaxed signal or leaves the coordinator unpinned (that is fine — the coordinator is orchestration-only); `reconciler.on_before_agent`.
4. **`before_model_callback`** (coordinator's turn) — not cancelled; `PromptShaper` injections are SKIPPED (observation-only, gated inside the shaper); `max_output_tokens` ratcheted to 16384; instrumentation stashes `_invocation_llm_pending[inv_A]` and spawns `goldfive_llm_watcher_inv_A`; returns `None`.
5. **`after_model_callback`** (coordinator emitted an AgentTool function_call) — pops `_invocation_llm_pending[inv_A]`, cancels the watcher, logs duration; `_invocation_tool_calls[inv_A] += 1`; feeds `observe`; reasoning observed if present.
6. **`before_tool_callback`** (tool=AgentTool→research_agent, `function_call_id = fc_1`) — not cancelled; not a reporting tool; enters the AgentTool branch: `_maybe_pin_delegation_task` binds `t1`'s assignee to `research_agent` and pins `session.current_task_id = t1`; emits `DelegationObserved(from=coordinator, to=research_agent, task_id=t1, tool_args_json=...)`; extends `task_lineage[t1].add("research_agent")`; `_maybe_emit_capability_mismatch` runs the detector on `research_agent`'s tools vs `t1` (passes → emits `capability_check` no-drift); `_pin_delegation_task_id` stamps `pending_delegations[fc_1] = {task_id: t1, ...}`; runaway count = 1 (< cap); F3 redirect classifier returns `None` (t1 is non-terminal); returns `None` → AgentTool runs.
7. **`before_run_callback`** (sub-Runner) — new `inv_B`; `parent_inv_id = inv_A`; `_invocation_parents[inv_B] = inv_A`; `_invocation_tasks[inv_B] = current_task()`.
8. **`before_agent_callback`** (agent=`research_agent`, `inv_B`) — `InvocationBoundaryEntered(inv_B)`; `current_agent_id = "research_agent"`; the pin ladder's signal 1 finds `pending_delegations[fc_1]`... but note signal 1 keys on the CURRENT invocation's `function_call_id` — here the ladder more commonly lands via signal 2 (DAG-ready exactly-1: `t1` now has assignee `research_agent`), stamping `current_task_id = t1` with source `single_match`.
9. **`before_model_callback`** / **`after_model_callback`** (research_agent's turn) — same shape as steps 4–5 under `inv_B`.
10. **`before_tool_callback`** (tool=`report_task_started`, reporting tool, no `task_id` arg) — `_inject_task_id_from_state` resolves via `_resolve_pinned_task_id`: delegation-site map (`pending_delegations[fc]`) or the agent-turn pin → `t1`; routes through `invoke_tool` → the handler transitions `t1` to RUNNING; returns the handler dict.
11. **`after_tool_callback`** — `_tool_loop_tracker.observe_tool_call(inv_B, research_agent, report_task_started, ...)`; since it's a progress report AND acknowledged, `on_task_progress` resets the window; records `recent_tool_observations`.
12. **`after_agent_callback`** (research_agent) — `reconciler.on_after_agent`; emits `InvocationBoundaryExited(inv_B, reason="completed")`.
13. **`after_run_callback`** (sub-Runner `inv_B`) — `_maybe_emit_confabulation_risk` (research task, but it used a tool → no fire); `_maybe_emit_tool_loop_no_drift` (tracker ran, no loop → aggregated `no_drift`); emits `AgentInvocationCompleted`; drops `inv_B` counters/handle.
14. **... back in the coordinator's `inv_A`** — the coordinator sees the sub-agent result, may finish; eventually `after_agent_callback` + `after_run_callback` for `inv_A` fire, closing `InvocationBoundaryExited(inv_A)`.
15. **`clear_active_context`** (adapter `finally`) — `_invocation_tasks.clear()`, cancel stall watchdog (none), `_active_ctx = None`, clear every per-dispatch map, cancel straggling LLM watchers.

Every emit above is best-effort; every cancel checkpoint short-circuits if the steerer flagged the invocation. Trace this whenever you are unsure "which callback owns X".

---

## 6. Cooperative cancellation

Cancellation is a plugin-instance state machine, NOT an ADK feature. The steerer calls `request_invocation_cancel` on the adapter (which forwards to the plugin); every callback checks a flag at the top of its body and short-circuits. This works with zero agent cooperation — invariant 1.

### 6.1 The state fields and their semantics

- `_cancel_state: dict[str, CancellationRequest]` — authoritative "cancel requested" flags. Stored on the plugin instance, NOT ADK `session.state`, because `InMemorySessionService` shallow-copies state (the same rationale as `_active_ctx`).
- `_cancelled_invocations: set[str]` — the **sticky** set. Once a callback consumes a `_cancel_state` entry (consume-once, so `InvocationCancelled` fires exactly once), the id lands here so EVERY subsequent callback on the same invocation still short-circuits.

`is_invocation_cancelled` is sticky by design:

```python:goldfive/adapters/_adk_plugin.py
        def is_invocation_cancelled(self, invocation_id: str) -> bool:
            if not invocation_id:
                return False
            inv_id = str(invocation_id)
            if inv_id in self._cancelled_invocations:
                return True
            return self._cancel_state.get(inv_id) is not None
```

Without the sticky bit, after the first cancellation a follow-up `before_model` / `before_tool` on the SAME `invocation_id` would see an empty `_cancel_state` and let the LLM call / tool dispatch proceed — exactly the demo-v12.log regression where a single `LLM_CALL_TIMEOUT` on `e-1e9e1f05` was followed by three more watcher firings on the same invocation.

**The consume/emit pattern every cancel-checkpoint uses:**

```python
if inv_id and self.is_invocation_cancelled(inv_id):
    pending = self._cancel_state.get(inv_id)
    if pending is not None:
        request = self.consume_cancel_for_invocation(inv_id)   # read + clear
        self._cancelled_invocations.add(inv_id)                # make sticky
        await self._emit_invocation_cancelled(...)             # fires once
    return None  # or the short-circuit response
```

The `if pending is not None` guard is what makes it exactly-once: the first callback consumes and emits; later callbacks see `pending is None` (already popped) but `is_invocation_cancelled` still returns `True` via the sticky set, so they short-circuit silently.

`consume_cancel_for_invocation` is read+clear (`self._cancel_state.pop(...)`); `peek_cancel_for_invocation` is a non-side-effecting read the adapter's invoke loop uses to check whether a cancel was flagged without consuming it.

### 6.2 Return-value discipline per callback

The short-circuit *value* differs by callback, and getting it wrong is a real bug:

| Callback | Cancelled short-circuit returns | Why |
|----------|--------------------------------|-----|
| `before_run` | `None` | The run just doesn't start. |
| `before_agent` | `None` (after emitting boundary-exited) | The agent turn is skipped. |
| `before_model` | `_make_cancelled_llm_response()` (**non-`None`**) | ADK only short-circuits the LLM call on a non-`None` return. Returning `None` lets the LLM run — the exact demo-v12 bug. |
| `before_tool` (AgentTool only) | `{"status": "cancelled"}` (minimal) | Non-`None` dict short-circuits the tool. FunctionTools are NOT short-circuited. |
| `after_model` | reasoning-observe + `note_llm_call` skipped | Avoid judging zombie work. |

The `{"status": "cancelled"}` response is **deliberately minimal** — no `reason`/`detail`/`drift_kind`. Richer shapes become prompt-injection vectors (lessons from goldfive#250/#252/#253 where LLMs pattern-matched on error strings and invented workarounds). Rich operator context lives on the `InvocationCancelled` sink event.

`_make_cancelled_llm_response()` builds a synthetic ADK `LlmResponse` with a `[goldfive: cancelled]` text part; if ADK types can't be imported/constructed it falls back to `{"goldfive_cancelled": True}` — ADK treats any non-`None` as a short-circuit either way.

### 6.3 request_invocation_cancel — propagation and deferred task cancel

```python:goldfive/adapters/_adk_plugin.py
        def request_invocation_cancel(
            self,
            *,
            invocation_id: str,
            request: Any,
            propagate_to_children: bool = True,
            cancel_inflight_task: bool = False,
        ) -> list[str]:
```

- **Propagation:** walks `_invocation_parents` breadth-first via `descendants_of_invocation` and flags every transitive descendant, so a cancelled coordinator's mid-flight AgentTool child short-circuits cleanly. Tree-agnostic — the plugin has no "coordinator" concept.
- **First-writer-wins:** `self._cancel_state.setdefault(flagged_id, request)` — a parent cancel with `reason="user_steer"` is not silently overwritten by a descendant-propagation pass reusing the parent's request object.
- **`cancel_inflight_task=True`** (goldfive#271 follow-up, v15 concurrent-invocation bug): ALSO fires `task.cancel()` on the registered asyncio.Task for each flagged invocation — but **deferred via `loop.call_soon(_safe_task_cancel, t, flagged_id)`**. This is critical: drift paths can reach this method *synchronously* from inside the same task that drives the dispatch (e.g. `PlanReconciler` emits PLAN_DIVERGENCE inline; the steerer's post-refine helper calls right before `_emit_plan_revised`). A direct `task.cancel()` would schedule `CancelledError` onto the very next `await` in the caller's chain — including the paired `_emit_plan_revised` — losing the on-the-wire event. `call_soon` queues the cancel for the next loop turn. A non-running loop falls back to a direct `cancel()` so `asyncio.run`-driven tests still see it.
- Default `cancel_inflight_task=False` keeps the pre-refine cancel paths flag-only; only the post-refine `_cancel_inflight_for_revision` opts in (so the cancel fires AFTER a superseding plan is installed).

The deferred-cancel loop, verbatim — `call_soon` for a running loop, direct `cancel()` fallback for none:

```python:goldfive/adapters/_adk_plugin.py
            if cancel_inflight_task:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                for flagged_id in flagged:
                    t = self._invocation_tasks.get(flagged_id)
                    if t is None or t.done():
                        continue
                    if loop is not None:
                        loop.call_soon(_safe_task_cancel, t, flagged_id)
                    else:
                        try:
                            if t.cancel():
                                log.info("goldfive.cancel.task: ... (no running loop; direct cancel)", flagged_id)
                        except Exception as exc:  # noqa: BLE001
                            log.debug("... task.cancel() for %s raised: %s", flagged_id, exc)
```

`_safe_task_cancel(task, invocation_id)` is a free function (not a bound method) precisely so `loop.call_soon` holds only the task + id strings, keeping the deferred callback reference-free of mutable plugin state. It no-ops on an already-done task and swallows any `cancel()` failure so a finished task doesn't surface a spurious traceback to the loop's exception handler.

### 6.4 The invocation-task registry view

`_invocation_tasks` looks like a `dict` but is a `_InvocationTaskRegistryView` over `StateStore` (goldfive#271 Phase 3.5). The storage moved onto `StateStore` (per the Phase-0 state-ownership contract) keyed by `Session.id`; the view forwards dict operations onto the store, resolving it lazily from `_active_ctx.session`. Reads/writes against an unbound plugin (no active context) silently no-op — matching the pre-migration empty-dict behavior.

| Dict op | Forwards to |
|---------|-------------|
| `view[inv] = task` | `store.register_invocation_task(inv, task)` |
| `view[inv]` | `store.get_invocation_task(inv)` (raises `KeyError` on miss) |
| `view.get(inv, default)` | `store.get_invocation_task` or default |
| `view.pop(inv, *default)` | `store.get_invocation_task` + `store.deregister_invocation_task` |
| `inv in view` | `store.get_invocation_task(inv) is not None` |
| `view.clear()` | `store.clear_active_invocations()` |
| `view.keys()` / `iter` / `len` | `store.active_invocation_ids()` |

The `_store()` resolver returns `None` when `_active_ctx` is `None`, and every method treats `None` as an empty registry. This is the deliberate no-op-when-unbound behavior — a unit test that constructs the plugin and probes `_invocation_tasks` without driving a callback sees an empty registry, not a crash.

**DON'T** replace this view with a plain `dict` "for simplicity" — the storage is intentionally owned by `StateStore` so per-session orchestration state is not stranded on a reused plugin instance.

### 6.5 The cancellation state machine, at a glance

```
request_invocation_cancel(inv, request, propagate, cancel_inflight)
        │
        ├─ setdefault _cancel_state[inv] = request        (first-writer-wins)
        ├─ propagate → flag every descendant via _invocation_parents
        └─ cancel_inflight? → loop.call_soon(_safe_task_cancel, task, inv)   (deferred)

        ▼   (next callback on inv, any of before_run/agent/model/tool)
is_invocation_cancelled(inv)?
        │  sticky = inv ∈ _cancelled_invocations  OR  _cancel_state[inv] is not None
        ▼ yes
   pending = _cancel_state.get(inv)
        ├─ pending is not None:  consume (pop) + add to _cancelled_invocations + emit InvocationCancelled  (exactly-once)
        └─ pending is None:      already consumed — short-circuit silently via the sticky bit
        ▼
   short-circuit per callback:
     before_run/agent  → return None (+ boundary-exited for agent)
     before_model      → return _make_cancelled_llm_response()   (NON-None!)
     before_tool       → {"status":"cancelled"} (AgentTool only)
     after_model       → skip observe_reasoning + note_llm_call

        ▼   (invocation ends)
after_run_callback: _cancelled_invocations.discard(inv); _invocation_tasks.pop(inv)
clear_active_context: _cancel_state.clear(); _cancelled_invocations.clear(); _invocation_parents.clear()
```

The three states of an invocation w.r.t. cancel: **clean** (not in either structure), **flagged** (`_cancel_state` has an unconsumed entry), **consumed-sticky** (`_cancelled_invocations` has it; `_cancel_state` popped). `is_invocation_cancelled` returns `True` for both flagged and consumed-sticky; that is the whole point.

---

## 6pin. The delegation-pin tier system

"Pinning" answers: *when a sub-agent's turn starts (or its reporting tool fires), which plan task is it enacting?* The model does not supply `task_id` — goldfive#241 hides it from the reporting-tool schema — so the plugin must resolve it structurally. There are **two** pin producers and one reader, plus a separate observational assignee re-population.

### 6pin.1 Why not just trust the assignee?

goldfive#252 zeroed `Task.assignee_agent_id` at plan-parse time so the LLM cannot pre-*declare* which sub-agent picks up a task (that would be predictive, violating invariant 4). Assignment is **observational** — described from what actually happened. So the pin machinery reconstructs the binding from observed delegation, keyed only on structurally-stable ids.

### 6pin.2 Producer A — `_pin_current_task_id_for_agent` (agent-turn pin, 8 signals)

Called from `before_agent_callback`. It is an 8-signal ladder that picks the first signal yielding a single best candidate. From the docstring:

| # | Signal | Keys on |
|---|--------|---------|
| 1 | **Delegation-site pin** | `pending_delegations[function_call_id]` stamped by the parent's `before_tool_callback`. Authoritative. |
| 2 | **DAG-ready exactly-1** | assignee match + status PENDING/RUNNING + all predecessors COMPLETED. The pre-existing happy path. |
| 3 | **Tool-arg scoring** | when (2) returns 2+, score each DAG-ready candidate against the parent AgentTool's args via `_score_candidates_by_args`. |
| 4 | **DAG gate relaxed** | drop the upstream-completion check; WARN + low-confidence sink event. |
| 5 | **Parent-pin downstream** | prefer candidates downstream of the parent's pinned task in `plan.edges` (reads `_invocation_pinned_task_id`). |
| 6 | **Correction targeting** | `goldfive.pending_corrections.<agent>.<task_id>` from the revision pipeline. |
| 7 | **Assignee normalization** | re-run 2–4 with bare/compound `agent_name` forms (defence-in-depth for compound assignees). |
| 8 | **Low-confidence best-guess** | highest-scoring candidate; emits `pin_resolved_low_confidence`. |

Every successful pin emits ONE `pin_resolved` sink event labelled `via_signal` so operators can chart how often the happy path short-circuits vs. how often relaxed signals fire (a leading indicator that pin invariants are weakening). The actual write goes through `_stamp_current_task_id` → `StateStore.set_pin_current_task(...)` with a `BindingSource` (mapped via `_BINDING_SOURCE_BY_LADDER`) and the plan `revision` (goldfive#266, so the report-time classifier can tell a fresh pin from a stale one). It also records `_invocation_pinned_task_id[invocation_id] = task_id` so a child invocation's signal 5 can read this pin.

**This ladder was reframed** from the original "exactly-1 DAG-ready single match" gate after live operator feedback: *if an agent was invoked, something precipitated the call.* The old gate gave up silently on 0/multiple matches, the agent ran unpinned, every reporting-tool call no-op'd, and the orchestration loop stagnated.

The ladder is implemented as a set of small closure-local helpers inside `_pin_current_task_id_for_agent`, each returning a candidate or `None`. When you extend the ladder, add a helper of the same shape and slot it in priority order:

| Helper | Signal |
|--------|--------|
| `_candidates_for_agent(tasks, agent_name)` | Assignee + PENDING/RUNNING filter (the base candidate set). |
| `_filter_dag_ready(...)` | Keep only candidates whose predecessors are all COMPLETED (signal 2). |
| `_scoring_args_for(...)` | Resolve the args to score against (delegation-pin tool_args → steer body → goals). |
| `_task_from_parent_pin_downstream(...)` | Signal 5 — downstream of the parent's pinned task in `plan.edges`. |
| `_task_from_reasoning_binding(...)` | A reasoning-derived binding, when present. |
| `_task_from_pending_correction(...)` | Signal 6 — `goldfive.pending_corrections.<agent>.<task_id>`. |
| `_alternate_agent_name_form(name)` | Signal 7 — bare/compound name swap. |
| `_low_confidence_best_guess(...)` | Signal 8 — highest-scoring candidate as a last resort. |
| `_emit_pin_resolved(...)` | Emits the single `pin_resolved` sink event with `via_signal`. |

Each helper is `_safe_attr`-defensive and non-raising; the ladder short-circuits on the first non-`None`. Do NOT reorder the helpers without understanding the confidence gradient — signal 1 (delegation-site pin) is authoritative and must stay first; signal 8 (low-confidence) must stay last and MUST emit `pin_resolved_low_confidence` so the weakening is visible.

### 6pin.3 Producer B — `_pin_delegation_task_id` (delegation-site pin)

Called from `before_tool_callback` on an AgentTool dispatch. When a coordinator fires N parallel AgentTool calls to the SAME sub-agent in one turn, each spawns its own sub-invocation but they all share `(agent_name, session.state)` — so `before_agent_callback` cannot disambiguate. This producer resolves the candidate for THIS dispatch and stamps it on `StateStore.set_pending_delegation(function_call_id, task_id=..., revision=..., tool_args=...)` keyed by the **`function_call_id`** (which the reporting-tool callback DOES see). Algorithm: collect DAG-ready PENDING/RUNNING candidates for the agent (`_predecessors_completed`); exactly-1 → pin; multiple → `_score_candidates_by_args`; tie/zero → no pin (fall through to Producer A's single-match path).

### 6pin.4 The reader — `_resolve_pinned_task_id`

Called from `_inject_task_id_from_state` (reporting-tool path). Resolution order: delegation-site map first (keyed by `function_call_id`), then the agent-turn pin (`StateStore.pin_current_task()`). Returns `""` when neither resolves — the caller then short-circuits with a bare `{"acknowledged": True}` (§5.4 step 4). The `function_call_id` is pulled via `_function_call_id_from_tool_context` (public `function_call_id` attr, else `""` → falls back to the agent-turn pin).

### 6pin.5 `_maybe_pin_delegation_task` — observational assignee tiers (goldfive#265)

A separate observational re-population that runs at `delegation_observed` time: it stamps `task.assignee_agent_id` (via `replace_task` + `set_session_plan` under `channel_processor_active()`) and pins `session.current_task_id`. Its multi-eligible disambiguation:

- **Tier 1 — required-tools cover:** `_select_by_required_tools` — unique candidate whose non-empty `Task.required_tools` is fully covered by the agent's live tool names (`_agent_tool_names(invoked_agent)`). Grounded in the same surface `detect_capability_mismatch` Rule B consumes. Highest confidence.
- **Tier 2 — agent-name semantic match:** `_select_by_agent_name_stems` — unique candidate whose title+description contains a stem extracted from the agent name (`reviewer_agent → reviewer`). **Bi-directional substring against a fixed role-suffix list — NO regex** (invariant 2). Issue #405 added a post-Tier-2 Rule A guard: if the invoked agent has ONLY AgentTool wrappers (delegation-only) and the chosen task is a leaf task, treat the stem match as a false positive and fall through to Tier-3.
- **Tier 3 — topic-args scorer + topo-order fallback:** `_score_candidates_by_args`, then first-eligible by plan order (`eligible[0]`).

Tier 1 wins on conflict because it runs first and short-circuits. `invoked_agent` is optional; legacy callers without an ADK agent object get the Tier-2 + Tier-3 path.

The tier dispatch, verbatim — each tier short-circuits on a unique pick; `eligible[0]` is the topo-order last resort:

```python:goldfive/adapters/_adk_plugin.py
            if len(eligible) == 1:
                chosen = eligible[0]
            else:
                chosen = None
                tier1_agent_tool_names = _agent_tool_names(invoked_agent)
                if tier1_agent_tool_names:
                    chosen = _select_by_required_tools(eligible, tier1_agent_tool_names)
                if chosen is None:
                    chosen = _select_by_agent_name_stems(eligible, invoked_agent_name, invoked_agent)
                if chosen is None:
                    scored = _score_candidates_by_args(eligible, tool_args)
                    chosen = scored if scored is not None else eligible[0]
```

The assignee stamp itself happens under `channel_processor_active()` (the single-writer envelope, goldfive#247) via `replace_task(plan, chosen_id, assignee_agent_id=invoked_agent_name)` + `set_session_plan(...)`, skipped when the assignee already matches (idempotent on parallel same-agent calls). It does NOT emit `PlanRevised` — the stamp is observational, like `NEW_WORK_DISCOVERED`.

**Tier-3 caveat:** the topo-order fallback (`eligible[0]`) is a *guess* — it picks the first DAG-ready task even when the args gave zero signal. It exists because leaving the agent unpinned is worse (stagnation), but it CAN mis-bind. This is why the assignee stamp does NOT emit `PlanRevised` (it is observational, like `NEW_WORK_DISCOVERED`) and why a wrong bind surfaces later as a proper terminal-task / capability failure rather than being masked.

### 6pin.6 The shared DAG-readiness predicate

`_predecessors_completed(task_id, edges=..., completed_ids=...)` is the single readiness predicate shared by both pin producers, `_maybe_pin_delegation_task`, AND the F3 redirect. Its semantics are **COMPLETED, not merely terminal**: a FAILED / CANCELLED / NOT_NEEDED predecessor produced no output for a successor to consume, so the pins never bind such a successor. F3 was aligned to this exact predicate (goldfive#481) so it never redirects the coordinator toward a task the pin machinery would then decline to bind. A task with no incoming edges trivially passes.

---

## 6cap. Capability-mismatch at delegation time (goldfive#253)

`_maybe_emit_capability_mismatch` runs on every AgentTool dispatch in `before_tool_callback`. It replaced the planner-LLM `PLAN_DIVERGENCE` comparison for the wrong-assignee case with a **ground-truth structural check** on the invoked agent's live tool surface. It is the third consumer of `_predecessors_completed`-adjacent structural logic and the place the descriptive-growth path lives.

### 6cap.1 Two-strategy task lookup

The detector needs a task to consult. It resolves the bound task two ways:

1. **Explicit-assignee path:** prefer a PENDING/RUNNING task whose `assignee_agent_id == invoked_agent_name`. Covers test fixtures and any caller with declarative binding.
2. **Post-#252 fallback:** since the planner zeroes `assignee_agent_id` at parse time, read `session.current_task_id` (then `StateStore.pin_current_task()`) — the pin `before_tool_callback` already stamped for this delegation via `_maybe_pin_delegation_task`. That pin describes the work this delegation is enacting.

If neither resolves a live task, skip the detector — Rule A wants a task to consult and would otherwise have to invent one.

### 6cap.2 The three rules

`detect_capability_mismatch(invoked_agent_name, invoked_agent_tools, task, all_pending_tasks)` (see `07-deterministic-drift-detection.md` for the detector internals) returns a drift or `None`:

- **Rule A** — the agent has ONLY AgentTool wrappers (delegation-only) but was bound to a leaf authoring task.
- **Rule B** — the task's `required_tools` are not covered by the agent's live tools.
- **Rule C** — out-of-DAG-order: the pin landed on a task whose role-stem mismatches the invoked agent, but another PENDING task carries that stem. Identified by `_is_rule_c_verdict(drift)` (substring match on the `_RULE_C_DETAIL_MARKER = "delegated out of DAG order"`).

### 6cap.3 Descriptive-growth suppression of Rule C (goldfive#423)

When the drift is **specifically Rule C** AND `SteeringConfig.descriptive_growth_enabled` is `True` (`_descriptive_growth_enabled(steerer)`), the plugin does NOT dispatch the Rule C drift. Instead `_attempt_descriptive_growth` calls `PlanReviser.install_descriptive_growth(session, agent_name=..., tool_args_json=..., delegation_event_id=...)` to synthesise a `discovered=True` task that carries the agent's role naturally — so the structural mismatch dissolves at the source rather than being papered over with a refine that has no clean answer. On success, `_repin_delegation_to_discovered` re-pins `session.current_task_id` onto the discovered task (under the `channel_processor_active()` single-writer envelope) and the Rule C drift is suppressed. Rule A / Rule B always dispatch normally — those are skill-gap signals, not pin-mismatch signals.

The `tool_args_json` threaded here is the agent-authored args off the `DelegationObserved` proto field (`_safe_jsonify_tool_args`), NOT a goldfive-side intercept — invariant 4. The growth helper dedups by `discovery_identity_hash` so two simultaneous growth calls cannot grow the plan twice for the same `(agent_name, args-token-set)`.

The Rule-C dispatch guard, verbatim — the ONLY branch that suppresses a drift; Rules A/B fall straight to `handle_drift`:

```python:goldfive/adapters/_adk_plugin.py
            if _is_rule_c_verdict(drift) and _descriptive_growth_enabled(steerer):
                grew = await _attempt_descriptive_growth(
                    steerer=steerer, session=ctx.session, agent_name=invoked_agent_name,
                    tool_args_json=tool_args_json, delegation_event_id=delegation_event_id,
                )
                if grew is not None:
                    await _repin_delegation_to_discovered(
                        ctx=ctx, discovered_task=grew, invoked_agent_name=invoked_agent_name,
                    )
                    log.info("... descriptive growth absorbed Rule C verdict ... (Rule C drift SUPPRESSED)", ...)
                    return
            handle = getattr(getattr(steerer, "drift", None), "handle_drift", None)
            if callable(handle):
                await handle(drift, ctx.session)
                return
```

When the detector returns `None` (capability check passed) the plugin emits `emit_no_drift_decision(detector_name="capability_check", ...)` — but only after a task was resolved and the detector actually ran; the earlier bails (no plan / no pin / import failure) never emit, because "no_drift" there would be a false-negative record.

**Deferred:** the `descriptive_growth_enabled` flag defaults `False` on main; the flip is future work. Do NOT present descriptive growth as always-on.

### 6cap.4 The runaway-delegation cap (goldfive#130)

Also in the AgentTool path: when `_agent_tool_cap > 0`, `_agent_tool_spawn_count` increments per spawn. On crossing the cap it sets `runaway_delegation_tripped = True` (one-shot) and emits `_emit_runaway_delegation_drift` (CRITICAL `RUNAWAY_DELEGATION`, built directly — the cap is an observed invariant violation, not a heuristic). Every subsequent AgentTool call in the invocation returns a `{"skipped": True, ...}` dict so the runner wraps up quickly; the adapter's invoke loop notices the tripped flag between events and breaks out. The count is reset in both `set_active_context` and `clear_active_context`.

The cap logic, verbatim — count BEFORE short-circuiting so the drift fires exactly once at the threshold crossing:

```python:goldfive/adapters/_adk_plugin.py
                if self._agent_tool_cap > 0:
                    self._agent_tool_spawn_count += 1
                    if (
                        self._agent_tool_spawn_count > self._agent_tool_cap
                        and not self.runaway_delegation_tripped
                    ):
                        self.runaway_delegation_tripped = True
                        await self._emit_runaway_delegation_drift(...)
                    if self.runaway_delegation_tripped:
                        return {
                            "skipped": True,
                            "reason": "goldfive_runaway_delegation_cap",
                            "tool_name": tool_name,
                            "detail": f"AgentTool-per-invoke cap of {self._agent_tool_cap} exceeded ...",
                        }
```

`_emit_runaway_delegation_drift` prefers `handle_drift` (so the planner gets a refine hook) and falls back to a direct `_emit_drift_detected` sink emission for steerer stubs — the same prefer-handle-fallback-emit pattern every plugin-built drift uses.

---

## 7. Watchers — spawn/cancel discipline via tracked background tasks

There are TWO background watchers, both `asyncio.Task`s on the wrapped tree's own event loop. Both follow the same discipline: **spawn a tracked task, always hold the handle, always cancel it in teardown.** Untracked `asyncio.create_task(...)` is a bug here.

### 7.1 Per-LLM-call timeout watcher

- **Spawn:** in `before_model_callback`, when `_llm_call_timeout_ms > 0` and a `ctx` exists and the invocation isn't already cancelled. `asyncio.create_task(self._run_llm_call_timeout_watcher(...), name=f"goldfive_llm_watcher_{inv_id}")`; the handle is stashed as `pending["watcher"]` in `_invocation_llm_pending[inv_id]`. Wrapped in `try/except RuntimeError` for the no-running-loop case (some unit harnesses).
- **Cancel (normal):** in `after_model_callback`, after popping `pending` and computing duration: `if watcher is not None and not watcher.done(): watcher.cancel()`. The LLM call returned within budget, so the sleep is no longer needed. Tolerates a watcher that already fired (race at the exact budget boundary).
- **Cancel (teardown):** `clear_active_context` cancels any straggling watchers.
- **On fire:** the watcher slept past `timeout_s`, so it emits a CRITICAL `LLM_CALL_TIMEOUT` drift (`steerer.drift.observe` + `_emit_drift_detected`), then — **gated on `_is_observation_only(ctx)`** — writes the cancel flag via `request_invocation_cancel`. Under passive mode the drift/telemetry still fires but the cancel-flag write is SKIPPED (goldfive#476): a healthy local model can genuinely need longer than the budget, and a passive observer must not discard work the unwrapped system would complete.

The watcher does NOT terminate the in-flight LLM call mid-stream (ADK exposes no hook). It flags cooperative cancel so the *next* checkpoint short-circuits.

The fire path, verbatim — telemetry ALWAYS fires; the cancel-flag write is gated:

```python:goldfive/adapters/_adk_plugin.py
            try:
                await asyncio.sleep(timeout_s)
            except asyncio.CancelledError:
                return
            # ... build + observe the CRITICAL LLM_CALL_TIMEOUT drift (telemetry) ...
            if _is_observation_only(ctx):
                log.info("goldfive.llm.timeout ... observation_only=True — cancel-flag write skipped", ...)
                return
            request = CancellationRequest(
                invocation_id=invocation_id, reason="llm_call_timeout",
                severity=DriftSeverity.CRITICAL,
                drift_kind=DriftKind.LLM_CALL_TIMEOUT.value, ...)
            self.request_invocation_cancel(
                invocation_id=invocation_id, request=request, propagate_to_children=True,
            )
```

The `except asyncio.CancelledError: return` at the top is the normal exit — `after_model_callback` cancels the watcher when the LLM call returns in budget. The gate (`_is_observation_only`) is between the telemetry emit and the cancel-flag write, exactly the strict-passive shape from §9.

### 7.2 Stall watchdog (goldfive#487, flag-gated, default OFF)

- **Config:** `SteeringConfig.stall_watchdog_enabled` (default `False`) and `stall_timeout_s` (default `600`), stashed on the steerer as `_stall_watchdog_enabled` / `_stall_timeout_s`.
- **Spawn:** `_maybe_start_stall_watchdog(ctx)` from `set_active_context`. Cancels any prior watchdog first (at most one per plugin), then `asyncio.create_task(self._run_stall_watchdog(...), name=f"goldfive_stall_watchdog_{session.id}")`. The un-flagged path costs one attribute read and spawns nothing. `RuntimeError` (no running loop) degrades to off, like the LLM watcher.
- **Cancel:** `_cancel_stall_watchdog()` from `clear_active_context` (idempotent — nulls the field then cancels if not done).
- **Liveness watermark:** `_stall_liveness_watermark(session, floor=started)` = max of every `task_last_progress_at` stamp and `session.last_observed_event_at` (the liveness stamp every observation updates), floored at watchdog-start so a fresh session isn't instantly stale.
- **Behavior per poll tick:** watermark advanced → reset episode bookkeeping; idle past `GOAL_DRIFT_IDLE_SECONDS` → trigger the idle goal-drift judge once per episode (`_trigger_idle_goal_drift_check` → `_spawn_goal_drift_judge_background`); idle past `timeout_s` → emit `TASK_TIMEOUT` at WARNING escalating to CRITICAL at each further multiple (graduated severity, same shape as tool_loops), routed through `handle_drift` (telemetry-only under `observation_only`). **SKIPPED while an LLM call is in flight under its own budget** (`_llm_watcher_inflight()`) — that hang is the per-call watcher's `LLM_CALL_TIMEOUT` case; firing here too would double-report.

**Honest limitation (documented in the module + method docstrings):** the watchdog is an asyncio task on the wrapped tree's own event loop. A synchronously-blocking tool (a `def` tool doing blocking I/O or CPU work without yielding) starves the loop and the watchdog cannot fire until the block ends. Detecting sync-blocked loops is out of scope. The covered cases are hung *async* tool calls and idle-with-no-transitions runs. Do not claim the watchdog catches sync-blocked stalls.

`_goal_drift_idle_seconds()` re-reads `goldfive.drift.goals.GOAL_DRIFT_IDLE_SECONDS` **per poll** (not captured at spawn) so an optimization-manifest `setattr` on the knob takes effect on a running watchdog — floored at `0.001` so a zero/negative override can't turn the idle trigger into a per-poll judge storm.

The poll loop core, verbatim — note the graduated-severity multiple, the `_llm_watcher_inflight` skip, and the `CancelledError` clean exit:

```python:goldfive/adapters/_adk_plugin.py
            while True:
                idle_goal_s = self._goal_drift_idle_seconds()
                await asyncio.sleep(max(0.005, min(timeout_s, idle_goal_s) / 8.0))
                watermark = self._stall_liveness_watermark(session, floor=started)
                if watermark > last_watermark:
                    last_watermark = watermark
                    episode_fires = 0
                    goal_judge_fired = False
                idle_s = time.monotonic() - watermark
                if not goal_judge_fired and idle_s >= idle_goal_s:
                    goal_judge_fired = True
                    self._trigger_idle_goal_drift_check(steerer, session, idle_s)
                if idle_s < timeout_s * (episode_fires + 1):
                    continue
                if self._llm_watcher_inflight():
                    continue
                severity = DriftSeverity.WARNING if episode_fires == 0 else DriftSeverity.CRITICAL
                episode_fires += 1
                # ... build TASK_TIMEOUT drift, route through handle_drift ...
```

The whole loop is wrapped in `try: ... except asyncio.CancelledError: return` so `clear_active_context`'s `_cancel_stall_watchdog()` exits it cleanly. The inner `handle_drift` catch re-raises `CancelledError` (so teardown always wins) and swallows everything else.

---

## 8. Reasoning-channel disarm warning (goldfive#263 follow-up)

Non-thinking models (Gemma, Mistral, several base deployments) never emit a separate reasoning/thinking stream, so `observe_reasoning` never fires and every reasoning-judge detector (OFF_TOPIC, GOAL_DRIFT, INTENT_DIVERGENCE, LOOPING_REASONING) silently disarms for the whole run. `_note_reasoning_channel_signal` makes this visible.

Counting rules:
- A turn that fed the channel (real reasoning OR content-fallback) **resets** the per-agent streak.
- A turn with empty reasoning but a non-empty text body **increments** it.
- Function-call-only / empty turns neither count nor reset (thinking models frequently omit the stream on pure tool turns; counting them would false-positive).

The counting core, verbatim:

```python:goldfive/adapters/_adk_plugin.py
            key = agent_name or "?"
            if reasoning:
                self._no_reasoning_streak.pop(key, None)   # real reasoning resets
                return
            body = " ".join(t for t in texts if t).strip()
            if not body:
                return                                       # empty / tool-only turn: neither count nor reset
            streak = self._no_reasoning_streak.get(key, 0) + 1
            self._no_reasoning_streak[key] = streak
            if streak < _NO_REASONING_WARN_STREAK or key in self._no_reasoning_warned:
                return
            self._no_reasoning_warned.add(key)               # one-shot per agent
```

After `_NO_REASONING_WARN_STREAK = 3` consecutive empty-reasoning turns, it fires a **one-shot** WARNING per agent naming the remedy (`ReasoningDriftConfig.fallback_to_content_when_no_reasoning=True` / `GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT=1`) plus a record-only sink event (`DriftKind.CUSTOM`, `INFO`, via `_emit_drift_detected`). The one-shot is why `_no_reasoning_warned` is plugin-lifetime state (NOT cleared in `clear_active_context`) — the warning stays one-shot per agent across dispatches on the same wrapped tree. The fallback is NOT auto-enabled — that behavior change is reserved for the operator (goldfive#476 one-shot-per-run tightening applies to the disarm warning path).

---

## 9. observation_only — the ONE accessor

Under `observation_only=True` (the production default) the plugin must be strictly passive. The ONLY sanctioned kill-switch read in this file is:

```python:goldfive/adapters/_adk_plugin.py
def _is_observation_only(ctx: Any) -> bool:
    return not steering_is_active(_safe_attr(ctx, "steerer", None))
```

`steering_is_active` delegates to `DefaultSteerer.is_active_steering()` (single predicate, goldfive#488) and resolves a missing / broken / raising steerer as **passive** — the fail-safe direction for a surface whose whole purpose is to NOT fire when the operator opted into passive observation.

The intervention surfaces that gate on it:
1. **The per-LLM-call watcher's cancel-flag write** (§7.1) — telemetry fires, cancel is skipped (goldfive#476).
2. **The F3 pre-dispatch redirect** in `before_tool_callback` (goldfive#481). When suppressed it logs and emits `_emit_policy_applied_from_plugin(policy_name="observation_only_gate", outcome="suppressed", ...)` and lets the AgentTool dispatch proceed untouched.
3. **The request-side `ContextEditor.apply(...)`** (goldfive#397) — `observation_only` is passed through as a hard gate inside `apply()`.

The `before_model` prompt injections (planner instruction, runtime tools hint) gate INSIDE `PromptShaper`, not here — the shaper short-circuits unless `steering_is_active(ctx.steerer)`.

The F3 suppressed-branch, verbatim — the canonical shape every plugin-side intervention gate should copy:

```python:goldfive/adapters/_adk_plugin.py
                if _is_observation_only(ctx):
                    log.info("before_tool_callback: observation_only=True — F3 redirect for %s SUPPRESSED ...", to_agent, ...)
                    await _emit_policy_applied_from_plugin(
                        session=ctx.session, steerer=ctx.steerer,
                        policy_name="observation_only_gate", outcome="suppressed",
                        reason="observation_only=True",
                        detail=f"intervention=f3_predispatch_redirect target_agent={to_agent} redirect_to={...}",
                    )
                else:
                    log.info("before_tool_callback: F3 redirect — ... redirecting coordinator to %s", ...)
                    return redirect
                # Fall through: AgentTool still runs, we're just observing.
```

Note that under suppression the code does NOT `return redirect` — it falls through and lets the AgentTool dispatch proceed. That fall-through IS the strict-passive guarantee: the wrapped system behaves exactly as it would unwrapped, and the only trace is the `PolicyApplied(suppressed)` telemetry.

### 9.1 Why "missing steerer → passive" is the safe default

`steering_is_active(None)` returns `False`, so `_is_observation_only` returns `True` (passive) for a missing / broken / raising steerer. This is the fail-safe direction: a surface whose whole purpose is to intervene should, when it cannot determine the mode, choose to do NOTHING rather than risk discarding work the operator wanted preserved. The inverse (defaulting to active on an unreadable steerer) could cancel a legitimate LLM call or refuse a legitimate dispatch on a stub steerer — exactly the failure the strict-passive pattern exists to prevent. Every gate in this file inherits this default by routing through the one accessor.

**Rule:** do NOT read `steerer._steering_config.observation_only`, `SteeringConfig.observation_only`, or any other field directly to decide whether to intervene. Route through `_is_observation_only(ctx)` (or `steering_is_active(steerer)`). goldfive#488 deleted the module-global test hook + autouse fixture specifically so the suite runs the shipped `observation_only=True` default (~90 tests explicitly opt into active mode); a stray direct read reintroduces the exact divergence that fixture masked.

---

## 10. F1 directive acks and F3 pre-dispatch redirect

- **F3 pre-dispatch redirect** (`_maybe_redirect_completed_agent`): returns a redirect-error dict ONLY when the coordinator invokes an AgentTool whose plan tasks are ALL terminal AND a non-terminal `next_pending` task exists assigned to a *different* agent. Returns `None` (allow the dispatch) in every other case: the target has PENDING/RUNNING/BLOCKED work; the target is off-plan (PLAN_DIVERGENCE handles that — no double-handling); no plan installed; or `next_pending` is on the same agent (legitimate follow-up). `next_pending` uses `_predecessors_completed` (COMPLETED semantics) so a redirect never points at a task the pin would decline to bind. The redirect is gated on `_is_observation_only` at the call site in `before_tool_callback` (§9). The response is a plain `{"error": "...", "redirect_to": ...}` — a deliberately thin shape the LLM reads as "go to the other agent"; operator observability lives on the drift/telemetry surface.
- **F1 directive acks** are the steerer's proactive anchor (the directive payload it lands in state that the coordinator reads next turn); F3 is the structural fence. The two are complementary — see `09-steering-ladder-and-gates.md` for F1's producer side.

### 10.1 The F3 classifier, step by step

`_maybe_redirect_completed_agent(ctx, target_agent)` is the whole F3 decision. Trace its early-returns (each returns `None` = allow the dispatch):

1. No `target_agent` → `None`.
2. No plan / no tasks → `None`.
3. Collect tasks assigned to `target_agent`, matching on the BARE agent name (`rsplit(".", 1)[-1]`) so fully-qualified ADK paths like `coordinator.research_agent` round-trip. No assigned tasks → `None` (off-plan agent; PLAN_DIVERGENCE owns that).
4. If ANY assigned task is non-terminal (`status not in TERMINAL_TASK_STATUSES`) → `None` (legitimate dispatch).
5. All assigned terminal → find the next PENDING task whose every predecessor is COMPLETED (`_predecessors_completed`). None found → `None` (plan effectively done).
6. If that `next_pending` is assigned to `target_agent` (bare) → `None` (legitimate follow-up work).
7. Otherwise return the redirect dict: `{"error": "All plan tasks for <agent> are complete. Next pending task is '<title>' assigned to <other>. Please invoke that agent.", "redirect_to": <other>}`.

Only step 7 is a redirect; everything else allows the dispatch. `TERMINAL_TASK_STATUSES` (goldfive#485) is the canonical terminal set (includes `NOT_NEEDED`), so the "all terminal" check and the `_predecessors_completed` COMPLETED check agree on which sweeps count. The call site then applies the observation-only gate (§9): under passive mode the redirect is logged + `PolicyApplied(suppressed)` and the AgentTool runs anyway.

---

## 11. Request char-count and metrics — captured vs consumed

`before_model_callback` calls `_measure_request_chars(llm_request)` → `(chars, messages_count)` and stashes them with `start_mono` on `_invocation_llm_pending[inv_id]`. `after_model_callback` pops the slot, computes `llm.call.duration_ms`, extracts usage via `_extract_usage_metadata` (`prompt_token_count → prompt_tokens`, `candidates_token_count → completion_tokens`, `total_token_count → total_tokens`; only present+non-zero fields), and:

- **Logs** `goldfive.llm.request` (before) and `goldfive.llm.response` (after) at INFO with structured fields so an operator tailing stderr can correlate context growth against post-steer slowdown (issue #172, hypothesis 1).
- **Enriches** the observation `raw["metrics"]` dict so custom steerer sinks can surface metrics alongside each LLM turn.

The metrics are **captured for observability, not consumed for control** — no drift or intervention keys off `chars` or token counts. The per-call *timeout* (§7.1) keys off wall-clock duration, not tokens. Do not add a control decision that reads these metrics without an explicit design + gate.

`_extract_reasoning` walks provider-specific shapes in priority order (first non-empty wins):

| Priority | Shape | Providers |
|----------|-------|-----------|
| 1 | `content.parts[i].thought == True` → `.text` | Google Gemini (thought parts) |
| 2 | `choices[0].message.reasoning_content` (or `.reasoning`) | Qwen3.5 via LiteLLM, o1-series, Deepseek |
| 3 | `content[i].type == "thinking"` → `.thinking` | Anthropic Claude extended thinking |
| 4 | flat `reasoning` / `reasoning_content` / `thinking` attr | tolerant fallback |

`_infer_provider(llm_response)` tags the observation so the steerer's per-provider judge bucketing works. When extraction returns empty AND `ReasoningDriftConfig.fallback_to_content_when_no_reasoning=True`, `_choose_reasoning_text` synthesises a `content_fallback` source from the response body; otherwise it returns `("", "")` and the reasoning-judge detectors receive no input for that turn (which is what the §8 disarm warning surfaces). Reasoning-related toggles (`/no_think`, `enable_thinking`) are handled in the one LLM-call module `goldfive/_llm.py` (goldfive#491), which restricts the Qwen thinking-disable path to the Qwen/litellm family via `THINKING_DISABLE_CAPABILITIES` — the plugin only *reads* whatever reasoning the response carried.

---

## 11b. Module-level helper catalogue

The file has ~40 module-level (non-method) helpers. Nearly all are pure, `_safe_attr`-defensive, and never raise. When you need one of these behaviors, use the existing helper — do NOT reimplement it inline. Grouped by concern:

### Context / session resolution
| Helper | Returns / does |
|--------|----------------|
| `session_context_from_invocation(inv_ctx)` | Live `SessionContext` via plugin-manager tree-walk (§2.2). |
| `_session_state_from_callback(ctx)` | ADK session.state mapping across all known shapes; `{}` fallback. |
| `_session_context_from_callback(ctx)` | `SessionContext` from the `SESSION_CONTEXT_STATE_KEY` stash (unit-test path). |
| `_goldfive_session_from_tool_context(tc)` | goldfive `Session` from a `ToolContext` (tree-walk then legacy stash). |
| `_safe_attr(obj, name, default)` | Defensive `getattr` normalizing `None`→default and swallowing a raising `__getattr__`. |

### Task-id / pin resolution
| Helper | Returns / does |
|--------|----------------|
| `_is_placeholder_task_id(v)` | True if missing / obvious placeholder (`""`, `placeholder`, `unknown`, `todo`, `none`, `null`, `n/a`; non-str). |
| `_is_reporting_tool_name(name)` | True for `report_task_*` or `report_awaiting_approval`. |
| `_inject_task_id_from_state(tool_name, tool_args, tool_context)` | Populates `tool_args["task_id"]` from the pin; returns whether a usable id is present. |
| `_resolve_pinned_task_id(tool_context)` | Delegation-site pin then agent-turn pin; `""` if neither. |
| `_function_call_id_from_tool_context(tc)` / `_function_call_id(tc)` | Extract `function_call_id` (the latter mints `adk-<uuid>` fallback). |
| `_delegation_pin_task_id` / `_delegation_pin_revision` / `_delegation_pin_tool_args` | Unpack a pending-delegations entry (tolerates str + versioned Mapping shapes). |

### Delegation / DAG / scoring
| Helper | Returns / does |
|--------|----------------|
| `_is_agent_tool_dispatch(tool)` | True if AgentTool (isinstance then `.agent` duck-type). |
| `_agent_has_pending_candidates(ctx, agent)` | True if the plan has any PENDING/RUNNING task for the agent. |
| `_completed_task_ids(tasks)` | Set of COMPLETED task ids. |
| `_predecessors_completed(task_id, edges, completed_ids)` | The shared DAG-readiness predicate (COMPLETED semantics). |
| `_maybe_redirect_completed_agent(ctx, target_agent)` | The F3 redirect classifier (returns redirect dict or `None`). |
| `_score_candidates_by_args(candidates, tool_args)` | Best token-overlap candidate; `None` on tie / zero. |
| `_agent_tool_names(agent)` | Public tool names off an ADK agent. |
| `_select_by_required_tools(candidates, names)` | Tier-1 required-tools cover. |
| `_select_by_agent_name_stems(candidates, name, agent)` | Tier-2 stem match + #405 Rule A guard. |

### Response / reasoning extraction
| Helper | Returns / does |
|--------|----------------|
| `_extract_usage_metadata(resp)` | prompt/completion/total token dict (present+non-zero only). |
| `_extract_text_parts(resp)` | Text parts list. |
| `_extract_reasoning(resp)` | Per-provider reasoning/thinking text. |
| `_choose_reasoning_text(resp, fallback_enabled)` | `(text, source)` where source ∈ `reasoning` / `content_fallback` / `""`. |
| `_infer_provider(resp)` | Provider tag for judge bucketing. |
| `_extract_function_calls(resp)` | List of `{name, args}`. |
| `_is_progress_report_success(resp)` | True only on `acknowledged=True` with no `error` key. |

### Approval / confirmation / drift-shape
| Helper | Returns / does |
|--------|----------------|
| `_tool_requires_confirmation(tool, args)` | True if `_require_confirmation` / `require_confirmation` (bool or callable). |
| `_tool_approval_prompt(tool, name, args)` | Human prompt (explicit `approval_prompt` or synthesised). |
| `_await_tool_approval(...)` | Flow-B gate: registers a waiter, emits `ApprovalRequested`, awaits APPROVE/REJECT. |
| `_is_rule_c_verdict(drift)` | True if the CAPABILITY_MISMATCH Rule C marker is in the detail. |
| `_descriptive_growth_enabled(steerer)` | Reads `SteeringConfig.descriptive_growth_enabled` (default False). |
| `_attempt_descriptive_growth(...)` / `_repin_delegation_to_discovered(...)` | The Rule-C-absorption growth + re-pin pair. |

### JSON / misc
| Helper | Returns / does |
|--------|----------------|
| `_jsonable(v)` | Best-effort JSON-serialisable coercion. |
| `_safe_jsonify_tool_args(args)` | Stable sorted JSON for proto carriage; `""` on failure. |
| `_as_observation(kind, detail, raw, task, agent_id)` | The observation dict handed to `steerer.drift.observe`. |
| `_make_cancelled_llm_response()` | Synthetic `LlmResponse` for the before_model cancel short-circuit. |
| `_is_observation_only(ctx)` | THE kill-switch accessor (§9). |
| `_safe_task_cancel(task, inv_id)` | Deferred `loop.call_soon` cancel callback. |

### The reporting-tool dispatch (verbatim)

The registered-reporting-tool branch of `before_tool_callback` routes through `invoke_tool` and returns its dict verbatim:

```python:goldfive/adapters/_adk_plugin.py
            tool_names_registered = {spec.name for spec in ctx.tools}
            if tool_name in tool_names_registered:
                ...
                try:
                    result = await invoke_tool(
                        ctx.tools, tool_name, args_map, ctx.session, ctx.steerer,
                    )
                except Exception as exc:  # noqa: BLE001
                    return {"acknowledged": True, "error": str(exc)}
                if isinstance(result, dict):
                    return result
                return {"acknowledged": True}
```

`invoke_tool` (in `_tool_invocation.py`, covered in `13-reporting-tools-and-approval.md`) owns the terminal-task rejection, idempotency, and loop-guard layers. Do NOT call a reporting handler directly — always route through `invoke_tool` so those layers fire.

---

## 11c. The state-protocol key names (`_adk_state_protocol.py`)

The `goldfive.*` state keys are named constants in `goldfive/adapters/_adk_state_protocol.py`. **These names are a stable external contract** — harmonograf and custom consumers read them. Do not rename them without a coordinated submodule bump. All are prefixed `GOLDFIVE_PREFIX = "goldfive."`.

| Constant | Key | Role |
|----------|-----|------|
| `KEY_CURRENT_TASK_ID` | `goldfive.current_task_id` | The agent-turn pin the reporting handlers default from. |
| `KEY_CURRENT_TASK_TITLE` / `_DESCRIPTION` / `_ASSIGNEE` / `_REVISION` | `goldfive.current_task_*` | Descriptive context for the current task. |
| `KEY_PLAN_ID` / `KEY_PLAN_SUMMARY` / `KEY_RUN_ID` | `goldfive.plan_id` etc. | Run/plan identity. |
| `KEY_AVAILABLE_TASKS` / `KEY_TOOLS_AVAILABLE` | `goldfive.available_tasks` / `goldfive.tools_available` | The runtime-tools-hint surface. |
| `KEY_COMPLETED_TASK_RESULTS` | `goldfive.completed_task_results` | Upstream outputs for a successor. |
| `KEY_PENDING_CORRECTIONS` | `goldfive.pending_corrections` | Pin signal 6 source (`.<agent>.<task_id>`). |
| `KEY_ACTIVE_STEER_BODY` / `_AT_TURN` | `goldfive.active_steer.*` | The steer body the pin-scoring can consult. |
| `KEY_CANCEL_REQUESTED` | `goldfive.cancel_requested` | Documented contract for the cancel flag; the plugin's `_cancel_state` dict is the live source of truth (§6.1). |
| `KEY_INVOCATION_PARENTS` | `goldfive.invocation_parents` | The parent map `descendants_of_invocation` walks for cancel propagation. |
| `PENDING_DELEGATIONS_KEY` (from `state_store`) | — | The delegation-site pin map keyed by `function_call_id`. Re-exported here as `_PENDING_DELEGATIONS_KEY` for legacy imports. |

Note the layering: the state-protocol module *documents* the key semantics as a stable contract, but the live source of truth for cancellation, pins, and the invocation-task registry is the plugin instance / `StateStore` (because ADK shallow-copies `session.state` — §2.1). `descendants_of_invocation` is the one function from this module the plugin calls at runtime (in `request_invocation_cancel`).

---

## 12. Duck-typing hazards — the rules for touching ADK reaches

This file reaches into ADK-private attributes everywhere (`_invocation_context`, `_function_call_id`, `_require_confirmation`). ADK's callback objects have changed shape across versions, so nearly every read is defensive. There is a discipline; follow it exactly.

### 12.1 `_safe_attr` is the base primitive

```python:goldfive/adapters/_adk_plugin.py
def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    return value if value is not None else default
```

It swallows a raising `__getattr__` (some ADK proxies raise) AND normalizes `None` to the default. Use it for every attribute read off an ADK object whose shape you don't control. **DON'T** use bare `getattr(obj, name)` on ADK callback objects — a property that raises will crash the callback and abort the run.

### 12.2 Multi-shape chains

ADK has used several shapes for the same thing. The canonical fallback chains you will see and must preserve:

- **Invocation context:** `_safe_attr(cb, "_invocation_context", None) or _safe_attr(cb, "invocation_context", None)`. Both spellings appear across ADK versions. Always try both.
- **Session state:** `_session_state_from_callback` tries `ctx.session.state`, `ctx._invocation_context.session.state`, `ctx.invocation_context.session.state`, then `ctx.state`, then `{}`.
- **function_call_id:** `_function_call_id` tries `function_call_id` then `_function_call_id`, then mints `adk-<uuid>` so correlation still works with a minimal stub.

### 12.3 AgentTool detection

`_is_agent_tool_dispatch` prefers `isinstance(tool, AgentTool)` when the `adk` extra imports, with a duck-typed fallback: AgentTool carries a `.agent` `BaseAgent` pointer; FunctionTool carries `.func`. Presence of `.agent` is the discriminator. When unsure, treat the tool as a plain function and let it run (conservative — short-circuiting a plain function loses committed side-effect work).

### 12.4 The rules

1. **Every new ADK attribute read goes through `_safe_attr`** (or a `try/except` for chained/iterated reads).
2. **Every ADK type import is lazy + `try/except`** inside the function (never at module load).
3. **When you add a new callback body, it must not raise into ADK.** Wrap risky work in `try/except Exception: log.debug(...)` and continue. The pattern is everywhere: instrumentation/observability must never shadow the real response path.
4. **Prefer a documented public ADK surface over a private one when both exist,** but keep the private fallback (e.g. `require_confirmation` public then `_require_confirmation`). ADK's public surface is not stable enough to drop the private reach.
5. **When ADK changes a shape,** add a new branch to the fallback chain — do NOT replace the existing branch (older ADK versions in the field still use it).

---

## 13. Common mistakes (this file bites weak models hardest)

Each row is a concrete wrong edit and its correct alternative.

### 13.1 Adding an intervention without the observation-only gate

**WRONG:** you add a new "refuse this dispatch" or "cancel this invocation" branch in a callback and return the refusal unconditionally.

**RIGHT:** gate it. Any surface that *changes what the wrapped system would do* must be a no-op under passive mode:

```python
if _is_observation_only(ctx):
    log.info("... SUPPRESSED under observation_only ...")
    await _emit_policy_applied_from_plugin(
        session=ctx.session, steerer=ctx.steerer,
        policy_name="observation_only_gate", outcome="suppressed",
        reason="observation_only=True", detail="intervention=<name> ...",
    )
    # let the dispatch proceed untouched
else:
    return <the intervention>
```

Telemetry (drift emission, logging) may still fire in passive mode; only the *action* is gated. Mirror the F3 redirect and the LLM-watcher cancel-flag write.

### 13.2 Spawning an untracked asyncio task

**WRONG:** `asyncio.create_task(self._some_watcher(...))` with no stored handle.

**RIGHT:** store the handle (on `_invocation_llm_pending[inv_id]["watcher"]`, or a dedicated `self._x_task` field), cancel it on the paired completion callback, AND cancel it in `clear_active_context`. An untracked task becomes an orphan that fires a drift against a dead dispatch (the demo-v12 multi-fire bug) or leaks across dispatches. Wrap the `create_task` in `try/except RuntimeError` for the no-running-loop case (synchronous test harnesses) — degrade to "watcher off", never crash. Give the task a `name=` so it is identifiable in a task dump.

### 13.3 Keying per-invocation state on an unstable id

**WRONG:** keying a pin, dedup, or gate on anything the model minted (a task_id the LLM typed, an agent-chosen slug) or on a per-condition churning value.

**RIGHT:** key on ADK `invocation_id`, `function_call_id`, `session.run_id`, or a plan task id. These are structurally stable (invariant 6). The delegation-site pin keys on `function_call_id` precisely because parallel same-agent calls share `(agent_name, session.state)` but each has a distinct `function_call_id`. If a churning id forces a coarser key, fix the churn upstream — do not coarsen the key (that lesson cost real debugging time; see the lifecycle-gate memory note).

### 13.4 Returning `None` from `before_model_callback` when you meant to short-circuit

**WRONG:** returning `None` to "skip" the LLM call.

**RIGHT:** return a non-`None` value — `_make_cancelled_llm_response()`. Per ADK's contract, `None` from `before_model_callback` lets the request proceed. This is the exact demo-v12 regression: a single `LLM_CALL_TIMEOUT` returned `None`, the LLM ran anyway, and the watcher re-fired. For `before_tool_callback`, return a **dict** to short-circuit, `None` to let the tool run.

### 13.5 Leaking a prompt-injection surface into a tool response

**WRONG:** returning `{"acknowledged": False, "error": "pin_unresolved", "detail": "..."}` or a rich `{"status": "cancelled", "reason": ...}` to the model.

**RIGHT:** return the minimal shape — bare `{"acknowledged": True}` for an unresolved pin, `{"status": "cancelled"}` for a cancel. Tool responses go back to the LLM verbatim; any editorializing string is read as actionable context (observed live: `research_agent` read `error: pin_unresolved` and bypassed the reporting contract, "Let me try a different approach"). Operator visibility goes on a sink event (`_emit_pin_unresolved_drift`, `InvocationCancelled`), never in the LLM-visible payload.

### 13.6 Reading ADK `session.state` as a cross-callback channel

**WRONG:** writing a goldfive key to ADK `session.state` in one callback and reading it in another.

**RIGHT:** the goldfive `Session` (via `_active_ctx` / `session_context_from_invocation`) and `StateStore` are the reliable channels. `InMemorySessionService` shallow-copies state on every `get_session`, so cross-callback ADK-state writes land on stranded copies. Every pin, cancel flag, and per-invocation map is stored on the plugin instance or `StateStore` for this reason (§2.1).

### 13.7 Forgetting to clear a new per-dispatch field

**WRONG:** adding `self._my_map: dict[str,X] = {}` in `__init__` and never clearing it.

**RIGHT:** clear it in `clear_active_context` alongside the other per-dispatch fields — UNLESS you deliberately want plugin-lifetime state (like `_no_reasoning_warned`), in which case document why in the `__init__` comment. Uncleared per-dispatch state leaks into the next `invoke` on the reused adapter.

### 13.8 Blaming the LLM / model for a slow or looping run

**WRONG:** attributing a 5-minute turn or a delegation loop to the model and adding a prompt tweak.

**RIGHT:** it is almost always plugin/steerer code — unbounded `max_output_tokens` (the `agent_max_output_tokens` cap exists for this), a drift loop, or a missing wall-clock budget (the LLM-timeout watcher and stall watchdog exist for this). Prompt tweaks that require agent cooperation also violate invariant 1. Add a structural fence, not a prompt. Before concluding "the model is slow", run the empirical baseline (drive the same tree without steering).

### 13.9 Reintroducing a regex heuristic for name/task matching

**WRONG:** matching an agent name to a task with a regex (`re.match(r"(\w+)_agent", name)`) or a keyword list scan.

**RIGHT:** the tier-2 disambiguator uses `stem_token_match` (bi-directional substring against a fixed role-suffix list) — no regex. Exact-equality / hash matching of structured ids IS allowed. goldfive#166/#167 retired `_GENERIC_VERB_PREFIX_RE` / `_FACTUAL_QUESTION_RE`; do not resurrect that shape. If you need NL classification, use an LLM classifier or design it away.

### 13.10 Short-circuiting a FunctionTool on cancel

**WRONG:** extending the cancel short-circuit in `before_tool_callback` to ALL tools.

**RIGHT:** only AgentTool dispatches are short-circuited (`_is_agent_tool_dispatch(tool)`). FunctionTools (write_webpage, patch_file, user side-effect helpers) have already had their args chosen and their side-effect committed; discarding them strands committed work for no benefit — the next `before_model` short-circuit ends the dispatch cleanly anyway (Bug C, v23 validation).

### 13.11 Making a "predictive" pin by intercepting agent state

**WRONG:** reading the agent's internal state at pin time to guess which task it *will* work on.

**RIGHT:** capture the observed fact. `DelegationObserved` carries `tool_args_json` from `_safe_jsonify_tool_args(tool_args)` — the args the agent actually authored — so descriptive growth and any adaptive consumer read the agent-authored event, not a goldfive-side intercept (invariant 4; design §13). The pin ladder resolves from observed delegation + DAG structure, never from a prediction.

### 13.12 Copying code from the agency-preservation branch

**WRONG:** porting a twin-refine-pipeline extraction, evidence-ledger, or checkpoint-rollback helper you saw referenced.

**RIGHT:** those are on the **unmerged** `agency-preservation` branch (#453–#474) behind default-OFF flags, LOCKED on user sign-off (step 13b). Main-side code must not copy from it, and your doc/code must not claim its features exist on main. The ~7 stacked `handle_drift` suppression gates and the twin-refine extraction are known *future* work, not current.

### 13.13 Touching a PROTECTED KEEP surface without sign-off

**WRONG:** deleting `LOOPING_TOOL_CALL` / `LOOPING_REASONING` machinery, the `PLAN_DIVERGENCE` code, or `reconciler.get_missed_tasks` because it "looks dead".

**RIGHT:** these are PROTECTED KEEP decisions (goldfive#204/#206, #252-disabled-but-KEEP, #163). The tool-loop path deliberately emits `LOOPING_REASONING` with NUDGE-first CRITICAL routing. Never delete or "fix" these without explicit human sign-off. If you think one is dead code, flag it in your report — do not delete it.

### 13.14 Landmine quick-reference

One-line reminders for the traps above, for a final pre-commit skim:

| Landmine | The rule |
|----------|----------|
| Top-level ADK import | Never. Lazy + `try/except` inside functions. |
| Bare `getattr` on ADK objects | Use `_safe_attr`. |
| `before_model` returning `None` to skip | Return `_make_cancelled_llm_response()`. |
| Intervention without a gate | `if _is_observation_only(ctx): suppress + PolicyApplied`. |
| Direct `observation_only` read | Route through `_is_observation_only` / `steering_is_active`. |
| Untracked `create_task` | Store handle, cancel on completion + in `clear_active_context`, `name=`, `try/except RuntimeError`. |
| New per-dispatch field | Clear it in `clear_active_context`. |
| Reordering `clear_active_context` | `_invocation_tasks.clear()` and `_cancel_stall_watchdog()` BEFORE `_active_ctx = None`. |
| Rich LLM-visible tool response | Minimal shape only; rich context on sink events. |
| ADK `session.state` as a channel | Use `_active_ctx` / `StateStore`. |
| Keying on an LLM-minted id | Key on `invocation_id` / `function_call_id` / `session.run_id` / task id. |
| Regex for name↔task matching | `stem_token_match`; no regex. |
| Short-circuiting a FunctionTool on cancel | AgentTool only (`_is_agent_tool_dispatch`). |
| Predictive pin (reading agent state) | Capture observed facts (`tool_args_json`). |
| Deleting LOOPING/PLAN_DIVERGENCE/get_missed_tasks | PROTECTED KEEP — needs human sign-off. |
| Copying from agency-preservation branch | Unmerged; do not copy, do not claim on main. |
| Dropping the boundary pair | Verify `close_open_boundaries` covers your new bypass path. |
| Negative-class emit at an early bail | Only emit when the detector actually ran. |
| Renaming a `goldfive.*` state key | External contract; needs a coordinated submodule bump. |
| Removing `__goldfive_adk_plugin__` | Breaks `session_context_from_invocation`. |

---

## 13b. Recipes

Concrete, numbered procedures for the edits weak models most often need here. Follow every step; none is optional.

### Recipe A — Add a new ADK callback (or a new branch to one)

1. Confirm ADK actually invokes the callback name you want (grep the installed `google.adk` for `before_run_callback` etc.; ADK's `BasePlugin` defines the surface). Do NOT invent a callback name.
2. Add the `async def <name>_callback(self, *, ...) -> ...:` inside `_GoldfiveADKPlugin`, matching ADK's kwarg names exactly (`callback_context`, `llm_request`, `tool`, `tool_args`, `tool_context`, `invocation_context`, `event`, `error`, `result`, `llm_response`).
3. First line of the body: `ctx = self._resolve_ctx(<the adk ctx arg>)` then `if ctx is None: return None`.
4. If the callback can intervene: resolve `inv_id` and add the cooperative-cancel checkpoint (the consume/emit pattern from §6.1) at the TOP, before any side-effecting work.
5. Wrap every risky read/emit in `try/except Exception: log.debug(...)`. The callback must not raise into ADK.
6. Respect the return contract (§5): `before_model` returns non-`None` to short-circuit; `before_tool` returns a dict to short-circuit; observation callbacks return `None`.
7. Add a targeted test under `tests/` driving the callback directly with a hand-built `SessionContext` stashed under `SESSION_CONTEXT_STATE_KEY`.

### Recipe B — Add a new intervention surface

1. Write the surface so it does nothing by default and only acts when a config flag is set (numeric-or-`None`, cheap short-circuit when disabled).
2. Gate the *action* on `if _is_observation_only(ctx): <log + _emit_policy_applied_from_plugin(outcome="suppressed") + let the wrapped behavior proceed>` (see §13.1). Telemetry may still fire; the action must not.
3. Emit a `PolicyApplied` on the suppressed branch so the gate is observable (mirror the F3 redirect).
4. Add a test in the `test_observation_only_*` family asserting the surface is a strict no-op under the shipped `observation_only=True` default AND fires under an explicit active-mode steerer.

### Recipe C — Add a new pin signal to the ladder

1. Decide which producer: the agent-turn ladder (`_pin_current_task_id_for_agent`, §6pin.2) or the delegation-site pin (`_pin_delegation_task_id`, §6pin.3).
2. Key ONLY on a structurally-stable id (`invocation_id`, `function_call_id`, `session.run_id`, task id). Never on an LLM-minted value (invariant 6).
3. Write through `_stamp_current_task_id(source="<your_label>", ...)` and add `"<your_label>": BindingSource.<X>` to `_BINDING_SOURCE_BY_LADDER` so the pin carries typed attribution.
4. The signal must return a SINGLE best candidate or fall through — never guess between ties (that is the low-confidence signal's job, and it emits `pin_resolved_low_confidence` so operators see the weakening).
5. Add a case to `tests/test_pin_resolution_ladder.py`.

### Recipe D — Add a new background watcher

1. Store the task handle on a dedicated field (or in `_invocation_llm_pending[inv_id]`). Never leave a `create_task` handle-less (§13.2).
2. Spawn with `asyncio.create_task(coro, name="goldfive_<name>_<id>")` wrapped in `try/except RuntimeError` (no-running-loop harnesses → degrade to off).
3. Cancel it on the paired completion callback AND in `clear_active_context`.
4. If it can intervene on fire, gate the intervention on `_is_observation_only(ctx)` (telemetry fires, action gated — mirror the LLM watcher, §7.1).
5. If it emits drift, stamp `observed_revision_index` from the live plan (goldfive#245) so the dispatch-time staleness gate can drop a stale verdict.
6. Handle `asyncio.CancelledError` by returning cleanly (teardown), re-raising it out of any inner `handle_drift` catch.

### Recipe E — Emit a new sink event from the plugin

1. Add the typed factory to `goldfive/events.py` and the proto (see `12-events-sinks-telemetry.md`); do NOT ship a dict-envelope (the `InvocationCancelled` dict-envelope was explicitly removed).
2. Write a small `async def _emit_<name>(self, ...)` following the shared pattern: resolve `_active_ctx`→`steerer._sinks`; bail if no sinks; compute `run_id`/`session_id`/`seq`; lazy-import the factory; `await emit(sinks, evt)`; swallow failures at DEBUG.
3. Stamp `session_id` (fall back to `run_id`) so harmonograf's session rollup keys correctly.
4. Never let the emit raise into the callback — observability is best-effort.

### Recipe F — Change the LLM-call budget or the token cap

1. These are `make_adk_plugin` kwargs threaded from `AgentConfig` via `goldfive.wrap` — change the default in `make_adk_plugin` AND check what `goldfive.wrap` passes (it overrides with a tighter 120s call timeout).
2. Keep the `<= 0` disable path working (the `if self._llm_call_timeout_ms > 0` / `if self._agent_max_output_tokens > 0` guards).
3. The token cap ratchets DOWN only (`_apply_agent_max_output_tokens_cap` takes the smaller). Do not make it ratchet up.
4. Re-run `tests/test_stall_watchdog.py` and any adapter tests that assert on the budget.

---

## 14. Verification checklist

After touching `_adk_plugin.py`, run these in order. All paths are from repo root `/home/sunil/git/goldfive`.

### 14.1 Lint (must stay clean)

```bash
uv run ruff check goldfive/adapters/_adk_plugin.py
uv run ruff check .
```

`ruff check` must be clean. The repo is NOT `ruff-format`-clean — do NOT run `ruff format` on this file or mass-reformat it.

### 14.2 Targeted test files by subsystem

| You touched... | Run |
|----------------|-----|
| Cancellation / cancel-state / sticky flag | `uv run pytest -q tests/test_cooperative_cancellation.py tests/test_cancel_propagation.py tests/test_cancel_inflight_on_refine.py tests/test_iter11d_cancel_race.py tests/test_cancel_reason.py` |
| Delegation pin / task-id injection | `uv run pytest -q tests/test_delegation_pin.py tests/test_pin_resolution_ladder.py tests/test_pin_versioning.py tests/test_goldfive_session_pin_outer.py` |
| Observation-only gates | `uv run pytest -q tests/test_observation_only.py tests/test_observation_only_strict_passive.py tests/test_observation_only_nudge_gate.py tests/test_observation_only_acks.py` |
| Stall watchdog | `uv run pytest -q tests/test_stall_watchdog.py tests/test_runner_install_revision_stall.py` |
| Reasoning-channel disarm | `uv run pytest -q tests/test_reasoning_channel_disarm_warning.py` |
| Runaway-delegation cap | `uv run pytest -q tests/test_runaway_delegation_cap.py` |
| Tool observations / after_tool | `uv run pytest -q tests/test_adk_plugin_tool_observations.py tests/test_session_recent_tool_observations.py` |
| Adapter wiring / concurrent sessions | `uv run pytest -q tests/test_adk_adapter.py tests/test_adk_adapter_concurrent_sessions.py tests/test_adk_adapter_overlay.py tests/test_adk_adapter_pending_tool_isolation.py` |
| wrap contract | `uv run pytest -q tests/test_wrap_adk.py tests/test_adk_wrap_passthrough.py` |

### 14.3 Full suite

```bash
uv sync --extra dev --extra adk
uv run pytest -q
```

Expect roughly `~2912 passed, ~61 skipped` in ~30s. A drop in passed count or a new failure in a cancellation/pin/observation-only test is the signal your edit broke an invariant.

### 14.4 Grep audits (invariant tripwires)

```bash
# Invariant 5: no direct observation_only reads outside the one accessor.
# The ONLY hits should be inside _is_observation_only, docstrings/comments.
grep -n "observation_only" goldfive/adapters/_adk_plugin.py

# Invariant 5: the sanctioned kill-switch reads.
grep -n "steering_is_active\|_is_observation_only" goldfive/adapters/_adk_plugin.py

# Watcher discipline: every create_task must have a name= and a stored handle.
grep -n "create_task" goldfive/adapters/_adk_plugin.py

# Invariant 1: no top-level ADK import (should return nothing).
grep -nE "^(import google|from google)" goldfive/adapters/_adk_plugin.py

# Duck-typing: bare getattr on ADK objects is suspicious — prefer _safe_attr.
grep -n "getattr(" goldfive/adapters/_adk_plugin.py | grep -iv "_safe_attr"

# Per-dispatch fields must be cleared — eyeball clear_active_context body.
grep -n "def clear_active_context" goldfive/adapters/_adk_plugin.py
```

If `grep -n "observation_only"` shows a new intervention branch reading the flag directly (not via `_is_observation_only`), that is a bug — reroute it. If a `create_task` has no stored handle or is not cancelled in `clear_active_context`, that is an orphan-task bug.

### 14.5 How the tests drive the plugin (so you can write one)

Most plugin tests do NOT spin up a real ADK runner. They construct the plugin via `make_adk_plugin(...)`, build a `SessionContext`, and either (a) call `plugin.set_active_context(ctx)` then drive callbacks directly, or (b) stash the `SessionContext` on a hand-built ADK-state dict under `SESSION_CONTEXT_STATE_KEY` and pass a fake `callback_context` / `tool_context` whose `_invocation_context.session.state` returns that dict. Pattern (a) exercises the live-run resolution path; pattern (b) exercises the unit-test fallback (§2.1).

To assert a callback short-circuited: check its return value (`None` vs a dict vs a synthetic `LlmResponse`). To assert an emit fired: install a recording sink on the steerer's `_sinks` and inspect the events. To assert a pin landed: read `StateStore.for_session(session).pin_current_task()`. To assert cancellation: call `plugin.request_invocation_cancel(invocation_id=..., request=...)` then drive the next callback and assert the short-circuit.

Fakes need only the attributes the callback reads (duck-typing cuts both ways — a stub `tool` with `.name` and `.agent`/`.func` is enough for `before_tool_callback`). Look at `tests/test_delegation_pin.py` and `tests/test_cooperative_cancellation.py` for the canonical fixtures.

### 14.6 State-ownership tripwire (when you touched a state write)

```bash
GOLDFIVE_STRICT_STATE_OWNERSHIP=1 uv run pytest -q tests/test_cancellation_stash_audit.py tests/test_cancellation_stash_tripwire.py
```

A callback that writes goldfive state outside the catalogued keys trips the audit under this flag. See `11-state-ownership.md`.

---

## 14b. History — why the code looks the way it does

Load-bearing PR/issue lineage. When a comment cites one of these, this is the context. (Full canon in `17-invariants-hazards-history.md`.)

| PR/issue | What it did to this file |
|----------|--------------------------|
| #128 | CONFABULATION_RISK classifier in `after_run_callback`. |
| #130 | AgentTool-per-invoke cap (`agent_tool_cap`, `_emit_runaway_delegation_drift`). |
| #143 | GOAL_DRIFT trajectory buffer (`note_agent_activity` / `note_agent_turn`); the idle goal-judge trigger the stall watchdog now produces. |
| #151 | Reconciler `parent_invocation_id` kwarg (the `TypeError`-fallback pattern). |
| #166/#167 | Retired regex NL classifiers; tier-2 uses `stem_token_match`, not regex. |
| #172 | Per-LLM-call instrumentation (`_measure_request_chars`, duration/usage logging). |
| #181/#192/#204 | Tool-loop tracker + acknowledged-success reset gate; LOOPING_REASONING NUDGE-first routing. |
| #191/#195/#241 | Task-id pinning layers; `task_id` hidden from the reporting schema; delegation-site pin. |
| #245 | `observed_revision_index` stamping on every plugin-emitted drift (dispatch-time staleness gate). |
| #250/#252/#253 | Assignee zeroed at parse; minimal LLM-visible payloads (no error-string leaks); structural CAPABILITY_MISMATCH at delegation. |
| #251 | Cooperative cancellation (`_cancel_state`, `request_invocation_cancel`, `InvocationCancelled`). |
| #256 | `agent_max_output_tokens` ratchet-down cap. |
| #263 | Reasoning extraction + content-fallback; the reasoning-channel disarm warning. |
| #264/#265/#266/#268/#405 | The 8-signal pin ladder; the delegation-assignee tiers; pin revision stamp; shared `capability_check` helpers; Rule A false-positive guard. |
| #271 | The big one: SessionContext resolution rework (Phase 2.0), StateStore-backed pins (Phase 2.1), boundary wrapper + StateStore task registry (Phase 3.5), strict-passive `observation_only`, state-audit callback wrapping (Phase 0). |
| #397 | Request-side `ContextEditor`. |
| #420 | `session_run_id` tool-loop bucketing. |
| #423 | `DelegationObserved.tool_args_json`; descriptive-growth absorption of Rule C. |
| #476 | LLM watcher cancel-flag skipped under observation_only; one-shot reasoning-disarm warning. |
| #481 | F3 redirect gated + aligned to `_predecessors_completed`. |
| #484 | Tool-loop name-axis capped at INFO without exact-repeat corroboration. |
| #485 | Canonical `TERMINAL_TASK_STATUSES` (incl. NOT_NEEDED). |
| #487 | Flag-gated stall watchdog + `last_observed_event_at` liveness stamp. |
| #488 | `is_active_steering` / `steering_is_active` single predicate; module test-hook + autouse fixture deleted. |
| #489 | `Runner._abort_turn` helper + `_run_overlay` decomposition (adapter-side). |
| #490/#491 | Dead-code deletion; single LLM-call module `goldfive/_llm.py`. |

### Deferred / not-on-main (do not present as current)

- **Twin-refine-pipeline extraction** and the **evidence-ledger replacement** of the ~7 stacked `handle_drift` suppression gates — blocked on the agency-preservation branch-merge decision.
- **Descriptive-growth default flip** — the `descriptive_growth_enabled` flag defaults `False` on main.
- **Checkpoint-rollback / tool-gating hold / fork-and-judge** — Stage-4, bench-gated.
- The **shared-Runner (7c) convergence** in `docs/design/SHARED-RUNNER-REFACTOR.md` — a proposed alternative to eliminate the dual-Runner split; NOT implemented. The split is current reality.

---

## 15. Cross-references

- `06-adapters-and-instrumentation.md` — the `ADKAdapter` that owns the plugin, calls `set_active_context` / `clear_active_context`, and the `adk_llm_instrumentation` helpers (`_measure_request_chars`, `_apply_agent_max_output_tokens_cap`, `_build_runtime_tools_hint`).
- `09-steering-ladder-and-gates.md` — the steerer this plugin feeds; `handle_drift`, the intervention ladder, F1 directive production, and `is_active_steering`.
- `07-deterministic-drift-detection.md` — `ToolLoopTracker`, `detect_capability_mismatch` Rules A/B/C, the tool-loop name-axis cap (#484).
- `08-llm-judges.md` — the reasoning judges the disarm warning protects, and the idle goal-drift judge the stall watchdog triggers.
- `11-state-ownership.md` — `StateStore`, the pin/pending-delegation storage, the invocation-task registry, and the state-audit tripwire.
- `13-reporting-tools-and-approval.md` — `invoke_tool`, the reporting-tool handlers, and the Flow-B approval bridge (`_await_tool_approval`) this plugin dispatches into.
- `12-events-sinks-telemetry.md` — `DelegationObserved`, `InvocationBoundaryEntered/Exited`, `InvocationCancelled`, `PolicyApplied`, `pin_resolved`, `DriftDetected` — every sink event the plugin emits.
- `17-invariants-hazards-history.md` — the full canon of PROTECTED KEEP decisions and the deferred agency-preservation program.
