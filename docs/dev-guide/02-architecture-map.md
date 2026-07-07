# 02. Architecture Map — The End-to-End Data Flow

## Read this chapter when...

- You are about to touch **any** file that participates in a run and need to know
  who calls you, what you may return, and who is downstream of your edit.
- You changed something and a *different* subsystem broke — this chapter is the
  map that tells you which hop you actually perturbed.
- You need to know whether a call is `await`-inline (on the agent's critical
  path) or fire-and-forget (background task, drained at the run boundary). Get
  this wrong and you either serialize every tool call behind an LLM judge, or you
  leak a coroutine past `RunCompleted`.
- You are tracing a symptom ("the nudge never fired", "the plan reset between
  turns", "the judge verdict arrived after the run ended") back to the exact
  method that owns it.

This is the **orientation spine**. Every other chapter drills into one hop named
here. When a hop has its own chapter, this chapter names it. Cross-references:
`03-runner-and-conversation.md`, `04-executors-and-control.md`,
`05-adk-plugin.md`, `06-adapters-and-instrumentation.md`,
`07-deterministic-drift-detection.md`, `08-llm-judges.md`,
`09-steering-ladder-and-gates.md`, `10-planning-and-revision.md`,
`11-state-ownership.md`, `12-events-sinks-telemetry.md`,
`13-reporting-tools-and-approval.md`, `14-config-reference.md`.

## Files covered

This chapter cites, but does not fully document, the following. Each bullet names
the chapter that owns the deep dive.

| File | Role in the data flow | Owning chapter |
|------|----------------------|----------------|
| `goldfive/convenience.py` | `wrap()` / `run()` construction | this chapter (§1) |
| `goldfive/runner.py` | `Runner.run` → `_run_locked` turn lifecycle | `03` |
| `goldfive/conversation.py` | per-outer-session cross-turn state | `03`, `11` |
| `goldfive/executors/sequential.py` | overlay + legacy per-task loops | `04` |
| `goldfive/executors/_control.py`, `_shared.py` | control-channel draining helpers | `04` |
| `goldfive/control.py` | `ControlChannel` / `ControlMessage` / `ControlKind` | `04` |
| `goldfive/adapters/adk.py` | `ADKAdapter` (invoke, passthrough, cancel boundary) | `06` |
| `goldfive/adapters/_adk_plugin.py` | the `BasePlugin` callback surface | `05` |
| `goldfive/adapters/adk_wrap.py` | `GoldfiveADKAgent` dual runner/BaseAgent bridge | `06` |
| `goldfive/reconciler.py` | `PlanReconciler` observation→task transitions | `10` |
| `goldfive/steerer.py` | `DefaultSteerer` facade, `InterventionLevel`, ladder tables | `09` |
| `goldfive/drift_observer.py` | `DriftObserver` (`observe`, `observe_reasoning`, `handle_drift`) | `07`, `08`, `09` |
| `goldfive/plan_reviser.py` | `PlanReviser` (`_apply_revision`, `observe_refine`, `PlanRevised` emission) | `10` |
| `goldfive/task_state_machine.py` | `TaskStateMachine` (`mark_task_*`, cascade cancel) | `11` |
| `goldfive/protocols.py` | the five `Protocol` seams | this chapter (§ contract table) |
| `goldfive/events.py` | `new_event` / `emit` fan-out | `12` |

## Invariants that bind you here

These are the CANON invariants that this chapter's flow must never violate. If an
edit anywhere on the spine breaks one, the edit is wrong.

1. **No prompt-cooperation contract.** Termination, control, and observability
   must work even if the agent never calls a goldfive reporting tool. The whole
   observation path (`PlanReconciler`, plugin callbacks, `DriftObserver.observe`)
   is *passive* — it watches ADK's own callbacks; it does not require the agent to
   volunteer anything.
2. **No regex/keyword NL classification.** The classifiers on this spine
   (`LLMPlanner.handle_turn`, the reasoning judge, the goal-drift judge) are LLM
   calls or design-away. Exact-equality / hash matching of *structured* data (task
   ids, `(name, args_hash)` tuples, plan revision indices) is allowed and used
   heavily.
3. **Any ADK tree shape.** Coordinator + `AgentTool`, `sub_agents`, single
   `LlmAgent` — all drive through the same overlay `invoke_passthrough`. The
   plugin is installed on *every* per-agent runner (see `wrap(plugins=...)`,
   goldfive#121), not just the root.
4. **Adaptive, not predictive.** The reconciler and observers capture *observed*
   facts (this agent ran, this delegation happened) — they never pre-guess which
   agent will run next.
5. **`observation_only=True` is the production default and is STRICTLY passive.**
   The only sanctioned read of the kill-switch is
   `DefaultSteerer.is_active_steering()` (`goldfive/steerer.py:1339`) or the module
   helper `steering_is_active(steerer)` (`goldfive/steerer.py:107`). Missing /
   `None` / raising → PASSIVE. Never read `_observation_only` directly from a
   consumer.
6. **Lifecycle gates need stable identity keys.** The keys on this spine — the
   per-key `Conversation` map, the in-flight-refine key, the coalescing queue key,
   the verdict watermark key — are all built from stable ids (session id, drift
   kind, task id). Never key a gate on an LLM-minted, churning id.

---

## The 30-second mental model

goldfive is an **overlay**. It does not replace the agent's control loop; it wraps
the agent's ADK tree, lets the tree run its own natural flow via ONE
`invoke_passthrough` call, and watches from the side through ADK's plugin
callbacks. Everything goldfive does decomposes into three verbs:

- **Observe** — plugin callbacks fire on every ADK agent/model/tool boundary;
  they build `ObservedAction` snapshots and feed them to `DriftObserver.observe`
  / `observe_reasoning` and to the `PlanReconciler`.
- **Detect** — deterministic detectors (loop, tool-error, refusal) run inline;
  LLM judges (reasoning-drift, goal-drift, custom) run **fire-and-forget** in
  background tasks bounded by a semaphore.
- **Steer** — when a drift clears the ladder, the steerer routes it to an
  intervention surface (nudge queue, `GOLDFIVE_STEER` control message,
  pause-escalate). Under the default `observation_only=True`, every one of these
  surfaces is gated OFF and stamps decision telemetry instead of acting.

The single async event loop runs all of it. There is no thread pool. Background
work is `asyncio.create_task`, tracked in sets, and drained at the run boundary.

---

## Component diagram

```
                       goldfive.wrap(agent)  [convenience.py]
                                 │  builds, in order:
                                 │  detect_llm → judges → steerer → executor
                                 │  → adapter → reporting-tool selection
                                 │  → dynamic instructions → planner attach
                                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │                          Runner  [runner.py]                        │
     │   run() → per-key asyncio.Lock → _run_locked()                      │
     │   Conversation map (per outer session)   [conversation.py]          │
     └───────────────────────────────────────────────────────────────────┘
        │ goals            │ handle_turn / generate      │ executor.run(...)
        ▼                  ▼                             ▼
  GoalDeriver         Planner (LLMPlanner)        SequentialExecutor(overlay_mode=True)
  [goal_deriver.py]   [planner.py]                [executors/sequential.py]
                                                        │  _run_overlay loop
                                                        │  races:  adapter invoke
                                                        │          ⨯ ControlChannel
                                                        ▼
                                         ADKAdapter.invoke_passthrough  [adapters/adk.py]
                                                        │  ONE inner ADK runner.run_async
                                                        ▼
                          ┌────────────────────────────────────────────────┐
                          │  _GoldfiveADKPlugin (BasePlugin) [_adk_plugin]  │
                          │  before_run / before_agent / before_model /     │
                          │  before_tool / after_tool / after_model /       │
                          │  on_event / after_agent / after_run             │
                          └────────────────────────────────────────────────┘
                            │ observations              │ reasoning text
                            ▼                            ▼
                    PlanReconciler            DriftObserver.observe / observe_reasoning
                    [reconciler.py]           [drift_observer.py]
                    task transitions           │ inline: loop detector
                          │                     │ background: reasoning judge (semaphore)
                          ▼                     ▼
                    TaskStateMachine        DriftObserver.handle_drift
                    [task_state_machine.py]  │ freshness gate → ladder → level
                          │                   ▼
                          │        InterventionLevel   [steerer.py]
                          │        OBSERVE/ABSORB/NUDGE/CANCEL_REINVOKE/
                          │        PAUSE_ESCALATE/TERMINATE
                          │                   │  (all injection gated on
                          │                   │   is_active_steering())
                          │                   ▼
                          │        Planner.refine → PlanReviser._apply_revision
                          │        [planner.py + plan_reviser.py]  → GOLDFIVE_STEER control msg
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                            events.emit(sinks, event_pb)   [events.py]
                                    │  concurrent fan-out, exceptions isolated
                                    ▼
                  EventSink[]   (LoggingSink, JSONL sink → harmonograf / zicato)
```

The dashed truth: **every** arrow into `events.emit` is observability only. A sink
that raises cannot abort a run (`events.emit` collects exceptions with
`return_exceptions=True`, re-raising only `CancelledError` —
`goldfive/events.py:104`). See `12-events-sinks-telemetry.md`.

---

## §1 — `wrap()` construction: what is built, in what order

`goldfive.wrap(agent, **kwargs)` in `goldfive/convenience.py` (the `wrap` function
begins at `convenience.py:119`) is the single assembly point. Read this order once
and you will never again wonder "where does the judge LLM come from" or "why is
the executor in overlay mode". The steps below are in source order.

### 1.1 Dynamic-instruction install (ADK only)

```python
# convenience.py — inside wrap(), first thing after the docstring
if is_adk_agent(agent):
    from goldfive.adapters.adk_llm_instrumentation import (
        install_dynamic_instructions, log_dynamic_instruction_opt_out,
    )
    if dynamic_instruction:
        touched = install_dynamic_instructions(agent)
```

`dynamic_instruction` defaults `True` (goldfive#251). It rewrites every reachable
`LlmAgent.instruction` string into a callable resolver that re-reads the current
task from `session.state` every turn (plan-causal prompting). #477 hardened this:
the resolver preserves ADK `{var}` templating via `inject_session_state`. No-op on
non-ADK trees. Deep dive: `06-adapters-and-instrumentation.md`.

### 1.2 Runtime config resolution and module install

```python
resolved_runtime: RuntimeConfig = runtime if runtime is not None else RuntimeConfig.from_env()
_embed_module.configure(resolved_runtime.embedding)
_reasoning_module.configure(resolved_runtime.reasoning_drift)
_tool_loops_module.configure(resolved_runtime.tool_loops)
```

`RuntimeConfig` (`goldfive/config.py`, goldfive#225) bundles every typed knob:
embedding backend, tool-loop thresholds, reasoning-drift, goal-drift,
`JudgeConfig`, `AgentConfig`, `SteeringConfig`. When the caller passes nothing,
`from_env()` reads `GOLDFIVE_*` env vars so pre-#225 callers are byte-identical.
The three `configure(...)` calls install **process-wide** module state — this is
the documented multi-Runner caveat (two Runners in one process share reasoning-drift
thresholds). See `14-config-reference.md`.

### 1.3 Sinks → ContextEditor → adapter

Sinks resolve first (default `[LoggingSink()]`; explicit `[]` suppresses all) so
the `ContextEditor` (goldfive#397, opt-in via
`SteeringConfig.context_editor_rules`) can emit onto the same fan-out. Then:

```python
adapter = auto_adapter(
    agent,
    plugins=plugins,
    llm_call_timeout_ms=resolved_runtime.agent.call_timeout_ms,
    agent_max_output_tokens=resolved_runtime.agent.max_output_tokens,
    context_editor=context_editor,
)
```

`auto_adapter` (`goldfive/adapters/auto.py`) picks the concrete adapter: an
existing `AgentAdapter` passes through; an ADK `BaseAgent`/`Runner` → `ADKAdapter`;
a Claude SDK factory → `claude.py`; an async `(task, session, tools)` callable →
`CallableAdapter`. The `plugins=` list is installed on **every** per-agent runner
the ADK adapter builds (coordinator + every `AgentTool`/`sub_agent`), not just the
`App(plugins=[...])` root — goldfive#121, invariant 3.

### 1.4 LLM detection and the judge-routing precedence chain

`detect_llm(agent)` (`goldfive/_llm_detect.py`) introspects an ADK tree for a
usable `(call_llm, model_name)` pair. It runs **whenever the caller did not pass
`call_llm=`** — the guard used to be `(planner is None or goal_deriver is None)`,
which caused a silent-disarm bug (judges inert when the caller supplied their own
planner + goal_deriver). Do not re-add that guard.

The judges' callable is resolved by a strict precedence (`convenience.py` around
the "Judge routing" comment):

1. Explicit `wrap(call_llm=...)` — wins outright.
2. `resolved_runtime.judge.base_url` (dedicated judge endpoint) via
   `_build_judge_call_llm` → `make_default_openai_call_llm` (`goldfive/_llm.py`,
   the one LLM-call module after #491).
3. Auto-detected tree LLM (`detect_llm`).

The **planner + goal_deriver** always stay on `resolved_call_llm`; only the two
drift judges may route to `JudgeConfig`. When the judges inherit the tree LLM
(case 3), `wrap` logs a **named-model WARNING** naming the model and the concurrent
judge-call cost so a billed cloud endpoint is visible in logs.

### 1.5 Planner, goal-deriver, executor, steerer

The defaults branch on `judge_only` and on LLM availability:

| Component | `judge_only=False`, LLM present | `judge_only=True` | no LLM |
|-----------|-------------------------------|-------------------|--------|
| planner | `LLMPlanner(call_llm, model)` | `StaticPlanner` (one framing task, `_build_judge_only_planner`) | `PassthroughPlanner` |
| goal_deriver | `LLMGoalDeriver` | `LiteralGoalDeriver` (no LLM call) | `LiteralGoalDeriver` |
| executor | `SequentialExecutor(max_task_invocations=..., overlay_mode=True)` | same | same |

`overlay_mode=True` is the `wrap()` default (goldfive#141). An explicit
`executor=` keeps full control.

The steerer default is `DefaultSteerer(...)` wired with the **judge** callable
(not the planner callable) — see the three-arm `if steerer is not None / elif
judge_call_llm is not None / else` block in `convenience.py`. When no judge
callable is available and `reasoning_drift_mode in ("judge","both")`, `wrap` logs a
WARNING that LLM drift detection is disabled. Deep dive on the steerer:
`09-steering-ladder-and-gates.md`.

### 1.6 Judges installation and the Runner

```python
if judges is not None and disable_judges is not None:
    raise TypeError(...)                      # mutually exclusive
resolved_judges = list(judges) if judges is not None \
    else default_judges(disable=disable_judges)   # goldfive/builtin_judges.py
set_judges = getattr(resolved_steerer, "set_judges", None)
if callable(set_judges):
    set_judges(resolved_judges)               # steerer.py:875
```

Then the `Runner(...)` is constructed (`convenience.py` near the end) with the
resolved components, `control`, `max_task_invocations`, and `drift_self_reporting`.
If the judges were routed through a dedicated `JudgeConfig` endpoint, `wrap`
registers `judge_call_llm.close` as a Runner close-hook so the HTTP session is torn
down on `runner.close()`.

### 1.7 The return-type twist

```python
if is_adk_agent(agent):
    from goldfive.adapters.adk_wrap import GoldfiveADKAgent
    return GoldfiveADKAgent(inner=agent, runner=runner)
return runner
```

For an ADK `BaseAgent`, `wrap` returns a `GoldfiveADKAgent` (`adapters/adk_wrap.py:222`)
— a `BaseAgent` subclass that *also* proxies the Runner surface (`.run`,
`.run_streamed`, `.close`). The same object works both programmatically
(`await wrapped.run(...)`) and as the `root_agent` of an `adk web` app (ADK calls
`_run_async_impl`, which internally drives `self._runner.run_streamed`). The
declared return type stays `Runner` for ergonomics. See
`06-adapters-and-instrumentation.md`.

---

## §2 — A single healthy turn

Trace one `await runner.run("do the thing")` with zero drift. Hops are in call
order; each names its file:symbol.

### 2.1 Entry and per-key serialization — `Runner.run` (`runner.py:519`)

```python
convo_key = self._conversation_key(session_id)      # session_id or ""
async with self._lock_for(convo_key):               # runner.py:499 (per-key asyncio.Lock)
    return await self._run_locked(user_input, context=..., session_id=..., convo_key=...)
```

`session_id` optionally pins the outer ADK session (adk-web passes `ctx.session.id`
so every goldfive event carries the same id as harmonograf spans — goldfive#161).
Two concurrent calls on the **same** key serialize on the lock so the second turn's
prior-plan seeding sees the first turn's post-install plan. Different keys run in
parallel. Everything below is inside `_run_locked` (`runner.py:572`).

### 2.2 Session build and Conversation seed

```python
convo = self._conversation_for(convo_key)           # runner.py:483
session = convo.next_turn_session()                  # fresh run_id; stable conversation_id
if pinned: session.run_id = session_id
```

The per-key `Conversation` (`goldfive/conversation.py`) owns cross-turn state:
`goals`, `completed_results`, `turns`, the wire-sequence cursor, and the prior-plan
stash. Keying the whole `Conversation` by outer-session id (not just the stash)
fixed a cross-session leak (validation v4 Class 1). See `03` and `11`.

### 2.3 `RunStarted` and plan seed

```python
await self._emit_run_started(session, user_input)    # first event of the turn
prior_plan = convo.prior_plan_for(session.id, pinned=pinned)
with channel_processor_active():                      # single-writer guard
    set_session_plan(session, dataclasses.replace(prior_plan, run_id=session.run_id)
                     if prior_plan is not None else Plan.empty(run_id=session.run_id))
_ostate.set_current_plan(session.state, session.plan)
```

`session.plan` is **always non-None** by the time the planner runs (`Plan.empty()`
on the very first turn). `Plan` is frozen (goldfive#247); every mutation is a
`dataclasses.replace` under `channel_processor_active()` / `set_session_plan`. See
`11-state-ownership.md` for the single-writer contract.

### 2.4 Goals — `_resolve_goals` (`runner.py:1703`)

`GoalDeriver.derive(user_input, context=...)` returns `list[Goal]`. Newly-derived
goals are appended to `session.goals` by id, with **collision renumbering** (F9 /
goldfive#322 Layer 4): if the deriver re-emits `g1`, it is `dataclasses.replace`d
to a fresh `gN` so multi-turn sessions accumulate goals instead of dropping them.
When the caller passed `user_input` as `list[Goal]`, derivation is bypassed.
`_ostate.refresh_goals_summary` updates `goldfive.goals_summary`, then
`GoalDerived` is emitted.

### 2.5 Per-turn planning — `_invoke_handle_turn` (`runner.py:1537`)

```python
if self._planner_gate is not None and isinstance(user_input, str) \
        and hasattr(self.planner, "handle_turn"):
    next_plan = await self._invoke_handle_turn(user_input=..., session=..., ...)
    decided = True
```

`Planner.handle_turn` (goldfive#271 Phase 4) is ONE LLM call that both classifies
("does this input warrant a plan change?") and produces the next `Plan` revision.
The classification is *emergent*: a returned `Plan` means change; `None` means
"purely conversational, reuse `session.plan`". This replaced the old regex
short-circuits + separate gate + refine pipeline (invariant 2). See
`10-planning-and-revision.md`.

Fallback to `planner.generate` (`runner.py` `needs_generate_fallback`) fires when
`handle_turn` was skipped, raised, or returned `None` on an empty first-turn seed —
so `PassthroughPlanner` / `StaticPlanner` / non-LLM planners still land a plan.

### 2.6 Install or reuse the plan — `_install_revision` (`runner.py:1583`)

- `next_plan is not None` → `_install_revision` applies it as the next revision
  (`revision_index += 1`, stable `plan.id`) and `PlanRevised` fires. On validator
  rejection the default (`fail_fast_on_revision_rejection=False`) keeps the prior
  plan, emits a `HUMAN_INTERVENTION_REQUIRED` INFO drift for observability, and
  continues. See PLAN-LIFECYCLE §4.5.1.
- `next_plan is None` and there is a real prior plan → conversational turn; reuse
  `session.plan`, set `session._conversational_turn = True`.
- `next_plan is None` and empty seed → `_abort_turn` with "no plan generated".

### 2.7 Register reporting tools, bind everything (steps 5–6d)

In strict source order (`runner.py` after the install block):

1. `self.agent.register_reporting_tools(select_reporting_tools(self.drift_self_reporting))`
   — `drift_self_reporting=False` default registers only the lifecycle subset
   (goldfive#196). See `13-reporting-tools-and-approval.md`.
2. `self.steerer.bind(sinks=..., planner=...)` — wires sinks + planner into the
   steerer so drift handlers can emit and refine.
3. `adapter.bind_steerer(self.steerer)` — without this the plugin's
   `_emit_observability` short-circuits on `SessionContext.steerer is None` and the
   `AgentInvocationStarted` / `DelegationObserved` events never fire.
4. `steerer.bind_adapter(self.agent)` — lets the steerer tag the adapter's next
   cancel with an LLM-actionable reason (goldfive#139).
5. `steerer.bind_control_channel(self._control)` — lets the steerer mint
   `GOLDFIVE_STEER` / `GOLDFIVE_PAUSE_ESCALATE` onto the same channel user
   directives ride. All four bind calls are duck-typed (`getattr` + `callable`) so
   custom steerers/adapters degrade cleanly.

### 2.8 Hand off to the executor — `SequentialExecutor.run` (`sequential.py:325`)

```python
with _state_audit.cancellation_stash_audited("Runner.run.executor_drive"):
    executor_kwargs = dict(plan=session.plan, session=session, adapter=self.agent,
                           steerer=self.steerer, planner=self.planner, sinks=list(self.sinks))
    if self.control is not None: executor_kwargs["control"] = self.control
    if isinstance(user_input, str) and "user_input" in inspect.signature(self.executor.run).parameters:
        executor_kwargs["user_input"] = executor_user_input   # F6 wrap gated by PromptShaper
    outcome = await self.executor.run(**executor_kwargs)
```

The F6 conversational directive is applied **only** through
`PromptShaper.wrap_conversational_input` (`goldfive/prompt_shaper.py`), which is a
no-op under `observation_only=True` — invariant 5. The whole call site is wrapped
in the cancellation-stash audit context (Phase 3.5, goldfive#271); the compliance
marker fires in the `finally`.

### 2.9 Overlay dispatch — `_run_overlay` (`sequential.py:921`)

Because `wrap()` set `overlay_mode=True` and `ADKAdapter` exposes
`invoke_passthrough`, `run()` delegates to `_run_overlay`. It:

1. Builds a `PlanReconciler(session, steerer, host_agent_name)` (`reconciler.py:109`).
2. Enters `while True:` and calls `_race_control` (`sequential.py:1142`), which
   runs ONE `_invoke_passthrough_with_control` — the passthrough invocation raced
   against the `ControlChannel`. `_race_control` returns `(kind, payload)` where
   `kind ∈ {result, cancelled, adapter_error, steer, goldfive_steer, goldfive_pause}`.
3. On `kind == "result"` (the healthy path): `_drain_nudges` finds no queued
   nudge → the loop breaks.
4. `_sweep_unreachable_pending` transitions leftover PENDING tasks to `NOT_NEEDED`
   (goldfive#163 removed the soft follow-up loop that used to re-dispatch them).
5. `_classify_fatal_failure` and `evaluate_goal_predicates` find nothing →
   `_drain_steerer_at_run_boundary(steerer, session)` drains background judges →
   `run_completed_event` is emitted → returns `ExecutionOutcome(success=True, session)`.

See `04-executors-and-control.md` for the full branch table.

### 2.10 Inside the passthrough — `ADKAdapter.invoke_passthrough` (`adapters/adk.py:1866`)

```python
with reentry(ReentryKind.OVERLAY_REPLAY):
    return await self._invoke_internal(task=None, session=session,
                                       new_message=_passthrough_message_parts(user_message),
                                       reconciler=reconciler)
```

The user's message is sent **verbatim** — no task framing, no goldfive jargon — so
a flow-prompted coordinator runs its natural pipeline (invariant 1, 3). `task` is
`None`: there is no single "current task" during a passthrough. `_invoke_internal`
(`adapters/adk.py:1945`) drives ONE inner ADK `runner.run_async` stream and is the
canonical `CancelledError` boundary catch site (Phase 3.5, `adapters/adk.py:2104`).

### 2.11 Observation — the plugin callback surface (`_adk_plugin.py`)

While the inner ADK runner streams, `_GoldfiveADKPlugin` (a `BasePlugin`,
`_adk_plugin.py:2060`) fires on every boundary. In rough temporal order:

| Callback | `_adk_plugin.py` | What it feeds |
|----------|------------------|---------------|
| `before_run_callback` | `:2648` | state-protocol writes onto `session.state` |
| `before_agent_callback` | `:2777` | opens a boundary; reconciler `on_before_agent`; capability checks |
| `before_model_callback` | `:5575` | ContextEditor request-side edits; per-LLM-call watchdog stash; structural `max_output_tokens` |
| `before_tool_callback` | `:5842` | tool-loop tracking; pre-dispatch redirect (F3, gated #481) |
| `after_tool_callback` | `:6789` | reconciler + tool-loop corroboration |
| `on_tool_error_callback` | `:6727` | `TOOL_ERROR` classification |
| `after_model_callback` | `:6453` | **reasoning extraction + observe** (see §3) |
| `on_event_callback` | `:6702` | delegation observation |
| `after_agent_callback` | `:3901` | closes the boundary; reconciler `on_after_agent` |
| `after_run_callback` | `:3962` | `CONFABULATION_RISK` check; boundary teardown |

`after_model_callback` builds an `llm_response` `ObservedAction` and calls
`await ctx.steerer.drift.observe(observation, ctx.session)` inline
(`_adk_plugin.py:6589`). Every callback short-circuits on
`ctx is None or ctx.steerer is None`. See `05-adk-plugin.md`.

### 2.12 Reconciler → task transitions (`reconciler.py`)

`PlanReconciler.on_before_agent` / `on_after_agent` / `on_delegation_observed`
(`reconciler.py:160/286/352`) map observed agent activity back to plan tasks and
call `steerer.tasks.mark_task_*` (via `steerer.transition`) to advance them —
adaptively, from what actually ran (invariant 4). `get_missed_tasks`
(`reconciler.py:407`) is a PROTECTED KEEP surface (goldfive#163) — do not delete.

### 2.13 Events out

Every `mark_task_*`, every drift, and the terminal `RunCompleted` route through
`events.emit(sinks, event_pb)` (`events.py:104`). Fan-out is concurrent per sink;
a raising sink is logged and dropped for that sink only. See `12`.

### 2.14 Turn teardown (still inside `_run_locked`)

```python
finally:
    if session.plan is not None and session.plan.tasks:
        convo.stash_plan(session, pinned=pinned)     # for next turn's handle_turn seed
    _state_audit.mark_stash_completed()
_ostate.clear_current_task(session.state)
_ostate.clear_active_steer(session.state)
convo.absorb_turn(outcome, user_input_summary=..., pinned=pinned)
return outcome
```

The stash lives in `finally` so it lands even when ADK cancels the runner
mid-stream (`CancelledError` is a `BaseException`, not caught by `except
Exception`). `absorb_turn` folds goals / completed_results / stash into the
`Conversation`. This is the run boundary for one turn — the *Runner* is not closed;
`runner.close()` is the separate process-teardown path (§4).

---

## §3 — A drifting turn

Same turn, but now the agent's chain-of-thought loops or wanders. Trace the drift
from the thinking block to the intervention surface.

### 3.1 Reasoning extraction — `after_model_callback` (`_adk_plugin.py:6453`)

```python
reasoning, reasoning_source = _choose_reasoning_text(llm_response, fallback_enabled=_fallback_enabled)
...
await ctx.steerer.drift.observe(observation, ctx.session)     # llm_response observation, inline
if reasoning and not (inv_id and self.is_invocation_cancelled(inv_id)):
    await observe_reasoning(reasoning, task=ctx.task, session=ctx.session,
                            provider=_infer_provider(llm_response), agent_name=reasoning_agent_name)
```

`_choose_reasoning_text` pulls the real chain-of-thought on thinking-capable models
(OpenAI `reasoning_content`, Anthropic thinking blocks, Google thought parts). On
non-thinking models the opt-in
`ReasoningDriftConfig.fallback_to_content_when_no_reasoning` (default `False`,
goldfive#263) synthesizes a signal from the body. Cancelled invocations skip
`observe_reasoning` — judging zombie reasoning wastes LLM calls and produces
spurious drift.

### 3.2 `observe_reasoning` — inline detector + fire-and-forget judge (`drift_observer.py:1842`)

```python
session.mark_reasoning_turn()                       # #441 logical-turn counter (freshness)
history.append(text); trim to reasoning_history_max
self._dispatch_custom_judges(text=text, session=session, agent_name=agent_name)   # #437, fire-and-forget
drift = detect_looping_reasoning(text, session)     # ALWAYS-ON, cheap, INLINE
if drift is not None:
    await self.handle_drift(drift, session); return  # short-circuits the judge
```

The **always-on loop detector** runs inline and its verdict can affect the current
turn. It short-circuits the LLM judge. If it does not fire, the mode-selected
pipeline schedules a judge:

```python
rl_call_llm = self._maybe_take_reasoning_judge_slot(session, agent_name=agent_name)  # rate limit
if mode == "off": return
if mode == "judge" and rl_call_llm is None: return   # rate-limited or globally disabled
pinned_history = list(session.reasoning_history)      # snapshot at schedule time
queue_key = (session.id, agent_name, session.current_task_id)
# COALESCE onto a still-QUEUED window for the same key; newest wins, granted slot never downgraded
bg_task = asyncio.create_task(self._run_judge_background(...),
                              name=f"goldfive-reasoning-judge:{session.id}")
self._steerer._background_judges.add(bg_task)
bg_task.add_done_callback(self._steerer._background_judges.discard)
```

**Why fire-and-forget** (`observe_reasoning` docstring): this runs on the ADK
model-response callback, which is on the critical path for tool dispatch. Awaiting a
minute-long local-llama judge inline would serialize every subsequent tool call
behind it. The consequence — a judge verdict can arrive AFTER same-turn tool calls
have dispatched — is handled by the supersede machinery downstream; late refines
just apply to an already-advanced plan. See `08-llm-judges.md`.

### 3.3 Judge scheduling guards (#483)

- **Semaphore** — `_run_judge_background` (`drift_observer.py:2081`) waits on
  `self._judge_semaphore` (`drift_observer.py:434`, default `max_concurrent_judges=3`,
  clamped `>= 1`) so at most N background judge calls run at once.
- **Coalescing** — while a request is QUEUED (task created, semaphore not yet
  acquired), a newer observation for the same `queue_key` folds into it
  (`_QueuedJudgeWindow`, `drift_observer.py:184`) instead of scheduling a second
  task. On semaphore acquire the entry is deleted (QUEUED → RUNNING) before the
  payload is read, so a still-newer observation schedules fresh.
- **Verdict-utility ledger** — `_verdict_ledger(session)` (`drift_observer.py:2358`)
  counts `{acted_on, emitted_late, emitted_redundant, parse_fail}`, summarized as a
  `reasoning_judge_utility_summary` event at the run boundary (§4).
- **Malformed severity → INFO** (#479): a judge that returns a bad severity is
  downgraded to INFO, not dropped.

### 3.4 The judge fires a drift → `handle_drift` (`drift_observer.py:3690`)

`handle_drift` is "the single most-fixed method in the codebase." It is a thin
guard wrapper around `_handle_drift_dispatch` (`drift_observer.py:3855`). Guards, in
order:

1. **PLAN_DIVERGENCE drop** — disabled at the top (goldfive#252); external
   producers cannot revive it. (The *machinery* is a PROTECTED KEEP, just gated
   off.)
2. **`authored_by` normalization** — USER_* → `"user"`, else `"goldfive"`.
3. **Verdict-freshness gate** (goldfive#245, #480 telemetry) — every observation
   stamps `observed_revision_index` before its LLM await. A verdict whose
   `(kind, current_task_id)` was already addressed at a **later** revision
   (`last_addressed_revision_by_drift_key`) is dropped as redundant: emit
   `DriftDetected` with `decision_outcome="drift_dropped_stale"`, bump
   `emitted_redundant`, return. User-authored drifts and unstamped
   (`observed_revision_index == 0`) drifts bypass the gate.
4. **In-flight-refine key** (goldfive#405) — a second concurrent judge on the same
   `(session.id, kind.value, current_task_id)` sees the in-flight entry and skips
   with `decision_outcome="drift_dropped_inflight"`. The key is added
   synchronously, cleared in the `finally`. **This is a stable-key lifecycle gate —
   invariant 6.**

### 3.5 Dispatch — `_handle_drift_dispatch` → ladder (`drift_observer.py:3855`)

```python
self._tag_adapter_cancel_reason(drift, session=session)          # #139: LLM-actionable cancel text
if drift.kind is USER_STEER: await self._apply_user_steer_state(drift, session)
promote_to_steer = self._should_promote_to_steer(drift, session) # severity + fresh-user-steer suppression
await self._emit_drift_detected(session, drift)
if drift.suppressed_by_user_steer: return                        # a user steer is already active
if self._should_request_cancel_for_drift(drift):                 # CRITICAL-only cooperative cancel
    self._cancelled_drift_ids.add(drift_id)
    await self.request_invocation_cancel(drift=drift, session=session)   # GATED on is_active_steering
if promote_to_steer: await self._promote_drift_to_steer(drift, session); return
occurrence_count = self._occurrence_count_for_ladder(session, drift)
level = self._ladder_level_for(drift.kind, drift.severity, occurrence_count)   # :3656
await self._emit_ladder_transition(...)
```

`_ladder_level_for` (`drift_observer.py:3656`) reads the per-kind `_LADDER` table
and the drift severity, plus `is_repeat = occurrence_count >= REFINE_FAILURE_THRESHOLD`
(2). `InterventionLevel` (`steerer.py:149`) is the ordered ladder:
`OBSERVE(0) < ABSORB(1) < NUDGE(2) < CANCEL_REINVOKE(3) < PAUSE_ESCALATE(4) < TERMINATE(5)`.
The dispatch tail:

```python
if level is OBSERVE: return
if level is NUDGE: await self._dispatch_nudge(drift, session); return
if level is PAUSE_ESCALATE: await self._dispatch_pause_escalate(drift, session); return
if level is TERMINATE: await self._dispatch_pause_escalate(drift, session, terminate=True); return
# ABSORB and CANCEL_REINVOKE both call planner.refine + _apply_revision:
#   outcome gate (refine_outcomes: skip if succeeded / >= threshold failed)
#   progress-stall escalation (PROGRESS_STALL_THRESHOLD_SECONDS → HUMAN_INTERVENTION_REQUIRED)
#   planner.refine(...)  →  _apply_revision  →  PlanRevised
#   CANCEL_REINVOKE additionally: _dispatch_goldfive_steer_control (GOLDFIVE_STEER msg)
```

`TERMINATE` (Level 5, #482) is a pause-with-deadline: same channel as
`PAUSE_ESCALATE` but the payload always carries a `deadline_s` (configured
`pause_escalate_deadline_s`, or the 600s built-in `DEFAULT_TERMINATE_PAUSE_DEADLINE_S`)
so the executor's pause wait aborts the run (`RunAborted` with escalation lineage)
instead of blocking forever — pre-#482 this silently degraded into another
`PAUSE_ESCALATE`. Deep dive on the ladder tables, promotion policy, and the
protected `LOOPING_TOOL_CALL` / `LOOPING_REASONING` routing (#204/#206):
`09-steering-ladder-and-gates.md`.

### 3.6 The `observation_only` gate — where every injection stops

Under the production default `observation_only=True`, detection, refine, and
`PlanRevised` (with `dry_run=True`) all still run, but every surface that would
*touch the live invocation* is gated. Each gate reads `is_active_steering()` and,
when passive, logs the would-be action at INFO and stamps an
`observation_only_gate` policy event:

| Injection surface | `drift_observer.py` | PR |
|-------------------|---------------------|----|
| `_dispatch_nudge` (nudge enqueue) | `:4416` (`if not self._steerer.is_active_steering()`) | #475 |
| `_dispatch_goldfive_steer_control` (GOLDFIVE_STEER msg) | `:4465` | #254 |
| `request_invocation_cancel` (plugin cancel flag) | (cancel path) | #254 |
| plan mutation in `PlanReviser._apply_revision` | `plan_reviser.py` | #254 |
| F1/F3/F6 prompt-shape + pre-dispatch gates | via `steering_is_active` | #481 |

`is_active_steering()` (`steerer.py:1339`) returns `not self._observation_only`.
**Consumers that hold a maybe-steerer** (executors, plugin, prompt shaper,
reporting acks) must go through `steering_is_active(steerer)` (`steerer.py:107`),
which returns `False` for `None`/missing/raising — the fail-safe direction
(invariant 5). #488 collapsed these to the single predicate + module helper and
deleted the module-global test hook and autouse fixture, so the suite now runs the
shipped `observation_only=True` default (~90 tests explicitly opt into active mode).

### 3.7 Executor replay/restart (only when active)

When steering IS active, a `GOLDFIVE_STEER` control message lands on the
`ControlChannel`. `_race_control` returns `kind == "goldfive_steer"`;
`_restart_after_goldfive_steer` (`sequential.py:1365`) supersedes the stuck task and
re-invokes the passthrough with the corrective body. A queued nudge is drained by
`_drain_nudges` (`sequential.py:1498`) on `kind == "result"`, capped at
`_MAX_NUDGE_REPLAYS = 3` (goldfive#202) so a pathological nudge-queueing drift cannot
re-introduce the #163 amplification. CANCEL vs supersede discrimination lives on
`session._supersede_pending`. See `04-executors-and-control.md`.

---

## §4 — Teardown

Two teardown scopes: the **run boundary** (per turn) and **runner close** (process
/ conversation end). Do not conflate them.

### 4.1 Run boundary — `_drain_steerer_at_run_boundary` (`sequential.py:107`)

Called by `_run_overlay` immediately before **every** terminal
`run_completed_event` / `run_aborted_event`. It delegates to
`DriftObserver.drain_session_background_tasks(session_id=session.id)`
(`drift_observer.py:2469`), which:

1. Filters `_background_drifts` / `_background_judges` by the `:session_id` suffix
   on each task's name (goldfive#243) — draining ONLY the terminating run's tasks,
   leaving other concurrent sessions alone. **Refuses to drain when `session_id` is
   empty** (that would match everything).
2. Bounded-waits (default 2.0s) then cancels stragglers.
3. Emits the `reasoning_judge_utility_summary` via `_emit_verdict_utility_summary`
   (`drift_observer.py:2392`) — the `{acted_on, emitted_late, emitted_redundant,
   parse_fail}` ledger, popped and summarized **before** the terminal RunAborted/
   RunCompleted so the summary rides inside the run.

This is why a drift cascade dispatched at the end of turn N (e.g. a refine from a
`report_*` tool) does not outlive turn N and burn compute on a dead session.

### 4.2 Stall watchdog — the `TASK_TIMEOUT` producer (#487)

Flag-gated (`SteeringConfig.stall_watchdog_enabled`, default `False`;
`stall_timeout_s` default 600.0, `config.py:775`). `DriftObserver._stamp_last_observed`
(`drift_observer.py:1763`) refreshes `Session.last_observed_event_at` on every
observed event. When enabled and the watermark goes silent past `stall_timeout_s`,
the watchdog fires `TASK_TIMEOUT` (WARNING → CRITICAL on further multiples). The
idle goal-judge trigger consumes `GOAL_DRIFT_IDLE_SECONDS` (300,
`drift/goals.py:84`). See `07-deterministic-drift-detection.md`.

### 4.3 Runner close — `Runner.close` (`runner.py:1406`)

Idempotent (`self._closed`). In order:

1. For every announced `Conversation` slot: `_audit_conversation_pending_at_close`
   cancels orphan PENDING tasks (goldfive#212), then `ConversationEnded` is emitted.
2. `steerer.drift.shutdown()` (`DriftObserver.shutdown`, `drift_observer.py:2438`) —
   drains remaining background judge/drift tasks (bounded, default 5.0s) and flushes
   any verdict ledger whose session never hit a run boundary. `runner.close` reaches
   it via a duck-typed `getattr(self.steerer, "shutdown", None)`; because
   `DefaultSteerer` exposes no top-level `shutdown`, on the default path this step is
   currently a no-op and the effective drain is the per-run
   `steerer.drift.drain_session_background_tasks` (§4.1).
3. Every sink's `close()` (exceptions logged, never fatal).
4. `maybe_close_call_llm` on `planner._call_llm` and `goal_deriver._call_llm` (SDK
   clients own aiohttp sessions that leak otherwise).
5. Registered close-hooks in order (including the `JudgeConfig` client close from
   §1.6). A raising hook is logged and does not block the rest.

---

## §3b — The other drift path: trajectory-level GOAL_DRIFT

§3 traced the *per-thinking-message* reasoning judge. There is a second,
orthogonal judge on the spine: the **trajectory-level GOAL_DRIFT judge**
(goldfive#143/#218). It asks a different question — "is the tree as a whole still
progressing toward `session.goals`?" — and it is triggered on two clocks, not on
reasoning blocks.

### 3b.1 Turn-counter trigger — `note_agent_turn` (`drift_observer.py:3142`)

The ADK plugin's `after_run_callback` (`_adk_plugin.py:3962`) calls
`note_agent_turn(session)` once per completed agent invocation.

```python
if self._steerer._goal_drift_call_llm is None: return    # no-op unless configured
session._agent_turns_since_goal_check += 1
if session._agent_turns_since_goal_check < self._steerer._goal_drift_check_interval:
    return
session._agent_turns_since_goal_check = 0                 # reset BEFORE spawn (no double-fire)
self._spawn_goal_drift_judge_background(session)          # fire-and-forget
```

The counter is **trajectory-level** — unlike the reasoning-judge rate-limit bucket,
it is NOT reset on task transitions, because GOAL_DRIFT is about the whole tree's
direction, not one task's progress. The callable comes from `wrap`'s judge
precedence (§1.4); a bare `DefaultSteerer()` with no judge callable no-ops here.

### 3b.2 Task-boundary trigger — `_maybe_run_goal_drift_on_task_boundary` (`drift_observer.py:3179`)

Task completions / failures / cancellations are natural "am I still on plan?"
checkpoints, so the judge also fires on task transitions (goldfive#219) — short
pipelines that finish before `goal_drift_check_interval` turns would otherwise never
trigger it. Rate-limited by `_GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` (two
transitions within the window fire only once) and resets the turn counter so a
boundary landing exactly on the interval does not pay for two back-to-back calls.

### 3b.3 Why fire-and-forget here too (the v22 regression)

`_spawn_goal_drift_judge_background` (`drift_observer.py:3311`) →
`_run_goal_drift_judge_background` (`drift_observer.py:3375`) is dispatched as a
tracked background task (`name=f"goldfive-goal-drift-judge:{session.id}"`, added to
`_background_judges`). The reason is subtle: `after_run_callback` runs *on the
agent's invocation task* — the same cancellable scope a sibling drift can target via
`request_invocation_cancel`. An inline await on the judge would die when that
invocation is cancelled (the v22 `judge_goal_drift` span vanished exactly this way).
When the judge produces a verdict it routes into the same `handle_drift`
(`drift_observer.py:3690`) → ladder → `observation_only` gate as every other drift.
GOAL_DRIFT resolves only at a task-terminal transition — it emits
`DRIFT_LIFECYCLE_RESOLVED` there (#486). Deep dive: `08-llm-judges.md`.

### 3b.4 The idle trigger (watchdog-gated)

When `stall_watchdog_enabled=True` (§4.2), an idle goal-judge trigger also fires
after `GOAL_DRIFT_IDLE_SECONDS` (300s, `drift/goals.py:84`) of observed silence.
This is off by default — do not assume it runs in production.

---

## §5 — Contrast: the legacy per-task loop

`wrap()` always builds `overlay_mode=True`, so §2/§3 traced the overlay path. But
`SequentialExecutor` has a **second** loop — the legacy per-task loop
(`sequential.py:387` onward, reached when `overlay_mode=False` OR the adapter lacks
`invoke_passthrough`). You will meet it in tests and in custom-executor callers.
Know how it differs so you do not accidentally port overlay assumptions into it.

| Aspect | Overlay loop (`_run_overlay`) | Legacy per-task loop (`run`, `overlay_mode=False`) |
|--------|------------------------------|---------------------------------------------------|
| Dispatch | ONE `invoke_passthrough(user_input)` | `_pick_next_task` → `adapter.invoke(task)` per eligible task |
| Who drives task order | the agent's own natural flow; reconciler observes | the executor, via `_pick_next_task` topological walk |
| PENDING at end | swept to `NOT_NEEDED` | naturally exhausted by the walk |
| Plan revisions | picked up by re-invoke on `goldfive_steer` | detected by plan-id/revision-index swap → `PlanRevised` re-scan |
| Invocation cap | nudge-replay cap `_MAX_NUDGE_REPLAYS` | `max_task_invocations` + per-lineage `max_retries_per_task_lineage` (3) |
| Control channel | `_race_control` per invocation | `_apply_pre_task_controls` between tasks (`sequential.py:1831`) |

The per-task loop's runaway guards are the `max_task_invocations` ceiling (whole
run) and the per-lineage cap (`_lineage_root` strips `retry_`/`retryN_` prefixes so
`t0`, `retry_t0`, `retry2_retry_t0` share the root `t0`; cap 3). When a lineage cap
trips, the task is marked `FAILED` **without invoking the adapter**. Both loops share
the terminal-status set (`TERMINAL_TASK_STATUSES`, canonical home
`goldfive/types.py:106`, promoted to shared executor helpers in #485) and the
shared executor helpers in `executors/_shared.py`; the parallel scheduler
(`executors/parallel.py`) skips terminal tasks (including `NOT_NEEDED`) using the
same canonical set. Before #485 each executor re-derived its own terminal check —
if you add a new terminal status, add it to the one frozenset in `types.py`, never
to a local copy. See `04-executors-and-control.md`.

---

## §6 — Reading one run's event stream

The fastest way to verify the spine is intact is to read the events a run emits, in
order. Every event flows through `events.emit` (`events.py:104`) to the sinks; the
`LoggingSink` prints them, the JSONL sink feeds harmonograf/zicato. A healthy
single-turn overlay run emits, roughly:

```
ConversationStarted        (first turn per key; runner.py:_emit_conversation_started)
RunStarted                 (runner.py:_emit_run_started)
GoalDerived                (runner.py:_emit_goal_derived)
PlanRevised                (first plan install; revision_index=1)
AgentInvocationStarted     (plugin _emit_observability, needs bind_steerer)
  DelegationObserved       (per AgentTool/sub_agent hop)
  ReasoningJudgeInvoked     (per judge call; proto fields 12-15 focused_task_id/
                             focus_confidence/stated_intent/provenance, #480)
  TaskTransitioned...       (reconciler → mark_task_*)
AgentInvocationCompleted
reasoning_judge_utility_summary   (drain_session_background_tasks, #483)
RunCompleted               (or RunAborted with reason)
ConversationEnded          (only at runner.close())
```

When drift fires, interleaved you also see:

```
DriftDetected              (always, even when suppressed/gated — with decision_outcome)
LadderTransition           (to_level=observe/absorb/nudge/... )
PolicyApplied              (observation_only_gate / refine_failure_threshold / ...)
RefineAttempted → RefineSucceeded|RefineFailed   (ABSORB/CANCEL_REINVOKE only)
PlanRevised                (dry_run=True under observation_only)
DRIFT_LIFECYCLE_RESOLVED   (#486, at task-terminal or staleness-guarded on-task)
```

### Decision-telemetry outcomes you will see (and what they mean)

These string outcomes ride on `DriftDetected` / `PolicyApplied` and are how you
diagnose "the drift fired but nothing happened":

| Outcome / policy | Producer | Means |
|------------------|----------|-------|
| `drift_dropped_stale` | freshness gate (`handle_drift`) | verdict observed at an already-addressed revision |
| `drift_dropped_inflight` | in-flight-refine key | a concurrent refine for the same `(kind, target)` is running |
| `observation_only_gate` | every injection surface | `observation_only=True` — action logged, not taken |
| `refine_outcome_succeeded_skip` | outcome gate | a prior refine already fixed this `(kind, task)` this turn |
| `refine_failure_threshold` | outcome gate | `>= REFINE_FAILURE_THRESHOLD` (2) failures — task marked FAILED |
| tool-loop `severity_capped_from` | #484 | name-axis loop capped at INFO without exact `(name,args_hash)` corroboration |

`DriftEvent.detector_name` (#480) tells you which detector/judge produced the drift.
See `12-events-sinks-telemetry.md` for the full event catalogue and proto field
map.

---

## §7 — Control channel: the one junction for all steering

Both user directives and goldfive-authored interventions ride ONE
`ControlChannel` (`goldfive/control.py`) so the executor's invoke loop has a single
place to consult. `ControlKind` (`control.py`):

| Kind | Origin | Payload | Handled by |
|------|--------|---------|-----------|
| `PAUSE` / `RESUME` / `CANCEL` | external (UI/CLI/test) | — | `_apply_pre_task_controls` / `_race_control` |
| `STEER` | external | `{note, suggested_action}` | `apply_steer` → USER_STEER drift → refine → restart |
| `REWIND_TO` | external | `{task_id}` | executor rewind |
| `APPROVE` / `REJECT` | external | `{target_id, detail}` | reporting-approval (`13`) |
| `INTERCEPT_TRANSFER` / `INJECT_MESSAGE` | external | varies | plugin/executor |
| `GOLDFIVE_STEER` | **goldfive-internal** | `{drift_kind, drift_id, body, superseded_task_ids, replacement_task_ids}` | `_restart_after_goldfive_steer` |
| `GOLDFIVE_PAUSE_ESCALATE` | **goldfive-internal** | `{reason, drift_id}` (+ `deadline_s` for TERMINATE) | `_handle_goldfive_pause` |

The two `GOLDFIVE_*` kinds are minted by the steerer
(`_dispatch_goldfive_steer_control` / `_dispatch_goldfive_pause_control`) so
autonomous drift takes the *same* cancel-and-restart junction as an operator
`STEER`. External bridges must NOT originate the `GOLDFIVE_*` kinds — they encode a
goldfive-side decision. `ControlKind` stays in lockstep with the proto enum
(`proto/goldfive/v1/control.proto`); `tests/test_control_proto.py` enforces both
directions. See `04-executors-and-control.md`.

---

## The cross-component contract table

Every arrow on the spine, its direction, sync/async, and what may be `None`. When
you edit a producer, this tells you what the consumer is allowed to assume; when
you edit a consumer, it tells you what you must tolerate. Protocol seams are in
`goldfive/protocols.py`.

| Caller → Callee | Method / seam | Sync/async | May be `None` / absent? | Notes |
|-----------------|---------------|-----------|--------------------------|-------|
| user → `Runner` | `run(user_input, *, context, session_id)` | async | `context`, `session_id` optional | per-key `asyncio.Lock` serializes same-key calls |
| `Runner` → `GoalDeriver` | `derive(user_input, *, context)` | async | `context` optional; returns `list[Goal]` (never `None`) | `runner.py:1703` |
| `Runner` → `Planner` | `handle_turn(*, user_input, session, conversation_history, available_agents, context)` | async | **optional method** (probed via `hasattr`); returns `Plan \| None` | `None` ⇒ reuse `session.plan` |
| `Runner` → `Planner` | `generate(*, goals, available_agents, context)` | async | `available_agents`/`context` may be `None`; returns `Plan \| None` | fallback path |
| `DriftObserver` → `Planner` | `refine(*, plan, drift, goals, observed_actions?, available_agents?)` | async | optional kwargs; returns `Plan \| None` | `_planner_refine_accepts_available_agents` probes the signature |
| `Runner` → `Executor` | `run(*, plan, session, adapter, steerer, planner, sinks, control?, user_input?)` | async | `control`, `user_input` optional (probed via `inspect.signature`) | returns `ExecutionOutcome` |
| `Runner`/`Executor` → `Steerer` | `bind(*, sinks, planner)` | sync | required | idempotent; executor may re-bind |
| `Runner` → adapter | `bind_steerer(steerer)` | sync | **duck-typed** (`getattr`) | without it, obs events don't fire |
| `Runner` → `Steerer` | `bind_adapter(adapter)`, `bind_control_channel(channel)` | sync | **duck-typed**; `channel` may be `None` | #139, path-duality |
| `Runner` → adapter | `register_reporting_tools(list[ReportingToolSpec])` | async | list may be empty | `runner.py` step 5 |
| `Executor` → adapter | `invoke_passthrough(user_message, *, session, reconciler?, ctx?)` | async | `reconciler`/`ctx` optional; `task` is `None` | overlay path |
| `Executor` ⇄ `ControlChannel` | `receive(timeout?)` / `ack(ack)` | async | `receive` returns `None` on timeout/closed | `control.py` |
| adapter plugin → `Steerer.drift` | `observe(observation, session)` | async | `ctx.steerer` may be `None` (callback returns) | `_adk_plugin.py:6589` |
| adapter plugin → `Steerer.drift` | `observe_reasoning(text, *, task?, session, provider?, agent_name?)` | async | `task` may be `None`; empty `text` no-ops | schedules bg judge |
| adapter plugin → `PlanReconciler` | `on_before_agent` / `on_after_agent` / `on_delegation_observed` | async | reconciler may be `None` in some paths | adaptive task transitions |
| `DriftObserver` → `Planner.refine` → `Steerer.plans` | `planner.refine(...)` → `_apply_revision` → `_emit_plan_revised` (refine is the Planner's; `PlanReviser` applies the revision) | async | `planner`/`session.plan` `None` ⇒ early return | mutation gated by `is_active_steering` |
| `DriftObserver` → `Steerer.tasks` | `transition` → `mark_task_*` | async | — | single writer via `channel_processor_active` |
| any → `is_active_steering()` | `DefaultSteerer.is_active_steering()` | sync | consumers holding maybe-steerer use `steering_is_active(steerer)` | `None`/raising ⇒ `False` |
| any → sinks | `events.emit(sinks, event_pb)` | async | empty list no-ops | one sink raising ≠ run abort |
| `Steerer`/`DriftObserver` → `ControlChannel` | mint `GOLDFIVE_STEER` / `GOLDFIVE_PAUSE_ESCALATE` | async | channel may be unbound (`False` landed) | best-effort |

### The five Protocol seams (`goldfive/protocols.py`)

- `GoalDeriver.derive` — `runtime_checkable`, one async method.
- `Planner` — `generate` + `refine` required; `handle_turn` optional (`hasattr`).
- `Steerer` — a thin facade with three sub-objects: `tasks`
  (`TaskStateMachine`), `plans` (`PlanReviser`), `drift` (`DriftObserver`).
  Callers reach `steerer.tasks.X` / `steerer.plans.X` / `steerer.drift.X`
  directly (goldfive#410 facade cleanup). Plus `is_active_steering`, `transition`,
  `bind`.
- `AgentAdapter` — `register_reporting_tools`, `invoke`, `emit_reasoning`,
  `available_agents`. Note `available_agents_tree` (goldfive#151) is deliberately
  NOT in the Protocol so legacy adapters still `isinstance`-check; call sites use
  `getattr` + fall back to the flat list.
- `Executor` — `run`. `EventSink` — `emit` + `close`.

---

## Threading and async model

There is **one asyncio event loop**. No threads, no process pool. Every hop above
is a coroutine on that loop. The concurrency primitives are:

1. **Per-key `asyncio.Lock`** (`Runner._convo_locks`, `runner.py:499`) — serializes
   same-outer-session `run()` calls; different keys run concurrently. The
   `setdefault(key, asyncio.Lock())` is atomic under asyncio's single-thread model,
   so two concurrent first-time lookups land on the same Lock.
2. **Fire-and-forget background sets** — `self._steerer._background_judges` and
   `_background_drifts`. Tasks are created via `asyncio.create_task(..., name=
   f"goldfive-<kind>:{session.id}")`, added to the set, and self-removed via
   `add_done_callback(...discard)`. The `:{session.id}` suffix is load-bearing:
   `drain_session_background_tasks` filters by it (§4.1). **Never spawn a background
   task without the session-id suffix in its name** — the drain will either miss it
   (leak) or refuse to drain (empty session id).
3. **Judge concurrency semaphore** (`_judge_semaphore`, `drift_observer.py:434`,
   default 3) + the QUEUED-window coalescing map (#483). Bounds background judge
   fan-out so ~N calls hit the (possibly shared) judge endpoint at once. An
   endpoint-contention WARNING fires when the queue backs up.
4. **Tracked-invoke machinery** — `ADKAdapter._inflight_invoke_task` (per session,
   `adk.py:1490`) holds the current passthrough task so the executor's control race
   can cancel it. The cancel is cooperative: the steerer writes a flag and the
   plugin's next `before_*` callback short-circuits (`drift_observer.py` cancel
   path); the `CancelledError` boundary catch is `_invoke_internal` (`adk.py:2104`).
5. **`ContextVar` isolation** — `_active_session_var` (drift emitter context),
   `channel_processor_active()` (single-writer plan mutation), the re-entry pin
   `current_reentry_kind` (`adapters/adk_reentry.py`), and (since #491) per-call
   `LlmCallDiagnostics` — all `ContextVar`-scoped so concurrent runs on one loop do
   not stomp each other.

### Inline vs background — the rule you must not break

| On the critical path (INLINE, `await`ed) | Background (fire-and-forget) |
|------------------------------------------|------------------------------|
| `detect_looping_reasoning` (cheap, affects current turn) | reasoning-drift LLM judge |
| deterministic detectors (tool-error, refusal) | goal-drift trajectory judge |
| `DriftObserver.observe` (builds observation) | custom judges (#437) |
| task transitions / `mark_task_*` | — |
| `events.emit` (fast fan-out) | — |

If you make a background judge inline, you serialize every downstream ADK tool call
behind a multi-second LLM round-trip (the exact regression `observe_reasoning`'s
async dispatch fixed). If you make an inline detector background, you lose its
ability to affect the current turn.

---

## Common mistakes

Concrete wrong edits a weak model would plausibly make on this spine, each with the
correct alternative.

### 1. Reading `steerer._observation_only` directly in a consumer

**Wrong:** `if steerer._observation_only: return` in the executor or plugin.
**Right:** `from goldfive.steerer import steering_is_active` then
`if not steering_is_active(steerer): <passive branch>`. Consumers must fail-safe to
passive when the steerer is `None`/stubbed/raising (invariant 5, #488). The only
place `_observation_only` is read is inside `DefaultSteerer.is_active_steering`.

### 2. Awaiting a judge inline in `observe_reasoning`

**Wrong:** `verdict = await self._run_judge_window(...)` directly in
`observe_reasoning`.
**Right:** keep the `asyncio.create_task(self._run_judge_background(...),
name=f"goldfive-reasoning-judge:{session.id}")` + `_background_judges` pattern. The
method runs on the ADK model-response callback; inline awaits serialize tool
dispatch. Late verdicts are handled by supersedes downstream.

### 3. Spawning a background task without the session-id suffix

**Wrong:** `asyncio.create_task(self._run_goal_drift_judge_background(...))`.
**Right:** `asyncio.create_task(..., name=f"goldfive-goal-drift-judge:{session.id}")`
and add to `_background_judges`/`_background_drifts`. Without the suffix,
`drain_session_background_tasks` (§4.1) cannot scope the drain — the task either
leaks past `RunCompleted` or forces the empty-session-id refusal.

### 4. Assuming `session.plan` can be `None` when the planner runs

**Wrong:** guarding `if session.plan is None: session.plan = Plan.empty(...)` inside
a planner or executor.
**Right:** the Runner guarantees non-None `session.plan` at step 3a
(`Plan.empty(run_id=...)` on first turn). Do not re-seed downstream; you will
clobber the prior-plan carry-forward. Mutate only via `set_session_plan` under
`channel_processor_active()` (goldfive#247 single-writer; see `11`).

### 5. Reviving `PLAN_DIVERGENCE` handling

**Wrong:** removing the `if drift.kind is DriftKind.PLAN_DIVERGENCE: return` guard
at the top of `handle_drift` because "it looks dead".
**Right:** it is intentionally disabled (goldfive#252) but the machinery is a
PROTECTED KEEP (#252-disabled branch). Leave both the guard and the machinery.

### 6. Keying a lifecycle gate on an LLM-minted id

**Wrong:** keying the in-flight-refine set or the coalescing map on
`drift.id` (fresh per emit) or an LLM-supplied goal id.
**Right:** key on stable structured identity — `(session.id, drift.kind.value,
current_task_id)` for in-flight refine; `(session.id, agent_name, current_task_id)`
for the judge queue. A churning key opens a fresh entry per observation and the gate
never engages (invariant 6).

### 7. Calling `planner.generate` unconditionally every turn

**Wrong:** replacing the `handle_turn` gate with a straight `generate` call to
"simplify".
**Right:** `handle_turn` is the goldfive#271 Phase 4 consolidated LLM call
(classify + produce in one shot). Calling `generate` every turn re-plans
conversational turns and re-mints plan ids, breaking `plan_id` stability. Fall
through to `generate` only when `handle_turn` is absent/raised/first-turn-empty.

### 8. Deleting the `finally` stash in `_run_locked`

**Wrong:** moving `convo.stash_plan` into the success path.
**Right:** it must be in `finally` — ADK closing the runner raises `CancelledError`
(a `BaseException`, not caught by `except Exception`), and the stash must still land
so the next turn's `handle_turn` seed is correct (goldfive#271 Gap 1). The
`mark_stash_completed()` tripwire asserts this.

### 9. Treating a sink exception as a run failure

**Wrong:** wrapping `events.emit` in a `try/except` that aborts the run.
**Right:** `events.emit` already isolates sink exceptions (`return_exceptions=True`,
`events.py:104`, hardened #479). Sinks are observability, never control flow. Only
`CancelledError` re-propagates.

### 10. Adding a prompt-cooperation requirement to make control work

**Wrong:** "the coordinator must call `report_task_completed` for termination to
work."
**Right:** termination/control/observability must work with zero tool calls from
the agent (invariant 1). The overlay path (`invoke_passthrough` + plugin callbacks +
`PlanReconciler` + generator-end termination) is passive by construction. Reporting
tools are an *augmentation*, never a contract.

### 11. Making `wrap()` skip `detect_llm` when planner+goal_deriver are supplied

**Wrong:** re-adding `if planner is None or goal_deriver is None:` around
`detect_llm`.
**Right:** `detect_llm` runs whenever `call_llm` is absent. The judges are wired
from the detected callable independently of the planner/goal_deriver overrides;
guarding it re-introduces the silent-disarm (judges inert). Only an explicit
`call_llm=` suppresses detection.

---

## Verification checklist

Run these after touching any hop on the spine. Commands assume repo root
`/home/sunil/git/goldfive` with `uv sync --extra dev --extra adk` already done.

1. **Full suite (fast).** ~30s, expect ~2912 passed / 61 skipped:
   ```
   uv run pytest -q
   ```
2. **Lint must stay clean** (repo is NOT ruff-format-clean — do not mass-reformat):
   ```
   ruff check .
   ```
3. **`wrap()` construction** — after editing `convenience.py`:
   ```
   uv run pytest -q tests/ -k "wrap or convenience or judge_only or detect_llm"
   ```
4. **Turn lifecycle** — after editing `runner.py` / `conversation.py`:
   ```
   uv run pytest -q tests/test_intra_session_plan_carry_forward.py tests/ -k "run_locked or handle_turn or conversation"
   ```
5. **Overlay + control** — after editing `executors/sequential.py` / `control.py`:
   ```
   uv run pytest -q tests/ -k "overlay or steer or control or nudge or supersede"
   ```
6. **Drift dispatch + ladder + gate** — after editing `drift_observer.py` /
   `steerer.py`:
   ```
   uv run pytest -q tests/ -k "handle_drift or ladder or observation_only or judge_semaphore or coalesc"
   ```
7. **The `observation_only` invariant** — confirm no consumer reads the bare
   attribute directly. Use a word-boundary grep for the attribute access:
   ```
   grep -rn "\._observation_only\b" goldfive/ | grep -v test
   ```
   Expect exactly two hits in `steerer.py`: the assignment in `__init__` and the
   read inside `is_active_steering` (`steerer.py:1370`). Any OTHER
   `._observation_only` attribute read is a bug — route it through
   `is_active_steering()` / `steering_is_active(steerer)`. (The helper names
   `_is_observation_only` / `_observation_only_active` are correct delegators, not
   violations.)
8. **Background-task naming** — every `create_task` on the drift path must carry the
   session-id suffix:
   ```
   grep -n "create_task" goldfive/drift_observer.py
   ```
   Confirm each has `name=f"goldfive-<kind>:{session.id}"`.
9. **Contract-table drift** — if you added/removed a bind call or a Protocol
   method, re-diff `goldfive/protocols.py` against the actual call sites:
   ```
   grep -n "bind_steerer\|bind_adapter\|bind_control_channel\|register_reporting_tools" goldfive/runner.py
   ```
10. **Control-proto lockstep** — if you touched `ControlKind`:
    ```
    uv run pytest -q tests/test_control_proto.py
    ```

---

## Where to go next

- The turn lifecycle in full (Conversation, prior-plan carry-forward, abort paths):
  `03-runner-and-conversation.md`.
- The overlay loop branch table, control channel, supersede/cancel discrimination:
  `04-executors-and-control.md`.
- The plugin callback surface in detail: `05-adk-plugin.md`.
- Adapters, `invoke_passthrough`, the cancel boundary, dynamic instructions,
  `GoldfiveADKAgent`: `06-adapters-and-instrumentation.md`.
- Detectors vs judges: `07-deterministic-drift-detection.md`, `08-llm-judges.md`.
- The ladder tables, promotion policy, `observation_only` gates:
  `09-steering-ladder-and-gates.md`.
- Planning, `handle_turn`, refine, `PlanRevised`: `10-planning-and-revision.md`.
- The single-writer plan contract and `session.state` ownership:
  `11-state-ownership.md`.
- Events, sinks, decision telemetry, the verdict-utility summary:
  `12-events-sinks-telemetry.md`.
- The full invariants + hazards + history register: `17-invariants-hazards-history.md`.
