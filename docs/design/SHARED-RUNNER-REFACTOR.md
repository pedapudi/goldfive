# Shared Runner / Session refactor — design alternative

**Status.** Proposed. This document specifies an **alternative** to the
wrapper-propagation work currently landing under goldfive#196 / #197
("Option 7(a)" — plumb inner ADK events through
`GoldfiveADKAgent._run_async_impl` while preserving the dual-Runner
architecture). The team is shipping 7(a) now for schedule reasons; this
doc captures **7(c)** — converge onto ADK's native
`parent_context=` composition — so a future prototype can pick it up
with full context.

Not a commitment. Not yet scheduled. File before the knowledge evaporates.

## 1. Summary

Today, `goldfive.wrap(adk_tree)` returns a `GoldfiveADKAgent` — an ADK
`BaseAgent` subclass whose `_run_async_impl` spins up goldfive's own
`Runner`, which in turn holds an `ADKAdapter` that owns a **separate**
`InMemoryRunner` wrapped around the inner tree. When `adk web` invokes
the outer agent, two ADK runners exist simultaneously and each owns its
own `InvocationContext` / `Session`. Everything that looks like
goldfive scaffolding — three-layer session pinning, wrapper-invocation
dedup, orphan-span sweeps, per-event `session_id` proto fields, the
`_GoldfiveADKPlugin` state mirror — is there to repair the split.

The alternative is to stop creating the split. Use ADK's documented
child-invocation primitive (`inner_agent.run_async(parent_context=ctx)`)
so goldfive orchestration runs as **one** ADK invocation, sharing
Runner, Session, plugin manager, and event stream with the inner tree.
The inner agent's events reach adk-web natively; the steerer and
reconciler run via plugin callbacks on the same Runner; the
intervention ladder translates to callback-held-open `asyncio.Event`
waits and `LiveRequestQueue` content injection.

**What changes:** `GoldfiveADKAgent._run_async_impl` yields events
directly from `self._inner.run_async(parent_context=ctx)`; `ADKAdapter`
short-circuits to child-invocation mode when called from the wrapper
(skipping its own `InMemoryRunner` construction); Levels 3 and 4 of the
intervention ladder migrate off the `cancel + re-invoke` / executor-
blocking path onto mid-flight `LiveRequestQueue` injection and
callback-held-open waits.

**What doesn't change:** the `goldfive.Session` object and every typed
field on it; the public `Runner.run(user_input)` surface; every
`EventSink` shape and the proto `Event` schema; the intervention
ladder's six levels and drift kinds; non-ADK adapters
(`CallableAdapter`, `ClaudeAgentSDKAdapter`) which keep the current
Runner / ADKAdapter path because they were never affected by the
split in the first place.

## 2. Motivation — the cumulative cost of self-inflicted split

The dual-Runner architecture was never a design goal. It fell out of a
sensible early call: `ADKAdapter` is **framework-specific** and should
own whatever it needs to talk to ADK — including an `InMemoryRunner` —
so every `AgentAdapter` looks symmetric from the Runner's perspective.
That symmetry served `CallableAdapter` and `ClaudeAgentSDKAdapter`
well. It cost nothing under the **bare programmatic** use case
(`runner = goldfive.Runner(agent=ADKAdapter(root))`) because there was
no outer ADK invocation to collide with — one runner, one session, one
event stream.

The cost started accruing when `goldfive.wrap(adk_tree)` gained the
polymorphic `BaseAgent` surface so adk-web could mount goldfive
directly (the `GoldfiveADKAgent` class in
`goldfive/adapters/adk_wrap.py`). adk-web now invokes the wrapper with
its own `InvocationContext`, which has its own `Runner` and
`Session.id`. Inside the wrapper, goldfive's `Runner.run` kicks off the
executor, the executor calls `ADKAdapter.invoke_passthrough`, and that
method constructs its **own** `InvocationContext` on its **own**
`InMemoryRunner` with its **own** `Session.id`. Two runners, two
sessions, one conceptual turn.

Every issue below is a direct consequence of that split:

| Scaffolding | Purpose | Issue(s) |
|---|---|---|
| **3-layer session pinning** (`_pin_outer_session_on_adapter`, `_outer_session_id`, `_session_id`, `session_id=` override on `Runner.run`) | Force adk-web session id, goldfive `Session.run_id`, and ADKAdapter `_session_id` to all be the same string so spans and events roll up in harmonograf. | goldfive#161, goldfive#164 |
| **Per-event `session_id` proto field** (field 5 on the Event envelope) | Let `HarmonografSink` route events by id because the Runner that emitted them is not the same Runner the span was emitted under. | goldfive#155, goldfive#157 |
| **`_GoldfiveADKPlugin` state mirror** (`_adk_state_protocol`) | Bridge `goldfive.Session.state["goldfive.*"]` into the ADK runtime session.state so `GoldfivePlanner` sees active steer body, current task id, and goals summary. Two sessions → the writer side isn't the reader side. | goldfive#170, goldfive#173 |
| **Orphan-span sweep / `unexpected_orphan_on_normal_exit` healing** | Close spans opened by the inner runner when the wrapper exits without the inner runner's generator naturally ending (e.g. we cancelled mid-flight). | goldfive#196 (in flight) |
| **Wrapper-invocation dedup** | Suppress duplicate INVOCATION spans — outer wrapper emits one, inner runner emits another, with different ids but conceptually the same turn. | part of #196 / #197 |
| **Plugin re-registration** (`_register_plugin_on_runner` walks `AgentTool` sub-Runners) | ADK propagates plugins into AgentTool sub-Runners — but only on the runner the adapter owns, not back to the outer wrapper's runner. Goldfive installs on its own and relies on ADK's propagation; the wrapper's runner is untouched. | goldfive#122 |

Every one of these is correct code. Every one is also **repair work
for a split we create ourselves**. The split does not come from
anything ADK forces on us — ADK ships a documented primitive
(`parent_context=`) whose entire purpose is to let one agent run
**inside** another's invocation context. We just don't use it.

Secondary costs:

- **Adapter API surface.** `ADKAdapter.invoke_passthrough` is
  essentially "build an invocation context and run the tree"; under a
  shared Runner, that's `ctx.inner_agent.run_async(parent_context=ctx)`
  — two lines, no bookkeeping.
- **Test fixtures.** E2e tests spin up both runners; `adk_ctx` fakes
  have to mimic both session layers. Most fixture complexity is
  session-id coordination we'd delete.
- **Documentation burden.** "Three session ids, usually equal but the
  override points are A / B / C" is a whole section of the user guide
  (`docs/guides/runner-and-session.md` §"Session id property"). Under
  a shared Runner there is one session id, full stop.
- **Mental model friction.** New contributors routinely ask "why do I
  see two INVOCATION spans in the trace?" The answer ("because we run
  two ADK runners for one conceptual turn") is a surprise, not a
  feature.

The pragmatic read: 7(a) is the right patch **now**. It lets the
wrapper propagate inner events without restructuring anything.
7(c) is the right **direction** — convergence onto one Runner deletes
the scaffolding rather than adding a pass-through layer on top of it.

## 3. Proposed architecture

### 3.1 ADK's `parent_context=` primitive

ADK's `BaseAgent.run_async` accepts a keyword `parent_context:
InvocationContext | None = None`. When supplied, the agent runs as a
**child invocation** of the caller: it inherits

- the caller's `Runner` (and therefore its `session_service`,
  `artifact_service`, `memory_service`);
- the caller's `session` (same `session.id`, same `session.state`
  dict — live, not copied);
- the caller's `plugin_manager` (the goldfive plugin and the
  harmonograf telemetry plugin stay installed; their callbacks fire
  for the child agent the same way they fire for the parent);
- the caller's `invocation_id` as the `parent_invocation_id` on the
  child context (ADK mints a fresh child `invocation_id` for span
  attribution but keeps the chain).

The child's `run_async` yields `Event` objects to the caller's async
loop. From the outer runner's perspective the child's events are
**the caller's events** — they flow through the same plugin stack,
stamp the same session id, and reach adk-web over the same stream.

This is the primitive `AgentTool` and `transfer_to_agent` use
internally. Using it from our own wrapper is not exotic; it's the
existing ADK composition pattern applied one layer up.

### 3.2 Redesigned `GoldfiveADKAgent._run_async_impl`

Today (abbreviated, see `goldfive/adapters/adk_wrap.py`):

```python
async def _run_async_impl(self, ctx: InvocationContext):
    outer_sid = self._outer_session_id_from_ctx(ctx)
    self._pin_outer_session_on_adapter(outer_sid)    # repair split
    user_input = _extract_user_input(ctx)
    outcome = await self._runner.run(                # spin up our runner
        user_input,
        context={"adk_ctx": ctx},
        session_id=outer_sid or None,                # pin layer 2 to layer 1
    )
    async for adk_event in _outcome_to_adk_events(   # synthesize summary
        outcome, ctx, author=self.name
    ):
        yield adk_event
```

Under the shared-Runner model:

```python
async def _run_async_impl(self, ctx: InvocationContext):
    # 1. Run goldfive's pre-execution pipeline (goal derive + plan).
    #    These phases don't touch ADK; they emit goldfive events to
    #    sinks and stamp session.state["goldfive.*"] keys live.
    session = await self._runner._prepare_session_from_ctx(ctx)
    goals = await self._runner._derive_goals(user_input, session=session)
    plan = await self._runner._generate_plan(goals, session=session)
    # PlanSubmitted has now been emitted to every sink; session.state
    # carries goldfive.current_plan_id / goldfive.goals_summary.

    # 2. Hand off to the inner tree as a child invocation. Its events
    #    reach adk-web over the caller's own event stream. Plugins
    #    (steerer observations, reconciler, GoldfivePlanner) fire on
    #    the same Runner.
    async for event in self._inner.run_async(parent_context=ctx):
        # Steerer / reconciler / drift detectors ran in plugin
        # callbacks as events were produced; the generator just
        # yields them forward.
        yield event

    # 3. Emit terminal goldfive events to sinks.
    outcome = await self._runner._finalize(session)
    # RunCompleted / RunAborted goes to every sink. adk-web has
    # already seen a terminal ADK event from the inner tree, so it
    # doesn't need a synthesized terminal event from us.
```

Key properties:

- **One `InvocationContext`.** The inner agent runs under `ctx` — the
  one adk-web handed us. No second runner exists.
- **One `session.id`.** The goldfive `Session` adopts `ctx.session.id`
  (as it already does today via `session_id=`). The ADK runtime
  session IS the goldfive Session's backing state dict — no mirror
  needed.
- **One event stream.** The inner agent yields events; we yield them
  forward. The `_outcome_to_adk_events` summariser goes away — the
  ADK UI saw the real events, not a reconstructed narrative.
- **Plugin stack unchanged.** `_GoldfiveADKPlugin` still installs on
  the (now sole) Runner. Its `before_agent_callback` /
  `before_model_callback` / `before_tool_callback` surface is exactly
  the observation and intervention point we need (§4).

### 3.3 Redesigned `ADKAdapter`

`ADKAdapter` keeps its current shape **for non-wrapper callers**. A
bare `Runner(agent=ADKAdapter(tree))` continues to work — the adapter
owns an `InMemoryRunner` and `invoke_passthrough` behaves as today.
This is the "goldfive as library" path and it was never affected by
the split.

In wrapper mode the adapter short-circuits:

```python
class ADKAdapter:
    def __init__(self, agent_or_runner, ..., parent_ctx: InvocationContext | None = None):
        ...
        # NEW: when constructed from GoldfiveADKAgent, the wrapper
        # stashes the outer ctx here. invoke_passthrough detects it
        # and runs as a child invocation instead of building its own
        # runner.
        self._parent_ctx = parent_ctx

    async def invoke_passthrough(self, user_message, *, session, reconciler=None, ctx=None):
        if self._parent_ctx is not None:
            # Shared-Runner path. No session creation, no runner mint.
            # Dispatch the user_message via parent_ctx's live request
            # queue (or, in the wrapper's redesigned _run_async_impl,
            # this method isn't called at all — the wrapper drives the
            # child invocation directly).
            return await self._invoke_child(user_message, session=session)
        # Legacy path — unchanged.
        return await self._invoke_internal(...)
```

In practice the wrapper's redesigned `_run_async_impl` drives the child
invocation directly (§3.2) and doesn't go through
`invoke_passthrough` at all. The adapter method is kept as a fallback
for non-wrapper code paths that still want the "one adapter call per
turn" shape (e.g. custom executors). But the hot path under
`goldfive.wrap` becomes wrapper → `self._inner.run_async(parent_context=...)`,
bypassing the adapter's own runner entirely.

### 3.4 Where the Runner's phases live

Under a shared Runner the goldfive `Runner.run` pipeline splits across
two call sites:

```
wrapper._run_async_impl(ctx):
    ┌─────────────────────────────────────────────────────────┐
    │ goldfive.Runner pre-execution (delegated)               │
    │   RunStarted, GoalDerived, PlanSubmitted emit to sinks  │
    │   session.state["goldfive.*"] stamped live               │
    │   NO ADK events yet                                      │
    └─────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────┐
    │ self._inner.run_async(parent_context=ctx)               │
    │   ADK events flow through; adk-web sees them natively    │
    │   Plugins fire on every event                            │
    │     - _GoldfiveADKPlugin: steerer observations, drift    │
    │       detection, reporting-tool interception, tool-loop  │
    │       cap, intervention-ladder dispatch                  │
    │     - HarmonografTelemetryPlugin: span emission          │
    │     - GoldfivePlanner (per-LlmAgent): orchestration      │
    │       block injection + response-side classification     │
    │   Reconciler observes before/after_agent pairs and       │
    │   writes session.state["goldfive.current_task_id"] live  │
    └─────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────┐
    │ goldfive.Runner finalization (delegated)                │
    │   PENDING-task NOT_NEEDED sweep (goldfive#163)           │
    │   RunCompleted or RunAborted emits to sinks              │
    └─────────────────────────────────────────────────────────┘
```

The goldfive `Runner` stays a real object — it owns the Session, the
sinks, the control channel, the plan, the steerer, the planner. What
changes is that in wrapper mode it doesn't own an AgentAdapter-driven
executor loop; it exposes its phases (derive, generate, finalize) as
discrete coroutines the wrapper invokes around the ADK child
invocation. `Runner.run(user_input)` still works for non-wrapper
callers — the library surface composes those same phases with an
executor loop in between, as it does today.

## 4. Intervention ladder translation table

The ladder today (see `goldfive/steerer.py::DefaultSteerer` and
`docs/design/CONTROL.md §7.d`) has six levels, 0 through 5. Each
level's mechanism under the current dual-Runner architecture, and
the proposed shared-Runner equivalent:

### Level 0 — OBSERVE

**Today:** `DriftDetected` emits to sinks. No in-flight action.

**Shared Runner:** Unchanged. The steerer is a plugin; emission is
sink-side. No coupling to the Runner layout.

**Risk:** None.

### Level 1 — ABSORB (plan revision)

**Today:** `planner.refine(plan, drift, goals)` is called, the revised
plan installs, the executor re-reads `session.plan` on the next
iteration. Under overlay mode the executor is already a single
`invoke_passthrough` call — ABSORB revises the plan mid-run without
interrupting execution. USER_STEER lands here at WARNING severity.

**Shared Runner:** Unchanged for sink behaviour
(`PlanRevised` event fires). The one subtlety is that "the executor
re-reads `session.plan` on the next iteration" no longer applies the
same way — under shared Runner, ABSORB during an active child
invocation means the revised plan installs in `session.state` and the
`GoldfivePlanner`'s next per-turn `before_model_callback` picks it up
via the orchestration block. The inner tree's next LLM call sees the
revised plan. Same behaviour, different seam.

**Risk:** Low. GoldfivePlanner already reads from `session.state`
(`goldfive.current_plan_id`, `goldfive.goals_summary`) on every LLM
call; that path doesn't change.

### Level 2 — NUDGE (soft follow-up)

**Today:** `session.pending_nudges.append(text)`; the overlay loop
pops one on each invocation-end and dispatches as a soft follow-up
user message. Not on the default table today (reserved for future
policies) but the plumbing exists.

**Shared Runner:** The child invocation is a single `run_async`
stream — there's no "next invocation" to hook a nudge onto inside a
turn. Two options:

- **Within-turn nudge (preferred):** Inject the nudge via
  `LiveRequestQueue` on `ctx` mid-flight. Same mechanic proposed for
  Level 3 (§below). This actually makes NUDGE **better** than today —
  the nudge lands while the tree is still thinking, not at the end of
  its current turn.
- **Next-turn nudge (fallback):** Stash on `session.pending_nudges`
  and prepend to the next user turn's input in the wrapper. Matches
  today's semantics but defers the nudge further.

**Risk:** Low. `LiveRequestQueue` content injection is documented ADK;
the semantics of "arrives at the next model turn" need confirming by
prototype.

### Level 3 — CANCEL_REINVOKE

**Today:** `ADKAdapter._cancel_in_flight_invocation` cancels the
running `runner.run_async` generator; `_heal_pending_tool_calls`
closes any pending function_call/function_response pairs;
`session.pending_corrective_message` carries the reframed user
message; the overlay loop restarts `invoke_passthrough` with that
message as input. This is how USER_STEER mid-flight works today
(goldfive#149, #150, #152).

**Shared Runner:** Two approaches, prototype to pick:

**(a) Preferred — `LiveRequestQueue` mid-flight content injection.**
ADK's `InvocationContext` carries a `live_request_queue` (or
equivalent — exact attribute name needs verification). Steerer writes
the corrective user message to that queue; the inner model gets the
injected content on its next turn **without** cancelling the current
generator. No tool-call healing needed because we don't cancel; we
redirect. Cleaner than today, but depends on the queue semantics
being "injected content is observed on the next LLM call with proper
conversation continuity" — which is the documented behaviour for
streaming agents but hasn't been stress-tested by goldfive for
batch-turn agents under mid-tool-call injection.

**(b) Fallback — wrapper-local re-invoke loop.** If (a) doesn't carry
conversation state cleanly across the injection, the wrapper can
re-drive the child invocation in a loop:

```python
async def _run_async_impl(self, ctx):
    ... derive + plan ...
    corrective: str | None = None
    while True:
        async for event in self._inner.run_async(parent_context=ctx):
            yield event
            # Steerer may have set session.pending_corrective_message
            # in a plugin callback during this loop. If so, fall
            # through to re-invoke.
        if session.pending_corrective_message is None:
            break
        # Append the corrective message to the session's event history
        # as a user turn; the next run_async picks up the amended
        # conversation natively.
        corrective = session.pending_corrective_message
        session.pending_corrective_message = None
        await self._append_user_event(ctx, corrective)
```

This still cancels — mid-flight cancellation propagates into the
generator's active `run_async` call. Tool-call healing is still
required. But it keeps the shared-Runner architecture and avoids
depending on `LiveRequestQueue` semantics we haven't verified.

**Risk:** HIGH. Needs prototyping (see §9). The cancellation-
propagation semantics of an active `run_async` child invocation under
a parent context are not a well-trodden path.

### Level 4 — PAUSE_ESCALATE

**Today:** Steerer sets `session.paused_for_human_intervention = True`;
the executor's pre-task loop in `SequentialExecutor._check_controls`
notices the flag and blocks on `control.receive()` until
CONTROL_RESUME / CONTROL_STEER. The pause happens **between** tasks
under the overlay loop — the in-flight task runs to completion first.

**Shared Runner:** The "between tasks" seam disappears under overlay;
there's just one `run_async` generator. Pause needs to happen
**inside** a plugin callback that the Runner holds open:

```python
class _GoldfiveADKPlugin:
    async def before_agent_callback(self, ctx, agent):
        if ctx.session.state.get("goldfive.paused_for_human_intervention"):
            event = self._pause_event_for(ctx.session)
            await event.wait()   # held until DefaultSteerer.resume()
```

The steerer resolves the `asyncio.Event` on CONTROL_RESUME /
CONTROL_STEER. Mechanically this looks like the overlay loop's
blocking `control.receive()` today, just hung off the plugin callback
instead of the executor.

**Risk:** HIGH. Holding a plugin callback open for seconds-to-minutes
is not a normal ADK pattern. Specific concerns:

- **Cancellation propagation.** If the outer `ctx` is cancelled (e.g.
  adk-web client disconnect), does the held-open callback's
  `event.wait()` receive a `CancelledError`? It should, via asyncio,
  but ADK's runner may not be propagating task cancellation into
  plugin tasks cleanly. Needs verification.
- **Tool-use state.** If PAUSE triggers between
  `before_model_callback` and `after_model_callback`, is the Runner
  in a consistent state to resume? Almost certainly — `before_agent`
  fires at agent boundaries, not model boundaries, so this is the
  right seam — but test coverage needed.
- **adk-web streaming.** While paused, the UI has to show "waiting
  for operator". Since the generator is held inside a plugin
  callback, the last yielded event is whatever preceded PAUSE. An
  explicit `DriftDetected(HUMAN_INTERVENTION_REQUIRED)` event from
  the steerer still emits to sinks before the wait, so harmonograf
  shows the pause reason. But the ADK UI's own "agent is thinking"
  spinner will just keep spinning — acceptable for the ask, but
  worth noting.

### Level 5 — TERMINATE

**Today:** Executor short-circuits into `RunAborted`; the run ends.
The control channel drains, `channel.close()` fires on close.

**Shared Runner:** Steerer raises an exception from the plugin
callback. ADK propagates it out of `run_async`; the wrapper catches,
emits `RunAborted`, and re-yields a synthesized terminal event to
adk-web (the one case where we still synthesize, because ADK may not
produce a clean turn-complete event on exception).

**Risk:** Medium. Exception propagation across the parent-child
boundary needs verification — specifically, that a child exception
surfaces cleanly at the wrapper's `async for` without losing
traceback fidelity.

### Summary table

| Level | Name | Today | Shared Runner | Risk |
|---|---|---|---|---|
| 0 | OBSERVE | Sink emit | Same | — |
| 1 | ABSORB | `planner.refine` + session.plan swap; executor re-reads | `planner.refine` + session.state; GoldfivePlanner picks up on next LLM call | Low |
| 2 | NUDGE | `session.pending_nudges` + overlay pop | `LiveRequestQueue` inject mid-flight; or next-turn queue fallback | Low |
| 3 | CANCEL_REINVOKE | Cancel + heal + restart `invoke_passthrough` | `LiveRequestQueue` inject (preferred); or wrapper-local re-invoke loop with cancellation + heal (fallback) | **HIGH** |
| 4 | PAUSE_ESCALATE | Executor blocks on `control.receive()` | Plugin callback holds open on `asyncio.Event.wait()` | **HIGH** |
| 5 | TERMINATE | `RunAborted` via executor short-circuit | Plugin raises; wrapper catches + emits terminal | Medium |

## 5. What falls out for free

Each of the scaffolding items in §2 goes away or collapses:

- **3-layer session pinning.** There is one session — `ctx.session`.
  `goldfive.Session.id` is set to `ctx.session.id` at the top of
  `_run_async_impl` and never needs a second write. No
  `_outer_session_id`, no `_pin_outer_session_on_adapter`, no
  `session_id=` override on `Runner.run` (it can stay as a convenience
  for the non-wrapper path, but it's no longer load-bearing).
- **Per-event `session_id` proto field** stays in the schema (for
  non-ADK sinks and cross-run audit) but its role collapses: every
  event in a wrapper run carries the same session id because there's
  only one. The `HarmonografSink` no longer needs per-event routing
  disambiguation for the wrapper case.
- **Wrapper-invocation dedup.** There is no outer wrapper INVOCATION
  span and inner-runner INVOCATION span duplicate — the wrapper
  doesn't invoke a separate runner, so the harmonograf telemetry
  plugin emits one INVOCATION span per turn, period.
- **Orphan-span sweep / `unexpected_orphan_on_normal_exit`.**
  Mid-flight cancellation still happens (Level 3 fallback path), but
  it happens in-place on the one runner; the plugin's
  `after_agent_callback` fires on cancellation and closes its own
  spans. No cross-runner orphan reconciliation needed.
- **`_GoldfiveADKPlugin` state mirror.** `goldfive.Session.state` IS
  `ctx.session.state` — same dict. Writers stamp once; readers read
  the same live dict. The `before_run_callback` that mirrors keys
  from the outer session to the inner runner's session disappears.
  The state-protocol module stays (it owns the key names and the
  `goldfive.*` prefix enforcement) but its "mirror" responsibility
  folds into ordinary dict writes.
- **Plugin re-registration / subtree walking.** One Runner, one plugin
  install, full stop. ADK already propagates the plugin manager into
  AgentTool sub-Runners spawned from the shared Runner, so the
  `_register_plugin_on_runner` walk goes away.

## 6. What stays unchanged

- **`goldfive.Session` with typed fields.** Every field on the dataclass
  (`goals`, `plan`, `completed_results`, `task_progress`,
  `pending_approvals`, `paused_for_human_intervention`,
  `pending_nudges`, `pending_corrective_message`, `state`, etc.) keeps
  its shape and owner. Readers and writers are the same. The change is
  that `Session.state` happens to BE the live ADK session.state dict
  rather than a mirror of one.
- **Every `EventSink` shape.** `InMemorySink`, `LoggingSink`,
  `JSONLPersistenceSink`, `SQLitePersistenceSink`, `GRPCSink`,
  `HarmonografSink` — all consume proto `Event` envelopes, and every
  goldfive event still flows through them. The Event schema itself
  (including `session_id` field 5) does not change — backwards
  compatibility with archived JSONL / SQLite runs is preserved.
- **Non-ADK adapters** (`CallableAdapter`, `ClaudeAgentSDKAdapter`)
  keep the current Runner + AgentAdapter + Executor path verbatim.
  They were never affected by the dual-Runner split — there was never
  an outer ADK invocation to share with. The shared-Runner refactor
  is scoped to the `goldfive.wrap(adk_tree)` call path.
- **Public Runner API.** `Runner.run(user_input)` still works; the
  `ExecutionOutcome` it returns still carries `success`, `session`,
  `reason`. The library surface is stable.
- **Intervention ladder — six levels, mapping table.** The drift-kind
  → level mapping (`_LADDER` in `DefaultSteerer`) does not change.
  Only the mechanism a given level uses to actually land its effect
  changes (§4).
- **Control channel + proto.** `ControlChannel`, `ControlKind`, the
  proto enum, STEER idempotency, APPROVE / REJECT dual flow — all
  unchanged. Shared-Runner is an internals refactor; the control-plane
  contract is public API.

## 7. Regression risk inventory

Ordered by severity, highest first. Each item calls out what to
prototype, what to measure, and what invariant must hold.

### 7.1 `parent_context=` composition edge cases — SEVERITY: HIGH

Unknowns:

- **Error propagation.** Does an exception raised inside the child
  agent's LLM call surface cleanly at the wrapper's `async for`, or
  does ADK swallow it into a synthetic error event? Needs a direct
  prototype: raise from inside a FunctionTool, raise from inside
  `before_model_callback`, raise from inside the child's own
  `_run_async_impl`.
- **Cancellation propagation.** When the outer `ctx`'s task is
  cancelled (adk-web client disconnect, shared-Runner Level 3 cancel),
  does the child's `run_async` generator receive `CancelledError`
  promptly? Specifically: if cancellation lands between
  `before_tool_callback` and the tool's execution, does the tool run
  anyway?
- **Streaming content deltas.** For agents that stream tokens
  (`LiveRequestQueue`-driven), do delta events propagate through the
  child-to-parent yield chain without coalescing?
- **Plugin scoping.** If the inner tree itself has a
  `parent_context=`-spawned sub-invocation (e.g. the caller's own
  tree uses the pattern for nested orchestration), does the plugin
  still fire at every level? Or does plugin propagation short-circuit
  at the first re-entry?

Invariant to hold: **every event the inner tree produces must reach
adk-web and every sink, without reordering.** Test: run a known tree
(e.g. `examples/adk_presentation`) under both modes and diff the
event sequences.

### 7.2 Level 3 `LiveRequestQueue` semantics — SEVERITY: HIGH

Prototype before anything else (§9.1).

Unknowns:

- Does a content item pushed onto the live request queue **during an
  active model turn** arrive before the next model call, or after the
  current call's tool-use cycle fully drains?
- If it arrives mid-tool-call, does the conversation history the
  model sees on the next turn include both the user-injected
  redirect and the partial tool-use sequence that was in flight?
- If not, is there an API to inject the content as a **new user
  turn** at the root of the conversation (rather than inside the
  current turn)?

The fallback path (§4 Level 3(b)) is architected specifically
because we haven't answered these yet. If the prototype shows the
queue semantics are clean, Level 3 becomes simpler and
cancellation-free. If not, we fall back to cancellation + heal + re-
drive, which works but preserves most of today's complexity just in a
different seam.

### 7.3 Level 4 callback-held-open — SEVERITY: HIGH

Unknowns:

- Cancellation propagation through a `before_agent_callback` that's
  blocking on `asyncio.Event.wait()` for minutes. ADK's runner likely
  wraps callbacks in a task; does that task inherit the outer
  cancellation scope? If not, a disconnected adk-web client could
  wedge the Runner indefinitely.
- Memory / resource cost of holding the runner's event loop on a
  paused callback. Probably fine — asyncio scales to millions of
  sleeping coroutines — but worth spot-checking under harmonograf
  session reconnection scenarios.
- Interaction with ADK's own 500-LLM-call ceiling. If PAUSE fires at
  call 499, resumes, and the agent re-queues work, do we trip the
  ceiling? (Probably not — pause holds the agent entry, doesn't queue
  further LLM calls — but call it out.)

Invariant to hold: **a paused run resumes to the same state it
paused in, preserving every mid-turn partial artifact.**

### 7.4 Plan events before child invocation — SEVERITY: MEDIUM

The wrapper's redesigned `_run_async_impl` emits `RunStarted`,
`GoalDerived`, `PlanSubmitted` to goldfive sinks **before** starting
the child invocation (§3.2 step 1). This preserves today's causal
order — sinks see a plan before they see task events.

Risk: if `GoldfivePlanner` on the very first LLM call of the child
invocation renders its orchestration block from `session.state`,
those state keys (`goldfive.current_plan_id`,
`goldfive.goals_summary`) must be stamped **before** the child runs.
Today this is enforced by the executor being a separate code path
from the ADK invocation; under shared Runner, the wrapper has to
stamp before yielding to the child.

Test: assert state-key stamp ordering with an explicit
`before_model_callback` that reads the keys and fails the test if
they're missing.

### 7.5 Budget / `max_task_invocations` enforcement — SEVERITY: MEDIUM

Today the executor counts adapter invocations and aborts the run
with `RunAborted(reason="max_task_invocations exceeded")`. Under
shared Runner there is no executor loop to count against — the child
invocation is one `run_async` call. The enforcement moves to a
`before_agent_callback` in the plugin that counts model turns and
raises when the budget is exceeded.

Risk: "model turns" ≠ "task invocations". The mapping needs calling
out in the spec (1:1 under overlay because each invocation is one
conversation turn; not 1:1 under legacy per-task execution, which
we've deprecated).

### 7.6 AgentTool cap (goldfive#130) — SEVERITY: LOW

Today the cap lives in the plugin and short-circuits further AgentTool
spawns in a single invocation. It's keyed off the plugin manager's
state, which is already on the shared Runner. No change needed for
the cap itself — it works the same under shared Runner because it
was always plugin-based.

### 7.7 `ADKAdapter._runner` as a distinct object — SEVERITY: LOW

Audit required: anything keyed off the identity of
`ADKAdapter._runner` as a separate object from the outer runner. A
grep of the codebase finds:

- Test fixtures: several tests construct an `ADKAdapter` directly and
  assert on `adapter._runner.plugin_manager`. Under wrapper mode this
  attribute still exists but may reference the shared runner; tests
  that depended on the adapter owning a *fresh* runner need an
  explicit `ADKAdapter(parent_ctx=None)` construction to keep the
  legacy path.
- `_register_plugin_on_runner`: walks `_runner.plugin_manager`. Under
  wrapper mode this is the shared runner's plugin manager; the walk
  still works, but the idempotency check (plugin already installed
  by name) becomes more important because the shared runner may
  already have the plugin from the outer installation path.

Mitigation: keep the legacy `ADKAdapter(agent)` construction
fully functional (non-wrapper mode) and gate shared-Runner behaviour
on `parent_ctx` being passed in. Tests that explicitly want two
runners can keep constructing two adapters.

### 7.8 `HarmonografTelemetryPlugin` — SEVERITY: LOW

The plugin emits INVOCATION / AGENT / MODEL_CALL spans. Today the
outer wrapper's runner emits one INVOCATION span (for the outer
turn) and the inner runner emits a second (for the inner turn).
harmonograf client code already handles the duplicate in goldfive#196
(either by suppressing the outer span or by nesting them). Under
shared Runner only the one span emits — **harmonograf client code
that was paying attention to the outer INVOCATION span as a "goldfive
turn starting" marker needs to handle its absence.** Specifically,
the `Hello` / home-session rollup logic from harmonograf#61 uses
the first INVOCATION span on the session to trigger the rollup; under
shared Runner that span has `ctx.session.id` directly, no remap
needed. Simpler, not more complex.

## 8. Test strategy

Flag-gated dual-mode CI. The refactor is too invasive to land
atomically on default behaviour; it ships behind a flag and graduates
over a soak period.

### 8.1 The flag

`goldfive.wrap(tree, use_shared_runner: bool = False)`. Default off
through the prototype phase; default on after the graduation
criteria (§8.4) are met; flag removed after the dual-Runner
scaffolding retires.

When `use_shared_runner=True`, `wrap` returns a `GoldfiveADKAgent`
whose `_run_async_impl` takes the shared-Runner path (§3.2). When
`False`, behaviour is exactly today's.

### 8.2 Dual-mode CI

The e2e test suite runs every test under **both** modes. Mechanism:

```python
@pytest.mark.parametrize("use_shared_runner", [False, True])
async def test_overlay_e2e(use_shared_runner):
    runner = goldfive.wrap(tree, use_shared_runner=use_shared_runner, ...)
    ...
```

Tests that can't run in one mode (e.g. a test that explicitly asserts
on the dual-Runner shape) skip conditionally with a clear reason.
Coverage target: every test in `tests/test_overlay_*.py`,
`tests/test_live_steering_*.py`, and the ADK-specific reconciler /
drift-detector tests.

### 8.3 Specific test additions

New tests specific to shared-Runner semantics:

- `test_shared_runner_session_id_single`: assert `ctx.session.id ==
  session.id == event.session_id` on every emitted event.
- `test_shared_runner_plan_events_before_child`: assert the first ADK
  event's `invocation_id` is **after** `PlanSubmitted` in the sink's
  event log.
- `test_shared_runner_level3_live_inject`: drive a STEER mid-flight,
  assert the corrective message lands without generator cancellation.
  (May be expected to fail initially; gates flag flip.)
- `test_shared_runner_level3_fallback_reinvoke`: same but with the
  fallback path; assert generator cancels, corrective message lands
  as next user turn, tool-calls heal.
- `test_shared_runner_level4_pause_resume`: PAUSE mid-flight, RESUME,
  assert the run completes to the same terminal state as the
  unpaused baseline.
- `test_shared_runner_level4_cancelled_client`: PAUSE mid-flight,
  cancel the outer `ctx`, assert the held-open callback unblocks
  within 5s.
- `test_shared_runner_budget_enforcement`: set
  `max_task_invocations=3`, drive a run that would consume 5,
  assert RunAborted with the right reason.
- `test_shared_runner_non_adk_unaffected`: wrap a `CallableAdapter`,
  confirm the flag is ignored (non-ADK path).

### 8.4 Graduation criteria for flag flip

Default flips from `False` to `True` when all hold:

1. Every dual-mode test passes under both modes for ≥ 2 weeks of CI.
2. Level 3 LiveRequestQueue test passes (or fallback path is
   confirmed acceptable and documented as the default).
3. Level 4 held-open-callback test passes including the cancelled-
   client case.
4. At least one in-tree example (e.g. `examples/adk_presentation`)
   has run end-to-end under shared Runner with harmonograf
   observation for a full day of manual dogfooding.
5. No open `regression: shared-runner` label on the issue tracker.

### 8.5 Scaffolding retirement

After the flag defaults to on, keep the dual-Runner code path alive
for at least one release so external callers who pinned
`use_shared_runner=False` for any reason have a migration window.
Then:

- Remove `_pin_outer_session_on_adapter` and the 3-layer session
  pinning code.
- Simplify `_GoldfiveADKPlugin`'s state-mirror to ordinary writes
  (the mirror direction was always orchestration → ADK; that becomes
  "write once to session.state").
- Remove the `session_id=` override path on `Runner.run` if no non-
  wrapper caller uses it. (Leave it if any does.)
- Retire the `use_shared_runner` flag itself.

## 9. Rollout plan

Ordered, with sizing. A prototype engineer should be able to pick
this up and execute it.

### 9.1 Level 3 LiveRequestQueue spike — 1 day

Minimum viable prototype:

- Build a throwaway ADK agent with a known tool-call pattern.
- From outside the Runner, push a `Content(role="user", parts=...)`
  onto `ctx.live_request_queue` mid-flight.
- Observe where the injected content lands: is it visible on the
  next model call's conversation history? Mid-current-call? Not
  at all?

Deliverable: a 1-page findings doc appended to this design doc as
§9.1.a. If the semantics support mid-flight injection, Level 3 takes
the preferred path (§4 Level 3(a)). If not, Level 3 takes the
fallback path (§4 Level 3(b)) and cancellation + heal stays part of
the shared-Runner architecture.

### 9.2 Minimum viable shared-Runner wrapper — 3-5 days

- Add `GoldfiveADKAgent._run_async_impl_shared` (behind the flag).
- Wire pre-execution phases (derive + plan) as callable coroutines
  off the `Runner`.
- Drive the child invocation with `parent_context=ctx`.
- Yield events forward; emit `RunCompleted` to sinks.
- NOT yet wired: Levels 2 / 3 / 4 — those come in phase 9.3. Levels
  0, 1, 5 work via the existing plugin surface.

Deliverable: a `Runner` that passes the basic overlay e2e under
shared Runner, with Levels 3 and 4 still deferring to the legacy
path (via a mode-switch inside the plugin — test-only crutch to let
phase 9.2 land without Level 3/4 ready).

### 9.3 Intervention-ladder rewiring — 5-10 days

- Level 3: implement whichever path §9.1 selected.
- Level 4: implement callback-held-open `asyncio.Event.wait()`.
- Level 2: within-turn `LiveRequestQueue` injection, with next-turn
  fallback.
- New dual-mode tests (§8.3) land in CI.

Deliverable: all six ladder levels working under shared Runner for
the supported drift kinds. Flag still defaults to off.

### 9.4 Dual-mode CI soak — ≥ 2 weeks

Flag-gated parametrization across the full e2e suite. Fix any
regressions that surface. Manual dogfooding on
`examples/adk_presentation` and one real harmonograf-observed run.

### 9.5 Flag flip — 1 day

Change default to `use_shared_runner=True`. Land in a minor version
bump with a clearly-called-out migration note in the CHANGELOG. Keep
the flag itself so callers can opt out temporarily.

### 9.6 Scaffolding retirement — 2-3 days

Remove the code paths and tests that only applied to dual-Runner.
Remove the flag. Update `docs/guides/runner-and-session.md` to drop
the "three session ids" section. Close this design doc out (replace
the "proposed" status with "implemented in #...") and link from
RATIONALE.md.

Total: ~4-6 weeks elapsed, ~3-4 engineer-weeks of work. High variance
on §9.3 depending on what §9.1 surfaces.

## 10. Alternatives considered

### 10.1 Goldfive as an ADK plugin

A tempting direction: make goldfive's orchestration a `BasePlugin`
subclass. The caller's ADK tree runs under their own Runner;
goldfive plugs in via `before_agent_callback` /
`after_agent_callback` / `before_model_callback` hooks, reads
session.state, writes session.state, dispatches the intervention
ladder from hooks.

**Rejected, three reasons:**

1. **Regresses pre-execution plan gating.** Today's architecture
   derives goals and generates the plan **before** the first LLM
   call. A plugin-only architecture fires on the first
   `before_agent_callback`, which is mid-turn — too late to gate
   whether to run at all. Workarounds (synthesize a "planning
   phase" sub-agent the tree is supposed to call first) push
   orchestration concerns into user prompts, which goldfive's
   no-prompt-cooperation contract (feedback doc
   `feedback_no_prompt_contract.md`) explicitly rejects.
2. **Puts Level 3 / 4 on worse footing.** Plugin callbacks don't own
   the Runner's event loop the way a wrapper does; cancellation
   across a plugin-triggered state change is ADK's responsibility,
   not goldfive's. Today's wrapper at least has full control over
   the Runner it wraps; a plugin is at the mercy of ADK's plugin
   lifecycle semantics, which are less stable than
   `parent_context=`.
3. **Doesn't apply to non-ADK adapters.** `CallableAdapter` and
   `ClaudeAgentSDKAdapter` would need parallel plugin-like surfaces
   in their respective frameworks. The Runner / AgentAdapter
   abstraction exists specifically so non-ADK frameworks get the
   same orchestration without per-framework plugin development. A
   plugin-only goldfive bifurcates the ADK path from the rest.

Shared-Runner (§3) keeps the AgentAdapter abstraction intact for
non-ADK frameworks — it only changes what `ADKAdapter` does
internally when driven from the wrapper. That preserves both the
Runner abstraction and the no-prompt-cooperation contract while
still eliminating the split.

### 10.2 Drop `goldfive.wrap` for ADK; require bare `Runner` construction

Another option: say adk-web integration is out of scope. Users who
want goldfive on ADK construct `Runner(agent=ADKAdapter(tree))`
explicitly; they don't get the polymorphic `BaseAgent` wrapper.

**Rejected.** The wrapper is the primary onramp — `adk web` + one
call to `goldfive.wrap` is how most users experience goldfive for
the first time. Removing it would bifurcate documentation and lose
the demo path that drives adoption.

### 10.3 Option 7(a) — plumb inner events through the wrapper forever

What's shipping now. `GoldfiveADKAgent._run_async_impl` yields not
just synthesized summary events but also the inner runner's events,
bridged across the dual-Runner boundary. Fixes the immediate
symptom (adk-web not seeing inner events) without restructuring.

**Not rejected — this is what we're doing now and for the
foreseeable future.** 7(c) is a strict superset: it fixes the
root cause rather than patching around it. But 7(a) is ~2 engineer-
weeks; 7(c) is ~4-6 engineer-weeks including a 2-week soak.
7(a) buys us the user-visible fix immediately; 7(c) buys us the
architecture we wish we had. Both can be right at different times.

## 11. References

### Internal code

- `goldfive/adapters/adk_wrap.py` — `GoldfiveADKAgent`, the outer
  wrapper; `_run_async_impl` is the redesign target.
- `goldfive/adapters/adk.py` — `ADKAdapter`, `_build_runner`,
  `_register_plugin_on_runner`, `_ensure_session`,
  `invoke_passthrough`, `_invoke_internal`.
- `goldfive/runner.py` — `Runner.run` lifecycle. Under shared Runner,
  exposes `_derive_goals`, `_generate_plan`, `_finalize` as
  publicly-callable phases for the wrapper.
- `goldfive/executors/sequential.py` — `_run_overlay`,
  `_check_controls` (Level 4 pause wait); today's pre-task loop is
  the Level 4 seam being moved into a plugin callback.
- `goldfive/steerer.py` — `DefaultSteerer`, `_LADDER`,
  `_handle_drift`, `_apply_user_steer_state`,
  `_apply_pause_escalate`.
- `goldfive/adapters/_adk_state_protocol.py` — `_GoldfiveADKPlugin`,
  state mirror logic (shrinks to ordinary writes under shared
  Runner).

### Internal docs

- `docs/design/ARCHITECTURE.md` — top-level architecture, six
  primitives, the Runner.
- `docs/design/CONTROL.md` — control channel, intervention ladder
  §7.d.
- `docs/design/DRIFT.md` — intervention ladder Levels 0-5 per drift
  kind.
- `docs/design/RATIONALE.md` — design rationales; this doc will add
  a §"Why shared-Runner" once implemented.
- `docs/guides/runner-and-session.md` — user guide; will simplify
  once the three-session-id section is no longer needed.

### ADK documentation

- `google.adk.agents.BaseAgent.run_async(parent_context=...)` — the
  child-invocation primitive.
- `google.adk.runners.InMemoryRunner` — the runner the wrapper no
  longer needs a second copy of.
- `google.adk.agents.invocation_context.InvocationContext` — the
  live request queue, the session, the plugin manager.
- ADK plugin lifecycle: `before_run_callback`,
  `before_agent_callback`, `before_model_callback`,
  `before_tool_callback`, and their `after_` counterparts.

### Tracking

- GitHub issue: **Prototype shared Runner/Session via `parent_context=`
  (design alternative to wrapper)** — to be filed as part of this
  design doc's PR. Links back here.
- Related current work: goldfive#196, #197 (7(a), in flight).
- Prior art to read: goldfive#161 (session pinning), #164 (outer
  session id adoption), #155 / #157 (per-event session_id), #170 /
  #173 (state mirror), #122 (plugin propagation), #130 (AgentTool
  cap).
