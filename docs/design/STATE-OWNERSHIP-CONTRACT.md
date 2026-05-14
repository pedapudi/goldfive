# State-ownership contract

**Status.** Phase 0 of [goldfive#271](https://github.com/pedapudi/goldfive/issues/271) — foundation. No behavior change. Defines who owns which state surface and prohibits writes that cross the line.

Related: [ARCHITECTURE.md](ARCHITECTURE.md), [PROTOCOLS.md](PROTOCOLS.md), [PLAN-LIFECYCLE.md](PLAN-LIFECYCLE.md), [STATE-MACHINE.md](STATE-MACHINE.md), the modules `goldfive/orchestration_state.py` and `goldfive/adapters/_adk_state_protocol.py`.

This document is normative for the goldfive `feat/state-ownership-contract` branch and every PR after it. Phases 1+ migrate the catalogued violations one site at a time; the runtime tripwire (`goldfive/_state_audit.py`) prevents new ones.

## 1. Why this exists

Today's E2E (Wave 3) found that goldfive's wrap mutates ADK `session.state` from inside ADK callback paths. ADK considers `session.state` writes exclusive to its own `session_service.append_event` machinery — every direct mutation races with ADK's optimistic-concurrency contract. Symptom in production: stale-session `ValueError` → steerer torn down → 0 observability events emitted. Documented in [goldfive#275](https://github.com/pedapudi/goldfive/issues/275).

The framing question this contract answers: **when a goldfive callback fires inside ADK, what is the callback allowed to write to?**

The answer (Phase 0): **nothing on `ADK Session.state`.** Reads are fine; writes go through goldfive's own orchestration store, and the bridge to ADK happens through ADK's blessed `state_delta` / `append_event` mechanism — not direct `dict.__setitem__`.

Phase 0 itself does not migrate any code to that target. It documents the rule, audits every current violation, and installs a runtime tripwire so future PRs can't add new violations while the migration is in flight.

## 2. The four state surfaces

goldfive code touches four physically-distinct dicts. Conflating them is the root cause of #275.

| Surface | Owner | Lifetime | Mutator |
|---|---|---|---|
| `goldfive.types.Session.state` | **goldfive** | Per `Runner.run()` invocation | Any goldfive component (steerer, reconciler, executor, adapter) writes directly. |
| ADK `Session.state` (the live `session_service`-managed dict) | **ADK** | Per `Runner.run_async` invocation; persisted across turns by the `SessionService` | **ADK only**, via `session_service.append_event` (which carries an `EventActions(state_delta=...)`). Goldfive callbacks must not mutate this dict directly. |
| Plugin-instance state (`self._cancel_state`, `self._invocation_pinned_task_id`, `self._invocation_parents`) | **goldfive plugin** | Lifetime of one `_GoldfiveADKPlugin` instance (typically one `Runner`) | Plugin internals; not part of any session contract. |
| Tool-call argument dicts (`tool_args`) | **ADK** (it constructed the dict from the LLM's `function_call`) | Single tool dispatch | ADK passes the dict through `before_tool_callback`. Mutating it is technically inside ADK's perimeter but ADK's contract treats `tool_args` as caller-owned for the duration of the dispatch — see §6.4. |

Goldfive's "orchestration state" (the first row) is a framework-agnostic dict with its own namespace and writers in `goldfive.orchestration_state`. The bridge between the orchestration dict and ADK `session.state` (the second row) is a copy operation; today that copy is a direct `state[k] = v` against the live ADK dict, which is the violation. Phase 2 replaces it with `append_event(... state_delta=...)`.

## 3. The contract

### 3.1 Goldfive owns

- `goldfive.types.Session.state` — direct read/write from any goldfive component. The `goldfive.orchestration_state` module is the conventional namespace owner; ad-hoc keys are tolerated for app-level use but discouraged in goldfive core.
- `goldfive.types.Session.plan`, `Session.completed_results`, `Session.goals`, `Session.run_id`. Typed fields, not strings — read these instead of poking through `Session.state` keys for the same fact.
- Every plugin-instance dict (`self._cancel_state`, etc.). These are not part of any session contract; they live and die with the plugin instance.

### 3.2 ADK owns

- ADK `Session.state` — read-only from goldfive callback paths. Reads are unrestricted.
- ADK `Session.events` — read-only. Goldfive must never construct an `Event` and `append_event` it onto an ADK session except through `_heal_pending_tool_calls` (§6.5), which is the one currently-blessed exception and itself a target for Phase 3 cleanup.
- The set of `Event` ids and `state_delta` causality. ADK assumes the only writer to `state_delta` is itself; the migration target for Phase 2 is to wrap goldfive's writes in synthetic `Event(actions=EventActions(state_delta={...}))` objects appended via `session_service.append_event` so ADK's optimistic-concurrency machinery sees them.

### 3.3 The cross-boundary rule

> **From inside any goldfive ADK callback (`before_run_callback`, `before_agent_callback`, `before_model_callback`, `before_tool_callback`, `after_tool_callback`, `after_agent_callback`, `after_run_callback`), goldfive code MUST NOT mutate `callback_context.state`, `tool_context.state`, `invocation_context.session.state`, or any alias thereof, by direct subscripting / `pop` / `update` / `setdefault`.**

The same rule covers anything reachable by walking attributes from the callback context to the live ADK session — see `_session_state_from_callback` for the concrete chain.

### 3.4 Reads are allowed

`callback_context.state.get("goldfive.current_task_id", "")`, iteration of `state.items()`, and any other read pattern is fine. The contract is about writes only.

### 3.5 What about `Session.state` (the goldfive one)?

That dict is goldfive's. Writing to it from a callback is allowed and frequently necessary — the steerer / reconciler are themselves invoked from callback paths and must update orchestration state. The rule above is specifically about the **ADK** `Session.state`, which is what `_session_state_from_callback` returns.

The two dicts are easy to confuse at the call site because both end in `.state`. The migration target makes the distinction syntactic: orchestration writes go through `goldfive.orchestration_state.write`, ADK writes go through a future `OrchestrationStore.publish_to_adk(session, key, value)` helper that emits an event under the hood.

## 4. The migration roadmap (#271)

| Phase | Scope | Status |
|---|---|---|
| 0 | This document, audit catalog (§5), runtime tripwire (`goldfive/_state_audit.py`) | **Done (#278)** |
| 1 | Introduce `OrchestrationStore` — single typed handle that owns goldfive's orchestration state. Extract reasoning / pin / cancel state off `Session.state` onto the store. No ADK-side change yet. | **Done (#279)** |
| 2.0 | Eliminate the bridge (V2) and the now-unused initial seeds (V1, V5). The dynamic-instruction resolver and `GoldfivePlanner` read goldfive `Session.state` directly via the `SessionContext` stash + `OrchestrationStore`. Closes goldfive#275. | **Done (Phase 2.0)** |
| 2.1 | Migrate the per-agent pin (V3) and the delegation-site pin (V4). Both move to goldfive `Session.state` exclusively via `OrchestrationStore`; readers consult goldfive Session via the plugin reference. After this phase, no callback-time write to ADK `session.state` from inside the wrap remains. | **Done (Phase 2.1)** |
| 2.x | Clean up V7 / V8 — the `SessionContext` stash is dead in production paths but legacy tests still drive through it. | Planned |
| 3 | Tripwire flips on by default in production. Catalog is empty. | Planned |

A future contributor can verify a Phase-2 PR by:

1. Reading the §5 catalog: every entry it removes is one fewer violation.
2. Running the tripwire suite: tests that exercise the migrated path now pass without an opt-out.
3. Confirming the catalog and the tripwire's `_KNOWN_OPT_OUT_CALLERS` allowlist drop in lockstep — when both are empty, the migration is complete.

## 5. Audit catalog

Every place goldfive writes to ADK `session.state` (or a structure ADK considers exclusively its own). Sorted by severity. "ADK callback context" means the write happens inside one of ADK's `before_*` / `after_*` plugin hooks; "Adapter entry" means the write happens before ADK's runner starts streaming.

### Severity legend

- **blocker** — implicated in #275 today; race-prone with ADK's optimistic-concurrency model. Phase 2 must migrate.
- **mitigated** — racy in theory but currently lands during a quiescent window (e.g. before any ADK event has been appended). Phase 2 can defer.
- **cosmetic** — write technically violates the rule but the value is plugin-private, ADK never reads it, and no race is observable. Phase 2 can collapse onto an instance dict.

### 5.1 Plugin-callback writes (blockers)

#### V1 — `before_run_callback`: initial seed — **MIGRATED (Phase 2.0)**

- **Status.** Eliminated by Phase 2.0 of goldfive#271. Nothing on the ADK side reads the seeded keys: the dynamic-instruction resolver and `GoldfivePlanner.build_planning_instruction` now read goldfive `Session.state` directly via the `SessionContext` stash (V7) and the `OrchestrationStore` typed accessor.
- **Original target.** `goldfive/adapters/_adk_plugin.py:1656-1664` — wrote `goldfive.run_id`, `goldfive.plan_id`, `goldfive.plan_summary`, `goldfive.available_tasks`, `goldfive.completed_task_results`, `goldfive.current_task_id`, `goldfive.current_task_title`, `goldfive.current_task_description`, `goldfive.current_task_assignee`, `goldfive.tools_available`.

#### V2 — `before_run_callback`: orchestration-state bridge — **MIGRATED (Phase 2.0)**

- **Status.** Eliminated by Phase 2.0 of goldfive#271. The literal site of goldfive#275. The bridge function `_bridge_orchestration_state` and its inline subroutine `_bridge_pending_corrections` are deleted; the resolver / planner read goldfive `Session.state` directly via `OrchestrationStore`.
- **Original target.** `goldfive/adapters/_adk_plugin.py:1679` called `_bridge_orchestration_state` to mirror `goldfive.active_steer.body`, `goldfive.active_steer.at_turn`, `goldfive.goals_summary`, `goldfive.cancelled_function_call_ids`, and every `goldfive.pending_corrections.<agent>.<task>` key onto ADK `session.state`.

#### V3 — `_stamp_current_task_id` (called from `before_agent_callback`) — **MIGRATED (Phase 2.1)**

- **Status.** Eliminated by Phase 2.1 of goldfive#271. The per-agent pin lands on goldfive `Session.state` exclusively via `OrchestrationStore.set_pin_current_task`. Readers (the dynamic-instruction resolver, the reporting handlers, `_resolve_pinned_task_id`) consult goldfive Session via the plugin reference (`session_context_from_invocation`) — no callback-time write to ADK `session.state` remains.
- **Original target.** `goldfive/adapters/_adk_plugin.py:2620-2628` wrote `goldfive.current_task_id`, `goldfive.current_task_title`, `goldfive.current_task_description`, `goldfive.current_task_assignee`, `goldfive.current_task_revision` onto both surfaces. The protocol-module writers (`_sp.write_current_task` / `_sp.write_current_task_id` / `_sp.write_current_task_revision` / `_sp.clear_current_task`) are deleted; only the read-side key constants remain.

#### V4 — `_pin_delegation_task_id` (called from `before_tool_callback`) — **MIGRATED (Phase 2.1)**

- **Status.** Eliminated by Phase 2.1 of goldfive#271. The per-`function_call_id` pin lands on goldfive `Session.state` exclusively via `OrchestrationStore.set_pending_delegation`. The reporting-tool callback's `_resolve_pinned_task_id` reads the same store via the plugin reference.
- **Original target.** `goldfive/adapters/_adk_plugin.py:2778-2783` wrote `goldfive.pending_delegations` (a dict keyed by `function_call_id`) onto both `ctx.session.state` (orchestration) AND the ADK `tool_context` session.state.

#### V5 — `before_model_callback`: defensive duplicate seed — **MIGRATED (Phase 2.0)**

- **Status.** Eliminated by Phase 2.0 of goldfive#271. Falls out together with V1: the same keys, the same race surface, the same readers (resolver / planner) all migrated to read goldfive `Session.state` directly. Only the planner's request-side `_inject_goldfive_planner_instruction` path remains in `before_model_callback` (it does not write state — it appends to `llm_request.config.system_instruction`).

### 5.2 Plugin-callback writes (cosmetic)

#### V6 — `_inject_task_id_from_state`: tool_args mutation

- **File:line.** `goldfive/adapters/_adk_plugin.py:362-364`
- **State written.** `tool_args["task_id"]` — mutates the dict ADK passes into `before_tool_callback`.
- **Why.** Reporting-tool schemas hide `task_id` from the LLM (#241), so every reporting tool call lands here with no `task_id`. This injects the resolved id from session.state so the handler can dispatch.
- **Severity.** **cosmetic.** ADK's contract on `tool_args` is fuzzy: the dict is constructed from the LLM's `function_call` and ADK doesn't re-read it after `before_tool_callback` returns (ADK trusts the callback to fully resolve args). No ADK race observed; the failure mode if migrated would be functional, not concurrent. Listed here for completeness.
- **Migration target.** Stay. This is the one "write" we expect to keep — it lives inside a perimeter ADK has already handed to the callback, and replacing it with `append_event` makes no sense (the value is per-dispatch transient).
- **Tripwire status.** Not flagged by the tripwire — it monitors `session.state` writes specifically. `tool_args` is a separate dict.

### 5.3 Adapter-entry writes (mitigated)

#### V7 — `ADKAdapter.invoke`: SessionContext stash

- **File:line.** `goldfive/adapters/adk.py:1635`
- **State written.** `goldfive._session_context` (a `SessionContext` instance, not a goldfive.* prefixed key).
- **Why.** Best-effort fallback for legacy unit tests that construct a plain `tool_context` holding a populated state dict and drive the plugin directly. The live-run path does not depend on this write.
- **Severity.** **mitigated.** Fires once per `invoke` *before* the `Runner.run_async` loop starts — so before any ADK event exists to race with. The "live-run path does not depend on this write" comment in source is accurate; deleting the line breaks only legacy unit tests that drive the plugin out of band.
- **Migration target.** Phase 2 cleanup. Delete the line, rewrite the affected unit tests to plumb the `SessionContext` through the plugin's public API. Acceptable to defer past Phase 2 and tackle as part of a test-cleanup pass.

#### V8 — `ADKAdapter.invoke`: SessionContext cleanup

- **File:line.** `goldfive/adapters/adk.py:1770`
- **State written.** `state.pop("goldfive._session_context", None)` in the `invoke` finally clause.
- **Why.** Companion cleanup to V7 — clears the stashed `SessionContext` once the invoke completes so a later invoke against the same session doesn't see stale context.
- **Severity.** **mitigated.** Fires after the `run_async` loop has fully drained, so again no live race.
- **Migration target.** Same as V7 — both go away together.

### 5.4 Implicit writes via the heal path (mitigated)

#### V9 — `_heal_pending_tool_calls`: cancelled-id stamping

- **File:line.** `goldfive/adapters/adk.py:1908-1913` calls `_ostate.append_cancelled_function_call_ids` against `session.state`.
- **State written.** `goldfive.cancelled_function_call_ids` (on the goldfive `Session.state` dict, not ADK's). Then the bridge V2 mirrors it onto ADK `session.state` on the next `before_run_callback`.
- **Why.** When an ADK invocation cancels mid-flight with pending tool calls, goldfive synthesises `function_response` events to satisfy the dangling tool-call ids and records them on orchestration state so downstream planners / prompt templates know which ids were cancelled.
- **Severity.** **mitigated.** The write itself is to goldfive's dict (compliant). The bridge V2 propagates it to ADK on the next invocation — also outside any racing live ADK event because `before_run_callback` fires before any new events.
- **Migration target.** Phase 2. Recategorised once V2 migrates: when the bridge is gone, this entry collapses.

### 5.5 Already-compliant writes (sanity check)

The following writes touch `goldfive.types.Session.state` (the goldfive-owned dict) and are NOT violations. Listed so future contributors don't double-flag them.

| File:line | What | Why compliant |
|---|---|---|
| `goldfive/orchestration_state.py:179` | `state[key] = value` inside `write()` | Caller passes goldfive `Session.state`; module's docstring asserts the surface is goldfive's. |
| `goldfive/orchestration_state.py:185` | `state.pop(key, None)` | Same as above. |
| `goldfive/_correction_injection.py:217` | `state[key] = dict(correction)` | Caller is `write_correction`, takes a `goldfive.Session`; key is `goldfive.pending_corrections.*`. The bridge V2 mirrors to ADK. |
| `goldfive/_correction_injection.py:352, :394` | `state.pop(key, None)` | Same. |
| `goldfive/plan_reviser.py:1322` | `state[_ostate.KEY_CURRENT_TASK_ID] = resolved` | Inside `PlanReviser._emit_plan_revised`'s supersession-rewrite path; `state` is the goldfive `Session.state`. (Moved from `goldfive/steerer.py` in #410 facade-cleanup.) |
| `goldfive/adapters/_adk_state_protocol.py:159` etc. | `state[key] = value` inside the protocol module's `_set` and the `set_*_on_adk_state` helpers. | These are the *helpers* that target the ADK side. The violations are at the **call sites** (V1-V5), not in the helper itself. The tripwire flags the call sites (which run inside callbacks) and lets the helpers themselves run uninstrumented. |

## 6. Edge cases the contract explicitly addresses

### 6.1 Sub-Runner propagation

`AgentTool` spawns a sub-Runner with its own `Session`. The plugin's `before_run_callback` fires against that sub-session — it's not the same dict as the parent. The current bridge re-runs on every sub-invocation (V2 above) so propagation works. Post-migration: `append_event` is per-session too, so the same shape applies — the sub-Runner publishes its own events.

### 6.2 Concurrent sinks

Sinks can be coroutines that run concurrently with the adapter loop (`harmonograf.GoldfiveSink` is one). Sinks must not write to ADK `session.state` either — same rule. The tripwire flags any goldfive code (sink, drift detector, planner, anything) that mutates ADK state from within an active goldfive callback context.

### 6.3 The `tool_args` carve-out

V6 mutates `tool_args["task_id"]`. As §3.3 / V6 explain, this is not technically a `session.state` write — `tool_args` is a separate dict ADK passes through the callback expecting it to be filled in. The tripwire does not flag this. Phase 2 keeps it.

### 6.4 The heal-path `append_event`

`ADKAdapter._heal_pending_tool_calls` calls `session_service.append_event` to synthesise `function_response` events for orphan tool calls. **This is the blessed mechanism** — same one Phase 2 will migrate the bridge onto. Heal-path retains its `append_event` calls; future bridge code adds more.

### 6.5 The plugin-instance state dicts

`self._cancel_state`, `self._invocation_pinned_task_id`, `self._invocation_parents` are plugin-local. They are not part of any session contract; they live and die with the plugin instance. Reads / writes to these are unrestricted and the tripwire ignores them. (The cancellation-flag entry the brief mentions, `KEY_CANCEL_REQUESTED`, is currently held in `self._cancel_state` — a plugin-local dict — not on ADK `session.state`. The state-protocol module exposes `read_cancel_request` / `write_cancel_request` against an arbitrary `MutableMapping`; goldfive's plugin uses the local dict, harmonograf-side bridges use the helper against their own.)

## 7. The runtime tripwire

`goldfive/_state_audit.py` installs an opt-in guard at the boundary between goldfive callback code and ADK's session.state mutation surface. Mechanism (Phase-0 minimal):

1. **Active-callback ContextVar.** `_active_callback` is a `contextvars.ContextVar` set / cleared at the entry / exit of every goldfive plugin callback. `wrap_plugin_callbacks(plugin)` (called once from `make_adk_plugin`) wraps each of the eight callback methods so the bookkeeping is automatic — no decorator at the call site required.
2. **Funnel patch.** `_install_protocol_patch` rewrites `_adk_state_protocol._set` (the single funnel through which every protocol-module writer goes) to call `_check_caller(...)` before performing the underlying mutation.
3. **Stack walk against catalog.** `_check_caller` walks the live call stack for a frame whose `(filename suffix, qualname suffix)` matches an entry in `_KNOWN_CALLERS`. Each entry is one violation in the §5 catalog. Match -> allow; no match -> raise.

Phase 0 leaves all current call sites intact (no behaviour change); the catalog covers every existing site, so a clean baseline is silent. New writes at un-catalogued (file, function) pairs raise.

### 7.1 Why `StateOwnershipViolation` inherits from `BaseException`

Many of the catalogued sites are inside broad `try / except Exception` blocks — state writes are best-effort by design, so the existing code swallows arbitrary exceptions to avoid crashing a run on a missing key. If `StateOwnershipViolation` were an `Exception` subclass, those defensive blocks would silently swallow the audit's signal — the same failure mode #275 is suffering from in production today, where ADK's stale-session `ValueError` is itself caught and dropped by the same blocks.

Inheriting from `BaseException` ensures the violation propagates through `except Exception` and is only caught by an explicit `except BaseException` (or unwrapped propagation to the test runner). In production with the tripwire disabled, the audit never raises, so the `BaseException` choice is invisible. In tests with the tripwire enabled, a new violation surfaces loudly even when triggered through a defensively-wrapped call site.

### 7.2 Defaults

The tripwire is **off by default in production** (`GOLDFIVE_STRICT_STATE_OWNERSHIP` unset on a fresh deploy). It is **on by default in tests** — `tests/conftest.py` sets the env var via the auto-applied `_state_audit_enabled` fixture. CI runs with it enabled.

| Env var | Effect |
|---|---|
| `GOLDFIVE_STRICT_STATE_OWNERSHIP=1` | Force-enable everywhere. |
| `GOLDFIVE_STRICT_STATE_OWNERSHIP=0` | Force-disable everywhere (tests opt out via `no_state_audit` fixture). |
| unset | Auto: on inside pytest (`"pytest" in sys.modules`), off elsewhere. |

See the module docstring of `goldfive/_state_audit.py` for the full API.

## 8. What changes if Phase 2 fails

If Phase 2's `append_event` migration has worse performance characteristics than the direct-write today, the fall-back is to keep the direct write but *gate it behind ADK's session lock*. The state-ownership rule still holds: writes from goldfive go through one helper, which is the place ADK can grow a lock-holding wrapper. Today's nine call sites can't be locked individually without a refactor, which is itself the Phase 1 work. So Phase 1 stands either way.
