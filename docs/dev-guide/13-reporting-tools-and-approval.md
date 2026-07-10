# 13. Reporting Tools and Approval

## Read this chapter when...

- You are adding, removing, or renaming a reporting tool (anything in the
  `report_task_*` / `report_*` / `declare_task_*` families).
- You are touching the JSON schema an agent sees for a reporting tool, or
  changing what the tool's response looks like on the model's next turn.
- You are working on the human-in-the-loop approval flow
  (`report_awaiting_approval`, `ApprovalRequested` / `ApprovalGranted` /
  `ApprovalRejected`, the control-channel `APPROVE` / `REJECT` dispatch).
- You are debugging a "the agent called `report_task_completed` and nothing
  happened" or "the run hung on an approval" report.
- You are wiring a new adapter and need to know how reporting-tool specs turn
  into native tool definitions and how their handlers get dispatched.
- You need to understand *why* these tools are an optional accelerator and how
  the framework still works when an agent never calls a single one.

## Files covered

| File | Role |
| --- | --- |
| `goldfive/reporting/__init__.py` | Public re-export surface; the package split map. |
| `goldfive/reporting/handlers.py` | The contract: `ReportingToolSpec`, the `BUILTIN_REPORTING_TOOLS` catalog, every `_handle_*` async handler, `select_reporting_tools`, declaration constants. |
| `goldfive/reporting/schemas.py` | JSON-schema parameter blocks (`_SCHEMA_*`) sent over the wire to LLMs. |
| `goldfive/reporting/rendering.py` | Response-shape construction: directive acks, `plan_state`, idempotent / invalid / refused / missing-arg shapes. |
| `goldfive/reporting/_internal.py` | Private helpers: arg coercion, `task_id` pin resolution, pin-freshness classification, sink emitters. |
| `goldfive/adapters/adk.py` | `register_reporting_tools`, `_augment_subtree_with_reporting`, `_build_function_tool` / `_build_ack_shim`, the two `RuntimeError` integrity guards. |
| `goldfive/adapters/_adk_plugin.py` | `before_tool_callback` interception; `task_id` injection; the Flow-B tool-confirmation bridge. |
| `goldfive/executors/_control.py` | `_resolve_approval` — the control-channel side of the approval handshake. |
| `goldfive/config.py` | `DEFAULT_APPROVAL_TIMEOUT_MS`, `SteeringConfig.approval_default_timeout_ms`. |
| `goldfive/runner.py` | Step 5 of `Runner.run` — the single registration call site. |
| `docs/design/APPROVAL.md` | Design ref for the three human-in-the-loop scenarios (code on main wins where they disagree). |

## Invariants that bind you here

1. **No prompt-cooperation contracts.** (CANON invariant 1.) Reporting tools
   are an *optional accelerator*. Termination, control, and observability MUST
   work when the agent never calls a single reporting tool. Never move
   load-bearing logic into a handler such that the framework degrades to
   *broken* (rather than *slower / less precise*) when the tool is not called.
   The degradation paths are enumerated in
   [§ The no-cooperation tension](#the-no-cooperation-tension-these-tools-are-optional).
2. **No regex/keyword heuristics for NL classification.** (CANON invariant 2.)
   Handlers do *structured* classification only — exact-equality status checks
   (`_classify_transition`), `(kind, task_id)` string-key dedup, revision-int
   comparison. That is allowed. Do not add a handler that string-matches free
   text to decide what a tool "meant."
3. **Any ADK tree shape must work, including coordinator + AgentTool.** (CANON
   invariant 3.) `register_reporting_tools` must land the tools on *every*
   reachable agent — root, `sub_agents`, `inner_agent`, and nested
   `AgentTool.agent`. The `RuntimeError` integrity guard in
   `ADKAdapter.register_reporting_tools` enforces this; do not weaken it.
4. **Adaptive over predictive.** (CANON invariant 4.) A handler records the
   fact the agent reported (a declaration, a transition) and lets the
   observation pipeline reconcile. It does not predict what the agent will do
   next. `declare_task_*` deliberately does NOT mutate the plan.
5. **`observation_only=True` is the strict-passive production default.** (CANON
   invariant 5.) Any goldfive-authored *directive* the model reads — here, the
   `plan_state` block on a directive ack — is a steering surface and must be
   gated on the single sanctioned predicate
   `goldfive.steerer.steering_is_active(steerer)`. Under strict-passive the ack
   keeps only the factual echo of what the agent itself reported. See
   [§ The observation_only gate on plan_state](#the-observation_only-gate-on-plan_state)
   and `09-steering-ladder-and-gates.md`.
6. **Lifecycle gates need stable identity keys.** (CANON invariant 6.) The
   declaration dedup key is `f"{kind}:{task_id}"` — both components are stable
   plan-side ids, never LLM-minted churning strings. Do not re-key any gate in
   this subsystem on a value the model authors turn-to-turn.

---

## 1. What a reporting tool is, and the inventory

A **reporting tool** is a framework-agnostic tool that an agent *may* call to
tell goldfive what it is doing: "I started task t7", "I finished t7", "I found
new work", "please get a human to approve this." goldfive routes the call into
the steerer's task state machine / drift pipeline, emits observability events,
and returns a payload the model reads on its next turn.

The tool is not the source of truth. The steerer's transition machinery and the
observation pipeline (reconciler, judges, detectors) are. A reporting tool is a
*faster, more precise* signal than observation — the agent telling you directly
beats inferring from delegation events — but the framework never *depends* on
it. See [§ The no-cooperation tension](#the-no-cooperation-tension-these-tools-are-optional).

### 1.1 The canonical inventory: 8 + 2 = 10 tools

The canonical names are pinned in `REPORTING_TOOL_NAMES` in
`goldfive/reporting/handlers.py`:

```python
# goldfive/reporting/handlers.py
REPORTING_TOOL_NAMES: tuple[str, ...] = (
    "report_task_started",
    "report_task_progress",
    "report_task_completed",
    "report_task_failed",
    "report_task_blocked",
    "report_new_work_discovered",
    "report_plan_divergence",
    "report_awaiting_approval",
    "declare_task_skipped",
    "declare_task_not_needed",
)
```

> **Note on the count.** The module docstrings historically say "the eight
> canonical reporting tools" and the `REPORTING_TOOL_NAMES` comment says "The
> ten canonical reporting tool names." The tuple is the ground truth: **ten**
> tool names. The "eight" phrasing predates the goldfive#271 Phase-3 addition
> of `declare_task_skipped` / `declare_task_not_needed`. When you touch this
> module, treat the `REPORTING_TOOL_NAMES` tuple + `BUILTIN_REPORTING_TOOLS`
> list as canonical and ignore the stale "eight" prose in the docstrings (fix
> it if you are in there anyway — it is a #492-style accuracy nit).

Grouped by behaviour:

| Group | Tools | What the handler does |
| --- | --- | --- |
| **Lifecycle (7)** | `report_task_started`, `report_task_progress`, `report_task_completed`, `report_task_failed`, `report_task_blocked`, `report_new_work_discovered`, `report_awaiting_approval` | Drives the steerer's task state machine / drift pipeline. Default-**on**. |
| **Drift self-report (3, opt-in)** | `report_plan_divergence`, `declare_task_skipped`, `declare_task_not_needed` | Off by default. Gated behind `Runner(drift_self_reporting=...)`. |

Note the split is **7 lifecycle + 3 drift**, not "7 lifecycle + 3 drift +
2 declare" — `declare_task_skipped` and `declare_task_not_needed` *are* two of
the three opt-in drift tools. The third is `report_plan_divergence`. This is
enumerated by `DRIFT_SELF_REPORTING_TOOL_NAMES`:

```python
# goldfive/reporting/handlers.py
DRIFT_SELF_REPORTING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "report_plan_divergence",
        "declare_task_skipped",
        "declare_task_not_needed",
    }
)
```

`report_new_work_discovered` is deliberately **not** in the drift-opt-in set —
there is no observation analog for an agent surfacing genuinely new work, so it
stays default-on (goldfive#196). Do not "tidy" it into the drift set.

### 1.2 The 7/3 split rationale (goldfive#196)

The drift tools overlap with observation-driven detectors: the reconciler
(`goldfive.reconciler.PlanReconciler`), the trajectory-level goal-drift judge
(`classify_goal_drift`), and the steerer's refine machinery. Registering the
drift tools on *every* sub-agent is pure downside when observation already
covers them:

- Each tool inflates prompt size by **~200–400 tokens** (schema + description).
- Each tool expands the agent's **hallucination surface**: a confused model can
  confabulate a `report_plan_divergence` call and trigger a spurious replan.

So the Runner gates the three drift opinions behind an opt-in flag while the
seven lifecycle tools stay default-on. `LIFECYCLE_REPORTING_TOOLS` is *derived*
(the subset of `BUILTIN_REPORTING_TOOLS` whose name is not in
`DRIFT_SELF_REPORTING_TOOL_NAMES`), so adding a new lifecycle tool flows through
automatically — do not hand-enumerate it.

---

## 2. Schemas: why `task_id` is hidden

The over-the-wire parameter blocks live in `goldfive/reporting/schemas.py`. Each
is a plain `dict[str, Any]` built by `_object_schema(required=..., properties=...)`.
Adapters embed them verbatim into native tool definitions.

### 2.1 `task_id` is absent from every `required[]` list — on purpose

This is the single most counterintuitive fact about the schemas. Read the
module docstring:

```python
# goldfive/reporting/schemas.py
# NOTE: ``task_id`` is intentionally omitted from every schema's
# ``required`` list (goldfive#191). The adapter stamps
# ``goldfive.current_task_id`` onto session state at delegation time
# so the handler can default from state when the model doesn't supply
# the arg. Handlers still reject with the canonical
# ``missing_task_id`` shape when neither source resolves a value —
# so strictness is enforced at the handler layer, not the schema.
```

There are actually **two** levels of `task_id` hiding, and you must not confuse
them:

1. **Not in `required[]` (schema level).** `task_id` still appears in the
   `properties` block of the task-scoped schemas — the model *may* supply it,
   but is not forced to.
2. **Not in the ADK tool *declaration* at all (adapter level).** For the ADK
   adapter, `_apply_llm_signature` (`goldfive/adapters/adk.py`) attaches a
   restricted `__signature__` to the FunctionTool shim so ADK's introspection
   builds a declaration that *omits `task_id` entirely* for the five built-in
   task-scoped tools. The model never sees the field. The plugin's
   `before_tool_callback` fills it in from state before the handler runs
   (goldfive#241).

Why go to this trouble? Live evidence (quoted in `_build_ack_shim`'s docstring):

> "the function is still trying to use a task_id parameter even though the
> schema says it doesn't require any parameters"

Exposing `task_id` to the model led LLMs to either hallucinate bad ids or
**abandon the reporting protocol entirely** out of optional-arg confusion. The
pin — `goldfive.current_task_id`, stamped by the plugin's
`before_agent_callback` when the sub-agent has exactly one PENDING/RUNNING task
assigned — is almost always the right answer anyway, so hiding the field and
defaulting from state is both more robust and more accurate.

### 2.2 The `required[]` content fields (enforced at the handler)

`required[]` lists the *content* fields the LLM must actually author:

| Schema | `required` | Optional `properties` |
| --- | --- | --- |
| `_SCHEMA_TASK_STARTED` | *(none)* | `task_id`, `detail` |
| `_SCHEMA_TASK_PROGRESS` | *(none)* | `task_id`, `fraction` (0.0–1.0), `detail` |
| `_SCHEMA_TASK_COMPLETED` | `summary` | `task_id`, `artifacts` (map<str,str>) |
| `_SCHEMA_TASK_FAILED` | `reason` | `task_id`, `recoverable` (bool) |
| `_SCHEMA_TASK_BLOCKED` | `blocker` | `task_id`, `needed` |
| `_SCHEMA_NEW_WORK_DISCOVERED` | `parent_task_id`, `title`, `description` | `assignee` |
| `_SCHEMA_PLAN_DIVERGENCE` | `note` | `suggested_action` |
| `_SCHEMA_AWAITING_APPROVAL` | `prompt` | `task_id`, `timeout_ms` (int ≥ 0) |
| `_SCHEMA_DECLARE_TASK_SKIPPED` | `reason` | `task_id` |
| `_SCHEMA_DECLARE_TASK_NOT_NEEDED` | `reason` | `task_id` |

`required[]` is enforced by `_validate_required` in `handlers.py` at handler
entry — see [§ 4.1](#41-required-field-validation-the-v15-cascade-fix). The
`required[]` list never includes `task_id` even where the handler needs one;
that rejection flows through `_missing_task_id_response` *after* the pin
fallback.

Every schema uses `additionalProperties: False`. The `schemas.py` docstring
notes that some engines don't honour that — the marshaling layer is responsible
for stripping it if needed, not the schema author.

---

## 3. Registration: from spec to every reachable agent

### 3.1 The single call site (Runner step 5)

`Runner.run` registers reporting tools exactly once per turn, in step 5:

```python
# goldfive/runner.py (Runner.run, step 5)
try:
    await self.agent.register_reporting_tools(
        select_reporting_tools(self.drift_self_reporting)
    )
except Exception as exc:  # noqa: BLE001
    log.exception("register_reporting_tools raised")
    return await self._abort_turn(...)
```

`select_reporting_tools(drift_self_reporting)` (in `handlers.py`) picks the spec
list:

- `False` (Runner default) → `LIFECYCLE_REPORTING_TOOLS` only.
- `True` → full `BUILTIN_REPORTING_TOOLS` (legacy: every tool).
- iterable of names → lifecycle subset **plus** the named drift tools. Names not
  in `DRIFT_SELF_REPORTING_TOOL_NAMES` are silently ignored (a typo can't
  accidentally enable a non-drift tool; lifecycle tools are always on). An empty
  iterable collapses to the `False` case.

`_abort_turn` on failure is the correct fail-closed behaviour: if the tools
can't be registered we abort the turn rather than run half-instrumented. Do not
downgrade this to a `log.warning` and proceed.

### 3.2 ADK: `register_reporting_tools` → subtree augmentation

`ADKAdapter.register_reporting_tools` (`goldfive/adapters/adk.py`) does the
heavy lifting for the ADK tree:

1. For each spec, record `self._tool_handlers[name] = handler` (this is the map
   the plugin dispatches through — the FunctionTool body itself is a no-op
   shim, see below), and build a `FunctionTool` via `_build_function_tool`.
2. Attach the tools to the **root** agent.
3. Call `_augment_subtree_with_reporting(self._agent, function_tools, names)` to
   walk the tree and append the tools to every reachable agent.
4. Run the **integrity check** (see [§ 3.4](#34-the-runtimeerror-integrity-guard)).

`_augment_subtree_with_reporting` traverses **three** edge kinds so every ADK
shape is covered (CANON invariant 3):

- `sub_agents` — native ADK agent tree.
- `inner_agent` — wrapper agents.
- `AgentTool.agent` — agents exposed to a parent as tools.

It is idempotent: an agent already carrying any canonical reporting-tool name is
skipped. Coverage across the *whole* tree matters because an AgentTool
sub-invocation can itself report terminal status for the outer task — so every
reachable agent needs the tools regardless of which one drives each turn. This
is also what makes early-exit on `_task_is_terminal` work inside an AgentTool
sub-invocation.

### 3.3 The no-op shim: the handler runs in the plugin, not the tool body

`_build_ack_shim(name, description)` returns a sync function whose body is just
`return {"acknowledged": True}`. **This shim is never actually executed as the
tool.** Its only jobs are (a) give ADK a callable with a proper `__name__` /
docstring for the declaration, and (b) carry the restricted `__signature__` that
hides `task_id`.

The real routing happens in `_GoldfiveADKPlugin.before_tool_callback`
(`goldfive/adapters/_adk_plugin.py`): the plugin recognises the tool name,
injects `task_id` from state (`_inject_task_id_from_state`), looks up the async
handler in `self._tool_handlers`, `await`s it, and returns the handler's dict as
the tool response the model sees. Returning a non-None dict from an ADK
`before_tool_callback` tells ADK to skip the underlying tool and use the dict as
the response — so the shim body never runs.

> **Weak-model trap.** If you "fix" a reporting tool by editing the shim body in
> `_build_ack_shim` to do real work, nothing will happen at runtime — the shim
> is dead code by design. The behaviour lives in the `_handle_*` functions in
> `handlers.py` and their dispatch in the plugin. Edit those.

### 3.4 The `RuntimeError` integrity guard

After augmentation, `register_reporting_tools` re-walks the tree (same three
edges) and asserts every reachable named agent carries every reporting-tool name
just registered. A gap raises:

```python
# goldfive/adapters/adk.py (register_reporting_tools, tail)
raise RuntimeError(
    f"ADKAdapter: reporting-tool set did not land on "
    f"{len(missing)} reachable agent(s): {details}. "
    f"Expected every reachable agent to carry the "
    f"reporting tools so terminal status can flow back "
    f"from an AgentTool sub-invocation. See "
    f"_augment_subtree_with_reporting."
)
```

Rationale: a partial augmentation leaves a sub-agent with no way to report
terminal status. If a coordinator delegates to it via AgentTool, the outer task
cannot finish via the early-exit optimization. The guard is skipped only in
`_degraded_prebuilt_runner` mode (the adapter only sees the root there).

There is a **second, unrelated** `RuntimeError` in `ADKAdapter.__init__`
(around the plugin-install check): it fires when the goldfive plugin failed to
install on the runner, because then "reporting callbacks, state-protocol writes,
and drift observation would all be broken." Don't conflate the two — the first
is about tool *coverage*, the second about the *plugin* being present at all.

---

## 4. Handler flow, per tool

Every handler has the signature `ReportingHandler`:

```python
# goldfive/reporting/handlers.py
ReportingHandler = Callable[
    [dict[str, Any], "Session", "Steerer"],
    Awaitable[dict[str, Any]],
]
```

It receives decoded args, the live `Session`, and the bound `Steerer`, and
returns a JSON-serializable dict. The task-scoped lifecycle handlers
(`report_task_started/progress/completed/failed/blocked`) all follow the same
pipeline; the two drift handlers and the two declarations are simpler.

### 4.1 Required-field validation (the v15-cascade fix)

`_validate_required(args, schema, tool_name)` runs at handler entry for every
handler with a non-empty `required[]`. It rejects a missing key, an explicit
`None`, or a whitespace-only string, returning the canonical
`missing_required_field` shape *instead of driving the steerer*.

Why it exists: pre-fix, handlers `_str`-coerced missing fields to `""` and
forwarded the empty payload to the steerer, where it became drift detail like
`"new work under : : "` — semantically-empty signals the planner correctly
declined to act on, leaving the agent in a no-op revision-tool loop. Numbers and
booleans are accepted as-is when present (a literal `0` / `False` is valid; only
absence is a violation). `task_id` is never in `required[]`, so this helper only
guards *content* fields.

### 4.2 The task-scoped pipeline (worked example: `report_task_completed`)

`_handle_task_completed` in `handlers.py` is the canonical shape. In order:

1. `_validate_required(args, _SCHEMA_TASK_COMPLETED, "report_task_completed")` —
   reject if `summary` missing/empty.
2. `task_id, source = _resolve_task_id_with_source(args, session)` — resolve the
   target task (explicit arg > adapter pin). `source` is `"llm_report"` or
   `"handler_default"` for later transition provenance.
3. Coerce `summary` / `artifacts`.
4. `if not task_id: return _missing_task_id_response(...)` — neither source
   resolved a value.
5. `await _await_plan_stable(session, steerer)` — soft barrier against a
   concurrent fire-and-forget refine mutating the plan mid-read (goldfive a4).
   Best-effort: duck-types `steerer.plans._wait_plan_stable`; times out and
   proceeds.
6. `_classify_and_route_pin(...)` — the goldfive#266 pin-freshness gate. Returns
   `(effective_task_id, refusal_or_None, rerouted)`. If `refusal is not None`
   the handler returns it verbatim (a `_refused_response()` ack). If `rerouted`,
   set `source = "supersedes_reroute"`. See [§ 5](#5-pin-resolution-and-freshness-the-266-classifier).
7. `task = _find_task_in_session(session, task_id)`; if found,
   `_classify_transition(...)` → one of `"idempotent"` / `"invalid"` /
   `"transition"`. Idempotent returns `_idempotent_response`; invalid returns
   `_invalid_transition_response`; otherwise fall through.
8. `await steerer.tasks.mark_task_completed(task_id, ..., source=source)` — the
   actual state-machine drive.
9. `_rotate_after_terminal(session, task)` — advance the `current_task_id` pin
   now the task is terminal.
10. `return _directive_ack(session=..., new_status=TaskStatus.COMPLETED,
    steerer=steerer)` — the F1 payload (see [§ 6](#6-response-shapes-rendering)).

`report_task_started`, `report_task_failed`, and `report_task_blocked` are
structurally identical with different target statuses and (for started/failed)
the terminal-rotation step. `report_task_progress` skips validation
(`required[]` is empty), skips terminal rotation, and treats a `RUNNING` task as
a legal liveness tick and everything else as `invalid`.

### 4.3 `report_task_started`'s extra step: correction GC in `finally`

`_handle_task_started` wraps the `mark_task_running` await in `try/finally` and
calls `_clear_correction_on_started(session, task)` in the `finally`. This clears
any queued correction scoped to this `(agent, task_id)` — once the agent has
*acknowledged* the (possibly corrected) task, further turns should see the
unadorned instruction. The `finally` is load-bearing: a `CancelledError` (a
`BaseException`) mid-await would otherwise leave the pending correction on
session state and re-inject the correction block on a task the agent already
acknowledged (Phase 3.5, CANCELLATION-CONTRACT.md §C5). The whole thing runs
inside `_state_audit.cancellation_stash_audited(...)` and marks
`_state_audit.mark_stash_completed()`.

`report_task_failed` deliberately does **not** clear the correction — failure is
not acknowledgment, and a re-invocation still needs the correction.

### 4.4 The two drift handlers

`_handle_new_work_discovered` and `_handle_plan_divergence` validate required
fields, then route straight into the steerer's drift component
(`steerer.drift.report_new_work_discovered` / `report_plan_divergence`) and
return a bare `dict(_ACK)`. No task-scoped pipeline, no pin classification, no
`plan_state`. `report_new_work_discovered` is a lifecycle tool (default-on);
`report_plan_divergence` is a drift-opt-in tool.

### 4.5 The two declarations (observability-only)

`declare_task_skipped` / `declare_task_not_needed` share `_handle_declaration`.
They are **observability-only** — they emit `TaskDeclarationReceived` on the
sink bus and return an ack, but **do not mutate plan state**. The steerer's
`_apply_revision` machinery remains the only path that can transition a task
(adaptive-over-predictive, CANON invariant 4). A declaration is an advisory
signal a future refine may consume (Phase-4 work).

Idempotency is via `_record_declaration(session, kind, task_id, reason)`, keyed
by `f"{kind}:{task_id}"` on `session.state[DECLARATIONS_KEY]`
(`"goldfive.task_declarations"`). A second declaration of the same
`(kind, task_id)` pair is a no-op (returns `False`, no event re-emitted) and the
handler returns `{"acknowledged": True, "idempotent": True}`. The recorded body
keeps the **first** reason — late declarations don't rewrite history. The key
components are stable plan-side ids (CANON invariant 6); never re-key this on an
LLM-authored value.

---

## 5. Pin resolution and freshness: the #266 classifier

Task-scoped handlers must decide *which* task a call actually drives. Two
concerns stack:

1. **Which id?** `_resolve_task_id_with_source` (in `_internal.py`): explicit
   `args["task_id"]` wins (`source="llm_report"`); else the adapter pin
   `StateStore.for_session(session).pin_current_task()`
   (`source="handler_default"`); else `("", "")` → `missing_task_id`. The read
   funnels through `StateStore` so handlers are decoupled from raw `Session.state`
   key strings (goldfive#271 Phase 2.1).

2. **Is the pin *fresh*?** `_classify_and_route_pin` (in `handlers.py`) applies
   the goldfive#266 pin-freshness classifier. It reads the pin's stamped
   revision (`_read_pin_revision`) and the live plan revision
   (`_read_plan_revision`) and classifies via `_classify_pin_freshness`:

| Freshness | Meaning | Handler action |
| --- | --- | --- |
| `"match"` | `pin_revision >= current_revision` | Proceed. Still runs `_reroute_if_superseded` so a fresh pin on a terminal task with a REPLACE successor routes. |
| `"stale_replace"` | pin older + REPLACE-kind (or legacy UNSPECIFIED) successor | Route onto the successor id; caller sets `source="supersedes_reroute"`. |
| `"stale_correct"` | pin older + CORRECT-kind successor | **Refuse.** Old task's terminal state is historical fact; correction is a separate work unit. |
| `"stale_ambiguous"` | pin older + no successor | **Refuse.** Operator must disambiguate. |

On refuse, the handler emits a `task_transition_refused` sink event
(`_emit_task_transition_refused`, a typed proto envelope) and returns
`_refused_response()` — an ack-only `{"acknowledged": True}`. The LLM never sees
a structured error (see [§ 6.4](#64-the-refused-shape-why-ack-only)).

When there is no pin-revision stamp at all (`_read_pin_revision` returns `None`
— legacy session, custom adapter, pre-#266 test stub) the classifier is bypassed
and the handler falls back to the historical `_reroute_if_superseded` semantics.
This preserves backward compatibility; the stamp is not load-bearing for
correctness, only for precision.

`_resolve_effective_task_id` (the supersession walker) collapses A→B→C chains,
caps the walk at depth 8, and — critically — **skips CORRECT-kind links** (a
CORRECT supersedes retains the old COMPLETED node for DAG history; a late report
on it is an idempotent no-op, not a reroute). Only REPLACE / legacy-UNSPECIFIED
links reroute.

---

## 6. Response shapes (rendering)

Everything the model reads back lives in `goldfive/reporting/rendering.py`. There
are four families of response, each a *distinct* shape so loop-detector and
observability layers can tell them apart.

### 6.1 Directive ack (`_directive_ack`) + `plan_state`

A successful transition returns more than `{"acknowledged": True}`. It embeds
the live `plan_state` so a coordinator that just completed a task sees the *next
pending hand-off* instead of an information-free ack and looping back onto the
just-finished work. This is the F1 / Tier-1 loop-prevention pattern.

```python
# goldfive/reporting/rendering.py (_directive_ack)
response: dict[str, Any] = {
    "acknowledged": True,
    "task": {"id": task_id, "status": new_status.value},
}
if steering_is_active(steerer):
    response["plan_state"] = _build_plan_state(getattr(session, "plan", None))
return response
```

`_build_plan_state(plan)` returns:

```
{
  "completed_task_ids": [...sorted ids of COMPLETED tasks...],
  "next_pending": {
     "id": ..., "title": ..., "assigned_to": <bare agent name>,
     "predecessors_completed": True,
  } | None,
}
```

`next_pending` is chosen by `_next_pending_with_completed_predecessors`: the
first PENDING task whose every incoming-edge predecessor is terminal (a task
with no incoming edges trivially passes). `assigned_to` is run through
`_bare_agent_name` (last dot-separated segment) so the model sees a name it can
pass back as the AgentTool target — display-only; correction keys still use the
verbatim assignee id.

### 6.2 The `observation_only` gate on `plan_state`

`plan_state` is a **goldfive-authored directive** — it tells the model what to do
next — so it is a steering surface and rides the same
`observation_only` kill-switch as the prompt-shaping sites. It is only added
when `steering_is_active(steerer)` is truthy (CANON invariant 5):

- The **only** sanctioned read of the kill-switch is `steering_is_active(steerer)`
  (module helper) / `DefaultSteerer.is_active_steering()`.
  Missing / `None` / raising → **passive** (no `plan_state`).
- Under strict-passive the ack keeps only the *factual echo* of the transition
  the agent itself reported (`acknowledged` + `task`). goldfive observes; it does
  not nudge the model toward the next hand-off.

Both `_directive_ack` and `_idempotent_response` apply this gate identically. See
`09-steering-ladder-and-gates.md` for the kill-switch contract and the full list
of gated surfaces.

> **Weak-model trap.** Do not read `observation_only` off the config object, off
> `session`, or via any other attribute here. There is exactly one predicate.
> Adding a second read path is a CANON-invariant-5 violation and will diverge
> from every other gate.

### 6.3 Idempotent / invalid shapes

`_idempotent_response(current_status, session=..., task_id=..., steerer=...)` —
returned when the call is a no-op (task already in the target status). Shape:
`{"acknowledged": True, "idempotent": True, "current_status": ...}` plus a
`task` echo and (gated) `plan_state`. This is the goldfive#201 fix: retries from
a confused model no longer masquerade as tool-loop spam. Without a `session` the
helper degrades to the pre-F1 shape (legacy callers / stubs).

`_invalid_transition_response(...)` — returned when the current status cannot
legally transition under this tool (e.g. `report_task_started` on a COMPLETED
task). Shape: `{"acknowledged": False, "error": "invalid_transition", ...,
"current_status": ..., "attempted": ..., "message": "... do not retry."}`. This
is a real "agent is confused about state" signal, distinct from a benign retry —
loop-detector owners can surface it directly.

The `"idempotent"` vs `"invalid"` vs `"transition"` decision is made by
`_classify_transition` in `_internal.py`, using the `_TOOL_TARGET_STATUS` and
`_TOOL_VALID_SOURCES` tables (exact-equality status matching — CANON invariant 2
allows this; it is structured, not NL, classification).

### 6.4 The refused shape (why ack-only)

`_refused_response()` returns `dict(_ACK)` — a plain `{"acknowledged": True}`.
Even though the framework *refused* the stale-pin transition, the LLM sees a
success ack, **not** an error payload:

```python
# goldfive/reporting/rendering.py (_refused_response)
# The LLM still sees ``{"acknowledged": True}`` rather than an error
# payload — surfacing the refusal as a structured error would create a
# prompt-injection surface (the LLM might reason against the rejection
# and bypass the contract). Operators see the refusal via the
# ``task_transition_refused`` sink event.
```

This is deliberate and subtle: a structured refusal error would give the model
something to argue against and route around. Operators still get the full audit
trail via the `task_transition_refused` proto event. Do not "improve" this by
returning an error dict.

### 6.5 Missing-arg rejections

`_missing_task_id_response(tool_name)` and `_missing_required_field_response(...)`
are the pre-dispatch rejection bodies. Both are `{"acknowledged": False, "error":
..., "tool": ..., "message": ...}`; the missing-field variant adds `field`,
`reason`, `expected` (the property block), and `required` so the model can
self-correct next turn. `_missing_task_id_response` mirrors the shape the ADK
`invoke_tool` dispatcher returns so direct handler callers surface the same
structured error.

---

## 7. Approval: `report_awaiting_approval` end-to-end (post-#478)

`report_awaiting_approval` is the task-level half of the human-in-the-loop flow
(`docs/design/APPROVAL.md` Flow A). The agent blocks a task pending a human
yes/no; the control dispatcher lands `APPROVE` / `REJECT`.

The handler `_handle_awaiting_approval` returns one of **four** decisions:
`"approve"`, `"reject"`, `"timeout"`, `"unavailable"`. The `#478` program made
this path *never hang* — historically it awaited a waiter that could never be
set (no channel) or waited forever (no finite timeout). Read the handler flow in
order.

### 7.1 Validation and terminal guard

1. `_validate_required(args, _SCHEMA_AWAITING_APPROVAL, ...)` — `prompt`
   required.
2. `task_id, source = _resolve_task_id_with_source(args, session)`;
   `prompt = _str(...)`; `timeout_ms = _int(args, "timeout_ms", 0)`.
3. `if not task_id: return _missing_task_id_response(...)`.
4. **Terminal guard (goldfive#201):** if the task is already terminal, return
   `_invalid_transition_response`. An already-BLOCKED / already-RUNNING task
   falls through to the waiter-reuse path (that is the semantic idempotency for
   approvals — block on the existing Event, not a no-op ack).

### 7.2 No control channel → immediate `"unavailable"` (the #478 no-hang fix)

```python
# goldfive/reporting/handlers.py (_handle_awaiting_approval)
if getattr(steerer, "_control_channel", _UNKNOWN_CHANNEL) is None:
    log.warning(...)
    return {
        "acknowledged": True,
        "decision": "unavailable",
        "detail": (
            "no approval controller is attached to this run; the "
            "request cannot be received or answered by a human"
        ),
    }
```

The default `goldfive.wrap()` posture binds no control channel
(`DefaultSteerer._control_channel is None`). Then **no `APPROVE`/`REJECT` can
ever arrive**, so waiting any amount would wedge the tool call. The handler
returns immediately **without** blocking the task, registering a waiter, or
emitting `ApprovalRequested` (there is no controller to render it to). The agent
learns approval is unavailable and decides for itself.

The `_UNKNOWN_CHANNEL` sentinel distinguishes three cases:

- `steerer._control_channel is None` → **known** no channel → `"unavailable"`.
- `steerer` has no `_control_channel` attr (`getattr` returns `_UNKNOWN_CHANNEL`)
  → custom/stub steerer → *fall through* to the finite-default wait (a controller
  *might* exist).
- `steerer._control_channel` is a real channel → wait for a decision.

### 7.3 Register waiter, block the task, emit `ApprovalRequested`

If a channel is (or might be) bound:

1. Reuse or create the per-task `asyncio.Event` on `session.pending_approvals[task_id]`
   (idempotent — reuse an existing waiter).
2. `session.pending_approvals_meta.setdefault(task_id, {"kind": "task", "prompt":
   ..., "task_id": ...})`; refresh `meta["prompt"]`.
3. `await steerer.tasks.mark_task_blocked(task_id, blocker="awaiting_approval",
   needed=prompt, source=source)` — the task gets a concrete BLOCKED status so
   UIs have something to render *before* the await.
4. `_emit_approval_requested(...)` → `ApprovalRequested` on the sink bus.

### 7.4 Finite default timeout (the other half of #478)

```python
# goldfive/reporting/handlers.py (_handle_awaiting_approval)
effective_timeout_ms = timeout_ms
if effective_timeout_ms <= 0:
    config = getattr(steerer, "_steering_config", None)
    effective_timeout_ms = int(getattr(config, "approval_default_timeout_ms", 0) or 0)
    if effective_timeout_ms <= 0:
        effective_timeout_ms = DEFAULT_APPROVAL_TIMEOUT_MS
try:
    await asyncio.wait_for(waiter.wait(), timeout=effective_timeout_ms / 1000.0)
except TimeoutError:
    await _emit_approval_timeout_drift(...)
    return {"acknowledged": True, "decision": "timeout",
            "detail": f"no decision after {effective_timeout_ms}ms"}
```

`timeout_ms <= 0` (including the omitted default) no longer means "wait forever."
No invocation wall clock covers tool waits, so infinity was a run-hang. The
substitution order is:

1. Explicit positive `timeout_ms` from the agent — wins verbatim.
2. Else `SteeringConfig.approval_default_timeout_ms` (default
   `DEFAULT_APPROVAL_TIMEOUT_MS = 600_000` ms = **600 s**; env
   `GOLDFIVE_STEER_APPROVAL_DEFAULT_TIMEOUT_MS`).
3. Else the module constant `DEFAULT_APPROVAL_TIMEOUT_MS` (600 s).

Both `DEFAULT_APPROVAL_TIMEOUT_MS` and `SteeringConfig.approval_default_timeout_ms`
live in `goldfive/config.py`.

### 7.5 Timeout → `HUMAN_INTERVENTION_REQUIRED` drift

On timeout the handler returns `decision="timeout"`, **leaves the task BLOCKED**
(a later `APPROVE` / `REJECT` still resolves via `session.pending_approvals`),
and calls `_emit_approval_timeout_drift`, which emits:

```python
# goldfive/reporting/handlers.py (_emit_approval_timeout_drift)
DriftEvent(
    kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
    severity=DriftSeverity.WARNING,
    detail=(f"approval request for task {task_id!r} received no "
            f"APPROVE/REJECT within {timeout_ms}ms; task remains "
            "BLOCKED pending a decision"),
    current_task_id=task_id,
    authored_by="goldfive",
)
```

It is **emit-only** — it fires `steerer.drift._emit_drift_detected` directly and
does **not** run the intervention ladder (mirrors the Runner's revision-rejection
observability drift). Best-effort: steerers without a `drift` component skip
silently. The point is that operators see an *unresolved approval* rather than a
silently-BLOCKED task.

### 7.6 Decision arrives → `_resolve_approval`

The control-channel side is `_resolve_approval` in
`goldfive/executors/_control.py`, reached from the control dispatcher when a
`ControlKind.APPROVE` / `REJECT` message with `payload.target_id` lands:

```python
# goldfive/executors/_control.py (_resolve_approval)
waiter = session.pending_approvals.get(target_id)
if waiter is None:
    return False          # UI gets a FAILURE ack — the click didn't land
meta = session.pending_approvals_meta.setdefault(target_id, {})
meta["decision"] = decision
meta["detail"] = detail
# ... emit ApprovalGranted / ApprovalRejected BEFORE setting the event ...
waiter.set()
return True
```

Ordering matters: the resolution event is emitted *before* `waiter.set()` so the
stream order is "resolution event visible → waiter releases → tool-call returns."
Back in the handler, `waiter.wait()` unblocks and the handler reads
`meta["decision"]` / `meta["detail"]` and returns
`{"acknowledged": True, "decision": decision, "detail": detail}` (default
`"approve"` if unset).

goldfive does **not** force a task transition on the decision. On `approve` the
agent typically calls `report_task_completed`; on `reject`,
`report_task_failed` with a user-rejection reason. `ApprovalGranted` /
`ApprovalRejected` are *resolution* events on a separate channel, not task
transitions — the agent decides.

### 7.7 `plan_state` stripped from approval acks under observation_only (#478)

The approval acks (`unavailable` / `timeout` / decision) carry no `plan_state`
at all — they are already stripped of the goldfive-authored directive under
observation_only. This is part of the same #478 program: an ack under
strict-passive carries only what the agent needs to decide, never a goldfive
steering opinion.

### 7.8 Flow B — ADK tool confirmation (a *different* approval path)

Do not confuse Flow A with **Flow B** (`docs/design/APPROVAL.md`). Flow B is
ADK-specific: when a `FunctionTool` is built with `require_confirmation=True`,
the plugin's `before_tool_callback` (`goldfive/adapters/_adk_plugin.py`)
intercepts, registers a waiter keyed by the ADK `function_call_id` (kind
`"tool"`, not `"task"`), emits `ApprovalRequested`, awaits, and on reject returns
`{"skipped": True, "reason": "user_rejected"}` (ADK skips the tool) or on approve
returns `None` (ADK runs the tool). It reuses the *same* `session.pending_approvals`
map and the same `APPROVE`/`REJECT` control dispatch, but the target id is a
tool-call id, not a task id. When you edit the approval control path, remember it
serves both flows.

---

## 8. The no-cooperation tension: these tools are optional

This is the single most important design fact and the easiest to violate. CANON
invariant 1: **termination / control / observability must work even if the agent
never calls a goldfive tool.** Reporting tools are an *accelerator*, not a
dependency. Concretely, here is what still works with **zero** reporting calls:

| Concern | Reporting-tool path (fast) | Degradation path (always works) |
| --- | --- | --- |
| Task completion | `report_task_completed` → `mark_task_completed` | `PlanReconciler.on_after_agent` observes the agent's before/after pair and drives `steerer.transition(..., COMPLETED)` (or FAILED on error). See `goldfive/reconciler.py`. |
| Task failure | `report_task_failed` | Same reconciler path transitions to FAILED when the observed agent raised. |
| Missed tasks | agent self-reports | `PlanReconciler.get_missed_tasks(plan)` surfaces tasks never observed running. (Protected keep — CANON: `reconciler.get_missed_tasks` per #163.) |
| Run termination | — | Generator-end termination: the run ends when the ADK generator completes; goldfive does not wait for a terminal `report_*` call. Drift detectors + the (flag-gated) stall watchdog (`stall_watchdog_enabled`, #487) also drive termination. |
| Drift | `report_plan_divergence` / `declare_*` | Deterministic detectors + the LLM goal-drift judge over thinking tokens. See `07-deterministic-drift-detection.md`, `08-llm-judges.md`. |
| Approval unavailable | `report_awaiting_approval` returns `"unavailable"` | The agent proceeds; goldfive forces nothing. |

The design consequence: **no handler may become the only path to a correctness
guarantee.** If you find yourself writing "the run can't finish unless the agent
calls `report_task_completed`," you have violated the invariant — the reconciler
must still close the task. Test both paths (there are reconciler tests that run
*without* any reporting tools registered).

This is also why the drift tools are opt-in and default-off: the observation
pipeline already covers them, so the tools are pure prompt-cost + hallucination
surface for no new capability (goldfive#196).

---

## 9. Tool-surface cost honesty

For the weak-model reader deciding whether to *add* a tool: every reporting tool
you register has a real, measurable cost paid on every turn of every reachable
agent.

- **Token overhead:** ~200–400 tokens per tool (schema + description), multiplied
  by every reachable agent in the tree (the augmentation lands the tool on all of
  them). A 5-agent tree with 3 extra tools is ~3–6k tokens of prompt bloat per
  turn.
- **Hallucination surface:** each tool is something the model can call *wrongly*.
  A confused model can confabulate `report_plan_divergence` and trigger a
  spurious replan; can call `report_task_completed` on the wrong task; can
  abandon the protocol entirely when an optional arg confuses it (the exact
  reason `task_id` is hidden — [§ 2.1](#21-task_id-is-absent-from-every-required-list--on-purpose)).
- **Weak-model confusion:** more tools = more surface for a weaker model to pick
  the wrong one. The maintainer bias is *fewer tools, default-off for anything
  observation already covers.*

The maintainer rationale for the 7/3 split ([§ 1.2](#12-the-73-split-rationale-goldfive196))
is exactly this cost accounting: keep the seven lifecycle tools (no observation
analog cheap enough / precise enough to replace them) default-on; gate the three
drift opinions off because observation already produces the same signal.

---

## 10. Common mistakes

Each row is a wrong edit a weaker model would plausibly make, with the correct
alternative.

### 10.1 Adding a tool without pin resolution

**Wrong:** add `report_task_paused`, read `task_id = args["task_id"]` directly,
and drive the steerer.

**Why it breaks:** the ADK adapter *hides* `task_id` from the tool declaration —
the model can't supply it. Your handler will always see `task_id` absent and
`missing_task_id`-reject every real call. You also skipped the #266 freshness
gate, so a stale pin drives a superseded task.

**Right:** resolve via `_resolve_task_id_with_source(args, session)`, short-circuit
with `_missing_task_id_response` if empty, run `_await_plan_stable`, then
`_classify_and_route_pin(...)` before touching the steerer. Copy the shape of
`_handle_task_blocked` verbatim and change the target status. Add the tool name +
declared signature to `_reporting_tool_signatures()` in `adk.py` if it takes a
`task_id` you want hidden.

### 10.2 Making core logic depend on a reporting call

**Wrong:** "the run ends when the coordinator calls `report_task_completed` on the
root task." Or: gate cleanup / termination / a correctness check on a handler
firing.

**Why it breaks:** CANON invariant 1. An agent that never calls the tool (a bare
ADK tree with no cooperation) would then never terminate / never clean up.

**Right:** the reconciler + generator-end termination are the source of truth
([§ 8](#8-the-no-cooperation-tension-these-tools-are-optional)). The handler is a
*faster* path to the same state, never the only path. If you need a new
guarantee, put it in the observation pipeline (reconciler, detectors, watchdog),
not in a handler.

### 10.3 Response shapes that instruct rather than inform under observation_only

**Wrong:** always attach `plan_state` / "next, delegate to research_agent" / any
"do X next" hint to the ack, unconditionally.

**Why it breaks:** CANON invariant 5. `plan_state` is a goldfive-authored
directive — a steering surface. Under strict-passive (`observation_only=True`,
the production default) goldfive must not nudge the model. An unconditional hint
turns a passive observer into a controller-in-disguise.

**Right:** gate every directive field on `steering_is_active(steerer)` exactly as
`_directive_ack` / `_idempotent_response` do. Under passive, return only the
factual echo (`acknowledged` + `task` + factual error fields). Never read the
kill-switch any other way.

### 10.4 Returning a structured error on a stale-pin refusal

**Wrong:** make `_refused_response()` return `{"acknowledged": False, "error":
"stale_pin", ...}` so the model "knows" it was refused.

**Why it breaks:** it creates a prompt-injection surface — the model reasons
against the rejection and routes around the contract. See
[§ 6.4](#64-the-refused-shape-why-ack-only).

**Right:** keep the ack-only `dict(_ACK)`. Operators learn about the refusal from
the `task_transition_refused` sink event, not the model.

### 10.5 Editing the shim body to "make the tool do something"

**Wrong:** put logic in `_build_ack_shim`'s inner `_shim(**kwargs)`.

**Why it breaks:** the shim is never executed — ADK's `before_tool_callback`
short-circuits it with the handler's dict ([§ 3.3](#33-the-no-op-shim-the-handler-runs-in-the-plugin-not-the-tool-body)).

**Right:** edit the `_handle_*` function in `handlers.py`. If you need new dispatch
behaviour, edit the plugin's `before_tool_callback` in `_adk_plugin.py`.

### 10.6 Making a declaration mutate the plan

**Wrong:** have `declare_task_not_needed` transition the task to `NOT_NEEDED`
directly "since the agent said so."

**Why it breaks:** CANON invariant 4 (adaptive over predictive). Declarations are
observability-only by design — the agent's *stated intent*, not an authoritative
transition. The steerer's `_apply_revision` machinery is the only path that may
transition a task. Auto-mutating on a self-report lets a confused agent delete
plan tasks.

**Right:** emit `TaskDeclarationReceived` and let the next refine *consider* the
declaration. Keep `_handle_declaration` mutation-free.

### 10.7 Re-keying the declaration gate on an LLM-authored value

**Wrong:** key declarations on `reason` text, or on an LLM-minted call id, "to be
more specific."

**Why it breaks:** CANON invariant 6. Churning keys open a fresh entry per
observation, so the idempotency gate never engages and every repeat re-emits.

**Right:** keep `f"{kind}:{task_id}"` — both are stable plan-side ids.

### 10.8 Hand-enumerating `LIFECYCLE_REPORTING_TOOLS`

**Wrong:** add your new lifecycle tool to a hand-written list of lifecycle names.

**Why it breaks:** `LIFECYCLE_REPORTING_TOOLS` is *derived* (`BUILTIN_REPORTING_TOOLS`
minus `DRIFT_SELF_REPORTING_TOOL_NAMES`). A parallel hand-list drifts out of sync.

**Right:** add the spec to `BUILTIN_REPORTING_TOOLS`. If it is a drift opinion,
also add its name to `DRIFT_SELF_REPORTING_TOOL_NAMES`; otherwise it is
automatically lifecycle/default-on.

### 10.9 Assuming `report_awaiting_approval` blocks forever

**Wrong:** "the approval tool blocks until a human answers, so I can await it
indefinitely in a test."

**Why it breaks:** post-#478 the wait is always finite (600 s default) and
returns `"timeout"` / `"unavailable"` on the no-answer / no-channel paths. A test
that expects an unbounded block will get a `"timeout"` decision.

**Right:** in tests, either bind a control channel and dispatch `APPROVE`/`REJECT`,
or assert the `"unavailable"` / `"timeout"` degraded decisions. See
`tests/test_approval_flow.py`.

### 10.10 Adding a fourth pin-freshness routing outcome that mutates history

**Wrong:** make a `stale_correct` pin *route* onto the correction task "so the
agent's report lands somewhere."

**Why it breaks:** a CORRECT supersedes retains the old COMPLETED node as
historical fact; rerouting a report onto the correction either destroys fact or
shadows the correction. That is exactly why `stale_correct` *refuses*.

**Right:** leave CORRECT-kind links refusing. Only REPLACE / legacy-UNSPECIFIED
links route.

---

## 11. Verification checklist

Run these after touching anything in this chapter's file set. Commands assume the
repo root and the dev+adk extras (`uv sync --extra dev --extra adk`).

### 11.1 Targeted test files

```bash
uv run pytest -q \
  tests/test_reporting.py \
  tests/test_reporting_idempotency.py \
  tests/test_reporting_declarations.py \
  tests/test_reporting_tool_required_validation.py \
  tests/test_drift_reporting_optin.py \
  tests/test_approval_flow.py \
  tests/test_plan_reconciler.py
```

- `test_reporting.py` / `test_reporting_idempotency.py` — handler pipeline,
  directive acks, idempotent/invalid shapes.
- `test_reporting_declarations.py` — the two `declare_task_*` tools + dedup.
- `test_reporting_tool_required_validation.py` — `_validate_required` rejections.
- `test_drift_reporting_optin.py` — `select_reporting_tools` / the 7/3 split.
- `test_approval_flow.py` — Flow A end-to-end incl. `unavailable` / `timeout` /
  decision paths.
- `test_plan_reconciler.py` — the no-cooperation degradation path (tasks close
  with **no** reporting call).

### 11.2 Full suite + lint

```bash
uv run pytest -q          # ~30s, expect ~2912 passed / 61 skipped
ruff check .              # must stay clean
```

Do **not** run a formatter over the repo — it is intentionally not
ruff-format-clean (mass reformat = unreviewable diff).

### 11.3 Grep checks for invariant compliance

After editing a handler or a response shape:

```bash
# (a) Every directive/plan_state emission is gated on the one predicate.
grep -n "plan_state" goldfive/reporting/rendering.py
grep -n "steering_is_active" goldfive/reporting/rendering.py
#   Expect: every `response["plan_state"] = ...` is inside an
#   `if steering_is_active(steerer):` block. No other observation_only read.

# (b) No handler reads the kill-switch any other way.
grep -rn "observation_only" goldfive/reporting/
#   Expect: NO hits (the gate lives behind steering_is_active in rendering.py).

# (c) The tool inventory is consistent across the three lists.
grep -n "REPORTING_TOOL_NAMES\|DRIFT_SELF_REPORTING_TOOL_NAMES" goldfive/reporting/handlers.py

# (d) A new task-scoped handler resolves task_id via the pin helper,
#     not by reading args["task_id"] directly.
grep -n "_resolve_task_id_with_source\|args\[.task_id.\]" goldfive/reporting/handlers.py
#   Expect: task-scoped handlers use _resolve_task_id_with_source; a bare
#   args["task_id"] read in a handler is a red flag.

# (e) If you added an ADK-hidden-task_id tool, it is in the signature map.
grep -n "_reporting_tool_signatures" goldfive/adapters/adk.py
```

### 11.4 Adapter coverage sanity

If you touched `register_reporting_tools` or `_augment_subtree_with_reporting`,
run the ADK adapter tests and confirm the integrity `RuntimeError` still fires on
a partial-augmentation fixture:

```bash
uv run pytest -q tests/ -k "adk and (reporting or augment or reachable)"
```

Expect: a coordinator + AgentTool fixture registers the tools on *every* reachable
agent, and a deliberately-broken augmentation raises the "did not land on N
reachable agent(s)" `RuntimeError`.

---

## 12. Cross-references

- `03-runner-and-conversation.md` — `Runner.run` step 5, `_abort_turn`.
- `05-adk-plugin.md` — `before_tool_callback` dispatch, `task_id` injection,
  Flow-B tool-confirmation bridge.
- `04-executors-and-control.md` — the control channel, `drain_controls` /
  `dispatch_control`, `_resolve_approval`.
- `07-deterministic-drift-detection.md` / `08-llm-judges.md` — the observation
  detectors that make the drift tools redundant (hence opt-in).
- `09-steering-ladder-and-gates.md` — the `observation_only` kill-switch and the
  full list of gated goldfive-authored surfaces.
- `10-planning-and-revision.md` — refine / `_apply_revision`, which declarations
  advise but never bypass.
- `11-state-ownership.md` — `StateStore`, the `current_task_id` pin, revision
  stamping, `rotate_current_task_id`.
- `12-events-sinks-telemetry.md` — `ApprovalRequested` / `Granted` / `Rejected`,
  `TaskTransitionRefused`, `TaskDeclarationReceived` event shapes.
- `14-config-reference.md` — `approval_default_timeout_ms`,
  `DEFAULT_APPROVAL_TIMEOUT_MS`, `drift_self_reporting`.
- `17-invariants-hazards-history.md` — the no-cooperation invariant, protected
  keep-decisions.
