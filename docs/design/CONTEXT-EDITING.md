# Context Editing as a Steering Capability

## Status

**Phase 1 shipped** (goldfive#397) — `ContextEditor` skeleton with all five invariants enforced, `PruneCancelledReasoningRule` registered, gated behind `SteeringConfig.context_editor_rules`.

**PR 6b shipped** (AGENCY-PRESERVATION.md Stage 2) — the three follow-up rules land: `PruneTransientErrorRule`, `PruneStaleSteerRule`, `CompactPriorReasoningRule`. The compaction rule introduces the **byte-monotonic-replace rule class** (the relaxation of Phase 1's drop-only invariant — see "Rule classes" below). Every rule is dormant on healthy turns: it edits `contents` ONLY on a tripped guardrail counter or drift verdict (AGENCY-PRESERVATION.md §0). All four rules opt in by name via `SteeringConfig.context_editor_rules`; default is still no rules wired.

## Implementation deltas from the original proposal

The proposal landed as written with four pragmatic adjustments:

* **Single-method rule protocol.** The proposed `applies(...) + edit(...)` shape was collapsed into a single `edit(contents, ctx) -> list[Content] | None` — `None` means "no change at this revision; skip." Equivalent surface, one fewer call to misuse.
* **Drop-only invariant made explicit, then scoped by rule class (PR 6b).** "No injection" shipped (Phase 1) as a structural byte- and content-count monotonicity gate: a rule's output `contents` MUST have ≤ the input's byte total AND ≤ the input's content count. Violations revert and emit `ContextEditRejected` with `reason='not_drop_only'`. PR 6b keeps that gate for *every* rule and layers a per-rule `rule_class` on top (see "Rule classes" below): `drop_only` rules may only remove whole entries (structurally enforced — `reason='injected_content'` on violation), while `byte_monotonic_replace` rules may redact/summarize in place under the same byte/count gate.
* **State-audit extension is documentation-only in Phase 1.** `goldfive/_state_audit.py` today guards ADK `session.state` writes through a single funnel (`_adk_state_protocol._set`). The `llm_request.contents` list has no analogous funnel — a runtime tripwire would require wrapping the list in a tracked proxy with measurable hot-path overhead. Phase 1 catalogs `ContextEditor.apply` as the sole authorised site via the new `_REQUEST_CONTENTS_AUTHORISED_SITES` constant in `_state_audit.py`; a runtime extension is deferred to a focused follow-up if a regression surfaces.
* **Events use dict envelopes.** `ContextEdited` and `ContextEditRejected` ship via `goldfive.events.make_event` (no proto regen) — mirrors the `pin_resolved` precedent in `_adk_plugin.py::_emit_pin_resolved`. Adding proto slots is left for a focused proto-schema PR.

## Motivation

Today goldfive's intervention ladder treats the LLM transcript as immutable. All steering routes through:

| Mechanism | Editorial direction | Surface |
|---|---|---|
| Plan revision | Plan / Task state only | `steerer._emit_plan_revised`, `set_session_plan` |
| Synthetic `USER_STEER` | **Add** new user event | `_install_revision` → executor injects |
| Cancel + reinvoke | **Terminate** in-flight only | `_safe_task_cancel`, `_heal_pending_tool_calls` |
| Prompt-shaping | **Add** to `system_instruction` | `PromptShaper` (post-Wave-B1) |
| `max_output_tokens` ratchet | Config cap | `_ratchet_max_tokens` |

Conspicuously absent: any way to *edit* the conversation the next model call sees. The LLM's view of history is whatever ADK has accumulated — `function_call` / `function_response` pairs, prior user messages, prior model responses — and goldfive has no mechanism to redact, prune, or rewrite any of it.

This leaves real steering capability on the table:

- **Drift-loop history pruning.** When the reasoning judge fires `LOOPING_REASONING` and the steerer cancels-and-retries, the failed reasoning trail is still in context for the next attempt. The model can re-anchor on the same failed approach. Relates to #173, #193, #210.
- **Transient tool-error redaction.** A 429 / parse failure / network blip that propagated through a `function_response` becomes a permanent fixture in the transcript. Subsequent turns waste tokens re-reading it and may bias their reasoning toward the failure.
- **Cancelled-invocation reasoning strip.** #230 silenced *judges* on cancelled reasoning. The orthogonal direction — strip the cancelled-invocation reasoning from the *model's own* context before the next call — is unimplemented.
- **Stale plan-revision summary cleanup.** When goldfive emits a revised plan, its own injected `system_instruction` prefix gets the strip-and-replace treatment (`_strip_prior_runtime_tools_hint`). The same idea applied to historical user messages that contained obsolete steering directives would close a long-tail leak.

## ADK surface (what's available)

From `google.adk.plugins.base_plugin.BasePlugin`:

```python
async def before_model_callback(
    self, *, callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]: ...

async def on_event_callback(
    self, *, invocation_context: InvocationContext, event: Event
) -> Optional[Event]: ...
```

Both receive *mutable* arguments. The plugin can:

- **`before_model_callback`** — mutate `llm_request.contents` (the message list sent to the model), edit `config.system_instruction`, edit `config.tools`, cap `config.max_output_tokens`, or return a synthesized `LlmResponse` to short-circuit the call.
- **`on_event_callback`** — return a replacement `Event`, or `None` to drop the event entirely.

ADK's base flow (`google/adk/flows/llm_flows/base_llm_flow.py`) reads `llm_request.contents` *after* the plugin callback chain, so any mutation lands in the request that hits the model.

## Goldfive's current usage (verified audit)

`goldfive/adapters/_adk_plugin.py`:

- **`before_model_callback`** (line 5232) — reads `contents` via `_measure_request_chars` (line 1180) for instrumentation. **Never writes to `contents`.** Writes only to `config.system_instruction` (additive injects, plus strip-and-replace of goldfive's own prior hint via `_strip_prior_runtime_tools_hint`, line 1919) and `config.max_output_tokens` (`_ratchet_max_tokens`, line 2052).
- **`after_model_callback`** (line 5925) — observation only.
- **`on_event_callback`** (line 6151) — detects `transfer_to_agent` / `escalate` actions, calls `steerer.observe`, **always returns `None`**. Never replaces an event.

Repository grep (verified) confirms no `llm_request.contents = ...`, `del llm_request.contents`, `contents.pop`, `contents.remove`, or list-slice assignment anywhere in the package. The transcript is read-only from goldfive's perspective.

The closest things to "editing context" goldfive does today:

- **`_heal_pending_tool_calls`** (`adapters/adk.py:2369`) — *appends* synthetic `function_response` events to `session.events` after a cancel to close orphan `tool_call_id`s. Necessary because mid-tool cancels leave half-pairs that ADK errors on. Adds; does not remove.
- **`_strip_prior_runtime_tools_hint`** (`adapters/_adk_plugin.py:1919`) — strips and rewrites goldfive's *own* injected prefix in `system_instruction` before re-injecting. The only redaction-shaped operation in the codebase, scoped to goldfive's own text.

## Proposed capability: `ContextEditor`

New module `goldfive/context_editor.py` exposing a `ContextEditor` class that runs in `before_model_callback`, **after** `PromptShaper` and **before** the LLM call dispatch. It applies a sequence of registered **editor rules** to `llm_request.contents`. Each rule has the signature:

```python
class ContextEditRule(Protocol):
    name: str
    def edit(self, contents: list[Content], ctx: ContextEditContext) -> list[Content] | None: ...
```

`ctx` carries the goldfive `Session`, the host agent name, and the snapshot `observed_revision_index` captured at the top of `ContextEditor.apply`. Rules returning `None` are skipped (no change at this revision); returning a new list triggers the invariant chain.

Rules are registered at adapter construction; the editor walks them in registration order, each receiving the output of the prior. The editor itself runs as a single `before_model_callback` site so all `contents` mutation is centralised (mirrors PromptShaper's "single gate, single module" pattern).

### Gates (mandatory)

1. **`observation_only` gate.** Strict-passive mode MUST skip the entire editor pipeline. Cataloged in `_state_audit.py`. Mirrors the discipline established by #271.
2. **`tool_call_id` pairing invariant.** After the editor runs, every `function_call` part must still have its matching `function_response` (or both absent). The editor invokes a `Plan.validate()`-style structural check on `contents` and **reverts** the edit on violation. `_heal_pending_tool_calls` shows this is non-negotiable — ADK errors hard on orphans.
3. **Byte/count monotonicity + rule-class scoping.** Editors may never grow the transcript: a rule's output `contents` MUST have ≤ the input's byte total AND ≤ its content count (revert + `reason='not_drop_only'`). On top of that gate, each rule's `rule_class` bounds the *shape* of edit it may make — see "Rule classes" below. Free-form additive shaping (injecting brand-new guidance) stays in PromptShaper's lane where it's auditable.
4. **Idempotence per-revision.** The same `(rule, observed_revision_index, contents-hash)` must produce the same output for the same input. Catches accidental nondeterminism.
5. **Logged-out-of-loop.** Edits are append-only against the *persisted* transcript — goldfive's sinks see the original event stream via `on_event_callback`; only the model's request is edited. Editor emits a `ContextEdited` event so harmonograf can show what the model didn't see.

### Rule classes (PR 6b)

Phase 1's "drop-only" framing was binary: a rule could only drop or truncate, never synthesize. PR 6b's `CompactPriorReasoningRule` needs to *summarize* a run of identical failed calls — i.e. emit a short goldfive-authored sentence in place of much longer removed material. That is still subtractive in aggregate, but it injects new text, which the strict drop-only contract forbade. The relaxation introduces a per-rule `rule_class` attribute; the structural byte/count monotonicity gate (Invariant 3) binds for **both** classes — the class only decides whether *in-place rewrite/synthesis* is permitted on top of it.

| `rule_class` | May do | May NOT do | Enforcement | Rules |
|---|---|---|---|---|
| `drop_only` (default) | remove whole `Content` entries | modify or synthesize any entry | every output entry must be **identity-present** in the input — else revert with `reason='injected_content'` | `PruneCancelledReasoningRule`, `PruneStaleSteerRule` |
| `byte_monotonic_replace` | redact a payload in place; replace a run of entries with one shorter summary | grow byte total or content count | the byte/count gate only (identity check skipped) | `PruneTransientErrorRule`, `CompactPriorReasoningRule` |

Discipline for `byte_monotonic_replace` rules: build NEW `Content`/part objects (`copy.deepcopy`) rather than mutating the live `contents` objects, so an invariant revert restores the original transcript byte-for-byte. The emitted `ContextEdited` event carries the `rule_class` so harmonograf can distinguish a pure subtraction from an in-place rewrite. An unrecognised `rule_class` value is coerced to the strictest class (`drop_only`) and logged — a typo never silently grants replace privileges.

### Dormancy (AGENCY-PRESERVATION.md §0)

Context editing is a steering surface, so it obeys the dormant-supervisor contract: a rule edits `contents` ONLY on a tripped guardrail counter or a drift verdict — never on a healthy turn. Each rule self-gates on a non-healthy trigger it can observe: cancelled function-call ids on session state (`PruneCancelledReasoningRule`), a transient-error response present in `contents` (`PruneTransientErrorRule`), a stale goldfive note present (`PruneStaleSteerRule`), or ≥N identical failed tool calls (`CompactPriorReasoningRule`). The editor additionally hands every rule `ContextEditContext.active_drift_kinds` — the set of conditions currently OPEN/ESCALATING on the session (`state_store.list_active_drifts`) — so a rule can arm on a recorded *verdict* too: `CompactPriorReasoningRule` lowers its repeat threshold from 3 to 2 when a `looping_tool_call` / `looping_reasoning` condition is already tripped. A healthy transcript (no cancels, no errors, no notes, no repeats, no active drifts) passes through every rule **byte-identical** — pinned by `test_healthy_transcript_is_byte_identical_with_all_rules`.

### Invariants to preserve

- **Plan / Task immutability** (#247) — editors don't touch state, only the model's view of history.
- **`observed_revision_index` semantics** — editor decisions stamped at this index; re-checked at next call.
- **ControlMessage channel** — editor activations emit a `ContextEdited` ControlMessage for observability (carries rule name, byte delta, content-count delta).
- **State-ownership audit** — `_state_audit.py` is extended to police request-side mutation. Editor cataloged as the authorised write site for `llm_request.contents`. Today the audit covers session-state writes only; this proposal *expands* its remit.

### Failure modes

| Failure | Mitigation |
|---|---|
| Rule strips half of a `function_call` / `function_response` pair | Pairing-invariant check reverts the edit; emit `ContextEditRejected` event. |
| Rule produces an empty `contents` list | Reject; ADK requires at least one user turn. |
| Two rules disagree across plugins | Editor runs after PromptShaper, before model — single ownership of `contents` editing. |
| Edit confuses the model into a *different* loop | Existing drift detectors observe next turn's reasoning; if `LOOPING_REASONING` re-fires, escalate to cancel-reinvoke (existing ladder). |
| Edit hides information the user expects in logs | Persisted transcript untouched; harmonograf shows the `ContextEdited` event with byte delta and rule name. |
| Rule is non-deterministic | Idempotence-per-revision invariant catches divergence. |

### Rule set (all shipped — opt in by name)

1. **`PruneCancelledReasoningRule`** (`prune_cancelled_reasoning`, `drop_only`) — strip `function_call` / `function_response` pairs from invocations that were cancelled via `_safe_task_cancel`. Companion to #230 (which silenced *judges* on cancelled reasoning); this silences the *model itself*. Trigger: `goldfive.cancelled_function_call_ids` non-empty.
2. **`PruneTransientErrorRule`** (`prune_transient_error`, `byte_monotonic_replace`) — **redact** (not drop) `function_response` payloads matching known transient errors (429/5xx/timeout/network blip/parse failure). Redacts in place so the `function_call`/`function_response` pair stays intact (dropping the response half alone would orphan the call). Configurable allowlist of status codes + markers. Conservative + structural: only a dict response with a recognised status code or an explicit error shape whose error text matches a transient marker. **The marker allowlist is restricted to machine-generated error signatures** — HTTP status reason phrases, SDK/runtime exception class names, structured error codes — emitted by infrastructure, never authored by the agent. The rule MUST NOT be extended into semantic matching of natural-language prose (agent reasoning or free-text tool output); that is the #166/#167 anti-pattern. The boundary is "machine error payload", never "looks like an error in English". Trigger: such a response present in `contents`.
3. **`PruneStaleSteerRule`** (`prune_stale_steer`, `drop_only`) — strip prior goldfive synthetic steer / observer-note user-messages once stale. Identified by goldfive's OWN stable constants — `observer_notes.OBSERVER_NOTE_MARKER_PREFIX` (`[GOLDFIVE OBSERVER NOTE`) + `observer_notes.ADVISORY_FOOTER`, both **single-sourced in `goldfive/observer_notes.py`** (the same module PR 6's channel renders them from, so the writer and this reader can never drift). Stale when no longer the currently-active steer (`goldfive.active_steer.body` cleared/superseded once the correction took) — a stable-keyed proxy for "the steered revision is COMPLETED", never an NL heuristic. Full coverage of legacy plain-text notes arrives once PR 6's channel wraps every delivered note in the marker. Trigger: a stale goldfive note present.
4. **`CompactPriorReasoningRule`** (`compact_prior_reasoning`, `byte_monotonic_replace`) — collapse N identical FAILED tool-call pairs into one summarized survivor (keep the first call+response, summarize its response, drop the rest). Only collapses a group when doing so strictly reduces bytes (so a run of tiny failures is left alone rather than reverted). Trigger: ≥`min_repeats` (default 3, or 2 when a `looping_*` verdict is tripped) identical failed calls.

All four are opt-in via `SteeringConfig.context_editor_rules`; none is on by default. Rules 1-3 are conservative subtractions; rule 4 synthesizes a bounded summary under the byte-monotonic-replace contract.

## Implementation plan

| PR | Scope | Status |
|---|---|---|
| 1 | `ContextEditor` skeleton + 5 invariants + `observation_only` gate + state-audit hook. No rules registered. Smoke test: editor is a no-op when rule set is empty. | ✅ goldfive#397 |
| 2 | `PruneCancelledReasoningRule`. Unit tests with synthetic cancel scenarios + e2e against the drift-loop reproducer. | ✅ goldfive#397 |
| 6b | `PruneTransientErrorRule` (redact, configurable allowlist) + `PruneStaleSteerRule` (stable-keyed staleness off active-steer state) + `CompactPriorReasoningRule` (byte-monotonic-replace rule class). Dormancy wiring + healthy-transcript byte-identity test. | ✅ AGENCY-PRESERVATION.md Stage 2 |

(The original plan split rules 3-5 across separate PRs; AGENCY-PRESERVATION.md folded the three follow-up rules into one PR 6b, since they share the rule-class relaxation and the dormancy wiring.)

Each rule lands behind its own opt-in name in `SteeringConfig.context_editor_rules: list[str]` so e2e regressions can be bisected per rule.

## Alternatives considered

- **Replay via fresh session.** Instead of editing in-place, cancel the session and replay with a curated `contents` list. Heavyweight: re-creates ADK session, re-pays tool-registration cost, breaks observability continuity. Editing is cheaper.
- **Move pruning into the planner prompt.** Have the planner summarise history at refine time and inject the summary. Doesn't help — the model that sees the planner's summary is the same model whose context we're trying to clean.
- **Re-emit `Event`s via `on_event_callback`.** ADK supports returning a replacement event there, but it's per-event and doesn't compose well across many events. `before_model_callback` operates on the assembled `contents` list — natural scope for multi-event pruning rules.
- **Do nothing; rely on drift detectors.** Today's stance. Works for many cases but leaves the failure modes listed in *Motivation* unaddressed — they show up as drift loops (#173, #193) and inflated LLM-call ratios (#210).

## Open questions

- Should the editor reflect its edits into the harmonograf event stream (so the UI can visualise "what the model saw vs. what happened")? Recommendation: **yes** — emit `ContextEdited` with byte-delta + rule name. Without this, debugging a "the model ignored X" report becomes impossible.
- ~~Should rules be allowed to *rewrite* (e.g., shorten) parts, or only drop?~~ **Resolved (PR 6b).** Phase 1 was drop-only; PR 6b adds the `byte_monotonic_replace` rule class (see "Rule classes") for `PruneTransientErrorRule` (redact) and `CompactPriorReasoningRule` (summarize). The "is the shortened version semantically equivalent?" risk is bounded structurally (byte/count monotonicity gate, `copy.deepcopy` so reverts are clean, `rule_class` stamped on the `ContextEdited` event) and operationally (each rule is opt-in, dormant on healthy turns, conservative in what it flags).
- Does this interact with `_heal_pending_tool_calls`? Yes — the editor runs *after* prior-turn healing has happened, so pairing invariants are already in place when editing begins. New documentation of the order in CANCELLATION-CONTRACT.md needed.
- Where does `ContextEditor` live in the new modular architecture (post Wave B1 / C)? Likely as a peer of `PromptShaper`: a goldfive-owned `before_model_callback` participant, ordered explicitly in the adapter glue.

## References

### Code

- `goldfive/adapters/_adk_plugin.py:5232` — `before_model_callback` (current implementation; never writes `contents`)
- `goldfive/adapters/_adk_plugin.py:6151` — `on_event_callback` (always returns `None`)
- `goldfive/adapters/_adk_plugin.py:1180` — `_measure_request_chars` (current read-only `contents` access)
- `goldfive/adapters/_adk_plugin.py:1919` — `_strip_prior_runtime_tools_hint` (existing strip-shaped operation, scoped to goldfive's own text)
- `goldfive/adapters/_adk_plugin.py:2052` — `_ratchet_max_tokens` (existing config edit)
- `goldfive/adapters/adk.py:2369` — `_heal_pending_tool_calls`
- `goldfive/_state_audit.py` — state-ownership audit (to be extended to request-side)

### Existing design docs

- `docs/design/STATE-OWNERSHIP-CONTRACT.md` — audit pattern this proposal extends
- `docs/design/CONTROL-CHANNEL.md` — ControlMessage pattern used for `ContextEdited` event
- `docs/design/EVENT-MODEL.md` — event-stream contracts
- `docs/design/CANCELLATION-CONTRACT.md` — `_heal_pending_tool_calls` ordering (needs amendment for editor)

### Related issues

- #173, #193, #210 — drift-loop / LLM-call ratio (motivates `PruneCancelledReasoningRule`)
- #230 — quiet judges on cancelled reasoning (orthogonal direction; this proposal silences the model itself)
- #271 — strict-passive `observation_only` (gate pattern this proposal mirrors)
