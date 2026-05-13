# Context Editing as a Steering Capability

## Status

**Proposal — not yet implemented.** Filed for tracking; ships behind a feature flag if accepted.

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
    def applies(self, request: LlmRequest, ctx: SteeringContext) -> bool: ...
    def edit(self, contents: list[Content]) -> list[Content]: ...
```

Rules are registered at adapter construction; the editor walks them in registration order, each receiving the output of the prior. The editor itself runs as a single `before_model_callback` site so all `contents` mutation is centralised (mirrors PromptShaper's "single gate, single module" pattern).

### Gates (mandatory)

1. **`observation_only` gate.** Strict-passive mode MUST skip the entire editor pipeline. Cataloged in `_state_audit.py`. Mirrors the discipline established by #271.
2. **`tool_call_id` pairing invariant.** After the editor runs, every `function_call` part must still have its matching `function_response` (or both absent). The editor invokes a `Plan.validate()`-style structural check on `contents` and **reverts** the edit on violation. `_heal_pending_tool_calls` shows this is non-negotiable — ADK errors hard on orphans.
3. **No content injection.** Editors can drop or rewrite existing messages; they cannot inject material that wasn't there. Injection stays in PromptShaper's lane where it's auditable.
4. **Idempotence per-revision.** The same `(rule, observed_revision_index, contents-hash)` must produce the same output for the same input. Catches accidental nondeterminism.
5. **Logged-out-of-loop.** Edits are append-only against the *persisted* transcript — goldfive's sinks see the original event stream via `on_event_callback`; only the model's request is edited. Editor emits a `ContextEdited` event so harmonograf can show what the model didn't see.

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

### Initial rule set

1. **`PruneCancelledReasoningRule`** — strip `function_call` / `function_response` pairs from invocations that were cancelled via `_safe_task_cancel`. Companion to #230 (which silenced *judges* on cancelled reasoning); this silences the *model itself*. Highest-value rule; recommend ship first.
2. **`PruneTransientErrorRule`** — strip `function_response` parts whose payload matches known transient errors (429, timeout, network blip). Configurable allowlist of patterns. Conservative — only flagged statuses, not anything that *looks* like an error.
3. **`PruneStaleSteerRule`** — strip prior synthetic `USER_STEER` messages once the steered plan revision is `COMPLETED`. Today they stick in the transcript forever even after the steering took effect, wasting tokens and biasing the model.
4. **`CompactPriorReasoningRule`** *(optional, off by default)* — replace long-form reasoning blocks from completed tasks with a one-line summary. Risk: model may want to re-read reasoning for context. Phase 2 if validated; not shipped initially.

Rules 1-3 are conservative (only strip clearly-stale material). Rule 4 is opt-in.

## Implementation plan

| PR | Scope |
|---|---|
| 1 | `ContextEditor` skeleton + 5 invariants + `observation_only` gate + state-audit hook. No rules registered. Smoke test: editor is a no-op when rule set is empty. |
| 2 | `PruneCancelledReasoningRule`. Unit tests with synthetic cancel scenarios + e2e against the drift-loop reproducer. |
| 3 | `PruneTransientErrorRule`. Configurable allowlist. Unit tests for each known transient pattern. |
| 4 | `PruneStaleSteerRule`. Coordinates with refine-cycle to know when a steer has "taken". |
| 5 | *(Optional)* `CompactPriorReasoningRule`. Off by default. |

Each rule lands behind its own config flag (`SteeringConfig.context_editor_rules: list[str]`) so e2e regressions can be bisected.

## Alternatives considered

- **Replay via fresh session.** Instead of editing in-place, cancel the session and replay with a curated `contents` list. Heavyweight: re-creates ADK session, re-pays tool-registration cost, breaks observability continuity. Editing is cheaper.
- **Move pruning into the planner prompt.** Have the planner summarise history at refine time and inject the summary. Doesn't help — the model that sees the planner's summary is the same model whose context we're trying to clean.
- **Re-emit `Event`s via `on_event_callback`.** ADK supports returning a replacement event there, but it's per-event and doesn't compose well across many events. `before_model_callback` operates on the assembled `contents` list — natural scope for multi-event pruning rules.
- **Do nothing; rely on drift detectors.** Today's stance. Works for many cases but leaves the failure modes listed in *Motivation* unaddressed — they show up as drift loops (#173, #193) and inflated LLM-call ratios (#210).

## Open questions

- Should the editor reflect its edits into the harmonograf event stream (so the UI can visualise "what the model saw vs. what happened")? Recommendation: **yes** — emit `ContextEdited` with byte-delta + rule name. Without this, debugging a "the model ignored X" report becomes impossible.
- Should rules be allowed to *rewrite* (e.g., shorten) parts, or only drop? Phase 1 should be drop-only — rewriting opens correctness questions ("is the shortened version semantically equivalent?") that need their own design.
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
