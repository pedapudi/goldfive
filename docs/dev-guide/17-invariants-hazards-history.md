# 17. Invariants, Hazards, and History

## Read this chapter when...

- **Always, before any nontrivial edit.** This is the first chapter a weaker model must internalize. The rules here are not style preferences — each one is a scar from a specific incident that cost hours (sometimes days) of wrong fixes. `00-index.md` enforces reading this chapter before the subsystem chapters.
- You are about to add, move, or delete anything that could **intervene** on a running agent — cancel an invocation, inject a message, mutate ADK/session state, refuse a dispatch, install a plan, block on a human. Section 4 (Hazard Catalog) is your pre-flight.
- You are editing the **refine flow**, the **judge scheduler**, the **drift-condition lifecycle**, or any **gate** with a dictionary keyed on drift identity. Sections 1 and 4 tell you which keys are stable and which will silently disable your gate.
- You found code that "looks dead", "looks redundant", or "looks like a leftover" and you want to delete or "simplify" it. Section 2 (Protected List) exists precisely because several of those look-dead surfaces are load-bearing KEEP decisions guarded by history.
- You are tempted to copy a clever mechanism you saw on the `agency-preservation` branch onto `main`. Section 2 forbids it and Section 3 explains what is deferred and why.
- You are re-deriving "why is it shaped this way?" from scratch. Section 5 (History) is the curated timeline so you stop guessing intent.
- You are about to open a PR. Section 6 is the single consolidated checklist every change must pass.

If you read only one sentence: **goldfive's job is to watch agents and, by default, do nothing. Every line you add is guilty until proven passive.** The production default is `observation_only=True`, and under that default the correct behavior for almost every new code path is to observe and emit telemetry, never to touch the agent.

## Files covered

This chapter is cross-cutting; it references the whole tree. The load-bearing files for the rules here are:

| File | Why it matters to this chapter |
| --- | --- |
| `goldfive/steerer.py` | `DefaultSteerer.is_active_steering()` (line ~1339) and `steering_is_active(steerer)` (line ~107) — the ONLY sanctioned kill-switch reads (Invariant 5). `RefineExhausted`, the `InterventionLevel` ladder. |
| `goldfive/drift_observer.py` | The twin refine pipelines `_handle_drift_dispatch` (~3855) and `_promote_drift_to_steer` (~5457) that Section 3 requires you keep in sync; the judge semaphore + verdict-utility ledger; the drift-condition lifecycle; all the stacked suppression gates. |
| `goldfive/config.py` | Every frozen default (`observation_only=True`, `stall_watchdog_enabled=False`, `pause_escalate_deadline_s=None`, `name_axis_max_severity="info"`, `max_concurrent_judges=3`). Section 2 marks the bench-frozen ones. |
| `goldfive/reconciler.py` | `get_missed_tasks` (#163, KEEP), the disabled-but-KEEP `PLAN_DIVERGENCE` machinery (#252). |
| `goldfive/drift/tool_loops.py` + `goldfive/types.py` | `LOOPING_TOOL_CALL` enum + the tool-loop severity surfaces (#204/#206, KEEP). |
| `goldfive/_llm.py` | The single LLM-call module (#491) — `THINKING_DISABLE_CAPABILITIES`, `LlmCallDiagnostics` via `ContextVar`. Section 4's duck-typing / provider-fragility rules live here. |
| `goldfive/_correction_injection.py` | Correction keys use the full agent path (#479) — the unstable-key hazard made concrete. |
| `docs/design/AGENCY-PRESERVATION.md` (main) + same file on branch `agency-preservation` | The branch-boundary rules (Section 2.4) and the §6 as-built / deviation record. |
| `tests/conftest.py` | The active-mode opt-in fixtures (`active_steering_config`, `make_active_steerer`) after the autouse fixture was deleted (#488). Section 6 depends on these. |

## Invariants that bind you here

All six CANON invariants apply everywhere in the tree; this chapter is where they are defined in full. They are restated as Section 1 below with the incident, the historical catch, and the guarding grep/test for each. Do not treat any of them as advisory. A single violation of Invariant 1 or Invariant 5 is a release blocker because it changes behavior for every operator running the shipped default.

**The six invariants on one screen** (full treatment in Section 1):

| # | One-line rule | The forbidden shape | The guard |
| --- | --- | --- | --- |
| 1 | No prompt-cooperation contracts. | Control that only works if the agent obeys/calls a tool. | `tests/test_cooperative_cancellation.py`; enforcement is structural. |
| 2 | No regex/keyword NL classification. | A compiled regex / wordlist deciding a drift kind or severity. | `grep re.compile goldfive/drift/`; LLM judge instead. |
| 3 | Any ADK tree shape works. | `agent.sub_agents[0]`-style topology assumptions. | run all `tests/test_adk_*`; resolve via plugin registry. |
| 4 | Adaptive over predictive. | Reacting to a forecast of the agent's next move. | design test: "recorded fact or forecast?" |
| 5 | `observation_only` strictly passive; one predicate. | Any raw `_observation_only` read; any ungated mutation. | `grep _observation_only`; `steering_is_active` (fail PASSIVE). |
| 6 | Lifecycle gates need stable keys. | A gate keyed on `drift.id` / an LLM-minted id. | two-observation-collapse test; stable-key registry. |

---

# 1. The Invariants

There are six hard invariants. For each: **what it forbids**, **why (the incident/PR)**, **how violations were caught historically**, and **the grep/test that guards it now**. Memorize the shape: an invariant is not "be careful" — it is "this exact edit is forbidden, here is the command that catches you doing it."

## 1.0 Grounding: the enforcement vocabulary the invariants govern

Before the invariants, learn the two vocabularies they constrain, so the rules have referents. If you already know the ladder and the control kinds, skim.

**The intervention ladder** (`InterventionLevel`, `goldfive/steerer.py` ~149, an `enum.IntEnum`). This is the escalation ordering. Higher levels are more forceful and are the ones Invariant 1 and Invariant 5 most constrain:

| Level | Name | Mechanism | Requires agent cooperation? | Gated on `observation_only`? |
| --- | --- | --- | --- | --- |
| 1 | (observe) | Emit telemetry only; touch nothing. | No | Always safe (this is the passive default). |
| 2 | `NUDGE` | Queue a soft follow-up on `session.pending_nudges`; the overlay drains it. | Best-effort — agent may ignore it. | **Yes** (#475). |
| 3 | `CANCEL_REINVOKE` | Dispatch `GOLDFIVE_STEER`; executor cancels the in-flight invoke and restarts with a framed corrective. | No — structural. | **Yes**. |
| 4 | `PAUSE_ESCALATE` | Dispatch `GOLDFIVE_PAUSE_ESCALATE`; parks the run in a blocking wait for operator action. | No — structural. | **Yes**. |
| 5 | `TERMINATE` | Pause-with-deadline: same channel as Level 4 but bounded by `pause_escalate_deadline_s` or the built-in **600s** fallback; expiry CANCELs non-terminal tasks and emits `RunAborted` with escalation lineage (#482). | No — structural. | **Yes**. |

Levels 3–5 are **structural** — they drive the executor and plugin, never an instruction the agent can ignore (Invariant 1). Level 2 (NUDGE) is the only best-effort rung, and it is still gated (Invariant 5) and must be truthful (Hazard 4.2). Some low-severity kinds are NUDGE-absorb-only (`_ABSORB_NUDGE_KINDS`, `goldfive/steerer.py` ~174) — they never climb past Level 2 and recover at the next task boundary.

**The control channel** (`ControlKind`, a `StrEnum`; see `goldfive/executors/_control.py` module docstring). Control messages are how steers, pauses, approvals, and termination reach the executor. The goldfive-internal kinds are minted by the steerer, never by the agent:

| ControlKind | Origin | Effect |
| --- | --- | --- |
| `PAUSE` / `RESUME` | operator | pause/unpause the run loop between tasks. |
| `CANCEL` | operator | abort the run; executor emits `RunAborted`. |
| `STEER` | operator | feed to `steerer.drift.observe`; planner produces a fresh plan on a `USER_STEER` drift. |
| `REWIND_TO` | operator | mark a task + downstream `PENDING` for re-walk. |
| `STATUS_QUERY` | operator | **read-only** probe; returns a snapshot string; emits **no** sink events (polling must not register as drift). |
| `INTERCEPT_TRANSFER` | operator | toggle `session._intercept_transfer` so honoring adapters refuse transfers. |
| `APPROVE` / `REJECT` | operator | resolve a `session.pending_approvals` waiter; emits `ApprovalGranted`/`ApprovalRejected`. |
| `GOLDFIVE_STEER` | **steerer** | the internal cancel-and-restart junction for goldfive-detected drift; the steerer has **already swapped `session.plan`** before dispatch. |
| `GOLDFIVE_PAUSE_ESCALATE` | **steerer** | internal pause; may carry `deadline_s`; expiry aborts the run with escalation lineage (replaces the deleted `session.paused_for_human_intervention` flag). |

Two design facts you must not break: `STATUS_QUERY` is read-only and silent (do not make it emit events); and the `GOLDFIVE_*` kinds mint from the steerer *after* the plan swap, so the control message is *only* the cancel-and-restart signal, not the plan itself (`#402`/`#403` closed the ordering / partial-apply windows here).

## Invariant 1 — No prompt-cooperation contracts

**What it forbids.** goldfive's termination, cancellation, pause, and observability must work **even if the agent never calls a goldfive tool and never reads an injected instruction.** You may not build any control or observability mechanism whose correctness depends on the agent:

- calling a specific tool (e.g. `report_progress`, `report_awaiting_approval`);
- following a system-prompt instruction ("when you are done, call X", "if you are stuck, stop");
- honoring an injected "please stop / please refocus" message.

Injected nudges and corrective messages are allowed as **best-effort additions**, never as the load-bearing enforcement path. The enforcement path is always structural: the executor stops scheduling, the plugin sets a cancel flag, the plan is swapped, the run is aborted. The agent's cooperation is a bonus, never a precondition.

**Why — the incident.** goldfive is a wrapper for *other people's* agent trees. Operators bring their own coordinator prompts (see memory `feedback_no_prompt_contract`). If goldfive required the agent to call a goldfive tool to be terminable, then any operator whose prompt didn't mention that tool would have an unkillable run. The whole `goldfive.wrap` contract (memory `feedback_goldfive_wrap_contract`) is "any ADK tree, unmodified." An intervention the agent can ignore is not an intervention.

**How violations were caught historically.** The `#271` program (see Section 5) repeatedly found control paths that only worked if the agent behaved. The fix arc — `#297` (serialize per-key run so the stash can't race), `#303`/`#307`/`#310` (goldfive-owned invocation boundary + cancellation-stash tripwire), `#315` (close open invocation boundaries on terminal drifts) — moved enforcement out of "ask the agent" and into goldfive-owned structure. `#476` (2026-07) is the modern echo: under `observation_only`, a reasoning-channel disarm no longer *cancels* — it emits one loud warning, because cancellation is an intervention and warning is observation.

**Guarding grep / test.**

```bash
# Any new "enforcement" that reads an agent tool call as the trigger is suspect.
grep -rn "report_awaiting_approval\|report_progress" goldfive/executors/ goldfive/runner.py
# TERMINATE / CANCEL must be driven by the executor, never by an agent tool result:
grep -rn "TERMINATE\|RunAborted\|request_invocation_cancel" goldfive/executors/_control.py
```

Tests: `tests/test_cooperative_cancellation.py`, `tests/test_cancel_propagation.py`, `tests/test_control_primitive.py`. The `report_awaiting_approval` no-channel path (`tests/test_approval_flow.py`) proves the run does **not** hang when nobody is listening — cooperation absent, run still progresses (#478).

**DO / DON'T.**

| DON'T (violates Inv 1) | DO instead |
| --- | --- |
| Inject "STOP — you are off task" and rely on the agent halting. | Dispatch `GOLDFIVE_STEER`/`GOLDFIVE_PAUSE_ESCALATE`; the executor cancels the invoke regardless of the agent. |
| Add a system-prompt line "call `report_done()` when finished" and drive termination off that call. | Terminate on generator-end + drift detectors + the stall watchdog — none of which need the agent to call anything. |
| Make a feature that only works if the operator's coordinator prompt mentions goldfive. | Make it work on an unmodified tree (`goldfive.wrap` contract). |

**Concrete wrong edit a weak model makes:** "The agent ignores my nudge, so I'll make the nudge a hard requirement by refusing to proceed until it acknowledges." Wrong — that is a cooperation contract. Correct: if the nudge (Level 2) is ignored and the drift persists, the *ladder* escalates to Level 3 CANCEL_REINVOKE, which is structural and needs no acknowledgment.

## Invariant 2 — No regex/keyword heuristics for natural-language classification

**What it forbids.** You may **not** classify agent free-text (thinking tokens, tool args, output prose) with regexes, keyword lists, verb-prefix matching, or any lexical heuristic to decide a drift kind, a severity, or an intervention. Natural-language classification goes through an **LLM classifier/judge**, or you redesign so the classification is unnecessary. What **is** allowed: exact-equality and hash matching of *structured* data — drift ids, `(name, args_hash)` tuples, revision indices, `(kind, task_id)` keys. Structured equality is not "NL classification."

**Why — the incident.** Two retired heuristics: `_GENERIC_VERB_PREFIX_RE` (#166) and `_FACTUAL_QUESTION_RE` (#167). Earlier still, a regex CONFUSION detector (#358 revert) and the R1/R2 regex heuristics (#331 revert). They all produced the same failure: brittle false positives/negatives that shifted with prompt wording, un-tunable, and un-explainable. Memory `feedback_no_regex_heuristics` records the standing ban. `#484` (2026-07) is the current-era discipline: a tool-loop's *name-axis* similarity is capped at `INFO` **unless** there is exact-repeat corroboration (`>=2` identical `(name, args_hash)`), because "same tool name, different args" is a lexical near-match that must not by itself escalate.

**How violations were caught historically.** Code review + the revert PRs above. The tell is a module-level compiled regex or a `set` of trigger words consulted during drift decisions.

**Guarding grep / test.**

```bash
# No compiled regexes deciding drift/severity. (Structural regexes for parsing wire formats are fine; NL-classification regexes are not.)
grep -rn "re\.compile\|re\.search\|re\.match" goldfive/drift/ goldfive/drift_observer.py goldfive/judges/ goldfive/builtin_judges.py
# The retired names must never come back:
grep -rn "_GENERIC_VERB_PREFIX_RE\|_FACTUAL_QUESTION_RE\|CONFUSION" goldfive/
```

The tool-loop corroboration rule is tested in `tests/` under the tool-loop severity-cap tests (grep `severity_capped_from` / `name_axis_max_severity`). If you add a detector, its severity must come from a deterministic structural signal or an LLM verdict — never a wordlist.

**DO / DON'T.**

| DON'T (violates Inv 2) | DO instead |
| --- | --- |
| `if re.search(r"\b(confused|stuck|lost)\b", thinking):` → raise CONFUSION drift. | Send the thinking window to the reasoning judge; consume its typed `JudgeVerdict`. |
| Keyword-match tool args to decide "this looks like a retry loop". | Hash `(name, args_hash)`; require `>=2` exact repeats before escalating past INFO (#484). |
| Add a `set` of "off-topic phrases" to bump severity. | Use the embedding-similarity band (deterministic, structural) or a judge. |

**What structured matching is allowed:** hashing `(name, args_hash)` for exact-repeat detection, comparing `(kind.value, task_id)` for gate keys, equality on revision indices, sha1 of stable structured fields for condition ids. None of these read natural language.

**Concrete wrong edit:** "The judge is slow, so I'll pre-filter with a cheap regex that only sends 'suspicious' thinking to the judge." Wrong — the regex *is* an NL classifier deciding what's suspicious, and it will systematically drop cases it wasn't written for. Correct: use the judge scheduler's semaphore + coalescing (#483) to control cost, not a lexical pre-filter.

## Invariant 3 — Any ADK tree shape must work

**What it forbids.** You may not assume a tree shape. A flat single agent, a `SequentialAgent`, a `ParallelAgent`, and — the hard case — a **coordinator that calls sub-agents via `AgentTool`** must all wrap and orchestrate identically. Cancel must resolve invocation ids through the plugin registry, not through a fixed parent/child relationship. Do not blame the tree shape when orchestration fails on coordinator+AgentTool — that is a goldfive bug to fix (memory `feedback_goldfive_wrap_contract`).

**Why — the incident.** The coordinator+AgentTool pattern hides sub-agent invocations behind a tool call, so naive "watch the top-level agent" logic sees nothing. The adapter/plugin layer (`goldfive/adapters/_adk_plugin.py`) was built to observe *every* invocation regardless of nesting. `request_invocation_cancel` must no-op cleanly on an unbound adapter, a non-ADK adapter, or an empty invocation-id list — because the resolver runs on trees where the id set is legitimately empty.

**How violations were caught historically.** E2E runs on the coordinator+AgentTool driver (see memory `feedback_e2e_validation_layered`). Unit tests that only build a flat agent pass while the real tree breaks — memory `feedback_integration_not_unit` warns exactly this.

**Guarding grep / test.**

```bash
grep -rn "AgentTool\|coordinator" tests/ | head
```

Tests: `tests/test_adk_adapter_concurrent_sessions.py`, `tests/test_adk_adapter_overlay.py`, `tests/test_adk_reentry.py`, `tests/test_adk_wrap_passthrough.py`, `tests/test_delegation_pin.py`. After touching the adapter, run all `tests/test_adk_*` — a flat-agent-only test passing is not evidence.

**Concrete wrong edit:** "Cancel the sub-agent by reaching into `agent.sub_agents[0]`." Wrong — a coordinator+AgentTool tree has no `sub_agents` list for the tool-invoked agent; the invocation lives behind a tool call. Correct: `request_invocation_cancel` resolves ids through the plugin registry and no-ops on an empty set, so it works for both shapes without knowing the topology.

## Invariant 4 — Adaptive over predictive

**What it forbids.** You may not **predict** what the agent will do and pre-empt it. Capture **observed facts** and react to them. Concretely: gates read state that has already been recorded (`occurrence_count` from `session.refine_outcomes`, `observed_revision_index` from the plan snapshot, `last_observed_event_at` liveness stamp), never a forecast of the agent's next move. Do not intercept at pin/dispatch time to "get ahead" of behavior the agent has not yet exhibited; extend the observed-fact protos/events instead.

**Why — the incident.** Memory `feedback_dont_predict_agent_behavior`: "Predictive must be adaptive (or just adaptive)." Agents have agency; predictive interception both violates that and is wrong more often than it is right (the model does something you didn't forecast). The `#423` "plan-descriptive growth" arc reframed the planner around *describing what happened* (unmatched delegations become descriptive tasks) rather than *predicting* the delegation graph up front.

**How violations were caught historically.** Design review; the tell is code that mutates control state based on a guess about a future turn. The correction is to add an observed-fact field to `DelegationObserved` / `DriftEvent` / a new event, and let a later observation drive the reaction.

**Guarding grep / test.** No single grep. The design test: for every new reaction, ask "is the fact I'm reacting to already recorded, or am I forecasting?" If forecasting, stop. Cross-reference `07-deterministic-drift-detection.md` and `10-planning-and-revision.md`.

**DO / DON'T.**

| DON'T (violates Inv 4) | DO instead |
| --- | --- |
| Pin task N+1's assignee up front because "the coordinator usually delegates to X next." | Observe the actual delegation, then describe it (`DelegationObserved`, `Task.discovered`). |
| Fire drift when the agent's action isn't in your predicted plan. | Grow the plan descriptively (#423); reserve drift for observed-fact guardrails or goal-referenced judges. |
| Intercept a tool call at dispatch to pre-empt a predicted loop. | Let it happen, observe `(name, args_hash)`, fire only on `>=2` exact repeats (#484). |
| Add a proto field that stores a *prediction* of the agent's next move. | Add a field that records an *observed fact* after it happens. |

**Concrete wrong edit:** "I can save a turn by pre-marking the next task IN_PROGRESS when I see the coordinator about to delegate." Wrong — you are predicting the delegation; the coordinator may delegate elsewhere, and now the task state lies. Correct: mark it when the delegation is *observed*.

## Invariant 5 — `observation_only=True` is the production default and is STRICTLY passive

**What it forbids.** This is the single most safety-critical rule in the codebase. Since the Waves 1–4 hardening, `observation_only` is **strictly passive**: when it is `True` (the shipped default), goldfive observes and emits telemetry and does **nothing** that mutates the agent, the session's control state, or the plan install. The ONLY sanctioned reads of the kill-switch are:

- `DefaultSteerer.is_active_steering()` — router-internal, the single source of truth (`goldfive/steerer.py` ~1339);
- `steering_is_active(steerer)` — the one documented module-level fallback for any external consumer holding a maybe-steerer (`goldfive/steerer.py` ~107).

Both fail **PASSIVE**: a `None` steerer, a steerer missing the predicate, or a predicate that raises → `False` (not active). You may **not** read `_observation_only` directly anywhere except inside `is_active_steering()` itself. Every intervention surface — cancel, nudge, plan install, prompt-shape injection, context edit, directive ack carrying goldfive-authored state — must be gated on this predicate.

Here is the exact fail-safe helper you must route through (`goldfive/steerer.py`):

```python
def steering_is_active(steerer: Any) -> bool:
    predicate = getattr(steerer, "is_active_steering", None)
    if not callable(predicate):
        return False
    try:
        return bool(predicate())
    except Exception:  # noqa: BLE001
        return False
```

**Why — the incident.** `observation_only` (goldfive#254) is the master kill-switch operators trust to run goldfive in shadow mode against production agents. If any single write path is ungated, that path fires for every operator on the default config — a silent, universal breach of "shadow mode." Before `#488`, the kill-switch was read in many places via raw `getattr(..., "_observation_only", ...)`, several of which defaulted to **ACTIVE** on a missing attribute — the wrong fail direction. `#488` collapsed all reads to the one predicate, deleted the module-global test hook and the autouse conftest fixture that had been flipping the whole suite to active mode, and flipped the fallback direction to PASSIVE at `PromptShaper.should_inject`, the `sequential.py` executor carve-outs, and the F1 directive-ack gate. The 2026-07 program then swept every remaining surface: `#475` (nudge path gated + truthful text), `#476` (LLM_CALL_TIMEOUT no longer cancels under `observation_only`), `#478` (`plan_state` stripped from acks under `observation_only`), `#481` (F3 pre-dispatch redirect gated). See memory `project_agency_preservation_roadmap` and `#394` (strict-passive: strip goldfive's own helper prompt directives, closes #271).

**How violations were caught historically.** The Waves 1–4 audit walked *every* mutation site and asked "is this gated?" Four separate production leaks were found this way (nudge, timeout-cancel, plan_state ack, F3 redirect). A unit test in active mode passing is not evidence — the leak only shows in passive mode. Hence the both-modes rule (Section 6).

**Guarding grep / test.**

```bash
# The kill-switch must be read ONLY through the predicate. Any other read is a bug.
grep -rn "_observation_only" goldfive/ | grep -v "def is_active_steering" | grep -v "steering_is_active"
# Every consumer of the flag should route through steering_is_active / is_active_steering:
grep -rn "steering_is_active\|is_active_steering" goldfive/
```

Tests: `tests/conftest.py` provides `active_steering_config` and `make_active_steerer` — active mode is now **opt-in**; ~90 tests explicitly request it and the rest run the shipped `observation_only=True` default (#488). Every new intervention needs a **passive test** (asserts nothing happens) AND an **active test** (asserts the intervention fires). See `09-steering-ladder-and-gates.md` for the surface-by-surface gate map.

**DO / DON'T.**

| DON'T (violates Inv 5) | DO instead |
| --- | --- |
| `if not steerer._observation_only:` anywhere outside `is_active_steering()`. | `if steering_is_active(steerer):` (or `self.is_active_steering()` inside `DefaultSteerer`). |
| `active = getattr(steerer, "_observation_only", False)` then invert it — fails ACTIVE on a missing attr. | `steering_is_active(steerer)` — fails **PASSIVE** on missing/None/raising. |
| Gate only the unit test to active mode; leave the runtime path ungated. | Gate the runtime path; write BOTH a passive and an active test. |
| Emit a directive ack carrying goldfive-authored `plan_state` unconditionally. | Strip goldfive-authored state from acks under `observation_only` (#478). |

**The four historical leaks** — memorize them as the archetypes, because your new surface will resemble one:

1. **NUDGE injection ungated** (#475) — a message written to the agent's context in passive mode.
2. **Timeout→cancel** (#476) — an infrastructure timeout cancelling healthy work in passive mode.
3. **`plan_state` in acks** (#478) — goldfive-authored state leaking to the agent via a "harmless" ack.
4. **F3 pre-dispatch redirect** (#481) — a dispatch rewrite firing in passive mode.

Every one of these *looked* passive ("it's just a message", "it's just a timeout", "it's just an ack field") and was not. When you add a surface, ask: does anything the agent can observe or that changes control flow differ between passive and active? If yes, it is an intervention and it gates.

**Concrete wrong edit:** "I'll add a small `session.state['goldfive_hint'] = ...` write so the agent knows it drifted; it's read-only advice." Wrong — a state write the agent can read is an intervention that fires in passive mode. Correct: emit a sink event / drift telemetry (observers see it, the agent does not), and only write to the agent's context when `steering_is_active(steerer)`.

**"Strictly passive" means goldfive's own prompt directives are stripped too (#394).** Before #394, even in `observation_only` goldfive injected its own helper-prompt directives (framing the agent to call goldfive tools, describing the plan, etc.). That is a per-turn footprint — a violation of the dormant contract (Section 5.9) even though it's not a "steer." `#394` (closes #271) strips those directives under `observation_only`: in passive mode the agent sees its *native* prompt, unmodified by goldfive. **Rule:** if you add any prompt content goldfive authors, it must be gated — passive mode has **zero** goldfive-authored prompt footprint. This is stricter than "don't intervene"; it is "don't even shape the prompt."

### 1.5.1 The complete `observation_only` gate-surface map

This is the authoritative list of every write-path the kill-switch suppresses (from `DefaultSteerer.is_active_steering()` and `SteeringConfig.observation_only` docstrings). When `observation_only=True`, **all** of these are skipped; everything else — detection, `planner.refine_steer`, `DriftDetected`/`PlanRevised` emission (with `dry_run=True`), logging, drift lifecycle, suppression accounting — keeps running.

| Write-path | Method | File |
| --- | --- | --- |
| Plan mutation (replace `session.plan` + `last_addressed_revision_by_drift_key`) | `PlanReviser._apply_revision` | `goldfive/plan_reviser.py` |
| `GOLDFIVE_STEER` ControlMessage enqueue | `DriftObserver._dispatch_goldfive_steer_control` | `goldfive/drift_observer.py` |
| Plugin cancel flag | `DriftObserver.request_invocation_cancel` | `goldfive/drift_observer.py` |
| Level-2 nudge enqueue onto `session.pending_nudges` (+ post-ABSORB handoff, #202) | `DriftObserver._dispatch_nudge` | `goldfive/drift_observer.py` |
| Prompt-shape injections | `PromptShaper.should_inject` (four bodies) | `goldfive/prompt_shaper.py` |
| Executor carve-out (defense-in-depth for subclasses that bypass the dispatcher, #264) | overlay drain gates | `goldfive/executors/sequential.py` |
| Plugin pre-dispatch gates (incl. F3 redirect, #481; LLM-timeout cancel, #476) | `_is_observation_only(ctx)` | `goldfive/adapters/_adk_plugin.py` |
| F1 directive acks (strip goldfive-authored `plan_state`, #478) | `_directive_ack` | `goldfive/reporting/rendering.py` |

**The nudge is an injection, not an observation.** A common misread: "queuing a nudge is passive, it's just adding to a list." No — the overlay *drains* `pending_nudges` into a **synthetic goldfive-authored user turn** and re-invokes the tree (#202). That is a write to the agent's context, so it gates.

**Three revision categories always land even under `observation_only`** (they are not "corrective goldfive-authored" revisions — the gate suppresses only those):

1. **bootstrap** — first install on a cold session (`prev is None`).
2. **user-authored** — operator `ControlMessage` STEER (`drift.authored_by == "user"`).
3. **discovery** — `DriftKind.NEW_WORK_DISCOVERED` revisions (#258): the planner/sub-agent *describing observed work*, both the turn-1 `install_initial_plan` (seeded with `Plan.empty()`) and the N+1 `install_revision_for_drift`.

**Mistake:** gating a discovery/bootstrap revision behind `observation_only` "for consistency." Wrong — those are not corrective interventions; suppressing them would leave the session with no plan at all in passive mode, breaking observation itself. Only *corrective* revisions gate. The env override is `GOLDFIVE_STEER_OBSERVATION_ONLY=0` (or `false`/`no`) to graduate to active steering.

### 1.5.2 Intervention surfaces you might not recognize

The obvious surfaces are nudge/cancel/plan-install. These four are easy to miss — each is a real intervention and each is gated:

- **`ContextEditor.apply`** (`goldfive/context_editor.py`, #397) — request-side context editing (drop-only, no injection). `observation_only` is a **hard gate (Invariant 1)**: `if observation_only: return` is a complete no-op at the top of `apply` (~481). It also honors Invariant 3 (drop-only / no-injection — byte total must not grow) and Invariant 4 (idempotence-per-revision). A weak model editing context-editing logic must keep the `observation_only` no-op the very first thing `apply` does.
- **`PromptShaper`'s four injection bodies** (`goldfive/prompt_shaper.py`) — Site 1 (conversational follow-up wrap) and three others, all consulting the **single** gate `should_inject(steerer) → steering_is_active(steerer)`. Pre-refactor this site defaulted ACTIVE on a missing attr; the flip to PASSIVE fallback is deliberate (#488). Do not add a fifth injection body with its own gate — route through `should_inject`.
- **F1 directive acks** (`goldfive/reporting/rendering.py`) — F1's directive payload is a proactive anchor; under `observation_only` the goldfive-authored `plan_state` is stripped from the ack (#478). The ack itself is fine; the *goldfive-authored state* in it is the intervention.
- **F3 pre-dispatch redirect** (`goldfive/adapters/_adk_plugin.py` ~646) — Tier-1 redirect for AgentTool calls on **completed** (not merely terminal) work. Gated on `observation_only` (#481), and its predicate is aligned with the delegation pin (`_maybe_pin_delegation_task`) so it uses the same COMPLETED semantics — F3's earlier local copy counted merely-terminal predecessors, so a `NOT_NEEDED` sweep could make it redirect against work that was never actually done.

**The reconciler is a forecast-grader — the agency landmine.** `docs/design/AGENCY-PRESERVATION.md` §1 names the reconciler grading every delegation against a forecast plan as *manufacturing* the drift signal that justifies engagement (a violation of the dormant contract, not of a hard invariant — but the branch's whole point). On main this is why `PLAN_DIVERGENCE` was disabled (#252, Section 2.2) in favor of `CAPABILITY_MISMATCH`, and why the descriptive-growth arc (#423) reframed unmatched delegations as *descriptions* rather than *divergences*. **Rule for your edits:** do not add new forecast-grading that fires drift for "the agent did something not in my predicted plan." That is predictive (Inv 4) and it manufactures signal. Describe what happened; only fire drift on *observed-fact* guardrails (loops, stalls, budgets) or a *goal*-referenced judge.

## Invariant 6 — Lifecycle gates need stable identity keys

**What it forbids.** Any gate — a freshness watermark, an in-flight set, a suppression window, a drift-condition ledger, a correction map — must key on a **stable** identity tuple. You may **not** key a gate on:

- a per-event `drift.id` (a fresh UUID4 per emit);
- any LLM-minted id (task ids the model invented, agent ids it renamed mid-run);
- a global writer-centric counter that churns under multi-observer fan-in.

If a component in your key churns, the gate opens a fresh entry per observation and **never engages** — it silently does nothing. Fix the churn upstream; do not make the key coarser to paper over it.

**Why — the incident.** Memory `feedback_stable_keys_for_lifecycle_gates` and `feedback_reader_centric_versioning`. A per-condition gate keyed on a churning id opens a new entry every observation, so the suppression/freshness logic never fires. `#479` made correction keys use the **full agent path** (not a bare agent name) precisely so two agents in different subtrees get distinct, stable keys (`goldfive/_correction_injection.py`, `pending_correction_key(agent_name, task_id)` with `_normalize_agent_name` producing the full path). `#442` keyed the user-steer suppression window on a **logical-turn counter** rather than anything the LLM controls. The drift-condition lifecycle keys on `sha1(kind, task_id, agent_id, turn_id)` — a content hash of stable structured fields, not the event id.

**How violations were caught historically.** A gate that "should suppress" but never does, observed as duplicate refines / duplicate drifts on the wire. `#420` fixed exactly this class (`tool_loops` bucket scoping + freshness-gate atomicity + double-cancel dedup).

**Guarding grep / test.**

```bash
# Gates keyed on drift.id are almost always wrong. Audit any dict keyed on a UUID.
grep -rn "\.id\]" goldfive/drift_observer.py
grep -rn "pending_correction_key\|_normalize_agent_name" goldfive/_correction_injection.py
```

Tests: `tests/test_correction_injection.py`, and the freshness/in-flight gate tests in the drift-observer test set. When you add a gate, write a test with **two observations that must collapse to one** — if they don't, your key is unstable.

**The stable-key registry** (memorize which key each gate uses; a new gate should match the pattern of the gate nearest it):

| Gate | Stable key | NOT the key |
| --- | --- | --- |
| Freshness watermark | `(kind.value, current_task_id)` | `drift.id` |
| In-flight refine | `(session_id, kind.value, current_task_id)` | `drift.id` |
| Refine-outcome | `(kind.value, current_task_id)` | `drift.id` |
| Drift condition | `sha1(kind, task_id, agent_id, turn_id)` | the raw event id |
| Correction injection | full agent path + `task_id` (`pending_correction_key`) | bare agent name |
| User-steer suppression | logical-turn counter (#442) | wall-clock time / LLM turn label |

**Concrete wrong edit:** "The freshness gate over-suppresses across tasks, so I'll add `drift.id` to the key to make it more specific." Wrong — `drift.id` is a fresh UUID4 per emit, so *every* observation gets a unique key and the gate never suppresses anything (it silently becomes a no-op). Correct: if the gate over-suppresses, the *task* dimension is the axis to refine (`current_task_id`), and if a churning id is the root cause, fix the churn upstream (that is what #479's full-agent-path correction key did).

---

# 1.7 Recipes: adding a new X without breaking an invariant

The four most common "add a thing" tasks, each as an ordered checklist that folds in the invariants this chapter defines. These are the invariant-critical steps only — see the `.agents/how-to-*.md` skills for the full mechanical walkthroughs (they win on mechanics; this chapter wins on which invariant each step protects).

## 1.7a Adding a new intervention surface (the highest-risk task)

An intervention surface = any code that cancels, injects, mutates ADK/session control state, refuses a dispatch, installs a plan, or emits a directive ack carrying goldfive-authored state.

1. **Locate the nearest existing surface** in the Section 1.5.1 map and copy its gate shape.
2. **Gate it:** wrap the mutation in `if steering_is_active(steerer):` (or `if self.is_active_steering():` inside `DefaultSteerer`). Fail PASSIVE — never `getattr(..., "_observation_only", ...)`.
3. **Make injected text truthful** (Hazard 4.2): every factual claim comes from state verified this turn.
4. **Bound any wait** (Hazard 4.3): finite deadline + defined expiry behavior.
5. **Emit the decision telemetry** (Section 3.6): both the positive fire and, if it's a detector, the `no_drift` negative class, with the right `detector_name`.
6. **Write BOTH tests:** a passive test asserting nothing happens, an active test (`active_steering_config` / `make_active_steerer`) asserting it fires.
7. **Verify the tree-shape** (Inv 3): if it cancels, resolve ids through the plugin registry; run all `tests/test_adk_*`.

## 1.7b Adding a new `DriftKind`

1. **Python enum** in `goldfive/types.py::DriftKind`, placed with its category; comment the default severity + trigger + any feature gate.
2. **Proto enum** in `proto/goldfive/v1/types.proto` — **append** `DRIFT_KIND_<NAME>` with the next number; never renumber (Hazard 4.11). The Python↔proto bridge is by name (`getattr(types_pb2, f"DRIFT_KIND_{kind.name}")`), so the names must match exactly. Regenerate stubs; don't hand-edit `_pb2`.
3. **Classifier/detector** — the severity comes from a deterministic structural signal or an LLM verdict, **never a wordlist** (Inv 2).
4. **Steerer dispatch** — add the ladder entry; decide the intervention level. If it's a guardrail (loop/stall/budget), it's always-armed and cheap; if it's steering, it engages only on drift and gates on `observation_only`.
5. **Gate keys** (Inv 6): any suppression you add for the new kind keys on `(kind.value, current_task_id)`-shaped stable tuples, never `drift.id`.
6. **Both refine pipelines** (Section 3.1) if the kind flows through refine.
7. **Tests + `docs/design/DRIFT.md` taxonomy row.**
8. **Manifest** (Hazard 4.13): if the kind introduces a tunable threshold, add its `optimization/manifest.toml` entry.

## 1.7c Adding a new event / proto field

1. **Append** the field with the next number (Hazard 4.11); reserve-and-append, never insert/renumber.
2. **Route the emit through `emit`/`make_event`** so `event_id` is stamped (Hazard 4.12); don't build envelopes by hand.
3. **Keep the JSONL camelCase** (Hazard 4.13); regenerate stubs.
4. **If it's a decision event,** use the `outcome` vocabulary consistently (Section 3.6).
5. **harmonograf:** if it's a framework-authored drift, mark it synthetic (#302) so the UI filters it from interventions.

## 1.7d Adding a new config knob

1. **Add the field** to the right dataclass in `goldfive/config.py` with a conservative default and a docstring citing the PR.
2. **Add the env reader** (`_read_bool_env` / `_read_int_env` / `_read_float_env` / `_read_optional_float_env` in `config.py`; `_read_severity_env` lives in `goldfive/drift/tool_loops.py` and is imported) in the `from_env` path, matching the sibling knobs.
3. **Decide the default direction:** if the knob enables an intervention, default it **OFF/passive** (like `stall_watchdog_enabled=False`, `pause_escalate_deadline_s=None`). A new-feature knob defaults to the *dormant* behavior.
4. **Manifest** (Hazard 4.13): add the `optimization/manifest.toml` entry so zicato can tune it; the AST-liveness test (#487) enforces the entry points at a live symbol.
5. **Do not** make the new knob a Section 2.5 bench-frozen default that flips existing behavior without sign-off.

## 1.8 The shape of a correct intervention (worked)

This is the canonical shape every new intervention surface should match. It is illustrative pseudo-code assembling the rules — not a verbatim excerpt — but every line maps to a rule above.

```python
async def _maybe_intervene(self, drift: DriftEvent, session: Session) -> None:
    # 1. ALWAYS observe — telemetry fires regardless of mode (dormant contract:
    #    observation has zero dependence on the kill-switch).
    await self._emit_drift_detected(session, drift)

    # 2. Decision telemetry incl. the negative class (Section 3.6). Even if we
    #    do NOT intervene, the decision is recorded so zicato can measure.
    if not self._detector_fired(drift):
        await self.emit_no_drift_decision(session, drift, detector_name="my_detector")
        return

    # 3. GATE the mutation on the single predicate, failing PASSIVE (Inv 5).
    #    In observation_only (the default) we stop HERE — detection ran,
    #    refine_steer may run in dry_run, but the agent is untouched.
    if not self.is_active_steering():
        return

    # 4. Bound every wait (Hazard 4.3) and key any suppression stably (Inv 6).
    key = (drift.kind.value, session.current_task_id)   # NOT drift.id
    if self._recently_addressed(key):
        return

    # 5. Injected text must be TRUE at emit time (Hazard 4.2): re-read the count.
    count = self._observed_repeat_count(session)         # fresh, not cached
    nudge = compose_corrective_user_message(drift, observed_repeats=count)

    # 6. Enforcement is STRUCTURAL, not a please-stop the agent can ignore (Inv 1).
    #    The nudge is a best-effort addition; the ladder escalates structurally.
    await self._dispatch_nudge(session, nudge)           # gated Level-2 surface
```

**Read that top-to-bottom as the invariant order:** observe → record the negative class → gate PASSIVE → stable key → truthful text → structural enforcement. If your new surface can't be written in that shape, it is probably violating one of the six invariants — find which.

## 1.9 Which invariants bind which files (fast lookup)

Before editing a file, check which invariants are live there. `✓` = this invariant is a primary concern in that file; edits here are the classic way to violate it.

| File | Inv 1 (no coop) | Inv 2 (no regex) | Inv 3 (tree shape) | Inv 4 (adaptive) | Inv 5 (passive) | Inv 6 (stable keys) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `steerer.py` | ✓ | | | | ✓ | ✓ |
| `drift_observer.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `plan_reviser.py` | ✓ | | | ✓ | ✓ | ✓ |
| `drift/*.py` (detectors) | | ✓ | | ✓ | | ✓ |
| `judges/`, `builtin_judges.py` | | ✓ | | | | |
| `reconciler.py` | | | ✓ | ✓ | | ✓ |
| `executors/sequential.py` | ✓ | | ✓ | | ✓ | |
| `executors/parallel.py` | ✓ | | ✓ | | ✓ | |
| `adapters/_adk_plugin.py` | ✓ | | ✓ | ✓ | ✓ | ✓ |
| `reporting/*.py` | ✓ | | | | ✓ | ✓ |
| `prompt_shaper.py` | ✓ | | | | ✓ | |
| `context_editor.py` | ✓ | | | ✓ | ✓ | |
| `_correction_injection.py` | | | | | | ✓ |
| `_llm.py` | | | ✓ | | | |
| `events.py`, `pb/` | | | | | | ✓ |
| `state_store.py`, `types.py` | | | | | | ✓ |
| `config.py` | | | | | ✓ | |

Read the corresponding invariant section for every `✓` in the row of the file you are about to touch.

## 1.10 Guardrail vs steering: which drift kinds are always-armed

The dormant-supervisor framing (Section 5.9) splits drift kinds into **guardrails** (observed-fact, cheap, always-armed even in dormant intent) and **steering** (goal-referenced, LLM-judged, engages only on drift, gates on `observation_only`). Knowing which is which tells you how much gating a kind needs.

| Kind | Class | Why |
| --- | --- | --- |
| `LOOPING_TOOL_CALL` / `LOOPING_REASONING` | Guardrail | Observed structural repetition; cheap; low false-positive (with #484 corroboration). |
| `TASK_TIMEOUT` (stall watchdog) | Guardrail | Wall-clock liveness; observed fact; but the *producer* is flag-gated OFF by default (#487). |
| budget / runaway-delegation | Guardrail | Counted observed facts. |
| refusal / escape | Guardrail | Observed. |
| `GOAL_DRIFT` / `OFF_TOPIC` / `INTENT_DIVERGENCE` | Steering | Requires a goal reference + LLM judge; expensive/fallible; resolves at task-terminal (#486). |
| `CAPABILITY_MISMATCH` | Steering (planning artifact) | The one the AGENCY doc flags: it ended up with the same enforcement machinery as a loop detector, which the branch aims to separate. |
| `USER_STEER` | (operator) | Always lands — user authority, not a framework correction. |
| `NEW_WORK_DISCOVERED` | (descriptive) | Discovery, always lands — describing observed work, not correcting (#258). |

**Mistake:** wiring a new guardrail (say a new loop shape) through the full steering refine machinery. Guardrails *stop* runaway behavior with a cheap structural signal; they should not require an LLM judge or a goal reference. Wire a guardrail as a deterministic detector; reserve the judge/refine path for goal-referenced steering.

---

# 2. The Protected List — KEEP decisions and frozen defaults

These are surfaces that **look** deletable, "fixable", or tunable but are not, without **explicit human sign-off**. A weaker model's instinct — "this is dead / redundant / disabled, remove it" or "this default seems conservative, raise it" — is wrong for every item below. When in doubt: leave it, cite this section in your PR, and ask.

## 2.1 KEEP: `LOOPING_TOOL_CALL` surfaces (#204 / #206)

- **Enum:** `DriftKind.LOOPING_TOOL_CALL = "looping_tool_call"` (`goldfive/types.py` ~138).
- **Detector + ladder + promotion + planner surfaces:** `goldfive/drift/tool_loops.py`, and the ladder/promotion routing in `goldfive/drift_observer.py`.
- **Deliberate design:** tool loops emit `LOOPING_REASONING` with **NUDGE-first CRITICAL routing** — the escalation deliberately leads with a nudge even at CRITICAL severity, per the #204/#206 history. Do **not** "simplify" this to a straight CANCEL, and do not delete the `LOOPING_TOOL_CALL` enum member because "nothing seems to raise it directly." The enum, the ladder rung, the promotion path, and the planner-facing surface are all load-bearing and were shaped by the #204/#206 iterations (see `#356` "graceful refine fallback escalates to PAUSE_ESCALATE (iter-12 #204)").
- **Interaction with #484:** `#484` capped the *name-axis* of tool-loop detection at INFO without exact-repeat corroboration. That cap is a severity floor on an uncorroborated signal — it does **not** retire the `LOOPING_TOOL_CALL` machinery. Keep them distinct in your head: #484 tunes when the signal is trustworthy; #204/#206 own what happens once it is.

**Grep before touching:** `grep -rn "LOOPING_TOOL_CALL\|LOOPING_REASONING\|name_axis_max_severity" goldfive/`.

## 2.2 KEEP: `PLAN_DIVERGENCE` machinery (#252-disabled, branch KEEP)

- **Location:** `goldfive/reconciler.py` (the emit path ~line 229, the disabled guard ~575) and `goldfive/steerer.py` (`DriftKind.PLAN_DIVERGENCE` ladder entry ~206).
- **Status:** `#252`/`#253` **disabled** the `PLAN_DIVERGENCE` drift emission (replaced by `CAPABILITY_MISMATCH`), but the machinery is a **KEEP branch** — do not delete it. The reconciler still carries the off-plan-agent detection scaffolding and the explicit "disabled here" comment (`goldfive/reconciler.py` ~575: "goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH (#253) — disabled here").
- **Why keep disabled code:** the branch encodes a design option that may be re-enabled; the surrounding logic (off-plan detection, intermediate-coordinator suppression) is still consulted. Deleting the disabled branch would also delete the scaffolding the live `CAPABILITY_MISMATCH` path leans on.

**Grep before touching:** `grep -rn "PLAN_DIVERGENCE" goldfive/reconciler.py goldfive/steerer.py`.

## 2.3 KEEP: `reconciler.get_missed_tasks` (#163)

- **Location:** `goldfive/reconciler.py` ~407, `def get_missed_tasks(self, plan: Plan | None = None) -> list[Task]`.
- **Status:** KEEP (#163). Even if a static call-graph pass shows few or no callers on a given day, this is a deliberately retained reconciliation surface. Do not delete it during a "dead code" sweep. (Contrast with `#490`, which deleted *verified*-dead code — `get_missed_tasks` was explicitly excluded from that class.)

**Grep before touching:** `grep -rn "get_missed_tasks" goldfive/`.

## 2.4 The `agency-preservation` branch boundary (never copy from it)

The branch `agency-preservation` (unmerged; PRs #453–#474) holds Stages 1–3 of the agency-preservation roadmap behind **default-OFF** flags. It is a parallel universe. The rules:

1. **Never copy code from the branch onto `main`.** Main-side code must not import, paste, or re-derive the branch's mechanisms. Doc text on main must not claim the branch's features exist on main.
2. **Expect merge conflicts** in exactly these files when the branch is eventually merged: `goldfive/drift_observer.py`, `goldfive/executors/sequential.py`, `goldfive/steerer.py`, `goldfive/config.py`. These are the four files both the 2026-07 hardening program and the branch heavily edit. Do not pre-resolve those conflicts on main by importing branch shape.
3. **The branch records its own deviations.** `docs/design/AGENCY-PRESERVATION.md` **on the branch** has a §6 "As-built" section documenting what shipped, seven reviewed deviations (each with its PR), what is still default-OFF, and the 13b pre-flip checklist. The **main** copy of that file describes the roadmap and the dormant-supervisor framing but must not assert branch features are live.
4. **Step 13b is LOCKED.** The branch's final step — a three-arm bench run, measurement-gated default flips, and hard deletions — is locked on **explicit user sign-off**. Merging the branch to main is a **separate** user decision. Do not flip any default or delete any protected surface in anticipation of 13b. See memory `project_agency_preservation_roadmap`.

**All branch flags default OFF on the branch:** `plan_mode=forecast`, `signal_channel=legacy`, `observation_only=True`, `signal_telemetry=False`. If you find yourself wanting one of these on main, you are about to violate rule 1.

## 2.5 Bench-frozen / measurement-gated defaults

These `config.py` defaults are **not** free to tune. They were chosen conservatively and are frozen until a measurement gate (13b bench, or a per-subsystem regression harness) justifies moving them. Raising them "because it seems too cautious" is a change of behavior for every operator.

| Default | Value | Where | Why frozen |
| --- | --- | --- | --- |
| `observation_only` | `True` | `SteeringConfig`, `config.py` ~697 | The master kill-switch. Flipping the default to active is a 13b-class decision. |
| `stall_watchdog_enabled` | `False` | `SteeringConfig`, `config.py` ~775 | `#487` shipped the watchdog **flag-gated OFF**. It is the only `TASK_TIMEOUT` producer; enabling it by default changes termination behavior. Leave OFF until measured. |
| `stall_timeout_s` | `600.0` | `config.py` ~782 | Wall-clock budget for the stall watchdog. |
| `pause_escalate_deadline_s` | `None` (feature off) | `config.py` ~760 | `None` means "no operator-set deadline"; the built-in 600s TERMINATE fallback (#482) still bounds the ladder. Do not set a default deadline. |
| `name_axis_max_severity` | `"info"` | `ToolLoopConfig`, `config.py` ~405 | The #484 uncorroborated tool-loop cap. Raising it re-introduces the lexical-escalation risk Invariant 2 forbids. |
| `max_concurrent_judges` | `3` | `SteeringConfig`, `config.py` ~463 | The #483 judge-concurrency cap. Raising it increases endpoint contention (a real cost — Section 4). |
| `GOAL_DRIFT_IDLE_SECONDS` | `300` | `goldfive/drift/goals.py` ~84 | The idle goal-judge trigger threshold consumed by the watchdog and the plugin. |

**Rule:** to move any of these, you need either the 13b bench (for `observation_only` and the watchdog) or a dedicated regression measurement (for the judge/tool-loop knobs), plus explicit sign-off. Cite this table in the PR if you touch a default. See `14-config-reference.md` for the full config surface.

**Knobs you MAY tune (with care, and with a manifest entry).** Not everything is frozen. These are legitimately operator/optimizer-tunable — but they still have behavioral consequences, and if you change a *default* you still update the manifest (Hazard 4.13) and note the rationale:

| Knob | Default | Consequence of raising | Consequence of lowering |
| --- | --- | --- | --- |
| `threshold` (`SteeringConfig`) | `"warning"` | fewer drifts promote to steer (more permissive) | more drifts steer (more aggressive) |
| `suppression_window_turns` | `3` | steers suppressed longer after a user steer | user steers re-drift sooner |
| `max_concurrent_judges` | `3` | more judge parallelism, more endpoint contention (bench-frozen — see above) | more queuing, less contention |
| `GOAL_DRIFT_IDLE_SECONDS` | `300` | idle goal-judge fires later | fires sooner (more judge calls) |
| `LOOPING_REASONING_HASH_WINDOW` | `5` (`drift/reasoning.py`) | wider loop-detection window | tighter |
| `exact_threshold` (`ToolLoopConfig`) | `3` | more exact repeats needed to fire | fewer |

The line between "frozen" (Section 2.5) and "tunable" (here) is: frozen knobs change the *safety posture* (does goldfive intervene at all, does it auto-terminate, does an uncorroborated signal escalate); tunable knobs change *sensitivity within an already-safe posture*. When unsure which side a knob is on, treat it as frozen and ask.

## 2.6 How to delete code safely (the #490 archaeology method)

Deleting "dead" code is allowed and encouraged — but only after **archaeology**, and never for a Section 2.1–2.3 protected surface. `#490` deleted four verified-dead surfaces and is the template. The method, per deletion:

1. **Full-tree grep for consumers.** `grep -rn "<symbol>" goldfive/ tests/`. If any production caller imports it, stop.
2. **`git -S` archaeology on the symbol.** `git log -S'<symbol>' --oneline` to see when it was introduced and whether it was ever wired. `#490` proved `_LADDER_BY_VALUE` was *declared empty and never written* — so the first lookup in `_ladder_level_for` always missed (dead by construction).
3. **Confirm it is not the documented API surface.** If it is exported in `__init__.py` or named in `.agents/*.md`, it is not safe to delete silently.
4. **Confirm it matches a retired direction.** `#490` deleted `detect_unreferenced_keyword` + `Session.unreferenced_keyword_flagged` precisely because they were the *lexical keyword heuristic* unwired in #226/#230 — deleting them *advances* the Invariant-2 direction.

What `#490` deleted (all verified-dead): `DriftObserver._LADDER_BY_VALUE`, `drift.registry.classify()`, `detect_unreferenced_keyword`/`_has_unreferenced_keyword` + `Session.unreferenced_keyword_flagged`, and `PlanReviser.apply_user_steer_with_plan` (a deprecated #271 Option A shim warned since #316). What it **kept** (production-imported, do not confuse with the above): `DetectorConfig`, `truncate_for_observability`, `format_goals_block`, `list_registered`, `_ensure_registered`.

**The trap:** deleting something that *looks* like the dead siblings above but is actually one of the KEEP surfaces (2.1–2.3) or a production import. `get_missed_tasks` looks exactly like dead reconciler scaffolding — it is a KEEP (#163). When archaeology is ambiguous, leave it and ask.

## 2.7 When to stop and ask for explicit sign-off

Some changes are never a solo decision. Use this table: if your change is in the left column, **stop and get explicit human sign-off** before doing it — do not infer permission from the task description.

| Change | Why it needs sign-off | Where it's documented |
| --- | --- | --- |
| Flip `observation_only` default to active. | Changes behavior for every operator; a 13b-class decision. | Section 2.5; `AGENCY-PRESERVATION.md` §6.4. |
| Enable `stall_watchdog_enabled` by default. | Introduces auto-termination; measurement-gated. | Section 2.5 (#487). |
| Raise `name_axis_max_severity` / `max_concurrent_judges` defaults. | Re-opens lexical-escalation / endpoint-contention risk. | Section 2.5. |
| Delete/modify a KEEP surface (`LOOPING_TOOL_CALL`, `PLAN_DIVERGENCE`, `get_missed_tasks`). | Load-bearing history; looks dead but isn't. | Section 2.1–2.3. |
| Merge the `agency-preservation` branch to main. | Separate user decision; carries the four-file conflict. | Section 2.4. |
| Run PR 13b (bench + flips + hard deletions). | LOCKED on explicit sign-off. | Section 2.4; `AGENCY-PRESERVATION.md` §6. |
| Wipe/reset harmonograf data. | NEVER unless explicitly told. | memory `feedback_auto_merge`. |

Everything **not** in this table follows the auto-merge cadence (review→merge→bump) on goldfive/harmonograf without asking (memory `feedback_auto_merge`). The asymmetry is deliberate: routine changes flow, behavior-of-record changes gate.

---

# 3. Deferred-Work Register

These are known-future items. **Do not present them as current** (they are not on `main`), and **do not start them speculatively** — each is blocked on a specific gate. But **do** know they exist, because each implies an **interim rule** for edits you make today. The interim rules are the actionable part for a weaker model.

**Deferred-work at a glance:**

| Item | Blocked on | Your interim rule |
| --- | --- | --- |
| Twin refine-pipeline extraction (3.1) | agency-preservation merge | Edit BOTH `_handle_drift_dispatch` and `_promote_drift_to_steer`. |
| Evidence-ledger replacing the ~7 gates (3.2) | agency-preservation merge | Don't add an 8th gate; make an existing key stable. |
| Judge windowing/cadence expansion (3.3) | a judge regression harness | Don't widen the window; watch the verdict-utility ledger. |
| Judge-facade dispatch authority (3.4) | design decision | Keep judge (verdict) separate from steerer (dispatch). |
| Stage-4 actuators (3.5) | bench + missing infra | Out of scope for main; don't build a partial. |

## 3.1 Twin refine-pipeline extraction (blocked on agency-preservation merge)

**What:** there are currently **two** near-identical refine dispatch pipelines in `goldfive/drift_observer.py`:

- `_handle_drift_dispatch` (~3855) — the primary drift-handling dispatch;
- `_promote_drift_to_steer` (~5457) — the promotion-to-steer path.

Both end by calling `_record_refine_outcome(session, drift, succeeded=...)` (dispatch path at ~4186/4233/4282/4313; promote path at ~5696/5736/5771/5795). The planned extraction folds them into one pipeline, but that refactor is **blocked** on the agency-preservation branch-merge decision (the branch edits the same code heavily; extracting now guarantees a painful conflict).

**Interim rule (ACT ON THIS):** until the extraction lands, **every edit to the refine flow must land in BOTH pipelines.** Before you commit a refine-flow change, run:

```bash
grep -n "_handle_drift_dispatch\|_promote_drift_to_steer\|_record_refine_outcome" goldfive/drift_observer.py
```

Read both methods. If your change touches gate ordering, outcome recording, cancel-inflight behavior, or the escalate-on-exhaustion path, it must be mirrored. A change to only one pipeline is a latent divergence bug. See `09-steering-ladder-and-gates.md` for the gate ordering both share.

**What the two pipelines actually do (so you know what to mirror):**

| Aspect | `_handle_drift_dispatch` (~3855) | `_promote_drift_to_steer` (~5457) |
| --- | --- | --- |
| Role | The ladder dispatch after `handle_drift`'s entry guards (freshness watermark, USER_STEER side effects) already ran; jumps straight into cancel + refine. | Promote a **goldfive-detected** drift into a full steer (the goldfive analogue of USER_STEER). |
| Cancel tagging | `_tag_adapter_cancel_reason(drift, session=...)` — symbolic reason on the adapter's next cancel. | `_tag_adapter_cancel_reason_for_promotion(...)` → `"goldfive_<kind>"`; also sets `session._last_cancel_reason_prefix`. |
| Steer bookkeeping | `_apply_user_steer_state` for `USER_STEER` (synthesize a durable Goal). | Stamp `goldfive.active_steer.*` on `session.state`; record `drift.id` in `goldfive.processed_steer_ids` (retry dedup). |
| Refine entry | `planner.refine` (ladder-driven). | `LLMPlanner.refine_steer` (or `planner.refine` fallback) with `source="goldfive"`. |
| Ordering invariant | dispatch follows `_apply_revision`/`_cancel_inflight_for_revision`/`_emit_plan_revised`. | **Same** — `GOLDFIVE_STEER` dispatch fires AFTER `_emit_plan_revised` swaps the plan (#402 fix); pre-fix it fired before refine and re-invoked stale `replacement_task_ids`. |
| Shared tail | `_record_refine_outcome(session, drift, succeeded=...)` on every exit. | **Same** `_record_refine_outcome` calls. |

Both import `RefineExhausted` and `_planner_refine_accepts_available_agents` from `goldfive.steerer`. Both cancel *the adapter tag + a queued restart*, not a direct `task.cancel()` — the actual cancel is the executor's job (`SequentialExecutor._invoke_with_control`), and `#241` reverted a direct `adapter.request_cancel` that propagated `CancelledError` too aggressively. If your change alters any row above, it changes it in **both** columns.

## 3.2 Evidence-ledger replacement of the stacked `handle_drift` suppression gates (blocked on agency-preservation merge)

**What:** `handle_drift` sits behind roughly seven stacked suppression gates (freshness watermark, in-flight-refine keys, refine-outcome gate, progress-stall gate, late-drift gate, and the observation_only + condition gates). The roadmap replaces this stack with a single **evidence ledger** (the agency-preservation branch's design). **Blocked** on the same branch-merge decision.

**The current gate stack (the ~7 you must not add an 8th to):**

1. **`observation_only` gate** — is active steering permitted at all (`is_active_steering()`).
2. **Freshness watermark** — key `(kind.value, current_task_id)`; drops a drift whose watermark says a fresher one already dispatched (`drift_dropped_stale` outcome, #480).
3. **In-flight refine keys** — key `(session_id, kind.value, current_task_id)`; drops a drift while a refine for the same key is running (`drift_dropped_inflight`/`emitted_redundant`).
4. **Refine-outcome gate** — key `(kind.value, current_task_id)`; consults `session.refine_outcomes` (occurrence count, prior success/failure).
5. **Progress-stall gate** — `PROGRESS_STALL_THRESHOLD_SECONDS`; don't refine again if no progress since the last refine.
6. **Late-drift gate** — the invocation already terminated (#319); the verdict is `emitted_late`, not `acted_on`.
7. **Drift-condition gate** — the lifecycle state (`OPENED`/`ESCALATING`/`RESOLVED`/`HUMAN_INTERVENTION_REQUIRED`) keyed on `sha1(kind, task_id, agent_id, turn_id)`.

**Interim rule:** do **not** add an eighth ad-hoc gate. If you need to suppress a new class of spurious drift, first check whether an existing gate's key can be made stable (Invariant 6) rather than stacking another gate. If you genuinely must add suppression, key it stably and document it next to the others so the eventual ledger migration can see it. Grep the gate cluster: `grep -n "fresh\|in.flight\|refine_outcome\|late.drift\|_should_" goldfive/drift_observer.py`.

## 3.3 Judge windowing / cadence expansion (blocked on a judge regression harness)

**What:** widening the reasoning-judge's observation window and firing cadence. **Blocked** until a judge **regression harness** exists to prove a wider window doesn't regress precision. `#483` shipped the scheduling *guards* (per-steerer semaphore default 3, queued-window coalescing, the verdict-utility ledger `{acted_on, emitted_late, emitted_redundant, parse_fail}`) — those are the *instrumentation* the future expansion will be measured against, not the expansion itself.

**Interim rule:** do not widen the window or raise the cadence "to catch more." Use the verdict-utility ledger (`_verdict_ledger`, `goldfive/drift_observer.py` ~2383) as the measurement surface: if `emitted_redundant` / `emitted_late` dominate `acted_on`, the current cadence is already over-firing. See `08-llm-judges.md`.

**The judge-scheduling guards you must respect** (#483 — these are the instrumentation the expansion will be measured against; do not weaken them):

| Guard | What it does | Where |
| --- | --- | --- |
| Per-steerer semaphore (default 3) | Caps concurrent judge calls; a request over the cap is QUEUED, not dropped. | `self._judge_semaphore = asyncio.Semaphore(max(1, _judge_limit))`, `drift_observer.py` ~434 |
| Queued-window coalescing | A newer observation folds into an already-queued window instead of firing a second call. | `_run_judge_window` region, `drift_observer.py` ~1967 |
| Verdict-utility ledger | Per-session `{acted_on, emitted_late, emitted_redundant, parse_fail, elapsed_ms}`; teardown emits a `reasoning_judge_utility_summary` dict envelope (no proto change) with p50/p95 latency. | `_verdict_ledger` / `_emit_verdict_utility_summary`, `drift_observer.py` ~2383 |
| Endpoint-contention warning | Warns when judge concurrency saturates the endpoint. | `drift_observer.py` |

The ledger's four counters are the honest accounting of whether judges earn their cost: `acted_on` (dispatched past the late gate), `emitted_late` (invocation already terminated, #319), `emitted_redundant` (hit the addressed-watermark / in-flight-refine gates), `parse_fail` (empty classification sentinel). Quiet runs (no judge activity) create no ledger and emit no summary — the pop is idempotent. **Mistake:** raising `max_concurrent_judges` to "reduce queuing" — that trades queuing for endpoint contention (a real cost, Hazard 4.5), and the contention warning exists precisely to catch it.

## 3.4 Judge-facade dispatch authority (deferred)

**What:** giving a judge facade authority to dispatch interventions directly. **Deferred** — judges today produce verdicts; the steerer owns dispatch. **Interim rule:** keep the judge/steerer separation strict. A `Judge` (`goldfive/protocols.py`, the pluggable protocol from #439) returns a typed `JudgeVerdict` (enum-typed `drift_kind`/`severity`, #443) and emits a `JudgementEmitted` event; it does **not** call cancel/nudge/plan-install. The steerer consumes the verdict and decides the intervention (and gates it on `observation_only`). A weak model tempted to "let the judge just cancel the run when it's confident" is collapsing this separation — the judge is fallible (that's why steering is the conditional half, Section 5.9), so its output is *advice to the steerer*, never a direct actuator. This separation also makes judge-only mode possible: the default steerer retains detector and event handling while its drift-response dispatch is disabled.

## 3.5 Stage-4 actuators: checkpoint-rollback / tool-gating hold / fork-and-judge (bench-gated, unbuilt)

**What:** the ambitious intervention surfaces — rolling back to a checkpoint, holding a tool call pending judgment, forking a run and judging both. These are **Stage-4** on the agency-preservation roadmap, **unbuilt** and **bench-gated**. `docs/design/AGENCY-PRESERVATION.md` (main) §5 and the branch's §6.4 describe them as exploratory.

**Interim rule:** do not build a partial version of these on main. They require checkpoint/fork infrastructure that does not exist and a bench that has not run. If a task seems to want one, it is out of scope for `main`.

**The correctness discipline these deferred items will follow** (`AGENCY-PRESERVATION.md` §5, on main as design text) — apply these to *any* risky change you make, not just the deferred ones:

1. **No-op by default; one-line revertible flips** (§5.1) — a new capability ships behind a default-OFF flag whose flip is one line. This is why every #453–#474 branch flag defaults OFF, and why #487's watchdog is flag-gated.
2. **Invariant contract tests written before the code** (§5.2) — write the both-modes / stable-key / no-hang test first, so the test fails before your change and passes after.
3. **Decision-table snapshot tests** (§5.3) — for ladder/severity logic, snapshot the `(kind, severity, occurrence) → action` table so an accidental routing change is caught.
4. **Shadow / differential validation before authority** (§5.4) — a new intervention runs in shadow (observe-only) and is compared against the incumbent before it gets authority to act. This is the `observation_only` philosophy generalized.
5. **Golden traces + property-based interleaving tests** (§5.5) — for concurrency (the stash race, the freshness-gate atomicity), test interleavings, not just the happy path.
6. **Integration disciplines** (§5.6) — grep for real call sites (memory `feedback_integration_not_unit`); a helper that passes unit tests can be dead code.
7. **Layered e2e with functional pass criteria** (§5.7) — DB-only checks miss UI/health regressions (memory `feedback_e2e_validation_layered`).
8. **Bench-gated flips** (§5.8) — the default flip itself is gated on a measured bench (PR 13b), not on "it looks better."

**The intervention-surface hierarchy (branch design, for context only).** The branch's §5 orders interventions from lightest to heaviest: (0) observe/emit; (1) **observer-note** — an honestly-attributed advisory note the agent may read (the branch's `signal_channel`); (2) plan-as-ledger refine (goal-anchored OUTCOME tasks); (3) nudge; (4) cancel-reinvoke; (5) pause/terminate; and the unbuilt Stage-4 actuators above. The design intent is to make the *common* correction a light observer-note rather than a wheel-grab plan-swap. **None of levels 1–2's ledger/observer-note machinery exists on main** — do not reference it in main-side code or claim it. On main the ladder is still NUDGE→CANCEL_REINVOKE→PAUSE_ESCALATE→TERMINATE (Section 1.0).

## 3.6 Decision-telemetry reference (what zicato reads; keep it honest)

Not deferred, but adjacent: the decision telemetry the meta-loop consumes. `SteeringDecisionMade` (`events_pb2.pyi` ~527) carries `detector_name, outcome, reason, score, considered_severity, chosen_severity, considered_intervention_level, chosen_intervention_level, drift_id, decided_at, invocation_id, task_id, agent_name`. The **`outcome`** vocabulary you must use consistently (a wrong label is a #480-class corruption):

| `outcome` | Meaning |
| --- | --- |
| (positive fire) | a drift was detected and dispatched. |
| `no_drift` | the detector ran and found nothing — the **negative class**, emitted via `emit_no_drift_decision` (`drift_observer.py` ~701). Do not skip it. |
| `drift_dropped_stale` | dropped by the freshness watermark. |
| `drift_dropped_inflight` | dropped because a refine for the same key was in flight. |

`detector_name` pairs a decision to its detector; `_detector_name_for_drift` (`drift_observer.py` ~629) resolves it (a `detector_name` stamped on the drift wins; else the kind maps, e.g. `CAPABILITY_MISMATCH → "capability_check"`). **Rule:** every detector emits both its positive fires AND its `no_drift` decisions, with the correct `detector_name`; otherwise zicato computes precision on a truncated denominator. Related decision events: `LadderTransitionDecided`, `DetectorDispatchOrdered`, `PolicyApplied`, `RetryBudgetSpent`, `JudgementEmitted` (#439). See `12-events-sinks-telemetry.md`.

---

# 4. Hazard Catalog

These are the recurring do-no-harm lessons, phrased as **permanent tests every new change must pass**. Unlike the invariants (which forbid specific things), these are traps: patterns that look correct and are not. For each: the trap, the rule, and how to check.

## 4.1 Every intervention surface must be gated

**Trap:** you add a code path that touches the agent (cancel, inject, mutate, refuse, install) and gate it in the *unit test* but forget the runtime gate — or you gate it but the fallback direction is ACTIVE.

**Rule:** an intervention surface is any code that: cancels an invocation, injects/rewrites a message, mutates ADK or session control state, refuses a dispatch, installs a plan, or emits a directive ack carrying goldfive-authored state. Each one gates on `steering_is_active(steerer)` / `is_active_steering()`, failing PASSIVE. This is Invariant 5 as a hazard: the four leaks (#475/#476/#478/#481) were all "surface added, gate missing or mis-directed."

**Check:** for every new surface, write two tests — passive asserts nothing happens, active asserts it fires. Grep the surface for the predicate. See Section 6, item 2.

## 4.2 Injected text must be factually true

**Trap:** a nudge or corrective message asserts something about the agent's state that isn't true ("you have not made progress", "you called tool X three times") based on a stale or wrong read — the agent then acts on a lie.

**Rule:** any text goldfive injects into the agent's context must be **factually accurate at emit time**. `#475` explicitly paired the nudge-path gate with **truthful nudge text** — the two were fixed together because an ungated nudge and a false nudge are the same class of harm. If you can't cheaply verify the claim, don't make it; emit a neutral refocus, not a false accusation.

**Check:** read the exact string your surface injects. Does every factual claim in it come from state you have verified this turn? If it interpolates a count or a task name, is that value current? Cross-reference `09-steering-ladder-and-gates.md` (corrective-message composition).

## 4.3 Nothing may hang unboundedly

**Trap:** you add a wait — on a human, on a control channel, on a judge, on a lock — with no finite deadline. Under the wrong conditions (no operator, no channel, a wedged judge) the run hangs forever.

**Rule:** every wait has a finite, bounded deadline **and** a defined behavior on expiry.

- `report_awaiting_approval` (#478): no channel → **immediate** `'unavailable'` ack (never blocks); finite default timeout **600s**; on expiry emits `HUMAN_INTERVENTION_REQUIRED` (`goldfive/reporting/handlers.py` ~793, ~835).
- The escalation ladder (#482): `pause_escalate_deadline_s`, or a built-in **600s** TERMINATE fallback, bounds PAUSE_ESCALATE so it terminates rather than blocking forever.
- The stall watchdog (#487): wall-clock `stall_timeout_s` (600s) is the `TASK_TIMEOUT` producer, stamped against `Session.last_observed_event_at`.
- goldfive-internal LLM calls (#298 lineage, now `_llm.py`): bounded max_output_tokens + timeouts so a runaway model can't wedge the observer (memory `feedback_dont_blame_llm_for_slowness`).

The `report_awaiting_approval` no-channel path is the canonical shape — return immediately rather than register a waiter (`goldfive/reporting/handlers.py`):

```python
# reporting/handlers.py — no controller attached: never block.
if getattr(steerer, "_control_channel", _UNKNOWN_CHANNEL) is None:
    log.warning(
        "report_awaiting_approval(task_id=%r): no control channel "
        "attached to this run; returning decision='unavailable' "
        "without blocking",
        task_id,
    )
    return {
        "acknowledged": True,
        "decision": "unavailable",
        "detail": (
            "no approval controller is attached to this run; the "
            "request cannot be received or answered by a human"
        ),
    }
```

(Note: this reads `_control_channel` with a `_UNKNOWN_CHANNEL` sentinel, not `None`, so a steerer that *doesn't expose the attribute at all* falls through to the finite-default wait rather than being mistaken for "no channel." That distinction — missing-attr vs explicit-None — is deliberate; don't collapse it.)

**Check:** every `await` on an external actor (human, channel, judge, lock) needs a timeout. Grep: `grep -rn "wait_for\|timeout\|deadline" goldfive/reporting/ goldfive/executors/ goldfive/drift_observer.py`. If you add a block with no timeout, you have a hang bug.

## 4.4 Nothing may cancel healthy work in passive mode (or on a weak signal)

**Trap:** a timeout, a judge parse failure, or a malformed verdict is treated as a reason to cancel — cancelling healthy work.

**Rule:** in `observation_only` (the default), **no** path cancels. Even in active mode, weak/failed signals downgrade to INFO rather than escalate:

- `#476`: `LLM_CALL_TIMEOUT` no longer cancels under `observation_only`; the reasoning channel disarm emits **one** loud warning, not a cancel.
- `#479`: a **malformed judge severity → INFO** (not an escalation); sink exceptions never abort runs; the embedding breaker recovers via half-open probe rather than dying.
- `#484`: uncorroborated tool-loop name-axis capped at INFO.

The principle (memory `feedback_empirical_baselines`): if a failure correlates with a STEER/CANCEL/PAUSE, run the same driver **without** steering before blaming the model — the cancel is often the cause, not the cure.

**Check:** trace every `cancel` / `abort` / escalation to its trigger. Is the trigger a *confirmed drift on healthy-work evidence*, or is it an *infrastructure hiccup* (timeout, parse fail, sink error)? Infrastructure hiccups must degrade to observation. Tests: `tests/test_cancel_reason.py`, and the judge-hardening tests for #479.

## 4.5 Tool-surface and endpoint costs are real deltas

**Trap:** you add a judge call, a nudge, a report tool, or an embedding call and reason "it's just one more call." Across a run at scale these compound into endpoint contention and cost.

**Rule:** treat added LLM calls, added tools exposed to the agent, and added embedding calls as **real deltas** to measure, not free. `#483` added the per-steerer judge **semaphore (default 3)**, **queued-window coalescing** (fold a newer observation into a queued window instead of firing twice), the verdict-utility ledger, and an **endpoint-contention warning**. Adding a tool to the agent's surface also changes the agent's behavior (more choices) — that is itself a behavioral delta, not just a cost.

**Check:** if your change adds an LLM/embedding call per drift/turn/tool, ask: is it coalesced with existing calls? Is it under the semaphore? Does the verdict-utility ledger show it earning its cost (`acted_on` vs `emitted_redundant`)? See `08-llm-judges.md` and `12-events-sinks-telemetry.md`.

**Concrete wrong edit:** "I'll add a quick confirmation judge call on every tool result to catch errors early." Wrong on two counts: (1) a judge *per tool result* is a large endpoint-cost delta at scale — it must go through the semaphore + coalescing, and the verdict-utility ledger will likely show it mostly `emitted_redundant`; (2) it may also add a *tool* the agent sees, changing the agent's own behavior. Correct: use the existing `TOOL_ERROR` builtin judge on the existing cadence, or a deterministic detector — not a new per-result LLM call. The embedding backend has its own cost guard: a circuit breaker (`goldfive/drift/_embed.py`) that trips on repeated backend failure and recovers via a timed half-open probe (#479) — do not disable it to "get more signal"; a tripped breaker is protecting you from wasted I/O.

## 4.6 Duck-typing / ADK-version fragility

**Trap:** you `getattr` into an ADK object assuming a field/method exists, or you assume the ADK session object round-trips writes — and a version bump or a shallow-copy silently breaks it.

**Rules:**

- **Shallow-copy state handoffs.** ADK (and ADK-family SDKs) can return **shallow copies** of `session.state`; a write on one side may be invisible to the callback side. Memory `feedback_callback_context_handoff` records this costing ~8h of wrong fixes. When handing state across a plugin callback boundary, verify the read side actually sees the write — do not assume the object is shared.
- **Duck-typed predicates fail PASSIVE.** `steering_is_active` uses `getattr(steerer, "is_active_steering", None)` and treats missing/raising as `False`. This is the correct pattern for optional-capability probing on maybe-objects: probe, and default to the **safe** direction when the capability is absent.
- **`getattr` into ADK objects needs a default and a reason.** The `GOAL_DRIFT_IDLE_SECONDS` live read (`goldfive/adapters/_adk_plugin.py` ~5413) uses `max(0.001, float(getattr(_goals, "GOAL_DRIFT_IDLE_SECONDS", 300)))` — a default plus a floor. Follow that shape.
- **Provider-family gating.** `#491` scoped Qwen `/no_think` + `enable_thinking` to the **Qwen/litellm family only** via `THINKING_DISABLE_CAPABILITIES` (`goldfive/_llm.py` ~247), matched by lowercase substring. Do not apply provider-specific thinking-control to all models; match the capability table. The lookup normalizes litellm-prefixed names so `"openai/Qwen3-32B"` and `"hosted_vllm/Qwen/Qwen3-32B"` route to the same family, and unknown models get **empty caps** — no vendor hacks:

```python
# goldfive/_llm.py — provider-family capability lookup, not a blanket switch.
THINKING_DISABLE_CAPABILITIES: tuple[tuple[str, ThinkingDisableCaps], ...] = (
    ("qwen", ThinkingDisableCaps(openai_enable_thinking_field=True, no_think_prompt_prefix=True)),
)

def thinking_disable_caps(model_name: str) -> ThinkingDisableCaps:
    lowered = (model_name or "").lower()
    for marker, caps in THINKING_DISABLE_CAPABILITIES:
        if marker in lowered:
            return caps
    return _NO_VENDOR_THINKING_CAPS  # unknown model → no vendor hacks
```

  (If you need Claude/Anthropic-family behavior here, add a capability row — do **not** special-case a model id inline. The provider abstraction is deliberate.)

**Check:** grep `getattr(` in adapter/plugin code and confirm each has a sensible default. Verify state-handoff reads in `tests/test_adk_plugin_*`. Cross-reference `05-adk-plugin.md` and `06-adapters-and-instrumentation.md`.

## 4.7 Unstable-key gates (Invariant 6 as a hazard)

**Trap:** a dict keyed on something that churns. Covered as Invariant 6 — repeated here because it is the single most common *silent* failure. The gate compiles, tests that don't check collapse pass, and in production the gate never engages. Correction keys on the **full agent path** (#479), suppression on a **logical-turn counter** (#442), conditions on a **content hash of stable fields**. Never `drift.id`.

**Check:** for every dict/set gate, ask "what is the key, and can any component of it change between two observations that should collapse?" Write the two-observation-collapse test.

## 4.8 Last-writer-wins globals

**Trap:** module-global mutable state (a default flag, a closure attribute, a monkeypatch hook) that the last writer clobbers — especially across concurrent sessions or test files.

**Rule:** prefer `ContextVar`-scoped per-call state over module globals. `#491` replaced the closure-attribute diagnostics side channel with a per-call `LlmCallDiagnostics` bound to `LLM_CALL_DIAGNOSTICS_VAR` (a `ContextVar`), and per-callsite caps live in `MAX_OUTPUT_TOKENS_VAR` / `THINKING_DISABLED_VAR` — all `ContextVar`, not globals. `#488` **deleted** the module-global test hook (`_OBSERVATION_ONLY_DEFAULT` + `_resolve_observation_only_default`) and the autouse fixture that mutated global default state, because they made the whole suite lie about the shipped default. Concurrent goldfive sessions must never share one mutable gate (`goldfive/drift_observer.py` ~423: sized/scoped so "processes never share one gate").

**Check:** any new module-level mutable (`_FOO: dict = {}`, a flag you flip) is suspect under concurrency. Can two sessions race it? If yes, scope it per-session or per-call via `ContextVar`. Grep: `grep -rn "^_[A-Z].*= \|ContextVar" goldfive/_llm.py goldfive/drift_observer.py`. Also memory `feedback_reader_centric_versioning` for the multi-observer version-compare trap.

## 4.9 Verify the running build before diagnosing

**Trap:** a symptom appears after a recent change; you assume the recent change caused it and start editing — but the running process predates the merge, or the log you're reading is from a stale build.

**Rule (memory `feedback_verify_running_build`):** when a symptom follows a change, first check what is *actually running* (process start time vs merge time, `lsof` on the log). And (memory `feedback_external_evidence_trumps_logs`): when GPU fans / model-server logs / harmonograf state contradict your "no activity" reading, dig deeper — `ss` showing `0 0` queues is not proof of idle.

**Check:** before diagnosing a regression, confirm the deployed build includes the suspect commit. E2E validation must be layered (memory `feedback_e2e_validation_layered`): DB-only checks miss UI and slow-burn health regressions.

## 4.10 Concurrent-agent worktree safety

**Trap:** you run a driver/agent from the main `~/git/goldfive` checkout while another agent is working there — corrupting each other's tree.

**Rule (memory `workflow_goldfive_concurrent_agents`):** the main checkout is unsafe when other agents run. Use an isolated `/tmp` worktree for concurrent work. This chapter is read-only ground truth; if you need to *run* something, do it in a worktree.

## 4.11 Proto/telemetry changes are additive-only; field numbers are forever

**Trap:** you "clean up" a proto by renumbering fields, reusing a deleted field number, or reordering — and every existing wire consumer (harmonograf, zicato) misreads old data.

**Rule:** proto changes are **additive**. New fields get the **next** number; you never reuse or renumber. `#480` added `ReasoningJudgeInvoked` fields **12–15** (`focused_task_id`, `focus_confidence`, `stated_intent`, `provenance`) — appended, not inserted. `#318` added drift-as-condition fields additively. `event_id` (#289) is Phase 3 *Addition* B for the same reason. When you extend an event, append.

**Check:** `git diff proto/goldfive/v1/*.proto` — every changed line should be an addition at the end of a message, never a renumber. Regenerate the `_pb2`/`_pb2.pyi` stubs; do not hand-edit them.

## 4.12 `event_id` must stay globally unique on the wire

**Trap:** you emit an event without threading `Session.next_event_id`, or you reuse an id, and multi-observer fan-in (harmonograf) collapses distinct events.

**Rule:** every emitted envelope carries a globally-unique `event_id`. The preferred value is minted via `Session.next_event_id` (so `(run_id, sequence)` is monotone); the fallback is `f"{run_id}:{sequence}:{seeded_uuid4().hex[:8]}"` (`goldfive/events.py` ~100) so the id is always non-empty AND unique even for un-sequenced emits. Do not emit an event with an empty or hand-built id.

**Check:** `grep -rn "event_id" goldfive/events.py`; any new emit path should route through `emit` (which stamps it), not construct envelopes by hand.

## 4.13 The sink JSONL contract is camelCase + deterministic (the zicato surface)

**Trap:** you add a field to the JSONL sink in snake_case, or you introduce nondeterminism (unsorted dict, wall-clock in a "determinism" path), and zicato's offline meta-loop reads garbage.

**Rule:** goldfive's JSONL sink emits **camelCase** (memory `project_zicato_optimization_surface`). Determinism is a contract (`#436` shipped determinism guarantees + a testkit). zicato reads the telemetry + `optimization/manifest.toml`; a nondeterministic or mis-cased field breaks the meta-loop silently. When you add a tunable knob, add its `optimization/manifest.toml` entry (`source = "file:SYMBOL"`, `python_attr = "module:SYMBOL"`) — `#487` even ships an **AST-based manifest-liveness test** that fails if a manifest entry points at a symbol that no longer exists.

**Check:** `grep -rn "GOAL_DRIFT_IDLE_SECONDS\|<your knob>" goldfive/optimization/manifest.toml`; run the determinism test (`tests/test_determinism.py`) and the manifest-liveness test after adding/renaming a knob.

## 4.14 `CancellationStashViolation` is a `BaseException` — don't swallow it

**Trap:** you wrap a block in `except Exception:` and accidentally swallow the cancellation-stash tripwire, hiding a real state-corruption bug.

**Rule:** `CancellationStashViolation` (`goldfive/_state_audit.py` ~106) is deliberately a `BaseException`, not an `Exception`, so a bare `except Exception:` does **not** catch it. Do not "fix" it to subclass `Exception`, and do not add a `except BaseException:` that would swallow it. It must propagate to surface the violation.

**Check:** `grep -rn "except BaseException" goldfive/` — should be rare and deliberate. The tripwire fires when a cancelled block exits without stashing its prior plan (the #287 invariant).

## 4.15 Terminal-status handling must use the canonical set (incl. `NOT_NEEDED`)

**Trap:** you hand-write a terminal-status check like `status in {COMPLETED, FAILED, CANCELLED}` and forget `NOT_NEEDED`, so the scheduler re-walks a task the planner already marked optional/superseded.

**Rule:** use the canonical `TERMINAL_TASK_STATUSES` (`goldfive/types.py` ~106 = `{COMPLETED, FAILED, CANCELLED, NOT_NEEDED}`) and the shared executor helpers (#485). The parallel scheduler skips terminal tasks including `NOT_NEEDED`. Never inline your own terminal set.

**Check:** `grep -rn "COMPLETED.*FAILED.*CANCELLED" goldfive/` — any inline set that omits `NOT_NEEDED` is a bug; replace with `TERMINAL_TASK_STATUSES`.

## 4.16 The stall watchdog is the only `TASK_TIMEOUT` producer — and it is OFF by default

**Trap:** you assume a hung task will eventually time out on its own, or you add a *second* timeout producer that races the watchdog.

**Rule:** the flag-gated wall-clock **stall watchdog** (#487) is the single `TASK_TIMEOUT` producer. It is `stall_watchdog_enabled=False` by default (Section 2.5) — so on the shipped config, **nothing** auto-times-out a wedged task; termination comes from generator-end, drift detectors, or the escalation-ladder terminus (#482). The watchdog stamps `Session.last_observed_event_at` on every observed event as a liveness watermark; if the watermark goes silent for `stall_timeout_s` (600s) it fires. It also triggers the idle goal-judge (consuming `GOAL_DRIFT_IDLE_SECONDS`, 300s). Do not add a competing timeout; if you need timeout behavior, enable the watchdog (a Section 2.5 sign-off decision), don't reinvent it.

**Check:** `grep -rn "TASK_TIMEOUT\|last_observed_event_at\|stall_watchdog" goldfive/` — there should be exactly one producer.

## 4.17 The reporting tools are optional augmentation, never a dependency

**Trap:** you make an observability or control feature *depend* on the agent calling one of the eight reporting tools (`report_task_started`, `report_task_progress`, `report_task_completed`, `report_task_failed`, `report_task_blocked`, `report_new_work_discovered`, `report_plan_divergence`, `report_awaiting_approval`; `goldfive/reporting/handlers.py`).

**Rule (Invariant 1 corollary):** the reporting tools **augment** goldfive's observability when the agent chooses to call them, but nothing may **require** them. Task lifecycle, drift detection, and termination must all work for an agent that never calls a single `report_*` tool. `report_awaiting_approval` is the sharpest example: it participates in the approval flow **when called**, but its no-channel path returns `unavailable` immediately (#478) precisely so a run without an approval controller doesn't hang — the tool is a cooperative convenience, not a gate. When you add a feature, ask: "does this break if the agent never calls a `report_*` tool?" If yes, you have built a cooperation contract.

**The two approval flows** (both resolve via `session.pending_approvals` + `APPROVE`/`REJECT` control messages): **Flow A** — a task-level `report_awaiting_approval` waiter (agent-initiated); **Flow B** — an ADK `require_confirmation=True` tool-level waiter (framework-initiated). Both must not hang (Hazard 4.3); both emit `ApprovalGranted`/`ApprovalRejected`. Do not collapse the two flows — they have different initiators and different waiter registration.

**Check:** `grep -rn "report_task_started\|report_awaiting_approval" goldfive/executors/ goldfive/runner.py` — the executor/runner must not *require* a `report_*` call to advance lifecycle.

## 4.18 Diagnostic table: symptom → likely goldfive cause (not the model)

When a run misbehaves, the reflex "the LLM did something weird" is usually wrong (memory `feedback_dont_blame_llm_for_slowness`, `feedback_empirical_baselines`). Map the symptom to its likely goldfive-side cause first.

| Symptom | Likely goldfive cause (check first) | The lesson |
| --- | --- | --- |
| 5+ min per turn. | Unbounded `max_tokens`, a drift loop re-refining, a missing wall-clock budget. | `feedback_dont_blame_llm_for_slowness`; check `_llm.py` caps. |
| Run "hangs" doing nothing. | An unbounded wait (approval / channel / lock) with no deadline. | Hazard 4.3; grep `wait_for`/`Event().wait`. |
| Healthy run gets cancelled. | A timeout / parse-fail / sink error routed to cancel; or an active-mode escalation on a weak signal. | Hazard 4.4; run the same driver WITHOUT steering (`feedback_empirical_baselines`). |
| A gate "should suppress" but duplicates fire. | Unstable key (`drift.id` / churning id) → gate never engages. | Inv 6; check the key. |
| A guard/helper "passes tests" but has no effect. | It's dead code — never wired into the real dispatch path. | `feedback_integration_not_unit`; grep call sites. |
| A callback reads state that another side wrote — and sees nothing. | ADK shallow-copy of `session.state`. | Hazard 4.6; `feedback_callback_context_handoff`. |
| harmonograf shows spurious "interventions." | A framework-authored drift not marked synthetic (#302). | Section 5.11. |
| zicato optimizes the wrong knob / can't compute precision. | Mislabeled decision telemetry, or a missing `no_drift` negative class. | Section 3.6 (#480). |
| Symptom appeared "after my change" but change looks innocent. | The running build predates the merge; you're reading stale logs. | Hazard 4.9; check process start vs merge time. |
| "No activity" but GPU fans spin / model-server logs move. | Your idle read is wrong; `ss` `0 0` queues ≠ idle. | `feedback_external_evidence_trumps_logs`. |
| A `USER_STEER`/`CANCEL`/`PAUSE` correlates with the failure. | The intervention is the cause, not the cure — baseline without it. | `feedback_empirical_baselines`. |

**Method:** reproduce with steering OFF (or `observation_only=True`) first. If the symptom vanishes, it is a goldfive intervention, not the model. If it persists, then investigate the agent/tree. Do not skip the baseline.

---

# 5. History — the load-bearing arcs

Weaker models re-derive intent from scratch and get it wrong. Here is the curated timeline of the arcs that shaped the code, so you can look up "why" instead of guessing. Each arc: what it was, what it changed, and the residue you'll still see in the code.

## 5.1 Single-Runner revert (goldfive#128 family)

**What:** an earlier design (#120) introduced N runners with a registry-dispatch model. It was rolled back (2026-04-20, memory `project_single_runner_revert`). Termination moved to **generator-end + drift detectors**, while keeping #120's state-protocol, reporting-tool augmentation, and sink events.

**Residue:** the single-`Runner` shape you see today (`goldfive/runner.py`), with termination driven structurally rather than by a runner registry. `#489` extracted `Runner._abort_turn` (the shared abort tail repeated at 8 call sites in `_run_locked` — grep `_abort_turn` in `runner.py`); that consolidation is the modern shape of the single-runner's failure handling. If you find references to multi-runner dispatch in old docs, the code won. **Lesson for edits:** termination is *emergent* (generator end + detectors + ladder terminus), not *dispatched* — do not add a "runner registry" or a central kill authority; that pattern was reverted for cause. See `03-runner-and-conversation.md` and `docs/design/SHARED-RUNNER-REFACTOR.md`.

## 5.2 Overlay refactor (goldfive#141–144)

**What:** replaced per-task driving with an **observation-driven overlay** + soft follow-up + intervention ladder (2026-04-21, memory `project_overlay_refactor_plan`). Fixed a coordinator-flow-looping regression.

**Residue:** the overlay loop in `goldfive/executors/sequential.py` (`_run_overlay`), which *observes* and *drains* rather than *drives*. `#489` later decomposed that ~620-line loop into named stage methods (`_clear_stale_supersede`, `_race_control`, `_handle_invoke_cancelled`, `_restart_after_user_steer`, `_restart_after_goldfive_steer`, `_handle_goldfive_pause`, `_drain_nudges`, `_sweep_unreachable_pending`, `_classify_fatal_failure`, `_abort_overlay`) with per-turn state in `_OverlayTurnState` — behavior-preserving. See `04-executors-and-control.md`.

**Editing-discipline note (#489 decomposition):** the stage methods are a *behavior-preserving* extraction — the loop skeleton lives in `_run_overlay` and per-turn state threads through `_OverlayTurnState`. Two traps: (1) the `#489` commit moved code **verbatim** and preserved the issue-number comments next to the logic they explain — do not "tidy" those comments away; they are the breadcrumbs (Section 5.12). (2) State that must survive across stages goes on `_OverlayTurnState`, not on a local in one stage — a value set in `_race_control` and read in `_drain_nudges` must be a field on the dataclass. The `_drain_nudges` stage is where the gated nudge injection actually reaches the tree (Section 1.5.1); it carries a defense-in-depth `observation_only` gate (#264) for steerer subclasses that bypass the dispatcher, so do not remove that check as "redundant."

## 5.3 Structural steering (goldfive#151–155)

**What:** tree-aware planner, orchestration state, `GoldfivePlanner(BasePlanner)`, goal-aware refine, and a proto fix for per-event `session_id` (2026-04-21, memory `project_structural_steering_plan`). **NO cooldown**, per explicit user directive.

**Residue:** the planner architecture in `goldfive/planner.py` / `goldfive/planners/`, and the absence of any cooldown timer in the steering path — if you're tempted to add a cooldown, don't; it was directively excluded. **Why no cooldown matters:** a weak model debugging "goldfive re-refines too fast" will reach for a time-based cooldown. That was explicitly excluded by user directive. The *correct* rate control is the **structural** gate stack (Section 3.2: freshness watermark, in-flight keys, refine-outcome, progress-stall) — event-driven suppression keyed on stable identity, not a wall-clock timer. If refines are too frequent, tighten a gate's engagement, don't add a timer. See `10-planning-and-revision.md`.

## 5.4 The #271 intent program (drift-as-stateful-condition + strict boundaries)

**What:** the largest single arc. `#271` reframed drift as a **stateful condition** with a lifecycle, and hardened every control boundary so enforcement doesn't depend on agent cooperation (Invariant 1). Fully validated 2026-04-25 (memory `project_271_intent_validated`): Phase 0/1/2.X + Gap 1 v2 (try/finally stash) + Gap 2 v2 (Conversation cursor), via an 8-invariant validation.

**Residue everywhere.** The #271 program is the single richest source of "why is this here?" answers. A curated map from the git log — when you find one of these mechanisms and wonder why, this is the reason:

| PR | Mechanism it created (the residue you see today) |
| --- | --- |
| #286 | Preserve goal qualifications across user steers (`planner`, `steerer`). |
| #287 | Stash prior plan in a `finally` so `CancelledError` can't skip the stash (Gap 1 v2). |
| #288 | Cross-turn wire **sequence cursor** on the Conversation (Gap 2 v2). |
| #289 | Globally-unique `event_id` (`{run_id}:{sequence}:{uuid8}` via `Session.next_event_id`), Phase 3 Addition B. |
| #291 | Collapse `planner_gate` into `Planner.handle_turn`. |
| #295 | Preserve qualifications + valid revision shape in `handle_turn`. |
| #297 | Serialize per-key `Runner.run` so concurrent same-session turns don't race the prior-plan stash. |
| #298 | Bound goldfive-internal LLM calls + escalate runaway drift loops. |
| #299 | Sticky cancel gate on `before_model_callback`. |
| #301 / #304 | Scope `ADKAdapter` cancel-reason / session-id / pending-tool state **per goldfive session** (not global). |
| #302 | Mark `Runner._install_revision` `USER_STEER` drift **synthetic** so harmonograf filters it from interventions. |
| #303 | Cancel the in-flight invocation task on refine. |
| #306 / #307 / #310 | Cancellation-stash conversions + goldfive-owned invocation-boundary wrapper + `CancellationStashViolation` tripwire. |
| #311 / #312 / #313 | Qwen thinking-model token budgets; disable-thinking on internal calls; forbid meta-commentary in the plan summary. |
| #314 | Thread `agent_name` through reasoning-judge dispatch. |
| #315 | Close open invocation boundaries on terminal drifts. |
| #316 / #317 | Decouple plan installs from drift events (Option A); replace the plan-revision count cap with **structural** guarantees. |
| #318 | Drift-as-stateful-condition fields on `DriftDetected` (additive). |
| #391 / #392 / #393 / #394 | Gate supersedes-integration on `dry_run`; require user-input grounding for justified-deviation; `capability_check` Rule C; strict-passive (strip goldfive's own helper prompt directives). |

`RefineExhausted` (`goldfive/steerer.py` ~124) is the #271 escalation primitive: a refine handler that cannot produce a meaningful change raises it, and the steerer catches it and emits `HUMAN_INTERVENTION_REQUIRED`, pausing the run for operator action. Most planners never raise it explicitly — the steerer's structural no-op-revision check catches the exhaustion for them. The `CancellationStashViolation` tripwire (`goldfive/_state_audit.py` ~106, a `BaseException` so it can't be swallowed by a bare `except Exception`) is the enforcement that a cancelled turn always stashes its prior plan — the invariant #287 established, now guarded at runtime (#326).

## 5.5 The steerer split (buckets 1–3, 2026-05)

**What:** `DefaultSteerer` was decomposed. `TaskStateMachine` (bucket 1, commit `258c810`), `PlanReviser` (bucket 2, `8c07649`), and `DriftObserver` (buckets 3a/3b/3c, `0214ae8`/`52f732f`/`92b2293`) were extracted out of the monolithic steerer.

**Residue:** `goldfive/task_state_machine.py`, `goldfive/plan_reviser.py`, `goldfive/drift_observer.py`. The steerer is now a **router** owning shared mutable state; the drift-routing surface lives in `DriftObserver` (constructed as `steerer.drift`). `#402` (dispatch `GOLDFIVE_STEER` after plan swap) and `#403` (defer `set_session_plan` into the lock to close a partial-apply window) are from this era. See `09-steering-ladder-and-gates.md` and `11-state-ownership.md`.

**Why this matters for your edits:** the split means responsibilities are now *placed*, not scattered. Plan mutation lives in `PlanReviser`; task-status transitions in `TaskStateMachine`; drift routing in `DriftObserver`; shared mutable state (sinks, planner, adapter, control channel, `_observation_only`, plan locks, background-task sets, the `_active_session_var` ContextVar) on `DefaultSteerer`. If you find yourself adding plan-mutation logic to the steerer or drift logic to the reviser, you are undoing the split — put it where the split placed it. `860cffa` (drop router shims, namespace components as properties) finished the cleanup: the components are properties (`steerer.drift`, `steerer.plans`, `steerer.tasks`), not free-floating.

## 5.5b The consolidation wave (2026-05, one-module-per-concern)

**What:** a series of "centralize the scattered thing" refactors that immediately preceded the zicato surface, each collapsing duplicated logic into one owner:

| Commit | Consolidation |
| --- | --- |
| `c197d8b` | Centralize detector boilerplate in `goldfive/drift/registry.py` (register / get_config / list_registered). |
| `89dc48c` | Merge `orchestration_state` + `orchestration_store` into a unified `goldfive/state_store.py` `StateStore`. |
| `fb541cf` | Centralize prompt-shaping gates in `goldfive/prompt_shaper.py` `PromptShaper` (the four injection bodies). |
| `5bdd7c1` | Split `reporting.py` into a `reporting/` package (handlers / schemas / rendering). |
| `5cdd141` | `ContextEditor` — request-side context editing (#397), hard-gated on `observation_only`. |
| `c0d563b` | Consolidate ADK LLM instrumentation into one module (the precursor to #491's `_llm.py`). |
| `321c332` | Merge `recent_agent_activity` + `recent_tool_observations` into one `recent_events` (#239). |

**Why this matters:** each concern now has exactly one home. When you need to touch prompt shaping, drift registration, or state, there is one file — do not re-scatter. The 2026-07 `_llm.py` consolidation (#491) is the same discipline applied to LLM calls: one module owns the two internal call shapes, the capability table, and the diagnostics ContextVar. Adding a *second* place that calls the model directly (bypassing `_llm.py`) re-opens the exact duplication these refactors closed.

## 5.6 Plan-descriptive growth + "adaptive over predictive" (#423, 2026-05)

**What:** the planner learned to **describe** unmatched delegations as tasks rather than **predict** the delegation graph. Codified the "adaptive over predictive" principle (Invariant 4). `Task.discovered`, `Plan.validate`, `DelegationObserved.tool_args` were added (`#423` PR 1 of 5).

**Residue:** the descriptive-growth fallback in `drift_observer`/`plan_reviser`, and `capability_check` Rule C (agent role-stem vs bound-task mismatch, #268/#393). Memory `feedback_dont_predict_agent_behavior`.

**The mechanic and why it matters.** Pre-#423, when a coordinator delegated to a sub-agent that wasn't in the forecast plan, goldfive treated it as a *divergence* (a drift) — manufacturing signal against the agent's legitimate choices (the forecast-grader problem, Section 1.5.2). Post-#423, an unmatched delegation grows the plan **descriptively**: a new `Task` with `discovered=True` is appended, recording what the agent actually did, and `DelegationObserved.tool_args` captures the observed call. `Plan.validate` keeps the grown plan well-formed. This is Invariant 4 in the planner: describe observed facts, don't grade against predictions. **For your edits:** when the agent does something not in the plan, the default response is to *grow the plan to describe it*, not to fire drift. Reserve drift for genuine guardrail violations (loops, stalls) or goal-referenced steering — see the guardrail/steering split (Section 1.10). The five-PR arc (`#423` PRs 1–5, commits `498320e`/`1abee69`/`c0952a5` among them) landed this incrementally.

## 5.7 The zicato optimization surface (2026-05-16)

**What:** zicato is the third ecosystem repo — an **offline meta-loop optimizer** reading goldfive's telemetry + `optimization/manifest.toml` (memory `project_zicato_optimization_surface`). goldfive#436/439/440/442 shipped the manifest (expanded to ~60 entries), `SteeringDecisionMade` decision telemetry, a testkit, and determinism guarantees. goldfive's JSONL sink emits **camelCase**.

**Residue:** `goldfive/optimization/manifest.toml` (the tunable-knob registry — every entry names a `source` file:symbol and a `python_attr`), `SteeringDecisionMade` and the `#480` decision-telemetry label fixes. When you add or rename a tunable knob, update the manifest entry (grep the knob name in `manifest.toml`). See `12-events-sinks-telemetry.md`.

**The manifest entry shape** (`goldfive/optimization/manifest.toml`) — each tunable knob is a `[[mutation]]` table with `id`, `kind`, `source`, and `python_attr` at minimum:

```toml
[[mutation]]
id = "goal_drift_idle_seconds"
kind = "numeric"
source = "goldfive/drift/goals.py:GOAL_DRIFT_IDLE_SECONDS"
python_attr = "goldfive.drift.goals:GOAL_DRIFT_IDLE_SECONDS"
```

The `python_attr` is how zicato *imports and sets* the knob at runtime; the `source` is the human breadcrumb. The manifest grew to ~60 entries (#440). `#436` shipped determinism guarantees + a testkit so the meta-loop's reads are reproducible. **The AST-liveness test (#487)** parses the manifest and fails CI if any `python_attr` points at a symbol that no longer exists — so renaming a knob without updating the manifest is caught. **Rule:** rename a knob → update its manifest entry in the same commit, or the liveness test fails.

## 5.8 The judge/config maturation (2026-06, #437–#447)

**What:** pluggable `Judge` protocol + `JudgementEmitted` event (#439/#437); enum-typed `JudgeVerdict.drift_kind`/`severity` (#443); typed `BuiltinJudge` selector + `wrap(disable_judges=...)` (#444); gradeable actual-output capture (#447); first-class **judge-only mode** — native run + judges, no planning/steering overlay (#446). `#442` keyed the user-steer suppression window on a logical-turn counter (Invariant 6).

**Residue:** `goldfive/judges/`, `goldfive/builtin_judges.py`, the `Judge` protocol in `goldfive/protocols.py`, and judge-only mode. See `08-llm-judges.md`.

**The three run modes (don't conflate them).** goldfive has three distinct operating modes; a common confusion is treating `observation_only` and judge-only as the same thing. They are orthogonal:

| Mode | How to enter | What runs | What doesn't |
| --- | --- | --- | --- |
| **Active steering** | `wrap(..., RuntimeConfig(steering=SteeringConfig(observation_only=False)))` | Everything: detection, judges, planning overlay, interventions. | — |
| **Observation-only** (default) | shipped default (`observation_only=True`) | Detection, judges, `planner.refine_steer` (dry_run), full telemetry. | Interventions (the Section 1.5.1 write-paths). |
| **Judge-only** | `wrap(..., judge_only=True)` | A native agent run with built-in and custom judges. A `StaticPlanner` supplies one framing task, and the default steerer emits judgement and drift events. | The default steerer returns before cancellation, promotion, the intervention ladder, refinement, and escalation. |

`observation_only` suppresses writes after the planning and steering machinery has run, including dry-run refinement. Judge-only keeps the observation machinery and removes the response machinery. `disable_judges=[...]` further trims which built-in judges run.

## 5.9 The agency-preservation branch (#449–#474, unmerged)

**What:** diagnosed goldfive as a "controller in disguise" and reframed it around the **dormant-supervisor** identity: lane-keep assist, not autopilot (memory `project_agency_preservation_roadmap`, `docs/design/AGENCY-PRESERVATION.md`). Stages 1–3 (ledger + observer-note + intervention-surface hierarchy) shipped **to the branch** behind default-OFF flags. Step 13b (three-arm bench + measurement-gated flips + hard deletions) is LOCKED on sign-off. **The branch is not on main.** Docs #449–#452 (the roadmap, the correctness strategy §5, the intervention-surface hierarchy §5, the dormant-supervisor reframing) *are* on main as design text.

**The framing that explains every invariant in this chapter** (`docs/design/AGENCY-PRESERVATION.md` §0, on main as design text). goldfive's purpose decomposes into three behaviors with *different contracts* — internalize these, because they are the "why" behind `observation_only` and Invariant 1:

- **Dormant (steady state):** while the agent makes progress toward the user's goal, goldfive has **zero trajectory footprint** — no per-turn injection, no grading against a forecast, no preemption. Observation and event emission only. This is what `observation_only=True` operationalizes today.
- **Guardrails (always armed):** hard limits on *observed facts* — tool loops, reasoning loops, stalls, budgets, runaway delegation, refusals. Cheap, low false-positive, legitimately always on. Their job is to **stop** runaway behavior, not redirect it.
- **Steering (engaged only on drift):** corrective influence when the trajectory diverges from the *user's goal*. Requires a reference + an LLM judge; expensive and fallible, so it engages **conditionally, proportionally, honestly attributed**.

The diagnosis that produced `observation_only=True` as the default: "the active half of the product is currently too disruptive to leave on." Today's steer = swap the plan + kill the in-flight invocation + restart the tree with goldfive's text as a synthetic user turn — a "wheel grab," not proportional steering. The branch's Stages 1–3 (evidence ledger, observer-note as a lighter-weight influence, intervention-surface hierarchy) are the planned fix. **On main, your job is to keep the dormant contract honest:** when nothing drifts, footprint must be zero. Any per-turn tax you add to the healthy path violates the dormant contract even if it never "intervenes."

**Residue on main:** `docs/design/AGENCY-PRESERVATION.md` (the framing + roadmap), the strict-passive discipline the 2026-07 program then enforced in code. See Section 2.4 for the branch-boundary rules.

## 5.10 The 2026-07 hardening program (#475–#492)

The most recent arc, all on `main`. This is the program that made `observation_only` strictly passive in code, bounded every wait, and paid down the refactor debt. One line each:

| PR | One-line intent |
| --- | --- |
| #475 | Gate the NUDGE injection path on `observation_only`; pair with truthful nudge text. |
| #476 | `observation_only` no longer cancels on `LLM_CALL_TIMEOUT`; one-shot loud warning when the reasoning channel disarms. |
| #477 | Preserve ADK `{var}` session-state templating under `dynamic_instruction` (via `inject_session_state`). |
| #478 | `report_awaiting_approval` cannot hang: no-channel → immediate `'unavailable'` ack; finite 600s default timeout; expiry emits `HUMAN_INTERVENTION_REQUIRED`; strip `plan_state` from acks under `observation_only`. |
| #479 | Five hardening fixes: malformed judge severity→INFO; embedding-breaker half-open recovery; sink exceptions never abort runs; correction keys use full agent path; judge history pinned by snapshot-passing. |
| #480 | Fix four decision-telemetry label corruptions (`DriftEvent.detector_name`, `drift_dropped_stale`/`inflight` outcomes, `capability_check` negative class, `ReasoningJudgeInvoked` proto fields 12–15: `focused_task_id`/`focus_confidence`/`stated_intent`/`provenance`). |
| #481 | Gate the F3 pre-dispatch redirect on `observation_only`; align its predicate with the delegation pin. |
| #482 | Bound the ladder's terminus: `pause_escalate_deadline_s` + real TERMINATE (600s built-in deadline) + `RunAborted` carrying escalation lineage. |
| #483 | Judge-scheduling guards: per-steerer semaphore (default 3), queued-window coalescing, verdict-utility ledger + teardown summary event, endpoint-contention warning. |
| #484 | Cap uncorroborated tool-loop name-axis at INFO without exact-repeat corroboration (`>=2` identical `(name, args_hash)`); knob `name_axis_max_severity`; `raw["severity_capped_from"]`; aggregated no-drift decision. |
| #485 | Canonical `TERMINAL_TASK_STATUSES` + shared executor helpers; parallel scheduler skips terminal tasks (incl. `NOT_NEEDED`). |
| #486 | Wire drift-condition resolution: task-terminal transitions + staleness-guarded on-task verdicts emit `DRIFT_LIFECYCLE_RESOLVED` (GOAL_DRIFT resolves only at task-terminal). |
| #487 | Flag-gated wall-clock stall watchdog (`stall_watchdog_enabled` default False, `stall_timeout_s` 600) = the `TASK_TIMEOUT` producer; `Session.last_observed_event_at` liveness stamp; idle goal-judge trigger; AST-based manifest-liveness test. |
| #488 | One public predicate for the kill-switch (`is_active_steering()`/`steering_is_active()`); delete the module-global test hook + autouse fixture; suite runs the shipped `observation_only=True` default (~90 tests opt into active mode). |
| #489 | Extract `Runner._abort_turn` (8 copy-paste sites) + decompose `_run_overlay` into named stage methods. |
| #490 | Delete verified-dead code with archaeology (`_LADDER_BY_VALUE`, `registry.classify`, keyword detector, deprecated shims). |
| #491 | One internal LLM-call module `goldfive/_llm.py`; `THINKING_DISABLE_CAPABILITIES`; per-call `LlmCallDiagnostics` via `ContextVar` replacing closure attributes; Qwen `/no_think` + `enable_thinking` now Qwen/litellm-family only. |
| #492 | Design-doc accuracy sweep — reconcile design docs with live code. |

**How to use this table:** when you touch a subsystem, grep the git log for its PR to read the original commit message — those messages are the primary source for "what problem was this solving." Example: `git show cf81d52` for #476.

## 5.11 The ecosystem: harmonograf and zicato (contracts you can break from inside goldfive)

goldfive is the middle of a three-repo ecosystem. Edits inside goldfive can silently break the siblings; know the contracts.

- **harmonograf** — the observability UI. It consumes goldfive's **event stream** (protos + sink envelopes). Contracts you can break: (1) proto additivity (Hazard 4.11) — a renumber corrupts harmonograf's decode; (2) `event_id` global uniqueness (Hazard 4.12) — collisions collapse distinct events in its fan-in; (3) the **synthetic-drift** flag (#302) — `Runner._install_revision`'s `USER_STEER` drift is marked synthetic so harmonograf filters it from the "interventions" view; if you emit a framework-authored drift without that flag it pollutes the operator's intervention list; (4) session-rollup stamping (harmonograf#61, memory `project_session_rollup_fix`) — spans + goldfive events collapse onto the client's Hello session, with the ADK sub-session preserved as a span attribute. **NEVER wipe harmonograf data** unless explicitly told (memory `feedback_auto_merge`).
- **zicato** — the offline meta-loop optimizer (the 3rd ecosystem repo, memory `project_zicato_optimization_surface`). It reads goldfive's **telemetry + `optimization/manifest.toml`** and proposes knob values. Contracts you can break: (1) JSONL **camelCase** + determinism (Hazard 4.13); (2) the **manifest** — every tunable knob has a `manifest.toml` entry pointing `source`/`python_attr` at a live symbol; the AST-liveness test (#487) fails if you rename a knob without updating the manifest; (3) **decision telemetry** — `SteeringDecisionMade` + the `DriftEvent.detector_name` pairing (#480) is how zicato attributes an outcome to a knob; a mislabeled decision (the four #480 corruptions) makes the meta-loop optimize the wrong lever. The decision-telemetry **negative class** (the "no drift" / "capability OK" case) must be emitted (#480 fixed `capability_check`'s missing negative class), or zicato can't compute precision.

**The auto-merge cadence** (memory `feedback_auto_merge`): on goldfive/harmonograf, proceed through review → merge → submodule bump without asking — but this is a *guide chapter*, read-only; the rule is context for how the ecosystem is operated, not a license to merge from here.

## 5.12 Recipe: reading history to recover intent (stop guessing)

When you find code you don't understand, do not guess its purpose — recover it. The commit messages in this repo are unusually detailed and are the primary intent source.

```bash
# Who last touched this line and why (the PR message is the intent):
git log -1 --format='%H %s' -L <start>,<end>:goldfive/<file>.py
# When was this symbol introduced / removed / last edited:
git log -S'<symbol>' --oneline -- goldfive/
# Read a specific PR's full rationale (commit message + diff):
git show <hash>
# Find the PR that closed an issue referenced in a comment:
git log --oneline --all | grep -i "#<issue>"
```

The `#NNN` numbers in code comments are load-bearing: a comment like "goldfive#402 (fixed): the GOLDFIVE_STEER dispatch fires AFTER _emit_plan_revised" tells you both *what* the ordering is and *why* it can't be reordered. Do not "clean up" a comment that cites a PR number — it is the breadcrumb the next agent needs.

## 5.13 Consolidated operational lessons (the memory-file canon)

These are the hard-won lessons from prior sessions (the `feedback_*` / `workflow_*` auto-memory). They are canonical guidance; a weak model that ignores them repeats an expensive mistake.

| Lesson | Rule |
| --- | --- |
| Isolated worktrees | The main `~/git/goldfive` checkout is unsafe when other agents run; use a `/tmp` worktree (`workflow_goldfive_concurrent_agents`). |
| Empirical baseline first | If a failure correlates with a STEER/CANCEL/PAUSE, run the same driver **without** steering before blaming the model/#92 (`feedback_empirical_baselines`). |
| Integration, not unit | After adding a middleware helper/guard, grep for real call sites — unit tests pass on dead code (`feedback_integration_not_unit`). |
| Shallow-copy handoffs | ADK/ADK-family SDKs return shallow copies of `session.state`; verify the read side sees the write (`feedback_callback_context_handoff`, cost ~8h). |
| wrap contract | `goldfive.wrap` must handle any ADK tree incl. coordinator+AgentTool; failures there are goldfive bugs, not tree-shape problems (`feedback_goldfive_wrap_contract`). |
| Self-review before reporting | Bake correctness/simplicity/alignment review into the work; read the diff once more before pushing (`feedback_agent_self_review`). |
| No prompt contracts in core | Users bring their own coordinator prompts; termination/control/observability must work without agents calling specific tools (`feedback_no_prompt_contract`). |
| Don't blame the LLM for slowness | 5+ min/turn is almost always goldfive code (unbounded `max_tokens`, drift loops, missing wall-clock budgets), not the model (`feedback_dont_blame_llm_for_slowness`). |
| Layered E2E validation | DB-only checks miss UI + slow-burn health regressions; use the six-layer protocol (sanity, drive, DB, UI, health, output) (`feedback_e2e_validation_layered`). |
| Narrow criteria pass on broken runs | Bake functional-completion + steer-honoured checks into regression criteria (`feedback_validation_criteria_too_narrow`). |
| No regex NL heuristics | Retired `_GENERIC_VERB_PREFIX_RE` (#166) / `_FACTUAL_QUESTION_RE` (#167); use LLM classifiers or design-away (`feedback_no_regex_heuristics`). |
| Stable keys for gates | A per-condition gate is only as good as its key; fix upstream churn, don't coarsen the key (`feedback_stable_keys_for_lifecycle_gates`). |
| Verify the running build | When a symptom follows a change, check what's actually running (process start vs merge time, `lsof` on logs) before diagnosing (`feedback_verify_running_build`). |
| External evidence trumps logs | GPU fans / model-server logs / harmonograf state contradicting your "no activity" read → dig deeper (`feedback_external_evidence_trumps_logs`). |
| Reader-centric versioning | Freshness gates key on the claim's identity (kind, target), not a global writer counter; naive `version` compares over-reject under fan-in (`feedback_reader_centric_versioning`). |
| Adaptive over predictive | Capture observed facts via extended protos/events, not interception at pin/dispatch time (`feedback_dont_predict_agent_behavior`). |
| No Claude co-author trailer | The user scrubs them from goldfive/harmonograf history; omit by default (`feedback_no_claude_coauthor`). |
| Don't ask before review/merge/bump | Auto-proceed through review→merge→submodule bump on goldfive/harmonograf; NEVER wipe harmonograf data unless told (`feedback_auto_merge`). |
| Team-lead pattern | For large parallelizable work, coordinate specialized agents with proactive status updates; ship tests+docs with code (`feedback_team_lead_pattern`). |

## 5.14 The design-doc map (where the "why" is written — but code wins)

The `docs/design/*.md` files are the authoritative *intent* source, kept honest by the #492 accuracy sweep. Use them to understand design; use the **code on main** as ground truth for behavior. Where a doc and the code disagree, the code wins — and you should fix the doc (that is what #492 did).

| Doc | Covers |
| --- | --- |
| `AGENCY-PRESERVATION.md` | The dormant-supervisor framing, the roadmap, §5 correctness strategy, §6 as-built (on the branch). Read §0 first. |
| `ARCHITECTURE.md` | The overall wrap→observe→detect→steer shape. |
| `DRIFT.md` | The drift taxonomy — every `DriftKind` row. |
| `CONTROL.md` / `CONTROL-CHANNEL.md` | The `ControlKind` vocabulary and the control channel. |
| `CANCELLATION-CONTRACT.md` | The cancel semantics + the stash invariant (#287/#326). |
| `STATE-MACHINE.md` / `STATE-OWNERSHIP-CONTRACT.md` | `TaskStateMachine`, `TERMINAL_TASK_STATUSES`, who owns which state. |
| `TASK-LIFECYCLE.md` | Task status transitions incl. `NOT_NEEDED`. |
| `PLAN-LIFECYCLE.md` / `PLAN-DESCRIPTIVE-GROWTH.md` | Plan install/revision + the #423 descriptive-growth reframe. |
| `EVENT-MODEL.md` | The proto event schema + additivity rules. |
| `APPROVAL.md` | `report_awaiting_approval`, Flow A / Flow B, the no-hang contract. |
| `CONTEXT-EDITING.md` | `ContextEditor` (#397), the drop-only / no-injection contract. |
| `PROTOCOLS.md` | The `Judge` / planner / adapter protocols. |
| `VOCABULARY.md` | The canonical term glossary (superset of Section 9). |
| `SHARED-RUNNER-REFACTOR.md` | The single-runner shape (#128 family). |
| `RATIONALE.md` | Cross-cutting "why" decisions. |

**Rule:** cite the doc for intent in your PR, but verify the behavior against the code. If you change behavior, update the doc in the same PR (Pre-PR item 17). A stale doc that contradicts the code is a #492-class bug — the code is what runs.

---

# 6. The Pre-PR Checklist

Every change must pass this single consolidated checklist before you open a PR. It folds in the invariants, the hazards, and the mechanical hygiene. Do them in order; do not skip an item because the change "looks trivial" — Invariant-5 leaks and unstable-key gates both look trivial.

## 6.1 Correctness & invariants

1. **Kill-switch gate (Invariant 5).** Does your change add or move any intervention surface (cancel / inject / mutate / refuse / install / directive-ack-with-goldfive-state)? If yes: it reads the kill-switch ONLY via `is_active_steering()` / `steering_is_active(steerer)`, fails PASSIVE, and you have BOTH a passive test (asserts nothing happens) and an active test (asserts it fires). Grep to confirm no raw `_observation_only` read leaked in:
   ```bash
   grep -rn "_observation_only" goldfive/ | grep -v "def is_active_steering" | grep -v "steering_is_active"
   ```
2. **No prompt-cooperation (Invariant 1).** Does any new enforcement depend on the agent calling a tool or obeying an instruction? If yes, redesign — enforcement is structural (executor/plugin/plan), cooperation is best-effort only.
3. **No NL regex/keyword heuristics (Invariant 2).** Did you add a regex/wordlist that classifies agent free-text into a drift/severity/intervention? If yes, remove it — use an LLM classifier or design it away. Structured equality/hashing is fine.
4. **Any tree shape (Invariant 3).** If you touched the adapter/plugin/cancel path, run all `tests/test_adk_*` (not just a flat-agent test). Cancel no-ops cleanly on empty/unbound/non-ADK.
5. **Adaptive not predictive (Invariant 4).** Are you reacting to a recorded fact or forecasting a future move? If forecasting, add an observed-fact field and react later.
6. **Stable gate keys (Invariant 6).** Every new dict/set gate keys on a stable tuple (full agent path, logical-turn counter, content hash of structured fields) — NEVER `drift.id` or an LLM-minted id. Write a two-observation-collapse test.
7. **Negative class present.** If you added a detector/telemetry classifier, does it emit the **negative** decision too (the "no drift" / "capability OK" case), not just the positive fire? (#480 fixed exactly this for `capability_check`.) Decision telemetry must record both classes so zicato can measure precision.
8. **Both refine pipelines (Section 3.1).** If you touched the refine flow, did you mirror the change in BOTH `_handle_drift_dispatch` AND `_promote_drift_to_steer`? Grep both:
   ```bash
   grep -n "_handle_drift_dispatch\|_promote_drift_to_steer\|_record_refine_outcome" goldfive/drift_observer.py
   ```

## 6.2 Hazard sweep

9. **Nothing hangs (4.3).** Every new `await` on a human/channel/judge/lock has a finite deadline and a defined expiry behavior.
10. **Nothing cancels healthy work on a weak signal (4.4).** Timeouts, parse failures, sink errors, and malformed verdicts degrade to INFO/observation — they do not cancel. Zero cancels in `observation_only`.
11. **Injected text is true (4.2).** Every factual claim in a nudge/corrective message comes from state verified this turn.
12. **Cost delta acknowledged (4.5).** Any added LLM/embedding/tool call is under the judge semaphore / coalesced where applicable, and you can name the cost.
13. **No new module-global mutable (4.8).** Per-call/per-session state uses `ContextVar` or session scope, not a module global.
14. **Protected list respected (Section 2).** You did not delete/simplify `LOOPING_TOOL_CALL` surfaces, `PLAN_DIVERGENCE` machinery, or `get_missed_tasks`; you did not tune a bench-frozen default; you did not copy from the `agency-preservation` branch.

## 6.3 Mechanical hygiene

15. **Full test suite green.**
    ```bash
    uv sync --extra dev --extra adk
    uv run pytest -q      # ~30s, expect ~2912 passed / 61 skipped
    ```
    A subset passing is not evidence — run the whole suite.
16. **Lint clean.**
    ```bash
    ruff check .          # must stay clean
    ```
    Do **NOT** run `ruff format` across the repo — it is not format-clean and you would generate a giant spurious diff. Format only lines you actually changed, matching surrounding style.
17. **Design docs touched if behavior changed.** If you changed observable behavior or a default, update the relevant `docs/design/*.md` (the #492 accuracy sweep is the precedent). Where docs and code disagree, code wins — fix the doc.
18. **Conventional-commit title.** `feat(...)`, `fix(...)`, `refactor(...)`, `docs(...)`, `chore(...)` with a scope and the closing `(closes #NNN)` / `(#NNN)`. Match the style in `git log`.
19. **NO Claude co-author trailer.** The user scrubs these from history in goldfive and harmonograf (memory `feedback_no_claude_coauthor`). Do not add `Co-Authored-By: Claude` to goldfive commits.
20. **Self-review the diff once more (memory `feedback_agent_self_review`).** Read the whole diff for correctness, simplicity, and alignment before pushing. Autonomous agents must self-review before reporting; ship tests + docs *with* the code, not after (memory `feedback_team_lead_pattern`).

## 6.4 The one-paragraph summary for a hurried reader

Under the shipped default (`observation_only=True`), goldfive **watches and does nothing to the agent.** Any new code that touches the agent must gate on `steering_is_active(steerer)` (fail PASSIVE), be tested in both modes, never depend on agent cooperation, never classify free-text with regexes, never cancel healthy work on a weak signal, never hang unboundedly, and never key a gate on a churning id. Refine-flow edits go in **both** pipelines. Don't delete the protected surfaces, don't tune the bench-frozen defaults, and don't copy from the `agency-preservation` branch. Run the full suite + `ruff check .`, update the design doc if behavior changed, use a conventional-commit title, and omit the Claude co-author trailer. When unsure whether something is an intervention: assume it is, and gate it.

## 6.5 PR-description template

Fill this in for any change that touches a subsystem this chapter governs. It forces you to name which invariants you considered.

```
## What
<one-line summary; closes #NNN>

## Invariant impact (delete the lines that don't apply)
- Inv 1 (no coop): <this change's enforcement is structural because ...>
- Inv 2 (no regex): <no NL heuristics added / N/A>
- Inv 3 (tree shape): <ran all tests/test_adk_* / N/A>
- Inv 4 (adaptive): <reacts to recorded fact X, not a forecast / N/A>
- Inv 5 (passive): <new surface gated via steering_is_active; both-modes tests added / N/A>
- Inv 6 (stable keys): <gate keyed on (kind, task_id); collapse test added / N/A>

## Protected / deferred touchpoints
- [ ] No KEEP surface (2.1-2.3) deleted
- [ ] No bench-frozen default (2.5) changed without sign-off
- [ ] Both refine pipelines edited (if refine-flow) — grep confirms
- [ ] No copy from agency-preservation branch

## Verification
- [ ] uv run pytest -q  (2912+ passed)
- [ ] ruff check .  (clean; no ruff format)
- [ ] design doc updated (if behavior changed)
```

Omit the Claude co-author trailer. A reviewer who sees the invariant-impact block populated knows you actually considered them; a blank block is a red flag.

---

# 7. Common mistakes

The distilled catalog of concrete wrong edits a weaker model plausibly makes in this codebase, each with the correct alternative. This is a superset of the per-invariant "concrete wrong edit" notes above, gathered so you can scan them in one place before you touch anything.

| # | The wrong edit (what a weak model does) | Why it's wrong | The correct alternative |
| --- | --- | --- | --- |
| 1 | Read `steerer._observation_only` directly in a new consumer. | Bypasses the single predicate; likely wrong fail direction (Inv 5). | `steering_is_active(steerer)` (external) / `self.is_active_steering()` (in `DefaultSteerer`). |
| 2 | Add an intervention, test it in active mode only. | The passive leak (the dangerous case) is untested. | Add BOTH a passive test (nothing happens) and an active test (it fires). |
| 3 | Key a new gate on `drift.id`. | UUID4 per emit → gate never engages, silently a no-op (Inv 6). | Key on `(kind.value, current_task_id)` or a sha1 of stable structured fields. |
| 4 | Make the correction key the bare agent name. | Two agents in different subtrees collide; churn if renamed (Inv 6). | Full agent path via `pending_correction_key` / `_normalize_agent_name` (#479). |
| 5 | Add a regex to classify thinking tokens into a drift kind. | NL classification by heuristic (Inv 2); brittle, un-tunable. | Reasoning judge → typed `JudgeVerdict`. |
| 6 | Escalate a tool loop on "same tool name, different args". | Lexical near-match (Inv 2); no exact-repeat evidence. | Cap at INFO unless `>=2` identical `(name, args_hash)` (#484). |
| 7 | Inject "please stop, you are off-task" and rely on it. | Cooperation contract (Inv 1); agent may ignore. | Level-3 `CANCEL_REINVOKE` via `GOLDFIVE_STEER` — structural. |
| 8 | Terminate the run off an agent tool call ("call `done()`"). | Cooperation contract (Inv 1). | Generator-end + drift detectors + stall watchdog. |
| 9 | Cancel healthy work when an LLM call times out. | Infrastructure hiccup treated as drift (Hazard 4.4); also fires in passive mode. | Under `observation_only`: emit one warning, do not cancel (#476). |
| 10 | Treat a malformed judge verdict as a CRITICAL drift. | A parse failure is not evidence of drift (Hazard 4.4). | Malformed severity → INFO (#479). |
| 11 | `await` on a human approval with no timeout. | Hangs forever if no operator (Hazard 4.3). | Finite 600s default; no channel → immediate `unavailable` (#478). |
| 12 | Reach into `agent.sub_agents[0]` to cancel a sub-agent. | Breaks on coordinator+AgentTool (Inv 3). | `request_invocation_cancel` via the plugin registry; no-op on empty. |
| 13 | Edit `_handle_drift_dispatch` only. | The twin `_promote_drift_to_steer` diverges (Section 3.1). | Mirror the change in both; grep both before commit. |
| 14 | Delete `LOOPING_TOOL_CALL` / `get_missed_tasks` / `PLAN_DIVERGENCE` as "dead". | Protected KEEP decisions (Section 2). | Leave them; cite Section 2 in the PR; ask for sign-off. |
| 15 | Flip `observation_only`, `stall_watchdog_enabled`, or `name_axis_max_severity` default. | Bench-frozen (Section 2.5); changes behavior for all operators. | Leave; needs 13b bench / regression measurement + sign-off. |
| 16 | Copy a mechanism from the `agency-preservation` branch to main. | Branch-boundary violation (Section 2.4). | Re-derive on main only if in scope; expect the four-file merge conflict later. |
| 17 | Add a module-global mutable flag for per-call state. | Last-writer-wins race across sessions/tests (Hazard 4.8). | `ContextVar` (see `_llm.py` `LLM_CALL_DIAGNOSTICS_VAR`). |
| 18 | Interpolate a stale count into a nudge ("you called X 5 times"). | Injected text must be true (Hazard 4.2). | Re-read the count this turn, or emit a neutral refocus. |
| 19 | Make `STATUS_QUERY` emit a sink event "for observability". | Polling must not register as drift; breaks the read-only contract. | Keep it silent; return the snapshot via the ack `detail` only. |
| 20 | Assume a `session.state` write is visible to the plugin callback. | ADK may hand back a shallow copy (Hazard 4.6). | Verify the read side sees the write; use the documented handoff. |
| 21 | Add an eighth ad-hoc suppression gate to `handle_drift`. | Increases the debt the evidence-ledger will replace (Section 3.2). | Make an existing gate's key stable instead; if you must add, key it stably and co-locate. |
| 22 | `ruff format` the repo to "clean it up". | Repo is not format-clean; produces a massive spurious diff. | Format only your changed lines; run `ruff check .` (lint), not format. |
| 23 | Add `Co-Authored-By: Claude` to a goldfive commit. | The user scrubs these (memory `feedback_no_claude_coauthor`). | Omit the trailer. |
| 24 | Emit only the positive drift decision, not the "no drift" case. | Decision telemetry needs the negative class for precision (#480). | Emit both classes so zicato can measure. |
| 25 | Predict the delegation graph and pre-pin tasks. | Predictive, not adaptive (Inv 4). | Describe unmatched delegations after they happen (#423). |
| 26 | Renumber or reuse a proto field number to "clean up." | Breaks every wire consumer (Hazard 4.11). | Append the next number; never renumber; regenerate stubs. |
| 27 | Let a confident judge cancel the run directly. | Collapses the judge/steerer separation (Section 3.4); judges are fallible. | Judge returns a `JudgeVerdict`; the steerer decides + gates. |
| 28 | Fire drift because the agent went "off the forecast plan." | Forecast-grading manufactures signal (Section 1.5.2, Inv 4). | Grow the plan descriptively; drift only on guardrails/goal-judge. |
| 29 | Inject a goldfive helper-prompt directive even in passive mode. | Passive mode must have zero goldfive prompt footprint (#394). | Gate all goldfive-authored prompt content on `is_active_steering()`. |
| 30 | Add a second code path that calls the model directly. | Re-opens the duplication #491's `_llm.py` consolidation closed. | Route through `goldfive/_llm.py`. |
| 31 | Add a wall-clock cooldown to slow down refines. | Cooldown was directively excluded (Section 5.3). | Tighten a structural gate's engagement instead. |
| 32 | Require the agent to call `report_*` for lifecycle to advance. | Cooperation contract (Hazard 4.17, Inv 1). | Lifecycle/termination work with zero `report_*` calls. |

## 7.1 The drift-condition lifecycle (a mistake magnet)

Because #486 wired drift resolution, a common new-code mistake is mis-resolving a condition. The lifecycle enum (`goldfive/drift_observer.py` `_drift_lifecycle_pb_value`, mapping to `types_pb2`):

- `DRIFT_LIFECYCLE_OPENED` — the condition is live.
- `DRIFT_LIFECYCLE_ESCALATING` — it climbed the ladder.
- `DRIFT_LIFECYCLE_RESOLVED` — it cleared. **GOAL_DRIFT resolves ONLY at task-terminal** (a task reached a `TERMINAL_TASK_STATUSES` state); an on-task verdict resolves only when staleness-guarded (#486).
- `DRIFT_LIFECYCLE_HUMAN_INTERVENTION_REQUIRED` — parked for an operator (e.g. `RefineExhausted`).

**Mistake:** resolving a `GOAL_DRIFT` because a single on-task judge verdict came back clean mid-task. Wrong — a mid-task clean read is not terminal; the goal can re-drift. Correct: `GOAL_DRIFT` resolves only at task-terminal (#486). This is observability-truth only — it does not itself intervene.

---

# 8. Verification checklist (exact commands)

Run these after touching any subsystem this chapter governs. They are the concrete grep/test commands that catch invariant and hazard violations. Copy-paste them; do not approximate.

**Environment (once per session):**

```bash
cd ~/git/goldfive          # or an isolated /tmp worktree if other agents run (memory workflow_goldfive_concurrent_agents)
uv sync --extra dev --extra adk
```

**Invariant 5 — no raw kill-switch read leaked in:**

```bash
grep -rn "_observation_only" goldfive/ | grep -v "def is_active_steering" | grep -v "steering_is_active"
# Expect: only the definition of is_active_steering() itself. Anything else is a bug.
```

**Invariant 2 — no NL-classification regexes / retired heuristics:**

```bash
grep -rn "re\.compile\|re\.search\|re\.match" goldfive/drift/ goldfive/drift_observer.py goldfive/judges/ goldfive/builtin_judges.py
grep -rn "_GENERIC_VERB_PREFIX_RE\|_FACTUAL_QUESTION_RE\|CONFUSION" goldfive/
# Expect: no NL-classification regex; retired names absent.
```

**Invariant 6 — no gate keyed on a churning id:**

```bash
grep -rn "\.id\]" goldfive/drift_observer.py
grep -rn "pending_correction_key\|_normalize_agent_name" goldfive/_correction_injection.py
```

**Section 3.1 — both refine pipelines touched together:**

```bash
grep -n "_handle_drift_dispatch\|_promote_drift_to_steer\|_record_refine_outcome" goldfive/drift_observer.py
```

**Protected list — you did not touch a KEEP surface:**

```bash
grep -rn "LOOPING_TOOL_CALL\|LOOPING_REASONING" goldfive/
grep -rn "PLAN_DIVERGENCE" goldfive/reconciler.py goldfive/steerer.py
grep -rn "get_missed_tasks" goldfive/
```

**Hazard 4.3 — nothing hangs unboundedly:**

```bash
grep -rn "wait_for\|timeout\|deadline\|Event().wait" goldfive/reporting/ goldfive/executors/ goldfive/drift_observer.py
```

**Targeted test runs (pick the ones your change touches):**

```bash
# ADK / tree-shape (Inv 3) — run ALL of them, not one:
uv run pytest -q tests/test_adk_adapter.py tests/test_adk_adapter_overlay.py \
  tests/test_adk_adapter_concurrent_sessions.py tests/test_adk_reentry.py \
  tests/test_adk_wrap_passthrough.py tests/test_delegation_pin.py
# Cancellation / control (Inv 1):
uv run pytest -q tests/test_cooperative_cancellation.py tests/test_cancel_propagation.py \
  tests/test_cancel_reason.py tests/test_control_primitive.py
# Approval no-hang (Hazard 4.3):
uv run pytest -q tests/test_approval_flow.py
# Gate keys (Inv 6):
uv run pytest -q tests/test_correction_injection.py
# Decision telemetry / negative class (#480):
uv run pytest -q tests/test_decision_telemetry.py
# LLM diagnostics ContextVar (Hazard 4.8):
uv run pytest -q tests/test_call_llm_diagnostic.py
```

**Full gate (always, before PR):**

```bash
uv run pytest -q          # ~30s; expect ~2912 passed / 61 skipped
ruff check .              # must be clean; do NOT run ruff format
```

If any expected-empty grep returns hits, or the suite count regresses, stop and reconcile before opening the PR.

## 8.0 Test-file → what it guards (pick by the subsystem you touched)

| If you touched... | Run these test files |
| --- | --- |
| Cancellation / control | `test_cooperative_cancellation.py`, `test_cancel_propagation.py`, `test_cancel_reason.py`, `test_cancel_inflight_on_refine.py`, `test_control_primitive.py`, `test_control_proto.py` |
| The ADK adapter / plugin | `test_adk_adapter.py`, `test_adk_adapter_overlay.py`, `test_adk_adapter_concurrent_sessions.py`, `test_adk_adapter_pending_tool_isolation.py`, `test_adk_plugin_tool_observations.py`, `test_adk_reentry.py`, `test_adk_wrap_passthrough.py` |
| Delegation / pinning | `test_delegation_pin.py`, `test_descriptive_growth_capability_mismatch_fallback.py`, `test_before_tool_task_id_injection.py` |
| Approval flow | `test_approval_flow.py` |
| Correction injection (gate keys) | `test_correction_injection.py`, `test_corrective_message.py` |
| Decision telemetry (negative class) | `test_decision_telemetry.py` |
| Determinism / zicato surface | `test_determinism.py` |
| The cancellation-stash tripwire | `test_cancellation_stash_audit.py`, `test_cancellation_stash_tripwire.py` |
| Capability mismatch | `test_capability_mismatch.py` |
| LLM diagnostics ContextVar | `test_call_llm_diagnostic.py` |
| Context editing | `test_context_editor.py` |
| Conversation / cursor | `test_conv.py`, `test_conv_strict.py`, `test_conversation.py` |
| Agent budget / guardrails | `test_agent_budget.py` |
| Claude adapter | `test_claude_adapter.py` |
| Callable / prebuilt adapters | `test_callable_adapter.py`, `test_degrade_prebuilt_runner.py` |
| Adapter cancellation plugin | `test_adapter_on_cancellation_plugin.py` |

When in doubt, run the full suite — it's ~30s. A targeted subset is for iterating, not for the final green. If your change adds a new intervention surface, the *new* tests you add matter more than the existing ones you run: a passive test and an active test (Section 8.1) are the evidence that the gate works.

## 8.1 The both-modes test pattern (post-#488)

Since #488 the suite runs the **shipped `observation_only=True` default**; the autouse fixture that used to flip everything to active is gone. This changes how you write intervention tests. The pattern:

```python
# PASSIVE test — the default fixture path. Assert the intervention does NOT fire.
async def test_my_surface_is_dormant_by_default(...):
    steerer = DefaultSteerer(...)                 # observation_only=True (shipped default)
    await run_the_flow(steerer)
    assert session.pending_nudges == []           # nothing injected
    assert session.plan is baseline_plan          # no corrective revision installed
    # (DriftDetected / telemetry MAY still be present — observation is expected.)

# ACTIVE test — opt in explicitly via the fixture. Assert it DOES fire.
async def test_my_surface_fires_in_active_mode(make_active_steerer):
    steerer = make_active_steerer(...)            # observation_only=False
    await run_the_flow(steerer)
    assert session.pending_nudges != []           # injected as expected
```

Opt-in mechanisms (`tests/conftest.py`): the `active_steering_config` fixture (a `SteeringConfig(observation_only=False)`), the `make_active_steerer` factory, or an inline `SteeringConfig(observation_only=False)`, or a stub whose `is_active_steering()` returns `True`. **A test that only checks active behavior is half a test** — the passive assertion is the one that catches an Invariant-5 leak, because a leak IS "fires in passive mode." Roughly ~90 tests opt into active mode; if your new feature is an intervention, add to both counts.

**Do not** re-introduce a module-global default flip or an autouse fixture to make active mode the suite default — that was deleted for cause (#488, Hazard 4.8): it made the whole corpus lie about the shipped behavior.

---

# 9. Glossary of load-bearing terms

The terms this chapter uses that a weaker model most often misreads. (Full vocabulary: `docs/design/VOCABULARY.md`.)

| Term | Meaning in goldfive |
| --- | --- |
| **observation_only** | The master kill-switch (`SteeringConfig`, default `True`). Strictly passive since Waves 1–4: no interventions, no goldfive-authored prompt footprint. Read ONLY via `is_active_steering()` / `steering_is_active()`. |
| **dormant supervisor** | goldfive's identity: zero trajectory footprint while the agent makes progress; guardrails always armed; steering only on drift. The "why" behind `observation_only`. |
| **intervention surface** | Any code that cancels / injects / mutates control state / refuses a dispatch / installs a plan / acks with goldfive-authored state. Must be gated. |
| **guardrail** | An always-armed, observed-fact detector (loops, stalls, budgets). Cheap, structural, no goal reference. |
| **steering** | Goal-referenced, LLM-judged, engages only on drift; the expensive/fallible half; gates on `observation_only`. |
| **the ladder** | `InterventionLevel`: NUDGE(2) → CANCEL_REINVOKE(3) → PAUSE_ESCALATE(4) → TERMINATE(5). Levels 3–5 are structural. |
| **refine** | The planner producing a revised plan in response to drift; runs even in `observation_only` (dry_run), only *installs* in active mode. |
| **the twin pipelines** | `_handle_drift_dispatch` and `_promote_drift_to_steer` — two near-identical refine dispatch paths that must be edited together (Section 3.1). |
| **drift condition** | A drift as a stateful lifecycle (OPENED/ESCALATING/RESOLVED/HUMAN_INTERVENTION_REQUIRED), not a one-shot event (#271/#318/#486). |
| **stable key** | A gate key made of non-churning structured fields (`(kind.value, task_id)`, sha1 of stable fields, full agent path) — never `drift.id`. |
| **synthetic drift** | A framework-authored drift marked so harmonograf filters it from the operator's interventions view (#302). |
| **verdict-utility ledger** | Per-session `{acted_on, emitted_late, emitted_redundant, parse_fail}` accounting of whether judges earn their cost (#483). |
| **the branch** | `agency-preservation` (unmerged, #453–#474). Never copy from it; expect a four-file conflict at merge. |
| **13b** | The LOCKED final step: three-arm bench + measurement-gated default flips + hard deletions. Needs explicit sign-off. |
| **negative class** | The "no drift" / "capability OK" decision a detector must also emit (`emit_no_drift_decision`), so zicato can compute precision. |
| **dry_run** | A `PlanRevised` emitted under `observation_only` showing what the planner *would* have installed, without installing it. |

---

## Cross-references

- `09-steering-ladder-and-gates.md` — the surface-by-surface gate map and the twin refine pipelines in detail (the most safety-critical subsystem chapter).
- `04-executors-and-control.md` — the overlay loop, `TERMINAL_TASK_STATUSES`, the TERMINATE terminus.
- `07-deterministic-drift-detection.md` — tool-loop detection and the #484 corroboration cap.
- `08-llm-judges.md` — the judge scheduler, semaphore, and verdict-utility ledger.
- `11-state-ownership.md` — session state, the shallow-copy handoff hazard, `StateStore`.
- `12-events-sinks-telemetry.md` — decision telemetry, the negative class, the zicato manifest.
- `13-reporting-tools-and-approval.md` — `report_awaiting_approval` and the no-hang contract.
- `14-config-reference.md` — every default in the bench-frozen table, in full.
- `15-testing-guide.md` — the active-mode opt-in fixtures and the both-modes test pattern.
- `02-architecture-map.md` — the wrap→observe→detect→steer shape this chapter's rules protect.
- `10-planning-and-revision.md` — the planner, refine, and the no-cooldown directive.
- `16-recipes.md` — task-oriented recipes; this chapter's Section 1.7 is the invariant-critical subset.

---

## If you remember nothing else

1. **goldfive is dormant by default.** `observation_only=True` ships; a healthy run must feel zero goldfive footprint — not even a prompt directive (#394).
2. **Every mutation gates on one predicate, failing PASSIVE.** `steering_is_active(steerer)` / `is_active_steering()`. Never read `_observation_only`. Test both modes.
3. **Enforcement is structural; classification is LLM.** No cooperation contracts (Inv 1); no regex NL classification (Inv 2).
4. **Gates key on stable identity.** Never `drift.id`. A gate on a churning key is a silent no-op (Inv 6).
5. **Two refine pipelines, seven gates, protected surfaces, a locked branch.** Edit both pipelines; don't add an eighth gate; don't delete the KEEP list; don't copy from `agency-preservation`.
6. **The history is written down.** Read the PR message before editing (`git show <hash>`); the `#NNN` breadcrumbs in comments are load-bearing. Don't re-derive intent — look it up.

When these conflict with a task instruction, the invariants win, and you stop and ask (Section 2.7). This chapter is the contract; the subsystem chapters are the details.

**One last operational reminder.** You are reading this in a read-only worktree of ground truth. If your task asks you to *change* code:

- Work in an isolated `/tmp` worktree if other agents may be running (memory `workflow_goldfive_concurrent_agents`).
- Reproduce with steering OFF before blaming the model (Hazard 4.18, memory `feedback_empirical_baselines`).
- Ship tests + docs *with* the code, self-review the diff once more, then follow the auto-merge cadence for routine changes and stop-and-ask for the Section 2.7 list.

The invariants are not bureaucracy — they are the compressed memory of every expensive mistake this project already made so you don't have to make them again.
