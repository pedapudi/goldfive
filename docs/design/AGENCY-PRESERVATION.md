# Agency Preservation: from always-on controller to dormant supervisor

Status: ROADMAP (not yet implemented)

## 0. What goldfive is

goldfive is neither an overlay nor a controller. The "overlay, not
controller" slogan (goldfive#141) describes a *mechanism* — wrap the tree,
observe via callbacks, don't drive per-task — and says nothing about
purpose. goldfive's purpose is to provide **steering and guardrails if
things drift**: a *dormant supervisor*. Lane-keep assist, not autopilot
and not a dashcam.

That identity decomposes into three behaviors with different contracts:

- **Dormant (the steady state)**: while the wrapped agent is making
  progress toward the user's goal, goldfive has ZERO trajectory footprint —
  no per-turn prompt injection, no grading of delegations against a
  forecast, no preemption. Observation and event emission only.
- **Guardrails (always armed)**: hard limits on *observed facts* — tool
  loops, reasoning loops, stalls, budgets, runaway delegation, refusals.
  These need no plan and no judgment call; they are cheap, low
  false-positive, and legitimately always on. Their job is to *stop*
  runaway behavior, not to redirect it.
- **Steering (engaged only on drift)**: corrective influence when the
  trajectory diverges from the *user's goal*. This requires a reference
  and an LLM judge; it is expensive and fallible, so it engages
  conditionally, proportionally, and honestly attributed.

The dichotomy matters because guardrails and steering are different
products with different trigger conditions and different authority — and
the current implementation runs both through one machine (one drift
taxonomy, one ladder, one refine path), which is how a planning artifact
like CAPABILITY_MISMATCH ended up with the same enforcement machinery as
a genuine loop detector.

## 1. Problem statement

Measured against the dormant-supervisor identity, the implementation
fails on both sides of the "if things drift" condition:

- **When nothing is drifting, goldfive is not dormant.** Prompt shaping
  injects every turn; the reconciler grades every delegation against a
  forecast plan, *manufacturing* the drift signal that justifies
  engagement; plan bookkeeping writes preempt in-flight work. The system
  has a per-turn tax and a hair trigger.
- **When something is drifting, the response is not proportional
  steering — it is a wheel grab.** "Steer" today means: swap the plan,
  kill the in-flight invocation, restart the tree with goldfive's text as
  a synthetic user turn prescribing which task and which agent comes next.

This measurably damages the wrapped agent's trajectory and undermines its
agency. The fact that `SteeringConfig.observation_only=True` had to become
the production default (goldfive#254) is the symptom: the active half of
the product is currently too disruptive to leave on.

The four defects, verified in code:

### 1.1 Every plan write preempts the agent

`DriftObserver._cancel_inflight_for_revision` fires on **every**
drift-driven plan install — before the intervention-ladder level is even
consulted (`drift_observer.py:3850`), and from both `PlanReviser` install
paths (`plan_reviser.py:456`, `plan_reviser.py:620`), **including
`NEW_WORK_DISCOVERED` descriptive-growth installs**. Even a Level-1 ABSORB
("refine the plan and continue") and the agency-friendly ledger write of
goldfive#423 fire `task.cancel()` on the in-flight invocation within ~one
event-loop tick (`drift_observer.py:4642-4746`). The ladder docstring
("ABSORB: continue") is false in practice. This is the single largest
trajectory destroyer — larger than CANCEL_REINVOKE itself.

### 1.2 The Plan is a forecast the agent is graded against

`LLMPlanner.generate` predicts 5–20 tasks of agent behavior up front
(`planner.py:73-130`). The delegation pin tiers (`_adk_plugin.py:~4047`)
and CAPABILITY_MISMATCH Rules A/C then treat legitimate autonomous choices
— calling a different sub-agent, decomposing differently, doing un-forecast
work — as drift, triggering refines that frequently land near-identical
plans (goldfive#305). Real evidence: a coordinator delegated to
debugger_agent 20+ times and each delegation fired CAPABILITY_MISMATCH →
refine (e2e session 2d27ff4a, 2026-05-13). This violates the project's own
"adaptive, not predictive" principle (PLAN-DESCRIPTIVE-GROWTH.md §13).

Additionally, descriptive growth currently only rescues the Rule-C verdict
path (`_adk_plugin.py:4447-4473`), not the tier-3 fallback that
PLAN-DESCRIPTIVE-GROWTH.md §4.3 specifies — so tier-3 misattribution
(`_score_candidates_by_args`) still mispins, and Rule-A misfires bypass
growth entirely.

### 1.3 Interventions command means and arrive as user turns

Nudge replays re-invoke the whole tree with goldfive's corrective text as
the next user message (`executors/sequential.py:1354-1383`, composers at
`1940-2068`). The templates prescribe *means*: "proceed to
'{next_task_title}' via {next_task_agent}", "do NOT retry"
(`steerer.py:165-218`). Prompt shaping pins `[CURRENT ASSIGNED TASK]` into
every agent turn and the conversational follow-up wrap orders "do NOT call
any AgentTool" (`prompt_shaper.py:119-185`, `474-615`). The agent's own
judgment about decomposition, delegation, and retries — the core of its
agency — is overridden rather than informed.

### 1.4 Drift→steer promotion converts opinions into plan swaps

WARNING-severity judge verdicts auto-promote to a full steer (plan swap +
GOLDFIVE_STEER dispatch + deferred cancel) under the default threshold
(`drift_observer.py:4801-5241`).

## 2. Design principle: an explicit authority split

The dormant-supervisor identity (§0) is operationalized as an authority
split:

- **goldfive owns**: GOALS (what the user wants), BUDGETS/SAFETY (loops,
  stalls, refusals, runaway delegation, timeouts — the guardrails),
  OBSERVABILITY (sinks → harmonograf/zicato), and USER-authority relay
  (USER_STEER / USER_CANCEL / USER_PAUSE remain absolute).
- **The wrapped agent owns**: MEANS — decomposition, delegation, ordering,
  retries.
- **The Plan becomes a ledger**: goal-anchored OUTCOME tasks (what success
  looks like) plus a descriptively-grown record of what the agent actually
  did — not a forecast the agent is graded against. This is what makes
  "if things drift" measurable correctly: drift is divergence from the
  user's goal given the agent's own observed trajectory, not divergence
  from goldfive's upfront guess at how the agent would decompose the work.
- **Interventions become advisory observer notes**: honestly attributed,
  observation+goal content, delivered without destroying in-flight context.
  Control (cancel/pause/terminate) stays on the control channel and never
  depends on agent cooperation — the no-prompt-contract rule is preserved.
- **Guardrails and steering separate**: guardrail kinds keep hard ladder
  rows (always armed, observed facts, stop-not-redirect); steering kinds
  engage only on goal-referenced drift, through the note channel, with
  grace-window pacing.
- **Interventions prefer the least-invasive surface.** A supervisor at
  the callback layer has five surfaces, ordered from most to least
  trajectory-preserving: (1) **shape the choice set** — select among the
  agent's own candidate actions before commitment; (2) **edit
  perception** — subtract poisoned context so the model stops
  re-anchoring on it; (3) **speak** — attributed advisory notes;
  (4) **constrain structurally** — tools, budgets, decoding;
  (5) **stop** — pause / cancel / terminate. Historically goldfive used
  only (3)-as-command and (5). Stages 1–3 repair (3) and (5); Stage 4
  builds (1), (2), and (4).

The roadmap's stages map onto the identity: Stage 1 + the prompt-shaping
diet restore *dormancy* (zero steady-state footprint); the drift triage
keeps *guardrails* armed while demoting forecast-mismatch kinds to
observability; the observer-note channel and ledger plan mode rebuild
*steering* as conditional, proportional, goal-referenced influence; Stage 4
is the ambition layer — the techniques that make an active goldfive beat a
disabled one, which is exactly the bar the PR-13 counterfactual bench sets.

End state: `observation_only=False` becomes the default again only when the
new regime is measurably non-inferior to no-steering (judge-only
counterfactual baseline).

## 3. Roadmap

### Stage 1 — Stop the bleeding

**PR 1 — Gate the in-flight cancel on authority.**
`_cancel_inflight_for_revision` (`drift_observer.py:4642`) and its call
sites (`drift_observer.py:3850`, `plan_reviser.py:456,620`): cancel only
for user-authored drift (USER_STEER/USER_CANCEL) or hard safety (budget
exhaustion, runaway delegation, TERMINATE). Goldfive-authored revisions
install for bookkeeping; the in-flight invocation runs to completion.
Risk note: this changes semantics the v15 concurrent-invocation fix
depended on (a long refine overlapping a still-generating coordinator).
Until PR 6 lands, the verdict-freshness gate (goldfive#245) and
no-op-revision rejection bound the loop risk; ship with an env
kill-switch.

**PR 2 — Finish goldfive#423.**
Flip `descriptive_growth_enabled` default ON (`config.py`); move the growth
trigger from the Rule-C verdict path to the tier-3 fallback in
`_maybe_pin_delegation_task` per §4.3 (also fixes the Rule-A-bypass gap);
soft-retire Rule C behind an env flag; add a growth trigger from
`PlanReconciler`'s unmatched-agent branch (`reconciler.py:228-260`) so
transfer-style trees grow the ledger too.

**PR 3 — Ladder demotions** (`drift_observer.py:3114-3196`, legacy table
`steerer.py:175`). Forecast-mismatch kinds become observability-only
(DriftDetected still emits; no refine/steer):

| Kind | Change |
|---|---|
| PLAN_DIVERGENCE | OBSERVE at all severities (reconciler emitter already dead per #252; keep executor reachability audit + reporting tool) |
| CAPABILITY_MISMATCH Rule A | OBSERVE — stem-matching NL classification, the #166/#167 anti-pattern |
| CAPABILITY_MISMATCH Rule B | keep (user-declared `required_tools` is genuine intent) |
| NEW_WORK_DISCOVERED (agent-authored, `drift_observer.py:1180`) | reroute to `install_descriptive_growth` instead of refine |
| WRONG_AGENT | deprecate (no production emitter exists) |
| budget/safety set (LOOPING_*, TIMEOUT, REFUSAL, TOO_MANY_STEPS, …) | keep ladder rows |

**PR 4 — Intervention content rewrite** (no control-flow change).
Replace `_CORRECTIVE_TEMPLATES` / `compose_corrective_user_message`
(`steerer.py:165-248`) and `_compose_goldfive_steer_body`
(`drift_observer.py:5244`) with observation+goal composers in a new
`goldfive/observer_notes.py`: a factual observation ("`search_web` called
5× with identical args; results unchanged") + the user's goal + an explicit
"advisory; how to proceed is your decision" footer. No
`{next_task_agent}`, no "do NOT retry", no task-id commands. LLM judges
author the observation in the same call that produced the verdict (add
`note_to_agent` to `JudgeVerdict`; prompts in `drift/reasoning_judge.py`,
`drift/goals.py`); deterministic detectors render facts they already hold.
Preserve the neutral-framing lesson from `_correction_injection.py`; add
adversarial tests asserting rendered notes contain no imperative
means-verbs. When judge confidence is low, prefer *question-form* notes
("Does the current approach still serve the goal of X?") over statements —
self-generated corrections integrate into a trajectory better than
external ones, and a question is the lowest-footprint speech act
available. Statement form stays for hard observed facts.

### Stage 2 — Observer-note channel, cancel policy, pacing

**PR 5 — Telemetry first** (works under observation_only).
New events `SignalDelivered` and `SignalOutcome` (outcome ∈
self_corrected_unaided | self_corrected_after_signal | escalated |
user_intervened) + a `SignalLedger` keyed `(kind, task)` recording
grace-window bookkeeping without gating anything. Dry-run note emission
under observation_only establishes the agent self-correction base rate
*before* behavior changes. Additive protos only.

**PR 6 — The observer-note channel.**
`ObserverNoteQueue` (StateStore-backed) with per-request coalescing (≤1
rendered block per LLM request). Delivery points, in preference order:

1. ADK `before_model_callback` (`_adk_plugin.py:~5119`) via a new
   `PromptShaper.inject_observer_note` using the existing marker
   strip-and-refresh pattern from the plan-state hint. Reaches a
   *mid-invocation* agent on its next model call — removing the only
   remaining justification for cancel-as-information-delivery.
2. Invocation-boundary replay (existing scoped loop,
   `sequential.py:1354-1383`) consumes the queue instead of
   `pending_nudges`.
3. claude-agent-sdk adapter: system-prompt section + PostToolUse hook
   `additionalContext`.
4. Tool-result annotation for loop-shaped drift: an append-only,
   attributed reminder on the tool result itself —
   `[goldfive observer: this is the 5th identical call; the result has
   not changed]` — landing adjacent to the evidence at the moment of
   maximal relevance (the battle-tested system-reminder pattern). Append
   only, never modify the real result: the line between annotation and
   corruption is attribution plus preservation.

Rendered block shape:

```
[GOLDFIVE OBSERVER NOTE — from an external monitoring layer, not from the user]
Observation: <factual description of what was observed>
The user's goal: <from session.goals>
Status: <completed / open work snapshot>
This note is advisory. How to proceed is your decision; the user's
instructions remain authoritative.
[/GOLDFIVE OBSERVER NOTE]
```

Config `signal_channel: "request_context" | "legacy_user_message"`, default
legacy in this PR.

**PR 6b — Context-editing rules (finish goldfive#397): steering by
subtraction.**
PR 1 stops cancelling in-flight work, which means failed reasoning trails
*stay* in context — and models re-anchor on their own failed attempts;
that is largely what looping *is*. Context editing is the cancel gate's
natural complement: prune the poison instead of killing the invocation.
The `ContextEditor` skeleton shipped in #397 Phase 1 (drop-only invariant,
`PruneCancelledReasoningRule` registered); land the designed follow-up
rules — `PruneTransientErrorRule` (redact 429/parse-blip
function_responses), `PruneStaleSteerRule`, `CompactPriorReasoningRule`
(collapse N identical failed tool calls into one summarized entry). The
compaction rule requires relaxing Phase 1's drop-only invariant to a
*byte-monotonic replace* rule class (output ≤ input bytes and content
count, but summarize-in-place permitted); keep the `ContextEdited` /
`ContextEditRejected` event contract. Rules fire only on tripped
guardrail counters or drift verdicts — never on healthy turns
(dormancy). Files: `goldfive/context_editor.py`, `config.py`
(`context_editor_rules`), `_state_audit.py` authorised-sites constant,
CONTEXT-EDITING.md.

**PR 7 — Cancel policy + ladder restructure.**
Rename NUDGE → SIGNAL; replace every CANCEL_REINVOKE cell in
goldfive-authored rows with SIGNAL; repeat-escalation becomes
PAUSE_ESCALATE (stop-and-ask preserves trajectory; cancel-and-redirect does
not). CANCEL_REINVOKE survives only for the user-steer junction and hard
safety. Strip steering side-effects from `_promote_drift_to_steer`
(`drift_observer.py:4863+`): no cancel-reason tagging, no GOLDFIVE_STEER
dispatch, no `active_steer(source="goldfive")` stamp; keep refine_steer +
PlanRevised + note enqueue. `GOLDFIVE_STEER_LEGACY_LADDER=1` escape hatch
for one release.

**PR 8 — Pacing (minimum-intervention).**
Grace window: after a note for `(kind, task)` is delivered, that key cannot
re-signal or escalate for `grace_window_turns` logical turns (default 3, on
`Session._reasoning_turn` — the goldfive#441-correct clock); detectors keep
running; a key resolving inside the window records
`self_corrected_after_signal`. Escalation: second signal is re-authored
quoting the first; third occurrence → PAUSE_ESCALATE (reuses
REFINE_FAILURE_THRESHOLD semantics). Unify with the #441 user-steer
suppression window as ordered gates in the ledger.

**PR 9 — Prompt-shaping diet** (`prompt_shaper.py`).
Site 1 (conversational wrap): keep plan-summary context; delete "Do NOT
call any AgentTool"; verify whether the tool-surface-tightening interceptor
referenced in its docstring ever shipped (`session._conversational_turn` is
set at `runner.py:~960` with no apparent consumer) and delete whichever
exists. Site 3 (plan-state hint): fold into the observer note's Status
section; drop "Choose the agent whose tasks are still PENDING". Site 4:
remove the `[CURRENT ASSIGNED TASK]` pin by default (escape hatch
`pin_assigned_task` for trees built around it); pending corrections migrate
to the note queue; delete the instruction-mutation read path in
`adk_llm_instrumentation.py`. Site 2 (GoldfivePlanner attachment contract,
goldfive#153) stays. Net: one injection surface plus the planner contract.

### Stage 3 — Ledger plan mode

**PR 10 — Outcome plan mode (flagged, default off).**
Additive `Task.kind` (FORECAST | OUTCOME | DISCOVERED; the `discovered`
bool stays for wire compat) and `Task.contributes_to` in `types.py` +
`proto/goldfive/v1/types.proto`. `SteeringConfig.plan_mode: "forecast" |
"ledger"`. In ledger mode `LLMPlanner.generate` produces 1–5 goal-anchored
OUTCOME tasks (deliverables, not behavior forecasts — a new short prompt
replacing the "5 to 20 tasks" directive); no edges unless the goals
themselves are ordered; all means-level structure arrives as DISCOVERED
tasks via the existing growth machinery; pin tiers bypassed (dedup-hash →
grow → pin). `PlanSubmitted`/`PlanRevised` keep firing — reporting
*improves*: an outcomes lane plus an actual-trajectory lane, and a cleaner
zicato signal (outcome completion rate, loop/budget drift — observed facts,
not forecast-match noise). StaticPlanner users keep forecast semantics — a
hand-authored plan is genuine prescriptive intent.

**PR 11 — Goal-grounded judging.**
(a) `classify_goal_drift` (`drift/goals.py:187`): graduated severity
(progressing / uncertain bands) so WARNING can fire before CRITICAL; feed
it the ledger as the trajectory summary; it becomes the primary steering
trigger. (b) Reasoning judge re-grounded: goals primary, bound task as
context (in ledger mode the bound task is the agent's own observed intent —
self-consistent, adaptive). (c) New `drift/outcome_progress.py`: an LLM
judge at task-boundary cadence (reuse
`_maybe_run_goal_drift_on_task_boundary`) + run end that transitions
OUTCOME tasks {met → COMPLETED, unmet at exit → FAILED} and stamps
`contributes_to`; user-supplied goal predicates remain authoritative; the
goldfive#447 full-output capture is its evidence. Run completion = all
outcomes terminal — decided by goldfive's judges, no agent cooperation
required.

**PR 12 — Refine retirement in ledger mode.**
Gate the drift-triggered forecast-repair refine branch
(`drift_observer.py:3535+`) on `plan_mode == "forecast"`. In ledger mode
the same rungs: mark the bound ledger task FAILED where applicable, enqueue
a note, or pause. The looping deterministic fallback reduces to force-FAIL
(there is no forecast to route around). Refine survives for exactly three
authors: USER_STEER, turn-level `handle_turn` replans, and descriptive
absorption. `_compose_instruction` renders a `[GOALS]` block instead of the
task block for DISCOVERED pins.

**PR 13 — Measurement + default flips + hard deletions.**
Three-arm bench (extend `bench/run_100_tasks.py`): (a) `judge_only=True`
counterfactual baseline (goldfive#446), (b) new SIGNAL regime with
`observation_only=False`, (c) legacy regime via the escape hatch. Metrics
from goldfive#447 captured artifacts: goal-predicate success, turns/tokens,
intervention count, post-signal drift re-fire rate, abort rate, and the
PR-5 self-correction base rate. Flip `plan_mode="ledger"` default, then
`observation_only=False` (each a one-line revertable PR) only when arm (b)
is non-inferior to arm (a) on success and not worse on turns/tokens beyond
an agreed margin across ≥2 tree shapes. Then delete: Rule A/C, the tier-3
scorer (`_score_candidates_by_args`), `_PLAN_DIVERGENCE_SYSTEM_PROMPT` +
the observed-actions refine path, and the legacy ladder/channel values.
Update PLAN-LIFECYCLE.md, PLAN-DESCRIPTIVE-GROWTH.md, DRIFT.md,
CONTROL-CHANNEL.md, RATIONALE.md alongside.

### Stage 4 — Beyond repair: the ambition layer (exploratory)

Stages 1–3 are the *repair* layer: they stop goldfive from harming
trajectories and fix the drift reference. Stage 4 is what makes an active
goldfive measurably better than a disabled one. Each item ships
individually flagged, default-off, and must beat the judge-only baseline
on the PR-13 three-arm bench before its flag flips. No fixed PR numbers —
sequence by what Stage 1–3 telemetry shows is the binding constraint.

**4a — Fork-and-judge selection at commitment points** (surface 1:
shape the choice set). At a high-stakes juncture — the first model call
after a drift verdict, an irreversible tool call, a major delegation —
`before_model_callback` samples K candidate completions *from the agent's
own distribution* (re-issue the request at temperature), a judge selects
the goal-aligned candidate, and the synthesized-`LlmResponse`
short-circuit (already available in the ADK plugin surface; see
CONTEXT-EDITING.md "ADK surface") returns it. Not one goldfive-authored
token enters the context; every candidate is the agent's own intention;
the agent never knows. This converts steering from post-hoc correction to
pre-commitment selection — test-time compute scaling applied at the
orchestration layer. Cost is K× on only the calls where it triggers;
cheap on a self-hosted serving stack. The only Stage-4 technique with
literally zero context footprint.

**4b — Intent-self-reference drift** (the strongest legitimate
reference). The ledger records observed *actions*; the goldfive#271
machinery already captures stated *intent*. Combine them: treat the
agent's own emitted commitments as a co-equal drift reference and detect
*self-inconsistency* — "the agent said it would verify before deleting;
it is deleting without verifying." Lower false-positive rate than
goal-judges (no interpretation gap), and maximally agency-respecting:
the note quotes the agent back to itself. Feeds the reasoning judge as a
third grounding alongside goals and the ledger.

**4c — Structural guardrails on the tool surface** (surface 4). When a
loop detector trips on tool T with identical args — an observed fact —
`before_model` drops T from `config.tools` for one call, or forces a
text-only "exhale" turn (no tools) so the model must articulate state
before acting again. Honest, structural, zero cooperation required.
Discipline: fires ONLY on tripped guardrail counters; narrowing the tool
surface on healthy turns is the "do NOT delegate" mistake Stage 1
deletes. Same family: generalize the existing `_ratchet_max_tokens`
mechanism into graduated per-task budgets (tool-call and token) that
tighten as guardrail counters accumulate — a ramp instead of the binary
pause.

**4d — Decision-theoretic gate → learned intervention policy.** The
grace window (PR 8) is a fixed-delay heuristic. The principled gate:
intervene iff `P(real drift) × cost(unchecked drift) >
expected(intervention damage)` — and PR-5 `SignalOutcome` telemetry
estimates *both* sides empirically (self-corrected_unaided rate ≈ how
often intervention was unnecessary; post-signal re-fire rate ≈ how often
it didn't help). Long-term this is zicato's job: a contextual bandit
over {wait, signal, escalate}, learned offline per tree shape and model,
closing the loop the telemetry opens.

**Considered and rejected:** activation steering / representation
engineering (self-hosted-only, research-grade fragile) and logit-biasing
a looping tool's name tokens (brittle under tokenization and paraphrase).
Worth taking from the decoding layer: guided/constrained decoding for the
*judges'* structured outputs on self-hosted stacks — cheaper,
schema-reliable verdicts, which matters once judges gate interventions.

## 4. Verification

- Per PR: unit tests; forecast mode stays default through Stage 2 so
  existing suites pass unmodified; ledger tests are new files. High-churn
  suites: `test_intervention_ladder.py`, `test_delegation_pin.py`,
  `test_capability_mismatch.py`, `test_descriptive_growth_*`,
  `test_drift_taxonomy.py`, reconciler/overlay tests.
- Integration discipline: grep call sites for every new gate/helper — no
  dead middleware.
- E2E layered protocol (sanity → drive a real coordinator+AgentTool tree →
  DB → UI → health → functional-completion + steer-honoured checks).
  Specifically re-run the repeated-delegation scenario (2d27ff4a shape) and
  assert: zero goldfive-authored in-flight cancels, zero
  CAPABILITY_MISMATCH refine storms, ledger grows exactly one discovered
  task per unique (agent, args) pair.
- Counterfactual gate: the PR-13 three-arm bench is the only authority for
  default flips; PR-5 telemetry must show the self-correction base rate
  before any signal regime is enabled by default. Stage-4 items are held
  to the same bar individually: each flag flips only when its arm beats
  the judge-only baseline.
