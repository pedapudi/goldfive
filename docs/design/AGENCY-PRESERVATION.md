# Agency Preservation: from always-on controller to dormant supervisor

Status: **IMPLEMENTED** on the `agency-preservation` branch (PRs #453–#472),
**default-OFF** pending the §6.4 bench gate (PR 13b). The roadmap below
(§§0–5) is preserved as the original design rationale; **§6 records the
as-built status, the deliberate deviations from this roadmap, what is still
default-OFF, and the 13b pre-flip checklist.** Read §6 first for "what
actually shipped."

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

> **Implementation status:** Stages 1–3 (PRs 1–12) and their fast-follows
> are **IMPLEMENTED and merged** on the `agency-preservation` branch
> (PRs #453–#472), default-OFF pending PR 13b. Stage 4 is unbuilt
> (exploratory). The per-PR text below is the original plan; see **§6** for
> the as-built status table, the deliberate deviations from this plan
> (each with its PR), and the 13b pre-flip checklist. PR 13b
> (measurement + default flips + hard deletions) has **not** started.

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
*Correctness requirements (§5.2):* this PR ships with (a) an explicit
inventory of every guarantee the unconditional cancel currently provides
— loop-break for the v15 scenario, plan coherence for reporting, the
restart boundary that nudge delivery relied on, supersede-flag
consumption in the executor's cancelled branch — plus a test per item
demonstrating what supplies that guarantee in the new regime; and (b) a
pinned v15 regression test: slow refine + still-generating coordinator
must not loop the refine.

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

*Correctness requirements (§5.3):* before any cell changes, land a
decision-table snapshot test over the full
`(kind, severity, occurrence, config) → action` surface so this PR's
review shows every cell change as an explicit table diff — no silent
collateral edits to unrelated rows. Verify demoted kinds that no longer
refine also no longer write `refine_outcomes` entries that other gates
read.

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
*Correctness requirements (§5.4, §5.5):* the dry-run events must carry
enough payload to diff "what the legacy regime would have done" against
"what the new regime would do" on the same real runs; the exit criterion
for enabling any later behavior PR is a reviewed divergence report over
real traffic. `SignalLedger` keys use goldfive-minted stable task ids,
never LLM-minted identifiers (churning keys make lifecycle gates never
engage); ship hypothesis-based interleaving tests for the ledger
(concurrent judge verdicts, late drifts, user steers).

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
*Correctness requirements (§5.2):* an exactly-once delivery contract test
across all delivery points — a note enqueued while an invocation is
mid-flight and ALSO present at the next boundary must render once, not
twice (the classic two-mode double-delivery bug); marker strip-and-refresh
idempotency tests (two consecutive `before_model` calls never stack
blocks).

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
*Correctness requirements (§5.3):* the PR-3 decision-table snapshot is
re-baselined here with the full diff in the PR description; the deep test
coverage on the executor's supersede/cancel branches
(`sequential.py:1023-1110`) is migrated deliberately — each deleted
assertion is either re-pointed at the new behavior or justified in the
PR, never dropped silently.

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

## 5. Correctness strategy

This roadmap touches the most entangled code in the repo —
`drift_observer.py`'s dispatch path is ~2,000 lines of interacting gates
(verdict freshness #245, in-flight refine registry #405, outcome gates
#215, suppression #441, late-drift tolerance #319) — and PR 1 removes a
behavior that is *load-bearing* (the unconditional cancel was the v15
concurrent-invocation fix and feeds the executor's supersede contract).
The risk concentrates in three places: PR 1/7 (dispatch + cancel
semantics), PR 3/7 (ladder cell changes silently altering what other
gates read), and the Stage 2–3 two-mode period (legacy/new forks doubling
branch surface; double-delivery is the classic bug shape). Defense in
depth, eight layers:

### 5.1 No-op by default; one-line revertible flips

Every behavior change lands behind a flag whose default preserves legacy
behavior (`signal_channel` legacy in PR 6, `plan_mode=forecast` through
Stage 2, `GOLDFIVE_STEER_LEGACY_LADDER`, the PR-1 env kill-switch).
Default flips are separate one-line PRs. Existing test suites (~195
files) must keep passing UNMODIFIED until the cutover PR — a refactor
that breaks one is signal, not churn to be fixed up in place.

### 5.2 Invariant contract tests written before the code changes

The guarantees currently *implicit* in the cancel-everything design are
written down and tested so they hold in both regimes:

- every installed plan revision is eventually observed by the executor;
- no queued nudge/note is dropped, and none is delivered twice across
  channels;
- `session._supersede_pending` is always consumed or provably harmless;
- USER_STEER is honored within N logical turns regardless of regime;
- run termination still occurs without agent cooperation (no
  prompt-contract regression).

PR 1 additionally ships the cancel-guarantee inventory and the pinned
v15 regression test (see its entry). PR 6 ships the exactly-once
delivery contract.

### 5.3 Decision-table snapshot tests

`_ladder_level_for` is a pure function
`(kind, severity, occurrence, config) → action`. The full table is
snapshotted in both regimes so PRs 3 and 7 surface every cell change as
a reviewable diff — no silent collateral edits. Side-effect coupling is
checked alongside: demoted kinds must not leave stale `refine_outcomes`
writes that other gates consume.

### 5.4 Shadow / differential validation before authority

The reason PR 5 (telemetry) precedes every behavior change: under
`observation_only`, the NEW decision logic runs dry against real traffic,
emitting `SignalDelivered(dry_run=true)` with full decision payloads.
We diff legacy-would-do vs. new-would-do on the same real runs and
review divergences before either regime is enabled. The new code path
accumulates production mileage with zero production authority — the same
trick this roadmap applies to goldfive itself (observe before steering).

### 5.5 Golden traces and property-based interleaving tests

The JSONL sink already records full event sequences; recorded traces
from real sessions — including the 2d27ff4a refine-storm shape — become
deterministic dispatch-level regression fixtures. New concurrent state
(`SignalLedger`, `ObserverNoteQueue`) gets hypothesis-based interleaving
tests (concurrent judge verdicts, late drifts, user steers, restarts) —
this codebase's race history (#405 dedup registry, the per-session plan
lock, growth dedup linearisability) says concurrency is where its bugs
live.

### 5.6 Integration disciplines (project scar tissue, applied per PR)

- Grep call sites for every new gate/helper — unit tests pass on dead
  middleware that no dispatch path calls.
- Verify callback state handoffs are read-readable on the ADK side
  (`session.state` shallow-copy trap).
- Lifecycle-gate keys (SignalLedger, note dedup) use goldfive-minted
  stable ids, never LLM-minted identifiers — churning keys make gates
  never engage.
- Verify the running build (process start time vs. merge time) before
  interpreting any e2e result.

### 5.7 Layered e2e with functional pass criteria

Narrow regression checks are not accepted as validation (harnesses have
PASSED on functionally broken runs). Every stage gets the six-layer
pass — sanity → drive a real coordinator+AgentTool tree → DB → UI →
health → output — with *functional completion* and *steer-honoured* as
explicit criteria. The pinned 2d27ff4a replay asserts: zero
goldfive-authored in-flight cancels, no refine storms, exactly one
discovered task per unique (agent, args) pair.

### 5.8 Bench-gated flips and the residual risk

Arm (c) of the PR-13 bench keeps the legacy regime alive precisely so
regressions are measurable rather than argued about; flips being
one-line PRs makes rollback a 60-second operation. The honest residual
risk is emergent behavior in the two-mode period that no test
anticipates — that is what §5.4 and §5.5 exist for: shadow mileage on
real workloads before any new path gets authority.

## 6. As-built: implementation status, deviations, and the 13b gate

Stages 1–3 of this roadmap, plus their fast-follows, are **implemented and
merged on the `agency-preservation` branch** (NOT `main`). The roadmap
above (§§0–5) is the design rationale as authored; this section is the
honest as-built record: it marks what shipped, documents every deliberate
deviation from the roadmap (each a reviewed decision, with its PR), states
what is still default-OFF, and reproduces the PR-13b pre-flip checklist so
it survives outside the task tracker. Stage 4 remains exploratory (unbuilt).

### 6.1 Status by stage (PRs #453–#472)

| Roadmap item | PR(s) | Status |
|---|---|---|
| PR 1 — gate in-flight cancel on authority (`cancel_inflight_scope`) | #453 | merged |
| PR 2 — finish #423 (descriptive growth at pin time) | #454 | merged |
| PR 3 — ladder demotions + decision-table snapshot | #457 (+ #458 pin) | merged |
| PR 4 — intervention content rewrite (`observer_notes.py`) | #455 | merged |
| PR 5 — signal telemetry (`SignalDelivered`/`SignalOutcome`, `SignalLedger`) | #456 | merged |
| PR 6 — observer-note channel (`ObserverNoteQueue` + 4 surfaces) | #462 | merged |
| PR 6b — context-editing rules (finish #397) | #459, #463 | merged |
| PR 7 — cancel policy + ladder restructure (NUDGE→SIGNAL) | #467 | merged |
| PR 8 — pacing / grace windows (visibility-keyed) | #470 (+ #472 follow-ups) | merged |
| PR 9 — prompt-shaping diet (sites 1/3/4) | #466 | merged |
| PR 9 follow-up — agent-scoped note delivery + correction migration + cross-surface fold | #468 | merged |
| PR 10 — ledger plan mode foundations | (ledger line) | merged |
| PR 11 — goal-grounded judging | #464 | merged |
| PR 12 — refine retirement in ledger mode + `[GOALS]` block | #469 | merged |
| PR 13a — three-arm bench harness + shadow-diff tooling | (bench line) | merged |
| PR 13b — run bench, evaluate, **default flips** | — | **NOT STARTED** (§6.4) |

### 6.2 Still default-OFF (nothing flips until 13b)

Every new regime ships behind a flag whose **default preserves legacy
behavior** (§5.1). As of #472 the production defaults are unchanged:

- `SteeringConfig.plan_mode = "forecast"` (ledger mode opt-in)
- `SteeringConfig.signal_channel = "legacy_user_message"` (observer-note
  channel + pacing + the PR-9 diet + correction-via-notes are all
  `request_context`-only)
- `SteeringConfig.observation_only = True` (active steering opt-in)
- `SteeringConfig.signal_telemetry = False` (the §5.4 shadow campaign must
  enable it explicitly)

The full implementation accumulates production mileage with zero production
authority until the 13b bench gate flips these (each flip a separate
one-line, revertable PR).

### 6.3 Deliberate deviations & discoveries

Each item below is a reviewed decision that departs from, or was discovered
during, the roadmap as written. They are intentional — not drift.

1. **#208 outcome semantics — uncertain stays PENDING, not FAILED**
   (#464, PR 11). §3 PR 11 said OUTCOME tasks transition "unmet at exit →
   FAILED." As built: **met → COMPLETED; *confidently*-unmet → FAILED at
   run end only; uncertain → stays PENDING** and carries forward (the #208
   reachable-PENDING rule; run end is usually itself a turn boundary). No
   manufactured failures. Consequence: `run.success` can be `True` with
   OUTCOME tasks still PENDING — runs are graded on **goal predicates +
   OUTCOME-task terminality**, never on `run.success` alone (see the 13b
   grading rule, §6.4).

2. **Shared budget-row fix — PAUSE_ESCALATE-first in both regimes**
   (#467, PR 7). The budget/timeout guardrail kinds
   (`RESOURCE_EXHAUSTED`, `TOO_MANY_STEPS`, `TASK_TIMEOUT`,
   `LLM_CALL_TIMEOUT`) are **PAUSE_ESCALATE-first in BOTH the legacy and
   the new ladder** — "a restart can't refund a spent budget," so
   cancel-reinvoke was never the right response for them. `RUNAWAY_DELEGATION`
   stays cancel-first (it protects against unbounded fan-out, not a spent
   budget). Bench consequence: **arm C is the pre-PR-7 *steering policy*,
   not a byte-exact pre-PR-7 build** — re-introducing the budget-row bug
   into arm C would bias the B-vs-C comparison toward B (a confound). The
   `GOLDFIVE_STEER_LEGACY_LADDER` escape hatch restores the legacy ladder
   *cells + promotion side-effects only*.

3. **Two-path refine gate** (#469, PR 12). The roadmap cited only the
   ladder ABSORB/CANCEL_REINVOKE refine branch as needing the
   `plan_mode == "forecast"` gate. Implementation found the **promotion
   `refine_steer` path also needed gating** — it fires *before* the ladder.
   Both refine entry points are now gated on `plan_mode`.

4. **Force-FAIL contract — deterministic detectors only** (#469, PR 12).
   Ledger-mode force-FAIL is restricted to the **two deterministic looping
   detectors** (`LOOPING_TOOL_CALL`, `LOOPING_REASONING`). This is the dual
   of PR 11's "no judge-manufactured terminal state" (deviation 1): no
   terminal task disposition is ever minted by an LLM judge — only by a
   deterministic counter or a user/predicate.

5. **Agent-aware-surfaces principle** (#468). Agent-specific notes (notably
   per-(agent, task) corrections) render **only on agent-aware surfaces**:
   the ADK `before_model` surface (resolves the running agent) and the
   claude surface (one agent per invoke, scoped to `task.assignee_agent_id`).
   The boundary-replay surface is **broadcast-only** (it re-invokes at the
   coordinator level and cannot attribute to a sub-agent); the tool-result
   annotation **excludes correction-origin notes** (keyed on the shared
   `CORRECTION_DRIFT_ID_PREFIX`). Governing rule: **"better undelivered
   than misdelivered"** — an agent-specific note that finds no agent-aware
   surface stays pending, and under PR-8 visibility-keyed attribution a
   drift that resolves anyway is correctly recorded `self_corrected_unaided`
   (the truthful record), whereas misdelivery to the wrong agent would be a
   silent correctness bug. This realises surface (2)/(3) of §2's
   least-invasive-surface ordering for the correction case.

6. **The Site-1 "tool-surface tightening interceptor" never shipped**
   (#466, PR 9 — closes the doc's open verification item). §1.3 / the
   PromptShaper docstrings described a parallel ADK-plugin "pre-dispatch
   interceptor," keyed off `session._conversational_turn`, that would
   tighten the tool surface on a conversational follow-up so the
   coordinator literally could not delegate. **Verified: it was never
   built.** `_conversational_turn` is consumed *only* by the runner's own
   wrap gating (it decides whether to call
   `wrap_conversational_input`); no ADK-plugin consumer exists. The stale
   docstring/comment claims were deleted; the flag is kept (load-bearing
   for the wrap gating). The PR-9 diet therefore *informs* on a follow-up
   (keeps the plan context, drops the "do NOT call any AgentTool" command);
   it never structurally constrains the tool surface.

7. **Emission vs. visibility split** (#462 + #470, PRs 6 + 8).
   `SignalDelivered` fires at the **dispatch decision point**
   (`DriftObserver._route_corrective_note`) — it is the §5.4 *decision*
   record ("what the regime decided to do"), emitted once, including the
   `dry_run=True` shadow form under `observation_only`. The note queue's
   `delivered_turn` / `delivered_surface` is a **separate *visibility*
   record** ("when/where a note actually reached the model"). PR 8's grace
   windows and the `self_corrected_after_signal` attribution key on
   **visibility**, not emission. Pacing is **`request_context`-only** —
   the legacy channel stays unpaced, which keeps bench arm C's policy pure
   (deviation 2).

### 6.4 PR-13b pre-flip checklist (the bench gate)

PR 13b — run the three-arm bench on real workloads, produce the
non-inferiority evaluation, then (gated on results **and explicit user
sign-off**) flip `plan_mode=ledger` and `observation_only=False` as
separate one-line PRs, followed by the §3 PR-13 hard deletions + the
sibling-doc updates (PLAN-LIFECYCLE.md, PLAN-DESCRIPTIVE-GROWTH.md,
DRIFT.md, CONTROL-CHANNEL.md, RATIONALE.md). **Do NOT start without bench
results and explicit user sign-off on the flips.** Requirements
accumulated during implementation (reproduced here so they survive outside
the task tracker):

1. **Grade ledger runs on goal predicates + OUTCOME-task terminality, NOT
   `run.success`** (bench, #464 review). Uncertain outcomes legitimately
   stay PENDING across turn boundaries (#208 carry-forward; deviation 1),
   and a force-FAILED looping DISCOVERED task can fail the run via the
   fatal gate.

2. **Stage-3 layered e2e must assert "ledger-mode runaway → clean PAUSE"**
   (#469 review; shipped as the §5.7 runaway e2e). The
   no-hang/no-silent-death contract is an *integration* property (executor
   pause-block + ledger plan structure + run-end disposition) that unit
   tests cannot show — §5.7 scar tissue: narrow criteria pass on broken
   runs. Shipped standalone as `tests/test_ledger_runaway_e2e.py` (#471);
   keep standalone per bench's harness ruling — the canonical pause probes
   are `outcome.reason` + the HIR drift on the sink + OUTCOME-stays-PENDING.

3. **The §5.4 shadow campaign must enable `signal_telemetry` explicitly**
   (channel, #462) — it defaults OFF (§6.2).

4. **Bench arm definitions** (#464/#467/#469 reviews):
   - **A (baseline)** = `judge_only` counterfactual.
   - **B (signal regime)** = `signal_channel=request_context` + the new
     ladder + `plan_mode=ledger` + `observation_only=False`.
   - **C (legacy)** = `GOLDFIVE_STEER_LEGACY_LADDER=1` +
     `observation_only=False` — the pre-PR-7 *steering policy*, NOT a
     byte-exact pre-PR-7 build (deviation 2: re-introducing the budget-row
     bug would confound B-vs-C).

   The flips proceed only when arm B is non-inferior to arm A on goal
   success and not worse on turns/tokens beyond the agreed margin across
   ≥2 tree shapes (§4 / §5.8).
