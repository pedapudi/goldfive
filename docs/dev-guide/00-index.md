# 00. The goldfive Development Guide — Index

This is the developer's guide to the **goldfive** codebase: the internal
reference an agent (or a human) reads before making a nontrivial change to
goldfive itself. It is not the user-facing "how do I wrap my agent" material
(that lives in `docs/guides/` and `.agents/use-goldfive.md`). It is the
map you read when you are about to *edit* the observe → detect → intervene
pipeline and you need to know what the code actually does, why it is shaped
that way, and what will bite you.

Seventeen chapters, ~24k lines, each self-contained and grounded in the live
code on `main`. Every chapter opens with a "Read this chapter when…" block, a
"Files covered" list, and the invariants that bind edits in that area; every
chapter ends with a "Common mistakes" section (recipe chapters embed the
pitfalls per-recipe instead) and a "Verification checklist" of exact
grep/pytest commands.

---

## ⚠ Currency caveat: the agency-preservation merge

The **agency-preservation re-architecture is merged** (PRs #453–#504 via the
`agency-preservation` integration branch): interventions are advisory
**observer notes** (`goldfive/observer_notes.py`, `observer_note_queue.py`),
the ladder's Level-2 rung is **`SIGNAL`** (renamed from `NUDGE`; enum value 2
unchanged), corrective command templates are retired behind a deprecating
shim, a **plan-as-ledger** mode (`TaskKind`, `plan_mode`) and
**signal telemetry** (`signal_ledger.py`, `SignalDelivered`/`SignalOutcome`)
exist behind default-OFF flags, and in-flight cancellation is gated on drift
**authority** (`cancel_inflight_scope`). All new-regime flags default OFF, so
default-config behavior descriptions below remain accurate — but chapters
**04, 09, 10, 12, 13, 14** predate the merge and describe the pre-merge
mechanics of nudges/corrective templates/ladder cells and omit the new
modules and config knobs. Until those chapters are refreshed:
**`docs/design/AGENCY-PRESERVATION.md` §6 is the authoritative as-built
record** for everything the merge changed; where this guide and that doc
disagree, §6 wins. The invariants in chapter 17 are unchanged and remain
binding (the kill-switch predicate gained siblings:
`steerer.signal_channel()` / `steerer.plan_mode_is_ledger()` — same
one-default fail-safe discipline).

## READ-FIRST rule

**Before any nontrivial edit, read [17-invariants-hazards-history.md](17-invariants-hazards-history.md).**

Chapter 17 is the constitution. It holds:

- The six **hard invariants** (no prompt-cooperation contracts; no
  regex/keyword NL heuristics; any ADK tree shape; adaptive-over-predictive;
  `observation_only=True` is strictly passive with one sanctioned kill-switch
  read; lifecycle gates need stable identity keys).
- The **Protected List** — KEEP decisions that must never be "cleaned up"
  without explicit human sign-off (LOOPING_TOOL_CALL machinery, PLAN_DIVERGENCE
  machinery, `reconciler.get_missed_tasks`), and the bench-frozen default flags.
- The **Deferred-Work Register** — features that are planned but NOT on `main`
  (twin-refine extraction, evidence-ledger, judge windowing, judge-facade
  authority, Stage-4 actuators). Do not document or build these as if they
  exist.
- The **Hazard Catalog** (18 recurring failure modes) and the **Pre-PR
  Checklist**.

If you skip chapter 17 you will, with high probability, either reintroduce a
retired anti-pattern (a keyword classifier, an unstable gate key, a
last-writer-wins global), break the passivity contract, or "fix" a deliberate
KEEP. Read it first. When you remember nothing else, remember: **the agent may
never cooperate — termination, control, and observability must work anyway.**

The second thing to keep open is [16-recipes.md](16-recipes.md): twelve
copy-pasteable, end-to-end procedures for the common extensions (add a
DriftKind, a detector, a judge, an intervention surface, a config knob, a proto
field, a sink, an observation point, a reporting tool, an adapter; safely
delete dead code; update a design doc). If your task matches a recipe, follow
the recipe rather than reconstructing the steps from the subsystem chapters.

---

## What goldfive is (one paragraph)

goldfive wraps an ADK agent tree (`goldfive.wrap`) to **observe** it, **detect
drift** (deterministic detectors plus an LLM-as-judge reading thinking tokens),
and **steer** it (nudge / steer / cancel-reinvoke / pause-escalate / terminate)
— **without requiring the agent to cooperate**. It sits between the runner and
the agent, watching events and reasoning, and it can intervene even if the
agent never calls a goldfive tool or follows an instruction. Its two ecosystem
siblings are **harmonograf** (the observability UI that ingests goldfive's
events) and **zicato** (an offline meta-loop optimizer that reads goldfive's
telemetry plus `optimization/manifest.toml`). The production default is
`observation_only=True`: goldfive watches and records but does not act.

---

## Routing table — task → chapters

Read them in the order listed (first is the primary home; the rest are the
blast radius). Chapter 17 is implied before every row.

| I want to… | Chapters (in order) |
|---|---|
| Get oriented / first change ever | 01, then 02 |
| Understand the end-to-end data flow | 02, 03, 04 |
| Change the Runner or Conversation state | 03, 11, 16 |
| Change the sequential/parallel executor or control channel | 04, 03, 09, 16 |
| Touch the ADK plugin / callbacks / delegation pin | 05, 06, 11, 16 |
| Add or change an adapter | 06, 05, 16 (Recipe 10) |
| Add a deterministic drift detector | 07, 16 (Recipe 2), 15, 17 |
| Add or change an LLM judge | 08, 07, 16 (Recipe 3), 15 |
| Add a new DriftKind end-to-end | 16 (Recipe 1), 07 or 08, 09, 12, 15, 17 |
| Change the steering ladder / gates / passivity | 09, 17, 16 (Recipe 4), 15 |
| Add an intervention surface | 16 (Recipe 4), 09, 17, 15 |
| Change planning / revision / reconciliation | 10, 03, 11, 16 |
| Touch state ownership / session state | 11, 05, 16 |
| Add an event, payload, sink, or proto field | 12, 16 (Recipes 6 & 7), 15 |
| Add or change a reporting tool / approval flow | 13, 05, 16 (Recipe 9) |
| Add or change a config knob | 14, 16 (Recipe 5), 15 |
| Write or fix a test | 15, then the subsystem chapter |
| Understand the invariants / history / hazards | 17 |
| Safely delete dead code | 16 (Recipe 11), 17 |
| Update a design doc | 16 (Recipe 12), 17 |

---

## Chapter list

Each summary is drawn from the chapter's own front matter and section list.

**[01. Orientation](01-orientation.md)** (964 lines) — Why goldfive exists (you
must steer agents that will not cooperate), the observe → detect → intervene
pipeline at a glance, the three-repo ecosystem, the no-cooperation contract
stated concretely, the vocabulary, the full 41-member DriftKind taxonomy with
producer/no-producer annotations, the intervention ladder at a glance, two
worked traces, what is deferred, a guided tour of where to find what, how to
read the history, and your first change. Start here if goldfive is new to you.

**[02. Architecture Map](02-architecture-map.md)** (1202 lines) — The
end-to-end data flow. `wrap()` construction order, a healthy turn, a drifting
turn, teardown, the trajectory-level GOAL_DRIFT path, the contrast with the
legacy per-task loop, how to read one run's event stream, and the single
control-channel junction where all steering converges. The cross-component
contract table and the threading/async model live here.

**[03. Runner and Conversation State](03-runner-and-conversation.md)** (1133
lines) — What a Runner is and the six components it composes, every
construction knob, the per-session Conversation map and lock discipline, the
`_run_locked` phase pipeline, the single `_abort_turn` path (post-#489), plan
install routing, cross-turn Conversation state, and `run_streamed` / `resume` /
`new_conversation` / `close`. Notes that `resume` is replay-only (open
TODO #15) — it does not continue execution.

**[04. Executors and the Control Channel](04-executors-and-control.md)** (1601
lines) — The sequential and parallel executors and the one control channel.
The invocation contract, the legacy run loop, overlay mode and its stage
methods, the passthrough control race, pre-task/pre-stage control draining and
the pause loop, `goldfive/control.py` and `_control.py` dispatch, the pause
deadline and real TERMINATE (#482), shared helpers, the parallel DAG executor,
and the honest sequential-vs-parallel parity table (including the shipped
asymmetries).

**[05. The ADK Plugin](05-adk-plugin.md)** (1610 lines) — The densest chapter;
weak models struggle here. How the plugin is built, `SessionContext` and how
callbacks reach goldfive state, the callbacks one at a time, cooperative
cancellation, the delegation-pin tier system, capability-mismatch at
delegation, watcher spawn/cancel discipline, the reasoning-channel disarm
warning, the ONE sanctioned `observation_only` accessor, F1 acks and the F3
pre-dispatch redirect, request char-count vs consumed metrics, the
state-protocol key names, and the duck-typing hazard rules for touching ADK
internals.

**[06. Adapters and Instrumentation](06-adapters-and-instrumentation.md)** (1339
lines) — The `AgentAdapter` protocol (required vs optional), `auto_adapter` and
`goldfive.wrap` dispatch, `CallableAdapter` (the reference implementation),
`ClaudeAgentSDKAdapter`, `ADKAdapter` and the one-runner model, tree
augmentation (reporting tools, GoldfivePlanner, agent-tree walk), the overlay
invoke paths, where reasoning actually gets extracted, ADK event fan-out, the
request-side LLM instrumentation, the dynamic-instruction resolver and the #477
`inject_session_state` fix, the parity table, and the add-an-adapter checklist.

**[07. Deterministic Drift Detection](07-deterministic-drift-detection.md)**
(1444 lines) — The structural detectors that need no LLM. The shape of a
detector, tool-loop detection (with the #484 name-axis INFO cap), the reasoning
detectors, the embedding backends, capability-mismatch, the structural
classifiers in `drift/__init__.py`, the stall watchdog as a detector (#487),
the post-#490 registry surface, the boundary with the LLM judges, and
condition-lifecycle resolution (#486) — where `current_task_id` /
`current_agent_id` become the identity key.

**[08. LLM Judges](08-llm-judges.md)** (1619 lines) — The two judges. The
reasoning-drift judge pipeline, three-state classification with provenance and
JUSTIFIED_DEVIATION, the #480 attribution wire, the quiet-fail sentinel,
severity parsing and the malformed→INFO rule (#479), the one internal LLM-call
module `goldfive/_llm.py` (#491), judge scheduling (#483: semaphore,
coalescing, utility ledger), the goal-drift judge, the disarm and
endpoint-contention warnings, plan-revision snapshotting, the pluggable
`judges/` package, and what NOT to build or delete here.

**[09. Steering: the Ladder, the Gates, and the observation_only
Contract](09-steering-ladder-and-gates.md)** (1606 lines) — The router +
observer subsystem. `InterventionLevel`, the verbatim `_LADDER` table,
`handle_drift` and `_handle_drift_dispatch` entry/dispatch gates in order, the
promotion path, the twin refine pipelines (keep in sync), the `observation_only`
spec, pause escalation and TERMINATE (#482), drift-condition lifecycle (#486),
the #480 decision telemetry, config knobs, cancel internals, the
REPEATED_FAILURE counter, two worked traces, and a deep appendix set (escalation
emitters, symbol index, USER_STEER lifecycle, mode-by-mode wire behavior).

**[10. Planning and Revision](10-planning-and-revision.md)** (1463 lines) — The
frozen Plan/Task data model, supersession vocabulary and revision topology, the
`LLMPlanner` entry points, the `TaskStateMachine` sanctioned transitions,
`PlanReviser` install paths, `PlanReconciler` (observation → transition),
`GoldfivePlanner` the ADK `BasePlanner`, and CORRECT-kind correction injection.
Includes the deferred-work note (descriptive-growth default is still off).

**[11. State Ownership](11-state-ownership.md)** (1189 lines) — Who owns which
state surface. The `Session` dataclass, goldfive `Session.state['goldfive.*']`
via `state_store.py`, the typed `StateStore` handle, drift conditions on
session state, module-level registries, plugin-instance dicts, ADK
`session.state['goldfive.*']` and the state protocol, the shallow-copy handoff
hazard (the 8-hour lesson), the ContextVars in use, the state-ownership
contract and tripwire, the "where to add new state" decision tree, and
stable-key discipline.

**[12. Events, Sinks, and Telemetry](12-events-sinks-telemetry.md)** (1403
lines) — The `Event` envelope, the kept dual-envelope wart, the payload
inventory, `events.py` factories and `emit()`, the sink catalog, decision
telemetry as a spec, `RunAborted` lineage encoding (#482), the proto workflow
for adding a field or payload, and what harmonograf and zicato consume
(including the camelCase JSONL-sink contract).

**[13. Reporting Tools and Approval](13-reporting-tools-and-approval.md)** (1098
lines) — What a reporting tool is and the ten-tool inventory, why `task_id` is
hidden from schemas, registration to every reachable agent, per-tool handler
flow, pin resolution and freshness (the #266 classifier), response shapes,
`report_awaiting_approval` end-to-end (post-#478: never hangs), the
no-cooperation tension (these tools are strictly optional), and tool-surface
cost honesty.

**[14. Config Reference](14-config-reference.md)** (1521 lines) — Every config
object and knob: the env-parse helpers (reuse them), `SteeringConfig`,
`AgentConfig`, `ReasoningDriftConfig`, `ToolLoopConfig`, `GoalDriftConfig`,
`EmbeddingConfig`, `JudgeConfig`, `wrap()` / `Runner` / executor / steerer
kwargs, low-level environment compatibility fallbacks, the `_llm.py` knob surface (#491), the
`GOLDFIVE_*` names that are NOT env vars, `optimization/manifest.toml` (the
zicato-facing inventory), precedence rules, the sign-off-gated frozen defaults,
and a worked "add a knob end-to-end" example.

**[15. Testing Guide](15-testing-guide.md)** (1195 lines) — The 30-second loop,
naming/placement/structure conventions, ADK-specific hazards, the
single-most-important post-#488 rule (**mode discipline**: no autouse fixture
flips the default; active-mode tests must opt in explicitly), the
`is_active_steering` silent-no-op trap, the harness toolbox, `importorskip`
conventions, how to test each artifact type, flaky-avoidance rules, two fully
worked examples, assertion-granularity discipline, and the coverage rule (a fix
PR must include a test that failed pre-fix).

**[16. Recipes](16-recipes.md)** (1951 lines) — Twelve end-to-end procedures:
add a DriftKind (1), a deterministic detector (2), an LLM judge (3), an
intervention surface (4 — the critical one), a config knob (5), a proto field
or payload (6), an event sink (7), an ADK-tree observation point (8), a
reporting tool (9), an adapter (10); safe dead-code deletion (11, the
archaeology protocol); update a design doc (12); plus the universal post-change
gate. Follow the recipe rather than reconstructing steps.

**[17. Invariants, Hazards, and History](17-invariants-hazards-history.md)**
(1443 lines) — **Read this first.** The six invariants with enforcement
vocabulary and per-file binding, the Protected List (KEEP decisions +
bench-frozen defaults), the Deferred-Work Register, the 18-entry Hazard
Catalog, the load-bearing history arcs (single-runner revert, overlay refactor,
structural steering, the #271 program, the steerer split, the #475-492
hardening program, the agency-preservation branch boundary), the Pre-PR
Checklist, a 32-row common-mistakes table, and the glossary.

---

## Maintaining this guide

The guide is a living reference, not a frozen document. The rules:

1. **Code wins.** Where a chapter and the code disagree, the code is right and
   the chapter is stale — fix the chapter. Several chapters explicitly flag
   places where a *design doc* is stale relative to code; treat those the same
   way. Never "fix" the code to match the doc without a KEEP/sign-off check
   (chapter 17 §2).

2. **Update the chapter in the same PR that changes its subsystem.** If you
   change the executor control channel, update chapter 04 in the same PR. A
   guide that lags the code by even one PR starts eroding trust. The
   subsystem-to-chapter mapping is the "Files covered" block at the top of each
   chapter.

3. **Keep citations symbol-anchored.** Cite a function/class/constant by name,
   not a bare line number — line numbers rot on the next edit. Where a chapter
   gives a line number, it is paired with a durable symbol anchor; preserve
   that discipline when you edit.

4. **Do not document deferred work as current.** The Deferred-Work Register
   (chapter 17 §3) lists features that are planned or live only on the unmerged
   agency-preservation branch. Present them as known future work with
   rationale, never as behavior on `main`.

5. **Respect the four-block shape.** Every non-recipe chapter has: a front block
   (Read-this-when / Files covered / Invariants), a body, a Common-mistakes
   section, and a Verification checklist. Keep new chapters to that shape.

6. **Each chapter carries its own caveats.** The writers recorded what they did
   NOT independently verify (unread sibling files, quoted-not-rerun test counts,
   line numbers verified-at-writing-time). Those caveats are reproduced in the
   PR that introduced the guide; when you resolve one, delete the caveat.

The terse, task-shaped `.agents/*.md` skills point *into* this guide — they say
"do X, see chapter NN." When you extend a subsystem, update both: the skill's
pointer and the chapter's prose.
