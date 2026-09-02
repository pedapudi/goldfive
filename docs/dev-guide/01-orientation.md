# 01. Orientation: What goldfive Is and the Rules of the Game

## Read this chapter when…

Read this chapter when ANY of the following is true:

1. This is your first task in the goldfive repo. Read all of it before editing anything.
2. You are unsure what a term means (drift kind vs. condition vs. verdict, pin, overlay,
   absorb, nudge, corrective, session vs. invocation vs. run vs. turn). Jump to
   [Vocabulary](#vocabulary).
3. You need to know which file owns a behavior. Jump to
   [Guided tour: where to find what](#guided-tour-where-to-find-what).
4. You are about to add, remove, or reroute a `DriftKind`. Read the
   [DriftKind taxonomy](#driftkind-taxonomy-all-41-members) FIRST — several members are
   deliberately producer-less and several are protected keep decisions you must not "fix".
5. You are deciding whether a change belongs in goldfive at all, or in harmonograf/zicato.
   Jump to [The three-repo ecosystem](#the-three-repo-ecosystem).

**Files covered** (this chapter maps the whole repo at one sentence per file; deep dives
live in the sibling chapters listed per row):

- `goldfive/__init__.py` — the public API surface (what `import goldfive` exports).
- `goldfive/types.py` — `DriftKind`, `DriftSeverity`, `TaskStatus`, `Task`, `Plan`,
  `DriftEvent`, `Session`, `Goal` (the data model everything else moves through).
- `goldfive/convenience.py` — `goldfive.wrap()` / `goldfive.run()` entry points.
- `docs/design/VOCABULARY.md` — the historical glossary this chapter reconciles against
  the code (the code wins; divergences are called out inline below).
- Every other `goldfive/*.py` and package, at orientation depth, in the guided tour.

**Invariants that bind you here** (from the repo-wide canon; each is expanded in
17-invariants-hazards-history.md):

| # | Invariant | What it means in THIS chapter's scope |
|---|---|---|
| 1 | No prompt-cooperation contracts | Termination, control, and observability MUST work even if the wrapped agent never calls a goldfive reporting tool and ignores every instruction. Never make a correctness path depend on the agent doing what it was asked. |
| 2 | No regex/keyword heuristics for NL classification | `#166` retired `_GENERIC_VERB_PREFIX_RE`, `#167` retired `_FACTUAL_QUESTION_RE`. New natural-language classification must be an LLM classifier or designed away. Exact-equality/hash matching of STRUCTURED data (tool names, args hashes, ids) is allowed. |
| 3 | Any ADK tree shape must work | Including coordinator+`AgentTool`. If orchestration breaks on a tree shape, that is a goldfive bug, never a "wrong tree" problem. |
| 4 | Adaptive over predictive | Capture observed facts (events the agent authored); do not intercept at pin/dispatch time to predict what the agent will do. |
| 5 | `observation_only=True` is the production default and strictly passive | The ONLY sanctioned read of the kill-switch is `DefaultSteerer.is_active_steering()` or the module helper `steering_is_active(steerer)` in `goldfive/steerer.py`. Missing/None/raising resolves to PASSIVE. Never read `_observation_only` directly. |
| 6 | Lifecycle gates need stable identity keys | Never key a gate/cooldown/dedup on an LLM-minted or otherwise churning id; fix the churn upstream. |

## The problem: steering agents that will not cooperate

goldfive exists to answer one question: **how do you keep a multi-agent LLM system on
target when you do not control its prompts, its tools, or its willingness to follow
instructions?**

The setting: a user brings an arbitrary ADK agent tree — a single `LlmAgent`, a
coordinator that delegates via `AgentTool`, a deep `sub_agents` hierarchy, anything —
plus their own prompts. The tree may:

- loop forever on the same tool call,
- delegate endlessly instead of finishing,
- wander off-topic in its chain-of-thought,
- refuse, stall, confabulate, or silently skip work,
- and — critically — **never call any tool goldfive gives it and never follow any
  instruction goldfive injects.**

The naive fixes are all forbidden here:

- "Add a `report_done` tool and tell the agent to call it" — that is a
  prompt-cooperation contract (invariant 1). goldfive DOES ship reporting tools
  (`goldfive/reporting/`), but they are an optional accelerant, never the correctness
  path. The `PlanReconciler` (`goldfive/reconciler.py`) transitions tasks from
  *observed* agent activity even when no reporting tool is ever called.
- "Grep the output for 'I cannot'" as the primary refusal detector — regex/keyword NL
  classification is the anti-pattern this repo has retired twice (invariant 2). The
  surviving marker lists in `classify_refusal` (`goldfive/drift/__init__.py`) are a
  legacy, deliberately-conservative surface; anything NEW must be an LLM judge.
- "Predict which sub-agent the coordinator will pick and pre-wire the plan to it" —
  predictive interception (invariant 4). goldfive instead observes the delegation the
  coordinator actually made (`DelegationObserved`) and adapts the plan to it
  (plan-descriptive-growth, `Task.discovered`).

So goldfive's stance is: **wrap, observe, judge, and only then intervene — with a
graduated ladder, and with the whole intervention arm behind a default-ON kill-switch
(`SteeringConfig.observation_only = True` in `goldfive/config.py`).** In production
default, goldfive is a pure observer: it plans, watches, detects, and reports, but does
not mutate the run.

The one-call entry point:

```python goldfive/convenience.py
def wrap(
    agent: Any,
    *,
    planner: Planner | None = None,
    goal_deriver: GoalDeriver | None = None,
    executor: Executor | None = None,
    steerer: Steerer | None = None,
    sinks: list[EventSink] | None = None,
    control: ControlChannel | None = None,
    call_llm: CallLLM | None = None,
    model: str | None = None,
    max_task_invocations: int | None = None,
    plugins: list[Any] | None = None,
    runtime: RuntimeConfig | None = None,
    dynamic_instruction: bool = True,
    drift_self_reporting: bool | list[str] = False,
    judge_only: bool = False,
    llm_detector: Any = None,
    judge_call_llm_builder: Any = None,
    judges: list[Any] | None = None,
    disable_judges: Iterable[BuiltinJudge | str] | None = None,
    **legacy_kwargs: Any,
) -> Runner:
```

`goldfive.wrap(agent)` returns a `Runner` (`goldfive/runner.py`); `await runner.run(user_input)`
executes one run. `judge_only=True` runs the native tree with judges attached.
The default steerer emits judgement and drift evidence, then stops before the
intervention ladder. Planning LLM calls and drift-response actions therefore
stay out of the run. A one-task `StaticPlanner` still installs the framing plan
that the native agent executes.

## The pipeline at a glance: observe → detect → intervene

Every goldfive run is one instance of a three-stage pipeline. Memorize this shape; every
sibling chapter is a zoom-in on one box.

```
 user_input
     │
     ▼
 Runner.run()  (goldfive/runner.py)
     │  GoalDeriver.derive → session.goals
     │  Planner.handle_turn / generate → session.plan  (revision of Plan.empty seed)
     ▼
 Executor  (goldfive/executors/sequential.py — overlay model)
     │  ONE adapter.invoke_passthrough(user_input) per turn
     ▼
 ┌───────────────────────── OBSERVE ─────────────────────────┐
 │ ADKAdapter + _GoldfiveADKPlugin (goldfive/adapters/)      │
 │  callbacks: before/after agent, model, tool; thinking     │
 │  tokens via emit_reasoning; delegation pins; LLM timing   │
 │ PlanReconciler (goldfive/reconciler.py)                   │
 │  maps observed agent activity → task transitions          │
 └────────────┬──────────────────────────────────────────────┘
              ▼
 ┌───────────────────────── DETECT ──────────────────────────┐
 │ Deterministic detectors (goldfive/drift/):                │
 │  tool_loops, capability_check, classify_tool_error,       │
 │  classify_stop_reason, classify_confabulation_risk, …     │
 │ LLM judges over thinking tokens:                          │
 │  reasoning_judge (per-thinking-message), goals            │
 │  (trajectory-level GOAL_DRIFT), pluggable Judge protocol  │
 │  (goldfive/judges/)                                       │
 │        → DriftEvent(kind, severity, …)                    │
 └────────────┬──────────────────────────────────────────────┘
              ▼
 ┌──────────────────────── INTERVENE ────────────────────────┐
 │ DriftObserver.handle_drift (goldfive/drift_observer.py)   │
 │  suppression gates → _ladder_level_for(kind, sev, count)  │
 │  Ladder: 0 OBSERVE · 1 ABSORB (planner.refine) ·          │
 │          2 NUDGE · 3 CANCEL_REINVOKE · 4 PAUSE_ESCALATE · │
 │          5 TERMINATE                                      │
 │  EVERY mutating arm gated by is_active_steering()         │
 │  (observation_only=True ⇒ record decision, do nothing)    │
 └────────────┬──────────────────────────────────────────────┘
              ▼
 EventSinks (goldfive/sinks/): InMemory, Logging, JSONL, SQLite, gRPC
 → harmonograf (live UI) / zicato (offline optimizer) / tests
```

Key structural facts a weak model must not get wrong:

1. **The steerer is a facade of three components** (post-#410 split). The `Steerer`
   protocol (`goldfive/protocols.py`) exposes `steerer.tasks`
   (`TaskStateMachine`, `goldfive/task_state_machine.py`), `steerer.plans`
   (`PlanReviser`, `goldfive/plan_reviser.py`), and `steerer.drift`
   (`DriftObserver`, `goldfive/drift_observer.py`). Older docs say
   "`DefaultSteerer._handle_drift`"; on main the body lives in
   `DriftObserver.handle_drift` in `goldfive/drift_observer.py`. See
   09-steering-ladder-and-gates.md.
2. **The overlay model is the default execution model** (#141). The executor does NOT
   dispatch plan tasks one at a time; it invokes the tree once (passthrough) and the
   `PlanReconciler` maps what the tree naturally did onto the plan. Re-invocations
   happen only via the ladder (nudge / cancel-reinvoke) or multi-turn conversation.
   See 03-runner-and-conversation.md and 04-executors-and-control.md.
3. **Detection has two arms**: deterministic (structured-data matching — allowed) and
   LLM-as-judge (natural-language classification — the only sanctioned NL arm).
   See 07-deterministic-drift-detection.md and 08-llm-judges.md.
4. **Intervention is graduated and default-disarmed.** Under the shipped default
   (`observation_only=True`), `handle_drift` still computes the ladder level and emits
   full decision telemetry (`SteeringDecisionMade`), but every mutating arm no-ops.
   Since Waves 1–4 (#488) this passivity is strict: even the nudge path (#475) and the
   `LLM_CALL_TIMEOUT` cancel (#476) are gated.

## The three-repo ecosystem

goldfive is one of three sibling repos. Know exactly what crosses each boundary so you
never implement a feature on the wrong side.

| Repo | Role | Runs when |
|---|---|---|
| **goldfive** (this repo) | The wrapping/observation/steering runtime. Everything in-process with the agent tree. | Live, inside the user's process. |
| **harmonograf** | Observability UI + storage. Renders runs, plans, drift timelines, interventions; hosts the human control surface (pause/steer/cancel buttons, approval prompts). | Live, out-of-process. |
| **zicato** | Offline meta-loop optimizer. Reads goldfive telemetry across many runs and proposes changes to goldfive's steering prompts/thresholds. | Offline, between runs. |

### goldfive ⇄ harmonograf boundary

- **Out (goldfive → harmonograf):** proto `Event` envelopes streamed by `GRPCSink`
  (`goldfive/sinks/grpc_sink.py`) to a `GoldfiveIngress` server (reference
  implementation: `goldfive/server/grpc_server.py`). Plus goldfive-internal LLM call
  spans (`goldfive/_llm_span.py`). Session unification (#161): the ADK
  `ctx.session.id`, `goldfive.Session.id`, and harmonograf's home session id are the
  same string, stamped per-event as `Event.session_id`.
- **In (harmonograf → goldfive):** `ControlMessage`s (`goldfive/control.py`) over a
  `ControlChannel` bridge — `PAUSE` / `RESUME` / `CANCEL` / `STEER` / `REWIND_TO`,
  plus approval decisions for `report_awaiting_approval`
  (see 13-reporting-tools-and-approval.md).
- **Never crosses:** Python objects, `Session` state, plan dataclasses. Everything on
  the wire is proto (see 12-events-sinks-telemetry.md).

### goldfive ⇄ zicato boundary

- **Out (goldfive → zicato):** JSONL telemetry written by `JSONLPersistenceSink`
  (`goldfive/sinks/persistence.py`; serialized with `MessageToJson`, so JSON keys are
  camelCase — unlike `LoggingSink`, which uses
  `preserving_proto_field_name=True` and emits snake_case), plus the
  **optimization manifest** `goldfive/optimization/manifest.toml` — the
  source-of-truth inventory of every knob a downstream optimizer is allowed to mutate
  (prompt bodies under `goldfive/optimization/prompts/*.md`, numeric thresholds as
  `goldfive/<module>.py:<ATTR>` entries).
- **In (zicato → goldfive):** proposed values for manifest-listed targets only.
  zicato has NO runtime hook into a live goldfive run.
- Loader/validator: `goldfive/optimization/manifest.py`; coverage pinned by
  `tests/test_optimization_manifest.py` (a code-side default bump must update the
  manifest or that test fails). Manifest-liveness is additionally AST-checked (#487).

Rule of thumb: if a change renders pixels or stores history → harmonograf. If it tunes
prompts/thresholds across runs → zicato (goldfive only exposes the knob in the
manifest). Everything that must happen while the agent is running → goldfive.

## The no-cooperation contract: what it forbids concretely

Invariant 1 is the repo's identity. "No prompt-cooperation contracts" means: **goldfive
must deliver termination, control, and observability against an agent that behaves as
if goldfive does not exist.** Concretely:

| # | FORBIDDEN | REQUIRED INSTEAD | Where enforced today |
|---|---|---|---|
| 1 | Making task-completion detection depend on the agent calling `report_task_completed`. | `PlanReconciler` observes `before_agent`/`after_agent` callbacks and transitions tasks from observed activity; reporting tools are a faster optional path that converges at `steerer.transition`. | `goldfive/reconciler.py::PlanReconciler`; both paths dedupe at terminal absorption. |
| 2 | Terminating a run by instructing the agent to stop. | Generator-end termination plus a real `TERMINATE` ladder level with a built-in 600s deadline and `RunAborted` carrying escalation lineage (#482). | `goldfive/drift_observer.py` (ladder level 5), executor teardown. |
| 3 | Requiring the coordinator prompt to describe tasks in a goldfive-parsable format. | The planner adapts to whatever the tree does: delegation pins map observed `AgentTool` calls onto plan tasks; discovered tasks (`Task.discovered`, #423) are minted from observed delegations the forecast missed. | `goldfive/adapters/_adk_plugin.py::_GoldfiveADKPlugin._maybe_pin_delegation_task`; `goldfive/types.py::discovery_identity_hash`. |
| 4 | Blocking the run until the agent acknowledges a control message. | Control is executor-side: the executor drains the `ControlChannel` and acts (pause blocks the executor loop, cancel aborts the invocation) regardless of agent behavior. | `goldfive/executors/_control.py::dispatch_control`. |
| 5 | An approval flow that hangs when no channel/human exists. | `report_awaiting_approval` never hangs: no channel → immediate `'unavailable'` ack; finite default timeout 600s; expiry emits `HUMAN_INTERVENTION_REQUIRED` (#478). | `goldfive/reporting/handlers.py`. |
| 6 | Nudge text that lies about what happened to coax compliance. | Nudge text must be truthful (#475) — e.g. the GOAL_DRIFT corrective only asserts "already complete" when the plan actually shows COMPLETED (`_GOAL_DRIFT_NOT_COMPLETE_TEMPLATE` fallback in `goldfive/steerer.py`). | `goldfive/steerer.py::compose_corrective_user_message`. |
| 7 | Coupling drift detection to agent self-reports. | Self-report kinds (`SELF_REPORTED_STUCK`, `UNCERTAIN_PROGRESS`, `drift_self_reporting=` tools) are opt-in extras; the load-bearing detectors read observed events and thinking tokens. | `goldfive/drift/` detectors; `wrap(drift_self_reporting=False)` default. |

The practical test for any new feature: **"does this still work if the agent never calls
our tools and ignores our text?"** If the answer is no, the design is wrong for goldfive
core (users bring their own coordinator prompts; see the memory note
`feedback_no_prompt_contract`).

## Vocabulary

This section reconciles `docs/design/VOCABULARY.md` against the code on main. Where the
doc and the code disagree, **the code wins**; divergences are flagged with ⚠. Every term
below is load-bearing — plain-English readings will mislead you.

### Run / turn / session / invocation / conversation

These four nesting levels are the most commonly confused names in the repo.

| Term | One run of what | Identity | Anchor |
|---|---|---|---|
| **Run** | One `await runner.run(user_input)` call, end to end: goal derivation → plan → execution → `ExecutionOutcome`. Emits `RunStarted` … `RunCompleted`/`RunAborted`. | `Session.run_id` (uuid per run) | `goldfive/runner.py::Runner.run` |
| **Turn** | The same thing viewed from the conversation: each `run()` on the same `Runner` is one turn; state (goals, completed_results, prior plan) carries across turns. | `TurnRecord` appended per turn | `goldfive/conversation.py::Conversation`, `TurnRecord` |
| **Session** | Two related things. (a) `goldfive.Session` (`goldfive/types.py`): the mutable per-run state bag — `goals`, `plan`, `current_task_id`, `completed_results`/`completed_outputs`, `reasoning_history`, `pending_nudges`, `refine_outcomes`, approval waiters. Docstring: "Live state for one Runner.run() invocation." (b) The OUTER session id (ADK `ctx.session.id`) that pins a `Conversation` and stamps every `Event.session_id` (#161 session unification). `runner.run(..., session_id=...)` adopts the outer id. | `Session.run_id` + `conversation_id`; outer id via `session_id=` | `goldfive/types.py::Session`; `goldfive/runner.py::Runner.run` |
| **Invocation** | One dispatch of the agent tree: `adapter.invoke(task, session)` (legacy per-task) or `adapter.invoke_passthrough(user_input)` (overlay — ONE per turn, plus ladder-driven re-invokes). ADK mints `invocation_id`; cancellation requests, delegation caps, and LLM-call watchers are keyed per invocation. | ADK `invocation_id` | `goldfive/protocols.py::AgentAdapter.invoke`; `goldfive/types.py::CancellationRequest` |
| **Conversation** | The cross-turn container. A `Runner` owns a map of `Conversation`s keyed by outer-session id so cross-turn state never leaks between distinct outer ADK sessions sharing one Runner (#271 follow-up); programmatic unpinned callers share the `""` key. | outer-session id (or `""`) | `goldfive/conversation.py` module docstring |

Nesting: **Conversation ⊃ turns; each turn = one run = one `Session`; each run contains
one or more invocations.**

### The four semantic enums

Four StrEnums classify run state. They never share values; they *bridge*.

| Enum | Module | What it classifies | Emitted by | Consumed by |
|---|---|---|---|---|
| `ControlKind` | `goldfive/control.py` | Verbs an EXTERNAL controller (UI/CLI/tests) issues to a running Runner — plus two goldfive-internal routing kinds. | Bridges via `ControlChannel.send(ControlMessage)`; the steerer mints the `GOLDFIVE_*` kinds. | Executors drain `ControlChannel.receive()` and dispatch in `goldfive/executors/_control.py::dispatch_control`. |
| `DriftKind` | `goldfive/types.py` | Categories of observed divergence. A classification, not a verb. | Detectors, judges, state machine, adapters, reporting handlers (see taxonomy table). | `DriftObserver.handle_drift` → ladder; sinks receive `DriftDetected`. |
| `DriftSeverity` | `goldfive/types.py` | Ordinal urgency: `INFO` (0) / `WARNING` (1) / `CRITICAL` (2); ranks via `severity_rank`. | The producer of each `DriftEvent`. | `_ladder_level_for` picks the ladder column by severity. |
| `TaskStatus` | `goldfive/types.py` | One task's lifecycle position (seven states, four terminal). | Transitions owned by `TaskStateMachine.mark_task_*` (`goldfive/task_state_machine.py`). | Executors filter by it; reconciler drives it; sinks emit `Task*` events on transitions. |

### ControlKind: all 12 members on main

⚠ VOCABULARY.md §2 says "five `ControlKind` values"; main has **12**
(`goldfive/control.py::ControlKind`, kept in lockstep with
`proto/goldfive/v1/control.proto` by `tests/test_control_proto.py`):

| Member | Origin | Payload | Effect on main |
|---|---|---|---|
| `PAUSE` | external | — | Executor sets its paused flag and blocks its loop on `channel.receive()` until RESUME/STEER/CANCEL; `USER_PAUSE` drift synthesized at INFO for audit. |
| `RESUME` | external | — | Clears the paused flag. No drift. |
| `CANCEL` | external | — | Executor short-circuits to abort (`RunAborted`); `USER_CANCEL` (CRITICAL) synthesized for audit only — refine never runs. |
| `REWIND_TO` | external | `{"task_id"}` | Structural reset: downstream tasks re-marked PENDING inline in the control dispatcher. No drift, no planner decision. |
| `STEER` | external | `{"note", "suggested_action"}` | THE canonical live-replan trigger: bridged to `DriftKind.USER_STEER` (WARNING) in `DriftObserver.observe`, then refine. Idempotent by `annotation_id` / message id (#171). |
| `APPROVE` / `REJECT` | external | `{"target_id", "detail"}` | Resolve a pending approval waiter (`Session.pending_approvals`) registered by `report_awaiting_approval` (Flow A) or ADK `require_confirmation` (Flow B). See 13-reporting-tools-and-approval.md. |
| `STATUS_QUERY` | external | — | Read-only snapshot returned via the ack's `detail` field; emits NO drift. |
| `INTERCEPT_TRANSFER` | external | `{"enabled"}` | Toggles a session flag consulted by adapters that honour transfer interception. Not a `DriftKind`. |
| `INJECT_MESSAGE` | external | `{"role", "text"}` | Injects a message into the run's transcript path (executor-side). |
| `GOLDFIVE_STEER` | **goldfive-internal** | `{drift_kind, drift_id, body, superseded_task_ids, replacement_task_ids}` | Minted by the steerer's promotion path so goldfive-authored drift rides the SAME cancel-and-restart junction as a user STEER. External bridges must NOT originate it. |
| `GOLDFIVE_PAUSE_ESCALATE` | **goldfive-internal** | `{reason, drift_id}` | Ladder level 4: makes the executor block awaiting an operator verb; paired with `HUMAN_INTERVENTION_REQUIRED`. |

The one distinction that matters: **`ControlKind` is an imperative verb** (no severity,
no task context); **`DriftKind` is a categorized observation** (severity + task context
+ detail meant for the refine prompt). External verbs that need replanning are bridged
into `USER_*` drift kinds because everything downstream of `observe` is drift-shaped.

### TaskStatus: the seven-state task lifecycle

Canonical terminal set (single source of truth — never duplicate it):

```python goldfive/types.py
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.NOT_NEEDED}
)
```

| State | Terminal? | Meaning | Reached via |
|---|---|---|---|
| `PENDING` | no | In the plan, not started. | Every new task. |
| `RUNNING` | no | Reconciler observed the agent working on it, OR a reporting tool / executor marked it. | `mark_task_running`. |
| `COMPLETED` | yes | Success; agent-authored summary in `session.completed_results[task_id]`, full-fidelity output in `session.completed_outputs[task_id]` (zicato#12 — graders read `completed_outputs`). | `mark_task_completed` (reporting or reconciler-observed `after_agent`). |
| `FAILED` | yes | Failure; fires `TASK_FAILED_RECOVERABLE`/`_FATAL` drift alongside. | `mark_task_failed`. |
| `CANCELLED` | yes | User/system cancelled, or upstream cascade; `Task.cancel_reason` carries a structured tag (#205). | `mark_task_cancelled` / cascade. |
| `BLOCKED` | no | Waiting on something external; may return to RUNNING after a refine resolves the blocker. Dual-natured: the transition also emits `DriftKind.BLOCKED` (WARNING) so the plan can adapt. | `mark_task_blocked`. |
| `NOT_NEEDED` | yes | Overlay-only (#141/#163): PENDING task the tree never exercised, stamped at invocation end by the reconciler. Distinct from CANCELLED so sinks can render "tree chose not to run" differently. Parallel scheduler skips it like every terminal (#485). | `reconciler.get_missed_tasks` path (protected keep). |

Terminal states are absorbing: `mark_task_*` on a terminal task is a silent no-op, and
`Plan.validate(for_revision=True, prior=...)` rejects revisions that regress or drop a
terminal task (PLAN-LIFECYCLE.md §3.1/§3.2). One more structural rule worth knowing on
day one: no CANCELLED/FAILED/NOT_NEEDED → PENDING edges (a PENDING task behind an
absorbing predecessor is definitionally unexecutable — `Plan.validate` step 7, #137).

### Drift kind, drift event, condition, verdict

| Term | Meaning | Anchor |
|---|---|---|
| **Drift kind** | A *category of divergence observation* — `DriftKind` StrEnum in `goldfive/types.py` (41 members; full table below). A classification, not a verb: knowing the kind does not by itself dictate an action. | `goldfive/types.py::DriftKind` |
| **Drift event** | The in-memory record of one observation: `DriftEvent(kind, severity, detail, current_task_id, current_agent_id, raw, id, trigger_input, authored_by, observed_revision_index, detector_name, …)`. `authored_by` is `"user"` for `USER_*` kinds, `"goldfive"` for detector-minted drifts (normalized at `handle_drift` entry). `observed_revision_index` (#245) is stamped BEFORE any LLM await so stale verdicts can be dropped. `detector_name` (#480) disambiguates producers that share a kind (the tool-loop tracker deliberately emits `LOOPING_REASONING` with `detector_name="tool_loops"`). | `goldfive/types.py::DriftEvent` |
| **Condition** | The *lifecycle identity* of a recurring drift: multiple `DriftEvent`s about the same underlying problem collapse onto one stable `condition_id` (computed from kind + target — reader-centric identity, invariant 6), whose `lifecycle` field walks toward `DRIFT_LIFECYCLE_RESOLVED`. #486: task-terminal transitions and staleness-guarded on-task verdicts resolve conditions; `GOAL_DRIFT` conditions resolve only at task-terminal. | `goldfive/drift_observer.py` (condition stamping around `_ostate.compute_condition_id`); `goldfive/state_store.py` lifecycle helpers |
| **Verdict** | A judge's output. Two layers: (a) the pluggable `JudgeVerdict` dataclass (`goldfive/judges/base.py`) with four flavours — drift, rubric, boolean, numeric; (b) the reasoning judge's three-state classification of a thinking block — on-task / off-task (`OFF_TOPIC`) / `JUSTIFIED_DEVIATION` (`goldfive/drift/reasoning_judge.py`). Verdicts are judged against a **pinned snapshot** (#479) and pass the freshness gate before acting. Post-#483 each verdict's utility is ledgered (`acted_on` / `emitted_late` / `emitted_redundant` / `parse_fail`). | `goldfive/judges/base.py::JudgeVerdict`; `goldfive/drift/reasoning_judge.py` |

### The intervention ladder and its verbs

| Term | Meaning | Anchor |
|---|---|---|
| **Ladder** | `InterventionLevel` IntEnum (goldfive#142): `OBSERVE=0, ABSORB=1, NUDGE=2, CANCEL_REINVOKE=3, PAUSE_ESCALATE=4, TERMINATE=5`. `(kind, severity, occurrence_count)` maps to exactly one level via `DriftObserver._ladder_level_for` reading the `_LADDER` table (`goldfive/drift_observer.py`, populated lazily in `_load_ladder_tables`). ⚠ VOCABULARY.md §7 still describes a single `severity >= WARNING` refine threshold in `DefaultSteerer._handle_drift`; on main the ladder table in `DriftObserver` is the policy and the enum lives in `goldfive/steerer.py::InterventionLevel`. | `goldfive/steerer.py::InterventionLevel`; `goldfive/drift_observer.py::DriftObserver._LADDER` |
| **Observe** (level 0) | Emit `DriftDetected` + decision telemetry; take no action. The only level whose behavior is identical with steering armed or disarmed. | `goldfive/drift_observer.py` |
| **Absorb** (level 1) | Fold the drift into the plan: call `planner.refine(plan, drift, goals, …)` and install the revision — **without interrupting the in-flight invocation**. The agent is never told directly; it learns via dynamic instructions on its next turn. For `_ABSORB_NUDGE_KINDS` (`LOOPING_REASONING`, `LOOPING_TOOL_CALL`, `SELF_REPORTED_STUCK`, `GOAL_DRIFT` — `goldfive/steerer.py`), a successful absorb ALSO queues a nudge (#202) since a mid-invocation coordinator cannot otherwise learn its plan changed. | `goldfive/drift_observer.py::handle_drift`; `goldfive/plan_reviser.py::PlanReviser._apply_revision` |
| **Refine** | The planner verb behind absorb: `Planner.refine(*, plan, drift, goals, observed_actions=None, available_agents=None) -> Plan \| None` (`goldfive/protocols.py`). Returns a REVISED plan validated by `Plan.validate(for_revision=True, prior=...)`; `None` means "decline". Parse/validation exhaustion emits `REFINE_VALIDATION_FAILED` (never re-refined — infinite-loop risk, #133). A planner may raise `RefineExhausted` (`goldfive/steerer.py`) to escalate straight to `HUMAN_INTERVENTION_REQUIRED`. | `goldfive/protocols.py::Planner.refine`; `goldfive/planner.py::LLMPlanner` |
| **Nudge** (level 2) | Enqueue a short synthesized user message onto `Session.pending_nudges` (`goldfive/types.py`); the overlay executor drains the queue at invocation end and re-invokes the passthrough with it. It is a *suggestion the agent is free to ignore* — the softest injection. Gated by `is_active_steering()` (#475) and the text must be truthful. Dispatch: `DriftObserver._dispatch_nudge` (`goldfive/drift_observer.py`). | `goldfive/drift_observer.py::_dispatch_nudge` |
| **Corrective** | The message content used at level 3 (and for nudges): `compose_corrective_user_message(drift=, refined_plan=)` in `goldfive/steerer.py` picks a per-kind template from `_CORRECTIVE_TEMPLATES` — short, action-focused, no goldfive jargon, truthful (see the GOAL_DRIFT not-complete fallback). | `goldfive/steerer.py::compose_corrective_user_message` |
| **Cancel-reinvoke** (level 3) | Cooperatively cancel the in-flight invocation (a `CancellationRequest` keyed by `invocation_id` is written into ADK `session.state`; every adapter callback checks it and short-circuits, the LLM sees only `{"status": "cancelled"}`), refine the plan, then re-invoke with a corrective message. | `goldfive/types.py::CancellationRequest`; `goldfive/adapters/_adk_plugin.py` |
| **Pause-escalate** (level 4) | Dispatch a `GOLDFIVE_PAUSE_ESCALATE` control message so the executor blocks awaiting an operator `RESUME`/`STEER`/`CANCEL`; emits `HUMAN_INTERVENTION_REQUIRED`. #482 added `pause_escalate_deadline_s`. | `goldfive/drift_observer.py`; `goldfive/executors/_control.py` |
| **Terminate** (level 5) | Real termination (#482): abort the run with a 600s built-in deadline, emit `RunAborted` with escalation lineage. Reached from `HUMAN_INTERVENTION_REQUIRED` on repeat (see ladder table). | `goldfive/drift_observer.py` |
| **Steer** | Two distinct senses. (a) `ControlKind.STEER` — the EXTERNAL verb: a user redirects the run; bridged to `DriftKind.USER_STEER` (WARNING) in `DriftObserver.observe`, then refine runs (the canonical live-replan trigger; idempotent by `annotation_id`, #171). (b) *goldfive-authored steer promotion*: for `_GOLDFIVE_STEER_ELIGIBLE_KINDS` (`OFF_TOPIC`, `INTENT_DIVERGENCE`, `UNEXPECTED_OUTPUT`, `CONFABULATION_RISK`, loop kinds, `PLAN_DIVERGENCE`; `goldfive/drift_observer.py`), a detector drift at/above the promotion threshold is promoted into the same steer machinery as a user steer — suppressed while a recent user-authored steer is fresh (`DriftEvent.suppressed_by_user_steer`). | `goldfive/control.py::ControlKind`; `goldfive/drift_observer.py::_should_promote_to_steer` |

### Structural terms

| Term | Meaning | Anchor |
|---|---|---|
| **Overlay** | The default execution model (#141): ONE `adapter.invoke_passthrough(user_input)` per turn; the plan is a descriptive overlay on the tree's natural flow, reconciled after the fact — vs. the legacy "per-task driving" model where the executor dispatched each task. | `goldfive/executors/sequential.py::SequentialExecutor._run_overlay` (decomposed into named stage methods, #489) |
| **Reconciler** | `PlanReconciler` — maps observed `before_agent`/`after_agent` pairs onto plan-task transitions via `steerer.transition`; at invocation end stamps never-exercised PENDING tasks `NOT_NEEDED` (`get_missed_tasks` — protected keep, #163). | `goldfive/reconciler.py` |
| **Pin** | Three related senses, all "attach a stable identity to something observed". (a) **Delegation pin**: when the coordinator's LLM calls an `AgentTool`, the plugin pins `session.current_task_id` to the matching plan task (`_GoldfiveADKPlugin._maybe_pin_delegation_task`, per-`function_call_id` pins in `goldfive/state_store.py`); the F3 pre-dispatch redirect predicate is aligned with this pin (#481). (b) **Session pin**: an outer ADK session id pins a `Conversation` (see above). (c) **Snapshot pinning**: judges receive a pinned copy of `session.reasoning_history` (`pinned_history`, snapshot-passing, #479) so an await cannot tear their input. | `goldfive/adapters/_adk_plugin.py`; `goldfive/conversation.py`; `goldfive/drift_observer.py` |
| **Passthrough** | Forwarding the user's input verbatim to the tree root and letting the tree run natively (no per-task prompt synthesis). | `AgentAdapter.invoke_passthrough` (adapter-level; see 06-adapters-and-instrumentation.md) |
| **Judge** | Anything implementing the async `Judge` protocol (`goldfive/judges/base.py`): `evaluate(ctx: JudgeContext) -> JudgeVerdict \| None`. Built-ins wrap existing detectors (`goldfive/judges/builtins.py`); scheduling is guarded per-steerer (semaphore default 3, queued-window coalescing, #483). | `goldfive/judges/` |
| **Detector** | A deterministic classifier producing `DriftEvent`s from structured observations (tool results, stop reasons, call sequences, tool surfaces). Registered with per-detector config in `goldfive/drift/registry.py` (note: the registry's old `classify` dispatch was deleted in #490; `register`/`get_config`/`list_registered` remain). | `goldfive/drift/` |
| **Sink** | Anything implementing `EventSink` (`emit(event_pb)` / `close()`). Sink exceptions never abort runs (#479). | `goldfive/protocols.py::EventSink`; `goldfive/sinks/` |
| **Goal / Plan / Task** | `Goal`: derived intent unit on `session.goals`. `Plan`: frozen DAG of frozen `Task`s + `TaskEdge`s (#247 immutability — every "mutation" builds a NEW plan; single-writer enforced by `set_session_plan` + the channel-processor contextvar). `Task.supersedes` + `SupersessionKind` (REPLACE/CORRECT) model replacements/corrections; `Task.discovered` marks reactively-minted tasks. | `goldfive/types.py` |
| **Observation-only / active steering** | The master kill-switch `SteeringConfig.observation_only: bool = True` (`goldfive/config.py`). Predicate: `DefaultSteerer.is_active_steering()` (`goldfive/steerer.py`) or module helper `steering_is_active(steerer)` — the ONLY sanctioned reads (invariant 5; #488 deleted the module-global test hook and autouse fixture, so the suite now runs the shipped passive default and ~90 tests explicitly opt into active mode). | `goldfive/steerer.py::steering_is_active` |
| **Dynamic instruction** | Default-on (#251): every reachable `LlmAgent`'s static `instruction` is replaced with a resolver that re-reads current-task context from `session.state` each turn — plan-causal prompting without transcript rewrite. #477: preserves ADK `{var}` templating via `inject_session_state`. | `goldfive/convenience.py::wrap(dynamic_instruction=)`; `goldfive/adapters/adk_wrap.py` |

### Plan-revision and telemetry vocabulary

| Term | Meaning | Anchor |
|---|---|---|
| **Revision** | Every plan install after the seed is a revision: `Plan.revision_index` increments, `revision_kind`/`revision_severity` carry the triggering drift's shape, `revision_reason` the detail. Post-#271 Phase 4 there is ONE install path — the Runner seeds `Plan.empty()` on turn 1 and every `handle_turn` result installs as a revision, so `PlanRevised` (not `PlanSubmitted`) fires uniformly. | `goldfive/types.py::Plan`; `goldfive/runner.py::Runner._install_revision` |
| **`revision_trigger_event_id`** | Opaque id joining a revision to its cause: annotation id for user-control refines, `DriftEvent.id` for autonomous refines, chained across validator retries (#199). Non-empty for every revision. | `goldfive/types.py::Plan.revision_trigger_event_id` |
| **Supersedes / REPLACE / CORRECT** | `Task.supersedes = <old_id>` links a replacement to what it replaces; `Task.supersedes_kind` (`SupersessionKind`, #251) records WHY: `REPLACE` (old was PENDING/RUNNING; reporting reroutes old→new) vs `CORRECT` (old already COMPLETED but drift-contaminated; old stays as history, new is a correction child, reporting does NOT reroute). Corrective-predecessor topology is validator-enforced (`Plan.validate` step 8, #248). | `goldfive/types.py::SupersessionKind`; `goldfive/reporting/_internal.py::_resolve_effective_task_id` |
| **Discovered task / descriptive growth** | A task minted reactively at delegation-observation time (`Task.discovered=True`, #423) because the tree did work the forecast plan did not predict — the "adaptive over predictive" invariant made structural. Dedup across repeated delegations via `Task.discovery_identity_hash` (`discovery_identity_hash(agent_id, tool_args)` — sha256 over normalized arg tokens, stable across processes). | `goldfive/types.py::discovery_identity_hash`; `docs/design/PLAN-DESCRIPTIVE-GROWTH.md` |
| **Annotation id** | Source identifier of a user-control message; powers STEER idempotency and sink-side dedup of the drift row against the annotation row (`DriftDetected.annotation_id`). | `proto/goldfive/v1/control.proto`; `goldfive/drift_observer.py` |
| **Session unification** | The convention `adk ctx.session.id == goldfive Session id == harmonograf home session id` (#161); `Event.session_id` stamps it per event. | `goldfive/runner.py::Runner.run(session_id=)` |
| **Decision telemetry** | The `SteeringDecisionMade` / `LadderTransitionDecided` / `PolicyApplied` / `DetectorDispatchOrdered` / `RetryBudgetSpent` event family — goldfive narrating its own decisions (crucial under observation_only, where decisions are the ONLY output). #480 fixed label truthfulness (`DriftEvent.detector_name`, `drift_dropped_stale`/`drift_dropped_inflight` outcomes, capability_check negative class). | `goldfive/events.py::steering_decision_made_event` and siblings |
| **Verdict-utility ledger** | Per-judge-verdict outcome accounting (#483): `acted_on` / `emitted_late` / `emitted_redundant` / `parse_fail`, summarized in a teardown event — the raw material for zicato to grade judge value. | `goldfive/drift_observer.py` (ledger + teardown summary) |
| **Drift self-reporting** | Opt-in tool surface (`wrap(drift_self_reporting=True or [names])`) exposing `DRIFT_SELF_REPORTING_TOOLS` so a WILLING agent can report divergence itself — an accelerant on top of, never a substitute for, observation (invariant 1). | `goldfive/reporting/__init__.py::DRIFT_SELF_REPORTING_TOOLS` |
| **Reasoning history** | `Session.reasoning_history` — bounded ring (default max 20) of thinking blocks fed by `adapter.emit_reasoning`; the input surface for every reasoning detector and judge. | `goldfive/types.py::Session.reasoning_history` |
| **Orchestration state** | The `goldfive.*`-prefixed key namespace on session state, bridged to ADK's `session.state` via `goldfive/adapters/_adk_state_protocol.py`; ALL reads/writes go through `goldfive/state_store.py` accessors (never raw dict access — see 11-state-ownership.md). | `goldfive/state_store.py` |
| **Judge-only mode** | `wrap(judge_only=True)`: native run with built-in and custom judges active. The default steerer emits judgement and drift telemetry while skipping every drift response. | `goldfive/convenience.py::_build_judge_only_planner`; `goldfive.steerer.DefaultSteerer(dispatch_drift_interventions=False)` |

### Reporting tools: the canonical ten

`REPORTING_TOOL_NAMES` in `goldfive/reporting/handlers.py` (⚠ VOCABULARY.md §8 lists
nine including `report_task_cancelled`; main has TEN and no
`report_task_cancelled` tool — cancellation is framework-driven, not self-reported):

| Tool | Class | Effect |
|---|---|---|
| `report_task_started` / `report_task_progress` / `report_task_completed` / `report_task_failed` / `report_task_blocked` | lifecycle | Drive the corresponding `mark_task_*` transition (converging with the reconciler at `steerer.transition`). |
| `report_new_work_discovered` | lifecycle (default-on; no observation analog — #196) | Emits `NEW_WORK_DISCOVERED` drift so refine can grow the plan. |
| `report_awaiting_approval` | lifecycle | Human-in-the-loop gate; never hangs (#478 — no channel → immediate `'unavailable'` ack, 600s default timeout, expiry → `HUMAN_INTERVENTION_REQUIRED`; `plan_state` stripped from acks under observation_only). |
| `report_plan_divergence` | drift self-reporting (opt-in) | Agent volunteers "I diverged"; historical `PLAN_DIVERGENCE` producer (handling disabled #252). |
| `declare_task_skipped` / `declare_task_not_needed` | drift self-reporting (opt-in), observability-only | Emit `TaskDeclarationReceived`; no plan mutation — the imperative surface or reconciler later confirms or contradicts. |

`LIFECYCLE_REPORTING_TOOLS` vs `DRIFT_SELF_REPORTING_TOOLS` is the split
`wrap(drift_self_reporting=...)` selects over; all of it remains optional per
invariant 1.

The kill-switch helper, verbatim — memorize its fail-safe direction:

```python goldfive/steerer.py
def steering_is_active(steerer: Any) -> bool:
    """Return ``True`` iff ``steerer`` permits active-steering interventions.
    ...
    """
    predicate = getattr(steerer, "is_active_steering", None)
    if not callable(predicate):
        return False
    try:
        return bool(predicate())
    except Exception:  # noqa: BLE001
        return False
```

### ⚠ Known divergences: docs/design/VOCABULARY.md vs. code on main

| VOCABULARY.md claim | Code on main (wins) |
|---|---|
| §5: "`DriftKind`: 26 values total (25 named kinds + CUSTOM)" | 41 members in `goldfive/types.py::DriftKind` (the doc's own "Count check" admits more exist but still omits `LLM_CALL_TIMEOUT`, `JUSTIFIED_DEVIATION`, `CAPABILITY_MISMATCH`). Use the table below. |
| §7: refine policy is one `severity >= WARNING` comparison in `DefaultSteerer._handle_drift` | Policy is the per-kind `_LADDER` table read by `DriftObserver._ladder_level_for` in `goldfive/drift_observer.py`; the dispatch entry point is `DriftObserver.handle_drift`. |
| §4: `TaskStatus` "six states" | Seven — `NOT_NEEDED` is a first-class terminal member (the doc's own table includes it; the prose count is stale). Canonical terminal set: `TERMINAL_TASK_STATUSES` in `goldfive/types.py` (#485). |
| §5.a rows imply producers for `MODEL_REFUSAL`, `HALLUCINATION_SUSPECTED`, `STOPPED_EARLY`, `UNEXPECTED_OUTPUT`, `SCHEMA_VIOLATION` ("adapter-detected", "caller-supplied") | Several have NO production producer on main — see the "producer" column below before assuming a kind fires. |
| §2/§6 attribute drift handling and event emission to "Steerer" methods | Post-#410 facade split, they live on the three components (`steerer.tasks` / `steerer.plans` / `steerer.drift`). |
| §2: "`ControlKind` … one of five values" | 12 members on main (`goldfive/control.py`), including APPROVE/REJECT, STATUS_QUERY, INTERCEPT_TRANSFER, INJECT_MESSAGE, and the goldfive-internal GOLDFIVE_STEER / GOLDFIVE_PAUSE_ESCALATE. |
| §5.e: `INTERCEPT_TRANSFER` "not currently in the DriftKind enum … recognized as a control kind-by-string" | It is a first-class `ControlKind` member on main; still not a `DriftKind`. |

## DriftKind taxonomy: all 41 members

Ground truth: `goldfive/types.py::DriftKind` (string values are the lower-snake
literals). "Producer" = the code on main that constructs a `DriftEvent` of that kind
(verified by grep on 2026-07-05, post-#492). "No producer" means the enum member is
declared (proto compatibility, external emitters, tests) but nothing in
`goldfive/` production code mints it — do NOT invent a producer to "complete" it.

Ladder column: the `(INFO, WARNING, CRITICAL-first / CRITICAL-repeat)` levels from
`DriftObserver._LADDER` (`goldfive/drift_observer.py::_load_ladder_tables`).
"default" = kind has no `_LADDER` row, so the fallback in `_ladder_level_for` applies:
INFO→OBSERVE, WARNING→ABSORB, CRITICAL→ABSORB / PAUSE_ESCALATE-on-repeat. A `None`
slot inside a row also falls back to OBSERVE. "Repeat" means
`occurrence_count >= DefaultSteerer.REFINE_FAILURE_THRESHOLD` (class constant, `= 2`,
`goldfive/steerer.py`).

| # | Member | Producer (file :: symbol) | Default severity | Ladder (INFO / WARN / CRIT-first→repeat) |
|---|---|---|---|---|
| 1 | `TOOL_ERROR` | `goldfive/drift/__init__.py::classify_tool_error` | WARNING | OBSERVE / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 2 | `AGENT_REFUSAL` | `goldfive/drift/__init__.py::classify_refusal` | tier-graded INFO/WARNING/CRITICAL (marker tier; CRITICAL scanned first) | OBSERVE / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 3 | `NEW_WORK_DISCOVERED` | `goldfive/drift_observer.py` (reporting-tool `report_new_work_discovered` path, WARNING); `goldfive/plan_reviser.py` + `goldfive/runner.py` (descriptive-growth notices, INFO) | WARNING (reporting) / INFO (growth) | default |
| 4 | `PLAN_DIVERGENCE` | **Disabled**: `DriftObserver.handle_drift` drops it at entry (#252, superseded by `CAPABILITY_MISMATCH` #253). Machinery is a PROTECTED KEEP — do not delete the enum, ladder row, or planner surfaces. | WARNING (historical) | row exists (OBSERVE / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE) but unreachable via `handle_drift` |
| 5 | `USER_STEER` | `goldfive/drift_observer.py::DriftObserver.observe` (from `ControlKind.STEER`); also `goldfive/plan_reviser.py` | WARNING | default (WARNING→ABSORB = refine; the canonical live-replan) |
| 6 | `USER_CANCEL` | `goldfive/drift_observer.py` (audit synthesis; executor short-circuits to `RunAborted` first) | CRITICAL | default (moot — executor aborts) |
| 7 | `USER_PAUSE` | `goldfive/drift_observer.py` (audit synthesis; pause effect is the executor's blocking wait) | INFO | default (OBSERVE) |
| 8 | `TASK_FAILED_RECOVERABLE` | `goldfive/task_state_machine.py::TaskStateMachine.mark_task_failed(recoverable=True)` | WARNING | default |
| 9 | `TASK_FAILED_FATAL` | same site, `recoverable=False` | CRITICAL | default |
| 10 | `CONTEXT_PRESSURE` | `goldfive/drift/__init__.py::classify_stop_reason` (MAX_TOKENS / LENGTH / TRUNCATED / CONTENT_FILTER / MAX_OUTPUT_TOKENS) | WARNING | default |
| 11 | `BLOCKED` | `goldfive/task_state_machine.py::mark_task_blocked` (paired with the `TaskStatus.BLOCKED` transition + `TaskBlocked` event) | WARNING | default |
| 12 | `WRONG_AGENT` | **no producer** on main | — | default |
| 13 | `AGENT_TRANSFER` | **no producer** on main | — | default |
| 14 | `MODEL_REFUSAL` | **no production producer** (only `goldfive/testkit/adversarial.py` synthesizes it); ladder row + corrective template kept for external emitters | CRITICAL (by convention) | OBSERVE / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 15 | `STOPPED_EARLY` | **no producer** on main | — | default |
| 16 | `TOO_MANY_STEPS` | **no producer** on main | — | default |
| 17 | `GOAL_UNREACHABLE` | **no producer** on main | — | default |
| 18 | `TASK_TIMEOUT` | `goldfive/adapters/_adk_plugin.py` stall watchdog (#487; flag-gated `SteeringConfig.stall_watchdog_enabled=False`, `stall_timeout_s=600.0`; liveness watermark `Session.last_observed_event_at`) | WARNING, escalating CRITICAL on continued silence | OBSERVE / NUDGE / PAUSE_ESCALATE→PAUSE_ESCALATE |
| 19 | `REPEATED_FAILURE` | `goldfive/drift_observer.py` (same task failed ≥ N in one run) | CRITICAL | default |
| 20 | `UNEXPECTED_OUTPUT` | **no producer** on main (listed in `_GOLDFIVE_STEER_ELIGIBLE_KINDS` for external emitters) | — | default |
| 21 | `SCHEMA_VIOLATION` | `goldfive/plan_reviser.py` (refine JSON unparseable, CRITICAL); `goldfive/drift_observer.py` (malformed-judge downgrades, INFO — #479) | CRITICAL / INFO by site | default |
| 22 | `HALLUCINATION_SUSPECTED` | **no producer** on main | — | default |
| 23 | `SAFETY_CONCERN` | **no producer** on main | — | default |
| 24 | `RESOURCE_EXHAUSTED` | **no producer** on main | — | default |
| 25 | `AMBIGUOUS_INTENT` | **no producer** on main | — | default |
| 26 | `CUSTOM` | escape hatch: `goldfive/adapters/_adk_plugin.py`, `goldfive/drift_observer.py`, `goldfive/executors/_shared.py` (all INFO); callers pick severity | caller's choice | default |
| 27 | `LOOPING_TOOL_CALL` | **no producer on main — deliberate** (PROTECTED KEEP, #204/#206): the tool-loop tracker emits `LOOPING_REASONING` with `detector_name="tool_loops"` so tool loops get NUDGE-first CRITICAL routing. Do not "fix" this by making `tool_loops.py` emit `LOOPING_TOOL_CALL`. | — | row kept: (None) / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 28 | `LOOPING_REASONING` | `goldfive/drift/reasoning.py` (embedding-similarity cliff, WARNING) and `goldfive/drift/tool_loops.py::ToolLoopTracker` (loop patterns; #484 caps the name-axis at INFO unless ≥2 identical `(name, args_hash)` corroborate — knob `name_axis_max_severity`, provenance in `raw["severity_capped_from"]`) | WARNING (INFO when capped) | (None) / ABSORB / **NUDGE**→PAUSE_ESCALATE |
| 29 | `REASONING_CLUSTER_TIGHTENING` | `goldfive/drift/reasoning.py` (0.75 ≤ cosine < 0.9; one-shot per task via `Session.reasoning_cluster_flagged`) | INFO | OBSERVE / OBSERVE / OBSERVE→OBSERVE (never intervenes) |
| 30 | `OFF_TOPIC` | `goldfive/drift/reasoning.py` (embedding, WARNING); `goldfive/drift/reasoning_judge.py` (judge off-task verdict, judge-set severity); `goldfive/adapters/_adk_plugin.py` (WARNING) | WARNING | OBSERVE / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 31 | `INTENT_DIVERGENCE` | `goldfive/drift/reasoning.py::detect_intent_divergence` | graduated INFO/WARNING/CRITICAL by similarity band | OBSERVE / ABSORB / PAUSE_ESCALATE→PAUSE_ESCALATE |
| 32 | `UNCERTAIN_PROGRESS` | `goldfive/drift_observer.py` (opt-in reflective self-progress: "making progress" with confidence < 0.5) | INFO | default (OBSERVE) |
| 33 | `SELF_REPORTED_STUCK` | `goldfive/drift_observer.py` (opt-in reflective self-progress: "not making progress") | WARNING | (None) / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 34 | `CONFABULATION_RISK` | `goldfive/drift/__init__.py::classify_confabulation_risk` (external-data-shaped task, non-empty output, zero tool calls; keyword set `CONFABULATION_TRIGGER_KEYWORDS` matches TASK TEXT, a legacy conservative surface) | INFO | OBSERVE / ABSORB / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 35 | `RUNAWAY_DELEGATION` | `goldfive/adapters/_adk_plugin.py` (`ADKAdapter(agent_tool_cap=N)`, default 16, per invocation; #130) | CRITICAL | (None) / (None) / CANCEL_REINVOKE→PAUSE_ESCALATE |
| 36 | `REFINE_VALIDATION_FAILED` | `goldfive/planner.py::LLMPlanner` (refine retry budget exhausted; #133 — NEVER re-refined) | CRITICAL | (None) / (None) / PAUSE_ESCALATE→PAUSE_ESCALATE |
| 37 | `GOAL_DRIFT` | `goldfive/drift/goals.py::classify_goal_drift` (trajectory-level judge every `GoalDriftConfig.check_interval=5` invocations; idle trigger consumes `GOAL_DRIFT_IDLE_SECONDS`, #487) | CRITICAL (classifier); WARNING path exists via ladder | (None) / NUDGE / NUDGE→CANCEL_REINVOKE (F4 loop-prevention: plan is right, agent's next action is stuck — nudge first) |
| 38 | `HUMAN_INTERVENTION_REQUIRED` | `goldfive/drift_observer.py` (structural escalation, CRITICAL); `goldfive/reporting/handlers.py` (approval expiry, WARNING, #478); `goldfive/runner.py` (INFO notice) | CRITICAL / WARNING / INFO by site | (None) / (None) / PAUSE_ESCALATE→**TERMINATE** |
| 39 | `LLM_CALL_TIMEOUT` | `goldfive/adapters/_adk_plugin.py` per-call watcher. The watcher's budget is `make_adk_plugin(llm_call_timeout_ms=...)`, whose own parameter default is `DEFAULT_LLM_CALL_TIMEOUT_MS = 1_800_000` (30 min, a pathological-hang ceiling). Through the `wrap()` path the effective budget is **120s**, because `wrap()` threads `AgentConfig.call_timeout_ms` (default `120_000`, `goldfive/config.py`) into `ADKAdapter(llm_call_timeout_ms=...)`. #476: under `observation_only` it no longer cancels the invocation and a one-shot reasoning-channel-disarm warning is emitted. | CRITICAL | default |
| 40 | `JUSTIFIED_DEVIATION` | `goldfive/drift/reasoning_judge.py` (iter-10 three-state judge: deviation plausibly provoked by reality — tool error, surprising result, discovered dependency) | judge-set | OBSERVE / ABSORB / ABSORB→ABSORB (never escalates — reality-provoked deviation is plan-extension input) |
| 41 | `CAPABILITY_MISMATCH` | `goldfive/drift/capability_check.py::detect_capability_mismatch` (at `delegation_observed`: invoked agent's tool surface structurally cannot do the bound task, or `Task.required_tools` unsatisfied; #253. #480 fixed its negative-class telemetry label.) | CRITICAL | default |

Reading rules for this table:

1. **"No producer" is not an invitation.** Members 12–17, 20, 22–25 exist for proto
   stability and external emitters (custom adapters, bridges, tests may synthesize
   them and feed `steerer.drift.observe`). Adding a producer is a design decision —
   see `.agents/how-to-add-a-drift-kind.md` and 07-deterministic-drift-detection.md —
   not a gap-fill.
2. **Three members are PROTECTED KEEP decisions** requiring explicit human sign-off to
   touch: `LOOPING_TOOL_CALL` (enum/ladder/promotion/planner surfaces, #204/#206),
   `PLAN_DIVERGENCE` (disabled-but-keep, #252), and `reconciler.get_missed_tasks`
   (#163, the `NOT_NEEDED` stamping path — not a DriftKind but same protection class).
3. **Severity is chosen by the producer, not the enum.** The same kind can fire at
   different severities from different sites (`SCHEMA_VIOLATION`,
   `HUMAN_INTERVENTION_REQUIRED`, `NEW_WORK_DISCOVERED`). Malformed judge output is
   downgraded to INFO (#479), never propagated as CRITICAL.

## The intervention ladder at a glance

Full treatment in 09-steering-ladder-and-gates.md; here is the minimum a first-time
editor needs.

```python goldfive/steerer.py
class InterventionLevel(enum.IntEnum):
    OBSERVE = 0
    ABSORB = 1
    NUDGE = 2
    CANCEL_REINVOKE = 3
    PAUSE_ESCALATE = 4
    TERMINATE = 5
```

Resolution order inside `DriftObserver.handle_drift` (`goldfive/drift_observer.py`):

1. **Entry guards**: `PLAN_DIVERGENCE` dropped (#252); `authored_by` normalized;
   verdict-freshness gate (#245) — a goldfive-authored drift whose
   `observed_revision_index` was already addressed for the SAME `(kind, target)` at a
   later revision is emitted for observability and otherwise dropped
   (`drift_dropped_stale` telemetry, #480); an in-flight-refine registry keyed
   `(kind, current_task_id)` short-circuits concurrent duplicates
   (`drift_dropped_inflight`).
2. **Level selection**: `_ladder_level_for(kind, severity, occurrence_count)` reads
   `_LADDER`; occurrence counting keys on the refine-outcome ledger
   (`Session.refine_outcomes`, per `(drift_kind_value, task_id)`, reset each turn).
   Correction keys use the full agent path (#479) — stable identity, invariant 6.
3. **Dispatch**: each level's mutating arm checks `is_active_steering()` first. Under
   the shipped default everything above OBSERVE records a decision
   (`SteeringDecisionMade`, with honest labels per #480) and does nothing.

Also load-bearing at this altitude: a successful ABSORB for the coordinator-stuck kinds
queues a follow-up NUDGE (`_ABSORB_NUDGE_KINDS`, #202), and user-authored drifts bypass
the freshness gate unconditionally (operator directives always win, #242).

## Two worked traces

### Trace A — a user steer, click to revised plan

The richest external path, updated to where each step actually lives on main
(VOCABULARY.md §2 tells the same story with pre-facade-split code homes):

1. **UI → bridge.** The operator clicks "Steer" in harmonograf with a note. The bridge
   (external to goldfive) builds
   `ControlMessage(kind=ControlKind.STEER, payload={"note": ..., "suggested_action": ...})`
   and calls `channel.send(msg)` on the run's `ControlChannel` (`goldfive/control.py`).
2. **Executor drains.** The overlay executor's control poll receives the message and
   routes it through `dispatch_control` (`goldfive/executors/_control.py`); the
   resulting outcome carries the steer message forward and an ack
   (`ControlAck(control_id, result=AckResult.SUCCESS, ...)`) flows back out via
   `channel.ack` for the UI.
3. **Verb becomes observation.** The executor feeds the message to
   `steerer.drift.observe(msg, session)` (`goldfive/drift_observer.py::DriftObserver.observe`),
   which synthesizes `DriftEvent(kind=DriftKind.USER_STEER, severity=WARNING,
   authored_by="user")` — idempotent by `annotation_id` / message id (#171). From here
   on, `STEER` the verb is gone; only `USER_STEER` the observation remains.
4. **Ladder dispatch.** `DriftObserver.handle_drift` emits `DriftDetected` on every
   sink, skips the freshness gate (user-authored), and — `USER_STEER` having no
   `_LADDER` row — resolves WARNING to ABSORB via the default arm of
   `_ladder_level_for`.
5. **Refine.** `planner.refine(plan=..., drift=..., goals=...)` runs
   (`goldfive/planner.py::LLMPlanner`); for `USER_STEER` the semantics are
   preserve-completed / re-plan-pending honoring the note. The revision must pass
   `Plan.validate(for_revision=True, prior=...)`.
6. **Install + announce.** `PlanReviser._apply_revision` (`goldfive/plan_reviser.py`)
   installs the new plan via `set_session_plan` (single-writer contextvar honored) and
   `_emit_plan_revised` fires `PlanRevised` with `revision_kind="user_steer"`,
   `revision_severity="warning"`, and `revision_trigger_event_id` set to the source
   annotation id — harmonograf joins the revision back to the click.
7. **The tree learns.** No transcript rewrite: the affected agent's next turn sees the
   revised task via dynamic instructions (#251/#477), and — if the coordinator is stuck
   mid-invocation — the steer restart machinery re-invokes with a
   `[GOLDFIVE STEERING CONTROL ...]`-framed corrective body.

Note what did NOT happen: no agent was asked to acknowledge anything; every step is
executor/steerer-side. That is the no-cooperation contract in action.

### Trace B — a detector drift under the shipped default (observation_only=True)

The same pipeline, autonomous producer, production config:

1. During the single passthrough invocation, the wrapped model emits thinking tokens;
   the ADK plugin forwards them through `adapter.emit_reasoning` into
   `session.reasoning_history` and `DriftObserver.observe_reasoning`.
2. The reasoning judge (`goldfive/drift/reasoning_judge.py`) is scheduled against a
   **pinned** history snapshot (#479), throttled by the per-steerer semaphore
   (default 3) with queued-window coalescing (#483). It stamps
   `observed_revision_index` BEFORE its LLM await (#245).
3. The judge returns an off-task verdict → `DriftEvent(kind=OFF_TOPIC,
   severity=WARNING, authored_by="goldfive", detector_name=...,
   trigger_input=<the reasoning block judged>)`.
4. `handle_drift` runs the gates: freshness (was this `(kind, target)` already
   addressed at a later revision? → `drift_dropped_stale`), in-flight refine registry
   (`drift_dropped_inflight`), user-steer suppression window
   (`suppressed_by_user_steer`). Say all pass.
5. `_ladder_level_for(OFF_TOPIC, WARNING, count)` → ABSORB per the `_LADDER` row.
6. **The kill-switch bites.** The ABSORB arm consults `is_active_steering()`; under
   the default `SteeringConfig.observation_only=True` it is `False`, so: NO refine
   install, NO nudge enqueue, NO cancel. What DOES happen: `DriftDetected` +
   `SteeringDecisionMade` (with the level goldfive WOULD have taken) + judge telemetry
   + verdict-utility ledger update go to every sink. Operators and zicato see the full
   decision trail; the run is untouched.
7. Flip `RuntimeConfig(steering=SteeringConfig(observation_only=False))` and step 6
   becomes the real intervention: refine, install, `PlanRevised`, and (for
   `_ABSORB_NUDGE_KINDS`) a queued nudge the overlay drains into a truthful synthetic
   user turn.

### The event stream at a glance

Everything both traces "emit" is a proto `Event` envelope built by a factory in
`goldfive/events.py` (never hand-construct envelopes — `new_event` stamps `run_id`,
monotonic `sequence` via `Session.next_sequence()`, and `session_id`). The core
factories, grouped (full spec: 12-events-sinks-telemetry.md):

| Group | Factories in `goldfive/events.py` | Emitted by |
|---|---|---|
| Run lifecycle | `run_started_event`, `run_completed_event`, `run_aborted_event`, `conversation_started_event`, `conversation_ended_event` | Runner / executor |
| Planning | `goal_derived_event`, `plan_submitted_event`, `plan_revised_event` | Runner / PlanReviser |
| Task lifecycle | `task_started_event`, `task_progress_event`, `task_completed_event`, `task_failed_event`, `task_blocked_event`, `task_cancelled_event`, `task_transitioned_event`, `task_transition_refused_event` | TaskStateMachine |
| Drift & control | `drift_detected_event`, `control_received_event`, `invocation_cancelled_event` | DriftObserver / dispatch_control |
| Observation | `agent_invocation_started_event`, `agent_invocation_completed_event`, `invocation_boundary_entered_event`/`_exited_event`, `delegation_observed_event` | ADK plugin / adapter |
| Approval | `approval_requested_event`, `approval_granted_event`, `approval_rejected_event` | reporting handlers / dispatch_control |
| Decision telemetry | `steering_decision_made_event`, `ladder_transition_decided_event`, `detector_dispatch_ordered_event`, `policy_applied_event`, `retry_budget_spent_event` | DriftObserver |
| Judge telemetry | `JudgementEmitted` / `ReasoningJudgeInvoked` payloads (built inline in `goldfive/drift_observer.py`; `ReasoningJudgeInvoked` fields 12–15 added in #480: `focused_task_id`, `focus_confidence`, `stated_intent`, `provenance`) | DriftObserver |

Two ordering guarantees sinks rely on: `sequence` is monotonic per run, and every
successful task transition emits exactly one task event (rejected transitions emit
`task_transition_refused_event`, not silence).

## What is deferred — do not build it as if it exists

These are KNOWN, deliberately-not-on-main directions. Present them as future work when
writing docs; never implement them opportunistically — each is blocked on a named
precondition.

| Deferred item | What it would be | Blocked on |
|---|---|---|
| Twin-refine-pipeline extraction | Consolidating the two parallel refine paths into one extracted pipeline. | The agency-preservation branch-merge decision (user-owned). |
| Evidence-ledger suppression | Replacing the ~7 stacked suppression gates in `handle_drift` (freshness, in-flight, user-steer window, …) with a single evidence-ledger mechanism. | Same branch-merge decision. |
| Judge windowing/cadence expansion | Richer scheduling windows and cadence policies for the LLM judges. | A judge regression harness (does not exist yet — without it, cadence changes are unverifiable). |
| Judge-facade dispatch authority | Letting the judge facade own dispatch decisions rather than routing through `handle_drift`. | Design decision pending. |
| Checkpoint-rollback, tool-gating hold, fork-and-judge | Stage-4 interventions (rewind the run to a checkpoint; hold a tool call pending judgment; fork the run and judge both branches). | Bench-gated (Stage-4 of the agency-preservation roadmap). |
| Agency-preservation Stages 1–3 on main | Ledger/observer-note machinery, plan_mode=forecast, signal channel, etc. (#453–#474). | Lives ONLY on the unmerged `agency-preservation` branch behind default-OFF flags; step 13b (three-arm bench + measurement-gated default flips + hard deletions) is LOCKED on explicit user sign-off, and merging the branch to main is a separate user decision. |

If a task seems to require one of these, the correct move is to say so and stop, not to
build a partial version on main.

## Guided tour: where to find what

One sentence per file/package. Deep-dive chapter in the last column.

### Top-level modules (`goldfive/*.py`)

| File | One sentence | Chapter |
|---|---|---|
| `__init__.py` | Public API: re-exports `wrap`, `run`, `quickstart`, `Runner`, the six protocols, enums/dataclasses, sinks, `steering_is_active`, judges. | 02 |
| `types.py` | The data model: `TaskStatus` + `TERMINAL_TASK_STATUSES`, `DriftKind`, `DriftSeverity`, frozen `Task`/`Plan`/`TaskEdge` with `Plan.validate`, `DriftEvent`, `Goal`, `Session`, `CancellationRequest`, plan-mutation helpers (`replace_task`, `with_task_status`, `add_tasks`, `replace_edges`, `bump_revision`, `set_session_plan`), discovery-identity hashing. | 11 |
| `protocols.py` | The six pluggable protocols: `GoalDeriver`, `Planner` (generate/refine/handle_turn), `Steerer` (facade: `tasks`/`plans`/`drift` + `is_active_steering`), `AgentAdapter`, `Executor`, `EventSink`. | 02 |
| `runner.py` | `Runner` — the single public entrypoint: composes the six components, seeds `Plan.empty` per turn, installs revisions, owns per-session locks and `_abort_turn` (#489). | 03 |
| `conversation.py` | `Conversation`/`TurnRecord` — cross-turn state (goals, results, prior plan) keyed by outer-session id. | 03 |
| `convenience.py` | `wrap()` and `run()` — one-call Runner factories, judge-only mode (#446), default component selection. | 03 |
| `quickstart.py` | Even smaller one-call factory wiring `SequentialExecutor` + passthrough defaults for new users. | 03 |
| `config.py` | Typed per-Runner config (#225): `RuntimeConfig` bundling `SteeringConfig` (`observation_only=True`, stall watchdog knobs), `EmbeddingConfig`, `ToolLoopConfig`, `ReasoningDriftConfig`, `GoalDriftConfig`, `JudgeConfig`; `from_env` readers. | 14 |
| `control.py` | `ControlChannel`/`ControlMessage`/`ControlKind`/`ControlAck`/`AckResult` — the bidirectional async control primitive. | 04 |
| `steerer.py` | `DefaultSteerer` facade + `InterventionLevel` + `steering_is_active` + `RefineExhausted` + corrective-message composition (`_CORRECTIVE_TEMPLATES`). | 09 |
| `drift_observer.py` | `DriftObserver` — the largest module: `observe`/`observe_reasoning`/`detect_drift`/`handle_drift`, the `_LADDER` table, suppression gates, condition lifecycle, steer promotion, nudge/cancel/pause dispatch. | 09 |
| `task_state_machine.py` | `TaskStateMachine` — `mark_task_*` transitions + per-status sink emission (Wave C extraction from steerer). | 11 |
| `plan_reviser.py` | `PlanReviser` — revision install, `_apply_revision`, `_emit_plan_revised`, refine-attempt bookkeeping (Wave C extraction). | 10 |
| `planner.py` | `LLMPlanner`/`StaticPlanner`/`PassthroughPlanner` — plan generation, `handle_turn` classification, refine prompts, structural validation retries. | 10 |
| `plan_reviser.py` + `planner.py` split rule | The planner PRODUCES plans; the reviser INSTALLS them — do not blur. | 10 |
| `goal_deriver.py` | `LLMGoalDeriver`/`LiteralGoalDeriver`/`PassthroughGoalDeriver` — free-form input → `list[Goal]`. | 03 |
| `reconciler.py` | `PlanReconciler` — overlay observation → plan transitions; `get_missed_tasks` (protected, #163) stamps `NOT_NEEDED`. | 04 |
| `reporting/` (pkg) | Agent-facing reporting tools: `handlers.py` (async handlers incl. `report_awaiting_approval`, #478), `schemas.py` (JSON-schema blocks), `rendering.py` (LLM-visible response shaping), `_internal.py`. | 13 |
| `events.py` | Proto `Event` envelope factories (`run_started_event`, `drift_detected_event`, …) — every wire event is built here. | 12 |
| `conv.py` | Dataclass ⇄ protobuf round-trip converters (lazy proto imports). | 12 |
| `results.py` | `ExecutionOutcome` / `InvocationResult` result dataclasses. | 03 |
| `state_store.py` | Single source of truth for `goldfive.*`-prefixed `Session.state` keys: unified read/write accessors, delegation pins, condition lifecycle helpers. | 11 |
| `context_editor.py` | Request-side context editing as a steering capability (#397) — the centralized gate for editing what the wrapped LLM sees. | 06 |
| `prompt_shaper.py` | `PromptShaper` — centralized gate + injection sites for every prompt augmentation goldfive performs (all `observation_only`-gated). | 06 |
| `_correction_injection.py` | Write-side for CORRECT-kind supersedes (#251 Stream D): threads plan-causal context into wrapped-agent prompts. | 10 |
| `_llm.py` | THE one internal LLM-call module (#491): `CallLLM` typing, budgets, builders, `THINKING_DISABLE_CAPABILITIES` table, per-call `LlmCallDiagnostics` via ContextVar; Qwen `/no_think` handling scoped to Qwen/litellm family. | 08 |
| `_llm_detect.py` | Detect a usable LLM surface on a wrapped agent (so `wrap()` can default planner/deriver LLMs). | 03 |
| `_llm_span.py` | Emits goldfive-internal LLM call spans for observability. | 12 |
| `_state_audit.py` | Debug-mode runtime assertion of the state-ownership contract (#271 Phase 0; `GOLDFIVE_STRICT_STATE_OWNERSHIP`). | 11 |
| `builtin_judges.py` | Public factory-registry name for built-in judges (thin re-export). | 08 |
| `runtime.py` | Process-wide determinism handles (`seeded_uuid4` etc.) for eval harnesses. | 15 |

### Packages

| Package | One sentence per member | Chapter |
|---|---|---|
| `adapters/` | `adk.py` (`ADKAdapter`), `_adk_plugin.py` (the ADK `BasePlugin` doing all callback observation, delegation pins, caps, watchdogs), `_adk_state_protocol.py` (state keys shared with ADK `session.state`), `adk_wrap.py` (Runner-as-ADK-agent polymorph, `GoldfiveADKAgent`), `adk_llm_instrumentation.py` (request-side LLM instrumentation), `adk_reentry.py` (re-entry contract for third-party plugins), `auto.py` (`auto_adapter` shape detection), `callable.py` (reference adapter), `claude.py`/`_claude_prompt.py` (Claude Agent SDK adapter), `_tool_invocation.py` (shared reporting-tool invocation helper). | 05, 06 |
| `drift/` | `__init__.py` (taxonomy + `classify_tool_error`/`classify_refusal`/`classify_stop_reason`/`classify_confabulation_risk`), `tool_loops.py` (`ToolLoopTracker`, #484 name-axis cap), `reasoning.py` (embedding detectors: intent divergence, loops, cluster tightening, off-topic), `reasoning_judge.py` (per-thinking-message LLM judge), `goals.py` (trajectory GOAL_DRIFT judge), `capability_check.py` (#253), `registry.py` (per-detector config + shared judge boilerplate; `classify` dispatch deleted #490), `_embed.py` (optional embedding backend + circuit breaker with half-open recovery, #479). | 07, 08 |
| `judges/` | `base.py` (`Judge` protocol, `JudgeContext`, `JudgeVerdict`), `builtins.py` (built-ins wrapping detectors), `__init__.py` (public surface). Scheduling guards (#483) live steerer-side. | 08 |
| `executors/` | `sequential.py` (`SequentialExecutor` — the overlay executor, stage methods per #489), `parallel.py` (`ParallelDAGExecutor` — skips terminal incl. `NOT_NEEDED`, #485), `_shared.py` (canonical shared helpers, #485), `_control.py` (`dispatch_control` + control outcome plumbing). | 04 |
| `planners/` | `goldfive_planner.py` — `GoldfivePlanner(BasePlanner)`, the ADK-native structural-steering planner (#153). | 10 |
| `sinks/` | `memory.py`, `logging_sink.py` (snake_case JSON), `persistence.py` (JSONL, camelCase, replay), `sqlite_sink.py`, `grpc_sink.py` (harmonograf ingress). | 12 |
| `server/` | `grpc_server.py` — reference `GoldfiveIngress` gRPC server. | 12 |
| `optimization/` | `manifest.py` + `manifest.toml` (zicato-mutable knob inventory), `prompts.py` + `prompts/*.md` (markdown-backed steering prompt catalog). | 14 |
| `reporting/` | see top-level table row above. | 13 |
| `testkit/` | `adversarial.py` (agents that deliberately misbehave per drift kind), `canned_call_llm.py` (deterministic `call_llm` stand-in). | 15 |
| `pb/` | GENERATED protobuf stubs (`goldfive/pb/goldfive/v1/`) — never hand-edit; regenerate via `make proto` from `proto/goldfive/v1/*.proto`. | 12 |

### Non-package directories

| Path | What it is |
|---|---|
| `proto/goldfive/v1/` | The `.proto` sources (types, events, control) — the wire contract with harmonograf/zicato. |
| `docs/design/` | Design docs (ARCHITECTURE, DRIFT, VOCABULARY, PLAN-LIFECYCLE, STATE-OWNERSHIP-CONTRACT, AGENCY-PRESERVATION, …). Accuracy-swept in #492, but the CODE ON MAIN is ground truth — when they disagree, the code wins. |
| `.agents/*.md` | Task-shaped skills (how-to-add-a-drift-kind, testing, debug-goldfive, …) — read the matching one before a task of that shape. |
| `examples/` | Runnable integration examples (adk_agent, adk_web_wrapped, approval_gated_agent, grpc_ingress, harmonograf_observed, …). |
| `bench/` | Benchmark harness assets. |
| `tests/` | ~2900 tests; see 15-testing-guide.md. |

## Reading the history

goldfive's git log is unusually load-bearing: most non-obvious code shapes exist
because a numbered PR fixed a real failure, and the comments cite those numbers.
`git log --oneline --grep '#<n>'` and `git log -S '<symbol>' --oneline` are your
first tools when a shape looks wrong. The eras, so PR numbers in comments mean
something to you:

| Era / PR range | What happened | Why you care today |
|---|---|---|
| #133–#144 (2026-04) | Overlay era: refine-exhaustion semantics (#133), observation-driven overlay + `PlanReconciler` replacing per-task driving (#141), the graduated intervention ladder (#142), goal-drift judge (#143), divergence-aware refine (#144). | The default execution model; `_run_overlay`, `NOT_NEEDED`, ladder levels. |
| #151–#155 | Structural steering: tree-aware planner, orchestration state namespace, `GoldfivePlanner(BasePlanner)`, goal-aware refine. | `goldfive/planners/`, `state_store` key discipline. |
| #161–#171 | Session unification, steer idempotency, retirement of the first regex heuristics (#166/#167). | Invariant 2's case law. |
| #181–#206 | Loop detection era: `ToolLoopTracker` (#181/#186), tool-loops-emit-`LOOPING_REASONING` decision (#204/#206 — protected keep). | Why `LOOPING_TOOL_CALL` has no producer. |
| #225, #247–#254 | Typed config (#225); Plan/Task immutability + torn-read fix (#247); corrective topology (#248); LLM-visible-response minimalism lessons (#250/#252/#253); `CAPABILITY_MISMATCH` replaces `PLAN_DIVERGENCE` handling (#252/#253); `observation_only` master switch (#254). | Most of `types.py`'s shape; the disabled-but-kept `PLAN_DIVERGENCE`. |
| #271 (+ phases) | "Intent fully validated": state-ownership contract, single plan-install path, conversation-per-outer-session, cancellation contract. | `_state_audit`, `Plan.empty` seeding, per-session locks. |
| #397–#423 | Context editing (#397), audit-driven hardening (#402/#405/#410 facade split), plan-descriptive-growth (#423). | `steerer.tasks/plans/drift` component layout; discovered tasks. |
| #436–#446 | zicato optimization surface (#436–#442); judge-only mode (#446). | `goldfive/optimization/`, `wrap(judge_only=True)`. |
| #453–#474 | Agency-preservation Stages 1–3 — **branch only, unmerged**. | Do not copy to main; doc text must not claim these exist on main. |
| #475–#492 (2026-07, on main) | The hardening program this guide documents: truthful gated nudges (#475), timeout behavior under observation_only (#476), templating preservation (#477), approval never-hangs (#478), judge/sink/breaker robustness (#479), telemetry truthfulness (#480), F3 gating (#481), real TERMINATE (#482), judge scheduling guards (#483), name-axis severity cap (#484), canonical terminal statuses (#485), drift-condition resolution (#486), stall watchdog (#487), single kill-switch predicate (#488), `_abort_turn` + overlay decomposition (#489), dead-code deletion with archaeology (#490), one-LLM-call module (#491), design-doc sweep (#492). | The as-built semantics cited throughout this chapter. |

Verification pattern when a comment cites a PR you doubt:

```bash
git -C /home/sunil/git/goldfive log --oneline --all --grep "#488" | head
git -C /home/sunil/git/goldfive log -S "steering_is_active" --oneline | head
```

## Your first change

Follow this exact sequence for your first edit in this repo. Do not skip steps.

1. **Set up and baseline.** From the repo root:

   ```bash
   cd /home/sunil/git/goldfive
   uv sync --extra dev --extra adk
   uv run pytest -q          # expect ~2912 passed, ~61 skipped, ~30s
   ruff check .              # expect: clean (zero findings)
   ```

   If the baseline is not green, STOP and report — do not build on a broken base.
   Note: the repo is deliberately NOT `ruff format`-clean. Never run a mass reformat;
   only `ruff check` (lint) must stay clean.

2. **If other agents may be running against this checkout, use an isolated worktree**
   (the main checkout is unsafe under concurrent agents):

   ```bash
   git -C /home/sunil/git/goldfive worktree add /tmp/gf-work -b my-change origin/main
   ```

3. **Classify your change** and open the matching chapter BEFORE editing:
   - new/changed drift detection → 07-deterministic-drift-detection.md or
     08-llm-judges.md, plus `.agents/how-to-add-a-drift-kind.md`;
   - ladder/intervention behavior → 09-steering-ladder-and-gates.md;
   - plan/refine behavior → 10-planning-and-revision.md;
   - anything touching `Session.state` or plan installs → 11-state-ownership.md;
   - events/sinks/proto → 12-events-sinks-telemetry.md;
   - config knobs → 14-config-reference.md (and check whether
     `goldfive/optimization/manifest.toml` + `tests/test_optimization_manifest.py`
     need a matching row);
   - step-by-step task templates → 16-recipes.md.

4. **Check the invariants table** at the top of this chapter against your plan. In
   particular: if your change reads the kill-switch, it must go through
   `steering_is_active(steerer)` / `is_active_steering()`; if it classifies natural
   language, it must be an LLM classifier; if it keys any gate, the key must be
   framework-minted and stable.

5. **Check the protected list.** If your diff touches `LOOPING_TOOL_CALL` surfaces,
   `PLAN_DIVERGENCE` machinery, or `reconciler.get_missed_tasks` — stop and ask for
   explicit human sign-off first. If your change would replicate anything from the
   unmerged `agency-preservation` branch (#453–#474) onto main — stop; that merge is
   a locked user decision.

6. **Make the edit, then verify** per 15-testing-guide.md: run the focused test file
   first, then the full suite, then lint:

   ```bash
   uv run pytest tests/<focused_file>.py -q
   uv run pytest -q
   ruff check .
   ```

7. **Self-review before reporting**: read your full diff once more, checking (a)
   correctness against the failure scenario you set out to fix, (b) that every new
   helper is actually wired into a real dispatch path (grep for call sites — unit
   tests pass even when a guard is dead code), (c) tests and docs ship WITH the code.

8. **Commit style**: no Claude co-author trailer in goldfive commits. CI runs
   lint-and-test on Python 3.11 and 3.12 with dev+adk+proto extras.

### Worked micro-example: your literal first edit

A safe, complete first task to calibrate the workflow — add a regression test for the
corrective-message fallback (`goldfive/steerer.py::compose_corrective_user_message`):

1. Read the function and its templates (`_CORRECTIVE_TEMPLATES`,
   `_GOAL_DRIFT_NOT_COMPLETE_TEMPLATE`) in `goldfive/steerer.py`. The contract you are
   pinning: a `GOAL_DRIFT` drift whose `current_task_id` is NOT COMPLETED in the
   refined plan must get the directive template ("Set task ... aside"), never the
   "already complete" assertion.
2. Find the existing test home: `grep -rln "compose_corrective_user_message" tests/`
   and add your case to that file (never create a parallel test file when one exists —
   see 15-testing-guide.md for suite layout and fixture conventions).
3. Build inputs with plain dataclasses — no mocks needed:

   ```python
   from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Plan, Task, TaskStatus
   from goldfive.steerer import compose_corrective_user_message

   def test_goal_drift_corrective_never_asserts_uncompleted_task_is_complete():
       plan = Plan(id="p", run_id="r", goal_ids=(), edges=(), tasks=(
           Task(id="t1", title="research", status=TaskStatus.RUNNING),
           Task(id="t2", title="draft slides", assignee_agent_id="app.drafter",
                status=TaskStatus.PENDING),
       ))
       drift = DriftEvent(kind=DriftKind.GOAL_DRIFT, severity=DriftSeverity.CRITICAL,
                          current_task_id="t1")
       msg = compose_corrective_user_message(drift=drift, refined_plan=plan)
       assert "already complete" not in msg      # truthfulness (#475)
       assert "draft slides" in msg              # points at next PENDING task
       assert "drafter" in msg                   # bare agent name, not dotted path
   ```

4. Run it focused, then the full gate:

   ```bash
   uv run pytest tests/ -q -k "corrective"
   uv run pytest -q && ruff check .
   ```

This touches no production code, exercises the vocabulary from this chapter
(drift event, corrective, truthful nudge text, PENDING/COMPLETED semantics), and
follows the exact loop every real change uses. Graduate to 16-recipes.md for
production-code recipes (add a detector, add a config knob, add an event field).

## Common mistakes

Concrete wrong edits a first-time (or weak-model) editor plausibly makes in
orientation-level territory, with the correct alternative.

| # | Wrong edit | Why it is wrong | Correct alternative |
|---|---|---|---|
| 1 | Add `if steerer._observation_only: return` (or read `config.steering.observation_only`) in a new intervention path. | Invariant 5: the only sanctioned kill-switch reads are the predicate pair; direct field reads bypass the fail-safe (missing/None/raising ⇒ PASSIVE) and were purged in #488. | `from goldfive.steerer import steering_is_active` and gate on `steering_is_active(steerer)`; on `DefaultSteerer` itself, call `self.is_active_steering()`. |
| 2 | "Fix the mismatch" by making `goldfive/drift/tool_loops.py` emit `DriftKind.LOOPING_TOOL_CALL` instead of `LOOPING_REASONING`. | PROTECTED KEEP (#204/#206): tool loops deliberately ride the `LOOPING_REASONING` ladder row (NUDGE-first CRITICAL routing); the producer is disambiguated by `DriftEvent.detector_name="tool_loops"`. | Leave the kind as-is. If routing must change, change the `_LADDER` row for `LOOPING_REASONING`/`LOOPING_TOOL_CALL` with human sign-off — see 09-steering-ladder-and-gates.md. |
| 3 | Add `re.compile(r"i can'?t|unable to")` (or a keyword list) to classify agent text in a new detector. | Invariant 2 — the exact anti-pattern retired in #166/#167. The `DriftKind.CONFUSION` retirement comment in `goldfive/types.py` documents this at length. | Route the text through an LLM judge (`goldfive/drift/reasoning_judge.py` pattern, or a pluggable `Judge`), or redesign so classification is unnecessary. Structured-data equality (tool names, `args_hash`) remains fine. |
| 4 | Delete "dead" enum members (`WRONG_AGENT`, `STOPPED_EARLY`, …) or the unreachable `PLAN_DIVERGENCE` ladder row during a cleanup. | Proto enum values are wire contract with harmonograf/zicato; `PLAN_DIVERGENCE` is an explicit KEEP (#252); external emitters may mint producer-less kinds. #490's deletions were archaeology-verified one by one. | Leave enum members and protected machinery alone. Dead-code deletion requires per-symbol history archaeology and, for protected items, human sign-off. |
| 5 | Give the agent an instruction ("call report_task_completed when done") and rely on it for run termination or task accounting. | Invariant 1: no prompt-cooperation contracts; the agent may never call it. | Rely on the `PlanReconciler` observation path; treat reporting tools as an optional accelerant that converges at `steerer.transition`. |
| 6 | Key a new cooldown/dedup/gate on `task.title`, an LLM-minted id, or a judge-minted condition string. | Invariant 6: churning keys mean the gate opens a fresh entry per observation and never engages. | Key on framework-minted stable identity: full agent path, `(kind, current_task_id)`, `discovery_identity_hash`, `condition_id` from `state_store` helpers. Fix upstream churn rather than coarsening the key. |
| 7 | Mutate `session.plan.tasks[i].status = ...` or `session.plan = new_plan` directly. | `Plan`/`Task` are frozen (#247); direct assignment raises or violates the single-writer invariant (WARNING, or raise under `GOLDFIVE_STRICT_STATE_OWNERSHIP=1`). | Use the `goldfive/types.py` helpers (`with_task_status`, `replace_task`, …) and install via `set_session_plan` / the `PlanReviser` paths. See 11-state-ownership.md. |
| 8 | Implement a feature described in `docs/design/AGENCY-PRESERVATION.md` "because the doc says it exists". | Stages 1–3 live ONLY on the unmerged `agency-preservation` branch (#453–#474), default-OFF; main must not copy from it, and step 13b is locked on user sign-off. | Treat that doc's §6 as-built status as branch-scoped. On main, implement only what main's code already supports. |
| 9 | Edit files under `goldfive/pb/` to change an event field. | Generated code; hand edits are overwritten and desync the wire contract. | Edit `proto/goldfive/v1/*.proto`, run `make proto`, then update `goldfive/conv.py` / `goldfive/events.py` round-trips. See 12-events-sinks-telemetry.md. |
| 10 | Add a new terminal `TaskStatus` member and update only one of the two terminal sources. | `TaskStatus.is_terminal` and `TERMINAL_TASK_STATUSES` are documented lock-step twins (`goldfive/types.py`); consumers import the frozenset (#485). | Update BOTH, and grep consumers: `grep -rn "TERMINAL_TASK_STATUSES" goldfive/`. |
| 11 | Run `ruff format .` (or an editor auto-format) across touched files. | The repo is intentionally not format-clean; mass reformat pollutes every future diff. | Match local style by hand; only `ruff check .` must pass. |
| 12 | Trust `docs/design/VOCABULARY.md` counts/locations when writing code (e.g. dispatch on "26 drift kinds" or patch `DefaultSteerer._handle_drift`). | The doc predates the facade split and taxonomy growth; several claims are stale (see the divergence table above). | Verify against `goldfive/types.py` and `goldfive/drift_observer.py` before citing; where doc and code disagree, the code wins. |
| 13 | Silence a failing sink or judge by letting its exception propagate "so we notice". | #479 hardened the opposite direction: sink exceptions never abort runs; malformed judge output degrades to INFO. Runs must survive observability failures. | Catch, log, degrade — follow the existing patterns in `goldfive/sinks/` and the judge-parse paths. |
| 14 | Hand-construct a proto `Event` envelope (or a bare dict) and push it to sinks. | Bypasses `run_id`/`sequence`/`session_id` stamping; breaks per-run monotonic ordering and harmonograf joins; `GRPCSink` skips dict envelopes. | Use (or add) a factory in `goldfive/events.py` built on `new_event`; see 12-events-sinks-telemetry.md. |
| 15 | In a new detector, read `session.plan.revision_index` AFTER an `await call_llm(...)` (or not at all) when stamping the drift. | The #245 freshness gate depends on `observed_revision_index` being captured BEFORE the LLM round-trip; stamping after (or never) either defeats staleness detection or bypasses the gate entirely. | Capture `session.plan.revision_index` at the TOP of the detector, before any await, and stamp it on the `DriftEvent` — the documented pattern in `goldfive/types.py::DriftEvent.observed_revision_index`. |
| 16 | Blame the wrapped model for slow turns or "no activity" and add model-side workarounds. | Repo history: 5+ min/turn is almost always goldfive code (unbounded max_tokens, drift loops, missing wall-clock budgets); and external evidence (GPU activity, server logs) trumps an incomplete log reading. | Check goldfive-side budgets first (`llm_call_timeout_ms`, stall watchdog, judge semaphore); verify what build is actually running before diagnosing a recent change. |

## Verification checklist

Run these after ANY change made while following this chapter. Commands are exact; run
from the repo root (`/home/sunil/git/goldfive`).

1. Full suite and lint (the universal gate):

   ```bash
   uv run pytest -q        # ~2912 passed / ~61 skipped in ~30s
   ruff check .            # must be clean; do NOT run `ruff format`
   ```

2. If you touched anything near the kill-switch — prove no new direct reads exist
   (expect matches only in `goldfive/steerer.py`, `goldfive/config.py`, and tests):

   ```bash
   grep -rn "_observation_only" goldfive/ | grep -v "steerer.py\|config.py"
   grep -rn "is_active_steering\|steering_is_active" goldfive/ | head -50
   ```

3. If you touched `DriftKind` or the ladder — confirm enum/proto/ladder coherence:

   ```bash
   grep -c "= \"" goldfive/types.py | head -1          # sanity: count enum literals region
   grep -n "class DriftKind" goldfive/types.py proto/goldfive/v1/types.proto goldfive/pb/goldfive/v1/types_pb2.pyi
   grep -n "DriftKind\." goldfive/drift_observer.py | grep "_IL\." | head -40   # ladder rows
   uv run pytest tests/ -q -k "drift and (kind or ladder or taxonomy)"
   ```

4. If you added any NL-classification logic — prove it is not regex/keyword-based
   (expect NO new hits beyond the documented legacy surfaces:
   `_DISCOVERY_TOKEN_RE` in `types.py`, refusal markers and
   `CONFABULATION_TRIGGER_KEYWORDS` in `drift/__init__.py`):

   ```bash
   grep -rn "re.compile\|re.search\|re.match" goldfive/drift/ goldfive/drift_observer.py
   ```

5. If you added a helper that gates or guards a dispatch path — prove it is wired
   (dead-code guard hazard):

   ```bash
   grep -rn "<your_helper_name>" goldfive/ --include="*.py" | grep -v "def <your_helper_name>\|test"
   ```

   Zero call sites outside its definition means the guard is dead code — wire it or
   remove it.

6. If you touched config defaults — keep the optimization manifest honest:

   ```bash
   uv run pytest tests/test_optimization_manifest.py -q
   ```

7. If you touched protos or events:

   ```bash
   make proto
   uv run pytest tests/ -q -k "event or sink or conv"
   ```

8. Cross-references sanity: this chapter's claims about pipeline shape, ladder rows,
   and file map were verified against main on 2026-07-05 (post-#492). If you are
   reading this much later, spot-check the two most drift-prone anchors before
   relying on them:

   ```bash
   grep -n "class DriftKind" goldfive/types.py
   grep -n "_load_ladder_tables\|def _ladder_level_for" goldfive/drift_observer.py
   ```

Next: 02-architecture-map.md for the component wiring in depth, 15-testing-guide.md
before your first test run, 16-recipes.md for step-by-step task templates, and
17-invariants-hazards-history.md for the full invariant catalog with history.
