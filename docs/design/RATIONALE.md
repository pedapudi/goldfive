# Design rationale

This document answers the question "why is goldfive shaped this way?"
for each major abstraction. Every entry uses the same template:

- **Observation** — what someone new to the code will notice first.
- **Intent** — what the design is trying to achieve.
- **Alternatives considered** — sibling designs that were rejected.
- **Tradeoffs** — what we gave up.
- **Signals this might be wrong** — the futures that would send us back
  to the drawing board.
- **Related** — pointers into the rest of the docs.

If you are reading goldfive for the first time and any choice feels
arbitrary, check here before assuming it is.

Related:

- [VOCABULARY.md](VOCABULARY.md) — the type-system reference.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the six primitives.
- [PROTOCOLS.md](PROTOCOLS.md) — the protocol contracts.
- [STATE-MACHINE.md](STATE-MACHINE.md) — task lifecycle.
- [DRIFT.md](DRIFT.md) — drift taxonomy.
- [EVENT-MODEL.md](EVENT-MODEL.md) — event envelope.

## Why `Goal` is first-class

**Observation.** `Runner.run` takes `user_input: str | list[Goal]`.
Every run begins with a `list[Goal]` in `session.goals`. The `Goal`
dataclass has `id`, `summary`, `success_predicate`, and `metadata`. The
plan's `goal_ids` list names goals explicitly.

**Intent.** Separate *what the user asked for* from *what the agent
is currently working on*. A `Goal` outlives a plan revision; a `Task`
does not. When `planner.refine` produces a fresh plan after a
`USER_STEER`, the goals stay the same while tasks get rewritten.

**Alternatives considered.**

1. `user_input: str` only — drop `Goal` entirely. Rejected: every
   planner would have to re-derive its own interpretation of intent,
   and a refine after STEER would have no stable handle on "the user's
   original request."
2. `Goal` = `str` alias — no dataclass. Rejected: we need the
   `success_predicate` escape hatch for closed-loop goals (e.g. "don't
   stop until this predicate returns True on the session"), and a
   `metadata` dict for planner-private context. Promoting to a
   dataclass costs almost nothing.
3. Merge `Goal` and `Plan` — always have the caller supply a plan.
   Rejected: most callers either want the planner to produce the plan
   or want a single-goal passthrough. Forcing plan authorship removes
   the main ergonomic win.

**Tradeoffs.** Two-layer modelling (goal → plan) is slightly more to
teach than one layer. In return, refine has a stable anchor and logs
carry enough provenance to reconstruct intent after many revisions.

**Signals this might be wrong.** If in practice every goal is the
literal user prompt with no predicate and no metadata, we have paid
for a layer nobody uses. The `LiteralGoalDeriver` / `PassthroughGoalDeriver`
pair makes this cheap, so even in that limit the cost is minimal.

**Related.** [PROTOCOLS.md §GoalDeriver](PROTOCOLS.md#goalderiver),
[VOCABULARY.md §"`Goal`"](VOCABULARY.md#1-the-four-semantic-enums).

## Why `Plan` and `Session` are separate

**Observation.** `Plan` is a DAG of tasks; `Session` holds `plan:
Plan | None` plus `run_id`, `goals`, `completed_results`,
`task_progress`, `agent_notes`, `history`, `divergence_flag`, and a
sequence counter. The Plan can be swapped by `refine` mid-run; the
Session persists across the swap.

**Intent.** Make the plan mutable-by-replacement while keeping durable
state (what completed, what the agent wrote down) on an object that
does not churn on every revision. When refine returns a new plan, the
Steerer reassigns `session.plan = revised` — everything else stays put.

**Alternatives considered.**

1. One big `RunState` object — fold the plan's tasks and edges into
   `Session`. Rejected: refine would need to reach into the Session and
   surgically edit tasks, losing the nice property that "a plan is a
   value you can log, serialize, and diff."
2. Multiple `Session` objects per run, one per plan revision. Rejected:
   sequence counters, `completed_results`, and progress state have to
   be continuous; threading them across session objects is error-prone.
3. Put `completed_results` on the plan itself. Rejected: completed
   results are tied to a `Task.id`, and we preserve `Task.id` across
   revisions — but we also don't want to assume *every* completed task
   survives a revision as-is. Keeping results on the session means a
   refine can drop a completed task from the plan without destroying
   its recorded output. (In practice refine must not drop completed
   tasks, but the invariant belongs in the planner contract, not in
   the data layout.)

**Tradeoffs.** Two types to teach instead of one. In return, refine is
a single-pointer reassignment and the serialization story for plans
(log them into `PlanSubmitted` / `PlanRevised` events verbatim) falls
out for free.

**Signals this might be wrong.** If callers frequently want to iterate
a session's task history and find themselves reaching for
`session.plan.tasks + session.history` patterns, we may need a helper
method or a flatter layout.

**Related.** [ARCHITECTURE.md](ARCHITECTURE.md),
[VOCABULARY.md §4](VOCABULARY.md#4-taskstatus-state-machine).

## Why `GoalDeriver` is a separate protocol from `Planner`

**Observation.** `GoalDeriver.derive(user_input) -> list[Goal]` and
`Planner.generate(goals, available_agents, context) -> Plan | None` are
two different protocols. In many runs they are backed by the same LLM.

**Intent.** Separate the concerns so a caller can replace one without
replacing the other. The set of goals is the contract the run is judged
against; the plan is a tactical artifact. Callers that already know
their goals (a REST API, a cron job) bypass derivation entirely by
passing `list[Goal]` to `Runner.run`. Callers that want to own planning
but not derivation (a product with a fixed goal vocabulary but flexible
agent dispatch) swap only the planner.

**Alternatives considered.**

1. One `PlanGenerator` interface — `(user_input, available_agents) ->
   (goals, plan)`. Rejected: fuses two independently-useful swaps.
2. `Goal` is derived by the planner on the fly during `generate`.
   Rejected: `PlanSubmitted` events would have to carry goals
   retroactively, and `Runner.run(list[Goal])` would have to simulate a
   passthrough planner.
3. Put derivation on the adapter. Rejected: deriving goals is
   framework-agnostic reasoning about user intent; it should not depend
   on the agent framework.

**Tradeoffs.** Two protocols instead of one. Callers writing both
must decide how to share an LLM between them; `wrap` handles this by
detecting the LLM and passing the same `call_llm` to both.

**Signals this might be wrong.** If in every real deployment the
`GoalDeriver` and `Planner` share an LLM, a shared prompt, and are
always replaced together, the separation is paying no dividend.

**Related.** [PROTOCOLS.md §GoalDeriver](PROTOCOLS.md#goalderiver),
[PROTOCOLS.md §Planner](PROTOCOLS.md#planner).

## Why `Runner` is the public entry point instead of calling `Executor` directly

**Observation.** You cannot use goldfive without a `Runner` — even
`goldfive.wrap` returns a `Runner`. Yet the executor's `run(...)`
method has a full signature (plan, session, adapter, steerer, planner,
sinks). It would be tempting to construct a session by hand and call
`executor.run(...)` directly.

**Intent.** The Runner owns the `Run*` lifecycle semantics that are
universal and which should not be re-implemented per executor:

- Generating `run_id`.
- Emitting `RunStarted` / `GoalDerived` / `PlanSubmitted` at the right
  moments.
- Calling `goal_deriver.derive` (or accepting `list[Goal]` directly).
- Calling `planner.generate` and swapping the result onto the session.
- Registering the seven canonical reporting tools on the adapter.
- Binding the steerer to sinks+planner.
- Emitting `RunAborted` on any pre-executor failure.

Without the Runner, every new executor would re-implement those eight
steps, and the event-ownership invariants would drift. With the
Runner, executors only need to own the per-task loop, task events, and
the terminal `RunCompleted`/`RunAborted`.

**Alternatives considered.**

1. Make the Executor the entry point; have the Runner be a thin wrapper
   that future-deprecates. Rejected: the Runner is not a wrapper, it
   is the *composition* of all six primitives. It is where sink
   fan-out, pre-executor aborts, and `close()` live.
2. Make the user construct a `Session` and hand it to the Executor.
   Rejected: too much ceremony; quickstart would become five lines
   longer for no gain.

**Tradeoffs.** Two concepts to teach (Runner vs Executor) instead of
one. In return, the event ownership map is stable: Runner-owned events
vs Executor-owned events vs Steerer-owned events never overlap.

**Signals this might be wrong.** If a new executor wanted to skip one
of the Runner's pre-executor steps (say, decline to register reporting
tools), we would need a way to opt out — at which point the current
split would bend. Currently no such need has surfaced.

**Related.** `goldfive/runner.py` module docstring,
[ARCHITECTURE.md §"The full lifecycle"](ARCHITECTURE.md#the-full-lifecycle).

## Why `Executor` is a protocol instead of a function

**Observation.** `SequentialExecutor` and `ParallelDAGExecutor` both
satisfy a single protocol. The protocol has one method, `async run`.
You could imagine compressing it into a callable.

**Intent.** Executors hold non-trivial per-run state (in-flight task
map, rewind bookkeeping, reinvocation counter, paused flag, control
drain helpers). A protocol lets each executor keep its own state
without forcing every caller to thread the state through function
parameters. It also means third parties can drop in a custom executor
(domain-specific scheduling, cloud-native fan-out) without goldfive
needing to know about it.

**Alternatives considered.**

1. `Executor = Callable[[...], Awaitable[ExecutionOutcome]]`.
   Rejected: in-flight state has to go somewhere; closures or context
   objects end up reconstituting a class with extra steps.
2. One concrete `Executor` class with a `mode` enum. Rejected: fans
   out the implementation into conditionals, and closes the door on
   custom executors that don't fit either mode.

**Tradeoffs.** Callers pay the cost of knowing about "protocols" and
the naming overhead of each concrete class.

**Signals this might be wrong.** If in every non-trivial use the
parallel/sequential distinction collapses into a single implementation
with a concurrency knob, we have two classes where we needed one.

**Related.** [PROTOCOLS.md §Executor](PROTOCOLS.md#executor),
[VOCABULARY.md §6](VOCABULARY.md#6-event-payload-kinds).

## Why `Steerer` is a protocol, and what `DefaultSteerer` does

**Observation.** The Steerer protocol has four methods: `observe`,
`transition`, `detect_drift`, `bind`. `DefaultSteerer` is the canonical
implementation and owns:

- The task state machine (`mark_task_*` → `TaskStatus` transitions).
- Drift classification (`detect_drift`).
- Drift handling (`_handle_drift` → emit `DriftDetected`, optionally
  call `planner.refine`, apply the revision).
- Event fan-out to sinks for every transition.

**Intent.** The Steerer is the **only** component that mutates live
run state. Executors compute, adapters run agents, planners compute
plans — the Steerer is the funnel through which all state-changing
decisions flow. Making it a protocol lets callers swap in a custom
policy (different refine threshold, different drift taxonomy, an LLM-
driven drift classifier) without replacing the executor.

**Alternatives considered.**

1. Fold the state machine into the Executor. Rejected: every executor
   would re-implement the same transitions with the same off-by-one
   bugs. And a custom executor wouldn't get drift detection for free.
2. Fold drift detection into the Adapter. Rejected: drift signals come
   from multiple places — tool errors, text refusals, stop reasons,
   reporting-tool calls, user steers. Putting all of that in the
   adapter layer would make adapters enormous and framework-specific.
3. One monolithic `Engine` that combines Steerer + Executor. Rejected:
   blurs the "what is going on" (Steerer) vs "what do I do next"
   (Executor) split that makes the rest of the contract clean.

**Tradeoffs.** Three protocols touch each other (Executor calls
Steerer, Steerer calls Planner, all emit to Sinks). In return, every
responsibility has a name and a contract.

**Signals this might be wrong.** If a user of goldfive wants to add a
new reporting tool and finds they have to edit both the Adapter
(register) and the Steerer (mutate on call), the handoff may be too
costly. Today the reporting-tool protocol in
`goldfive/reporting.py` keeps this small.

**Related.** [PROTOCOLS.md §Steerer](PROTOCOLS.md#steerer),
`goldfive/steerer.py`.

## Why `AgentAdapter` exists and isn't the agent itself

**Observation.** You cannot hand a Google ADK `BaseAgent` or a Claude
SDK client to `Runner` directly. You must wrap it in an
`AgentAdapter` (or let `goldfive.wrap` do it for you).

**Intent.** Separate "how to invoke" from "how to be invoked." The
adapter knows how to:

- Render the current `Task` + `Session` into whatever input channel
  the framework expects.
- Register the seven canonical reporting tools in the framework's
  tool-registration format.
- Intercept tool calls and route them through the Steerer.
- Return an `InvocationResult` with the stop reason and final text.

The agent itself — the `BaseAgent`, the client, the callable — does
none of this. It just answers. Putting the adapter concerns on the
agent would force every agent author to learn goldfive's contract.

**Alternatives considered.**

1. Subclass `BaseAgent` with goldfive hooks. Rejected: requires
   framework modification, prevents dropping in an off-the-shelf agent.
2. Monkeypatch the agent at wrap time. Rejected: opaque, fragile, and
   forbidden in many agent frameworks.
3. Make the adapter optional — accept "either an AgentAdapter or a
   bare agent" in `Runner`. Rejected: the logic to figure out which is
   which still lives somewhere; we simply moved the decision inside
   `Runner` instead of making the caller pass `wrap`.

**Tradeoffs.** One extra class per supported framework. We ship three
(`CallableAdapter`, `ADKAdapter`, `ClaudeAgentSDKAdapter`) and
`auto_adapter` auto-selects at wrap time.

**Signals this might be wrong.** If a new framework resists being
adapted — say its invocation model is not request/response but
infinite streaming — we may need to generalize `AgentAdapter.invoke`
or add a sibling protocol.

**Related.** [PROTOCOLS.md §AgentAdapter](PROTOCOLS.md#agentadapter),
`goldfive/adapters/`, [.agents/adapters.md](../../.agents/adapters.md).

## Why `EventSink` protocol is proto-Event-shaped, not dict-shaped

**Observation.** `EventSink.emit(event_pb: Any)` takes a proto `Event`
envelope. Typed factories in `goldfive/events.py` build those
envelopes. There is also a `make_event(run_id, sequence, kind,
payload)` helper that returns a `dict`.

**Intent.** Proto is the source of truth. The schema in
`proto/goldfive/v1/events.proto` gives us:

- **Wire stability.** Field numbers are frozen; sinks written against
  v0.1 keep working on future versions.
- **Observability integration.** Downstream systems (harmonograf, gRPC
  clients, SQLite readers) can deserialize without a goldfive import.
- **Forward compatibility.** New event kinds extend the `oneof`
  payload; old sinks ignore unknowns.

The `make_event` dict path coexists as an **escape hatch**: tests and
lightweight tools that don't want the `proto` extra installed can still
emit dict envelopes, and dict-aware sinks (`JSONLPersistenceSink`,
`InMemorySink`, `LoggingSink`) round-trip them.

**Alternatives considered.**

1. Dict-only schema. Rejected: no wire stability, no
   cross-language consumption, types drift.
2. Proto-only with no escape hatch. Rejected: tests that don't need
   proto would have to install the extra, and `make_event` is the
   single place where a minimal dict envelope gets formatted.

**Tradeoffs.** Two envelope shapes exist in parallel. Sinks must
duck-type on `DESCRIPTOR`. Issue #53 tracked a bug caused by mixing
shapes incorrectly; the fix was careful sink authoring, not removal
of either path.

**Signals this might be wrong.** If we grow a third envelope shape
(some new serialization format), we should seriously consider
consolidating back to proto-only.

**Related.** [EVENT-MODEL.md](EVENT-MODEL.md),
[.agents/debug-goldfive.md §"Common pitfalls"](../../.agents/debug-goldfive.md#common-pitfalls).

<!-- TODO: once docs/design/CONTROL.md lands, cross-link from the
ControlChannel, STEER "delete-and-replan", and "Phase 1 mid-task cancel"
entries below. -->

## Why `ControlChannel` is a primitive rather than methods on Runner

**Observation.** `Runner.pause()` / `Runner.cancel()` / `Runner.steer()`
do not exist. Instead, callers construct a `ControlChannel`, pass it
to `Runner(control=...)` (or `wrap(control=...)`), and send
`ControlMessage` objects through the channel.

**Intent.** Control is **bidirectional** and **external**. The UI runs
in a different process. A test harness runs in a different task. A CLI
runs in a different thread. Making the control surface an async
queue-pair means:

- The Runner never has to be reachable by the caller mid-run — the
  channel is.
- Acks flow back through the same primitive, so external callers know
  whether their control took effect.
- Bridges (e.g. `harmonograf_client.observe`) can translate their own
  wire protocol into `ControlMessage` and plug in without knowing
  anything about goldfive's internals.

**Alternatives considered.**

1. `Runner.pause()` etc. as methods. Rejected: requires a stable
   reference to the Runner across processes, couples control semantics
   to the Runner lifetime, and has no natural place for async acks.
2. A single `asyncio.Queue[ControlMessage]` (no acks). Rejected: the
   UI needs to know whether its cancel was honored or whether it
   arrived after a terminal state.
3. Signals (SIGINT, SIGUSR1). Rejected: not expressive enough for
   STEER's payload; signals don't cross process boundaries cleanly in
   general agent deployments.

**Tradeoffs.** Two queues instead of zero. Callers must understand
that they drain acks on a separate async iterator. In practice the
bridge hides both queues from the UI caller.

**Signals this might be wrong.** If every real-world caller ends up
writing the same ack-draining bridge by hand, we should either ship a
default bridge or build ack handling into the Runner.

**Related.** [VOCABULARY.md §2](VOCABULARY.md#2-controlkind-vs-driftkind--a-worked-example),
`goldfive/control.py`.

## Why STEER is "delete-and-replan" instead of "modify-in-place"

**Observation.** When the user steers, the planner's `refine` for
`USER_STEER` drops every non-completed task and generates a fresh task
list. Completed tasks are preserved. It does not surgically modify the
existing plan.

**Intent.** User steering is a **semantic signal**, not a diff. When
the user says "focus on slide 3 instead", they are not asking for the
planner to tweak task descriptions; they are asking for a new
conception of the remaining work that takes their note seriously.
Delete-and-replan guarantees:

- Completed work's identity and output is preserved (invariant from
  Planner contract, cross-referenced below).
- Every remaining task was *generated in light of the steering note*,
  not inherited from a pre-steering plan.
- The revision is visible and auditable: `PlanRevised` carries the
  full new plan, not a patch.

**Alternatives considered.**

1. In-place edit: planner modifies task titles, shuffles
   dependencies, leaves some pending tasks. Rejected: the invariant
   "every pending task reflects current user intent" becomes hard to
   state, and diff-style refine output complicates the event record.
2. User-supplied task list (no LLM). Rejected: takes the intelligence
   out of the loop for the very case where we need it most.
3. Refine is "appendonly" — new tasks are appended to existing pending
   tasks. Rejected: the user's steer is a redirection, not an
   addition; appendonly makes that case awkward.

**Tradeoffs.** Wasted compute on pending tasks whose work would have
overlapped with the new plan. In practice, the pending tasks have not
yet executed, so no agent-time is lost — only planning time.

**Signals this might be wrong.** If deployments show heavy LLM-planner
cost per steer, or if callers write custom planners that *do* do
in-place edits and want first-class support, we'd revisit.

**Related.** `DriftKind.USER_STEER` in
[VOCABULARY.md §5.d](VOCABULARY.md#5d-user-driven--an-external-verb-entered-the-pipeline).

## Why `goldfive.wrap()` exists when `Runner(...)` would do

**Observation.** `Runner.__init__` requires five named pieces:
`agent`, `planner`, `executor`, plus the optional `goal_deriver`,
`steerer`, `sinks`, `control`, `max_task_invocations`. Most callers
have one of {`BaseAgent`, `Client`, `callable`} and want to say "just
run this with sensible defaults." `goldfive.wrap(agent)` does that in
one call.

**Intent.** Close the ergonomics gap between "I have an agent" and "I
have a `Runner`." `wrap` does three things:

1. **Auto-adapter detection.** `auto_adapter(agent)` dispatches to
   `ADKAdapter`, `ClaudeAgentSDKAdapter`, `CallableAdapter`, or
   passes through an existing `AgentAdapter`.
2. **LLM detection.** If the agent carries an LLM handle goldfive
   knows how to reuse (currently ADK), it threads that into an
   `LLMPlanner` and `LLMGoalDeriver`.
3. **Sensible defaults.** Falls back to `PassthroughPlanner` +
   `LiteralGoalDeriver` when no LLM is detected, so `wrap` never
   crashes at construction.

**Alternatives considered.**

1. Make `Runner.__init__` auto-detect too. Rejected: the Runner's
   signature is intentionally explicit — it's the contract. `wrap`
   can *relax* that contract; `Runner` cannot.
2. `quickstart(agent, goals)` only. Rejected: doesn't cover callers who
   want `.run(user_input)` with auto-derivation. `quickstart` already
   exists as a more opinionated cousin (static plan from goals).

**Tradeoffs.** Two entry points (`Runner` and `wrap`) — plus `run` and
`quickstart`. Three idioms to teach. The docs pick one canonical path
(`wrap` for most callers) and show the others as escape hatches.

**Signals this might be wrong.** If `wrap`'s auto-detection silently
picks a wrong adapter or planner, callers get baffling behavior. The
current debug-log messages help; a future `wrap(dry_run=True)` that
prints the resolved component list would help more.

**Related.** `goldfive/convenience.py`,
[README.md](../../README.md#hello-goldfive).

## Why harmonograf is a sink rather than an executor plugin

**Observation.** You plug harmonograf in by supplying a sink to
`Runner(sinks=[HarmonografSink(...)])` (or via
`harmonograf_client.observe(runner)`), not by passing an "observed"
executor. Observability lives one level outside orchestration.

**Intent.** Layer boundaries:

- **goldfive** is the **orchestration library**. It plans, executes,
  steers, and emits events.
- **harmonograf** is the **observability console**. It subscribes to
  events and optionally sends control messages back.

Keeping observability on the sink boundary means you can:

- Run goldfive with no observer (production CLI).
- Run goldfive with logging-only observability.
- Run goldfive with harmonograf.
- Run goldfive with your own custom observer.

…all without touching the orchestration layer.

**Alternatives considered.**

1. Executor-plugin API: `SequentialExecutor(plugins=[harmonograf])`.
   Rejected: conflates "what schedules work" with "what watches work",
   and every new executor would need its own plugin contract.
2. Direct harmonograf dependency from goldfive. Rejected: goldfive has
   to be usable without harmonograf installed.

**Tradeoffs.** `observe(runner)` is two different concepts from
`wrap(agent)` — the first adds a sink (and, for live steering, a
control bridge), the second adds an adapter. New users sometimes
conflate them. The docs split them cleanly by file and by
verb.

**Signals this might be wrong.** If observability features routinely
need hooks that sinks can't reach (executor decisions, steerer internal
state), we may need a more expressive contract. So far the event
stream has been enough.

**Related.** `goldfive/sinks/`, [harmonograf-integration guide][hg].

[hg]: ../guides/harmonograf-integration.md

## Why drift severity is a 3-level enum (`INFO`/`WARNING`/`CRITICAL`) not a number

**Observation.** Severities are discrete: `INFO`, `WARNING`,
`CRITICAL`. No `NOTICE`, `ERROR`, `FATAL`, or numeric scales.

**Intent.** Three levels map to three discrete operator actions:

- `INFO`: see it in the log, take no action.
- `WARNING`: take the "replan" action.
- `CRITICAL`: take the "abort or replan-then-maybe-abort" action.

Finer granularity would require each level to correspond to a specific
automated action, and we do not have more than three actions to
assign.

**Alternatives considered.**

1. Numeric severity (0-100). Rejected: forces every caller to pick a
   number, and forces the Steerer to pick a threshold that looks
   arbitrary.
2. Five-level (`DEBUG, INFO, WARN, ERROR, FATAL`). Rejected: extra
   levels without extra actions just confuse operators.
3. Boolean `is_recoverable` flag. Rejected: conflates severity with
   recoverability; some `WARNING`s are not recoverable (a permanent
   schema violation) and some `CRITICAL`s are recoverable with a
   major replan.

**Tradeoffs.** Callers with domain-specific gradation have to shoehorn
into three buckets. They can always extend the detail string or use a
caller-owned label in `metadata`.

**Signals this might be wrong.** If we find ourselves repeatedly
synthesizing drifts at "WARNING but really WARNING-er" severity and
writing custom threshold comparisons, the enum is too coarse.

**Related.** [DRIFT.md §"Severity levels"](DRIFT.md#severity-levels),
[VOCABULARY.md §7](VOCABULARY.md#7-severity-ladder).

## Why `make_event` (dict) coexists with typed factories (proto)

**Observation.** `goldfive/events.py` exports both
`run_started_event(...)` (proto) and `make_event(run_id, sequence,
kind, payload)` (dict). Tests and a couple of lightweight paths use
the dict shape. `GRPCSink` only accepts proto.

**Intent.** Keep goldfive usable without the `proto` optional
dependency. The proto stubs are large and require `grpcio-tools` at
generation time. Some tests and some downstream tools (the
`harmonograf_client` ack formatters, small scripts) don't need proto;
forcing it would bloat their install footprint.

**Alternatives considered.**

1. Proto-only. Rejected: the `proto` extra becomes effectively
   required, breaking the "pure Python" promise for some use cases.
2. Dict-only. Rejected: loses wire stability and forward compatibility.
3. Convert dict ↔ proto on every emission. Rejected: adds cost to the
   hot path and a subtle class of mixed-shape bugs.

**Tradeoffs.** Sinks have to duck-type on `DESCRIPTOR` to distinguish
the two shapes. An earlier bug (#53) came from `LoggingSink` assuming
proto; the fix was a dict fallback. We kept dict anyway because the
alternative — forcing proto everywhere — is worse.

**Signals this might be wrong.** If downstream tooling uniformly
installs the proto extra, the dict path is dead code. Today it is
not — `make_event` is called from several test scenarios and from
`harmonograf_client` fallbacks.

**Related.** [EVENT-MODEL.md §"Forward compatibility"](EVENT-MODEL.md#forward-compatibility),
[.agents/debug-goldfive.md §"Common pitfalls"](../../.agents/debug-goldfive.md#common-pitfalls).

## Why `max_task_invocations` is a budget, not a time limit

**Observation.** `Runner(..., max_task_invocations=N)` and its
per-executor equivalent cap the **number** of adapter invocations
(in the sequential executor) or plan refinements (in the parallel
executor) that a single run may issue before aborting. The default
is `None` — unbounded. There is no wall-clock timeout at this layer.

**Intent.** Differentiate **stuck** from **slow**. An agent that takes
ten minutes per task is slow (fine, maybe). An agent that loops
through five refines per minute is stuck (not fine). A time budget
would punish the slow case; an invocation budget only punishes the
loopy case. The unbounded default reflects that the primary guards
against runaway loops live closer to the work: per-task-lineage
caps (`max_retries_per_task_lineage`), per-tool-loop caps inside
the adapter, and the adapter's own max-LLM-call ceilings.

**Alternatives considered.**

1. Wall-clock timeout. Rejected: too blunt for mixed workloads.
2. Exponential-backoff between refines. Rejected: doesn't bound total
   work, just spaces it.
3. Finite default (historically 32). Rejected: the old name
   `max_plan_reinvocations` misled callers into thinking the value
   was a refine-count cap and some set it to the number of expected
   refinements (e.g. 8), which then tripped on routine large plans.
   Unbounded-by-default plus explicit opt-in avoids that footgun.

**Tradeoffs.** Without a configured cap a truly stuck agent keeps
burning invocations until the per-lineage cap or `fail_fast`
catches it. Callers who want a belt-and-suspenders ceiling can set
a finite integer.

**Historical note.** The parameter was formerly called
`max_plan_reinvocations`; that name implied a refine-count semantic
that did not match the sequential executor's "total adapter
invocations" behaviour. The rename is backwards-compatible for one
release via a deprecation shim.

**Signals this might be wrong.** If callers regularly want wall-clock
timeouts, we may need a sibling knob. Today none do.

**Related.** `goldfive/runner.py::Runner` constructor docstring,
[ARCHITECTURE.md §"5. Termination"](ARCHITECTURE.md#5-termination).

## Why `BLOCKED` is a task status rather than a drift kind

**Observation.** Both `TaskStatus.BLOCKED` and `DriftKind.BLOCKED`
exist. They represent different aspects of the same event.

**Intent.** Status and drift are orthogonal:

- `TaskStatus.BLOCKED` is a **position in the task lifecycle** — the
  task is still alive, not terminal, and may resume. Executors use the
  status to decide what to dispatch on the next tick.
- `DriftKind.BLOCKED` is the **observation that a task *just* entered
  BLOCKED** — it is a one-shot signal that flows through refine. The
  planner decides whether to route around the blocker, wait, or give
  up.

Most drift kinds do not need a status counterpart. `CONTEXT_PRESSURE`
is about an invocation, not a task lifecycle state. `TOOL_ERROR` is
about a tool call. Only `BLOCKED` is both a durable lifecycle state
and a refinable event.

**Alternatives considered.**

1. `BLOCKED` as drift only; no status. Rejected: the executor needs a
   way to say "this task is waiting" when it iterates. Without a
   status, it would have to cross-reference the drift history on
   every tick.
2. `BLOCKED` as status only; no drift. Rejected: refining out of
   blocked is the happy path. Without a drift, `_handle_drift` would
   not fire and the planner would not hear about the blocker.

**Tradeoffs.** Two `BLOCKED` symbols can confuse a reader. The tables
in [VOCABULARY.md §4](VOCABULARY.md#4-taskstatus-state-machine) and
[§5.c](VOCABULARY.md#5c-runtime--the-environment-limited-progress)
show them side-by-side so the dual role is visible.

**Signals this might be wrong.** If callers routinely want a task to
enter `BLOCKED` silently (no drift, no refine), we would need a
severity-zero variant or a separate `mark_task_waiting`. Today no such
need has surfaced.

**Related.** [VOCABULARY.md §4](VOCABULARY.md#4-taskstatus-state-machine),
[STATE-MACHINE.md §"Blocked vs non-blocked resume"](STATE-MACHINE.md#blocked-vs-non-blocked-resume).

## Why Phase 1 includes mid-task cancel

**Observation.** The sequential executor races `adapter.invoke(task,
session)` against `channel.receive()` so a CANCEL or STEER mid-task
interrupts an in-flight adapter call. The parallel executor does the
same per stage.

**Intent.** Live UX requires mid-task cancel. A user who clicks
"Cancel" while the agent is still typing expects it to stop — not in
30 seconds when the task finishes, but now. Between-task-only cancel
would be a regression from the console's current behavior.

**Best-effort, not guaranteed.** The executor calls `invoke_task.cancel()`
and waits up to a small timeout; after that, it abandons the coroutine
and proceeds with the cancel. If the adapter doesn't honor
`CancelledError` (e.g. wraps it in a `try/except Exception`), the
adapter coroutine continues running in the background until it
completes naturally. That's acceptable because:

- The cancel effect is about run control, not adapter resource
  cleanup. The run aborts either way.
- Adapters that *do* honor cancellation get the fast path.
- Adapters that don't will eventually finish and become orphan tasks;
  asyncio's garbage collection logs them.

**Alternatives considered.**

1. Between-task-only cancel. Rejected: UX regression, see above.
2. Forcible thread kill. Rejected: Python doesn't support it cleanly,
   and adapters may hold locks.
3. Wait arbitrarily long for cancel. Rejected: defeats the purpose.

**Tradeoffs.** "Best-effort" is a contract users have to accept.
Adapter authors who want first-class mid-task cancel must cooperate
with `CancelledError`.

**Signals this might be wrong.** If common adapters routinely leak
tasks past cancel, we may need a stricter kill mechanism (process
boundary, subprocess executor).

**Related.** `goldfive/executors/sequential.py::_invoke_with_control`,
`goldfive/executors/_control.py::_ControlCancelled`.

## Why we preserve completed task outputs across a STEER replan

**Observation.** After `USER_STEER` refine, `session.completed_results`
still contains every completed task's output. The planner contract
requires preserving completed tasks' identity and outcome in the
revised plan.

**Intent.** Avoid wasted work and preserve user-visible progress. A
user steers because the current direction is wrong *going forward*,
not because everything done so far was wrong. If steering threw away
the research phase output, users would learn to never steer — they'd
restart from zero instead, which is strictly worse.

Preserving completed tasks also:

- Makes `session.completed_results` a durable record across revisions.
- Gives the planner's refine prompt concrete context ("here's what
  you've already produced; incorporate or set aside, don't redo").
- Keeps `revision_index` meaningful — the run has a continuous history.

**Alternatives considered.**

1. Discard everything on STEER. Rejected: see above. Also degrades
   event history — replay would show work being done and then
   ignored.
2. Let the planner decide per-task. Rejected: leaks the "preserve
   identity of completed tasks" invariant into every planner
   implementation. The current contract makes it impossible to violate.
3. Branching runs: old run stops, new run starts with the outputs as
   seed data. Rejected: heavyweight, conflicts with `run_id` being a
   single identifier per `Runner.run()`.

**Tradeoffs.** If the steer is "abandon everything and start over,"
callers must send a `CANCEL` followed by a fresh `runner.run(...)`
rather than a `STEER` — because `STEER` always preserves completed
work.

**Signals this might be wrong.** If users routinely want "STEER that
also drops completed X", we'd add a `drop_completed_task_ids` payload
to the STEER message rather than change the default.

**Related.** [VOCABULARY.md §2](VOCABULARY.md#2-controlkind-vs-driftkind--a-worked-example),
[PROTOCOLS.md §Planner](PROTOCOLS.md#planner).

## Why the Steerer invokes `planner.refine` and not the Executor

**Observation.** The refine call lives in
`DefaultSteerer._handle_drift`, not in the executor's task loop. The
executor only observes the resulting `session.plan` change.

**Intent.** Refine is a *reaction to drift*, and drift is the Steerer's
domain. Keeping refine in the Steerer:

- Makes the drift pipeline uniform: every drift goes through one
  place, regardless of where it was synthesized (executor-detected
  failure, reporting-tool call, external STEER).
- Lets the Steerer be the single authority on "what drift means" —
  severity threshold, throttling, pre-refine emission of
  `DriftDetected`.
- Keeps executors simple: they check `session.plan` at each iteration,
  which is idempotent and doesn't care how the plan changed.

**Alternatives considered.**

1. Executor calls refine. Rejected: every executor would re-implement
   the drift handling pipeline, and custom steerer policies (throttle,
   skip, upgrade severity) would not have one place to live.
2. Planner self-refines on any drift (no pipeline). Rejected: the
   gating on severity and the `DriftDetected` emission have to live
   somewhere; the Steerer is the natural home.

**Tradeoffs.** The Steerer's `_handle_drift` is a bit magical — it
does four things (emit, gate, refine, apply). A reader has to follow
that thread. The alternative is to spread those four things across
three classes, which is worse for local reasoning.

**Signals this might be wrong.** If we grow per-executor refine
policies (e.g. parallel-specific logic that can't fit in a Steerer),
we may need to split the responsibility. So far the Steerer has been
enough.

**Related.** `goldfive/steerer.py::DefaultSteerer._handle_drift`,
[STATE-MACHINE.md](STATE-MACHINE.md).

## Why reasoning-based drift detection is its own channel

**Decision.** `Steerer.observe_reasoning(text, session)` is separate
from `observe(event, session)`, and the reasoning detectors live in
`goldfive/drift/reasoning.py` rather than being merged into the
event-shape classifiers in `goldfive/drift/__init__.py`.

**Why.** Tool-level drift detection (what `observe` handles) is a
half-loop too late. By the time an adapter surfaces a `TOOL_ERROR`
or a `LOOPING_TOOL_CALL` the model has already burned a full
inference round and spent a tool call. Reasoning content
(`reasoning_content` on OpenAI-compat surfaces, `thinking` blocks on
Anthropic, `thought` parts on Google) exposes the model's
chain-of-thought *before* the tool calls resolve — catching a loop
there costs half as many tokens per intervention. The four kinds
(`LOOPING_REASONING`, `CONFUSION`, `OFF_TOPIC`,
`INTENT_DIVERGENCE`) map one-for-one onto the most common reasoning
pathologies we see in practice.

The separation also keeps the existing classifier contract
(framework-neutral `event` shapes; no session state required) clean.
Reasoning detectors need `session.reasoning_history` to detect loops
and `session.current_task_id` / `session.goals` for off-topic and
intent-divergence checks. Shoehorning those into the generic
classifier signature would force every classifier to take a
`Session` argument it does not use.

**Cost envelope.** Pattern + hash detectors are O(1) per call, so
the default path pays nothing. The optional embedding path
(`goldfive[embedding]`) loads `all-MiniLM-L6-v2` once per process
(~23 MB, ~200 ms warm-up) and runs a single encode per observation
(~5 ms). The pipeline short-circuits on the first hit and caps at
one drift per call, so cost does not scale with reasoning-block
size.

**Tradeoffs.** Pattern-based `CONFUSION` detection will false-fire
on models that are merely verbose about their process (GPT-4o's
"Let me think about this again…" pattern is benign in many
contexts). We mitigate by leaving it at `INFO` severity so it does
not trigger refine by default — callers that want action can
subclass `DefaultSteerer` and override `should_refine`.
`INTENT_DIVERGENCE` uses *graduated* severity
(`INFO` / `WARNING` / `CRITICAL`) based on cosine similarity to the
goals + task topic. That keeps the kind stable (callers that filter
by kind see one signal) while letting refine policy hinge on
`severity` — a 0.5 similarity is worth surfacing but not worth
aborting over, whereas a 0.1 similarity plus an unrelated-keyword
hit is. The pattern fallback still requires an explicit "let's
change goals" phrasing *and* a token-overlap check against
`session.goals`, so false-positive run aborts remain rare without
embeddings.

**Signals this might be wrong.** If users report
`INTENT_DIVERGENCE` false positives at `CRITICAL` we should raise
the `INTENT_DIVERGENCE_WARNING_SIMILARITY` floor or tighten the
marker regex. If `LOOPING_REASONING` fires on models whose reasoning
is legitimately iterative (e.g. chain-of-thought enumerating
hypotheses), we should increase the hash-window or add a
"distinct-tokens" guard before firing.

**Related.** `goldfive/drift/reasoning.py`,
[DRIFT.md](DRIFT.md#reasoning-category-the-models-chain-of-thought-exposes-drift-before-the-tool-calls-do).
