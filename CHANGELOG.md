# Changelog

All notable changes to goldfive are documented in this file. Dates are ISO-8601.

## Unreleased

### Added

- #141 Overlay execution model — `goldfive.wrap(...)` no longer drives
  per-task `"Task: X"` messages through the wrapped agent tree.
  Instead, one `adapter.invoke_passthrough(user_input)` sends the
  caller's original request verbatim and the new
  `goldfive.reconciler.PlanReconciler` maps observed `before_agent` /
  `after_agent` / `DelegationObserved` callbacks to plan-task
  transitions. A post-invocation follow-up loop fires
  `adapter.invoke_follow_up(task)` (gentle "Also, please: ..."
  phrasing) for PENDING tasks the tree missed; tasks still PENDING
  after the follow-up rounds land in the new terminal
  `TaskStatus.NOT_NEEDED` state. Off-plan agents emit a
  `PLAN_DIVERGENCE` drift at INFO severity (the #142 intervention
  ladder decides escalation). `ADKAdapter` gains `invoke_passthrough`
  and `invoke_follow_up`; the legacy `invoke(task)` is preserved
  for back-compat and now also uses the gentle phrasing.
  `SequentialExecutor(overlay_mode=True)` opts a caller into the new
  path; `goldfive.wrap(...)` flips it on by default.
- `REASONING_CLUSTER_TIGHTENING` — graduated INFO-severity early-warning
  drift below the `LOOPING_REASONING` cliff. Fires once per task when
  max cosine similarity against the last N=5 reasoning blocks falls in
  `[0.75, 0.9)`, giving sinks a "may be looping soon" signal before the
  WARNING-tier cliff at `>= 0.9`. Embedding-only; silent when
  `sentence-transformers` is unavailable. No refine side-effects.
- #96 Reasoning-based drift detection — `Steerer.observe_reasoning()`
  + `AgentAdapter.emit_reasoning()` feed chain-of-thought from the
  model into the drift pipeline. Four new `DriftKind` values
  (`LOOPING_REASONING`, `CONFUSION`, `OFF_TOPIC`,
  `INTENT_DIVERGENCE`) catch loops and off-goal drift before the
  tool calls resolve. Pattern + hash detectors ship by default;
  install `goldfive[embedding]` to light up cosine-similarity
  detectors. `Session.reasoning_history` keeps the last 20 blocks.
  ADK `after_model_callback` extracts per-provider reasoning
  content (OpenAI `reasoning_content`, Anthropic `thinking` blocks,
  Google thought parts).
- #57 `goldfive.quickstart(agent, goals)` — one-call `Runner` factory
  wiring `SequentialExecutor`, `PassthroughGoalDeriver`, a
  one-task-per-goal `StaticPlanner`, and an `InMemorySink`.
- #58 `examples/harmonograf_observed/` — minimal `CallableAdapter`
  agent wired to `InMemorySink` + `HarmonografSink`.
- #59 `bench/run_100_tasks.py` orchestration-only benchmark and
  `docs/performance.md` baseline (v0.1 snapshot).
- #60 `docs/guides/choosing-a-sink.md` — decision matrix across all
  shipped sinks.
- #61 `examples/multi_sink_fanout.py`, `examples/drift_refinement.py`,
  `examples/parallel_dag.py`.
- #62 `docs/guides/troubleshooting.md`.
- #63 README pointer to the 10-minute observability walkthrough.
- #64 `docs/guides/observability-with-harmonograf.md`.
- #67 `goldfive.wrap(agent)` / `goldfive.run(agent, input)` — one-line
  wrapping with auto-detected `AgentAdapter` (callable, ADK, Claude
  SDK), LLM auto-detection from ADK agent trees, and `LLMPlanner` /
  `LLMGoalDeriver` by default. `goldfive.adapters.auto.auto_adapter`
  exposes the dispatch standalone.
- #68 `.agents/` — agent-facing skill folder.
- #98 Task-lifecycle soundness: adapter invoke-loop breaks on
  terminal task status, reporting-tool dispatch rejects calls on
  terminal tasks with a structured `task_already_terminal` ack, the
  per-task reporting-tool volume cap got the first full pass
  (50 calls, same `(task, tool)` bucket), and refine failures now
  surface as a CRITICAL follow-up `DriftDetected` so sinks do not
  lose a silently-swallowed refine. Closes the filler-loop class.
- #99 Per-`(drift_kind, task)` refine-failure counter
  (`session.refine_failure_counts`) + back-off. Two consecutive
  refine failures for the same key now mark the task FAILED and
  emit a CRITICAL `REPEATED_FAILURE` drift rather than looping
  until `max_task_invocations` trips. Threshold is tunable via the
  class attribute `DefaultSteerer.REFINE_FAILURE_THRESHOLD`.
- #100 `Plan.validate(for_revision=..., prior=...)` runs at plan
  creation (`LLMPlanner.generate`) and revision
  (`LLMPlanner.refine`, `DefaultSteerer._apply_revision`). Enforces
  unique task ids, edge referential integrity, acyclicity, PENDING
  status on creation, and (post-#105) terminal-task and
  terminal→terminal-edge preservation on revision. Malformed plans
  are rejected before being installed on the session.
- #101 ADK session conversation history is healed on mid-invocation
  cancel. The adapter tracks pending `function_call_id`s inside an
  invoke and synthesises `{"cancelled": true}` function responses
  for any tool-call that never got a matching response, so the
  next turn does not see an orphaned `function_call` that upsets
  the model.
- #102 `SequentialExecutor(max_retries_per_task_lineage=3)` per-task
  retry-lineage cap. A "lineage root" strips `retry_` / `retryN_` /
  `retry_retry_` prefixes so `t0`, `retry_t0`, and
  `retry2_retry_t0` all share the same budget. When a lineage
  reaches the cap the next task in that lineage is marked FAILED in
  place without invoking the adapter. Bounds blast-radius of a
  runaway refine-spawning-retry-tasks loop.
- #103 Cancellation cascade: any `mark_task_cancelled` (via control
  CANCEL, reporting tool, or `_apply_steer`) BFS-walks
  `plan.edges` forward from the cancelled task and cancels every
  reachable non-terminal PENDING / RUNNING / BLOCKED task with
  `reason="cascade from <task_id>"`. Closes the orphaned-PENDING
  soundness gap. See `PLAN-LIFECYCLE.md` §6.3.
- #104 `Goal.success_predicate: Callable[[Session], bool] | None`
  is now evaluated at run termination by
  `goldfive.results.evaluate_goal_predicates`. A predicate that
  returns False or raises produces `Outcome.success=False` with
  `reason="goal '<summary>' unmet"` (or `"goal '<summary>'
  predicate raised: <exc>"`). Predicates evaluate in
  `session.goals` order; a raise is logged WARNING and treated as
  unmet. See `PLAN-LIFECYCLE.md` §6.1.
- #105 `Plan.validate(for_revision=True, prior=...)` enforces
  terminal-task and terminal→terminal-edge preservation across
  revisions (PLAN-LIFECYCLE §3.1 + §3.2). A revision that drops a
  terminal task, regresses one to PENDING, or deletes a frozen
  terminal edge now raises `ValueError` before the plan is
  installed.
- #106 `PlanRevised` now carries a `PlanRevisionDiff` sidecar
  (`added_task_ids`, `removed_task_ids`, `modified_task_ids`,
  `added_edges`, `removed_edges`) so sinks can render "what
  changed" without re-fetching the prior plan. Populated by
  `DefaultSteerer._emit_plan_revised`.
- #107 `DefaultSteerer.mark_task_failed(recoverable=False)` and the
  cancellation cascade now fan out through **one** shared
  primitive — `Steerer.cascade_cancel_downstream(session,
  task_id)` on `goldfive/protocols.py`. Sinks see the same
  `TaskCancelled` event stream for downstream cancellations
  regardless of whether the cascade was seeded by a fatal FAILED
  or a CANCEL.
- #109 Tool-loop guard hardening. Schema layer rejects calls with
  missing / unknown `task_id` (so malformed-spam can't poison
  session counters); per-task loop guard flips the `(task, tool)`
  bucket into a hard-reject once a burst is detected so continued
  spam gets `loop_detected` instead of silent pass-through; new
  session-wide volume cap (> 50 calls of the same tool name across
  ALL tasks) catches adversarial callers that invent a fresh
  `task_id` every call.
- #110 `examples/adk_presentation/` (new canonical path); the
  pre-existing `examples/adk_agent.py` keeps running unchanged.
- #112 Opt-in **reflective self-progress check**. When operators
  enable it via
  `DefaultSteerer(reflective_check_interval=15, reflective_call_llm=...)`,
  every N LLM turns the steerer asks the model "are you making
  forward progress on task X?" and classifies the reply as either
  `UNCERTAIN_PROGRESS` (yes, but `confidence < 0.5`, INFO) or
  `SELF_REPORTED_STUCK` (no, WARNING → refine). Off by default:
  operators that don't supply `reflective_call_llm` pay no LLM
  cost. A parse / LLM failure emits an INFO `CUSTOM` drift
  prefixed `reflective_check_failed:` — the run is never broken
  by a bad reflective call.
- Extended `Runner` extension API (#87) with the post-construction
  `control` setter for late `ControlChannel` attachment.
- Shared `Steerer.cascade_cancel_downstream` primitive added to
  `goldfive/protocols.py`; reference impl on `DefaultSteerer`.

### Changed

- #108 **Critical fix.** Every adapter's reporting-tool dispatch is
  now routed through the central
  `goldfive.adapters._tool_invocation.invoke_tool` helper, which
  runs the four-layer guard stack (schema rejection, terminal-task
  rejection, per-task loop guard, session-wide volume cap) before
  calling `spec.handler`. Pre-#108 the ADK plugin called
  `spec.handler` directly from `before_tool_callback`, bypassing
  the idempotency and terminal-rejection layers — which is the
  root cause of the "guard defined but not firing" class of
  filler-loop bugs. No public API change; behaviour only.
- #111 `examples/adk_presentation/agent.py` simplified to a single
  `Agent` + `goldfive.wrap(...)` call. The previous hand-rolled
  coordinator+subagents tree was agent-design, not framework, and
  was hiding a root cause of the week's filler-loop investigation
  (`require_confirmation` combined with a bad coordinator prompt
  kept the agent silently gated instead of executing the task).
  The multi-agent reference moved to
  `harmonograf/tests/reference_agents/presentation_agent/`.
- #114 `INTENT_DIVERGENCE` now fires at **graduated severity**
  (`INFO` / `WARNING` / `CRITICAL`) based on cosine similarity
  between the reasoning block and `session.goals` + the current
  task topic. Bands: `sim >= 0.6` healthy, `>= 0.4` INFO, `>= 0.2`
  WARNING, `< 0.2` CRITICAL. An unreferenced-keyword mismatch bumps
  severity one step. The pattern-based fallback (no embeddings)
  now fires at `WARNING` with the same keyword-mismatch bump. The
  drift kind is stable; only `severity` changes. See
  `docs/design/DRIFT.md` for the full table and
  `goldfive/drift/reasoning.py` for thresholds.
- #115 Renamed `max_plan_reinvocations` to `max_task_invocations` on
  `SequentialExecutor`, `ParallelDAGExecutor`, `Runner`, and
  `goldfive.wrap()` / `goldfive.run()`. The default is now
  `None` (unbounded) — per-task / per-tool caps
  (`max_retries_per_task_lineage`, adapter-level loop guards) are
  the primary guards against runaway invocations. The old kwarg is
  still accepted for one release and emits a
  `DeprecationWarning` that maps to the new parameter. Callers that
  relied on the old 32 default should pass `max_task_invocations=32`
  explicitly, or an appropriate ceiling for their workload.
- #65 Follow-up cleanup from the Team Lead A drift audit
  (`SequentialExecutor.max_plan_reinvocations` default raised
  from 3 to 32; minor doc fixes). Superseded by the rename above.
- #69 Reconciled `HarmonografSink` API docs with the shipped
  `harmonograf_client.HarmonografSink(client)` shape and canonical
  `:7531` port.

### Fixed

- #98 Filler-loop ceiling regressions: adapters that ignored the
  terminal-task status post-`report_task_completed` could keep
  invoking and hit the ADK 500-call ceiling. The adapter invoke
  loop now breaks early on terminal status, and the steerer rejects
  terminal-task reporting calls at the dispatch boundary.
- #108 See **Changed**. Reporting-tool dispatch that skipped the
  guard layers was the class of bug that caused every filler-loop
  patch before this to look correct in isolation but ineffective
  in practice.
- #109 `report_task_completed(task_id="")` and other empty / unknown
  `task_id` payloads no longer reach the guard state counters.

## 0.1.0 — 2026-04-18

Initial public release. goldfive v0.1 is a framework-agnostic control loop
that wraps an agent with an explicit goal, a DAG-structured plan, per-turn
drift analysis, and a steering loop — wired to a protobuf event stream you
can persist, replay, or ship to an observability console. Adapters ship for
Google ADK, the Claude Agent SDK, and plain async callables. Planners,
executors, goal derivers, steerers, and event sinks are all pluggable
behind narrow protocols.

### Added

- #18 Scaffold goldfive package
- #19 Define core protocols, reporting tool spec, and result dataclasses
- #20 Add CallableAdapter (reference AgentAdapter)
- #21 Add core dataclasses, proto converters, and event helpers
- #22 Add GoalDeriver variants (Passthrough, Literal, LLM)
- #23 Add Parallel-DAG executor
- #24 Add built-in EventSinks (InMemory, Logging, JSONL persistence)
- #25 Implement Sequential executor
- #26 Implement Planner (Passthrough + LLM) [closes #7]
- #27 feat: Claude Agent SDK adapter (#14)
- #29 Add ADK AgentAdapter and plugin
- #30 Port _AdkState state machine into DefaultSteerer
- #33 Add typed event factories to goldfive.events
- #34 Wire up public API re-exports and fix event helpers
- #36 Define goldfive proto schema + wire up Python codegen
- #47 Add SQLitePersistenceSink
- #48 Add ADK presentation_agent reference example
- #52 Add goldfive gRPC transport (sink + server)

### Changed

- #28 Add umbrella test-suite for issue #16
- #31 Add design docs, user guides, and reference documentation
- #46 Verify ADK root-agent wrapping propagates across sub-agent tree
- #49 Verify design-doc code snippets against goldfive v0.1
- #50 Verify code snippets in docs/guides/ run against v0.1
- #51 Align docs/reference/ + README hello snippet with goldfive v0.1 API
- #54 Consistency audit: fix API drift

### Fixed

- #37 Executors auto-complete tasks when agent returns without reporting
- #38 JSONLPersistenceSink accepts non-proto events
- #55 Runner emits proto events consistently (fixes #53)
