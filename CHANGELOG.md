# Changelog

All notable changes to goldfive are documented in this file. Dates are ISO-8601.

## Unreleased

### Added

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

### Changed

- #65 Follow-up cleanup from the Team Lead A drift audit
  (`SequentialExecutor.max_plan_reinvocations` default raised
  from 3 to 32; minor doc fixes).
- #69 Reconciled `HarmonografSink` API docs with the shipped
  `harmonograf_client.HarmonografSink(client)` shape and canonical
  `:7531` port.

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
