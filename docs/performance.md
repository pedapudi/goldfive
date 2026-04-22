# Performance baseline

This page records goldfive's **orchestration-only** performance baseline:
how fast and how memory-efficient the runner / executor / sink path is
when the agent itself does no real work. It is documentation, not a CI
gate — the numbers below give us a yardstick to notice future
regressions when we look.

## Baseline workload

The benchmark in `bench/run_100_tasks.py` runs:

- A **100-task linear plan** (`t000 → t001 → … → t099`) built directly
  and handed to `StaticPlanner` so no LLM planning round-trip is in the
  measurement.
- A **trivial no-op `CallableAdapter` agent** that returns
  `InvocationResult(task_id=task.id, text="ok")` immediately. There is
  no LLM call, no network I/O, and no compute beyond constructing the
  result.
- The **`SequentialExecutor`** (single-threaded; one task at a time)
  with `max_task_invocations=101` so the budget cannot trip on the
  100-task plan.
- A single **`JSONLPersistenceSink`** writing proto-encoded events to a
  temporary file. JSONL is the highest-fidelity sink we ship and is the
  most realistic single-sink choice for a production deployment that
  wants crash recovery.

The point is to isolate **orchestration overhead** — runner setup,
sequential walking of the DAG, steerer transitions, event construction
and serialisation, file writes — from anything that depends on a real
agent.

## How to run

```sh
uv run --extra proto --extra dev python bench/run_100_tasks.py
```

The `proto` extra is required because `JSONLPersistenceSink` serialises
proto event messages. The script prints a single block of measurements
to stdout and unlinks its temp file before exiting. Total runtime is
well under a second on commodity hardware.

## Recorded baselines

### 2026-04-21 (current)

Median of five consecutive runs, commodity Linux laptop, single
core, no parallelism, CPython 3.12.13, post-overlay + tool-loop
detector.

| Metric             | Value          |
| ------------------ | -------------- |
| Wall time          | **0.104 s**    |
| Throughput         | **962 tasks/s** |
| Peak memory        | **0.33 MiB**   |
| JSONL file size    | **61.59 KiB**  |
| Python             | 3.12.13        |
| goldfive           | 0.1.0          |

JSONL file size grew ~20 % vs the 2026-04-18 baseline because the
proto `Event` envelope gained the `session_id` field (goldfive#155)
and `DriftDetected` gained `annotation_id` (goldfive#177). These
are per-event string fields stamped by the Runner / Steerer, not
per-turn overhead. Peak memory and wall time held flat relative to
baseline — the overlay execution model (goldfive#141) removes the
per-task re-invocation overhead, offsetting the new per-call
instrumentation (`goldfive.llm.request` / `goldfive.llm.response`
logs, goldfive#172/#174).

### 2026-04-18 (v0.1.0 release snapshot)

Median of five consecutive runs on the same hardware:

| Metric             | Value          |
| ------------------ | -------------- |
| Wall time          | **0.107 s**    |
| Throughput         | **937 tasks/s** |
| Peak memory        | **0.29 MiB**   |
| JSONL file size    | **51.28 KiB**  |
| Python             | 3.12.13        |
| goldfive           | 0.1.0          |

Run-to-run variance over the five samples is roughly ±10 % on
wall-time and effectively zero on peak memory and JSONL size.

## Methodology notes

- **Wall-clock** is `time.perf_counter()` deltas around `Runner.run`
  (after sink + runner construction; that constructor work is excluded
  to keep the measurement focused on the run path).
- **Peak memory** is `tracemalloc.get_traced_memory()[1]` — the high-
  water mark of Python-allocated memory between `tracemalloc.start()`
  and the snapshot taken immediately after `Runner.run` returns. It
  does not include C-extension allocations (e.g. inside `protobuf`),
  but it captures the dataclass and event churn that goldfive itself
  is responsible for.
- **JSONL file size** is `Path.stat().st_size` of the persistence file
  immediately before unlink. Each event is one proto-canonical JSON
  line with sorted keys.
- The agent is a no-op, so this measures **orchestration overhead
  only**. Real workloads will be dominated by LLM and tool latency,
  which goldfive does not control.

## Shape of the performance profile

The 2026-04-18 baseline measures a **legacy per-task** executor path
(`SequentialExecutor(overlay_mode=False)` + `StaticPlanner`, no LLM
in the loop). Real `goldfive.wrap`-driven runs have a different
profile:

- **LLM call count** scales with `tree_depth × task_count` in the
  per-task model; under the overlay model (goldfive#141, default for
  `goldfive.wrap`) it scales with whatever the wrapped tree does
  naturally for the single passthrough invocation. Goldfive adds
  one additional LLM call per refine, bounded by
  `LLMPlanner.DEFAULT_MAX_REFINE_ATTEMPTS=2`.
- **Tool-loop detector** (`ToolLoopTracker`) is **O(1) per tool call**
  modulo the ring-buffer window (default 7). Pure Python dict /
  deque ops; no embeddings, no LLM.
- **Reasoning-similarity detectors** are O(1) when embeddings are
  unavailable (hash-only path). With `goldfive[embedding]`
  installed, each reasoning block costs one embedding lookup against
  the last N=5 blocks in `session.reasoning_history`.
- **Per-LLM-call instrumentation** (goldfive#172/#174) adds a
  constant-time log line before and after each model call. No
  measurable overhead at this scale.
- **GoldfivePlanner** runs request-side and response-side on every
  LLM turn; both are O(parts_in_response) for the three-stage
  classifier. No allocations per call in the common case.
- **Session state bridge** (goldfive#170/#173) writes ~5 goldfive
  keys onto ADK `session.state` at `before_run_callback`; one dict
  copy per invocation, not per turn.

## Known limitations

- **Single-threaded**: `SequentialExecutor` walks one task at a time.
  The `ParallelDAGExecutor` will have a different profile and should
  get its own baseline once it stabilises.
- **No LLM, no network**: the headline numbers say nothing about how
  goldfive performs against real models. They are a floor on
  orchestration overhead, not a ceiling on end-to-end run time.
- **Single sink**: real deployments often fan events out to multiple
  sinks (logging + persistence + gRPC). Each additional sink adds its
  own serialisation cost.
- **Linear plan**: a 100-stage chain stresses sequential walking but
  not the topological-sort path that wide DAGs exercise.
- **Per-task executor**: the benchmark still runs the legacy per-task
  loop because the overlay path needs an adapter that implements
  `invoke_passthrough`. An overlay-mode baseline should be added
  once the benchmark is migrated.
- **CPython only**: numbers measured under CPython 3.12; PyPy and
  newer CPython releases may differ.

## Regression policy

If a future change pushes wall-time past **2× current baseline**
(~0.21 s) or peak memory past **2× current baseline** (~0.66 MiB) on
this exact workload, that warrants investigation before merge. We do
**not** gate CI on the benchmark — it is a tripwire for human
reviewers, not infrastructure.

To check after a change, simply re-run the benchmark a handful of times
and compare the medians against the tables above. If the workload
itself changes (different sink, different plan shape, different
adapter, overlay-mode path), record a new baseline rather than
comparing apples to oranges.
