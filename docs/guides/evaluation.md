# Grading agents: what to read, what not to read

If you score agents with goldfive — expectation predicates, rubric
judges, an exact-match check on returned ids, or a fitness function in
an optimization loop — **grade the agent's actual output, not its
self-report.** This guide names the gradeable artifact and the two
traps that historically made grades imprecise and non-deterministic.

Related: [Multi-turn conversations](multi-turn.md),
[Writing an agent adapter](writing-an-agent-adapter.md),
[api.md](../reference/api.md).

## TL;DR

| You want… | Read | Not |
|---|---|---|
| The agent's complete output for a task | `session.completed_outputs[task_id]` | `session.completed_results[task_id]` |
| The complete output of one invocation | `InvocationResult.full_text` (or `.text_turns`) | `InvocationResult.text` |
| Structured returned values (ids, citations, files) | `report_task_completed(..., artifacts={...})` event payload | substring-matching prose |

`completed_results` and `InvocationResult.text` are still populated and
still mean what they always did — they are kept for backward
compatibility. They are **summaries / the last turn**, not the full
output. Don't grade them.

## The two traps

### 1. The self-report is not the output

When an agent ends a task with `report_task_completed(summary=…)`, that
`summary` is an agent-authored **status line** ("Found the KEYWORD
table."), written to signal completion — not to faithfully reproduce
the answer. It lands in `session.completed_results[task_id]`.

Grading that summary means grading *how the agent narrated what it did*.
Two runs with byte-identical behavior can get opposite verdicts purely
on the phrasing of the summary: run A's summary happens to repeat the
exact id `KEYWORD_____ID_x_y_V2` → your token check passes; run B says
"the KEYWORD table" → it fails. The grade becomes a function of summary
verbosity, uncorrelated with correctness.

**Fix:** read `session.completed_outputs[task_id]`. goldfive records the
agent's *complete actual output* there for every task, **independent of
whether the agent self-reported**. The self-reported summary stays in
`completed_results` as separate metadata; it never shadows the real
output.

### 2. Only the last turn is the answer

An agent often emits its substantive answer (a list, a table of ids, a
JSON blob) in one turn and a terse wrap-up ("Done — let me know if you
need anything else.") in the next. `InvocationResult.text` keeps only
the **last** non-empty turn, so the substantive turn is silently
dropped.

**Fix:** read `InvocationResult.full_text` (every assistant text turn,
joined by `goldfive.results.TURN_SEPARATOR`) or `InvocationResult.text_turns`
(the ordered list). The executor records `full_text` into
`session.completed_outputs`, so a session-level grader already gets the
full-fidelity artifact.

## Previews are previews, not gradeable artifacts

Process judges that read `recent_events` tool-observation
`result_preview` strings are reading **truncated** previews (≈480
chars), bounded for observability. They are fine for "is the agent on
task?" reasoning but are **not** a faithful grading target — do not run
exact-match grading against a preview. For exact matching, use the
full-fidelity channel (`completed_outputs` / `full_text`).

## Structured output (exact-match-friendly)

When a task's success is defined by a set of returned artifacts (ids,
citations, file paths, numeric values), have the agent declare them
structurally via the existing reporting tool:

```python
report_task_completed(
    task_id="...",
    summary="Looked up the three matching rows.",
    artifacts={"row_ids": "KEYWORD_____ID_x_y_V2,KEYWORD_____ID_a_b_V1"},
)
```

`artifacts` is a typed `dict[str, str]` carried on the `TaskCompleted`
event. Grade it as exact-match on structured data instead of substring
matching on prose — that removes the entire "did the prose happen to
contain the token" class of fragility. `summary` remains free-form
metadata.

## Multi-turn

Both maps carry across conversation turns on the owning
`Conversation` (`completed_results` and `completed_outputs`), with
later-turns-win merge semantics. A turn-N grader and the planner both
see prior turns' full output, not only their summaries. See
[Multi-turn conversations](multi-turn.md).

## Backward compatibility

This is purely additive. If you already grade `completed_results` /
`InvocationResult.text`, nothing changed for you — but you are grading a
lossy channel, and migrating to `completed_outputs` / `full_text` will
raise your eval's signal-to-noise ratio. Adapters that cannot
distinguish turns leave `text_turns` empty; `full_text` then falls back
to `text`, and `completed_outputs` falls back to whatever the adapter
recorded, so a grader can always read the new fields safely.

## The three-arm counterfactual bench (`bench/`)

The AGENCY-PRESERVATION rollout (`docs/design/AGENCY-PRESERVATION.md`)
flips `observation_only=False` (and `plan_mode=ledger`) back to the
default **only** when the new steering regime is measurably non-inferior
to a *disabled* goldfive. That decision is gated on one artifact: the
three-arm bench in `bench/`, plus the §5.4 shadow-diff that must show a
*reviewed* legacy-vs-new divergence report before any behavior PR is
enabled.

The library lives in `bench/harness.py` (the harness) and
`bench/shadow_diff.py` (the divergence tool); `bench/run_100_tasks.py` is
the CLI. Running the real bench against a live model is the *next* task —
this is the tooling.

### The three arms

The harness runs the **same workload** under three steering regimes and
records per-arm metrics from *captured artifacts* (`completed_outputs`,
goal predicates, the `SignalLedger`) and the *sink event stream*
(`SignalDelivered` / `SignalOutcome` / `DriftDetected`) — never from
parsing agent prose:

| Arm | Kind | What it is |
|---|---|---|
| A | `baseline` | `wrap(judge_only=True)` — the judge-only counterfactual (goldfive#446). The agent runs natively, judges stay armed, zero steering authority. This is the bar arm B must be non-inferior to. |
| B | `signal` | the new SIGNAL regime — `observation_only=False` + the new-regime flags (`signal_channel=request_context`, `plan_mode=ledger`). |
| C | `legacy` | the legacy ladder — `GOLDFIVE_STEER_LEGACY_LADDER=1` + `cancel_inflight_scope=all`, kept alive so regressions are measurable rather than argued about (§5.8). |

Arms are defined as **flag dicts of environment variables**, not
`SteeringConfig` kwargs, so the harness does not hard-depend on unmerged
PRs. A flag whose env var this build's `RuntimeConfig.from_env` does not
yet consult (e.g. `GOLDFIVE_STEER_SIGNAL_CHANNEL` before PR 6) is applied
to the environment, reported as **pending**, and simply no-ops until the
PR that reads it lands — graceful degradation by construction.
`arm_flag_status(arm)` returns `(applied, pending)` so a pending flag is
never silently mistaken for an applied one.

```bash
uv run python bench/run_100_tasks.py three-arm --out-dir /tmp/bench
```

Per-arm metrics (`ArmMetrics`): goal-predicate success, turns/tokens to
completion, `intervention_count` (real, non-dry-run deliveries),
`post_signal_refire_rate` (from the `SignalLedger`), run-abort, and the
PR-5 **self-correction base rate** (`self_corrected_unaided` vs.
`self_corrected_after_signal` from `SignalOutcome`).

### `signal_telemetry` is DEFAULT OFF

`SteeringConfig.signal_telemetry` ships off (goldfive#456): a run that
does not set it emits **no** `SignalDelivered` events. Every arm here
enables it explicitly, and the tooling treats *zero parsed signal events*
as a **loud error**, never an empty report — `load_signals()` raises
`ShadowDiffError` on a zero-delivery log so "the flag was off" can never
masquerade as "no divergence". Pass `--allow-empty` only for a run known
to be genuinely drift-free.

### The §5.4 shadow diff

Shadow mode runs arms B/C with `observation_only=1` so the new decision
logic runs *dry* — `SignalDelivered(dry_run=true)` with a full decision
payload — and accrues production mileage with zero production authority.
The tool then diffs legacy-would-do vs. new-would-do **on the same
runs**, aligning deliveries by a stable cross-run key
`(kind, task_id, occurrence#)` (drift ids are per-run minted, so they are
never the join key — the project's stable-keys discipline):

```bash
uv run python bench/run_100_tasks.py shadow --out-dir /tmp/bench
# or diff two existing logs directly:
uv run python bench/shadow_diff.py --legacy C.jsonl --new B.jsonl
# single-log census of one shadow run:
uv run python bench/shadow_diff.py B.jsonl
```

Even before PR 7's ladder restructure lands, the diff surfaces a real,
merged divergence: a CRITICAL goldfive-authored `OFF_TOPIC` drift cancels
in-flight work under `cancel_inflight_scope=all` (legacy) but not under
`user_and_safety` (new) — the PR-1 authority split — so the report reads
`would_cancel_inflight: legacy=True -> new=False`. Once the behavior PRs
merge, `ladder_level` and channel divergences appear in the same report.
The reviewed two-log report is the **exit criterion** for enabling any
behavior PR (§5.4, §5.8).
