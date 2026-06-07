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
