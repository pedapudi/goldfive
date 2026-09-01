# 08. LLM Judges

## Read this chapter when...

You are touching any of goldfive's **LLM-as-a-judge** machinery: the
per-thinking-message reasoning-drift judge, the trajectory-level
goal-drift judge, the background scheduling that runs them off the
critical path, the single internal LLM-call module they dispatch
through, or the pluggable `judges/` package that lets operators install
custom judges. Concretely, read this before you:

- change a judge **prompt** (system or user template) — there is no
  automated regression harness for judge quality, so a prompt edit is
  higher-risk than it looks (see Common mistakes);
- add, remove, or reorder a **parsed field** in a judge verdict (e.g.
  `classification`, `provenance`, `focused_task_id`);
- touch the **three-state classification** logic (`on_task` /
  `justified_deviation` / `erroneous_deviation`) or the severity mapping;
- add a new judge **call site** (you must go through the semaphore + the
  `call_llm_budget` / `call_llm_thinking_disabled` / `llm_call_diagnostics`
  context managers — never a bare `await call_llm(...)`);
- change the **token budget**, the **thinking-disable** wiring, or the
  vendor capability table in `goldfive/_llm.py`;
- write a **custom judge** or change how `JudgeContext` / `JudgeVerdict`
  are populated;
- debug a judge that "never fires", "fires too much", or "returns
  `raw=''`".

### Files covered

| File | What lives there |
|------|------------------|
| `goldfive/drift/reasoning_judge.py` | The per-reasoning-block judge: prompt templates, truncation caps, three-state parse, attribution parse, severity parse, `ReasoningJudgeVerdict`, the `ReasoningJudgeInvoked` emitter. |
| `goldfive/drift/goals.py` | The trajectory-level GOAL_DRIFT judge: prompt, activity rendering, the `idle_note` entry, the post-LLM plan re-read. |
| `goldfive/_llm.py` | The **one** internal LLM-call module (#491): the two default builders, the `max_output_tokens` / thinking-disable / diagnostics ContextVars, the `THINKING_DISABLE_CAPABILITIES` vendor table. |
| `goldfive/drift_observer.py` (judge-scheduling parts) | The per-steerer semaphore, the coalescing registry, the verdict-utility ledger, the freshness/staleness gates, the goal-drift scheduling paths. |
| `goldfive/drift/reasoning.py` (judge dispatch parts) | `analyze_reasoning_with_focus`, `_run_judge_with_focus` — the mode selector that unpacks the session into the classifier. |
| `goldfive/judges/base.py` | `Judge` protocol, `JudgeContext`, `JudgeVerdict`. |
| `goldfive/judges/builtins.py` | Built-in judge shims, `BuiltinJudge` enum, `default_judges()`, `BUILTIN_JUDGE_NAMES` skip-list. |
| `goldfive/config.py` (judge parts) | `JudgeConfig`, `ReasoningDriftConfig`, `GoalDriftConfig`. |
| `goldfive/convenience.py` (judge parts) | The endpoint-contention WARNING at `goldfive.wrap` time. |
| `goldfive/adapters/_adk_plugin.py` (disarm parts) | The one-shot reasoning-channel-disarm WARNING (#476/#263) and the `LLM_CALL_TIMEOUT` watcher. |
| `proto/goldfive/v1/events.proto` | `ReasoningJudgeInvoked` message (fields 1-15). |

Sibling chapters you will cross into:
`07-deterministic-drift-detection.md` (the non-LLM detectors that run
inline before the judge), `09-steering-ladder-and-gates.md`
(`handle_drift`, promotion, the ladder that consumes judge verdicts),
`10-planning-and-revision.md` (`refine` / `refine_steer`, plan revision
that the freshness gate keys against), `11-state-ownership.md`
(`StateStore`, reasoning-extracted bindings), `12-events-sinks-telemetry.md`
(`ReasoningJudgeInvoked`, `reasoning_judge_utility_summary`, LLM spans),
`14-config-reference.md` (env vars), `15-testing-guide.md` (the harness).

### Invariants that bind you here

These are non-negotiable in this subsystem. Every one of them has a
history of being violated and reverted.

1. **No prompt-cooperation contract.** The judge observes the agent's
   thinking tokens; it never requires the agent to call a goldfive tool
   or follow an instruction. A judge that only worked when the agent
   "cooperated" would violate HARD INVARIANT 1. The judge is a passive
   reader of reasoning text.
2. **No regex/keyword NL classification.** The judge decides on-task vs
   deviation with an LLM, not a keyword list. The retired
   `_GENERIC_VERB_PREFIX_RE` (#166) and `_FACTUAL_QUESTION_RE` (#167)
   are gone; do not reintroduce anything like them here. Exact-equality
   / frozenset membership on **structured** enum strings
   (`_VALID_CLASSIFICATIONS`, `_VALID_PROVENANCES`, `_SEVERITY_MAP`) IS
   allowed — those match the judge's own structured output, not free
   natural language.
3. **Quiet on failure.** A judge that raises, returns non-JSON, returns
   JSON missing the decision key, or spends its whole token budget
   thinking (empty answer) MUST produce **no drift**. Only an explicit
   negative verdict (`classification` in `justified_deviation` /
   `erroneous_deviation`, or `progressing: false`) produces a
   `DriftEvent`. A flaky judge must never spam operator UIs with
   false-positive alarms.
4. **Observation stays passive.** The judge **detection** path runs in
   full even under `observation_only=True` (the production default). The
   judge computing a verdict is not an intervention; only `handle_drift`
   downstream write-paths (cancel, refine install, nudge enqueue) are
   gated. Do not add a kill-switch read inside the judge — the sanctioned
   read is `DefaultSteerer.is_active_steering()` /
   `steering_is_active(steerer)` and it lives on the dispatch side, not in
   the classifier. See `09-steering-ladder-and-gates.md`.
5. **Cost-bounded, single call.** Each classifier function awaits
   `call_llm` **at most once** per invocation. The caller owns the
   rate-limit / scheduling policy. No retries inside the classifier.
6. **Adaptive, not predictive.** The judge records observed facts
   (`focused_task_id`, `stated_intent`) onto the store/wire; it does not
   predict what the agent will do. Bindings are keyed by stable agent
   name, never by an LLM-minted id (INVARIANT 6).
7. **All judge dispatch goes through `goldfive/_llm.py` context
   managers.** `call_llm_budget(...)`, `call_llm_thinking_disabled()`,
   and `llm_call_diagnostics()` wrap the `await call_llm(...)`. Diagnostics
   are read from the yielded object, never via `getattr` on the callable
   (that pattern was deleted in #491).

---

## 1. The two judges at a glance

goldfive ships exactly two LLM-as-a-judge detectors. They are siblings
with the same "quiet on failure" contract but ask different questions at
different cadences.

| | Reasoning-drift judge | Goal-drift judge |
|---|---|---|
| File | `goldfive/drift/reasoning_judge.py` | `goldfive/drift/goals.py` |
| Question | "Is *this* chain-of-thought block on task?" | "Is the whole trajectory progressing toward the goals?" |
| Cadence | Per thinking message, rate-limited per `(agent, task)` bucket | Every N agent turns, on task-boundary transitions, and (flag-gated) on idle |
| Input | One reasoning block + task + goals + plan + lineage + recent tool observations + agent tree | Goals + plan tasks + recent agent activity |
| Entry point | `classify_reasoning_drift_with_focus(...)` (and the back-compat `classify_reasoning_drift(...)`) | `classify_goal_drift(...)` |
| Verdict type | `ReasoningJudgeVerdict` (drift + attribution) | `DriftEvent \| None` |
| Drift kinds emitted | `OFF_TOPIC` (erroneous_deviation), `JUSTIFIED_DEVIATION` | `GOAL_DRIFT` |
| Default severity of a fire | judge-chosen `info`/`warning`/`critical` (malformed → INFO) | always `CRITICAL` |
| Scheduling | fire-and-forget bg task, semaphore-gated, coalescing | fire-and-forget bg task (no semaphore) |
| History / PR | #226, #251, #271 (attribution), #244 (agent tree), iter-10 (three-state) | #143, #218, #219, #239, #245 |

Both classifiers are **framework-neutral**: they do not import from
`goldfive.steerer` or any adapter. They take the data they need via
keyword arguments and return a drift (or verdict). The steerer wires
them up; the module boundary keeps a circular import from forming and
lets tests call them directly.

Both register themselves into the drift registry at import time. The
reasoning judge registers `classify_reasoning_drift` under **two** kinds
(it can emit either), the goal judge under one:

```python
# goldfive/drift/reasoning_judge.py  (module bottom)
_register(DriftKind.OFF_TOPIC, classify_reasoning_drift, _REASONING_JUDGE_CONFIG, is_async=True)
_register(DriftKind.JUSTIFIED_DEVIATION, classify_reasoning_drift, _REASONING_JUDGE_CONFIG, is_async=True)
```

```python
# goldfive/drift/goals.py  (module bottom)
_register(DriftKind.GOAL_DRIFT, classify_goal_drift, _GOAL_DRIFT_CONFIG, is_async=True)
```

The `DetectorConfig` on each carries `uses_llm=True`, the per-callsite
`max_output_tokens`, and `disable_thinking=True` — see
`07-deterministic-drift-detection.md` for what `DetectorConfig` is and how
the dispatch-snapshot event reads it.

---

## 2. Reasoning-drift judge: the full pipeline

`classify_reasoning_drift_with_focus` in
`goldfive/drift/reasoning_judge.py` is the real implementation.
`classify_reasoning_drift` is a thin back-compat wrapper that returns
`verdict.drift` so pre-#271 callers keep their `DriftEvent | None` return
shape:

```python
# goldfive/drift/reasoning_judge.py
async def classify_reasoning_drift(...) -> DriftEvent | None:
    verdict = await classify_reasoning_drift_with_focus(...)
    return verdict.drift
```

### 2.1 Call chain into the judge

The judge is **not** called directly from adapters. The path is:

1. Adapter's model-response callback extracts reasoning content and calls
   `steerer.drift.observe_reasoning(text, session, agent_name=...)`
   (`DriftObserver.observe_reasoning` in `goldfive/drift_observer.py`).
2. `observe_reasoning` runs the cheap **inline** loop detector first
   (`detect_looping_reasoning`); a fire short-circuits before the judge.
3. It then takes a rate-limited judge slot
   (`_maybe_take_reasoning_judge_slot`), snapshots the reasoning history,
   and schedules a **background** task (`_run_judge_background`) gated on
   the per-steerer judge semaphore.
4. `_run_judge_background` → `_run_judge_window` →
   `analyze_reasoning_with_focus` (in `goldfive/drift/reasoning.py`) →
   `_run_judge_with_focus` → `classify_reasoning_drift_with_focus`.

`_run_judge_with_focus` is where the session is unpacked into the
classifier's keyword arguments (task, goals, plan, lineage, recent tool
observations). It filters `session.recent_events` down to the
`tool_observed` kind for the prompt's tool-observation block, and pulls
`session.task_lineage` for the lineage block:

```python
# goldfive/drift/reasoning.py — _run_judge_with_focus
return await classify_reasoning_drift_with_focus(
    reasoning=text,
    task=_current_task(session),
    goals=list(session.goals),
    plan=getattr(session, "plan", None),
    model=model,
    call_llm=call_llm,
    current_task_id=session.current_task_id,
    current_agent_id=agent_name,
    sink=sink,
    run_id=session.run_id,
    session_id=session.id,
    sequence_fn=session.next_sequence,
    task_lineage=getattr(session, "task_lineage", None),
    recent_tool_observations=tool_obs,
    available_agents=available_agents,
)
```

`analyze_reasoning_with_focus` picks the pipeline by
`reasoning_drift_mode` (`"judge"` — the default, from
`DEFAULT_REASONING_DRIFT_MODE = "judge"` in `goldfive/drift/reasoning.py`
— plus `"embedding"`, `"both"`, `"off"`). Only the `"judge"` / `"both"`
modes ever call the LLM judge. In `"both"` the embedding drift and the
judge verdict are merged worst-severity-wins with `dataclasses.replace`
so the judge's attribution fields survive even when the embedding drift
wins the severity tie-break.

### 2.2 Empty-reasoning guard (no LLM call)

The classifier's very first action is a cheap guard: whitespace-only
input returns an empty verdict **without** awaiting the LLM. This is the
`judge_ran=False` case (the verdict's `judge_ran` field stays `False`, so
the ledger downstream can tell "quiet-fail" from "never ran").

```python
# classify_reasoning_drift_with_focus
if not reasoning or not reasoning.strip():
    return ReasoningJudgeVerdict(drift=None)
```

### 2.3 Prompt structure

Two module-level constants pin the prompt. Operators can override them
without reimplementing the parse logic (pass `system_prompt=` /
`user_prompt_template=`), which is exactly why they are module constants
and not inline strings.

- `REASONING_DRIFT_SYSTEM_PROMPT` — one sentence: "You are assessing
  whether an autonomous agent's chain-of-thought is staying focused on
  its explicit task and goals. Reply with a single JSON object and
  nothing else."
- `REASONING_DRIFT_USER_PROMPT_TEMPLATE` — the big `.format(...)`
  template. It has these sections **in order** (each a `{placeholder}`):

  1. `PLAN TASKS (id -> title)` — `{plan_tasks_summary}` via
     `format_plan_tasks_summary(plan, available_agents=...)`.
  2. `CURRENTLY BOUND TASK` — `{task_block}` via `_format_task(task)`.
  3. `Currently reasoning agent:` — `{current_agent_id}`.
  4. `Task lineage:` — `{task_lineage_block}` via `_format_task_lineage(...)`.
  5. `GOALS` — `{goals_block}` via the shared `format_goals_block`.
  6. `RECENT TOOL OBSERVATIONS (last {tool_obs_count}, oldest first)` —
     `{tool_obs_block}` via `_format_tool_observations(...)`.
  7. `REASONING (the agent's most recent chain-of-thought block)` —
     `{reasoning_block}` via `_format_reasoning(reasoning)` (≤1500 chars).
  8. The "Decide THREE things" instruction block: **CLASSIFICATION**,
     **ATTRIBUTION**, **PROVENANCE**.
  9. The JSON shape spec.
  10. A `GUIDANCE` block with per-provenance criteria.
  11. Severity guidance for non-on_task classifications.

The template is `.format(...)`-ed in `classify_reasoning_drift_with_focus`
with exactly these keys:

```python
user = template.format(
    plan_tasks_summary=format_plan_tasks_summary(plan, available_agents=available_agents),
    goals_block=_format_goals(goals),
    task_block=_format_task(task),
    reasoning_block=_format_reasoning(reasoning),
    current_agent_id=current_agent_id or "(unknown)",
    task_lineage_block=task_lineage_block,
    tool_obs_block=tool_obs_block,
    tool_obs_count=tool_obs_count,
)
```

**If you add a section to the template you must add its key to this
`.format(...)` call, or the format raises `KeyError` and the judge
quiet-fails on every call** (the exception is caught and logged, no drift
emitted, but the judge is effectively dead). Conversely, if you add a
`{placeholder}` and a matching key, existing operator overrides of the
template that lack the placeholder keep working — Python's `str.format`
ignores unused kwargs. The reverse is not true.

The **agent tree** section (#244) is deliberately **not** a template
placeholder. When `available_agents` is non-empty, the code appends an
`AGENT TREE (...)` block to the already-formatted `user` string and
appends `AGENT_TREE_SYSTEM_PROMPT_SUFFIX` to the system prompt:

```python
agent_tree_block = format_available_agents_block(available_agents)
if agent_tree_block:
    user = f"{user}\n\nAGENT TREE (...):\n{agent_tree_block}"
    system = f"{system}{AGENT_TREE_SYSTEM_PROMPT_SUFFIX}"
```

The reason it appends rather than templating: existing tests render
`REASONING_DRIFT_USER_PROMPT_TEMPLATE.format(...)` with the current key
set, and adding a new placeholder would break them. When
`available_agents is None` (the default), the prompt is byte-identical to
the pre-#244 shape. The agent tree exists so the judge does **not** flag
legitimate coordinator → sub-agent delegation as OFF_TOPIC — see the
brussels-sprouts `web_developer_agent` false-positive documented in the
module comment. This is the same `goldfive.wrap` coordinator+AgentTool
contract from HARD INVARIANT 3: any tree shape must work. The steerer
resolves the tree via `DriftObserver._resolve_available_agents()`, which
reads `adapter.available_agents_tree` (structured) or falls back to
`adapter.available_agents` (flat names); with neither it stays `None`.

`format_plan_tasks_summary` also grows delegate-annotations when the
agent tree is passed: each task line can render
`[assignee=coordinator_agent; delegates to: research_agent, ...]` so the
judge sees at a glance that a coordinator-style assignee is allowed to
delegate for that task. With `available_agents=None` the render is the
byte-identical pre-#244 `- id -> title` form.

### 2.4 Input truncation caps — name them honestly

The judge sees a **bounded** slice of everything. These caps are real
recall limits: a deviation whose only evidence lives past the cap is
invisible to the judge. **Windowing (feeding the judge more history in a
sliding window) is DEFERRED work**, blocked on a judge regression harness
— do not implement it here without that harness. Until then, know the
caps:

| Constant | Value | Bounds |
|----------|-------|--------|
| `REASONING_DRIFT_MAX_REASONING_CHARS` | 1500 | The reasoning block sent to the judge. Longer → `text[:1500] + " ... [truncated]"`. ~300-400 tokens. |
| `PLAN_TASKS_SUMMARY_MAX_CHARS` | 2000 | The `id -> title` plan summary. Head-of-list preferred; drops tail with `... [N more task(s) elided]`. |
| `AGENT_TREE_BLOCK_MAX_CHARS` | 1200 | The AGENT TREE section (#244). Drops tail with `... [N more agent(s) elided]`. |
| `REASONING_DRIFT_TOOL_OBS_MAX_CHARS` | 1500 | Total rendered tool-observations block. |
| `REASONING_DRIFT_TOOL_OBS_MAX_ENTRIES` | 8 | Max tool observations rendered (tail of the list). |

Separately, the **observability** caps bound what goes onto the
`ReasoningJudgeInvoked` event (not what the judge sees):

| Constant | Value | Bounds |
|----------|-------|--------|
| `REASONING_JUDGE_MAX_REASONING_INPUT_CHARS` | 4096 | `reasoning_input` on the event + `trigger_input` on the drift. |
| `REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS` | 2048 | `raw_response` on the event. |

And the **output** budget:

| Constant | Value | Bounds |
|----------|-------|--------|
| `REASONING_JUDGE_MAX_OUTPUT_TOKENS` | 16384 | `max_output_tokens` for the judge dispatch. See §7.5 for why 16k and not 2k. |

`_format_tool_observations` filters the observations to the current task's
first, and only falls back to a global slice when the per-task slice is
empty (§3.4 design decision — a deviation rooted in an earlier task's tool
result is still useful context). It renders each entry as
`- {ERROR|ok} {agent} {tool}({args_preview}) -> {result_preview}` and
returns `(rendered_block, count_used)` so the template's
`(last N, oldest first)` header is honest about how many entries actually
fit. Per-entry truncation (args 240 chars / result 480 chars) is done at
**write** time by `DefaultSteerer.note_tool_observation`, not here.

### 2.5 The dispatch: budget + thinking-disable + diagnostics

The single `await call_llm(...)` is wrapped in **three** context managers
from `goldfive/_llm.py` plus a shared LLM span:

```python
from goldfive._llm import (
    call_llm_budget,
    call_llm_thinking_disabled,
    llm_call_diagnostics,
)

with (
    call_llm_budget(REASONING_JUDGE_MAX_OUTPUT_TOKENS),
    call_llm_thinking_disabled(),
    llm_call_diagnostics() as llm_diag,
):
    raw = await call_llm(system, user, model)
```

Wrapping the whole thing is `goldfive_llm_span(...)` (from
`goldfive/_llm_span.py`) so harmonograf renders the judge call as a span
named `judge_reasoning` on the goldfive lane, with `input_preview` set to
the reasoning block. The span's `output_preview` and `decision_summary`
are stamped **inside** the `with` block so they reflect the parsed verdict
before the span's End event fires. See `12-events-sinks-telemetry.md` for
the span contract.

`llm_diag` is an `LlmCallDiagnostics` object (see §7). After the call, if
`on_task_parsed is None` (unparseable) and the raw answer is empty but
`llm_diag.thought_count > 0`, the span records
`empty answer (N thought part(s); the model spent its budget thinking and
emitted no JSON)` instead of an indistinguishable `raw=''`. This is the
whole reason the diagnostics channel exists — it turns two days of
misdiagnosis (v16 / Qwen 35B) into a one-line span message.

---

## 3. Three-state classification, provenance, and JUSTIFIED_DEVIATION

iter-10 (PR 3) replaced the old binary `on_task: bool` verdict with a
three-state `classification`. This is the single most edit-sensitive part
of the judge — study §2.4 rules 1/2/3 in the code comments before
changing it.

### 3.1 The three states

The judge is asked to return one of:

- `on_task` — advances the bound task or the goals. Includes clarifying
  sub-steps, exploring tradeoffs, working through calculations. Emits
  **no drift**.
- `justified_deviation` — departs from the bound task, but a recent tool
  observation / surprising result / discovered dependency / new
  information **visible in the prompt** provoked it. Emits a
  `JUSTIFIED_DEVIATION` drift at the judge's severity.
- `erroneous_deviation` — departs with **no** such provoking signal in the
  context. Emits an `OFF_TOPIC` drift at the judge's severity.

The critical prompt discipline (INVARIANT 2 in spirit): the agent's
**claim** that a signal exists ("based on user instructions", "the user
asked for X") is **not** evidence. The judge must find X in the GOALS
section verbatim or in RECENT TOOL OBSERVATIONS. If it can't, classify
`erroneous_deviation` regardless of what the agent claims. This is baked
into the GUIDANCE block of the prompt, not into any Python heuristic.

### 3.2 Parse order: classification first, legacy `on_task` fallback

The parser (inside the span `with` block) reads `classification` first,
then falls back to the legacy boolean `on_task` for operator prompt
overrides that still produce the old shape:

```python
classification_raw = parsed.get("classification", "")
if isinstance(classification_raw, str):
    candidate = classification_raw.strip().lower()
    if candidate in _VALID_CLASSIFICATIONS:
        classification_parsed = candidate
if not classification_parsed:
    on_task_legacy = parsed.get("on_task")
    if isinstance(on_task_legacy, bool):
        classification_parsed = "on_task" if on_task_legacy else "erroneous_deviation"
```

`_VALID_CLASSIFICATIONS` is a `frozenset` of the three literals. A
misspelled classification with no legacy `on_task` fallback leaves
`classification_parsed == ""` — the **quiet-fail sentinel** (§5).

### 3.3 Provenance validation and demotion (§2.4 rule 3)

`justified_deviation` **requires** a valid provenance. The four valid
values are `_VALID_PROVENANCES = {"tool_error", "surprising_result",
"discovered_dependency", "new_information"}`. Anything else — missing key,
`"none"`, unknown free text — **demotes** the verdict to
`erroneous_deviation`:

```python
if classification_parsed == "justified_deviation":
    provenance_raw = parsed.get("provenance", "")
    candidate = provenance_raw.strip().lower() if isinstance(provenance_raw, str) else ""
    if candidate in _VALID_PROVENANCES:
        provenance_parsed = candidate
    else:
        classification_parsed = "erroneous_deviation"   # demote
        provenance_parsed = ""
        on_task_parsed = False
```

The demotion is logged at **INFO** (not WARNING) in the drift-construction
branch below, so operators can grep the rate of "model claimed justified
but couldn't name a provenance" in production. The log message
deliberately carries both the original raw `classification` value and the
post-demotion classification. Once demoted, the verdict **is**
`erroneous_deviation` by every downstream surface — the drift renders as
`OFF_TOPIC`, indistinguishable from a model-emitted `erroneous_deviation`.
That is intentional.

### 3.4 Drift construction

After the span block, the routing branch builds the drift:

| `classification_parsed` | Drift kind | Severity | detail prefix |
|-------------------------|-----------|----------|---------------|
| `""` (sentinel) | none | — | (no drift; quiet-fail logged at DEBUG) |
| `on_task` | none | — | (no drift; "on-track" logged at DEBUG) |
| `justified_deviation` | `JUSTIFIED_DEVIATION` | judge's severity | `justified deviation ({provenance}): {reason}` |
| `erroneous_deviation` | `OFF_TOPIC` | judge's severity | `reasoning drift: {reason}` |

Both drift kinds stamp `raw=reasoning`,
`trigger_input=truncate_for_observability(reasoning, 4096)`, and
`observed_revision_index` (captured **before** the LLM call — see §11).

Note `on_task_parsed` is a separate `bool | None` that mirrors the
classification for the legacy proto `on_task` field and the span's
`output_preview`. It is `True` for `on_task`, `False` for either
deviation, and `None` on quiet-fail. Do not delete it: the
`ReasoningJudgeInvoked.on_task` wire field (field 8) is computed for
back-compat with harmonograf columns.

---

## 4. Attribution fields and the #480 wire

Independently of the drift verdict, the judge extracts four attribution
fields. These are parsed on **every** verdict (on-task or off) because
off-task reasoning still has a focus — that is how the steerer learns the
agent has silently switched to a different plan task.

| Field | Type | Meaning | Default on failure |
|-------|------|---------|--------------------|
| `focused_task_id` | `str` | Plan-task id the judge says the reasoning is actually working on. `""` if off-plan. | `""` |
| `focus_confidence` | `float` | Subjective certainty, clamped to `[0.0, 1.0]`. | `0.0` |
| `stated_intent` | `str` | One-sentence summary of what the agent says it is doing. Observability-only. | `""` |
| `provenance` | `str` | The justified-deviation signal (post-demotion). | `""` |

Parse + clamp:

```python
focused_raw = parsed.get("focused_task_id", "")
if isinstance(focused_raw, str):
    focused_task_id_parsed = focused_raw.strip()
conf_raw = parsed.get("focus_confidence", 0.0)
try:
    focus_confidence_parsed = float(conf_raw)
except (TypeError, ValueError):
    focus_confidence_parsed = 0.0
focus_confidence_parsed = max(0.0, min(1.0, focus_confidence_parsed))
```

The clamp is defensive: the prompt asks for `0.0-1.0` but the code does
not trust the model to stay in range.

### 4.1 Where the attribution goes

`focused_task_id` + `focus_confidence` feed
`_maybe_record_reasoning_binding` in `drift_observer.py`, which records a
**reasoning-extracted binding** onto the `StateStore` when:

- `agent_name` is non-empty (bindings are keyed by agent name — a
  **stable** key, honoring INVARIANT 6, never an LLM-minted id);
- `focused_task_id` is non-empty;
- `focus_confidence >= _reasoning_binding_confidence_threshold`.

Lower-confidence verdicts are silently dropped so the pin-resolution
ladder does not consume noisy bindings. The binding is recorded
**regardless of the drift verdict** — an on-task verdict that names a
*different* plan task is itself a useful pin-resolution signal. Store
failures degrade silently: the judge's primary job is the drift signal,
not the binding. See `11-state-ownership.md` for the pin ladder.

### 4.2 The `ReasoningJudgeInvoked` wire fields (#480)

All four attribution fields, plus `classification`, ride the wire on the
`ReasoningJudgeInvoked` proto event so the telemetry carries the judge's
full verdict, not just the drift projection. The proto layout
(`proto/goldfive/v1/events.proto`, `message ReasoningJudgeInvoked`):

| Field # | Name | Notes |
|---------|------|-------|
| 1-7 | `run_id`, `task_id`, `subject_agent_id`, `model`, `elapsed_ms`, `reasoning_input`, `raw_response` | Envelope + truncated I/O. |
| 8 | `on_task` | Legacy boolean projection: `true` iff `classification == "on_task"`. |
| 9 | `severity` | `info`/`warning`/`critical` when non-on_task, else `""`. |
| 10 | `reason` | Judge's one-sentence explanation. |
| 11 | `classification` | Three-state string; `""` is the quiet-fail sentinel. |
| 12 | `focused_task_id` | (#480) |
| 13 | `focus_confidence` | `float`, clamped `[0,1]`. (#480) |
| 14 | `stated_intent` | (#480) |
| 15 | `provenance` | Post-demotion. (#480) |

The event is emitted by `_emit_judge_invoked` on **every** invocation,
regardless of verdict — on-task, off-task, and plumbing-failure paths all
produce one. It is emitted **after** the drift decision so it carries the
parsed outcome, but its emission is independent of whether a drift fired.
A broken sink must not break the run: every exception (including a
proto-import failure from a partially-regenerated tree) is caught and
logged at WARNING inside `_emit_judge_invoked`.

`on_task` for the event is computed from the final `classification_parsed`
(post-demotion), falling back to `drift is None` when there is no
classification, so the boolean stays consistent with the three-state field
even after a demotion:

```python
on_task_for_event = (
    classification_parsed == "on_task"
    if classification_parsed
    else (drift is None)
)
```

---

## 5. The quiet-fail sentinel

The empty string `classification == ""` is the **quiet-fail sentinel**.
Callers MUST treat it as identical to `on_task` for routing — no drift, no
cancel, no refine. This preserves the pre-iter-10 fail-quiet contract from
#143 / #226 (INVARIANT 3). The sentinel is produced by **any** of:

- `call_llm` raised (caught by the broad `except Exception`; `raw` set to
  a diagnostic string, `call_failed = True`, `parsed` stays `None`);
- the response was not JSON (`_parse_response` returns `None`);
- the JSON parsed but carried neither a recognized `classification` nor a
  boolean `on_task`;
- the model spent its whole budget thinking and returned an empty answer.

`ReasoningJudgeVerdict` (frozen dataclass) carries two extra fields that
let downstream distinguish these cases:

- `judge_ran: bool` — `True` iff the judge LLM was actually dispatched.
  `False` on the empty-reasoning early return, the embedding-only path,
  and `mode="off"`. So `judge_ran and not classification` == "the judge
  ran and quiet-failed"; `not judge_ran` == "the judge never ran".
- `elapsed_ms: int` — mirrors the value stamped on the event; `0` when
  `judge_ran` is `False`.

The verdict-utility ledger keys `parse_fail` on exactly
`judge_ran and not classification` (§8.3). **Do not "fix" a quiet-fail by
raising or by emitting a default drift.** Silence is the contract.

---

## 6. Severity parsing and the malformed→INFO rule (#479)

The judge's `severity` string maps to a `DriftSeverity` via `_SEVERITY_MAP`:

```python
_SEVERITY_MAP = {"info": DriftSeverity.INFO, "warning": DriftSeverity.WARNING, "critical": DriftSeverity.CRITICAL}

def _severity_from_verdict(raw):
    if isinstance(raw, str):
        severity = _SEVERITY_MAP.get(raw.strip().lower())
        if severity is not None:
            return severity
    log.debug("... severity %r missing or unrecognised; defaulting to INFO", raw)
    return DriftSeverity.INFO
```

**#479 rule: a malformed / missing / unrecognized severity defaults to
INFO, and the drift is still emitted.** The reasoning: the verdict must
never be silently swallowed (a real deviation with a garbled severity is
still a deviation), but a malformed severity string must **not** be
promotion-eligible. `DriftObserver._should_promote_to_steer` gates on
WARNING-and-up (see `09-steering-ladder-and-gates.md`), so an
INFO-defaulted drift lands on the wire for observability but does not
trigger cancel/refine. This is the safe direction: garbage in the severity
field cannot escalate an intervention.

`_severity_from_verdict` is exact-equality/hash matching on the judge's
**own structured enum output** — that is explicitly permitted under
INVARIANT 2 (it is not NL classification). Do not replace it with
fuzzy/substring matching.

---

## 7. `goldfive/_llm.py` — the one internal LLM-call module (#491)

Before #491 this logic lived as two divergent copies in
`goldfive/_llm_detect.py` and `goldfive/convenience.py`, and diagnostics
were smuggled as attributes mutated on the shared callable
(`call_llm.last_thought_count`) — last-writer-wins once concurrent
background judges dispatched through the same closure. #491 collapsed it
into one module and replaced the closure-attribute channel with a
ContextVar-bound per-call object. `_llm_detect` / `convenience` keep thin
re-export shims.

### 7.1 The `call_llm` contract

Everything speaks the opaque async signature `(system, user, model) -> str`
(`CallLLM`, a `runtime_checkable` `Protocol`). `ClosableCallLLM` extends it
with an optional async `close()`; `maybe_close_call_llm` probes
`getattr(call_llm, "close", None)` and awaits it if present, swallowing
exceptions (`Runner.close` uses it for callables the Runner owns).
`CallLLM`, `ClosableCallLLM`, and `maybe_close_call_llm` are public imports
from `goldfive`. A caller that supplies a callable retains ownership unless
the accepting API explicitly says otherwise. The signature is **opaque on
purpose**. Adding a `max_tokens` parameter would break every user-supplied
callable, so per-call knobs travel through ContextVars.

### 7.2 The three ContextVars

| ContextVar | Manager / reader | Purpose |
|-----------|------------------|---------|
| `MAX_OUTPUT_TOKENS_VAR` | `call_llm_budget(n)` / `get_max_output_tokens()` | Per-callsite output cap. `None`/`<=0` → `DEFAULT_MAX_OUTPUT_TOKENS = 4096`. |
| `THINKING_DISABLED_VAR` | `call_llm_thinking_disabled()` / `get_thinking_disabled()` | Suppress thinking for this dispatch. Default `False`. |
| `LLM_CALL_DIAGNOSTICS_VAR` | `llm_call_diagnostics()` (yields the object) / `record_llm_call_diagnostics(...)` | Per-call `LlmCallDiagnostics(thought_count, answer_count)`. |

All three managers restore the prior value on exit **even if the body
raises** (they use `ContextVar.reset(token)` in a `finally`). Judges,
goal_deriver, planner refine, and the reflective check all enter the
budget + thinking-disable managers around their `await call_llm(...)`.

**The operator-callable compatibility rule:** all three are **optional**
for user-supplied callables. A bare lambda that ignores the ContextVars
still works — the only cost of ignoring the budget var is the model
emitting to its natural stop (the very behavior that produced 9.6-minute
calls in #271 evidence), and ignoring the thinking var means the model
keeps thinking. goldfive's **default builders** read the vars; users are
never required to. `record_llm_call_diagnostics` is a **no-op** when no
consumer installed a diagnostics object, so a user callable that never
records leaves the counts at zero.

### 7.3 The two default builders

- `make_default_adk_call_llm(model)` — backs `call_llm` with an ADK
  `BaseLlm` (string alias, `BaseLlm` instance, or `LiteLlm`). Returns
  `None` when ADK is not installed or the model can't resolve. On this
  path the genai `ThinkingConfig(include_thoughts=False, thinking_budget=0)`
  opt-out is **first-class SDK surface** and applies to **every** model
  when `get_thinking_disabled()` is set. It counts real `thought=True`
  parts vs answer parts across the streamed response and calls
  `_note_dispatch_result`.
- `make_default_openai_call_llm(config: JudgeConfig)` — builds an
  `AsyncOpenAI` client against a dedicated judge endpoint
  (`{base_url}/v1`). Returns `(call_llm, model)` or `None`. Tolerates
  missing/placeholder `api_key` (`"not-needed"`) so llama.cpp / Ollama
  "just work". On this OpenAI-compatible wire, thinking-disable is a
  **Qwen-family-only** hack (see §7.4). It reports `reasoning_content`
  presence as a 0/1 sentinel for `thought_count`.

`JudgeConfig` and `make_default_openai_call_llm` are public imports from
`goldfive`. This is the supported construction path for an application that
wants Goldfive's OpenAI-compatible dispatch behavior while retaining endpoint
ownership. The application must call `await maybe_close_call_llm(call_llm)`
after its last dispatch. Supplying that callable as `judge_call_llm` to
`wrap` or `run` does not transfer ownership to Goldfive.

Both attach a `close` coroutine (via `_probe_close`, which duck-types
`aclose`/`close` on the target and a nested `._client`/`.client`). Both
call `_note_dispatch_result(transport=..., result=..., thought_count=...,
answer_count=...)`, which records the diagnostics AND logs the
all-thought-no-answer failure shape at INFO so the same observable shape
appears on success and failure.

### 7.4 The vendor capability table (`THINKING_DISABLE_CAPABILITIES`)

How a "disable thinking" request is expressed on the wire is a **vendor
convention**, not a transport property. The genai `ThinkingConfig` opt-out
is universal on the ADK path. The two Qwen-specific hacks — the
`extra_body={"enable_thinking": False}` field and the `/no_think`
system-prompt prefix — ride the OpenAI-compatible format and are
meaningful only to the Qwen/litellm family, so they are keyed off the
model name:

```python
@dataclass(frozen=True)
class ThinkingDisableCaps:
    openai_enable_thinking_field: bool = False
    no_think_prompt_prefix: bool = False

THINKING_DISABLE_CAPABILITIES = (
    ("qwen", ThinkingDisableCaps(openai_enable_thinking_field=True, no_think_prompt_prefix=True)),
)

def thinking_disable_caps(model_name):
    lowered = (model_name or "").lower()
    for marker, caps in THINKING_DISABLE_CAPABILITIES:
        if marker in lowered:
            return caps
    return _NO_VENDOR_THINKING_CAPS
```

This is a **lookup table of vendor conventions (configuration)**, matched
by lowercase substring so litellm-prefixed names (`"openai/Qwen3-32B"`,
`"hosted_vllm/Qwen/Qwen3-32B"`) route to the same family. **It is not NL
classification** — it matches a model identifier, not agent text, so it
does not violate INVARIANT 2. Unknown models get the empty caps (no vendor
hacks); the genai `ThinkingConfig` still applies on the ADK path. #491
narrowed the `/no_think` + `enable_thinking` hacks to the Qwen / litellm
family — do not re-broaden them to all OpenAI-compatible models.

The OpenAI builder also handles old SDKs that reject `extra_body`: it
catches `TypeError`, drops `extra_body`, and retries (the `/no_think`
prompt prefix still does its job for Qwen). Note this retry is **not** a
second judge dispatch in the INVARIANT-5 sense — it is a same-call retry
after the SDK rejected a kwarg, not a re-ask of the model.

### 7.5 Why 16k, not 2k (the sizing rationale)

`REASONING_JUDGE_MAX_OUTPUT_TOKENS = 16384` and
`GOAL_DRIFT_MAX_OUTPUT_TOKENS = 16384`. Qwen 3.5 thinking models combine
`<think>` reasoning and the final answer under one `max_output_tokens`
ceiling. Capping at 2048 produced empty (`raw=''`) responses on v16
because the model exhausted its budget inside the think block before
emitting a single JSON byte — no drift fired and the cascade never
started. The 16k cap was the **symptom** fix; `call_llm_thinking_disabled()`
is the **cause** fix (a JSON-shaped meta-cognition question does not need
thinking at all). Both ship together. The `DEFAULT_MAX_OUTPUT_TOKENS`
(4096) is the fallback for an unsupervised dispatch; the wall-clock
backstop is `DEFAULT_LLM_CALL_TIMEOUT_MS` in the ADK plugin (§10.1).

---

## 8. Judge scheduling (#483) in `drift_observer.py`

The reasoning judge is **fire-and-forget** off the critical path — see
`observe_reasoning`'s docstring for why (awaiting a minute-long
local-llama round-trip inline serialized every subsequent tool call behind
it). #483 added the guards that keep a burst of concurrent judge calls
from stampeding the endpoint.

### 8.1 The per-steerer semaphore + coalescing registry

**Concurrency cap.** `DriftObserver._judge_semaphore` is an
`asyncio.Semaphore` sized from
`ReasoningDriftConfig.max_concurrent_judges` (default **3**, env
`GOLDFIVE_DRIFT_MAX_CONCURRENT_JUDGES`), clamped to `>= 1`. It is
**per-steerer-instance, not module-global** — a multi-Runner process never
shares one gate. N agents thinking in the same event-loop tick used to
fire N parallel judge calls (each with the 16k ceiling) at the same
endpoint; the semaphore bounds the burst.

**Coalescing.** While a background judge task waits on the semaphore, its
request is **QUEUED** in `_queued_judge_windows`, a dict keyed by
`(session_id, agent_name, task_id)`. A newer reasoning observation for the
**same key** does not schedule a second task — it mutates the queued
`_QueuedJudgeWindow` in place (newest `text`/`pinned_history` wins,
`coalesced` counter bumps). A granted judge slot (`call_llm`) is never
downgraded by a slotless newer observation. Once the task acquires the
semaphore it is **RUNNING** and its entry leaves the registry — a RUNNING
call is never coalesced.

```python
# observe_reasoning
queue_key = (str(session.id or ""), agent_name or "", str(session.current_task_id or ""))
queued = self._queued_judge_windows.get(queue_key)
if queued is not None:
    queued.text = text
    queued.pinned_history = pinned_history
    if rl_call_llm is not None:
        queued.call_llm = rl_call_llm
    queued.coalesced += 1
    return                    # fold into the queued window; no new task
window = _QueuedJudgeWindow(text=text, pinned_history=pinned_history, call_llm=rl_call_llm)
self._queued_judge_windows[queue_key] = window
bg_task = asyncio.create_task(
    self._run_judge_background(queue_key=queue_key, window=window, session=session, ...),
    name=f"goldfive-reasoning-judge:{session.id}",
)
```

`_run_judge_background` acquires the semaphore, deletes the registry entry
(QUEUED → RUNNING) **before** reading the payload so a newer same-key
observation schedules a fresh request instead of mutating a window that is
already being judged, and calls `_run_judge_window`. A `finally` removes
the entry if the task is cancelled while still queued (run-boundary drain,
shutdown) so later observations cannot coalesce onto a dead window and
silently vanish:

```python
async def _run_judge_background(self, *, queue_key, window, session, judge_sink, agent_name=""):
    try:
        async with self._judge_semaphore:
            if self._queued_judge_windows.get(queue_key) is window:
                del self._queued_judge_windows[queue_key]     # QUEUED -> RUNNING
            await self._run_judge_window(text=window.text, session=session,
                                         call_llm=window.call_llm, judge_sink=judge_sink,
                                         pinned_history=window.pinned_history, agent_name=agent_name)
    finally:
        if self._queued_judge_windows.get(queue_key) is window:
            del self._queued_judge_windows[queue_key]
```

**The task name is load-bearing.** `goldfive-reasoning-judge:{session.id}`
(and `goldfive-goal-drift-judge:{...}`, `goldfive-custom-judge:{...}`) let
`drain_session_background_tasks` (#243) filter pending tasks by the run
boundary that is terminating, leaving other concurrent sessions' tasks
alone. If you spawn a new judge task, encode `session.id` in the name.

### 8.2 The rate-limit slot (`_maybe_take_reasoning_judge_slot`)

Before scheduling, `observe_reasoning` takes a rate-limit slot. Policy
(#226):

- First thinking message of every `(agent_name, task_id)` bucket always
  fires the judge.
- Subsequent messages skip `(rate_limit - 1)` and fire on the Nth
  (`reasoning_drift_rate_limit`, default **3**, `max(1, ...)`).
- Counters live in `session._reasoning_judge_counters` keyed by
  `(agent_name, task_id)` — a task transition **or** an agent switch
  resets the window lazily (the new tuple is simply absent from the dict).

```python
# _maybe_take_reasoning_judge_slot
key = (agent_name or "", session.current_task_id or "")
count = counters.get(key, 0)
fire = (count % self._steerer._reasoning_drift_rate_limit) == 0
counters[key] = count + 1
return self._steerer._reasoning_drift_call_llm if fire else None
```

The pre-fix key was a single `current_task_id or ""` string, so every
unpinned turn from every agent collapsed onto the `""` bucket and agent
B's first thinking block could skip the judge because agent A's unpinned
turn had already incremented the counter. Bucketing by `(agent, task)`
isolates each agent's cadence — this is the "stable identity key" lesson
(INVARIANT 6) applied to rate-limit gates.

Returns `None` (no slot) when the judge is globally disabled
(`_reasoning_drift_call_llm is None`, or mode not in `("judge","both")`)
or on a skip turn. In `mode="judge"` a `None` slot short-circuits before
scheduling; in `"embedding"`/`"both"` the task still schedules because the
embedding pipeline runs even with an empty judge slot.

### 8.3 The verdict-utility ledger

`_verdict_ledger(session)` is a plain per-session dict on the observer,
created lazily on first increment. Fields and where each increments:

| Field | Increment site | Meaning |
|-------|---------------|---------|
| `acted_on` | `_run_judge_window`, just before `handle_drift` | Reasoning-judge verdicts dispatched into the ladder (past the late gate). |
| `emitted_late` | `_run_judge_window`, the `_is_late_drift_for_terminated_invocation` branch | Verdict emitted-only because the originating invocation already terminated (#319). |
| `emitted_redundant` | `handle_drift`, both the addressed-watermark and the in-flight-refine gates | Verdict emitted-only at `handle_drift`'s entry gates. Counts any observation-stamped verdict that hits those gates. |
| `parse_fail` | `_run_judge_window`, when `judge_ran and not classification` | Judge calls that quiet-failed (empty-classification sentinel). |
| `elapsed_ms` | `_run_judge_window`, when `judge_ran` | Bounded latency sample list (cap `_LEDGER_ELAPSED_SAMPLES_CAP = 1024`). |

At the run boundary, `drain_session_background_tasks` drains the bg tasks
for that session, then `_emit_verdict_utility_summary(session_id)` pops the
ledger and emits a `reasoning_judge_utility_summary` **dict** event (via
`make_event` — **no proto change**) carrying the four counters,
`judge_calls` (= sample count), and nearest-rank `elapsed_ms_p50` /
`elapsed_ms_p95`. A session with no judge activity never created a ledger,
so quiet runs emit nothing; the pop makes repeat drains idempotent.
`shutdown` flushes any ledgers whose sessions never hit a run-boundary
drain (custom executors, aborted loops) as the teardown fallback.

The summary is emitted **before** the executor's terminal
`RunAborted`/`RunCompleted` so it rides inside the run. See
`12-events-sinks-telemetry.md` for consuming the dict event.

### 8.4 Late-drift tolerance and staleness gates

Because the judge is fire-and-forget, its verdict can land **after** the
invocation that produced the reasoning has already terminated (adk-web
outer-turn boundary crossed, agent moved on). `_run_judge_window` has a
`_is_late_drift_for_terminated_invocation` guard (#319): such a verdict is
emitted directly via `_emit_drift_detected` (so operators see "from past
turn") and **skips** the cancel + ladder dispatch — routing it through
cancel would kill an unrelated next invocation, and refining would target
a plan whose offending step is already complete. This guard is scoped to
the background path because only that path produces verdicts that outlive
the originating invocation; synchronous detectors always see a live one.

Symmetrically, an **on-task** verdict is the recovery signal for
conditions this same pipeline opened. It is gated on the same staleness
predicate (`_invocation_target_gone`) before it is allowed to resolve
conditions — a verdict landing after its invocation terminated cannot
resolve a fresh condition opened by a newer turn. When the judge is
on-task and live, `_resolve_conditions_on_on_task_verdict` runs and (per
#486) staleness-guarded on-task verdicts may emit
`DRIFT_LIFECYCLE_RESOLVED`.

The **freshness / staleness gates** a verdict must pass in `handle_drift`
(before any cancel/refine) are:

1. **Addressed-watermark gate.** Keyed by `(drift.kind.value,
   current_task_id or "")` against
   `session.last_addressed_revision_by_drift_key`. If the same
   `(kind, target)` was already addressed at a **later** revision than the
   verdict's `observed_revision_index`, drop it (emit for observability
   with `decision_outcome="drift_dropped_stale"`, bump `emitted_redundant`).
   This is per-`(kind, target)` — narrower than a naive
   `observed < live_revision` compare — so parallel judges firing on
   **orthogonal** concerns are not over-rejected. (This is the
   reader-centric-versioning lesson: key on the claim's identity, not a
   global writer counter.)
2. **In-flight-refine gate.** A synchronous `inflight_key =
   (session_id, kind, target)` in `self._inflight_refine_keys`, stamped
   before dispatch and cleared in `finally`. A second concurrent same-key
   judge sees the entry, emits for observability with
   `decision_outcome="drift_dropped_inflight"`, bumps `emitted_redundant`,
   and skips. This closes the race where two judges both read
   `last_addressed == 0` and both run a refine for one drift (the plan lock
   is not held across the multi-second `planner.refine`).

Both gates **bypass** unstamped drifts (`observed_revision_index == 0`,
legacy/external producers) and **user-authored** drifts
(`authored_by == "user"`) — an operator directive is honored regardless of
the plan cursor. See `09-steering-ladder-and-gates.md` for the full
`handle_drift` flow.

### 8.5 Snapshot-passing (#479)

The judge history is **pinned by snapshot-passing**, not read live.
`observe_reasoning` captures `pinned_history = list(session.reasoning_history)`
at schedule time and threads it through `_run_judge_window` →
`analyze_reasoning_with_focus(reasoning_history=pinned_history)`. Without
this, a history-window detector that slices `history[-N:-1]` (expecting
`text` to be the last entry) would see later-appended turns and either
self-match (false LOOPING) or compare against the wrong window. The live
`session.reasoning_history` list itself is **never mutated** by the
background path, so concurrent readers always see the live list. If you add
a detector that reads history, take the `reasoning_history` argument; do
not read `session.reasoning_history` directly in the background path.

---

## 9. The goal-drift judge (`goldfive/drift/goals.py`)

`classify_goal_drift` asks the trajectory-level question. It emits a
`GOAL_DRIFT` drift at **CRITICAL** severity when the judge returns
`{"progressing": false, "reason": "..."}`, and `None` in every other case
(progressing-true, malformed JSON, missing/non-boolean `progressing`,
`call_llm` raised). Same "quiet on failure" contract (INVARIANT 3) — a
false-positive trajectory-level alarm erodes trust faster than it helps.

### 9.1 Prompt

`GOAL_DRIFT_SYSTEM_PROMPT` + `GOAL_DRIFT_USER_PROMPT_TEMPLATE`. The user
template sections: `GOALS`, `PLANNED TASKS` (via `_format_tasks` — numbered
`[id] title (STATUS)` lines), and `RECENT AGENT ACTIVITY (most recent
{activity_count} invocations, newest last)` (via `_format_activity`). The
instruction asks for **strictly** one of two JSON shapes:
`{"progressing": true}` or `{"progressing": false, "reason": "..."}`. There
is no three-state, no severity field, no attribution — this judge is
binary.

### 9.2 Triggers (who calls it)

Three scheduling paths, all in `DriftObserver`, all **fire-and-forget** via
`_spawn_goal_drift_judge_background` → `_run_goal_drift_judge_background` →
`maybe_run_goal_drift_check`:

1. **Turn counter** — `note_agent_turn` increments
   `session._agent_turns_since_goal_check`; at `goal_drift_check_interval`
   (default **5**, `GoalDriftConfig.check_interval`) it fires and resets.
   Trajectory-level, so it is **not** reset on task transitions.
2. **Task boundary** — `_maybe_run_goal_drift_on_task_boundary` fires on
   task completion/failure/cancellation (natural "am I still on plan?"
   checkpoints), rate-limited to one call per
   `_GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S = 10.0` seconds via
   `session._last_goal_drift_check_ts`. Short pipelines that finish before
   5 turns would otherwise never trigger the judge (#219).
3. **Idle episode (#487, flag-gated)** — when
   `SteeringConfig.stall_watchdog_enabled` is on (default **OFF**) and the
   session's liveness watermark (`Session.last_observed_event_at`) has been
   silent for `GOAL_DRIFT_IDLE_SECONDS` (default **300**), the wall-clock
   stall watchdog triggers `maybe_run_goal_drift_check(session,
   idle_note=...)` once per idle episode. `GOAL_DRIFT_IDLE_SECONDS` is read
   **live** from the module attribute on every poll, so a zicato
   optimization-manifest `setattr` mutation takes effect on a running
   watchdog. See §10 and `07-deterministic-drift-detection.md` for the
   watchdog (the TASK_TIMEOUT producer).

**Why fire-and-forget (the v22 regression):** these call sites run on the
agent's ADK invocation task, which is registered for cooperative cancel. A
sibling drift firing `request_invocation_cancel` while the inline judge
awaited its LLM round-trip landed a `CancelledError` inside
`classify_goal_drift`; the `judge_goal_drift` span ended with
`error=CancelledError` and the verdict was lost. Detaching into a separate
`asyncio.Task` isolates the judge from the agent invocation's cancel scope
(asyncio Tasks do not form a parent-child cancel tree). The task is tracked
on `_background_judges` and drained by `shutdown`.

### 9.3 The `idle_note` activity entry

When `maybe_run_goal_drift_check` is called with a non-empty `idle_note`,
it appends a **synthetic** activity entry so the judge's activity block
renders the idle observation without any prompt-template change:

```python
activity = filter_recent_events_by_kind(session.recent_events, RECENT_EVENT_AGENT_ACTIVITY_KINDS)
if idle_note:
    activity = [*activity, {"kind": "idle_observed", "detail": idle_note}]
```

`_format_activity` then renders it as e.g.
`- idle_observed: 300s since last observed activity`. This is how the idle
watchdog tells the judge "nothing has happened for 300s" — the judge sees
an activity list that ends with an idle marker and can decide the
trajectory has stalled. Activity is read from the unified
`session.recent_events` buffer (#239) filtered to the agent-activity kinds
(byte-identical to the pre-merge `recent_agent_activity`).

### 9.4 The post-LLM plan re-read (#245)

The goal judge captures `observed_revision_index` and a snapshot of
`{task_id: status}` **before** the LLM call, then **re-reads**
`session.plan` after the round-trip and drops the verdict if the plan moved
under it. This kills the brussels/tomato false-positive class: the judge
complained "drafting still pending" against a snapshot where draft was
already DONE by the time the verdict arrived. Two tiers:

1. **Targeted** — if the reason text names a specific plan task by id and
   that task transitioned status during the call, drop.
2. **Generic** — otherwise compare `revision_index` + the `(id, status)`
   set against the pre-call snapshot; if either changed, drop.

`session` is optional (legacy callers keep pre-#245 behavior); the
dispatch-time gate in `handle_drift` (§8.4) is the second line of defense.
The re-read is inside the classifier; the watermark gate is in the observer
— they are complementary, not redundant.

The goal judge also stamps `trigger_input` with the activity block it saw
(truncated to `_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS = 2048`) so sinks can
render "why did goldfive think this was off-goal?" without re-fetching the
activity log. On a `None` drift it emits an `emit_no_drift_decision`
(detector `goal_drift_judge`, reason "judge verdict: progressing") so the
zicato optimizer sees the negative class, not just the firing path.

---

## 10. The disarm warning (#476/#263) and endpoint-contention warning (#483)

Two operator-facing WARNINGs are worth knowing because they are how a
silently-dead judge surfaces. Neither changes behavior — both are pure
visibility.

### 10.1 Reasoning-channel-disarm warning + `LLM_CALL_TIMEOUT` under observation_only (#476/#263)

Non-thinking models (Gemma 4, Mistral, several base-model deployments)
never emit a separate reasoning/thinking stream, so `observe_reasoning`
never fires and **every** LLM-judge reasoning detector (OFF_TOPIC,
GOAL_DRIFT, INTENT_DIVERGENCE, LOOPING_REASONING) silently disarms for the
whole run. `_note_reasoning_channel_signal` in
`goldfive/adapters/_adk_plugin.py` fixes the invisibility: after
`_NO_REASONING_WARN_STREAK = 3` consecutive turns that carried a text body
but no reasoning stream, it fires a **one-shot per-agent** WARNING
(`goldfive.reasoning.disarmed agent=... — N consecutive ...`) naming the
remedy: set `ReasoningDriftConfig.fallback_to_content_when_no_reasoning=True`
(env `GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT=1`) to synthesize a signal from
the response body. The fallback is **not** auto-enabled — that behavior
change is reserved for the operator.

Counting rules: a turn that fed the channel (real reasoning or
content-fallback) **resets** the per-agent streak; a turn with an empty
reasoning source but a non-empty text body **increments** it;
function-call-only / empty turns neither count nor reset (thinking models
omit the stream on pure tool turns, so counting them would false-positive).
The warning also emits a record-only `CUSTOM`/INFO drift via
`_emit_drift_detected` so wire-watchers (not just stderr) see the disarm —
safe under `observation_only` because `_emit_drift_detected` is a
record-only path with no policy dispatch.

#476 relatedly made `LLM_CALL_TIMEOUT` **not cancel** under
`observation_only`: the per-call timeout watcher
(`_run_llm_call_timeout_watcher`, `DEFAULT_LLM_CALL_TIMEOUT_MS = 1_800_000`
= 30 min) still emits the CRITICAL `LLM_CALL_TIMEOUT` drift for
observability, but the cancel-flag write is skipped when
`_is_observation_only(ctx)` is true:

```python
# _run_llm_call_timeout_watcher — after emitting the drift
if _is_observation_only(ctx):
    log.info("goldfive.llm.timeout invocation_id=%s observation_only=True — cancel-flag write skipped", invocation_id)
    return
# ... only past here: request_invocation_cancel(...)
```

`_is_observation_only(ctx)` delegates to `steering_is_active(ctx.steerer)`
(the ONE sanctioned kill-switch read — INVARIANT 4/5) and treats a missing
steerer / missing predicate as **passive** — the fail-safe direction for a
surface that cancels in-flight work. Detection runs, intervention is gated.

### 10.2 Endpoint-contention warning (#483)

At `goldfive.wrap` time, when the judges' callable was **inherited** from
`detect_llm` — i.e. the operator passed no `judge_call_llm=`, no
`call_llm=`, and no `JudgeConfig` — `goldfive/convenience.py` logs a
WARNING naming the cost explicitly. Judge traffic includes up to
`max_concurrent_judges` calls in flight. Each call has a
`REASONING_JUDGE_MAX_OUTPUT_TOKENS` budget and lands on the
**same** endpoint the agent tree bills against, competing for
capacity/rate-limits. It points to `GOLDFIVE_JUDGE_BASE_URL` /
`GOLDFIVE_JUDGE_MODEL` (i.e. a `JudgeConfig`) to route judges to a
dedicated endpoint. An explicit `judge_call_llm=`, `call_llm=`, or
`JudgeConfig` suppresses the warning because the operator selected a
route. The warning
prefers the agent's own `.name` over the Python class name so it names
*which* agent the LLM was detected from.

`JudgeConfig` (in `config.py`) is exactly this dedicated-endpoint escape
hatch: when `base_url` is set, `goldfive.wrap` routes the two judges (and
only the judges — planner/goal_deriver keep the tree LLM) through
`make_default_openai_call_llm(config)`. Precedence: explicit
`judge_call_llm=` > explicit `call_llm=` > `JudgeConfig.base_url` >
auto-detected tree LLM. `judge_model=` overrides the model name passed
through the selected callable. Goldfive registers a close hook only for
the client it constructs from `JudgeConfig`.

---

## 11. Plan-revision snapshotting (why `observed_revision_index`)

Both judges capture the plan's `revision_index` **before** rendering the
prompt or awaiting the LLM:

```python
observed_revision_index = int(getattr(plan, "revision_index", 0) or 0)
```

and stamp it onto every drift they emit. This is the input to the freshness
gates (§8.4, §9.4). The rule is: **capture the revision the judge is
observing at call-start**, because during the multi-second LLM round-trip
the reconciler may transition tasks and refines may bump the plan. A
verdict computed against revision N that arrives after the plan moved to
N+1 (for the same kind+target) is stale. Do not read `plan.revision_index`
after the `await` for this purpose — that would capture the moved-on value
and defeat the gate.

---

## 12. The `judges/` package — pluggable judges

`goldfive/judges/` is the **operator-facing** custom-judge surface (#437).
It is deliberately thin, and there are honest limits to what a custom judge
can do today.

### 12.1 `JudgeContext` / `JudgeVerdict` / `Judge`

`goldfive/judges/base.py` defines three frozen/protocol types:

- `Judge` — `Protocol` with a stable `name: str` and
  `async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict`. Called
  once per observation point. A judge with nothing to say returns an
  empty-default `JudgeVerdict` (no flavour populated) and the steerer skips
  emission for it. Errors raised by `evaluate` are caught and logged at
  WARNING — one misbehaving judge must not break the run or suppress other
  judges' verdicts.
- `JudgeContext` — frozen snapshot: `reasoning_text`, `plan`, `transcript`
  (bounded, oldest-first), `session_state` (the live `Session`),
  `current_task_id`, `current_agent_id`, and an open-ended `extras: dict`.
  Judges MAY read `session.reasoning_history` / `session.recent_events` but
  **MUST NOT mutate** either — the state-ownership audit flags stray writes
  (see `11-state-ownership.md`). The plan is read-only; refines are driven
  by the steerer, never a judge.
- `JudgeVerdict` — frozen result with four flavours; the runtime picks
  `verdict_kind` from the first populated flavour in order: **drift**
  (`drift_emitted=True` + `drift_kind` + `severity`), **rubric**
  (`rubric_score` + `rubric_dimensions`), **boolean** (`boolean_result`),
  **numeric** (`numeric_value` + `metric_name`). `detail` is a free-form
  one-line string every flavour can set. `__post_init__` normalizes a
  recognized lowercase-string `drift_kind`/`severity` up to its enum (both
  are `StrEnum`, so string-equality consumers still match). Frozen so a
  misbehaving judge can't mutate steerer state between emission and
  dispatch.

### 12.2 `evaluate_judges` and the per-judge timeout

`DefaultSteerer.evaluate_judges(ctx, session=, run_id=, judges=)` iterates
the judges, awaits each `judge.evaluate(ctx)` bounded by
`JUDGE_EVALUATE_TIMEOUT_S = 30.0` (`asyncio.wait_for`), and:

- a timeout or exception → logged at WARNING, treated as **no signal**, the
  loop continues (one bad judge can't suppress the others);
- an empty verdict → skipped;
- a populated verdict → emits a `JudgementEmitted` envelope keyed on
  `judge.name`, and if it is **drift-flavoured**, ALSO routes it through
  `drift.handle_drift` so the legacy `DriftDetected` + refine machinery
  still fires (back-compat). The drift is marked
  `_judge_emitted_judgement = True` so the paired-emission path does not
  double-fire a second `JudgementEmitted`.

### 12.3 Built-in judges and the BUILTIN skip-list

`goldfive/judges/builtins.py` provides a `Judge` shim per existing
detector, a `BuiltinJudge` `StrEnum` (member value == wire name), factory
functions (`reasoning_drift()`, `goal_drift()`, `refusal()`,
`tool_error()`, `stop_reason()`, `looping_reasoning()`, `looping_tool()`),
`default_judges(disable=...)`, and `BUILTIN_JUDGE_NAMES` (a `frozenset[str]`
derived from the enum).

Crucial mechanic — the **skip-list**: the auto-wired observation path
(`DriftObserver._dispatch_custom_judges`, called from `observe_reasoning`)
runs `evaluate_judges` against **only** the *custom* judges — those whose
`name` is **not** in `BUILTIN_JUDGE_NAMES`:

```python
custom = [j for j in judges if str(getattr(j, "name", "")) not in BUILTIN_JUDGE_NAMES]
```

Built-ins are excluded because their drift verdicts already ride the wire
via the legacy detector path and its paired `_emit_judgement_from_drift`
emission. Re-running the built-in wrappers here would **double-fire**
`DriftDetected` for the same logical signal (#437). This is why the
built-in shims (`ReasoningDriftJudge`, `GoalDriftJudge`, `LoopingToolJudge`)
return an **empty-default** `JudgeVerdict` from `evaluate` — they exist for
protocol conformance and operator opt-in/opt-out, not to compute a verdict.
The real reasoning / goal / tool-loop verdicts come from the steerer's own
background paths. (`LoopingReasoningJudge`, `RefusalJudge`, `ToolErrorJudge`,
`StopReasonJudge` DO compute a verdict in `evaluate` because their
underlying detectors are cheap/synchronous; but they are still in the
skip-list, so on the reasoning path their real signal rides the legacy
detector emission, not `evaluate_judges`.)

`default_judges(disable=[BuiltinJudge.TOOL_ERROR])` is the typed way to keep
the default set but drop a subset (e.g. an agent that makes no tool calls
dropping `tool_error`). Unknown entries are ignored (forward-compatible).
`goldfive.wrap` installs `default_judges()` when the caller does not supply
an explicit `judges=` list.

### 12.4 What custom judges can and cannot do today — honest limits

**Can:** a custom judge (any `name` not in the skip-list) runs on every
reasoning observation with `reasoning_text`, `plan`, `transcript`,
`session_state`, `current_task_id`, `current_agent_id`. It can return any
of the four verdict flavours; a drift-flavoured verdict flows through the
full refine machinery.

**Cannot (yet):** the `JudgeContext.extras` dict is **not fed** on the
auto-wired reasoning-observation path. `_dispatch_custom_judges` builds the
context with `reasoning_text`/`plan`/`transcript`/`session_state`/
`current_task_id`/`current_agent_id` and **no `extras`**:

```python
# _dispatch_custom_judges
ctx = JudgeContext(
    reasoning_text=text,
    plan=getattr(session, "plan", None),
    transcript=tuple(getattr(session, "reasoning_history", []) or ()),
    session_state=session,
    current_task_id=str(getattr(session, "current_task_id", "") or ""),
    current_agent_id=agent_name,
)   # note: no extras=
```

So the two built-in shims that *read* `extras` — `ToolErrorJudge`
(`ctx.extras["tool_event"]`) and `StopReasonJudge`
(`ctx.extras["stop_reason"]`) — get nothing when invoked as *custom*
judges through this path; they only work through the steerer's own
tool-observation / stop-reason plumbing that populates those signals
directly. If you write a custom judge that needs tool-event or stop-reason
context, know that the reasoning-observation path will hand you an empty
`extras`. **Do not document `extras` as a supported input for custom
reasoning judges — it is unfed today.** (Feeding it uniformly is plausible
future work; it is not on main.)

Custom judges are dispatched fire-and-forget (tracked on
`_background_drifts`, drained by `shutdown`), so a slow custom judge cannot
serialize the model-response callback, and the 30s per-judge timeout bounds
a hung one.

---

## 13. Deferred and protected — what NOT to build or delete here

Consistent with the program's DEFERRED / PROTECTED lists:

**Deferred (present as future work, do not implement on main):**

- **Judge windowing / cadence expansion** — feeding the judge a sliding
  window of history rather than the 1500-char single block, or running it
  more often. **Blocked on a judge regression harness.** The truncation
  caps in §2.4 are real recall limits until then; say so honestly rather
  than quietly widening a cap and calling it a fix.
- **Judge-facade dispatch authority** — letting a judge directly drive
  cancel/refine instead of routing through `handle_drift`. Not on main.
- **Twin-refine-pipeline / evidence-ledger** replacement of the stacked
  `handle_drift` suppression gates — blocked on the agency-preservation
  branch-merge decision. The two freshness gates in §8.4 are the current
  design; do not replace them with an evidence ledger here.
- **Checkpoint-rollback / tool-gating hold / fork-and-judge** (Stage-4,
  bench-gated).

**Protected (never delete/"fix" without explicit human sign-off):**

- `LOOPING_TOOL_CALL` enum/ladder/promotion/planner surfaces (#204/#206)
  and `LOOPING_REASONING` NUDGE-first CRITICAL routing — the `looping_tool`
  / `looping_reasoning` built-in judge shims are the operator-visible tip
  of that machinery.
- `PLAN_DIVERGENCE` machinery (#252-disabled but KEEP).
- `reconciler.get_missed_tasks` (#163).

The **agency-preservation branch** (#453-#474) holds unmerged Stages 1-3
behind default-OFF flags. Main-side judge code must **not** copy from it,
and doc text must not claim its features exist on main.

---

## Common mistakes

Each row is a concrete wrong edit a weaker model would plausibly make, and
the correct alternative.

### Changing a judge prompt without regression evidence

**Wrong:** "Tune" `REASONING_DRIFT_USER_PROMPT_TEMPLATE` or
`GOAL_DRIFT_USER_PROMPT_TEMPLATE` (reword the GUIDANCE, add a "be strict"
line, change the JSON shape spec) and rely on `pytest` passing. The unit
tests assert on the *parse* logic and prompt *shape*, not judge *quality*
— they will pass while your reword silently makes the judge fire on every
turn or never fire.

**Right:** There is **no automated judge-quality regression harness** (it
is DEFERRED — §13). So:
1. Keep the change behind the existing override seam — operators can pass
   `system_prompt=` / `user_prompt_template=`; prefer that over editing the
   module constant.
2. If you must edit the constant, **capture and manually evaluate**: run
   the judge against a saved corpus of reasoning blocks (drive a real
   session with a sink capturing `ReasoningJudgeInvoked`, or feed known
   blocks through `classify_reasoning_drift_with_focus` with a scripted
   `call_llm`), and eyeball the verdict distribution before/after.
3. Update the prompt-shape tests (`tests/test_reasoning_judge.py`,
   `tests/test_reasoning_judge_classification_table.py`) to match, and
   state in your PR that quality was manually evaluated because no harness
   exists.

### Adding a judge call outside the semaphore / context managers

**Wrong:** Add a new `await call_llm(system, user, model)` somewhere in
`drift_observer.py` or a new detector, directly, to "quickly ask the LLM".

**Right:** Every judge dispatch MUST (a) be scheduled through the
background/semaphore path if it is per-reasoning (`_run_judge_background`),
and (b) wrap the call in `call_llm_budget(...)`,
`call_llm_thinking_disabled()`, and `llm_call_diagnostics() as diag`, and
(c) wrap all of that in `goldfive_llm_span(...)`. A bare `await call_llm`
gets no token cap (→ 9-minute calls), keeps thinking on (→ empty `raw=''`),
records no diagnostics (→ unparseable-vs-all-thought indistinguishable),
and never appears as a span. Copy the pattern from
`classify_reasoning_drift_with_focus` verbatim.

### Reading diagnostics via `getattr` on the callable

**Wrong:** `thought_count = getattr(call_llm, "last_thought_count", 0)`.
This is the **deleted** #491 pattern — closure-attribute smuggling that was
last-writer-wins under concurrent judges.

**Right:** `with llm_call_diagnostics() as diag: raw = await call_llm(...)`
then read `diag.thought_count` / `diag.answer_count`. The object is
per-call (ContextVar-bound) and safe under concurrency. The default
builders call `record_llm_call_diagnostics(...)` internally; you read the
yielded object.

### "Fixing" a quiet-fail by raising or emitting a default drift

**Wrong:** In the parse block, add `else: raise ValueError("bad verdict")`
or `else: drift = DriftEvent(kind=OFF_TOPIC, severity=WARNING, ...)` when
`classification_parsed` is empty.

**Right:** Empty classification is the **sentinel** (INVARIANT 3). Leave
`drift = None`, log at DEBUG, and let the verdict be empty. A malformed
severity is the only case that defaults (to **INFO**, not WARNING, and only
when a real deviation classification was present — §6).

### Treating `justified_deviation` with no provenance as a fire

**Wrong:** Emit a `JUSTIFIED_DEVIATION` drift whenever
`classification == "justified_deviation"`, trusting the model.

**Right:** `justified_deviation` **requires** a value in
`_VALID_PROVENANCES`. Without one, **demote** to `erroneous_deviation`
(→ `OFF_TOPIC`), log the demotion at INFO with the original raw value
(§3.3). The demotion is the anti-hallucination guard: a model claiming
"justified" but unable to name a provenance is treated as an unprovoked
deviation.

### Adding a regex/keyword shortcut to the classification

**Wrong:** "The LLM is slow; let me pre-filter obvious on-task blocks with
a keyword check" or "detect refusals with a `re.search`". This
reintroduces exactly what #166/#167 retired.

**Right:** No regex/keyword NL classification (INVARIANT 2). If cost is the
concern, tune the rate-limit (`reasoning_drift_rate_limit`) or the
concurrency cap (`max_concurrent_judges`), or route judges to a cheap
dedicated endpoint via `JudgeConfig`. Frozenset membership on the judge's
**own structured enum output** is fine; substring/regex on the agent's
**natural-language reasoning** is not.

### Broadening the Qwen thinking-disable hacks to all models

**Wrong:** Move `enable_thinking`/`/no_think` out of
`THINKING_DISABLE_CAPABILITIES` and apply them to every OpenAI-compatible
model.

**Right:** Those are Qwen/litellm-family conventions (#491 narrowed them
there deliberately). Other vendors ignore or choke on them. To support a
new vendor's convention, add a row to `THINKING_DISABLE_CAPABILITIES` keyed
by a model-name substring — do not apply one vendor's hack universally.

### Keying the rate-limit or a gate on a churning id

**Wrong:** Key `_reasoning_judge_counters` on `invocation_id`, or key the
addressed-watermark gate on `drift.id`.

**Right:** Rate-limit is keyed by `(agent_name, task_id)` — stable across
an invocation (INVARIANT 6). The watermark gate is keyed by
`(kind, current_task_id)`. Both survive the LLM-minted-id churn that would
otherwise open a fresh slot per observation and never engage the gate.

### Reading the plan revision after the `await`

**Wrong:** Move `observed_revision_index = int(plan.revision_index)` to
after `raw = await call_llm(...)` "so it's fresher".

**Right:** Capture it **before** the call (§11). The freshness gate needs
the revision the judge *observed at call-start*; the post-`await` value has
already moved on and would make every verdict look fresh, defeating the
stale-drop.

### Populating `JudgeContext.extras` and assuming custom judges see it

**Wrong:** Write a custom judge that reads `ctx.extras["tool_event"]` and
expect it populated on the reasoning-observation path.

**Right:** `extras` is **unfed** on `_dispatch_custom_judges` today
(§12.4). Base a custom reasoning judge on `reasoning_text` / `transcript` /
`session_state`. If you genuinely need tool-event context, that is a
plumbing change (feed `extras` at the dispatch site) plus tests — not an
assumption you can make in the judge body.

### Deleting a "no-op" built-in judge shim

**Wrong:** Delete `ReasoningDriftJudge`/`GoalDriftJudge`/`LoopingToolJudge`
because their `evaluate` returns an empty verdict and "does nothing".

**Right:** They are the opt-in/opt-out tokens (`default_judges`,
`disable_judges`) and the source of `BUILTIN_JUDGE_NAMES`, which drives the
double-fire-prevention skip-list. Deleting one either double-fires its
signal (if removed from the skip-list) or removes an operator's ability to
disable it. `looping_tool` / `looping_reasoning` are also on the PROTECTED
list.

### Awaiting the goal-drift or reasoning judge inline

**Wrong:** "Simplify" `_spawn_goal_drift_judge_background` or
`_run_judge_background` by awaiting the classifier directly from the
`mark_task_*` / model-response callback.

**Right:** Both judges are fire-and-forget on purpose. The reasoning judge
inline would serialize every subsequent tool call behind a minute-long
round-trip; the goal judge inline dies to a sibling `CancelledError` (the
v22 regression, §9.2). Keep them detached on `_background_judges` and
drained by `shutdown` / `drain_session_background_tasks`.

---

## Verification checklist

Run after touching anything in this chapter's surface. Commands assume the
repo root and the dev+adk extras installed
(`uv sync --extra dev --extra adk`).

### 1. Targeted judge tests (fast, run these first)

```bash
uv run pytest -q \
  tests/test_reasoning_judge.py \
  tests/test_reasoning_judge_classification_table.py \
  tests/test_reasoning_judge_verdict_scaffolding.py \
  tests/test_reasoning_judge_emission.py \
  tests/test_reasoning_judge_agent_tree.py \
  tests/test_reasoning_judge_covers_delegated_agents.py \
  tests/test_judge_uses_real_agent_id.py \
  tests/test_goal_drift_classifier.py \
  tests/test_pluggable_judges.py
```

### 2. Scheduling, budget, and diagnostics

```bash
uv run pytest -q \
  tests/test_judge_scheduling_guards.py \
  tests/test_judge_token_caps.py \
  tests/test_judge_thinking_disabled.py \
  tests/test_judge_empty_response_no_retry.py \
  tests/test_judge_task_lifetime.py \
  tests/test_judge_lifetime_v22_regression.py \
  tests/test_call_llm_diagnostic.py \
  tests/test_llm_call_budget.py \
  tests/test_llm_call_budget_integration.py \
  tests/test_reasoning_mode_dispatch.py
```

### 3. Warnings, fallbacks, wiring

```bash
uv run pytest -q \
  tests/test_reasoning_channel_disarm_warning.py \
  tests/test_reasoning_content_fallback.py \
  tests/test_llm_call_timeout_watcher.py \
  tests/test_llm_call_timeout_watcher_sticky.py \
  tests/test_wrap_goal_drift_wiring.py \
  tests/test_wrap_judge_only.py \
  tests/test_task_binding_follows_revision.py
```

### 4. Full suite + lint (always before committing)

```bash
uv run pytest -q          # expect ~2912 passed / ~61 skipped, ~30s
ruff check .              # MUST stay clean
```

Do **not** run `ruff format` on the repo — it is not format-clean and a
mass reformat is out of scope (see `15-testing-guide.md`).

### 5. Grep checks after specific edits

- **After adding a template placeholder** — confirm the `.format(...)` call
  has the matching key:
  ```bash
  grep -n "template.format(" goldfive/drift/reasoning_judge.py
  grep -n "REASONING_DRIFT_USER_PROMPT_TEMPLATE" goldfive/drift/reasoning_judge.py
  ```
- **After touching classification/provenance enums** — confirm the
  frozensets and the parse branches agree:
  ```bash
  grep -n "_VALID_CLASSIFICATIONS\|_VALID_PROVENANCES\|_SEVERITY_MAP" goldfive/drift/reasoning_judge.py
  ```
- **After adding a judge call site** — confirm it is wrapped, not bare:
  ```bash
  grep -n "await call_llm(" goldfive/drift/reasoning_judge.py goldfive/drift/goals.py goldfive/drift_observer.py
  grep -n "call_llm_budget\|call_llm_thinking_disabled\|llm_call_diagnostics" goldfive/drift/reasoning_judge.py goldfive/drift/goals.py
  ```
  Every `await call_llm(` here should sit inside the three managers.
- **After touching diagnostics** — confirm no closure-attribute pattern
  crept back:
  ```bash
  grep -rn "last_thought_count\|getattr(call_llm" goldfive/    # expect no hits
  ```
- **After touching the wire fields** — confirm proto and emitter agree
  (fields 11-15):
  ```bash
  grep -n "classification\|focused_task_id\|focus_confidence\|stated_intent\|provenance" proto/goldfive/v1/events.proto
  grep -n "payload\." goldfive/drift/reasoning_judge.py   # _emit_judge_invoked
  ```
  If you changed the proto, regenerate with the proto extra and rerun CI
  (`lint-and-test` on 3.11/3.12 with dev+adk+proto).
- **After touching the semaphore/coalescing** — confirm the task-name
  convention survives (drain filters on it):
  ```bash
  grep -n "goldfive-reasoning-judge:\|goldfive-goal-drift-judge:\|goldfive-custom-judge:" goldfive/drift_observer.py
  ```

### 6. When code and docs disagree

The **code on main is ground truth**. `docs/design/*.md` and the
`.agents/*.md` skills are sources of *intent*, but where a design doc
describes judge behavior that the code does not implement (e.g. windowing),
the code wins — treat the doc as a statement of future work and say so in
your PR. #492 was a design-doc accuracy sweep; if you find a fresh
discrepancy, that is a doc bug, not a license to change the code to match
the doc.
