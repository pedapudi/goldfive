# Common failure modes

Catalog of the failure shapes goldfive has observed in the wild, with
the signature on the event stream, the root cause, and the recovery
path. Pair with [troubleshooting.md](troubleshooting.md) (install +
setup problems) and [insight-from-logs.md](insight-from-logs.md) (how
to read the stream itself). For the "it wasn't on this page"
catch-all, reach for
[.agents/debug-goldfive.md](../../.agents/debug-goldfive.md).

For the taxonomy of every drift kind, severity rules, and the refine
policy, see [../design/DRIFT.md](../design/DRIFT.md).

## 1. Tool-call loop — agent stuck calling the same tool

The canonical filler loop post-#181: the agent keeps calling the same
tool over and over without reaching a terminal task state. Covered
automatically by the **`ToolLoopTracker`** (auto-wired, no user
config). The detector fires at `after_tool_callback` on three
patterns:

- **Exact** — same `(tool_name, args_hash)` repeats ≥ 3 in the last 7
  calls → `LOOPING_REASONING` / WARNING.
- **Name** — same `tool_name` (any args) repeats ≥ 5 in the last 7
  with no task-state progress → `LOOPING_REASONING` / WARNING.
- **Alternating** — A,B,A,B,A pattern in the last 5 → `LOOPING_REASONING`
  / INFO (observational; does not trigger refine).

Reporting-tool calls (`report_task_*`, `report_plan_divergence`, etc.)
are excluded — they're progress signals, not work.

**Signature.**

- `DriftDetected{kind=looping_reasoning, severity=warning}` with
  `detail` starting `tool_loop_exact:`, `tool_loop_name:`, or
  `tool_loop_alternating:`.
- `raw.mode` on the drift identifies which mode fired.
- Downstream: the steerer escalates through the intervention ladder
  (Level 1 ABSORB → refine; escalates to Level 3 CANCEL_REINVOKE on
  repeat).

**Configuration.** Defaults tuned for a 10-call window. Override via
env vars:

- `GOLDFIVE_TOOL_LOOP_WINDOW` (default 10)
- `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` (default 3, work-WARNING only)
- `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` (default 5, work-WARNING only)
- `GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD` (default 5)

See `goldfive/drift/tool_loops.py` for the full contract.

**Recovery path.**

1. Read the drift detail: identifies the tool being looped.
2. If the tool is legitimate (scripted pipeline), raise the threshold
   or clear the buffer with `ToolLoopTracker.on_task_progress(...)`
   on your own progress signal.
3. If it's genuine drift, let the intervention ladder handle it —
   `planner.refine` typically produces a revised task that escapes
   the loop.

## 2. Plan divergence — agent ran an unplanned agent

The overlay reconciler observed a `before_agent_callback` for an agent
whose `name` doesn't match any PENDING plan task's `assignee_agent_id`
(after walking the parent chain for a contextual match,
goldfive#151). Emitted as `PLAN_DIVERGENCE` / INFO severity —
observational.

Separately, the three-stage `function_call` gate in
`GoldfivePlanner.process_planning_response` (goldfive#184) classifies
LLM-emitted tool calls:

| Input | Classification |
|---|---|
| `function_call` name is in the current agent's `tools` | legitimate; no drift |
| Name is a known agent in the tree registry but not in this agent's tools | cross-layer delegation attempt → `PLAN_DIVERGENCE` / WARNING |
| Name is nowhere (not a tool, not a known agent) | hallucination → `CONFABULATION_RISK` / WARNING |

`function_call` names prefixed `report_` (the reporting-tool namespace)
are always legitimate regardless of tool-list contents. Cancelled
function-call ids from `session.state['goldfive.cancelled_function_call_ids']`
are stripped before classification runs.

**Signature.**

- `DriftDetected{kind=plan_divergence, severity=warning, detail="function_call ... cross-layer"}`.
- `DriftDetected{kind=confabulation_risk, severity=warning, detail="function_call ... hallucinated"}`.
- Never blocks the call — it's a signal, not a gate. The steerer's
  ladder decides whether to escalate.

**Recovery path.**

- `PLAN_DIVERGENCE` → the refine path typically narrows the tool /
  agent scope or adjusts assignee hints.
- `CONFABULATION_RISK` → usually the prompt is wrong. Either (a)
  narrow the coordinator's instruction to only describe tools it
  actually has, or (b) add the missing tool / agent to the tree.

## 3. Refine validation failed

`LLMPlanner.refine` exhausted its retry budget — the LLM's response
couldn't be parsed or couldn't pass `Plan.validate(for_revision=True,
prior=plan)` after N attempts. The planner falls back to the prior
plan (or the deterministic fail-the-looper plan when the drift was
`LOOPING_REASONING`).

**Signature.**

- Logger `goldfive.planner` line: `LLMPlanner._refine_user_steer: attempt 2/2: <error>`.
- `DriftDetected{kind=refine_validation_failed, severity=critical}`.
- The steerer deliberately does **not** refine on this drift (infinite
  loop risk). The ladder escalates to Level 4 PAUSE_ESCALATE →
  `HUMAN_INTERVENTION_REQUIRED`.

**Root causes.**

- Planner LLM's `max_tokens` too small; output is truncated JSON.
- LLM returned a revision that drops a terminal task (§3.1 of
  PLAN-LIFECYCLE) or grafts PENDING tasks onto CANCELLED predecessors
  (reachability invariant, §7).
- Planner prompt template is wrong (custom planners).

**Recovery path.**

- Widen `max_tokens`.
- Log the raw `call_llm` response to see what the LLM produced. See
  [insight-from-logs.md](insight-from-logs.md).
- The operator resumes by steering again, cancelling, or accepting
  the fallback plan.

## 4. Runaway delegation — AgentTool cap exceeded

The coordinator's prompt describes a pipeline and its LLM keeps
delegating via `AgentTool`. The goldfive ADK plugin enforces a
per-invocation cap (default 16, configurable via
`ADKAdapter(agent_tool_cap=N)`).

**Signature.**

- `DriftDetected{kind=runaway_delegation, severity=critical}` once the
  cap trips.
- Further AgentTool spawns in the same invocation return a "skipped"
  dict.
- The current task is marked FAILED (CRITICAL drift flows through the
  planner's refine path).

**Root cause.** The coordinator's prompt describes a pipeline and the
LLM keeps re-routing. Goldfive cannot require prompt cooperation
(users bring their own trees).

**What catches it first, in order.**

1. **`ToolLoopTracker`** (§1) — catches tight AgentTool loops before
   the cap trips.
2. **Reasoning-content drift detectors.** `LOOPING_REASONING` (hash-
   or embedding-based) and `INTENT_DIVERGENCE` fire when the
   coordinator's chain-of-thought shows the pattern.
3. **Refine-driven recovery.** A WARNING-or-higher drift flows
   through the ladder into `planner.refine`, which can narrow the
   assignee hint or split into sub-tasks before the next turn.
4. **AgentTool cap.** The last-resort safety net.

**Recovery path.**

- Inspect `adapter.available_agents` after wrap: should list every
  agent in the tree. A one-entry list means the wrap target was a
  pre-built `Runner` rather than a `BaseAgent`.
- Tighten the coordinator's prompt to be task-focused (see the
  example in [adk-web-integration.md](adk-web-integration.md)).
- Raise `agent_tool_cap` only if legitimate delegation exceeds 16
  per turn.

## 5. Goal drift (opt-in)

Periodic trajectory-level check: every N agent turns, an LLM-judge
looks at the recent activity window and decides whether the tree is
advancing `session.goals` (goldfive#143). Emits `GOAL_DRIFT` /
CRITICAL when the judge concludes progress has stalled.

**Feature gate.** Opt-in via `DefaultSteerer(goal_drift_enabled=True,
goal_drift_call_llm=...)`. Operators who don't configure it never
trigger it and pay no LLM cost.

**Signature.**

- `DriftDetected{kind=goal_drift, severity=critical}`.
- Routes to Level 4 PAUSE_ESCALATE → `HUMAN_INTERVENTION_REQUIRED`.

## 6. Human intervention required

The steerer escalated a drift to Level 4. Paused the run on
`session.paused_for_human_intervention`; the executor blocks waiting
for a `CONTROL_RESUME` or `CONTROL_STEER`. Emitted for:

- Persistent refine failures.
- `GOAL_DRIFT` (CRITICAL).
- `REFINE_VALIDATION_FAILED`.
- `RUNAWAY_DELEGATION` on repeat.

**Signature.**

- `DriftDetected{kind=human_intervention_required, severity=critical}`.
- Run pauses; no new tasks start.
- Harmonograf UI's session header shows the pause; the Steer / Resume
  buttons are armed.

**Recovery path.** A user-initiated `CONTROL_RESUME` or
`CONTROL_STEER` clears the flag.

## 7. Qwen coordinator hallucinates tool success (model-specific)

Observed on Qwen3-Coder and similar weaker models under the
`presentation_agent_orchestrated` tree: the coordinator produces
fluent output claiming it wrote files via `write_webpage` but never
actually emits the tool call. The tree reports success; the
filesystem is empty.

**Not a goldfive regression.** The confabulation-risk classifier
(§2) catches the shape when the task's title/description implies
external data access (research, lookup, write, verify, …) and the
agent produced non-empty output without calling a single tool —
`CONFABULATION_RISK` / INFO. Record-only; does not trigger refine.

**Recovery path.**

- Switch to a stronger model (`USER_MODEL_NAME=openai/gpt-4o-mini`
  works; Gemini + `GOOGLE_API_KEY` works).
- Tighten the coordinator prompt to say "you MUST call
  `write_webpage` — do not claim success without the tool call."
- The INFO drift itself is informational; operators watching the UI
  can cancel on sight.

## 8. Refine cascade produced no follow-up plan

Not a failure per se — "incomplete but not broken". A USER_STEER
arrives, the current task is CANCELLED, the cascade cancels
downstream PENDINGs, and then `planner.refine` returns `None`. No
new work is installed. The run ends cleanly with `success=False`.

**Signature.**

- `DriftDetected{kind=user_steer, severity=warning}`.
- `TaskCancelled` for the current task, then a flurry of
  `TaskCancelled` events with `reason="cascade from <task_id>"`.
- No `PlanRevised` follows.
- `RunAborted` with reason like "goal '<…>' unmet" or "planner
  declined refine".

**Root causes.**

- The steer message was ambiguous and the planner genuinely couldn't
  produce a coherent follow-up.
- Custom planners whose refine logic doesn't know how to handle
  `USER_STEER` (the bundled `LLMPlanner` does).

**Recovery path.**

- Arguably correct behaviour — the planner decided the steer was
  unrecoverable. Start a new run with the steer text folded into
  the initial input.
- Customise your planner to never return `None` (return a trivial
  one-task plan as a fallback).

## 9. Drift severity not what you expected

`INTENT_DIVERGENCE` fires at graduated severity — INFO / WARNING /
CRITICAL based on cosine similarity against `session.goals` + the
current task topic. If you were relying on "INTENT_DIVERGENCE always
means refine", expect more INFO-severity signals that don't trigger
it.

**Signature.**

- `DriftDetected{kind=intent_divergence, severity=info}` — no refine
  follows. Expected.
- `DriftDetected{kind=intent_divergence, severity=critical}` —
  treated as unrecoverable; cascade fires and the run aborts.
- `LOOPING_REASONING` at INFO (the `tool_loop_alternating` variant) —
  observational, no refine.

**Recovery path.**

- Filter by `severity >= WARNING` if you only care about the
  refine-triggering band.
- Update custom `should_refine(drift)` overrides to inspect severity,
  not kind.

## 10. Session rows duplicated in harmonograf

Pre-#161 / #164: goldfive's `Session.id`, the `ADKAdapter`'s internal
`_session_id`, and adk-web's outer `ctx.session.id` were three
different UUIDs. Goldfive events carried one id; harmonograf spans
carried another; the UI showed two rows per run with the plan on one
and the execution on the other.

**Current state.** `GoldfiveADKAgent._run_async_impl` pins the outer
adk-web session id onto both the goldfive Session and the
ADKAdapter's internal state (`_outer_session_id`, `_session_id`)
before any sub-agent dispatch runs. Per-event `session_id` stamping
(goldfive#155) + `HarmonografSink` routing by it means one session
row per run.

**If you still see duplicates:** verify you're on goldfive ≥ #164
and harmonograf ≥ #85 (lazy Hello).

## Related

- [troubleshooting.md](troubleshooting.md) — install + setup problems (distinct from these run-time failure modes).
- [insight-from-logs.md](insight-from-logs.md) — how to read the event stream to identify which failure mode you're in.
- [telemetry-with-harmonograf.md](telemetry-with-harmonograf.md) — same diagnostics via the UI.
- [../../.agents/debug-goldfive.md](../../.agents/debug-goldfive.md) — the triage tree.
- [../../.agents/how-to-debug-a-filler-loop.md](../../.agents/how-to-debug-a-filler-loop.md) — deep dive on failure mode 1.
- [../design/DRIFT.md](../design/DRIFT.md) — the drift taxonomy.
- [../design/PLAN-LIFECYCLE.md](../design/PLAN-LIFECYCLE.md) — run termination predicate and cascade semantics.
