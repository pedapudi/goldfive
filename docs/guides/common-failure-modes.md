# Common failure modes

Catalog of the failure shapes goldfive has observed in the wild,
with the signature on the event stream, the root cause, and the
recovery path. Pair with
[troubleshooting.md](troubleshooting.md) (install + setup
problems) and [insight-from-logs.md](insight-from-logs.md) (how to
read the stream itself). For the "it wasn't on this page"
catch-all, reach for
[.agents/debug-goldfive.md](../../.agents/debug-goldfive.md).

## 1. Filler loop

The canonical goldfive failure: the agent loops making no forward
progress, and the guards — which exist — fail to stop it.
Pre-#98 / #108 / #109 hardening, this class of bug could ride all
the way to the framework's own ceiling (ADK 500 calls, Claude
max_turns) before the run aborted.

**Signature.** One of:

- `outcome.success=False, reason="exhausted max_task_invocations=N with pending task <id>"`
  with `id` being the same task every time.
- `outcome.success=False, reason="adapter.invoke raised ... max_turns_exceeded"`
  or similar framework-level abort.
- `sink.events` shows `DRIFT_KIND_LOOPING_TOOL_CALL` (post-guard)
  or a dominating count of `report_task_*` calls on one task
  (pre-guard).
- Task status is `COMPLETED` on the session plan, but reporting
  tool calls keep arriving for the same `task_id` (returned as
  `task_already_terminal` acks post-#98).

**Root causes (by likelihood).**

1. A new adapter bypassed `invoke_tool`. See
   [.agents/how-to-debug-a-filler-loop.md](../../.agents/how-to-debug-a-filler-loop.md).
   This was the #108 root cause in the ADK adapter.
2. A custom planner produced a task the agent can't actually
   satisfy (the agent reports "done" but the planner doesn't
   accept the completion, refine spawns a retry, loop).
3. The agent's instruction prompt is wrong — it thinks it's
   driving the whole workflow (see §Agent tree misdesign below).

**Recovery path.**

- Set `SequentialExecutor(max_task_invocations=<finite>)` as a
  belt-and-suspenders ceiling while investigating.
- Run [.agents/how-to-debug-a-filler-loop.md](../../.agents/how-to-debug-a-filler-loop.md).
- If the guard was bypassed, fix the adapter. If the agent is
  mis-prompted, fix the instruction. Do NOT add another cap.

## 2. Refine failure (truncated JSON, LLM unavailable)

The planner LLM returns something `LLMPlanner` can't parse. Three
common shapes:

- JSON that got truncated at the token limit — missing closing
  `}` or `]`.
- An empty string or a whitespace-only response.
- Valid JSON but in the wrong shape (no `tasks` field, or tasks
  missing required keys).

**Signature.**

- `DriftDetected` with detail starting `refine failed` (or
  `_refine_user_steer: empty/non-string`).
- Logger `goldfive.planner` DEBUG line with `failed to parse LLM
  output`.
- `session.refine_failure_counts` incrementing for
  `(drift.kind, current_task_id)` across ticks.
- Once the counter crosses
  `DefaultSteerer.REFINE_FAILURE_THRESHOLD` (default `2`),
  `DriftDetected{kind=REPEATED_FAILURE, severity=CRITICAL}` fires
  and the task is marked FAILED.

**Root causes.**

- Planner LLM's `max_tokens` too small for the plan size.
- Planner LLM provider down / rate-limited / returning 500s.
- Prompt template is wrong (e.g. `{tasks}` literal in output
  instead of a JSON array — shows up with custom planners).

**Recovery path.**

- Widen `max_tokens` on your planner LLM.
- Implement retries / backoff inside your `call_llm` callable
  (not at the goldfive level — the steerer's refine-failure
  counter is the coarse backoff).
- Log the raw `call_llm` response before returning so you can
  see what was actually truncated. See
  [insight-from-logs.md §Capturing planner reasoning](insight-from-logs.md).

## 3. Orphaned PENDINGs at run end

A run ends with `success=False,
reason="orphaned pending tasks after run: <ids>"`. Pre-#103 this
could also produce `success=True` with PENDING tasks still on
the plan — #103 and the reachability audit (§6.4) made that
impossible.

**Signature.**

- `outcome.success=False, reason="orphaned pending tasks after run: t4, t5, t6"`.
- The last event before `RunAborted` is a CRITICAL
  `DriftDetected{kind=PLAN_DIVERGENCE}` emitted by the
  reachability audit.
- Every orphan task is PENDING on the final plan with a
  predecessor that is CANCELLED or FAILED.

**Root causes.**

- `planner.refine` returned `None` after a USER_STEER or
  unrecoverable drift, but the cascade didn't fire — pre-#103 bug,
  should not recur.
- A custom steerer cancelled a task without calling
  `cascade_cancel_downstream` — the primitive is the source of
  truth; any direct `mark_task_cancelled` that skips the
  downstream walk regresses §6.3.

**Recovery path.**

- Confirm you're on goldfive ≥ #103. If so, file a bug — this
  shouldn't happen on current `main`.
- If you have a custom Steerer subclass, audit it for any
  `mark_task_cancelled` path that doesn't delegate to
  `cascade_cancel_downstream`.

## 4. Agent tree misdesign (coordinator-drives-whole-workflow)

Not a framework bug but easy to mistake for one: the agent's
prompt tells it to run the whole workflow in one turn, instead of
handling "just the task the orchestrator routed me". The
`examples/adk_presentation/` example hit this before #111 — a
hand-rolled coordinator + subagent tree where the coordinator's
instructions made it `transfer_to_agent` sub-agents in sequence
regardless of goldfive's task dispatch. Goldfive then marked the
task complete while the coordinator was still narrating, and the
coordinator kept calling `require_confirmation=True` tools that
silently gated on a UI that wasn't watching.

**Signature.**

- Multiple agents show up in `available_agents` but every
  `InvocationResult` originates from one.
- Plan generated by the LLM planner has one task per
  sub-workflow, but the agent's output never acknowledges the
  individual task ids — it just rolls through.
- Reasoning content (if extracted) mentions the whole workflow
  not the current task.
- `report_task_completed` is called with arguments derived from
  the *plan summary*, not the *current task id* — the agent
  doesn't know which task it's on.

**Root causes.**

- The agent's instruction says "execute the plan" rather than
  "execute the single task the orchestrator gave you."
- `require_confirmation=True` on sub-agent tools, combined with
  no UI watching approvals — the tool silently blocks, the
  framework reports `max_turns_exceeded`.
- A coordinator that uses `transfer_to_agent` as its primary
  control-flow mechanism, treating sub-agents as independent
  processes rather than goldfive-managed workers.

**Recovery path.**

- Simplify to one agent + `goldfive.wrap(agent)` first. If the
  run now succeeds, the agent tree was the problem, not
  goldfive. See `examples/adk_presentation/agent.py` for the
  minimal shape post-#111.
- If you need multiple agents, have each one handle only its
  assigned task (`task.assignee_agent_id`) and return. Let
  goldfive do the routing. The harmonograf
  `tests/reference_agents/presentation_agent` is a worked
  example with a real multi-agent tree that works with goldfive.
- Rewrite the instruction to explicitly scope to the current
  task: "Each message you receive is a single task; complete
  just that task and stop."

## 5. `require_confirmation` silent gating (pre-#111 behaviour)

Specific sub-case of §4 but distinctive enough to call out. ADK
sub-agents marked `require_confirmation=True` create a tool-call
that blocks on an APPROVE / REJECT decision. Pre-#83 goldfive
didn't have an APPROVE / REJECT control channel, so the tool
call hung forever; #83 added the bridge; #111 dropped the
unneeded `require_confirmation=True` from the example tree.

**Signature.**

- Agent "goes silent" — no new events after the first
  `TaskStarted`.
- `session.pending_approvals` has an entry keyed by an
  `adk-<uuid>` (ADK Flow B). Non-empty after the run ends is the
  red flag.
- After `max_task_invocations` trips, the `RunAborted.reason`
  blames the task that was waiting on approval, not the approval
  itself — the failure mode looks like a stuck task.

**Recovery path.**

- Drop `require_confirmation=True` from any ADK tool that doesn't
  actually need human-in-the-loop approval. This was the #111 fix.
- If you do need approval, wire up the control channel
  (`Runner(control=...)`) and route APPROVE / REJECT messages.
  See [../design/CONTROL.md §Flow B](../design/CONTROL.md) and
  [../design/APPROVAL.md](../design/APPROVAL.md).
- Verify harmonograf `observe()` is attached if you expect the UI
  to drive approvals — it enables HUMAN_IN_LOOP by default since
  harmonograf #44.

## 6. Cascade ran but refine didn't produce a follow-up plan

Not a failure per se — just "incomplete but not broken". A
USER_STEER arrives, the current task is CANCELLED, the cascade
cancels downstream PENDINGs, and then `planner.refine` returns
`None`. No new work is installed. The run ends cleanly with
`success=False`.

**Signature.**

- `DriftDetected{kind=USER_STEER, severity=WARNING}`.
- `TaskCancelled` for the current task, then a flurry of
  `TaskCancelled` events with `reason="cascade from <task_id>"`.
- No `PlanRevised` follows.
- `RunAborted` with a reason like "goal '<...>' unmet" or
  "planner declined refine".

**Root causes.**

- The steer message was ambiguous and the planner genuinely
  couldn't produce a coherent follow-up.
- The planner's prompt template doesn't know how to handle
  USER_STEER (most `LLMPlanner` configurations do; custom
  planners may not).

**Recovery path.**

- This is arguably correct behaviour — the planner decided the
  steer was unrecoverable. Treat the run as failed and start a
  new one with the steer text folded into the initial input.
- If you want the planner to always produce something, customise
  your planner's refine logic to never return `None` (return a
  trivial one-task plan as a fallback).

## 7. Drift kinds firing at unexpected severity

`INTENT_DIVERGENCE` fires at graduated severity since #114 —
INFO / WARNING / CRITICAL based on cosine similarity. If you
were relying on "INTENT_DIVERGENCE always means refine", expect
more INFO-severity signals that don't trigger refine.

**Signature.**

- `DriftDetected{kind=INTENT_DIVERGENCE, severity=INFO}` — no
  refine follows. This is expected and correct.
- `DriftDetected{kind=INTENT_DIVERGENCE, severity=CRITICAL}` —
  treated as unrecoverable; cascade fires and the run aborts.
- `UNCERTAIN_PROGRESS` (INFO) appearing mid-run — this is the
  opt-in reflective self-progress check from #112 emitting.
  Informational, no refine.
- `SELF_REPORTED_STUCK` (WARNING) — refine triggered. Expect a
  `PlanRevised` follow-up.

**Root causes.**

- Reading the `kind` without the `severity` — now the wrong
  abstraction.
- Relying on pre-#114 semantics where INTENT_DIVERGENCE was a
  single severity.

**Recovery path.**

- Filter by `severity >= WARNING` if you only care about the
  refine-triggering band.
- Update any custom `should_refine(drift)` override to inspect
  severity, not kind.

## 8. Deprecation warning: `max_plan_reinvocations`

`DeprecationWarning: SequentialExecutor(max_plan_reinvocations=...)
is deprecated; use max_task_invocations=... instead.`

**Root cause.** #115 renamed the parameter. The old kwarg is
still accepted for one release with a `DeprecationWarning`.

**Recovery.** Rename. The default changed from finite (32) to
`None` (unbounded) in the same PR — if you were relying on the
old default as a safety cap, set `max_task_invocations=32`
explicitly. See
[../design/RATIONALE.md](../design/RATIONALE.md) for why the
default changed.

## Related

- [troubleshooting.md](troubleshooting.md) — install + setup problems (distinct from these run-time failure modes).
- [insight-from-logs.md](insight-from-logs.md) — how to read the event stream to identify which failure mode you're in.
- [telemetry-with-harmonograf.md](telemetry-with-harmonograf.md) — same diagnostics via the UI.
- [../../.agents/debug-goldfive.md](../../.agents/debug-goldfive.md) — the triage tree.
- [../../.agents/how-to-debug-a-filler-loop.md](../../.agents/how-to-debug-a-filler-loop.md) — deep dive on failure mode 1.
- [../design/PLAN-LIFECYCLE.md](../design/PLAN-LIFECYCLE.md) — run termination predicate and cascade semantics.
