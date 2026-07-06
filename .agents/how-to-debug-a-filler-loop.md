---
name: how-to-debug-a-filler-loop
description: Diagnostic playbook for the "guards are defined but aren't firing" class of filler-loop bugs. Based on the postmortem from #98 / #108 / #109 / #116 / harmonograf #45.
applies-when: ["filler loop", "report_task_completed called repeatedly", "500-call ceiling", "guard not firing"]
---

# How to debug a filler loop

A **filler loop** is the class of bug where the agent makes no
forward progress and goldfive's guards — which exist — fail to
stop it. The five bugs the spring 2026 sprint found all fit this
mold: the guard was written correctly and had tests, but the data
path never reached it.

## Symptom signatures

Any of these is a high-confidence filler-loop signal:

- The run terminates on the adapter framework's own ceiling (ADK
  500-call, Claude max-turns) rather than on any goldfive
  guard.
- `sink.events` shows the same reporting tool called more than
  ~20 times in a single `invoke()` with the same or near-identical
  args, yet no `LOOPING_REASONING` drift fires from the
  `ToolLoopTracker`.
- `outcome.success=True` but a large fraction of tasks are still
  `PENDING` after the run.
- `outcome.success=False` with `reason="adapter.invoke raised ...
  max_turns_exceeded"` or similar framework-level error.
- The `RunAborted.reason` mentions exhausting `max_task_invocations`
  with one specific `task_id` pending every time.
- `task.status` sits at `COMPLETED` but new reporting-tool calls
  keep arriving for the same `task_id`.

## The diagnostic playbook

### Step 1 — Confirm it's structural, not symptomatic

Check which guard you expected to fire. The goldfive guards are:

| Guard | Where | Expected drift or ack |
|---|---|---|
| Schema rejection (missing / unknown `task_id`) | `_tool_invocation.invoke_tool` | `{acknowledged: False, error: "missing_task_id" \| "unknown_task_id"}` |
| Handler idempotency / invalid-transition (goldfive#201) | `goldfive.reporting._handle_task_*` | `{acknowledged: True, idempotent: True, ...}` on same-transition retry; `{acknowledged: False, error: "invalid_transition", ...}` on cross-transition |
| **`ToolLoopTracker`** (all ADK tool calls, goldfive#181 + #204) | `goldfive/drift/tool_loops.py`, fires at `after_tool_callback` | `LOOPING_REASONING` drift with `detail="tool_loop_exact:…"`/`tool_loop_name:…`/`tool_loop_alternating:…` and graduated INFO/WARNING/CRITICAL severity |
| Per-task retry-lineage cap | `SequentialExecutor._lineage_root` | task marked FAILED without invoking the adapter |
| Refine-failure threshold | `DefaultSteerer.REFINE_FAILURE_THRESHOLD` | `REPEATED_FAILURE` drift + task FAILED |
| AgentTool-per-invoke cap | `_GoldfiveADKPlugin.before_tool_callback` | `RUNAWAY_DELEGATION` drift (CRITICAL); further spawns return `{"status": "skipped"}` |

> **Retired (goldfive#206).** The per-task + session-wide
> `ToolLoopGuard` in `goldfive/adapters/_tool_loop_guard.py` used
> to sit between schema rejection and the handler. It was retired
> because it was firing CRITICAL `LOOPING_TOOL_CALL` drifts on
> benign idempotent retries (the exact=6-in-window-of-8
> behaviour pre-dated both handler-owned idempotency and the
> tracker's graduated severity).

If none of these ACK/drift shapes appear in your sink log, the
guard never ran. That is the bug — do not patch a new cap; find
why the existing guard was skipped.

### Step 1.5 — Check the `ToolLoopTracker` coverage (goldfive#181)

The `ToolLoopTracker` is auto-wired into the ADK plugin and observes
**every** tool call in the ADK tree (not just reporting tools). Since
goldfive#204 it uses graduated severity per category (meta vs work
tools) — see `goldfive/drift/tool_loops.py` for the full threshold
table. In brief:

- **Meta tools** (`report_task_*`, `report_awaiting_approval`):
  exact-match INFO at 3, WARNING at 6, CRITICAL at 10 in a window
  of 10.
- **Work tools** (everything else): exact-match INFO at 3, WARNING
  at 3, CRITICAL at 6; same-name (any args) WARNING at 5, CRITICAL
  at 7.
- **Alternating** A,B,A,B,A pattern always fires INFO only (window
  tail of 5).

Inspect its detection output with:

```python
import logging
logging.getLogger("goldfive.drift.tool_loops").setLevel(logging.DEBUG)
```

The tracker is keyed by `(invocation_id, agent_name)`, so parallel
sub-agents within one invocation are isolated. The "no task
progress" gate for the name-axis is cleared by
`ToolLoopTracker.on_task_progress(invocation_id=…, agent_name=…)` —
the plugin wires this to the `after_tool_callback` for
acknowledged-success reporting responses (goldfive#192).

Env-var overrides:

- `GOLDFIVE_TOOL_LOOP_WINDOW` (default 10)
- `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` (default 3, overrides the
  work-category WARNING tier only)
- `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` (default 5, overrides the
  work-category WARNING tier only)
- `GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD` (default 5)

If the tracker isn't firing on a run you know is looping, three
things to check:

1. The plugin may not be installed. Verify `adapter.add_plugin` /
   `_register_plugin_on_runner` ran.
2. The tool names being called may not be reaching the plugin
   (unusual — ADK routes every tool call through `before_tool_callback`).
3. The window size may be too small for the loop's phase — try
   `GOLDFIVE_TOOL_LOOP_WINDOW=15`.

### Step 2 — Instrument at the guard entry point

Add a single line at the top of `invoke_tool`:

```python
# goldfive/adapters/_tool_invocation.py
async def invoke_tool(tools, name, args, session, steerer):
    import logging; logging.getLogger("goldfive").warning(
        "invoke_tool entry: tool=%s task_id=%r session=%s",
        name, args.get("task_id"), id(session),
    )
    ...
```

Run the reproducer. If the line doesn't print for the looping
tool, `invoke_tool` is never called — the adapter is dispatching
`spec.handler` directly. This was the #108 bug.

If the line prints but the guard decisions look wrong, the
counters / keys / session are inconsistent. Proceed to step 3.

### Step 3 — Verify `invoke_tool` is actually wired

Check each adapter's tool-invocation hook:

- **ADK.** `goldfive/adapters/_adk_plugin.py::before_tool_callback`
  — the dispatch line should be
  `return await invoke_tool(self._adapter._tool_specs, name, args, session, steerer)`.
  Historically it was `return await spec.handler(args, session, steerer)`,
  which bypassed every guard. #108 fixed this.
- **Claude SDK.**
  `goldfive/adapters/claude.py::ClaudeAgentSDKAdapter._handle_tool_use`
  — look for `await invoke_tool(self._tools, name, args, ...)`.
- **Callable.** `CallableAdapter` hands the spec list to the
  callable; the callable is responsible for routing through
  `invoke_tool` if it wants guard coverage. Most don't, because
  most callables are test harnesses.
- **Custom.** Search the adapter for `spec.handler(` — every call
  is a bypass candidate.

### Step 4 — Check `ContextVar` propagation

If the adapter uses a `ContextVar` to carry per-invoke state
(session, steerer, task) into its callbacks:

- The `ContextVar` must be set in the coroutine that calls
  `adapter.invoke(...)`, not in the hook registration site.
- `asyncio.to_thread` / `loop.run_in_executor` does NOT propagate
  context vars across the thread boundary by default. Frameworks
  that dispatch callbacks through a thread pool will see `None`.
- `asyncio.gather` with `copy_context=True` (the default in
  3.11+) copies the current context into each child; mutations to
  a `ContextVar` inside a child task are invisible to the parent.

The fix, as landed in #116 for the ADK plugin: stop routing
state through `ContextVar` / `session.state` and bind the state
to the adapter/plugin instance directly. A
Python reference held by the adapter survives every SDK-internal
copy boundary.

### Step 5 — Check `session.state` copy semantics

If handoff between an adapter hook and a framework callback uses
the SDK's own session-state object (ADK's
`InvocationContext.session.state`, Claude's conversation state,
etc.), verify that the SDK does not copy state between turns.

ADK deep-copies `session.state` between every turn. Anything
written into it in turn N is **not** the same object in turn
N+1. Mutating it after turn boundaries has no effect from the
agent's perspective. The symptoms: the guard counter you see in
one callback is consistently different from the one the next
callback reads.

The reference memory note
(`~/.claude/.../memory/feedback_callback_context_handoff.md`
for the agent-local copy; on-repo analogue: the postmortem in
TASK-LIFECYCLE.md §7.5 and the rationale in ARCHITECTURE.md's
"architectural invariant" section) documents this class of bug
and the fix pattern: bind state to the adapter / plugin instance
(a Python object reference), not to the SDK's state store.

## The five bugs — what was actually broken

| PR | Symptom | Actual cause |
|---|---|---|
| #98 | Adapter kept invoking agent past `COMPLETED`; 500-call ADK ceiling | The invoke-loop didn't early-break on terminal status. Reporting-tool dispatch didn't reject calls on terminal tasks. |
| #108 | All guards "correctly defined", none firing under load | ADK plugin called `spec.handler` direct, bypassing `invoke_tool` and the four guard layers. |
| #109 | Guard fired once per task; agents that invented fresh `task_id` each call defeated it | No session-wide cap, and the per-task guard's "flag and continue" ack looked like a pass to the model, which then kept calling. |
| #116 | Post-#108 regression under ADK load; SessionContext kept vanishing | The plugin was reading its context out of `session.state`, which ADK copied between turns. Rebinding to the plugin instance fixed it. |
| harmonograf #45 | Reasoning content not appearing on spans despite #43 wiring | The span attribute name mismatch: server wrote `llm.reasoning`, client read `reasoning`. Not a goldfive bug but the same "two sides of the same wire" pattern. |

Common thread: each "fix" before the real structural fix added
another cap or threshold. The real fix was always finding the one
data path that should have been routing through the guard and
wasn't.

## Related

- **Deep reference:** why the guard exists and where it lives —
  [docs/dev-guide/07-deterministic-drift-detection.md](../docs/dev-guide/07-deterministic-drift-detection.md)
  (tool-loop detector, the #484 name-axis INFO cap, the stall
  watchdog), [09-steering-ladder-and-gates.md](../docs/dev-guide/09-steering-ladder-and-gates.md)
  (the ladder, the gates, and Appendix D "a drift fired but the agent
  kept going"), and [12-events-sinks-telemetry.md](../docs/dev-guide/12-events-sinks-telemetry.md)
  (reading the event stream / decision telemetry to see what fired).
- [docs/guides/common-failure-modes.md](../docs/guides/common-failure-modes.md) — each failure mode in detail with its recovery path.
- [docs/design/TASK-LIFECYCLE.md §5](../docs/design/TASK-LIFECYCLE.md) — the four guard layers and their ordering.
- [docs/design/TASK-LIFECYCLE.md §7.3 / §7.4](../docs/design/TASK-LIFECYCLE.md) — refine-failure threshold and ADK session heal.
- [debug-goldfive.md](debug-goldfive.md) — general debugging playbook; "structural vs symptomatic" heuristic.
- [how-to-add-a-new-adapter.md](how-to-add-a-new-adapter.md) — the `invoke_tool` wiring contract for new adapters.
