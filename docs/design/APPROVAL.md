# Human-in-the-loop approval & escalation

Three scenarios require a human action before the run proceeds. Two of
them are **approval flows** (A / B below) — the agent explicitly waits
for yes/no — and terminate in `ControlKind.APPROVE` /
`ControlKind.REJECT`. The third is **escalation** — goldfive detects
unrecoverable drift and pauses until the operator steers or resumes,
terminating in `HUMAN_INTERVENTION_REQUIRED` drift (Level 4 of the
intervention ladder).

All three surface through proto events the harmonograf UI (or any
sink) can render. Approvals use `ApprovalRequested` / `ApprovalGranted`
/ `ApprovalRejected`; escalation uses `DriftDetected` with kind
`HUMAN_INTERVENTION_REQUIRED` plus the session's
`GOLDFIVE_PAUSE_ESCALATE` channel dispatch (Phase 2 of #246 replaced the deleted `paused_for_human_intervention` flag).

## Flow A — Task-level approval (framework-agnostic)

The agent explicitly blocks a task pending human yes/no. Works for any adapter
(CallableAdapter, ADK, Claude SDK): the agent calls the 8th canonical reporting
tool, `report_awaiting_approval`, and its handler `await`s on a per-task
`asyncio.Event` until the control dispatcher resolves it.

```
agent: call report_awaiting_approval(task_id="t7", prompt="ok to spend $500?")
  ↓
handler:
    steerer.mark_task_blocked(task_id, blocker="awaiting_approval", needed=prompt)
    session.pending_approvals[task_id] = asyncio.Event()
    session.pending_approvals_meta[task_id] = {"kind": "task", "prompt": ...}
    emit(ApprovalRequested{target_id=task_id, kind="task", prompt=...})
    await session.pending_approvals[task_id].wait()
    decision = session.pending_approvals_meta[task_id]["decision"]
    return {"decision": decision, "detail": ...}
  ↓
UI sees ApprovalRequested → renders [Approve] [Reject]
  ↓
human clicks → harmonograf ControlEvent(APPROVE|REJECT, payload={"target_id": "t7"})
  ↓
bridge → ControlChannel → executor drain_controls → dispatch_control
  ↓
dispatch_control looks up session.pending_approvals["t7"]:
    on APPROVE: stores decision="approve", sets event, emits ApprovalGranted.
    on REJECT:  stores decision="reject",  sets event, emits ApprovalRejected.
  ↓
handler resumes, returns the tool-call ack to the agent.
  ↓
agent resumes the task. It may call report_task_completed (on approve) or
report_task_failed (on reject) — goldfive does not force either transition;
the agent decides based on the decision it got back from the tool.
```

Notes:

- The task is transitioned to `BLOCKED` by the handler *before* it awaits, so
  observers and UIs have a concrete proto status to render against.
- The `ApprovalGranted` / `ApprovalRejected` events are a separate channel from
  the task state machine — they are *resolution* events, not transitions. A
  rejected approval does not automatically fail the task; the agent chooses.
- A `timeout_ms` parameter on `report_awaiting_approval` is accepted but
  currently implemented as a pure `asyncio.wait_for`. On timeout the handler
  returns `{"decision": "timeout", "detail": ...}` and leaves the task blocked.

## Flow B — ADK tool confirmation (ADK-specific)

ADK has its own `tool.require_confirmation` flag. When a `FunctionTool` is
constructed with `require_confirmation=True`, ADK's native behavior is to
short-circuit tool execution, yield an `adk_request_confirmation` function call
back to the model, and wait for a `FunctionResponse` carrying a
`ToolConfirmation{confirmed=true}` before running the tool body.

The goldfive ADKAdapter plugin intercepts that mechanism in
`before_tool_callback` so the same UI works for both flows:

```
ADK: model wants to call T (require_confirmation=True)
  ↓
goldfive _GoldfiveADKPlugin.before_tool_callback:
    tool_call_id = tool_context.function_call_id (ADK generates "adk-<uuid>")
    session.pending_approvals[tool_call_id] = asyncio.Event()
    session.pending_approvals_meta[tool_call_id] = {
        "kind": "tool", "tool_name": tool.name, "args": tool_args,
        "task_id": session.current_task_id,
    }
    emit(ApprovalRequested{target_id=tool_call_id, kind="tool", prompt=...,
                            metadata={"tool_name": ..., "args_json": ...}})
    await session.pending_approvals[tool_call_id].wait()
    decision = session.pending_approvals_meta[tool_call_id]["decision"]
    if decision == "reject":
        emit(ApprovalRejected{target_id=tool_call_id, ...})
        return {"skipped": True, "reason": "user_rejected"}
    emit(ApprovalGranted{target_id=tool_call_id, ...})
    return None  # fall through; ADK runs the tool
```

The return shape is the one ADK uses everywhere else in its plugin API:
- Returning a non-None dict tells ADK to skip the tool and use the dict as the
  response the model sees (`plugin_manager.run_before_tool_callback` line 512+).
- Returning None tells ADK to proceed with the original args.

We bypass ADK's native confirmation flow — the UI we target is goldfive's
approval event, not ADK's `adk_request_confirmation` function call. The model
never sees the interception; from its perspective the tool just took a while.

## ADK API reference (findings — April 2026, `google-adk 0.1.x`)

Researched against `third_party/adk-python/src/google/adk/`.

### Declaring "needs confirmation" on a tool

`FunctionTool.__init__` accepts a keyword arg `require_confirmation`:

```python
# src/google/adk/tools/function_tool.py:46-87
def __init__(
    self,
    func: Callable[..., Any],
    *,
    require_confirmation: Union[bool, Callable[..., bool]] = False,
):
```

Type: `bool | Callable[..., bool]`. Stored on the instance as
`self._require_confirmation`. A callable receives the tool args and returns a
bool — so individual calls can opt in or out dynamically.

Check site: `FunctionTool.run_async` (function_tool.py:191-220). If the tool is
flagged but `tool_context.tool_confirmation.confirmed` is not yet set, ADK
returns a structured error + registers a confirmation request on
`EventActions.requested_tool_confirmations` (keyed by `function_call_id`).

### Plugin callback signature + return contract

`BasePlugin.before_tool_callback` (plugins/base_plugin.py:297-319):

```python
async def before_tool_callback(
    self,
    *,
    tool: BaseTool,
    tool_args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict]: ...
```

Return contract:
- **None** → proceed normally; ADK runs `tool.run_async(args=tool_args, tool_context=...)`.
- **non-None dict** → *skip* the tool entirely. The dict becomes the tool's
  response as the model sees it. No exception, no confirmation loop, no
  double-execution.

You may *mutate* `tool_args` in-place before returning None to transform inputs.

The callback is awaited before ADK hits its own `require_confirmation` check,
so intercepting here cleanly bypasses the native flow.

### Correlation id

ADK generates the tool's function-call id in
`flows/llm_flows/functions.py:186-187`:

```python
def generate_client_function_call_id() -> str:
    return f"adk-{uuid}"
```

It is exposed on `tool_context.function_call_id`. We use this id as
`target_id` in the `ApprovalRequested` event — it's stable across the
lifetime of the call and unique within the ADK session.

### What `ToolContext` looks like in the callback

`ToolContext` is an alias for `Context` (tool_context.py:30), defined in
`agents/context.py:45`. It exposes:

- `function_call_id: str` — the `adk-…` id
- `tool_confirmation: Optional[ToolConfirmation]` — populated when the
  human-in-the-loop resolution has landed; read-only at runtime
- `state` — the ADK session state dict (we already use this to stash the
  `SessionContext` object that links back to the goldfive Session)

## Protos

Three new oneof variants on `Event.payload` (events.proto):

```proto
ApprovalRequested approval_requested = 23;
ApprovalGranted   approval_granted   = 24;
ApprovalRejected  approval_rejected  = 25;

message ApprovalRequested {
  string target_id = 1;        // task_id (Flow A) or tool_call_id (Flow B)
  string kind = 2;             // "task" or "tool"
  string prompt = 3;
  string task_id = 4;          // current task context (empty for root-task approvals)
  map<string,string> metadata = 5;  // tool_name, args_json, etc.
}
message ApprovalGranted  { string target_id = 1; string detail = 2; }
message ApprovalRejected { string target_id = 1; string detail = 2; }
```

## ControlKind extension

Issue #80 presumed `APPROVE` and `REJECT` were already in the enum. They are
not — this PR adds them to `goldfive.control.ControlKind`. Payload shape:

```python
ControlMessage(kind=ControlKind.APPROVE, payload={"target_id": "...", "detail": "..."})
ControlMessage(kind=ControlKind.REJECT,  payload={"target_id": "...", "detail": "..."})
```

The target_id routes to the same `session.pending_approvals` map regardless of
whether it was a task-level or tool-level request.

## Flow C — `HUMAN_INTERVENTION_REQUIRED` escalation (goldfive#142)

Unlike flows A and B — which are agent-initiated approvals — flow C is
**goldfive-initiated**. When the intervention ladder
(`DefaultSteerer._ladder_level_for`) routes a drift to Level 4
(`PAUSE_ESCALATE`), the steerer:

1. Dispatches a `ControlMessage(kind=GOLDFIVE_PAUSE_ESCALATE, ...)` on
   the bound `ControlChannel` (Phase 2 of #246 — see CANCELLATION-CONTRACT.md §4).
   The executor's pre-task loop honours `request_pause=True` from the
   dispatch outcome, blocking on the channel until a `RESUME` /
   `STEER` / `CANCEL` arrives.
2. Emits a `DriftDetected` event of kind
   `HUMAN_INTERVENTION_REQUIRED` (proto enum value 37, CRITICAL
   severity) carrying `detail="escalated from {orig_kind}: {orig_detail}"`.

Typical Level-4 triggers:

- `GOAL_DRIFT` at CRITICAL (every occurrence — the trajectory LLM
  judge says the tree is no longer advancing goals).
- `REFINE_VALIDATION_FAILED` at CRITICAL (planner exhausted its
  retry budget; steerer deliberately does NOT re-refine to avoid an
  infinite loop).
- `INTENT_DIVERGENCE` at CRITICAL (or at WARNING on repeat).
- `RUNAWAY_DELEGATION` on repeat.
- Any `CRITICAL` drift whose first-occurrence Level-3 CANCEL_REINVOKE
  didn't resolve and it repeats.

The Runner's overlay loop blocks before picking up the next step
when a `GOLDFIVE_PAUSE_ESCALATE` ControlMessage has been drained
from the channel (the dispatch outcome's `request_pause=True` flips
the executor's pre-task loop into the same blocking wait a user
`PAUSE` uses). To unpause, the operator sends one of:

- **`ControlKind.RESUME`** — clear the flag and continue (if the
  operator believes the drift was spurious).
- **`ControlKind.STEER`** — provide a new direction. The refine runs,
  the plan is revised, and the overlay restarts with the steer body.
- **`ControlKind.CANCEL`** — abort the run cleanly.

If the pause never resolves and the same `HUMAN_INTERVENTION_REQUIRED`
drift fires a second time (rare — would require the operator to
send a STEER that re-triggers the same escalation), the ladder
escalates to Level 5 (`TERMINATE`) — a run-level abort.

Sinks should render the pause as a visible state (badge + paused
Gantt) and surface the approve/reject-style buttons as Resume /
Steer / Cancel affordances. The UI path mirrors flow A's
"awaiting" card shape but with a distinct drift source and a
different control verb set.

See [DRIFT.md §"Intervention ladder"](DRIFT.md#intervention-ladder-levels-0-5)
for the full per-drift-kind level mapping.

## harmonograf follow-up (out of scope for this PR)

The harmonograf UI needs to render `ApprovalRequested` events with
approve/reject buttons and route clicks through its existing ControlRouter as
`ControlEvent(APPROVE|REJECT, payload={"target_id": ...})`. Filed separately.
