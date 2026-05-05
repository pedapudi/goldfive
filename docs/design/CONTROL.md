# Live steering and the control channel

goldfive runs are not fire-and-forget. While the executor walks the
plan, an external operator — a harmonograf UI user, a CLI, a test
harness — can reach in and redirect it: pause between tasks, cancel
the run, steer the planner into a different shape, rewind to an
earlier task, approve or reject a tool call waiting for a human.

Every one of those verbs travels through a single primitive: the
**`ControlChannel`**. This document is the canonical reference for
that primitive — what it is, what every verb means, how each one
lands in the executor, and how a UI on the other side of a network
hooks into it.

Type-system view (enum tables, bridge tables, severity ladder) is in
[VOCABULARY.md](VOCABULARY.md). Why-this-shape is in
[RATIONALE.md](RATIONALE.md). This doc focuses on the mechanics.

## 1. Why live steering exists

Three forces made live steering first-class.

**Long runs.** A slide-deck agent may take ten minutes end-to-end.
An operator watching the harmonograf Gantt often realises at minute
three that the plan is heading the wrong way. Mid-run steering
replaces remaining work without losing what's already completed.

**Human-in-the-loop approval.** Some tool calls must not land without
a human yes/no — a billing call, a `DELETE FROM`, a production deploy.
goldfive cannot reach the UI directly; the UI cannot touch goldfive-
internal state directly. The ControlChannel plus
`session.pending_approvals` is that bridge.

**Tests and CLIs.** The same primitive that drives the UI drives test
harnesses. Asserting "PAUSE emits `USER_PAUSE`" shouldn't need a UI
server. `ControlChannel` is framework-neutral; tests instantiate one
directly.

The design constraint: **no polling, anywhere**. Both sides `await`
queues (`asyncio.Queue` in-process, gRPC streams cross-process).

### 1.a Single source of truth — the wire format lives in goldfive's proto

The on-the-wire control plane (`ControlEvent`, `ControlKind`,
`ControlAckResult`, `ControlAck`, the typed payload oneof) is defined
**once**, in `proto/goldfive/v1/control.proto`. harmonograf, any other
bridge, and any future non-Python language binding imports goldfive's
proto rather than declaring its own. The bridge becomes a transport
relay — it moves `ControlEvent` bytes from the UI process into the
goldfive process and hands the already-decoded message to
`ControlChannel.send`. No enum translation. No per-kind JSON schema.

This was not always true. Harmonograf used to define a parallel
`ControlEvent` / `ControlKind` in its own proto and the
`harmonograf_client` bridge hand-maintained a `_KIND_MAP`. That drifted
twice — once after goldfive added APPROVE / REJECT (harmonograf #34) —
so in issue #88 the control types moved under `goldfive.v1`. The
Python dataclasses in `goldfive/control.py` stay as an ergonomic
in-process surface; the converters in `goldfive/conv.py`
(`to_pb_control_event` / `from_pb_control_event` / `to_pb_control_ack`
/ `from_pb_control_ack`) round-trip between them. A tripwire test
asserts the Python `ControlKind` StrEnum and the proto enum stay in
lockstep; adding a member to one without the other fails CI.

Typed payloads replace what was previously `bytes payload`. Each kind
that carries data has a dedicated message (`SteerPayload`,
`RewindPayload`, `ApprovePayload`, `RejectPayload`,
`InjectMessagePayload`) selected by a `oneof`. The dataclass keeps
`payload: dict[str, Any]` for Python-side ergonomics; the converter is
the boundary that maps `dict` → oneof branch and back.

### 1.b `SteerPayload` fields (goldfive#171)

`SteerPayload` (`proto/goldfive/v1/control.proto`) carries four
strings:

| Field | # | Meaning |
|---|---|---|
| `note` | 1 | The free-text steering instruction the operator authored. |
| `suggested_action` | 2 | Optional short verb the UI may surface as a hint (e.g. "drop research phase"). |
| `author` | 3 | Operator identity from the originating annotation. Empty when the bridge doesn't source annotations. Used for audit trails and prompt attribution (appears in the USER_STEER drift detail as `"by {author}: {note}"`). |
| `annotation_id` | 4 | Source annotation id. Used for idempotency (see §1.c) and for stamping `DriftDetected.annotation_id` on the resulting USER_STEER drift so sinks can dedup against the annotation row. |

Bridges that don't source annotations may leave `author` and
`annotation_id` empty — the in-process dedupe falls back to
`ControlEvent.id` / `ControlMessage.id`.

### 1.c STEER idempotency contract (goldfive#171)

Every STEER carries a dedupe id: `annotation_id` when the bridge
populated one, otherwise the `ControlMessage.id`. `DefaultSteerer`
records processed ids on
`session.state["goldfive.processed_steer_ids"]` (a bounded FIFO) and
drops retries before they reach `steerer._handle_drift`:

- **Retries of the same STEER** (same annotation, repeated
  `channel.send`) — dropped silently; the steerer does not cascade-
  cancel or call `planner.refine` a second time.
- **New STEER with a fresh id** — processed normally; emits
  USER_STEER drift, cancels the in-flight invocation, runs refine.
- **Bridge that doesn't source annotation_id** — falls back to the
  `ControlMessage.id`; retries at the transport layer still dedupe
  because the bridge re-uses the same message id.

Dispatcher and ack still fire on every delivery (so the bridge sees
an `AckResult.SUCCESS` back) — only the side-effects (drift + refine)
are suppressed for duplicates.

## 2. The `ControlChannel` primitive

One class, two queues, a dozen lines of interface. Lives at
`goldfive/control.py` and re-exports at the package root
(`goldfive.ControlChannel`).

```python
from goldfive import ControlChannel, ControlMessage, ControlKind, ControlAck, AckResult

channel = ControlChannel()

# Sender side (UI bridge, CLI, test)
await channel.send(ControlMessage(kind=ControlKind.PAUSE))

# Runner-internal side (executor)
msg = await channel.receive(timeout=0.1)
if msg is not None:
    await channel.ack(ControlAck(control_id=msg.id, result=AckResult.SUCCESS))

# Sender side: drain acks
async for ack in channel.acks():
    ...  # surface to the UI
```

### 2.a The shapes

```python
class ControlKind(StrEnum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    REWIND_TO = "REWIND_TO"              # payload: {"task_id": "..."}
    STEER = "STEER"                      # payload: {"note": "...", "suggested_action": "..."}
    APPROVE = "APPROVE"                  # payload: {"target_id": "...", "detail": "..."}
    REJECT = "REJECT"                    # payload: {"target_id": "...", "detail": "..."}
    STATUS_QUERY = "STATUS_QUERY"        # no payload
    INTERCEPT_TRANSFER = "INTERCEPT_TRANSFER"  # payload: {"enabled": bool}
    INJECT_MESSAGE = "INJECT_MESSAGE"    # payload: {"role": "...", "text": "..."}


class AckResult(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ControlMessage:
    kind: ControlKind
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: dict[str, Any] = field(default_factory=dict)
    issued_at_ms: int = 0


@dataclass
class ControlAck:
    control_id: str
    result: AckResult
    detail: str = ""
    acked_at_ms: int = 0
```

Every member above has a matching `CONTROL_KIND_<NAME>` value in
`proto/goldfive/v1/control.proto`; the alignment test in
`tests/test_control_proto.py` keeps the two in lockstep.

### 2.b Send / receive / ack semantics

**Bidirectional but asymmetric.** Two internal queues: `_inbox`
(external senders push, runner consumes) and `_outbox` (runner
pushes, external consumers iterate). Both sides `await`; neither
polls.

**Every message produces exactly one ack.** The executor dispatches,
publishes the ack, then applies the effect. Errors after the ack
surface as proto events, not late acks.

**IDs are UUIDs, not sequence numbers.** Senders correlate
`ControlAck.control_id` back to `ControlMessage.id`. Acks land in
send order in practice because the runner consumes the inbox
sequentially, but the contract imposes no ordering beyond that.

**Close is idempotent.** `channel.close()` marks both queues closed.
`receive()` returns `None`; `acks()` exits via an internal sentinel.
In-flight messages drain first.

### 2.c Lifecycle binding

The channel is constructed by whoever owns the external end and
passed to the Runner via `control=`:

```python
from goldfive import Runner, ControlChannel, SequentialExecutor

channel = ControlChannel()
runner = Runner(
    agent=...,
    planner=...,
    executor=SequentialExecutor(),
    control=channel,
)
```

The Runner forwards `control=channel` into `executor.run(...)`. Both
`SequentialExecutor` and `ParallelDAGExecutor` drain it between tasks
and race it against the adapter via `asyncio.wait(...,
return_when=FIRST_COMPLETED)` for mid-task cancel/steer.

## 3. End-to-end path — UI click to plan revision

Every arrow below is an `await` or a proto event on the wire;
nothing polls.

```
┌───────────────────────────────────────────────────────────────────┐
│ browser (harmonograf UI)                                           │
│   user clicks [Steer]; form = {note: "focus on slide 3", ...}     │
└──────────────────────────────────┬────────────────────────────────┘
                                   │ gRPC-Web → ControlEvent(STEER)
┌──────────────────────────────────▼────────────────────────────────┐
│ harmonograf server (Python, gRPC :7531)                            │
│   looks up the Client for this run_id; writes onto its control    │
│   stream                                                           │
└──────────────────────────────────┬────────────────────────────────┘
                                   │ gRPC server-streaming
┌──────────────────────────────────▼────────────────────────────────┐
│ harmonograf_client.Client (in goldfive's process)                  │
│   Client.observe() yields ControlEvent; bridge translates:        │
│     harmonograf.ControlEvent(STEER, note, suggested_action)        │
│       → goldfive.ControlMessage(kind=ControlKind.STEER,            │
│                                  payload={"note": ...})            │
│   await goldfive_channel.send(msg)                                 │
└──────────────────────────────────┬────────────────────────────────┘
                                   │ asyncio.Queue.put
┌──────────────────────────────────▼────────────────────────────────┐
│ goldfive.ControlChannel (_inbox) → channel.receive() unblocks      │
├───────────────────────────────────────────────────────────────────┤
│ SequentialExecutor / ParallelDAGExecutor                           │
│   between tasks: drain_controls(channel, ...)                      │
│   mid-task:      asyncio.wait({invoke, recv}, FIRST_COMPLETED)    │
│   dispatch_control(msg) → ControlOutcome(steer_message=msg, ack)  │
│   channel.ack(outcome.ack)  (flows back out)                       │
└──────────────────────────────────┬────────────────────────────────┘
                                   │ outcome.steer_message
┌──────────────────────────────────▼────────────────────────────────┐
│ steerer.observe(msg, session)                                      │
│   → DriftEvent(USER_STEER, WARNING, note)                          │
│ DefaultSteerer._handle_drift                                       │
│   → emits DriftDetected; severity >= WARNING → planner.refine     │
├───────────────────────────────────────────────────────────────────┤
│ LLMPlanner.refine (USER_STEER, "delete-and-replan")                │
│   drops pending, preserves completed, returns fresh Plan           │
│ Steerer._apply_revision + PlanRevised event                        │
│   session.plan = revised; revision_kind=USER_STEER                 │
└───────────────────────────────────────────────────────────────────┘

         Ack side mirrors outbound: _outbox → Client.acks()
                 → harmonograf → gRPC-Web → UI toast.
```

Four processes, one proto schema, one logical channel. The UI
round-trip for STEER is typically < 100 ms; total latency is
dominated by the LLM refine call (seconds), not the transport.

## 4. Every ControlKind — behaviour by kind

Per-kind: between-task behaviour, mid-task behaviour, effect on the
in-flight task / plan / session. Dispatcher lives in
`goldfive/executors/_control.py::dispatch_control`.

### `PAUSE`

Pause the outer loop; keep the in-flight task running until it
terminates on its own (aborting a partial artifact is usually worse).

- *Between tasks:* drain sets `paused=True`; executor blocks on
  `await control.receive()` until RESUME / CANCEL / STEER / REWIND_TO.
- *Mid-task:* ack only; invoke runs to completion; pause takes effect
  at the next between-task point.
- *Session/plan:* unchanged. *Drift:* `USER_PAUSE/INFO` — visible in
  the stream, below refine threshold.

### `RESUME`

Clear the paused flag. Between tasks (paused): exits the blocking
`receive`. Otherwise: ack, no-op. Session/plan/drift: none.

### `CANCEL`

Abort the run. The only kind that cancels an in-flight invocation.

- *Between tasks:* raises `_ControlCancelled`; executor emits
  `RunAborted(reason=payload.reason or "cancelled by control")`.
- *Mid-task:* `invoke_task.cancel()` with a 5 s grace. Adapters that
  ignore cancellation are orphaned with a warning — we never wedge the
  run.
- *Session/plan:* running tasks cascade to CANCELLED. *Drift:*
  `USER_CANCEL/CRITICAL` (audit only — executor short-circuits to
  abort without routing through refine).

### `STEER`

Redirect the run — the canonical live-replan trigger. Payload
`{"note": "...", "suggested_action": "..."}`. See §5 for semantics.

- *Between tasks:* fed to `steerer.observe` → `USER_STEER/WARNING` →
  `planner.refine`. Executor re-reads `session.plan` next iteration.
- *Mid-task:* adapter task cancelled (5 s grace, same as CANCEL), then
  the steer applies. Cancelled task → CANCELLED; refined plan
  typically reintroduces it (or a replacement) as PENDING.
- *Session/plan:* replaced; `revision_index` incremented. Completed
  preserved; pending dropped. *Drift:* `USER_STEER/WARNING`.

### `REWIND_TO`

Reset a task and every downstream task to PENDING. Payload
`{"task_id": "..."}`; unknown id → FAILURE ack.

- *Between tasks:* `_rewind_plan` walks the reachable subgraph via
  `plan.edges`, sets each status to `PENDING`, drops entries from
  `session.completed_results` and `session.task_progress`.
- *Mid-task:* ack only; reset lands at the next checkpoint. To cancel
  the current task first, send CANCEL → REWIND_TO → RESUME.
- *Session/plan:* structural reset — no plan revision, no refine.
  *Drift:* none.

### `STATUS_QUERY`

Emit a run-state snapshot onto the sink stream — explicit "what are
you working on?" ping rather than inferring from the last event.

- *Any time:* emits `DriftDetected(kind=CUSTOM, severity=INFO)` with
  detail `status_query control_id=... current_task=... completed=N/M
  pending=t3,t4`.
- *Session/plan/drift:* no side effects (CUSTOM/INFO is below refine).

### `INTERCEPT_TRANSFER`

Toggle `session._intercept_transfer`. Adapters that honour the flag
(ADK) refuse subsequent agent-to-agent transfers and surface them as
drift. Payload `{"enabled": true|false}`, default `true`.

- *Any time:* flag set on the session; next invocation sees it.
  Mid-task flips apply to the next task.
- *Session:* `session._intercept_transfer = flag`. *Drift:* none
  directly; ADK may synthesize `WRONG_AGENT` / `AGENT_TRANSFER`.

### `APPROVE` / `REJECT`

Resolve a pending approval waiter in `session.pending_approvals`.
`target_id` can be a `task_id` (Flow A, task-level via
`report_awaiting_approval`) or a `tool_call_id` (Flow B, ADK tool
confirmation). Payload `{"target_id": "...", "detail": "..."}`. See
§6 for the dual flow, [APPROVAL.md](APPROVAL.md) for the design.

- *Any time:* looks up `pending_approvals[target_id]`; stamps
  `meta["decision"]`, emits `ApprovalGranted` / `ApprovalRejected`,
  sets the `asyncio.Event`. Absent target → FAILURE ack.
- *Session:* `pending_approvals[target_id]` cleared. *Drift:* none —
  approval is a separate stream.

## 5. STEER — "delete-and-replan" semantics

`STEER` is the only control kind that invokes the planner. On
`USER_STEER` drift, `LLMPlanner.refine` follows a
**delete-and-replan** branch in
`goldfive/planner.py::_refine_user_steer`:

1. **Preserve every COMPLETED task.** Its output in
   `session.completed_results` stays available as context.
2. **Drop every PENDING / RUNNING / BLOCKED / CANCELLED task.** The
   steer implicitly says "what you were going to do is wrong."
3. **Ask the LLM for a fresh plan** with the user's `note` /
   `suggested_action` as primary directive, original goals as
   background, completed outputs as ground truth.
4. **Stamp revision metadata** — `revision_index = prev + 1`,
   `revision_kind = USER_STEER`, `revision_severity = WARNING`,
   `revision_reason = drift.detail`.

### Example

Four-task linear plan, mid-`draft`, user clicks [Steer] with
`note="focus on fewer, higher-quality slides"`:

```
before:
  t1 research   COMPLETED   results[t1] = "industry landscape..."
  t2 draft      RUNNING     (partial output)
  t3 review     PENDING
  t4 publish    PENDING

mid-task STEER cancels t2 (grace-window), observe → USER_STEER drift,
refine runs _refine_user_steer, LLM returns a 2-task plan.

after:
  t1 research          COMPLETED   (preserved)
  t5 draft_focused     PENDING
  t6 polish_and_ship   PENDING
  # revision_index = 1, revision_kind = USER_STEER

events (in order):
  TaskCancelled(t2)
  DriftDetected(USER_STEER, WARNING, "focus on fewer...")
  PlanRevised(revision_index=1, revision_kind=USER_STEER)
```

The next executor iteration picks up `t5` (no predecessors, PENDING)
and dispatches. From the sink the cancel + revision straddle the
steer; from the user the timeline re-orients.

### Why delete rather than edit

Refine-as-diff is cheaper but fragile — an LLM authoring a patch has
to reason about edge semantics and per-task status, and mistakes
compound across revisions. Deleting pending work and re-asking
produces clean plans every time and matches user expectation:
[Steer] feels like "redirect from here", not "edit the task list."
See [RATIONALE.md](RATIONALE.md).

## 6. APPROVE / REJECT — the dual flow

Two flows both resolve via `APPROVE`/`REJECT` and the same three
proto events (`ApprovalRequested`, `ApprovalGranted`,
`ApprovalRejected`), so a single UI affordance works for either.

### Flow A — task-level (any adapter)

The agent calls the 8th reporting tool,
`report_awaiting_approval(task_id, prompt, timeout_ms=...)`. Its
handler:

1. `steerer.mark_task_blocked(task_id, blocker="awaiting_approval")`.
2. Registers `session.pending_approvals[task_id] = asyncio.Event()` and
   stashes metadata in `session.pending_approvals_meta[task_id]`.
3. Emits `ApprovalRequested{target_id=task_id, kind="task", prompt=...}`.
4. `await`s the event.
5. Returns `{"decision": "approve"|"reject"|"timeout", "detail": ...}`
   to the agent.

The UI renders buttons on `ApprovalRequested` and sends
`ControlMessage(kind=APPROVE|REJECT, payload={"target_id": task_id})`.
The agent decides what to do with the returned decision — goldfive
does not auto-transition the task.

### Flow B — ADK tool confirmation

An ADK `FunctionTool` declared with `require_confirmation=True`.
goldfive's ADK plugin intercepts in `before_tool_callback`:

1. Uses `tool_context.function_call_id` (`adk-<uuid>`) as `target_id`.
2. Registers `pending_approvals[tool_call_id]` with metadata
   `{"kind": "tool", "tool_name": ..., "args": ...,
   "task_id": session.current_task_id}`.
3. Emits `ApprovalRequested{kind="tool", metadata={...}}`.
4. `await`s the event.
5. On APPROVE: emits `ApprovalGranted`, returns `None` → ADK runs the
   tool.
6. On REJECT: emits `ApprovalRejected`, returns
   `{"skipped": True, "reason": "user_rejected"}` → ADK uses that as
   the tool's response.

The model never sees the interception; from its perspective the tool
just took a while. Full ADK plugin reference is in
[APPROVAL.md](APPROVAL.md).

### Why one kind for two flows

Both flows need a yes/no, a UI-renderable event, a correlation id,
and a wait-then-resume mechanic. One pair of kinds + one pending map
beats two parallel plumbings. `target_id` disambiguates: task ids are
planner-authored; tool-call ids are ADK-authored (`adk-<uuid>`).

## 7. Writing a custom control handler

If you need a bespoke verb — "checkpoint", "inject a memo", "wait
for task X" — two escape hatches.

### 7.a Send a custom kind from the bridge

Unknown kinds still deliver; the default dispatcher acks them as
`UNSUPPORTED` until the runner-side dispatcher knows them:

```python
from goldfive import ControlChannel, ControlMessage

channel = ControlChannel()
await channel.send(ControlMessage(kind="CHECKPOINT", payload={"label": "pre-refine"}))
```

### 7.b Subclass the executor

`drain_controls` and `dispatch_control` in
`goldfive/executors/_control.py` are module-level helpers. Override
the drain loop in your executor subclass and fall back to
`dispatch_control` for kinds you don't handle:

```python
from goldfive.executors.sequential import SequentialExecutor
from goldfive.executors._control import ControlOutcome, dispatch_control
from goldfive.control import AckResult, ControlAck


class CheckpointingExecutor(SequentialExecutor):
    async def _handle_custom(self, msg, *, session, **kw):
        kind = str(getattr(msg.kind, "value", msg.kind)).upper()
        if kind == "CHECKPOINT":
            label = str(msg.payload.get("label", ""))
            session.agent_notes.setdefault("_checkpoints", []).append(label)
            return ControlOutcome(
                ack=ControlAck(
                    control_id=msg.id,
                    result=AckResult.SUCCESS,
                    detail=f"checkpointed at {label}",
                ),
            )
        return await dispatch_control(msg, session=session, **kw)
```

No registry yet — custom dispatchers usually also want custom
executor state (the `_checkpoints` list above), so the override point
is naturally the executor. File an issue if you write more than one.

### 7.c Testing and running end-to-end

`ControlChannel` is `asyncio.Queue`-backed. Tests instantiate one
directly, drive it with `await channel.send(...)`, and assert on the
acks iterator or on events captured by an `InMemorySink`. Patterns:
`tests/test_control_primitive.py`, `tests/test_executor_control.py`,
`tests/test_live_steering_e2e.py`. A runnable demo covering PAUSE /
STEER / CANCEL against an offline canned-LLM planner lives at
`examples/live_steering.py`.

## 7.d Intervention ladder (goldfive#142)

A STEER control message is the user-initiated entry into a broader
intervention story. Goldfive-internal drift detectors (loop, refusal,
goal drift, ...) enter the same pipeline via synthesized drifts, and
both paths route through `DefaultSteerer._handle_drift` which maps
`(drift_kind, severity, occurrence_count)` to one of six levels:

| Level | Name | What happens |
|---|---|---|
| **0** | `OBSERVE` | Emit `DriftDetected`; no further action. |
| **1** | `ABSORB` | Call `planner.refine`; install revised plan; continue. This is where USER_STEER lands. |
| **2** | `NUDGE` | Queue a corrective user message on `session.pending_nudges`. Not on the default table today; reserved for future policies. |
| **3** | `CANCEL_REINVOKE` | Refine; install revised plan; **dispatch a `GOLDFIVE_STEER` ControlMessage on the bound channel** so the executor's invoke loop cancels the in-flight invocation and restarts with a `[GOLDFIVE STEERING CONTROL …]` framed corrective. |
| **4** | `PAUSE_ESCALATE` | Emit `HUMAN_INTERVENTION_REQUIRED`; **dispatch a `GOLDFIVE_PAUSE_ESCALATE` ControlMessage on the bound channel** so the executor's pre-task loop blocks for user input. (Phase 2 of #246 replaced the deleted `session.paused_for_human_intervention` flag with this channel-routed signal.) |
| **5** | `TERMINATE` | Run-level abort. Only reached on repeat Level-4 that didn't resolve. |

The per-`(drift_kind, severity)` mapping lives in
`DefaultSteerer._LADDER` (see `goldfive/steerer.py`). A subclass can
override `_ladder_level_for` to tune the table.

USER_STEER specifically maps to **Level 1 (ABSORB)** at WARNING (the
default severity) — refine runs synchronously, the revised plan
installs, and the overlay loop restarts with the steer body as the
new user input (framed via `_compose_steer_restart_message`, see
goldfive#152). USER_CANCEL is special-cased to short-circuit the
ladder into an unconditional `RunAborted`.

See [DRIFT.md §"Intervention ladder (Levels 0-5)"](DRIFT.md#intervention-ladder-levels-0-5)
for the full per-drift-kind mapping and the corresponding code
path in `goldfive/steerer.py`.

## 8. What ControlChannel is *not*

- **Not a scheduler.** Send-order dispatch; no priority, no deadline
  ordering, no cancelling pending messages. PAUSE then CANCEL pauses
  first, then cancels.
- **Not a transport.** Process-local (`asyncio.Queue`). Cross-process
  transport is the bridge's problem (§3); the contract is the
  `ControlMessage` dataclass.
- **Not a persistence layer.** Raw control messages aren't logged to
  sinks; their *effects* are (DriftDetected, TaskCancelled,
  PlanRevised, ApprovalGranted). For an audit trail of raw messages,
  emit a `CUSTOM`-kind drift from a custom dispatcher.
- **Not required.** `Runner(control=None)` — the default — runs
  end-to-end with zero live-steering surface.

## See also

- [VOCABULARY.md](VOCABULARY.md) — type-system reference, the
  `ControlKind` × `DriftKind` bridge tables, worked STEER →
  USER_STEER example.
- [RATIONALE.md](RATIONALE.md) — why one channel, why STEER deletes,
  why approval rides the control path.
- [APPROVAL.md](APPROVAL.md) — approval design + ADK plugin API.
- [DRIFT.md](DRIFT.md), [STATE-MACHINE.md](STATE-MACHINE.md),
  [PROTOCOLS.md](PROTOCOLS.md) — related design docs.
- [../reference/api.md](../reference/api.md) — control types in the
  public API.
- [../guides/getting-started.md](../guides/getting-started.md),
  [../guides/observability-with-harmonograf.md](../guides/observability-with-harmonograf.md)
  — hands-on walkthroughs.
