---
name: debug-goldfive
description: Diagnose a broken goldfive run — decision tree from symptom to root cause, plus the tools to see what's happening.
applies-when: ["debug", "something is broken", "goldfive fails", "task stuck", "sink not receiving"]
---

# Debug goldfive

Broken run? Start here. This is a triage tree. For the full symptom
catalogue with quoted error messages, see
[docs/guides/troubleshooting.md](../docs/guides/troubleshooting.md) —
don't duplicate it; send people there for the specifics.

## Decision tree

```
run fails →
├─ outcome.reason == "no plan generated"
│    → planner returned None. StaticPlanner(tasks=[])? LLMPlanner.call_llm
│      raised or returned non-JSON? Empty goals?
│    → see docs/guides/troubleshooting.md § "no plan generated"
│
├─ outcome.reason starts with "planner.generate raised:"
│    → custom planner raised. Enable DEBUG on goldfive.runner.
│
├─ outcome.reason starts with "executor.run raised:"
│    → executor crashed. Inspect InMemorySink.events before the crash.
│
├─ outcome.success is True but task never started
│    → dependency cycle in plan.edges OR assignee_agent_id outside
│      adapter.available_agents. Print plan.topological_stages().
│
├─ task stuck in RUNNING / PENDING
│    → adapter returned without auto-completing AND didn't call
│      report_task_completed / _failed / _blocked. See "state machine"
│      in docs/guides/troubleshooting.md.
│
├─ sink registered but events is empty
│    → sink passed somewhere other than Runner(sinks=[...])? forgot
│      await runner.close()? run aborted before the first emission?
│
├─ ImportError: google.adk / anthropic / grpcio / LoggingSink
│    → the optional extra isn't installed. See use-goldfive.md § Install.
│
└─ adapter never sees reporting tools
     → your register_reporting_tools is a no-op; implement it. See
       adapters.md.
```

## Observation tools

### `InMemorySink` — the ground-truth event log

```python
from goldfive import InMemorySink, Runner

sink = InMemorySink()
runner = Runner(..., sinks=[sink])
outcome = await runner.run("go")
await runner.close()

for e in sink.events:
    kind = e.WhichOneof("payload") if hasattr(e, "DESCRIPTOR") else e.get("kind")
    seq = e.sequence if hasattr(e, "DESCRIPTOR") else e.get("sequence")
    print(f"seq={seq:>3}  kind={kind}")
```

Current `main` emits every event as a proto `Event` message with a
`oneof` `payload`. `goldfive.events.make_event` still exists as a dict
fallback for callers that don't want proto; old examples (and the
troubleshooting guide) reference it — duck-type on `DESCRIPTOR` to
tolerate both shapes.

### `LoggingSink` — every event flying past

```python
import logging

from goldfive.sinks import LoggingSink

logging.basicConfig(level=logging.DEBUG)
runner = Runner(..., sinks=[LoggingSink()])
```

One JSON line per event. Pair this with `InMemorySink` when you want
both a live tail and a post-run assertion surface.

### Runner-internal logging

```python
import logging
logging.getLogger("goldfive.runner").setLevel(logging.DEBUG)
logging.getLogger("goldfive.planner").setLevel(logging.DEBUG)
```

Exception traces from `planner.generate`, `executor.run`,
`register_reporting_tools`, and `steerer.bind` are logged at `ERROR`
with the outcome `reason` field summarising them.

## Event ownership map

When an event goes missing, look at who emits it:

| Event | Emitted by |
|---|---|
| `RunStarted`, `GoalDerived`, `PlanSubmitted`, pre-executor `RunAborted` | `Runner` |
| `TaskStarted`, `TaskProgress`, `TaskCompleted`, `TaskFailed`, `TaskBlocked`, `TaskCancelled` | `Steerer` (via `mark_task_*`) |
| `PlanRevised`, terminal `RunCompleted` / `RunAborted` | `Executor` |
| `DriftDetected` | `Steerer` |

If your sink never sees `TaskStarted`, the executor never dispatched or
the steerer is bypassed. If it never sees `RunCompleted`, the executor
aborted — check `outcome.reason`.

## Drift kinds and taxonomy

25+ `DriftKind` values (`TOOL_ERROR`, `AGENT_REFUSAL`, `CONTEXT_PRESSURE`,
...). Full catalogue with classification rules:
[docs/design/DRIFT.md](../docs/design/DRIFT.md). The classifiers
(`classify_refusal`, `classify_stop_reason`, `classify_tool_error`) are
pure functions; call them on your own upstream signals to mint drift
events outside the built-in detector.

## Consulting VOCABULARY to diagnose

When a bug is shaped like "two names that sound similar behave
differently" or "I can't tell which enum value to reach for",
[docs/design/VOCABULARY.md](../docs/design/VOCABULARY.md) is the
single-source answer. It enumerates every `TaskStatus`, `DriftKind`,
`DriftSeverity`, `ControlKind`, every `Event` payload variant, and the
bridges between them — with emitter + consumer listed for each.

Quick triage table:

| Symptom | VOCABULARY section |
|---|---|
| "I sent a STEER but saw `DriftKind.USER_STEER`, is that right?" | [§2](../docs/design/VOCABULARY.md#2-controlkind-vs-driftkind--a-worked-example) — the STEER → USER_STEER flow diagram |
| "Which `DriftKind` should I synthesize for X?" | [§5](../docs/design/VOCABULARY.md#5-driftkind-taxonomy) — all 26 kinds grouped by trigger |
| "Who emits `PlanRevised`?" | [§6](../docs/design/VOCABULARY.md#6-event-payload-kinds) — event factory + emitter + "when" table |
| "Can a task go from BLOCKED to COMPLETED?" | [§4](../docs/design/VOCABULARY.md#4-taskstatus-state-machine) — transition ownership table + state diagram |
| "Does severity WARN trigger refine, or just log?" | [§7](../docs/design/VOCABULARY.md#7-severity-ladder) — severity ladder with the exact `_severity_ge` comparison |
| "What are all the ControlKinds and which ones touch the planner?" | [§3](../docs/design/VOCABULARY.md#3-all-control--drift-bridges) — control → drift bridge table |

For the "why is it this way" questions that pair with the vocabulary
answers — e.g. *why* does STEER delete-and-replan, *why* is there both
a status and a drift kind named `BLOCKED` — see
[docs/design/RATIONALE.md](../docs/design/RATIONALE.md).

## Quick reference

```python
# minimal inspection harness
import asyncio, logging
from goldfive import InMemorySink, Runner

logging.basicConfig(level=logging.DEBUG)

sink = InMemorySink()
runner = Runner(..., sinks=[sink])
try:
    outcome = await runner.run(user_input)
finally:
    await runner.close()

print(f"success={outcome.success}  reason={outcome.reason!r}")
print(f"{len(sink.events)} events")
```

## Control channel (live steering)

When `Runner(control=channel)` is wired, a new class of failure modes
shows up. Triage tree:

```
control message sent, nothing happens →
├─ channel.acks() never yields anything
│    → channel is not the same instance the Runner got. Print
│      id(channel) on both ends to verify.
├─ ack.result == AckResult.UNSUPPORTED
│    → dispatcher doesn't recognize the kind. Phase-1 enum:
│      PAUSE, RESUME, CANCEL, STEER, REWIND_TO, APPROVE, REJECT.
│      STATUS_QUERY + INTERCEPT_TRANSFER accepted as string-kind.
│      Anything else → you need a custom dispatcher (CONTROL.md §7).
├─ ack.result == AckResult.FAILURE
│    → the specific kind rejected the payload. Common causes:
│        REWIND_TO with missing / unknown payload.task_id
│        APPROVE/REJECT with target_id not in pending_approvals
│      The ack.detail string names the exact reason.
├─ STEER sent but plan never revises
│    → planner.refine raised or returned None. Enable DEBUG on
│      goldfive.planner. For LLMPlanner check call_llm output parses.
├─ PAUSE sent, executor keeps running tasks
│    → expected mid-task — the current task finishes, then pause
│      takes effect. PAUSE never cancels in-flight work; only CANCEL
│      and STEER do (both via invoke_task.cancel() with 5 s grace).
├─ CANCEL sent, adapter keeps running after 5 s
│    → adapter is ignoring asyncio.CancelledError. The run is
│      unwedged (orphaned task + warning log) but the adapter process
│      is leaking. Fix the adapter's cancellation handling.
└─ APPROVE sent, report_awaiting_approval tool still blocked
     → target_id mismatch. Flow A uses the task_id; Flow B uses
       tool_context.function_call_id (shape `adk-<uuid>`). Check
       session.pending_approvals keys against payload.target_id.
```

Full protocol and kind-by-kind behaviour:
[docs/design/CONTROL.md](../docs/design/CONTROL.md). Approval-specific
design: [docs/design/APPROVAL.md](../docs/design/APPROVAL.md).

## "Why is a `LOOPING_REASONING` drift firing?"

```
A LOOPING_REASONING drift appeared on the stream — what triggered it?
│
├─ Always ask first: is the session running with an adapter that
│  surfaces reasoning? Only OpenAI-compat models with
│  `reasoning_content`, Anthropic extended-thinking, and Google
│  thought-part responses feed the detector. If none fire,
│  chain-of-thought never enters the pipeline.
│
├─ Check which detector fired (the drift.detail carries the reason):
│   detail contains "hash=" → byte-identical loop detection tripped
│   detail contains "cosine=" → embedding-based similarity tripped
│
├─ Byte-identical hash match:
│     goldfive/drift/reasoning.py::reasoning_hash normalises
│     whitespace + case before hashing. Two "reasoning" blocks that
│     look different but trim to the same tokens collide. Inspect
│     ``session.reasoning_history[-5:]`` — the previous 5 entries are
│     what the detector compared against.
│
├─ Semantic (cosine) match:
│     Only fires when `goldfive[embedding]` is installed and the
│     model loaded. Threshold is
│     `LOOPING_REASONING_SIMILARITY_THRESHOLD` = 0.9. If you're seeing
│     false positives on legitimately iterative reasoning (e.g. the
│     model enumerating sub-hypotheses), subclass `DefaultSteerer`
│     and override `observe_reasoning` to raise the threshold or
│     shrink the window (`LOOPING_REASONING_HASH_WINDOW`, default 5).
│
└─ Not firing when you expect a loop?
     goldfive/drift/reasoning.py slices
     ``session.reasoning_history[-WINDOW-1 : -1]`` — it excludes the
     most recent entry because the steerer appends the current text
     before analysis. A manual caller must append current-to-history
     first.
```

For `CONFUSION` / `OFF_TOPIC` / `INTENT_DIVERGENCE` the same detail
field explains which pattern or distance threshold tripped. The four
kinds live in `goldfive/drift/reasoning.py`; tune the module-level
constants (`CONFUSION_MIN_HITS`, `OFF_TOPIC_DISTANCE_THRESHOLD`,
`CONFUSION_MARKERS` regex, `_INTENT_DIVERGENCE_MARKERS` regex) to
project needs.

## Common pitfalls

- Reading `sink.events` before `await runner.close()` — buffered sinks
  haven't flushed. `InMemorySink` is fine either way.
- Assuming every event is proto. Current `main` is; historical logs
  and `goldfive.events.make_event` still produce dict envelopes, so
  duck-type on `DESCRIPTOR`.
- Using `GRPCSink` alone with a sink that emits dict envelopes — it
  only forwards objects with a proto `DESCRIPTOR`. Pair with
  `JSONLPersistenceSink` if you mix shapes.
- Mixing two different `events_pb2.Event` classes (goldfive's vs a
  downstream-regenerated copy) → `isinstance` fails. See
  `docs/guides/troubleshooting.md` § "Proto and types".

## Related

- [docs/guides/troubleshooting.md](../docs/guides/troubleshooting.md) — detailed symptom → fix catalogue.
- [docs/design/CONTROL.md](../docs/design/CONTROL.md) — live-steering protocol, per-kind behaviour, custom dispatcher hook.
- [docs/design/VOCABULARY.md](../docs/design/VOCABULARY.md) — exhaustive type-system reference (start here when a name is confusing).
- [docs/design/RATIONALE.md](../docs/design/RATIONALE.md) — design-rationale docs (start here when a choice feels arbitrary).
- [docs/design/DRIFT.md](../docs/design/DRIFT.md) — drift taxonomy.
- [docs/design/EVENT-MODEL.md](../docs/design/EVENT-MODEL.md) — event ownership, sequence semantics.
- [events.md](events.md) — emitting a new event.
- [sinks.md](sinks.md) — sink contract.
