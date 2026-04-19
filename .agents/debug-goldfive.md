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
- [docs/design/DRIFT.md](../docs/design/DRIFT.md) — drift taxonomy.
- [docs/design/EVENT-MODEL.md](../docs/design/EVENT-MODEL.md) — event ownership, sequence semantics.
- [events.md](events.md) — emitting a new event.
- [sinks.md](sinks.md) — sink contract.
