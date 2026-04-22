# Harmonograf integration

[harmonograf](https://github.com/pedapudi/harmonograf) is a console for
observing and coordinating multi-agent systems — Gantt timeline,
drawer inspector, live plan-diff banners. goldfive is designed to feed
harmonograf as a first-class `EventSink`.

This guide covers how the integration works and the contract
goldfive provides. The concrete `HarmonografSink` implementation
lives in the harmonograf repo (`harmonograf_client.HarmonografSink`),
not in goldfive itself. For the end-to-end runnable walkthrough, see
[observability-with-harmonograf.md](observability-with-harmonograf.md).

Related: [ARCHITECTURE.md](../design/ARCHITECTURE.md),
[EVENT-MODEL.md](../design/EVENT-MODEL.md),
[writing-an-event-sink.md](writing-an-event-sink.md).

## Why this pairing

goldfive and harmonograf were born together and split on purpose:

- **harmonograf** owns the observability and human-coordination
  surface: the Gantt, the drawer, the control tabs, the timeline
  store. It is a product-shaped UI plus a server and a client
  library.
- **goldfive** owns the orchestration semantics: goal derivation,
  planning, task tracking, drift classification, refine. It is a
  framework-agnostic library.

The interface between them is the goldfive proto event stream. Every
orchestration decision goldfive makes produces an event; harmonograf
consumes events to render the timeline, the plan-diff banner, the
drift markers.

## Architectural placement

```
┌───────────────────────┐        ┌──────────────────────┐
│   your agent stack    │        │    harmonograf       │
│ (ADK / Claude SDK /   │        │                      │
│  callable / …)        │        │  server + frontend   │
│                       │        │  (timeline, drawer)  │
└──────────┬────────────┘        └──────────▲───────────┘
           │                                │
           │                                │
┌──────────▼────────────┐    proto events   │
│       goldfive        ├──────────────────▶│
│       Runner          │    via EventSink  │
└───────────────────────┘                   │
                                            │
           (separate repo)                  │
           HarmonografSink ─────────────────┘
           (goldfive EventSink that forwards
            to harmonograf's gRPC stream)
```

Your code imports goldfive for orchestration and harmonograf's
goldfive-adapter package for observability. The two are composed at
the sink layer.

## Proto alignment

goldfive's proto is the wire surface (per [issue #3](https://github.com/pedapudi/goldfive/issues/3)).
harmonograf adopts goldfive's proto directly — the same
`TaskStatus`, `DriftKind`, `Plan`, `Task`, and `TaskEdge` messages
flow from goldfive into harmonograf without translation.

This is D1 and D7 in the [vision doc](https://github.com/pedapudi/goldfive/issues/1):

> - **D1**: goldfive owns the proto layer; sinks (including
>   harmonograf) consume goldfive's proto.
> - **D7**: Harmonograf adopts goldfive's proto directly (replaces its
>   own TaskPlan / UpdatedTaskStatus).

The practical consequence: there is no field-by-field mapping layer.
A `goldfive.v1.Plan` message is the same bytes harmonograf stores on
disk and renders on-screen.

## The `HarmonografSink`

`HarmonografSink` ships in harmonograf's client library
(`client/harmonograf_client/sink.py`). It takes a pre-built
`Client` and forwards each goldfive `Event` through the client's
telemetry stream:

```python
from __future__ import annotations

from typing import Any

from harmonograf_client import Client
from goldfive.protocols import EventSink


class HarmonografSink:
    """goldfive EventSink that forwards events to a harmonograf server."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def emit(self, event_pb: Any) -> None:
        # harmonograf's telemetry stream expects goldfive events directly
        await self._client.emit_goldfive_event(event_pb)
```

Usage pattern from a goldfive caller:

```python
from goldfive import Runner
from goldfive.sinks import JSONLPersistenceSink
from harmonograf_client import Client, HarmonografSink


client = Client(name="my-agent", server_addr="127.0.0.1:7531")
runner = Runner(
    agent=my_agent_adapter,
    planner=my_planner,
    executor=my_executor,
    sinks=[
        HarmonografSink(client),
        JSONLPersistenceSink(path=f"./runs/{run_id}.jsonl"),
    ],
)
outcome = await runner.run("build me a slide deck about Python")
await runner.close()
client.shutdown(flush_timeout=5.0)
```

The run emits events to both sinks. harmonograf's UI lights up in
real time with the plan, the task states, drift markers, and plan
revisions. JSONL captures the same stream for crash recovery and
offline replay.

## What harmonograf renders, per event

An informal mapping of goldfive events to harmonograf UI elements.
The authoritative mapping lives in harmonograf's frontend
(`frontend/src/gantt/`).

| goldfive event | harmonograf UI effect |
|---|---|
| `RunStarted` | New session appears in the session picker; root agent row created. |
| `GoalDerived` | Goal chips appear in the session header. |
| `PlanSubmitted` | Initial Gantt bars materialize, one per task, in PENDING state. |
| `PlanRevised` | Plan-diff banner shows added / removed / modified tasks; a revision marker is placed on the timeline. |
| `TaskStarted` | Bar transitions to RUNNING color; liveness indicator starts pulsing. |
| `TaskProgress` | Bar fills to the reported fraction; drawer shows latest `detail`. |
| `TaskCompleted` | Bar transitions to COMPLETED color; `summary` surfaces in the drawer. |
| `TaskFailed` | Bar transitions to FAILED color; `reason` surfaces with an error badge. |
| `TaskBlocked` | Bar shows a blocked indicator; `blocker` surfaces in the drawer. |
| `TaskCancelled` | Bar greys out; drawer shows the cancellation reason. |
| `DriftDetected` | A drift marker drops onto the timeline with the `DriftKind` icon and color. |
| `RunCompleted` / `RunAborted` | Session shows terminal badge; timeline scroll locks. |

Drift markers are positioned at the drift's `emitted_at` timestamp.
`PlanRevised` markers chain off of the drift that caused them so the
UI can render a "drift → refine" trail.

## Replacing harmonograf's existing `TaskPlan`

harmonograf today has its own `TaskPlan` / `UpdatedTaskStatus`
messages in its proto (see `proto/harmonograf/v1/types.proto` in
harmonograf). Per D7 in the vision doc, those are superseded by
goldfive's messages once the integration lands.

The migration is mechanical:

1. Add `goldfive.v1.Plan` and friends to harmonograf's proto imports
   (or, ideally, drop harmonograf's task proto entirely and re-export
   from goldfive).
2. Switch `TaskRegistry.upsertPlan(...)` to consume a `goldfive.v1.Plan`.
3. Switch the drift-detection path to consume
   `goldfive.v1.DriftDetected` events from the stream.
4. Retire harmonograf's internal `_AdkState` state machine; state now
   comes from goldfive's `TaskStarted` / `TaskCompleted` / etc. events.

Because the proto values mirror harmonograf's existing ones
(`TaskStatus.COMPLETED` = `"COMPLETED"`, `DriftKind.NEW_WORK_DISCOVERED`
= `"new_work_discovered"`, etc.), the frontend needs no changes other
than pointing at the new messages.

## Bidirectional control

goldfive's `ControlChannel` is the inbound control surface. Pass one
to `Runner(control=...)` (or via `goldfive.wrap(control=...)`) and
the harmonograf UI's Steer / Pause / Cancel / Approve / Reject
buttons become `ControlMessage` values in a companion task that
bridges `client.observe()` → `channel.send(...)`. Acks ride back
through `channel.acks()` → the harmonograf client's ack stream.

STEER messages carry an `annotation_id` (goldfive#171) that
propagates through to the drift detail, the plan's `revision_reason`,
and the resulting `DriftDetected.annotation_id` field — letting the
UI deduplicate redundant clicks and show author metadata on each
refine. See [../design/CONTROL.md](../design/CONTROL.md) for the
complete protocol.

## The two harmonograf hooks: sink + telemetry plugin

Full observability of an ADK run needs **both** sides wired:

```python
import goldfive
from google.adk.apps.app import App
from harmonograf_client import Client, HarmonografSink, HarmonografTelemetryPlugin

client = Client(name="my-agent", server_addr="127.0.0.1:7531")

wrapped = goldfive.wrap(root_agent, sinks=[HarmonografSink(client)])
app = App(
    name="my-demo",
    root_agent=wrapped,
    plugins=[HarmonografTelemetryPlugin(client)],
)
```

| Hook | Where | Captures |
|---|---|---|
| `HarmonografSink` | goldfive runner `sinks=[...]` | goldfive `Event` proto (RunStarted, PlanSubmitted, TaskStarted, TaskCompleted, DriftDetected, PlanRevised, …) |
| `HarmonografTelemetryPlugin` | `App(plugins=[...])` or `goldfive.wrap(plugins=[...])` | ADK-native spans: INVOCATION (per-agent), LLM_CALL (per model request), TOOL_CALL (per tool call) |

Both hooks dedupe by plugin `name` (goldfive#166). If you wire the
plugin both at the App level and the goldfive.wrap level, only one
instance lands on the runner.

Per-LLM-call instrumentation logs (goldfive#172) ride alongside the
spans:

- `goldfive.llm.request invocation_id=… agent=… chars=… messages=…`
- `goldfive.llm.response invocation_id=… agent=… duration_ms=… chars=… usage=…`

These surface in harmonograf's drawer inspector as the LLM call's
input/output preview.

## Running the two together

For the end-to-end runnable walkthrough (install harmonograf, boot
the server and UI, run the bundled example, watch events light up
the Gantt), see
[observability-with-harmonograf.md](observability-with-harmonograf.md).
The bundled example at
[`examples/harmonograf_observed/`](../../examples/harmonograf_observed/)
is the reference wiring.

## FAQ

**Will goldfive depend on harmonograf?** No. The dependency is
one-way: harmonograf has an optional dep on goldfive (for the proto).
goldfive has no concept of harmonograf at the code level — it sees
only an `EventSink`.

**What about harmonograf's reporting tools?** They are the same seven
tools goldfive defines (see [tool-protocol.md](../reference/tool-protocol.md)).
The intent is that harmonograf's client library stops defining them
and re-exports from goldfive.

**What happens if goldfive emits an event harmonograf doesn't
recognize?** harmonograf ignores unknown fields (standard proto3
behavior). Adding new events or fields to goldfive is
backwards-compatible for harmonograf.

**What happens if the harmonograf server is down when a goldfive run
fires?** The `HarmonografSink` buffers with bounded capacity; on
overflow, events drop (with a log warning). The other sinks
(especially `JSONLPersistenceSink`) are unaffected — the durable log
is intact regardless of harmonograf's health.

## Tracking issue

- Main vision issue: [goldfive#1](https://github.com/pedapudi/goldfive/issues/1).
- Proto definition: [goldfive#3](https://github.com/pedapudi/goldfive/issues/3).
- `HarmonografSink` ships from harmonograf's client library
  (`harmonograf_client.HarmonografSink`); see
  [harmonograf/docs/goldfive-integration.md](https://github.com/pedapudi/harmonograf/blob/main/docs/goldfive-integration.md)
  for the harmonograf-side reference.
