# Telemetry with harmonograf

This guide is for the person staring at the harmonograf UI, asking
"what is my agent actually doing and why did that happen?" It is
the companion to
[observability-with-harmonograf.md](observability-with-harmonograf.md) —
that guide gets you to the point of seeing events on the wire;
this one walks through how to read what the wire is saying.

If you already have harmonograf running and your run is flowing
through, skip to [§What you'll see](#what-youll-see-in-the-ui).

## Minimum setup

Four pieces:

1. `uv sync --extra proto` in `~/git/goldfive` (or wherever).
2. `make server-run` in `~/git/harmonograf` (Python gRPC server on
   `127.0.0.1:7531`).
3. `pnpm dev --port 5173 --strictPort` in `~/git/harmonograf/frontend`
   (the React UI).
4. Wrap your runner:

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

   Two hooks. `HarmonografSink` carries the goldfive `Event` stream
   (run / plan / task / drift); `HarmonografTelemetryPlugin` carries
   per-agent spans. Both route by `session_id` (goldfive#155 stamps
   each event; goldfive#161 pins adk-web's session id onto the
   goldfive Session so there's one row per run in harmonograf).

   `harmonograf_client.observe(wrapped)` is the legacy entry point —
   still works, appends a `HarmonografSink`, enables STEERING +
   HUMAN_IN_LOOP capabilities — but the newer path attaches the
   telemetry plugin explicitly at the `App` level so adk-web's
   runner sees it.

Runnable examples:

- [`examples/harmonograf_observed/agent.py`](../../examples/harmonograf_observed/agent.py)
  — minimal CallableAdapter + sink.
- [`examples/presentation_agent/agent.py`](../../examples/presentation_agent/agent.py)
  — full ADK tree + both hooks.

## What you'll see in the UI

Open `http://127.0.0.1:5173`. The chrome is four panels:

- **Sessions** (top-left) — every `run_id` harmonograf has seen.
  Newest first. Click one to focus the other views on it.
- **Agents timeline** — Gantt chart of agent invocations across
  the run. Each bar is an `INVOCATION` span; children are
  `LLM_CALL` and `TOOL_CALL` spans.
- **DAG** — the current plan, rendered as the topological DAG.
  Tasks show status badges (PENDING / RUNNING / COMPLETED /
  FAILED / CANCELLED / BLOCKED) and colour-code by severity. When
  the plan revises mid-run, the DAG morphs in place; the revision
  history is accessible via the "revisions" picker.
- **Task list** — every task in the current plan, with status,
  duration, and a preview of the final outcome text. Sortable.
- **Live Activity** — stream of events as they land. Useful for
  "watch the agent think" during a live run.

Four secondary panels live in the right rail:

- **Notes** — any `TaskProgress` notes and agent annotations.
- **Drifts** — every `DriftDetected` event on this session, with
  kind, severity, and `detail`.
- **Plan revisions** — when the plan has been revised, a timeline
  showing what was added / removed / modified at each revision
  (populated from the `PlanRevisionDiff` sidecar — goldfive #106).
- **Approvals** — open `ApprovalRequested` waiters with
  Approve / Reject buttons.

## Reading the Gantt view

The Agents timeline is the densest surface. Three things to look
for:

**Span hierarchy.** Each top-level bar is one call to
`adapter.invoke(task, session)` — the `INVOCATION` span. Its
children are `LLM_CALL` spans (one per round-trip to the model)
and `TOOL_CALL` spans (one per tool the model called, including
the goldfive reporting tools). Clicking a parent expands its
children.

**Duration columns.** Hover any span for the exact
start / end / duration. The column widths are proportional to
wall-clock duration — a fat `LLM_CALL` is typically where your
latency is, not the orchestration overhead (which is usually
< 1 ms per transition; see [performance.md](../performance.md)).

**Cancellation strikethroughs.** A span that was cancelled
mid-flight (CANCEL control, unrecoverable drift cascade, STEER
re-route) renders struck-through. You'll see this on every
downstream task after a cascade — each one is one
`TaskCancelled` event from the
`cascade_cancel_downstream` primitive (goldfive #103 + #107).

**Revision markers.** Each time the plan is revised, a vertical
marker appears across the timeline at the `PlanRevised` event's
timestamp. Hover it to see the triggering drift (kind + severity
+ detail) and the diff summary.

**Delegation bars (ADK).** When goldfive wraps an ADK tree with
`AgentTool`-wrapped specialists, each
`adapter.invoke(task, session)` dispatch starts a top-level
`INVOCATION` bar tagged with the dispatched agent's name
(`AgentInvocationStarted.agent_name`). If that agent calls an
`AgentTool` mid-turn, a nested bar appears for the wrapped
agent's sub-Runner invocation, connected to its parent by a
dashed delegation edge. All bars in the nested chain share the
same goldfive-dispatched `task_id` — the edge is drawn from the
`DelegationObserved` event, and the parent / child relationship
is reconstructed from `AgentInvocationStarted.parent_invocation_id`.
The three events are documented in
[EVENT-MODEL.md §"Agent-invocation events"](../design/EVENT-MODEL.md#agent-invocation-events).

## Clicking on a span

Left-click opens a popover with four actions:

| Action | When to use it |
|---|---|
| **Steer** | The run is live and you want to redirect it — add guidance, pivot to a different approach, or nudge it back on goal. Sends a `ControlKind.STEER` with the free-text message you type. Deletes any downstream PENDING tasks and refines. |
| **Annotate** | You want to leave a note (for yourself or a reviewer) on this span without affecting the run. Notes are persisted and show up in the Notes panel. |
| **Copy id** | Grab the `span_id` / `event_id` / `run_id` for pasting into a ticket or SQL query. |
| **Open drawer** | Open the full Inspector Drawer for this span. See next section. |

"Steer" is the destructive one — use it when the run is still
live and you've decided the current direction is wrong. "Annotate"
is cheap and reversible; reach for it first when you're still
deciding.

## The Inspector Drawer

Click "Open drawer" (or double-click a span) to open the right-side
Inspector. Five tabs:

- **Reasoning.** Chain-of-thought blocks extracted by the
  adapter: OpenAI `reasoning_content`, Anthropic `thinking`
  blocks, Google thought parts. Surfaced via
  `Steerer.observe_reasoning` and stored in the
  `llm.reasoning` span attribute (harmonograf #43 + #45). This is
  the first place to look when asking "why did the agent make
  that choice" — the reasoning often reveals intent that didn't
  surface in the visible output.
- **Tool input / output.** The exact `args` payload the model
  sent and the `result` the tool returned. For goldfive reporting
  tools (`report_task_*`, `report_plan_divergence`, etc.), the
  result is the ACK dict — an `{"acknowledged": False, "error":
  "task_already_terminal"}` here is the signal that a guard
  layer rejected the call (see [common-failure-modes.md](common-failure-modes.md)).
- **Context window.** Preview of the messages goldfive rendered
  for this task — the task context, the completed-results
  summary, the goal summary.
- **Span links.** Forward / back navigation to parent and child
  spans. Useful for walking a deep tree.
- **Raw.** The proto event JSON. Last resort when the cooked
  views are hiding something.

## Live steering from the UI

With `observe()` attached (which enables STEERING +
HUMAN_IN_LOOP capabilities since harmonograf #44), the UI shows
two always-on controls in the run header:

- **Pause / Resume** — sends `ControlKind.PAUSE` / `RESUME`. The
  current in-flight task finishes, then the executor blocks on
  the control channel until a RESUME arrives. Useful when you
  want to pause a run, read the reasoning carefully, and decide
  what to do.
- **Cancel** — sends `ControlKind.CANCEL`. The current task is
  cancelled (adapter is signalled; has 5 s grace), the cascade
  fires, and the run ends with
  `outcome.reason="cancelled by control"`.

Plus the context-sensitive actions on every span:

- **Steer** (from the span popover) — `ControlKind.STEER` with
  the user text as the message. The steerer handles it as a
  `USER_STEER` drift: the current task is CANCELLED, the
  cascade cancels downstream PENDINGs, and `planner.refine` is
  called with the steer as the drift payload. If the refine
  returns a new plan, it is installed on the session and
  execution continues against the new plan.
- **Cancel & redirect** — shorthand for CANCEL + STEER. Current
  task stops immediately; the steer text redirects the
  replanner.
- **Add to queue** — when you want the steer to apply at the
  next task boundary, not mid-task. Enqueues the steer without
  cancelling the current task.

Verifying a steer took effect: watch the Drifts panel for a new
`USER_STEER` event (severity WARNING); the Plan revisions panel
for a new revision carrying the `added_task_ids` from the
refine; the DAG for the morphed plan. If the drift arrives but
no revision follows, `planner.refine` declined (returned
`None`); the task CANCEL still fired, and the run ends cleanly
as "incomplete but not broken."

## Diagnosing "why did my run fail"

Three places to look, in order:

1. **`outcome.success` + `outcome.reason`.** Printed by the
   reproducer or visible in the UI's session header. See
   [common-failure-modes.md](common-failure-modes.md) for the
   reason → root cause table.
2. **The last `DriftDetected` event.** Filter the Drifts panel
   by severity CRITICAL; the last one is almost always the one
   that cascaded into `RunAborted`. Its `kind` tells you what
   class of failure; its `detail` usually names the specific
   task.
3. **The reasoning of the last `LLM_CALL` before the abort.**
   Open the drawer on the last span before the `RunAborted`
   marker; read the Reasoning tab. The model will often narrate
   that it gave up, got confused, or decided the task was
   unreachable — which `detect_drift` then classified.

When the three disagree (e.g. reason says "orphaned pending",
last drift is INFO `OFF_TOPIC`, reasoning is unremarkable),
look at [insight-from-logs.md](insight-from-logs.md) — the raw
log-level view is what you want.

## Plan revisions in the UI

When `planner.refine` produces a new plan, harmonograf renders
the revision with three affordances:

- **Diff summary on the revision marker.** Hover shows
  "+2 tasks, -1 task, 3 edges modified" — sourced from the
  `PlanRevisionDiff` sidecar on the `PlanRevised` event
  (goldfive #106).
- **Diff detail in the Plan revisions panel.** Expanded, you
  see the exact `added_task_ids`, `removed_task_ids`,
  `modified_task_ids`, `added_edges`, `removed_edges`. No need
  to fetch the prior plan to figure out what changed.
- **Morphed DAG.** The current-state DAG switches to the new
  plan. Terminal tasks from the prior plan are preserved
  (PLAN-LIFECYCLE §3.1 — goldfive #105), so a COMPLETED task
  in revision N shows up COMPLETED in revision N+1 and you
  don't lose history.

A cascade visualised: when a task FAILED-fatal or CANCELLED
triggers the cascade, you'll see exactly one `TaskCancelled`
event per downstream task on the Live Activity stream, all
within a few ms of each other, each with
`reason="cascade from <task_id>"`. That's
`cascade_cancel_downstream` (goldfive #107) doing its BFS walk.

## Related

- [observability-with-harmonograf.md](observability-with-harmonograf.md) — one-time setup walkthrough.
- [harmonograf-integration.md](harmonograf-integration.md) — sink protocol + client factory.
- [insight-from-logs.md](insight-from-logs.md) — same diagnostics without the UI.
- [common-failure-modes.md](common-failure-modes.md) — catalog of failure shapes + recovery path.
- [../design/CONTROL.md](../design/CONTROL.md) — the live-steering protocol the UI speaks.
- [../design/EVENT-MODEL.md](../design/EVENT-MODEL.md) — what each event on the wire carries.
