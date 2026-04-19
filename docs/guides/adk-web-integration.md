# Using goldfive with `adk web`

`adk web` is Google ADK's local web UI for driving an agent
interactively. It expects the module it loads to expose an
`App(root_agent=...)` whose `root_agent` is a
`google.adk.agents.BaseAgent`. Before goldfive Phase 2,
`goldfive.wrap(agent)` always returned a `goldfive.Runner` — great for
programmatic use, but *not* a `BaseAgent`, so the ADK UI couldn't load
it.

As of issue [#77], `goldfive.wrap(adk_agent)` returns a polymorphic
[`GoldfiveADKAgent`](../../goldfive/adapters/adk_wrap.py). The same
object now works in both contexts:

| Call site | Needs | What goldfive returns |
|---|---|---|
| `adk web` | `BaseAgent` with `run_async(ctx)` | `GoldfiveADKAgent` — it IS a `BaseAgent`. |
| Programmatic | `await runner.run(user_input)` | `GoldfiveADKAgent` — same object, same method. |

No new entry points, no second wrap variant — the existing call site
just works.

## The recipe

Swap the line that constructs the `root_agent` for `goldfive.wrap(...)`
and leave the rest of your ADK code untouched:

```python
# agent.py — what `adk web` loads
from google.adk.agents import Agent
from google.adk.apps.app import App
import goldfive

real_agent = Agent(
    name="coordinator",
    model="gpt-4o-mini",
    sub_agents=[...],
)

# Before: root_agent = real_agent
root_agent = goldfive.wrap(real_agent)

app = App(name="my-demo", root_agent=root_agent)
```

Run it:

```bash
uv pip install -e '.[adk]'
adk web agent.py
```

Every user turn the UI submits now flows through goldfive's pipeline —
goal-derive → plan → execute → emit events — and the ADK UI renders a
short stream of `Event` objects summarising the plan + each completed
task.

A complete runnable file lives at
[`examples/adk_web_wrapped.py`](../../examples/adk_web_wrapped.py).

## What the UI sees each turn

When ADK invokes `root_agent.run_async(ctx)`, goldfive:

1. Extracts the latest user text from `ctx.user_content` (falling back
   to the session's event history).
2. Runs one `Runner.run(user_input, context={"adk_ctx": ctx})` pass.
3. Synthesises an `Event` stream from the resulting
   [`ExecutionOutcome`](../../goldfive/results.py):
   - A **plan summary** event (the first message).
   - One event per completed task, keyed by `Task.title`.
   - One line per drift event observed during the turn.
   - A terminal `turn_complete=True` event closing the turn.

This is deliberately minimal — enough for `adk web` to render a coherent
turn without duplicating what goldfive emits into its own event sinks.
Richer views come from attaching a sink to the wrapped runner (see
below).

## Programmatic use still works

```python
root_agent = goldfive.wrap(real_agent)

# Same object. Different call site.
outcome = await root_agent.run("plan a presentation about waffles")
```

`GoldfiveADKAgent.run(user_input, **kwargs)` delegates straight to the
inner `Runner.run(...)`, so every Runner knob — `context=`, cancellation
via `ControlChannel`, drift handling — behaves as before.

## Harmonograf observability composes cleanly

The wrapper exposes the inner `Runner`'s sink list as a property, so
`harmonograf_client.observe()` (which appends a `HarmonografSink`) works
unchanged:

```python
import goldfive
import harmonograf_client

root_agent = harmonograf_client.observe(goldfive.wrap(real_agent))
# observe() appended a sink to root_agent.sinks — the returned object is
# still the same GoldfiveADKAgent. adk web will load it just the same.

app = App(name="observed-demo", root_agent=root_agent)
```

## What is and is not shared with the adk-web session

The goldfive pipeline that runs for each turn uses its own internal
`InMemoryRunner` (via `ADKAdapter`). That is independent of the ADK
session that `adk web` hosts. In practice this means:

- Per-turn goldfive state (plan, drift history, reporting tool calls)
  lives in goldfive's `Session` and on goldfive's sinks.
- The ADK UI sees the synthesised `Event` stream from step 3 above.
- Cross-turn memory / state shared at the *adk web session* level is on
  the Phase 3 roadmap (the sibling agent currently wires
  conversation-level continuity through `run()`; see issue #71).

## Limitations

- `ControlChannel` steering from a harmonograf UI does not yet reach the
  adk-web-driven pipeline — Phase 4 follow-up.
- Only ADK-shaped agents get the polymorphic return. Callable agents and
  Claude SDK factories still return a plain `Runner`; that's intentional
  (they can't satisfy `BaseAgent` anyway).

[#77]: https://github.com/pedapudi/goldfive/issues/77
