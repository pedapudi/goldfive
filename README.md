# goldfive

**Stay on target.**

goldfive is a small, framework-agnostic Python library that wraps an agent
with the orchestration scaffolding most agents quietly need: an explicit
**goal**, a **plan** broken into tasks, per-turn **drift analysis**, and a
**steering** loop that nudges the agent back on course when it wanders.

It does not ship an LLM client, a prompt DSL, or a tool registry. It wraps
whatever agent runtime you already use (Google ADK, the Anthropic SDK, a
plain callable, ...) behind a narrow `AgentAdapter` protocol and gives you:

- a `Runner` that drives the agent turn by turn against a `Goal`
- pluggable `Planner`, `DriftAnalyzer`, and `Steerer` components
- a `TelemetrySink` stream of structured events you can log, render, or
  ship to an observability console

goldfive is the orchestration half of
[harmonograf](https://github.com/pedapudi/harmonograf), extracted so you
can use the control loop without the console.

## Install

```bash
uv add goldfive           # recommended
# or
pip install goldfive
```

Optional extras: `goldfive[adk]`, `goldfive[claude]`, `goldfive[dev]`.

## Hello goldfive

> **Intended API — coming in #15.** The types below do not exist yet; this
> snippet is a sketch of how the package will feel once the core lands.

```python
import asyncio
from goldfive import Runner, Goal, CallableAdapter, InMemorySink

async def agent(prompt: str) -> str:
    return f"echo: {prompt}"

async def main() -> None:
    sink = InMemorySink()
    runner = Runner(
        adapter=CallableAdapter(agent),
        goal=Goal("Summarize today's standup in three bullets."),
        sink=sink,
    )
    await runner.run()
    for event in sink.events:
        print(event)

asyncio.run(main())
```

## Docs

See [`docs/guides/getting-started.md`](docs/guides/getting-started.md) for
a longer walkthrough (doc coming in a later PR).

## License

Apache-2.0.
