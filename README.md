# goldfive

**Stay on target.**

goldfive is a small, framework-agnostic Python library that wraps an agent
with the orchestration scaffolding most agents quietly need: an explicit
**goal**, a **plan** broken into tasks, per-turn **drift analysis**, and a
**steering** loop that nudges the agent back on course when it wanders.

It does not ship an LLM client, a prompt DSL, or a tool registry. It wraps
whatever agent runtime you already use (Google ADK, the Anthropic SDK, a
plain callable, ...) behind a narrow `AgentAdapter` protocol and gives you:

- a `Runner` (or one-line `goldfive.wrap` / `goldfive.run`) that drives
  the agent turn by turn against a `Goal`
- pluggable `GoalDeriver`, `Planner`, `Executor`, and `Steerer` components
- an `EventSink` stream of proto-encoded events you can log, persist, or
  ship to an observability console

goldfive is the orchestration half of
[harmonograf](https://github.com/pedapudi/harmonograf), extracted so you
can use the control loop without the console.

## Get running in 10 minutes

The fastest way to see goldfive work end-to-end is with its observability
console, [harmonograf](https://github.com/pedapudi/harmonograf). The
walkthrough installs both, boots a local stack, runs the
`examples/harmonograf_observed/` agent, and shows every event flowing
into the UI — no LLM credentials required.

1. `uv sync` in this repo (Python 3.11+, `uv` on your PATH).
2. Clone and `make demo` in harmonograf (server + UI on :7531 and :5173).
3. `uv run python examples/harmonograf_observed/agent.py`.

Full walkthrough: **[observability-with-harmonograf.md](docs/guides/observability-with-harmonograf.md)**.

## Install

```bash
uv add goldfive           # recommended
# or
pip install goldfive
```

Optional extras:

- `goldfive[adk]` — Google ADK adapter (`google-adk`).
- `goldfive[claude]` — Claude Agent SDK adapter (`anthropic`).
- `goldfive[examples]` — runtime deps for the scripts in [`examples/`](examples/) (`rich`).
- `goldfive[proto]` — regenerate proto stubs with `make proto` (`grpcio`, `grpcio-tools`, `mypy-protobuf`).
- `goldfive[dev]` — test + lint tooling used by the repo itself (`pytest`, `ruff`, `mypy`, ...).

## Hello goldfive

The fastest path to a goldfive-wrapped agent is a single call to
`goldfive.run`. It picks the right adapter for your agent, reuses
the agent's LLM when it can detect one, and returns an
`ExecutionOutcome`:

```python
import asyncio
import goldfive

# `agent` is any of: an ADK BaseAgent, a Claude SDK client factory,
# an async (task, session, tools) -> InvocationResult callable, or
# anything implementing goldfive.AgentAdapter.
outcome = await goldfive.run(agent, "make a presentation about waffles")
```

Prefer to keep the runner around (for `.resume()`, custom sinks, or
multiple runs)? Use `goldfive.wrap`:

```python
runner = goldfive.wrap(agent, sinks=[my_sink])
outcome = await runner.run("make a presentation about waffles")
```

Every default component is overridable — pass `planner=`,
`executor=`, `sinks=`, `call_llm=`, `model=`, or
`max_plan_reinvocations=` as keyword arguments to either function.

A runnable demo lives in
[`examples/hello_callable.py`](examples/hello_callable.py).

## Docs

**Start with [`docs/guides/getting-started.md`](docs/guides/getting-started.md)** —
install, run your first goldfive-wrapped agent in about ten minutes,
inspect the event stream. Concrete and runnable.

### Design

- [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md) — overview of the six primitives, how they compose, full lifecycle.
- [`docs/design/PROTOCOLS.md`](docs/design/PROTOCOLS.md) — the six protocol contracts with minimal implementations.
- [`docs/design/STATE-MACHINE.md`](docs/design/STATE-MACHINE.md) — task lifecycle state diagram, transition rules, invariants.
- [`docs/design/DRIFT.md`](docs/design/DRIFT.md) — full drift-kind taxonomy (25+), classification rules, refine policy.
- [`docs/design/EVENT-MODEL.md`](docs/design/EVENT-MODEL.md) — proto event taxonomy, sequence semantics, `EventSink` contract.

### Guides

- [`docs/guides/getting-started.md`](docs/guides/getting-started.md) — install + first agent.
- [`docs/guides/observability-with-harmonograf.md`](docs/guides/observability-with-harmonograf.md) — ten-minute end-to-end with the harmonograf UI.
- [`docs/guides/writing-an-agent-adapter.md`](docs/guides/writing-an-agent-adapter.md) — wrap a new framework.
- [`docs/guides/writing-an-event-sink.md`](docs/guides/writing-an-event-sink.md) — build a custom sink.
- [`docs/guides/choosing-a-sink.md`](docs/guides/choosing-a-sink.md) — decision matrix across the five shipped sinks.
- [`docs/guides/goals-and-plans.md`](docs/guides/goals-and-plans.md) — authoring custom `GoalDeriver` / `Planner`.
- [`docs/guides/persistence-and-recovery.md`](docs/guides/persistence-and-recovery.md) — JSONL + SQLite persistence, `Runner.resume()`.
- [`docs/guides/grpc-transport.md`](docs/guides/grpc-transport.md) — `GRPCSink` + `GoldfiveIngressServer` for out-of-process observers.
- [`docs/guides/harmonograf-integration.md`](docs/guides/harmonograf-integration.md) — plugging harmonograf in as a sink.
- [`docs/guides/troubleshooting.md`](docs/guides/troubleshooting.md) — common setup / runtime failures.

### Reference

- [`docs/reference/api.md`](docs/reference/api.md) — public API surface.
- [`docs/reference/tool-protocol.md`](docs/reference/tool-protocol.md) — the seven reporting tools.

## License

Apache-2.0.
