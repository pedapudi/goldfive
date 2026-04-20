# Multi-agent presentation reference

A production-shaped goldfive + ADK example: a coordinator plus four
specialists (research, web_developer, reviewer, debugger) with real
`write_webpage` / `read_presentation_files` / `patch_file` tools. The
coordinator tree is plain ADK; goldfive handles planning, task dispatch,
drift detection, and steering. Optional harmonograf telemetry watches
the run.

This tree is the canonical source for harmonograf's
`tests/reference_agents/presentation_agent/` — harmonograf re-exports
these agents so the two repos share a single definition.

For the minimal "how do I wrap a single agent" lesson, see
[`../adk_presentation`](../adk_presentation/). This example assumes you
already understand that pattern.

## Running

### Mock mode — no credentials

```bash
uv pip install -e '.[adk]'
uv run python examples/presentation_agent/agent.py --mock
```

Canned planner + goal-deriver + in-process `_MockLlm` for every
subagent. Walks four sequential tasks (research → build → review →
debug) and exits with `success=True`.

### Live mode — OpenAI

```bash
uv pip install -e '.[adk]'
pip install openai litellm
export OPENAI_API_KEY=sk-...
uv run python examples/presentation_agent/agent.py --topic "the Voyager missions"
```

The subagents hit `openai/gpt-4o-mini` (override with
`GOLDFIVE_EXAMPLE_MODEL`); the planner + goal-deriver use the `openai`
SDK directly (override with `GOLDFIVE_EXAMPLE_PLANNER_MODEL`). The
`web_developer_agent` writes files under `output/<topic>/`.

### Under `adk web`

```bash
uv pip install -e '.[adk]'
adk web examples/presentation_agent
```

The module exposes `app` lazily (PEP 562 `__getattr__`), so importing
it is side-effect free. On first access `app` builds a
`goldfive.wrap(root_agent, ...)` root agent and hands it to ADK's
`App`. If `OPENAI_API_KEY` is unset the `App` falls back to mock mode
so `adk web` still loads offline.

### With harmonograf telemetry

```bash
export HARMONOGRAF_SERVER=127.0.0.1:7531
uv run python examples/presentation_agent/agent.py --topic waffles
```

Attaches a `HarmonografSink` to the runner and a
`HarmonografTelemetryPlugin` to the `App` so per-span TOOL_CALL /
LLM_CALL events land in the harmonograf UI. Install harmonograf's
client extra (`pip install harmonograf-client`) first. Both hooks are
optional — unset `HARMONOGRAF_SERVER` or skip the package and the
example runs silently.

## Flags

| Flag | Effect |
|---|---|
| `--topic TEXT` | Override the presentation topic. Default: `waffles`. |
| `--mock` | Replace every LLM with an in-process mock. No network. |
| `--verbose`, `-v` | INFO-level logs from goldfive sinks. |
