# Minimal goldfive + ADK example

The smallest possible demo showing how `goldfive.wrap(agent, ...)`
plugs a plain Google ADK agent into goldfive's planner + reconciler
+ steerer stack.

```python
agent  = <your ADK Agent>                 # any BaseAgent
runner = goldfive.wrap(                   # goldfive handles decomposition,
    agent,                                # overlay dispatch, drift, and steering
    planner=LLMPlanner(call_llm=...),
    goal_deriver=LLMGoalDeriver(call_llm=...),
)
await runner.run("make a presentation about waffles")
```

One agent, no coordinator, no hand-rolled delegation. `goldfive.wrap`
returns a `GoldfiveADKAgent` — a `BaseAgent` subclass that also
exposes `Runner.run`, so the same object works under `adk web` and
programmatically.

## What this demonstrates

- **One ADK `Agent`, goldfive does the rest.** The planner emits a
  task DAG; the overlay reconciler observes the agent running via
  `before_agent` / `after_agent` callbacks and maps invocations to
  plan tasks. No subagent tree, no `AgentTool`s, no coordinator with
  hardcoded routing.
- **`GoldfivePlanner` auto-attached** to the agent's `LlmAgent`
  (goldfive#153). A per-turn orchestration context block is injected
  into the LLM's system instruction so the agent reads the current
  task from `session.state['goldfive.*']` without the caller wiring
  prompts manually.
- **Planner + goal deriver are pluggable.** Wire any model provider
  behind a `call_llm` callable with signature
  `(system_prompt, user_prompt, model) -> str`.
- **Events per task.** Goldfive's executor emits `TaskStarted` /
  `TaskCompleted` for each planner-generated task. Drop in a
  `HarmonografSink` to watch the run live in harmonograf's UI.

## Running

### Mock mode — no credentials

```bash
uv pip install -e '.[adk]'
uv run python examples/adk_presentation/agent.py --mock
```

Canned planner / goal deriver + in-process `_MockLlm` for the agent.
The run walks three sequential tasks (research → outline → writeup) and
exits with `success=True`.

### Live mode — OpenAI

```bash
uv pip install -e '.[adk]'
pip install openai litellm
export OPENAI_API_KEY=sk-...
uv run python examples/adk_presentation/agent.py --topic "the Voyager missions"
```

Live mode uses the `openai` SDK for planner + goal deriver and a LiteLLM
model string (default `openai/gpt-4o-mini`) for the ADK agent. Overrides:

| Env var | What it controls |
|---|---|
| `GOLDFIVE_EXAMPLE_MODEL` | ADK agent model (LiteLLM format) |
| `GOLDFIVE_EXAMPLE_PLANNER_MODEL` | Planner `call_llm` model id |

## Flags

| Flag | Effect |
|---|---|
| `--mock` | Replace every LLM with an in-process mock. No network. |
| `--topic TEXT` | Override the presentation topic. Default: `waffles`. |
| `--verbose`, `-v` | INFO-level logs from goldfive's `LoggingSink`. |

## A richer example

For a production-shaped multi-agent ADK tree (coordinator + research /
developer / reviewer / debugger specialists with real file tools) see
**harmonograf's `presentation_agent`** at
`tests/reference_agents/presentation_agent/agent.py` in the
[harmonograf repo](https://github.com/pedapudi/harmonograf). That module
already exports a `build_goldfive_runner` helper that assembles the same
goldfive `Runner` shape as this example, but over the full subagent
tree, so you can compare the two side-by-side.

The split is deliberate: this repo's example shows you *how to wrap*;
harmonograf's example shows you *how to structure a real multi-agent
ADK tree*. The two concerns compose but live in different files so
neither obscures the other.
