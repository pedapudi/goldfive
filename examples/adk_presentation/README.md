# ADK presentation example

A reference implementation showing how `goldfive.Runner` wraps a multi-subagent
Google ADK tree. Ported from harmonograf's `presentation_agent` demo, with the
orchestration layer swapped from `HarmonografAgent` to
`goldfive.adapters.adk.ADKAdapter`.

## What this demonstrates

- **A coordinator that delegates to four specialists** — researcher, web
  developer, reviewer, and debugger — via `AgentTool`.
- **One wrap covers the whole tree.** Only the root `coordinator_agent`
  is handed to `ADKAdapter(...)`. The adapter walks the tree (`sub_agents`,
  `inner_agent`, and nested `AgentTool.agent`) and attaches goldfive's
  seven canonical reporting tools to every subagent automatically.
- **Planner + goal deriver are pluggable.** The example wires
  `LLMPlanner` and `LLMGoalDeriver` behind a `call_llm` callable so you
  can swap in any model provider without touching goldfive internals.
- **Events per subagent.** Goldfive's executor emits `TaskStarted` /
  `TaskCompleted` events for each of the four specialist tasks, even
  though the adapter only invokes the coordinator root.

## Agent tree

```
coordinator_agent
├── research_agent         (AgentTool)
├── web_developer_agent    (AgentTool, tool: write_webpage)
├── reviewer_agent         (AgentTool, tool: read_presentation_files)
└── debugger_agent         (AgentTool, tool: patch_file)
```

## Running

### Mock mode — no credentials required

```bash
uv pip install -e '.[adk]'
uv run python examples/adk_presentation/agent.py --mock
```

In this mode every ADK agent's model is a deterministic in-process
`BaseLlm` subclass, and the planner / goal deriver use canned JSON. The
run completes with `success=True` and emits a 13-event stream
culminating in `run_completed`.

### Live mode — OpenAI

```bash
uv pip install -e '.[adk]'
pip install openai litellm
export OPENAI_API_KEY=sk-...
uv run python examples/adk_presentation/agent.py --topic "the Voyager missions"
```

Live mode uses the `openai` Python SDK for the planner / goal deriver
and a LiteLLM model string (default `openai/gpt-4o-mini`) for the ADK
subagents. Override the models via:

- `GOLDFIVE_EXAMPLE_MODEL` — ADK agent model (LiteLLM format, e.g.
  `openai/gpt-4o-mini`).
- `GOLDFIVE_EXAMPLE_PLANNER_MODEL` — model id passed to the planner's
  `call_llm`. Defaults to `gpt-4o-mini`.
- `GOLDFIVE_EXAMPLE_GOAL_MODEL` — model id for the goal deriver.
  Defaults to `gpt-4o-mini`.

Generated slideshow files land under
`examples/adk_presentation/output/<topic>/` (`index.html`,
`styles.css`, `script.js`).

## Flags

| Flag | Effect |
|---|---|
| `--mock` | Replace every LLM with an in-process mock. No network. |
| `--topic TEXT` | Override the presentation topic. |
| `--verbose`, `-v` | Enable INFO logging from goldfive's `LoggingSink`. |
