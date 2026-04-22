# Multi-agent presentation reference

A production-shaped goldfive + ADK example: a coordinator plus four
specialists (research, web_developer, reviewer, debugger) with real
`write_webpage` / `read_presentation_files` / `patch_file` tools. The
coordinator tree is plain ADK; goldfive handles goal derivation,
planning, overlay-mode dispatch, drift detection, and steering.
Optional harmonograf telemetry watches the run per-agent.

This tree is the canonical source for harmonograf's
`tests/reference_agents/presentation_agent/` — harmonograf re-exports
these agents so the two repos share a single definition.

For the minimal "how do I wrap a single agent" lesson, see
[`../adk_presentation`](../adk_presentation/). This example assumes
you already understand that pattern.

## What this demonstrates

- **`goldfive.wrap(root_agent)` on a full tree.** One
  `InMemoryRunner` around the coordinator; ADK's `AgentTool`
  delegation routes to specialists; goldfive observes via the
  overlay `PlanReconciler` (goldfive#141).
- **`GoldfivePlanner(BasePlanner)` auto-attached to every
  `LlmAgent`** (goldfive#153). Per-turn orchestration context block
  (current task id/title, goals, active steer, cancelled
  function-call ids) is injected into each LLM's system
  instruction.
- **`HarmonografTelemetryPlugin` on the `App`.** Per-agent spans
  land in the harmonograf Gantt (one row per sub-agent, goldfive#170
  + harmonograf#80).
- **`HarmonografSink` on the runner.** Goldfive events (run / plan
  / task / drift / PlanRevised) stream to harmonograf.
- **Mock mode for CI.** Zero network; canned planner / goal-deriver
  / `_MockLlm`.

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
`GOLDFIVE_EXAMPLE_MODEL`); the planner + goal-deriver use the
`openai` SDK directly (override with
`GOLDFIVE_EXAMPLE_PLANNER_MODEL`). The `web_developer_agent` writes
files under `output/<topic>/`.

### Live mode — local LLM via kikuchi / Qwen

```bash
export GOLDFIVE_EXAMPLE_MODEL=openai/qwen3-coder-30b
export OPENAI_BASE_URL=http://kikuchi:8000/v1
export OPENAI_API_KEY=sk-anything   # not validated
export GOLDFIVE_EXAMPLE_PLANNER_MODEL=qwen3-coder-30b
uv run python examples/presentation_agent/agent.py --topic waffles
```

Qwen sometimes hallucinates `write_webpage` success without calling
the tool. The goldfive `CONFABULATION_RISK` classifier catches this
at INFO severity — record-only, does not trigger refine. See
[docs/guides/common-failure-modes.md §7](../../docs/guides/common-failure-modes.md).

### Under `adk web`

```bash
uv pip install -e '.[adk]'
export USER_MODEL_NAME=openai/gpt-4o-mini  # or qwen, etc.
export OPENAI_API_KEY=sk-...
adk web examples/presentation_agent
```

The module exposes `app` lazily (PEP 562 `__getattr__`), so
importing it is side-effect free. On first access `app` builds the
goldfive-wrapped tree and hands it to ADK's `App`. If
`OPENAI_API_KEY` is unset the `App` falls back to mock mode so `adk
web` still loads offline.

**Don't skip `USER_MODEL_NAME`** when hitting a non-Gemini backend.
The default is `gemini-2.5-flash`; without `GOOGLE_API_KEY` every
run terminates instantly with "goldfive run complete." and an
`AttributeError: '_async_httpx_client'` at teardown. See
[docs/guides/troubleshooting.md](../../docs/guides/troubleshooting.md).

### With harmonograf telemetry

```bash
export HARMONOGRAF_SERVER=127.0.0.1:7531
uv run python examples/presentation_agent/agent.py --topic waffles
```

Attaches a `HarmonografSink` to the runner and a
`HarmonografTelemetryPlugin` to the `App` so per-span INVOCATION /
LLM_CALL / TOOL_CALL spans land in harmonograf with one Gantt row
per sub-agent. Install harmonograf's client extra
(`pip install harmonograf-client`) first. Both hooks are optional —
unset `HARMONOGRAF_SERVER` or skip the package and the example runs
silently.

## What to expect in the harmonograf UI

- One session row per run (session id pinned to the outer adk-web
  session, goldfive#161).
- Plan: four tasks (research / build / review / debug) in a
  sequential DAG. Each transitions PENDING → RUNNING → COMPLETED
  as the overlay reconciler credits tree invocations to tasks.
- Agents timeline: one Gantt row per sub-agent, children showing
  LLM_CALL and TOOL_CALL spans. AgentTool delegations show as
  dashed edges.
- Drifts panel: on a clean run, empty. Under Qwen, occasional
  `CONFABULATION_RISK` / INFO.
- Plan revisions panel: empty on a clean run; populated if you
  click Steer in the UI.

## Flags

| Flag | Effect |
|---|---|
| `--topic TEXT` | Override the presentation topic. Default: `waffles`. |
| `--mock` | Replace every LLM with an in-process mock. No network. |
| `--verbose`, `-v` | INFO-level logs from goldfive sinks. |
