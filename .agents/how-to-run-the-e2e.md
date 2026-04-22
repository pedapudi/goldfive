---
name: how-to-run-the-e2e
description: How to drive a goldfive run end-to-end locally with harmonograf observability — submodule setup, server spin-up, monitor scripts, success validation.
applies-when: ["run e2e", "local stack", "harmonograf server", "reproduce a bug", "full rollout"]
---

# Run a goldfive end-to-end run locally

This is the canonical loop for reproducing a real goldfive run
against harmonograf — the same loop we used all week to chase
filler-loop regressions. It assumes `~/git/goldfive` and
`~/git/harmonograf` are sibling checkouts.

## One-time setup

### 1. goldfive

```bash
cd ~/git/goldfive
uv sync --extra adk --extra claude --extra proto --extra dev
uv run pytest -q        # sanity check
```

The `adk` extra is required for the ADK examples; `claude` for
the Claude SDK adapter; `proto` for the event protobufs; `dev`
for pytest / ruff / mypy.

### 2. harmonograf

```bash
cd ~/git/harmonograf
git clone https://github.com/google/adk-python.git third_party/adk-python
make install        # installs server + client + frontend deps
```

The `third_party/adk-python` clone is required even for runs that
don't go through ADK — the harmonograf `make install` target
treats it as an editable path dep.

### 3. LLM backend

The canonical e2e uses a local LLM served by kikuchi. Point the
environment at it before running adk-web:

```bash
# kikuchi-hosted Qwen on LAN (adjust host as needed)
export USER_MODEL_NAME=openai/qwen3-coder-30b
export OPENAI_BASE_URL=http://kikuchi:8000/v1
export OPENAI_API_KEY=sk-anything   # kikuchi doesn't check

# or OpenAI cloud
export USER_MODEL_NAME=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...

# planner model (defaults to USER_MODEL_NAME via litellm)
export GOLDFIVE_EXAMPLE_PLANNER_MODEL=gpt-4o-mini
```

`USER_MODEL_NAME` is the env var the reference agents
(`presentation_agent_orchestrated` etc.) read. **Do not** leave it
unset — the default `gemini-2.5-flash` silently breaks without
`GOOGLE_API_KEY` (see
[docs/guides/troubleshooting.md §"Run terminates immediately"](../docs/guides/troubleshooting.md)).

## Driving a run

### The `/tmp/e2e-*.py` pattern

The reproducer scripts land in `/tmp/e2e-<topic>.py` — never
committed, ephemeral, one per investigation. Shape:

```python
# /tmp/e2e-plain-waffles.py — reproducer for issue #XXX
import asyncio, logging

import goldfive
from goldfive.sinks import LoggingSink
from harmonograf_client import HarmonografClient, HarmonografSink
# ... imports for the specific agent you're driving ...

logging.basicConfig(level=logging.INFO)
logging.getLogger("goldfive").setLevel(logging.DEBUG)

async def main():
    # 1. harmonograf client + sink.
    hg = HarmonografClient("localhost:7531")
    hg_sink = HarmonografSink(hg)

    # 2. wrap your agent (ADK / Claude / callable; see
    #    /tmp/e2e-*.py in your shell history for references).
    agent = ...  # your Agent / client factory / callable
    runner = goldfive.wrap(
        agent,
        sinks=[LoggingSink(), hg_sink],
    )

    # 3. or observe() for live-steering as well.
    # runner = hg.observe(runner)

    try:
        outcome = await runner.run("make a presentation about waffles")
    finally:
        await runner.close()
        await hg.close()

    print(f"success={outcome.success} reason={outcome.reason!r}")
    if outcome.session.plan is not None:
        for t in outcome.session.plan.tasks:
            print(f"  {t.status.value:<10} {t.id}  {t.title}")

asyncio.run(main())
```

Why `/tmp/` and not `examples/`: these scripts change shape every
few hours and carry local config. Committing them creates churn;
`examples/` should stay stable and runnable by readers.

### Booting the harmonograf stack

Three terminals for the canonical e2e (server + frontend + adk-web):

```bash
# terminal 1 — gRPC server on :7531 + gRPC-Web on :7532
cd ~/git/harmonograf
make server-run

# terminal 2 — frontend dev server on :5173
cd ~/git/harmonograf/frontend
pnpm dev --port 5173 --strictPort

# terminal 3 — adk-web hosting the reference presentation agent
cd ~/git/harmonograf/.demo-agents
export USER_MODEL_NAME=openai/qwen3-coder-30b   # or OpenAI / Gemini
export OPENAI_BASE_URL=http://kikuchi:8000/v1
export OPENAI_API_KEY=sk-anything
export HARMONOGRAF_SERVER=127.0.0.1:7531
uv run adk web presentation_agent_orchestrated
```

Open `http://127.0.0.1:5173` for the harmonograf UI and
`http://127.0.0.1:8000` for the adk-web UI.

If you want the one-shot demo (server + frontend + adk-web all in
one command), `make demo` in the harmonograf repo does all three;
skip the separate terminals in that case.

### Steer flow: `/tmp/steer.py` + browser-use

The canonical driver for STEER end-to-end testing is a small
script at `/tmp/steer.py` (ephemeral, tuned per session):

```python
# /tmp/steer.py — submit a prompt to adk-web, wait, then STEER
import asyncio, json, urllib.request

def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:8000{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req).read()

async def main():
    # 1. submit initial prompt
    post("/run", {"message": "Make a presentation about waffles"})
    await asyncio.sleep(10)
    # 2. steer mid-run
    post("/steer", {"message": "Switch to raccoons", "author": "e2e"})

asyncio.run(main())
```

Or drive via `mcp__browser-use__` tools from a parent agent to
open the adk-web / harmonograf UIs, click Steer buttons, and
verify PlanRevised events in the UI — same thing, just
higher-fidelity.

### Running the reproducer

```bash
cd ~/git/goldfive
uv run python /tmp/e2e-plain-waffles.py
```

The run writes events to harmonograf as it executes; watch the
Sessions view update live. The stdout of the reproducer gives
you `success` / `reason` / per-task final status for a quick
pass/fail check.

## Monitor scripts

Two patterns show up repeatedly:

### Live tail against a specific run

```python
# /tmp/monitor-latest.py
import asyncio
from harmonograf_client import HarmonografClient

async def main():
    hg = HarmonografClient("localhost:7531")
    sess = await hg.get_latest_session()
    async for event in hg.stream_events(sess.run_id):
        kind = event.WhichOneof("payload")
        print(f"[{event.sequence:4d}] {kind}")
    await hg.close()

asyncio.run(main())
```

### SQLite drill-down after the fact

Harmonograf persists every received event to
`~/git/harmonograf/data/harmonograf.db`. Direct SQL for
post-mortem drilling:

```bash
sqlite3 ~/git/harmonograf/data/harmonograf.db \
  "SELECT sequence, event_type, payload FROM events \
   WHERE run_id = '<run_id>' ORDER BY sequence" \
  | less
```

This is the pattern the postmortem used to reconstruct the
filler-loop sequence events when the UI truncated them.

## Validating success

A run is healthy when **all four** hold:

1. `outcome.success == True`.
2. Every task in `outcome.session.plan.tasks` has status `COMPLETED`.
3. No `DriftDetected` events of severity CRITICAL appear in the
   event stream.
4. The `RunCompleted` event is the last event on the run's stream.

Failure shapes you'll see in this order of likelihood, and what
each means:

- `success=False, reason="orphaned pending tasks after run"` —
  cascade didn't fire cleanly; see PLAN-LIFECYCLE §6.4.
- `success=False, reason="exhausted max_task_invocations=N with pending task ..."` —
  agent stuck in a loop; a finite `max_task_invocations` caught
  it. Drop to the filler-loop playbook
  ([how-to-debug-a-filler-loop.md](how-to-debug-a-filler-loop.md)).
- `success=False, reason="goal '<summary>' unmet"` — the
  `Goal.success_predicate` returned False; the tasks completed
  but the semantic goal wasn't met. #104.
- `success=False, reason="adapter.invoke raised ..."` — the
  underlying framework crashed. Look at the LoggingSink output;
  the traceback is in the preceding ERROR line.
- `success=True` but a task is `PENDING`/`FAILED` — can't happen
  post-#98 / #104; if it does, that's a regression worth filing.

## Tips

- Run with `logging.getLogger("goldfive").setLevel(logging.DEBUG)` —
  every planner call, every refine, every cascade step logs at
  DEBUG. Volume is manageable for a single run.
- Set `GOLDFIVE_EXAMPLE_MODEL=openai/gpt-4o-mini` (or whatever)
  to pin the model in examples that default to something expensive.
- Use `--mock` (in `examples/adk_presentation/agent.py` and
  similar) to run without any network. That isolates goldfive
  orchestration bugs from LLM / API flakiness.
- `outcome.session.plan.tasks` after the run tells you where the
  run ended. `outcome.session.reasoning_history` is the last 20
  reasoning blocks, useful for post-hoc reasoning-drift diagnosis.

## Related

- [docs/guides/observability-with-harmonograf.md](../docs/guides/observability-with-harmonograf.md) — the user-facing end-to-end walkthrough.
- [docs/guides/telemetry-with-harmonograf.md](../docs/guides/telemetry-with-harmonograf.md) — deriving insight from the UI.
- [docs/guides/harmonograf-integration.md](../docs/guides/harmonograf-integration.md) — the sink wiring and protocol.
- [docs/guides/common-failure-modes.md](../docs/guides/common-failure-modes.md) — catalog of failure shapes.
- [how-to-debug-a-filler-loop.md](how-to-debug-a-filler-loop.md) — what to do when the run doesn't progress.
- [debug-goldfive.md](debug-goldfive.md) — the triage tree.
