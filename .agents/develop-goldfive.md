---
name: develop-goldfive
description: Add a feature, adapter, sink, or bug fix to goldfive itself — branch, test, PR, merge loop.
applies-when: ["develop goldfive", "add a feature", "add an adapter", "contribute to goldfive"]
---

# Develop goldfive

You're changing goldfive itself. This skill covers the repo layout,
dev loop, and merge flow.

## Read first (before any nontrivial edit)

- **[docs/dev-guide/17-invariants-hazards-history.md](../docs/dev-guide/17-invariants-hazards-history.md)** —
  the six hard invariants, the Protected List (KEEP decisions you must
  not "clean up" without sign-off), the Deferred-Work Register, the
  hazard catalog, and the pre-PR checklist. This is the constitution;
  read it before you touch the observe/detect/intervene pipeline.
- **[docs/dev-guide/16-recipes.md](../docs/dev-guide/16-recipes.md)** —
  twelve copy-pasteable end-to-end procedures (add a DriftKind, detector,
  judge, intervention surface, config knob, proto field, sink,
  reporting tool, adapter; safe dead-code deletion; doc update). If your
  task matches a recipe, follow it.
- The full guide index with a task→chapter routing table is
  [docs/dev-guide/00-index.md](../docs/dev-guide/00-index.md).

**Mode discipline (post-#488):** the suite runs the shipped
`observation_only=True` default — there is no autouse fixture that flips
it. A test that needs active steering must construct the steerer in
active mode explicitly, and any intervention feature must be tested in
BOTH modes (passive: no wire action; active: the intervention fires).
The only sanctioned kill-switch read is
`DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)`
(missing/None/raising → PASSIVE). See
[docs/dev-guide/15-testing-guide.md](../docs/dev-guide/15-testing-guide.md).

**Pre-PR:** run the pre-PR checklist in chapter 17 §6 (invariants sweep,
hazard sweep, mechanical hygiene) before opening the PR.

## Repo layout

```
goldfive/
  adapters/     # CallableAdapter, ADKAdapter, ClaudeAgentSDKAdapter
  executors/    # SequentialExecutor, ParallelDAGExecutor
  sinks/        # InMemory, Logging, JSONL, SQLite, gRPC
  server/       # gRPC ingress server
  pb/           # generated proto stubs (do not hand-edit)
  runner.py     # the one public entrypoint
  planner.py    # Static, Passthrough, LLM planners
  steerer.py    # DefaultSteerer — state machine + drift detection
  goal_deriver.py
  drift.py      # classifiers + DriftKind/DriftSeverity re-export
  events.py     # typed event factories (proto path)
  reporting.py  # the seven canonical reporting tools
  protocols.py  # six Protocol interfaces
  types.py      # dataclasses (Goal, Plan, Task, Session, DriftEvent, ...)
  conv.py       # dataclass <-> proto converters
proto/          # .proto sources; `make proto` regenerates goldfive/pb/
tests/          # pytest-asyncio; see testing.md
examples/       # runnable end-to-end scripts
docs/           # prose (design/, guides/, reference/)
bench/          # perf benchmarks
```

## First-time setup

```bash
git clone git@github.com:pedapudi/goldfive.git
cd goldfive
uv sync --extra dev
uv run pytest -q
```

## Dev loop

```bash
uv run pytest -q                    # full test suite (~7s)
uv run pytest tests/test_runner.py  # target one module
uv run ruff check .                 # lint
uv run ruff format .                # auto-format
```

Tests use `pytest-asyncio` in auto mode (see `[tool.pytest.ini_options]`
in `pyproject.toml`) — `async def test_*` works out of the box.

## Adding a feature

1. **Branch off current `main`.**
   ```bash
   git fetch origin && git checkout -B my-feature origin/main
   ```
2. **Implement.** Keep the public surface narrow — if you add a public
   class, wire it into `goldfive/__init__.py` and the `__all__` list.
3. **Test.** Add tests under `tests/`. See [testing.md](testing.md).
4. **Lint.** `uv run ruff check .` and `uv run ruff format .`.
5. **Docs.** If the change is user-facing, update the relevant
   `docs/guides/*.md` and `docs/reference/api.md`.
6. **PR.** `gh pr create` with a summary and a test plan.

## Regenerating proto

After editing anything under `proto/`:

```bash
make proto
```

This runs `grpc_tools.protoc` via `uv run --extra proto` and writes
stubs into `goldfive/pb/goldfive/v1/`. Commit the regenerated stubs.

## Working with adapters, sinks, events

- Adding an adapter → [adapters.md](adapters.md).
- Adding a sink → [sinks.md](sinks.md).
- Adding an event or a new event factory → [events.md](events.md).

## Conventions

- **`from __future__ import annotations`** at the top of every file.
- **Async-native** everywhere in the core pipeline. Sync code is
  suspicious in `runner.py`, `steerer.py`, executors, adapters, sinks.
- **Line length 100** (ruff-enforced).
- **No emojis.** Not in code, docstrings, docs, commits, or PR bodies.
- **Dataclasses**, not proto messages, are what callers pass around.
  `goldfive.conv.to_pb_*` / `from_pb_*` bridges the two.
- **No ADK or Claude SDK imports** outside `goldfive/adapters/adk.py`
  and `goldfive/adapters/claude.py`. Lazy-import in the adapter module.
- **Terse docstrings.** One-line summary plus a short block if the
  behaviour isn't obvious.
- **No trailing-summary comments** that restate the diff.

## Merging

`main` is the single long-lived branch. PRs merge via
`gh pr merge --admin --squash --delete-branch` once CI is green and
review is complete.

## Quick reference

```bash
# start a feature
git fetch origin && git checkout -B feat/my-thing origin/main

# tight loop
uv run pytest -q && uv run ruff check .

# proto
make proto

# push + PR
git push -u origin feat/my-thing
gh pr create --title "Add X" --body "..."

# merge (after review / CI green)
gh pr merge --admin --squash --delete-branch
```

## Common pitfalls

- Editing `goldfive/pb/` by hand. These are generated — change
  `proto/` and re-run `make proto`.
- Adding a public symbol but forgetting to re-export from
  `goldfive/__init__.py`. The next consistency audit will revert it.
- Tests that depend on real LLMs / network. Use `CallableAdapter` or
  a mock `call_llm` to stay deterministic.
- Dropping `from __future__ import annotations` in a new module —
  breaks forward-reference typing on Python 3.11.

## Related

- [testing.md](testing.md) — test patterns and fixtures.
- [adapters.md](adapters.md) / [sinks.md](sinks.md) / [events.md](events.md).
- [release.md](release.md) — cutting a release.
- [docs/design/ARCHITECTURE.md](../docs/design/ARCHITECTURE.md) — why things are shaped this way.
