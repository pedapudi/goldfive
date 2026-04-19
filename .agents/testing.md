---
name: testing
description: How to write a goldfive test — pytest-asyncio patterns, CallableAdapter harness, optional-extra skips.
applies-when: ["write a test", "test goldfive", "pytest"]
---

# Testing

goldfive's test suite lives under `tests/` and runs in ~7 seconds.

## Setup

```bash
uv sync --extra dev
uv run pytest -q
```

`pytest-asyncio` is on in auto mode (`asyncio_mode = "auto"` in
`pyproject.toml`). `async def test_*` works without a decorator.

## Where tests live

- `tests/test_runner.py`, `tests/test_runner_integration.py` — end-to-end.
- `tests/test_<component>.py` — one file per component
  (`test_planner.py`, `test_steerer.py`, `test_sequential_executor.py`, ...).
- `tests/test_<adapter>_adapter.py` — adapter tests.
- `tests/test_<sink>.py` — sink tests.
- `tests/_pbsetup.py`, `tests/conftest.py` — shared fixtures / proto
  bootstrap.
- `tests/test_smoke.py` — top-level import / re-export smoke tests.

## The canonical integration test pattern

Every integration test uses `CallableAdapter` to remove LLM
non-determinism:

```python
from __future__ import annotations

import pytest

from goldfive import (
    CallableAdapter, InMemorySink, InvocationResult, Plan, ReportingToolSpec,
    Runner, SequentialExecutor, Session, StaticPlanner, Task, TaskEdge,
)


async def test_runner_completes_linear_plan() -> None:
    async def agent(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        return InvocationResult(task_id=task.id, text=task.title)

    plan = Plan(
        id="p", run_id="", goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="a", assignee_agent_id="worker"),
            Task(id="t2", title="b", assignee_agent_id="worker"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        summary="",
    )

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(agent, available_agents=["worker"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        sinks=[sink],
    )
    outcome = await runner.run("go")
    await runner.close()

    assert outcome.success
    kinds = [e.WhichOneof("payload") for e in sink.events if hasattr(e, "DESCRIPTOR")]
    assert "task_completed" in kinds
```

## Optional-extras tests

Tests that need `google-adk`, `anthropic`, or the `proto` extra must
`pytest.importorskip` so the suite stays green on a minimal install:

```python
import pytest

pytest.importorskip("google.adk")

from goldfive.adapters.adk import ADKAdapter  # noqa: E402

async def test_adk_adapter_wraps_root_agent() -> None:
    ...
```

For sinks that require `proto`:

```python
from goldfive.sinks import JSONLPersistenceSink

pytestmark = pytest.mark.skipif(
    JSONLPersistenceSink is None, reason="goldfive[proto] not installed"
)
```

## Fixtures

Most tests don't need fixtures beyond what's in `tests/conftest.py`.
When you do, keep them local to the test module — the suite avoids
shared mutable state across files.

## Running subsets

```bash
uv run pytest tests/test_runner.py              # one file
uv run pytest tests/test_runner.py::test_x      # one test
uv run pytest -k drift                          # by keyword
uv run pytest -q --lf                           # last-failed
```

## Deterministic LLM stubs

When a component takes `call_llm`, inject a pure-Python stub:

```python
async def fake_call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    return '{"tasks": [{"id": "t1", "title": "do it"}], "summary": ""}'

from goldfive.planner import LLMPlanner
planner = LLMPlanner(call_llm=fake_call_llm, model="stub")
```

## Quick reference

```bash
uv run pytest -q                        # full suite
uv run pytest tests/test_runner.py -v   # one module, verbose
uv run ruff check . && uv run ruff format --check .
```

## Common pitfalls

- Forgetting `await runner.close()` in a test → buffered sinks (none
  in the default set, but e.g. `GRPCSink`) drop events and the test
  flakes.
- Using the `@pytest.mark.asyncio` decorator — unnecessary in auto
  mode and inconsistent with the existing suite.
- Tests that depend on wall-clock time. Mock `time.time` / `time_ns`
  or assert on relative ordering.
- Hitting real networks or LLMs. Always use `CallableAdapter` +
  stubbed `call_llm` in tests.
- Expecting every event to be proto — current `main` is (PR #55), but
  legacy callers produce dict envelopes from `goldfive.events.make_event`.
  Duck-type on `DESCRIPTOR` if you assert across both.

## Related

- [develop-goldfive.md](develop-goldfive.md) — dev loop and merge flow.
- [adapters.md](adapters.md) / [sinks.md](sinks.md) — what you're testing.
- [docs/design/EVENT-MODEL.md](../docs/design/EVENT-MODEL.md) — event shapes.
