"""Shared fixtures for the goldfive test suite.

These fixtures are deliberately defensive: they import optional goldfive
submodules lazily so that individual test files can `pytest.importorskip`
them when a feature PR has not yet landed.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# stub_call_llm factory
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_call_llm() -> Callable[..., Callable[..., Any]]:
    """Return a factory that builds an async ``call_llm`` stub.

    The factory takes an iterable of canned responses (strings or dicts)
    and returns an async function that pops the next response on each call.
    Dicts are JSON-encoded so callers that expect a raw string get one,
    while callers that expect structured data can json.loads() the result.
    """

    def _factory(responses: Iterable[Any]) -> Callable[..., Any]:
        queue: list[Any] = list(responses)

        async def _call_llm(*args: Any, **kwargs: Any) -> str:
            if not queue:
                raise AssertionError("stub_call_llm exhausted; no more canned responses")
            resp = queue.pop(0)
            if isinstance(resp, (dict, list)):
                return json.dumps(resp)
            return str(resp)

        _call_llm.remaining = queue  # type: ignore[attr-defined]
        return _call_llm

    return _factory


# ---------------------------------------------------------------------------
# session_factory
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory() -> Callable[..., Any]:
    """Return a callable that builds a fresh ``Session`` with sensible defaults.

    Skips the test if ``goldfive.types`` is not importable yet.
    """

    types = pytest.importorskip("goldfive.types")

    def _factory(
        *,
        run_id: str | None = None,
        goals: list[Any] | None = None,
        plan: Any | None = None,
    ) -> Any:
        session = types.Session(
            run_id=run_id or f"run-{uuid.uuid4().hex[:8]}",
            goals=list(goals or []),
            plan=plan,
        )
        return session

    return _factory


# ---------------------------------------------------------------------------
# in_memory_runner
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_runner() -> Callable[..., Any]:
    """Build a ``Runner`` wired with in-memory/no-op collaborators.

    The returned factory accepts keyword overrides so individual tests can
    replace the adapter, planner, steerer, or sinks. The test is skipped
    if any required goldfive module has not yet been implemented.
    """

    runner_mod = pytest.importorskip("goldfive.runner")
    planners_mod = pytest.importorskip("goldfive.planners")
    executors_mod = pytest.importorskip("goldfive.executors")
    adapters_mod = pytest.importorskip("goldfive.adapters.callable")
    steerers_mod = pytest.importorskip("goldfive.steerers")
    sinks_mod = pytest.importorskip("goldfive.sinks")

    def _factory(**overrides: Any) -> Any:
        adapter = overrides.pop(
            "adapter",
            adapters_mod.CallableAdapter(handlers={}, available_agents=["default"]),
        )
        planner = overrides.pop("planner", planners_mod.PassthroughPlanner())
        executor = overrides.pop("executor", executors_mod.SequentialExecutor())
        steerer = overrides.pop("steerer", steerers_mod.DefaultSteerer())
        sinks = overrides.pop("sinks", [sinks_mod.InMemorySink()])
        return runner_mod.Runner(
            agent=adapter,
            planner=planner,
            executor=executor,
            steerer=steerer,
            sinks=sinks,
            **overrides,
        )

    return _factory


# ---------------------------------------------------------------------------
# tmp_jsonl_path
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_jsonl_path(tmp_path):
    """Return a pathlib.Path to a .jsonl file inside pytest tmp_path."""
    return tmp_path / "events.jsonl"
