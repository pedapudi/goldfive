"""Shared fixtures for the goldfive test suite.

These fixtures are deliberately defensive: they import optional goldfive
submodules lazily so that individual test files can `pytest.importorskip`
them when a feature PR has not yet landed.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# State-ownership audit (goldfive#271 Phase 0)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _state_audit_enabled() -> Iterator[None]:
    """Auto-applied fixture — enable the state-ownership tripwire in tests.

    See ``docs/design/STATE-OWNERSHIP-CONTRACT.md`` §7 for the
    contract. The tripwire is off in production by default; in tests
    we flip it on so any new ADK-state mutation from inside a goldfive
    callback raises :class:`StateOwnershipViolation`.

    Tests that need to drive a deliberate violation use
    ``goldfive._state_audit.expect_violation(reason)`` to suppress
    the guard for a specific block.

    Tests that want to disable the audit entirely can override this
    fixture or use the ``no_state_audit`` fixture below.
    """
    prior = os.environ.get("GOLDFIVE_STRICT_STATE_OWNERSHIP")
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("GOLDFIVE_STRICT_STATE_OWNERSHIP", None)
        else:
            os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = prior


@pytest.fixture
def no_state_audit() -> Iterator[None]:
    """Disable the state-ownership tripwire for the wrapped test.

    Used by the audit module's own tests to demonstrate the off-state
    sanity check (catalogued + un-catalogued writes both pass through
    when the tripwire is disabled).
    """
    prior = os.environ.get("GOLDFIVE_STRICT_STATE_OWNERSHIP")
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "0"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("GOLDFIVE_STRICT_STATE_OWNERSHIP", None)
        else:
            os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = prior

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
