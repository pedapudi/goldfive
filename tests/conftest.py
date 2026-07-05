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


@pytest.fixture(autouse=True)
def _goldfive_active_steering_default() -> Iterator[None]:
    """Flip :class:`SteeringConfig.observation_only`'s implicit default to
    ``False`` for the test suite (goldfive#254).

    Production default is ``True`` (passive observation — see the
    docstring on :class:`goldfive.config.SteeringConfig`). The existing
    test corpus was written against the prior active-steering default
    and asserts e.g. ``session.plan`` mutating after a refine; flipping
    every test to construct a ``RuntimeConfig`` explicitly is overkill.
    This fixture sets a module-level test-only override
    (``goldfive.config._OBSERVATION_ONLY_DEFAULT``) that the
    ``observation_only`` field consults via a ``default_factory`` — so
    any test that constructs ``SteeringConfig()`` (or builds a
    ``DefaultSteerer`` without a config) sees active steering.

    Tests that explicitly pass ``observation_only=True`` (or
    ``observation_only=False``) still win — the dataclass honours
    explicit kwargs over the default factory, and a
    ``SteeringConfig(observation_only=True)`` instance retains the
    True the test asked for regardless of this fixture.
    """
    try:
        from goldfive import config as _gf_config
    except ImportError:
        # Config module not importable yet — no-op for back-compat with
        # pre-#225 worktrees.
        yield
        return
    prior = _gf_config._OBSERVATION_ONLY_DEFAULT
    _gf_config._OBSERVATION_ONLY_DEFAULT = False
    try:
        yield
    finally:
        _gf_config._OBSERVATION_ONLY_DEFAULT = prior


@pytest.fixture(autouse=True)
def _isolate_orchestration_store_registries() -> Iterator[None]:
    """Clear StateStore's per-session registries between tests.

    ``_ACTIVE_INVOCATION_TASKS`` and ``_CANCEL_REQUESTED_INVOCATIONS``
    are module-level dicts in ``goldfive.state_store`` keyed
    by ``session.id`` (which is a property aliased to ``run_id``).
    Many tests share ``run_id="r1"``, so a test that calls
    ``request_invocation_cancel`` or ``register_invocation_task``
    without explicit cleanup leaks its bucket into the next test's
    session and silently flips the late-drift gate
    (``_is_late_drift_for_terminated_invocation``) — manifesting as a
    flaky CI-only failure of e.g.
    ``test_judge_verdict_with_live_invocation_proceeds_normally``
    where ``planner.refine`` is unexpectedly skipped because a stale
    cancel-pending entry from an earlier test makes the gate think
    the verdict is late.

    Clearing both dicts before and after every test makes test
    ordering irrelevant. Cheap (dict.clear() under the existing
    module-level lock) and idempotent.
    """
    try:
        from goldfive.state_store import (
            _ACTIVE_INVOCATION_LOCK,
            _ACTIVE_INVOCATION_TASKS,
            _CANCEL_REQUESTED_INVOCATIONS,
        )
    except ImportError:
        # Module not importable yet (pre-Phase-1 worktree) — degrade
        # to a no-op so this fixture doesn't block other tests.
        yield
        return
    with _ACTIVE_INVOCATION_LOCK:
        _ACTIVE_INVOCATION_TASKS.clear()
        _CANCEL_REQUESTED_INVOCATIONS.clear()
    try:
        yield
    finally:
        with _ACTIVE_INVOCATION_LOCK:
            _ACTIVE_INVOCATION_TASKS.clear()
            _CANCEL_REQUESTED_INVOCATIONS.clear()

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


# ---------------------------------------------------------------------------
# Env-var fixtures (Wave C bucket 3 cleanup)
# ---------------------------------------------------------------------------
#
# Each ``GOLDFIVE_*`` (and the small ``OPENAI_*``) env-var family that the
# config / from-env tests poke gets a dedicated fixture below. The fixtures
# are thin wrappers over ``pytest.MonkeyPatch.setenv`` / ``delenv`` — they
# do NOT touch ``os.environ`` directly, so the standard pytest
# monkeypatch-teardown still owns the cleanup. The point of the fixture
# is to (a) eliminate per-test boilerplate, (b) make the env surface a
# test consumes visible from the fixture name (``goldfive_agent_env``
# vs ``goldfive_embedding_env`` etc.), and (c) avoid each test inlining
# the variable-name string.
#
# A fixture is named after the **config domain** rather than the env
# var: callers ask for ``goldfive_steer_env`` and set
# ``observation_only="1"`` — never the underlying
# ``GOLDFIVE_STEER_OBSERVATION_ONLY`` string. The mapping is owned by
# the fixture; callers must not bypass it.
#
# Reset semantics: each fixture pre-clears its variables via
# ``delenv(..., raising=False)`` before yielding so tests start from a
# guaranteed clean slate. Teardown is automatic via the underlying
# monkeypatch fixture.


class _EnvController:
    """Thin controller backing the per-domain env-var fixtures.

    Construction takes an explicit allow-list of env-var names: the
    controller refuses ``set`` / ``unset`` calls for any other name so
    a fixture's tests cannot accidentally poke a sibling family's
    variables and confuse the reader of the fixture name.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        names: dict[str, str],
    ) -> None:
        self._monkeypatch = monkeypatch
        # ``names`` maps a short kwarg-style key (``base_url``) to the
        # underlying env-var name (``GOLDFIVE_EMBEDDING_BASE_URL``).
        self._names = dict(names)

    def _resolve(self, key: str) -> str:
        if key not in self._names:
            raise KeyError(
                f"{key!r} is not a recognised env-var key for this "
                f"fixture; expected one of {sorted(self._names)}"
            )
        return self._names[key]

    def set(self, **kwargs: str) -> None:
        """Set one or more env vars by short key.

        Values are str-coerced so callers can pass ``1`` / ``True`` /
        floats without ceremony.
        """
        for key, value in kwargs.items():
            self._monkeypatch.setenv(self._resolve(key), str(value))

    def unset(self, *keys: str) -> None:
        """Delete one or more env vars (no-op if absent)."""
        for key in keys:
            self._monkeypatch.delenv(self._resolve(key), raising=False)

    def clear(self) -> None:
        """Delete every env var owned by this fixture."""
        for env_name in self._names.values():
            self._monkeypatch.delenv(env_name, raising=False)

    def raw_setenv(self, env_name: str, value: str) -> None:
        """Escape-hatch for tests that need to set a variant string
        directly (e.g. the case-insensitivity / whitespace tests).

        Only accepts env-var names this controller owns.
        """
        if env_name not in self._names.values():
            raise KeyError(
                f"{env_name!r} is not owned by this fixture; "
                f"owned names: {sorted(self._names.values())}"
            )
        self._monkeypatch.setenv(env_name, value)


_AGENT_ENV: dict[str, str] = {
    "max_output_tokens": "GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS",
    "call_timeout_ms": "GOLDFIVE_AGENT_CALL_TIMEOUT_MS",
}

_EMBEDDING_ENV: dict[str, str] = {
    "base_url": "GOLDFIVE_EMBEDDING_BASE_URL",
    "model": "GOLDFIVE_EMBEDDING_MODEL",
    "api_key": "GOLDFIVE_EMBEDDING_API_KEY",
    "timeout_ms": "GOLDFIVE_EMBEDDING_TIMEOUT_MS",
    "breaker_cooldown_s": "GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S",
}

_STEER_ENV: dict[str, str] = {
    "observation_only": "GOLDFIVE_STEER_OBSERVATION_ONLY",
    "threshold": "GOLDFIVE_STEER_THRESHOLD",
    "suppression_window_turns": "GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS",
}

_TOOL_LOOP_ENV: dict[str, str] = {
    "window": "GOLDFIVE_TOOL_LOOP_WINDOW",
    "exact_threshold": "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD",
    "name_threshold": "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD",
    "alternating_threshold": "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD",
}

_REASONING_DRIFT_ENV: dict[str, str] = {
    "mode": "GOLDFIVE_DRIFT_REASONING_MODE",
    "off_topic_distance": "GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE",
    "intent_healthy_similarity": "GOLDFIVE_DRIFT_INTENT_HEALTHY_SIMILARITY",
    "intent_minor_similarity": "GOLDFIVE_DRIFT_INTENT_MINOR_SIMILARITY",
    "intent_warning_similarity": "GOLDFIVE_DRIFT_INTENT_WARNING_SIMILARITY",
    "looping_similarity": "GOLDFIVE_DRIFT_LOOPING_SIMILARITY",
    "cluster_similarity": "GOLDFIVE_DRIFT_CLUSTER_SIMILARITY",
    "looping_hash_window": "GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW",
    "fallback_to_content": "GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT",
}

_GOAL_DRIFT_ENV: dict[str, str] = {
    "check_interval": "GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL",
    "activity_window": "GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW",
}

_JUDGE_ENV: dict[str, str] = {
    "base_url": "GOLDFIVE_JUDGE_BASE_URL",
    "model": "GOLDFIVE_JUDGE_MODEL",
    "api_key": "GOLDFIVE_JUDGE_API_KEY",
    "timeout_ms": "GOLDFIVE_JUDGE_TIMEOUT_MS",
}

_FAIL_FAST_ENV: dict[str, str] = {
    "revision_rejection": "GOLDFIVE_FAIL_FAST_REVISION_REJECTION",
    "invoke_cancel": "GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL",
}

_EXAMPLES_ENV: dict[str, str] = {
    "topic": "GOLDFIVE_EXAMPLE_TOPIC",
    "openai_api_key": "OPENAI_API_KEY",
    "harmonograf_server": "HARMONOGRAF_SERVER",
}


def _make_env_fixture(env_map: dict[str, str]) -> Callable[..., Iterator[_EnvController]]:
    """Build a fixture function that yields an ``_EnvController`` over ``env_map``."""

    @pytest.fixture
    def _fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[_EnvController]:
        controller = _EnvController(monkeypatch, env_map)
        # Start from a clean slate: any of these env vars set in the
        # ambient environment must not leak into the test.
        controller.clear()
        yield controller
        # No teardown needed — monkeypatch unwinds set/delenv automatically.

    return _fixture


goldfive_agent_env = _make_env_fixture(_AGENT_ENV)
goldfive_embedding_env = _make_env_fixture(_EMBEDDING_ENV)
goldfive_steer_env = _make_env_fixture(_STEER_ENV)
goldfive_tool_loop_env = _make_env_fixture(_TOOL_LOOP_ENV)
goldfive_reasoning_drift_env = _make_env_fixture(_REASONING_DRIFT_ENV)
goldfive_goal_drift_env = _make_env_fixture(_GOAL_DRIFT_ENV)
goldfive_judge_env = _make_env_fixture(_JUDGE_ENV)
goldfive_fail_fast_env = _make_env_fixture(_FAIL_FAST_ENV)
goldfive_examples_env = _make_env_fixture(_EXAMPLES_ENV)


@pytest.fixture
def goldfive_runtime_env(
    goldfive_embedding_env: _EnvController,
    goldfive_tool_loop_env: _EnvController,
    goldfive_reasoning_drift_env: _EnvController,
    goldfive_goal_drift_env: _EnvController,
    goldfive_judge_env: _EnvController,
    goldfive_steer_env: _EnvController,
    goldfive_agent_env: _EnvController,
) -> dict[str, _EnvController]:
    """Bundle every ``RuntimeConfig`` sub-domain controller for tests
    that exercise the aggregate ``RuntimeConfig.from_env()`` path.

    Returned as a dict keyed by sub-domain so a single test can poke
    multiple families without claiming half a dozen named fixtures.
    """
    return {
        "embedding": goldfive_embedding_env,
        "tool_loop": goldfive_tool_loop_env,
        "reasoning_drift": goldfive_reasoning_drift_env,
        "goal_drift": goldfive_goal_drift_env,
        "judge": goldfive_judge_env,
        "steer": goldfive_steer_env,
        "agent": goldfive_agent_env,
    }
