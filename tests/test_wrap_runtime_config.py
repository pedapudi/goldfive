"""Tests that :func:`goldfive.wrap` threads a ``RuntimeConfig`` through
into the embedding backend, reasoning-drift module, tool-loops module,
and the default steerer.

Regression suite for goldfive#225. Key invariants:

* Passing ``runtime=cfg`` installs every sub-config.
* Omitting ``runtime=`` builds one from the environment so the
  pre-#225 contract is preserved (byte-identical behaviour to what
  ``wrap()`` did before).
* An explicit ``steerer=`` kwarg wins over the runtime-derived steerer
  — the caller keeps full control.
* The precedence rule on ``DefaultSteerer.__init__`` is: explicit
  individual kwarg > config dataclass > module default.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

import goldfive  # noqa: E402
from goldfive.config import (  # noqa: E402
    EmbeddingConfig,
    GoalDriftConfig,
    ReasoningDriftConfig,
    RuntimeConfig,
    ToolLoopConfig,
)
from goldfive.drift import _embed  # noqa: E402
from goldfive.drift import reasoning as _reasoning  # noqa: E402
from goldfive.drift import tool_loops as _tool_loops  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402


@pytest.fixture(autouse=True)
def _scrub_module_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Clear every module-level config slot and env var before each test.

    Tests that install a config via ``wrap()`` must leave the process
    in a clean state for downstream tests — otherwise the reasoning-
    drift detectors run against last-test thresholds.
    """
    _embed.set_model(None)
    _embed.configure(None)
    _reasoning.configure(None)
    _tool_loops.configure(None)
    for name in (
        "GOLDFIVE_EMBEDDING_BASE_URL",
        "GOLDFIVE_EMBEDDING_MODEL",
        "GOLDFIVE_EMBEDDING_API_KEY",
        "GOLDFIVE_EMBEDDING_TIMEOUT_MS",
        "GOLDFIVE_TOOL_LOOP_WINDOW",
        "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD",
        "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD",
        "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD",
        "GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE",
        "GOLDFIVE_DRIFT_CONFUSION_MIN_HITS",
        "GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL",
        "GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    _embed.set_model(None)
    _embed.configure(None)
    _reasoning.configure(None)
    _tool_loops.configure(None)


async def _noop_agent(task: Any, session: Any, tools: Any) -> InvocationResult:
    return InvocationResult(task_id=getattr(task, "id", ""), text="ok")


# ---------------------------------------------------------------------------
# Runtime config threads through wrap()
# ---------------------------------------------------------------------------


def test_wrap_threads_runtime_config() -> None:
    """A non-default ``RuntimeConfig`` propagates into every subsystem."""
    cfg = RuntimeConfig(
        embedding=EmbeddingConfig(
            base_url="http://runtime:9000",
            model="custom-embed",
            timeout_ms=7500,
        ),
        tool_loops=ToolLoopConfig(
            window=15, exact_threshold=4, name_threshold=6, alternating_threshold=6
        ),
        reasoning_drift=ReasoningDriftConfig(
            off_topic_distance_threshold=0.55,
            confusion_min_hits=6,
        ),
        goal_drift=GoalDriftConfig(check_interval=8, activity_window=25),
    )
    runner = goldfive.wrap(_noop_agent, runtime=cfg, sinks=[])

    # _embed module sees the embedding sub-config.
    assert _embed._CONFIG is cfg.embedding

    # Reasoning-drift module sees its sub-config.
    assert _reasoning._CONFIG is cfg.reasoning_drift
    # The helper accessors honour the installed config.
    assert _reasoning._off_topic_distance_threshold() == 0.55
    assert _reasoning._confusion_min_hits() == 6

    # Tool-loops module sees its sub-config.
    assert _tool_loops._CONFIG is cfg.tool_loops
    assert _tool_loops.resolve_thresholds() == {
        "window": 15,
        "exact_threshold": 4,
        "name_threshold": 6,
        "alternating_threshold": 6,
    }

    # Steerer picks up goal-drift + tool-loop + reasoning-drift configs.
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_check_interval == 8
    assert steerer._goal_drift_activity_window == 25
    assert steerer._tool_loop_config is cfg.tool_loops
    assert steerer._reasoning_drift_config is cfg.reasoning_drift
    assert steerer._goal_drift_config is cfg.goal_drift


def test_wrap_falls_back_to_from_env_when_runtime_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``runtime=None`` (the default) env vars drive the effective config."""
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL", "11")
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW", "30")
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_WINDOW", "13")
    monkeypatch.setenv("GOLDFIVE_DRIFT_CONFUSION_MIN_HITS", "7")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://envfall:1234")

    runner = goldfive.wrap(_noop_agent, sinks=[])

    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_check_interval == 11
    assert steerer._goal_drift_activity_window == 30

    assert _tool_loops.resolve_thresholds()["window"] == 13
    assert _reasoning._confusion_min_hits() == 7
    assert _embed._CONFIG is not None
    assert _embed._CONFIG.base_url == "http://envfall:1234"


def test_wrap_runtime_none_uses_defaults_when_env_empty() -> None:
    """With no runtime and no env, defaults match pre-#225 behaviour."""
    runner = goldfive.wrap(_noop_agent, sinks=[])
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    # Pre-#225 defaults: check_interval=5, activity_window=10.
    assert steerer._goal_drift_check_interval == 5
    assert steerer._goal_drift_activity_window == 10


def test_wrap_explicit_steerer_wins_over_runtime() -> None:
    """``steerer=`` bypasses the runtime-derived steerer configuration."""
    cfg = RuntimeConfig(goal_drift=GoalDriftConfig(check_interval=99))
    explicit = DefaultSteerer(goal_drift_check_interval=3)
    runner = goldfive.wrap(_noop_agent, runtime=cfg, steerer=explicit, sinks=[])
    assert runner.steerer is explicit
    assert runner.steerer._goal_drift_check_interval == 3
    # The embedding / reasoning / tool-loop modules are still configured
    # by wrap() itself — those channels are independent of the steerer.
    assert _embed._CONFIG is cfg.embedding


# ---------------------------------------------------------------------------
# DefaultSteerer precedence
# ---------------------------------------------------------------------------


def test_default_steerer_explicit_kwarg_wins_over_config() -> None:
    """explicit kwarg > config dataclass > module default."""
    cfg = GoalDriftConfig(check_interval=42, activity_window=99)
    # Explicit kwarg wins.
    steerer = DefaultSteerer(
        goal_drift_check_interval=7,
        goal_drift_config=cfg,
    )
    assert steerer._goal_drift_check_interval == 7
    # The non-overridden field still comes from the config.
    assert steerer._goal_drift_activity_window == 99


def test_default_steerer_config_wins_over_default() -> None:
    """When no explicit kwarg is passed, config values take effect."""
    cfg = GoalDriftConfig(check_interval=12, activity_window=20)
    steerer = DefaultSteerer(goal_drift_config=cfg)
    assert steerer._goal_drift_check_interval == 12
    assert steerer._goal_drift_activity_window == 20


def test_default_steerer_no_config_no_kwarg_uses_module_default() -> None:
    """Back-compat: bare ``DefaultSteerer()`` still yields the pre-#225 defaults."""
    steerer = DefaultSteerer()
    assert steerer._goal_drift_check_interval == 5
    assert steerer._goal_drift_activity_window == 10
    assert steerer._goal_drift_config is None
    assert steerer._tool_loop_config is None
    assert steerer._reasoning_drift_config is None


def test_default_steerer_installs_reasoning_config_on_init() -> None:
    """Passing ``reasoning_drift_config=`` installs it into the module."""
    cfg = ReasoningDriftConfig(off_topic_distance_threshold=0.42)
    DefaultSteerer(reasoning_drift_config=cfg)
    assert _reasoning._CONFIG is cfg
    assert _reasoning._off_topic_distance_threshold() == 0.42


def test_default_steerer_get_tool_loop_config_returns_stashed() -> None:
    """The steerer exposes the stashed config for plugins to consult."""
    cfg = ToolLoopConfig(window=15)
    steerer = DefaultSteerer(tool_loop_config=cfg)
    assert steerer.get_tool_loop_config() is cfg
    bare = DefaultSteerer()
    assert bare.get_tool_loop_config() is None
