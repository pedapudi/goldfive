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
    JudgeConfig,
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
        "GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW",
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
            looping_reasoning_hash_window=9,
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
    assert cfg.reasoning_drift.looping_reasoning_hash_window == 9

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
    # Mode flows from the config into the steerer.
    assert steerer._reasoning_drift_mode == "judge"


def test_wrap_threads_reasoning_drift_mode_from_config() -> None:
    """A non-default ``reasoning_drift.mode`` propagates into the steerer."""
    cfg = RuntimeConfig(
        reasoning_drift=ReasoningDriftConfig(mode="embedding"),
    )
    runner = goldfive.wrap(_noop_agent, runtime=cfg, sinks=[])
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._reasoning_drift_mode == "embedding"


def test_wrap_threads_reasoning_drift_mode_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GOLDFIVE_DRIFT_REASONING_MODE=both make demo` end-to-end."""
    monkeypatch.setenv("GOLDFIVE_DRIFT_REASONING_MODE", "both")
    runner = goldfive.wrap(_noop_agent, sinks=[])
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._reasoning_drift_mode == "both"


def test_wrap_falls_back_to_from_env_when_runtime_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``runtime=None`` (the default) env vars drive the effective config."""
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL", "11")
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW", "30")
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_WINDOW", "13")
    monkeypatch.setenv("GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW", "11")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://envfall:1234")

    runner = goldfive.wrap(_noop_agent, sinks=[])

    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_check_interval == 11
    assert steerer._goal_drift_activity_window == 30

    assert _tool_loops.resolve_thresholds()["window"] == 13
    assert _reasoning._CONFIG is not None
    assert _reasoning._CONFIG.looping_reasoning_hash_window == 11
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


def test_wrap_threads_steering_config() -> None:
    """RuntimeConfig.steering propagates into DefaultSteerer."""
    from goldfive.config import SteeringConfig

    cfg = RuntimeConfig(
        steering=SteeringConfig(threshold="critical", suppression_window_turns=9)
    )
    runner = goldfive.wrap(_noop_agent, runtime=cfg, sinks=[])
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goldfive_steer_threshold == "critical"
    assert steerer._goldfive_steer_suppression_window_turns == 9


def test_wrap_threads_steering_config_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GOLDFIVE_STEER_THRESHOLD env var threads through wrap()."""
    monkeypatch.setenv("GOLDFIVE_STEER_THRESHOLD", "off")
    monkeypatch.setenv("GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS", "11")
    runner = goldfive.wrap(_noop_agent, sinks=[])
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goldfive_steer_threshold == "off"
    assert steerer._goldfive_steer_suppression_window_turns == 11


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


# ---------------------------------------------------------------------------
# JudgeConfig routing (goldfive silent-disarm follow-up)
# ---------------------------------------------------------------------------


def test_wrap_uses_judge_config_over_detected_llm() -> None:
    """``JudgeConfig.base_url`` wins over the auto-detected tree LLM.

    When both an auto-detectable agent LLM AND a :class:`JudgeConfig`
    are present, the two drift judges should be wired from the
    JudgeConfig endpoint -- not from ``detect_llm``. Planner +
    goal_deriver still use the detected LLM; only the judges split.
    """
    detected_call_llm = _StubCallable("detected")
    judge_call_llm = _StubCallable("from-judge-config")

    def _fake_detect(_agent: Any) -> tuple[Any, str]:
        return detected_call_llm, "detected-tree-model"

    def _fake_build(cfg: JudgeConfig) -> tuple[Any, str]:
        assert cfg.base_url == "http://judge:9000"
        return judge_call_llm, cfg.model

    runtime = RuntimeConfig(
        judge=JudgeConfig(base_url="http://judge:9000", model="judge-model"),
    )
    runner = goldfive.wrap(
        _noop_agent,
        runtime=runtime,
        sinks=[],
        llm_detector=_fake_detect,
        judge_call_llm_builder=_fake_build,
    )

    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    # Judges use the JudgeConfig-built callable, NOT the detected one.
    assert steerer._reasoning_drift_call_llm is judge_call_llm
    assert steerer._goal_drift_call_llm is judge_call_llm
    assert steerer._reasoning_drift_model == "judge-model"
    assert steerer._goal_drift_model == "judge-model"


def test_wrap_explicit_call_llm_wins_over_judge_config() -> None:
    """Explicit ``call_llm=`` trumps both ``JudgeConfig`` and ``detect_llm``."""
    explicit = _StubCallable("explicit")

    def _must_not_build(_cfg: Any) -> Any:
        raise AssertionError("JudgeConfig path should be suppressed")

    runtime = RuntimeConfig(
        judge=JudgeConfig(base_url="http://should-not-build:9000"),
    )
    runner = goldfive.wrap(
        _noop_agent,
        call_llm=explicit,
        model="explicit-model",
        runtime=runtime,
        sinks=[],
        judge_call_llm_builder=_must_not_build,
    )

    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_call_llm is explicit
    assert steerer._reasoning_drift_call_llm is explicit
    assert steerer._goal_drift_model == "explicit-model"


def test_wrap_falls_back_when_judge_config_build_fails() -> None:
    """A ``JudgeConfig`` build failure falls back to the detected LLM."""
    detected = _StubCallable("detected")

    def _fake_detect(_agent: Any) -> tuple[Any, str]:
        return detected, "tree-model"

    def _fail_build(_cfg: Any) -> None:
        return None  # openai SDK missing / client construction failed

    runtime = RuntimeConfig(
        judge=JudgeConfig(base_url="http://judge:9000"),
    )
    runner = goldfive.wrap(
        _noop_agent,
        runtime=runtime,
        sinks=[],
        llm_detector=_fake_detect,
        judge_call_llm_builder=_fail_build,
    )

    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    # Falls back to the detected callable rather than disarming judges.
    assert steerer._goal_drift_call_llm is detected
    assert steerer._reasoning_drift_call_llm is detected


class _StubCallable:
    """Marker-only callable for identity assertions.

    Shape-compatible with :data:`goldfive._llm_detect.CallLLM`: async
    callable returning a string. We never actually invoke it in these
    tests -- the assertions are on identity, not behaviour.
    """

    def __init__(self, label: str) -> None:
        self._label = label

    async def __call__(self, system: str, user: str, model: str) -> str:
        return f"{self._label}:{model}"
