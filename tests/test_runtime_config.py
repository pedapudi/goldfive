"""Tests for the typed :class:`~goldfive.config.RuntimeConfig` dataclasses.

Regression suite for goldfive#225. Covers:

* Each sub-config's ``from_env`` classmethod reads every supported env
  var, and subsets fall back cleanly to field defaults.
* ``RuntimeConfig.from_env`` aggregates the four sub-``from_env``
  calls.
* The dataclasses are mutable (``frozen=False``) so operators can
  tweak a field after constructing from env -- this is an intentional
  design choice called out in the module docstring.
* Equality is value-based (default dataclass behaviour) so tests can
  assert ``config == expected`` cleanly.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from goldfive.config import (
    EmbeddingConfig,
    GoalDriftConfig,
    JudgeConfig,
    ReasoningDriftConfig,
    RuntimeConfig,
    ToolLoopConfig,
)

# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------


def test_embedding_config_defaults() -> None:
    """Field defaults match the pre-#225 fall-through behaviour."""
    cfg = EmbeddingConfig()
    assert cfg.base_url is None
    assert cfg.model == ""
    assert cfg.api_key is None
    assert cfg.timeout_ms == 10_000


def test_embedding_config_from_env_all_vars(
    goldfive_embedding_env: Any,
) -> None:
    """All four env vars map to the matching fields."""
    goldfive_embedding_env.set(
        base_url="http://llm.local:8080",
        model="qwen3-embed",
        api_key="secret-token",
        timeout_ms=2500,
    )
    cfg = EmbeddingConfig.from_env()
    assert cfg.base_url == "http://llm.local:8080"
    assert cfg.model == "qwen3-embed"
    assert cfg.api_key == "secret-token"
    assert cfg.timeout_ms == 2500


def test_embedding_config_from_env_subset(
    goldfive_embedding_env: Any,
) -> None:
    """A partial env landscape fills in remaining fields with defaults."""
    # ``goldfive_embedding_env`` pre-clears every variable in its setup,
    # so an explicit ``unset`` round is redundant; setting only the
    # field we care about leaves the rest at their dataclass defaults.
    goldfive_embedding_env.set(base_url="http://partial:1234")
    cfg = EmbeddingConfig.from_env()
    assert cfg.base_url == "http://partial:1234"
    # Remaining fields keep the built-in defaults.
    assert cfg.model == ""
    assert cfg.api_key is None
    assert cfg.timeout_ms == 10_000


def test_embedding_config_empty_base_url_is_none(
    goldfive_embedding_env: Any,
) -> None:
    """An explicit empty string is treated as "not set" (maps to ``None``)."""
    goldfive_embedding_env.set(base_url="   ")
    cfg = EmbeddingConfig.from_env()
    assert cfg.base_url is None


def test_embedding_config_invalid_timeout_falls_back(
    goldfive_embedding_env: Any,
) -> None:
    """Non-integer / non-positive timeouts fall back to the default."""
    goldfive_embedding_env.set(timeout_ms="abc")
    assert EmbeddingConfig.from_env().timeout_ms == 10_000
    goldfive_embedding_env.set(timeout_ms="0")
    assert EmbeddingConfig.from_env().timeout_ms == 10_000


# ---------------------------------------------------------------------------
# ToolLoopConfig
# ---------------------------------------------------------------------------


def test_tool_loop_config_defaults() -> None:
    """Defaults track :mod:`goldfive.drift.tool_loops` module constants."""
    cfg = ToolLoopConfig()
    assert cfg.window == 10
    assert cfg.exact_threshold == 3
    assert cfg.name_threshold == 5
    assert cfg.alternating_threshold == 5


def test_tool_loop_config_from_env_all_vars(
    goldfive_tool_loop_env: Any,
) -> None:
    """All four env vars map to the matching fields."""
    goldfive_tool_loop_env.set(
        window=12,
        exact_threshold=4,
        name_threshold=7,
        alternating_threshold=6,
    )
    cfg = ToolLoopConfig.from_env()
    assert cfg.window == 12
    assert cfg.exact_threshold == 4
    assert cfg.name_threshold == 7
    assert cfg.alternating_threshold == 6


def test_tool_loop_config_from_env_subset(
    goldfive_tool_loop_env: Any,
) -> None:
    """Missing vars fall back to the built-in defaults."""
    # Fixture pre-clears every tool-loop env var in its setup.
    goldfive_tool_loop_env.set(window=15)
    cfg = ToolLoopConfig.from_env()
    assert cfg.window == 15
    assert cfg.exact_threshold == 3
    assert cfg.name_threshold == 5
    assert cfg.alternating_threshold == 5


# ---------------------------------------------------------------------------
# ReasoningDriftConfig
# ---------------------------------------------------------------------------


def test_reasoning_drift_config_defaults() -> None:
    """Defaults mirror the module-level constants in
    :mod:`goldfive.drift.reasoning`.
    """
    cfg = ReasoningDriftConfig()
    assert cfg.mode == "judge"
    assert cfg.off_topic_distance_threshold == 0.7
    assert cfg.intent_divergence_healthy_similarity == 0.6
    assert cfg.intent_divergence_minor_similarity == 0.4
    assert cfg.intent_divergence_warning_similarity == 0.2
    assert cfg.looping_reasoning_similarity_threshold == 0.9
    assert cfg.reasoning_cluster_similarity_threshold == 0.75
    assert cfg.looping_reasoning_hash_window == 5


@pytest.mark.parametrize("mode", ["judge", "embedding", "both", "off"])
def test_reasoning_drift_config_from_env_mode(
    goldfive_reasoning_drift_env: Any, mode: str
) -> None:
    """`GOLDFIVE_DRIFT_REASONING_MODE` selects the pipeline mode."""
    goldfive_reasoning_drift_env.set(mode=mode)
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.mode == mode


def test_reasoning_drift_config_from_env_mode_is_case_insensitive(
    goldfive_reasoning_drift_env: Any,
) -> None:
    """Mode parsing tolerates case variations + surrounding whitespace."""
    goldfive_reasoning_drift_env.set(mode="  Embedding  ")
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.mode == "embedding"


def test_reasoning_drift_config_from_env_mode_invalid_warns_and_defaults(
    goldfive_reasoning_drift_env: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown mode logs a WARNING and falls back to the default."""
    goldfive_reasoning_drift_env.set(mode="bogus")
    with caplog.at_level("WARNING", logger="goldfive"):
        cfg = ReasoningDriftConfig.from_env()
    assert cfg.mode == "judge"
    warnings = [
        r for r in caplog.records if "GOLDFIVE_DRIFT_REASONING_MODE" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_reasoning_drift_config_from_env_all_vars(
    goldfive_reasoning_drift_env: Any,
) -> None:
    """The new ``GOLDFIVE_DRIFT_*`` env vars map to the matching fields."""
    goldfive_reasoning_drift_env.set(
        off_topic_distance="0.55",
        intent_healthy_similarity="0.7",
        intent_minor_similarity="0.5",
        intent_warning_similarity="0.3",
        looping_similarity="0.85",
        cluster_similarity="0.7",
        looping_hash_window="8",
    )
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.off_topic_distance_threshold == 0.55
    assert cfg.intent_divergence_healthy_similarity == 0.7
    assert cfg.intent_divergence_minor_similarity == 0.5
    assert cfg.intent_divergence_warning_similarity == 0.3
    assert cfg.looping_reasoning_similarity_threshold == 0.85
    assert cfg.reasoning_cluster_similarity_threshold == 0.7
    assert cfg.looping_reasoning_hash_window == 8


def test_reasoning_drift_config_from_env_missing_falls_back(
    goldfive_reasoning_drift_env: Any,
) -> None:
    """Missing env vars revert to defaults."""
    # ``goldfive_reasoning_drift_env`` pre-clears every reasoning-drift
    # env var in its setup so we don't need an explicit unset round.
    cfg = ReasoningDriftConfig.from_env()
    assert cfg == ReasoningDriftConfig()


def test_reasoning_drift_config_from_env_bad_float_falls_back(
    goldfive_reasoning_drift_env: Any,
) -> None:
    """A non-float env var falls back to the default, not a crash."""
    goldfive_reasoning_drift_env.set(off_topic_distance="not-a-float")
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.off_topic_distance_threshold == 0.7


# ---------------------------------------------------------------------------
# GoalDriftConfig
# ---------------------------------------------------------------------------


def test_goal_drift_config_defaults() -> None:
    cfg = GoalDriftConfig()
    assert cfg.check_interval == 5
    assert cfg.activity_window == 10


def test_goal_drift_config_from_env_all_vars(
    goldfive_goal_drift_env: Any,
) -> None:
    goldfive_goal_drift_env.set(check_interval=8, activity_window=25)
    cfg = GoalDriftConfig.from_env()
    assert cfg.check_interval == 8
    assert cfg.activity_window == 25


def test_goal_drift_config_from_env_subset(
    goldfive_goal_drift_env: Any,
) -> None:
    # Fixture pre-clears the goal-drift env surface in its setup.
    goldfive_goal_drift_env.set(check_interval=3)
    cfg = GoalDriftConfig.from_env()
    assert cfg.check_interval == 3
    assert cfg.activity_window == 10


# ---------------------------------------------------------------------------
# JudgeConfig
# ---------------------------------------------------------------------------


def test_judge_config_defaults() -> None:
    """Defaults route judges to inherit the tree LLM (``base_url=None``)."""
    cfg = JudgeConfig()
    assert cfg.base_url is None
    assert cfg.model == ""
    assert cfg.api_key is None
    assert cfg.timeout_ms == 10_000


def test_judge_config_from_env(goldfive_judge_env: Any) -> None:
    """All four ``GOLDFIVE_JUDGE_*`` env vars map to fields."""
    goldfive_judge_env.set(
        base_url="http://judge.local:9000",
        model="qwen3-judge",
        api_key="judge-token",
        timeout_ms=3500,
    )
    cfg = JudgeConfig.from_env()
    assert cfg.base_url == "http://judge.local:9000"
    assert cfg.model == "qwen3-judge"
    assert cfg.api_key == "judge-token"
    assert cfg.timeout_ms == 3500


def test_judge_config_from_env_subset(goldfive_judge_env: Any) -> None:
    """Missing env vars fall back to the built-in defaults."""
    # Fixture pre-clears the judge env surface in its setup.
    goldfive_judge_env.set(base_url="http://partial-judge:1234")
    cfg = JudgeConfig.from_env()
    assert cfg.base_url == "http://partial-judge:1234"
    assert cfg.model == ""
    assert cfg.api_key is None
    assert cfg.timeout_ms == 10_000


def test_judge_config_empty_base_url_is_none(
    goldfive_judge_env: Any,
) -> None:
    """Whitespace-only ``GOLDFIVE_JUDGE_BASE_URL`` is treated as unset."""
    goldfive_judge_env.set(base_url="   ")
    cfg = JudgeConfig.from_env()
    assert cfg.base_url is None


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


def test_runtime_config_defaults() -> None:
    """Default aggregate composes defaults of each sub-config."""
    cfg = RuntimeConfig()
    assert cfg.embedding == EmbeddingConfig()
    assert cfg.tool_loops == ToolLoopConfig()
    assert cfg.reasoning_drift == ReasoningDriftConfig()
    assert cfg.goal_drift == GoalDriftConfig()
    assert cfg.judge == JudgeConfig()


def test_runtime_config_includes_judge() -> None:
    """``RuntimeConfig.judge`` is present and independently replaceable."""
    cfg = RuntimeConfig()
    # Field is accessible and mutable (frozen=False).
    cfg.judge.base_url = "http://judge:9000"
    assert cfg.judge.base_url == "http://judge:9000"
    # Fresh instance keeps original default (no cross-contamination).
    fresh = RuntimeConfig()
    assert fresh.judge.base_url is None


def test_runtime_config_from_env_aggregates_all_four(
    goldfive_runtime_env: dict,
) -> None:
    """Each sub-``from_env`` is called and the results are aggregated."""
    goldfive_runtime_env["embedding"].set(base_url="http://agg:7000")
    goldfive_runtime_env["tool_loop"].set(window=14)
    goldfive_runtime_env["reasoning_drift"].set(looping_hash_window=9)
    goldfive_runtime_env["goal_drift"].set(check_interval=7)
    goldfive_runtime_env["judge"].set(base_url="http://judge-agg:9001")
    cfg = RuntimeConfig.from_env()
    assert cfg.embedding.base_url == "http://agg:7000"
    assert cfg.tool_loops.window == 14
    assert cfg.reasoning_drift.looping_reasoning_hash_window == 9
    assert cfg.goal_drift.check_interval == 7
    assert cfg.judge.base_url == "http://judge-agg:9001"


def test_runtime_config_equality_and_mutable_semantics() -> None:
    """Instances are value-equal and fields are mutable (``frozen=False``).

    The dataclasses are deliberately **not** frozen so operators can
    patch a single field after constructing from env (e.g. bumping
    ``goal_drift.check_interval`` for a debugging run). Callers who
    want an immutable snapshot should use :func:`dataclasses.replace`
    to derive a variant.
    """
    a = RuntimeConfig()
    b = RuntimeConfig()
    assert a == b

    # Mutation is allowed.
    a.goal_drift.check_interval = 99
    assert a != b

    # ``dataclasses.replace`` builds a divergent instance without
    # mutating the source.
    derived = dataclasses.replace(
        b, goal_drift=dataclasses.replace(b.goal_drift, check_interval=42)
    )
    assert derived.goal_drift.check_interval == 42
    assert b.goal_drift.check_interval == 5  # untouched
