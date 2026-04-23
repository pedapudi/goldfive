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

import pytest

from goldfive.config import (
    EmbeddingConfig,
    GoalDriftConfig,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four env vars map to the matching fields."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://llm.local:8080")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_MODEL", "qwen3-embed")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_API_KEY", "secret-token")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_TIMEOUT_MS", "2500")
    cfg = EmbeddingConfig.from_env()
    assert cfg.base_url == "http://llm.local:8080"
    assert cfg.model == "qwen3-embed"
    assert cfg.api_key == "secret-token"
    assert cfg.timeout_ms == 2500


def test_embedding_config_from_env_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial env landscape fills in remaining fields with defaults."""
    for name in (
        "GOLDFIVE_EMBEDDING_BASE_URL",
        "GOLDFIVE_EMBEDDING_MODEL",
        "GOLDFIVE_EMBEDDING_API_KEY",
        "GOLDFIVE_EMBEDDING_TIMEOUT_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://partial:1234")
    cfg = EmbeddingConfig.from_env()
    assert cfg.base_url == "http://partial:1234"
    # Remaining fields keep the built-in defaults.
    assert cfg.model == ""
    assert cfg.api_key is None
    assert cfg.timeout_ms == 10_000


def test_embedding_config_empty_base_url_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit empty string is treated as "not set" (maps to ``None``)."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "   ")
    cfg = EmbeddingConfig.from_env()
    assert cfg.base_url is None


def test_embedding_config_invalid_timeout_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-integer / non-positive timeouts fall back to the default."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_TIMEOUT_MS", "abc")
    assert EmbeddingConfig.from_env().timeout_ms == 10_000
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_TIMEOUT_MS", "0")
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four env vars map to the matching fields."""
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_WINDOW", "12")
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD", "4")
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD", "7")
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD", "6")
    cfg = ToolLoopConfig.from_env()
    assert cfg.window == 12
    assert cfg.exact_threshold == 4
    assert cfg.name_threshold == 7
    assert cfg.alternating_threshold == 6


def test_tool_loop_config_from_env_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing vars fall back to the built-in defaults."""
    for name in (
        "GOLDFIVE_TOOL_LOOP_WINDOW",
        "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD",
        "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD",
        "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_WINDOW", "15")
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
    assert cfg.off_topic_distance_threshold == 0.7
    assert cfg.intent_divergence_healthy_similarity == 0.6
    assert cfg.intent_divergence_minor_similarity == 0.4
    assert cfg.intent_divergence_warning_similarity == 0.2
    assert cfg.looping_reasoning_similarity_threshold == 0.9
    assert cfg.reasoning_cluster_similarity_threshold == 0.75
    assert cfg.looping_reasoning_hash_window == 5
    assert cfg.confusion_min_hits == 3


def test_reasoning_drift_config_from_env_all_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new ``GOLDFIVE_DRIFT_*`` env vars map to the matching fields."""
    monkeypatch.setenv("GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE", "0.55")
    monkeypatch.setenv("GOLDFIVE_DRIFT_INTENT_HEALTHY_SIMILARITY", "0.7")
    monkeypatch.setenv("GOLDFIVE_DRIFT_INTENT_MINOR_SIMILARITY", "0.5")
    monkeypatch.setenv("GOLDFIVE_DRIFT_INTENT_WARNING_SIMILARITY", "0.3")
    monkeypatch.setenv("GOLDFIVE_DRIFT_LOOPING_SIMILARITY", "0.85")
    monkeypatch.setenv("GOLDFIVE_DRIFT_CLUSTER_SIMILARITY", "0.7")
    monkeypatch.setenv("GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW", "8")
    monkeypatch.setenv("GOLDFIVE_DRIFT_CONFUSION_MIN_HITS", "5")
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.off_topic_distance_threshold == 0.55
    assert cfg.intent_divergence_healthy_similarity == 0.7
    assert cfg.intent_divergence_minor_similarity == 0.5
    assert cfg.intent_divergence_warning_similarity == 0.3
    assert cfg.looping_reasoning_similarity_threshold == 0.85
    assert cfg.reasoning_cluster_similarity_threshold == 0.7
    assert cfg.looping_reasoning_hash_window == 8
    assert cfg.confusion_min_hits == 5


def test_reasoning_drift_config_from_env_missing_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing env vars revert to defaults."""
    for name in (
        "GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE",
        "GOLDFIVE_DRIFT_INTENT_HEALTHY_SIMILARITY",
        "GOLDFIVE_DRIFT_INTENT_MINOR_SIMILARITY",
        "GOLDFIVE_DRIFT_INTENT_WARNING_SIMILARITY",
        "GOLDFIVE_DRIFT_LOOPING_SIMILARITY",
        "GOLDFIVE_DRIFT_CLUSTER_SIMILARITY",
        "GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW",
        "GOLDFIVE_DRIFT_CONFUSION_MIN_HITS",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = ReasoningDriftConfig.from_env()
    assert cfg == ReasoningDriftConfig()


def test_reasoning_drift_config_from_env_bad_float_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-float env var falls back to the default, not a crash."""
    monkeypatch.setenv("GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE", "not-a-float")
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL", "8")
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW", "25")
    cfg = GoalDriftConfig.from_env()
    assert cfg.check_interval == 8
    assert cfg.activity_window == 25


def test_goal_drift_config_from_env_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL", raising=False)
    monkeypatch.delenv("GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW", raising=False)
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL", "3")
    cfg = GoalDriftConfig.from_env()
    assert cfg.check_interval == 3
    assert cfg.activity_window == 10


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


def test_runtime_config_from_env_aggregates_all_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each sub-``from_env`` is called and the results are aggregated."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://agg:7000")
    monkeypatch.setenv("GOLDFIVE_TOOL_LOOP_WINDOW", "14")
    monkeypatch.setenv("GOLDFIVE_DRIFT_CONFUSION_MIN_HITS", "6")
    monkeypatch.setenv("GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL", "7")
    cfg = RuntimeConfig.from_env()
    assert cfg.embedding.base_url == "http://agg:7000"
    assert cfg.tool_loops.window == 14
    assert cfg.reasoning_drift.confusion_min_hits == 6
    assert cfg.goal_drift.check_interval == 7


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
