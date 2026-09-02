"""Regression guards for the persisted Goldfive runtime configuration."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

import goldfive
import goldfive.runtime_config_document as document_module
from goldfive import RuntimeConfigDocument
from goldfive.config import (
    AgentConfig,
    EmbeddingConfig,
    GoalDriftConfig,
    JudgeConfig,
    ReasoningDriftConfig,
    RuntimeConfig,
    SteeringConfig,
    ToolLoopConfig,
)


def _group(mapping: dict[str, Any], name: str) -> dict[str, Any]:
    group = mapping[name]
    assert isinstance(group, dict)
    return group


def test_document_is_exported_from_the_public_package() -> None:
    assert goldfive.RuntimeConfigDocument is RuntimeConfigDocument
    assert {"JsonValue", "RuntimeConfigDocument", "SecretResolver"} <= set(goldfive.__all__)


def test_empty_document_uses_explicit_defaults_and_ignores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient ``GOLDFIVE_*`` values cannot alter persisted configuration."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S", "999")
    monkeypatch.setenv("GOLDFIVE_CAPABILITY_RULE_A", "1")
    monkeypatch.setenv("GOLDFIVE_CAPABILITY_RULE_C", "1")
    monkeypatch.setenv("GOLDFIVE_FAIL_FAST_REVISION_REJECTION", "1")
    monkeypatch.setenv("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL", "1")
    monkeypatch.setenv("GOLDFIVE_STRICT_STATE_OWNERSHIP", "1")
    monkeypatch.setenv("GOLDFIVE_AGENT_CALL_TIMEOUT_MS", "42")

    document = RuntimeConfigDocument.from_mapping({})
    mapping = document.to_mapping()
    runtime = document.build(resolve_secret=lambda _name: None)

    assert _group(mapping, "embedding")["breaker_cooldown_s"] == 60.0
    assert _group(mapping, "agent")["call_timeout_ms"] == 120_000
    assert runtime.embedding.breaker_cooldown_s == 60.0
    assert runtime.steering.capability_rule_a_enabled is False
    assert runtime.steering.capability_rule_c_enabled is False
    assert runtime.fail_fast_on_revision_rejection is False
    assert runtime.fail_fast_on_invoke_cancel is False
    assert runtime.strict_state_ownership is False
    assert runtime.agent.call_timeout_ms == 120_000


def test_defaults_overlay_and_operator_precedence() -> None:
    zicato_defaults = {"agent": {"call_timeout_ms": 1_800_000}}

    inherited = RuntimeConfigDocument.from_mapping({}, defaults=zicato_defaults)
    overridden = RuntimeConfigDocument.from_mapping(
        {"agent": {"call_timeout_ms": 900_000}}, defaults=zicato_defaults
    )

    assert inherited.build(resolve_secret=lambda _name: None).agent.call_timeout_ms == 1_800_000
    assert overridden.build(resolve_secret=lambda _name: None).agent.call_timeout_ms == 900_000


def test_scaffold_parse_round_trip_is_canonical_and_independent() -> None:
    scaffold = RuntimeConfigDocument.scaffold()
    document = RuntimeConfigDocument.from_mapping(scaffold)

    assert document.to_mapping() == scaffold
    _group(scaffold, "agent")["call_timeout_ms"] = 1
    first_copy = document.to_mapping()
    _group(first_copy, "agent")["call_timeout_ms"] = 2
    assert _group(document.to_mapping(), "agent")["call_timeout_ms"] == 120_000


def _dataclass_field_paths() -> set[str]:
    groups = {
        "embedding": EmbeddingConfig,
        "tool_loops": ToolLoopConfig,
        "reasoning_drift": ReasoningDriftConfig,
        "goal_drift": GoalDriftConfig,
        "judge": JudgeConfig,
        "steering": SteeringConfig,
        "agent": AgentConfig,
    }
    paths = {
        f"{group}.{field.name}"
        for group, config_type in groups.items()
        for field in dataclasses.fields(config_type)
    }
    paths.update(
        field.name for field in dataclasses.fields(RuntimeConfig) if field.name not in groups
    )
    return paths


def _document_field_paths(value: dict[str, Any], prefix: str = "") -> set[str]:
    result: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_document_field_paths(item, path))
        else:
            result.add(path)
    return result


def test_every_runtime_behavior_field_has_one_document_field() -> None:
    """The complete scaffold covers runtime behavior without internal labels."""
    document_paths = _document_field_paths(RuntimeConfigDocument.scaffold())
    aliases = {
        "embedding.api_key": "embedding.api_key_env",
        "judge.api_key": "judge.api_key_env",
        "steering.capability_rule_a_enabled": (
            "steering.delegation_only_agent_leaf_task_detector_enabled"
        ),
        "steering.capability_rule_c_enabled": (
            "steering.pending_task_role_mismatch_detector_enabled"
        ),
        "steering.legacy_ladder": ("steering.cancel_and_reinvoke_interventions_enabled"),
    }
    expected_document_paths = {
        aliases.get(runtime_path, runtime_path) for runtime_path in _dataclass_field_paths()
    }

    assert document_paths == expected_document_paths | {
        "embedding.revision",
        "judge.revision",
    }
    rendered = json.dumps(RuntimeConfigDocument.scaffold())
    assert "capability_rule_a" not in rendered
    assert "capability_rule_c" not in rendered
    assert "legacy_ladder" not in rendered
    assert "legacy_user_message" not in rendered
    assert _group(RuntimeConfigDocument.scaffold(), "steering")["signal_channel"] == "user_message"


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": True},
        {"agent": {"unknown": True}},
        {"agent": {"call_timeout_ms": True}},
        {"agent": {"call_timeout_ms": -1}},
        {"tool_loops": {"window": 10**10_000}},
        {"judge": {"base_url": "https://judge.example", "timeout_ms": 10**10_000}},
        {"reasoning_drift": {"mode": "automatic"}},
        {"reasoning_drift": {"off_topic_distance_threshold": float("nan")}},
        {"reasoning_drift": {"off_topic_distance_threshold": 10**10_000}},
        {"reasoning_drift": {"intent_divergence_warning_similarity": -1.01}},
        {"reasoning_drift": {"looping_reasoning_similarity_threshold": float("inf")}},
        {"steering": {"context_editor_rules": ("prune_stale_steer",)}},
        {"steering": {"context_editor_rules": ["unknown_rule"]}},
        {"steering": {"signal_channel": "legacy_user_message"}},
        {"steering": {"stall_timeout_s": float("inf")}},
        {"steering": {"pause_escalate_deadline_s": 10**10_000}},
        {"judge": {"api_key": "must-never-be-accepted"}},
        {"judge": {"api_key_env": "JUDGE_KEY"}},
        {"embedding": {"revision": "model-v1"}},
        {"embedding": {"model": "inert-model"}},
        {"judge": {"timeout_ms": 20_000}},
        {
            "reasoning_drift": {
                "intent_divergence_healthy_similarity": 0.3,
                "intent_divergence_minor_similarity": 0.4,
            }
        },
    ],
)
def test_malformed_and_cross_field_invalid_documents_reject(
    raw: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        RuntimeConfigDocument.from_mapping(raw)


def test_defaults_are_validated_before_operator_overlay() -> None:
    """Operator input cannot repair an invalid defaults mapping after the fact."""
    with pytest.raises(ValueError, match="agent.call_timeout_ms"):
        RuntimeConfigDocument.from_mapping(
            {"agent": {"call_timeout_ms": 10}},
            defaults={"agent": {"call_timeout_ms": -1}},
        )


def test_disabling_boundaries_and_negative_cosine_thresholds_are_valid() -> None:
    """Validation retains Goldfive's documented disabling boundaries."""
    document = RuntimeConfigDocument.from_mapping(
        {
            "tool_loops": {"window": 2, "exact_threshold": 3},
            "reasoning_drift": {
                "intent_divergence_healthy_similarity": -0.1,
                "intent_divergence_minor_similarity": -0.1,
                "intent_divergence_warning_similarity": -0.3,
                "looping_reasoning_similarity_threshold": -0.4,
                "reasoning_cluster_similarity_threshold": -0.5,
            },
            "steering": {
                "suppression_window_turns": 0,
                "grace_window_turns": 0,
                "pause_escalate_deadline_s": None,
                "stall_timeout_s": 0,
            },
            "agent": {"max_output_tokens": 0, "call_timeout_ms": 0},
        }
    )

    runtime = document.build(resolve_secret=lambda _name: None)
    assert runtime.tool_loops.exact_threshold == 3
    assert runtime.reasoning_drift.intent_divergence_healthy_similarity == -0.1
    assert runtime.reasoning_drift.intent_divergence_minor_similarity == -0.1
    assert runtime.reasoning_drift.intent_divergence_warning_similarity == -0.3
    assert runtime.reasoning_drift.looping_reasoning_similarity_threshold == -0.4
    assert runtime.reasoning_drift.reasoning_cluster_similarity_threshold == -0.5
    assert runtime.steering.suppression_window_turns == 0
    assert runtime.steering.grace_window_turns == 0
    assert runtime.steering.pause_escalate_deadline_s is None
    assert runtime.steering.stall_timeout_s == 0
    assert runtime.agent.max_output_tokens == 0
    assert runtime.agent.call_timeout_ms == 0


def test_negative_finite_stall_timeout_is_valid() -> None:
    runtime = RuntimeConfigDocument.from_mapping({"steering": {"stall_timeout_s": -1}}).build(
        resolve_secret=lambda _name: None
    )

    assert runtime.steering.stall_timeout_s == -1


def test_public_user_message_channel_builds_internal_compatibility_value() -> None:
    runtime = RuntimeConfigDocument.from_mapping(
        {"steering": {"signal_channel": "user_message"}}
    ).build(resolve_secret=lambda _name: None)

    assert runtime.steering.signal_channel == "legacy_user_message"


def test_credentials_resolve_only_during_build_without_disclosure() -> None:
    secret = "credential-that-must-not-leak"
    calls: list[str] = []

    with pytest.raises(ValueError) as invalid_document:
        RuntimeConfigDocument.from_mapping({"judge": {"api_key": secret}})
    assert secret not in str(invalid_document.value)

    raw = {
        "embedding": {
            "base_url": "https://embedding.example/root",
            "api_key_env": "SHARED_BACKEND_KEY",
        },
        "judge": {
            "base_url": "https://judge.example/root",
            "api_key_env": "SHARED_BACKEND_KEY",
        },
    }

    document = RuntimeConfigDocument.from_mapping(raw)
    assert calls == []
    assert document.secret_env_names == ("SHARED_BACKEND_KEY",)
    assert secret not in repr(document)
    assert secret not in json.dumps(document.to_mapping())

    def resolve_secret(name: str) -> str:
        calls.append(name)
        return secret

    runtime = document.build(resolve_secret=resolve_secret)
    assert calls == ["SHARED_BACKEND_KEY"]
    assert runtime.embedding.api_key == secret
    assert runtime.judge.api_key == secret

    def unsafe_resolver(_name: str) -> str:
        raise RuntimeError(secret)

    with pytest.raises(ValueError) as caught:
        document.build(resolve_secret=unsafe_resolver)
    assert "SHARED_BACKEND_KEY" in str(caught.value)
    assert secret not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.test",
        "https://example.test/path?token=secret",
        "https://example.test/path#credential",
        "https://example.test/v1",
        "https://example.test/v1/",
    ],
)
@pytest.mark.parametrize("group", ["embedding", "judge"])
def test_endpoint_urls_reject_userinfo_query_fragment_and_v1_suffix(url: str, group: str) -> None:
    with pytest.raises(ValueError, match=rf"{group}\.base_url"):
        RuntimeConfigDocument.from_mapping({group: {"base_url": url}})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, frozenset()),
        ({"judge": {"base_url": "https://judge.example"}}, frozenset({"remote"})),
        (
            {"embedding": {"base_url": "https://embedding.example"}},
            frozenset(),
        ),
        (
            {
                "reasoning_drift": {"mode": "embedding"},
                "embedding": {"base_url": "https://embedding.example"},
            },
            frozenset({"remote"}),
        ),
        (
            {"reasoning_drift": {"mode": "embedding"}},
            frozenset({"embedding"}),
        ),
        (
            {
                "reasoning_drift": {"mode": "both"},
                "embedding": {"base_url": "https://embedding.example"},
            },
            frozenset({"remote"}),
        ),
    ],
)
def test_required_extras_follow_selected_backends(
    raw: dict[str, object], expected: frozenset[str]
) -> None:
    assert RuntimeConfigDocument.from_mapping(raw).required_extras == expected


def test_missing_runtime_capabilities_follow_backend_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        document_module,
        "_runtime_symbol_available",
        lambda module_name, symbol_name: (module_name, symbol_name) in available,
    )
    document = RuntimeConfigDocument.from_mapping(
        {
            "reasoning_drift": {"mode": "embedding"},
            "judge": {"base_url": "https://judge.example"},
            "embedding": {"base_url": "https://embedding.example"},
        }
    )
    assert document.missing_runtime_capabilities() == (
        "remote_judge",
        "remote_embedding",
    )

    available.add(("httpx", "Client"))
    assert document.missing_runtime_capabilities() == ("remote_judge",)
    available.add(("openai", "AsyncOpenAI"))
    assert document.missing_runtime_capabilities() == ()

    local = RuntimeConfigDocument.from_mapping({"reasoning_drift": {"mode": "embedding"}})
    assert local.missing_runtime_capabilities() == ("local_embedding",)
    available.add(("sentence_transformers", "SentenceTransformer"))
    assert local.missing_runtime_capabilities() == ()


def test_capability_probe_imports_only_selected_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def runtime_symbol_available(module_name: str, symbol_name: str) -> bool:
        calls.append(f"{module_name}.{symbol_name}")
        return True

    monkeypatch.setattr(document_module, "_runtime_symbol_available", runtime_symbol_available)

    assert RuntimeConfigDocument.from_mapping({}).missing_runtime_capabilities() == ()
    assert calls == []


def test_capability_probe_rejects_an_importable_but_broken_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discoverable package that raises during import remains unavailable."""
    calls: list[str] = []

    class CompatibleHttpx:
        class Client:
            pass

    class IncompatibleOpenAI:
        AsyncOpenAI = None

    def import_module(name: str) -> object:
        calls.append(name)
        if name == "openai":
            return IncompatibleOpenAI()
        return CompatibleHttpx()

    monkeypatch.setattr(document_module.importlib, "import_module", import_module)

    assert document_module._runtime_symbol_available("openai", "AsyncOpenAI") is False
    assert document_module._runtime_symbol_available("httpx", "Client") is True
    assert calls == ["openai", "httpx"]


def test_build_matches_every_canonical_behavior_field() -> None:
    """A fully populated document builds the corresponding runtime exactly."""
    document = RuntimeConfigDocument.from_mapping(
        {
            "embedding": {
                "base_url": "https://embedding.example/root",
                "model": "embedding-model",
                "revision": "embedding-revision",
                "api_key_env": "EMBEDDING_KEY",
                "timeout_ms": 1_001,
                "breaker_cooldown_s": 61,
            },
            "tool_loops": {
                "window": 20,
                "exact_threshold": 4,
                "name_threshold": 6,
                "alternating_threshold": 8,
                "name_axis_max_severity": "warning",
            },
            "reasoning_drift": {
                "mode": "both",
                "off_topic_distance_threshold": 0.8,
                "intent_divergence_healthy_similarity": 0.7,
                "intent_divergence_minor_similarity": 0.5,
                "intent_divergence_warning_similarity": 0.3,
                "looping_reasoning_similarity_threshold": 0.91,
                "reasoning_cluster_similarity_threshold": 0.76,
                "looping_reasoning_hash_window": 7,
                "max_concurrent_judges": 4,
                "fallback_to_content_when_no_reasoning": True,
            },
            "goal_drift": {"check_interval": 6, "activity_window": 12},
            "judge": {
                "base_url": "https://judge.example/root",
                "model": "judge-model",
                "revision": "judge-revision",
                "api_key_env": "JUDGE_KEY",
                "timeout_ms": 1_002,
            },
            "steering": {
                "threshold": "critical",
                "suppression_window_turns": 4,
                "observation_only": False,
                "delegation_only_agent_leaf_task_detector_enabled": True,
                "pending_task_role_mismatch_detector_enabled": True,
                "context_editor_rules": ["prune_stale_steer"],
                "descriptive_growth_enabled": True,
                "signal_telemetry": True,
                "cancel_inflight_scope": "all",
                "signal_channel": "request_context",
                "plan_mode": "ledger",
                "cancel_and_reinvoke_interventions_enabled": True,
                "pin_assigned_task": True,
                "grace_window_turns": 5,
                "approval_default_timeout_ms": 700_000,
                "pause_escalate_deadline_s": 40,
                "stall_watchdog_enabled": True,
                "stall_timeout_s": 41,
            },
            "agent": {"max_output_tokens": 17_000, "call_timeout_ms": 130_000},
            "fail_fast_on_revision_rejection": True,
            "fail_fast_on_invoke_cancel": True,
            "strict_state_ownership": True,
        }
    )

    runtime = document.build(
        resolve_secret=lambda name: {
            "EMBEDDING_KEY": "embedding-secret",
            "JUDGE_KEY": "judge-secret",
        }[name]
    )

    assert runtime == RuntimeConfig(
        embedding=EmbeddingConfig(
            base_url="https://embedding.example/root",
            model="embedding-model",
            api_key="embedding-secret",
            timeout_ms=1_001,
            breaker_cooldown_s=61.0,
        ),
        tool_loops=ToolLoopConfig(
            window=20,
            exact_threshold=4,
            name_threshold=6,
            alternating_threshold=8,
            name_axis_max_severity="warning",
        ),
        reasoning_drift=ReasoningDriftConfig(
            mode="both",
            off_topic_distance_threshold=0.8,
            intent_divergence_healthy_similarity=0.7,
            intent_divergence_minor_similarity=0.5,
            intent_divergence_warning_similarity=0.3,
            looping_reasoning_similarity_threshold=0.91,
            reasoning_cluster_similarity_threshold=0.76,
            looping_reasoning_hash_window=7,
            max_concurrent_judges=4,
            fallback_to_content_when_no_reasoning=True,
        ),
        goal_drift=GoalDriftConfig(check_interval=6, activity_window=12),
        judge=JudgeConfig(
            base_url="https://judge.example/root",
            model="judge-model",
            api_key="judge-secret",
            timeout_ms=1_002,
        ),
        steering=SteeringConfig(
            threshold="critical",
            suppression_window_turns=4,
            observation_only=False,
            capability_rule_a_enabled=True,
            capability_rule_c_enabled=True,
            context_editor_rules=["prune_stale_steer"],
            descriptive_growth_enabled=True,
            signal_telemetry=True,
            cancel_inflight_scope="all",
            signal_channel="request_context",
            plan_mode="ledger",
            legacy_ladder=True,
            pin_assigned_task=True,
            grace_window_turns=5,
            approval_default_timeout_ms=700_000,
            pause_escalate_deadline_s=40.0,
            stall_watchdog_enabled=True,
            stall_timeout_s=41.0,
        ),
        agent=AgentConfig(max_output_tokens=17_000, call_timeout_ms=130_000),
        fail_fast_on_revision_rejection=True,
        fail_fast_on_invoke_cancel=True,
        strict_state_ownership=True,
    )
