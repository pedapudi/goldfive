"""Serializable runtime configuration that omits credential values.

``RuntimeConfigDocument`` is the persisted counterpart to
:class:`goldfive.config.RuntimeConfig`. It accepts only JSON-compatible data,
fills every omitted field from stable library defaults, and resolves
credential-variable names only when :meth:`RuntimeConfigDocument.build`
creates the in-memory runtime object. It never reads the process environment.

The document uses public names that describe behavior. Three internal
``SteeringConfig`` compatibility fields therefore have different document
names:

* ``delegation_only_agent_leaf_task_detector_enabled`` controls the detector
  stored internally as ``capability_rule_a_enabled``.
* ``pending_task_role_mismatch_detector_enabled`` controls the detector stored
  internally as ``capability_rule_c_enabled``.
* ``cancel_and_reinvoke_interventions_enabled`` controls the behavior stored
  internally as ``legacy_ladder``.

Endpoint ``revision`` values remain document metadata for callers that use
them in configuration identity. ``RuntimeConfig`` has no corresponding field,
so :meth:`RuntimeConfigDocument.build` does not pass revisions to the runtime.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Literal, TypeAlias
from urllib.parse import urlsplit

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

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
SecretResolver: TypeAlias = Callable[[str], str | None]

__all__ = ["JsonValue", "RuntimeConfigDocument", "SecretResolver"]


_Validator: TypeAlias = Callable[[object, str], JsonValue]


def _type_error(path: str, expected: str) -> ValueError:
    return ValueError(f"{path} must be {expected}")


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise _type_error(path, "a boolean")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _type_error(path, "a string")
    return value


def _optional_nonempty_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _type_error(path, "null or a non-empty string")
    return value


def _integer_at_least(minimum: int) -> _Validator:
    def validate(value: object, path: str) -> int:
        if type(value) is not int or value < minimum:
            raise _type_error(path, f"an integer greater than or equal to {minimum}")
        return value

    return validate


def _finite_number(*, minimum: float | None = None, maximum: float | None = None) -> _Validator:
    def validate(value: object, path: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _type_error(path, "a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise _type_error(path, "a finite number")
        if minimum is not None and result < minimum:
            raise ValueError(f"{path} must be greater than or equal to {minimum}")
        if maximum is not None and result > maximum:
            raise ValueError(f"{path} must be less than or equal to {maximum}")
        return result

    return validate


def _optional_positive_number(value: object, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _type_error(path, "null or a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{path} must be greater than 0 when set")
    return result


def _one_of(*choices: str) -> _Validator:
    allowed = frozenset(choices)

    def validate(value: object, path: str) -> str:
        if not isinstance(value, str) or value not in allowed:
            rendered = ", ".join(sorted(allowed))
            raise ValueError(f"{path} must be one of: {rendered}")
        return value

    return validate


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _credential_variable(value: object, path: str) -> str | None:
    result = _optional_nonempty_string(value, path)
    if result is not None and _ENV_NAME.fullmatch(result) is None:
        raise ValueError(f"{path} must be an environment-variable name")
    return result


def _base_url(value: object, path: str) -> str | None:
    result = _optional_nonempty_string(value, path)
    if result is None:
        return None
    if any(character.isspace() for character in result):
        raise ValueError(f"{path} must not contain whitespace")
    try:
        parsed = urlsplit(result)
        _ = parsed.port
    except ValueError:
        raise ValueError(f"{path} must be a valid HTTP or HTTPS URL") from None
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"{path} must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{path} must not contain user information")
    if "?" in result:
        raise ValueError(f"{path} must not contain a query")
    if "#" in result:
        raise ValueError(f"{path} must not contain a fragment")
    if parsed.path.rstrip("/").endswith("/v1"):
        raise ValueError(f"{path} must omit /v1 because Goldfive appends it")
    return result


_CONTEXT_EDITOR_RULES = frozenset(
    {
        "compact_prior_reasoning",
        "prune_cancelled_reasoning",
        "prune_stale_steer",
        "prune_transient_error",
    }
)


def _context_editor_rules(value: object, path: str) -> list[JsonValue] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise _type_error(path, "null or a list of context-editor rule names")
    result: list[JsonValue] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str) or item not in _CONTEXT_EDITOR_RULES:
            rendered = ", ".join(sorted(_CONTEXT_EDITOR_RULES))
            raise ValueError(f"{item_path} must be one of: {rendered}")
        if item in seen:
            raise ValueError(f"{item_path} duplicates an earlier rule")
        seen.add(item)
        result.append(item)
    return result


_RUNTIME_GROUP_TYPES: dict[str, type[Any]] = {
    "embedding": EmbeddingConfig,
    "tool_loops": ToolLoopConfig,
    "reasoning_drift": ReasoningDriftConfig,
    "goal_drift": GoalDriftConfig,
    "judge": JudgeConfig,
    "steering": SteeringConfig,
    "agent": AgentConfig,
}

_STEERING_DOCUMENT_TO_RUNTIME = {
    "delegation_only_agent_leaf_task_detector_enabled": "capability_rule_a_enabled",
    "pending_task_role_mismatch_detector_enabled": "capability_rule_c_enabled",
    "cancel_and_reinvoke_interventions_enabled": "legacy_ladder",
}

_FIELD_VALIDATORS: dict[str, _Validator] = {
    "embedding.base_url": _base_url,
    "embedding.revision": _optional_nonempty_string,
    "embedding.api_key_env": _credential_variable,
    "judge.base_url": _base_url,
    "judge.revision": _optional_nonempty_string,
    "judge.api_key_env": _credential_variable,
    "reasoning_drift.mode": _one_of("judge", "embedding", "both", "off"),
    "reasoning_drift.off_topic_distance_threshold": _finite_number(minimum=0.0, maximum=2.0),
    "reasoning_drift.intent_divergence_healthy_similarity": _finite_number(
        minimum=-1.0, maximum=1.0
    ),
    "reasoning_drift.intent_divergence_minor_similarity": _finite_number(minimum=-1.0, maximum=1.0),
    "reasoning_drift.intent_divergence_warning_similarity": _finite_number(
        minimum=-1.0, maximum=1.0
    ),
    "reasoning_drift.looping_reasoning_similarity_threshold": _finite_number(
        minimum=-1.0, maximum=1.0
    ),
    "reasoning_drift.reasoning_cluster_similarity_threshold": _finite_number(
        minimum=-1.0, maximum=1.0
    ),
    "tool_loops.name_axis_max_severity": _one_of("info", "warning", "critical"),
    "steering.threshold": _one_of("off", "warning", "critical"),
    "steering.context_editor_rules": _context_editor_rules,
    "steering.cancel_inflight_scope": _one_of("user_and_safety", "all"),
    "steering.signal_channel": _one_of("legacy_user_message", "request_context"),
    "steering.plan_mode": _one_of("forecast", "ledger"),
    "steering.pause_escalate_deadline_s": _optional_positive_number,
    "steering.stall_timeout_s": _finite_number(),
}

_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "steering.suppression_window_turns",
        "steering.grace_window_turns",
        "agent.max_output_tokens",
        "agent.call_timeout_ms",
    }
)


def _hard_defaults() -> dict[str, JsonValue]:
    """Derive document defaults from ``RuntimeConfig`` without reading env."""
    runtime = RuntimeConfig()
    document: dict[str, JsonValue] = {}
    for group_name in _RUNTIME_GROUP_TYPES:
        group = dataclasses.asdict(getattr(runtime, group_name))
        if group_name in {"embedding", "judge"}:
            group.pop("api_key")
            group.update(revision=None, api_key_env=None)
        if group_name == "steering":
            runtime_to_document = {
                runtime_name: document_name
                for document_name, runtime_name in _STEERING_DOCUMENT_TO_RUNTIME.items()
            }
            group = {runtime_to_document.get(key, key): value for key, value in group.items()}
        document[group_name] = group

    # RuntimeConfig keeps these values nullable for low-level environment
    # compatibility. Persisted documents use explicit, environment-independent
    # behavior.
    embedding = _group(document, "embedding")
    steering = _group(document, "steering")
    embedding["breaker_cooldown_s"] = 60.0
    steering["delegation_only_agent_leaf_task_detector_enabled"] = False
    steering["pending_task_role_mismatch_detector_enabled"] = False
    for name in (
        "fail_fast_on_revision_rejection",
        "fail_fast_on_invoke_cancel",
        "strict_state_ownership",
    ):
        document[name] = False
    return document


def _normalize_json(
    value: object, path: str, *, ancestors: frozenset[int] = frozenset()
) -> JsonValue:
    if value is None or type(value) in {bool, int, str}:
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain a non-finite number")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} must not contain a reference cycle")
        nested_ancestors = ancestors | {identity}
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} must use string object keys")
            item_path = f"{path}.{key}" if path else key
            result[key] = _normalize_json(item, item_path, ancestors=nested_ancestors)
        return result
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} must not contain a reference cycle")
        nested_ancestors = ancestors | {identity}
        return [
            _normalize_json(item, f"{path}[{index}]", ancestors=nested_ancestors)
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} must contain only JSON-compatible values")


def _validate_field(value: object, default: JsonValue, path: str) -> JsonValue:
    validator = _FIELD_VALIDATORS.get(path)
    if validator is not None:
        return validator(value, path)
    if type(default) is bool:
        return _boolean(value, path)
    if type(default) is int:
        minimum = 0 if path in _NONNEGATIVE_INTEGER_FIELDS else 1
        return _integer_at_least(minimum)(value, path)
    if type(default) is float:
        return _finite_number(minimum=0.0)(value, path)
    if isinstance(default, str):
        return _string(value, path)
    raise RuntimeError(f"missing validator for document field {path!r}")


def _parse_overlay(
    value: Mapping[str, JsonValue],
    reference: Mapping[str, JsonValue],
    *,
    path: str = "",
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, raw_value in value.items():
        field_path = f"{path}.{key}" if path else key
        if key not in reference:
            raise ValueError(f"unknown configuration field {field_path!r}")
        default = reference[key]
        if isinstance(default, dict):
            if not isinstance(raw_value, dict):
                raise _type_error(field_path, "an object")
            result[key] = _parse_overlay(raw_value, default, path=field_path)
        else:
            result[key] = _validate_field(raw_value, default, field_path)
    return result


def _overlay(base: dict[str, JsonValue], changes: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    result = deepcopy(base)
    for key, value in changes.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            result[key] = _overlay(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _group(document: Mapping[str, JsonValue], name: str) -> dict[str, JsonValue]:
    value = document[name]
    assert isinstance(value, dict)
    return value


def _runtime_kwargs(
    group: Mapping[str, JsonValue],
    *,
    omit: frozenset[str] = frozenset(),
    rename: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Copy validated document fields into dataclass constructor keywords."""
    names = rename or {}
    return {names.get(key, key): deepcopy(value) for key, value in group.items() if key not in omit}


def _build_runtime_groups(
    document: Mapping[str, JsonValue], resolved_credentials: Mapping[str, str]
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name, config_type in _RUNTIME_GROUP_TYPES.items():
        source = _group(document, name)
        endpoint = name in {"embedding", "judge"}
        kwargs = _runtime_kwargs(
            source,
            omit=frozenset({"api_key_env", "revision"}) if endpoint else frozenset(),
            rename=_STEERING_DOCUMENT_TO_RUNTIME if name == "steering" else None,
        )
        if endpoint:
            reference = source["api_key_env"]
            kwargs["api_key"] = (
                resolved_credentials[reference] if isinstance(reference, str) else None
            )
        groups[name] = config_type(**kwargs)
    return groups


def _validate_endpoint_metadata(document: Mapping[str, JsonValue], name: str) -> None:
    endpoint = _group(document, name)
    if endpoint["base_url"] is None:
        runtime_default = _RUNTIME_GROUP_TYPES[name]()
        if name == "embedding" and endpoint["model"]:
            raise ValueError("embedding.model requires embedding.base_url")
        if endpoint["api_key_env"] is not None:
            raise ValueError(f"{name}.api_key_env requires {name}.base_url")
        if endpoint["revision"] is not None:
            raise ValueError(f"{name}.revision requires {name}.base_url")
        if endpoint["timeout_ms"] != runtime_default.timeout_ms:
            raise ValueError(f"{name}.timeout_ms can change only when {name}.base_url is set")


def _validate_cross_fields(document: Mapping[str, JsonValue]) -> None:
    reasoning = _group(document, "reasoning_drift")
    healthy = reasoning["intent_divergence_healthy_similarity"]
    minor = reasoning["intent_divergence_minor_similarity"]
    warning = reasoning["intent_divergence_warning_similarity"]
    assert isinstance(healthy, float)
    assert isinstance(minor, float)
    assert isinstance(warning, float)
    if not healthy >= minor >= warning:
        raise ValueError(
            "reasoning_drift intent-divergence similarities must satisfy "
            "healthy >= minor >= warning"
        )

    _validate_endpoint_metadata(document, "embedding")
    _validate_endpoint_metadata(document, "judge")


def _resolved_document(
    raw: Mapping[str, object], defaults: Mapping[str, object] | None
) -> dict[str, JsonValue]:
    result = _hard_defaults()
    result = _parse_overlay(result, result)
    _validate_cross_fields(result)
    if defaults is not None:
        normalized_defaults = _normalize_json(defaults, "defaults")
        assert isinstance(normalized_defaults, dict)
        parsed_defaults = _parse_overlay(normalized_defaults, result)
        result = _overlay(result, parsed_defaults)
        _validate_cross_fields(result)

    normalized_raw = _normalize_json(raw, "config")
    assert isinstance(normalized_raw, dict)
    parsed_raw = _parse_overlay(normalized_raw, result)
    result = _overlay(result, parsed_raw)
    _validate_cross_fields(result)
    return result


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 - broken optional installs are unavailable
        return False
    return True


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class RuntimeConfigDocument:
    """Validated configuration suitable for persistence and contract hashing.

    Construct documents with :meth:`from_mapping`. Direct construction is
    disabled so every instance has passed type, range, and cross-field
    validation. The stored canonical mapping contains credential-variable
    names. It never contains credential values.
    """

    _document: dict[str, JsonValue] = dataclasses.field(repr=False)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        defaults: Mapping[str, object] | None = None,
    ) -> RuntimeConfigDocument:
        """Validate ``defaults``, overlay ``raw``, and fill hard defaults.

        Both inputs must contain JSON-compatible values. Unknown keys, invalid
        types, out-of-range numbers, and invalid field combinations raise
        :class:`ValueError` with the affected document path. Process
        environment variables do not participate in resolution.
        """
        if not isinstance(raw, Mapping):
            raise _type_error("config", "an object")
        if defaults is not None and not isinstance(defaults, Mapping):
            raise _type_error("defaults", "an object")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_document", _resolved_document(raw, defaults))
        return instance

    @classmethod
    def scaffold(cls, *, defaults: Mapping[str, object] | None = None) -> dict[str, JsonValue]:
        """Return a complete valid mapping with stable, explicit defaults."""
        return cls.from_mapping({}, defaults=defaults).to_mapping()

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return a deep copy of the complete canonical document."""
        return deepcopy(self._document)

    @property
    def secret_env_names(self) -> tuple[str, ...]:
        """Return distinct credential-variable names in canonical order."""
        names: list[str] = []
        for group_name in ("embedding", "judge"):
            reference = _group(self._document, group_name)["api_key_env"]
            if isinstance(reference, str) and reference not in names:
                names.append(reference)
        return tuple(names)

    @property
    def required_extras(self) -> frozenset[Literal["remote", "embedding"]]:
        """Return optional Goldfive extras required by selected backends."""
        required: set[Literal["remote", "embedding"]] = set()
        embedding = _group(self._document, "embedding")
        judge = _group(self._document, "judge")
        reasoning = _group(self._document, "reasoning_drift")
        if judge["base_url"] is not None or embedding["base_url"] is not None:
            required.add("remote")
        if reasoning["mode"] in {"embedding", "both"} and embedding["base_url"] is None:
            required.add("embedding")
        return frozenset(required)

    def missing_runtime_capabilities(self) -> tuple[str, ...]:
        """Return selected backend capabilities unavailable in this process.

        Capability identifiers are ``remote_judge``, ``remote_embedding``, and
        ``local_embedding``. A remote judge needs ``openai``. A remote
        embedding endpoint can use either ``openai`` or its ``httpx`` fallback.
        Local embedding needs ``sentence-transformers``.
        """
        missing: list[str] = []
        embedding = _group(self._document, "embedding")
        judge = _group(self._document, "judge")
        reasoning = _group(self._document, "reasoning_drift")
        has_remote_judge = judge["base_url"] is not None
        has_remote_embedding = embedding["base_url"] is not None
        openai_available = (
            _module_available("openai") if has_remote_judge or has_remote_embedding else False
        )
        if has_remote_judge and not openai_available:
            missing.append("remote_judge")
        if has_remote_embedding and not openai_available and not _module_available("httpx"):
            missing.append("remote_embedding")
        if (
            reasoning["mode"] in {"embedding", "both"}
            and embedding["base_url"] is None
            and not _module_available("sentence_transformers")
        ):
            missing.append("local_embedding")
        return tuple(missing)

    def build(self, *, resolve_secret: SecretResolver) -> RuntimeConfig:
        """Build a ``RuntimeConfig``, resolving credentials at this boundary.

        A missing credential raises :class:`ValueError` that names its
        variable. Resolver failures are replaced with the same variable-only
        error, so an exception from a resolver cannot disclose a credential
        value.
        """
        resolved_credentials: dict[str, str] = {}
        for reference in self.secret_env_names:
            try:
                value = resolve_secret(reference)
            except Exception:  # noqa: BLE001 - replace unsafe resolver errors
                raise ValueError(
                    f"credential variable {reference!r} could not be resolved"
                ) from None
            if not isinstance(value, str) or not value:
                raise ValueError(f"credential variable {reference!r} could not be resolved")
            resolved_credentials[reference] = value

        return RuntimeConfig(
            **_build_runtime_groups(self._document, resolved_credentials),
            **_runtime_kwargs(
                self._document,
                omit=frozenset(_RUNTIME_GROUP_TYPES),
            ),
        )
