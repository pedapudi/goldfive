"""Unit tests for :mod:`goldfive.drift.registry`.

The registry centralises three pieces of boilerplate previously
duplicated across the goal-drift and reasoning-judge detectors:

* :func:`parse_json_response` — liberal JSON extractor.
* :func:`format_goals_block` — numbered ``[id] summary`` renderer.
* :func:`truncate_for_observability` — bounded text truncation with
  the uniform ``" … [truncated]"`` suffix.

The registry itself stores ``(DriftKind, classifier_fn, DetectorConfig)``
triples so callers can dispatch by kind via :func:`classify` without
importing the detector module directly. These tests cover the helper
behaviour (parity with the byte-identical pre-Wave-A copies) and the
register/dispatch contract.
"""

from __future__ import annotations

import pytest

from goldfive.drift import registry
from goldfive.drift.registry import (
    TRUNCATE_SUFFIX,
    DetectorConfig,
    classify,
    format_goals_block,
    get_config,
    list_registered,
    parse_json_response,
    register,
    truncate_for_observability,
)
from goldfive.types import DriftEvent, DriftKind, DriftSeverity

# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """Parity checks against the per-detector copies the registry replaces."""

    def test_plain_json_object(self) -> None:
        assert parse_json_response('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}

    def test_whitespace_trimmed_before_parse(self) -> None:
        assert parse_json_response('   {"x": true}\n') == {"x": True}

    def test_markdown_fence_falls_back_to_first_object(self) -> None:
        raw = """Here is the verdict:
```json
{"on_task": false, "reason": "off-topic"}
```
"""
        assert parse_json_response(raw) == {
            "on_task": False,
            "reason": "off-topic",
        }

    def test_prose_prefix_extracts_object(self) -> None:
        assert parse_json_response("verdict: {\"progressing\": true}") == {
            "progressing": True
        }

    def test_non_string_returns_none(self) -> None:
        assert parse_json_response(None) is None
        assert parse_json_response(42) is None
        assert parse_json_response({"already": "dict"}) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_json_response("") is None
        assert parse_json_response("   ") is None

    def test_garbage_returns_none(self) -> None:
        assert parse_json_response("not json at all") is None

    def test_top_level_non_object_returns_none(self) -> None:
        # The pre-Wave-A parser explicitly rejected lists/strings at the
        # top level — only dicts pass the type check. The "first ``{...}``"
        # fallback fires when ``json.loads`` of the entire string fails,
        # so a bare list never reaches the fallback path; we still want
        # the contract to hold.
        assert parse_json_response("[1, 2, 3]") is None

    def test_malformed_json_in_object_returns_none(self) -> None:
        assert parse_json_response("{not valid json}") is None


# ---------------------------------------------------------------------------
# format_goals_block
# ---------------------------------------------------------------------------


class _Goal:
    """Minimal duck-typed goal stand-in for parity checks."""

    def __init__(self, id: str = "", summary: str = "") -> None:
        self.id = id
        self.summary = summary


class TestFormatGoalsBlock:
    def test_empty_renders_placeholder(self) -> None:
        assert format_goals_block(None) == "(no goals recorded)"
        assert format_goals_block([]) == "(no goals recorded)"

    def test_ids_are_bracketed(self) -> None:
        goals = [_Goal(id="g-1", summary="ship feature"), _Goal(id="g-2", summary="write docs")]
        assert format_goals_block(goals) == "1. [g-1] ship feature\n2. [g-2] write docs"

    def test_missing_id_omits_brackets(self) -> None:
        goals = [_Goal(summary="ship feature")]
        assert format_goals_block(goals) == "1. ship feature"

    def test_string_goal_is_used_verbatim(self) -> None:
        assert format_goals_block(["just a string"]) == "1. just a string"

    def test_mixed_shape_renders(self) -> None:
        goals = [_Goal(id="g-1", summary="A"), "B"]
        assert format_goals_block(goals) == "1. [g-1] A\n2. B"


# ---------------------------------------------------------------------------
# truncate_for_observability
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert truncate_for_observability("hi", 100) == "hi"

    def test_long_text_truncated_with_suffix(self) -> None:
        text = "x" * 50
        out = truncate_for_observability(text, 10)
        assert out == "x" * 10 + TRUNCATE_SUFFIX
        assert TRUNCATE_SUFFIX in out

    def test_non_string_returns_empty(self) -> None:
        assert truncate_for_observability(None, 100) == ""
        assert truncate_for_observability(42, 100) == ""
        assert truncate_for_observability(["x"], 100) == ""

    def test_zero_limit_returns_input(self) -> None:
        # Per docstring: limit <= 0 → no truncation.
        assert truncate_for_observability("hello", 0) == "hello"
        assert truncate_for_observability("hello", -5) == "hello"

    def test_at_limit_unchanged(self) -> None:
        assert truncate_for_observability("xxxxx", 5) == "xxxxx"

    def test_suffix_is_pinned(self) -> None:
        # Pinning the literal value catches accidental edits to
        # ``TRUNCATE_SUFFIX`` since downstream UIs (harmonograf) match
        # against it as a sentinel.
        assert TRUNCATE_SUFFIX == " … [truncated]"


# ---------------------------------------------------------------------------
# register / classify
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Isolate registry state from the auto-registered detector entries.

    The detector modules register on import; we don't want to clobber
    those for tests that share the same process. This fixture swaps in
    a fresh dict for the duration of the test and restores on teardown.
    """
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    try:
        yield registry._REGISTRY
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


class TestRegisterAndClassify:
    def test_sync_classifier_dispatches(self, fresh_registry: dict) -> None:
        called: list[dict] = []

        def stub(**kwargs):
            called.append(kwargs)
            return DriftEvent(
                kind=DriftKind.TOOL_ERROR,
                severity=DriftSeverity.WARNING,
                detail="stub",
            )

        register(
            DriftKind.TOOL_ERROR,
            stub,
            DetectorConfig(uses_llm=False),
            is_async=False,
        )
        event = classify(kind=DriftKind.TOOL_ERROR, alpha=1, beta="two")
        assert isinstance(event, DriftEvent)
        assert event.kind is DriftKind.TOOL_ERROR
        assert called == [{"alpha": 1, "beta": "two"}]

    def test_async_classifier_returns_awaitable(self, fresh_registry: dict) -> None:
        async def stub(**_kwargs) -> DriftEvent | None:
            return None

        register(
            DriftKind.OFF_TOPIC,
            stub,
            DetectorConfig(uses_llm=True),
            is_async=True,
        )
        result = classify(kind=DriftKind.OFF_TOPIC, reasoning="x")
        # Async classifiers return a coroutine; the caller awaits it.
        import inspect

        assert inspect.iscoroutine(result)
        result.close()

    def test_unregistered_kind_raises(self, fresh_registry: dict) -> None:
        with pytest.raises(KeyError) as excinfo:
            classify(kind=DriftKind.TOOL_ERROR)
        assert "no drift detector registered" in str(excinfo.value)

    def test_get_config_returns_registered_config(
        self, fresh_registry: dict
    ) -> None:
        cfg = DetectorConfig(
            uses_llm=True,
            max_input_chars=512,
            max_output_tokens=4096,
            disable_thinking=True,
        )
        register(DriftKind.GOAL_DRIFT, lambda **_: None, cfg, is_async=False)
        assert get_config(DriftKind.GOAL_DRIFT) is cfg

    def test_get_config_returns_none_for_unknown_kind(
        self, fresh_registry: dict
    ) -> None:
        assert get_config(DriftKind.CAPABILITY_MISMATCH) is None

    def test_re_registration_overwrites(self, fresh_registry: dict) -> None:
        a = DetectorConfig(uses_llm=False, max_input_chars=1)
        b = DetectorConfig(uses_llm=False, max_input_chars=2)
        register(DriftKind.TOOL_ERROR, lambda **_: None, a, is_async=False)
        register(DriftKind.TOOL_ERROR, lambda **_: None, b, is_async=False)
        assert get_config(DriftKind.TOOL_ERROR) is b

    def test_list_registered_returns_insertion_order(
        self, fresh_registry: dict
    ) -> None:
        register(
            DriftKind.GOAL_DRIFT,
            lambda **_: None,
            DetectorConfig(),
            is_async=False,
        )
        register(
            DriftKind.OFF_TOPIC,
            lambda **_: None,
            DetectorConfig(),
            is_async=False,
        )
        assert list_registered() == (DriftKind.GOAL_DRIFT, DriftKind.OFF_TOPIC)


class TestAutoRegistration:
    """Detector modules self-register at import time."""

    def test_ensure_registered_populates_all_known_detectors(self) -> None:
        registry._ensure_registered()
        kinds = set(list_registered())
        # Every LLM-judge detector + the structural CAPABILITY_MISMATCH
        # rule should be discoverable via the registry after the
        # idempotent ``_ensure_registered`` call.
        assert DriftKind.GOAL_DRIFT in kinds
        assert DriftKind.OFF_TOPIC in kinds
        assert DriftKind.JUSTIFIED_DEVIATION in kinds
        assert DriftKind.CAPABILITY_MISMATCH in kinds

    def test_config_for_reasoning_judge_pins_input_cap(self) -> None:
        registry._ensure_registered()
        cfg = get_config(DriftKind.OFF_TOPIC)
        assert cfg is not None
        # Pinned by the reasoning-judge detector — bus contract that
        # observability emissions stay at 4096 chars.
        assert cfg.max_input_chars == 4096
        assert cfg.uses_llm is True
        assert cfg.disable_thinking is True

    def test_config_for_goal_drift_pins_trigger_input_cap(self) -> None:
        registry._ensure_registered()
        cfg = get_config(DriftKind.GOAL_DRIFT)
        assert cfg is not None
        assert cfg.max_input_chars == 2048
        assert cfg.uses_llm is True

    def test_config_for_capability_mismatch_is_structural(self) -> None:
        registry._ensure_registered()
        cfg = get_config(DriftKind.CAPABILITY_MISMATCH)
        assert cfg is not None
        assert cfg.uses_llm is False
        # Structural rules do not have an observability payload —
        # ``max_input_chars`` is therefore zero so the truncation helper
        # is a no-op for these detectors.
        assert cfg.max_input_chars == 0


# ---------------------------------------------------------------------------
# Parity check — the registry helpers must produce byte-identical output
# vs the per-detector copies they replaced
# ---------------------------------------------------------------------------


class TestSharedHelpersParityWithDetectors:
    """Sanity-check that goals/reasoning_judge still expose the helpers
    under the historical private names so external test fixtures that
    mock ``goals._parse_response`` etc. continue to work.
    """

    def test_goals_module_reexports_parse_response(self) -> None:
        from goldfive.drift import goals

        assert goals._parse_response is parse_json_response

    def test_goals_module_reexports_format_goals(self) -> None:
        from goldfive.drift import goals

        assert goals._format_goals is format_goals_block

    def test_reasoning_judge_reexports_parse_response(self) -> None:
        from goldfive.drift import reasoning_judge

        assert reasoning_judge._parse_response is parse_json_response

    def test_reasoning_judge_reexports_format_goals(self) -> None:
        from goldfive.drift import reasoning_judge

        assert reasoning_judge._format_goals is format_goals_block

    def test_reasoning_judge_reexports_truncate(self) -> None:
        from goldfive.drift import reasoning_judge

        assert reasoning_judge.truncate_for_observability is truncate_for_observability
