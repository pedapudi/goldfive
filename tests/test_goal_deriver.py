"""Tests for ``goldfive.goal_deriver``.

Covers all three deriver variants. We use plain stubs for ``call_llm``; no
network or real LLM is touched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from goldfive.goal_deriver import (
    DEFAULT_SYSTEM_PROMPT,
    LiteralGoalDeriver,
    LLMGoalDeriver,
    PassthroughGoalDeriver,
)
from goldfive.types import Goal


# ---------------------------------------------------------------------------
# PassthroughGoalDeriver
# ---------------------------------------------------------------------------
class TestPassthroughGoalDeriver:
    async def test_single_string_is_wrapped_as_one_goal(self) -> None:
        deriver = PassthroughGoalDeriver("ship the thing")
        goals = await deriver.derive("ignored input")
        assert goals == [Goal(id="g1", summary="ship the thing")]

    async def test_list_of_strings_each_become_a_goal(self) -> None:
        deriver = PassthroughGoalDeriver(["one", "two", "three"])
        goals = await deriver.derive("ignored")
        assert [g.summary for g in goals] == ["one", "two", "three"]
        assert [g.id for g in goals] == ["g1", "g2", "g3"]

    async def test_list_of_goal_objects_returned_verbatim(self) -> None:
        g1 = Goal(id="top", summary="be awesome", metadata={"priority": "high"})
        g2 = Goal(id="other", summary="write docs")
        deriver = PassthroughGoalDeriver([g1, g2])
        goals = await deriver.derive("anything")
        assert goals == [g1, g2]

    async def test_user_input_is_ignored(self) -> None:
        deriver = PassthroughGoalDeriver("fixed goal")
        a = await deriver.derive("first")
        b = await deriver.derive("second")
        assert a == b

    async def test_returned_list_is_independent_copy(self) -> None:
        """Mutating the returned list must not affect subsequent calls."""
        deriver = PassthroughGoalDeriver(["a", "b"])
        first = await deriver.derive("x")
        first.append(Goal(id="gX", summary="smuggled"))
        second = await deriver.derive("x")
        assert len(second) == 2

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            PassthroughGoalDeriver("")
        with pytest.raises(ValueError):
            PassthroughGoalDeriver("   ")

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError):
            PassthroughGoalDeriver([])

    async def test_mixed_list_of_strings_and_goals(self) -> None:
        g = Goal(id="custom", summary="custom summary")
        deriver = PassthroughGoalDeriver(["plain", g])
        goals = await deriver.derive("x")
        # String item -> Goal(id="g1", summary="plain"); Goal item -> verbatim.
        assert goals[0] == Goal(id="g1", summary="plain")
        assert goals[1] is g

    def test_non_string_non_list_rejected(self) -> None:
        with pytest.raises(TypeError):
            PassthroughGoalDeriver(42)  # type: ignore[arg-type]

    def test_list_with_invalid_item_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            PassthroughGoalDeriver(["ok", 99])  # type: ignore[list-item]

    def test_list_with_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            PassthroughGoalDeriver(["ok", ""])


# ---------------------------------------------------------------------------
# LiteralGoalDeriver
# ---------------------------------------------------------------------------
class TestLiteralGoalDeriver:
    async def test_wraps_user_input_as_single_goal(self) -> None:
        deriver = LiteralGoalDeriver()
        goals = await deriver.derive("write a haiku about cats")
        assert goals == [Goal(id="g1", summary="write a haiku about cats")]

    async def test_empty_string_rejected(self) -> None:
        deriver = LiteralGoalDeriver()
        with pytest.raises(ValueError):
            await deriver.derive("")

    async def test_whitespace_only_rejected(self) -> None:
        deriver = LiteralGoalDeriver()
        with pytest.raises(ValueError):
            await deriver.derive("   \n\t")

    async def test_non_string_rejected(self) -> None:
        deriver = LiteralGoalDeriver()
        with pytest.raises(ValueError):
            await deriver.derive(None)  # type: ignore[arg-type]

    async def test_context_is_ignored(self) -> None:
        deriver = LiteralGoalDeriver()
        goals_no_ctx = await deriver.derive("x")
        goals_ctx = await deriver.derive("x", context={"anything": "here"})
        assert goals_no_ctx == goals_ctx


# ---------------------------------------------------------------------------
# LLMGoalDeriver
# ---------------------------------------------------------------------------
class _CannedLLM:
    """Stub ``call_llm`` that records invocations and returns canned text."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, prompt: str, model: str) -> str:
        self.calls.append((system, prompt, model))
        return self.response


class _RaisingLLM:
    async def __call__(self, system: str, prompt: str, model: str) -> str:
        raise RuntimeError("boom")


class TestLLMGoalDeriver:
    async def test_single_goal_parsed(self) -> None:
        stub = _CannedLLM(json.dumps({"goals": [{"id": "g1", "summary": "do the thing"}]}))
        deriver = LLMGoalDeriver(stub, model="test-model")
        goals = await deriver.derive("please do the thing")
        assert goals == [Goal(id="g1", summary="do the thing")]
        # The default system prompt was used.
        assert stub.calls[0][0] == DEFAULT_SYSTEM_PROMPT
        # The user input is embedded in the prompt.
        assert "please do the thing" in stub.calls[0][1]
        # Model propagated.
        assert stub.calls[0][2] == "test-model"

    async def test_multiple_goals_parsed(self) -> None:
        stub = _CannedLLM(
            json.dumps(
                {
                    "goals": [
                        {"id": "g1", "summary": "ship feature"},
                        {"id": "g2", "summary": "write blog post"},
                    ]
                }
            )
        )
        deriver = LLMGoalDeriver(stub)
        goals = await deriver.derive("ship feature and announce it")
        assert [g.id for g in goals] == ["g1", "g2"]
        assert [g.summary for g in goals] == ["ship feature", "write blog post"]

    async def test_metadata_coerced_to_str(self) -> None:
        stub = _CannedLLM(
            json.dumps(
                {
                    "goals": [
                        {
                            "id": "g1",
                            "summary": "x",
                            "metadata": {"priority": 1, "owner": "alice"},
                        }
                    ]
                }
            )
        )
        deriver = LLMGoalDeriver(stub)
        goals = await deriver.derive("x")
        assert goals[0].metadata == {"priority": "1", "owner": "alice"}

    async def test_response_with_code_fences_is_parsed(self) -> None:
        fenced = (
            "```json\n"
            + json.dumps({"goals": [{"id": "g1", "summary": "clean"}]})
            + "\n```"
        )
        stub = _CannedLLM(fenced)
        deriver = LLMGoalDeriver(stub)
        goals = await deriver.derive("clean the house")
        assert goals == [Goal(id="g1", summary="clean")]

    async def test_response_with_bare_code_fence_is_parsed(self) -> None:
        fenced = "```\n" + json.dumps({"goals": [{"id": "g1", "summary": "ok"}]}) + "\n```"
        deriver = LLMGoalDeriver(_CannedLLM(fenced))
        goals = await deriver.derive("do it")
        assert goals[0].summary == "ok"

    async def test_missing_id_gets_auto_assigned(self) -> None:
        stub = _CannedLLM(json.dumps({"goals": [{"summary": "no id here"}]}))
        deriver = LLMGoalDeriver(stub)
        goals = await deriver.derive("no id here")
        assert goals == [Goal(id="g1", summary="no id here")]

    async def test_custom_system_prompt(self) -> None:
        stub = _CannedLLM(json.dumps({"goals": [{"id": "g1", "summary": "ok"}]}))
        deriver = LLMGoalDeriver(stub, system_prompt="CUSTOM")
        await deriver.derive("hi")
        assert stub.calls[0][0] == "CUSTOM"

    async def test_context_appears_in_prompt(self) -> None:
        stub = _CannedLLM(json.dumps({"goals": [{"id": "g1", "summary": "ok"}]}))
        deriver = LLMGoalDeriver(stub)
        await deriver.derive("hi", context={"prior_goal": "foo"})
        assert "prior_goal" in stub.calls[0][1]
        assert "foo" in stub.calls[0][1]

    async def test_unserialisable_context_does_not_crash(self) -> None:
        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        stub = _CannedLLM(json.dumps({"goals": [{"id": "g1", "summary": "ok"}]}))
        deriver = LLMGoalDeriver(stub)
        await deriver.derive("hi", context={"obj": Weird()})
        # Either a str() fallback or json with default=str kicked in — both
        # should end up producing SOME non-crashing prompt.
        assert "hi" in stub.calls[0][1]

    # ---- fallback paths ---------------------------------------------------
    async def test_llm_exception_falls_back_to_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        deriver = LLMGoalDeriver(_RaisingLLM())
        with caplog.at_level("WARNING"):
            goals = await deriver.derive("just do X")
        assert goals == [Goal(id="g1", summary="just do X")]
        assert any("falling back" in rec.message for rec in caplog.records)

    async def test_non_json_response_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        deriver = LLMGoalDeriver(_CannedLLM("not json at all"))
        with caplog.at_level("WARNING"):
            goals = await deriver.derive("hello")
        assert goals == [Goal(id="g1", summary="hello")]
        assert any("parse" in rec.message for rec in caplog.records)

    async def test_wrong_schema_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        # "goals" field missing.
        deriver = LLMGoalDeriver(_CannedLLM(json.dumps({"tasks": []})))
        with caplog.at_level("WARNING"):
            goals = await deriver.derive("hello")
        assert goals == [Goal(id="g1", summary="hello")]

    async def test_empty_goals_array_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        deriver = LLMGoalDeriver(_CannedLLM(json.dumps({"goals": []})))
        with caplog.at_level("WARNING"):
            goals = await deriver.derive("hello")
        assert goals == [Goal(id="g1", summary="hello")]
        assert any("zero goals" in rec.message for rec in caplog.records)

    async def test_goal_with_empty_summary_falls_back(self) -> None:
        deriver = LLMGoalDeriver(
            _CannedLLM(json.dumps({"goals": [{"id": "g1", "summary": "  "}]}))
        )
        goals = await deriver.derive("real input")
        assert goals == [Goal(id="g1", summary="real input")]

    async def test_non_object_goal_item_falls_back(self) -> None:
        deriver = LLMGoalDeriver(
            _CannedLLM(json.dumps({"goals": ["not-an-object"]}))
        )
        goals = await deriver.derive("real input")
        assert goals == [Goal(id="g1", summary="real input")]

    async def test_empty_user_input_rejected(self) -> None:
        deriver = LLMGoalDeriver(_CannedLLM(""))
        with pytest.raises(ValueError):
            await deriver.derive("")

    async def test_default_model_is_empty_string(self) -> None:
        stub = _CannedLLM(json.dumps({"goals": [{"id": "g1", "summary": "ok"}]}))
        deriver = LLMGoalDeriver(stub)
        await deriver.derive("hi")
        assert stub.calls[0][2] == ""


# ---------------------------------------------------------------------------
# Sanity: all three derivers satisfy the expected shape (structural check).
# ---------------------------------------------------------------------------
async def test_all_derivers_have_async_derive_returning_list_of_goal() -> None:
    class _OkLLM:
        async def __call__(self, system: str, prompt: str, model: str) -> str:
            return json.dumps({"goals": [{"id": "g1", "summary": "ok"}]})

    derivers: list[Any] = [
        PassthroughGoalDeriver("x"),
        LiteralGoalDeriver(),
        LLMGoalDeriver(_OkLLM()),
    ]
    for d in derivers:
        out = await d.derive("hello world")
        assert isinstance(out, list)
        assert all(isinstance(g, Goal) for g in out)
        assert len(out) >= 1
