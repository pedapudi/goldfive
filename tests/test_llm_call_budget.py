"""Per-LLM-call max_output_tokens budget (goldfive#271 follow-up).

Pre-fix: goldfive's own LLM dispatches had no upper bound on the LLM's
output, leaving plumbing-level ``call_llm`` callables to set their own
caps. The default ADK / OpenAI builders did not, which on a Qwen Q4
endpoint produced 9961-token / 9.6-minute responses (demo-v8.log).

These tests pin:

1. The :data:`goldfive._llm.MAX_OUTPUT_TOKENS_VAR` ContextVar threading
   from each consumer to the shared cap helper.
2. The default-ADK ``call_llm`` builder reading the var and applying
   it to ``GenerateContentConfig.max_output_tokens``.
3. The default-OpenAI ``call_llm`` builder reading the var and applying
   it to ``client.chat.completions.create(max_tokens=...)``.
4. Per-callsite class attributes wiring the right cap into each consumer
   (``LLMPlanner.MAX_OUTPUT_TOKENS``, ``LLMGoalDeriver.MAX_OUTPUT_TOKENS``,
   reasoning + goal drift judges, reflective check).
"""

from __future__ import annotations

import pytest

from goldfive._llm import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS_VAR,
    call_llm_budget,
    get_max_output_tokens,
)


def test_default_max_output_tokens_is_4096():
    """4096 is the agreed default — large enough for plan refines /
    handle_turn, small enough that wall-clock is bounded at ~4 minutes
    against a Q4 endpoint at typical 17 tok/sec."""
    assert DEFAULT_MAX_OUTPUT_TOKENS == 4096


def test_get_max_output_tokens_falls_back_when_unset():
    """Outside any ``call_llm_budget`` block, the helper returns the default."""
    assert MAX_OUTPUT_TOKENS_VAR.get() is None
    assert get_max_output_tokens() == DEFAULT_MAX_OUTPUT_TOKENS


def test_get_max_output_tokens_returns_set_value():
    """Inside a ``call_llm_budget(N)`` block, the helper returns ``N``."""
    with call_llm_budget(2048):
        assert get_max_output_tokens() == 2048
    # Restored on exit.
    assert get_max_output_tokens() == DEFAULT_MAX_OUTPUT_TOKENS


def test_call_llm_budget_resets_on_exception():
    """Even when the body raises, the var resets cleanly."""
    with pytest.raises(RuntimeError, match="boom"):
        with call_llm_budget(1024):
            assert get_max_output_tokens() == 1024
            raise RuntimeError("boom")
    assert get_max_output_tokens() == DEFAULT_MAX_OUTPUT_TOKENS


def test_call_llm_budget_nests():
    """Nested budgets restore the outer value on exit."""
    with call_llm_budget(4096):
        assert get_max_output_tokens() == 4096
        with call_llm_budget(1024):
            assert get_max_output_tokens() == 1024
        assert get_max_output_tokens() == 4096
    assert get_max_output_tokens() == DEFAULT_MAX_OUTPUT_TOKENS


def test_call_llm_budget_none_resets_to_default():
    """``None`` inside the block falls back to the default."""
    with call_llm_budget(None):
        assert get_max_output_tokens() == DEFAULT_MAX_OUTPUT_TOKENS


def test_call_llm_budget_zero_or_negative_falls_back():
    """Zero / negative values are treated as "no cap" — use the default
    rather than passing 0 to the underlying SDK (which would refuse to
    emit anything)."""
    with call_llm_budget(0):
        assert get_max_output_tokens() == DEFAULT_MAX_OUTPUT_TOKENS
    with call_llm_budget(-1):
        assert get_max_output_tokens() == DEFAULT_MAX_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# Per-consumer class-attribute caps
# ---------------------------------------------------------------------------


def test_llm_planner_cap_is_16384():
    from goldfive.planner import LLMPlanner

    assert LLMPlanner.MAX_OUTPUT_TOKENS == 16384


def test_llm_goal_deriver_cap_is_8192():
    from goldfive.goal_deriver import LLMGoalDeriver

    assert LLMGoalDeriver.MAX_OUTPUT_TOKENS == 8192


def test_goal_drift_cap_is_16384():
    from goldfive.drift.goals import GOAL_DRIFT_MAX_OUTPUT_TOKENS

    assert GOAL_DRIFT_MAX_OUTPUT_TOKENS == 16384


def test_reasoning_judge_cap_is_16384():
    from goldfive.drift.reasoning_judge import REASONING_JUDGE_MAX_OUTPUT_TOKENS

    assert REASONING_JUDGE_MAX_OUTPUT_TOKENS == 16384


def test_reflective_check_cap_is_16384():
    from goldfive.drift_observer import DriftObserver

    assert DriftObserver.REFLECTIVE_MAX_OUTPUT_TOKENS == 16384


# ---------------------------------------------------------------------------
# Consumer threading: planner / goal_deriver / judges / reflective check
# all set the var around their ``await call_llm(...)`` so a wrapping
# ``call_llm`` shim sees the cap.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_handle_turn_threads_budget():
    """``LLMPlanner.handle_turn`` must set the budget around its
    ``call_llm`` so a built-in ADK / OpenAI builder applies the cap."""
    from goldfive.planner import LLMPlanner
    from goldfive.types import Plan, Session

    seen_budgets: list[int] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_budgets.append(get_max_output_tokens())
        # handle_turn returns None on missing/unparseable LLM responses;
        # an empty string is one such trigger.
        return ""

    planner = LLMPlanner(call_llm=stub_call_llm, model="x")
    session = Session(run_id="r1")
    session.plan = Plan.empty(run_id="r1")
    result = await planner.handle_turn(
        user_input="hello",
        session=session,
    )
    assert result is None  # empty LLM response → no plan
    assert seen_budgets == [LLMPlanner.MAX_OUTPUT_TOKENS]


@pytest.mark.asyncio
async def test_goal_deriver_threads_budget():
    """``LLMGoalDeriver.derive`` must set the goal-deriver cap (1024)."""
    from goldfive.goal_deriver import LLMGoalDeriver

    seen_budgets: list[int] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_budgets.append(get_max_output_tokens())
        return '{"goals": [{"id": "g1", "summary": "test"}]}'

    deriver = LLMGoalDeriver(stub_call_llm)
    goals = await deriver.derive("hello world")
    assert len(goals) == 1
    assert seen_budgets == [LLMGoalDeriver.MAX_OUTPUT_TOKENS]
    assert seen_budgets == [8192]


@pytest.mark.asyncio
async def test_goal_drift_judge_threads_budget():
    """``classify_goal_drift`` must set the goal-drift cap (2048)."""
    from goldfive.drift.goals import (
        GOAL_DRIFT_MAX_OUTPUT_TOKENS,
        classify_goal_drift,
    )
    from goldfive.types import Goal

    seen_budgets: list[int] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_budgets.append(get_max_output_tokens())
        return '{"progressing": true, "reason": "looks fine"}'

    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship")],
        plan=None,
        observed_actions=[{"kind": "tool_call", "agent_name": "agent", "detail": "ran a tool"}],
        call_llm=stub_call_llm,
        model="x",
        run_id="r",
        session_id="s",
    )
    # progressing=True → no drift emitted, but the call still happened.
    assert drift is None
    assert seen_budgets == [GOAL_DRIFT_MAX_OUTPUT_TOKENS]
