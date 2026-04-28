"""Judge token caps for Qwen 3.5 thinking models (goldfive#271 follow-up).

Empirical context (v16 on Qwen 35B): the reasoning + goal-drift judges
returned ``raw=''`` because Qwen 3.5 thinking models share ``<think>``
+ answer under one ``max_output_tokens`` ceiling. A 2048-token cap was
exhausted inside the think block before any JSON was emitted, so
goldfive's parser fell through to "no drift" and the cascade never
fired.

These tests pin the post-fix caps (16k for the judges / planner /
reflective check, 8k for the goal deriver) AND assert each call site
threads the cap through :func:`goldfive._llm.call_llm_budget` so a
wrapping ``call_llm`` shim sees the per-callsite value.

Each consumer's ``call_llm`` is replaced with a stub that captures
``get_max_output_tokens()`` at the moment the judge dispatches. That is
the only thing the underlying SDK builders ever see, so it is the only
contract we need to pin — the actual SDK plumbing is exercised by
:mod:`tests.test_llm_call_budget_integration`.
"""

from __future__ import annotations

import pytest

from goldfive._llm import get_max_output_tokens

# ---------------------------------------------------------------------------
# Constant pins — fail loudly if anyone halves the cap and reintroduces
# the v16 / Qwen 35B empty-judge regression.
# ---------------------------------------------------------------------------


def test_reasoning_judge_cap_is_16k():
    from goldfive.drift.reasoning_judge import REASONING_JUDGE_MAX_OUTPUT_TOKENS

    assert REASONING_JUDGE_MAX_OUTPUT_TOKENS == 16384


def test_goal_drift_judge_cap_is_16k():
    from goldfive.drift.goals import GOAL_DRIFT_MAX_OUTPUT_TOKENS

    assert GOAL_DRIFT_MAX_OUTPUT_TOKENS == 16384


def test_llm_planner_cap_is_16k():
    from goldfive.planner import LLMPlanner

    assert LLMPlanner.MAX_OUTPUT_TOKENS == 16384


def test_reflective_check_cap_is_16k():
    from goldfive.steerer import DefaultSteerer

    assert DefaultSteerer.REFLECTIVE_MAX_OUTPUT_TOKENS == 16384


def test_goal_deriver_cap_is_8k():
    from goldfive.goal_deriver import LLMGoalDeriver

    assert LLMGoalDeriver.MAX_OUTPUT_TOKENS == 8192


# ---------------------------------------------------------------------------
# Threading: each call site must set the ContextVar around its
# ``await call_llm(...)``. We replace ``call_llm`` with a stub that
# captures ``get_max_output_tokens()`` at invocation time and assert the
# captured value matches the consumer's class-attribute cap.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_judge_threads_16k_budget():
    """``classify_reasoning_drift`` must set ``call_llm_budget(16384)``
    around its ``await call_llm(...)`` so a wrapping shim — which is
    what ADK / OpenAI builders are — applies the cap to the SDK call."""
    from goldfive.drift.reasoning_judge import (
        REASONING_JUDGE_MAX_OUTPUT_TOKENS,
        classify_reasoning_drift,
    )
    from goldfive.types import Task

    seen_budgets: list[int] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_budgets.append(get_max_output_tokens())
        # Quiet-on-failure: an empty response means no drift emitted
        # (matches the v16 / 35B observation), but the call itself
        # still happened so we capture the budget.
        return ""

    drift = await classify_reasoning_drift(
        reasoning="some reasoning to judge",
        task=Task(id="t1", title="Ship the feature"),
        goals=None,
        model="x",
        call_llm=stub_call_llm,
    )
    assert drift is None
    assert seen_budgets == [REASONING_JUDGE_MAX_OUTPUT_TOKENS]
    assert seen_budgets == [16384]


@pytest.mark.asyncio
async def test_goal_drift_judge_threads_16k_budget():
    """``classify_goal_drift`` must set ``call_llm_budget(16384)``."""
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
        observed_actions=[
            {"kind": "tool_call", "agent_name": "agent", "detail": "ran a tool"}
        ],
        call_llm=stub_call_llm,
        model="x",
        run_id="r",
        session_id="s",
    )
    assert drift is None  # progressing=True → no drift
    assert seen_budgets == [GOAL_DRIFT_MAX_OUTPUT_TOKENS]
    assert seen_budgets == [16384]


@pytest.mark.asyncio
async def test_planner_handle_turn_threads_16k_budget():
    """``LLMPlanner.handle_turn`` must set ``call_llm_budget(16384)``."""
    from goldfive.planner import LLMPlanner
    from goldfive.types import Plan, Session

    seen_budgets: list[int] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_budgets.append(get_max_output_tokens())
        return ""

    planner = LLMPlanner(call_llm=stub_call_llm, model="x")
    session = Session(run_id="r1")
    session.plan = Plan.empty(run_id="r1")
    result = await planner.handle_turn(user_input="hello", session=session)
    assert result is None
    assert seen_budgets == [LLMPlanner.MAX_OUTPUT_TOKENS]
    assert seen_budgets == [16384]


@pytest.mark.asyncio
async def test_goal_deriver_threads_8k_budget():
    """``LLMGoalDeriver.derive`` must set ``call_llm_budget(8192)``."""
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
