"""Judge dispatches must run with thinking disabled (goldfive#271 follow-up to #311).

#311 raised the judge / planner / goal-deriver max_output_tokens caps from
2048 to 16384 because v16 / Qwen 35B was returning ``raw=''`` — the
*symptom* fix. The *cause* is that goldfive's judges are meta-cognition
(small JSON questions like "is this on-task?") and have no business
running through Qwen / Gemini "thinking" mode at all. Letting the model
share the 16k cap with ``<think>`` reasoning is the same failure mode,
just one cap-bump away.

These tests pin the contract:

1. ``call_llm_thinking_disabled()`` flips ``THINKING_DISABLED_VAR`` on
   for the duration of the with-block.
2. The default ADK builder reads the var and attaches
   ``ThinkingConfig(include_thoughts=False, thinking_budget=0)`` to the
   ``GenerateContentConfig`` it sends to ``BaseLlm.generate_content_async``.
3. The default OpenAI / Qwen-via-litellm judge builder reads the var
   and sends ``extra_body={"enable_thinking": False}`` plus a
   ``/no_think`` system-prompt prefix.
4. Each judge / goal_deriver / planner-refine call site enters
   :func:`call_llm_thinking_disabled` around its
   ``await call_llm(...)`` so a wrapping shim sees the flag.
5. Agent-side dispatches (no enclosing context manager) keep thinking
   enabled — that's the user's model behaviour, not goldfive's choice.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from goldfive._llm import (
    call_llm_thinking_disabled,
    get_thinking_disabled,
)

# ---------------------------------------------------------------------------
# ContextVar plumbing
# ---------------------------------------------------------------------------


def test_default_thinking_is_enabled():
    """Outside any with-block, ``get_thinking_disabled()`` returns ``False``.

    Agent-side LLM calls (coordinator / research / web_developer) keep
    their natural thinking behaviour. Goldfive only flips the flag for
    its own meta-cognition dispatches.
    """
    assert get_thinking_disabled() is False


def test_call_llm_thinking_disabled_flips_var():
    """``call_llm_thinking_disabled()`` flips the var inside the block."""
    assert get_thinking_disabled() is False
    with call_llm_thinking_disabled():
        assert get_thinking_disabled() is True
    assert get_thinking_disabled() is False


def test_call_llm_thinking_disabled_resets_on_exception():
    """The var is reset even when the body raises."""
    with pytest.raises(RuntimeError):
        with call_llm_thinking_disabled():
            assert get_thinking_disabled() is True
            raise RuntimeError("boom")
    assert get_thinking_disabled() is False


# ---------------------------------------------------------------------------
# ADK builder integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adk_builder_attaches_thinking_config_when_disabled():
    """``make_default_adk_call_llm`` must attach a ``ThinkingConfig`` with
    ``include_thoughts=False, thinking_budget=0`` when the per-callsite
    disable-thinking flag is set."""
    pytest.importorskip("google.adk")
    pytest.importorskip("google.genai")

    from goldfive._llm_detect import make_default_adk_call_llm

    captured_requests: list[Any] = []

    from google.adk.models.base_llm import BaseLlm  # type: ignore[import-not-found]
    from google.adk.models.llm_response import (  # type: ignore[import-not-found]
        LlmResponse,
    )
    from google.genai import types as genai_types  # type: ignore[import-not-found]

    class _StubLLM(BaseLlm):
        async def generate_content_async(self, req, stream=False):  # type: ignore[override]
            captured_requests.append(req)
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
            )

    stub = _StubLLM(model="stub")
    call_llm = make_default_adk_call_llm(stub)
    assert call_llm is not None

    # Default (no disable-thinking flag) — no ThinkingConfig on the request.
    out = await call_llm("system", "user", "stub")
    assert out == "ok"
    last_config = captured_requests[-1].config
    assert getattr(last_config, "thinking_config", None) is None, (
        "agent-side calls must keep their natural thinking behaviour"
    )

    # With disable-thinking flag — ThinkingConfig must be present and
    # configured to suppress the think prelude entirely.
    with call_llm_thinking_disabled():
        await call_llm("system", "user", "stub")
    last_config = captured_requests[-1].config
    tc = getattr(last_config, "thinking_config", None)
    assert tc is not None, "judge-side dispatches must attach ThinkingConfig"
    assert tc.include_thoughts is False
    assert tc.thinking_budget == 0


# ---------------------------------------------------------------------------
# OpenAI / Qwen builder integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_builder_threads_enable_thinking_false_when_disabled():
    """``_build_judge_call_llm`` must send ``extra_body={"enable_thinking":
    False}`` and prepend ``/no_think`` to the system prompt when the
    disable-thinking flag is set."""
    pytest.importorskip("openai")
    from goldfive.config import JudgeConfig
    from goldfive.convenience import _build_judge_call_llm

    config = JudgeConfig(
        base_url="http://stub-judge.invalid",
        model="stub-judge",
        api_key="not-needed",
    )

    built = _build_judge_call_llm(config)
    assert built is not None
    call_llm, _model = built

    captured_kwargs: list[dict[str, Any]] = []
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]
    # No reasoning_content attribute — typical for non-thinking models.
    type(fake_response.choices[0].message).reasoning_content = ""  # type: ignore[attr-defined]

    async def fake_create(**kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return fake_response

    client_cell = None
    for c in call_llm.__closure__ or ():
        if hasattr(c.cell_contents, "chat"):
            client_cell = c
            break
    assert client_cell is not None
    client_cell.cell_contents.chat.completions.create = fake_create

    # Default: no extra_body, no /no_think prefix.
    out = await call_llm("system", "user", "stub-judge")
    assert out == "ok"
    assert "extra_body" not in captured_kwargs[-1]
    sys_msg = captured_kwargs[-1]["messages"][0]["content"]
    assert "/no_think" not in sys_msg

    # With disable-thinking flag: extra_body present, /no_think prepended.
    with call_llm_thinking_disabled():
        await call_llm("system", "user", "stub-judge")
    last = captured_kwargs[-1]
    assert last["extra_body"] == {"enable_thinking": False}
    sys_msg = last["messages"][0]["content"]
    assert sys_msg.startswith("/no_think"), "Qwen prompt-level fallback must be prepended"


# ---------------------------------------------------------------------------
# Per-call-site contract: each judge / goal-deriver / planner-refine site
# must enter ``call_llm_thinking_disabled()`` around its
# ``await call_llm(...)``. Same shape as ``test_judge_token_caps.py`` but
# capturing the disable-thinking flag rather than the token cap.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_judge_disables_thinking():
    """``classify_reasoning_drift`` must enter
    ``call_llm_thinking_disabled()`` around its ``await call_llm(...)``."""
    from goldfive.drift.reasoning_judge import classify_reasoning_drift
    from goldfive.types import Task

    seen_flags: list[bool] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_flags.append(get_thinking_disabled())
        return ""

    drift = await classify_reasoning_drift(
        reasoning="some reasoning to judge",
        task=Task(id="t1", title="Ship the feature"),
        goals=None,
        model="x",
        call_llm=stub_call_llm,
    )
    assert drift is None
    assert seen_flags == [True]


@pytest.mark.asyncio
async def test_goal_drift_judge_disables_thinking():
    """``classify_goal_drift`` must enter
    ``call_llm_thinking_disabled()``."""
    from goldfive.drift.goals import classify_goal_drift
    from goldfive.types import Goal

    seen_flags: list[bool] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_flags.append(get_thinking_disabled())
        return '{"progressing": true}'

    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship")],
        plan=None,
        observed_actions=[{"kind": "tool_call", "agent_name": "agent", "detail": "ran a tool"}],
        call_llm=stub_call_llm,
        model="x",
        run_id="r",
        session_id="s",
    )
    assert drift is None
    assert seen_flags == [True]


@pytest.mark.asyncio
async def test_planner_handle_turn_disables_thinking():
    """``LLMPlanner.handle_turn`` must enter
    ``call_llm_thinking_disabled()``."""
    from goldfive.planner import LLMPlanner
    from goldfive.types import Plan, Session

    seen_flags: list[bool] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_flags.append(get_thinking_disabled())
        return ""

    planner = LLMPlanner(call_llm=stub_call_llm, model="x")
    session = Session(run_id="r1")
    session.plan = Plan.empty(run_id="r1")
    result = await planner.handle_turn(user_input="hello", session=session)
    assert result is None
    assert seen_flags == [True]


@pytest.mark.asyncio
async def test_goal_deriver_disables_thinking():
    """``LLMGoalDeriver.derive`` must enter
    ``call_llm_thinking_disabled()``."""
    from goldfive.goal_deriver import LLMGoalDeriver

    seen_flags: list[bool] = []

    async def stub_call_llm(system: str, user: str, model: str) -> str:
        seen_flags.append(get_thinking_disabled())
        return '{"goals": [{"id": "g1", "summary": "test"}]}'

    deriver = LLMGoalDeriver(stub_call_llm)
    goals = await deriver.derive("hello world")
    assert len(goals) == 1
    assert seen_flags == [True]
