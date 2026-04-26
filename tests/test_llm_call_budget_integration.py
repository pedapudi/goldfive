"""Integration: ``max_output_tokens`` flows from goldfive consumer to
the underlying ADK / OpenAI client (goldfive#271 follow-up).

The unit tests in :mod:`tests.test_llm_call_budget` pin the ContextVar
threading inside each consumer. These integration tests verify the
*other* end of the contract — the default-ADK and default-OpenAI
``call_llm`` builders read the var and forward it to the SDK's
underlying call.

Without this end-to-end check, a regression that breaks the var-read
inside the builder would leave the unit tests passing while the wire
budget silently reverts to "no cap" — the exact pre-fix bug.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from goldfive._llm import call_llm_budget


@pytest.mark.asyncio
async def test_adk_builder_threads_max_output_tokens_to_genai_config():
    """``make_default_adk_call_llm`` builds a ``call_llm`` whose
    ``GenerateContentConfig`` carries the per-callsite ``max_output_tokens``."""
    pytest.importorskip("google.adk")
    pytest.importorskip("google.genai")

    from goldfive._llm_detect import make_default_adk_call_llm

    captured_requests: list[Any] = []

    # Build a fake BaseLlm that captures every LlmRequest passed in
    # and returns one minimal LlmResponse.
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

    # Default cap (no ContextVar set) — falls back to 4096.
    out = await call_llm("system", "user", "stub")
    assert out == "ok"
    assert captured_requests[-1].config.max_output_tokens == 4096

    # Per-callsite override (1024) — what LLMGoalDeriver would set.
    with call_llm_budget(1024):
        await call_llm("system", "user", "stub")
    assert captured_requests[-1].config.max_output_tokens == 1024

    # Per-callsite override (2048) — what reasoning/goal-drift judges
    # would set.
    with call_llm_budget(2048):
        await call_llm("system", "user", "stub")
    assert captured_requests[-1].config.max_output_tokens == 2048


@pytest.mark.asyncio
async def test_openai_builder_threads_max_tokens_to_chat_completions():
    """``_build_judge_call_llm`` builds a ``call_llm`` whose
    ``client.chat.completions.create`` is invoked with ``max_tokens=N``
    set from the per-callsite ContextVar."""
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

    # Replace the underlying AsyncOpenAI client with a mock so we can
    # capture the ``max_tokens`` kwarg on every dispatch without
    # actually hitting an HTTP endpoint. The builder stashes the
    # client in a closure — we reach in via the closure cell.
    captured_kwargs: list[dict[str, Any]] = []

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]

    async def fake_create(**kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return fake_response

    # The AsyncOpenAI client lives under the name "client" in the
    # builder closure — find by attribute presence rather than name
    # so a future rename of the local doesn't silently break this.
    client_cell = None
    for c in call_llm.__closure__ or ():
        if hasattr(c.cell_contents, "chat"):
            client_cell = c
            break
    assert client_cell is not None, (
        "could not find AsyncOpenAI client in _build_judge_call_llm closure"
    )
    client_cell.cell_contents.chat.completions.create = fake_create

    # Default cap (no ContextVar set) → 4096.
    out = await call_llm("system", "user", "stub-judge")
    assert out == "ok"
    assert captured_kwargs[-1]["max_tokens"] == 4096

    # Per-callsite 2048 (judge default).
    with call_llm_budget(2048):
        await call_llm("system", "user", "stub-judge")
    assert captured_kwargs[-1]["max_tokens"] == 2048
