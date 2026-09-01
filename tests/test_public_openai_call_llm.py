"""Public construction and lifecycle contract for OpenAI-compatible LLM calls."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

import goldfive
from goldfive._llm import (
    call_llm_budget,
    call_llm_thinking_disabled,
    llm_call_diagnostics,
)


def test_openai_call_llm_types_and_helpers_are_public() -> None:
    """Endpoint users must not import Goldfive's private LLM module."""
    expected = {
        "CallLLM",
        "ClosableCallLLM",
        "JudgeConfig",
        "make_default_openai_call_llm",
        "maybe_close_call_llm",
    }

    assert expected <= set(goldfive.__all__)
    for name in expected:
        assert getattr(goldfive, name) is not None


async def test_public_openai_builder_preserves_dispatch_and_caller_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public builder must retain Goldfive's dispatch logic and caller ownership."""
    clients: list[Any] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.requests: list[dict[str, Any]] = []
            self.close_count = 0
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )
            clients.append(self)

        async def _create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if "extra_body" in kwargs:
                raise TypeError("client does not accept extra_body")
            message = types.SimpleNamespace(content="", reasoning_content="reasoning")
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)]
            )

        async def close(self) -> None:
            self.close_count += 1

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai)

    built = goldfive.make_default_openai_call_llm(
        goldfive.JudgeConfig(
            base_url="http://judge.invalid/root/",
            model="qwen3-judge",
            api_key="secret",
            timeout_ms=2500,
        )
    )

    assert built is not None
    call_llm, model = built
    assert model == "qwen3-judge"
    assert len(clients) == 1
    client = clients[0]
    assert client.init_kwargs == {
        "base_url": "http://judge.invalid/root/v1",
        "api_key": "secret",
        "timeout": 2.5,
    }

    with (
        call_llm_budget(733),
        call_llm_thinking_disabled(),
        llm_call_diagnostics() as diag,
    ):
        result = await call_llm("judge instruction", "trajectory", model)

    assert result == ""
    assert diag.thought_count == 1
    assert diag.answer_count == 0
    assert len(client.requests) == 2
    assert client.requests[0]["extra_body"] == {"enable_thinking": False}
    assert "extra_body" not in client.requests[1]
    assert client.requests[1]["max_tokens"] == 733
    assert client.requests[1]["messages"][0]["content"].startswith("/no_think\n")

    async def agent(task: Any, _session: Any, _tools: Any) -> goldfive.InvocationResult:
        return goldfive.InvocationResult(task_id=task.id, text="unused")

    runner = goldfive.wrap(agent, judge_call_llm=call_llm, judge_model=model)
    await runner.close()

    # Supplying the callable does not transfer its lifecycle to the Runner.
    assert client.close_count == 0
    await goldfive.maybe_close_call_llm(call_llm, label="judge_call_llm")
    assert client.close_count == 1


async def test_public_close_helper_is_safe_for_optional_and_failed_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cleanup is optional and cannot turn shutdown into a new failure."""

    async def bare_call_llm(system: str, user: str, model: str) -> str:
        return ""

    class FailedClose:
        async def close(self) -> None:
            raise RuntimeError("cleanup failed")

    await goldfive.maybe_close_call_llm(None)
    await goldfive.maybe_close_call_llm(bare_call_llm)
    with caplog.at_level(logging.WARNING, logger="goldfive.llm"):
        await goldfive.maybe_close_call_llm(FailedClose(), label="dedicated_judge")

    assert "dedicated_judge.close() raised cleanup failed; ignored" in caplog.text
