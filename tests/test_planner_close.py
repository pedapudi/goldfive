"""Runner.close awaits optional close() on planner / goal_deriver call_llm.

A common SDK pattern (OpenAI ``AsyncClient``, ADK ``LiteLlm`` over
litellm) hands back a callable backed by an ``aiohttp.ClientSession``
that leaks at process exit unless explicitly closed. We extend the
``call_llm`` shape with an optional async ``close`` and have the
Runner walk the planner / goal_deriver to fire it.
"""

from __future__ import annotations

from typing import Any

from goldfive.adapters.callable import CallableAdapter
from goldfive.executors.sequential import SequentialExecutor
from goldfive.goal_deriver import LLMGoalDeriver
from goldfive.planner import LLMPlanner
from goldfive.results import InvocationResult
from goldfive.runner import Runner
from goldfive.types import Task


class _ClosableLLM:
    """A minimal call_llm with a ``close`` shim and a record-keeping flag."""

    def __init__(self, response: str = "{}") -> None:
        self._response = response
        self.calls = 0
        self.closed = False

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls += 1
        return self._response

    async def close(self) -> None:
        self.closed = True


class _BareLLM:
    """A call_llm with NO close attribute — Runner must tolerate this."""

    async def __call__(self, system: str, user: str, model: str) -> str:
        return "{}"


async def _stub_agent(task: Task, session: Any, tools: Any) -> InvocationResult:
    return InvocationResult(task_id=task.id, text="ok", stop_reason="end_turn")


def _runner_with(planner_llm: Any, goal_llm: Any) -> Runner:
    adapter = CallableAdapter(_stub_agent, available_agents=["root"])
    return Runner(
        agent=adapter,
        planner=LLMPlanner(call_llm=planner_llm),
        executor=SequentialExecutor(),
        goal_deriver=LLMGoalDeriver(goal_llm),
    )


async def test_runner_close_awaits_planner_call_llm_close() -> None:
    planner_llm = _ClosableLLM()
    goal_llm = _BareLLM()
    runner = _runner_with(planner_llm, goal_llm)

    await runner.close()

    assert planner_llm.closed is True


async def test_runner_close_awaits_goal_deriver_call_llm_close() -> None:
    planner_llm = _BareLLM()
    goal_llm = _ClosableLLM()
    runner = _runner_with(planner_llm, goal_llm)

    await runner.close()

    assert goal_llm.closed is True


async def test_runner_close_tolerates_bare_callable() -> None:
    """Lambda-style call_llm without a close attribute: no error."""
    runner = _runner_with(_BareLLM(), _BareLLM())
    await runner.close()  # must not raise


async def test_runner_close_swallows_exception_from_close() -> None:
    """A raising close() must not propagate or break subsequent hooks."""

    class _BoomLLM(_ClosableLLM):
        async def close(self) -> None:
            raise RuntimeError("boom")

    boom = _BoomLLM()
    fired: list[str] = []

    async def hook() -> None:
        fired.append("hook")

    runner = _runner_with(boom, _BareLLM())
    runner.add_close_hook(hook)
    await runner.close()  # must not raise

    # The hook still fired despite the failing close().
    assert fired == ["hook"]


async def test_runner_close_is_idempotent() -> None:
    planner_llm = _ClosableLLM()
    runner = _runner_with(planner_llm, _BareLLM())

    await runner.close()
    await runner.close()  # second call is a no-op (Runner._closed guard)

    # Closer was called exactly once because the second close() short-circuits.
    assert planner_llm.closed is True
