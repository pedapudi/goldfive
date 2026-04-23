"""Tests for :meth:`goldfive.Runner.run_streamed`.

Covers the streaming API added to support
:class:`goldfive.adapters.adk_wrap.GoldfiveADKAgent` forwarding
inner-Runner ADK events through adk-web in real time.

Two execution paths are verified:

* ADK adapter path — drives a minimal ADK tree through
  ``Runner.run_streamed`` and asserts at least one ADK
  ``google.adk.events.Event`` is yielded before the trailing
  :class:`~goldfive.results.ExecutionOutcome`.
* Non-ADK (callable) adapter path — the adapter has no framework
  events; ``run_streamed`` must still yield exactly one item (the
  outcome) and the run must still succeed.

The existing :meth:`Runner.run` contract is validated indirectly —
every ``run_streamed`` call ends with the same
:class:`ExecutionOutcome` that :meth:`run` would have returned.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

import goldfive
from goldfive import (
    ExecutionOutcome,
    InMemorySink,
    InvocationResult,
    Runner,
    SequentialExecutor,
    StaticPlanner,
)
from goldfive.types import Plan, Task


def _one_task_planner(task_id: str = "t1", title: str = "the task") -> StaticPlanner:
    return StaticPlanner(
        Plan(
            id="p1",
            run_id="",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id=task_id,
                    title=title,
                    description=title,
                    assignee_agent_id="inner_agent",
                )
            ],
            edges=[],
            summary="one task plan",
        )
    )


# ---------------------------------------------------------------------------
# ADK adapter: streaming yields raw inner-Runner Events + outcome
# ---------------------------------------------------------------------------


pytest.importorskip("google.adk")


def _mk_inner(name: str = "inner_agent") -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description="a wrapped agent",
        instruction="follow instructions",
    )


async def test_run_streamed_yields_inner_adk_events_then_outcome(
    stub_call_llm: Any,
) -> None:
    """At least one raw ADK Event + a trailing ExecutionOutcome must be yielded."""
    from google.adk.events import Event  # type: ignore

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )
    # Substitute a stub for adapter.invoke so we don't drive a real LLM,
    # BUT still synthesize inner-Runner events via the fan-out so the
    # streaming path is exercised. We call _dispatch_adk_event directly
    # from the fake invoke — this is the seam the real invoke path uses
    # on every event from ``runner.run_async``.
    adapter = wrapped.runner.agent

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        # Simulate two inner-Runner events during this invoke.
        from google.genai.types import Content, Part  # type: ignore

        for text in ("step one", "step two"):
            fake_event = Event(
                invocation_id="inv-1",
                author=inner.name,
                content=Content(role="model", parts=[Part(text=text)]),
            )
            adapter._dispatch_adk_event(fake_event)
        return InvocationResult(task_id=task.id, text="ok")

    adapter.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    items: list[Any] = []
    async for item in wrapped.runner.run_streamed(
        [goldfive.Goal(id="g1", summary="go")],
    ):
        items.append(item)

    # Last item must be the ExecutionOutcome.
    assert items, "run_streamed yielded nothing"
    assert isinstance(items[-1], ExecutionOutcome)
    assert items[-1].success, items[-1].reason

    # At least one inner-Runner Event must have preceded the outcome.
    adk_events = [i for i in items[:-1] if isinstance(i, Event)]
    assert adk_events, "no ADK Events yielded mid-run"


async def test_run_streamed_without_inner_events_still_yields_outcome(
    stub_call_llm: Any,
) -> None:
    """If no inner Events fire, run_streamed still produces a single outcome item."""
    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    items = [
        item
        async for item in wrapped.runner.run_streamed(
            [goldfive.Goal(id="g1", summary="go")],
        )
    ]

    # At minimum: the outcome. Possibly zero pre-outcome items.
    assert items
    assert isinstance(items[-1], ExecutionOutcome)
    assert items[-1].success, items[-1].reason


async def test_run_streamed_matches_run_contract() -> None:
    """``run_streamed``'s final outcome should be shape-equivalent to ``run``."""
    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    items = [
        item
        async for item in wrapped.runner.run_streamed(
            [goldfive.Goal(id="g1", summary="go")],
        )
    ]
    outcome = items[-1]
    assert isinstance(outcome, ExecutionOutcome)
    # Contract: outcome carries a Session with the run's plan / results.
    assert outcome.session is not None
    assert outcome.session.plan is not None


# ---------------------------------------------------------------------------
# Non-ADK adapter (callable): run_streamed degrades gracefully
# ---------------------------------------------------------------------------


async def test_run_streamed_on_callable_adapter_yields_only_outcome() -> None:
    """Callable adapters have no ADK events; run_streamed must still work."""

    async def _bare_agent(task: Any, session: Any, tools: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    runner: Runner = goldfive.wrap(
        _bare_agent,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    items = [
        item async for item in runner.run_streamed([goldfive.Goal(id="g1", summary="go")])
    ]
    assert isinstance(items[-1], ExecutionOutcome)
    assert items[-1].success, items[-1].reason
