"""Tests for :meth:`GoldfiveADKAgent._run_async_impl` streaming behaviour.

Validates that inner-Runner ADK Events are forwarded through to adk-web
in real time (the "plumb events through" fix), not collapsed into a
handful of synthetic summary Events emitted only after
:meth:`Runner.run` returns.

The test drives :meth:`GoldfiveADKAgent._run_async_impl` with a mock
:class:`InvocationContext`, substitutes a fake adapter ``invoke`` that
synthesizes inner-Runner Events via the adapter's event fan-out, and
asserts:

* At least one inner-tree ADK Event reaches adk-web (the real fix).
* The plan-summary header still fires BEFORE the first inner Event
  (goldfive-owned framing is preserved).
* The terminal turn-complete Event still fires last.
* The last yielded item is a proper ``turn_complete`` Event.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.adk")

import goldfive
from goldfive import InMemorySink, InvocationResult, SequentialExecutor, StaticPlanner
from goldfive.types import Plan, Task


def _one_task_planner() -> StaticPlanner:
    return StaticPlanner(
        Plan(
            id="p1",
            run_id="",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id="t1",
                    title="the task",
                    description="the task",
                    assignee_agent_id="inner_agent",
                )
            ],
            edges=[],
            summary="one task plan",
        )
    )


def _mk_inner(name: str = "inner_agent") -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description="a wrapped agent",
        instruction="follow instructions",
    )


class _FakeSession:
    def __init__(self) -> None:
        self.id = ""
        self.events: list[Any] = []


class _FakeCtx:
    def __init__(self, text: str) -> None:
        from google.genai.types import Content, Part  # type: ignore

        self.user_content = Content(role="user", parts=[Part(text=text)])
        self.session = _FakeSession()
        self.invocation_id = "invocation-test-passthrough"
        self.end_invocation = False


def _event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or ()
    return "\n".join(str(getattr(p, "text", "") or "") for p in parts).strip()


# ---------------------------------------------------------------------------
# Inner events plumb through to adk-web
# ---------------------------------------------------------------------------


async def test_run_async_impl_forwards_inner_adk_events(stub_call_llm: Any) -> None:
    """Inner-Runner events fanned out by the adapter reach the outer Event stream."""
    from google.adk.events import Event  # type: ignore
    from google.genai.types import Content, Part  # type: ignore

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    adapter = wrapped.runner.agent

    inner_event_1 = Event(
        invocation_id="inv-1",
        author=inner.name,
        content=Content(role="model", parts=[Part(text="inner step 1")]),
    )
    inner_event_2 = Event(
        invocation_id="inv-1",
        author=inner.name,
        content=Content(role="model", parts=[Part(text="inner step 2")]),
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        # Simulate the real invoke path fanning ADK events out to
        # subscribed listeners (that's what the adapter does in its
        # real ``async for event in self._runner.run_async(...)`` loop).
        adapter._dispatch_adk_event(inner_event_1)
        adapter._dispatch_adk_event(inner_event_2)
        return InvocationResult(task_id=task.id, text="ok: the task")

    adapter.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    ctx = _FakeCtx("make a thing")
    events = [evt async for evt in wrapped._run_async_impl(ctx)]

    # 1) At least one of the real inner events came through verbatim.
    forwarded_by_identity = [e for e in events if e is inner_event_1 or e is inner_event_2]
    assert forwarded_by_identity, (
        "inner-Runner ADK Events were not forwarded to the outer stream; "
        "adk-web UI would still be blank"
    )

    # 2) Plan summary still emitted.
    plan_summary_events = [
        e for e in events if _event_text(e).startswith("**one task plan**")
    ]
    assert plan_summary_events, "plan-summary framing event missing"

    # 3) Terminal turn_complete event still fires last.
    assert events[-1].turn_complete is True


async def test_run_async_impl_plan_summary_precedes_inner_events(
    stub_call_llm: Any,
) -> None:
    """The goldfive plan summary must land BEFORE any real inner event.

    adk-web renders events in order; the plan summary is the "header"
    that tells the UI which task block the upcoming tree activity
    belongs to. If inner events arrive first, the UI shows tree
    activity with no context.
    """
    from google.adk.events import Event  # type: ignore
    from google.genai.types import Content, Part  # type: ignore

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )
    adapter = wrapped.runner.agent

    inner_event = Event(
        invocation_id="inv-1",
        author=inner.name,
        content=Content(role="model", parts=[Part(text="inner event")]),
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        adapter._dispatch_adk_event(inner_event)
        return InvocationResult(task_id=task.id, text="ok")

    adapter.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    ctx = _FakeCtx("go")
    events = [evt async for evt in wrapped._run_async_impl(ctx)]

    # First plan-summary (by text) must be at a strictly earlier index
    # than the inner event (by identity).
    plan_idx = next(
        (i for i, e in enumerate(events) if _event_text(e).startswith("**one task plan**")),
        None,
    )
    inner_idx = next((i for i, e in enumerate(events) if e is inner_event), None)
    assert plan_idx is not None, "plan-summary event missing"
    assert inner_idx is not None, "inner event was not forwarded"
    assert plan_idx < inner_idx, (
        f"plan summary landed at index {plan_idx}, after inner event at {inner_idx}; "
        f"ordering contract broken"
    )


async def test_run_async_impl_still_emits_framing_when_no_inner_events(
    stub_call_llm: Any,
) -> None:
    """Runs with zero inner events must still deliver plan + task + terminal."""
    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok: the task")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    ctx = _FakeCtx("go")
    events = [evt async for evt in wrapped._run_async_impl(ctx)]

    # plan summary + at least one task-result block + terminal.
    assert len(events) >= 2
    assert any(_event_text(e).startswith("**one task plan**") for e in events)
    assert events[-1].turn_complete is True
