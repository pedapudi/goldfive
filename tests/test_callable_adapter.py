"""Unit tests for :class:`goldfive.adapters.CallableAdapter`.

These tests use minimal stand-in ``report_task_started`` and
``report_task_completed`` tool handlers. Issue #5/#6 owns the real handler
implementations — what we care about here is that the adapter:

1. Stores tool specs on ``register_reporting_tools``.
2. Forwards them to the wrapped callable on ``invoke``.
3. Returns the callable's :class:`InvocationResult` verbatim.
4. Exposes ``available_agents`` as configured.

We also exercise the ``_tool_invocation`` helper via the stub agent, since
it is the shared lookup path every adapter will use.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.adapters import CallableAdapter
from goldfive.adapters._tool_invocation import find_tool, invoke_tool
from goldfive.reporting import ReportingToolSpec
from goldfive.results import InvocationResult
from goldfive.types import Session, Task, TaskStatus


class _StubSteerer:
    """Minimal ``Steerer``-shaped recorder used for tool handler tests.

    Real handlers (issue #6) will route through the live steerer, but here
    we only need something the stub handlers can call ``transition`` on and
    that the adapter never inspects.
    """

    def __init__(self) -> None:
        self.transitions: list[tuple[str, TaskStatus, str]] = []

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
        cancel_reason: str = "",  # noqa: ARG002
    ) -> None:
        self.transitions.append((task_id, to, detail))
        # Mirror real steerer behaviour: update session state.
        session.current_task_id = task_id
        if session.plan is not None:
            for t in session.plan.tasks:
                if t.id == task_id:
                    t.status = to

    # Unused in these tests but kept so the stub is Steerer-shaped.
    async def observe(self, event: Any, session: Session) -> None:  # pragma: no cover
        pass

    def detect_drift(self, event: Any, session: Session):  # pragma: no cover
        return None

    def bind(self, *, sinks, planner) -> None:  # pragma: no cover
        pass


def _make_started_tool(steerer: _StubSteerer) -> ReportingToolSpec:
    async def handler(args: dict[str, Any], session: Session, st) -> dict[str, Any]:
        await st.transition(
            args["task_id"],
            TaskStatus.RUNNING,
            detail=args.get("detail", ""),
            session=session,
        )
        return {"ok": True}

    return ReportingToolSpec(
        name="report_task_started",
        description="Mark a task as started.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "detail": {"type": "string"},
            },
            "required": ["task_id"],
        },
        handler=handler,
    )


def _make_completed_tool(steerer: _StubSteerer) -> ReportingToolSpec:
    async def handler(args: dict[str, Any], session: Session, st) -> dict[str, Any]:
        await st.transition(
            args["task_id"],
            TaskStatus.COMPLETED,
            detail=args.get("summary", ""),
            session=session,
        )
        session.completed_results[args["task_id"]] = args.get("summary", "")
        return {"ok": True}

    return ReportingToolSpec(
        name="report_task_completed",
        description="Mark a task as completed.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "summary": {"type": "string"},
                "artifacts": {"type": "object"},
            },
            "required": ["task_id", "summary"],
        },
        handler=handler,
    )


def _fresh_session() -> tuple[Session, Task]:
    from goldfive.types import Plan

    task = Task(id="t1", title="Do the thing")
    session = Session(
        run_id="run-1",
        plan=Plan(
            id="p1",
            run_id="run-1",
            goal_ids=["g1"],
            tasks=[task],
            edges=[],
        ),
    )
    return session, task


async def test_register_reporting_tools_stores_specs() -> None:
    async def noop_agent(task, session, tools):
        return InvocationResult(task_id=task.id, text="ok")

    adapter = CallableAdapter(noop_agent)
    started = _make_started_tool(_StubSteerer())
    completed = _make_completed_tool(_StubSteerer())

    await adapter.register_reporting_tools([started, completed])

    # Private attr intentionally inspected — adapter internals are small
    # and this confirms forwarding is wired correctly.
    assert adapter._tools == [started, completed]


async def test_available_agents_defaults_to_empty_list() -> None:
    async def noop_agent(task, session, tools):
        return InvocationResult(task_id=task.id)

    adapter = CallableAdapter(noop_agent)
    assert adapter.available_agents == []


async def test_available_agents_returns_configured_list() -> None:
    async def noop_agent(task, session, tools):
        return InvocationResult(task_id=task.id)

    adapter = CallableAdapter(noop_agent, available_agents=["alpha", "beta"])
    assert adapter.available_agents == ["alpha", "beta"]

    # Returned list is a copy — caller mutations must not leak into adapter state.
    adapter.available_agents.append("gamma")
    assert adapter.available_agents == ["alpha", "beta"]


async def test_invoke_forwards_tools_and_returns_result() -> None:
    captured: dict[str, Any] = {}

    async def agent(task, session, tools):
        captured["task"] = task
        captured["session"] = session
        captured["tools"] = tools
        return InvocationResult(task_id=task.id, text="hello", stop_reason="end_turn")

    tool = _make_started_tool(_StubSteerer())
    adapter = CallableAdapter(agent)
    await adapter.register_reporting_tools([tool])

    session, task = _fresh_session()
    result = await adapter.invoke(task, session)

    assert result.task_id == "t1"
    assert result.text == "hello"
    assert result.stop_reason == "end_turn"
    assert captured["task"] is task
    assert captured["session"] is session
    assert captured["tools"] == [tool]


async def test_tool_routing_drives_session_transitions() -> None:
    """The canonical acceptance test for issue #12.

    A stub agent calls ``report_task_started`` then ``report_task_completed``
    via the ``_tool_invocation`` helper. We verify the task moves PENDING ->
    RUNNING -> COMPLETED and that the completion summary lands in
    ``session.completed_results``.
    """
    steerer = _StubSteerer()
    started = _make_started_tool(steerer)
    completed = _make_completed_tool(steerer)

    async def agent(task, session, tools):
        await invoke_tool(
            tools,
            "report_task_started",
            {"task_id": task.id, "detail": "kicking off"},
            session,
            steerer,
        )
        await invoke_tool(
            tools,
            "report_task_completed",
            {"task_id": task.id, "summary": "done deal"},
            session,
            steerer,
        )
        return InvocationResult(task_id=task.id, text="done deal", stop_reason="end_turn")

    adapter = CallableAdapter(agent)
    await adapter.register_reporting_tools([started, completed])

    session, task = _fresh_session()
    assert task.status == TaskStatus.PENDING

    result = await adapter.invoke(task, session)

    assert result.task_id == "t1"
    assert result.text == "done deal"
    assert task.status == TaskStatus.COMPLETED
    assert session.completed_results == {"t1": "done deal"}
    assert steerer.transitions == [
        ("t1", TaskStatus.RUNNING, "kicking off"),
        ("t1", TaskStatus.COMPLETED, "done deal"),
    ]


async def test_invoke_tool_raises_on_unknown_name() -> None:
    session, _ = _fresh_session()
    with pytest.raises(KeyError, match="unknown reporting tool"):
        await invoke_tool([], "report_task_started", {}, session, _StubSteerer())


def test_find_tool_returns_none_for_missing_name() -> None:
    steerer = _StubSteerer()
    tools = [_make_started_tool(steerer)]
    assert find_tool(tools, "report_task_started") is tools[0]
    assert find_tool(tools, "nope") is None


async def test_register_reporting_tools_replaces_previous_tools() -> None:
    """Re-registering drops the old spec list — executors re-bind per run."""

    async def agent(task, session, tools):
        return InvocationResult(task_id=task.id)

    adapter = CallableAdapter(agent)
    first = _make_started_tool(_StubSteerer())
    await adapter.register_reporting_tools([first])
    second = _make_completed_tool(_StubSteerer())
    await adapter.register_reporting_tools([second])
    assert adapter._tools == [second]


async def test_adapter_conforms_to_agent_adapter_protocol() -> None:
    from goldfive.protocols import AgentAdapter

    async def agent(task, session, tools):
        return InvocationResult(task_id=task.id)

    adapter = CallableAdapter(agent)
    assert isinstance(adapter, AgentAdapter)
