"""approval_gated_agent — human-in-the-loop APPROVE / REJECT demo.

Exercises both flows from ``docs/design/APPROVAL.md`` against a live
session:

* **Flow A (task-level)**: a callable agent calls
  ``report_awaiting_approval`` for the current task. The driver below
  watches the ``pending_approvals`` map and pushes an APPROVE control
  message via :func:`dispatch_control` once the waiter appears.
* **Flow B (tool-level)**: the ADKAdapter's plugin hook
  (``_await_tool_approval``) is exercised directly against a
  ``Session``. We skip spinning a real ADK agent turn so the demo
  runs without a model binding — the point is to show how the control
  channel resolves the gate.

Run with::

    uv run python examples/approval_gated_agent.py

Expected output: two sections, one per flow, each showing the
ApprovalRequested → ApprovalGranted / ApprovalRejected trace.
"""

from __future__ import annotations

import asyncio
from typing import Any

from goldfive.adapters._adk_plugin import SessionContext, _await_tool_approval
from goldfive.control import ControlKind, ControlMessage
from goldfive.executors._control import dispatch_control
from goldfive.reporting import BUILTIN_REPORTING_TOOLS
from goldfive.steerer import DefaultSteerer
from goldfive.types import Goal, Plan, Session, Task


class PrintingSink:
    """Sink that prints each event's payload kind as it arrives."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)
        kind = (
            event_pb.WhichOneof("payload")
            if hasattr(event_pb, "WhichOneof")
            else "?"
        )
        detail = ""
        if kind == "approval_requested":
            req = event_pb.approval_requested
            detail = f" target={req.target_id!r} prompt={req.prompt!r}"
        elif kind in ("approval_granted", "approval_rejected"):
            field = getattr(event_pb, kind)
            detail = f" target={field.target_id!r} detail={field.detail!r}"
        print(f"  [{self.label}] {kind}{detail}")

    async def close(self) -> None:
        pass


def _find_handler(name: str) -> Any:
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec.handler
    raise AssertionError(f"no tool {name!r}")


def _session(task_id: str) -> Session:
    plan = Plan(
        id="p1",
        run_id="demo",
        goal_ids=["g1"],
        tasks=[Task(id=task_id, title="Charge customer card")],
        edges=[],
    )
    return Session(
        run_id="demo",
        goals=[Goal(id="g1", summary="process payment")],
        plan=plan,
        current_task_id=task_id,
    )


async def flow_a_task_level() -> None:
    print("== Flow A: task-level approval ==")
    session = _session("charge")
    sink = PrintingSink("A")
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)

    handler = _find_handler("report_awaiting_approval")

    async def agent_blocks_for_approval() -> dict[str, Any]:
        # Pretend the agent is asking for sign-off before charging.
        print("  agent calls report_awaiting_approval(prompt='charge $500?')")
        return await handler(
            {"task_id": "charge", "prompt": "charge $500?"}, session, steerer
        )

    agent_task = asyncio.create_task(agent_blocks_for_approval())
    # Wait for the waiter to land on the session.
    while "charge" not in session.pending_approvals:
        await asyncio.sleep(0.01)

    print("  UI: human clicks APPROVE")
    await dispatch_control(
        ControlMessage(
            kind=ControlKind.APPROVE,
            payload={"target_id": "charge", "detail": "cfo signed off"},
        ),
        session=session,
        steerer=steerer,
        sinks=[sink],
    )

    result = await agent_task
    print(f"  agent received: decision={result['decision']!r} detail={result['detail']!r}")
    print()


async def flow_b_tool_level() -> None:
    print("== Flow B: ADK tool-level approval (require_confirmation=True) ==")
    session = _session("write")
    sink = PrintingSink("B")

    class _StubSteerer:
        def __init__(self, sinks: list[Any]) -> None:
            self._sinks = list(sinks)

    class _FakeTool:
        name = "write_file"
        _require_confirmation = True
        approval_prompt = "Write /etc/passwd?"

    class _FakeCtx:
        function_call_id = "adk-demo-42"

    session_ctx = SessionContext(
        session=session,
        steerer=_StubSteerer([sink]),
        task=session.plan.tasks[0],
        tool_handlers={},
        host_agent_name="demo_agent",
    )

    print("  ADK: agent wants to call write_file(path='/etc/passwd')")
    tool_task = asyncio.create_task(
        _await_tool_approval(
            tool=_FakeTool(),
            tool_name="write_file",
            tool_args={"path": "/etc/passwd"},
            tool_context=_FakeCtx(),
            session_ctx=session_ctx,
        )
    )
    while "adk-demo-42" not in session.pending_approvals:
        await asyncio.sleep(0.01)

    print("  UI: human clicks REJECT")
    await dispatch_control(
        ControlMessage(
            kind=ControlKind.REJECT,
            payload={"target_id": "adk-demo-42", "detail": "never"},
        ),
        session=session,
        steerer=session_ctx.steerer,
        sinks=[sink],
    )

    result = await tool_task
    print(f"  ADK: tool skipped — result = {result}")


async def main() -> None:
    await flow_a_task_level()
    await flow_b_tool_level()


if __name__ == "__main__":
    asyncio.run(main())
