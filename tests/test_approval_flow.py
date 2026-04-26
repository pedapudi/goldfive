"""Tests for the human-in-the-loop approval flows (issue #80).

Two flows share the ``APPROVE`` / ``REJECT`` control kinds and the
``ApprovalRequested`` / ``ApprovalGranted`` / ``ApprovalRejected`` event
triple:

* **Flow A (task-level)**: an agent calls ``report_awaiting_approval``;
  the task blocks; the control dispatcher resumes it on APPROVE /
  REJECT. Handler returns the decision back to the agent.
* **Flow B (tool-level)**: an ADK tool flagged
  ``require_confirmation=True`` is intercepted in
  ``_GoldfiveADKPlugin.before_tool_callback``; the plugin awaits the
  same control signal and either falls through (APPROVE) or returns a
  skipped-tool dict (REJECT).

Both scenarios are exercised directly — without spinning up a Runner —
by invoking the handlers against a ``Session`` and pushing
``ControlMessage`` values through ``dispatch_control``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SessionContext,
    _await_tool_approval,
    _tool_requires_confirmation,
)
from goldfive.control import (  # noqa: E402
    ControlKind,
    ControlMessage,
)
from goldfive.executors._control import dispatch_control  # noqa: E402
from goldfive.reporting import (  # noqa: E402
    BUILTIN_REPORTING_TOOLS,
    REPORTING_TOOL_NAMES,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
)


class ListSink:
    """Minimal EventSink recording every envelope it sees."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass

    def payload_kinds(self) -> list[str]:
        return [e.WhichOneof("payload") for e in self.events if hasattr(e, "WhichOneof")]


class _StubSteerer:
    """Minimal steerer for ADK-plugin tests — only needs ``_sinks``."""

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = list(sinks)

    def bind(self, *, sinks: list[Any], planner: Any) -> None:
        self._sinks = list(sinks)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _plan_with_task(task_id: str = "t1") -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=task_id, title=task_id)],
        edges=[],
    )


def _session_with_plan(task_id: str = "t1") -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=_plan_with_task(task_id),
        current_task_id=task_id,
    )


def _find_handler(name: str) -> Any:
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec.handler
    raise AssertionError(f"no reporting tool named {name!r}")


# ---------------------------------------------------------------------------
# Basic registration / vocabulary
# ---------------------------------------------------------------------------


def test_eighth_reporting_tool_is_report_awaiting_approval() -> None:
    # report_awaiting_approval is the 8th canonical reporting tool. The
    # ninth and tenth are ``declare_task_skipped`` and
    # ``declare_task_not_needed`` (goldfive#271 Phase 3 — observability-
    # only structural declarations) — the awaiting_approval position
    # in the tuple is preserved.
    assert REPORTING_TOOL_NAMES[7] == "report_awaiting_approval"
    assert len(REPORTING_TOOL_NAMES) == 10
    names = {spec.name for spec in BUILTIN_REPORTING_TOOLS}
    assert "report_awaiting_approval" in names


def test_control_kind_has_approve_reject() -> None:
    assert ControlKind.APPROVE.value == "APPROVE"
    assert ControlKind.REJECT.value == "REJECT"


def test_session_has_pending_approvals_maps() -> None:
    session = Session(run_id="r")
    assert session.pending_approvals == {}
    assert session.pending_approvals_meta == {}


# ---------------------------------------------------------------------------
# Flow A — task-level approval
# ---------------------------------------------------------------------------


async def test_task_level_approve_resumes_handler() -> None:
    """APPROVE control unblocks `report_awaiting_approval` and returns 'approve'."""
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)

    handler = _find_handler("report_awaiting_approval")

    async def run_handler() -> dict[str, Any]:
        return await handler(
            {"task_id": "t1", "prompt": "ok to spend $500?"},
            session,
            steerer,
        )

    task = asyncio.create_task(run_handler())
    # Let the handler block on the waiter.
    for _ in range(20):
        if "t1" in session.pending_approvals:
            break
        await asyncio.sleep(0.01)
    assert "t1" in session.pending_approvals
    assert session.plan.tasks[0].status == TaskStatus.BLOCKED

    # Dispatch APPROVE control.
    msg = ControlMessage(
        kind=ControlKind.APPROVE,
        payload={"target_id": "t1", "detail": "looks fine"},
    )
    outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=[sink])
    assert outcome.ack.result.value == "SUCCESS"

    result = await asyncio.wait_for(task, timeout=1.0)
    assert result["decision"] == "approve"
    assert result["detail"] == "looks fine"

    kinds = sink.payload_kinds()
    # TaskBlocked from mark_task_blocked, DriftDetected (BLOCKED), then
    # ApprovalRequested, then ApprovalGranted.
    assert "task_blocked" in kinds
    assert "approval_requested" in kinds
    assert "approval_granted" in kinds
    assert "approval_rejected" not in kinds


async def test_task_level_reject_resumes_handler() -> None:
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)

    handler = _find_handler("report_awaiting_approval")

    task = asyncio.create_task(
        handler(
            {"task_id": "t1", "prompt": "delete user data?"},
            session,
            steerer,
        )
    )
    for _ in range(20):
        if "t1" in session.pending_approvals:
            break
        await asyncio.sleep(0.01)

    msg = ControlMessage(
        kind=ControlKind.REJECT,
        payload={"target_id": "t1", "detail": "absolutely not"},
    )
    outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=[sink])
    assert outcome.ack.result.value == "SUCCESS"

    result = await asyncio.wait_for(task, timeout=1.0)
    assert result["decision"] == "reject"
    assert result["detail"] == "absolutely not"
    assert "approval_rejected" in sink.payload_kinds()


async def test_task_level_approval_timeout_returns_timeout_decision() -> None:
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)

    handler = _find_handler("report_awaiting_approval")

    result = await handler(
        {"task_id": "t1", "prompt": "?", "timeout_ms": 50},
        session,
        steerer,
    )
    assert result["decision"] == "timeout"
    # Task remains blocked; next APPROVE / REJECT still resolves via the map.
    assert "t1" in session.pending_approvals
    assert session.plan.tasks[0].status == TaskStatus.BLOCKED


async def test_dispatch_approve_unknown_target_fails_ack() -> None:
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)

    msg = ControlMessage(
        kind=ControlKind.APPROVE,
        payload={"target_id": "nope", "detail": ""},
    )
    outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=[sink])
    assert outcome.ack.result.value == "FAILURE"
    assert "no pending approval" in outcome.ack.detail


async def test_dispatch_approve_missing_target_fails_ack() -> None:
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)

    msg = ControlMessage(kind=ControlKind.APPROVE, payload={})
    outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=[sink])
    assert outcome.ack.result.value == "FAILURE"
    assert "requires payload.target_id" in outcome.ack.detail


async def test_awaiting_approval_requires_task_id() -> None:
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)
    handler = _find_handler("report_awaiting_approval")

    result = await handler({"prompt": "x"}, session, steerer)
    assert result.get("error")


# ---------------------------------------------------------------------------
# Flow B — ADK tool-level approval
# ---------------------------------------------------------------------------


class _FakeTool:
    """Minimal stand-in for an ADK ``FunctionTool`` with require_confirmation."""

    def __init__(
        self,
        *,
        name: str = "write_file",
        require_confirmation: Any = True,
        approval_prompt: str = "",
    ) -> None:
        self.name = name
        # FunctionTool uses the private attribute name.
        self._require_confirmation = require_confirmation
        if approval_prompt:
            self.approval_prompt = approval_prompt


class _FakeToolContext:
    """Minimal ADK-style tool_context that exposes function_call_id."""

    def __init__(self, function_call_id: str) -> None:
        self.function_call_id = function_call_id


def test_tool_requires_confirmation_bool() -> None:
    assert _tool_requires_confirmation(_FakeTool(require_confirmation=True), {})
    assert not _tool_requires_confirmation(_FakeTool(require_confirmation=False), {})


def test_tool_requires_confirmation_callable() -> None:
    tool = _FakeTool(require_confirmation=lambda *, path="": path.startswith("/etc"))
    assert _tool_requires_confirmation(tool, {"path": "/etc/passwd"})
    assert not _tool_requires_confirmation(tool, {"path": "/tmp/x"})


async def test_tool_level_approve_falls_through() -> None:
    """On APPROVE, the plugin hook returns None so ADK runs the tool body."""
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = _StubSteerer([sink])
    tool = _FakeTool(name="write_file")
    ctx = _FakeToolContext(function_call_id="adk-abc123")
    session_ctx = SessionContext(
        session=session,
        steerer=steerer,
        task=session.plan.tasks[0],
        tool_handlers={},
        host_agent_name="root",
    )

    task = asyncio.create_task(
        _await_tool_approval(
            tool=tool,
            tool_name="write_file",
            tool_args={"path": "/etc/passwd"},
            tool_context=ctx,
            session_ctx=session_ctx,
        )
    )
    for _ in range(20):
        if "adk-abc123" in session.pending_approvals:
            break
        await asyncio.sleep(0.01)
    assert "adk-abc123" in session.pending_approvals
    assert session.pending_approvals_meta["adk-abc123"]["tool_name"] == "write_file"

    msg = ControlMessage(
        kind=ControlKind.APPROVE,
        payload={"target_id": "adk-abc123", "detail": "proceed"},
    )
    outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=[sink])
    assert outcome.ack.result.value == "SUCCESS"

    result = await asyncio.wait_for(task, timeout=1.0)
    assert result is None  # ADK will run the tool
    kinds = sink.payload_kinds()
    assert "approval_requested" in kinds
    assert "approval_granted" in kinds


async def test_tool_level_reject_skips_tool() -> None:
    """On REJECT, the plugin hook returns a skipped-tool dict."""
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = _StubSteerer([sink])
    tool = _FakeTool(name="send_email", approval_prompt="Send email to legal?")
    ctx = _FakeToolContext(function_call_id="adk-xyz789")
    session_ctx = SessionContext(
        session=session,
        steerer=steerer,
        task=session.plan.tasks[0],
        tool_handlers={},
        host_agent_name="root",
    )

    task = asyncio.create_task(
        _await_tool_approval(
            tool=tool,
            tool_name="send_email",
            tool_args={"to": "legal@example.com"},
            tool_context=ctx,
            session_ctx=session_ctx,
        )
    )
    for _ in range(20):
        if "adk-xyz789" in session.pending_approvals:
            break
        await asyncio.sleep(0.01)

    msg = ControlMessage(
        kind=ControlKind.REJECT,
        payload={"target_id": "adk-xyz789", "detail": "not now"},
    )
    await dispatch_control(msg, session=session, steerer=steerer, sinks=[sink])

    result = await asyncio.wait_for(task, timeout=1.0)
    assert isinstance(result, dict)
    assert result["skipped"] is True
    assert result["reason"] == "user_rejected"
    assert result["tool_name"] == "send_email"

    # Verify the emitted ApprovalRequested carries the custom prompt.
    req = [e for e in sink.events if e.WhichOneof("payload") == "approval_requested"]
    assert len(req) == 1
    assert req[0].approval_requested.prompt == "Send email to legal?"
    assert req[0].approval_requested.kind == "tool"
    assert req[0].approval_requested.metadata["tool_name"] == "send_email"


async def test_tool_level_uses_generated_target_id_when_no_function_call_id() -> None:
    """Plugin falls back to a fresh adk-<uuid> when ctx lacks function_call_id."""
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = _StubSteerer([sink])
    tool = _FakeTool(name="risky_op")

    class _BareCtx:
        pass

    session_ctx = SessionContext(
        session=session,
        steerer=steerer,
        task=session.plan.tasks[0],
        tool_handlers={},
        host_agent_name="root",
    )

    task = asyncio.create_task(
        _await_tool_approval(
            tool=tool,
            tool_name="risky_op",
            tool_args={},
            tool_context=_BareCtx(),
            session_ctx=session_ctx,
        )
    )
    for _ in range(20):
        if session.pending_approvals:
            break
        await asyncio.sleep(0.01)
    [generated] = list(session.pending_approvals.keys())
    assert generated.startswith("adk-")

    msg = ControlMessage(
        kind=ControlKind.APPROVE,
        payload={"target_id": generated, "detail": ""},
    )
    await dispatch_control(msg, session=session, steerer=steerer, sinks=[sink])
    await asyncio.wait_for(task, timeout=1.0)


# ---------------------------------------------------------------------------
# Cross-flow: both waiters on the same session resolve independently
# ---------------------------------------------------------------------------


async def test_task_and_tool_waiters_resolve_independently() -> None:
    session = _session_with_plan("t1")
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)

    # Register a task-level waiter via the reporting handler.
    task_handler = _find_handler("report_awaiting_approval")
    task_fut = asyncio.create_task(
        task_handler({"task_id": "t1", "prompt": "task-level?"}, session, steerer)
    )
    for _ in range(20):
        if "t1" in session.pending_approvals:
            break
        await asyncio.sleep(0.01)

    # Register a tool-level waiter via the plugin helper.
    session_ctx = SessionContext(
        session=session,
        steerer=steerer,
        task=session.plan.tasks[0],
        tool_handlers={},
        host_agent_name="root",
    )
    tool_fut = asyncio.create_task(
        _await_tool_approval(
            tool=_FakeTool(name="tx"),
            tool_name="tx",
            tool_args={"amount": 10},
            tool_context=_FakeToolContext(function_call_id="adk-tool"),
            session_ctx=session_ctx,
        )
    )
    for _ in range(20):
        if "adk-tool" in session.pending_approvals:
            break
        await asyncio.sleep(0.01)

    # Reject the tool, approve the task.
    await dispatch_control(
        ControlMessage(kind=ControlKind.REJECT, payload={"target_id": "adk-tool"}),
        session=session,
        steerer=steerer,
        sinks=[sink],
    )
    await dispatch_control(
        ControlMessage(kind=ControlKind.APPROVE, payload={"target_id": "t1"}),
        session=session,
        steerer=steerer,
        sinks=[sink],
    )

    task_result = await asyncio.wait_for(task_fut, timeout=1.0)
    tool_result = await asyncio.wait_for(tool_fut, timeout=1.0)
    assert task_result["decision"] == "approve"
    assert tool_result["skipped"] is True
