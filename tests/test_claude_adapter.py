"""Unit tests for :mod:`goldfive.adapters.claude`.

These tests are gated on ``claude_agent_sdk`` being importable. When the
optional dependency is missing the whole module is skipped via
``pytest.importorskip`` — this matches the optional-install contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

# Imports below are safe once the skip above has passed.
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from goldfive.adapters._claude_prompt import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)
from goldfive.adapters.claude import ClaudeAgentSDKAdapter  # noqa: E402
from goldfive.drift import classify_stop_reason  # noqa: E402
from goldfive.reporting import ReportingToolSpec  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftKind,
    Goal,
    Plan,
    Session,
    Task,
)

# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


class _RecordingSteerer:
    """Collects every event ``observe`` sees so tests can assert on them."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.transitions: list[tuple[str, str]] = []
        # The adapter routes observations through ``steerer.drift.observe``
        # (see ``_safe_observe``); expose this object as its own drift
        # observer so the recorded events actually accumulate.
        self.drift = self

    async def observe(self, event: Any, session: Session) -> None:
        self.events.append(event)

    async def transition(
        self,
        task_id: str,
        to: Any,
        *,
        detail: str = "",
        session: Session,
        cancel_reason: str = "",  # noqa: ARG002
    ) -> None:
        self.transitions.append((task_id, str(to)))

    def detect_drift(self, event: Any, session: Session) -> None:
        return None

    def bind(self, *, sinks: list[Any], planner: Any) -> None:
        pass


class _StubClient:
    """Minimal stand-in for :class:`claude_agent_sdk.ClaudeSDKClient`.

    The real client streams from a subprocess; we hard-wire a scripted
    message list so tests never touch the network.
    """

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages
        self.options: Any = None
        self.queries: list[str] = []
        self.connected = False

    async def connect(self, prompt: Any = None) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.queries.append(prompt)

    async def receive_response(self) -> AsyncIterator[Any]:
        for msg in self._messages:
            yield msg


def _make_result_message(stop_reason: str = "end_turn") -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=5,
        is_error=False,
        num_turns=1,
        session_id="test",
        stop_reason=stop_reason,
        result="final",
    )


def _make_assistant_message(blocks: list[Any]) -> AssistantMessage:
    return AssistantMessage(content=blocks, model="test-model", stop_reason="tool_use")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_prompt_render_default_template() -> None:
    """The default template fills every placeholder without KeyError."""

    task = Task(id="t1", title="Write report", description="Summarize findings")
    goals = [Goal(id="g1", summary="deliver quarterly report")]
    rendered = render_system_prompt(
        None,
        task=task,
        goals=goals,
        plan_summary="t1 -> t2",
        completed={"t0": "kickoff done"},
    )
    assert "t1" in rendered
    assert "Write report" in rendered
    assert "deliver quarterly report" in rendered
    assert "t0: kickoff done" in rendered
    assert "t1 -> t2" in rendered


def test_prompt_render_override_template() -> None:
    """A custom template is honored — placeholders are resolved."""

    template = (
        "GOALS:\n{goal_block}\nPLAN:{plan_summary}\nDONE:{completed_block}\nTASK:{task_block}\n"
    )
    rendered = render_system_prompt(
        template,
        task=Task(id="x", title="t"),
        goals=[],
        plan_summary="",
        completed={},
    )
    assert rendered.startswith("GOALS:\n(no goals declared)\n")
    assert "(no active plan)" in rendered
    assert "(none)" in rendered


def test_default_template_exports_required_placeholders() -> None:
    """Sanity check: the published default contains every placeholder."""

    for placeholder in ("{goal_block}", "{task_block}", "{plan_summary}", "{completed_block}"):
        assert placeholder in DEFAULT_SYSTEM_PROMPT_TEMPLATE


def test_classify_stop_reason_benign() -> None:
    """``tool_use`` and unknown reasons do not produce drift."""

    assert classify_stop_reason("tool_use") is None
    assert classify_stop_reason(None) is None
    assert classify_stop_reason("") is None
    assert classify_stop_reason("some_future_reason") is None


def test_classify_stop_reason_context_pressure() -> None:
    # ``classify_stop_reason`` is the shared context-pressure classifier
    # (its earlier per-kind behaviour moved elsewhere when the signature
    # was narrowed to ``(reason)``); the adapter stamps task/agent
    # attribution itself after calling it.
    drift = classify_stop_reason("max_tokens")
    assert drift is not None
    assert drift.kind is DriftKind.CONTEXT_PRESSURE


def test_classify_stop_reason_non_pressure_reasons_are_benign() -> None:
    # Reasons outside the context-pressure set produce no drift from
    # this classifier (refusal / step-count classification live in
    # their own detectors).
    assert classify_stop_reason("max_turns") is None
    assert classify_stop_reason("refusal") is None
    assert classify_stop_reason("end_turn") is None


def test_adapter_invoke_routes_tool_use_to_steerer() -> None:
    """A streamed tool_use for a reporting tool reaches the steerer and the handler."""

    handler_calls: list[dict[str, Any]] = []

    async def _report_completed(
        args: dict[str, Any],
        session: Session,
        steerer: Any,
    ) -> dict[str, Any]:
        handler_calls.append(args)
        return {"ok": True}

    spec = ReportingToolSpec(
        name="report_task_completed",
        description="Mark a task complete",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["task_id", "summary"],
        },
        handler=_report_completed,
    )

    # Scripted message stream: assistant says something, then emits a
    # tool_use for report_task_completed, then the result message.
    messages = [
        _make_assistant_message(
            [
                TextBlock(text="Working on it."),
                ToolUseBlock(
                    id="tu_1",
                    name="mcp__goldfive_reporting__report_task_completed",
                    input={"task_id": "t1", "summary": "done"},
                ),
            ]
        ),
        _make_result_message(stop_reason="end_turn"),
    ]
    stub_client = _StubClient(messages)

    steerer = _RecordingSteerer()
    adapter = ClaudeAgentSDKAdapter(
        client_factory=lambda: stub_client,
        steerer=steerer,
    )

    async def _run() -> None:
        await adapter.register_reporting_tools([spec])
        # Simulate the SDK-side hook firing for the tool_use block. The
        # adapter's invoke() streams messages straight from the stub,
        # but the hook is only invoked by the real SDK runtime — we call
        # it manually here to exercise the routing logic end-to-end.
        hook = adapter._make_pretooluse_hook(  # noqa: SLF001 - test hook
            Session(run_id="r1")
        )
        ack = await hook(
            {
                "tool_name": "mcp__goldfive_reporting__report_task_completed",
                "tool_input": {"task_id": "t1", "summary": "done"},
            },
            "tu_1",
            None,
        )
        # Hook should deny the no-op stub and embed the handler's ack.
        spec_out = ack.get("hookSpecificOutput", {})
        assert spec_out.get("permissionDecision") == "deny"
        assert "ok" in spec_out.get("permissionDecisionReason", "")

        session = Session(
            run_id="r1",
            goals=[Goal(id="g1", summary="complete t1")],
            plan=Plan(
                id="p1",
                run_id="r1",
                goal_ids=["g1"],
                tasks=[Task(id="t1", title="Do thing")],
                edges=[],
                summary="single-task plan",
            ),
        )
        result = await adapter.invoke(Task(id="t1", title="Do thing"), session)
        assert result.task_id == "t1"
        assert result.stop_reason == "end_turn"
        assert "Working on it." in result.text

    asyncio.run(_run())

    # The handler fired exactly once with the correct args.
    assert handler_calls == [{"task_id": "t1", "summary": "done"}]
    # The steerer observed messages from the stream: at minimum the
    # assistant message, the result message, and the tool-call observation.
    observed_types = [type(e).__name__ for e in steerer.events]
    assert "AssistantMessage" in observed_types
    assert "ResultMessage" in observed_types
    assert "_ToolCallObservation" in observed_types


def test_adapter_invoke_reports_drift_on_max_tokens() -> None:
    """A context-pressure stop_reason is classified, attributed to the
    invoked task/agent, and observed as drift."""

    messages = [_make_result_message(stop_reason="max_tokens")]
    stub_client = _StubClient(messages)
    steerer = _RecordingSteerer()
    adapter = ClaudeAgentSDKAdapter(
        client_factory=lambda: stub_client,
        steerer=steerer,
    )

    async def _run() -> None:
        session = Session(run_id="r2")
        result = await adapter.invoke(
            Task(id="t1", title="Do thing", assignee_agent_id="writer"), session
        )
        assert result.stop_reason == "max_tokens"

    asyncio.run(_run())

    drift_events = [e for e in steerer.events if type(e).__name__ == "DriftEvent"]
    assert len(drift_events) == 1
    assert drift_events[0].kind is DriftKind.CONTEXT_PRESSURE
    # The adapter stamps the attribution the classifier no longer takes.
    assert drift_events[0].current_task_id == "t1"
    assert drift_events[0].current_agent_id == "writer"


def test_adapter_invoke_benign_stop_reason_emits_no_drift() -> None:
    """``end_turn`` completes cleanly: no drift, no TypeError from the
    classifier call (regression: the old kwargs call raised on every
    completed invocation)."""

    messages = [_make_result_message(stop_reason="end_turn")]
    steerer = _RecordingSteerer()
    adapter = ClaudeAgentSDKAdapter(
        client_factory=lambda: _StubClient(messages),
        steerer=steerer,
    )

    async def _run() -> None:
        result = await adapter.invoke(
            Task(id="t1", title="Do thing"), Session(run_id="r3")
        )
        assert result.error is None
        assert result.stop_reason == "end_turn"

    asyncio.run(_run())
    assert not [e for e in steerer.events if type(e).__name__ == "DriftEvent"]


def test_adapter_available_agents_passthrough() -> None:
    adapter = ClaudeAgentSDKAdapter(
        client_factory=lambda: _StubClient([]),
        available_agents=["planner", "executor"],
    )
    assert adapter.available_agents == ["planner", "executor"]
    # Returned list is a copy — mutations don't leak.
    adapter.available_agents.append("other")
    assert adapter.available_agents == ["planner", "executor"]


def test_pretooluse_hook_passes_through_non_reporting_tools() -> None:
    """Non-reporting tools are not short-circuited."""

    async def _noop(
        args: dict[str, Any],
        session: Session,
        steerer: Any,
    ) -> dict[str, Any]:
        return {"ok": True}

    spec = ReportingToolSpec(
        name="report_task_started",
        description="start",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_noop,
    )

    adapter = ClaudeAgentSDKAdapter(client_factory=lambda: _StubClient([]))

    async def _run() -> dict[str, Any]:
        await adapter.register_reporting_tools([spec])
        hook = adapter._make_pretooluse_hook(Session(run_id="r"))  # noqa: SLF001
        return await hook(
            {"tool_name": "Bash", "tool_input": {"command": "ls"}},
            "tu_bash",
            None,
        )

    result = asyncio.run(_run())
    assert result == {}


# --------------------------------------------------------------------------- #
# Regression guard — every reporting-tool dispatch MUST route through
# :func:`goldfive.adapters._tool_invocation.invoke_tool` so the three
# protection layers (terminal-task rejection, idempotency, loop guard)
# fire. The Claude adapter's pre-tool-use hook previously invoked
# ``spec.handler(...)`` directly, silently bypassing all three. See
# ``docs/design/TASK-LIFECYCLE.md`` §5.
# --------------------------------------------------------------------------- #


def test_pretooluse_hook_on_terminal_task_returns_structured_rejection() -> None:
    """A cross-transition on an already-FAILED task must surface the
    structured ``invalid_transition`` error inside the hook's
    ``permissionDecisionReason`` — proof the hook routes through
    ``invoke_tool`` into the handler's idempotency matrix
    (goldfive#201).
    """
    import json as _json

    from goldfive.reporting import BUILTIN_REPORTING_TOOLS
    from goldfive.types import TaskStatus

    # Use the real built-in spec so the handler's idempotency matrix
    # runs. Pre-goldfive#201 this test used a boom-handler to prove
    # invoke_tool rejected BEFORE the handler; now the handler itself
    # owns the decision so we exercise the real one.
    spec = next(t for t in BUILTIN_REPORTING_TOOLS if t.name == "report_task_progress")

    adapter = ClaudeAgentSDKAdapter(client_factory=lambda: _StubClient([]))

    async def _run() -> dict[str, Any]:
        await adapter.register_reporting_tools([spec])
        task = Task(id="t1", title="x", status=TaskStatus.FAILED)
        session = Session(
            run_id="r",
            plan=Plan(
                id="p",
                run_id="r",
                goal_ids=[],
                tasks=[task],
                edges=[],
            ),
        )
        hook = adapter._make_pretooluse_hook(session)  # noqa: SLF001
        return await hook(
            {
                "tool_name": "mcp__goldfive_reporting__report_task_progress",
                "tool_input": {"task_id": "t1", "fraction": 0.3},
            },
            "tu_x",
            None,
        )

    result = asyncio.run(_run())
    spec_out = result.get("hookSpecificOutput", {})
    assert spec_out.get("permissionDecision") == "deny"
    reason = spec_out.get("permissionDecisionReason", "")
    # The hook encodes the ACK dict as JSON in the reason; parse it back
    # to inspect the structured rejection payload.
    parsed = _json.loads(reason)
    assert parsed.get("acknowledged") is False, (
        "expected structured rejection; got acknowledged=true. If this "
        "fails, _make_pretooluse_hook is bypassing invoke_tool."
    )
    assert parsed.get("error") == "invalid_transition"
    assert parsed.get("task_id") == "t1"
    assert parsed.get("current_status") == "FAILED"
