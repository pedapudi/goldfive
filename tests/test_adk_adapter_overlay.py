"""Tests for the goldfive#141 overlay path on :class:`ADKAdapter`.

Covers:

* :meth:`ADKAdapter.invoke_passthrough` sends the user's message
  verbatim (no ``"Task: X"`` framing, no goldfive jargon).
* :meth:`ADKAdapter.invoke_follow_up` uses the gentle
  ``"Also, please: <title>. <description>"`` phrasing.
* The plugin attaches the :class:`PlanReconciler` for the
  passthrough invocation and clears it afterwards.
* Legacy :meth:`invoke` also uses the gentle phrasing now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.types import Plan, Session, Task


def _make_agent() -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name="test_agent",
        model="fake-model",
        description="Test",
        instruction="x",
    )


@dataclass
class _RecordingFakeRunner:
    """Runner stub that captures the ``new_message`` handed to ``run_async``.

    Emits zero events so the adapter loop exits immediately on generator
    end; the test asserts on ``captured_new_message``.
    """

    captured_new_message: Any = None
    captured_user_id: str = ""
    captured_session_id: str = ""
    session_service: Any = None
    plugin_manager: Any = field(default=None)

    async def run_async(self, **kwargs: Any):
        self.captured_new_message = kwargs.get("new_message")
        self.captured_user_id = str(kwargs.get("user_id") or "")
        self.captured_session_id = str(kwargs.get("session_id") or "")
        if False:  # pragma: no cover -- generator shape
            yield None


def _user_text(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    return "\n".join(str(getattr(p, "text", "") or "") for p in parts if getattr(p, "text", None))


async def test_invoke_passthrough_sends_user_message_verbatim() -> None:
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_make_agent())
    adapter._runner = _RecordingFakeRunner()
    adapter._session_id = "stub-session"

    session = Session(run_id="r1", plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[], edges=[]))
    await adapter.invoke_passthrough(
        "make a presentation about solar panels",
        session=session,
    )

    captured = adapter._runner.captured_new_message
    text = _user_text(captured)
    assert text == "make a presentation about solar panels", (
        f"passthrough should send user input verbatim; got {text!r}"
    )
    assert "Task:" not in text
    assert "goldfive." not in text
    assert "Also, please" not in text


async def test_invoke_follow_up_uses_gentle_phrasing() -> None:
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_make_agent())
    adapter._runner = _RecordingFakeRunner()
    adapter._session_id = "stub-session"

    task = Task(
        id="t0",
        title="Gather 3 bullet points",
        description="Summarise key facts.",
    )
    session = Session(
        run_id="r1",
        plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )
    await adapter.invoke_follow_up(task, session)

    captured = adapter._runner.captured_new_message
    text = _user_text(captured)
    assert text.startswith("Also, please:"), f"follow-up should use gentle prefix; got {text!r}"
    assert "Gather 3 bullet points" in text
    assert "Summarise key facts." in text
    assert "Task:" not in text
    assert "goldfive." not in text


async def test_legacy_invoke_uses_follow_up_phrasing() -> None:
    """The old ``invoke(task)`` path now delegates to the gentle
    follow-up shape — the jargon-heavy ``"Task: X. Use the
    goldfive.* session-state keys ..."`` message is gone.
    """
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_make_agent())
    adapter._runner = _RecordingFakeRunner()
    adapter._session_id = "stub-session"

    task = Task(id="t0", title="do work", description="the thing")
    session = Session(
        run_id="r1",
        plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )
    await adapter.invoke(task=task, session=session)

    text = _user_text(adapter._runner.captured_new_message)
    assert text.startswith("Also, please:"), text
    assert "goldfive." not in text


async def test_invoke_passthrough_attaches_then_clears_reconciler() -> None:
    """The plugin should have the reconciler for the duration of the
    invocation and have it cleared in ``finally``."""
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reconciler import PlanReconciler

    class _ProbingRunner(_RecordingFakeRunner):
        """Runner that probes the plugin state while run_async is active."""

        plugin_ref: Any = None
        saw_reconciler_during_run: bool = False

        async def run_async(self, **kwargs: Any):
            self.captured_new_message = kwargs.get("new_message")
            self.saw_reconciler_during_run = self.plugin_ref._reconciler is not None
            if False:  # pragma: no cover
                yield None

    adapter = ADKAdapter(_make_agent())
    runner = _ProbingRunner()
    runner.plugin_ref = adapter._plugin
    adapter._runner = runner
    adapter._session_id = "stub-session"

    class _NullSteerer:
        _sinks: list[Any] = []

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def _handle_drift(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    session = Session(run_id="r1", plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[], edges=[]))
    rec = PlanReconciler(session=session, steerer=_NullSteerer(), host_agent_name="test_agent")

    await adapter.invoke_passthrough("hello", session=session, reconciler=rec)

    assert runner.saw_reconciler_during_run is True
    # Cleared in the adapter's ``finally`` via clear_active_context.
    assert adapter._plugin._reconciler is None


async def test_plugin_forwards_before_agent_to_reconciler() -> None:
    """End-to-end: plugin.before_agent_callback → reconciler.on_before_agent
    with the running agent's name."""
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reconciler import PlanReconciler

    adapter = ADKAdapter(_make_agent())

    class _RecordingSteerer:
        _sinks: list[Any] = []

        def __init__(self) -> None:
            self.transitions: list[tuple[str, Any]] = []

        async def transition(
            self,
            task_id: str,
            to: Any,
            *,
            detail: str = "",  # noqa: ARG002
            session: Any,
        ) -> None:
            self.transitions.append((task_id, to))
            if session.plan is None:
                return
            for t in session.plan.tasks:
                if t.id == task_id:
                    t.status = to
                    return

        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def _handle_drift(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    steerer = _RecordingSteerer()
    session = Session(
        run_id="r1",
        plan=Plan(
            id="p0",
            run_id="r1",
            goal_ids=[],
            tasks=[Task(id="t0", title="a", assignee_agent_id="research_agent")],
            edges=[],
        ),
    )
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="test_agent")
    adapter._plugin.set_active_context_ = adapter._plugin.set_active_context  # alias for clarity

    # Install the active ctx + reconciler the way invoke_passthrough does.
    from goldfive.adapters._adk_plugin import SessionContext

    ctx = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        tools=[],
        host_agent_name="test_agent",
    )
    adapter._plugin.set_active_context(ctx)
    adapter._plugin.set_reconciler(rec)

    # Fabricate a minimal ADK agent + callback context for the plugin.
    class _Agent:
        name = "research_agent"

    class _Ctx:
        _invocation_context = None  # absent is fine

    await adapter._plugin.before_agent_callback(agent=_Agent(), callback_context=_Ctx())

    from goldfive.types import TaskStatus

    assert steerer.transitions == [("t0", TaskStatus.RUNNING)]

    # Cleanup — same guarantee the adapter's finally would provide.
    adapter._plugin.clear_active_context()
    assert adapter._plugin._reconciler is None
