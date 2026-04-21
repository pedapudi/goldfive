"""Dispatch-by-assignee tests.

Phase 2 of feat/registry-dispatch-model: ``ADKAdapter.invoke`` routes
to ``self._runners[task.assignee_agent_id]`` — not to the wrap target's
root. Covers the happy path, error paths, and sequential cross-agent
dispatch (no state leakage between invocations).

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("google.adk")


def _mk(name: str) -> Any:
    from google.adk.agents.llm_agent import LlmAgent

    return LlmAgent(name=name, model="fake-model", description=name, instruction="x")


@dataclass
class _Event:
    """ADK-event duck-type for dispatch recording (no text / fn calls)."""

    marker: str = ""
    content: Any = None


@dataclass
class _RecordingRunner:
    """Runner stub that records every ``run_async`` call on a shared log.

    Used as a spy to prove registry dispatch routed to the right
    per-agent runner (not the coordinator's runner).
    """

    name: str = ""
    log: list = field(default_factory=list)
    session_service: Any = None
    plugin_manager: Any = None
    app_name: str = ""

    async def run_async(self, **kwargs: Any):  # noqa: ARG002
        self.log.append(self.name)
        yield _Event(marker=self.name)


def _wire_recording_runners(adapter: Any, shared_log: list) -> None:
    """Swap every per-agent runner on ``adapter`` with a
    :class:`_RecordingRunner` sharing ``shared_log``. Pre-seed session
    ids so the adapter skips real ADK session creation."""
    for agent_name in list(adapter._runners):
        adapter._runners[agent_name] = _RecordingRunner(
            name=agent_name,
            log=shared_log,
            app_name=agent_name,
        )
    root_name = str(getattr(adapter._agent, "name", "") or "")
    adapter._runner = adapter._runners.get(
        root_name, next(iter(adapter._runners.values()))
    )
    adapter._session_ids = {name: f"sess-{name}" for name in adapter._runners}


# ---------------------------------------------------------------------------
# Assignee-name dispatch
# ---------------------------------------------------------------------------


async def test_invoke_dispatches_to_named_assignee_only() -> None:
    """A spy on one specific runner fires — others stay silent."""
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    research = _mk("research_agent")
    web = _mk("web_agent")
    coord = _mk("coord")
    coord.tools = [AgentTool(research), AgentTool(web)]

    adapter = ADKAdapter(coord)
    log: list[str] = []
    _wire_recording_runners(adapter, log)

    task = Task(id="t1", title="gather data", assignee_agent_id="research_agent")
    session = Session(
        run_id="r1",
        plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )
    await adapter.invoke(task=task, session=session)

    # ONLY research_agent got dispatched — coord/web stay silent.
    assert log == ["research_agent"]


async def test_invoke_raises_clear_error_for_unknown_assignee() -> None:
    """Plan bug → ValueError with an ``available:`` hint naming candidates."""
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    research = _mk("research_agent")
    coord = _mk("coord")
    coord.tools = [AgentTool(research)]

    adapter = ADKAdapter(coord)
    task = Task(id="t1", title="x", assignee_agent_id="nobody")
    session = Session(
        run_id="r1",
        plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )

    with pytest.raises(ValueError) as excinfo:
        await adapter.invoke(task=task, session=session)

    msg = str(excinfo.value)
    assert "unknown agent" in msg
    assert "'nobody'" in msg
    # Available-names hint is included so the caller can fix the plan.
    assert "available" in msg.lower()
    assert "research_agent" in msg
    assert "coord" in msg


async def test_invoke_empty_assignee_dispatches_to_root() -> None:
    """Empty assignee preserves single-agent wrap contract — dispatch to root."""
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    sub = _mk("sub")
    coord = _mk("coord")
    coord.tools = [AgentTool(sub)]

    adapter = ADKAdapter(coord)
    log: list[str] = []
    _wire_recording_runners(adapter, log)

    # assignee_agent_id defaults to "".
    task = Task(id="t1", title="x")
    session = Session(
        run_id="r1",
        plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )
    await adapter.invoke(task=task, session=session)

    # Root coordinator's runner got it — NOT sub.
    assert log == ["coord"]


# ---------------------------------------------------------------------------
# Sequential dispatch across assignees
# ---------------------------------------------------------------------------


async def test_sequential_dispatch_to_different_assignees_clean_between() -> None:
    """Three back-to-back invocations route to the named runner each time.

    No leakage: each call logs exactly one name, and the ordering
    ``A, B, A`` is preserved — proves the dispatch resolution is
    stateless across invocations.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    a = _mk("a")
    b = _mk("b")
    coord = _mk("coord")
    coord.tools = [AgentTool(a), AgentTool(b)]

    adapter = ADKAdapter(coord)
    log: list[str] = []
    _wire_recording_runners(adapter, log)

    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[
                Task(id="t1", title="do a", assignee_agent_id="a"),
                Task(id="t2", title="do b", assignee_agent_id="b"),
                Task(id="t3", title="again a", assignee_agent_id="a"),
            ],
            edges=[],
        ),
    )

    for task in session.plan.tasks:
        await adapter.invoke(task=task, session=session)

    assert log == ["a", "b", "a"]


async def test_sequential_dispatch_plugin_active_ctx_cleared_between_invokes() -> None:
    """After each invoke the plugin's ``_active_ctx`` returns to ``None``.

    Regression: if ``clear_active_context`` misfires (or never fires),
    subsequent invocations on other agents would see the previous
    task's context — cross-task leakage that breaks the reliability
    contract.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    a = _mk("a")
    coord = _mk("coord")
    coord.tools = [AgentTool(a)]

    adapter = ADKAdapter(coord)
    log: list[str] = []
    _wire_recording_runners(adapter, log)

    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[Task(id="t1", title="x", assignee_agent_id="a")],
            edges=[],
        ),
    )

    task = Task(id="t1", title="x", assignee_agent_id="a")
    await adapter.invoke(task=task, session=session)

    # After invoke returns, the plugin MUST have cleared its active ctx
    # — otherwise a stale ctx would bleed into the next dispatch.
    assert adapter._plugin._active_ctx is None
    # And the top_invocation_id pin must also be released.
    assert adapter._plugin._top_invocation_id == ""


async def test_sequential_dispatch_shares_one_session_id_across_agents() -> None:
    """Per-agent runners all share the SAME ADK session id string
    (goldfive#123). The HarmonografTelemetryPlugin stamps
    ``ctx.session.id`` onto every span, so a uniform id across runners
    rolls an adk-web run's spans up under one harmonograf session
    (before the fix, sub-agent runners minted their own uuids and the
    UI scattered one run across 3+ sessions).

    Safety: each :class:`InMemoryRunner` has its own
    :class:`InMemorySessionService`; ADK looks sessions up as
    ``(app_name, user_id, session_id)`` on the PER-RUNNER service, so
    sharing the id string across runners does NOT collide.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    a = _mk("a")
    b = _mk("b")
    coord = _mk("coord")
    coord.tools = [AgentTool(a), AgentTool(b)]

    adapter = ADKAdapter(coord)
    log: list[str] = []
    _wire_recording_runners(adapter, log)
    # Clear the pre-seeded ids so each agent's session id is assigned
    # on first dispatch (exercising the real _ensure_session_for flow
    # against our recording runners, which have no session_service).
    adapter._session_ids = {}
    adapter._outer_session_id = None

    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[
                Task(id="t1", title="x", assignee_agent_id="a"),
                Task(id="t2", title="y", assignee_agent_id="b"),
            ],
            edges=[],
        ),
    )

    for task in session.plan.tasks:
        await adapter.invoke(task=task, session=session)

    # Both agents got the SAME shared session id.
    assert "a" in adapter._session_ids
    assert "b" in adapter._session_ids
    assert adapter._session_ids["a"] == adapter._session_ids["b"], (
        "per-agent session ids must share ONE outer id so telemetry "
        "plugins that stamp ctx.session.id onto spans roll up under a "
        "single harmonograf session (goldfive#123)"
    )
    # And that id is cached on the adapter for reuse.
    assert adapter._outer_session_id == adapter._session_ids["a"]
