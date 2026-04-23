"""Tests for task_id wiring (goldfive#191 Layers 1 + 2).

Covers:

* Layer 1 — ``_GoldfiveADKPlugin.before_agent_callback`` pins
  ``goldfive.current_task_id`` onto BOTH the ADK session.state and the
  goldfive orchestration session.state when the starting sub-agent has
  exactly one PENDING / RUNNING task assigned to it. Ambiguous
  (multiple) and absent (zero) matches leave state unset so the
  ``missing_task_id`` error path still fires.
* Layer 2 — reporting-tool handlers fall back to
  ``session.state["goldfive.current_task_id"]`` when the model's tool
  call omits the ``task_id`` arg, yield the canonical
  ``missing_task_id`` error when neither source supplies a value, and
  honour an explicit arg over the state fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID  # noqa: E402
from goldfive.adapters._tool_invocation import invoke_tool  # noqa: E402
from goldfive.reporting import (  # noqa: E402
    BUILTIN_REPORTING_TOOLS,
    ReportingToolSpec,
)
from goldfive.types import (  # noqa: E402
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StateCtx:
    """Minimal ADK-callback-context stub with a ``session.state`` dict."""

    class _Session:
        def __init__(self, state: dict) -> None:
            self.state = state

    def __init__(self, state: dict) -> None:
        self._state = state

    @property
    def session(self) -> Any:
        return _StateCtx._Session(self._state)


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name


class _RecordingSteerer:
    """Steerer stub that records transitions + drifts for assertions."""

    def __init__(self) -> None:
        self.transitions: list[tuple[str, Any, str]] = []
        self.drifts: list[Any] = []

    async def mark_task_running(self, task_id: str, *, session: Any, detail: str = "") -> None:
        self.transitions.append((task_id, "RUNNING", detail))

    async def mark_task_progress(
        self, task_id: str, *, session: Any, fraction: float = 0.0, detail: str = ""
    ) -> None:
        self.transitions.append((task_id, "PROGRESS", detail))

    async def mark_task_completed(
        self, task_id: str, *, session: Any, summary: str = "", artifacts: Any = None
    ) -> None:
        self.transitions.append((task_id, "COMPLETED", summary))

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        session: Any,
        reason: str = "",
        recoverable: bool = True,
    ) -> None:
        self.transitions.append((task_id, "FAILED", reason))

    async def mark_task_blocked(
        self, task_id: str, *, session: Any, blocker: str = "", needed: str = ""
    ) -> None:
        self.transitions.append((task_id, "BLOCKED", blocker))

    async def report_new_work_discovered(self, **kwargs: Any) -> None:
        pass

    async def report_plan_divergence(self, **kwargs: Any) -> None:
        pass

    async def observe(self, event: Any, session: Any) -> None:
        self.drifts.append(event)

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, event: Any, session: Any) -> Any:
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _plan_with(*tasks: Task, edges: list[TaskEdge] | None = None) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=list(tasks),
        edges=list(edges or []),
        summary="",
    )


def _session_with(plan: Plan | None) -> Session:
    return Session(run_id="r1", plan=plan)


def _ctx_for(session: Session, agent_name: str) -> tuple[dict, Any]:
    """Build a ({adk_state}, callback_context) pair for plugin callbacks."""
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=None,
            task=None,
            tool_handlers={spec.name: spec.handler for spec in BUILTIN_REPORTING_TOOLS},
            host_agent_name=agent_name,
        ),
    }
    return state, _StateCtx(state)


# ---------------------------------------------------------------------------
# Layer 1 — before_agent_callback pins current_task_id
# ---------------------------------------------------------------------------


async def test_before_agent_callback_pins_unambiguous_task() -> None:
    """Exactly one PENDING task assigned to the starting agent → state pinned."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="research", assignee_agent_id="research_agent"),
            Task(id="t2", title="code", assignee_agent_id="web_developer"),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    # ADK session.state side (sub-agent-visible).
    assert state[KEY_CURRENT_TASK_ID] == "t1"
    # Goldfive orchestration-state side (handler fallback).
    assert session.state["goldfive.current_task_id"] == "t1"


async def test_before_agent_callback_ambiguous_match_leaves_unset() -> None:
    """Two PENDING tasks for the same agent → no stamp."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="first", assignee_agent_id="research_agent"),
            Task(id="t2", title="second", assignee_agent_id="research_agent"),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert KEY_CURRENT_TASK_ID not in state
    assert "goldfive.current_task_id" not in session.state


async def test_before_agent_callback_no_match_leaves_unset() -> None:
    """Zero PENDING tasks for the agent (off-plan agent) → no stamp."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="first", assignee_agent_id="research_agent"),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    # Agent name doesn't match any assignee in the plan.
    await plugin.before_agent_callback(
        agent=_Agent("rogue_agent"),
        callback_context=ctx,
    )

    assert KEY_CURRENT_TASK_ID not in state
    assert "goldfive.current_task_id" not in session.state


async def test_before_agent_callback_includes_running_tasks() -> None:
    """A RUNNING task for the agent also counts (re-entry / re-spawn path)."""
    plugin = make_adk_plugin(host_agent_name="coord")
    running_task = Task(
        id="t1",
        title="research",
        assignee_agent_id="research_agent",
        status=TaskStatus.RUNNING,
    )
    session = _session_with(_plan_with(running_task))
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert state[KEY_CURRENT_TASK_ID] == "t1"
    assert session.state["goldfive.current_task_id"] == "t1"


async def test_before_agent_callback_skips_terminal_tasks() -> None:
    """A COMPLETED task does not count; if it's the only match, state stays unset."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(
                id="t1",
                title="already done",
                assignee_agent_id="research_agent",
                status=TaskStatus.COMPLETED,
            ),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert KEY_CURRENT_TASK_ID not in state
    assert "goldfive.current_task_id" not in session.state


# ---------------------------------------------------------------------------
# Layer 2 — reporting-tool handlers default task_id from state
# ---------------------------------------------------------------------------


def _get_spec(name: str) -> ReportingToolSpec:
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"builtin tool {name!r} missing")


async def test_reporting_tool_uses_state_task_id_when_arg_missing() -> None:
    """State fallback reaches the handler via ``invoke_tool``."""
    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec("report_task_started")
    result = await invoke_tool(
        [spec],
        "report_task_started",
        {"detail": "starting"},  # no task_id
        session,
        steerer,
    )

    assert result == {"acknowledged": True}
    # The handler ran with the state-derived id.
    assert steerer.transitions == [("t-1", "RUNNING", "starting")]


async def test_reporting_tool_honors_explicit_arg_over_state() -> None:
    """An explicit ``task_id`` wins against the session-state fallback."""
    steerer = _RecordingSteerer()
    # Plan carries both ids so the ``unknown_task_id`` guard doesn't mask
    # the behavior we're trying to observe.
    session = _session_with(
        _plan_with(
            Task(id="t-1", title="a", assignee_agent_id="a"),
            Task(id="t-2", title="b", assignee_agent_id="b"),
        )
    )
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec("report_task_started")
    result = await invoke_tool(
        [spec],
        "report_task_started",
        {"task_id": "t-2", "detail": "explicit wins"},
        session,
        steerer,
    )

    assert result == {"acknowledged": True}
    # Explicit arg is what reached the handler.
    assert steerer.transitions == [("t-2", "RUNNING", "explicit wins")]


async def test_reporting_tool_errors_when_neither_arg_nor_state() -> None:
    """Neither arg nor state → canonical ``missing_task_id`` rejection."""
    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))
    # session.state intentionally empty — no fallback.

    spec = _get_spec("report_task_started")
    result = await invoke_tool(
        [spec],
        "report_task_started",
        {"detail": "starting"},
        session,
        steerer,
    )

    assert result["acknowledged"] is False
    assert result["error"] == "missing_task_id"
    assert result["tool"] == "report_task_started"
    # No transition — handler must not run.
    assert steerer.transitions == []


@pytest.mark.parametrize(
    "tool_name,extra_args",
    [
        ("report_task_started", {}),
        ("report_task_progress", {"fraction": 0.5}),
        ("report_task_completed", {"summary": "done"}),
        ("report_task_failed", {"reason": "oops", "recoverable": True}),
        ("report_task_blocked", {"blocker": "missing input"}),
    ],
)
async def test_all_task_scoped_reporting_tools_default_task_id(
    tool_name: str, extra_args: dict[str, Any]
) -> None:
    """Every task-scoped reporting tool defaults ``task_id`` from state.

    Parameterized over the five task-scoped tools in
    :data:`BUILTIN_REPORTING_TOOLS`. The remaining three (the plan-level
    ``report_plan_divergence`` + ``report_new_work_discovered`` and the
    waiter-based ``report_awaiting_approval``) are not task-id-gated by
    the invoke_tool schema layer (``report_plan_divergence`` is
    plan-level; ``report_new_work_discovered`` takes ``parent_task_id``;
    ``report_awaiting_approval`` blocks on a waiter this test won't set
    up), so they follow a different shape — covered separately.
    """
    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec(tool_name)
    args: dict[str, Any] = dict(extra_args)  # no task_id supplied
    result = await invoke_tool([spec], tool_name, args, session, steerer)

    # Each tool either ACKs or passes through to the handler with the
    # state-derived id; neither should produce a ``missing_task_id``.
    assert result.get("error") != "missing_task_id", (
        f"{tool_name} did not default task_id from state: {result!r}"
    )
    # Handler ran with the resolved id.
    assert any(t[0] == "t-1" for t in steerer.transitions), (
        f"{tool_name} did not invoke the handler with t-1: transitions={steerer.transitions!r}"
    )


async def test_report_awaiting_approval_accepts_state_task_id() -> None:
    """``report_awaiting_approval`` also resolves task_id from state.

    Unlike the other tools, it fails at the ``prompt`` schema check if
    absent, but we're testing the state-fallback behavior specifically.
    We don't drive the waiter to completion — we just verify the
    handler does NOT return ``missing_task_id`` when state supplies
    the id.
    """
    import asyncio

    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec("report_awaiting_approval")
    # Set the waiter in advance so the handler doesn't block forever.
    waiter = asyncio.Event()
    session.pending_approvals["t-1"] = waiter
    session.pending_approvals_meta["t-1"] = {
        "kind": "task",
        "prompt": "ok?",
        "task_id": "t-1",
        "decision": "approve",
        "detail": "",
    }
    waiter.set()

    result = await invoke_tool(
        [spec],
        "report_awaiting_approval",
        {"prompt": "approve this?"},  # no task_id
        session,
        steerer,
    )

    assert result.get("error") != "missing_task_id"
    assert result.get("decision") == "approve"


async def test_missing_task_id_still_rejects_when_state_empty() -> None:
    """Regression guard: the existing ``missing_task_id`` rejection path
    still fires when neither ``args.task_id`` nor session-state supplies
    a value. The fallback must not degrade the schema guard."""
    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))

    spec = _get_spec("report_task_failed")
    for bad_args in (
        {},
        {"task_id": ""},
        {"task_id": "   "},
        {"task_id": None, "reason": "x"},
        {"reason": "x"},
    ):
        result = await invoke_tool([spec], "report_task_failed", dict(bad_args), session, steerer)
        assert result["acknowledged"] is False, bad_args
        assert result["error"] == "missing_task_id", bad_args
    # No handler runs → no transitions.
    assert steerer.transitions == []


# ---------------------------------------------------------------------------
# Compound-assignee end-to-end (goldfive#214)
# ---------------------------------------------------------------------------


async def test_before_agent_callback_pins_when_plan_came_from_compound_json() -> None:
    """Plans built from compound-form planner JSON still pin the right task.

    Regression guard for #214: the planner's JSON→TaskPlan path normalizes
    ``"<client>:<agent>"`` to the bare ADK name, so the reconciler /
    plugin match (`assignee == agent_name`) finds exactly one candidate.
    """
    from goldfive.planner import _plan_from_json

    payload = {
        "summary": "s",
        "tasks": [
            {
                "id": "t1",
                "title": "research",
                "assignee_agent_id": "presentation-orchestrated-9b2b3a9c7289:research_agent",
            },
            {
                "id": "t2",
                "title": "code",
                "assignee_agent_id": "presentation-orchestrated-9b2b3a9c7289:web_developer",
            },
        ],
    }
    plan = _plan_from_json(payload, run_id="r1", goal_ids=[])
    assert plan is not None
    # Sanity: normalization happened at parse time.
    assert [t.assignee_agent_id for t in plan.tasks] == [
        "research_agent",
        "web_developer",
    ]

    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(plan)
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert state[KEY_CURRENT_TASK_ID] == "t1"
    assert session.state["goldfive.current_task_id"] == "t1"
