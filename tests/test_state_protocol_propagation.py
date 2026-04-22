"""State-protocol reliability tests for the registry-dispatch model.

CRITICAL: goldfive's state-protocol writes cannot be best-effort. They
MUST propagate into AgentTool-spawned sub-Runners so the sub-agent can
read the active task, plan context, run id, and tools_available off its
own live session state.

Phase 1 moved the authoritative write into the plugin's
``before_run_callback`` (against the LIVE invocation session) precisely
so the write lands on the session the sub-Runner actually runs against.
These tests pin that contract.

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.types import Plan, Session, Task

# ---------------------------------------------------------------------------
# Scripted LLMs used to trigger AgentTool dispatch
# ---------------------------------------------------------------------------


def _make_agent_a(tool_name: str) -> Any:
    """Build agent A whose first LLM turn calls ``tool_name`` (an AgentTool).

    Turn 1: emit a ``function_call`` for ``tool_name``.
    Turn 2: return ``turn_complete=True`` after the sub-agent responds.
    """
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _ScriptedA(BaseLlm):
        model: str = "fake-model"
        _step: int = 0

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            self._step += 1
            if self._step == 1:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    id="call_b",
                                    name=tool_name,
                                    args={"request": "do it"},
                                )
                            ),
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="ok")],
                    ),
                    turn_complete=True,
                )

    return _ScriptedA


def _make_agent_b_llm() -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _ScriptedB(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="done")],
                ),
                turn_complete=True,
            )

    return _ScriptedB


class _StubSteerer:
    async def observe(self, *a: Any, **kw: Any) -> None:
        pass

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, *a: Any, **kw: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _read_state_via_before_model(observed: dict[str, Any]):
    """Return a ``before_model_callback`` that snapshots every
    ``goldfive.*`` state key into ``observed`` on entry.

    ADK invokes this as a sync function receiving ``callback_context``
    and ``llm_request``. Navigates the several shapes callback_context
    may expose (``_invocation_context.session.state`` / ``session.state``)
    so the snapshot works across ADK versions.
    """
    from goldfive.adapters._adk_state_protocol import (
        KEY_CURRENT_TASK_ASSIGNEE,
        KEY_CURRENT_TASK_ID,
        KEY_CURRENT_TASK_TITLE,
        KEY_PLAN_ID,
        KEY_PLAN_SUMMARY,
        KEY_RUN_ID,
        KEY_TOOLS_AVAILABLE,
    )

    def _before_model(callback_context: Any, llm_request: Any) -> None:  # noqa: ARG001
        inv_ctx = getattr(callback_context, "_invocation_context", None)
        state = None
        if inv_ctx is not None:
            sess = getattr(inv_ctx, "session", None)
            state = getattr(sess, "state", None)
        if state is None:
            sess = getattr(callback_context, "session", None)
            state = getattr(sess, "state", None)
        if state is None:
            return None
        # Snapshot only the first time — some B callbacks may fire more
        # than once if the scripted LLM hands more turns to B.
        if observed:
            return None
        for key in (
            KEY_RUN_ID,
            KEY_PLAN_ID,
            KEY_PLAN_SUMMARY,
            KEY_CURRENT_TASK_ID,
            KEY_CURRENT_TASK_TITLE,
            KEY_CURRENT_TASK_ASSIGNEE,
            KEY_TOOLS_AVAILABLE,
        ):
            try:
                observed[key] = state.get(key)
            except Exception:
                observed[key] = None
        return None

    return _before_model


# ---------------------------------------------------------------------------
# State propagation through an AgentTool boundary
# ---------------------------------------------------------------------------


async def test_state_protocol_current_task_id_visible_in_sub_runner() -> None:
    """B's ``before_model_callback`` sees ``goldfive.current_task_id`` on
    the sub-Runner's live session — the authoritative state-protocol
    write must cross the AgentTool boundary.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID
    from goldfive.adapters.adk import ADKAdapter

    observed: dict[str, Any] = {}
    agent_b = Agent(
        name="agent_b",
        model=_make_agent_b_llm()(),
        instruction="",
        before_model_callback=_read_state_via_before_model(observed),
    )
    agent_a = Agent(
        name="agent_a",
        model=_make_agent_a("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    adapter.bind_steerer(_StubSteerer())

    task = Task(id="my-task-id", title="compound", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    assert observed, "B's before_model_callback never ran — AgentTool dispatch broke"
    assert observed.get(KEY_CURRENT_TASK_ID) == "my-task-id", (
        f"B's sub-Runner saw current_task_id={observed.get(KEY_CURRENT_TASK_ID)!r}; "
        "expected 'my-task-id'. The authoritative state write in "
        "plugin.before_run_callback did not propagate through AgentTool."
    )


async def test_state_protocol_run_id_visible_in_sub_runner() -> None:
    """``goldfive.run_id`` is visible on B's sub-session."""
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters._adk_state_protocol import KEY_RUN_ID
    from goldfive.adapters.adk import ADKAdapter

    observed: dict[str, Any] = {}
    agent_b = Agent(
        name="agent_b",
        model=_make_agent_b_llm()(),
        instruction="",
        before_model_callback=_read_state_via_before_model(observed),
    )
    agent_a = Agent(
        name="agent_a",
        model=_make_agent_a("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    adapter.bind_steerer(_StubSteerer())

    task = Task(id="t1", title="x", assignee_agent_id="agent_a")
    plan = Plan(id="p-xyz", run_id="run-9876", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="run-9876", plan=plan)
    await adapter.invoke(task=task, session=session)

    assert observed.get(KEY_RUN_ID) == "run-9876"


async def test_state_protocol_plan_context_visible_in_sub_runner() -> None:
    """``goldfive.plan_id`` and ``plan_summary`` propagate to the sub-Runner."""
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters._adk_state_protocol import KEY_PLAN_ID, KEY_PLAN_SUMMARY
    from goldfive.adapters.adk import ADKAdapter

    observed: dict[str, Any] = {}
    agent_b = Agent(
        name="agent_b",
        model=_make_agent_b_llm()(),
        instruction="",
        before_model_callback=_read_state_via_before_model(observed),
    )
    agent_a = Agent(
        name="agent_a",
        model=_make_agent_a("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    adapter.bind_steerer(_StubSteerer())

    task = Task(id="t1", title="x", assignee_agent_id="agent_a")
    plan = Plan(
        id="plan-ABC",
        run_id="r1",
        goal_ids=[],
        tasks=[task],
        edges=[],
        summary="compound task plan",
    )
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    assert observed.get(KEY_PLAN_ID) == "plan-ABC"
    assert observed.get(KEY_PLAN_SUMMARY) == "compound task plan"


async def test_state_protocol_tools_available_visible_in_sub_runner() -> None:
    """``goldfive.tools_available`` on the sub-Runner lists the reporting
    tools goldfive registered on the adapter.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters._adk_state_protocol import KEY_TOOLS_AVAILABLE
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import ReportingToolSpec

    observed: dict[str, Any] = {}
    agent_b = Agent(
        name="agent_b",
        model=_make_agent_b_llm()(),
        instruction="",
        before_model_callback=_read_state_via_before_model(observed),
    )
    agent_a = Agent(
        name="agent_a",
        model=_make_agent_a("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    adapter.bind_steerer(_StubSteerer())

    async def _noop(args: dict, session: Any, steerer: Any) -> dict:  # noqa: ARG001
        return {"acknowledged": True}

    # Register a single reporting tool so tools_available has a concrete
    # name to look for.
    spec = ReportingToolSpec(
        name="report_task_started",
        description="Mark started.",
        parameters={"type": "object", "properties": {}},
        handler=_noop,
    )
    await adapter.register_reporting_tools([spec])

    task = Task(id="t1", title="x", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    tools = observed.get(KEY_TOOLS_AVAILABLE) or []
    assert "report_task_started" in tools, (
        f"sub-Runner state.tools_available={tools!r}; expected our "
        "registered reporting tool. tools_available is written by "
        "plugin.before_run_callback on the live sub-Runner session."
    )


# ---------------------------------------------------------------------------
# Direct before_run_callback plumbing (state writes land on ctx session)
# ---------------------------------------------------------------------------


async def test_before_run_callback_writes_state_on_live_session_directly() -> None:
    """When ``before_run_callback`` is invoked against a constructed
    invocation_context, the state-protocol keys land on THAT context's
    session — not on the adapter's or a cached copy.

    This is the unit-level correctness test for the reliability
    contract. The integration test above proves the callback fires at
    the right time; this test proves that when it fires, it writes to
    the right place.
    """
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin
    from goldfive.adapters._adk_state_protocol import (
        KEY_CURRENT_TASK_ID,
        KEY_CURRENT_TASK_TITLE,
        KEY_PLAN_ID,
        KEY_RUN_ID,
        KEY_TOOLS_AVAILABLE,
    )

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(
        run_id="run-42",
        plan=Plan(
            id="plan-42",
            run_id="run-42",
            goal_ids=[],
            tasks=[Task(id="t1", title="xyz")],
            edges=[],
            summary="summary",
        ),
    )
    task = Task(id="t1", title="xyz")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={"report_task_started": lambda *a, **kw: None},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = type("A", (), {"name": "agent_a"})()

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    # The state write landed on THIS session — not a sibling.
    assert inv_ctx.session.state[KEY_RUN_ID] == "run-42"
    assert inv_ctx.session.state[KEY_PLAN_ID] == "plan-42"
    assert inv_ctx.session.state[KEY_CURRENT_TASK_ID] == "t1"
    assert inv_ctx.session.state[KEY_CURRENT_TASK_TITLE] == "xyz"
    assert "report_task_started" in (inv_ctx.session.state[KEY_TOOLS_AVAILABLE] or [])


async def test_before_run_callback_no_op_when_no_active_ctx() -> None:
    """Without an active ``SessionContext``, ``before_run_callback`` is a no-op.

    Regression: eager writes when ``_active_ctx is None`` would crash
    tests that instantiate the plugin outside the adapter. The callback
    returns silently and writes nothing.
    """
    from goldfive.adapters._adk_plugin import make_adk_plugin

    plugin = make_adk_plugin(host_agent_name="agent_a")

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = None

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    # No keys written.
    assert inv_ctx.session.state == {}


async def test_top_level_invocation_id_pinned_then_released() -> None:
    """Top-level ``before_run`` pins ``_top_invocation_id``; matching
    ``after_run`` releases it. Ensures nested sub-Runners can correctly
    attribute themselves via ``parent_invocation_id``.
    """
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(run_id="run-1")
    task = Task(id="t1", title="x")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self, inv_id: str) -> None:
            self.session = _Session()
            self.invocation_id = inv_id
            self.agent = type("A", (), {"name": "agent_a"})()

    top = _InvCtx("inv-top")
    await plugin.before_run_callback(invocation_context=top)
    assert plugin._top_invocation_id == "inv-top"

    # Nested sub-Runner fires before_run too — must NOT overwrite the pin.
    nested = _InvCtx("inv-nested")
    await plugin.before_run_callback(invocation_context=nested)
    assert plugin._top_invocation_id == "inv-top", (
        "nested AgentTool sub-Runner overwrote the top-level invocation pin"
    )

    # Nested after_run does NOT release (its id != top).
    await plugin.after_run_callback(invocation_context=nested)
    assert plugin._top_invocation_id == "inv-top"

    # Top-level after_run RELEASES.
    await plugin.after_run_callback(invocation_context=top)
    assert plugin._top_invocation_id == ""


# ---------------------------------------------------------------------------
# Orchestration-state bridge (goldfive#170)
#
# DefaultSteerer (#152) writes active_steer / goals_summary /
# cancelled_function_call_ids onto ``goldfive.Session.state`` —
# framework-agnostic orchestration dict. GoldfivePlanner (#153) reads
# the same logical keys off the ADK session.state for its per-turn
# injection. Before #170 there was no bridge between the two
# surfaces, so the planner always rendered ``(none)``.
#
# The bridge lives in :meth:`_GoldfiveADKPlugin.before_run_callback`
# and fires on every invocation — including AgentTool sub-Runners
# whose own ``before_run_callback`` repeats the bridge against their
# own live session. These tests pin the contract.
# ---------------------------------------------------------------------------


async def test_bridge_active_steer_copies_from_orchestration_state_to_adk_state() -> None:
    """When goldfive.Session.state carries an active steer, the plugin's
    ``before_run_callback`` bridges it onto the live ADK session.state.
    """
    from goldfive import orchestration_state as _ostate
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin
    from goldfive.adapters._adk_state_protocol import (
        KEY_ACTIVE_STEER_AT_TURN,
        KEY_ACTIVE_STEER_BODY,
    )

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(run_id="run-42")
    # Stamp the orchestration-level active_steer as DefaultSteerer would.
    _ostate.set_active_steer(
        session.state,
        body="focus on the cost angle",
        at_turn=7,
    )

    task = Task(id="t1", title="x", assignee_agent_id="agent_a")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = type("A", (), {"name": "agent_a"})()

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    assert inv_ctx.session.state[KEY_ACTIVE_STEER_BODY] == "focus on the cost angle"
    assert inv_ctx.session.state[KEY_ACTIVE_STEER_AT_TURN] == 7


async def test_bridge_goals_summary_copies_from_orchestration_state_to_adk_state() -> None:
    """``goldfive.goals_summary`` crosses the bridge."""
    from goldfive import orchestration_state as _ostate
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin
    from goldfive.adapters._adk_state_protocol import KEY_GOALS_SUMMARY
    from goldfive.types import Goal

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship the widget")],
    )
    _ostate.refresh_goals_summary(session.state, session.goals)

    task = Task(id="t1", title="x")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = type("A", (), {"name": "agent_a"})()

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    adk_summary = inv_ctx.session.state[KEY_GOALS_SUMMARY]
    assert "g1" in adk_summary
    assert "ship the widget" in adk_summary


async def test_bridge_cancelled_function_call_ids_copies_to_adk_state() -> None:
    """``goldfive.cancelled_function_call_ids`` crosses the bridge."""
    from goldfive import orchestration_state as _ostate
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin
    from goldfive.adapters._adk_state_protocol import (
        KEY_CANCELLED_FUNCTION_CALL_IDS,
    )

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(run_id="r1")
    _ostate.append_cancelled_function_call_ids(session.state, ["fc-a", "fc-b"])

    task = Task(id="t1", title="x")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = type("A", (), {"name": "agent_a"})()

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    assert inv_ctx.session.state[KEY_CANCELLED_FUNCTION_CALL_IDS] == ["fc-a", "fc-b"]


async def test_bridge_clears_stale_active_steer_when_orchestration_state_empty() -> None:
    """An earlier bridged write does NOT linger across a subsequent
    invocation whose orchestration-state has no active steer.
    """
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin
    from goldfive.adapters._adk_state_protocol import (
        KEY_ACTIVE_STEER_AT_TURN,
        KEY_ACTIVE_STEER_BODY,
    )

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(run_id="r1")
    # No orchestration-level active steer.

    task = Task(id="t1", title="x")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            # Start with a stale active_steer on ADK state (e.g. from an
            # earlier turn that has since been cleared).
            self.state: dict[str, Any] = {
                KEY_ACTIVE_STEER_BODY: "stale steer",
                KEY_ACTIVE_STEER_AT_TURN: 3,
            }

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = type("A", (), {"name": "agent_a"})()

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    # Both keys were cleared because orchestration-state had no active steer.
    assert KEY_ACTIVE_STEER_BODY not in inv_ctx.session.state
    assert KEY_ACTIVE_STEER_AT_TURN not in inv_ctx.session.state


async def test_bridge_preserves_legacy_state_keys_regression() -> None:
    """Regression: adding the orchestration bridge MUST NOT regress the
    legacy-key writes (run_id, plan context, current_task, tools).
    """
    from goldfive import orchestration_state as _ostate
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin
    from goldfive.adapters._adk_state_protocol import (
        KEY_ACTIVE_STEER_BODY,
        KEY_CURRENT_TASK_ID,
        KEY_CURRENT_TASK_TITLE,
        KEY_PLAN_ID,
        KEY_RUN_ID,
        KEY_TOOLS_AVAILABLE,
    )

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(
        run_id="run-42",
        plan=Plan(
            id="plan-42",
            run_id="run-42",
            goal_ids=[],
            tasks=[Task(id="t1", title="xyz")],
            edges=[],
            summary="summary",
        ),
    )
    _ostate.set_active_steer(session.state, body="steer it", at_turn=1)

    task = Task(id="t1", title="xyz")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={"report_task_started": lambda *a, **kw: None},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = type("A", (), {"name": "agent_a"})()

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    # Legacy keys still land.
    assert inv_ctx.session.state[KEY_RUN_ID] == "run-42"
    assert inv_ctx.session.state[KEY_PLAN_ID] == "plan-42"
    assert inv_ctx.session.state[KEY_CURRENT_TASK_ID] == "t1"
    assert inv_ctx.session.state[KEY_CURRENT_TASK_TITLE] == "xyz"
    assert "report_task_started" in (inv_ctx.session.state[KEY_TOOLS_AVAILABLE] or [])
    # Bridged key lands too.
    assert inv_ctx.session.state[KEY_ACTIVE_STEER_BODY] == "steer it"


async def test_bridge_active_steer_visible_in_sub_runner() -> None:
    """Steer body bridged via DefaultSteerer is visible on the sub-Runner
    session.state during its own ``before_model_callback`` — i.e. the
    bridge runs automatically across AgentTool boundaries via each
    sub-Runner's ``before_run_callback``.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive import orchestration_state as _ostate
    from goldfive.adapters._adk_state_protocol import KEY_ACTIVE_STEER_BODY
    from goldfive.adapters.adk import ADKAdapter

    observed_steer: dict[str, Any] = {}

    def _snapshot_steer(callback_context: Any, llm_request: Any) -> None:  # noqa: ARG001
        inv_ctx = getattr(callback_context, "_invocation_context", None)
        state = None
        if inv_ctx is not None:
            sess = getattr(inv_ctx, "session", None)
            state = getattr(sess, "state", None)
        if state is None:
            return None
        # Only snapshot the sub-Runner's first before_model.
        if observed_steer:
            return None
        try:
            observed_steer["body"] = state.get(KEY_ACTIVE_STEER_BODY)
            observed_steer["agent_name"] = getattr(
                getattr(inv_ctx, "agent", None), "name", ""
            )
        except Exception:
            observed_steer["body"] = None
        return None

    agent_b = Agent(
        name="agent_b",
        model=_make_agent_b_llm()(),
        instruction="",
        before_model_callback=_snapshot_steer,
    )
    agent_a = Agent(
        name="agent_a",
        model=_make_agent_a("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )
    adapter = ADKAdapter(agent_a)
    adapter.bind_steerer(_StubSteerer())

    task = Task(id="t1", title="compound", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    # Fire an orchestration-level active steer before invoking. The
    # plugin's before_run_callback will bridge this onto BOTH the top
    # invocation's session AND the sub-Runner's session.
    _ostate.set_active_steer(
        session.state,
        body="bridge this to sub-runner",
        at_turn=1,
    )
    await adapter.invoke(task=task, session=session)

    assert observed_steer, (
        "sub-Runner's before_model_callback never ran — AgentTool dispatch broke"
    )
    assert observed_steer.get("body") == "bridge this to sub-runner", (
        f"sub-Runner saw active_steer.body={observed_steer.get('body')!r}; "
        "orchestration-state bridge did not propagate through AgentTool."
    )
    # Sanity: the sub-Runner observation came from agent_b's callback
    # (not agent_a's), so this is genuinely the nested sub-Runner.
    assert observed_steer.get("agent_name") == "agent_b"


async def test_bridge_end_to_end_steerer_to_goldfive_planner_instruction() -> None:
    """Fire USER_STEER through DefaultSteerer; start an ADK invocation;
    assert GoldfivePlanner's injected instruction contains the steer
    body (not ``(none)``).

    This is the highest-level assertion for #170: the data path from
    operator-authored steer → DefaultSteerer._apply_user_steer_state
    → goldfive.Session.state → bridge (before_run_callback) → ADK
    session.state → GoldfivePlanner.build_planning_instruction.
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.control import ControlKind, ControlMessage
    from goldfive.steerer import DefaultSteerer

    # Capture the system_instruction the LLM sees so we can assert
    # GoldfivePlanner's orchestration block landed with the real
    # steer body rather than ``(none)``.
    captured_instructions: list[str] = []

    class _ScriptedLLM(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            # Pull the system instruction for assertion. ADK stashes it
            # under ``config.system_instruction``.
            config = getattr(llm_request, "config", None)
            si = getattr(config, "system_instruction", "") if config is not None else ""
            if isinstance(si, str):
                captured_instructions.append(si)
            else:
                # Some ADK versions wrap it in a Content or list of parts.
                captured_instructions.append(str(si))
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                turn_complete=True,
            )

    agent = Agent(
        name="agent_a",
        model=_ScriptedLLM(),
        instruction="",
    )
    adapter = ADKAdapter(agent)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)  # type: ignore[arg-type]
    steerer.bind_adapter(adapter)
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="ship it", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)

    # Fire a USER_STEER via the steerer — same path a control-channel
    # STEER takes in production. This writes the active_steer onto
    # goldfive.Session.state via _apply_user_steer_state.
    await steerer.observe(
        ControlMessage(
            kind=ControlKind.STEER,
            payload={"note": "pivot toward reliability"},
        ),
        session,
    )

    await adapter.invoke(task=task, session=session)

    assert captured_instructions, (
        "LLM generate_content_async never ran — invocation did not dispatch"
    )
    first = captured_instructions[0]
    assert "[GOLDFIVE ORCHESTRATION CONTEXT]" in first, (
        "GoldfivePlanner's orchestration block missing from the system instruction"
    )
    assert "pivot toward reliability" in first, (
        f"Steer body missing from injected instruction; saw:\n{first}"
    )
    assert "Active user steer (if any): (none)" not in first, (
        "Active steer rendered as (none) — bridge did not propagate the body"
    )
