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
