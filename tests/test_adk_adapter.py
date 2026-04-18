"""Tests for :mod:`goldfive.adapters.adk`.

Skipped entirely when ``google.adk`` is not installed (optional dep).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.reporting import ReportingToolSpec
from goldfive.types import Plan, Session, Task

# ---------------------------------------------------------------------------
# Optional-import guard
# ---------------------------------------------------------------------------


def test_optional_import_guard_message() -> None:
    """When ADK IS installed the module must import cleanly."""
    import importlib

    mod = importlib.import_module("goldfive.adapters.adk")
    assert hasattr(mod, "ADKAdapter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> Any:
    """Construct a bare ADK ``LlmAgent`` for plugin wiring tests.

    The tests never actually drive an LLM turn — they exercise tool
    registration and plugin routing via direct callback invocation.
    """
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name="test_agent",
        model="fake-model",
        description="Test agent",
        instruction="Test.",
    )


@pytest.fixture
def state_ctx_cls():
    """Factory for a minimal ADK-callback-context stub with a state dict."""

    class _Ctx:
        class _Session:
            def __init__(self, state: dict) -> None:
                self.state = state

        def __init__(self, state: dict) -> None:
            self._state = state

        @property
        def session(self) -> Any:
            return _Ctx._Session(self._state)

    return _Ctx


class _RecordingSteerer:
    """Minimal async-capable steerer stub for plugin observation tests."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def observe(self, event: Any, session: Any) -> None:
        self.events.append(event)

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, event: Any, session: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Reporting-tool registration + routing
# ---------------------------------------------------------------------------


async def test_reporting_tools_registered_on_root_agent() -> None:
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    adapter = ADKAdapter(agent)

    async def handler(args, session, steerer):
        return {"acknowledged": True, "echo": dict(args)}

    spec = ReportingToolSpec(
        name="report_task_started",
        description="Mark a task as started.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "detail": {"type": "string"},
            },
        },
        handler=handler,
    )

    await adapter.register_reporting_tools([spec])

    names = [
        getattr(t, "name", None)
        or getattr(getattr(t, "func", None), "__name__", None)
        for t in getattr(agent, "tools", None) or ()
    ]
    assert "report_task_started" in names


async def test_plugin_routes_reporting_tool_through_handler(state_ctx_cls) -> None:
    """before_tool_callback intercepts the tool and returns the handler result."""
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
    )
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    adapter = ADKAdapter(agent)

    invocations: list[dict] = []

    async def handler(args, session, steerer):
        invocations.append(dict(args))
        return {"acknowledged": True, "routed": True}

    spec = ReportingToolSpec(
        name="report_task_completed",
        description="Mark a task completed.",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    await adapter.register_reporting_tools([spec])

    session = Session(run_id="run-1")
    task = Task(id="t1", title="Do the thing")
    state_dict: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=None,
            task=task,
            tool_handlers=adapter._tool_handlers,
            host_agent_name="test_agent",
        )
    }

    class _Tool:
        name = "report_task_completed"

    result = await adapter._plugin.before_tool_callback(
        tool=_Tool(),
        tool_args={"task_id": "t1", "summary": "done"},
        tool_context=state_ctx_cls(state_dict),
    )

    assert invocations == [{"task_id": "t1", "summary": "done"}]
    assert isinstance(result, dict)
    assert result.get("routed") is True


async def test_non_reporting_tool_passes_through(state_ctx_cls) -> None:
    """The plugin returns None for tools it doesn't recognize."""
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=Session(run_id="r"),
            steerer=None,
            task=Task(id="t", title="x"),
            tool_handlers={},
            host_agent_name="test_agent",
        )
    }

    class _Tool:
        name = "web_search"

    result = await plugin.before_tool_callback(
        tool=_Tool(),
        tool_args={"q": "hello"},
        tool_context=state_ctx_cls(state),
    )
    assert result is None


# ---------------------------------------------------------------------------
# State protocol writes
# ---------------------------------------------------------------------------


async def test_before_model_writes_goldfive_state_keys(state_ctx_cls) -> None:
    """before_model_callback seeds goldfive.* keys for the agent to read."""
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )
    from goldfive.adapters._adk_state_protocol import (
        KEY_CURRENT_TASK_ID,
        KEY_CURRENT_TASK_TITLE,
        KEY_PLAN_ID,
        KEY_RUN_ID,
        KEY_TOOLS_AVAILABLE,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(
        run_id="run-abc",
        plan=Plan(
            id="plan-1",
            run_id="run-abc",
            goal_ids=[],
            tasks=[Task(id="t1", title="Alpha", assignee_agent_id="worker")],
            edges=[],
            summary="test plan",
        ),
        completed_results={},
    )
    task = Task(id="t1", title="Alpha")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=None,
            task=task,
            tool_handlers={"report_task_started": lambda *a, **kw: None},
            host_agent_name="test_agent",
        )
    }

    await plugin.before_model_callback(
        callback_context=state_ctx_cls(state), llm_request=None
    )

    assert state.get(KEY_RUN_ID) == "run-abc"
    assert state.get(KEY_PLAN_ID) == "plan-1"
    assert state.get(KEY_CURRENT_TASK_ID) == "t1"
    assert state.get(KEY_CURRENT_TASK_TITLE) == "Alpha"
    assert "report_task_started" in (state.get(KEY_TOOLS_AVAILABLE) or [])


# ---------------------------------------------------------------------------
# Drift observation
# ---------------------------------------------------------------------------


async def test_on_tool_error_calls_steerer_observe(state_ctx_cls) -> None:
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    steerer = _RecordingSteerer()
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=Session(run_id="run-err"),
            steerer=steerer,
            task=Task(id="t9", title="Broken step"),
            tool_handlers={},
            host_agent_name="test_agent",
        )
    }

    class _Tool:
        name = "web_search"

    await plugin.on_tool_error_callback(
        tool=_Tool(),
        tool_args={"q": "x"},
        tool_context=state_ctx_cls(state),
        error=RuntimeError("upstream 500"),
    )

    assert len(steerer.events) == 1
    evt = steerer.events[0]
    assert evt["kind"] == "tool_error"
    assert "web_search" in evt["detail"]


async def test_on_event_transfer_calls_steerer_observe(state_ctx_cls) -> None:
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    steerer = _RecordingSteerer()
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=Session(run_id="run-xfer"),
            steerer=steerer,
            task=Task(id="t1", title="Handoff"),
            tool_handlers={},
            host_agent_name="test_agent",
        )
    }

    class _Actions:
        transfer_to_agent = "other_agent"
        escalate = False

    class _Event:
        actions = _Actions()

    await plugin.on_event_callback(
        invocation_context=state_ctx_cls(state), event=_Event()
    )

    assert len(steerer.events) == 1
    assert steerer.events[0]["kind"] == "agent_transfer"
    assert "other_agent" in steerer.events[0]["detail"]


# ---------------------------------------------------------------------------
# AgentAdapter protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_conforms_to_protocol() -> None:
    """ADKAdapter must satisfy the AgentAdapter Protocol."""
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.protocols import AgentAdapter

    adapter = ADKAdapter(_make_agent())
    assert isinstance(adapter, AgentAdapter)
    # available_agents includes the root agent name at minimum.
    assert "test_agent" in adapter.available_agents


# ---------------------------------------------------------------------------
# Subtree propagation — wrapping the root coordinates every sub-agent
# ---------------------------------------------------------------------------


def _tool_names(agent: Any) -> set[str]:
    names: set[str] = set()
    for t in getattr(agent, "tools", None) or ():
        n = getattr(t, "name", None) or getattr(
            getattr(t, "func", None), "__name__", None
        )
        if n:
            names.add(str(n))
    return names


async def test_register_reporting_tools_propagates_across_three_level_tree() -> None:
    """Wrapping the root agent must attach reporting tools to every descendant.

    Tree shape:

        root
        ├── child_a           (via sub_agents)
        │   └── grandchild_a  (via sub_agents)
        └── child_b_as_tool   (via AgentTool in root.tools)
            └── grandchild_b  (via sub_agents)
    """
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore
    from google.adk.tools.agent_tool import AgentTool  # type: ignore

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS, REPORTING_TOOL_NAMES

    def _mk(name: str) -> Any:
        return LlmAgent(
            name=name, model="fake-model", description=name, instruction="x"
        )

    grandchild_a = _mk("grandchild_a")
    grandchild_b = _mk("grandchild_b")
    child_a = _mk("child_a")
    child_a.sub_agents = [grandchild_a]
    child_b_as_tool = _mk("child_b_as_tool")
    child_b_as_tool.sub_agents = [grandchild_b]

    root = _mk("root")
    root.sub_agents = [child_a]
    root.tools = [AgentTool(agent=child_b_as_tool)]

    adapter = ADKAdapter(root)
    await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))

    expected = set(REPORTING_TOOL_NAMES)
    for agent in (root, child_a, grandchild_a, child_b_as_tool, grandchild_b):
        have = _tool_names(agent)
        missing = expected - have
        assert not missing, (
            f"agent {agent.name!r} missing reporting tools: {sorted(missing)}"
        )

    # available_agents discovers every node in the tree.
    discovered = set(adapter.available_agents)
    assert {
        "root",
        "child_a",
        "grandchild_a",
        "child_b_as_tool",
        "grandchild_b",
    } <= discovered


async def test_register_reporting_tools_is_idempotent() -> None:
    """Registering twice must not duplicate the reporting tools on any agent."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS, REPORTING_TOOL_NAMES

    child = LlmAgent(
        name="child", model="fake-model", description="c", instruction="x"
    )
    root = LlmAgent(
        name="root", model="fake-model", description="r", instruction="x"
    )
    root.sub_agents = [child]

    adapter = ADKAdapter(root)
    await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))
    await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))

    for agent in (root, child):
        names = [
            getattr(t, "name", None)
            or getattr(getattr(t, "func", None), "__name__", None)
            for t in getattr(agent, "tools", None) or ()
        ]
        for reporting_name in REPORTING_TOOL_NAMES:
            assert names.count(reporting_name) == 1, (
                f"{agent.name}: {reporting_name} registered "
                f"{names.count(reporting_name)} times"
            )
