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
        getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", None)
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

    await plugin.before_model_callback(callback_context=state_ctx_cls(state), llm_request=None)

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


async def test_after_run_emits_confabulation_risk_on_zero_tools(state_ctx_cls) -> None:
    """Research-shaped task + non-empty final text + 0 tool calls → INFO drift.

    Plugin-level integration test for issue #128: drives the plugin's
    before_run → after_model (no tool calls in the llm response) →
    after_run callback chain by hand and asserts that
    CONFABULATION_RISK fires through the same path AGENT_REFUSAL uses.
    """
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity

    plugin = make_adk_plugin(host_agent_name="research_agent")

    class _ConfabulationAwareSteerer(_RecordingSteerer):
        """Steerer stub that records drift emitted via _handle_drift."""

        def __init__(self) -> None:
            super().__init__()
            self.drifts: list[DriftEvent] = []

        async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
            self.drifts.append(drift)

    steerer = _ConfabulationAwareSteerer()
    session = Session(run_id="run-confab")
    task = Task(
        id="t1",
        title="Research the latest exchange rates",
        description="Look up current USD/EUR and USD/JPY rates.",
        assignee_agent_id="research_agent",
    )
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=task,
            tool_handlers={},
            host_agent_name="research_agent",
        )
    }

    class _InvContext:
        def __init__(self, invocation_id: str) -> None:
            self.invocation_id = invocation_id
            self.session = state_ctx_cls(state).session
            self.agent = type("_A", (), {"name": "research_agent"})()

    class _CbContext:
        """Minimal callback-context that also exposes ``_invocation_context``.

        The after_model_callback path reads ``_invocation_context`` off
        the callback context to attribute the turn to an invocation_id;
        without it, the plugin cannot correlate tool-call counts to the
        invocation in after_run.
        """

        def __init__(self, state: dict, invocation_id: str) -> None:
            self._state = state
            self._invocation_context = _InvContext(invocation_id)

        @property
        def session(self):
            return self._invocation_context.session

    # LLM response shape: non-empty text, NO function calls.
    class _Part:
        def __init__(self, text: str = "", thought: bool = False) -> None:
            self.text = text
            self.thought = thought
            self.function_call = None

    class _Content:
        def __init__(self, parts: list[_Part]) -> None:
            self.parts = parts

    class _LlmResponse:
        def __init__(self, text: str) -> None:
            self.content = _Content([_Part(text=text)])
            self.finish_reason = None

    inv_id = "inv-research-1"

    # Drive the full lifecycle.
    await plugin.before_run_callback(invocation_context=_InvContext(inv_id))
    await plugin.after_model_callback(
        callback_context=_CbContext(state, inv_id),
        llm_response=_LlmResponse("The current USD/EUR rate is 1.08 and USD/JPY is 151.4."),
    )
    await plugin.after_run_callback(invocation_context=_InvContext(inv_id))

    assert len(steerer.drifts) == 1, (
        f"expected exactly one CONFABULATION_RISK drift, got {steerer.drifts!r}"
    )
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.CONFABULATION_RISK
    assert drift.severity is DriftSeverity.INFO
    assert drift.current_task_id == "t1"
    assert drift.current_agent_id == "research_agent"


async def test_after_run_no_confabulation_when_tool_was_called(state_ctx_cls) -> None:
    """Research task that called a tool → no CONFABULATION_RISK drift.

    Counterpart to ``test_after_run_emits_confabulation_risk_on_zero_tools``:
    the tool call in after_model bumps the per-invocation counter above
    zero, so the after_run check falls through silently.
    """
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )
    from goldfive.types import DriftEvent

    plugin = make_adk_plugin(host_agent_name="research_agent")

    class _Steerer(_RecordingSteerer):
        def __init__(self) -> None:
            super().__init__()
            self.drifts: list[DriftEvent] = []

        async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
            self.drifts.append(drift)

    steerer = _Steerer()
    session = Session(run_id="run-ok")
    task = Task(
        id="t2",
        title="Research the latest exchange rates",
        description="",
        assignee_agent_id="research_agent",
    )
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=task,
            tool_handlers={},
            host_agent_name="research_agent",
        )
    }

    class _InvContext:
        def __init__(self, invocation_id: str) -> None:
            self.invocation_id = invocation_id
            self.session = state_ctx_cls(state).session
            self.agent = type("_A", (), {"name": "research_agent"})()

    class _CbContext:
        def __init__(self, state: dict, invocation_id: str) -> None:
            self._state = state
            self._invocation_context = _InvContext(invocation_id)

        @property
        def session(self):
            return self._invocation_context.session

    class _FunctionCall:
        def __init__(self, name: str) -> None:
            self.name = name
            self.args = {"q": "USD EUR rate"}

    class _Part:
        def __init__(
            self,
            text: str = "",
            thought: bool = False,
            function_call: Any = None,
        ) -> None:
            self.text = text
            self.thought = thought
            self.function_call = function_call

    class _Content:
        def __init__(self, parts: list[_Part]) -> None:
            self.parts = parts

    class _LlmResponse:
        def __init__(self, parts: list[_Part]) -> None:
            self.content = _Content(parts)
            self.finish_reason = None

    inv_id = "inv-ok-1"

    await plugin.before_run_callback(invocation_context=_InvContext(inv_id))
    # Turn 1: the model calls a web_search tool.
    await plugin.after_model_callback(
        callback_context=_CbContext(state, inv_id),
        llm_response=_LlmResponse([_Part(function_call=_FunctionCall("web_search"))]),
    )
    # Turn 2: the model returns its final answer as text.
    await plugin.after_model_callback(
        callback_context=_CbContext(state, inv_id),
        llm_response=_LlmResponse(
            [_Part(text="The USD/EUR rate is 1.08 according to the search results.")]
        ),
    )
    await plugin.after_run_callback(invocation_context=_InvContext(inv_id))

    # No CONFABULATION_RISK drift — a tool was called so the pattern is
    # exactly the expected shape for a research task.
    assert steerer.drifts == [], f"no drift expected when a tool was called; got {steerer.drifts!r}"


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

    await plugin.on_event_callback(invocation_context=state_ctx_cls(state), event=_Event())

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
        n = getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", None)
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
        return LlmAgent(name=name, model="fake-model", description=name, instruction="x")

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
        assert not missing, f"agent {agent.name!r} missing reporting tools: {sorted(missing)}"

    # available_agents discovers every node in the tree.
    discovered = set(adapter.available_agents)
    assert {
        "root",
        "child_a",
        "grandchild_a",
        "child_b_as_tool",
        "grandchild_b",
    } <= discovered


# ---------------------------------------------------------------------------
# add_plugin — post-construction plugin install
# ---------------------------------------------------------------------------


async def test_add_plugin_installs_on_inner_runner() -> None:
    """ADKAdapter.add_plugin must attach the plugin to the InMemoryRunner."""
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore

    from goldfive.adapters.adk import ADKAdapter

    class _FakePlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="fake_plugin")

    adapter = ADKAdapter(_make_agent())
    plugin = _FakePlugin()
    adapter.add_plugin(plugin)

    installed = list(getattr(adapter._runner.plugin_manager, "plugins", []))
    assert plugin in installed


async def test_add_plugin_no_op_when_runner_lacks_plugin_manager() -> None:
    """add_plugin must not raise when the inner runner has no plugin support."""
    from goldfive.adapters.adk import ADKAdapter

    # Build the adapter normally, then swap in a runner stub with no
    # plugin_manager. ADKAdapter.add_plugin should DEBUG-log + no-op.
    adapter = ADKAdapter(_make_agent())

    class _BareRunner:
        agent = adapter._agent
        run_async = adapter._runner.run_async
        session_service = adapter._runner.session_service

    adapter._runner = _BareRunner()
    adapter.add_plugin(object())  # must not raise


async def test_goldfive_adk_agent_add_plugin_delegates() -> None:
    """GoldfiveADKAgent.add_plugin routes through ADKAdapter on the inner Runner."""
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore

    import goldfive
    from goldfive.adapters.adk_wrap import GoldfiveADKAgent

    class _FakePlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="wrap_fake_plugin")

    wrapped = goldfive.wrap(_make_agent())
    assert isinstance(wrapped, GoldfiveADKAgent)
    plugin = _FakePlugin()
    wrapped.add_plugin(plugin)

    adapter = wrapped._runner.agent
    installed = list(getattr(adapter._runner.plugin_manager, "plugins", []))
    assert plugin in installed


async def test_invoke_breaks_when_task_reported_terminal_mid_stream() -> None:
    """Adapter must stop driving the ADK runner once the agent has
    reported the current task as terminal.

    Without this, the ADK generator keeps running — letting the agent
    take more LLM turns on an already-done task and burn through ADK's
    500-LLM-call ceiling reporting redundant status. The fix checks
    ``session.plan.tasks[task_id].status`` after each streamed event
    and exits the ``async for`` loop as soon as it's terminal.
    """
    from dataclasses import dataclass, field

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task, TaskStatus

    @dataclass
    class _Event:
        # ADK duck-types: no is_final_response → _is_final_event returns False.
        marker: int = 0
        content: Any = None

    # A run_async that yields 5 events. On event #2, a "tool call" flips
    # the task status to FAILED. The adapter should break at event #2 —
    # events #3, #4, #5 must never be observed.
    observed: list[int] = []

    @dataclass
    class _FakeRunner:
        session_service: Any = None
        plugin_manager: Any = field(default=None)

        async def run_async(self, **kwargs: Any):  # noqa: ARG002
            for i in range(5):
                observed.append(i)
                if i == 2:
                    # Simulate a reporting-tool handler marking the task terminal.
                    session.plan.tasks[0].status = TaskStatus.FAILED
                yield _Event(marker=i)

    task = Task(id="t1", title="do the thing")
    session = Session(
        run_id="r1",
        goals=[],
        plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )

    adapter = ADKAdapter(_make_agent())
    # Single-Runner model: one runner drives everything; monkey-patch
    # it directly.
    adapter._runner = _FakeRunner()
    adapter._session_id = "stub-session"

    result = await adapter.invoke(task=task, session=session)

    # Events 0, 1, 2 should be observed; 3 and 4 must not be.
    assert observed == [0, 1, 2], (
        f"adapter should have broken after event #2 (terminal transition); observed {observed}"
    )
    assert result.stop_reason == "task_terminal"
    assert result.task_id == "t1"


async def test_invoke_runs_to_completion_when_task_stays_non_terminal() -> None:
    """If the task is never reported terminal, the adapter drains all
    events — the new break must not short-circuit normal runs.
    """
    from dataclasses import dataclass, field

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    @dataclass
    class _Event:
        marker: int = 0
        content: Any = None

    observed: list[int] = []

    @dataclass
    class _FakeRunner:
        session_service: Any = None
        plugin_manager: Any = field(default=None)

        async def run_async(self, **kwargs: Any):  # noqa: ARG002
            for i in range(3):
                observed.append(i)
                yield _Event(marker=i)

    task = Task(id="t2", title="normal run")
    session = Session(
        run_id="r1",
        goals=[],
        plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )

    adapter = ADKAdapter(_make_agent())
    # Single-Runner model: monkey-patch the one runner.
    adapter._runner = _FakeRunner()
    adapter._session_id = "stub-session"

    await adapter.invoke(task=task, session=session)
    assert observed == [0, 1, 2]


async def test_register_reporting_tools_is_idempotent() -> None:
    """Registering twice must not duplicate the reporting tools on any agent."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS, REPORTING_TOOL_NAMES

    child = LlmAgent(name="child", model="fake-model", description="c", instruction="x")
    root = LlmAgent(name="root", model="fake-model", description="r", instruction="x")
    root.sub_agents = [child]

    adapter = ADKAdapter(root)
    await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))
    await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))

    for agent in (root, child):
        names = [
            getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", None)
            for t in getattr(agent, "tools", None) or ()
        ]
        for reporting_name in REPORTING_TOOL_NAMES:
            assert names.count(reporting_name) == 1, (
                f"{agent.name}: {reporting_name} registered {names.count(reporting_name)} times"
            )


# ---------------------------------------------------------------------------
# Mid-invocation cancel — heal ADK session history for orphan tool_call_ids
# (TASK-LIFECYCLE.md §7.4). See goldfive.adapters.adk._heal_pending_tool_calls.
# ---------------------------------------------------------------------------


class _FakeADKSession:
    """Minimal stand-in for ``google.adk.sessions.Session`` for heal tests.

    Only the attributes ``events`` and ``state`` that
    :meth:`BaseSessionService.append_event` needs are provided.
    """

    def __init__(self) -> None:
        self.events: list = []
        self.state: dict = {}


class _FakeSessionService:
    """Records ``append_event`` calls so heal-path tests can assert on them."""

    def __init__(self) -> None:
        self._session = _FakeADKSession()
        self.appended: list = []

    async def create_session(self, **_kwargs) -> _FakeADKSession:
        return self._session

    async def get_session(self, **_kwargs) -> _FakeADKSession:
        return self._session

    async def append_event(self, *, session, event):
        self.appended.append(event)
        session.events.append(event)
        return event


class _FakeRunner:
    """Runner stub whose ``run_async`` yields a scripted sequence of events.

    ``events_factory(cancel_event)`` is a zero-arg callable returning an async
    generator (or an async function). The fixture tests pass a coroutine-like
    factory that yields function_call events, optionally pauses on a
    :class:`asyncio.Event`, and can raise :class:`asyncio.CancelledError` to
    simulate the sequential executor's mid-invocation cancel.
    """

    def __init__(self, run_async_impl, agent) -> None:
        self._run_async_impl = run_async_impl
        self.agent = agent
        self.app_name = getattr(agent, "name", "fake-app")
        self.session_service = _FakeSessionService()
        self.plugin_manager = None
        self.plugins: list = []

    async def run_async(self, **kwargs):  # noqa: ARG002 — runner signature
        async for event in self._run_async_impl():
            yield event


def _mk_function_call_event(*, call_id: str, name: str, invocation_id: str = "inv-1"):
    """Build an ADK ``Event`` carrying a single ``function_call`` part."""
    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    part = types.Part(function_call=types.FunctionCall(id=call_id, name=name))
    return Event(
        invocation_id=invocation_id,
        author="test_agent",
        content=types.Content(role="model", parts=[part]),
    )


def _mk_function_response_event(*, call_id: str, name: str, invocation_id: str = "inv-1"):
    """Build an ADK ``Event`` carrying a matching ``function_response`` part."""
    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    part = types.Part.from_function_response(name=name, response={"ok": True})
    part.function_response.id = call_id
    return Event(
        invocation_id=invocation_id,
        author="test_agent",
        content=types.Content(role="user", parts=[part]),
    )


async def test_cancel_mid_tool_call_heals_session() -> None:
    """Cancellation mid-tool-round-trip must synthesize a function_response.

    Scenario: runner yields a function_call event for ``call-1`` then the
    task is cancelled (simulating SequentialExecutor._cancel_invoke_task on
    a USER_STEER). ADKAdapter must append a synthetic function_response
    event whose ``id`` matches ``call-1`` so the next invoke() doesn't hit
    "Missing tool results for tool_call_id".
    """
    import asyncio

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    async def _run():
        yield _mk_function_call_event(call_id="call-1", name="search")
        # Hang forever; the outer task.cancel() will inject CancelledError.
        await asyncio.Event().wait()
        # Unreachable.
        yield None  # pragma: no cover

    runner = _FakeRunner(_run, agent)
    adapter = ADKAdapter(runner, session_id="sess-1")

    invoke_task = asyncio.create_task(adapter.invoke(Task(id="t1", title="x"), Session(run_id="r")))
    # Give the runner a moment to emit the function_call event and start
    # awaiting the never-set asyncio.Event, then cancel.
    await asyncio.sleep(0.01)
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    appended = runner.session_service.appended
    assert len(appended) == 1, f"expected 1 synthetic response; got {len(appended)}"
    synth = appended[0]
    responses = synth.get_function_responses()
    assert len(responses) == 1
    assert responses[0].id == "call-1"
    # Reason-differentiated content (goldfive#139). A bare
    # task.cancel() without a steerer-set _next_cancel_reason falls
    # through to the generic content variant — LLM-actionable, not
    # goldfive-internal jargon.
    payload = responses[0].response
    assert payload.get("status") == "cancelled"
    assert "Do NOT retry" in payload.get("instruction", "")


async def test_multiple_pending_tool_calls_all_healed() -> None:
    """When multiple function_calls are outstanding at cancel time, all heal.

    Three parallel function_calls are yielded on a single "model turn" event.
    Cancelling before any response arrives must produce three synthetic
    function_response events, one per id.
    """
    import asyncio

    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    # Single event carrying three parallel function_call parts, mimicking
    # ADK's "parallel tool call" shape.
    multi_parts = [
        types.Part(function_call=types.FunctionCall(id=f"call-{i}", name=f"t{i}"))
        for i in (1, 2, 3)
    ]
    multi_event = Event(
        invocation_id="inv-multi",
        author="test_agent",
        content=types.Content(role="model", parts=multi_parts),
    )

    async def _run():
        yield multi_event
        await asyncio.Event().wait()
        yield None  # pragma: no cover

    runner = _FakeRunner(_run, agent)
    adapter = ADKAdapter(runner, session_id="sess-multi")

    invoke_task = asyncio.create_task(adapter.invoke(Task(id="t1", title="x"), Session(run_id="r")))
    await asyncio.sleep(0.01)
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    appended = runner.session_service.appended
    healed_ids = set()
    for ev in appended:
        for fr in ev.get_function_responses():
            healed_ids.add(fr.id)
    assert healed_ids == {"call-1", "call-2", "call-3"}, (
        f"expected all three orphan ids healed; got {healed_ids}"
    )


async def test_heal_is_noop_when_no_pending_calls() -> None:
    """Clean exit with no orphan function_calls appends nothing.

    Also covers the case where every emitted function_call received a
    matching function_response before the stream ended — _pending_tool_call_ids
    should be empty at exit and no synthetic event should be appended.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    async def _run_clean():
        # Paired call/response, then quiet exit.
        yield _mk_function_call_event(call_id="call-ok", name="search")
        yield _mk_function_response_event(call_id="call-ok", name="search")

    runner_clean = _FakeRunner(_run_clean, agent)
    adapter_clean = ADKAdapter(runner_clean, session_id="sess-clean")
    result = await adapter_clean.invoke(Task(id="t1", title="x"), Session(run_id="r"))
    assert runner_clean.session_service.appended == [], (
        "healed-path should not fire on a clean exit"
    )
    assert result.stop_reason != "cancelled"

    # Also: a truly empty stream (no function_calls at all) — heal must no-op.
    async def _run_empty():
        if False:  # pragma: no cover — never yields
            yield None
        return

    runner_empty = _FakeRunner(_run_empty, agent)
    adapter_empty = ADKAdapter(runner_empty, session_id="sess-empty")
    await adapter_empty.invoke(Task(id="t1", title="x"), Session(run_id="r"))
    assert runner_empty.session_service.appended == []


# ---------------------------------------------------------------------------
# goldfive#139 — reason-differentiated synthetic response content +
# USER_STEER user-role primer event. The prior generic content shape
# ("goldfive_cancelled" / "preserve session history well-formedness")
# was written for human log viewers and confused Qwen + other smaller
# LLMs reading it on the next turn, triggering retry loops and
# reasoning spirals. These tests pin the new content variants and the
# primer-event append behaviour.
# ---------------------------------------------------------------------------


def test_build_cancelled_response_event_user_steer_content() -> None:
    """``reason=user_steer`` → status/instruction aimed at the next LLM turn.

    The content must carry the "prior task abandoned" instruction and
    not leak goldfive-internal jargon ("goldfive_cancelled",
    "preserve session history well-formedness", etc.) the prior
    generic shape emitted. See goldfive#139.
    """
    from goldfive.adapters.adk import (
        SYMBOLIC_REASON_USER_STEER,
        _build_cancelled_response_event,
    )

    event = _build_cancelled_response_event(
        function_call_id="call-x",
        tool_name="search",
        author="agent",
        invocation_id="inv-1",
        reason=SYMBOLIC_REASON_USER_STEER,
    )
    responses = event.get_function_responses()
    assert len(responses) == 1
    fr = responses[0]
    assert fr.id == "call-x"
    payload = fr.response
    assert payload.get("status") == "cancelled_by_user_steering"
    instruction = payload.get("instruction", "")
    assert "ABANDONED" in instruction
    assert "Do NOT retry" in instruction
    # No goldfive-internal jargon leaks into the LLM-visible payload.
    assert "goldfive_cancelled" not in payload
    assert "session history" not in instruction


def test_build_cancelled_response_event_replan_content() -> None:
    """``reason=replan`` → status/instruction describing a plan update.

    Milder wording than USER_STEER — a replan is expected to reference
    prior context the agent may still need.
    """
    from goldfive.adapters.adk import (
        SYMBOLIC_REASON_REPLAN,
        _build_cancelled_response_event,
    )

    event = _build_cancelled_response_event(
        function_call_id="call-x",
        tool_name="search",
        author="agent",
        invocation_id="inv-1",
        reason=SYMBOLIC_REASON_REPLAN,
    )
    fr = event.get_function_responses()[0]
    assert fr.response.get("status") == "cancelled_by_replan"
    instruction = fr.response.get("instruction", "")
    assert "plan was updated" in instruction.lower()
    assert "Do NOT retry" in instruction


def test_build_cancelled_response_event_generic_content() -> None:
    """Legacy / unknown reasons fall through to the neutral variant."""
    from goldfive.adapters.adk import _build_cancelled_response_event

    for legacy_reason in (
        "cancelled_mid_invocation",
        "error:ConnectionError",
        "unexpected_orphan_on_normal_exit",
        "",
    ):
        event = _build_cancelled_response_event(
            function_call_id="call-x",
            tool_name="search",
            author="agent",
            invocation_id="inv-1",
            reason=legacy_reason,
        )
        fr = event.get_function_responses()[0]
        assert fr.response.get("status") == "cancelled", (
            f"reason {legacy_reason!r} should resolve to generic content"
        )
        assert "Do NOT retry" in fr.response.get("instruction", "")


async def test_user_steer_primer_appended_after_function_response() -> None:
    """USER_STEER cancel with 2 orphan calls → 2 function_response + 1 primer.

    Covers the belt-and-braces primer-event append that reinforces the
    pivot for the next LLM turn. Primer must come AFTER the
    function_response events (ordering matters for ADK's chronological
    replay).
    """
    import asyncio

    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    from goldfive.adapters.adk import SYMBOLIC_REASON_USER_STEER, ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    multi_parts = [
        types.Part(function_call=types.FunctionCall(id=f"call-{i}", name=f"t{i}")) for i in (1, 2)
    ]
    multi_event = Event(
        invocation_id="inv-steer",
        author="test_agent",
        content=types.Content(role="model", parts=multi_parts),
    )

    async def _run():
        yield multi_event
        await asyncio.Event().wait()
        yield None  # pragma: no cover

    runner = _FakeRunner(_run, agent)
    adapter = ADKAdapter(runner, session_id="sess-steer")

    invoke_task = asyncio.create_task(adapter.invoke(Task(id="t1", title="x"), Session(run_id="r")))
    await asyncio.sleep(0.01)
    # Simulate the executor tagging the adapter before triggering cancel.
    adapter._next_cancel_reason = SYMBOLIC_REASON_USER_STEER
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    appended = runner.session_service.appended
    # Exactly 3 events: 2 function_response + 1 user-role primer, in that order.
    assert len(appended) == 3, f"expected 3 events; got {len(appended)}"
    # First two events are function_response pairings for the orphans.
    fr_ids = set()
    for ev in appended[:2]:
        for fr in ev.get_function_responses():
            fr_ids.add(fr.id)
    assert fr_ids == {"call-1", "call-2"}
    # Third event is the user-role primer reinforcing the pivot.
    primer = appended[2]
    assert primer.content is not None
    assert primer.content.role == "user"
    # Primer is plain text, not a function_response.
    assert primer.get_function_responses() == []
    text = "".join((p.text or "") for p in (primer.content.parts or []))
    assert "STEERING NOTICE" in text
    assert "cancelled by user steering" in text
    assert "supersedes" in text.lower()


async def test_replan_cancel_does_not_append_primer() -> None:
    """REPLAN cancel heals orphans but does NOT append a primer event.

    The replan variant is a less-jolting pivot — the function_response
    content is sufficient signal; adding a primer would be noise.
    """
    import asyncio

    from goldfive.adapters.adk import SYMBOLIC_REASON_REPLAN, ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    async def _run():
        yield _mk_function_call_event(call_id="call-r", name="search")
        await asyncio.Event().wait()
        yield None  # pragma: no cover

    runner = _FakeRunner(_run, agent)
    adapter = ADKAdapter(runner, session_id="sess-replan")

    invoke_task = asyncio.create_task(adapter.invoke(Task(id="t1", title="x"), Session(run_id="r")))
    await asyncio.sleep(0.01)
    adapter._next_cancel_reason = SYMBOLIC_REASON_REPLAN
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    appended = runner.session_service.appended
    # Exactly 1 event — the function_response pairing. No primer.
    assert len(appended) == 1
    fr = appended[0].get_function_responses()[0]
    assert fr.response.get("status") == "cancelled_by_replan"


async def test_generic_cancel_does_not_append_primer() -> None:
    """A bare (untagged) cancel also gets no primer event."""
    import asyncio

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    async def _run():
        yield _mk_function_call_event(call_id="call-g", name="search")
        await asyncio.Event().wait()
        yield None  # pragma: no cover

    runner = _FakeRunner(_run, agent)
    adapter = ADKAdapter(runner, session_id="sess-generic")

    invoke_task = asyncio.create_task(adapter.invoke(Task(id="t1", title="x"), Session(run_id="r")))
    await asyncio.sleep(0.01)
    # Do NOT set _next_cancel_reason — simulate the legacy code path
    # where the cancel arrives without a symbolic tag.
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    appended = runner.session_service.appended
    # Exactly 1 event — the function_response pairing. No primer.
    assert len(appended) == 1
    fr = appended[0].get_function_responses()[0]
    assert fr.response.get("status") == "cancelled"


async def test_pending_tool_call_pairing_preserved_under_new_content() -> None:
    """Regression: the function_response.id MUST still match the
    orphan function_call.id across all content variants.

    ADK's ``find_event_by_function_call_id`` pairs on the id, not the
    content; if we ever broke the pairing the next turn would still
    see "Missing tool results for tool_call_id". Pin the invariant
    across USER_STEER, REPLAN, and generic reasons.
    """
    from goldfive.adapters.adk import (
        SYMBOLIC_REASON_REPLAN,
        SYMBOLIC_REASON_USER_STEER,
        _build_cancelled_response_event,
    )

    for reason in (
        SYMBOLIC_REASON_USER_STEER,
        SYMBOLIC_REASON_REPLAN,
        "cancelled_mid_invocation",
        "error:SomeError",
    ):
        event = _build_cancelled_response_event(
            function_call_id="call-paired",
            tool_name="whatever",
            author="agent",
            invocation_id="inv-1",
            reason=reason,
        )
        fr = event.get_function_responses()[0]
        assert fr.id == "call-paired", (
            f"reason {reason!r} broke the function_call/function_response "
            f"pairing — ADK would see an orphan call on next turn"
        )


async def test_next_cancel_reason_cleared_after_consumption() -> None:
    """After a cancel fires, ``_next_cancel_reason`` is cleared.

    A stale tag would bleed into the NEXT cancel (of a different
    invocation), causing the wrong content variant to be emitted.
    """
    import asyncio

    from goldfive.adapters.adk import SYMBOLIC_REASON_USER_STEER, ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    async def _run():
        yield _mk_function_call_event(call_id="call-c", name="search")
        await asyncio.Event().wait()
        yield None  # pragma: no cover

    runner = _FakeRunner(_run, agent)
    adapter = ADKAdapter(runner, session_id="sess-clear")

    invoke_task = asyncio.create_task(adapter.invoke(Task(id="t1", title="x"), Session(run_id="r")))
    await asyncio.sleep(0.01)
    adapter._next_cancel_reason = SYMBOLIC_REASON_USER_STEER
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    # Tag was consumed during _heal_pending_tool_calls and cleared.
    assert adapter._next_cancel_reason == ""


async def test_next_cancel_reason_cleared_on_clean_exit() -> None:
    """A clean (non-cancelled) exit also clears any stale tag.

    Defensive: if someone sets the tag speculatively and the invoke
    then finishes normally, we don't want the tag to leak into the
    next cancel.
    """
    from goldfive.adapters.adk import SYMBOLIC_REASON_USER_STEER, ADKAdapter
    from goldfive.types import Session, Task

    agent = _make_agent()

    async def _run_clean():
        if False:  # pragma: no cover
            yield None
        return

    runner_clean = _FakeRunner(_run_clean, agent)
    adapter = ADKAdapter(runner_clean, session_id="sess-clean-clear")
    adapter._next_cancel_reason = SYMBOLIC_REASON_USER_STEER

    await adapter.invoke(Task(id="t1", title="x"), Session(run_id="r"))

    assert adapter._next_cancel_reason == ""


# ---------------------------------------------------------------------------
# Regression guard — every reporting-tool dispatch MUST route through
# :func:`goldfive.adapters._tool_invocation.invoke_tool` so the three
# protection layers (terminal-task rejection, idempotency, loop guard)
# fire. A prior version of ``before_tool_callback`` called the handler
# directly and silently bypassed all three — 500-LLM-call ceilings were
# being hit on already-terminal tasks with nothing but bland
# ``{"acknowledged": True}`` responses reaching the agent. These tests
# are the regression guard for that wiring. See
# ``docs/design/TASK-LIFECYCLE.md`` §5.
# ---------------------------------------------------------------------------


async def test_reporting_tool_dispatch_routes_through_invoke_tool(state_ctx_cls) -> None:
    """15 calls to the same reporting tool on one task must trip the
    volume-cap loop guard (15+ cumulative calls → LOOPING_TOOL_CALL
    drift), proving the plugin's ``before_tool_callback`` runs the
    dispatch through :func:`invoke_tool` and not the handler directly.

    If the wiring regresses (callback calls ``handler(...)`` directly),
    no drift will be emitted and the 15th call will return a plain
    ``{"acknowledged": True}`` — this test catches that.
    """
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
    )
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS
    from goldfive.types import DriftEvent, DriftKind, Plan, TaskEdge

    # _RecordingSteerer-shaped double that the tool-loop-guard's
    # ``emit_loop_drift`` can push a DriftEvent into. The builtin
    # reporting handlers call ``mark_task_*`` on the steerer, so we
    # implement the two we need.
    class _Steerer:
        def __init__(self) -> None:
            self.drifts: list[DriftEvent] = []
            self.blocked_calls: int = 0

        async def mark_task_blocked(self, task_id: str, *, session: Any, **kwargs: Any) -> None:
            self.blocked_calls += 1

        async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
            self.drifts.append(drift)

    spec = next(t for t in BUILTIN_REPORTING_TOOLS if t.name == "report_task_blocked")

    agent = _make_agent()
    adapter = ADKAdapter(agent)
    await adapter.register_reporting_tools([spec])

    steerer = _Steerer()
    task = Task(id="t1", title="x")
    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[task, Task(id="t2", title="y")],
            edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        ),
    )

    ctx = SessionContext(
        session=session,
        steerer=steerer,
        task=task,
        tools=[spec],
        host_agent_name="test_agent",
    )
    state = {SESSION_CONTEXT_STATE_KEY: ctx}

    class _Tool:
        name = "report_task_blocked"

    # Fire 16 calls on the same task, with varied ``blocked_on`` strings
    # so every signature is unique — the volume cap (>= 15 cumulative
    # calls) is the layer that catches this pattern. If the dispatch
    # bypasses invoke_tool, no drift ever fires and the handler is
    # invoked all 16 times.
    results: list[dict[str, Any]] = []
    for i in range(16):
        result = await adapter._plugin.before_tool_callback(
            tool=_Tool(),
            tool_args={"task_id": "t1", "blocked_on": f"dep-{i}"},
            tool_context=state_ctx_cls(state),
        )
        assert isinstance(result, dict)
        results.append(result)

    # The loop guard must have emitted exactly one LOOPING_TOOL_CALL
    # drift — proof that invoke_tool's loop-detection layer ran.
    assert len(steerer.drifts) == 1, (
        f"expected one LOOPING_TOOL_CALL drift; got {len(steerer.drifts)}. "
        "If this fails, before_tool_callback is bypassing invoke_tool "
        "(the regression from PR #94/#98)."
    )
    assert steerer.drifts[0].kind is DriftKind.LOOPING_TOOL_CALL
    assert steerer.drifts[0].current_task_id == "t1"

    # Calls 1..14 reach the handler; call 15 is the one that trips the
    # volume cap — it STILL reaches the handler (the drift is a side
    # effect, not a short-circuit). The handler count therefore ends at
    # 15 (first 15 calls) because the 16th call has the same task_id
    # so is NOT subject to further short-circuiting by the volume
    # check. The exact arithmetic is less important than: the drift
    # fired at all. The regression-catching assertion is the drift
    # count above; the handler count here is a secondary sanity check.
    assert steerer.blocked_calls >= 1, (
        "handler should have run at least once on a non-terminal task"
    )


async def test_reporting_tool_on_terminal_task_returns_structured_rejection(
    state_ctx_cls,
) -> None:
    """A reporting call on an already-terminal task must get the
    structured ``task_already_terminal`` error response — NOT a bland
    ``{"acknowledged": True}``.

    This is the terminal-task rejection layer (layer 1 in
    ``docs/design/TASK-LIFECYCLE.md`` §5). Fires from inside
    ``invoke_tool``; this test proves the ADK plugin's dispatch path
    reaches that layer.
    """
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
    )
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS
    from goldfive.types import Plan, TaskStatus

    # The handler must NEVER run when the task is already terminal —
    # route the call through a spec whose handler raises, to catch a
    # regression where invoke_tool's short-circuit is skipped.
    invoked_count = [0]

    async def _boom_handler(args, session, steerer):
        invoked_count[0] += 1
        raise AssertionError(
            "handler must not run on a terminal task; dispatch should "
            "short-circuit via invoke_tool's terminal rejection"
        )

    spec_template = next(t for t in BUILTIN_REPORTING_TOOLS if t.name == "report_task_progress")
    spec = ReportingToolSpec(
        name=spec_template.name,
        description=spec_template.description,
        parameters=spec_template.parameters,
        handler=_boom_handler,
    )

    agent = _make_agent()
    adapter = ADKAdapter(agent)
    await adapter.register_reporting_tools([spec])

    task = Task(id="t1", title="x", status=TaskStatus.FAILED)
    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[task],
            edges=[],
        ),
    )

    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tools=[spec],
        host_agent_name="test_agent",
    )
    state = {SESSION_CONTEXT_STATE_KEY: ctx}

    class _Tool:
        name = "report_task_progress"

    result = await adapter._plugin.before_tool_callback(
        tool=_Tool(),
        tool_args={"task_id": "t1", "fraction": 0.5, "detail": "too late"},
        tool_context=state_ctx_cls(state),
    )

    assert isinstance(result, dict)
    assert result.get("acknowledged") is False, (
        "expected structured rejection; got acknowledged=true. If this "
        "fails, before_tool_callback is bypassing invoke_tool's "
        "terminal-task rejection layer."
    )
    assert result.get("error") == "task_already_terminal"
    assert result.get("task_id") == "t1"
    assert result.get("current_status") == "FAILED"
    # Handler must have stayed cold.
    assert invoked_count[0] == 0


async def test_reporting_tool_duplicate_returns_duplicate_ack(state_ctx_cls) -> None:
    """A byte-identical follow-up call must get the idempotency ACK
    ``{"acknowledged": True, "duplicate": True}`` instead of re-entering
    the handler — proof that layer 2 (idempotency) is wired.
    """
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
    )
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS
    from goldfive.types import Plan, TaskEdge

    handler_calls: list[dict[str, Any]] = []

    async def _recording_handler(args, session, steerer):
        handler_calls.append(dict(args))
        return {"acknowledged": True}

    spec_template = next(t for t in BUILTIN_REPORTING_TOOLS if t.name == "report_task_started")
    spec = ReportingToolSpec(
        name=spec_template.name,
        description=spec_template.description,
        parameters=spec_template.parameters,
        handler=_recording_handler,
    )

    agent = _make_agent()
    adapter = ADKAdapter(agent)
    await adapter.register_reporting_tools([spec])

    task = Task(id="t1", title="x")
    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[task, Task(id="t2", title="y")],
            edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        ),
    )
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tools=[spec],
        host_agent_name="test_agent",
    )
    state = {SESSION_CONTEXT_STATE_KEY: ctx}

    class _Tool:
        name = "report_task_started"

    args = {"task_id": "t1", "detail": "kick"}
    first = await adapter._plugin.before_tool_callback(
        tool=_Tool(), tool_args=args, tool_context=state_ctx_cls(state)
    )
    second = await adapter._plugin.before_tool_callback(
        tool=_Tool(), tool_args=args, tool_context=state_ctx_cls(state)
    )

    # First call: handler runs, plain ACK.
    assert first == {"acknowledged": True}
    # Second call: handler must NOT re-run; duplicate ACK returned.
    assert second == {"acknowledged": True, "duplicate": True}, (
        "expected duplicate ACK; if this fails, idempotency layer is "
        "bypassed (before_tool_callback is routing around invoke_tool)."
    )
    assert len(handler_calls) == 1


async def test_adversarial_agent_with_varying_task_ids_is_stopped_at_session_cap(
    state_ctx_cls,
) -> None:
    """Live-run regression: an adversarial agent fires
    ``report_task_failed`` over and over, inventing a FRESH
    ``task_id`` each time so the per-task volume cap never trips
    (each per-task bucket stays at 1 call). The session-wide volume
    cap (Layer 4 in ``docs/design/TASK-LIFECYCLE.md`` §5) must catch
    this before ADK's 500-LLM-call ceiling bites — observed in the
    wild as 237 consecutive plain ACKs and no intervention.
    """
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
    )
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS
    from goldfive.types import DriftEvent, DriftKind, Plan

    handler_calls: list[dict[str, Any]] = []

    # Use a permissive handler (not the built-in that would need a
    # functioning Steerer) so we can count invocations directly.
    async def _recording_handler(args, session, steerer):
        handler_calls.append(dict(args))
        return {"acknowledged": True}

    template = next(t for t in BUILTIN_REPORTING_TOOLS if t.name == "report_task_failed")
    spec = ReportingToolSpec(
        name=template.name,
        description=template.description,
        parameters=template.parameters,
        handler=_recording_handler,
    )

    class _Steerer:
        def __init__(self) -> None:
            self.drifts: list[DriftEvent] = []

        async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
            self.drifts.append(drift)

    agent = _make_agent()
    adapter = ADKAdapter(agent)
    await adapter.register_reporting_tools([spec])
    steerer = _Steerer()

    # 60 distinct tasks pre-populated so the ``unknown_task_id`` check
    # doesn't short-circuit before the session counter increments.
    tasks = [Task(id=f"t{i}", title=f"task-{i}") for i in range(60)]
    session = Session(
        run_id="r1",
        plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=tasks, edges=[]),
    )

    ctx = SessionContext(
        session=session,
        steerer=steerer,
        task=tasks[0],
        tools=[spec],
        host_agent_name="test_agent",
    )
    state = {SESSION_CONTEXT_STATE_KEY: ctx}

    class _Tool:
        name = "report_task_failed"

    results: list[dict[str, Any]] = []
    for i in range(60):
        result = await adapter._plugin.before_tool_callback(
            tool=_Tool(),
            tool_args={"task_id": f"t{i}", "reason": f"fresh #{i}"},
            tool_context=state_ctx_cls(state),
        )
        assert isinstance(result, dict)
        results.append(result)

    # The session-wide cap fires exactly one drift — not ADK's 500-call
    # ceiling, and not per-task drifts (each per-task bucket stayed at 1).
    assert len(steerer.drifts) == 1, (
        f"expected session-wide LOOPING_TOOL_CALL drift; got {len(steerer.drifts)}. "
        "If this fails, the session-wide volume cap is not wired through "
        "invoke_tool (the live-run failure from 237 passes of report_task_failed)."
    )
    assert steerer.drifts[0].kind is DriftKind.LOOPING_TOOL_CALL
    assert "session-wide" in steerer.drifts[0].detail.lower()

    # Handler runs for the first 49 (below the 50 threshold); call 50
    # trips the drift and is itself rejected; calls 51..60 stay rejected.
    assert len(handler_calls) == 49
    for r in results[:49]:
        assert r == {"acknowledged": True}
    for r in results[49:]:
        assert r["acknowledged"] is False
        assert r["error"] == "loop_detected"
        assert r["scope"] == "session"
        assert r["tool"] == "report_task_failed"


# ---------------------------------------------------------------------------
# Live-run regression: the SessionContext handoff must survive ADK's
# ``InMemorySessionService`` state-copying semantics.
# ---------------------------------------------------------------------------


async def test_reporting_tool_guards_fire_under_real_adk_runner() -> None:
    """Drive a REAL ``google.adk.runners.InMemoryRunner`` end-to-end and
    verify the reporting-tool handler (and therefore every guard layer)
    is actually invoked.

    This is the regression test for the filler-loop outage: every
    reporting-tool call in a live ADK run was silently falling through
    to the :func:`_build_ack_shim` because the adapter's session-context
    handoff relied on mutating the dict returned by
    ``InMemorySessionService.get_session`` — which returns a shallow
    *copy* of the stored session state on every call. The copy the
    adapter wrote to was discarded; the runner's own
    ``get_session`` produced a second, empty copy for the invocation;
    ``before_tool_callback`` saw no goldfive SessionContext and returned
    ``None``; ADK then called the shim which returned
    ``{"acknowledged": true}`` — bypassing terminal rejection,
    idempotency, per-task and session-wide loop guards entirely.

    The fix hands the context to the goldfive plugin instance directly,
    sidestepping ADK state. This test exercises that path by running
    the *actual* ADK runner with a scripted LLM so the copy-state
    behaviour is real, not simulated. A regression here would once again
    produce 500+ plain ACKs per run with every protection layer silent.
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    from goldfive.adapters.adk import ADKAdapter

    # A scripted LLM that: turn 1 — emits a report_task_completed
    # function call with concrete args; turn 2 — responds "done" and
    # ends. If the guards were bypassed, step 1 would still return an
    # ACK but the handler below would never be called.
    class _ScriptedLLM(BaseLlm):
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
                                    id="call_1",
                                    name="report_task_completed",
                                    args={
                                        "task_id": "raccoon_research",
                                        "summary": "done",
                                    },
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

    agent = Agent(name="presenter", model=_ScriptedLLM(), instruction="")
    adapter = ADKAdapter(agent)

    handler_calls: list[dict[str, Any]] = []

    async def _handler(args: dict[str, Any], session: Any, steerer: Any) -> dict:
        handler_calls.append(dict(args))
        # Mirror what the real handler does: mark the task COMPLETED so
        # the adapter's post-event check tears the run-loop down cleanly.
        from goldfive.types import TaskStatus

        for t in session.plan.tasks:
            if t.id == args.get("task_id"):
                t.status = TaskStatus.COMPLETED
        return {"acknowledged": True, "routed_via_invoke_tool": True}

    spec = ReportingToolSpec(
        name="report_task_completed",
        description="Mark a task completed.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "summary": {"type": "string"},
            },
        },
        handler=_handler,
    )
    await adapter.register_reporting_tools([spec])

    class _StubSteerer:
        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    adapter.bind_steerer(_StubSteerer())

    task = Task(id="raccoon_research", title="research raccoons")
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[task],
        edges=[],
    )
    session = Session(run_id="r1", plan=plan)

    result = await adapter.invoke(task=task, session=session)

    # If the fix regresses, handler_calls stays empty (the shim swallows
    # the call) and the task status never flips to COMPLETED.
    assert handler_calls == [{"task_id": "raccoon_research", "summary": "done"}], (
        "handler never ran: reporting-tool dispatch fell through to the "
        "ACK shim, so every guard layer (terminal rejection, idempotency, "
        "per-task loop cap, session volume cap) was silently bypassed. "
        "This is the regression from the live run that produced 500+ "
        "plain ACKs on a single task."
    )
    assert result.stop_reason in {"final_response", "task_terminal"}


async def test_reporting_tool_guards_fire_across_back_to_back_invocations() -> None:
    """Two sequential ``adapter.invoke`` calls must BOTH route through
    the handler — the plugin-instance handoff must be correctly reset
    between invocations so the second invocation gets its own ctx.

    Specifically covers the filler-loop pattern from the live outage:
    after a mid-run STEER cancels the first invocation and the
    sequential executor starts a fresh invocation on the refined plan,
    the NEW invocation's reporting-tool calls must still reach the
    real handler (not the ACK shim). A regression where
    ``clear_active_context`` is called too aggressively — or where the
    second ``invoke`` fails to re-set the active ctx — would look
    identical in the sink stream: first task healthy, second task
    emits N plain ACKs with no handler invocation.
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    from goldfive.adapters.adk import ADKAdapter

    class _ScriptedLLM(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            # Inspect the LAST event in the conversation to decide:
            # * If the last event is a user turn carrying an
            #   "Also, please: <title>" follow-up (goldfive#141 overlay
            #   phrasing) that hasn't been acted on yet, emit a
            #   function_call for that task id.
            # * Otherwise (the last event is a function_response, meaning
            #   the handler just acked), close out with ``turn_complete``.
            contents = getattr(llm_request, "contents", None) or []
            last_user_task_id = ""
            last_was_function_response_for_task: str | None = None
            for c in contents:
                role = getattr(c, "role", "")
                parts = getattr(c, "parts", None) or []
                for p in parts:
                    text = getattr(p, "text", None)
                    fr = getattr(p, "function_response", None)
                    if text and role == "user":
                        # Extract the title from the adapter's gentle
                        # follow-up prompt "Also, please: <title>. <desc>"
                        # (goldfive#141). Falls back to the legacy
                        # "Task: X" prefix so older test harnesses keep
                        # working.
                        for line in text.splitlines():
                            stripped = line.strip()
                            for prefix in ("Also, please: ", "Task: "):
                                if stripped.startswith(prefix):
                                    tail = stripped[len(prefix) :]
                                    # Drop trailing punctuation / descriptions.
                                    for sep in (". ", "\n"):
                                        idx = tail.find(sep)
                                        if idx >= 0:
                                            tail = tail[:idx]
                                            break
                                    last_user_task_id = tail.rstrip(".").strip()
                                    break
                        last_was_function_response_for_task = None
                    if fr is not None:
                        last_was_function_response_for_task = getattr(fr, "name", "") or ""
            if last_was_function_response_for_task is None and last_user_task_id:
                # Map the nudge title back to the task id we expect.
                task_id = {
                    "waffles": "waffle_research",
                    "raccoons": "raccoon_research",
                }.get(last_user_task_id, last_user_task_id)
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    id="call_" + task_id,
                                    name="report_task_completed",
                                    args={
                                        "task_id": task_id,
                                        "summary": "done",
                                    },
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

    agent = Agent(name="presenter", model=_ScriptedLLM(), instruction="")
    adapter = ADKAdapter(agent)

    handler_calls: list[dict[str, Any]] = []

    async def _handler(args: dict[str, Any], session: Any, steerer: Any) -> dict:
        handler_calls.append(dict(args))
        from goldfive.types import TaskStatus

        for t in session.plan.tasks:
            if t.id == args.get("task_id"):
                t.status = TaskStatus.COMPLETED
        return {"acknowledged": True}

    spec = ReportingToolSpec(
        name="report_task_completed",
        description="Mark a task completed.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "summary": {"type": "string"},
            },
        },
        handler=_handler,
    )
    await adapter.register_reporting_tools([spec])

    class _StubSteerer:
        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    adapter.bind_steerer(_StubSteerer())

    # First invocation: the "pre-STEER" task.
    t_waffle = Task(id="waffle_research", title="waffles")
    plan_rev0 = Plan(id="p0", run_id="r1", goal_ids=[], tasks=[t_waffle], edges=[])
    session = Session(run_id="r1", plan=plan_rev0)
    await adapter.invoke(task=t_waffle, session=session)
    assert len(handler_calls) == 1
    assert handler_calls[0]["task_id"] == "waffle_research"

    # Simulate the STEER-driven plan revision: swap the plan on the
    # session to rev 1 with a new task, which is exactly what
    # ``steerer._apply_revision`` does mid-run.
    t_raccoon = Task(id="raccoon_research", title="raccoons")
    plan_rev1 = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[t_raccoon],
        edges=[],
        revision_index=1,
    )
    session.plan = plan_rev1

    # Second invocation: the "post-STEER" task. This is the one the
    # live run filler-looped on because the plugin-instance handoff
    # was previously via ADK state, which the session-service copy
    # discarded.
    await adapter.invoke(task=t_raccoon, session=session)

    assert len(handler_calls) == 2, (
        "second invocation's reporting-tool call never reached the "
        "handler — the plugin-instance handoff is not being re-set "
        "across back-to-back invocations (the filler-loop regression)."
    )
    assert handler_calls[1]["task_id"] == "raccoon_research"


# ---------------------------------------------------------------------------
# Single-Runner model (goldfive#130)
# ---------------------------------------------------------------------------


def test_single_runner_built_around_root_agent() -> None:
    """``wrap(root)`` produces ONE ``InMemoryRunner`` around the root.

    Under the single-Runner model (goldfive#130) the adapter doesn't
    build per-agent runners — delegation within the tree is ADK's job
    (AgentTool, transfer_to_agent, sub_agents).
    """
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    def _mk(name: str) -> Any:
        return LlmAgent(name=name, model="fake-model", description=name, instruction="x")

    research = _mk("research")
    web = _mk("web")
    reviewer = _mk("reviewer")
    coordinator = _mk("coordinator")
    coordinator.tools = [AgentTool(research), AgentTool(web), AgentTool(reviewer)]

    adapter = ADKAdapter(coordinator)
    # Exactly one runner exists and it wraps the coordinator (the root).
    assert adapter._agent is coordinator
    # available_agents reports every reachable agent name — advisory for
    # the planner, not a dispatch registry.
    assert adapter.available_agents == sorted(["coordinator", "research", "web", "reviewer"])


def test_available_agents_reports_reachable_names() -> None:
    """``available_agents`` walks ``sub_agents`` / ``inner_agent`` /
    ``AgentTool.agent`` edges so the planner can see delegation targets.
    """
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    def _mk(name: str) -> Any:
        return LlmAgent(name=name, model="fake-model", description=name, instruction="x")

    leaf = _mk("leaf")
    mid = _mk("mid")
    mid.tools = [AgentTool(leaf)]
    root = _mk("root")
    root.sub_agents = [mid]

    adapter = ADKAdapter(root)
    assert adapter.available_agents == ["leaf", "mid", "root"]


# ---------------------------------------------------------------------------
# available_agents_tree — tree-aware planner constraints (goldfive#151)
# ---------------------------------------------------------------------------


def _mk_llm(name: str) -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(name=name, model="fake-model", description=name, instruction="x")


def test_available_agents_tree_single_llm_agent() -> None:
    """Fixture 1: single LlmAgent. Tree has one root leaf."""
    from goldfive.adapters.adk import ADKAdapter

    solo = _mk_llm("solo")
    adapter = ADKAdapter(solo)
    tree = adapter.available_agents_tree
    assert len(tree) == 1
    entry = tree[0]
    assert entry["name"] == "solo"
    assert entry["depth"] == 0
    assert entry["parent"] == ""
    assert entry["role"] == "root"
    assert entry["kind"] == "LlmAgent"


def test_available_agents_tree_flat_specialist_tree() -> None:
    """Fixture 2: coordinator with three specialist sub_agents.

    Tree shape: coordinator (root) -> A, B, C (leaves, depth=1).
    """
    from goldfive.adapters.adk import ADKAdapter

    a = _mk_llm("agent_a")
    b = _mk_llm("agent_b")
    c = _mk_llm("agent_c")
    coordinator = _mk_llm("coordinator")
    coordinator.sub_agents = [a, b, c]

    adapter = ADKAdapter(coordinator)
    tree = adapter.available_agents_tree
    by_name = {e["name"]: e for e in tree}
    assert set(by_name) == {"coordinator", "agent_a", "agent_b", "agent_c"}
    assert by_name["coordinator"]["depth"] == 0
    assert by_name["coordinator"]["role"] == "root"
    assert by_name["coordinator"]["parent"] == ""
    for leaf_name in ("agent_a", "agent_b", "agent_c"):
        entry = by_name[leaf_name]
        assert entry["depth"] == 1
        assert entry["role"] == "leaf"
        assert entry["parent"] == "coordinator"
        assert entry["kind"] == "LlmAgent"


def test_available_agents_tree_deep_hierarchy() -> None:
    """Fixture 3: deep hierarchy (depth >= 3)."""
    from goldfive.adapters.adk import ADKAdapter

    leaf1 = _mk_llm("leaf1")
    leaf2 = _mk_llm("leaf2")
    coord1 = _mk_llm("coord1")
    coord1.sub_agents = [leaf1, leaf2]
    root = _mk_llm("root")
    root.sub_agents = [coord1]

    adapter = ADKAdapter(root)
    tree = adapter.available_agents_tree
    by_name = {e["name"]: e for e in tree}
    assert set(by_name) == {"root", "coord1", "leaf1", "leaf2"}
    assert by_name["root"]["depth"] == 0
    assert by_name["root"]["role"] == "root"
    assert by_name["coord1"]["depth"] == 1
    assert by_name["coord1"]["role"] == "intermediate"
    assert by_name["coord1"]["parent"] == "root"
    assert by_name["leaf1"]["depth"] == 2
    assert by_name["leaf1"]["parent"] == "coord1"
    assert by_name["leaf1"]["role"] == "leaf"
    assert by_name["leaf2"]["depth"] == 2
    assert by_name["leaf2"]["parent"] == "coord1"
    assert by_name["leaf2"]["role"] == "leaf"


def test_available_agents_tree_agnostic_to_agent_class() -> None:
    """Custom BaseAgent subclasses pass through ``kind`` unchanged.

    Tree-agnostic invariant (goldfive#151): the walker reports the raw
    class name of whatever agent is in the tree — it does NOT
    special-case LlmAgent / SequentialAgent / ParallelAgent / BaseAgent.
    """
    from google.adk.agents.base_agent import BaseAgent  # type: ignore

    from goldfive.adapters.adk import ADKAdapter

    class CustomAgent(BaseAgent):
        """A caller-supplied subclass the walker must not interpret."""

        async def _run_async_impl(self, ctx):  # noqa: D401, ANN001
            return
            yield  # pragma: no cover - required async generator shape

    custom_leaf = CustomAgent(name="my_custom_leaf", description="x")
    root = _mk_llm("root")
    root.sub_agents = [custom_leaf]

    adapter = ADKAdapter(root)
    tree = adapter.available_agents_tree
    by_name = {e["name"]: e for e in tree}
    assert by_name["my_custom_leaf"]["kind"] == "CustomAgent"
    assert by_name["my_custom_leaf"]["role"] == "leaf"
    assert by_name["my_custom_leaf"]["parent"] == "root"
    assert by_name["root"]["kind"] == "LlmAgent"


async def test_invoke_always_drives_the_one_runner() -> None:
    """Every ``invoke`` drives the one runner around the root, regardless
    of ``task.assignee_agent_id``.

    Delegation within the tree happens via ADK's native mechanisms —
    goldfive does not route on assignee. The assignee field is still
    carried on the task for observability + the planner's delegation
    hints in prompts.
    """
    from dataclasses import dataclass, field

    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    @dataclass
    class _Event:
        marker: str = ""
        content: Any = None

    called_count = {"n": 0}

    @dataclass
    class _FakeRunner:
        session_service: Any = None
        plugin_manager: Any = field(default=None)
        app_name: str = "coordinator"
        agent: Any = None

        async def run_async(self, **kwargs: Any):  # noqa: ARG002
            called_count["n"] += 1
            yield _Event(marker="tick")

    def _mk(name: str) -> Any:
        return LlmAgent(name=name, model="fake-model", description=name, instruction="x")

    research = _mk("research")
    web = _mk("web")
    coordinator = _mk("coordinator")
    coordinator.tools = [AgentTool(research), AgentTool(web)]

    adapter = ADKAdapter(coordinator)
    adapter._runner = _FakeRunner(agent=coordinator)
    adapter._session_id = "sess-1"

    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[Task(id="t1", title="do research", assignee_agent_id="research")],
            edges=[],
        ),
    )
    # Even with a non-empty, non-root assignee the one runner is driven.
    await adapter.invoke(
        task=Task(id="t1", title="do research", assignee_agent_id="research"),
        session=session,
    )
    await adapter.invoke(
        task=Task(id="t2", title="build ui", assignee_agent_id="web"),
        session=session,
    )
    # Even an unknown assignee does not raise — the assignee isn't routed.
    await adapter.invoke(
        task=Task(id="t3", title="x", assignee_agent_id="does_not_exist"),
        session=session,
    )
    assert called_count["n"] == 3


# ---------------------------------------------------------------------------
# State-protocol reliability (feat/registry-dispatch-model §4)
# ---------------------------------------------------------------------------


async def test_state_protocol_writes_propagate_through_agent_tool_subtree() -> None:
    """RELIABILITY CONTRACT: task T dispatched to agent A with
    ``AgentTool(B)`` must let B's ``before_model_callback`` see
    ``state[goldfive.current_task_id] == T.id``.

    Previously these state writes were done against a shallow copy of
    the session returned by ``InMemorySessionService.get_session`` and
    were flagged "best-effort". They now happen inside the plugin's
    ``before_run_callback`` against the LIVE invocation session, so
    they propagate through AgentTool sub-Runners automatically (each
    sub-Runner inherits the plugin and fires its own
    ``before_run_callback`` that seeds the sub-session's live state).
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types as genai_types

    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID
    from goldfive.adapters.adk import ADKAdapter

    observed_task_ids_in_B: list[str] = []

    class _ScriptedB(BaseLlm):
        model: str = "fake-model"
        _step: int = 0

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            self._step += 1
            # Check the session state at the time B's LLM is called.
            # We'll pull it from llm_request.contents is not enough — we
            # need the live session state. Use the next yielded response
            # to carry the observation back via a side channel (the
            # list captured by closure).
            #
            # Approach: yield a turn_complete final and rely on
            # the adapter's callback-driven state writes. The observation
            # comes from B's ``before_model_callback`` which fires on
            # the sub-Runner (same goldfive plugin, same ctx) — but our
            # plugin write happens in before_run_callback on the live
            # session, which is invisible here. Instead, read state
            # when we're about to generate by querying via an
            # injected tool. Simplest path: require B to be directly
            # dispatched and drive it that way — see test below.
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="done")],
                ),
                turn_complete=True,
            )

    # Simplify: directly register a tool on B whose handler reads the
    # sub-session's live state so we can assert the seed is there.
    class _ScriptedA(BaseLlm):
        model: str = "fake-model"
        _step: int = 0

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            self._step += 1
            if self._step == 1:
                # Turn 1: call the B AgentTool so ADK spawns a sub-Runner.
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    id="call_b",
                                    name="agent_b",
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

    # Agent B — registers a pre-model callback that records the
    # goldfive.current_task_id on the sub-session's live state.
    def _b_before_model(callback_context: Any, llm_request: Any) -> None:  # noqa: ARG001
        state = getattr(getattr(callback_context, "_invocation_context", None), "session", None)
        if state is not None:
            state = getattr(state, "state", None)
        if state is None:
            state = getattr(getattr(callback_context, "session", None), "state", None)
        tid = None
        if state is not None:
            try:
                tid = state.get(KEY_CURRENT_TASK_ID)
            except Exception:
                tid = None
        observed_task_ids_in_B.append(str(tid or ""))

    agent_b = Agent(
        name="agent_b",
        model=_ScriptedB(),
        instruction="",
        before_model_callback=_b_before_model,
    )
    agent_a = Agent(
        name="agent_a",
        model=_ScriptedA(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)

    class _StubSteerer:
        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    adapter.bind_steerer(_StubSteerer())

    task = Task(id="task_xyz", title="compound", assignee_agent_id="agent_a")
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[task],
        edges=[],
    )
    session = Session(run_id="r1", plan=plan)

    await adapter.invoke(task=task, session=session)

    # B's before_model_callback must have observed the goldfive task id
    # on the sub-session's live state — proof the reliability contract
    # holds across the AgentTool boundary.
    assert observed_task_ids_in_B, "B's before_model_callback never fired"
    assert observed_task_ids_in_B[0] == "task_xyz", (
        f"B saw current_task_id={observed_task_ids_in_B[0]!r}, expected "
        "'task_xyz'. This is the state-protocol reliability contract "
        "for AgentTool sub-Runners: if writes only landed on the "
        "top-level session (not the sub-Runner's live session), B would "
        "see an empty id here. See plugin.before_run_callback."
    )


# ---------------------------------------------------------------------------
# Sink events — AgentInvocationStarted / Completed / DelegationObserved
# ---------------------------------------------------------------------------


async def test_agent_invocation_started_and_completed_emitted_to_sinks() -> None:
    """An invoke() must emit AgentInvocationStarted at run entry and
    AgentInvocationCompleted at run exit on the session's sinks."""
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.sinks.memory import InMemorySink

    class _Scripted(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="hi")],
                ),
                turn_complete=True,
            )

    agent = Agent(name="worker", model=_Scripted(), instruction="")
    adapter = ADKAdapter(agent)

    sink = InMemorySink()

    class _SinkingSteerer:
        def __init__(self) -> None:
            self._sinks = [sink]

        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    adapter.bind_steerer(_SinkingSteerer())

    task = Task(id="t1", title="go", assignee_agent_id="worker")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)

    await adapter.invoke(task=task, session=session)

    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert "agent_invocation_started" in kinds
    assert "agent_invocation_completed" in kinds
    started = next(e for e in sink.events if e.WhichOneof("payload") == "agent_invocation_started")
    assert started.agent_invocation_started.agent_name == "worker"
    assert started.agent_invocation_started.task_id == "t1"
    # parent_invocation_id is empty on the top-level dispatch.
    assert started.agent_invocation_started.parent_invocation_id == ""


async def test_delegation_observed_emitted_on_agent_tool_call() -> None:
    """When an AgentTool is invoked, the plugin emits DelegationObserved
    with from_agent = host, to_agent = wrapped.
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types as genai_types

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.sinks.memory import InMemorySink

    class _B(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="b done")],
                ),
                turn_complete=True,
            )

    class _A(BaseLlm):
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
                                    id="c1", name="agent_b", args={"q": "go"}
                                )
                            ),
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="a done")],
                    ),
                    turn_complete=True,
                )

    agent_b = Agent(name="agent_b", model=_B(), instruction="")
    agent_a = Agent(
        name="agent_a",
        model=_A(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )
    adapter = ADKAdapter(agent_a)

    sink = InMemorySink()

    class _SinkingSteerer:
        def __init__(self) -> None:
            self._sinks = [sink]

        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    adapter.bind_steerer(_SinkingSteerer())

    task = Task(id="t1", title="do a", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    delegations = [
        e.delegation_observed
        for e in sink.events
        if e.WhichOneof("payload") == "delegation_observed"
    ]
    assert len(delegations) >= 1, "expected at least one DelegationObserved"
    first = delegations[0]
    assert first.from_agent == "agent_a"
    assert first.to_agent == "agent_b"
    assert first.task_id == "t1"


# ---------------------------------------------------------------------------
# Caller-supplied plugin propagation (goldfive#121)
# ---------------------------------------------------------------------------


def test_build_runner_without_plugins() -> None:
    """``_build_runner(agent)`` must not pass a ``plugins`` kwarg to
    ``InMemoryRunner`` when the caller didn't supply any.

    Without the ``if plugins:`` guard, callers would always pay for a
    ``plugins=[]`` kwarg which also masks ADK versions that omit the
    parameter from their ``__init__``.
    """
    from goldfive.adapters import adk as adk_mod

    captured: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    import google.adk.runners as adk_runners  # type: ignore

    orig = adk_runners.InMemoryRunner
    adk_runners.InMemoryRunner = _FakeRunner  # type: ignore[assignment]
    try:
        adk_mod._build_runner(_make_agent())
    finally:
        adk_runners.InMemoryRunner = orig  # type: ignore[assignment]

    assert "plugins" not in captured, (
        "no plugins supplied — _build_runner must omit the kwarg entirely"
    )
    assert captured["app_name"] == "test_agent"


def test_build_runner_with_plugins() -> None:
    """``_build_runner(agent, plugins=[...])`` must forward the plugin
    list into :class:`InMemoryRunner` via the ``plugins=`` kwarg.
    """
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore

    from goldfive.adapters import adk as adk_mod

    class _StubPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="stub")

    captured: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    plugin = _StubPlugin()
    import google.adk.runners as adk_runners  # type: ignore

    orig = adk_runners.InMemoryRunner
    adk_runners.InMemoryRunner = _FakeRunner  # type: ignore[assignment]
    try:
        adk_mod._build_runner(_make_agent(), plugins=[plugin])
    finally:
        adk_runners.InMemoryRunner = orig  # type: ignore[assignment]

    assert captured.get("plugins") == [plugin]


def test_adk_adapter_init_accepts_plugins() -> None:
    """``ADKAdapter(tree, plugins=[...])`` stores the plugin list and
    propagates it onto EVERY per-agent runner's plugin manager.

    This is the goldfive#121 fix: without this, caller-supplied
    observability plugins only land on the coordinator's runner because
    the ADK ``App(plugins=[...])`` surface only covers the root.
    """
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore
    from google.adk.tools.agent_tool import AgentTool  # type: ignore

    from goldfive.adapters.adk import ADKAdapter

    class _StubPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="adapter_init_stub")

    def _mk(name: str) -> Any:
        return LlmAgent(name=name, model="fake-model", description=name, instruction="x")

    research = _mk("research")
    web = _mk("web")
    coordinator = _mk("coordinator")
    coordinator.tools = [AgentTool(research), AgentTool(web)]

    plugin = _StubPlugin()
    adapter = ADKAdapter(coordinator, plugins=[plugin])

    assert adapter._plugins == [plugin]
    # Single-Runner model: the caller plugin is installed on the one
    # runner. ADK's AgentTool propagates the plugin_manager into sub-
    # Runners automatically, so coverage across the tree is inherent.
    installed = list(getattr(adapter._runner.plugin_manager, "plugins", []))
    assert plugin in installed, "caller-supplied plugin did not land on the runner"


def test_wrap_accepts_plugins() -> None:
    """``goldfive.wrap(tree, plugins=[...])`` must forward the plugin
    list through to the underlying :class:`ADKAdapter`.
    """
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore
    from google.adk.tools.agent_tool import AgentTool  # type: ignore

    import goldfive
    from goldfive.adapters.adk_wrap import GoldfiveADKAgent

    class _StubPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="wrap_stub")

    def _mk(name: str) -> Any:
        return LlmAgent(name=name, model="fake-model", description=name, instruction="x")

    sub = _mk("sub")
    coordinator = _mk("coordinator")
    coordinator.tools = [AgentTool(sub)]

    plugin = _StubPlugin()
    wrapped = goldfive.wrap(coordinator, plugins=[plugin])
    assert isinstance(wrapped, GoldfiveADKAgent)

    adapter = wrapped._runner.agent
    assert adapter._plugins == [plugin]
    installed = list(getattr(adapter._runner.plugin_manager, "plugins", []))
    assert plugin in installed, "wrap(plugins=[...]) did not propagate onto the runner"


async def test_caller_plugin_fires_on_sub_agent_invocation_via_agent_tool() -> None:
    """ADK propagates the runner's plugin_manager into AgentTool sub-
    Runners, so a caller plugin's ``before_run_callback`` fires for
    both the coordinator and for any sub-agent spawned via AgentTool.

    This is what lets harmonograf observe the entire delegation tree
    under the single-Runner model (goldfive#130) with no special
    registration path — the ADK plugin manager is the propagation seam.
    """
    from google.adk.agents import Agent  # type: ignore
    from google.adk.models.base_llm import BaseLlm  # type: ignore
    from google.adk.models.llm_response import LlmResponse  # type: ignore
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore
    from google.adk.tools.agent_tool import AgentTool  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    from goldfive.adapters.adk import ADKAdapter

    class _RecordingPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="recording")
            self.agents_seen: list[str] = []

        async def before_run_callback(self, *, invocation_context: Any) -> None:  # noqa: ARG002
            agent = getattr(invocation_context, "agent", None)
            name = getattr(agent, "name", "") if agent is not None else ""
            self.agents_seen.append(str(name))

    class _SubLlm(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="sub done")],
                ),
                turn_complete=True,
            )

    class _CoordLlm(BaseLlm):
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
                                    id="call_sub",
                                    name="sub_agent",
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
                        parts=[genai_types.Part(text="coord done")],
                    ),
                    turn_complete=True,
                )

    sub = Agent(name="sub_agent", model=_SubLlm(), instruction="")
    coordinator = Agent(
        name="coordinator",
        model=_CoordLlm(),
        instruction="",
        tools=[AgentTool(sub)],
    )

    plugin = _RecordingPlugin()
    adapter = ADKAdapter(coordinator, plugins=[plugin])

    class _StubSteerer:
        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

    adapter.bind_steerer(_StubSteerer())

    task = Task(id="t1", title="do it")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    # The caller plugin fires on BOTH the coordinator invocation and
    # the AgentTool-spawned sub-Runner invocation — ADK propagates its
    # plugin_manager through AgentTool automatically, so the same
    # observability plugin sees the whole delegation tree.
    assert "coordinator" in plugin.agents_seen, (
        f"plugin never fired for the coordinator; saw {plugin.agents_seen!r}"
    )
    assert "sub_agent" in plugin.agents_seen, (
        "plugin never fired for the AgentTool-spawned sub-Runner — ADK's "
        "plugin_manager propagation broke. Saw: "
        f"{plugin.agents_seen!r}"
    )


# ---------------------------------------------------------------------------
# Duplicate-plugin dedup (goldfive#166)
# ---------------------------------------------------------------------------


def test_dedupe_plugins_by_name_drops_later_duplicates() -> None:
    """``_dedupe_plugins_by_name`` keeps the first occurrence and drops
    later instances sharing a ``name``. Plugins without a ``name`` pass
    through unchanged — only identically-named duplicates are a problem.
    """
    from dataclasses import dataclass

    from goldfive.adapters import adk as adk_mod

    @dataclass
    class _NamedPlugin:
        name: str

    class _NamelessPlugin:
        pass

    a = _NamedPlugin(name="telemetry")
    b = _NamedPlugin(name="telemetry")  # same name -> duplicate
    c = _NamedPlugin(name="goldfive_adk_plugin")
    d = _NamelessPlugin()  # no name attribute
    e = _NamelessPlugin()

    deduped = adk_mod._dedupe_plugins_by_name([a, b, c, d, e])

    assert deduped == [a, c, d, e]


def test_build_runner_dedupes_plugins_before_inmemoryrunner() -> None:
    """``_build_runner`` must collapse same-``name`` duplicates before
    forwarding to :class:`InMemoryRunner`.

    Without this, ADK's ``PluginManager.register_plugin`` raises
    ``ValueError`` for the duplicate and the caller's intent (install
    one telemetry plugin) is lost. See goldfive #166.
    """
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore

    from goldfive.adapters import adk as adk_mod

    class _StubPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="duplicate-stub")

    captured: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    first = _StubPlugin()
    second = _StubPlugin()  # same name — should be dropped
    import google.adk.runners as adk_runners  # type: ignore

    orig = adk_runners.InMemoryRunner
    adk_runners.InMemoryRunner = _FakeRunner  # type: ignore[assignment]
    try:
        adk_mod._build_runner(_make_agent(), plugins=[first, second])
    finally:
        adk_runners.InMemoryRunner = orig  # type: ignore[assignment]

    passed = captured.get("plugins")
    assert passed == [first], (
        "duplicate plugin instance must be dropped before reaching InMemoryRunner; "
        f"got {passed!r}"
    )


def test_register_plugin_on_runner_is_idempotent_by_name() -> None:
    """When the runner already carries a plugin with the same name, the
    helper must skip the install rather than appending a second copy.

    This is the belt-and-braces guard for goldfive #166: the harmonograf
    plugin's own dedup is primary, but goldfive's install path must also
    refuse to stack two instances of the same name. Regression test for
    the pre-fix behaviour where a ``register_plugin`` ValueError was
    caught and the helper fell through to ``plugins.append(plugin)``.
    """
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore

    from goldfive.adapters.adk import _register_plugin_on_runner

    class _Stub(BasePlugin):
        def __init__(self, name: str) -> None:
            super().__init__(name=name)

    first = _Stub("telemetry")
    second = _Stub("telemetry")  # same name

    class _FakeRunner:
        class plugin_manager:  # type: ignore[misc]
            plugins: list[Any] = []

            @classmethod
            def register_plugin(cls, plugin: Any) -> None:
                # Mirror ADK's PluginManager: raise on duplicate name.
                if any(p.name == plugin.name for p in cls.plugins):
                    raise ValueError(
                        f"Plugin with name '{plugin.name}' already registered."
                    )
                cls.plugins.append(plugin)

    runner = _FakeRunner()
    assert _register_plugin_on_runner(runner, first) is True
    assert _register_plugin_on_runner(runner, second) is True
    assert runner.plugin_manager.plugins == [first], (
        "dedup guard must short-circuit; got "
        f"{runner.plugin_manager.plugins!r}"
    )


async def test_wrap_skips_duplicate_plugin_on_same_tree() -> None:
    """``ADKAdapter(tree, plugins=[p, p'])`` where both plugins share a
    name must install only one of them on the runner.

    Exercises the integration path end-to-end: pass two instances of the
    same-named plugin and assert the runner's plugin list carries just
    one. Goldfive #166 regression.
    """
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore

    from goldfive.adapters.adk import ADKAdapter

    class _Stub(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="tele-stub")

    p1 = _Stub()
    p2 = _Stub()  # same name
    adapter = ADKAdapter(_make_agent(), plugins=[p1, p2])

    installed = list(getattr(adapter._runner.plugin_manager, "plugins", []))
    tele = [p for p in installed if getattr(p, "name", "") == "tele-stub"]
    assert len(tele) == 1, (
        f"expected a single telemetry-stub plugin after dedup, got {tele!r}"
    )
    # First instance wins.
    assert tele[0] is p1


async def test_add_plugin_skips_when_same_name_already_installed() -> None:
    """``ADKAdapter.add_plugin`` must not stack a second instance when a
    plugin of the same name already sits on the runner.

    Under ``goldfive.wrap`` + ``adk web``, the outer App carries the
    telemetry plugin and a downstream call (``observe()``,
    ``add_plugin``) can end up trying to attach a second instance. The
    guard prevents the duplicate span-doubling pathology (goldfive #166)
    even when the harmonograf plugin's own dedup is disabled or absent.
    """
    from google.adk.plugins.base_plugin import BasePlugin  # type: ignore

    from goldfive.adapters.adk import ADKAdapter

    class _Stub(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="observability")

    adapter = ADKAdapter(_make_agent(), plugins=[_Stub()])
    before = list(getattr(adapter._runner.plugin_manager, "plugins", []))
    before_count = sum(1 for p in before if getattr(p, "name", "") == "observability")
    assert before_count == 1

    # User / library tries to install a second same-named instance.
    adapter.add_plugin(_Stub())

    after = list(getattr(adapter._runner.plugin_manager, "plugins", []))
    after_count = sum(1 for p in after if getattr(p, "name", "") == "observability")
    assert after_count == 1, (
        f"add_plugin installed a duplicate; counts before={before_count} after={after_count}"
    )
