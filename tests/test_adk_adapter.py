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
    # The payload flags this as goldfive-synthesized so downstream tools can
    # tell it apart from a real tool return.
    assert responses[0].response.get("goldfive_cancelled") is True


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
