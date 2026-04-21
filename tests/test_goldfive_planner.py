"""Tests for :class:`goldfive.planners.goldfive_planner.GoldfivePlanner`
and the plugin-side request-side instruction injection (goldfive#153).

Skipped entirely when ``google.adk`` is not installed (optional dep).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    _inject_goldfive_planner_instruction,
    make_adk_plugin,
)
from goldfive.adapters._adk_state_protocol import (  # noqa: E402
    KEY_CURRENT_TASK_ID,
    KEY_CURRENT_TASK_TITLE,
)
from goldfive.adapters.adk import (  # noqa: E402
    GOLDFIVE_PLANNER_OPT_OUT_ATTR,
    ADKAdapter,
    _attach_goldfive_planner_to_tree,
    _rebind_goldfive_planners,
)
from goldfive.planners.goldfive_planner import (  # noqa: E402
    KEY_ACTIVE_STEER_BODY,
    KEY_CANCELLED_FUNCTION_CALL_IDS,
    KEY_GOALS_SUMMARY,
    GoldfivePlanner,
)
from goldfive.types import DriftEvent, DriftKind, Session, Task  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_agent(name: str = "agent_under_test") -> Any:
    """Construct a bare ADK ``LlmAgent`` for planner attachment tests."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description="Test agent",
        instruction="Test.",
    )


@dataclass
class _FakeReadonlyContext:
    """Minimal ReadonlyContext stand-in — only needs ``.state`` for our planner."""

    state: dict


class _FakeFunctionCall:
    """Duck-type of ``types.FunctionCall`` carrying id + name."""

    def __init__(self, *, id: str, name: str, args: dict | None = None) -> None:
        self.id = id
        self.name = name
        self.args = args or {}


class _FakePart:
    """Duck-type of ``types.Part`` with either ``function_call`` or ``text``."""

    def __init__(
        self,
        *,
        function_call: _FakeFunctionCall | None = None,
        text: str = "",
    ) -> None:
        self.function_call = function_call
        self.text = text


class _RecordingSteerer:
    """Async-capable steerer stub that records _handle_drift calls."""

    def __init__(self) -> None:
        self.drifts: list[DriftEvent] = []
        self._sinks: list = []

    async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
        self.drifts.append(drift)

    async def observe(self, event: Any, session: Any) -> None:
        pass


class _RecordingUserPlanner:
    """Test double for a user-supplied BasePlanner.

    Records the calls and returns canned values so composition
    order is observable.
    """

    def __init__(
        self,
        *,
        build_return: str | None = "USER_BLOCK",
        process_return: list[_FakePart] | None = None,
    ) -> None:
        # Import here so non-ADK test collection doesn't break.
        from google.adk.planners.base_planner import BasePlanner  # type: ignore

        # We intentionally DON'T inherit from BasePlanner as a class so we
        # can be duck-typed; but to make isinstance(...) checks work we
        # ALSO register via a tiny BasePlanner subclass wrapper.
        self._BasePlanner = BasePlanner
        self.build_calls: list[tuple] = []
        self.process_calls: list[tuple] = []
        self._build_return = build_return
        self._process_return = process_return

    def build_planning_instruction(self, readonly_context, llm_request):
        self.build_calls.append((readonly_context, llm_request))
        return self._build_return

    def process_planning_response(self, callback_context, response_parts):
        self.process_calls.append((callback_context, list(response_parts)))
        return self._process_return


def _make_user_planner(**kwargs) -> Any:
    """Build a BasePlanner-subclass wrapper around :class:`_RecordingUserPlanner`."""
    from google.adk.planners.base_planner import BasePlanner  # type: ignore

    rec = _RecordingUserPlanner(**kwargs)

    class _BPShim(BasePlanner):
        def build_planning_instruction(self, readonly_context, llm_request):
            return rec.build_planning_instruction(readonly_context, llm_request)

        def process_planning_response(self, callback_context, response_parts):
            return rec.process_planning_response(callback_context, response_parts)

    shim = _BPShim()
    # Stick the recorder on the shim so tests can read its call log.
    shim.recorder = rec  # type: ignore[attr-defined]
    return shim


# ---------------------------------------------------------------------------
# build_planning_instruction
# ---------------------------------------------------------------------------


def test_goldfive_planner_build_instruction_reads_state() -> None:
    """State values render into the emitted orchestration block."""
    planner = GoldfivePlanner()
    state = {
        KEY_CURRENT_TASK_ID: "task-42",
        KEY_CURRENT_TASK_TITLE: "Summarise findings",
        KEY_GOALS_SUMMARY: "deliver a 500-word brief",
        KEY_ACTIVE_STEER_BODY: "focus on the cost angle",
    }
    ctx = _FakeReadonlyContext(state=state)
    out = planner.build_planning_instruction(ctx, llm_request=None)

    assert out is not None
    assert "[GOLDFIVE ORCHESTRATION CONTEXT]" in out
    assert "task-42" in out
    assert "Summarise findings" in out
    assert "deliver a 500-word brief" in out
    assert "focus on the cost angle" in out


def test_goldfive_planner_build_instruction_defaults_when_state_empty() -> None:
    """Missing state keys render as ``(none)`` so the block is never empty."""
    planner = GoldfivePlanner()
    ctx = _FakeReadonlyContext(state={})
    out = planner.build_planning_instruction(ctx, llm_request=None)

    assert out is not None
    assert "[GOLDFIVE ORCHESTRATION CONTEXT]" in out
    # Every placeholder defaults to ``(none)`` when state is empty.
    assert out.count("(none)") >= 3


def test_goldfive_planner_build_instruction_tree_agnostic() -> None:
    """No domain / presentation-agent vocabulary in the emitted block."""
    planner = GoldfivePlanner()
    state = {
        KEY_CURRENT_TASK_ID: "t1",
        KEY_CURRENT_TASK_TITLE: "Do research",
        KEY_GOALS_SUMMARY: "investigate widgets",
    }
    out = (
        planner.build_planning_instruction(_FakeReadonlyContext(state=state), llm_request=None)
        or ""
    )

    banned = (
        "presentation",
        "presenter",
        "slide",
        "deck",
        "researcher",
        "coordinator",
        "specialist",
    )
    low = out.lower()
    for term in banned:
        assert term not in low, (
            f"tree-agnostic contract violated: emitted block contains {term!r}\nblock:\n{out}"
        )


def test_goldfive_planner_composes_with_user_planner_build() -> None:
    """User planner's result is prepended ahead of goldfive's block."""
    user_planner = _make_user_planner(build_return="USER_META_INSTRUCTION")
    planner = GoldfivePlanner(user_planner=user_planner)
    state = {KEY_CURRENT_TASK_ID: "t1", KEY_CURRENT_TASK_TITLE: "Step 1"}

    out = (
        planner.build_planning_instruction(_FakeReadonlyContext(state=state), llm_request=None)
        or ""
    )

    assert "USER_META_INSTRUCTION" in out
    assert "[GOLDFIVE ORCHESTRATION CONTEXT]" in out
    # User block is prepended — its text appears before goldfive's.
    assert out.index("USER_META_INSTRUCTION") < out.index("[GOLDFIVE ORCHESTRATION CONTEXT]")
    # Composition called through.
    assert len(user_planner.recorder.build_calls) == 1


def test_goldfive_planner_handles_user_planner_raising() -> None:
    """A raising user planner's error is swallowed; goldfive block still emits."""
    from google.adk.planners.base_planner import BasePlanner  # type: ignore

    class _Boom(BasePlanner):
        def build_planning_instruction(self, *a, **kw):
            raise RuntimeError("boom")

        def process_planning_response(self, *a, **kw):
            return None

    planner = GoldfivePlanner(user_planner=_Boom())
    out = planner.build_planning_instruction(_FakeReadonlyContext(state={}), llm_request=None) or ""
    assert "[GOLDFIVE ORCHESTRATION CONTEXT]" in out


# ---------------------------------------------------------------------------
# process_planning_response
# ---------------------------------------------------------------------------


async def test_goldfive_planner_process_response_strips_cancelled_call_ids() -> None:
    """function_call parts with cancelled ids are filtered out."""
    planner = GoldfivePlanner()
    state = {KEY_CANCELLED_FUNCTION_CALL_IDS: ["fc-x", "fc-y"]}
    parts = [
        _FakePart(function_call=_FakeFunctionCall(id="fc-x", name="do_a")),
        _FakePart(text="some text"),
        _FakePart(function_call=_FakeFunctionCall(id="fc-ok", name="do_b")),
        _FakePart(function_call=_FakeFunctionCall(id="fc-y", name="do_c")),
    ]
    ctx = _FakeReadonlyContext(state=state)

    out = planner.process_planning_response(ctx, parts)

    assert out is not None
    ids = [p.function_call.id for p in out if p.function_call is not None]
    assert ids == ["fc-ok"]
    # Non-function_call parts preserved.
    assert any(p.text == "some text" for p in out)


def test_goldfive_planner_process_response_noop_when_no_filters_fire() -> None:
    """Return ``None`` when no cancels and no divergence signal — ADK skip-flag."""
    planner = GoldfivePlanner()
    ctx = _FakeReadonlyContext(state={})
    parts = [_FakePart(text="hello")]
    out = planner.process_planning_response(ctx, parts)
    assert out is None


async def test_goldfive_planner_process_response_emits_divergence_on_off_registry_agent() -> None:
    """A function_call to an unknown agent name triggers PLAN_DIVERGENCE."""
    steerer = _RecordingSteerer()
    session = Session(run_id="r")
    planner = GoldfivePlanner(
        agent_registry=["researcher", "writer"],
        steerer=steerer,
        session=session,
    )
    parts = [
        _FakePart(
            function_call=_FakeFunctionCall(id="fc-1", name="researcher", args={}),
        ),
        _FakePart(
            function_call=_FakeFunctionCall(id="fc-2", name="rogue_agent", args={}),
        ),
        # Reporting tool — must not trigger divergence.
        _FakePart(
            function_call=_FakeFunctionCall(
                id="fc-3", name="report_task_started", args={"task_id": "t"}
            ),
        ),
    ]
    ctx = _FakeReadonlyContext(state={})

    out = planner.process_planning_response(ctx, parts)

    # All three parts retained; divergence is signal-only.
    assert out is not None
    names = [p.function_call.name for p in out]
    assert names == ["researcher", "rogue_agent", "report_task_started"]

    # Yield once so the scheduled _handle_drift task runs.
    await asyncio.sleep(0)

    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.PLAN_DIVERGENCE
    assert "rogue_agent" in drift.detail
    assert drift.current_agent_id == "rogue_agent"


async def test_goldfive_planner_process_response_composes_user_planner() -> None:
    """User planner's process runs after goldfive's structural filtering."""
    transformed = [_FakePart(text="transformed_by_user")]
    user_planner = _make_user_planner(process_return=transformed)

    planner = GoldfivePlanner(user_planner=user_planner)
    state = {KEY_CANCELLED_FUNCTION_CALL_IDS: ["fc-bad"]}
    parts = [
        _FakePart(function_call=_FakeFunctionCall(id="fc-bad", name="x")),
        _FakePart(function_call=_FakeFunctionCall(id="fc-ok", name="y")),
    ]
    ctx = _FakeReadonlyContext(state=state)

    out = planner.process_planning_response(ctx, parts)

    # User planner saw the already-filtered list (fc-bad removed).
    assert len(user_planner.recorder.process_calls) == 1
    call_ctx, call_parts = user_planner.recorder.process_calls[0]
    ids_passed_to_user = [p.function_call.id for p in call_parts if p.function_call]
    assert ids_passed_to_user == ["fc-ok"]

    # User planner's return wins the final output.
    assert list(out) == list(transformed)
    assert out[0].text == "transformed_by_user"


# ---------------------------------------------------------------------------
# Auto-attachment in goldfive.wrap (via ADKAdapter construction)
# ---------------------------------------------------------------------------


def test_auto_attachment_attaches_goldfive_planner_to_single_agent() -> None:
    """Single LlmAgent gets a GoldfivePlanner by default."""
    agent = _make_llm_agent()
    assert agent.planner is None
    _attach_goldfive_planner_to_tree(agent)
    assert isinstance(agent.planner, GoldfivePlanner)
    assert agent.planner.user_planner is None


def test_auto_attachment_composes_when_user_planner_already_set() -> None:
    """Existing user planner is preserved via ``user_planner=`` composition."""
    agent = _make_llm_agent()
    user_planner = _make_user_planner()
    agent.planner = user_planner
    _attach_goldfive_planner_to_tree(agent)
    assert isinstance(agent.planner, GoldfivePlanner)
    assert agent.planner.user_planner is user_planner


def test_auto_attachment_is_idempotent() -> None:
    """Re-running attachment does not re-wrap a GoldfivePlanner."""
    agent = _make_llm_agent()
    _attach_goldfive_planner_to_tree(agent)
    first = agent.planner
    _attach_goldfive_planner_to_tree(agent)
    assert agent.planner is first
    # The second pass didn't wrap the first-pass planner as a user_planner.
    assert agent.planner.user_planner is None


def test_opt_out_marker_skips_attachment() -> None:
    """Agents with ``_goldfive_planner_opt_out = True`` stay untouched."""
    agent = _make_llm_agent()
    setattr(agent, GOLDFIVE_PLANNER_OPT_OUT_ATTR, True)
    _attach_goldfive_planner_to_tree(agent)
    assert agent.planner is None


def test_auto_attachment_covers_flat_specialists_tree() -> None:
    """Every LlmAgent in a flat specialists tree gets a GoldfivePlanner."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    a = _make_llm_agent("a")
    b = _make_llm_agent("b")
    c = _make_llm_agent("c")
    root = LlmAgent(
        name="root",
        model="fake-model",
        description="root",
        instruction="root",
        sub_agents=[a, b, c],
    )
    _attach_goldfive_planner_to_tree(root)
    for node in (root, a, b, c):
        assert isinstance(node.planner, GoldfivePlanner), (
            f"{node.name} did not receive a GoldfivePlanner"
        )


def test_auto_attachment_covers_deep_hierarchy() -> None:
    """LlmAgents nested at depth 3+ get a GoldfivePlanner."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    leaf = _make_llm_agent("leaf")
    mid = LlmAgent(
        name="mid",
        model="fake-model",
        description="mid",
        instruction="mid",
        sub_agents=[leaf],
    )
    inner = LlmAgent(
        name="inner",
        model="fake-model",
        description="inner",
        instruction="inner",
        sub_agents=[mid],
    )
    root = LlmAgent(
        name="root",
        model="fake-model",
        description="root",
        instruction="root",
        sub_agents=[inner],
    )
    _attach_goldfive_planner_to_tree(root)
    for node in (root, inner, mid, leaf):
        assert isinstance(node.planner, GoldfivePlanner)


def test_auto_attachment_skips_opt_out_but_walks_subtree() -> None:
    """Opt-out marker skips only the marked agent; children still get attached."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    child = _make_llm_agent("child")
    marked = LlmAgent(
        name="marked",
        model="fake-model",
        description="marked",
        instruction="marked",
        sub_agents=[child],
    )
    setattr(marked, GOLDFIVE_PLANNER_OPT_OUT_ATTR, True)
    _attach_goldfive_planner_to_tree(marked)
    assert marked.planner is None
    assert isinstance(child.planner, GoldfivePlanner)


def test_rebind_populates_agent_registry_and_steerer() -> None:
    """``_rebind_goldfive_planners`` flows the runtime collaborators in."""
    agent = _make_llm_agent()
    _attach_goldfive_planner_to_tree(agent)
    planner = agent.planner
    assert isinstance(planner, GoldfivePlanner)
    assert planner._agent_registry is None
    assert planner._steerer is None

    steerer = _RecordingSteerer()
    session = Session(run_id="r")
    _rebind_goldfive_planners(
        agent,
        agent_registry=["agent_under_test", "helper"],
        steerer=steerer,
        session=session,
    )
    assert planner._agent_registry == {"agent_under_test", "helper"}
    assert planner._steerer is steerer
    assert planner._session is session


def test_adk_adapter_init_attaches_goldfive_planner_to_tree() -> None:
    """Constructing ``ADKAdapter`` auto-attaches GoldfivePlanner (non-degraded)."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    leaf = _make_llm_agent("leaf")
    root = LlmAgent(
        name="root",
        model="fake-model",
        description="root",
        instruction="root",
        sub_agents=[leaf],
    )
    ADKAdapter(root)
    assert isinstance(root.planner, GoldfivePlanner)
    assert isinstance(leaf.planner, GoldfivePlanner)


# ---------------------------------------------------------------------------
# Plugin-side request injection
# ---------------------------------------------------------------------------


class _FakeLlmRequestConfig:
    def __init__(self) -> None:
        self.system_instruction: str | None = None


class _FakeLlmRequest:
    """Minimal LlmRequest stub exposing ``append_instructions`` + ``config``."""

    def __init__(self) -> None:
        self.config = _FakeLlmRequestConfig()
        self.calls: list[list[str]] = []

    def append_instructions(self, instructions: list[str]) -> list:
        self.calls.append(list(instructions))
        if not self.config.system_instruction:
            self.config.system_instruction = "\n\n".join(instructions)
        else:
            self.config.system_instruction += "\n\n" + "\n\n".join(instructions)
        return []


class _FakeInvocationContext:
    def __init__(self, *, agent: Any, state: dict) -> None:
        self.agent = agent

        class _S:
            def __init__(self, st):
                self.state = st

        self.session = _S(state)
        self.invocation_id = "inv-1"
        self.user_content = None
        self.user_id = "u"
        self.branch = None
        self.run_config = None


class _FakeCallbackContext:
    """Stand-in for ADK's CallbackContext carrying ``_invocation_context``."""

    def __init__(self, *, inv_ctx: _FakeInvocationContext) -> None:
        self._invocation_context = inv_ctx
        self.session = inv_ctx.session


async def test_plugin_before_model_callback_injects_system_instruction() -> None:
    """The plugin's ``before_model_callback`` appends GoldfivePlanner output."""
    agent = _make_llm_agent()
    _attach_goldfive_planner_to_tree(agent)

    state = {
        KEY_CURRENT_TASK_ID: "t-42",
        KEY_CURRENT_TASK_TITLE: "Prepare brief",
    }
    inv_ctx = _FakeInvocationContext(agent=agent, state=state)
    cb = _FakeCallbackContext(inv_ctx=inv_ctx)
    llm_request = _FakeLlmRequest()

    await _inject_goldfive_planner_instruction(
        callback_context=cb,
        llm_request=llm_request,
    )

    assert llm_request.config.system_instruction is not None
    assert "[GOLDFIVE ORCHESTRATION CONTEXT]" in llm_request.config.system_instruction
    assert "t-42" in llm_request.config.system_instruction
    assert "Prepare brief" in llm_request.config.system_instruction


async def test_plugin_skips_injection_for_plan_re_act_planner() -> None:
    """Agent carrying a PlanReActPlanner is NOT double-injected by the plugin."""
    from google.adk.planners.plan_re_act_planner import PlanReActPlanner  # type: ignore

    agent = _make_llm_agent()
    agent.planner = PlanReActPlanner()
    # NOTE: we do NOT run _attach_goldfive_planner_to_tree — this test
    # asks the plugin to skip even when the planner is NOT a
    # GoldfivePlanner. ADK handles PlanReActPlanner natively via
    # _nl_planning.py — the plugin's hook must be a no-op so the
    # instruction isn't injected twice.

    inv_ctx = _FakeInvocationContext(agent=agent, state={})
    cb = _FakeCallbackContext(inv_ctx=inv_ctx)
    llm_request = _FakeLlmRequest()

    await _inject_goldfive_planner_instruction(
        callback_context=cb,
        llm_request=llm_request,
    )

    assert llm_request.config.system_instruction is None
    assert llm_request.calls == []


async def test_plugin_skips_injection_when_no_planner() -> None:
    """Agent with no planner attached — plugin no-ops."""
    agent = _make_llm_agent()
    assert agent.planner is None

    inv_ctx = _FakeInvocationContext(agent=agent, state={})
    cb = _FakeCallbackContext(inv_ctx=inv_ctx)
    llm_request = _FakeLlmRequest()

    await _inject_goldfive_planner_instruction(
        callback_context=cb,
        llm_request=llm_request,
    )
    assert llm_request.config.system_instruction is None


async def test_plugin_skips_injection_for_custom_base_planner_subclass() -> None:
    """A user BasePlanner subclass (not GoldfivePlanner) is skipped by the hook.

    Rationale: the hook is the GoldfivePlanner-specific workaround for
    ADK's PlanReActPlanner-only request gate. Custom BasePlanner
    subclasses are expected to work via ADK's response-side dispatch
    only; the plugin must not unilaterally inject for them.
    """
    from google.adk.planners.base_planner import BasePlanner  # type: ignore

    class _MyPlanner(BasePlanner):
        def build_planning_instruction(self, ro, req):
            return "USER_BLOCK"

        def process_planning_response(self, cb, parts):
            return None

    agent = _make_llm_agent()
    agent.planner = _MyPlanner()

    inv_ctx = _FakeInvocationContext(agent=agent, state={})
    cb = _FakeCallbackContext(inv_ctx=inv_ctx)
    llm_request = _FakeLlmRequest()
    await _inject_goldfive_planner_instruction(callback_context=cb, llm_request=llm_request)
    assert llm_request.config.system_instruction is None


async def test_plugin_injection_via_full_before_model_callback() -> None:
    """End-to-end: running the plugin's real ``before_model_callback`` injects."""
    plugin = make_adk_plugin(host_agent_name="h")
    agent = _make_llm_agent()
    _attach_goldfive_planner_to_tree(agent)

    session = Session(run_id="r")
    state = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=None,
            task=Task(id="t-99", title="Build the thing"),
            tool_handlers={},
            host_agent_name="h",
        )
    }

    inv_ctx = _FakeInvocationContext(agent=agent, state=state)
    cb = _FakeCallbackContext(inv_ctx=inv_ctx)
    llm_request = _FakeLlmRequest()

    await plugin.before_model_callback(
        callback_context=cb,
        llm_request=llm_request,
    )

    # before_model_callback also re-seeds current_task via the state protocol;
    # assert the instruction injection picked the new state up.
    assert llm_request.config.system_instruction is not None
    assert "t-99" in llm_request.config.system_instruction
    assert "Build the thing" in llm_request.config.system_instruction


async def test_plugin_injection_skips_built_in_planner() -> None:
    """BuiltInPlanner (thinking-config only) is not injected for by the plugin."""
    from google.adk.planners.built_in_planner import BuiltInPlanner  # type: ignore
    from google.genai import types  # type: ignore

    agent = _make_llm_agent()
    agent.planner = BuiltInPlanner(thinking_config=types.ThinkingConfig())

    inv_ctx = _FakeInvocationContext(agent=agent, state={})
    cb = _FakeCallbackContext(inv_ctx=inv_ctx)
    llm_request = _FakeLlmRequest()
    await _inject_goldfive_planner_instruction(callback_context=cb, llm_request=llm_request)
    assert llm_request.config.system_instruction is None
