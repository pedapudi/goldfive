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
from goldfive.types import DriftEvent, DriftKind, Plan, Session, Task  # noqa: E402

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


@pytest.mark.skip(reason="goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH (#253)")
async def test_goldfive_planner_process_response_emits_divergence_on_off_registry_agent() -> None:
    """Three-stage classification: cross-layer → PLAN_DIVERGENCE;
    hallucinated → CONFABULATION_RISK; report_ prefix → skipped.

    Ctx has no ``_invocation_context`` so ``_extract_own_tool_names``
    returns an empty set — every function_call falls through stage 1
    and is classified against the registry (stage 2) or as
    hallucination (stage 3).
    """
    steerer = _RecordingSteerer()
    session = Session(run_id="r")
    planner = GoldfivePlanner(
        agent_registry=["researcher", "writer"],
        steerer=steerer,
        session=session,
    )
    parts = [
        # Stage 2 — name matches a registry agent but wasn't exposed
        # as a tool to this agent: PLAN_DIVERGENCE.
        _FakePart(
            function_call=_FakeFunctionCall(id="fc-1", name="researcher", args={}),
        ),
        # Stage 3 — name is neither a tool nor a known agent:
        # CONFABULATION_RISK.
        _FakePart(
            function_call=_FakeFunctionCall(id="fc-2", name="rogue_agent", args={}),
        ),
        # Reporting tool — must not trigger any drift.
        _FakePart(
            function_call=_FakeFunctionCall(
                id="fc-3", name="report_task_started", args={"task_id": "t"}
            ),
        ),
    ]
    ctx = _FakeReadonlyContext(state={})

    out = planner.process_planning_response(ctx, parts)

    # All three parts retained; classification is signal-only.
    assert out is not None
    names = [p.function_call.name for p in out]
    assert names == ["researcher", "rogue_agent", "report_task_started"]

    # Yield once so the scheduled _handle_drift tasks run.
    await asyncio.sleep(0)

    # Two drifts: stage 2 (PLAN_DIVERGENCE) + stage 3 (CONFABULATION_RISK).
    by_kind = {d.kind: d for d in steerer.drifts}
    assert DriftKind.PLAN_DIVERGENCE in by_kind, (
        f"expected PLAN_DIVERGENCE drift, got {steerer.drifts!r}"
    )
    assert DriftKind.CONFABULATION_RISK in by_kind, (
        f"expected CONFABULATION_RISK drift, got {steerer.drifts!r}"
    )
    div = by_kind[DriftKind.PLAN_DIVERGENCE]
    assert "researcher" in div.detail
    assert div.current_agent_id == "researcher"
    assert div.severity.value == "warning"

    conf = by_kind[DriftKind.CONFABULATION_RISK]
    assert "rogue_agent" in conf.detail
    assert conf.current_agent_id == "rogue_agent"
    assert conf.severity.value == "warning"


class _FakeTool:
    """Duck-typed ADK tool exposing a ``.name`` attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAgent:
    """Duck-typed ADK agent exposing ``.tools`` for the planner's own-tool lookup."""

    def __init__(self, *, name: str, tool_names: list[str]) -> None:
        self.name = name
        self.tools = [_FakeTool(n) for n in tool_names]


class _FakeInvocationContextAgent:
    """Stand-in for ``CallbackContext._invocation_context`` carrying an agent."""

    def __init__(self, *, agent: _FakeAgent, state: dict) -> None:
        self.agent = agent

        class _S:
            def __init__(self, st):
                self.state = st

        self.session = _S(state)


class _FakeCallbackContextWithAgent:
    """CallbackContext stand-in wiring ``_invocation_context.agent.tools``.

    Also satisfies the state-extraction contract via
    ``._invocation_context.session.state`` so the cancelled-id filter
    still exercises the real chain.
    """

    def __init__(self, *, agent: _FakeAgent, state: dict | None = None) -> None:
        self._invocation_context = _FakeInvocationContextAgent(agent=agent, state=state or {})
        # Mirror ADK CallbackContext behaviour: a direct ``.state`` alias
        # for simple readers.
        self.state = state or {}


async def test_own_tool_call_no_drift() -> None:
    """Stage 1: function_call name in agent's own tools → no drift, part retained.

    A ``web_developer_agent`` exposes ``write_webpage`` directly on its
    tool list. An LLM turn emitting ``function_call(name="write_webpage")``
    must NOT fire any drift — this is the over-firing bug #178 fixed.
    """
    steerer = _RecordingSteerer()
    session = Session(run_id="r")
    planner = GoldfivePlanner(
        agent_registry=["web_developer_agent", "research_agent", "coordinator_agent"],
        steerer=steerer,
        session=session,
    )
    agent = _FakeAgent(
        name="web_developer_agent",
        tool_names=["write_webpage", "read_presentation_files", "patch_file"],
    )
    ctx = _FakeCallbackContextWithAgent(agent=agent)
    parts = [
        _FakePart(function_call=_FakeFunctionCall(id="fc-1", name="write_webpage")),
        _FakePart(function_call=_FakeFunctionCall(id="fc-2", name="patch_file")),
    ]

    out = planner.process_planning_response(ctx, parts)

    # No drift fired — all function_calls are the agent's own tools.
    await asyncio.sleep(0)
    assert steerer.drifts == [], f"expected no drifts for own-tool calls, got {steerer.drifts!r}"

    # Parts retained verbatim. When no cancellations and no drift
    # classification fired, the planner returns ``None`` (ADK skip-flag)
    # to preserve the original response untouched.
    assert out is None


@pytest.mark.skip(reason="goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH (#253)")
async def test_cross_layer_agent_call_fires_plan_divergence() -> None:
    """Stage 2: name matches a registry agent but is not in this agent's tools.

    ``research_agent`` (the current agent) has its own tools list
    without ``coordinator_agent``. The coordinator IS in the tree's
    registry — so emitting ``function_call(name="coordinator_agent")``
    from research_agent's LLM is a cross-layer delegation attempt →
    PLAN_DIVERGENCE (WARNING).
    """
    steerer = _RecordingSteerer()
    session = Session(run_id="r")
    planner = GoldfivePlanner(
        agent_registry=["research_agent", "coordinator_agent", "writer_agent"],
        steerer=steerer,
        session=session,
    )
    agent = _FakeAgent(
        name="research_agent",
        tool_names=["web_search", "read_file"],  # NOT coordinator_agent
    )
    ctx = _FakeCallbackContextWithAgent(agent=agent)
    parts = [
        _FakePart(function_call=_FakeFunctionCall(id="fc-1", name="coordinator_agent")),
    ]

    out = planner.process_planning_response(ctx, parts)

    await asyncio.sleep(0)
    assert len(steerer.drifts) == 1, f"expected exactly 1 drift, got {steerer.drifts!r}"
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.PLAN_DIVERGENCE
    assert drift.severity.value == "warning"
    assert "coordinator_agent" in drift.detail
    assert drift.current_agent_id == "coordinator_agent"

    # Part retained; classification is signal-only.
    assert out is not None
    assert [p.function_call.name for p in out if p.function_call] == ["coordinator_agent"]


async def test_hallucinated_tool_fires_confabulation_risk() -> None:
    """Stage 3: name is neither a tool nor a known agent → CONFABULATION_RISK.

    ``flux_capacitor_42`` is in no agent's tool list and not in the
    tree's registry — pure hallucination. Fire CONFABULATION_RISK at
    WARNING.
    """
    steerer = _RecordingSteerer()
    session = Session(run_id="r")
    planner = GoldfivePlanner(
        agent_registry=["research_agent", "writer_agent"],
        steerer=steerer,
        session=session,
    )
    agent = _FakeAgent(
        name="writer_agent",
        tool_names=["write_text", "read_text"],
    )
    ctx = _FakeCallbackContextWithAgent(agent=agent)
    parts = [
        _FakePart(function_call=_FakeFunctionCall(id="fc-1", name="flux_capacitor_42")),
    ]

    out = planner.process_planning_response(ctx, parts)

    await asyncio.sleep(0)
    assert len(steerer.drifts) == 1, f"expected exactly 1 drift, got {steerer.drifts!r}"
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.CONFABULATION_RISK
    assert drift.severity.value == "warning"
    assert "flux_capacitor_42" in drift.detail
    assert drift.current_agent_id == "flux_capacitor_42"

    # Part retained.
    assert out is not None
    assert [p.function_call.name for p in out if p.function_call] == ["flux_capacitor_42"]


async def test_cancelled_id_stripped_across_all_stages() -> None:
    """Cancelled-id filter runs regardless of drift stage.

    Mix parts that would fall in stages 1/2/3 AND carry cancelled ids;
    verify every cancelled id is stripped and the remaining parts flow
    into the drift classifier exactly once each.
    """
    steerer = _RecordingSteerer()
    session = Session(run_id="r")
    planner = GoldfivePlanner(
        agent_registry=["coordinator_agent", "web_developer_agent"],
        steerer=steerer,
        session=session,
    )
    agent = _FakeAgent(
        name="web_developer_agent",
        tool_names=["write_webpage"],
    )
    state = {
        KEY_CANCELLED_FUNCTION_CALL_IDS: [
            "fc-cancel-stage1",
            "fc-cancel-stage2",
            "fc-cancel-stage3",
        ],
    }
    ctx = _FakeCallbackContextWithAgent(agent=agent, state=state)

    parts = [
        # Cancelled stage-1 (own tool) — stripped.
        _FakePart(function_call=_FakeFunctionCall(id="fc-cancel-stage1", name="write_webpage")),
        # Retained stage-1 (own tool) — no drift.
        _FakePart(function_call=_FakeFunctionCall(id="fc-ok-1", name="write_webpage")),
        # Cancelled stage-2 (cross-layer agent) — stripped before classification.
        _FakePart(function_call=_FakeFunctionCall(id="fc-cancel-stage2", name="coordinator_agent")),
        # Retained stage-2 — PLAN_DIVERGENCE.
        _FakePart(function_call=_FakeFunctionCall(id="fc-ok-2", name="coordinator_agent")),
        # Cancelled stage-3 (hallucination) — stripped before classification.
        _FakePart(function_call=_FakeFunctionCall(id="fc-cancel-stage3", name="made_up_tool")),
        # Retained stage-3 — CONFABULATION_RISK.
        _FakePart(function_call=_FakeFunctionCall(id="fc-ok-3", name="made_up_tool")),
    ]

    out = planner.process_planning_response(ctx, parts)

    # Every cancelled id removed from the output across all three stages.
    assert out is not None
    remaining_ids = [p.function_call.id for p in out if p.function_call]
    assert remaining_ids == ["fc-ok-1", "fc-ok-2", "fc-ok-3"]
    assert "fc-cancel-stage1" not in remaining_ids
    assert "fc-cancel-stage2" not in remaining_ids
    assert "fc-cancel-stage3" not in remaining_ids

    # Drifts fired only for the retained hallucinated part — the
    # cancelled ones never reach classification because the cancelled-id
    # filter runs first.
    #
    # goldfive#252: PLAN_DIVERGENCE is silenced (replaced by
    # CAPABILITY_MISMATCH in #253); the retained stage-2 cross-layer
    # part is still classified but produces no drift. Only the stage-3
    # hallucination still fires CONFABULATION_RISK.
    await asyncio.sleep(0)
    kinds = sorted(d.kind.value for d in steerer.drifts)
    assert kinds == ["confabulation_risk"], (
        f"expected one CONFABULATION_RISK (PLAN_DIVERGENCE silenced per "
        f"#252), got {steerer.drifts!r}"
    )


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
    """End-to-end: running the plugin's real ``before_model_callback`` injects.

    Phase 2.0 of goldfive#271 — the planner reads goldfive
    ``Session.state`` directly via the ``SessionContext`` stash. The
    test pins the current task on goldfive ``Session.state`` (the same
    surface V3's ``before_agent_callback`` writes in production); the
    planner's injection picks it up via ``StateStore``.
    """
    from goldfive import state_store as _ostate

    plugin = make_adk_plugin(host_agent_name="h")
    agent = _make_llm_agent()
    _attach_goldfive_planner_to_tree(agent)

    task = Task(id="t-99", title="Build the thing")
    session = Session(
        run_id="r",
        plan=Plan(
            id="p1",
            run_id="r",
            goal_ids=[],
            tasks=[task],
            edges=[],
        ),
    )
    # Pin on goldfive Session.state — the surface V3's
    # ``before_agent_callback`` writes in production. The planner
    # reads pin / title via ``StateStore.for_session``.
    _ostate.set_current_task(session.state, task)

    state = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=None,
            task=task,
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
