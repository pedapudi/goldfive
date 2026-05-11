"""Structural capability-mismatch detector (goldfive#253).

Covers :func:`goldfive.drift.capability_check.detect_capability_mismatch`
plus the integration wiring through the ADK adapter's
``before_tool_callback`` so a delegation to an under-qualified agent
fires a CRITICAL ``CAPABILITY_MISMATCH`` drift through the steerer's
``_handle_drift`` hook.

Detection rules under test:

* Rule A — coordinator-style leaf-assignment: agent has only AgentTool
  wrappers; bound task uses leaf-task verbs ("draft", "write", ...).
* Rule B — task ``required_tools`` is non-empty and the invoked agent
  doesn't cover the required names.

Negative cases mirror each positive: the same shapes WITHOUT the
trigger condition must return ``None``.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.drift.capability_check import (
    DELEGATION_VERB_MARKERS,
    detect_capability_mismatch,
    is_agent_tool,
)
from goldfive.types import (
    DriftKind,
    DriftSeverity,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Lightweight tool stubs (no ADK import required for unit tests).
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal ADK BaseAgent stand-in — only ``.name`` is read."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAgentTool:
    """AgentTool stub. Detector picks this up via the ``.agent`` duck-type
    fallback when google.adk is not importable in the test env.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent = _FakeAgent(agent_name)
        self.name = agent_name


class _FakeFunctionTool:
    """FunctionTool stub. No ``.agent`` attribute, so it never matches
    Rule A's "all-AgentTool" predicate.
    """

    def __init__(self, name: str) -> None:
        self.name = name

        def _func() -> None:
            return None

        _func.__name__ = name
        self.func = _func


# ---------------------------------------------------------------------------
# Rule A — coordinator-style leaf-assignment
# ---------------------------------------------------------------------------


def test_rule_a_positive_only_agent_tools_on_leaf_task() -> None:
    """An agent whose tools are ALL AgentTool wrappers, bound to a leaf
    authoring task ("Draft a presentation"), must fire CAPABILITY_MISMATCH.
    """
    agent_tools: list[Any] = [_FakeAgentTool("writer"), _FakeAgentTool("reviewer")]
    task = Task(
        id="t-draft",
        title="Draft a presentation about LLM observability",
        description="Produce slides covering the goldfive design.",
        assignee_agent_id="coordinator",
    )

    drift = detect_capability_mismatch(
        invoked_agent_name="coordinator",
        invoked_agent_tools=agent_tools,
        task=task,
    )

    assert drift is not None, "Rule A must fire for coordinator-only on leaf task"
    assert drift.kind is DriftKind.CAPABILITY_MISMATCH
    assert drift.severity is DriftSeverity.CRITICAL
    assert drift.current_task_id == "t-draft"
    assert drift.current_agent_id == "coordinator"
    # Detail names the structural gap so operators can read it.
    assert "AgentTool" in drift.detail
    assert "coordinator" in drift.detail


def test_rule_a_negative_agent_has_function_tool() -> None:
    """An agent with a real FunctionTool (e.g. ``write_webpage``) is
    NOT all-AgentTool, so Rule A is suppressed even on a leaf task.
    """
    agent_tools: list[Any] = [_FakeFunctionTool("write_webpage")]
    task = Task(id="t-draft", title="Draft a presentation about X")

    drift = detect_capability_mismatch(
        invoked_agent_name="writer",
        invoked_agent_tools=agent_tools,
        task=task,
    )

    assert drift is None, "Rule A must NOT fire when agent has a leaf-capability tool"


def test_rule_a_negative_delegation_verb_in_title() -> None:
    """A task whose title uses a delegation verb ("Coordinate the
    workflow") matches the agent's actual capability; Rule A is
    suppressed even though the agent is all-AgentTool.
    """
    agent_tools: list[Any] = [_FakeAgentTool("writer")]
    task = Task(id="t-coord", title="Coordinate the presentation workflow")

    drift = detect_capability_mismatch(
        invoked_agent_name="coordinator",
        invoked_agent_tools=agent_tools,
        task=task,
    )

    assert drift is None, "Rule A must NOT fire when task is itself delegation-shaped"


@pytest.mark.parametrize(
    "verb",
    list(DELEGATION_VERB_MARKERS),
)
def test_rule_a_suppressed_for_each_delegation_marker(verb: str) -> None:
    """Every documented delegation marker must suppress Rule A."""
    agent_tools: list[Any] = [_FakeAgentTool("sub")]
    task = Task(id="t-x", title=f"Please {verb} the work to the right team")

    drift = detect_capability_mismatch(
        invoked_agent_name="coordinator",
        invoked_agent_tools=agent_tools,
        task=task,
    )

    assert drift is None, f"Rule A must be suppressed by marker {verb!r}"


def test_rule_a_negative_review_research_are_leaf_verbs() -> None:
    """"Review" and "research" are LEAF-task verbs — a coordinator
    structurally cannot do them. Self-review guard from the brief.
    """
    agent_tools: list[Any] = [_FakeAgentTool("writer"), _FakeAgentTool("reader")]

    review_task = Task(id="t-r1", title="Review the draft slides")
    review_drift = detect_capability_mismatch(
        invoked_agent_name="coordinator",
        invoked_agent_tools=agent_tools,
        task=review_task,
    )
    assert review_drift is not None
    assert review_drift.kind is DriftKind.CAPABILITY_MISMATCH

    research_task = Task(id="t-r2", title="Research the latest LLM benchmarks")
    research_drift = detect_capability_mismatch(
        invoked_agent_name="coordinator",
        invoked_agent_tools=agent_tools,
        task=research_task,
    )
    assert research_drift is not None
    assert research_drift.kind is DriftKind.CAPABILITY_MISMATCH


# ---------------------------------------------------------------------------
# Rule B — required_tools advisory
# ---------------------------------------------------------------------------


def test_rule_b_positive_required_tool_missing() -> None:
    """``Task.required_tools`` is non-empty and the agent's tool names
    do not cover the required names — fire CRITICAL.
    """
    agent_tools: list[Any] = [_FakeFunctionTool("read_presentation_files")]
    task = Task(
        id="t-write",
        title="Author the slides",
        required_tools=("write_webpage",),
    )

    drift = detect_capability_mismatch(
        invoked_agent_name="reader",
        invoked_agent_tools=agent_tools,
        task=task,
    )

    assert drift is not None
    assert drift.kind is DriftKind.CAPABILITY_MISMATCH
    assert drift.severity is DriftSeverity.CRITICAL
    assert "write_webpage" in drift.detail


def test_rule_b_negative_required_tool_present() -> None:
    """Required tool is in the agent's tool surface — no drift."""
    agent_tools: list[Any] = [_FakeFunctionTool("write_webpage")]
    task = Task(
        id="t-write",
        title="Author the slides",
        required_tools=("write_webpage",),
    )

    drift = detect_capability_mismatch(
        invoked_agent_name="writer",
        invoked_agent_tools=agent_tools,
        task=task,
    )

    assert drift is None


def test_rule_b_skipped_when_required_tools_empty() -> None:
    """Empty ``required_tools`` is the legacy / unset default — Rule B
    is a no-op regardless of the agent's tool surface.
    """
    # Even with zero tools and a leaf-shaped task, Rule B alone wouldn't
    # fire here. Use a task without delegation language so Rule A would
    # otherwise be in play; force the agent to have something
    # non-AgentTool so Rule A is also out.
    agent_tools: list[Any] = [_FakeFunctionTool("some_fn")]
    task = Task(id="t-leg", title="Draft something", required_tools=())

    drift = detect_capability_mismatch(
        invoked_agent_name="writer",
        invoked_agent_tools=agent_tools,
        task=task,
    )

    assert drift is None


# ---------------------------------------------------------------------------
# Empty-input safety
# ---------------------------------------------------------------------------


def test_empty_inputs_return_none() -> None:
    """No tools + no required_tools -> nothing to decide -> None."""
    task = Task(id="t-empty", title="")
    drift = detect_capability_mismatch(
        invoked_agent_name="ghost",
        invoked_agent_tools=[],
        task=task,
    )
    assert drift is None


def test_none_task_returns_none() -> None:
    """Defensive: a missing task (e.g. plan was never built) is a no-op."""
    drift = detect_capability_mismatch(
        invoked_agent_name="any",
        invoked_agent_tools=[_FakeAgentTool("x")],
        task=None,  # type: ignore[arg-type]
    )
    assert drift is None


def test_is_agent_tool_duck_type() -> None:
    """``is_agent_tool`` falls back to the ``.agent`` duck-type when ADK
    is not importable; FunctionTool stubs without ``.agent`` are not
    AgentTools.
    """
    assert is_agent_tool(_FakeAgentTool("x")) is True
    assert is_agent_tool(_FakeFunctionTool("y")) is False
    assert is_agent_tool(None) is False


# ---------------------------------------------------------------------------
# Integration — ADK adapter wires the detector through ``_handle_drift``
# ---------------------------------------------------------------------------


pytest.importorskip("google.adk")


class _RecordingSteerer:
    """Test stub matching ``test_runaway_delegation_cap``: records every
    drift routed through ``_handle_drift`` so the test can assert on
    CAPABILITY_MISMATCH emission without spinning up a full
    DefaultSteerer.
    """

    def __init__(self) -> None:
        self._sinks: list[Any] = []
        self.drifts: list[Any] = []

    async def observe(self, *a: Any, **kw: Any) -> None:
        pass

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, *a: Any, **kw: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass

    async def _handle_drift(self, drift: Any, session: Any) -> None:  # noqa: ARG002
        self.drifts.append(drift)


def _make_one_shot_delegating_llm(tool_name: str) -> Any:
    """LLM that fires one AgentTool call then yields a final text."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _OneShot(BaseLlm):
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
                                    id="c1",
                                    name=tool_name,
                                    args={"request": "go"},
                                )
                            ),
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="done")],
                    ),
                    turn_complete=True,
                )

    return _OneShot


def _make_quiet_llm() -> Any:
    """Sub-agent LLM that just says "ok" and terminates."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _Quiet(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                turn_complete=True,
            )

    return _Quiet


async def test_integration_capability_mismatch_flows_through_handle_drift() -> None:
    """End-to-end: a coordinator delegates to a sub-agent whose only
    tools are AgentTool wrappers (so it cannot perform a leaf task).
    The plan task is assigned to that sub-agent. The capability detector
    fires and the drift lands on the steerer's ``_handle_drift``.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    # Inner-most leaf: never invoked, just needed so AgentTool wraps an
    # actual ADK BaseAgent.
    inner = Agent(name="inner", model=_make_quiet_llm()(), instruction="")
    # Underqualified sub-agent: only tool is an AgentTool wrapping inner.
    underqualified = Agent(
        name="underqualified",
        model=_make_quiet_llm()(),
        instruction="",
        tools=[AgentTool(inner)],
    )
    # Coordinator delegates ONCE to the underqualified sub-agent.
    coord = Agent(
        name="coord",
        model=_make_one_shot_delegating_llm("underqualified")(),
        instruction="",
        tools=[AgentTool(underqualified)],
    )
    adapter = ADKAdapter(coord)
    steerer = _RecordingSteerer()
    adapter.bind_steerer(steerer)

    # Plan task is the leaf authoring task assigned to the
    # underqualified sub-agent — exactly the structural mismatch.
    # goldfive#259: the coord task is marked RUNNING (not PENDING) so
    # the observational delegation-time pin (#259) only treats the leaf
    # task as eligible. Otherwise the pin would pick the
    # delegation-shaped coord_task by plan order (zero-overlap topic
    # match falls back to first) and Rule A's "Coordinate" delegation
    # marker would suppress the drift.
    leaf_task = Task(
        id="t-draft",
        title="Draft a presentation about LLM observability",
        assignee_agent_id="underqualified",
    )
    coord_task = Task(
        id="t-coord",
        title="Coordinate the presentation",
        assignee_agent_id="coord",
        status=TaskStatus.RUNNING,
    )
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=(),
        tasks=(coord_task, leaf_task),
        edges=(),
    )
    session = Session(run_id="r1", plan=plan)

    await adapter.invoke(task=coord_task, session=session)

    # The capability-mismatch drift reached the steerer.
    capability_drifts = [
        d for d in steerer.drifts if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert len(capability_drifts) >= 1, (
        f"expected at least one CAPABILITY_MISMATCH drift; got "
        f"{[d.kind for d in steerer.drifts]}"
    )
    drift = capability_drifts[0]
    assert drift.severity is DriftSeverity.CRITICAL
    assert drift.current_task_id == "t-draft"
    assert drift.current_agent_id == "underqualified"


async def test_integration_no_capability_mismatch_when_agent_has_leaf_tool() -> None:
    """Same coordinator+delegation shape, but the sub-agent has a real
    FunctionTool — Rule A no longer applies and no CAPABILITY_MISMATCH
    fires.
    """
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    def write_webpage(content: str) -> dict[str, Any]:  # noqa: ARG001
        return {"ok": True}

    qualified = Agent(
        name="qualified",
        model=_make_quiet_llm()(),
        instruction="",
        tools=[FunctionTool(write_webpage)],
    )
    coord = Agent(
        name="coord",
        model=_make_one_shot_delegating_llm("qualified")(),
        instruction="",
        tools=[AgentTool(qualified)],
    )
    adapter = ADKAdapter(coord)
    steerer = _RecordingSteerer()
    adapter.bind_steerer(steerer)

    # goldfive#259: mark coord_task RUNNING (not PENDING) so only the
    # leaf task is eligible for the observational delegation-time pin.
    # Otherwise the pin would pick coord_task by plan order (zero-
    # overlap topic match → first eligible), and Rule A would be
    # suppressed by the "Coordinate" delegation marker anyway.
    leaf_task = Task(
        id="t-draft",
        title="Draft a presentation",
        assignee_agent_id="qualified",
    )
    coord_task = Task(
        id="t-coord",
        title="Coordinate the work",
        assignee_agent_id="coord",
        status=TaskStatus.RUNNING,
    )
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=(),
        tasks=(coord_task, leaf_task),
        edges=(),
    )
    session = Session(run_id="r1", plan=plan)

    await adapter.invoke(task=coord_task, session=session)

    capability_drifts = [
        d for d in steerer.drifts if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert capability_drifts == [], (
        f"Rule A must NOT fire when sub-agent has a leaf-capability "
        f"tool; got {capability_drifts}"
    )
