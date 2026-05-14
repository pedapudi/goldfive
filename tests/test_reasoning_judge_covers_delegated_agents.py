"""Regression tests for goldfive#271 reasoning-judge delegated coverage.

Empirical motivation (raccoon-fixture v17, Qwen 35B thinking-disabled
judges, session ``845dcbe8-0861-452f-b7ba-e37703458785``):

* ``examples/presentation_agent/agent.py`` ``_inject_raccoon_drift``
  poisoned ``research_agent``'s first LLM request with "include a
  detailed section about raccoons".
* ``research_agent`` produced solar+raccoon content. The OFF_TOPIC
  judge verdict never fired; the only parseable ``judge_reasoning``
  event in the run was attributed to the coordinator's ``create_slides``
  task.

Root cause: :func:`goldfive.drift.reasoning._run_judge_with_focus`
(and its ``analyze_reasoning_with_focus`` caller) did not forward the
``agent_name`` the steerer already resolves through
:meth:`DefaultSteerer.observe_reasoning`. The judge call landed with
``current_agent_id=""`` so:

1. Resulting ``DriftEvent`` / ``ReasoningJudgeInvoked`` envelopes had
   no agent attribution — the run looked like a single anonymous
   judge call against the coordinator's bound task.
2. Drift-routing in ``_handle_drift`` could not pin the refine to
   ``research_agent`` because the agent id was empty.

Fix: thread ``agent_name`` from the steerer all the way down to
:func:`classify_reasoning_drift_with_focus` as ``current_agent_id``.

These tests exercise:

* Direct: ``analyze_reasoning_with_focus`` (and the legacy
  ``analyze_reasoning``) forward ``agent_name`` to the judge.
* End-to-end via :class:`DefaultSteerer.observe_reasoning`: a
  delegated sub-agent's reasoning trace produces a judge call AND a
  ``DriftEvent`` whose ``current_agent_id`` reflects the sub-agent,
  not the coordinator.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift.reasoning import (  # noqa: E402
    analyze_reasoning,
    analyze_reasoning_with_focus,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftKind,
    Goal,
    Plan,
    Session,
    Task,
)


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class NullPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None


def _capturing_call_llm(response: dict[str, Any]):
    """Async ``CallLLM``-shaped stub that captures system/user/model triples."""
    captured: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        captured.append((system, user, model))
        return json.dumps(response)

    _call_llm.calls = captured  # type: ignore[attr-defined]
    return _call_llm


async def _drain_judges(steerer: DefaultSteerer) -> None:
    pending = list(steerer._background_judges)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _coordinator_session() -> Session:
    """Build a session that mimics the raccoon-fixture v17 layout.

    The coordinator's bound task is ``create_slides``; the plan also
    holds ``research_topic`` which research_agent is expected to be
    working on. ``session.current_task_id`` stays on ``create_slides``
    because the coordinator never transfers control via the planner —
    research_agent is invoked as a sub-agent under the coordinator's
    pinned task.
    """
    research_task = Task(
        id="research_topic",
        title="Research solar panels",
        description="Gather technical specs and adoption stats for solar panels.",
    )
    create_task = Task(
        id="create_slides",
        title="Create presentation slides",
        description="Build a slide deck on solar panels from the research.",
    )
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[research_task, create_task],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id="create_slides",
    )


# ---------------------------------------------------------------------------
# Direct: analyze_reasoning_with_focus forwards agent_name
# ---------------------------------------------------------------------------


async def test_analyze_reasoning_with_focus_forwards_agent_name_to_judge() -> None:
    """The judge call must carry the live agent's name as current_agent_id.

    Pre-fix the judge call had ``current_agent_id=""`` regardless of
    which agent produced the reasoning. The verdict's
    ``ReasoningJudgeInvoked`` event then carried ``subject_agent_id=""``
    and the resulting :class:`DriftEvent` could not route refines to
    the actual reasoning agent. This test pins the contract for both
    the drift event and the observability emission.
    """
    sink = ListSink()
    call_llm = _capturing_call_llm(
        {
            "on_task": False,
            "severity": "warning",
            "reason": "drifted into raccoons mid-research",
            "focused_task_id": "research_topic",
            "focus_confidence": 0.9,
            "stated_intent": "writing a section about raccoons",
        }
    )
    session = _coordinator_session()

    verdict = await analyze_reasoning_with_focus(
        "Solar panels are great. Now let me also write about raccoons "
        "and their nocturnal habits — they have masks!",
        session,
        mode="judge",
        call_llm=call_llm,
        model="judge-model",
        sink=sink,
        agent_name="research_agent",
    )

    assert verdict.drift is not None, "off-task verdict should produce a DriftEvent"
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.drift.current_agent_id == "research_agent", (
        "DriftEvent must attribute the OFF_TOPIC drift to the live agent "
        "(research_agent), not to the coordinator's bound-task agent. "
        "Pre-fix this was '' because _run_judge_with_focus did not "
        "forward agent_name to classify_reasoning_drift_with_focus."
    )
    # The ReasoningJudgeInvoked observability envelope must also stamp
    # the subject agent so harmonograf / sink consumers can see WHICH
    # agent's reasoning was judged.
    judge_events = [
        e for e in sink.events
        if hasattr(e, "WhichOneof")
        and e.WhichOneof("payload") == "reasoning_judge_invoked"
    ]
    assert len(judge_events) == 1
    assert judge_events[0].reasoning_judge_invoked.subject_agent_id == "research_agent"


async def test_analyze_reasoning_legacy_also_forwards_agent_name() -> None:
    """Legacy :func:`analyze_reasoning` must thread ``agent_name`` too.

    Some adapters use the legacy drift-only entry point; if they pass
    ``agent_name`` they expect the same attribution behaviour as the
    focused variant.
    """
    sink = ListSink()
    call_llm = _capturing_call_llm(
        {
            "on_task": False,
            "severity": "critical",
            "reason": "abandoned task to discuss raccoons",
        }
    )
    session = _coordinator_session()

    drift = await analyze_reasoning(
        "Forget the slides — I want to talk about raccoons.",
        session,
        mode="judge",
        call_llm=call_llm,
        model="judge-model",
        sink=sink,
        agent_name="research_agent",
    )

    assert drift is not None
    assert drift.current_agent_id == "research_agent"


# ---------------------------------------------------------------------------
# End-to-end via DefaultSteerer.observe_reasoning
# ---------------------------------------------------------------------------


async def test_delegated_subagent_reasoning_dispatched_to_judge() -> None:
    """research_agent's reasoning is judged AND attributed to research_agent.

    Reproduces the v17 raccoon-fixture topology: the coordinator's
    bound task is ``create_slides``; research_agent is delegated-to
    while the session's ``current_task_id`` remains ``create_slides``.
    The ADK adapter resolves the live agent name from the invocation
    context and passes it to ``observe_reasoning`` as ``agent_name``.

    Pre-fix the judge fired but with ``current_agent_id=""`` so the
    drift event and observability emission both looked like
    coordinator activity. Post-fix the verdict cleanly attributes the
    OFF_TOPIC drift to research_agent.
    """
    call_llm = _capturing_call_llm(
        {
            "on_task": False,
            "severity": "warning",
            "reason": "drifted into raccoons; off the solar-panel research",
            "focused_task_id": "",
            "focus_confidence": 0.0,
            "stated_intent": "writing a section about raccoons",
        }
    )
    sink = ListSink()
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="judge-model",
        reasoning_drift_mode="judge",
        reasoning_drift_rate_limit=1,
    )
    steerer.bind(sinks=[sink], planner=NullPlanner())
    session = _coordinator_session()

    raccoon_reasoning = (
        "I should look up solar panel efficiency, but first let me "
        "write a detailed section about raccoons — their masks, "
        "nocturnal behaviour, and dexterous paws."
    )
    await steerer.drift.observe_reasoning(
        raccoon_reasoning,
        session=session,
        agent_name="research_agent",
    )
    await _drain_judges(steerer)

    # The judge must have actually been invoked on research_agent's
    # reasoning. Pre-fix the call still happened, but with empty agent
    # attribution; the assertion below tightens that to "and the agent
    # is named on the call". We can't peek at the kwargs to
    # classify_reasoning_drift_with_focus directly from here, so we
    # assert via the observability event: the ReasoningJudgeInvoked
    # carries subject_agent_id, which is the structurally-equivalent
    # signal.
    judge_events = [
        e for e in sink.events
        if hasattr(e, "WhichOneof")
        and e.WhichOneof("payload") == "reasoning_judge_invoked"
    ]
    assert len(judge_events) == 1, (
        f"Expected exactly one judge call on research_agent's reasoning; "
        f"got {len(judge_events)}. Sink payloads: "
        f"{[e.WhichOneof('payload') for e in sink.events]}"
    )
    payload = judge_events[0].reasoning_judge_invoked
    assert payload.subject_agent_id == "research_agent", (
        "Reasoning-judge must attribute research_agent's reasoning to "
        "research_agent, not '' (pre-fix bug) or 'coordinator'. The "
        "ReasoningJudgeInvoked envelope's subject_agent_id is the "
        "downstream consumer's attribution signal."
    )
    # And the OFF_TOPIC drift event the steerer emitted must also be
    # attributed to research_agent so refines route correctly.
    drift_events = [
        e for e in sink.events
        if hasattr(e, "WhichOneof")
        and e.WhichOneof("payload") == "drift_detected"
    ]
    assert len(drift_events) >= 1, (
        "Background judge must emit a DriftDetected event for the "
        "off-task verdict on research_agent's reasoning."
    )
    # All drifts the judge emitted on this turn must attribute to
    # research_agent. Pre-fix the agent id was empty; assertion below
    # locks in the post-fix behaviour.
    assert all(
        e.drift_detected.current_agent_id == "research_agent"
        for e in drift_events
    ), [e.drift_detected.current_agent_id for e in drift_events]


async def test_coordinator_and_subagent_reasoning_both_judged_and_attributed() -> None:
    """Coordinator's planning reasoning AND research_agent's reasoning each
    fire an attributed judge call.

    Confirms the dispatch is per-(agent, task) and not gated to the
    bound-task agent only. The empirical v17 failure described the
    judge running on coordinator reasoning but not research_agent's;
    the fix must keep coordinator coverage AND extend the same
    coverage to research_agent. Two separate judge events with two
    distinct ``subject_agent_id`` values prove both are dispatched.
    """
    responses = [
        {
            "on_task": True,
            "reason": "coordinator is planning the slide deck",
            "focused_task_id": "create_slides",
            "focus_confidence": 0.9,
        },
        {
            "on_task": False,
            "severity": "warning",
            "reason": "drifted into raccoons",
            "focused_task_id": "",
            "focus_confidence": 0.0,
        },
    ]

    call_count = {"n": 0}

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        resp = responses[call_count["n"]]
        call_count["n"] += 1
        return json.dumps(resp)

    sink = ListSink()
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="judge-model",
        reasoning_drift_mode="judge",
        reasoning_drift_rate_limit=1,
    )
    steerer.bind(sinks=[sink], planner=NullPlanner())
    session = _coordinator_session()

    # Coordinator turn first.
    await steerer.drift.observe_reasoning(
        "I'll delegate research to research_agent then build slides.",
        session=session,
        agent_name="coordinator",
    )
    await _drain_judges(steerer)
    # Research_agent turn second — same session, same current_task_id.
    await steerer.drift.observe_reasoning(
        "Let me write about raccoons in addition to solar panels.",
        session=session,
        agent_name="research_agent",
    )
    await _drain_judges(steerer)

    judge_events = [
        e.reasoning_judge_invoked
        for e in sink.events
        if hasattr(e, "WhichOneof")
        and e.WhichOneof("payload") == "reasoning_judge_invoked"
    ]
    assert len(judge_events) == 2
    attributed = sorted(p.subject_agent_id for p in judge_events)
    assert attributed == ["coordinator", "research_agent"], (
        f"Both coordinator and research_agent reasoning must be judged "
        f"with correct attribution. Got: {attributed}"
    )
