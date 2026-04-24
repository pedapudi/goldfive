"""Tests for the decision-context fields on ``goldfive_llm_span``.

Covers:
  1. Unit: ``input_preview`` / ``target_agent_id`` / ``target_task_id``
     kwargs propagate to the emitted Start/End events.
  2. Unit: the yielded handle's ``output_preview`` /
     ``decision_summary`` attributes are read on exit and stamped onto
     ``GoldfiveLLMCallEnd``.
  3. Unit: every field is truncated at the documented limits with the
     " … [truncated]" suffix.
  4. Unit: handle values persist onto the End event even when the
     wrapped body raises.
  5. Integration: every wrap site in the production code emits
     non-empty decision-context fields with the right shape.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("goldfive.pb.goldfive.v1.events_pb2")

from goldfive._llm_span import (  # noqa: E402  -- after importorskip
    _MAX_DECISION_SUMMARY_CHARS,
    _MAX_PREVIEW_CHARS,
    GoldfiveLLMSpanHandle,
    goldfive_llm_span,
)
from goldfive.sinks import InMemorySink  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span_events(sink: InMemorySink) -> list[Any]:
    out: list[Any] = []
    for evt in sink.events:
        case = evt.WhichOneof("payload")
        if case in ("goldfive_llm_call_start", "goldfive_llm_call_end"):
            out.append(evt)
    return out


def _start(evt: Any) -> Any:
    return evt.goldfive_llm_call_start


def _end(evt: Any) -> Any:
    return evt.goldfive_llm_call_end


# ---------------------------------------------------------------------------
# Unit: input_preview / target_agent_id / target_task_id propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_preview_and_targets_propagate_to_start_and_end() -> None:
    sink = InMemorySink()
    async with goldfive_llm_span(
        sinks=[sink],
        name="refine_steer",
        model="gpt-4o",
        input_preview="drift: OFF_TOPIC/WARNING\ncurrent plan: rev3, 4 task(s)",
        target_agent_id="research_agent",
        target_task_id="research_solar",
    ):
        pass

    events = _span_events(sink)
    assert len(events) == 2
    start, end = _start(events[0]), _end(events[1])

    assert start.input_preview == (
        "drift: OFF_TOPIC/WARNING\ncurrent plan: rev3, 4 task(s)"
    )
    assert start.target_agent_id == "research_agent"
    assert start.target_task_id == "research_solar"

    assert end.input_preview == start.input_preview
    assert end.target_agent_id == "research_agent"
    assert end.target_task_id == "research_solar"


@pytest.mark.asyncio
async def test_target_task_id_defaults_to_task_id() -> None:
    """When target_task_id is unset, the legacy ``task_id`` is mirrored."""
    sink = InMemorySink()
    async with goldfive_llm_span(
        sinks=[sink],
        name="judge_reasoning",
        model="gpt-judge",
        task_id="t-driving",
        target_agent_id="research_agent",
    ):
        pass
    events = _span_events(sink)
    assert events
    assert _start(events[0]).target_task_id == "t-driving"
    assert _end(events[1]).target_task_id == "t-driving"


# ---------------------------------------------------------------------------
# Unit: handle propagates output_preview / decision_summary to End event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_output_and_decision_reach_end_event() -> None:
    sink = InMemorySink()
    async with goldfive_llm_span(
        sinks=[sink],
        name="refine_steer",
        model="gpt-4o",
        input_preview="drift: USER_STEER/WARNING\n...",
        target_agent_id="research_agent",
        target_task_id="research_solar",
    ) as span:
        assert isinstance(span, GoldfiveLLMSpanHandle)
        assert span.output_preview == ""
        assert span.decision_summary == ""
        span.output_preview = (
            "revision_index=4 | tasks=2 | titles=[Corrected task, New task]"
        )
        span.decision_summary = (
            "refined plan in response to USER_STEER drift on "
            "research_solar; produced 2 tasks"
        )
    events = _span_events(sink)
    start, end = _start(events[0]), _end(events[1])
    # Start carries no output/decision — those aren't known yet.
    # (the proto has no such fields on Start, so this is really a
    # no-op assertion at the Python level; the End field is what
    # matters.)
    assert end.output_preview == (
        "revision_index=4 | tasks=2 | titles=[Corrected task, New task]"
    )
    assert end.decision_summary == (
        "refined plan in response to USER_STEER drift on "
        "research_solar; produced 2 tasks"
    )
    # And the End also echoes the Start-side input / target fields.
    assert end.input_preview == start.input_preview
    assert end.target_agent_id == start.target_agent_id
    assert end.target_task_id == start.target_task_id


@pytest.mark.asyncio
async def test_handle_defaults_are_empty_strings_not_none() -> None:
    sink = InMemorySink()
    # A caller that doesn't touch the handle must still produce an End
    # event with empty (not unset) output/decision fields.
    async with goldfive_llm_span(sinks=[sink], name="goal_derive", model=""):
        pass
    events = _span_events(sink)
    end = _end(events[1])
    assert end.output_preview == ""
    assert end.decision_summary == ""


# ---------------------------------------------------------------------------
# Unit: truncation at the documented limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_preview_truncated_at_4096_chars() -> None:
    huge = "x" * (_MAX_PREVIEW_CHARS + 500)
    sink = InMemorySink()
    async with goldfive_llm_span(
        sinks=[sink], name="judge_reasoning", model="", input_preview=huge
    ):
        pass
    events = _span_events(sink)
    assert _start(events[0]).input_preview.endswith(" … [truncated]")
    assert len(_start(events[0]).input_preview) <= _MAX_PREVIEW_CHARS
    assert _end(events[1]).input_preview == _start(events[0]).input_preview


@pytest.mark.asyncio
async def test_output_preview_truncated_at_4096_chars() -> None:
    huge = "y" * (_MAX_PREVIEW_CHARS + 500)
    sink = InMemorySink()
    async with goldfive_llm_span(sinks=[sink], name="refine", model="") as span:
        span.output_preview = huge
    events = _span_events(sink)
    assert _end(events[1]).output_preview.endswith(" … [truncated]")
    assert len(_end(events[1]).output_preview) <= _MAX_PREVIEW_CHARS


@pytest.mark.asyncio
async def test_decision_summary_truncated_at_512_chars() -> None:
    huge = "z" * (_MAX_DECISION_SUMMARY_CHARS + 100)
    sink = InMemorySink()
    async with goldfive_llm_span(sinks=[sink], name="refine", model="") as span:
        span.decision_summary = huge
    events = _span_events(sink)
    assert _end(events[1]).decision_summary.endswith(" … [truncated]")
    assert len(_end(events[1]).decision_summary) <= _MAX_DECISION_SUMMARY_CHARS


# ---------------------------------------------------------------------------
# Unit: partial handle values reach the End event even on exception
# ---------------------------------------------------------------------------


class _Boom(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_handle_values_persist_through_exception() -> None:
    sink = InMemorySink()
    with pytest.raises(_Boom):
        async with goldfive_llm_span(
            sinks=[sink],
            name="judge_reasoning",
            model="",
            input_preview="input that triggered the boom",
            target_agent_id="research_agent",
            target_task_id="t-x",
        ) as span:
            span.output_preview = "partial output before failure"
            span.decision_summary = "attempted judge call; raised"
            raise _Boom("kapow")
    events = _span_events(sink)
    end = _end(events[1])
    assert end.status == "failed"
    assert end.output_preview == "partial output before failure"
    assert end.decision_summary == "attempted judge call; raised"
    # Start-side context still stamped.
    assert end.input_preview == "input that triggered the boom"
    assert end.target_agent_id == "research_agent"
    assert end.target_task_id == "t-x"


# ---------------------------------------------------------------------------
# Integration: classify_reasoning_drift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_reasoning_drift_stamps_decision_context_on_task() -> None:
    from goldfive.drift.reasoning_judge import classify_reasoning_drift

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps({"on_task": True, "reason": "staying focused"})

    sink = InMemorySink()
    await classify_reasoning_drift(
        reasoning="writing the report section per the task",
        task=None,
        goals=[],
        model="gpt-judge",
        call_llm=fake_call_llm,
        current_task_id="t-write",
        current_agent_id="writer_agent",
        sink=sink,
    )
    events = _span_events(sink)
    assert len(events) == 2
    start, end = _start(events[0]), _end(events[1])
    assert "writing the report section" in start.input_preview
    assert start.target_agent_id == "writer_agent"
    assert start.target_task_id == "t-write"
    # On-task verdict wording.
    assert "on_task=True" in end.output_preview
    assert "writer_agent" in end.decision_summary
    assert "t-write" in end.decision_summary
    assert "on-task" in end.decision_summary


@pytest.mark.asyncio
async def test_classify_reasoning_drift_off_task_decision_wording() -> None:
    from goldfive.drift.reasoning_judge import classify_reasoning_drift

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps(
            {
                "on_task": False,
                "severity": "warning",
                "reason": "agent is discussing tangential topics",
            }
        )

    sink = InMemorySink()
    await classify_reasoning_drift(
        reasoning="I should research cats instead of solar panels",
        task=None,
        goals=[],
        model="gpt-judge",
        call_llm=fake_call_llm,
        current_task_id="research_solar",
        current_agent_id="research_agent",
        sink=sink,
    )
    events = _span_events(sink)
    end = _end(events[1])
    assert "on_task=False" in end.output_preview
    assert "warning" in end.output_preview
    assert "tangential" in end.output_preview
    assert "off-task" in end.decision_summary
    # Severity surfaced in UPPERCASE.
    assert "WARNING" in end.decision_summary
    assert "research_agent" in end.decision_summary


# ---------------------------------------------------------------------------
# Integration: classify_goal_drift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_goal_drift_stamps_decision_context() -> None:
    from goldfive.drift.goals import classify_goal_drift

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps({"progressing": True, "reason": "on track"})

    sink = InMemorySink()
    await classify_goal_drift(
        goals=[],
        plan=None,
        observed_actions=[
            {"kind": "agent_started", "agent_name": "a", "task_id": "t", "detail": "x"}
        ],
        model="gpt-goals",
        call_llm=fake_call_llm,
        current_task_id="t-current",
        sinks=[sink],
    )
    events = _span_events(sink)
    start, end = _start(events[0]), _end(events[1])
    # input_preview carries the activity block (the stub produced one
    # observed action).
    assert start.input_preview
    # Trajectory-level: target fields stay empty.
    assert start.target_agent_id == ""
    # (target_task_id mirrors task_id because goal-drift passes
    # current_task_id as task_id; that's fine — sinks use it as the
    # driving-task attribution.)
    assert start.target_task_id == "t-current"
    assert "progressing=True" in end.output_preview
    assert "on-track" in end.decision_summary


@pytest.mark.asyncio
async def test_classify_goal_drift_off_track_decision() -> None:
    from goldfive.drift.goals import classify_goal_drift

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps(
            {"progressing": False, "reason": "team not converging on the goal"}
        )

    sink = InMemorySink()
    await classify_goal_drift(
        goals=[],
        plan=None,
        observed_actions=[],
        model="gpt-goals",
        call_llm=fake_call_llm,
        sinks=[sink],
    )
    events = _span_events(sink)
    end = _end(events[1])
    assert "progressing=False" in end.output_preview
    assert "off-track" in end.decision_summary
    assert "not converging" in end.decision_summary


# ---------------------------------------------------------------------------
# Integration: LLMGoalDeriver.derive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_goal_deriver_stamps_decision_context() -> None:
    from goldfive.goal_deriver import LLMGoalDeriver

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps(
            {
                "goals": [
                    {"id": "g1", "summary": "research solar panels"},
                    {"id": "g2", "summary": "draft the report"},
                ]
            }
        )

    deriver = LLMGoalDeriver(fake_call_llm, model="gpt-derive")
    sink = InMemorySink()
    seq = iter(range(1000))
    user_request = "research solar panels and write a report"
    await deriver.derive(
        user_request,
        context={
            "sinks": [sink],
            "run_id": "r-d",
            "session_id": "s-d",
            "next_sequence": lambda: next(seq),
        },
    )
    events = _span_events(sink)
    start, end = _start(events[0]), _end(events[1])
    assert start.input_preview == user_request
    assert start.target_agent_id == ""
    assert start.target_task_id == ""
    assert "research solar panels" in end.output_preview
    assert "draft the report" in end.output_preview
    assert end.decision_summary == "derived 2 goals from user request"


# ---------------------------------------------------------------------------
# Integration: LLMPlanner._user_steer_one_attempt (refine_user_steer span)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refine_user_steer_stamps_decision_context() -> None:
    from goldfive.planner import LLMPlanner
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Plan, Task

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-new",
                        "title": "Do the steered thing",
                        "description": "per operator",
                        "assignee_agent_id": "",
                        "status": "pending",
                    }
                ],
                "edges": [],
            }
        )

    planner = LLMPlanner(call_llm=fake_call_llm, model="gpt-refine")
    sink = InMemorySink()
    seq = iter(range(1000))

    def provider() -> Any:
        return ([sink], "r-refine", "s-refine", "t-driving", lambda: next(seq))

    planner.set_span_context_provider(provider)
    prior = Plan(
        id="p1",
        run_id="r-refine",
        goal_ids=["g1"],
        summary="prior",
        tasks=[Task(id="t-old", title="Old", description="", assignee_agent_id="")],
        edges=[],
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="focus on analysis instead",
        current_task_id="t-driving",
        current_agent_id="research_agent",
    )
    await planner._refine_steer(
        prior, drift, [Goal(id="g1", summary="do stuff")], None, source="user"
    )
    events = _span_events(sink)
    assert events
    start = _start(events[0])
    end = _end(events[1])
    # Refine input_preview carries drift + plan summary (DriftKind
    # values are lowercase on the ``value`` attribute).
    assert "drift: user_steer/warning" in start.input_preview
    assert "current plan: rev0" in start.input_preview
    assert start.target_agent_id == "research_agent"
    assert start.target_task_id == "t-driving"
    # Decision summary mentions the steer source + drift kind.
    assert end.decision_summary
    assert "user steer" in end.decision_summary.lower()
    assert "user_steer" in end.decision_summary.lower()


@pytest.mark.asyncio
async def test_refine_steer_goldfive_source_decision_wording() -> None:
    from goldfive.planner import LLMPlanner
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Plan, Task

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps({"tasks": [], "edges": []})

    planner = LLMPlanner(call_llm=fake_call_llm, model="gpt-refine")
    sink = InMemorySink()
    seq = iter(range(1000))

    def provider() -> Any:
        return ([sink], "r-refine", "s-refine", "t-driving", lambda: next(seq))

    planner.set_span_context_provider(provider)
    prior = Plan(
        id="p1",
        run_id="r-refine",
        goal_ids=["g1"],
        summary="prior",
        tasks=[Task(id="t-old", title="Old", description="", assignee_agent_id="")],
        edges=[],
    )
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.CRITICAL,
        detail="agent went off-topic",
        current_task_id="t-driving",
        current_agent_id="research_agent",
    )
    await planner._refine_steer(
        prior, drift, [Goal(id="g1", summary="do stuff")], None, source="goldfive"
    )
    events = _span_events(sink)
    assert events
    end = _end(events[1])
    assert "goldfive steer" in end.decision_summary.lower()
    assert "off_topic" in end.decision_summary.lower()


# ---------------------------------------------------------------------------
# Integration: planner_gate.classify_turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_gate_stamps_decision_context() -> None:
    from goldfive.planner_gate import classify_turn
    from goldfive.types import Plan, Task

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps({"classification": "new_work", "reason": "fresh request"})

    sink = InMemorySink()
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        summary="prior",
        tasks=[Task(id="t1", title="x", description="", assignee_agent_id="")],
        edges=[],
    )
    await classify_turn(
        prior_plan=prior,
        completed_results={},
        user_input="please research solar",
        conversation_id="c1",
        call_llm=fake_call_llm,
        model="gpt-gate",
        sinks=[sink],
        run_id="r1",
        session_id="s1",
    )
    events = _span_events(sink)
    assert events
    start, end = _start(events[0]), _end(events[1])
    assert "please research solar" in start.input_preview
    assert "c1" in start.input_preview
    assert "verdict=" in end.output_preview
    assert "planner gate verdict:" in end.decision_summary


# ---------------------------------------------------------------------------
# Integration: LLMPlanner.synthesize_goal_from_steer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_goal_from_steer_stamps_decision_context() -> None:
    from goldfive.planner import LLMPlanner

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps(
            {"goal": {"id": "g-steer", "summary": "focus on X"}, "mode": "append"}
        )

    planner = LLMPlanner(call_llm=fake_call_llm, model="gpt-x")
    sink = InMemorySink()
    seq = iter(range(1000))

    def provider() -> Any:
        return ([sink], "r1", "s1", "", lambda: next(seq))

    planner.set_span_context_provider(provider)
    await planner.synthesize_goal_from_steer("please focus on X")
    events = _span_events(sink)
    assert events
    start = _start(events[0])
    end = _end(events[1])
    assert start.input_preview == "please focus on X"
    assert "synthesized goal from user steer" in end.decision_summary


# ---------------------------------------------------------------------------
# Integration: LLMPlanner.generate (plan_generate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_generate_stamps_decision_context() -> None:
    from goldfive.planner import LLMPlanner
    from goldfive.types import Goal

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps(
            {
                "tasks": [
                    {
                        "id": "t1",
                        "title": "do it",
                        "description": "",
                        "assignee_agent_id": "",
                        "status": "pending",
                    }
                ],
                "edges": [],
            }
        )

    planner = LLMPlanner(call_llm=fake_call_llm, model="gpt-x")
    sink = InMemorySink()
    seq = iter(range(1000))

    def provider() -> Any:
        return ([sink], "r-gen", "s-gen", "", lambda: next(seq))

    planner.set_span_context_provider(provider)
    plan = await planner.generate(
        goals=[Goal(id="g1", summary="ship the thing")],
        available_agents=None,
        context={"run_id": "r-gen", "user_request": "please ship the thing"},
    )
    assert plan is not None
    events = _span_events(sink)
    assert events
    start, end = _start(events[0]), _end(events[1])
    assert "please ship the thing" in start.input_preview
    assert "ship the thing" in start.input_preview
    # plan_generate is trajectory-level (initial plan).
    assert start.target_agent_id == ""
    assert start.target_task_id == ""
    assert end.output_preview
    assert "plan_generate attempt" in end.decision_summary


# ---------------------------------------------------------------------------
# Integration: DefaultSteerer.maybe_run_reflective_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflective_check_stamps_decision_context() -> None:
    from goldfive.sinks import InMemorySink
    from goldfive.steerer import DefaultSteerer
    from goldfive.types import Plan, Session, Task

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps(
            {
                "making_progress": True,
                "confidence": 0.9,
                "reason": "writing as planned",
            }
        )

    steerer = DefaultSteerer(
        reflective_call_llm=fake_call_llm,
        reflective_model="gpt-reflective",
        reflective_check_interval=1,
    )
    sink = InMemorySink()

    class _NullPlanner:
        async def generate(self, **_: Any) -> Any:
            return None

        async def refine(self, **_: Any) -> Any:
            return None

    steerer.bind(sinks=[sink], planner=_NullPlanner())

    session = Session(
        run_id="r1",
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id="t1",
                    title="Write the report",
                    description="draft the outline",
                    assignee_agent_id="writer_agent",
                )
            ],
            edges=[],
        ),
        current_task_id="t1",
    )
    # Call the reflective check directly (bypasses the interval gate
    # so we don't have to reason about LLM-call accounting state).
    await steerer.maybe_run_reflective_check(session)
    events = _span_events(sink)
    assert events, "expected at least one span pair"
    start = _start(events[0])
    end = _end(events[1])
    assert "task=t1" in start.input_preview
    assert start.target_agent_id == "writer_agent"
    assert start.target_task_id == "t1"
    assert "making_progress=True" in end.output_preview
    assert "progressing" in end.decision_summary
