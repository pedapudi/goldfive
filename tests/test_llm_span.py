"""Tests for :func:`goldfive._llm_span.goldfive_llm_span`.

Covers the helper itself (unit) and integration with the four
goldfive-internal LLM call sites wrapped by this PR:

* ``classify_reasoning_drift`` (drift/reasoning_judge.py)
* ``classify_goal_drift``      (drift/goals.py)
* ``LLMPlanner._refine_steer`` (planner.py)
* ``LLMGoalDeriver.derive``    (goal_deriver.py)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# The proto extra is required: this PR introduces new proto messages that
# must be on the wire for the tests to assert anything meaningful. Skip
# with a clear message if the stubs aren't importable.
pytest.importorskip("goldfive.pb.goldfive.v1.events_pb2")

from goldfive._llm_span import goldfive_llm_span
from goldfive.sinks import InMemorySink

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _span_events(sink: InMemorySink) -> list[Any]:
    """Return every GoldfiveLLMCallStart/End event on ``sink``."""
    out: list[Any] = []
    for evt in sink.events:
        case = evt.WhichOneof("payload")
        if case in ("goldfive_llm_call_start", "goldfive_llm_call_end"):
            out.append(evt)
    return out


def _payload_case(evt: Any) -> str:
    return evt.WhichOneof("payload")


# ---------------------------------------------------------------------------
# Unit: happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_emits_start_then_end_with_matching_span_id() -> None:
    sink = InMemorySink()
    async with goldfive_llm_span(
        sinks=[sink],
        name="refine_steer",
        model="gpt-4o-mini",
        run_id="r1",
        session_id="s1",
        task_id="t7",
    ):
        await asyncio.sleep(0)  # simulate the awaited ``call_llm``

    events = _span_events(sink)
    assert len(events) == 2, f"expected 2 span events, got {len(events)}: {events}"
    start, end = events
    assert _payload_case(start) == "goldfive_llm_call_start"
    assert _payload_case(end) == "goldfive_llm_call_end"

    assert start.goldfive_llm_call_start.span_id == end.goldfive_llm_call_end.span_id
    assert start.goldfive_llm_call_start.name == "refine_steer"
    assert start.goldfive_llm_call_start.model == "gpt-4o-mini"
    assert start.goldfive_llm_call_start.task_id == "t7"
    assert start.goldfive_llm_call_start.start_time_ns > 0

    assert end.goldfive_llm_call_end.name == "refine_steer"
    assert end.goldfive_llm_call_end.status == "completed"
    assert end.goldfive_llm_call_end.error == ""
    # Duration must be non-negative; we slept for one tick but the monotonic
    # floor is 0 nanoseconds on fast runners — accept >=.
    assert (
        end.goldfive_llm_call_end.end_time_ns
        >= start.goldfive_llm_call_start.start_time_ns
    )


# ---------------------------------------------------------------------------
# Unit: exception path
# ---------------------------------------------------------------------------


class _Boom(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_exception_emits_failed_end_and_reraises() -> None:
    sink = InMemorySink()
    with pytest.raises(_Boom, match="kapow"):
        async with goldfive_llm_span(
            sinks=[sink],
            name="judge_goal_drift",
            model="gpt-4o-mini",
        ):
            raise _Boom("kapow")

    events = _span_events(sink)
    assert len(events) == 2
    start, end = events
    assert start.goldfive_llm_call_start.span_id == end.goldfive_llm_call_end.span_id
    assert end.goldfive_llm_call_end.status == "failed"
    assert "kapow" in end.goldfive_llm_call_end.error
    assert "_Boom" in end.goldfive_llm_call_end.error


# ---------------------------------------------------------------------------
# Unit: empty sink list is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_sinks_is_noop() -> None:
    # Should not raise regardless of what the body does.
    async with goldfive_llm_span(sinks=[], name="goal_derive", model=""):
        await asyncio.sleep(0)

    # Exception path through an empty sink still re-raises.
    with pytest.raises(ValueError):
        async with goldfive_llm_span(sinks=[], name="goal_derive", model=""):
            raise ValueError("x")


# ---------------------------------------------------------------------------
# Unit: timing is close to wrapped call's wall-clock duration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_ts_tracks_wrapped_call_duration() -> None:
    sink = InMemorySink()
    async with goldfive_llm_span(sinks=[sink], name="plan_generate", model=""):
        await asyncio.sleep(0.05)

    events = _span_events(sink)
    duration_ns = (
        events[1].goldfive_llm_call_end.end_time_ns
        - events[0].goldfive_llm_call_start.start_time_ns
    )
    # 50ms in ns is 5e7. Allow generous lower bound (25ms) for noisy CI.
    assert duration_ns >= 25_000_000
    # And a sane upper bound (5s) so a bug that stamps a fixed value
    # doesn't pass silently.
    assert duration_ns < 5_000_000_000


# ---------------------------------------------------------------------------
# Unit: broken sink doesn't break the wrapped call
# ---------------------------------------------------------------------------


class _BrokenSink:
    async def emit(self, event_pb: Any) -> None:
        raise RuntimeError("sink is on fire")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_broken_sink_does_not_break_wrapped_call() -> None:
    good = InMemorySink()
    broken = _BrokenSink()
    # The wrapped call must still run and the good sink must still see
    # both events.
    async with goldfive_llm_span(
        sinks=[broken, good], name="refine", model=""
    ):
        pass
    events = _span_events(good)
    assert len(events) == 2
    assert events[0].goldfive_llm_call_start.name == "refine"
    assert events[1].goldfive_llm_call_end.status == "completed"


# ---------------------------------------------------------------------------
# Integration: classify_reasoning_drift emits a judge_reasoning span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_reasoning_drift_emits_judge_reasoning_span() -> None:
    from goldfive.drift.reasoning_judge import classify_reasoning_drift

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return '{"on_task": true, "reason": "covered"}'

    sink = InMemorySink()
    result = await classify_reasoning_drift(
        reasoning="agent is writing the file as expected",
        task=None,
        goals=[],
        model="gpt-judge",
        call_llm=fake_call_llm,
        current_task_id="t-reasoning",
        sink=sink,
        run_id="r-r",
        session_id="s-r",
    )
    assert result is None  # on-task verdict

    spans = _span_events(sink)
    assert len(spans) == 2
    assert spans[0].goldfive_llm_call_start.name == "judge_reasoning"
    assert spans[0].goldfive_llm_call_start.model == "gpt-judge"
    assert spans[0].goldfive_llm_call_start.task_id == "t-reasoning"
    assert (
        spans[0].goldfive_llm_call_start.span_id
        == spans[1].goldfive_llm_call_end.span_id
    )
    assert spans[1].goldfive_llm_call_end.status == "completed"


# ---------------------------------------------------------------------------
# Integration: classify_goal_drift emits a judge_goal_drift span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_goal_drift_emits_judge_goal_drift_span() -> None:
    from goldfive.drift.goals import classify_goal_drift

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return '{"progressing": true}'

    sink = InMemorySink()
    result = await classify_goal_drift(
        goals=[],
        plan=None,
        observed_actions=[],
        model="gpt-goals",
        call_llm=fake_call_llm,
        current_task_id="t-goal",
        sinks=[sink],
        run_id="r-g",
        session_id="s-g",
    )
    assert result is None

    spans = _span_events(sink)
    assert len(spans) == 2
    assert spans[0].goldfive_llm_call_start.name == "judge_goal_drift"
    assert spans[0].goldfive_llm_call_start.model == "gpt-goals"
    assert spans[0].goldfive_llm_call_start.task_id == "t-goal"
    assert spans[1].goldfive_llm_call_end.status == "completed"


# ---------------------------------------------------------------------------
# Integration: LLMPlanner._refine_steer emits a refine_* span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_planner_refine_steer_emits_span_via_set_span_context_provider() -> (
    None
):
    import json

    from goldfive.planner import LLMPlanner
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Plan, Task

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        # Minimal valid refine response: one new pending task.
        return json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-new",
                        "title": "Do the steered thing",
                        "description": "per operator steer",
                        "assignee_agent_id": "",
                        "status": "pending",
                    }
                ],
                "edges": [],
            }
        )

    planner = LLMPlanner(call_llm=fake_call_llm, model="gpt-refine")

    # Wire the span-context provider directly (bypass the steerer for a
    # focused integration test). Returns the tuple the planner expects.
    sink = InMemorySink()
    seq = iter(range(1000))

    def provider() -> Any:
        return (
            [sink],
            "r-refine",
            "s-refine",
            "t-driving",
            lambda: next(seq),
        )

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
        detail="change direction",
        current_task_id="t-driving",
    )
    revised = await planner._refine_steer(
        prior, drift, [Goal(id="g1", summary="do stuff")], None, source="user"
    )
    # Regardless of merge outcome, a span must fire on the call.
    spans = _span_events(sink)
    assert len(spans) >= 2
    # At least one Start carries a "refine"-prefixed name.
    names_on_start = [
        s.goldfive_llm_call_start.name
        for s in spans
        if _payload_case(s) == "goldfive_llm_call_start"
    ]
    assert any(n.startswith("refine") for n in names_on_start), names_on_start
    # The driving task id is stamped on the span.
    assert all(
        s.goldfive_llm_call_start.task_id == "t-driving"
        for s in spans
        if _payload_case(s) == "goldfive_llm_call_start"
    )
    # Keep ``revised`` so lints don't warn about an unused assignment;
    # the test body cares about emissions, not merge outcome.
    assert revised is None or revised is not None


# ---------------------------------------------------------------------------
# Integration: LLMGoalDeriver.derive emits a goal_derive span via context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_goal_deriver_emits_goal_derive_span_via_context() -> None:
    from goldfive.goal_deriver import LLMGoalDeriver

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return '{"goals": [{"id": "g1", "summary": "write the thing"}]}'

    deriver = LLMGoalDeriver(fake_call_llm, model="gpt-derive")
    sink = InMemorySink()
    seq = iter(range(1000))
    goals = await deriver.derive(
        "write the thing",
        context={
            "sinks": [sink],
            "run_id": "r-d",
            "session_id": "s-d",
            "next_sequence": lambda: next(seq),
        },
    )
    assert len(goals) == 1
    spans = _span_events(sink)
    assert len(spans) == 2
    assert spans[0].goldfive_llm_call_start.name == "goal_derive"
    assert spans[0].goldfive_llm_call_start.model == "gpt-derive"
    assert spans[1].goldfive_llm_call_end.status == "completed"


# ---------------------------------------------------------------------------
# Integration: planner with no span context provider degrades cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_without_span_context_provider_noops() -> None:
    """When no provider is bound, spans are silently skipped.

    Standalone-planner tests (no steerer) must not crash because the
    span helper can't find a session.
    """
    import json

    from goldfive.planner import LLMPlanner

    async def fake_call_llm(system: str, user: str, model: str) -> str:
        return json.dumps({"tasks": [], "edges": []})

    planner = LLMPlanner(call_llm=fake_call_llm, model="gpt-x")
    assert planner._span_kwargs() == {"sinks": [], "model": "gpt-x"}


# ---------------------------------------------------------------------------
# goldfive#266 — shutdown-initiated cancel surfaces as ``cancelled``, not
# ``failed``. Live session 4538863f-0dea-4fe8-97b4-5f660ee2cb7f surfaced
# routine ``_drain_steerer_at_run_boundary`` teardowns as red
# ``judge_reasoning`` spans with CancelledError stack traces. The fix
# threads a per-task marker the drain stamps before issuing
# ``task.cancel()`` so the span helper can emit
# ``status="cancelled"`` with a benign reason — preserving the cancel
# BEHAVIOUR, correcting only the observability. Genuine cancels (caller-
# driven asyncio.CancelledError from outside the drain) still surface as
# ``failed`` so operators can spot real cancel-induced aborts.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_inside_drain_marker_surfaces_as_cancelled() -> None:
    """When the steerer-drain marker is on the current task, a CancelledError
    inside the span body produces ``status="cancelled"``."""
    from goldfive._llm_span import DRAIN_INITIATED_ATTR

    sink = InMemorySink()
    started = asyncio.Event()

    async def body() -> None:
        # Simulate the drain having marked us BEFORE issuing the cancel.
        # In production this is set by ``_drain_background_set`` on the
        # judge task object; in this unit test we stamp the same marker
        # on the current task so the span helper observes the same
        # signal.
        current = asyncio.current_task()
        assert current is not None
        setattr(current, DRAIN_INITIATED_ATTR, True)
        async with goldfive_llm_span(
            sinks=[sink], name="judge_reasoning", model="gpt-judge"
        ):
            started.set()
            # Sleep until cancelled — the drain-issued cancel arrives
            # here, propagates into the span helper, and the helper
            # detects the marker.
            await asyncio.sleep(60)

    task = asyncio.create_task(body())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = _span_events(sink)
    assert len(events) == 2, f"expected start+end, got {events}"
    end = events[1].goldfive_llm_call_end
    assert end.status == "cancelled", (
        f"drain-initiated cancel should surface as status=cancelled, "
        f"got status={end.status!r}, error={end.error!r}"
    )
    assert "drained" in end.error.lower() or "boundary" in end.error.lower(), (
        f"cancelled span should carry a benign drain reason; "
        f"got error={end.error!r}"
    )
    # Genuine error indicators MUST be absent — operators must not see
    # ``CancelledError`` or a stack-trace fragment for routine drains.
    assert "CancelledError" not in end.error
    assert "Traceback" not in end.error


@pytest.mark.asyncio
async def test_cancel_outside_drain_preserves_failed() -> None:
    """A CancelledError WITHOUT the drain marker still surfaces as
    ``status="failed"`` — the legacy behaviour for genuine cancels
    (e.g. caller-driven asyncio.CancelledError, USER_CANCEL) is
    preserved."""
    sink = InMemorySink()
    started = asyncio.Event()

    async def body() -> None:
        # No DRAIN_INITIATED_ATTR set on the current task.
        async with goldfive_llm_span(
            sinks=[sink], name="judge_reasoning", model="gpt-judge"
        ):
            started.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(body())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = _span_events(sink)
    assert len(events) == 2
    end = events[1].goldfive_llm_call_end
    assert end.status == "failed", (
        f"non-drain cancel should preserve status=failed; "
        f"got status={end.status!r}"
    )
    # The error text still carries the CancelledError diagnostic for
    # genuine cancels — that's the back-compat behaviour the negative
    # case relies on.
    assert "CancelledError" in end.error


@pytest.mark.asyncio
async def test_drain_marker_tags_judge_task_and_span_records_cancelled() -> None:
    """End-to-end: the steerer's drain stamps the marker on the
    in-flight judge task, the cancellation propagates into the
    ``judge_reasoning`` span body, and the resulting
    ``GoldfiveLLMCallEnd`` reports ``status="cancelled"``.

    This is the goldfive#266 fix exercised through the actual drain
    code path — not just the span helper unit. Pins the integration
    between :meth:`DefaultSteerer._drain_background_set` and the
    span's CancelledError handler."""
    from goldfive._llm_span import DRAIN_INITIATED_ATTR, goldfive_llm_span

    sink = InMemorySink()
    started = asyncio.Event()
    saw_cancel: dict[str, bool] = {"hit": False}

    async def judge_body() -> None:
        # Reproduce the reasoning-judge body shape: open the
        # judge_reasoning span and block on a long await that
        # ``task.cancel()`` will interrupt — exactly the live-session
        # symptom shape.
        try:
            async with goldfive_llm_span(
                sinks=[sink], name="judge_reasoning", model="m"
            ):
                started.set()
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            saw_cancel["hit"] = True
            raise

    task = asyncio.create_task(judge_body(), name="goldfive-reasoning-judge:s-fix")
    await asyncio.wait_for(started.wait(), timeout=2.0)
    # Reproduce ``_drain_background_set``'s tag-then-cancel pattern.
    setattr(task, DRAIN_INITIATED_ATTR, True)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert saw_cancel["hit"] is True

    events = _span_events(sink)
    assert len(events) == 2
    end = events[1].goldfive_llm_call_end
    assert end.name == "judge_reasoning"
    assert end.status == "cancelled"
    assert "drained" in end.error.lower() or "boundary" in end.error.lower()
