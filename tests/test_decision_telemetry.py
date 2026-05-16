"""Tests for the manifest-and-decision-telemetry decision events.

Covers four new proto events introduced at events.proto tags 40-43:

* ``LadderTransitionDecided`` — emitted when the intervention ladder
  picks a level for a freshly-emitted drift.
* ``DetectorDispatchOrdered`` — emitted once per session snapshotting
  the detector dispatch order.
* ``PolicyApplied`` — emitted at refine-failure-threshold,
  refine-outcome-succeeded, and observation-only gates.
* ``RetryBudgetSpent`` — emitted on each refine attempt with the
  remaining budget.

Each test exercises the production emit site under a controlled
fixture, asserting the envelope lands on the sink with the expected
payload fields.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.events import (  # noqa: E402
    detector_dispatch_ordered_event,
    ladder_transition_decided_event,
    policy_applied_event,
    retry_budget_spent_event,
)
from goldfive.pb.goldfive.v1 import events_pb2 as pb  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Session,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _NullPlanner:
    async def generate(self, **_: Any) -> Any:
        return None

    async def refine(self, **_: Any) -> Any:
        return None


def _build_steerer() -> tuple[DefaultSteerer, _ListSink, Session]:
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    session = Session(run_id="run-test")
    return steerer, sink, session


def _payloads(sink: _ListSink, kind: str) -> list[Any]:
    return [e for e in sink.events if e.WhichOneof("payload") == kind]


# ---------------------------------------------------------------------------
# Event factories
# ---------------------------------------------------------------------------


def test_ladder_transition_decided_factory_basic_fields() -> None:
    evt = ladder_transition_decided_event(
        "run-1",
        9,
        from_level="observe",
        to_level="nudge",
        reason="first occurrence",
        drift_kind="DRIFT_KIND_LOOPING_REASONING",
        drift_id="d-7",
        severity="warning",
    )
    assert evt.run_id == "run-1"
    assert evt.sequence == 9
    payload = evt.ladder_transition_decided
    assert payload.from_level == "observe"
    assert payload.to_level == "nudge"
    assert payload.reason == "first occurrence"
    assert payload.drift_kind == "DRIFT_KIND_LOOPING_REASONING"
    assert payload.drift_id == "d-7"
    assert payload.severity == "warning"


def test_detector_dispatch_ordered_factory_basic_fields() -> None:
    evt = detector_dispatch_ordered_event(
        "run-1",
        2,
        dispatch_order=["reasoning_judge", "goal_drift_judge", "tool_loops"],
        reason="default",
    )
    payload = evt.detector_dispatch_ordered
    assert list(payload.dispatch_order) == [
        "reasoning_judge",
        "goal_drift_judge",
        "tool_loops",
    ]
    assert payload.reason == "default"


def test_policy_applied_factory_basic_fields() -> None:
    evt = policy_applied_event(
        "run-1",
        3,
        policy_name="observation_only_gate",
        outcome="suppressed",
        reason="observation_only=True",
        detail="kind=off_topic task_id=t2",
    )
    payload = evt.policy_applied
    assert payload.policy_name == "observation_only_gate"
    assert payload.outcome == "suppressed"
    assert payload.reason == "observation_only=True"
    assert payload.detail == "kind=off_topic task_id=t2"


def test_retry_budget_spent_factory_basic_fields() -> None:
    evt = retry_budget_spent_event(
        "run-1",
        4,
        operation="refine",
        attempt=2,
        budget_remaining=0,
        reason="budget_exhausted",
    )
    payload = evt.retry_budget_spent
    assert payload.operation == "refine"
    assert payload.attempt == 2
    assert payload.budget_remaining == 0
    assert payload.reason == "budget_exhausted"


def test_retry_budget_spent_factory_coerces_non_numeric_to_zero() -> None:
    evt = retry_budget_spent_event(
        "run-1",
        5,
        operation="refine",
        attempt="not a number",  # type: ignore[arg-type]
        budget_remaining=None,  # type: ignore[arg-type]
        reason="",
    )
    payload = evt.retry_budget_spent
    assert payload.attempt == 0
    assert payload.budget_remaining == 0


# ---------------------------------------------------------------------------
# Proto envelope shape
# ---------------------------------------------------------------------------


def test_new_events_are_in_event_payload_oneof() -> None:
    """Each of the four new payloads must be accepted by Event.payload."""
    e1 = pb.Event()
    e1.ladder_transition_decided.to_level = "nudge"
    assert e1.WhichOneof("payload") == "ladder_transition_decided"

    e2 = pb.Event()
    e2.detector_dispatch_ordered.reason = "default"
    assert e2.WhichOneof("payload") == "detector_dispatch_ordered"

    e3 = pb.Event()
    e3.policy_applied.policy_name = "x"
    assert e3.WhichOneof("payload") == "policy_applied"

    e4 = pb.Event()
    e4.retry_budget_spent.operation = "refine"
    assert e4.WhichOneof("payload") == "retry_budget_spent"


# ---------------------------------------------------------------------------
# LadderTransitionDecided emit site: fires when handle_drift routes via ladder
# ---------------------------------------------------------------------------


async def test_ladder_transition_decided_emitted_on_tool_error() -> None:
    """A WARNING TOOL_ERROR drift (not steer-eligible) goes through the ladder."""
    steerer, sink, session = _build_steerer()
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="tool error",
        current_task_id="t1",
        current_agent_id="researcher",
    )
    await steerer.drift.handle_drift(drift, session)
    rows = _payloads(sink, "ladder_transition_decided")
    assert len(rows) == 1
    payload = rows[0].ladder_transition_decided
    assert payload.to_level  # some non-empty level
    assert payload.drift_kind  # stamped
    assert payload.drift_id == drift.id
    assert payload.severity == "warning"


async def test_ladder_transition_first_vs_repeat_reason() -> None:
    """The ladder reason distinguishes first occurrence from repeat."""
    steerer, sink, session = _build_steerer()
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="tool error",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)
    rows = _payloads(sink, "ladder_transition_decided")
    assert rows[0].ladder_transition_decided.reason == "first occurrence"


# ---------------------------------------------------------------------------
# DetectorDispatchOrdered emit site: once-per-session via observe entry point
# ---------------------------------------------------------------------------


async def test_detector_dispatch_ordered_fires_on_first_observe() -> None:
    """The first observe call snapshots the dispatch order."""
    steerer, sink, session = _build_steerer()
    await steerer.drift.observe({"text": "nothing"}, session)
    rows = _payloads(sink, "detector_dispatch_ordered")
    assert len(rows) == 1
    payload = rows[0].detector_dispatch_ordered
    # The registry includes at least the reasoning + goal-drift detectors.
    assert len(payload.dispatch_order) >= 2
    assert payload.reason == "default"


async def test_detector_dispatch_ordered_idempotent_per_session() -> None:
    """The snapshot fires at most once per session even on repeat observes."""
    steerer, sink, session = _build_steerer()
    await steerer.drift.observe({"text": "first"}, session)
    await steerer.drift.observe({"text": "second"}, session)
    await steerer.drift.observe({"text": "third"}, session)
    rows = _payloads(sink, "detector_dispatch_ordered")
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# PolicyApplied emit site: refine-failure-threshold gate
# ---------------------------------------------------------------------------


async def test_policy_applied_fires_on_refine_failure_threshold() -> None:
    """When the refine-outcome counter hits the threshold, PolicyApplied fires."""
    from goldfive.types import RefineOutcome

    # Use TOOL_ERROR so the steer-promotion path is skipped and the ladder
    # routes to a refine-eligible level where the failure-threshold gate is
    # consulted before the (no-op planner) refine.
    class _StubPlannerWithRefine:
        async def generate(self, **_: Any) -> Any:
            return None

        async def refine(self, **_: Any) -> Any:
            return None

    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_StubPlannerWithRefine())
    session = Session(run_id="run-test")
    # Pre-seed a failed-twice refine outcome for TOOL_ERROR/t1.
    outcome_key = (DriftKind.TOOL_ERROR.value, "t1")
    session.refine_outcomes[outcome_key] = RefineOutcome(
        state="failed",
        fail_count=2,
    )
    # Install a minimal plan so the refine path isn't short-circuited.
    from goldfive.types import Plan, Task, TaskStatus

    session.plan = Plan(
        id="p1",
        run_id="run-test",
        summary="prior",
        tasks=(Task(id="t1", title="x", description="y", status=TaskStatus.PENDING),),
        edges=(),
        goal_ids=("g1",),
    )
    # Now fire a new TOOL_ERROR drift — the ladder reaches a refine level,
    # then the refine-failure-threshold gate suppresses.
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="tool error again",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)
    rows = _payloads(sink, "policy_applied")
    # The failure-threshold policy fired.
    threshold_rows = [
        e
        for e in rows
        if e.policy_applied.policy_name == "refine_failure_threshold"
    ]
    assert len(threshold_rows) == 1
    payload = threshold_rows[0].policy_applied
    assert payload.outcome == "suppressed"
    assert payload.reason == "threshold_reached"
    assert "t1" in payload.detail


# ---------------------------------------------------------------------------
# RetryBudgetSpent emit site: planner.refine attempt loop
# ---------------------------------------------------------------------------


async def test_retry_budget_emitter_fires_on_success() -> None:
    """A successful refine attempt emits one RetryBudgetSpent with reason='validated'."""
    from goldfive.planner import LLMPlanner

    # Build a planner with a stubbed call_llm and a stub retry-budget emitter.
    emitted: list[tuple[str, int, int, str]] = []

    async def collect(
        operation: str, attempt: int, budget_remaining: int, reason: str
    ) -> None:
        emitted.append((operation, attempt, budget_remaining, reason))

    async def fake_llm(system: str, user: str, model: str) -> str:
        # Return a minimal valid JSON plan.
        return (
            '{"summary": "test", "tasks": [{"id": "t1", "title": "x", '
            '"description": "y"}], "edges": []}'
        )

    planner = LLMPlanner(call_llm=fake_llm, model="test")
    planner.set_retry_budget_emitter(collect)

    # Build a minimal prior plan to refine against.
    from goldfive.types import Goal, Plan, Task, TaskStatus

    prior = Plan(
        id="p1",
        run_id="r1",
        summary="prior",
        tasks=(Task(id="t1", title="x", description="y", status=TaskStatus.PENDING),),
        edges=(),
        goal_ids=("g1",),
    )
    goals = [Goal(id="g1", summary="ship")]

    result, err, rejected = await planner._call_and_validate_refine(
        system_prompt="sys",
        base_user_prompt="user",
        prior_plan=prior,
        goals=goals,
        log_prefix="LLMPlanner.refine",
    )
    assert result is not None
    assert not rejected
    assert err == ""
    # Exactly one RetryBudgetSpent emission on the success path.
    assert len(emitted) == 1
    op, attempt, remaining, reason = emitted[0]
    assert op == "refine"
    assert attempt == 1
    # max_refine_attempts defaults to 2; first-attempt success leaves 1 unused.
    assert remaining == 1
    assert reason == "validated"


async def test_retry_budget_emitter_fires_on_exhaustion() -> None:
    """All attempts failing emits the final RetryBudgetSpent with budget_remaining=0."""
    from goldfive.planner import LLMPlanner
    from goldfive.types import Goal, Plan, Task, TaskStatus

    emitted: list[tuple[str, int, int, str]] = []

    async def collect(
        operation: str, attempt: int, budget_remaining: int, reason: str
    ) -> None:
        emitted.append((operation, attempt, budget_remaining, reason))

    async def broken_llm(system: str, user: str, model: str) -> str:
        # Returns malformed JSON on every attempt — every call exhausts.
        return "not valid json"

    planner = LLMPlanner(call_llm=broken_llm, model="test", max_refine_attempts=2)
    planner.set_retry_budget_emitter(collect)

    prior = Plan(
        id="p1",
        run_id="r1",
        summary="prior",
        tasks=(Task(id="t1", title="x", description="y", status=TaskStatus.PENDING),),
        edges=(),
        goal_ids=("g1",),
    )
    goals = [Goal(id="g1", summary="ship")]
    result, err, rejected = await planner._call_and_validate_refine(
        system_prompt="sys",
        base_user_prompt="user",
        prior_plan=prior,
        goals=goals,
        log_prefix="LLMPlanner.refine",
    )
    assert result is None
    assert not rejected
    assert err  # non-empty error string
    # One exhaustion row; reason carries the last error.
    assert len(emitted) == 1
    op, attempt, remaining, reason = emitted[0]
    assert op == "refine"
    assert attempt == 2
    assert remaining == 0
    assert "JSON parse failed" in reason


async def test_retry_budget_emitter_fires_on_reject_sentinel() -> None:
    """The reject-sentinel terminal outcome emits a RetryBudgetSpent row.

    The reject sentinel returns early from the attempt loop; without an
    explicit emit the optimizer would silently miss reject outcomes
    even though the planner did converge on a verdict.
    """
    from goldfive.planner import LLMPlanner
    from goldfive.types import Goal, Plan, Task, TaskStatus

    emitted: list[tuple[str, int, int, str]] = []

    async def collect(
        operation: str, attempt: int, budget_remaining: int, reason: str
    ) -> None:
        emitted.append((operation, attempt, budget_remaining, reason))

    async def reject_llm(system: str, user: str, model: str) -> str:
        return '{"reject": true, "reason": "request is unsatisfiable"}'

    planner = LLMPlanner(call_llm=reject_llm, model="test", max_refine_attempts=2)
    planner.set_retry_budget_emitter(collect)

    prior = Plan(
        id="p1",
        run_id="r1",
        summary="prior",
        tasks=(Task(id="t1", title="x", description="y", status=TaskStatus.PENDING),),
        edges=(),
        goal_ids=("g1",),
    )
    goals = [Goal(id="g1", summary="ship")]
    result, reason_out, rejected = await planner._call_and_validate_refine(
        system_prompt="sys",
        base_user_prompt="user",
        prior_plan=prior,
        goals=goals,
        log_prefix="LLMPlanner.refine",
        allow_reject=True,
    )
    assert result is None
    assert rejected
    assert "unsatisfiable" in reason_out
    # Exactly one row, stamped on the first attempt with reason=rejected.
    assert len(emitted) == 1
    op, attempt, remaining, reason = emitted[0]
    assert op == "refine"
    assert attempt == 1
    assert remaining == 1
    assert reason == "rejected"


async def test_retry_budget_emitter_attempt_number_on_empty_response() -> None:
    """An empty LLM response breaks the loop early; the emitted attempt
    number must reflect the attempt that actually ran, not the budget.
    """
    from goldfive.planner import LLMPlanner
    from goldfive.types import Goal, Plan, Task, TaskStatus

    emitted: list[tuple[str, int, int, str]] = []

    async def collect(
        operation: str, attempt: int, budget_remaining: int, reason: str
    ) -> None:
        emitted.append((operation, attempt, budget_remaining, reason))

    async def empty_llm(system: str, user: str, model: str) -> str:
        # Small-model artefact: no final answer. The loop treats this as
        # terminal and breaks on the FIRST attempt without retrying.
        return ""

    planner = LLMPlanner(call_llm=empty_llm, model="test", max_refine_attempts=3)
    planner.set_retry_budget_emitter(collect)

    prior = Plan(
        id="p1",
        run_id="r1",
        summary="prior",
        tasks=(Task(id="t1", title="x", description="y", status=TaskStatus.PENDING),),
        edges=(),
        goal_ids=("g1",),
    )
    goals = [Goal(id="g1", summary="ship")]
    result, err, rejected = await planner._call_and_validate_refine(
        system_prompt="sys",
        base_user_prompt="user",
        prior_plan=prior,
        goals=goals,
        log_prefix="LLMPlanner.refine",
    )
    assert result is None
    assert not rejected
    # Budget is 3 but the empty-response branch broke on attempt 1 — the
    # emitted attempt must be 1, not the full budget of 3.
    assert len(emitted) == 1
    op, attempt, remaining, reason = emitted[0]
    assert op == "refine"
    assert attempt == 1
    assert remaining == 0
