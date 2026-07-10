"""``observation_only`` must gate the NUDGE injection path.

Regression tests for the observation-mode injection leak: with the
production default ``SteeringConfig(observation_only=True)`` a
NUDGE-routed drift used to enqueue a synthetic corrective message on
``session.pending_nudges`` (``_dispatch_nudge`` and the post-ABSORB
handoff carried no ``_should_inject()`` gate) and the overlay drain
(``SequentialExecutor._run_overlay``) consumed it and RE-INVOKED the
wrapped tree with a goldfive-authored user turn.

Fixed behaviour under ``observation_only=True``:

* drift detection + ``DriftDetected`` telemetry still fire;
* the nudge enqueue is suppressed (``PolicyApplied`` gate stamp);
* the overlay never re-invokes the tree (defense-in-depth gate on the
  drain for steerer subclasses that bypass the dispatcher).

Active mode (``observation_only=False``) keeps the nudge-replay path,
and the injected text is truthful: the GOAL_DRIFT corrective only
claims "already complete" for a COMPLETED task, and the replay header
only claims a plan revision when one was actually installed.

Note: tests pass ``observation_only`` EXPLICITLY — tests/conftest.py
flips the *implicit* default to False for the legacy corpus, so
``observation_only=True`` here is exactly the production default.
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

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.executors.sequential import SequentialExecutor  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class StubPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> None:
        self.refine_calls.append(kwargs)
        return None


class OverlayStubAdapter:
    """Overlay-path adapter: scripted coordinator tree."""

    def __init__(self, *, passthrough_effect: Any = None) -> None:
        self._passthrough_effect = passthrough_effect
        self.passthrough_calls: list[str] = []

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
        return None

    async def invoke(self, task: Task, session: Session) -> InvocationResult:  # noqa: ARG002
        return InvocationResult(task_id=task.id, text="")

    async def invoke_passthrough(
        self,
        user_message: str,
        *,
        session: Session,
        reconciler: Any = None,
        ctx: Any = None,  # noqa: ARG002
    ) -> InvocationResult:
        self.passthrough_calls.append(user_message)
        if self._passthrough_effect is not None:
            result = await self._passthrough_effect(user_message, session, reconciler)
            if result is not None:
                return result
        return InvocationResult(task_id="", text="")


def _stub_call_llm(responses: list[Any]):
    queue = list(responses)

    async def _call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        return str(resp)

    return _call_llm


async def _drain_background_judges(steerer: DefaultSteerer) -> None:
    pending = list(steerer._background_judges)
    if not pending:
        return
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)


def _make_session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Gather facts", description="Collect sources"),
            Task(id="t2", title="Write draft", description="Draft the memo"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a memo on topic X")],
        plan=plan,
        current_task_id="t1",
    )


def _payload_kinds(sink: ListSink) -> list[str]:
    return [e.WhichOneof("payload") for e in sink.events if hasattr(e, "WhichOneof")]


def _gate_stamps(sink: ListSink) -> list[Any]:
    return [
        e.policy_applied
        for e in sink.events
        if hasattr(e, "WhichOneof")
        and e.WhichOneof("payload") == "policy_applied"
        and e.policy_applied.policy_name == "observation_only_gate"
    ]


# ---------------------------------------------------------------------------
# Observation-only: the steerer must not enqueue nudges.
# ---------------------------------------------------------------------------


async def test_goal_drift_judge_does_not_enqueue_nudge_under_observation_only() -> None:
    """Full built-in judge pipeline: the goal-drift judge fires and the
    ladder routes to NUDGE, but under ``observation_only=True`` nothing
    lands on ``session.pending_nudges`` — only telemetry."""
    call_llm = _stub_call_llm([{"progressing": False, "reason": "off in the weeds"}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=2,
        goal_drift_call_llm=call_llm,
        steering_config=SteeringConfig(observation_only=True),
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    assert steerer._should_inject() is False

    for _ in range(2):
        await steerer.drift.note_agent_turn(session)
    await _drain_background_judges(steerer)

    # Detection telemetry still fires in full.
    assert "drift_detected" in _payload_kinds(sink)
    # The injection is suppressed and the gate is stamped.
    assert session.pending_nudges == []
    assert planner.refine_calls == []
    stamps = _gate_stamps(sink)
    assert stamps and all(s.outcome == "suppressed" for s in stamps)


async def test_dispatch_nudge_suppressed_under_observation_only() -> None:
    """Direct Level 2 dispatch: ``_dispatch_nudge`` honours the same
    ``_should_inject()`` discipline as steer/pause dispatch."""
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=True))
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    drift = DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.WARNING,
        detail="synthetic WARNING goal drift",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)

    assert session.pending_nudges == []
    stamps = _gate_stamps(sink)
    assert len(stamps) == 1
    assert "intervention=nudge" in stamps[0].detail


async def test_post_absorb_nudge_suppressed_under_observation_only() -> None:
    """The post-ABSORB nudge handoff (goldfive#202) is equally gated:
    a LOOPING_REASONING drift whose refine returns a revised plan must
    not queue the mid-invocation rescue nudge in observation mode."""
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Gather facts", status=TaskStatus.FAILED),
            Task(id="t1_v2", title="Gather facts (v2)", status=TaskStatus.PENDING),
            Task(id="t2", title="Write draft"),
        ],
        edges=[TaskEdge(from_task_id="t1_v2", to_task_id="t2")],
        revision_index=1,
    )

    class _RefiningPlanner(StubPlanner):
        async def refine(self, **kwargs: Any) -> Plan:
            self.refine_calls.append(kwargs)
            return revised

    steerer = DefaultSteerer(
        goldfive_steer_threshold="off",
        steering_config=SteeringConfig(observation_only=True, threshold="off"),
    )
    planner = _RefiningPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    drift = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="tool-loop detector fired",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)

    # Refine still ran (dry-run visibility) but nothing was queued.
    assert planner.refine_calls
    assert session.pending_nudges == []
    assert session.pending_nudges_revision_installed is False
    stamps = _gate_stamps(sink)
    assert any("intervention=post_absorb_nudge" in s.detail for s in stamps)


# ---------------------------------------------------------------------------
# Observation-only, end to end: one user turn == one tree invocation.
# ---------------------------------------------------------------------------


async def test_overlay_never_reinvokes_tree_under_observation_only() -> None:
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=True))
    session = _make_session()
    sink = ListSink()

    async def _passthrough(
        user_message: str,
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        # Mid-invocation, a NUDGE-routed drift fires.
        drift = DriftEvent(
            kind=DriftKind.GOAL_DRIFT,
            severity=DriftSeverity.WARNING,
            detail="synthetic WARNING goal drift",
            current_task_id="t1",
        )
        await steerer.drift.handle_drift(drift, session)
        await steerer.transition("t1", TaskStatus.COMPLETED, session=session)
        await steerer.transition("t2", TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id="", text="done")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=session.plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="write the memo",
    )

    assert outcome.success is True
    # ONE user turn, ONE invocation — no goldfive-authored re-invoke.
    assert adapter.passthrough_calls == ["write the memo"]
    assert session.pending_nudges == []


async def test_overlay_drain_gate_blocks_bypassing_writers_under_observation_only() -> None:
    """Defense-in-depth (goldfive#264 pattern): even when something
    bypasses the dispatcher and writes ``session.pending_nudges``
    directly, an observation-only steerer's drain must not re-invoke;
    the queue is discarded so it cannot inject later."""
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=True))
    session = _make_session()
    sink = ListSink()

    async def _passthrough(
        user_message: str,
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        session.pending_nudges.append("bypassed the dispatcher")
        await steerer.transition("t1", TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True, fail_fast=False)
    await executor.run(
        plan=session.plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="go",
    )

    assert adapter.passthrough_calls == ["go"]
    assert session.pending_nudges == []


# ---------------------------------------------------------------------------
# Active mode: the nudge path still works, and the text is truthful.
# ---------------------------------------------------------------------------


async def test_active_mode_nudge_still_enqueued_and_drained() -> None:
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=False))
    session = _make_session()
    sink = ListSink()

    async def _passthrough(
        user_message: str,
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        if len(adapter.passthrough_calls) == 1:
            drift = DriftEvent(
                kind=DriftKind.GOAL_DRIFT,
                severity=DriftSeverity.WARNING,
                detail="synthetic WARNING goal drift",
                current_task_id="t1",
            )
            await steerer.drift.handle_drift(drift, session)
            return InvocationResult(task_id="", text="healthy turn output")
        await steerer.transition("t1", TaskStatus.COMPLETED, session=session)
        await steerer.transition("t2", TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id="", text="done")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=session.plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="write the memo",
    )

    assert outcome.success is True
    assert len(adapter.passthrough_calls) == 2
    assert adapter.passthrough_calls[0] == "write the memo"
    replay = adapter.passthrough_calls[1]
    assert "GOLDFIVE" in replay
    # Truthful text: t1 was PENDING when the drift fired, so the
    # corrective must not assert completion; no refine ran, so the
    # header must not claim a plan revision.
    assert "already complete" not in replay
    assert "PLAN REVISION" not in replay
    assert "revised the active plan" not in replay
    assert session.pending_nudges == []
    assert _gate_stamps(sink) == []


def test_goal_drift_corrective_claims_completion_only_when_completed() -> None:
    """The GOAL_DRIFT corrective never asserts a completion the plan
    does not show (goldfive#475 truthfulness). The observer-note
    composer (AGENCY-PRESERVATION.md PR 4, which replaced the
    ``_CORRECTIVE_TEMPLATES`` machinery and its conditional
    "already complete" claim) satisfies this by construction: the note
    carries the judge's observation + the user's goal and makes no
    task-status claims at all — for ANY status, including COMPLETED.
    """
    from goldfive.steerer import compose_corrective_user_message

    def _plan(t1_status: TaskStatus) -> Plan:
        return Plan(
            id="p1",
            run_id="r1",
            goal_ids=[],
            tasks=[
                Task(id="t1", title="Gather facts", status=t1_status),
                Task(
                    id="t2",
                    title="Write draft",
                    assignee_agent_id="writer",
                    status=TaskStatus.PENDING,
                ),
            ],
            edges=[],
        )

    drift = DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.WARNING,
        detail="grinding",
        current_task_id="t1",
    )
    for status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED):
        msg = compose_corrective_user_message(drift=drift, refined_plan=_plan(status))
        assert "already complete" not in msg, (status, msg)
        assert "t1" in msg
    completed_msg = compose_corrective_user_message(
        drift=drift, refined_plan=_plan(TaskStatus.COMPLETED)
    )
    # Stronger than the pre-PR-4 contract: the note asserts no completion
    # even when the plan DOES show the task COMPLETED — status claims are
    # the plan-state block's job, not the advisory note's.
    assert "already complete" not in completed_msg


def test_replay_header_claims_revision_only_when_installed() -> None:
    nudges = ["Proceed with the next step."]
    plain = SequentialExecutor._compose_nudge_replay_message(nudges, plan_revised=False)
    assert "PLAN REVISION" not in plain
    assert "revised the active plan" not in plain
    assert "superseded" not in plain
    assert "GOLDFIVE" in plain and nudges[0] in plain
    revised = SequentialExecutor._compose_nudge_replay_message(nudges, plan_revised=True)
    assert "PLAN REVISION" in revised
    assert nudges[0] in revised
