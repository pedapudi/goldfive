"""Observation-only gating of the F1 ``plan_state`` directive surface.

The report_task_* directive / idempotent acks embed goldfive's live
``plan_state`` (completed ids + the ``next_pending`` hand-off with
``assigned_to``) — a goldfive-authored steering signal fed straight to
the coordinator's next turn. Under
``SteeringConfig(observation_only=True)`` — the production default —
that surface must be suppressed exactly like the four prompt-shaping
sites: the agent sees a neutral ack (acknowledged + factual task echo)
with no ``plan_state``. Active mode keeps the full directive payload.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.reporting import BUILTIN_REPORTING_TOOLS  # noqa: E402
from goldfive.reporting.rendering import (  # noqa: E402
    _directive_ack,
    _idempotent_response,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _StubPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return None


def _handler(name: str) -> Any:
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec.handler
    raise AssertionError(f"missing builtin tool {name!r}")


def _session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Research",
                assignee_agent_id="researcher",
                status=TaskStatus.RUNNING,
            ),
            Task(id="t2", title="Draft", assignee_agent_id="writer", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    return Session(run_id="r1", goals=[Goal(id="g1", summary="brief")], plan=plan)


def _steerer(*, observation_only: bool, sink: _ListSink) -> DefaultSteerer:
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=observation_only))
    steerer.bind(sinks=[sink], planner=_StubPlanner())
    return steerer


async def test_directive_ack_omits_plan_state_under_observation_only() -> None:
    session = _session()
    steerer = _steerer(observation_only=True, sink=_ListSink())

    out = await _handler("report_task_completed")(
        {"task_id": "t1", "summary": "done"}, session, steerer
    )

    # Neutral ack: the transition the agent itself reported is echoed,
    # but no goldfive-authored plan_state directive rides along.
    assert out["acknowledged"] is True
    assert out["task"] == {"id": "t1", "status": TaskStatus.COMPLETED.value}
    assert "plan_state" not in out
    # The transition itself still landed — observation_only gates the
    # directive surface, not the reporting-driven state machine.
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


async def test_directive_ack_includes_plan_state_in_active_mode() -> None:
    session = _session()
    steerer = _steerer(observation_only=False, sink=_ListSink())

    out = await _handler("report_task_completed")(
        {"task_id": "t1", "summary": "done"}, session, steerer
    )

    assert out["acknowledged"] is True
    assert out["plan_state"]["completed_task_ids"] == ["t1"]
    assert out["plan_state"]["next_pending"]["id"] == "t2"
    assert out["plan_state"]["next_pending"]["assigned_to"] == "writer"


async def test_idempotent_ack_omits_plan_state_under_observation_only() -> None:
    session = _session()
    steerer = _steerer(observation_only=True, sink=_ListSink())
    handler = _handler("report_task_completed")

    await handler({"task_id": "t1", "summary": "done"}, session, steerer)
    second = await handler({"task_id": "t1", "summary": "again"}, session, steerer)

    assert second["acknowledged"] is True
    assert second["idempotent"] is True
    assert second["task"] == {"id": "t1", "status": "COMPLETED"}
    assert "plan_state" not in second


async def test_idempotent_ack_includes_plan_state_in_active_mode() -> None:
    session = _session()
    steerer = _steerer(observation_only=False, sink=_ListSink())
    handler = _handler("report_task_completed")

    await handler({"task_id": "t1", "summary": "done"}, session, steerer)
    second = await handler({"task_id": "t1", "summary": "again"}, session, steerer)

    assert second["idempotent"] is True
    assert second["plan_state"]["next_pending"]["id"] == "t2"


def test_rendering_helpers_default_to_directive_shape_without_steerer() -> None:
    # Legacy callers / stubs that pass no steerer keep the pre-gate
    # shape — same tolerance as PromptShaper.should_inject.
    session = _session()
    out = _directive_ack(session=session, task_id="t1", new_status=TaskStatus.COMPLETED)
    assert "plan_state" in out
    idem = _idempotent_response(TaskStatus.COMPLETED, session=session, task_id="t1")
    assert "plan_state" in idem
