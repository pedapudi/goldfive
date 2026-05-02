"""Tests for the runtime-reasoning agent pin (``Session.current_agent_id``).

The reflective self-progress check (``maybe_run_reflective_check``)
constructs ``DriftEvent``\\s of kind ``SELF_REPORTED_STUCK`` /
``UNCERTAIN_PROGRESS``. Pre-fix it stamped ``current_agent_id`` from
``task.assignee_agent_id`` — the static plan intent. When a coordinator
delegates to a child via AgentTool the child reasons under the parent's
task pin but the steerer still attributed the resulting drift to the
coordinator.

The fix reads ``session.current_agent_id`` (set by the ADK plugin's
``before_agent_callback``) when non-empty, falling back to
``task.assignee_agent_id`` for back-compat (pre-pin races, non-ADK
adapters that don't populate the session pin).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.pb.goldfive.v1 import types_pb2  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:  # pragma: no cover - convenience
        return None


class _NullPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None


def _stub_call_llm(response: dict[str, Any]):
    """Async ``call_llm``-shaped stub. Captures (system, user, model)."""
    captured: list[tuple[str, str, str]] = []

    async def _call(system: str, user: str, model: str) -> str:
        captured.append((system, user, model))
        return json.dumps(response)

    _call.calls = captured  # type: ignore[attr-defined]
    return _call


def _build_running_session(
    *,
    assignee: str,
    current_agent_id: str = "",
) -> tuple[Session, Task]:
    """Build a session whose single task is RUNNING with the given assignee."""
    task = Task(id="t1", title="Demo", assignee_agent_id=assignee)
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="demo")],
        plan=plan,
        current_task_id=task.id,
        current_agent_id=current_agent_id,
    )
    return session, task


async def _capture_drifts(steerer: DefaultSteerer, sink: _ListSink) -> list[Any]:
    """Pull the ``DriftDetected`` payloads from the sink."""
    drifts: list[Any] = []
    for evt in sink.events:
        if hasattr(evt, "WhichOneof") and evt.WhichOneof("payload") == "drift_detected":
            drifts.append(evt.drift_detected)
    return drifts


async def test_reflective_drift_uses_session_current_agent_id_when_set() -> None:
    """When ``session.current_agent_id`` is set the reflective drift uses it.

    Reproduces the topology: ``task.assignee_agent_id == "coordinator"``
    but the actual reasoner is ``research_agent`` (delegated via
    AgentTool). The reflective judge returns ``making_progress=False``;
    the resulting ``SELF_REPORTED_STUCK`` drift must be attributed to
    ``research_agent``, not the coordinator.
    """
    sink = _ListSink()
    reflective_call = _stub_call_llm(
        {"making_progress": False, "confidence": 0.9, "reason": "stuck"}
    )
    steerer = DefaultSteerer(reflective_call_llm=reflective_call)
    steerer.bind(sinks=[sink], planner=_NullPlanner())

    session, task = _build_running_session(
        assignee="coordinator",
        current_agent_id="research_agent",
    )

    await steerer.maybe_run_reflective_check(session)

    drifts = await _capture_drifts(steerer, sink)
    # ``_handle_drift`` may emit a lifecycle follow-up (ESCALATING) on top
    # of the initial ENGAGED drift; we only care about the initial event.
    assert drifts, "reflective check should have emitted at least one drift"
    assert drifts[0].kind == types_pb2.DRIFT_KIND_SELF_REPORTED_STUCK
    assert drifts[0].current_agent_id == "research_agent", (
        "reflective drift must be attributed to the runtime reasoner, "
        "not the static plan assignee"
    )


async def test_reflective_drift_falls_back_to_assignee_when_pin_empty() -> None:
    """Empty session pin → falls back to ``task.assignee_agent_id``.

    Back-compat path: a non-ADK adapter (or a pre-pin race during the
    very first invocation before ``before_agent_callback`` has fired)
    leaves ``session.current_agent_id`` empty. The drift must still
    carry an attribution rather than ending up with empty
    ``current_agent_id`` — fall back to the static plan assignee.
    """
    sink = _ListSink()
    reflective_call = _stub_call_llm(
        {"making_progress": False, "confidence": 0.9, "reason": "stuck"}
    )
    steerer = DefaultSteerer(reflective_call_llm=reflective_call)
    steerer.bind(sinks=[sink], planner=_NullPlanner())

    session, task = _build_running_session(
        assignee="coordinator",
        current_agent_id="",  # empty pin
    )

    await steerer.maybe_run_reflective_check(session)

    drifts = await _capture_drifts(steerer, sink)
    assert drifts, "reflective check should have emitted at least one drift"
    assert drifts[0].current_agent_id == "coordinator", (
        "empty session pin must fall back to the static assignee"
    )


async def test_uncertain_progress_drift_also_uses_session_pin() -> None:
    """The ``UNCERTAIN_PROGRESS`` (low-confidence) path carries the pin too.

    The reflective check has two drift exits: ``SELF_REPORTED_STUCK``
    when ``making_progress=False`` and ``UNCERTAIN_PROGRESS`` when
    ``making_progress=True`` but ``confidence < 0.5``. Both exits
    construct a ``DriftEvent`` and must read from the session pin.
    """
    sink = _ListSink()
    reflective_call = _stub_call_llm(
        {"making_progress": True, "confidence": 0.2, "reason": "vague"}
    )
    steerer = DefaultSteerer(reflective_call_llm=reflective_call)
    steerer.bind(sinks=[sink], planner=_NullPlanner())

    session, task = _build_running_session(
        assignee="coordinator",
        current_agent_id="research_agent",
    )

    await steerer.maybe_run_reflective_check(session)

    drifts = await _capture_drifts(steerer, sink)
    assert drifts, "reflective check should have emitted at least one drift"
    assert drifts[0].kind == types_pb2.DRIFT_KIND_UNCERTAIN_PROGRESS
    assert drifts[0].current_agent_id == "research_agent"


def test_session_default_current_agent_id_is_empty_string() -> None:
    """``Session.current_agent_id`` defaults to ``""`` for legacy callers."""
    s = Session(run_id="r1")
    assert s.current_agent_id == ""
