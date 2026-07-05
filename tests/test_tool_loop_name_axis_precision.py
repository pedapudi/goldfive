"""Name-axis precision for the tool-loop detector + negative class.

The tool-loop tracker's name axis (same tool name, varied args) used
to fire WARNING at 5 and CRITICAL at 7 calls in the window -- a false
positive on definitionally-healthy behaviour (an agent reading six
different files with the same tool), amplified by run-scoped
accumulation across re-invocations (goldfive#420). Downstream those
drifts drove ``planner.refine`` and, at CRITICAL, ladder escalation.

Fixed behaviour (this suite):

* An uncorroborated name-axis hit emits at most INFO; the drift routes
  to ladder Level 0 ``OBSERVE`` -- no refine, no nudge, no
  ``goldfive.active_steer.*`` stamps -- in BOTH observation-only and
  active modes.
* Exact-repeat corroboration in the window restores the tier's full
  severity; the pure exact axis is untouched.
* The ADK plugin emits one aggregated
  ``SteeringDecisionMade(outcome="no_drift", detector_name="tool_loops")``
  per invocation whose tracker ran and fired nothing (the negative
  class for threshold-tuning optimizers).

Note: tests pass ``observation_only`` EXPLICITLY -- tests/conftest.py
flips the *implicit* default to False for the legacy corpus, so
``observation_only=True`` here is exactly the production default.
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
from goldfive.drift.tool_loops import ToolLoopTracker  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
)

# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------


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


def _make_session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Read the corpus", description="Read source files"),
            Task(id="t2", title="Write summary", description="Summarise findings"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Summarise the corpus")],
        plan=plan,
        current_task_id="t1",
    )


def _payload_kinds(sink: ListSink) -> list[str]:
    return [e.WhichOneof("payload") for e in sink.events if hasattr(e, "WhichOneof")]


def _active_steer_keys(session: Session) -> list[str]:
    return [k for k in session.state if "active_steer" in str(k)]


def _varied_args_drifts(n: int = 6) -> list[DriftEvent]:
    """Tracker output for ``n`` same-name-varied-args calls (default config)."""
    tracker = ToolLoopTracker()
    fired: list[DriftEvent] = []
    for i in range(n):
        fired.extend(
            tracker.observe_tool_call(
                invocation_id="inv-1",
                agent_name="reader_agent",
                tool_name="read_file",
                args={"path": f"chapter_{i}.md"},
                task_id="t1",
                session_run_id="r1",
            )
        )
    return fired


# ---------------------------------------------------------------------------
# Detector -> steerer: uncorroborated name-axis drifts stay OBSERVE-level.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observation_only", [True, False])
async def test_varied_args_burst_never_refines_or_stamps_steer(
    observation_only: bool,
) -> None:
    """6 same-name-varied-args calls -> INFO only, OBSERVE routing.

    Both modes: the production default (observation_only=True) and
    active mode must agree here because the fix is at the detector --
    an INFO drift routes to Level 0 before any mode gate is consulted.
    """
    fired = _varied_args_drifts(6)
    assert fired, "the name tier should still produce INFO telemetry"
    assert all(d.severity is DriftSeverity.INFO for d in fired), (
        f"expected INFO only, got {[d.severity for d in fired]}"
    )

    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=observation_only))
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    for drift in fired:
        await steerer.drift.handle_drift(drift, session)

    # Telemetry fires; intervention machinery stays cold.
    assert "drift_detected" in _payload_kinds(sink)
    assert planner.refine_calls == []
    assert session.pending_nudges == []
    assert _active_steer_keys(session) == []


async def test_exact_repeat_burst_still_refines_in_active_mode() -> None:
    """Control case: the exact axis is untouched by the cap.

    Three identical work calls fire WARNING and, in active mode, still
    route to ABSORB (``planner.refine``) exactly as before.
    """
    tracker = ToolLoopTracker()
    fired: list[DriftEvent] = []
    for _ in range(3):
        fired.extend(
            tracker.observe_tool_call(
                invocation_id="inv-1",
                agent_name="worker",
                tool_name="patch_file",
                args={"path": "a.py", "diff": "x"},
                task_id="t1",
                session_run_id="r1",
            )
        )
    assert [d.severity for d in fired] == [DriftSeverity.WARNING]

    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=False))
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer.drift.handle_drift(fired[0], session)

    assert len(planner.refine_calls) == 1


# ---------------------------------------------------------------------------
# Negative class: aggregated no-drift decision from the ADK plugin.
# ---------------------------------------------------------------------------


class _NoDriftRecordingDrift:
    """Drift sub-component recording drifts + ``emit_no_drift_decision``."""

    def __init__(self) -> None:
        self.drifts: list[DriftEvent] = []
        self.no_drift_calls: list[dict[str, Any]] = []

    async def observe(self, event: Any, session: Any) -> None:
        pass

    async def handle_drift(self, drift: DriftEvent, session: Any) -> None:
        self.drifts.append(drift)

    async def emit_no_drift_decision(self, **kwargs: Any) -> None:
        self.no_drift_calls.append(kwargs)


class _NoDriftRecordingSteerer:
    def __init__(self) -> None:
        self.drift = _NoDriftRecordingDrift()
        self._sinks: list[Any] = []


def _plugin_with_ctx(task: Task, session: Session, steerer: Any):
    pytest.importorskip("google.adk")
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=task,
            tool_handlers={},
            host_agent_name="test_agent",
        )
    }
    plugin.set_active_context(state[SESSION_CONTEXT_STATE_KEY])
    return plugin, state


class _ToolStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _InvCtxStub:
    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self.invocation_id = invocation_id
        self.agent = type("_A", (), {"name": agent_name})()


class _ToolCtxStub:
    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self._invocation_context = _InvCtxStub(invocation_id, agent_name)


async def test_clean_invocation_emits_one_aggregated_no_drift_decision() -> None:
    """Tracker ran, fired nothing -> exactly one no_drift decision."""
    steerer = _NoDriftRecordingSteerer()
    session = Session(run_id="run-negclass-1")
    task = Task(id="t1", title="Fetch", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    tool_ctx = _ToolCtxStub("inv-clean", "worker")
    # Two distinct tools, distinct args -- far below every threshold.
    await plugin.after_tool_callback(
        tool=_ToolStub("read_file"),
        tool_args={"path": "a.md"},
        tool_context=tool_ctx,
        result={"ok": True},
    )
    await plugin.after_tool_callback(
        tool=_ToolStub("web_search"),
        tool_args={"q": "raccoons"},
        tool_context=tool_ctx,
        result={"ok": True},
    )
    assert steerer.drift.drifts == []

    await plugin.after_run_callback(invocation_context=_InvCtxStub("inv-clean", "worker"))

    assert len(steerer.drift.no_drift_calls) == 1
    call = steerer.drift.no_drift_calls[0]
    assert call["detector_name"] == "tool_loops"
    assert call["invocation_id"] == "inv-clean"
    assert call["agent_name"] == "worker"
    assert "2 tool call(s)" in call["reason"]

    # The stats entry is consumed: a second after_run for the same
    # invocation does not double-emit.
    await plugin.after_run_callback(invocation_context=_InvCtxStub("inv-clean", "worker"))
    assert len(steerer.drift.no_drift_calls) == 1


async def test_no_decision_when_tracker_fired_a_drift() -> None:
    """A window that fired (even INFO) is not a clean window."""
    steerer = _NoDriftRecordingSteerer()
    session = Session(run_id="run-negclass-2")
    task = Task(id="t1", title="Fetch", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    tool_ctx = _ToolCtxStub("inv-loop", "worker")
    tool = _ToolStub("patch_file")
    args = {"path": "a.py", "diff": "x"}
    for _ in range(3):  # exact repeat x 3 -> WARNING fires
        await plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_ctx, result={"ok": True}
        )
    assert steerer.drift.drifts, "expected the exact axis to fire"

    await plugin.after_run_callback(invocation_context=_InvCtxStub("inv-loop", "worker"))

    assert steerer.drift.no_drift_calls == []


async def test_no_decision_when_tracker_never_ran() -> None:
    """No tool calls -> no negative-class decision (tracker never ran)."""
    steerer = _NoDriftRecordingSteerer()
    session = Session(run_id="run-negclass-3")
    task = Task(id="t1", title="Fetch", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    await plugin.after_run_callback(invocation_context=_InvCtxStub("inv-idle", "worker"))

    assert steerer.drift.no_drift_calls == []


async def test_no_drift_decision_reaches_the_wire_via_real_steerer() -> None:
    """End-to-end: with a real DefaultSteerer the aggregated decision
    lands on the sink as ``steering_decision_made(outcome="no_drift")``."""
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=True))
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()
    task = Task(id="t1", title="Read the corpus", assignee_agent_id="reader_agent")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    tool_ctx = _ToolCtxStub("inv-wire", "reader_agent")
    await plugin.after_tool_callback(
        tool=_ToolStub("read_file"),
        tool_args={"path": "a.md"},
        tool_context=tool_ctx,
        result={"ok": True},
    )
    await plugin.after_run_callback(invocation_context=_InvCtxStub("inv-wire", "reader_agent"))

    decisions = [
        e.steering_decision_made
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "steering_decision_made"
    ]
    ours = [d for d in decisions if d.detector_name == "tool_loops"]
    assert len(ours) == 1
    assert ours[0].outcome == "no_drift"
    assert ours[0].invocation_id == "inv-wire"
