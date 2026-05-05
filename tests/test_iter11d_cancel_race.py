"""Regression tests for goldfive#242: close the race between
``request_invocation_cancel`` (synchronous flag flip) and
``OrchestrationStore.active_invocation_ids()`` (transitions to empty
only AFTER ADK winds down each cancelled invocation, ~4-8s later).

The bug
-------

iter-11D introduced
:meth:`DefaultSteerer._is_late_drift_for_terminated_invocation` to gate
background-judge drifts that arrive after the originating invocation
has terminated. The gate consulted ``active_invocation_ids()``: when
empty, the drift was treated as late and refine was skipped.

Empirically, ``request_invocation_cancel`` only flips a flag — ADK then
takes several seconds to wind the invocation down and remove it from
the active-task registry. Live evidence from the brussels-sprouts run:

* 21:49:22 — ``request_invocation_cancel`` fired for 5 invocations on
  ``goal_drift severity=critical``
* 21:49:26 — JUSTIFIED_DEVIATION drift fired; gate did NOT skip;
  refine dispatched
* 21:49:30 — second JUSTIFIED_DEVIATION; ``active_invocation_ids``
  is now empty; gate correctly skipped

The 4-second window between the cancel landing and the registry
draining is when goldfive-authored drifts would mis-dispatch.

The fix
-------

A companion per-session set of cancel-pending invocation ids on
:class:`~goldfive.orchestration_store.OrchestrationStore`, stamped
synchronously at the top of ``request_invocation_cancel``. The gate
now treats "cancel-pending OR not-active" as late.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.orchestration_store import OrchestrationStore  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class NullPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return None


class _RecordingPlugin:
    """Stand-in plugin used to validate that
    ``request_invocation_cancel`` reaches the plugin while the
    cancel-pending flag is stamped synchronously beforehand. The
    plugin never removes the invocation from the active-task
    registry, mirroring the real-world race window where ADK has
    not yet wound the invocation down.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request_invocation_cancel(
        self,
        *,
        invocation_id: str,
        request: Any,
        cancel_inflight_task: bool = False,
    ) -> list[str]:
        self.calls.append(
            {
                "invocation_id": invocation_id,
                "request": request,
                "cancel_inflight_task": cancel_inflight_task,
            }
        )
        return [invocation_id]


class _StubAdapter:
    def __init__(self, plugin: _RecordingPlugin) -> None:
        self._plugin = plugin


def _session(run_id: str = "r-242", task_id: str = "t-242") -> Session:
    task = Task(id=task_id, title="Task 242", description="cancel race")
    plan = Plan(
        id="p-242",
        run_id=run_id,
        goal_ids=["g-242"],
        tasks=[task],
        edges=[],
    )
    return Session(
        run_id=run_id,
        goals=[Goal(id="g-242", summary="goal")],
        plan=plan,
        current_task_id=task_id,
    )


def _drift(
    *,
    kind: DriftKind = DriftKind.JUSTIFIED_DEVIATION,
    severity: DriftSeverity = DriftSeverity.WARNING,
    agent: str = "agent-x",
    task: str = "t-242",
    authored_by: str = "goldfive",
) -> DriftEvent:
    return DriftEvent(
        id="d-242",
        kind=kind,
        severity=severity,
        current_agent_id=agent,
        current_task_id=task,
        authored_by=authored_by,
        detail="brussels race",
        trigger_input="x",
    )


# ---------------------------------------------------------------------------
# OrchestrationStore: cancel-pending registry primitives
# ---------------------------------------------------------------------------


def test_mark_invocation_cancel_requested_round_trips() -> None:
    """``mark_*`` then ``is_*`` / ``cancel_requested_invocation_ids`` agree."""
    session = _session(run_id="r-prim")
    store = OrchestrationStore.for_session(session)
    try:
        assert store.cancel_requested_invocation_ids() == []
        assert not store.is_invocation_cancel_requested("inv-A")

        store.mark_invocation_cancel_requested("inv-A")
        store.mark_invocation_cancel_requested("inv-B")
        # Idempotent — duplicate stamp doesn't grow the set.
        store.mark_invocation_cancel_requested("inv-A")

        ids = sorted(store.cancel_requested_invocation_ids())
        assert ids == ["inv-A", "inv-B"]
        assert store.is_invocation_cancel_requested("inv-A")
        assert store.is_invocation_cancel_requested("inv-B")
        assert not store.is_invocation_cancel_requested("inv-C")
    finally:
        store.clear_active_invocations()


def test_clear_active_invocations_drops_cancel_pending_too() -> None:
    """Session teardown wipes both registries in one shot."""
    session = _session(run_id="r-clear")
    store = OrchestrationStore.for_session(session)
    store.mark_invocation_cancel_requested("inv-X")
    assert store.cancel_requested_invocation_ids() == ["inv-X"]

    store.clear_active_invocations()
    assert store.cancel_requested_invocation_ids() == []


# ---------------------------------------------------------------------------
# Late-drift gate: cancel-pending short-circuits even with active list non-empty
# ---------------------------------------------------------------------------


async def test_gate_skips_drift_when_cancel_pending_with_active_invocations() -> None:
    """The race-window scenario: cancel was requested, the registry
    still lists the invocation as active, a goldfive-authored drift
    fires, and the gate must classify it as late.
    """
    steerer = DefaultSteerer()
    session = _session(run_id="r-race")
    store = OrchestrationStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(1.0)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-live", fake_task)
    try:
        # Sanity: pre-cancel, gate does NOT classify the drift as late.
        drift = _drift()
        assert steerer._is_late_drift_for_terminated_invocation(drift, session) is False

        # Synchronous cancel-pending stamp (the part
        # request_invocation_cancel does at its top).
        store.mark_invocation_cancel_requested("inv-live")

        # The active-task registry is intentionally still populated —
        # this is the 4-8s window the bug lived in. The gate must
        # nonetheless treat the drift as late.
        assert store.active_invocation_ids() == ["inv-live"]
        assert steerer._is_late_drift_for_terminated_invocation(drift, session) is True
    finally:
        store.deregister_invocation_task("inv-live")
        store.clear_active_invocations()
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)


async def test_gate_still_classifies_empty_active_list_as_late() -> None:
    """Pre-existing semantics preserved: empty active list still gates."""
    steerer = DefaultSteerer()
    session = _session(run_id="r-empty")
    store = OrchestrationStore.for_session(session)
    try:
        assert store.active_invocation_ids() == []
        assert store.cancel_requested_invocation_ids() == []
        drift = _drift()
        assert steerer._is_late_drift_for_terminated_invocation(drift, session) is True
    finally:
        store.clear_active_invocations()


async def test_gate_lets_user_authored_drift_through_even_with_cancel_pending() -> None:
    """User-authored drifts always bypass the gate — preserved."""
    steerer = DefaultSteerer()
    session = _session(run_id="r-user")
    store = OrchestrationStore.for_session(session)
    store.mark_invocation_cancel_requested("inv-user")
    try:
        for kind in (
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.USER_PAUSE,
        ):
            drift = _drift(kind=kind, authored_by="user")
            assert (
                steerer._is_late_drift_for_terminated_invocation(drift, session)
                is False
            ), (
                f"user-authored drift kind={kind!r} must bypass the late-drift "
                "gate even when cancel-pending is set"
            )
    finally:
        store.clear_active_invocations()


# ---------------------------------------------------------------------------
# Integration: request_invocation_cancel flips the flag synchronously
# ---------------------------------------------------------------------------


async def test_request_invocation_cancel_stamps_cancel_pending_synchronously() -> None:
    """The flag flip happens at the TOP of ``request_invocation_cancel``,
    before the plugin call. After the method returns, the gate sees
    cancel-pending non-empty even though the active-task registry is
    untouched.
    """
    plugin = _RecordingPlugin()
    adapter = _StubAdapter(plugin)
    steerer = DefaultSteerer()
    steerer._adapter = adapter  # type: ignore[assignment]

    session = _session(run_id="r-stamp")
    store = OrchestrationStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(1.0)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-A", fake_task)

    # Wire the steerer's resolver to find inv-A. The simplest path is to
    # stamp ``_top_invocation_id`` on the plugin — the resolver's
    # fallback branch picks it up.
    plugin._top_invocation_id = "inv-A"  # type: ignore[attr-defined]

    try:
        cancel_drift = _drift(
            kind=DriftKind.GOAL_DRIFT,
            severity=DriftSeverity.CRITICAL,
        )
        flagged = await steerer.request_invocation_cancel(
            drift=cancel_drift, session=session
        )
        assert flagged == ["inv-A"]
        # Plugin was called.
        assert len(plugin.calls) == 1
        assert plugin.calls[0]["invocation_id"] == "inv-A"

        # Cancel-pending stamped synchronously.
        assert store.is_invocation_cancel_requested("inv-A")

        # Active-task registry still populated (mirroring the
        # real-world race window).
        assert store.active_invocation_ids() == ["inv-A"]

        # A goldfive-authored drift firing immediately after is gated.
        late_drift = _drift(kind=DriftKind.JUSTIFIED_DEVIATION)
        assert (
            steerer._is_late_drift_for_terminated_invocation(late_drift, session)
            is True
        )
    finally:
        store.deregister_invocation_task("inv-A")
        store.clear_active_invocations()
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)
