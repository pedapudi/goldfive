"""Goldfive boundary wrapper tests (goldfive#271 Phase 3.5 component 1).

The boundary is the canonical ``try / finally`` arc wrapping each ADK
agent invocation:

* ``before_agent_callback`` emits ``InvocationBoundaryEntered`` and
  registers ``asyncio.current_task()`` on
  :class:`~goldfive.orchestration_store.OrchestrationStore`.
* ``after_agent_callback`` emits the paired
  ``InvocationBoundaryExited`` (reason="completed").
* When ``CancelledError`` propagates out of the ADK runner, the
  canonical catch site in :meth:`ADKAdapter._invoke_internal`
  (``except asyncio.CancelledError``) calls
  :meth:`_GoldfiveADKPlugin.close_open_boundaries` to fire
  ``InvocationBoundaryExited(reason="cancelled")`` for every
  still-open boundary.
* The structured marker :class:`_InvocationCancelled` documents the
  catch-site contract; the rest of the runtime sees a normal
  completion shape.

These tests pin the contract: entry/exit pair on the wire, registry
relocated to OrchestrationStore (NOT plugin), CancelledError caught
once at the boundary.
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

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SessionContext,
    _InvocationCancelled,
    _InvocationTaskRegistryView,
    make_adk_plugin,
)
from goldfive.orchestration_store import OrchestrationStore  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


def _events_of_kind(sink: ListSink, kind: str) -> list[Any]:
    out: list[Any] = []
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") == kind:
                out.append(evt)
        except Exception:
            continue
    return out


class _FakeAgent:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _FakeADKSession:
    def __init__(self, *, state: dict[str, Any], run_id: str = "run-test") -> None:
        self.state = state
        self.run_id = run_id
        self.id = "session-test"


class _FakeCallbackContext:
    def __init__(self, *, invocation_context: Any) -> None:
        self._invocation_context = invocation_context


class _FakeInvocationContext:
    def __init__(
        self,
        *,
        invocation_id: str,
        session_state: dict[str, Any] | None = None,
        agent_name: str = "coordinator",
    ) -> None:
        self.invocation_id = invocation_id
        self.session = _FakeADKSession(state=session_state or {})
        self.agent = _FakeAgent(name=agent_name)


def _make_session() -> Session:
    return Session(
        run_id="run-test",
        goals=[Goal(id="g1", summary="ship")],
        plan=Plan(
            id="p1",
            run_id="run-test",
            goal_ids=["g1"],
            tasks=[
                Task(id="t1", title="T1", status=TaskStatus.RUNNING),
                Task(id="t2", title="T2", status=TaskStatus.PENDING),
            ],
            edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        ),
        current_task_id="t1",
    )


def _bind_plugin_with_sink() -> tuple[Any, Session, ListSink]:
    plugin = make_adk_plugin(host_agent_name="coordinator")
    session = _make_session()
    sink = ListSink()
    steerer = DefaultSteerer()
    # Sinks attached directly — the Steerer.bind helper requires a
    # planner, but for boundary-emit tests we only need ``_sinks``.
    steerer._sinks = [sink]
    ctx = SessionContext(
        session=session,
        steerer=steerer,
        tools=(),
        tool_handlers={},
        host_agent_name="coordinator",
        task=session.plan.tasks[0],
    )
    plugin.set_active_context(ctx)
    return plugin, session, sink


# ---------------------------------------------------------------------------
# 1. before_agent_callback emits InvocationBoundaryEntered
# ---------------------------------------------------------------------------


async def test_before_agent_emits_boundary_entered() -> None:
    """Entering the boundary fires the ``InvocationBoundaryEntered``
    event paired with the agent name + task id from the active
    SessionContext."""
    plugin, _session, sink = _bind_plugin_with_sink()
    inv_ctx = _FakeInvocationContext(invocation_id="inv-A")
    cb_ctx = _FakeCallbackContext(invocation_context=inv_ctx)
    agent = _FakeAgent(name="coordinator")

    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)

    entered = _events_of_kind(sink, "invocation_boundary_entered")
    assert len(entered) == 1, "exactly one InvocationBoundaryEntered must fire"
    payload = entered[0].invocation_boundary_entered
    assert payload.invocation_id == "inv-A"
    assert payload.agent_name == "coordinator"
    assert payload.task_id == "t1"


# ---------------------------------------------------------------------------
# 2. after_agent_callback emits the paired InvocationBoundaryExited
# ---------------------------------------------------------------------------


async def test_after_agent_emits_boundary_exited_completed() -> None:
    """The normal completion path emits ``Exited(reason="completed")``."""
    plugin, _session, sink = _bind_plugin_with_sink()
    inv_ctx = _FakeInvocationContext(invocation_id="inv-A")
    cb_ctx = _FakeCallbackContext(invocation_context=inv_ctx)
    agent = _FakeAgent(name="coordinator")

    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)
    await plugin.after_agent_callback(agent=agent, callback_context=cb_ctx)

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 1, "exactly one InvocationBoundaryExited must fire"
    payload = exited[0].invocation_boundary_exited
    assert payload.invocation_id == "inv-A"
    assert payload.agent_name == "coordinator"
    assert payload.reason == "completed"


# ---------------------------------------------------------------------------
# 3. Boundary exit is exactly-once: a second after-callback no-ops
# ---------------------------------------------------------------------------


async def test_boundary_exit_is_exactly_once() -> None:
    """Two ``after_agent_callback`` calls for the same invocation_id
    must NOT emit two exit events. The boundary pair is exactly-once."""
    plugin, _session, sink = _bind_plugin_with_sink()
    inv_ctx = _FakeInvocationContext(invocation_id="inv-A")
    cb_ctx = _FakeCallbackContext(invocation_context=inv_ctx)
    agent = _FakeAgent(name="coordinator")

    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)
    await plugin.after_agent_callback(agent=agent, callback_context=cb_ctx)
    await plugin.after_agent_callback(agent=agent, callback_context=cb_ctx)

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 1


# ---------------------------------------------------------------------------
# 4. Boundary entry is exactly-once: a second before-callback no-ops
# ---------------------------------------------------------------------------


async def test_boundary_entry_is_exactly_once() -> None:
    """A re-entrant ``before_agent_callback`` (transfer-to-agent inside
    one invocation) must not fire a second Entered event."""
    plugin, _session, sink = _bind_plugin_with_sink()
    inv_ctx = _FakeInvocationContext(invocation_id="inv-A")
    cb_ctx = _FakeCallbackContext(invocation_context=inv_ctx)
    agent = _FakeAgent(name="coordinator")

    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)
    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)

    entered = _events_of_kind(sink, "invocation_boundary_entered")
    assert len(entered) == 1


# ---------------------------------------------------------------------------
# 5. close_open_boundaries fires Exited(reason="cancelled") for every
#    still-open boundary — the canonical CancelledError catch contract
# ---------------------------------------------------------------------------


async def test_close_open_boundaries_emits_cancelled_exit() -> None:
    """The boundary's canonical exit-on-cancel path emits
    ``Exited(reason="cancelled")`` for every still-open invocation.

    This is the path the adapter's ``except asyncio.CancelledError``
    site exercises when ADK skips ``after_agent_callback`` because
    the runner generator was closed mid-stream.
    """
    plugin, _session, sink = _bind_plugin_with_sink()
    # Open two boundaries (e.g. a parent + nested AgentTool) without
    # firing the paired after_agent_callback.
    for inv_id in ("inv-A", "inv-B"):
        cb_ctx = _FakeCallbackContext(
            invocation_context=_FakeInvocationContext(invocation_id=inv_id),
        )
        agent = _FakeAgent(name="coordinator")
        await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)

    entered = _events_of_kind(sink, "invocation_boundary_entered")
    assert len(entered) == 2

    # The canonical catch site closes both boundaries.
    await plugin.close_open_boundaries(reason="cancelled")

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 2
    reasons = {e.invocation_boundary_exited.reason for e in exited}
    assert reasons == {"cancelled"}
    inv_ids = {e.invocation_boundary_exited.invocation_id for e in exited}
    assert inv_ids == {"inv-A", "inv-B"}


# ---------------------------------------------------------------------------
# 6. close_open_boundaries is idempotent — second call is a no-op
# ---------------------------------------------------------------------------


async def test_close_open_boundaries_idempotent() -> None:
    """A second ``close_open_boundaries`` after every boundary already
    closed must NOT re-emit Exited events."""
    plugin, _session, sink = _bind_plugin_with_sink()
    cb_ctx = _FakeCallbackContext(
        invocation_context=_FakeInvocationContext(invocation_id="inv-A"),
    )
    agent = _FakeAgent(name="coordinator")
    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)
    await plugin.close_open_boundaries(reason="cancelled")
    await plugin.close_open_boundaries(reason="cancelled")

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 1


# ---------------------------------------------------------------------------
# 7. Boundary exit emits "cancelled" for cancel-checkpoint short-circuit
# ---------------------------------------------------------------------------


async def test_boundary_exits_cancelled_on_cancel_checkpoint() -> None:
    """When ``before_agent_callback`` finds a pending cancel and
    short-circuits, the boundary's exit emit fires with
    ``reason="cancelled"`` so the entry/exit pair is still visible on
    the wire."""
    from goldfive.types import CancellationRequest, DriftSeverity

    plugin, _session, sink = _bind_plugin_with_sink()
    inv_ctx = _FakeInvocationContext(invocation_id="inv-A")
    cb_ctx = _FakeCallbackContext(invocation_context=inv_ctx)
    agent = _FakeAgent(name="coordinator")

    plugin.request_invocation_cancel(
        invocation_id="inv-A",
        request=CancellationRequest(
            invocation_id="inv-A",
            reason="user_steer",
            severity=DriftSeverity.CRITICAL,
            drift_kind="user_cancel",
        ),
    )

    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)

    entered = _events_of_kind(sink, "invocation_boundary_entered")
    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(entered) == 1
    assert len(exited) == 1
    assert exited[0].invocation_boundary_exited.reason == "cancelled"


# ---------------------------------------------------------------------------
# 8. Registry storage relocated: lives on OrchestrationStore (NOT plugin)
# ---------------------------------------------------------------------------


def test_invocation_tasks_registry_is_view_not_dict() -> None:
    """The plugin attribute is a backwards-compat view, not the actual
    storage. The Phase 3.5 contract requires the registry to live on
    OrchestrationStore — verifies the storage truly relocated rather
    than being a duplicate dict."""
    plugin = make_adk_plugin(host_agent_name="coordinator")
    assert isinstance(plugin._invocation_tasks, _InvocationTaskRegistryView)
    # The view is NOT a dict — it forwards to OrchestrationStore.
    assert not isinstance(plugin._invocation_tasks, dict)


async def test_registry_lives_on_orchestration_store() -> None:
    """A task registered through the plugin is visible via the
    OrchestrationStore lookup; clearing the store-side registry empties
    the plugin-side view."""
    plugin, session, _sink = _bind_plugin_with_sink()
    # Register a task through the legacy attribute path.
    fake_task = asyncio.create_task(asyncio.sleep(5.0))
    try:
        plugin._invocation_tasks["inv-X"] = fake_task

        store = OrchestrationStore.for_session(session)
        # The store is the source of truth.
        assert store.get_invocation_task("inv-X") is fake_task
        assert "inv-X" in store.active_invocation_ids()

        # Clearing the store empties the plugin-side view.
        store.clear_active_invocations()
        assert "inv-X" not in plugin._invocation_tasks
        assert plugin._invocation_tasks.get("inv-X") is None
    finally:
        fake_task.cancel()
        try:
            await fake_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# 9. Structured _InvocationCancelled marker exists and carries the id
# ---------------------------------------------------------------------------


def test_invocation_cancelled_marker_dataclass() -> None:
    """The structured marker is the canonical conversion of
    ``CancelledError`` at the boundary. Documents the catch-site
    contract."""
    marker = _InvocationCancelled(invocation_id="inv-X", detail="LLM_CALL_TIMEOUT")
    assert marker.invocation_id == "inv-X"
    assert marker.reason == "cancelled"
    assert marker.detail == "LLM_CALL_TIMEOUT"
    # Frozen dataclass — mutation is rejected.
    import dataclasses as _dc

    with pytest.raises(_dc.FrozenInstanceError):
        marker.invocation_id = "inv-Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 10. Boundary entry survives no-active-context (defensive default)
# ---------------------------------------------------------------------------


async def test_boundary_entry_no_active_context_is_silent() -> None:
    """Without an active SessionContext, the boundary emit is a no-op
    (not a raise) — matches the "observability never blocks a callback"
    contract."""
    plugin = make_adk_plugin(host_agent_name="coordinator")
    # No set_active_context — _active_ctx is None.
    cb_ctx = _FakeCallbackContext(
        invocation_context=_FakeInvocationContext(invocation_id="inv-A"),
    )
    agent = _FakeAgent(name="coordinator")
    # Must not raise.
    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)
    await plugin.after_agent_callback(agent=agent, callback_context=cb_ctx)
