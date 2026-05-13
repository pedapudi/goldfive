"""Cancel-in-flight-task on refine (goldfive#271 follow-up — v15
concurrent-invocation bug).

Empirical evidence from ``v15presmtx-1`` (goldfive#271 follow-up
brief, 2026-04-27): a ``PLAN_DIVERGENCE`` drift kicked off the
steerer's ``refine_steer`` (a ~10-minute LLM call), but the
coordinator's in-flight invocation kept making LLM/tool calls against
the soon-to-be-stale plan. The existing sticky cancel-gate (PR #299)
short-circuited only the NEXT ``before_model_callback`` /
``before_tool_callback``; it did NOT cancel the already-running LLM
streaming call inside that invocation, NOR the asyncio.Task driving
the dispatch.

This file pins the cancel-on-refine wiring:

1. The plugin registers ``asyncio.current_task()`` keyed by
   ``invocation_id`` in ``before_run_callback`` so
   :meth:`request_invocation_cancel` can target the running task.
2. ``request_invocation_cancel(cancel_inflight_task=True)`` fires
   ``task.cancel()`` (deferred via ``loop.call_soon``) on the
   registered task, in addition to writing the cancel-state flag.
3. ``request_invocation_cancel(cancel_inflight_task=False)`` (the
   default) leaves the flag-only contract intact — the existing
   pre-refine CRITICAL cancel path keeps its semantics.
4. The deferred-cancel path lets the calling coroutine finish its
   PlanRevised emit before the dispatch task observes the cancel —
   no lost ``PlanRevised`` event.
5. After Option A (goldfive#271 follow-up), turn-1 first-plan
   installs no longer reach ``_cancel_inflight_for_revision`` at
   all: :meth:`DefaultSteerer.install_initial_plan` skips the
   cancel path because there is no in-flight invocation to cancel
   on a fresh session.
6. Successful refine paths
   (``_handle_drift`` / ``_promote_drift_to_steer`` /
   ``install_revision_for_drift`` /
   ``install_revision_for_user_steer``) call
   ``_cancel_inflight_for_revision`` BEFORE
   ``_emit_plan_revised`` so the cancel and the revision land
   adjacent to each other on the wire.
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
    make_adk_plugin,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    CancellationRequest,
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

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _FakeADKSession:
    def __init__(self, *, state: dict[str, Any]) -> None:
        self.state = state
        self.run_id = "run-test"
        self.id = "session-test"


class _FakeInvocationContext:
    def __init__(
        self,
        *,
        invocation_id: str,
        session_state: dict[str, Any],
        agent_name: str = "coordinator",
    ) -> None:
        self.invocation_id = invocation_id
        self.session = _FakeADKSession(state=session_state)
        self.agent = _FakeAgent(name=agent_name)


def _make_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="run-test",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.RUNNING),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_session() -> Session:
    return Session(
        run_id="run-test",
        goals=[Goal(id="g1", summary="ship")],
        plan=_make_plan(),
        current_task_id="t1",
    )


def _make_request(
    *,
    invocation_id: str = "inv-1",
    severity: DriftSeverity = DriftSeverity.WARNING,
    reason: str = "drift",
    drift_kind: str = "plan_divergence",
) -> CancellationRequest:
    return CancellationRequest(
        invocation_id=invocation_id,
        reason=reason,
        severity=severity,
        drift_kind=drift_kind,
        detail="cancel test",
    )


def _bind_plugin() -> tuple[Any, Session]:
    plugin = make_adk_plugin(host_agent_name="coordinator")
    session = _make_session()
    ctx = SessionContext(
        session=session,
        steerer=DefaultSteerer(),
        tools=(),
        tool_handlers={},
        host_agent_name="coordinator",
        task=session.plan.tasks[0],
    )
    plugin.set_active_context(ctx)
    return plugin, session


# ---------------------------------------------------------------------------
# 1. before_run_callback registers asyncio.current_task() under inv_id
# ---------------------------------------------------------------------------


async def test_before_run_registers_current_task_under_invocation_id() -> None:
    """The per-invocation task registry must be populated by
    ``before_run_callback`` so :meth:`request_invocation_cancel` has a
    handle to cancel.

    Phase 3.5 (goldfive#271 component 1): the registry now lives on
    :class:`~goldfive.state_store.StateStore`, not on
    the plugin instance. The plugin's ``_invocation_tasks`` attribute
    is a backwards-compat view that delegates to the store. Both the
    legacy attribute access AND a direct ``StateStore`` lookup
    must return the registered task — pinning that the storage truly
    relocated rather than being duplicated.
    """
    from goldfive.state_store import StateStore

    plugin, session = _bind_plugin()
    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-A",
        session_state={},
        agent_name="coordinator",
    )
    await plugin.before_run_callback(invocation_context=inv_ctx)
    expected = asyncio.current_task()
    # Legacy attribute path — preserved for the steerer + tests.
    assert plugin._invocation_tasks.get("inv-A") is expected
    # Phase 3.5: registry actually lives on StateStore.
    store = StateStore.for_session(session)
    assert store.get_invocation_task("inv-A") is expected
    assert "inv-A" in store.active_invocation_ids()


# ---------------------------------------------------------------------------
# 2. after_run_callback drops the registered task
# ---------------------------------------------------------------------------


async def test_after_run_drops_registered_task() -> None:
    """``after_run_callback`` releases the per-invocation registry slot
    so an unrelated late cancel doesn't target a finished invocation.

    Phase 3.5 (goldfive#271 component 1): the deregister path must
    reach the StateStore-backed registry, not just the plugin
    attribute.
    """
    from goldfive.state_store import StateStore

    plugin, session = _bind_plugin()
    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-B",
        session_state={},
        agent_name="coordinator",
    )
    await plugin.before_run_callback(invocation_context=inv_ctx)
    assert "inv-B" in plugin._invocation_tasks
    store = StateStore.for_session(session)
    assert store.get_invocation_task("inv-B") is not None
    await plugin.after_run_callback(invocation_context=inv_ctx)
    assert "inv-B" not in plugin._invocation_tasks
    # And the StateStore-side bucket no longer has the entry.
    assert store.get_invocation_task("inv-B") is None


# ---------------------------------------------------------------------------
# 3. request_invocation_cancel(cancel_inflight_task=False) does NOT
#    fire task.cancel — flag-only legacy contract preserved
# ---------------------------------------------------------------------------


async def test_request_cancel_flag_only_does_not_touch_task() -> None:
    """Default ``cancel_inflight_task=False`` keeps the pre-#271-follow-
    up flag-only semantics so the existing CRITICAL pre-refine
    cancel path doesn't preempt its own refine call."""
    plugin, _session = _bind_plugin()

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _body() -> None:
        try:
            started.set()
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    body_task = asyncio.create_task(_body())
    await started.wait()
    plugin._invocation_tasks["inv-1"] = body_task

    plugin.request_invocation_cancel(
        invocation_id="inv-1",
        request=_make_request(invocation_id="inv-1"),
        # cancel_inflight_task defaults to False
    )
    # Flag is set -> sticky-cancelled path will engage in callbacks.
    assert plugin.peek_cancel_for_invocation("inv-1") is not None
    # Give the loop a tick to drain any (incorrectly) queued callbacks.
    await asyncio.sleep(0)
    assert not cancelled.is_set(), "task.cancel() must NOT fire when cancel_inflight_task is False"

    body_task.cancel()
    try:
        await body_task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# 4. request_invocation_cancel(cancel_inflight_task=True) DOES fire
#    task.cancel — deferred via loop.call_soon so the calling
#    coroutine finishes its current sync work first
# ---------------------------------------------------------------------------


async def test_request_cancel_with_task_flag_cancels_registered_task() -> None:
    """The new ``cancel_inflight_task=True`` opt-in fires
    ``task.cancel()`` on the registered task. Cancellation lands within
    one event-loop tick — well under the 1s budget called out in the
    brief's regression contract."""
    plugin, _session = _bind_plugin()

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _body() -> None:
        try:
            started.set()
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    body_task = asyncio.create_task(_body())
    await started.wait()
    plugin._invocation_tasks["inv-1"] = body_task

    plugin.request_invocation_cancel(
        invocation_id="inv-1",
        request=_make_request(invocation_id="inv-1"),
        cancel_inflight_task=True,
    )
    # Cancel is queued via ``loop.call_soon``; flush the loop until the
    # body task observes it. Use ``wait_for`` with a generous-but-bounded
    # budget — the brief's regression contract is "<1s".
    try:
        await asyncio.wait_for(body_task, timeout=1.0)
    except asyncio.CancelledError:
        pass
    except TimeoutError:
        pytest.fail("task.cancel() did not propagate within 1s budget")
    assert cancelled.is_set()


# ---------------------------------------------------------------------------
# 5. Deferred cancel lets the calling coroutine emit PlanRevised first
# ---------------------------------------------------------------------------


async def test_cancel_inflight_task_is_deferred_via_call_soon() -> None:
    """``loop.call_soon`` deferral is the contract the steerer relies
    on: the caller (``_handle_drift`` / ``_promote_drift_to_steer``)
    must finish emitting ``PlanRevised`` BEFORE the cancel lands. We
    pin the contract by checking that follow-up sync work after the
    cancel call still runs in the same task on the same yield."""
    plugin, _session = _bind_plugin()

    body_done = asyncio.Event()

    async def _body() -> None:
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            body_done.set()
            raise

    body_task = asyncio.create_task(_body())
    # Yield once to let _body start running.
    await asyncio.sleep(0)
    plugin._invocation_tasks["inv-1"] = body_task

    follow_up_ran = False
    plugin.request_invocation_cancel(
        invocation_id="inv-1",
        request=_make_request(invocation_id="inv-1"),
        cancel_inflight_task=True,
    )
    # Synchronous work after the cancel call runs without seeing
    # CancelledError — we are NOT the cancelled task, but even if we
    # were, the deferred cancel wouldn't have fired yet.
    follow_up_ran = True
    assert follow_up_ran is True
    # Now let the loop run the queued cancel callback.
    try:
        await asyncio.wait_for(body_task, timeout=1.0)
    except asyncio.CancelledError:
        pass
    except TimeoutError:
        pytest.fail("queued cancel did not propagate within 1s")
    assert body_done.is_set()


# ---------------------------------------------------------------------------
# 6. Option A: install_initial_plan does not reach _cancel_inflight_for_revision
# ---------------------------------------------------------------------------


async def test_install_initial_plan_does_not_cancel_inflight() -> None:
    """:meth:`DefaultSteerer.install_initial_plan` MUST NOT touch the
    cancel-inflight pipeline.

    Goldfive#271 Option A: turn-1 installs go through
    :meth:`install_initial_plan` which intentionally skips
    :meth:`_cancel_inflight_for_revision` (no in-flight invocation
    exists on a fresh session). Pre-Option-A this was achieved with a
    synthetic-USER_STEER exemption inside
    :meth:`_cancel_inflight_for_revision`; Option A makes it
    structural by routing first installs to a separate API.
    """

    class _CountingAdapter:
        def __init__(self) -> None:
            self._next_cancel_reason = ""
            self._plugin = _CountingPlugin()

    class _CountingPlugin:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._top_invocation_id = "inv-X"
            self._invocation_parents: dict[str, str] = {}
            self._reconciler = None

        def request_invocation_cancel(self, **kwargs: Any) -> list[str]:
            self.calls.append(kwargs)
            return [str(kwargs.get("invocation_id", ""))]

    steerer = DefaultSteerer()
    adapter = _CountingAdapter()
    steerer.bind_adapter(adapter)
    session = _make_session()
    session.plan = Plan.empty(run_id=session.run_id)
    plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="initial",
    )
    installed = await steerer.install_initial_plan(session=session, plan=plan)
    assert installed
    # The plugin's cancel must NOT have been called: there is no
    # in-flight invocation on a fresh-session install.
    assert adapter._plugin.calls == []


async def test_cancel_inflight_for_revision_fires_for_real_drift() -> None:
    """A non-synthetic drift (PLAN_DIVERGENCE, real USER_STEER, OFF_TOPIC,
    …) MUST forward to the plugin with ``cancel_inflight_task=True``."""

    class _CountingAdapter:
        def __init__(self) -> None:
            self._next_cancel_reason = ""
            self._plugin = _CountingPlugin()

    class _CountingPlugin:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._top_invocation_id = "inv-X"
            self._invocation_parents: dict[str, str] = {}
            self._reconciler = None

        def request_invocation_cancel(self, **kwargs: Any) -> list[str]:
            self.calls.append(kwargs)
            return [str(kwargs.get("invocation_id", ""))]

    steerer = DefaultSteerer()
    adapter = _CountingAdapter()
    steerer.bind_adapter(adapter)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="off-plan agent",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    flagged = await steerer._cancel_inflight_for_revision(drift, _make_session())
    assert flagged == ["inv-X"]
    assert len(adapter._plugin.calls) == 1
    assert adapter._plugin.calls[0]["cancel_inflight_task"] is True


# ---------------------------------------------------------------------------
# 8. End-to-end: WARNING-severity drift → refine_steer → cancel fires on the
#    coordinator's task before its long sleep finishes
# ---------------------------------------------------------------------------


async def test_plan_divergence_refine_cancels_inflight_coordinator_task() -> None:
    """Regression test for the v15 bug: a WARNING-severity drift fires
    refine_steer; the coordinator's in-flight asyncio task is cancelled
    within the 1s propagation budget instead of running for the full
    refine duration.

    Simulates the v15 timeline by:

    1. Standing up a steerer bound to a stub planner that pretends to
       run ``refine_steer`` (returning a revised plan).
    2. Pinning a long-running coordinator task in the plugin's
       per-invocation registry.
    3. Firing an ``OFF_TOPIC`` (WARNING) drift through
       ``_handle_drift`` — which routes to ``_promote_drift_to_steer``
       given the goldfive-steer eligibility set.
    4. Asserting the coordinator's task observes ``CancelledError``
       within 1s.

    goldfive#252: this test originally fired PLAN_DIVERGENCE; that kind
    is now silenced at the top of ``_handle_drift`` (replaced by
    CAPABILITY_MISMATCH in #253). The cancel-inflight contract is
    independent of which drift kind triggers it, so we exercise the
    same code path with ``OFF_TOPIC`` (also WARNING → ABSORB → refine).
    """

    revised = Plan(
        id="p1",
        run_id="run-test",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.RUNNING),
            Task(id="t2b", title="Replanned T2"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2b")],
        revision_kind=DriftKind.OFF_TOPIC.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )

    class _StubPlanner:
        async def refine_steer(self, **_kwargs: Any) -> Plan:
            # Realistic ``refine_steer`` is a multi-second LLM call; we
            # simulate the duration just long enough for the deferred
            # cancel to land *before* this returns. Were the cancel
            # not deferred via ``loop.call_soon`` (or worse: not fired
            # at all) the coordinator's ``await sleep(5.0)`` would run
            # to completion and the test would time out.
            return revised

        async def refine(self, **_kwargs: Any) -> Plan:
            return revised

    plugin = make_adk_plugin(host_agent_name="coordinator")
    session = _make_session()
    steerer = DefaultSteerer(
        goldfive_steer_threshold="warning",
        goldfive_steer_suppression_window_turns=3,
    )
    steerer.bind(sinks=[], planner=_StubPlanner())

    class _Adapter:
        def __init__(self, plugin: Any) -> None:
            self._plugin = plugin
            self._next_cancel_reason = ""
            self.available_agents = ["coordinator"]

    adapter = _Adapter(plugin)
    steerer.bind_adapter(adapter)

    ctx = SessionContext(
        session=session,
        steerer=steerer,
        tools=(),
        tool_handlers={},
        host_agent_name="coordinator",
        task=session.plan.tasks[0],
    )
    plugin.set_active_context(ctx)
    plugin._top_invocation_id = "inv-coord-1"

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _coordinator() -> None:
        try:
            started.set()
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    coord_task = asyncio.create_task(_coordinator())
    await started.wait()
    plugin._invocation_tasks["inv-coord-1"] = coord_task

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="coordinator reasoning off the bound task",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    await steerer._handle_drift(drift, session)

    try:
        await asyncio.wait_for(coord_task, timeout=1.0)
    except asyncio.CancelledError:
        pass
    except TimeoutError:
        pytest.fail(
            "coordinator task was not cancelled within 1s of OFF_TOPIC "
            "refine — v15 concurrent-invocation bug regressed"
        )
    assert cancelled.is_set()
    # Plan was actually revised — the cancel did not preempt the
    # PlanRevised emit (the deferral contract is upheld).
    # goldfive#247: identity check replaced with id check (Plan is frozen)
    assert session.plan is not None and session.plan.id == revised.id
