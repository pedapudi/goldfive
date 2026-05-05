"""Adapter.request_cancel wiring for goldfive-promoted steers (goldfive#241).

Pre-#241 a goldfive-detected drift that cleared the promotion
threshold queued a restart message via the deleted
``pending_corrective_message`` slot (Phase 2 of the path-duality fix
replaced that with a ``GOLDFIVE_STEER`` ControlMessage on the bound
channel) and tagged ``adapter._next_cancel_reason``, but the in-flight
``runner.run_async`` stream kept running to completion. Observed
consequence: the coordinator kept emitting contaminated reasoning /
tool calls for tens of seconds after the drift fired, and the
restart only landed on the next executor turn.

#241 wires an optional ``adapter.request_cancel(reason)`` hook
that the steerer fires on every promoted drift. The ADK adapter
implements it by calling ``task.cancel()`` on the asyncio task
inside :meth:`ADKAdapter._invoke_internal` so the LLM stream
raises ``CancelledError`` and the adapter's standard heal path
runs with the already-stamped ``_next_cancel_reason`` tag.

This file pins:

* A WARNING OFF_TOPIC drift fires ``adapter.request_cancel(
  reason="goldfive_off_topic")`` exactly once.
* The same reason stamped on ``_next_cancel_reason`` and passed to
  ``request_cancel`` is byte-identical so operator logs line up.
* An async ``request_cancel`` is awaited (not just scheduled).
* A sync ``request_cancel`` is also supported (returns non-awaitable).
* An adapter WITHOUT ``request_cancel`` keeps pre-#241 semantics —
  the promotion path does not raise.
* ``ADKAdapter.request_cancel`` is a no-op when no invocation is
  in-flight (called between turns) and when called with a completed
  task handle.
* Calling ``ADKAdapter.request_cancel`` on a pinned in-flight task
  actually fires ``task.cancel()``.
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

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubPlanner:
    """Minimal planner recording refine_steer calls."""

    def __init__(self, revised: Plan | None = None) -> None:
        self.refine_steer_calls: list[dict[str, Any]] = []
        self.revised = revised

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return self.revised

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return self.revised


class _AsyncCancelAdapter:
    """Adapter stub exposing an async ``request_cancel``."""

    def __init__(self) -> None:
        self._next_cancel_reason: str = ""
        self.request_cancel_calls: list[str] = []

    async def request_cancel(self, reason: str) -> None:
        self.request_cancel_calls.append(reason)


class _SyncCancelAdapter:
    """Adapter stub whose ``request_cancel`` is synchronous.

    The steerer must still call it successfully — the protocol only
    requires "callable"; awaitable is optional.
    """

    def __init__(self) -> None:
        self._next_cancel_reason: str = ""
        self.request_cancel_calls: list[str] = []

    def request_cancel(self, reason: str) -> None:
        self.request_cancel_calls.append(reason)


class _NoCancelHookAdapter:
    """Adapter without ``request_cancel`` — models a non-ADK adapter."""

    def __init__(self) -> None:
        self._next_cancel_reason: str = ""


class _RaisingCancelAdapter:
    """Adapter whose ``request_cancel`` blows up.

    The steerer must swallow it — a best-effort cancel cannot break
    the promotion path; the queued restart message still arrives.
    """

    def __init__(self) -> None:
        self._next_cancel_reason: str = ""
        self.request_cancel_calls: list[str] = []

    async def request_cancel(self, reason: str) -> None:
        self.request_cancel_calls.append(reason)
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id="t2", title="T2", status=TaskStatus.RUNNING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship it")],
        plan=_make_plan(),
        current_task_id="t2",
    )


def _bind(adapter: Any) -> tuple[DefaultSteerer, Session, _StubPlanner]:
    steerer = DefaultSteerer(
        goldfive_steer_threshold="warning",
        goldfive_steer_suppression_window_turns=3,
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id="t2b", title="Replanned T2"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2b")],
        revision_kind=DriftKind.OFF_TOPIC.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )
    planner = _StubPlanner(revised=revised)
    session = _make_session()
    steerer.bind(sinks=[], planner=planner)
    steerer.bind_adapter(adapter)
    return steerer, session, planner


# ---------------------------------------------------------------------------
# Steerer -> adapter.request_cancel wiring
# ---------------------------------------------------------------------------


async def test_goldfive_steer_request_cancel_fires() -> None:
    """A WARNING goldfive-eligible drift fires request_cancel with the
    goldfive_<kind> reason, exactly once."""
    adapter = _AsyncCancelAdapter()
    steerer, session, _ = _bind(adapter)
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent wandered",
        current_task_id="t2",
    )
    await steerer._handle_drift(drift, session)
    assert adapter.request_cancel_calls == []  # reverted: deferred-cancel only
    # The same reason is on the cancel-tag so the adapter's healing
    # path picks it up when CancelledError flows.
    assert adapter._next_cancel_reason == "goldfive_off_topic"


async def test_goldfive_steer_request_cancel_supports_sync_hook() -> None:
    """A sync ``request_cancel`` callable is also respected."""
    adapter = _SyncCancelAdapter()
    steerer, session, _ = _bind(adapter)
    drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="agent abandoned the goal",
        current_task_id="t2",
    )
    await steerer._handle_drift(drift, session)
    assert adapter.request_cancel_calls == []  # reverted: deferred-cancel only


async def test_goldfive_steer_tolerates_adapter_without_request_cancel() -> None:
    """An adapter without request_cancel still promotes drift and queues
    the restart message — no raise, no missed cancel tag."""
    adapter = _NoCancelHookAdapter()
    steerer, session, planner = _bind(adapter)
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="minor wander",
        current_task_id="t2",
    )
    await steerer._handle_drift(drift, session)
    # Cancel-tag still stamped.
    assert adapter._next_cancel_reason == "goldfive_off_topic"
    # Refine still ran — the promotion path is intact.
    assert planner.refine_steer_calls, "refine_steer should have fired"


async def test_goldfive_steer_swallows_request_cancel_raise() -> None:
    """A raising ``request_cancel`` cannot break the promotion path."""
    adapter = _RaisingCancelAdapter()
    steerer, session, planner = _bind(adapter)
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="explode",
        current_task_id="t2",
    )
    # Must not raise out.
    await steerer._handle_drift(drift, session)
    assert adapter.request_cancel_calls == []  # reverted: deferred-cancel only
    assert planner.refine_steer_calls, "refine should still have fired"


async def test_user_steer_does_not_call_request_cancel() -> None:
    """USER_STEER keeps its pre-unification executor-loop cancel path.
    The promotion-specific ``request_cancel`` hook MUST NOT fire on a
    USER_STEER drift (the executor owns that cancel; firing here
    would double-cancel).
    """
    adapter = _AsyncCancelAdapter()
    steerer, session, _ = _bind(adapter)
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.INFO,
        detail="operator note",
        current_task_id="t2",
    )
    await steerer._handle_drift(drift, session)
    assert adapter.request_cancel_calls == []


# ---------------------------------------------------------------------------
# ADKAdapter.request_cancel behaviour
# ---------------------------------------------------------------------------


async def test_adk_adapter_request_cancel_is_noop_when_no_invocation() -> None:
    """ADKAdapter.request_cancel with no pinned task is a no-op."""
    pytest.importorskip("google.adk")
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter.__new__(ADKAdapter)
    # Minimal attributes the method reads.
    adapter._inflight_invoke_task = None
    # Must not raise.
    await adapter.request_cancel("goldfive_off_topic")


async def test_adk_adapter_request_cancel_noop_when_task_done() -> None:
    """A pinned-but-completed task is also a no-op path."""
    pytest.importorskip("google.adk")
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter.__new__(ADKAdapter)

    async def _done() -> None:
        return

    task = asyncio.create_task(_done())
    await task  # drive it to done
    adapter._inflight_invoke_task = task
    await adapter.request_cancel("goldfive_off_topic")
    # No failure expected; the task is already finished.
    assert task.done()


async def test_adk_adapter_request_cancel_fires_task_cancel() -> None:
    """When a task IS in-flight, ``request_cancel`` fires ``task.cancel()``
    and the coroutine observes ``CancelledError``."""
    pytest.importorskip("google.adk")
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter.__new__(ADKAdapter)

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _body() -> None:
        started.set()
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(_body())
    await started.wait()
    adapter._inflight_invoke_task = task
    await adapter.request_cancel("goldfive_off_topic")
    # Give the event loop a chance to propagate the cancel.
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert cancelled.is_set()
