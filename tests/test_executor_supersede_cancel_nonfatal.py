"""Tests for the goldfive-internal supersede-cancel non-fatal contract.

Bug A from v22 validation (session
``49b0eb10-5636-465d-b96b-9e9d03d91e81``): a goldfive-internal cancel
fired by :meth:`DefaultSteerer._cancel_inflight_for_revision` to switch
the in-flight invocation onto a freshly-installed revised plan was
falling through the executor's overlay ``cancelled`` branch and
emitting ``run_aborted``, terminating the user's turn — even though
the supersede contract is "switch to the new plan, do not end the
turn."

Fix shape (mirrors PR #332's ``fail_fast_on_revision_rejection``
principle): the steerer stamps ``session._supersede_pending = True``
inside ``_cancel_inflight_for_revision``; the executor's overlay
cancelled branch reads the flag, treats the cancel as a restart
trigger (resets the reconciler, ``continue``s the loop on the new
plan), and clears the flag. External cancels (USER_CANCEL via the
control channel, asyncio.CancelledError from the caller) NEVER set
the flag, so they retain the legacy abort behaviour. Opt-in
``fail_fast_on_invoke_cancel`` (kwarg or
``GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL=1`` env) restores the pre-fix
abort even on a supersede.

These tests pin:

* Default + supersede flag → restart, no ``run_aborted``.
* Default + no flag (external cancel via control channel) →
  ``run_aborted`` (current behaviour preserved).
* ``fail_fast_on_invoke_cancel=True`` + supersede flag → still abort.
* Env var ``GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL=1`` honoured.
* No double-clear: after a supersede restart, ``_supersede_pending``
  is False so a subsequent (external) cancel is correctly classified.
* Steerer's ``_cancel_inflight_for_revision`` actually stamps the
  flag (the supersede contract from the docstring).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from goldfive.control import (
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.executors.sequential import SequentialExecutor
from goldfive.protocols import EventSink
from goldfive.results import InvocationResult
from goldfive.steerer import DefaultSteerer
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs (mirror tests/test_overlay_steer.py shapes).
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.closed = False

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        self.closed = True

    def payload_kinds(self) -> list[str]:
        kinds: list[str] = []
        for e in self.events:
            if hasattr(e, "WhichOneof"):
                kinds.append(e.WhichOneof("payload") or "")
            elif isinstance(e, dict):
                kinds.append(e.get("kind", ""))
            else:
                kinds.append(getattr(e, "payload_kind", ""))
        return kinds


class _MinimalStubSteerer:
    """Bare steerer that records observations but does no drift work.

    These tests drive the executor directly with pre-arranged session
    state — they don't need the full DefaultSteerer pipeline. The
    steerer surface here is the minimum the executor calls into.
    """

    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.observed: list[Any] = []

    def bind(self, *, sinks: list[EventSink], planner: Any) -> None:
        self._sinks = sinks
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:  # noqa: ARG002
        self.observed.append(event)

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:  # noqa: ARG002
        return None

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",  # noqa: ARG002
        session: Session,
        cancel_reason: str = "",  # noqa: ARG002
    ) -> None:
        if session.plan is None:
            return
        # goldfive#247: Plan + Task are frozen — derive new plan + swap.
        if not any(t.id == task_id for t in session.plan.tasks):
            return
        from goldfive.types import (
            channel_processor_active,
            set_session_plan,
            with_task_status,
        )
        with channel_processor_active():
            set_session_plan(session, with_task_status(session.plan, task_id, to))

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:  # noqa: ARG002
        return None


class _StubPlanner:
    async def generate(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class _OverlayStubAdapter:
    def __init__(
        self,
        *,
        passthrough_effect: Callable[
            [str, Session, Any], Awaitable[InvocationResult | None]
        ]
        | None = None,
    ) -> None:
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


# ---------------------------------------------------------------------------
# Plan helpers.
# ---------------------------------------------------------------------------


def _two_task_plan(plan_id: str = "p0", revision_index: int = 0) -> Plan:
    return Plan(
        id=plan_id,
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="task 0", assignee_agent_id="agent_t0"),
            Task(id="t1", title="task 1", assignee_agent_id="agent_t1"),
        ],
        edges=[TaskEdge(from_task_id="t0", to_task_id="t1")],
        revision_index=revision_index,
    )


def _revised_plan(revision_index: int = 1) -> Plan:
    return Plan(
        id=f"p{revision_index}",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="r0", title="revised 0", assignee_agent_id="agent_r0"),
        ],
        edges=[],
        revision_reason="off-topic supersede",
        revision_kind=DriftKind.OFF_TOPIC.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=revision_index,
    )


# ---------------------------------------------------------------------------
# 1. Default behaviour: supersede flag set → executor restarts, no abort.
# ---------------------------------------------------------------------------


async def test_overlay_supersede_cancel_restarts_loop_by_default() -> None:
    """The v22 Bug A repro, in miniature.

    The first passthrough simulates the steerer stamping
    ``session._supersede_pending = True`` (as
    ``DefaultSteerer._cancel_inflight_for_revision`` would in
    production) and then raising CancelledError to mimic the
    asyncio.Task.cancel() the plugin fires. The executor MUST treat
    this as a restart, not an abort.
    """
    plan = _two_task_plan()
    session = Session(run_id="r1")
    # goldfive#247: route through helper
    from tests._immutable_plan_helpers import force_plan as _fp_helper
    _fp_helper(session, plan)
    refined = _revised_plan()
    steerer = _MinimalStubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        # First invocation: simulate the steerer's supersede pipeline:
        # 1. swap session.plan to the refined plan (mirrors
        #    DefaultSteerer._apply_revision running before the cancel).
        # 2. stamp _supersede_pending (mirrors
        #    _cancel_inflight_for_revision before request_invocation_cancel).
        # 3. raise CancelledError (mirrors the plugin firing
        #    task.cancel() on the registered asyncio.Task).
        if len(adapter.passthrough_calls) == 1:
            # goldfive#247: route through helper
            from tests._immutable_plan_helpers import force_plan as _fp_helper
            _fp_helper(session, refined)
            session._supersede_pending = True  # type: ignore[attr-defined]
            raise asyncio.CancelledError()
        # Second invocation: run the revised plan to completion. The
        # reconciler was reset by the executor on the supersede
        # restart, so claim agent_r0's task fresh.
        if session.plan is not None:
            for t in list(session.plan.tasks):
                await reconciler.on_before_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
                await reconciler.on_after_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
        return InvocationResult(task_id="", text="revised plan done")

    adapter = _OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    outcome = await asyncio.wait_for(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=_StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="original goal",
        ),
        timeout=5.0,
    )

    # No abort — the supersede was treated as a restart trigger.
    assert outcome.success is True, (
        f"expected success after supersede restart, got reason={outcome.reason!r}"
    )
    # Two passthrough invocations: pre-supersede (cancelled) and
    # post-supersede (completed).
    assert len(adapter.passthrough_calls) == 2, (
        f"expected 2 invocations (cancel + restart), got {adapter.passthrough_calls}"
    )
    # No run_aborted in the sink stream.
    assert "run_aborted" not in sink.payload_kinds(), (
        f"unexpected run_aborted in sink stream: {sink.payload_kinds()}"
    )
    # Final event is run_completed.
    assert sink.payload_kinds()[-1] == "run_completed"
    # session.plan is the revised plan and its tasks are terminal.
    # goldfive#247: identity check replaced with id check (Plan is frozen).
    # Read tasks from session.plan (live) — the local ``refined`` is
    # the pre-mutation snapshot.
    assert session.plan is not None and session.plan.id == refined.id
    for t in session.plan.tasks:
        assert t.status in (TaskStatus.COMPLETED, TaskStatus.NOT_NEEDED), (
            f"revised task {t.id} ended in {t.status}"
        )
    # Flag was consumed (cleared) by the executor on restart.
    assert getattr(session, "_supersede_pending", False) is False, (
        "executor must clear _supersede_pending after consuming it"
    )


# ---------------------------------------------------------------------------
# 2. External cancel (no supersede flag) → abort preserved.
# ---------------------------------------------------------------------------


async def test_overlay_external_cancel_still_aborts() -> None:
    """A USER_CANCEL on the control channel does not set
    ``_supersede_pending`` and MUST still emit ``run_aborted`` —
    external cancels reflect a genuine decision to end the run.
    """
    plan = _two_task_plan()
    session = Session(run_id="r1")
    # goldfive#247: route through helper
    from tests._immutable_plan_helpers import force_plan as _fp_helper
    _fp_helper(session, plan)
    steerer = _MinimalStubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()

    started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        return InvocationResult(task_id="", text="unreachable")

    adapter = _OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=_StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="go",
        )
    )
    await started.wait()
    # Send an external CANCEL — the supersede flag is NOT set.
    await channel.send(
        ControlMessage(kind=ControlKind.CANCEL, payload={"reason": "user aborted"})
    )

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    # Aborted as before #332's principle DOES NOT extend to external
    # cancels — those reflect an external decision.
    assert outcome.success is False
    assert "run_aborted" in sink.payload_kinds()
    # Only the original passthrough — no restart.
    assert len(adapter.passthrough_calls) == 1


# ---------------------------------------------------------------------------
# 3. fail_fast_on_invoke_cancel=True → abort even on supersede.
# ---------------------------------------------------------------------------


async def test_overlay_supersede_cancel_aborts_when_fail_fast_kwarg_set() -> None:
    plan = _two_task_plan()
    session = Session(run_id="r1")
    # goldfive#247: route through helper
    from tests._immutable_plan_helpers import force_plan as _fp_helper
    _fp_helper(session, plan)
    steerer = _MinimalStubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        # Simulate the supersede contract: stamp the flag and cancel.
        session._supersede_pending = True  # type: ignore[attr-defined]
        raise asyncio.CancelledError()

    adapter = _OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(
        overlay_mode=True, fail_fast_on_invoke_cancel=True
    )

    outcome = await asyncio.wait_for(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=_StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="go",
        ),
        timeout=5.0,
    )

    # Strict mode: even a supersede aborts.
    assert outcome.success is False
    assert "run_aborted" in sink.payload_kinds()
    assert len(adapter.passthrough_calls) == 1
    # Flag cleared so a downstream Session reuse doesn't inherit it.
    assert getattr(session, "_supersede_pending", False) is False


# ---------------------------------------------------------------------------
# 4. Env var GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL=1 honoured.
# ---------------------------------------------------------------------------


async def test_overlay_env_var_enables_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL", "1")

    plan = _two_task_plan()
    session = Session(run_id="r1")
    # goldfive#247: route through helper
    from tests._immutable_plan_helpers import force_plan as _fp_helper
    _fp_helper(session, plan)
    steerer = _MinimalStubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        session._supersede_pending = True  # type: ignore[attr-defined]
        raise asyncio.CancelledError()

    adapter = _OverlayStubAdapter(passthrough_effect=_passthrough)
    # Construct AFTER setenv so the constructor reads the env.
    executor = SequentialExecutor(overlay_mode=True)
    assert executor._fail_fast_on_invoke_cancel is True

    outcome = await asyncio.wait_for(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=_StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="go",
        ),
        timeout=5.0,
    )

    assert outcome.success is False
    assert "run_aborted" in sink.payload_kinds()


# ---------------------------------------------------------------------------
# 5. Explicit kwarg=False overrides env var (kwarg wins).
# ---------------------------------------------------------------------------


async def test_overlay_kwarg_false_overrides_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL", "1")
    executor = SequentialExecutor(
        overlay_mode=True, fail_fast_on_invoke_cancel=False
    )
    assert executor._fail_fast_on_invoke_cancel is False


# ---------------------------------------------------------------------------
# 6. Default (no kwarg, no env) → False.
# ---------------------------------------------------------------------------


def test_overlay_default_fail_fast_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL", raising=False)
    executor = SequentialExecutor(overlay_mode=True)
    assert executor._fail_fast_on_invoke_cancel is False


# ---------------------------------------------------------------------------
# 7. No double-clear: subsequent external cancel after supersede
#    restart is correctly classified.
# ---------------------------------------------------------------------------


async def test_overlay_supersede_then_external_cancel_aborts() -> None:
    """After a supersede restart, ``_supersede_pending`` is False. A
    subsequent external cancel on the restarted invocation MUST abort
    (it has no supersede marker)."""
    plan = _two_task_plan()
    session = Session(run_id="r1")
    # goldfive#247: route through helper
    from tests._immutable_plan_helpers import force_plan as _fp_helper
    _fp_helper(session, plan)
    refined = _revised_plan()
    steerer = _MinimalStubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()

    second_started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        if len(adapter.passthrough_calls) == 1:
            # goldfive#247: route through helper
            from tests._immutable_plan_helpers import force_plan as _fp_helper
            _fp_helper(session, refined)
            session._supersede_pending = True  # type: ignore[attr-defined]
            raise asyncio.CancelledError()
        # Second invocation: signal then block. The test will then
        # send an external CANCEL.
        second_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        return InvocationResult(task_id="", text="unreachable")

    adapter = _OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=_StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="go",
        )
    )
    await second_started.wait()
    # Sanity: flag was cleared by the supersede restart.
    assert getattr(session, "_supersede_pending", False) is False
    # External cancel on the restarted invocation.
    await channel.send(
        ControlMessage(kind=ControlKind.CANCEL, payload={"reason": "user aborted"})
    )

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    # External cancel after the restart aborts as expected.
    assert outcome.success is False
    assert "run_aborted" in sink.payload_kinds()
    assert len(adapter.passthrough_calls) == 2


# ---------------------------------------------------------------------------
# 7b. Defensive clear: a stale supersede flag from a STEER branch (where
#     observe → install_user_steer → _cancel_inflight_for_revision sets
#     the flag as a side-effect, but the STEER branch — not the
#     cancelled branch — handles the restart) must NOT misclassify a
#     subsequent external cancel as a supersede.
# ---------------------------------------------------------------------------


async def test_overlay_stale_supersede_flag_cleared_per_iteration() -> None:
    """If the supersede flag was left set by a previous code path that
    handles its own restart (e.g. the STEER branch), the next external
    CANCEL on a fresh invocation MUST still abort. The executor's
    defensive per-iteration clear ensures this.

    Simulates: iteration 1's effect sets ``_supersede_pending=True``
    AND completes normally (mimicking a STEER-like side effect that
    the cancelled branch never gets to consume). Iteration 2 starts —
    the executor must clear the flag before invoking. Then we send an
    external CANCEL on iteration 2; with the defensive clear in
    place, the cancel branch sees flag=False and correctly aborts.

    Without the per-iteration clear, the stale True flag would route
    iteration 2's external cancel into the supersede-restart branch,
    silently dropping the user's CANCEL.
    """
    plan = _two_task_plan()
    session = Session(run_id="r1")
    # goldfive#247: route through helper
    from tests._immutable_plan_helpers import force_plan as _fp_helper
    _fp_helper(session, plan)
    # Pre-stamp the flag as if some prior side-effect set it. The
    # executor's per-iteration clear at the top of the while loop must
    # wipe this before the first invocation completes.
    session._supersede_pending = True  # type: ignore[attr-defined]
    steerer = _MinimalStubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()

    started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        # Iteration 1: signal then block. Test sends external CANCEL.
        # The defensive clear at the top of the loop should have
        # already wiped the pre-set flag, so the cancel will be
        # correctly classified as external.
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        return InvocationResult(task_id="", text="unreachable")

    adapter = _OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=_StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="go",
        )
    )
    await started.wait()
    # External cancel — flag was True at session entry but the
    # executor cleared it at the top of the loop iteration.
    await channel.send(
        ControlMessage(kind=ControlKind.CANCEL, payload={"reason": "user aborted"})
    )

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    # External cancel correctly aborts (flag was cleared before invoke).
    assert outcome.success is False
    assert "run_aborted" in sink.payload_kinds()
    assert len(adapter.passthrough_calls) == 1


# ---------------------------------------------------------------------------
# 8. Steerer contract: _cancel_inflight_for_revision stamps the flag.
# ---------------------------------------------------------------------------


async def test_steerer_cancel_inflight_stamps_supersede_flag() -> None:
    """The steerer's own ``_cancel_inflight_for_revision`` MUST stamp
    ``session._supersede_pending = True`` before delegating to
    ``request_invocation_cancel``. Without this stamp, the executor's
    cancelled branch can't disambiguate the supersede.
    """
    steerer = DefaultSteerer()
    session = Session(run_id="r1")

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="off topic",
    )

    # No adapter bound → request_invocation_cancel returns []. The
    # supersede stamp must STILL fire (it's set before the delegation).
    flagged = await steerer._cancel_inflight_for_revision(drift, session)
    assert flagged == []
    assert getattr(session, "_supersede_pending", False) is True, (
        "DefaultSteerer._cancel_inflight_for_revision must stamp "
        "session._supersede_pending=True before initiating the cancel"
    )


# ---------------------------------------------------------------------------
# 9. Env var = "0" or unset → False.
# ---------------------------------------------------------------------------


def test_overlay_env_var_zero_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL", "0")
    executor = SequentialExecutor(overlay_mode=True)
    assert executor._fail_fast_on_invoke_cancel is False


# ---------------------------------------------------------------------------
# 10. Truthy non-"1" values are NOT treated as enabled (strict match).
# ---------------------------------------------------------------------------


def test_overlay_env_var_strict_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env-var contract is strict: only the literal "1" enables
    fail-fast (mirrors PR #332's ``GOLDFIVE_FAIL_FAST_REVISION_REJECTION``
    parsing). "true" / "yes" / "on" do NOT enable.
    """
    monkeypatch.setenv("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL", "true")
    executor = SequentialExecutor(overlay_mode=True)
    assert executor._fail_fast_on_invoke_cancel is False


# ---------------------------------------------------------------------------
# Sanity: env teardown isolated.
# ---------------------------------------------------------------------------


def test_env_var_isolation() -> None:
    """Confirm pytest's monkeypatch teardown cleans the env so tests
    that follow this module see a fresh environment.
    """
    # If a previous test leaked the env var, this would fail.
    assert os.environ.get("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL") in (None, "0")
