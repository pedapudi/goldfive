"""Shared control-message helpers for executors.

Both :class:`~goldfive.executors.SequentialExecutor` and
:class:`~goldfive.executors.ParallelDAGExecutor` interpret the same
:class:`~goldfive.control.ControlMessage` vocabulary. This module centralizes
the dispatch logic so the two executors stay in sync.

The helpers here:

* :func:`dispatch_control` — handle one message, mutate session state
  and/or call into the steerer, and return a :class:`ControlOutcome`
  the executor should apply. The ack is pre-built on the outcome; the
  caller publishes it via ``channel.ack(outcome.ack)``.
* :func:`drain_controls` — non-blocking drain of every message currently
  queued on the channel, dispatching each in order and returning the
  outcomes for the caller.
* :class:`_ControlCancelled` — internal exception the executor raises
  when it needs to short-circuit the run loop on CANCEL.
* :class:`ControlOutcome` — rich per-message result the caller uses to
  decide what to do next (abort the run, apply a steer, rewind, ...).

Control kinds recognized (strings match both ``ControlKind`` enum
members and their raw string equivalents — ``ControlKind`` is a
``StrEnum``):

* ``PAUSE`` / ``RESUME`` — pause / unpause the run loop between tasks.
* ``CANCEL`` — abort the run; executor emits ``RunAborted``.
* ``STEER`` — feed the message to ``steerer.observe`` so the planner
  can produce a fresh plan on the ``USER_STEER`` drift.
* ``REWIND_TO`` — mark a task and every downstream task ``PENDING``
  so the executor re-walks them.
* ``STATUS_QUERY`` — synthesize a status-snapshot event on the sinks
  (forward-compat; not in the Phase-1 ``ControlKind`` enum, but
  recognized if an external caller sends the string).
* ``INTERCEPT_TRANSFER`` — toggle ``session._intercept_transfer`` so
  adapters that honour the flag refuse transfers.
* ``APPROVE`` / ``REJECT`` — resolve a pending human-in-the-loop
  approval registered on ``session.pending_approvals`` (either a
  ``report_awaiting_approval`` task-level waiter, Flow A, or an ADK
  ``require_confirmation=True`` tool-level waiter, Flow B). Emits
  ``ApprovalGranted`` / ``ApprovalRejected`` through the sinks.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

from goldfive.types import Session, TaskStatus

if TYPE_CHECKING:
    from goldfive.control import ControlAck, ControlChannel, ControlMessage
    from goldfive.protocols import EventSink, Steerer

log = logging.getLogger(__name__)


__all__ = [
    "ControlOutcome",
    "_ControlCancelled",
    "dispatch_control",
    "drain_controls",
    "emit_status_report",
]


class _ControlCancelled(BaseException):
    """Internal signal: a CANCEL control message reached the executor.

    Uses :class:`BaseException` (not :class:`Exception`) so stray ``except
    Exception`` handlers inside adapter code do not swallow it.
    """

    def __init__(self, detail: str = "cancelled by control") -> None:
        super().__init__(detail)
        self.detail = detail


@dataclasses.dataclass
class ControlOutcome:
    """Per-message dispatch result the caller folds into its run loop."""

    # The ack that should be published back on the channel.
    ack: ControlAck
    # True when the executor should stop its outer loop and abort the run.
    cancel_run: bool = False
    # The STEER ControlMessage (if any) — the executor feeds this to the
    # steerer so planner.refine can build a fresh plan.
    steer_message: ControlMessage | None = None
    # True when the executor should enter its paused wait after the
    # current in-flight task (if any) finishes.
    request_pause: bool = False
    # True when a RESUME arrived — clears any pending pause.
    request_resume: bool = False
    # task_id the REWIND target should reset to, or "" for none.
    rewind_task_id: str = ""
    # Reason string, populated when cancel_run=True.
    cancel_reason: str = ""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _build_ack(
    msg: ControlMessage,
    *,
    result: Any = None,
    detail: str = "",
) -> ControlAck:
    """Construct a :class:`ControlAck` for ``msg``."""
    from goldfive.control import AckResult, ControlAck

    if result is None:
        result = AckResult.SUCCESS
    return ControlAck(
        control_id=msg.id,
        result=result,
        detail=detail,
        acked_at_ms=_now_ms(),
    )


def _kind_value(msg: ControlMessage) -> str:
    """Return the control-kind as an uppercased string for matching."""
    raw = getattr(msg, "kind", "")
    return str(getattr(raw, "value", raw)).upper()


async def emit_status_report(
    *,
    session: Session,
    sinks: list[EventSink],
    control_id: str,
) -> None:
    """Emit a synthetic status report as a ``DriftDetected`` event.

    STATUS_QUERY messages ask the runner "what are you working on right
    now?" Reuses the existing ``DriftDetected`` payload (no proto regen
    for Phase 1 — see issue #71) with kind ``CUSTOM`` and a detail
    string that encodes a compact status snapshot. External observers
    pick the message up off the sink stream and surface it to the UI.
    """
    from goldfive.events import drift_detected_event
    from goldfive.events import emit as emit_event
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity

    plan = session.plan
    total = len(plan.tasks) if plan is not None else 0
    completed_ids: list[str] = []
    pending_ids: list[str] = []
    current = session.current_task_id or ""
    if plan is not None:
        for t in plan.tasks:
            if t.status == TaskStatus.COMPLETED:
                completed_ids.append(t.id)
            elif t.status in (
                TaskStatus.PENDING,
                TaskStatus.RUNNING,
                TaskStatus.BLOCKED,
            ):
                pending_ids.append(t.id)
    detail = (
        f"status_query control_id={control_id} current_task={current} "
        f"completed={len(completed_ids)}/{total} "
        f"pending={','.join(pending_ids) or '-'}"
    )
    drift = DriftEvent(
        kind=DriftKind.CUSTOM,
        severity=DriftSeverity.INFO,
        detail=detail,
        current_task_id=current,
    )
    try:
        evt = drift_detected_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            drift=drift,
        )
    except Exception as exc:  # noqa: BLE001 — proto stubs may be missing
        log.debug("emit_status_report: proto event build failed: %s", exc)
        return
    try:
        await emit_event(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("emit_status_report: sink emit raised: %s", exc)


async def dispatch_control(
    msg: ControlMessage,
    *,
    session: Session,
    steerer: Steerer,
    sinks: list[EventSink],
) -> ControlOutcome:
    """Dispatch a single control message; return the outcome to apply."""
    from goldfive.control import AckResult

    kind = _kind_value(msg)

    if kind == "CANCEL":
        reason = str(msg.payload.get("reason", "")) or "cancelled by control"
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="cancel acknowledged"),
            cancel_run=True,
            cancel_reason=reason,
        )

    if kind == "PAUSE":
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="paused"),
            request_pause=True,
        )

    if kind == "RESUME":
        # Clear the steerer-initiated pause flag (goldfive#142). A
        # RESUME here unwinds both the control-channel's own pause
        # state (via request_resume) AND any Level 4 intervention-
        # ladder pause the steerer set on the session. Always reset
        # even when the flag was never set -- no-op by design.
        session.paused_for_human_intervention = False
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="resumed"),
            request_resume=True,
        )

    if kind == "STEER":
        # A STEER also clears a steerer-initiated pause (goldfive#142):
        # user-supplied corrective intent is itself the resolution the
        # pause was waiting for. The steer message is queued for the
        # executor to feed through ``steerer.observe`` so the planner
        # can produce a USER_STEER-driven plan revision.
        session.paused_for_human_intervention = False
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="steer queued"),
            steer_message=msg,
        )

    if kind == "REWIND_TO":
        target = str(msg.payload.get("task_id", ""))
        if not target:
            return ControlOutcome(
                ack=_build_ack(
                    msg,
                    result=AckResult.FAILURE,
                    detail="rewind_to requires payload.task_id",
                ),
            )
        ok = _rewind_plan(session, target)
        if not ok:
            return ControlOutcome(
                ack=_build_ack(
                    msg,
                    result=AckResult.FAILURE,
                    detail=f"rewind_to: unknown task_id={target!r}",
                ),
            )
        return ControlOutcome(
            ack=_build_ack(
                msg,
                result=AckResult.SUCCESS,
                detail=f"rewound to {target}",
            ),
            rewind_task_id=target,
        )

    if kind == "STATUS_QUERY":
        await emit_status_report(session=session, sinks=sinks, control_id=msg.id)
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="status emitted"),
        )

    if kind == "INTERCEPT_TRANSFER":
        flag = bool(msg.payload.get("enabled", True))
        session._intercept_transfer = flag
        return ControlOutcome(
            ack=_build_ack(
                msg,
                result=AckResult.SUCCESS,
                detail=f"intercept_transfer={'on' if flag else 'off'}",
            ),
        )

    if kind in ("APPROVE", "REJECT"):
        decision = "approve" if kind == "APPROVE" else "reject"
        target_id = str(msg.payload.get("target_id", ""))
        detail = str(msg.payload.get("detail", ""))
        if not target_id:
            return ControlOutcome(
                ack=_build_ack(
                    msg,
                    result=AckResult.FAILURE,
                    detail=f"{kind.lower()} requires payload.target_id",
                ),
            )
        resolved = await _resolve_approval(
            session=session,
            sinks=sinks,
            target_id=target_id,
            decision=decision,
            detail=detail,
        )
        if not resolved:
            return ControlOutcome(
                ack=_build_ack(
                    msg,
                    result=AckResult.FAILURE,
                    detail=(f"no pending approval for target_id={target_id!r}"),
                ),
            )
        return ControlOutcome(
            ack=_build_ack(
                msg,
                result=AckResult.SUCCESS,
                detail=f"{decision} dispatched to {target_id}",
            ),
        )

    # Unknown / unsupported kind.
    return ControlOutcome(
        ack=_build_ack(
            msg,
            result=AckResult.UNSUPPORTED,
            detail=f"unsupported control kind: {kind!r}",
        ),
    )


async def drain_controls(
    channel: ControlChannel | None,
    *,
    session: Session,
    steerer: Steerer,
    sinks: list[EventSink],
) -> list[ControlOutcome]:
    """Drain every message currently queued on ``channel`` and dispatch.

    Non-blocking: returns an empty list when ``channel`` is ``None`` or
    empty. The caller is responsible for applying each outcome; the
    ack has already been published on the channel before return.
    """
    if channel is None:
        return []
    outcomes: list[ControlOutcome] = []
    inbox = getattr(channel, "_inbox", None)
    if inbox is None:
        return outcomes
    while not inbox.empty():
        try:
            msg = inbox.get_nowait()
        except Exception:  # noqa: BLE001 — empty race is benign
            break
        outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=sinks)
        outcomes.append(outcome)
        try:
            await channel.ack(outcome.ack)
        except Exception as exc:  # noqa: BLE001
            log.debug("drain_controls: channel.ack raised: %s", exc)
    return outcomes


async def _resolve_approval(
    *,
    session: Session,
    sinks: list[EventSink],
    target_id: str,
    decision: str,
    detail: str,
) -> bool:
    """Resolve a pending approval waiter and emit the resolution event.

    Returns ``True`` if the ``target_id`` was registered on
    ``session.pending_approvals`` and the waiter was set. Returns
    ``False`` if no waiter exists — the caller turns that into a
    FAILURE ack so UIs know their button click did not land.

    Emits ``ApprovalGranted`` / ``ApprovalRejected`` through the sinks
    before setting the event so the stream ordering is "resolution
    event visible → waiter releases → tool-call returns".
    """
    waiter = session.pending_approvals.get(target_id)
    if waiter is None:
        return False
    meta = session.pending_approvals_meta.setdefault(target_id, {})
    meta["decision"] = decision
    meta["detail"] = detail

    try:
        from goldfive.events import (
            approval_granted_event,
            approval_rejected_event,
            emit,
        )

        if decision == "approve":
            evt = approval_granted_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                target_id=target_id,
                detail=detail,
            )
        else:
            evt = approval_rejected_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                target_id=target_id,
                detail=detail,
            )
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001 — proto/sink failure shouldn't block resolution
        log.debug("_resolve_approval: event emit failed: %s", exc)

    waiter.set()
    return True


def _rewind_plan(session: Session, target_task_id: str) -> bool:
    """Reset ``target`` and every downstream task back to ``PENDING``.

    Any task transitively reachable from ``target`` via ``plan.edges``
    is considered downstream. Returns ``True`` if ``target`` exists
    in the session's plan, else ``False``.
    """
    plan = session.plan
    if plan is None:
        return False
    tasks_by_id = {t.id: t for t in plan.tasks if t.id}
    if target_task_id not in tasks_by_id:
        return False

    children: dict[str, list[str]] = {tid: [] for tid in tasks_by_id}
    for e in plan.edges:
        if e.from_task_id in tasks_by_id and e.to_task_id in tasks_by_id:
            children[e.from_task_id].append(e.to_task_id)

    affected: set[str] = set()
    stack = [target_task_id]
    while stack:
        cur = stack.pop()
        if cur in affected:
            continue
        affected.add(cur)
        for child in children.get(cur, []):
            if child not in affected:
                stack.append(child)

    for tid in affected:
        task = tasks_by_id[tid]
        task.status = TaskStatus.PENDING
        session.completed_results.pop(tid, None)
        session.task_progress.pop(tid, None)
    return True
