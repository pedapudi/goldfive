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
* ``STATUS_QUERY`` — read-only probe. Returns a compact status
  snapshot string via the ack's ``detail`` field. Does NOT emit any
  sink events: status polling must not register as drift or pollute
  observer streams.
* ``INTERCEPT_TRANSFER`` — toggle ``session._intercept_transfer`` so
  adapters that honour the flag refuse transfers.
* ``APPROVE`` / ``REJECT`` — resolve a pending human-in-the-loop
  approval registered on ``session.pending_approvals`` (either a
  ``report_awaiting_approval`` task-level waiter, Flow A, or an ADK
  ``require_confirmation=True`` tool-level waiter, Flow B). Emits
  ``ApprovalGranted`` / ``ApprovalRejected`` through the sinks.
* ``GOLDFIVE_STEER`` — goldfive-internal kind minted by the steerer
  (Phase 2 of the path-duality fix). Routes goldfive-detected drift
  through the same cancel-and-restart junction as user-authored
  ``STEER``. The executor cancels the in-flight invoke and restarts
  the passthrough with a ``[GOLDFIVE STEERING CONTROL …]`` framed
  corrective. The steerer has already swapped ``session.plan`` before
  dispatching; this control message is purely the cancel-and-restart
  signal + the corrective body.
* ``GOLDFIVE_PAUSE_ESCALATE`` — goldfive-internal kind minted by the
  steerer (Phase 2 of the path-duality fix) replacing the deleted
  ``session.paused_for_human_intervention`` flag. Cancels in-flight
  work and parks the run in the same blocking wait as a user-issued
  ``PAUSE`` — the next ``RESUME`` / ``CANCEL`` / ``STEER`` unwinds it.
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
    "build_status_snapshot",
    "dispatch_control",
    "drain_controls",
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
    # The GOLDFIVE_STEER ControlMessage (if any). Phase 2 of the path-
    # duality fix: goldfive-authored drift dispatches this kind so the
    # executor's invoke loop cancels in-flight work and restarts with a
    # ``[GOLDFIVE STEERING CONTROL …]`` framed corrective. Distinct
    # from ``steer_message`` so the executor keeps the operator-vs-
    # goldfive provenance straight when composing the restart prompt.
    # The plan swap has already happened on the steerer side — this
    # outcome only carries the cancel-and-restart signal + the
    # corrective body to inject.
    goldfive_steer_message: ControlMessage | None = None
    # The GOLDFIVE_PAUSE_ESCALATE ControlMessage (if any). Phase 2 of
    # the path-duality fix: replaces the dead
    # ``session.paused_for_human_intervention`` flag-set with a
    # channel-routed signal. The executor's invoke loop cancels the
    # in-flight invocation, then the pre-task loop blocks waiting for
    # an operator RESUME / CANCEL / STEER (the existing PAUSE channel
    # state).
    goldfive_pause_message: ControlMessage | None = None
    # True when the executor should enter its paused wait after the
    # current in-flight task (if any) finishes.
    request_pause: bool = False
    # True when a RESUME arrived — clears any pending pause.
    request_resume: bool = False
    # task_id the REWIND target should reset to, or "" for none.
    rewind_task_id: str = ""
    # Reason string, populated when cancel_run=True.
    cancel_reason: str = ""
    # goldfive#205: structured cancel reason prefix for harmonograf's
    # Trajectory view. Empty unless the control message was a CANCEL.
    # When populated, looks like ``user_cancel:<annotation_id>`` (or just
    # ``user_cancel:cancelled_by_control`` when the bridge did not carry
    # an annotation id). Threaded through to ``steerer.transition`` so
    # the emitted ``TaskCancelled`` carries the reason.
    cancel_reason_prefix: str = ""


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


def build_status_snapshot(*, session: Session, control_id: str) -> str:
    """Build a compact status-snapshot string for a STATUS_QUERY ack.

    STATUS_QUERY messages ask the runner "what are you working on right
    now?" This is a read-only probe — it MUST NOT emit any drift events
    or pollute the sink stream. The snapshot string is returned via the
    control-channel ack's ``detail`` field so the frontend can poll
    cheaply without generating observer-visible drift markers.

    Previous versions (pre goldfive#XXX) synthesised a ``DriftDetected``
    event here, which meant every status poll produced 2 drift rows in
    the sink chain. A 5-minute e2e run saw 33,666 bogus
    ``drift_detected`` events with ``kind=0`` (UNSPECIFIED) — this
    helper is the fix.
    """
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
    return (
        f"status_query control_id={control_id} current_task={current} "
        f"completed={len(completed_ids)}/{total} "
        f"pending={','.join(pending_ids) or '-'}"
    )


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
        # goldfive#205: compose a structured reason prefix so downstream
        # per-task cancel emits can carry provenance ("user_cancel:<id>").
        # Prefers the bridge-supplied annotation_id when present (the
        # stable id harmonograf uses to dedupe annotation rows against
        # drift rows, goldfive#176); falls back to the control message's
        # own id so every user-cancel row still has a non-empty prefix.
        ann_id = str(msg.payload.get("annotation_id", "") or "")
        cancel_prefix = (
            f"user_cancel:{ann_id}" if ann_id else f"user_cancel:{msg.id}"
        )
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="cancel acknowledged"),
            cancel_run=True,
            cancel_reason=reason,
            cancel_reason_prefix=cancel_prefix,
        )

    if kind == "PAUSE":
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="paused"),
            request_pause=True,
        )

    if kind == "RESUME":
        # RESUME unwinds the control channel's pause state (via
        # request_resume). Phase 2 of the path-duality fix: the
        # in-process ``GOLDFIVE_PAUSE_ESCALATE`` dispatch now drives
        # the same pause state as a user-issued PAUSE, so a RESUME
        # here unblocks both transparently — no separate Session flag
        # to clear.
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="resumed"),
            request_resume=True,
        )

    if kind == "STEER":
        # The steer message is queued for the executor to feed through
        # ``steerer.observe`` so the planner can produce a USER_STEER-
        # driven plan revision. Phase 2 of the path-duality fix: a
        # STEER also acts as a RESUME for any goldfive-initiated pause
        # — the executor honours that by treating ``steer_message`` as
        # an implicit unblock in the pre-task loop.
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail="steer queued"),
            steer_message=msg,
        )

    if kind == "GOLDFIVE_STEER":
        # Phase 2 of the path-duality fix: a goldfive-authored steer.
        # The steerer has already swapped ``session.plan`` and queued
        # the corrective body; the channel message instructs the
        # executor to cancel any in-flight invoke task and restart the
        # passthrough with a ``[GOLDFIVE STEERING CONTROL …]`` framed
        # corrective. The executor branch consumes ``payload["body"]``
        # (and the optional ``superseded_task_ids`` /
        # ``replacement_task_ids`` lists) to compose the restart
        # message.
        return ControlOutcome(
            ack=_build_ack(
                msg, result=AckResult.SUCCESS, detail="goldfive_steer queued"
            ),
            goldfive_steer_message=msg,
        )

    if kind == "GOLDFIVE_PAUSE_ESCALATE":
        # Phase 2 of the path-duality fix: replaces the dead
        # ``session.paused_for_human_intervention`` flag-set. Cancels
        # any in-flight invoke task and parks the run in the same
        # blocking wait as a user-issued PAUSE — the next RESUME /
        # CANCEL / STEER on the channel unwinds it.
        return ControlOutcome(
            ack=_build_ack(
                msg,
                result=AckResult.SUCCESS,
                detail="goldfive_pause_escalate queued",
            ),
            goldfive_pause_message=msg,
            request_pause=True,
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
        transitions = _rewind_plan(session, target)
        if transitions is None:
            return ControlOutcome(
                ack=_build_ack(
                    msg,
                    result=AckResult.FAILURE,
                    detail=f"rewind_to: unknown task_id={target!r}",
                ),
            )
        # F10 / goldfive#251 R4: emit one ``TaskTransitioned`` with
        # ``source="control_rewind"`` per affected task so operator
        # triage can distinguish a control-driven rewind from a
        # cancellation cascade or a plan_revision flip. The steerer's
        # _emit_task_transitioned helper is duck-typed on purpose
        # (custom Steerers may not implement it); failures are
        # swallowed at debug — observability MUST NOT break the
        # control path.
        emit_transition = getattr(steerer, "_emit_task_transitioned", None)
        if callable(emit_transition):
            for task, prev_status in transitions:
                try:
                    await emit_transition(
                        session,
                        task,
                        from_status=prev_status,
                        to_status=TaskStatus.PENDING,
                        source="control_rewind",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "dispatch_control: TaskTransitioned emit raised for "
                        "REWIND_TO task=%s: %s",
                        getattr(task, "id", "?"),
                        exc,
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
        # Read-only probe: return the status snapshot via the ack's
        # `detail` field. Do NOT emit a drift event — status polls must
        # not pollute the sink stream or register as drift markers in
        # the frontend. See the module docstring for the background
        # (goldfive#XXX: 33k bogus drift_detected events from
        # status_query polling in a single 5-min e2e run).
        snapshot = build_status_snapshot(session=session, control_id=msg.id)
        return ControlOutcome(
            ack=_build_ack(msg, result=AckResult.SUCCESS, detail=snapshot),
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
                session_id=session.id,
            )
        else:
            evt = approval_rejected_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                target_id=target_id,
                detail=detail,
                session_id=session.id,
            )
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001 — proto/sink failure shouldn't block resolution
        log.debug("_resolve_approval: event emit failed: %s", exc)

    waiter.set()
    return True


def _rewind_plan(
    session: Session, target_task_id: str
) -> list[tuple[Any, TaskStatus]] | None:
    """Reset ``target`` and every downstream task back to ``PENDING``.

    Any task transitively reachable from ``target`` via ``plan.edges``
    is considered downstream. Returns ``None`` if ``target`` does not
    exist in the session's plan; otherwise returns a list of
    ``(task, previous_status)`` pairs for every task whose status
    actually changed (i.e. was non-PENDING before the rewind). The
    caller uses this list to emit one ``TaskTransitioned`` event per
    affected task with ``source="control_rewind"``.

    F10 / goldfive#251 R4: emitting from the dispatcher (rather than
    here) keeps this helper sync + dependency-free, while still giving
    the typed-observability layer a precise per-task transition row
    that operators can distinguish from ``cancellation`` cascades and
    ``plan_revision`` flips.
    """
    plan = session.plan
    if plan is None:
        return None
    tasks_by_id = {t.id: t for t in plan.tasks if t.id}
    if target_task_id not in tasks_by_id:
        return None

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

    transitions: list[tuple[Any, TaskStatus]] = []
    for tid in affected:
        task = tasks_by_id[tid]
        prev = task.status
        if prev is not TaskStatus.PENDING:
            transitions.append((task, prev))
        task.status = TaskStatus.PENDING
        session.completed_results.pop(tid, None)
        session.task_progress.pop(tid, None)
    return transitions
