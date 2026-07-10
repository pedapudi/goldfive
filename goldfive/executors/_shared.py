"""Helpers shared verbatim by both executors.

:class:`~goldfive.executors.SequentialExecutor` and
:class:`~goldfive.executors.ParallelDAGExecutor` need the same glue for
steer-driven cancels, cancel bookkeeping, and drift-pipeline failure
reporting. Control-message *dispatch* lives in
:mod:`goldfive.executors._control`; this module hosts the non-dispatch
helpers so the two executors cannot drift apart.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from goldfive.events import emit, new_event
from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Session,
    TaskStatus,
)

if TYPE_CHECKING:
    from goldfive.protocols import EventSink, Steerer

log = logging.getLogger(__name__)


__all__ = [
    "CANCEL_REASON_USER_STEER",
    "apply_steer",
    "emit_drift_event",
    "emit_pipeline_failure_drift",
    "mark_cancelled_if_live",
    "tag_adapter_cancel_user_steer",
]


# Symbolic cancel-reason for USER_STEER. Mirrors
# :data:`goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER` but duplicated
# as a plain string to avoid importing the optional ADK adapter module
# from the provider-agnostic executors. Keep in sync. See goldfive#139.
CANCEL_REASON_USER_STEER: str = "user_steer"


def tag_adapter_cancel_user_steer(adapter: Any, session: Any = None) -> None:
    """Tag the adapter's next mid-invocation cancel with the USER_STEER reason.

    Called just before the executor triggers ``task.cancel()`` on an
    in-flight invoke task so the adapter's mid-invocation cancel
    handler picks up the tag and appends an LLM-actionable synthetic
    ``function_response`` (instead of the legacy generic jargon). See
    goldfive#139.

    Routes through :meth:`ADKAdapter.set_next_cancel_reason` when the
    adapter exposes it (PR #294 audit / goldfive#271 follow-up) so the
    tag is keyed by ``session.id`` and cannot bleed across concurrent
    goldfive sessions sharing one adapter. Falls back to the bare
    ``_next_cancel_reason`` attribute for adapters / stubs that
    predate the helper.
    """
    setter = getattr(adapter, "set_next_cancel_reason", None)
    if callable(setter) and session is not None:
        try:
            setter(session, CANCEL_REASON_USER_STEER)
            return
        except Exception as exc:  # noqa: BLE001
            log.debug("executor: set_next_cancel_reason raised: %s", exc)
    try:
        adapter._next_cancel_reason = CANCEL_REASON_USER_STEER
    except Exception as exc:  # noqa: BLE001
        log.debug("executor: could not tag adapter cancel reason: %s", exc)


async def mark_cancelled_if_live(
    *,
    task_id: str,
    steerer: Steerer,
    session: Session,
    cancel_reason: str = "",
) -> None:
    """Transition a not-yet-terminal task to CANCELLED.

    ``cancel_reason`` (goldfive#205): structured reason stamped on the
    emitted ``TaskCancelled``. Defaults to a generic
    ``user_cancel:cancelled_by_control`` when the caller does not pass
    something more specific (e.g. an annotation_id).
    """
    if session.plan is None:
        return
    for t in session.plan.tasks:
        if t.id == task_id:
            if t.status in TERMINAL_TASK_STATUSES:
                return
            reason_value = cancel_reason or "user_cancel:cancelled_by_control"
            try:
                await steerer.transition(
                    task_id,
                    TaskStatus.CANCELLED,
                    detail="cancelled by control",
                    cancel_reason=reason_value,
                    session=session,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("executor: cancelled transition raised: %s", exc)
            return


async def apply_steer(
    message: object,
    *,
    steerer: Steerer,
    session: Session,
) -> None:
    """Feed a STEER :class:`ControlMessage` to the steerer."""
    try:
        await steerer.drift.observe(message, session)
    except Exception as exc:  # noqa: BLE001
        log.warning("executor: steerer.drift.observe(STEER) raised: %s", exc)


async def emit_drift_event(
    *,
    session: Session,
    sinks: list[EventSink],
    drift: DriftEvent,
) -> None:
    """Build a DriftDetected envelope with proto enum mapping, then emit.

    Uses the same enum-mapping shape
    :meth:`DefaultSteerer._emit_drift_detected` uses — the
    :func:`drift_detected_event` helper in :mod:`goldfive.events` does
    a best-effort name lookup that silently fails for StrEnum-style
    kind/severity names (stored as lowercase like ``critical``), which
    would leave the event with enum value ``0`` (UNSPECIFIED).
    """
    from goldfive.pb.goldfive.v1 import types_pb2

    evt = new_event(session.run_id, session.next_sequence(), session_id=session.id)
    evt.drift_detected.kind = getattr(
        types_pb2,
        f"DRIFT_KIND_{drift.kind.name}",
        getattr(types_pb2, "DRIFT_KIND_CUSTOM", 0),
    )
    evt.drift_detected.severity = getattr(
        types_pb2,
        f"DRIFT_SEVERITY_{drift.severity.name}",
        getattr(types_pb2, "DRIFT_SEVERITY_UNSPECIFIED", 0),
    )
    evt.drift_detected.detail = drift.detail
    evt.drift_detected.current_task_id = drift.current_task_id or ""
    evt.drift_detected.current_agent_id = drift.current_agent_id or ""
    try:
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("emit_drift_event: sink emit raised: %s", exc)


async def emit_pipeline_failure_drift(
    *,
    session: Session,
    sinks: list[EventSink],
    task_id: str,
    reason: str,
) -> None:
    """Emit an INFO ``CUSTOM`` drift when the drift pipeline itself raised.

    Surfaces plumbing failures in ``steerer.drift.observe`` /
    ``detect_drift`` that would otherwise be swallowed. INFO severity
    so this is record-only and does not trigger another refine. Sinks
    that care can filter on the ``drift_pipeline_failed:`` detail
    prefix. See goldfive#134.
    """
    drift = DriftEvent(
        kind=DriftKind.CUSTOM,
        severity=DriftSeverity.INFO,
        detail=reason,
        current_task_id=task_id or "",
    )
    await emit_drift_event(session=session, sinks=sinks, drift=drift)
