"""Default :class:`Steerer` implementation.

The steerer is the state-machine driver for a run. It:

* ``observe`` — absorbs raw events emitted by the adapter/executor,
  forwards progress + drift notifications to the sinks, runs drift
  detection on arbitrary events.
* ``transition`` — flips task status on the session's plan and emits
  the matching ``TaskStarted`` / ``TaskCompleted`` / ``TaskFailed`` /
  ``TaskBlocked`` / ``TaskCancelled`` event through the sinks.
* ``detect_drift`` — default returns ``None``; subclasses can override
  to pattern-match on adapter-specific events.
* ``bind`` — stores the sinks + planner wiring supplied by the runner
  or executor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from goldfive.events import emit, make_event
from goldfive.types import DriftEvent, Session, TaskStatus

if TYPE_CHECKING:
    from goldfive.protocols import EventSink, Planner

log = logging.getLogger("goldfive.steerer")


_STATUS_TO_EVENT: dict[TaskStatus, str] = {
    TaskStatus.RUNNING: "TaskStarted",
    TaskStatus.COMPLETED: "TaskCompleted",
    TaskStatus.FAILED: "TaskFailed",
    TaskStatus.BLOCKED: "TaskBlocked",
    TaskStatus.CANCELLED: "TaskCancelled",
}


class DefaultSteerer:
    """Reference :class:`Steerer` implementation.

    Designed to be used unchanged by most callers. Subclass and override
    :meth:`detect_drift` when the underlying adapter surfaces framework-
    specific drift signals (e.g., ADK ``InvocationContext`` states).
    """

    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Planner | None = None

    def bind(self, *, sinks: list[EventSink], planner: Planner) -> None:
        self._sinks = list(sinks)
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:
        """Observe a raw event — record it, emit progress/drift, detect drift."""
        session.history.append(event)

        if isinstance(event, dict) and event.get("kind") == "task_progress":
            payload = {
                "task_id": str(event.get("task_id") or ""),
                "fraction": float(event.get("fraction") or 0.0),
                "detail": str(event.get("detail") or ""),
            }
            evt = make_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                kind="TaskProgress",
                payload=payload,
            )
            await emit(self._sinks, evt)
            return

        if isinstance(event, dict) and event.get("kind") == "drift":
            drift = event.get("drift")
            if isinstance(drift, DriftEvent):
                await self._emit_drift(drift, session)
                return

        drift = self.detect_drift(event, session)
        if drift is not None:
            await self._emit_drift(drift, session)

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None:
        """Flip task status on the session's plan and emit a status event."""
        if session.plan is not None:
            for t in session.plan.tasks:
                if t.id == task_id:
                    t.status = to
                    break
        if to == TaskStatus.RUNNING:
            session.current_task_id = task_id
        kind = _STATUS_TO_EVENT.get(to)
        if kind is None:
            return
        payload: dict[str, Any] = {"task_id": task_id, "detail": detail, "status": str(to)}
        evt = make_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            kind=kind,
            payload=payload,
        )
        await emit(self._sinks, evt)

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:
        """Default drift detection — returns ``None``. Override to extend."""
        return None

    async def _emit_drift(self, drift: DriftEvent, session: Session) -> None:
        payload = {
            "kind": str(drift.kind),
            "severity": str(drift.severity),
            "detail": drift.detail,
            "current_task_id": drift.current_task_id,
            "current_agent_id": drift.current_agent_id,
        }
        evt = make_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            kind="DriftDetected",
            payload=payload,
        )
        await emit(self._sinks, evt)


__all__ = ["DefaultSteerer"]
