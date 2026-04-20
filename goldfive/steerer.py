"""The framework-agnostic :class:`DefaultSteerer`.

Port of harmonograf's ``_AdkState`` task state machine and drift detector,
with all ADK callback plumbing, ``session.state`` writes, and gRPC calls
stripped out. The steerer's job is:

* Mutate :class:`Session`'s plan task statuses in response to reporting
  tool calls (the ``mark_task_*`` family).
* Emit a corresponding proto ``Event`` to every bound ``EventSink`` for
  each transition, drift detection, and plan revision.
* Observe raw upstream events, classify drift, and — if the drift rises
  to ``WARNING`` or above — call ``Planner.refine`` and apply the new
  plan (emitting ``PlanRevised``).

The steerer holds no gRPC / client references and touches no adapter
internals. Executors call :meth:`DefaultSteerer.bind` to wire in the
sinks list and planner at run start.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from goldfive.drift import (
    classify_refusal,
    classify_stop_reason,
    classify_tool_error,
)
from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskStatus,
)

if TYPE_CHECKING:
    from goldfive.protocols import EventSink, Planner

log = logging.getLogger(__name__)


__all__ = ["DefaultSteerer"]


# Task statuses that are terminal (no further transitions allowed).
# Re-exported from :mod:`goldfive.types` — the canonical definition —
# under the historical private name so existing imports in this module
# keep working. Do NOT redefine; new consumers should import
# ``TERMINAL_TASK_STATUSES`` from ``goldfive.types`` directly.
_TERMINAL_TASK_STATUSES = TERMINAL_TASK_STATUSES


# Ordered severities so we can compare with ``>=``.
_SEVERITY_ORDER: dict[DriftSeverity, int] = {
    DriftSeverity.INFO: 0,
    DriftSeverity.WARNING: 1,
    DriftSeverity.CRITICAL: 2,
}


def _severity_ge(a: DriftSeverity, b: DriftSeverity) -> bool:
    return _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b]


# ---------------------------------------------------------------------------
# DefaultSteerer
# ---------------------------------------------------------------------------


class DefaultSteerer:
    """The canonical :class:`Steerer` implementation.

    Bind via :meth:`bind`, then call the ``mark_task_*`` family (usually
    from reporting-tool handlers) to drive task transitions. The executor
    calls :meth:`observe` for each raw upstream event to let the steerer
    classify drift and (optionally) trigger a plan refine.
    """

    def __init__(self) -> None:
        self._sinks: list[Any] = []
        self._planner: Any | None = None
        # Per-session, per-kind last-refine bookkeeping. Purely advisory:
        # callers can subclass to throttle on top of this if needed.
        self._last_refine_kind: dict[tuple[str, DriftKind], int] = {}

    # ------------------------------------------------------------------
    # Protocol-required: wiring
    # ------------------------------------------------------------------

    def bind(self, *, sinks: list[EventSink], planner: Planner) -> None:
        self._sinks = list(sinks)
        self._planner = planner

    # ------------------------------------------------------------------
    # Protocol-required: transition (generic)
    # ------------------------------------------------------------------

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None:
        """Generic transition entry point.

        Dispatches to the corresponding ``mark_task_*`` method. Unknown
        target statuses are a no-op (we don't invent new transitions).
        """
        if to is TaskStatus.RUNNING:
            await self.mark_task_running(task_id, session=session, detail=detail)
        elif to is TaskStatus.COMPLETED:
            await self.mark_task_completed(task_id, summary=detail, session=session)
        elif to is TaskStatus.FAILED:
            await self.mark_task_failed(task_id, reason=detail, session=session)
        elif to is TaskStatus.BLOCKED:
            await self.mark_task_blocked(task_id, blocker=detail, session=session)
        elif to is TaskStatus.CANCELLED:
            await self.mark_task_cancelled(task_id, reason=detail, session=session)
        # PENDING and UNSPECIFIED are intentionally not reachable from
        # here; transitions are always forward in the lifecycle.

    # ------------------------------------------------------------------
    # Task state machine
    # ------------------------------------------------------------------

    async def mark_task_running(
        self,
        task_id: str,
        *,
        session: Session,
        detail: str = "",
    ) -> None:
        """Transition ``task_id`` to ``RUNNING`` and emit ``TaskStarted``."""
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        task.status = TaskStatus.RUNNING
        session.current_task_id = task_id
        if detail:
            session.agent_notes[task_id] = detail
        await self._emit_task_started(session, task_id, detail)

    async def mark_task_progress(
        self,
        task_id: str,
        *,
        session: Session,
        fraction: float = 0.0,
        detail: str = "",
    ) -> None:
        """Record mid-task progress and emit ``TaskProgress``.

        No status transition — a progress update is a liveness ping only.
        ``fraction`` is clamped to ``[0.0, 1.0]``.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        try:
            frac = float(fraction)
        except (TypeError, ValueError):
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        session.task_progress[task_id] = frac
        if detail:
            session.agent_notes[task_id] = detail
        await self._emit_task_progress(session, task_id, frac, detail)

    async def mark_task_completed(
        self,
        task_id: str,
        *,
        session: Session,
        summary: str = "",
        artifacts: dict[str, str] | None = None,
    ) -> None:
        """Transition ``task_id`` to ``COMPLETED`` and emit ``TaskCompleted``."""
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        task.status = TaskStatus.COMPLETED
        if summary:
            session.completed_results[task_id] = summary
        await self._emit_task_completed(session, task_id, summary, artifacts or {})

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        recoverable: bool = True,
    ) -> None:
        """Transition ``task_id`` to ``FAILED`` and emit ``TaskFailed``.

        Also fires a drift event of kind ``TASK_FAILED_RECOVERABLE`` or
        ``TASK_FAILED_FATAL``. The drift event is dispatched through the
        same drift pipeline as observer-detected drift: if severity is
        ``>= WARNING`` (both of these are) we invoke ``planner.refine``.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        task.status = TaskStatus.FAILED
        await self._emit_task_failed(session, task_id, reason, recoverable)
        kind = DriftKind.TASK_FAILED_RECOVERABLE if recoverable else DriftKind.TASK_FAILED_FATAL
        severity = DriftSeverity.WARNING if recoverable else DriftSeverity.CRITICAL
        drift = DriftEvent(
            kind=kind,
            severity=severity,
            detail=f"task {task_id} failed: {reason}" if reason else f"task {task_id} failed",
            current_task_id=task_id,
        )
        await self._handle_drift(drift, session)

    async def mark_task_blocked(
        self,
        task_id: str,
        *,
        session: Session,
        blocker: str = "",
        needed: str = "",
    ) -> None:
        """Transition ``task_id`` to ``BLOCKED`` and emit ``TaskBlocked``.

        Also fires a drift event of kind ``BLOCKED`` which flows through
        the standard drift pipeline (WARNING severity → refine).
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        # BLOCKED is not a terminal status but we still guard against
        # re-blocking a task that's already blocked (idempotent).
        task.status = TaskStatus.BLOCKED
        if blocker or needed:
            session.agent_notes[task_id] = f"blocked: {blocker}" + (
                f" (needed: {needed})" if needed else ""
            )
        await self._emit_task_blocked(session, task_id, blocker, needed)
        detail = f"task {task_id} blocked: {blocker}" + (f" (needed: {needed})" if needed else "")
        drift = DriftEvent(
            kind=DriftKind.BLOCKED,
            severity=DriftSeverity.WARNING,
            detail=detail,
            current_task_id=task_id,
        )
        await self._handle_drift(drift, session)

    async def mark_task_cancelled(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
    ) -> None:
        """Transition ``task_id`` to ``CANCELLED`` and emit ``TaskCancelled``."""
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        task.status = TaskStatus.CANCELLED
        await self._emit_task_cancelled(session, task_id, reason)

    # ------------------------------------------------------------------
    # Observer: drift detection + refine
    # ------------------------------------------------------------------

    async def observe(self, event: Any, session: Session) -> None:
        """Inspect ``event``, classify drift, and refine if severe enough.

        ``ControlMessage`` values are handled first — they carry explicit
        user intent (STEER / CANCEL / PAUSE) and map directly to the
        corresponding ``USER_*`` drift kinds without going through the
        heuristic classifiers. Every other event falls through to
        :meth:`detect_drift`.
        """
        drift = self._drift_from_control(event, session)
        if drift is None:
            drift = self.detect_drift(event, session)
        if drift is None:
            return
        await self._handle_drift(drift, session)

    async def observe_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,  # noqa: ARG002 -- reserved for future detectors
        session: Session,
        provider: str = "",  # noqa: ARG002 -- reserved for per-provider dispatch
    ) -> None:
        """Feed a chain-of-thought / reasoning block into the drift pipeline.

        Appends ``text`` to ``session.reasoning_history`` (bounded by
        ``session.reasoning_history_max``), then runs the reasoning
        detectors. Emits at most one drift per call.

        Adapters call this from their model-response callback once they
        have extracted reasoning_content (OpenAI), thinking blocks
        (Anthropic), or thought parts (Google). Safe to call with empty
        text -- the pipeline no-ops.
        """
        if not text:
            return
        history = session.reasoning_history
        history.append(text)
        cap = getattr(session, "reasoning_history_max", 20) or 20
        overflow = len(history) - cap
        if overflow > 0:
            del history[:overflow]
        from goldfive.drift.reasoning import analyze_reasoning

        drift = analyze_reasoning(text, session)
        if drift is None:
            return
        await self._handle_drift(drift, session)

    @staticmethod
    def _drift_from_control(event: Any, session: Session) -> DriftEvent | None:
        """Map a :class:`ControlMessage` to the matching ``USER_*`` drift.

        Returns ``None`` for anything that is not a ``ControlMessage`` so
        the caller can fall through to the classifier pipeline. Unknown
        control kinds return ``None`` as well — they are dispatched by
        the executor, not the steerer.
        """
        from goldfive.control import ControlKind, ControlMessage

        if not isinstance(event, ControlMessage):
            return None
        raw_kind = getattr(event, "kind", None)
        kind_str = str(getattr(raw_kind, "value", raw_kind) or "").upper()
        payload = event.payload if isinstance(event.payload, dict) else {}
        note = str(payload.get("note", "") or "")
        reason = str(payload.get("reason", "") or "")
        if kind_str == ControlKind.STEER.value:
            return DriftEvent(
                kind=DriftKind.USER_STEER,
                severity=DriftSeverity.WARNING,
                detail=note,
                current_task_id=session.current_task_id,
            )
        if kind_str == ControlKind.CANCEL.value:
            return DriftEvent(
                kind=DriftKind.USER_CANCEL,
                severity=DriftSeverity.CRITICAL,
                detail=reason,
                current_task_id=session.current_task_id,
            )
        if kind_str == ControlKind.PAUSE.value:
            return DriftEvent(
                kind=DriftKind.USER_PAUSE,
                severity=DriftSeverity.INFO,
                detail=note,
                current_task_id=session.current_task_id,
            )
        return None

    def detect_drift(
        self,
        event: Any,
        session: Session,  # noqa: ARG002 — reserved for future heuristics
    ) -> DriftEvent | None:
        """Classify ``event`` via the modular classifiers in :mod:`drift`.

        Classifiers are tried in order of specificity: tool-error shapes
        first (most structured), then refusal markers in text, then
        stop-reason tokens. The first match wins.
        """
        drift = classify_tool_error(event)
        if drift is not None:
            return drift

        # Refusal scan — tolerates raw strings, dicts, objects.
        drift = classify_refusal(event)
        if drift is not None:
            return drift

        # Stop-reason scan — prefer explicit field on dicts / objects.
        stop_reason: Any = None
        if isinstance(event, dict):
            stop_reason = event.get("stop_reason") or event.get("finish_reason")
        else:
            stop_reason = getattr(event, "stop_reason", None) or getattr(
                event, "finish_reason", None
            )
        if stop_reason is not None:
            drift = classify_stop_reason(stop_reason)
            if drift is not None:
                return drift

        return None

    # ------------------------------------------------------------------
    # Plan-mutation drift hooks (invoked by reporting-tool handlers)
    # ------------------------------------------------------------------

    async def report_new_work_discovered(
        self,
        *,
        session: Session,
        parent_task_id: str,
        title: str,
        description: str,
        assignee: str = "",
    ) -> None:
        """Fire a ``NEW_WORK_DISCOVERED`` drift event → triggers refine."""
        detail = f"new work under {parent_task_id}: {title}: {description}" + (
            f" (assignee={assignee})" if assignee else ""
        )
        drift = DriftEvent(
            kind=DriftKind.NEW_WORK_DISCOVERED,
            severity=DriftSeverity.WARNING,
            detail=detail,
            current_task_id=parent_task_id,
            current_agent_id=assignee,
        )
        await self._handle_drift(drift, session)

    async def report_plan_divergence(
        self,
        *,
        session: Session,
        note: str,
        suggested_action: str = "",
    ) -> None:
        """Fire a ``PLAN_DIVERGENCE`` drift event → triggers refine."""
        session.divergence_flag = True
        detail = f"{note} (suggested: {suggested_action})" if suggested_action else note
        drift = DriftEvent(
            kind=DriftKind.PLAN_DIVERGENCE,
            severity=DriftSeverity.WARNING,
            detail=detail,
            current_task_id=session.current_task_id,
        )
        await self._handle_drift(drift, session)

    # ==================================================================
    # Internals
    # ==================================================================

    # --- Plan lookup --------------------------------------------------

    @staticmethod
    def _find_task(session: Session, task_id: str) -> Task | None:
        if not task_id or session.plan is None:
            return None
        for t in session.plan.tasks:
            if t.id == task_id:
                return t
        return None

    # --- Drift dispatch ----------------------------------------------

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:
        """Emit a ``DriftDetected`` event and (if severe enough) refine."""
        await self._emit_drift_detected(session, drift)
        if not _severity_ge(drift.severity, DriftSeverity.WARNING):
            return
        if self._planner is None or session.plan is None:
            return
        try:
            revised = await self._planner.refine(
                plan=session.plan,
                drift=drift,
                goals=list(session.goals),
            )
        except Exception as exc:  # noqa: BLE001 — refine errors must not break the run
            # Surface the failure via logging + a synthetic follow-up
            # drift so operators don't silently see the same plan loop
            # forever. Without this, a refine that raises (e.g. malformed
            # LLM JSON after a mid-invocation cancel poisons the session)
            # leaves session.plan unchanged and the executor re-enters
            # the same state on the next tick.
            log.warning(
                "DefaultSteerer._handle_drift: planner.refine(kind=%s) raised %s; plan unchanged",
                drift.kind.value,
                exc,
            )
            await self._emit_refine_failure(session, drift, reason=str(exc))
            return
        if revised is None:
            log.warning(
                "DefaultSteerer._handle_drift: planner.refine(kind=%s) returned None; "
                "plan unchanged",
                drift.kind.value,
            )
            await self._emit_refine_failure(
                session, drift, reason="planner returned no revised plan"
            )
            return
        self._apply_revision(session, revised, drift)
        await self._emit_plan_revised(session, revised, drift)

    async def _emit_refine_failure(
        self, session: Session, source: DriftEvent, *, reason: str
    ) -> None:
        """Surface a failed refine as a follow-up CRITICAL drift.

        Reuses the source drift's kind and prefixes ``detail`` with
        ``refine failed`` so sinks (and the harmonograf UI) get a durable,
        CRITICAL signal that a prior drift's refine did not succeed —
        without this event, a silently-swallowed refine leaves the
        session pinned to the stale plan and the executor re-enters the
        same state on the next tick.
        """
        failure = DriftEvent(
            kind=source.kind,
            severity=DriftSeverity.CRITICAL,
            detail=f"refine failed ({source.kind.value}): {reason}",
            current_task_id=source.current_task_id,
            current_agent_id=source.current_agent_id,
        )
        await self._emit_drift_detected(session, failure)

    @staticmethod
    def _apply_revision(session: Session, revised: Plan, drift: DriftEvent) -> None:
        """Stamp revision metadata and install ``revised`` on the session.

        Preserves the existing ``revision_index`` monotonicity: the new
        plan's index is at least ``old.revision_index + 1``.
        """
        prev = session.plan
        next_index = (prev.revision_index + 1) if prev is not None else 1
        if revised.revision_index < next_index:
            revised.revision_index = next_index
        if not revised.revision_kind:
            revised.revision_kind = drift.kind.value
        if not revised.revision_severity:
            revised.revision_severity = drift.severity.value
        if not revised.revision_reason:
            revised.revision_reason = drift.detail
        session.plan = revised

    # --- Event construction ------------------------------------------

    def _new_envelope(self, session: Session) -> Any:
        """Build a fresh ``Event`` envelope via :mod:`goldfive.events`."""
        from goldfive.events import new_event

        return new_event(session.run_id, session.next_sequence())

    async def _emit(self, event_pb: Any) -> None:
        from goldfive.events import emit

        await emit(self._sinks, event_pb)

    @staticmethod
    def _drift_kind_pb_value(kind: DriftKind) -> int:
        from goldfive.pb.goldfive.v1 import types_pb2

        name = f"DRIFT_KIND_{kind.name}"
        return getattr(types_pb2, name, getattr(types_pb2, "DRIFT_KIND_CUSTOM", 0))

    @staticmethod
    def _drift_severity_pb_value(severity: DriftSeverity) -> int:
        from goldfive.pb.goldfive.v1 import types_pb2

        name = f"DRIFT_SEVERITY_{severity.name}"
        return getattr(types_pb2, name, getattr(types_pb2, "DRIFT_SEVERITY_UNSPECIFIED", 0))

    # --- Concrete emitters -------------------------------------------

    async def _emit_task_started(self, session: Session, task_id: str, detail: str) -> None:
        evt = self._new_envelope(session)
        evt.task_started.task_id = task_id
        evt.task_started.detail = detail
        await self._emit(evt)

    async def _emit_task_progress(
        self, session: Session, task_id: str, fraction: float, detail: str
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_progress.task_id = task_id
        evt.task_progress.fraction = fraction
        evt.task_progress.detail = detail
        await self._emit(evt)

    async def _emit_task_completed(
        self,
        session: Session,
        task_id: str,
        summary: str,
        artifacts: dict[str, str],
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_completed.task_id = task_id
        evt.task_completed.summary = summary
        for k, v in artifacts.items():
            evt.task_completed.artifacts[k] = v
        await self._emit(evt)

    async def _emit_task_failed(
        self, session: Session, task_id: str, reason: str, recoverable: bool
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_failed.task_id = task_id
        evt.task_failed.reason = reason
        evt.task_failed.recoverable = recoverable
        await self._emit(evt)

    async def _emit_task_blocked(
        self, session: Session, task_id: str, blocker: str, needed: str
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_blocked.task_id = task_id
        evt.task_blocked.blocker = blocker
        evt.task_blocked.needed = needed
        await self._emit(evt)

    async def _emit_task_cancelled(self, session: Session, task_id: str, reason: str) -> None:
        evt = self._new_envelope(session)
        evt.task_cancelled.task_id = task_id
        evt.task_cancelled.reason = reason
        await self._emit(evt)

    async def _emit_drift_detected(self, session: Session, drift: DriftEvent) -> None:
        evt = self._new_envelope(session)
        evt.drift_detected.kind = self._drift_kind_pb_value(drift.kind)
        evt.drift_detected.severity = self._drift_severity_pb_value(drift.severity)
        evt.drift_detected.detail = drift.detail
        evt.drift_detected.current_task_id = drift.current_task_id
        evt.drift_detected.current_agent_id = drift.current_agent_id
        await self._emit(evt)

    async def _emit_plan_revised(self, session: Session, revised: Plan, drift: DriftEvent) -> None:
        from goldfive.conv import to_pb_plan

        evt = self._new_envelope(session)
        evt.plan_revised.plan.CopyFrom(to_pb_plan(revised))
        evt.plan_revised.drift_kind = self._drift_kind_pb_value(drift.kind)
        evt.plan_revised.severity = self._drift_severity_pb_value(drift.severity)
        evt.plan_revised.reason = drift.detail
        evt.plan_revised.revision_index = revised.revision_index
        await self._emit(evt)
