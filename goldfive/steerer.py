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

import json
import logging
import re
from collections.abc import Awaitable, Callable
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

# Shape of the opt-in reflective LLM callable. Matches the signature of
# the ``call_llm`` used by ``LLMPlanner`` (``(system, user, model) ->
# str``) so operators can pass the same callable they already configure
# for planning.
ReflectiveCallLLM = Callable[[str, str, str], Awaitable[str]]

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

    def __init__(
        self,
        *,
        reflective_check_interval: int = 15,
        reflective_call_llm: ReflectiveCallLLM | None = None,
        reflective_model: str = "",
    ) -> None:
        """Build a steerer.

        Parameters
        ----------
        reflective_check_interval:
            Number of LLM invocations (as reported via
            :meth:`note_llm_call`) between reflective self-progress
            checks. Defaults to ``15``. Ignored when
            ``reflective_call_llm`` is ``None``.
        reflective_call_llm:
            Optional async callable ``(system_prompt, user_prompt,
            model) -> str`` used to ask the agent "are you making
            progress?" once the counter reaches the configured
            interval. The whole feature is **off by default** — pass a
            callable to opt in. Operators who don't want the extra LLM
            cost never trigger it. The shape deliberately matches
            :class:`~goldfive.planner.LLMPlanner` so the same callable
            can be reused.
        reflective_model:
            Model name forwarded to ``reflective_call_llm``. Empty
            string is permitted; the callable may substitute its own
            default.

        See ``docs/design/DRIFT.md`` §"Reflective self-progress check"
        for the full feature-gate semantics.
        """
        self._sinks: list[Any] = []
        self._planner: Any | None = None
        # Per-session, per-kind last-refine bookkeeping. Purely advisory:
        # callers can subclass to throttle on top of this if needed.
        self._last_refine_kind: dict[tuple[str, DriftKind], int] = {}
        # Reflective check wiring. When ``_reflective_call_llm`` is None
        # every entry point short-circuits so the feature is inert.
        self._reflective_call_llm: ReflectiveCallLLM | None = reflective_call_llm
        self._reflective_check_interval = max(1, int(reflective_check_interval))
        self._reflective_model = reflective_model

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

        When ``recoverable=False`` the failure is fatal for this task
        lineage: **cascade-cancel every reachable downstream non-terminal
        task** via :meth:`cascade_cancel_downstream`, so the plan lands
        in a consistent terminal-only shape instead of orphaning
        dependents that would sit PENDING forever. This is the
        implementation of PLAN-LIFECYCLE.md §6.2 step 3 and it shares
        its primitive with the §6.3 cancel cascade path
        (:meth:`mark_task_cancelled`). The downstream cascade fires
        *before* we dispatch the fatal drift through ``_handle_drift``
        so that planner.refine sees the post-cascade plan shape and a
        refine-failure back-off does not leave orphans behind. See
        ``STATE-MACHINE.md §"Cascade semantics on unrecoverable drift"``.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        task.status = TaskStatus.FAILED
        await self._emit_task_failed(session, task_id, reason, recoverable)
        # Fatal failures cascade downstream via the same primitive used
        # by mark_task_cancelled, so both §6.2 and §6.3 produce the
        # same TaskCancelled event stream and share rejection guards.
        if not recoverable:
            await self.cascade_cancel_downstream(session, task_id)
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
        """Transition ``task_id`` to ``CANCELLED`` and emit ``TaskCancelled``.

        Also **cascades** the cancellation forward through the plan's
        edges: every non-terminal task reachable from ``task_id`` is
        transitioned to ``CANCELLED`` with a "cascade from <task_id>"
        reason. Without this cascade, downstream PENDING tasks with a
        CANCELLED predecessor would never satisfy the executor's
        "all deps COMPLETED" check — they would sit PENDING forever and
        the executor would report the run as successful while leaving
        them orphaned. See TASK-LIFECYCLE.md §"Cancellation cascade" and
        STATE-MACHINE.md §"Cascade semantics on unrecoverable drift".
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            # Already terminal (including already CANCELLED) — no-op, and
            # crucially do NOT re-run the cascade: we would double-emit
            # TaskCancelled events for downstream tasks on every call.
            return
        task.status = TaskStatus.CANCELLED
        await self._emit_task_cancelled(session, task_id, reason)
        await self.cascade_cancel_downstream(session, task_id)

    async def cascade_cancel_downstream(
        self,
        session: Session,
        cancelled_id: str,
    ) -> None:
        """BFS-cancel every downstream non-terminal task of ``cancelled_id``.

        Shared primitive for both cascade codepaths
        (PLAN-LIFECYCLE.md §6.2 unrecoverable cascade and §6.3
        cancellation cascade):

        - The recoverable path
          (:meth:`mark_task_cancelled`) calls it after transitioning
          the initiator to ``CANCELLED``.
        - The unrecoverable path
          (:meth:`mark_task_failed` with ``recoverable=False``) calls
          it after transitioning the initiator to ``FAILED``.

        Both paths therefore produce the same ``TaskCancelled`` event
        stream for the downstream set and share the rejection guards
        (terminal tasks are skipped; diamond DAGs are de-duplicated).
        The initiator's own transition-emission is caller-controlled —
        this method only emits for the *downstream* set.

        Walks ``session.plan.edges`` forward from ``cancelled_id`` and
        transitions every reachable non-terminal task to ``CANCELLED``
        in-place, emitting one ``TaskCancelled`` event per transition.
        Terminal tasks (COMPLETED / FAILED / CANCELLED) are skipped so a
        diamond DAG does not re-cancel a task through two paths and so
        already-COMPLETED dependents are preserved verbatim.

        Implemented as an iterative BFS on a precomputed adjacency list
        (rather than recursing into :meth:`mark_task_cancelled`) so a
        single top-level cancel produces one summary log line and a
        predictable number of emitted events, independent of graph shape.
        """
        plan = session.plan
        if plan is None:
            return
        # Precompute forward adjacency once.
        downstream: dict[str, list[str]] = {}
        for e in plan.edges:
            downstream.setdefault(e.from_task_id, []).append(e.to_task_id)
        tasks_by_id: dict[str, Task] = {t.id: t for t in plan.tasks if t.id}

        cascade_reason = f"cascade from {cancelled_id}"
        cascaded: list[str] = []
        queue: list[str] = list(downstream.get(cancelled_id, []))
        visited: set[str] = set()
        while queue:
            next_id = queue.pop(0)
            if next_id in visited:
                continue
            visited.add(next_id)
            dep = tasks_by_id.get(next_id)
            if dep is None:
                continue
            if dep.status in _TERMINAL_TASK_STATUSES:
                # Already terminal (COMPLETED/FAILED/CANCELLED) — preserve
                # and do not traverse its children. A COMPLETED task that
                # sits downstream of a late-cancelled ancestor keeps its
                # completion; cascading past it would mean cancelling
                # tasks whose preserved prerequisite is still valid.
                continue
            # Transition in place and emit. We deliberately do NOT
            # recurse through ``mark_task_cancelled`` here; we fan out
            # via our own BFS queue so the surrounding summary log and
            # emission count stay deterministic.
            dep.status = TaskStatus.CANCELLED
            await self._emit_task_cancelled(session, next_id, cascade_reason)
            cascaded.append(next_id)
            for grandchild in downstream.get(next_id, []):
                if grandchild not in visited:
                    queue.append(grandchild)
        if cascaded:
            log.info(
                "DefaultSteerer: cascade-cancelled %d downstream task(s) from %s: %s",
                len(cascaded),
                cancelled_id,
                ", ".join(cascaded),
            )

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

    # ------------------------------------------------------------------
    # Reflective self-progress check (opt-in)
    # ------------------------------------------------------------------

    # Prompt templates. Pulled out as class attributes so subclasses can
    # override the wording without re-implementing the full check.
    REFLECTIVE_SYSTEM_PROMPT: str = (
        "You are assessing your own progress on a task. Answer truthfully. "
        "Reply with a single JSON object and nothing else."
    )

    REFLECTIVE_USER_PROMPT_TEMPLATE: str = (
        "You are assessing your own progress on a task.\n\n"
        "CURRENT TASK:\n"
        "id: {task_id}\n"
        "title: {task_title}\n"
        "description: {task_description}\n\n"
        "WHAT YOU HAVE DONE IN THE LAST {window} LLM TURNS (summarized):\n"
        "- recent tool calls: {tool_call_summary}\n"
        "- recent reasoning (last 3 blocks): {reasoning_summary}\n\n"
        "Q: Are you making forward progress on the task? Reply with a "
        "single JSON object:\n"
        '{{"making_progress": true|false, "confidence": 0.0-1.0, '
        '"reason": "one-sentence explanation"}}'
    )

    async def note_llm_call(self, session: Session) -> None:
        """Record one LLM invocation against ``session``.

        Adapters call this once per LLM turn. Increments
        ``session._llm_calls_since_check``. When the counter reaches the
        configured ``reflective_check_interval`` (and a
        ``reflective_call_llm`` is configured), fires
        :meth:`maybe_run_reflective_check` and resets the counter.

        The counter is also reset (without firing a check) when the
        session's ``current_task_id`` changes — a new task gets a fresh
        window so the check is always scoped to the current task.

        No-ops when ``reflective_call_llm`` was not configured. The
        counter is only updated when the feature is enabled, so
        operators who never opt in pay no memory or call cost.
        """
        if self._reflective_call_llm is None:
            return
        # Reset window on task transitions so the check is always scoped
        # to the current task. Tracks the task id the counter currently
        # belongs to; when it changes (including the first call after a
        # session starts with no current task), we start fresh.
        current = session.current_task_id
        if current != session._reflective_check_task_id:
            session._reflective_check_task_id = current
            session._llm_calls_since_check = 0
        session._llm_calls_since_check += 1
        if session._llm_calls_since_check < self._reflective_check_interval:
            return
        # Reset before running so a check that itself triggers further
        # LLM calls in the agent loop doesn't double-fire.
        session._llm_calls_since_check = 0
        await self.maybe_run_reflective_check(session)

    async def maybe_run_reflective_check(self, session: Session) -> None:
        """Ask the agent "are you making progress?" and emit a drift.

        Opt-in, feature-gated by ``reflective_call_llm``. Does NOT
        advance the counter — callers that want counter-driven
        scheduling go through :meth:`note_llm_call`. This method is
        public so operators can also trigger a one-shot check from
        outside the interval (e.g. on a long-running task boundary).

        Outcomes:

        * ``making_progress=true`` with ``confidence >= 0.5`` → no drift.
        * ``making_progress=true`` with ``confidence < 0.5`` →
          ``UNCERTAIN_PROGRESS`` (INFO severity, observational only).
        * ``making_progress=false`` → ``SELF_REPORTED_STUCK`` (WARNING
          severity; flows through :meth:`_handle_drift` and may trigger
          ``planner.refine``).
        * Reflective LLM raises, returns empty/unparseable JSON, or
          returns JSON missing the expected keys → INFO ``CUSTOM``
          drift noting the reflective check itself failed. The run is
          never broken by a bad reflective call.
        """
        call_llm = self._reflective_call_llm
        if call_llm is None or session.plan is None:
            return
        task = self._find_task(session, session.current_task_id)
        if task is None:
            # No task to assess. Nothing useful to ask the model.
            return
        tool_call_summary = self._summarize_recent_tool_calls(session)
        reasoning_summary = self._summarize_recent_reasoning(session)
        user_prompt = self.REFLECTIVE_USER_PROMPT_TEMPLATE.format(
            task_id=task.id,
            task_title=task.title or "",
            task_description=task.description or "",
            window=self._reflective_check_interval,
            tool_call_summary=tool_call_summary,
            reasoning_summary=reasoning_summary,
        )
        try:
            raw = await call_llm(
                self.REFLECTIVE_SYSTEM_PROMPT,
                user_prompt,
                self._reflective_model,
            )
        except Exception as exc:  # noqa: BLE001 - never break the run
            log.warning("DefaultSteerer.maybe_run_reflective_check: call_llm raised %s", exc)
            await self._emit_reflective_failure(
                session,
                task_id=task.id,
                reason=f"reflective call_llm raised: {exc}",
            )
            return
        parsed = self._parse_reflective_response(raw)
        if parsed is None:
            await self._emit_reflective_failure(
                session,
                task_id=task.id,
                reason=f"reflective response was not valid JSON: {raw!r:.200}",
            )
            return
        making_progress = parsed.get("making_progress")
        confidence = parsed.get("confidence")
        reason = str(parsed.get("reason", "") or "")
        if not isinstance(making_progress, bool):
            await self._emit_reflective_failure(
                session,
                task_id=task.id,
                reason=(f"reflective response missing boolean 'making_progress': {raw!r:.200}"),
            )
            return
        try:
            conf_val = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            conf_val = 0.0
        if not making_progress:
            drift = DriftEvent(
                kind=DriftKind.SELF_REPORTED_STUCK,
                severity=DriftSeverity.WARNING,
                detail=(
                    f"self-reported stuck on task {task.id}"
                    + (f": {reason}" if reason else "")
                    + f" (confidence={conf_val:.2f})"
                ),
                current_task_id=task.id,
                current_agent_id=task.assignee_agent_id,
            )
            await self._handle_drift(drift, session)
            return
        if conf_val < 0.5:
            drift = DriftEvent(
                kind=DriftKind.UNCERTAIN_PROGRESS,
                severity=DriftSeverity.INFO,
                detail=(
                    f"uncertain progress on task {task.id} "
                    f"(confidence={conf_val:.2f})" + (f": {reason}" if reason else "")
                ),
                current_task_id=task.id,
                current_agent_id=task.assignee_agent_id,
            )
            await self._handle_drift(drift, session)
            return
        # making_progress=true, confidence >= 0.5 -- no drift.
        return

    async def _emit_reflective_failure(
        self, session: Session, *, task_id: str, reason: str
    ) -> None:
        """Emit an INFO ``CUSTOM`` drift when the reflective check itself
        could not be interpreted.

        Uses ``CUSTOM`` (rather than a new kind) because this is not a
        property of the agent's behaviour — it's a plumbing failure in
        the reflective check. Sinks that want to surface it specifically
        can look for the ``reflective_check_failed:`` prefix on detail.
        """
        drift = DriftEvent(
            kind=DriftKind.CUSTOM,
            severity=DriftSeverity.INFO,
            detail=f"reflective_check_failed: {reason}",
            current_task_id=task_id,
        )
        # INFO drifts never trigger refine; emit directly.
        await self._emit_drift_detected(session, drift)

    # --- Reflective prompt helpers -----------------------------------

    # Liberal JSON extractor: tolerates markdown code fences and leading /
    # trailing prose around the object.
    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

    @classmethod
    def _parse_reflective_response(cls, raw: Any) -> dict[str, Any] | None:
        """Extract the first JSON object from ``raw`` or return None.

        Tolerates markdown code fences (``\\`\\`\\`json ... \\`\\`\\``) and
        prose wrapping, which real LLMs emit even with strong "reply JSON
        only" instructions. Returns ``None`` for any shape that is not a
        dict once parsed, so downstream code can check one failure mode.
        """
        if not isinstance(raw, str) or not raw.strip():
            return None
        stripped = raw.strip()
        # Fast path: parse verbatim.
        try:
            decoded = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Try extracting the first {...} block.
            match = cls._JSON_OBJECT_RE.search(stripped)
            if match is None:
                return None
            try:
                decoded = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(decoded, dict):
            return None
        return decoded

    @staticmethod
    def _summarize_recent_tool_calls(session: Session, *, limit: int = 10) -> str:
        """Build a short human-readable summary of the last N tool calls.

        Reads from ``session.history`` when the adapter writes tool-call
        observations there; falls back to "(no recent tool calls)" when
        no tool-call-shaped entries are found. Args are trimmed to 120
        chars per call to keep the prompt bounded.

        Adapters that want richer summaries can subclass and override.
        """
        hist = getattr(session, "history", None)
        if not hist:
            return "(no recent tool calls)"
        lines: list[str] = []
        for entry in reversed(list(hist)):
            name = ""
            args: Any = None
            if isinstance(entry, dict):
                if entry.get("kind") == "tool_call":
                    name = str(entry.get("name", "") or "")
                    args = entry.get("args")
            else:
                kind = getattr(entry, "kind", "") or ""
                if kind == "tool_call":
                    name = str(getattr(entry, "name", "") or "")
                    args = getattr(entry, "args", None)
            if not name:
                continue
            args_str = repr(args)[:120] if args is not None else ""
            lines.append(f"{name}({args_str})")
            if len(lines) >= limit:
                break
        if not lines:
            return "(no recent tool calls)"
        # Oldest first for readability.
        return ", ".join(reversed(lines))

    @staticmethod
    def _summarize_recent_reasoning(session: Session, *, limit: int = 3) -> str:
        """Return the last ``limit`` reasoning blocks, truncated.

        Pulls directly from ``session.reasoning_history`` (populated by
        :meth:`observe_reasoning`). Each block is capped at 240 chars so
        the prompt stays bounded for long chains of thought.
        """
        hist = getattr(session, "reasoning_history", None) or []
        if not hist:
            return "(no recent reasoning)"
        tail = list(hist)[-limit:]
        trimmed = [r[:240] + ("…" if len(r) > 240 else "") for r in tail]
        return " | ".join(trimmed)

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
        """Emit a ``DriftDetected`` event and (if severe enough) refine.

        Refine failures (either a raised exception or a ``None`` return)
        are tracked per ``(drift.kind.value, drift.current_task_id)`` on
        ``session.refine_failure_counts``. After
        :attr:`REFINE_FAILURE_THRESHOLD` consecutive failures for the
        same key we skip refine entirely, mark the offending task
        ``FAILED`` (non-recoverable), and emit a CRITICAL
        ``REPEATED_FAILURE`` drift so sinks see the back-off. This
        bounds the loop that would otherwise re-fire the same drift
        every tick until ``SequentialExecutor.max_task_invocations``
        tripped (see TASK-LIFECYCLE.md §7.3).
        """
        await self._emit_drift_detected(session, drift)
        if not _severity_ge(drift.severity, DriftSeverity.WARNING):
            return
        if self._planner is None or session.plan is None:
            return
        counter_key = (drift.kind.value, drift.current_task_id)
        # If this (kind, task) already tripped the threshold on a prior
        # tick, stop trying to refine. The task is FAILED by now so any
        # further drift for it will short-circuit at ``mark_task_failed``.
        if session.refine_failure_counts.get(counter_key, 0) >= self.REFINE_FAILURE_THRESHOLD:
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
            await self._register_refine_failure(session, drift, counter_key)
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
            await self._register_refine_failure(session, drift, counter_key)
            return
        try:
            revised.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            # Reject the revision and surface the failure as a CRITICAL
            # DriftDetected so operators can see the bad plan upstream.
            # The session keeps its old plan. A bad revision also counts
            # as a refine failure for backoff purposes — the planner
            # produced an unusable plan, which is functionally the same
            # as returning None. Passing ``prior=session.plan`` enables
            # PLAN-LIFECYCLE.md §3.1 (terminal task preservation) and
            # §3.2 (terminal->terminal edge preservation) on top of the
            # usual structural checks.
            await self._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.CRITICAL,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                ),
            )
            await self._register_refine_failure(session, drift, counter_key)
            return
        # Successful refine — reset the back-off counter for this key
        # so a future drift of the same kind starts fresh.
        session.refine_failure_counts.pop(counter_key, None)
        # Capture the outgoing plan BEFORE _apply_revision installs the
        # revised one; _emit_plan_revised diffs the two to populate the
        # PlanRevisionDiff sidecar (PLAN-LIFECYCLE.md §2, §8 gap #4).
        prev_plan = session.plan
        self._apply_revision(session, revised, drift)
        await self._emit_plan_revised(session, revised, drift, prev_plan=prev_plan)

    # Consecutive refine failures tolerated per (drift_kind, task_id)
    # before we give up and mark the task FAILED. Class attribute so
    # subclasses / tests can tune it without poking at instance state.
    REFINE_FAILURE_THRESHOLD: int = 2

    async def _register_refine_failure(
        self,
        session: Session,
        drift: DriftEvent,
        counter_key: tuple[str, str],
    ) -> None:
        """Bump the per-(kind, task) refine-failure counter and, if we
        just crossed :attr:`REFINE_FAILURE_THRESHOLD`, mark the task
        ``FAILED`` and emit a ``REPEATED_FAILURE`` drift.

        Below the threshold this is a no-op beyond the increment:
        callers return normally so the next trigger of the same drift
        gets another chance to refine. See TASK-LIFECYCLE.md §7.3.
        """
        count = session.refine_failure_counts.get(counter_key, 0) + 1
        session.refine_failure_counts[counter_key] = count
        if count < self.REFINE_FAILURE_THRESHOLD:
            return
        task_id = drift.current_task_id
        reason = f"refine repeatedly failed for {drift.kind.value}"
        # mark_task_failed routes through _handle_drift for the TASK_FAILED
        # drift; that path keys on a different drift kind so it cannot
        # feed back into *this* counter and recurse.
        if task_id:
            await self.mark_task_failed(
                task_id,
                reason=reason,
                recoverable=False,
                session=session,
            )
        repeated = DriftEvent(
            kind=DriftKind.REPEATED_FAILURE,
            severity=DriftSeverity.CRITICAL,
            detail=(
                f"refine failed {count} consecutive times for "
                f"{drift.kind.value} (task {task_id or 'n/a'})"
            ),
            current_task_id=task_id,
            current_agent_id=drift.current_agent_id,
        )
        # Emit directly — do NOT go back through _handle_drift, which
        # would try to refine again on the fresh drift.
        await self._emit_drift_detected(session, repeated)

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

    async def _emit_plan_revised(
        self,
        session: Session,
        revised: Plan,
        drift: DriftEvent,
        *,
        prev_plan: Plan | None = None,
    ) -> None:
        from goldfive.conv import to_pb_plan
        from goldfive.events import build_plan_revision_diff

        evt = self._new_envelope(session)
        evt.plan_revised.plan.CopyFrom(to_pb_plan(revised))
        evt.plan_revised.drift_kind = self._drift_kind_pb_value(drift.kind)
        evt.plan_revised.severity = self._drift_severity_pb_value(drift.severity)
        evt.plan_revised.reason = drift.detail
        evt.plan_revised.revision_index = revised.revision_index
        # Populate the minimal cross-revision diff so sinks that want a
        # "what changed" view don't have to re-fetch and diff the two
        # plans client-side. prev_plan may be None on the first revision
        # of a run that never received an initial plan — the helper
        # treats that as "everything in revised is newly added".
        evt.plan_revised.diff.CopyFrom(build_plan_revision_diff(prev_plan, revised))
        await self._emit(evt)
