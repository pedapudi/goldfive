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

Intervention ladder
-------------------
Drift handling routes through an explicit six-level ladder (goldfive#142)
so "when does goldfive interrupt the tree" is a single table, not a
tangle of conditionals. Levels, ordered by intrusiveness:

* Level 0 — OBSERVE: record the drift, no action.
* Level 1 — ABSORB: call ``planner.refine``; continue.
* Level 2 — NUDGE: queue a soft follow-up message on the session for
  the Runner's overlay loop to pick up at the next invocation boundary.
* Level 3 — CANCEL_REINVOKE: cancel in-flight, refine, and compose a
  corrective user message for the overlay loop to re-invoke with.
* Level 4 — PAUSE_ESCALATE: emit ``HUMAN_INTERVENTION_REQUIRED``, set
  ``session.paused_for_human_intervention``, and wait for user action.
* Level 5 — TERMINATE: run-level abort (currently only reached when an
  unhandled Level 4 times out; actual termination is driven by the
  executor, not the steerer).

The mapping from (drift_kind, severity, occurrence_count) to level
lives in :meth:`DefaultSteerer._ladder_level_for`. Level dispatch is
handled by :meth:`DefaultSteerer._dispatch_ladder_level`, which wraps
the existing refine flow for Level 1 and short-circuits the other
levels. See goldfive#142 for the full table.
"""

from __future__ import annotations

import enum
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from goldfive import orchestration_state as _ostate
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
    Goal,
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


__all__ = [
    "DefaultSteerer",
    "InterventionLevel",
    "compose_corrective_user_message",
]


class InterventionLevel(enum.IntEnum):
    """The graduated intervention ladder (goldfive#142).

    Ordered by intrusiveness. Every drift handled by
    :class:`DefaultSteerer` maps to exactly one level via
    :meth:`DefaultSteerer._ladder_level_for`; the level dictates what
    the steerer does in response.
    """

    OBSERVE = 0
    ABSORB = 1
    NUDGE = 2
    CANCEL_REINVOKE = 3
    PAUSE_ESCALATE = 4
    TERMINATE = 5


# goldfive#202: drift kinds for which a successful ABSORB (Level 1
# refine) also queues a Level 2 nudge onto ``session.pending_nudges``.
# The executor's overlay loop consumes the nudge at invocation end and
# re-invokes the passthrough with a synthesized user message describing
# the plan revision — the only way for a coordinator that is still
# mid-invocation (retrying the superseded task) to learn its plan
# changed. Scoped to "coordinator-stuck" shapes; other ABSORB kinds
# recover at the next task boundary or via Level 3 CANCEL_REINVOKE.
_ABSORB_NUDGE_KINDS: frozenset[DriftKind] = frozenset(
    {
        DriftKind.LOOPING_REASONING,
        DriftKind.LOOPING_TOOL_CALL,
        DriftKind.SELF_REPORTED_STUCK,
    }
)


# Default drift messages per kind, used by
# :func:`compose_corrective_user_message` when the drift carries no
# kind-specific override. Keep SHORT, action-focused; no goldfive jargon
# ("synthetic", "healed", "orphan", "drift") in user-facing copy.
_CORRECTIVE_TEMPLATES: dict[DriftKind, str] = {
    DriftKind.LOOPING_REASONING: (
        "The prior attempt looped on {current_task_id}. "
        "Refined plan: {next_task_title}. Please try a different approach."
    ),
    DriftKind.LOOPING_TOOL_CALL: (
        "The prior attempt kept retrying the same tool call on "
        "{current_task_id} without progress. Refined plan: "
        "{next_task_title}. Please try a different approach."
    ),
    DriftKind.PLAN_DIVERGENCE: (
        "The tree's prior activity diverged from the plan. "
        "Refined plan: proceed with {next_task_title}."
    ),
    DriftKind.AGENT_REFUSAL: (
        "The prior attempt could not complete {current_task_id}. "
        "Refined plan: try {next_task_title}."
    ),
    DriftKind.MODEL_REFUSAL: (
        "The model declined to proceed on {current_task_id}. Refined plan: try {next_task_title}."
    ),
    DriftKind.INTENT_DIVERGENCE: (
        "The prior attempt strayed from the stated intent for "
        "{current_task_id}. Refined plan: proceed with "
        "{next_task_title}."
    ),
    DriftKind.TOOL_ERROR: (
        "The prior attempt hit a tool error on {current_task_id}. "
        "Refined plan: proceed with {next_task_title}."
    ),
    DriftKind.RUNAWAY_DELEGATION: (
        "The prior attempt kept delegating without finishing "
        "{current_task_id}. Refined plan: proceed with "
        "{next_task_title} directly."
    ),
    DriftKind.SELF_REPORTED_STUCK: (
        "The prior attempt reported being stuck on {current_task_id}. "
        "Refined plan: try {next_task_title}."
    ),
    DriftKind.CONFUSION: (
        "The prior attempt showed uncertainty on {current_task_id}. "
        "Refined plan: proceed with {next_task_title}."
    ),
    DriftKind.CONFABULATION_RISK: (
        "The prior attempt may have produced {current_task_id} "
        "without consulting external data. Refined plan: "
        "{next_task_title}."
    ),
}


def compose_corrective_user_message(
    *,
    drift: DriftEvent,
    refined_plan: Plan | None,
    observed_actions: list[Any] | None = None,  # noqa: ARG001
) -> str:
    """Build a short directive user message for Level 3 re-invoke.

    Shape varies by drift kind (see :data:`_CORRECTIVE_TEMPLATES`). The
    message is deliberately short, action-focused, and avoids goldfive
    jargon -- the consumer is the agent's LLM, which should read a
    natural instruction rather than a framework postmortem.

    ``observed_actions`` is accepted for forward-compat with
    goldfive#144 (PLAN_DIVERGENCE refine with observed_actions=...) but
    is NOT interpolated today -- the planner owns action summarization
    and the composer just stitches drift + refined plan. Adding the
    parameter now keeps the signature stable when #144 lands.
    """
    current = drift.current_task_id or "the current task"
    next_title = _next_pending_task_title(refined_plan) or "the next planned step"
    template = _CORRECTIVE_TEMPLATES.get(drift.kind)
    if template is None:
        # Generic fallback for drift kinds that didn't get a
        # custom shape. Keep it tight and action-focused.
        template = (
            "The prior attempt on {current_task_id} did not complete "
            "successfully. Refined plan: proceed with {next_task_title}."
        )
    return template.format(
        current_task_id=current,
        next_task_title=next_title,
    )


def _next_pending_task_title(plan: Plan | None) -> str:
    """Return the title of the next PENDING task in topological order.

    Falls back to the task id if no title is set. Returns an empty
    string when there is no eligible task.
    """
    if plan is None:
        return ""
    for t in plan.tasks:
        if t.status is TaskStatus.PENDING:
            return (t.title or t.id or "").strip()
    return ""


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


def _planner_refine_accepts_available_agents(planner: Any) -> bool:
    """Return True if ``planner.refine`` accepts ``available_agents=``.

    The #151 registry kwarg is additive — the main goldfive planners
    (``LLMPlanner``, ``PassthroughPlanner``, ``StaticPlanner``) all
    accept it, but user-supplied / test-stub planners that predate
    #151 have a refine signature without the kwarg. We probe the
    signature once per drift and fall back to the legacy kwarg-set
    when the kwarg would raise ``TypeError: unexpected keyword
    argument``. Planners declared with ``**kwargs`` are assumed to
    accept (the kwarg passes through).
    """
    import inspect

    refine = getattr(planner, "refine", None)
    if refine is None:
        return False
    try:
        sig = inspect.signature(refine)
    except (TypeError, ValueError):
        # Unintrospectable callable — safest to not pass the kwarg.
        return False
    params = sig.parameters
    if "available_agents" in params:
        return True
    for p in params.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


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
        goal_drift_check_interval: int = 5,
        goal_drift_call_llm: ReflectiveCallLLM | None = None,
        goal_drift_model: str = "",
        goal_drift_activity_window: int = 10,
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
        goal_drift_check_interval:
            Number of agent-invocation turns (as reported via
            :meth:`note_agent_turn`) between trajectory-level
            GOAL_DRIFT checks. Defaults to ``5``. Ignored when
            ``goal_drift_call_llm`` is ``None``.
        goal_drift_call_llm:
            Optional async callable ``(system_prompt, user_prompt,
            model) -> str`` used by
            :meth:`maybe_run_goal_drift_check` to ask an LLM-judge
            whether the tree's recent activity is progressing toward
            ``session.goals``. Opt-in: pass a callable (typically
            ``Runner`` wires its planner LLM here when
            ``goal_drift_enabled`` is on). Shape matches
            ``reflective_call_llm`` so the same callable can be
            reused.
        goal_drift_model:
            Model name forwarded to ``goal_drift_call_llm``.
        goal_drift_activity_window:
            Number of recent-activity entries retained on
            ``session.recent_agent_activity`` and passed to the
            judge. Bounds the prompt size; defaults to ``10``.

        See ``docs/design/DRIFT.md`` §"Reflective self-progress check"
        for the full feature-gate semantics. The GOAL_DRIFT check
        follows the same opt-in contract (see goldfive#143).
        """
        self._sinks: list[Any] = []
        self._planner: Any | None = None
        # Optional adapter back-reference. Wired via :meth:`bind_adapter`
        # so the steerer can tag the next mid-invocation cancel with a
        # symbolic reason (``user_steer``) before the executor triggers
        # ``task.cancel()`` on the invoke task. See goldfive#139 and
        # :attr:`goldfive.adapters.adk.ADKAdapter._next_cancel_reason`.
        self._adapter: Any | None = None
        # Per-session, per-kind last-refine bookkeeping. Purely advisory:
        # callers can subclass to throttle on top of this if needed.
        self._last_refine_kind: dict[tuple[str, DriftKind], int] = {}
        # Scratchpad the steerer uses to plumb the active session into
        # the planner's drift-emitter callback. Set just before calling
        # ``planner.refine`` and cleared afterwards in ``_handle_drift``;
        # ``None`` outside that window. Only consulted by the emitter
        # the planner calls when its retry budget is spent
        # (goldfive#133).
        self._active_session: Session | None = None
        # Reflective check wiring. When ``_reflective_call_llm`` is None
        # every entry point short-circuits so the feature is inert.
        self._reflective_call_llm: ReflectiveCallLLM | None = reflective_call_llm
        self._reflective_check_interval = max(1, int(reflective_check_interval))
        self._reflective_model = reflective_model
        # GOAL_DRIFT (goldfive#143) wiring. Same opt-in contract as the
        # reflective check: feature is inert unless a callable is passed.
        self._goal_drift_call_llm: ReflectiveCallLLM | None = goal_drift_call_llm
        self._goal_drift_check_interval = max(1, int(goal_drift_check_interval))
        self._goal_drift_model = goal_drift_model
        self._goal_drift_activity_window = max(1, int(goal_drift_activity_window))

    # ------------------------------------------------------------------
    # Protocol-required: wiring
    # ------------------------------------------------------------------

    def bind(self, *, sinks: list[EventSink], planner: Planner) -> None:
        self._sinks = list(sinks)
        self._planner = planner
        # If the planner supports the optional drift-emitter hook (see
        # :class:`~goldfive.planner.LLMPlanner.set_drift_emitter`), wire
        # it up now so the planner can signal ``REFINE_VALIDATION_FAILED``
        # through the normal event pipeline when its retry budget is
        # spent. Duck-typed on purpose so custom ``Planner``
        # implementations don't have to implement the hook.
        setter = getattr(planner, "set_drift_emitter", None)
        if callable(setter):
            setter(self._emit_planner_refine_validation_failed)

    def bind_adapter(self, adapter: Any) -> None:
        """Attach the active adapter for cancel-reason tagging.

        Optional wiring (goldfive#139). When set, the steerer tags the
        adapter's next mid-invocation cancel with a symbolic reason
        (``user_steer``) so the synthetic ``function_response`` event
        the adapter appends on cancel carries LLM-actionable content
        instead of the legacy goldfive-internal jargon. Adapters that
        don't expose ``_next_cancel_reason`` are tolerated: the setter
        is a no-op and the cancel falls through to the generic content
        variant.
        """
        self._adapter = adapter

    async def _emit_planner_refine_validation_failed(self, drift: DriftEvent) -> None:
        """Emit a planner-side drift through the DriftDetected pipeline.

        Dispatched as the drift emitter passed to
        :meth:`~goldfive.planner.LLMPlanner.set_drift_emitter` in
        :meth:`bind`. The planner calls this when it exhausts its
        refine retry budget with a ``REFINE_VALIDATION_FAILED``
        ``DriftEvent`` pre-built; we emit it but deliberately do NOT go
        back through :meth:`_handle_drift` (the steerer must not try to
        refine again on this kind -- infinite-loop risk).
        """
        # The planner's emitter isn't bound to a specific session, so
        # we route through the most recently-active session -- which in
        # practice is the only session a single-threaded planner is
        # handling. Store it on the instance when _handle_drift runs.
        session = self._active_session
        if session is None:
            log.warning(
                "DefaultSteerer: planner emitted %s but no active session bound; dropping signal",
                drift.kind.value,
            )
            return
        await self._emit_drift_detected(session, drift)

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
        cancel_reason: str = "",
    ) -> None:
        """Generic transition entry point.

        Dispatches to the corresponding ``mark_task_*`` method. Unknown
        target statuses are a no-op (we don't invent new transitions).

        ``cancel_reason`` (goldfive#205): structured reason string stamped
        on the emitted ``TaskCancelled`` / ``TaskFailed`` envelope when the
        transition is to CANCELLED or FAILED. Takes precedence over
        ``detail`` for reason-field population. Conventional formats
        recognised by harmonograf's Trajectory view:

        * ``upstream_failed:<upstream_task_id>`` — cascade from a failed
          or cancelled ancestor.
        * ``superseded_by_revision:<replacement_task_id>`` — refine
          replaced this task with a new one.
        * ``run_aborted:<abort_reason>`` — fail_fast / validation / budget.
        * ``user_cancel:<annotation_id>`` — user-initiated CANCEL control.
        * ``adk_cancellation:<invocation_id>`` — ADK mid-invocation cancel.
        * ``steerer_policy:<drift_kind>`` — steerer-imposed cancel via
          intervention ladder.

        Sinks that don't know the format still surface the raw string,
        so it's safe to evolve the vocabulary without a proto change.

        Passing ``cancel_reason`` for non-terminal transitions is a
        no-op; the value is only consulted when ``to`` is CANCELLED or
        FAILED.
        """
        if to is TaskStatus.RUNNING:
            await self.mark_task_running(task_id, session=session, detail=detail)
        elif to is TaskStatus.COMPLETED:
            await self.mark_task_completed(task_id, summary=detail, session=session)
        elif to is TaskStatus.FAILED:
            reason = cancel_reason or detail
            await self.mark_task_failed(task_id, reason=reason, session=session)
        elif to is TaskStatus.BLOCKED:
            await self.mark_task_blocked(task_id, blocker=detail, session=session)
        elif to is TaskStatus.CANCELLED:
            reason = cancel_reason or detail
            await self.mark_task_cancelled(task_id, reason=reason, session=session)
        elif to is TaskStatus.NOT_NEEDED:
            await self.mark_task_not_needed(task_id, reason=detail, session=session)
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
        # goldfive#152: stamp current_task_* on the orchestration-state
        # dict so downstream prompt templates / refine paths see it.
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.RUNNING)
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
        # goldfive#152: clear current_task_* if we were the active task.
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.COMPLETED)
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
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.FAILED)
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
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.CANCELLED)
        await self._emit_task_cancelled(session, task_id, reason)
        await self.cascade_cancel_downstream(session, task_id)

    async def mark_task_not_needed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
    ) -> None:
        """Transition ``task_id`` to ``NOT_NEEDED`` terminally.

        Introduced by the overlay-model refactor (goldfive#141). Unlike
        :meth:`mark_task_cancelled` this path does NOT cascade — a task
        the :class:`~goldfive.reconciler.PlanReconciler` deemed "not
        needed" post-invocation is an observation about that specific
        plan entry, not a signal that downstream work is invalid. The
        reconciler independently evaluates each PENDING task.

        Idempotent on terminal tasks. Emits ``TaskCancelled`` at the
        proto level (there's no dedicated NOT_NEEDED event — the
        status lives on the task itself and sinks can distinguish via
        ``task.status`` on the next ``TaskCancelled`` / ``PlanRevised``
        envelope). The reason string carries the distinguishing
        context ("not needed: superseded by ...").
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        task.status = TaskStatus.NOT_NEEDED
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.NOT_NEEDED)
        # There is no dedicated ``TaskNotNeeded`` proto message;
        # reuse TaskCancelled with the reason prefix so sinks that
        # inspect reason can differentiate if they wish. The live
        # ``task.status`` on the plan is the authoritative signal.
        await self._emit_task_cancelled(
            session, task_id, f"not_needed: {reason}" if reason else "not_needed"
        )

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

        # goldfive#205: structured reason consumed by harmonograf's
        # Trajectory view. Old ``cascade from <id>`` form preserved in a
        # human-readable tail after the colon so sinks that render the
        # raw reason keep their existing copy; new sinks parse the
        # ``upstream_failed:`` prefix to categorise the cancel.
        cascade_reason = f"upstream_failed:{cancelled_id}"
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

        STEER ControlMessages are deduped by their source annotation id
        (goldfive#171): a delivery retry or UI double-fire of the same
        STEER lands here twice, but cascade-cancel + refine must only
        happen once. The dedupe set lives on ``session.state`` under
        :data:`orchestration_state.KEY_PROCESSED_STEER_IDS` with FIFO
        eviction after :data:`PROCESSED_STEER_IDS_CAP` entries. Content-
        based drifts (LOOPING_REASONING, tool errors, …) are NOT
        deduped — they're heuristic signals, not user actions.
        """
        if self._is_duplicate_steer(event, session):
            steer_id = self._steer_dedupe_id(event)
            log.debug("DefaultSteerer.observe: dropping duplicate STEER id=%s", steer_id)
            return
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

    # ------------------------------------------------------------------
    # GOAL_DRIFT — trajectory-level periodic check (opt-in, goldfive#143)
    # ------------------------------------------------------------------

    def note_agent_activity(
        self,
        session: Session,
        *,
        kind: str,
        agent_name: str = "",
        task_id: str = "",
        detail: str = "",
    ) -> None:
        """Record a recent agent-activity entry on ``session``.

        Push-only: adapters (or executors) call this once per
        ``AgentInvocationStarted`` / ``AgentInvocationCompleted`` so the
        GOAL_DRIFT judge has a rolling view of the trajectory. The ring
        buffer is trimmed to ``goal_drift_activity_window`` so the
        prompt stays bounded regardless of run length.

        Always safe to call (feature-gate is enforced at check time, not
        at record time) -- unlike :meth:`note_agent_turn`, this method
        does not short-circuit when ``goal_drift_call_llm`` is
        unconfigured so that sinks / tests can observe the recorded
        activity independently.
        """
        if not kind:
            return
        entry: dict[str, Any] = {"kind": kind}
        if agent_name:
            entry["agent_name"] = agent_name
        if task_id:
            entry["task_id"] = task_id
        if detail:
            # Keep individual entries bounded so a pathological detail
            # cannot blow up the prompt even before trimming.
            entry["detail"] = detail[:500]
        hist = session.recent_agent_activity
        hist.append(entry)
        overflow = len(hist) - self._goal_drift_activity_window
        if overflow > 0:
            del hist[:overflow]

    async def note_agent_turn(self, session: Session) -> None:
        """Record one agent invocation against ``session``.

        Adapters call this once per completed agent invocation
        (``after_run_callback`` on ADK, or the equivalent hook on other
        frameworks). Increments
        ``session._agent_turns_since_goal_check``; when the counter
        reaches ``goal_drift_check_interval`` (and a
        ``goal_drift_call_llm`` is configured), fires
        :meth:`maybe_run_goal_drift_check` and resets the counter.

        No-ops when ``goal_drift_call_llm`` was not configured, so
        operators who never opt in pay no memory or LLM cost. Unlike
        :meth:`note_llm_call`, the counter is trajectory-level and is
        NOT reset on task transitions -- GOAL_DRIFT is about the whole
        tree's direction, not one task's progress.
        """
        if self._goal_drift_call_llm is None:
            return
        session._agent_turns_since_goal_check += 1
        if session._agent_turns_since_goal_check < self._goal_drift_check_interval:
            return
        # Reset before running so a check that itself triggers further
        # invocations in the agent loop doesn't double-fire.
        session._agent_turns_since_goal_check = 0
        await self.maybe_run_goal_drift_check(session)

    async def maybe_run_goal_drift_check(self, session: Session) -> None:
        """Run the trajectory-level GOAL_DRIFT judge once, cost-bounded.

        Opt-in, feature-gated by ``goal_drift_call_llm``. Does NOT
        advance the counter -- callers that want counter-driven
        scheduling go through :meth:`note_agent_turn`. Public so
        operators can trigger a one-shot check from outside the
        interval (e.g. on a long idle period with no task transitions).

        Outcomes:

        * Judge returns ``{"progressing": true}`` → no drift emitted.
        * Judge returns ``{"progressing": false, "reason": "..."}`` →
          ``GOAL_DRIFT`` drift at CRITICAL severity; flows through
          :meth:`_handle_drift` so the #142 ladder (once merged) can
          route it to Level 4.
        * Judge raises, returns malformed JSON, or returns a dict
          missing / with a non-boolean ``progressing`` field → no
          drift emitted. False positives on plumbing failures would
          spam operators; see goldfive#143 rationale.
        """
        call_llm = self._goal_drift_call_llm
        if call_llm is None:
            return
        from goldfive.drift.goals import classify_goal_drift

        # Snapshot activity so subsequent appends during the await do
        # not perturb the prompt the judge saw.
        activity = list(session.recent_agent_activity)
        drift = await classify_goal_drift(
            goals=session.goals,
            plan=session.plan,
            observed_actions=activity,
            model=self._goal_drift_model,
            call_llm=call_llm,
            current_task_id=session.current_task_id,
        )
        if drift is None:
            return
        await self._handle_drift(drift, session)

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
    def _steer_dedupe_id(event: Any) -> str:
        """Return the dedupe id for a STEER ``ControlMessage``, or ``""``.

        Prefers the source ``annotation_id`` when the bridge forwarded
        one (goldfive#171), falling back to the ``ControlMessage.id``
        so callers that don't source annotations still get retry dedupe.
        Returns ``""`` for non-ControlMessages, non-STEER kinds, or
        ids the bridge didn't populate — callers treat an empty id as
        "nothing to dedupe".
        """
        from goldfive.control import ControlKind, ControlMessage

        if not isinstance(event, ControlMessage):
            return ""
        raw_kind = getattr(event, "kind", None)
        kind_str = str(getattr(raw_kind, "value", raw_kind) or "").upper()
        if kind_str != ControlKind.STEER.value:
            return ""
        payload = event.payload if isinstance(event.payload, dict) else {}
        ann_id = str(payload.get("annotation_id", "") or "")
        if ann_id:
            return ann_id
        return str(getattr(event, "id", "") or "")

    @staticmethod
    def _unpack_steer_context(drift: DriftEvent) -> tuple[str, str, str]:
        """Extract ``(raw_body, author, dedupe_id)`` from a USER_STEER drift.

        Prefers the originating :class:`ControlMessage` stashed on
        :attr:`DriftEvent.raw` so the raw body survives the ``"by
        {author}: {body}"`` rewrite applied to :attr:`DriftEvent.detail`.
        When ``raw`` is absent (e.g. a test that builds a USER_STEER
        drift directly), falls back to parsing the detail string — a
        ``"by X: Y"`` prefix is treated as ``(Y, X, "")``; anything
        else becomes ``(detail, "", "")``.
        """
        from goldfive.control import ControlMessage

        raw = getattr(drift, "raw", None)
        if isinstance(raw, ControlMessage):
            payload = raw.payload if isinstance(raw.payload, dict) else {}
            body = str(payload.get("note", "") or "")
            author = str(payload.get("author", "") or "").strip()
            ann_id = str(payload.get("annotation_id", "") or "")
            dedupe_id = ann_id or str(getattr(raw, "id", "") or "")
            return body, author, dedupe_id
        # Fallback: parse "by {author}: {body}" out of detail so the
        # back-compat DriftEvent-only code path preserves the author in
        # state writes. No dedupe id is recoverable here.
        detail = str(getattr(drift, "detail", "") or "")
        if detail.startswith("by ") and ": " in detail:
            prefix, _, tail = detail.partition(": ")
            author = prefix[len("by ") :].strip()
            return tail, author, ""
        return detail, "", ""

    @classmethod
    def _is_duplicate_steer(cls, event: Any, session: Session) -> bool:
        """True when ``event`` is a STEER ControlMessage already processed.

        See :meth:`_steer_dedupe_id` for the id-selection rules. An
        empty id always returns ``False`` (nothing to compare against).
        """
        steer_id = cls._steer_dedupe_id(event)
        if not steer_id:
            return False
        return _ostate.has_processed_steer_id(session.state, steer_id)

    @staticmethod
    def _drift_from_control(event: Any, session: Session) -> DriftEvent | None:
        """Map a :class:`ControlMessage` to the matching ``USER_*`` drift.

        Returns ``None`` for anything that is not a ``ControlMessage`` so
        the caller can fall through to the classifier pipeline. Unknown
        control kinds return ``None`` as well — they are dispatched by
        the executor, not the steerer.

        For STEER, the operator ``author`` (when the bridge forwarded
        one) is prefixed onto the drift detail so downstream consumers
        — prompt templates, sinks, UI — see audit-trail attribution
        inline without having to peek into ``session.state``
        (goldfive#171). The raw body still lands on
        ``goldfive.active_steer.body`` untouched.
        """
        from goldfive.control import ControlKind, ControlMessage

        if not isinstance(event, ControlMessage):
            return None
        raw_kind = getattr(event, "kind", None)
        kind_str = str(getattr(raw_kind, "value", raw_kind) or "").upper()
        payload = event.payload if isinstance(event.payload, dict) else {}
        note = str(payload.get("note", "") or "")
        reason = str(payload.get("reason", "") or "")
        author = str(payload.get("author", "") or "").strip()
        if kind_str == ControlKind.STEER.value:
            if author:
                detail = f"by {author}: {note}"
            else:
                detail = note
            return DriftEvent(
                kind=DriftKind.USER_STEER,
                severity=DriftSeverity.WARNING,
                detail=detail,
                current_task_id=session.current_task_id,
                raw=event,
            )
        if kind_str == ControlKind.CANCEL.value:
            return DriftEvent(
                kind=DriftKind.USER_CANCEL,
                severity=DriftSeverity.CRITICAL,
                detail=reason,
                current_task_id=session.current_task_id,
                raw=event,
            )
        if kind_str == ControlKind.PAUSE.value:
            return DriftEvent(
                kind=DriftKind.USER_PAUSE,
                severity=DriftSeverity.INFO,
                detail=note,
                current_task_id=session.current_task_id,
                raw=event,
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
        """Emit a ``DriftDetected`` event and dispatch via the intervention ladder.

        The ladder (goldfive#142) maps (drift_kind, severity,
        occurrence_count) to one of six :class:`InterventionLevel`
        values and dispatches accordingly. Level 1 (ABSORB) preserves
        the historical refine-on-WARNING behaviour; other levels
        short-circuit, queue follow-ups, or escalate to a paused state
        for user intervention.

        Refine failures (either a raised exception or a ``None`` return)
        from Level 1 dispatch are tracked per
        ``(drift.kind.value, drift.current_task_id)`` on
        ``session.refine_failure_counts``. After
        :attr:`REFINE_FAILURE_THRESHOLD` consecutive failures for the
        same key we skip refine entirely, mark the offending task
        ``FAILED`` (non-recoverable), and emit a CRITICAL
        ``REPEATED_FAILURE`` drift so sinks see the back-off. This
        bounds the loop that would otherwise re-fire the same drift
        every tick until ``SequentialExecutor.max_task_invocations``
        tripped (see TASK-LIFECYCLE.md §7.3).
        """
        # Tag the bound adapter's next cancel with a symbolic reason so
        # the synthetic function_response the adapter appends on cancel
        # carries LLM-actionable content. Done BEFORE the drift event
        # is emitted so a sink that reacts by cancelling the invoke
        # sees the tag. Harmless if the adapter doesn't expose the
        # attribute (duck-typed) or no adapter is bound. See
        # goldfive#139 and
        # :func:`goldfive.adapters.adk._build_cancelled_response_event`.
        self._tag_adapter_cancel_reason(drift)
        # goldfive#152: USER_STEER-specific side effects -- write the
        # active-steer bookkeeping onto the orchestration-state dict
        # and synthesize a durable Goal from the steer body so
        # subsequent refines see the pivot as a first-class goal,
        # not a one-shot user message. Done BEFORE the drift event
        # is emitted (the state writes are cheap) and BEFORE the
        # ladder dispatches to planner.refine (which reads
        # ``session.goals`` we just mutated) so the refine sees the
        # new goal shape in the same dispatch.
        if drift.kind is DriftKind.USER_STEER:
            await self._apply_user_steer_state(drift, session)
        await self._emit_drift_detected(session, drift)
        # Route through the intervention ladder. The per-(kind, task)
        # occurrence count drives the "first vs repeat" distinction in
        # the ladder table -- we read it BEFORE any mutation so the
        # mapping sees the state at drift-fire time.
        counter_key = (drift.kind.value, drift.current_task_id)
        occurrence_count = session.refine_failure_counts.get(counter_key, 0)
        level = self._ladder_level_for(drift.kind, drift.severity, occurrence_count)
        log.debug(
            "DefaultSteerer._handle_drift: kind=%s severity=%s occurrence=%d -> level=%s",
            drift.kind.value,
            drift.severity.value,
            occurrence_count,
            level.name,
        )
        if level is InterventionLevel.OBSERVE:
            return
        if level is InterventionLevel.NUDGE:
            await self._dispatch_nudge(drift, session)
            return
        if level is InterventionLevel.PAUSE_ESCALATE:
            await self._dispatch_pause_escalate(drift, session)
            return
        if level is InterventionLevel.TERMINATE:
            # Level 5 is reserved for a future Runner-side timeout on a
            # stuck Level 4 pause. Today we fall back to PAUSE_ESCALATE
            # so no code path silently drops the drift.
            await self._dispatch_pause_escalate(drift, session)
            return
        # ABSORB and CANCEL_REINVOKE both call ``planner.refine`` and
        # install the revised plan. CANCEL_REINVOKE additionally queues
        # a corrective message on the session for the overlay loop
        # (goldfive#141). The refine call itself is identical so we
        # share the implementation below and read the level at the end
        # to decide whether to emit the follow-up handoff.
        if self._planner is None or session.plan is None:
            return
        if drift.kind is DriftKind.REFINE_VALIDATION_FAILED:
            # Terminal planner signal (goldfive#133). Do NOT call refine
            # again on it. The ladder already routes this to Level 4 so
            # control flow normally won't reach here, but belt-and-braces.
            return
        if session.refine_failure_counts.get(counter_key, 0) >= self.REFINE_FAILURE_THRESHOLD:
            return
        # Plumb the session into the planner's drift-emitter callback
        # for the duration of this refine call so the planner can emit
        # REFINE_VALIDATION_FAILED drifts through the normal event
        # pipeline. Cleared in a ``finally`` so exceptions don't leave
        # a stale session pointer. See goldfive#133.
        self._active_session = session
        try:
            # Thread the adapter's available_agents_tree (goldfive#151)
            # through refine so the LLM is constrained to pick real
            # tree assignees. Adapters without the property fall back
            # to ``available_agents`` (list[str]); custom/legacy adapters
            # without either surface produce ``None`` and the planner
            # keeps its pre-#151 behaviour. Planners whose refine does
            # not accept the kwarg (test stubs, pre-#151 custom
            # planners) are called the old way so nothing breaks.
            available_agents: Any = None
            adapter = self._adapter
            if adapter is not None:
                tree = getattr(adapter, "available_agents_tree", None)
                if isinstance(tree, list) and tree:
                    available_agents = list(tree)
                else:
                    flat = getattr(adapter, "available_agents", None)
                    if flat:
                        available_agents = list(flat)
            refine_accepts_registry = _planner_refine_accepts_available_agents(self._planner)
            try:
                if refine_accepts_registry:
                    revised = await self._planner.refine(
                        plan=session.plan,
                        drift=drift,
                        goals=list(session.goals),
                        available_agents=available_agents,
                    )
                else:
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
                    "DefaultSteerer._handle_drift: planner.refine(kind=%s) raised "
                    "%s; plan unchanged",
                    drift.kind.value,
                    exc,
                )
                await self._emit_refine_failure(session, drift, reason=str(exc))
                await self._register_refine_failure(session, drift, counter_key)
                return
        finally:
            self._active_session = None
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
        # Level 3 (CANCEL_REINVOKE) handoff: compose a corrective user
        # message from the drift + refined plan and stash it on the
        # session so the Runner's overlay loop (goldfive#141) can cancel
        # the in-flight invocation and re-invoke with the composed text.
        # Until #141 lands, this slot is inert -- nobody reads it -- but
        # the data is durably attached to the session and a later-landing
        # overlay will pick it up automatically.
        if level is InterventionLevel.CANCEL_REINVOKE:
            session.pending_corrective_message = compose_corrective_user_message(
                drift=drift,
                refined_plan=session.plan,
            )
        # goldfive#202: for drifts where the coordinator has no way to
        # observe the plan revision on its own (it is still mid-
        # invocation, retrying the superseded task), ALSO queue a
        # Level 2 nudge after a successful ABSORB. The overlay loop's
        # scoped nudge-replay path (see SequentialExecutor._run_overlay)
        # picks this up at invocation end and re-invokes the
        # passthrough with the nudge as the next user message — the
        # only way for the coordinator to learn its plan changed.
        #
        # Scoped to drift kinds whose mid-invocation signature is
        # "coordinator is stuck on a task goldfive just replaced":
        # LOOPING_REASONING / LOOPING_TOOL_CALL (detector fires while
        # the coordinator retries the same tool call), SELF_REPORTED_STUCK
        # (reflective self-check reports no progress). Other ABSORB
        # kinds (CONFUSION, CONFABULATION_RISK, etc.) do not need
        # mid-invocation rescue — their corrective path fires at the
        # next task boundary or via Level 3 CANCEL_REINVOKE.
        if level is InterventionLevel.ABSORB and drift.kind in _ABSORB_NUDGE_KINDS:
            nudge_msg = compose_corrective_user_message(
                drift=drift,
                refined_plan=session.plan,
            )
            session.pending_nudges.append(nudge_msg)
            log.debug(
                "DefaultSteerer._handle_drift: queued post-ABSORB nudge for kind=%s task=%s: %s",
                drift.kind.value,
                drift.current_task_id or "-",
                nudge_msg,
            )

    # --- Intervention ladder -----------------------------------------
    #
    # Mapping table for :meth:`_ladder_level_for`. Keys are
    # :class:`DriftKind` values. Each value is a 3-tuple
    # ``(info_level, warning_level, critical_level)`` holding the level
    # for that severity tier. CRITICAL uses a (first, repeat) pair --
    # repeat applies once the refine-failure counter has crossed
    # :attr:`REFINE_FAILURE_THRESHOLD` for the (kind, task) pair. A
    # level of ``None`` means "not applicable at this severity -- drop
    # through to OBSERVE". See goldfive#142 for the rationale.
    #
    # Drifts without an entry here fall through to a conservative
    # default that preserves the pre-ladder behaviour: WARNING -> ABSORB,
    # CRITICAL -> ABSORB first, CANCEL_REINVOKE on repeat.
    #
    # A note on INFO-tier preservation: the pre-ladder steerer
    # short-circuited every INFO-severity drift at "no refine" (the
    # early return below ``_severity_ge(drift.severity, WARNING)``).
    # The ladder preserves that invariant by mapping every INFO entry
    # to :data:`InterventionLevel.OBSERVE`. The issue's suggested
    # table labels some INFO tiers as "Level 1 (absorb)" but a Level 1
    # at INFO would trigger refine for every INFO hint (CONFUSION
    # detector, CONFABULATION_RISK, etc.), which regresses existing
    # behaviour. If an operator later wants a refine-on-hint policy,
    # they subclass and override :meth:`_ladder_level_for`.
    _LADDER: dict[
        DriftKind,
        tuple[
            InterventionLevel | None,  # INFO
            InterventionLevel | None,  # WARNING
            tuple[InterventionLevel, InterventionLevel],  # CRITICAL (first, repeat)
        ],
    ] = {
        DriftKind.CONFUSION: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.CONFABULATION_RISK: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.AGENT_REFUSAL: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.MODEL_REFUSAL: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # LOOPING_REASONING: severity is now graduated (goldfive#204)
        # -- the tool-loop detector emits INFO / WARNING / CRITICAL
        # based on count + category (meta vs work). The ladder mirrors
        # that graduation:
        #
        # * INFO  -> OBSERVE (default fallback; benign meta-tool retries
        #   at the first threshold should not mutate the plan).
        # * WARNING -> ABSORB (refine plan; unchanged from pre-#204).
        # * CRITICAL first -> NUDGE (refine AND queue a soft corrective
        #   follow-up for the overlay loop -- Agent B wires the nudge
        #   consumption in goldfive#forward-progress).
        # * CRITICAL repeat -> PAUSE_ESCALATE (escalate to human if the
        #   loop persists past nudge).
        DriftKind.LOOPING_REASONING: (
            None,
            InterventionLevel.ABSORB,
            (InterventionLevel.NUDGE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.LOOPING_TOOL_CALL: (
            None,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.REASONING_CLUSTER_TIGHTENING: (
            InterventionLevel.OBSERVE,
            None,
            (InterventionLevel.OBSERVE, InterventionLevel.OBSERVE),
        ),
        DriftKind.PLAN_DIVERGENCE: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.INTENT_DIVERGENCE: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.PAUSE_ESCALATE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.TOOL_ERROR: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.RUNAWAY_DELEGATION: (
            None,
            None,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.REFINE_VALIDATION_FAILED: (
            None,
            None,
            (InterventionLevel.PAUSE_ESCALATE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # HUMAN_INTERVENTION_REQUIRED is always CRITICAL. First fire =>
        # PAUSE_ESCALATE. A repeat fire (occurrence_count crosses the
        # threshold) escalates to TERMINATE because it means the pause
        # was already issued and the situation didn't resolve.
        DriftKind.HUMAN_INTERVENTION_REQUIRED: (
            None,
            None,
            (InterventionLevel.PAUSE_ESCALATE, InterventionLevel.TERMINATE),
        ),
        # GOAL_DRIFT (goldfive#143) -- trajectory-level judgment that
        # the tree is no longer advancing ``session.goals``. CRITICAL
        # severity only; routed to Level 4 both on first occurrence
        # and on repeat because goal drift is structural and refine
        # cannot recover from it. Per the goldfive#142 table.
        DriftKind.GOAL_DRIFT: (
            None,
            None,
            (InterventionLevel.PAUSE_ESCALATE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # Self-reported stuck (from reflective check). WARNING by
        # default -- preserve pre-ladder behaviour (ABSORB / refine).
        DriftKind.SELF_REPORTED_STUCK: (
            None,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # BLOCKED, NEW_WORK_DISCOVERED, and user-control derived drifts
        # go through the default fallback below (WARNING -> ABSORB,
        # CRITICAL -> ABSORB first, PAUSE_ESCALATE on repeat), which
        # matches the pre-ladder refine-on-WARNING behaviour verbatim.
    }

    # Ladder entries keyed by drift-kind *value* so future siblings can
    # register entries without depending on a newly-introduced enum
    # member. Consulted BEFORE the enum-keyed table in
    # :meth:`_ladder_level_for` and takes precedence when both match.
    # Empty today -- goldfive#143's GOAL_DRIFT has landed as a real
    # enum member, so the enum-keyed table suffices. Left in place for
    # future cross-PR coordination.
    _LADDER_BY_VALUE: dict[
        str,
        tuple[
            InterventionLevel | None,
            InterventionLevel | None,
            tuple[InterventionLevel, InterventionLevel],
        ],
    ] = {}

    def _ladder_level_for(
        self,
        kind: DriftKind,
        severity: DriftSeverity,
        occurrence_count: int,
    ) -> InterventionLevel:
        """Return the intervention level for ``(kind, severity, count)``.

        The mapping is the single source of truth for the ladder table
        documented on this module's docstring and goldfive#142. Drifts
        with no explicit entry fall through to a safe default that
        preserves the pre-ladder behaviour:

        * INFO -> OBSERVE (no action)
        * WARNING -> ABSORB (refine)
        * CRITICAL, first occurrence -> ABSORB (refine)
        * CRITICAL, repeat occurrence -> PAUSE_ESCALATE

        Subclasses can override this method to tune the table without
        re-implementing :meth:`_handle_drift`.
        """
        entry = self._LADDER_BY_VALUE.get(kind.value)
        if entry is None:
            entry = self._LADDER.get(kind)
        is_repeat = occurrence_count >= self.REFINE_FAILURE_THRESHOLD
        if entry is not None:
            info_level, warning_level, critical_pair = entry
            if severity is DriftSeverity.INFO:
                return info_level or InterventionLevel.OBSERVE
            if severity is DriftSeverity.WARNING:
                return warning_level or InterventionLevel.OBSERVE
            # CRITICAL
            return critical_pair[1] if is_repeat else critical_pair[0]
        # Default fallback for drifts not explicitly in the table.
        if severity is DriftSeverity.INFO:
            return InterventionLevel.OBSERVE
        if severity is DriftSeverity.WARNING:
            return InterventionLevel.ABSORB
        # CRITICAL with no explicit entry -- ABSORB first, escalate on repeat.
        return InterventionLevel.PAUSE_ESCALATE if is_repeat else InterventionLevel.ABSORB

    async def _dispatch_nudge(self, drift: DriftEvent, session: Session) -> None:
        """Level 2 dispatch: queue a soft follow-up message on the session.

        The Runner's overlay loop (goldfive#141) picks up the queued
        nudge at the next invocation boundary and sends it as a gentle
        corrective user message. Until #141 lands, the queue is
        observable but inert; nothing consumes it.
        """
        msg = compose_corrective_user_message(
            drift=drift,
            refined_plan=session.plan,
        )
        session.pending_nudges.append(msg)
        log.debug(
            "DefaultSteerer: queued nudge for kind=%s task=%s: %s",
            drift.kind.value,
            drift.current_task_id or "-",
            msg,
        )

    async def _dispatch_pause_escalate(
        self,
        drift: DriftEvent,
        session: Session,
    ) -> None:
        """Level 4 dispatch: emit HUMAN_INTERVENTION_REQUIRED and pause.

        Does NOT call ``planner.refine`` -- Level 4 signals that the
        planner cannot recover. Sets
        ``session.paused_for_human_intervention`` so the Runner's loop
        blocks before the next task, and emits a CRITICAL
        ``HUMAN_INTERVENTION_REQUIRED`` drift so sinks / the UI can
        surface the pause and let the user decide what to do.

        When the drift reaching Level 4 is *already* a
        ``HUMAN_INTERVENTION_REQUIRED`` (e.g. landed here via the
        generic fallback), we pause but do not re-emit the same drift
        a second time -- the original DriftDetected emission at the
        top of :meth:`_handle_drift` already carried the signal.
        """
        session.paused_for_human_intervention = True
        if drift.kind is DriftKind.HUMAN_INTERVENTION_REQUIRED:
            # Already emitted at the top of _handle_drift; just pause.
            return
        escalation = DriftEvent(
            kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
            severity=DriftSeverity.CRITICAL,
            detail=(
                f"escalated from {drift.kind.value}: {drift.detail}"
                if drift.detail
                else f"escalated from {drift.kind.value}"
            ),
            current_task_id=drift.current_task_id,
            current_agent_id=drift.current_agent_id,
        )
        # Emit directly; do NOT go back through _handle_drift (would
        # infinite-loop at CRITICAL).
        await self._emit_drift_detected(session, escalation)

    # Symbolic cancel-reason tags — mirror
    # :mod:`goldfive.adapters.adk` constants but duplicated as plain
    # strings here to avoid a hard import of the optional ADK adapter
    # module from the provider-agnostic steerer. Keep in sync with
    # :data:`goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER` etc.
    _ADAPTER_CANCEL_REASON_USER_STEER: str = "user_steer"

    def _tag_adapter_cancel_reason(self, drift: DriftEvent) -> None:
        """Set ``adapter._next_cancel_reason`` based on ``drift.kind``.

        USER_STEER drift -> ``"user_steer"``. Other kinds currently leave
        the tag unset so the adapter falls through to the generic
        content variant. Tolerates adapters that don't carry the
        attribute (no-op) and an unbound adapter (no-op). See
        goldfive#139.
        """
        adapter = self._adapter
        if adapter is None:
            return
        if drift.kind is DriftKind.USER_STEER:
            reason = self._ADAPTER_CANCEL_REASON_USER_STEER
        else:
            return
        try:
            adapter._next_cancel_reason = reason
        except Exception as exc:  # noqa: BLE001
            # Adapter doesn't expose the attribute — tolerated.
            log.debug(
                "DefaultSteerer: could not tag adapter cancel reason: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # USER_STEER state handler (goldfive#152)
    # ------------------------------------------------------------------

    async def _apply_user_steer_state(
        self,
        drift: DriftEvent,
        session: Session,
    ) -> None:
        """Side-effects for USER_STEER drift that aren't refine: state
        bookkeeping + goal synthesis.

        Called from :meth:`_handle_drift` just before ``_emit_drift_detected``
        and well before ``planner.refine`` runs, so:

        1. The ``goldfive.active_steer.*`` keys are set so downstream
           observers see the steer before the drift event.
        2. The synthesized Goal is appended / replaced onto
           ``session.goals`` BEFORE ``planner.refine`` reads
           ``list(session.goals)``, so the refined plan sees the
           pivot as a goal, not only as a drift detail string.
        3. The source annotation / control id is appended to
           ``goldfive.processed_steer_ids`` so a retry or UI double-fire
           of the same STEER is a no-op (goldfive#171 dedupe).

        Never raises: a planner that doesn't implement
        ``synthesize_goal_from_steer`` or a synthesis call that fails
        falls through to a minimal passthrough (wrap the steer body as
        a Goal, mode APPEND). The steerer must never break the run on
        a missing optional hook.
        """
        # Recover the raw body + operator author from the originating
        # ControlMessage when it's available on drift.raw (goldfive#171).
        # Falling back to drift.detail preserves back-compat for tests
        # that synthesize a USER_STEER DriftEvent directly without a
        # ControlMessage behind it.
        raw_body, author, steer_id = self._unpack_steer_context(drift)
        body = raw_body.strip()
        # Stamp the active_steer keys regardless of synthesis outcome
        # so readers see "a steer is active as of turn N" even when
        # the planner can't synthesize. ``at_turn`` uses the session's
        # monotonic sequence counter which increments on every emitted
        # event — a cheap, always-available "turn" proxy.
        at_turn = getattr(session, "_next_sequence", 0) or 0
        try:
            _ostate.set_active_steer(session.state, body=body, at_turn=at_turn, author=author)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._apply_user_steer_state: set_active_steer raised: %s",
                exc,
            )
        # Record the dedupe id. Safe to call even with an empty id
        # (the helper no-ops). Done AFTER the active_steer stamp so a
        # reader that inspects ``state`` mid-dispatch always sees the
        # most recent steer is reflected.
        if steer_id:
            try:
                _ostate.record_processed_steer_id(session.state, steer_id)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer._apply_user_steer_state: record_processed_steer_id raised: %s",
                    exc,
                )
        if not body:
            # Empty steer body: nothing to synthesize into a goal. The
            # active_steer.* keys still landed (readers may want to know
            # "a steer was fired even if empty"). Keep goals as-is.
            return
        synth_goal, mode = await self._synthesize_goal_from_steer(body)
        if synth_goal is None:
            return
        mode_norm = (mode or "append").strip().lower()
        if mode_norm == "replace":
            session.goals.clear()
            session.goals.append(synth_goal)
        else:
            # Default / "append" mode: add unless an id collision
            # exists (belt-and-braces if the synthesizer reuses an id).
            existing = {g.id for g in session.goals if g.id}
            if synth_goal.id and synth_goal.id in existing:
                # Replace the colliding goal in-place so the synthesizer
                # can refine a previously-appended steer goal.
                for i, g in enumerate(session.goals):
                    if g.id == synth_goal.id:
                        session.goals[i] = synth_goal
                        break
            else:
                session.goals.append(synth_goal)
        # Refresh the goals_summary so downstream consumers (refine
        # prompt templates, GoldfivePlanner in goldfive#153) see the
        # new shape immediately.
        try:
            _ostate.refresh_goals_summary(session.state, session.goals)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._apply_user_steer_state: refresh_goals_summary raised: %s",
                exc,
            )

    async def _synthesize_goal_from_steer(
        self,
        steer_body: str,
    ) -> tuple[Goal | None, str]:
        """Call ``planner.synthesize_goal_from_steer`` if available.

        Returns ``(goal, mode)`` where ``mode`` is ``"append"`` or
        ``"replace"``. Falls back to a passthrough goal (APPEND) when
        the planner doesn't implement the hook or the call fails.
        The fallback keeps the steer body durable on
        ``session.goals`` even when the planner is a minimal stub
        (PassthroughPlanner / StaticPlanner / tests).
        """
        planner = self._planner
        synth = getattr(planner, "synthesize_goal_from_steer", None)
        if not callable(synth):
            return Goal(id="steer", summary=steer_body), "append"
        try:
            result = await synth(steer_body)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "DefaultSteerer: planner.synthesize_goal_from_steer raised "
                "%s; falling back to passthrough append",
                exc,
            )
            return Goal(id="steer", summary=steer_body), "append"
        if result is None:
            return Goal(id="steer", summary=steer_body), "append"
        # Accept two shapes: a bare Goal (mode defaults to "append") or
        # a ``(Goal, mode)`` tuple. The tuple shape is what the
        # synthesizer should emit in the common case; the bare form is
        # a courtesy for callers / tests that only care about the goal.
        if isinstance(result, tuple) and len(result) == 2:
            goal, mode = result
            if not isinstance(goal, Goal):
                log.warning(
                    "DefaultSteerer: synthesize_goal_from_steer returned "
                    "tuple without Goal; falling back"
                )
                return Goal(id="steer", summary=steer_body), "append"
            return goal, str(mode or "append")
        if isinstance(result, Goal):
            return result, "append"
        log.warning(
            "DefaultSteerer: synthesize_goal_from_steer returned "
            "unrecognised shape %r; falling back",
            type(result),
        )
        return Goal(id="steer", summary=steer_body), "append"

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
        # goldfive#199: stamp the trigger_event_id from the drift onto the
        # plan so out-of-band PlanRevised emitters (the SequentialExecutor's
        # plan-swap detector) can thread it through without needing the
        # drift in scope. Resolution mirrors
        # :func:`goldfive.events._trigger_id_from_drift`: source
        # annotation_id for user-control drifts, ``drift.id`` otherwise.
        # Non-empty for every revision because every ``DriftEvent``
        # dataclass defaults to a UUID4 ``id``. Preserves any pre-existing
        # stamp (e.g. validator-retry chains that re-use the original
        # attempt's trigger id).
        if not revised.revision_trigger_event_id:
            trig_id = DefaultSteerer._drift_annotation_id(drift) or str(
                getattr(drift, "id", "") or ""
            )
            if trig_id:
                revised.revision_trigger_event_id = trig_id
        session.plan = revised
        # goldfive#152: refresh the orchestration-state current plan id
        # so downstream reads see the revised id, not the stale one.
        _ostate.set_current_plan(session.state, revised)

    # --- Event construction ------------------------------------------

    def _new_envelope(self, session: Session) -> Any:
        """Build a fresh ``Event`` envelope via :mod:`goldfive.events`."""
        from goldfive.events import new_event

        return new_event(session.run_id, session.next_sequence(), session_id=session.id)

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
        # goldfive#199: stamp the drift's own id on the wire so a
        # subsequent ``PlanRevised.trigger_event_id`` can strict-match the
        # drift row in harmonograf. Always non-empty — ``DriftEvent``
        # defaults ``id`` to a UUID4.
        drift_id = str(getattr(drift, "id", "") or "")
        if drift_id:
            evt.drift_detected.id = drift_id
        # Stamp the source annotation_id for USER_STEER / USER_CANCEL drifts
        # minted from a ControlMessage with a bridge-supplied annotation_id
        # (goldfive#171). Sinks use this to dedup the drift row against the
        # source annotation — without it a single user STEER surfaces as
        # three cards (annotation row + drift row + plan_revised row) in
        # harmonograf's Intervention view. See goldfive#176 / harmonograf#75.
        ann_id = self._drift_annotation_id(drift)
        if ann_id:
            evt.drift_detected.annotation_id = ann_id
        await self._emit(evt)

    @staticmethod
    def _drift_annotation_id(drift: DriftEvent) -> str:
        """Return the source annotation id for a user-control drift, or "".

        Looks at :attr:`DriftEvent.raw` — populated by
        :meth:`_drift_from_control` when the drift was minted from a STEER
        / CANCEL ControlMessage — and extracts
        ``payload["annotation_id"]`` (set by the bridge per goldfive#171).
        Returns "" for drifts that goldfive minted itself (loop detection,
        goal drift, etc), whose ``raw`` is either absent or not a
        ControlMessage. Non-string payloads are coerced to str so a
        mis-typed bridge still flows the id through.
        """
        from goldfive.control import ControlMessage

        raw = getattr(drift, "raw", None)
        if not isinstance(raw, ControlMessage):
            return ""
        payload = raw.payload if isinstance(raw.payload, dict) else {}
        return str(payload.get("annotation_id", "") or "")

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
        # goldfive#199: stamp ``trigger_event_id`` on the PlanRevised
        # envelope for EVERY refine — user-control (via source
        # annotation_id) and autonomous (via drift.id). Harmonograf's
        # intervention aggregator merges PlanRevised rows by strict id
        # only (legacy time-window fallback is behind a disabled env
        # flag). Priority: pre-stamped ``revision_trigger_event_id`` on
        # the revised plan (from ``_apply_revision`` or validator-retry
        # chain) → source annotation_id from the drift → drift.id.
        trig_id = revised.revision_trigger_event_id or (
            self._drift_annotation_id(drift) or str(getattr(drift, "id", "") or "")
        )
        if trig_id:
            evt.plan_revised.trigger_event_id = trig_id
        # Populate the minimal cross-revision diff so sinks that want a
        # "what changed" view don't have to re-fetch and diff the two
        # plans client-side. prev_plan may be None on the first revision
        # of a run that never received an initial plan — the helper
        # treats that as "everything in revised is newly added".
        evt.plan_revised.diff.CopyFrom(build_plan_revision_diff(prev_plan, revised))
        await self._emit(evt)
