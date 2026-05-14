"""Drift event lifecycle, classification, observation, and dispatch helpers.

Extracted from :mod:`goldfive.steerer` in Wave C of the steerer split.
Buckets **3a** (observability primitives), **3b** (observation entry
points + judge orchestration), and **3c** (dispatch + ladder + promotion)
all now live here. :class:`DefaultSteerer` is the thin router that owns
the shared mutable state (sinks, planner, adapter, control_channel,
ContextVar, background-task sets, plan locks, ``observation_only`` flag,
config, ``REFINE_FAILURE_THRESHOLD`` / ``PROGRESS_STALL_THRESHOLD_SECONDS``
class constants) plus the public shim layer that tests + external
callers (the planner, executors) historically hooked at the
``steerer.X`` bare-attribute name.

This module owns the drift-event observability + classification surface
plus the observation entry points (``observe`` / ``observe_reasoning``),
the background judge orchestration (``_run_judge_background`` /
``_run_goal_drift_judge_background`` / ``_spawn_*_background``), the
reflective self-progress check (``maybe_run_reflective_check`` /
``note_llm_call`` / ``_emit_reflective_failure``), the GOAL_DRIFT
trajectory-level check (``maybe_run_goal_drift_check`` /
``note_agent_turn`` / ``_maybe_run_goal_drift_on_task_boundary``), the
bounded note-buffer family (``note_agent_activity`` /
``note_tool_observation``), and the **dispatch + ladder + promotion**
machinery: ``handle_drift`` (the central drift-routing method),
``_promote_drift_to_steer`` (audit issue #402 — dispatch-before-plan-swap
ordering is preserved here, not fixed), the intervention ladder
(``_ladder_level_for`` / ``_LADDER`` / ``_LADDER_BY_VALUE``), ladder
dispatch (``_dispatch_nudge`` / ``_dispatch_goldfive_steer_control`` /
``_dispatch_goldfive_pause_control`` / ``_dispatch_pause_escalate``),
adapter cancel tagging, the late-drift gate
(``_is_late_drift_for_terminated_invocation``),
``request_invocation_cancel``, ``_cancel_inflight_for_revision``,
``_apply_user_steer_state`` (#199 / #183 synthetic-USER_STEER
suppression plumbing), and refine-outcome bookkeeping
(``_record_refine_outcome`` / ``reset_for_turn`` /
``_occurrence_count_for_ladder`` /
``_escalate_refine_failure_as_critical_drift``).

Responsibilities (this PR)
--------------------------

* :meth:`_emit_drift_detected` — the canonical ``DriftDetected`` event
  emit. Stamps wire fields, normalises ``authored_by`` belt-and-braces
  for direct callers (``_dispatch_pause_escalate`` /
  ``_escalate_refine_failure_as_critical_drift``), routes through
  state_store lifecycle helpers (#271) for ``condition_id`` /
  ``lifecycle`` / ``prev_severity``, and — on **terminal drifts**
  (``HUMAN_INTERVENTION_REQUIRED`` / ``REPEATED_FAILURE``) — asks the
  bound adapter's plugin to close every still-open boundary so
  observability sinks don't render permanently-open spans.

* :meth:`_stamp_drift_lifecycle` + :meth:`_drift_lifecycle_pb_value` —
  the drift-as-stateful-condition stamping (#271): the same
  ``(kind, task, agent, run_id)`` tuple in one turn collapses onto a
  stable ``condition_id`` whose ``lifecycle`` field walks from
  ``OPENED`` → ``ESCALATING``.

* :meth:`_is_terminal_drift` + :attr:`_TERMINAL_DRIFT_KINDS` — the
  whitelist of drift kinds that trigger plugin-side boundary cleanup
  on emit. ``LOOPING_REASONING`` is **deliberately excluded** — its
  CRITICAL-first tier maps to ``NUDGE`` (recoverable); the eventual
  ``HUMAN_INTERVENTION_REQUIRED`` emission on escalation is the
  cleanup trigger.

* :meth:`_resolve_authored_by` + :attr:`_USER_AUTHORED_DRIFT_KINDS`
  + :meth:`_drift_annotation_id` — source-attribution helpers.
  ``USER_STEER`` / ``USER_CANCEL`` / ``USER_PAUSE`` default to
  ``"user"``; everything else defaults to ``"goldfive"``. The
  annotation id is extracted from the originating
  :class:`ControlMessage` payload (#171) when one is present.

* :meth:`detect_drift` — primitive classifier dispatch (tool-error →
  refusal → stop-reason). Stamps ``observed_revision_index`` (#245)
  at observation time so the dispatch-time gate in ``_handle_drift``
  can drop verdicts whose revision is older than the live plan's.

* :meth:`_drift_from_control` + :meth:`_unpack_steer_context` +
  :meth:`_is_duplicate_steer` + :meth:`_steer_dedupe_id` — the
  :class:`~goldfive.control.ControlMessage` → ``USER_*`` drift mapping
  with annotation-id dedupe (#171). STEER messages whose source
  annotation id was already processed on this session no-op at the
  ``observe`` entry point.

* :meth:`report_new_work_discovered` + :meth:`report_plan_divergence`
  — reporting-tool hooks that mint drifts directly. PLAN_DIVERGENCE
  is structurally disabled (#252 → CAPABILITY_MISMATCH at #253) but
  the report path remains so external callers / replays don't crash.

* Display helpers: :meth:`_truncate_trigger_input`,
  :meth:`_summarize_recent_tool_calls`,
  :meth:`_summarize_recent_reasoning`. Bounded summarisation used by
  the reflective check prompt and by ``trigger_input`` stamping.

* :meth:`_parse_reflective_response` — tolerant JSON-from-LLM parser
  used by the reflective check verdict path.

The module DOES NOT own
----------------------

* Task-status transitions — moved in bucket 1
  (:mod:`goldfive.task_state_machine`).
* Plan-revision install + refine observability — moved in bucket 2
  (:mod:`goldfive.plan_reviser`).
* Class constants ``REFINE_FAILURE_THRESHOLD`` and
  ``PROGRESS_STALL_THRESHOLD_SECONDS`` — stay on
  :class:`DefaultSteerer` so subclasses + tests that historically
  tuned them at ``DefaultSteerer.X`` / ``steerer.X`` keep working;
  read from this module as ``self._steerer.X``.
* The :class:`~goldfive.steerer.InterventionLevel` enum — stays as a
  module-level export of :mod:`goldfive.steerer` for back-compat with
  callers that import it as ``from goldfive.steerer import
  InterventionLevel``.

Audit issue #402 — dispatch-before-plan-swap (fixed)
----------------------------------------------------

:meth:`_promote_drift_to_steer` dispatches the ``GOLDFIVE_STEER``
ControlMessage **after** ``planner.refine_steer`` has produced a
revised plan and :meth:`_emit_plan_revised` has swapped it onto the
session. Pre-fix the dispatch fired BEFORE refine, so the payload's
``replacement_task_ids`` carried the prior plan's task ids and the
executor's overlay loop re-invoked against ids the imminent revision
was about to remove / cancel. Ordering now mirrors
:meth:`_handle_drift`'s CANCEL_REINVOKE branch.

All cross-component calls go through the router back-reference passed
to :meth:`DriftObserver.__init__` (``self._steerer``). This keeps the
components decoupled and lets the router own the shared state
(sinks, adapter, ContextVar for the active session, plan locks,
``observation_only`` flag, config).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive import _state_audit
from goldfive import state_store as _ostate
from goldfive.drift import (
    classify_refusal,
    classify_stop_reason,
    classify_tool_error,
)
from goldfive.types import (
    CancellationRequest,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    RefineOutcome,
    Session,
    Task,
    TaskStatus,
)

if TYPE_CHECKING:
    from goldfive.steerer import DefaultSteerer, InterventionLevel

# Shape of the opt-in reflective LLM callable. Re-exported here so the
# DriftObserver methods that own the reflective + goal-drift + reasoning
# judge plumbing can carry their typed signatures without round-tripping
# through :mod:`goldfive.steerer` (avoids a circular import).
ReflectiveCallLLM = Callable[[str, str, str], Awaitable[str]]

log = logging.getLogger(__name__)
# Wave C bucket 3b/3c post-cleanup: the module previously kept a
# sibling ``_steerer_log = logging.getLogger("goldfive.steerer")``
# because the test corpus asserted on ``record.name == "goldfive.steerer"``.
# Those assertions have been widened to message-content predicates so
# we no longer need the sibling — every log line on this module now
# lands on the natural ``goldfive.drift_observer`` logger (``log``).
# Operators / harmonograf consumers should grep on message content
# (``"DefaultSteerer."`` prefixes, ``"stale judge verdict"``, etc.)
# or on the parent ``goldfive`` logger.


class DriftObserver:
    """Drift event observability + classification + dispatch helpers.

    Constructed by :class:`DefaultSteerer` and exposed publicly as
    ``DefaultSteerer.drift`` (goldfive#410). Callers — executors,
    adapters, the runner, planners, tests — reach the drift-emit /
    detection / attribution / dispatch surface directly as
    ``steerer.drift.X``.

    Buckets retained from the original steerer-split:

    * bucket 3a — observability primitives (``_emit_drift_detected``,
      lifecycle stamping, ``_close_open_boundaries_for_terminal_drift``,
      attribution helpers);
    * bucket 3b — observation entry points + judge orchestration
      (``observe`` / ``observe_reasoning`` / ``maybe_run_*_check``);
    * bucket 3c — dispatch + ladder + promotion + refine-outcome
      bookkeeping (``handle_drift`` / ``_ladder_level_for`` /
      ``_promote_drift_to_steer`` / ``_record_refine_outcome``).
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    # goldfive#271 follow-up: drift kinds that are unrecoverable on
    # emit. Boundary cleanup hooks fire on these to close any
    # still-open spans the cooperative-cancel path would otherwise
    # leave dangling.
    #
    # Inclusion rationale (and why ``LOOPING_REASONING`` is NOT here
    # despite being listed in the v15 stuck-span evidence):
    #
    # * ``HUMAN_INTERVENTION_REQUIRED`` — the ladder always emits this
    #   at CRITICAL with PAUSE_ESCALATE / TERMINATE semantics; the run
    #   pauses for an operator and no normal ``after_agent_callback``
    #   will fire on the open invocations.
    # * ``REPEATED_FAILURE`` — emitted from
    #   :meth:`_record_refine_failure` ONLY after the offending task is
    #   marked ``FAILED`` non-recoverable; the executor will not
    #   resume it.
    # * ``LOOPING_REASONING`` is deliberately NOT here despite being
    #   listed in the v15 evidence: it is graduated (INFO / WARNING /
    #   CRITICAL) and CRITICAL-first maps to ``NUDGE`` (recoverable —
    #   refine + corrective follow-up). Closing on the LOOPING_REASONING
    #   emission itself would corrupt the boundary pair when the run
    #   actually recovers. The CRITICAL-repeat path escalates to
    #   ``PAUSE_ESCALATE``, which emits a fresh
    #   ``HUMAN_INTERVENTION_REQUIRED`` drift; that emission triggers
    #   the close, so the v15 stuck-spans symptom is still cleaned up
    #   on the actual terminal step.
    _TERMINAL_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.HUMAN_INTERVENTION_REQUIRED,
            DriftKind.REPEATED_FAILURE,
        }
    )

    # goldfive-steer-unification: drift kinds that are always "user"-
    # authored when no explicit source was stamped. Any other kind
    # defaults to "goldfive" (the detector path).
    _USER_AUTHORED_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.USER_PAUSE,
        }
    )

    # Drift kinds whose origin is a user intervention (USER_STEER /
    # USER_CANCEL) or a trajectory-level signal that has its own rate
    # limit (GOAL_DRIFT — task-boundary throttle via
    # ``_last_goal_drift_check_ts``). These kinds bypass the time-based
    # cooldown and the progress-stall escalation: user intent is always
    # honoured, and trajectory-wide drifts have no single task whose
    # progress could be measured.
    _USER_OR_TRAJECTORY_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.GOAL_DRIFT,
        }
    )

    # Liberal JSON extractor: tolerates markdown code fences and leading /
    # trailing prose around the object.
    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, steerer: DefaultSteerer) -> None:
        # Back-reference to the router. Used to reach the shared
        # event-emission primitives (``_new_envelope`` / ``_emit`` /
        # ``_drift_kind_pb_value`` / ``_drift_severity_pb_value``),
        # the cross-component cooperators (the bound adapter for
        # boundary cleanup, the planner, the plan-revision install +
        # refine-observability surface owned by
        # :class:`~goldfive.plan_reviser.PlanReviser` and the
        # task-state-machine surface owned by
        # :class:`~goldfive.task_state_machine.TaskStateMachine`), and
        # the router-owned shared state (the per-async-task
        # ``_active_session_var`` ContextVar plumbing for the planner
        # span-context provider, the ``_observation_only`` gate read
        # via :meth:`DefaultSteerer._should_inject`, and the
        # ``REFINE_FAILURE_THRESHOLD`` / ``PROGRESS_STALL_THRESHOLD_SECONDS``
        # class constants).
        self._steerer = steerer

    # ------------------------------------------------------------------
    # Drift event emission + lifecycle stamping
    # ------------------------------------------------------------------

    async def _emit_drift_detected(self, session: Session, drift: DriftEvent) -> None:
        evt = self._steerer._new_envelope(session)
        evt.drift_detected.kind = self._steerer._drift_kind_pb_value(drift.kind)
        evt.drift_detected.severity = self._steerer._drift_severity_pb_value(drift.severity)
        evt.drift_detected.detail = drift.detail
        evt.drift_detected.current_task_id = drift.current_task_id
        evt.drift_detected.current_agent_id = drift.current_agent_id
        # goldfive#245 — forward the observation-time plan revision so
        # downstream sinks can render "this verdict was against
        # revision N, current is M" and dedup gate-skipped drifts.
        if drift.observed_revision_index:
            evt.drift_detected.observed_revision_index = int(
                drift.observed_revision_index
            )
        # goldfive-steer-unification: source attribution. Normalise a
        # missing ``authored_by`` on the drift here so downstream sinks
        # never see an unattributed event from goldfive-internal paths
        # (the ladder dispatcher normalises pre-emit; this is a belt-
        # and-braces for direct ``_emit_drift_detected`` callers like
        # ``_dispatch_pause_escalate`` / ``_escalate_refine_failure_as_critical_drift``).
        evt.drift_detected.authored_by = self._resolve_authored_by(drift)
        evt.drift_detected.suppressed_by_user_steer = bool(drift.suppressed_by_user_steer)
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
        # Forward the detector-supplied trigger_input onto the wire so
        # sinks that render a Gantt / timeline can explain "why did
        # goldfive flag this?" without re-fetching raw agent transcripts.
        # Always truncated by the detector before it lands on the drift;
        # we belt-and-braces truncate here in case an out-of-tree
        # detector forgot. Empty string for user-control drifts (their
        # explanation lives on the source annotation).
        trigger_input = getattr(drift, "trigger_input", "") or ""
        if trigger_input:
            evt.drift_detected.trigger_input = self._truncate_trigger_input(trigger_input)
        # goldfive#271 PR1 — drift-as-stateful-condition. Route the emit
        # through the state_store lifecycle helpers so multiple
        # emits for the same logical condition (kind+task+agent within
        # the current turn) collapse onto one ``condition_id`` and the
        # wire carries lifecycle / prev_severity. Additive: legacy
        # fields (kind/severity/detail/synthetic/id/...) are unchanged
        # so any sink that doesn't know the new fields still sees one
        # row per emit and renders it identically.
        self._stamp_drift_lifecycle(session, drift, evt)
        await self._steerer._emit(evt)
        # goldfive#271 follow-up: when a terminal drift fires the run
        # cannot recover on its own — any boundary still open at this
        # point belongs to an invocation that will not get a paired
        # ``after_agent_callback`` (the executor is about to pause or
        # tear down). Walk the plugin's still-open boundaries and emit
        # the paired ``InvocationBoundaryExited(reason=terminal_drift:
        # <kind>)`` so observability sinks (and harmonograf's Gantt)
        # don't render permanently-open spans for coordinator /
        # research / refine_steer LLM_CALLs that v15 left in
        # ``dur=(open)``.
        if self._is_terminal_drift(drift):
            await self._close_open_boundaries_for_terminal_drift(drift)

    @classmethod
    def _is_terminal_drift(cls, drift: DriftEvent) -> bool:
        """Return True iff ``drift`` should trigger boundary cleanup.

        Membership-only check against :attr:`_TERMINAL_DRIFT_KINDS`;
        every kind in the set is unconditionally terminal at emit
        time. See the set definition for the rationale on which kinds
        are included (and why ``LOOPING_REASONING`` is NOT — its
        CRITICAL-first tier is still recoverable; the eventual
        ``HUMAN_INTERVENTION_REQUIRED`` emission on escalation triggers
        cleanup instead).
        """
        return drift.kind in cls._TERMINAL_DRIFT_KINDS

    async def _close_open_boundaries_for_terminal_drift(self, drift: DriftEvent) -> None:
        """Ask the bound adapter's plugin to close every still-open boundary.

        Reuses the canonical ``close_open_boundaries`` helper from
        PR #307 so the cleanup path is identical to the
        ``except CancelledError`` arc in
        :meth:`ADKAdapter._invoke_internal`. The reason string is
        ``terminal_drift:<kind>`` so sink consumers can distinguish a
        steerer-driven cleanup from the cancel / error paths.

        Best-effort: tolerates an unbound adapter, an adapter without
        the plugin attribute, a plugin without the helper (third-party
        / legacy), and any exception from the plugin (logged at DEBUG).
        Never re-raises — the drift was already emitted on the wire,
        and a failed cleanup must not corrupt the steerer's pause /
        escalate flow.
        """
        adapter = self._steerer._adapter
        if adapter is None:
            return
        plugin = getattr(adapter, "_plugin", None)
        if plugin is None:
            return
        helper = getattr(plugin, "close_open_boundaries", None)
        if not callable(helper):
            return
        reason = f"terminal_drift:{drift.kind.value}"
        try:
            await helper(reason=reason)
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            log.debug(
                "DefaultSteerer._close_open_boundaries_for_terminal_drift: "
                "plugin.close_open_boundaries(reason=%r) raised: %s",
                reason,
                exc,
            )

    def _stamp_drift_lifecycle(
        self,
        session: Session,
        drift: DriftEvent,
        evt: Any,
    ) -> None:
        """Stamp ``condition_id`` / ``lifecycle`` / ``prev_severity`` on ``evt``.

        Routes the emit through
        :func:`state_store.open_or_escalate_drift` keyed by
        ``(kind, current_task_id, current_agent_id, run_id)``. The first
        emit for a given tuple in a turn opens a new condition and stamps
        ``DRIFT_LIFECYCLE_OPENED``; subsequent emits stamp
        ``DRIFT_LIFECYCLE_ESCALATING`` and carry the previous severity in
        ``prev_severity``. The drift's intrinsic ``id`` is NOT used as
        the condition key — the condition is a logical group that the
        same kind+task+agent can re-open within a turn, and the
        per-event id (#199) is intentionally distinct from the
        condition id (#271).

        Tolerant of partial state: a drift with empty ``current_task_id``
        / ``current_agent_id`` still produces a stable condition_id (the
        sha1 just hashes the empty strings), so user-control drifts
        without a pinned task collapse onto a single condition per turn
        per kind.

        Synthetic drifts are routed through the helpers as well so the
        wire still carries the lifecycle metadata; sinks that filter
        ``synthetic == true`` already drop them and continue to do so.
        """
        try:
            from goldfive.pb.goldfive.v1 import types_pb2

            turn_id = str(getattr(session, "run_id", "") or "")
            tracked = _ostate.open_or_escalate_drift(
                session.state,
                kind=drift.kind,
                task_id=str(getattr(drift, "current_task_id", "") or ""),
                agent_id=str(getattr(drift, "current_agent_id", "") or ""),
                turn_id=turn_id,
                severity=drift.severity,
            )
            evt.drift_detected.condition_id = tracked.condition_id
            evt.drift_detected.lifecycle = self._drift_lifecycle_pb_value(
                tracked.lifecycle, types_pb2
            )
            if tracked.prev_severity is not None:
                evt.drift_detected.prev_severity = self._steerer._drift_severity_pb_value(
                    tracked.prev_severity
                )
        except Exception as exc:  # noqa: BLE001
            # Lifecycle stamping is observability-only; never let a
            # bookkeeping bug break the wire emit. Log and fall through
            # to the legacy single-shot view (UNSPECIFIED lifecycle,
            # empty condition_id).
            log.debug("DefaultSteerer: drift-lifecycle stamping skipped (%s)", exc)

    @staticmethod
    def _drift_lifecycle_pb_value(lifecycle: str, types_pb2: Any) -> int:
        """Map an :mod:`state_store` lifecycle string to the proto enum."""
        mapping = {
            _ostate.LIFECYCLE_OPENED: "DRIFT_LIFECYCLE_OPENED",
            _ostate.LIFECYCLE_ESCALATING: "DRIFT_LIFECYCLE_ESCALATING",
            _ostate.LIFECYCLE_RESOLVED: "DRIFT_LIFECYCLE_RESOLVED",
            _ostate.LIFECYCLE_HUMAN_INTERVENTION_REQUIRED: (
                "DRIFT_LIFECYCLE_HUMAN_INTERVENTION_REQUIRED"
            ),
        }
        name = mapping.get(lifecycle, "DRIFT_LIFECYCLE_UNSPECIFIED")
        return getattr(types_pb2, name, getattr(types_pb2, "DRIFT_LIFECYCLE_UNSPECIFIED", 0))

    # ------------------------------------------------------------------
    # Source attribution helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_authored_by(cls, drift: DriftEvent) -> str:
        """Return the effective ``authored_by`` value for ``drift``.

        Honours an explicit value on the dataclass first; otherwise
        derives from the drift kind. User-control kinds → ``"user"``;
        everything else → ``"goldfive"`` (the detector path).
        """
        explicit = str(getattr(drift, "authored_by", "") or "").strip()
        if explicit:
            return explicit
        if drift.kind in cls._USER_AUTHORED_DRIFT_KINDS:
            return "user"
        return "goldfive"

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

    # ------------------------------------------------------------------
    # Drift detection / classification primitives
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        event: Any,
        session: Session,
    ) -> DriftEvent | None:
        """Classify ``event`` via the modular classifiers in :mod:`drift`.

        Classifiers are tried in order of specificity: tool-error shapes
        first (most structured), then refusal markers in text, then
        stop-reason tokens. The first match wins.

        The primitive classifiers in :mod:`goldfive.drift` (tool-error,
        refusal, stop-reason) don't take a session, so we stamp the
        observation-time plan revision (goldfive#245) here on the
        positive side of the funnel — same observation moment, same
        snapshot the call sees. The dispatch-time gate in
        :meth:`_handle_drift` then drops verdicts whose revision is
        older than the live plan's.
        """
        observed_revision_index = 0
        plan = getattr(session, "plan", None)
        if plan is not None:
            observed_revision_index = int(getattr(plan, "revision_index", 0) or 0)

        def _stamp(d: DriftEvent | None) -> DriftEvent | None:
            if d is None:
                return None
            # Only stamp when unset so explicit observation-time stamps
            # from inner classifiers win.
            if not d.observed_revision_index and observed_revision_index:
                d.observed_revision_index = observed_revision_index
            return d

        drift = _stamp(classify_tool_error(event))
        if drift is not None:
            return drift

        # Refusal scan — tolerates raw strings, dicts, objects.
        drift = _stamp(classify_refusal(event))
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
            drift = _stamp(classify_stop_reason(stop_reason))
            if drift is not None:
                return drift

        return None

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
                authored_by="user",
            )
        if kind_str == ControlKind.CANCEL.value:
            return DriftEvent(
                kind=DriftKind.USER_CANCEL,
                severity=DriftSeverity.CRITICAL,
                detail=reason,
                current_task_id=session.current_task_id,
                raw=event,
                authored_by="user",
            )
        if kind_str == ControlKind.PAUSE.value:
            return DriftEvent(
                kind=DriftKind.USER_PAUSE,
                severity=DriftSeverity.INFO,
                detail=note,
                current_task_id=session.current_task_id,
                raw=event,
                authored_by="user",
            )
        return None

    # ------------------------------------------------------------------
    # Reporting-tool drift hooks
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
        await self.handle_drift(drift, session)

    async def report_plan_divergence(
        self,
        *,
        session: Session,
        note: str,
        suggested_action: str = "",
    ) -> None:
        """No-op: PLAN_DIVERGENCE drift is disabled (goldfive#252).

        # goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH
        (#253) — disabled here. The detector path still records the
        ``divergence_flag`` so observers see "something happened", but
        no drift fires through the steerer pipeline.
        """
        session.divergence_flag = True
        detail = f"{note} (suggested: {suggested_action})" if suggested_action else note
        log.debug(
            "DefaultSteerer.report_plan_divergence: PLAN_DIVERGENCE "
            "drift disabled (goldfive#252); detector observed %r",
            detail,
        )
        return

    # ------------------------------------------------------------------
    # Display + bounded-summarisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_trigger_input(text: str, limit: int = 2048) -> str:
        """Truncate ``text`` for use as a ``DriftDetected.trigger_input``.

        Uses the same suffix convention as the reasoning-judge
        observability event so consumers see one truncation marker
        regardless of which detector produced the drift.
        """
        if not isinstance(text, str):
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + " … [truncated]"

    @staticmethod
    def _summarize_recent_tool_calls(session: Session, *, limit: int = 10) -> str:
        """Build a short human-readable summary of the last N tool calls.

        Reads from ``session.recent_tool_observations`` (populated by
        :meth:`note_tool_observation` from the adapter's
        ``after_tool_callback`` / ``on_tool_error_callback`` hooks).
        Falls back to "(no recent tool calls)" when the buffer is empty.

        Each rendered entry is ``tool_name(args_preview)`` with an
        ``[ERROR: ...]`` suffix when the observation was flagged as an
        error. ``args_preview`` is already truncated to 240 chars by the
        writer; we further trim to 120 here to keep the reflective-check
        prompt bounded. The most recent ``limit`` entries are emitted
        oldest-first for readability.

        Adapters that want richer summaries can subclass and override.
        """
        hist = getattr(session, "recent_tool_observations", None) or []
        if not hist:
            return "(no recent tool calls)"
        # Take the tail (most recent ``limit`` entries) oldest-first.
        tail = list(hist)[-limit:]
        lines: list[str] = []
        for entry in tail:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("tool_name", "") or "")
            if not name:
                continue
            args_preview = str(entry.get("args_preview", "") or "")[:120]
            rendered = f"{name}({args_preview})"
            if entry.get("is_error"):
                err = str(entry.get("error_message", "") or "")[:80]
                rendered += f" [ERROR: {err}]" if err else " [ERROR]"
            lines.append(rendered)
        if not lines:
            return "(no recent tool calls)"
        return ", ".join(lines)

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

    # ------------------------------------------------------------------
    # Structural-escalation helpers (progress-stall, handler-exhausted)
    # ------------------------------------------------------------------
    #
    # ``_is_task_progress_stalled`` is a pure predicate (no router-state
    # mutation, no side effects) — extracted here so the
    # ``_USER_OR_TRAJECTORY_DRIFT_KINDS`` constant has one owner.
    # ``_emit_progress_stalled_escalation`` and
    # ``_emit_handler_exhausted_escalation`` route through the still-on-
    # router pause-control dispatcher; they were originally clustered
    # with the drift-emit primitives and stay grouped here so the
    # PROGRESS_STALL_THRESHOLD_SECONDS reference reads off the router.

    def _is_task_progress_stalled(self, drift: DriftEvent, session: Session) -> bool:
        """Return ``True`` iff the drift's task has had no progress recently.

        goldfive#271 — replaces the deleted count-based cap with a
        progress-grounded structural guarantee. A productively-iterating
        task continually emits progress events
        (``mark_task_running`` / ``mark_task_progress`` /
        ``_emit_task_transitioned``); a stuck task does not. When a
        drift fires for a task whose ``Session.task_last_progress_at``
        is older than :attr:`PROGRESS_STALL_THRESHOLD_SECONDS`, we
        treat the drift as unresolvable by another refine and escalate
        to ``HUMAN_INTERVENTION_REQUIRED``.

        Returns ``False`` (no gate) when:

        * The threshold is non-positive (disabled).
        * The drift kind is a user / trajectory-level drift (always
          honoured / has its own rate limit).
        * The drift carries no ``current_task_id`` (trajectory-wide
          signals cannot be progress-stalled).
        * The task has no recorded progress yet (a freshly-running
          task may not have stamped ``task_last_progress_at`` if the
          drift fires before the first transition is processed).
        """
        threshold = self._steerer.PROGRESS_STALL_THRESHOLD_SECONDS
        if threshold <= 0:
            return False
        if drift.kind in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            return False
        task_id = drift.current_task_id
        if not task_id:
            return False
        last_at = session.task_last_progress_at.get(task_id)
        if last_at is None:
            # No progress signal yet — give the task the benefit of the
            # doubt. The first ``mark_task_running`` stamps the table,
            # so this branch only fires for the very first tick of a
            # fresh task or a task that never transitioned.
            return False
        age = time.monotonic() - last_at
        if age < threshold:
            return False
        log.warning(
            "task progress stalled (task=%r kind=%s age=%.1fs threshold=%.1fs); "
            "escalating to HUMAN_INTERVENTION_REQUIRED",
            task_id,
            drift.kind.value,
            age,
            threshold,
        )
        return True

    async def _emit_progress_stalled_escalation(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Emit a ``HUMAN_INTERVENTION_REQUIRED`` drift + pause the runner.

        Called from ``_handle_drift`` / ``_promote_drift_to_steer`` when
        :meth:`_is_task_progress_stalled` returns True. Phase 2 of the
        path-duality fix: dispatches a ``GOLDFIVE_PAUSE_ESCALATE``
        ControlMessage so the executor's pre-task loop blocks via the
        same channel state as a user ``PAUSE``. Emits a CRITICAL drift
        carrying the underlying (kind, task) so sinks / the UI can
        surface the stall.
        """
        task_id = drift.current_task_id
        last_at = session.task_last_progress_at.get(task_id) if task_id else None
        age = (time.monotonic() - last_at) if last_at is not None else 0.0
        reason = (
            f"task progress stalled for {drift.kind.value} on task "
            f"{task_id or '(trajectory)'}: "
            f"{age:.0f}s since last progress, threshold "
            f"{self._steerer.PROGRESS_STALL_THRESHOLD_SECONDS:.0f}s"
        )
        await self._dispatch_goldfive_pause_control(drift, session, reason=reason)
        escalation = DriftEvent(
            kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
            severity=DriftSeverity.CRITICAL,
            detail=reason,
            current_task_id=task_id,
            current_agent_id=drift.current_agent_id,
        )
        # Emit directly; do NOT recurse through ``_handle_drift``.
        await self._emit_drift_detected(session, escalation)

    async def _emit_handler_exhausted_escalation(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Emit a ``HUMAN_INTERVENTION_REQUIRED`` drift for handler exhaustion.

        goldfive#271 — drift-handler exhaustion as the escalation
        primitive. Called when a refine handler has tried and cannot
        produce a meaningful change for this drift (today: a
        structurally identical revision; future: explicit
        ``RefineExhausted`` sentinel from a planner). Phase 2 of the
        path-duality fix: dispatches a ``GOLDFIVE_PAUSE_ESCALATE``
        ControlMessage so the executor's pre-task loop blocks via the
        same channel state as a user ``PAUSE``. Emits a CRITICAL
        drift so the operator can decide whether to cancel or steer.
        """
        reason = (
            f"refine handler exhausted for {drift.kind.value} on task "
            f"{drift.current_task_id or '(trajectory)'}: "
            f"planner cannot produce a meaningful change"
        )
        await self._dispatch_goldfive_pause_control(drift, session, reason=reason)
        escalation = DriftEvent(
            kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
            severity=DriftSeverity.CRITICAL,
            detail=reason,
            current_task_id=drift.current_task_id,
            current_agent_id=drift.current_agent_id,
        )
        # Emit directly; do NOT recurse through ``_handle_drift``.
        await self._emit_drift_detected(session, escalation)

    # ==================================================================
    # Bucket 3b: observation entry points + judge orchestration
    # ==================================================================
    #
    # All methods below were moved from :class:`DefaultSteerer` verbatim
    # in bucket 3b of the steerer split. They reach back into the router
    # via ``self._steerer`` for:
    #
    # * ``_handle_drift`` — dispatch + ladder (stays on router until 3c)
    # * ``_is_late_drift_for_terminated_invocation`` — late-drift gate
    #   (stays on router; preserves goldfive#242 / #230 / #319 semantics)
    # * ``_find_task`` — plan-task lookup helper (staticmethod on router)
    # * ``_background_judges`` / ``_background_drifts`` — task-tracking
    #   sets (still owned by the router so 3c can cancel/drain them
    #   alongside dispatch state)
    # * ``_sinks`` / ``_adapter`` / ``_reasoning_drift_*`` /
    #   ``_goal_drift_*`` / ``_reflective_*`` — config + wiring (owned
    #   by the router constructor)

    # ------------------------------------------------------------------
    # Reflective check — prompt + budget constants
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

    # Per-callsite ``max_output_tokens`` budget (goldfive#271 follow-up).
    # The reflective check returns a small JSON verdict, but Qwen 3.5
    # thinking models share think+answer under one ceiling — 16384
    # covers the think prelude on the 35B variant without permitting
    # unbounded essays. See
    # :func:`goldfive._llm.call_llm_budget` docstring for sizing rationale.
    REFLECTIVE_MAX_OUTPUT_TOKENS: int = 16384

    # Minimum spacing between two task-boundary-triggered GOAL_DRIFT
    # judge calls, in seconds (goldfive#219). Task transitions can
    # arrive back-to-back (e.g. a cascade-cancel fan-out or a fast
    # research→write→review pipeline), and we don't want to pay for
    # N LLM calls per burst; one is enough to catch drift. Turn-based
    # scheduling has its own interval (``goal_drift_check_interval``)
    # and is not affected by this guard.
    _GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S: float = 10.0

    # ------------------------------------------------------------------
    # Observation entry points
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
        :data:`state_store.KEY_PROCESSED_STEER_IDS` with FIFO
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
            # Subclasses that need custom detection can subclass
            # :class:`DriftObserver` and override
            # :meth:`detect_drift`. The router-level shim that used
            # to dispatch back here was retired in goldfive#410.
            drift = self.detect_drift(event, session)
        if drift is None:
            return
        await self.handle_drift(drift, session)

    async def observe_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,  # noqa: ARG002 -- reserved for future detectors
        session: Session,
        provider: str = "",  # noqa: ARG002 -- reserved for per-provider dispatch
        agent_name: str = "",
    ) -> None:
        """Feed a chain-of-thought / reasoning block into the drift pipeline.

        Appends ``text`` to ``session.reasoning_history`` (bounded by
        ``session.reasoning_history_max``), then runs the reasoning
        detectors. Emits at most one drift per call.

        Pipeline dispatch (goldfive#226, refined in #251):

        * Always-on detector — :func:`~goldfive.drift.reasoning.detect_looping_reasoning`
          runs first on every call. It catches the byte-identical /
          near-identical repetition pattern that the LLM judge does
          not, and it is cheap. Its drift verdict is handled
          SYNCHRONOUSLY: callers awaiting this method see the resulting
          ``DriftDetected`` sink emission and any refine dispatch
          before control returns.
        * Mode-selected pipeline — :func:`~goldfive.drift.reasoning.analyze_reasoning`
          runs in the configured ``reasoning_drift_mode``. The LLM judge
          path is rate-limited to at most one call every
          ``reasoning_drift_rate_limit`` thinking messages per task; the
          first thinking message of every task always fires a judge
          call. Counters reset on task transition. This path is
          **fire-and-forget**: the judge is scheduled via
          :func:`asyncio.create_task` and tracked on
          ``self._steerer._background_judges`` so :meth:`shutdown` can
          drain it at run end. Its drift verdict may therefore arrive
          AFTER tool calls from the same turn have already dispatched —
          the refine machinery handles "drift arrives mid-run" via
          supersedes, so there is no correctness regression; late
          refines simply apply to a plan state that has already advanced.

        Why the judge path is async: this method is called from the
        adapter's model-response callback, which is on the critical
        path for ADK tool dispatch. Awaiting a minute-long local-llama
        judge round-trip inline serialized every subsequent tool call
        behind it.

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
        from goldfive.drift.reasoning import detect_looping_reasoning

        # Always-on loop detector. A fire short-circuits before the
        # mode-selected pipeline so it remains the canonical signal
        # for "repetitive" reasoning regardless of mode. Cheap, and
        # its verdict can affect the current turn, so it stays inline.
        drift = detect_looping_reasoning(text, session)
        if drift is not None:
            # Populate ``trigger_input`` on drifts produced by the
            # always-on detector (it does not set it itself — it is
            # framework-agnostic).
            if not drift.trigger_input:
                drift.trigger_input = self._truncate_trigger_input(text)
            await self.handle_drift(drift, session)
            return

        # Mode-selected pipeline (judge / embedding / both / off).
        # The judge path is rate-limited per-(agent, task) bucket.
        # Historically this awaited ``analyze_reasoning`` inline; as of
        # goldfive#251 the judge is fire-and-forget so the
        # model-response callback can return immediately.
        rl_call_llm = self._maybe_take_reasoning_judge_slot(session, agent_name=agent_name)
        # Fast-exit when there's nothing for ``analyze_reasoning`` to
        # do: ``mode="off"``, or ``mode="judge"`` with no judge slot
        # (rate-limited or globally disabled). Embedding and "both"
        # modes always schedule — their embedding pipeline runs even
        # when the judge slot is empty.
        if self._steerer._reasoning_drift_mode == "off":
            return
        if self._steerer._reasoning_drift_mode == "judge" and rl_call_llm is None:
            return
        # Thread the first bound sink into the judge path so a
        # ``ReasoningJudgeInvoked`` event fires on every judge call,
        # regardless of verdict. ``None`` when no sinks are bound —
        # the classifier then stays sink-less and behaves as before.
        judge_sink = self._steerer._sinks[0] if self._steerer._sinks else None
        # Snapshot the reasoning-history position at schedule time so
        # the bg pipeline sees the same view the inline pattern
        # detectors just saw, even if subsequent turns append more
        # entries before the bg task runs. Without this, a detector
        # that slices ``history[-N:-1]`` (expecting ``text`` to be the
        # last entry) would see ``text`` itself in the comparison
        # window and trivially self-match (goldfive#251 ordering
        # regression surfaced by the cluster-tightening one-shot
        # test). ``history_length`` is the length AFTER ``text`` was
        # appended — the bg path trims ``session.reasoning_history``
        # to this length for its invocation.
        history_length = len(session.reasoning_history)
        bg_task = asyncio.create_task(
            self._run_judge_background(
                text=text,
                session=session,
                call_llm=rl_call_llm,
                judge_sink=judge_sink,
                history_length=history_length,
                agent_name=agent_name,
            ),
            # goldfive#243: encode session.id in the task name so
            # :meth:`drain_session_background_tasks` can filter pending
            # tasks by the run boundary that's terminating, leaving any
            # other concurrent session's tasks alone.
            name=f"goldfive-reasoning-judge:{session.id}",
        )
        self._steerer._background_judges.add(bg_task)
        bg_task.add_done_callback(self._steerer._background_judges.discard)

    async def _run_judge_background(
        self,
        *,
        text: str,
        session: Session,
        call_llm: ReflectiveCallLLM | None,
        judge_sink: Any,
        history_length: int,
        agent_name: str = "",
    ) -> None:
        """Run the mode-selected reasoning drift pipeline off the critical path.

        Scheduled by :meth:`observe_reasoning` as an
        :func:`asyncio.create_task` so the adapter's model-response
        callback can return before ADK dispatches the response's tool
        calls. Awaits :func:`~goldfive.drift.reasoning.analyze_reasoning`
        and, if it yields a :class:`DriftEvent`, routes it through
        :meth:`_handle_drift` — same effect as the historical inline
        path, just resolving later.

        ``history_length`` pins the ``session.reasoning_history`` view
        the pipeline sees to the same tail index that was in effect
        when the bg task was scheduled. Later turns that append to
        the shared history (this same session receiving more
        reasoning blocks before the bg task runs) would otherwise
        shift the detectors' "exclude self" slice and generate false
        self-match LOOPING signals. We temporarily truncate the
        session view for the duration of this bg invocation and
        restore it after; concurrent bg tasks serialize on the same
        session's reasoning_history via the asyncio event loop (no
        threading) so the save/restore pattern is safe in practice.

        Never raises: any exception (from the judge LLM, the embedding
        pipeline, or ``_handle_drift``) is logged at ``WARNING`` and
        swallowed. The background task must not crash the run; the
        adapter callback that scheduled us has long since returned.
        """
        try:
            from goldfive.drift.reasoning import analyze_reasoning_with_focus

            # Save the shared live history and swap in a list snapshot
            # truncated to the length captured at schedule time. Using
            # list slicing (not mutation) keeps any already-escaped
            # reference (e.g. a concurrent detector) pointing at the
            # original list. We restore the live reference in a
            # ``finally`` so intervening appends are not lost.
            original_history = session.reasoning_history
            pinned_history = list(original_history[:history_length])
            session.reasoning_history = pinned_history
            try:
                # Phase 1 of goldfive#271 — call the focused-verdict
                # path so we get the judge's plan-task attribution
                # alongside the drift signal. ``analyze_reasoning_with_focus``
                # is a sibling of ``analyze_reasoning`` that threads a
                # :class:`ReasoningJudgeVerdict` instead of just the
                # drift; legacy callers of ``analyze_reasoning`` keep
                # their existing return shape.
                #
                # goldfive#244 — also forward the wrapped agent tree so
                # the judge can recognise legitimate coordinator → sub-
                # agent delegation as ON-TASK rather than OFF_TOPIC.
                # Reuses the same shape the planner already consumes
                # (``ADKAdapter.available_agents_tree``); legacy adapters
                # without the property fall back to a flat
                # ``available_agents`` list, and adapters with neither
                # leave ``available_agents=None`` — the judge prompt
                # then renders byte-identically to pre-#244.
                judge_available_agents = self._resolve_available_agents()
                verdict = await analyze_reasoning_with_focus(
                    text,
                    session,
                    mode=self._steerer._reasoning_drift_mode,
                    call_llm=call_llm,
                    model=self._steerer._reasoning_drift_model,
                    sink=judge_sink,
                    agent_name=agent_name,
                    available_agents=judge_available_agents,
                )
            finally:
                # Restore the live history. Any entries appended by
                # subsequent turns are preserved because we pointed
                # ``session.reasoning_history`` at a separate list for
                # our window.
                session.reasoning_history = original_history

            # Record the reasoning-extracted binding onto the
            # orchestration store regardless of the drift verdict —
            # an on-task verdict that names a different plan task is
            # itself a useful pin-resolution signal (the agent has
            # silently moved to a different task without reporting).
            self._maybe_record_reasoning_binding(
                session=session,
                verdict=verdict,
                agent_name=agent_name,
            )

            drift = verdict.drift
            if drift is None:
                return
            if not drift.trigger_input:
                drift.trigger_input = self._truncate_trigger_input(text)
            # Late-drift tolerance (goldfive#319). The judge is
            # fire-and-forget so its verdict can land after the
            # invocation that produced the reasoning has already
            # terminated — adk-web outer-turn boundary crossed, agent
            # moved on. Routing such a verdict through the cancel +
            # ladder dispatch would either cancel an unrelated next
            # invocation or refine against a plan whose offending step
            # is already complete. We still want the drift on the wire
            # for observability ("from past turn"), so we emit it
            # directly via :meth:`_emit_drift_detected` and skip the
            # rest of the dispatch. The guard is scoped to the
            # background-judge path because only that path produces
            # verdicts that may outlive the originating invocation —
            # synchronous detectors run inline on the model-response
            # callback and always see a live invocation.
            if self._is_late_drift_for_terminated_invocation(drift, session):
                log.info(
                    "DefaultSteerer: stale judge verdict; invocation for "
                    "agent=%r task=%r already terminated; drift kind=%s "
                    "recorded but refine skipped",
                    drift.current_agent_id or "-",
                    drift.current_task_id or "-",
                    drift.kind.value,
                )
                if not drift.authored_by:
                    drift.authored_by = self._resolve_authored_by(drift)
                await self._emit_drift_detected(session, drift)
                return
            await self.handle_drift(drift, session)
        except asyncio.CancelledError:
            # Propagate cancellation so :meth:`shutdown` / event-loop
            # teardown can cleanly abort a still-running judge without
            # the WARNING log below muddying the signal.
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            log.warning(
                "DefaultSteerer: background reasoning-judge raised (swallowed): %s",
                exc,
            )

    def _maybe_record_reasoning_binding(
        self,
        *,
        session: Session,
        verdict: Any,
        agent_name: str,
    ) -> None:
        """Stamp a reasoning-extracted binding onto the StateStore.

        Phase 1 of goldfive#271. Called from
        :meth:`_run_judge_background` after the LLM judge returns its
        :class:`~goldfive.drift.reasoning_judge.ReasoningJudgeVerdict`.
        Records a binding when:

        * the verdict carries a non-empty ``focused_task_id``,
        * the agent name is non-empty (we key bindings by agent),
        * ``focus_confidence`` is at least the configured threshold.

        Lower-confidence verdicts are silently dropped so the pin
        ladder doesn't consume noisy bindings. Failures inside the
        store helper degrade silently — the judge's primary job is
        the drift signal, not the binding.
        """
        if not agent_name:
            return
        focused = getattr(verdict, "focused_task_id", "")
        if not focused:
            return
        confidence = float(getattr(verdict, "focus_confidence", 0.0) or 0.0)
        threshold = self._steerer._reasoning_binding_confidence_threshold
        if confidence < threshold:
            log.debug(
                "DefaultSteerer: reasoning binding for agent=%r "
                "task=%r dropped (confidence=%.2f < threshold=%.2f)",
                agent_name,
                focused,
                confidence,
                threshold,
            )
            return
        try:
            from goldfive.state_store import StateStore

            store = StateStore.for_session(session)
            recorded = store.record_reasoning_extracted_binding(
                agent_name=agent_name,
                task_id=focused,
                confidence=confidence,
                recorded_at_turn=session.next_sequence(),
                run_id=session.run_id,
                session_id=session.id,
            )
            if recorded is not None:
                log.info(
                    "DefaultSteerer: recorded reasoning-extracted binding "
                    "agent=%r task=%r confidence=%.2f",
                    agent_name,
                    focused,
                    confidence,
                )
        except Exception as exc:  # noqa: BLE001 — never break the run
            log.warning(
                "DefaultSteerer: record_reasoning_extracted_binding raised (swallowed): %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Background-task lifecycle (drain + shutdown)
    # ------------------------------------------------------------------

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Drain background reasoning-judge + drift tasks with a bounded wait.

        Called at run / runner teardown so ``asyncio.create_task``
        handles scheduled by :meth:`observe_reasoning` (judges) and
        :meth:`mark_task_failed` / :meth:`mark_task_blocked` (drift
        cascades, iter-11A) do not leak beyond the event loop's
        lifetime. Waits at most ``timeout`` seconds (default 5.0) for
        all tracked tasks to finish; any still-running tasks past the
        timeout are cancelled and awaited briefly so their
        ``CancelledError`` propagation settles before we return.

        Idempotent: a second call when both tracking sets are empty
        is a no-op.
        """
        # Drain reasoning-judge tasks.
        if self._steerer._background_judges:
            await self._drain_background_set(
                self._steerer._background_judges, label="judge", timeout=timeout
            )
        # Drain drift-handler tasks (iter-11A).
        if self._steerer._background_drifts:
            await self._drain_background_set(
                self._steerer._background_drifts, label="drift", timeout=timeout
            )

    async def drain_session_background_tasks(
        self, *, session_id: str, timeout: float = 2.0
    ) -> None:
        """Drain background drift / judge tasks for a single session at run end.

        Goldfive#243. The pre-existing drain in :meth:`shutdown` only
        fires from :meth:`Runner.close`, which on long-running adk-web
        / shared-Runner deployments is invoked at process shutdown,
        NOT between user turns. A drift cascade dispatched at the end
        of turn N (e.g. a JUSTIFIED_DEVIATION refine triggered from a
        ``report_*`` tool) outlives turn N's ``RunAborted`` /
        ``RunCompleted`` and runs against an abandoned session — burning
        compute on retry-buried HTTP attempts and emitting spurious
        post-abort drifts (the brussels-sprouts e2e leaked ~10 minutes
        of compute and produced a HUMAN_INTERVENTION_REQUIRED on a
        long-dead session).

        Executors call this right before each terminal
        ``run_aborted_event`` / ``run_completed_event`` emission so the
        symmetry the iter-11A docstring already claims ("drained at run
        end") actually holds at run boundaries, not just process
        teardown. Same bounded-wait + cancel-stragglers semantics as
        :meth:`shutdown`; idempotent (second call shortly after the
        first is a no-op because the tracking sets are empty).

        Filtering: each background task is named
        ``goldfive-<kind>:<session_id>`` (see :meth:`_spawn_*_background`)
        so this method drains ONLY the tasks belonging to the run that
        is terminating, leaving any other concurrent session's tasks
        alone. Tasks predating goldfive#243 (or future spawns that
        forget the suffix) fall back to a session-prefix-aware match;
        if the session_id is the empty string we drain nothing and
        warn — that signals a caller bug rather than legitimate work.

        User-authored drifts (``USER_STEER`` / ``USER_CANCEL`` /
        ``USER_PAUSE``) are dispatched through :meth:`_handle_drift`
        synchronously from :meth:`observe`, so they never land on
        ``_background_drifts`` and are therefore not affected by this
        drain — operator intent survives across turns by construction.
        """
        if not session_id:
            log.warning(
                "DefaultSteerer.drain_session_background_tasks: empty "
                "session_id; refusing to drain (would otherwise match "
                "every pending background task)",
            )
            return
        suffix = f":{session_id}"
        drift_subset = {
            t for t in self._steerer._background_drifts if t.get_name().endswith(suffix)
        }
        judge_subset = {
            t for t in self._steerer._background_judges if t.get_name().endswith(suffix)
        }
        if drift_subset:
            await self._drain_background_set(
                drift_subset, label="drift", timeout=timeout
            )
        if judge_subset:
            await self._drain_background_set(
                judge_subset, label="judge", timeout=timeout
            )

    async def _drain_background_set(
        self,
        bg_set: set[asyncio.Task[Any]],
        *,
        label: str,
        timeout: float,
    ) -> None:
        """Bounded-wait drain for a background-task tracking set.

        Shared between :attr:`_background_judges` and
        :attr:`_background_drifts` (iter-11A). ``label`` is used in
        log messages only.
        """
        # Snapshot: tasks may be removed from the set by their
        # done-callbacks while we're iterating.
        pending = list(bg_set)
        if not pending:
            return
        # goldfive#266 — tag every pending task BEFORE the bounded wait
        # so the in-flight ``goldfive_llm_span`` context manager (e.g.
        # inside a ``judge_reasoning`` call) can read the marker off
        # ``asyncio.current_task()`` on its CancelledError path and emit
        # ``status="cancelled"`` with a benign reason rather than
        # ``status="failed"`` with a CancelledError traceback. We stamp
        # upfront — not just on the post-timeout straggler-cancel branch
        # — because :func:`asyncio.wait_for` itself cancels its inner
        # awaitable on timeout, and that cancellation can race ahead of
        # the explicit ``task.cancel()`` below; tagging before the wait
        # makes the marker race-free. Tasks that complete naturally
        # within the timeout never observe a CancelledError so the
        # attribute is harmless on them. The cancel BEHAVIOUR is
        # unchanged; only the span-status observability is corrected so
        # live sessions stop showing red judge spans for routine
        # run-boundary teardowns.
        from goldfive._llm_span import DRAIN_INITIATED_ATTR

        for task in pending:
            try:
                setattr(task, DRAIN_INITIATED_ATTR, True)
            except (AttributeError, TypeError):  # noqa: PERF203 — defensive
                # asyncio.Task allows arbitrary attribute writes in
                # CPython today; the try/except is purely defensive
                # against future hardening / alternate implementations.
                pass
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=max(0.0, float(timeout)),
            )
        except TimeoutError:
            # Cancel the stragglers and give them a beat to unwind so
            # we don't leave "pending task" warnings on loop close. The
            # drain-initiated marker is already on each task from the
            # tagging loop above.
            still_pending = [t for t in pending if not t.done()]
            for task in still_pending:
                task.cancel()
            if still_pending:
                try:
                    await asyncio.wait(still_pending, timeout=0.5)
                except Exception:  # noqa: BLE001 — defensive
                    pass
            # Phase 2.X (goldfive#271 Gap 3): only WARN when stragglers
            # were actually cancelled. The TimeoutError can fire even
            # when every task completed in the same instant the timeout
            # expired (gather scheduling vs. wait_for race) — those
            # cases logged ``cancelled 0 tasks`` which was both
            # confusing and noisy in the demo log. The DEBUG line
            # preserves visibility for diagnostics while keeping INFO
            # / WARNING reserved for the real "we cancelled work"
            # signal.
            if still_pending:
                log.warning(
                    "DefaultSteerer.shutdown: %d background %s task(s) "
                    "exceeded %.2fs timeout; cancelled",
                    len(still_pending),
                    label,
                    float(timeout),
                )
            else:
                log.debug(
                    "DefaultSteerer.shutdown: %.2fs timeout expired but "
                    "all %s tasks completed in the same instant; nothing "
                    "to cancel",
                    float(timeout),
                    label,
                )

    def _resolve_available_agents(self) -> list[str] | list[dict[str, Any]] | None:
        """Return the wrapped agent tree for a downstream prompt.

        Mirrors the resolution used by ``_handle_drift`` when threading
        ``available_agents`` into ``planner.refine``: prefer the
        structured ``ADKAdapter.available_agents_tree`` (goldfive#151,
        list of dicts with name/parent/role/kind/depth); fall back to
        the flat ``available_agents`` (list[str]) when the structured
        property is missing or empty; return ``None`` when the adapter
        is missing or exposes neither surface.

        Used by :meth:`_run_judge_background` to feed the reasoning
        judge's :data:`~goldfive.drift.reasoning_judge.AGENT_TREE_BLOCK_MAX_CHARS`
        bounded "AGENT TREE" prompt section (goldfive#244) so the judge
        can recognise legitimate coordinator → sub-agent delegation as
        ON-TASK rather than OFF_TOPIC. ``None`` keeps the judge's
        prompt byte-identical to the pre-#244 shape.
        """
        adapter = self._steerer._adapter
        if adapter is None:
            return None
        tree = getattr(adapter, "available_agents_tree", None)
        if isinstance(tree, list) and tree:
            return list(tree)
        flat = getattr(adapter, "available_agents", None)
        if flat:
            return list(flat)
        return None

    def _maybe_take_reasoning_judge_slot(
        self,
        session: Session,
        *,
        agent_name: str = "",
    ) -> ReflectiveCallLLM | None:
        """Return the judge ``call_llm`` when this turn is a judge turn.

        Rate-limit policy (goldfive#226):

        * First thinking message of every (agent, task) bucket always fires.
        * Subsequent messages skip ``(N-1)`` and then fire on the Nth.
        * Counters are scoped per-(agent, task) via
          ``session._reasoning_judge_counters`` so a task transition
          OR an agent switch resets the window lazily -- the next
          ``(agent, task_id)`` tuple is simply not in the dict yet, so
          its first message falls into the "count=0" branch.

        Pre-fix the key was a single string keyed on
        ``current_task_id or ""``. Every unpinned turn from every agent
        collapsed onto the ``""`` bucket, so agent B's first thinking
        block could legitimately skip the judge because unrelated
        agent A's unpinned turn had already incremented the counter.
        Bucketing by ``(agent_name, task_id)`` isolates each agent's
        cadence.

        Returns ``None`` when the judge is globally disabled (mode
        skips it, or ``reasoning_drift_call_llm`` is unconfigured).
        Also ``None`` on skip turns even when armed.
        """
        if self._steerer._reasoning_drift_call_llm is None:
            return None
        if self._steerer._reasoning_drift_mode not in ("judge", "both"):
            return None
        task_id = session.current_task_id or ""
        key = (agent_name or "", task_id)
        counters = session._reasoning_judge_counters
        count = counters.get(key, 0)
        # count=0 -> fire (first message on this (agent, task) bucket),
        # reset to 1. Otherwise fire when count % rate_limit == 0.
        fire = (count % self._steerer._reasoning_drift_rate_limit) == 0
        counters[key] = count + 1
        return self._steerer._reasoning_drift_call_llm if fire else None

    # ------------------------------------------------------------------
    # Reflective self-progress check (opt-in)
    # ------------------------------------------------------------------

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
        if self._steerer._reflective_call_llm is None:
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
        if session._llm_calls_since_check < self._steerer._reflective_check_interval:
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
        call_llm = self._steerer._reflective_call_llm
        if call_llm is None or session.plan is None:
            return
        task = self._steerer._find_task(session, session.current_task_id)
        if task is None:
            # No task to assess. Nothing useful to ask the model.
            return
        tool_call_summary = self._summarize_recent_tool_calls(session)
        reasoning_summary = self._summarize_recent_reasoning(session)
        user_prompt = self.REFLECTIVE_USER_PROMPT_TEMPLATE.format(
            task_id=task.id,
            task_title=task.title or "",
            task_description=task.description or "",
            window=self._steerer._reflective_check_interval,
            tool_call_summary=tool_call_summary,
            reasoning_summary=reasoning_summary,
        )
        from goldfive._llm_span import goldfive_llm_span

        # ``reflective_check`` targets a specific task / agent, so stamp
        # the driver agent + task onto the span and feed a composed
        # input_preview (tool calls + reasoning window) so operators can
        # answer "what did the reflective check see?" from the Gantt.
        reflective_input_preview = (
            f"task={task.id} ({task.title or ''})\n"
            f"tool_calls:\n{tool_call_summary}\n\n"
            f"reasoning:\n{reasoning_summary}"
        )
        parsed: dict[str, Any] | None = None
        try:
            async with goldfive_llm_span(
                sinks=self._steerer._sinks,
                name="reflective_check",
                model=self._steerer._reflective_model,
                session_id=session.id,
                run_id=session.run_id,
                task_id=task.id,
                sequence_fn=session.next_sequence,
                input_preview=reflective_input_preview,
                target_agent_id=task.assignee_agent_id or "",
                target_task_id=task.id,
            ) as span:
                # Bound the dispatch — see ``REFLECTIVE_MAX_OUTPUT_TOKENS``.
                # Also disable thinking (goldfive#271 follow-up to #311):
                # this is meta-cognition asking the agent if it's making
                # progress, not deep reasoning.
                from goldfive._llm import (
                    call_llm_budget,
                    call_llm_thinking_disabled,
                )

                with (
                    call_llm_budget(self.REFLECTIVE_MAX_OUTPUT_TOKENS),
                    call_llm_thinking_disabled(),
                ):
                    raw = await call_llm(
                        self.REFLECTIVE_SYSTEM_PROMPT,
                        user_prompt,
                        self._steerer._reflective_model,
                    )
                parsed = self._parse_reflective_response(raw)
                if parsed is None:
                    # Distinguish "model returned all thinking, no
                    # answer" from "model returned garbage" — see
                    # goldfive#271 follow-up to #311. ``call_llm`` is
                    # the closure built by ``make_default_adk_call_llm``
                    # / ``_build_judge_call_llm`` which stashes part
                    # counts on itself.
                    _thought_n = int(getattr(call_llm, "last_thought_count", 0) or 0)
                    _raw_str = raw if isinstance(raw, str) else ""
                    if not _raw_str.strip() and _thought_n > 0:
                        span.output_preview = (
                            f"empty answer ({_thought_n} thought "
                            f"part(s); the model spent its budget thinking "
                            f"and emitted no JSON)"
                        )
                    else:
                        span.output_preview = f"unparseable verdict; raw={raw!r:.200}"
                    span.decision_summary = f"reflective check on {task.id}: unparseable verdict"
                else:
                    making_progress_inline = parsed.get("making_progress")
                    conf_inline = parsed.get("confidence")
                    reason_inline = str(parsed.get("reason", "") or "")
                    span.output_preview = (
                        f"making_progress={making_progress_inline}, "
                        f"confidence={conf_inline}, "
                        f"reason={reason_inline or '(none)'}"
                    )
                    if isinstance(making_progress_inline, bool):
                        verdict_str = "progressing" if making_progress_inline else "stuck"
                    else:
                        verdict_str = "malformed"
                    span.decision_summary = f"reflective check on {task.id}: {verdict_str}"
        except Exception as exc:  # noqa: BLE001 - never break the run
            log.warning(
                "DefaultSteerer.maybe_run_reflective_check: call_llm raised %s", exc
            )
            await self._emit_reflective_failure(
                session,
                task_id=task.id,
                reason=f"reflective call_llm raised: {exc}",
            )
            return
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
        # Prefer the runtime-reasoning agent pin (set by the ADK
        # plugin's ``before_agent_callback``) over the static plan
        # assignee — when a coordinator delegates to a child the
        # child's reasoning produced this drift, not the assignee's.
        # Fall back to ``task.assignee_agent_id`` when the session
        # pin is empty (pre-pin race or non-ADK adapter that doesn't
        # populate it) so we keep back-compat.
        agent_id_for_drift = session.current_agent_id or task.assignee_agent_id
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
                current_agent_id=agent_id_for_drift,
            )
            await self.handle_drift(drift, session)
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
                current_agent_id=agent_id_for_drift,
            )
            await self.handle_drift(drift, session)
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
        overflow = len(hist) - self._steerer._goal_drift_activity_window
        if overflow > 0:
            del hist[:overflow]

    def note_tool_observation(
        self,
        session: Session,
        *,
        agent_name: str,
        task_id: str,
        tool_name: str,
        args: Any,
        result: Any,
        error: Exception | str | None = None,
    ) -> None:
        """Append a bounded tool-observation entry to ``session.recent_tool_observations``.

        Iter-10 PR 2. Population path for the three-state reasoning
        judge (PR 3 reads this buffer to distinguish a provoked
        deviation from an unprovoked one). Adapters call this from
        their ``after_tool_callback`` (success + acknowledged-failure)
        and ``on_tool_error_callback`` hooks.

        Push-only and trim-on-write — mirrors
        :meth:`note_agent_activity`. The buffer is bounded by
        ``session.recent_tool_observations_max`` (default 16) so the
        prompt the judge eventually reads stays small regardless of
        run length. Per-task filtering happens at READ time in the
        judge's prompt renderer; this writer captures every call.

        Always swallow internal errors. Observability must never break
        tool dispatch — a malformed ``args`` / ``result`` repr, a
        broken clock, or a pathological session must not raise out of
        an ADK callback. The catch is intentionally broad.
        """
        try:
            ts_ms = time.monotonic_ns() // 1_000_000
            try:
                args_preview = repr(args)[:240]
            except Exception:  # noqa: BLE001
                args_preview = "(unrepresentable args)"
            if result is None:
                result_preview = "(none)"
            else:
                try:
                    result_preview = repr(result)[:480]
                except Exception:  # noqa: BLE001
                    result_preview = "(unrepresentable result)"
            # Error detection: an explicit ``error=`` from the caller
            # (the on_tool_error path) wins; otherwise look for the
            # acknowledged-failure shape ``{"error": ...}`` in the
            # tool result. The reporting tools and most goldfive
            # tools return that shape on a soft failure.
            is_error = False
            error_message = ""
            if error is not None:
                is_error = True
                try:
                    error_message = str(error)[:240]
                except Exception:  # noqa: BLE001
                    error_message = "(unrepresentable error)"
            elif isinstance(result, dict) and "error" in result:
                is_error = True
                try:
                    error_message = str(result.get("error", ""))[:240]
                except Exception:  # noqa: BLE001
                    error_message = "(unrepresentable error)"
            entry: dict[str, Any] = {
                "ts_ms": ts_ms,
                "agent_name": agent_name,
                "task_id": task_id,
                "tool_name": tool_name,
                "args_preview": args_preview,
                "result_preview": result_preview,
                "is_error": is_error,
                "error_message": error_message,
            }
            hist = session.recent_tool_observations
            hist.append(entry)
            # Cap defaults to 16 (§3.1) but honour any session-local
            # override; clamp to >=1 so a pathological 0 / negative
            # value doesn't disable the buffer entirely (we always
            # want at least the most-recent entry).
            try:
                cap_raw = int(session.recent_tool_observations_max)
            except (TypeError, ValueError):
                cap_raw = 16
            cap = max(1, cap_raw)
            overflow = len(hist) - cap
            if overflow > 0:
                # Slice-delete is amortized O(1) on average for the
                # bounded ``overflow == 1`` case (the steady state once
                # the buffer is full), and is the same pattern
                # ``note_agent_activity`` uses.
                del hist[:overflow]
        except Exception as exc:  # noqa: BLE001
            log.debug("note_tool_observation: swallowed: %s", exc)

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

        Spawn-and-detach (goldfive v22 regression fix). The judge is
        dispatched as a fire-and-forget background task — see the
        rationale on :meth:`_maybe_run_goal_drift_on_task_boundary`.
        ``after_run_callback`` runs on the agent's invocation task,
        which is the same cancellable scope a sibling drift can target
        via :meth:`request_invocation_cancel`; an inline await on the
        judge would die the same way the v22 ``judge_goal_drift`` span
        did. Tests that drove the inline path can drain via
        ``await asyncio.gather(*list(steerer._background_judges))``.
        """
        if self._steerer._goal_drift_call_llm is None:
            return
        session._agent_turns_since_goal_check += 1
        if session._agent_turns_since_goal_check < self._steerer._goal_drift_check_interval:
            return
        # Reset before running so a check that itself triggers further
        # invocations in the agent loop doesn't double-fire.
        session._agent_turns_since_goal_check = 0
        self._spawn_goal_drift_judge_background(session)

    async def _maybe_run_goal_drift_on_task_boundary(self, session: Session) -> None:
        """Fire :meth:`maybe_run_goal_drift_check` on a task transition.

        Task completions / failures / cancellations are natural
        "am I still on plan?" checkpoints, so we fire the judge here
        in addition to the turn-counter-driven path (goldfive#219).
        Short pipelines that finish before ``goal_drift_check_interval``
        turns would otherwise never trigger the judge.

        Rate-limited: if two task transitions happen within
        :data:`_GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` seconds of
        each other, only the first fires a judge call. Callers pass
        a fresh ``time.time()`` implicitly via the session-stored
        ``_last_goal_drift_check_ts``.

        Also resets ``session._agent_turns_since_goal_check`` so a
        task boundary that lands on exactly the interval boundary
        does not pay for two back-to-back judge calls.

        No-ops when ``goal_drift_call_llm`` is unconfigured — that
        gate is enforced inside :meth:`maybe_run_goal_drift_check`;
        we short-circuit here only to avoid bumping the timestamp
        when no judge will run.

        Spawn-and-detach (goldfive v22 regression fix). The judge LLM
        call is dispatched as a fire-and-forget background task on
        :attr:`_background_judges` rather than awaited inline. The
        ``mark_task_*`` callers run on the agent's invocation task —
        which is registered with the ADK plugin's ``_invocation_tasks``
        for cooperative cancel — so a sibling cancel (supersede,
        runaway delegation, refine-driven preempt) firing
        ``task.cancel()`` on the agent's invocation task while the
        inline judge was awaiting its LLM round-trip would surface a
        ``CancelledError`` inside ``classify_goal_drift``. The
        ``judge_goal_drift`` span ended with ``error=CancelledError``
        and an empty stack, the verdict was lost, and operator-visible
        evidence (v22 trace) showed the cancel landing the moment the
        span opened. Detaching the judge from the cancellable task
        scope — same pattern as the reasoning judge at
        :meth:`_run_judge_background` — keeps it alive across cancel
        propagation and drainable at :meth:`shutdown`.
        """
        if self._steerer._goal_drift_call_llm is None:
            return
        now = time.time()
        last = getattr(session, "_last_goal_drift_check_ts", 0.0)
        if now - last < self._GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S:
            return
        session._last_goal_drift_check_ts = now
        # Reset the turn counter so the next turn-interval check starts
        # fresh rather than firing one more judge call on the next turn.
        session._agent_turns_since_goal_check = 0
        self._spawn_goal_drift_judge_background(session)

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
        call_llm = self._steerer._goal_drift_call_llm
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
            model=self._steerer._goal_drift_model,
            call_llm=call_llm,
            current_task_id=session.current_task_id,
            sinks=self._steerer._sinks,
            run_id=session.run_id,
            session_id=session.id,
            sequence_fn=session.next_sequence,
            # goldfive#245 — pass the live session so the judge can
            # post-LLM re-read ``session.plan`` after its await and
            # drop verdicts whose target task transitioned during the
            # round-trip.
            session=session,
        )
        if drift is None:
            return
        await self.handle_drift(drift, session)

    def _spawn_goal_drift_judge_background(self, session: Session) -> None:
        """Spawn :meth:`maybe_run_goal_drift_check` as a fire-and-forget task.

        Goldfive v22 regression fix. The trajectory-level GOAL_DRIFT
        judge used to be awaited inline from
        :meth:`_maybe_run_goal_drift_on_task_boundary` (called from
        ``mark_task_*``) and from :meth:`note_agent_turn` (called from
        the ADK plugin's ``after_run_callback``). Both call sites run
        on the agent's ADK invocation task, which is registered with
        ``_GoldfiveADKPlugin._invocation_tasks`` for cooperative
        cancellation. A sibling drift firing
        :meth:`request_invocation_cancel(cancel_inflight_task=True)`
        could therefore land a ``CancelledError`` inside the judge's
        own ``await call_llm(...)`` — the v22 trace
        (49b0eb10-5636-465d-b96b-9e9d03d91e81) shows exactly that:
        immediately after research_panels transitioned to COMPLETED
        the ``judge_goal_drift`` span opened and failed with
        ``CancelledError`` and an empty stack, no LLM duration
        recorded.

        Detaching the judge into a separate ``asyncio.Task`` isolates
        it from the agent invocation's cancel scope: ``task.cancel()``
        on the agent's task does NOT propagate to children spawned via
        :func:`asyncio.create_task` (asyncio Tasks do not form a
        parent-child cancel tree the way ``asyncio.TaskGroup`` does).
        The judge is tracked on :attr:`_background_judges` so
        :meth:`shutdown` (called from ``Runner.close``) can drain it
        with the same bounded wait the reasoning judge uses
        (goldfive#251). Done-callback removes the entry on completion
        so there is no per-turn leak.

        No-op when no event loop is running (defensive — keeps
        synchronous test harnesses that build a steerer outside an
        async context from raising). No-op when no judge ``call_llm``
        is configured.
        """
        if self._steerer._goal_drift_call_llm is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop — fall through silently. The synchronous callers
            # of ``mark_task_*`` outside an async context (rare; only
            # tests / synthetic harnesses) won't get a goal-drift
            # check, but they wouldn't have anywhere to await the
            # judge anyway.
            return
        bg_task = asyncio.create_task(
            self._run_goal_drift_judge_background(session),
            # goldfive#243: encode session.id in the task name so
            # :meth:`drain_session_background_tasks` can filter pending
            # tasks by the run boundary that's terminating, leaving any
            # other concurrent session's tasks alone.
            name=f"goldfive-goal-drift-judge:{session.id}",
        )
        self._steerer._background_judges.add(bg_task)
        bg_task.add_done_callback(self._steerer._background_judges.discard)

    async def _run_goal_drift_judge_background(self, session: Session) -> None:
        """Body of the fire-and-forget GOAL_DRIFT judge task.

        Mirrors :meth:`_run_judge_background` (the reasoning-judge
        equivalent): swallows every exception so a flaky judge cannot
        crash the run, and re-raises ``CancelledError`` cleanly so
        :meth:`shutdown` can cancel still-running judges at teardown
        without a stray ``WARNING`` muddying the signal.

        Calls :meth:`maybe_run_goal_drift_check` directly — the public
        method's synchronous semantics are preserved for operator-side
        one-shot triggers; this background path just bypasses the
        cancellable agent task that hosted us.
        """
        try:
            await self.maybe_run_goal_drift_check(session)
        except asyncio.CancelledError:
            # Propagate so :meth:`shutdown` / teardown sees a clean
            # cancel. The shutdown path expects this and counts it
            # against the still-pending tally without warning.
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            log.warning(
                "DefaultSteerer: background goal-drift judge raised "
                "(swallowed): %s",
                exc,
            )

    # ------------------------------------------------------------------
    # iter-11A: fire-and-forget drift-cascade dispatch.
    #
    # ``mark_task_failed`` / ``mark_task_blocked`` previously awaited
    # ``_handle_drift`` inline. The cascade traverses planner.refine
    # (an LLM round-trip), supersedes integration, and downstream
    # cancellation — on a slow local LLM (e.g. Qwen3.6-35B-A3B-FP8) the
    # full chain can take 60-120s. Awaiting that from the reporting
    # tool blocked the tool's return, which blocked the agent's next
    # ADK turn end-to-end. Spawning the cascade lets the tool ack the
    # transition immediately; the cascade's side effects
    # (PlanRevised emission, supersedes, follow-up nudges) land on the
    # sink bus exactly as before, just slightly later.
    #
    # Mirrors :meth:`_spawn_goal_drift_judge_background`. ``shutdown``
    # drains :attr:`_background_drifts` symmetrically with
    # :attr:`_background_judges`.
    # ------------------------------------------------------------------
    def _spawn_drift_handler_background(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Dispatch :meth:`_handle_drift` off the critical path.

        Mirrors :meth:`_spawn_goal_drift_judge_background`. The
        reporting tool that triggered this drift returns immediately;
        the downstream cascade (refine, supersedes, cancellation)
        happens asynchronously on a tracked task.

        No-op when no event loop is running (defensive — keeps
        synchronous test harnesses that build a steerer outside an
        async context from raising; the inline-awaiting callers of
        ``mark_task_*`` outside an async context are vanishingly rare
        and cannot drive a refine round-trip anyway).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop — fall through silently. Same defensive pattern
            # as :meth:`_spawn_goal_drift_judge_background`.
            log.debug(
                "DefaultSteerer._spawn_drift_handler_background: no running "
                "loop; skipping spawn for kind=%s",
                drift.kind.value,
            )
            return
        bg_task = asyncio.create_task(
            self._run_drift_handler_background(drift, session),
            # goldfive#243: encode session.id in the task name so
            # :meth:`drain_session_background_tasks` can filter pending
            # tasks by the run boundary that's terminating, leaving any
            # other concurrent session's tasks alone.
            name=f"goldfive-drift-{drift.kind.value}:{session.id}",
        )
        self._steerer._background_drifts.add(bg_task)
        bg_task.add_done_callback(self._steerer._background_drifts.discard)

    async def _run_drift_handler_background(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Body of the fire-and-forget drift handler.

        Mirrors :meth:`_run_goal_drift_judge_background`: swallows
        every exception so a flaky cascade cannot crash the run, and
        re-raises ``CancelledError`` cleanly so :meth:`shutdown` can
        cancel still-running cascades at teardown without a stray
        ``WARNING`` muddying the signal.
        """
        try:
            await self.handle_drift(drift, session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            log.warning(
                "DefaultSteerer: background drift handler raised "
                "(swallowed): kind=%s exc=%s",
                drift.kind.value,
                exc,
            )

    async def _wait_background_drifts_idle(self) -> None:
        """Wait for every pending background drift task to settle.

        Test helper. Mirrors the goal-drift drain pattern used by
        :func:`tests.test_goal_drift_classifier._drain_background_judges`.
        Production callers should never need this — the run-end
        :meth:`shutdown` drains pending cascades with a bounded wait.
        """
        pending = list(self._steerer._background_drifts)
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
        # One yield so the ``add_done_callback(...discard)`` has run
        # and the set is fully empty for the next assertion / spawn.
        await asyncio.sleep(0)

    # ==================================================================
    # Bucket 3c — Dispatch + intervention ladder + promotion
    # ==================================================================
    #
    # Methods below were extracted verbatim from
    # :class:`goldfive.steerer.DefaultSteerer` in the bucket-3c step of
    # the steerer split. ``self.X`` references that target router-level
    # state (sinks, planner, adapter, control_channel, plan locks,
    # observation_only flag, REFINE_FAILURE_THRESHOLD /
    # PROGRESS_STALL_THRESHOLD_SECONDS class constants, the
    # ContextVar-isolated active-session plumbing) read via
    # ``self._steerer.X``; ``self.X`` references that target the
    # observability primitives extracted in buckets 3a/3b (drift emit,
    # detect, ladder constants, structural escalation helpers) read
    # directly off this class.
    #
    # The most-fixed of the moved methods is :meth:`handle_drift` (the
    # ex-``_handle_drift``); the line count goes well past terseness
    # norms because the source did. :meth:`_promote_drift_to_steer`
    # contains audit issue #402 (dispatch-before-plan-swap) which is
    # **preserved**, not fixed — the fix has its own queued PR.

    # ------------------------------------------------------------------
    # Intervention ladder table (goldfive#142)
    # ------------------------------------------------------------------
    #
    # See the long-form rationale comments above
    # :attr:`goldfive.steerer.DefaultSteerer._LADDER` (now removed from
    # the router; this is the canonical home post bucket-3c).

    _LADDER: dict[
        DriftKind,
        tuple[
            InterventionLevel | None,
            InterventionLevel | None,
            tuple[InterventionLevel, InterventionLevel],
        ],
    ] = {}  # populated lazily in :meth:`_load_ladder_tables` to avoid an
    # import-cycle with :mod:`goldfive.steerer` (which defines the
    # :class:`InterventionLevel` enum).

    _LADDER_BY_VALUE: dict[
        str,
        tuple[
            InterventionLevel | None,
            InterventionLevel | None,
            tuple[InterventionLevel, InterventionLevel],
        ],
    ] = {}

    _LADDER_LOADED: bool = False

    @classmethod
    def _load_ladder_tables(cls) -> None:
        """Populate :attr:`_LADDER` / :attr:`_LADDER_BY_VALUE` on first use.

        Lazy load defers the :class:`InterventionLevel` import so this
        module can be imported before :mod:`goldfive.steerer` finishes
        its own imports (the steerer constructs a :class:`DriftObserver`
        in its ``__init__``).
        """
        if cls._LADDER_LOADED:
            return
        from goldfive.steerer import InterventionLevel as _IL

        cls._LADDER = {
            DriftKind.CONFABULATION_RISK: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.AGENT_REFUSAL: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.MODEL_REFUSAL: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.LOOPING_REASONING: (
                None,
                _IL.ABSORB,
                (_IL.NUDGE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.LOOPING_TOOL_CALL: (
                None,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.REASONING_CLUSTER_TIGHTENING: (
                _IL.OBSERVE,
                None,
                (_IL.OBSERVE, _IL.OBSERVE),
            ),
            DriftKind.PLAN_DIVERGENCE: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.OFF_TOPIC: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.JUSTIFIED_DEVIATION: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.ABSORB, _IL.ABSORB),
            ),
            DriftKind.INTENT_DIVERGENCE: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.PAUSE_ESCALATE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.TOOL_ERROR: (
                _IL.OBSERVE,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.RUNAWAY_DELEGATION: (
                None,
                None,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.REFINE_VALIDATION_FAILED: (
                None,
                None,
                (_IL.PAUSE_ESCALATE, _IL.PAUSE_ESCALATE),
            ),
            DriftKind.HUMAN_INTERVENTION_REQUIRED: (
                None,
                None,
                (_IL.PAUSE_ESCALATE, _IL.TERMINATE),
            ),
            DriftKind.GOAL_DRIFT: (
                None,
                _IL.NUDGE,
                (_IL.NUDGE, _IL.CANCEL_REINVOKE),
            ),
            DriftKind.SELF_REPORTED_STUCK: (
                None,
                _IL.ABSORB,
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),
            ),
        }
        cls._LADDER_LOADED = True

    def _ladder_level_for(
        self,
        kind: DriftKind,
        severity: DriftSeverity,
        occurrence_count: int,
    ) -> InterventionLevel:
        """Return the intervention level for ``(kind, severity, count)``.

        See :meth:`goldfive.steerer.DefaultSteerer._ladder_level_for` —
        identical semantics; lifted verbatim into bucket 3c so the
        whole drift-dispatch surface has one home.
        """
        from goldfive.steerer import InterventionLevel as _IL

        self._load_ladder_tables()
        entry = self._LADDER_BY_VALUE.get(kind.value)
        if entry is None:
            entry = self._LADDER.get(kind)
        is_repeat = occurrence_count >= self._steerer.REFINE_FAILURE_THRESHOLD
        if entry is not None:
            info_level, warning_level, critical_pair = entry
            if severity is DriftSeverity.INFO:
                return info_level or _IL.OBSERVE
            if severity is DriftSeverity.WARNING:
                return warning_level or _IL.OBSERVE
            return critical_pair[1] if is_repeat else critical_pair[0]
        if severity is DriftSeverity.INFO:
            return _IL.OBSERVE
        if severity is DriftSeverity.WARNING:
            return _IL.ABSORB
        return _IL.PAUSE_ESCALATE if is_repeat else _IL.ABSORB

    # ------------------------------------------------------------------
    # The central drift-routing entry point.
    # ------------------------------------------------------------------

    async def handle_drift(self, drift: DriftEvent, session: Session) -> None:
        """Emit a ``DriftDetected`` event and dispatch via the intervention ladder.

        Verbatim move of :meth:`DefaultSteerer._handle_drift` (the
        single most-fixed method on the codebase after
        :meth:`_emit_plan_revised`). See the original docstring in
        :class:`DefaultSteerer` for the long-form contract; the body
        and behaviour are unchanged here.
        """
        from goldfive.steerer import (
            _ABSORB_NUDGE_KINDS,
            InterventionLevel,
            RefineExhausted,
            _planner_refine_accepts_available_agents,
            compose_corrective_user_message,
        )

        # goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH
        # (#253) — disabled here. Guard at the very top so any external
        # producer (legacy callers, replays, sinks) cannot revive it.
        if drift.kind is DriftKind.PLAN_DIVERGENCE:
            log.debug(
                "DefaultSteerer._handle_drift: PLAN_DIVERGENCE drift "
                "received but handling is disabled (goldfive#252); "
                "detail=%r",
                drift.detail,
            )
            return
        # Normalise source attribution early so every downstream
        # consumer (sinks, promotion policy, prompt framing) sees a
        # non-empty ``authored_by``. USER_* kinds → "user"; anything
        # else → "goldfive". Honours an explicit non-empty value on
        # the drift (e.g. callers that already attributed) via
        # :meth:`_resolve_authored_by`.
        if not drift.authored_by:
            drift.authored_by = self._resolve_authored_by(drift)
        # goldfive#245 — verdict-freshness gate. Every observation/
        # detector stamps ``observed_revision_index`` from
        # ``session.plan.revision_index`` BEFORE its LLM await; the
        # reconciler may transition tasks during that round-trip, so a
        # verdict that arrives after the framework moved on is moot.
        # Drop it here: emit ``DriftDetected`` for observability
        # (operators see the detector ran) and skip the cancel + refine
        # machinery.
        #
        # Bypasses:
        #   * Unstamped drifts (``observed_revision_index == 0``) are
        #     legacy / external producers / pre-#245 emit paths — flow
        #     through unchanged so the gate is purely additive.
        #   * User-authored drifts (USER_STEER / USER_CANCEL /
        #     USER_PAUSE) bypass the gate even when stamped: an
        #     operator directive must be honoured regardless of the
        #     framework's plan-state cursor (preserves the iter-11D /
        #     #242 contract).
        if drift.observed_revision_index and (drift.authored_by or "").lower() != "user":
            # Per-(kind, target) addressed-watermark check — narrower
            # than naive ``observed < live_revision`` gating so parallel
            # judges firing on *orthogonal* concerns aren't over-rejected.
            # Naive gating drops a GOAL_DRIFT verdict observed at N just
            # because an unrelated OFF_TOPIC refine bumped the plan to
            # N+1; this gate drops only when the SAME (kind, target) was
            # already addressed at a later revision (genuinely redundant).
            #
            # ``last_addressed_revision_by_drift_key`` is stamped by
            # :meth:`_apply_revision` after every successful goldfive-
            # authored refine. Empty target (``""``) coalesces trajectory-
            # level drifts on one key, so trajectory-wide addressing
            # works correctly.
            key = (drift.kind.value, drift.current_task_id or "")
            last_addressed = int(
                session.last_addressed_revision_by_drift_key.get(key, 0)
            )
            if last_addressed and drift.observed_revision_index < last_addressed:
                log.info(
                    "DefaultSteerer._handle_drift: redundant verdict — "
                    "drift kind=%s target=%r observed revision %d but "
                    "same (kind, target) was already addressed at "
                    "revision %d; skipping dispatch",
                    drift.kind.value,
                    drift.current_task_id or "<trajectory>",
                    drift.observed_revision_index,
                    last_addressed,
                )
                # Emit for observability so operators see the detector
                # ran; do NOT cancel / refine on a redundant view.
                await self._emit_drift_detected(session, drift)
                return
        # Tag the bound adapter's next cancel with a symbolic reason so
        # the synthetic function_response the adapter appends on cancel
        # carries LLM-actionable content. Done BEFORE the drift event
        # is emitted so a sink that reacts by cancelling the invoke
        # sees the tag. Harmless if the adapter doesn't expose the
        # attribute (duck-typed) or no adapter is bound. See
        # goldfive#139 and
        # :func:`goldfive.adapters.adk._build_cancelled_response_event`.
        self._tag_adapter_cancel_reason(drift, session=session)
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
            # Plumb the session into the planner's span-context provider
            # for the duration of synthesize_goal_from_steer so its LLM
            # call shows up as a span on the Gantt. Cleared in a
            # ``finally`` so exceptions don't leave a stale pointer.
            _token = self._steerer._active_session_var.set(session)
            try:
                await self._apply_user_steer_state(drift, session)
            finally:
                self._steerer._active_session_var.reset(_token)
        # goldfive-steer-unification: consult the severity-aware
        # promotion policy BEFORE emitting DriftDetected so that a
        # suppressed goldfive steer carries the ``suppressed_by_user_steer``
        # flag on the wire (sinks can surface the suppression
        # decision). ``_should_promote_to_steer`` returns ``True`` iff
        # the drift is goldfive-authored, clears the configured
        # severity threshold, and is not blocked by an active fresh
        # user steer; as a side effect it stamps
        # ``drift.suppressed_by_user_steer=True`` when the suppression
        # path wins.
        promote_to_steer = self._should_promote_to_steer(drift, session)
        await self._emit_drift_detected(session, drift)
        if drift.suppressed_by_user_steer:
            # Suppression path: the goldfive drift fired, was observed
            # via DriftDetected, and — per the fresh user-steer
            # suppression window — we neither cancel nor refine. The
            # pre-unification passive ladder dispatch is also skipped:
            # a user steer is already active, its refine has already
            # happened, and running another refine for this signal
            # would race against it.
            return
        # Cooperative cancellation (goldfive#251 Stream C / 7a). Severity
        # ladder decision: CRITICAL drifts (and ONLY critical drifts)
        # flag the currently-active invocation(s) for cooperative
        # cancel before the refine / promote path runs. INFO + WARNING
        # severities do NOT cancel — they flow through the usual
        # observe / absorb / nudge channels. User-authored drifts
        # (USER_STEER / USER_CANCEL / USER_PAUSE) additionally bypass
        # the severity gate because an operator directive must be
        # honoured even when emitted at a lower severity tier.
        #
        # The actual short-circuit happens in the ADK plugin's next
        # ``before_agent_callback`` / ``before_model_callback`` /
        # ``before_tool_callback``; this call just writes the flag.
        # Whether to re-dispatch after the cancel is the parent
        # agent's decision, informed by plan-causal prompting from
        # Stream B — the framework itself does NOT auto-reinvoke.
        if self._should_request_cancel_for_drift(drift):
            try:
                await self.request_invocation_cancel(drift=drift, session=session)
            except Exception as exc:  # noqa: BLE001 — cancel is best-effort
                log.debug(
                    "DefaultSteerer._handle_drift: request_invocation_cancel raised: %s",
                    exc,
                )
        if promote_to_steer:
            await self._promote_drift_to_steer(drift, session)
            return
        # Route through the intervention ladder. The per-(kind, task)
        # occurrence count drives the "first vs repeat" distinction in
        # the ladder table -- we read it BEFORE any mutation so the
        # mapping sees the state at drift-fire time. ``occurrence_count``
        # is derived from ``session.refine_outcomes`` via
        # :meth:`_occurrence_count_for_ladder` (goldfive#215 iter-8 P2:
        # the outcome dict replaces the deleted ``refine_failure_counts``
        # int counter).
        occurrence_count = self._occurrence_count_for_ladder(session, drift)
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
        if self._steerer._planner is None or session.plan is None:
            return
        if drift.kind is DriftKind.REFINE_VALIDATION_FAILED:
            # Terminal planner signal (goldfive#133). Do NOT call refine
            # again on it. The ladder already routes this to Level 4 so
            # control flow normally won't reach here, but belt-and-braces.
            return
        # Outcome-based gate (goldfive#215 iter-8 P2 — unified G1+G3).
        # Skip refine when ``(kind, task)`` already has a terminal
        # outcome on this turn: a prior refine already succeeded (the
        # current drift is a same-turn replay of an addressed
        # condition), or prior refines have failed
        # >= REFINE_FAILURE_THRESHOLD times (the threshold trip already
        # marked the task FAILED + emitted REPEATED_FAILURE; a third
        # tick must not retry). USER_STEER / USER_CANCEL / GOAL_DRIFT
        # bypass the gate — operator intent always honoured,
        # trajectory drifts have their own rate limiters.
        if drift.kind not in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            outcome_key = (drift.kind.value, drift.current_task_id or "")
            outcome = session.refine_outcomes.get(outcome_key)
            if outcome is not None:
                if outcome.state == "succeeded":
                    log.debug(
                        "refine skipped: prior succeeded outcome (kind=%s task=%r)",
                        drift.kind.value,
                        drift.current_task_id,
                    )
                    return
                if outcome.fail_count >= self._steerer.REFINE_FAILURE_THRESHOLD:
                    log.debug(
                        "refine skipped: failure threshold reached (kind=%s task=%r count=%d)",
                        drift.kind.value,
                        drift.current_task_id,
                        outcome.fail_count,
                    )
                    return
        # Progress-based escalation (goldfive#271). Orthogonal to the
        # outcome gate: a task that has been silent past the configured
        # stall threshold escalates to HUMAN_INTERVENTION_REQUIRED
        # instead of looping the planner. A productively-iterating task
        # has continuous progress events; a stuck task does not.
        if self._is_task_progress_stalled(drift, session):
            await self._emit_progress_stalled_escalation(drift, session)
            return
        # Plumb the session into the planner's drift-emitter callback
        # for the duration of this refine call so the planner can emit
        # REFINE_VALIDATION_FAILED drifts through the normal event
        # pipeline. Cleared in a ``finally`` so exceptions don't leave
        # a stale session pointer. ContextVar isolation keeps concurrent
        # runs from stomping each other (goldfive#133, PR #294 audit).
        _active_session_token = self._steerer._active_session_var.set(session)
        # goldfive a4: mint a refine-attempt id for correlation across
        # ``refine_attempted`` and the paired success/failure event.
        attempt_id = self._steerer.plans._new_attempt_id()
        await self._steerer.plans._emit_refine_attempted(session, drift, attempt_id=attempt_id)
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
            adapter = self._steerer._adapter
            if adapter is not None:
                tree = getattr(adapter, "available_agents_tree", None)
                if isinstance(tree, list) and tree:
                    available_agents = list(tree)
                else:
                    flat = getattr(adapter, "available_agents", None)
                    if flat:
                        available_agents = list(flat)
            refine_accepts_registry = _planner_refine_accepts_available_agents(
                self._steerer._planner
            )
            # Phase 3.5 (goldfive#271) tripwire wrapper — the
            # ``except BaseException: stash; raise`` arm below is the
            # compliance branch (CANCELLATION-CONTRACT.md §1.2). The
            # boundary catch site at ``ADKAdapter._invoke_internal``
            # asserts ``mark_stash_completed()`` fired before the
            # cancel propagated past us.
            with _state_audit.cancellation_stash_audited("DefaultSteerer._handle_drift.refine"):
                try:
                    if refine_accepts_registry:
                        revised = await self._steerer._planner.refine(
                            plan=session.plan,
                            drift=drift,
                            goals=list(session.goals),
                            available_agents=available_agents,
                        )
                    else:
                        revised = await self._steerer._planner.refine(
                            plan=session.plan,
                            drift=drift,
                            goals=list(session.goals),
                        )
                except RefineExhausted as exc:
                    # goldfive#271: planner explicitly signals it cannot
                    # produce a meaningful change. Same escalation path
                    # as the structural no-op detector — pause for
                    # human intervention rather than retrying.
                    log.info(
                        "DefaultSteerer._handle_drift: planner.refine raised "
                        "RefineExhausted for kind=%s task=%r: %s",
                        drift.kind.value,
                        drift.current_task_id,
                        exc,
                    )
                    await self._steerer.plans._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="refine_exhausted",
                        reason=str(exc) or "planner signalled handler exhaustion",
                        detail="",
                    )
                    await self._emit_handler_exhausted_escalation(drift, session)
                    return
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
                    await self._steerer.plans._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="llm_error",
                        reason=str(exc),
                        detail=type(exc).__name__,
                    )
                    await self._escalate_refine_failure_as_critical_drift(
                        session, drift, reason=str(exc)
                    )
                    await self._record_refine_outcome(session, drift, succeeded=False)
                    return
                except BaseException as exc:  # noqa: BLE001
                    # Phase 3.5 (CANCELLATION-CONTRACT.md §C4): ``CancelledError``
                    # bypasses the ``except Exception`` branch (it is a
                    # ``BaseException`` since Py 3.8). Emit the paired
                    # ``refine_failed`` observability event so a refine cancelled
                    # mid-flight does not leave sinks with an unmatched
                    # ``refine_attempted``. The ``finally`` below still resets
                    # ``_active_session_var``; we only own the paired-event
                    # stash here. Re-raise so cancellation continues to
                    # propagate per the asyncio contract.
                    await self._steerer.plans._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="cancelled",
                        reason=f"refine cancelled: {type(exc).__name__}",
                        detail=type(exc).__name__,
                    )
                    # Phase 3.5 tripwire compliance marker (§1.2 form).
                    _state_audit.mark_stash_completed()
                    raise
        finally:
            self._steerer._active_session_var.reset(_active_session_token)
        if revised is None:
            # iter-12 (#204): refine returning None at the steerer level
            # means the planner has already exhausted its internal retry
            # budget (iter-11C's repeat-rejection guard). Treat as
            # handler exhaustion and pause for human intervention rather
            # than emitting a follow-up CRITICAL drift that would
            # recurse through ``_handle_drift`` and eventually abort the
            # run. Mirrors the ``RefineExhausted`` and no-op-revision
            # escalation paths.
            log.warning(
                "DefaultSteerer._handle_drift: planner.refine(kind=%s) returned None; "
                "plan unchanged — escalating to HUMAN_INTERVENTION_REQUIRED",
                drift.kind.value,
            )
            await self._steerer.plans._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="parse_error",
                reason="planner returned no revised plan",
                detail="",
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # I4 fix: fold runtime terminal statuses from the prior plan
        # onto the revised plan BEFORE validation. A task that was
        # cancelled / failed / NOT_NEEDED out-of-band between revisions
        # (e.g. overlay reap → NOT_NEEDED, executor reachability audit
        # → CANCELLED, coordinator reporting-tool → COMPLETED) should
        # carry that status into the persisted snapshot, even when the
        # LLM's view of the prior plan was stale.
        # goldfive#247: fold returns a NEW Plan (Plan is frozen).
        revised = self._steerer.plans._fold_runtime_terminal_statuses(revised, session.plan)
        try:
            revised.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            # iter-12 (#204): the revised plan is structurally invalid
            # AND the planner has already exhausted its internal
            # validator-rejection retry budget (iter-11C). Treat as
            # handler exhaustion and pause for human intervention.
            #
            # Operator visibility: the SCHEMA_VIOLATION drift is
            # preserved at INFO severity (observability-only — does NOT
            # recurse through ``_handle_drift``) so harmonograf and
            # other sinks still see the schema-failure signal carrying
            # the validator's reason. The actionable signal is the
            # paired ``refine_failed(validator_rejected)`` envelope and
            # the HUMAN_INTERVENTION_REQUIRED escalation that follows.
            #
            # Passing ``prior=session.plan`` to ``validate`` enables
            # PLAN-LIFECYCLE.md §3.1 (terminal task preservation) and
            # §3.2 (terminal->terminal edge preservation) on top of the
            # usual structural checks.
            await self._steerer.plans._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="validator_rejected",
                reason=f"plan validation failed: {exc}",
                detail=type(exc).__name__,
            )
            await self._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.INFO,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                ),
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # No-op revision rejection (goldfive#271 — replaces the deleted
        # count cap). If the LLM produced a "refine" that is structurally
        # identical to the prior plan (same task ids, edges, assignees,
        # statuses), treat the handler as exhausted: the planner cannot
        # produce a meaningful change for this drift. Escalate to
        # HUMAN_INTERVENTION_REQUIRED rather than bumping the revision
        # index for a no-op, which would otherwise loop forever on a
        # judge that keeps re-firing on a corrected task.
        if self._steerer.plans._plans_structurally_identical(session.plan, revised):
            log.info(
                "no-op revision skipped (kind=%s task=%r); escalating to "
                "HUMAN_INTERVENTION_REQUIRED",
                drift.kind.value,
                drift.current_task_id,
            )
            await self._steerer.plans._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="no_op_revision",
                reason="planner returned structurally identical plan",
                detail="",
            )
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # Successful refine — record the "succeeded" outcome so a
        # follow-up same-(kind, task) drift on this turn skips refine
        # (the prior refine already addressed it).
        await self._record_refine_outcome(session, drift, succeeded=True)
        # Capture the outgoing plan BEFORE _apply_revision installs the
        # revised one; _emit_plan_revised diffs the two to populate the
        # PlanRevisionDiff sidecar (PLAN-LIFECYCLE.md §2, §8 gap #4).
        prev_plan = session.plan
        # goldfive#247: _apply_revision returns the stamped instance.
        # goldfive#255: _apply_revision returns ``(revised, was_installed)``
        # so the caller can thread the install outcome into PlanRevised's
        # ``dry_run`` marker.
        revised, was_installed = self._steerer.plans._apply_revision(session, revised, drift)
        # Cancel the in-flight coordinator invocation now that the plan
        # it was reasoning against has been superseded (goldfive#271
        # follow-up — v15 concurrent-invocation bug). Order: cancel
        # BEFORE PlanRevised emit so the synthetic InvocationCancelled
        # sink event lands adjacent to the revision in the wire log
        # and operators can correlate the two. Best-effort, never
        # raises — a no-op cancel still leaves the new plan installed.
        await self._cancel_inflight_for_revision(drift, session)
        await self._steerer.plans._emit_plan_revised(
            session,
            revised,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
            dry_run=not was_installed,
        )
        # Level 3 (CANCEL_REINVOKE) handoff (Phase 2 of the path-
        # duality fix). Pre-Phase-2 this stuffed
        # ``session.pending_corrective_message`` — a write-only slot
        # nobody read after the overlay loop took shape, leaving the
        # coordinator running its original chain blind to the plan
        # swap. Phase 2 dispatches a ``GOLDFIVE_STEER`` ControlMessage
        # so the executor's invoke loop cancels in-flight work and
        # restarts with the corrective body framed as ``[GOLDFIVE
        # STEERING CONTROL …]`` — the same junction USER_STEER uses.
        if level is InterventionLevel.CANCEL_REINVOKE:
            await self._dispatch_goldfive_steer_control(drift, session)
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
        # kinds (CONFABULATION_RISK, etc.) do not need mid-invocation
        # rescue — their corrective path fires at the next task
        # boundary or via Level 3 CANCEL_REINVOKE.
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

    # ------------------------------------------------------------------
    # Level dispatch (#142) — nudge / steer / pause
    # ------------------------------------------------------------------

    async def _dispatch_nudge(self, drift: DriftEvent, session: Session) -> None:
        """Level 2 dispatch: queue a soft follow-up message on the session.

        The Runner's overlay loop (goldfive#141) picks up the queued
        nudge at the next invocation boundary and sends it as a gentle
        corrective user message. Until #141 lands, the queue is
        observable but inert; nothing consumes it.
        """
        from goldfive.steerer import compose_corrective_user_message

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

    async def _dispatch_goldfive_steer_control(
        self,
        drift: DriftEvent,
        session: Session,
        *,
        body_override: str = "",
    ) -> bool:
        """Mint and dispatch a ``GOLDFIVE_STEER`` ControlMessage.

        Phase 2 of the path-duality fix. Replaces the dead
        ``session.pending_corrective_message`` write at every
        CANCEL_REINVOKE / promote-to-steer site so goldfive-authored
        drift rides the same cancel-and-restart junction as
        user-authored ``STEER``.

        ``body_override``: optional text to use as the corrective
        body. The promotion path passes its already-composed
        :meth:`_compose_goldfive_steer_body` output; the Level 3
        CANCEL_REINVOKE path leaves it empty and falls back to
        :func:`compose_corrective_user_message` against the freshly
        revised plan.

        Returns ``True`` on successful dispatch, ``False`` on no
        bound channel / send failure (best-effort — see
        :meth:`DefaultSteerer._dispatch_goldfive_control`).
        """
        from goldfive.control import ControlKind, ControlMessage
        from goldfive.steerer import compose_corrective_user_message

        if body_override:
            body = body_override
        else:
            body = compose_corrective_user_message(
                drift=drift,
                refined_plan=session.plan,
            )
        superseded_ids = (
            [str(drift.current_task_id)] if drift.current_task_id else []
        )
        # Replacement task ids: pick the first PENDING task on the
        # revised plan as the natural successor — the executor uses
        # this to render an explicit "pick these up instead" block in
        # the restart message.
        replacement_ids: list[str] = []
        plan = session.plan
        if plan is not None:
            for task in plan.tasks:
                if task.status is TaskStatus.PENDING and task.id:
                    replacement_ids.append(task.id)
                    break
        msg = ControlMessage(
            kind=ControlKind.GOLDFIVE_STEER,
            payload={
                "drift_kind": drift.kind.value,
                "drift_id": str(getattr(drift, "id", "") or ""),
                "body": body,
                "superseded_task_ids": superseded_ids,
                "replacement_task_ids": replacement_ids,
            },
        )
        # goldfive#254 — observation-only: skip the actual ControlMessage
        # enqueue but log the would-be payload at INFO so operators can
        # see what would have been dispatched (drift kind, task id, body).
        # No cancel-and-restart fires on the executor; the live invocation
        # continues against the prior plan.
        if not self._steerer._should_inject():
            log.info(
                "DefaultSteerer._dispatch_goldfive_steer_control: "
                "observation_only=True — SKIPPING GOLDFIVE_STEER enqueue. "
                "would_have_dispatched kind=%s task=%s drift_id=%s "
                "superseded=%s replacement=%s body=%r",
                drift.kind.value,
                drift.current_task_id or "-",
                str(getattr(drift, "id", "") or ""),
                superseded_ids,
                replacement_ids,
                body[:200],
            )
            return False
        landed = await self._steerer._dispatch_goldfive_control(msg)
        log.debug(
            "DefaultSteerer._dispatch_goldfive_steer_control: "
            "kind=%s task=%s landed=%s",
            drift.kind.value,
            drift.current_task_id or "-",
            landed,
        )
        return landed

    async def _dispatch_goldfive_pause_control(
        self,
        drift: DriftEvent,
        session: Session,
        *,
        reason: str,
    ) -> bool:
        """Mint and dispatch a ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage.

        Phase 2 of the path-duality fix. Replaces the dead
        ``session.paused_for_human_intervention = True`` flag-set at
        every Level-4 / progress-stall / handler-exhausted escalation
        site so the executor's pre-task loop blocks via the same
        channel state as a user-issued ``PAUSE``.

        Returns ``True`` on successful dispatch, ``False`` on no
        bound channel / send failure.

        goldfive#264 — observation-only carve-out. Under
        ``SteeringConfig.observation_only`` the would-be
        ``GOLDFIVE_PAUSE_ESCALATE`` is SKIPPED: dispatching it on the
        channel sets ``goldfive_pause_message`` on the executor's
        :class:`~goldfive.executors._control.ControlOutcome`, which in
        turn drives ``_cancel_invoke_task`` and ends the overlay turn.
        That kills the live invocation — exactly the enforcement
        ``observation_only`` exists to suppress. The originating
        ``HUMAN_INTERVENTION_REQUIRED`` drift emitted by the caller
        (e.g. :meth:`_emit_handler_exhausted_escalation`,
        :meth:`_emit_progress_stall_escalation`,
        :meth:`_dispatch_pause_escalate`) is OUTSIDE this dispatch and
        continues to fire — observers/sinks still see the escalation,
        the operator can still react, but goldfive does NOT cancel the
        in-flight invocation. Mirrors the gate pattern at
        :meth:`_dispatch_goldfive_steer_control` (goldfive#254) and
        :meth:`request_invocation_cancel`.

        Live reproduction (2026-05-11, session
        ``4538863f-0dea-4fe8-97b4-5f660ee2cb7f``): an OFF_TOPIC drift
        under ``observation_only=True`` reached refine handler
        exhaustion (#271 no-op-revision path), which called this
        method, which dispatched the channel message, which cancelled
        the in-flight invoke. The carve-out below stops that chain.
        """
        from goldfive.control import ControlKind, ControlMessage

        if not self._steerer._should_inject():
            log.info(
                "DefaultSteerer._dispatch_goldfive_pause_control: "
                "observation_only=True — SKIPPING GOLDFIVE_PAUSE_ESCALATE "
                "dispatch. would_have_dispatched kind=%s task=%s "
                "drift_id=%s reason=%r",
                drift.kind.value,
                drift.current_task_id or "-",
                str(getattr(drift, "id", "") or ""),
                reason,
            )
            return False
        msg = ControlMessage(
            kind=ControlKind.GOLDFIVE_PAUSE_ESCALATE,
            payload={
                "reason": reason,
                "drift_id": str(getattr(drift, "id", "") or ""),
                "drift_kind": drift.kind.value,
            },
        )
        landed = await self._steerer._dispatch_goldfive_control(msg)
        log.debug(
            "DefaultSteerer._dispatch_goldfive_pause_control: "
            "kind=%s task=%s landed=%s reason=%r",
            drift.kind.value,
            drift.current_task_id or "-",
            landed,
            reason,
        )
        return landed

    async def _dispatch_pause_escalate(
        self,
        drift: DriftEvent,
        session: Session,
    ) -> None:
        """Level 4 dispatch: emit HUMAN_INTERVENTION_REQUIRED and pause.

        Does NOT call ``planner.refine`` -- Level 4 signals that the
        planner cannot recover. Phase 2 of the path-duality fix:
        dispatches a ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage on the
        bound channel so the executor's pre-task loop blocks via the
        same channel state as a user ``PAUSE``. Pre-Phase-2 this
        flipped ``session.paused_for_human_intervention = True`` — a
        flag the executor read on its next iteration; the indirection
        was synonymous with the channel signal but parallel-tracked
        from the user-PAUSE path.

        Emits a CRITICAL ``HUMAN_INTERVENTION_REQUIRED`` drift so
        sinks / the UI can surface the pause and let the user decide
        what to do.

        When the drift reaching Level 4 is *already* a
        ``HUMAN_INTERVENTION_REQUIRED`` (e.g. landed here via the
        generic fallback), we pause but do not re-emit the same drift
        a second time -- the original DriftDetected emission at the
        top of :meth:`handle_drift` already carried the signal.
        """
        await self._dispatch_goldfive_pause_control(
            drift,
            session,
            reason=(
                f"pause_escalate from {drift.kind.value}: {drift.detail}"
                if drift.detail
                else f"pause_escalate from {drift.kind.value}"
            ),
        )
        if drift.kind is DriftKind.HUMAN_INTERVENTION_REQUIRED:
            # Already emitted at the top of _handle_drift; just pause.
            return
        # goldfive#271 PR1 — close the originating condition with
        # ``human_intervention_required`` so consumers tracking the
        # condition_id see the terminal lifecycle on the *original*
        # condition, not just on the synthesized HUMAN_INTERVENTION
        # row. The originating drift was already emitted at the top of
        # ``_handle_drift`` (legacy path) under its own condition_id;
        # this call swaps that condition's recorded lifecycle so a
        # later get_active_drift returns the terminal state.
        try:
            origin_cid = _ostate.compute_condition_id(
                kind=drift.kind,
                task_id=str(getattr(drift, "current_task_id", "") or ""),
                agent_id=str(getattr(drift, "current_agent_id", "") or ""),
                turn_id=str(getattr(session, "run_id", "") or ""),
            )
            _ostate.escalate_drift_to_human_intervention(session.state, origin_cid)
        except Exception as exc:  # noqa: BLE001
            log.debug("DefaultSteerer: drift-lifecycle escalate skipped (%s)", exc)
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

    # ------------------------------------------------------------------
    # Adapter cancel-reason tagging
    # ------------------------------------------------------------------
    #
    # Symbolic cancel-reason tags — mirror
    # :mod:`goldfive.adapters.adk` constants but duplicated as plain
    # strings here to avoid a hard import of the optional ADK adapter
    # module from the provider-agnostic steerer. Keep in sync with
    # :data:`goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER` etc.
    _ADAPTER_CANCEL_REASON_USER_STEER: str = "user_steer"

    def _tag_adapter_cancel_reason(
        self, drift: DriftEvent, *, session: Session | None = None
    ) -> None:
        """Set the next adapter cancel reason based on ``drift.kind``.

        USER_STEER drift -> ``"user_steer"``. Other kinds currently leave
        the tag unset so the adapter falls through to the generic
        content variant. Tolerates adapters that don't carry the
        attribute (no-op) and an unbound adapter (no-op). See
        goldfive#139.

        Routes the write through
        :meth:`ADKAdapter.set_next_cancel_reason` when the adapter
        exposes that helper (PR #294 audit / goldfive#271 follow-up)
        so the tag is keyed by ``session.id`` and cannot bleed across
        concurrent goldfive sessions sharing one adapter. Falls back
        to the bare attribute write for adapters / stubs that predate
        the helper.

        The goldfive-steer-unification promotion path uses a separate
        helper (:meth:`_tag_adapter_cancel_reason_for_promotion`) to
        stamp a ``"goldfive_<drift_kind>"`` reason when promoting a
        detector drift to a full steer; keeping the two call sites
        distinct avoids muddling the pre-unification tag semantics for
        unpromoted paths.
        """
        adapter = self._steerer._adapter
        if adapter is None:
            return
        if drift.kind is DriftKind.USER_STEER:
            reason = self._ADAPTER_CANCEL_REASON_USER_STEER
        else:
            return
        self._write_adapter_cancel_reason(adapter, reason, session)

    def _tag_adapter_cancel_reason_for_promotion(
        self, drift: DriftEvent, *, session: Session | None = None
    ) -> str:
        """Stamp a goldfive-specific cancel reason on the bound adapter.

        Returns the reason string stamped (or synthesised) so callers
        can record it on the session for downstream observability.
        Mirrors :meth:`_tag_adapter_cancel_reason` semantics: adapters
        without the per-session helper are tolerated.
        """
        reason = f"goldfive_{drift.kind.name.lower()}"
        adapter = self._steerer._adapter
        if adapter is None:
            return reason
        self._write_adapter_cancel_reason(adapter, reason, session)
        return reason

    @staticmethod
    def _write_adapter_cancel_reason(adapter: Any, reason: str, session: Session | None) -> None:
        """Route the cancel-reason tag through the session-aware helper.

        Falls back to the legacy bare-attribute write for adapters /
        stubs that don't expose :meth:`set_next_cancel_reason`. See
        :meth:`ADKAdapter.set_next_cancel_reason` for the rationale.
        """
        setter = getattr(adapter, "set_next_cancel_reason", None)
        if callable(setter) and session is not None:
            try:
                setter(session, reason)
                return
            except Exception as exc:  # noqa: BLE001
                log.debug("DefaultSteerer: set_next_cancel_reason raised: %s", exc)
        try:
            adapter._next_cancel_reason = reason
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer: could not tag adapter cancel reason: %s",
                exc,
            )

    async def _request_adapter_cancel(self, reason: str) -> None:
        """Invoke the optional ``adapter.request_cancel(reason)`` hook.

        goldfive#241 — a goldfive-promoted steer needs the in-flight
        LLM call to stop NOW so the contaminated reasoning / tool
        calls don't keep writing to the session while we queue the
        restart. The ADK adapter exposes :meth:`ADKAdapter.request_cancel`
        which fires ``task.cancel()`` on the asyncio task driving
        ``runner.run_async`` so the stream raises ``CancelledError``
        and the adapter's standard heal path runs with the already-
        stamped ``_next_cancel_reason`` tag.

        Optional protocol: adapters that don't implement the method
        (Claude adapter, callable adapter, test stubs without live
        invocations) keep the legacy deferred-cancel semantics —
        ``_next_cancel_reason`` is still tagged, the restart message
        is still queued, and the next executor checkpoint still
        terminates the invocation. Tolerates an unbound adapter and
        swallows every failure so a best-effort cancel cannot break
        the promotion path.
        """
        adapter = self._steerer._adapter
        if adapter is None:
            return
        fn = getattr(adapter, "request_cancel", None)
        if not callable(fn):
            return
        try:
            result = fn(reason)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._request_adapter_cancel(reason=%r): adapter raised: %s",
                reason,
                exc,
            )

    # ------------------------------------------------------------------
    # Cooperative cancellation (goldfive#251 Stream C / 7a)
    # ------------------------------------------------------------------

    def _is_late_drift_for_terminated_invocation(
        self, drift: DriftEvent, session: Session
    ) -> bool:
        """Return True iff a goldfive-authored drift's target is gone (goldfive#319).

        Background reasoning-judge tasks (goldfive#251) run off the
        critical path so the adapter's model-response callback can return
        before the LLM judge finishes. With goldfive#319's removal of the
        per-turn cancel-drain, a slow judge spawned in turn N may now
        produce its verdict in turn N+1 — well after the original agent
        invocation has terminated. Routing such a verdict through the
        cancel + ladder dispatch is a category error: it could cancel an
        unrelated invocation or trigger a refine against a plan whose
        offending step is already complete. The drift is still emitted
        on the sink (observability preserved); this guard short-circuits
        the dispatch.

        The check uses :class:`StateStore` as the live registry
        of in-flight invocations (Phase 3.5 component 1, goldfive#271).
        Two conditions count as "late":

        * **No active invocations** — every agent has finished its turn
          and any drift currently being handled is by definition stale.
        * **Cancel-pending on the session** (goldfive#242) — a previous
          drift already requested a cooperative cancel for one or more
          invocations. The active-task registry takes 4-8s to drain
          while ADK winds those invocations down; during that window
          any newly-arriving goldfive-authored drift would dispatch a
          refine against an effectively-dead session. Stamping the
          cancel-pending flag synchronously at
          :meth:`request_invocation_cancel` time closes that race.

        User-authored drifts (USER_STEER / USER_CANCEL / USER_PAUSE)
        always bypass this guard — they are forward-looking operator
        directives, not tied to a specific in-flight invocation.
        """
        # User-authored drifts always pass through. ``authored_by`` was
        # normalised at the top of :meth:`handle_drift`.
        if (drift.authored_by or "").lower() == "user":
            return False
        try:
            from goldfive.state_store import StateStore

            store = StateStore.for_session(session)
            active = store.active_invocation_ids()
            cancel_pending = store.cancel_requested_invocation_ids()
        except Exception as exc:  # noqa: BLE001 — defensive
            log.debug(
                "DefaultSteerer._is_late_drift_for_terminated_invocation: "
                "active_invocation_ids lookup raised (treating as not-late): %s",
                exc,
            )
            return False
        # Symmetric predicate: late when the active list is empty OR
        # any cancel is pending on the session. The cancel-pending
        # branch closes the iter-11D race (goldfive#242) where the
        # cancel-request has landed but ADK hasn't yet finished
        # winding down the cancelled invocation.
        return (not active) or bool(cancel_pending)

    def _resolve_active_invocation_ids(self, drift: DriftEvent, session: Session) -> list[str]:
        """Resolve which invocation_id(s) a cancel should target.

        Returns an ordered list of invocation ids that are "active"
        with respect to the triggering drift. The primary source is
        the reconciler's invocation bookkeeping (goldfive#151
        introduced the ``_invocation_agent`` / ``_invocation_parent``
        maps). When the reconciler is unavailable or empty, falls
        back to the drift's ``current_agent_id``-keyed invocation
        (best effort via the adapter's active-context invocation id)
        and finally returns an empty list.

        Tree-agnostic: the method does NOT special-case "the
        coordinator" or "the root agent" — it targets whichever
        invocation matches the drift's context and lets the plugin's
        child-propagation logic flag the rest of the sub-tree.
        """
        candidates: list[str] = []
        reconciler = getattr(session, "_reconciler", None)
        if reconciler is None:
            # The steerer doesn't hold a direct reference to the
            # reconciler; the adapter's plugin does. Walk it via the
            # adapter when the plugin exposes the attribute.
            adapter = self._steerer._adapter
            plugin = getattr(adapter, "_plugin", None) if adapter is not None else None
            reconciler = getattr(plugin, "_reconciler", None) if plugin is not None else None
        if reconciler is not None:
            try:
                inv_agent = getattr(reconciler, "_invocation_agent", None)
                if isinstance(inv_agent, Mapping) and drift.current_agent_id:
                    # Match by agent name — most drifts carry
                    # ``current_agent_id`` set to the running agent's name.
                    for inv_id, agent_name in inv_agent.items():
                        if agent_name == drift.current_agent_id and inv_id:
                            candidates.append(str(inv_id))
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer._resolve_active_invocation_ids: reconciler lookup raised: %s",
                    exc,
                )
        # Fallback: the adapter's plugin pins a top-level invocation_id
        # for the currently-driving dispatch. When the reconciler lookup
        # produced nothing, the top-level id is the best we can do —
        # cancel propagation from there will flag any sub-invocations.
        if not candidates:
            adapter = self._steerer._adapter
            plugin = getattr(adapter, "_plugin", None) if adapter is not None else None
            top = str(getattr(plugin, "_top_invocation_id", "") or "")
            if top:
                candidates.append(top)
        return candidates

    async def request_invocation_cancel(
        self,
        *,
        drift: DriftEvent,
        session: Session,
        cancel_inflight_task: bool = False,
    ) -> list[str]:
        """Flag the invocation(s) associated with ``drift`` for
        cooperative cancellation (goldfive#251 Stream C / 7a).

        Called from :meth:`handle_drift` and
        :meth:`_promote_drift_to_steer` when the drift's severity is
        CRITICAL — the only tier on the ladder that reaches the hard
        cancel per the severity decision. INFO / WARNING drifts flow
        through their usual nudge / absorb paths without touching
        this method.

        Writes a :class:`~goldfive.types.CancellationRequest` onto the
        adapter's plugin state for every resolved active invocation
        id. The plugin propagates to children automatically. Returns
        the list of flagged invocation ids (including children) for
        observability; callers can log / sink-emit from the list.

        When ``cancel_inflight_task=True`` (goldfive#271 follow-up —
        v15 concurrent-invocation bug), the plugin ALSO fires
        ``task.cancel()`` on the registered asyncio.Task driving each
        flagged invocation, deferred via ``loop.call_soon`` so an
        inline same-task caller still completes its current emission
        work before the cancel lands. Default False so the existing
        pre-refine cancel paths keep their flag-only semantics; the
        post-refine helper :meth:`_cancel_inflight_for_revision`
        opts in explicitly so the cancel only fires AFTER a
        superseding plan has been installed.

        Guard rails:

        * No-op when no adapter is bound.
        * No-op when no active invocation can be resolved (e.g. the
          drift was synthesized before any agent turn started) — this
          is the "empty invocation-id guard" called out in the brief.
        * Tolerates missing plugin methods (third-party adapters that
          don't implement :meth:`request_invocation_cancel`) by
          falling through silently; the rest of the ladder (refine,
          restart message) still runs and eventually catches up at
          the next task boundary.
        * Plugins whose ``request_invocation_cancel`` predates
          ``cancel_inflight_task`` (TypeError on the kwarg) fall back
          to the kwarg-less call so older third-party plugins don't
          break — the task-cancel step is silently skipped.

        Observation-only mode (goldfive#254): when
        :meth:`DefaultSteerer._should_inject` is ``False`` this method
        returns ``[]`` without consulting the plugin or stamping
        ``cancel_requested_invocation_ids``. Logged at INFO so an
        operator can see WHAT would have been cancelled (drift kind,
        task / agent id) without the cancel actually firing on the
        live invocation.
        """
        if not self._steerer._should_inject():
            log.info(
                "DefaultSteerer.request_invocation_cancel: "
                "observation_only=True — SKIPPING cancel for "
                "drift kind=%s severity=%s agent=%s task=%s",
                drift.kind.value,
                drift.severity.value,
                drift.current_agent_id or "-",
                drift.current_task_id or "-",
            )
            return []
        adapter = self._steerer._adapter
        if adapter is None:
            return []
        plugin = getattr(adapter, "_plugin", None)
        if plugin is None:
            return []
        fn = getattr(plugin, "request_invocation_cancel", None)
        if not callable(fn):
            return []
        invocation_ids = self._resolve_active_invocation_ids(drift, session)
        # Stamp the cancel-pending flag SYNCHRONOUSLY before any
        # plugin / async work (goldfive#242). The active-task
        # registry takes 4-8s to drain while ADK winds the cancelled
        # invocations down; during that window the late-drift gate
        # would otherwise see ``active_invocation_ids()`` non-empty
        # and let a freshly-arriving goldfive-authored drift dispatch
        # a refine against an effectively-dead session. Flipping the
        # flag here closes the race: any drift handled after this
        # point sees ``cancel_requested_invocation_ids()`` non-empty
        # via :meth:`_is_late_drift_for_terminated_invocation` and
        # short-circuits.
        if invocation_ids:
            try:
                from goldfive.state_store import StateStore

                store = StateStore.for_session(session)
                for inv_id in invocation_ids:
                    store.mark_invocation_cancel_requested(inv_id)
            except Exception as exc:  # noqa: BLE001 — defensive
                log.debug(
                    "DefaultSteerer.request_invocation_cancel: "
                    "cancel-pending stamp raised (continuing): %s",
                    exc,
                )
        if not invocation_ids:
            # Empty invocation-id guard — drift has no identifiable
            # in-flight invocation. Don't fabricate one; the cancel
            # would misfire on whatever invocation happens to share a
            # blank id. The drift still observed, refine still runs;
            # cancel is a best-effort add-on.
            log.debug(
                "DefaultSteerer.request_invocation_cancel: no active invocation "
                "for drift kind=%s agent=%s task=%s — skipping cancel",
                drift.kind.value,
                drift.current_agent_id or "-",
                drift.current_task_id or "-",
            )
            return []
        # Build the request once and reuse for every targeted id so
        # sink events from propagation share a common fingerprint.
        import time as _time_mod

        request = CancellationRequest(
            invocation_id=invocation_ids[0],
            reason=self._cancel_reason_for_drift(drift),
            severity=drift.severity,
            drift_id=str(getattr(drift, "id", "") or ""),
            drift_kind=drift.kind.value,
            requested_at_ms=int(_time_mod.time() * 1000),
            detail=(drift.detail or "")[:200],
        )
        flagged: list[str] = []
        for inv_id in invocation_ids:
            try:
                result = fn(
                    invocation_id=inv_id,
                    request=request,
                    cancel_inflight_task=cancel_inflight_task,
                )
            except TypeError:
                # Older plugin without the ``cancel_inflight_task``
                # kwarg (third-party / pre-#271-follow-up). Fall back
                # to the legacy signature; the task-cancel step is
                # silently skipped, but the flag-only contract is
                # preserved.
                try:
                    result = fn(invocation_id=inv_id, request=request)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "DefaultSteerer.request_invocation_cancel: "
                        "plugin.request_invocation_cancel(%s) raised: %s",
                        inv_id,
                        exc,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer.request_invocation_cancel: "
                    "plugin.request_invocation_cancel(%s) raised: %s",
                    inv_id,
                    exc,
                )
                continue
            if isinstance(result, list):
                flagged.extend(str(x) for x in result)
            else:
                flagged.append(inv_id)
        if flagged:
            log.info(
                "DefaultSteerer.request_invocation_cancel: flagged "
                "invocations=%s for drift kind=%s severity=%s",
                flagged,
                drift.kind.value,
                drift.severity.value,
            )
        return flagged

    @staticmethod
    def _should_request_cancel_for_drift(drift: DriftEvent) -> bool:
        """Decide whether a drift warrants a cooperative cancel.

        Severity ladder (goldfive#251 design decision):

        * ``DriftSeverity.INFO`` — never cancels. Info drifts are
          either periodic-check signals or soft one-shots; cancel
          would be disproportionate.
        * ``DriftSeverity.WARNING`` — never cancels. Warning drifts
          route to the existing ABSORB / NUDGE ladder paths; the
          refined plan lands on the next task boundary without
          preempting the in-flight turn.
        * ``DriftSeverity.CRITICAL`` — cancels. The in-flight turn's
          output is likely to contaminate its parent's transcript
          (stale prompt, wrong scope, broken tool); short-circuit
          cleanly and let the parent see ``{"status": "cancelled"}``.

        User-authored drifts (``USER_STEER`` / ``USER_CANCEL`` /
        ``USER_PAUSE``) bypass the severity gate — an operator
        directive must be honoured even when the ControlMessage-to-
        DriftEvent coercion landed on a lower severity tier.
        """
        if drift.kind in (
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.USER_PAUSE,
        ):
            return True
        return drift.severity is DriftSeverity.CRITICAL

    @staticmethod
    def _cancel_reason_for_drift(drift: DriftEvent) -> str:
        """Map a drift into a short symbolic reason for the
        :class:`~goldfive.types.CancellationRequest`.

        USER_STEER / USER_CANCEL / USER_PAUSE get the matching
        ``"user_*"`` shorthand; everything else uses ``"drift"`` as
        the generic tag. The reason is OPERATOR-visible only (lives
        on the InvocationCancelled sink event), so this string is
        free to be descriptive without prompt-injection concerns.
        """
        kind = drift.kind
        if kind is DriftKind.USER_STEER:
            return "user_steer"
        if kind is DriftKind.USER_CANCEL:
            return "user_cancel"
        if kind is DriftKind.USER_PAUSE:
            return "user_pause"
        return "drift"

    async def _cancel_inflight_for_revision(self, drift: DriftEvent, session: Session) -> list[str]:
        """Cancel the in-flight invocation(s) that produced ``drift``.

        Called from every drift-driven PlanRevised emission path right
        after the revised plan has been applied to the session and
        BEFORE the ``PlanRevised`` event is emitted. Closes the gap
        behind the v15 concurrent-invocation bug: a ``refine_steer``
        call (10+ minutes on a slow planner) used to overlap the
        coordinator's invocation for its full duration because the
        existing cancel-state flag only gates SUBSEQUENT callbacks —
        the already-running LLM streaming call kept generating output
        that triggered more drift, looping the refine.

        After Option A (goldfive#271 follow-up), turn-1 first-plan
        installs no longer reach this path:
        :meth:`install_initial_plan` skips it directly because there
        is no in-flight invocation to cancel on a fresh session. Every
        drift-driven install (refine from drift, refine_steer from
        goldfive-steer promotion, operator USER_STEER from a real
        ControlMessage, NEW_WORK_DISCOVERED from an N+1 user message)
        flows through this method. The plugin's
        :meth:`request_invocation_cancel` then writes the cancel-state
        flag (sticky-gate from PR #299) AND fires ``task.cancel()`` on
        the registered asyncio.Task (goldfive#271 follow-up) so the
        coordinator's in-flight LLM call observes ``CancelledError``
        within ~one event-loop tick instead of ~the LLM-call's
        full duration.

        Supersede contract (Bug A fix from v22 validation): every call
        to this method represents a goldfive-INTERNAL cancel — the
        revised plan has just been applied, and the cancel is the
        mechanism by which the in-flight agent is switched onto it.
        Stamps ``session._supersede_pending = True`` BEFORE initiating
        the cancel so the executor's overlay loop
        (:meth:`SequentialExecutor._run_overlay` cancelled branch) can
        distinguish this internal supersede from an external cancel
        (USER_CANCEL via control channel, asyncio.CancelledError from
        above) and restart the passthrough loop with the new plan
        instead of aborting the turn. The executor consumes and clears
        the flag; an unconsumed flag (e.g. cancel never lands because
        the invocation already completed) is harmless — the next
        cancel-branch entry will see and clear it, or the run finishes
        normally and the Session is discarded.

        Best-effort: an unbound adapter, a non-ADK adapter without
        :meth:`request_invocation_cancel`, or an empty resolved
        invocation-id list each result in a no-op (the refined plan
        still lands; the in-flight invocation simply runs to
        completion under the older, less aggressive contract). The
        supersede flag is still stamped in the no-op case — it costs
        nothing and a downstream overlay that DOES observe a cancel
        from a separate path stays correctly classified.
        """
        # Stamp the supersede marker so the overlay loop can
        # distinguish this internal cancel from an external one. See
        # the supersede-contract paragraph in the docstring above.
        try:
            session._supersede_pending = True  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — flag is best-effort
            log.debug(
                "DefaultSteerer._cancel_inflight_for_revision: "
                "could not stamp supersede flag on session: %s",
                exc,
            )
        # Issue #405 LOW #7: also stamp the per-invocation supersede
        # registry on the StateStore. Each active invocation that's
        # about to be cancelled by ``request_invocation_cancel`` gets
        # its own entry, so a concurrent overlay iteration's defensive
        # ``_supersede_pending = False`` clear cannot drop the signal
        # for an unrelated invocation. Best-effort; failure is harmless
        # because the legacy bool above is still set.
        try:
            from goldfive.state_store import StateStore  # noqa: PLC0415 — lazy

            store = StateStore.for_session(session)
            for inv_id in self._resolve_active_invocation_ids(drift, session):
                store.mark_supersede_pending(inv_id)
        except Exception as exc:  # noqa: BLE001 — registry is best-effort
            log.debug(
                "DefaultSteerer._cancel_inflight_for_revision: "
                "could not stamp per-invocation supersede registry: %s",
                exc,
            )
        try:
            return await self.request_invocation_cancel(
                drift=drift,
                session=session,
                cancel_inflight_task=True,
            )
        except Exception as exc:  # noqa: BLE001 — cancel is best-effort
            log.debug(
                "DefaultSteerer._cancel_inflight_for_revision: "
                "request_invocation_cancel raised: %s",
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # goldfive-steer-unification: promotion policy + handler
    # ------------------------------------------------------------------

    # Drift kinds eligible for ladder-promoted steer treatment when
    # goldfive-authored. Mirrors the "content drifts the coordinator
    # acknowledges but doesn't correct" list from the unification
    # design brief: off-topic / intent / unexpected output /
    # confabulation / loop kinds. Other detector kinds (SCHEMA_VIOLATION,
    # REFINE_VALIDATION_FAILED, REPEATED_FAILURE, GOAL_DRIFT, …) keep
    # their pre-unification ladder mapping so escalation / repeated-
    # failure semantics aren't rerouted into the cancel-in-flight path.
    _GOLDFIVE_STEER_ELIGIBLE_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.OFF_TOPIC,
            DriftKind.INTENT_DIVERGENCE,
            DriftKind.UNEXPECTED_OUTPUT,
            DriftKind.CONFABULATION_RISK,
            DriftKind.LOOPING_REASONING,
            DriftKind.LOOPING_TOOL_CALL,
            DriftKind.PLAN_DIVERGENCE,
        }
    )

    def _severity_meets_promotion_threshold(self, severity: DriftSeverity) -> bool:
        """True iff ``severity`` satisfies the configured promotion threshold."""
        threshold = self._steerer._goldfive_steer_threshold
        if threshold == "off":
            return False
        if threshold == "critical":
            return severity is DriftSeverity.CRITICAL
        # "warning" — promote WARNING and CRITICAL.
        return severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)

    def _should_promote_to_steer(self, drift: DriftEvent, session: Session) -> bool:
        """Evaluate the drift against the unification promotion policy.

        Returns ``True`` iff the drift should be dispatched through
        :meth:`_promote_drift_to_steer` instead of the legacy passive
        ladder. Side-effect: stamps ``drift.suppressed_by_user_steer``
        when a fresh user steer is blocking promotion so the subsequent
        ``DriftDetected`` emission reflects the suppression decision.

        The policy:

        1. User-authored drifts (USER_STEER / USER_CANCEL / USER_PAUSE)
           keep their pre-unification handling — USER_STEER already
           routes through the refine path with cancel-in-flight wired
           by the executor. Return ``False``.
        2. The drift kind must be in
           :data:`_GOLDFIVE_STEER_ELIGIBLE_KINDS` — other kinds keep
           their legacy ladder mapping.
        3. The severity must clear the configured ``threshold``.
        4. If a user-authored steer is within the freshness window
           (``suppression_window_turns`` turns), stamp the suppression
           flag and return ``False``. Otherwise return ``True``.
        """
        if drift.kind in self._USER_AUTHORED_DRIFT_KINDS:
            return False
        authored_by = self._resolve_authored_by(drift)
        if authored_by != "goldfive":
            return False
        if drift.kind not in self._GOLDFIVE_STEER_ELIGIBLE_KINDS:
            return False
        if not self._severity_meets_promotion_threshold(drift.severity):
            return False
        # Consult the active user steer freshness window.
        # Phase 1 of goldfive#271 — read through StateStore so
        # the active-steer slot reads from a single named accessor; the
        # underlying ``_ostate.read`` calls still funnel through the
        # goldfive Session.state dict, just behind a typed surface.
        window = self._steerer._goldfive_steer_suppression_window_turns
        if window > 0:
            from goldfive.state_store import StateStore

            active = StateStore.for_session(session).get_active_steer()
            if active is not None and active.source.lower() == "user":
                current_turn = int(getattr(session, "_next_sequence", 0) or 0)
                age = current_turn - active.at_turn
                if 0 <= age < window:
                    drift.suppressed_by_user_steer = True
                    log.info(
                        "goldfive steer suppressed: user steer %r is active "
                        "(age=%d turns, window=%d)",
                        active.body,
                        age,
                        window,
                    )
                    return False
        return True

    async def _promote_drift_to_steer(self, drift: DriftEvent, session: Session) -> None:
        """Promote a goldfive-detected drift into a full steer.

        Ordered side effects (mirrors the USER_STEER path):

        1. Tag the bound adapter's ``_next_cancel_reason`` with a
           ``"goldfive_<drift_kind>"`` symbolic reason so the in-flight
           invocation's synthetic ``function_response`` carries an
           LLM-actionable explanation.
        2. Stamp ``goldfive.active_steer.*`` onto ``session.state``
           (body = derived from :meth:`_compose_goldfive_steer_body`,
           author = ``"goldfive"``, source = ``"goldfive"``).
        3. Record ``drift.id`` in ``goldfive.processed_steer_ids`` so
           the same drift cannot re-promote on a delivery retry.
        4. Call :meth:`LLMPlanner.refine_steer` (or the generic
           ``planner.refine`` fallback when the planner doesn't expose
           the goldfive-specific entry point) with ``source="goldfive"``
           semantics so the refine prompt frames the pivot as a
           correction, not as an operator directive.
        5. Install the revised plan + emit ``PlanRevised``.

        Note on cancel-in-flight: the actual ``task.cancel()`` on the
        adapter invocation is the executor's responsibility
        (:meth:`SequentialExecutor._invoke_with_control` performs it
        when a ``STEER`` ControlMessage arrives). The steerer tags the
        adapter and queues a restart message so that the **next** time
        the executor reaches a cancel / steer checkpoint (either
        because a sink callback requested cancel, or because the
        overlay loop picks up the pending restart message), the
        contaminated invocation is preempted. For the common case
        where the drift is detected from a mid-invocation reasoning
        block and the overlay loop is already streaming, the queued
        restart message reaches the LLM on the next turn — cancel
        semantics identical to USER_STEER.

        Audit issue #402 (fixed): the ``GOLDFIVE_STEER`` ControlMessage
        dispatch fires AFTER :meth:`_emit_plan_revised` has swapped the
        plan to the revised version. Pre-fix the dispatch fired BEFORE
        refine, so the payload's ``replacement_task_ids`` were derived
        from the prior plan — the executor's overlay loop would
        re-invoke against tasks that the imminent revision was about to
        remove / cancel. Ordering now matches :meth:`_handle_drift`'s
        CANCEL_REINVOKE branch (dispatch follows
        ``_apply_revision`` / ``_cancel_inflight_for_revision`` /
        ``_emit_plan_revised``).
        """
        from goldfive.steerer import (
            RefineExhausted,
            _planner_refine_accepts_available_agents,
        )

        # 1. Tag adapter cancel reason.
        cancel_reason = self._tag_adapter_cancel_reason_for_promotion(drift, session=session)
        # Session-visible cancel prefix so ``_mark_cancelled_if_live``
        # stamps it on any TaskCancelled the executor emits for the
        # in-flight task as part of the promotion.
        try:
            session._last_cancel_reason_prefix = cancel_reason
        except Exception:  # noqa: BLE001
            pass
        # 1a. NOTE (#241 emergency revert): previously we fired
        # ``adapter.request_cancel(reason)`` here to terminate the
        # in-flight LLM call immediately. In practice that
        # ``task.cancel()`` propagated a ``CancelledError`` past the
        # executor's invocation-scope catch and killed the entire run
        # — observed as ``run_aborted`` immediately after a
        # goldfive-detected drift. Reverted to the pre-#241 deferred-
        # cancel semantics: we stamp ``_next_cancel_reason`` (above)
        # and queue a restart message; the executor loop sees the
        # queue at the next invocation boundary and resumes with the
        # refined plan. The taint of letting the in-flight call run
        # to completion is a lesser evil than aborting the run.
        # Proper fix (future): scope the cancel to the LLM stream
        # only, or catch ``CancelledError`` at the goldfive-steer
        # boundary and continue.
        # 2. Stamp active-steer state + compose the restart body.
        at_turn = int(getattr(session, "_next_sequence", 0) or 0)
        body = self._compose_goldfive_steer_body(drift)
        try:
            _ostate.set_active_steer(
                session.state,
                body=body,
                at_turn=at_turn,
                author="goldfive",
                source="goldfive",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._promote_drift_to_steer: set_active_steer raised: %s",
                exc,
            )
        # NOTE: the ``GOLDFIVE_STEER`` ControlMessage dispatch used to
        # fire HERE, BEFORE the refine + plan install below. That left
        # the payload carrying the PRIOR plan's task ids in
        # ``superseded_task_ids`` / ``replacement_task_ids`` — the
        # executor's overlay loop would re-invoke against ids that the
        # imminent revision was about to remove / cancel (audit #402,
        # HIGH). The dispatch has been moved to AFTER
        # :meth:`_emit_plan_revised` so the payload reads the NEW
        # plan's task ids. This mirrors :meth:`_handle_drift`'s
        # CANCEL_REINVOKE branch, which already orders dispatch after
        # ``_apply_revision`` / ``_cancel_inflight_for_revision`` /
        # ``_emit_plan_revised`` (the canonical pattern).
        #
        # 3. Record the drift id in processed_steer_ids so a redelivery
        # (same drift id) doesn't re-cancel / re-refine.
        drift_id = str(getattr(drift, "id", "") or "")
        if drift_id:
            try:
                _ostate.record_processed_steer_id(session.state, drift_id)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer._promote_drift_to_steer: record_processed_steer_id raised: %s",
                    exc,
                )
        # 4. Route to planner.refine_steer (source="goldfive") — falls
        # back to planner.refine for planners that don't expose the
        # goldfive-specific entry point.
        if self._steerer._planner is None or session.plan is None:
            return
        # Outcome-based gate (goldfive#215 iter-8 P2). Mirror of the
        # gate in ``_handle_drift``: skip refine_steer when (kind, task)
        # already has a terminal outcome on this turn. USER_STEER /
        # USER_CANCEL / GOAL_DRIFT bypass.
        if drift.kind not in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            outcome_key = (drift.kind.value, drift.current_task_id or "")
            outcome = session.refine_outcomes.get(outcome_key)
            if outcome is not None:
                if outcome.state == "succeeded":
                    log.debug(
                        "refine_steer skipped: prior succeeded outcome (kind=%s task=%r)",
                        drift.kind.value,
                        drift.current_task_id,
                    )
                    return
                if outcome.fail_count >= self._steerer.REFINE_FAILURE_THRESHOLD:
                    log.debug(
                        "refine_steer skipped: failure threshold reached "
                        "(kind=%s task=%r count=%d)",
                        drift.kind.value,
                        drift.current_task_id,
                        outcome.fail_count,
                    )
                    return
        # Progress-based escalation (goldfive#271). Orthogonal to the
        # outcome gate: see parallel check in ``_handle_drift``.
        if self._is_task_progress_stalled(drift, session):
            await self._emit_progress_stalled_escalation(drift, session)
            return
        # ContextVar plumbing for the planner-side drift-emitter and
        # span-context callbacks; per-async-task so concurrent runs
        # sharing this Steerer keep their session pointers isolated.
        _active_session_token = self._steerer._active_session_var.set(session)
        # goldfive a4: same attempt-id correlation contract as
        # ``_handle_drift``.
        attempt_id = self._steerer.plans._new_attempt_id()
        await self._steerer.plans._emit_refine_attempted(session, drift, attempt_id=attempt_id)
        # Resolve the registry constraint (goldfive#151) the same way
        # ``_handle_drift`` does so the goldfive steer refine honours it.
        planner = self._steerer._planner
        available_agents: Any = None
        adapter = self._steerer._adapter
        if adapter is not None:
            tree = getattr(adapter, "available_agents_tree", None)
            if isinstance(tree, list) and tree:
                available_agents = list(tree)
            else:
                flat = getattr(adapter, "available_agents", None)
                if flat:
                    available_agents = list(flat)
        # Phase 3.5 (goldfive#271) tripwire wrapper — see §C4.
        with _state_audit.cancellation_stash_audited(
            "DefaultSteerer._promote_drift_to_steer.refine"
        ):
            try:
                # Call ``planner.refine_steer`` when available; fall back
                # to ``planner.refine``. The fallback exists for test
                # stubs / third-party planners that don't expose the
                # goldfive-specific entry point — the generic path is
                # better than no refine at all.
                refine_steer = getattr(planner, "refine_steer", None)
                if callable(refine_steer):
                    revised = await refine_steer(
                        plan=session.plan,
                        drift=drift,
                        goals=list(session.goals),
                        available_agents=available_agents,
                    )
                elif _planner_refine_accepts_available_agents(planner):
                    revised = await planner.refine(
                        plan=session.plan,
                        drift=drift,
                        goals=list(session.goals),
                        available_agents=available_agents,
                    )
                else:
                    revised = await planner.refine(
                        plan=session.plan,
                        drift=drift,
                        goals=list(session.goals),
                    )
            except RefineExhausted as exc:
                # goldfive#271: planner explicitly signalled handler
                # exhaustion. Pause for human intervention.
                log.info(
                    "DefaultSteerer._promote_drift_to_steer: refine raised "
                    "RefineExhausted for kind=%s task=%r: %s",
                    drift.kind.value,
                    drift.current_task_id,
                    exc,
                )
                await self._steerer.plans._emit_refine_failed(
                    session,
                    drift,
                    attempt_id=attempt_id,
                    failure_kind="refine_exhausted",
                    reason=str(exc) or "planner signalled handler exhaustion",
                    detail="",
                )
                await self._emit_handler_exhausted_escalation(drift, session)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "DefaultSteerer._promote_drift_to_steer: refine raised %s; plan unchanged",
                    exc,
                )
                await self._steerer.plans._emit_refine_failed(
                    session,
                    drift,
                    attempt_id=attempt_id,
                    failure_kind="llm_error",
                    reason=str(exc),
                    detail=type(exc).__name__,
                )
                await self._escalate_refine_failure_as_critical_drift(
                    session, drift, reason=str(exc)
                )
                await self._record_refine_outcome(session, drift, succeeded=False)
                return
            except BaseException as exc:  # noqa: BLE001
                # Phase 3.5 (CANCELLATION-CONTRACT.md §C4): ``CancelledError``
                # bypasses the ``except Exception`` branch above. Emit the
                # paired ``refine_failed`` so cancelled goldfive-steer refines
                # do not leave sinks with an unmatched ``refine_attempted``.
                # Re-raise to preserve asyncio cancellation propagation.
                await self._steerer.plans._emit_refine_failed(
                    session,
                    drift,
                    attempt_id=attempt_id,
                    failure_kind="cancelled",
                    reason=f"refine cancelled: {type(exc).__name__}",
                    detail=type(exc).__name__,
                )
                # Phase 3.5 tripwire compliance marker (§1.2 form).
                _state_audit.mark_stash_completed()
                raise
            finally:
                self._steerer._active_session_var.reset(_active_session_token)
        if revised is None:
            # iter-12 (#204): mirror the ``_handle_drift`` graceful
            # fallback — refine returning None means the planner has
            # exhausted its internal retry budget; escalate to
            # HUMAN_INTERVENTION_REQUIRED rather than emitting a
            # recursing CRITICAL follow-up drift that would eventually
            # abort the run.
            log.warning(
                "DefaultSteerer._promote_drift_to_steer: refine returned None; "
                "plan unchanged — escalating to HUMAN_INTERVENTION_REQUIRED"
            )
            await self._steerer.plans._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="parse_error",
                reason="planner returned no revised plan",
                detail="",
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # I4 fix: fold runtime terminal statuses from the prior plan
        # onto the revised plan BEFORE validation (see _handle_drift
        # for the full rationale). goldfive#247: returns a NEW Plan.
        revised = self._steerer.plans._fold_runtime_terminal_statuses(revised, session.plan)
        try:
            revised.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            # iter-12 (#204): mirror the ``_handle_drift`` graceful
            # fallback — keep the SCHEMA_VIOLATION emission at INFO
            # severity for operator/sink observability (does NOT
            # recurse through ``_handle_drift``) and escalate to
            # HUMAN_INTERVENTION_REQUIRED. The actionable signal is
            # the paired ``refine_failed(validator_rejected)`` envelope
            # plus the escalation drift.
            await self._steerer.plans._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="validator_rejected",
                reason=f"plan validation failed: {exc}",
                detail=type(exc).__name__,
            )
            await self._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.INFO,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                    authored_by="goldfive",
                ),
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # No-op revision rejection (goldfive#271). Same handler-
        # exhaustion semantics as ``_handle_drift`` — a structurally
        # identical plan means the planner cannot make progress on this
        # drift; escalate to HUMAN_INTERVENTION_REQUIRED.
        if self._steerer.plans._plans_structurally_identical(session.plan, revised):
            log.info(
                "no-op refine_steer revision skipped (kind=%s task=%r); "
                "escalating to HUMAN_INTERVENTION_REQUIRED",
                drift.kind.value,
                drift.current_task_id,
            )
            await self._steerer.plans._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="no_op_revision",
                reason="planner returned structurally identical plan",
                detail="",
            )
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        await self._record_refine_outcome(session, drift, succeeded=True)
        prev_plan = session.plan
        # goldfive#247: rebind to the stamped instance.
        # goldfive#255: thread ``was_installed`` into PlanRevised.dry_run.
        revised, was_installed = self._steerer.plans._apply_revision(session, revised, drift)
        # Cancel the in-flight coordinator invocation now that
        # ``refine_steer`` produced a superseding plan (goldfive#271
        # follow-up — v15 concurrent-invocation bug). This is the
        # path that empirically motivated the fix: a goldfive-steer-
        # eligible drift (PLAN_DIVERGENCE / OFF_TOPIC / …) at WARNING
        # severity got refined while the coordinator's LLM call kept
        # running for the full ``refine_steer`` duration, generating
        # contaminated output that triggered more drift. Cancelling
        # here preempts the in-flight LLM call so its remaining
        # output can't loop the refine.
        #
        # Order: cancel BEFORE PlanRevised emit so the synthetic
        # InvocationCancelled sink event lands adjacent to the
        # revision and operators can correlate the two on the
        # gantt timeline.
        await self._cancel_inflight_for_revision(drift, session)
        await self._steerer.plans._emit_plan_revised(
            session,
            revised,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
            dry_run=not was_installed,
        )
        # Audit #402 fix: dispatch the ``GOLDFIVE_STEER`` ControlMessage
        # AFTER ``_emit_plan_revised`` has swapped ``session.plan`` to
        # the revised version. ``_dispatch_goldfive_steer_control``
        # re-reads ``session.plan`` to derive ``replacement_task_ids``,
        # so dispatching here ensures the payload points the
        # executor's restart at the NEW plan's PENDING tasks rather
        # than the prior plan's (which may have been removed /
        # cancelled by the just-installed revision). Mirrors the
        # ordering in :meth:`_handle_drift`'s CANCEL_REINVOKE branch.
        await self._dispatch_goldfive_steer_control(
            drift, session, body_override=body
        )

    @staticmethod
    def _compose_goldfive_steer_body(drift: DriftEvent) -> str:
        """Derive the steer body for a goldfive-promoted drift.

        Prefers ``drift.detail`` verbatim — the reasoning judge and
        other LLM-as-a-judge paths already emit human-readable reasons
        like "agent acknowledged discrepancy but chose to adopt
        expanded topic" that are directly usable as a corrective. When
        ``detail`` is empty, synthesise a generic template from the
        drift's kind / severity / task context.
        """
        detail = str(getattr(drift, "detail", "") or "").strip()
        if detail:
            return detail
        task_id = drift.current_task_id or "the current task"
        return (
            f"Goldfive detected {drift.kind.name} drift "
            f"(severity={drift.severity.name}). The preceding agent "
            f"output did not match the task: {task_id}. Discard prior "
            "work on this task and proceed with the corrective plan."
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
        bookkeeping.

        Called from :meth:`handle_drift` and
        :meth:`apply_user_steer_with_plan` just before
        ``_emit_drift_detected`` and well before any plan install so:

        1. The ``goldfive.active_steer.*`` keys are set so downstream
           observers see the steer before the drift event.
        2. The source annotation / control id is appended to
           ``goldfive.processed_steer_ids`` so a retry or UI double-fire
           of the same STEER is a no-op (goldfive#171 dedupe).

        Never raises.

        Phase 4 (goldfive#271): goal synthesis was previously done
        here via ``planner.synthesize_goal_from_steer`` plus a
        regex-based qualification-merge post-process. That is now the
        :meth:`Planner.handle_turn` LLM's job — it produces the
        revised plan with qualifications already merged in one shot.
        This method retains only the bookkeeping-side effects.
        """
        # Recover the raw body + operator author from the originating
        # ControlMessage when it's available on drift.raw (goldfive#171).
        # Falling back to drift.detail preserves back-compat for tests
        # that synthesize a USER_STEER DriftEvent directly without a
        # ControlMessage behind it.
        raw_body, author, steer_id = self._unpack_steer_context(drift)
        body = raw_body.strip()
        # Stamp the active_steer keys regardless so readers see "a
        # steer is active as of turn N". ``at_turn`` uses the
        # session's monotonic sequence counter which increments on
        # every emitted event — a cheap, always-available "turn"
        # proxy.
        at_turn = getattr(session, "_next_sequence", 0) or 0
        try:
            _ostate.set_active_steer(
                session.state,
                body=body,
                at_turn=at_turn,
                author=author,
                source="user",
            )
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

    # ------------------------------------------------------------------
    # Refine-outcome bookkeeping (goldfive#215 iter-8 P2)
    # ------------------------------------------------------------------

    async def _record_refine_outcome(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        succeeded: bool,
    ) -> None:
        """Record the outcome of a refine attempt for ``(kind, task)``.

        On ``succeeded=True`` writes a ``RefineOutcome(state="succeeded",
        fail_count=0)`` entry. The "succeeded" state still encodes the
        "attempted" signal so a follow-up same-(kind, task) drift on
        the same turn skips refine — the prior refine already produced
        a landed revision, re-running it is a no-op replay.

        On ``succeeded=False`` increments ``fail_count`` (or initialises
        to 1 if no prior failure entry) and, when the count crosses
        :attr:`DefaultSteerer.REFINE_FAILURE_THRESHOLD`, marks the
        offending task FAILED (non-recoverable) and emits a CRITICAL
        ``REPEATED_FAILURE`` drift directly (NOT through
        :meth:`handle_drift` — the REPEATED_FAILURE drift keys on a
        different (kind, task) tuple than the source so it does not
        feed back into this counter).

        ``USER_STEER`` / ``USER_CANCEL`` / ``GOAL_DRIFT`` bypass the
        write entirely — operator intent must always be honoured and
        trajectory-level drifts have their own rate limiters.
        """
        if drift.kind in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            return
        key = (drift.kind.value, drift.current_task_id or "")
        if succeeded:
            session.refine_outcomes[key] = RefineOutcome(state="succeeded", fail_count=0)
            return
        prior = session.refine_outcomes.get(key)
        new_count = (prior.fail_count + 1) if prior is not None and prior.state == "failed" else 1
        session.refine_outcomes[key] = RefineOutcome(state="failed", fail_count=new_count)
        if new_count < self._steerer.REFINE_FAILURE_THRESHOLD:
            return
        # Crossed the threshold: mark the offending task FAILED (which
        # routes through _handle_drift on a TASK_FAILED_FATAL key —
        # different (kind, task) tuple, so no recursion into this
        # counter) and emit REPEATED_FAILURE directly via
        # _emit_drift_detected (NOT _handle_drift, which would try to
        # refine again on the fresh drift). See TASK-LIFECYCLE.md §7.3.
        task_id = drift.current_task_id
        reason = f"refine repeatedly failed for {drift.kind.value}"
        if task_id:
            await self._steerer.tasks.mark_task_failed(
                task_id,
                reason=reason,
                recoverable=False,
                session=session,
            )
        repeated = DriftEvent(
            kind=DriftKind.REPEATED_FAILURE,
            severity=DriftSeverity.CRITICAL,
            detail=(
                f"refine failed {new_count} consecutive times for "
                f"{drift.kind.value} (task {task_id or 'n/a'})"
            ),
            current_task_id=task_id,
            current_agent_id=drift.current_agent_id,
        )
        await self._emit_drift_detected(session, repeated)

    def reset_for_turn(self, session: Session) -> None:
        """Clear per-turn refine-outcome bookkeeping.

        Wired from :meth:`Runner.run` immediately after the
        ``run_started`` event so each turn starts with an empty
        outcome table. The (kind, task) retry budget is naturally
        per-turn — a wedged drift from a prior turn should not
        carry over its failure count and short-circuit a fresh
        refine attempt on the new turn.
        """
        session.refine_outcomes.clear()

    def _occurrence_count_for_ladder(self, session: Session, drift: DriftEvent) -> int:
        """Return the per-(kind, task) failure count consumed by the ladder.

        Maps ``RefineOutcome`` back onto the int the
        :meth:`_ladder_level_for` table reads. ``"succeeded"`` returns
        ``0`` so a fresh same-(kind, task) drift is treated as the
        first occurrence (the gate above the ladder will short-circuit
        anyway, but keeping the ladder invariant intact is cheaper
        than re-deriving the ``is_repeat`` semantics inside the ladder).
        """
        outcome = session.refine_outcomes.get((drift.kind.value, drift.current_task_id or ""))
        if outcome is None or outcome.state == "succeeded":
            return 0
        return outcome.fail_count

    async def _escalate_refine_failure_as_critical_drift(
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
