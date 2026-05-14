"""Drift event lifecycle + classification + observation helpers for :class:`DefaultSteerer`.

Extracted from :mod:`goldfive.steerer` in Wave C of the steerer split.
Buckets **3a** (observability primitives) and **3b** (observation entry
points + judge orchestration) now live here; bucket **3c** (dispatch +
ladder + promotion) remains on :class:`DefaultSteerer` and is the next
follow-up PR.

This module owns the drift-event observability + classification surface
plus the observation entry points (``observe`` / ``observe_reasoning``),
the background judge orchestration (``_run_judge_background`` /
``_run_goal_drift_judge_background`` / ``_spawn_*_background``), the
reflective self-progress check (``maybe_run_reflective_check`` /
``note_llm_call`` / ``_emit_reflective_failure``), the GOAL_DRIFT
trajectory-level check (``maybe_run_goal_drift_check`` /
``note_agent_turn`` / ``_maybe_run_goal_drift_on_task_boundary``), and
the bounded note-buffer family (``note_agent_activity`` /
``note_tool_observation``). The dispatch + ladder + promotion machinery
(``_handle_drift`` / ``_promote_drift_to_steer`` / the intervention
ladder / cancel-inflight / late-drift gate) stays on
:class:`DefaultSteerer` for now — that is bucket 3c.

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

The module DOES NOT own (yet — follow-up PR)
--------------------------------------------

* :meth:`_handle_drift` (753 lines), the intervention ladder
  (:meth:`_ladder_level_for`, :attr:`_LADDER` / :attr:`_LADDER_BY_VALUE`),
  ladder dispatch (``_dispatch_nudge`` / ``_dispatch_goldfive_steer_control``
  / ``_dispatch_goldfive_pause_control`` / ``_dispatch_pause_escalate``),
  adapter cancel tagging, late-drift gate
  (``_is_late_drift_for_terminated_invocation``),
  ``request_invocation_cancel``, ``_cancel_inflight_for_revision``,
  promotion (``_should_promote_to_steer``, ``_promote_drift_to_steer``,
  ``_compose_goldfive_steer_body``), ``_apply_user_steer_state``,
  refine-outcome bookkeeping (``_record_refine_outcome`` /
  ``reset_for_turn`` / ``_occurrence_count_for_ladder`` /
  ``_escalate_refine_failure_as_critical_drift``), structural
  escalation (``_is_task_progress_stalled`` /
  ``_emit_progress_stalled_escalation`` /
  ``_emit_handler_exhausted_escalation``). Moved in **bucket 3c** —
  the largest and most fragile cut, kept separate so audit issues
  #402 (dispatch-before-plan-swap) and #403 (lock window) can be
  reviewed against a single focused diff.
* Task-status transitions — moved in bucket 1
  (:mod:`goldfive.task_state_machine`).
* Plan-revision install + refine observability — moved in bucket 2
  (:mod:`goldfive.plan_reviser`).

All cross-component calls go through the router back-reference passed
to :meth:`DriftObserver.__init__` (``self._steerer``). This keeps the
components decoupled and lets the router own the shared state
(sinks, adapter, ContextVar for the active session, plan locks,
``observation_only`` flag, config).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from goldfive import state_store as _ostate
from goldfive.drift import (
    classify_refusal,
    classify_stop_reason,
    classify_tool_error,
)
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Session,
    Task,
)

if TYPE_CHECKING:
    from goldfive.steerer import DefaultSteerer

# Shape of the opt-in reflective LLM callable. Re-exported here so the
# DriftObserver methods that own the reflective + goal-drift + reasoning
# judge plumbing can carry their typed signatures without round-tripping
# through :mod:`goldfive.steerer` (avoids a circular import).
ReflectiveCallLLM = Callable[[str, str, str], Awaitable[str]]

log = logging.getLogger(__name__)
# Tests + harmonograf consumers grep for ``"DefaultSteerer."`` prefixes
# and the ``goldfive.steerer`` logger name across log records the
# observation + judge surface emits. Keep emitting under that logger
# name so the contract survives the bucket-3b extraction byte-for-byte
# (cf. the structural ``"stale judge verdict"`` INFO line asserted by
# :file:`tests/test_judge_task_lifetime.py`). The module's other logs
# stay on the ``goldfive.drift_observer`` logger so log-routing /
# filtering scoped to this component still works.
_steerer_log = logging.getLogger("goldfive.steerer")


class DriftObserver:
    """Drift event observability + classification helpers.

    Constructed by :class:`DefaultSteerer` and held on
    ``DefaultSteerer._drift_observer``. The router delegates the
    drift-emit / detection / attribution surface to the matching method
    on this class via thin shims that forward arguments verbatim. Tests
    that historically poked the bare-attribute names on the steerer
    (``steerer._emit_drift_detected``, ``steerer._resolve_authored_by``,
    ``steerer.detect_drift``, etc.) keep working through those shims;
    nothing on the public surface changes.

    This is **bucket 3a** of the steerer split — the observability
    cluster. The dispatch + ladder + promotion machinery is owned by
    :class:`DefaultSteerer` for now and will migrate in subsequent
    follow-up PRs (3b, 3c).
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
        # ``_drift_kind_pb_value`` / ``_drift_severity_pb_value``) and
        # the cross-component cooperators (the bound adapter for
        # boundary cleanup, the dispatch / ladder methods that still
        # live on :class:`DefaultSteerer` and will migrate in
        # follow-up bucket-3 PRs).
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
        await self._steerer._handle_drift(drift, session)

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
        await self._steerer._dispatch_goldfive_pause_control(drift, session, reason=reason)
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
        await self._steerer._dispatch_goldfive_pause_control(drift, session, reason=reason)
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
            _steerer_log.debug("DefaultSteerer.observe: dropping duplicate STEER id=%s", steer_id)
            return
        drift = self._drift_from_control(event, session)
        if drift is None:
            # Route through the router so subclasses overriding
            # :meth:`DefaultSteerer.detect_drift` are still consulted
            # (legacy extension surface; the router's shim forwards
            # back into :meth:`DriftObserver.detect_drift` when the
            # subclass does not override).
            drift = self._steerer.detect_drift(event, session)
        if drift is None:
            return
        await self._steerer._handle_drift(drift, session)

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
            await self._steerer._handle_drift(drift, session)
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
            if self._steerer._is_late_drift_for_terminated_invocation(drift, session):
                _steerer_log.info(
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
            await self._steerer._handle_drift(drift, session)
        except asyncio.CancelledError:
            # Propagate cancellation so :meth:`shutdown` / event-loop
            # teardown can cleanly abort a still-running judge without
            # the WARNING log below muddying the signal.
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            _steerer_log.warning(
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
            _steerer_log.debug(
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
                _steerer_log.info(
                    "DefaultSteerer: recorded reasoning-extracted binding "
                    "agent=%r task=%r confidence=%.2f",
                    agent_name,
                    focused,
                    confidence,
                )
        except Exception as exc:  # noqa: BLE001 — never break the run
            _steerer_log.warning(
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
            _steerer_log.warning(
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
                _steerer_log.warning(
                    "DefaultSteerer.shutdown: %d background %s task(s) "
                    "exceeded %.2fs timeout; cancelled",
                    len(still_pending),
                    label,
                    float(timeout),
                )
            else:
                _steerer_log.debug(
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
            _steerer_log.warning(
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
            await self._steerer._handle_drift(drift, session)
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
            await self._steerer._handle_drift(drift, session)
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
            _steerer_log.debug("note_tool_observation: swallowed: %s", exc)

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
        await self._steerer._handle_drift(drift, session)

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
            _steerer_log.warning(
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
            _steerer_log.debug(
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
            await self._steerer._handle_drift(drift, session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            _steerer_log.warning(
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
