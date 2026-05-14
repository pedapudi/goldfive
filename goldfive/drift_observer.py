"""Drift event lifecycle + classification helpers for :class:`DefaultSteerer`.

Extracted from :mod:`goldfive.steerer` in Wave C of the steerer split —
**bucket 3a of 3** (the first cut of the DriftObserver split). This
module owns the drift-event observability and classification surface:
the ``DriftDetected`` emit path, lifecycle stamping (#271), source
attribution helpers, control-message-to-drift mapping, and the
content-based detectors. The dispatch + ladder + promotion machinery
(``_handle_drift`` / ``_promote_drift_to_steer`` / the intervention
ladder / cancel-inflight) stays on :class:`DefaultSteerer` for now —
those are the next two follow-up PRs in the bucket-3 family.

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

The module DOES NOT own (yet — follow-up PRs)
---------------------------------------------

* The dispatch entry points :meth:`observe` / :meth:`observe_reasoning`
  and the background-judge plumbing (``_run_judge_background``,
  ``_maybe_record_reasoning_binding``, ``_resolve_available_agents``,
  ``_maybe_take_reasoning_judge_slot``) — moved in **bucket 3b**.
* The reflective self-progress + GOAL_DRIFT periodic checks — moved
  in **bucket 3b**.
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

import json
import logging
import re
import time
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
)

if TYPE_CHECKING:
    from goldfive.steerer import DefaultSteerer

log = logging.getLogger(__name__)


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
