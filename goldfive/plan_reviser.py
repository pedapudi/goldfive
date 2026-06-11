"""Plan-revision install + refine-attempt bookkeeping for :class:`DefaultSteerer`.

Extracted from :mod:`goldfive.steerer` in Wave C of the steerer split
(bucket 2 of 3, after :class:`~goldfive.task_state_machine.TaskStateMachine`
landed in bucket 1). This module owns every code path that takes a
candidate :class:`~goldfive.types.Plan` and lands it onto
:attr:`Session.plan` — the canonical write path for plan installs.

Responsibilities
----------------

* The four public install entry points:

  - :meth:`install_initial_plan` — turn-1 install on a fresh session.
  - :meth:`install_revision_for_drift` — autonomous detector-promoted
    refine or planner-driven replan.
  - :meth:`install_revision_for_user_steer` — operator
    :class:`~goldfive.control.ControlMessage` STEER deliveries.
  - :meth:`install_user_steer` — the always-lands user-steer path with
    a deterministic minimum-evolution fallback (PLAN-LIFECYCLE.md §4.2.1).
  - :meth:`install_descriptive_growth` — reactive plan growth at
    delegation-observation time for unmatched delegations (goldfive#423
    PR 2). See ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.3 + §5
    Option D for the lock-acquiring synchronous growth contract.
  - :meth:`apply_user_steer_with_plan` — deprecated back-compat shim.

* The shared install pipeline :meth:`_install_with_drift`:
  ``DriftDetected`` emit → fold runtime terminals → validate →
  ``_apply_revision`` → cancel in-flight → ``_emit_plan_revised``.

* :meth:`_apply_revision` — the **observation-only gate** (goldfive#254 /
  #255 / #258 / #267). Stamp revision metadata; suppress the live
  ``set_session_plan`` write and ``last_addressed_revision_by_drift_key``
  watermark when the run is in observation-only mode AND the drift is not
  one of the three carve-outs (bootstrap / user-authored / discovery).
  Returns ``(plan, was_installed)`` so the caller threads
  ``dry_run = not was_installed`` into :meth:`_emit_plan_revised`.

* :meth:`_emit_plan_revised` — the 351-line plan-revision emit:
  supersedes integration, pending-corrections GC + queue,
  ``current_task_id`` repin, ``PlanRevised`` envelope build,
  diff + refine-context observability, paired correlation envelope
  (#a4), and three dry_run-gated mutation sites (#267) that suppress
  side effects on observation-only previews while keeping the
  ``PlanRevised`` event on the wire.

* :meth:`_integrate_correction_supersedes` and
  :meth:`_repin_current_task_on_supersedes` — DAG rewiring for
  CORRECT-kind supersedes (goldfive#251) and pin migration to
  replacement tasks (goldfive#237).

* :meth:`_fold_runtime_terminal_statuses` — the I4 fix
  (PR #371 + goldfive#247) ensuring out-of-band terminal transitions
  are not regressed by an LLM-produced revision.

* :meth:`_plans_structurally_identical` — no-op revision rejection
  (goldfive#271 / closes #305 loop pattern).

* :meth:`_build_minimal_steer_evolution` — the deterministic always-
  valid revision shape for user-steer fallback (PLAN-LIFECYCLE.md §4.2).

* Refine-attempt observability: :meth:`_emit_refine_attempted`,
  :meth:`_emit_refine_failed`, :meth:`observe_refine` (the
  ``Phase 3.5``-audited async context manager wrapping every
  ``planner.refine`` call), :meth:`_emit_plan_revised_correlation`,
  :meth:`_new_attempt_id`.

* :meth:`_get_plan_lock` + :meth:`_wait_plan_stable` — per-session
  plan-state mutation lock (goldfive a4) used by reporting handlers
  to coordinate with fire-and-forget judge-triggered refines (#254).

* :meth:`_build_refine_input_summary` / :meth:`_build_refine_output_summary`
  — refine-context observability summaries stamped onto
  ``PlanRevised``.

The module DOES NOT own:

* Drift classification / dispatch / refine-outcome counting (lives in
  :class:`~goldfive.drift_observer.DriftObserver`).
* Task-status transitions (lives in
  :class:`~goldfive.task_state_machine.TaskStateMachine`).
* The shared event-emission primitives (``_new_envelope`` / ``_emit``
  / pb-value helpers) — those live on the :class:`DefaultSteerer`
  router because every component emits through them.

All cross-component calls go through the router back-reference passed
to :meth:`PlanReviser.__init__` (``self._steerer``). This keeps the
three components decoupled and lets the router own the shared state
(sinks, planner, adapter, ContextVar for the active session, plan
locks, ``observation_only`` flag, config).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import uuid
import warnings
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from goldfive import _state_audit
from goldfive import state_store as _ostate
from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskKind,
    TaskStatus,
    add_tasks,
    bump_revision,
    channel_processor_active,
    discovery_identity_hash,
    replace_edges,
    set_session_plan,
)

if TYPE_CHECKING:
    from goldfive.steerer import DefaultSteerer

log = logging.getLogger(__name__)


class PlanReviser:
    """Plan-install + refine-attempt observability owner.

    Constructed by :class:`DefaultSteerer` and exposed publicly as
    ``DefaultSteerer.plans`` (goldfive#410). Callers — the Runner,
    executors, planners, tests — reach the ``install_*`` family
    directly as ``steerer.plans.install_X``.
    """

    def __init__(self, steerer: DefaultSteerer) -> None:
        # Back-reference to the router. Used to reach the shared
        # event-emission primitives (``_new_envelope`` / ``_emit`` /
        # ``_drift_*_pb_value``) and the cross-component cooperators:
        # ``steerer.drift`` for ``_emit_drift_detected`` /
        # ``_cancel_inflight_for_revision`` / ``_apply_user_steer_state``
        # / ``_unpack_steer_context`` / ``_resolve_authored_by`` /
        # ``_drift_annotation_id``, and ``steerer.tasks`` for
        # ``_emit_plan_revision_transitions`` when a revision changes
        # task statuses out-of-band.
        self._steerer = steerer

    def _ledger_mode(self) -> bool:
        """Return True iff ``SteeringConfig.plan_mode == "ledger"``.

        AGENCY-PRESERVATION.md Stage 3 PR 10. Reads through
        ``steerer._steering_config.plan_mode`` (the typed config), exactly
        as the pin path reads ``descriptive_growth_enabled``. Defensive:
        any read failure (custom steerer without a typed config, test
        stub) resolves to forecast mode, keeping the legacy behaviour.
        """
        try:
            cfg = getattr(self._steerer, "_steering_config", None)
            if cfg is None:
                return False
            return str(getattr(cfg, "plan_mode", "forecast")).strip().lower() == "ledger"
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Public install entry points (4 + 1 back-compat shim)
    # ------------------------------------------------------------------

    async def install_initial_plan(
        self,
        *,
        session: Session,
        plan: Plan,
        is_pivot: bool = False,
    ) -> bool:
        """Install ``plan`` as the very first revision (rev 1) of ``session.plan``.

        Used on turn 1 of a fresh conversation when ``session.plan`` is
        a :meth:`Plan.empty` seed. Emits :class:`PlanRevised` with
        ``revision_index = 1`` and **no** :class:`DriftDetected` event:
        installing the first plan is not a corrective intervention,
        and stamping a USER_STEER drift here was the category error
        Option A (goldfive#271 follow-up) eliminates.

        ``is_pivot`` (F5, goldfive#322 Layer 2 / #204): when ``True``,
        the caller has classified the user's intent as a PIVOT —
        replacement of the prior plan rather than a revision of it.
        The validator runs WITHOUT ``prior`` so Rule 6
        (terminal-task / terminal->terminal-edge preservation) does
        not gate the new plan against a structurally-unrelated
        predecessor. The runner sets this when
        :meth:`Planner.handle_turn` flagged ``replaces_prior`` on the
        produced plan.

        The internal ``DriftEvent`` placeholder this method passes to
        :meth:`_apply_revision` and :meth:`_emit_plan_revised` carries
        ``DriftKind.NEW_WORK_DISCOVERED`` (``severity=INFO``) so the
        :class:`PlanRevised` envelope's ``drift_kind`` field has a
        coherent value — that field is required by the proto and
        downstream consumers (harmonograf) read it for revision
        framing. The placeholder is **never emitted** as a
        ``DriftDetected``.

        Returns ``True`` on success, ``False`` on validation failure.
        Never raises.
        """
        try:
            if is_pivot:
                # Pivot: validate structurally only. Rule 6 (terminal
                # preservation) is intentionally skipped — the user is
                # replacing the prior plan, not revising it.
                #
                # No fold for pivots — the user is replacing the prior
                # plan; runtime terminal statuses from the discarded
                # plan are not relevant to the new sub-DAG.
                plan.validate(for_revision=True, prior=None)
            else:
                # I4 fix: fold runtime terminal statuses from the prior
                # plan onto the candidate before validation.
                # goldfive#247: returns a NEW Plan; assign so the caller
                # uses the folded variant downstream.
                plan = self._fold_runtime_terminal_statuses(plan, session.plan)
                plan.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            await self._steerer.drift._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.CRITICAL,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                    authored_by="goldfive",
                ),
            )
            return False
        # Placeholder drift used only to thread metadata through
        # :meth:`_apply_revision` / :meth:`_emit_plan_revised`. Not
        # emitted as a DriftDetected.
        placeholder = DriftEvent(
            kind=DriftKind.NEW_WORK_DISCOVERED,
            severity=DriftSeverity.INFO,
            detail="initial plan install",
            authored_by="goldfive",
        )
        prev_plan = session.plan
        # goldfive#247: rebind to the stamped instance.
        # goldfive#255: bootstrap installs always land — the carve-out
        # inside ``_apply_revision`` (``prev is None``) covers the
        # fresh-session case; the ``Plan.empty`` seed path bypasses the
        # gate via this method's contract (``install_initial_plan`` is
        # structural, never a corrective intervention).
        plan, was_installed = self._apply_revision(session, plan, placeholder)
        # No cancel-in-flight: nothing is running yet on the very
        # first install.
        await self._emit_plan_revised(
            session,
            plan,
            placeholder,
            prev_plan=prev_plan,
            attempt_id=None,
            dry_run=not was_installed,
        )
        return True

    async def install_revision_for_drift(
        self,
        *,
        session: Session,
        drift: DriftEvent,
        revised_plan: Plan,
    ) -> bool:
        """Install ``revised_plan`` in response to a real :class:`DriftEvent`.

        The general-purpose install path for non-user-steer revisions:
        an LLM-driven replan after the user's next-turn message
        (``DriftKind.NEW_WORK_DISCOVERED``), an autonomous
        detector-promoted refine, or any other drift-driven plan
        revision the caller has already classified.

        Pipeline:

        * :meth:`_emit_drift_detected` — ``DriftDetected`` carrying
          the **real** drift kind/severity/detail
        * validate against prior plan; emit ``SCHEMA_VIOLATION`` and
          return ``False`` on failure
        * :meth:`_apply_revision` — bump ``revision_index`` + stamp
          metadata
        * :meth:`_cancel_inflight_for_revision` — preempt any
          in-flight invocation, IF the drift's authority permits it
          (AGENCY-PRESERVATION.md PR 1: user-authored / hard-safety
          only under the default ``cancel_inflight_scope``; a
          goldfive-authored install lands for bookkeeping while the
          invocation runs to completion)
        * :meth:`_emit_plan_revised` — ``PlanRevised`` + the paired
          refine-attempted / -success sidecar envelopes

        Refuses :class:`DriftKind.USER_STEER` — callers must route
        genuine operator steers through
        :meth:`install_revision_for_user_steer` so the active_steer
        bookkeeping and dedupe fire correctly.

        Returns ``True`` on success, ``False`` on validation failure.
        Never raises.
        """
        if drift.kind is DriftKind.USER_STEER:
            raise ValueError(
                "install_revision_for_drift refuses USER_STEER drifts; "
                "use install_revision_for_user_steer for genuine "
                "operator-pushed STEER ControlMessages."
            )
        if not drift.authored_by:
            drift.authored_by = self._steerer.drift._resolve_authored_by(drift)
        return await self._install_with_drift(
            session=session,
            drift=drift,
            revised_plan=revised_plan,
            apply_user_steer_state=False,
        )

    async def install_revision_for_user_steer(
        self,
        *,
        session: Session,
        raw: Any,
        revised_plan: Plan,
    ) -> bool:
        """Install ``revised_plan`` in response to an operator
        :class:`~goldfive.control.ControlMessage` STEER.

        ``raw`` is the originating :class:`ControlMessage`; this method
        builds the ``USER_STEER`` :class:`DriftEvent` internally so
        callers cannot accidentally fabricate a USER_STEER from
        plumbing (the category error #199/#302 papered over).

        Pipeline:

        * :meth:`_apply_user_steer_state` — active_steer bookkeeping +
          dedup (always — every call here represents genuine operator
          action)
        * :meth:`_emit_drift_detected` — ``USER_STEER`` ``DriftDetected``
          with ``raw`` populated and ``authored_by="user"``
        * validate revised plan; emit ``SCHEMA_VIOLATION`` on failure
        * :meth:`_apply_revision` + :meth:`_cancel_inflight_for_revision`
          + :meth:`_emit_plan_revised`

        Returns ``True`` on success, ``False`` on validation failure.
        Never raises.
        """
        body, author, _dedupe = self._steerer.drift._unpack_steer_context(
            DriftEvent(
                kind=DriftKind.USER_STEER,
                severity=DriftSeverity.WARNING,
                raw=raw,
            )
        )
        detail = f"by {author}: {body}" if author else body
        drift = DriftEvent(
            kind=DriftKind.USER_STEER,
            severity=DriftSeverity.WARNING,
            detail=detail,
            raw=raw,
            authored_by="user",
        )
        return await self._install_with_drift(
            session=session,
            drift=drift,
            revised_plan=revised_plan,
            apply_user_steer_state=True,
        )

    async def install_user_steer(
        self,
        *,
        drift: DriftEvent,
        prior: Plan,
        llm_revision: Plan | None,
        session: Session,
    ) -> Plan:
        """Install a user-authored revision. ALWAYS returns a valid Plan.

        Contract (see ``docs/design/PLAN-LIFECYCLE.md`` §4.2.1): user-steer
        rejection is **structurally impossible**. The return type is
        ``Plan`` (never ``None``), and this method does not raise
        ``ValueError`` from validation. If the LLM-produced revision
        fails ``Plan.validate(for_revision=True, prior=...)``, this
        method falls back to the deterministic minimum evolution shape
        (per §4.2): preserve every terminal task verbatim, cancel every
        PENDING / RUNNING / BLOCKED task, drop edges incident to the
        cancelled set. The minimum is provably valid by construction.

        Order of preference:

        1. ``llm_revision`` if non-None and validates against ``prior``.
        2. :meth:`_build_minimal_steer_evolution` — deterministic, always
           valid, intentionally produces a plan with no PENDING tasks.

        The deterministic minimum lands the user's pivot as a clean
        terminal-only frontier; the next refine cycle or coordinator
        turn can populate the new sub-DAG. This is acceptable
        degradation — the turn does not abort. The contract sacrifices
        a bit of "the LLM's first attempt drove forward progress" for
        the much stronger "the user's intent ALWAYS lands".

        Side effects (regardless of which branch fires):

        * :meth:`_apply_user_steer_state` writes the
          ``goldfive.active_steer.*`` slot from ``drift``.
        * :meth:`_emit_drift_detected` emits the ``USER_STEER`` drift.
        * :meth:`_apply_revision` swaps ``session.plan`` and bumps
          ``revision_index``.
        * :meth:`_cancel_inflight_for_revision` preempts in-flight work
          (USER_STEER is user-authored, so the AGENCY-PRESERVATION PR-1
          authority gate always permits this cancel).
        * :meth:`_emit_plan_revised` fires ``PlanRevised``.

        The deterministic-fallback branch deliberately does NOT touch
        ``session.refine_outcomes`` — that table governs goldfive-
        authored autonomous refines (§4.5), not user-driven changes.
        A USER_STEER never escalates via REPEATED_FAILURE.

        Never raises.
        """
        # Normalise the drift's authored_by so downstream observability
        # (DriftDetected.authored_by) carries the right attribution.
        if not drift.authored_by:
            drift.authored_by = "user"
        # Branch 1: try the LLM's revision if it parses + validates.
        chosen: Plan | None = None
        if llm_revision is not None:
            # I4 fix: fold runtime terminal statuses from the prior plan
            # onto the LLM revision before validation. Without this, an
            # NOT_NEEDED reaped task that the LLM regressed to PENDING
            # would force the deterministic-minimum fallback even when
            # the LLM's *new* work was otherwise sound.
            # goldfive#247: fold returns a NEW Plan; rebind so the
            # validator + downstream selection see the folded variant.
            llm_revision = self._fold_runtime_terminal_statuses(llm_revision, prior)
            try:
                llm_revision.validate(for_revision=True, prior=prior)
                chosen = llm_revision
            except ValueError as exc:
                log.warning(
                    "DefaultSteerer.install_user_steer: LLM revision rejected "
                    "by validator (%s); falling back to deterministic minimum "
                    "evolution shape (PLAN-LIFECYCLE.md §4.2.1)",
                    exc,
                )
        # Branch 2: deterministic minimum. Always valid by construction.
        if chosen is None:
            chosen = self._build_minimal_steer_evolution(prior, drift)
        # Always run the user-steer state bookkeeping — every call to
        # this method represents a genuine operator action.
        await self._steerer.drift._apply_user_steer_state(drift, session)
        await self._steerer.drift._emit_drift_detected(session, drift)
        # No-op short-circuit: the deterministic minimum on a prior with
        # no PENDING/RUNNING/BLOCKED tasks degenerates to a structurally
        # identical plan. Skip the install (avoids a misleading
        # PlanRevised with empty diff) but still return ``prior`` so the
        # contract (always a Plan) holds.
        if self._plans_structurally_identical(prior, chosen):
            log.info(
                "DefaultSteerer.install_user_steer: deterministic minimum "
                "is structurally identical to prior (no mutable tasks to "
                "cancel); install skipped, returning prior plan"
            )
            return prior
        prev_plan = session.plan
        attempt_id = self._new_attempt_id()
        await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
        # goldfive#247: rebind to the stamped instance.
        # goldfive#255: user-authored revisions always land (the carve-out
        # inside ``_apply_revision`` honours ``drift.authored_by == "user"``)
        # so ``was_installed`` is True even under observation_only.
        chosen, was_installed = self._apply_revision(session, chosen, drift)
        await self._steerer.drift._cancel_inflight_for_revision(drift, session)
        await self._emit_plan_revised(
            session,
            chosen,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
            dry_run=not was_installed,
        )
        return chosen

    def _build_minimal_steer_evolution(
        self, prior: Plan, drift: DriftEvent
    ) -> Plan:
        """Construct the canonical evolution shape per PLAN-LIFECYCLE.md §4.2.

        Deterministic. Preserves terminal tasks verbatim (§3.1), cancels
        every PENDING / RUNNING / BLOCKED task (so they enter the
        absorbing CANCELLED terminal), and drops every edge incident to
        a cancelled task. The result always passes
        ``Plan.validate(for_revision=True, prior=prior)`` because:

        * Every prior-terminal task is preserved with the same status →
          §3.1 holds.
        * Every prior terminal->terminal edge is preserved verbatim →
          §3.2 holds.
        * No PENDING tasks remain → reachability invariant (§5 rule 7,
          goldfive#137) is vacuously satisfied (no PENDING task can
          have a CANCELLED predecessor because there ARE no PENDING
          tasks).
        * Edges only span surviving terminal endpoints → no dangling
          edges.

        Uses :func:`dataclasses.replace` so the prior plan and tasks
        are not mutated. The deriver caches Tasks by identity in a few
        places (Tier 2 #323 found that mutating shared Task references
        corrupts the cache); fresh copies sidestep that risk.

        ``drift`` is consulted only for revision metadata (kind /
        severity / detail go onto the new plan via
        :meth:`_apply_revision`). It is not strictly required here, but
        keeping the parameter mirrors the steerer's other revision
        builders and makes future extensions (e.g. tagging which task
        the steer named) easier.
        """
        _ = drift  # reserved for future per-task framing; see docstring
        new_tasks: list[Task] = []
        cancelled_ids: set[str] = set()
        for t in prior.tasks:
            if t.status.is_terminal:
                # Preserve verbatim — fresh copy so callers cannot
                # accidentally mutate the prior plan's task identity.
                new_tasks.append(dataclasses.replace(t))
            else:
                # PENDING / RUNNING / BLOCKED → CANCELLED. Stamp a
                # provenance reason so harmonograf's intervention view
                # can attribute the cancel to a user-steer rollover.
                cancelled = dataclasses.replace(
                    t,
                    status=TaskStatus.CANCELLED,
                    cancel_reason=f"user_steer_rollover:{drift.id}"
                    if getattr(drift, "id", "")
                    else "user_steer_rollover",
                )
                new_tasks.append(cancelled)
                cancelled_ids.add(t.id)
        # Edges: drop any edge incident to a cancelled task. The
        # surviving edges are exactly the prior terminal->terminal set
        # plus any pre-existing terminal->cancelled (now both terminal,
        # but we still drop those because the to-task transitioned in
        # this revision and §3.2 only freezes edges that were
        # terminal->terminal in PRIOR — not in the revision).
        # Simpler: drop any edge touching a cancelled-this-rev id.
        new_edges: list[TaskEdge] = []
        for e in prior.edges:
            if e.from_task_id in cancelled_ids or e.to_task_id in cancelled_ids:
                continue
            new_edges.append(dataclasses.replace(e))
        # Construct the revised plan. ``revision_index`` is bumped by
        # :meth:`_apply_revision`; ``revision_*`` metadata stamping
        # happens there too. We populate ``id`` / ``run_id`` /
        # ``goal_ids`` / ``summary`` from prior so identity stays
        # stable across the revision (the plan_id-stable-across-turns
        # invariant from goldfive#271 Phase 4).
        return Plan(
            id=prior.id,
            run_id=prior.run_id,
            goal_ids=tuple(prior.goal_ids),
            tasks=tuple(new_tasks),
            edges=tuple(new_edges),
            summary=prior.summary,
        )

    async def _install_with_drift(
        self,
        *,
        session: Session,
        drift: DriftEvent,
        revised_plan: Plan,
        apply_user_steer_state: bool,
    ) -> bool:
        """Shared install pipeline for the two drift-driven install APIs.

        Emits ``DriftDetected`` then validates + installs the revision
        + emits ``PlanRevised``. The ``apply_user_steer_state`` flag
        gates the ``goldfive.active_steer.*`` bookkeeping so genuine
        operator STEERs write the slot and other drift-driven
        installs do not.
        """
        if apply_user_steer_state:
            await self._steerer.drift._apply_user_steer_state(drift, session)
        await self._steerer.drift._emit_drift_detected(session, drift)
        # I4 fix: fold runtime terminal statuses from the prior plan
        # onto the revised plan BEFORE validation. This is the path
        # that NEW_WORK_DISCOVERED installs (Runner._install_revision)
        # and USER_STEER ControlMessage installs travel through, which
        # is where the v24 phantom-state regression was observed.
        # goldfive#247: returns a NEW Plan; rebind so validation +
        # _apply_revision below see the folded variant.
        revised_plan = self._fold_runtime_terminal_statuses(revised_plan, session.plan)
        try:
            revised_plan.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            await self._steerer.drift._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.CRITICAL,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                    authored_by="goldfive",
                ),
            )
            return False
        # No-op revision rejection (goldfive#271 — replaces the deleted
        # count cap). If the install would be structurally identical to
        # the prior plan (same task ids, edges, assignees, statuses),
        # skip the install entirely: bumping ``revision_index`` for an
        # unchanged plan would emit a misleading PlanRevised with no
        # actual diff. Returns False so the caller can surface the
        # no-op. INFO-level so operators see why the install dropped.
        if self._plans_structurally_identical(session.plan, revised_plan):
            log.info(
                "no-op revision skipped on _install_with_drift "
                "(kind=%s task=%r); install dropped",
                drift.kind.value,
                drift.current_task_id,
            )
            return False
        # Capture prev_plan BEFORE _apply_revision swaps it; the
        # PlanRevisionDiff sidecar in _emit_plan_revised diffs the
        # two.
        prev_plan = session.plan
        attempt_id = self._new_attempt_id()
        await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
        # goldfive#247: rebind to the stamped instance.
        # goldfive#255: thread ``was_installed`` into PlanRevised.dry_run.
        # ``install_revision_for_user_steer`` and the user-steer routing
        # in ``apply_user_steer_with_plan`` enter through here too — the
        # ``authored_by == "user"`` carve-out inside ``_apply_revision``
        # makes ``was_installed`` True for those even under observation_only.
        revised_plan, was_installed = self._apply_revision(
            session, revised_plan, drift
        )
        await self._steerer.drift._cancel_inflight_for_revision(drift, session)
        await self._emit_plan_revised(
            session,
            revised_plan,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
            dry_run=not was_installed,
        )
        return True

    async def apply_user_steer_with_plan(
        self,
        *,
        drift: DriftEvent,
        session: Session,
        revised_plan: Plan,
    ) -> bool:
        """Back-compat shim — prefer :meth:`install_revision_for_drift`
        or :meth:`install_revision_for_user_steer` instead.

        Routes based on ``drift.kind`` + ``drift.raw``:

        * ``USER_STEER`` with ``raw`` populated → routed to
          :meth:`install_revision_for_user_steer`. The ``raw`` from
          the supplied drift is forwarded; ``drift.detail`` /
          ``drift.authored_by`` are ignored (the new API rebuilds
          them from ``raw`` deterministically).
        * ``USER_STEER`` with ``raw is None`` → was the
          :meth:`Runner._install_revision` synthetic install path
          before Option A. The new Runner path no longer reaches this
          shim; callers in this state probably mean
          :meth:`install_initial_plan` (turn 1) or
          :meth:`install_revision_for_drift` with a real drift kind
          (turn N+1). Routed defensively to ``install_initial_plan``
          when ``session.plan`` is empty, otherwise to
          ``install_revision_for_drift`` with a synthesized
          ``NEW_WORK_DISCOVERED`` drift so legacy callers keep
          working — but a deprecation warning fires.
        * Any other drift kind → routed to
          :meth:`install_revision_for_drift`.

        Slated for removal once external callers migrate.
        """
        warnings.warn(
            "DefaultSteerer.apply_user_steer_with_plan is deprecated; "
            "use install_initial_plan / install_revision_for_drift / "
            "install_revision_for_user_steer (goldfive#271 Option A).",
            DeprecationWarning,
            stacklevel=2,
        )
        if drift.kind is DriftKind.USER_STEER and getattr(drift, "raw", None) is not None:
            return await self.install_revision_for_user_steer(
                session=session,
                raw=drift.raw,
                revised_plan=revised_plan,
            )
        if drift.kind is DriftKind.USER_STEER:
            # Legacy synthetic-install path. Pick the new-API equivalent.
            if session.plan is None or not session.plan.tasks:
                return await self.install_initial_plan(
                    session=session, plan=revised_plan
                )
            replan_drift = DriftEvent(
                kind=DriftKind.NEW_WORK_DISCOVERED,
                severity=DriftSeverity.INFO,
                detail=drift.detail,
                authored_by="goldfive",
            )
            return await self.install_revision_for_drift(
                session=session,
                drift=replan_drift,
                revised_plan=revised_plan,
            )
        return await self.install_revision_for_drift(
            session=session, drift=drift, revised_plan=revised_plan
        )

    # ------------------------------------------------------------------
    # Descriptive growth — synchronous, lock-acquiring plan growth at
    # delegation_observed time (goldfive#423 PR 2).
    # ------------------------------------------------------------------

    async def install_descriptive_growth(
        self,
        session: Session,
        *,
        agent_name: str,
        tool_args_json: str,
        delegation_event_id: str = "",
        title: str | None = None,
        description: str | None = None,
    ) -> Task:
        """Grow ``session.plan`` with a ``discovered=True`` task and return it.

        The descriptive-growth fallback for unmatched delegations (design
        doc §4.3 + §5 Option D). Synthesises a new :class:`Task` carrying
        ``discovered=True`` and a stable
        :attr:`Task.discovery_identity_hash` computed from
        ``(agent_name, tool_args_json)`` via the §4.3.0 helper, then
        installs it onto ``session.plan`` under the per-session plan
        lock so the swap linearises against concurrent refines.

        Idempotent by ``discovery_identity_hash``. Inside the lock the
        method re-reads ``session.plan`` (the lock acquisition is the
        linearisation point — any concurrent refine has either completed
        before the read or is queued behind it) and checks for an
        existing NON-TERMINAL task with the same hash. If found, the
        existing task is returned and the plan is NOT grown. Two
        delegations of the same ``(agent, args-token-set)`` arriving
        simultaneously thus produce ONE discovered task, not two
        (§11.6 dedup linearisability). The dedup window follows the
        §11.1 TTL: once the discovered task reaches a terminal status,
        a fresh delegation with the same hash is a genuinely new unit
        of work and grows the plan again.

        The new task lands as an independent sub-DAG root: no predecessor
        edges, no supersedes link. Rule 7 of :meth:`Plan.validate` allows
        this because the predecessor set is empty.

        The ``tool_args_json`` argument MUST come from the observed
        :class:`~goldfive.types.DelegationObserved` event — i.e., from
        the agent-authored proto field, NOT from a goldfive-side
        intercept of agent state at pin time. See §13 for the underlying
        "adaptive, not predictive" principle. Empty / missing
        ``tool_args_json`` (legacy events from before PR 1) degrades to
        a coarser per-``(agent_name, "")`` hash, which is the §9
        forward-compat fallback PR 2 must tolerate.

        ``delegation_event_id`` is the originating
        ``DelegationObserved.id`` — threaded into the placeholder
        :class:`DriftEvent`'s ``id`` so harmonograf's intervention
        aggregator can correlate the resulting ``PlanRevised`` /
        ``DriftDetected`` envelopes back to the delegation that
        triggered the growth. Empty when the caller has no event id on
        hand; the helper still works.

        ``title`` / ``description`` are optional verbatim overrides
        (AGENCY-PRESERVATION.md PR 3). ``None`` (the default) derives
        both from ``tool_args_json`` exactly as before — pin-time and
        reconciler growth are unaffected. The agent-authored
        ``report_new_work_discovered`` reroute passes the agent's own
        title/description so absorb-as-growth keeps the reported text
        instead of an auto-derived ``agent: request`` label.

        Pipeline (under lock):

        1. Compute ``identity_hash`` from ``(agent_name, tool_args_json)``.
        2. Acquire ``_get_plan_lock(session)``.
        3. Re-read ``session.plan``. If any NON-TERMINAL task already
           carries ``discovery_identity_hash == identity_hash``,
           return it (dedup; no growth — §11.1 TTL).
        4. Build the new :class:`Task` with ``discovered=True``,
           ``discovery_identity_hash=identity_hash``,
           ``status=PENDING``, ``assignee_agent_id=agent_name``, and a
           title derived from ``agent_name``.
        5. Synthesise a ``NEW_WORK_DISCOVERED`` :class:`DriftEvent`
           (INFO severity, ``authored_by="goldfive"``) so the existing
           :meth:`install_revision_for_drift` carve-out at
           :meth:`_apply_revision` (the goldfive#258 discovery exemption
           from the observation-only gate) lets the revision land in
           both steering AND observation mode.
        6. Build the revised :class:`Plan` via :func:`add_tasks` and
           hand it to :meth:`install_revision_for_drift`, which owns the
           full ``PlanRevised`` + ``DriftDetected`` emit path and is
           already lock-aware.
        7. Release the lock; return the new task.

        Returns the discovered :class:`Task` (either the freshly
        installed one or the deduped pre-existing one). The caller can
        then re-pin ``session.current_task_id`` onto its id.

        Never raises on validation or install failures; on rejection
        returns a fresh Task instance representing the would-have-been
        discovery so callers can still pin observationally. (Production
        flow always lands — discovered tasks satisfy
        :meth:`Plan.validate` by construction; rejection means
        something pathological happened upstream.)

        Design ref: ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.3,
        §5 Option D (lock-acquiring), §11.6 (race-test acceptance), §13
        (adaptive not predictive).
        """
        from goldfive.conv import to_pb_plan
        from goldfive.events import build_plan_revision_diff

        identity_hash = discovery_identity_hash(agent_name, tool_args_json or None)
        # AGENCY-PRESERVATION.md Stage 3 PR 10 — ledger plan mode. In
        # ledger mode the descriptively-grown task is the means-level
        # DISCOVERED trajectory record; in forecast mode the ledger
        # taxonomy is unused, so the grown task keeps the FORECAST default
        # and forecast-mode behaviour stays byte-identical (the
        # ``discovered=True`` bool is the only overlay it carries, exactly
        # as before PR 10). Read off the steerer's typed config the same
        # way the pin path reads ``descriptive_growth_enabled``.
        discovered_kind = (
            TaskKind.DISCOVERED if self._ledger_mode() else TaskKind.FORECAST
        )
        # Explicit ``title`` / ``description`` overrides (None → derive
        # from the observed ``tool_args_json`` as before) let the
        # agent-authored ``report_new_work_discovered`` reroute
        # (AGENCY-PRESERVATION.md PR 3) preserve the agent's own verbatim
        # title/description while still landing as a discovered ledger
        # task. The identity hash still keys on ``(agent_name,
        # tool_args_json)``, so callers wanting per-report dedup encode
        # the distinguishing fields into ``tool_args_json``. Pin-time and
        # reconciler callers pass neither and keep the derived titles.
        title = (
            title
            if title is not None
            else self._derive_discovered_task_title(agent_name, tool_args_json)
        )
        description = (
            description
            if description is not None
            else self._derive_discovered_task_description(tool_args_json)
        )
        # Mint a fresh id so two concurrent growths on the same
        # (agent, args-token-set) cannot collide on plan-task id even
        # if both miss the dedup check (the lock makes them sequential
        # — the second will find the first inside the lock — but we
        # belt-and-braces the id space anyway).
        new_task_id = f"discovered-{uuid.uuid4().hex[:12]}"

        # The placeholder DriftEvent threads metadata through the
        # PlanRevised + DriftDetected emit below. INFO severity per
        # design doc §4.6: framework-synthesised discoveries are
        # observational, not corrective.
        drift = DriftEvent(
            kind=DriftKind.NEW_WORK_DISCOVERED,
            severity=DriftSeverity.INFO,
            detail=(
                f"descriptive growth: agent={agent_name!r} "
                f"hash={identity_hash} task_id={new_task_id}"
            ),
            current_task_id=new_task_id,
            current_agent_id=agent_name or "",
            authored_by="goldfive",
        )
        if delegation_event_id:
            try:
                drift.id = delegation_event_id
            except Exception:  # noqa: BLE001 — defensive
                pass

        # Captured-after-lock state used by the OFF-lock PlanRevised
        # emit. Initialised to "no install" so an early dedup return
        # cleanly skips the emit.
        installed_task: Task | None = None
        revised_plan: Plan | None = None
        prev_plan: Plan | None = None

        lock = self._get_plan_lock(session)
        async with lock:
            # Linearisation point: any concurrent refine has either
            # completed before this read or is queued behind it. Two
            # simultaneous descriptive-growth calls also serialise here.
            current_plan = session.plan
            if current_plan is not None:
                for existing in current_plan.tasks:
                    if (
                        getattr(existing, "discovery_identity_hash", "") or ""
                    ) == identity_hash and bool(
                        getattr(existing, "discovered", False)
                    ):
                        # Dedup TTL (design doc §11.1): the window is
                        # "until the discovered task reaches a terminal
                        # status". A fresh delegation matching a TERMINAL
                        # discovered task is a genuinely new unit of work
                        # and grows the plan again.
                        if (
                            getattr(existing, "status", None)
                            in TERMINAL_TASK_STATUSES
                        ):
                            continue
                        # Dedup hit — a prior delegation already grew the
                        # plan for this (agent, args-token-set). Re-pin
                        # to the existing task; no growth.
                        log.info(
                            "DefaultSteerer.install_descriptive_growth: "
                            "dedup hit on identity_hash=%s — reusing "
                            "existing discovered task id=%s for agent=%r",
                            identity_hash,
                            existing.id,
                            agent_name,
                        )
                        return existing

            new_task = Task(
                id=new_task_id,
                title=title,
                description=description,
                assignee_agent_id=agent_name or "",
                status=TaskStatus.PENDING,
                discovered=True,
                discovery_identity_hash=identity_hash,
                kind=discovered_kind,
            )

            if current_plan is None:
                # Defensive: no prior plan on the session. The runner
                # seeds session.plan with Plan.empty() before turn 1
                # in normal flow, so this branch is unusual — but we
                # still produce a single-task plan so the discovered
                # task lands as the seed.
                revised_plan = Plan(
                    id=f"discovered-plan-{uuid.uuid4().hex[:12]}",
                    run_id=session.run_id,
                    goal_ids=tuple(g.id for g in session.goals),
                    tasks=(new_task,),
                    edges=(),
                    revision_index=1,
                )
            else:
                # Sub-DAG root: no predecessor edges, no supersedes
                # link. Rule 7 of Plan.validate allows this because the
                # predecessor set is empty (design doc §4.3 closing
                # paragraph).
                grown = add_tasks(current_plan, [new_task])
                revised_plan = bump_revision(
                    grown,
                    revision_index=current_plan.revision_index + 1,
                    revision_kind=drift.kind.value,
                    revision_severity=drift.severity.value,
                    revision_reason=drift.detail,
                    revision_trigger_event_id=str(getattr(drift, "id", "") or ""),
                )

            # Validate the revised plan before swapping. A discovered
            # task that adds no edges is provably valid by construction
            # (sub-DAG root); the validate is cheap and catches the
            # pathological case (e.g. a malformed agent_name producing
            # an empty task id) before mutating session state. On
            # rejection, log + return the would-have-been Task without
            # installing — the next delegation's dedup will fail and
            # re-attempt.
            try:
                revised_plan.validate(for_revision=True, prior=current_plan)
            except ValueError as exc:
                log.warning(
                    "DefaultSteerer.install_descriptive_growth: "
                    "validation failed (%s); skipping install for "
                    "agent=%r hash=%s",
                    exc,
                    agent_name,
                    identity_hash,
                )
                return new_task

            prev_plan = current_plan
            # SINGLE WRITER under the lock — set_session_plan + the
            # channel_processor envelope serialise this against any
            # concurrent refine, and the lock acquisition above ensures
            # _emit_plan_revised's lock holder cannot interleave. This
            # is the §5 Option D contract: single writer, inside the
            # lock, full stop.
            with channel_processor_active():
                set_session_plan(session, revised_plan)
            # Refresh the orchestration-state current plan id so
            # downstream reads see the revised id. Mirrors the slice of
            # _emit_plan_revised that owns this stamp post-#403.
            try:
                _ostate.set_current_plan(session.state, revised_plan)
            except Exception as exc:  # noqa: BLE001 — defensive
                log.debug(
                    "DefaultSteerer.install_descriptive_growth: "
                    "set_current_plan raised: %s",
                    exc,
                )
            installed_task = new_task

        # OFF-LOCK: emit PlanRevised + paired DriftDetected. The
        # snapshot we carry was captured inside the lock and cannot
        # tear. Fire-and-forget-style — observability cannot block
        # the delegation. Per design doc §5.1: "The paired PlanRevised
        # + DriftDetected emit is scheduled off-lock as fire-and-forget;
        # the snapshot it carries is captured inside the lock and
        # cannot tear."
        if installed_task is not None and revised_plan is not None:
            try:
                await self._steerer.drift._emit_drift_detected(session, drift)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer.install_descriptive_growth: "
                    "_emit_drift_detected raised: %s",
                    exc,
                )
            try:
                evt = self._steerer._new_envelope(session)
                evt.plan_revised.plan.CopyFrom(to_pb_plan(revised_plan))
                evt.plan_revised.drift_kind = (
                    self._steerer._drift_kind_pb_value(drift.kind)
                )
                evt.plan_revised.severity = (
                    self._steerer._drift_severity_pb_value(drift.severity)
                )
                evt.plan_revised.reason = drift.detail
                evt.plan_revised.revision_index = revised_plan.revision_index
                trig_id = str(getattr(drift, "id", "") or "")
                if trig_id:
                    evt.plan_revised.trigger_event_id = trig_id
                evt.plan_revised.diff.CopyFrom(
                    build_plan_revision_diff(prev_plan, revised_plan)
                )
                evt.plan_revised.refine_input_summary = (
                    self._build_refine_input_summary(drift, prev_plan)
                )
                evt.plan_revised.refine_output_summary = (
                    self._build_refine_output_summary(revised_plan)
                )
                evt.plan_revised.target_agent_id = drift.current_agent_id or ""
                # Discovery growth is observational by construction —
                # the work is already happening; we are describing it,
                # not proposing a revision. dry_run=False keeps
                # harmonograf rendering it as a real revision.
                evt.plan_revised.dry_run = False
                await self._steerer._emit(evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer.install_descriptive_growth: "
                    "PlanRevised emit raised: %s",
                    exc,
                )

        log.info(
            "DefaultSteerer.install_descriptive_growth: grew plan with "
            "discovered task id=%s agent=%r hash=%s",
            new_task_id,
            agent_name,
            identity_hash,
        )
        return installed_task if installed_task is not None else new_task

    @staticmethod
    def _derive_discovered_task_title(agent_name: str, tool_args_json: str) -> str:
        """Render a stable, human-readable title for a discovered task.

        Priority per design doc §4.3.1:

        1. The ``request`` / ``task`` / ``goal`` arg off ``tool_args_json``
           (the conventional ``AgentTool`` payload key), truncated to 80
           chars.
        2. Fallback: ``f"{agent_name}: discovered work"``.

        The §4.3.1 second tier (first reasoning trace via
        ``Steerer.observe_reasoning``) is left for a follow-up PR — at
        pin / delegation-observed time we do not yet have a reasoning
        trace from the discovered sub-agent.
        """
        import json as _json

        if tool_args_json:
            try:
                parsed = _json.loads(tool_args_json)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                for key in ("request", "task", "goal"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        snippet = val.strip()
                        if len(snippet) > 80:
                            snippet = snippet[:77].rstrip() + "..."
                        return f"{agent_name}: {snippet}" if agent_name else snippet
        return f"{agent_name}: discovered work" if agent_name else "discovered work"

    @staticmethod
    def _derive_discovered_task_description(tool_args_json: str) -> str:
        """Render a compact description from the observed tool_args.

        Truncated to 256 chars per design doc §4.3 pseudocode.
        ``tool_args_json`` is the raw JSON payload from the
        :class:`~goldfive.types.DelegationObserved` proto; we keep it
        verbatim (with truncation) so operators can see exactly what
        the agent invoked.
        """
        if not tool_args_json:
            return ""
        if len(tool_args_json) > 256:
            return tool_args_json[:253].rstrip() + "..."
        return tool_args_json

    # ------------------------------------------------------------------
    # Per-session plan lock + refine attempt observability
    # ------------------------------------------------------------------

    def _get_plan_lock(self, session: Session) -> asyncio.Lock:
        """Return the per-session plan-state mutation lock, creating on first use.

        Keyed by ``session.id``. Multiple Sessions on the same Steerer
        each get an independent lock so concurrent runs never serialise
        on each other. The dict is unbounded for the steerer's lifetime;
        live runs share a steerer with a small number of sessions so
        this is acceptable. (If a future use-case introduces churn —
        many short-lived sessions — add a cleanup hook on session end.)
        """
        sid = session.id or session.run_id or ""
        lock = self._steerer._plan_locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._steerer._plan_locks[sid] = lock
        return lock

    async def _wait_plan_stable(
        self,
        session: Session,
        *,
        timeout: float | None = 1.0,
    ) -> bool:
        """Block until the per-session plan-state mutation region is idle.

        Acquires + immediately releases the per-session plan lock so the
        caller observes either pre-revision or post-revision plan state
        — never a partial apply. Used by report_task_* handlers and
        ``_resolve_effective_task_id`` callers to coordinate with the
        fire-and-forget judge-triggered refines introduced in #254.

        Returns ``True`` when the wait completed cleanly; ``False`` when
        ``timeout`` elapsed (in which case the caller MUST proceed
        anyway — atomicity is best-effort, not a hard barrier — and the
        worst case degrades to the pre-fix racy read). The default
        timeout is intentionally short (1s): a refine's mutation region
        is bounded by a handful of in-memory operations, so a timeout
        here means something pathological is happening and blocking
        a report indefinitely is worse than a stale read.

        ``timeout=None`` waits forever. Pass a positive float to bound
        the wait. Passing ``timeout<=0`` returns immediately with the
        lock-free reading semantics (does not check the lock at all);
        callers wanting a strict barrier should use a positive timeout.
        """
        if timeout is not None and timeout <= 0:
            return True
        lock = self._get_plan_lock(session)
        if not lock.locked():
            return True
        try:
            if timeout is None:
                async with lock:
                    pass
                return True
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            try:
                pass
            finally:
                lock.release()
            return True
        except TimeoutError:
            log.warning(
                "DefaultSteerer._wait_plan_stable: timed out after %.2fs "
                "waiting for plan lock on session %s; proceeding with "
                "best-effort racy read",
                timeout,
                session.id,
            )
            return False

    @staticmethod
    def _new_attempt_id() -> str:
        """Mint a fresh refine-attempt UUID for correlation between
        ``refine_attempted`` and the paired ``refine_failed`` /
        ``plan_revised`` events.
        """
        return str(uuid.uuid4())

    async def _emit_refine_attempted(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        attempt_id: str,
    ) -> None:
        """Emit a ``refine_attempted`` dict envelope onto the sink bus.

        Fired at the start of a refine call (both the autonomous
        ``_handle_drift`` path and the goldfive-steer
        ``_promote_drift_to_steer`` path). Pairs with exactly one of
        ``refine_failed`` / ``plan_revised`` carrying the same
        ``attempt_id``. Dict envelope (not proto) — promote to proto
        when the Stream C (#256) follow-up gets prioritised.
        """
        from goldfive.events import emit, make_event

        drift_id = str(getattr(drift, "id", "") or "")
        payload = {
            "attempt_id": attempt_id,
            "drift_id": drift_id,
            "trigger_kind": drift.kind.value,
            "trigger_severity": drift.severity.value,
            "current_task_id": drift.current_task_id or "",
            "current_agent_id": drift.current_agent_id or "",
        }
        try:
            evt = make_event(
                session.run_id,
                session.next_sequence(),
                "refine_attempted",
                payload,
                session_id=session.id,
            )
            await emit(self._steerer._sinks, evt)
        except Exception as exc:  # noqa: BLE001 — observability must never break the run
            log.debug(
                "DefaultSteerer._emit_refine_attempted: failed to emit: %s",
                exc,
            )

    async def _emit_refine_failed(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        attempt_id: str,
        failure_kind: str,
        reason: str,
        detail: str = "",
    ) -> None:
        """Emit a ``refine_failed`` dict envelope onto the sink bus.

        ``failure_kind`` is one of ``parse_error`` / ``validator_rejected``
        / ``llm_error`` / ``other`` (string, not enum, so the surface is
        forward-compatible without proto changes). ``reason`` is a short
        human-readable summary; ``detail`` may carry a longer
        free-form payload (e.g. the validator's exception text).
        Crucially, this event is emitted WITHOUT bumping
        ``revision_index`` — the attempt_id disambiguates failures
        across otherwise-incrementing revisions.
        """
        from goldfive.events import emit, make_event

        drift_id = str(getattr(drift, "id", "") or "")
        payload = {
            "attempt_id": attempt_id,
            "drift_id": drift_id,
            "trigger_kind": drift.kind.value,
            "trigger_severity": drift.severity.value,
            "failure_kind": failure_kind,
            "reason": reason,
            "detail": detail,
            "current_task_id": drift.current_task_id or "",
            "current_agent_id": drift.current_agent_id or "",
        }
        try:
            evt = make_event(
                session.run_id,
                session.next_sequence(),
                "refine_failed",
                payload,
                session_id=session.id,
            )
            await emit(self._steerer._sinks, evt)
        except Exception as exc:  # noqa: BLE001 — observability must never break the run
            log.debug(
                "DefaultSteerer._emit_refine_failed: failed to emit: %s",
                exc,
            )

    @contextlib.asynccontextmanager
    async def observe_refine(
        self,
        session: Session,
        drift: DriftEvent,
    ) -> AsyncIterator[str]:
        """Async context manager that wraps a ``planner.refine`` call with
        observability emission.

        On enter:

        * Mints a fresh ``attempt_id``.
        * Stamps the per-async-task ``_active_session_var`` ContextVar
          to ``session`` so the planner's ``_span_ctx_provider`` resolves
          correctly (this is what powers the planner-side
          ``refine_orphaned_tasks`` emission and the
          ``GoldfiveLLMCallStart/End`` spans). ContextVar isolation
          keeps concurrent runs sharing one Steerer from stomping each
          other's session pointer.
        * Emits ``refine_attempted`` to the bound sinks.

        On exception:

        * Emits ``refine_failed`` with ``failure_kind="llm_error"``,
          stamped with the same ``attempt_id``, then re-raises.

        On clean exit (no exception):

        * Resets ``_active_session_var``.
        * Caller is responsible for emitting either ``plan_revised``
          (success) or ``refine_failed`` (returned ``None`` / validator
          rejected) — the helper has no way to introspect the caller's
          decision tree from here. Pair with :meth:`_emit_refine_failed`
          / ``_emit_plan_revised`` using the yielded ``attempt_id``.

        Used by:

        * :meth:`_handle_drift` / :meth:`_promote_drift_to_steer` —
          the steerer's own refine call sites.
        * :class:`~goldfive.executors.parallel.ParallelDAGExecutor._refine` —
          the executor-side refine fallback. Without this helper, the
          parallel path's refines emit no ``refine_attempted`` /
          ``refine_failed`` / ``refine_orphaned_tasks`` events, since
          they bypass the steerer's hand-rolled emission blocks.
        """
        attempt_id = self._new_attempt_id()
        # Setting the per-async-task ``_active_session_var`` ContextVar
        # before refine lets the planner's internal
        # ``_emit_refine_orphaned_tasks`` resolve a sink target via the
        # bound span-context provider. Without this, the planner's
        # validator computes orphans, logs the WARNING, but no sink event
        # lands — exactly the symptom Bug A describes.
        _active_session_token = self._steerer._active_session_var.set(session)
        # Phase 3.5 (goldfive#271) tripwire wrapper — see §C4. The
        # ``except BaseException: stash; raise`` arm below is the
        # compliance branch (CANCELLATION-CONTRACT.md §1.2).
        with _state_audit.cancellation_stash_audited("DefaultSteerer.observe_refine"):
            try:
                await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
                try:
                    yield attempt_id
                except Exception as exc:  # noqa: BLE001 — refine errors must not break observability
                    # Emit failure event with the same attempt_id so consumers
                    # can pair attempted ↔ failed. We do NOT swallow the
                    # exception — re-raise so the caller's existing error path
                    # (e.g. _escalate_refine_failure_as_critical_drift / fallback plans) runs.
                    await self._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="llm_error",
                        reason=str(exc),
                        detail=type(exc).__name__,
                    )
                    raise
                except BaseException as exc:  # noqa: BLE001
                    # Phase 3.5 (CANCELLATION-CONTRACT.md §C4): ``CancelledError``
                    # is a ``BaseException`` (not ``Exception``) since Py 3.8, so
                    # the ``except Exception`` branch above does NOT catch it. If
                    # a refine is cancelled mid-flight (e.g. ADK closes the
                    # runner, harness interrupts the loop) the paired
                    # ``refine_failed`` observability event would be skipped,
                    # leaving sinks with an unmatched ``refine_attempted``.
                    # Emit the pair-completing failure event AND re-raise so
                    # cancellation still propagates per the asyncio contract.
                    await self._emit_refine_failed(
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

    async def _emit_plan_revised_correlation(
        self,
        session: Session,
        revised: Plan,
        drift: DriftEvent,
        *,
        attempt_id: str,
    ) -> None:
        """Emit a ``plan_revised`` dict envelope stamped with ``attempt_id``.

        Companion to the proto ``PlanRevised`` event so dict-event
        consumers correlate successful refines with their preceding
        ``refine_attempted`` event. The proto event carries the full
        payload for primary consumers; this dict envelope is purely a
        correlation side-car. When the Stream C (#256) proto follow-up
        promotes ``attempt_id`` onto ``PlanRevised``, this emitter goes
        away.
        """
        from goldfive.events import emit, make_event

        drift_id = str(getattr(drift, "id", "") or "")
        payload = {
            "attempt_id": attempt_id,
            "drift_id": drift_id,
            "trigger_kind": drift.kind.value,
            "trigger_severity": drift.severity.value,
            "revision_index": int(revised.revision_index),
            "current_task_id": drift.current_task_id or "",
            "current_agent_id": drift.current_agent_id or "",
        }
        try:
            evt = make_event(
                session.run_id,
                session.next_sequence(),
                "plan_revised",
                payload,
                session_id=session.id,
            )
            await emit(self._steerer._sinks, evt)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._emit_plan_revised_correlation: failed to emit: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Revision-shape helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fold_runtime_terminal_statuses(revised: Plan, prior: Plan | None) -> Plan:
        """Fold runtime terminal statuses from ``prior`` onto ``revised``.

        The persistence-boundary fix for the I4 phantom-state class of
        bugs (escalation report iter_1 §I4, v24 session
        ``2a324f78``): runtime terminal transitions emitted out-of-band
        between revisions — the overlay-reaper's NOT_NEEDED reap, the
        SequentialExecutor's reachability-audit cancels, an explicit
        ``mark_task_*`` call from a coordinator's reporting-tool — all
        flip the live plan's task status, but the next
        ``planner.refine`` / ``planner.handle_turn`` invocation builds
        its candidate plan from the LLM's view, which may have lost or
        regressed those terminal statuses.

        For each task ``t`` in ``revised`` whose id matches a task in
        ``prior`` with a status in :data:`TERMINAL_TASK_STATUSES`:

        * If ``revised``'s entry is non-terminal (PENDING / RUNNING /
          BLOCKED), OVERWRITE its status with the prior terminal status
          and copy ``cancel_reason`` so the persisted snapshot matches
          what actually happened.
        * If ``revised`` already carries the same terminal status, no-op.
        * If ``revised`` carries a *different* terminal status — a
          genuine regression we must NOT silently rewrite — leave it
          alone so the validator catches it.

        Returns a NEW :class:`Plan` (goldfive#247: Plan is frozen). When
        no folds are needed the input is returned unchanged so callers
        can share the reference. The fold list is logged at INFO when
        non-empty for downstream observability.

        Anti-pattern note: this is **not** a validator relaxation. The
        validator (``Plan.validate(for_revision=True, prior=...)``)
        remains the source of truth for terminal-task preservation. The
        fold corrects the LLM's output to match runtime reality
        BEFORE validation runs, so the validator only fires on a true
        regression (e.g. terminal→different-terminal) rather than on
        an ordinary "the LLM forgot a NOT_NEEDED reap fired since its
        prompt was authored."
        """
        if prior is None or not getattr(prior, "tasks", None):
            return revised
        prior_terminal: dict[str, Task] = {
            t.id: t
            for t in prior.tasks
            if t.id and t.status in TERMINAL_TASK_STATUSES
        }
        if not prior_terminal:
            return revised
        new_tasks: list[Task] = []
        folded: list[str] = []
        for t in revised.tasks:
            prior_t = prior_terminal.get(t.id)
            if prior_t is None:
                new_tasks.append(t)
                continue
            if t.status is prior_t.status:
                new_tasks.append(t)
                continue
            if t.status in TERMINAL_TASK_STATUSES:
                # Different terminal in the revised plan — a genuine
                # regression. Do not silently rewrite; let the
                # validator surface it as SCHEMA_VIOLATION.
                new_tasks.append(t)
                continue
            # Non-terminal in revised, terminal in prior → fold.
            replacement = dataclasses.replace(
                t,
                status=prior_t.status,
                cancel_reason=t.cancel_reason or prior_t.cancel_reason,
            )
            new_tasks.append(replacement)
            folded.append(t.id)
        if not folded:
            return revised
        log.info(
            "DefaultSteerer._fold_runtime_terminal_statuses: "
            "folded %d task(s) from prior runtime state: %s",
            len(folded),
            ", ".join(folded),
        )
        return dataclasses.replace(revised, tasks=tuple(new_tasks))

    @staticmethod
    def _plans_structurally_identical(prior: Plan | None, revised: Plan) -> bool:
        """Return ``True`` iff ``revised`` has the same structural shape as ``prior``.

        goldfive#271 — no-op revision rejection (subsumes #188 / closes
        the post-#305 loop pattern). Compares task ids, edges, assignees,
        and statuses. Differences in plan id, revision metadata
        (``revision_index`` / ``revision_kind`` / ``revision_severity`` /
        ``revision_reason`` / ``revision_trigger_event_id``), summaries,
        timing predictions, descriptions, and span bindings are
        ignored — these can change without the plan actually meaning
        anything different to the executor.

        Returns ``False`` when ``prior`` is ``None`` (the seed case in
        :meth:`Runner._install_revision`).
        """
        if prior is None:
            return False
        # Tasks: id + assignee + status (in declared order — order
        # matters because executor scheduling reads tasks in list order
        # for the topological tie-breaker).
        if len(prior.tasks) != len(revised.tasks):
            return False
        for old, new in zip(prior.tasks, revised.tasks, strict=True):
            if old.id != new.id:
                return False
            if old.assignee_agent_id != new.assignee_agent_id:
                return False
            if old.status != new.status:
                return False
        # Edges: order-independent, structural set comparison.
        old_edges = {(e.from_task_id, e.to_task_id) for e in prior.edges}
        new_edges = {(e.from_task_id, e.to_task_id) for e in revised.edges}
        if old_edges != new_edges:
            return False
        return True

    @staticmethod
    def _integrate_correction_supersedes(revised: Plan) -> Plan:
        """Rewire DAG edges for every ``CORRECT``-kind supersedes link.

        goldfive#251 Option B topology. For a new task
        ``new.supersedes == old_id`` with ``new.supersedes_kind ==
        SupersessionKind.CORRECT``:

        * The old task is NOT marked superseded / hidden — it stays in
          the plan as a historical COMPLETED node.
        * An edge ``old -> new`` is added (unless already present), so
          the new correction-task has the old as its upstream.
        * Every existing edge ``old -> X`` for some X != new is
          rewritten to ``new -> X`` so downstream work that used to
          depend on the old task now flows through the correction.

        The in-revision edges from the refiner sometimes already
        reflect this topology (the LLM may emit the rewired shape);
        this method is idempotent and re-runnable in that case.

        Does nothing when no task carries a CORRECT-kind supersedes.
        REPLACE-kind links are intentionally left alone — the pre-#251
        behaviour (old task marked terminal / hidden by the refiner;
        downstream edges rewritten to the replacement) was already
        correct and this method does not touch that path.

        goldfive#247: Plan.edges is an immutable tuple of frozen
        :class:`TaskEdge`. The rewrite builds a fresh edge list and
        returns a new :class:`Plan` via :func:`replace_edges`. When no
        rewrite is needed the original is returned unchanged so callers
        can keep their reference.

        Runs BEFORE :meth:`_repin_current_task_on_supersedes` so that
        helper's downstream rewrites see the already-correct DAG.
        """
        if revised is None:
            return revised
        tasks_by_id: dict[str, Task] = {t.id: t for t in revised.tasks if t.id}
        corrections: list[tuple[str, str]] = []  # (old_id, new_id)
        for task in revised.tasks:
            if task.supersedes_kind is not SupersessionKind.CORRECT:
                continue
            old_id = (task.supersedes or "").strip()
            new_id = (task.id or "").strip()
            if not old_id or not new_id or old_id == new_id:
                continue
            if old_id not in tasks_by_id:
                # Structural validator will reject; skip the rewrite.
                continue
            corrections.append((old_id, new_id))
        if not corrections:
            return revised
        # Build a fresh edge list as plain tuples; coerce to TaskEdges
        # at the end via :func:`replace_edges` (which also handles
        # dedup-while-preserving-order).
        edges: list[tuple[str, str]] = [(e.from_task_id, e.to_task_id) for e in revised.edges]
        existing_edges: set[tuple[str, str]] = set(edges)
        for old_id, new_id in corrections:
            # 1. Ensure old -> new edge exists.
            if (old_id, new_id) not in existing_edges:
                edges.append((old_id, new_id))
                existing_edges.add((old_id, new_id))
            # 2. Rewrite outgoing edges of the old task to originate
            #    from the new (correction) task. Skip the old -> new
            #    edge we just ensured.
            for i, edge in enumerate(edges):
                frm, to = edge
                if frm != old_id:
                    continue
                if to == new_id:
                    continue
                # Avoid duplicating an edge that already exists from the
                # new task to the same downstream.
                if (new_id, to) in existing_edges:
                    # Mark for dedup below; same content as existing.
                    edges[i] = (new_id, to)
                    continue
                existing_edges.discard((old_id, to))
                edges[i] = (new_id, to)
                existing_edges.add((new_id, to))
        # Final dedup: rewriting may have produced structurally-duplicate
        # edges. Preserve insertion order while dropping repeats.
        seen: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str]] = []
        for e in edges:
            if e in seen:
                continue
            seen.add(e)
            deduped.append(e)
        return replace_edges(revised, deduped)

    def _repin_current_task_on_supersedes(
        self,
        session: Session,
        revised: Plan,
    ) -> None:
        """Re-pin ``current_task_id`` onto replacement tasks after revision.

        When a revision's tasks carry a non-empty ``supersedes`` link
        (goldfive#237), treat it as the explicit "this task replaces
        that one" signal that older heuristic id-suffix matching was
        unable to express. Walk the map and:

        * Update ``session.current_task_id`` if it matches a superseded
          id — so agent-facing reporting-tool calls land on the live
          replacement rather than the FAILED/CANCELLED original.
        * Update the goldfive orchestration ``session.state`` pin
          (``goldfive.current_task_id`` key) when it matches a
          superseded id. This is the key the reporting-handler fallback
          (:func:`goldfive.reporting._resolve_task_id`) reads when the
          LLM's tool call omits the arg.
        * Ask the bound adapter (if any) to rewrite any per-agent ADK
          ``session.state`` copies whose current-task pin matches a
          superseded id. Best-effort: adapters without the hook no-op.

        The supersession map is built fresh from ``revised`` every call
        so A→B→C chains across multiple revisions compose naturally
        (each refine sees B.supersedes=A at revision N and
        C.supersedes=B at revision N+1; we never need to chase
        transitive links because the pin can only point at one id at a
        time and each revision fires this hook independently).
        """
        if revised is None:
            return
        # Build fresh per-revision. Old -> new. A planner producing
        # `C.supersedes = B` in the SAME revision that also ages
        # `B.supersedes = A` is handled transitively: we follow the
        # chain from the current pin forward to the first task that is
        # NOT itself superseded within the revision. In practice the
        # chain is rarely >1 hop per revision but the loop is cheap.
        supersession: dict[str, str] = {}
        for task in getattr(revised, "tasks", None) or ():
            old_id = str(getattr(task, "supersedes", "") or "").strip()
            new_id = str(getattr(task, "id", "") or "").strip()
            if not old_id or not new_id or old_id == new_id:
                continue
            supersession[old_id] = new_id
        if not supersession:
            return

        def _resolve_chain(start: str) -> str:
            """Walk the supersession map from ``start`` to its latest end."""
            seen: set[str] = {start}
            current = start
            while current in supersession:
                nxt = supersession[current]
                if nxt in seen:
                    # Defensive: a cycle shouldn't exist but guard
                    # against an adversarial planner before looping.
                    break
                seen.add(nxt)
                current = nxt
            return current

        # 1. goldfive Session pin.
        pinned = str(getattr(session, "current_task_id", "") or "")
        if pinned and pinned in supersession:
            resolved = _resolve_chain(pinned)
            if resolved != pinned:
                log.info(
                    "goldfive#237: re-pinning session.current_task_id %s -> %s (supersedes)",
                    pinned,
                    resolved,
                )
                session.current_task_id = resolved

        # 2. goldfive orchestration session.state pin (the reporting-
        # tool fallback's source of truth). Use the canonical state key
        # so tests that inspect the state dict directly see the update.
        # Phase 1 of goldfive#271 — read through StateStore;
        # the write stays at this call site (Phase 2 migration target
        # per the catalog).
        state = getattr(session, "state", None)
        if isinstance(state, dict):
            from goldfive.state_store import StateStore

            store = StateStore.for_state(state)
            state_pinned_s = store.pin_current_task().strip()
            if state_pinned_s and state_pinned_s in supersession:
                resolved = _resolve_chain(state_pinned_s)
                if resolved != state_pinned_s:
                    log.info(
                        "goldfive#237: re-pinning session.state %s -> %s (supersedes)",
                        state_pinned_s,
                        resolved,
                    )
                    state[_ostate.KEY_CURRENT_TASK_ID] = resolved

        # 3. Per-agent ADK session.state copies (when the adapter
        # exposes a hook). Optional wiring: most test-path adapters
        # don't — we guard with hasattr and swallow exceptions so a
        # missing hook never breaks revision emission.
        adapter = self._steerer._adapter
        if adapter is None:
            return
        hook = getattr(adapter, "rewrite_pinned_task_ids", None)
        if not callable(hook):
            return
        try:
            hook(supersession)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "goldfive#237: adapter.rewrite_pinned_task_ids raised: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # The observation-only gate + the canonical PlanRevised emit
    # ------------------------------------------------------------------

    def _apply_revision(
        self, session: Session, revised: Plan, drift: DriftEvent
    ) -> tuple[Plan, bool]:
        """Stamp revision metadata and decide whether ``revised`` should install.

        Returns ``(revised, was_installed)``: ``was_installed`` is
        ``True`` iff the revision **should be** swapped onto
        ``session.plan`` (so the caller — and downstream
        :meth:`_emit_plan_revised` — can stamp ``dry_run`` on the
        emitted ``PlanRevised`` faithfully).

        goldfive#403: this method NO LONGER mutates session state. Pre-
        #403 it called :func:`set_session_plan`, stamped
        ``session.last_addressed_revision_by_drift_key``, and pushed
        ``_ostate.set_current_plan`` here. All three sites were
        OUTSIDE the per-session plan lock — the caller then awaited
        :meth:`_cancel_inflight_for_revision` (which yields the event
        loop) before :meth:`_emit_plan_revised` acquired the lock,
        leaving a partial-apply window where ``_wait_plan_stable``
        readers observed a bumped ``revision_index`` with the
        un-rewired (pre-supersedes) edge DAG. All three mutations now
        run inside :meth:`_emit_plan_revised`'s lock, gated on the
        ``was_installed`` decision encoded by this method's return
        value. The contract for direct callers (none in production —
        every caller of this method also routes to
        :meth:`_emit_plan_revised`) is now "compute, don't install";
        tests that previously asserted post-call ``session.plan``
        identity must also invoke :meth:`_emit_plan_revised` to
        observe the install.

        Preserves the existing ``revision_index`` monotonicity: the new
        plan's index is at least ``old.revision_index + 1``.

        Observation-only mode (goldfive#254 / #255 / #258): when
        ``observation_only`` is set the gate suppresses **only**
        goldfive-authored **corrective** drifts (``OFF_TOPIC``,
        ``GOAL_DRIFT``, ``LOOPING_*``, ``TASK_FAILED_*``, ``BLOCKED``,
        ``CAPABILITY_MISMATCH``, etc.). The carve-outs land as **real**
        revisions even under ``observation_only=True``:

        * **bootstrap** (``prev is None``) — a cold start with no prior
          plan on the session. Structural, not corrective.
        * **user-authored** (``drift.authored_by == "user"``) — genuine
          operator STEER ``ControlMessage`` deliveries always land. The
          operator has the authority to override observation mode.
        * **discovery** (``drift.kind is DriftKind.NEW_WORK_DISCOVERED``,
          goldfive#258) — a description of work the planner or a
          sub-agent reported / discovered, not a framework-driven
          correction. This covers two paths the ``prev is None``
          bootstrap predicate missed:

          - The runner's turn-1 install through
            :meth:`install_initial_plan`: the runner seeds
            ``session.plan = Plan.empty(run_id=...)`` before
            :meth:`Planner.handle_turn` runs, so by the time the
            placeholder ``NEW_WORK_DISCOVERED`` drift reaches
            ``_apply_revision`` ``prev`` is the empty seed (non-None).
          - The runner's turn N+1 replan through
            :meth:`install_revision_for_drift` with a
            ``NEW_WORK_DISCOVERED`` drift: the planner integrated new
            work from the user's fresh message. This is a description
            of what happened, not a corrective intervention.

          Sub-agent ``report_new_work_discovered`` calls fold into the
          same kind for the same reason.

        observation-only is about suppressing framework-driven
        **corrections** — not framework-driven **observability** of
        what the planner / agents are doing.

        When the gate fires, the revision metadata is STILL stamped (so
        the returned Plan accurately reflects the index/kind/severity
        the planner produced and downstream :meth:`_emit_plan_revised`
        can render a faithful preview), but the actual
        ``set_session_plan`` write to ``session.plan`` and the
        ``last_addressed_revision_by_drift_key`` stamp are SKIPPED —
        the live agent keeps reasoning against the prior plan. The
        paired ``PlanRevised`` event from :meth:`_emit_plan_revised`
        carries ``dry_run=True`` so consumers can distinguish a
        would-have-applied revision from a real one.

        goldfive#247: returns the post-stamp Plan that was actually
        installed onto :attr:`Session.plan`. Pre-#247 the function
        mutated ``revised`` in place AND ``session.plan = revised``, so
        callers who reused their local ``revised`` reference saw the
        stamped metadata. With frozen Plan, the stamp produces a NEW
        instance; the helper returns it so callers can rebind their
        local variable and pass the same instance to
        :meth:`_emit_plan_revised`.

        Phase 2.X / goldfive#271 Gap 2: log the install at INFO so the
        prior_plan_id → revised_plan_id transition is grep-able in the
        demo log. The validation E2E found 2 of 4 task_plans rows
        without corresponding plan events; without this log line a
        silent install (e.g. an exception in ``_emit_plan_revised``
        right after) leaves no goldfive-side trace of the swap.

        Defensive fold (I4 fix): re-applies
        :meth:`_fold_runtime_terminal_statuses` against ``session.plan``
        even though install paths fold before validation. Idempotent —
        a no-op if the caller already folded — but a last-line guard
        against any future install path that forgets to fold before
        calling here.
        """
        prev = session.plan
        # I4 fix (defensive): fold runtime terminal statuses even if the
        # caller already did. Idempotent — if every task's status
        # already matches prior's terminal, this is a no-op. Returns a
        # NEW Plan (goldfive#247: Plan is frozen).
        revised = self._fold_runtime_terminal_statuses(revised, prev)
        prior_id = (getattr(prev, "id", "") or "") if prev is not None else ""
        next_index = (prev.revision_index + 1) if prev is not None else 1
        # goldfive#247: Plan is frozen — derive a new instance with the
        # stamped revision metadata via :func:`bump_revision`. Preserves
        # caller-supplied non-empty values (matches the legacy
        # "only set if blank" guards).
        new_index = max(int(revised.revision_index), next_index)
        new_kind = revised.revision_kind or drift.kind.value
        new_severity = revised.revision_severity or drift.severity.value
        new_reason = revised.revision_reason or drift.detail
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
        new_trigger_id = revised.revision_trigger_event_id
        if not new_trigger_id:
            new_trigger_id = self._steerer.drift._drift_annotation_id(drift) or str(
                getattr(drift, "id", "") or ""
            )
        revised = bump_revision(
            revised,
            revision_index=new_index,
            revision_kind=new_kind,
            revision_severity=new_severity,
            revision_reason=new_reason,
            revision_trigger_event_id=new_trigger_id,
        )
        # goldfive#255 / #258: refine the observation-only gate so it
        # captures ONLY goldfive-authored corrective drifts. The three
        # carve-outs — bootstrap, user-authored, and discovery — are
        # named explicitly so a future reader can grep for the reasons
        # the gate does NOT fire:
        #
        # * bootstrap (``prev is None``) — cold session.
        # * user-authored (``drift.authored_by == "user"``) — operator
        #   STEER deliveries.
        # * discovery (``drift.kind is DriftKind.NEW_WORK_DISCOVERED``,
        #   #258) — the planner / a sub-agent describing new work, not
        #   a framework-driven correction. Covers both turn-1 installs
        #   through ``install_initial_plan`` (where ``prev`` is the
        #   ``Plan.empty()`` seed, NOT None) and turn N+1 replans
        #   through ``install_revision_for_drift``.
        is_bootstrap = prev is None
        is_user_authored = (drift.authored_by or "").lower() == "user"
        is_discovery = drift.kind is DriftKind.NEW_WORK_DISCOVERED
        gate_active = (
            (not is_bootstrap)
            and (not is_user_authored)
            and (not is_discovery)
            and (not self._steerer._should_inject())
        )
        if gate_active:
            log.info(
                "DefaultSteerer._apply_revision: observation_only=True — "
                "SKIPPING plan install (gate_active; bootstrap=%s user=%s "
                "discovery=%s). prior_plan_id=%s "
                "would_have_installed_plan_id=%s revision_index=%d "
                "drift_kind=%s",
                is_bootstrap,
                is_user_authored,
                is_discovery,
                prior_id[:16] or "<none>",
                (revised.id or "")[:16] or "<empty>",
                int(revised.revision_index),
                drift.kind.value,
            )
            # Observation-only: do NOT swap ``session.plan``, do NOT stamp
            # the per-(kind, target) addressed watermark, do NOT update the
            # orchestration-state current_plan pointer. The stamped
            # ``revised`` instance is still returned so
            # :meth:`_emit_plan_revised` can render the would-have-applied
            # preview into ``PlanRevised`` (with ``dry_run=True``).
            return revised, False
        log.info(
            "DefaultSteerer._apply_revision: prior_plan_id=%s "
            "revised_plan_id=%s revision_index=%d drift_kind=%s "
            "(decision=install; session.plan swap deferred to "
            "_emit_plan_revised under lock — goldfive#403)",
            prior_id[:16] or "<none>",
            (revised.id or "")[:16] or "<empty>",
            int(revised.revision_index),
            drift.kind.value,
        )
        # goldfive#403: do NOT mutate ``session.plan``,
        # ``session.last_addressed_revision_by_drift_key``, or
        # ``session.state`` here. Pre-#403 this method called
        # :func:`set_session_plan` immediately, stamped the per-(kind,
        # target) watermark, and pushed the orchestration-state pointer
        # — all OUTSIDE the per-session plan lock. The caller then
        # awaited :meth:`_cancel_inflight_for_revision` (which yields
        # the event loop) before :meth:`_emit_plan_revised` acquired
        # the lock, leaving a window where readers calling
        # :meth:`_wait_plan_stable` saw the bumped ``revision_index``
        # and stamped watermark but with the un-rewired (pre-supersedes)
        # edge DAG and the un-repinned ``current_task_id``. The
        # docstring on :meth:`_emit_plan_revised` promises that
        # ``_wait_plan_stable`` callers see pre- or post-revision state
        # but never a partial apply; the only way to honour that is to
        # defer every session-mutation site into the lock-protected
        # region of :meth:`_emit_plan_revised`. This method now returns
        # the stamped Plan plus the install decision; the caller is
        # responsible for routing it to :meth:`_emit_plan_revised`,
        # which performs the actual install atomically.
        return revised, True

    async def _emit_plan_revised(
        self,
        session: Session,
        revised: Plan,
        drift: DriftEvent,
        *,
        prev_plan: Plan | None = None,
        attempt_id: str | None = None,
        dry_run: bool | None = None,
    ) -> None:
        from goldfive._correction_injection import (
            clear_obsolete_corrections_on_revision,
            queue_corrections_for_revision,
        )
        from goldfive.conv import to_pb_plan
        from goldfive.events import build_plan_revision_diff

        # goldfive a4 / goldfive#403: serialise the consistency-critical
        # region of plan mutation. Held across EVERY session-mutation
        # site (``session.plan`` swap, watermark stamp,
        # orchestration-state pointer, supersedes-integration re-swap,
        # current_task_id repin, pending-corrections GC + queue) plus
        # the PlanRevised emit — NOT across ``planner.refine`` itself,
        # which the caller owns. Pre-#403 the first ``set_session_plan``
        # call lived in :meth:`_apply_revision` (OUTSIDE this lock) and
        # the caller's ``await _cancel_inflight_for_revision`` yielded
        # the event loop before this method ran, leaving a partial-
        # apply window where ``_wait_plan_stable`` callers observed a
        # bumped ``revision_index`` with the un-rewired edge DAG.
        # Reports calling :meth:`_wait_plan_stable` now observe either
        # the pre- or post-revision state, never a partial apply (e.g.
        # supersedes integrated but pin not yet repinned, or
        # revision_index bumped but PlanRevised not yet emitted). Fixes
        # the race between fire-and-forget judge-triggered refines
        # (#254) and imperative report_task_* handlers (goldfive a4)
        # and closes the partial-apply window opened by the two-phase
        # plan swap that goldfive#403 audited as HIGH severity.
        lock = self._get_plan_lock(session)
        async with lock:
            # goldfive#267: resolve the effective dry_run for THIS emit so
            # every side-effect site below can gate consistently. Mirrors
            # the wire-stamp resolution further down (caller threads through
            # ``not was_installed``; legacy callers pass ``None`` and fall
            # back to ``not self._should_inject()``). Computed once,
            # consumed by:
            #   * the supersedes-integration ``set_session_plan`` swap;
            #   * ``clear_obsolete_corrections_on_revision`` (state pop);
            #   * ``queue_corrections_for_revision`` (state write);
            #   * ``_repin_current_task_on_supersedes`` (session +
            #     session.state pin + adapter rewrite hook).
            # Under ``dry_run=True`` (observation_only's dry-run preview),
            # every one of these is suppressed so the would-have-applied
            # revision never end-runs ``_apply_revision``'s gate via the
            # emit path. The PlanRevised wire event still fires — dry_run
            # is observability, not silence.
            effective_dry_run = (
                bool(dry_run)
                if dry_run is not None
                else (not self._steerer._should_inject())
            )
            prior_plan_id_short = (
                (session.plan.id if session.plan is not None else "")[:16]
                or "<none>"
            )
            would_be_plan_id_short = (revised.id or "")[:16] or "<empty>"

            # goldfive#251: integrate CORRECT-kind supersedes links into
            # the DAG. The old task stays in the plan as a historical
            # COMPLETED node; the new correction-task is inserted as a
            # child with an edge old -> new, and any downstream edges of
            # the old task are rewritten so work flows through the
            # correction. No-op for REPLACE-kind (existing behaviour is
            # preserved) and for plans without supersedes.
            # goldfive#247: returns a NEW Plan (Plan is frozen). With
            # frozen types, the integrated variant replaces ``revised``
            # so the wire payload and the session pointer agree on the
            # rewired DAG.
            #
            # goldfive#267 / goldfive#403: we ALWAYS compute the
            # integrated plan (the PlanRevised emit payload needs the
            # rewired DAG so operators can preview what the corrective
            # refine WOULD have installed under ``dry_run=True``, and
            # the live install path needs it so the FIRST and ONLY
            # ``set_session_plan`` call below lands the rewired shape).
            # Rebinding ``revised`` is unconditional; the actual session
            # mutation is gated below on ``effective_dry_run``.
            revised = self._integrate_correction_supersedes(revised)

            # goldfive#403: single, atomic install site for the
            # session-mutation triad — ``session.plan`` swap, per-(kind,
            # target) addressed watermark, and orchestration-state
            # current-plan pointer — all performed under the lock with
            # the integrated (supersedes-rewired) DAG. Pre-#403 the
            # first two of these lived in :meth:`_apply_revision`
            # (outside the lock) and ran BEFORE the caller's
            # ``await _cancel_inflight_for_revision``, leaving the
            # bumped ``revision_index`` + stamped watermark visible on
            # ``session.plan`` with the un-rewired edge DAG and
            # un-repinned ``current_task_id`` until this method's lock
            # acquisition. The supersedes-integration ``set_session_plan``
            # is now folded into this single call (was a second swap
            # below). Under ``effective_dry_run`` every site is
            # suppressed exactly as before — the would-have-applied
            # revision never end-runs ``_apply_revision``'s gate via
            # the emit path. The PlanRevised wire event still fires —
            # dry_run is observability, not silence.
            if effective_dry_run:
                log.info(
                    "DefaultSteerer._emit_plan_revised: dry_run=True — "
                    "SKIPPING session.plan install + watermark stamp + "
                    "orchestration-state pointer. prior_plan_id=%s "
                    "would_have_installed_plan_id=%s revision_index=%d "
                    "drift_kind=%s",
                    prior_plan_id_short,
                    would_be_plan_id_short,
                    int(revised.revision_index),
                    drift.kind.value,
                )
            else:
                with channel_processor_active():
                    set_session_plan(session, revised)
                # goldfive#245 follow-up / goldfive#403: stamp the
                # per-(kind, target) addressed watermark so the
                # verdict-freshness gate in :meth:`_handle_drift` can
                # drop subsequent same-(kind, target) verdicts observed
                # at older revisions as redundant. User-authored drifts
                # bypass the gate entirely so they don't stamp here.
                # Moved from :meth:`_apply_revision` into the lock so
                # the watermark is never visible without the matching
                # ``session.plan`` swap.
                if (drift.authored_by or "").lower() != "user":
                    key = (drift.kind.value, drift.current_task_id or "")
                    session.last_addressed_revision_by_drift_key[key] = int(
                        revised.revision_index
                    )
                # goldfive#152 / goldfive#403: refresh the
                # orchestration-state current plan id so downstream
                # reads see the revised id, not the stale one. Moved
                # from :meth:`_apply_revision` into the lock so the
                # orchestration-state pointer is never visible without
                # the matching ``session.plan`` swap.
                _ostate.set_current_plan(session.state, revised)

            # goldfive#251 Stream D: GC corrections for tasks superseded by
            # this revision BEFORE queuing new ones. A task whose correction
            # is about to be obsoleted (because the new revision supersedes
            # the correction task itself) must have its stale correction
            # dropped. Runs first so a same-revision CORRECT->CORRECT chain
            # (T -> T' -> T'') doesn't race: the T correction is cleared
            # here, then T''s correction is written below.
            #
            # goldfive#267: both clear_obsolete and queue_corrections write
            # to ``session.state`` (``goldfive.pending_corrections.*``
            # keys); under ``dry_run=True`` those writes would end-run the
            # gate by injecting correction directives into the next-turn
            # agent prompt for a revision that was never installed. Skip
            # both under dry_run.
            if effective_dry_run:
                log.info(
                    "DefaultSteerer._emit_plan_revised: dry_run=True — "
                    "SKIPPING pending-corrections GC + queue. "
                    "prior_plan_id=%s would_have_installed_plan_id=%s "
                    "revision_index=%d drift_kind=%s",
                    prior_plan_id_short,
                    would_be_plan_id_short,
                    int(revised.revision_index),
                    drift.kind.value,
                )
            else:
                clear_obsolete_corrections_on_revision(session, revised)

                # goldfive#251 Stream D: for every NEW task with supersedes_kind
                # == CORRECT, stamp a structured correction dict on the
                # orchestration session state under
                # ``goldfive.pending_corrections.<agent_name>.<task_id>``. The
                # dynamic instruction resolver (Stream B) reads this on the next
                # turn and appends a directive-style correction block to the
                # agent's system prompt. No-op on refines with no CORRECT links.
                #
                # AGENCY-PRESERVATION.md task #11: in the request_context regime
                # with the Site-4 pin retired, corrections ride the agent-scoped
                # ObserverNoteQueue instead of that slot (the suppressed resolver
                # no longer reads it). ``pin_assigned_task`` keeps the pin — and
                # hence the slot read — so corrections stay on the slot there.
                # Legacy channel: unchanged.
                _channel = getattr(
                    self._steerer, "_signal_channel", "legacy_user_message"
                )
                _pin = bool(
                    getattr(
                        getattr(self._steerer, "_steering_config", None),
                        "pin_assigned_task",
                        False,
                    )
                )
                queue_corrections_for_revision(
                    session=session,
                    revised=revised,
                    prev_plan=prev_plan,
                    drift=drift,
                    corrections_via_notes=(_channel == "request_context" and not _pin),
                )

            # goldfive#237: re-pin ``current_task_id`` onto any replacement
            # task the revision introduces. Without this, agents keep
            # reporting on the superseded (FAILED/CANCELLED) task and the
            # replacement stays PENDING despite active work — the contradiction
            # live sessions surfaced. Done before the event is emitted so
            # downstream observers see the revised pin consistently with the
            # revised plan. Additive: when no task has ``supersedes`` set,
            # nothing changes.
            #
            # goldfive#267: under ``dry_run=True`` skip the repin — it
            # mutates ``session.current_task_id``, the orchestration
            # ``session.state`` pin, and per-agent ADK state copies via
            # the adapter hook. All three are session-pointer writes that
            # would end-run the gate (the next coordinator turn would pick
            # up the would-have-been-installed corrective task because
            # the pin was moved onto it).
            if effective_dry_run:
                log.info(
                    "DefaultSteerer._emit_plan_revised: dry_run=True — "
                    "SKIPPING goldfive#237 current_task_id repin "
                    "(session + session.state + adapter hook). "
                    "prior_plan_id=%s would_have_installed_plan_id=%s "
                    "revision_index=%d drift_kind=%s",
                    prior_plan_id_short,
                    would_be_plan_id_short,
                    int(revised.revision_index),
                    drift.kind.value,
                )
            else:
                self._repin_current_task_on_supersedes(session, revised)

            evt = self._steerer._new_envelope(session)
            evt.plan_revised.plan.CopyFrom(to_pb_plan(revised))
            evt.plan_revised.drift_kind = self._steerer._drift_kind_pb_value(drift.kind)
            evt.plan_revised.severity = self._steerer._drift_severity_pb_value(drift.severity)
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
                self._steerer.drift._drift_annotation_id(drift)
                or str(getattr(drift, "id", "") or "")
            )
            if trig_id:
                evt.plan_revised.trigger_event_id = trig_id
            # Populate the minimal cross-revision diff so sinks that want a
            # "what changed" view don't have to re-fetch and diff the two
            # plans client-side. prev_plan may be None on the first revision
            # of a run that never received an initial plan — the helper
            # treats that as "everything in revised is newly added".
            evt.plan_revised.diff.CopyFrom(build_plan_revision_diff(prev_plan, revised))
            # Refine-context observability (judge-observability event). Sinks
            # rendering a Gantt / timeline want to explain WHY a refine was
            # requested and WHAT the planner produced without re-fetching
            # the drift and both plans.
            evt.plan_revised.refine_input_summary = self._build_refine_input_summary(
                drift, prev_plan
            )
            evt.plan_revised.refine_output_summary = self._build_refine_output_summary(revised)
            evt.plan_revised.target_agent_id = drift.current_agent_id or ""
            # goldfive#254 — stamp the dry_run marker so consumers can
            # distinguish a would-have-applied revision (observation-only
            # mode, plan was NOT installed onto ``session.plan``, no
            # GOLDFIVE_STEER ControlMessage was enqueued, no cancel was
            # requested) from a real revision. Wire-default is ``false``
            # so legacy producers and historical events round-trip as
            # "real revision" without surprise.
            #
            # goldfive#255: the caller threads the value through from
            # :meth:`_apply_revision`'s ``was_installed`` return so the
            # marker reflects whether this SPECIFIC revision was actually
            # suppressed (a bootstrap install or user-authored steer
            # under observation_only is a REAL revision — dry_run False
            # — even though ``self._should_inject()`` is False). Legacy
            # callers that don't pass ``dry_run`` fall back to the
            # pre-#255 behaviour for back-compat.
            #
            # goldfive#267: resolved once at the top of this method into
            # ``effective_dry_run`` (same fallback logic) so the wire
            # stamp and every side-effect carve-out above key off the
            # SAME value — a future refactor that drifts those two
            # resolutions apart would silently re-introduce the bug
            # this issue closed (gate-skipped on install, applied on
            # emit).
            evt.plan_revised.dry_run = effective_dry_run
            # Phase 2.X / goldfive#271 Gap 2: log the emission so a
            # raise-mid-fire scenario (proto build OK, sink emit raises)
            # leaves a goldfive-side trace before the harmonograf side
            # observes the gap. Pair with the warning on empty
            # plan_id / run_id below — those are the harmonograf#197
            # gate preconditions.
            plan_id_short = (revised.id or "")[:16] or "<empty>"
            run_id_short = (session.run_id or "")[:16] or "<empty>"
            if not session.run_id:
                log.warning(
                    "DefaultSteerer._emit_plan_revised: empty run_id for "
                    "plan_id=%s — harmonograf will drop both the audit "
                    "row AND the task_plans dispatch (harmonograf#197 "
                    "gate); this would silently lose the revision",
                    plan_id_short,
                )
            if not revised.id:
                log.warning(
                    "DefaultSteerer._emit_plan_revised: empty plan_id on "
                    "revised plan — harmonograf will drop the task_plans "
                    "row (no upsert key); this would silently lose the "
                    "revision",
                )
            log.info(
                "DefaultSteerer._emit_plan_revised: plan_id=%s "
                "revision_index=%d drift_kind=%s severity=%s run_id=%s",
                plan_id_short,
                int(revised.revision_index),
                drift.kind.value,
                drift.severity.value,
                run_id_short,
            )
            await self._steerer._emit(evt)
            # goldfive#251 R4 — every per-task status change carried by the
            # refine (e.g. ``_force_looper_failed`` stamping FAILED on the
            # looper, a CORRECT-supersedes integration cancelling the
            # superseded task, a REPLACE supersession marking the old task
            # CANCELLED) gets a paired ``TaskTransitioned`` sink event with
            # ``source="plan_revision"`` so operators see the refine-driven
            # transitions on the same observability lane as LLM-driven
            # ones. The transition events come AFTER ``PlanRevised`` so a
            # consumer that processes events strictly in order sees the
            # plan flip first, then the per-task status changes that flow
            # from it.
            await self._steerer.tasks._emit_plan_revision_transitions(session, prev_plan, revised)
            # goldfive a4: paired correlation envelope. The proto
            # ``PlanRevised`` carries no ``attempt_id`` field today; emit
            # a sidecar dict event so consumers can pair this success
            # with its preceding ``refine_attempted`` by attempt_id. The
            # proto event remains the primary surface; this is purely
            # correlation. ``attempt_id`` is ``None`` on legacy callers
            # that haven't been threaded through the new pipeline (e.g.
            # the executor's plan-swap detector) — those callers skip
            # the sidecar and behave exactly as before.
            if attempt_id:
                await self._emit_plan_revised_correlation(
                    session, revised, drift, attempt_id=attempt_id
                )

    # ------------------------------------------------------------------
    # Refine-context observability summaries
    # ------------------------------------------------------------------

    @staticmethod
    def _build_refine_input_summary(
        drift: DriftEvent,
        prev_plan: Plan | None,
    ) -> str:
        """Render a short summary of what goldfive sent to ``planner.refine``.

        Intentionally terse — we pair the drift's ``kind`` / ``severity``
        / ``detail`` with a compact plan census (task count + status
        tallies) so a sink can answer "why was this refine requested,
        what did the planner see?" at a glance. Truncated via the same
        convention used by ``trigger_input`` to keep event sinks bounded.
        """
        from goldfive.drift_observer import DriftObserver

        parts: list[str] = []
        parts.append(f"drift={drift.kind.value}/{drift.severity.value}")
        if drift.current_task_id:
            parts.append(f"task={drift.current_task_id}")
        if drift.detail:
            parts.append(f"detail={drift.detail}")
        if prev_plan is not None:
            tasks = getattr(prev_plan, "tasks", None) or []
            parts.append(f"prior_plan=rev{prev_plan.revision_index}:{len(tasks)}tasks")
            if tasks:
                status_counts: dict[str, int] = {}
                for t in tasks:
                    status = getattr(t, "status", None)
                    key = str(getattr(status, "value", status) or "unspecified")
                    status_counts[key] = status_counts.get(key, 0) + 1
                tally = ",".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
                parts.append(f"prior_statuses={tally}")
        else:
            parts.append("prior_plan=none")
        text = " | ".join(parts)
        return DriftObserver._truncate_trigger_input(text)

    @staticmethod
    def _build_refine_output_summary(revised: Plan) -> str:
        """Render a short summary of the plan the planner returned."""
        from goldfive.drift_observer import DriftObserver

        tasks = getattr(revised, "tasks", None) or []
        parts: list[str] = [
            f"revision_index={revised.revision_index}",
            f"tasks={len(tasks)}",
        ]
        # Include the first few task titles so a Gantt can show the
        # revised plan's shape without fetching the full plan payload.
        titles = [str(getattr(t, "title", "") or "") for t in tasks[:6]]
        titles = [t for t in titles if t]
        if titles:
            parts.append("titles=[" + ", ".join(titles) + "]")
        text = " | ".join(parts)
        return DriftObserver._truncate_trigger_input(text)
