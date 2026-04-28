"""Sequential executor.

Drives a :class:`~goldfive.types.Plan` one task at a time, re-invoking the
underlying agent (via the adapter) with a next-task nudge until the plan
terminates or a safety cap trips.

This is a port of ``_run_orchestrated`` / ``_run_sequential`` from
``harmonograf_client/agent.py`` with the ADK-specific session/runner plumbing
stripped out. The harmonograf version drives ADK's generator model directly;
here we speak to the adapter through :meth:`AgentAdapter.invoke`, which is
defined to return exactly once per call after the reporting tools inside have
mutated task state via the Steerer.

Key behaviors
-------------
* Honor the plan's topological order: only invoke tasks whose predecessors
  are ``COMPLETED``. Walk tasks one-at-a-time (no parallelism).
* After each invocation, re-read plan state: tasks may have moved to
  ``COMPLETED`` / ``FAILED`` / ``BLOCKED`` via reporting tools; the steerer
  may also have mutated the plan in response to drift (``PlanRevised``).
* Optionally budget the total number of adapter invocations with
  ``max_task_invocations`` so a stuck agent cannot spin forever. The
  default is ``None`` (unbounded); per-task / per-tool caps are the
  primary guards against runaway loops.
* Terminate with :class:`RunCompleted` on success or :class:`RunAborted`
  when ``fail_fast`` is set and a task fails fatally.

Lifecycle ownership: the executor does NOT emit ``RunStarted``. The
:class:`Runner` owns ``Run*`` lifecycle events; the executor owns the
terminal ``RunCompleted`` / ``RunAborted`` plus ``PlanRevised`` and the
per-task ``Task*`` events (the latter via the steerer).
"""

from __future__ import annotations

import asyncio
import logging
import re
import warnings
from typing import TYPE_CHECKING, Any

from goldfive.adapters.adk_reentry import ReentryKind, reentry
from goldfive.events import (
    emit,
    new_event,
    plan_revised_event,
    run_aborted_event,
    run_completed_event,
)
from goldfive.executors._control import (
    _ControlCancelled,
    dispatch_control,
    drain_controls,
)
from goldfive.protocols import AgentAdapter, EventSink, Executor, Planner, Steerer
from goldfive.results import ExecutionOutcome, evaluate_goal_predicates
from goldfive.types import DriftKind, DriftSeverity, Plan, Session, Task, TaskStatus

if TYPE_CHECKING:
    from goldfive.control import ControlChannel

log = logging.getLogger(__name__)


# Symbolic cancel-reason for USER_STEER. Mirrors
# :data:`goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER` but duplicated
# as a plain string to avoid importing the optional ADK adapter module
# from the provider-agnostic executor. Keep in sync. See goldfive#139.
_CANCEL_REASON_USER_STEER: str = "user_steer"


def _steer_cancel_reason_prefix(steer_msg: Any) -> str:
    """Return a structured cancel reason for a STEER-interrupted task.

    Format: ``user_steer:<annotation_id>`` when the bridge forwarded an
    annotation id on ``payload["annotation_id"]`` (goldfive#171); falls
    back to ``user_steer:<control_id>`` when no annotation id is
    present (e.g. a direct SDK caller); final fallback is the generic
    ``user_steer:steer`` so the reason field is never empty on a
    steer-driven cancel. See goldfive#205.
    """
    payload = getattr(steer_msg, "payload", None)
    if isinstance(payload, dict):
        ann = str(payload.get("annotation_id", "") or "")
        if ann:
            return f"user_steer:{ann}"
    msg_id = str(getattr(steer_msg, "id", "") or "")
    if msg_id:
        return f"user_steer:{msg_id}"
    return "user_steer:steer"


def _tag_adapter_cancel_user_steer(adapter: Any, session: Any = None) -> None:
    """Tag the adapter's next mid-invocation cancel with the USER_STEER reason.

    Called just before the executor triggers ``task.cancel()`` on the
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
            setter(session, _CANCEL_REASON_USER_STEER)
            return
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "SequentialExecutor: set_next_cancel_reason raised: %s", exc
            )
    try:
        adapter._next_cancel_reason = _CANCEL_REASON_USER_STEER
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "SequentialExecutor: could not tag adapter cancel reason: %s",
            exc,
        )


_DEFAULT_MAX_NUDGE_REPLAYS: int = 3


class SequentialExecutor(Executor):
    """Single-threaded executor that drives one task at a time.

    Parameters
    ----------
    max_task_invocations:
        Optional upper bound on the number of adapter ``invoke()`` calls
        a single :meth:`run` may issue. Each eligible task consumes one
        invocation; when drift triggers a plan revision, the new plan's
        remaining tasks share the same budget. Defaults to ``None``
        (unbounded) — the run proceeds until every task reaches a
        terminal state or a drift / ``fail_fast`` / per-task-lineage
        cap terminates it. Set to an integer to enforce a ceiling on
        total adapter invocations for defensive, belt-and-suspenders
        containment.
    max_retries_per_task_lineage:
        Upper bound on the number of adapter ``invoke()`` calls that may
        be spent on any one task "lineage" — the original task plus any
        refine-spawned retries of it. A lineage is identified by stripping
        ``retry_`` / ``retry2_`` / ``retryN_`` prefixes from the task id,
        so e.g. ``t0``, ``retry_t0``, and ``retry2_retry_t0`` all share
        the lineage root ``t0``. When the cap is reached, the next task
        in that lineage is skipped (never sent to the adapter) and marked
        ``FAILED`` in place; downstream tasks then block naturally via
        ``_pick_next_task``'s dependency check. Defaults to ``3``, which
        bounds worst-case blast radius to a small constant multiple of
        the plan size even when a misbehaving refine keeps spawning
        ``retry_<task>`` clones. Set to a very large number to disable.
        See ``TASK-LIFECYCLE.md §7.7``.
    fail_fast:
        When ``True`` (default), the first task that ends up ``FAILED`` causes
        the executor to emit ``RunAborted`` and stop. When ``False``, failed
        tasks are recorded and the executor continues walking remaining
        eligible tasks.
    """

    def __init__(
        self,
        *,
        max_task_invocations: int | None = None,
        max_retries_per_task_lineage: int = 3,
        fail_fast: bool = True,
        overlay_mode: bool = False,
        **legacy_kwargs: Any,
    ) -> None:
        # Backwards-compatible alias: accept the old name for one release
        # and emit a DeprecationWarning mapping it to the new one. Remove
        # in a future release.
        if "max_plan_reinvocations" in legacy_kwargs:
            legacy_value = legacy_kwargs.pop("max_plan_reinvocations")
            warnings.warn(
                "SequentialExecutor(max_plan_reinvocations=...) is deprecated; "
                "use max_task_invocations=... instead. The parameter has been "
                "renamed for clarity — it caps total adapter invocations per "
                "run, not plan refinements.",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_task_invocations is None:
                max_task_invocations = legacy_value
        # ``max_follow_up_rounds`` was the cap on the overlay's soft
        # follow-up loop. goldfive#163 removed that loop entirely
        # (flow-prompted coordinators were re-running their full
        # pipeline on every follow-up, amplifying a ~10min run into
        # 40+ minutes). The kwarg is accepted here for back-compat
        # with a DeprecationWarning; it has no effect. Remove in a
        # future release.
        if "max_follow_up_rounds" in legacy_kwargs:
            legacy_kwargs.pop("max_follow_up_rounds")
            warnings.warn(
                "SequentialExecutor(max_follow_up_rounds=...) is deprecated "
                "and has no effect; the overlay's soft follow-up loop was "
                "removed in goldfive#163. PENDING tasks at the end of the "
                "passthrough invocation are transitioned to NOT_NEEDED "
                "instead of being re-dispatched. STEER remains the user-"
                "driven path for exercising uncovered tasks.",
                DeprecationWarning,
                stacklevel=2,
            )
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"SequentialExecutor got unexpected keyword argument(s): {unexpected}")
        self.max_task_invocations: int | None = (
            None if max_task_invocations is None else int(max_task_invocations)
        )
        self.max_retries_per_task_lineage = int(max_retries_per_task_lineage)
        self.fail_fast = bool(fail_fast)
        # Overlay model (goldfive#141, refined in goldfive#163). When
        # ``overlay_mode`` is True:
        #   1. Call ``adapter.invoke_passthrough(goal_text)`` ONCE
        #      with the user's original request and a plugin-attached
        #      :class:`~goldfive.reconciler.PlanReconciler` watching
        #      the agent tree run its natural flow.
        #   2. When the invocation ends, mark any PENDING tasks as
        #      NOT_NEEDED. The tree did what it naturally does; the
        #      reconciler recorded the coverage; goldfive does not
        #      drive per-task. Users who want uncovered tasks
        #      exercised explicitly can STEER.
        # When False (default for direct SequentialExecutor() callers)
        # we keep the legacy per-task loop so tests and callers that
        # already encode that model keep working. The ``goldfive.wrap``
        # convenience flips this to True by default.
        self.overlay_mode = bool(overlay_mode)

    # goldfive#202: cap on the overlay's nudge-driven re-invoke loop.
    # Each pass that the steerer queues a Level 2 nudge burns one.
    # Class attribute (not a constructor kwarg) so subclasses can tune
    # it without expanding the public constructor surface; also keeps
    # it out of ``legacy_kwargs`` handling above.
    _MAX_NUDGE_REPLAYS: int = _DEFAULT_MAX_NUDGE_REPLAYS

    # ------------------------------------------------------------------
    # Executor protocol
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        plan: Plan,
        session: Session,
        adapter: AgentAdapter,
        steerer: Steerer,
        planner: Planner,
        sinks: list[EventSink],
        control: ControlChannel | None = None,
        user_input: str = "",
    ) -> ExecutionOutcome:
        """Walk ``plan`` end-to-end, driving ``adapter`` once per eligible task.

        Returns an :class:`ExecutionOutcome` summarizing whether the run
        succeeded and carrying the final ``session`` so callers can inspect
        completed results / agent notes.

        When :attr:`overlay_mode` is True the executor switches to the
        goldfive#141 overlay path: one
        :meth:`AgentAdapter.invoke_passthrough` with ``user_input``.
        When the invocation ends any PENDING tasks are transitioned to
        ``NOT_NEEDED`` — goldfive#163 removed the soft follow-up loop
        that used to re-dispatch missed tasks (it amplified slow
        flow-prompted coordinators into rework loops). STEER remains
        the user-driven path for exercising uncovered work. Falls
        through to the legacy per-task loop when ``overlay_mode`` is
        False.
        """
        # Pin the plan onto the session so the steerer / reporting handlers
        # see the same object the executor is iterating.
        session.plan = plan

        # Wire sinks + planner into the steerer so reporting-tool handlers
        # can emit events and trigger planner.refine on drift.
        steerer.bind(sinks=sinks, planner=planner)

        # Overlay-mode dispatch: single passthrough + reconciled follow-ups.
        # Only engages when the adapter exposes ``invoke_passthrough`` —
        # duck-typed so third-party AgentAdapter implementations that
        # predate the overlay refactor still work under ``overlay_mode=False``
        # (the caller's choice).
        if self.overlay_mode and callable(getattr(adapter, "invoke_passthrough", None)):
            return await self._run_overlay(
                plan=plan,
                session=session,
                adapter=adapter,
                steerer=steerer,
                planner=planner,
                sinks=sinks,
                control=control,
                user_input=user_input,
            )

        # NOTE: RunStarted is emitted by Runner, not by the executor. See
        # Runner._emit_run_started.

        invocations = 0
        failure_reason = ""
        run_failed = False

        # Per-lineage invocation count. The "lineage root" of a task is
        # its id with any chain of ``retry_`` / ``retryN_`` prefixes
        # stripped, so ``t0``, ``retry_t0``, and ``retry2_retry_t0`` all
        # collapse to the same root ``t0``. Bounding this per-run bounds
        # the worst-case cost of an LLM that keeps refining a failing
        # task into new ``retry_<...>`` tasks forever. See
        # ``max_retries_per_task_lineage``.
        lineage_invocations: dict[str, int] = {}

        # Track the plan identity so we detect mid-run revisions by the steerer.
        last_plan_id = plan.id
        last_revision_index = plan.revision_index

        # Cap by max_task_invocations: this is the number of adapter
        # invocations we'll allow for the whole run. Each eligible task
        # burns one; mid-run plan revisions do not refund any. ``None``
        # means unbounded — the loop exits when no eligible task remains
        # or when ``fail_fast`` / the per-lineage cap trips.
        while self.max_task_invocations is None or invocations < self.max_task_invocations:
            # ------------------------------------------------------------------
            # Control channel: drain any pending messages before picking
            # the next task. PAUSE here blocks until RESUME arrives.
            # ------------------------------------------------------------------
            try:
                stop, steer_msg = await self._apply_pre_task_controls(
                    control=control,
                    session=session,
                    steerer=steerer,
                    sinks=sinks,
                )
            except _ControlCancelled as cancelled:
                failure_reason = cancelled.detail
                run_failed = True
                break
            if stop:
                failure_reason = "cancelled by control"
                run_failed = True
                break
            if steer_msg is not None:
                await self._apply_steer(steer_msg, steerer=steerer, session=session)
                # The steerer swapped session.plan; pick up the revision
                # on the next outer iteration.

            current_plan = session.plan or plan

            # Detect an out-of-band plan revision (steerer swapped the plan
            # on the session). Emit PlanRevised so sinks see the boundary.
            if (
                current_plan.id != last_plan_id
                or current_plan.revision_index != last_revision_index
            ):
                # Phase 2.X / goldfive#271 Gap 2: log the executor-side
                # detection at INFO so it's clear who emitted this
                # PlanRevised. Two emitters exist for the same plan
                # swap (steerer's _emit_plan_revised + this swap
                # detector); the source is implicit in the call site,
                # but explicit logs help cross-reference with the
                # steerer's INFO line.
                log.info(
                    "SequentialExecutor: plan-swap detected, emitting "
                    "PlanRevised plan_id=%s revision_index=%d "
                    "(prior=%s) drift_kind=%s",
                    (current_plan.id or "")[:16] or "<empty>",
                    int(current_plan.revision_index),
                    last_plan_id[:16] if last_plan_id else "<none>",
                    current_plan.revision_kind or "<none>",
                )
                await emit(
                    sinks,
                    plan_revised_event(
                        run_id=session.run_id,
                        sequence=session.next_sequence(),
                        plan=current_plan,
                        drift_kind=current_plan.revision_kind,
                        severity=current_plan.revision_severity,
                        reason=current_plan.revision_reason,
                        revision_index=current_plan.revision_index,
                        session_id=session.id,
                    ),
                )
                last_plan_id = current_plan.id
                last_revision_index = current_plan.revision_index
                # Restart iteration from the current cursor: pick_next_task
                # below will naturally re-scan the (possibly new) plan.

            task = _pick_next_task(current_plan)
            if task is None:
                # No eligible PENDING task: either the plan is done or
                # everything remaining is blocked by failures. Break out
                # and let the post-loop block decide success.
                break

            # Per-task-lineage cap. Before we spend an invocation on
            # ``task``, check how many invocations we have already spent
            # on tasks that share its lineage root. If the cap is
            # already reached, do not invoke the adapter: transition
            # ``task`` to FAILED in place and let the outer loop pick up
            # the next eligible task (or abort if fail_fast). This keeps
            # a runaway refine loop (``t0`` -> ``retry_t0`` -> ``retry2_retry_t0``
            # -> ...) from ever reaching the adapter more than N times
            # per lineage, independent of how many plan revisions happen.
            lineage_root = _lineage_root(task.id)
            lineage_count = lineage_invocations.get(lineage_root, 0)
            if lineage_count >= self.max_retries_per_task_lineage:
                log.warning(
                    "SequentialExecutor: lineage cap reached for task=%s "
                    "(lineage_root=%s, count=%d, cap=%d); marking FAILED "
                    "without invoking adapter",
                    task.id,
                    lineage_root,
                    lineage_count,
                    self.max_retries_per_task_lineage,
                )
                await steerer.transition(
                    task.id,
                    TaskStatus.FAILED,
                    detail=(
                        f"retry-lineage cap reached: "
                        f"{lineage_count} invocations already spent on "
                        f"lineage root '{lineage_root}' "
                        f"(cap={self.max_retries_per_task_lineage})"
                    ),
                    session=session,
                )
                failure_reason = (
                    f"task {task.id} skipped: retry-lineage cap "
                    f"({self.max_retries_per_task_lineage}) reached for "
                    f"lineage root '{lineage_root}'"
                )
                if self.fail_fast:
                    run_failed = True
                    break
                # fail_fast=False: loop around; downstream tasks that
                # depended on this one will be blocked by _pick_next_task.
                continue

            session.current_task_id = task.id
            invocations += 1
            lineage_invocations[lineage_root] = lineage_count + 1

            log.debug(
                "SequentialExecutor: invoking task=%s (invocation %d/%s, "
                "lineage_root=%s, lineage_count=%d/%d)",
                task.id,
                invocations,
                "unbounded"
                if self.max_task_invocations is None
                else str(self.max_task_invocations),
                lineage_root,
                lineage_count + 1,
                self.max_retries_per_task_lineage,
            )

            # Auto-announce the task as RUNNING before invoke so agents
            # that never call report_task_started (the common callable
            # case) still produce a TaskStarted event. The steerer's
            # transition is idempotent once the task leaves PENDING.
            if task.status == TaskStatus.PENDING:
                await steerer.transition(task.id, TaskStatus.RUNNING, session=session)

            # Race the adapter invocation against the control channel so a
            # CANCEL / STEER / PAUSE arriving mid-task can cancel (or at
            # least observe) the in-flight task. The helper returns
            # ``(kind, payload)``:
            #   ("result", InvocationResult | None)   normal completion
            #   ("adapter_error", BaseException)      invoke raised
            #   ("cancelled", reason_str)             CANCEL interrupted
            #   ("steer", ControlMessage)             STEER interrupted
            outcome_kind, outcome_payload = await self._invoke_with_control(
                adapter=adapter,
                task=task,
                session=session,
                steerer=steerer,
                sinks=sinks,
                control=control,
            )

            if outcome_kind == "cancelled":
                # goldfive#205: the dispatch stashed a structured cancel
                # prefix on the session (``user_cancel:<annotation_id>``
                # for user-initiated CANCEL); thread it through so the
                # emitted TaskCancelled carries the reason.
                cancel_prefix = getattr(session, "_last_cancel_reason_prefix", "")
                session._last_cancel_reason_prefix = ""
                await self._mark_cancelled_if_live(
                    task_id=task.id,
                    steerer=steerer,
                    session=session,
                    cancel_reason=cancel_prefix,
                )
                failure_reason = str(outcome_payload) or "cancelled by control"
                run_failed = True
                break

            if outcome_kind == "adapter_error":
                exc = outcome_payload
                log.exception(
                    "SequentialExecutor: adapter.invoke raised for task=%s",
                    task.id,
                )
                failure_reason = f"adapter.invoke raised for task={task.id}: {exc}"
                run_failed = True
                break

            if outcome_kind == "steer":
                # goldfive#205: the in-flight task is being cancelled in
                # favour of a steer-driven plan revision. Stamp a
                # ``user_steer:<annotation_id>`` reason so sinks can
                # distinguish "superseded by user steering" from
                # generic run-aborts / cascade-cancels.
                steer_prefix = _steer_cancel_reason_prefix(outcome_payload)
                await self._mark_cancelled_if_live(
                    task_id=task.id,
                    steerer=steerer,
                    session=session,
                    cancel_reason=steer_prefix,
                )
                await self._apply_steer(outcome_payload, steerer=steerer, session=session)
                continue

            # outcome_kind == "result"
            result = outcome_payload

            # If the adapter reported an error on the invocation envelope
            # (but didn't raise), record it; the auto-transition block
            # below routes that through steerer.mark_task_failed unless
            # the agent already transitioned the task itself.
            invocation_error = result is not None and getattr(result, "error", None) is not None
            if invocation_error:
                log.warning(
                    "SequentialExecutor: InvocationResult.error=%s task=%s",
                    result.error,
                    task.id,
                )

            # Route the invocation result through the steerer's observer so
            # drift classification runs on the sequential path too. This
            # mirrors ParallelDAGExecutor and lets drift enter via the raw
            # invocation envelope (not only via reporting-tool handlers).
            if result is not None:
                try:
                    await steerer.observe(result, session)
                except Exception as observe_exc:  # noqa: BLE001
                    # Plumbing failure inside the drift pipeline: surface
                    # it so sinks see a signal, rather than silently
                    # treating the task's output as benign. INFO CUSTOM
                    # so the run continues but the failure is durably
                    # recorded. See goldfive#134.
                    log.warning(
                        "SequentialExecutor: steerer.observe raised for task=%s: %s",
                        task.id,
                        observe_exc,
                    )
                    await _emit_pipeline_failure_drift(
                        session=session,
                        sinks=sinks,
                        task_id=task.id,
                        reason=f"drift_pipeline_failed: {observe_exc}",
                    )

            # Re-read the task's tracked status after the invocation.
            tracked = _find_task(session.plan or current_plan, task.id)
            tracked_status = tracked.status if tracked is not None else TaskStatus.PENDING

            # If the agent returned without emitting a terminal reporting
            # call, auto-transition on its behalf: complete on a clean
            # return, fail if the invocation carried an error. The steerer
            # routes the transition (which dispatches to mark_task_*),
            # and mark_task_* is idempotent on terminal states, so agents
            # that DID emit reporting calls are unaffected.
            if tracked_status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                if invocation_error:
                    detail = str(result.error) if result is not None else ""
                    await steerer.transition(
                        task.id,
                        TaskStatus.FAILED,
                        detail=detail,
                        session=session,
                    )
                else:
                    summary = (result.text if result is not None else "") or ""
                    await steerer.transition(
                        task.id,
                        TaskStatus.COMPLETED,
                        detail=summary,
                        session=session,
                    )
                tracked = _find_task(session.plan or current_plan, task.id)
                tracked_status = tracked.status if tracked is not None else TaskStatus.PENDING

            if tracked_status == TaskStatus.FAILED:
                failure_reason = f"task {task.id} failed"
                if self.fail_fast:
                    # goldfive#202: a FAILED task with a live replacement
                    # in the current plan (refine-spawned successor, e.g.
                    # ``retry_<id>`` or ``<id>_v2``) is NOT fatal — the
                    # replacement is the forward-progress path. Only
                    # abort when no replacement exists.
                    live_plan = session.plan or current_plan
                    failed_task = _find_task(live_plan, task.id)
                    has_replacement = failed_task is not None and _has_live_replacement(
                        live_plan, failed_task
                    )
                    if has_replacement:
                        log.info(
                            "SequentialExecutor: fail_fast skipped for task=%s — "
                            "live replacement present in current plan revision",
                            task.id,
                        )
                        failure_reason = ""  # not actually fatal
                    else:
                        run_failed = True
                        break
                # else: continue to next eligible task.

            # Session.current_task_id gets reset by the next iteration.

        # ------------------------------------------------------------------
        # Terminal emission.
        # ------------------------------------------------------------------

        if run_failed:
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=failure_reason or "run aborted",
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(
                success=False, session=session, reason=failure_reason or "run aborted"
            )

        # If we exhausted the invocation budget with work still pending,
        # that is also a failure (stuck agent). Only applies when a
        # finite cap was configured.
        remaining = _pick_next_task(session.plan or plan)
        if (
            self.max_task_invocations is not None
            and invocations >= self.max_task_invocations
            and remaining is not None
        ):
            reason = (
                f"exhausted max_task_invocations={self.max_task_invocations} "
                f"with pending task {remaining.id}"
            )
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=reason,
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=reason)

        # If fail_fast=False and some task ended FAILED with no live
        # replacement, the run is only "successful" in the best-effort
        # sense. Match harmonograf: report success=False with a reason
        # when any task is truly terminal-failed (goldfive#202: a FAILED
        # task whose refine-spawned successor is live is not fatal).
        fatally_failed = _fatally_failed_task_ids(session.plan or plan)
        if fatally_failed and not self.fail_fast:
            reason = (
                "one or more tasks failed without a live replacement "
                f"(fail_fast=False): {', '.join(tid for tid in fatally_failed if tid)}"
            )
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=reason,
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=reason)

        # Reachability audit: belt-and-suspenders for any cancellation path
        # that failed to cascade. If we are about to report success but
        # there are still PENDING tasks — and ``_pick_next_task`` already
        # declined to return any of them — those tasks are orphaned (e.g.
        # every path to them crosses a CANCELLED or FAILED predecessor
        # and no refine produced a replacement plan). Mark them CANCELLED
        # in place, emit a CRITICAL PLAN_DIVERGENCE drift so sinks see
        # the incomplete ending, and fail the run. This catches cases
        # the Steerer's cascade missed (and is a cheap safety net even
        # when the cascade worked correctly).
        orphaned = _pending_task_ids(session.plan or plan)
        if orphaned:
            log.warning(
                "SequentialExecutor: %d task(s) still PENDING with no "
                "eligible next task — run ending incomplete: %s",
                len(orphaned),
                ", ".join(orphaned),
            )
            # goldfive#205: structured cancel reason so harmonograf can
            # answer "why was this task cancelled?" in the Trajectory view.
            # Prefix the reason with ``run_aborted:`` and keep the legacy
            # humane tail so existing sinks that render the raw reason
            # unchanged still see intent.
            cancel_reason_value = (
                "run_aborted:orphaned by plan revision failure"
            )
            for orphan_id in orphaned:
                await steerer.transition(
                    orphan_id,
                    TaskStatus.CANCELLED,
                    detail="orphaned by plan revision failure",
                    cancel_reason=cancel_reason_value,
                    session=session,
                )
            reason = (
                f"{len(orphaned)} task(s) left PENDING with no eligible "
                f"next task (orphaned by plan revision failure): "
                f"{', '.join(orphaned)}"
            )
            await emit(
                sinks,
                _plan_divergence_drift_event(session, reason),
            )
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=reason,
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=reason)

        # Goal success-predicate gate (PLAN-LIFECYCLE.md §6.1, third
        # clause). Tasks are terminal and no orphans remain — now
        # verify the caller's semantic goals. A predicate that returns
        # False or raises fails the run with a descriptive reason.
        unmet = evaluate_goal_predicates(session)
        if unmet is not None:
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=unmet,
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=unmet)

        await emit(
            sinks,
            run_completed_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                outcome_summary=_outcome_summary(session),
                session_id=session.id,
            ),
        )
        return ExecutionOutcome(success=True, session=session)

    # ------------------------------------------------------------------
    # Overlay dispatch (goldfive#141)
    # ------------------------------------------------------------------

    async def _run_overlay(
        self,
        *,
        plan: Plan,
        session: Session,
        adapter: AgentAdapter,
        steerer: Steerer,
        planner: Planner,  # noqa: ARG002 -- reserved for future refine hooks
        sinks: list[EventSink],
        control: ControlChannel | None,
        user_input: str,
    ) -> ExecutionOutcome:
        """Overlay-model run loop: single passthrough, no soft follow-ups.

        See :meth:`run` for the high-level contract. In order:

        1. Instantiate a :class:`~goldfive.reconciler.PlanReconciler`
           bound to the session and steerer.
        2. Call ``adapter.invoke_passthrough(user_input, session=...,
           reconciler=...)`` ONCE. While the generator runs, the
           plugin forwards before/after_agent observations to the
           reconciler, which transitions plan tasks.
        3. When the invocation ends, mark any PENDING tasks as
           ``NOT_NEEDED``. The tree did what it naturally does;
           goldfive does not drive per-task. See goldfive#163:
           the previous soft follow-up loop re-dispatched each
           PENDING task as a new user message, and flow-prompted
           coordinators re-ran their full pipeline on every such
           message — turning a ~10 min run into 40+ min. Users who
           want uncovered tasks exercised explicitly can STEER.
        4. Emit terminal RunCompleted / RunAborted.

        STEER handling inside the passthrough loop is preserved
        (goldfive#149): a steer cancels the in-flight invocation,
        feeds the message to the steerer for USER_STEER drift +
        refine, then restarts ``invoke_passthrough`` with the steer
        body as the new user input.
        """
        from goldfive.reconciler import PlanReconciler

        # Belt-and-suspenders: a caller that passed overlay_mode=True
        # with no user_input can still fall back to an empty string;
        # the passthrough message will just be empty and the tree
        # runs off the last user turn in the session.
        user_input = user_input or ""

        host_agent_name = ""
        agent_obj = getattr(adapter, "_agent", None)
        if agent_obj is not None:
            host_agent_name = str(getattr(agent_obj, "name", "") or "")

        reconciler = PlanReconciler(
            session=session,
            steerer=steerer,
            host_agent_name=host_agent_name,
        )

        # --- Passthrough invocation loop. --------------------------
        # STEER control messages arriving mid-invocation require us
        # to (a) feed the STEER through the steerer so USER_STEER
        # drift → cascade-cancel + planner.refine runs and the
        # session's plan is swapped in place, then (b) re-invoke the
        # passthrough with the steer body as the new user input so
        # the tree runs the revised plan. This is why the loop is a
        # ``while True`` — a steer restarts the invocation against
        # the new plan; ``cancelled`` / ``adapter_error`` terminates
        # the run; ``result`` falls through to a (scoped) nudge-replay
        # check (goldfive#202) and then the NOT_NEEDED sweep.
        # See goldfive#149 for the regression this guards against.
        #
        # goldfive#202 re-introduces — in a narrowly scoped form — a
        # post-invocation re-invoke that #163 removed wholesale. The
        # #163 removal was correct for the "every PENDING at invocation
        # end triggers a follow-up" case (flow-prompted coordinators
        # re-ran their entire pipeline on every such message, turning
        # a 10-min run into 40+ min). But when the steerer explicitly
        # queues a nudge via ``session.pending_nudges`` in response to
        # an autonomous drift + plan revision (e.g. LOOPING_REASONING
        # → refine spawned ``<task>_v2``), the coordinator has no way
        # to know its plan changed; without a follow-up it keeps
        # retrying the superseded task. The nudge-replay path below
        # fires ONLY when the steerer explicitly asked for it (a nudge
        # is queued) AND there is still live work to do; capped at
        # ``_MAX_NUDGE_REPLAYS`` so a pathological nudge-queueing drift
        # cannot re-introduce the #163 amplification.
        current_user_input = user_input
        failure_reason = ""
        nudge_replays = 0
        # Re-entry contract (harmonograf#234). The very first iteration
        # of this loop carries the operator's verbatim user_input — for
        # plugins observing the inner runner that's still a goldfive
        # OVERLAY_REPLAY (the outer adk-web runner already emitted the
        # USER_TURN); ADKAdapter.invoke_passthrough pins OVERLAY_REPLAY
        # itself, so the default value here is "no executor-level pin".
        # Subsequent iterations triggered by a STEER or queued nudge
        # MUST carry the more-specific kind so plugins can attribute
        # the replay to the correct cause (operator-issued steer vs
        # autonomous drift-driven nudge); see ReentryKind.reentry()
        # for stack-precedence rules.
        next_reentry_kind: ReentryKind | None = None
        while True:
            # ContextVars snapshot at ``asyncio.create_task`` time
            # (which happens inside _invoke_passthrough_with_control),
            # so the ``reentry()`` block must wrap the call site here.
            if next_reentry_kind is None:
                kind, payload = await self._invoke_passthrough_with_control(
                    adapter=adapter,
                    session=session,
                    steerer=steerer,
                    sinks=sinks,
                    control=control,
                    reconciler=reconciler,
                    user_input=current_user_input,
                )
            else:
                with reentry(next_reentry_kind):
                    kind, payload = await self._invoke_passthrough_with_control(
                        adapter=adapter,
                        session=session,
                        steerer=steerer,
                        sinks=sinks,
                        control=control,
                        reconciler=reconciler,
                        user_input=current_user_input,
                    )
                # One-shot: clear after consumption so the next iteration
                # falls back to whatever the next branch decides.
                next_reentry_kind = None
            if kind == "cancelled":
                failure_reason = str(payload) or "cancelled by control"
                # goldfive#205: overlay path doesn't have a single
                # "current task" to stamp — the orphan sweep below
                # handles PENDING tasks. Clear the transient prefix so
                # it doesn't leak to a subsequent run.
                session._last_cancel_reason_prefix = ""
                await emit(
                    sinks,
                    run_aborted_event(
                        run_id=session.run_id,
                        sequence=session.next_sequence(),
                        reason=failure_reason,
                        session_id=session.id,
                    ),
                )
                return ExecutionOutcome(success=False, session=session, reason=failure_reason)
            if kind == "adapter_error":
                exc = payload
                failure_reason = f"adapter.invoke_passthrough raised: {exc}"
                log.exception("SequentialExecutor._run_overlay: passthrough raised")
                await emit(
                    sinks,
                    run_aborted_event(
                        run_id=session.run_id,
                        sequence=session.next_sequence(),
                        reason=failure_reason,
                        session_id=session.id,
                    ),
                )
                return ExecutionOutcome(success=False, session=session, reason=failure_reason)
            if kind == "steer":
                # Feed the STEER through the steerer so USER_STEER
                # drift fires → cascade-cancel + planner.refine runs
                # → session.plan is replaced with the revised plan.
                # Without this call the overlay would just mark the
                # pre-steer plan's tasks NOT_NEEDED on the ORIGINAL
                # plan and miss the steer entirely (the goldfive#149
                # regression, preserved here post-#163).
                log.info(
                    "SequentialExecutor._run_overlay: STEER received; "
                    "feeding steerer.observe for USER_STEER drift + refine",
                )
                await self._apply_steer(payload, steerer=steerer, session=session)
                # goldfive#152: wrap the steer body in a goldfive-authored
                # override header so the LLM sees it as a USER STEERING
                # CONTROL directive, not a fresh user turn. Supersedes
                # goldfive#149's raw-body handoff.
                current_user_input = self._compose_steer_restart_message(
                    payload, fallback=current_user_input
                )
                # Reset reconciler bookkeeping so the revised plan's
                # tasks map fresh — stale task_id → agent claims from
                # the pre-steer plan must not leak into the replay.
                reconciler.reset_for_new_plan(session.plan)
                # Re-entry contract (harmonograf#234): the next iteration
                # re-feeds a goldfive-composed steer-restart message
                # wrapped in USER STEERING CONTROL framing. Plugins
                # observing the inner runner's user_message hook should
                # see STEER_REPLAY, not USER_TURN, to suppress duplicate
                # emission.
                next_reentry_kind = ReentryKind.STEER_REPLAY
                # Restart the invocation with the steer body as the
                # new user input.
                continue
            # kind == "result": invocation ended normally. Before
            # falling through to the NOT_NEEDED sweep, check whether
            # the steerer queued a Level 2 nudge during this invocation
            # (e.g. LOOPING_REASONING drift → refine spawned a
            # replacement task → nudge queued describing the pivot).
            # If so, and there is still live work for the tree to do,
            # consume the queued nudge(s) as the next user message
            # and re-invoke. Bounded by ``_MAX_NUDGE_REPLAYS`` to
            # prevent the #163-style amplification: a coordinator
            # whose tree keeps producing nudge-eligible drift on every
            # turn must eventually stop triggering re-invokes.
            pending = list(session.pending_nudges)
            if (
                pending
                and nudge_replays < self._MAX_NUDGE_REPLAYS
                and _has_live_pending_or_running(session.plan or plan)
            ):
                session.pending_nudges.clear()
                nudge_replays += 1
                current_user_input = self._compose_nudge_replay_message(pending)
                log.info(
                    "SequentialExecutor._run_overlay: nudge replay %d/%d "
                    "(nudges=%d) — re-invoking passthrough with queued nudge",
                    nudge_replays,
                    self._MAX_NUDGE_REPLAYS,
                    len(pending),
                )
                # Reset reconciler bookkeeping so the revised plan's
                # tasks map fresh. The refine that queued the nudge
                # likely added new PENDING tasks that reconciler-side
                # agent claims should re-match against.
                reconciler.reset_for_new_plan(session.plan)
                # Re-entry contract (harmonograf#234): the next iteration
                # re-feeds a goldfive-composed nudge body that the
                # steerer authored in response to autonomous drift +
                # plan revision (e.g. LOOPING_REASONING → refine spawned
                # ``<task>_v2``). Plugins observing the inner runner's
                # user_message hook should see NUDGE_REPLAY, not
                # USER_TURN.
                next_reentry_kind = ReentryKind.NUDGE_REPLAY
                continue
            break

        # --- PENDING → NOT_NEEDED on invocation end (goldfive#163). ----
        # The tree finished its natural flow. Any task still PENDING
        # was not exercised by the tree. Mark terminal so sinks do not
        # see stale PENDING entries and downstream runs do not wedge.
        # We deliberately do NOT dispatch a follow-up: flow-prompted
        # coordinators re-run their full pipeline on every new user
        # message, which amplifies a ~10 min run into 40+ min. STEER
        # is the user-driven path for exercising uncovered work.
        live_plan = session.plan or plan
        pending_ids = [
            t.id
            for t in list(getattr(live_plan, "tasks", None) or ())
            if t.status is TaskStatus.PENDING and t.id
        ]
        if pending_ids:
            log.info(
                "SequentialExecutor._run_overlay: marking %d PENDING task(s) "
                "NOT_NEEDED at invocation end (no soft follow-up per #163): %s",
                len(pending_ids),
                ", ".join(pending_ids),
            )
            for tid in pending_ids:
                await steerer.transition(
                    tid,
                    TaskStatus.NOT_NEEDED,
                    detail="overlay: tree did not exercise; no follow-up dispatched (goldfive#163)",
                    session=session,
                )

        # --- Terminal emission: success if no failures. -----------
        # goldfive#202: a FAILED task with a live replacement (refine
        # spawned a successor like ``retry_<id>`` / ``<id>_v2``) is not
        # fatal — the replacement is the forward-progress path. Only
        # abort when at least one FAILED task has no live replacement
        # in the current plan revision.
        fatally_failed = _fatally_failed_task_ids(session.plan or plan)
        if fatally_failed and self.fail_fast:
            reason = (
                "one or more tasks failed without a live replacement: "
                f"{', '.join(tid for tid in fatally_failed if tid)}"
            )
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=reason,
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=reason)

        unmet = evaluate_goal_predicates(session)
        if unmet is not None:
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=unmet,
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=unmet)

        await emit(
            sinks,
            run_completed_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                outcome_summary=_outcome_summary(session),
                session_id=session.id,
            ),
        )
        return ExecutionOutcome(success=True, session=session)

    async def _invoke_passthrough_with_control(
        self,
        *,
        adapter: AgentAdapter,
        session: Session,
        steerer: Steerer,
        sinks: list[EventSink],
        control: ControlChannel | None,
        reconciler: Any,
        user_input: str,
    ) -> tuple[str, object | None]:
        """Drive one ``invoke_passthrough`` while watching the control channel.

        Mirrors :meth:`_invoke_with_control` for the overlay path.
        Returns ``(kind, payload)`` where ``kind`` is ``"result"``,
        ``"adapter_error"``, ``"cancelled"``, or ``"steer"``.
        """
        invoke_task: asyncio.Task = asyncio.create_task(
            adapter.invoke_passthrough(
                user_input,
                session=session,
                reconciler=reconciler,
            ),
            name="goldfive-invoke-passthrough",
        )

        if control is None:
            try:
                result = await invoke_task
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return ("adapter_error", exc)
            return ("result", result)

        while True:
            recv_task = asyncio.create_task(control.receive(), name="control-recv")
            done, _pending = await asyncio.wait(
                {invoke_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if invoke_task in done:
                if not recv_task.done():
                    recv_task.cancel()
                    try:
                        await recv_task
                    except BaseException:  # noqa: BLE001
                        pass
                try:
                    result = invoke_task.result()
                except asyncio.CancelledError:
                    return ("cancelled", "invoke cancelled")
                except BaseException as exc:  # noqa: BLE001
                    return ("adapter_error", exc)
                return ("result", result)

            try:
                msg = recv_task.result()
            except BaseException:  # noqa: BLE001
                msg = None
            if msg is None:
                try:
                    result = await invoke_task
                except BaseException as exc:  # noqa: BLE001
                    if isinstance(exc, asyncio.CancelledError):
                        return ("cancelled", "invoke cancelled")
                    return ("adapter_error", exc)
                return ("result", result)

            outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=sinks)
            try:
                await control.ack(outcome.ack)
            except Exception:  # noqa: BLE001
                pass

            if outcome.cancel_run:
                await self._cancel_invoke_task(invoke_task)
                if outcome.cancel_reason_prefix:
                    session._last_cancel_reason_prefix = outcome.cancel_reason_prefix
                return ("cancelled", outcome.cancel_reason or "cancelled by control")

            if outcome.steer_message is not None:
                await self._cancel_invoke_task(invoke_task)
                # For overlay-mode we don't loop on steer here — the
                # caller (Runner) will pick up session.plan (which the
                # steerer swapped) and re-enter on the next run() cycle.
                return ("steer", outcome.steer_message)

            # Non-cancelling controls: keep waiting.

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    async def _apply_pre_task_controls(
        self,
        *,
        control: ControlChannel | None,
        session: Session,
        steerer: Steerer,
        sinks: list[EventSink],
    ) -> tuple[bool, object | None]:
        """Drain queued controls before the next task; honour PAUSE.

        Returns ``(cancel_run, steer_message)``. The steer message is
        the last STEER control observed (the steerer consumes them one
        at a time, so queued STEERs after the first are applied in
        order on subsequent loop iterations).

        A PAUSE here blocks on ``channel.receive()`` until a RESUME (or
        CANCEL) arrives. The steerer's intervention-ladder Level 4
        pause (goldfive#142) triggers the same blocking wait via
        ``session.paused_for_human_intervention``, so a Level 4
        escalation is indistinguishable from an explicit user-initiated
        PAUSE from the executor's perspective.
        """
        if control is None:
            # Without a control channel the ladder-initiated pause has
            # nothing to wait on -- the run would wedge forever. Clear
            # the flag and let the drift event the steerer already
            # emitted stand as the signal.
            if session.paused_for_human_intervention:
                log.warning(
                    "SequentialExecutor: session.paused_for_human_intervention "
                    "is set but no control channel is attached; clearing flag "
                    "so the run does not wedge. Sinks still see the "
                    "HUMAN_INTERVENTION_REQUIRED drift."
                )
                session.paused_for_human_intervention = False
            return False, None

        outcomes = await drain_controls(control, session=session, steerer=steerer, sinks=sinks)

        cancel_run = False
        cancel_reason = ""
        cancel_prefix = ""
        steer_msg: object | None = None
        paused = False
        for o in outcomes:
            if o.cancel_run:
                cancel_run = True
                cancel_reason = o.cancel_reason
                cancel_prefix = o.cancel_reason_prefix
                break
            if o.steer_message is not None:
                steer_msg = o.steer_message
            if o.request_pause:
                paused = True
            if o.request_resume:
                paused = False

        if cancel_run:
            # goldfive#205: stash the structured cancel prefix on the
            # session so any per-task cancel cascade triggered by this
            # abort picks it up.
            if cancel_prefix:
                session._last_cancel_reason_prefix = cancel_prefix
            raise _ControlCancelled(cancel_reason or "cancelled by control")

        # Intervention-ladder pause (goldfive#142). The steerer sets
        # this flag from Level 4 PAUSE_ESCALATE dispatch; block the
        # same way we block on an explicit user PAUSE. RESUME / STEER
        # handlers clear the flag via dispatch_control.
        if session.paused_for_human_intervention:
            paused = True

        # Honour PAUSE by blocking on the channel until a RESUME /
        # CANCEL / STEER arrives.
        while paused:
            msg = await control.receive()
            if msg is None:
                # Channel closed — treat as resume so we don't wedge.
                paused = False
                break
            outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=sinks)
            try:
                await control.ack(outcome.ack)
            except Exception:  # noqa: BLE001
                pass
            if outcome.cancel_run:
                if outcome.cancel_reason_prefix:
                    session._last_cancel_reason_prefix = outcome.cancel_reason_prefix
                raise _ControlCancelled(outcome.cancel_reason or "cancelled by control")
            if outcome.request_resume:
                paused = False
            if outcome.steer_message is not None:
                steer_msg = outcome.steer_message
                paused = False
            if outcome.rewind_task_id:
                paused = False

        return False, steer_msg

    async def _invoke_with_control(
        self,
        *,
        adapter: AgentAdapter,
        task: Task,
        session: Session,
        steerer: Steerer,
        sinks: list[EventSink],
        control: ControlChannel | None,
    ) -> tuple[str, object | None]:
        """Run ``adapter.invoke(task, session)`` while watching ``control``.

        Returns ``(kind, payload)`` where ``kind`` is one of
        ``"result"`` (normal completion), ``"adapter_error"``,
        ``"cancelled"`` (CANCEL received), or ``"steer"`` (STEER
        received). Non-cancelling controls (PAUSE, RESUME, REWIND_TO,
        STATUS_QUERY, INTERCEPT_TRANSFER) are acked and do not
        interrupt the task.
        """
        invoke_task: asyncio.Task = asyncio.create_task(
            adapter.invoke(task, session),
            name=f"goldfive-invoke-{task.id}",
        )

        if control is None:
            try:
                result = await invoke_task
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return ("adapter_error", exc)
            return ("result", result)

        while True:
            recv_task = asyncio.create_task(control.receive(), name="control-recv")
            done, _pending = await asyncio.wait(
                {invoke_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if invoke_task in done:
                if not recv_task.done():
                    recv_task.cancel()
                    try:
                        await recv_task
                    except BaseException:  # noqa: BLE001
                        pass
                try:
                    result = invoke_task.result()
                except asyncio.CancelledError:
                    return ("cancelled", "invoke cancelled")
                except BaseException as exc:  # noqa: BLE001
                    return ("adapter_error", exc)
                return ("result", result)

            # recv_task completed first: a control message arrived.
            try:
                msg = recv_task.result()
            except BaseException:  # noqa: BLE001
                msg = None
            if msg is None:
                # Channel closed; fall back to awaiting the adapter task.
                try:
                    result = await invoke_task
                except BaseException as exc:  # noqa: BLE001
                    if isinstance(exc, asyncio.CancelledError):
                        return ("cancelled", "invoke cancelled")
                    return ("adapter_error", exc)
                return ("result", result)

            outcome = await dispatch_control(msg, session=session, steerer=steerer, sinks=sinks)
            try:
                await control.ack(outcome.ack)
            except Exception:  # noqa: BLE001
                pass

            if outcome.cancel_run:
                await self._cancel_invoke_task(invoke_task)
                # goldfive#205: stash the structured cancel prefix on the
                # session so the per-task cancel emit below picks it up.
                if outcome.cancel_reason_prefix:
                    session._last_cancel_reason_prefix = outcome.cancel_reason_prefix
                return ("cancelled", outcome.cancel_reason or "cancelled by control")

            if outcome.steer_message is not None:
                # Tag the adapter's next mid-invocation cancel with the
                # USER_STEER symbolic reason BEFORE triggering the
                # actual cancel, so the adapter's CancelledError
                # handler (and the synthetic function_response it
                # appends) carries LLM-actionable content instead of
                # the legacy generic jargon. See goldfive#139.
                _tag_adapter_cancel_user_steer(adapter, session=session)
                await self._cancel_invoke_task(invoke_task)
                return ("steer", outcome.steer_message)

            # Non-cancelling control: keep waiting on the adapter task.
            # (PAUSE mid-task lets the current task finish per spec;
            # REWIND_TO / STATUS_QUERY / INTERCEPT_TRANSFER are applied
            # in-line and execution continues.)

    @staticmethod
    async def _cancel_invoke_task(invoke_task: asyncio.Task) -> None:
        """Cancel ``invoke_task`` with a 5s grace window.

        Defensive: adapters that ignore ``task.cancel()`` shouldn't wedge
        the run. Polls for up to 5s; if the task still isn't done, we
        log a warning and return (the orphaned task is left for the
        event loop to reap).
        """
        if invoke_task.done():
            return
        invoke_task.cancel()
        import time as _time

        deadline = _time.monotonic() + 5.0
        while _time.monotonic() < deadline:
            if invoke_task.done():
                return
            await asyncio.sleep(0.05)
        log.warning(
            "SequentialExecutor: adapter ignored task.cancel(); abandoning after 5s grace window"
        )

    @staticmethod
    async def _mark_cancelled_if_live(
        *,
        task_id: str,
        steerer: Steerer,
        session: Session,
        cancel_reason: str = "",
    ) -> None:
        """Transition a not-yet-terminal task to CANCELLED.

        ``cancel_reason`` (goldfive#205): structured reason stamped on the
        emitted ``TaskCancelled``. Defaults to a generic
        ``user_cancel:cancelled_by_control`` when the caller does not
        pass something more specific (e.g. an annotation_id).
        """
        if session.plan is None:
            return
        for t in session.plan.tasks:
            if t.id == task_id:
                if t.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                ):
                    return
                reason_value = cancel_reason or "user_cancel:cancelled_by_control"
                await steerer.transition(
                    task_id,
                    TaskStatus.CANCELLED,
                    detail="cancelled by control",
                    cancel_reason=reason_value,
                    session=session,
                )
                return

    @staticmethod
    async def _apply_steer(
        message: object,
        *,
        steerer: Steerer,
        session: Session,
    ) -> None:
        """Feed a STEER :class:`ControlMessage` to the steerer."""
        try:
            await steerer.observe(message, session)
        except Exception as exc:  # noqa: BLE001
            log.warning("SequentialExecutor: steerer.observe(STEER) raised: %s", exc)

    @staticmethod
    def _compose_nudge_replay_message(nudges: list[str]) -> str:
        """Wrap queued nudges in a goldfive-authored framing header.

        Mirrors :meth:`_compose_steer_restart_message` but for the
        autonomous-nudge path (goldfive#202). The LLM sees:

        * A header distinguishing this from a fresh user turn: the
          operator did not intervene; goldfive detected drift (e.g.
          repeated ``report_task_completed`` calls), revised the plan,
          and is directing the coordinator to the new next task.
        * Each queued nudge verbatim (short, action-focused strings
          composed by :func:`compose_corrective_user_message`).
        * A brief instruction to continue with the revised plan.

        The scoped replay path is the carefully-narrowed successor to
        the blanket follow-up loop that goldfive#163 removed. #163's
        removal was correct when every PENDING task triggered a
        follow-up; this path only fires when the STEERER explicitly
        queued a nudge in response to a tracked drift + plan
        revision — not on every PENDING-at-invocation-end.
        """
        body = "\n".join(f"- {n}" for n in nudges if n)
        return (
            "[GOLDFIVE PLAN REVISION — replace superseded task(s)]\n"
            "\n"
            "Goldfive detected drift during the prior turn and revised "
            "the active plan. The task you were last working on has "
            "been superseded by a replacement. Proceed with the "
            "replacement; do NOT retry the prior task.\n"
            "\n"
            f"{body}\n"
            "\n"
            "Notes:\n"
            "- Continue the run with the revised plan. Resume with the "
            "next unfinished task — the replacement mentioned above, "
            "or any other PENDING task your tree still owns.\n"
            "- Do not re-invoke reporting tools for the superseded task."
        )

    @staticmethod
    def _compose_steer_restart_message(
        msg: object,
        *,
        fallback: str,
        source: str = "user",
        superseded_task_ids: list[str] | None = None,
        replacement_task_ids: list[str] | None = None,
    ) -> str:
        """Wrap a STEER body in a goldfive-authored override header.

        The STEER payload shape on the goldfive side is
        ``{"note": str, "suggested_action": str}`` — the harmonograf
        server maps ``PostAnnotation(body=...)`` onto
        ``ControlEvent.steer.note`` (see
        ``server/harmonograf_server/rpc/frontend.py``), and the
        control bridge rehydrates that into
        ``ControlMessage.payload["note"]``. We also accept
        ``payload["body"]`` as a courtesy for callers building STEER
        messages directly from annotation shapes, so either key works.

        ``source`` (goldfive-steer-unification) selects the framing:

        * ``"user"`` (default) — ``[USER STEERING CONTROL — supersedes
          prior task context]`` header: this is an operator override,
          not a fresh user turn.
        * ``"goldfive"`` — ``[GOLDFIVE STEERING CONTROL — supersedes
          prior task context]`` header: goldfive's drift ladder
          promoted a detector signal into a steer and is directing the
          coordinator away from contaminated work. When the optional
          ``superseded_task_ids`` / ``replacement_task_ids`` lists are
          provided, a task-id block is appended so the LLM knows
          precisely which ids are void and what replaced them.

        Falls back to ``fallback`` when the extracted text is empty,
        then wraps that fallback in the same header so the re-invocation
        always shows the override semantics.
        """
        payload = getattr(msg, "payload", None)
        body = ""
        if isinstance(payload, dict):
            body = str(payload.get("note", "") or payload.get("body", "") or "")
        effective = body or fallback
        # Build the override-framing message. Keep the header line
        # short and machine-readable so prompt templates / tests can
        # match on the prefix ("[USER STEERING CONTROL" or
        # "[GOLDFIVE STEERING CONTROL").
        source_norm = (source or "user").strip().lower()
        if source_norm == "goldfive":
            header = "[GOLDFIVE STEERING CONTROL — supersedes prior task context]"
            extra_notes: list[str] = []
            sup = [str(t) for t in (superseded_task_ids or []) if t]
            rep = [str(t) for t in (replacement_task_ids or []) if t]
            if sup:
                extra_notes.append(
                    "- Superseded task ids (do NOT resume these): "
                    + ", ".join(sup)
                )
            if rep:
                extra_notes.append(
                    "- Replacement task ids (pick these up instead): "
                    + ", ".join(rep)
                )
            extra_block = ("\n" + "\n".join(extra_notes)) if extra_notes else ""
            return (
                f"{header}\n"
                "\n"
                f"{effective}\n"
                "\n"
                "Notes:\n"
                "- Goldfive detected drift in the prior turn's activity. "
                "Prior research, partial work, or planned tasks from the "
                "contaminated step are superseded unless this message "
                "explicitly references them.\n"
                "- Proceed with the corrective direction above. Do not "
                "retry the superseded task."
                f"{extra_block}"
            )
        return (
            "[USER STEERING CONTROL — supersedes prior task context]\n"
            "\n"
            f"{effective}\n"
            "\n"
            "Notes:\n"
            "- Prior research, partial work, or planned tasks from the pre-steer "
            "conversation are superseded unless this message explicitly "
            "references them.\n"
            "- Proceed with the new direction. Do not continue prior work unless "
            "doing so directly serves this steer."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_next_task(plan: Plan) -> Task | None:
    """Return the first PENDING task whose deps are all COMPLETED.

    Walks ``plan.topological_stages()`` in order. Status authority is
    ``Task.status`` on the plan's tasks (the steerer mutates these in
    place via reporting tool handlers).
    """
    tasks_by_id = {t.id: t for t in plan.tasks if t.id}
    deps_by_task: dict[str, list[str]] = {}
    for e in plan.edges:
        deps_by_task.setdefault(e.to_task_id, []).append(e.from_task_id)

    for stage in plan.topological_stages():
        for task in stage:
            if task.status != TaskStatus.PENDING:
                continue
            blocked = False
            for dep_id in deps_by_task.get(task.id, []):
                dep = tasks_by_id.get(dep_id)
                if dep is None or dep.status != TaskStatus.COMPLETED:
                    blocked = True
                    break
            if not blocked:
                return task
    return None


def _find_task(plan: Plan, task_id: str) -> Task | None:
    for t in plan.tasks:
        if t.id == task_id:
            return t
    return None


# Precompiled once-per-module: matches a single ``retry_`` or
# ``retry<N>_`` prefix (N >= 1), e.g. ``retry_``, ``retry2_``, ``retry42_``.
_RETRY_PREFIX_RE = re.compile(r"^retry(?:\d+)?_")


def _lineage_root(task_id: str) -> str:
    """Return ``task_id`` with any chain of retry prefixes stripped.

    Goldfive does not pin a single spelling for retry-task ids — an LLM
    planner may emit ``retry_t0``, ``retry2_t0``, or even nested forms
    like ``retry_retry_t0`` when regenerating a plan after a failure.
    For per-lineage budgeting we collapse all of those to the same root
    (here: ``t0``) by repeatedly stripping a leading ``retry_`` or
    ``retry<N>_`` prefix.

    Empty / None-like ids pass through unchanged so we do not crash on
    malformed plans.
    """
    if not task_id:
        return task_id
    root = task_id
    # Bound the loop so a pathological id like
    # "retry_retry_retry_..." can't spin forever.
    for _ in range(16):
        stripped = _RETRY_PREFIX_RE.sub("", root, count=1)
        if stripped == root:
            break
        root = stripped
    return root


def _any_failed(plan: Plan) -> bool:
    return any(t.status == TaskStatus.FAILED for t in plan.tasks)


def _has_live_pending_or_running(plan: Plan) -> bool:
    """Return True if the plan has any PENDING or RUNNING task left.

    Used by the overlay's nudge-replay gate (goldfive#202): a queued
    nudge should only trigger a re-invoke when there is actually
    outstanding work for the coordinator to do. Guards against replaying
    against a terminated plan.
    """
    return any(t.status in (TaskStatus.PENDING, TaskStatus.RUNNING) for t in plan.tasks)


def _has_live_replacement(plan: Plan, failed: Task) -> bool:
    """Return True iff ``failed`` has a *live* replacement task in ``plan``.

    A replacement task is one the planner spawned to supersede
    ``failed`` — a forward-progress successor, not a predecessor. We
    require it to be PENDING or RUNNING (never COMPLETED) so a
    COMPLETED sibling lineage peer (``t0`` when ``retry_retry_t0`` is
    the failure) does NOT mask the failure — that's a predecessor, not
    a replacement. See goldfive#202.

    Matched via one of two id conventions goldfive's refine path
    produces (structural inference; no proto ``replaces`` field):

    * Shared retry lineage: ``_lineage_root(R.id) == _lineage_root(failed.id)``.
      Catches ``retry_<id>``, ``retry2_<id>`` etc. — the pattern the
      refine system prompt historically emitted (see PLAN-LIFECYCLE.md
      §7.3).
    * Versioned replacement: ``R.id`` starts with ``<failed.id>_``
      (``define_structure`` → ``define_structure_v2``, ``..._retry``,
      etc.). Empirically the shape LLM planners emit when the refine
      prompt does NOT encode a ``retry_`` convention.

    Additionally require ``R.assignee_agent_id == failed.assignee_agent_id``
    when both are populated, so a task owned by a different agent doesn't
    accidentally mask a genuine failure.

    Lives in the executor (not a proto field on :class:`Task`) to avoid
    a cross-cutting contract change through planner JSON shapes and
    prompt templates. Structural inference is adequate: the refine
    output is always validated against :meth:`Plan.validate`, so the
    shapes here are the shapes the planner actually emits.
    """
    failed_root = _lineage_root(failed.id)
    for r in plan.tasks:
        if r.id == failed.id or not r.id:
            continue
        # FAILED / CANCELLED are obviously not replacements.
        if r.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            continue
        # Convention match. The two patterns have different chronological
        # semantics w.r.t. the FAILED task:
        # * Versioned pattern (``<failed_id>_<suffix>`` like
        #   ``define_structure_v2``) is conventionally a SUCCESSOR — a
        #   planner would not emit ``..._v2`` before ``v1`` had run —
        #   so PENDING / RUNNING / COMPLETED all count as a live
        #   replacement.
        # * Shared retry lineage (``retry_<id>``, ``retry2_<id>``)
        #   is CHRONOLOGY-AMBIGUOUS — ``t0`` COMPLETED + ``retry_t0``
        #   COMPLETED + ``retry_retry_t0`` FAILED describes predecessors
        #   of the FAILED task, not replacements. So retry-lineage peers
        #   only count when PENDING / RUNNING (clear "still to run").
        versioned = r.id.startswith(f"{failed.id}_")
        same_lineage = (
            not versioned  # don't double-count versioned-and-lineage
            and _lineage_root(r.id) == failed_root
            and r.id != failed.id
        )
        if versioned:
            pass  # any non-FAILED / non-CANCELLED state counts
        elif same_lineage:
            if r.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                continue
        else:
            continue
        # Assignee scoping when both are populated — narrow false positives.
        if (
            failed.assignee_agent_id
            and r.assignee_agent_id
            and r.assignee_agent_id != failed.assignee_agent_id
        ):
            continue
        return True
    return False


def _fatally_failed_task_ids(plan: Plan) -> list[str]:
    """Return FAILED task ids with NO live replacement in ``plan``.

    Used by :class:`SequentialExecutor`'s ``fail_fast`` gate so a FAILED
    task whose refine-time replacement is still live (PENDING / RUNNING
    / COMPLETED) does not abort the run. Without this check,
    ``fail_fast=True`` would see the refine's FAILED mark and abort
    before the replacement gets a chance to execute — defeating the
    point of the refine. See goldfive#202.
    """
    fatal: list[str] = []
    for t in plan.tasks:
        if t.status != TaskStatus.FAILED:
            continue
        if _has_live_replacement(plan, t):
            continue
        fatal.append(t.id or "")
    return fatal


def _pending_task_ids(plan: Plan) -> list[str]:
    """Return the ids of every task still in PENDING state.

    Used by the reachability audit at loop exit: if ``_pick_next_task``
    returned None but some tasks are still PENDING, those tasks are
    orphaned (every path to them crosses a CANCELLED / FAILED predecessor
    that no refine replaced). The audit then cancels them in place so
    sinks see a coherent plan-end state instead of "stuck PENDING forever."
    """
    return [t.id for t in plan.tasks if t.status == TaskStatus.PENDING and t.id]


async def _emit_pipeline_failure_drift(
    *,
    session: Session,
    sinks: list[EventSink],
    task_id: str,
    reason: str,
) -> None:
    """Emit an INFO ``CUSTOM`` drift when the drift pipeline itself raised.

    Mirrors the helper in
    :mod:`goldfive.executors.parallel`: a bug in the steerer's observe
    path must not silently disappear. INFO severity so the run does not
    trigger another refine — the goal is to make the plumbing failure
    visible, not to recover from it. Sinks that care can filter on
    the ``drift_pipeline_failed:`` detail prefix. See goldfive#134.

    Uses the steerer's proto-enum mapping shape
    (``DRIFT_KIND_<NAME>``) so the emitted envelope carries a real
    enum value rather than UNSPECIFIED — the
    :func:`goldfive.events.drift_detected_event` helper does a
    best-effort lookup that silently drops StrEnum-style values.
    """
    from goldfive.pb.goldfive.v1 import types_pb2

    evt = new_event(session.run_id, session.next_sequence(), session_id=session.id)
    evt.drift_detected.kind = getattr(
        types_pb2,
        f"DRIFT_KIND_{DriftKind.CUSTOM.name}",
        0,
    )
    evt.drift_detected.severity = getattr(
        types_pb2,
        f"DRIFT_SEVERITY_{DriftSeverity.INFO.name}",
        0,
    )
    evt.drift_detected.detail = reason
    evt.drift_detected.current_task_id = task_id or ""
    try:
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("_emit_pipeline_failure_drift: sink emit raised: %s", exc)


def _plan_divergence_drift_event(session: Session, detail: str) -> object:
    """Build a CRITICAL PLAN_DIVERGENCE ``DriftDetected`` envelope.

    Mirrors ``DefaultSteerer._emit_drift_detected`` but built here so
    the reachability audit can emit a drift event from the executor
    directly (without routing through the steerer, which would
    otherwise try to re-refine). Uses ``types_pb2``'s named enum
    constants so the emitted event carries the real ``PLAN_DIVERGENCE``
    / ``CRITICAL`` enum values rather than ``UNSPECIFIED``.
    """
    from goldfive.pb.goldfive.v1 import types_pb2

    evt = new_event(session.run_id, session.next_sequence(), session_id=session.id)
    evt.drift_detected.kind = getattr(
        types_pb2,
        f"DRIFT_KIND_{DriftKind.PLAN_DIVERGENCE.name}",
        0,
    )
    evt.drift_detected.severity = getattr(
        types_pb2,
        f"DRIFT_SEVERITY_{DriftSeverity.CRITICAL.name}",
        0,
    )
    evt.drift_detected.detail = detail
    evt.drift_detected.current_task_id = session.current_task_id or ""
    return evt


def _outcome_summary(session: Session) -> str:
    """One-line summary for the RunCompleted event."""
    plan = session.plan
    if plan is None:
        return "run completed"
    total = len(plan.tasks)
    completed = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
    return f"{completed}/{total} tasks completed"


def build_task_nudge(task: Task) -> str:
    """Return the canonical next-task nudge string.

    Ported from harmonograf's ``_build_nudge_content`` (minus the
    ADK-specific ``genai_types.Content`` wrapping). Adapters that want
    harmonograf-identical prompting can feed this string to their underlying
    agent as the user turn that precedes each per-task invocation.
    """
    tid = task.id or ""
    title = task.title or ""
    description = task.description or ""
    return (
        f"Continue executing the plan. Your next task is {tid}: {title}. "
        f"{description} "
        "Execute it now, then proceed to the next pending task assigned "
        "to you whose dependencies are complete."
    ).strip()
