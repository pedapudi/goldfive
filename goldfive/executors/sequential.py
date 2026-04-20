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
* Budget the total number of adapter invocations with ``max_plan_reinvocations``
  so a stuck agent cannot spin forever.
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
from typing import TYPE_CHECKING

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
from goldfive.results import ExecutionOutcome
from goldfive.types import DriftKind, DriftSeverity, Plan, Session, Task, TaskStatus

if TYPE_CHECKING:
    from goldfive.control import ControlChannel

log = logging.getLogger(__name__)


class SequentialExecutor(Executor):
    """Single-threaded executor that drives one task at a time.

    Parameters
    ----------
    max_plan_reinvocations:
        Upper bound on the number of adapter ``invoke()`` calls a single
        :meth:`run` may issue. Each eligible task consumes one invocation;
        when drift triggers a plan revision, the new plan's remaining tasks
        share the same budget. Defaults to ``32`` — comfortably covers a
        plan with 10+ tasks plus a few refinement cycles. A stuck agent
        still aborts via ``fail_fast`` or (in pathological cases) the
        budget; the old default of ``3`` was tuned to the harmonograf
        re-invocation cap and surprised callers running realistic plans.
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
        max_plan_reinvocations: int = 32,
        max_retries_per_task_lineage: int = 3,
        fail_fast: bool = True,
    ) -> None:
        self.max_plan_reinvocations = int(max_plan_reinvocations)
        self.max_retries_per_task_lineage = int(max_retries_per_task_lineage)
        self.fail_fast = bool(fail_fast)

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
    ) -> ExecutionOutcome:
        """Walk ``plan`` end-to-end, driving ``adapter`` once per eligible task.

        Returns an :class:`ExecutionOutcome` summarizing whether the run
        succeeded and carrying the final ``session`` so callers can inspect
        completed results / agent notes.
        """
        # Pin the plan onto the session so the steerer / reporting handlers
        # see the same object the executor is iterating.
        session.plan = plan

        # Wire sinks + planner into the steerer so reporting-tool handlers
        # can emit events and trigger planner.refine on drift.
        steerer.bind(sinks=sinks, planner=planner)

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

        # Cap by max_plan_reinvocations: this is the number of adapter
        # invocations we'll allow for the whole run. Each eligible task
        # burns one; mid-run plan revisions do not refund any.
        while invocations < self.max_plan_reinvocations:
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
                "SequentialExecutor: invoking task=%s (invocation %d/%d, "
                "lineage_root=%s, lineage_count=%d/%d)",
                task.id,
                invocations,
                self.max_plan_reinvocations,
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
                await self._mark_cancelled_if_live(
                    task_id=task.id, steerer=steerer, session=session
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
                await self._mark_cancelled_if_live(
                    task_id=task.id, steerer=steerer, session=session
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
                    log.debug(
                        "SequentialExecutor: steerer.observe raised: %s",
                        observe_exc,
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
                ),
            )
            return ExecutionOutcome(
                success=False, session=session, reason=failure_reason or "run aborted"
            )

        # If we exhausted the invocation budget with work still pending,
        # that is also a failure (stuck agent).
        remaining = _pick_next_task(session.plan or plan)
        if invocations >= self.max_plan_reinvocations and remaining is not None:
            reason = (
                f"exhausted max_plan_reinvocations={self.max_plan_reinvocations} "
                f"with pending task {remaining.id}"
            )
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=reason,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=reason)

        # If fail_fast=False and some task ended FAILED, the run is only
        # "successful" in the best-effort sense. Match harmonograf: report
        # success=False with a reason when any task is terminal-failed.
        any_failed = _any_failed(session.plan or plan)
        if any_failed and not self.fail_fast:
            reason = "one or more tasks failed (fail_fast=False)"
            await emit(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=reason,
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
            for orphan_id in orphaned:
                await steerer.transition(
                    orphan_id,
                    TaskStatus.CANCELLED,
                    detail="orphaned by plan revision failure",
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
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=reason)

        await emit(
            sinks,
            run_completed_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                outcome_summary=_outcome_summary(session),
            ),
        )
        return ExecutionOutcome(success=True, session=session)

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
        CANCEL) arrives.
        """
        if control is None:
            return False, None

        outcomes = await drain_controls(control, session=session, steerer=steerer, sinks=sinks)

        cancel_run = False
        cancel_reason = ""
        steer_msg: object | None = None
        paused = False
        for o in outcomes:
            if o.cancel_run:
                cancel_run = True
                cancel_reason = o.cancel_reason
                break
            if o.steer_message is not None:
                steer_msg = o.steer_message
            if o.request_pause:
                paused = True
            if o.request_resume:
                paused = False

        if cancel_run:
            raise _ControlCancelled(cancel_reason or "cancelled by control")

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
                return ("cancelled", outcome.cancel_reason or "cancelled by control")

            if outcome.steer_message is not None:
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
    ) -> None:
        """Transition a not-yet-terminal task to CANCELLED."""
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
                await steerer.transition(
                    task_id,
                    TaskStatus.CANCELLED,
                    detail="cancelled by control",
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


def _pending_task_ids(plan: Plan) -> list[str]:
    """Return the ids of every task still in PENDING state.

    Used by the reachability audit at loop exit: if ``_pick_next_task``
    returned None but some tasks are still PENDING, those tasks are
    orphaned (every path to them crosses a CANCELLED / FAILED predecessor
    that no refine replaced). The audit then cancels them in place so
    sinks see a coherent plan-end state instead of "stuck PENDING forever."
    """
    return [t.id for t in plan.tasks if t.status == TaskStatus.PENDING and t.id]


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

    evt = new_event(session.run_id, session.next_sequence())
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
