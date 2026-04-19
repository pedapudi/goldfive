"""Parallel-DAG executor.

Ports the rigid-DAG batch walker from harmonograf's
``_run_orchestrator_walker`` (``client/harmonograf_client/agent.py``
lines 1260-1450) onto the goldfive ``Executor`` protocol. ADK-specific
context vars are dropped; all state flows through the ``Session`` object
the caller provides.

The executor walks :meth:`Plan.topological_stages`. Each stage is a set
of tasks whose predecessors already finished; within a stage we
``asyncio.gather`` calls to ``adapter.invoke(task, session)``, bounded
by an optional concurrency cap. Drift detection runs per task via the
bound ``Steerer``; on a drift whose severity is at or above WARNING the
``drift_policy`` decides whether to (a) cancel the remaining in-flight
tasks in the stage or (b) let siblings finish before refining. Plan
refinement never happens mid-stage — that matches the harmonograf
walker's semantics and keeps the DAG walk deterministic.

Lifecycle ownership: ``RunStarted`` is emitted by :class:`~goldfive.Runner`,
not by the executor. The executor owns the terminal ``RunCompleted`` /
``RunAborted`` and the ``PlanRevised`` events it generates when the
walker swaps in a refined plan.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Literal

from goldfive.events import (
    emit as emit_event,
)
from goldfive.events import (
    plan_revised_event,
    run_aborted_event,
    run_completed_event,
)
from goldfive.executors._control import (
    ControlOutcome,
    _ControlCancelled,
    dispatch_control,
    drain_controls,
)
from goldfive.protocols import (
    AgentAdapter,
    EventSink,
    Executor,
    Planner,
    Steerer,
)
from goldfive.results import ExecutionOutcome, InvocationResult
from goldfive.types import (
    DriftEvent,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskStatus,
    severity_rank,
)

if TYPE_CHECKING:
    from goldfive.control import ControlChannel

log = logging.getLogger(__name__)

_WARNING_RANK = severity_rank(DriftSeverity.WARNING)

_TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


# A (task, result_or_drift) carrier for stage gather. Using a dataclass
# would add import weight; a tuple is fine and stays local to this file.
_StageResult = tuple[Task, InvocationResult | None, DriftEvent | None, BaseException | None]


class ParallelDAGExecutor:
    """Parallel rigid-DAG executor.

    Parameters
    ----------
    max_concurrency:
        Upper bound on concurrent invocations within a single stage. ``0``
        disables the cap (full-fan-out via ``asyncio.gather``). Set to
        ``1`` to force sequential behaviour (useful for tests and for
        adapters that aren't re-entrant).
    drift_policy:
        * ``"cancel_stage"`` — first drift at ``>= WARNING`` cancels
          every other in-flight task in the stage; refinement runs
          before the next stage.
        * ``"finish_stage"`` (default) — the stage is allowed to
          complete; refinement runs before the next stage.
    max_plan_reinvocations:
        Safety cap on the number of times the planner may replace the
        plan during a single ``run()``. Protects against refine loops.
    """

    def __init__(
        self,
        max_concurrency: int = 0,
        drift_policy: Literal["cancel_stage", "finish_stage"] = "finish_stage",
        max_plan_reinvocations: int = 3,
    ) -> None:
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be >= 0")
        if drift_policy not in ("cancel_stage", "finish_stage"):
            raise ValueError(
                f"drift_policy must be 'cancel_stage' or 'finish_stage', got {drift_policy!r}"
            )
        if max_plan_reinvocations < 0:
            raise ValueError("max_plan_reinvocations must be >= 0")
        self.max_concurrency = max_concurrency
        self.drift_policy: Literal["cancel_stage", "finish_stage"] = drift_policy
        self.max_plan_reinvocations = max_plan_reinvocations

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
        session.plan = plan

        # Bind sinks + planner into the steerer so its own observe/
        # transition calls fan out alongside ours.
        try:
            steerer.bind(sinks=sinks, planner=planner)
        except Exception as exc:  # noqa: BLE001
            log.debug("ParallelDAGExecutor: steerer.bind raised: %s", exc)

        # NOTE: RunStarted is emitted by Runner, not by the executor. See
        # Runner._emit_run_started.

        refinements_used = 0
        completed_stage_ids: set[str] = set()
        abort_reason = ""
        try:
            while True:
                # Pre-stage: drain pending controls; honour PAUSE by
                # blocking on the channel until RESUME/CANCEL/STEER
                # arrives.
                try:
                    stop, pending_steer = await self._apply_pre_stage_controls(
                        control=control,
                        session=session,
                        steerer=steerer,
                        sinks=sinks,
                    )
                except _ControlCancelled as cancelled:
                    abort_reason = cancelled.detail
                    break
                if stop:
                    abort_reason = "cancelled by control"
                    break
                if pending_steer is not None:
                    await self._apply_steer(
                        pending_steer, steerer=steerer, session=session
                    )
                    # Fall through: refresh plan on next iteration.

                # Recompute stages from current plan each outer iteration;
                # tasks already completed in previous stages are filtered
                # out by ``topological_stages`` (they sit in a terminal
                # status). This mirrors harmonograf's approach of
                # re-querying plan state after every batch.
                current_plan = session.plan or plan
                stages = current_plan.topological_stages()
                # Drop any stage composed entirely of tasks we already ran
                # (defensive — planner.refine may leave older tasks in).
                pending_stages = [
                    stage
                    for stage in stages
                    if any(t.id not in completed_stage_ids for t in stage)
                ]
                if not pending_stages:
                    break

                stage = pending_stages[0]
                # Filter already-completed tasks within this stage so we
                # don't re-invoke them after a refinement.
                stage_tasks = [t for t in stage if t.id not in completed_stage_ids]
                if not stage_tasks:
                    continue

                stage_results, drift, control_outcome = await self._run_stage(
                    stage_tasks, session, adapter, steerer, sinks, control
                )

                if control_outcome is not None and control_outcome.cancel_run:
                    abort_reason = (
                        control_outcome.cancel_reason or "cancelled by control"
                    )
                    for task, _inv, _drift, _err in stage_results:
                        await self._mark_cancelled_if_live(
                            task_id=task.id, steerer=steerer, session=session
                        )
                    break

                # Fold terminal statuses + results back into the session.
                # If the agent already transitioned the task via reporting
                # tools (status is terminal), leave it alone. Otherwise
                # auto-transition on its behalf so clean returns count as
                # COMPLETED, not silently PENDING/RUNNING.
                for task, inv, _task_drift, error in stage_results:
                    already_terminal = task.status in _TERMINAL_TASK_STATUSES
                    if error is not None and not isinstance(error, asyncio.CancelledError):
                        if not already_terminal:
                            await steerer.transition(
                                task.id,
                                TaskStatus.FAILED,
                                detail=str(error),
                                session=session,
                            )
                    elif isinstance(error, asyncio.CancelledError):
                        if not already_terminal:
                            task.status = TaskStatus.CANCELLED
                    elif inv is not None and inv.error is not None:
                        if not already_terminal:
                            await steerer.transition(
                                task.id,
                                TaskStatus.FAILED,
                                detail=str(inv.error),
                                session=session,
                            )
                    else:
                        summary = inv.text if inv is not None else ""
                        if not already_terminal:
                            await steerer.transition(
                                task.id,
                                TaskStatus.COMPLETED,
                                detail=summary,
                                session=session,
                            )
                        if inv is not None:
                            session.completed_results[task.id] = inv.text
                    completed_stage_ids.add(task.id)

                # If a STEER arrived mid-stage, apply it now (stage was
                # cancelled by _run_stage, so the fold-back above has
                # already recorded the tasks as CANCELLED).
                if (
                    control_outcome is not None
                    and control_outcome.steer_message is not None
                ):
                    await self._apply_steer(
                        control_outcome.steer_message,
                        steerer=steerer,
                        session=session,
                    )
                    # New plan installed on session; drop the stage-level
                    # drift (if any) so we don't double-refine.
                    drift = None

                if drift is not None:
                    if refinements_used >= self.max_plan_reinvocations:
                        log.warning(
                            "ParallelDAGExecutor: drift detected but reinvocation "
                            "budget exhausted (%d) — aborting",
                            self.max_plan_reinvocations,
                        )
                        abort_reason = (
                            f"plan reinvocation budget exhausted "
                            f"({self.max_plan_reinvocations})"
                        )
                        break

                    refined = await self._refine(plan=session.plan or plan,
                                                 drift=drift, planner=planner,
                                                 session=session)
                    if refined is not None and refined is not (session.plan or plan):
                        refinements_used += 1
                        session.plan = refined
                        await emit_event(
                            sinks,
                            plan_revised_event(
                                run_id=session.run_id,
                                sequence=session.next_sequence(),
                                plan=refined,
                                drift=drift,
                            ),
                        )
                        # Falls through to loop top: stages recomputed.
                        continue

                # No drift (or no refinement): continue to next stage.
        except BaseException as exc:  # noqa: BLE001 — propagate reason, then re-raise
            abort_reason = f"{type(exc).__name__}: {exc}"
            await emit_event(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=abort_reason,
                ),
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            return ExecutionOutcome(success=False, session=session, reason=abort_reason)

        if abort_reason:
            await emit_event(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=abort_reason,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=abort_reason)

        await emit_event(
            sinks,
            run_completed_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                outcome_summary=_outcome_summary(completed_stage_ids),
            ),
        )
        return ExecutionOutcome(success=True, session=session)

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    async def _run_stage(
        self,
        stage_tasks: list[Task],
        session: Session,
        adapter: AgentAdapter,
        steerer: Steerer,
        sinks: list[EventSink],
        control: ControlChannel | None = None,
    ) -> tuple[list[_StageResult], DriftEvent | None, ControlOutcome | None]:
        """Run one stage's tasks concurrently.

        Returns ``(results, first_drift, control_outcome)``:

        * ``results`` — the per-task results in stage order.
        * ``first_drift`` — the first drift (at severity >= WARNING)
          observed while the stage ran, or ``None``.
        * ``control_outcome`` — the first cancelling :class:`ControlOutcome`
          (cancel_run or steer_message) received mid-stage, or ``None``.
          Callers inspect this to decide whether to abort the run or
          apply a STEER before moving on.

        Under ``cancel_stage`` the surviving in-flight tasks are
        cancelled as soon as a qualifying drift is spotted. A CANCEL
        or STEER control also cancels every in-flight task in the
        stage, regardless of drift_policy.
        """
        sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(self.max_concurrency) if self.max_concurrency > 0 else None
        )

        # Each per-task coroutine invokes the adapter, passes the result
        # through the steerer's drift detector, and returns a structured
        # tuple. Exceptions from ``invoke`` are captured, not raised —
        # gather() with ``return_exceptions=True`` still lets a single
        # failure cascade into cancellation when drift_policy demands.
        async def run_one(task: Task) -> _StageResult:
            if sem is not None:
                await sem.acquire()
            try:
                task.status = TaskStatus.RUNNING
                try:
                    inv: InvocationResult | None = await adapter.invoke(task, session)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # noqa: BLE001
                    return (task, None, None, exc)

                drift: DriftEvent | None = None
                try:
                    # Hand the result to the steerer for observation
                    # (sinks + any book-keeping) and drift detection.
                    await steerer.observe(inv, session)
                    drift = steerer.detect_drift(inv, session)
                except asyncio.CancelledError:
                    raise
                except Exception as detect_exc:  # noqa: BLE001
                    log.debug(
                        "ParallelDAGExecutor: steerer.detect_drift raised: %s",
                        detect_exc,
                    )
                    drift = None

                # Adapter-reported error counts as drift only if the
                # steerer flagged it; we don't synthesize one here.
                return (task, inv, drift, None)
            finally:
                if sem is not None:
                    sem.release()

        # Create tasks in deterministic order so cancellation and gather
        # ordering match the input stage ordering.
        aio_tasks: list[asyncio.Task[_StageResult]] = [
            asyncio.create_task(run_one(t), name=f"goldfive-task-{t.id}")
            for t in stage_tasks
        ]
        task_by_aio: dict[asyncio.Task[_StageResult], Task] = dict(
            zip(aio_tasks, stage_tasks, strict=True)
        )

        first_drift: DriftEvent | None = None
        stage_control_outcome: ControlOutcome | None = None
        results: dict[str, _StageResult] = {}
        pending: set[asyncio.Task[_StageResult]] = set(aio_tasks)

        recv_task: asyncio.Task | None = None

        def _ensure_recv_task() -> asyncio.Task | None:
            nonlocal recv_task
            if control is None:
                return None
            if recv_task is None or recv_task.done():
                recv_task = asyncio.create_task(
                    control.receive(), name="goldfive-stage-control"
                )
            return recv_task

        async def _cancel_stage_tasks() -> None:
            if not pending:
                return
            for p in pending:
                p.cancel()
            cancelled_done, _still = await asyncio.wait(
                pending, return_when=asyncio.ALL_COMPLETED
            )
            for cd in cancelled_done:
                t2 = task_by_aio[cd]
                try:
                    results[t2.id] = cd.result()
                except asyncio.CancelledError:
                    results[t2.id] = (t2, None, None, asyncio.CancelledError())
                except BaseException as exc:  # noqa: BLE001
                    results[t2.id] = (t2, None, None, exc)
            pending.clear()

        try:
            while pending:
                waitables: set[asyncio.Future] = set(pending)
                rt = _ensure_recv_task()
                if rt is not None:
                    waitables.add(rt)
                done, _pending_set = await asyncio.wait(
                    waitables, return_when=asyncio.FIRST_COMPLETED
                )

                # Process the control receive first so a mid-stage CANCEL
                # / STEER short-circuits before we fold in stage results.
                if rt is not None and rt in done:
                    try:
                        msg = rt.result()
                    except BaseException:  # noqa: BLE001
                        msg = None
                    recv_task = None
                    if msg is not None:
                        outcome = await dispatch_control(
                            msg,
                            session=session,
                            steerer=steerer,
                            sinks=sinks,
                        )
                        try:
                            await control.ack(outcome.ack)  # type: ignore[union-attr]
                        except Exception:  # noqa: BLE001
                            pass
                        if outcome.cancel_run or outcome.steer_message is not None:
                            # Fold any tasks that already completed in
                            # this done-set before cancelling the stage.
                            for d in done - {rt}:
                                task = task_by_aio[d]
                                pending.discard(d)
                                try:
                                    results[task.id] = d.result()
                                except asyncio.CancelledError:
                                    results[task.id] = (
                                        task, None, None, asyncio.CancelledError()
                                    )
                                except BaseException as exc:  # noqa: BLE001
                                    results[task.id] = (task, None, None, exc)
                            stage_control_outcome = outcome
                            await _cancel_stage_tasks()
                            break
                        # Non-interrupting control (PAUSE/RESUME/
                        # REWIND_TO/STATUS_QUERY/INTERCEPT_TRANSFER):
                        # Apply and keep the stage running. PAUSE mid-
                        # stage lets the stage finish per spec.

                for d in done:
                    if d is rt:
                        continue
                    pending.discard(d)
                    task = task_by_aio[d]
                    try:
                        res = d.result()
                    except asyncio.CancelledError:
                        res = (task, None, None, asyncio.CancelledError())
                    except BaseException as exc:  # noqa: BLE001
                        res = (task, None, None, exc)
                    results[task.id] = res

                    _, _, drift, _ = res
                    if (
                        first_drift is None
                        and drift is not None
                        and severity_rank(drift.severity) >= _WARNING_RANK
                    ):
                        first_drift = drift
                        if self.drift_policy == "cancel_stage" and pending:
                            await _cancel_stage_tasks()
        except asyncio.CancelledError:
            # Outer cancellation: make sure every inner task is reaped.
            for p in pending:
                p.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if recv_task is not None and not recv_task.done():
                recv_task.cancel()
            raise
        finally:
            if recv_task is not None and not recv_task.done():
                recv_task.cancel()
                try:
                    await recv_task
                except BaseException:  # noqa: BLE001
                    pass

        # Preserve stage-input order in the return.
        ordered: list[_StageResult] = [
            results[t.id] for t in stage_tasks if t.id in results
        ]
        return ordered, first_drift, stage_control_outcome

    # ------------------------------------------------------------------
    # Plan refinement
    # ------------------------------------------------------------------

    async def _refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        planner: Planner,
        session: Session,
    ) -> Plan | None:
        try:
            refined = await planner.refine(
                plan=plan, drift=drift, goals=list(session.goals)
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("ParallelDAGExecutor: planner.refine raised: %s", exc)
            return None
        return refined

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    async def _apply_pre_stage_controls(
        self,
        *,
        control: ControlChannel | None,
        session: Session,
        steerer: Steerer,
        sinks: list[EventSink],
    ) -> tuple[bool, object | None]:
        """Drain queued controls before the next stage; honour PAUSE.

        Returns ``(cancel_run, steer_message)``. Pre-stage PAUSE blocks
        on :meth:`ControlChannel.receive` until RESUME / CANCEL / STEER
        arrives.
        """
        if control is None:
            return False, None

        outcomes = await drain_controls(
            control, session=session, steerer=steerer, sinks=sinks
        )

        cancel_reason = ""
        cancel_run = False
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

        while paused:
            msg = await control.receive()
            if msg is None:
                paused = False
                break
            outcome = await dispatch_control(
                msg, session=session, steerer=steerer, sinks=sinks
            )
            try:
                await control.ack(outcome.ack)
            except Exception:  # noqa: BLE001
                pass
            if outcome.cancel_run:
                raise _ControlCancelled(
                    outcome.cancel_reason or "cancelled by control"
                )
            if outcome.request_resume:
                paused = False
            if outcome.steer_message is not None:
                steer_msg = outcome.steer_message
                paused = False
            if outcome.rewind_task_id:
                paused = False

        return False, steer_msg

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
                if t.status in _TERMINAL_TASK_STATUSES:
                    return
                try:
                    await steerer.transition(
                        task_id,
                        TaskStatus.CANCELLED,
                        detail="cancelled by control",
                        session=session,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "ParallelDAGExecutor: cancelled transition raised: %s",
                        exc,
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
            log.warning(
                "ParallelDAGExecutor: steerer.observe(STEER) raised: %s", exc
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome_summary(completed_task_ids: set[str]) -> str:
    return f"{len(completed_task_ids)} tasks completed"


# Runtime conformance check — the class must satisfy the Executor protocol.
assert isinstance(ParallelDAGExecutor(), Executor)
