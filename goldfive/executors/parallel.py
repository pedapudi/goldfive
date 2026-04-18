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
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from goldfive.events import emit as emit_event
from goldfive.events import make_event
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

log = logging.getLogger(__name__)

_WARNING_RANK = severity_rank(DriftSeverity.WARNING)


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
    ) -> ExecutionOutcome:
        session.plan = plan

        # Bind sinks + planner into the steerer so its own observe/
        # transition calls fan out alongside ours.
        try:
            steerer.bind(sinks=sinks, planner=planner)
        except Exception as exc:  # noqa: BLE001
            log.debug("ParallelDAGExecutor: steerer.bind raised: %s", exc)

        await self._emit(
            session,
            sinks,
            kind="RunStarted",
            payload={"plan_id": plan.id, "goal_ids": list(plan.goal_ids)},
        )

        refinements_used = 0
        completed_stage_ids: set[str] = set()
        abort_reason = ""
        try:
            while True:
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

                stage_index = len(completed_stage_ids)  # ordinal-ish, for logs
                await self._emit(
                    session,
                    sinks,
                    kind="StageStarted",
                    payload={
                        "stage_index": stage_index,
                        "task_ids": [t.id for t in stage_tasks],
                    },
                )

                stage_results, drift = await self._run_stage(
                    stage_tasks, session, adapter, steerer, sinks
                )

                # Fold terminal statuses + results back into the session.
                for task, inv, _task_drift, error in stage_results:
                    if error is not None and not isinstance(error, asyncio.CancelledError):
                        task.status = TaskStatus.FAILED
                    elif isinstance(error, asyncio.CancelledError):
                        task.status = TaskStatus.CANCELLED
                    elif inv is not None and inv.error is not None:
                        task.status = TaskStatus.FAILED
                    else:
                        task.status = TaskStatus.COMPLETED
                        if inv is not None:
                            session.completed_results[task.id] = inv.text
                    completed_stage_ids.add(task.id)

                await self._emit(
                    session,
                    sinks,
                    kind="StageCompleted",
                    payload={
                        "stage_index": stage_index,
                        "task_ids": [t.id for t, *_ in stage_results],
                    },
                )

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
                        await self._emit(
                            session,
                            sinks,
                            kind="PlanRevised",
                            payload={
                                "plan_id": refined.id,
                                "revision_index": refined.revision_index,
                                "reason": refined.revision_reason,
                                "kind": refined.revision_kind,
                                "severity": refined.revision_severity,
                            },
                        )
                        # Falls through to loop top: stages recomputed.
                        continue

                # No drift (or no refinement): continue to next stage.
        except BaseException as exc:  # noqa: BLE001 — propagate reason, then re-raise
            abort_reason = f"{type(exc).__name__}: {exc}"
            await self._emit(
                session,
                sinks,
                kind="RunAborted",
                payload={"reason": abort_reason},
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            return ExecutionOutcome(success=False, session=session, reason=abort_reason)

        if abort_reason:
            await self._emit(
                session,
                sinks,
                kind="RunAborted",
                payload={"reason": abort_reason},
            )
            return ExecutionOutcome(success=False, session=session, reason=abort_reason)

        await self._emit(
            session,
            sinks,
            kind="RunCompleted",
            payload={"completed_task_ids": sorted(completed_stage_ids)},
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
    ) -> tuple[list[_StageResult], DriftEvent | None]:
        """Run one stage's tasks concurrently.

        Returns the per-task results in stage order plus the first drift
        event (at severity >= WARNING) observed while the stage ran, or
        ``None`` if the stage finished clean. Under ``cancel_stage`` the
        surviving in-flight tasks are cancelled as soon as a qualifying
        drift is spotted.
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
        results: dict[str, _StageResult] = {}
        pending: set[asyncio.Task[_StageResult]] = set(aio_tasks)

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for d in done:
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
                            for p in pending:
                                p.cancel()
                            # Drain the cancelled tasks so they don't leak.
                            cancelled_done, _ = await asyncio.wait(
                                pending, return_when=asyncio.ALL_COMPLETED
                            )
                            for cd in cancelled_done:
                                t2 = task_by_aio[cd]
                                try:
                                    results[t2.id] = cd.result()
                                except asyncio.CancelledError:
                                    results[t2.id] = (
                                        t2, None, None, asyncio.CancelledError()
                                    )
                                except BaseException as exc:  # noqa: BLE001
                                    results[t2.id] = (t2, None, None, exc)
                            pending = set()
        except asyncio.CancelledError:
            # Outer cancellation: make sure every inner task is reaped.
            for p in pending:
                p.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise

        # Preserve stage-input order in the return.
        ordered: list[_StageResult] = [
            results[t.id] for t in stage_tasks if t.id in results
        ]
        return ordered, first_drift

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
    # Event emission
    # ------------------------------------------------------------------

    async def _emit(
        self,
        session: Session,
        sinks: list[EventSink],
        *,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        ev = make_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            kind=kind,
            payload=payload,
        )
        await emit_event(sinks, ev)


# Runtime conformance check — the class must satisfy the Executor protocol.
assert isinstance(ParallelDAGExecutor(), Executor)
