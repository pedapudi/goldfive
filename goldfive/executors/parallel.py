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
import warnings
from typing import TYPE_CHECKING, Any, Literal

from goldfive import _state_audit
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
from goldfive.results import ExecutionOutcome, InvocationResult, evaluate_goal_predicates
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    RefineOutcome,
    Session,
    Task,
    TaskStatus,
    bump_revision,
    channel_processor_active,
    set_session_plan,
    severity_rank,
    with_task_status,
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


# Symbolic cancel-reason for USER_STEER. Mirrors
# :data:`goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER` but duplicated
# as a plain string to avoid importing the optional ADK adapter module
# from the provider-agnostic executor. Keep in sync. See goldfive#139.
_CANCEL_REASON_USER_STEER: str = "user_steer"


def _tag_adapter_cancel_user_steer(adapter: Any, session: Any = None) -> None:
    """Tag the adapter's next mid-invocation cancel with the USER_STEER reason.

    Called just before the executor triggers ``task.cancel()`` on any
    in-flight stage invoke task so the adapter's mid-invocation cancel
    handler picks up the tag and appends an LLM-actionable synthetic
    ``function_response`` (instead of the legacy generic jargon). See
    goldfive#139.

    Routes through :meth:`ADKAdapter.set_next_cancel_reason` when the
    adapter exposes it (PR #294 audit / goldfive#271 follow-up) so
    the tag is keyed by ``session.id`` and cannot bleed across
    concurrent goldfive sessions sharing one adapter. Falls back to
    the bare attribute write for adapters / stubs that predate the
    helper.
    """
    setter = getattr(adapter, "set_next_cancel_reason", None)
    if callable(setter) and session is not None:
        try:
            setter(session, _CANCEL_REASON_USER_STEER)
            return
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "ParallelDAGExecutor: set_next_cancel_reason raised: %s", exc
            )
    try:
        adapter._next_cancel_reason = _CANCEL_REASON_USER_STEER
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "ParallelDAGExecutor: could not tag adapter cancel reason: %s",
            exc,
        )


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
    max_task_invocations:
        Optional safety cap on the number of times the planner may replace
        the plan during a single ``run()``. Protects against refine loops.
        Defaults to ``None`` (unbounded); per-task / per-tool caps are the
        primary guards.

        Note: in the parallel executor this counter increments on plan
        refinement (not per task invocation as in
        :class:`SequentialExecutor`); the parameter is unified for naming
        consistency and backwards compatibility.
    """

    def __init__(
        self,
        max_concurrency: int = 0,
        drift_policy: Literal["cancel_stage", "finish_stage"] = "finish_stage",
        max_task_invocations: int | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        if "max_plan_reinvocations" in legacy_kwargs:
            legacy_value = legacy_kwargs.pop("max_plan_reinvocations")
            warnings.warn(
                "ParallelDAGExecutor(max_plan_reinvocations=...) is deprecated; "
                "use max_task_invocations=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_task_invocations is None:
                max_task_invocations = legacy_value
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"ParallelDAGExecutor got unexpected keyword argument(s): {unexpected}")
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be >= 0")
        if drift_policy not in ("cancel_stage", "finish_stage"):
            raise ValueError(
                f"drift_policy must be 'cancel_stage' or 'finish_stage', got {drift_policy!r}"
            )
        if max_task_invocations is not None and max_task_invocations < 0:
            raise ValueError("max_task_invocations must be >= 0")
        self.max_concurrency = max_concurrency
        self.drift_policy: Literal["cancel_stage", "finish_stage"] = drift_policy
        self.max_task_invocations: int | None = max_task_invocations

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
        # goldfive#247: even the initial pin runs through the
        # channel-processor primitive so the runtime check stays
        # consistent. ``set_session_plan`` warns (or, in strict
        # mode, raises) when called outside
        # :func:`channel_processor_active` — that's the structural
        # enforcement of "single writer onto session.plan".
        with channel_processor_active():
            set_session_plan(session, plan)

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
                    await self._apply_steer(pending_steer, steerer=steerer, session=session)
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
                    stage for stage in stages if any(t.id not in completed_stage_ids for t in stage)
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
                    abort_reason = control_outcome.cancel_reason or "cancelled by control"
                    # goldfive#205: propagate the structured cancel prefix
                    # onto every task this CANCEL interrupted.
                    cancel_prefix = control_outcome.cancel_reason_prefix or (
                        "user_cancel:cancelled_by_control"
                    )
                    for task, _inv, _drift, _err in stage_results:
                        await self._mark_cancelled_if_live(
                            task_id=task.id,
                            steerer=steerer,
                            session=session,
                            cancel_reason=cancel_prefix,
                        )
                    break

                # Fold terminal statuses + results back into the session.
                # If the agent already transitioned the task via reporting
                # tools (status is terminal), leave it alone. Otherwise
                # auto-transition on its behalf so clean returns count as
                # COMPLETED, not silently PENDING/RUNNING.
                # goldfive#247: ``task`` is the pre-mutation snapshot;
                # the framework auto-start via ``steerer.transition(...
                # RUNNING)`` produced a NEW Plan, so the captured ``task``
                # reference still says PENDING. Refresh status from the
                # live ``session.plan`` so terminal-task detection (set
                # by reporting tools through the steerer) is honoured.
                live_plan = session.plan
                live_status_by_id: dict[str, TaskStatus] = (
                    {t.id: t.status for t in live_plan.tasks} if live_plan is not None else {}
                )
                for task, inv, _task_drift, error in stage_results:
                    live_status = live_status_by_id.get(task.id, task.status)
                    already_terminal = live_status in _TERMINAL_TASK_STATUSES
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
                            # goldfive#205: route through the steerer so
                            # an observable ``TaskCancelled`` envelope with
                            # a structured reason reaches sinks. Previously
                            # this mutated task.status silently.
                            await steerer.transition(
                                task.id,
                                TaskStatus.CANCELLED,
                                detail="cancelled by asyncio (stage)",
                                cancel_reason="adk_cancellation:stage_task",
                                session=session,
                            )
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
                            # zicato#12 mechanism 1: full-fidelity actual
                            # output (every turn), recorded as the canonical
                            # gradeable artifact alongside the legacy
                            # last-turn ``completed_results`` write.
                            full_output = inv.full_text or inv.text or ""
                            if full_output:
                                session.completed_outputs[task.id] = full_output
                    completed_stage_ids.add(task.id)

                # If a STEER arrived mid-stage, apply it now (stage was
                # cancelled by _run_stage, so the fold-back above has
                # already recorded the tasks as CANCELLED).
                if control_outcome is not None and control_outcome.steer_message is not None:
                    await self._apply_steer(
                        control_outcome.steer_message,
                        steerer=steerer,
                        session=session,
                    )
                    # New plan installed on session; drop the stage-level
                    # drift (if any) so we don't double-refine.
                    drift = None

                if drift is not None:
                    if (
                        self.max_task_invocations is not None
                        and refinements_used >= self.max_task_invocations
                    ):
                        log.warning(
                            "ParallelDAGExecutor: drift detected but reinvocation "
                            "budget exhausted (%d) — aborting",
                            self.max_task_invocations,
                        )
                        abort_reason = (
                            f"plan reinvocation budget exhausted ({self.max_task_invocations})"
                        )
                        break

                    refined, refine_attempt_id = await self._refine(
                        plan=session.plan or plan,
                        drift=drift,
                        planner=planner,
                        session=session,
                        sinks=sinks,
                        steerer=steerer,
                    )
                    if refined is not None and refined is not (session.plan or plan):
                        refinements_used += 1
                        # Refine succeeded: record the "succeeded" outcome
                        # so a follow-up same-(kind, task) drift on this
                        # turn skips refine (goldfive#215 iter-8 P2). The
                        # outcome dict is the single source of truth — no
                        # separate ``refine_failure_counts`` to pop.
                        session.refine_outcomes[
                            (drift.kind.value, drift.current_task_id or "")
                        ] = RefineOutcome(state="succeeded", fail_count=0)
                        with channel_processor_active():
                            set_session_plan(session, refined)
                        await emit_event(
                            sinks,
                            plan_revised_event(
                                run_id=session.run_id,
                                sequence=session.next_sequence(),
                                plan=refined,
                                drift=drift,
                                session_id=session.id,
                            ),
                        )
                        # Pair the success with its preceding
                        # refine_attempted event so dict-event consumers
                        # can correlate by attempt_id (mirrors the steerer's
                        # _emit_plan_revised_correlation contract). Skipped
                        # when the legacy path (no steerer) was taken —
                        # there's no attempt_id to pair against.
                        if refine_attempt_id:
                            emit_corr = getattr(
                                steerer, "_emit_plan_revised_correlation", None
                            )
                            if callable(emit_corr):
                                try:
                                    await emit_corr(
                                        session,
                                        refined,
                                        drift,
                                        attempt_id=refine_attempt_id,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    log.debug(
                                        "ParallelDAGExecutor: "
                                        "steerer.plans._emit_plan_revised_correlation raised: %s",
                                        exc,
                                    )
                        # Falls through to loop top: stages recomputed.
                        continue

                    # Refine returned None: _refine has already emitted a
                    # CRITICAL follow-up DriftDetected. Bump the per-(kind,
                    # task) counter and abort the run if we've exceeded
                    # the threshold — silently re-entering the same stage
                    # with the same plan is the exact stall goldfive#134
                    # targets.
                    if refined is None:
                        failure_count = self._bump_refine_failure(session=session, drift=drift)
                        if failure_count >= self.REFINE_FAILURE_THRESHOLD:
                            abort_reason = (
                                f"refine failed {failure_count} consecutive "
                                f"times for {drift.kind.value} "
                                f"(task {drift.current_task_id or 'n/a'}); "
                                f"aborting to avoid silent loop"
                            )
                            log.warning("ParallelDAGExecutor: %s", abort_reason)
                            break
                        # Below threshold: fall through to the next stage
                        # and give the planner another chance on the next
                        # drift of the same kind.

                # No drift (or refine below threshold): continue to next stage.
        except BaseException as exc:  # noqa: BLE001 — propagate reason, then re-raise
            abort_reason = f"{type(exc).__name__}: {exc}"
            await emit_event(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=abort_reason,
                    session_id=session.id,
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
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=abort_reason)

        # Goal success-predicate gate (PLAN-LIFECYCLE.md §6.1, third
        # clause). Every stage has completed — now verify the caller's
        # semantic goals. A predicate that returns False or raises
        # fails the run.
        unmet = evaluate_goal_predicates(session)
        if unmet is not None:
            await emit_event(
                sinks,
                run_aborted_event(
                    run_id=session.run_id,
                    sequence=session.next_sequence(),
                    reason=unmet,
                    session_id=session.id,
                ),
            )
            return ExecutionOutcome(success=False, session=session, reason=unmet)

        await emit_event(
            sinks,
            run_completed_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                outcome_summary=_outcome_summary(completed_stage_ids),
                session_id=session.id,
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
                # F10 / goldfive#251 R4: framework auto-start of a task
                # is a real status transition (PENDING -> RUNNING) and
                # deserves a dedicated source label so operators can
                # tell it apart from ``handler_default`` (LLM tool call
                # where ``task_id`` defaulted) and ``other`` (catch-all).
                # Emit BEFORE the adapter invoke so the transition row
                # lands in the wire order operators expect (transition
                # then activity).
                # F10 / goldfive#251 R4: framework auto-start of a task
                # is a real PENDING -> RUNNING transition; emit
                # TaskTransitioned with source="executor_dispatch" so
                # operators distinguish it from LLM reporting-tool
                # (handler_default) and other (catch-all). goldfive#247:
                # the in-place ``task.status = RUNNING`` mutation is
                # gone; we derive a NEW Plan via :func:`with_task_status`
                # and swap, then emit. The captured ``task`` reference
                # used by the surrounding fold-back stays stale (it's a
                # Task snapshot) but the live ``session.plan.tasks[*]``
                # carries the new status — :meth:`_run_stage`'s caller
                # consults the live plan post-fold for terminal
                # detection.
                prev_status = task.status
                if prev_status is not TaskStatus.RUNNING and prev_status not in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.NOT_NEEDED,
                ):
                    if session.plan is not None and any(
                        t.id == task.id for t in session.plan.tasks
                    ):
                        with channel_processor_active():
                            set_session_plan(
                                session,
                                with_task_status(session.plan, task.id, TaskStatus.RUNNING),
                            )
                    emit_transition = getattr(
                        getattr(steerer, "tasks", None), "_emit_task_transitioned", None
                    )
                    if callable(emit_transition):
                        # Refresh the task reference so the emit reads
                        # the new status from the swapped plan.
                        live_task = task
                        if session.plan is not None:
                            live_task = next(
                                (t for t in session.plan.tasks if t.id == task.id),
                                task,
                            )
                        try:
                            await emit_transition(
                                session,
                                live_task,
                                from_status=prev_status,
                                to_status=TaskStatus.RUNNING,
                                source="executor_dispatch",
                            )
                        except Exception as exc:  # noqa: BLE001
                            log.debug(
                                "ParallelDAGExecutor: TaskTransitioned emit raised "
                                "for task=%s: %s",
                                task.id,
                                exc,
                            )
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
                    await steerer.drift.observe(inv, session)
                    drift = steerer.drift.detect_drift(inv, session)
                except asyncio.CancelledError:
                    raise
                except Exception as detect_exc:  # noqa: BLE001
                    # Plumbing failure inside the drift pipeline: surface
                    # it so sinks see a signal, instead of silently
                    # treating the task's output as benign. An INFO
                    # CUSTOM drift mirrors the pattern
                    # ``DefaultSteerer._emit_reflective_failure`` uses
                    # for the reflective-check plumbing: the run
                    # continues but operators can see the failure in
                    # the event stream. See goldfive#134.
                    log.warning(
                        "ParallelDAGExecutor: steerer.drift.observe/detect_drift "
                        "raised for task=%s: %s",
                        task.id,
                        detect_exc,
                    )
                    await _emit_pipeline_failure_drift(
                        session=session,
                        sinks=sinks,
                        task_id=task.id,
                        reason=f"drift_pipeline_failed: {detect_exc}",
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
            asyncio.create_task(run_one(t), name=f"goldfive-task-{t.id}") for t in stage_tasks
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
                recv_task = asyncio.create_task(control.receive(), name="goldfive-stage-control")
            return recv_task

        async def _cancel_stage_tasks() -> None:
            if not pending:
                return
            for p in pending:
                p.cancel()
            cancelled_done, _still = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
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
                                    results[task.id] = (task, None, None, asyncio.CancelledError())
                                except BaseException as exc:  # noqa: BLE001
                                    results[task.id] = (task, None, None, exc)
                            stage_control_outcome = outcome
                            # Tag the adapter's next mid-invocation
                            # cancel with USER_STEER when that's the
                            # control cause, so the synthetic
                            # function_response carries LLM-actionable
                            # content instead of the legacy generic
                            # jargon. See goldfive#139.
                            if outcome.steer_message is not None:
                                _tag_adapter_cancel_user_steer(adapter, session=session)
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
        ordered: list[_StageResult] = [results[t.id] for t in stage_tasks if t.id in results]
        return ordered, first_drift, stage_control_outcome

    # ------------------------------------------------------------------
    # Plan refinement
    # ------------------------------------------------------------------

    # Consecutive refine failures tolerated per (drift_kind, task_id)
    # before the parallel executor gives up and aborts the run. Mirrors
    # ``DefaultSteerer.REFINE_FAILURE_THRESHOLD`` so a stage-level refine
    # that fails validation or raises does not loop silently. See
    # goldfive#134.
    REFINE_FAILURE_THRESHOLD: int = 2

    async def _refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        planner: Planner,
        session: Session,
        sinks: list[EventSink],
        steerer: Steerer | None = None,
    ) -> tuple[Plan | None, str]:
        """Ask ``planner.refine`` for a revision; validate and signal failures.

        Unlike the previous quiet-null version, every failure mode
        (``refine`` raises, returns ``None``, or returns a plan that
        fails structural validation) emits a CRITICAL follow-up
        ``DriftDetected`` event so sinks see "refine failed" instead of
        silently observing the old plan. A per-``(kind, task_id)``
        failure counter is bumped on ``session.refine_outcomes``
        (goldfive#215 P2) and mirrors the steerer's back-off threshold.
        Returns the validated revised plan on success or ``None`` on
        any failure;
        callers should fall through to the next stage when ``None`` is
        returned (downstream tasks that depended on a replacement still
        block via ``_pick_next_task`` and the reachability audit).

        ``steerer`` (optional): when bound and the steerer exposes the
        :meth:`~goldfive.steerer.DefaultSteerer.observe_refine` async
        context manager, refine attempt is wrapped so every refine
        produces a paired ``refine_attempted`` + (``refine_failed`` |
        ``plan_revised``) event regardless of which dispatch path
        triggered it. This unifies emission across the steerer-driven
        and executor-driven refine paths — without it, refines via this
        executor emit no observability and the planner's
        ``refine_orphaned_tasks`` validator no-ops (it depends on the
        steerer's span-context provider which is wired through
        ``_active_session``). See goldfive#263 / #264.

        Returns ``(plan, attempt_id)``. ``plan`` is the validated
        revised plan or ``None`` on any failure. ``attempt_id`` is the
        empty string when no steerer was used or no observation was
        established; otherwise the UUID minted inside ``observe_refine``
        so callers can stamp it onto the success-path ``plan_revised``
        correlation event.

        See goldfive#134.
        """
        # If a steerer with observe_refine is bound, route refine
        # attempt+failure emission through it. Otherwise fall back to
        # the legacy direct call (test stubs / custom steerers).
        observe_refine = (
            getattr(getattr(steerer, "plans", None), "observe_refine", None)
            if steerer is not None
            else None
        )
        attempt_id: str = ""
        if observe_refine is not None and callable(observe_refine):
            cm = observe_refine(session, drift)
            with _state_audit.cancellation_stash_audited(
                "ParallelDAGExecutor._refine.observed"
            ):
                try:
                    async with cm as ctx_attempt_id:
                        attempt_id = ctx_attempt_id
                        refined = await planner.refine(
                            plan=plan, drift=drift, goals=list(session.goals)
                        )
                except Exception as exc:  # noqa: BLE001
                    # observe_refine has already emitted refine_failed; we
                    # ALSO emit the CRITICAL DriftDetected mirror so the
                    # legacy operator-visible signal still lands.
                    log.warning("ParallelDAGExecutor: planner.refine raised: %s", exc)
                    await self._escalate_refine_failure_as_critical_drift(
                        session=session,
                        sinks=sinks,
                        source=drift,
                        reason=f"refine raised: {exc}",
                    )
                    return None, attempt_id
                except BaseException as exc:  # noqa: BLE001
                    # Phase 3.5 (CANCELLATION-CONTRACT.md §C2): ``CancelledError``
                    # bypasses ``except Exception`` (since Py 3.8 it is a
                    # ``BaseException``). The CRITICAL drift mirror is the
                    # operator-visible "refine failed" signal; without this
                    # branch a refine cancelled mid-flight would leave
                    # sinks observing only the original drift, with no
                    # follow-up explaining why the plan didn't change.
                    # Re-raise so cancellation continues to propagate.
                    log.warning(
                        "ParallelDAGExecutor: planner.refine cancelled: %s",
                        type(exc).__name__,
                    )
                    await self._escalate_refine_failure_as_critical_drift(
                        session=session,
                        sinks=sinks,
                        source=drift,
                        reason=f"refine cancelled: {type(exc).__name__}",
                    )
                    # Phase 3.5 tripwire compliance marker: the
                    # ``except BaseException: stash; raise`` form (§1.2)
                    # has run its stash (the refine_failure mirror).
                    _state_audit.mark_stash_completed()
                    raise
            if refined is None:
                log.warning(
                    "ParallelDAGExecutor: planner.refine(kind=%s) returned None; plan unchanged",
                    drift.kind.value,
                )
                # Emit refine_failed via the steerer for parity with the
                # exception path above. observe_refine has already cleared
                # _active_session by now, so we go through the steerer's
                # direct emitter.
                await self._steerer_emit_refine_failed(
                    steerer=steerer,
                    session=session,
                    drift=drift,
                    attempt_id=attempt_id,
                    failure_kind="parse_error",
                    reason="planner returned no revised plan",
                    detail="",
                )
                await self._escalate_refine_failure_as_critical_drift(
                    session=session,
                    sinks=sinks,
                    source=drift,
                    reason="planner returned no revised plan",
                )
                return None, attempt_id
            try:
                refined.validate(for_revision=True, prior=plan)
            except ValueError as exc:
                log.warning(
                    "ParallelDAGExecutor: revised plan failed validation (%s); keeping prior plan",
                    exc,
                )
                await self._steerer_emit_refine_failed(
                    steerer=steerer,
                    session=session,
                    drift=drift,
                    attempt_id=attempt_id,
                    failure_kind="validator_rejected",
                    reason=f"plan validation failed: {exc}",
                    detail=type(exc).__name__,
                )
                await self._escalate_refine_failure_as_critical_drift(
                    session=session,
                    sinks=sinks,
                    source=drift,
                    reason=f"plan validation failed: {exc}",
                )
                return None, attempt_id
        else:
            # Legacy path — no steerer or no observe_refine. No
            # refine_attempted/refine_failed emission, but the
            # CRITICAL DriftDetected mirror is preserved.
            with _state_audit.cancellation_stash_audited(
                "ParallelDAGExecutor._refine.legacy"
            ):
                try:
                    refined = await planner.refine(
                        plan=plan, drift=drift, goals=list(session.goals)
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("ParallelDAGExecutor: planner.refine raised: %s", exc)
                    await self._escalate_refine_failure_as_critical_drift(
                        session=session,
                        sinks=sinks,
                        source=drift,
                        reason=f"refine raised: {exc}",
                    )
                    return None, attempt_id
                except BaseException as exc:  # noqa: BLE001
                    # Phase 3.5 (CANCELLATION-CONTRACT.md §C2): ``CancelledError``
                    # bypasses ``except Exception``; emit the CRITICAL drift
                    # mirror so the operator-visible "refine cancelled" signal
                    # still lands, then re-raise to preserve cancellation
                    # propagation per the asyncio contract.
                    log.warning(
                        "ParallelDAGExecutor: planner.refine cancelled (legacy path): %s",
                        type(exc).__name__,
                    )
                    await self._escalate_refine_failure_as_critical_drift(
                        session=session,
                        sinks=sinks,
                        source=drift,
                        reason=f"refine cancelled: {type(exc).__name__}",
                    )
                    # Phase 3.5 tripwire compliance marker.
                    _state_audit.mark_stash_completed()
                    raise
            if refined is None:
                log.warning(
                    "ParallelDAGExecutor: planner.refine(kind=%s) returned None; plan unchanged",
                    drift.kind.value,
                )
                await self._escalate_refine_failure_as_critical_drift(
                    session=session,
                    sinks=sinks,
                    source=drift,
                    reason="planner returned no revised plan",
                )
                return None, attempt_id
            try:
                refined.validate(for_revision=True, prior=plan)
            except ValueError as exc:
                log.warning(
                    "ParallelDAGExecutor: revised plan failed validation (%s); keeping prior plan",
                    exc,
                )
                await self._escalate_refine_failure_as_critical_drift(
                    session=session,
                    sinks=sinks,
                    source=drift,
                    reason=f"plan validation failed: {exc}",
                )
                return None, attempt_id
        # goldfive#199: stamp the trigger_event_id on the plan for every
        # refine so harmonograf can strict-id-merge plan-revision rows
        # regardless of whether the executor refined via the steerer or
        # inline here. Resolution mirrors steerer.plans._apply_revision: source
        # annotation_id (user-control) → drift.id (autonomous). Preserves
        # any pre-existing stamp from the planner path.
        # goldfive#247: Plan is frozen — derive a new instance via
        # :func:`bump_revision` rather than mutating in place.
        if not refined.revision_trigger_event_id:
            from goldfive.events import _trigger_id_from_drift

            trig_id = _trigger_id_from_drift(drift)
            if trig_id:
                refined = bump_revision(
                    refined,
                    revision_index=refined.revision_index,
                    revision_trigger_event_id=trig_id,
                )
        return refined, attempt_id

    @staticmethod
    async def _steerer_emit_refine_failed(
        *,
        steerer: Steerer | None,
        session: Session,
        drift: DriftEvent,
        attempt_id: str,
        failure_kind: str,
        reason: str,
        detail: str,
    ) -> None:
        """Best-effort delegate to ``DefaultSteerer._emit_refine_failed``.

        The parallel executor's refine path emits ``refine_failed``
        events via the bound steerer when one is available so observers
        can pair attempted/failed/plan-revised by ``attempt_id``. Custom
        steerers without the ``_emit_refine_failed`` method are tolerated
        — the call is duck-typed and silently no-ops, which preserves
        backwards compatibility for tests that pass a stub Steerer.

        Failures inside the steerer's emit are logged and swallowed:
        observability must never break the run.
        """
        if steerer is None:
            return
        emit_failed = getattr(steerer, "_emit_refine_failed", None)
        if not callable(emit_failed):
            return
        try:
            await emit_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind=failure_kind,
                reason=reason,
                detail=detail,
            )
        except Exception as exc:  # noqa: BLE001 — observability must never break the run
            log.debug(
                "ParallelDAGExecutor: steerer.plans._emit_refine_failed raised: %s",
                exc,
            )

    async def _escalate_refine_failure_as_critical_drift(
        self,
        *,
        session: Session,
        sinks: list[EventSink],
        source: DriftEvent,
        reason: str,
    ) -> None:
        """Surface a failed refine as a CRITICAL follow-up ``DriftDetected``.

        Mirrors ``DefaultSteerer._escalate_refine_failure_as_critical_drift`` so the parallel
        executor's direct refine path produces the same sink-level
        signal the sequential path does. Without this, a planner.refine
        that raises or returns a garbage plan leaves the session pinned
        to the stale plan and the next stage re-enters the same state.
        See goldfive#134.
        """
        failure = DriftEvent(
            kind=source.kind,
            severity=DriftSeverity.CRITICAL,
            detail=f"refine failed ({source.kind.value}): {reason}",
            current_task_id=source.current_task_id,
            current_agent_id=source.current_agent_id,
        )
        await _emit_drift_event(session=session, sinks=sinks, drift=failure)

    def _bump_refine_failure(self, *, session: Session, drift: DriftEvent) -> int:
        """Bump the per-``(kind, task_id)`` refine-failure counter.

        Returns the new count. Callers compare against
        :attr:`REFINE_FAILURE_THRESHOLD` to decide whether to abort the
        run. Uses ``session.refine_outcomes`` so the counter is shared
        with :class:`~goldfive.steerer.DefaultSteerer` (goldfive#215
        iter-8 P2 — outcome dict replaces the deleted
        ``refine_failure_counts`` int counter).
        """
        key = (drift.kind.value, drift.current_task_id or "")
        prior = session.refine_outcomes.get(key)
        new_count = (
            prior.fail_count + 1 if prior is not None and prior.state == "failed" else 1
        )
        session.refine_outcomes[key] = RefineOutcome(state="failed", fail_count=new_count)
        return new_count

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
        arrives. The steerer's intervention-ladder Level 4 pause
        (goldfive#142) triggers the same blocking wait by dispatching
        a ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage (Phase 2 of the
        path-duality fix) — that message sets ``request_pause=True``
        so the executor's pause loop is indistinguishable from an
        explicit user-initiated PAUSE.
        """
        if control is None:
            # Without a control channel the ladder-initiated pause has
            # nowhere to wait. The steerer's
            # ``GOLDFIVE_PAUSE_ESCALATE`` dispatch is best-effort; when
            # there is no channel attached the dispatch is dropped at
            # the source. The originating
            # ``HUMAN_INTERVENTION_REQUIRED`` drift on the sink stream
            # remains the durable signal so observers can react.
            return False, None

        outcomes = await drain_controls(control, session=session, steerer=steerer, sinks=sinks)

        cancel_reason = ""
        cancel_run = False
        steer_msg: object | None = None
        paused = False
        cancel_prefix = ""
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
            # goldfive#205: stash structured cancel prefix for downstream
            # per-task cancel emits (``user_cancel:<annotation_id>``).
            if cancel_prefix:
                session._last_cancel_reason_prefix = cancel_prefix
            raise _ControlCancelled(cancel_reason or "cancelled by control")

        while paused:
            msg = await control.receive()
            if msg is None:
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

    @staticmethod
    async def _mark_cancelled_if_live(
        *,
        task_id: str,
        steerer: Steerer,
        session: Session,
        cancel_reason: str = "",
    ) -> None:
        """Transition a not-yet-terminal task to CANCELLED.

        ``cancel_reason`` (goldfive#205): structured reason stamped on
        the emitted ``TaskCancelled``. Defaults to a generic
        ``user_cancel:cancelled_by_control`` when unspecified.
        """
        if session.plan is None:
            return
        for t in session.plan.tasks:
            if t.id == task_id:
                if t.status in _TERMINAL_TASK_STATUSES:
                    return
                reason_value = cancel_reason or "user_cancel:cancelled_by_control"
                try:
                    await steerer.transition(
                        task_id,
                        TaskStatus.CANCELLED,
                        detail="cancelled by control",
                        cancel_reason=reason_value,
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
            await steerer.drift.observe(message, session)
        except Exception as exc:  # noqa: BLE001
            log.warning("ParallelDAGExecutor: steerer.drift.observe(STEER) raised: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome_summary(completed_task_ids: set[str]) -> str:
    return f"{len(completed_task_ids)} tasks completed"


async def _emit_pipeline_failure_drift(
    *,
    session: Session,
    sinks: list[EventSink],
    task_id: str,
    reason: str,
) -> None:
    """Emit an INFO ``CUSTOM`` drift when the drift pipeline itself raised.

    Surfaces plumbing failures in ``steerer.drift.observe`` / ``detect_drift``
    that would otherwise be swallowed. INFO severity so this is
    record-only and does not trigger another refine. Sinks that care
    can filter on the ``drift_pipeline_failed:`` detail prefix. See
    goldfive#134.
    """
    drift = DriftEvent(
        kind=DriftKind.CUSTOM,
        severity=DriftSeverity.INFO,
        detail=reason,
        current_task_id=task_id or "",
    )
    await _emit_drift_event(session=session, sinks=sinks, drift=drift)


async def _emit_drift_event(
    *,
    session: Session,
    sinks: list[EventSink],
    drift: DriftEvent,
) -> None:
    """Build a DriftDetected envelope with proto enum mapping, then emit.

    Uses the same enum-mapping shape
    :meth:`DefaultSteerer._emit_drift_detected` uses — the
    :func:`drift_detected_event` helper in :mod:`goldfive.events` does
    a best-effort name lookup that silently fails for StrEnum-style
    kind/severity names (stored as lowercase like ``critical``), which
    would leave the event with enum value ``0`` (UNSPECIFIED).
    """
    from goldfive.events import new_event
    from goldfive.pb.goldfive.v1 import types_pb2

    evt = new_event(session.run_id, session.next_sequence(), session_id=session.id)
    evt.drift_detected.kind = getattr(
        types_pb2,
        f"DRIFT_KIND_{drift.kind.name}",
        getattr(types_pb2, "DRIFT_KIND_CUSTOM", 0),
    )
    evt.drift_detected.severity = getattr(
        types_pb2,
        f"DRIFT_SEVERITY_{drift.severity.name}",
        getattr(types_pb2, "DRIFT_SEVERITY_UNSPECIFIED", 0),
    )
    evt.drift_detected.detail = drift.detail
    evt.drift_detected.current_task_id = drift.current_task_id or ""
    evt.drift_detected.current_agent_id = drift.current_agent_id or ""
    try:
        await emit_event(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("_emit_drift_event: sink emit raised: %s", exc)


# Runtime conformance check — the class must satisfy the Executor protocol.
assert isinstance(ParallelDAGExecutor(), Executor)
