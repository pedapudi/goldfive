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
* Emit :class:`RunStarted` once, then iterate tasks.
* Honor the plan's topological order: only invoke tasks whose predecessors
  are ``COMPLETED``. Walk tasks one-at-a-time (no parallelism).
* After each invocation, re-read plan state: tasks may have moved to
  ``COMPLETED`` / ``FAILED`` / ``BLOCKED`` via reporting tools; the steerer
  may also have mutated the plan in response to drift (``PlanRevised``).
* Budget the total number of adapter invocations with ``max_plan_reinvocations``
  so a stuck agent cannot spin forever.
* Terminate with :class:`RunCompleted` on success or :class:`RunAborted`
  when ``fail_fast`` is set and a task fails fatally.
"""

from __future__ import annotations

import logging

from goldfive.events import (
    emit,
    plan_revised_event,
    run_aborted_event,
    run_completed_event,
    run_started_event,
)
from goldfive.protocols import AgentAdapter, EventSink, Executor, Planner, Steerer
from goldfive.results import ExecutionOutcome
from goldfive.types import Plan, Session, Task, TaskStatus

log = logging.getLogger(__name__)


class SequentialExecutor(Executor):
    """Single-threaded executor that drives one task at a time.

    Parameters
    ----------
    max_plan_reinvocations:
        Upper bound on the number of adapter ``invoke()`` calls a single
        :meth:`run` may issue. Each eligible task consumes one invocation;
        when drift triggers a plan revision, the new plan's remaining tasks
        share the same budget. Defaults to ``3`` to mirror the harmonograf
        re-invocation cap.
    fail_fast:
        When ``True`` (default), the first task that ends up ``FAILED`` causes
        the executor to emit ``RunAborted`` and stop. When ``False``, failed
        tasks are recorded and the executor continues walking remaining
        eligible tasks.
    """

    def __init__(
        self,
        *,
        max_plan_reinvocations: int = 3,
        fail_fast: bool = True,
    ) -> None:
        self.max_plan_reinvocations = int(max_plan_reinvocations)
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

        # Emit RunStarted.
        await emit(
            sinks,
            run_started_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                goal_summary=_goal_summary(session, plan),
            ),
        )

        invocations = 0
        failure_reason = ""
        run_failed = False

        # Track the plan identity so we detect mid-run revisions by the steerer.
        last_plan_id = plan.id
        last_revision_index = plan.revision_index

        # Cap by max_plan_reinvocations: this is the number of adapter
        # invocations we'll allow for the whole run. Each eligible task
        # burns one; mid-run plan revisions do not refund any.
        while invocations < self.max_plan_reinvocations:
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

            session.current_task_id = task.id
            invocations += 1

            log.debug(
                "SequentialExecutor: invoking task=%s (invocation %d/%d)",
                task.id, invocations, self.max_plan_reinvocations,
            )

            # Adapter.invoke drives the agent, which calls reporting tools,
            # whose handlers route through the steerer to mutate task state
            # (PENDING -> RUNNING -> COMPLETED/FAILED/BLOCKED) and emit the
            # corresponding proto events via sinks.
            try:
                result = await adapter.invoke(task, session)
            except Exception as exc:  # noqa: BLE001
                # Hard adapter failure: nothing to do except abort.
                log.exception(
                    "SequentialExecutor: adapter.invoke raised for task=%s",
                    task.id,
                )
                failure_reason = f"adapter.invoke raised for task={task.id}: {exc}"
                run_failed = True
                break

            # If the adapter reported an error on the invocation envelope
            # (but didn't raise), surface it as a failure when fail_fast is
            # set. A well-behaved adapter will usually have already called
            # report_task_failed through the agent.
            if result is not None and getattr(result, "error", None) is not None:
                log.warning(
                    "SequentialExecutor: InvocationResult.error=%s task=%s",
                    result.error, task.id,
                )

            # Re-read the task's tracked status after the invocation.
            tracked = _find_task(session.plan or current_plan, task.id)
            tracked_status = (
                tracked.status if tracked is not None else TaskStatus.PENDING
            )
            if tracked_status == TaskStatus.PENDING:
                # The agent did not report a terminal state for this task.
                # Treat as a soft failure: mark it failed locally so
                # _pick_next_task does not pick it again next iteration.
                # The steerer is the authority on status; if it cares, it
                # can transition this itself on a follow-up observe().
                log.info(
                    "SequentialExecutor: task=%s returned PENDING post-invoke; "
                    "marking FAILED locally to avoid repick",
                    task.id,
                )
                if tracked is not None:
                    tracked.status = TaskStatus.FAILED
                tracked_status = TaskStatus.FAILED

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

        await emit(
            sinks,
            run_completed_event(
                run_id=session.run_id,
                sequence=session.next_sequence(),
                outcome_summary=_outcome_summary(session),
            ),
        )
        return ExecutionOutcome(success=True, session=session)


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


def _any_failed(plan: Plan) -> bool:
    return any(t.status == TaskStatus.FAILED for t in plan.tasks)


def _goal_summary(session: Session, plan: Plan) -> str:
    """Best-effort one-line summary for the RunStarted event."""
    if session.goals:
        first = session.goals[0]
        summary = getattr(first, "summary", "") or ""
        if summary:
            return summary
    return plan.summary or f"plan {plan.id}"


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
