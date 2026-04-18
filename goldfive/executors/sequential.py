"""Sequential executor — walks the plan one task at a time.

Simplest correct :class:`Executor` implementation: for every task in
topological order, invoke the adapter and advance the status. Useful
for deterministic tests, adapters that aren't re-entrant, and CLIs that
prefer linear output.

The executor intentionally does NOT emit a ``RunStarted`` event — the
runner is responsible for that so the lifecycle envelope stays under
the runner's control. The executor emits ``RunCompleted`` (on success)
or ``RunAborted`` (on failure / stop-on-failure) as the terminal
envelope.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from goldfive.events import emit, make_event
from goldfive.results import ExecutionOutcome
from goldfive.types import Plan, Session, TaskStatus

if TYPE_CHECKING:
    from goldfive.protocols import AgentAdapter, EventSink, Planner, Steerer

log = logging.getLogger("goldfive.executors.sequential")


class SequentialExecutor:
    """Run a plan one task at a time in topological order."""

    def __init__(self, *, stop_on_failure: bool = True) -> None:
        self.stop_on_failure = stop_on_failure

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
        # Bind sinks + planner on the steerer so transition() events flow.
        try:
            steerer.bind(sinks=sinks, planner=planner)
        except Exception as exc:  # noqa: BLE001
            log.debug("steerer.bind raised: %s", exc)

        session.plan = plan
        stages = plan.topological_stages()
        try:
            for stage in stages:
                for task in stage:
                    await steerer.transition(
                        task.id, TaskStatus.RUNNING, detail="", session=session
                    )
                    try:
                        result = await adapter.invoke(task, session)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("adapter.invoke raised for task %s", task.id)
                        await steerer.transition(
                            task.id,
                            TaskStatus.FAILED,
                            detail=f"adapter.invoke raised: {exc}",
                            session=session,
                        )
                        if self.stop_on_failure:
                            reason = f"task {task.id} failed: {exc}"
                            await self._emit_aborted(session, sinks, reason)
                            return ExecutionOutcome(
                                success=False, session=session, reason=reason
                            )
                        continue

                    # Let the steerer fan the raw result through its drift
                    # detection and progress-forwarding machinery.
                    await steerer.observe(result, session)

                    cur = _current_status(session, task.id)
                    if cur == TaskStatus.RUNNING:
                        # Agent did not transition via a reporting tool —
                        # auto-complete so the plan advances.
                        summary = result.text or ""
                        if summary:
                            session.completed_results[task.id] = summary
                        await steerer.transition(
                            task.id,
                            TaskStatus.COMPLETED,
                            detail=summary,
                            session=session,
                        )
                    elif cur == TaskStatus.FAILED and self.stop_on_failure:
                        reason = f"task {task.id} reported FAILED"
                        await self._emit_aborted(session, sinks, reason)
                        return ExecutionOutcome(
                            success=False, session=session, reason=reason
                        )
        except Exception as exc:  # noqa: BLE001
            log.exception("SequentialExecutor.run raised")
            reason = f"executor raised: {exc}"
            await self._emit_aborted(session, sinks, reason)
            return ExecutionOutcome(success=False, session=session, reason=reason)

        evt = make_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            kind="RunCompleted",
            payload={"plan_id": plan.id},
        )
        await emit(sinks, evt)
        return ExecutionOutcome(success=True, session=session, reason="")

    @staticmethod
    async def _emit_aborted(
        session: Session, sinks: list[EventSink], reason: str
    ) -> None:
        evt = make_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            kind="RunAborted",
            payload={"reason": reason},
        )
        await emit(sinks, evt)


def _current_status(session: Session, task_id: str) -> Any:
    if session.plan is None:
        return None
    for t in session.plan.tasks:
        if t.id == task_id:
            return t.status
    return None


__all__ = ["SequentialExecutor"]
