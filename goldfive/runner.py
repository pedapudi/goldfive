"""The :class:`Runner` — goldfive's single public entrypoint.

A Runner composes the six pluggable components of a goldfive run:

* :class:`GoalDeriver` — turns ``user_input`` into ``list[Goal]``.
* :class:`Planner` — turns goals into a :class:`Plan`.
* :class:`Executor` — walks the plan, dispatches to the adapter.
* :class:`AgentAdapter` — talks to the underlying agent framework.
* :class:`Steerer` — runs the state machine, detects drift.
* :class:`EventSink` — persists / observes the event stream.

``Runner.run`` emits a ``RunStarted`` event, derives goals, generates a
plan, registers the seven canonical reporting tools on the adapter,
binds the steerer to the sinks+planner, and hands everything to the
executor. The returned :class:`ExecutionOutcome` carries the final live
:class:`Session` so callers can inspect completed tasks / artifacts.

Event lifecycle ownership
-------------------------
* The Runner owns ``Run*`` lifecycle events (``RunStarted``,
  ``GoalDerived``, ``PlanSubmitted``, and pre-executor ``RunAborted``).
* Executors own ``Task*`` events, ``PlanRevised``, and the terminal
  ``RunCompleted`` / ``RunAborted`` they emit when their own state
  machine reaches the end of the run.
* The Steerer owns ``DriftDetected`` and the per-task ``mark_task_*``
  emissions.

All sink emissions are proto :class:`Event` envelopes — built via the
typed factories in :mod:`goldfive.events`.

No ADK or Claude Agent SDK imports live in this module. Optional
adapter implementations live under ``goldfive.adapters.<framework>``
and are loaded lazily by callers.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from goldfive.events import (
    emit,
    goal_derived_event,
    plan_submitted_event,
    run_aborted_event,
    run_started_event,
)
from goldfive.goal_deriver import PassthroughGoalDeriver
from goldfive.reporting import BUILTIN_REPORTING_TOOLS
from goldfive.results import ExecutionOutcome
from goldfive.steerer import DefaultSteerer
from goldfive.types import Goal, Session

if TYPE_CHECKING:
    from goldfive.protocols import (
        AgentAdapter,
        EventSink,
        Executor,
        GoalDeriver,
        Planner,
        Steerer,
    )

log = logging.getLogger("goldfive.runner")


def _now_ms() -> int:
    return int(time.time() * 1000)


class Runner:
    """The public entrypoint for a goldfive run.

    Parameters
    ----------
    agent:
        An :class:`AgentAdapter` wrapping the underlying agent framework.
    planner:
        A :class:`Planner` instance. Pass a planner configured with a
        pre-baked plan when you already know the tasks (see
        :class:`PassthroughGoalDeriver` for an analogous convenience on
        the goals side).
    executor:
        An :class:`Executor` (e.g. :class:`SequentialExecutor` or
        :class:`ParallelDAGExecutor`).
    goal_deriver:
        Optional — defaults to ``PassthroughGoalDeriver("run")``. When
        the caller passes ``user_input`` as ``list[Goal]`` the deriver
        is bypassed entirely.
    steerer:
        Optional — defaults to :class:`DefaultSteerer`.
    sinks:
        Optional list of :class:`EventSink` instances. Defaults to ``[]``.
    max_plan_reinvocations:
        Safety cap on plan refinement cycles. Passed through to the
        executor (via ``context``) so executors that honour it can
        enforce the cap.
    """

    def __init__(
        self,
        *,
        agent: AgentAdapter,
        planner: Planner,
        executor: Executor,
        goal_deriver: GoalDeriver | None = None,
        steerer: Steerer | None = None,
        sinks: list[EventSink] | None = None,
        max_plan_reinvocations: int = 3,
    ) -> None:
        self.agent = agent
        self.planner = planner
        self.executor = executor
        self.goal_deriver: GoalDeriver = goal_deriver or PassthroughGoalDeriver("run")
        self.steerer: Steerer = steerer or DefaultSteerer()
        self.sinks: list[EventSink] = list(sinks) if sinks else []
        self.max_plan_reinvocations = max_plan_reinvocations

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str | list[Goal],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ExecutionOutcome:
        """Execute one end-to-end goldfive run and return the outcome."""

        # 1. Build Session with a fresh run_id.
        session = Session(
            run_id=uuid.uuid4().hex,
            started_at_ms=_now_ms(),
        )

        # 2. Emit RunStarted before anything else.
        await self._emit_run_started(session, user_input)

        # 3. Derive (or accept) goals.
        try:
            goals = await self._resolve_goals(user_input, context)
        except Exception as exc:  # noqa: BLE001
            reason = f"goal derivation failed: {exc}"
            log.exception("goal derivation failed")
            await self._emit_run_aborted(session, reason)
            return ExecutionOutcome(success=False, session=session, reason=reason)
        session.goals = list(goals)

        await self._emit_goal_derived(session)

        # Build the context passed to the planner — stamp run_id so LLM
        # planners can include it in the plan envelope.
        planner_context: dict[str, Any] = dict(context) if context else {}
        planner_context.setdefault("run_id", session.run_id)
        planner_context.setdefault("max_plan_reinvocations", self.max_plan_reinvocations)

        # 4. Generate the plan.
        try:
            plan = await self.planner.generate(
                goals=session.goals,
                available_agents=list(self.agent.available_agents),
                context=planner_context,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"planner.generate raised: {exc}"
            log.exception("planner.generate raised")
            await self._emit_run_aborted(session, reason)
            return ExecutionOutcome(success=False, session=session, reason=reason)

        if plan is None:
            reason = "no plan generated"
            await self._emit_run_aborted(session, reason)
            return ExecutionOutcome(success=False, session=session, reason=reason)

        # Planners may leave run_id blank; stamp ours so downstream sinks
        # correlate cleanly.
        if not plan.run_id:
            plan.run_id = session.run_id
        session.plan = plan

        await self._emit_plan_submitted(session, plan)

        # 5. Register the seven canonical reporting tools on the adapter.
        try:
            await self.agent.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))
        except Exception as exc:  # noqa: BLE001
            reason = f"register_reporting_tools raised: {exc}"
            log.exception("register_reporting_tools raised")
            await self._emit_run_aborted(session, reason)
            return ExecutionOutcome(success=False, session=session, reason=reason)

        # 6. Bind the steerer. (The executor may re-bind — that's fine.)
        try:
            self.steerer.bind(sinks=list(self.sinks), planner=self.planner)
        except Exception as exc:  # noqa: BLE001
            reason = f"steerer.bind raised: {exc}"
            log.exception("steerer.bind raised")
            await self._emit_run_aborted(session, reason)
            return ExecutionOutcome(success=False, session=session, reason=reason)

        # 7. Hand off to the executor.
        try:
            outcome = await self.executor.run(
                plan=session.plan,
                session=session,
                adapter=self.agent,
                steerer=self.steerer,
                planner=self.planner,
                sinks=list(self.sinks),
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"executor.run raised: {exc}"
            log.exception("executor.run raised")
            await self._emit_run_aborted(session, reason)
            return ExecutionOutcome(success=False, session=session, reason=reason)

        return outcome

    # ------------------------------------------------------------------
    # resume — best-effort replay
    # ------------------------------------------------------------------

    async def resume(self, persistence_path: str) -> ExecutionOutcome:
        """Replay a JSONL persistence log and return the recovered outcome.

        Best-effort for v0.1: uses ``goldfive.sinks.reconstruct_session``
        when the proto stubs are available (JSONL events are proto-
        encoded). Returns the reconstructed session as an
        :class:`ExecutionOutcome` reflecting the latest terminal marker
        (``RunCompleted`` / ``RunAborted``) seen in the log.

        We do **not** continue execution from the latest cursor — full
        resume semantics require planner/executor co-operation that is
        out-of-scope for this PR. Callers who need a live continuation
        should construct a new :class:`Runner` with the goals recovered
        from the log.

        TODO(#15): once executors grow a ``resume_from`` hook, continue
        execution from the latest un-finished task rather than returning
        the recovered session as-is.
        """
        try:
            from goldfive.sinks import reconstruct_session, replay_from_jsonl
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "goldfive.sinks.reconstruct_session is not available; "
                "install the `proto` extra to enable JSONL replay."
            ) from exc

        events = replay_from_jsonl(persistence_path)
        session = reconstruct_session(events)

        success = False
        reason = "run did not complete before persistence ended"
        for evt in events:
            payload = getattr(evt, "WhichOneof", lambda _: None)("payload")
            if payload == "run_completed":
                success = True
                reason = ""
            elif payload == "run_aborted":
                success = False
                reason = getattr(evt.run_aborted, "reason", "")

        return ExecutionOutcome(success=success, session=session, reason=reason)

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close every sink. Best-effort — individual errors are logged."""
        for sink in self.sinks:
            try:
                await sink.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("sink.close() raised: %s", exc)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _resolve_goals(
        self,
        user_input: str | list[Goal],
        context: Mapping[str, Any] | None,
    ) -> list[Goal]:
        if isinstance(user_input, list):
            if not user_input:
                raise ValueError("Runner.run: empty goal list")
            if not all(isinstance(g, Goal) for g in user_input):
                raise TypeError("Runner.run: list input must be list[Goal]")
            return list(user_input)
        if not isinstance(user_input, str):
            raise TypeError(
                f"Runner.run: user_input must be str or list[Goal], "
                f"got {type(user_input).__name__}"
            )
        goals = await self.goal_deriver.derive(user_input, context=context)
        if not goals:
            raise ValueError("GoalDeriver returned an empty goals list")
        return list(goals)

    async def _emit_run_started(
        self, session: Session, user_input: str | list[Goal]
    ) -> None:
        evt = run_started_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            goal_summary=_initial_goal_summary(user_input),
        )
        await emit(self.sinks, evt)

    async def _emit_goal_derived(self, session: Session) -> None:
        evt = goal_derived_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            goals=list(session.goals),
        )
        await emit(self.sinks, evt)

    async def _emit_plan_submitted(self, session: Session, plan: Any) -> None:
        evt = plan_submitted_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            plan=plan,
        )
        await emit(self.sinks, evt)

    async def _emit_run_aborted(self, session: Session, reason: str) -> None:
        evt = run_aborted_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            reason=reason,
        )
        await emit(self.sinks, evt)


def _initial_goal_summary(user_input: str | list[Goal]) -> str:
    """Best-effort one-liner for the RunStarted event before goals derive."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list) and user_input:
        first = user_input[0]
        return getattr(first, "summary", "") or ""
    return ""


__all__ = ["Runner"]
