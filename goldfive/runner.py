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

import inspect
import logging
import warnings
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive import orchestration_state as _ostate
from goldfive._llm import maybe_close_call_llm
from goldfive.conversation import Conversation
from goldfive.events import (
    conversation_ended_event,
    conversation_started_event,
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
    from goldfive.control import ControlChannel
    from goldfive.protocols import (
        AgentAdapter,
        EventSink,
        Executor,
        GoalDeriver,
        Planner,
        Steerer,
    )

log = logging.getLogger("goldfive.runner")


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
    control:
        Optional :class:`~goldfive.control.ControlChannel` for live
        pause / resume / cancel / steer / rewind from an external
        controller (harmonograf UI, CLI, tests). When provided, the
        Runner forwards it into the executor, which polls the channel
        between tasks and races against adapter invocations mid-task.
    max_task_invocations:
        Optional safety cap on adapter invocations per run. Stamped onto
        the planner context so executors that honour it can enforce the
        cap. Defaults to ``None`` (unbounded); per-task / per-tool caps
        are the primary guards against runaway loops.
    goal_drift_enabled:
        Opt-in switch for the trajectory-level GOAL_DRIFT periodic
        check (goldfive#143). ``True`` (default) leaves the steerer's
        own ``goal_drift_call_llm`` wiring intact -- operators who
        pass a steerer configured with a judge callable get the
        check. ``False`` forcibly disables it by detaching
        ``_goal_drift_call_llm`` on the steerer, which is the shape
        unit tests driving mock runners want so they never see
        spurious GOAL_DRIFT firings from the bookkeeping path.
        Has no effect when the steerer was never configured with a
        ``goal_drift_call_llm`` (the feature is already inert).
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
        control: ControlChannel | None = None,
        max_task_invocations: int | None = None,
        conversation: Conversation | None = None,
        goal_drift_enabled: bool = True,
        **legacy_kwargs: Any,
    ) -> None:
        if "max_plan_reinvocations" in legacy_kwargs:
            legacy_value = legacy_kwargs.pop("max_plan_reinvocations")
            warnings.warn(
                "Runner(max_plan_reinvocations=...) is deprecated; use "
                "max_task_invocations=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_task_invocations is None:
                max_task_invocations = legacy_value
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Runner got unexpected keyword argument(s): {unexpected}")
        self.agent = agent
        self.planner = planner
        self.executor = executor
        self.goal_deriver: GoalDeriver = goal_deriver or PassthroughGoalDeriver("run")
        self.steerer: Steerer = steerer or DefaultSteerer()
        # goldfive#143: opt-in gate for the trajectory-level GOAL_DRIFT
        # periodic check. ``True`` (default) is a no-op -- the steerer's
        # own ``goal_drift_call_llm`` wiring governs whether the check
        # fires. ``False`` forcibly detaches the callable so mock-only
        # runs never see GOAL_DRIFT firings, even if a test accidentally
        # wires a callable through. Guarded on ``hasattr`` so custom
        # ``Steerer`` implementations that predate this attribute still
        # construct cleanly.
        self.goal_drift_enabled: bool = goal_drift_enabled
        if not goal_drift_enabled and hasattr(self.steerer, "_goal_drift_call_llm"):
            self.steerer._goal_drift_call_llm = None
        self.sinks: list[EventSink] = list(sinks) if sinks else []
        self._control: ControlChannel | None = control
        self._close_hooks: list[Callable[[], Awaitable[None]]] = []
        self._closed: bool = False
        self.max_task_invocations: int | None = max_task_invocations
        self._conversation: Conversation = conversation or Conversation.new()
        # Tracks whether we've emitted the ConversationStarted event for
        # the current Conversation. Flips back to False on new_conversation().
        self._conversation_announced: bool = False
        # Last turn's Session, held past the turn so that ConversationEnded
        # can piggy-back on its ``next_sequence()`` counter (sequence must
        # be monotonic within a run_id, so the terminal marker needs the
        # session that produced the run's other events).
        self._last_session: Session | None = None

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str | list[Goal],
        *,
        context: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ExecutionOutcome:
        """Execute one end-to-end goldfive run and return the outcome.

        ``session_id`` optionally overrides the ``Session.run_id`` /
        ``Session.id`` that :class:`Conversation.next_turn_session`
        would otherwise mint. Used by :class:`GoldfiveADKAgent` to
        adopt the outer adk-web ``InvocationContext.session.id`` so
        every goldfive Event emitted through sinks stamps the same
        session id that harmonograf spans carry (goldfive#161). Empty
        / ``None`` preserves the legacy uuid4 mint so bare programmatic
        Runner callers see no behaviour change.
        """

        # 1. Build Session seeded by the Conversation. The Session's
        #    run_id is fresh for this turn; conversation_id is stable
        #    across turns; goals / completed_results are pre-populated
        #    with prior-turn state.
        session = self._conversation.next_turn_session()
        # Outer-session pin (goldfive#161): when the caller supplies a
        # non-empty ``session_id`` (typically ``ctx.session.id`` from
        # adk-web), override the freshly-minted ``run_id`` so every
        # Event emitted this turn carries that id. Sinks stamp
        # ``Event.session_id`` from ``Session.id`` (= ``run_id``), so
        # this aligns goldfive events with the ADK session that
        # harmonograf spans already target — resolving the
        # "plan view has empty Gantt" regression from the overlay
        # architecture.
        if session_id:
            session.run_id = session_id
        self._last_session = session

        # 2. Announce the Conversation on the first turn.
        if not self._conversation_announced:
            await self._emit_conversation_started(session)
            self._conversation_announced = True

        # 3. Emit RunStarted before anything else for this turn.
        await self._emit_run_started(session, user_input)

        # 4. Derive (or accept) goals for this turn. Cross-turn state
        #    lives on ``session.goals`` already (seeded by the
        #    Conversation); we append newly-derived goals that weren't
        #    already present by id so the planner sees the full history.
        try:
            new_goals = await self._resolve_goals(user_input, context)
        except Exception as exc:  # noqa: BLE001
            reason = f"goal derivation failed: {exc}"
            log.exception("goal derivation failed")
            await self._emit_run_aborted(session, reason)
            outcome = ExecutionOutcome(success=False, session=session, reason=reason)
            self._conversation.absorb_turn(
                outcome, user_input_summary=_initial_goal_summary(user_input)
            )
            return outcome

        existing_ids = {g.id for g in session.goals if g.id}
        for g in new_goals:
            if g.id and g.id in existing_ids:
                continue
            session.goals.append(g)
            if g.id:
                existing_ids.add(g.id)

        # goldfive#152: refresh the orchestration-state goals summary
        # so prompt templates / refine paths / downstream planners
        # see an up-to-date ``goldfive.goals_summary``.
        _ostate.refresh_goals_summary(session.state, session.goals)

        await self._emit_goal_derived(session)

        # Build the context passed to the planner. Stamp run_id (so LLM
        # planners can include it in the plan envelope), max
        # reinvocations, and cross-turn context from the Conversation.
        # Caller-supplied context wins on key collisions.
        planner_context: dict[str, Any] = {
            "run_id": session.run_id,
            "max_task_invocations": self.max_task_invocations,
        }
        planner_context.update(self._conversation.prior_turn_context())
        if context:
            planner_context.update(context)
        planner_context["run_id"] = session.run_id

        # 4. Generate the plan.
        # Prefer the richer tree shape (goldfive#151) when the adapter
        # exposes it so the planner can constrain assignee_agent_id to
        # real tree names and render the tree in its prompt. Adapters
        # that don't implement the property fall through to the legacy
        # flat list — keeps back-compat with custom adapters.
        available_agents: Any
        tree = getattr(self.agent, "available_agents_tree", None)
        if isinstance(tree, list) and tree:
            available_agents = list(tree)
        else:
            available_agents = list(self.agent.available_agents)
        try:
            plan = await self.planner.generate(
                goals=session.goals,
                available_agents=available_agents,
                context=planner_context,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"planner.generate raised: {exc}"
            log.exception("planner.generate raised")
            await self._emit_run_aborted(session, reason)
            outcome = ExecutionOutcome(success=False, session=session, reason=reason)
            self._conversation.absorb_turn(
                outcome, user_input_summary=_initial_goal_summary(user_input)
            )
            return outcome

        if plan is None:
            reason = "no plan generated"
            await self._emit_run_aborted(session, reason)
            outcome = ExecutionOutcome(success=False, session=session, reason=reason)
            self._conversation.absorb_turn(
                outcome, user_input_summary=_initial_goal_summary(user_input)
            )
            return outcome

        # Planners may leave run_id blank; stamp ours so downstream sinks
        # correlate cleanly.
        if not plan.run_id:
            plan.run_id = session.run_id
        session.plan = plan

        # goldfive#152: record the installed plan id on the
        # orchestration-state dict so downstream components don't
        # need to reach into ``session.plan`` to read the current
        # plan id.
        _ostate.set_current_plan(session.state, plan)

        await self._emit_plan_submitted(session, plan)

        # 5. Register the seven canonical reporting tools on the adapter.
        try:
            await self.agent.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))
        except Exception as exc:  # noqa: BLE001
            reason = f"register_reporting_tools raised: {exc}"
            log.exception("register_reporting_tools raised")
            await self._emit_run_aborted(session, reason)
            outcome = ExecutionOutcome(success=False, session=session, reason=reason)
            self._conversation.absorb_turn(
                outcome, user_input_summary=_initial_goal_summary(user_input)
            )
            return outcome

        # 6. Bind the steerer. (The executor may re-bind — that's fine.)
        try:
            self.steerer.bind(sinks=list(self.sinks), planner=self.planner)
        except Exception as exc:  # noqa: BLE001
            reason = f"steerer.bind raised: {exc}"
            log.exception("steerer.bind raised")
            await self._emit_run_aborted(session, reason)
            outcome = ExecutionOutcome(success=False, session=session, reason=reason)
            self._conversation.absorb_turn(
                outcome, user_input_summary=_initial_goal_summary(user_input)
            )
            return outcome

        # 6b. Wire the steerer into the adapter. Adapter plugin callbacks
        # (e.g. ADKAdapter's ``_emit_observability``) short-circuit when
        # ``SessionContext.steerer`` is ``None`` — without this call the
        # new sink events (``AgentInvocationStarted`` /
        # ``AgentInvocationCompleted`` / ``DelegationObserved``) never
        # fire. Every built-in adapter (ADK, Claude, Callable) exposes
        # ``bind_steerer``; probe with getattr so third-party adapters
        # that predate the protocol addition still work.
        bind_adapter_steerer = getattr(self.agent, "bind_steerer", None)
        if bind_adapter_steerer is not None:
            try:
                bind_adapter_steerer(self.steerer)
            except Exception as exc:  # noqa: BLE001
                reason = f"adapter.bind_steerer raised: {exc}"
                log.exception("adapter.bind_steerer raised")
                await self._emit_run_aborted(session, reason)
                outcome = ExecutionOutcome(success=False, session=session, reason=reason)
                self._conversation.absorb_turn(
                    outcome, user_input_summary=_initial_goal_summary(user_input)
                )
                return outcome

        # 6c. Wire the adapter back into the steerer. Optional hook
        # (goldfive#139) the steerer uses to tag the adapter's next
        # mid-invocation cancel with a symbolic reason on USER_STEER
        # drift, so the synthetic function_response the adapter appends
        # on cancel carries LLM-actionable content. Duck-typed on
        # purpose — custom Steerers that don't implement
        # ``bind_adapter`` skip this silently.
        bind_steerer_adapter = getattr(self.steerer, "bind_adapter", None)
        if callable(bind_steerer_adapter):
            try:
                bind_steerer_adapter(self.agent)
            except Exception as exc:  # noqa: BLE001
                log.debug("steerer.bind_adapter raised: %s", exc)

        # 7. Hand off to the executor.
        try:
            executor_kwargs: dict[str, Any] = dict(
                plan=session.plan,
                session=session,
                adapter=self.agent,
                steerer=self.steerer,
                planner=self.planner,
                sinks=list(self.sinks),
            )
            if self.control is not None:
                executor_kwargs["control"] = self.control
            # Overlay model (goldfive#141): pass the original user
            # request through to the executor so an overlay-capable
            # :class:`SequentialExecutor` can hand it verbatim to
            # ``adapter.invoke_passthrough``. Best-effort via
            # inspection — executors that don't accept
            # ``user_input=`` keep working with the legacy kwargs.
            if isinstance(user_input, str):
                run_sig = inspect.signature(self.executor.run)
                if "user_input" in run_sig.parameters:
                    executor_kwargs["user_input"] = user_input
            outcome = await self.executor.run(**executor_kwargs)
        except Exception as exc:  # noqa: BLE001
            reason = f"executor.run raised: {exc}"
            log.exception("executor.run raised")
            await self._emit_run_aborted(session, reason)
            aborted = ExecutionOutcome(success=False, session=session, reason=reason)
            self._conversation.absorb_turn(
                aborted, user_input_summary=_initial_goal_summary(user_input)
            )
            return aborted

        # goldfive#152: clear the current_task_* stamp at run end.
        # The plan id + goals summary stay (they remain meaningful
        # cross-turn on the owning Conversation).
        _ostate.clear_current_task(session.state)
        _ostate.clear_active_steer(session.state)

        self._conversation.absorb_turn(
            outcome, user_input_summary=_initial_goal_summary(user_input)
        )
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
    # cross-turn conversation
    # ------------------------------------------------------------------

    @property
    def conversation_id(self) -> str:
        """The current Conversation's stable id. Changes only via :meth:`new_conversation`."""
        return self._conversation.id

    @property
    def conversation(self) -> Conversation:
        """The live :class:`Conversation` object. Read-only handle for inspection."""
        return self._conversation

    async def new_conversation(self, *, reason: str = "") -> None:
        """Reset cross-turn state. The next :meth:`run` starts a fresh Conversation.

        Emits a ``ConversationEnded`` event for the outgoing Conversation
        (if it had any turns), then installs a fresh one. The new
        Conversation is announced lazily — ``ConversationStarted`` fires
        on the next :meth:`run` call.
        """
        outgoing = self._conversation
        if self._conversation_announced and self._last_session is not None:
            await self._emit_conversation_ended(
                conversation=outgoing,
                session_anchor=self._last_session,
                reason=reason or "new_conversation",
            )
        self._conversation = Conversation.new()
        self._conversation_announced = False
        self._last_session = None

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close every sink, then invoke registered close hooks. Idempotent.

        Emits ``ConversationEnded`` for the active Conversation (if any
        turns ran) before closing sinks, so persisted logs have a clean
        terminal marker. Close hooks registered via
        :meth:`add_close_hook` run in registration order AFTER sinks are
        closed; a raising hook is logged and does not prevent subsequent
        hooks from running. A second call to :meth:`close` is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        if self._conversation_announced and self._last_session is not None:
            try:
                await self._emit_conversation_ended(
                    conversation=self._conversation,
                    session_anchor=self._last_session,
                    reason="runner_close",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("conversation_ended emission raised: %s", exc)
            self._conversation_announced = False
        for sink in self.sinks:
            try:
                await sink.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("sink.close() raised: %s", exc)
        # Auto-close LLM callables on the planner and goal-deriver if they
        # implement the optional ``close`` shape (see goldfive._llm).
        # Standard SDK clients (OpenAI AsyncOpenAI, ADK LiteLlm, …) own
        # an aiohttp session that leaks unless explicitly closed.
        await maybe_close_call_llm(
            getattr(self.planner, "_call_llm", None), label="planner.call_llm"
        )
        await maybe_close_call_llm(
            getattr(self.goal_deriver, "_call_llm", None),
            label="goal_deriver.call_llm",
        )
        for hook in self._close_hooks:
            try:
                await hook()
            except Exception as exc:  # noqa: BLE001
                log.warning("close hook raised: %s", exc)

    # ------------------------------------------------------------------
    # Extension API — post-construction wiring for sinks, hooks, control
    # ------------------------------------------------------------------

    def add_sink(self, sink: EventSink) -> None:
        """Register an additional :class:`EventSink`.

        Takes effect for events emitted by subsequent calls to
        :meth:`run`. In-flight runs continue with whatever sink list
        they were handed to the executor at kickoff.
        """
        self.sinks.append(sink)

    def add_close_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        """Register an async callable invoked by :meth:`close` after sinks.

        Hooks fire in registration order, AFTER the Runner's internal
        teardown (sinks closed). An exception in one hook is logged
        via :mod:`logging` and does not prevent subsequent hooks from
        running — failing cleanup must not hang a process.
        """
        self._close_hooks.append(hook)

    @property
    def control(self) -> ControlChannel | None:
        """The attached :class:`~goldfive.control.ControlChannel`, if any."""
        return self._control

    @control.setter
    def control(self, value: ControlChannel) -> None:
        """Attach a :class:`ControlChannel` post-construction.

        Idempotent when the same channel (by identity, ``is``) is
        re-attached. Raises :class:`RuntimeError` if a different
        channel is already attached — callers must construct a fresh
        Runner rather than swap channels mid-lifetime.
        """
        if self._control is value:
            return
        if self._control is not None:
            raise RuntimeError(
                "Runner already has a control channel attached; "
                "detach it first or construct the runner with a specific one."
            )
        self._control = value

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
                f"Runner.run: user_input must be str or list[Goal], got {type(user_input).__name__}"
            )
        goals = await self.goal_deriver.derive(user_input, context=context)
        if not goals:
            raise ValueError("GoalDeriver returned an empty goals list")
        return list(goals)

    async def _emit_run_started(self, session: Session, user_input: str | list[Goal]) -> None:
        evt = run_started_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            goal_summary=_initial_goal_summary(user_input),
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _emit_goal_derived(self, session: Session) -> None:
        evt = goal_derived_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            goals=list(session.goals),
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _emit_plan_submitted(self, session: Session, plan: Any) -> None:
        evt = plan_submitted_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            plan=plan,
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _emit_run_aborted(self, session: Session, reason: str) -> None:
        evt = run_aborted_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            reason=reason,
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _emit_conversation_started(self, session: Session) -> None:
        evt = conversation_started_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            conversation_id=self._conversation.id,
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _emit_conversation_ended(
        self,
        *,
        conversation: Conversation,
        session_anchor: Session,
        reason: str,
    ) -> None:
        # Piggy-back on the last turn's sequence counter so the
        # envelope's sequence field stays monotonic within its run_id.
        evt = conversation_ended_event(
            run_id=session_anchor.run_id,
            sequence=session_anchor.next_sequence(),
            conversation_id=conversation.id,
            turn_count=len(conversation.turns),
            reason=reason,
            session_id=session_anchor.id,
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
