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

import asyncio
import inspect
import logging
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive import orchestration_state as _ostate
from goldfive._llm import maybe_close_call_llm
from goldfive.conversation import Conversation
from goldfive.events import (
    conversation_ended_event,
    conversation_started_event,
    emit,
    goal_derived_event,
    run_aborted_event,
    run_started_event,
)
from goldfive.goal_deriver import PassthroughGoalDeriver
from goldfive.reporting import BUILTIN_REPORTING_TOOLS
from goldfive.results import ExecutionOutcome
from goldfive.steerer import DefaultSteerer
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
)

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
    planner_gate:
        Per-turn planning behaviour. Goldfive#271 Phase 4 collapsed
        the prior gate-then-refine pipeline into the planner's own
        :meth:`Planner.handle_turn` method, so the gate is no longer
        a separate layer. This kwarg is retained as a feature switch:

        * ``"auto"`` (default) — call ``planner.handle_turn`` on
          every turn. The planner LLM either produces the next plan
          (warrants change) or returns ``None`` (purely conversational
          — the Runner reuses ``session.plan`` unchanged). The
          "classification" is emergent: did the LLM produce a plan
          or not. Recommended production setting.
        * ``None`` — disable handle_turn entirely; every turn falls
          through to ``Planner.generate`` (pre-#271 behaviour, useful
          for deterministic replay).

        Skipped when ``user_input`` is already a ``list[Goal]`` (the
        caller has opted out of natural-language derivation).
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
        planner_gate: Any = "auto",
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
        elif (
            goal_drift_enabled
            and hasattr(self.steerer, "_goal_drift_call_llm")
            and self.steerer._goal_drift_call_llm is None
        ):
            # Soft-fail: the docstring at ``DefaultSteerer.__init__`` says
            # the Runner wires its planner LLM here when the feature is
            # on. If no callable is present the judge can never fire;
            # surface that once rather than failing silently. Don't raise
            # -- existing Runner(...) callers that build a steerer
            # without a judge intentionally (mock tests, degraded LLMs)
            # must still construct cleanly. See goldfive#217.
            log.warning(
                "goal_drift_enabled=True but no call_llm wired on steerer; "
                "goal-drift judge disabled"
            )
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
        # Turn-aware planning gate. Goldfive#271 Phase 4: the gate is
        # now ``planner.handle_turn`` (a single LLM call that classifies
        # AND produces the merged plan). ``"auto"`` (default) calls it
        # on every turn after the first when the planner exposes the
        # method; ``None`` disables it entirely so every turn re-plans
        # via ``Planner.generate`` (pre-#271 behaviour, useful for
        # deterministic replay).
        self._planner_gate: Any = planner_gate
        # Held across turns so the ``conversational`` path can carry
        # the prior plan forward onto the freshly minted session
        # without re-running the planner.
        self._last_plan: Plan | None = None

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

        # 3a. Seed session.plan with the prior plan (or Plan.empty()
        # on the very first turn) so :meth:`Planner.handle_turn` always
        # sees a non-None ``session.plan``. The Runner has a single
        # install path post-Phase-4: every plan landed by the planner
        # becomes the next revision of this seed (revision_index += 1).
        if self._last_plan is not None:
            self._last_plan.run_id = session.run_id
            session.plan = self._last_plan
        else:
            session.plan = Plan.empty(run_id=session.run_id)
        _ostate.set_current_plan(session.state, session.plan)

        # 4. Derive (or accept) goals for this turn. Cross-turn state
        #    lives on ``session.goals`` already (seeded by the
        #    Conversation); we append newly-derived goals that weren't
        #    already present by id so the planner sees the full history.
        try:
            new_goals = await self._resolve_goals(user_input, context, session=session)
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
        # so prompt templates / handle_turn / downstream planners see
        # an up-to-date ``goldfive.goals_summary``.
        _ostate.refresh_goals_summary(session.state, session.goals)

        await self._emit_goal_derived(session)

        # 4a. Per-turn planner decision (goldfive#271 Phase 4).
        # ``handle_turn`` is a single LLM call that decides whether
        # the new user_input warrants a plan change and, when it
        # does, produces the next revision of session.plan in one
        # shot. Replaces the prior multi-stage pipeline (regex
        # short-circuits + LLM gate + synthesize_goal_from_steer +
        # qualification-merge regex + planner.refine). All plan
        # changes are revisions: the conversation's plan_id is
        # stable, revision_index increments monotonically.
        #
        # Returns ``None`` when the user_input is purely
        # conversational and the current revision still describes
        # the right work — the Runner reuses ``session.plan`` for
        # this turn (driving the executor over the existing plan).
        #
        # Returns the next ``Plan`` revision when a change is
        # warranted — the Runner installs it via the unified
        # ``_install_revision`` path so PlanRevised fires uniformly.
        #
        # Skipped when ``user_input`` is already a ``list[Goal]``
        # (caller has opted out of NL derivation), when
        # ``planner_gate=None`` (deterministic replay mode), and
        # when the planner doesn't implement ``handle_turn`` (legacy
        # PassthroughPlanner / third-party stubs — Runner falls
        # through to ``planner.generate`` once for back-compat).
        next_plan: Plan | None = None
        decided = False
        if (
            self._planner_gate is not None
            and isinstance(user_input, str)
            and hasattr(self.planner, "handle_turn")
        ):
            try:
                next_plan = await self._invoke_handle_turn(
                    user_input=user_input,
                    session=session,
                    context=context,
                )
                decided = True
                log.info(
                    "Runner.run: handle_turn produced_plan=%s "
                    "(source=runner-inline) prior_plan_id=%s "
                    "user_input_first=%r",
                    "yes" if next_plan is not None else "no",
                    (session.plan.id or "")[:16] or "<none>",
                    user_input[:80],
                )
            except Exception as exc:  # noqa: BLE001
                # A misbehaving handle_turn must never break the run;
                # fall through to generate (legacy first-turn path).
                log.warning(
                    "planner.handle_turn raised; falling through to "
                    "generate: %s",
                    exc,
                )
                decided = False

        # If handle_turn was skipped, raised, OR the planner doesn't
        # implement handle_turn meaningfully (returns None on the
        # very first turn against an empty seed — true for
        # PassthroughPlanner / StaticPlanner / non-LLM planners),
        # fall through to ``planner.generate`` so a brand-new plan
        # still lands. ``planner.generate`` is the legacy path the
        # Runner used pre-Phase-4; preserved for back-compat with
        # planners that don't implement Phase 4's per-turn LLM call.
        first_turn_seed = not session.plan.tasks
        needs_generate_fallback = (not decided) or (
            decided and next_plan is None and first_turn_seed
        )
        if needs_generate_fallback:
            available_agents: Any
            tree = getattr(self.agent, "available_agents_tree", None)
            if isinstance(tree, list) and tree:
                available_agents = list(tree)
            else:
                available_agents = list(self.agent.available_agents)
            planner_context: dict[str, Any] = {
                "run_id": session.run_id,
                "max_task_invocations": self.max_task_invocations,
            }
            planner_context.update(self._conversation.prior_turn_context())
            if context:
                planner_context.update(context)
            planner_context["run_id"] = session.run_id
            try:
                next_plan = await self.planner.generate(
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

        # Install the produced plan as the next revision of
        # session.plan, OR (when next_plan is None and a real prior
        # exists) reuse session.plan unchanged so the executor drives
        # the coordinator over existing context.
        if next_plan is not None:
            installed = await self._install_revision(
                session=session,
                user_input=user_input,
                revised_plan=next_plan,
            )
            if not installed:
                reason = "plan revision rejected by validator"
                await self._emit_run_aborted(session, reason)
                outcome = ExecutionOutcome(success=False, session=session, reason=reason)
                self._conversation.absorb_turn(
                    outcome, user_input_summary=_initial_goal_summary(user_input)
                )
                return outcome
        elif not session.plan.tasks:
            # First turn AND handle_turn returned None (purely
            # conversational on an empty seed). No plan to drive the
            # executor over — abort cleanly.
            reason = "no plan generated"
            await self._emit_run_aborted(session, reason)
            outcome = ExecutionOutcome(success=False, session=session, reason=reason)
            self._conversation.absorb_turn(
                outcome, user_input_summary=_initial_goal_summary(user_input)
            )
            return outcome
        else:
            # Conversational follow-up on a real prior plan. Reuse
            # session.plan unchanged. No PlanRevised — the prior
            # revision is still the right one for this turn.
            log.info(
                "Runner.run: conversational turn — reusing prior "
                "plan_id=%s revision_index=%d",
                (session.plan.id or "")[:16] or "<none>",
                int(session.plan.revision_index),
            )

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
        finally:
            # planner-gate: snapshot the turn's final plan so the next
            # turn's planner_gate can classify against it.
            #
            # Rationale (Phase 2.X v2 / goldfive#271 Gap 1): the prior
            # post-success-path stash (PR #282) was bypassed when ADK
            # closed the runner mid-flight — the executor coroutine was
            # cancelled, ``CancelledError`` propagated out of
            # ``await self.executor.run(...)``, and since Py 3.8
            # ``CancelledError`` is a ``BaseException`` (not an
            # ``Exception``) the ``except Exception`` handler above did
            # NOT catch it. Control flowed out of ``run`` entirely and
            # the stash was skipped, leaving ``_last_plan = None`` for
            # the next turn even though the turn produced a real plan.
            # The ADK-web user-steer flow hit this on validation v2:
            # zero "stashed prior plan" log lines across 4 turns and
            # "GoldfiveADKAgent: gate skipped — no prior plan" 4 times.
            #
            # Putting the stash in ``finally`` runs it regardless of how
            # the executor exited — normal success, ``Exception`` (e.g.
            # planner bind error), or ``BaseException`` (e.g.
            # ``CancelledError`` from ADK closing the runner mid-stream).
            # The exception still propagates after the stash; this block
            # does not swallow it.
            if session.plan is not None and session.plan.tasks:
                self._last_plan = session.plan
                log.info(
                    "Runner.run: stashed prior plan for next turn's "
                    "handle_turn (plan_id=%s revision_index=%d)",
                    (session.plan.id or "")[:16] or "<none>",
                    int(session.plan.revision_index),
                )

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
    # run_streamed — yield inner-adapter framework events in real time
    # ------------------------------------------------------------------

    async def run_streamed(
        self,
        user_input: str | list[Goal],
        *,
        context: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Any]:
        """Execute a run and stream inner-adapter framework events out as they arrive.

        Async generator that yields — in order — every framework-native
        event the adapter observes mid-invocation (e.g. ADK ``Event``
        objects: ``transfer_to_agent``, model text parts, function
        calls, function responses) followed by exactly one trailing
        :class:`~goldfive.results.ExecutionOutcome` as the final
        yielded element when the run finishes.

        The trailing ``ExecutionOutcome`` is how callers recover the
        completed run's success flag, reason, and live
        :class:`~goldfive.types.Session`. Consumers distinguish the
        two yielded shapes via ``isinstance(item, ExecutionOutcome)``.

        The equivalent of :meth:`run` — same lifecycle, same sinks, same
        plugin callbacks, same conversation bookkeeping — is driven in
        the background. :meth:`run_streamed` does NOT call :meth:`run`
        recursively; it subscribes to the adapter's event fan-out
        (:meth:`ADKAdapter.subscribe_adk_events`, when available) and
        forwards those events through an :class:`asyncio.Queue` so
        backpressure from the consumer cannot stall the inner Runner.

        Non-ADK adapters (callable, Claude SDK) have no streamable
        framework events; :meth:`run_streamed` still works for them —
        it simply yields no mid-run events and produces the outcome at
        the end, exactly as :meth:`run` would. Callers do not need to
        switch on adapter type.

        This is the primary path used by
        :class:`~goldfive.adapters.adk_wrap.GoldfiveADKAgent` so
        ``adk web`` sees per-agent activity (LLM responses, tool calls,
        agent transitions) in its UI while the goldfive pipeline runs.

        Parameters mirror :meth:`run` — see that docstring for
        ``session_id`` semantics.
        """
        # Import here so non-ADK consumers don't pay the optional-ADK
        # import cost when they never call run_streamed.
        queue: asyncio.Queue[Any] = asyncio.Queue()

        # Subscribe a sync listener to the adapter's raw-event fan-out
        # when the adapter supports it. The listener enqueues every
        # event into ``queue`` for us to re-yield. Non-ADK adapters
        # simply don't expose ``subscribe_adk_events`` — the run still
        # completes and we yield only the final outcome.
        subscribe = getattr(self.agent, "subscribe_adk_events", None)
        unsubscribe = getattr(self.agent, "unsubscribe_adk_events", None)

        def _listener(event: Any) -> None:
            # ``put_nowait`` is correct here: the queue is unbounded
            # so it never raises, and we must NOT block the adapter's
            # event loop on a consumer that's slow to pull.
            try:
                queue.put_nowait(event)
            except Exception:  # noqa: BLE001 — defensive; unbounded queue shouldn't raise
                log.debug("run_streamed: queue.put_nowait unexpectedly raised")

        if callable(subscribe):
            subscribe(_listener)

        # Sentinel that tells the consumer loop the run is done and
        # any remaining events have already been enqueued.
        _DONE = object()

        async def _drive() -> ExecutionOutcome:
            try:
                return await self.run(
                    user_input,
                    context=context,
                    session_id=session_id,
                )
            finally:
                # Signal end-of-stream regardless of success / failure.
                # The consumer drains any remaining buffered events,
                # then stops when it sees the sentinel.
                queue.put_nowait(_DONE)

        run_task: asyncio.Task[ExecutionOutcome] = asyncio.create_task(_drive())

        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                yield item
            outcome = await run_task
            yield outcome
        except (asyncio.CancelledError, GeneratorExit):
            # Upstream cancelled us (adk-web disconnect) OR the caller
            # aclose()'d the generator early. Propagate the cancel into
            # the driver so its ``try/finally`` teardown runs, then
            # await it to collect the final state — suppressing the
            # CancelledError so the generator exits cleanly.
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            raise
        finally:
            if callable(unsubscribe):
                try:
                    unsubscribe(_listener)
                except Exception as exc:  # noqa: BLE001
                    log.debug("run_streamed: unsubscribe raised: %s", exc)

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
        # planner-gate: reset turn-aware gate state so the first turn
        # of the new conversation runs full planning.
        self._last_plan = None

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
        # Drain background reasoning-judge tasks the steerer scheduled
        # via its fire-and-forget judge path (goldfive#251). Bounded
        # shutdown so a hung LLM judge doesn't stall close. Duck-typed
        # — custom steerers without ``shutdown`` fall through cleanly.
        steerer_shutdown = getattr(self.steerer, "shutdown", None)
        if callable(steerer_shutdown):
            try:
                await steerer_shutdown()
            except Exception as exc:  # noqa: BLE001
                log.warning("steerer.shutdown() raised: %s", exc)
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

    async def _invoke_handle_turn(
        self,
        *,
        user_input: str,
        session: Session,
        context: Mapping[str, Any] | None,
    ) -> Plan | None:
        """Invoke ``planner.handle_turn`` with the runner's per-turn context.

        Goldfive#271 Phase 4 entry point. The planner reads the prior
        plan + goals off ``session.plan`` / ``session.goals``; the
        Runner threads the available agents and the per-turn context
        (run_id, max_task_invocations, prior_turns) so the planner has
        everything it needs in one call.
        """
        # Prefer the richer tree shape (goldfive#151) when the adapter
        # exposes it. Adapters that don't implement the property fall
        # through to the legacy flat list — keeps back-compat.
        available_agents: Any
        tree = getattr(self.agent, "available_agents_tree", None)
        if isinstance(tree, list) and tree:
            available_agents = list(tree)
        else:
            available_agents = list(self.agent.available_agents)
        planner_context: dict[str, Any] = {
            "run_id": session.run_id,
            "max_task_invocations": self.max_task_invocations,
        }
        planner_context.update(self._conversation.prior_turn_context())
        if context:
            planner_context.update(context)
        planner_context["run_id"] = session.run_id
        return await self.planner.handle_turn(
            user_input=user_input,
            session=session,
            conversation_history=list(self._conversation.turns),
            available_agents=available_agents,
            context=planner_context,
        )

    async def _install_revision(
        self,
        *,
        session: Session,
        user_input: str | list[Goal],
        revised_plan: Plan,
    ) -> bool:
        """Install ``revised_plan`` as the next revision of ``session.plan``.

        Goldfive#271 Phase 4 unified install path: every plan change
        becomes a revision (revision_index += 1, plan_id preserved).
        On the very first turn ``session.plan`` was seeded with
        :meth:`Plan.empty` so this still produces revision 1 with a
        fresh PlanRevised event.

        Routes through :meth:`DefaultSteerer.apply_user_steer_with_plan`
        so the steerer's USER_STEER bookkeeping (active_steer state,
        dedup), drift event emission, validation, ``_apply_revision``
        (revision_index bump, metadata stamp, current_plan refresh),
        and ``_emit_plan_revised`` (with paired RefineAttempted
        envelopes) all fire uniformly.

        Returns ``True`` on success, ``False`` on validation failure
        (the caller should surface RunAborted in that case).
        """
        # Bind the steerer + adapter so the install pipeline has
        # sinks + planner + adapter wiring. bind() is idempotent —
        # the executor handoff below re-binds with the same args.
        try:
            self.steerer.bind(sinks=list(self.sinks), planner=self.planner)
            bind_adapter_steerer = getattr(self.agent, "bind_steerer", None)
            if bind_adapter_steerer is not None:
                bind_adapter_steerer(self.steerer)
            bind_steerer_adapter = getattr(self.steerer, "bind_adapter", None)
            if callable(bind_steerer_adapter):
                bind_steerer_adapter(self.agent)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Runner._install_revision: steerer/adapter bind raised: %s", exc
            )
            return False
        # Stamp run_id on the revised plan so sink emissions correlate
        # with this turn.
        if not revised_plan.run_id:
            revised_plan.run_id = session.run_id
        # Goldfive#271 Phase 4: USER_STEER drift coerces the natural-
        # language input into the unified install pipeline. The
        # steerer's apply_user_steer_with_plan does the bookkeeping +
        # validation + revision install + PlanRevised emit.
        user_text = (
            user_input.strip()
            if isinstance(user_input, str)
            else _initial_goal_summary(user_input)
        )
        drift = DriftEvent(
            kind=DriftKind.USER_STEER,
            severity=DriftSeverity.WARNING,
            detail=user_text,
        )
        try:
            installed = await self.steerer.apply_user_steer_with_plan(
                drift=drift,
                session=session,
                revised_plan=revised_plan,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Runner._install_revision: apply_user_steer_with_plan "
                "raised: %s",
                exc,
            )
            return False
        return bool(installed)

    async def _resolve_goals(
        self,
        user_input: str | list[Goal],
        context: Mapping[str, Any] | None,
        session: Session | None = None,
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
        # Merge span-emission context (sinks + session correlation) into
        # the context dict the deriver sees so an ``LLMGoalDeriver`` can
        # emit ``GoldfiveLLMCallStart/End`` spans around its internal
        # call. Overrides caller-supplied values deliberately — the
        # Runner owns the sink list and session id.
        span_ctx: dict[str, Any] = dict(context or {})
        if session is not None:
            span_ctx.setdefault("run_id", session.run_id)
            span_ctx.setdefault("session_id", session.id)
            span_ctx.setdefault("next_sequence", session.next_sequence)
        if self.sinks:
            span_ctx.setdefault("sinks", list(self.sinks))
        goals = await self.goal_deriver.derive(user_input, context=span_ctx)
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
