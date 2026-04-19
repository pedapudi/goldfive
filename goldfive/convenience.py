"""One-line ergonomics for wrapping an existing agent in goldfive.

Most callers want a two-step flow:

    runner = goldfive.wrap(my_agent)
    outcome = await runner.run("do the thing")

or, even shorter:

    outcome = await goldfive.run(my_agent, "do the thing")

:func:`wrap` picks a concrete :class:`~goldfive.protocols.AgentAdapter`
via :func:`goldfive.adapters.auto.auto_adapter`, tries to reuse the
agent's existing LLM surface (currently only ADK), and wires a
:class:`~goldfive.Runner` with the usual defaults — a
:class:`LLMPlanner` / :class:`LLMGoalDeriver` pair when an LLM is
available, else the degraded :class:`PassthroughPlanner` /
:class:`LiteralGoalDeriver` pair. Every component is overridable via
keyword argument.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from goldfive._llm_detect import CallLLM, detect_llm
from goldfive.adapters.auto import auto_adapter
from goldfive.executors.sequential import SequentialExecutor
from goldfive.goal_deriver import LiteralGoalDeriver, LLMGoalDeriver
from goldfive.planner import LLMPlanner, PassthroughPlanner
from goldfive.protocols import (
    EventSink,
    Executor,
    GoalDeriver,
    Planner,
    Steerer,
)
from goldfive.results import ExecutionOutcome
from goldfive.runner import Runner
from goldfive.sinks import LoggingSink
from goldfive.steerer import DefaultSteerer
from goldfive.types import Goal

if TYPE_CHECKING:
    from goldfive.control import ControlChannel

log = logging.getLogger("goldfive.wrap")


def wrap(
    agent: Any,
    *,
    planner: Planner | None = None,
    goal_deriver: GoalDeriver | None = None,
    executor: Executor | None = None,
    steerer: Steerer | None = None,
    sinks: list[EventSink] | None = None,
    control: ControlChannel | None = None,
    call_llm: CallLLM | None = None,
    model: str | None = None,
    max_plan_reinvocations: int = 32,
) -> Runner:
    """Build a :class:`Runner` that drives ``agent`` with goldfive.

    Parameters
    ----------
    agent:
        Any of the shapes :func:`auto_adapter` accepts: an existing
        :class:`AgentAdapter`, an ADK ``BaseAgent`` / ``Runner``, a
        Claude SDK client factory, or an async ``(task, session, tools)
        -> InvocationResult`` callable.
    planner:
        Optional :class:`Planner` override. Wins over the default
        :class:`LLMPlanner` / :class:`PassthroughPlanner`.
    goal_deriver:
        Optional :class:`GoalDeriver` override. Wins over the default
        :class:`LLMGoalDeriver` / :class:`LiteralGoalDeriver`.
    executor:
        Optional :class:`Executor` override. Defaults to
        :class:`SequentialExecutor(max_plan_reinvocations=...)`.
    steerer:
        Optional :class:`Steerer` override. Defaults to
        :class:`DefaultSteerer`.
    sinks:
        Optional sink list override. Defaults to ``[LoggingSink()]``.
        Passing an explicit empty list suppresses all sinks.
    control:
        Optional :class:`~goldfive.control.ControlChannel` forwarded
        into the :class:`Runner`. Enables live pause / resume / cancel /
        steer / rewind from an external controller.
    call_llm:
        Optional async ``(system, user, model) -> str`` callable. When
        provided, it is used for both the default planner and the
        default goal deriver, overriding any LLM surface detected on
        the agent.
    model:
        Optional model name passed to :class:`LLMPlanner` /
        :class:`LLMGoalDeriver`. Ignored when ``call_llm`` is omitted
        and no LLM is detected on the agent.
    max_plan_reinvocations:
        Cap on how many times the executor may re-invoke the planner
        for refine loops. Default ``32``; flowed into both the
        :class:`SequentialExecutor` default and the :class:`Runner`.

    Returns
    -------
    Runner
        A ready-to-use :class:`Runner`. Call ``await runner.run(...)``
        or use :func:`goldfive.run` for the one-line variant.
    """
    adapter = auto_adapter(agent)

    resolved_call_llm: CallLLM | None = call_llm
    resolved_model: str = model or ""

    if resolved_call_llm is None and (planner is None or goal_deriver is None):
        detected = detect_llm(agent)
        if detected is not None:
            resolved_call_llm, detected_model = detected
            if not resolved_model:
                resolved_model = detected_model

    resolved_planner: Planner
    if planner is not None:
        resolved_planner = planner
    elif resolved_call_llm is not None:
        resolved_planner = LLMPlanner(call_llm=resolved_call_llm, model=resolved_model)
    else:
        log.debug(
            "goldfive.wrap: no call_llm and no detectable LLM on agent "
            "(%s); falling back to PassthroughPlanner",
            type(agent).__name__,
        )
        resolved_planner = PassthroughPlanner()

    resolved_goal_deriver: GoalDeriver
    if goal_deriver is not None:
        resolved_goal_deriver = goal_deriver
    elif resolved_call_llm is not None:
        resolved_goal_deriver = LLMGoalDeriver(
            call_llm=resolved_call_llm,
            model=resolved_model,
        )
    else:
        log.debug(
            "goldfive.wrap: no call_llm available; falling back to "
            "LiteralGoalDeriver (user_input becomes a single Goal)"
        )
        resolved_goal_deriver = LiteralGoalDeriver()

    resolved_executor: Executor = executor or SequentialExecutor(
        max_plan_reinvocations=max_plan_reinvocations
    )
    resolved_steerer: Steerer = steerer or DefaultSteerer()
    resolved_sinks: list[EventSink] = (
        list(sinks) if sinks is not None else [LoggingSink()]
    )

    return Runner(
        agent=adapter,
        planner=resolved_planner,
        executor=resolved_executor,
        goal_deriver=resolved_goal_deriver,
        steerer=resolved_steerer,
        sinks=resolved_sinks,
        control=control,
        max_plan_reinvocations=max_plan_reinvocations,
    )


async def run(
    agent: Any,
    user_input: str | list[Goal],
    *,
    context: Mapping[str, Any] | None = None,
    **wrap_kwargs: Any,
) -> ExecutionOutcome:
    """Wrap ``agent`` and run it against ``user_input`` in one call.

    Equivalent to::

        runner = goldfive.wrap(agent, **wrap_kwargs)
        return await runner.run(user_input, context=context)

    Any keyword arguments other than ``context`` are forwarded to
    :func:`wrap`. Does not call :meth:`Runner.close`; callers that
    need deterministic sink teardown should build their own runner.
    """
    runner = wrap(agent, **wrap_kwargs)
    return await runner.run(user_input, context=context)


__all__ = ["run", "wrap"]
