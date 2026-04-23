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
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from goldfive._llm_detect import CallLLM, detect_llm
from goldfive.adapters.auto import auto_adapter, is_adk_agent
from goldfive.config import RuntimeConfig
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
    max_task_invocations: int | None = None,
    plugins: list[Any] | None = None,
    runtime: RuntimeConfig | None = None,
    **legacy_kwargs: Any,
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
        :class:`SequentialExecutor(max_task_invocations=...)`.
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
    max_task_invocations:
        Optional cap on total adapter invocations per run. Defaults to
        ``None`` (unbounded); flowed into both the
        :class:`SequentialExecutor` default and the :class:`Runner`.
        Accepts the deprecated ``max_plan_reinvocations`` kwarg for one
        release with a :class:`DeprecationWarning`.
    plugins:
        Optional list of ADK ``BasePlugin`` instances to install on
        every per-agent runner built for an ADK wrap target. Forwards
        to :class:`~goldfive.adapters.adk.ADKAdapter` so sub-agent
        dispatches (``AgentTool(...)``, ``sub_agents``) observe the
        same plugins as the coordinator — not just the
        ``App(plugins=[...])``-level root runner. Ignored for
        non-ADK agents. See goldfive#121.
    runtime:
        Optional :class:`~goldfive.config.RuntimeConfig` (goldfive#225)
        bundling the four typed-config surfaces: embedding backend,
        tool-loop detector thresholds, reasoning-drift thresholds, and
        goal-drift scheduling. When ``None`` (the default) ``wrap()``
        builds an instance from the environment via
        :meth:`RuntimeConfig.from_env` so pre-#225 callers get
        byte-identical behaviour. When provided, the config is
        installed into the per-process embedding backend and
        reasoning-drift module, and threaded into the default
        :class:`DefaultSteerer` via its config kwargs. An explicit
        ``steerer=`` kwarg wins — the caller keeps full control
        over the steerer they build themselves.

    Returns
    -------
    Runner
        A ready-to-use :class:`Runner`. Call ``await runner.run(...)``
        or use :func:`goldfive.run` for the one-line variant.

        When ``agent`` is an ADK ``BaseAgent``, the returned object is
        a :class:`~goldfive.adapters.adk_wrap.GoldfiveADKAgent` — a
        ``BaseAgent`` subclass that *also* exposes the Runner surface,
        so the same call site works both programmatically and as the
        ``root_agent`` of an ``adk web`` app. The declared return type
        stays :class:`Runner` for ergonomics; callers who rely on the
        ``BaseAgent`` side can annotate
        ``cast(GoldfiveADKAgent, goldfive.wrap(...))`` themselves.
    """
    adapter = auto_adapter(agent, plugins=plugins)

    # Resolve the runtime config (goldfive#225). When ``runtime`` is
    # not supplied we build one from the environment so pre-#225
    # callers — and the env-var tests they already ship — keep working
    # without modification. The resolved config is then installed
    # eagerly into the embedding backend's module-level state; the
    # reasoning-drift module is installed later inside ``DefaultSteerer``
    # when the steerer-default branch runs (so callers who pass their
    # own ``steerer=`` can make their own installation decisions).
    resolved_runtime: RuntimeConfig = (
        runtime if runtime is not None else RuntimeConfig.from_env()
    )
    from goldfive.drift import _embed as _embed_module
    from goldfive.drift import reasoning as _reasoning_module
    from goldfive.drift import tool_loops as _tool_loops_module

    _embed_module.configure(resolved_runtime.embedding)
    _reasoning_module.configure(resolved_runtime.reasoning_drift)
    _tool_loops_module.configure(resolved_runtime.tool_loops)

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

    if "max_plan_reinvocations" in legacy_kwargs:
        legacy_value = legacy_kwargs.pop("max_plan_reinvocations")
        warnings.warn(
            "goldfive.wrap(max_plan_reinvocations=...) is deprecated; use "
            "max_task_invocations=... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if max_task_invocations is None:
            max_task_invocations = legacy_value
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"goldfive.wrap got unexpected keyword argument(s): {unexpected}")

    # Default executor: overlay mode ON for the wrap() convenience
    # because :class:`ADKAdapter` exposes ``invoke_passthrough`` and
    # that's the safe path for coordinator trees (goldfive#141).
    # Callers supplying their own executor keep full control.
    resolved_executor: Executor = executor or SequentialExecutor(
        max_task_invocations=max_task_invocations,
        overlay_mode=True,
    )
    # Wire the resolved call_llm into the default steerer so the
    # trajectory-level GOAL_DRIFT judge (goldfive#143) actually fires.
    # Without this, ``DefaultSteerer()`` gets ``goal_drift_call_llm=None``
    # and ``note_agent_turn`` short-circuits before the counter advances —
    # the docstring on :class:`DefaultSteerer` promises the Runner wires
    # the planner LLM here when ``goal_drift_enabled`` is on, but until
    # now nothing fulfilled that promise. Never override an
    # explicit user-supplied steerer. See goldfive#217.
    resolved_steerer: Steerer
    if steerer is not None:
        resolved_steerer = steerer
    elif resolved_call_llm is not None:
        resolved_steerer = DefaultSteerer(
            goal_drift_call_llm=resolved_call_llm,
            goal_drift_model=resolved_model,
            goal_drift_config=resolved_runtime.goal_drift,
            tool_loop_config=resolved_runtime.tool_loops,
            reasoning_drift_config=resolved_runtime.reasoning_drift,
        )
    else:
        resolved_steerer = DefaultSteerer(
            goal_drift_config=resolved_runtime.goal_drift,
            tool_loop_config=resolved_runtime.tool_loops,
            reasoning_drift_config=resolved_runtime.reasoning_drift,
        )
    resolved_sinks: list[EventSink] = list(sinks) if sinks is not None else [LoggingSink()]

    runner = Runner(
        agent=adapter,
        planner=resolved_planner,
        executor=resolved_executor,
        goal_deriver=resolved_goal_deriver,
        steerer=resolved_steerer,
        sinks=resolved_sinks,
        control=control,
        max_task_invocations=max_task_invocations,
    )

    if is_adk_agent(agent):
        # Lazy import so callers without the ADK extra don't pay for it.
        from goldfive.adapters.adk_wrap import GoldfiveADKAgent

        return GoldfiveADKAgent(inner=agent, runner=runner)

    return runner


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
