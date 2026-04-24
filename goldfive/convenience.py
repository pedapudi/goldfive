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
from goldfive.config import JudgeConfig, RuntimeConfig
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


def _build_judge_call_llm(config: JudgeConfig) -> tuple[CallLLM, str] | None:
    """Construct an OpenAI-compatible ``CallLLM`` from a :class:`JudgeConfig`.

    Returns ``(call_llm, model)`` or ``None`` when the ``openai``
    package is not importable / the client cannot be built. Shape
    mirrors :func:`goldfive._llm_detect.make_default_adk_call_llm`: the
    returned callable exposes a ``close`` coroutine so
    :class:`Runner` can tear down its HTTP session on shutdown.

    Design parallels :class:`goldfive.drift._embed._OpenAIEmbeddingBackend`
    — we intentionally tolerate missing / placeholder ``api_key`` so
    llama.cpp / Ollama endpoints "just work" (they don't check the
    header).
    """
    base_url = (config.base_url or "").strip()
    if not base_url:
        return None
    try:
        from openai import AsyncOpenAI  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "goldfive.wrap: openai SDK not importable for JudgeConfig "
            "(base_url=%r): %s",
            base_url,
            exc,
        )
        return None
    timeout_s = max(0.1, config.timeout_ms / 1000.0)
    try:
        client: Any = AsyncOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=config.api_key or "not-needed",
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "goldfive.wrap: AsyncOpenAI client construction failed for "
            "JudgeConfig (base_url=%r): %s",
            base_url,
            exc,
        )
        return None

    model_name = config.model or ""

    async def _call_llm(system: str, user: str, model_str: str) -> str:
        # Prefer the model argument supplied by the caller (matches the
        # contract used by :class:`~goldfive.planner.LLMPlanner`); fall
        # back to the config model when the caller passes the empty
        # string. An empty-string model is tolerated by llama.cpp /
        # Ollama even against an OpenAI endpoint that requires one,
        # because we only hit endpoints the operator configured.
        effective_model = model_str or model_name
        resp = await client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        try:
            content = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return ""
        return str(content)

    async def _close() -> None:
        for attr_name in ("aclose", "close"):
            target = getattr(client, attr_name, None)
            if callable(target):
                try:
                    result = target()
                    if hasattr(result, "__await__"):
                        await result
                    return
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "goldfive.wrap: JudgeConfig client.%s raised %s",
                        attr_name,
                        exc,
                    )
                    return

    _call_llm.close = _close  # type: ignore[attr-defined]
    return _call_llm, model_name


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
        bundling the typed-config surfaces: embedding backend,
        tool-loop detector thresholds, reasoning-drift thresholds,
        goal-drift scheduling, and (added in the silent-disarm
        follow-up) a dedicated :class:`~goldfive.config.JudgeConfig`
        for routing the two drift judges to their own LLM endpoint.
        When ``None`` (the default) ``wrap()``
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

    # Track whether the judge-bound callable came from detect_llm so we
    # can emit the named-model WARNING below. Explicit ``call_llm=`` or
    # an explicit :class:`JudgeConfig` suppresses that warning.
    _call_llm_from_detect: bool = False
    _detected_model_name: str = ""

    # Always auto-detect when the caller did not supply an explicit
    # ``call_llm``. Prior to this fix the auto-detect was guarded by
    # ``(planner is None or goal_deriver is None)`` on the assumption that
    # ``call_llm`` existed only to feed those two. That stopped being
    # true when #218 / #226 wired the goal-drift and reasoning-drift
    # judges through the same callable — callers who supply their own
    # planner + goal_deriver still need the judges armed. Leaving the
    # guard in place produced a silent-disarm: both judges stayed
    # inert, and drift in the agent's reasoning went undetected despite
    # the detectors being "on". See ``docs/design/DRIFT.md`` and the
    # live-session evidence in harmonograf session
    # ``1aa68419-00f3-41eb-bf6e-22d0bdff21ed``.
    if resolved_call_llm is None:
        detected = detect_llm(agent)
        if detected is not None:
            resolved_call_llm, detected_model = detected
            _call_llm_from_detect = True
            _detected_model_name = detected_model
            if not resolved_model:
                resolved_model = detected_model

    # Judge routing (goldfive JudgeConfig + named-model WARNING).
    # Precedence for the two drift judges' call_llm / model:
    #   1. Explicit ``goldfive.wrap(call_llm=...)`` — wins outright.
    #   2. ``resolved_runtime.judge.base_url`` — dedicated judge endpoint.
    #   3. Auto-detected tree LLM (``detect_llm``).
    # The planner + goal_deriver stay on ``resolved_call_llm`` regardless;
    # only the two judges get routed to JudgeConfig when it is set.
    judge_call_llm: CallLLM | None = resolved_call_llm
    judge_model: str = resolved_model
    judge_from_config: bool = False
    if call_llm is None and resolved_runtime.judge.base_url:
        built = _build_judge_call_llm(resolved_runtime.judge)
        if built is not None:
            judge_call_llm, judge_config_model = built
            if judge_config_model:
                judge_model = judge_config_model
            judge_from_config = True
        else:
            log.warning(
                "goldfive.wrap: JudgeConfig.base_url=%r is set but a "
                "CallLLM could not be constructed (openai SDK missing "
                "or client rejected the config); judges will fall back "
                "to the tree LLM.",
                resolved_runtime.judge.base_url,
            )

    # Named-model WARNING (goldfive silent-disarm follow-up). When the
    # judges' callable was inherited from ``detect_llm`` — i.e. the
    # operator did not pass ``call_llm=`` and did not configure a
    # :class:`JudgeConfig` — surface which model is now handling judge
    # traffic so billed / rate-limited cloud endpoints are visible
    # from logs rather than hidden inside the adapter. An explicit
    # ``call_llm=`` or an explicit ``JudgeConfig`` suppresses the
    # warning (the operator has already made a deliberate choice).
    if _call_llm_from_detect and not judge_from_config:
        # Prefer the agent's own ``.name`` ("coordinator_agent", "research_agent",
        # ...) over the Python class name ("LlmAgent", "BaseAgent", ...) — the
        # latter is nearly useless for identifying *which* agent in a tree the
        # LLM was detected from. Fall through to the class name only when the
        # object has no usable ``.name`` attribute (non-ADK shapes).
        _agent_label = getattr(agent, "name", "") or type(agent).__name__
        log.warning(
            "goldfive.wrap: judge LLM not explicitly configured; inheriting "
            "%r from agent %r (detected via ADK model attribute). Set "
            "GOLDFIVE_JUDGE_BASE_URL / GOLDFIVE_JUDGE_MODEL to route "
            "goldfive's judges to a dedicated endpoint.",
            _detected_model_name,
            _agent_label,
        )

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
    elif judge_call_llm is not None:
        # The two drift judges share a single callable: the
        # trajectory-level GOAL_DRIFT judge (goldfive#218) and the
        # per-thinking-message reasoning judge (goldfive#226). Default
        # ``reasoning_drift_mode`` on the steerer is ``"judge"`` -- see
        # :class:`DefaultSteerer`. ``judge_call_llm`` comes from the
        # precedence chain above: explicit > JudgeConfig > detected.
        resolved_steerer = DefaultSteerer(
            goal_drift_call_llm=judge_call_llm,
            goal_drift_model=judge_model,
            goal_drift_config=resolved_runtime.goal_drift,
            tool_loop_config=resolved_runtime.tool_loops,
            reasoning_drift_config=resolved_runtime.reasoning_drift,
            reasoning_drift_mode=resolved_runtime.reasoning_drift.mode,
            reasoning_drift_call_llm=judge_call_llm,
            reasoning_drift_model=judge_model,
        )
    else:
        resolved_steerer = DefaultSteerer(
            goal_drift_config=resolved_runtime.goal_drift,
            tool_loop_config=resolved_runtime.tool_loops,
            reasoning_drift_config=resolved_runtime.reasoning_drift,
            reasoning_drift_mode=resolved_runtime.reasoning_drift.mode,
        )
        # Judges inherit ``call_llm`` from :func:`goldfive.wrap`; with
        # no callable wired here, both the trajectory-level GOAL_DRIFT
        # judge and the per-thinking-message reasoning-drift judge are
        # silently inert. Fail loud so operators can diagnose without
        # trawling source — the most common cause is forgetting to
        # pass ``call_llm=`` (or an ADK agent that ``detect_llm``
        # cannot introspect). The reasoning-drift mode is still
        # honoured; it just won't produce drift events until a
        # callable is wired.
        mode = resolved_runtime.reasoning_drift.mode
        if mode in ("judge", "both"):
            log.warning(
                "goldfive.wrap: reasoning_drift_mode=%r but no call_llm "
                "wired — LLM-as-a-judge drift detection is disabled for "
                "this Runner. Pass call_llm=... to goldfive.wrap() or "
                "use an ADK agent detect_llm() can introspect.",
                mode,
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

    # When the judges were routed through a dedicated JudgeConfig
    # endpoint we constructed our own ``AsyncOpenAI`` client — register
    # its ``close`` as a Runner close-hook so the HTTP session is torn
    # down on ``runner.close()`` rather than leaking until process
    # exit. (Judges inheriting the tree LLM already close via the
    # planner/goal_deriver close path.)
    if judge_from_config and judge_call_llm is not None:
        _close = getattr(judge_call_llm, "close", None)
        if callable(_close):
            runner.add_close_hook(_close)

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
