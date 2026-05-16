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
from collections.abc import Iterable, Mapping
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
    from goldfive.judges.builtins import BuiltinJudge

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
            "goldfive.wrap: openai SDK not importable for JudgeConfig (base_url=%r): %s",
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
        # Pull the per-callsite cap (set by goldfive consumers via
        # :func:`goldfive._llm.call_llm_budget`). Default ``4096`` is
        # large enough for plan refines while bounding the worst case
        # under typical Q4 throughput. Pre-fix: unbounded → 9961-token
        # responses (goldfive#271 demo-v8.log).
        from goldfive._llm import get_max_output_tokens, get_thinking_disabled

        # Pull the per-callsite "disable thinking" signal (goldfive#271
        # follow-up to #311). When goldfive's judges / goal_deriver /
        # planner-refine dispatch we set ``enable_thinking=False`` via
        # ``extra_body`` (the Qwen-via-litellm convention) AND prepend
        # ``/no_think`` to the system prompt as a model-prompt-level
        # fallback. Vendors that don't recognise the kwarg drop it
        # server-side; vendors that don't recognise ``/no_think`` ignore
        # the line. Belt-and-suspenders so a misconfigured endpoint
        # still exits the think prelude.
        thinking_disabled = get_thinking_disabled()
        effective_system = system
        extra_body: dict[str, Any] = {}
        if thinking_disabled:
            extra_body["enable_thinking"] = False
            # ``/no_think`` is the Qwen prompt-level toggle. Cheap to
            # include for non-Qwen models — they treat it as ordinary
            # text and ignore it.
            if "/no_think" not in (system or ""):
                effective_system = f"/no_think\n{system}" if system else "/no_think"

        create_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user},
            ],
            "max_tokens": get_max_output_tokens(),
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        try:
            resp = await client.chat.completions.create(**create_kwargs)
        except TypeError as exc:
            # Older OpenAI client versions don't accept ``extra_body``.
            # Retry without it — the ``/no_think`` system-prompt prefix
            # still does its job for Qwen. Other TypeErrors are real
            # failures; fall through.
            if "extra_body" not in create_kwargs:
                raise
            log.debug(
                "goldfive.wrap: AsyncOpenAI rejected extra_body=%r (%s); retrying without it",
                extra_body,
                exc,
            )
            create_kwargs.pop("extra_body", None)
            resp = await client.chat.completions.create(**create_kwargs)
        try:
            content = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return ""
        # Diagnostic for empty-content + non-empty reasoning_content
        # (the OpenAI-compatible analogue of "all-thought, no-answer").
        # Qwen-via-litellm returns reasoning text on a sibling field
        # (``reasoning_content``); when ``content == ""`` but reasoning
        # is present, the model spent its budget thinking and produced
        # no answer. Surface this rather than letting the parser see an
        # indistinguishable empty string.
        result = str(content)
        reasoning_content = ""
        try:
            reasoning_content = getattr(resp.choices[0].message, "reasoning_content", "") or ""
        except Exception:  # noqa: BLE001
            reasoning_content = ""
        _call_llm.last_thought_count = (  # type: ignore[attr-defined]
            1 if reasoning_content else 0
        )
        _call_llm.last_answer_count = 1 if result else 0  # type: ignore[attr-defined]
        if not result and reasoning_content:
            log.info(
                "goldfive.wrap._build_judge_call_llm: model returned "
                "reasoning_content (%d chars) with empty content — check "
                "thinking-mode config or max_output_tokens (the model spent "
                "its budget thinking and emitted no answer).",
                len(reasoning_content),
            )
        return result

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
    dynamic_instruction: bool = True,
    drift_self_reporting: bool | list[str] = False,
    llm_detector: Any = None,
    judge_call_llm_builder: Any = None,
    judges: list[Any] | None = None,
    disable_judges: Iterable[BuiltinJudge | str] | None = None,
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
    dynamic_instruction:
        Default-on (goldfive#251). When ``True`` (the default) every
        reachable ``LlmAgent`` in an ADK wrap target has its static
        ``instruction`` string replaced with a callable resolver that
        re-reads the agent's current-task context from ``session.state``
        at every turn. The effect is plan-causal prompting: when refine
        lands a revised description mid-run, the NEXT turn of the
        affected agent automatically sees the new task without
        transcript rewrite. Pass ``dynamic_instruction=False`` to keep
        the original static strings (e.g. for deterministic prompts
        under test, or when the caller is already managing their own
        dynamic resolution). Ignored for non-ADK agents.
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
    judges:
        Optional list of :class:`~goldfive.judges.Judge` instances
        (goldfive#437). When provided, the steerer iterates this list
        at every observation point and emits a
        :class:`JudgementEmitted` envelope for each populated verdict
        (drift / rubric / boolean / numeric). Drift-flavoured verdicts
        ALSO fire :class:`DriftDetected` exactly as the pre-judges code
        path did, so existing sinks see no behavioural change.

        Defaults to :func:`goldfive.builtin_judges.default_judges` —
        the built-in detector set wrapped as :class:`Judge` instances.
        Pass an explicit empty list (``judges=[]``) to opt out of the
        new event surface entirely (the legacy hardcoded detector
        path still runs but emits no
        :class:`JudgementEmitted` envelopes). Pass a custom mix to
        register agent-specific quality signals alongside (or in
        place of) the built-ins::

            runner = goldfive.wrap(
                agent,
                judges=[
                    goldfive.builtin_judges.reasoning_drift(),
                    MyCustomLengthJudge(),
                    MyRubricJudge(rubric="..."),
                ],
            )

    disable_judges:
        Optional iterable of
        :class:`~goldfive.builtin_judges.BuiltinJudge` members (or their
        wire-name strings) to drop from the **default** judge set. The
        typed, surgical opt-out: keep every built-in detector except the
        ones named here, without having to re-list the rest via
        ``judges=``. Mutually exclusive with an explicit ``judges=``
        list — when ``judges=`` is supplied the caller is already
        specifying the exact set, so ``disable_judges`` is rejected with
        a :class:`TypeError` rather than silently ignored::

            # Keep the full default set minus the tool-error judge —
            # e.g. an agent that legitimately makes no tool calls.
            runner = goldfive.wrap(
                agent,
                disable_judges=[goldfive.builtin_judges.BuiltinJudge.TOOL_ERROR],
            )

        An unrecognised entry is ignored (forward-compatible). ``None``
        (the default) keeps the full default set.

    drift_self_reporting:
        Forwarded to :class:`Runner`. Default ``False`` (goldfive#196):
        only the lifecycle reporting tools (``report_task_started`` /
        ``_progress`` / ``_completed`` / ``_failed`` / ``_blocked`` /
        ``_awaiting_approval`` / ``report_new_work_discovered``) are
        registered on the agent. The drift opinions —
        ``report_plan_divergence``, ``declare_task_skipped``,
        ``declare_task_not_needed`` — are NOT registered, so the
        prompt is smaller and the model can't hallucinate a drift
        call. The framework's observation paths
        (``classify_goal_drift``, :class:`PlanReconciler`, the
        steerer's refine machinery) remain the canonical detectors.
        Pass ``True`` to restore the full pre-#196 set, or a list of
        drift tool names to enable a subset.

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
    # Dynamic instruction resolver (goldfive#251). Default-on per user
    # review decision. Mutates each reachable LlmAgent's ``instruction``
    # field in-place so the LLM sees a plan-causal prompt every turn,
    # not just the baked-in string from construction time. Only
    # meaningful for ADK agents (LlmAgent has the ``instruction``
    # field); the installer is a no-op on other tree shapes.
    if is_adk_agent(agent):
        from goldfive.adapters.adk_llm_instrumentation import (
            install_dynamic_instructions,
            log_dynamic_instruction_opt_out,
        )

        if dynamic_instruction:
            touched = install_dynamic_instructions(agent)
            if touched:
                log.debug(
                    "goldfive.wrap: dynamic_instruction installed on %d agent(s)",
                    touched,
                )
        else:
            log_dynamic_instruction_opt_out(agent)

    # Resolve the runtime config (goldfive#225). When ``runtime`` is
    # not supplied we build one from the environment so pre-#225
    # callers — and the env-var tests they already ship — keep working
    # without modification. The resolved config is then installed
    # eagerly into the embedding backend's module-level state; the
    # reasoning-drift module is installed later inside ``DefaultSteerer``
    # when the steerer-default branch runs (so callers who pass their
    # own ``steerer=`` can make their own installation decisions).
    resolved_runtime: RuntimeConfig = runtime if runtime is not None else RuntimeConfig.from_env()
    from goldfive.drift import _embed as _embed_module
    from goldfive.drift import reasoning as _reasoning_module
    from goldfive.drift import tool_loops as _tool_loops_module

    _embed_module.configure(resolved_runtime.embedding)
    _reasoning_module.configure(resolved_runtime.reasoning_drift)
    _tool_loops_module.configure(resolved_runtime.tool_loops)

    # Resolve sinks first so the ContextEditor (built next) can emit
    # ``context_edited`` / ``context_edit_rejected`` events onto the
    # same fan-out the rest of the runtime uses. Default to
    # ``[LoggingSink()]`` mirroring the historical behaviour.
    resolved_sinks: list[EventSink] = list(sinks) if sinks is not None else [LoggingSink()]

    # Request-side ContextEditor (goldfive#397). Built ONLY when the
    # operator opted in via ``SteeringConfig.context_editor_rules``;
    # otherwise stays ``None`` and the ADK plugin's
    # ``before_model_callback`` short-circuits with one ``is None``
    # check (zero overhead for non-users — the contract goldfive#397
    # demands).
    from goldfive.context_editor import build_editor_from_config  # noqa: PLC0415

    context_editor = build_editor_from_config(
        resolved_runtime.steering.context_editor_rules,
        sinks=resolved_sinks,
    )

    # Thread :class:`AgentConfig` (goldfive#256) into the ADK adapter so
    # the plugin's structural ``max_output_tokens`` ceiling AND the
    # per-LLM-call wall-clock budget reflect the runtime config. Both
    # kwargs are ignored for non-ADK adapter shapes (the typed config
    # has no analogous surface on Claude / callable adapters today).
    # ``context_editor`` is forwarded too; non-ADK adapters ignore it.
    adapter = auto_adapter(
        agent,
        plugins=plugins,
        llm_call_timeout_ms=resolved_runtime.agent.call_timeout_ms,
        agent_max_output_tokens=resolved_runtime.agent.max_output_tokens,
        context_editor=context_editor,
    )

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
    # Test seam (goldfive#cleanup-monkeypatch): when a caller passes
    # ``llm_detector=...`` it replaces :func:`detect_llm` for the
    # duration of this call. Production code leaves it ``None``;
    # tests use it to script a ``(call_llm, model_name)`` pair without
    # rebinding the module-level ``detect_llm`` symbol.
    detector = llm_detector if llm_detector is not None else detect_llm
    if resolved_call_llm is None:
        detected = detector(agent)
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
    # Test seam: ``judge_call_llm_builder`` replaces
    # :func:`_build_judge_call_llm` for this call. Same pattern as
    # ``llm_detector`` above — leave ``None`` in production.
    judge_builder = (
        judge_call_llm_builder
        if judge_call_llm_builder is not None
        else _build_judge_call_llm
    )
    if call_llm is None and resolved_runtime.judge.base_url:
        built = judge_builder(resolved_runtime.judge)
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
            steering_config=resolved_runtime.steering,
        )
    else:
        resolved_steerer = DefaultSteerer(
            goal_drift_config=resolved_runtime.goal_drift,
            tool_loop_config=resolved_runtime.tool_loops,
            reasoning_drift_config=resolved_runtime.reasoning_drift,
            reasoning_drift_mode=resolved_runtime.reasoning_drift.mode,
            steering_config=resolved_runtime.steering,
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
    # Pluggable-judges installation (goldfive#437). Operators pass a
    # custom judge list via ``goldfive.wrap(judges=[...])``. When the
    # caller does not supply one, the goldfive default judge set is
    # installed so the new :class:`JudgementEmitted` envelope is
    # populated for every default-detector verdict alongside the
    # legacy :class:`DriftDetected` emit (back-compat preserved).
    # Passing an explicit empty list (``judges=[]``) is the opt-out
    # token — the steerer's hardcoded detector path still runs but no
    # ``JudgementEmitted`` envelopes ride the sink stream.
    #
    # ``disable_judges=`` drops named built-ins from the *default* set.
    # It is meaningless alongside an explicit ``judges=`` (which already
    # spells out the exact set), so the combination is rejected loudly
    # rather than silently ignoring one of the two.
    if judges is not None and disable_judges is not None:
        raise TypeError(
            "goldfive.wrap: pass either judges= (the explicit judge list) "
            "or disable_judges= (drop built-ins from the default set), not "
            "both — an explicit judges= list already specifies the exact set."
        )
    resolved_judges: list[Any]
    if judges is not None:
        resolved_judges = list(judges)
    else:
        from goldfive.builtin_judges import default_judges as _default_judges

        resolved_judges = _default_judges(disable=disable_judges)
    set_judges = getattr(resolved_steerer, "set_judges", None)
    if callable(set_judges):
        set_judges(resolved_judges)

    runner = Runner(
        agent=adapter,
        planner=resolved_planner,
        executor=resolved_executor,
        goal_deriver=resolved_goal_deriver,
        steerer=resolved_steerer,
        sinks=resolved_sinks,
        control=control,
        max_task_invocations=max_task_invocations,
        drift_self_reporting=drift_self_reporting,
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
