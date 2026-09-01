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

from goldfive._llm import make_default_openai_call_llm
from goldfive._llm_detect import CallLLM, detect_llm
from goldfive.adapters.auto import auto_adapter, is_adk_agent
from goldfive.config import JudgeConfig, RuntimeConfig
from goldfive.executors.sequential import SequentialExecutor
from goldfive.goal_deriver import LiteralGoalDeriver, LLMGoalDeriver
from goldfive.planner import LLMPlanner, PassthroughPlanner, StaticPlanner
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
from goldfive.types import Goal, Plan, Task

if TYPE_CHECKING:
    from goldfive.control import ControlChannel
    from goldfive.judges.builtins import BuiltinJudge

log = logging.getLogger("goldfive.wrap")


#: Stable task id for the synthetic single-task plan that judge-only mode
#: installs. Kept distinct so operators can recognise it in transcripts /
#: telemetry as the framing task the native run executed under.
_JUDGE_ONLY_TASK_ID = "judge_only_native_run"


def _build_judge_only_planner() -> StaticPlanner:
    """Return the planner that drives a NATIVE, un-steered agent run.

    Judge-only mode runs the wrapped agent natively while keeping the
    drift judges armed, and issues ZERO planning / steering LLM calls.
    The recipe (validated downstream): a :class:`StaticPlanner` carrying
    a single task.

    Why a one-task :class:`StaticPlanner` rather than
    :class:`PassthroughPlanner`:

    * :class:`PassthroughPlanner` returns ``None`` from ``generate`` /
      ``handle_turn``, so the Runner has no plan to execute and the run
      aborts with an EMPTY transcript — nothing for the judges to score.
    * :class:`StaticPlanner` returns a baked single-task plan. Under the
      overlay executor (``SequentialExecutor(overlay_mode=True)``, the
      ``wrap()`` default) that drives ONE ``invoke_passthrough`` of the
      native agent tree against the user's input — a real transcript is
      produced — and ``refine`` returns ``None`` so no refine / steer
      planning call ever fires.

    The single task's ``assignee_agent_id`` is intentionally left empty:
    overlay dispatch runs the native tree regardless of assignee, and
    the framework populates assignment observationally (goldfive#252).
    The task is pure framing so the Gantt / transcript has a node to
    hang native activity under.
    """
    return StaticPlanner(
        Plan(
            id="",
            run_id="",
            goal_ids=(),
            tasks=(
                Task(
                    id=_JUDGE_ONLY_TASK_ID,
                    title="Native agent run (judge-only)",
                    description=(
                        "Run the wrapped agent natively with no planning or "
                        "steering overlay; drift judges stay armed."
                    ),
                ),
            ),
            edges=(),
            summary="Native agent run (judge-only mode)",
        )
    )


def _build_judge_call_llm(config: JudgeConfig) -> tuple[CallLLM, str] | None:
    """Back-compat shim over :func:`goldfive._llm.make_default_openai_call_llm`.

    Kept because ``wrap(judge_call_llm_builder=...)`` documents this
    symbol as the default and external callers / tests import it here.
    """
    return make_default_openai_call_llm(config)


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
    judge_call_llm: CallLLM | None = None,
    judge_model: str | None = None,
    max_task_invocations: int | None = None,
    plugins: list[Any] | None = None,
    runtime: RuntimeConfig | None = None,
    dynamic_instruction: bool = True,
    drift_self_reporting: bool | list[str] = False,
    judge_only: bool = False,
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
    judge_call_llm:
        Optional async ``(system, user, model) -> str`` callable used
        only by the default goal-drift and reasoning-drift judges. It
        takes precedence over ``call_llm`` and ``runtime.judge``.
        ``wrap()`` does not register a judge close hook for a
        caller-supplied callable. An explicit ``steerer=`` retains full
        control and is not modified.
    judge_model:
        Optional model name passed only to the default goal-drift and
        reasoning-drift judges. When omitted, the judges use the model
        resolved from ``model``, ``runtime.judge``, or the agent tree,
        according to the callable source.
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
    judge_only:
        First-class JUDGE-ONLY mode. Default ``False`` — behaviour is
        byte-identical to today (the full planning overlay runs:
        goal-derivation, per-turn planning, refine, and drift-reactive
        steering).

        When ``True`` the wrapped agent runs NATIVELY while the drift
        judges stay armed, and NO planning / steering LLM call is ever
        issued (no goal-derive, no plan / refine, no drift-reactive
        steering). This is the mode an evaluation / benchmarking harness
        wants: judge an agent's own NATIVE behaviour without goldfive
        steering it.

        It is a convenience that sets the defaults for ``planner`` and
        ``goal_deriver``; it does NOT touch the judges (they remain wired
        through the same ``judge_call_llm`` / ``call_llm`` / detected
        tree LLM / ``JudgeConfig`` resolution used in full mode).
        Concretely, when the caller did not supply them:

        * ``planner`` defaults to a :class:`StaticPlanner` carrying a
          single framing task. Under the overlay executor that drives ONE
          native ``invoke_passthrough`` of the agent tree — a real
          transcript is produced — and its ``refine`` returns ``None`` so
          no refine / steer planning call fires. (A
          :class:`PassthroughPlanner` would instead return ``None`` from
          ``generate`` and ABORT the run with an empty transcript — the
          trap this mode exists to avoid.)
        * ``goal_deriver`` defaults to :class:`LiteralGoalDeriver`, which
          wraps the user input as a single goal WITHOUT an LLM call (vs
          the goal-derive LLM call :class:`LLMGoalDeriver` would make).

        An explicit ``planner=`` / ``goal_deriver=`` / ``steerer=`` still
        wins — ``judge_only`` only supplies the defaults. Note that
        ``SteeringConfig.observation_only`` does NOT achieve this: it
        gates only the three drift-reactive INJECTION points, while the
        planner's goal-derivation / per-turn planning / refine still run
        and burn LLM calls.

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

    # Judge routing (explicit judge override + JudgeConfig + named-model WARNING).
    # Precedence for the two drift judges' call_llm / model:
    #   1. Explicit ``goldfive.wrap(judge_call_llm=...)``.
    #   2. Explicit ``goldfive.wrap(call_llm=...)`` (shared route).
    #   3. ``resolved_runtime.judge.base_url`` — dedicated judge endpoint.
    #   4. Auto-detected tree LLM (``detect_llm``).
    # The planner + goal_deriver stay on ``resolved_call_llm`` regardless;
    # only the two judges use the explicit judge route or JudgeConfig.
    resolved_judge_call_llm: CallLLM | None = (
        judge_call_llm if judge_call_llm is not None else resolved_call_llm
    )
    resolved_judge_model: str = judge_model if judge_model is not None else resolved_model
    judge_from_config: bool = False
    # Test seam: ``judge_call_llm_builder`` replaces
    # :func:`_build_judge_call_llm` for this call. Same pattern as
    # ``llm_detector`` above — leave ``None`` in production.
    judge_builder = (
        judge_call_llm_builder if judge_call_llm_builder is not None else _build_judge_call_llm
    )
    if judge_call_llm is None and call_llm is None and resolved_runtime.judge.base_url:
        built = judge_builder(resolved_runtime.judge)
        if built is not None:
            resolved_judge_call_llm, judge_config_model = built
            if judge_model is None and judge_config_model:
                resolved_judge_model = judge_config_model
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
    # ``judge_call_llm=``, ``call_llm=``, or an explicit ``JudgeConfig``
    # suppresses the warning because the operator selected a route.
    if _call_llm_from_detect and judge_call_llm is None and not judge_from_config:
        # Prefer the agent's own ``.name`` ("coordinator_agent", "research_agent",
        # ...) over the Python class name ("LlmAgent", "BaseAgent", ...) — the
        # latter is nearly useless for identifying *which* agent in a tree the
        # LLM was detected from. Fall through to the class name only when the
        # object has no usable ``.name`` attribute (non-ADK shapes).
        _agent_label = getattr(agent, "name", "") or type(agent).__name__
        # Name the cost explicitly: background reasoning-judge calls
        # (up to ``max_concurrent_judges`` in flight, each with the
        # judge's output-token ceiling) land on the SAME endpoint the
        # agent tree bills against, competing with the tree's own
        # calls for capacity / rate limits.
        from goldfive.drift.reasoning_judge import REASONING_JUDGE_MAX_OUTPUT_TOKENS

        log.warning(
            "goldfive.wrap: judge LLM not explicitly configured; inheriting "
            "%r from agent %r (detected via ADK model attribute). Judge "
            "traffic (up to %d concurrent background calls, %d output "
            "tokens each) will share this endpoint with — and compete "
            "against — the agent tree's own calls. Set "
            "GOLDFIVE_JUDGE_BASE_URL / GOLDFIVE_JUDGE_MODEL to route "
            "goldfive's judges to a dedicated endpoint.",
            _detected_model_name,
            _agent_label,
            max(1, int(resolved_runtime.reasoning_drift.max_concurrent_judges)),
            REASONING_JUDGE_MAX_OUTPUT_TOKENS,
        )

    resolved_planner: Planner
    if planner is not None:
        resolved_planner = planner
    elif judge_only:
        # Judge-only mode: a one-task StaticPlanner drives a native
        # overlay run (real transcript) while issuing zero planning
        # LLM calls — no generate-time plan synthesis, and refine
        # returns None so no refine / steer call fires. See the
        # ``judge_only`` docstring and :func:`_build_judge_only_planner`.
        resolved_planner = _build_judge_only_planner()
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
    elif judge_only:
        # Judge-only mode: wrap the user input as a single goal without
        # an LLM call (LLMGoalDeriver would issue a goal-derive call).
        resolved_goal_deriver = LiteralGoalDeriver()
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
    elif resolved_judge_call_llm is not None:
        # The two drift judges share a single callable: the
        # trajectory-level GOAL_DRIFT judge (goldfive#218) and the
        # per-thinking-message reasoning judge (goldfive#226). Default
        # ``reasoning_drift_mode`` on the steerer is ``"judge"`` -- see
        # :class:`DefaultSteerer`. ``resolved_judge_call_llm`` comes from
        # the precedence chain above.
        resolved_steerer = DefaultSteerer(
            goal_drift_call_llm=resolved_judge_call_llm,
            goal_drift_model=resolved_judge_model,
            goal_drift_config=resolved_runtime.goal_drift,
            tool_loop_config=resolved_runtime.tool_loops,
            reasoning_drift_config=resolved_runtime.reasoning_drift,
            reasoning_drift_mode=resolved_runtime.reasoning_drift.mode,
            reasoning_drift_call_llm=resolved_judge_call_llm,
            reasoning_drift_model=resolved_judge_model,
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
    if judge_from_config and resolved_judge_call_llm is not None:
        _close = getattr(resolved_judge_call_llm, "close", None)
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
    judge_call_llm: CallLLM | None = None,
    judge_model: str | None = None,
    **wrap_kwargs: Any,
) -> ExecutionOutcome:
    """Wrap ``agent`` and run it against ``user_input`` in one call.

    Equivalent to::

        runner = goldfive.wrap(
            agent,
            judge_call_llm=judge_call_llm,
            judge_model=judge_model,
            **wrap_kwargs,
        )
        return await runner.run(user_input, context=context)

    ``judge_call_llm`` and ``judge_model`` are forwarded to
    :func:`wrap` as the dedicated built-in-judge route. All remaining
    keyword arguments are also forwarded. This function does not call
    :meth:`Runner.close`; callers that need deterministic sink teardown
    should build their own runner.
    """
    if judge_call_llm is not None:
        wrap_kwargs["judge_call_llm"] = judge_call_llm
    if judge_model is not None:
        wrap_kwargs["judge_model"] = judge_model
    runner = wrap(agent, **wrap_kwargs)
    return await runner.run(user_input, context=context)


__all__ = ["run", "wrap"]
