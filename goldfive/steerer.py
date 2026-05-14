"""The framework-agnostic :class:`DefaultSteerer`.

Port of harmonograf's ``_AdkState`` task state machine and drift detector,
with all ADK callback plumbing, ``session.state`` writes, and gRPC calls
stripped out. The steerer's job is:

* Mutate :class:`Session`'s plan task statuses in response to reporting
  tool calls (the ``mark_task_*`` family).
* Emit a corresponding proto ``Event`` to every bound ``EventSink`` for
  each transition, drift detection, and plan revision.
* Observe raw upstream events, classify drift, and — if the drift rises
  to ``WARNING`` or above — call ``Planner.refine`` and apply the new
  plan (emitting ``PlanRevised``).

The steerer holds no gRPC / client references and touches no adapter
internals. Executors call :meth:`DefaultSteerer.bind` to wire in the
sinks list and planner at run start.

Intervention ladder
-------------------
Drift handling routes through an explicit six-level ladder (goldfive#142)
so "when does goldfive interrupt the tree" is a single table, not a
tangle of conditionals. Levels, ordered by intrusiveness:

* Level 0 — OBSERVE: record the drift, no action.
* Level 1 — ABSORB: call ``planner.refine``; continue.
* Level 2 — NUDGE: queue a soft follow-up message on the session for
  the Runner's overlay loop to pick up at the next invocation boundary.
* Level 3 — CANCEL_REINVOKE: dispatch a ``GOLDFIVE_STEER`` control
  message on the bound channel so the executor cancels in-flight work
  and restarts with a goldfive-authored corrective. Phase 2 of the
  path-duality fix routes this through the same junction as USER_STEER.
* Level 4 — PAUSE_ESCALATE: dispatch a ``GOLDFIVE_PAUSE_ESCALATE``
  control message and emit ``HUMAN_INTERVENTION_REQUIRED`` so the
  executor's pre-task loop blocks waiting for operator action.
* Level 5 — TERMINATE: run-level abort (currently only reached when an
  unhandled Level 4 times out; actual termination is driven by the
  executor, not the steerer).

The mapping from (drift_kind, severity, occurrence_count) to level
lives in :meth:`DefaultSteerer._ladder_level_for`. Level dispatch is
handled by :meth:`DefaultSteerer._dispatch_ladder_level`, which wraps
the existing refine flow for Level 1 and short-circuits the other
levels. See goldfive#142 for the full table.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from goldfive.drift.reasoning import (
    DEFAULT_REASONING_DRIFT_MODE,
    ReasoningDriftMode,
)
from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskStatus,
)

if TYPE_CHECKING:
    from goldfive.config import (
        GoalDriftConfig,
        ReasoningDriftConfig,
        SteeringConfig,
        ToolLoopConfig,
    )
    from goldfive.protocols import EventSink, Planner

# Shape of the opt-in reflective LLM callable. Matches the signature of
# the ``call_llm`` used by ``LLMPlanner`` (``(system, user, model) ->
# str``) so operators can pass the same callable they already configure
# for planning.
ReflectiveCallLLM = Callable[[str, str, str], Awaitable[str]]

log = logging.getLogger(__name__)


__all__ = [
    "DefaultSteerer",
    "InterventionLevel",
    "RefineExhausted",
    "compose_corrective_user_message",
]


class RefineExhausted(Exception):
    """Sentinel raised by ``Planner.refine`` when no meaningful change is possible.

    goldfive#271 — drift-handler exhaustion as the escalation primitive.
    A refine handler that has tried and cannot produce a meaningful
    change for a drift may raise this exception to signal "this drift
    is unresolvable" up to the steerer. The steerer catches it and
    emits ``HUMAN_INTERVENTION_REQUIRED`` for the originating drift,
    pausing the runner for operator action.

    Most planners do NOT need to raise this explicitly — the steerer's
    structural no-op revision check (in
    :meth:`DefaultSteerer._plans_structurally_identical`) catches the
    common case where a refine returns a plan with no real change.
    This sentinel is for planners that can detect exhaustion ahead of
    producing a plan (e.g., the LLM explicitly responded "I cannot
    refine this further").
    """


class InterventionLevel(enum.IntEnum):
    """The graduated intervention ladder (goldfive#142).

    Ordered by intrusiveness. Every drift handled by
    :class:`DefaultSteerer` maps to exactly one level via
    :meth:`DefaultSteerer._ladder_level_for`; the level dictates what
    the steerer does in response.
    """

    OBSERVE = 0
    ABSORB = 1
    NUDGE = 2
    CANCEL_REINVOKE = 3
    PAUSE_ESCALATE = 4
    TERMINATE = 5


# goldfive#202: drift kinds for which a successful ABSORB (Level 1
# refine) also queues a Level 2 nudge onto ``session.pending_nudges``.
# The executor's overlay loop consumes the nudge at invocation end and
# re-invokes the passthrough with a synthesized user message describing
# the plan revision — the only way for a coordinator that is still
# mid-invocation (retrying the superseded task) to learn its plan
# changed. Scoped to "coordinator-stuck" shapes; other ABSORB kinds
# recover at the next task boundary or via Level 3 CANCEL_REINVOKE.
_ABSORB_NUDGE_KINDS: frozenset[DriftKind] = frozenset(
    {
        DriftKind.LOOPING_REASONING,
        DriftKind.LOOPING_TOOL_CALL,
        DriftKind.SELF_REPORTED_STUCK,
        # Tier 1 / F4 (loop prevention): GOAL_DRIFT now lives on the
        # NUDGE path. The judge's signal is "agent stuck on completed
        # work" — exactly the shape a corrective user-message can fix
        # without refining the plan (the plan is correct; only the
        # agent's next-action reasoning is stuck). When ABSORB fires
        # (WARNING severity), queueing a nudge gives the overlay the
        # mid-invocation rescue path; CRITICAL routes through NUDGE
        # directly via the ladder table below.
        DriftKind.GOAL_DRIFT,
    }
)


# Default drift messages per kind, used by
# :func:`compose_corrective_user_message` when the drift carries no
# kind-specific override. Keep SHORT, action-focused; no goldfive jargon
# ("synthetic", "healed", "orphan", "drift") in user-facing copy.
_CORRECTIVE_TEMPLATES: dict[DriftKind, str] = {
    DriftKind.LOOPING_REASONING: (
        "The prior attempt looped on {current_task_id}. "
        "Refined plan: {next_task_title}. Please try a different approach."
    ),
    DriftKind.LOOPING_TOOL_CALL: (
        "The prior attempt kept retrying the same tool call on "
        "{current_task_id} without progress. Refined plan: "
        "{next_task_title}. Please try a different approach."
    ),
    DriftKind.PLAN_DIVERGENCE: (
        "The tree's prior activity diverged from the plan. "
        "Refined plan: proceed with {next_task_title}."
    ),
    DriftKind.AGENT_REFUSAL: (
        "The prior attempt could not complete {current_task_id}. "
        "Refined plan: try {next_task_title}."
    ),
    DriftKind.MODEL_REFUSAL: (
        "The model declined to proceed on {current_task_id}. Refined plan: try {next_task_title}."
    ),
    DriftKind.INTENT_DIVERGENCE: (
        "The prior attempt strayed from the stated intent for "
        "{current_task_id}. Refined plan: proceed with "
        "{next_task_title}."
    ),
    DriftKind.TOOL_ERROR: (
        "The prior attempt hit a tool error on {current_task_id}. "
        "Refined plan: proceed with {next_task_title}."
    ),
    DriftKind.RUNAWAY_DELEGATION: (
        "The prior attempt kept delegating without finishing "
        "{current_task_id}. Refined plan: proceed with "
        "{next_task_title} directly."
    ),
    DriftKind.SELF_REPORTED_STUCK: (
        "The prior attempt reported being stuck on {current_task_id}. "
        "Refined plan: try {next_task_title}."
    ),
    DriftKind.CONFABULATION_RISK: (
        "The prior attempt may have produced {current_task_id} "
        "without consulting external data. Refined plan: "
        "{next_task_title}."
    ),
    # Tier 1 / F4 — GOAL_DRIFT corrective. The judge's signal is "agent
    # is grinding on completed work / not advancing the goal". The plan
    # itself is fine; the agent just needs a pointer at the next hand-
    # off. Includes ``{next_task_agent}`` so the coordinator can route
    # to the assignee directly rather than re-invoking the stuck agent.
    DriftKind.GOAL_DRIFT: (
        "Task '{current_task_id}' is already complete. "
        "Please proceed to '{next_task_title}' via {next_task_agent}."
    ),
}


def compose_corrective_user_message(
    *,
    drift: DriftEvent,
    refined_plan: Plan | None,
) -> str:
    """Build a short directive user message for Level 3 re-invoke.

    Shape varies by drift kind (see :data:`_CORRECTIVE_TEMPLATES`). The
    message is deliberately short, action-focused, and avoids goldfive
    jargon -- the consumer is the agent's LLM, which should read a
    natural instruction rather than a framework postmortem.
    """
    current = drift.current_task_id or "the current task"
    next_title = _next_pending_task_title(refined_plan) or "the next planned step"
    next_agent = _next_pending_task_agent(refined_plan) or "the next assigned agent"
    template = _CORRECTIVE_TEMPLATES.get(drift.kind)
    if template is None:
        # Generic fallback for drift kinds that didn't get a
        # custom shape. Keep it tight and action-focused.
        template = (
            "The prior attempt on {current_task_id} did not complete "
            "successfully. Refined plan: proceed with {next_task_title}."
        )
    return template.format(
        current_task_id=current,
        next_task_title=next_title,
        next_task_agent=next_agent,
    )


def _next_pending_task_title(plan: Plan | None) -> str:
    """Return the title of the next PENDING task in topological order.

    Falls back to the task id if no title is set. Returns an empty
    string when there is no eligible task.
    """
    if plan is None:
        return ""
    for t in plan.tasks:
        if t.status is TaskStatus.PENDING:
            return (t.title or t.id or "").strip()
    return ""


def _next_pending_task_agent(plan: Plan | None) -> str:
    """Return the bare agent name assigned to the next PENDING task.

    Tier 1 / F4 — used by the GOAL_DRIFT corrective template, which
    redirects the LLM's next action to a different agent. Returns the
    last dot-separated segment of ``assignee_agent_id`` so the LLM sees
    a name it can pass back as the AgentTool target. Empty string when
    no eligible task exists or the task has no assignee.
    """
    if plan is None:
        return ""
    for t in plan.tasks:
        if t.status is TaskStatus.PENDING:
            assignee = (getattr(t, "assignee_agent_id", "") or "").strip()
            if "." in assignee:
                return assignee.rsplit(".", 1)[-1]
            return assignee
    return ""


# Task statuses that are terminal (no further transitions allowed).
# Re-exported from :mod:`goldfive.types` — the canonical definition —
# under the historical private name so existing imports in this module
# keep working. Do NOT redefine; new consumers should import
# ``TERMINAL_TASK_STATUSES`` from ``goldfive.types`` directly.
_TERMINAL_TASK_STATUSES = TERMINAL_TASK_STATUSES


# Ordered severities so we can compare with ``>=``.
_SEVERITY_ORDER: dict[DriftSeverity, int] = {
    DriftSeverity.INFO: 0,
    DriftSeverity.WARNING: 1,
    DriftSeverity.CRITICAL: 2,
}


def _severity_ge(a: DriftSeverity, b: DriftSeverity) -> bool:
    return _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b]


def _planner_refine_accepts_available_agents(planner: Any) -> bool:
    """Return True if ``planner.refine`` accepts ``available_agents=``.

    The #151 registry kwarg is additive — the main goldfive planners
    (``LLMPlanner``, ``PassthroughPlanner``, ``StaticPlanner``) all
    accept it, but user-supplied / test-stub planners that predate
    #151 have a refine signature without the kwarg. We probe the
    signature once per drift and fall back to the legacy kwarg-set
    when the kwarg would raise ``TypeError: unexpected keyword
    argument``. Planners declared with ``**kwargs`` are assumed to
    accept (the kwarg passes through).
    """

    refine = getattr(planner, "refine", None)
    if refine is None:
        return False
    try:
        sig = inspect.signature(refine)
    except (TypeError, ValueError):
        # Unintrospectable callable — safest to not pass the kwarg.
        return False
    params = sig.parameters
    if "available_agents" in params:
        return True
    for p in params.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


# ---------------------------------------------------------------------------
# DefaultSteerer
# ---------------------------------------------------------------------------


class DefaultSteerer:
    """The canonical :class:`Steerer` implementation.

    Bind via :meth:`bind`, then call the ``mark_task_*`` family (usually
    from reporting-tool handlers) to drive task transitions. The executor
    calls :meth:`observe` for each raw upstream event to let the steerer
    classify drift and (optionally) trigger a plan refine.
    """

    def __init__(
        self,
        *,
        reflective_check_interval: int = 15,
        reflective_call_llm: ReflectiveCallLLM | None = None,
        reflective_model: str = "",
        goal_drift_check_interval: int | None = None,
        goal_drift_call_llm: ReflectiveCallLLM | None = None,
        goal_drift_model: str = "",
        goal_drift_activity_window: int | None = None,
        goal_drift_config: GoalDriftConfig | None = None,
        tool_loop_config: ToolLoopConfig | None = None,
        reasoning_drift_config: ReasoningDriftConfig | None = None,
        reasoning_drift_mode: ReasoningDriftMode = DEFAULT_REASONING_DRIFT_MODE,
        reasoning_drift_call_llm: ReflectiveCallLLM | None = None,
        reasoning_drift_model: str = "",
        reasoning_drift_rate_limit: int = 3,
        reasoning_binding_confidence_threshold: float = 0.7,
        steering_config: SteeringConfig | None = None,
        goldfive_steer_threshold: str | None = None,
        goldfive_steer_suppression_window_turns: int | None = None,
    ) -> None:
        """Build a steerer.

        Parameters
        ----------
        reflective_check_interval:
            Number of LLM invocations (as reported via
            :meth:`note_llm_call`) between reflective self-progress
            checks. Defaults to ``15``. Ignored when
            ``reflective_call_llm`` is ``None``.
        reflective_call_llm:
            Optional async callable ``(system_prompt, user_prompt,
            model) -> str`` used to ask the agent "are you making
            progress?" once the counter reaches the configured
            interval. The whole feature is **off by default** — pass a
            callable to opt in. Operators who don't want the extra LLM
            cost never trigger it. The shape deliberately matches
            :class:`~goldfive.planner.LLMPlanner` so the same callable
            can be reused.
        reflective_model:
            Model name forwarded to ``reflective_call_llm``. Empty
            string is permitted; the callable may substitute its own
            default.
        goal_drift_check_interval:
            Number of agent-invocation turns (as reported via
            :meth:`note_agent_turn`) between trajectory-level
            GOAL_DRIFT checks. Defaults to ``5`` when
            ``goal_drift_config`` is also ``None``. Ignored when
            ``goal_drift_call_llm`` is ``None``.
        goal_drift_call_llm:
            Optional async callable ``(system_prompt, user_prompt,
            model) -> str`` used by
            :meth:`maybe_run_goal_drift_check` to ask an LLM-judge
            whether the tree's recent activity is progressing toward
            ``session.goals``. Opt-in: pass a callable (typically
            ``Runner`` wires its planner LLM here when
            ``goal_drift_enabled`` is on). Shape matches
            ``reflective_call_llm`` so the same callable can be
            reused.
        goal_drift_model:
            Model name forwarded to ``goal_drift_call_llm``.
        goal_drift_activity_window:
            Number of recent-activity entries retained on
            ``session.recent_agent_activity`` and passed to the
            judge. Bounds the prompt size; defaults to ``10`` when
            ``goal_drift_config`` is also ``None``.
        goal_drift_config:
            Optional :class:`~goldfive.config.GoalDriftConfig` (see
            goldfive#225). When provided, its fields supply fallback
            defaults for ``goal_drift_check_interval`` and
            ``goal_drift_activity_window``. **Precedence**: an
            explicit individual kwarg (``goal_drift_check_interval=``
            or ``goal_drift_activity_window=``) wins over the config
            dataclass; the config wins over the module-level
            built-in. This matches :func:`goldfive.wrap`'s contract
            where operators can pass a fully-typed ``RuntimeConfig``
            at the top level and still selectively override a single
            knob on a per-steerer basis.
        tool_loop_config:
            Optional :class:`~goldfive.config.ToolLoopConfig`. Stored
            on the steerer so callers (the ADK plugin, custom
            adapters) can pull thresholds via
            :meth:`get_tool_loop_config` instead of re-reading env
            vars. The steerer itself does NOT instantiate a
            :class:`~goldfive.drift.tool_loops.ToolLoopTracker` — the
            plugin still owns that — but exposing the config here
            keeps the four typed knobs co-located on a single
            component. Precedence: the config field is advisory; the
            plugin is free to honour or ignore it. Added in #225.
        reasoning_drift_config:
            Optional :class:`~goldfive.config.ReasoningDriftConfig`.
            When provided, :meth:`observe_reasoning` installs it via
            :func:`goldfive.drift.reasoning.configure` so the
            detector thresholds pick it up. Process-wide (see the
            module docstring on :mod:`goldfive.drift.reasoning` for
            the multi-Runner caveat). Added in #225.
        reasoning_drift_mode:
            Pipeline selection for :meth:`observe_reasoning`.
            ``"judge"`` (default) runs the LLM judge (goldfive#226);
            ``"embedding"`` runs the legacy embedding pipeline;
            ``"both"`` runs both with higher-severity-wins reconciliation;
            ``"off"`` disables off-topic detection (the always-on loop
            detector continues to run).
        reasoning_drift_call_llm:
            Optional async ``(system_prompt, user_prompt, model) -> str``
            callable used by the LLM-as-a-judge reasoning-drift detector.
            Required for ``reasoning_drift_mode`` in ``"judge"`` / ``"both"``
            to fire; silently no-ops when ``None`` so tests without a
            live LLM do not crash. Shape matches ``goal_drift_call_llm``
            so the same callable can be reused.
        reasoning_drift_model:
            Model name forwarded to ``reasoning_drift_call_llm``.
        reasoning_drift_rate_limit:
            Run the judge once every N thinking messages per task.
            ``N=1`` fires on every thinking message; ``N=3`` (the
            default) fires on the 1st, 4th, 7th ... per task. The
            first thinking message of every task always gets a judge
            call; counters reset on task transition.

        See ``docs/design/DRIFT.md`` §"Reflective self-progress check"
        for the full feature-gate semantics. The GOAL_DRIFT check
        follows the same opt-in contract (see goldfive#143).
        """
        self._sinks: list[Any] = []
        self._planner: Any | None = None
        # Optional adapter back-reference. Wired via :meth:`bind_adapter`
        # so the steerer can tag the next mid-invocation cancel with a
        # symbolic reason (``user_steer``) before the executor triggers
        # ``task.cancel()`` on the invoke task. See goldfive#139 and
        # :attr:`goldfive.adapters.adk.ADKAdapter._next_cancel_reason`.
        self._adapter: Any | None = None
        # Optional control-channel back-reference (Phase 2 of the path-
        # duality fix). Wired via :meth:`bind_control_channel` so the
        # steerer can mint ``GOLDFIVE_STEER`` and
        # ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessages onto the same
        # channel that user-issued PAUSE / RESUME / CANCEL / STEER ride.
        # ``None`` outside a bound run; dispatches are best-effort
        # no-ops when unbound (the originating drift event on the sink
        # stream remains the durable signal). Cleared at run boundary
        # by the Runner.
        self._control_channel: Any | None = None
        # Per-session, per-kind last-refine bookkeeping. Purely advisory:
        # callers can subclass to throttle on top of this if needed.
        self._last_refine_kind: dict[tuple[str, DriftKind], int] = {}
        # Per-async-task scratchpad the steerer uses to plumb the active
        # session into the planner's drift-emitter and span-context
        # callbacks. Set just before calling ``planner.refine`` /
        # ``planner.synthesize_goal_from_steer`` and cleared afterwards;
        # ``None`` outside that window. Only consulted by the emitter the
        # planner calls when its retry budget is spent (goldfive#133) and
        # by ``_span_context_for_planner``.
        #
        # ContextVar (not a plain attribute) so that two concurrent
        # ``runner.run(...)`` calls sharing one Steerer (and therefore
        # one Planner) do not stomp each other's session pointer.
        # Without this isolation, session A can refine while session B
        # is mid-refine, B's value overwrites A's, and A's planner-side
        # span / drift callbacks then resolve to B's run_id -- the
        # leak observed empirically across v12 -> v14 sessions in
        # demo-v14.log. Per-instance ContextVar (not module-level) so
        # parallel test cases instantiating their own Steerer never
        # collide. See PR #294 audit + the regression test in
        # ``tests/test_steerer_concurrent_sessions.py``.
        self._active_session_var: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
            f"goldfive_active_session_{id(self)}", default=None
        )
        # Reflective check wiring. When ``_reflective_call_llm`` is None
        # every entry point short-circuits so the feature is inert.
        self._reflective_call_llm: ReflectiveCallLLM | None = reflective_call_llm
        self._reflective_check_interval = max(1, int(reflective_check_interval))
        self._reflective_model = reflective_model
        # GOAL_DRIFT (goldfive#143) wiring. Same opt-in contract as the
        # reflective check: feature is inert unless a callable is passed.
        #
        # Precedence resolution (goldfive#225): an explicit individual
        # kwarg wins over the config dataclass which wins over the
        # module-level default. ``None`` on the individual kwarg means
        # "not explicitly set", which is the trigger to fall through to
        # the config / default.
        self._goal_drift_call_llm: ReflectiveCallLLM | None = goal_drift_call_llm
        if goal_drift_check_interval is not None:
            _check_interval = goal_drift_check_interval
        elif goal_drift_config is not None:
            _check_interval = goal_drift_config.check_interval
        else:
            _check_interval = 5
        self._goal_drift_check_interval = max(1, int(_check_interval))
        self._goal_drift_model = goal_drift_model
        if goal_drift_activity_window is not None:
            _activity_window = goal_drift_activity_window
        elif goal_drift_config is not None:
            _activity_window = goal_drift_config.activity_window
        else:
            _activity_window = 10
        self._goal_drift_activity_window = max(1, int(_activity_window))
        # Typed-config stash (goldfive#225). Retained as-is so
        # downstream consumers (plugins, tests) can introspect the
        # effective configuration without reconstructing it from the
        # scalar fields above.
        self._goal_drift_config: GoalDriftConfig | None = goal_drift_config
        self._tool_loop_config: ToolLoopConfig | None = tool_loop_config
        self._reasoning_drift_config: ReasoningDriftConfig | None = reasoning_drift_config
        # Install the reasoning-drift thresholds eagerly so any
        # ``observe_reasoning`` call on any session sees the
        # Runner-scoped config. Process-wide — see the installation-
        # site docstring on :mod:`goldfive.drift.reasoning`.
        if reasoning_drift_config is not None:
            from goldfive.drift import reasoning as _reasoning

            _reasoning.configure(reasoning_drift_config)
        # Per-thinking-message reasoning-drift judge wiring (goldfive#226).
        # Mode selects which detectors run in :meth:`observe_reasoning`;
        # the judge callable / model are forwarded to the LLM-as-a-judge
        # path. ``None`` callable silently no-ops the judge (tests without
        # a live LLM stay green).
        self._reasoning_drift_mode: ReasoningDriftMode = reasoning_drift_mode
        self._reasoning_drift_call_llm: ReflectiveCallLLM | None = reasoning_drift_call_llm
        self._reasoning_drift_model = reasoning_drift_model
        self._reasoning_drift_rate_limit = max(1, int(reasoning_drift_rate_limit))
        # Phase 1 of goldfive#271 — confidence threshold for recording a
        # reasoning-extracted binding onto StateStore. The
        # judge returns a focus_confidence in [0.0, 1.0]; bindings with
        # confidence >= this threshold are stamped, lower-confidence
        # ones are discarded so the pin-resolution ladder doesn't
        # consume noisy attributions. Clamped to [0.0, 1.0]. Defaults
        # to 0.7 — empirically the band where the judge's
        # plan-task attribution stops being a guess.
        try:
            _conf_threshold = float(reasoning_binding_confidence_threshold)
        except (TypeError, ValueError):
            _conf_threshold = 0.7
        self._reasoning_binding_confidence_threshold: float = max(0.0, min(1.0, _conf_threshold))
        # goldfive-steer-unification: policy knob + freshness window for
        # promoting goldfive-detected drifts into the USER_STEER-style
        # cancel+refine+restart machinery. Precedence mirrors the other
        # goldfive#225 knobs: explicit individual kwarg wins over the
        # ``SteeringConfig`` dataclass, which wins over the built-in
        # defaults.
        if goldfive_steer_threshold is not None:
            _threshold = str(goldfive_steer_threshold).strip().lower()
        elif steering_config is not None:
            _threshold = str(steering_config.threshold).strip().lower()
        else:
            _threshold = "warning"
        if _threshold not in {"off", "warning", "critical"}:
            log.warning(
                "DefaultSteerer: unknown goldfive_steer_threshold=%r; falling back to 'warning'",
                _threshold,
            )
            _threshold = "warning"
        self._goldfive_steer_threshold: str = _threshold
        if goldfive_steer_suppression_window_turns is not None:
            _window = int(goldfive_steer_suppression_window_turns)
        elif steering_config is not None:
            _window = int(steering_config.suppression_window_turns)
        else:
            _window = 3
        self._goldfive_steer_suppression_window_turns = max(0, _window)
        self._steering_config: SteeringConfig | None = steering_config
        # goldfive#254: observation-only mode. Detection still runs in
        # full and ``planner.refine_steer`` still runs (operators see the
        # would-have-applied plan via ``PlanRevised`` with
        # ``dry_run=True``), but the three actual injection points
        # (``session.plan`` mutation in ``_apply_revision``, the
        # ``GOLDFIVE_STEER`` ControlMessage enqueue, the
        # ``request_invocation_cancel`` plugin call) are gated by
        # :meth:`_should_inject`. The flag lives on :class:`SteeringConfig`
        # — NOT a constructor parameter on this class — so operators
        # set it via ``RuntimeConfig(steering=SteeringConfig(...))``
        # at :func:`goldfive.wrap` time. Default for the bare
        # ``DefaultSteerer()`` constructor (no config) is the safer
        # passive observation (``True``) — matches the production
        # default on :class:`SteeringConfig` and avoids surprising
        # third-party callers who construct ``DefaultSteerer`` directly.
        if steering_config is not None:
            self._observation_only: bool = bool(steering_config.observation_only)
        else:
            # Honour the test-only override hook so the autouse fixture
            # in ``tests/conftest.py`` can flip the implicit default for
            # the entire test suite without touching every call site.
            from goldfive.config import _resolve_observation_only_default

            self._observation_only = _resolve_observation_only_default()
        # Background reasoning-judge tasks (goldfive#251). The LLM-judge
        # path in :meth:`observe_reasoning` is fire-and-forget so the
        # adapter's model-response callback can return immediately and
        # ADK can dispatch tool calls without waiting on a minute-long
        # local-llama judge round-trip. This set holds the live
        # ``asyncio.Task`` handles so :meth:`shutdown` can drain them
        # at run end. Tasks auto-discard themselves via
        # ``add_done_callback(self._background_judges.discard)``.
        self._background_judges: set[asyncio.Task[Any]] = set()
        # Background drift-handler tasks (iter-11A). The drift cascade
        # triggered from ``mark_task_failed`` / ``mark_task_blocked``
        # awaits ``planner.refine`` (an LLM round-trip) plus its
        # cancellation / supersedes side effects. Awaiting that inline
        # from a reporting-tool call site (e.g. ``report_task_failed``)
        # blocked the tool from returning for ~minutes on slow local
        # LLMs, which in turn blocked the agent's next ADK turn. We
        # now spawn the cascade fire-and-forget through this set,
        # mirroring :attr:`_background_judges`. ``shutdown`` drains
        # both sets symmetrically.
        self._background_drifts: set[asyncio.Task[None]] = set()
        # Per-session plan-state mutation lock. Held only across the
        # consistency-critical region of ``_emit_plan_revised`` (revision
        # index bump + supersedes integration + correction GC + repin +
        # PlanRevised emit). NOT held across ``planner.refine`` itself —
        # that would serialise concurrent refines and defeat the
        # fire-and-forget judge path from #254. Reports that must observe
        # a consistent plan call :meth:`_wait_plan_stable` to acquire +
        # immediately release the lock. Keyed by ``session.id``.
        self._plan_locks: dict[str, asyncio.Lock] = {}
        # Component construction. The router holds shared mutable state
        # (sinks, planner, adapter, control_channel, ContextVar,
        # background-task sets, plan locks, observation_only flag,
        # config) and each component takes a back-reference to the
        # router so cross-component calls go through one indirection.
        # Order matters only insofar as a component constructed first
        # cannot reference one constructed later inside its own
        # ``__init__`` — none of them do today; every cross-call lands
        # at runtime.
        from goldfive.drift_observer import DriftObserver
        from goldfive.plan_reviser import PlanReviser
        from goldfive.task_state_machine import TaskStateMachine

        self._task_state_machine: TaskStateMachine = TaskStateMachine(self)
        self._plan_reviser: PlanReviser = PlanReviser(self)
        self._drift_observer: DriftObserver = DriftObserver(self)

    # ------------------------------------------------------------------
    # Protocol-required: wiring
    # ------------------------------------------------------------------

    def bind(self, *, sinks: list[EventSink], planner: Planner) -> None:
        self._sinks = list(sinks)
        self._planner = planner
        # If the planner supports the optional drift-emitter hook (see
        # :class:`~goldfive.planner.LLMPlanner.set_drift_emitter`), wire
        # it up now so the planner can signal ``REFINE_VALIDATION_FAILED``
        # through the normal event pipeline when its retry budget is
        # spent. Duck-typed on purpose so custom ``Planner``
        # implementations don't have to implement the hook.
        setter = getattr(planner, "set_drift_emitter", None)
        if callable(setter):
            setter(self._emit_planner_refine_validation_failed)
        # Wire the span-context provider so every planner-internal
        # ``call_llm`` site emits ``GoldfiveLLMCallStart/End`` pairs onto
        # the sink bus and shows up as a proper span on harmonograf's
        # Gantt. Duck-typed — pre-spans Planner implementations simply
        # skip this hook.
        span_setter = getattr(planner, "set_span_context_provider", None)
        if callable(span_setter):
            span_setter(self._span_context_for_planner)

    def bind_adapter(self, adapter: Any) -> None:
        """Attach the active adapter for cancel-reason tagging.

        Optional wiring (goldfive#139). When set, the steerer tags the
        adapter's next mid-invocation cancel with a symbolic reason
        (``user_steer``) so the synthetic ``function_response`` event
        the adapter appends on cancel carries LLM-actionable content
        instead of the legacy goldfive-internal jargon. Adapters that
        don't expose ``_next_cancel_reason`` are tolerated: the setter
        is a no-op and the cancel falls through to the generic content
        variant.
        """
        self._adapter = adapter

    def bind_control_channel(self, channel: Any | None) -> None:
        """Attach (or detach, on ``None``) the active control channel.

        Phase 2 of the path-duality fix. The steerer mints
        ``GOLDFIVE_STEER`` and ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessages
        onto this channel so goldfive-authored drift rides the same
        cancel-and-restart junction as user-authored ``STEER`` /
        ``PAUSE``. Dispatches are best-effort: when no channel is
        bound (or ``channel is None`` was passed at run end), the
        steerer falls back to the originating drift event on the sink
        stream as the durable signal — no exception, no wedge.

        Wired by ``Runner.run`` immediately after :meth:`bind` and
        unwired (passed ``None``) at the run boundary so a Steerer
        shared across runs cannot leak a stale channel into a later
        unrelated run.
        """
        self._control_channel = channel

    async def _dispatch_goldfive_control(
        self, msg: Any
    ) -> bool:
        """Send a goldfive-internal ControlMessage on the bound channel.

        Returns ``True`` when the dispatch landed on a channel,
        ``False`` when no channel is bound or the send raised. The
        send raise path is logged at DEBUG and swallowed: the
        originating drift event on the sink stream remains the
        durable signal regardless of channel state.
        """
        channel = self._control_channel
        if channel is None:
            return False
        try:
            await channel.send(msg)
        except Exception as exc:  # noqa: BLE001 — best-effort dispatch
            log.debug(
                "DefaultSteerer._dispatch_goldfive_control: "
                "channel.send raised (kind=%s, swallowed): %s",
                getattr(getattr(msg, "kind", None), "value", "?"),
                exc,
            )
            return False
        return True

    def get_tool_loop_config(self) -> ToolLoopConfig | None:
        """Return the :class:`~goldfive.config.ToolLoopConfig` stashed at init, if any.

        Plugins (the ADK plugin) call this when constructing a
        :class:`~goldfive.drift.tool_loops.ToolLoopTracker` so the
        tracker's thresholds come from the :class:`RuntimeConfig`
        threaded through :func:`goldfive.wrap` instead of from env
        vars. Returns ``None`` when the steerer was built without a
        config, in which case callers should fall back to
        :func:`~goldfive.drift.tool_loops.load_thresholds_from_env`.
        """
        return self._tool_loop_config

    def _span_context_for_planner(self) -> Any | None:
        """Snapshot the currently-active session into span-emission context.

        Returns the ``(sinks, run_id, session_id, task_id, sequence_fn)``
        tuple that :func:`goldfive._llm_span.goldfive_llm_span` expects,
        or ``None`` when no session is in scope (e.g. tests that
        exercise the planner standalone).

        The steerer plumbs ``self._active_session_var`` just before calling
        ``planner.refine`` / ``planner.refine_steer`` /
        ``planner.synthesize_goal_from_steer`` so every LLM call site
        inside those methods has a valid session to stamp onto its
        spans. When called outside that window (the ContextVar is
        ``None``), spans are no-ops. The ContextVar is per-async-task,
        so concurrent ``runner.run`` calls sharing one Steerer keep
        their session pointers isolated (PR #294 / regression test
        ``tests/test_steerer_concurrent_sessions.py``).
        """
        session = self._active_session_var.get()
        if session is None:
            return None
        return (
            list(self._sinks),
            session.run_id,
            session.id,
            session.current_task_id,
            session.next_sequence,
        )

    async def _emit_planner_refine_validation_failed(self, drift: DriftEvent) -> None:
        """Emit a planner-side drift through the DriftDetected pipeline.

        Dispatched as the drift emitter passed to
        :meth:`~goldfive.planner.LLMPlanner.set_drift_emitter` in
        :meth:`bind`. The planner calls this when it exhausts its
        refine retry budget with a ``REFINE_VALIDATION_FAILED``
        ``DriftEvent`` pre-built; we emit it but deliberately do NOT go
        back through :meth:`_handle_drift` (the steerer must not try to
        refine again on this kind -- infinite-loop risk).
        """
        # The planner's emitter isn't bound to a specific session, so
        # we route through the per-async-task ContextVar set by the
        # caller (``_handle_drift`` / ``_promote_drift_to_steer`` /
        # ``observe_refine``). ContextVar isolation ensures concurrent
        # runs sharing one Steerer route their planner-emitted drifts
        # to the correct session.
        session = self._active_session_var.get()
        if session is None:
            log.warning(
                "DefaultSteerer: planner emitted %s but no active session bound; dropping signal",
                drift.kind.value,
            )
            return
        await self._emit_drift_detected(session, drift)

    # ------------------------------------------------------------------
    # Protocol-required: transition (generic)
    # ------------------------------------------------------------------

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
        cancel_reason: str = "",
        source: str = "other",
    ) -> None:
        """Generic transition entry point.

        Dispatches to the corresponding ``mark_task_*`` method. Unknown
        target statuses are a no-op (we don't invent new transitions).

        ``cancel_reason`` (goldfive#205): structured reason string stamped
        on the emitted ``TaskCancelled`` / ``TaskFailed`` envelope when the
        transition is to CANCELLED or FAILED. Takes precedence over
        ``detail`` for reason-field population. Conventional formats
        recognised by harmonograf's Trajectory view:

        * ``upstream_failed:<upstream_task_id>`` — cascade from a failed
          or cancelled ancestor.
        * ``superseded_by_revision:<replacement_task_id>`` — refine
          replaced this task with a new one.
        * ``run_aborted:<abort_reason>`` — fail_fast / validation / budget.
        * ``user_cancel:<annotation_id>`` — user-initiated CANCEL control.
        * ``adk_cancellation:<invocation_id>`` — ADK mid-invocation cancel.
        * ``steerer_policy:<drift_kind>`` — steerer-imposed cancel via
          intervention ladder.

        Sinks that don't know the format still surface the raw string,
        so it's safe to evolve the vocabulary without a proto change.

        Passing ``cancel_reason`` for non-terminal transitions is a
        no-op; the value is only consulted when ``to`` is CANCELLED or
        FAILED.
        """
        if to is TaskStatus.RUNNING:
            await self.mark_task_running(task_id, session=session, detail=detail, source=source)
        elif to is TaskStatus.COMPLETED:
            await self.mark_task_completed(task_id, summary=detail, session=session, source=source)
        elif to is TaskStatus.FAILED:
            reason = cancel_reason or detail
            await self.mark_task_failed(task_id, reason=reason, session=session, source=source)
        elif to is TaskStatus.BLOCKED:
            await self.mark_task_blocked(task_id, blocker=detail, session=session, source=source)
        elif to is TaskStatus.CANCELLED:
            reason = cancel_reason or detail
            await self.mark_task_cancelled(task_id, reason=reason, session=session, source=source)
        elif to is TaskStatus.NOT_NEEDED:
            await self.mark_task_not_needed(task_id, reason=detail, session=session, source=source)
        # PENDING and UNSPECIFIED are intentionally not reachable from
        # here; transitions are always forward in the lifecycle.

    # ------------------------------------------------------------------
    # Task state machine — delegated to :class:`TaskStateMachine`.
    #
    # See :mod:`goldfive.task_state_machine` for the implementation.
    # These thin shims preserve the historical public surface
    # (``DefaultSteerer.mark_task_*``) so executors / reporting handlers
    # / tests that bind to the router don't have to change.
    # ------------------------------------------------------------------

    async def mark_task_running(
        self,
        task_id: str,
        *,
        session: Session,
        detail: str = "",
        source: str = "other",
    ) -> None:
        await self._task_state_machine.mark_task_running(
            task_id, session=session, detail=detail, source=source
        )

    async def mark_task_progress(
        self,
        task_id: str,
        *,
        session: Session,
        fraction: float = 0.0,
        detail: str = "",
    ) -> None:
        await self._task_state_machine.mark_task_progress(
            task_id, session=session, fraction=fraction, detail=detail
        )

    async def mark_task_completed(
        self,
        task_id: str,
        *,
        session: Session,
        summary: str = "",
        artifacts: dict[str, str] | None = None,
        source: str = "other",
    ) -> None:
        await self._task_state_machine.mark_task_completed(
            task_id,
            session=session,
            summary=summary,
            artifacts=artifacts,
            source=source,
        )

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        recoverable: bool = True,
        source: str = "other",
    ) -> None:
        await self._task_state_machine.mark_task_failed(
            task_id,
            session=session,
            reason=reason,
            recoverable=recoverable,
            source=source,
        )

    async def mark_task_blocked(
        self,
        task_id: str,
        *,
        session: Session,
        blocker: str = "",
        needed: str = "",
        source: str = "other",
    ) -> None:
        await self._task_state_machine.mark_task_blocked(
            task_id,
            session=session,
            blocker=blocker,
            needed=needed,
            source=source,
        )

    async def mark_task_cancelled(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        source: str = "other",
    ) -> None:
        await self._task_state_machine.mark_task_cancelled(
            task_id, session=session, reason=reason, source=source
        )

    async def mark_task_not_needed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        source: str = "other",
    ) -> None:
        await self._task_state_machine.mark_task_not_needed(
            task_id, session=session, reason=reason, source=source
        )

    async def cascade_cancel_downstream(
        self,
        session: Session,
        cancelled_id: str,
        *,
        source: str = "cancellation",
    ) -> None:
        await self._task_state_machine.cascade_cancel_downstream(
            session, cancelled_id, source=source
        )

    # ------------------------------------------------------------------
    # Observer: drift detection + refine
    # ------------------------------------------------------------------

    async def observe(self, event: Any, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver.observe`."""
        await self._drift_observer.observe(event, session)

    async def observe_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        agent_name: str = "",
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver.observe_reasoning`."""
        await self._drift_observer.observe_reasoning(
            text,
            task=task,
            session=session,
            provider=provider,
            agent_name=agent_name,
        )

    async def _run_judge_background(
        self,
        *,
        text: str,
        session: Session,
        call_llm: ReflectiveCallLLM | None,
        judge_sink: Any,
        history_length: int,
        agent_name: str = "",
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._run_judge_background`."""
        await self._drift_observer._run_judge_background(
            text=text,
            session=session,
            call_llm=call_llm,
            judge_sink=judge_sink,
            history_length=history_length,
            agent_name=agent_name,
        )

    def _maybe_record_reasoning_binding(
        self,
        *,
        session: Session,
        verdict: Any,
        agent_name: str,
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._maybe_record_reasoning_binding`."""
        self._drift_observer._maybe_record_reasoning_binding(
            session=session, verdict=verdict, agent_name=agent_name
        )

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Shim — delegate to :meth:`DriftObserver.shutdown`."""
        await self._drift_observer.shutdown(timeout=timeout)

    async def drain_session_background_tasks(
        self, *, session_id: str, timeout: float = 2.0
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver.drain_session_background_tasks`."""
        await self._drift_observer.drain_session_background_tasks(
            session_id=session_id, timeout=timeout
        )

    async def _drain_background_set(
        self,
        bg_set: set[asyncio.Task[Any]],
        *,
        label: str,
        timeout: float,
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._drain_background_set`."""
        await self._drift_observer._drain_background_set(
            bg_set, label=label, timeout=timeout
        )

    @staticmethod
    def _truncate_trigger_input(text: str, limit: int = 2048) -> str:
        """Shim — delegate to :meth:`DriftObserver._truncate_trigger_input`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._truncate_trigger_input(text, limit=limit)

    def _resolve_available_agents(self) -> list[str] | list[dict[str, Any]] | None:
        """Shim — delegate to :meth:`DriftObserver._resolve_available_agents`."""
        return self._drift_observer._resolve_available_agents()

    def _maybe_take_reasoning_judge_slot(
        self,
        session: Session,
        *,
        agent_name: str = "",
    ) -> ReflectiveCallLLM | None:
        """Shim — delegate to :meth:`DriftObserver._maybe_take_reasoning_judge_slot`."""
        return self._drift_observer._maybe_take_reasoning_judge_slot(
            session, agent_name=agent_name
        )

    # ------------------------------------------------------------------
    # Reflective check — prompt + budget constants re-exported
    # ------------------------------------------------------------------
    #
    # Re-exported as class attributes so callers / subclasses / tests
    # that read ``DefaultSteerer.REFLECTIVE_*`` directly keep working.
    # The canonical definitions live on :class:`DriftObserver`.

    REFLECTIVE_SYSTEM_PROMPT: str = (
        "You are assessing your own progress on a task. Answer truthfully. "
        "Reply with a single JSON object and nothing else."
    )

    REFLECTIVE_USER_PROMPT_TEMPLATE: str = (
        "You are assessing your own progress on a task.\n\n"
        "CURRENT TASK:\n"
        "id: {task_id}\n"
        "title: {task_title}\n"
        "description: {task_description}\n\n"
        "WHAT YOU HAVE DONE IN THE LAST {window} LLM TURNS (summarized):\n"
        "- recent tool calls: {tool_call_summary}\n"
        "- recent reasoning (last 3 blocks): {reasoning_summary}\n\n"
        "Q: Are you making forward progress on the task? Reply with a "
        "single JSON object:\n"
        '{{"making_progress": true|false, "confidence": 0.0-1.0, '
        '"reason": "one-sentence explanation"}}'
    )

    REFLECTIVE_MAX_OUTPUT_TOKENS: int = 16384

    async def note_llm_call(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver.note_llm_call`."""
        await self._drift_observer.note_llm_call(session)

    async def maybe_run_reflective_check(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver.maybe_run_reflective_check`."""
        await self._drift_observer.maybe_run_reflective_check(session)

    def note_agent_activity(
        self,
        session: Session,
        *,
        kind: str,
        agent_name: str = "",
        task_id: str = "",
        detail: str = "",
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver.note_agent_activity`."""
        self._drift_observer.note_agent_activity(
            session,
            kind=kind,
            agent_name=agent_name,
            task_id=task_id,
            detail=detail,
        )

    def note_tool_observation(
        self,
        session: Session,
        *,
        agent_name: str,
        task_id: str,
        tool_name: str,
        args: Any,
        result: Any,
        error: Exception | str | None = None,
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver.note_tool_observation`."""
        self._drift_observer.note_tool_observation(
            session,
            agent_name=agent_name,
            task_id=task_id,
            tool_name=tool_name,
            args=args,
            result=result,
            error=error,
        )

    async def note_agent_turn(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver.note_agent_turn`."""
        await self._drift_observer.note_agent_turn(session)

    # Minimum spacing between two task-boundary-triggered GOAL_DRIFT
    # judge calls (seconds). Re-exported here as a class attribute so
    # tests / subclasses that read ``DefaultSteerer._GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S``
    # keep working. The canonical definition lives on :class:`DriftObserver`.
    _GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S: float = 10.0

    async def _maybe_run_goal_drift_on_task_boundary(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver._maybe_run_goal_drift_on_task_boundary`."""
        await self._drift_observer._maybe_run_goal_drift_on_task_boundary(session)

    async def maybe_run_goal_drift_check(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver.maybe_run_goal_drift_check`."""
        await self._drift_observer.maybe_run_goal_drift_check(session)

    def _spawn_goal_drift_judge_background(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver._spawn_goal_drift_judge_background`."""
        self._drift_observer._spawn_goal_drift_judge_background(session)

    async def _run_goal_drift_judge_background(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver._run_goal_drift_judge_background`."""
        await self._drift_observer._run_goal_drift_judge_background(session)

    def _spawn_drift_handler_background(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._spawn_drift_handler_background`."""
        self._drift_observer._spawn_drift_handler_background(drift, session)

    async def _run_drift_handler_background(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._run_drift_handler_background`."""
        await self._drift_observer._run_drift_handler_background(drift, session)

    async def _wait_background_drifts_idle(self) -> None:
        """Shim — delegate to :meth:`DriftObserver._wait_background_drifts_idle`."""
        await self._drift_observer._wait_background_drifts_idle()

    async def _emit_reflective_failure(
        self, session: Session, *, task_id: str, reason: str
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._emit_reflective_failure`."""
        await self._drift_observer._emit_reflective_failure(
            session, task_id=task_id, reason=reason
        )

    # --- Reflective prompt helpers -----------------------------------

    # Liberal JSON extractor: tolerates markdown code fences and leading /
    # trailing prose around the object. Re-exported here for any
    # subclass / test that pokes the bare-attribute name; the canonical
    # definition lives on :class:`DriftObserver`.
    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

    @classmethod
    def _parse_reflective_response(cls, raw: Any) -> dict[str, Any] | None:
        """Shim — delegate to :meth:`DriftObserver._parse_reflective_response`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._parse_reflective_response(raw)

    @staticmethod
    def _summarize_recent_tool_calls(session: Session, *, limit: int = 10) -> str:
        """Shim — delegate to :meth:`DriftObserver._summarize_recent_tool_calls`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._summarize_recent_tool_calls(session, limit=limit)

    @staticmethod
    def _summarize_recent_reasoning(session: Session, *, limit: int = 3) -> str:
        """Shim — delegate to :meth:`DriftObserver._summarize_recent_reasoning`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._summarize_recent_reasoning(session, limit=limit)

    @staticmethod
    def _steer_dedupe_id(event: Any) -> str:
        """Shim — delegate to :meth:`DriftObserver._steer_dedupe_id`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._steer_dedupe_id(event)

    @staticmethod
    def _unpack_steer_context(drift: DriftEvent) -> tuple[str, str, str]:
        """Shim — delegate to :meth:`DriftObserver._unpack_steer_context`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._unpack_steer_context(drift)

    @classmethod
    def _is_duplicate_steer(cls, event: Any, session: Session) -> bool:
        """Shim — delegate to :meth:`DriftObserver._is_duplicate_steer`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._is_duplicate_steer(event, session)

    @staticmethod
    def _drift_from_control(event: Any, session: Session) -> DriftEvent | None:
        """Shim — delegate to :meth:`DriftObserver._drift_from_control`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._drift_from_control(event, session)

    def detect_drift(
        self,
        event: Any,
        session: Session,
    ) -> DriftEvent | None:
        """Shim — delegate to :meth:`DriftObserver.detect_drift`."""
        return self._drift_observer.detect_drift(event, session)

    # ------------------------------------------------------------------
    # Plan-mutation drift hooks (invoked by reporting-tool handlers)
    # ------------------------------------------------------------------

    async def report_new_work_discovered(
        self,
        *,
        session: Session,
        parent_task_id: str,
        title: str,
        description: str,
        assignee: str = "",
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver.report_new_work_discovered`."""
        await self._drift_observer.report_new_work_discovered(
            session=session,
            parent_task_id=parent_task_id,
            title=title,
            description=description,
            assignee=assignee,
        )

    async def report_plan_divergence(
        self,
        *,
        session: Session,
        note: str,
        suggested_action: str = "",
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver.report_plan_divergence`."""
        await self._drift_observer.report_plan_divergence(
            session=session,
            note=note,
            suggested_action=suggested_action,
        )

    # ==================================================================
    # Internals
    # ==================================================================

    # --- Plan lookup --------------------------------------------------

    @staticmethod
    def _find_task(session: Session, task_id: str) -> Task | None:
        # Identity helper — kept as a router-level staticmethod so the
        # historical call surface (``DefaultSteerer._find_task``) keeps
        # working unchanged. Mirrors
        # :meth:`goldfive.task_state_machine.TaskStateMachine._find_task`.
        if not task_id or session.plan is None:
            return None
        for t in session.plan.tasks:
            if t.id == task_id:
                return t
        return None

    # ------------------------------------------------------------------
    # Drift dispatch + intervention ladder + promotion — thin shims to
    # :class:`DriftObserver`. The real implementations live on
    # ``self._drift_observer`` (bucket 3c of the steerer split). The
    # router keeps these shims so the wide collection of tests, the
    # planner's structural-drift hook, and any third-party subclasses
    # that historically poked the bare-attribute names on
    # :class:`DefaultSteerer` keep working byte-equivalently.
    # ------------------------------------------------------------------

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver.handle_drift`.

        The central drift-routing entry point. See
        :meth:`DriftObserver.handle_drift` for the contract.
        """
        await self._drift_observer.handle_drift(drift, session)

    def _ladder_level_for(
        self,
        kind: DriftKind,
        severity: DriftSeverity,
        occurrence_count: int,
    ) -> InterventionLevel:
        """Shim — delegate to :meth:`DriftObserver._ladder_level_for`."""
        return self._drift_observer._ladder_level_for(kind, severity, occurrence_count)

    async def _dispatch_nudge(self, drift: DriftEvent, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver._dispatch_nudge`."""
        await self._drift_observer._dispatch_nudge(drift, session)

    async def _dispatch_goldfive_steer_control(
        self,
        drift: DriftEvent,
        session: Session,
        *,
        body_override: str = "",
    ) -> bool:
        """Shim — delegate to :meth:`DriftObserver._dispatch_goldfive_steer_control`."""
        return await self._drift_observer._dispatch_goldfive_steer_control(
            drift, session, body_override=body_override
        )

    async def _dispatch_goldfive_pause_control(
        self,
        drift: DriftEvent,
        session: Session,
        *,
        reason: str,
    ) -> bool:
        """Shim — delegate to :meth:`DriftObserver._dispatch_goldfive_pause_control`."""
        return await self._drift_observer._dispatch_goldfive_pause_control(
            drift, session, reason=reason
        )

    async def _dispatch_pause_escalate(
        self,
        drift: DriftEvent,
        session: Session,
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._dispatch_pause_escalate`."""
        await self._drift_observer._dispatch_pause_escalate(drift, session)

    def _tag_adapter_cancel_reason(
        self, drift: DriftEvent, *, session: Session | None = None
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._tag_adapter_cancel_reason`."""
        self._drift_observer._tag_adapter_cancel_reason(drift, session=session)

    def _tag_adapter_cancel_reason_for_promotion(
        self, drift: DriftEvent, *, session: Session | None = None
    ) -> str:
        """Shim — delegate to :meth:`DriftObserver._tag_adapter_cancel_reason_for_promotion`."""
        return self._drift_observer._tag_adapter_cancel_reason_for_promotion(
            drift, session=session
        )

    async def _request_adapter_cancel(self, reason: str) -> None:
        """Shim — delegate to :meth:`DriftObserver._request_adapter_cancel`."""
        await self._drift_observer._request_adapter_cancel(reason)

    def _is_late_drift_for_terminated_invocation(
        self, drift: DriftEvent, session: Session
    ) -> bool:
        """Shim — delegate to :meth:`DriftObserver._is_late_drift_for_terminated_invocation`."""
        return self._drift_observer._is_late_drift_for_terminated_invocation(drift, session)

    def _resolve_active_invocation_ids(
        self, drift: DriftEvent, session: Session
    ) -> list[str]:
        """Shim — delegate to :meth:`DriftObserver._resolve_active_invocation_ids`."""
        return self._drift_observer._resolve_active_invocation_ids(drift, session)

    async def request_invocation_cancel(
        self,
        *,
        drift: DriftEvent,
        session: Session,
        cancel_inflight_task: bool = False,
    ) -> list[str]:
        """Shim — delegate to :meth:`DriftObserver.request_invocation_cancel`."""
        return await self._drift_observer.request_invocation_cancel(
            drift=drift,
            session=session,
            cancel_inflight_task=cancel_inflight_task,
        )

    @staticmethod
    def _should_request_cancel_for_drift(drift: DriftEvent) -> bool:
        """Shim — delegate to :meth:`DriftObserver._should_request_cancel_for_drift`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._should_request_cancel_for_drift(drift)

    @staticmethod
    def _cancel_reason_for_drift(drift: DriftEvent) -> str:
        """Shim — delegate to :meth:`DriftObserver._cancel_reason_for_drift`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._cancel_reason_for_drift(drift)

    async def _cancel_inflight_for_revision(
        self, drift: DriftEvent, session: Session
    ) -> list[str]:
        """Shim — delegate to :meth:`DriftObserver._cancel_inflight_for_revision`."""
        return await self._drift_observer._cancel_inflight_for_revision(drift, session)

    def _severity_meets_promotion_threshold(self, severity: DriftSeverity) -> bool:
        """Shim — delegate to :meth:`DriftObserver._severity_meets_promotion_threshold`."""
        return self._drift_observer._severity_meets_promotion_threshold(severity)

    def _should_promote_to_steer(self, drift: DriftEvent, session: Session) -> bool:
        """Shim — delegate to :meth:`DriftObserver._should_promote_to_steer`."""
        return self._drift_observer._should_promote_to_steer(drift, session)

    async def _promote_drift_to_steer(self, drift: DriftEvent, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver._promote_drift_to_steer`.

        Audit issue #402 (dispatch-before-plan-swap) is preserved on
        the DriftObserver side; the shim simply forwards.
        """
        await self._drift_observer._promote_drift_to_steer(drift, session)

    @staticmethod
    def _compose_goldfive_steer_body(drift: DriftEvent) -> str:
        """Shim — delegate to :meth:`DriftObserver._compose_goldfive_steer_body`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._compose_goldfive_steer_body(drift)

    async def _apply_user_steer_state(
        self,
        drift: DriftEvent,
        session: Session,
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._apply_user_steer_state`."""
        await self._drift_observer._apply_user_steer_state(drift, session)

    async def _record_refine_outcome(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        succeeded: bool,
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._record_refine_outcome`."""
        await self._drift_observer._record_refine_outcome(
            session, drift, succeeded=succeeded
        )

    def reset_for_turn(self, session: Session) -> None:
        """Shim — delegate to :meth:`DriftObserver.reset_for_turn`.

        Wired from :meth:`Runner.run` immediately after the
        ``run_started`` event so each turn starts with an empty
        outcome table.
        """
        self._drift_observer.reset_for_turn(session)

    def _occurrence_count_for_ladder(self, session: Session, drift: DriftEvent) -> int:
        """Shim — delegate to :meth:`DriftObserver._occurrence_count_for_ladder`."""
        return self._drift_observer._occurrence_count_for_ladder(session, drift)

    async def _escalate_refine_failure_as_critical_drift(
        self, session: Session, source: DriftEvent, *, reason: str
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._escalate_refine_failure_as_critical_drift`."""
        await self._drift_observer._escalate_refine_failure_as_critical_drift(
            session, source, reason=reason
        )

    # ------------------------------------------------------------------
    # ``observation_only`` gate — stays on the router because it reads
    # the router-owned :attr:`_observation_only` flag (set in
    # :meth:`__init__` from :class:`~goldfive.config.SteeringConfig`).
    # ------------------------------------------------------------------

    def _should_inject(self) -> bool:
        """Return ``True`` iff the steerer should actually inject side effects.

        Single named gate for the three steering injection points
        (goldfive#254):

        * plan mutation in :meth:`_apply_revision`
          (``set_session_plan`` + ``last_addressed_revision_by_drift_key``);
        * ``GOLDFIVE_STEER`` ControlMessage enqueue in
          :meth:`_dispatch_goldfive_steer_control`;
        * the plugin ``request_invocation_cancel`` flag in
          :meth:`request_invocation_cancel`.

        ``False`` when :class:`~goldfive.config.SteeringConfig.observation_only`
        is in effect — detection still runs, ``planner.refine_steer``
        still runs, ``PlanRevised`` still emits (with ``dry_run=True``),
        but the in-flight invocation is not touched. Defined as a tiny
        helper rather than inlining ``not self._observation_only`` at
        three sites so the intent is grep-able and a future fourth
        injection point has a single gate to honour.
        """
        return not self._observation_only

    # ------------------------------------------------------------------
    # Plan install entry points — thin shims to :class:`PlanReviser`
    # ------------------------------------------------------------------

    async def install_initial_plan(
        self,
        *,
        session: Session,
        plan: Plan,
        is_pivot: bool = False,
    ) -> bool:
        """Shim — delegate to :meth:`PlanReviser.install_initial_plan`."""
        return await self._plan_reviser.install_initial_plan(
            session=session, plan=plan, is_pivot=is_pivot
        )

    async def install_revision_for_drift(
        self,
        *,
        session: Session,
        drift: DriftEvent,
        revised_plan: Plan,
    ) -> bool:
        """Shim — delegate to :meth:`PlanReviser.install_revision_for_drift`."""
        return await self._plan_reviser.install_revision_for_drift(
            session=session, drift=drift, revised_plan=revised_plan
        )

    async def install_revision_for_user_steer(
        self,
        *,
        session: Session,
        raw: Any,
        revised_plan: Plan,
    ) -> bool:
        """Shim — delegate to :meth:`PlanReviser.install_revision_for_user_steer`."""
        return await self._plan_reviser.install_revision_for_user_steer(
            session=session, raw=raw, revised_plan=revised_plan
        )

    async def install_user_steer(
        self,
        *,
        drift: DriftEvent,
        prior: Plan,
        llm_revision: Plan | None,
        session: Session,
    ) -> Plan:
        """Shim — delegate to :meth:`PlanReviser.install_user_steer`."""
        return await self._plan_reviser.install_user_steer(
            drift=drift,
            prior=prior,
            llm_revision=llm_revision,
            session=session,
        )

    def _build_minimal_steer_evolution(
        self, prior: Plan, drift: DriftEvent
    ) -> Plan:
        """Shim — delegate to :meth:`PlanReviser._build_minimal_steer_evolution`."""
        return self._plan_reviser._build_minimal_steer_evolution(prior, drift)

    async def _install_with_drift(
        self,
        *,
        session: Session,
        drift: DriftEvent,
        revised_plan: Plan,
        apply_user_steer_state: bool,
    ) -> bool:
        """Shim — delegate to :meth:`PlanReviser._install_with_drift`."""
        return await self._plan_reviser._install_with_drift(
            session=session,
            drift=drift,
            revised_plan=revised_plan,
            apply_user_steer_state=apply_user_steer_state,
        )

    async def apply_user_steer_with_plan(
        self,
        *,
        drift: DriftEvent,
        session: Session,
        revised_plan: Plan,
    ) -> bool:
        """Shim — delegate to :meth:`PlanReviser.apply_user_steer_with_plan`."""
        return await self._plan_reviser.apply_user_steer_with_plan(
            drift=drift, session=session, revised_plan=revised_plan
        )

    # Consecutive refine failures tolerated per (drift_kind, task_id)
    # before we give up and mark the task FAILED. Class attribute so
    # subclasses / tests can tune it without poking at instance state.
    REFINE_FAILURE_THRESHOLD: int = 2

    # Wall-clock seconds of task silence before a drift escalates to
    # HUMAN_INTERVENTION_REQUIRED (goldfive#271 — replaces the deleted
    # count cap). A productively-iterating task emits continuous
    # progress events (``mark_task_running`` / ``mark_task_progress`` /
    # ``_emit_task_transitioned`` updates ``Session.task_last_progress_at``);
    # a stuck task does not. When a drift fires for a task whose last
    # progress is older than this threshold, the steerer treats it as
    # structurally unrecoverable and pauses for human intervention.
    # Default 600s (10 minutes) — generous enough that legitimate slow
    # work (model reasoning, multi-step research) is not interrupted,
    # tight enough that a Qwen judge re-firing on a wedged task does
    # not loop forever. ``0`` disables the gate (useful for tests).
    PROGRESS_STALL_THRESHOLD_SECONDS: float = 600.0


    # --- Refine atomicity + observability (goldfive a4) --------------

    def _get_plan_lock(self, session: Session) -> asyncio.Lock:
        """Shim — delegate to :meth:`PlanReviser._get_plan_lock`."""
        return self._plan_reviser._get_plan_lock(session)

    async def _wait_plan_stable(
        self,
        session: Session,
        *,
        timeout: float | None = 1.0,
    ) -> bool:
        """Shim — delegate to :meth:`PlanReviser._wait_plan_stable`."""
        return await self._plan_reviser._wait_plan_stable(session, timeout=timeout)

    @staticmethod
    def _new_attempt_id() -> str:
        """Shim — delegate to :meth:`PlanReviser._new_attempt_id`."""
        from goldfive.plan_reviser import PlanReviser

        return PlanReviser._new_attempt_id()

    async def _emit_refine_attempted(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        attempt_id: str,
    ) -> None:
        """Shim — delegate to :meth:`PlanReviser._emit_refine_attempted`."""
        await self._plan_reviser._emit_refine_attempted(
            session, drift, attempt_id=attempt_id
        )

    async def _emit_refine_failed(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        attempt_id: str,
        failure_kind: str,
        reason: str,
        detail: str = "",
    ) -> None:
        """Shim — delegate to :meth:`PlanReviser._emit_refine_failed`."""
        await self._plan_reviser._emit_refine_failed(
            session,
            drift,
            attempt_id=attempt_id,
            failure_kind=failure_kind,
            reason=reason,
            detail=detail,
        )

    def observe_refine(
        self,
        session: Session,
        drift: DriftEvent,
    ) -> contextlib.AbstractAsyncContextManager[str]:
        """Shim — delegate to :meth:`PlanReviser.observe_refine`.

        Note: this returns the underlying async context manager directly
        rather than wrapping it in another ``@asynccontextmanager`` so
        ``async with steerer.observe_refine(...)`` resolves to the same
        ``_AsyncGeneratorContextManager`` instance that ``async with
        steerer._plan_reviser.observe_refine(...)`` would resolve to.
        """
        return self._plan_reviser.observe_refine(session, drift)

    async def _emit_plan_revised_correlation(
        self,
        session: Session,
        revised: Plan,
        drift: DriftEvent,
        *,
        attempt_id: str,
    ) -> None:
        """Shim — delegate to :meth:`PlanReviser._emit_plan_revised_correlation`."""
        await self._plan_reviser._emit_plan_revised_correlation(
            session, revised, drift, attempt_id=attempt_id
        )

    @staticmethod
    def _fold_runtime_terminal_statuses(revised: Plan, prior: Plan | None) -> Plan:
        """Shim — delegate to :meth:`PlanReviser._fold_runtime_terminal_statuses`."""
        from goldfive.plan_reviser import PlanReviser

        return PlanReviser._fold_runtime_terminal_statuses(revised, prior)

    def _apply_revision(
        self, session: Session, revised: Plan, drift: DriftEvent
    ) -> tuple[Plan, bool]:
        """Shim — delegate to :meth:`PlanReviser._apply_revision`."""
        return self._plan_reviser._apply_revision(session, revised, drift)

    # --- Event construction ------------------------------------------

    def _new_envelope(self, session: Session) -> Any:
        """Build a fresh ``Event`` envelope via :mod:`goldfive.events`."""
        from goldfive.events import new_event

        return new_event(session.run_id, session.next_sequence(), session_id=session.id)

    async def _emit(self, event_pb: Any) -> None:
        from goldfive.events import emit

        await emit(self._sinks, event_pb)

    @staticmethod
    def _drift_kind_pb_value(kind: DriftKind) -> int:
        from goldfive.pb.goldfive.v1 import types_pb2

        name = f"DRIFT_KIND_{kind.name}"
        return getattr(types_pb2, name, getattr(types_pb2, "DRIFT_KIND_CUSTOM", 0))

    @staticmethod
    def _drift_severity_pb_value(severity: DriftSeverity) -> int:
        from goldfive.pb.goldfive.v1 import types_pb2

        name = f"DRIFT_SEVERITY_{severity.name}"
        return getattr(types_pb2, name, getattr(types_pb2, "DRIFT_SEVERITY_UNSPECIFIED", 0))

    # --- Concrete emitters -------------------------------------------
    #
    # Per-status proto emit helpers — delegated to
    # :class:`~goldfive.task_state_machine.TaskStateMachine`. Thin shims
    # so test fixtures that mock ``DefaultSteerer._emit_task_*`` keep
    # intercepting at the router level and so other components inside
    # this module can keep calling ``self._emit_task_*`` unchanged.

    async def _emit_task_started(self, session: Session, task_id: str, detail: str) -> None:
        await self._task_state_machine._emit_task_started(session, task_id, detail)

    async def _emit_task_progress(
        self, session: Session, task_id: str, fraction: float, detail: str
    ) -> None:
        await self._task_state_machine._emit_task_progress(session, task_id, fraction, detail)

    async def _emit_task_completed(
        self,
        session: Session,
        task_id: str,
        summary: str,
        artifacts: dict[str, str],
    ) -> None:
        await self._task_state_machine._emit_task_completed(
            session, task_id, summary, artifacts
        )

    async def _emit_task_failed(
        self, session: Session, task_id: str, reason: str, recoverable: bool
    ) -> None:
        await self._task_state_machine._emit_task_failed(
            session, task_id, reason, recoverable
        )

    async def _emit_task_blocked(
        self, session: Session, task_id: str, blocker: str, needed: str
    ) -> None:
        await self._task_state_machine._emit_task_blocked(
            session, task_id, blocker, needed
        )

    async def _emit_task_cancelled(self, session: Session, task_id: str, reason: str) -> None:
        await self._task_state_machine._emit_task_cancelled(session, task_id, reason)

    async def _emit_task_transitioned(
        self,
        session: Session,
        task: Task,
        *,
        from_status: TaskStatus,
        to_status: TaskStatus,
        source: str,
    ) -> None:
        await self._task_state_machine._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=to_status,
            source=source,
        )

    async def _emit_plan_revision_transitions(
        self,
        session: Session,
        prev_plan: Plan | None,
        revised: Plan,
    ) -> None:
        await self._task_state_machine._emit_plan_revision_transitions(
            session, prev_plan, revised
        )

    def _resolve_invocation_id_for_agent(self, agent_name: str) -> str:
        return self._task_state_machine._resolve_invocation_id_for_agent(agent_name)

    # ------------------------------------------------------------------
    # DriftObserver shims — drift-event emission + lifecycle stamping +
    # source attribution + structural escalation primitives. The real
    # implementations live on :class:`goldfive.drift_observer.DriftObserver`
    # held on ``self._drift_observer``; these are byte-equivalent shims
    # for callers that historically poked the bare-attribute names on
    # :class:`DefaultSteerer`.
    # ------------------------------------------------------------------

    # Class-level constants re-exported for callers (and methods still
    # on this router) that read ``self._TERMINAL_DRIFT_KINDS`` /
    # ``self._USER_AUTHORED_DRIFT_KINDS`` /
    # ``self._USER_OR_TRAJECTORY_DRIFT_KINDS``. The canonical
    # definitions live on :class:`DriftObserver`; aliasing them here
    # keeps subclasses + tests that re-read these sets working.
    _TERMINAL_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.HUMAN_INTERVENTION_REQUIRED,
            DriftKind.REPEATED_FAILURE,
        }
    )

    _USER_AUTHORED_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.USER_PAUSE,
        }
    )

    _USER_OR_TRAJECTORY_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.GOAL_DRIFT,
        }
    )

    async def _emit_drift_detected(self, session: Session, drift: DriftEvent) -> None:
        """Shim — delegate to :meth:`DriftObserver._emit_drift_detected`."""
        await self._drift_observer._emit_drift_detected(session, drift)

    @classmethod
    def _is_terminal_drift(cls, drift: DriftEvent) -> bool:
        """Shim — delegate to :meth:`DriftObserver._is_terminal_drift`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._is_terminal_drift(drift)

    async def _close_open_boundaries_for_terminal_drift(self, drift: DriftEvent) -> None:
        """Shim — delegate to :meth:`DriftObserver._close_open_boundaries_for_terminal_drift`."""
        await self._drift_observer._close_open_boundaries_for_terminal_drift(drift)

    def _stamp_drift_lifecycle(
        self,
        session: Session,
        drift: DriftEvent,
        evt: Any,
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._stamp_drift_lifecycle`."""
        self._drift_observer._stamp_drift_lifecycle(session, drift, evt)

    @staticmethod
    def _drift_lifecycle_pb_value(lifecycle: str, types_pb2: Any) -> int:
        """Shim — delegate to :meth:`DriftObserver._drift_lifecycle_pb_value`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._drift_lifecycle_pb_value(lifecycle, types_pb2)

    @classmethod
    def _resolve_authored_by(cls, drift: DriftEvent) -> str:
        """Shim — delegate to :meth:`DriftObserver._resolve_authored_by`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._resolve_authored_by(drift)

    @staticmethod
    def _drift_annotation_id(drift: DriftEvent) -> str:
        """Shim — delegate to :meth:`DriftObserver._drift_annotation_id`."""
        from goldfive.drift_observer import DriftObserver

        return DriftObserver._drift_annotation_id(drift)

    def _is_task_progress_stalled(self, drift: DriftEvent, session: Session) -> bool:
        """Shim — delegate to :meth:`DriftObserver._is_task_progress_stalled`."""
        return self._drift_observer._is_task_progress_stalled(drift, session)

    async def _emit_progress_stalled_escalation(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._emit_progress_stalled_escalation`."""
        await self._drift_observer._emit_progress_stalled_escalation(drift, session)

    async def _emit_handler_exhausted_escalation(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Shim — delegate to :meth:`DriftObserver._emit_handler_exhausted_escalation`."""
        await self._drift_observer._emit_handler_exhausted_escalation(drift, session)

    @staticmethod
    def _plans_structurally_identical(prior: Plan | None, revised: Plan) -> bool:
        """Shim — delegate to :meth:`PlanReviser._plans_structurally_identical`."""
        from goldfive.plan_reviser import PlanReviser

        return PlanReviser._plans_structurally_identical(prior, revised)

    @staticmethod
    def _integrate_correction_supersedes(revised: Plan) -> Plan:
        """Shim — delegate to :meth:`PlanReviser._integrate_correction_supersedes`."""
        from goldfive.plan_reviser import PlanReviser

        return PlanReviser._integrate_correction_supersedes(revised)

    def _repin_current_task_on_supersedes(
        self,
        session: Session,
        revised: Plan,
    ) -> None:
        """Shim — delegate to :meth:`PlanReviser._repin_current_task_on_supersedes`."""
        self._plan_reviser._repin_current_task_on_supersedes(session, revised)

    async def _emit_plan_revised(
        self,
        session: Session,
        revised: Plan,
        drift: DriftEvent,
        *,
        prev_plan: Plan | None = None,
        attempt_id: str | None = None,
        dry_run: bool | None = None,
    ) -> None:
        """Shim — delegate to :meth:`PlanReviser._emit_plan_revised`."""
        await self._plan_reviser._emit_plan_revised(
            session,
            revised,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
            dry_run=dry_run,
        )

    @staticmethod
    def _build_refine_input_summary(
        drift: DriftEvent,
        prev_plan: Plan | None,
    ) -> str:
        """Shim — delegate to :meth:`PlanReviser._build_refine_input_summary`."""
        from goldfive.plan_reviser import PlanReviser

        return PlanReviser._build_refine_input_summary(drift, prev_plan)

    @staticmethod
    def _build_refine_output_summary(revised: Plan) -> str:
        """Shim — delegate to :meth:`PlanReviser._build_refine_output_summary`."""
        from goldfive.plan_reviser import PlanReviser

        return PlanReviser._build_refine_output_summary(revised)
