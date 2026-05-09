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
import dataclasses
import enum
import inspect
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive import _state_audit
from goldfive import orchestration_state as _ostate
from goldfive.drift import (
    classify_refusal,
    classify_stop_reason,
    classify_tool_error,
)
from goldfive.drift.reasoning import (
    DEFAULT_REASONING_DRIFT_MODE,
    ReasoningDriftMode,
)
from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    CancellationRequest,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    RefineOutcome,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
    bump_revision,
    channel_processor_active,
    replace_edges,
    set_session_plan,
    with_task_status,
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
    import inspect

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
        # reasoning-extracted binding onto OrchestrationStore. The
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
    # Task state machine
    # ------------------------------------------------------------------

    async def mark_task_running(
        self,
        task_id: str,
        *,
        session: Session,
        detail: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``RUNNING`` and emit ``TaskStarted``.

        ``source`` (goldfive#251 R4) is the attribution string emitted on
        the paired ``TaskTransitioned`` sink event — see
        :func:`goldfive.events.task_transitioned_event` for the
        vocabulary. Defaults to ``"other"`` for callers that haven't been
        threaded through (back-compat); the live LLM-driven path through
        :mod:`goldfive.reporting` passes ``"llm_report"`` /
        ``"handler_default"`` / ``"supersedes_reroute"`` as appropriate.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: Plan + Task are frozen. Mutate by deriving a new
        # plan via :func:`with_task_status` and swapping the pointer
        # under :func:`channel_processor_active`. The local ``task``
        # reference becomes stale; refresh from the live plan so
        # downstream side-effects see the new status.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.RUNNING))
        task = self._find_task(session, task_id) or task
        session.current_task_id = task_id
        if detail:
            session.agent_notes[task_id] = detail
        # goldfive#152: stamp current_task_* on the orchestration-state
        # dict so downstream prompt templates / refine paths see it.
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.RUNNING)
        # goldfive#271: stamp task progress liveness for the structural
        # progress-stall escalation. A drift firing on this task within
        # ``PROGRESS_STALL_THRESHOLD_SECONDS`` of any progress signal is
        # considered productively iterating; outside that window, the
        # next drift escalates to HUMAN_INTERVENTION_REQUIRED.
        session.task_last_progress_at[task_id] = time.monotonic()
        # Seed the observed-agent lineage with the static plan
        # assignee. ``before_tool_callback`` extends the set with each
        # delegated child agent so consumers (e.g. the reasoning judge)
        # can distinguish "child of a delegation chain rooted at the
        # assignee" from "off-plan agent". Cleared on every terminal
        # transition.
        session.task_lineage[task_id] = (
            {task.assignee_agent_id} if task.assignee_agent_id else set()
        )
        await self._emit_task_started(session, task_id, detail)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.RUNNING,
            source=source,
        )

    async def mark_task_progress(
        self,
        task_id: str,
        *,
        session: Session,
        fraction: float = 0.0,
        detail: str = "",
    ) -> None:
        """Record mid-task progress and emit ``TaskProgress``.

        No status transition — a progress update is a liveness ping only.
        ``fraction`` is clamped to ``[0.0, 1.0]``.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        try:
            frac = float(fraction)
        except (TypeError, ValueError):
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        session.task_progress[task_id] = frac
        if detail:
            session.agent_notes[task_id] = detail
        # goldfive#271: refresh progress liveness so a productively
        # iterating task is not flagged as stalled by the structural
        # escalation gate.
        session.task_last_progress_at[task_id] = time.monotonic()
        await self._emit_task_progress(session, task_id, frac, detail)

    async def mark_task_completed(
        self,
        task_id: str,
        *,
        session: Session,
        summary: str = "",
        artifacts: dict[str, str] | None = None,
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``COMPLETED`` and emit ``TaskCompleted``.

        See :meth:`mark_task_running` for the ``source`` contract.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.COMPLETED))
        task = self._find_task(session, task_id) or task
        if summary:
            session.completed_results[task_id] = summary
        # goldfive#152: clear current_task_* if we were the active task.
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.COMPLETED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        # iter-11B: pair the prior ``agent_invocation_started`` entry
        # so the GOAL_DRIFT judge does not see an orphan-start +
        # task-COMPLETED shape and false-positive on "looping".
        # ``after_run_callback`` will append the real
        # ``agent_invocation_completed`` slightly later when the agent
        # actually returns; duplicate completed entries are harmless
        # (each is benign and the ring buffer trims naturally).
        if task.assignee_agent_id:
            self.note_agent_activity(
                session,
                kind="agent_invocation_completed",
                agent_name=task.assignee_agent_id,
                task_id=task_id,
            )
        await self._emit_task_completed(session, task_id, summary, artifacts or {})
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.COMPLETED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        await self._maybe_run_goal_drift_on_task_boundary(session)

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        recoverable: bool = True,
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``FAILED`` and emit ``TaskFailed``.

        Also fires a drift event of kind ``TASK_FAILED_RECOVERABLE`` or
        ``TASK_FAILED_FATAL``. The drift event is dispatched through the
        same drift pipeline as observer-detected drift: if severity is
        ``>= WARNING`` (both of these are) we invoke ``planner.refine``.

        When ``recoverable=False`` the failure is fatal for this task
        lineage: **cascade-cancel every reachable downstream non-terminal
        task** via :meth:`cascade_cancel_downstream`, so the plan lands
        in a consistent terminal-only shape instead of orphaning
        dependents that would sit PENDING forever. This is the
        implementation of PLAN-LIFECYCLE.md §6.2 step 3 and it shares
        its primitive with the §6.3 cancel cascade path
        (:meth:`mark_task_cancelled`). The downstream cascade fires
        *before* we dispatch the fatal drift through ``_handle_drift``
        so that planner.refine sees the post-cascade plan shape and a
        refine-failure back-off does not leave orphans behind. See
        ``STATE-MACHINE.md §"Cascade semantics on unrecoverable drift"``.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.FAILED))
        task = self._find_task(session, task_id) or task
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.FAILED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        # iter-11B: pair the prior ``agent_invocation_started`` entry
        # so the GOAL_DRIFT judge does not see an orphan-start +
        # task-FAILED shape and false-positive on "looping".  See the
        # matching write in :meth:`mark_task_completed` for rationale.
        if task.assignee_agent_id:
            self.note_agent_activity(
                session,
                kind="agent_invocation_completed",
                agent_name=task.assignee_agent_id,
                task_id=task_id,
            )
        await self._emit_task_failed(session, task_id, reason, recoverable)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.FAILED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        await self._maybe_run_goal_drift_on_task_boundary(session)
        # Fatal failures cascade downstream via the same primitive used
        # by mark_task_cancelled, so both §6.2 and §6.3 produce the
        # same TaskCancelled event stream and share rejection guards.
        if not recoverable:
            # The cascade is a propagation of the same source-attribution
            # decision (e.g. an LLM-reported fatal failure cascades as
            # ``"cancellation"`` from the framework's perspective — the
            # cascaded tasks weren't moved by the LLM directly).
            await self.cascade_cancel_downstream(session, task_id, source="cancellation")
        kind = DriftKind.TASK_FAILED_RECOVERABLE if recoverable else DriftKind.TASK_FAILED_FATAL
        severity = DriftSeverity.WARNING if recoverable else DriftSeverity.CRITICAL
        drift = DriftEvent(
            kind=kind,
            severity=severity,
            detail=f"task {task_id} failed: {reason}" if reason else f"task {task_id} failed",
            current_task_id=task_id,
        )
        # iter-11A: spawn the drift cascade fire-and-forget so the
        # reporting tool that triggered us (``report_task_failed``)
        # can return immediately. The cascade
        # (refine → supersedes → cancellation) lands asynchronously;
        # tests that need the post-cascade plan state await
        # :meth:`_wait_background_drifts_idle`.
        self._spawn_drift_handler_background(drift, session)

    async def mark_task_blocked(
        self,
        task_id: str,
        *,
        session: Session,
        blocker: str = "",
        needed: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``BLOCKED`` and emit ``TaskBlocked``.

        Also fires a drift event of kind ``BLOCKED`` which flows through
        the standard drift pipeline (WARNING severity → refine).
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        # BLOCKED is not a terminal status but we still guard against
        # re-blocking a task that's already blocked (idempotent).
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.BLOCKED))
        task = self._find_task(session, task_id) or task
        if blocker or needed:
            session.agent_notes[task_id] = f"blocked: {blocker}" + (
                f" (needed: {needed})" if needed else ""
            )
        await self._emit_task_blocked(session, task_id, blocker, needed)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.BLOCKED,
            source=source,
        )
        detail = f"task {task_id} blocked: {blocker}" + (f" (needed: {needed})" if needed else "")
        drift = DriftEvent(
            kind=DriftKind.BLOCKED,
            severity=DriftSeverity.WARNING,
            detail=detail,
            current_task_id=task_id,
        )
        # iter-11A: spawn the drift cascade fire-and-forget so the
        # reporting tool that triggered us (``report_task_blocked``)
        # can return immediately. See :meth:`mark_task_failed` for
        # the matching call-site comment.
        self._spawn_drift_handler_background(drift, session)

    async def mark_task_cancelled(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``CANCELLED`` and emit ``TaskCancelled``.

        Also **cascades** the cancellation forward through the plan's
        edges: every non-terminal task reachable from ``task_id`` is
        transitioned to ``CANCELLED`` with a "cascade from <task_id>"
        reason. Without this cascade, downstream PENDING tasks with a
        CANCELLED predecessor would never satisfy the executor's
        "all deps COMPLETED" check — they would sit PENDING forever and
        the executor would report the run as successful while leaving
        them orphaned. See TASK-LIFECYCLE.md §"Cancellation cascade" and
        STATE-MACHINE.md §"Cascade semantics on unrecoverable drift".
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            # Already terminal (including already CANCELLED) — no-op, and
            # crucially do NOT re-run the cascade: we would double-emit
            # TaskCancelled events for downstream tasks on every call.
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.CANCELLED))
        task = self._find_task(session, task_id) or task
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.CANCELLED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        await self._emit_task_cancelled(session, task_id, reason)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.CANCELLED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        # Fire before cascade so the judge sees the initiator's transition;
        # cascade-cancel downstream tasks share the same rate-limit bucket
        # and will no-op as subsequent boundary fires fall within the
        # 10s guard.
        await self._maybe_run_goal_drift_on_task_boundary(session)
        await self.cascade_cancel_downstream(session, task_id, source="cancellation")

    async def mark_task_not_needed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``NOT_NEEDED`` terminally.

        Introduced by the overlay-model refactor (goldfive#141). Unlike
        :meth:`mark_task_cancelled` this path does NOT cascade — a task
        the :class:`~goldfive.reconciler.PlanReconciler` deemed "not
        needed" post-invocation is an observation about that specific
        plan entry, not a signal that downstream work is invalid. The
        reconciler independently evaluates each PENDING task.

        Idempotent on terminal tasks. Emits ``TaskCancelled`` at the
        proto level (there's no dedicated NOT_NEEDED event — the
        status lives on the task itself and sinks can distinguish via
        ``task.status`` on the next ``TaskCancelled`` / ``PlanRevised``
        envelope). The reason string carries the distinguishing
        context ("not needed: superseded by ...").
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        assert session.plan is not None
        with channel_processor_active():
            set_session_plan(
                session,
                with_task_status(session.plan, task_id, TaskStatus.NOT_NEEDED),
            )
        task = self._find_task(session, task_id) or task
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.NOT_NEEDED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        # There is no dedicated ``TaskNotNeeded`` proto message;
        # reuse TaskCancelled with the reason prefix so sinks that
        # inspect reason can differentiate if they wish. The live
        # ``task.status`` on the plan is the authoritative signal.
        await self._emit_task_cancelled(
            session, task_id, f"not_needed: {reason}" if reason else "not_needed"
        )
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.NOT_NEEDED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        await self._maybe_run_goal_drift_on_task_boundary(session)

    async def cascade_cancel_downstream(
        self,
        session: Session,
        cancelled_id: str,
        *,
        source: str = "cancellation",
    ) -> None:
        """BFS-cancel every downstream non-terminal task of ``cancelled_id``.

        Shared primitive for both cascade codepaths
        (PLAN-LIFECYCLE.md §6.2 unrecoverable cascade and §6.3
        cancellation cascade):

        - The recoverable path
          (:meth:`mark_task_cancelled`) calls it after transitioning
          the initiator to ``CANCELLED``.
        - The unrecoverable path
          (:meth:`mark_task_failed` with ``recoverable=False``) calls
          it after transitioning the initiator to ``FAILED``.

        Both paths therefore produce the same ``TaskCancelled`` event
        stream for the downstream set and share the rejection guards
        (terminal tasks are skipped; diamond DAGs are de-duplicated).
        The initiator's own transition-emission is caller-controlled —
        this method only emits for the *downstream* set.

        Walks ``session.plan.edges`` forward from ``cancelled_id`` and
        transitions every reachable non-terminal task to ``CANCELLED``
        in-place, emitting one ``TaskCancelled`` event per transition.
        Terminal tasks (COMPLETED / FAILED / CANCELLED) are skipped so a
        diamond DAG does not re-cancel a task through two paths and so
        already-COMPLETED dependents are preserved verbatim.

        Implemented as an iterative BFS on a precomputed adjacency list
        (rather than recursing into :meth:`mark_task_cancelled`) so a
        single top-level cancel produces one summary log line and a
        predictable number of emitted events, independent of graph shape.
        """
        plan = session.plan
        if plan is None:
            return
        # Precompute forward adjacency once.
        downstream: dict[str, list[str]] = {}
        for e in plan.edges:
            downstream.setdefault(e.from_task_id, []).append(e.to_task_id)
        tasks_by_id: dict[str, Task] = {t.id: t for t in plan.tasks if t.id}

        # goldfive#205: structured reason consumed by harmonograf's
        # Trajectory view. Old ``cascade from <id>`` form preserved in a
        # human-readable tail after the colon so sinks that render the
        # raw reason keep their existing copy; new sinks parse the
        # ``upstream_failed:`` prefix to categorise the cancel.
        cascade_reason = f"upstream_failed:{cancelled_id}"
        cascaded: list[str] = []
        queue: list[str] = list(downstream.get(cancelled_id, []))
        visited: set[str] = set()
        while queue:
            next_id = queue.pop(0)
            if next_id in visited:
                continue
            visited.add(next_id)
            dep = tasks_by_id.get(next_id)
            if dep is None:
                continue
            if dep.status in _TERMINAL_TASK_STATUSES:
                # Already terminal (COMPLETED/FAILED/CANCELLED) — preserve
                # and do not traverse its children. A COMPLETED task that
                # sits downstream of a late-cancelled ancestor keeps its
                # completion; cascading past it would mean cancelling
                # tasks whose preserved prerequisite is still valid.
                continue
            # Transition by deriving a new plan and swapping it in. We
            # deliberately do NOT recurse through ``mark_task_cancelled``
            # here; we fan out via our own BFS queue so the surrounding
            # summary log and emission count stay deterministic.
            #
            # goldfive#247: the local ``dep`` reference becomes stale
            # after the swap (frozen Task) — re-resolve from the live
            # plan so the emit reads the new status. Note that
            # ``tasks_by_id`` was built from the *original* plan; the
            # iteration loop only inspects ``status`` for terminal
            # gating which is monotonic (a task that wasn't terminal
            # at loop start is the one we're cancelling).
            dep_from = dep.status
            assert session.plan is not None
            with channel_processor_active():
                set_session_plan(
                    session,
                    with_task_status(session.plan, next_id, TaskStatus.CANCELLED),
                )
            # Drop the observed-agent lineage now the task is terminal —
            # mirrors the cleanup that ``mark_task_cancelled`` performs
            # for the initiator (this BFS does not recurse through that
            # method so the cleanup is duplicated explicitly).
            session.task_lineage.pop(next_id, None)
            await self._emit_task_cancelled(session, next_id, cascade_reason)
            # Re-fetch so the transition emit reads the swapped task.
            updated_dep = self._find_task(session, next_id) or dep
            await self._emit_task_transitioned(
                session,
                updated_dep,
                from_status=dep_from,
                to_status=TaskStatus.CANCELLED,
                source=source,
            )
            cascaded.append(next_id)
            for grandchild in downstream.get(next_id, []):
                if grandchild not in visited:
                    queue.append(grandchild)
        if cascaded:
            log.info(
                "DefaultSteerer: cascade-cancelled %d downstream task(s) from %s: %s",
                len(cascaded),
                cancelled_id,
                ", ".join(cascaded),
            )

    # ------------------------------------------------------------------
    # Observer: drift detection + refine
    # ------------------------------------------------------------------

    async def observe(self, event: Any, session: Session) -> None:
        """Inspect ``event``, classify drift, and refine if severe enough.

        ``ControlMessage`` values are handled first — they carry explicit
        user intent (STEER / CANCEL / PAUSE) and map directly to the
        corresponding ``USER_*`` drift kinds without going through the
        heuristic classifiers. Every other event falls through to
        :meth:`detect_drift`.

        STEER ControlMessages are deduped by their source annotation id
        (goldfive#171): a delivery retry or UI double-fire of the same
        STEER lands here twice, but cascade-cancel + refine must only
        happen once. The dedupe set lives on ``session.state`` under
        :data:`orchestration_state.KEY_PROCESSED_STEER_IDS` with FIFO
        eviction after :data:`PROCESSED_STEER_IDS_CAP` entries. Content-
        based drifts (LOOPING_REASONING, tool errors, …) are NOT
        deduped — they're heuristic signals, not user actions.
        """
        if self._is_duplicate_steer(event, session):
            steer_id = self._steer_dedupe_id(event)
            log.debug("DefaultSteerer.observe: dropping duplicate STEER id=%s", steer_id)
            return
        drift = self._drift_from_control(event, session)
        if drift is None:
            drift = self.detect_drift(event, session)
        if drift is None:
            return
        await self._handle_drift(drift, session)

    async def observe_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,  # noqa: ARG002 -- reserved for future detectors
        session: Session,
        provider: str = "",  # noqa: ARG002 -- reserved for per-provider dispatch
        agent_name: str = "",
    ) -> None:
        """Feed a chain-of-thought / reasoning block into the drift pipeline.

        Appends ``text`` to ``session.reasoning_history`` (bounded by
        ``session.reasoning_history_max``), then runs the reasoning
        detectors. Emits at most one drift per call.

        Pipeline dispatch (goldfive#226, refined in #251):

        * Always-on detector — :func:`~goldfive.drift.reasoning.detect_looping_reasoning`
          runs first on every call. It catches the byte-identical /
          near-identical repetition pattern that the LLM judge does
          not, and it is cheap. Its drift verdict is handled
          SYNCHRONOUSLY: callers awaiting this method see the resulting
          ``DriftDetected`` sink emission and any refine dispatch
          before control returns.
        * Mode-selected pipeline — :func:`~goldfive.drift.reasoning.analyze_reasoning`
          runs in the configured ``reasoning_drift_mode``. The LLM judge
          path is rate-limited to at most one call every
          ``reasoning_drift_rate_limit`` thinking messages per task; the
          first thinking message of every task always fires a judge
          call. Counters reset on task transition. This path is
          **fire-and-forget**: the judge is scheduled via
          :func:`asyncio.create_task` and tracked on
          ``self._background_judges`` so :meth:`shutdown` can drain it
          at run end. Its drift verdict may therefore arrive AFTER tool
          calls from the same turn have already dispatched — the refine
          machinery handles "drift arrives mid-run" via supersedes, so
          there is no correctness regression; late refines simply apply
          to a plan state that has already advanced.

        Why the judge path is async: this method is called from the
        adapter's model-response callback, which is on the critical
        path for ADK tool dispatch. Awaiting a minute-long local-llama
        judge round-trip inline serialized every subsequent tool call
        behind it.

        Adapters call this from their model-response callback once they
        have extracted reasoning_content (OpenAI), thinking blocks
        (Anthropic), or thought parts (Google). Safe to call with empty
        text -- the pipeline no-ops.
        """
        if not text:
            return
        history = session.reasoning_history
        history.append(text)
        cap = getattr(session, "reasoning_history_max", 20) or 20
        overflow = len(history) - cap
        if overflow > 0:
            del history[:overflow]
        from goldfive.drift.reasoning import detect_looping_reasoning

        # Always-on loop detector. A fire short-circuits before the
        # mode-selected pipeline so it remains the canonical signal
        # for "repetitive" reasoning regardless of mode. Cheap, and
        # its verdict can affect the current turn, so it stays inline.
        drift = detect_looping_reasoning(text, session)
        if drift is not None:
            # Populate ``trigger_input`` on drifts produced by the
            # always-on detector (it does not set it itself — it is
            # framework-agnostic).
            if not drift.trigger_input:
                drift.trigger_input = self._truncate_trigger_input(text)
            await self._handle_drift(drift, session)
            return

        # Mode-selected pipeline (judge / embedding / both / off).
        # The judge path is rate-limited per-(agent, task) bucket.
        # Historically this awaited ``analyze_reasoning`` inline; as of
        # goldfive#251 the judge is fire-and-forget so the
        # model-response callback can return immediately.
        rl_call_llm = self._maybe_take_reasoning_judge_slot(session, agent_name=agent_name)
        # Fast-exit when there's nothing for ``analyze_reasoning`` to
        # do: ``mode="off"``, or ``mode="judge"`` with no judge slot
        # (rate-limited or globally disabled). Embedding and "both"
        # modes always schedule — their embedding pipeline runs even
        # when the judge slot is empty.
        if self._reasoning_drift_mode == "off":
            return
        if self._reasoning_drift_mode == "judge" and rl_call_llm is None:
            return
        # Thread the first bound sink into the judge path so a
        # ``ReasoningJudgeInvoked`` event fires on every judge call,
        # regardless of verdict. ``None`` when no sinks are bound —
        # the classifier then stays sink-less and behaves as before.
        judge_sink = self._sinks[0] if self._sinks else None
        # Snapshot the reasoning-history position at schedule time so
        # the bg pipeline sees the same view the inline pattern
        # detectors just saw, even if subsequent turns append more
        # entries before the bg task runs. Without this, a detector
        # that slices ``history[-N:-1]`` (expecting ``text`` to be the
        # last entry) would see ``text`` itself in the comparison
        # window and trivially self-match (goldfive#251 ordering
        # regression surfaced by the cluster-tightening one-shot
        # test). ``history_length`` is the length AFTER ``text`` was
        # appended — the bg path trims ``session.reasoning_history``
        # to this length for its invocation.
        history_length = len(session.reasoning_history)
        bg_task = asyncio.create_task(
            self._run_judge_background(
                text=text,
                session=session,
                call_llm=rl_call_llm,
                judge_sink=judge_sink,
                history_length=history_length,
                agent_name=agent_name,
            ),
            # goldfive#243: encode session.id in the task name so
            # :meth:`drain_session_background_tasks` can filter pending
            # tasks by the run boundary that's terminating, leaving any
            # other concurrent session's tasks alone.
            name=f"goldfive-reasoning-judge:{session.id}",
        )
        self._background_judges.add(bg_task)
        bg_task.add_done_callback(self._background_judges.discard)

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
        """Run the mode-selected reasoning drift pipeline off the critical path.

        Scheduled by :meth:`observe_reasoning` as an
        :func:`asyncio.create_task` so the adapter's model-response
        callback can return before ADK dispatches the response's tool
        calls. Awaits :func:`~goldfive.drift.reasoning.analyze_reasoning`
        and, if it yields a :class:`DriftEvent`, routes it through
        :meth:`_handle_drift` — same effect as the historical inline
        path, just resolving later.

        ``history_length`` pins the ``session.reasoning_history`` view
        the pipeline sees to the same tail index that was in effect
        when the bg task was scheduled. Later turns that append to
        the shared history (this same session receiving more
        reasoning blocks before the bg task runs) would otherwise
        shift the detectors' "exclude self" slice and generate false
        self-match LOOPING signals. We temporarily truncate the
        session view for the duration of this bg invocation and
        restore it after; concurrent bg tasks serialize on the same
        session's reasoning_history via the asyncio event loop (no
        threading) so the save/restore pattern is safe in practice.

        Never raises: any exception (from the judge LLM, the embedding
        pipeline, or ``_handle_drift``) is logged at ``WARNING`` and
        swallowed. The background task must not crash the run; the
        adapter callback that scheduled us has long since returned.
        """
        try:
            from goldfive.drift.reasoning import analyze_reasoning_with_focus

            # Save the shared live history and swap in a list snapshot
            # truncated to the length captured at schedule time. Using
            # list slicing (not mutation) keeps any already-escaped
            # reference (e.g. a concurrent detector) pointing at the
            # original list. We restore the live reference in a
            # ``finally`` so intervening appends are not lost.
            original_history = session.reasoning_history
            pinned_history = list(original_history[:history_length])
            session.reasoning_history = pinned_history
            try:
                # Phase 1 of goldfive#271 — call the focused-verdict
                # path so we get the judge's plan-task attribution
                # alongside the drift signal. ``analyze_reasoning_with_focus``
                # is a sibling of ``analyze_reasoning`` that threads a
                # :class:`ReasoningJudgeVerdict` instead of just the
                # drift; legacy callers of ``analyze_reasoning`` keep
                # their existing return shape.
                #
                # goldfive#244 — also forward the wrapped agent tree so
                # the judge can recognise legitimate coordinator → sub-
                # agent delegation as ON-TASK rather than OFF_TOPIC.
                # Reuses the same shape the planner already consumes
                # (``ADKAdapter.available_agents_tree``); legacy adapters
                # without the property fall back to a flat
                # ``available_agents`` list, and adapters with neither
                # leave ``available_agents=None`` — the judge prompt
                # then renders byte-identically to pre-#244.
                judge_available_agents = self._resolve_available_agents()
                verdict = await analyze_reasoning_with_focus(
                    text,
                    session,
                    mode=self._reasoning_drift_mode,
                    call_llm=call_llm,
                    model=self._reasoning_drift_model,
                    sink=judge_sink,
                    agent_name=agent_name,
                    available_agents=judge_available_agents,
                )
            finally:
                # Restore the live history. Any entries appended by
                # subsequent turns are preserved because we pointed
                # ``session.reasoning_history`` at a separate list for
                # our window.
                session.reasoning_history = original_history

            # Record the reasoning-extracted binding onto the
            # orchestration store regardless of the drift verdict —
            # an on-task verdict that names a different plan task is
            # itself a useful pin-resolution signal (the agent has
            # silently moved to a different task without reporting).
            self._maybe_record_reasoning_binding(
                session=session,
                verdict=verdict,
                agent_name=agent_name,
            )

            drift = verdict.drift
            if drift is None:
                return
            if not drift.trigger_input:
                drift.trigger_input = self._truncate_trigger_input(text)
            # Late-drift tolerance (goldfive#319). The judge is
            # fire-and-forget so its verdict can land after the
            # invocation that produced the reasoning has already
            # terminated — adk-web outer-turn boundary crossed, agent
            # moved on. Routing such a verdict through the cancel +
            # ladder dispatch would either cancel an unrelated next
            # invocation or refine against a plan whose offending step
            # is already complete. We still want the drift on the wire
            # for observability ("from past turn"), so we emit it
            # directly via :meth:`_emit_drift_detected` and skip the
            # rest of the dispatch. The guard is scoped to the
            # background-judge path because only that path produces
            # verdicts that may outlive the originating invocation —
            # synchronous detectors run inline on the model-response
            # callback and always see a live invocation.
            if self._is_late_drift_for_terminated_invocation(drift, session):
                log.info(
                    "DefaultSteerer: stale judge verdict; invocation for "
                    "agent=%r task=%r already terminated; drift kind=%s "
                    "recorded but refine skipped",
                    drift.current_agent_id or "-",
                    drift.current_task_id or "-",
                    drift.kind.value,
                )
                if not drift.authored_by:
                    drift.authored_by = self._resolve_authored_by(drift)
                await self._emit_drift_detected(session, drift)
                return
            await self._handle_drift(drift, session)
        except asyncio.CancelledError:
            # Propagate cancellation so :meth:`shutdown` / event-loop
            # teardown can cleanly abort a still-running judge without
            # the WARNING log below muddying the signal.
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            log.warning(
                "DefaultSteerer: background reasoning-judge raised (swallowed): %s",
                exc,
            )

    def _maybe_record_reasoning_binding(
        self,
        *,
        session: Session,
        verdict: Any,
        agent_name: str,
    ) -> None:
        """Stamp a reasoning-extracted binding onto the OrchestrationStore.

        Phase 1 of goldfive#271. Called from
        :meth:`_run_judge_background` after the LLM judge returns its
        :class:`~goldfive.drift.reasoning_judge.ReasoningJudgeVerdict`.
        Records a binding when:

        * the verdict carries a non-empty ``focused_task_id``,
        * the agent name is non-empty (we key bindings by agent),
        * ``focus_confidence`` is at least the configured threshold.

        Lower-confidence verdicts are silently dropped so the pin
        ladder doesn't consume noisy bindings. Failures inside the
        store helper degrade silently — the judge's primary job is
        the drift signal, not the binding.
        """
        if not agent_name:
            return
        focused = getattr(verdict, "focused_task_id", "")
        if not focused:
            return
        confidence = float(getattr(verdict, "focus_confidence", 0.0) or 0.0)
        if confidence < self._reasoning_binding_confidence_threshold:
            log.debug(
                "DefaultSteerer: reasoning binding for agent=%r "
                "task=%r dropped (confidence=%.2f < threshold=%.2f)",
                agent_name,
                focused,
                confidence,
                self._reasoning_binding_confidence_threshold,
            )
            return
        try:
            from goldfive.orchestration_store import OrchestrationStore

            store = OrchestrationStore.for_session(session)
            recorded = store.record_reasoning_extracted_binding(
                agent_name=agent_name,
                task_id=focused,
                confidence=confidence,
                recorded_at_turn=session.next_sequence(),
                run_id=session.run_id,
                session_id=session.id,
            )
            if recorded is not None:
                log.info(
                    "DefaultSteerer: recorded reasoning-extracted binding "
                    "agent=%r task=%r confidence=%.2f",
                    agent_name,
                    focused,
                    confidence,
                )
        except Exception as exc:  # noqa: BLE001 — never break the run
            log.warning(
                "DefaultSteerer: record_reasoning_extracted_binding raised (swallowed): %s",
                exc,
            )

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Drain background reasoning-judge + drift tasks with a bounded wait.

        Called at run / runner teardown so ``asyncio.create_task``
        handles scheduled by :meth:`observe_reasoning` (judges) and
        :meth:`mark_task_failed` / :meth:`mark_task_blocked` (drift
        cascades, iter-11A) do not leak beyond the event loop's
        lifetime. Waits at most ``timeout`` seconds (default 5.0) for
        all tracked tasks to finish; any still-running tasks past the
        timeout are cancelled and awaited briefly so their
        ``CancelledError`` propagation settles before we return.

        Idempotent: a second call when both tracking sets are empty
        is a no-op.
        """
        # Drain reasoning-judge tasks.
        if self._background_judges:
            await self._drain_background_set(
                self._background_judges, label="judge", timeout=timeout
            )
        # Drain drift-handler tasks (iter-11A).
        if self._background_drifts:
            await self._drain_background_set(
                self._background_drifts, label="drift", timeout=timeout
            )

    async def drain_session_background_tasks(
        self, *, session_id: str, timeout: float = 2.0
    ) -> None:
        """Drain background drift / judge tasks for a single session at run end.

        Goldfive#243. The pre-existing drain in :meth:`shutdown` only
        fires from :meth:`Runner.close`, which on long-running adk-web
        / shared-Runner deployments is invoked at process shutdown,
        NOT between user turns. A drift cascade dispatched at the end
        of turn N (e.g. a JUSTIFIED_DEVIATION refine triggered from a
        ``report_*`` tool) outlives turn N's ``RunAborted`` /
        ``RunCompleted`` and runs against an abandoned session — burning
        compute on retry-buried HTTP attempts and emitting spurious
        post-abort drifts (the brussels-sprouts e2e leaked ~10 minutes
        of compute and produced a HUMAN_INTERVENTION_REQUIRED on a
        long-dead session).

        Executors call this right before each terminal
        ``run_aborted_event`` / ``run_completed_event`` emission so the
        symmetry the iter-11A docstring already claims ("drained at run
        end") actually holds at run boundaries, not just process
        teardown. Same bounded-wait + cancel-stragglers semantics as
        :meth:`shutdown`; idempotent (second call shortly after the
        first is a no-op because the tracking sets are empty).

        Filtering: each background task is named
        ``goldfive-<kind>:<session_id>`` (see :meth:`_spawn_*_background`)
        so this method drains ONLY the tasks belonging to the run that
        is terminating, leaving any other concurrent session's tasks
        alone. Tasks predating goldfive#243 (or future spawns that
        forget the suffix) fall back to a session-prefix-aware match;
        if the session_id is the empty string we drain nothing and
        warn — that signals a caller bug rather than legitimate work.

        User-authored drifts (``USER_STEER`` / ``USER_CANCEL`` /
        ``USER_PAUSE``) are dispatched through :meth:`_handle_drift`
        synchronously from :meth:`observe`, so they never land on
        ``_background_drifts`` and are therefore not affected by this
        drain — operator intent survives across turns by construction.
        """
        if not session_id:
            log.warning(
                "DefaultSteerer.drain_session_background_tasks: empty "
                "session_id; refusing to drain (would otherwise match "
                "every pending background task)",
            )
            return
        suffix = f":{session_id}"
        drift_subset = {
            t for t in self._background_drifts if t.get_name().endswith(suffix)
        }
        judge_subset = {
            t for t in self._background_judges if t.get_name().endswith(suffix)
        }
        if drift_subset:
            await self._drain_background_set(
                drift_subset, label="drift", timeout=timeout
            )
        if judge_subset:
            await self._drain_background_set(
                judge_subset, label="judge", timeout=timeout
            )

    async def _drain_background_set(
        self,
        bg_set: set[asyncio.Task[Any]],
        *,
        label: str,
        timeout: float,
    ) -> None:
        """Bounded-wait drain for a background-task tracking set.

        Shared between :attr:`_background_judges` and
        :attr:`_background_drifts` (iter-11A). ``label`` is used in
        log messages only.
        """
        # Snapshot: tasks may be removed from the set by their
        # done-callbacks while we're iterating.
        pending = list(bg_set)
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=max(0.0, float(timeout)),
            )
        except TimeoutError:
            # Cancel the stragglers and give them a beat to unwind so
            # we don't leave "pending task" warnings on loop close.
            still_pending = [t for t in pending if not t.done()]
            for task in still_pending:
                task.cancel()
            if still_pending:
                try:
                    await asyncio.wait(still_pending, timeout=0.5)
                except Exception:  # noqa: BLE001 — defensive
                    pass
            # Phase 2.X (goldfive#271 Gap 3): only WARN when stragglers
            # were actually cancelled. The TimeoutError can fire even
            # when every task completed in the same instant the timeout
            # expired (gather scheduling vs. wait_for race) — those
            # cases logged ``cancelled 0 tasks`` which was both
            # confusing and noisy in the demo log. The DEBUG line
            # preserves visibility for diagnostics while keeping INFO
            # / WARNING reserved for the real "we cancelled work"
            # signal.
            if still_pending:
                log.warning(
                    "DefaultSteerer.shutdown: %d background %s task(s) "
                    "exceeded %.2fs timeout; cancelled",
                    len(still_pending),
                    label,
                    float(timeout),
                )
            else:
                log.debug(
                    "DefaultSteerer.shutdown: %.2fs timeout expired but "
                    "all %s tasks completed in the same instant; nothing "
                    "to cancel",
                    float(timeout),
                    label,
                )

    @staticmethod
    def _truncate_trigger_input(text: str, limit: int = 2048) -> str:
        """Truncate ``text`` for use as a ``DriftDetected.trigger_input``.

        Uses the same suffix convention as the reasoning-judge
        observability event so consumers see one truncation marker
        regardless of which detector produced the drift.
        """
        if not isinstance(text, str):
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + " … [truncated]"

    def _resolve_available_agents(self) -> list[str] | list[dict[str, Any]] | None:
        """Return the wrapped agent tree for a downstream prompt.

        Mirrors the resolution used by ``_handle_drift`` when threading
        ``available_agents`` into ``planner.refine``: prefer the
        structured ``ADKAdapter.available_agents_tree`` (goldfive#151,
        list of dicts with name/parent/role/kind/depth); fall back to
        the flat ``available_agents`` (list[str]) when the structured
        property is missing or empty; return ``None`` when the adapter
        is missing or exposes neither surface.

        Used by :meth:`_run_judge_background` to feed the reasoning
        judge's :data:`~goldfive.drift.reasoning_judge.AGENT_TREE_BLOCK_MAX_CHARS`
        bounded "AGENT TREE" prompt section (goldfive#244) so the judge
        can recognise legitimate coordinator → sub-agent delegation as
        ON-TASK rather than OFF_TOPIC. ``None`` keeps the judge's
        prompt byte-identical to the pre-#244 shape.
        """
        adapter = self._adapter
        if adapter is None:
            return None
        tree = getattr(adapter, "available_agents_tree", None)
        if isinstance(tree, list) and tree:
            return list(tree)
        flat = getattr(adapter, "available_agents", None)
        if flat:
            return list(flat)
        return None

    def _maybe_take_reasoning_judge_slot(
        self,
        session: Session,
        *,
        agent_name: str = "",
    ) -> ReflectiveCallLLM | None:
        """Return the judge ``call_llm`` when this turn is a judge turn.

        Rate-limit policy (goldfive#226):

        * First thinking message of every (agent, task) bucket always fires.
        * Subsequent messages skip ``(N-1)`` and then fire on the Nth.
        * Counters are scoped per-(agent, task) via
          ``session._reasoning_judge_counters`` so a task transition
          OR an agent switch resets the window lazily -- the next
          ``(agent, task_id)`` tuple is simply not in the dict yet, so
          its first message falls into the "count=0" branch.

        Pre-fix the key was a single string keyed on
        ``current_task_id or ""``. Every unpinned turn from every agent
        collapsed onto the ``""`` bucket, so agent B's first thinking
        block could legitimately skip the judge because unrelated
        agent A's unpinned turn had already incremented the counter.
        Bucketing by ``(agent_name, task_id)`` isolates each agent's
        cadence.

        Returns ``None`` when the judge is globally disabled (mode
        skips it, or ``reasoning_drift_call_llm`` is unconfigured).
        Also ``None`` on skip turns even when armed.
        """
        if self._reasoning_drift_call_llm is None:
            return None
        if self._reasoning_drift_mode not in ("judge", "both"):
            return None
        task_id = session.current_task_id or ""
        key = (agent_name or "", task_id)
        counters = session._reasoning_judge_counters
        count = counters.get(key, 0)
        # count=0 -> fire (first message on this (agent, task) bucket),
        # reset to 1. Otherwise fire when count % rate_limit == 0.
        fire = (count % self._reasoning_drift_rate_limit) == 0
        counters[key] = count + 1
        return self._reasoning_drift_call_llm if fire else None

    # ------------------------------------------------------------------
    # Reflective self-progress check (opt-in)
    # ------------------------------------------------------------------

    # Prompt templates. Pulled out as class attributes so subclasses can
    # override the wording without re-implementing the full check.
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

    # Per-callsite ``max_output_tokens`` budget (goldfive#271 follow-up).
    # The reflective check returns a small JSON verdict, but Qwen 3.5
    # thinking models share think+answer under one ceiling — 16384
    # covers the think prelude on the 35B variant without permitting
    # unbounded essays. See
    # :func:`goldfive._llm.call_llm_budget` docstring for sizing rationale.
    REFLECTIVE_MAX_OUTPUT_TOKENS: int = 16384

    async def note_llm_call(self, session: Session) -> None:
        """Record one LLM invocation against ``session``.

        Adapters call this once per LLM turn. Increments
        ``session._llm_calls_since_check``. When the counter reaches the
        configured ``reflective_check_interval`` (and a
        ``reflective_call_llm`` is configured), fires
        :meth:`maybe_run_reflective_check` and resets the counter.

        The counter is also reset (without firing a check) when the
        session's ``current_task_id`` changes — a new task gets a fresh
        window so the check is always scoped to the current task.

        No-ops when ``reflective_call_llm`` was not configured. The
        counter is only updated when the feature is enabled, so
        operators who never opt in pay no memory or call cost.
        """
        if self._reflective_call_llm is None:
            return
        # Reset window on task transitions so the check is always scoped
        # to the current task. Tracks the task id the counter currently
        # belongs to; when it changes (including the first call after a
        # session starts with no current task), we start fresh.
        current = session.current_task_id
        if current != session._reflective_check_task_id:
            session._reflective_check_task_id = current
            session._llm_calls_since_check = 0
        session._llm_calls_since_check += 1
        if session._llm_calls_since_check < self._reflective_check_interval:
            return
        # Reset before running so a check that itself triggers further
        # LLM calls in the agent loop doesn't double-fire.
        session._llm_calls_since_check = 0
        await self.maybe_run_reflective_check(session)

    async def maybe_run_reflective_check(self, session: Session) -> None:
        """Ask the agent "are you making progress?" and emit a drift.

        Opt-in, feature-gated by ``reflective_call_llm``. Does NOT
        advance the counter — callers that want counter-driven
        scheduling go through :meth:`note_llm_call`. This method is
        public so operators can also trigger a one-shot check from
        outside the interval (e.g. on a long-running task boundary).

        Outcomes:

        * ``making_progress=true`` with ``confidence >= 0.5`` → no drift.
        * ``making_progress=true`` with ``confidence < 0.5`` →
          ``UNCERTAIN_PROGRESS`` (INFO severity, observational only).
        * ``making_progress=false`` → ``SELF_REPORTED_STUCK`` (WARNING
          severity; flows through :meth:`_handle_drift` and may trigger
          ``planner.refine``).
        * Reflective LLM raises, returns empty/unparseable JSON, or
          returns JSON missing the expected keys → INFO ``CUSTOM``
          drift noting the reflective check itself failed. The run is
          never broken by a bad reflective call.
        """
        call_llm = self._reflective_call_llm
        if call_llm is None or session.plan is None:
            return
        task = self._find_task(session, session.current_task_id)
        if task is None:
            # No task to assess. Nothing useful to ask the model.
            return
        tool_call_summary = self._summarize_recent_tool_calls(session)
        reasoning_summary = self._summarize_recent_reasoning(session)
        user_prompt = self.REFLECTIVE_USER_PROMPT_TEMPLATE.format(
            task_id=task.id,
            task_title=task.title or "",
            task_description=task.description or "",
            window=self._reflective_check_interval,
            tool_call_summary=tool_call_summary,
            reasoning_summary=reasoning_summary,
        )
        from goldfive._llm_span import goldfive_llm_span

        # ``reflective_check`` targets a specific task / agent, so stamp
        # the driver agent + task onto the span and feed a composed
        # input_preview (tool calls + reasoning window) so operators can
        # answer "what did the reflective check see?" from the Gantt.
        reflective_input_preview = (
            f"task={task.id} ({task.title or ''})\n"
            f"tool_calls:\n{tool_call_summary}\n\n"
            f"reasoning:\n{reasoning_summary}"
        )
        parsed: dict[str, Any] | None = None
        try:
            async with goldfive_llm_span(
                sinks=self._sinks,
                name="reflective_check",
                model=self._reflective_model,
                session_id=session.id,
                run_id=session.run_id,
                task_id=task.id,
                sequence_fn=session.next_sequence,
                input_preview=reflective_input_preview,
                target_agent_id=task.assignee_agent_id or "",
                target_task_id=task.id,
            ) as span:
                # Bound the dispatch — see ``REFLECTIVE_MAX_OUTPUT_TOKENS``.
                # Also disable thinking (goldfive#271 follow-up to #311):
                # this is meta-cognition asking the agent if it's making
                # progress, not deep reasoning.
                from goldfive._llm import (
                    call_llm_budget,
                    call_llm_thinking_disabled,
                )

                with (
                    call_llm_budget(self.REFLECTIVE_MAX_OUTPUT_TOKENS),
                    call_llm_thinking_disabled(),
                ):
                    raw = await call_llm(
                        self.REFLECTIVE_SYSTEM_PROMPT,
                        user_prompt,
                        self._reflective_model,
                    )
                parsed = self._parse_reflective_response(raw)
                if parsed is None:
                    # Distinguish "model returned all thinking, no
                    # answer" from "model returned garbage" — see
                    # goldfive#271 follow-up to #311. ``call_llm`` is
                    # the closure built by ``make_default_adk_call_llm``
                    # / ``_build_judge_call_llm`` which stashes part
                    # counts on itself.
                    _thought_n = int(getattr(call_llm, "last_thought_count", 0) or 0)
                    _raw_str = raw if isinstance(raw, str) else ""
                    if not _raw_str.strip() and _thought_n > 0:
                        span.output_preview = (
                            f"empty answer ({_thought_n} thought "
                            f"part(s); the model spent its budget thinking "
                            f"and emitted no JSON)"
                        )
                    else:
                        span.output_preview = f"unparseable verdict; raw={raw!r:.200}"
                    span.decision_summary = f"reflective check on {task.id}: unparseable verdict"
                else:
                    making_progress_inline = parsed.get("making_progress")
                    conf_inline = parsed.get("confidence")
                    reason_inline = str(parsed.get("reason", "") or "")
                    span.output_preview = (
                        f"making_progress={making_progress_inline}, "
                        f"confidence={conf_inline}, "
                        f"reason={reason_inline or '(none)'}"
                    )
                    if isinstance(making_progress_inline, bool):
                        verdict_str = "progressing" if making_progress_inline else "stuck"
                    else:
                        verdict_str = "malformed"
                    span.decision_summary = f"reflective check on {task.id}: {verdict_str}"
        except Exception as exc:  # noqa: BLE001 - never break the run
            log.warning("DefaultSteerer.maybe_run_reflective_check: call_llm raised %s", exc)
            await self._emit_reflective_failure(
                session,
                task_id=task.id,
                reason=f"reflective call_llm raised: {exc}",
            )
            return
        if parsed is None:
            await self._emit_reflective_failure(
                session,
                task_id=task.id,
                reason=f"reflective response was not valid JSON: {raw!r:.200}",
            )
            return
        making_progress = parsed.get("making_progress")
        confidence = parsed.get("confidence")
        reason = str(parsed.get("reason", "") or "")
        if not isinstance(making_progress, bool):
            await self._emit_reflective_failure(
                session,
                task_id=task.id,
                reason=(f"reflective response missing boolean 'making_progress': {raw!r:.200}"),
            )
            return
        try:
            conf_val = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            conf_val = 0.0
        # Prefer the runtime-reasoning agent pin (set by the ADK
        # plugin's ``before_agent_callback``) over the static plan
        # assignee — when a coordinator delegates to a child the
        # child's reasoning produced this drift, not the assignee's.
        # Fall back to ``task.assignee_agent_id`` when the session
        # pin is empty (pre-pin race or non-ADK adapter that doesn't
        # populate it) so we keep back-compat.
        agent_id_for_drift = session.current_agent_id or task.assignee_agent_id
        if not making_progress:
            drift = DriftEvent(
                kind=DriftKind.SELF_REPORTED_STUCK,
                severity=DriftSeverity.WARNING,
                detail=(
                    f"self-reported stuck on task {task.id}"
                    + (f": {reason}" if reason else "")
                    + f" (confidence={conf_val:.2f})"
                ),
                current_task_id=task.id,
                current_agent_id=agent_id_for_drift,
            )
            await self._handle_drift(drift, session)
            return
        if conf_val < 0.5:
            drift = DriftEvent(
                kind=DriftKind.UNCERTAIN_PROGRESS,
                severity=DriftSeverity.INFO,
                detail=(
                    f"uncertain progress on task {task.id} "
                    f"(confidence={conf_val:.2f})" + (f": {reason}" if reason else "")
                ),
                current_task_id=task.id,
                current_agent_id=agent_id_for_drift,
            )
            await self._handle_drift(drift, session)
            return
        # making_progress=true, confidence >= 0.5 -- no drift.
        return

    # ------------------------------------------------------------------
    # GOAL_DRIFT — trajectory-level periodic check (opt-in, goldfive#143)
    # ------------------------------------------------------------------

    def note_agent_activity(
        self,
        session: Session,
        *,
        kind: str,
        agent_name: str = "",
        task_id: str = "",
        detail: str = "",
    ) -> None:
        """Record a recent agent-activity entry on ``session``.

        Push-only: adapters (or executors) call this once per
        ``AgentInvocationStarted`` / ``AgentInvocationCompleted`` so the
        GOAL_DRIFT judge has a rolling view of the trajectory. The ring
        buffer is trimmed to ``goal_drift_activity_window`` so the
        prompt stays bounded regardless of run length.

        Always safe to call (feature-gate is enforced at check time, not
        at record time) -- unlike :meth:`note_agent_turn`, this method
        does not short-circuit when ``goal_drift_call_llm`` is
        unconfigured so that sinks / tests can observe the recorded
        activity independently.
        """
        if not kind:
            return
        entry: dict[str, Any] = {"kind": kind}
        if agent_name:
            entry["agent_name"] = agent_name
        if task_id:
            entry["task_id"] = task_id
        if detail:
            # Keep individual entries bounded so a pathological detail
            # cannot blow up the prompt even before trimming.
            entry["detail"] = detail[:500]
        hist = session.recent_agent_activity
        hist.append(entry)
        overflow = len(hist) - self._goal_drift_activity_window
        if overflow > 0:
            del hist[:overflow]

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
        """Append a bounded tool-observation entry to ``session.recent_tool_observations``.

        Iter-10 PR 2. Population path for the three-state reasoning
        judge (PR 3 reads this buffer to distinguish a provoked
        deviation from an unprovoked one). Adapters call this from
        their ``after_tool_callback`` (success + acknowledged-failure)
        and ``on_tool_error_callback`` hooks.

        Push-only and trim-on-write — mirrors
        :meth:`note_agent_activity`. The buffer is bounded by
        ``session.recent_tool_observations_max`` (default 16) so the
        prompt the judge eventually reads stays small regardless of
        run length. Per-task filtering happens at READ time in the
        judge's prompt renderer; this writer captures every call.

        Always swallow internal errors. Observability must never break
        tool dispatch — a malformed ``args`` / ``result`` repr, a
        broken clock, or a pathological session must not raise out of
        an ADK callback. The catch is intentionally broad.
        """
        try:
            ts_ms = time.monotonic_ns() // 1_000_000
            try:
                args_preview = repr(args)[:240]
            except Exception:  # noqa: BLE001
                args_preview = "(unrepresentable args)"
            if result is None:
                result_preview = "(none)"
            else:
                try:
                    result_preview = repr(result)[:480]
                except Exception:  # noqa: BLE001
                    result_preview = "(unrepresentable result)"
            # Error detection: an explicit ``error=`` from the caller
            # (the on_tool_error path) wins; otherwise look for the
            # acknowledged-failure shape ``{"error": ...}`` in the
            # tool result. The reporting tools and most goldfive
            # tools return that shape on a soft failure.
            is_error = False
            error_message = ""
            if error is not None:
                is_error = True
                try:
                    error_message = str(error)[:240]
                except Exception:  # noqa: BLE001
                    error_message = "(unrepresentable error)"
            elif isinstance(result, dict) and "error" in result:
                is_error = True
                try:
                    error_message = str(result.get("error", ""))[:240]
                except Exception:  # noqa: BLE001
                    error_message = "(unrepresentable error)"
            entry: dict[str, Any] = {
                "ts_ms": ts_ms,
                "agent_name": agent_name,
                "task_id": task_id,
                "tool_name": tool_name,
                "args_preview": args_preview,
                "result_preview": result_preview,
                "is_error": is_error,
                "error_message": error_message,
            }
            hist = session.recent_tool_observations
            hist.append(entry)
            # Cap defaults to 16 (§3.1) but honour any session-local
            # override; clamp to >=1 so a pathological 0 / negative
            # value doesn't disable the buffer entirely (we always
            # want at least the most-recent entry).
            try:
                cap_raw = int(session.recent_tool_observations_max)
            except (TypeError, ValueError):
                cap_raw = 16
            cap = max(1, cap_raw)
            overflow = len(hist) - cap
            if overflow > 0:
                # Slice-delete is amortized O(1) on average for the
                # bounded ``overflow == 1`` case (the steady state once
                # the buffer is full), and is the same pattern
                # ``note_agent_activity`` uses.
                del hist[:overflow]
        except Exception as exc:  # noqa: BLE001
            log.debug("note_tool_observation: swallowed: %s", exc)

    async def note_agent_turn(self, session: Session) -> None:
        """Record one agent invocation against ``session``.

        Adapters call this once per completed agent invocation
        (``after_run_callback`` on ADK, or the equivalent hook on other
        frameworks). Increments
        ``session._agent_turns_since_goal_check``; when the counter
        reaches ``goal_drift_check_interval`` (and a
        ``goal_drift_call_llm`` is configured), fires
        :meth:`maybe_run_goal_drift_check` and resets the counter.

        No-ops when ``goal_drift_call_llm`` was not configured, so
        operators who never opt in pay no memory or LLM cost. Unlike
        :meth:`note_llm_call`, the counter is trajectory-level and is
        NOT reset on task transitions -- GOAL_DRIFT is about the whole
        tree's direction, not one task's progress.

        Spawn-and-detach (goldfive v22 regression fix). The judge is
        dispatched as a fire-and-forget background task — see the
        rationale on :meth:`_maybe_run_goal_drift_on_task_boundary`.
        ``after_run_callback`` runs on the agent's invocation task,
        which is the same cancellable scope a sibling drift can target
        via :meth:`request_invocation_cancel`; an inline await on the
        judge would die the same way the v22 ``judge_goal_drift`` span
        did. Tests that drove the inline path can drain via
        ``await asyncio.gather(*list(steerer._background_judges))``.
        """
        if self._goal_drift_call_llm is None:
            return
        session._agent_turns_since_goal_check += 1
        if session._agent_turns_since_goal_check < self._goal_drift_check_interval:
            return
        # Reset before running so a check that itself triggers further
        # invocations in the agent loop doesn't double-fire.
        session._agent_turns_since_goal_check = 0
        self._spawn_goal_drift_judge_background(session)

    # Minimum spacing between two task-boundary-triggered GOAL_DRIFT
    # judge calls, in seconds (goldfive#219). Task transitions can
    # arrive back-to-back (e.g. a cascade-cancel fan-out or a fast
    # research→write→review pipeline), and we don't want to pay for
    # N LLM calls per burst; one is enough to catch drift. Turn-based
    # scheduling has its own interval (``goal_drift_check_interval``)
    # and is not affected by this guard.
    _GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S: float = 10.0

    async def _maybe_run_goal_drift_on_task_boundary(self, session: Session) -> None:
        """Fire :meth:`maybe_run_goal_drift_check` on a task transition.

        Task completions / failures / cancellations are natural
        "am I still on plan?" checkpoints, so we fire the judge here
        in addition to the turn-counter-driven path (goldfive#219).
        Short pipelines that finish before ``goal_drift_check_interval``
        turns would otherwise never trigger the judge.

        Rate-limited: if two task transitions happen within
        :data:`_GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` seconds of
        each other, only the first fires a judge call. Callers pass
        a fresh ``time.time()`` implicitly via the session-stored
        ``_last_goal_drift_check_ts``.

        Also resets ``session._agent_turns_since_goal_check`` so a
        task boundary that lands on exactly the interval boundary
        does not pay for two back-to-back judge calls.

        No-ops when ``goal_drift_call_llm`` is unconfigured — that
        gate is enforced inside :meth:`maybe_run_goal_drift_check`;
        we short-circuit here only to avoid bumping the timestamp
        when no judge will run.

        Spawn-and-detach (goldfive v22 regression fix). The judge LLM
        call is dispatched as a fire-and-forget background task on
        :attr:`_background_judges` rather than awaited inline. The
        ``mark_task_*`` callers run on the agent's invocation task —
        which is registered with the ADK plugin's ``_invocation_tasks``
        for cooperative cancel — so a sibling cancel (supersede,
        runaway delegation, refine-driven preempt) firing
        ``task.cancel()`` on the agent's invocation task while the
        inline judge was awaiting its LLM round-trip would surface a
        ``CancelledError`` inside ``classify_goal_drift``. The
        ``judge_goal_drift`` span ended with ``error=CancelledError``
        and an empty stack, the verdict was lost, and operator-visible
        evidence (v22 trace) showed the cancel landing the moment the
        span opened. Detaching the judge from the cancellable task
        scope — same pattern as the reasoning judge at
        :meth:`_run_judge_background` — keeps it alive across cancel
        propagation and drainable at :meth:`shutdown`.
        """
        if self._goal_drift_call_llm is None:
            return
        now = time.time()
        last = getattr(session, "_last_goal_drift_check_ts", 0.0)
        if now - last < self._GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S:
            return
        session._last_goal_drift_check_ts = now
        # Reset the turn counter so the next turn-interval check starts
        # fresh rather than firing one more judge call on the next turn.
        session._agent_turns_since_goal_check = 0
        self._spawn_goal_drift_judge_background(session)

    async def maybe_run_goal_drift_check(self, session: Session) -> None:
        """Run the trajectory-level GOAL_DRIFT judge once, cost-bounded.

        Opt-in, feature-gated by ``goal_drift_call_llm``. Does NOT
        advance the counter -- callers that want counter-driven
        scheduling go through :meth:`note_agent_turn`. Public so
        operators can trigger a one-shot check from outside the
        interval (e.g. on a long idle period with no task transitions).

        Outcomes:

        * Judge returns ``{"progressing": true}`` → no drift emitted.
        * Judge returns ``{"progressing": false, "reason": "..."}`` →
          ``GOAL_DRIFT`` drift at CRITICAL severity; flows through
          :meth:`_handle_drift` so the #142 ladder (once merged) can
          route it to Level 4.
        * Judge raises, returns malformed JSON, or returns a dict
          missing / with a non-boolean ``progressing`` field → no
          drift emitted. False positives on plumbing failures would
          spam operators; see goldfive#143 rationale.
        """
        call_llm = self._goal_drift_call_llm
        if call_llm is None:
            return
        from goldfive.drift.goals import classify_goal_drift

        # Snapshot activity so subsequent appends during the await do
        # not perturb the prompt the judge saw.
        activity = list(session.recent_agent_activity)
        drift = await classify_goal_drift(
            goals=session.goals,
            plan=session.plan,
            observed_actions=activity,
            model=self._goal_drift_model,
            call_llm=call_llm,
            current_task_id=session.current_task_id,
            sinks=self._sinks,
            run_id=session.run_id,
            session_id=session.id,
            sequence_fn=session.next_sequence,
            # goldfive#245 — pass the live session so the judge can
            # post-LLM re-read ``session.plan`` after its await and
            # drop verdicts whose target task transitioned during the
            # round-trip.
            session=session,
        )
        if drift is None:
            return
        await self._handle_drift(drift, session)

    def _spawn_goal_drift_judge_background(self, session: Session) -> None:
        """Spawn :meth:`maybe_run_goal_drift_check` as a fire-and-forget task.

        Goldfive v22 regression fix. The trajectory-level GOAL_DRIFT
        judge used to be awaited inline from
        :meth:`_maybe_run_goal_drift_on_task_boundary` (called from
        ``mark_task_*``) and from :meth:`note_agent_turn` (called from
        the ADK plugin's ``after_run_callback``). Both call sites run
        on the agent's ADK invocation task, which is registered with
        ``_GoldfiveADKPlugin._invocation_tasks`` for cooperative
        cancellation. A sibling drift firing
        :meth:`request_invocation_cancel(cancel_inflight_task=True)`
        could therefore land a ``CancelledError`` inside the judge's
        own ``await call_llm(...)`` — the v22 trace
        (49b0eb10-5636-465d-b96b-9e9d03d91e81) shows exactly that:
        immediately after research_panels transitioned to COMPLETED
        the ``judge_goal_drift`` span opened and failed with
        ``CancelledError`` and an empty stack, no LLM duration
        recorded.

        Detaching the judge into a separate ``asyncio.Task`` isolates
        it from the agent invocation's cancel scope: ``task.cancel()``
        on the agent's task does NOT propagate to children spawned via
        :func:`asyncio.create_task` (asyncio Tasks do not form a
        parent-child cancel tree the way ``asyncio.TaskGroup`` does).
        The judge is tracked on :attr:`_background_judges` so
        :meth:`shutdown` (called from ``Runner.close``) can drain it
        with the same bounded wait the reasoning judge uses
        (goldfive#251). Done-callback removes the entry on completion
        so there is no per-turn leak.

        No-op when no event loop is running (defensive — keeps
        synchronous test harnesses that build a steerer outside an
        async context from raising). No-op when no judge ``call_llm``
        is configured.
        """
        if self._goal_drift_call_llm is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop — fall through silently. The synchronous callers
            # of ``mark_task_*`` outside an async context (rare; only
            # tests / synthetic harnesses) won't get a goal-drift
            # check, but they wouldn't have anywhere to await the
            # judge anyway.
            return
        bg_task = asyncio.create_task(
            self._run_goal_drift_judge_background(session),
            # goldfive#243: encode session.id in the task name so
            # :meth:`drain_session_background_tasks` can filter pending
            # tasks by the run boundary that's terminating, leaving any
            # other concurrent session's tasks alone.
            name=f"goldfive-goal-drift-judge:{session.id}",
        )
        self._background_judges.add(bg_task)
        bg_task.add_done_callback(self._background_judges.discard)

    async def _run_goal_drift_judge_background(self, session: Session) -> None:
        """Body of the fire-and-forget GOAL_DRIFT judge task.

        Mirrors :meth:`_run_judge_background` (the reasoning-judge
        equivalent): swallows every exception so a flaky judge cannot
        crash the run, and re-raises ``CancelledError`` cleanly so
        :meth:`shutdown` can cancel still-running judges at teardown
        without a stray ``WARNING`` muddying the signal.

        Calls :meth:`maybe_run_goal_drift_check` directly — the public
        method's synchronous semantics are preserved for operator-side
        one-shot triggers; this background path just bypasses the
        cancellable agent task that hosted us.
        """
        try:
            await self.maybe_run_goal_drift_check(session)
        except asyncio.CancelledError:
            # Propagate so :meth:`shutdown` / teardown sees a clean
            # cancel. The shutdown path expects this and counts it
            # against the still-pending tally without warning.
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            log.warning(
                "DefaultSteerer: background goal-drift judge raised "
                "(swallowed): %s",
                exc,
            )

    # ------------------------------------------------------------------
    # iter-11A: fire-and-forget drift-cascade dispatch.
    #
    # ``mark_task_failed`` / ``mark_task_blocked`` previously awaited
    # ``_handle_drift`` inline. The cascade traverses planner.refine
    # (an LLM round-trip), supersedes integration, and downstream
    # cancellation — on a slow local LLM (e.g. Qwen3.6-35B-A3B-FP8) the
    # full chain can take 60-120s. Awaiting that from the reporting
    # tool blocked the tool's return, which blocked the agent's next
    # ADK turn end-to-end. Spawning the cascade lets the tool ack the
    # transition immediately; the cascade's side effects
    # (PlanRevised emission, supersedes, follow-up nudges) land on the
    # sink bus exactly as before, just slightly later.
    #
    # Mirrors :meth:`_spawn_goal_drift_judge_background`. ``shutdown``
    # drains :attr:`_background_drifts` symmetrically with
    # :attr:`_background_judges`.
    # ------------------------------------------------------------------
    def _spawn_drift_handler_background(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Dispatch :meth:`_handle_drift` off the critical path.

        Mirrors :meth:`_spawn_goal_drift_judge_background`. The
        reporting tool that triggered this drift returns immediately;
        the downstream cascade (refine, supersedes, cancellation)
        happens asynchronously on a tracked task.

        No-op when no event loop is running (defensive — keeps
        synchronous test harnesses that build a steerer outside an
        async context from raising; the inline-awaiting callers of
        ``mark_task_*`` outside an async context are vanishingly rare
        and cannot drive a refine round-trip anyway).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop — fall through silently. Same defensive pattern
            # as :meth:`_spawn_goal_drift_judge_background`.
            log.debug(
                "DefaultSteerer._spawn_drift_handler_background: no running "
                "loop; skipping spawn for kind=%s",
                drift.kind.value,
            )
            return
        bg_task = asyncio.create_task(
            self._run_drift_handler_background(drift, session),
            # goldfive#243: encode session.id in the task name so
            # :meth:`drain_session_background_tasks` can filter pending
            # tasks by the run boundary that's terminating, leaving any
            # other concurrent session's tasks alone.
            name=f"goldfive-drift-{drift.kind.value}:{session.id}",
        )
        self._background_drifts.add(bg_task)
        bg_task.add_done_callback(self._background_drifts.discard)

    async def _run_drift_handler_background(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Body of the fire-and-forget drift handler.

        Mirrors :meth:`_run_goal_drift_judge_background`: swallows
        every exception so a flaky cascade cannot crash the run, and
        re-raises ``CancelledError`` cleanly so :meth:`shutdown` can
        cancel still-running cascades at teardown without a stray
        ``WARNING`` muddying the signal.
        """
        try:
            await self._handle_drift(drift, session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — background task
            log.warning(
                "DefaultSteerer: background drift handler raised "
                "(swallowed): kind=%s exc=%s",
                drift.kind.value,
                exc,
            )

    async def _wait_background_drifts_idle(self) -> None:
        """Wait for every pending background drift task to settle.

        Test helper. Mirrors the goal-drift drain pattern used by
        :func:`tests.test_goal_drift_classifier._drain_background_judges`.
        Production callers should never need this — the run-end
        :meth:`shutdown` drains pending cascades with a bounded wait.
        """
        pending = list(self._background_drifts)
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
        # One yield so the ``add_done_callback(...discard)`` has run
        # and the set is fully empty for the next assertion / spawn.
        await asyncio.sleep(0)

    async def _emit_reflective_failure(
        self, session: Session, *, task_id: str, reason: str
    ) -> None:
        """Emit an INFO ``CUSTOM`` drift when the reflective check itself
        could not be interpreted.

        Uses ``CUSTOM`` (rather than a new kind) because this is not a
        property of the agent's behaviour — it's a plumbing failure in
        the reflective check. Sinks that want to surface it specifically
        can look for the ``reflective_check_failed:`` prefix on detail.
        """
        drift = DriftEvent(
            kind=DriftKind.CUSTOM,
            severity=DriftSeverity.INFO,
            detail=f"reflective_check_failed: {reason}",
            current_task_id=task_id,
        )
        # INFO drifts never trigger refine; emit directly.
        await self._emit_drift_detected(session, drift)

    # --- Reflective prompt helpers -----------------------------------

    # Liberal JSON extractor: tolerates markdown code fences and leading /
    # trailing prose around the object.
    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

    @classmethod
    def _parse_reflective_response(cls, raw: Any) -> dict[str, Any] | None:
        """Extract the first JSON object from ``raw`` or return None.

        Tolerates markdown code fences (``\\`\\`\\`json ... \\`\\`\\``) and
        prose wrapping, which real LLMs emit even with strong "reply JSON
        only" instructions. Returns ``None`` for any shape that is not a
        dict once parsed, so downstream code can check one failure mode.
        """
        if not isinstance(raw, str) or not raw.strip():
            return None
        stripped = raw.strip()
        # Fast path: parse verbatim.
        try:
            decoded = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Try extracting the first {...} block.
            match = cls._JSON_OBJECT_RE.search(stripped)
            if match is None:
                return None
            try:
                decoded = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(decoded, dict):
            return None
        return decoded

    @staticmethod
    def _summarize_recent_tool_calls(session: Session, *, limit: int = 10) -> str:
        """Build a short human-readable summary of the last N tool calls.

        Reads from ``session.recent_tool_observations`` (populated by
        :meth:`note_tool_observation` from the adapter's
        ``after_tool_callback`` / ``on_tool_error_callback`` hooks).
        Falls back to "(no recent tool calls)" when the buffer is empty.

        Each rendered entry is ``tool_name(args_preview)`` with an
        ``[ERROR: ...]`` suffix when the observation was flagged as an
        error. ``args_preview`` is already truncated to 240 chars by the
        writer; we further trim to 120 here to keep the reflective-check
        prompt bounded. The most recent ``limit`` entries are emitted
        oldest-first for readability.

        Adapters that want richer summaries can subclass and override.
        """
        hist = getattr(session, "recent_tool_observations", None) or []
        if not hist:
            return "(no recent tool calls)"
        # Take the tail (most recent ``limit`` entries) oldest-first.
        tail = list(hist)[-limit:]
        lines: list[str] = []
        for entry in tail:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("tool_name", "") or "")
            if not name:
                continue
            args_preview = str(entry.get("args_preview", "") or "")[:120]
            rendered = f"{name}({args_preview})"
            if entry.get("is_error"):
                err = str(entry.get("error_message", "") or "")[:80]
                rendered += f" [ERROR: {err}]" if err else " [ERROR]"
            lines.append(rendered)
        if not lines:
            return "(no recent tool calls)"
        return ", ".join(lines)

    @staticmethod
    def _summarize_recent_reasoning(session: Session, *, limit: int = 3) -> str:
        """Return the last ``limit`` reasoning blocks, truncated.

        Pulls directly from ``session.reasoning_history`` (populated by
        :meth:`observe_reasoning`). Each block is capped at 240 chars so
        the prompt stays bounded for long chains of thought.
        """
        hist = getattr(session, "reasoning_history", None) or []
        if not hist:
            return "(no recent reasoning)"
        tail = list(hist)[-limit:]
        trimmed = [r[:240] + ("…" if len(r) > 240 else "") for r in tail]
        return " | ".join(trimmed)

    @staticmethod
    def _steer_dedupe_id(event: Any) -> str:
        """Return the dedupe id for a STEER ``ControlMessage``, or ``""``.

        Prefers the source ``annotation_id`` when the bridge forwarded
        one (goldfive#171), falling back to the ``ControlMessage.id``
        so callers that don't source annotations still get retry dedupe.
        Returns ``""`` for non-ControlMessages, non-STEER kinds, or
        ids the bridge didn't populate — callers treat an empty id as
        "nothing to dedupe".
        """
        from goldfive.control import ControlKind, ControlMessage

        if not isinstance(event, ControlMessage):
            return ""
        raw_kind = getattr(event, "kind", None)
        kind_str = str(getattr(raw_kind, "value", raw_kind) or "").upper()
        if kind_str != ControlKind.STEER.value:
            return ""
        payload = event.payload if isinstance(event.payload, dict) else {}
        ann_id = str(payload.get("annotation_id", "") or "")
        if ann_id:
            return ann_id
        return str(getattr(event, "id", "") or "")

    @staticmethod
    def _unpack_steer_context(drift: DriftEvent) -> tuple[str, str, str]:
        """Extract ``(raw_body, author, dedupe_id)`` from a USER_STEER drift.

        Prefers the originating :class:`ControlMessage` stashed on
        :attr:`DriftEvent.raw` so the raw body survives the ``"by
        {author}: {body}"`` rewrite applied to :attr:`DriftEvent.detail`.
        When ``raw`` is absent (e.g. a test that builds a USER_STEER
        drift directly), falls back to parsing the detail string — a
        ``"by X: Y"`` prefix is treated as ``(Y, X, "")``; anything
        else becomes ``(detail, "", "")``.
        """
        from goldfive.control import ControlMessage

        raw = getattr(drift, "raw", None)
        if isinstance(raw, ControlMessage):
            payload = raw.payload if isinstance(raw.payload, dict) else {}
            body = str(payload.get("note", "") or "")
            author = str(payload.get("author", "") or "").strip()
            ann_id = str(payload.get("annotation_id", "") or "")
            dedupe_id = ann_id or str(getattr(raw, "id", "") or "")
            return body, author, dedupe_id
        # Fallback: parse "by {author}: {body}" out of detail so the
        # back-compat DriftEvent-only code path preserves the author in
        # state writes. No dedupe id is recoverable here.
        detail = str(getattr(drift, "detail", "") or "")
        if detail.startswith("by ") and ": " in detail:
            prefix, _, tail = detail.partition(": ")
            author = prefix[len("by ") :].strip()
            return tail, author, ""
        return detail, "", ""

    @classmethod
    def _is_duplicate_steer(cls, event: Any, session: Session) -> bool:
        """True when ``event`` is a STEER ControlMessage already processed.

        See :meth:`_steer_dedupe_id` for the id-selection rules. An
        empty id always returns ``False`` (nothing to compare against).
        """
        steer_id = cls._steer_dedupe_id(event)
        if not steer_id:
            return False
        return _ostate.has_processed_steer_id(session.state, steer_id)

    @staticmethod
    def _drift_from_control(event: Any, session: Session) -> DriftEvent | None:
        """Map a :class:`ControlMessage` to the matching ``USER_*`` drift.

        Returns ``None`` for anything that is not a ``ControlMessage`` so
        the caller can fall through to the classifier pipeline. Unknown
        control kinds return ``None`` as well — they are dispatched by
        the executor, not the steerer.

        For STEER, the operator ``author`` (when the bridge forwarded
        one) is prefixed onto the drift detail so downstream consumers
        — prompt templates, sinks, UI — see audit-trail attribution
        inline without having to peek into ``session.state``
        (goldfive#171). The raw body still lands on
        ``goldfive.active_steer.body`` untouched.
        """
        from goldfive.control import ControlKind, ControlMessage

        if not isinstance(event, ControlMessage):
            return None
        raw_kind = getattr(event, "kind", None)
        kind_str = str(getattr(raw_kind, "value", raw_kind) or "").upper()
        payload = event.payload if isinstance(event.payload, dict) else {}
        note = str(payload.get("note", "") or "")
        reason = str(payload.get("reason", "") or "")
        author = str(payload.get("author", "") or "").strip()
        if kind_str == ControlKind.STEER.value:
            if author:
                detail = f"by {author}: {note}"
            else:
                detail = note
            return DriftEvent(
                kind=DriftKind.USER_STEER,
                severity=DriftSeverity.WARNING,
                detail=detail,
                current_task_id=session.current_task_id,
                raw=event,
                authored_by="user",
            )
        if kind_str == ControlKind.CANCEL.value:
            return DriftEvent(
                kind=DriftKind.USER_CANCEL,
                severity=DriftSeverity.CRITICAL,
                detail=reason,
                current_task_id=session.current_task_id,
                raw=event,
                authored_by="user",
            )
        if kind_str == ControlKind.PAUSE.value:
            return DriftEvent(
                kind=DriftKind.USER_PAUSE,
                severity=DriftSeverity.INFO,
                detail=note,
                current_task_id=session.current_task_id,
                raw=event,
                authored_by="user",
            )
        return None

    def detect_drift(
        self,
        event: Any,
        session: Session,
    ) -> DriftEvent | None:
        """Classify ``event`` via the modular classifiers in :mod:`drift`.

        Classifiers are tried in order of specificity: tool-error shapes
        first (most structured), then refusal markers in text, then
        stop-reason tokens. The first match wins.

        The primitive classifiers in :mod:`goldfive.drift` (tool-error,
        refusal, stop-reason) don't take a session, so we stamp the
        observation-time plan revision (goldfive#245) here on the
        positive side of the funnel — same observation moment, same
        snapshot the call sees. The dispatch-time gate in
        :meth:`_handle_drift` then drops verdicts whose revision is
        older than the live plan's.
        """
        observed_revision_index = 0
        plan = getattr(session, "plan", None)
        if plan is not None:
            observed_revision_index = int(getattr(plan, "revision_index", 0) or 0)

        def _stamp(d: DriftEvent | None) -> DriftEvent | None:
            if d is None:
                return None
            # Only stamp when unset so explicit observation-time stamps
            # from inner classifiers win.
            if not d.observed_revision_index and observed_revision_index:
                d.observed_revision_index = observed_revision_index
            return d

        drift = _stamp(classify_tool_error(event))
        if drift is not None:
            return drift

        # Refusal scan — tolerates raw strings, dicts, objects.
        drift = _stamp(classify_refusal(event))
        if drift is not None:
            return drift

        # Stop-reason scan — prefer explicit field on dicts / objects.
        stop_reason: Any = None
        if isinstance(event, dict):
            stop_reason = event.get("stop_reason") or event.get("finish_reason")
        else:
            stop_reason = getattr(event, "stop_reason", None) or getattr(
                event, "finish_reason", None
            )
        if stop_reason is not None:
            drift = _stamp(classify_stop_reason(stop_reason))
            if drift is not None:
                return drift

        return None

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
        """Fire a ``NEW_WORK_DISCOVERED`` drift event → triggers refine."""
        detail = f"new work under {parent_task_id}: {title}: {description}" + (
            f" (assignee={assignee})" if assignee else ""
        )
        drift = DriftEvent(
            kind=DriftKind.NEW_WORK_DISCOVERED,
            severity=DriftSeverity.WARNING,
            detail=detail,
            current_task_id=parent_task_id,
            current_agent_id=assignee,
        )
        await self._handle_drift(drift, session)

    async def report_plan_divergence(
        self,
        *,
        session: Session,
        note: str,
        suggested_action: str = "",
    ) -> None:
        """No-op: PLAN_DIVERGENCE drift is disabled (goldfive#252).

        # goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH
        (#253) — disabled here. The detector path still records the
        ``divergence_flag`` so observers see "something happened", but
        no drift fires through the steerer pipeline.
        """
        session.divergence_flag = True
        detail = f"{note} (suggested: {suggested_action})" if suggested_action else note
        log.debug(
            "DefaultSteerer.report_plan_divergence: PLAN_DIVERGENCE "
            "drift disabled (goldfive#252); detector observed %r",
            detail,
        )
        return

    # ==================================================================
    # Internals
    # ==================================================================

    # --- Plan lookup --------------------------------------------------

    @staticmethod
    def _find_task(session: Session, task_id: str) -> Task | None:
        if not task_id or session.plan is None:
            return None
        for t in session.plan.tasks:
            if t.id == task_id:
                return t
        return None

    # --- Drift dispatch ----------------------------------------------

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:
        """Emit a ``DriftDetected`` event and dispatch via the intervention ladder.

        The ladder (goldfive#142) maps (drift_kind, severity,
        occurrence_count) to one of six :class:`InterventionLevel`
        values and dispatches accordingly. Level 1 (ABSORB) preserves
        the historical refine-on-WARNING behaviour; other levels
        short-circuit, queue follow-ups, or escalate to a paused state
        for user intervention.

        Refine outcomes (success / failure) from Level 1 dispatch are
        tracked per ``(drift.kind.value, drift.current_task_id)`` on
        ``session.refine_outcomes`` (goldfive#215 iter-8 P2). A prior
        ``"succeeded"`` outcome on this turn short-circuits a follow-up
        same-key drift entirely (the prior refine already addressed
        it). After :attr:`REFINE_FAILURE_THRESHOLD` consecutive
        ``"failed"`` outcomes for the same key we skip refine, mark
        the offending task ``FAILED`` (non-recoverable), and emit a
        CRITICAL ``REPEATED_FAILURE`` drift so sinks see the back-off.
        This bounds the loop that would otherwise re-fire the same
        drift every tick until ``SequentialExecutor.max_task_invocations``
        tripped (see TASK-LIFECYCLE.md). The outcome table is
        cleared on every ``run_started`` boundary.

        goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH
        (#253) — disabled here. Drop the drift at the very top of the
        handler so any external producer (legacy callers, replays,
        sinks that build raw ``DriftEvent`` instances) cannot revive
        the signal. Detection coverage moves to CAPABILITY_MISMATCH,
        which is grounded in actual agent tools rather than the
        planner's declared ``assignee_agent_id``.
        """
        # goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH
        # (#253) — disabled here. Guard at the very top so any external
        # producer (legacy callers, replays, sinks) cannot revive it.
        if drift.kind is DriftKind.PLAN_DIVERGENCE:
            log.debug(
                "DefaultSteerer._handle_drift: PLAN_DIVERGENCE drift "
                "received but handling is disabled (goldfive#252); "
                "detail=%r",
                drift.detail,
            )
            return
        # Normalise source attribution early so every downstream
        # consumer (sinks, promotion policy, prompt framing) sees a
        # non-empty ``authored_by``. USER_* kinds → "user"; anything
        # else → "goldfive". Honours an explicit non-empty value on
        # the drift (e.g. callers that already attributed) via
        # :meth:`_resolve_authored_by`.
        if not drift.authored_by:
            drift.authored_by = self._resolve_authored_by(drift)
        # goldfive#245 — verdict-freshness gate. Every observation/
        # detector stamps ``observed_revision_index`` from
        # ``session.plan.revision_index`` BEFORE its LLM await; the
        # reconciler may transition tasks during that round-trip, so a
        # verdict that arrives after the framework moved on is moot.
        # Drop it here: emit ``DriftDetected`` for observability
        # (operators see the detector ran) and skip the cancel + refine
        # machinery.
        #
        # Bypasses:
        #   * Unstamped drifts (``observed_revision_index == 0``) are
        #     legacy / external producers / pre-#245 emit paths — flow
        #     through unchanged so the gate is purely additive.
        #   * User-authored drifts (USER_STEER / USER_CANCEL /
        #     USER_PAUSE) bypass the gate even when stamped: an
        #     operator directive must be honoured regardless of the
        #     framework's plan-state cursor (preserves the iter-11D /
        #     #242 contract).
        if drift.observed_revision_index and (drift.authored_by or "").lower() != "user":
            # Per-(kind, target) addressed-watermark check — narrower
            # than naive ``observed < live_revision`` gating so parallel
            # judges firing on *orthogonal* concerns aren't over-rejected.
            # Naive gating drops a GOAL_DRIFT verdict observed at N just
            # because an unrelated OFF_TOPIC refine bumped the plan to
            # N+1; this gate drops only when the SAME (kind, target) was
            # already addressed at a later revision (genuinely redundant).
            #
            # ``last_addressed_revision_by_drift_key`` is stamped by
            # :meth:`_apply_revision` after every successful goldfive-
            # authored refine. Empty target (``""``) coalesces trajectory-
            # level drifts on one key, so trajectory-wide addressing
            # works correctly.
            key = (drift.kind.value, drift.current_task_id or "")
            last_addressed = int(
                session.last_addressed_revision_by_drift_key.get(key, 0)
            )
            if last_addressed and drift.observed_revision_index < last_addressed:
                log.info(
                    "DefaultSteerer._handle_drift: redundant verdict — "
                    "drift kind=%s target=%r observed revision %d but "
                    "same (kind, target) was already addressed at "
                    "revision %d; skipping dispatch",
                    drift.kind.value,
                    drift.current_task_id or "<trajectory>",
                    drift.observed_revision_index,
                    last_addressed,
                )
                # Emit for observability so operators see the detector
                # ran; do NOT cancel / refine on a redundant view.
                await self._emit_drift_detected(session, drift)
                return
        # Tag the bound adapter's next cancel with a symbolic reason so
        # the synthetic function_response the adapter appends on cancel
        # carries LLM-actionable content. Done BEFORE the drift event
        # is emitted so a sink that reacts by cancelling the invoke
        # sees the tag. Harmless if the adapter doesn't expose the
        # attribute (duck-typed) or no adapter is bound. See
        # goldfive#139 and
        # :func:`goldfive.adapters.adk._build_cancelled_response_event`.
        self._tag_adapter_cancel_reason(drift, session=session)
        # goldfive#152: USER_STEER-specific side effects -- write the
        # active-steer bookkeeping onto the orchestration-state dict
        # and synthesize a durable Goal from the steer body so
        # subsequent refines see the pivot as a first-class goal,
        # not a one-shot user message. Done BEFORE the drift event
        # is emitted (the state writes are cheap) and BEFORE the
        # ladder dispatches to planner.refine (which reads
        # ``session.goals`` we just mutated) so the refine sees the
        # new goal shape in the same dispatch.
        if drift.kind is DriftKind.USER_STEER:
            # Plumb the session into the planner's span-context provider
            # for the duration of synthesize_goal_from_steer so its LLM
            # call shows up as a span on the Gantt. Cleared in a
            # ``finally`` so exceptions don't leave a stale pointer.
            _token = self._active_session_var.set(session)
            try:
                await self._apply_user_steer_state(drift, session)
            finally:
                self._active_session_var.reset(_token)
        # goldfive-steer-unification: consult the severity-aware
        # promotion policy BEFORE emitting DriftDetected so that a
        # suppressed goldfive steer carries the ``suppressed_by_user_steer``
        # flag on the wire (sinks can surface the suppression
        # decision). ``_should_promote_to_steer`` returns ``True`` iff
        # the drift is goldfive-authored, clears the configured
        # severity threshold, and is not blocked by an active fresh
        # user steer; as a side effect it stamps
        # ``drift.suppressed_by_user_steer=True`` when the suppression
        # path wins.
        promote_to_steer = self._should_promote_to_steer(drift, session)
        await self._emit_drift_detected(session, drift)
        if drift.suppressed_by_user_steer:
            # Suppression path: the goldfive drift fired, was observed
            # via DriftDetected, and — per the fresh user-steer
            # suppression window — we neither cancel nor refine. The
            # pre-unification passive ladder dispatch is also skipped:
            # a user steer is already active, its refine has already
            # happened, and running another refine for this signal
            # would race against it.
            return
        # Cooperative cancellation (goldfive#251 Stream C / 7a). Severity
        # ladder decision: CRITICAL drifts (and ONLY critical drifts)
        # flag the currently-active invocation(s) for cooperative
        # cancel before the refine / promote path runs. INFO + WARNING
        # severities do NOT cancel — they flow through the usual
        # observe / absorb / nudge channels. User-authored drifts
        # (USER_STEER / USER_CANCEL / USER_PAUSE) additionally bypass
        # the severity gate because an operator directive must be
        # honoured even when emitted at a lower severity tier.
        #
        # The actual short-circuit happens in the ADK plugin's next
        # ``before_agent_callback`` / ``before_model_callback`` /
        # ``before_tool_callback``; this call just writes the flag.
        # Whether to re-dispatch after the cancel is the parent
        # agent's decision, informed by plan-causal prompting from
        # Stream B — the framework itself does NOT auto-reinvoke.
        if self._should_request_cancel_for_drift(drift):
            try:
                await self.request_invocation_cancel(drift=drift, session=session)
            except Exception as exc:  # noqa: BLE001 — cancel is best-effort
                log.debug(
                    "DefaultSteerer._handle_drift: request_invocation_cancel raised: %s",
                    exc,
                )
        if promote_to_steer:
            await self._promote_drift_to_steer(drift, session)
            return
        # Route through the intervention ladder. The per-(kind, task)
        # occurrence count drives the "first vs repeat" distinction in
        # the ladder table -- we read it BEFORE any mutation so the
        # mapping sees the state at drift-fire time. ``occurrence_count``
        # is derived from ``session.refine_outcomes`` via
        # :meth:`_occurrence_count_for_ladder` (goldfive#215 iter-8 P2:
        # the outcome dict replaces the deleted ``refine_failure_counts``
        # int counter).
        occurrence_count = self._occurrence_count_for_ladder(session, drift)
        level = self._ladder_level_for(drift.kind, drift.severity, occurrence_count)
        log.debug(
            "DefaultSteerer._handle_drift: kind=%s severity=%s occurrence=%d -> level=%s",
            drift.kind.value,
            drift.severity.value,
            occurrence_count,
            level.name,
        )
        if level is InterventionLevel.OBSERVE:
            return
        if level is InterventionLevel.NUDGE:
            await self._dispatch_nudge(drift, session)
            return
        if level is InterventionLevel.PAUSE_ESCALATE:
            await self._dispatch_pause_escalate(drift, session)
            return
        if level is InterventionLevel.TERMINATE:
            # Level 5 is reserved for a future Runner-side timeout on a
            # stuck Level 4 pause. Today we fall back to PAUSE_ESCALATE
            # so no code path silently drops the drift.
            await self._dispatch_pause_escalate(drift, session)
            return
        # ABSORB and CANCEL_REINVOKE both call ``planner.refine`` and
        # install the revised plan. CANCEL_REINVOKE additionally queues
        # a corrective message on the session for the overlay loop
        # (goldfive#141). The refine call itself is identical so we
        # share the implementation below and read the level at the end
        # to decide whether to emit the follow-up handoff.
        if self._planner is None or session.plan is None:
            return
        if drift.kind is DriftKind.REFINE_VALIDATION_FAILED:
            # Terminal planner signal (goldfive#133). Do NOT call refine
            # again on it. The ladder already routes this to Level 4 so
            # control flow normally won't reach here, but belt-and-braces.
            return
        # Outcome-based gate (goldfive#215 iter-8 P2 — unified G1+G3).
        # Skip refine when ``(kind, task)`` already has a terminal
        # outcome on this turn: a prior refine already succeeded (the
        # current drift is a same-turn replay of an addressed
        # condition), or prior refines have failed
        # >= REFINE_FAILURE_THRESHOLD times (the threshold trip already
        # marked the task FAILED + emitted REPEATED_FAILURE; a third
        # tick must not retry). USER_STEER / USER_CANCEL / GOAL_DRIFT
        # bypass the gate — operator intent always honoured,
        # trajectory drifts have their own rate limiters.
        if drift.kind not in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            outcome_key = (drift.kind.value, drift.current_task_id or "")
            outcome = session.refine_outcomes.get(outcome_key)
            if outcome is not None:
                if outcome.state == "succeeded":
                    log.debug(
                        "refine skipped: prior succeeded outcome (kind=%s task=%r)",
                        drift.kind.value,
                        drift.current_task_id,
                    )
                    return
                if outcome.fail_count >= self.REFINE_FAILURE_THRESHOLD:
                    log.debug(
                        "refine skipped: failure threshold reached (kind=%s task=%r count=%d)",
                        drift.kind.value,
                        drift.current_task_id,
                        outcome.fail_count,
                    )
                    return
        # Progress-based escalation (goldfive#271). Orthogonal to the
        # outcome gate: a task that has been silent past the configured
        # stall threshold escalates to HUMAN_INTERVENTION_REQUIRED
        # instead of looping the planner. A productively-iterating task
        # has continuous progress events; a stuck task does not.
        if self._is_task_progress_stalled(drift, session):
            await self._emit_progress_stalled_escalation(drift, session)
            return
        # Plumb the session into the planner's drift-emitter callback
        # for the duration of this refine call so the planner can emit
        # REFINE_VALIDATION_FAILED drifts through the normal event
        # pipeline. Cleared in a ``finally`` so exceptions don't leave
        # a stale session pointer. ContextVar isolation keeps concurrent
        # runs from stomping each other (goldfive#133, PR #294 audit).
        _active_session_token = self._active_session_var.set(session)
        # goldfive a4: mint a refine-attempt id for correlation across
        # ``refine_attempted`` and the paired success/failure event.
        attempt_id = self._new_attempt_id()
        await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
        try:
            # Thread the adapter's available_agents_tree (goldfive#151)
            # through refine so the LLM is constrained to pick real
            # tree assignees. Adapters without the property fall back
            # to ``available_agents`` (list[str]); custom/legacy adapters
            # without either surface produce ``None`` and the planner
            # keeps its pre-#151 behaviour. Planners whose refine does
            # not accept the kwarg (test stubs, pre-#151 custom
            # planners) are called the old way so nothing breaks.
            available_agents: Any = None
            adapter = self._adapter
            if adapter is not None:
                tree = getattr(adapter, "available_agents_tree", None)
                if isinstance(tree, list) and tree:
                    available_agents = list(tree)
                else:
                    flat = getattr(adapter, "available_agents", None)
                    if flat:
                        available_agents = list(flat)
            refine_accepts_registry = _planner_refine_accepts_available_agents(self._planner)
            # Phase 3.5 (goldfive#271) tripwire wrapper — the
            # ``except BaseException: stash; raise`` arm below is the
            # compliance branch (CANCELLATION-CONTRACT.md §1.2). The
            # boundary catch site at ``ADKAdapter._invoke_internal``
            # asserts ``mark_stash_completed()`` fired before the
            # cancel propagated past us.
            with _state_audit.cancellation_stash_audited("DefaultSteerer._handle_drift.refine"):
                try:
                    if refine_accepts_registry:
                        revised = await self._planner.refine(
                            plan=session.plan,
                            drift=drift,
                            goals=list(session.goals),
                            available_agents=available_agents,
                        )
                    else:
                        revised = await self._planner.refine(
                            plan=session.plan,
                            drift=drift,
                            goals=list(session.goals),
                        )
                except RefineExhausted as exc:
                    # goldfive#271: planner explicitly signals it cannot
                    # produce a meaningful change. Same escalation path
                    # as the structural no-op detector — pause for
                    # human intervention rather than retrying.
                    log.info(
                        "DefaultSteerer._handle_drift: planner.refine raised "
                        "RefineExhausted for kind=%s task=%r: %s",
                        drift.kind.value,
                        drift.current_task_id,
                        exc,
                    )
                    await self._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="refine_exhausted",
                        reason=str(exc) or "planner signalled handler exhaustion",
                        detail="",
                    )
                    await self._emit_handler_exhausted_escalation(drift, session)
                    return
                except Exception as exc:  # noqa: BLE001 — refine errors must not break the run
                    # Surface the failure via logging + a synthetic follow-up
                    # drift so operators don't silently see the same plan loop
                    # forever. Without this, a refine that raises (e.g. malformed
                    # LLM JSON after a mid-invocation cancel poisons the session)
                    # leaves session.plan unchanged and the executor re-enters
                    # the same state on the next tick.
                    log.warning(
                        "DefaultSteerer._handle_drift: planner.refine(kind=%s) raised "
                        "%s; plan unchanged",
                        drift.kind.value,
                        exc,
                    )
                    await self._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="llm_error",
                        reason=str(exc),
                        detail=type(exc).__name__,
                    )
                    await self._escalate_refine_failure_as_critical_drift(
                        session, drift, reason=str(exc)
                    )
                    await self._record_refine_outcome(session, drift, succeeded=False)
                    return
                except BaseException as exc:  # noqa: BLE001
                    # Phase 3.5 (CANCELLATION-CONTRACT.md §C4): ``CancelledError``
                    # bypasses the ``except Exception`` branch (it is a
                    # ``BaseException`` since Py 3.8). Emit the paired
                    # ``refine_failed`` observability event so a refine cancelled
                    # mid-flight does not leave sinks with an unmatched
                    # ``refine_attempted``. The ``finally`` below still resets
                    # ``_active_session_var``; we only own the paired-event
                    # stash here. Re-raise so cancellation continues to
                    # propagate per the asyncio contract.
                    await self._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="cancelled",
                        reason=f"refine cancelled: {type(exc).__name__}",
                        detail=type(exc).__name__,
                    )
                    # Phase 3.5 tripwire compliance marker (§1.2 form).
                    _state_audit.mark_stash_completed()
                    raise
        finally:
            self._active_session_var.reset(_active_session_token)
        if revised is None:
            # iter-12 (#204): refine returning None at the steerer level
            # means the planner has already exhausted its internal retry
            # budget (iter-11C's repeat-rejection guard). Treat as
            # handler exhaustion and pause for human intervention rather
            # than emitting a follow-up CRITICAL drift that would
            # recurse through ``_handle_drift`` and eventually abort the
            # run. Mirrors the ``RefineExhausted`` and no-op-revision
            # escalation paths.
            log.warning(
                "DefaultSteerer._handle_drift: planner.refine(kind=%s) returned None; "
                "plan unchanged — escalating to HUMAN_INTERVENTION_REQUIRED",
                drift.kind.value,
            )
            await self._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="parse_error",
                reason="planner returned no revised plan",
                detail="",
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # I4 fix: fold runtime terminal statuses from the prior plan
        # onto the revised plan BEFORE validation. A task that was
        # cancelled / failed / NOT_NEEDED out-of-band between revisions
        # (e.g. overlay reap → NOT_NEEDED, executor reachability audit
        # → CANCELLED, coordinator reporting-tool → COMPLETED) should
        # carry that status into the persisted snapshot, even when the
        # LLM's view of the prior plan was stale.
        # goldfive#247: fold returns a NEW Plan (Plan is frozen).
        revised = self._fold_runtime_terminal_statuses(revised, session.plan)
        try:
            revised.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            # iter-12 (#204): the revised plan is structurally invalid
            # AND the planner has already exhausted its internal
            # validator-rejection retry budget (iter-11C). Treat as
            # handler exhaustion and pause for human intervention.
            #
            # Operator visibility: the SCHEMA_VIOLATION drift is
            # preserved at INFO severity (observability-only — does NOT
            # recurse through ``_handle_drift``) so harmonograf and
            # other sinks still see the schema-failure signal carrying
            # the validator's reason. The actionable signal is the
            # paired ``refine_failed(validator_rejected)`` envelope and
            # the HUMAN_INTERVENTION_REQUIRED escalation that follows.
            #
            # Passing ``prior=session.plan`` to ``validate`` enables
            # PLAN-LIFECYCLE.md §3.1 (terminal task preservation) and
            # §3.2 (terminal->terminal edge preservation) on top of the
            # usual structural checks.
            await self._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="validator_rejected",
                reason=f"plan validation failed: {exc}",
                detail=type(exc).__name__,
            )
            await self._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.INFO,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                ),
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # No-op revision rejection (goldfive#271 — replaces the deleted
        # count cap). If the LLM produced a "refine" that is structurally
        # identical to the prior plan (same task ids, edges, assignees,
        # statuses), treat the handler as exhausted: the planner cannot
        # produce a meaningful change for this drift. Escalate to
        # HUMAN_INTERVENTION_REQUIRED rather than bumping the revision
        # index for a no-op, which would otherwise loop forever on a
        # judge that keeps re-firing on a corrected task.
        if self._plans_structurally_identical(session.plan, revised):
            log.info(
                "no-op revision skipped (kind=%s task=%r); escalating to "
                "HUMAN_INTERVENTION_REQUIRED",
                drift.kind.value,
                drift.current_task_id,
            )
            await self._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="no_op_revision",
                reason="planner returned structurally identical plan",
                detail="",
            )
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # Successful refine — record the "succeeded" outcome so a
        # follow-up same-(kind, task) drift on this turn skips refine
        # (the prior refine already addressed it).
        await self._record_refine_outcome(session, drift, succeeded=True)
        # Capture the outgoing plan BEFORE _apply_revision installs the
        # revised one; _emit_plan_revised diffs the two to populate the
        # PlanRevisionDiff sidecar (PLAN-LIFECYCLE.md §2, §8 gap #4).
        prev_plan = session.plan
        # goldfive#247: _apply_revision returns the stamped instance.
        revised = self._apply_revision(session, revised, drift)
        # Cancel the in-flight coordinator invocation now that the plan
        # it was reasoning against has been superseded (goldfive#271
        # follow-up — v15 concurrent-invocation bug). Order: cancel
        # BEFORE PlanRevised emit so the synthetic InvocationCancelled
        # sink event lands adjacent to the revision in the wire log
        # and operators can correlate the two. Best-effort, never
        # raises — a no-op cancel still leaves the new plan installed.
        await self._cancel_inflight_for_revision(drift, session)
        await self._emit_plan_revised(
            session, revised, drift, prev_plan=prev_plan, attempt_id=attempt_id
        )
        # Level 3 (CANCEL_REINVOKE) handoff (Phase 2 of the path-
        # duality fix). Pre-Phase-2 this stuffed
        # ``session.pending_corrective_message`` — a write-only slot
        # nobody read after the overlay loop took shape, leaving the
        # coordinator running its original chain blind to the plan
        # swap. Phase 2 dispatches a ``GOLDFIVE_STEER`` ControlMessage
        # so the executor's invoke loop cancels in-flight work and
        # restarts with the corrective body framed as ``[GOLDFIVE
        # STEERING CONTROL …]`` — the same junction USER_STEER uses.
        if level is InterventionLevel.CANCEL_REINVOKE:
            await self._dispatch_goldfive_steer_control(drift, session)
        # goldfive#202: for drifts where the coordinator has no way to
        # observe the plan revision on its own (it is still mid-
        # invocation, retrying the superseded task), ALSO queue a
        # Level 2 nudge after a successful ABSORB. The overlay loop's
        # scoped nudge-replay path (see SequentialExecutor._run_overlay)
        # picks this up at invocation end and re-invokes the
        # passthrough with the nudge as the next user message — the
        # only way for the coordinator to learn its plan changed.
        #
        # Scoped to drift kinds whose mid-invocation signature is
        # "coordinator is stuck on a task goldfive just replaced":
        # LOOPING_REASONING / LOOPING_TOOL_CALL (detector fires while
        # the coordinator retries the same tool call), SELF_REPORTED_STUCK
        # (reflective self-check reports no progress). Other ABSORB
        # kinds (CONFABULATION_RISK, etc.) do not need mid-invocation
        # rescue — their corrective path fires at the next task
        # boundary or via Level 3 CANCEL_REINVOKE.
        if level is InterventionLevel.ABSORB and drift.kind in _ABSORB_NUDGE_KINDS:
            nudge_msg = compose_corrective_user_message(
                drift=drift,
                refined_plan=session.plan,
            )
            session.pending_nudges.append(nudge_msg)
            log.debug(
                "DefaultSteerer._handle_drift: queued post-ABSORB nudge for kind=%s task=%s: %s",
                drift.kind.value,
                drift.current_task_id or "-",
                nudge_msg,
            )

    # --- Intervention ladder -----------------------------------------
    #
    # Mapping table for :meth:`_ladder_level_for`. Keys are
    # :class:`DriftKind` values. Each value is a 3-tuple
    # ``(info_level, warning_level, critical_level)`` holding the level
    # for that severity tier. CRITICAL uses a (first, repeat) pair --
    # repeat applies once the refine-failure counter has crossed
    # :attr:`REFINE_FAILURE_THRESHOLD` for the (kind, task) pair. A
    # level of ``None`` means "not applicable at this severity -- drop
    # through to OBSERVE". See goldfive#142 for the rationale.
    #
    # Drifts without an entry here fall through to a conservative
    # default that preserves the pre-ladder behaviour: WARNING -> ABSORB,
    # CRITICAL -> ABSORB first, CANCEL_REINVOKE on repeat.
    #
    # A note on INFO-tier preservation: the pre-ladder steerer
    # short-circuited every INFO-severity drift at "no refine" (the
    # early return below ``_severity_ge(drift.severity, WARNING)``).
    # The ladder preserves that invariant by mapping every INFO entry
    # to :data:`InterventionLevel.OBSERVE`. The issue's suggested
    # table labels some INFO tiers as "Level 1 (absorb)" but a Level 1
    # at INFO would trigger refine for every INFO hint
    # (CONFABULATION_RISK, etc.), which regresses existing behaviour.
    # If an operator later wants a refine-on-hint policy, they
    # subclass and override :meth:`_ladder_level_for`.
    _LADDER: dict[
        DriftKind,
        tuple[
            InterventionLevel | None,  # INFO
            InterventionLevel | None,  # WARNING
            tuple[InterventionLevel, InterventionLevel],  # CRITICAL (first, repeat)
        ],
    ] = {
        DriftKind.CONFABULATION_RISK: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.AGENT_REFUSAL: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.MODEL_REFUSAL: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # LOOPING_REASONING: severity is now graduated (goldfive#204)
        # -- the tool-loop detector emits INFO / WARNING / CRITICAL
        # based on count + category (meta vs work). The ladder mirrors
        # that graduation:
        #
        # * INFO  -> OBSERVE (default fallback; benign meta-tool retries
        #   at the first threshold should not mutate the plan).
        # * WARNING -> ABSORB (refine plan; unchanged from pre-#204).
        # * CRITICAL first -> NUDGE (refine AND queue a soft corrective
        #   follow-up for the overlay loop -- Agent B wires the nudge
        #   consumption in goldfive#forward-progress).
        # * CRITICAL repeat -> PAUSE_ESCALATE (escalate to human if the
        #   loop persists past nudge).
        DriftKind.LOOPING_REASONING: (
            None,
            InterventionLevel.ABSORB,
            (InterventionLevel.NUDGE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.LOOPING_TOOL_CALL: (
            None,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.REASONING_CLUSTER_TIGHTENING: (
            InterventionLevel.OBSERVE,
            None,
            (InterventionLevel.OBSERVE, InterventionLevel.OBSERVE),
        ),
        DriftKind.PLAN_DIVERGENCE: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # OFF_TOPIC is plan-context drift from the reasoning judge — the
        # agent is reasoning about something that doesn't fit the bound
        # task. Mirrors PLAN_DIVERGENCE's ladder mapping so the ABSORB
        # path engages the goal-aware refine prompt (planner.refine
        # routes both PLAN_DIVERGENCE and OFF_TOPIC through
        # ``_PLAN_DIVERGENCE_SYSTEM_PROMPT``). Without this entry the
        # default fallback (WARNING -> ABSORB) still triggered refine
        # but the planner picked the generic ``_REFINE_SYSTEM_PROMPT``
        # which has no goal-alignment guidance and could silently
        # absorb off-goal reasoning into the plan.
        DriftKind.OFF_TOPIC: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # JUSTIFIED_DEVIATION (iter-10 PR 3+4) is plan-context drift
        # WITH provenance — the agent saw a real provoking signal (tool
        # error, surprising result, discovered dependency, new
        # information) that pulled it off the bound task. The refine
        # path is identical to OFF_TOPIC (goal-aware ABSORB/REJECT via
        # ``_PLAN_DIVERGENCE_SYSTEM_PROMPT``), but we never escalate:
        # CRITICAL repeat does NOT escalate to PAUSE_ESCALATE because a
        # provoked deviation is the right input for plan-extension at
        # every severity. Penalising it would punish the agent for
        # responding to reality.
        #
        # Backstops against runaway justified_deviation are upstream of
        # the ladder, not on it (per design §7.3):
        #
        # * The per-(kind, task_id) refine cooldown ``_is_plan_revision_
        #   gated`` collapses a cluster of JUSTIFIED_DEVIATION drifts on
        #   the same task to one refine attempt; subsequent ones drop.
        # * The ``task_last_progress_at`` stall gate
        #   (``_is_task_progress_stalled``) catches the pathological
        #   case where the agent KEEPS justifying drift on the same
        #   task without making progress, escalating to
        #   HUMAN_INTERVENTION_REQUIRED.
        DriftKind.JUSTIFIED_DEVIATION: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.ABSORB, InterventionLevel.ABSORB),
        ),
        DriftKind.INTENT_DIVERGENCE: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.PAUSE_ESCALATE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.TOOL_ERROR: (
            InterventionLevel.OBSERVE,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.RUNAWAY_DELEGATION: (
            None,
            None,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        DriftKind.REFINE_VALIDATION_FAILED: (
            None,
            None,
            (InterventionLevel.PAUSE_ESCALATE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # HUMAN_INTERVENTION_REQUIRED is always CRITICAL. First fire =>
        # PAUSE_ESCALATE. A repeat fire (occurrence_count crosses the
        # threshold) escalates to TERMINATE because it means the pause
        # was already issued and the situation didn't resolve.
        DriftKind.HUMAN_INTERVENTION_REQUIRED: (
            None,
            None,
            (InterventionLevel.PAUSE_ESCALATE, InterventionLevel.TERMINATE),
        ),
        # GOAL_DRIFT (goldfive#143) -- trajectory-level judgment that
        # the tree is no longer advancing ``session.goals``. The judge's
        # signal in practice is "coordinator is looping on completed
        # work; should advance to the next hand-off."
        #
        # Tier 1 / F4 (loop prevention): NUDGE-first, not PAUSE-first.
        # ABSORB-side refine cannot recover structural goal drift —
        # the plan is usually correct and the agent is stuck — but
        # NUDGE *doesn't* refine the plan; it injects a corrective user
        # message via ``compose_corrective_user_message`` that re-
        # anchors the LLM on the next hand-off. Repeat occurrence
        # escalates to CANCEL_REINVOKE (cancel + restart with the
        # corrective body); only after CANCEL_REINVOKE didn't break
        # the loop do we fall through to PAUSE_ESCALATE in the default
        # ladder path. WARNING is also routed to NUDGE (the judge
        # rarely emits WARNING but if it does, the corrective is the
        # same shape).
        DriftKind.GOAL_DRIFT: (
            None,
            InterventionLevel.NUDGE,
            (InterventionLevel.NUDGE, InterventionLevel.CANCEL_REINVOKE),
        ),
        # Self-reported stuck (from reflective check). WARNING by
        # default -- preserve pre-ladder behaviour (ABSORB / refine).
        DriftKind.SELF_REPORTED_STUCK: (
            None,
            InterventionLevel.ABSORB,
            (InterventionLevel.CANCEL_REINVOKE, InterventionLevel.PAUSE_ESCALATE),
        ),
        # BLOCKED, NEW_WORK_DISCOVERED, and user-control derived drifts
        # go through the default fallback below (WARNING -> ABSORB,
        # CRITICAL -> ABSORB first, PAUSE_ESCALATE on repeat), which
        # matches the pre-ladder refine-on-WARNING behaviour verbatim.
    }

    # Ladder entries keyed by drift-kind *value* so future siblings can
    # register entries without depending on a newly-introduced enum
    # member. Consulted BEFORE the enum-keyed table in
    # :meth:`_ladder_level_for` and takes precedence when both match.
    # Empty today -- goldfive#143's GOAL_DRIFT has landed as a real
    # enum member, so the enum-keyed table suffices. Left in place for
    # future cross-PR coordination.
    _LADDER_BY_VALUE: dict[
        str,
        tuple[
            InterventionLevel | None,
            InterventionLevel | None,
            tuple[InterventionLevel, InterventionLevel],
        ],
    ] = {}

    def _ladder_level_for(
        self,
        kind: DriftKind,
        severity: DriftSeverity,
        occurrence_count: int,
    ) -> InterventionLevel:
        """Return the intervention level for ``(kind, severity, count)``.

        The mapping is the single source of truth for the ladder table
        documented on this module's docstring and goldfive#142. Drifts
        with no explicit entry fall through to a safe default that
        preserves the pre-ladder behaviour:

        * INFO -> OBSERVE (no action)
        * WARNING -> ABSORB (refine)
        * CRITICAL, first occurrence -> ABSORB (refine)
        * CRITICAL, repeat occurrence -> PAUSE_ESCALATE

        Subclasses can override this method to tune the table without
        re-implementing :meth:`_handle_drift`.
        """
        entry = self._LADDER_BY_VALUE.get(kind.value)
        if entry is None:
            entry = self._LADDER.get(kind)
        is_repeat = occurrence_count >= self.REFINE_FAILURE_THRESHOLD
        if entry is not None:
            info_level, warning_level, critical_pair = entry
            if severity is DriftSeverity.INFO:
                return info_level or InterventionLevel.OBSERVE
            if severity is DriftSeverity.WARNING:
                return warning_level or InterventionLevel.OBSERVE
            # CRITICAL
            return critical_pair[1] if is_repeat else critical_pair[0]
        # Default fallback for drifts not explicitly in the table.
        if severity is DriftSeverity.INFO:
            return InterventionLevel.OBSERVE
        if severity is DriftSeverity.WARNING:
            return InterventionLevel.ABSORB
        # CRITICAL with no explicit entry -- ABSORB first, escalate on repeat.
        return InterventionLevel.PAUSE_ESCALATE if is_repeat else InterventionLevel.ABSORB

    async def _dispatch_nudge(self, drift: DriftEvent, session: Session) -> None:
        """Level 2 dispatch: queue a soft follow-up message on the session.

        The Runner's overlay loop (goldfive#141) picks up the queued
        nudge at the next invocation boundary and sends it as a gentle
        corrective user message. Until #141 lands, the queue is
        observable but inert; nothing consumes it.
        """
        msg = compose_corrective_user_message(
            drift=drift,
            refined_plan=session.plan,
        )
        session.pending_nudges.append(msg)
        log.debug(
            "DefaultSteerer: queued nudge for kind=%s task=%s: %s",
            drift.kind.value,
            drift.current_task_id or "-",
            msg,
        )

    async def _dispatch_goldfive_steer_control(
        self,
        drift: DriftEvent,
        session: Session,
        *,
        body_override: str = "",
    ) -> bool:
        """Mint and dispatch a ``GOLDFIVE_STEER`` ControlMessage.

        Phase 2 of the path-duality fix. Replaces the dead
        ``session.pending_corrective_message`` write at every
        CANCEL_REINVOKE / promote-to-steer site so goldfive-authored
        drift rides the same cancel-and-restart junction as
        user-authored ``STEER``.

        ``body_override``: optional text to use as the corrective
        body. The promotion path passes its already-composed
        :meth:`_compose_goldfive_steer_body` output; the Level 3
        CANCEL_REINVOKE path leaves it empty and falls back to
        :func:`compose_corrective_user_message` against the freshly
        revised plan.

        Returns ``True`` on successful dispatch, ``False`` on no
        bound channel / send failure (best-effort — see
        :meth:`_dispatch_goldfive_control`).
        """
        from goldfive.control import ControlKind, ControlMessage

        if body_override:
            body = body_override
        else:
            body = compose_corrective_user_message(
                drift=drift,
                refined_plan=session.plan,
            )
        superseded_ids = (
            [str(drift.current_task_id)] if drift.current_task_id else []
        )
        # Replacement task ids: pick the first PENDING task on the
        # revised plan as the natural successor — the executor uses
        # this to render an explicit "pick these up instead" block in
        # the restart message.
        replacement_ids: list[str] = []
        plan = session.plan
        if plan is not None:
            for task in plan.tasks:
                if task.status is TaskStatus.PENDING and task.id:
                    replacement_ids.append(task.id)
                    break
        msg = ControlMessage(
            kind=ControlKind.GOLDFIVE_STEER,
            payload={
                "drift_kind": drift.kind.value,
                "drift_id": str(getattr(drift, "id", "") or ""),
                "body": body,
                "superseded_task_ids": superseded_ids,
                "replacement_task_ids": replacement_ids,
            },
        )
        landed = await self._dispatch_goldfive_control(msg)
        log.debug(
            "DefaultSteerer._dispatch_goldfive_steer_control: "
            "kind=%s task=%s landed=%s",
            drift.kind.value,
            drift.current_task_id or "-",
            landed,
        )
        return landed

    async def _dispatch_goldfive_pause_control(
        self,
        drift: DriftEvent,
        session: Session,
        *,
        reason: str,
    ) -> bool:
        """Mint and dispatch a ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage.

        Phase 2 of the path-duality fix. Replaces the dead
        ``session.paused_for_human_intervention = True`` flag-set at
        every Level-4 / progress-stall / handler-exhausted escalation
        site so the executor's pre-task loop blocks via the same
        channel state as a user-issued ``PAUSE``.

        Returns ``True`` on successful dispatch, ``False`` on no
        bound channel / send failure.
        """
        from goldfive.control import ControlKind, ControlMessage

        msg = ControlMessage(
            kind=ControlKind.GOLDFIVE_PAUSE_ESCALATE,
            payload={
                "reason": reason,
                "drift_id": str(getattr(drift, "id", "") or ""),
                "drift_kind": drift.kind.value,
            },
        )
        landed = await self._dispatch_goldfive_control(msg)
        log.debug(
            "DefaultSteerer._dispatch_goldfive_pause_control: "
            "kind=%s task=%s landed=%s reason=%r",
            drift.kind.value,
            drift.current_task_id or "-",
            landed,
            reason,
        )
        return landed

    async def _dispatch_pause_escalate(
        self,
        drift: DriftEvent,
        session: Session,
    ) -> None:
        """Level 4 dispatch: emit HUMAN_INTERVENTION_REQUIRED and pause.

        Does NOT call ``planner.refine`` -- Level 4 signals that the
        planner cannot recover. Phase 2 of the path-duality fix:
        dispatches a ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage on the
        bound channel so the executor's pre-task loop blocks via the
        same channel state as a user ``PAUSE``. Pre-Phase-2 this
        flipped ``session.paused_for_human_intervention = True`` — a
        flag the executor read on its next iteration; the indirection
        was synonymous with the channel signal but parallel-tracked
        from the user-PAUSE path.

        Emits a CRITICAL ``HUMAN_INTERVENTION_REQUIRED`` drift so
        sinks / the UI can surface the pause and let the user decide
        what to do.

        When the drift reaching Level 4 is *already* a
        ``HUMAN_INTERVENTION_REQUIRED`` (e.g. landed here via the
        generic fallback), we pause but do not re-emit the same drift
        a second time -- the original DriftDetected emission at the
        top of :meth:`_handle_drift` already carried the signal.
        """
        await self._dispatch_goldfive_pause_control(
            drift,
            session,
            reason=(
                f"pause_escalate from {drift.kind.value}: {drift.detail}"
                if drift.detail
                else f"pause_escalate from {drift.kind.value}"
            ),
        )
        if drift.kind is DriftKind.HUMAN_INTERVENTION_REQUIRED:
            # Already emitted at the top of _handle_drift; just pause.
            return
        # goldfive#271 PR1 — close the originating condition with
        # ``human_intervention_required`` so consumers tracking the
        # condition_id see the terminal lifecycle on the *original*
        # condition, not just on the synthesized HUMAN_INTERVENTION
        # row. The originating drift was already emitted at the top of
        # ``_handle_drift`` (legacy path) under its own condition_id;
        # this call swaps that condition's recorded lifecycle so a
        # later get_active_drift returns the terminal state.
        try:
            origin_cid = _ostate.compute_condition_id(
                kind=drift.kind,
                task_id=str(getattr(drift, "current_task_id", "") or ""),
                agent_id=str(getattr(drift, "current_agent_id", "") or ""),
                turn_id=str(getattr(session, "run_id", "") or ""),
            )
            _ostate.escalate_drift_to_human_intervention(session.state, origin_cid)
        except Exception as exc:  # noqa: BLE001
            log.debug("DefaultSteerer: drift-lifecycle escalate skipped (%s)", exc)
        escalation = DriftEvent(
            kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
            severity=DriftSeverity.CRITICAL,
            detail=(
                f"escalated from {drift.kind.value}: {drift.detail}"
                if drift.detail
                else f"escalated from {drift.kind.value}"
            ),
            current_task_id=drift.current_task_id,
            current_agent_id=drift.current_agent_id,
        )
        # Emit directly; do NOT go back through _handle_drift (would
        # infinite-loop at CRITICAL).
        await self._emit_drift_detected(session, escalation)

    # Symbolic cancel-reason tags — mirror
    # :mod:`goldfive.adapters.adk` constants but duplicated as plain
    # strings here to avoid a hard import of the optional ADK adapter
    # module from the provider-agnostic steerer. Keep in sync with
    # :data:`goldfive.adapters.adk.SYMBOLIC_REASON_USER_STEER` etc.
    _ADAPTER_CANCEL_REASON_USER_STEER: str = "user_steer"

    def _tag_adapter_cancel_reason(
        self, drift: DriftEvent, *, session: Session | None = None
    ) -> None:
        """Set the next adapter cancel reason based on ``drift.kind``.

        USER_STEER drift -> ``"user_steer"``. Other kinds currently leave
        the tag unset so the adapter falls through to the generic
        content variant. Tolerates adapters that don't carry the
        attribute (no-op) and an unbound adapter (no-op). See
        goldfive#139.

        Routes the write through
        :meth:`ADKAdapter.set_next_cancel_reason` when the adapter
        exposes that helper (PR #294 audit / goldfive#271 follow-up)
        so the tag is keyed by ``session.id`` and cannot bleed across
        concurrent goldfive sessions sharing one adapter. Falls back
        to the bare attribute write for adapters / stubs that predate
        the helper.

        The goldfive-steer-unification promotion path uses a separate
        helper (:meth:`_tag_adapter_cancel_reason_for_promotion`) to
        stamp a ``"goldfive_<drift_kind>"`` reason when promoting a
        detector drift to a full steer; keeping the two call sites
        distinct avoids muddling the pre-unification tag semantics for
        unpromoted paths.
        """
        adapter = self._adapter
        if adapter is None:
            return
        if drift.kind is DriftKind.USER_STEER:
            reason = self._ADAPTER_CANCEL_REASON_USER_STEER
        else:
            return
        self._write_adapter_cancel_reason(adapter, reason, session)

    def _tag_adapter_cancel_reason_for_promotion(
        self, drift: DriftEvent, *, session: Session | None = None
    ) -> str:
        """Stamp a goldfive-specific cancel reason on the bound adapter.

        Returns the reason string stamped (or synthesised) so callers
        can record it on the session for downstream observability.
        Mirrors :meth:`_tag_adapter_cancel_reason` semantics: adapters
        without the per-session helper are tolerated.
        """
        reason = f"goldfive_{drift.kind.name.lower()}"
        adapter = self._adapter
        if adapter is None:
            return reason
        self._write_adapter_cancel_reason(adapter, reason, session)
        return reason

    @staticmethod
    def _write_adapter_cancel_reason(adapter: Any, reason: str, session: Session | None) -> None:
        """Route the cancel-reason tag through the session-aware helper.

        Falls back to the legacy bare-attribute write for adapters /
        stubs that don't expose :meth:`set_next_cancel_reason`. See
        :meth:`ADKAdapter.set_next_cancel_reason` for the rationale.
        """
        setter = getattr(adapter, "set_next_cancel_reason", None)
        if callable(setter) and session is not None:
            try:
                setter(session, reason)
                return
            except Exception as exc:  # noqa: BLE001
                log.debug("DefaultSteerer: set_next_cancel_reason raised: %s", exc)
        try:
            adapter._next_cancel_reason = reason
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer: could not tag adapter cancel reason: %s",
                exc,
            )

    async def _request_adapter_cancel(self, reason: str) -> None:
        """Invoke the optional ``adapter.request_cancel(reason)`` hook.

        goldfive#241 — a goldfive-promoted steer needs the in-flight
        LLM call to stop NOW so the contaminated reasoning / tool
        calls don't keep writing to the session while we queue the
        restart. The ADK adapter exposes :meth:`ADKAdapter.request_cancel`
        which fires ``task.cancel()`` on the asyncio task driving
        ``runner.run_async`` so the stream raises ``CancelledError``
        and the adapter's standard heal path runs with the already-
        stamped ``_next_cancel_reason`` tag.

        Optional protocol: adapters that don't implement the method
        (Claude adapter, callable adapter, test stubs without live
        invocations) keep the legacy deferred-cancel semantics —
        ``_next_cancel_reason`` is still tagged, the restart message
        is still queued, and the next executor checkpoint still
        terminates the invocation. Tolerates an unbound adapter and
        swallows every failure so a best-effort cancel cannot break
        the promotion path.
        """
        adapter = self._adapter
        if adapter is None:
            return
        fn = getattr(adapter, "request_cancel", None)
        if not callable(fn):
            return
        try:
            result = fn(reason)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._request_adapter_cancel(reason=%r): adapter raised: %s",
                reason,
                exc,
            )

    # ------------------------------------------------------------------
    # Cooperative cancellation (goldfive#251 Stream C / 7a)
    # ------------------------------------------------------------------

    def _is_late_drift_for_terminated_invocation(
        self, drift: DriftEvent, session: Session
    ) -> bool:
        """Return True iff a goldfive-authored drift's target is gone (goldfive#319).

        Background reasoning-judge tasks (goldfive#251) run off the
        critical path so the adapter's model-response callback can return
        before the LLM judge finishes. With goldfive#319's removal of the
        per-turn cancel-drain, a slow judge spawned in turn N may now
        produce its verdict in turn N+1 — well after the original agent
        invocation has terminated. Routing such a verdict through the
        cancel + ladder dispatch is a category error: it could cancel an
        unrelated invocation or trigger a refine against a plan whose
        offending step is already complete. The drift is still emitted
        on the sink (observability preserved); this guard short-circuits
        the dispatch.

        The check uses :class:`OrchestrationStore` as the live registry
        of in-flight invocations (Phase 3.5 component 1, goldfive#271).
        Two conditions count as "late":

        * **No active invocations** — every agent has finished its turn
          and any drift currently being handled is by definition stale.
        * **Cancel-pending on the session** (goldfive#242) — a previous
          drift already requested a cooperative cancel for one or more
          invocations. The active-task registry takes 4-8s to drain
          while ADK winds those invocations down; during that window
          any newly-arriving goldfive-authored drift would dispatch a
          refine against an effectively-dead session. Stamping the
          cancel-pending flag synchronously at
          :meth:`request_invocation_cancel` time closes that race.

        User-authored drifts (USER_STEER / USER_CANCEL / USER_PAUSE)
        always bypass this guard — they are forward-looking operator
        directives, not tied to a specific in-flight invocation.
        """
        # User-authored drifts always pass through. ``authored_by`` was
        # normalised at the top of :meth:`_handle_drift`.
        if (drift.authored_by or "").lower() == "user":
            return False
        try:
            from goldfive.orchestration_store import OrchestrationStore

            store = OrchestrationStore.for_session(session)
            active = store.active_invocation_ids()
            cancel_pending = store.cancel_requested_invocation_ids()
        except Exception as exc:  # noqa: BLE001 — defensive
            log.debug(
                "DefaultSteerer._is_late_drift_for_terminated_invocation: "
                "active_invocation_ids lookup raised (treating as not-late): %s",
                exc,
            )
            return False
        # Symmetric predicate: late when the active list is empty OR
        # any cancel is pending on the session. The cancel-pending
        # branch closes the iter-11D race (goldfive#242) where the
        # cancel-request has landed but ADK hasn't yet finished
        # winding down the cancelled invocation.
        return (not active) or bool(cancel_pending)

    def _resolve_active_invocation_ids(self, drift: DriftEvent, session: Session) -> list[str]:
        """Resolve which invocation_id(s) a cancel should target.

        Returns an ordered list of invocation ids that are "active"
        with respect to the triggering drift. The primary source is
        the reconciler's invocation bookkeeping (goldfive#151
        introduced the ``_invocation_agent`` / ``_invocation_parent``
        maps). When the reconciler is unavailable or empty, falls
        back to the drift's ``current_agent_id``-keyed invocation
        (best effort via the adapter's active-context invocation id)
        and finally returns an empty list.

        Tree-agnostic: the method does NOT special-case "the
        coordinator" or "the root agent" — it targets whichever
        invocation matches the drift's context and lets the plugin's
        child-propagation logic flag the rest of the sub-tree.
        """
        candidates: list[str] = []
        reconciler = getattr(session, "_reconciler", None)
        if reconciler is None:
            # The steerer doesn't hold a direct reference to the
            # reconciler; the adapter's plugin does. Walk it via the
            # adapter when the plugin exposes the attribute.
            adapter = self._adapter
            plugin = getattr(adapter, "_plugin", None) if adapter is not None else None
            reconciler = getattr(plugin, "_reconciler", None) if plugin is not None else None
        if reconciler is not None:
            try:
                inv_agent = getattr(reconciler, "_invocation_agent", None)
                if isinstance(inv_agent, Mapping) and drift.current_agent_id:
                    # Match by agent name — most drifts carry
                    # ``current_agent_id`` set to the running agent's name.
                    for inv_id, agent_name in inv_agent.items():
                        if agent_name == drift.current_agent_id and inv_id:
                            candidates.append(str(inv_id))
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer._resolve_active_invocation_ids: reconciler lookup raised: %s",
                    exc,
                )
        # Fallback: the adapter's plugin pins a top-level invocation_id
        # for the currently-driving dispatch. When the reconciler lookup
        # produced nothing, the top-level id is the best we can do —
        # cancel propagation from there will flag any sub-invocations.
        if not candidates:
            adapter = self._adapter
            plugin = getattr(adapter, "_plugin", None) if adapter is not None else None
            top = str(getattr(plugin, "_top_invocation_id", "") or "")
            if top:
                candidates.append(top)
        return candidates

    async def request_invocation_cancel(
        self,
        *,
        drift: DriftEvent,
        session: Session,
        cancel_inflight_task: bool = False,
    ) -> list[str]:
        """Flag the invocation(s) associated with ``drift`` for
        cooperative cancellation (goldfive#251 Stream C / 7a).

        Called from :meth:`_handle_drift` and
        :meth:`_promote_drift_to_steer` when the drift's severity is
        CRITICAL — the only tier on the ladder that reaches the hard
        cancel per the severity decision. INFO / WARNING drifts flow
        through their usual nudge / absorb paths without touching
        this method.

        Writes a :class:`~goldfive.types.CancellationRequest` onto the
        adapter's plugin state for every resolved active invocation
        id. The plugin propagates to children automatically. Returns
        the list of flagged invocation ids (including children) for
        observability; callers can log / sink-emit from the list.

        When ``cancel_inflight_task=True`` (goldfive#271 follow-up —
        v15 concurrent-invocation bug), the plugin ALSO fires
        ``task.cancel()`` on the registered asyncio.Task driving each
        flagged invocation, deferred via ``loop.call_soon`` so an
        inline same-task caller still completes its current emission
        work before the cancel lands. Default False so the existing
        pre-refine cancel paths keep their flag-only semantics; the
        post-refine helper :meth:`_cancel_inflight_for_revision`
        opts in explicitly so the cancel only fires AFTER a
        superseding plan has been installed.

        Guard rails:

        * No-op when no adapter is bound.
        * No-op when no active invocation can be resolved (e.g. the
          drift was synthesized before any agent turn started) — this
          is the "empty invocation-id guard" called out in the brief.
        * Tolerates missing plugin methods (third-party adapters that
          don't implement :meth:`request_invocation_cancel`) by
          falling through silently; the rest of the ladder (refine,
          restart message) still runs and eventually catches up at
          the next task boundary.
        * Plugins whose ``request_invocation_cancel`` predates
          ``cancel_inflight_task`` (TypeError on the kwarg) fall back
          to the kwarg-less call so older third-party plugins don't
          break — the task-cancel step is silently skipped.
        """
        adapter = self._adapter
        if adapter is None:
            return []
        plugin = getattr(adapter, "_plugin", None)
        if plugin is None:
            return []
        fn = getattr(plugin, "request_invocation_cancel", None)
        if not callable(fn):
            return []
        invocation_ids = self._resolve_active_invocation_ids(drift, session)
        # Stamp the cancel-pending flag SYNCHRONOUSLY before any
        # plugin / async work (goldfive#242). The active-task
        # registry takes 4-8s to drain while ADK winds the cancelled
        # invocations down; during that window the late-drift gate
        # would otherwise see ``active_invocation_ids()`` non-empty
        # and let a freshly-arriving goldfive-authored drift dispatch
        # a refine against an effectively-dead session. Flipping the
        # flag here closes the race: any drift handled after this
        # point sees ``cancel_requested_invocation_ids()`` non-empty
        # via :meth:`_is_late_drift_for_terminated_invocation` and
        # short-circuits.
        if invocation_ids:
            try:
                from goldfive.orchestration_store import OrchestrationStore

                store = OrchestrationStore.for_session(session)
                for inv_id in invocation_ids:
                    store.mark_invocation_cancel_requested(inv_id)
            except Exception as exc:  # noqa: BLE001 — defensive
                log.debug(
                    "DefaultSteerer.request_invocation_cancel: "
                    "cancel-pending stamp raised (continuing): %s",
                    exc,
                )
        if not invocation_ids:
            # Empty invocation-id guard — drift has no identifiable
            # in-flight invocation. Don't fabricate one; the cancel
            # would misfire on whatever invocation happens to share a
            # blank id. The drift still observed, refine still runs;
            # cancel is a best-effort add-on.
            log.debug(
                "DefaultSteerer.request_invocation_cancel: no active invocation "
                "for drift kind=%s agent=%s task=%s — skipping cancel",
                drift.kind.value,
                drift.current_agent_id or "-",
                drift.current_task_id or "-",
            )
            return []
        # Build the request once and reuse for every targeted id so
        # sink events from propagation share a common fingerprint.
        import time as _time_mod

        request = CancellationRequest(
            invocation_id=invocation_ids[0],
            reason=self._cancel_reason_for_drift(drift),
            severity=drift.severity,
            drift_id=str(getattr(drift, "id", "") or ""),
            drift_kind=drift.kind.value,
            requested_at_ms=int(_time_mod.time() * 1000),
            detail=(drift.detail or "")[:200],
        )
        flagged: list[str] = []
        for inv_id in invocation_ids:
            try:
                result = fn(
                    invocation_id=inv_id,
                    request=request,
                    cancel_inflight_task=cancel_inflight_task,
                )
            except TypeError:
                # Older plugin without the ``cancel_inflight_task``
                # kwarg (third-party / pre-#271-follow-up). Fall back
                # to the legacy signature; the task-cancel step is
                # silently skipped, but the flag-only contract is
                # preserved.
                try:
                    result = fn(invocation_id=inv_id, request=request)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "DefaultSteerer.request_invocation_cancel: "
                        "plugin.request_invocation_cancel(%s) raised: %s",
                        inv_id,
                        exc,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer.request_invocation_cancel: "
                    "plugin.request_invocation_cancel(%s) raised: %s",
                    inv_id,
                    exc,
                )
                continue
            if isinstance(result, list):
                flagged.extend(str(x) for x in result)
            else:
                flagged.append(inv_id)
        if flagged:
            log.info(
                "DefaultSteerer.request_invocation_cancel: flagged "
                "invocations=%s for drift kind=%s severity=%s",
                flagged,
                drift.kind.value,
                drift.severity.value,
            )
        return flagged

    @staticmethod
    def _should_request_cancel_for_drift(drift: DriftEvent) -> bool:
        """Decide whether a drift warrants a cooperative cancel.

        Severity ladder (goldfive#251 design decision):

        * ``DriftSeverity.INFO`` — never cancels. Info drifts are
          either periodic-check signals or soft one-shots; cancel
          would be disproportionate.
        * ``DriftSeverity.WARNING`` — never cancels. Warning drifts
          route to the existing ABSORB / NUDGE ladder paths; the
          refined plan lands on the next task boundary without
          preempting the in-flight turn.
        * ``DriftSeverity.CRITICAL`` — cancels. The in-flight turn's
          output is likely to contaminate its parent's transcript
          (stale prompt, wrong scope, broken tool); short-circuit
          cleanly and let the parent see ``{"status": "cancelled"}``.

        User-authored drifts (``USER_STEER`` / ``USER_CANCEL`` /
        ``USER_PAUSE``) bypass the severity gate — an operator
        directive must be honoured even when the ControlMessage-to-
        DriftEvent coercion landed on a lower severity tier.
        """
        if drift.kind in (
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.USER_PAUSE,
        ):
            return True
        return drift.severity is DriftSeverity.CRITICAL

    @staticmethod
    def _cancel_reason_for_drift(drift: DriftEvent) -> str:
        """Map a drift into a short symbolic reason for the
        :class:`~goldfive.types.CancellationRequest`.

        USER_STEER / USER_CANCEL / USER_PAUSE get the matching
        ``"user_*"`` shorthand; everything else uses ``"drift"`` as
        the generic tag. The reason is OPERATOR-visible only (lives
        on the InvocationCancelled sink event), so this string is
        free to be descriptive without prompt-injection concerns.
        """
        kind = drift.kind
        if kind is DriftKind.USER_STEER:
            return "user_steer"
        if kind is DriftKind.USER_CANCEL:
            return "user_cancel"
        if kind is DriftKind.USER_PAUSE:
            return "user_pause"
        return "drift"

    async def _cancel_inflight_for_revision(self, drift: DriftEvent, session: Session) -> list[str]:
        """Cancel the in-flight invocation(s) that produced ``drift``.

        Called from every drift-driven PlanRevised emission path right
        after the revised plan has been applied to the session and
        BEFORE the ``PlanRevised`` event is emitted. Closes the gap
        behind the v15 concurrent-invocation bug: a ``refine_steer``
        call (10+ minutes on a slow planner) used to overlap the
        coordinator's invocation for its full duration because the
        existing cancel-state flag only gates SUBSEQUENT callbacks —
        the already-running LLM streaming call kept generating output
        that triggered more drift, looping the refine.

        After Option A (goldfive#271 follow-up), turn-1 first-plan
        installs no longer reach this path:
        :meth:`install_initial_plan` skips it directly because there
        is no in-flight invocation to cancel on a fresh session. Every
        drift-driven install (refine from drift, refine_steer from
        goldfive-steer promotion, operator USER_STEER from a real
        ControlMessage, NEW_WORK_DISCOVERED from an N+1 user message)
        flows through this method. The plugin's
        :meth:`request_invocation_cancel` then writes the cancel-state
        flag (sticky-gate from PR #299) AND fires ``task.cancel()`` on
        the registered asyncio.Task (goldfive#271 follow-up) so the
        coordinator's in-flight LLM call observes ``CancelledError``
        within ~one event-loop tick instead of ~the LLM-call's
        full duration.

        Supersede contract (Bug A fix from v22 validation): every call
        to this method represents a goldfive-INTERNAL cancel — the
        revised plan has just been applied, and the cancel is the
        mechanism by which the in-flight agent is switched onto it.
        Stamps ``session._supersede_pending = True`` BEFORE initiating
        the cancel so the executor's overlay loop
        (:meth:`SequentialExecutor._run_overlay` cancelled branch) can
        distinguish this internal supersede from an external cancel
        (USER_CANCEL via control channel, asyncio.CancelledError from
        above) and restart the passthrough loop with the new plan
        instead of aborting the turn. The executor consumes and clears
        the flag; an unconsumed flag (e.g. cancel never lands because
        the invocation already completed) is harmless — the next
        cancel-branch entry will see and clear it, or the run finishes
        normally and the Session is discarded.

        Best-effort: an unbound adapter, a non-ADK adapter without
        :meth:`request_invocation_cancel`, or an empty resolved
        invocation-id list each result in a no-op (the refined plan
        still lands; the in-flight invocation simply runs to
        completion under the older, less aggressive contract). The
        supersede flag is still stamped in the no-op case — it costs
        nothing and a downstream overlay that DOES observe a cancel
        from a separate path stays correctly classified.
        """
        # Stamp the supersede marker so the overlay loop can
        # distinguish this internal cancel from an external one. See
        # the supersede-contract paragraph in the docstring above.
        try:
            session._supersede_pending = True  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — flag is best-effort
            log.debug(
                "DefaultSteerer._cancel_inflight_for_revision: "
                "could not stamp supersede flag on session: %s",
                exc,
            )
        try:
            return await self.request_invocation_cancel(
                drift=drift,
                session=session,
                cancel_inflight_task=True,
            )
        except Exception as exc:  # noqa: BLE001 — cancel is best-effort
            log.debug(
                "DefaultSteerer._cancel_inflight_for_revision: "
                "request_invocation_cancel raised: %s",
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # goldfive-steer-unification: promotion policy + handler
    # ------------------------------------------------------------------

    # Drift kinds eligible for ladder-promoted steer treatment when
    # goldfive-authored. Mirrors the "content drifts the coordinator
    # acknowledges but doesn't correct" list from the unification
    # design brief: off-topic / intent / unexpected output /
    # confabulation / loop kinds. Other detector kinds (SCHEMA_VIOLATION,
    # REFINE_VALIDATION_FAILED, REPEATED_FAILURE, GOAL_DRIFT, …) keep
    # their pre-unification ladder mapping so escalation / repeated-
    # failure semantics aren't rerouted into the cancel-in-flight path.
    _GOLDFIVE_STEER_ELIGIBLE_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.OFF_TOPIC,
            DriftKind.INTENT_DIVERGENCE,
            DriftKind.UNEXPECTED_OUTPUT,
            DriftKind.CONFABULATION_RISK,
            DriftKind.LOOPING_REASONING,
            DriftKind.LOOPING_TOOL_CALL,
            DriftKind.PLAN_DIVERGENCE,
        }
    )

    def _severity_meets_promotion_threshold(self, severity: DriftSeverity) -> bool:
        """True iff ``severity`` satisfies the configured promotion threshold."""
        threshold = self._goldfive_steer_threshold
        if threshold == "off":
            return False
        if threshold == "critical":
            return severity is DriftSeverity.CRITICAL
        # "warning" — promote WARNING and CRITICAL.
        return severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)

    def _should_promote_to_steer(self, drift: DriftEvent, session: Session) -> bool:
        """Evaluate the drift against the unification promotion policy.

        Returns ``True`` iff the drift should be dispatched through
        :meth:`_promote_drift_to_steer` instead of the legacy passive
        ladder. Side-effect: stamps ``drift.suppressed_by_user_steer``
        when a fresh user steer is blocking promotion so the subsequent
        ``DriftDetected`` emission reflects the suppression decision.

        The policy:

        1. User-authored drifts (USER_STEER / USER_CANCEL / USER_PAUSE)
           keep their pre-unification handling — USER_STEER already
           routes through the refine path with cancel-in-flight wired
           by the executor. Return ``False``.
        2. The drift kind must be in
           :data:`_GOLDFIVE_STEER_ELIGIBLE_KINDS` — other kinds keep
           their legacy ladder mapping.
        3. The severity must clear the configured ``threshold``.
        4. If a user-authored steer is within the freshness window
           (``suppression_window_turns`` turns), stamp the suppression
           flag and return ``False``. Otherwise return ``True``.
        """
        if drift.kind in self._USER_AUTHORED_DRIFT_KINDS:
            return False
        authored_by = self._resolve_authored_by(drift)
        if authored_by != "goldfive":
            return False
        if drift.kind not in self._GOLDFIVE_STEER_ELIGIBLE_KINDS:
            return False
        if not self._severity_meets_promotion_threshold(drift.severity):
            return False
        # Consult the active user steer freshness window.
        # Phase 1 of goldfive#271 — read through OrchestrationStore so
        # the active-steer slot reads from a single named accessor; the
        # underlying ``_ostate.read`` calls still funnel through the
        # goldfive Session.state dict, just behind a typed surface.
        window = self._goldfive_steer_suppression_window_turns
        if window > 0:
            from goldfive.orchestration_store import OrchestrationStore

            active = OrchestrationStore.for_session(session).get_active_steer()
            if active is not None and active.source.lower() == "user":
                current_turn = int(getattr(session, "_next_sequence", 0) or 0)
                age = current_turn - active.at_turn
                if 0 <= age < window:
                    drift.suppressed_by_user_steer = True
                    log.info(
                        "goldfive steer suppressed: user steer %r is active "
                        "(age=%d turns, window=%d)",
                        active.body,
                        age,
                        window,
                    )
                    return False
        return True

    async def _promote_drift_to_steer(self, drift: DriftEvent, session: Session) -> None:
        """Promote a goldfive-detected drift into a full steer.

        Ordered side effects (mirrors the USER_STEER path):

        1. Tag the bound adapter's ``_next_cancel_reason`` with a
           ``"goldfive_<drift_kind>"`` symbolic reason so the in-flight
           invocation's synthetic ``function_response`` carries an
           LLM-actionable explanation.
        2. Stamp ``goldfive.active_steer.*`` onto ``session.state``
           (body = derived from :meth:`_compose_goldfive_steer_body`,
           author = ``"goldfive"``, source = ``"goldfive"``).
        3. Record ``drift.id`` in ``goldfive.processed_steer_ids`` so
           the same drift cannot re-promote on a delivery retry.
        4. Call :meth:`LLMPlanner.refine_steer` (or the generic
           ``planner.refine`` fallback when the planner doesn't expose
           the goldfive-specific entry point) with ``source="goldfive"``
           semantics so the refine prompt frames the pivot as a
           correction, not as an operator directive.
        5. Install the revised plan + emit ``PlanRevised``.

        Note on cancel-in-flight: the actual ``task.cancel()`` on the
        adapter invocation is the executor's responsibility
        (:meth:`SequentialExecutor._invoke_with_control` performs it
        when a ``STEER`` ControlMessage arrives). The steerer tags the
        adapter and queues a restart message so that the **next** time
        the executor reaches a cancel / steer checkpoint (either
        because a sink callback requested cancel, or because the
        overlay loop picks up the pending restart message), the
        contaminated invocation is preempted. For the common case
        where the drift is detected from a mid-invocation reasoning
        block and the overlay loop is already streaming, the queued
        restart message reaches the LLM on the next turn — cancel
        semantics identical to USER_STEER.
        """
        # 1. Tag adapter cancel reason.
        cancel_reason = self._tag_adapter_cancel_reason_for_promotion(drift, session=session)
        # Session-visible cancel prefix so ``_mark_cancelled_if_live``
        # stamps it on any TaskCancelled the executor emits for the
        # in-flight task as part of the promotion.
        try:
            session._last_cancel_reason_prefix = cancel_reason
        except Exception:  # noqa: BLE001
            pass
        # 1a. NOTE (#241 emergency revert): previously we fired
        # ``adapter.request_cancel(reason)`` here to terminate the
        # in-flight LLM call immediately. In practice that
        # ``task.cancel()`` propagated a ``CancelledError`` past the
        # executor's invocation-scope catch and killed the entire run
        # — observed as ``run_aborted`` immediately after a
        # goldfive-detected drift. Reverted to the pre-#241 deferred-
        # cancel semantics: we stamp ``_next_cancel_reason`` (above)
        # and queue a restart message; the executor loop sees the
        # queue at the next invocation boundary and resumes with the
        # refined plan. The taint of letting the in-flight call run
        # to completion is a lesser evil than aborting the run.
        # Proper fix (future): scope the cancel to the LLM stream
        # only, or catch ``CancelledError`` at the goldfive-steer
        # boundary and continue.
        # 2. Stamp active-steer state + compose the restart body.
        at_turn = int(getattr(session, "_next_sequence", 0) or 0)
        body = self._compose_goldfive_steer_body(drift)
        try:
            _ostate.set_active_steer(
                session.state,
                body=body,
                at_turn=at_turn,
                author="goldfive",
                source="goldfive",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._promote_drift_to_steer: set_active_steer raised: %s",
                exc,
            )
        # Phase 2 of the path-duality fix: dispatch a
        # ``GOLDFIVE_STEER`` ControlMessage on the bound channel so
        # the executor's invoke loop cancels the in-flight invocation
        # and restarts the passthrough with a ``[GOLDFIVE STEERING
        # CONTROL …]`` framed corrective. The body, drift kind, and
        # superseded task ids ride the message payload; the executor
        # composes the restart text from those fields. Pre-Phase-2
        # this wrote ``session.pending_corrective_message`` — a
        # write-only slot that left the coordinator blind to the plan
        # swap.
        await self._dispatch_goldfive_steer_control(
            drift, session, body_override=body
        )
        # 3. Record the drift id in processed_steer_ids so a redelivery
        # (same drift id) doesn't re-cancel / re-refine.
        drift_id = str(getattr(drift, "id", "") or "")
        if drift_id:
            try:
                _ostate.record_processed_steer_id(session.state, drift_id)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer._promote_drift_to_steer: record_processed_steer_id raised: %s",
                    exc,
                )
        # 4. Route to planner.refine_steer (source="goldfive") — falls
        # back to planner.refine for planners that don't expose the
        # goldfive-specific entry point.
        if self._planner is None or session.plan is None:
            return
        # Outcome-based gate (goldfive#215 iter-8 P2). Mirror of the
        # gate in ``_handle_drift``: skip refine_steer when (kind, task)
        # already has a terminal outcome on this turn. USER_STEER /
        # USER_CANCEL / GOAL_DRIFT bypass.
        if drift.kind not in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            outcome_key = (drift.kind.value, drift.current_task_id or "")
            outcome = session.refine_outcomes.get(outcome_key)
            if outcome is not None:
                if outcome.state == "succeeded":
                    log.debug(
                        "refine_steer skipped: prior succeeded outcome (kind=%s task=%r)",
                        drift.kind.value,
                        drift.current_task_id,
                    )
                    return
                if outcome.fail_count >= self.REFINE_FAILURE_THRESHOLD:
                    log.debug(
                        "refine_steer skipped: failure threshold reached "
                        "(kind=%s task=%r count=%d)",
                        drift.kind.value,
                        drift.current_task_id,
                        outcome.fail_count,
                    )
                    return
        # Progress-based escalation (goldfive#271). Orthogonal to the
        # outcome gate: see parallel check in ``_handle_drift``.
        if self._is_task_progress_stalled(drift, session):
            await self._emit_progress_stalled_escalation(drift, session)
            return
        # ContextVar plumbing for the planner-side drift-emitter and
        # span-context callbacks; per-async-task so concurrent runs
        # sharing this Steerer keep their session pointers isolated.
        _active_session_token = self._active_session_var.set(session)
        # goldfive a4: same attempt-id correlation contract as
        # ``_handle_drift``.
        attempt_id = self._new_attempt_id()
        await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
        # Resolve the registry constraint (goldfive#151) the same way
        # ``_handle_drift`` does so the goldfive steer refine honours it.
        planner = self._planner
        available_agents: Any = None
        adapter = self._adapter
        if adapter is not None:
            tree = getattr(adapter, "available_agents_tree", None)
            if isinstance(tree, list) and tree:
                available_agents = list(tree)
            else:
                flat = getattr(adapter, "available_agents", None)
                if flat:
                    available_agents = list(flat)
        # Phase 3.5 (goldfive#271) tripwire wrapper — see §C4.
        with _state_audit.cancellation_stash_audited(
            "DefaultSteerer._promote_drift_to_steer.refine"
        ):
            try:
                # Call ``planner.refine_steer`` when available; fall back
                # to ``planner.refine``. The fallback exists for test
                # stubs / third-party planners that don't expose the
                # goldfive-specific entry point — the generic path is
                # better than no refine at all.
                refine_steer = getattr(planner, "refine_steer", None)
                if callable(refine_steer):
                    revised = await refine_steer(
                        plan=session.plan,
                        drift=drift,
                        goals=list(session.goals),
                        available_agents=available_agents,
                    )
                elif _planner_refine_accepts_available_agents(planner):
                    revised = await planner.refine(
                        plan=session.plan,
                        drift=drift,
                        goals=list(session.goals),
                        available_agents=available_agents,
                    )
                else:
                    revised = await planner.refine(
                        plan=session.plan,
                        drift=drift,
                        goals=list(session.goals),
                    )
            except RefineExhausted as exc:
                # goldfive#271: planner explicitly signalled handler
                # exhaustion. Pause for human intervention.
                log.info(
                    "DefaultSteerer._promote_drift_to_steer: refine raised "
                    "RefineExhausted for kind=%s task=%r: %s",
                    drift.kind.value,
                    drift.current_task_id,
                    exc,
                )
                await self._emit_refine_failed(
                    session,
                    drift,
                    attempt_id=attempt_id,
                    failure_kind="refine_exhausted",
                    reason=str(exc) or "planner signalled handler exhaustion",
                    detail="",
                )
                await self._emit_handler_exhausted_escalation(drift, session)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "DefaultSteerer._promote_drift_to_steer: refine raised %s; plan unchanged",
                    exc,
                )
                await self._emit_refine_failed(
                    session,
                    drift,
                    attempt_id=attempt_id,
                    failure_kind="llm_error",
                    reason=str(exc),
                    detail=type(exc).__name__,
                )
                await self._escalate_refine_failure_as_critical_drift(
                    session, drift, reason=str(exc)
                )
                await self._record_refine_outcome(session, drift, succeeded=False)
                return
            except BaseException as exc:  # noqa: BLE001
                # Phase 3.5 (CANCELLATION-CONTRACT.md §C4): ``CancelledError``
                # bypasses the ``except Exception`` branch above. Emit the
                # paired ``refine_failed`` so cancelled goldfive-steer refines
                # do not leave sinks with an unmatched ``refine_attempted``.
                # Re-raise to preserve asyncio cancellation propagation.
                await self._emit_refine_failed(
                    session,
                    drift,
                    attempt_id=attempt_id,
                    failure_kind="cancelled",
                    reason=f"refine cancelled: {type(exc).__name__}",
                    detail=type(exc).__name__,
                )
                # Phase 3.5 tripwire compliance marker (§1.2 form).
                _state_audit.mark_stash_completed()
                raise
            finally:
                self._active_session_var.reset(_active_session_token)
        if revised is None:
            # iter-12 (#204): mirror the ``_handle_drift`` graceful
            # fallback — refine returning None means the planner has
            # exhausted its internal retry budget; escalate to
            # HUMAN_INTERVENTION_REQUIRED rather than emitting a
            # recursing CRITICAL follow-up drift that would eventually
            # abort the run.
            log.warning(
                "DefaultSteerer._promote_drift_to_steer: refine returned None; "
                "plan unchanged — escalating to HUMAN_INTERVENTION_REQUIRED"
            )
            await self._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="parse_error",
                reason="planner returned no revised plan",
                detail="",
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # I4 fix: fold runtime terminal statuses from the prior plan
        # onto the revised plan BEFORE validation (see _handle_drift
        # for the full rationale). goldfive#247: returns a NEW Plan.
        revised = self._fold_runtime_terminal_statuses(revised, session.plan)
        try:
            revised.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            # iter-12 (#204): mirror the ``_handle_drift`` graceful
            # fallback — keep the SCHEMA_VIOLATION emission at INFO
            # severity for operator/sink observability (does NOT
            # recurse through ``_handle_drift``) and escalate to
            # HUMAN_INTERVENTION_REQUIRED. The actionable signal is
            # the paired ``refine_failed(validator_rejected)`` envelope
            # plus the escalation drift.
            await self._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="validator_rejected",
                reason=f"plan validation failed: {exc}",
                detail=type(exc).__name__,
            )
            await self._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.INFO,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                    authored_by="goldfive",
                ),
            )
            await self._record_refine_outcome(session, drift, succeeded=False)
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        # No-op revision rejection (goldfive#271). Same handler-
        # exhaustion semantics as ``_handle_drift`` — a structurally
        # identical plan means the planner cannot make progress on this
        # drift; escalate to HUMAN_INTERVENTION_REQUIRED.
        if self._plans_structurally_identical(session.plan, revised):
            log.info(
                "no-op refine_steer revision skipped (kind=%s task=%r); "
                "escalating to HUMAN_INTERVENTION_REQUIRED",
                drift.kind.value,
                drift.current_task_id,
            )
            await self._emit_refine_failed(
                session,
                drift,
                attempt_id=attempt_id,
                failure_kind="no_op_revision",
                reason="planner returned structurally identical plan",
                detail="",
            )
            await self._emit_handler_exhausted_escalation(drift, session)
            return
        await self._record_refine_outcome(session, drift, succeeded=True)
        prev_plan = session.plan
        # goldfive#247: rebind to the stamped instance.
        revised = self._apply_revision(session, revised, drift)
        # Cancel the in-flight coordinator invocation now that
        # ``refine_steer`` produced a superseding plan (goldfive#271
        # follow-up — v15 concurrent-invocation bug). This is the
        # path that empirically motivated the fix: a goldfive-steer-
        # eligible drift (PLAN_DIVERGENCE / OFF_TOPIC / …) at WARNING
        # severity got refined while the coordinator's LLM call kept
        # running for the full ``refine_steer`` duration, generating
        # contaminated output that triggered more drift. Cancelling
        # here preempts the in-flight LLM call so its remaining
        # output can't loop the refine.
        #
        # Order: cancel BEFORE PlanRevised emit so the synthetic
        # InvocationCancelled sink event lands adjacent to the
        # revision and operators can correlate the two on the
        # gantt timeline.
        await self._cancel_inflight_for_revision(drift, session)
        await self._emit_plan_revised(
            session, revised, drift, prev_plan=prev_plan, attempt_id=attempt_id
        )

    @staticmethod
    def _compose_goldfive_steer_body(drift: DriftEvent) -> str:
        """Derive the steer body for a goldfive-promoted drift.

        Prefers ``drift.detail`` verbatim — the reasoning judge and
        other LLM-as-a-judge paths already emit human-readable reasons
        like "agent acknowledged discrepancy but chose to adopt
        expanded topic" that are directly usable as a corrective. When
        ``detail`` is empty, synthesise a generic template from the
        drift's kind / severity / task context.
        """
        detail = str(getattr(drift, "detail", "") or "").strip()
        if detail:
            return detail
        task_id = drift.current_task_id or "the current task"
        return (
            f"Goldfive detected {drift.kind.name} drift "
            f"(severity={drift.severity.name}). The preceding agent "
            f"output did not match the task: {task_id}. Discard prior "
            "work on this task and proceed with the corrective plan."
        )

    # ------------------------------------------------------------------
    # USER_STEER state handler (goldfive#152)
    # ------------------------------------------------------------------

    async def _apply_user_steer_state(
        self,
        drift: DriftEvent,
        session: Session,
    ) -> None:
        """Side-effects for USER_STEER drift that aren't refine: state
        bookkeeping.

        Called from :meth:`_handle_drift` and
        :meth:`apply_user_steer_with_plan` just before
        ``_emit_drift_detected`` and well before any plan install so:

        1. The ``goldfive.active_steer.*`` keys are set so downstream
           observers see the steer before the drift event.
        2. The source annotation / control id is appended to
           ``goldfive.processed_steer_ids`` so a retry or UI double-fire
           of the same STEER is a no-op (goldfive#171 dedupe).

        Never raises.

        Phase 4 (goldfive#271): goal synthesis was previously done
        here via ``planner.synthesize_goal_from_steer`` plus a
        regex-based qualification-merge post-process. That is now the
        :meth:`Planner.handle_turn` LLM's job — it produces the
        revised plan with qualifications already merged in one shot.
        This method retains only the bookkeeping-side effects.
        """
        # Recover the raw body + operator author from the originating
        # ControlMessage when it's available on drift.raw (goldfive#171).
        # Falling back to drift.detail preserves back-compat for tests
        # that synthesize a USER_STEER DriftEvent directly without a
        # ControlMessage behind it.
        raw_body, author, steer_id = self._unpack_steer_context(drift)
        body = raw_body.strip()
        # Stamp the active_steer keys regardless so readers see "a
        # steer is active as of turn N". ``at_turn`` uses the
        # session's monotonic sequence counter which increments on
        # every emitted event — a cheap, always-available "turn"
        # proxy.
        at_turn = getattr(session, "_next_sequence", 0) or 0
        try:
            _ostate.set_active_steer(
                session.state,
                body=body,
                at_turn=at_turn,
                author=author,
                source="user",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._apply_user_steer_state: set_active_steer raised: %s",
                exc,
            )
        # Record the dedupe id. Safe to call even with an empty id
        # (the helper no-ops). Done AFTER the active_steer stamp so a
        # reader that inspects ``state`` mid-dispatch always sees the
        # most recent steer is reflected.
        if steer_id:
            try:
                _ostate.record_processed_steer_id(session.state, steer_id)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "DefaultSteerer._apply_user_steer_state: record_processed_steer_id raised: %s",
                    exc,
                )

    async def install_initial_plan(
        self,
        *,
        session: Session,
        plan: Plan,
        is_pivot: bool = False,
    ) -> bool:
        """Install ``plan`` as the very first revision (rev 1) of ``session.plan``.

        Used on turn 1 of a fresh conversation when ``session.plan`` is
        a :meth:`Plan.empty` seed. Emits :class:`PlanRevised` with
        ``revision_index = 1`` and **no** :class:`DriftDetected` event:
        installing the first plan is not a corrective intervention,
        and stamping a USER_STEER drift here was the category error
        Option A (goldfive#271 follow-up) eliminates.

        ``is_pivot`` (F5, goldfive#322 Layer 2 / #204): when ``True``,
        the caller has classified the user's intent as a PIVOT —
        replacement of the prior plan rather than a revision of it.
        The validator runs WITHOUT ``prior`` so Rule 6
        (terminal-task / terminal->terminal-edge preservation) does
        not gate the new plan against a structurally-unrelated
        predecessor. The runner sets this when
        :meth:`Planner.handle_turn` flagged ``replaces_prior`` on the
        produced plan.

        The internal ``DriftEvent`` placeholder this method passes to
        :meth:`_apply_revision` and :meth:`_emit_plan_revised` carries
        ``DriftKind.NEW_WORK_DISCOVERED`` (``severity=INFO``) so the
        :class:`PlanRevised` envelope's ``drift_kind`` field has a
        coherent value — that field is required by the proto and
        downstream consumers (harmonograf) read it for revision
        framing. The placeholder is **never emitted** as a
        ``DriftDetected``.

        Returns ``True`` on success, ``False`` on validation failure.
        Never raises.
        """
        try:
            if is_pivot:
                # Pivot: validate structurally only. Rule 6 (terminal
                # preservation) is intentionally skipped — the user is
                # replacing the prior plan, not revising it.
                #
                # No fold for pivots — the user is replacing the prior
                # plan; runtime terminal statuses from the discarded
                # plan are not relevant to the new sub-DAG.
                plan.validate(for_revision=True, prior=None)
            else:
                # I4 fix: fold runtime terminal statuses from the prior
                # plan onto the candidate before validation.
                # goldfive#247: returns a NEW Plan; assign so the caller
                # uses the folded variant downstream.
                plan = self._fold_runtime_terminal_statuses(plan, session.plan)
                plan.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            await self._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.CRITICAL,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                    authored_by="goldfive",
                ),
            )
            return False
        # Placeholder drift used only to thread metadata through
        # :meth:`_apply_revision` / :meth:`_emit_plan_revised`. Not
        # emitted as a DriftDetected.
        placeholder = DriftEvent(
            kind=DriftKind.NEW_WORK_DISCOVERED,
            severity=DriftSeverity.INFO,
            detail="initial plan install",
            authored_by="goldfive",
        )
        prev_plan = session.plan
        # goldfive#247: rebind to the stamped instance.
        plan = self._apply_revision(session, plan, placeholder)
        # No cancel-in-flight: nothing is running yet on the very
        # first install.
        await self._emit_plan_revised(
            session,
            plan,
            placeholder,
            prev_plan=prev_plan,
            attempt_id=None,
        )
        return True

    async def install_revision_for_drift(
        self,
        *,
        session: Session,
        drift: DriftEvent,
        revised_plan: Plan,
    ) -> bool:
        """Install ``revised_plan`` in response to a real :class:`DriftEvent`.

        The general-purpose install path for non-user-steer revisions:
        an LLM-driven replan after the user's next-turn message
        (``DriftKind.NEW_WORK_DISCOVERED``), an autonomous
        detector-promoted refine, or any other drift-driven plan
        revision the caller has already classified.

        Pipeline:

        * :meth:`_emit_drift_detected` — ``DriftDetected`` carrying
          the **real** drift kind/severity/detail
        * validate against prior plan; emit ``SCHEMA_VIOLATION`` and
          return ``False`` on failure
        * :meth:`_apply_revision` — bump ``revision_index`` + stamp
          metadata
        * :meth:`_cancel_inflight_for_revision` — preempt any
          in-flight invocation per the drift's severity
        * :meth:`_emit_plan_revised` — ``PlanRevised`` + the paired
          refine-attempted / -success sidecar envelopes

        Refuses :class:`DriftKind.USER_STEER` — callers must route
        genuine operator steers through
        :meth:`install_revision_for_user_steer` so the active_steer
        bookkeeping and dedupe fire correctly.

        Returns ``True`` on success, ``False`` on validation failure.
        Never raises.
        """
        if drift.kind is DriftKind.USER_STEER:
            raise ValueError(
                "install_revision_for_drift refuses USER_STEER drifts; "
                "use install_revision_for_user_steer for genuine "
                "operator-pushed STEER ControlMessages."
            )
        if not drift.authored_by:
            drift.authored_by = self._resolve_authored_by(drift)
        return await self._install_with_drift(
            session=session,
            drift=drift,
            revised_plan=revised_plan,
            apply_user_steer_state=False,
        )

    async def install_revision_for_user_steer(
        self,
        *,
        session: Session,
        raw: Any,
        revised_plan: Plan,
    ) -> bool:
        """Install ``revised_plan`` in response to an operator
        :class:`~goldfive.control.ControlMessage` STEER.

        ``raw`` is the originating :class:`ControlMessage`; this method
        builds the ``USER_STEER`` :class:`DriftEvent` internally so
        callers cannot accidentally fabricate a USER_STEER from
        plumbing (the category error #199/#302 papered over).

        Pipeline:

        * :meth:`_apply_user_steer_state` — active_steer bookkeeping +
          dedup (always — every call here represents genuine operator
          action)
        * :meth:`_emit_drift_detected` — ``USER_STEER`` ``DriftDetected``
          with ``raw`` populated and ``authored_by="user"``
        * validate revised plan; emit ``SCHEMA_VIOLATION`` on failure
        * :meth:`_apply_revision` + :meth:`_cancel_inflight_for_revision`
          + :meth:`_emit_plan_revised`

        Returns ``True`` on success, ``False`` on validation failure.
        Never raises.
        """
        body, author, _dedupe = self._unpack_steer_context(
            DriftEvent(
                kind=DriftKind.USER_STEER,
                severity=DriftSeverity.WARNING,
                raw=raw,
            )
        )
        detail = f"by {author}: {body}" if author else body
        drift = DriftEvent(
            kind=DriftKind.USER_STEER,
            severity=DriftSeverity.WARNING,
            detail=detail,
            raw=raw,
            authored_by="user",
        )
        return await self._install_with_drift(
            session=session,
            drift=drift,
            revised_plan=revised_plan,
            apply_user_steer_state=True,
        )

    async def install_user_steer(
        self,
        *,
        drift: DriftEvent,
        prior: Plan,
        llm_revision: Plan | None,
        session: Session,
    ) -> Plan:
        """Install a user-authored revision. ALWAYS returns a valid Plan.

        Contract (see ``docs/design/PLAN-LIFECYCLE.md`` §4.2.1): user-steer
        rejection is **structurally impossible**. The return type is
        ``Plan`` (never ``None``), and this method does not raise
        ``ValueError`` from validation. If the LLM-produced revision
        fails ``Plan.validate(for_revision=True, prior=...)``, this
        method falls back to the deterministic minimum evolution shape
        (per §4.2): preserve every terminal task verbatim, cancel every
        PENDING / RUNNING / BLOCKED task, drop edges incident to the
        cancelled set. The minimum is provably valid by construction.

        Order of preference:

        1. ``llm_revision`` if non-None and validates against ``prior``.
        2. :meth:`_build_minimal_steer_evolution` — deterministic, always
           valid, intentionally produces a plan with no PENDING tasks.

        The deterministic minimum lands the user's pivot as a clean
        terminal-only frontier; the next refine cycle or coordinator
        turn can populate the new sub-DAG. This is acceptable
        degradation — the turn does not abort. The contract sacrifices
        a bit of "the LLM's first attempt drove forward progress" for
        the much stronger "the user's intent ALWAYS lands".

        Side effects (regardless of which branch fires):

        * :meth:`_apply_user_steer_state` writes the
          ``goldfive.active_steer.*`` slot from ``drift``.
        * :meth:`_emit_drift_detected` emits the ``USER_STEER`` drift.
        * :meth:`_apply_revision` swaps ``session.plan`` and bumps
          ``revision_index``.
        * :meth:`_cancel_inflight_for_revision` preempts in-flight work.
        * :meth:`_emit_plan_revised` fires ``PlanRevised``.

        The deterministic-fallback branch deliberately does NOT touch
        ``session.refine_outcomes`` — that table governs goldfive-
        authored autonomous refines (§4.5), not user-driven changes.
        A USER_STEER never escalates via REPEATED_FAILURE.

        Never raises.
        """
        # Normalise the drift's authored_by so downstream observability
        # (DriftDetected.authored_by) carries the right attribution.
        if not drift.authored_by:
            drift.authored_by = "user"
        # Branch 1: try the LLM's revision if it parses + validates.
        chosen: Plan | None = None
        if llm_revision is not None:
            # I4 fix: fold runtime terminal statuses from the prior plan
            # onto the LLM revision before validation. Without this, an
            # NOT_NEEDED reaped task that the LLM regressed to PENDING
            # would force the deterministic-minimum fallback even when
            # the LLM's *new* work was otherwise sound.
            # goldfive#247: fold returns a NEW Plan; rebind so the
            # validator + downstream selection see the folded variant.
            llm_revision = self._fold_runtime_terminal_statuses(llm_revision, prior)
            try:
                llm_revision.validate(for_revision=True, prior=prior)
                chosen = llm_revision
            except ValueError as exc:
                log.warning(
                    "DefaultSteerer.install_user_steer: LLM revision rejected "
                    "by validator (%s); falling back to deterministic minimum "
                    "evolution shape (PLAN-LIFECYCLE.md §4.2.1)",
                    exc,
                )
        # Branch 2: deterministic minimum. Always valid by construction.
        if chosen is None:
            chosen = self._build_minimal_steer_evolution(prior, drift)
        # Always run the user-steer state bookkeeping — every call to
        # this method represents a genuine operator action.
        await self._apply_user_steer_state(drift, session)
        await self._emit_drift_detected(session, drift)
        # No-op short-circuit: the deterministic minimum on a prior with
        # no PENDING/RUNNING/BLOCKED tasks degenerates to a structurally
        # identical plan. Skip the install (avoids a misleading
        # PlanRevised with empty diff) but still return ``prior`` so the
        # contract (always a Plan) holds.
        if self._plans_structurally_identical(prior, chosen):
            log.info(
                "DefaultSteerer.install_user_steer: deterministic minimum "
                "is structurally identical to prior (no mutable tasks to "
                "cancel); install skipped, returning prior plan"
            )
            return prior
        prev_plan = session.plan
        attempt_id = self._new_attempt_id()
        await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
        # goldfive#247: rebind to the stamped instance.
        chosen = self._apply_revision(session, chosen, drift)
        await self._cancel_inflight_for_revision(drift, session)
        await self._emit_plan_revised(
            session,
            chosen,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
        )
        return chosen

    def _build_minimal_steer_evolution(
        self, prior: Plan, drift: DriftEvent
    ) -> Plan:
        """Construct the canonical evolution shape per PLAN-LIFECYCLE.md §4.2.

        Deterministic. Preserves terminal tasks verbatim (§3.1), cancels
        every PENDING / RUNNING / BLOCKED task (so they enter the
        absorbing CANCELLED terminal), and drops every edge incident to
        a cancelled task. The result always passes
        ``Plan.validate(for_revision=True, prior=prior)`` because:

        * Every prior-terminal task is preserved with the same status →
          §3.1 holds.
        * Every prior terminal->terminal edge is preserved verbatim →
          §3.2 holds.
        * No PENDING tasks remain → reachability invariant (§5 rule 7,
          goldfive#137) is vacuously satisfied (no PENDING task can
          have a CANCELLED predecessor because there ARE no PENDING
          tasks).
        * Edges only span surviving terminal endpoints → no dangling
          edges.

        Uses :func:`dataclasses.replace` so the prior plan and tasks
        are not mutated. The deriver caches Tasks by identity in a few
        places (Tier 2 #323 found that mutating shared Task references
        corrupts the cache); fresh copies sidestep that risk.

        ``drift`` is consulted only for revision metadata (kind /
        severity / detail go onto the new plan via
        :meth:`_apply_revision`). It is not strictly required here, but
        keeping the parameter mirrors the steerer's other revision
        builders and makes future extensions (e.g. tagging which task
        the steer named) easier.
        """
        _ = drift  # reserved for future per-task framing; see docstring
        new_tasks: list[Task] = []
        cancelled_ids: set[str] = set()
        for t in prior.tasks:
            if t.status.is_terminal:
                # Preserve verbatim — fresh copy so callers cannot
                # accidentally mutate the prior plan's task identity.
                new_tasks.append(dataclasses.replace(t))
            else:
                # PENDING / RUNNING / BLOCKED → CANCELLED. Stamp a
                # provenance reason so harmonograf's intervention view
                # can attribute the cancel to a user-steer rollover.
                cancelled = dataclasses.replace(
                    t,
                    status=TaskStatus.CANCELLED,
                    cancel_reason=f"user_steer_rollover:{drift.id}"
                    if getattr(drift, "id", "")
                    else "user_steer_rollover",
                )
                new_tasks.append(cancelled)
                cancelled_ids.add(t.id)
        # Edges: drop any edge incident to a cancelled task. The
        # surviving edges are exactly the prior terminal->terminal set
        # plus any pre-existing terminal->cancelled (now both terminal,
        # but we still drop those because the to-task transitioned in
        # this revision and §3.2 only freezes edges that were
        # terminal->terminal in PRIOR — not in the revision).
        # Simpler: drop any edge touching a cancelled-this-rev id.
        new_edges: list[TaskEdge] = []
        for e in prior.edges:
            if e.from_task_id in cancelled_ids or e.to_task_id in cancelled_ids:
                continue
            new_edges.append(dataclasses.replace(e))
        # Construct the revised plan. ``revision_index`` is bumped by
        # :meth:`_apply_revision`; ``revision_*`` metadata stamping
        # happens there too. We populate ``id`` / ``run_id`` /
        # ``goal_ids`` / ``summary`` from prior so identity stays
        # stable across the revision (the plan_id-stable-across-turns
        # invariant from goldfive#271 Phase 4).
        return Plan(
            id=prior.id,
            run_id=prior.run_id,
            goal_ids=tuple(prior.goal_ids),
            tasks=tuple(new_tasks),
            edges=tuple(new_edges),
            summary=prior.summary,
        )

    async def _install_with_drift(
        self,
        *,
        session: Session,
        drift: DriftEvent,
        revised_plan: Plan,
        apply_user_steer_state: bool,
    ) -> bool:
        """Shared install pipeline for the two drift-driven install APIs.

        Emits ``DriftDetected`` then validates + installs the revision
        + emits ``PlanRevised``. The ``apply_user_steer_state`` flag
        gates the ``goldfive.active_steer.*`` bookkeeping so genuine
        operator STEERs write the slot and other drift-driven
        installs do not.
        """
        if apply_user_steer_state:
            await self._apply_user_steer_state(drift, session)
        await self._emit_drift_detected(session, drift)
        # I4 fix: fold runtime terminal statuses from the prior plan
        # onto the revised plan BEFORE validation. This is the path
        # that NEW_WORK_DISCOVERED installs (Runner._install_revision)
        # and USER_STEER ControlMessage installs travel through, which
        # is where the v24 phantom-state regression was observed.
        # goldfive#247: returns a NEW Plan; rebind so validation +
        # _apply_revision below see the folded variant.
        revised_plan = self._fold_runtime_terminal_statuses(revised_plan, session.plan)
        try:
            revised_plan.validate(for_revision=True, prior=session.plan)
        except ValueError as exc:
            await self._emit_drift_detected(
                session,
                DriftEvent(
                    kind=DriftKind.SCHEMA_VIOLATION,
                    severity=DriftSeverity.CRITICAL,
                    detail=f"plan validation failed: {exc}",
                    current_task_id=session.current_task_id,
                    authored_by="goldfive",
                ),
            )
            return False
        # No-op revision rejection (goldfive#271 — replaces the deleted
        # count cap). If the install would be structurally identical to
        # the prior plan (same task ids, edges, assignees, statuses),
        # skip the install entirely: bumping ``revision_index`` for an
        # unchanged plan would emit a misleading PlanRevised with no
        # actual diff. Returns False so the caller can surface the
        # no-op. INFO-level so operators see why the install dropped.
        if self._plans_structurally_identical(session.plan, revised_plan):
            log.info(
                "no-op revision skipped on _install_with_drift "
                "(kind=%s task=%r); install dropped",
                drift.kind.value,
                drift.current_task_id,
            )
            return False
        # Capture prev_plan BEFORE _apply_revision swaps it; the
        # PlanRevisionDiff sidecar in _emit_plan_revised diffs the
        # two.
        prev_plan = session.plan
        attempt_id = self._new_attempt_id()
        await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
        # goldfive#247: rebind to the stamped instance.
        revised_plan = self._apply_revision(session, revised_plan, drift)
        await self._cancel_inflight_for_revision(drift, session)
        await self._emit_plan_revised(
            session,
            revised_plan,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
        )
        return True

    async def apply_user_steer_with_plan(
        self,
        *,
        drift: DriftEvent,
        session: Session,
        revised_plan: Plan,
    ) -> bool:
        """Back-compat shim — prefer :meth:`install_revision_for_drift`
        or :meth:`install_revision_for_user_steer` instead.

        Routes based on ``drift.kind`` + ``drift.raw``:

        * ``USER_STEER`` with ``raw`` populated → routed to
          :meth:`install_revision_for_user_steer`. The ``raw`` from
          the supplied drift is forwarded; ``drift.detail`` /
          ``drift.authored_by`` are ignored (the new API rebuilds
          them from ``raw`` deterministically).
        * ``USER_STEER`` with ``raw is None`` → was the
          :meth:`Runner._install_revision` synthetic install path
          before Option A. The new Runner path no longer reaches this
          shim; callers in this state probably mean
          :meth:`install_initial_plan` (turn 1) or
          :meth:`install_revision_for_drift` with a real drift kind
          (turn N+1). Routed defensively to ``install_initial_plan``
          when ``session.plan`` is empty, otherwise to
          ``install_revision_for_drift`` with a synthesized
          ``NEW_WORK_DISCOVERED`` drift so legacy callers keep
          working — but a deprecation warning fires.
        * Any other drift kind → routed to
          :meth:`install_revision_for_drift`.

        Slated for removal once external callers migrate.
        """
        import warnings

        warnings.warn(
            "DefaultSteerer.apply_user_steer_with_plan is deprecated; "
            "use install_initial_plan / install_revision_for_drift / "
            "install_revision_for_user_steer (goldfive#271 Option A).",
            DeprecationWarning,
            stacklevel=2,
        )
        if drift.kind is DriftKind.USER_STEER and getattr(drift, "raw", None) is not None:
            return await self.install_revision_for_user_steer(
                session=session,
                raw=drift.raw,
                revised_plan=revised_plan,
            )
        if drift.kind is DriftKind.USER_STEER:
            # Legacy synthetic-install path. Pick the new-API equivalent.
            if session.plan is None or not session.plan.tasks:
                return await self.install_initial_plan(
                    session=session, plan=revised_plan
                )
            replan_drift = DriftEvent(
                kind=DriftKind.NEW_WORK_DISCOVERED,
                severity=DriftSeverity.INFO,
                detail=drift.detail,
                authored_by="goldfive",
            )
            return await self.install_revision_for_drift(
                session=session,
                drift=replan_drift,
                revised_plan=revised_plan,
            )
        return await self.install_revision_for_drift(
            session=session, drift=drift, revised_plan=revised_plan
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

    # --- Refine outcome tracking (goldfive#215 iter-8 P2) ------------
    #
    # Single outcome-tracked state machine replacing the split
    # ``refine_failure_counts`` (numerical cap) +
    # ``KEY_ACTIVE_DRIFTS`` lifecycle gate. Reset every turn boundary
    # by :meth:`reset_for_turn`.

    async def _record_refine_outcome(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        succeeded: bool,
    ) -> None:
        """Record the outcome of a refine attempt for ``(kind, task)``.

        On ``succeeded=True`` writes a ``RefineOutcome(state="succeeded",
        fail_count=0)`` entry. The "succeeded" state still encodes the
        "attempted" signal so a follow-up same-(kind, task) drift on
        the same turn skips refine — the prior refine already produced
        a landed revision, re-running it is a no-op replay.

        On ``succeeded=False`` increments ``fail_count`` (or initialises
        to 1 if no prior failure entry) and, when the count crosses
        :attr:`REFINE_FAILURE_THRESHOLD`, marks the offending task
        FAILED (non-recoverable) and emits a CRITICAL
        ``REPEATED_FAILURE`` drift directly (NOT through
        ``_handle_drift`` — the REPEATED_FAILURE drift keys on a
        different (kind, task) tuple than the source so it does not
        feed back into this counter).

        ``USER_STEER`` / ``USER_CANCEL`` / ``GOAL_DRIFT`` bypass the
        write entirely — operator intent must always be honoured and
        trajectory-level drifts have their own rate limiters.
        """
        if drift.kind in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            return
        key = (drift.kind.value, drift.current_task_id or "")
        if succeeded:
            session.refine_outcomes[key] = RefineOutcome(state="succeeded", fail_count=0)
            return
        prior = session.refine_outcomes.get(key)
        new_count = (prior.fail_count + 1) if prior is not None and prior.state == "failed" else 1
        session.refine_outcomes[key] = RefineOutcome(state="failed", fail_count=new_count)
        if new_count < self.REFINE_FAILURE_THRESHOLD:
            return
        # Crossed the threshold: mark the offending task FAILED (which
        # routes through _handle_drift on a TASK_FAILED_FATAL key —
        # different (kind, task) tuple, so no recursion into this
        # counter) and emit REPEATED_FAILURE directly via
        # _emit_drift_detected (NOT _handle_drift, which would try to
        # refine again on the fresh drift). See TASK-LIFECYCLE.md §7.3.
        task_id = drift.current_task_id
        reason = f"refine repeatedly failed for {drift.kind.value}"
        if task_id:
            await self.mark_task_failed(
                task_id,
                reason=reason,
                recoverable=False,
                session=session,
            )
        repeated = DriftEvent(
            kind=DriftKind.REPEATED_FAILURE,
            severity=DriftSeverity.CRITICAL,
            detail=(
                f"refine failed {new_count} consecutive times for "
                f"{drift.kind.value} (task {task_id or 'n/a'})"
            ),
            current_task_id=task_id,
            current_agent_id=drift.current_agent_id,
        )
        await self._emit_drift_detected(session, repeated)

    def reset_for_turn(self, session: Session) -> None:
        """Clear per-turn refine-outcome bookkeeping.

        Wired from :meth:`Runner.run` immediately after the
        ``run_started`` event so each turn starts with an empty
        outcome table. The (kind, task) retry budget is naturally
        per-turn — a wedged drift from a prior turn should not
        carry over its failure count and short-circuit a fresh
        refine attempt on the new turn.
        """
        session.refine_outcomes.clear()

    def _occurrence_count_for_ladder(self, session: Session, drift: DriftEvent) -> int:
        """Return the per-(kind, task) failure count consumed by the ladder.

        Maps ``RefineOutcome`` back onto the int the
        :meth:`_ladder_level_for` table reads. ``"succeeded"`` returns
        ``0`` so a fresh same-(kind, task) drift is treated as the
        first occurrence (the gate above the ladder will short-circuit
        anyway, but keeping the ladder invariant intact is cheaper
        than re-deriving the ``is_repeat`` semantics inside the ladder).
        """
        outcome = session.refine_outcomes.get((drift.kind.value, drift.current_task_id or ""))
        if outcome is None or outcome.state == "succeeded":
            return 0
        return outcome.fail_count

    async def _escalate_refine_failure_as_critical_drift(
        self, session: Session, source: DriftEvent, *, reason: str
    ) -> None:
        """Surface a failed refine as a follow-up CRITICAL drift.

        Reuses the source drift's kind and prefixes ``detail`` with
        ``refine failed`` so sinks (and the harmonograf UI) get a durable,
        CRITICAL signal that a prior drift's refine did not succeed —
        without this event, a silently-swallowed refine leaves the
        session pinned to the stale plan and the executor re-enters the
        same state on the next tick.
        """
        failure = DriftEvent(
            kind=source.kind,
            severity=DriftSeverity.CRITICAL,
            detail=f"refine failed ({source.kind.value}): {reason}",
            current_task_id=source.current_task_id,
            current_agent_id=source.current_agent_id,
        )
        await self._emit_drift_detected(session, failure)

    # --- Refine atomicity + observability (goldfive a4) --------------

    def _get_plan_lock(self, session: Session) -> asyncio.Lock:
        """Return the per-session plan-state mutation lock, creating on first use.

        Keyed by ``session.id``. Multiple Sessions on the same Steerer
        each get an independent lock so concurrent runs never serialise
        on each other. The dict is unbounded for the steerer's lifetime;
        live runs share a steerer with a small number of sessions so
        this is acceptable. (If a future use-case introduces churn —
        many short-lived sessions — add a cleanup hook on session end.)
        """
        sid = session.id or session.run_id or ""
        lock = self._plan_locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._plan_locks[sid] = lock
        return lock

    async def _wait_plan_stable(
        self,
        session: Session,
        *,
        timeout: float | None = 1.0,
    ) -> bool:
        """Block until the per-session plan-state mutation region is idle.

        Acquires + immediately releases the per-session plan lock so the
        caller observes either pre-revision or post-revision plan state
        — never a partial apply. Used by report_task_* handlers and
        ``_resolve_effective_task_id`` callers to coordinate with the
        fire-and-forget judge-triggered refines introduced in #254.

        Returns ``True`` when the wait completed cleanly; ``False`` when
        ``timeout`` elapsed (in which case the caller MUST proceed
        anyway — atomicity is best-effort, not a hard barrier — and the
        worst case degrades to the pre-fix racy read). The default
        timeout is intentionally short (1s): a refine's mutation region
        is bounded by a handful of in-memory operations, so a timeout
        here means something pathological is happening and blocking
        a report indefinitely is worse than a stale read.

        ``timeout=None`` waits forever. Pass a positive float to bound
        the wait. Passing ``timeout<=0`` returns immediately with the
        lock-free reading semantics (does not check the lock at all);
        callers wanting a strict barrier should use a positive timeout.
        """
        if timeout is not None and timeout <= 0:
            return True
        lock = self._get_plan_lock(session)
        if not lock.locked():
            return True
        try:
            if timeout is None:
                async with lock:
                    pass
                return True
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            try:
                pass
            finally:
                lock.release()
            return True
        except TimeoutError:
            log.warning(
                "DefaultSteerer._wait_plan_stable: timed out after %.2fs "
                "waiting for plan lock on session %s; proceeding with "
                "best-effort racy read",
                timeout,
                session.id,
            )
            return False

    @staticmethod
    def _new_attempt_id() -> str:
        """Mint a fresh refine-attempt UUID for correlation between
        ``refine_attempted`` and the paired ``refine_failed`` /
        ``plan_revised`` events.
        """
        return str(uuid.uuid4())

    async def _emit_refine_attempted(
        self,
        session: Session,
        drift: DriftEvent,
        *,
        attempt_id: str,
    ) -> None:
        """Emit a ``refine_attempted`` dict envelope onto the sink bus.

        Fired at the start of a refine call (both the autonomous
        ``_handle_drift`` path and the goldfive-steer
        ``_promote_drift_to_steer`` path). Pairs with exactly one of
        ``refine_failed`` / ``plan_revised`` carrying the same
        ``attempt_id``. Dict envelope (not proto) — promote to proto
        when the Stream C (#256) follow-up gets prioritised.
        """
        from goldfive.events import emit, make_event

        drift_id = str(getattr(drift, "id", "") or "")
        payload = {
            "attempt_id": attempt_id,
            "drift_id": drift_id,
            "trigger_kind": drift.kind.value,
            "trigger_severity": drift.severity.value,
            "current_task_id": drift.current_task_id or "",
            "current_agent_id": drift.current_agent_id or "",
        }
        try:
            evt = make_event(
                session.run_id,
                session.next_sequence(),
                "refine_attempted",
                payload,
                session_id=session.id,
            )
            await emit(self._sinks, evt)
        except Exception as exc:  # noqa: BLE001 — observability must never break the run
            log.debug(
                "DefaultSteerer._emit_refine_attempted: failed to emit: %s",
                exc,
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
        """Emit a ``refine_failed`` dict envelope onto the sink bus.

        ``failure_kind`` is one of ``parse_error`` / ``validator_rejected``
        / ``llm_error`` / ``other`` (string, not enum, so the surface is
        forward-compatible without proto changes). ``reason`` is a short
        human-readable summary; ``detail`` may carry a longer
        free-form payload (e.g. the validator's exception text).
        Crucially, this event is emitted WITHOUT bumping
        ``revision_index`` — the attempt_id disambiguates failures
        across otherwise-incrementing revisions.
        """
        from goldfive.events import emit, make_event

        drift_id = str(getattr(drift, "id", "") or "")
        payload = {
            "attempt_id": attempt_id,
            "drift_id": drift_id,
            "trigger_kind": drift.kind.value,
            "trigger_severity": drift.severity.value,
            "failure_kind": failure_kind,
            "reason": reason,
            "detail": detail,
            "current_task_id": drift.current_task_id or "",
            "current_agent_id": drift.current_agent_id or "",
        }
        try:
            evt = make_event(
                session.run_id,
                session.next_sequence(),
                "refine_failed",
                payload,
                session_id=session.id,
            )
            await emit(self._sinks, evt)
        except Exception as exc:  # noqa: BLE001 — observability must never break the run
            log.debug(
                "DefaultSteerer._emit_refine_failed: failed to emit: %s",
                exc,
            )

    @contextlib.asynccontextmanager
    async def observe_refine(
        self,
        session: Session,
        drift: DriftEvent,
    ) -> AsyncIterator[str]:
        """Async context manager that wraps a ``planner.refine`` call with
        observability emission.

        On enter:

        * Mints a fresh ``attempt_id``.
        * Stamps the per-async-task ``_active_session_var`` ContextVar
          to ``session`` so the planner's ``_span_ctx_provider`` resolves
          correctly (this is what powers the planner-side
          ``refine_orphaned_tasks`` emission and the
          ``GoldfiveLLMCallStart/End`` spans). ContextVar isolation
          keeps concurrent runs sharing one Steerer from stomping each
          other's session pointer.
        * Emits ``refine_attempted`` to the bound sinks.

        On exception:

        * Emits ``refine_failed`` with ``failure_kind="llm_error"``,
          stamped with the same ``attempt_id``, then re-raises.

        On clean exit (no exception):

        * Resets ``_active_session_var``.
        * Caller is responsible for emitting either ``plan_revised``
          (success) or ``refine_failed`` (returned ``None`` / validator
          rejected) — the helper has no way to introspect the caller's
          decision tree from here. Pair with :meth:`_emit_refine_failed`
          / ``_emit_plan_revised`` using the yielded ``attempt_id``.

        Used by:

        * :meth:`_handle_drift` / :meth:`_promote_drift_to_steer` —
          the steerer's own refine call sites.
        * :class:`~goldfive.executors.parallel.ParallelDAGExecutor._refine` —
          the executor-side refine fallback. Without this helper, the
          parallel path's refines emit no ``refine_attempted`` /
          ``refine_failed`` / ``refine_orphaned_tasks`` events, since
          they bypass the steerer's hand-rolled emission blocks.
        """
        attempt_id = self._new_attempt_id()
        # Setting the per-async-task ``_active_session_var`` ContextVar
        # before refine lets the planner's internal
        # ``_emit_refine_orphaned_tasks`` resolve a sink target via the
        # bound span-context provider. Without this, the planner's
        # validator computes orphans, logs the WARNING, but no sink event
        # lands — exactly the symptom Bug A describes.
        _active_session_token = self._active_session_var.set(session)
        # Phase 3.5 (goldfive#271) tripwire wrapper — see §C4. The
        # ``except BaseException: stash; raise`` arm below is the
        # compliance branch (CANCELLATION-CONTRACT.md §1.2).
        with _state_audit.cancellation_stash_audited("DefaultSteerer.observe_refine"):
            try:
                await self._emit_refine_attempted(session, drift, attempt_id=attempt_id)
                try:
                    yield attempt_id
                except Exception as exc:  # noqa: BLE001 — refine errors must not break observability
                    # Emit failure event with the same attempt_id so consumers
                    # can pair attempted ↔ failed. We do NOT swallow the
                    # exception — re-raise so the caller's existing error path
                    # (e.g. _escalate_refine_failure_as_critical_drift / fallback plans) runs.
                    await self._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="llm_error",
                        reason=str(exc),
                        detail=type(exc).__name__,
                    )
                    raise
                except BaseException as exc:  # noqa: BLE001
                    # Phase 3.5 (CANCELLATION-CONTRACT.md §C4): ``CancelledError``
                    # is a ``BaseException`` (not ``Exception``) since Py 3.8, so
                    # the ``except Exception`` branch above does NOT catch it. If
                    # a refine is cancelled mid-flight (e.g. ADK closes the
                    # runner, harness interrupts the loop) the paired
                    # ``refine_failed`` observability event would be skipped,
                    # leaving sinks with an unmatched ``refine_attempted``.
                    # Emit the pair-completing failure event AND re-raise so
                    # cancellation still propagates per the asyncio contract.
                    await self._emit_refine_failed(
                        session,
                        drift,
                        attempt_id=attempt_id,
                        failure_kind="cancelled",
                        reason=f"refine cancelled: {type(exc).__name__}",
                        detail=type(exc).__name__,
                    )
                    # Phase 3.5 tripwire compliance marker (§1.2 form).
                    _state_audit.mark_stash_completed()
                    raise
            finally:
                self._active_session_var.reset(_active_session_token)

    async def _emit_plan_revised_correlation(
        self,
        session: Session,
        revised: Plan,
        drift: DriftEvent,
        *,
        attempt_id: str,
    ) -> None:
        """Emit a ``plan_revised`` dict envelope stamped with ``attempt_id``.

        Companion to the proto ``PlanRevised`` event so dict-event
        consumers correlate successful refines with their preceding
        ``refine_attempted`` event. The proto event carries the full
        payload for primary consumers; this dict envelope is purely a
        correlation side-car. When the Stream C (#256) proto follow-up
        promotes ``attempt_id`` onto ``PlanRevised``, this emitter goes
        away.
        """
        from goldfive.events import emit, make_event

        drift_id = str(getattr(drift, "id", "") or "")
        payload = {
            "attempt_id": attempt_id,
            "drift_id": drift_id,
            "trigger_kind": drift.kind.value,
            "trigger_severity": drift.severity.value,
            "revision_index": int(revised.revision_index),
            "current_task_id": drift.current_task_id or "",
            "current_agent_id": drift.current_agent_id or "",
        }
        try:
            evt = make_event(
                session.run_id,
                session.next_sequence(),
                "plan_revised",
                payload,
                session_id=session.id,
            )
            await emit(self._sinks, evt)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._emit_plan_revised_correlation: failed to emit: %s",
                exc,
            )

    @staticmethod
    def _fold_runtime_terminal_statuses(revised: Plan, prior: Plan | None) -> Plan:
        """Fold runtime terminal statuses from ``prior`` onto ``revised``.

        The persistence-boundary fix for the I4 phantom-state class of
        bugs (escalation report iter_1 §I4, v24 session
        ``2a324f78``): runtime terminal transitions emitted out-of-band
        between revisions — the overlay-reaper's NOT_NEEDED reap, the
        SequentialExecutor's reachability-audit cancels, an explicit
        ``mark_task_*`` call from a coordinator's reporting-tool — all
        flip the live plan's task status, but the next
        ``planner.refine`` / ``planner.handle_turn`` invocation builds
        its candidate plan from the LLM's view, which may have lost or
        regressed those terminal statuses.

        For each task ``t`` in ``revised`` whose id matches a task in
        ``prior`` with a status in :data:`TERMINAL_TASK_STATUSES`:

        * If ``revised``'s entry is non-terminal (PENDING / RUNNING /
          BLOCKED), OVERWRITE its status with the prior terminal status
          and copy ``cancel_reason`` so the persisted snapshot matches
          what actually happened.
        * If ``revised`` already carries the same terminal status, no-op.
        * If ``revised`` carries a *different* terminal status — a
          genuine regression we must NOT silently rewrite — leave it
          alone so the validator catches it.

        Returns a NEW :class:`Plan` (goldfive#247: Plan is frozen). When
        no folds are needed the input is returned unchanged so callers
        can share the reference. The fold list is logged at INFO when
        non-empty for downstream observability.

        Anti-pattern note: this is **not** a validator relaxation. The
        validator (``Plan.validate(for_revision=True, prior=...)``)
        remains the source of truth for terminal-task preservation. The
        fold corrects the LLM's output to match runtime reality
        BEFORE validation runs, so the validator only fires on a true
        regression (e.g. terminal→different-terminal) rather than on
        an ordinary "the LLM forgot a NOT_NEEDED reap fired since its
        prompt was authored."
        """
        if prior is None or not getattr(prior, "tasks", None):
            return revised
        prior_terminal: dict[str, Task] = {
            t.id: t
            for t in prior.tasks
            if t.id and t.status in TERMINAL_TASK_STATUSES
        }
        if not prior_terminal:
            return revised
        new_tasks: list[Task] = []
        folded: list[str] = []
        for t in revised.tasks:
            prior_t = prior_terminal.get(t.id)
            if prior_t is None:
                new_tasks.append(t)
                continue
            if t.status is prior_t.status:
                new_tasks.append(t)
                continue
            if t.status in TERMINAL_TASK_STATUSES:
                # Different terminal in the revised plan — a genuine
                # regression. Do not silently rewrite; let the
                # validator surface it as SCHEMA_VIOLATION.
                new_tasks.append(t)
                continue
            # Non-terminal in revised, terminal in prior → fold.
            replacement = dataclasses.replace(
                t,
                status=prior_t.status,
                cancel_reason=t.cancel_reason or prior_t.cancel_reason,
            )
            new_tasks.append(replacement)
            folded.append(t.id)
        if not folded:
            return revised
        log.info(
            "DefaultSteerer._fold_runtime_terminal_statuses: "
            "folded %d task(s) from prior runtime state: %s",
            len(folded),
            ", ".join(folded),
        )
        return dataclasses.replace(revised, tasks=tuple(new_tasks))

    @staticmethod
    def _apply_revision(session: Session, revised: Plan, drift: DriftEvent) -> Plan:
        """Stamp revision metadata and install ``revised`` on the session.

        Preserves the existing ``revision_index`` monotonicity: the new
        plan's index is at least ``old.revision_index + 1``.

        goldfive#247: returns the post-stamp Plan that was actually
        installed onto :attr:`Session.plan`. Pre-#247 the function
        mutated ``revised`` in place AND ``session.plan = revised``, so
        callers who reused their local ``revised`` reference saw the
        stamped metadata. With frozen Plan, the stamp produces a NEW
        instance; the helper returns it so callers can rebind their
        local variable and pass the same instance to
        :meth:`_emit_plan_revised`.

        Phase 2.X / goldfive#271 Gap 2: log the install at INFO so the
        prior_plan_id → revised_plan_id transition is grep-able in the
        demo log. The validation E2E found 2 of 4 task_plans rows
        without corresponding plan events; without this log line a
        silent install (e.g. an exception in ``_emit_plan_revised``
        right after) leaves no goldfive-side trace of the swap.

        Defensive fold (I4 fix): re-applies
        :meth:`_fold_runtime_terminal_statuses` against ``session.plan``
        even though install paths fold before validation. Idempotent —
        a no-op if the caller already folded — but a last-line guard
        against any future install path that forgets to fold before
        calling here.
        """
        prev = session.plan
        # I4 fix (defensive): fold runtime terminal statuses even if the
        # caller already did. Idempotent — if every task's status
        # already matches prior's terminal, this is a no-op. Returns a
        # NEW Plan (goldfive#247: Plan is frozen).
        revised = DefaultSteerer._fold_runtime_terminal_statuses(revised, prev)
        prior_id = (getattr(prev, "id", "") or "") if prev is not None else ""
        next_index = (prev.revision_index + 1) if prev is not None else 1
        # goldfive#247: Plan is frozen — derive a new instance with the
        # stamped revision metadata via :func:`bump_revision`. Preserves
        # caller-supplied non-empty values (matches the legacy
        # "only set if blank" guards).
        new_index = max(int(revised.revision_index), next_index)
        new_kind = revised.revision_kind or drift.kind.value
        new_severity = revised.revision_severity or drift.severity.value
        new_reason = revised.revision_reason or drift.detail
        # goldfive#199: stamp the trigger_event_id from the drift onto the
        # plan so out-of-band PlanRevised emitters (the SequentialExecutor's
        # plan-swap detector) can thread it through without needing the
        # drift in scope. Resolution mirrors
        # :func:`goldfive.events._trigger_id_from_drift`: source
        # annotation_id for user-control drifts, ``drift.id`` otherwise.
        # Non-empty for every revision because every ``DriftEvent``
        # dataclass defaults to a UUID4 ``id``. Preserves any pre-existing
        # stamp (e.g. validator-retry chains that re-use the original
        # attempt's trigger id).
        new_trigger_id = revised.revision_trigger_event_id
        if not new_trigger_id:
            new_trigger_id = DefaultSteerer._drift_annotation_id(drift) or str(
                getattr(drift, "id", "") or ""
            )
        revised = bump_revision(
            revised,
            revision_index=new_index,
            revision_kind=new_kind,
            revision_severity=new_severity,
            revision_reason=new_reason,
            revision_trigger_event_id=new_trigger_id,
        )
        log.info(
            "DefaultSteerer._apply_revision: prior_plan_id=%s "
            "revised_plan_id=%s revision_index=%d drift_kind=%s",
            prior_id[:16] or "<none>",
            (revised.id or "")[:16] or "<empty>",
            int(revised.revision_index),
            drift.kind.value,
        )
        with channel_processor_active():
            set_session_plan(session, revised)
        # goldfive#245 follow-up — stamp the per-(kind, target) addressed
        # watermark so the verdict-freshness gate in :meth:`_handle_drift`
        # can drop subsequent same-(kind, target) verdicts observed at
        # older revisions as redundant. User-authored drifts bypass the
        # gate entirely so they don't stamp here.
        if (drift.authored_by or "").lower() != "user":
            key = (drift.kind.value, drift.current_task_id or "")
            session.last_addressed_revision_by_drift_key[key] = int(
                revised.revision_index
            )
        # goldfive#152: refresh the orchestration-state current plan id
        # so downstream reads see the revised id, not the stale one.
        _ostate.set_current_plan(session.state, revised)
        return revised

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

    async def _emit_task_started(self, session: Session, task_id: str, detail: str) -> None:
        evt = self._new_envelope(session)
        evt.task_started.task_id = task_id
        evt.task_started.detail = detail
        await self._emit(evt)

    async def _emit_task_progress(
        self, session: Session, task_id: str, fraction: float, detail: str
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_progress.task_id = task_id
        evt.task_progress.fraction = fraction
        evt.task_progress.detail = detail
        await self._emit(evt)

    async def _emit_task_completed(
        self,
        session: Session,
        task_id: str,
        summary: str,
        artifacts: dict[str, str],
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_completed.task_id = task_id
        evt.task_completed.summary = summary
        for k, v in artifacts.items():
            evt.task_completed.artifacts[k] = v
        await self._emit(evt)

    async def _emit_task_failed(
        self, session: Session, task_id: str, reason: str, recoverable: bool
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_failed.task_id = task_id
        evt.task_failed.reason = reason
        evt.task_failed.recoverable = recoverable
        await self._emit(evt)

    async def _emit_task_blocked(
        self, session: Session, task_id: str, blocker: str, needed: str
    ) -> None:
        evt = self._new_envelope(session)
        evt.task_blocked.task_id = task_id
        evt.task_blocked.blocker = blocker
        evt.task_blocked.needed = needed
        await self._emit(evt)

    async def _emit_task_cancelled(self, session: Session, task_id: str, reason: str) -> None:
        evt = self._new_envelope(session)
        evt.task_cancelled.task_id = task_id
        evt.task_cancelled.reason = reason
        await self._emit(evt)

    async def _emit_task_transitioned(
        self,
        session: Session,
        task: Task,
        *,
        from_status: TaskStatus,
        to_status: TaskStatus,
        source: str,
    ) -> None:
        """Emit a ``TaskTransitioned`` envelope (goldfive#251 R4).

        Sink-only observability. Called from every site that mutates a
        task's status — both the imperative ``mark_task_*`` path and
        the cascade path inside :meth:`cascade_cancel_downstream`. The
        LLM never sees this event; the ``report_task_*`` surface still
        returns ``{"acknowledged": True}``.

        Source attribution is the caller's responsibility (defaults to
        ``"other"`` on un-threaded callers); see
        :func:`goldfive.events.task_transitioned_event` for the
        vocabulary.

        ``agent_name`` resolves to ``task.assignee_agent_id``; that's
        the most stable surface goldfive owns. ``invocation_id`` is a
        best-effort lookup against the reconciler's
        ``_invocation_agent`` map (goldfive#151) when available; empty
        when no in-flight invocation matches the assignee. Tolerant of
        missing maps / proto stubs — emission failures are swallowed
        with a debug log rather than breaking the transition path.
        """
        # goldfive#271: stamp task progress liveness on every transition
        # so the structural progress-stall escalation gate sees the task
        # as productively iterating. Done BEFORE the sink check because
        # progress liveness must be tracked even when sinks are missing
        # (test scenarios) — the gate consults this map regardless.
        task_id_for_progress = str(getattr(task, "id", "") or "")
        if task_id_for_progress:
            session.task_last_progress_at[task_id_for_progress] = time.monotonic()
        sinks = self._sinks
        if not sinks:
            return
        try:
            from goldfive.events import emit, task_transitioned_event
        except Exception as exc:  # noqa: BLE001 — proto stubs may be missing
            log.debug(
                "DefaultSteerer._emit_task_transitioned: events module unavailable: %s",
                exc,
            )
            return

        agent_name = str(getattr(task, "assignee_agent_id", "") or "")
        invocation_id = self._resolve_invocation_id_for_agent(agent_name)
        revision_stamp = 0
        plan = getattr(session, "plan", None)
        if plan is not None:
            try:
                revision_stamp = int(getattr(plan, "revision_index", 0) or 0)
            except (TypeError, ValueError):
                revision_stamp = 0
        try:
            evt = task_transitioned_event(
                session.run_id,
                session.next_sequence(),
                task_id=str(getattr(task, "id", "") or ""),
                from_status=str(getattr(from_status, "value", from_status) or ""),
                to_status=str(getattr(to_status, "value", to_status) or ""),
                source=str(source or "other"),
                revision_stamp=revision_stamp,
                agent_name=agent_name,
                invocation_id=invocation_id,
                session_id=session.id,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._emit_task_transitioned: proto event build failed: %s",
                exc,
            )
            return
        try:
            await emit(sinks, evt)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._emit_task_transitioned: sink emit raised: %s",
                exc,
            )

    async def _emit_plan_revision_transitions(
        self,
        session: Session,
        prev_plan: Plan | None,
        revised: Plan,
    ) -> None:
        """Emit ``TaskTransitioned`` events for status changes carried by a refine.

        Compares ``prev_plan`` vs ``revised`` task-by-task and emits one
        ``TaskTransitioned`` event per task whose status changed (or
        whose ``status`` is now non-PENDING and the task didn't exist
        in ``prev_plan`` — a refine-introduced task that arrived in a
        non-PENDING state, e.g. a CORRECT-kind successor that the
        planner pre-stamped).

        Source is always ``"plan_revision"``: the refine is the
        authoritative driver. Tasks that exist in both plans with the
        same status are skipped (no transition happened).

        ``prev_plan`` may be ``None`` on the first revision after a run
        with no initial plan; in that case every task in ``revised``
        with non-PENDING status emits a "(implicit) PENDING ->
        actual_status" event so operators see the post-revision state
        on the wire.
        """
        if not self._sinks:
            return
        prev_by_id: dict[str, Task] = {}
        if prev_plan is not None:
            for t in getattr(prev_plan, "tasks", []) or []:
                tid = str(getattr(t, "id", "") or "")
                if tid:
                    prev_by_id[tid] = t
        for t in getattr(revised, "tasks", []) or []:
            tid = str(getattr(t, "id", "") or "")
            if not tid:
                continue
            new_status = getattr(t, "status", None)
            if not isinstance(new_status, TaskStatus):
                continue
            old = prev_by_id.get(tid)
            if old is None:
                old_status: TaskStatus = TaskStatus.PENDING
            else:
                old_status = getattr(old, "status", TaskStatus.PENDING)
            if old_status == new_status:
                continue
            # No transition to record when the new status is the
            # default PENDING and the task is brand-new — sinks would
            # render that as a phantom "started in PENDING" row.
            if old is None and new_status is TaskStatus.PENDING:
                continue
            await self._emit_task_transitioned(
                session,
                t,
                from_status=old_status,
                to_status=new_status,
                source="plan_revision",
            )

    def _resolve_invocation_id_for_agent(self, agent_name: str) -> str:
        """Best-effort lookup of an active invocation_id for ``agent_name``.

        Mirrors the reconciler-walk pattern in
        :meth:`_resolve_active_invocation_ids` but scopes the match to a
        single agent name (the assignee of the transitioning task). The
        most-recent matching invocation_id wins; empty string when no
        match (no reconciler, no in-flight invocation under that
        agent, etc.). Tolerant of every failure mode — never raises.
        """
        if not agent_name:
            return ""
        adapter = self._adapter
        plugin = getattr(adapter, "_plugin", None) if adapter is not None else None
        reconciler = getattr(plugin, "_reconciler", None) if plugin is not None else None
        if reconciler is None:
            return ""
        try:
            inv_agent = getattr(reconciler, "_invocation_agent", None)
            if not isinstance(inv_agent, Mapping):
                return ""
            # Iterate insertion-order; later writes win.
            match = ""
            for inv_id, name in inv_agent.items():
                if name == agent_name and inv_id:
                    match = str(inv_id)
            return match
        except Exception:  # noqa: BLE001
            return ""

    async def _emit_drift_detected(self, session: Session, drift: DriftEvent) -> None:
        evt = self._new_envelope(session)
        evt.drift_detected.kind = self._drift_kind_pb_value(drift.kind)
        evt.drift_detected.severity = self._drift_severity_pb_value(drift.severity)
        evt.drift_detected.detail = drift.detail
        evt.drift_detected.current_task_id = drift.current_task_id
        evt.drift_detected.current_agent_id = drift.current_agent_id
        # goldfive#245 — forward the observation-time plan revision so
        # downstream sinks can render "this verdict was against
        # revision N, current is M" and dedup gate-skipped drifts.
        if drift.observed_revision_index:
            evt.drift_detected.observed_revision_index = int(
                drift.observed_revision_index
            )
        # goldfive-steer-unification: source attribution. Normalise a
        # missing ``authored_by`` on the drift here so downstream sinks
        # never see an unattributed event from goldfive-internal paths
        # (the ladder dispatcher normalises pre-emit; this is a belt-
        # and-braces for direct ``_emit_drift_detected`` callers like
        # ``_dispatch_pause_escalate`` / ``_escalate_refine_failure_as_critical_drift``).
        evt.drift_detected.authored_by = self._resolve_authored_by(drift)
        evt.drift_detected.suppressed_by_user_steer = bool(drift.suppressed_by_user_steer)
        # goldfive#199: stamp the drift's own id on the wire so a
        # subsequent ``PlanRevised.trigger_event_id`` can strict-match the
        # drift row in harmonograf. Always non-empty — ``DriftEvent``
        # defaults ``id`` to a UUID4.
        drift_id = str(getattr(drift, "id", "") or "")
        if drift_id:
            evt.drift_detected.id = drift_id
        # Stamp the source annotation_id for USER_STEER / USER_CANCEL drifts
        # minted from a ControlMessage with a bridge-supplied annotation_id
        # (goldfive#171). Sinks use this to dedup the drift row against the
        # source annotation — without it a single user STEER surfaces as
        # three cards (annotation row + drift row + plan_revised row) in
        # harmonograf's Intervention view. See goldfive#176 / harmonograf#75.
        ann_id = self._drift_annotation_id(drift)
        if ann_id:
            evt.drift_detected.annotation_id = ann_id
        # Forward the detector-supplied trigger_input onto the wire so
        # sinks that render a Gantt / timeline can explain "why did
        # goldfive flag this?" without re-fetching raw agent transcripts.
        # Always truncated by the detector before it lands on the drift;
        # we belt-and-braces truncate here in case an out-of-tree
        # detector forgot. Empty string for user-control drifts (their
        # explanation lives on the source annotation).
        trigger_input = getattr(drift, "trigger_input", "") or ""
        if trigger_input:
            evt.drift_detected.trigger_input = self._truncate_trigger_input(trigger_input)
        # goldfive#271 PR1 — drift-as-stateful-condition. Route the emit
        # through the orchestration_state lifecycle helpers so multiple
        # emits for the same logical condition (kind+task+agent within
        # the current turn) collapse onto one ``condition_id`` and the
        # wire carries lifecycle / prev_severity. Additive: legacy
        # fields (kind/severity/detail/synthetic/id/...) are unchanged
        # so any sink that doesn't know the new fields still sees one
        # row per emit and renders it identically.
        self._stamp_drift_lifecycle(session, drift, evt)
        await self._emit(evt)
        # goldfive#271 follow-up: when a terminal drift fires the run
        # cannot recover on its own — any boundary still open at this
        # point belongs to an invocation that will not get a paired
        # ``after_agent_callback`` (the executor is about to pause or
        # tear down). Walk the plugin's still-open boundaries and emit
        # the paired ``InvocationBoundaryExited(reason=terminal_drift:
        # <kind>)`` so observability sinks (and harmonograf's Gantt)
        # don't render permanently-open spans for coordinator /
        # research / refine_steer LLM_CALLs that v15 left in
        # ``dur=(open)``.
        if self._is_terminal_drift(drift):
            await self._close_open_boundaries_for_terminal_drift(drift)

    # goldfive#271 follow-up: drift kinds that are unrecoverable on
    # emit. Boundary cleanup hooks fire on these to close any
    # still-open spans the cooperative-cancel path would otherwise
    # leave dangling.
    #
    # Inclusion rationale (and why ``LOOPING_REASONING`` is NOT here
    # despite being listed in the v15 stuck-span evidence):
    #
    # * ``HUMAN_INTERVENTION_REQUIRED`` — the ladder always emits this
    #   at CRITICAL with PAUSE_ESCALATE / TERMINATE semantics; the run
    #   pauses for an operator and no normal ``after_agent_callback``
    #   will fire on the open invocations.
    # * ``REPEATED_FAILURE`` — emitted from
    #   :meth:`_record_refine_failure` ONLY after the offending task is
    #   marked ``FAILED`` non-recoverable; the executor will not
    #   resume it.
    # * ``LOOPING_REASONING`` is deliberately NOT here despite being
    #   listed in the v15 evidence: it is graduated (INFO / WARNING /
    #   CRITICAL) and CRITICAL-first maps to ``NUDGE`` (recoverable —
    #   refine + corrective follow-up). Closing on the LOOPING_REASONING
    #   emission itself would corrupt the boundary pair when the run
    #   actually recovers. The CRITICAL-repeat path escalates to
    #   ``PAUSE_ESCALATE``, which emits a fresh
    #   ``HUMAN_INTERVENTION_REQUIRED`` drift; that emission triggers
    #   the close, so the v15 stuck-spans symptom is still cleaned up
    #   on the actual terminal step.
    _TERMINAL_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.HUMAN_INTERVENTION_REQUIRED,
            DriftKind.REPEATED_FAILURE,
        }
    )

    @classmethod
    def _is_terminal_drift(cls, drift: DriftEvent) -> bool:
        """Return True iff ``drift`` should trigger boundary cleanup.

        Membership-only check against :attr:`_TERMINAL_DRIFT_KINDS`;
        every kind in the set is unconditionally terminal at emit
        time. See the set definition for the rationale on which kinds
        are included (and why ``LOOPING_REASONING`` is NOT — its
        CRITICAL-first tier is still recoverable; the eventual
        ``HUMAN_INTERVENTION_REQUIRED`` emission on escalation triggers
        cleanup instead).
        """
        return drift.kind in cls._TERMINAL_DRIFT_KINDS

    async def _close_open_boundaries_for_terminal_drift(self, drift: DriftEvent) -> None:
        """Ask the bound adapter's plugin to close every still-open boundary.

        Reuses the canonical ``close_open_boundaries`` helper from
        PR #307 so the cleanup path is identical to the
        ``except CancelledError`` arc in
        :meth:`ADKAdapter._invoke_internal`. The reason string is
        ``terminal_drift:<kind>`` so sink consumers can distinguish a
        steerer-driven cleanup from the cancel / error paths.

        Best-effort: tolerates an unbound adapter, an adapter without
        the plugin attribute, a plugin without the helper (third-party
        / legacy), and any exception from the plugin (logged at DEBUG).
        Never re-raises — the drift was already emitted on the wire,
        and a failed cleanup must not corrupt the steerer's pause /
        escalate flow.
        """
        adapter = self._adapter
        if adapter is None:
            return
        plugin = getattr(adapter, "_plugin", None)
        if plugin is None:
            return
        helper = getattr(plugin, "close_open_boundaries", None)
        if not callable(helper):
            return
        reason = f"terminal_drift:{drift.kind.value}"
        try:
            await helper(reason=reason)
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            log.debug(
                "DefaultSteerer._close_open_boundaries_for_terminal_drift: "
                "plugin.close_open_boundaries(reason=%r) raised: %s",
                reason,
                exc,
            )

    def _stamp_drift_lifecycle(
        self,
        session: Session,
        drift: DriftEvent,
        evt: Any,
    ) -> None:
        """Stamp ``condition_id`` / ``lifecycle`` / ``prev_severity`` on ``evt``.

        Routes the emit through
        :func:`orchestration_state.open_or_escalate_drift` keyed by
        ``(kind, current_task_id, current_agent_id, run_id)``. The first
        emit for a given tuple in a turn opens a new condition and stamps
        ``DRIFT_LIFECYCLE_OPENED``; subsequent emits stamp
        ``DRIFT_LIFECYCLE_ESCALATING`` and carry the previous severity in
        ``prev_severity``. The drift's intrinsic ``id`` is NOT used as
        the condition key — the condition is a logical group that the
        same kind+task+agent can re-open within a turn, and the
        per-event id (#199) is intentionally distinct from the
        condition id (#271).

        Tolerant of partial state: a drift with empty ``current_task_id``
        / ``current_agent_id`` still produces a stable condition_id (the
        sha1 just hashes the empty strings), so user-control drifts
        without a pinned task collapse onto a single condition per turn
        per kind.

        Synthetic drifts are routed through the helpers as well so the
        wire still carries the lifecycle metadata; sinks that filter
        ``synthetic == true`` already drop them and continue to do so.
        """
        try:
            from goldfive.pb.goldfive.v1 import types_pb2

            turn_id = str(getattr(session, "run_id", "") or "")
            tracked = _ostate.open_or_escalate_drift(
                session.state,
                kind=drift.kind,
                task_id=str(getattr(drift, "current_task_id", "") or ""),
                agent_id=str(getattr(drift, "current_agent_id", "") or ""),
                turn_id=turn_id,
                severity=drift.severity,
            )
            evt.drift_detected.condition_id = tracked.condition_id
            evt.drift_detected.lifecycle = self._drift_lifecycle_pb_value(
                tracked.lifecycle, types_pb2
            )
            if tracked.prev_severity is not None:
                evt.drift_detected.prev_severity = self._drift_severity_pb_value(
                    tracked.prev_severity
                )
        except Exception as exc:  # noqa: BLE001
            # Lifecycle stamping is observability-only; never let a
            # bookkeeping bug break the wire emit. Log and fall through
            # to the legacy single-shot view (UNSPECIFIED lifecycle,
            # empty condition_id).
            log.debug("DefaultSteerer: drift-lifecycle stamping skipped (%s)", exc)

    @staticmethod
    def _drift_lifecycle_pb_value(lifecycle: str, types_pb2: Any) -> int:
        """Map an :mod:`orchestration_state` lifecycle string to the proto enum."""
        mapping = {
            _ostate.LIFECYCLE_OPENED: "DRIFT_LIFECYCLE_OPENED",
            _ostate.LIFECYCLE_ESCALATING: "DRIFT_LIFECYCLE_ESCALATING",
            _ostate.LIFECYCLE_RESOLVED: "DRIFT_LIFECYCLE_RESOLVED",
            _ostate.LIFECYCLE_HUMAN_INTERVENTION_REQUIRED: (
                "DRIFT_LIFECYCLE_HUMAN_INTERVENTION_REQUIRED"
            ),
        }
        name = mapping.get(lifecycle, "DRIFT_LIFECYCLE_UNSPECIFIED")
        return getattr(types_pb2, name, getattr(types_pb2, "DRIFT_LIFECYCLE_UNSPECIFIED", 0))

    # goldfive-steer-unification: drift kinds that are always "user"-
    # authored when no explicit source was stamped. Any other kind
    # defaults to "goldfive" (the detector path).
    _USER_AUTHORED_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.USER_PAUSE,
        }
    )

    @classmethod
    def _resolve_authored_by(cls, drift: DriftEvent) -> str:
        """Return the effective ``authored_by`` value for ``drift``.

        Honours an explicit value on the dataclass first; otherwise
        derives from the drift kind. User-control kinds → ``"user"``;
        everything else → ``"goldfive"`` (the detector path).
        """
        explicit = str(getattr(drift, "authored_by", "") or "").strip()
        if explicit:
            return explicit
        if drift.kind in cls._USER_AUTHORED_DRIFT_KINDS:
            return "user"
        return "goldfive"

    @staticmethod
    def _drift_annotation_id(drift: DriftEvent) -> str:
        """Return the source annotation id for a user-control drift, or "".

        Looks at :attr:`DriftEvent.raw` — populated by
        :meth:`_drift_from_control` when the drift was minted from a STEER
        / CANCEL ControlMessage — and extracts
        ``payload["annotation_id"]`` (set by the bridge per goldfive#171).
        Returns "" for drifts that goldfive minted itself (loop detection,
        goal drift, etc), whose ``raw`` is either absent or not a
        ControlMessage. Non-string payloads are coerced to str so a
        mis-typed bridge still flows the id through.
        """
        from goldfive.control import ControlMessage

        raw = getattr(drift, "raw", None)
        if not isinstance(raw, ControlMessage):
            return ""
        payload = raw.payload if isinstance(raw.payload, dict) else {}
        return str(payload.get("annotation_id", "") or "")

    # ------------------------------------------------------------------
    # Plan-revision cooldown + structural escalation primitives
    # ------------------------------------------------------------------

    # Drift kinds whose origin is a user intervention (USER_STEER /
    # USER_CANCEL) or a trajectory-level signal that has its own rate
    # limit (GOAL_DRIFT — task-boundary throttle via
    # ``_last_goal_drift_check_ts``). These kinds bypass the time-based
    # cooldown and the progress-stall escalation: user intent is always
    # honoured, and trajectory-wide drifts have no single task whose
    # progress could be measured.
    _USER_OR_TRAJECTORY_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
        {
            DriftKind.USER_STEER,
            DriftKind.USER_CANCEL,
            DriftKind.GOAL_DRIFT,
        }
    )

    def _is_task_progress_stalled(self, drift: DriftEvent, session: Session) -> bool:
        """Return ``True`` iff the drift's task has had no progress recently.

        goldfive#271 — replaces the deleted count-based cap with a
        progress-grounded structural guarantee. A productively-iterating
        task continually emits progress events
        (``mark_task_running`` / ``mark_task_progress`` /
        ``_emit_task_transitioned``); a stuck task does not. When a
        drift fires for a task whose ``Session.task_last_progress_at``
        is older than :attr:`PROGRESS_STALL_THRESHOLD_SECONDS`, we
        treat the drift as unresolvable by another refine and escalate
        to ``HUMAN_INTERVENTION_REQUIRED``.

        Returns ``False`` (no gate) when:

        * The threshold is non-positive (disabled).
        * The drift kind is a user / trajectory-level drift (always
          honoured / has its own rate limit).
        * The drift carries no ``current_task_id`` (trajectory-wide
          signals cannot be progress-stalled).
        * The task has no recorded progress yet (a freshly-running
          task may not have stamped ``task_last_progress_at`` if the
          drift fires before the first transition is processed).
        """
        threshold = self.PROGRESS_STALL_THRESHOLD_SECONDS
        if threshold <= 0:
            return False
        if drift.kind in self._USER_OR_TRAJECTORY_DRIFT_KINDS:
            return False
        task_id = drift.current_task_id
        if not task_id:
            return False
        last_at = session.task_last_progress_at.get(task_id)
        if last_at is None:
            # No progress signal yet — give the task the benefit of the
            # doubt. The first ``mark_task_running`` stamps the table,
            # so this branch only fires for the very first tick of a
            # fresh task or a task that never transitioned.
            return False
        age = time.monotonic() - last_at
        if age < threshold:
            return False
        log.warning(
            "task progress stalled (task=%r kind=%s age=%.1fs threshold=%.1fs); "
            "escalating to HUMAN_INTERVENTION_REQUIRED",
            task_id,
            drift.kind.value,
            age,
            threshold,
        )
        return True

    async def _emit_progress_stalled_escalation(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Emit a ``HUMAN_INTERVENTION_REQUIRED`` drift + pause the runner.

        Called from ``_handle_drift`` / ``_promote_drift_to_steer`` when
        :meth:`_is_task_progress_stalled` returns True. Phase 2 of the
        path-duality fix: dispatches a ``GOLDFIVE_PAUSE_ESCALATE``
        ControlMessage so the executor's pre-task loop blocks via the
        same channel state as a user ``PAUSE``. Emits a CRITICAL drift
        carrying the underlying (kind, task) so sinks / the UI can
        surface the stall.
        """
        task_id = drift.current_task_id
        last_at = session.task_last_progress_at.get(task_id) if task_id else None
        age = (time.monotonic() - last_at) if last_at is not None else 0.0
        reason = (
            f"task progress stalled for {drift.kind.value} on task "
            f"{task_id or '(trajectory)'}: "
            f"{age:.0f}s since last progress, threshold "
            f"{self.PROGRESS_STALL_THRESHOLD_SECONDS:.0f}s"
        )
        await self._dispatch_goldfive_pause_control(drift, session, reason=reason)
        escalation = DriftEvent(
            kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
            severity=DriftSeverity.CRITICAL,
            detail=reason,
            current_task_id=task_id,
            current_agent_id=drift.current_agent_id,
        )
        # Emit directly; do NOT recurse through ``_handle_drift``.
        await self._emit_drift_detected(session, escalation)

    async def _emit_handler_exhausted_escalation(
        self, drift: DriftEvent, session: Session
    ) -> None:
        """Emit a ``HUMAN_INTERVENTION_REQUIRED`` drift for handler exhaustion.

        goldfive#271 — drift-handler exhaustion as the escalation
        primitive. Called when a refine handler has tried and cannot
        produce a meaningful change for this drift (today: a
        structurally identical revision; future: explicit
        ``RefineExhausted`` sentinel from a planner). Phase 2 of the
        path-duality fix: dispatches a ``GOLDFIVE_PAUSE_ESCALATE``
        ControlMessage so the executor's pre-task loop blocks via the
        same channel state as a user ``PAUSE``. Emits a CRITICAL
        drift so the operator can decide whether to cancel or steer.
        """
        reason = (
            f"refine handler exhausted for {drift.kind.value} on task "
            f"{drift.current_task_id or '(trajectory)'}: "
            f"planner cannot produce a meaningful change"
        )
        await self._dispatch_goldfive_pause_control(drift, session, reason=reason)
        escalation = DriftEvent(
            kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
            severity=DriftSeverity.CRITICAL,
            detail=reason,
            current_task_id=drift.current_task_id,
            current_agent_id=drift.current_agent_id,
        )
        # Emit directly; do NOT recurse through ``_handle_drift``.
        await self._emit_drift_detected(session, escalation)

    @staticmethod
    def _plans_structurally_identical(prior: Plan | None, revised: Plan) -> bool:
        """Return ``True`` iff ``revised`` has the same structural shape as ``prior``.

        goldfive#271 — no-op revision rejection (subsumes #188 / closes
        the post-#305 loop pattern). Compares task ids, edges, assignees,
        and statuses. Differences in plan id, revision metadata
        (``revision_index`` / ``revision_kind`` / ``revision_severity`` /
        ``revision_reason`` / ``revision_trigger_event_id``), summaries,
        timing predictions, descriptions, and span bindings are
        ignored — these can change without the plan actually meaning
        anything different to the executor.

        Returns ``False`` when ``prior`` is ``None`` (the seed case in
        :meth:`Runner._install_revision`).
        """
        if prior is None:
            return False
        # Tasks: id + assignee + status (in declared order — order
        # matters because executor scheduling reads tasks in list order
        # for the topological tie-breaker).
        if len(prior.tasks) != len(revised.tasks):
            return False
        for old, new in zip(prior.tasks, revised.tasks, strict=True):
            if old.id != new.id:
                return False
            if old.assignee_agent_id != new.assignee_agent_id:
                return False
            if old.status != new.status:
                return False
        # Edges: order-independent, structural set comparison.
        old_edges = {(e.from_task_id, e.to_task_id) for e in prior.edges}
        new_edges = {(e.from_task_id, e.to_task_id) for e in revised.edges}
        if old_edges != new_edges:
            return False
        return True

    @staticmethod
    def _integrate_correction_supersedes(revised: Plan) -> Plan:
        """Rewire DAG edges for every ``CORRECT``-kind supersedes link.

        goldfive#251 Option B topology. For a new task
        ``new.supersedes == old_id`` with ``new.supersedes_kind ==
        SupersessionKind.CORRECT``:

        * The old task is NOT marked superseded / hidden — it stays in
          the plan as a historical COMPLETED node.
        * An edge ``old -> new`` is added (unless already present), so
          the new correction-task has the old as its upstream.
        * Every existing edge ``old -> X`` for some X != new is
          rewritten to ``new -> X`` so downstream work that used to
          depend on the old task now flows through the correction.

        The in-revision edges from the refiner sometimes already
        reflect this topology (the LLM may emit the rewired shape);
        this method is idempotent and re-runnable in that case.

        Does nothing when no task carries a CORRECT-kind supersedes.
        REPLACE-kind links are intentionally left alone — the pre-#251
        behaviour (old task marked terminal / hidden by the refiner;
        downstream edges rewritten to the replacement) was already
        correct and this method does not touch that path.

        goldfive#247: Plan.edges is an immutable tuple of frozen
        :class:`TaskEdge`. The rewrite builds a fresh edge list and
        returns a new :class:`Plan` via :func:`replace_edges`. When no
        rewrite is needed the original is returned unchanged so callers
        can keep their reference.

        Runs BEFORE :meth:`_repin_current_task_on_supersedes` so that
        helper's downstream rewrites see the already-correct DAG.
        """
        if revised is None:
            return revised
        tasks_by_id: dict[str, Task] = {t.id: t for t in revised.tasks if t.id}
        corrections: list[tuple[str, str]] = []  # (old_id, new_id)
        for task in revised.tasks:
            if task.supersedes_kind is not SupersessionKind.CORRECT:
                continue
            old_id = (task.supersedes or "").strip()
            new_id = (task.id or "").strip()
            if not old_id or not new_id or old_id == new_id:
                continue
            if old_id not in tasks_by_id:
                # Structural validator will reject; skip the rewrite.
                continue
            corrections.append((old_id, new_id))
        if not corrections:
            return revised
        # Build a fresh edge list as plain tuples; coerce to TaskEdges
        # at the end via :func:`replace_edges` (which also handles
        # dedup-while-preserving-order).
        edges: list[tuple[str, str]] = [(e.from_task_id, e.to_task_id) for e in revised.edges]
        existing_edges: set[tuple[str, str]] = set(edges)
        for old_id, new_id in corrections:
            # 1. Ensure old -> new edge exists.
            if (old_id, new_id) not in existing_edges:
                edges.append((old_id, new_id))
                existing_edges.add((old_id, new_id))
            # 2. Rewrite outgoing edges of the old task to originate
            #    from the new (correction) task. Skip the old -> new
            #    edge we just ensured.
            for i, edge in enumerate(edges):
                frm, to = edge
                if frm != old_id:
                    continue
                if to == new_id:
                    continue
                # Avoid duplicating an edge that already exists from the
                # new task to the same downstream.
                if (new_id, to) in existing_edges:
                    # Mark for dedup below; same content as existing.
                    edges[i] = (new_id, to)
                    continue
                existing_edges.discard((old_id, to))
                edges[i] = (new_id, to)
                existing_edges.add((new_id, to))
        # Final dedup: rewriting may have produced structurally-duplicate
        # edges. Preserve insertion order while dropping repeats.
        seen: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str]] = []
        for e in edges:
            if e in seen:
                continue
            seen.add(e)
            deduped.append(e)
        return replace_edges(revised, deduped)

    def _repin_current_task_on_supersedes(
        self,
        session: Session,
        revised: Plan,
    ) -> None:
        """Re-pin ``current_task_id`` onto replacement tasks after revision.

        When a revision's tasks carry a non-empty ``supersedes`` link
        (goldfive#237), treat it as the explicit "this task replaces
        that one" signal that older heuristic id-suffix matching was
        unable to express. Walk the map and:

        * Update ``session.current_task_id`` if it matches a superseded
          id — so agent-facing reporting-tool calls land on the live
          replacement rather than the FAILED/CANCELLED original.
        * Update the goldfive orchestration ``session.state`` pin
          (``goldfive.current_task_id`` key) when it matches a
          superseded id. This is the key the reporting-handler fallback
          (:func:`goldfive.reporting._resolve_task_id`) reads when the
          LLM's tool call omits the arg.
        * Ask the bound adapter (if any) to rewrite any per-agent ADK
          ``session.state`` copies whose current-task pin matches a
          superseded id. Best-effort: adapters without the hook no-op.

        The supersession map is built fresh from ``revised`` every call
        so A→B→C chains across multiple revisions compose naturally
        (each refine sees B.supersedes=A at revision N and
        C.supersedes=B at revision N+1; we never need to chase
        transitive links because the pin can only point at one id at a
        time and each revision fires this hook independently).
        """
        if revised is None:
            return
        # Build fresh per-revision. Old -> new. A planner producing
        # `C.supersedes = B` in the SAME revision that also ages
        # `B.supersedes = A` is handled transitively: we follow the
        # chain from the current pin forward to the first task that is
        # NOT itself superseded within the revision. In practice the
        # chain is rarely >1 hop per revision but the loop is cheap.
        supersession: dict[str, str] = {}
        for task in getattr(revised, "tasks", None) or ():
            old_id = str(getattr(task, "supersedes", "") or "").strip()
            new_id = str(getattr(task, "id", "") or "").strip()
            if not old_id or not new_id or old_id == new_id:
                continue
            supersession[old_id] = new_id
        if not supersession:
            return

        def _resolve_chain(start: str) -> str:
            """Walk the supersession map from ``start`` to its latest end."""
            seen: set[str] = {start}
            current = start
            while current in supersession:
                nxt = supersession[current]
                if nxt in seen:
                    # Defensive: a cycle shouldn't exist but guard
                    # against an adversarial planner before looping.
                    break
                seen.add(nxt)
                current = nxt
            return current

        # 1. goldfive Session pin.
        pinned = str(getattr(session, "current_task_id", "") or "")
        if pinned and pinned in supersession:
            resolved = _resolve_chain(pinned)
            if resolved != pinned:
                log.info(
                    "goldfive#237: re-pinning session.current_task_id %s -> %s (supersedes)",
                    pinned,
                    resolved,
                )
                session.current_task_id = resolved

        # 2. goldfive orchestration session.state pin (the reporting-
        # tool fallback's source of truth). Use the canonical state key
        # so tests that inspect the state dict directly see the update.
        # Phase 1 of goldfive#271 — read through OrchestrationStore;
        # the write stays at this call site (Phase 2 migration target
        # per the catalog).
        state = getattr(session, "state", None)
        if isinstance(state, dict):
            from goldfive.orchestration_store import OrchestrationStore

            store = OrchestrationStore.for_state(state)
            state_pinned_s = store.pin_current_task().strip()
            if state_pinned_s and state_pinned_s in supersession:
                resolved = _resolve_chain(state_pinned_s)
                if resolved != state_pinned_s:
                    log.info(
                        "goldfive#237: re-pinning session.state %s -> %s (supersedes)",
                        state_pinned_s,
                        resolved,
                    )
                    state[_ostate.KEY_CURRENT_TASK_ID] = resolved

        # 3. Per-agent ADK session.state copies (when the adapter
        # exposes a hook). Optional wiring: most test-path adapters
        # don't — we guard with hasattr and swallow exceptions so a
        # missing hook never breaks revision emission.
        adapter = self._adapter
        if adapter is None:
            return
        hook = getattr(adapter, "rewrite_pinned_task_ids", None)
        if not callable(hook):
            return
        try:
            hook(supersession)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "goldfive#237: adapter.rewrite_pinned_task_ids raised: %s",
                exc,
            )

    async def _emit_plan_revised(
        self,
        session: Session,
        revised: Plan,
        drift: DriftEvent,
        *,
        prev_plan: Plan | None = None,
        attempt_id: str | None = None,
    ) -> None:
        from goldfive._correction_injection import (
            clear_obsolete_corrections_on_revision,
            queue_corrections_for_revision,
        )
        from goldfive.conv import to_pb_plan
        from goldfive.events import build_plan_revision_diff

        # goldfive a4: serialise the consistency-critical region of plan
        # mutation. Held only across the in-memory mutations + the
        # PlanRevised emit — NOT across ``planner.refine`` itself, which
        # the caller owns. Reports calling :meth:`_wait_plan_stable`
        # observe either the pre- or post-revision state, never a
        # partial apply (e.g. supersedes integrated but pin not yet
        # repinned, or revision_index bumped but PlanRevised not yet
        # emitted). Fixes the race between fire-and-forget
        # judge-triggered refines (#254) and imperative report_task_*
        # handlers.
        lock = self._get_plan_lock(session)
        async with lock:
            # goldfive#251: integrate CORRECT-kind supersedes links into
            # the DAG. The old task stays in the plan as a historical
            # COMPLETED node; the new correction-task is inserted as a
            # child with an edge old -> new, and any downstream edges of
            # the old task are rewritten so work flows through the
            # correction. No-op for REPLACE-kind (existing behaviour is
            # preserved) and for plans without supersedes.
            # goldfive#247: returns a NEW Plan (Plan is frozen). The
            # pre-frozen code mutated ``revised`` in place; with frozen
            # types, we swap the new variant onto ``session.plan`` so
            # the live pointer matches the rewired DAG that's about to
            # be emitted as PlanRevised.
            integrated = self._integrate_correction_supersedes(revised)
            if integrated is not revised:
                revised = integrated
                with channel_processor_active():
                    set_session_plan(session, revised)

            # goldfive#251 Stream D: GC corrections for tasks superseded by
            # this revision BEFORE queuing new ones. A task whose correction
            # is about to be obsoleted (because the new revision supersedes
            # the correction task itself) must have its stale correction
            # dropped. Runs first so a same-revision CORRECT->CORRECT chain
            # (T -> T' -> T'') doesn't race: the T correction is cleared
            # here, then T''s correction is written below.
            clear_obsolete_corrections_on_revision(session, revised)

            # goldfive#251 Stream D: for every NEW task with supersedes_kind
            # == CORRECT, stamp a structured correction dict on the
            # orchestration session state under
            # ``goldfive.pending_corrections.<agent_name>.<task_id>``. The
            # dynamic instruction resolver (Stream B) reads this on the next
            # turn and appends a directive-style correction block to the
            # agent's system prompt. No-op on refines with no CORRECT links.
            queue_corrections_for_revision(
                session=session,
                revised=revised,
                prev_plan=prev_plan,
                drift=drift,
            )

            # goldfive#237: re-pin ``current_task_id`` onto any replacement
            # task the revision introduces. Without this, agents keep
            # reporting on the superseded (FAILED/CANCELLED) task and the
            # replacement stays PENDING despite active work — the contradiction
            # live sessions surfaced. Done before the event is emitted so
            # downstream observers see the revised pin consistently with the
            # revised plan. Additive: when no task has ``supersedes`` set,
            # nothing changes.
            self._repin_current_task_on_supersedes(session, revised)

            evt = self._new_envelope(session)
            evt.plan_revised.plan.CopyFrom(to_pb_plan(revised))
            evt.plan_revised.drift_kind = self._drift_kind_pb_value(drift.kind)
            evt.plan_revised.severity = self._drift_severity_pb_value(drift.severity)
            evt.plan_revised.reason = drift.detail
            evt.plan_revised.revision_index = revised.revision_index
            # goldfive#199: stamp ``trigger_event_id`` on the PlanRevised
            # envelope for EVERY refine — user-control (via source
            # annotation_id) and autonomous (via drift.id). Harmonograf's
            # intervention aggregator merges PlanRevised rows by strict id
            # only (legacy time-window fallback is behind a disabled env
            # flag). Priority: pre-stamped ``revision_trigger_event_id`` on
            # the revised plan (from ``_apply_revision`` or validator-retry
            # chain) → source annotation_id from the drift → drift.id.
            trig_id = revised.revision_trigger_event_id or (
                self._drift_annotation_id(drift) or str(getattr(drift, "id", "") or "")
            )
            if trig_id:
                evt.plan_revised.trigger_event_id = trig_id
            # Populate the minimal cross-revision diff so sinks that want a
            # "what changed" view don't have to re-fetch and diff the two
            # plans client-side. prev_plan may be None on the first revision
            # of a run that never received an initial plan — the helper
            # treats that as "everything in revised is newly added".
            evt.plan_revised.diff.CopyFrom(build_plan_revision_diff(prev_plan, revised))
            # Refine-context observability (judge-observability event). Sinks
            # rendering a Gantt / timeline want to explain WHY a refine was
            # requested and WHAT the planner produced without re-fetching
            # the drift and both plans.
            evt.plan_revised.refine_input_summary = self._build_refine_input_summary(
                drift, prev_plan
            )
            evt.plan_revised.refine_output_summary = self._build_refine_output_summary(revised)
            evt.plan_revised.target_agent_id = drift.current_agent_id or ""
            # Phase 2.X / goldfive#271 Gap 2: log the emission so a
            # raise-mid-fire scenario (proto build OK, sink emit raises)
            # leaves a goldfive-side trace before the harmonograf side
            # observes the gap. Pair with the warning on empty
            # plan_id / run_id below — those are the harmonograf#197
            # gate preconditions.
            plan_id_short = (revised.id or "")[:16] or "<empty>"
            run_id_short = (session.run_id or "")[:16] or "<empty>"
            if not session.run_id:
                log.warning(
                    "DefaultSteerer._emit_plan_revised: empty run_id for "
                    "plan_id=%s — harmonograf will drop both the audit "
                    "row AND the task_plans dispatch (harmonograf#197 "
                    "gate); this would silently lose the revision",
                    plan_id_short,
                )
            if not revised.id:
                log.warning(
                    "DefaultSteerer._emit_plan_revised: empty plan_id on "
                    "revised plan — harmonograf will drop the task_plans "
                    "row (no upsert key); this would silently lose the "
                    "revision",
                )
            log.info(
                "DefaultSteerer._emit_plan_revised: plan_id=%s "
                "revision_index=%d drift_kind=%s severity=%s run_id=%s",
                plan_id_short,
                int(revised.revision_index),
                drift.kind.value,
                drift.severity.value,
                run_id_short,
            )
            await self._emit(evt)
            # goldfive#251 R4 — every per-task status change carried by the
            # refine (e.g. ``_force_looper_failed`` stamping FAILED on the
            # looper, a CORRECT-supersedes integration cancelling the
            # superseded task, a REPLACE supersession marking the old task
            # CANCELLED) gets a paired ``TaskTransitioned`` sink event with
            # ``source="plan_revision"`` so operators see the refine-driven
            # transitions on the same observability lane as LLM-driven
            # ones. The transition events come AFTER ``PlanRevised`` so a
            # consumer that processes events strictly in order sees the
            # plan flip first, then the per-task status changes that flow
            # from it.
            await self._emit_plan_revision_transitions(session, prev_plan, revised)
            # goldfive a4: paired correlation envelope. The proto
            # ``PlanRevised`` carries no ``attempt_id`` field today; emit
            # a sidecar dict event so consumers can pair this success
            # with its preceding ``refine_attempted`` by attempt_id. The
            # proto event remains the primary surface; this is purely
            # correlation. ``attempt_id`` is ``None`` on legacy callers
            # that haven't been threaded through the new pipeline (e.g.
            # the executor's plan-swap detector) — those callers skip
            # the sidecar and behave exactly as before.
            if attempt_id:
                await self._emit_plan_revised_correlation(
                    session, revised, drift, attempt_id=attempt_id
                )

    @staticmethod
    def _build_refine_input_summary(
        drift: DriftEvent,
        prev_plan: Plan | None,
    ) -> str:
        """Render a short summary of what goldfive sent to ``planner.refine``.

        Intentionally terse — we pair the drift's ``kind`` / ``severity``
        / ``detail`` with a compact plan census (task count + status
        tallies) so a sink can answer "why was this refine requested,
        what did the planner see?" at a glance. Truncated via the same
        convention used by ``trigger_input`` to keep event sinks bounded.
        """
        parts: list[str] = []
        parts.append(f"drift={drift.kind.value}/{drift.severity.value}")
        if drift.current_task_id:
            parts.append(f"task={drift.current_task_id}")
        if drift.detail:
            parts.append(f"detail={drift.detail}")
        if prev_plan is not None:
            tasks = getattr(prev_plan, "tasks", None) or []
            parts.append(f"prior_plan=rev{prev_plan.revision_index}:{len(tasks)}tasks")
            if tasks:
                status_counts: dict[str, int] = {}
                for t in tasks:
                    status = getattr(t, "status", None)
                    key = str(getattr(status, "value", status) or "unspecified")
                    status_counts[key] = status_counts.get(key, 0) + 1
                tally = ",".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
                parts.append(f"prior_statuses={tally}")
        else:
            parts.append("prior_plan=none")
        text = " | ".join(parts)
        return DefaultSteerer._truncate_trigger_input(text)

    @staticmethod
    def _build_refine_output_summary(revised: Plan) -> str:
        """Render a short summary of the plan the planner returned."""
        tasks = getattr(revised, "tasks", None) or []
        parts: list[str] = [
            f"revision_index={revised.revision_index}",
            f"tasks={len(tasks)}",
        ]
        # Include the first few task titles so a Gantt can show the
        # revised plan's shape without fetching the full plan payload.
        titles = [str(getattr(t, "title", "") or "") for t in tasks[:6]]
        titles = [t for t in titles if t]
        if titles:
            parts.append("titles=[" + ", ".join(titles) + "]")
        text = " | ".join(parts)
        return DefaultSteerer._truncate_trigger_input(text)
