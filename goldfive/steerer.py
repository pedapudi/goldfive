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
import time  # noqa: F401 — re-exported for tests that monkeypatch ``goldfive.steerer.time.monotonic`` / ``.time``
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive import _state_audit
from goldfive import state_store as _ostate
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
        # goldfive#255: _apply_revision returns ``(revised, was_installed)``
        # so the caller can thread the install outcome into PlanRevised's
        # ``dry_run`` marker.
        revised, was_installed = self._apply_revision(session, revised, drift)
        # Cancel the in-flight coordinator invocation now that the plan
        # it was reasoning against has been superseded (goldfive#271
        # follow-up — v15 concurrent-invocation bug). Order: cancel
        # BEFORE PlanRevised emit so the synthetic InvocationCancelled
        # sink event lands adjacent to the revision in the wire log
        # and operators can correlate the two. Best-effort, never
        # raises — a no-op cancel still leaves the new plan installed.
        await self._cancel_inflight_for_revision(drift, session)
        await self._emit_plan_revised(
            session,
            revised,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
            dry_run=not was_installed,
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
        # goldfive#254 — observation-only: skip the actual ControlMessage
        # enqueue but log the would-be payload at INFO so operators can
        # see what would have been dispatched (drift kind, task id, body).
        # No cancel-and-restart fires on the executor; the live invocation
        # continues against the prior plan.
        if not self._should_inject():
            log.info(
                "DefaultSteerer._dispatch_goldfive_steer_control: "
                "observation_only=True — SKIPPING GOLDFIVE_STEER enqueue. "
                "would_have_dispatched kind=%s task=%s drift_id=%s "
                "superseded=%s replacement=%s body=%r",
                drift.kind.value,
                drift.current_task_id or "-",
                str(getattr(drift, "id", "") or ""),
                superseded_ids,
                replacement_ids,
                body[:200],
            )
            return False
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

        goldfive#264 — observation-only carve-out. Under
        ``SteeringConfig.observation_only`` the would-be
        ``GOLDFIVE_PAUSE_ESCALATE`` is SKIPPED: dispatching it on the
        channel sets ``goldfive_pause_message`` on the executor's
        :class:`~goldfive.executors._control.ControlOutcome`, which in
        turn drives ``_cancel_invoke_task`` and ends the overlay turn.
        That kills the live invocation — exactly the enforcement
        ``observation_only`` exists to suppress. The originating
        ``HUMAN_INTERVENTION_REQUIRED`` drift emitted by the caller
        (e.g. :meth:`_emit_handler_exhausted_escalation`,
        :meth:`_emit_progress_stall_escalation`,
        :meth:`_dispatch_pause_escalate`) is OUTSIDE this dispatch and
        continues to fire — observers/sinks still see the escalation,
        the operator can still react, but goldfive does NOT cancel the
        in-flight invocation. Mirrors the gate pattern at
        :meth:`_dispatch_goldfive_steer_control` (goldfive#254) and
        :meth:`request_invocation_cancel`.

        Live reproduction (2026-05-11, session
        ``4538863f-0dea-4fe8-97b4-5f660ee2cb7f``): an OFF_TOPIC drift
        under ``observation_only=True`` reached refine handler
        exhaustion (#271 no-op-revision path), which called this
        method, which dispatched the channel message, which cancelled
        the in-flight invoke. The carve-out below stops that chain.
        """
        from goldfive.control import ControlKind, ControlMessage

        if not self._should_inject():
            log.info(
                "DefaultSteerer._dispatch_goldfive_pause_control: "
                "observation_only=True — SKIPPING GOLDFIVE_PAUSE_ESCALATE "
                "dispatch. would_have_dispatched kind=%s task=%s "
                "drift_id=%s reason=%r",
                drift.kind.value,
                drift.current_task_id or "-",
                str(getattr(drift, "id", "") or ""),
                reason,
            )
            return False
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

        The check uses :class:`StateStore` as the live registry
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
            from goldfive.state_store import StateStore

            store = StateStore.for_session(session)
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

        Observation-only mode (goldfive#254): when
        :meth:`_should_inject` is ``False`` this method returns ``[]``
        without consulting the plugin or stamping
        ``cancel_requested_invocation_ids``. Logged at INFO so an
        operator can see WHAT would have been cancelled (drift kind,
        task / agent id) without the cancel actually firing on the
        live invocation.
        """
        if not self._should_inject():
            log.info(
                "DefaultSteerer.request_invocation_cancel: "
                "observation_only=True — SKIPPING cancel for "
                "drift kind=%s severity=%s agent=%s task=%s",
                drift.kind.value,
                drift.severity.value,
                drift.current_agent_id or "-",
                drift.current_task_id or "-",
            )
            return []
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
                from goldfive.state_store import StateStore

                store = StateStore.for_session(session)
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
        # Phase 1 of goldfive#271 — read through StateStore so
        # the active-steer slot reads from a single named accessor; the
        # underlying ``_ostate.read`` calls still funnel through the
        # goldfive Session.state dict, just behind a typed surface.
        window = self._goldfive_steer_suppression_window_turns
        if window > 0:
            from goldfive.state_store import StateStore

            active = StateStore.for_session(session).get_active_steer()
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
        # goldfive#255: thread ``was_installed`` into PlanRevised.dry_run.
        revised, was_installed = self._apply_revision(session, revised, drift)
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
            session,
            revised,
            drift,
            prev_plan=prev_plan,
            attempt_id=attempt_id,
            dry_run=not was_installed,
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
