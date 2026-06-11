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
import contextvars
import enum
import inspect
import logging
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
    # AGENCY-PRESERVATION.md PR 7: NUDGE renamed to SIGNAL. The level now
    # enqueues an advisory observer note (the request_context channel / the
    # legacy nudge replay) WITHOUT a refine/cancel/steer — the proportional,
    # trajectory-preserving response that replaces the goldfive-authored
    # CANCEL_REINVOKE cells. The enum value (2) is unchanged.
    SIGNAL = 2
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


def compose_corrective_user_message(
    *,
    drift: DriftEvent,
    refined_plan: Plan | None,
) -> str:
    """DEPRECATED shim — delegates to :mod:`goldfive.observer_notes`.

    AGENCY-PRESERVATION.md PR 4 retired the ``_CORRECTIVE_TEMPLATES``
    command templates this function used to render ("proceed to
    '{next_task_title}' via {next_task_agent}", "do NOT retry", …):
    goldfive owns goals / budgets / observability, the wrapped agent
    owns MEANS (decomposition, delegation, ordering, retries), so the
    agent-facing message is now an observation+goal advisory note, not
    a directive about which task or agent comes next.

    Kept as a thin delegating shim because the name is in the public
    ``__all__`` and external callers / tests import it. In-tree call
    sites (``goldfive.drift_observer``) call
    :func:`goldfive.observer_notes.compose_note_for_drift` directly —
    it accepts the ``session`` so the note carries the user's goals;
    this shim cannot (signature compatibility), so notes it renders
    carry the "(no goals recorded for this run)" placeholder. New code
    should not call this.
    """
    from goldfive.observer_notes import compose_note_for_drift

    return compose_note_for_drift(drift=drift, plan=refined_plan)


def _enum_or_str_value(value: Any) -> str:
    """Project a ``DriftKind``/``DriftSeverity``-or-string onto its string value.

    :class:`~goldfive.judges.JudgeVerdict` accepts either the typed
    :class:`DriftKind` / :class:`DriftSeverity` enum (preferred) or a
    legacy lowercase string for its ``drift_kind`` / ``severity``
    fields. Proto envelopes carry plain strings, so the emit path needs
    one place that flattens either shape: an enum member yields its
    ``.value``, anything else is ``str()``-coerced. ``None`` / empty
    yields the empty string.
    """
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value or "")


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
        cancel_inflight_scope: str | None = None,
        judges: list[Any] | None = None,
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
            Number of agent-activity entries retained in the
            ``recent_events`` buffer (goldfive#239) and passed to the
            judge. Bounds the prompt size; defaults to ``10`` when
            ``goal_drift_config`` is also ``None``. The ``tool_observed``
            subset of the same buffer is bounded independently by
            ``session.recent_tool_observations_max`` so a flood of one
            kind cannot evict the other.
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
        # AGENCY-PRESERVATION.md PR 1 (goldfive#449/#452): authority
        # scope for cancelling the wrapped agent's in-flight invocation
        # on a drift-driven plan install. Precedence mirrors the other
        # goldfive#225 knobs: explicit kwarg > ``SteeringConfig`` >
        # built-in default (``"user_and_safety"``). Consumed by
        # :meth:`DriftObserver._drift_authorizes_inflight_cancel`;
        # ``"all"`` is the §5.1 kill-switch restoring the legacy
        # cancel-on-every-install behaviour (env:
        # ``GOLDFIVE_CANCEL_INFLIGHT_SCOPE=all`` via
        # :meth:`SteeringConfig.from_env`).
        if cancel_inflight_scope is not None:
            _cancel_scope = str(cancel_inflight_scope).strip().lower()
        elif steering_config is not None:
            _cancel_scope = str(steering_config.cancel_inflight_scope).strip().lower()
        else:
            _cancel_scope = "user_and_safety"
        if _cancel_scope not in {"user_and_safety", "all"}:
            log.warning(
                "DefaultSteerer: unknown cancel_inflight_scope=%r; "
                "falling back to 'user_and_safety'",
                _cancel_scope,
            )
            _cancel_scope = "user_and_safety"
        self._cancel_inflight_scope: str = _cancel_scope
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
        # AGENCY-PRESERVATION.md PR 5 (#449/#452): signal telemetry —
        # SignalDelivered / SignalOutcome events + the SignalLedger
        # bookkeeping. Default OFF so PR 5 is a true no-op (§5.1
        # "no-op by default"): with the flag off the drift-observer's signal
        # helpers early-return before touching the ledger or the wire, so the
        # event stream every existing suite asserts on is byte-for-byte
        # unchanged. The shadow/differential-validation campaign (§5.4) and
        # PR 8's grace-window pacing enable it explicitly via
        # ``SteeringConfig(signal_telemetry=True)``. Gates nothing either way.
        self._signal_telemetry_enabled: bool = bool(
            getattr(steering_config, "signal_telemetry", False)
        )
        # AGENCY-PRESERVATION.md PR 6: observer-note delivery channel.
        # ``"legacy_user_message"`` (the default) keeps corrective notes on
        # ``session.pending_nudges`` (the pre-PR-6 boundary replay);
        # ``"request_context"`` routes them through the
        # :class:`~goldfive.observer_note_queue.ObserverNoteQueue` and the four
        # observer-note delivery surfaces. The drift observer (enqueue) and the
        # executor / adapters (delivery) read this flag; with the legacy
        # default the queue is never populated and the new surfaces are inert,
        # so PR 6 ships dark (§5.1). Normalised + validated on
        # :class:`SteeringConfig`; a bare ``DefaultSteerer()`` with no config
        # defaults to legacy.
        _channel = str(
            getattr(steering_config, "signal_channel", "legacy_user_message")
            or "legacy_user_message"
        ).strip().lower()
        if _channel not in {"legacy_user_message", "request_context"}:
            log.warning(
                "DefaultSteerer: unknown signal_channel=%r; "
                "falling back to 'legacy_user_message'",
                _channel,
            )
            _channel = "legacy_user_message"
        self._signal_channel: str = _channel
        # AGENCY-PRESERVATION.md PR 7: the one-release legacy-ladder escape
        # hatch. When True the pre-PR-7 ladder cells (CANCEL_REINVOKE in the
        # goldfive-authored rows) and the full _promote_drift_to_steer
        # side-effects are restored; the drift observer reads this flag in
        # ``_ladder_level_for`` and ``_promote_drift_to_steer``. Default OFF
        # (the new SIGNAL ladder). The two deferred correctness fixes
        # (hard-safety stop, PLAN_DIVERGENCE removal) are NOT gated by it.
        self._legacy_ladder: bool = bool(
            getattr(steering_config, "legacy_ladder", False)
        )
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
        # Pluggable-judges surface (goldfive#437). Operators register a
        # list of :class:`~goldfive.judges.Judge` instances via
        # :func:`goldfive.wrap(judges=[...])`; the runtime calls
        # :meth:`evaluate_judges` at observation points and emits
        # ``JudgementEmitted`` for every populated verdict (plus the
        # legacy ``DriftDetected`` envelope when a judge returns a
        # drift-flavoured verdict). ``None`` from the constructor
        # defers default-set installation until :func:`goldfive.wrap`
        # decides — see :meth:`set_judges`. Empty list is an explicit
        # operator opt-out (no judges run).
        self._judges: list[Any] = list(judges) if judges is not None else []
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
        #
        # Components are exposed as public properties (``steerer.tasks``,
        # ``steerer.plans``, ``steerer.drift``) so callers — executors,
        # the runner, reporting handlers, planners, tests — can address
        # the right component directly rather than going through a
        # router shim. See goldfive#410.
        from goldfive.drift_observer import DriftObserver
        from goldfive.plan_reviser import PlanReviser
        from goldfive.task_state_machine import TaskStateMachine

        self.tasks: TaskStateMachine = TaskStateMachine(self)
        self.plans: PlanReviser = PlanReviser(self)
        self.drift: DriftObserver = DriftObserver(self)

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
        # Wire the per-attempt retry-budget telemetry callback
        # (manifest-and-decision-telemetry). Each refine attempt emits
        # a ``RetryBudgetSpent`` row through the normal sink pipeline
        # so downstream optimizers can correlate planner attempt counts
        # against drift outcomes. Duck-typed on purpose.
        budget_setter = getattr(planner, "set_retry_budget_emitter", None)
        if callable(budget_setter):
            budget_setter(self._emit_planner_retry_budget)
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

    def set_judges(self, judges: list[Any]) -> None:
        """Install the pluggable-judges list (goldfive#437).

        Operator entry point — :func:`goldfive.wrap` forwards its
        ``judges=`` kwarg here so the steerer's built-in detector
        emit path can additionally publish :class:`JudgementEmitted`
        for every populated verdict. The legacy
        :class:`DriftDetected` envelope is unchanged: drift-flavoured
        verdicts still fire both events for back-compat.

        Replaces any previously-installed judge list; pass an empty
        list to disable the surface entirely (no judges run, only
        the legacy hardcoded detector path remains active).
        """
        self._judges = list(judges)

    def get_judges(self) -> list[Any]:
        """Return the currently-installed judge list (goldfive#437)."""
        return list(self._judges)

    #: Wall-clock budget, in seconds, for a single ``Judge.evaluate``
    #: call inside :meth:`evaluate_judges`. A custom judge that hangs
    #: (network rubric grader with no client-side timeout, a deadlocked
    #: lock) MUST NOT stall the run — the auto-wired observation path
    #: dispatches :meth:`evaluate_judges` from the model-response
    #: critical path. A judge that overruns the budget is cancelled,
    #: logged at WARNING, and treated as "no signal" (goldfive#437).
    JUDGE_EVALUATE_TIMEOUT_S: float = 30.0

    async def evaluate_judges(
        self,
        ctx: Any,
        *,
        session: Session | None = None,
        run_id: str = "",
        judges: list[Any] | None = None,
    ) -> list[Any]:
        """Evaluate every installed judge against ``ctx`` and emit results.

        For every judge in :attr:`_judges`:

        * call ``judge.evaluate(ctx)`` and await the
          :class:`~goldfive.judges.JudgeVerdict`, bounded by
          :attr:`JUDGE_EVALUATE_TIMEOUT_S` — a judge that overruns
          the budget is cancelled and treated as "no signal";
        * pick ``verdict_kind`` from the first populated flavour
          (drift / rubric / boolean / numeric);
        * skip emission entirely when no flavour is populated (judges
          that have nothing to say stay silent on the wire);
        * emit a :class:`JudgementEmitted` envelope onto the bound
          sinks for every populated verdict;
        * forward drift-flavoured verdicts back to the legacy
          :meth:`drift.handle_drift` path so ``DriftDetected`` still
          fires and the refine machinery still runs (back-compat
          contract).

        Errors raised by a judge — and timeouts — are caught and
        logged at WARNING; a misbehaving judge MUST NOT break the run
        or suppress other judges' verdicts. Returns the list of
        verdicts collected so callers can inspect them in tests.

        ``session`` is forwarded onto the emitted event envelope (for
        ``session_id``) and used as the back-channel for drift-
        flavoured verdicts that need to route through
        :meth:`drift.handle_drift`. ``run_id`` is the active run's
        identifier — falls back to ``session.run_id`` when omitted.
        ``judges`` overrides the installed list for this call only —
        the auto-wired observation path passes the operator-supplied
        custom judges (built-ins excluded; their drift verdicts ride
        the legacy detector path's paired emission instead).
        """
        verdicts: list[Any] = []
        active = list(self._judges) if judges is None else list(judges)
        for judge in active:
            judge_name = str(getattr(judge, "name", "") or type(judge).__name__)
            try:
                verdict = await asyncio.wait_for(
                    judge.evaluate(ctx), timeout=self.JUDGE_EVALUATE_TIMEOUT_S
                )
            except TimeoutError:
                log.warning(
                    "DefaultSteerer.evaluate_judges: judge %r exceeded the "
                    "%.1fs evaluate budget; cancelled and treated as no signal",
                    judge_name,
                    self.JUDGE_EVALUATE_TIMEOUT_S,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — judges must not crash the run
                log.warning(
                    "DefaultSteerer.evaluate_judges: judge %r raised %s (%s); "
                    "swallowed",
                    judge_name,
                    type(exc).__name__,
                    exc,
                )
                continue
            if verdict is None:
                continue
            verdicts.append(verdict)
            await self._emit_judgement(
                verdict, judge_name=judge_name, session=session, run_id=run_id
            )
            # Drift-flavoured verdicts ALSO fire the legacy
            # ``DriftDetected`` envelope via the existing handle_drift
            # path so pre-judges consumers see no behavioural change.
            if getattr(verdict, "drift_emitted", False) and session is not None:
                drift = self._drift_from_judge_verdict(verdict, judge_name=judge_name)
                if drift is not None:
                    # Mark the drift so the ``_emit_drift_detected``
                    # paired-emission path does NOT emit a second
                    # ``JudgementEmitted`` — ``_emit_judgement`` above
                    # already published one keyed on the judge's real
                    # ``name``. A non-wire runtime attribute (the proto
                    # ``DriftDetected`` envelope is unaffected).
                    drift._judge_emitted_judgement = True  # type: ignore[attr-defined]
                    try:
                        await self.drift.handle_drift(drift, session)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "DefaultSteerer.evaluate_judges: handle_drift raised "
                            "%s (%s) for judge %r; swallowed",
                            type(exc).__name__,
                            exc,
                            judge_name,
                        )
        return verdicts

    def _drift_from_judge_verdict(
        self, verdict: Any, *, judge_name: str
    ) -> DriftEvent | None:
        """Project a drift-flavoured verdict back onto a :class:`DriftEvent`.

        Used by :meth:`evaluate_judges` so drift-flavoured verdicts
        still flow through the legacy refine machinery. Returns
        ``None`` when the verdict does not carry a recognisable
        :class:`DriftKind` so a malformed judge can't crash the
        handler — the :class:`JudgementEmitted` envelope is still
        emitted in that case.

        ``JudgeVerdict.drift_kind`` / ``severity`` may be a typed
        :class:`DriftKind` / :class:`DriftSeverity` enum (the preferred
        form) or a legacy lowercase string; ``DriftKind(...)`` /
        ``DriftSeverity(...)`` accept both an enum member and a string,
        so this handles either shape.
        """
        try:
            kind = DriftKind(verdict.drift_kind)
        except (ValueError, AttributeError):
            log.debug(
                "DefaultSteerer._drift_from_judge_verdict: judge %r returned "
                "unrecognised drift_kind=%r; emitting JudgementEmitted only",
                judge_name,
                getattr(verdict, "drift_kind", ""),
            )
            return None
        try:
            severity = DriftSeverity(verdict.severity)
        except (ValueError, AttributeError):
            severity = DriftSeverity.INFO
        return DriftEvent(
            kind=kind,
            severity=severity,
            detail=str(getattr(verdict, "detail", "") or ""),
            # AGENCY-PRESERVATION.md PR 4 — thread the judge-authored
            # agent-facing observation onto the drift so
            # ``goldfive.observer_notes`` can prefer it over ``detail``
            # when composing notes. ``getattr`` keeps pre-PR-4 verdict
            # shapes (no field) degrading to the empty string.
            note_to_agent=str(getattr(verdict, "note_to_agent", "") or ""),
        )

    async def _emit_judgement(
        self,
        verdict: Any,
        *,
        judge_name: str,
        session: Session | None,
        run_id: str,
    ) -> None:
        """Emit a :class:`JudgementEmitted` envelope for ``verdict``.

        Picks ``verdict_kind`` from the first populated flavour:
        drift, then rubric, then boolean, then numeric. An empty-
        default verdict produces no event (the judge had nothing to
        say). Errors raised by the sink are absorbed at WARNING so
        a broken sink can't crash the run.
        """
        verdict_kind = ""
        if getattr(verdict, "drift_emitted", False):
            verdict_kind = "drift"
        elif getattr(verdict, "rubric_score", None) is not None or getattr(
            verdict, "rubric_dimensions", None
        ):
            verdict_kind = "rubric"
        elif getattr(verdict, "boolean_result", None) is not None:
            verdict_kind = "boolean"
        elif getattr(verdict, "numeric_value", None) is not None or getattr(
            verdict, "metric_name", ""
        ):
            verdict_kind = "numeric"
        if not verdict_kind:
            # Empty-default verdict — judge had nothing to say.
            return
        if not self._sinks:
            return
        try:
            from goldfive.events import emit, new_event
            from goldfive.pb.goldfive.v1 import events_pb2 as _pb
        except Exception as exc:  # noqa: BLE001 — pb stubs missing
            log.debug(
                "DefaultSteerer._emit_judgement: pb stubs unavailable (%s); "
                "judge %r verdict not emitted",
                exc,
                judge_name,
            )
            return
        active_session = session if session is not None else self._active_session_var.get()
        sess_id = str(getattr(active_session, "id", "") or "")
        resolved_run_id = run_id or str(getattr(active_session, "run_id", "") or "")
        try:
            if active_session is not None:
                seq, event_id = active_session.next_sequence_and_event_id()
            else:
                seq, event_id = 0, ""
        except Exception:  # noqa: BLE001 — older Session shapes lack the helpers
            seq, event_id = 0, ""
        evt = new_event(resolved_run_id, seq, sess_id, event_id=event_id)
        payload = _pb.JudgementEmitted()
        payload.judge_name = judge_name
        payload.verdict_kind = verdict_kind
        # ``drift_kind`` / ``severity`` on a :class:`JudgeVerdict` may be
        # a typed :class:`DriftKind` / :class:`DriftSeverity` enum or a
        # legacy lowercase string. The proto field is a plain string;
        # ``_enum_or_str_value`` projects either shape onto its string
        # value so the wire form is identical regardless.
        payload.drift_kind = _enum_or_str_value(getattr(verdict, "drift_kind", ""))
        payload.severity = _enum_or_str_value(getattr(verdict, "severity", ""))
        rubric_score = getattr(verdict, "rubric_score", None)
        if rubric_score is not None:
            payload.rubric_score = float(rubric_score)
        rubric_dimensions = getattr(verdict, "rubric_dimensions", None) or {}
        for dim_name, dim_score in rubric_dimensions.items():
            payload.rubric_dimensions[str(dim_name)] = float(dim_score)
        boolean_result = getattr(verdict, "boolean_result", None)
        if boolean_result is not None:
            payload.boolean_result = bool(boolean_result)
        numeric_value = getattr(verdict, "numeric_value", None)
        if numeric_value is not None:
            payload.numeric_value = float(numeric_value)
        payload.metric_name = str(getattr(verdict, "metric_name", "") or "")
        payload.detail = str(getattr(verdict, "detail", "") or "")
        evt.judgement_emitted.CopyFrom(payload)
        try:
            await emit(list(self._sinks), evt)
        except Exception as exc:  # noqa: BLE001 — broken sink must not crash run
            log.warning(
                "DefaultSteerer._emit_judgement: emit raised %s (%s) for judge %r; "
                "swallowed",
                type(exc).__name__,
                exc,
                judge_name,
            )

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
        await self.drift._emit_drift_detected(session, drift)

    async def _emit_planner_retry_budget(
        self,
        operation: str,
        attempt: int,
        budget_remaining: int,
        reason: str,
    ) -> None:
        """Forward a per-attempt retry-budget telemetry row to sinks.

        Wired in :meth:`bind` via
        :meth:`~goldfive.planner.LLMPlanner.set_retry_budget_emitter`.
        Routes through the same ``_active_session_var`` ContextVar the
        drift emitter uses so concurrent runs sharing one steerer
        instance still attribute rows to the right session. Best-effort:
        emission failure must never break the refine retry loop.
        """
        session = self._active_session_var.get()
        if session is None:
            # No active session — the planner is being exercised
            # standalone (tests). Drop the row; no caller depends on
            # it.
            return
        try:
            from goldfive.events import retry_budget_spent_event
        except ModuleNotFoundError:  # pragma: no cover -- proto-less env
            return
        try:
            seq, event_id = session.next_sequence_and_event_id()
            evt = retry_budget_spent_event(
                session.run_id,
                seq,
                operation=operation,
                attempt=attempt,
                budget_remaining=budget_remaining,
                reason=reason,
                session_id=session.id,
                event_id=event_id,
            )
            await self._emit(evt)
        except Exception as exc:  # noqa: BLE001 -- telemetry best-effort
            log.debug("retry_budget_spent emit failed: %s", exc)

    # ------------------------------------------------------------------
    # Protocol-required: transition (generic)
    # ------------------------------------------------------------------

    async def finalize_outcomes(self, session: Session) -> None:
        """Judge + finalize ledger OUTCOME deliverables at the run boundary.

        AGENCY-PRESERVATION.md Stage 3 PR 11(c). Delegates to
        :meth:`DriftObserver.finalize_outcomes` (the judge + transition
        logic lives drift-side); the executor calls this single method at
        the overlay run boundary, mirroring the
        ``_drain_steerer_at_run_boundary`` contract. Ledger-gated and
        quiet on failure — a no-op in forecast mode.
        """
        await self.drift.finalize_outcomes(session)

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
            await self.tasks.mark_task_running(
                task_id, session=session, detail=detail, source=source
            )
        elif to is TaskStatus.COMPLETED:
            await self.tasks.mark_task_completed(
                task_id, summary=detail, session=session, source=source
            )
        elif to is TaskStatus.FAILED:
            reason = cancel_reason or detail
            await self.tasks.mark_task_failed(
                task_id, reason=reason, session=session, source=source
            )
        elif to is TaskStatus.BLOCKED:
            await self.tasks.mark_task_blocked(
                task_id, blocker=detail, session=session, source=source
            )
        elif to is TaskStatus.CANCELLED:
            reason = cancel_reason or detail
            await self.tasks.mark_task_cancelled(
                task_id, reason=reason, session=session, source=source
            )
        elif to is TaskStatus.NOT_NEEDED:
            await self.tasks.mark_task_not_needed(
                task_id, reason=detail, session=session, source=source
            )
        # PENDING and UNSPECIFIED are intentionally not reachable from
        # here; transitions are always forward in the lifecycle.

    # ==================================================================
    # Internals (router-level, not delegated to a component)
    # ==================================================================

    # --- Plan lookup --------------------------------------------------

    @staticmethod
    def _find_task(session: Session, task_id: str) -> Task | None:
        # Identity helper. Mirrors
        # :meth:`goldfive.task_state_machine.TaskStateMachine._find_task`
        # and remains here as a router-level staticmethod for callers
        # (the reflective-check path on :class:`DriftObserver`, tests)
        # that look up tasks without holding a component reference.
        if not task_id or session.plan is None:
            return None
        for t in session.plan.tasks:
            if t.id == task_id:
                return t
        return None

    # ------------------------------------------------------------------
    # ``observation_only`` gate — stays on the router because it reads
    # the router-owned :attr:`_observation_only` flag (set in
    # :meth:`__init__` from :class:`~goldfive.config.SteeringConfig`).
    # ------------------------------------------------------------------

    def _should_inject(self) -> bool:
        """Return ``True`` iff the steerer should actually inject side effects.

        Single named gate for the three steering injection points
        (goldfive#254):

        * plan mutation in :meth:`PlanReviser._apply_revision`
          (``set_session_plan`` + ``last_addressed_revision_by_drift_key``);
        * ``GOLDFIVE_STEER`` ControlMessage enqueue in
          :meth:`DriftObserver._dispatch_goldfive_steer_control`;
        * the plugin ``request_invocation_cancel`` flag in
          :meth:`DriftObserver.request_invocation_cancel`.

        ``False`` when :class:`~goldfive.config.SteeringConfig.observation_only`
        is in effect — detection still runs, ``planner.refine_steer``
        still runs, ``PlanRevised`` still emits (with ``dry_run=True``),
        but the in-flight invocation is not touched. Defined as a tiny
        helper rather than inlining ``not self._observation_only`` at
        three sites so the intent is grep-able and a future fourth
        injection point has a single gate to honour.
        """
        return not self._observation_only

    # Consecutive refine failures tolerated per (drift_kind, task_id)
    # before we give up and mark the task FAILED. Class attribute so
    # subclasses / tests can tune it without poking at instance state.
    # Canonical definition; :class:`DriftObserver` and :class:`PlanReviser`
    # both read it as ``self._steerer.REFINE_FAILURE_THRESHOLD``.
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
    # Canonical definition; :class:`DriftObserver` reads it as
    # ``self._steerer.PROGRESS_STALL_THRESHOLD_SECONDS``.
    PROGRESS_STALL_THRESHOLD_SECONDS: float = 600.0

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

