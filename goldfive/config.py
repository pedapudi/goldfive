"""Typed, per-Runner configuration for goldfive (goldfive#225).

goldfive has accumulated ad-hoc configuration knobs across four
subsystems, each with a different mechanism:

1. **Embedding backend** (#221). ``GOLDFIVE_EMBEDDING_BASE_URL`` /
   ``_MODEL`` / ``_API_KEY`` / ``_TIMEOUT_MS`` read at first call in
   :mod:`goldfive.drift._embed`.
2. **Tool-loop detector**. ``GOLDFIVE_TOOL_LOOP_WINDOW`` /
   ``_EXACT_THRESHOLD`` / ``_NAME_THRESHOLD`` / ``_ALTERNATING_THRESHOLD``
   read in :mod:`goldfive.drift.tool_loops`.
3. **Reasoning-drift thresholds**. Module-level constants in
   :mod:`goldfive.drift.reasoning` with no env wiring at all.
4. **Goal-drift scheduling**. Kwargs on
   :class:`~goldfive.steerer.DefaultSteerer` that ``goldfive.wrap()``
   does not thread through.

Steering policy (goldfive#254). :class:`SteeringConfig` now also
gates the THREE actual steering injection points in
:class:`~goldfive.steerer.DefaultSteerer`:

* the would-be revised plan replacing ``session.plan`` in
  :meth:`~goldfive.steerer.DefaultSteerer._apply_revision`;
* the ``GOLDFIVE_STEER`` ControlMessage being enqueued onto the
  executor's control channel in
  :meth:`~goldfive.steerer.DefaultSteerer._dispatch_goldfive_steer_control`;
* the ``request_invocation_cancel`` plugin call in
  :meth:`~goldfive.steerer.DefaultSteerer.request_invocation_cancel`.

The ``observation_only`` field on :class:`SteeringConfig` controls
whether those three injections fire. Default is ``True`` (passive
observation) — detection still runs in full, ``planner.refine_steer``
still runs (operators can see what the planner WOULD have produced via
``PlanRevised`` with ``dry_run=True``), but the in-flight invocation is
not touched. Operators graduate to active steering explicitly via
``RuntimeConfig(steering=SteeringConfig(observation_only=False))``.

This module introduces a typed :class:`RuntimeConfig` dataclass with
four sub-configs that collapses those four surfaces into a single
object operators can pass to :func:`goldfive.wrap`. The ``from_env()``
classmethods preserve the existing env-var surface (names unchanged
where they already exist; new ``GOLDFIVE_DRIFT_*`` / ``GOLDFIVE_GOAL_DRIFT_*``
names for the knobs that did not previously have env wiring) so
``goldfive.wrap(tree)`` with no ``runtime=`` kwarg remains byte-
identical to pre-#225 behaviour.

Dataclasses are deliberately **mutable** (``frozen=False``). Operators
commonly tweak a field after constructing from env (e.g. load
defaults then bump ``goal_drift.check_interval`` for a debugging run).
Callers who want a snapshot can :func:`dataclasses.replace` to build a
variant without mutating the source.

See :func:`goldfive.wrap` for the installation path and
``docs/design/DRIFT.md`` §"Per-Runner runtime config" for the design
note.
"""

from __future__ import annotations

import dataclasses
import logging
import os

from goldfive.drift.reasoning import (
    DEFAULT_REASONING_DRIFT_MODE,
    ReasoningDriftMode,
)

__all__ = [
    "AgentConfig",
    "EmbeddingConfig",
    "GoalDriftConfig",
    "JudgeConfig",
    "ReasoningDriftConfig",
    "RuntimeConfig",
    "SteeringConfig",
    "ToolLoopConfig",
]


_VALID_STEER_THRESHOLDS: frozenset[str] = frozenset({"off", "warning", "critical"})


# Test-only override hook for :class:`SteeringConfig.observation_only`'s
# default (goldfive#254). Production code path: this stays ``None`` and
# every fresh ``SteeringConfig()`` instance gets the documented production
# default of ``True`` (passive observation). The pytest autouse fixture
# ``tests/conftest.py::_goldfive_active_steering_default`` flips this to
# ``False`` for the test suite so the broad existing test corpus —
# written against the prior active-steering default — stays green
# without per-test surgery. Tests that explicitly pass
# ``observation_only=True`` (or ``=False``) still win — the override
# only applies when the field was not explicitly set by the caller.
_OBSERVATION_ONLY_DEFAULT: bool | None = None


def _resolve_observation_only_default() -> bool:
    """Resolve the active default for :class:`SteeringConfig.observation_only`.

    Reads :data:`_OBSERVATION_ONLY_DEFAULT`. ``None`` means "no test
    override is in effect" — return the production default (``True``).
    Anything else means a test fixture has flipped the default
    explicitly; honour the override.
    """
    if _OBSERVATION_ONLY_DEFAULT is None:
        return True
    return bool(_OBSERVATION_ONLY_DEFAULT)


def _read_bool_env(name: str, default: bool) -> bool:
    """Read ``os.environ[name]`` as a boolean; fall back to ``default``.

    Accepted truthy values (case-insensitive): ``1``, ``true``, ``yes``,
    ``on``, ``y``, ``t``. Accepted falsy values: ``0``, ``false``,
    ``no``, ``off``, ``n``, ``f``, ``""``. Anything else logs a WARNING
    and falls back to ``default`` so a typo never silently flips an
    operator-visible policy knob.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if value in {"0", "false", "no", "off", "n", "f", ""}:
        return False
    log.warning(
        "ignoring unknown %s=%r (expected a boolean literal); using default %r",
        name,
        raw,
        default,
    )
    return default


def _read_steer_threshold_env(name: str, default: str) -> str:
    """Return ``os.environ[name]`` as a steer threshold literal, or ``default``.

    Accepts ``"off"`` / ``"warning"`` / ``"critical"`` (case-insensitive).
    Anything else logs a WARNING and falls back so a typo doesn't
    silently disable the promotion policy.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _VALID_STEER_THRESHOLDS:
        return value
    log.warning(
        "ignoring unknown %s=%r (expected one of %s); using default %r",
        name,
        raw,
        sorted(_VALID_STEER_THRESHOLDS),
        default,
    )
    return default


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _read_int_env(name: str, default: int) -> int:
    """Best-effort positive-integer env override; returns default on any failure.

    Matches the semantics of
    :func:`goldfive.drift.tool_loops._read_int_env` (and, indirectly,
    :func:`goldfive.drift._embed._try_load_openai_backend`'s timeout
    read). Non-integer / non-positive / missing values silently fall
    back to the default.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        log.debug(
            "runtime-config: ignoring non-integer %s=%r (using default %d)",
            name,
            raw,
            default,
        )
        return default
    if val <= 0:
        log.debug(
            "runtime-config: ignoring non-positive %s=%d (using default %d)",
            name,
            val,
            default,
        )
        return default
    return val


def _read_float_env(name: str, default: float) -> float:
    """Best-effort float env override; returns default on any failure.

    Unlike :func:`_read_int_env` we do **not** require the value to be
    positive — a reasoning-drift threshold of ``0.0`` is a valid (if
    degenerate) configuration. Parse failures fall back to the default
    with a debug log.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.debug(
            "runtime-config: ignoring non-float %s=%r (using default %s)",
            name,
            raw,
            default,
        )
        return default


def _read_optional_float_env(name: str, default: float | None) -> float | None:
    """Best-effort optional-float env override; returns default on any failure.

    Used for fields whose Python type is ``float | None`` where ``None``
    means "feature disabled" (e.g. ``pause_escalate_deadline_s``). A
    non-positive value is treated as an explicit "disable" and maps to
    ``None``; parse failures fall back to the default with a debug log.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.debug(
            "runtime-config: ignoring non-float %s=%r (using default %s)",
            name,
            raw,
            default,
        )
        return default
    return value if value > 0 else None


_VALID_REASONING_DRIFT_MODES: frozenset[str] = frozenset(
    {"judge", "embedding", "both", "off"}
)


def _read_reasoning_drift_mode_env(
    name: str, default: ReasoningDriftMode
) -> ReasoningDriftMode:
    """Return ``os.environ[name]`` as a ``ReasoningDriftMode``, or ``default``.

    Accepts exactly the four literal values of
    :data:`~goldfive.drift.reasoning.ReasoningDriftMode`
    (``"judge"`` / ``"embedding"`` / ``"both"`` / ``"off"``).
    Anything else logs a WARNING and falls back to ``default`` so a
    typo in the env never silently disables drift detection.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _VALID_REASONING_DRIFT_MODES:
        return value  # type: ignore[return-value]
    log.warning(
        "ignoring unknown %s=%r (expected one of %s); using default %r",
        name,
        raw,
        sorted(_VALID_REASONING_DRIFT_MODES),
        default,
    )
    return default


def _read_str_env(name: str, default: str) -> str:
    """Return ``os.environ[name]`` or ``default`` when missing.

    The empty string is a legitimate value for model names (llama.cpp
    tolerates ``model=""``), so we do NOT treat empty as "missing" here
    — the caller may explicitly want to clear a default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw


def _read_optional_str_env(name: str, default: str | None) -> str | None:
    """Return ``os.environ[name]`` or ``default``. Empty string -> ``None``.

    Used for fields whose Python type is ``str | None`` (e.g.
    ``api_key``, ``base_url``) where the semantic "unset" value is
    ``None``, not the empty string.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    stripped = raw.strip()
    if not stripped:
        return None
    return stripped


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EmbeddingConfig:
    """Configuration for the embedding backend used by reasoning-drift detectors.

    When ``base_url`` is ``None`` the OpenAI-compatible HTTP backend is
    skipped and the lazy-load path in :mod:`goldfive.drift._embed`
    falls through to sentence-transformers (if the ``goldfive[embedding]``
    extra is installed). When ``base_url`` is set, the HTTP backend is
    used exclusively — no silent fall-through to sentence-transformers
    on HTTP failure (matches the pre-#225 env-driven contract).
    """

    base_url: str | None = None
    model: str = ""
    api_key: str | None = None
    timeout_ms: int = 10_000

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        """Read ``GOLDFIVE_EMBEDDING_*`` env vars into an instance.

        Missing vars fall back to the field defaults. Preserves the
        exact env-var surface documented in
        :mod:`goldfive.drift._embed`.
        """
        defaults = cls()
        return cls(
            base_url=_read_optional_str_env(
                "GOLDFIVE_EMBEDDING_BASE_URL", defaults.base_url
            ),
            model=_read_str_env("GOLDFIVE_EMBEDDING_MODEL", defaults.model),
            api_key=_read_optional_str_env(
                "GOLDFIVE_EMBEDDING_API_KEY", defaults.api_key
            ),
            timeout_ms=_read_int_env(
                "GOLDFIVE_EMBEDDING_TIMEOUT_MS", defaults.timeout_ms
            ),
        )


@dataclasses.dataclass
class JudgeConfig:
    """Configuration for a dedicated LLM endpoint for goldfive's judges.

    When ``base_url`` is set, :func:`goldfive.wrap` routes the two
    LLM-as-a-judge drift detectors (the trajectory-level GOAL_DRIFT
    judge from goldfive#218 and the per-thinking-message
    reasoning-drift judge from goldfive#226) through this dedicated
    endpoint instead of inheriting the tree's LLM. Planner and
    goal_deriver keep using the tree's LLM — only the judges split
    off.

    This lets operators run the judges on a cheap local model (e.g.
    a llama.cpp / Ollama endpoint) while the tree's agent keeps
    billing against a cloud model. Without ``JudgeConfig``, the
    detect_llm path silently inherits the tree's LLM for the judges
    as well; see the named-model WARNING in :func:`goldfive.wrap`
    for the guardrail that surfaces that inheritance.

    When ``base_url`` is ``None`` (the default), the explicit
    ``goldfive.wrap(call_llm=...)`` argument — or the auto-detected
    tree LLM — is used for the judges. Precedence order:

    1. Explicit ``goldfive.wrap(call_llm=...)`` kwarg.
    2. ``JudgeConfig.base_url``.
    3. Auto-detected LLM from the agent (``detect_llm``).
    """

    base_url: str | None = None
    model: str = ""
    api_key: str | None = None
    timeout_ms: int = 10_000

    @classmethod
    def from_env(cls) -> JudgeConfig:
        """Read ``GOLDFIVE_JUDGE_*`` env vars into an instance.

        Missing vars fall back to the field defaults. Mirrors the
        shape of :meth:`EmbeddingConfig.from_env`.
        """
        defaults = cls()
        return cls(
            base_url=_read_optional_str_env(
                "GOLDFIVE_JUDGE_BASE_URL", defaults.base_url
            ),
            model=_read_str_env("GOLDFIVE_JUDGE_MODEL", defaults.model),
            api_key=_read_optional_str_env(
                "GOLDFIVE_JUDGE_API_KEY", defaults.api_key
            ),
            timeout_ms=_read_int_env(
                "GOLDFIVE_JUDGE_TIMEOUT_MS", defaults.timeout_ms
            ),
        )


@dataclasses.dataclass
class ToolLoopConfig:
    """Configuration for :class:`~goldfive.drift.tool_loops.ToolLoopTracker`.

    ``exact_threshold`` / ``name_threshold`` override the **work**
    category's WARNING tier only; the graduated CRITICAL tiers and
    the meta-category thresholds remain module constants. This
    preserves the pre-#204 single-threshold semantics for work tools.
    See :mod:`goldfive.drift.tool_loops` §"Graduated thresholds per
    category" for the full table.

    ``name_axis_max_severity`` caps the severity of same-name-varied-
    args (name-axis) hits that lack exact-repeat corroboration in the
    window -- ``"info"`` (default) keeps the uncorroborated name axis
    signal-only; ``"critical"`` restores the legacy uncapped
    behaviour. See :mod:`goldfive.drift.tool_loops` §"Name-axis
    precision".
    """

    window: int = 10
    exact_threshold: int = 3
    name_threshold: int = 5
    alternating_threshold: int = 5
    name_axis_max_severity: str = "info"

    @classmethod
    def from_env(cls) -> ToolLoopConfig:
        """Read ``GOLDFIVE_TOOL_LOOP_*`` env vars into an instance.

        Names preserved from :func:`goldfive.drift.tool_loops.load_thresholds_from_env`.
        """
        from goldfive.drift.tool_loops import _read_severity_env

        defaults = cls()
        return cls(
            window=_read_int_env("GOLDFIVE_TOOL_LOOP_WINDOW", defaults.window),
            exact_threshold=_read_int_env(
                "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD", defaults.exact_threshold
            ),
            name_threshold=_read_int_env(
                "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD", defaults.name_threshold
            ),
            alternating_threshold=_read_int_env(
                "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD",
                defaults.alternating_threshold,
            ),
            name_axis_max_severity=_read_severity_env(
                "GOLDFIVE_TOOL_LOOP_NAME_AXIS_MAX_SEVERITY",
                defaults.name_axis_max_severity,
            ),
        )


@dataclasses.dataclass
class ReasoningDriftConfig:
    """Thresholds for the reasoning-drift detectors.

    Field defaults match the module-level constants in
    :mod:`goldfive.drift.reasoning` one-for-one. Operators who want to
    tune these previously had to fork the module; now they can either
    set the corresponding env var (``GOLDFIVE_DRIFT_*``) or pass a
    :class:`RuntimeConfig` explicitly.

    Installation is process-wide via
    :func:`goldfive.drift.reasoning.configure` — "last Runner wins"
    for processes that host multiple Runners concurrently. The
    tradeoff is documented at the install site. If two-Runners-in-
    one-process with different drift thresholds becomes a real use
    case, #225's follow-up plan is to move the config onto
    :class:`~goldfive.types.Session` and read it per-session in each
    detector.
    """

    mode: ReasoningDriftMode = DEFAULT_REASONING_DRIFT_MODE
    off_topic_distance_threshold: float = 0.7
    intent_divergence_healthy_similarity: float = 0.6
    intent_divergence_minor_similarity: float = 0.4
    intent_divergence_warning_similarity: float = 0.2
    looping_reasoning_similarity_threshold: float = 0.9
    reasoning_cluster_similarity_threshold: float = 0.75
    looping_reasoning_hash_window: int = 5
    max_concurrent_judges: int = 3
    """Per-steerer cap on concurrently RUNNING background reasoning-judge
    LLM calls (judge-scheduling guards).

    N agents thinking in the same event-loop tick used to fire N
    parallel judge calls (each with a 16384-token output ceiling) at
    the judge endpoint — which, under the default :func:`goldfive.wrap`
    auto-detect, is the agent tree's OWN model. The cap bounds the
    burst; requests beyond it queue on a per-steerer
    ``asyncio.Semaphore`` and per-(agent, task) queued windows coalesce
    onto the newest observation. Values below 1 are clamped to 1.
    Env: ``GOLDFIVE_DRIFT_MAX_CONCURRENT_JUDGES``.
    """
    fallback_to_content_when_no_reasoning: bool = False
    """Synthesize a reasoning signal from the response body on
    non-thinking models (goldfive#263).

    Some local models (Gemma 4, Mistral, several base-model
    deployments) do not emit a separate chain-of-thought stream.
    :func:`goldfive.adapters._adk_plugin._extract_reasoning` then
    returns an empty string and the
    :meth:`~goldfive.steerer.DefaultSteerer.observe_reasoning` path
    never fires — so the LLM-as-a-judge reasoning detectors
    (OFF_TOPIC, GOAL_DRIFT, INTENT_DIVERGENCE, LOOPING_REASONING)
    silently disarm across the entire run. Reproduced live
    2026-05-11 on Gemma-4 (session ``4a721a07``): zero
    ``reasoning_judge`` invocations across the run despite agent
    flows that would have tripped on a thinking-capable model.

    When this flag is True, the ADK plugin's
    ``after_model_callback`` falls back to feeding the response's
    ``content`` body (the agent's answer text) into
    ``observe_reasoning`` if and only if the real reasoning
    extraction returned empty. The trade-off is intentionally
    lossy: "what the agent decided" mixes with "what it reasoned
    about", so OFF_TOPIC / GOAL_DRIFT signals will be noisier than
    on a thinking model that emits a clean chain-of-thought. The
    user accepts this — a lossy reasoning signal is strictly
    better than no signal at all on Gemma-class deployments.

    Default ``False`` so the behaviour change is opt-in. Operators
    running on non-thinking models flip this to True (typed config
    or ``GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT=1``). The flag has no
    effect on responses that DO carry real reasoning content: real
    reasoning always wins; the fallback only kicks in on a genuine
    empty.
    """

    @classmethod
    def from_env(cls) -> ReasoningDriftConfig:
        """Read ``GOLDFIVE_DRIFT_*`` env vars into an instance.

        New env surface introduced by #225 — the reasoning-drift
        thresholds had no env wiring before. Names are chosen to be
        self-descriptive and lowercased versions match the dataclass
        fields verbatim.
        """
        defaults = cls()
        return cls(
            mode=_read_reasoning_drift_mode_env(
                "GOLDFIVE_DRIFT_REASONING_MODE", defaults.mode
            ),
            off_topic_distance_threshold=_read_float_env(
                "GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE",
                defaults.off_topic_distance_threshold,
            ),
            intent_divergence_healthy_similarity=_read_float_env(
                "GOLDFIVE_DRIFT_INTENT_HEALTHY_SIMILARITY",
                defaults.intent_divergence_healthy_similarity,
            ),
            intent_divergence_minor_similarity=_read_float_env(
                "GOLDFIVE_DRIFT_INTENT_MINOR_SIMILARITY",
                defaults.intent_divergence_minor_similarity,
            ),
            intent_divergence_warning_similarity=_read_float_env(
                "GOLDFIVE_DRIFT_INTENT_WARNING_SIMILARITY",
                defaults.intent_divergence_warning_similarity,
            ),
            looping_reasoning_similarity_threshold=_read_float_env(
                "GOLDFIVE_DRIFT_LOOPING_SIMILARITY",
                defaults.looping_reasoning_similarity_threshold,
            ),
            reasoning_cluster_similarity_threshold=_read_float_env(
                "GOLDFIVE_DRIFT_CLUSTER_SIMILARITY",
                defaults.reasoning_cluster_similarity_threshold,
            ),
            looping_reasoning_hash_window=_read_int_env(
                "GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW",
                defaults.looping_reasoning_hash_window,
            ),
            max_concurrent_judges=_read_int_env(
                "GOLDFIVE_DRIFT_MAX_CONCURRENT_JUDGES",
                defaults.max_concurrent_judges,
            ),
            fallback_to_content_when_no_reasoning=_read_bool_env(
                "GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT",
                defaults.fallback_to_content_when_no_reasoning,
            ),
        )


@dataclasses.dataclass
class GoalDriftConfig:
    """Scheduling for the trajectory-level GOAL_DRIFT judge (#143).

    ``check_interval`` is the number of agent-invocation turns between
    judge calls; ``activity_window`` bounds the agent-activity subset of
    ``session.recent_events`` (goldfive#239 — the unified buffer that
    replaced the historical ``recent_agent_activity``) and hence the
    prompt size. Both were previously ``DefaultSteerer`` kwargs with no
    env or ``wrap()``-level override; this config surfaces them.
    """

    check_interval: int = 5
    activity_window: int = 10

    @classmethod
    def from_env(cls) -> GoalDriftConfig:
        """Read ``GOLDFIVE_GOAL_DRIFT_*`` env vars into an instance."""
        defaults = cls()
        return cls(
            check_interval=_read_int_env(
                "GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL",
                defaults.check_interval,
            ),
            activity_window=_read_int_env(
                "GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW",
                defaults.activity_window,
            ),
        )


#: Fallback wait budget, in milliseconds, for a ``report_awaiting_approval``
#: call that omits ``timeout_ms`` (or passes ``<= 0``) while a control
#: channel is attached. Referenced by
#: :attr:`SteeringConfig.approval_default_timeout_ms` and by the handler's
#: fallback when the steerer carries no :class:`SteeringConfig` at all.
DEFAULT_APPROVAL_TIMEOUT_MS: int = 600_000


@dataclasses.dataclass
class SteeringConfig:
    """Drift → steer promotion policy (goldfive-steer-unification).

    Controls how goldfive-detected drifts (reasoning judge, embedding
    detectors, goal-drift, loop detectors, tool-loops, confabulation)
    interact with the cancel-in-flight + refine + restart-message
    machinery previously reserved for ``USER_STEER``.

    ``threshold``:

    * ``"off"`` — every goldfive-detected drift stays on the legacy
      passive ``REFINE_PLAN`` path (no cancel-in-flight, no
      restart-message). For operators who want the softer pre-
      unification semantics.
    * ``"warning"`` (default) — ``WARNING``+ goldfive-detected drifts
      are promoted to a full steer (cancel in-flight + stamp
      ``goldfive.active_steer.*`` + refine + restart message).
    * ``"critical"`` — only ``CRITICAL`` goldfive-detected drifts are
      promoted. Useful for high-noise trees where the WARNING judge
      fires too aggressively.

    ``suppression_window_turns`` is the number of *logical turns*
    within which a fresh user-authored steer suppresses any
    goldfive-authored steer promotion. A logical turn is one reasoning
    observation (``Session._reasoning_turn`` — one tick per model
    response fed through the drift pipeline), NOT a raw event-sequence
    increment: keying on the per-event ``_next_sequence`` let
    observability-event volume shrink the effective window
    (goldfive#441). The goldfive drift still surfaces as a
    ``DriftDetected`` event with ``suppressed_by_user_steer=true``; no
    cancel or refine fires. Default ``3`` keeps a live operator
    override dominant across a few agent turns.

    ``observation_only`` (goldfive#254) gates the actual steering
    injection points on :class:`~goldfive.steerer.DefaultSteerer`:

    * the would-be revised plan replacing ``session.plan`` in
      :meth:`~goldfive.steerer.DefaultSteerer._apply_revision`;
    * the ``GOLDFIVE_STEER`` ControlMessage enqueue in
      :meth:`~goldfive.steerer.DefaultSteerer._dispatch_goldfive_steer_control`;
    * the ``request_invocation_cancel`` plugin call in
      :meth:`~goldfive.steerer.DefaultSteerer.request_invocation_cancel`;
    * the Level 2 nudge enqueues onto ``session.pending_nudges``
      (``_dispatch_nudge`` and the post-ABSORB handoff, goldfive#202) —
      the overlay would otherwise drain the queue into a
      goldfive-authored synthetic user turn and re-invoke the tree.
      The executor's drain carries a matching defense-in-depth gate
      (goldfive#264 pattern) for steerer subclasses that bypass the
      dispatcher.

    The plan-install gate suppresses only **corrective**
    goldfive-authored revisions. Three categories always land as real
    revisions even under ``observation_only=True``:

    * **bootstrap** — first install on a cold session (``prev is None``);
    * **user-authored** — operator ``ControlMessage`` STEER deliveries
      (``drift.authored_by == "user"``);
    * **discovery** — ``DriftKind.NEW_WORK_DISCOVERED`` revisions
      (goldfive#258), covering both the runner's turn-1 install through
      :meth:`install_initial_plan` (where ``session.plan`` was seeded
      with ``Plan.empty()`` so ``prev is None`` no longer holds) and
      the turn N+1 replan through :meth:`install_revision_for_drift`.
      Discovery is the planner / a sub-agent describing observed work,
      not a framework-driven correction.

    Detection still runs in full (reasoning judges, embedding
    detectors, goal-drift, looping detectors, CAPABILITY_MISMATCH, …)
    and ``planner.refine_steer`` still runs — operators can see what
    the planner WOULD have produced via the ``PlanRevised`` event with
    ``dry_run=True``. The in-flight invocation is otherwise untouched.

    **Default is ``True``** — a deliberate behaviour change from the
    pre-#254 implicit active-steering default. Operators graduate to
    active steering explicitly by passing
    ``RuntimeConfig(steering=SteeringConfig(observation_only=False))``
    to :func:`goldfive.wrap`, or by setting the env var
    ``GOLDFIVE_STEER_OBSERVATION_ONLY=0`` / ``false`` / ``no``.

    The ``DriftDetected`` event still fires unchanged in observation
    mode — detection is independent of injection. Only the three
    write-paths above are skipped; everything else (logging, sink
    emission, drift lifecycle, suppression accounting) keeps running.
    """

    threshold: str = "warning"
    suppression_window_turns: int = 3
    observation_only: bool = dataclasses.field(
        default_factory=_resolve_observation_only_default
    )
    #: Names of :class:`~goldfive.context_editor.ContextEditRule` rules to
    #: register on the ADK plugin's :class:`~goldfive.context_editor.ContextEditor`
    #: (goldfive#397). ``None`` (the default) AND an empty list both leave
    #: the editor unwired — the plugin's ``before_model_callback`` never
    #: even instantiates the editor and the codepath is zero-overhead.
    #:
    #: Set to a list of rule names (e.g. ``["prune_cancelled_reasoning"]``)
    #: to opt in. Unknown rule names are logged and ignored at registration
    #: time; an empty list after filtering also keeps the editor unwired.
    #:
    #: Per-rule (rather than a single master switch) so e2e regressions
    #: from a single rule can be bisected without disabling the whole
    #: capability. See ``docs/design/CONTEXT-EDITING.md`` for the rule
    #: catalog and the rationale behind drop-only edits.
    context_editor_rules: list[str] | None = None
    #: Plan-descriptive growth fallback for unmatched delegations
    #: (goldfive#423 PR 2). When ``True`` and the structural capability
    #: detector returns a Rule C (out-of-DAG-order) verdict, the steerer
    #: synthesises a ``discovered=True`` task via
    #: :meth:`~goldfive.plan_reviser.PlanReviser.install_descriptive_growth`
    #: and re-pins the delegation to it instead of firing the
    #: CAPABILITY_MISMATCH drift. Rule A and Rule B are unaffected. When
    #: ``False`` (the default) the existing CAPABILITY_MISMATCH Rule C
    #: path fires as today, so PR 2 lands behind the flag without
    #: behaviour change for existing tests/runs. PR 4 flips the default
    #: after sufficient validation. Env: ``GOLDFIVE_STEER_DESCRIPTIVE_GROWTH``.
    #:
    #: Design ref: ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.3 + §9
    #: (PR table). The flag gates ONLY the §4.3 fallback; the data-model
    #: fields shipped in PR 1 (``Task.discovered``,
    #: ``Task.discovery_identity_hash``, ``DelegationObserved.tool_args_json``)
    #: are always available regardless of this flag.
    descriptive_growth_enabled: bool = False
    #: Wait budget, in milliseconds, substituted when a
    #: ``report_awaiting_approval`` call omits ``timeout_ms`` (or passes
    #: ``<= 0``) and a control channel is attached. Historically that
    #: path awaited the APPROVE / REJECT waiter with no bound, so an
    #: operator who never dispatched a decision hung the tool call — and
    #: the run — forever (no invocation wall clock covers tool waits).
    #: On expiry the handler returns ``decision="timeout"`` and emits a
    #: ``HUMAN_INTERVENTION_REQUIRED`` WARNING drift so operators see
    #: the unresolved approval. An explicit positive ``timeout_ms`` from
    #: the agent still wins; a non-positive value here falls back to
    #: :data:`DEFAULT_APPROVAL_TIMEOUT_MS`.
    #: Env: ``GOLDFIVE_STEER_APPROVAL_DEFAULT_TIMEOUT_MS``.
    approval_default_timeout_ms: int = DEFAULT_APPROVAL_TIMEOUT_MS
    #: Deadline, in seconds, on the executor's pause wait after a Level-4
    #: ``GOLDFIVE_PAUSE_ESCALATE`` dispatch. The pre-task / pre-stage
    #: pause loop historically awaited ``ControlChannel.receive()`` with
    #: no bound, so an unattended deployment whose ladder escalated to a
    #: pause hung forever waiting for an operator RESUME / CANCEL /
    #: STEER that never came. When set, the executor aborts the run on
    #: expiry: background steerer/judge tasks are drained, every
    #: non-terminal task is CANCELLED, and a ``RunAborted`` carrying the
    #: escalation lineage (originating drift kind + ladder level) is
    #: emitted. ``None`` (the default) preserves the block-forever
    #: behaviour for Level 4 — EXCEPT for Level 5 (TERMINATE), which by
    #: definition must terminate: with no configured deadline it falls
    #: back to
    #: :data:`goldfive.drift_observer.DEFAULT_TERMINATE_PAUSE_DEADLINE_S`.
    #: Operator-issued ``PAUSE`` controls are never bounded by this knob.
    #: Env: ``GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S``.
    pause_escalate_deadline_s: float | None = None
    #: Wall-clock stall watchdog (default OFF). When enabled, the ADK
    #: plugin spawns one background task per dispatch that watches a
    #: liveness watermark — the max of every
    #: :attr:`~goldfive.types.Session.task_last_progress_at` stamp and
    #: :attr:`~goldfive.types.Session.last_observed_event_at` (stamped on
    #: every observation dispatched into the drift pipeline). When the
    #: watermark goes silent for :attr:`stall_timeout_s` seconds, a
    #: ``TASK_TIMEOUT`` drift fires at WARNING, escalating to CRITICAL
    #: on continued silence. Routed through the normal drift dispatch,
    #: so under ``observation_only`` (the production default) the drift
    #: is telemetry-only; in active mode the intervention ladder handles
    #: it. This is the producer for runs wedged in a hung async tool
    #: call or idling with no task transitions — which previously
    #: produced ZERO signal. Env: ``GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED``.
    stall_watchdog_enabled: bool = False
    #: Idle threshold, in seconds, before the stall watchdog fires its
    #: first ``TASK_TIMEOUT`` WARNING; each further multiple of the
    #: timeout with no fresh activity fires a CRITICAL. Non-positive
    #: values disable the watchdog even when
    #: :attr:`stall_watchdog_enabled` is True.
    #: Env: ``GOLDFIVE_STEER_STALL_TIMEOUT_S``.
    stall_timeout_s: float = 600.0

    @classmethod
    def from_env(cls) -> SteeringConfig:
        """Read ``GOLDFIVE_STEER_*`` env vars into an instance.

        Env surface:

        * ``GOLDFIVE_STEER_THRESHOLD`` — ``off`` / ``warning`` /
          ``critical`` (case-insensitive).
        * ``GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS`` — positive int.
        * ``GOLDFIVE_STEER_OBSERVATION_ONLY`` — boolean
          (``1``/``true``/``yes``/``on`` truthy; ``0``/``false``/
          ``no``/``off`` falsy; case-insensitive). Defaults to the
          built-in default (``True`` in production, flipped to
          ``False`` for the goldfive test suite via the autouse
          ``_goldfive_active_steering_default`` fixture).
        * ``GOLDFIVE_STEER_CONTEXT_EDITOR_RULES`` — comma-separated rule
          names (goldfive#397). Empty / unset → ``None`` (editor unwired).
          Example: ``GOLDFIVE_STEER_CONTEXT_EDITOR_RULES=prune_cancelled_reasoning``.
        * ``GOLDFIVE_STEER_APPROVAL_DEFAULT_TIMEOUT_MS`` — positive int
          (milliseconds); non-positive / non-integer values fall back
          to the default.
        * ``GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S`` — positive float
          (seconds); non-positive values disable the deadline (``None``),
          non-float values fall back to the default.
        * ``GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED`` — boolean (same
          literals as ``GOLDFIVE_STEER_OBSERVATION_ONLY``). Default
          False (watchdog off).
        * ``GOLDFIVE_STEER_STALL_TIMEOUT_S`` — float (seconds);
          non-float values fall back to the default (600).
        """
        defaults = cls()
        raw_rules = os.environ.get("GOLDFIVE_STEER_CONTEXT_EDITOR_RULES", "").strip()
        rules: list[str] | None
        if not raw_rules:
            rules = defaults.context_editor_rules
        else:
            rules = [r.strip() for r in raw_rules.split(",") if r.strip()] or None
        return cls(
            threshold=_read_steer_threshold_env(
                "GOLDFIVE_STEER_THRESHOLD", defaults.threshold
            ),
            suppression_window_turns=_read_int_env(
                "GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS",
                defaults.suppression_window_turns,
            ),
            observation_only=_read_bool_env(
                "GOLDFIVE_STEER_OBSERVATION_ONLY",
                defaults.observation_only,
            ),
            context_editor_rules=rules,
            descriptive_growth_enabled=_read_bool_env(
                "GOLDFIVE_STEER_DESCRIPTIVE_GROWTH",
                defaults.descriptive_growth_enabled,
            ),
            approval_default_timeout_ms=_read_int_env(
                "GOLDFIVE_STEER_APPROVAL_DEFAULT_TIMEOUT_MS",
                defaults.approval_default_timeout_ms,
            ),
            pause_escalate_deadline_s=_read_optional_float_env(
                "GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S",
                defaults.pause_escalate_deadline_s,
            ),
            stall_watchdog_enabled=_read_bool_env(
                "GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED",
                defaults.stall_watchdog_enabled,
            ),
            stall_timeout_s=_read_float_env(
                "GOLDFIVE_STEER_STALL_TIMEOUT_S",
                defaults.stall_timeout_s,
            ),
        )


@dataclasses.dataclass
class AgentConfig:
    """Per-agent LLM-call budget for ADK sub-agent invocations (goldfive#256).

    Distinct from :class:`JudgeConfig` (goldfive's internal judge calls)
    and :attr:`goldfive.planner.LLMPlanner.MAX_OUTPUT_TOKENS` (the planner's
    own LLM calls); this covers the user-tree sub-agent calls flowing
    through ADK / litellm — coordinator, research_agent, web_developer_agent,
    reviewer_agent, debugger_agent, and anyone else wrapped by
    :func:`goldfive.wrap`.

    ``max_output_tokens`` caps the generation per ADK sub-agent LLM call.
    Without it, a wandering / looping generation can run for many minutes
    and consume tens of thousands of tokens with no framework
    intervention (live e2e 2026-05-11, ice cream session ``62dde1a6``:
    30K+ tokens for a 5-bullet-point research task at ~55 tok/s with
    sustained 100% speculative-decoding acceptance — low-entropy
    repetitive output that nothing was bounding). The cap is applied as
    a STRUCTURAL CEILING in
    :meth:`goldfive.adapters._adk_plugin._GoldfiveADKPlugin.before_model_callback`:
    if the sub-agent (or ADK's defaults) already supplied a smaller
    ``max_output_tokens`` on ``llm_request.config``, that smaller value
    wins — goldfive only ratchets the cap DOWN, never up.

    ``call_timeout_ms`` is the wall-clock budget per call. On expiry an
    ``LLM_CALL_TIMEOUT`` drift fires (CRITICAL, capacity-shaped) so the
    existing drift dispatch handles escalation; in active mode
    (``SteeringConfig.observation_only=False``) the invocation is also
    flagged for cooperative cancel. Under ``observation_only`` (the
    production default) the drift is telemetry-only — the in-flight
    call is never cancelled, since healthy models can genuinely need
    longer than the budget (see below).
    Default 120s — Qwen 35B-class models can take 60-90s on long
    prompts; operators who genuinely need longer (slow judges, weak
    hardware, multi-step research synthesis) override via env or the
    typed config. This default is dramatically tighter than the
    pre-goldfive#256
    :data:`goldfive.adapters._adk_plugin.DEFAULT_LLM_CALL_TIMEOUT_MS`
    (30 min), which was sized as a pathological-hang ceiling, not a
    latency SLO. Operators who want the legacy 30-minute backstop pass
    ``call_timeout_ms=1_800_000`` (or set
    ``GOLDFIVE_AGENT_CALL_TIMEOUT_MS=1800000``).

    Both fields can be overridden via env (``GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS``
    / ``GOLDFIVE_AGENT_CALL_TIMEOUT_MS``) so operators tune without code
    changes.
    """

    max_output_tokens: int = 16384
    call_timeout_ms: int = 120_000

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Read ``GOLDFIVE_AGENT_*`` env vars into an instance.

        Missing vars fall back to the field defaults. Non-integer /
        non-positive values are ignored (same semantics as the other
        sub-configs — see :func:`_read_int_env`).
        """
        defaults = cls()
        return cls(
            max_output_tokens=_read_int_env(
                "GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS",
                defaults.max_output_tokens,
            ),
            call_timeout_ms=_read_int_env(
                "GOLDFIVE_AGENT_CALL_TIMEOUT_MS",
                defaults.call_timeout_ms,
            ),
        )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RuntimeConfig:
    """Per-Runner typed configuration aggregate.

    Pass to :func:`goldfive.wrap` via the ``runtime=`` kwarg to install
    all four sub-configs at once. When the kwarg is omitted, ``wrap()``
    builds an instance from the environment via :meth:`from_env` —
    byte-identical to pre-#225 behaviour for callers that relied on
    env vars or accepted the built-in defaults.
    """

    embedding: EmbeddingConfig = dataclasses.field(default_factory=EmbeddingConfig)
    tool_loops: ToolLoopConfig = dataclasses.field(default_factory=ToolLoopConfig)
    reasoning_drift: ReasoningDriftConfig = dataclasses.field(
        default_factory=ReasoningDriftConfig
    )
    goal_drift: GoalDriftConfig = dataclasses.field(default_factory=GoalDriftConfig)
    judge: JudgeConfig = dataclasses.field(default_factory=JudgeConfig)
    steering: SteeringConfig = dataclasses.field(default_factory=SteeringConfig)
    agent: AgentConfig = dataclasses.field(default_factory=AgentConfig)

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        """Build a :class:`RuntimeConfig` by reading every supported env var.

        Aggregates the sub-``from_env`` calls. Each sub-config is
        independent: a missing env var in one subsystem does not affect
        the others. The result is a fresh instance; callers may mutate
        it in place or :func:`dataclasses.replace` to derive a variant.
        """
        return cls(
            embedding=EmbeddingConfig.from_env(),
            tool_loops=ToolLoopConfig.from_env(),
            reasoning_drift=ReasoningDriftConfig.from_env(),
            goal_drift=GoalDriftConfig.from_env(),
            judge=JudgeConfig.from_env(),
            steering=SteeringConfig.from_env(),
            agent=AgentConfig.from_env(),
        )
