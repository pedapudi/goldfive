"""Core types for the pluggable-judges surface.

* :class:`Judge` — async ``evaluate(ctx) -> JudgeVerdict`` protocol.
  Stateless from goldfive's perspective: the runtime calls
  ``evaluate`` once per observation point and emits the returned
  verdict. Implementations are free to keep their own state across
  calls (rate limiters, cached embeddings, accumulated counters).
* :class:`JudgeContext` — frozen observation snapshot fed into
  :meth:`Judge.evaluate`. Carries reasoning text, plan, transcript,
  session state, and current task / agent pinning so a judge can
  inspect "what the agent just thought / did and why".
* :class:`JudgeVerdict` — frozen result. Carries optional fields for
  each of the four verdict flavours (drift / rubric / boolean /
  numeric); judges populate the subset relevant to their signal and
  leave the others at their default-empty values.

The dataclasses are deliberately frozen so consumers can hash / cache
them, and so a misbehaving judge can't mutate state held by the
steerer between emission and dispatch.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from goldfive.types import DriftKind, DriftSeverity

if TYPE_CHECKING:
    from goldfive.types import Plan, Session


def _normalize_drift_kind(value: DriftKind | str) -> DriftKind | str:
    """Coerce a verdict ``drift_kind`` to :class:`DriftKind` when possible.

    Accepts either a :class:`DriftKind` enum member (the preferred,
    typed form) or the legacy lowercase string. A string that matches a
    known :class:`DriftKind` value is upgraded to the enum so consumers
    get a real enum off ``JudgeVerdict.drift_kind``; an empty string or
    an unrecognised custom string is returned unchanged so a forward-
    compatible / domain-specific judge is not broken. :class:`DriftKind`
    is a :class:`~enum.StrEnum`, so the upgraded value still compares
    equal to the original lowercase string — back-compat is preserved.
    """
    if isinstance(value, DriftKind) or not value:
        return value
    try:
        return DriftKind(str(value))
    except ValueError:
        return value


def _normalize_drift_severity(value: DriftSeverity | str) -> DriftSeverity | str:
    """Coerce a verdict ``severity`` to :class:`DriftSeverity` when possible.

    Mirrors :func:`_normalize_drift_kind`: a :class:`DriftSeverity` enum
    member passes through, a recognised legacy lowercase string is
    upgraded to the enum, and an empty / unrecognised string is returned
    unchanged. :class:`DriftSeverity` is a :class:`~enum.StrEnum`, so the
    upgrade is transparent to existing string-equality consumers.
    """
    if isinstance(value, DriftSeverity) or not value:
        return value
    try:
        return DriftSeverity(str(value))
    except ValueError:
        return value


@dataclasses.dataclass(frozen=True)
class JudgeContext:
    """Snapshot fed into :meth:`Judge.evaluate`.

    Fields
    ------
    reasoning_text:
        The chain-of-thought / thinking block the model just emitted,
        if any. Empty string when the observation point is not a
        reasoning emit (e.g. tool-loops, goal-drift trajectory check).
    plan:
        The current plan at observation time. Read-only — judges MUST
        NOT mutate the plan; refines are driven by the steerer via
        :class:`~goldfive.protocols.Planner`.
    transcript:
        Recent reasoning / activity blocks for the session, bounded
        and ordered oldest-first. Stable across calls only within a
        single observation point; judges that need history-of-history
        should keep their own state.
    session_state:
        The live :class:`~goldfive.types.Session`. Judges MAY read
        ``session.reasoning_history`` / ``session.recent_events`` for
        context but MUST NOT mutate either — goldfive's state-
        ownership audit will flag stray writes.
    current_task_id:
        Task id pinned at observation time (the task the agent was
        executing). Empty when no task was active.
    current_agent_id:
        Agent name pinned at observation time. Empty when no agent
        was active (trajectory-level observation points).
    extras:
        Open-ended dict for observation-point specific extras (e.g.
        the tool name + args for a tool-loop check, the model name
        for a cost / latency judge). Judges MUST tolerate missing
        keys.
    """

    reasoning_text: str = ""
    plan: Plan | None = None
    transcript: tuple[str, ...] = ()
    session_state: Session | None = None
    current_task_id: str = ""
    current_agent_id: str = ""
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class JudgeVerdict:
    """Result returned by :meth:`Judge.evaluate`.

    Verdict flavours (any combination of fields may be populated;
    the runtime picks ``verdict_kind`` from the first populated
    flavour in the order: drift, rubric, boolean, numeric):

    * **drift** — set ``drift_emitted = True`` and populate
      ``drift_kind`` + ``severity``. Both fields accept the typed
      :class:`~goldfive.types.DriftKind` /
      :class:`~goldfive.types.DriftSeverity` enums (the preferred
      form). The legacy lowercase-string form (``"tool_error"``,
      ``"critical"``, ...) is still accepted for back-compat: a string
      matching a known enum value is normalised to the enum at
      construction, so a verdict built either way exposes a real enum
      on ``verdict.drift_kind`` / ``verdict.severity``. Because both
      enums are :class:`~enum.StrEnum`, the normalised value still
      compares equal to its lowercase string — existing
      string-equality consumers see no change. An empty string (no
      drift) or an unrecognised custom string is left untouched. The
      steerer fires both :class:`DriftDetected` (back-compat) and
      :class:`JudgementEmitted`.
    * **rubric** — set ``rubric_score`` (an aggregate in [0, 1] or any
      domain-defined range) and optionally ``rubric_dimensions`` with
      per-dimension sub-scores.
    * **boolean** — set ``boolean_result`` to ``True`` / ``False`` for a
      pass / fail contract.
    * **numeric** — set ``numeric_value`` and ``metric_name`` for a
      single named metric (cost_usd, latency_ms, tokens_out, ...).

    ``detail`` is a free-form one-line human-readable explanation that
    every flavour can populate. Surfaced on
    :class:`JudgementEmitted.detail` for the UI.
    """

    # drift-flavored (back-compat with existing detectors). ``drift_kind``
    # / ``severity`` accept either the typed enum (preferred) or the
    # legacy lowercase string; ``__post_init__`` normalises a recognised
    # string to its enum so consumers always get a real enum value back.
    drift_emitted: bool = False
    drift_kind: DriftKind | str = ""
    severity: DriftSeverity | str = ""
    # rubric-flavored
    rubric_score: float | None = None
    rubric_dimensions: dict[str, float] = dataclasses.field(default_factory=dict)
    # boolean-flavored
    boolean_result: bool | None = None
    # numeric
    numeric_value: float | None = None
    metric_name: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        # The dataclass is ``frozen``; ``object.__setattr__`` is the
        # sanctioned way to write a derived/normalised value during
        # construction. Normalising here (rather than at every read
        # site) means every consumer — the steerer, sinks, external
        # callers — sees the typed enum without each having to coerce.
        normalized_kind = _normalize_drift_kind(self.drift_kind)
        if normalized_kind is not self.drift_kind:
            object.__setattr__(self, "drift_kind", normalized_kind)
        normalized_severity = _normalize_drift_severity(self.severity)
        if normalized_severity is not self.severity:
            object.__setattr__(self, "severity", normalized_severity)


@runtime_checkable
class Judge(Protocol):
    """Async judge protocol.

    Implementations expose a stable ``name`` (used as the wire key on
    :class:`JudgementEmitted.judge_name` and as the operator-facing
    opt-in / opt-out token) and an async ``evaluate`` method.

    The runtime calls ``evaluate`` once per observation point. Judges
    that have nothing to say SHOULD return an empty-default
    :class:`JudgeVerdict` (no flavour populated); the steerer then
    skips :class:`JudgementEmitted` emission for that judge on that
    point. Judges that always want a verdict on the wire (for telemetry
    completeness) can populate ``metric_name`` + ``numeric_value = 0.0``
    to land a numeric heartbeat.

    Errors raised by ``evaluate`` are caught by the steerer and logged
    at WARNING; one misbehaving judge MUST NOT break the run or
    suppress other judges' verdicts.
    """

    name: str

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict: ...
