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
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from goldfive.types import DriftKind, DriftSeverity

if TYPE_CHECKING:
    from goldfive.types import Plan, Session


_DriftEnum = TypeVar("_DriftEnum", DriftKind, DriftSeverity)


def _normalize_drift_field(
    value: _DriftEnum | str, enum_cls: type[_DriftEnum]
) -> _DriftEnum | str:
    """Coerce a verdict drift field to its enum when the value matches a member.

    Used by :meth:`JudgeVerdict.__post_init__` to normalise the
    ``drift_kind`` / ``severity`` fields. Accepts either an ``enum_cls``
    member — :class:`DriftKind` or :class:`DriftSeverity`, the preferred
    typed form — or the legacy lowercase string. A string matching a
    known ``enum_cls`` value is upgraded to the enum so consumers get a
    real enum off ``JudgeVerdict``; an empty string (no drift) or an
    unrecognised custom string is returned unchanged so a forward-
    compatible / domain-specific judge is not broken. Both enums are
    :class:`~enum.StrEnum`, so the upgraded value still compares equal to
    the original lowercase string — back-compat is preserved.
    """
    if isinstance(value, enum_cls) or not value:
        return value
    try:
        return enum_cls(str(value))
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

    ``note_to_agent`` (AGENCY-PRESERVATION.md PR 4) is an optional
    observation addressed to the *wrapped agent*, authored by the judge
    in the same evaluation that produced the verdict: one or two
    neutral, factual sentences stating what was observed relative to
    the user's goal — no commands, no fault language; question form
    when the judge's confidence is low. Drift-flavoured verdicts thread
    it onto :attr:`~goldfive.types.DriftEvent.note_to_agent` (see
    ``DefaultSteerer._drift_from_judge_verdict``) where
    :mod:`goldfive.observer_notes` prefers it verbatim over ``detail``
    when composing agent-facing notes. Optional and additive: judges
    that never populate it (including every pre-PR-4 judge) degrade
    gracefully to the ``detail`` fallback. Not emitted on the
    ``JudgementEmitted`` proto envelope in this PR (no proto regen;
    the wire surface belongs to the PR 5 telemetry pass).
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
    # AGENCY-PRESERVATION.md PR 4 — judge-authored agent-facing
    # observation (see class docstring). Follows the additive-field
    # pattern the enum-typed drift fields established: optional,
    # default-empty, back-compat with every existing constructor call.
    note_to_agent: str = ""

    def __post_init__(self) -> None:
        # The dataclass is ``frozen``; ``object.__setattr__`` is the
        # sanctioned way to write a derived/normalised value during
        # construction. Normalising here (rather than at every read
        # site) means every consumer — the steerer, sinks, external
        # callers — sees the typed enum without each having to coerce.
        object.__setattr__(
            self, "drift_kind", _normalize_drift_field(self.drift_kind, DriftKind)
        )
        object.__setattr__(
            self, "severity", _normalize_drift_field(self.severity, DriftSeverity)
        )


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
