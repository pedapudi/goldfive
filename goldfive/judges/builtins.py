"""Built-in :class:`Judge` implementations wrapping existing detectors.

Each built-in judge is a thin :class:`Judge` shim around the
detector function that already lives under :mod:`goldfive.drift`.
The wrapper is deliberately stateless: when a verdict fires, it
populates the drift-flavoured fields of :class:`JudgeVerdict` so the
steerer can both:

* fire ``DriftDetected`` exactly as the pre-judges path did
  (back-compat — every existing sink + downstream consumer keeps
  seeing the same wire event), AND
* fire ``JudgementEmitted`` keyed on ``judge_name`` so operators
  who switch to the judge-centric event surface get uniform
  telemetry across drift / rubric / boolean / numeric verdicts.

The factory functions (``reasoning_drift()``, ``goal_drift()``, ...)
return ready-to-install judge instances. Operators pass them to
:func:`goldfive.wrap` via the ``judges=`` kwarg::

    runner = goldfive.wrap(
        agent,
        judges=[
            goldfive.builtin_judges.reasoning_drift(),
            goldfive.builtin_judges.refusal(),
            MyCustomLengthJudge(),
        ],
    )

Each factory returns a fresh instance so callers can construct
multiple runners without state-bleed.
"""

from __future__ import annotations

from typing import Any

from goldfive.drift import (
    classify_refusal,
    classify_stop_reason,
    classify_tool_error,
)
from goldfive.judges.base import Judge, JudgeContext, JudgeVerdict
from goldfive.types import DriftEvent


def _verdict_from_drift(drift: DriftEvent | None) -> JudgeVerdict:
    """Project a :class:`DriftEvent` onto a drift-flavoured verdict.

    Returns an empty-default :class:`JudgeVerdict` when ``drift`` is
    ``None`` — the steerer skips :class:`JudgementEmitted` emission
    for empty verdicts (no signal == no event).
    """
    if drift is None:
        return JudgeVerdict()
    return JudgeVerdict(
        drift_emitted=True,
        drift_kind=str(drift.kind),
        severity=str(drift.severity),
        detail=drift.detail or "",
    )


class ReasoningDriftJudge:
    """Wraps the LLM-as-a-judge reasoning-drift classifier.

    The actual classifier lives in
    :mod:`goldfive.drift.reasoning_judge` and is driven by the
    steerer's background-judge orchestration in
    :mod:`goldfive.drift_observer`. This wrapper exists so the
    operator-facing surface is uniform — listing
    ``reasoning_drift()`` in :func:`goldfive.wrap(judges=[...])` is
    the opt-in token; the steerer keeps owning the rate-limited /
    fire-and-forget execution because that path has subtle ordering
    contracts (history pinning, late-drift tolerance) that don't
    belong in user code.

    Returns an empty-default verdict from :meth:`evaluate` so the
    public-judge code path stays a no-op for the built-in case — the
    canonical event still flows from the steerer's existing
    emission site. Subclasses that want to short-circuit the
    background-judge path can override :meth:`evaluate` and return
    a populated verdict.
    """

    name: str = "reasoning_drift"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:  # noqa: ARG002
        # The real verdict is produced by the steerer's
        # ``_run_judge_background`` path (see drift_observer.py); this
        # method exists for protocol conformance and for operator
        # opt-in via the judges list.
        return JudgeVerdict()


class LoopingReasoningJudge:
    """Wraps :func:`goldfive.drift.reasoning.detect_looping_reasoning`.

    Runs synchronously on every reasoning observation; the underlying
    detector is cheap (hash-based with an optional embedding fallback).
    """

    name: str = "looping_reasoning"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:
        from goldfive.drift.reasoning import detect_looping_reasoning

        if not ctx.reasoning_text or ctx.session_state is None:
            return JudgeVerdict()
        drift = detect_looping_reasoning(ctx.reasoning_text, ctx.session_state)
        return _verdict_from_drift(drift)


class GoalDriftJudge:
    """Wraps the trajectory-level GOAL_DRIFT classifier.

    The detector is invoked by the steerer's
    ``_maybe_run_goal_drift_on_task_boundary`` path; the wrapper is
    a protocol-conformance shim so operators can list / opt-out via
    the ``judges=`` kwarg.
    """

    name: str = "goal_drift"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:  # noqa: ARG002
        return JudgeVerdict()


class RefusalJudge:
    """Wraps :func:`goldfive.drift.classify_refusal`.

    Emits a drift verdict when the reasoning text or the latest
    transcript entry trips a refusal marker.
    """

    name: str = "refusal"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:
        candidate = ctx.reasoning_text
        if not candidate and ctx.transcript:
            candidate = ctx.transcript[-1]
        drift = classify_refusal(candidate) if candidate else None
        return _verdict_from_drift(drift)


class ToolErrorJudge:
    """Wraps :func:`goldfive.drift.classify_tool_error`.

    Inspects ``ctx.extras["tool_event"]`` for a tool-call result and
    emits a drift verdict when the call failed. Operators that
    install this judge pass the tool event dict via the steerer's
    tool-observation path.
    """

    name: str = "tool_error"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:
        tool_event: Any = ctx.extras.get("tool_event")
        drift = classify_tool_error(tool_event) if tool_event is not None else None
        return _verdict_from_drift(drift)


class StopReasonJudge:
    """Wraps :func:`goldfive.drift.classify_stop_reason`.

    Inspects ``ctx.extras["stop_reason"]`` for a model finish-reason
    value and emits a CONTEXT_PRESSURE drift when it matches the
    known truncation markers.
    """

    name: str = "stop_reason"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:
        reason = ctx.extras.get("stop_reason")
        drift = classify_stop_reason(reason) if reason is not None else None
        return _verdict_from_drift(drift)


class LoopingToolJudge:
    """Wraps the tool-loop detector tracker.

    The actual matching state is owned by
    :class:`~goldfive.drift.tool_loops.ToolLoopTracker` on the
    steerer; this shim is a protocol-conformance stub so operators
    can list / opt-out from the ``judges=`` kwarg.
    """

    name: str = "looping_tool"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:  # noqa: ARG002
        return JudgeVerdict()


# ---------------------------------------------------------------------------
# Factory functions (the public opt-in surface)
# ---------------------------------------------------------------------------
#
# Each factory returns a FRESH instance so callers can construct
# multiple :func:`goldfive.wrap` invocations without judge-state
# bleed between them. Factories take no arguments today; future
# tuning surfaces (e.g. per-judge thresholds) can be added as
# keyword arguments without breaking callers.


def reasoning_drift() -> Judge:
    """Return a :class:`ReasoningDriftJudge` instance."""
    return ReasoningDriftJudge()


def looping_reasoning() -> Judge:
    """Return a :class:`LoopingReasoningJudge` instance."""
    return LoopingReasoningJudge()


def goal_drift() -> Judge:
    """Return a :class:`GoalDriftJudge` instance."""
    return GoalDriftJudge()


def refusal() -> Judge:
    """Return a :class:`RefusalJudge` instance."""
    return RefusalJudge()


def tool_error() -> Judge:
    """Return a :class:`ToolErrorJudge` instance."""
    return ToolErrorJudge()


def stop_reason() -> Judge:
    """Return a :class:`StopReasonJudge` instance."""
    return StopReasonJudge()


def looping_tool() -> Judge:
    """Return a :class:`LoopingToolJudge` instance."""
    return LoopingToolJudge()


def default_judges() -> list[Judge]:
    """Return the goldfive default judge set.

    Mirrors the detectors the pre-judges code path armed by default.
    :func:`goldfive.wrap` installs this set when the caller does not
    supply an explicit ``judges=`` list.
    """
    return [
        reasoning_drift(),
        looping_reasoning(),
        goal_drift(),
        refusal(),
        tool_error(),
        stop_reason(),
        looping_tool(),
    ]


__all__ = [
    "GoalDriftJudge",
    "LoopingReasoningJudge",
    "LoopingToolJudge",
    "ReasoningDriftJudge",
    "RefusalJudge",
    "StopReasonJudge",
    "ToolErrorJudge",
    "default_judges",
    "goal_drift",
    "looping_reasoning",
    "looping_tool",
    "reasoning_drift",
    "refusal",
    "stop_reason",
    "tool_error",
]
