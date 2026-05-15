"""Tests for the pluggable-judges surface (goldfive#437).

Covers:

* :class:`~goldfive.judges.JudgeContext` and :class:`JudgeVerdict` are
  frozen dataclasses with the documented default-empty values.
* The protocol-conformance shape: any object that exposes
  ``name: str`` plus an async ``evaluate`` method is treated as a
  :class:`Judge`.
* :func:`goldfive.builtin_judges.default_judges` returns the
  documented built-in set and each factory returns a fresh instance.
* :meth:`DefaultSteerer.evaluate_judges` calls every installed
  judge, emits :class:`JudgementEmitted` for every populated
  verdict, and skips emission for empty-default verdicts.
* :meth:`DefaultSteerer.evaluate_judges` picks ``verdict_kind`` from
  the first populated flavour (drift > rubric > boolean > numeric).
* Drift-flavoured verdicts ALSO emit ``DriftDetected`` (back-compat).
* A judge whose ``evaluate`` raises is swallowed and the next judge
  still runs.
* :func:`goldfive.wrap(judges=[...])` forwards the list onto the
  default steerer.
* Built-in drift detectors emit BOTH ``DriftDetected`` (legacy) AND
  a paired ``JudgementEmitted`` (the new judge-centric surface)
  through the existing ``_emit_drift_detected`` path.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

import goldfive  # noqa: E402
from goldfive import builtin_judges  # noqa: E402
from goldfive.judges import Judge, JudgeContext, JudgeVerdict  # noqa: E402
from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Session  # noqa: E402

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _NullPlanner:
    """Minimal Planner stub for steerer.bind() in unit tests."""

    async def generate(self, **kwargs: Any) -> Any:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> Any:  # noqa: ARG002
        return None


class ListSink:
    """Bare-bones EventSink collecting proto envelopes into a list."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


def _judgement_events(sink: ListSink) -> list[Any]:
    return [
        e.judgement_emitted
        for e in sink.events
        if e.WhichOneof("payload") == "judgement_emitted"
    ]


def _drift_events(sink: ListSink) -> list[Any]:
    return [
        e.drift_detected
        for e in sink.events
        if e.WhichOneof("payload") == "drift_detected"
    ]


def _make_session() -> Session:
    """Minimal Session suitable for steerer.evaluate_judges() emission."""
    return Session(run_id="run-test")


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


def test_judge_context_is_frozen_with_default_empty_values() -> None:
    ctx = JudgeContext()
    assert ctx.reasoning_text == ""
    assert ctx.plan is None
    assert ctx.transcript == ()
    assert ctx.session_state is None
    assert ctx.current_task_id == ""
    assert ctx.current_agent_id == ""
    assert ctx.extras == {}
    with pytest.raises(dataclasses_error()):
        ctx.reasoning_text = "mutated"  # type: ignore[misc]


def test_judge_verdict_is_frozen_with_default_empty_values() -> None:
    v = JudgeVerdict()
    assert v.drift_emitted is False
    assert v.drift_kind == ""
    assert v.severity == ""
    assert v.rubric_score is None
    assert v.rubric_dimensions == {}
    assert v.boolean_result is None
    assert v.numeric_value is None
    assert v.metric_name == ""
    assert v.detail == ""
    with pytest.raises(dataclasses_error()):
        v.detail = "mutated"  # type: ignore[misc]


def dataclasses_error() -> type[Exception]:
    """The exception raised by frozen dataclasses on attribute assignment.

    On stdlib dataclasses this is :class:`dataclasses.FrozenInstanceError`,
    a subclass of :class:`AttributeError`. Returning the broader parent
    keeps the test happy across CPython versions that don't (yet) expose
    the subclass under the same public name.
    """
    import dataclasses as _dc

    return _dc.FrozenInstanceError


def test_judge_protocol_accepts_duck_typed_implementation() -> None:
    class _MyJudge:
        name = "my_judge"

        async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:  # noqa: ARG002
            return JudgeVerdict()

    j = _MyJudge()
    assert isinstance(j, Judge)
    assert j.name == "my_judge"


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------


def test_default_judges_returns_documented_set() -> None:
    names = [j.name for j in builtin_judges.default_judges()]
    assert "reasoning_drift" in names
    assert "goal_drift" in names
    assert "refusal" in names
    assert "tool_error" in names
    assert "stop_reason" in names
    assert "looping_reasoning" in names
    assert "looping_tool" in names


def test_factories_return_fresh_instances() -> None:
    a = builtin_judges.reasoning_drift()
    b = builtin_judges.reasoning_drift()
    assert a is not b


async def test_refusal_judge_emits_drift_verdict_on_match() -> None:
    judge = builtin_judges.refusal()
    ctx = JudgeContext(reasoning_text="I cannot help with that request")
    verdict = await judge.evaluate(ctx)
    assert verdict.drift_emitted is True
    assert verdict.drift_kind  # populated
    assert verdict.severity   # populated


async def test_refusal_judge_emits_empty_verdict_on_no_match() -> None:
    judge = builtin_judges.refusal()
    ctx = JudgeContext(reasoning_text="happy to help")
    verdict = await judge.evaluate(ctx)
    assert verdict.drift_emitted is False
    assert verdict.drift_kind == ""


async def test_tool_error_judge_reads_extras_tool_event() -> None:
    judge = builtin_judges.tool_error()
    ctx = JudgeContext(extras={"tool_event": {"error": "boom", "tool": "search"}})
    verdict = await judge.evaluate(ctx)
    assert verdict.drift_emitted is True
    assert verdict.drift_kind == str(DriftKind.TOOL_ERROR)


# ---------------------------------------------------------------------------
# evaluate_judges() — emission contract
# ---------------------------------------------------------------------------


class _StubJudge:
    """Returns a hand-built verdict for assertion purposes."""

    def __init__(self, name: str, verdict: JudgeVerdict) -> None:
        self.name = name
        self._verdict = verdict
        self.calls: list[JudgeContext] = []

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:
        self.calls.append(ctx)
        return self._verdict


class _RaisingJudge:
    name = "raising_judge"

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:  # noqa: ARG002
        raise RuntimeError("kaboom")


async def test_evaluate_judges_emits_judgement_for_populated_verdict() -> None:
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges([_StubJudge("rubric_j", JudgeVerdict(rubric_score=0.83))])
    session = _make_session()
    await steerer.evaluate_judges(JudgeContext(), session=session)
    hits = _judgement_events(sink)
    assert len(hits) == 1
    assert hits[0].judge_name == "rubric_j"
    assert hits[0].verdict_kind == "rubric"
    assert hits[0].rubric_score == pytest.approx(0.83)


async def test_evaluate_judges_skips_empty_default_verdict() -> None:
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges([_StubJudge("silent", JudgeVerdict())])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    assert _judgement_events(sink) == []


async def test_evaluate_judges_picks_first_populated_flavour() -> None:
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    # Boolean verdict — should be classified "boolean" (drift / rubric empty).
    steerer.set_judges([_StubJudge("bool_j", JudgeVerdict(boolean_result=True))])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    hits = _judgement_events(sink)
    assert len(hits) == 1
    assert hits[0].verdict_kind == "boolean"
    assert hits[0].boolean_result is True


async def test_evaluate_judges_numeric_flavour_includes_metric_name() -> None:
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges(
        [
            _StubJudge(
                "cost_j",
                JudgeVerdict(numeric_value=0.42, metric_name="cost_usd"),
            )
        ]
    )
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    hits = _judgement_events(sink)
    assert len(hits) == 1
    assert hits[0].verdict_kind == "numeric"
    assert hits[0].metric_name == "cost_usd"
    assert hits[0].numeric_value == pytest.approx(0.42)


async def test_evaluate_judges_drift_flavour_emits_judgement_with_kind() -> None:
    """A drift-flavoured verdict produces a ``JudgementEmitted`` with
    ``verdict_kind = "drift"`` and the drift_kind / severity fields
    populated from the verdict.

    The paired :class:`DriftDetected` envelope is covered by
    :func:`test_existing_drift_emit_also_fires_judgement_emitted` —
    that exercises the legacy ``_emit_drift_detected`` path directly
    rather than the full ``handle_drift`` dispatch (which needs a
    bound planner / adapter to walk the intervention ladder).
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    drift_verdict = JudgeVerdict(
        drift_emitted=True,
        drift_kind=str(DriftKind.AGENT_REFUSAL),
        severity=str(DriftSeverity.WARNING),
        detail="refusal marker matched",
    )
    # Stub handle_drift so the test doesn't depend on the full
    # ladder-dispatch machinery (planner.refine, adapter cancel
    # tagging). The forwarded drift is captured and asserted.
    captured_drifts: list[DriftEvent] = []

    async def _capture(drift: DriftEvent, session: Session) -> None:  # noqa: ARG001
        captured_drifts.append(drift)

    steerer.drift.handle_drift = _capture  # type: ignore[method-assign]
    steerer.set_judges([_StubJudge("refusal_j", drift_verdict)])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    judgements = _judgement_events(sink)
    assert len(judgements) == 1
    assert judgements[0].verdict_kind == "drift"
    assert judgements[0].judge_name == "refusal_j"
    assert judgements[0].drift_kind == str(DriftKind.AGENT_REFUSAL)
    # And the verdict was forwarded to the legacy handler for the
    # DriftDetected back-compat fan-out.
    assert len(captured_drifts) == 1
    assert captured_drifts[0].kind is DriftKind.AGENT_REFUSAL
    assert captured_drifts[0].severity is DriftSeverity.WARNING


async def test_evaluate_judges_swallows_judge_exception() -> None:
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    good = _StubJudge("good", JudgeVerdict(boolean_result=False))
    steerer.set_judges([_RaisingJudge(), good])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    # The raising judge produced no event; the good one still ran.
    hits = _judgement_events(sink)
    assert len(hits) == 1
    assert hits[0].judge_name == "good"
    assert good.calls, "good judge must have been invoked after the failing one"


async def test_evaluate_judges_no_judges_emits_nothing() -> None:
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges([])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    assert _judgement_events(sink) == []


# ---------------------------------------------------------------------------
# Legacy drift detectors emit paired JudgementEmitted (back-compat)
# ---------------------------------------------------------------------------


async def test_existing_drift_emit_also_fires_judgement_emitted() -> None:
    """The legacy ``_emit_drift_detected`` path emits a paired
    ``JudgementEmitted`` envelope (goldfive#437) keyed on the drift
    kind so downstream consumers of the judge-centric surface see
    every built-in detector verdict alongside operator-supplied
    judge verdicts.

    The paired emission is gated on the steerer having a non-empty
    installed judges list — :func:`goldfive.wrap` installs the
    default set, so the paired-emit path is the wrap-time default.
    Bare ``DefaultSteerer()`` constructions without ``set_judges``
    preserve the legacy single-event contract (see
    ``test_bare_steerer_without_judges_preserves_single_drift_emit``).
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges(builtin_judges.default_judges())
    session = _make_session()
    drift = DriftEvent(
        kind=DriftKind.AGENT_REFUSAL,
        severity=DriftSeverity.WARNING,
        detail="refusal marker matched 'i cannot'",
    )
    await steerer.drift._emit_drift_detected(session, drift)
    drifts = _drift_events(sink)
    judgements = _judgement_events(sink)
    assert len(drifts) == 1, "DriftDetected wire envelope must still fire"
    assert len(judgements) == 1, "JudgementEmitted paired envelope must fire"
    assert judgements[0].verdict_kind == "drift"
    assert judgements[0].drift_kind == str(DriftKind.AGENT_REFUSAL)
    assert judgements[0].severity == str(DriftSeverity.WARNING)


async def test_bare_steerer_without_judges_preserves_single_drift_emit() -> None:
    """A ``DefaultSteerer`` that never had judges installed continues
    to fire only ``DriftDetected`` (no paired ``JudgementEmitted``)
    on the legacy ``_emit_drift_detected`` path.

    Pins the back-compat contract relied on by the existing drift /
    lifecycle test corpus (``assert len(sink.events) == 1`` after
    a single drift).
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    session = _make_session()
    drift = DriftEvent(
        kind=DriftKind.AGENT_REFUSAL,
        severity=DriftSeverity.WARNING,
        detail="refusal marker matched",
    )
    await steerer.drift._emit_drift_detected(session, drift)
    assert len(_drift_events(sink)) == 1
    assert len(_judgement_events(sink)) == 0


# ---------------------------------------------------------------------------
# goldfive.wrap(judges=) wiring
# ---------------------------------------------------------------------------


async def test_wrap_installs_default_judge_set_when_kwarg_omitted() -> None:
    """``goldfive.wrap`` without an explicit ``judges=`` installs the
    built-in default set on the steerer.
    """
    # Minimal callable adapter; the wrap() machinery only needs an
    # adapter that auto_adapter accepts.
    async def _agent(task: Any, session: Any, tools: Any) -> Any:  # noqa: ARG001
        from goldfive.results import InvocationResult

        return InvocationResult(output="ok")

    runner = goldfive.wrap(_agent, sinks=[])
    steerer = runner.steerer  # type: ignore[attr-defined]
    judges = steerer.get_judges()
    names = {j.name for j in judges}
    assert "reasoning_drift" in names
    assert "refusal" in names
    await runner.close()


async def test_wrap_judges_kwarg_overrides_default_set() -> None:
    async def _agent(task: Any, session: Any, tools: Any) -> Any:  # noqa: ARG001
        from goldfive.results import InvocationResult

        return InvocationResult(output="ok")

    custom = _StubJudge("custom_only", JudgeVerdict(rubric_score=1.0))
    runner = goldfive.wrap(_agent, sinks=[], judges=[custom])
    steerer = runner.steerer  # type: ignore[attr-defined]
    judges = steerer.get_judges()
    assert [j.name for j in judges] == ["custom_only"]
    await runner.close()


async def test_wrap_judges_empty_list_disables_judges() -> None:
    async def _agent(task: Any, session: Any, tools: Any) -> Any:  # noqa: ARG001
        from goldfive.results import InvocationResult

        return InvocationResult(output="ok")

    runner = goldfive.wrap(_agent, sinks=[], judges=[])
    steerer = runner.steerer  # type: ignore[attr-defined]
    assert steerer.get_judges() == []
    await runner.close()
