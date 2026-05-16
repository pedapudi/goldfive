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

import asyncio
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


def test_judge_verdict_accepts_drift_enums() -> None:
    """``JudgeVerdict`` accepts the typed ``DriftKind`` / ``DriftSeverity``
    enums (the preferred form) and stores them as enum members."""
    v = JudgeVerdict(
        drift_emitted=True,
        drift_kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.CRITICAL,
    )
    assert v.drift_kind is DriftKind.TOOL_ERROR
    assert v.severity is DriftSeverity.CRITICAL
    assert isinstance(v.drift_kind, DriftKind)
    assert isinstance(v.severity, DriftSeverity)


def test_judge_verdict_normalizes_legacy_string_form() -> None:
    """A legacy lowercase-string ``drift_kind`` / ``severity`` is
    normalised to the matching enum at construction, so consumers always
    read a real enum — yet string equality still holds (StrEnum)."""
    v = JudgeVerdict(
        drift_emitted=True,
        drift_kind="tool_error",
        severity="critical",
    )
    # Normalised to the enum.
    assert v.drift_kind is DriftKind.TOOL_ERROR
    assert v.severity is DriftSeverity.CRITICAL
    # Back-compat: still compares equal to the legacy lowercase string.
    assert v.drift_kind == "tool_error"
    assert v.severity == "critical"
    assert str(v.drift_kind) == "tool_error"
    assert str(v.severity) == "critical"


def test_judge_verdict_empty_default_leaves_drift_fields_blank() -> None:
    """The empty-default verdict keeps ``drift_kind`` / ``severity`` as
    empty strings — no spurious enum coercion of the no-drift sentinel."""
    v = JudgeVerdict()
    assert v.drift_kind == ""
    assert v.severity == ""
    assert not isinstance(v.drift_kind, DriftKind)
    assert not isinstance(v.severity, DriftSeverity)


def test_judge_verdict_unrecognised_string_passes_through() -> None:
    """An unrecognised custom ``drift_kind`` / ``severity`` string is
    left untouched so a forward-compatible / domain-specific judge is
    not broken by normalisation."""
    v = JudgeVerdict(
        drift_emitted=True,
        drift_kind="some_future_kind",
        severity="unusual",
    )
    assert v.drift_kind == "some_future_kind"
    assert v.severity == "unusual"
    assert not isinstance(v.drift_kind, DriftKind)


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


def test_builtin_judge_enum_values_match_judge_names() -> None:
    """Every :class:`BuiltinJudge` member's value equals the wire ``name``
    of the judge its factory builds — the enum is a typed alias for the
    historical magic-string judge names."""
    from goldfive.builtin_judges import BuiltinJudge

    enum_values = {str(j) for j in BuiltinJudge}
    factory_names = {j.name for j in builtin_judges.default_judges()}
    assert enum_values == factory_names
    # StrEnum: a member compares equal to its wire name.
    assert BuiltinJudge.TOOL_ERROR == "tool_error"


def test_builtin_judge_names_derives_from_enum() -> None:
    """``BUILTIN_JUDGE_NAMES`` stays a ``frozenset[str]`` (back-compat)
    and is exactly the set of :class:`BuiltinJudge` values."""
    from goldfive.builtin_judges import BuiltinJudge
    from goldfive.judges.builtins import BUILTIN_JUDGE_NAMES

    assert BUILTIN_JUDGE_NAMES == {str(j) for j in BuiltinJudge}
    # Membership works with either a plain string or a BuiltinJudge.
    assert "reasoning_drift" in BUILTIN_JUDGE_NAMES
    assert BuiltinJudge.REFUSAL in BUILTIN_JUDGE_NAMES


def test_default_judges_disable_by_enum() -> None:
    """``default_judges(disable=[BuiltinJudge...])`` drops exactly the
    named built-ins and keeps the rest."""
    from goldfive.builtin_judges import BuiltinJudge

    full = builtin_judges.default_judges()
    filtered = builtin_judges.default_judges(
        disable=[BuiltinJudge.TOOL_ERROR, BuiltinJudge.GOAL_DRIFT]
    )
    names = {j.name for j in filtered}
    assert "tool_error" not in names
    assert "goal_drift" not in names
    assert "reasoning_drift" in names
    assert len(filtered) == len(full) - 2


def test_default_judges_disable_accepts_legacy_strings() -> None:
    """``disable=`` also accepts the legacy wire-name strings."""
    filtered = builtin_judges.default_judges(disable=["refusal"])
    assert "refusal" not in {j.name for j in filtered}


def test_default_judges_disable_ignores_unknown_entries() -> None:
    """An unrecognised ``disable=`` entry is ignored (forward-compatible)."""
    full = builtin_judges.default_judges()
    filtered = builtin_judges.default_judges(disable=["not_a_real_judge"])
    assert len(filtered) == len(full)


def test_default_judges_no_disable_returns_full_set() -> None:
    """``default_judges()`` / ``default_judges(disable=None)`` return the
    full set unchanged — back-compat with the no-arg call."""
    assert len(builtin_judges.default_judges()) == 7
    assert len(builtin_judges.default_judges(disable=None)) == 7


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


async def test_evaluate_judges_drift_flavour_accepts_enum_typed_verdict() -> None:
    """A drift-flavoured verdict built with the typed ``DriftKind`` /
    ``DriftSeverity`` enums (rather than legacy strings) is consumed
    correctly: the ``JudgementEmitted`` envelope carries the string
    value and the forwarded ``DriftEvent`` carries the real enums.

    Pins the steerer-side half of the enum-typed judge API — the
    consumer that reads ``JudgeVerdict`` and fires ``DriftDetected``
    handles both the enum form and the legacy-string form.
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    # Built entirely from enums — no magic strings.
    drift_verdict = JudgeVerdict(
        drift_emitted=True,
        drift_kind=DriftKind.AGENT_REFUSAL,
        severity=DriftSeverity.WARNING,
        detail="enum-typed refusal verdict",
    )
    captured_drifts: list[DriftEvent] = []

    async def _capture(drift: DriftEvent, session: Session) -> None:  # noqa: ARG001
        captured_drifts.append(drift)

    steerer.drift.handle_drift = _capture  # type: ignore[method-assign]
    steerer.set_judges([_StubJudge("enum_refusal_j", drift_verdict)])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    judgements = _judgement_events(sink)
    assert len(judgements) == 1
    assert judgements[0].verdict_kind == "drift"
    # Proto carries the plain string value regardless of input shape.
    assert judgements[0].drift_kind == str(DriftKind.AGENT_REFUSAL)
    assert judgements[0].severity == str(DriftSeverity.WARNING)
    # The forwarded DriftEvent carries the real enum members.
    assert len(captured_drifts) == 1
    assert captured_drifts[0].kind is DriftKind.AGENT_REFUSAL
    assert captured_drifts[0].severity is DriftSeverity.WARNING


async def test_evaluate_judges_drift_flavour_with_unrecognised_kind_emits_only_judgement() -> None:
    """A drift verdict whose ``drift_kind`` is an unrecognised custom
    string (not a :class:`DriftKind` member) still emits a
    ``JudgementEmitted`` envelope but forwards NO ``DriftEvent``.

    ``__post_init__`` leaves an unrecognised string untouched (a
    forward-compatible / domain-specific judge is not broken), so the
    steerer's ``_drift_from_judge_verdict`` cannot project it onto a
    :class:`DriftKind`. It returns ``None`` — the legacy refine
    machinery is skipped, but the typed judge signal still reaches the
    wire via ``JudgementEmitted``.
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    drift_verdict = JudgeVerdict(
        drift_emitted=True,
        drift_kind="domain_specific_signal",  # not a DriftKind member
        severity="critical",
        detail="custom judge with a bespoke drift kind",
    )
    # The unrecognised string is left as-is — not coerced to an enum.
    assert drift_verdict.drift_kind == "domain_specific_signal"
    assert not isinstance(drift_verdict.drift_kind, DriftKind)
    captured_drifts: list[DriftEvent] = []

    async def _capture(drift: DriftEvent, session: Session) -> None:  # noqa: ARG001
        captured_drifts.append(drift)

    steerer.drift.handle_drift = _capture  # type: ignore[method-assign]
    steerer.set_judges([_StubJudge("bespoke_j", drift_verdict)])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    # JudgementEmitted still reaches the wire, carrying the raw string.
    judgements = _judgement_events(sink)
    assert len(judgements) == 1
    assert judgements[0].verdict_kind == "drift"
    assert judgements[0].drift_kind == "domain_specific_signal"
    # ...but no DriftEvent is projected from the unrecognised kind.
    assert captured_drifts == []


async def test_custom_drift_judge_emits_exactly_one_judgement() -> None:
    """A custom drift-flavoured judge produces exactly ONE
    ``JudgementEmitted`` — keyed on the judge's real ``name``.

    The drift verdict also routes through ``_emit_drift_detected``
    (the back-compat ``DriftDetected`` fan-out). That path has its own
    paired ``JudgementEmitted`` emission for legacy detectors; it MUST
    be suppressed here so the custom judge does not land two events
    (one keyed on ``judge_name``, one on the drift kind) and break the
    "join on judge_name" telemetry contract.
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges(builtin_judges.default_judges())  # paired-emit armed
    drift_verdict = JudgeVerdict(
        drift_emitted=True,
        drift_kind=str(DriftKind.AGENT_REFUSAL),
        severity=str(DriftSeverity.WARNING),
        detail="custom refusal grader matched",
    )

    # Route handle_drift straight to the real _emit_drift_detected so
    # the paired-emission path runs without the full ladder machinery.
    async def _emit_only(drift: DriftEvent, session: Session) -> None:
        await steerer.drift._emit_drift_detected(session, drift)

    steerer.drift.handle_drift = _emit_only  # type: ignore[method-assign]
    custom = _StubJudge("my_grader", drift_verdict)
    await steerer.evaluate_judges(
        JudgeContext(), session=_make_session(), judges=[custom]
    )
    judgements = _judgement_events(sink)
    assert len(judgements) == 1, "exactly one JudgementEmitted for the drift judge"
    assert judgements[0].judge_name == "my_grader"
    assert judgements[0].verdict_kind == "drift"
    # The legacy DriftDetected envelope still fires (back-compat).
    assert len(_drift_events(sink)) == 1


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


async def test_wrap_disable_judges_drops_named_builtins() -> None:
    """``goldfive.wrap(disable_judges=[BuiltinJudge...])`` installs the
    default set minus the named built-ins."""
    from goldfive.builtin_judges import BuiltinJudge

    async def _agent(task: Any, session: Any, tools: Any) -> Any:  # noqa: ARG001
        from goldfive.results import InvocationResult

        return InvocationResult(output="ok")

    runner = goldfive.wrap(
        _agent, sinks=[], disable_judges=[BuiltinJudge.TOOL_ERROR]
    )
    steerer = runner.steerer  # type: ignore[attr-defined]
    names = {j.name for j in steerer.get_judges()}
    assert "tool_error" not in names
    assert "reasoning_drift" in names
    assert len(names) == 6
    await runner.close()


async def test_wrap_disable_judges_rejects_combination_with_judges() -> None:
    """Passing both ``judges=`` and ``disable_judges=`` is a ``TypeError``
    — an explicit ``judges=`` list already names the exact set."""
    from goldfive.builtin_judges import BuiltinJudge

    async def _agent(task: Any, session: Any, tools: Any) -> Any:  # noqa: ARG001
        from goldfive.results import InvocationResult

        return InvocationResult(output="ok")

    with pytest.raises(TypeError, match="disable_judges"):
        goldfive.wrap(
            _agent,
            sinks=[],
            judges=[],
            disable_judges=[BuiltinJudge.REFUSAL],
        )


# ---------------------------------------------------------------------------
# observe_reasoning auto-wires custom judges (goldfive#437)
# ---------------------------------------------------------------------------


async def _drain_background(steerer: Any) -> None:
    """Await background custom-judge tasks scheduled by observe_reasoning."""
    pending = list(getattr(steerer, "_background_drifts", set())) + list(
        getattr(steerer, "_background_judges", set())
    )
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def test_observe_reasoning_runs_custom_judge() -> None:
    """A custom (non-built-in) judge is invoked on every reasoning
    observation and its verdict reaches the sink as ``JudgementEmitted``.

    Pins the headline goldfive#437 contract: ``goldfive.wrap(judges=[
    MyRubricJudge()])`` actually runs the judge during a run rather
    than leaving ``evaluate_judges`` as dead code.
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    custom = _StubJudge("my_rubric", JudgeVerdict(rubric_score=0.7))
    steerer.set_judges([custom])
    session = _make_session()
    await steerer.drift.observe_reasoning("a thought", session=session)
    await _drain_background(steerer)
    hits = _judgement_events(sink)
    assert len(hits) == 1
    assert hits[0].judge_name == "my_rubric"
    assert hits[0].verdict_kind == "rubric"
    assert custom.calls, "custom judge must have been invoked"
    # The JudgeContext carried the reasoning text.
    assert custom.calls[0].reasoning_text == "a thought"


async def test_observe_reasoning_skips_builtin_judges() -> None:
    """Built-in judges are NOT re-run by the observe_reasoning auto-wire
    path — their drift verdicts ride the legacy detector path's paired
    emission, so re-running them here would double-fire.

    Installs the full default set; a benign reasoning block produces
    no ``JudgementEmitted`` from the custom-judge path (all installed
    judges are built-ins).
    """
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges(builtin_judges.default_judges())
    session = _make_session()
    await steerer.drift.observe_reasoning("a benign thought", session=session)
    await _drain_background(steerer)
    assert _judgement_events(sink) == []


async def test_observe_reasoning_custom_judge_exception_does_not_crash() -> None:
    """A custom judge that raises is swallowed; observe_reasoning still
    returns normally and the reasoning history is still appended."""
    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.set_judges([_RaisingJudge()])
    session = _make_session()
    await steerer.drift.observe_reasoning("a thought", session=session)
    await _drain_background(steerer)
    assert _judgement_events(sink) == []
    assert session.reasoning_history == ["a thought"]


async def test_evaluate_judges_times_out_slow_judge() -> None:
    """A judge whose ``evaluate`` hangs past the budget is cancelled and
    treated as no signal; other judges still run."""

    class _SlowJudge:
        name = "slow"

        async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:  # noqa: ARG002
            await asyncio.sleep(3600)
            return JudgeVerdict(boolean_result=True)

    sink = ListSink()
    steerer = goldfive.DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())  # type: ignore[arg-type]
    steerer.JUDGE_EVALUATE_TIMEOUT_S = 0.05  # type: ignore[misc]
    good = _StubJudge("good", JudgeVerdict(boolean_result=False))
    steerer.set_judges([_SlowJudge(), good])
    await steerer.evaluate_judges(JudgeContext(), session=_make_session())
    hits = _judgement_events(sink)
    assert len(hits) == 1
    assert hits[0].judge_name == "good"
    assert good.calls, "good judge must run after the slow one is cancelled"
