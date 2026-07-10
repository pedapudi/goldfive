"""Unit tests for ``ReasoningJudgeInvoked`` emission from
:func:`goldfive.drift.reasoning_judge.classify_reasoning_drift`.

Covers (judge-observability event):

* ``sink`` kwarg is optional — omitting it preserves the pre-existing
  sinkless contract (no event emission, return value unchanged).
* When ``sink`` is provided, a ``ReasoningJudgeInvoked`` event is
  emitted on EVERY invocation regardless of verdict: on-task,
  off-task (every severity), malformed JSON, missing key, and
  ``call_llm`` raising all produce exactly one event per call.
* ``reasoning_input`` is truncated at
  ``REASONING_JUDGE_MAX_REASONING_INPUT_CHARS`` with the
  ``" … [truncated]"`` suffix; ``raw_response`` at
  ``REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS`` similarly.
* Parsed-verdict fields (``on_task`` / ``severity`` / ``reason``)
  mirror the judge's JSON when the response is well-formed.
* A broken sink (``emit`` raises) does NOT break the run: the return
  value still reflects the judge's verdict.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import reasoning_judge as rjudge  # noqa: E402
from goldfive.types import DriftKind, DriftSeverity, Goal, Task  # noqa: E402


class ListSink:
    """Bare-bones EventSink collecting proto envelopes into a list."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class RaisingSink:
    """Sink whose ``emit`` always raises, to exercise absorb-on-failure."""

    def __init__(self) -> None:
        self.calls = 0

    async def emit(self, event_pb: Any) -> None:  # noqa: ARG002
        self.calls += 1
        raise RuntimeError("sink is broken")

    async def close(self) -> None:
        return None


def _stub_call_llm(responses: list[Any]):
    """Async ``CallLLM``-shaped stub popping responses in order."""
    queue = list(responses)
    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        if isinstance(resp, Exception):
            raise resp
        return str(resp)

    _call_llm.calls = calls  # type: ignore[attr-defined]
    return _call_llm


def _task() -> Task:
    return Task(id="t1", title="Research solar panels", description="Find specs")


def _goals() -> list[Goal]:
    return [Goal(id="g1", summary="Publish a memo on solar panels")]


def _one_judge_event(sink: ListSink) -> Any:
    """Assert exactly one ReasoningJudgeInvoked landed and return it."""
    hits = [
        e for e in sink.events
        if e.WhichOneof("payload") == "reasoning_judge_invoked"
    ]
    assert len(hits) == 1, [e.WhichOneof("payload") for e in sink.events]
    return hits[0].reasoning_judge_invoked


# ---------------------------------------------------------------------------
# Sink-less back-compat
# ---------------------------------------------------------------------------


async def test_sinkless_call_preserves_legacy_return_value() -> None:
    """Omitting ``sink`` keeps the pre-existing (no emission) contract."""
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "drifted"}]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert drift is not None
    assert drift.kind is DriftKind.OFF_TOPIC
    assert drift.severity is DriftSeverity.WARNING


# ---------------------------------------------------------------------------
# Emission on every invocation regardless of verdict
# ---------------------------------------------------------------------------


async def test_on_task_true_emits_judge_invoked() -> None:
    """On-task verdicts must ALSO fire ReasoningJudgeInvoked (goldfive gap)."""
    sink = ListSink()
    call_llm = _stub_call_llm([{"on_task": True}])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="working through the spec sheet",
        task=_task(),
        goals=_goals(),
        model="judge-model",
        call_llm=call_llm,
        current_task_id="t1",
        current_agent_id="researcher",
        sink=sink,
        run_id="run-123",
    )
    assert drift is None
    payload = _one_judge_event(sink)
    assert payload.on_task is True
    assert payload.severity == ""  # no severity on on-task
    assert payload.task_id == "t1"
    assert payload.subject_agent_id == "researcher"
    assert payload.model == "judge-model"
    assert payload.run_id == "run-123"
    assert payload.reasoning_input == "working through the spec sheet"


async def test_off_task_drift_emits_judge_invoked_with_parsed_verdict() -> None:
    """Drift verdicts carry severity + reason on the observability event."""
    sink = ListSink()
    reason = "drifted to raccoons"
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "critical", "reason": reason}]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="raccoons have masks but this is about solar panels",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
        current_task_id="t1",
        current_agent_id="researcher",
        sink=sink,
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.CRITICAL
    payload = _one_judge_event(sink)
    assert payload.on_task is False
    assert payload.severity == "critical"
    assert payload.reason == reason


async def test_malformed_json_still_emits_judge_invoked() -> None:
    """Parse failure path must still fire the observability event."""
    sink = ListSink()
    call_llm = _stub_call_llm(["not json at all, sorry"])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    assert drift is None
    payload = _one_judge_event(sink)
    # Treated as on-task (no drift produced) because the judge didn't
    # return a usable off-task verdict — the timeline still shows the
    # call happened with this raw response.
    assert payload.on_task is True
    assert "not json" in payload.raw_response


async def test_missing_on_task_field_still_emits_judge_invoked() -> None:
    """Dict without a bool ``on_task`` → event fires; no drift returned."""
    sink = ListSink()
    call_llm = _stub_call_llm([{"verdict": "unclear"}])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    assert drift is None
    payload = _one_judge_event(sink)
    assert payload.on_task is True  # no off-task verdict parsed
    assert '"verdict"' in payload.raw_response or "verdict" in payload.raw_response


async def test_call_llm_raises_still_emits_judge_invoked() -> None:
    """Exception in call_llm path still produces an observability event."""
    sink = ListSink()

    async def _raises(system: str, user: str, model: str) -> str:  # noqa: ARG001
        raise RuntimeError("boom")

    drift = await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=_raises,
        sink=sink,
    )
    assert drift is None
    payload = _one_judge_event(sink)
    assert payload.on_task is True  # no verdict parseable
    # raw_response carries a synthetic placeholder describing the failure
    assert "call_llm raised" in payload.raw_response


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


async def test_long_reasoning_input_is_truncated_on_event() -> None:
    """Huge reasoning block → event field capped with " … [truncated]" suffix."""
    sink = ListSink()
    call_llm = _stub_call_llm([{"on_task": True}])
    big = "x" * (rjudge.REASONING_JUDGE_MAX_REASONING_INPUT_CHARS + 500)
    await rjudge.classify_reasoning_drift(
        reasoning=big,
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    payload = _one_judge_event(sink)
    assert payload.reasoning_input.endswith(" … [truncated]")
    # The truncation cap is the configured limit + suffix length, nothing more.
    assert (
        len(payload.reasoning_input)
        == rjudge.REASONING_JUDGE_MAX_REASONING_INPUT_CHARS
        + len(" … [truncated]")
    )


async def test_long_raw_response_is_truncated_on_event() -> None:
    """Chatty judge response → event field capped with truncation suffix."""
    sink = ListSink()
    huge_response = "z" * (rjudge.REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS + 500)
    call_llm = _stub_call_llm([huge_response])
    await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    payload = _one_judge_event(sink)
    assert payload.raw_response.endswith(" … [truncated]")
    assert (
        len(payload.raw_response)
        == rjudge.REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS
        + len(" … [truncated]")
    )


# ---------------------------------------------------------------------------
# Absorb-on-failure
# ---------------------------------------------------------------------------


async def test_broken_sink_does_not_break_run() -> None:
    """Sink.emit raising must be absorbed; judge return value preserved."""
    sink = RaisingSink()
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "off"}]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.WARNING
    # One attempt was made per emitted event, none broke the run:
    # ``ReasoningJudgeInvoked`` (1) + ``GoldfiveLLMCallStart`` (1) +
    # ``GoldfiveLLMCallEnd`` (1) = 3 since the span-wrapper PR.
    assert sink.calls == 3


# ---------------------------------------------------------------------------
# elapsed_ms and sequence
# ---------------------------------------------------------------------------


async def test_elapsed_ms_is_non_negative_int() -> None:
    """``elapsed_ms`` is a measured duration, never negative."""
    sink = ListSink()
    call_llm = _stub_call_llm([{"on_task": True}])
    await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    payload = _one_judge_event(sink)
    assert isinstance(payload.elapsed_ms, int)
    assert payload.elapsed_ms >= 0


async def test_sequence_fn_advances_envelope_sequence() -> None:
    """``sequence_fn`` advances on every emission; its value lands on each envelope."""
    sink = ListSink()
    call_llm = _stub_call_llm([{"on_task": True}])
    counter = {"n": 41}

    def seq() -> int:
        counter["n"] += 1
        return counter["n"]

    await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
        sequence_fn=seq,
        run_id="r",
    )
    # Three emissions since the span-wrapper PR:
    # ``GoldfiveLLMCallStart`` (span enter) + ``ReasoningJudgeInvoked``
    # (verdict) + ``GoldfiveLLMCallEnd`` (span exit). Each pops a fresh
    # sequence number from ``seq()``.
    assert counter["n"] == 44
    judge_events = [
        e for e in sink.events
        if e.WhichOneof("payload") == "reasoning_judge_invoked"
    ]
    assert len(judge_events) == 1
    # Emission order is span-start, span-end (after call_llm returns),
    # then ReasoningJudgeInvoked — so the judge envelope carries the
    # final sequence value of 44.
    assert judge_events[0].sequence == 44


# ---------------------------------------------------------------------------
# DriftEvent.trigger_input is populated on the returned drift
# ---------------------------------------------------------------------------


async def test_returned_drift_carries_trigger_input() -> None:
    """The off-task DriftEvent carries the reasoning as ``trigger_input``."""
    sink = ListSink()
    reasoning = "raccoons have masks but this is about solar panels"
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "off"}]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning=reasoning,
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    assert drift is not None
    assert drift.trigger_input == reasoning


async def test_returned_drift_trigger_input_is_truncated_when_long() -> None:
    """Truncation applies to ``DriftEvent.trigger_input`` too."""
    sink = ListSink()
    reasoning = "y" * (rjudge.REASONING_JUDGE_MAX_REASONING_INPUT_CHARS + 500)
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "off"}]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning=reasoning,
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    assert drift is not None
    assert drift.trigger_input.endswith(" … [truncated]")


# ---------------------------------------------------------------------------
# Parsed attribution fields land on the wire
# ---------------------------------------------------------------------------


async def test_attribution_fields_land_on_judge_invoked_event() -> None:
    """focused_task_id / focus_confidence / stated_intent / provenance
    reach the ``ReasoningJudgeInvoked`` payload and survive a proto
    serialize/parse round-trip.
    """
    sink = ListSink()
    call_llm = _stub_call_llm(
        [
            {
                "classification": "justified_deviation",
                "provenance": "tool_error",
                "severity": "info",
                "reason": "tool failed, agent adapting",
                "focused_task_id": "t1",
                "focus_confidence": 0.8,
                "stated_intent": "retrying the fetch with a fallback endpoint",
            }
        ]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="the fetch tool errored; switching to the mirror",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        current_task_id="t1",
        sink=sink,
    )
    assert drift is not None
    assert drift.kind is DriftKind.JUSTIFIED_DEVIATION
    hits = [
        e for e in sink.events
        if e.WhichOneof("payload") == "reasoning_judge_invoked"
    ]
    assert len(hits) == 1
    # Round-trip through the wire encoding — the fields must not be
    # local-only attributes on the in-memory message.
    from goldfive.pb.goldfive.v1 import events_pb2 as pb

    parsed = pb.Event.FromString(hits[0].SerializeToString())
    payload = parsed.reasoning_judge_invoked
    assert payload.focused_task_id == "t1"
    assert payload.focus_confidence == pytest.approx(0.8)
    assert payload.stated_intent == "retrying the fetch with a fallback endpoint"
    assert payload.provenance == "tool_error"


async def test_attribution_fields_populated_on_on_task_verdict() -> None:
    """Attribution is extracted regardless of verdict; confidence is
    clamped to [0.0, 1.0] and provenance stays empty outside
    justified_deviation.
    """
    sink = ListSink()
    call_llm = _stub_call_llm(
        [
            {
                "classification": "on_task",
                "focused_task_id": "t1",
                "focus_confidence": 1.7,
                "stated_intent": "comparing panel specs",
            }
        ]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="comparing spec sheets",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    assert drift is None
    payload = _one_judge_event(sink)
    assert payload.focused_task_id == "t1"
    assert payload.focus_confidence == pytest.approx(1.0)  # clamped
    assert payload.stated_intent == "comparing panel specs"
    assert payload.provenance == ""


async def test_attribution_fields_default_empty_on_malformed_response() -> None:
    """Quiet-fail path leaves the attribution fields at proto defaults."""
    sink = ListSink()
    call_llm = _stub_call_llm(["not json"])
    await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="m",
        call_llm=call_llm,
        sink=sink,
    )
    payload = _one_judge_event(sink)
    assert payload.focused_task_id == ""
    assert payload.focus_confidence == pytest.approx(0.0)
    assert payload.stated_intent == ""
    assert payload.provenance == ""
