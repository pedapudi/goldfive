"""Scaffolding tests for the iter-10 three-state reasoning-judge verdict
(see ``docs/design/`` and the iter-10 design doc / PR 1).

PR 1 ships *pure scaffolding* ahead of the parser change in PR 3 and the
routing change in PR 4:

* ``ReasoningJudgeVerdict`` gains ``classification`` + ``provenance``
  fields (both default ``""``).
* ``DriftKind.JUSTIFIED_DEVIATION`` is added (StrEnum + proto enum).
* The ``ReasoningJudgeInvoked`` proto event gains a ``classification``
  string field; the ``_emit_judge_invoked`` builder accepts and forwards
  it (default ``""`` so existing callers don't break).

These tests pin those scaffolding properties only — the behavioural
contracts (parser tolerance, routing) are exercised by PR 3 / PR 4
tests. Asserts target the ABI / wire surface so the scaffolding is
discoverable by sinks and downstream consumers without yet having
behaviour to drive it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import reasoning_judge as rjudge  # noqa: E402
from goldfive.types import DriftEvent, DriftKind, DriftSeverity  # noqa: E402

# ---------------------------------------------------------------------------
# ReasoningJudgeVerdict dataclass scaffolding
# ---------------------------------------------------------------------------


def test_verdict_dataclass_back_compat() -> None:
    """Pre-iter-10 construction with the original four fields still works.

    The two new fields (``classification``, ``provenance``) MUST default
    to the empty string so existing callers — including the back-compat
    wrapper :func:`classify_reasoning_drift` and any operator-side test
    helpers — continue to construct verdicts positionally / by keyword
    without breakage.
    """
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="reasoning drift: drifted to raccoons",
    )
    verdict = rjudge.ReasoningJudgeVerdict(
        drift=drift,
        focused_task_id="t1",
        focus_confidence=0.8,
        stated_intent="researching solar panels",
    )
    # Old surface preserved.
    assert verdict.drift is drift
    assert verdict.focused_task_id == "t1"
    assert verdict.focus_confidence == 0.8
    assert verdict.stated_intent == "researching solar panels"
    # New iter-10 fields default to the quiet-fail sentinel ("").
    assert verdict.classification == ""
    assert verdict.provenance == ""


def test_verdict_dataclass_three_state() -> None:
    """Each of the three classification strings can be stored on the verdict.

    PR 1 only checks construction-and-readback; PR 3 wires the parser
    to actually populate ``classification`` from the judge's response.
    """
    expected = ("on_task", "justified_deviation", "erroneous_deviation")
    for value in expected:
        verdict = rjudge.ReasoningJudgeVerdict(
            drift=None,
            focused_task_id="t1",
            classification=value,
        )
        assert verdict.classification == value
        # provenance is optional even when classification is set; only
        # PR 3's parser enforces "justified_deviation requires a real
        # provenance bucket".
        assert verdict.provenance == ""


def test_verdict_dataclass_provenance_field_round_trip() -> None:
    """The ``provenance`` field stores the four planned signal buckets.

    The provenance enum is informally an enum-of-strings; PR 3 will
    validate the value at parse time. PR 1 only checks the field is
    present and stores arbitrary strings.
    """
    for prov in (
        "tool_error",
        "surprising_result",
        "discovered_dependency",
        "new_information",
    ):
        verdict = rjudge.ReasoningJudgeVerdict(
            drift=None,
            classification="justified_deviation",
            provenance=prov,
        )
        assert verdict.provenance == prov
        assert verdict.classification == "justified_deviation"


def test_verdict_dataclass_is_frozen() -> None:
    """The verdict dataclass remains frozen post-extension.

    The frozen contract pre-dates iter-10; this regression guard catches
    accidental ``frozen=False`` slips that would let a downstream
    consumer mutate the verdict in place.
    """
    verdict = rjudge.ReasoningJudgeVerdict(drift=None)
    with pytest.raises(dataclasses_FrozenInstanceError()):
        verdict.classification = "on_task"  # type: ignore[misc]


def dataclasses_FrozenInstanceError() -> type[BaseException]:
    """Local indirection for the FrozenInstanceError class name."""
    import dataclasses

    return dataclasses.FrozenInstanceError


# ---------------------------------------------------------------------------
# DriftKind.JUSTIFIED_DEVIATION (StrEnum + proto enum) round-trip
# ---------------------------------------------------------------------------


def test_drift_kind_justified_deviation_strenum_value() -> None:
    """The Python ``DriftKind`` StrEnum exposes ``JUSTIFIED_DEVIATION``."""
    assert hasattr(DriftKind, "JUSTIFIED_DEVIATION")
    assert DriftKind.JUSTIFIED_DEVIATION.value == "justified_deviation"
    # Distinct from the OFF_TOPIC kind it doesn't share semantics with
    # (iter-10 keeps them separable so condition_id and per-(kind,
    # task) cooldown lifecycle stays per-kind, not pooled).
    assert DriftKind.JUSTIFIED_DEVIATION is not DriftKind.OFF_TOPIC


def test_drift_kind_justified_deviation_round_trip() -> None:
    """The new enum value round-trips through the proto stub.

    Parses the enum value off the regenerated ``types_pb2`` module and
    confirms it has the ABI contract iter-10 PR 4 routing relies on:
    a stable integer wire value (40 by design) + name resolution. If
    a future ``make proto`` reshuffles enum numbers, this catches it
    before the harmonograf consumer does.
    """
    from goldfive.pb.goldfive.v1 import types_pb2  # type: ignore[import]

    # Symbol is exported.
    assert hasattr(types_pb2, "DRIFT_KIND_JUSTIFIED_DEVIATION")
    proto_value = types_pb2.DRIFT_KIND_JUSTIFIED_DEVIATION
    assert isinstance(proto_value, int)
    # Numbering picked the next free slot after LLM_CALL_TIMEOUT (39).
    assert proto_value == 40

    # Name <-> value resolution via the descriptor matches.
    enum_descriptor = types_pb2.DriftKind.DESCRIPTOR
    by_name = enum_descriptor.values_by_name["DRIFT_KIND_JUSTIFIED_DEVIATION"]
    by_number = enum_descriptor.values_by_number[proto_value]
    assert by_name.number == 40
    assert by_number.name == "DRIFT_KIND_JUSTIFIED_DEVIATION"


def test_drift_kind_justified_deviation_serialises_on_drift_detected() -> None:
    """A ``DriftDetected`` proto carries the new enum value losslessly.

    Sets ``DriftDetected.kind = DRIFT_KIND_JUSTIFIED_DEVIATION``,
    serialises, parses, and asserts the kind field round-trips. This is
    the wire path harmonograf will see once PR 4 starts emitting the
    new kind. PR 1 ships only the proto+stub; this test pins the wire
    contract so PR 4 can wire the steerer without a separate proto
    bump.
    """
    from goldfive.pb.goldfive.v1 import events_pb2, types_pb2  # type: ignore[import]

    src = events_pb2.DriftDetected(
        kind=types_pb2.DRIFT_KIND_JUSTIFIED_DEVIATION,
        severity=types_pb2.DRIFT_SEVERITY_WARNING,
        detail="iter-10 PR 1 scaffolding round-trip",
        current_task_id="t1",
        current_agent_id="researcher",
        id="drift-1",
    )
    blob = src.SerializeToString()
    dst = events_pb2.DriftDetected()
    dst.ParseFromString(blob)
    assert dst.kind == types_pb2.DRIFT_KIND_JUSTIFIED_DEVIATION
    assert dst.detail == "iter-10 PR 1 scaffolding round-trip"


# ---------------------------------------------------------------------------
# ReasoningJudgeInvoked.classification proto field
# ---------------------------------------------------------------------------


def test_reasoning_judge_invoked_event_has_classification_field() -> None:
    """The proto message exposes ``classification`` (string, default '')."""
    from goldfive.pb.goldfive.v1 import events_pb2  # type: ignore[import]

    msg = events_pb2.ReasoningJudgeInvoked()
    # Default is the empty string (proto3 string default; quiet-fail
    # sentinel matches the dataclass default).
    assert msg.classification == ""
    msg.classification = "justified_deviation"
    assert msg.classification == "justified_deviation"


def test_reasoning_judge_invoked_event_classification_round_trips() -> None:
    """Wire round-trip preserves ``classification`` for every three-state value."""
    from goldfive.pb.goldfive.v1 import events_pb2  # type: ignore[import]

    for value in ("on_task", "justified_deviation", "erroneous_deviation", ""):
        src = events_pb2.ReasoningJudgeInvoked(
            run_id="r1",
            task_id="t1",
            subject_agent_id="researcher",
            model="judge",
            on_task=(value == "on_task"),
            severity="warning" if value != "on_task" else "",
            reason="iter-10 scaffolding",
            classification=value,
        )
        dst = events_pb2.ReasoningJudgeInvoked()
        dst.ParseFromString(src.SerializeToString())
        assert dst.classification == value
        assert dst.on_task is (value == "on_task")
        assert dst.run_id == "r1"
        assert dst.subject_agent_id == "researcher"


# ---------------------------------------------------------------------------
# _emit_judge_invoked builder forwards ``classification`` to the event
# ---------------------------------------------------------------------------


class _ListSink:
    """Bare-bones EventSink collecting proto envelopes into a list."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


async def test_emit_judge_invoked_default_classification_is_empty() -> None:
    """When the caller doesn't pass ``classification`` the field stays ''.

    Default-empty preserves the pre-iter-10 emission contract: existing
    call sites continue to work without source change, and operators
    inspecting old vs. new event payloads see the new field absent /
    empty until PR 3 starts populating it.
    """
    sink = _ListSink()
    await rjudge._emit_judge_invoked(
        sink=sink,
        run_id="run-1",
        session_id="sess-1",
        sequence_fn=None,
        current_task_id="t1",
        current_agent_id="researcher",
        model="judge",
        elapsed_ms=12,
        reasoning_input="thinking",
        raw_response='{"on_task": true}',
        on_task=True,
        severity="",
        reason="",
        # classification omitted on purpose — must default to "".
    )
    assert len(sink.events) == 1
    payload = sink.events[0].reasoning_judge_invoked
    assert payload.classification == ""
    assert payload.on_task is True


async def test_emit_judge_invoked_forwards_classification() -> None:
    """A non-empty ``classification`` reaches the proto payload as-is.

    Proves the kwarg is wired through. PR 3 will start passing the
    parser's verdict here; PR 1 only proves the channel exists.
    """
    sink = _ListSink()
    await rjudge._emit_judge_invoked(
        sink=sink,
        run_id="run-2",
        session_id="sess-2",
        sequence_fn=None,
        current_task_id="t1",
        current_agent_id="researcher",
        model="judge",
        elapsed_ms=42,
        reasoning_input="thinking",
        raw_response='{"classification": "justified_deviation"}',
        on_task=False,
        severity="warning",
        reason="tool returned 503",
        classification="justified_deviation",
    )
    assert len(sink.events) == 1
    payload = sink.events[0].reasoning_judge_invoked
    assert payload.classification == "justified_deviation"
    # Legacy fields still carried alongside the new one (no replacement).
    assert payload.on_task is False
    assert payload.severity == "warning"
    assert payload.reason == "tool returned 503"
