"""Round-trip + enum-alignment tests for the control proto.

These tests require the generated proto stubs under ``goldfive.pb``
(produced by ``make proto``). When those stubs are not present the
module is skipped so ``pytest`` stays green on branches where the proto
extra isn't installed — mirrors ``tests/test_conv.py``.
"""

from __future__ import annotations

import importlib.util

import pytest

from goldfive.control import AckResult, ControlAck, ControlKind, ControlMessage


def _pb_available() -> bool:
    try:
        return (
            importlib.util.find_spec("goldfive.pb.goldfive.v1.control_pb2") is not None
        )
    except (ModuleNotFoundError, ImportError):
        return False


_PB_AVAILABLE = _pb_available()

pytestmark = pytest.mark.skipif(
    not _PB_AVAILABLE,
    reason="goldfive protobuf stubs not generated yet (run `make proto`)",
)

if _PB_AVAILABLE:
    from goldfive.conv import (  # noqa: E402
        from_pb_control_ack,
        from_pb_control_event,
        to_pb_control_ack,
        to_pb_control_event,
    )
    from goldfive.pb.goldfive.v1 import control_pb2  # noqa: E402


# ---------------------------------------------------------------------------
# Enum alignment (the drift guard)
# ---------------------------------------------------------------------------


def test_control_kind_enum_alignment() -> None:
    """Every Python ``ControlKind`` member has a matching proto member.

    The whole point of moving the control wire format into goldfive's
    proto is to make drift impossible. This test is the tripwire: if
    someone adds a Python member without a proto member (or vice versa)
    the test fails before the drift reaches harmonograf.
    """
    py_names = {k.name for k in ControlKind}
    pb_names = {
        control_pb2.ControlKind.Name(v)
        for v in control_pb2.ControlKind.values()
        if control_pb2.ControlKind.Name(v) != "CONTROL_KIND_UNSPECIFIED"
    }
    # Proto names are CONTROL_KIND_<NAME>; Python names are <NAME>.
    pb_stripped = {name.removeprefix("CONTROL_KIND_") for name in pb_names}
    assert py_names == pb_stripped


def test_control_ack_result_enum_alignment() -> None:
    py_names = {r.name for r in AckResult}
    pb_names = {
        control_pb2.ControlAckResult.Name(v)
        for v in control_pb2.ControlAckResult.values()
        if control_pb2.ControlAckResult.Name(v) != "CONTROL_ACK_RESULT_UNSPECIFIED"
    }
    pb_stripped = {name.removeprefix("CONTROL_ACK_RESULT_") for name in pb_names}
    assert py_names == pb_stripped


# ---------------------------------------------------------------------------
# Per-kind round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        ControlKind.PAUSE,
        ControlKind.RESUME,
        ControlKind.CANCEL,
        ControlKind.STATUS_QUERY,
        ControlKind.INTERCEPT_TRANSFER,
    ],
)
def test_payloadless_kind_round_trip(kind: ControlKind) -> None:
    """PAUSE / RESUME / CANCEL / STATUS_QUERY / INTERCEPT_TRANSFER carry
    no structured payload; the oneof stays unset on the wire."""
    msg = ControlMessage(kind=kind, id="ctl-1", issued_at_ms=1700000000000)
    pb_msg = to_pb_control_event(msg)
    assert pb_msg.WhichOneof("payload") is None
    recovered = from_pb_control_event(pb_msg)
    assert recovered.kind == kind
    assert recovered.id == "ctl-1"
    assert recovered.payload == {}
    assert recovered.issued_at_ms == 1700000000000


def test_steer_round_trip() -> None:
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-steer",
        payload={
            "note": "focus on slide 3",
            "suggested_action": "narrow scope",
            "author": "alice",
            "annotation_id": "ann_abc123",
        },
    )
    pb_msg = to_pb_control_event(msg)
    assert pb_msg.WhichOneof("payload") == "steer"
    assert pb_msg.steer.note == "focus on slide 3"
    assert pb_msg.steer.author == "alice"
    assert pb_msg.steer.annotation_id == "ann_abc123"
    recovered = from_pb_control_event(pb_msg)
    assert recovered.kind == ControlKind.STEER
    assert recovered.payload == msg.payload


def test_steer_round_trip_without_author_or_annotation_id() -> None:
    """Back-compat: callers that don't set author / annotation_id
    round-trip through empty strings (not missing keys)."""
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-steer-bare",
        payload={"note": "pivot", "suggested_action": ""},
    )
    pb_msg = to_pb_control_event(msg)
    assert pb_msg.steer.author == ""
    assert pb_msg.steer.annotation_id == ""
    recovered = from_pb_control_event(pb_msg)
    assert recovered.payload["note"] == "pivot"
    assert recovered.payload["author"] == ""
    assert recovered.payload["annotation_id"] == ""


def test_rewind_round_trip() -> None:
    msg = ControlMessage(
        kind=ControlKind.REWIND_TO,
        id="ctl-rew",
        payload={"task_id": "t7"},
    )
    pb_msg = to_pb_control_event(msg)
    assert pb_msg.WhichOneof("payload") == "rewind"
    assert pb_msg.rewind.task_id == "t7"
    assert from_pb_control_event(pb_msg).payload == {"task_id": "t7"}


def test_approve_round_trip() -> None:
    msg = ControlMessage(
        kind=ControlKind.APPROVE,
        id="ctl-ok",
        payload={"target_id": "task-42", "detail": "looks good"},
    )
    pb_msg = to_pb_control_event(msg)
    assert pb_msg.WhichOneof("payload") == "approve"
    assert pb_msg.approve.target_id == "task-42"
    assert pb_msg.approve.detail == "looks good"
    assert from_pb_control_event(pb_msg).payload == msg.payload


def test_reject_round_trip() -> None:
    msg = ControlMessage(
        kind=ControlKind.REJECT,
        id="ctl-no",
        payload={"target_id": "adk-abc", "detail": "too risky"},
    )
    pb_msg = to_pb_control_event(msg)
    assert pb_msg.WhichOneof("payload") == "reject"
    assert pb_msg.reject.target_id == "adk-abc"
    assert from_pb_control_event(pb_msg).payload == msg.payload


def test_inject_message_round_trip() -> None:
    msg = ControlMessage(
        kind=ControlKind.INJECT_MESSAGE,
        id="ctl-inj",
        payload={"role": "user", "text": "by the way, skip section 4"},
    )
    pb_msg = to_pb_control_event(msg)
    assert pb_msg.WhichOneof("payload") == "inject_message"
    assert pb_msg.inject_message.role == "user"
    assert pb_msg.inject_message.text == "by the way, skip section 4"
    assert from_pb_control_event(pb_msg).payload == msg.payload


# ---------------------------------------------------------------------------
# Ack round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [AckResult.SUCCESS, AckResult.FAILURE, AckResult.UNSUPPORTED],
)
def test_ack_round_trip(result: AckResult) -> None:
    ack = ControlAck(
        control_id="ctl-1",
        result=result,
        detail="round-tripped",
        acked_at_ms=1700000000123,
    )
    recovered = from_pb_control_ack(to_pb_control_ack(ack))
    assert recovered == ack


def test_ack_accepts_raw_string_result() -> None:
    """Loosely-typed callers may pass ``result="SUCCESS"``; the converter
    should coerce rather than crash."""
    ack = ControlAck(control_id="ctl-1", result="SUCCESS", detail="ok")  # type: ignore[arg-type]
    pb_msg = to_pb_control_ack(ack)
    assert pb_msg.result == control_pb2.CONTROL_ACK_RESULT_SUCCESS


# ---------------------------------------------------------------------------
# Every kind dispatches via the shared converter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(ControlKind))
def test_every_kind_round_trips(kind: ControlKind) -> None:
    """Exhaustively round-trip every enum member so adding a new kind
    without its payload wiring fails loudly in CI."""
    payload: dict[str, object] = {}
    if kind == ControlKind.STEER:
        payload = {
            "note": "n",
            "suggested_action": "s",
            "author": "",
            "annotation_id": "",
        }
    elif kind == ControlKind.REWIND_TO:
        payload = {"task_id": "t"}
    elif kind in (ControlKind.APPROVE, ControlKind.REJECT):
        payload = {"target_id": "x", "detail": "d"}
    elif kind == ControlKind.INJECT_MESSAGE:
        payload = {"role": "user", "text": "hi"}
    msg = ControlMessage(kind=kind, id=f"ctl-{kind.value}", payload=payload)
    recovered = from_pb_control_event(to_pb_control_event(msg))
    assert recovered.kind == kind
    assert recovered.id == msg.id
    assert recovered.payload == payload
