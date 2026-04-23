"""Strict-enum tests for goldfive.conv (issue #211 regression guard).

Before this refactor the private ``_*_to_pb`` helpers took a ``pb``
argument and used ``getattr(pb, NAME, 0)`` to look up enum ints. That
silent fallback meant a caller passing the wrong proto module (``events_pb2``
instead of ``types_pb2``) got ``UNSPECIFIED`` (= 0) on the wire, with no
indication of the mistake. The status_query dispatch path hit this bug
in goldfive#211 and downgraded 50,212 drift events to ``kind=0``.

The fix: the helpers take no ``pb`` argument — they resolve the module
internally — and missing names raise ``ValueError`` instead of returning
0. These tests pin the new contract and the expected wire values.
"""

from __future__ import annotations

import importlib.util

import pytest

from goldfive.types import DriftKind, DriftSeverity, TaskStatus


def _pb_available() -> bool:
    try:
        return importlib.util.find_spec("goldfive.pb.goldfive.v1.types_pb2") is not None
    except (ModuleNotFoundError, ImportError):
        return False


_PB_AVAILABLE = _pb_available()

pytestmark = pytest.mark.skipif(
    not _PB_AVAILABLE,
    reason="goldfive protobuf stubs not generated yet (depends on issue #3)",
)


# ---------------------------------------------------------------------------
# DriftKind: verify explicit wire values for the key motivating cases.
#
# These are the values called out in the goldfive#211 writeup. If the
# proto source of truth ever renumbers, this test fails loud — which is
# exactly what we want; wire-value changes need a migration, not a
# silent downgrade.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected_wire_value",
    [
        (DriftKind.CUSTOM, 25),
        (DriftKind.PLAN_DIVERGENCE, 4),
        (DriftKind.LOOPING_REASONING, 27),
        (DriftKind.USER_STEER, 5),
        (DriftKind.TASK_FAILED_FATAL, 8),
    ],
)
def test_drift_kind_to_pb_pins_expected_wire_values(kind, expected_wire_value) -> None:
    from goldfive.conv import _drift_kind_to_pb
    from goldfive.pb.goldfive.v1 import types_pb2

    wire = _drift_kind_to_pb(kind)
    assert wire == expected_wire_value
    # Double-check: the wire int really does resolve to the expected name
    # on the types_pb2 module (not events_pb2).
    assert types_pb2.DriftKind.Name(wire) == f"DRIFT_KIND_{kind.name}"


def test_drift_kind_to_pb_takes_no_pb_argument() -> None:
    """Regression guard for goldfive#211. The helper used to take a ``pb``
    argument and silently returned 0 when the wrong module was passed.
    The new signature takes no module — a caller *cannot* pass the wrong
    thing even if they try.
    """
    from goldfive.conv import _drift_kind_to_pb
    from goldfive.pb.goldfive.v1 import events_pb2

    # Positional-only TypeError path: passing events_pb2 is now a runtime
    # TypeError rather than a silent 0.
    with pytest.raises(TypeError):
        _drift_kind_to_pb(DriftKind.CUSTOM, events_pb2)  # type: ignore[call-arg]

    # Without the stray argument, CUSTOM resolves to its real wire value.
    assert _drift_kind_to_pb(DriftKind.CUSTOM) == 25


def test_drift_kind_user_pause_degrades_to_custom() -> None:
    """``DriftKind.USER_PAUSE`` exists on the Python side but intentionally
    is not on the wire (see proto/goldfive/v1/types.proto). It legitimately
    degrades to DRIFT_KIND_CUSTOM rather than raising.
    """
    from goldfive.conv import _drift_kind_to_pb

    assert _drift_kind_to_pb(DriftKind.USER_PAUSE) == 25  # DRIFT_KIND_CUSTOM


# ---------------------------------------------------------------------------
# TaskStatus round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", list(TaskStatus))
def test_task_status_round_trip(status) -> None:
    from goldfive.conv import _task_status_from_pb, _task_status_to_pb

    wire = _task_status_to_pb(status)
    assert wire != 0, f"{status} serialized to UNSPECIFIED"
    assert _task_status_from_pb(wire) == status


# ---------------------------------------------------------------------------
# DriftSeverity round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("severity", list(DriftSeverity))
def test_drift_severity_round_trip(severity) -> None:
    from goldfive.conv import _drift_severity_from_pb, _drift_severity_to_pb

    wire = _drift_severity_to_pb(severity)
    assert wire != 0, f"{severity} serialized to UNSPECIFIED"
    assert _drift_severity_from_pb(wire) == severity


# ---------------------------------------------------------------------------
# DriftKind round-trip (every wire-defined member should round-trip
# without loss; USER_PAUSE is excluded per the comment above).
# ---------------------------------------------------------------------------


_DRIFT_KINDS_ON_WIRE = [k for k in DriftKind if k is not DriftKind.USER_PAUSE]


@pytest.mark.parametrize("kind", _DRIFT_KINDS_ON_WIRE)
def test_drift_kind_round_trip(kind) -> None:
    from goldfive.conv import _drift_kind_from_pb, _drift_kind_to_pb

    wire = _drift_kind_to_pb(kind)
    assert wire != 0, f"{kind} serialized to UNSPECIFIED"
    assert _drift_kind_from_pb(wire) == kind


# ---------------------------------------------------------------------------
# Strict lookup fails loud on unknown names.
# ---------------------------------------------------------------------------


def test_strict_enum_value_raises_on_missing_name() -> None:
    """The new ``_strict_enum_value`` helper is what replaced the silent
    ``getattr(pb, name, 0)`` pattern. Missing names must raise so that a
    future proto regeneration that drops a kind surfaces in tests instead
    of silently downgrading the wire.
    """
    from goldfive.conv import _strict_enum_value
    from goldfive.pb.goldfive.v1 import types_pb2

    with pytest.raises(ValueError, match="unknown proto enum value"):
        _strict_enum_value(types_pb2, "DRIFT_KIND_THIS_NAME_DOES_NOT_EXIST")

    # And passing the wrong module (events_pb2) for a types-enum name also
    # fails loud. This is the concrete goldfive#211 regression path.
    from goldfive.pb.goldfive.v1 import events_pb2

    with pytest.raises(ValueError, match="unknown proto enum value"):
        _strict_enum_value(events_pb2, "DRIFT_KIND_CUSTOM")
