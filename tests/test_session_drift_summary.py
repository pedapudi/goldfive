"""Tests for :class:`Session.drift_summary` (zicato-optimization-surface).

Covers:

* An empty session returns an empty summary that is falsy.
* Drifts pushed via :meth:`DriftObserver._emit_drift_detected` are
  reflected in the summary in emission order.
* The summary groups by kind and severity correctly.
* ``total_severity_weight`` matches the documented 1/3/10 scale.
* The dedupe guard prevents the same drift id from being counted
  twice across multiple :meth:`_emit_drift_detected` calls (e.g. when
  a drift event is replayed at the stale-verdict gate).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    DriftSummary,
    Session,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _NullPlanner:
    async def generate(self, **_: Any) -> Any:
        return None

    async def refine(self, **_: Any) -> Any:
        return None


def _build_steerer() -> tuple[DefaultSteerer, Session]:
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())
    session = Session(run_id="run-summary")
    return steerer, session


# ---------------------------------------------------------------------------
# Empty session
# ---------------------------------------------------------------------------


def test_empty_session_drift_summary_is_empty_and_falsy() -> None:
    session = Session(run_id="r-empty")
    summary = session.drift_summary
    assert isinstance(summary, DriftSummary)
    assert summary.by_kind == {}
    assert summary.by_severity == {}
    assert summary.events == ()
    assert summary.total_severity_weight == 0.0
    assert not summary
    assert len(summary) == 0


# ---------------------------------------------------------------------------
# Aggregation after emits
# ---------------------------------------------------------------------------


async def test_drift_summary_reflects_emitted_drifts_in_order() -> None:
    steerer, session = _build_steerer()
    drifts = [
        DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.INFO),
        DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING),
        DriftEvent(kind=DriftKind.GOAL_DRIFT, severity=DriftSeverity.CRITICAL),
    ]
    for d in drifts:
        await steerer.drift._emit_drift_detected(session, d)

    summary = session.drift_summary
    assert len(summary) == 3
    assert summary.by_kind == {
        DriftKind.OFF_TOPIC: 2,
        DriftKind.GOAL_DRIFT: 1,
    }
    assert summary.by_severity == {
        DriftSeverity.INFO: 1,
        DriftSeverity.WARNING: 1,
        DriftSeverity.CRITICAL: 1,
    }
    # Emission order is preserved.
    assert [e.kind for e in summary.events] == [
        DriftKind.OFF_TOPIC,
        DriftKind.OFF_TOPIC,
        DriftKind.GOAL_DRIFT,
    ]


async def test_drift_summary_total_severity_weight_uses_documented_scale() -> None:
    steerer, session = _build_steerer()
    await steerer.drift._emit_drift_detected(
        session,
        DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.INFO),
    )
    await steerer.drift._emit_drift_detected(
        session,
        DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING),
    )
    await steerer.drift._emit_drift_detected(
        session,
        DriftEvent(kind=DriftKind.GOAL_DRIFT, severity=DriftSeverity.CRITICAL),
    )
    summary = session.drift_summary
    # 1 (INFO) + 3 (WARNING) + 10 (CRITICAL) = 14
    assert summary.total_severity_weight == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# Dedupe guard
# ---------------------------------------------------------------------------


async def test_drift_summary_dedupes_same_drift_id_across_replays() -> None:
    """A drift emitted twice with the same id is recorded once."""
    steerer, session = _build_steerer()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
    )
    await steerer.drift._emit_drift_detected(session, drift)
    await steerer.drift._emit_drift_detected(session, drift)
    summary = session.drift_summary
    assert len(summary) == 1
    assert summary.by_kind == {DriftKind.OFF_TOPIC: 1}


async def test_drift_summary_does_not_dedupe_distinct_drifts_of_same_kind() -> None:
    """Two distinct DriftEvents (different ids) of the same kind both count."""
    steerer, session = _build_steerer()
    a = DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING)
    b = DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING)
    assert a.id != b.id
    await steerer.drift._emit_drift_detected(session, a)
    await steerer.drift._emit_drift_detected(session, b)
    assert len(session.drift_summary) == 2


# ---------------------------------------------------------------------------
# DriftSummary frozen contract
# ---------------------------------------------------------------------------


def test_drift_summary_is_frozen() -> None:
    session = Session(run_id="r")
    summary = session.drift_summary
    from dataclasses import FrozenInstanceError

    with pytest.raises((FrozenInstanceError, AttributeError)):
        summary.total_severity_weight = 99.0  # type: ignore[misc]


def test_drift_summary_snapshot_is_independent_of_later_emits() -> None:
    """Returning a summary captures a snapshot — later emits don't mutate it."""
    session = Session(run_id="r")
    session.drift_events.append(
        DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.INFO)
    )
    snapshot_before = session.drift_summary
    assert len(snapshot_before) == 1
    session.drift_events.append(
        DriftEvent(kind=DriftKind.GOAL_DRIFT, severity=DriftSeverity.WARNING)
    )
    snapshot_after = session.drift_summary
    assert len(snapshot_before) == 1  # unchanged
    assert len(snapshot_after) == 2
