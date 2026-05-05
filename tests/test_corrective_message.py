"""Tests for :func:`goldfive.steerer.compose_corrective_user_message`.

Pins the shape of the directive user message the Steerer hands off to
the Runner's overlay loop on Level 3 (CANCEL_REINVOKE) of the
intervention ladder. Messages are short, action-focused, and avoid
goldfive jargon (see goldfive#142).
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import compose_corrective_user_message  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Task,
    TaskStatus,
)


def _plan_with_next(next_title: str = "Summarize findings") -> Plan:
    """Build a plan whose next PENDING task has ``next_title``."""
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="Research", status=TaskStatus.COMPLETED),
            Task(id="t1", title=next_title, status=TaskStatus.PENDING),
        ],
        edges=[],
    )


def _drift(kind: DriftKind, task_id: str = "t0") -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=DriftSeverity.CRITICAL,
        detail=f"synthetic {kind.value}",
        current_task_id=task_id,
    )


def test_looping_reasoning_message_shape() -> None:
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.LOOPING_REASONING, task_id="t0"),
        refined_plan=_plan_with_next("Draft the final note"),
    )
    assert "looped on t0" in msg
    assert "Draft the final note" in msg
    assert "different approach" in msg


def test_plan_divergence_message_shape() -> None:
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.PLAN_DIVERGENCE),
        refined_plan=_plan_with_next("Summarize findings"),
    )
    assert "diverged from the plan" in msg
    assert "Summarize findings" in msg


def test_agent_refusal_message_shape() -> None:
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.AGENT_REFUSAL, task_id="t7"),
        refined_plan=_plan_with_next("Alternative approach"),
    )
    assert "t7" in msg
    assert "Alternative approach" in msg
    assert "could not complete" in msg


def test_tool_error_message_shape() -> None:
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.TOOL_ERROR, task_id="t4"),
        refined_plan=_plan_with_next("Second attempt"),
    )
    assert "tool error" in msg
    assert "Second attempt" in msg
    assert "t4" in msg


def test_runaway_delegation_message_shape() -> None:
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.RUNAWAY_DELEGATION, task_id="coord"),
        refined_plan=_plan_with_next("Do the work directly"),
    )
    assert "kept delegating" in msg
    assert "coord" in msg
    assert "Do the work directly" in msg


def test_message_avoids_goldfive_jargon() -> None:
    """No 'synthetic', 'healed', 'orphan', 'drift' in user-facing copy.

    Enumerated from goldfive#142's "keep messages SHORT, action-focused,
    no goldfive jargon, no 'synthetic'/'healed'/'orphan' language."
    """
    # "Refined plan" is allowed (the issue's own example uses that
    # phrasing); jargon-gate is specifically on the goldfive-internal
    # vocabulary ("synthetic", "healed", "orphan", "steerer") plus the
    # word "drift" which is a postmortem term for end-users.
    forbidden = ("synthetic", "healed", "orphan", "drift", "steerer")
    for kind in (
        DriftKind.LOOPING_REASONING,
        DriftKind.PLAN_DIVERGENCE,
        DriftKind.AGENT_REFUSAL,
        DriftKind.INTENT_DIVERGENCE,
        DriftKind.TOOL_ERROR,
        DriftKind.RUNAWAY_DELEGATION,
        DriftKind.CONFABULATION_RISK,
        DriftKind.SELF_REPORTED_STUCK,
    ):
        msg = compose_corrective_user_message(
            drift=_drift(kind, task_id="t0"),
            refined_plan=_plan_with_next(),
        )
        lower = msg.lower()
        for bad in forbidden:
            assert bad not in lower, (
                f"{kind.value} message contains forbidden jargon {bad!r}: {msg!r}"
            )


def test_message_is_short() -> None:
    """A corrective message should be at most ~250 chars -- a single
    directive, not a postmortem. Pins the "short, action-focused" rule.
    """
    for kind in (
        DriftKind.LOOPING_REASONING,
        DriftKind.PLAN_DIVERGENCE,
        DriftKind.AGENT_REFUSAL,
        DriftKind.TOOL_ERROR,
        DriftKind.RUNAWAY_DELEGATION,
    ):
        msg = compose_corrective_user_message(
            drift=_drift(kind, task_id="task-with-long-identifier"),
            refined_plan=_plan_with_next("A fairly descriptive next step title"),
        )
        assert len(msg) < 260, f"{kind.value} message too long ({len(msg)}): {msg!r}"


def test_handles_missing_plan() -> None:
    """A composer called with a missing refined plan falls back to a
    generic 'next planned step' rather than interpolating an empty
    title -- prevents malformed outputs like 'proceed with  .'."""
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.PLAN_DIVERGENCE),
        refined_plan=None,
    )
    assert "next planned step" in msg
    # No double-space artifacts from an empty title.
    assert "  " not in msg or msg.count("  ") <= 0


def test_handles_missing_task_id() -> None:
    drift = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.CRITICAL,
        detail="",
        current_task_id="",
    )
    msg = compose_corrective_user_message(
        drift=drift,
        refined_plan=_plan_with_next("Try option B"),
    )
    # Falls back to a readable placeholder, not an empty interpolation.
    assert "the current task" in msg
    assert "Try option B" in msg


def test_unknown_drift_kind_uses_generic_fallback() -> None:
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.CUSTOM, task_id="t3"),
        refined_plan=_plan_with_next("Move forward"),
    )
    assert "t3" in msg
    assert "Move forward" in msg
    # Still action-focused: ends with a directive.
    assert msg.lower().endswith(".") or "proceed" in msg.lower()
