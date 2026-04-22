"""Table-driven tests for :class:`DefaultSteerer`'s intervention ladder.

Pins the (drift_kind, severity, occurrence_count) -> InterventionLevel
mapping from goldfive#142. Every branch of the ladder gets one
representative case so a future reorg that silently changes routing
trips one of these tests.

See ``docs/design/DRIFT.md`` and the module-level docstring on
``goldfive.steerer`` for the full ladder description.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer, InterventionLevel  # noqa: E402
from goldfive.types import DriftKind, DriftSeverity  # noqa: E402

# Each case is (kind, severity, occurrence_count, expected_level).
# REFINE_FAILURE_THRESHOLD is 2 so occurrence_count >= 2 means "repeat".
_CASES: list[tuple[DriftKind, DriftSeverity, int, InterventionLevel]] = [
    # CONFUSION (INFO is observe-only to preserve pre-ladder behaviour).
    (DriftKind.CONFUSION, DriftSeverity.INFO, 0, InterventionLevel.OBSERVE),
    (DriftKind.CONFUSION, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.CONFUSION,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    (
        DriftKind.CONFUSION,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # CONFABULATION_RISK: identical shape to CONFUSION.
    (DriftKind.CONFABULATION_RISK, DriftSeverity.INFO, 0, InterventionLevel.OBSERVE),
    (
        DriftKind.CONFABULATION_RISK,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.ABSORB,
    ),
    # AGENT_REFUSAL: WARNING -> absorb (refine), CRITICAL first -> cancel+re-invoke.
    (DriftKind.AGENT_REFUSAL, DriftSeverity.INFO, 0, InterventionLevel.OBSERVE),
    (DriftKind.AGENT_REFUSAL, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.AGENT_REFUSAL,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    (
        DriftKind.AGENT_REFUSAL,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # MODEL_REFUSAL mirrors AGENT_REFUSAL.
    (DriftKind.MODEL_REFUSAL, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.MODEL_REFUSAL,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    # LOOPING_REASONING: graduated severity landed in goldfive#204.
    # INFO (meta-tool retries at low counts) -> OBSERVE; WARNING ->
    # ABSORB (unchanged); CRITICAL first -> NUDGE (refine + queue
    # corrective message for overlay loop); CRITICAL repeat ->
    # PAUSE_ESCALATE.
    (DriftKind.LOOPING_REASONING, DriftSeverity.INFO, 0, InterventionLevel.OBSERVE),
    (DriftKind.LOOPING_REASONING, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.LOOPING_REASONING,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.NUDGE,
    ),
    (
        DriftKind.LOOPING_REASONING,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # LOOPING_TOOL_CALL mirrors LOOPING_REASONING.
    (DriftKind.LOOPING_TOOL_CALL, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.LOOPING_TOOL_CALL,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    # REASONING_CLUSTER_TIGHTENING: INFO-only early-warning; always OBSERVE.
    (
        DriftKind.REASONING_CLUSTER_TIGHTENING,
        DriftSeverity.INFO,
        0,
        InterventionLevel.OBSERVE,
    ),
    # PLAN_DIVERGENCE.
    (DriftKind.PLAN_DIVERGENCE, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.PLAN_DIVERGENCE,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    (
        DriftKind.PLAN_DIVERGENCE,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # INTENT_DIVERGENCE: CRITICAL -> pause-escalate even on first occurrence.
    (DriftKind.INTENT_DIVERGENCE, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.INTENT_DIVERGENCE,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    (
        DriftKind.INTENT_DIVERGENCE,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # TOOL_ERROR.
    (DriftKind.TOOL_ERROR, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.TOOL_ERROR,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    # RUNAWAY_DELEGATION: CRITICAL-only shape.
    (
        DriftKind.RUNAWAY_DELEGATION,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    (
        DriftKind.RUNAWAY_DELEGATION,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # REFINE_VALIDATION_FAILED: CRITICAL -> PAUSE_ESCALATE (terminal planner signal).
    (
        DriftKind.REFINE_VALIDATION_FAILED,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    (
        DriftKind.REFINE_VALIDATION_FAILED,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # HUMAN_INTERVENTION_REQUIRED: CRITICAL -> pause first, terminate on repeat.
    (
        DriftKind.HUMAN_INTERVENTION_REQUIRED,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    (
        DriftKind.HUMAN_INTERVENTION_REQUIRED,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.TERMINATE,
    ),
    # GOAL_DRIFT (goldfive#143): CRITICAL -> PAUSE_ESCALATE both first
    # and repeat; refine can't recover structural goal drift.
    (
        DriftKind.GOAL_DRIFT,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    (
        DriftKind.GOAL_DRIFT,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # SELF_REPORTED_STUCK (WARNING by default).
    (
        DriftKind.SELF_REPORTED_STUCK,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.ABSORB,
    ),
    # Default fallback: a drift kind with no table entry. NEW_WORK_DISCOVERED
    # is WARNING severity by default -> ABSORB.
    (
        DriftKind.NEW_WORK_DISCOVERED,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.ABSORB,
    ),
    # Default fallback, CRITICAL repeat -> PAUSE_ESCALATE.
    (
        DriftKind.NEW_WORK_DISCOVERED,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # INFO on a kind without a table entry -> OBSERVE.
    (DriftKind.CUSTOM, DriftSeverity.INFO, 0, InterventionLevel.OBSERVE),
]


@pytest.mark.parametrize(
    "kind,severity,occurrence,expected",
    _CASES,
    ids=[f"{k.value}-{s.value}-occ{o}-{exp.name}" for k, s, o, exp in _CASES],
)
def test_ladder_mapping(
    kind: DriftKind,
    severity: DriftSeverity,
    occurrence: int,
    expected: InterventionLevel,
) -> None:
    steerer = DefaultSteerer()
    level = steerer._ladder_level_for(kind, severity, occurrence)
    assert level is expected, (
        f"ladder({kind.value}, {severity.value}, occ={occurrence}) "
        f"= {level.name}, expected {expected.name}"
    )


def test_ladder_level_ordering_is_monotonic_by_intrusiveness() -> None:
    """Enum values progress from least to most intrusive.

    A Runner reading the enum ordinal can compare levels with ``<`` /
    ``>``; the names encode the semantics. Pins the ordering so a
    future reshuffle is caught.
    """
    assert InterventionLevel.OBSERVE < InterventionLevel.ABSORB
    assert InterventionLevel.ABSORB < InterventionLevel.NUDGE
    assert InterventionLevel.NUDGE < InterventionLevel.CANCEL_REINVOKE
    assert InterventionLevel.CANCEL_REINVOKE < InterventionLevel.PAUSE_ESCALATE
    assert InterventionLevel.PAUSE_ESCALATE < InterventionLevel.TERMINATE


def test_ladder_covers_goal_drift() -> None:
    """goldfive#143 ``GOAL_DRIFT`` -> Level 4 always per the goldfive#142 table."""
    steerer = DefaultSteerer()
    # CRITICAL first -> PAUSE_ESCALATE.
    assert (
        steerer._ladder_level_for(DriftKind.GOAL_DRIFT, DriftSeverity.CRITICAL, 0)
        is InterventionLevel.PAUSE_ESCALATE
    )
    # CRITICAL repeat -> still PAUSE_ESCALATE (refine can't recover from
    # structural goal drift).
    assert (
        steerer._ladder_level_for(
            DriftKind.GOAL_DRIFT,
            DriftSeverity.CRITICAL,
            DefaultSteerer.REFINE_FAILURE_THRESHOLD,
        )
        is InterventionLevel.PAUSE_ESCALATE
    )
