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
    # CONFABULATION_RISK (INFO is observe-only to preserve pre-ladder behaviour).
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
    # PLAN_DIVERGENCE — DEMOTED to OBSERVE at all severities
    # (AGENCY-PRESERVATION.md PR 3: forecast-mismatch, not goal drift;
    # also already dropped at the top of handle_drift per #252). Was
    # WARNING→ABSORB, CRITICAL→CANCEL_REINVOKE/PAUSE_ESCALATE.
    (DriftKind.PLAN_DIVERGENCE, DriftSeverity.WARNING, 0, InterventionLevel.OBSERVE),
    (
        DriftKind.PLAN_DIVERGENCE,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.OBSERVE,
    ),
    (
        DriftKind.PLAN_DIVERGENCE,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.OBSERVE,
    ),
    # OFF_TOPIC: plan-context drift from the reasoning judge. Ladder
    # mirrors PLAN_DIVERGENCE so the ABSORB path can route to the
    # goal-aware refine prompt rather than the generic refine prompt.
    (DriftKind.OFF_TOPIC, DriftSeverity.INFO, 0, InterventionLevel.OBSERVE),
    (DriftKind.OFF_TOPIC, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (
        DriftKind.OFF_TOPIC,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    (
        DriftKind.OFF_TOPIC,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
    # JUSTIFIED_DEVIATION (iter-10 PR 4): provoked deviation. Always-
    # ABSORB at every severity; CRITICAL repeat does NOT escalate
    # because the deviation is a response to a real provoking signal.
    # Backstops upstream of the ladder (per-(kind, task_id) refine
    # cooldown + task_last_progress_at stall gate) catch runaway
    # justified_deviation without involving PAUSE_ESCALATE.
    (
        DriftKind.JUSTIFIED_DEVIATION,
        DriftSeverity.INFO,
        0,
        InterventionLevel.OBSERVE,
    ),
    (
        DriftKind.JUSTIFIED_DEVIATION,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.ABSORB,
    ),
    (
        DriftKind.JUSTIFIED_DEVIATION,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.ABSORB,
    ),
    (
        DriftKind.JUSTIFIED_DEVIATION,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.ABSORB,
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
    # GOAL_DRIFT (goldfive#143, retuned for Tier 1 / F4): the judge
    # signals "agent is stuck on completed work" -- a NUDGE (corrective
    # user message) is the right first response, since the plan is
    # correct and only the agent's next-action reasoning needs an
    # anchor. CRITICAL repeat escalates to CANCEL_REINVOKE; PAUSE_
    # ESCALATE is the last-resort fallback that the default ladder
    # path provides (no explicit entry needed once CANCEL_REINVOKE
    # didn't break the loop).
    (
        DriftKind.GOAL_DRIFT,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.NUDGE,
    ),
    (
        DriftKind.GOAL_DRIFT,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.NUDGE,
    ),
    (
        DriftKind.GOAL_DRIFT,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    # SELF_REPORTED_STUCK (WARNING by default).
    (
        DriftKind.SELF_REPORTED_STUCK,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.ABSORB,
    ),
    # NEW_WORK_DISCOVERED — DEMOTED to OBSERVE at all severities
    # (AGENCY-PRESERVATION.md PR 3). The agent-authored reporting tool
    # reroutes to descriptive growth instead of refine, and the explicit
    # _LADDER row makes "non-steering at every severity" the documented
    # mapping. Was the default-fallback ABSORB / PAUSE_ESCALATE below.
    (
        DriftKind.NEW_WORK_DISCOVERED,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.OBSERVE,
    ),
    (
        DriftKind.NEW_WORK_DISCOVERED,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.OBSERVE,
    ),
    # Default fallback: a drift kind with NO _LADDER entry. CUSTOM is the
    # representative (NEW_WORK_DISCOVERED gained an explicit row in PR 3,
    # so it no longer exercises this path). WARNING -> ABSORB; CRITICAL
    # repeat -> PAUSE_ESCALATE; INFO -> OBSERVE.
    (DriftKind.CUSTOM, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (DriftKind.CUSTOM, DriftSeverity.CRITICAL, 2, InterventionLevel.PAUSE_ESCALATE),
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
    level = steerer.drift._ladder_level_for(kind, severity, occurrence)
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
    """Tier 1 / F4: GOAL_DRIFT routes through NUDGE first, not PAUSE.

    The judge's signal is "agent is grinding on completed work"; the
    plan is fine and a corrective user message (NUDGE) re-anchors the
    LLM without refining the plan. CRITICAL repeat escalates to
    CANCEL_REINVOKE — only if that also fails to break the loop does
    the default-fallback path eventually surface PAUSE_ESCALATE.
    """
    steerer = DefaultSteerer()
    # WARNING -> NUDGE (corrective message; no plan change needed).
    assert (
        steerer.drift._ladder_level_for(DriftKind.GOAL_DRIFT, DriftSeverity.WARNING, 0)
        is InterventionLevel.NUDGE
    )
    # CRITICAL first occurrence -> NUDGE.
    assert (
        steerer.drift._ladder_level_for(DriftKind.GOAL_DRIFT, DriftSeverity.CRITICAL, 0)
        is InterventionLevel.NUDGE
    )
    # CRITICAL repeat -> CANCEL_REINVOKE (cancel + restart with the
    # corrective body as the new user input).
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.GOAL_DRIFT,
            DriftSeverity.CRITICAL,
            DefaultSteerer.REFINE_FAILURE_THRESHOLD,
        )
        is InterventionLevel.CANCEL_REINVOKE
    )


def test_justified_deviation_routes_to_absorb_at_all_severities() -> None:
    """iter-10 PR 4: JUSTIFIED_DEVIATION always-ABSORBs.

    A provoked deviation is the right input for plan-extension at
    every severity — penalising it with a PAUSE_ESCALATE on CRITICAL
    repeat would punish the agent for responding to reality. The
    backstops are upstream (per-(kind, task_id) refine cooldown +
    task_last_progress_at stall gate), not on the ladder.
    """
    steerer = DefaultSteerer()
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.JUSTIFIED_DEVIATION, DriftSeverity.INFO, 0
        )
        is InterventionLevel.OBSERVE
    )
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.JUSTIFIED_DEVIATION, DriftSeverity.WARNING, 0
        )
        is InterventionLevel.ABSORB
    )
    # CRITICAL first occurrence — ABSORB, NOT cancel-reinvoke.
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.JUSTIFIED_DEVIATION, DriftSeverity.CRITICAL, 0
        )
        is InterventionLevel.ABSORB
    )
    # CRITICAL repeat — STILL ABSORB. Specifically NOT pause_escalate.
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.JUSTIFIED_DEVIATION,
            DriftSeverity.CRITICAL,
            DefaultSteerer.REFINE_FAILURE_THRESHOLD,
        )
        is InterventionLevel.ABSORB
    )
    # And way past the repeat threshold for paranoia.
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.JUSTIFIED_DEVIATION, DriftSeverity.CRITICAL, 99
        )
        is InterventionLevel.ABSORB
    )


def test_off_topic_unchanged_by_justified_deviation_addition() -> None:
    """Regression: #345's OFF_TOPIC ladder row didn't move.

    iter-10 PR 4 adds a sibling JUSTIFIED_DEVIATION row but must not
    perturb OFF_TOPIC routing. Pin the four key cells.
    """
    steerer = DefaultSteerer()
    assert (
        steerer.drift._ladder_level_for(DriftKind.OFF_TOPIC, DriftSeverity.INFO, 0)
        is InterventionLevel.OBSERVE
    )
    assert (
        steerer.drift._ladder_level_for(DriftKind.OFF_TOPIC, DriftSeverity.WARNING, 0)
        is InterventionLevel.ABSORB
    )
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.OFF_TOPIC, DriftSeverity.CRITICAL, 0
        )
        is InterventionLevel.CANCEL_REINVOKE
    )
    assert (
        steerer.drift._ladder_level_for(
            DriftKind.OFF_TOPIC,
            DriftSeverity.CRITICAL,
            DefaultSteerer.REFINE_FAILURE_THRESHOLD,
        )
        is InterventionLevel.PAUSE_ESCALATE
    )
