"""Golden decision-table snapshot for the intervention ladder.

AGENCY-PRESERVATION.md §5.3 ("Decision-table snapshot tests") makes
:meth:`goldfive.drift_observer.DriftObserver._ladder_level_for` — the
pure function ``(kind, severity, occurrence) -> InterventionLevel`` that
is the ONLY live ladder (goldfive#142; the ``steerer.py`` table
reference in older docs is stale) — into a checked-in golden surface so
every cell change shows up as an explicit, reviewable diff in this file.

Why a whole-surface snapshot and not more per-case asserts? The
roadmap's risk analysis (§5, "PR 3/7 — ladder cell changes silently
altering what other gates read") is that a demotion of one kind quietly
moves an unrelated row. ``test_intervention_ladder.py`` pins one
representative case per branch; this test pins **every** cell of the
``(kind × severity × occurrence-bucket)`` grid at once, so a stray edit
to a row PR 3 never meant to touch fails here with a one-line diff.

Buckets (§5.3, "occurrence"): ``first`` is ``occurrence_count == 0`` and
``repeat`` is ``occurrence_count >= REFINE_FAILURE_THRESHOLD`` — the only
boundary ``_ladder_level_for`` keys on (``is_repeat`` flips the
``critical_pair``). INFO and WARNING cannot differ by bucket (the
function returns ``info_level`` / ``warning_level`` without reading
``occurrence_count``), so the snapshot renders them once and bucket-
splits only the CRITICAL column. That rendering choice is itself pinned
by :func:`test_info_and_warning_columns_are_bucket_invariant` — if a
future edit makes INFO/WARNING occurrence-sensitive, that guard fails and
tells us to widen the snapshot format rather than silently lose coverage.

Commit-pairing contract (AGENCY-PRESERVATION.md PR 3): this file lands in
its OWN commit BEFORE any ladder cell changes, pinning today's table.
The demotions commit updates :data:`EXPECTED_LADDER_SURFACE` in the same
commit as the ``_LADDER`` edits, so the snapshot diff IS the review
artifact for which cells moved.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift_observer import DriftObserver  # noqa: E402
from goldfive.steerer import DefaultSteerer, InterventionLevel  # noqa: E402
from goldfive.types import DriftKind, DriftSeverity  # noqa: E402


def render_ladder_surface(steerer: DefaultSteerer) -> str:
    """Render the full ``_ladder_level_for`` surface as deterministic text.

    One line per :class:`~goldfive.types.DriftKind` (sorted by enum
    value for a stable diff). INFO and WARNING carry their single level
    (bucket-invariant — see the module docstring); CRITICAL carries the
    ``[first|repeat]`` pair where ``first`` is occurrence 0 and
    ``repeat`` is occurrence ``REFINE_FAILURE_THRESHOLD``.
    """
    threshold = steerer.REFINE_FAILURE_THRESHOLD
    lines: list[str] = []
    for kind in sorted(DriftKind, key=lambda k: k.value):
        info = steerer.drift._ladder_level_for(kind, DriftSeverity.INFO, 0).name
        warning = steerer.drift._ladder_level_for(kind, DriftSeverity.WARNING, 0).name
        crit_first = steerer.drift._ladder_level_for(kind, DriftSeverity.CRITICAL, 0).name
        crit_repeat = steerer.drift._ladder_level_for(
            kind, DriftSeverity.CRITICAL, threshold
        ).name
        lines.append(
            f"{kind.value:28s} INFO={info:7s} WARNING={warning:7s} "
            f"CRITICAL=[{crit_first}|{crit_repeat}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden snapshot — regenerate via ``render_ladder_surface`` and hand-review
# every changed line. NEVER overwrite blindly: a diff here is either an
# intended ladder demotion (update + justify in the PR body) or a
# regression (fix the code, not the golden).
#
# AGENCY-PRESERVATION.md PR 7 restructured the ladder. Changes from the PR-3
# baseline (every other row byte-for-byte unchanged):
#   * NUDGE → SIGNAL rename (the enum member): goal_drift + looping_reasoning.
#   * goldfive-authored CANCEL_REINVOKE cells → SIGNAL (advisory note, no
#     refine/cancel/steer): agent_refusal, confabulation_risk, looping_tool_call,
#     model_refusal, off_topic, self_reported_stuck, tool_error CRITICAL-first.
#   * goal_drift CRITICAL-repeat CANCEL_REINVOKE → PAUSE_ESCALATE (repeat-
#     escalation is stop-and-ask) and WARNING/CRITICAL-first NUDGE → SIGNAL.
#   * Deferred Stage-1 fix — hard-safety budget/timeout kinds (resource_exhausted,
#     too_many_steps, task_timeout, llm_call_timeout) previously fell through to
#     the default whose CRITICAL-first is ABSORB ("redirect"); they now STOP:
#     CRITICAL [ABSORB|PAUSE_ESCALATE] → [PAUSE_ESCALATE|PAUSE_ESCALATE] and
#     WARNING ABSORB → OBSERVE. PAUSE_ESCALATE (not CANCEL_REINVOKE): restart
#     can't refund a spent budget. The immediate in-flight stop comes from the
#     PR-1 cancel-authority path (these kinds are in cancel scope); the ladder
#     cell's follow-on for a spent resource is halt-and-ask-human.
#   * runaway_delegation KEEPS [CANCEL_REINVOKE|PAUSE_ESCALATE] — the behaviour
#     is the problem (not a spent budget), so killing the runaway subtree and
#     continuing non-runaway work is plausibly productive.
# capability_mismatch / new_work_discovered / plan_divergence keep their PR-3
# OBSERVE demotions; wrong_agent's dead default-fallthrough line is moot.
# ---------------------------------------------------------------------------
EXPECTED_LADDER_SURFACE = """\
agent_refusal                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
agent_transfer               INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
ambiguous_intent             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
blocked                      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
capability_mismatch          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[OBSERVE|OBSERVE]
confabulation_risk           INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
context_pressure             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
custom                       INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
goal_drift                   INFO=OBSERVE WARNING=SIGNAL  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
goal_unreachable             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
hallucination_suspected      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
human_intervention_required  INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|TERMINATE]
intent_divergence            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
justified_deviation          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|ABSORB]
llm_call_timeout             INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
looping_reasoning            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
looping_tool_call            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
model_refusal                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
new_work_discovered          INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
off_topic                    INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
plan_divergence              INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
reasoning_cluster_tightening INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
refine_validation_failed     INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
repeated_failure             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
resource_exhausted           INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
runaway_delegation           INFO=OBSERVE WARNING=OBSERVE CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
safety_concern               INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
schema_violation             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
self_reported_stuck          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
stopped_early                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_failed_fatal            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_failed_recoverable      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_timeout                 INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
too_many_steps               INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
tool_error                   INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
uncertain_progress           INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
unexpected_output            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
user_cancel                  INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
user_pause                   INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
user_steer                   INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
wrong_agent                  INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]"""


# The ``legacy_ladder`` escape hatch (GOLDFIVE_STEER_LEGACY_LADDER=1) restores
# the pre-PR-7 goldfive-authored CANCEL_REINVOKE cells. It differs from the new
# surface above on exactly the eight demoted rows; the NUDGE→SIGNAL rename and
# the hard-safety stop fix are NOT toggled (they apply in both regimes).
EXPECTED_LADDER_SURFACE_LEGACY = """\
agent_refusal                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
agent_transfer               INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
ambiguous_intent             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
blocked                      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
capability_mismatch          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[OBSERVE|OBSERVE]
confabulation_risk           INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
context_pressure             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
custom                       INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
goal_drift                   INFO=OBSERVE WARNING=SIGNAL  CRITICAL=[SIGNAL|CANCEL_REINVOKE]
goal_unreachable             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
hallucination_suspected      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
human_intervention_required  INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|TERMINATE]
intent_divergence            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
justified_deviation          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|ABSORB]
llm_call_timeout             INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
looping_reasoning            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[SIGNAL|PAUSE_ESCALATE]
looping_tool_call            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
model_refusal                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
new_work_discovered          INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
off_topic                    INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
plan_divergence              INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
reasoning_cluster_tightening INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
refine_validation_failed     INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
repeated_failure             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
resource_exhausted           INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
runaway_delegation           INFO=OBSERVE WARNING=OBSERVE CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
safety_concern               INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
schema_violation             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
self_reported_stuck          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
stopped_early                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_failed_fatal            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_failed_recoverable      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_timeout                 INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
too_many_steps               INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
tool_error                   INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
uncertain_progress           INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
unexpected_output            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
user_cancel                  INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
user_pause                   INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
user_steer                   INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
wrong_agent                  INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]"""


def test_ladder_decision_table_snapshot() -> None:
    """The full ladder surface matches the checked-in golden.

    A failure here is a ladder cell change. If intended (a demotion),
    regenerate :data:`EXPECTED_LADDER_SURFACE` and justify the diff in
    the PR body (§5.3); if not, the code regressed.
    """
    surface = render_ladder_surface(DefaultSteerer())
    assert surface == EXPECTED_LADDER_SURFACE, (
        "Ladder decision surface changed. Diff each line against the "
        "golden; an intended demotion updates the golden in the SAME "
        "commit and documents the cells in the PR body (§5.3)."
    )


def test_ladder_decision_table_snapshot_legacy_regime() -> None:
    """The ``legacy_ladder`` escape hatch restores the pre-PR-7 cells.

    §5.3 snapshots the ladder in BOTH regimes so the PR-7 restructure surfaces
    as an explicit diff. This golden differs from the default surface on
    exactly the eight goldfive-authored rows whose CANCEL_REINVOKE cells PR 7
    demoted (plus GOAL_DRIFT's CRITICAL-repeat); the NUDGE→SIGNAL rename and
    the hard-safety stop fix are shared (NOT toggled by the escape hatch).
    """
    from goldfive.config import SteeringConfig

    steerer = DefaultSteerer(steering_config=SteeringConfig(legacy_ladder=True))
    surface = render_ladder_surface(steerer)
    assert surface == EXPECTED_LADDER_SURFACE_LEGACY


def test_legacy_overrides_are_exactly_the_demoted_rows() -> None:
    """The two regimes differ on exactly the rows PR 7 demoted — no others.

    Guards the override map against accidentally restoring (or failing to
    restore) an unrelated row: the diff between the default and legacy goldens
    must be precisely the eight goldfive-authored CANCEL_REINVOKE rows.
    """
    new = EXPECTED_LADDER_SURFACE.splitlines()
    legacy = EXPECTED_LADDER_SURFACE_LEGACY.splitlines()
    differing = {
        n.split()[0] for n, leg in zip(new, legacy, strict=True) if n != leg
    }
    assert differing == {
        "agent_refusal",
        "confabulation_risk",
        "goal_drift",
        "looping_tool_call",
        "model_refusal",
        "off_topic",
        "self_reported_stuck",
        "tool_error",
    }


def test_every_drift_kind_is_in_the_surface() -> None:
    """No kind is silently dropped from the snapshot.

    The snapshot iterates ``DriftKind`` directly, so a newly added kind
    appears automatically — this guards the inverse: that the golden was
    not hand-trimmed to hide a kind.
    """
    rendered_kinds = {line.split()[0] for line in EXPECTED_LADDER_SURFACE.splitlines()}
    assert rendered_kinds == {k.value for k in DriftKind}


def test_info_and_warning_columns_are_bucket_invariant() -> None:
    """INFO/WARNING levels do not depend on the occurrence bucket.

    The snapshot renders these columns once (not ``[first|repeat]``).
    This guards that simplification: ``_ladder_level_for`` returns the
    INFO/WARNING level without consulting ``occurrence_count`` today, so
    first and repeat must agree. If they ever diverge, widen the
    snapshot to bucket-split these columns too.
    """
    steerer = DefaultSteerer()
    threshold = steerer.REFINE_FAILURE_THRESHOLD
    for kind in DriftKind:
        for severity in (DriftSeverity.INFO, DriftSeverity.WARNING):
            first = steerer.drift._ladder_level_for(kind, severity, 0)
            repeat = steerer.drift._ladder_level_for(kind, severity, threshold)
            assert first is repeat, (kind, severity, first, repeat)


def test_user_authored_rows_are_untouched_default_fallthrough() -> None:
    """USER_* kinds keep the default fallthrough mapping (§2 authority split).

    USER_STEER / USER_CANCEL / USER_PAUSE are never demoted by the
    agency-preservation ladder work: user authority is absolute. They
    are not in ``_LADDER`` and so resolve via the default fallthrough
    (INFO→OBSERVE, WARNING→ABSORB, CRITICAL first→ABSORB / repeat→
    PAUSE_ESCALATE). Pinning that here means a demotion commit that
    accidentally adds a USER_* row trips this test, not just the
    snapshot.
    """
    steerer = DefaultSteerer()
    threshold = steerer.REFINE_FAILURE_THRESHOLD
    for kind in (DriftKind.USER_STEER, DriftKind.USER_CANCEL, DriftKind.USER_PAUSE):
        assert (
            steerer.drift._ladder_level_for(kind, DriftSeverity.INFO, 0)
            is InterventionLevel.OBSERVE
        )
        assert (
            steerer.drift._ladder_level_for(kind, DriftSeverity.WARNING, 0)
            is InterventionLevel.ABSORB
        )
        assert (
            steerer.drift._ladder_level_for(kind, DriftSeverity.CRITICAL, 0)
            is InterventionLevel.ABSORB
        )
        assert (
            steerer.drift._ladder_level_for(kind, DriftSeverity.CRITICAL, threshold)
            is InterventionLevel.PAUSE_ESCALATE
        )


def test_hard_safety_kinds_stay_armed_at_critical() -> None:
    """Hard-safety (guardrail) kinds never drop to OBSERVE at CRITICAL.

    The §0/§2 authority split keeps the budget/safety guardrails always
    armed: a CRITICAL guardrail trip must still take a stop-or-steer
    action, never silently observe. This anchors the snapshot against
    :attr:`DriftObserver._HARD_SAFETY_DRIFT_KINDS` (#453) so a future
    ladder edit cannot demote a guardrail row by accident.

    PR 7 closed the latent mismatch this note used to flag: RESOURCE_EXHAUSTED
    / TOO_MANY_STEPS / TASK_TIMEOUT / LLM_CALL_TIMEOUT now map CRITICAL-first to
    PAUSE_ESCALATE (stop — restart can't refund a spent budget), not ABSORB
    (redirect); RUNAWAY_DELEGATION maps to CANCEL_REINVOKE — verified by the
    snapshot. This test additionally pins that hard-safety kinds never map
    CRITICAL to ABSORB (the §0 stop-not-redirect guarantee).
    """
    steerer = DefaultSteerer()
    threshold = steerer.REFINE_FAILURE_THRESHOLD
    for kind in DriftObserver._HARD_SAFETY_DRIFT_KINDS:
        first = steerer.drift._ladder_level_for(kind, DriftSeverity.CRITICAL, 0)
        repeat = steerer.drift._ladder_level_for(kind, DriftSeverity.CRITICAL, threshold)
        assert first is not InterventionLevel.OBSERVE, kind
        assert repeat is not InterventionLevel.OBSERVE, kind
        # PR 7 stop-not-redirect: a CRITICAL guardrail trip must STOP, never
        # ABSORB (refine-and-continue). The budget/timeout kinds +
        # HUMAN_INTERVENTION_REQUIRED map to PAUSE_ESCALATE (restart can't
        # refund a spent budget); RUNAWAY_DELEGATION to CANCEL_REINVOKE (kill
        # the runaway subtree, continue non-runaway work).
        assert first is not InterventionLevel.ABSORB, kind
