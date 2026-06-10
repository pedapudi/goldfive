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
# AGENCY-PRESERVATION.md PR 3 demoted exactly three rows from this table
# (the forecast-mismatch family); every other row is byte-for-byte the
# Commit-1 baseline:
#   * plan_divergence     WARNING ABSORB→OBSERVE, CRITICAL
#                         [CANCEL_REINVOKE|PAUSE_ESCALATE]→[OBSERVE|OBSERVE]
#   * capability_mismatch CRITICAL [ABSORB|PAUSE_ESCALATE]→[OBSERVE|OBSERVE]
#                         (WARNING stays ABSORB: Rule B / "WARNING-max").
#                         The CRITICAL→OBSERVE cells are unreachable in the
#                         DEFAULT config — Rule A and Rule C are both env-
#                         gated OFF and the only live emitter (Rule B) emits
#                         WARNING; the cells bite only under the Rule A/C
#                         escape hatches (belt-and-suspenders).
#   * new_work_discovered WARNING ABSORB→OBSERVE, CRITICAL
#                         [ABSORB|PAUSE_ESCALATE]→[OBSERVE|OBSERVE]
# wrong_agent is unchanged here: it is deprecated with no _LADDER row and
# no emitter, so its dead default-fallthrough line is moot (see the
# types.py deprecation note + the _load_ladder_tables comment).
# ---------------------------------------------------------------------------
EXPECTED_LADDER_SURFACE = """\
agent_refusal                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
agent_transfer               INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
ambiguous_intent             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
blocked                      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
capability_mismatch          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[OBSERVE|OBSERVE]
confabulation_risk           INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
context_pressure             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
custom                       INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
goal_drift                   INFO=OBSERVE WARNING=NUDGE   CRITICAL=[NUDGE|CANCEL_REINVOKE]
goal_unreachable             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
hallucination_suspected      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
human_intervention_required  INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|TERMINATE]
intent_divergence            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
justified_deviation          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|ABSORB]
llm_call_timeout             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
looping_reasoning            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[NUDGE|PAUSE_ESCALATE]
looping_tool_call            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
model_refusal                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
new_work_discovered          INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
off_topic                    INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
plan_divergence              INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
reasoning_cluster_tightening INFO=OBSERVE WARNING=OBSERVE CRITICAL=[OBSERVE|OBSERVE]
refine_validation_failed     INFO=OBSERVE WARNING=OBSERVE CRITICAL=[PAUSE_ESCALATE|PAUSE_ESCALATE]
repeated_failure             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
resource_exhausted           INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
runaway_delegation           INFO=OBSERVE WARNING=OBSERVE CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
safety_concern               INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
schema_violation             INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
self_reported_stuck          INFO=OBSERVE WARNING=ABSORB  CRITICAL=[CANCEL_REINVOKE|PAUSE_ESCALATE]
stopped_early                INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_failed_fatal            INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_failed_recoverable      INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
task_timeout                 INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
too_many_steps               INFO=OBSERVE WARNING=ABSORB  CRITICAL=[ABSORB|PAUSE_ESCALATE]
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

    Note (consistency, not a fix): RESOURCE_EXHAUSTED / TOO_MANY_STEPS /
    TASK_TIMEOUT / LLM_CALL_TIMEOUT are hard-safety (cancel-authorised)
    yet map CRITICAL-first to ABSORB (refine), i.e. "redirect" rather
    than the §0 "stop". That latent mismatch predates PR 3 and is the
    ladder restructure's concern (PR 7); this test only asserts they are
    not OBSERVE.
    """
    steerer = DefaultSteerer()
    threshold = steerer.REFINE_FAILURE_THRESHOLD
    for kind in DriftObserver._HARD_SAFETY_DRIFT_KINDS:
        first = steerer.drift._ladder_level_for(kind, DriftSeverity.CRITICAL, 0)
        repeat = steerer.drift._ladder_level_for(kind, DriftSeverity.CRITICAL, threshold)
        assert first is not InterventionLevel.OBSERVE, kind
        assert repeat is not InterventionLevel.OBSERVE, kind
