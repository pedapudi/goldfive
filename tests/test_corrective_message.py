"""Tests for the deprecated :func:`compose_corrective_user_message` shim.

AGENCY-PRESERVATION.md PR 4 re-point: this file used to pin the
``_CORRECTIVE_TEMPLATES`` command shapes ("Refined plan: proceed with
{next_task_title}", "Please proceed to '{next_task_title}' via
{next_task_agent}") per drift kind. Those templates are retired — the
agent owns MEANS, so goldfive's notes carry observation + goal +
advisory footer, never a next-task directive. The original assertions
map onto the new contract as follows:

* per-kind "message shape" tests (looping / plan-divergence / refusal /
  tool-error / runaway-delegation) → the shim delegates to
  :func:`goldfive.observer_notes.compose_note_for_drift`; one
  delegation test plus the means-command sweep below replace them.
* ``test_message_avoids_goldfive_jargon`` → kept (tightened: "drift" /
  "synthetic" / "healed" / "orphan" / "steerer" still banned) and
  extended with the §5 imperative means-verb wordlist.
* ``test_message_is_short`` → dropped: the note is deliberately a
  multi-line block (observation + goal + status + footer), not a
  one-line directive; the length cap was an artifact of the command
  format.
* ``test_handles_missing_plan`` / ``test_handles_missing_task_id`` /
  ``test_unknown_drift_kind_uses_generic_fallback`` → re-pointed at the
  note's graceful-degradation behaviour.

The full adversarial × golden coverage for the composers lives in
``tests/test_observer_notes.py``; this file only pins the shim.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.observer_notes import (  # noqa: E402
    ADVISORY_FOOTER,
    compose_note_for_drift,
)
from goldfive.steerer import compose_corrective_user_message  # noqa: E402
from goldfive.testkit.adversarial import find_means_commands  # noqa: E402
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


def _drift(kind: DriftKind, task_id: str = "t0", detail: str = "") -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=DriftSeverity.CRITICAL,
        detail=detail,
        current_task_id=task_id,
    )


# A representative sweep of drift kinds the retired template table
# carried entries for, plus CUSTOM for the generic fallback.
_SWEEP_KINDS = (
    DriftKind.LOOPING_REASONING,
    DriftKind.LOOPING_TOOL_CALL,
    DriftKind.PLAN_DIVERGENCE,
    DriftKind.AGENT_REFUSAL,
    DriftKind.MODEL_REFUSAL,
    DriftKind.INTENT_DIVERGENCE,
    DriftKind.TOOL_ERROR,
    DriftKind.RUNAWAY_DELEGATION,
    DriftKind.SELF_REPORTED_STUCK,
    DriftKind.CONFABULATION_RISK,
    DriftKind.GOAL_DRIFT,
    DriftKind.CUSTOM,
)


def test_shim_delegates_to_observer_notes() -> None:
    """The deprecated shim renders exactly what the new composer renders."""
    drift = _drift(DriftKind.LOOPING_REASONING)
    plan = _plan_with_next()
    assert compose_corrective_user_message(
        drift=drift, refined_plan=plan
    ) == compose_note_for_drift(drift=drift, plan=plan)


def test_every_kind_renders_advisory_note_shape() -> None:
    """Every kind renders the observation+goal+footer block."""
    for kind in _SWEEP_KINDS:
        msg = compose_corrective_user_message(
            drift=_drift(kind),
            refined_plan=_plan_with_next(),
        )
        assert msg.startswith("Observation: "), msg
        assert "The user's goal: " in msg, msg
        assert ADVISORY_FOOTER in msg, msg


def test_no_imperative_means_commands_any_kind() -> None:
    """§5 adversarial gate: no means-verbs directed at the agent.

    The retired templates commanded means ("proceed with …", "do NOT
    retry", "try {next_task_title}"); the note must not, for any kind,
    with or without a plan / detail.
    """
    for kind in _SWEEP_KINDS:
        for plan in (_plan_with_next(), None):
            msg = compose_corrective_user_message(
                drift=_drift(kind), refined_plan=plan
            )
            offending = find_means_commands(msg)
            assert not offending, (
                f"{kind.value} note contains means-command(s) "
                f"{offending!r}: {msg!r}"
            )


def test_no_next_task_routing_any_kind() -> None:
    """The next PENDING task's title never appears in the note.

    Replaces the per-kind "Refined plan: {next_task_title}" shape
    assertions: pointing the agent at goldfive's choice of next task
    is exactly the command surface PR 4 removes.
    """
    for kind in _SWEEP_KINDS:
        msg = compose_corrective_user_message(
            drift=_drift(kind),
            refined_plan=_plan_with_next("A very distinctive next step"),
        )
        assert "A very distinctive next step" not in msg, (kind, msg)


def test_message_avoids_goldfive_jargon() -> None:
    """No 'synthetic', 'healed', 'orphan', 'drift', 'steerer' in
    agent-facing copy (carried over from goldfive#142's content rule)."""
    forbidden = ("synthetic", "healed", "orphan", "drift", "steerer")
    for kind in _SWEEP_KINDS:
        msg = compose_corrective_user_message(
            drift=_drift(kind),
            refined_plan=_plan_with_next(),
        )
        lower = msg.lower()
        for bad in forbidden:
            assert bad not in lower, (
                f"{kind.value} message contains forbidden jargon {bad!r}: {msg!r}"
            )


def test_handles_missing_plan() -> None:
    """No plan → goal placeholder + task bookkeeping, no Status crash."""
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.PLAN_DIVERGENCE),
        refined_plan=None,
    )
    assert "t0" in msg
    assert ADVISORY_FOOTER in msg
    # No double-space artifacts from empty interpolations.
    assert "  " not in msg


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
    assert ADVISORY_FOOTER in msg


def test_unknown_drift_kind_uses_generic_fallback() -> None:
    msg = compose_corrective_user_message(
        drift=_drift(DriftKind.CUSTOM, task_id="t3"),
        refined_plan=_plan_with_next("Move forward"),
    )
    assert "t3" in msg
    assert msg.startswith("Observation: ")
    assert ADVISORY_FOOTER in msg


def test_detail_is_embedded_verbatim() -> None:
    """A detector's human-readable detail rides the observation line."""
    msg = compose_corrective_user_message(
        drift=_drift(
            DriftKind.SELF_REPORTED_STUCK,
            detail="the agent's own self-check reported no recent progress",
        ),
        refined_plan=_plan_with_next(),
    )
    assert "the agent's own self-check reported no recent progress" in msg
