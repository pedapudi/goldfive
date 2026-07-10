"""AGENCY-PRESERVATION.md PR 4 — observer-note content contract tests.

Four binding §5 requirements covered here:

1. **Adversarial content tests** — every rendered note, across all
   drift kinds × severities × composers (the note composer, the
   executor nudge-replay wrapper, the goldfive steer-restart wrapper),
   contains no imperative means-verbs directed at the agent. The
   wordlist checker lives in :mod:`goldfive.testkit.adversarial`
   (test-side only; never an NL classifier in production).
2. **Golden-output tests** — exact rendered notes for representative
   drift kinds, so future content changes are reviewable diffs.
3. **USER_STEER byte-identity** — the user-authored steer-restart
   rendering is pinned as a full-string equality; PR 4 changes only
   goldfive-authored content.
4. **Graceful degradation** — judges returning old-style responses
   (no ``note_to_agent``) leave the field empty and the composer falls
   back to ``detail``; the full fallback chain
   (``note_to_agent`` → detector facts → ``detail`` → per-kind
   template) is pinned.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.executors.sequential import SequentialExecutor  # noqa: E402
from goldfive.observer_notes import (  # noqa: E402
    ADVISORY_FOOTER,
    GOAL_QUESTION,
    compose_note_for_drift,
    compose_observer_note,
    compose_status_line,
    observation_for_drift,
    render_goals_text,
)
from goldfive.testkit.adversarial import find_means_commands  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="Research", status=TaskStatus.COMPLETED),
            Task(id="t1", title="Draft the summary", status=TaskStatus.RUNNING),
            Task(id="t2", title="Review", status=TaskStatus.PENDING),
        ],
        edges=[],
    )


def _session(plan: Plan | None = None) -> SimpleNamespace:
    """Duck-typed session: the composer reads only ``goals`` / ``plan``."""
    return SimpleNamespace(
        goals=[Goal(id="g1", summary="research raccoon habitats and draft a summary")],
        plan=plan if plan is not None else _plan(),
    )


def _drift(
    kind: DriftKind,
    *,
    severity: DriftSeverity = DriftSeverity.WARNING,
    detail: str = "",
    task_id: str = "t1",
    raw: object = None,
    note_to_agent: str = "",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=detail,
        current_task_id=task_id,
        raw=raw,
        note_to_agent=note_to_agent,
    )


# ---------------------------------------------------------------------------
# 1. Adversarial content sweep — kinds × severities × composers
# ---------------------------------------------------------------------------


def _all_rendered_messages() -> list[tuple[str, str]]:
    """Render (label, message) across every kind, severity, composer."""
    rendered: list[tuple[str, str]] = []
    session = _session()
    for kind in DriftKind:
        for severity in DriftSeverity:
            for detail in ("", "detector emitted a reason string"):
                drift = _drift(kind, severity=severity, detail=detail)
                label = f"{kind.value}/{severity.value}/detail={bool(detail)}"
                note = compose_note_for_drift(drift=drift, session=session)
                rendered.append((f"note:{label}", note))
                # Executor wrappers (delivery framing) on top of the
                # same note bodies.
                rendered.append(
                    (
                        f"nudge_replay:{label}",
                        SequentialExecutor._compose_nudge_replay_message([note]),
                    )
                )
                rendered.append(
                    (
                        f"steer_restart:{label}",
                        SequentialExecutor._compose_steer_restart_message(
                            None,
                            fallback=note,
                            source="goldfive",
                            superseded_task_ids=["t1"],
                            replacement_task_ids=["t1b"],
                        ),
                    )
                )
    # Tool-loop deterministic renderings (all three modes).
    for raw in (
        {
            "mode": "exact",
            "tool_name": "search_web",
            "args_hash": "a1b2c3d4",
            "count": 5,
            "window_len": 10,
        },
        {"mode": "name", "tool_name": "search_web", "count": 7, "window_len": 10},
        {"mode": "alternating", "tools": ["lint", "build"], "window_len": 5},
    ):
        drift = _drift(DriftKind.LOOPING_REASONING, raw=raw)
        rendered.append(
            (
                f"note:tool_loop/{raw['mode']}",
                compose_note_for_drift(drift=drift, session=session),
            )
        )
    return rendered


def test_no_imperative_means_verbs_in_any_rendering() -> None:
    """§5 adversarial gate over the full kind × severity × composer grid."""
    for label, msg in _all_rendered_messages():
        offending = find_means_commands(msg)
        assert not offending, (
            f"{label} contains means-command(s) {offending!r}:\n{msg}"
        )


def test_no_problem_naming_or_jargon_in_composed_observations() -> None:
    """goldfive-composed text avoids apology-bait problem naming.

    The ``_correction_injection`` lesson (#250/#252/#253): "failed" /
    "broken" / "wrong" provoke meta-commentary instead of work. Checked
    on goldfive-COMPOSED fallback observations (empty detail) — judge /
    detector ``detail`` strings pass through verbatim and are governed
    by the judge prompts instead. The Status line may carry the literal
    ledger status word (bookkeeping is exempt), so the check runs on
    the observation line only.
    """
    session = _session()
    banned = ("failed", "broken", "wrong", "incorrect", "drift", "steerer")
    for kind in DriftKind:
        drift = _drift(kind, detail="")
        observation, _question = observation_for_drift(drift)
        lower = observation.lower()
        for bad in banned:
            assert bad not in lower, (kind, bad, observation)
        # And the full note shape is always present.
        note = compose_note_for_drift(drift=drift, session=session)
        assert note.startswith("Observation: ")
        assert "The user's goal: " in note
        assert ADVISORY_FOOTER in note


# ---------------------------------------------------------------------------
# 2. Golden outputs — representative kinds (reviewable-diff contract)
# ---------------------------------------------------------------------------


def test_golden_tool_loop_exact_note() -> None:
    """Deterministic detector facts render verbatim into the observation."""
    drift = _drift(
        DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="tool_loop_exact: search_web x 5 in last 10 calls",
        raw={
            "mode": "exact",
            "tool_name": "search_web",
            "args_hash": "a1b2c3d4",
            "count": 5,
            "window_len": 10,
            "invocation_id": "inv-1",
            "category": "work",
            "tier": "warning",
        },
    )
    note = compose_note_for_drift(drift=drift, session=_session())
    assert note == (
        "Observation: `search_web` was invoked 5 times in the last 10 tool "
        "invocations with identical arguments (args fingerprint a1b2c3d4); "
        "no task progress was recorded in that window.\n"
        "The user's goal: research raccoon habitats and draft a summary\n"
        "Status: goldfive's tracking records task t1 as running (still "
        "open); its ledger shows 1 task(s) recorded complete and 2 still "
        "open.\n"
        "This note is advisory. How to proceed is your decision; the "
        "user's instructions remain authoritative."
    )


def test_golden_goal_drift_question_form_note() -> None:
    """A below-CRITICAL judge opinion renders in question form."""
    drift = _drift(
        DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.WARNING,
        detail="recent turns kept re-reading completed research",
        task_id="t0",
    )
    note = compose_note_for_drift(drift=drift, session=_session())
    assert note == (
        "Observation: recent turns kept re-reading completed research "
        "Does the current approach still serve the user's goal?\n"
        "The user's goal: research raccoon habitats and draft a summary\n"
        "Status: goldfive's tracking records task t0 as completed; its "
        "ledger shows 1 task(s) recorded complete and 2 still open.\n"
        "This note is advisory. How to proceed is your decision; the "
        "user's instructions remain authoritative."
    )


def test_golden_judge_authored_note_used_verbatim() -> None:
    """note_to_agent wins the chain and is rendered exactly as authored."""
    drift = _drift(
        DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="operator-facing reason",
        note_to_agent=(
            "The last three reasoning turns discussed raccoon migration, "
            "which does not appear in the user's request. Does that line "
            "of inquiry still serve the goal of drafting the summary?"
        ),
    )
    note = compose_note_for_drift(drift=drift, session=_session())
    assert note == (
        "Observation: The last three reasoning turns discussed raccoon "
        "migration, which does not appear in the user's request. Does "
        "that line of inquiry still serve the goal of drafting the "
        "summary?\n"
        "The user's goal: research raccoon habitats and draft a summary\n"
        "Status: goldfive's tracking records task t1 as running (still "
        "open); its ledger shows 1 task(s) recorded complete and 2 still "
        "open.\n"
        "This note is advisory. How to proceed is your decision; the "
        "user's instructions remain authoritative."
    )


def test_golden_per_kind_fallback_observations() -> None:
    """Pin the composed fallback observation per representative kind."""
    cases = {
        DriftKind.LOOPING_TOOL_CALL: (
            "The same tool invocation was observed several times on t1 "
            "without recorded progress."
        ),
        DriftKind.AGENT_REFUSAL: (
            "The most recent response on t1 declined to continue."
        ),
        DriftKind.SELF_REPORTED_STUCK: (
            "In a recent self-check, the agent's own assessment reported "
            "no progress on t1."
        ),
        DriftKind.CONFABULATION_RISK: (
            "Output on t1 referenced external data, but no tool "
            "invocation was recorded for it."
        ),
        DriftKind.CUSTOM: (
            "goldfive's monitoring raised a signal while t1 was active."
        ),
    }
    for kind, expected in cases.items():
        observation, _q = observation_for_drift(_drift(kind, detail=""))
        assert observation == expected, kind


def test_golden_nudge_replay_wrapper() -> None:
    """The executor's nudge framing: attribution header + note verbatim.

    ``plan_revised`` selects the header (goldfive#475 truthfulness): the
    PLAN REVISION framing only renders when the steerer recorded that a
    revision actually installed; the default (no revision) framing
    asserts nothing about the plan.
    """
    msg = SequentialExecutor._compose_nudge_replay_message(
        ["NOTE-BODY"], plan_revised=True
    )
    assert msg == (
        "[GOLDFIVE PLAN REVISION — observer note]\n"
        "\n"
        "goldfive — an external monitoring layer, not the user — revised "
        "its plan bookkeeping during the prior turn. The note(s) below "
        "describe what it observed.\n"
        "\n"
        "NOTE-BODY"
    )
    plain = SequentialExecutor._compose_nudge_replay_message(["NOTE-BODY"])
    assert plain == (
        "[GOLDFIVE OBSERVER NOTE]\n"
        "\n"
        "goldfive — an external monitoring layer, not the user — observed "
        "drift during the prior turn. Its plan bookkeeping is unchanged. "
        "The note(s) below describe what it observed.\n"
        "\n"
        "NOTE-BODY"
    )


def test_golden_goldfive_steer_restart_wrapper() -> None:
    """The goldfive steer framing: header + body + bookkeeping line."""
    msg = SequentialExecutor._compose_steer_restart_message(
        None,
        fallback="NOTE-BODY",
        source="goldfive",
        superseded_task_ids=["t2"],
        replacement_task_ids=["t2b"],
    )
    assert msg == (
        "[GOLDFIVE STEERING CONTROL — supersedes prior task context]\n"
        "\n"
        "NOTE-BODY\n"
        "\n"
        "goldfive bookkeeping (for reference): goldfive's plan tracking "
        "records task id(s) t2 as superseded by this revision; its ledger "
        "contains replacement entrie(s): t2b."
    )


def test_goldfive_steer_restart_wrapper_without_task_ids() -> None:
    """No bookkeeping block when there is nothing to report."""
    msg = SequentialExecutor._compose_steer_restart_message(
        None, fallback="NOTE-BODY", source="goldfive"
    )
    assert msg == (
        "[GOLDFIVE STEERING CONTROL — supersedes prior task context]\n"
        "\n"
        "NOTE-BODY"
    )


# ---------------------------------------------------------------------------
# 3. USER_STEER paths byte-identical (§5 pin)
# ---------------------------------------------------------------------------


def test_user_steer_restart_message_byte_identical() -> None:
    """Full-string pin of the pre-PR-4 user-steer rendering.

    PR 4 changes goldfive-authored content only; the USER-authority
    relay text must not change by a single byte.
    """
    msg = SequentialExecutor._compose_steer_restart_message(
        None, fallback="user body"
    )
    assert msg == (
        "[USER STEERING CONTROL — supersedes prior task context]\n"
        "\n"
        "user body\n"
        "\n"
        "Notes:\n"
        "- Prior research, partial work, or planned tasks from the "
        "pre-steer conversation are superseded unless this message "
        "explicitly references them.\n"
        "- Proceed with the new direction. Do not continue prior work "
        "unless doing so directly serves this steer."
    )


def test_user_steer_restart_message_payload_note_byte_identical() -> None:
    """Same pin through the ControlMessage payload path."""
    msg_obj = SimpleNamespace(payload={"note": "pivot to tomatoes"})
    msg = SequentialExecutor._compose_steer_restart_message(
        msg_obj, fallback="ignored"
    )
    assert msg == (
        "[USER STEERING CONTROL — supersedes prior task context]\n"
        "\n"
        "pivot to tomatoes\n"
        "\n"
        "Notes:\n"
        "- Prior research, partial work, or planned tasks from the "
        "pre-steer conversation are superseded unless this message "
        "explicitly references them.\n"
        "- Proceed with the new direction. Do not continue prior work "
        "unless doing so directly serves this steer."
    )


# ---------------------------------------------------------------------------
# 4. Observation fallback chain + question form
# ---------------------------------------------------------------------------


def test_chain_note_to_agent_beats_detector_facts_and_detail() -> None:
    drift = _drift(
        DriftKind.LOOPING_REASONING,
        detail="detector detail",
        raw={"mode": "exact", "tool_name": "x", "count": 3, "window_len": 5},
        note_to_agent="judge-authored note",
    )
    observation, question = observation_for_drift(drift)
    assert observation == "judge-authored note"
    assert question is False  # judge chose its own form


def test_chain_detector_facts_beat_detail() -> None:
    drift = _drift(
        DriftKind.LOOPING_REASONING,
        detail="tool_loop_exact: x x 3 in last 5 calls",
        raw={"mode": "exact", "tool_name": "x", "count": 3, "window_len": 5},
    )
    observation, question = observation_for_drift(drift)
    assert observation.startswith("`x` was invoked 3 times")
    assert question is False


def test_chain_detail_beats_fallback_template() -> None:
    drift = _drift(DriftKind.TOOL_ERROR, detail="the search tool returned a 429")
    observation, _q = observation_for_drift(drift)
    assert observation == "the search tool returned a 429"


def test_chain_malformed_raw_falls_back_to_detail() -> None:
    """A raw payload that isn't the tracker shape degrades to detail."""
    drift = _drift(
        DriftKind.LOOPING_REASONING,
        detail="loop detail",
        raw={"mode": "exact"},  # missing tool_name / count
    )
    observation, _q = observation_for_drift(drift)
    assert observation == "loop detail"


def test_question_form_only_for_low_confidence_judge_opinions() -> None:
    """Severity is the confidence proxy: judge kinds below CRITICAL ask;
    CRITICAL verdicts and deterministic facts state."""
    # Judge opinion, WARNING → question.
    _obs, question = observation_for_drift(
        _drift(DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING, detail="d")
    )
    assert question is True
    # Judge opinion, CRITICAL → statement.
    _obs, question = observation_for_drift(
        _drift(DriftKind.OFF_TOPIC, severity=DriftSeverity.CRITICAL, detail="d")
    )
    assert question is False
    # Deterministic kind, WARNING → statement.
    _obs, question = observation_for_drift(
        _drift(DriftKind.LOOPING_REASONING, severity=DriftSeverity.WARNING, detail="d")
    )
    assert question is False


def test_question_form_not_doubled_when_already_a_question() -> None:
    note = compose_observer_note(
        observation="Is this still on track?",
        goals_text="g",
        question_form=True,
    )
    assert note.count("?") == 1
    assert GOAL_QUESTION not in note


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_render_goals_text_joins_and_degrades() -> None:
    assert render_goals_text(None) == "(no goals recorded for this run)"
    assert render_goals_text([]) == "(no goals recorded for this run)"
    assert render_goals_text(["plain string goal"]) == "plain string goal"
    goals = [Goal(id="g1", summary="first"), Goal(id="g2", summary="second")]
    assert render_goals_text(goals) == "first; second"


def test_compose_status_line_handles_missing_everything() -> None:
    assert compose_status_line(None, "") == ""
    line = compose_status_line(None, "t9")
    assert line == "goldfive's tracking associates this note with task t9."


def test_status_line_suppressible_and_overridable() -> None:
    drift = _drift(DriftKind.TOOL_ERROR, detail="d")
    no_status = compose_note_for_drift(drift=drift, session=_session(), status="")
    assert "Status:" not in no_status
    custom = compose_note_for_drift(
        drift=drift, session=_session(), status="custom bookkeeping line."
    )
    assert "Status: custom bookkeeping line." in custom


# ---------------------------------------------------------------------------
# 5. Judges: note_to_agent authored in the verdict call; old-style
#    responses degrade gracefully
# ---------------------------------------------------------------------------


def _stub_call_llm(response: str):
    async def _call(_system: str, _user: str, _model: str) -> str:
        return response

    return _call


async def test_goal_drift_judge_stamps_note_to_agent() -> None:
    from goldfive.drift.goals import classify_goal_drift

    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship it")],
        plan=None,
        observed_actions=[],
        model="m",
        call_llm=_stub_call_llm(
            '{"progressing": false, "reason": "looping", '
            '"note_to_agent": "Recent activity repeated the same lookup; '
            'does it still serve the goal of shipping?"}'
        ),
    )
    assert drift is not None
    assert drift.note_to_agent == (
        "Recent activity repeated the same lookup; does it still serve "
        "the goal of shipping?"
    )
    # The composed note prefers the judge's note over detail: the
    # observation line is the note verbatim; the operator-facing
    # detail string does not leak into it.
    note = compose_note_for_drift(drift=drift)
    observation_line = note.split("\n")[0]
    assert observation_line == (
        "Observation: Recent activity repeated the same lookup; does it "
        "still serve the goal of shipping?"
    )


async def test_goal_drift_judge_old_style_response_degrades() -> None:
    """Pre-PR-4 response shape (no note_to_agent) → empty field,
    composer falls back to detail."""
    from goldfive.drift.goals import classify_goal_drift

    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship it")],
        plan=None,
        observed_actions=[],
        model="m",
        call_llm=_stub_call_llm('{"progressing": false, "reason": "stalled out"}'),
    )
    assert drift is not None
    assert drift.note_to_agent == ""
    note = compose_note_for_drift(drift=drift)
    assert "goal drift detected: stalled out" in note


async def test_reasoning_judge_stamps_note_to_agent() -> None:
    from goldfive.drift import reasoning_judge as rjudge

    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thinking about raccoons",
        task=None,
        goals=[Goal(id="g1", summary="ship it")],
        model="m",
        call_llm=_stub_call_llm(
            '{"classification": "erroneous_deviation", "severity": "warning", '
            '"reason": "off task", "provenance": "none", '
            '"focused_task_id": "", "focus_confidence": 0.2, '
            '"stated_intent": "exploring raccoons", '
            '"note_to_agent": "The recent reasoning explored raccoons, '
            'which is not part of the recorded goal."}'
        ),
        current_task_id="t1",
    )
    assert verdict.note_to_agent == (
        "The recent reasoning explored raccoons, which is not part of "
        "the recorded goal."
    )
    assert verdict.drift is not None
    assert verdict.drift.note_to_agent == verdict.note_to_agent


async def test_reasoning_judge_old_style_response_degrades() -> None:
    """Legacy {"on_task": false} shape (no note field) → empty note."""
    from goldfive.drift import reasoning_judge as rjudge

    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thinking",
        task=None,
        goals=None,
        model="m",
        call_llm=_stub_call_llm(
            '{"on_task": false, "severity": "warning", "reason": "off"}'
        ),
        current_task_id="t1",
    )
    assert verdict.note_to_agent == ""
    assert verdict.drift is not None
    assert verdict.drift.note_to_agent == ""
    # The composed note still renders (detail fallback).
    note = compose_note_for_drift(drift=verdict.drift)
    assert note.startswith("Observation: reasoning drift: off")


async def test_reasoning_judge_ignores_note_on_on_task_verdicts() -> None:
    """A chatty model filling note_to_agent on on_task is ignored —
    healthy turns stay note-free (dormancy)."""
    from goldfive.drift import reasoning_judge as rjudge

    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thinking",
        task=None,
        goals=None,
        model="m",
        call_llm=_stub_call_llm(
            '{"classification": "on_task", "severity": "info", "reason": "fine", '
            '"provenance": "none", "note_to_agent": "spurious chatter"}'
        ),
        current_task_id="t1",
    )
    assert verdict.note_to_agent == ""
    assert verdict.drift is None


def test_judge_verdict_note_threads_onto_drift_event() -> None:
    """DefaultSteerer._drift_from_judge_verdict carries note_to_agent."""
    from goldfive.judges import JudgeVerdict
    from goldfive.steerer import DefaultSteerer

    steerer = DefaultSteerer()
    verdict = JudgeVerdict(
        drift_emitted=True,
        drift_kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="operator detail",
        note_to_agent="agent-facing observation",
    )
    drift = steerer._drift_from_judge_verdict(verdict, judge_name="j")
    assert drift is not None
    assert drift.note_to_agent == "agent-facing observation"


def test_judge_verdict_without_note_field_degrades() -> None:
    """Duck-typed verdicts lacking the attribute degrade to ''."""
    from goldfive.steerer import DefaultSteerer

    steerer = DefaultSteerer()
    verdict = SimpleNamespace(
        drift_emitted=True,
        drift_kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="d",
    )
    drift = steerer._drift_from_judge_verdict(verdict, judge_name="j")
    assert drift is not None
    assert drift.note_to_agent == ""
