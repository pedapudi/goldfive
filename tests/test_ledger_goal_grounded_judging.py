"""Goal-grounded judging in ledger mode (AGENCY-PRESERVATION.md Stage 3 PR 11).

New, ledger-gated judging behaviour. Forecast mode is unchanged (the
existing goal-drift / reasoning-judge suites pass unmodified); these
tests exercise only the additive ledger surface:

* (a) ``classify_goal_drift(graduated=True)`` — three-band verdict so
  WARNING (``uncertain``) can fire before CRITICAL (``off_track``), with
  the plan rendered as a kind-annotated LEDGER. Forecast mode
  (``graduated=False``) keeps the binary / always-CRITICAL behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift.goals import (  # noqa: E402
    GOAL_DRIFT_GRADUATED_USER_PROMPT_TEMPLATE,
    classify_goal_drift,
)
from goldfive.drift.outcome_progress import (  # noqa: E402
    evaluate_outcome_progress,
    plan_outcome_transitions,
)
from goldfive.drift.reasoning_judge import (  # noqa: E402
    classify_reasoning_drift_with_focus,
)
from goldfive.types import (  # noqa: E402
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
    TaskKind,
    TaskStatus,
)


def _ledger_plan() -> Plan:
    return Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(
            Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),
            Task(
                id="d1",
                title="reviewer: drafted notes",
                discovered=True,
                kind=TaskKind.DISCOVERED,
            ),
        ),
        edges=(),
    )


def _goals() -> list[Goal]:
    return [Goal(id="g", summary="summarise the deck")]


def _run(call_llm, *, graduated: bool):
    return asyncio.run(
        classify_goal_drift(
            goals=_goals(),
            plan=_ledger_plan(),
            observed_actions=[],
            model="m",
            call_llm=call_llm,
            graduated=graduated,
        )
    )


# ---------------------------------------------------------------------------
# (a) Graduated goal-drift
# ---------------------------------------------------------------------------


def test_graduated_uncertain_band_fires_warning() -> None:
    captured: dict[str, str] = {}

    async def llm(system: str, user: str, model: str) -> str:
        captured["user"] = user
        return (
            '{"progressing": false, "band": "uncertain", '
            '"reason": "connection to the goal is unclear", '
            '"note_to_agent": "Does the current approach still serve goal X?"}'
        )

    drift = _run(llm, graduated=True)
    assert drift is not None
    assert drift.kind is DriftKind.GOAL_DRIFT
    assert drift.severity is DriftSeverity.WARNING  # fires BEFORE critical
    assert drift.note_to_agent  # judge authored the advisory note
    # The graduated prompt renders the plan as a kind-annotated LEDGER.
    assert "LEDGER" in captured["user"]
    assert "OUTCOME" in captured["user"]
    assert "DISCOVERED" in captured["user"]


def test_graduated_off_track_band_fires_critical() -> None:
    async def llm(system: str, user: str, model: str) -> str:
        return '{"progressing": false, "band": "off_track", "reason": "looping"}'

    drift = _run(llm, graduated=True)
    assert drift is not None
    assert drift.severity is DriftSeverity.CRITICAL


def test_graduated_missing_band_degrades_to_critical() -> None:
    # A graduated judge that omits the band must NOT silently soften below
    # the legacy CRITICAL default.
    async def llm(system: str, user: str, model: str) -> str:
        return '{"progressing": false, "reason": "off"}'

    drift = _run(llm, graduated=True)
    assert drift is not None
    assert drift.severity is DriftSeverity.CRITICAL


def test_graduated_progressing_emits_no_drift() -> None:
    async def llm(system: str, user: str, model: str) -> str:
        return '{"progressing": true}'

    assert _run(llm, graduated=True) is None


def test_forecast_mode_unchanged_binary_critical() -> None:
    captured: dict[str, str] = {}

    async def llm(system: str, user: str, model: str) -> str:
        captured["user"] = user
        return '{"progressing": false, "reason": "off-topic"}'

    drift = _run(llm, graduated=False)
    assert drift is not None
    assert drift.severity is DriftSeverity.CRITICAL  # always critical, pre-PR-11
    # Forecast prompt is the binary template — no LEDGER framing.
    assert "LEDGER" not in captured["user"]


def test_graduated_template_has_three_bands() -> None:
    # Belt-and-suspenders: the graduated template must offer all three
    # verdict shapes so the judge can express the WARNING band.
    tmpl = GOAL_DRIFT_GRADUATED_USER_PROMPT_TEMPLATE
    assert '"progressing": true' in tmpl
    assert '"band": "uncertain"' in tmpl
    assert '"band": "off_track"' in tmpl


# ---------------------------------------------------------------------------
# (b) Reasoning judge re-grounding (goals primary, bound task as context)
# ---------------------------------------------------------------------------


def _reasoning_task() -> Task:
    return Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME)


def _on_task_verdict() -> str:
    return (
        '{"classification": "on_task", "severity": "info", "reason": "ok", '
        '"provenance": "none", "focused_task_id": "o1", '
        '"focus_confidence": 1.0, "stated_intent": "summarising", '
        '"note_to_agent": ""}'
    )


def test_reasoning_ledger_grounds_goals_primary_task_as_context() -> None:
    captured: dict[str, str] = {}

    async def llm(system: str, user: str, model: str) -> str:
        captured["user"] = user
        return _on_task_verdict()

    asyncio.run(
        classify_reasoning_drift_with_focus(
            reasoning="let me summarise the deck",
            task=_reasoning_task(),
            goals=_goals(),
            plan=_ledger_plan(),
            model="m",
            call_llm=llm,
            ledger=True,
        )
    )
    user = captured["user"]
    # GOALS lead; the bound task is explicitly framed as context only.
    assert user.index("GOALS") < user.index("CURRENTLY BOUND TASK")
    assert "PRIMARY reference" in user
    assert "CONTEXT ONLY" in user
    assert "LEDGER" in user


def test_reasoning_forecast_mode_unchanged_task_first() -> None:
    captured: dict[str, str] = {}

    async def llm(system: str, user: str, model: str) -> str:
        captured["user"] = user
        return _on_task_verdict()

    asyncio.run(
        classify_reasoning_drift_with_focus(
            reasoning="let me summarise the deck",
            task=_reasoning_task(),
            goals=_goals(),
            plan=_ledger_plan(),
            model="m",
            call_llm=llm,
            ledger=False,
        )
    )
    user = captured["user"]
    # Forecast template (pre-PR-11): bound task precedes goals.
    assert user.index("CURRENTLY BOUND TASK") < user.index("GOALS")
    assert "PRIMARY reference" not in user


def test_reasoning_ledger_verdict_shape_unchanged() -> None:
    # The ledger template keeps the IDENTICAL verdict JSON shape, so the
    # parser still produces an on-task (no drift) verdict.
    async def llm(system: str, user: str, model: str) -> str:
        return _on_task_verdict()

    verdict = asyncio.run(
        classify_reasoning_drift_with_focus(
            reasoning="summarising",
            task=_reasoning_task(),
            goals=_goals(),
            plan=_ledger_plan(),
            model="m",
            call_llm=llm,
            ledger=True,
        )
    )
    assert verdict.drift is None
    assert verdict.focused_task_id == "o1"


# ---------------------------------------------------------------------------
# (c) Outcome-progress judge + pure transition planning
# ---------------------------------------------------------------------------


def _outcome_ledger() -> Plan:
    return Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(
            Task(
                id="o1",
                title="Summary delivered",
                kind=TaskKind.OUTCOME,
                description="has summary",
            ),
            Task(id="o2", title="Translation delivered", kind=TaskKind.OUTCOME),
            Task(
                id="d1",
                title="writer: drafted summary",
                discovered=True,
                kind=TaskKind.DISCOVERED,
                status=TaskStatus.COMPLETED,
            ),
        ),
        edges=(),
    )


def _outcome_llm(payload: str):
    async def llm(system: str, user: str, model: str) -> str:
        return payload

    return llm


def test_outcome_judge_grades_deliverables_against_evidence() -> None:
    captured: dict[str, str] = {}

    async def llm(system: str, user: str, model: str) -> str:
        captured["user"] = user
        return (
            '{"outcomes": ['
            '{"task_id": "o1", "assessment": "met", "reason": "summary present", '
            '"contributing_task_ids": ["d1"]},'
            '{"task_id": "o2", "assessment": "pending", "reason": "not yet", '
            '"contributing_task_ids": []}]}'
        )

    verdicts = asyncio.run(
        evaluate_outcome_progress(
            goals=_goals(),
            plan=_outcome_ledger(),
            completed_outputs={"d1": "Full summary of the deck: ...."},
            model="m",
            call_llm=llm,
        )
    )
    by_id = {v.task_id: v for v in verdicts}
    assert by_id["o1"].met is True
    assert by_id["o1"].assessment == "met"
    assert by_id["o1"].contributing_task_ids == ("d1",)
    assert by_id["o2"].met is False
    assert by_id["o2"].assessment == "pending"
    # The prompt carried the goals, deliverables, trajectory, and evidence.
    assert "DELIVERABLES TO JUDGE" in captured["user"]
    assert "EVIDENCE" in captured["user"]
    assert "[d1]" in captured["user"]


def test_outcome_judge_no_outcome_tasks_skips_llm() -> None:
    async def boom(system: str, user: str, model: str) -> str:
        raise AssertionError("must not call the LLM when there are no OUTCOME tasks")

    plan = Plan(
        id="p", run_id="r", goal_ids=["g"], tasks=(Task(id="f1", title="forecast"),), edges=()
    )
    assert (
        asyncio.run(
            evaluate_outcome_progress(
                goals=_goals(), plan=plan, completed_outputs={}, model="m", call_llm=boom
            )
        )
        == []
    )


def test_outcome_judge_quiet_on_malformed_json() -> None:
    verdicts = asyncio.run(
        evaluate_outcome_progress(
            goals=_goals(),
            plan=_outcome_ledger(),
            completed_outputs={},
            model="m",
            call_llm=_outcome_llm("not json at all"),
        )
    )
    assert verdicts == []


def test_outcome_judge_ignores_unknown_outcome_ids() -> None:
    # A verdict for an id that is not a non-terminal OUTCOME is dropped.
    verdicts = asyncio.run(
        evaluate_outcome_progress(
            goals=_goals(),
            plan=_outcome_ledger(),
            completed_outputs={},
            model="m",
            call_llm=_outcome_llm(
                '{"outcomes": [{"task_id": "bogus", "assessment": "met"}, '
                '{"task_id": "o1", "assessment": "met", "contributing_task_ids": ["nope"]}]}'
            ),
        )
    )
    by_id = {v.task_id: v for v in verdicts}
    assert set(by_id) == {"o1"}
    # contributing id "nope" is not a DISCOVERED task → filtered out.
    assert by_id["o1"].contributing_task_ids == ()


def test_plan_transitions_met_completes_and_stamps_contributes_to() -> None:
    from goldfive.drift.outcome_progress import OutcomeVerdict

    verdicts = [
        OutcomeVerdict(
            task_id="o1", assessment="met", reason="done", contributing_task_ids=("d1",)
        ),
        OutcomeVerdict(task_id="o2", assessment="pending", reason="not yet"),
    ]
    tr = plan_outcome_transitions(_outcome_ledger(), verdicts, run_ending=False)
    assert len(tr) == 1
    assert tr[0].task_id == "o1"
    assert tr[0].new_status is TaskStatus.COMPLETED
    assert tr[0].contributes_stamps == (("d1", "o1"),)


def test_plan_transitions_confident_fail_only_at_run_end() -> None:
    from goldfive.drift.outcome_progress import OutcomeVerdict

    # CONFIDENTLY-unmet ("failed") → FAILED only at run end.
    verdicts = [OutcomeVerdict(task_id="o2", assessment="failed", reason="user cancelled it")]
    assert plan_outcome_transitions(_outcome_ledger(), verdicts, run_ending=False) == []
    tr = plan_outcome_transitions(_outcome_ledger(), verdicts, run_ending=True)
    assert len(tr) == 1
    assert tr[0].task_id == "o2"
    assert tr[0].new_status is TaskStatus.FAILED


def test_plan_transitions_pending_never_fails_even_at_run_end() -> None:
    from goldfive.drift.outcome_progress import OutcomeVerdict

    # #208 carry-forward: not-yet-met ("pending") is NOT failed at run end
    # — it stays PENDING for the next turn.
    verdicts = [OutcomeVerdict(task_id="o2", assessment="pending", reason="still working")]
    assert plan_outcome_transitions(_outcome_ledger(), verdicts, run_ending=True) == []


def test_plan_transitions_predicate_authoritative_blocks_completion() -> None:
    from goldfive.drift.outcome_progress import OutcomeVerdict

    verdicts = [OutcomeVerdict(task_id="o1", assessment="met", reason="llm says done")]
    # User predicate is explicitly unmet → the deterministic predicate
    # overrides the LLM and the outcome is NOT completed.
    assert (
        plan_outcome_transitions(
            _outcome_ledger(), verdicts, run_ending=False, goal_predicates_met=False
        )
        == []
    )


def test_plan_transitions_skips_terminal_outcomes() -> None:
    from goldfive.drift.outcome_progress import OutcomeVerdict

    plan = Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(Task(id="o1", title="done", kind=TaskKind.OUTCOME, status=TaskStatus.COMPLETED),),
        edges=(),
    )
    verdicts = [OutcomeVerdict(task_id="o1", assessment="failed")]
    assert plan_outcome_transitions(plan, verdicts, run_ending=True) == []
