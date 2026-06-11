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
