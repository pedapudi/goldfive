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
