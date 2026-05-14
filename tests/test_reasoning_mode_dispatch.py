"""Mode-dispatch tests for :func:`goldfive.drift.reasoning.analyze_reasoning`.

For each of the four modes (``"judge"`` / ``"embedding"`` / ``"both"`` /
``"off"``) verify which detectors run, and that the
worst-severity-wins reconciliation in ``"both"`` is deterministic.

We monkeypatch the embedding helpers and stub ``call_llm`` so both
pipelines produce deterministic drifts; the ``both`` tests then flex
the severity ladder (INFO < WARNING < CRITICAL) to prove embedding
wins ties and judge wins strict higher severity.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import reasoning as dreason  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)


def _session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="Research solar panels", description="Find specs")],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id="t1",
    )


def _stub_judge_call_llm(responses: list[Any]):
    queue = list(responses)
    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        return json.dumps(resp) if isinstance(resp, (dict, list)) else str(resp)

    _call_llm.calls = calls  # type: ignore[attr-defined]
    return _call_llm


# ---------------------------------------------------------------------------
# mode="off"
# ---------------------------------------------------------------------------


async def test_mode_off_returns_none_regardless_of_input() -> None:
    """No embedding detector runs and no judge fires in off-mode."""
    embed_calls: list[str] = []

    def _embed_pipeline(text, session):
        embed_calls.append("embedding")
        return None

    judge_calls: list[str] = []

    async def _judge(text, session, **kwargs):
        judge_calls.append("judge")
        return None

    drift = await dreason.analyze_reasoning(
        "arbitrary reasoning",
        _session(),
        mode="off",
        call_llm=_stub_judge_call_llm([]),
        embedding_pipeline=_embed_pipeline,
        judge_classifier=_judge,
    )
    assert drift is None
    assert embed_calls == []
    assert judge_calls == []


# ---------------------------------------------------------------------------
# mode="embedding"
# ---------------------------------------------------------------------------


async def test_mode_embedding_runs_embedding_detectors_and_skips_judge() -> None:
    """``mode="embedding"`` invokes the embedding pipeline and never the
    judge.

    The embedding pipeline contract is "try every embedding detector
    in worst-signal-wins order"; we model that by handing
    ``analyze_reasoning`` an embedding pipeline stub that records its
    own invocation and short-circuits via the real detectors. The
    detector-level coverage lives in :file:`tests/test_drift_reasoning.py`;
    here we only care that the dispatch lands on the embedding side
    and not on the judge side.
    """
    embed_calls: list[str] = []

    def _embed_pipeline(text, session):
        embed_calls.append("embedding")
        return None

    judge_calls: list[str] = []

    async def _judge(text, session, **kwargs):
        judge_calls.append("judge")
        return None

    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="embedding",
        call_llm=_stub_judge_call_llm([{"on_task": False}]),
        embedding_pipeline=_embed_pipeline,
        judge_classifier=_judge,
    )
    assert drift is None
    assert embed_calls == ["embedding"]
    assert judge_calls == []


# ---------------------------------------------------------------------------
# mode="judge"
# ---------------------------------------------------------------------------


async def test_mode_judge_skips_embedding_and_runs_judge() -> None:
    embedding_calls: list[str] = []

    def _embed_pipeline(text, session):
        embedding_calls.append("embedding")
        return None

    call_llm = _stub_judge_call_llm([{"on_task": True}])
    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="judge",
        call_llm=call_llm,
        model="fake",
        embedding_pipeline=_embed_pipeline,
    )
    assert drift is None
    assert embedding_calls == []
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]


async def test_mode_judge_with_no_call_llm_silently_no_ops() -> None:
    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="judge", call_llm=None,
    )
    assert drift is None


# ---------------------------------------------------------------------------
# mode="both"
# ---------------------------------------------------------------------------


async def test_mode_both_embedding_wins_tie_on_equal_severity() -> None:
    """When both paths fire with the SAME severity, embedding wins
    (deterministic, synchronous path).
    """
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding path fired",
        current_task_id="t1",
    )
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "judge path fired"}]
    )

    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="both",
        call_llm=call_llm,
        model="fake",
        embedding_pipeline=lambda text, session: embedding_drift,
    )
    assert drift is embedding_drift


async def test_mode_both_judge_wins_higher_severity() -> None:
    """Judge drift at CRITICAL wins over embedding drift at WARNING."""
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding",
        current_task_id="t1",
    )
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "critical", "reason": "severe drift"}]
    )

    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="both",
        call_llm=call_llm,
        model="fake",
        embedding_pipeline=lambda text, session: embedding_drift,
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.CRITICAL
    assert "severe drift" in drift.detail  # came from judge


async def test_mode_both_embedding_wins_higher_severity() -> None:
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="embedding",
        current_task_id="t1",
    )
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "info", "reason": "mild"}]
    )

    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="both",
        call_llm=call_llm,
        model="fake",
        embedding_pipeline=lambda text, session: embedding_drift,
    )
    assert drift is embedding_drift


async def test_mode_both_judge_alone_when_embedding_silent() -> None:
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "off"}]
    )
    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="both",
        call_llm=call_llm,
        model="fake",
        embedding_pipeline=lambda text, session: None,
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.WARNING


async def test_mode_both_embedding_alone_when_judge_silent() -> None:
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding",
        current_task_id="t1",
    )
    call_llm = _stub_judge_call_llm([{"on_task": True}])
    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="both",
        call_llm=call_llm,
        model="fake",
        embedding_pipeline=lambda text, session: embedding_drift,
    )
    assert drift is embedding_drift


async def test_mode_both_without_call_llm_degrades_to_embedding() -> None:
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding",
        current_task_id="t1",
    )
    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="both",
        call_llm=None,
        embedding_pipeline=lambda text, session: embedding_drift,
    )
    assert drift is embedding_drift


# ---------------------------------------------------------------------------
# unknown modes fall back to embedding
# ---------------------------------------------------------------------------


async def test_unknown_mode_falls_back_to_embedding() -> None:
    fired: list[str] = []

    def _embed_pipeline(text, session):
        fired.append("embedding")
        return None

    drift = await dreason.analyze_reasoning(
        "text",
        _session(),
        mode="bogus",  # type: ignore[arg-type]
        embedding_pipeline=_embed_pipeline,
    )
    assert drift is None
    assert fired == ["embedding"]
