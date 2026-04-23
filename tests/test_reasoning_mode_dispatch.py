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


async def test_mode_off_returns_none_regardless_of_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No embedding detector runs and no judge fires in off-mode."""
    calls: list[str] = []

    def _track(name):
        def _f(text, session):
            calls.append(name)
            return None
        return _f

    for fn in (
        "detect_intent_divergence",
        "detect_looping_reasoning",
        "detect_off_topic",
        "detect_reasoning_cluster_tightening",
        "detect_confusion",
    ):
        monkeypatch.setattr(dreason, fn, _track(fn))

    judge_calls: list[str] = []

    async def _judge(**kwargs):
        judge_calls.append("judge")
        return None

    monkeypatch.setattr(dreason, "classify_reasoning_drift", _judge)

    drift = await dreason.analyze_reasoning(
        "arbitrary reasoning", _session(), mode="off", call_llm=_stub_judge_call_llm([])
    )
    assert drift is None
    assert calls == []
    assert judge_calls == []


# ---------------------------------------------------------------------------
# mode="embedding"
# ---------------------------------------------------------------------------


async def test_mode_embedding_runs_embedding_detectors_and_skips_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _track(name, returns=None):
        def _f(text, session):
            calls.append(name)
            return returns
        return _f

    for fn in (
        "detect_intent_divergence",
        "detect_looping_reasoning",
        "detect_off_topic",
        "detect_reasoning_cluster_tightening",
        "detect_confusion",
    ):
        monkeypatch.setattr(dreason, fn, _track(fn))

    judge_calls: list[str] = []

    async def _judge(**kwargs):
        judge_calls.append("judge")
        return None

    monkeypatch.setattr(dreason, "classify_reasoning_drift", _judge)

    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="embedding",
        call_llm=_stub_judge_call_llm([{"on_task": False}]),
    )
    assert drift is None
    # Every embedding detector got a chance (they all returned None).
    # ``detect_unreferenced_keyword`` is intentionally unwired post #226.
    assert calls == [
        "detect_intent_divergence",
        "detect_looping_reasoning",
        "detect_off_topic",
        "detect_reasoning_cluster_tightening",
        "detect_confusion",
    ]
    assert judge_calls == []


# ---------------------------------------------------------------------------
# mode="judge"
# ---------------------------------------------------------------------------


async def test_mode_judge_skips_embedding_and_runs_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_calls: list[str] = []
    for fn in (
        "detect_intent_divergence",
        "detect_looping_reasoning",
        "detect_off_topic",
        "detect_unreferenced_keyword",
        "detect_reasoning_cluster_tightening",
        "detect_confusion",
    ):
        def _maker(name):
            def _f(text, session):
                embedding_calls.append(name)
                return None
            return _f

        monkeypatch.setattr(dreason, fn, _maker(fn))

    call_llm = _stub_judge_call_llm([{"on_task": True}])
    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="judge", call_llm=call_llm, model="fake",
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


async def test_mode_both_embedding_wins_tie_on_equal_severity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both paths fire with the SAME severity, embedding wins
    (deterministic, synchronous path).
    """
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding path fired",
        current_task_id="t1",
    )
    monkeypatch.setattr(
        dreason,
        "_embedding_pipeline",
        lambda text, session: embedding_drift,
    )
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "judge path fired"}]
    )

    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="both", call_llm=call_llm, model="fake",
    )
    assert drift is embedding_drift


async def test_mode_both_judge_wins_higher_severity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge drift at CRITICAL wins over embedding drift at WARNING."""
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding",
        current_task_id="t1",
    )
    monkeypatch.setattr(
        dreason,
        "_embedding_pipeline",
        lambda text, session: embedding_drift,
    )
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "critical", "reason": "severe drift"}]
    )

    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="both", call_llm=call_llm, model="fake",
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.CRITICAL
    assert "severe drift" in drift.detail  # came from judge


async def test_mode_both_embedding_wins_higher_severity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="embedding",
        current_task_id="t1",
    )
    monkeypatch.setattr(
        dreason,
        "_embedding_pipeline",
        lambda text, session: embedding_drift,
    )
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "info", "reason": "mild"}]
    )

    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="both", call_llm=call_llm, model="fake",
    )
    assert drift is embedding_drift


async def test_mode_both_judge_alone_when_embedding_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dreason, "_embedding_pipeline", lambda text, session: None)
    call_llm = _stub_judge_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "off"}]
    )
    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="both", call_llm=call_llm, model="fake",
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.WARNING


async def test_mode_both_embedding_alone_when_judge_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding",
        current_task_id="t1",
    )
    monkeypatch.setattr(
        dreason,
        "_embedding_pipeline",
        lambda text, session: embedding_drift,
    )
    call_llm = _stub_judge_call_llm([{"on_task": True}])
    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="both", call_llm=call_llm, model="fake",
    )
    assert drift is embedding_drift


async def test_mode_both_without_call_llm_degrades_to_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="embedding",
        current_task_id="t1",
    )
    monkeypatch.setattr(
        dreason,
        "_embedding_pipeline",
        lambda text, session: embedding_drift,
    )
    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="both", call_llm=None,
    )
    assert drift is embedding_drift


# ---------------------------------------------------------------------------
# unknown modes fall back to embedding
# ---------------------------------------------------------------------------


async def test_unknown_mode_falls_back_to_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fired: list[str] = []
    monkeypatch.setattr(
        dreason,
        "_embedding_pipeline",
        lambda text, session: fired.append("embedding") or None,  # type: ignore[func-returns-value]
    )
    drift = await dreason.analyze_reasoning(
        "text", _session(), mode="bogus",  # type: ignore[arg-type]
    )
    assert drift is None
    assert fired == ["embedding"]
