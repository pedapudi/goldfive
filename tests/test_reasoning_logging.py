"""DEBUG-log assertions for the four embedding-driven drift detectors.

Operators reading the log should be able to distinguish "detector
didn't fire because the similarity was high" from "detector didn't
fire because the embedding model is unavailable." Each detector in
:mod:`goldfive.drift.reasoning` emits a single DEBUG line with the
cosine value and its threshold context whenever the embedding model
is consulted.

We install a fixed-similarity encoder so the cosine is deterministic
and can appear verbatim in the log assertion.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import reasoning as dreason  # noqa: E402
from goldfive.drift._embed import set_model  # noqa: E402
from goldfive.types import Goal, Plan, Session, Task  # noqa: E402


class _FixedSimilarityEncoder:
    """2-D unit-circle encoder: cosine between two registered texts
    equals ``cos(theta_a - theta_b)``."""

    def __init__(self) -> None:
        self._by_text: dict[str, tuple[float, float]] = {}

    def set(self, text: str, angle_rad: float) -> None:
        self._by_text[text] = (math.cos(angle_rad), math.sin(angle_rad))

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(self._by_text.get(t, (1.0, 0.0))) for t in texts]


def _session_basic(
    *,
    goals_summary: str = "produce a report on solar panels",
    title: str = "solar power panels",
    description: str = "write comparison of solar panels",
) -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title=title, description=description)],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary=goals_summary)],
        plan=plan,
        current_task_id="t1",
    )


@pytest.fixture(autouse=True)
def _reset_embed() -> Any:
    from goldfive.drift import _embed as embed_mod

    set_model(None)
    embed_mod._MODEL_UNAVAILABLE = True
    yield
    set_model(None)
    embed_mod._MODEL_UNAVAILABLE = True


# ---------------------------------------------------------------------------
# intent_divergence
# ---------------------------------------------------------------------------


def test_intent_divergence_logs_cosine(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reasoning_text = "alpha bravo charlie"
    goals_summary = "alpha bravo charlie delta"
    task_title = "echo foxtrot"
    task_description = "golf hotel india juliet"
    reference = f"{goals_summary} {task_title} {task_description}"

    encoder = _FixedSimilarityEncoder()
    encoder.set(reference, 0.0)
    # Use cos(theta)=0.8 -> healthy, no drift, but the DEBUG line still fires.
    encoder.set(reasoning_text, math.acos(0.8))
    set_model(encoder)

    session = _session_basic(
        goals_summary=goals_summary,
        title=task_title,
        description=task_description,
    )

    with caplog.at_level(logging.DEBUG, logger="goldfive.drift.reasoning"):
        dreason.detect_intent_divergence(reasoning_text, session)

    matching = [
        r
        for r in caplog.records
        if r.name == "goldfive.drift.reasoning"
        and "intent_divergence:" in r.getMessage()
    ]
    assert len(matching) == 1, [r.getMessage() for r in caplog.records]
    msg = matching[0].getMessage()
    assert "cosine=0.800" in msg
    assert "healthy=" in msg
    assert "text_head=" in msg


# ---------------------------------------------------------------------------
# off_topic
# ---------------------------------------------------------------------------


def test_off_topic_logs_distance(caplog: pytest.LogCaptureFixture) -> None:
    reasoning_text = "off topic reasoning"
    task_title = "solar power panels"
    task_description = "write comparison of solar panels"
    topic = f"{task_title} {task_description}"

    encoder = _FixedSimilarityEncoder()
    encoder.set(topic, 0.0)
    # distance = 0.2, below the 0.7 threshold: detector doesn't fire
    # but the DEBUG line must still land so operators can see why.
    encoder.set(reasoning_text, math.acos(0.8))
    set_model(encoder)

    session = _session_basic(title=task_title, description=task_description)

    with caplog.at_level(logging.DEBUG, logger="goldfive.drift.reasoning"):
        dreason.detect_off_topic(reasoning_text, session)

    matching = [
        r
        for r in caplog.records
        if r.name == "goldfive.drift.reasoning"
        and "off_topic:" in r.getMessage()
    ]
    assert len(matching) == 1, [r.getMessage() for r in caplog.records]
    msg = matching[0].getMessage()
    assert "distance=0.200" in msg
    assert "threshold=0.70" in msg
    assert "task=" in msg


# ---------------------------------------------------------------------------
# looping_reasoning (embedding tier)
# ---------------------------------------------------------------------------


def test_looping_reasoning_logs_cosine(caplog: pytest.LogCaptureFixture) -> None:
    current = "current reasoning text"
    past = "past reasoning text"

    encoder = _FixedSimilarityEncoder()
    encoder.set(current, 0.0)
    encoder.set(past, math.acos(0.5))  # cos = 0.5, below the 0.9 cliff.
    set_model(encoder)

    session = _session_basic()
    # Steerer contract: current lives at -1, priors before it.
    session.reasoning_history = [past, current]

    with caplog.at_level(logging.DEBUG, logger="goldfive.drift.reasoning"):
        dreason.detect_looping_reasoning(current, session)

    matching = [
        r
        for r in caplog.records
        if r.name == "goldfive.drift.reasoning"
        and "looping_reasoning:" in r.getMessage()
    ]
    assert len(matching) == 1, [r.getMessage() for r in caplog.records]
    msg = matching[0].getMessage()
    assert "cosine=0.500" in msg
    assert "prior turns" in msg
    assert "threshold=0.90" in msg


# ---------------------------------------------------------------------------
# reasoning_cluster_tightening
# ---------------------------------------------------------------------------


def test_cluster_tightening_logs_cosine(
    caplog: pytest.LogCaptureFixture,
) -> None:
    current = "current reasoning text"
    past = "past reasoning text"

    encoder = _FixedSimilarityEncoder()
    encoder.set(current, 0.0)
    # cos = 0.4 -- below the 0.75 cluster floor, so the detector
    # doesn't fire; the DEBUG line still logs.
    encoder.set(past, math.acos(0.4))
    set_model(encoder)

    session = _session_basic()
    session.reasoning_history = [past, current]

    with caplog.at_level(logging.DEBUG, logger="goldfive.drift.reasoning"):
        dreason.detect_reasoning_cluster_tightening(current, session)

    matching = [
        r
        for r in caplog.records
        if r.name == "goldfive.drift.reasoning"
        and "reasoning_cluster_tightening:" in r.getMessage()
    ]
    assert len(matching) == 1, [r.getMessage() for r in caplog.records]
    msg = matching[0].getMessage()
    assert "cosine=0.400" in msg
    assert "band=" in msg
