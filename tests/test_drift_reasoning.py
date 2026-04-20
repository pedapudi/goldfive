"""Unit tests for :mod:`goldfive.drift.reasoning` and the
``Steerer.observe_reasoning`` pipeline.

Covers:

* Pattern-based detectors (always on): CONFUSION, INTENT_DIVERGENCE.
* Hash-based loop detection (always on).
* Embedding-based detectors (gated on ``sentence-transformers``).
* Steerer integration: observe_reasoning -> drift emission -> refine.
* ADK reasoning extraction from per-provider response shapes.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import reasoning as dreason  # noqa: E402
from goldfive.drift._embed import set_model  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class NullPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return None


class RecordingPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        return None


def _session_with_task(
    task_id: str = "t1",
    *,
    title: str = "Review the slides",
    description: str = "Read every slide and list any typos found.",
    goals: list[Goal] | None = None,
) -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=task_id, title=title, description=description)],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=goals or [Goal(id="g1", summary="produce a slide review report")],
        plan=plan,
        current_task_id=task_id,
    )


@pytest.fixture(autouse=True)
def _clear_embedding_model() -> Any:
    """Ensure each test starts with no custom encoder installed."""
    set_model(None)
    yield
    set_model(None)


# ---------------------------------------------------------------------------
# Pattern-based CONFUSION
# ---------------------------------------------------------------------------


def test_confusion_fires_on_three_or_more_uncertainty_markers() -> None:
    text = (
        "Hmm, I'm not sure what to do here. I don't know if the user wants "
        "a summary. Wait, should I re-read the prompt?"
    )
    session = _session_with_task()
    drift = dreason.detect_confusion(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.CONFUSION
    assert drift.severity is DriftSeverity.INFO
    assert "markers" in drift.detail


def test_confusion_suppressed_below_threshold() -> None:
    text = "I'm not sure about this, but let's try."
    session = _session_with_task()
    drift = dreason.detect_confusion(text, session)
    assert drift is None


# ---------------------------------------------------------------------------
# Pattern-based INTENT_DIVERGENCE
# ---------------------------------------------------------------------------


def test_intent_divergence_fires_when_goal_proposes_unrelated_focus() -> None:
    session = _session_with_task(
        goals=[Goal(id="g1", summary="write a slide review report")]
    )
    text = (
        "Let me change goals -- actually, let's focus on building a "
        "cryptocurrency trading dashboard."
    )
    drift = dreason.detect_intent_divergence(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE
    assert drift.severity is DriftSeverity.CRITICAL


def test_intent_divergence_suppressed_when_focus_overlaps_goals() -> None:
    session = _session_with_task(
        goals=[Goal(id="g1", summary="write slide review report with examples")]
    )
    # The proposed focus reuses "slide review" tokens from the goal.
    text = "Actually, let's focus on the slide review examples in order."
    drift = dreason.detect_intent_divergence(text, session)
    assert drift is None


# ---------------------------------------------------------------------------
# Hash-based LOOPING_REASONING
# ---------------------------------------------------------------------------


def test_hash_based_loop_fires_on_byte_identical_repeat() -> None:
    session = _session_with_task()
    text = "Let me re-examine the slide content and look for typos."
    # Current reasoning lives at position -1 (the steerer appends before
    # analysis); older copies live before it.
    session.reasoning_history = [text, text, text]
    drift = dreason.detect_looping_reasoning(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.WARNING


def test_hash_based_loop_ignores_whitespace_and_case() -> None:
    session = _session_with_task()
    session.reasoning_history = [
        "Re-examine slides.",
        "Re-examine slides.",
        "re-examine   slides.",
    ]
    drift = dreason.detect_looping_reasoning("re-examine   slides.", session)
    assert drift is not None


def test_hash_based_loop_does_not_fire_without_history() -> None:
    session = _session_with_task()
    # Only current reasoning in history, nothing prior to compare to.
    session.reasoning_history = ["first reasoning"]
    drift = dreason.detect_looping_reasoning("first reasoning", session)
    assert drift is None


# ---------------------------------------------------------------------------
# Embedding-based detectors (gated on sentence-transformers)
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Deterministic vector encoder for tests.

    Maps each unique text to a vector by hashing tokens. Unrelated
    texts get near-orthogonal vectors; texts sharing many tokens get
    high cosine similarity.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        vocab_dim = 4096
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * vocab_dim
            for tok in t.lower().split():
                h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % vocab_dim
                vec[idx] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def test_embedding_based_loop_detects_semantic_duplicates() -> None:
    set_model(_StubEncoder())
    session = _session_with_task()
    # Match the steerer's contract: current text lives in history at
    # position -1 (appended just before analyze_reasoning runs).
    past = "I should check the slides again for typos"
    current = "should I check slides again typos for the I"
    session.reasoning_history = [past, current]
    drift = dreason.detect_looping_reasoning(current, session)
    assert drift is not None
    assert drift.kind is DriftKind.LOOPING_REASONING


def test_embedding_based_off_topic_fires_for_distant_reasoning() -> None:
    set_model(_StubEncoder())
    session = _session_with_task(
        title="Review slides", description="read slides list typos"
    )
    # Reasoning shares no tokens with the task -> distance ~1.0 >= 0.7.
    text = "Calculate the factorial sequence for quantum physics proofs"
    drift = dreason.detect_off_topic(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.OFF_TOPIC
    assert drift.severity is DriftSeverity.WARNING


def test_embedding_based_off_topic_no_fire_when_on_topic() -> None:
    set_model(_StubEncoder())
    session = _session_with_task(
        title="Review slides", description="read slides list typos"
    )
    text = "Review each slide and list typos for the report"
    drift = dreason.detect_off_topic(text, session)
    assert drift is None


def test_embedding_based_off_topic_silent_without_model() -> None:
    # No model installed; graceful degradation.
    set_model(None)
    session = _session_with_task()
    text = "wildly unrelated reasoning content goes here"
    drift = dreason.detect_off_topic(text, session)
    assert drift is None


# ---------------------------------------------------------------------------
# Pipeline ordering: CRITICAL > WARNING > INFO
# ---------------------------------------------------------------------------


def test_analyze_reasoning_prefers_intent_divergence_over_confusion() -> None:
    session = _session_with_task(
        goals=[Goal(id="g1", summary="write tax report")]
    )
    # Has uncertainty markers AND an off-goal proposal. INTENT_DIVERGENCE
    # wins because it is CRITICAL.
    text = (
        "Hmm, I'm not sure. Wait, let me change goals -- actually, let's "
        "focus on building a video game instead. I don't know about taxes."
    )
    drift = dreason.analyze_reasoning(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE


def test_analyze_reasoning_returns_none_on_clean_text() -> None:
    session = _session_with_task()
    text = "Focus: examine slide 3 for typos; log each typo with line number."
    assert dreason.analyze_reasoning(text, session) is None


# ---------------------------------------------------------------------------
# Steerer integration
# ---------------------------------------------------------------------------


async def test_observe_reasoning_appends_to_history_and_caps() -> None:
    steerer = DefaultSteerer()
    session = _session_with_task()
    session.reasoning_history_max = 3
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    for i in range(5):
        await steerer.observe_reasoning(f"thought {i}", session=session)

    assert session.reasoning_history == ["thought 2", "thought 3", "thought 4"]


async def test_observe_reasoning_emits_confusion_drift_but_no_refine() -> None:
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    text = (
        "Hmm, I'm not sure what the user wants. I don't know if they want "
        "a summary. Wait, should I re-check the prompt?"
    )
    await steerer.observe_reasoning(text, session=session)

    assert len(sink.events) == 1
    evt = sink.events[0]
    assert evt.WhichOneof("payload") == "drift_detected"
    # CONFUSION is INFO severity -> no refine.
    assert planner.refine_calls == []


async def test_observe_reasoning_emits_looping_drift_and_refines() -> None:
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    repeated = "I should re-check the slides for typos"
    # Prime the history so the next observation repeats.
    session.reasoning_history = [repeated, repeated, repeated]
    await steerer.observe_reasoning(repeated, session=session)

    assert len(planner.refine_calls) == 1
    drift = planner.refine_calls[0]["drift"]
    assert drift.kind is DriftKind.LOOPING_REASONING


async def test_observe_reasoning_noops_on_empty_text() -> None:
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    await steerer.observe_reasoning("", session=session)
    assert session.reasoning_history == []
    assert sink.events == []


# ---------------------------------------------------------------------------
# ADK reasoning extraction
# ---------------------------------------------------------------------------


def test_adk_extracts_thought_parts_from_content() -> None:
    from goldfive.adapters._adk_plugin import _extract_reasoning

    class Part:
        def __init__(self, text: str, thought: bool = False) -> None:
            self.text = text
            self.thought = thought

    class Content:
        def __init__(self, parts: list[Part]) -> None:
            self.parts = parts

    class Resp:
        def __init__(self, parts: list[Part]) -> None:
            self.content = Content(parts)

    resp = Resp(
        [
            Part("visible output"),
            Part("internal chain of thought", thought=True),
            Part("more visible text"),
        ]
    )
    assert _extract_reasoning(resp) == "internal chain of thought"


def test_adk_extracts_openai_reasoning_content() -> None:
    from goldfive.adapters._adk_plugin import _extract_reasoning

    class Msg:
        reasoning_content = "step-by-step cot from qwen"

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    assert _extract_reasoning(Resp()) == "step-by-step cot from qwen"


def test_adk_extracts_anthropic_thinking_block() -> None:
    from goldfive.adapters._adk_plugin import _extract_reasoning

    class Block:
        def __init__(self, type: str, thinking: str = "") -> None:
            self.type = type
            self.thinking = thinking

    class Resp:
        content = [Block("text"), Block("thinking", thinking="extended thoughts here")]

    assert _extract_reasoning(Resp()) == "extended thoughts here"


def test_adk_extract_reasoning_returns_empty_for_plain_text_response() -> None:
    from goldfive.adapters._adk_plugin import _extract_reasoning

    class Part:
        text = "hello"
        thought = False

    class Content:
        parts = [Part()]

    class Resp:
        content = Content()

    assert _extract_reasoning(Resp()) == ""


# ---------------------------------------------------------------------------
# Acceptance test from the issue: feed 3 "not sure" into DefaultSteerer,
# assert CONFUSION drift fires.
# ---------------------------------------------------------------------------


async def test_issue_acceptance_confusion_from_reasoning_block() -> None:
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    text = (
        "I'm not sure what to do. I'm not sure if the user wants a list. "
        "I'm not sure if I should stop here."
    )
    await steerer.observe_reasoning(text, session=session)

    # Exactly one drift event emitted, kind=CONFUSION, severity=INFO.
    kinds = [e for e in sink.events if e.WhichOneof("payload") == "drift_detected"]
    assert len(kinds) == 1
    evt = kinds[0]
    from goldfive.pb.goldfive.v1 import types_pb2

    assert evt.drift_detected.kind == types_pb2.DRIFT_KIND_CONFUSION
    assert evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_INFO


