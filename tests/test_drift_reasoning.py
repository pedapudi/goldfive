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
    """Ensure each test starts with embeddings *unavailable*.

    ``set_model(None)`` clears any cached encoder, and flipping
    ``_MODEL_UNAVAILABLE`` true short-circuits the lazy import in
    :func:`goldfive.drift._embed._get_model`. Tests that want the
    embedding path install a stub via ``set_model(<encoder>)``, which
    resets the flag.
    """
    from goldfive.drift import _embed as _embed_mod

    set_model(None)
    _embed_mod._MODEL_UNAVAILABLE = True
    yield
    set_model(None)
    _embed_mod._MODEL_UNAVAILABLE = True


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
# Pattern-based INTENT_DIVERGENCE (fallback path, no embedding model)
# ---------------------------------------------------------------------------


def test_intent_divergence_pattern_path_fires_when_goal_proposes_unrelated_focus() -> None:
    # No embedding model -> pattern-based fallback. Default severity is
    # WARNING; a keyword mismatch elsewhere in the text bumps it to
    # CRITICAL (covered in the dedicated test below). The "cryptocurrency
    # trading dashboard" text here has several 5+ char tokens absent
    # from the goal summary, so severity bumps to CRITICAL.
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
    # Proposal tokens disjoint AND unrelated keywords present -> bumped
    # from WARNING up to CRITICAL.
    assert drift.severity is DriftSeverity.CRITICAL


def test_intent_divergence_pattern_path_warning_without_keyword_mismatch() -> None:
    # The unreferenced-keyword bump only fires on 5+ char non-stopword
    # tokens that appear in the reasoning but not in goals+task. Here
    # every 5+ char token in the reasoning also shows up in the task
    # description, so the bump is suppressed and severity stays at
    # WARNING. Meanwhile the proposal tokens ("tactics", "champion",
    # "shortly") are disjoint from the goal summary, so the pattern
    # detector still fires.
    session = _session_with_task(
        title="pivot actually focus report",
        description="pivot actually focus report tactics champion shortly goal",
        goals=[Goal(id="g1", summary="write a slide review report")],
    )
    text = "pivot to tactics champion shortly"
    drift = dreason.detect_intent_divergence(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE
    assert drift.severity is DriftSeverity.WARNING


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
    # No model installed; graceful degradation. The autouse fixture
    # already forces embeddings unavailable, but we re-assert here so
    # the intent of the test is explicit: ``set_model(None)`` alone
    # is not sufficient because it resets the lazy-load gate, which
    # then allows the real sentence-transformers model to load when
    # the ``embedding`` extra is installed.
    from goldfive.drift import _embed as _embed_mod

    set_model(None)
    _embed_mod._MODEL_UNAVAILABLE = True
    session = _session_with_task()
    text = "wildly unrelated reasoning content goes here"
    drift = dreason.detect_off_topic(text, session)
    assert drift is None


# ---------------------------------------------------------------------------
# Graduated INTENT_DIVERGENCE (embedding path)
#
# We use a fixed-similarity encoder so cosine(text, reference) is exactly
# ``cos(angle_text - angle_reference)``. This keeps the tests independent
# of the hash-bucket distribution in ``_StubEncoder``.
# ---------------------------------------------------------------------------


class _FixedSimilarityEncoder:
    """Encoder that maps registered texts to pre-chosen unit vectors.

    All vectors live on a 2-D unit circle so cosine between two texts
    equals ``cos(theta_a - theta_b)``. Unregistered texts land at angle
    0 (cosine 1 against each other). Use ``set(text, angle_rad)`` to
    control the angle per text.
    """

    def __init__(self) -> None:
        import math

        self._math = math
        self._by_text: dict[str, tuple[float, float]] = {}

    def set(self, text: str, angle_rad: float) -> None:
        self._by_text[text] = (self._math.cos(angle_rad), self._math.sin(angle_rad))

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(self._by_text.get(t, (1.0, 0.0))) for t in texts]


def _intent_session(
    *,
    reasoning_text: str,
    reasoning_angle: float,
    goals_summary: str = "alpha bravo charlie delta",
    task_title: str = "echo foxtrot",
    task_description: str = "golf hotel india juliet",
) -> Session:
    """Build a session whose reference-text vector sits at angle 0 and
    whose reasoning vector sits at ``reasoning_angle``. Cosine becomes
    ``cos(reasoning_angle)``.

    Default goals / task topic use NATO-style placeholder tokens so
    tests can craft a reasoning text whose 5+ char tokens all appear
    in the reference (suppressing the unreferenced-keyword bump) and
    dial severity purely through the encoder angle. Tests that want
    the keyword bump should override the defaults with a reasoning
    text that includes an off-reference token.
    """
    import math

    encoder = _FixedSimilarityEncoder()
    # The detector concatenates goals_summary + task_title + task_description
    # with single spaces (see ``reasoning._goals_text`` + ``_task_topic``).
    # We register the exact concatenation at angle 0.
    reference = " ".join(
        p for p in (goals_summary, f"{task_title} {task_description}") if p
    ).strip()
    encoder.set(reference, 0.0)
    encoder.set(reasoning_text, reasoning_angle)
    set_model(encoder)
    session = _session_with_task(
        title=task_title,
        description=task_description,
        goals=[Goal(id="g1", summary=goals_summary)],
    )
    # Sanity: make sure our calibration matches what the detector will
    # see. If this assert ever trips a reviewer has changed the
    # _goals_text / _task_topic concatenation contract.
    assert math.isclose(
        math.cos(reasoning_angle),
        dreason._embed.max_similarity(reasoning_text, [reference]),
        abs_tol=1e-6,
    )
    return session


def test_intent_divergence_healthy_above_0_6() -> None:
    import math

    # cos(theta) = 0.8 -> healthy, no drift. Every 5+ char token in
    # ``reasoning_text`` already appears in the reference, so the
    # keyword-mismatch bump cannot fire.
    reasoning_text = "alpha bravo charlie"
    session = _intent_session(
        reasoning_text=reasoning_text,
        reasoning_angle=math.acos(0.8),
    )
    drift = dreason.detect_intent_divergence(reasoning_text, session)
    assert drift is None


def test_intent_divergence_minor_fires_info_at_0_4_0_6() -> None:
    import math

    # cos(theta) = 0.5 -> in [0.4, 0.6) -> INFO.
    reasoning_text = "alpha bravo charlie"
    session = _intent_session(
        reasoning_text=reasoning_text,
        reasoning_angle=math.acos(0.5),
    )
    drift = dreason.detect_intent_divergence(reasoning_text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE
    assert drift.severity is DriftSeverity.INFO


def test_intent_divergence_warning_at_0_2_0_4() -> None:
    import math

    # cos(theta) = 0.3 -> in [0.2, 0.4) -> WARNING.
    reasoning_text = "alpha bravo charlie"
    session = _intent_session(
        reasoning_text=reasoning_text,
        reasoning_angle=math.acos(0.3),
    )
    drift = dreason.detect_intent_divergence(reasoning_text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE
    assert drift.severity is DriftSeverity.WARNING


def test_intent_divergence_critical_below_0_2() -> None:
    import math

    # cos(theta) = 0.0 -> < 0.2 -> CRITICAL.
    reasoning_text = "alpha bravo charlie"
    session = _intent_session(
        reasoning_text=reasoning_text,
        reasoning_angle=math.acos(0.0),
    )
    drift = dreason.detect_intent_divergence(reasoning_text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE
    assert drift.severity is DriftSeverity.CRITICAL


def test_intent_divergence_keyword_mismatch_upgrades_severity() -> None:
    import math

    # Cosine sits in the INFO band (0.5). The reasoning text mentions
    # "blockchain" -- a 5+ char non-stopword absent from both the goal
    # summary and the task topic -> severity bumps INFO -> WARNING.
    # All other 5+ char reasoning tokens DO appear in the reference, so
    # only the off-topic keyword drives the bump.
    reasoning_text = "alpha bravo blockchain charlie"
    session = _intent_session(
        reasoning_text=reasoning_text,
        reasoning_angle=math.acos(0.5),
    )
    drift = dreason.detect_intent_divergence(reasoning_text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE
    # INFO (cosine band) + unreferenced keyword "blockchain" -> WARNING.
    assert drift.severity is DriftSeverity.WARNING


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


