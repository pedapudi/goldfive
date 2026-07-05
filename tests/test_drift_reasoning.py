"""Unit tests for :mod:`goldfive.drift.reasoning` and the
``Steerer.observe_reasoning`` pipeline.

Covers:

* Pattern-based detectors (always on): INTENT_DIVERGENCE.
* Hash-based loop detection (always on).
* Embedding-based detectors (gated on ``sentence-transformers``).
* Steerer integration: observe_reasoning -> drift emission -> refine.
* ADK reasoning extraction from per-provider response shapes.
"""

from __future__ import annotations

import asyncio
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


async def _wait_for_judges(steerer: DefaultSteerer) -> None:
    """Drain background reasoning-judge tasks scheduled by the steerer.

    Since goldfive#251 :meth:`DefaultSteerer.observe_reasoning` routes
    the mode-selected pipeline (judge / embedding / both) through
    ``asyncio.create_task``; tests that assert sink state need to
    await the pending tasks before inspecting the sink.
    """
    pending = list(steerer._background_judges)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


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
    """Ensure each test starts with no custom encoder installed, and the
    lazy-load path disabled.

    ``set_model(None)`` alone resets the cached model but leaves the
    ``_MODEL_UNAVAILABLE`` flag False, so the next
    :func:`goldfive.drift._embed._get_model` call will attempt to import
    ``sentence-transformers`` and — when the ``embedding`` extra is
    installed — load the real MiniLM model. The fixture pins
    ``_MODEL_UNAVAILABLE = True`` too so the default environment is
    "no model"; tests that want the stub encoder call
    ``set_model(_StubEncoder())``, which flips the flag back.
    """
    from goldfive.drift import _embed as _embed_mod

    set_model(None)
    _embed_mod._MODEL_UNAVAILABLE = True
    yield
    set_model(None)
    _embed_mod._MODEL_UNAVAILABLE = True


# ---------------------------------------------------------------------------
# Pattern-based INTENT_DIVERGENCE (fallback path, no embedding model)
# ---------------------------------------------------------------------------


def test_intent_divergence_pattern_path_fires_when_goal_proposes_unrelated_focus() -> None:
    # No embedding model -> pattern-based fallback. Severity is flat
    # WARNING post goldfive#226 -- the historical keyword-mismatch bump
    # to CRITICAL was removed because the lexical heuristic fired on
    # generic English vocabulary absent from task descriptions.
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
    assert drift.severity is DriftSeverity.WARNING


def test_intent_divergence_pattern_path_stays_warning() -> None:
    # Pattern-path severity is flat WARNING in all cases post-#226.
    # This test pins the invariant explicitly: even when every 5+ char
    # token in the reasoning also appears in the task description
    # (i.e. the pre-#226 keyword-mismatch bump would have been
    # suppressed), the pattern path still emits WARNING.
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


def test_embedding_based_off_topic_silent_without_model(
    request: pytest.FixtureRequest,
) -> None:
    # No model installed; graceful degradation. Force the lazy-load
    # path off so the real sentence-transformers model cannot be
    # pulled in when the ``embedding`` extra happens to be active —
    # the scenario under test is explicitly "embedding stack absent."
    from goldfive.drift import _embed as _embed_mod

    _embed_mod.force_unavailable()
    request.addfinalizer(lambda: _embed_mod.set_model(None))
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


def test_intent_divergence_keyword_mismatch_does_not_bump_severity() -> None:
    import math

    # Post goldfive#226 the historical keyword-mismatch severity bump
    # was removed -- it fired on generic English vocabulary absent from
    # task descriptions and noisily promoted real embedding triggers to
    # spurious CRITICAL severities. Cosine bands alone determine
    # severity. Here cosine is 0.5 (INFO band), and the stray 5+ char
    # off-reference token "blockchain" must NOT bump the verdict.
    reasoning_text = "alpha bravo blockchain charlie"
    session = _intent_session(
        reasoning_text=reasoning_text,
        reasoning_angle=math.acos(0.5),
    )
    drift = dreason.detect_intent_divergence(reasoning_text, session)
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE
    # Severity stays at INFO (cosine band only, no keyword bump).
    assert drift.severity is DriftSeverity.INFO


# ---------------------------------------------------------------------------
# REASONING_CLUSTER_TIGHTENING — graduated early-warning tier below the
# LOOPING_REASONING cliff. Uses ``_StubEncoder`` for deterministic
# cosine values: each text becomes a unit-normalised count vector over
# hashed tokens, so for current/past blocks with disjoint-besides-shared
# token sets of size N each, ``cos = shared / N``.
# ---------------------------------------------------------------------------


def _pad_tokens(prefix: str, n: int) -> str:
    """Return a whitespace-joined string of ``n`` unique-ish tokens sharing
    ``prefix``. Token ids are zero-padded so tokens hash distinctly.
    """
    return " ".join(f"{prefix}{i:03d}" for i in range(n))


def _cluster_session() -> Session:
    """Return a session whose task topic already overlaps with the
    ``shared*`` tokens the cluster-tightening tests use, so OFF_TOPIC
    (which runs before REASONING_CLUSTER_TIGHTENING in the pipeline)
    stays silent and the tightening detector owns the signal.
    """
    topic = _pad_tokens("shared", 10)
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title=topic, description=topic)],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary=topic)],
        plan=plan,
        current_task_id="t1",
    )


def test_reasoning_cluster_tightening_fires_at_0_75() -> None:
    set_model(_StubEncoder())
    session = _cluster_session()
    # Current: 10 unique tokens (all shared-prefix).
    current = _pad_tokens("shared", 10)
    # Each of 5 priors shares 8 tokens with current and adds 2 unique
    # tokens -> cos = 8 / sqrt(10 * 10) = 0.8, inside [0.75, 0.9).
    priors = [
        _pad_tokens("shared", 8) + f" unique{block:02d}00 unique{block:02d}01"
        for block in range(5)
    ]
    # Steerer contract: current text lives at position -1; priors are
    # everything before it.
    session.reasoning_history = [*priors, current]
    drift = dreason.detect_reasoning_cluster_tightening(current, session)
    assert drift is not None
    assert drift.kind is DriftKind.REASONING_CLUSTER_TIGHTENING
    assert drift.severity is DriftSeverity.INFO
    assert "max cosine" in drift.detail
    assert "0.80" in drift.detail
    # The one-shot flag is now set on the session.
    assert session.current_task_id in session.reasoning_cluster_flagged


def test_reasoning_cluster_tightening_does_not_fire_below_0_75() -> None:
    set_model(_StubEncoder())
    session = _cluster_session()
    current = _pad_tokens("shared", 10)
    # Each prior shares 1 token with current -> cos = 1/sqrt(10*10) = 0.1,
    # well below the 0.75 floor.
    priors = [
        "shared000 " + _pad_tokens(f"alien{block:02d}", 9) for block in range(5)
    ]
    session.reasoning_history = [*priors, current]
    drift = dreason.detect_reasoning_cluster_tightening(current, session)
    assert drift is None
    assert session.current_task_id not in session.reasoning_cluster_flagged


async def test_reasoning_cluster_tightening_is_one_shot_per_task() -> None:
    set_model(_StubEncoder())
    # Cluster-tightening is an embedding-pipeline signal; engage it
    # explicitly. The default ``mode="judge"`` with no ``call_llm`` would
    # silently skip the embedding path.
    steerer = DefaultSteerer(reasoning_drift_mode="embedding")
    session = _cluster_session()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # Prime priors once; they stay in history throughout the loop. Each
    # prior has 10 tokens: 8 shared + 2 prior-specific unique tokens.
    priors = [
        _pad_tokens("shared", 8) + f" uniqueprior{i:02d}00 uniqueprior{i:02d}01"
        for i in range(5)
    ]
    session.reasoning_history = list(priors)
    # Keep the cap generous so priors + 10 observations all fit.
    session.reasoning_history_max = 100

    for turn in range(10):
        # Each observation has the same 8 shared tokens plus 2 turn-
        # specific unique tokens so cos(current, any_prior) = 8/sqrt(10*10)
        # = 0.8 deterministically -- in the tightening band on every
        # turn. Hashes differ because the uniques rotate, so the
        # hash-based LOOPING_REASONING path stays silent.
        current = (
            _pad_tokens("shared", 8) + f" uniqueturn{turn:02d}00 uniqueturn{turn:02d}01"
        )
        await steerer.drift.observe_reasoning(current, session=session)
    # The embedding pipeline is now fire-and-forget (goldfive#251);
    # drain before inspecting the sink.
    await _wait_for_judges(steerer)

    drift_events = [
        e for e in sink.events if e.WhichOneof("payload") == "drift_detected"
    ]
    # Exactly one INFO REASONING_CLUSTER_TIGHTENING drift across all 10
    # observations, and no spurious cliff drifts either.
    from goldfive.pb.goldfive.v1 import types_pb2

    tightening = [
        e
        for e in drift_events
        if e.drift_detected.kind == types_pb2.DRIFT_KIND_REASONING_CLUSTER_TIGHTENING
    ]
    assert len(tightening) == 1
    assert tightening[0].drift_detected.severity == types_pb2.DRIFT_SEVERITY_INFO
    looping = [
        e
        for e in drift_events
        if e.drift_detected.kind == types_pb2.DRIFT_KIND_LOOPING_REASONING
    ]
    assert looping == []


async def test_reasoning_high_similarity_skips_tightening_fires_loop() -> None:
    set_model(_StubEncoder())
    session = _cluster_session()
    current = _pad_tokens("shared", 10)
    # Priors share all 10 current tokens plus one distinctive filler:
    # cos = 10 / sqrt(10 * 11) ~= 0.953, well above the 0.9 cliff. The
    # filler keeps each prior's hash distinct from current so the
    # hash-based LOOPING_REASONING path stays silent and the semantic
    # >= 0.9 path owns the detection.
    priors = [
        _pad_tokens("shared", 10) + f" filler{block:02d}" for block in range(3)
    ]
    session.reasoning_history = [*priors, current]
    drift = await dreason.analyze_reasoning(current, session, mode="embedding")
    assert drift is not None
    # Cliff owns the signal; tightening must not steal it.
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.WARNING
    # And calling the tightening detector directly in this regime is a
    # no-op -- the cliff band is exclusive.
    tightening = dreason.detect_reasoning_cluster_tightening(current, session)
    assert tightening is None


def test_reasoning_cluster_skipped_when_embeddings_unavailable(
    request: pytest.FixtureRequest,
) -> None:
    # Simulate "sentence-transformers not installed" by forcing the
    # embed helper's lazy-load path to short-circuit to None, matching
    # the detector's graceful-degrade contract.
    from goldfive.drift import _embed as embed_mod

    embed_mod.force_unavailable()
    request.addfinalizer(lambda: embed_mod.set_model(None))

    session = _cluster_session()
    current = _pad_tokens("shared", 10)
    priors = [
        _pad_tokens("shared", 8) + f" unique{block:02d}00 unique{block:02d}01"
        for block in range(5)
    ]
    session.reasoning_history = [*priors, current]
    drift = dreason.detect_reasoning_cluster_tightening(current, session)
    assert drift is None


# ---------------------------------------------------------------------------
# Pipeline ordering: CRITICAL > WARNING > INFO
# ---------------------------------------------------------------------------


async def test_analyze_reasoning_returns_intent_divergence_on_off_goal_proposal() -> None:
    session = _session_with_task(
        goals=[Goal(id="g1", summary="write tax report")]
    )
    # An off-goal proposal triggers INTENT_DIVERGENCE (CRITICAL) ahead
    # of any other detector in the pipeline.
    text = (
        "Let me change goals -- actually, let's focus on building a "
        "video game instead. Forget about taxes."
    )
    drift = await dreason.analyze_reasoning(text, session, mode="embedding")
    assert drift is not None
    assert drift.kind is DriftKind.INTENT_DIVERGENCE


async def test_analyze_reasoning_returns_none_on_clean_text() -> None:
    # Every 5+ char non-stopword token in the reasoning also appears in
    # the default goals+task reference ("produce a slide review report"
    # + "Review the slides" + "Read every slide and list any typos
    # found.") so the standalone ``detect_unreferenced_keyword`` stays
    # silent alongside the intent-divergence / off-topic paths.
    session = _session_with_task()
    text = "slide review: read each slide, list the typos found."
    assert await dreason.analyze_reasoning(text, session, mode="embedding") is None


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
        await steerer.drift.observe_reasoning(f"thought {i}", session=session)

    assert session.reasoning_history == ["thought 2", "thought 3", "thought 4"]


async def test_observe_reasoning_emits_looping_drift_and_refines() -> None:
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    repeated = "I should re-check the slides for typos"
    # Prime the history so the next observation repeats.
    session.reasoning_history = [repeated, repeated, repeated]
    await steerer.drift.observe_reasoning(repeated, session=session)

    assert len(planner.refine_calls) == 1
    drift = planner.refine_calls[0]["drift"]
    assert drift.kind is DriftKind.LOOPING_REASONING


async def test_observe_reasoning_noops_on_empty_text() -> None:
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    await steerer.drift.observe_reasoning("", session=session)
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
# DriftDetected.trigger_input enrichment (judge-observability event)
# ---------------------------------------------------------------------------


def _drift_pb_events(sink: ListSink) -> list[Any]:
    """Filter sink.events to ``DriftDetected``-bearing proto events.

    LOOPING_REASONING is WARNING severity, so when refine fails the
    sink also carries a ``refine_failed`` dict envelope (see
    :func:`goldfive.events.make_event`). The pb-typed predicate filters
    to drift events only.
    """
    out: list[Any] = []
    for e in sink.events:
        # Dict envelopes (make_event) lack ``WhichOneof``; skip.
        if not hasattr(e, "WhichOneof"):
            continue
        if e.WhichOneof("payload") == "drift_detected":
            out.append(e)
    return out


async def test_observe_reasoning_populates_drift_trigger_input() -> None:
    """The always-on loop detector populates ``DriftDetected.trigger_input``.

    The always-on pattern detector doesn't construct ``trigger_input``
    itself — ``observe_reasoning`` fills it in from the reasoning text
    before emitting the drift. Harmonograf uses this to explain "why
    did goldfive flag this reasoning block?".
    """
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    text = "I should re-check the slides for typos"
    # Prime the history so the next observation repeats and the loop
    # detector fires.
    session.reasoning_history = [text, text, text]
    await steerer.drift.observe_reasoning(text, session=session)

    # LOOPING_REASONING is WARNING, so the steerer attempts refine. With
    # NullPlanner the refine returns None and the steerer escalates,
    # emitting a second DriftDetected with lifecycle=ESCALATING. Both
    # carry the same trigger_input — the assertion targets the OPENED
    # event for clarity.
    drift_events = _drift_pb_events(sink)
    assert len(drift_events) >= 1
    assert drift_events[0].drift_detected.trigger_input == text


async def test_observe_reasoning_truncates_long_trigger_input() -> None:
    """trigger_input is capped so long reasoning does not blow up sinks."""
    steerer = DefaultSteerer()
    session = _session_with_task()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # Pathological repeated block: 3 KB+, well over the 2048-char cap.
    huge = (
        "Reviewing the next slide for typos and double-checking the "
        "summary against the task description. "
    ) * 60
    # Prime the history so the loop detector fires on the same text.
    session.reasoning_history = [huge, huge, huge]
    await steerer.drift.observe_reasoning(huge, session=session)

    # See note in test_observe_reasoning_populates_drift_trigger_input:
    # WARNING-severity drift + NullPlanner refine triggers an
    # ESCALATING follow-up. Inspect the first (OPENED) drift.
    drift_events = _drift_pb_events(sink)
    assert len(drift_events) >= 1
    trigger_input = drift_events[0].drift_detected.trigger_input
    assert trigger_input.endswith(" … [truncated]")
    # At the 2048-char cap + suffix length.
    assert len(trigger_input) == 2048 + len(" … [truncated]")



# ---------------------------------------------------------------------------
# Pinned-history threading (background judge snapshot)
# ---------------------------------------------------------------------------


async def test_analyze_with_focus_honours_pinned_history() -> None:
    """``reasoning_history`` pins the detectors' window: entries appended
    to the live session list after the snapshot cannot self-match."""
    session = _session_with_task()
    text = "Let me re-examine the slide content and look for typos."
    # Snapshot at schedule time: ``text`` is the last (and only) entry.
    pinned = [text]
    # A later turn appends a byte-identical block to the live list.
    session.reasoning_history = [text, text]

    verdict = await dreason.analyze_reasoning_with_focus(
        text, session, mode="embedding", reasoning_history=pinned
    )
    assert verdict.drift is None

    # Without the pin the live list is read and the duplicate matches.
    verdict = await dreason.analyze_reasoning_with_focus(
        text, session, mode="embedding"
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.LOOPING_REASONING
