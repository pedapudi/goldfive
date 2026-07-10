"""Reasoning-based drift detectors.

Qwen3.5, Claude extended-thinking, and o1-style models expose their
chain-of-thought via ``reasoning_content`` / ``thinking`` blocks. The
adapter captures those blocks and hands them to
:meth:`goldfive.steerer.DefaultSteerer.observe_reasoning`, which runs
the pipeline defined here.

Four drift kinds live in this module:

* :data:`~goldfive.types.DriftKind.LOOPING_REASONING` -- consecutive
  reasoning blocks semantically identical (hash-exact fallback; cosine
  similarity >= 0.9 when the ``embedding`` extra is installed). The
  "cliff" tier that triggers refine.
* :data:`~goldfive.types.DriftKind.REASONING_CLUSTER_TIGHTENING` --
  graduated early-warning tier, fires when cosine similarity is in
  ``[0.75, 0.9)`` against any of the last N reasoning blocks. INFO
  severity; informational only, no refine. One-shot per task.
* :data:`~goldfive.types.DriftKind.OFF_TOPIC` -- reasoning topic is
  far from the task description (requires the ``embedding`` extra).
* :data:`~goldfive.types.DriftKind.INTENT_DIVERGENCE` -- reasoning
  has drifted away from ``session.goals`` + the current task. Severity
  is graduated by cosine distance when embeddings are available and
  falls back to a pattern-based WARNING when they are not.

The pipeline emits at most one drift per call to keep cost bounded.
Detectors run in severity order: INTENT_DIVERGENCE (up to CRITICAL) ->
LOOPING_REASONING (WARNING) -> OFF_TOPIC (WARNING) ->
REASONING_CLUSTER_TIGHTENING (INFO).
When INTENT_DIVERGENCE resolves to a non-CRITICAL severity (INFO /
WARNING), the pipeline still returns it first — the kind is stable,
severity differentiates.

Note: a regex-based ``CONFUSION`` detector (uncertainty-marker count)
used to live here. It was retired -- see ``DriftKind.CONFUSION`` (now
absent) for the rationale. The LLM-as-a-judge path covers the same
semantic ground robustly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from goldfive.drift import _embed
from goldfive.drift.reasoning_judge import (
    CallLLM as JudgeCallLLM,
)
from goldfive.drift.reasoning_judge import (
    ReasoningJudgeVerdict,
    classify_reasoning_drift,  # noqa: F401 — re-export consumed by tests
    classify_reasoning_drift_with_focus,
)
from goldfive.types import (
    RECENT_EVENT_KIND_TOOL_OBSERVED,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    filter_recent_events_by_kind,
)

if TYPE_CHECKING:
    from goldfive.config import ReasoningDriftConfig
    from goldfive.types import Session, Task


log = logging.getLogger(__name__)


# Pipeline-selection mode. See :func:`analyze_reasoning` for semantics.
#
# * ``"judge"``    — LLM-as-a-judge only (plus the always-on loop
#                    detector which lives upstream in
#                    :meth:`DefaultSteerer.observe_reasoning`).
# * ``"embedding"`` — the legacy embedding-based pipeline.
# * ``"both"``     — run both; the higher-severity drift wins. Ties
#                    are broken by the embedding path (runs first,
#                    synchronous).
# * ``"off"``      — no off-topic / intent / keyword checks. The
#                    always-on loop detector continues to run upstream.
ReasoningDriftMode = Literal["judge", "embedding", "both", "off"]
DEFAULT_REASONING_DRIFT_MODE: ReasoningDriftMode = "judge"


__all__ = [
    "DEFAULT_REASONING_DRIFT_MODE",
    "INTENT_DIVERGENCE_HEALTHY_SIMILARITY",
    "INTENT_DIVERGENCE_MINOR_SIMILARITY",
    "INTENT_DIVERGENCE_WARNING_SIMILARITY",
    "LOOPING_REASONING_HASH_WINDOW",
    "LOOPING_REASONING_SIMILARITY_THRESHOLD",
    "OFF_TOPIC_DISTANCE_THRESHOLD",
    "REASONING_CLUSTER_SIMILARITY_THRESHOLD",
    "ReasoningDriftMode",
    "SENTENCE_LEVEL_MIN_BLOCK_LENGTH",
    "SENTENCE_LEVEL_MAX_SENTENCES",
    "analyze_reasoning",
    "analyze_reasoning_with_focus",
    "detect_intent_divergence",
    "detect_looping_reasoning",
    "detect_off_topic",
    "detect_reasoning_cluster_tightening",
    "reasoning_hash",
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


# Window of prior reasoning blocks to compare against for loop detection.
LOOPING_REASONING_HASH_WINDOW: int = 5

# Cosine-similarity threshold for semantic loop detection. Lower values
# flag looser similarity as a loop; 0.9 keeps false positives rare.
LOOPING_REASONING_SIMILARITY_THRESHOLD: float = 0.9

# Lower-tier cosine-similarity threshold for the graduated early-warning
# signal ``REASONING_CLUSTER_TIGHTENING``. Fires in the half-open band
# ``[REASONING_CLUSTER_SIMILARITY_THRESHOLD, LOOPING_REASONING_SIMILARITY_THRESHOLD)``;
# above the upper bound the cliff detector (LOOPING_REASONING) owns the
# signal and the tightening tier stays quiet.
REASONING_CLUSTER_SIMILARITY_THRESHOLD: float = 0.75

# Cosine-distance threshold for OFF_TOPIC. ``1 - cosine >= threshold``
# means the reasoning is far from the task description.
OFF_TOPIC_DISTANCE_THRESHOLD: float = 0.7

# Minimum character length for ``detect_off_topic`` to additionally run
# the per-sentence min-cosine check. Below this threshold a block is
# likely a single sentence already; splitting buys nothing.
SENTENCE_LEVEL_MIN_BLOCK_LENGTH: int = 200

# Upper bound on how many sentences ``detect_off_topic`` checks per call
# when the sentence-level path is engaged. Keeps HTTP-backed embedding
# cost bounded for pathologically long reasoning blocks.
SENTENCE_LEVEL_MAX_SENTENCES: int = 10

# Sentence boundary: period / exclamation / question mark followed by
# whitespace. Pragmatic — see ``_split_sentences`` docstring for the
# tradeoffs vs more elaborate NLTK-style tokenisation.
_SENTENCE_BOUNDARY: re.Pattern[str] = re.compile(r"[.!?]\s+")

# Minimum length for a sentence-level candidate. Fragments shorter
# than this (e.g. ``"OK."``, ``"Step 1."``) can't carry enough lexical
# content for cosine to be meaningful.
_SENTENCE_MIN_LENGTH: int = 10

# Regex looking for explicit "my goal is / let me change goals / new
# objective" style phrasing. Used by the pattern-based fallback path
# when embeddings are unavailable, and also to surface off-goal
# proposals the embedding path may dilute over a long reasoning block.
_INTENT_DIVERGENCE_MARKERS: re.Pattern[str] = re.compile(
    r"\b("
    r"my (?:real )?goal (?:is|should be)|"
    r"new (?:goal|objective)|"
    r"(?:let(?:'s| us)|i should) (?:change|switch) (?:goals?|objectives?|tasks?)|"
    r"(?:actually|instead),? (?:let'?s|i should) (?:focus|work) on|"
    r"forget (?:the|this) (?:task|plan|goal)|"
    r"abandon (?:this|the) (?:task|plan|goal)|"
    r"pivot to"
    r")",
    re.IGNORECASE,
)


# Cosine-similarity bands for graduated INTENT_DIVERGENCE severity.
# Similarity is measured between the reasoning block and the goals +
# current task topic text.
#   sim >= HEALTHY            -> no drift
#   MINOR <= sim < HEALTHY    -> INFO
#   WARNING <= sim < MINOR    -> WARNING
#   sim < WARNING             -> CRITICAL
INTENT_DIVERGENCE_HEALTHY_SIMILARITY: float = 0.6
INTENT_DIVERGENCE_MINOR_SIMILARITY: float = 0.4
INTENT_DIVERGENCE_WARNING_SIMILARITY: float = 0.2


# ---------------------------------------------------------------------------
# Runtime-config installation (goldfive#225)
# ---------------------------------------------------------------------------
#
# Module-level ``_CONFIG`` + :func:`configure` form the per-Runner
# threshold override channel. When ``_CONFIG`` is ``None`` (default)
# the detectors read the module-level constants above — byte-identical
# to pre-#225 behaviour. When a :class:`~goldfive.config.ReasoningDriftConfig`
# is installed, the detectors read from it instead.
#
# Installation is **process-wide**: two Runners in one process share
# the last-installed config. :class:`~goldfive.steerer.DefaultSteerer.observe_reasoning`
# is the sole entry point to the pipeline, and the observed race
# (Runner A installs config, Runner B installs config, Runner A's
# observe_reasoning fires against Runner B's thresholds) is minor in
# practice — the thresholds are for heuristic drift detection, not
# correctness-critical limits.
#
# Alternative: attach the config to :class:`~goldfive.types.Session`
# and thread it through every free-function detector. ~5x LOC and
# requires touching every detector signature. Revisit if two-Runners-
# in-one-process with differing thresholds becomes a real use case.

_CONFIG: ReasoningDriftConfig | None = None


def configure(config: ReasoningDriftConfig | None) -> None:
    """Install a :class:`~goldfive.config.ReasoningDriftConfig` for this process.

    Called by :func:`goldfive.wrap` with ``runtime.reasoning_drift``.
    Passing ``None`` clears the override so detectors fall back to the
    module-level constants — used in test teardown to avoid cross-test
    leakage.
    """
    global _CONFIG
    _CONFIG = config


def _looping_hash_window() -> int:
    """Return the active LOOPING_REASONING hash-window size.

    Either the installed config's field or the module-level constant.
    Wrapping the lookup in a helper keeps the detector call sites
    terse and makes the config/no-config branch obvious in tests.
    """
    if _CONFIG is not None:
        return _CONFIG.looping_reasoning_hash_window
    return LOOPING_REASONING_HASH_WINDOW


def _looping_similarity_threshold() -> float:
    if _CONFIG is not None:
        return _CONFIG.looping_reasoning_similarity_threshold
    return LOOPING_REASONING_SIMILARITY_THRESHOLD


def _cluster_similarity_threshold() -> float:
    if _CONFIG is not None:
        return _CONFIG.reasoning_cluster_similarity_threshold
    return REASONING_CLUSTER_SIMILARITY_THRESHOLD


def _off_topic_distance_threshold() -> float:
    if _CONFIG is not None:
        return _CONFIG.off_topic_distance_threshold
    return OFF_TOPIC_DISTANCE_THRESHOLD


def _intent_healthy_similarity() -> float:
    if _CONFIG is not None:
        return _CONFIG.intent_divergence_healthy_similarity
    return INTENT_DIVERGENCE_HEALTHY_SIMILARITY


def _intent_minor_similarity() -> float:
    if _CONFIG is not None:
        return _CONFIG.intent_divergence_minor_similarity
    return INTENT_DIVERGENCE_MINOR_SIMILARITY


def _intent_warning_similarity() -> float:
    if _CONFIG is not None:
        return _CONFIG.intent_divergence_warning_similarity
    return INTENT_DIVERGENCE_WARNING_SIMILARITY


def _fallback_to_content_when_no_reasoning() -> bool:
    """Return the installed
    :attr:`~goldfive.config.ReasoningDriftConfig.fallback_to_content_when_no_reasoning`
    flag, or ``False`` when no config is installed (goldfive#263).

    Read by :func:`goldfive.adapters._adk_plugin._choose_reasoning_text`
    in the ADK plugin's ``after_model_callback`` to decide whether to
    synthesise a reasoning signal from the response body on non-thinking
    models. The flag lives on the reasoning-drift config (rather than
    being threaded through ``make_adk_plugin``) so it tracks the rest of
    the reasoning-drift surface and stays consistent with how the other
    fields propagate via :func:`configure`.
    """
    if _CONFIG is not None:
        return _CONFIG.fallback_to_content_when_no_reasoning
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reasoning_hash(text: str) -> str:
    """Return a short, stable SHA-256 prefix of the normalised reasoning
    text. Used for byte-identical loop detection without paying the
    encode cost of embeddings.
    """
    if not text:
        return ""
    normalised = " ".join(text.split()).strip().lower().encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()[:16]


def _current_task(session: Session) -> Task | None:
    plan = getattr(session, "plan", None)
    if plan is None:
        return None
    tid = getattr(session, "current_task_id", "") or ""
    if not tid:
        return None
    for t in plan.tasks:
        if t.id == tid:
            return t
    return None


def _task_topic(task: Task | None) -> str:
    if task is None:
        return ""
    parts = [task.title or "", task.description or ""]
    return " ".join(p for p in parts if p).strip()


def _goals_text(session: Session) -> str:
    return " ".join(g.summary for g in session.goals if g.summary).strip()


def _observed_revision_index(session: Session) -> int:
    """Return ``session.plan.revision_index`` or ``0`` when no plan exists.

    Captured at observation time by every reasoning-detector (goldfive#245)
    and stamped onto the produced :class:`DriftEvent`. The dispatch-time
    gate in :meth:`goldfive.steerer.DefaultSteerer._handle_drift` drops
    drifts whose observed revision is older than the live plan's, so a
    detector that observed against revision ``N`` cannot move the
    framework on a state-snapshot the reconciler already advanced past.
    """
    plan = getattr(session, "plan", None)
    if plan is None:
        return 0
    return int(getattr(plan, "revision_index", 0) or 0)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def detect_intent_divergence(
    text: str, session: Session
) -> DriftEvent | None:
    """Return :data:`DriftKind.INTENT_DIVERGENCE` when the reasoning
    text has drifted from ``session.goals`` + the current task topic.

    Graduated severity by cosine similarity (embedding path):

    =====================  ===========
    similarity             severity
    =====================  ===========
    >= 0.6                 no drift
    0.4 <= sim < 0.6       INFO
    0.2 <= sim < 0.4       WARNING
    < 0.2                  CRITICAL
    =====================  ===========

    When embeddings are unavailable we fall back to the pattern path:
    an explicit "my goal is / let's focus on" phrase whose proposal
    tokens do not overlap with any goal summary fires at WARNING.

    The kind is stable. Severity differentiates -- callers that filter
    by kind see one signal, callers that care about urgency read the
    ``severity`` field.

    .. note::

       The historical ``_has_unreferenced_keyword`` severity-bump that
       promoted one tier on any 5+ char token absent from goals + task
       was removed (goldfive#226/#230), and the helper itself has since
       been deleted. The lexical heuristic fired on generic English
       vocabulary not present in the task description, contaminating
       the embedding signal with noise.
    """
    if not text:
        return None

    goals_text = _goals_text(session)
    task_topic = _task_topic(_current_task(session))
    reference = " ".join(p for p in (goals_text, task_topic) if p).strip()
    if not reference:
        # No goals or task to compare against -- cannot determine
        # divergence with any confidence.
        return None

    # Embedding path -- graduated similarity bands. Only engage when the
    # encoder is loadable; otherwise fall through to the pattern path.
    if _embed.available():
        sim = _embed.max_similarity(text, [reference])
        log.debug(
            "intent_divergence: cosine=%.3f "
            "(thresholds healthy=%.2f minor=%.2f warning=%.2f); "
            "text_head=%r",
            sim,
            _intent_healthy_similarity(),
            _intent_minor_similarity(),
            _intent_warning_similarity(),
            text[:80],
        )
        severity = _severity_from_similarity(sim)
        if severity is None:
            return None
        # NOTE: the historical ``_has_unreferenced_keyword`` severity-bump
        # was removed. The lexical heuristic fired on generic English
        # vocabulary ("wants", "asking", "interactive", "slideshow") that
        # isn't in the task description, which bumped real embedding
        # triggers to spurious CRITICAL severities. The cosine band alone
        # now determines severity. See PR description rationale.
        detail = (
            f"reasoning diverged from goals (cosine={sim:.2f}): "
            f"reference={reference[:80]!r}"
        )
        return DriftEvent(
            kind=DriftKind.INTENT_DIVERGENCE,
            severity=severity,
            detail=detail,
            current_task_id=session.current_task_id,
            raw=text,
            observed_revision_index=_observed_revision_index(session),
        )

    # Pattern-based fallback (no embeddings).
    return _pattern_intent_divergence(text, session, goals_text, task_topic)


def _pattern_intent_divergence(
    text: str,
    session: Session,
    goals_text: str,
    task_topic: str,  # noqa: ARG001 -- kept for signature stability
) -> DriftEvent | None:
    """Return an INTENT_DIVERGENCE drift from regex-only signals.

    Fires at WARNING when an off-goal "focus on X" phrase sits next to
    tokens that do not appear in the goal summary.

    .. note::

       The historical ``_has_unreferenced_keyword`` bump that pushed
       severity to CRITICAL was removed. The lexical heuristic fired on
       generic English vocabulary and degraded the reliability of the
       real pattern trigger. Pattern-path severity is flat WARNING now.
       ``task_topic`` is retained in the signature for backward
       compatibility with callers that pass it positionally.
    """
    match = _INTENT_DIVERGENCE_MARKERS.search(text)
    if match is None:
        return None
    goals_lower = goals_text.lower()
    if not goals_lower:
        return None
    # Use the 120 chars after the marker as a cheap proxy for the
    # proposed new goal. If any significant token (>=4 chars) also
    # appears in the goals text we assume the model is restating, not
    # diverging.
    start = match.end()
    proposal = text[start : start + 120].lower()
    tokens = {
        tok
        for tok in re.findall(r"[a-z]{4,}", proposal)
        if tok not in _STOPWORDS
    }
    if not tokens:
        return None
    if any(tok in goals_lower for tok in tokens):
        return None
    severity = DriftSeverity.WARNING
    snippet = (text[max(0, match.start() - 40) : match.end() + 80]).strip()
    return DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=severity,
        detail=f"reasoning proposes off-goal focus: {snippet!r}",
        current_task_id=session.current_task_id,
        raw=text,
        observed_revision_index=_observed_revision_index(session),
    )


def _severity_from_similarity(sim: float) -> DriftSeverity | None:
    """Map a cosine-similarity score to an INTENT_DIVERGENCE severity.

    Returns ``None`` for "healthy" scores (>= ``HEALTHY``) so the
    caller can suppress the drift entirely. Thresholds are read from
    the installed :class:`~goldfive.config.ReasoningDriftConfig` when
    one is set (goldfive#225), else from the module-level constants.
    """
    if sim >= _intent_healthy_similarity():
        return None
    if sim >= _intent_minor_similarity():
        return DriftSeverity.INFO
    if sim >= _intent_warning_similarity():
        return DriftSeverity.WARNING
    return DriftSeverity.CRITICAL


def _bump_severity(sev: DriftSeverity) -> DriftSeverity:
    """Return the next-higher severity, saturating at CRITICAL."""
    if sev is DriftSeverity.INFO:
        return DriftSeverity.WARNING
    if sev is DriftSeverity.WARNING:
        return DriftSeverity.CRITICAL
    return DriftSeverity.CRITICAL


def detect_looping_reasoning(
    text: str, session: Session, history: Sequence[str] | None = None
) -> DriftEvent | None:
    """Return :data:`DriftKind.LOOPING_REASONING` when ``text``
    byte-identically or semantically matches a recent entry in
    ``session.reasoning_history``.

    The hash-based check is always on; the semantic check fires only
    when the embedding model is loadable. ``session.reasoning_history``
    is inspected but not mutated here -- the steerer appends the new
    reasoning block before running the pipeline. ``history`` overrides
    the session's live list (background judge runs pass the snapshot
    captured at schedule time — ``text`` is its last entry).
    """
    if history is None:
        history = session.reasoning_history
    hash_window = _looping_hash_window()
    history = [
        h for h in history[-hash_window - 1 : -1]
        if h
    ]
    if not history or not text:
        return None
    current_hash = reasoning_hash(text)
    for past in history:
        if current_hash and reasoning_hash(past) == current_hash:
            return DriftEvent(
                kind=DriftKind.LOOPING_REASONING,
                severity=DriftSeverity.WARNING,
                detail=(
                    f"reasoning block repeats (hash={current_hash}) over "
                    f"{len(history) + 1} turns"
                ),
                current_task_id=session.current_task_id,
                raw=text,
                observed_revision_index=_observed_revision_index(session),
            )
    sim = _embed.max_similarity(text, history)
    loop_threshold = _looping_similarity_threshold()
    log.debug(
        "looping_reasoning: cosine=%.3f over %d prior turns (threshold=%.2f)",
        sim,
        len(history),
        loop_threshold,
    )
    if sim >= loop_threshold:
        return DriftEvent(
            kind=DriftKind.LOOPING_REASONING,
            severity=DriftSeverity.WARNING,
            detail=(
                f"reasoning semantically repeats (cosine={sim:.2f}) over "
                f"{len(history) + 1} turns"
            ),
            current_task_id=session.current_task_id,
            raw=text,
            observed_revision_index=_observed_revision_index(session),
        )
    return None


def detect_reasoning_cluster_tightening(
    text: str, session: Session, history: Sequence[str] | None = None
) -> DriftEvent | None:
    """Return :data:`DriftKind.REASONING_CLUSTER_TIGHTENING` when recent
    reasoning blocks are semantically clustering tight -- max cosine
    similarity against the last :data:`LOOPING_REASONING_HASH_WINDOW`
    prior blocks falls in the half-open band
    ``[REASONING_CLUSTER_SIMILARITY_THRESHOLD,
    LOOPING_REASONING_SIMILARITY_THRESHOLD)``.

    This is the graduated early-warning tier below the LOOPING_REASONING
    "cliff" at 0.9: the agent's chain-of-thought is repeating concepts
    but has not yet collapsed into a loop. INFO severity, so sinks see
    it but the planner is not disturbed.

    Embedding-only -- skipped silently when the embedding model is
    unavailable (same rule as :func:`detect_off_topic`), because the
    signal is semantic tightening rather than byte-identical repetition.

    One-shot per task: the detector fires at most once for any given
    ``session.current_task_id`` value, tracked via
    ``session.reasoning_cluster_flagged``. Avoids drift-spam when a run
    stays in the tight-cluster regime for many consecutive turns.
    """
    if not text:
        return None
    if history is None:
        history = session.reasoning_history
    task_id = session.current_task_id or ""
    if task_id and task_id in session.reasoning_cluster_flagged:
        return None
    hash_window = _looping_hash_window()
    history = [
        h for h in history[-hash_window - 1 : -1]
        if h
    ]
    if not history:
        return None
    sim = _embed.max_similarity(text, history)
    cluster_threshold = _cluster_similarity_threshold()
    loop_threshold = _looping_similarity_threshold()
    log.debug(
        "reasoning_cluster_tightening: cosine=%.3f over %d prior turns "
        "(band=[%.2f, %.2f))",
        sim,
        len(history),
        cluster_threshold,
        loop_threshold,
    )
    # max_similarity returns 0.0 both when the model is unavailable and
    # when the genuine cosine is zero; either way the early-warning tier
    # stays silent. This matches ``detect_off_topic``'s graceful-degrade
    # contract.
    if sim < cluster_threshold:
        return None
    if sim >= loop_threshold:
        # Cliff tier owns this regime -- do not double-fire.
        return None
    if task_id:
        session.reasoning_cluster_flagged.add(task_id)
    return DriftEvent(
        kind=DriftKind.REASONING_CLUSTER_TIGHTENING,
        severity=DriftSeverity.INFO,
        detail=(
            f"recent reasoning clustering (max cosine={sim:.2f}); "
            "agent may be looping soon"
        ),
        current_task_id=session.current_task_id,
        raw=text,
        observed_revision_index=_observed_revision_index(session),
    )


def detect_off_topic(text: str, session: Session) -> DriftEvent | None:
    """Return :data:`DriftKind.OFF_TOPIC` when the reasoning is far from
    the current task topic.

    Two checks, in order:

    1. **Whole-text distance.** The original signal — still useful for
       blocks that are uniformly off-topic ("FAR-OFF" regime).
    2. **Per-sentence min-distance.** When ``text`` is long enough to
       likely contain multiple sentences, split on sentence terminators
       and embed each sentence individually. If ANY non-trivial sentence
       has ``distance_to_topic >= OFF_TOPIC_DISTANCE_THRESHOLD`` the
       pipeline fires. Rationale: on real embedding models the shared
       vocabulary of a long reasoning block swamps a brief drift tangent,
       so the whole-text cosine stays high even when the block clearly
       drifted (see #223 for the raccoon stimulus calibration). A single
       sentence is short enough that the drift tokens are a large
       fraction of the embedding, so the existing 0.7 threshold applies
       roughly unchanged; we deliberately don't introduce a new knob.

    Rate-limited to :data:`SENTENCE_LEVEL_MAX_SENTENCES` sentences per
    call to bound HTTP cost on the OpenAI-compatible backend.

    Requires the embedding extra. Returns ``None`` when the model is
    unavailable or when there is no bound current task to compare
    against.
    """
    if not text:
        return None
    topic = _task_topic(_current_task(session))
    if not topic:
        return None
    dist = _embed.distance_to_topic(text, topic)
    off_topic_threshold = _off_topic_distance_threshold()
    log.debug(
        "off_topic: distance=%.3f (threshold=%.2f); task=%r",
        dist,
        off_topic_threshold,
        topic[:60],
    )
    if dist >= 0 and dist >= off_topic_threshold:
        return DriftEvent(
            kind=DriftKind.OFF_TOPIC,
            severity=DriftSeverity.WARNING,
            detail=(
                f"reasoning far from task (distance={dist:.2f} >= "
                f"{off_topic_threshold:.2f}): task={topic[:60]!r}"
            ),
            current_task_id=session.current_task_id,
            raw=text,
            observed_revision_index=_observed_revision_index(session),
        )

    # Sentence-level path. Whole-text cosine came in either healthy or
    # unavailable; only engage per-sentence splitting when the block is
    # plausibly multi-sentence. A short single-sentence block has already
    # been tested above.
    if not _looks_multi_sentence(text):
        return None
    sentences = _split_sentences(text)[:SENTENCE_LEVEL_MAX_SENTENCES]
    candidates = [s for s in sentences if _is_sentence_candidate(s)]
    if not candidates:
        return None
    per_sentence: list[tuple[float, str]] = []
    worst: tuple[float, str] | None = None
    for sentence in candidates:
        sdist = _embed.distance_to_topic(sentence, topic)
        if sdist < 0:
            # Embedding unavailable mid-loop — match the whole-text
            # graceful-degrade contract.
            return None
        per_sentence.append((sdist, sentence))
        if worst is None or sdist > worst[0]:
            worst = (sdist, sentence)
    log.debug(
        "off_topic: sentence-level scan (%d sentences, threshold=%.2f): %s",
        len(per_sentence),
        off_topic_threshold,
        [(round(d, 3), s[:40]) for d, s in per_sentence],
    )
    if worst is None or worst[0] < off_topic_threshold:
        return None
    worst_dist, worst_sentence = worst
    return DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail=(
            f"reasoning has off-topic sentence (distance={worst_dist:.2f} "
            f">= {off_topic_threshold:.2f}): {worst_sentence[:80]!r}"
        ),
        current_task_id=session.current_task_id,
        raw=text,
        observed_revision_index=_observed_revision_index(session),
    )


def _looks_multi_sentence(text: str) -> bool:
    """Return True when ``text`` is long enough to plausibly contain
    multiple sentences. Either a char-count heuristic or two+ sentence
    terminators suffices — short blocks with a stray period (``"OK. "``)
    still get treated as single-sentence and fall through.
    """
    if len(text) > SENTENCE_LEVEL_MIN_BLOCK_LENGTH:
        return True
    terminators = sum(1 for ch in text if ch in ".!?")
    return terminators >= 2


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into rough sentences on terminator + whitespace.

    Trailing terminators are trimmed from each piece for readability in
    the diagnostic ``detail`` field. No attempt is made to handle
    abbreviations ("Dr."), ellipses, or quoted dialogue — the calibration
    corpus (#223) has none of those, and a heavier tokeniser trades
    accuracy for dependency weight we don't want here.
    """
    if not text:
        return []
    pieces = _SENTENCE_BOUNDARY.split(text)
    return [p.strip().rstrip(".!?").strip() for p in pieces if p.strip()]


def _is_sentence_candidate(sentence: str) -> bool:
    """Return True when ``sentence`` has enough lexical content for a
    cosine check to be meaningful. Filters trivial fragments like
    ``"OK"`` or ``"Step 1"`` that lack a single 5+ char alpha token —
    embedding them burns HTTP budget without producing signal.
    """
    if len(sentence) < _SENTENCE_MIN_LENGTH:
        return False
    return bool(re.search(r"[a-zA-Z]{5,}", sentence))


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


_SEVERITY_ORDER: dict[DriftSeverity, int] = {
    DriftSeverity.INFO: 0,
    DriftSeverity.WARNING: 1,
    DriftSeverity.CRITICAL: 2,
}


def _embedding_pipeline(
    text: str, session: Session, history: Sequence[str] | None = None
) -> DriftEvent | None:
    """Run the embedding-based pipeline (``mode="embedding"``).

    Detectors run in worst-signal-wins order:
    INTENT_DIVERGENCE -> LOOPING_REASONING -> OFF_TOPIC (with the
    sentence-level min-cosine path from #224) ->
    REASONING_CLUSTER_TIGHTENING. Ordering rationale lives in
    :func:`analyze_reasoning`. ``history`` overrides the live
    ``session.reasoning_history`` for the history-window detectors
    (background judge runs pass a snapshot pinned at schedule time).

    .. note::

       The historical lexical keyword detector
       (``detect_unreferenced_keyword``) was unwired here in
       goldfive#226 and has since been deleted. It fired on generic
       English vocabulary not present in the task description
       (``wants``, ``asking``, ``interactive``, ``slideshow``),
       producing noisy CRITICAL drifts on routine reasoning.
    """
    drift = detect_intent_divergence(text, session)
    if drift is not None:
        return drift
    drift = detect_looping_reasoning(text, session, history)
    if drift is not None:
        return drift
    drift = detect_off_topic(text, session)
    if drift is not None:
        return drift
    drift = detect_reasoning_cluster_tightening(text, session, history)
    if drift is not None:
        return drift
    return None


async def _run_judge(
    text: str,
    session: Session,
    *,
    call_llm: JudgeCallLLM,
    model: str,
    sink: Any = None,
    agent_name: str = "",
    available_agents: list[str] | list[dict[str, Any]] | None = None,
) -> DriftEvent | None:
    """Dispatch :func:`classify_reasoning_drift` against the current task.

    Returns just the drift component of the extended verdict. Steerer
    callers that need the attribution fields call
    :func:`_run_judge_with_focus` directly.

    When ``sink`` is provided it is forwarded into the classifier so a
    ``ReasoningJudgeInvoked`` event is emitted on every invocation
    regardless of verdict. ``run_id`` / ``session_id`` / a gap-free
    sequence are stamped from the session so downstream consumers can
    correlate the observability event with the session's other events.

    ``agent_name`` is the live ADK agent whose reasoning produced
    ``text``. Forwarded to the classifier as ``current_agent_id`` so
    delegated sub-agent reasoning (e.g. ``research_agent`` invoked by
    a coordinator) is correctly attributed on the resulting
    :class:`DriftEvent` and the ``ReasoningJudgeInvoked`` observability
    event. See goldfive#271 reasoning-judge delegated coverage.

    ``available_agents`` (goldfive#244) is the wrapped agent tree (in
    the shape exposed by
    :attr:`goldfive.adapters.adk.ADKAdapter.available_agents_tree`)
    forwarded into the judge prompt so legitimate coordinator → sub-
    agent delegation is treated as ON-TASK execution rather than an
    OFF_TOPIC deviation. ``None`` (the default) preserves the byte-
    identical pre-#244 prompt.
    """
    verdict = await _run_judge_with_focus(
        text,
        session,
        call_llm=call_llm,
        model=model,
        sink=sink,
        agent_name=agent_name,
        available_agents=available_agents,
    )
    return verdict.drift


async def _run_judge_with_focus(
    text: str,
    session: Session,
    *,
    call_llm: JudgeCallLLM,
    model: str,
    sink: Any = None,
    agent_name: str = "",
    available_agents: list[str] | list[dict[str, Any]] | None = None,
    ledger: bool = False,
) -> ReasoningJudgeVerdict:
    """Dispatch :func:`classify_reasoning_drift_with_focus` against the current task.

    Phase 1 of goldfive#271 — returns the full verdict (drift +
    attribution fields) so the steerer can record a reasoning-extracted
    binding onto :class:`~goldfive.state_store.StateStore`
    when ``focus_confidence`` clears the configured threshold.

    Same shape as :func:`_run_judge` for the LLM call itself; the only
    delta is the return type. Stamps ``plan`` so the prompt's
    plan-tasks-attribution section is populated.

    ``agent_name`` is forwarded to the classifier as ``current_agent_id``
    so reasoning produced by a delegated sub-agent (whose bound task
    is still the parent's) is attributed to the actual agent on the
    drift event and observability emission. See goldfive#271
    reasoning-judge delegated coverage.
    """
    task = _current_task(session)
    # iter-10 PR 3: surface lineage + recent tool observations to the
    # judge as additional context. ``task_lineage`` was added in iter-9
    # (#344); recent tool observations were added in iter-10 PR 2
    # (#347). Both are passed by attribute lookup so older Session
    # snapshots without the fields still parse cleanly (the helpers
    # treat None as "no data").
    # Goldfive#239: the dedicated ``recent_tool_observations`` buffer
    # was merged into :attr:`Session.recent_events`; filter to the
    # ``tool_observed`` kind here so the judge prompt sees the same
    # subset it saw pre-merge (the kwarg name is preserved for the
    # public ``classify_reasoning_drift*`` API).
    recent_events_attr = getattr(session, "recent_events", None)
    tool_obs = (
        filter_recent_events_by_kind(recent_events_attr, RECENT_EVENT_KIND_TOOL_OBSERVED)
        if recent_events_attr
        else None
    )
    return await classify_reasoning_drift_with_focus(
        reasoning=text,
        task=task,
        goals=list(session.goals),
        plan=getattr(session, "plan", None),
        model=model,
        call_llm=call_llm,
        current_task_id=session.current_task_id,
        current_agent_id=agent_name,
        sink=sink,
        run_id=session.run_id,
        session_id=session.id,
        sequence_fn=session.next_sequence,
        task_lineage=getattr(session, "task_lineage", None),
        recent_tool_observations=tool_obs,
        available_agents=available_agents,
        # AGENCY-PRESERVATION.md PR 11(b) — ledger re-grounding flag.
        ledger=ledger,
    )


async def analyze_reasoning_with_focus(
    text: str,
    session: Session,
    *,
    mode: ReasoningDriftMode = "embedding",
    call_llm: JudgeCallLLM | None = None,
    model: str = "",
    sink: Any = None,
    agent_name: str = "",
    available_agents: list[str] | list[dict[str, Any]] | None = None,
    embedding_pipeline: Any = None,
    judge_runner: Any = None,
    ledger: bool = False,
    reasoning_history: Sequence[str] | None = None,
) -> ReasoningJudgeVerdict:
    """Phase-1 sibling of :func:`analyze_reasoning` returning a focused verdict.

    Same selection logic as :func:`analyze_reasoning`; the only
    difference is the return type. When the judge path runs, callers
    get the full
    :class:`~goldfive.drift.reasoning_judge.ReasoningJudgeVerdict` —
    drift + ``focused_task_id`` + ``focus_confidence`` +
    ``stated_intent``. The legacy embedding pipeline, which has no
    judge, returns a verdict with the embedding-derived drift and
    empty attribution fields (the embedding pipeline can't attribute
    against the plan-tasks list).

    Modes:

    * ``"judge"`` — runs only the judge; verdict carries judge fields.
    * ``"embedding"`` — runs the embedding pipeline; verdict has
      ``drift`` set when off-topic and empty attribution fields.
    * ``"both"`` — both pipelines fire; the worst-severity drift
      wins (same tie-break as :func:`analyze_reasoning`); attribution
      fields come from the judge regardless of which drift won.
    * ``"off"`` — empty verdict.

    The legacy :func:`analyze_reasoning` is unchanged — it now
    delegates to this function and returns ``verdict.drift`` for
    back-compat with every existing caller.

    The ``embedding_pipeline`` and ``judge_runner`` parameters are test
    seams. ``embedding_pipeline`` is a callable
    ``(text, session) -> DriftEvent | None`` that replaces the default
    :func:`_embedding_pipeline`; tests use it to script the embedding
    side without monkeypatching detector symbols. ``judge_runner`` is
    an async callable with the same signature as
    :func:`_run_judge_with_focus`; tests use it to short-circuit the
    judge round-trip. Both default to ``None`` (use the in-module
    implementations) so production callers see no behaviour change.

    ``reasoning_history`` pins the view the history-window detectors
    see instead of the live ``session.reasoning_history`` — the
    background judge passes the snapshot captured at schedule time so
    entries appended by later turns cannot produce false self-match
    LOOPING signals. ``None`` (the default) reads the live session
    list. A caller-supplied ``embedding_pipeline`` keeps its
    ``(text, session)`` signature and is responsible for its own
    history handling.
    """
    if embedding_pipeline is not None:
        embed = embedding_pipeline
    else:

        def embed(t: str, s: Session) -> DriftEvent | None:
            return _embedding_pipeline(t, s, reasoning_history)

    judge = judge_runner if judge_runner is not None else _run_judge_with_focus
    if not text or mode == "off":
        return ReasoningJudgeVerdict(drift=None)
    if mode == "embedding":
        return ReasoningJudgeVerdict(drift=embed(text, session))
    if mode == "judge":
        if call_llm is None:
            return ReasoningJudgeVerdict(drift=None)
        return await judge(
            text,
            session,
            call_llm=call_llm,
            model=model,
            sink=sink,
            agent_name=agent_name,
            available_agents=available_agents,
            ledger=ledger,
        )
    if mode == "both":
        embedding_drift = embed(text, session)
        judge_verdict: ReasoningJudgeVerdict | None = None
        if call_llm is not None:
            judge_verdict = await judge(
                text,
                session,
                call_llm=call_llm,
                model=model,
                sink=sink,
                agent_name=agent_name,
                available_agents=available_agents,
                ledger=ledger,
            )
        if judge_verdict is None:
            return ReasoningJudgeVerdict(drift=embedding_drift)
        if embedding_drift is None:
            return judge_verdict
        # Both fired — worst-severity wins. Embedding wins ties
        # (deterministic, synchronous path). Either way, the
        # attribution fields come from the judge — the embedding
        # pipeline doesn't produce them.
        # ``dataclasses.replace`` (vs field-by-field reconstruction)
        # keeps every judge-derived field — attribution, classification /
        # provenance, and the judge-scheduling measurement fields
        # (``judge_ran`` / ``elapsed_ms``) — on the returned verdict
        # even when the embedding drift wins.
        if judge_verdict.drift is None:
            return dataclasses.replace(judge_verdict, drift=embedding_drift)
        if _SEVERITY_ORDER[judge_verdict.drift.severity] > _SEVERITY_ORDER[
            embedding_drift.severity
        ]:
            return judge_verdict
        return dataclasses.replace(judge_verdict, drift=embedding_drift)
    log.warning(
        "analyze_reasoning_with_focus: unknown mode=%r; falling back "
        "to 'embedding'",
        mode,
    )
    return ReasoningJudgeVerdict(drift=embed(text, session))


async def analyze_reasoning(
    text: str,
    session: Session,
    *,
    mode: ReasoningDriftMode = "embedding",
    call_llm: JudgeCallLLM | None = None,
    model: str = "",
    sink: Any = None,
    agent_name: str = "",
    available_agents: list[str] | list[dict[str, Any]] | None = None,
    embedding_pipeline: Any = None,
    judge_classifier: Any = None,
) -> DriftEvent | None:
    """Run the reasoning-drift pipeline against ``text``.

    Emits at most one drift per call. The behaviour is selected by
    ``mode``:

    * ``"embedding"`` — embedding-based pipeline. Detectors are tried
      in worst-signal-wins order: INTENT_DIVERGENCE ->
      LOOPING_REASONING -> OFF_TOPIC (whole-text + sentence-level
      min-cosine from #224) -> REASONING_CLUSTER_TIGHTENING.
      INTENT_DIVERGENCE runs first even at INFO severity so its kind
      is stable; callers that only care about warning-and-up simply
      filter by ``severity``. LOOPING_REASONING (cosine >= 0.9) runs
      before REASONING_CLUSTER_TIGHTENING (0.75 <= cosine < 0.9) so
      tight-loop observations emit the cliff drift and never the
      INFO tier.

    * ``"judge"`` — LLM-as-a-judge (goldfive#226). Dispatches to
      :func:`classify_reasoning_drift` with ``call_llm`` / ``model``.
      When ``call_llm`` is ``None`` the judge path silently no-ops so
      tests without a live LLM do not crash. The cheap orthogonal
      always-on loop detector runs upstream in
      :meth:`DefaultSteerer.observe_reasoning` in every mode.

    * ``"both"`` — run the embedding pipeline and the judge; the
      higher-severity drift wins. Tie-breaker is embedding (runs first,
      synchronously). When ``call_llm`` is ``None`` this degrades to
      ``"embedding"``.

    * ``"off"`` — skip the mode-selected pipeline entirely. The
      always-on loop detector continues to run upstream in
      :meth:`DefaultSteerer.observe_reasoning`.

    .. note::

       The historical keyword heuristic (``detect_unreferenced_keyword``)
       was unwired from every mode in goldfive#226 and has since been
       deleted. It fired on generic English vocabulary that isn't in the
       task description (real examples: ``wants``, ``asking``,
       ``interactive``, ``slideshow``), producing noisy CRITICAL drifts
       on routine reasoning.

    The ``embedding_pipeline`` and ``judge_classifier`` parameters are
    test seams. ``embedding_pipeline`` is a callable
    ``(text, session) -> DriftEvent | None`` substituted for
    :func:`_embedding_pipeline`; ``judge_classifier`` is an async
    callable with the signature of :func:`_run_judge`. Both default to
    ``None`` (use the in-module implementations) so production callers
    see no behaviour change.
    """
    embed = embedding_pipeline if embedding_pipeline is not None else _embedding_pipeline
    judge = judge_classifier if judge_classifier is not None else _run_judge
    if not text:
        return None
    if mode == "off":
        return None
    if mode == "embedding":
        return embed(text, session)
    if mode == "judge":
        if call_llm is None:
            return None
        return await judge(
            text,
            session,
            call_llm=call_llm,
            model=model,
            sink=sink,
            agent_name=agent_name,
            available_agents=available_agents,
        )
    if mode == "both":
        embedding_drift = embed(text, session)
        judge_drift: DriftEvent | None = None
        if call_llm is not None:
            judge_drift = await judge(
                text,
                session,
                call_llm=call_llm,
                model=model,
                sink=sink,
                agent_name=agent_name,
                available_agents=available_agents,
            )
        if embedding_drift is None:
            return judge_drift
        if judge_drift is None:
            return embedding_drift
        # Both fired — worst-severity wins. Embedding wins ties
        # (deterministic, synchronous path).
        if _SEVERITY_ORDER[judge_drift.severity] > _SEVERITY_ORDER[
            embedding_drift.severity
        ]:
            return judge_drift
        return embedding_drift
    # Unknown mode -- log and fall back to the legacy pipeline so the
    # run is never broken by a typo'd config value.
    log.warning(
        "analyze_reasoning: unknown mode=%r; falling back to 'embedding'",
        mode,
    )
    return embed(text, session)


# Small stopword set used by the intent-divergence token-overlap check.
# We only need to strip truly common English connective tokens; the
# rest is left permissive to avoid false negatives on CRITICAL drift.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "because",
        "before",
        "could",
        "does",
        "doesn",
        "every",
        "from",
        "just",
        "like",
        "more",
        "only",
        "other",
        "over",
        "same",
        "some",
        "than",
        "that",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "were",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "would",
        "your",
    }
)
