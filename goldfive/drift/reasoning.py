"""Reasoning-based drift detectors.

Qwen3.5, Claude extended-thinking, and o1-style models expose their
chain-of-thought via ``reasoning_content`` / ``thinking`` blocks. The
adapter captures those blocks and hands them to
:meth:`goldfive.steerer.DefaultSteerer.observe_reasoning`, which runs
the pipeline defined here.

Five drift kinds live in this module:

* :data:`~goldfive.types.DriftKind.LOOPING_REASONING` -- consecutive
  reasoning blocks semantically identical (hash-exact fallback; cosine
  similarity >= 0.9 when the ``embedding`` extra is installed). The
  "cliff" tier that triggers refine.
* :data:`~goldfive.types.DriftKind.REASONING_CLUSTER_TIGHTENING` --
  graduated early-warning tier, fires when cosine similarity is in
  ``[0.75, 0.9)`` against any of the last N reasoning blocks. INFO
  severity; informational only, no refine. One-shot per task.
* :data:`~goldfive.types.DriftKind.CONFUSION` -- uncertainty markers
  in the reasoning text.
* :data:`~goldfive.types.DriftKind.OFF_TOPIC` -- reasoning topic is
  far from the task description (requires the ``embedding`` extra).
* :data:`~goldfive.types.DriftKind.INTENT_DIVERGENCE` -- reasoning
  has drifted away from ``session.goals`` + the current task. Severity
  is graduated by cosine distance when embeddings are available and
  falls back to a pattern-based WARNING when they are not.

The pipeline emits at most one drift per call to keep cost bounded.
Detectors run in severity order: INTENT_DIVERGENCE (up to CRITICAL) ->
LOOPING_REASONING (WARNING) -> OFF_TOPIC (WARNING) ->
REASONING_CLUSTER_TIGHTENING (INFO) -> CONFUSION (INFO).
When INTENT_DIVERGENCE resolves to a non-CRITICAL severity (INFO /
WARNING), the pipeline still returns it first — the kind is stable,
severity differentiates.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

from goldfive.drift import _embed
from goldfive.types import DriftEvent, DriftKind, DriftSeverity

if TYPE_CHECKING:
    from goldfive.types import Session, Task


log = logging.getLogger(__name__)


__all__ = [
    "CONFUSION_MARKERS",
    "CONFUSION_MIN_HITS",
    "INTENT_DIVERGENCE_HEALTHY_SIMILARITY",
    "INTENT_DIVERGENCE_MINOR_SIMILARITY",
    "INTENT_DIVERGENCE_WARNING_SIMILARITY",
    "LOOPING_REASONING_HASH_WINDOW",
    "LOOPING_REASONING_SIMILARITY_THRESHOLD",
    "OFF_TOPIC_DISTANCE_THRESHOLD",
    "REASONING_CLUSTER_SIMILARITY_THRESHOLD",
    "analyze_reasoning",
    "detect_confusion",
    "detect_intent_divergence",
    "detect_looping_reasoning",
    "detect_off_topic",
    "detect_reasoning_cluster_tightening",
    "reasoning_hash",
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


# Matches against uncertainty phrases in reasoning text. Three or more
# matches in a single block trips CONFUSION. Case-insensitive.
CONFUSION_MARKERS: re.Pattern[str] = re.compile(
    r"\b("
    r"i'?m not sure|"
    r"i don'?t know|"
    r"it'?s unclear|"
    r"should i |"
    r"need more info|"
    r"let me think(?: about this)? again|"
    r"wait,? |"
    r"actually,? |"
    r"hmm,? |"
    r"on second thought|"
    r"maybe i should|"
    r"i'?m confused"
    r")",
    re.IGNORECASE,
)

# Number of uncertainty-marker matches required for CONFUSION to fire.
CONFUSION_MIN_HITS: int = 3

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

    Either path may be bumped one step (INFO -> WARNING -> CRITICAL)
    when the reasoning text mentions a significant noun / keyword that
    does not appear in ``session.goals`` OR in the current task's
    title / description -- a cheap "talking about something unrelated"
    signal that catches soft divergence the cosine score alone may
    smooth over.

    The kind is stable. Severity differentiates -- callers that filter
    by kind see one signal, callers that care about urgency read the
    ``severity`` field.
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
            INTENT_DIVERGENCE_HEALTHY_SIMILARITY,
            INTENT_DIVERGENCE_MINOR_SIMILARITY,
            INTENT_DIVERGENCE_WARNING_SIMILARITY,
            text[:80],
        )
        severity = _severity_from_similarity(sim)
        if severity is None:
            return None
        if _has_unreferenced_keyword(text, goals_text, task_topic):
            severity = _bump_severity(severity)
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
        )

    # Pattern-based fallback (no embeddings).
    return _pattern_intent_divergence(text, session, goals_text, task_topic)


def _pattern_intent_divergence(
    text: str,
    session: Session,
    goals_text: str,
    task_topic: str,
) -> DriftEvent | None:
    """Return an INTENT_DIVERGENCE drift from regex-only signals.

    Fires at WARNING by default when an off-goal "focus on X" phrase
    sits next to tokens that do not appear in the goal summary. An
    unreferenced-keyword mismatch elsewhere in the text bumps severity
    to CRITICAL.
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
    if _has_unreferenced_keyword(text, goals_text, task_topic):
        severity = _bump_severity(severity)
    snippet = (text[max(0, match.start() - 40) : match.end() + 80]).strip()
    return DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=severity,
        detail=f"reasoning proposes off-goal focus: {snippet!r}",
        current_task_id=session.current_task_id,
        raw=text,
    )


def _severity_from_similarity(sim: float) -> DriftSeverity | None:
    """Map a cosine-similarity score to an INTENT_DIVERGENCE severity.

    Returns ``None`` for "healthy" scores (>= ``HEALTHY``) so the
    caller can suppress the drift entirely.
    """
    if sim >= INTENT_DIVERGENCE_HEALTHY_SIMILARITY:
        return None
    if sim >= INTENT_DIVERGENCE_MINOR_SIMILARITY:
        return DriftSeverity.INFO
    if sim >= INTENT_DIVERGENCE_WARNING_SIMILARITY:
        return DriftSeverity.WARNING
    return DriftSeverity.CRITICAL


def _bump_severity(sev: DriftSeverity) -> DriftSeverity:
    """Return the next-higher severity, saturating at CRITICAL."""
    if sev is DriftSeverity.INFO:
        return DriftSeverity.WARNING
    if sev is DriftSeverity.WARNING:
        return DriftSeverity.CRITICAL
    return DriftSeverity.CRITICAL


def _has_unreferenced_keyword(
    text: str, goals_text: str, task_topic: str
) -> bool:
    """Return True if the reasoning mentions a keyword not in goals or task.

    A "keyword" is any 5+ char alphabetic token from ``text`` that is
    not a stopword. If at least one such keyword is absent from both
    ``goals_text`` and ``task_topic`` (lower-cased, substring match),
    we treat the reasoning as talking about something unrelated.

    We require a 5-char minimum to avoid matching on generic English
    (``with``, ``from``); stopwords strip the common connectives that
    slip past the length gate. The check is deliberately conservative
    -- one odd token can bump severity by one step, never more.
    """
    if not text:
        return False
    reference = (goals_text + " " + task_topic).lower()
    if not reference.strip():
        return False
    for tok in re.findall(r"[a-z]{5,}", text.lower()):
        if tok in _STOPWORDS:
            continue
        if tok not in reference:
            return True
    return False


def detect_looping_reasoning(
    text: str, session: Session
) -> DriftEvent | None:
    """Return :data:`DriftKind.LOOPING_REASONING` when ``text``
    byte-identically or semantically matches a recent entry in
    ``session.reasoning_history``.

    The hash-based check is always on; the semantic check fires only
    when the embedding model is loadable. ``session.reasoning_history``
    is inspected but not mutated here -- the steerer appends the new
    reasoning block before running the pipeline.
    """
    history = [
        h for h in session.reasoning_history[-LOOPING_REASONING_HASH_WINDOW - 1 : -1]
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
            )
    sim = _embed.max_similarity(text, history)
    log.debug(
        "looping_reasoning: cosine=%.3f over %d prior turns (threshold=%.2f)",
        sim,
        len(history),
        LOOPING_REASONING_SIMILARITY_THRESHOLD,
    )
    if sim >= LOOPING_REASONING_SIMILARITY_THRESHOLD:
        return DriftEvent(
            kind=DriftKind.LOOPING_REASONING,
            severity=DriftSeverity.WARNING,
            detail=(
                f"reasoning semantically repeats (cosine={sim:.2f}) over "
                f"{len(history) + 1} turns"
            ),
            current_task_id=session.current_task_id,
            raw=text,
        )
    return None


def detect_reasoning_cluster_tightening(
    text: str, session: Session
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
    task_id = session.current_task_id or ""
    if task_id and task_id in session.reasoning_cluster_flagged:
        return None
    history = [
        h for h in session.reasoning_history[-LOOPING_REASONING_HASH_WINDOW - 1 : -1]
        if h
    ]
    if not history:
        return None
    sim = _embed.max_similarity(text, history)
    log.debug(
        "reasoning_cluster_tightening: cosine=%.3f over %d prior turns "
        "(band=[%.2f, %.2f))",
        sim,
        len(history),
        REASONING_CLUSTER_SIMILARITY_THRESHOLD,
        LOOPING_REASONING_SIMILARITY_THRESHOLD,
    )
    # max_similarity returns 0.0 both when the model is unavailable and
    # when the genuine cosine is zero; either way the early-warning tier
    # stays silent. This matches ``detect_off_topic``'s graceful-degrade
    # contract.
    if sim < REASONING_CLUSTER_SIMILARITY_THRESHOLD:
        return None
    if sim >= LOOPING_REASONING_SIMILARITY_THRESHOLD:
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
    )


def detect_off_topic(text: str, session: Session) -> DriftEvent | None:
    """Return :data:`DriftKind.OFF_TOPIC` when the reasoning vector is
    far from the current task topic vector.

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
    log.debug(
        "off_topic: distance=%.3f (threshold=%.2f); task=%r",
        dist,
        OFF_TOPIC_DISTANCE_THRESHOLD,
        topic[:60],
    )
    if dist < 0:
        return None
    if dist < OFF_TOPIC_DISTANCE_THRESHOLD:
        return None
    return DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail=(
            f"reasoning far from task (distance={dist:.2f} >= "
            f"{OFF_TOPIC_DISTANCE_THRESHOLD:.2f}): task={topic[:60]!r}"
        ),
        current_task_id=session.current_task_id,
        raw=text,
    )


def detect_confusion(text: str, session: Session) -> DriftEvent | None:
    """Return :data:`DriftKind.CONFUSION` when the reasoning text has
    at least :data:`CONFUSION_MIN_HITS` uncertainty markers.
    """
    if not text:
        return None
    hits = CONFUSION_MARKERS.findall(text)
    if len(hits) < CONFUSION_MIN_HITS:
        return None
    return DriftEvent(
        kind=DriftKind.CONFUSION,
        severity=DriftSeverity.INFO,
        detail=f"{len(hits)} uncertainty markers in reasoning",
        current_task_id=session.current_task_id,
        raw=text,
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def analyze_reasoning(text: str, session: Session) -> DriftEvent | None:
    """Run the reasoning-drift pipeline against ``text``.

    Emits at most one drift per call. Detectors are tried in the order
    that preserves the worst-signal-wins invariant:
    INTENT_DIVERGENCE (graduated INFO/WARNING/CRITICAL) ->
    LOOPING_REASONING (WARNING) -> OFF_TOPIC (WARNING) ->
    REASONING_CLUSTER_TIGHTENING (INFO) -> CONFUSION (INFO).

    INTENT_DIVERGENCE runs first even at INFO severity so its kind is
    stable; callers that only care about warning-and-up simply filter
    by ``severity``.

    LOOPING_REASONING (cosine >= 0.9) must run before
    REASONING_CLUSTER_TIGHTENING (0.75 <= cosine < 0.9) so that a
    tight-loop observation emits the cliff drift and never the INFO
    tier — the two are mutually exclusive by construction, and running
    LOOPING_REASONING first keeps the "no double-fire" invariant
    cheap to reason about.
    """
    if not text:
        return None
    drift = detect_intent_divergence(text, session)
    if drift is not None:
        return drift
    drift = detect_looping_reasoning(text, session)
    if drift is not None:
        return drift
    drift = detect_off_topic(text, session)
    if drift is not None:
        return drift
    drift = detect_reasoning_cluster_tightening(text, session)
    if drift is not None:
        return drift
    drift = detect_confusion(text, session)
    if drift is not None:
        return drift
    return None


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
