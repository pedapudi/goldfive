"""Reasoning-based drift detectors.

Qwen3.5, Claude extended-thinking, and o1-style models expose their
chain-of-thought via ``reasoning_content`` / ``thinking`` blocks. The
adapter captures those blocks and hands them to
:meth:`goldfive.steerer.DefaultSteerer.observe_reasoning`, which runs
the pipeline defined here.

Four drift kinds live in this module:

* :data:`~goldfive.types.DriftKind.LOOPING_REASONING` -- consecutive
  reasoning blocks semantically identical (hash-exact fallback; cosine
  similarity when the ``embedding`` extra is installed).
* :data:`~goldfive.types.DriftKind.CONFUSION` -- uncertainty markers
  in the reasoning text.
* :data:`~goldfive.types.DriftKind.OFF_TOPIC` -- reasoning topic is
  far from the task description (requires the ``embedding`` extra).
* :data:`~goldfive.types.DriftKind.INTENT_DIVERGENCE` -- reasoning
  mentions goals that are not in ``session.goals``.

The pipeline emits at most one drift per call to keep cost bounded.
Detectors run in severity order: INTENT_DIVERGENCE (CRITICAL) ->
LOOPING_REASONING (WARNING) -> OFF_TOPIC (WARNING) -> CONFUSION (INFO).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from goldfive.drift import _embed
from goldfive.types import DriftEvent, DriftKind, DriftSeverity

if TYPE_CHECKING:
    from goldfive.types import Session, Task


__all__ = [
    "CONFUSION_MARKERS",
    "CONFUSION_MIN_HITS",
    "LOOPING_REASONING_HASH_WINDOW",
    "LOOPING_REASONING_SIMILARITY_THRESHOLD",
    "OFF_TOPIC_DISTANCE_THRESHOLD",
    "analyze_reasoning",
    "detect_confusion",
    "detect_intent_divergence",
    "detect_looping_reasoning",
    "detect_off_topic",
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

# Cosine-distance threshold for OFF_TOPIC. ``1 - cosine >= threshold``
# means the reasoning is far from the task description.
OFF_TOPIC_DISTANCE_THRESHOLD: float = 0.7

# Regex looking for explicit "my goal is / let me change goals / new
# objective" style phrasing. Proof-by-wording only; embedding coverage
# is deferred to a follow-up.
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
    text proposes a goal that is not in ``session.goals``.

    Pattern-based only: looks for explicit "my goal is X / let's focus
    on Y" phrases and confirms that the mentioned focus does not
    overlap with any existing goal summary token. Conservative by
    design -- false positives here are costly (CRITICAL severity).
    """
    if not text:
        return None
    match = _INTENT_DIVERGENCE_MARKERS.search(text)
    if match is None:
        return None
    goals_text = _goals_text(session).lower()
    if not goals_text:
        # No goals to compare against -- cannot determine divergence
        # with any confidence. Skip rather than fire a CRITICAL.
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
    if any(tok in goals_text for tok in tokens):
        return None
    snippet = (text[max(0, match.start() - 40) : match.end() + 80]).strip()
    return DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail=f"reasoning proposes off-goal focus: {snippet!r}",
        current_task_id=session.current_task_id,
        raw=text,
    )


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

    Emits at most one drift per call. Detectors are tried in severity
    order so the worst signal wins: INTENT_DIVERGENCE (CRITICAL) ->
    LOOPING_REASONING (WARNING) -> OFF_TOPIC (WARNING) -> CONFUSION
    (INFO).
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
