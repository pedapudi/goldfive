"""Optional embedding helpers for reasoning-based drift detection.

The helpers here light up when ``sentence-transformers`` is installed
(``goldfive[embedding]`` extra). When the dependency is absent every
function in this module returns ``None`` (or the "no-signal" value)
silently so the caller can fall through to pattern / hash heuristics.

Design notes
------------
* Import of the heavy ML stack is deferred to first call. Importing
  :mod:`goldfive.drift._embed` is always safe, even from minimal
  installs.
* The model is loaded at most once per process and cached in a module
  global. It stays loaded for the life of the process -- a 23 MB
  residue is cheap and avoids the 200 ms warm-up on every reasoning
  observation.
* No public API is exposed via :mod:`goldfive.drift.__init__`. These
  helpers are detector-private; swap the model by replacing the
  singleton via :func:`set_model` in tests or custom runtimes.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("goldfive.drift.reasoning.embed")


_MODEL: Any | None = None
_MODEL_UNAVAILABLE: bool = False
_DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"


def set_model(model: Any | None) -> None:
    """Install a caller-supplied encoder (tests, custom runtimes).

    ``None`` clears the cached model so the next call falls back to
    lazy loading. Callers that pass a custom encoder should ensure it
    exposes ``encode(list[str]) -> list[ndarray-like]``.
    """
    global _MODEL, _MODEL_UNAVAILABLE
    _MODEL = model
    _MODEL_UNAVAILABLE = False


def _get_model() -> Any | None:
    """Return the lazily-loaded sentence-transformers model, or ``None``.

    The first failed import flips ``_MODEL_UNAVAILABLE`` so subsequent
    calls skip the import cost. Callers must check for ``None``.
    """
    global _MODEL, _MODEL_UNAVAILABLE
    if _MODEL is not None:
        return _MODEL
    if _MODEL_UNAVAILABLE:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001 -- any import issue disables
        log.debug(
            "sentence-transformers not available (%s); install the "
            "`goldfive[embedding]` extra to enable semantic reasoning "
            "drift detection",
            exc,
        )
        _MODEL_UNAVAILABLE = True
        return None
    try:
        _MODEL = SentenceTransformer(_DEFAULT_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "failed to load sentence-transformers model %r: %s",
            _DEFAULT_MODEL_NAME,
            exc,
        )
        _MODEL_UNAVAILABLE = True
        return None
    return _MODEL


def available() -> bool:
    """Return True iff the embedding model can be loaded right now."""
    return _get_model() is not None


def _cosine(a: Any, b: Any) -> float:
    """Cosine similarity between two vector-like arrays, in ``[-1, 1]``.

    Uses NumPy when available (transitive dep of sentence-transformers)
    and falls back to a pure-Python dot product otherwise.
    """
    try:
        import numpy as np

        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)
    except Exception:  # noqa: BLE001
        pass
    try:
        la = [float(x) for x in a]
        lb = [float(x) for x in b]
    except Exception:  # noqa: BLE001
        return 0.0
    if len(la) != len(lb) or not la:
        return 0.0
    dot = sum(x * y for x, y in zip(la, lb, strict=False))
    na = sum(x * x for x in la) ** 0.5
    nb = sum(x * x for x in lb) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _encode(model: Any, text: str) -> Any | None:
    if not text:
        return None
    try:
        vecs = model.encode([text])
    except Exception as exc:  # noqa: BLE001
        log.debug("embedding encode failed: %s", exc)
        return None
    try:
        return vecs[0]
    except Exception:  # noqa: BLE001
        return None


def max_similarity(current: str, history: list[str]) -> float:
    """Return the maximum cosine similarity between ``current`` and
    any entry in ``history``, or ``0.0`` if embeddings are unavailable.

    Returns ``0.0`` on any encoding error; callers compare against the
    threshold, so the zero floor is safe (it never triggers a false
    loop detection).
    """
    if not current or not history:
        return 0.0
    model = _get_model()
    if model is None:
        return 0.0
    cur = _encode(model, current)
    if cur is None:
        return 0.0
    best = 0.0
    for past in history:
        past_vec = _encode(model, past)
        if past_vec is None:
            continue
        sim = _cosine(cur, past_vec)
        if sim > best:
            best = sim
    return best


def distance_to_topic(text: str, topic: str) -> float:
    """Return ``1 - cosine(text, topic)`` in ``[0, 2]``, or ``-1.0`` if
    embeddings are unavailable.

    Higher = further from the topic. ``-1.0`` lets callers distinguish
    "could not compute" from a genuine zero distance.
    """
    if not text or not topic:
        return -1.0
    model = _get_model()
    if model is None:
        return -1.0
    tv = _encode(model, text)
    topic_v = _encode(model, topic)
    if tv is None or topic_v is None:
        return -1.0
    return 1.0 - _cosine(tv, topic_v)
