"""Optional embedding helpers for reasoning-based drift detection.

Two encoder backends are supported; the right one is chosen lazily the
first time :func:`_get_model` is called.

1. **OpenAI-compatible HTTP backend** -- selected when the
   ``GOLDFIVE_EMBEDDING_BASE_URL`` environment variable is set. Posts
   to ``{BASE_URL}/v1/embeddings`` against any llama.cpp / Ollama /
   OpenAI-compatible endpoint the caller already has running for
   inference. Zero extra local install; the user's LLM server is the
   embedding server. Env vars:

   * ``GOLDFIVE_EMBEDDING_BASE_URL`` -- e.g. ``http://kikuchi.lan:8081``
     (no trailing slash; ``/v1/embeddings`` is appended).
   * ``GOLDFIVE_EMBEDDING_MODEL`` -- model name for the request body.
     Defaults to ``""``; most llama.cpp / Ollama servers accept the
     empty model (they use the single loaded model).
   * ``GOLDFIVE_EMBEDDING_API_KEY`` -- optional bearer token.
   * ``GOLDFIVE_EMBEDDING_TIMEOUT_MS`` -- HTTP timeout, default 10000.

2. **sentence-transformers backend** -- fallback when the env var is
   unset. Requires the ``goldfive[embedding]`` extra to be installed.

When neither backend is reachable every function in this module
returns the "no-signal" value (``0.0`` / ``-1.0``) silently so the
caller can fall through to pattern / hash heuristics.

Design notes
------------
* Import / network I/O is deferred to first call. Importing this
  module is always safe, even from minimal installs.
* The model is loaded at most once per process and cached in a
  module global.
* ``set_model()`` is the public escape hatch for tests and custom
  runtimes. It accepts any object exposing
  ``encode(list[str]) -> list[vector-like]`` and overrides both the
  env-driven and the sentence-transformers paths.
* Per-encode results are additionally memoised in a small LRU
  (:data:`_CACHE_MAX`) keyed on ``(backend_name, text)`` so that
  repeated comparisons against history entries do not re-pay the
  HTTP round-trip. The cache is process-local; drop it by calling
  :func:`set_model` or :func:`_reset_cache`.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

log = logging.getLogger("goldfive.drift.reasoning.embed")


_MODEL: Any | None = None
_MODEL_UNAVAILABLE: bool = False
_DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"

# LRU cache for per-text encoded vectors.
_CACHE_MAX: int = 512
_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()


def set_model(model: Any | None) -> None:
    """Install a caller-supplied encoder (tests, custom runtimes).

    ``None`` clears the cached model so the next call falls back to
    the lazy-load path (env-driven OpenAI backend, else
    sentence-transformers). Callers that pass a custom encoder should
    ensure it exposes ``encode(list[str]) -> list[ndarray-like]``.
    """
    global _MODEL, _MODEL_UNAVAILABLE
    _MODEL = model
    _MODEL_UNAVAILABLE = False
    _reset_cache()


def _reset_cache() -> None:
    """Drop the per-encode LRU cache. Called from :func:`set_model`."""
    _CACHE.clear()


def _get_model() -> Any | None:
    """Return the lazily-loaded encoder, or ``None``.

    Preference order (first match wins):

    1. A test- or caller-installed model via :func:`set_model`.
    2. An OpenAI-compatible HTTP backend when
       ``GOLDFIVE_EMBEDDING_BASE_URL`` is set in the environment.
    3. A ``sentence-transformers`` model loaded from the
       ``goldfive[embedding]`` extra.

    The first failed path flips ``_MODEL_UNAVAILABLE`` so subsequent
    calls skip the import cost. Callers must check for ``None``.
    """
    global _MODEL, _MODEL_UNAVAILABLE
    if _MODEL is not None:
        return _MODEL
    if _MODEL_UNAVAILABLE:
        return None

    base_url = os.environ.get("GOLDFIVE_EMBEDDING_BASE_URL", "").strip()
    if base_url:
        backend = _try_load_openai_backend(base_url)
        if backend is not None:
            _MODEL = backend
            return _MODEL
        # Env var set but backend failed to build: do NOT silently fall
        # through to sentence-transformers -- the user configured an
        # HTTP endpoint; honour that by flipping to unavailable so they
        # see "no-signal" instead of surprise-local-encoding.
        _MODEL_UNAVAILABLE = True
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001 -- any import issue disables
        log.debug(
            "sentence-transformers not available (%s); install the "
            "`goldfive[embedding]` extra or set "
            "GOLDFIVE_EMBEDDING_BASE_URL to use an OpenAI-compatible "
            "remote embedding endpoint",
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


def _try_load_openai_backend(base_url: str) -> Any | None:
    """Construct an :class:`_OpenAIEmbeddingBackend`, or return ``None``.

    Splitting this out lets tests assert backend construction without
    going through the full ``_get_model`` state machine.
    """
    model_name = os.environ.get("GOLDFIVE_EMBEDDING_MODEL", "")
    api_key = os.environ.get("GOLDFIVE_EMBEDDING_API_KEY") or None
    try:
        timeout_ms = int(os.environ.get("GOLDFIVE_EMBEDDING_TIMEOUT_MS", "10000"))
    except ValueError:
        timeout_ms = 10000
    try:
        return _OpenAIEmbeddingBackend(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            timeout_ms=timeout_ms,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "failed to build OpenAI-compatible embedding backend "
            "(base_url=%r): %s",
            base_url,
            exc,
        )
        return None


class _OpenAIEmbeddingBackend:
    """Encoder backed by an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Exposes the same ``encode(list[str]) -> list[list[float]]`` surface
    as the sentence-transformers adapter, so drop-in usage works.

    Uses the ``openai`` SDK when importable (for auth / retry
    consistency with the rest of the harmonograf client stack); falls
    back to raw ``httpx`` when it is not.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "",
        api_key: str | None = None,
        timeout_ms: int = 10000,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_s = max(0.1, timeout_ms / 1000.0)
        self._openai_client: Any | None = None
        self._httpx_client: Any | None = None
        self._prefer_sdk = self._try_build_openai_client()

    def _try_build_openai_client(self) -> bool:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            return False
        try:
            # The OpenAI SDK requires ``api_key`` to be a non-empty
            # string; llama.cpp servers don't check it, so pass a
            # placeholder when the user hasn't configured one.
            self._openai_client = OpenAI(
                base_url=f"{self._base_url}/v1",
                api_key=self._api_key or "not-needed",
                timeout=self._timeout_s,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("openai SDK client construction failed: %s", exc)
            self._openai_client = None
            return False

    def _get_httpx_client(self) -> Any | None:
        if self._httpx_client is not None:
            return self._httpx_client
        try:
            import httpx
        except Exception as exc:  # noqa: BLE001
            log.debug("httpx not importable for embedding backend: %s", exc)
            return None
        self._httpx_client = httpx.Client(timeout=self._timeout_s)
        return self._httpx_client

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text.

        Network / parse errors yield an empty list; callers upstream
        treat that as "no signal" and fall through to the 0.0 / -1.0
        defaults.
        """
        if not texts:
            return []
        if self._prefer_sdk and self._openai_client is not None:
            vectors = self._encode_via_sdk(texts)
            if vectors is not None:
                return vectors
            # SDK path failed -- fall through to raw httpx before giving up.
        return self._encode_via_httpx(texts) or []

    def _encode_via_sdk(self, texts: list[str]) -> list[list[float]] | None:
        assert self._openai_client is not None
        try:
            resp = self._openai_client.embeddings.create(
                model=self._model or "",
                input=texts,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("openai SDK embeddings.create failed: %s", exc)
            return None
        return _parse_openai_response(resp)

    def _encode_via_httpx(self, texts: list[str]) -> list[list[float]] | None:
        client = self._get_httpx_client()
        if client is None:
            return None
        url = f"{self._base_url}/v1/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {"input": texts}
        if self._model:
            payload["model"] = self._model
        else:
            # llama.cpp / Ollama tolerate ``model=""`` or a missing
            # field; always send an explicit key so servers that
            # *require* the field (strict OpenAI) complain loudly
            # rather than embedding silence.
            payload["model"] = ""
        try:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            body = r.json()
        except Exception as exc:  # noqa: BLE001
            log.debug("httpx embedding POST to %s failed: %s", url, exc)
            return None
        return _parse_openai_response(body)


def _parse_openai_response(resp: Any) -> list[list[float]] | None:
    """Extract vectors from an OpenAI ``/v1/embeddings`` response shape.

    Accepts either a dict (raw JSON from ``httpx``) or an SDK
    ``CreateEmbeddingResponse``-like object. Returns ``None`` if the
    response doesn't match the expected shape -- callers treat that as
    "no signal".

    Defensive against three real-world footguns:

    * ``data`` missing entirely (error responses: ``{"error": ...}``).
    * ``data[i].embedding`` being a nested list (some llama.cpp builds
      wrap it one level deep for pooled vs per-token embeddings).
    * ``embedding`` items being non-numeric (strings, ``None``).
    """
    try:
        data = resp["data"] if isinstance(resp, dict) else resp.data  # noqa: B009
    except Exception:  # noqa: BLE001
        log.debug("embedding response missing 'data' field: %r", resp)
        return None
    if not isinstance(data, list) or not data:
        log.debug("embedding response 'data' is empty or not a list")
        return None
    out: list[list[float]] = []
    for item in data:
        try:
            emb = item["embedding"] if isinstance(item, dict) else item.embedding
        except Exception:  # noqa: BLE001
            log.debug("embedding item missing 'embedding' field: %r", item)
            return None
        # Unwrap one level of nesting for servers that return
        # ``[[...]]`` instead of ``[...]``.
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb = emb[0]
        try:
            vec = [float(x) for x in emb]
        except Exception:  # noqa: BLE001
            log.debug("embedding item has non-numeric values: %r", emb)
            return None
        out.append(vec)
    return out


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


def _backend_name(model: Any) -> str:
    """Return a stable identifier used as the first half of the cache key.

    A different encoder identity => a different cache bucket, so that
    swapping the model via :func:`set_model` in tests cannot return
    stale vectors from an earlier run.
    """
    cls = type(model).__name__
    return f"{cls}:{id(model)}"


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


def _cached_encode(model: Any, text: str) -> Any | None:
    """LRU-cached wrapper around :func:`_encode`.

    The sentence-transformers path is cheap, but the OpenAI HTTP path
    pays a round-trip per call and :func:`max_similarity` re-encodes
    each history entry on every new observation. The cache holds at
    most :data:`_CACHE_MAX` entries and evicts oldest-first.
    """
    if not text:
        return None
    key = (_backend_name(model), text)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    vec = _encode(model, text)
    if vec is None:
        return None
    _CACHE[key] = vec
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return vec


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
    cur = _cached_encode(model, current)
    if cur is None:
        return 0.0
    best = 0.0
    for past in history:
        past_vec = _cached_encode(model, past)
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
    tv = _cached_encode(model, text)
    topic_v = _cached_encode(model, topic)
    if tv is None or topic_v is None:
        return -1.0
    return 1.0 - _cosine(tv, topic_v)
