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
   * ``GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S`` -- half-open retry delay,
     default 60 seconds.

   :class:`goldfive.config.EmbeddingConfig` exposes the same settings to
   callers that use :func:`goldfive.wrap`; an installed typed config wins
   over these direct-module environment fallbacks.

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
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldfive.config import EmbeddingConfig

log = logging.getLogger("goldfive.drift.reasoning.embed")


_MODEL: Any | None = None
_MODEL_UNAVAILABLE: bool = False
_DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Test seam: when non-None, ``_try_load_openai_backend`` delegates to
# this callable instead of constructing an :class:`_OpenAIEmbeddingBackend`
# itself. Production code never sets this; the test escape hatch
# :func:`set_backend_loader` does. Restoring ``None`` returns to the
# default constructor path.
_BACKEND_LOADER: Callable[[str], Any | None] | None = None

# Test seam: when non-None, ``_try_load_openai_backend`` builds via this
# class instead of :class:`_OpenAIEmbeddingBackend`. Lets tests verify
# the constructor kwargs without reaching into the module to rebind the
# class symbol. ``set_backend_class(None)`` restores the default.
_BACKEND_CLASS_OVERRIDE: Any | None = None

# Runtime circuit breaker for the OpenAI-compatible HTTP backend.
# Previously ``_MODEL_UNAVAILABLE`` flipped only on *import* failures
# (sentence-transformers missing). An HTTP backend whose endpoint was
# unreachable at runtime kept retrying forever, paying the timeout on
# every single call from :func:`max_similarity` and
# :func:`distance_to_topic`. The counter below trips the same flag
# after :data:`_RUNTIME_FAILURE_THRESHOLD` consecutive failures so a
# dead endpoint degrades to "no signal" after a bounded period of
# wasted I/O. A tripped breaker recovers via a timed half-open probe
# (see :func:`_get_model`) so a transient outage does not disable
# embeddings for the process lifetime. ``reset_circuit_breaker()``
# exposes a test escape hatch.
_RUNTIME_FAILURE_COUNT: int = 0
_RUNTIME_FAILURE_THRESHOLD: int = 3
_RUNTIME_FAILURE_TRIPPED: bool = False
# Monotonic timestamp of the trip (or the last failed half-open probe).
# ``None`` while the breaker is closed.
_RUNTIME_TRIPPED_AT: float | None = None
# Cooldown before a tripped breaker admits a half-open probe encode.
# ``GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S`` overrides at read time —
# same env pattern as ``GOLDFIVE_EMBEDDING_TIMEOUT_MS``.
_RUNTIME_RECOVERY_COOLDOWN_S: float = 60.0

# Installed per-Runner embedding config (goldfive#225). When non-None,
# the :func:`_get_model` lazy-load path reads backend parameters from
# this object instead of environment variables. ``None`` preserves the
# pre-#225 env-driven behaviour exactly.
_CONFIG: EmbeddingConfig | None = None

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
    reset_circuit_breaker()
    _reset_cache()


def set_backend_loader(loader: Callable[[str], Any | None] | None) -> None:
    """Test seam: override the OpenAI-backend constructor.

    When ``loader`` is non-None, :func:`_try_load_openai_backend`
    delegates to it instead of building an
    :class:`_OpenAIEmbeddingBackend` directly. The loader receives the
    resolved ``base_url`` and must return an encoder-shaped object
    exposing ``encode(list[str]) -> list[list[float]]``, or ``None`` to
    signal "could not build". Pass ``None`` to restore the default
    constructor path. Tests use this to assert which ``base_url`` the
    lazy-load path selects without monkeypatching module internals.
    """
    global _BACKEND_LOADER
    _BACKEND_LOADER = loader


def set_backend_class(cls: Any | None) -> None:
    """Test seam: override the backend class used by the default loader.

    When non-None, :func:`_try_load_openai_backend` calls ``cls(**kwargs)``
    instead of :class:`_OpenAIEmbeddingBackend`. Lets tests assert which
    constructor kwargs the loader pulls from env / config without
    monkeypatching the class symbol on this module. Pass ``None`` to
    restore the default class.
    """
    global _BACKEND_CLASS_OVERRIDE
    _BACKEND_CLASS_OVERRIDE = cls


def set_cache_max(n: int) -> None:
    """Test seam: shrink (or restore) the per-text LRU cache cap.

    Production callers should not touch this — the default 512-entry
    cap suits all real workloads. Tests use it to verify eviction
    behaviour without filling 513 entries per case.
    """
    global _CACHE_MAX
    _CACHE_MAX = int(n)
    # Trim any current entries beyond the new cap so the next encode
    # sees the freshly-constrained state.
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def force_unavailable() -> None:
    """Test seam: mark embeddings as unavailable without an HTTP probe.

    Equivalent to ``set_model(None)`` followed by tripping the
    ``_MODEL_UNAVAILABLE`` flag. Used by graceful-degradation tests
    that need ``_get_model()`` to return ``None`` regardless of env /
    config state. :func:`set_model` (with a non-None encoder) or
    :func:`reset_circuit_breaker` restores the lazy-load path.
    """
    global _MODEL, _MODEL_UNAVAILABLE
    _MODEL = None
    _MODEL_UNAVAILABLE = True


def reset_circuit_breaker() -> None:
    """Clear the runtime-failure counter and un-trip the circuit breaker.

    Test-only escape hatch. Prod code resets the counter on any
    successful encode via :meth:`_OpenAIEmbeddingBackend.encode`; tests
    that exercise the trip path need a deterministic way to reset
    between cases.

    Also clears :data:`_MODEL_UNAVAILABLE` when it was set by the trip
    (the counter was at the threshold); leaves it alone otherwise, so
    a sentence-transformers import failure isn't accidentally papered
    over.
    """
    global _RUNTIME_FAILURE_COUNT, _RUNTIME_FAILURE_TRIPPED, _MODEL_UNAVAILABLE
    global _RUNTIME_TRIPPED_AT
    if _RUNTIME_FAILURE_TRIPPED:
        _MODEL_UNAVAILABLE = False
    _RUNTIME_FAILURE_COUNT = 0
    _RUNTIME_FAILURE_TRIPPED = False
    _RUNTIME_TRIPPED_AT = None


def configure(config: EmbeddingConfig | None) -> None:
    """Install a :class:`~goldfive.config.EmbeddingConfig` for this process.

    Called by :func:`goldfive.wrap` with the ``runtime.embedding``
    dataclass. Once installed the env-based auto-config path in
    :func:`_get_model` is skipped — the config wins over env vars.
    Passing ``None`` reverts to env-driven behaviour.

    This also drops any cached encoder so the next :func:`_get_model`
    call re-enters the lazy-load path with the new configuration.
    :func:`set_model` (the test escape hatch) is NOT cleared — callers
    who want to replace a test-installed encoder with a config-driven
    one should call ``set_model(None)`` first.
    """
    global _CONFIG, _MODEL, _MODEL_UNAVAILABLE
    _CONFIG = config
    # Only flush the cached backend when the caller did not previously
    # install their own via ``set_model``. Tests that set a fake
    # encoder + also install a config expect the fake to keep winning.
    if _MODEL is None or not isinstance(_MODEL, _OpenAIEmbeddingBackend):
        pass  # keep a test-installed model alive
    else:
        _MODEL = None
    _MODEL_UNAVAILABLE = False
    reset_circuit_breaker()
    _reset_cache()


def _reset_cache() -> None:
    """Drop the per-encode LRU cache. Called from :func:`set_model`."""
    _CACHE.clear()


def _recovery_cooldown_s() -> float:
    """Return the configured half-open cooldown.

    A concrete value in an installed :class:`EmbeddingConfig` wins over the
    legacy environment fallback. ``None`` preserves that fallback for callers
    whose embedding config predates the typed cooldown field.
    """
    if _CONFIG is not None and _CONFIG.breaker_cooldown_s is not None:
        return _CONFIG.breaker_cooldown_s
    raw = os.environ.get("GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S", "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return _RUNTIME_RECOVERY_COOLDOWN_S


def _breaker_cooldown_elapsed() -> bool:
    """True when a *tripped* breaker has waited out its cooldown.

    Always False for a non-tripped ``_MODEL_UNAVAILABLE`` (import
    failure / :func:`force_unavailable`) — those states have no
    recovery path by design.
    """
    if not _RUNTIME_FAILURE_TRIPPED or _RUNTIME_TRIPPED_AT is None:
        return False
    return time.monotonic() - _RUNTIME_TRIPPED_AT >= _recovery_cooldown_s()


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
    global _MODEL, _MODEL_UNAVAILABLE, _RUNTIME_TRIPPED_AT
    # Check ``_MODEL_UNAVAILABLE`` before ``_MODEL`` so the runtime
    # circuit breaker (see :func:`_note_backend_failure`) actually
    # short-circuits. The tripped flag is also cleared by
    # :func:`reset_circuit_breaker` / :func:`set_model` /
    # :func:`configure` for callers that want a fresh start.
    if _MODEL_UNAVAILABLE:
        if not _breaker_cooldown_elapsed():
            return None
        # Half-open: admit one probe encode. A success closes the
        # breaker (:func:`_note_backend_success`); a failure re-opens
        # it (:func:`_note_backend_failure`). Restart the cooldown
        # clock now so a probe that dies during backend construction
        # (below) also waits a full cooldown before the next attempt.
        _RUNTIME_TRIPPED_AT = time.monotonic()
        _MODEL_UNAVAILABLE = False
        log.info(
            "embedding circuit breaker half-open after %.0fs cooldown; "
            "probing backend",
            _recovery_cooldown_s(),
        )
    if _MODEL is not None:
        return _MODEL

    # Prefer an installed RuntimeConfig over env vars (goldfive#225).
    # When ``configure()`` has been called with a non-None base_url,
    # its values win; otherwise we fall back to env lookup to preserve
    # the pre-#225 contract for callers that never touched the new API.
    base_url = ""
    if _CONFIG is not None and _CONFIG.base_url:
        base_url = _CONFIG.base_url.strip()
    else:
        base_url = os.environ.get("GOLDFIVE_EMBEDDING_BASE_URL", "").strip()
    if base_url:
        backend = _try_load_openai_backend(base_url)
        if backend is not None:
            _MODEL = backend
            return _MODEL
        # Base URL configured but backend failed to build: do NOT
        # silently fall through to sentence-transformers -- the user
        # configured an HTTP endpoint; honour that by flipping to
        # unavailable so they see "no-signal" instead of surprise-
        # local-encoding.
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

    Under goldfive#225, parameters are pulled from the installed
    :class:`~goldfive.config.EmbeddingConfig` when one is set; missing
    fields fall back to env vars, then to the built-in defaults — the
    same precedence as :func:`_get_model` uses for ``base_url``.
    """
    # Test seam: when a loader has been installed via
    # :func:`set_backend_loader`, delegate entirely. The loader is
    # responsible for returning an encoder-shaped object (or ``None``).
    if _BACKEND_LOADER is not None:
        return _BACKEND_LOADER(base_url)
    if _CONFIG is not None:
        model_name = _CONFIG.model
        api_key = _CONFIG.api_key
        timeout_ms = _CONFIG.timeout_ms
    else:
        model_name = os.environ.get("GOLDFIVE_EMBEDDING_MODEL", "")
        api_key = os.environ.get("GOLDFIVE_EMBEDDING_API_KEY") or None
        try:
            timeout_ms = int(os.environ.get("GOLDFIVE_EMBEDDING_TIMEOUT_MS", "10000"))
        except ValueError:
            timeout_ms = 10000
    # Test seam: override the backend class without rebinding it on the
    # module. Defaults to the real :class:`_OpenAIEmbeddingBackend`.
    backend_cls = _BACKEND_CLASS_OVERRIDE or _OpenAIEmbeddingBackend
    try:
        return backend_cls(
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

        Empty results also feed the module-level circuit breaker. After
        :data:`_RUNTIME_FAILURE_THRESHOLD` consecutive failures the
        :data:`_MODEL_UNAVAILABLE` flag trips, so subsequent calls
        short-circuit through :func:`_get_model` without paying the
        network timeout. Any successful call (non-empty vectors)
        resets the counter. See :func:`reset_circuit_breaker`.
        """
        if not texts:
            return []
        if self._prefer_sdk and self._openai_client is not None:
            vectors = self._encode_via_sdk(texts)
            if vectors is not None:
                _note_backend_success()
                return vectors
            # SDK path failed -- fall through to raw httpx before giving up.
        vectors = self._encode_via_httpx(texts)
        if vectors:
            _note_backend_success()
            return vectors
        _note_backend_failure(self._base_url)
        return []

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


def _note_backend_success() -> None:
    """Reset the runtime-failure counter on any successful encode.

    Called by :meth:`_OpenAIEmbeddingBackend.encode` whenever a
    non-empty vector list comes back. Keeping the reset in one place
    means a transient outage (two failures, one success, two more
    failures) never trips the circuit breaker — only *consecutive*
    failures count. A success on a half-open probe fully closes a
    tripped breaker.
    """
    global _RUNTIME_FAILURE_COUNT, _RUNTIME_FAILURE_TRIPPED, _RUNTIME_TRIPPED_AT
    if _RUNTIME_FAILURE_TRIPPED:
        log.info(
            "embedding backend recovered on half-open probe; "
            "circuit breaker closed",
        )
    elif _RUNTIME_FAILURE_COUNT:
        log.debug(
            "embedding backend recovered after %d failures; "
            "resetting circuit breaker",
            _RUNTIME_FAILURE_COUNT,
        )
    _RUNTIME_FAILURE_COUNT = 0
    _RUNTIME_FAILURE_TRIPPED = False
    _RUNTIME_TRIPPED_AT = None


def _note_backend_failure(base_url: str) -> None:
    """Increment the runtime-failure counter and trip at threshold.

    Called from the OpenAI backend's ``encode`` when both the SDK and
    httpx paths return no vectors. After
    :data:`_RUNTIME_FAILURE_THRESHOLD` consecutive failures we flip
    :data:`_MODEL_UNAVAILABLE` and log a single WARNING; every
    :func:`max_similarity` / :func:`distance_to_topic` call after
    that short-circuits via :func:`_get_model` to the "no signal"
    default without paying the network timeout. The WARNING mentions
    ``GOLDFIVE_EMBEDDING_BASE_URL`` so operators can identify the
    unreachable endpoint from logs.

    A failure while the breaker is already tripped is a failed
    half-open probe (:func:`_get_model` admitted one encode after the
    cooldown): the breaker re-opens immediately and the cooldown
    restarts.
    """
    global _RUNTIME_FAILURE_COUNT, _MODEL_UNAVAILABLE
    global _RUNTIME_FAILURE_TRIPPED, _RUNTIME_TRIPPED_AT, _MODEL
    _RUNTIME_FAILURE_COUNT += 1
    log.debug(
        "embedding backend %r: failure %d/%d",
        base_url,
        _RUNTIME_FAILURE_COUNT,
        _RUNTIME_FAILURE_THRESHOLD,
    )
    if _RUNTIME_FAILURE_TRIPPED:
        _MODEL_UNAVAILABLE = True
        _MODEL = None
        _RUNTIME_TRIPPED_AT = time.monotonic()
        log.debug(
            "embedding backend %r: half-open probe failed; breaker "
            "re-opened for %.0fs",
            base_url,
            _recovery_cooldown_s(),
        )
        return
    if _RUNTIME_FAILURE_COUNT >= _RUNTIME_FAILURE_THRESHOLD:
        _RUNTIME_FAILURE_TRIPPED = True
        _MODEL_UNAVAILABLE = True
        # Drop the cached backend too: ``_get_model`` short-circuits on
        # ``_MODEL_UNAVAILABLE`` only when ``_MODEL is None``, otherwise
        # the already-cached (dead) backend keeps getting handed back.
        _MODEL = None
        _RUNTIME_TRIPPED_AT = time.monotonic()
        log.warning(
            "embedding backend at %s has failed %d times in a row; "
            "disabling for %.0fs, then probing once "
            "(set GOLDFIVE_EMBEDDING_BASE_URL=... to redirect or unset "
            "to disable silently)",
            base_url,
            _RUNTIME_FAILURE_COUNT,
            _recovery_cooldown_s(),
        )


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
