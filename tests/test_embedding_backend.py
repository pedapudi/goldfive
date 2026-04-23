"""Tests for the OpenAI-compatible embedding backend in
:mod:`goldfive.drift._embed`.

Covers:

* Env-driven backend selection via ``GOLDFIVE_EMBEDDING_BASE_URL``.
* Response parsing (happy path, malformed, network error).
* LRU cache hits and eviction.
* ``set_model`` overriding env-driven config.

All tests mock out the HTTP layer -- we never hit a real endpoint.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.drift import _embed


@pytest.fixture(autouse=True)
def _reset_embed_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Ensure each test starts from a clean slate.

    ``set_model(None)`` both clears any installed encoder and resets
    ``_MODEL_UNAVAILABLE`` so the lazy-load path can run again. We
    also scrub the env vars so one test leaking a ``setenv`` can't
    influence another.
    """
    _embed.set_model(None)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_TIMEOUT_MS", raising=False)
    yield
    _embed.set_model(None)


def _canonical_response(vectors: list[list[float]]) -> dict[str, Any]:
    """Build a canonical OpenAI ``/v1/embeddings`` response envelope."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


class _FakeBackend:
    """Test double with the same .encode shape as the real backend."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._by_text = vectors_by_text
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._by_text[t] for t in texts]


# ---------------------------------------------------------------------------
# Env-driven backend selection
# ---------------------------------------------------------------------------


def test_openai_backend_env_driven(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the env var lights up the HTTP backend, and the encode
    path returns a cosine in ``[-1, 1]``."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://fake:9999")

    def fake_build(base_url: str) -> Any:
        assert base_url == "http://fake:9999"
        return _FakeBackend(
            {
                "foo": [1.0, 0.0, 0.0],
                "bar": [0.0, 1.0, 0.0],
            }
        )

    monkeypatch.setattr(_embed, "_try_load_openai_backend", fake_build)

    assert _embed.available() is True
    sim = _embed.max_similarity("foo", ["bar"])
    assert -1.0 <= sim <= 1.0
    # Orthogonal unit vectors -> cosine == 0.
    assert abs(sim) < 1e-6


def test_openai_backend_response_parsing() -> None:
    """Feed the canonical OpenAI envelope and assert vectors come out."""
    resp = _canonical_response([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    vectors = _embed._parse_openai_response(resp)
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_openai_backend_sdk_object_response_parsing() -> None:
    """Parse an object-shaped SDK response (attrs, not dict keys)."""

    class _Item:
        def __init__(self, embedding: list[float]) -> None:
            self.embedding = embedding

    class _Resp:
        def __init__(self, items: list[_Item]) -> None:
            self.data = items

    resp = _Resp([_Item([1.0, 2.0]), _Item([3.0, 4.0])])
    vectors = _embed._parse_openai_response(resp)
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


def test_openai_backend_nested_embedding_unwrapped() -> None:
    """Some llama.cpp builds wrap embeddings as ``[[...]]``; unwrap it."""
    resp = {
        "data": [
            {"embedding": [[0.5, 0.5]]},
        ]
    }
    vectors = _embed._parse_openai_response(resp)
    assert vectors == [[0.5, 0.5]]


def test_openai_backend_malformed_response() -> None:
    """Missing ``data`` / non-numeric values -> ``None`` from parser,
    ``0.0`` from ``max_similarity``."""
    # No ``data`` field (error response shape).
    assert _embed._parse_openai_response({"error": "no embeddings mode"}) is None
    # ``data`` not a list.
    assert _embed._parse_openai_response({"data": "oops"}) is None
    # Non-numeric embedding entries.
    assert (
        _embed._parse_openai_response({"data": [{"embedding": ["not", "a", "number"]}]})
        is None
    )
    # Bare garbage (not a dict, no ``.data`` attr).
    assert _embed._parse_openai_response(42) is None


def test_openai_backend_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the backend's ``encode`` blows up, ``max_similarity`` /
    ``distance_to_topic`` return ``0.0`` / ``-1.0`` rather than
    propagating the exception."""

    class _ExplodingBackend:
        def encode(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("connection refused")

    _embed.set_model(_ExplodingBackend())
    assert _embed.max_similarity("a", ["b"]) == 0.0
    assert _embed.distance_to_topic("a", "b") == -1.0


def test_cached_encode_hits_on_repeat() -> None:
    """Calling the same text twice hits the cache the second time."""
    backend = _FakeBackend({"hello": [1.0, 0.0]})
    _embed.set_model(backend)

    first = _embed._cached_encode(backend, "hello")
    second = _embed._cached_encode(backend, "hello")
    assert first == [1.0, 0.0]
    assert second == [1.0, 0.0]
    # Underlying encode was called exactly once despite two reads.
    assert len(backend.calls) == 1


def test_cached_encode_lru_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fill the cache past its cap; oldest entry drops first."""
    # Shrink the cap for the test so we don't need 513 encodes.
    monkeypatch.setattr(_embed, "_CACHE_MAX", 3)
    _embed._reset_cache()

    vectors = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "c": [1.0, 1.0],
        "d": [2.0, 0.0],
    }
    backend = _FakeBackend(vectors)
    _embed.set_model(backend)

    # Fill the cache.
    for t in ("a", "b", "c"):
        _embed._cached_encode(backend, t)
    assert len(backend.calls) == 3
    # Adding a fourth entry evicts "a" (oldest).
    _embed._cached_encode(backend, "d")
    assert len(backend.calls) == 4
    # Re-reading "a" re-encodes (cache miss).
    _embed._cached_encode(backend, "a")
    assert len(backend.calls) == 5
    # Re-reading "d" (still warm) does not.
    _embed._cached_encode(backend, "d")
    assert len(backend.calls) == 5


def test_set_model_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``set_model(fake)`` wins over ``GOLDFIVE_EMBEDDING_BASE_URL``."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://should-not-be-used")
    # If ``_get_model`` ever reaches this, it would hit the real loader
    # and raise; we never want to see it happen when set_model is in
    # effect, so the mock is punitive.
    monkeypatch.setattr(
        _embed,
        "_try_load_openai_backend",
        lambda url: (_ for _ in ()).throw(AssertionError("env path used")),
    )

    fake = _FakeBackend({"foo": [1.0, 0.0], "bar": [0.0, 1.0]})
    _embed.set_model(fake)

    sim = _embed.max_similarity("foo", ["bar"])
    assert abs(sim) < 1e-6
    # Each text encoded exactly once (once the cache warmed up).
    assert fake.calls, "fake.encode must have been called"


def test_openai_backend_env_failure_does_not_fall_back_to_st(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the user configured an HTTP endpoint but the backend cannot
    build, we prefer "no signal" over silently loading a local
    sentence-transformers model the user did not ask for."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://fake:9999")
    monkeypatch.setattr(_embed, "_try_load_openai_backend", lambda _url: None)

    assert _embed.available() is False


def test_openai_backend_builder_honours_model_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_try_load_openai_backend`` reads every optional env var."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_MODEL", "qwen3-embed")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_TIMEOUT_MS", "2500")

    captured: dict[str, Any] = {}

    class _Spy:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(_embed, "_OpenAIEmbeddingBackend", _Spy)

    backend = _embed._try_load_openai_backend("http://host:8081/")
    assert backend is not None
    assert captured["base_url"] == "http://host:8081/"
    assert captured["model"] == "qwen3-embed"
    assert captured["api_key"] == "secret"
    assert captured["timeout_ms"] == 2500


def test_openai_backend_httpx_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the ``openai`` SDK is not available, the backend falls
    through to ``httpx`` and parses the same response shape."""
    # Force the SDK path off.
    monkeypatch.setattr(
        _embed._OpenAIEmbeddingBackend,
        "_try_build_openai_client",
        lambda self: False,
    )

    captured_request: dict[str, Any] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return _canonical_response([[1.0, 0.0], [0.0, 1.0]])

    class _FakeHttpxClient:
        def post(
            self, url: str, *, json: dict[str, Any], headers: dict[str, str]
        ) -> _FakeResponse:
            captured_request["url"] = url
            captured_request["json"] = json
            captured_request["headers"] = headers
            return _FakeResponse()

    backend = _embed._OpenAIEmbeddingBackend(
        base_url="http://srv:8081",
        model="qwen",
        api_key="k",
        timeout_ms=1000,
    )
    # Short-circuit lazy httpx client construction.
    backend._httpx_client = _FakeHttpxClient()

    out = backend.encode(["hello", "world"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]
    assert captured_request["url"] == "http://srv:8081/v1/embeddings"
    assert captured_request["json"]["model"] == "qwen"
    assert captured_request["json"]["input"] == ["hello", "world"]
    assert captured_request["headers"]["Authorization"] == "Bearer k"


def test_openai_backend_httpx_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An HTTP error on the httpx path returns ``None`` (no signal)
    instead of propagating."""

    class _Raising:
        def post(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    backend = _embed._OpenAIEmbeddingBackend(
        base_url="http://srv:8081",
        model="",
        api_key=None,
        timeout_ms=1000,
    )
    # Force SDK path off and inject the raising httpx client.
    backend._prefer_sdk = False
    backend._openai_client = None
    backend._httpx_client = _Raising()

    assert backend.encode(["foo"]) == []
