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
    influence another. Under goldfive#225 ``configure(None)`` also
    clears any :class:`~goldfive.config.EmbeddingConfig` a prior
    ``goldfive.wrap()`` call may have installed; without this, the
    env-var behaviour these tests exercise would be short-circuited
    by a leaked config from another test module.
    """
    _embed.set_model(None)
    _embed.configure(None)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("GOLDFIVE_EMBEDDING_TIMEOUT_MS", raising=False)
    yield
    _embed.set_model(None)
    _embed.configure(None)


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

    _embed.set_backend_loader(fake_build)
    try:
        assert _embed.available() is True
        sim = _embed.max_similarity("foo", ["bar"])
        assert -1.0 <= sim <= 1.0
        # Orthogonal unit vectors -> cosine == 0.
        assert abs(sim) < 1e-6
    finally:
        _embed.set_backend_loader(None)


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


def test_cached_encode_lru_eviction(request: pytest.FixtureRequest) -> None:
    """Fill the cache past its cap; oldest entry drops first."""
    # Shrink the cap for the test so we don't need 513 encodes. Restore
    # the production cap on teardown so other tests in this session
    # see the default LRU size.
    original_cap = _embed._CACHE_MAX
    _embed.set_cache_max(3)
    request.addfinalizer(lambda: _embed.set_cache_max(original_cap))
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


def test_set_model_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """``set_model(fake)`` wins over ``GOLDFIVE_EMBEDDING_BASE_URL``."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://should-not-be-used")
    # If ``_get_model`` ever reaches this, it would hit the real loader
    # and raise; we never want to see it happen when set_model is in
    # effect, so the loader is punitive.
    _embed.set_backend_loader(
        lambda url: (_ for _ in ()).throw(AssertionError("env path used"))
    )
    request.addfinalizer(lambda: _embed.set_backend_loader(None))

    fake = _FakeBackend({"foo": [1.0, 0.0], "bar": [0.0, 1.0]})
    _embed.set_model(fake)

    sim = _embed.max_similarity("foo", ["bar"])
    assert abs(sim) < 1e-6
    # Each text encoded exactly once (once the cache warmed up).
    assert fake.calls, "fake.encode must have been called"


def test_openai_backend_env_failure_does_not_fall_back_to_st(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """If the user configured an HTTP endpoint but the backend cannot
    build, we prefer "no signal" over silently loading a local
    sentence-transformers model the user did not ask for."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_BASE_URL", "http://fake:9999")
    _embed.set_backend_loader(lambda _url: None)
    request.addfinalizer(lambda: _embed.set_backend_loader(None))

    assert _embed.available() is False


def test_openai_backend_builder_honours_model_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """``_try_load_openai_backend`` reads every optional env var."""
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_MODEL", "qwen3-embed")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("GOLDFIVE_EMBEDDING_TIMEOUT_MS", "2500")

    captured: dict[str, Any] = {}

    class _Spy:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    _embed.set_backend_class(_Spy)
    request.addfinalizer(lambda: _embed.set_backend_class(None))

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


# ---------------------------------------------------------------------------
# Runtime circuit breaker (goldfive#225 follow-up)
# ---------------------------------------------------------------------------


def _make_always_failing_backend() -> Any:
    """Build a backend whose HTTP paths both fail, so ``encode`` returns ``[]``."""

    class _Raising:
        def post(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("connection refused")

    backend = _embed._OpenAIEmbeddingBackend(
        base_url="http://dead:9999",
        model="",
        api_key=None,
        timeout_ms=1000,
    )
    backend._prefer_sdk = False
    backend._openai_client = None
    backend._httpx_client = _Raising()
    return backend


def test_circuit_breaker_trips_after_three_http_5xx() -> None:
    """Three consecutive failed encodes flip ``_MODEL_UNAVAILABLE``."""
    _embed.reset_circuit_breaker()
    backend = _make_always_failing_backend()

    assert _embed._MODEL_UNAVAILABLE is False
    # First two failures must not trip.
    assert backend.encode(["a"]) == []
    assert _embed._MODEL_UNAVAILABLE is False
    assert backend.encode(["b"]) == []
    assert _embed._MODEL_UNAVAILABLE is False
    # Third failure trips the breaker.
    assert backend.encode(["c"]) == []
    assert _embed._MODEL_UNAVAILABLE is True
    assert _embed._RUNTIME_FAILURE_TRIPPED is True


def test_circuit_breaker_resets_on_success() -> None:
    """A successful encode clears the failure counter."""
    _embed.reset_circuit_breaker()

    class _Flaky:
        """Fails twice, then succeeds."""

        def __init__(self) -> None:
            self._calls = 0

        def post(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            raise RuntimeError("transient")

    backend = _embed._OpenAIEmbeddingBackend(
        base_url="http://flaky:9999",
        model="",
        api_key=None,
        timeout_ms=1000,
    )
    backend._prefer_sdk = False
    backend._openai_client = None
    backend._httpx_client = _Flaky()

    # Two failures, counter at 2.
    assert backend.encode(["a"]) == []
    assert backend.encode(["b"]) == []
    assert _embed._RUNTIME_FAILURE_COUNT == 2

    # Simulate a recovery by swapping in a working client.
    class _OK:
        def post(self, *args: Any, **kwargs: Any) -> Any:
            class _Resp:
                def raise_for_status(self) -> None:
                    pass

                def json(self) -> dict[str, Any]:
                    return _canonical_response([[1.0, 0.0]])

            return _Resp()

    backend._httpx_client = _OK()
    out = backend.encode(["c"])
    assert out == [[1.0, 0.0]]
    # Counter reset after the successful call.
    assert _embed._RUNTIME_FAILURE_COUNT == 0

    # Subsequent failures start counting from zero again -- would need
    # another THREE to trip.
    class _FailAgain:
        def post(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    backend._httpx_client = _FailAgain()
    assert backend.encode(["d"]) == []
    assert _embed._RUNTIME_FAILURE_COUNT == 1
    assert _embed._MODEL_UNAVAILABLE is False


def test_circuit_breaker_warns_once_on_trip(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only one WARNING is emitted, even if more failures follow."""
    import logging

    _embed.reset_circuit_breaker()
    backend = _make_always_failing_backend()

    with caplog.at_level(
        logging.WARNING, logger="goldfive.drift.reasoning.embed"
    ):
        for _ in range(5):
            backend.encode(["x"])

    warnings = [
        r
        for r in caplog.records
        if r.name == "goldfive.drift.reasoning.embed"
        and "has failed" in r.getMessage()
        and "in a row" in r.getMessage()
    ]
    assert len(warnings) == 1, (
        f"expected exactly one trip WARNING, got {len(warnings)}: "
        f"{[r.getMessage() for r in warnings]}"
    )
    assert "http://dead:9999" in warnings[0].getMessage()


def test_circuit_breaker_short_circuits_get_model() -> None:
    """Once tripped, :func:`_get_model` returns ``None`` -- no more HTTP."""
    _embed.reset_circuit_breaker()
    backend = _make_always_failing_backend()
    _embed.set_model(backend)

    # Trip the breaker.
    for _ in range(3):
        backend.encode(["x"])
    assert _embed._MODEL_UNAVAILABLE is True

    # ``_get_model`` now returns ``None`` -- the upstream helpers
    # (``max_similarity`` / ``distance_to_topic``) see "no signal".
    assert _embed._get_model() is None
    assert _embed.max_similarity("a", ["b"]) == 0.0
    assert _embed.distance_to_topic("a", "b") == -1.0


def test_reset_circuit_breaker_clears_state() -> None:
    """The test-only helper restores a pristine module state."""
    _embed.reset_circuit_breaker()
    backend = _make_always_failing_backend()
    for _ in range(3):
        backend.encode(["x"])
    assert _embed._MODEL_UNAVAILABLE is True
    assert _embed._RUNTIME_FAILURE_TRIPPED is True

    _embed.reset_circuit_breaker()
    assert _embed._RUNTIME_FAILURE_COUNT == 0
    assert _embed._RUNTIME_FAILURE_TRIPPED is False
