"""Reasoning-content fallback for non-thinking models (goldfive#263).

Background — reproduced live 2026-05-11 on Gemma-4 (session
``4a721a07``). Gemma's responses carry ``reasoning: null`` end-to-end,
so :func:`goldfive.adapters._adk_plugin._extract_reasoning` returns an
empty string and the plugin's ``after_model_callback`` never invokes
``steerer.drift.observe_reasoning``. Result: zero ``reasoning_judge``
invocations across the entire run, zero OFF_TOPIC / GOAL_DRIFT
detection from the reasoning surface, even on agent flows that would
have tripped on a thinking-capable model (Qwen3.5, Claude with
extended thinking, Gemini with thought parts).

The opt-in
:attr:`~goldfive.config.ReasoningDriftConfig.fallback_to_content_when_no_reasoning`
flag (``GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT``) tells the plugin to
synthesise a reasoning signal from the response BODY when real
reasoning extraction returns empty. The trade-off is intentionally
lossy ("what the agent decided" mixes with "what it reasoned about"),
but a lossy signal is strictly better than no signal at all on
Gemma-class deployments.

These tests pin five invariants:

1. **Flag off (default)** — empty reasoning + non-empty content does
   NOT trigger ``observe_reasoning``. Guards the default behaviour
   against an accidental always-on regression.
2. **Flag on, real reasoning present** — real chain-of-thought always
   wins; ``observe_reasoning`` gets the REASONING text, not the
   content body. The fallback is a fallback, not a replacement.
3. **Flag on, empty reasoning + content body** — the synthesised path
   fires and ``observe_reasoning`` gets the CONTENT text. The
   ``raw["reasoning_source"]`` annotation on the
   ``steerer.drift.observe(llm_response)`` payload reads
   ``"content_fallback"`` so downstream consumers can tell synthesised
   from real reasoning.
4. **Flag on, both empty** — defensive: ``observe_reasoning`` is NOT
   called when there's nothing to feed it (zero parts, no body).
5. **Env override** — ``GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT=1`` flips
   the :meth:`ReasoningDriftConfig.from_env` default to ``True``.

The integration tests reuse the
:mod:`tests.test_cooperative_cancellation` plugin / fake-context
harness so the test runs the real
``_GoldfiveADKPlugin.after_model_callback`` end-to-end without needing
a live LLM. The pure helper
:func:`goldfive.adapters._adk_plugin._choose_reasoning_text` is also
tested directly so future regressions land on the smallest possible
surface.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SessionContext,
    _choose_reasoning_text,
    make_adk_plugin,
)
from goldfive.config import ReasoningDriftConfig  # noqa: E402
from goldfive.drift import reasoning as _reasoning_module  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs — minimal LlmResponse shape covering both the real-reasoning
# path (ADK thought parts) and the content-fallback path (regular
# non-thought parts).
# ---------------------------------------------------------------------------


class _Part:
    """ADK ``content.parts[i]`` shape.

    ``thought=True`` flags a chain-of-thought part; the plugin's
    :func:`_extract_reasoning` reads only those, while
    :func:`_extract_text_parts` reads only the non-thought parts. A
    Gemma-shape response carries only non-thought parts (no reasoning
    stream); a Qwen-shape response carries both.
    """

    def __init__(self, *, text: str = "", thought: bool = False) -> None:
        self.text = text
        self.thought = thought
        self.function_call = None


class _Content:
    def __init__(self, parts: list[_Part]) -> None:
        self.parts = parts


class _LlmResponse:
    """LlmResponse-shaped stub.

    Pass ``content_text`` for a non-thought body, ``reasoning_text`` for
    a thought-flagged part, or both for the "real reasoning wins"
    invariant. ``None`` on either side maps to an empty (or absent)
    part so we can exercise the "both empty" defensive branch.
    """

    def __init__(
        self,
        *,
        content_text: str | None = None,
        reasoning_text: str | None = None,
    ) -> None:
        parts: list[_Part] = []
        if content_text is not None:
            parts.append(_Part(text=content_text, thought=False))
        if reasoning_text is not None:
            parts.append(_Part(text=reasoning_text, thought=True))
        # If both are None, materialise an empty parts list so
        # ``_extract_text_parts`` / ``_extract_reasoning`` both see a
        # well-formed but empty content block.
        self.content = _Content(parts) if parts else _Content([])
        self.finish_reason = None


class _FakeAgent:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _FakeADKSession:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.run_id = "run-test"
        self.id = "session-test"


class _FakeInvocationContext:
    def __init__(self, *, invocation_id: str, agent_name: str = "coordinator") -> None:
        self.invocation_id = invocation_id
        self.session = _FakeADKSession()
        self.agent = _FakeAgent(name=agent_name)


class _FakeCallbackContext:
    def __init__(self, *, invocation_context: _FakeInvocationContext) -> None:
        self._invocation_context = invocation_context
        self.state = invocation_context.session.state


class _DriftSpy:
    """Records every call into the drift sub-component (post #410)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.observe_calls: list[Any] = []
        self.observe_reasoning_calls: list[tuple[str, dict[str, Any]]] = []
        self.note_llm_calls = 0

    async def observe(self, observation: Any, session: Any) -> None:
        self.observe_calls.append(observation)
        await self._inner.observe(observation, session)

    async def observe_reasoning(self, text: str, **kwargs: Any) -> None:
        self.observe_reasoning_calls.append((text, dict(kwargs)))
        await self._inner.observe_reasoning(text, **kwargs)

    async def note_llm_call(self, session: Any) -> None:
        self.note_llm_calls += 1
        await self._inner.note_llm_call(session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SteererSpy:
    """Wraps a real steerer with a spying drift sub-component so a test
    can assert which observe/observe_reasoning/note_llm_call path the
    plugin took.  Component-namespaced per goldfive#410.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        # Replace the inner steerer's drift with a spying wrapper so
        # the plugin's ``steerer.drift.X`` calls land on the spy.
        self.drift = _DriftSpy(inner.drift)

    @property
    def observe_calls(self) -> list[Any]:
        return self.drift.observe_calls

    @property
    def observe_reasoning_calls(self) -> list[tuple[str, dict[str, Any]]]:
        return self.drift.observe_reasoning_calls

    @property
    def note_llm_calls(self) -> int:
        return self.drift.note_llm_calls

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _make_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="run-test",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.RUNNING),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_plugin_with_spy() -> tuple[Any, _SteererSpy]:
    plugin = make_adk_plugin(host_agent_name="coordinator")
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)
    session = Session(
        run_id="run-test",
        goals=[Goal(id="g1", summary="ship")],
        plan=_make_plan(),
        current_task_id="t1",
    )
    ctx = SessionContext(
        session=session,
        steerer=steerer,
        tools=(),
        tool_handlers={},
        host_agent_name="coordinator",
        task=session.plan.tasks[0],
    )
    plugin.set_active_context(ctx)
    spy = _SteererSpy(ctx.steerer)
    ctx.steerer = spy
    return plugin, spy


def _make_cb_context(*, invocation_id: str) -> _FakeCallbackContext:
    inv_ctx = _FakeInvocationContext(invocation_id=invocation_id)
    return _FakeCallbackContext(invocation_context=inv_ctx)


@pytest.fixture(autouse=True)
def _reset_reasoning_drift_config():
    """Snap the module-level
    :data:`goldfive.drift.reasoning._CONFIG` back to ``None`` after
    each test so the flag we install for one test doesn't leak.
    :class:`DefaultSteerer` does not install a config unless
    ``reasoning_drift_config=`` is passed, so the natural default is
    ``None`` and the plugin's accessor returns ``False`` (the
    pre-#263 behaviour).
    """
    previous = _reasoning_module._CONFIG  # noqa: SLF001 — test fixture
    yield
    _reasoning_module.configure(previous)


# ---------------------------------------------------------------------------
# 1. Pure helper: _choose_reasoning_text
# ---------------------------------------------------------------------------


def test_choose_reasoning_text_flag_off_empty_reasoning_returns_empty() -> None:
    """Default behaviour: when the flag is off, an empty reasoning
    stream returns ``("", "")`` even if a content body is present.
    Regression guard for pre-#263 callers.
    """
    response = _LlmResponse(content_text="agent's answer body", reasoning_text=None)
    text, source = _choose_reasoning_text(response, fallback_enabled=False)
    assert text == ""
    assert source == ""


def test_choose_reasoning_text_real_reasoning_wins_with_flag_on() -> None:
    """When the flag is on AND real reasoning is present, the real
    reasoning is returned (not the content body). The fallback only
    kicks in on a genuine empty.
    """
    response = _LlmResponse(
        content_text="agent's answer body",
        reasoning_text="actual chain of thought",
    )
    text, source = _choose_reasoning_text(response, fallback_enabled=True)
    assert text == "actual chain of thought"
    assert source == "reasoning"


def test_choose_reasoning_text_synthesises_from_content_when_flag_on() -> None:
    """When the flag is on AND real reasoning is empty, the helper
    returns the response body tagged as ``content_fallback``.
    """
    response = _LlmResponse(content_text="some answer text", reasoning_text=None)
    text, source = _choose_reasoning_text(response, fallback_enabled=True)
    assert text == "some answer text"
    assert source == "content_fallback"


def test_choose_reasoning_text_both_empty_returns_empty() -> None:
    """Defensive: with both reasoning and content empty, the helper
    returns ``("", "")`` regardless of the flag. Nothing to feed.
    """
    response = _LlmResponse(content_text=None, reasoning_text=None)
    text_off, source_off = _choose_reasoning_text(response, fallback_enabled=False)
    text_on, source_on = _choose_reasoning_text(response, fallback_enabled=True)
    assert text_off == "" and source_off == ""
    assert text_on == "" and source_on == ""


# ---------------------------------------------------------------------------
# 2. Integration: plugin's after_model_callback honours the flag
# ---------------------------------------------------------------------------


async def test_after_model_callback_flag_off_skips_observe_reasoning() -> None:
    """Default behaviour: empty reasoning + non-empty content does
    NOT trigger ``observe_reasoning``. This is the byte-identical
    pre-#263 path — the regression guard for operators who haven't
    opted in.
    """
    # Flag stays off (the autouse fixture restores ``None`` after each
    # test, and ``_fallback_to_content_when_no_reasoning`` returns
    # ``False`` when ``_CONFIG`` is None).
    plugin, spy = _make_plugin_with_spy()
    cb_ctx = _make_cb_context(invocation_id="inv-flag-off")
    response = _LlmResponse(content_text="Gemma's plain answer body", reasoning_text=None)

    await plugin.after_model_callback(callback_context=cb_ctx, llm_response=response)

    # Plain LLM-response observation still fires.
    assert len(spy.observe_calls) == 1
    # But reasoning observation is skipped — the default behaviour.
    assert spy.observe_reasoning_calls == []


async def test_after_model_callback_flag_on_real_reasoning_wins() -> None:
    """When the flag is on AND real reasoning is present, the plugin
    feeds the REASONING text to ``observe_reasoning`` (not the
    content body). Real reasoning always wins.
    """
    _reasoning_module.configure(
        ReasoningDriftConfig(fallback_to_content_when_no_reasoning=True)
    )
    plugin, spy = _make_plugin_with_spy()
    cb_ctx = _make_cb_context(invocation_id="inv-real-wins")
    response = _LlmResponse(
        content_text="here is my answer",
        reasoning_text="the user asked X so I will Y",
    )

    await plugin.after_model_callback(callback_context=cb_ctx, llm_response=response)

    assert len(spy.observe_reasoning_calls) == 1
    text, _kwargs = spy.observe_reasoning_calls[0]
    assert text == "the user asked X so I will Y"
    # The observation raw dict tags the source as the real reasoning
    # path, not content_fallback, so downstream consumers can tell.
    assert len(spy.observe_calls) == 1
    raw = spy.observe_calls[0]["raw"]
    assert raw["reasoning_source"] == "reasoning"


async def test_after_model_callback_flag_on_empty_reasoning_feeds_content(
    caplog: Any,
) -> None:
    """The core of goldfive#263: flag on + empty reasoning + content
    body → ``observe_reasoning`` gets the CONTENT text. The raw
    observation dict carries ``reasoning_source="content_fallback"``
    and a debug log line lands so operators can tell synthesised
    reasoning apart from real reasoning in the operational logs.
    """
    _reasoning_module.configure(
        ReasoningDriftConfig(fallback_to_content_when_no_reasoning=True)
    )
    plugin, spy = _make_plugin_with_spy()
    cb_ctx = _make_cb_context(invocation_id="inv-content-fb")
    response = _LlmResponse(
        content_text="Gemma's body text serving as fallback reasoning",
        reasoning_text=None,
    )

    import logging

    with caplog.at_level(logging.DEBUG, logger="goldfive"):
        await plugin.after_model_callback(
            callback_context=cb_ctx, llm_response=response
        )

    assert len(spy.observe_reasoning_calls) == 1
    text, _kwargs = spy.observe_reasoning_calls[0]
    assert text == "Gemma's body text serving as fallback reasoning"
    # Source annotation on the LLM-response observation.
    assert len(spy.observe_calls) == 1
    raw = spy.observe_calls[0]["raw"]
    assert raw["reasoning_source"] == "content_fallback"
    # Debug log records the fallback-firing event with the invocation
    # id so operators can audit which turns leaned on the synthesised
    # signal.
    fallback_logs = [
        r
        for r in caplog.records
        if "content_fallback" in r.getMessage()
        and "observe_reasoning" in r.getMessage()
    ]
    assert len(fallback_logs) == 1, (
        "expected exactly one content_fallback debug log; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )
    assert "inv-content-fb" in fallback_logs[0].getMessage()


async def test_after_model_callback_flag_on_both_empty_skips_observe_reasoning() -> None:
    """Defensive: flag on + reasoning empty + content empty →
    ``observe_reasoning`` is NOT called. Nothing meaningful to feed,
    and we don't want to spend a judge call on whitespace.
    """
    _reasoning_module.configure(
        ReasoningDriftConfig(fallback_to_content_when_no_reasoning=True)
    )
    plugin, spy = _make_plugin_with_spy()
    cb_ctx = _make_cb_context(invocation_id="inv-both-empty")
    response = _LlmResponse(content_text=None, reasoning_text=None)

    await plugin.after_model_callback(callback_context=cb_ctx, llm_response=response)

    assert spy.observe_reasoning_calls == []
    # Plain ``observe`` still fires (consistent with the cancel-gate
    # invariant: the regular fan-out is independent of reasoning).
    assert len(spy.observe_calls) == 1
    raw = spy.observe_calls[0]["raw"]
    # No text fed → empty source annotation.
    assert raw["reasoning_source"] == ""


# ---------------------------------------------------------------------------
# 3. Config env wiring
# ---------------------------------------------------------------------------


def test_reasoning_drift_config_default_is_false() -> None:
    """The new field defaults to ``False`` — opt-in behaviour."""
    cfg = ReasoningDriftConfig()
    assert cfg.fallback_to_content_when_no_reasoning is False


def test_reasoning_drift_config_env_flip_to_true(goldfive_reasoning_drift_env: Any) -> None:
    """``GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT=1`` flips the field to
    ``True`` via :meth:`ReasoningDriftConfig.from_env`.
    """
    goldfive_reasoning_drift_env.set(fallback_to_content="1")
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.fallback_to_content_when_no_reasoning is True


def test_reasoning_drift_config_env_other_truthy_literals(
    goldfive_reasoning_drift_env: Any,
) -> None:
    """``_read_bool_env`` accepts ``true`` / ``yes`` / ``on`` /
    ``y`` / ``t`` too; pick one to anchor the link.
    """
    goldfive_reasoning_drift_env.set(fallback_to_content="true")
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.fallback_to_content_when_no_reasoning is True


def test_reasoning_drift_config_env_unset_stays_false() -> None:
    """No env var → field stays at the dataclass default
    (``False``). Guards against an accidental flip via the env layer
    when no override is supplied.
    """
    # Ensure the env var is genuinely absent for this assertion.
    assert "GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT" not in os.environ
    cfg = ReasoningDriftConfig.from_env()
    assert cfg.fallback_to_content_when_no_reasoning is False
