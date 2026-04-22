"""Tests for :meth:`ADKAdapter._notify_plugins_on_cancellation` (goldfive#167).

When ``asyncio.CancelledError`` raises inside ``runner.run_async``
mid-invocation, ADK's :meth:`Runner._exec_with_plugin` does NOT fire
``after_run_callback`` (it's placed after the ``async with
Aclosing(...)`` block, not inside a ``finally``). So observability
plugins that track open spans via ``before_*_callback`` need an
alternate cleanup hook.

The adapter now iterates its caller-supplied ``plugins`` list in the
``except asyncio.CancelledError:`` branch and calls
``plugin.on_cancellation(invocation_id)`` on every plugin that defines
the method. This file pins that behaviour.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("google.adk")


def _mk(name: str) -> Any:
    from google.adk.agents.llm_agent import LlmAgent

    return LlmAgent(name=name, model="fake-model", description=name, instruction="x")


# ---------------------------------------------------------------------------
# Fakes (mirrors the shapes test_cancel_propagation.py uses)
# ---------------------------------------------------------------------------


class _FakeADKSession:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.state: dict[str, Any] = {}


class _FakeSessionService:
    def __init__(self) -> None:
        self._session = _FakeADKSession()
        self.appended: list[Any] = []

    async def create_session(self, **_kwargs: Any) -> _FakeADKSession:
        return self._session

    async def get_session(self, **_kwargs: Any) -> _FakeADKSession:
        return self._session

    async def append_event(self, *, session: Any, event: Any) -> Any:
        self.appended.append(event)
        session.events.append(event)
        return event


@dataclass
class _HangingRunner:
    """Yields one event (for invocation_id tracking), then hangs."""

    agent: Any = None
    session_service: Any = field(default_factory=_FakeSessionService)
    plugin_manager: Any = None
    plugins: list = field(default_factory=list)
    app_name: str = "fake-app"

    async def run_async(self, **kwargs: Any):  # noqa: ARG002
        from google.adk.events.event import Event
        from google.genai import types

        part = types.Part(function_call=types.FunctionCall(id="p1", name="search"))
        yield Event(
            invocation_id="inv-cancel",
            author="test_agent",
            content=types.Content(role="model", parts=[part]),
        )
        await asyncio.Event().wait()
        yield None  # pragma: no cover


# ---------------------------------------------------------------------------
# Spy plugins
# ---------------------------------------------------------------------------


class _SpyPlugin:
    """Records every ``on_cancellation`` call."""

    def __init__(self, name: str = "spy") -> None:
        self.name = name
        self.cancel_calls: list[str] = []

    def on_cancellation(self, invocation_id: str) -> None:
        self.cancel_calls.append(invocation_id)


class _NoHookPlugin:
    """Plugin without ``on_cancellation`` — must be tolerated silently."""

    def __init__(self) -> None:
        self.name = "no-hook"


class _RaisingPlugin:
    """Plugin whose ``on_cancellation`` raises — must NOT break cancel."""

    def __init__(self) -> None:
        self.name = "raiser"
        self.called = 0

    def on_cancellation(self, invocation_id: str) -> None:
        self.called += 1
        raise RuntimeError("plugin boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_adapter_invokes_on_cancellation_on_registered_plugins() -> None:
    """On ``asyncio.CancelledError``, every plugin with an
    ``on_cancellation`` method receives ``(invocation_id,)``.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    spy = _SpyPlugin()
    adapter = ADKAdapter(hanging, session_id="sess-1", plugins=[spy])

    invoke_task = asyncio.create_task(
        adapter.invoke(task=Task(id="t1", title="x"), session=Session(run_id="r1"))
    )
    await asyncio.sleep(0.01)
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    assert spy.cancel_calls == ["inv-cancel"]


async def test_adapter_tolerates_plugin_without_on_cancellation() -> None:
    """Plugins that do NOT define ``on_cancellation`` are skipped
    silently — no AttributeError, no log spam that breaks the cancel
    path.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    no_hook = _NoHookPlugin()
    spy = _SpyPlugin()
    adapter = ADKAdapter(hanging, session_id="sess-2", plugins=[no_hook, spy])

    invoke_task = asyncio.create_task(
        adapter.invoke(task=Task(id="t1", title="x"), session=Session(run_id="r1"))
    )
    await asyncio.sleep(0.01)
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    # The hook-carrying plugin still got its call; the hookless one was
    # tolerated.
    assert spy.cancel_calls == ["inv-cancel"]


async def test_plugin_exception_does_not_replace_cancel() -> None:
    """A plugin raising from ``on_cancellation`` must NOT leak an
    exception into the cancel path — the caller still sees a clean
    ``CancelledError``, not a chained ``RuntimeError``.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    raiser = _RaisingPlugin()
    spy = _SpyPlugin()
    adapter = ADKAdapter(hanging, session_id="sess-3", plugins=[raiser, spy])

    invoke_task = asyncio.create_task(
        adapter.invoke(task=Task(id="t1", title="x"), session=Session(run_id="r1"))
    )
    await asyncio.sleep(0.01)
    invoke_task.cancel()
    # Must be plain CancelledError — the RuntimeError is swallowed.
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    # Both plugins were still visited (one raiser doesn't prevent
    # the next plugin from seeing its notification).
    assert raiser.called == 1
    assert spy.cancel_calls == ["inv-cancel"]


async def test_on_cancellation_not_called_on_normal_completion() -> None:
    """Normal path: ``on_cancellation`` is NOT called on clean
    generator completion. Plugins rely on this — they use their usual
    ``after_run_callback`` on the success path and would double-close
    spans if the cancel hook also fired.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    @dataclass
    class _CleanRunner:
        agent: Any = None
        session_service: Any = field(default_factory=_FakeSessionService)
        plugin_manager: Any = None
        plugins: list = field(default_factory=list)
        app_name: str = "fake-app"

        async def run_async(self, **kwargs: Any):  # noqa: ARG002
            if False:  # pragma: no cover
                yield None
            return

    agent = _mk("worker")
    clean = _CleanRunner(agent=agent)
    spy = _SpyPlugin()
    adapter = ADKAdapter(clean, session_id="sess-4", plugins=[spy])

    await adapter.invoke(
        task=Task(id="t1", title="clean"), session=Session(run_id="r1")
    )
    assert spy.cancel_calls == []


def test_notify_plugins_empty_invocation_id_is_noop() -> None:
    """Direct unit test for the helper: an empty invocation id skips
    plugins entirely. Defensive — an empty id can't usefully be routed
    to span cleanup and would pollute plugin-side logs.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    spy = _SpyPlugin()
    adapter = ADKAdapter(hanging, session_id="sess-5", plugins=[spy])

    adapter._notify_plugins_on_cancellation("")
    assert spy.cancel_calls == []


def test_notify_plugins_empty_plugin_list_is_noop() -> None:
    """No plugins registered → no-op, no AttributeError."""
    from goldfive.adapters.adk import ADKAdapter

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    adapter = ADKAdapter(hanging, session_id="sess-6")
    # Must not raise.
    adapter._notify_plugins_on_cancellation("inv-X")
