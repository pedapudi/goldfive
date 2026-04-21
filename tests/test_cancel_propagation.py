"""Cancel-propagation tests for the registry-dispatch model.

When an ``adapter.invoke(task, session)`` task is cancelled
mid-invocation, the cancel must propagate naturally through
``runner.run_async()`` and through any nested AgentTool sub-Runner
awaits. After cancel:

* ``_heal_pending_tool_calls`` must fire (the synthetic
  ``function_response`` events are appended so the next turn sees a
  well-formed history).
* Adapter bookkeeping (``_pending_tool_call_ids``,
  ``_pending_tool_call_names``) returns to empty.
* A subsequent ``invoke`` must succeed without inheriting stale state.

Skipped entirely when ``google.adk`` is not installed.
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
# Fakes that let us drive the invoke loop manually
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
    """Runner that emits a function_call then hangs — exactly the
    shape the heal path must deal with on mid-invocation cancel."""

    agent: Any = None
    session_service: Any = field(default_factory=_FakeSessionService)
    plugin_manager: Any = None
    plugins: list = field(default_factory=list)
    app_name: str = "fake-app"

    async def run_async(self, **kwargs: Any):  # noqa: ARG002
        from google.adk.events.event import Event
        from google.genai import types

        # Emit one function_call for a tool that never returns.
        part = types.Part(function_call=types.FunctionCall(id="pending-1", name="search"))
        yield Event(
            invocation_id="inv-1",
            author="test_agent",
            content=types.Content(role="model", parts=[part]),
        )
        # Hang forever — the outer task.cancel() injects CancelledError here.
        await asyncio.Event().wait()
        yield None  # pragma: no cover


# ---------------------------------------------------------------------------
# Cancel-propagation tests
# ---------------------------------------------------------------------------


async def test_cancel_invoke_raises_cancelled_and_heals_orphan_tool_calls() -> None:
    """invoke() raises CancelledError on task.cancel(); session history
    is healed with a synthetic function_response for the orphan call.

    This is the registry-model analogue of the existing heal test in
    test_adk_adapter.py — covers the cancel path on a per-agent runner
    (not just the legacy single-runner path).
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    # Use the degraded-prebuilt path so the adapter takes the hanging
    # runner verbatim — no per-agent expansion or plugin integrity check
    # gets in our way.
    adapter = ADKAdapter(hanging, session_id="sess-1")

    invoke_task = asyncio.create_task(
        adapter.invoke(task=Task(id="t1", title="x"), session=Session(run_id="r1"))
    )
    # Let the runner yield the function_call + start awaiting.
    await asyncio.sleep(0.01)
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    # Heal path fired: one synthetic response for "pending-1".
    appended = hanging.session_service.appended
    assert len(appended) == 1
    responses = appended[0].get_function_responses()
    assert len(responses) == 1
    assert responses[0].id == "pending-1"
    assert responses[0].response.get("goldfive_cancelled") is True


async def test_after_cancel_pending_tool_call_bookkeeping_is_empty() -> None:
    """The adapter's internal pending-call state returns to empty after
    cancel. Leaked ids would cause the NEXT invoke to synthesize stale
    responses from this invoke's cancelled turn.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    adapter = ADKAdapter(hanging, session_id="sess-1")

    invoke_task = asyncio.create_task(
        adapter.invoke(task=Task(id="t1", title="x"), session=Session(run_id="r1"))
    )
    await asyncio.sleep(0.01)
    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    assert adapter._pending_tool_call_ids == set()
    assert adapter._pending_tool_call_names == {}


async def test_next_invoke_after_cancel_runs_cleanly() -> None:
    """A fresh invoke after a cancelled one must not inherit stale state.

    Regression: if ``_pending_tool_call_ids`` were leaked, or if the
    plugin's ``_active_ctx`` weren't cleared on the cancel path, the
    next invoke would either re-heal phantom calls or bind to the
    stale ctx. We swap the runner out and verify a clean second run.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    agent = _mk("worker")
    hanging = _HangingRunner(agent=agent)
    adapter = ADKAdapter(hanging, session_id="sess-1")

    # First invocation: cancelled.
    inv1 = asyncio.create_task(
        adapter.invoke(task=Task(id="t1", title="x"), session=Session(run_id="r1"))
    )
    await asyncio.sleep(0.01)
    inv1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inv1

    # Plugin's active ctx was released on the cancel path.
    assert adapter._plugin._active_ctx is None

    # Second invocation with a runner that runs to clean completion.
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

    clean = _CleanRunner(agent=agent)
    # Single-Runner model: swap the one runner.
    adapter._runner = clean
    # Force a fresh session lookup.
    adapter._session_id = None

    result = await adapter.invoke(task=Task(id="t2", title="clean"), session=Session(run_id="r2"))
    # Result carries the new task id; no cancel / no orphan healing.
    assert result.task_id == "t2"
    assert result.stop_reason != "cancelled"
    assert clean.session_service.appended == [], (
        "no synthetic healing events should have been appended on a "
        "clean second invocation — a non-empty list means stale "
        "pending-ids leaked from the cancelled first invoke"
    )


async def test_cancel_with_multiple_pending_calls_heals_every_id() -> None:
    """Two parallel function_calls outstanding at cancel time → two heals."""
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Session, Task

    @dataclass
    class _MultiHangingRunner:
        agent: Any = None
        session_service: Any = field(default_factory=_FakeSessionService)
        plugin_manager: Any = None
        plugins: list = field(default_factory=list)
        app_name: str = "fake-app"

        async def run_async(self, **kwargs: Any):  # noqa: ARG002
            from google.adk.events.event import Event
            from google.genai import types

            parts = [
                types.Part(function_call=types.FunctionCall(id=f"p-{i}", name=f"t{i}"))
                for i in (1, 2)
            ]
            yield Event(
                invocation_id="inv-m",
                author="test_agent",
                content=types.Content(role="model", parts=parts),
            )
            await asyncio.Event().wait()
            yield None  # pragma: no cover

    agent = _mk("worker")
    multi = _MultiHangingRunner(agent=agent)
    adapter = ADKAdapter(multi, session_id="sess-m")

    task = asyncio.create_task(
        adapter.invoke(task=Task(id="t1", title="x"), session=Session(run_id="r1"))
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    healed_ids = {
        fr.id for ev in multi.session_service.appended for fr in ev.get_function_responses()
    }
    assert healed_ids == {"p-1", "p-2"}
    # Bookkeeping state is clean.
    assert adapter._pending_tool_call_ids == set()
