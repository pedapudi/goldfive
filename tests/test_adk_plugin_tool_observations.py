"""Integration tests for ADK plugin tool-observation wiring.

iter-10 PR 2. Pins the contract that
:func:`goldfive.adapters._adk_plugin._GoldfiveADKPlugin.after_tool_callback`
and :meth:`_GoldfiveADKPlugin.on_tool_error_callback` write into
the ``tool_observed`` subset of ``session.recent_events`` (goldfive#239 —
the unified buffer that replaced ``recent_tool_observations``) via the
steerer's ``note_tool_observation`` method.

Three integration cases:

* Successful tool dispatch -> entry with ``is_error=False``.
* Acknowledged-failure dispatch (``result == {"error": ...}``) ->
  entry with ``is_error=True``.
* Hard tool exception (``on_tool_error_callback``) -> entry with
  ``is_error=True``.

Plus a robustness case: if ``note_tool_observation`` raises, tool
dispatch must still complete -- observability never breaks the agent.

The fixture mirrors ``tests/test_tool_loop_exemption_tightening.py`` so
the plugin entry point is exercised exactly the way the live ADK
runtime drives it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

pytest.importorskip("google.adk")

from goldfive.types import (  # noqa: E402
    RECENT_EVENT_KIND_TOOL_OBSERVED,
    DriftEvent,
    Session,
    Task,
    filter_recent_events_by_kind,
)


def _tool_obs(session: Session) -> list[dict[str, Any]]:
    """Pre-merge ``session.recent_tool_observations`` accessor.

    Goldfive#239: the dedicated buffer was merged into ``recent_events``;
    test assertions filter back to the ``tool_observed`` kind.
    """
    return filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_KIND_TOOL_OBSERVED
    )

# ---------------------------------------------------------------------------
# Stubs — minimal surface that mirrors what the live ADK runtime gives us.
# ---------------------------------------------------------------------------


class _RecordingDrift:
    """Drift sub-component capturing every observation + drift (post #410).

    Holds a real ``DriftObserver.note_tool_observation`` so the
    integration test exercises the steerer's writer end-to-end. Tests
    that want to override the writer (e.g. to assert dispatch survives
    a writer failure) can monkeypatch ``note_tool_observation`` after
    construction.
    """

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.drifts: list[DriftEvent] = []
        # Defer import until inside __init__ so the protobuf-stubs
        # gate at module import time is honoured.
        from goldfive.steerer import DefaultSteerer

        self._real_steerer = DefaultSteerer()

    async def observe(self, event: Any, session: Any) -> None:
        self.observations.append(event)

    async def handle_drift(self, drift: DriftEvent, session: Any) -> None:
        self.drifts.append(drift)

    def note_tool_observation(self, *args: Any, **kwargs: Any) -> None:
        # Delegate to the real implementation under test.
        self._real_steerer.drift.note_tool_observation(*args, **kwargs)


class _RecordingSteerer:
    """Async-capable steerer stub that captures every drift + every observation entry."""

    def __init__(self) -> None:
        self.drift = _RecordingDrift()

    @property
    def observations(self) -> list[Any]:
        return self.drift.observations

    @property
    def drifts(self) -> list[DriftEvent]:
        return self.drift.drifts


def _plugin_with_ctx(task: Task, session: Session, steerer: Any):
    """Build a goldfive ADK plugin + state-context for tool-observation tests."""
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=task,
            tool_handlers={},
            host_agent_name="test_agent",
        )
    }
    plugin.set_active_context(state[SESSION_CONTEXT_STATE_KEY])
    return plugin, state


class _ToolStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _InvCtxStub:
    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self.invocation_id = invocation_id
        self.agent = type("_A", (), {"name": agent_name})()


class _ToolCtxStub:
    """Minimal ADK tool_context shape exposing ``_invocation_context``."""

    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self._invocation_context = _InvCtxStub(invocation_id, agent_name)


# ---------------------------------------------------------------------------
# after_tool_callback — success path
# ---------------------------------------------------------------------------


async def test_after_tool_callback_records_observation() -> None:
    """A successful tool call appends an ``is_error=False`` entry."""
    steerer = _RecordingSteerer()
    session = Session(run_id="run-iter10-pr2-after-success")
    task = Task(id="t1", title="Fetch data", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    tool = _ToolStub("web_search")
    tool_ctx = _ToolCtxStub("inv-1", "worker")
    args = {"q": "raccoon facts"}
    result = {"results": ["fact a"]}

    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result=result
    )

    assert len(_tool_obs(session)) == 1
    entry = _tool_obs(session)[0]
    assert entry["tool_name"] == "web_search"
    assert entry["is_error"] is False
    assert entry["error_message"] == ""
    assert entry["task_id"] == "t1"
    # Agent name falls back to ADK-resolved name when current_agent_id
    # is empty (which it is on a bare Session for these tests).
    assert entry["agent_name"] == "worker"
    assert "raccoon" in entry["args_preview"]
    assert "fact a" in entry["result_preview"]


async def test_after_tool_callback_uses_pinned_agent_id_when_set() -> None:
    """``session.current_agent_id`` (iter-9 pin) wins over ADK-resolved name."""
    steerer = _RecordingSteerer()
    session = Session(run_id="run-iter10-pr2-pin")
    session.current_agent_id = "delegated_child"
    session.current_task_id = "t-pinned"
    task = Task(id="t1", title="Fetch", assignee_agent_id="parent")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    await plugin.after_tool_callback(
        tool=_ToolStub("get_url"),
        tool_args={"url": "https://x"},
        tool_context=_ToolCtxStub("inv-2", "parent"),
        result={"ok": True},
    )

    assert len(_tool_obs(session)) == 1
    entry = _tool_obs(session)[0]
    # Pin wins over the ADK-resolved ``parent`` name.
    assert entry["agent_name"] == "delegated_child"
    # And the task pin from iter-9 wins over ctx.task.id.
    assert entry["task_id"] == "t-pinned"


# ---------------------------------------------------------------------------
# after_tool_callback — acknowledged-failure path
# ---------------------------------------------------------------------------


async def test_after_tool_callback_records_error_result() -> None:
    """``result={"error": "..."}`` -> entry with ``is_error=True``.

    The acknowledged-failure shape from the reporting tools and most
    custom goldfive tools. The buffer must capture these so the
    three-state judge (PR 3) can recognise a provoked deviation.
    """
    steerer = _RecordingSteerer()
    session = Session(run_id="run-iter10-pr2-after-error")
    task = Task(id="t-err", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    err_result = {"acknowledged": False, "error": "missing_task_id"}

    await plugin.after_tool_callback(
        tool=_ToolStub("report_task_started"),
        tool_args={"task_id": "bogus"},
        tool_context=_ToolCtxStub("inv-err", "worker"),
        result=err_result,
    )

    assert len(_tool_obs(session)) == 1
    entry = _tool_obs(session)[0]
    assert entry["is_error"] is True
    assert entry["error_message"] == "missing_task_id"
    assert entry["tool_name"] == "report_task_started"


# ---------------------------------------------------------------------------
# on_tool_error_callback — raised-exception path
# ---------------------------------------------------------------------------


async def test_on_tool_error_callback_records_observation() -> None:
    """A raised tool exception appends an ``is_error=True`` entry."""
    steerer = _RecordingSteerer()
    session = Session(run_id="run-iter10-pr2-on-err")
    task = Task(id="t-raise", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    await plugin.on_tool_error_callback(
        tool=_ToolStub("web_search"),
        tool_args={"q": "x"},
        tool_context=_ToolCtxStub("inv-raise", "worker"),
        error=RuntimeError("upstream 500"),
    )

    assert len(_tool_obs(session)) == 1
    entry = _tool_obs(session)[0]
    assert entry["is_error"] is True
    assert "upstream 500" in entry["error_message"]
    assert entry["tool_name"] == "web_search"
    # ``on_tool_error`` doesn't get a result -- writer marks it as none.
    assert entry["result_preview"] == "(none)"
    # The existing one-shot ``Observation`` for steerer.drift.observe still
    # fires alongside the new buffer write.
    assert len(steerer.observations) == 1


async def test_on_tool_error_callback_uses_pinned_agent_id_when_set() -> None:
    """The on-error path also honours iter-9's runtime-reasoning pin."""
    steerer = _RecordingSteerer()
    session = Session(run_id="run-iter10-pr2-on-err-pin")
    session.current_agent_id = "child"
    task = Task(id="t-raise", title="Task", assignee_agent_id="parent")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    await plugin.on_tool_error_callback(
        tool=_ToolStub("net_call"),
        tool_args={"u": "https://x"},
        tool_context=_ToolCtxStub("inv", "parent"),
        error=RuntimeError("boom"),
    )

    entry = _tool_obs(session)[0]
    assert entry["agent_name"] == "child"


# ---------------------------------------------------------------------------
# Robustness — observability must never break dispatch.
# ---------------------------------------------------------------------------


async def test_observation_does_not_break_on_buffer_failure_after_tool() -> None:
    """A raising ``note_tool_observation`` must not break ``after_tool_callback``."""
    steerer = _RecordingSteerer()

    def _raise(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("buffer broken")

    steerer.drift.note_tool_observation = _raise  # type: ignore[assignment]

    session = Session(run_id="run-iter10-pr2-broken")
    task = Task(id="t1", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    # Must not raise.
    await plugin.after_tool_callback(
        tool=_ToolStub("web_search"),
        tool_args={"q": "x"},
        tool_context=_ToolCtxStub("inv", "worker"),
        result={"ok": True},
    )

    # Buffer is empty (the writer raised before appending) but no
    # exception escaped the callback.
    assert _tool_obs(session) == []


async def test_observation_does_not_break_on_buffer_failure_on_error() -> None:
    """A raising ``note_tool_observation`` must not break ``on_tool_error_callback``."""
    steerer = _RecordingSteerer()

    def _raise(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("buffer broken")

    steerer.drift.note_tool_observation = _raise  # type: ignore[assignment]

    session = Session(run_id="run-iter10-pr2-broken-on-err")
    task = Task(id="t1", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, steerer)

    # Must not raise.
    await plugin.on_tool_error_callback(
        tool=_ToolStub("web_search"),
        tool_args={"q": "x"},
        tool_context=_ToolCtxStub("inv", "worker"),
        error=RuntimeError("upstream 500"),
    )

    assert _tool_obs(session) == []
    # The original ``observe(observation, session)`` path still fires.
    assert len(steerer.observations) == 1


async def test_after_tool_callback_no_observation_when_steerer_lacks_writer() -> None:
    """A steerer stub without ``note_tool_observation`` is tolerated.

    Some legacy steerer stubs in the wild (e.g. older
    ``_RecordingToolLoopSteerer`` shapes) don't expose
    ``note_tool_observation``. The plugin must check for the method
    via ``getattr`` and skip when absent — never raise ``AttributeError``.
    """

    class _OldSteerer:
        async def observe(self, event: Any, session: Any) -> None:
            pass

        async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
            pass

    session = Session(run_id="run-iter10-pr2-old-steerer")
    task = Task(id="t1", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_ctx(task, session, _OldSteerer())

    await plugin.after_tool_callback(
        tool=_ToolStub("web_search"),
        tool_args={"q": "x"},
        tool_context=_ToolCtxStub("inv", "worker"),
        result={"ok": True},
    )
    # Buffer untouched, no exception.
    assert _tool_obs(session) == []


# ---------------------------------------------------------------------------
# Capture-everything semantics — multiple tasks land in the same buffer.
# ---------------------------------------------------------------------------


async def test_after_tool_callback_writes_across_task_boundaries() -> None:
    """Tool calls under two distinct tasks both land in the buffer.

    Locked decision (§3.1): per-task filtering is a READ-time concern.
    The writer captures everything so a deviation rooted in an earlier
    task's artefact remains visible to the judge.
    """
    steerer = _RecordingSteerer()
    session = Session(run_id="run-iter10-pr2-multi-task")
    task_a = Task(id="ta", title="A", assignee_agent_id="worker")
    plugin_a, _state_a = _plugin_with_ctx(task_a, session, steerer)

    await plugin_a.after_tool_callback(
        tool=_ToolStub("tool_a"),
        tool_args={"a": 1},
        tool_context=_ToolCtxStub("inv", "worker"),
        result={"ok": True},
    )

    # Re-use the same session under a new task ctx.
    task_b = Task(id="tb", title="B", assignee_agent_id="worker")
    plugin_b, _state_b = _plugin_with_ctx(task_b, session, steerer)
    await plugin_b.after_tool_callback(
        tool=_ToolStub("tool_b"),
        tool_args={"b": 2},
        tool_context=_ToolCtxStub("inv", "worker"),
        result={"ok": True},
    )

    task_ids = [e["task_id"] for e in _tool_obs(session)]
    tool_names = [e["tool_name"] for e in _tool_obs(session)]
    assert task_ids == ["ta", "tb"]
    assert tool_names == ["tool_a", "tool_b"]
