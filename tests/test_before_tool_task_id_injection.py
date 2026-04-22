"""Tests for goldfive#191 Layer 3 — ``before_tool_callback`` task_id injection.

The plugin's reporting-tool dispatch path (``_GoldfiveADKPlugin.
before_tool_callback``) is the outermost safety net that fills in a
missing / placeholder ``task_id`` from the pinned
``goldfive.current_task_id`` key on session.state. Layer 1
(``before_agent_callback``) is responsible for pinning that key at the
start of every agent turn; Layer 2 (the reporting-tool handler itself)
also defaults from state. This file pins the Layer-3 contract:

  * empty / placeholder ``task_id`` args are rewritten from state,
  * real-looking ``task_id`` args are left alone even if state disagrees,
  * non-reporting tools are never touched,
  * missing state leaves args unchanged (Layer 2 will handle the error),
  * a malformed callback_context never raises out of the plugin.

Skipped entirely when ``google.adk`` is not installed (optional dep).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.reporting import ReportingToolSpec
from goldfive.types import Session, Task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Ctx:
    """Minimal ADK-callback-context stub with a mutable state dict.

    Mirrors the ``state_ctx_cls`` fixture from ``test_adk_adapter.py`` but
    inlined here to avoid cross-file fixture coupling.
    """

    class _Session:
        def __init__(self, state: dict) -> None:
            self.state = state

    def __init__(self, state: dict) -> None:
        self._state = state

    @property
    def session(self) -> Any:
        return _Ctx._Session(self._state)


class _Tool:
    """Bare tool stub — the plugin only reads ``.name``."""

    def __init__(self, name: str) -> None:
        self.name = name


def _make_plugin_with_reporting_spec(tool_name: str):
    """Return (plugin, state_dict) wired for a single reporting tool.

    The plugin is built via :func:`make_adk_plugin` (the public factory)
    and wired to a :class:`SessionContext` so the reporting-tool match in
    ``before_tool_callback`` fires. The handler is a no-op that echoes
    the args the plugin dispatched with — callers inspect that echo to
    verify injection.
    """
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    captured: list[dict[str, Any]] = []

    async def handler(args: Any, session: Any, steerer: Any) -> dict[str, Any]:
        captured.append(dict(args))
        return {"acknowledged": True, "echo": dict(args)}

    spec = ReportingToolSpec(
        name=tool_name,
        description="",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    session_obj = Session(run_id="run-1")
    task = Task(id="t-ignored", title="x")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session_obj,
            steerer=None,
            task=task,
            tool_handlers={tool_name: handler},
            tools=[spec],
            host_agent_name="test_agent",
        )
    }
    return plugin, state, captured


# ---------------------------------------------------------------------------
# Injection tests — reporting tools
# ---------------------------------------------------------------------------


async def test_empty_arg_replaced_from_state() -> None:
    """An empty task_id on a report_task_* call is filled from pinned state."""
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID

    plugin, state, captured = _make_plugin_with_reporting_spec("report_task_completed")
    state[KEY_CURRENT_TASK_ID] = "t-1"

    args: dict[str, Any] = {"task_id": "", "summary": "done"}
    await plugin.before_tool_callback(
        tool=_Tool("report_task_completed"),
        tool_args=args,
        tool_context=_Ctx(state),
    )

    assert args["task_id"] == "t-1", (
        "Layer 3 must overwrite an empty task_id with the pinned "
        "goldfive.current_task_id from state."
    )
    # And the handler must have seen the corrected arg.
    assert captured == [{"task_id": "t-1", "summary": "done"}]


async def test_placeholder_arg_replaced() -> None:
    """Known placeholder strings ('placeholder', 'unknown', 'TODO') are replaced."""
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID

    for placeholder in ("placeholder", "unknown", "TODO", "  ", "none", "N/A"):
        plugin, state, captured = _make_plugin_with_reporting_spec(
            "report_task_progress"
        )
        state[KEY_CURRENT_TASK_ID] = "t-1"

        args: dict[str, Any] = {"task_id": placeholder, "detail": "x"}
        await plugin.before_tool_callback(
            tool=_Tool("report_task_progress"),
            tool_args=args,
            tool_context=_Ctx(state),
        )

        assert args["task_id"] == "t-1", (
            f"placeholder {placeholder!r} must be replaced with the pinned "
            "task_id from state."
        )


async def test_real_looking_arg_preserved() -> None:
    """A real-looking task_id is NOT overwritten, even if state disagrees.

    Wrong task_ids must surface as handler errors (terminal-task rejection
    / not-found) rather than being silently re-targeted. See goldfive#191
    Layer 3 design note: "Wrong task_ids are better surfaced as failures
    than masked."
    """
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID

    plugin, state, _ = _make_plugin_with_reporting_spec("report_task_failed")
    state[KEY_CURRENT_TASK_ID] = "t-1"

    args: dict[str, Any] = {"task_id": "t-42", "error": "boom"}
    await plugin.before_tool_callback(
        tool=_Tool("report_task_failed"),
        tool_args=args,
        tool_context=_Ctx(state),
    )

    assert args["task_id"] == "t-42", (
        "Real-looking task_ids must be preserved verbatim — Layer 3 is a "
        "safety net for missing/placeholder args, not a rewriter."
    )


async def test_awaiting_approval_also_injected() -> None:
    """report_awaiting_approval is part of the reporting-tool family."""
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID

    plugin, state, _ = _make_plugin_with_reporting_spec("report_awaiting_approval")
    state[KEY_CURRENT_TASK_ID] = "t-1"

    args: dict[str, Any] = {"task_id": ""}
    await plugin.before_tool_callback(
        tool=_Tool("report_awaiting_approval"),
        tool_args=args,
        tool_context=_Ctx(state),
    )

    assert args["task_id"] == "t-1"


# ---------------------------------------------------------------------------
# Non-injection paths
# ---------------------------------------------------------------------------


async def test_non_report_tool_untouched() -> None:
    """Non-reporting tools must never have their task_id rewritten."""
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID

    plugin = make_adk_plugin(host_agent_name="test_agent")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=Session(run_id="r"),
            steerer=None,
            task=Task(id="t", title="x"),
            tool_handlers={},
            host_agent_name="test_agent",
        ),
        KEY_CURRENT_TASK_ID: "t-1",
    }

    args: dict[str, Any] = {"task_id": ""}
    result = await plugin.before_tool_callback(
        tool=_Tool("write_file"),
        tool_args=args,
        tool_context=_Ctx(state),
    )

    # write_file is not a reporting tool -> plugin passes through with
    # no rewrite.
    assert result is None
    assert args["task_id"] == "", (
        "Non-reporting tools must be left alone — the injection is scoped "
        "to report_task_* / report_awaiting_approval only."
    )


async def test_missing_state_leaves_args_unchanged() -> None:
    """No pinned task_id in state → args untouched; Layer 2 will handle it."""
    plugin, state, _ = _make_plugin_with_reporting_spec("report_task_started")
    # Note: NOT setting KEY_CURRENT_TASK_ID.

    args: dict[str, Any] = {"task_id": ""}
    await plugin.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args=args,
        tool_context=_Ctx(state),
    )

    assert args["task_id"] == "", (
        "When state has no pinned task_id, Layer 3 must not guess — the "
        "handler (Layer 2) default-from-state or the eventual error is "
        "the correct outcome."
    )


async def test_injection_is_best_effort() -> None:
    """A malformed callback_context (state=None / no .session) must not raise.

    This pins the ``try/except`` in ``_inject_task_id_from_state`` — any
    exception from state inspection is swallowed at DEBUG level so the
    plugin's tool-dispatch path cannot be broken by a missing attribute
    on an odd ADK-context shape.
    """
    from goldfive.adapters._adk_plugin import _inject_task_id_from_state

    # Context with no ``.session`` and no ``.state``. The helper must
    # silently return without touching args.
    class _BrokenCtx:
        pass

    args: dict[str, Any] = {"task_id": ""}
    _inject_task_id_from_state(
        tool_name="report_task_completed",
        tool_args=args,
        tool_context=_BrokenCtx(),
    )
    assert args["task_id"] == ""  # unchanged — no state to read from.

    # A context whose ``.state`` attribute raises on access. The helper's
    # try/except must swallow this.
    class _RaisingCtx:
        @property
        def state(self) -> Any:
            raise RuntimeError("state access blew up")

        @property
        def session(self) -> Any:
            raise RuntimeError("session access blew up")

    args2: dict[str, Any] = {"task_id": ""}
    _inject_task_id_from_state(
        tool_name="report_task_completed",
        tool_args=args2,
        tool_context=_RaisingCtx(),
    )
    assert args2["task_id"] == ""  # unchanged.


# ---------------------------------------------------------------------------
# Ordering: Layer 1 (state stamp) → Layer 3 (injection)
# ---------------------------------------------------------------------------


async def test_injection_sees_layer1_state_stamp() -> None:
    """Sanity: a state write done BEFORE the tool callback is visible to it.

    This is the ordering contract Layer 3 depends on — Layer 1's
    ``before_agent_callback`` writes to session.state at agent-turn start,
    and every ``before_tool_callback`` within that turn sees the write.
    We simulate that by writing the state key, then dispatching, then
    asserting the injection happened.
    """
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID

    plugin, state, _ = _make_plugin_with_reporting_spec("report_task_blocked")

    # Layer-1-style write (simulating what before_agent_callback does).
    state[KEY_CURRENT_TASK_ID] = "t-layer1"

    # Layer-3 tool-call entry.
    args: dict[str, Any] = {"task_id": "", "blocked_on": "dep-x"}
    await plugin.before_tool_callback(
        tool=_Tool("report_task_blocked"),
        tool_args=args,
        tool_context=_Ctx(state),
    )

    assert args["task_id"] == "t-layer1"
