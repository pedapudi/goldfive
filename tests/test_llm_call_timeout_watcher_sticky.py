"""Sticky-cancel gate for the LLM-call timeout watcher (goldfive#271
follow-up; demo-v12.log regression).

Pre-fix: after the watcher fired CRITICAL on call N of an invocation,
the cancel marker was POPPED on the next ``before_model_callback``
(consume-once) and ``before_model_callback`` returned ``None`` —
which per ADK's contract lets the LLM request proceed. So call N+1
ran for another full budget, the watcher fired again, and the cycle
repeated. Demo-v12 showed 4 firings on a single invocation
``e-1e9e1f05`` and 5 on ``e-f342d38c``.

Post-fix: the plugin tracks a sticky ``_cancelled_invocations`` set;
every callback consults :meth:`is_invocation_cancelled` which is
True both when a fresh ``CancellationRequest`` is pending AND when
an earlier callback already consumed one. ``before_model_callback``
returns a synthetic ``LlmResponse`` (NOT ``None``) so ADK actually
short-circuits.

These tests pin the new contract:

* watcher fires once and flags the invocation;
* the next ``before_model_callback`` sees the sticky flag, returns a
  non-None response (LLM call skipped), and does NOT schedule a
  fresh watcher;
* the next ``before_tool_callback`` returns ``{"status": "cancelled"}``
  with no second ``InvocationCancelled`` sink emit.
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

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SessionContext,
    make_adk_plugin,
)
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    CancellationRequest,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs (mirrored from test_cooperative_cancellation.py to keep the
# file self-contained; the plugin only inspects the duck-typed shape)
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


def _cancelled_event_count(sink: _ListSink) -> int:
    out = 0
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") == "invocation_cancelled":
                out += 1
        except Exception:
            continue
    return out


class _FakeAgent:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _FakeADKSession:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.run_id = "run-test"
        self.id = "session-test"


class _FakeInvocationContext:
    def __init__(self, *, invocation_id: str) -> None:
        self.invocation_id = invocation_id
        self.session = _FakeADKSession()
        self.agent = _FakeAgent(name="coordinator")


class _FakeCallbackContext:
    def __init__(self, *, invocation_context: _FakeInvocationContext) -> None:
        self._invocation_context = invocation_context
        self.state = invocation_context.session.state


class _FakeToolContext(_FakeCallbackContext):
    def __init__(self, *, invocation_context: _FakeInvocationContext) -> None:
        super().__init__(invocation_context=invocation_context)
        self.function_call_id = "fc-1"


class _FakeTool:
    """FunctionTool-shaped stub (``.name`` + ``.func``)."""

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.func = lambda **_kw: {"ok": True}


class _FakeAgentTool:
    """AgentTool-shaped stub (``.name`` + ``.agent``).

    Trips :func:`goldfive.adapters._adk_plugin._is_agent_tool_dispatch`
    via the ``.agent is not None`` duck-typed branch so the
    cooperative-cancel short-circuit fires on this stub even when the
    optional ``adk`` extra is absent. FunctionTool dispatches must NOT
    short-circuit (Bug C / goldfive#211610) — those tests use
    :class:`_FakeTool` instead.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.agent = type("_FakeAgent", (), {"name": name})()


class _FakeLlmRequest:
    """Minimal ``llm_request`` stub.

    ``before_model_callback`` reaches into ``contents`` and ``config``
    via ``_measure_request_chars``; both can be empty / missing without
    raising.
    """

    def __init__(self) -> None:
        self.contents: list[Any] = []
        self.config = None


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


def _make_plugin_with_ctx() -> tuple[Any, _ListSink]:
    plugin = make_adk_plugin(host_agent_name="coordinator")
    sink = _ListSink()
    # Explicit active mode: the watcher's cancel-flag write under test
    # is suppressed under the shipped observation-only default.
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=False))
    steerer.bind(sinks=[sink], planner=None)
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
    return plugin, sink


def _make_request(invocation_id: str) -> CancellationRequest:
    return CancellationRequest(
        invocation_id=invocation_id,
        reason="llm_call_timeout",
        severity=DriftSeverity.CRITICAL,
        drift_kind="llm_call_timeout",
        detail="LLM call exceeded wall-clock budget (120.0s)",
    )


# ---------------------------------------------------------------------------
# is_invocation_cancelled — the sticky-aware predicate
# ---------------------------------------------------------------------------


def test_is_invocation_cancelled_false_when_unflagged() -> None:
    plugin, _ = _make_plugin_with_ctx()
    assert plugin.is_invocation_cancelled("inv-x") is False


def test_is_invocation_cancelled_true_with_pending_request() -> None:
    plugin, _ = _make_plugin_with_ctx()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request("inv-1"))
    assert plugin.is_invocation_cancelled("inv-1") is True


def test_is_invocation_cancelled_true_after_consume_via_sticky_set() -> None:
    """Even after :meth:`consume_cancel_for_invocation` pops the
    pending request, ``is_invocation_cancelled`` returns True as long
    as the invocation_id is in the sticky set. This is the bit that
    fixes /tmp/demo-v12.log."""
    plugin, _ = _make_plugin_with_ctx()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request("inv-1"))
    plugin.consume_cancel_for_invocation("inv-1")
    plugin._cancelled_invocations.add("inv-1")
    assert plugin.peek_cancel_for_invocation("inv-1") is None
    assert plugin.is_invocation_cancelled("inv-1") is True


def test_is_invocation_cancelled_empty_id_is_false() -> None:
    plugin, _ = _make_plugin_with_ctx()
    assert plugin.is_invocation_cancelled("") is False


# ---------------------------------------------------------------------------
# before_model_callback — short-circuit + sticky bit
# ---------------------------------------------------------------------------


async def test_before_model_short_circuits_when_cancel_pending() -> None:
    """A fresh cancel marker triggers the InvocationCancelled emit and
    a non-None LlmResponse return — ADK respects this as a
    short-circuit per the BasePlugin contract."""
    plugin, sink = _make_plugin_with_ctx()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request("inv-1"))
    inv_ctx = _FakeInvocationContext(invocation_id="inv-1")
    cb_ctx = _FakeCallbackContext(invocation_context=inv_ctx)

    response = await plugin.before_model_callback(
        callback_context=cb_ctx, llm_request=_FakeLlmRequest()
    )

    # Non-None: ADK short-circuits the LLM dispatch.
    assert response is not None
    # Sticky bit set so subsequent callbacks also short-circuit.
    assert plugin.is_invocation_cancelled("inv-1") is True
    # InvocationCancelled emitted exactly once.
    assert _cancelled_event_count(sink) == 1
    # No watcher scheduled — the pending dict has no entry, or the
    # entry has no live watcher task.
    pending = plugin._invocation_llm_pending.get("inv-1") or {}
    watcher = pending.get("watcher") if isinstance(pending, dict) else None
    assert watcher is None or watcher.done()


async def test_before_model_sticky_short_circuit_subsequent_call() -> None:
    """The fix for /tmp/demo-v12.log: after the watcher fired and the
    first callback consumed the cancel, the NEXT
    ``before_model_callback`` on the same invocation must still
    short-circuit (return non-None) and must NOT schedule another
    watcher. Pre-fix it returned None and ADK kept dispatching."""
    plugin, sink = _make_plugin_with_ctx()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request("inv-1"))
    inv_ctx = _FakeInvocationContext(invocation_id="inv-1")

    # Call 1 — consumes the marker, emits InvocationCancelled.
    first = await plugin.before_model_callback(
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
        llm_request=_FakeLlmRequest(),
    )
    assert first is not None
    assert _cancelled_event_count(sink) == 1
    assert plugin.peek_cancel_for_invocation("inv-1") is None

    # Call 2 — same invocation_id. Pre-fix: returned None, scheduled
    # a fresh watcher, the LLM call would proceed and the watcher
    # would eventually fire AGAIN (the demo-v12 regression).
    second = await plugin.before_model_callback(
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
        llm_request=_FakeLlmRequest(),
    )
    assert second is not None  # short-circuited again
    # No new InvocationCancelled emit — only the original.
    assert _cancelled_event_count(sink) == 1
    # No watcher scheduled on the second call. ``_invocation_llm_pending``
    # may have an entry from the first call, but no NEW watcher task.
    pending = plugin._invocation_llm_pending.get("inv-1") or {}
    watcher = pending.get("watcher") if isinstance(pending, dict) else None
    assert watcher is None or watcher.done()


async def test_before_model_no_short_circuit_for_unrelated_invocation() -> None:
    """The sticky bit is per-invocation. Cancelling ``inv-1`` does NOT
    short-circuit ``inv-2``."""
    plugin, _sink = _make_plugin_with_ctx()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request("inv-1"))
    # Sibling invocation must still proceed.
    inv2 = _FakeInvocationContext(invocation_id="inv-2")
    response = await plugin.before_model_callback(
        callback_context=_FakeCallbackContext(invocation_context=inv2),
        llm_request=_FakeLlmRequest(),
    )
    # Returning ``None`` means "proceed normally". The watcher should
    # have been scheduled for inv-2.
    assert response is None
    pending = plugin._invocation_llm_pending.get("inv-2") or {}
    watcher = pending.get("watcher") if isinstance(pending, dict) else None
    if watcher is not None:
        watcher.cancel()


# ---------------------------------------------------------------------------
# before_tool_callback — short-circuit + sticky bit
# ---------------------------------------------------------------------------


async def test_before_tool_short_circuits_with_cancelled_status() -> None:
    """AgentTool dispatch on a cancelled invocation short-circuits.

    Plain FunctionTool dispatches no longer short-circuit at this
    callback (Bug C / goldfive#211610) — see the companion FunctionTool
    test in tests/test_cooperative_cancellation.py.
    """
    plugin, sink = _make_plugin_with_ctx()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request("inv-1"))
    inv_ctx = _FakeInvocationContext(invocation_id="inv-1")
    response = await plugin.before_tool_callback(
        tool=_FakeAgentTool(name="researcher"),
        tool_args={"q": "x"},
        tool_context=_FakeToolContext(invocation_context=inv_ctx),
    )
    assert response == {"status": "cancelled"}
    assert plugin.is_invocation_cancelled("inv-1") is True
    assert _cancelled_event_count(sink) == 1


async def test_before_tool_sticky_short_circuit_no_double_emit() -> None:
    """Subsequent AgentTool calls on a cancelled invocation also return
    the cancelled status without re-emitting the sink event."""
    plugin, sink = _make_plugin_with_ctx()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request("inv-1"))
    inv_ctx = _FakeInvocationContext(invocation_id="inv-1")
    tool_ctx = _FakeToolContext(invocation_context=inv_ctx)

    # First AgentTool call — consumes the marker, emits InvocationCancelled.
    r1 = await plugin.before_tool_callback(
        tool=_FakeAgentTool(name="researcher"),
        tool_args={},
        tool_context=tool_ctx,
    )
    assert r1 == {"status": "cancelled"}
    assert _cancelled_event_count(sink) == 1
    # Second AgentTool call — sticky bit shorts the dispatch silently.
    r2 = await plugin.before_tool_callback(
        tool=_FakeAgentTool(name="reviewer"),
        tool_args={},
        tool_context=tool_ctx,
    )
    assert r2 == {"status": "cancelled"}
    assert _cancelled_event_count(sink) == 1  # no double-emit


# ---------------------------------------------------------------------------
# Watcher integration: after the watcher fires, the next
# before_model_callback must short-circuit (the demo-v12 regression).
# ---------------------------------------------------------------------------


async def test_watcher_fire_then_next_before_model_short_circuits() -> None:
    """Drive the watcher to completion, then call
    ``before_model_callback`` on the same invocation. Pre-fix the
    second callback returned None and the LLM dispatch proceeded —
    causing the watcher to fire AGAIN minutes later (demo-v12.log
    showed 4 firings on a single invocation_id). Post-fix: the second
    callback short-circuits via the sticky bit."""
    plugin, sink = _make_plugin_with_ctx()
    ctx = plugin._active_ctx

    # Drive the watcher with a tiny sleep so it fires synchronously.
    await plugin._run_llm_call_timeout_watcher(
        invocation_id="inv-1",
        timeout_s=0.01,
        ctx=ctx,
    )
    # Watcher set the cancel marker.
    assert plugin._cancel_state.get("inv-1") is not None

    inv_ctx = _FakeInvocationContext(invocation_id="inv-1")

    # Next before_model_callback — first time we see the marker.
    r1 = await plugin.before_model_callback(
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
        llm_request=_FakeLlmRequest(),
    )
    assert r1 is not None  # short-circuited
    assert _cancelled_event_count(sink) == 1
    # The marker is consumed, but the sticky bit is set.
    assert plugin.peek_cancel_for_invocation("inv-1") is None
    assert plugin.is_invocation_cancelled("inv-1") is True

    # Pre-fix bug: this second call returned None and scheduled a
    # fresh watcher. Post-fix: short-circuits.
    r2 = await plugin.before_model_callback(
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
        llm_request=_FakeLlmRequest(),
    )
    assert r2 is not None
    # No new sink emit, no live watcher task.
    assert _cancelled_event_count(sink) == 1
    pending = plugin._invocation_llm_pending.get("inv-1") or {}
    watcher = pending.get("watcher") if isinstance(pending, dict) else None
    assert watcher is None or watcher.done()


# ---------------------------------------------------------------------------
# Cleanup — the sticky set is released when the invocation ends so a
# future invocation_id collision doesn't inherit the cancel bit.
# ---------------------------------------------------------------------------


async def test_after_run_clears_sticky_cancelled_for_invocation() -> None:
    plugin, _sink = _make_plugin_with_ctx()
    plugin._cancelled_invocations.add("inv-1")
    inv_ctx = _FakeInvocationContext(invocation_id="inv-1")
    await plugin.after_run_callback(invocation_context=inv_ctx)
    assert "inv-1" not in plugin._cancelled_invocations


def test_clear_active_context_drops_sticky_set() -> None:
    plugin, _sink = _make_plugin_with_ctx()
    plugin._cancelled_invocations.update({"inv-1", "inv-2"})
    plugin.clear_active_context()
    assert plugin._cancelled_invocations == set()
