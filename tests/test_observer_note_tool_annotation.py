"""Observer-note channel surface 4 — tool-result annotation (PR 6).

AGENCY-PRESERVATION.md PR 6 delivery point 4: for loop-shaped drift, an
append-only attributed one-liner lands on the repeated tool's result, at the
moment of maximal relevance. ADK-gated (the plugin requires ``google.adk``);
the annotation's append-only discipline and the loop-kinds filter are the
contract pinned here. The shared queue mechanics it reuses (mark_delivered,
render_tool_annotation, peek_for_render kinds filter) are unit-tested in
``test_observer_note_queue.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SessionContext,
    make_adk_plugin,
)
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.observer_note_queue import ObserverNoteQueue  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import Session, Task  # noqa: E402


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _StubPlanner:
    async def generate(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class _ToolStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _InvCtxStub:
    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self.invocation_id = invocation_id
        self.agent = type("_A", (), {"name": agent_name})()


class _ToolCtxStub:
    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self._invocation_context = _InvCtxStub(invocation_id, agent_name)


def _ctx(session: Session, steerer: Any) -> SessionContext:
    return SessionContext(
        session=session,
        steerer=steerer,
        task=Task(id="t1", title="research", assignee_agent_id="worker"),
        tool_handlers={},
        host_agent_name="test_agent",
    )


def _enqueue_loop_note(session: Session, *, drift_id: str = "d1") -> None:
    ObserverNoteQueue.for_session(session).enqueue(
        body="Observation: `search_web` was invoked 5 times",
        observation="`search_web` was invoked 5 times with identical arguments",
        severity="warning",
        drift_id=drift_id,
        kind="looping_tool_call",
        task_id="t1",
        agent_id="worker",
        turn=0,
    )


def _steerer(*, active: bool = False) -> DefaultSteerer:
    """Build a ``request_context`` steerer.

    Bare (``active=False``) is PASSIVE — ``observation_only=True`` is the
    shipped production default and the branch's §5.4 shadow-campaign
    config. Surface 4 must consume-but-not-annotate in that mode. Tests
    that assert the active annotation delivery opt in via ``active=True``.
    """
    return DefaultSteerer(
        steering_config=SteeringConfig(
            signal_channel="request_context",
            observation_only=not active,
        )
    )


async def test_annotation_appends_without_modifying_real_result() -> None:
    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(run_id="r1")
    steerer = _steerer(active=True)
    ctx = _ctx(session, steerer)
    _enqueue_loop_note(session)

    result = {"content": [{"type": "text", "text": "search results..."}]}
    annotated = await plugin._maybe_annotate_tool_result(ctx, result)

    assert annotated is not None
    # Append-only: every real key is preserved verbatim.
    assert annotated["content"] == result["content"]
    # The annotation rides a reserved, namespaced key.
    assert annotated["goldfive_observer_note"].startswith("[goldfive observer:")
    # The note is marked delivered (exactly-once) so the block surfaces skip it.
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is True


async def test_annotation_skips_non_loop_notes() -> None:
    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(run_id="r1")
    ctx = _ctx(session, _steerer())
    # A goal-drift note is not loop-shaped — surface 4 leaves it for the
    # block surfaces.
    ObserverNoteQueue.for_session(session).enqueue(
        body="Observation: drifting", observation="drifting", severity="critical",
        drift_id="g1", kind="goal_drift", task_id="t1", turn=0,
    )
    annotated = await plugin._maybe_annotate_tool_result(ctx, {"content": "x"})
    assert annotated is None
    assert ObserverNoteQueue.for_session(session).get("g1").delivered is False


async def test_annotation_noop_for_non_dict_result() -> None:
    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(run_id="r1")
    ctx = _ctx(session, _steerer())
    _enqueue_loop_note(session)
    # A non-mapping result can't be annotated append-only; leave it untouched.
    assert await plugin._maybe_annotate_tool_result(ctx, "bare-string") is None
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is False


async def test_annotation_noop_in_legacy_channel() -> None:
    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(run_id="r1")
    legacy = DefaultSteerer(
        steering_config=SteeringConfig(signal_channel="legacy_user_message")
    )
    ctx = _ctx(session, legacy)
    _enqueue_loop_note(session)
    # Legacy default: surface 4 is inert (returns None, note untouched).
    assert await plugin._maybe_annotate_tool_result(ctx, {"content": "x"}) is None
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is False


async def test_annotation_suppressed_under_observation_only() -> None:
    """Surface-4 kill-switch gate: request_context + observation_only=True.

    The §5.4 shadow-campaign config (request_context channel, passive
    kill-switch — the branch default) must NOT annotate the tool result;
    nothing reaches the model. But the note is still consumed as a
    dry-run delivery (``mark_delivered`` runs for decision parity) so
    pacing / coalescing behave identically to the active path.
    """
    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(run_id="r1")
    # Bare steerer -> observation_only=True (production default).
    ctx = _ctx(session, _steerer())
    _enqueue_loop_note(session)

    result = {"content": [{"type": "text", "text": "search results..."}]}
    annotated = await plugin._maybe_annotate_tool_result(ctx, result)

    # No result replacement — after_tool_callback returns None and the
    # real tool result passes through UNANNOTATED.
    assert annotated is None
    # Dry-run consume: the note is marked delivered for decision parity.
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is True


async def _drive_repeated_loop(steerer: DefaultSteerer) -> tuple[Any, Session]:
    """Drive a real repeated tool call through ``after_tool_callback``.

    Exact-repeat x3 trips the tool-loop tracker's WARNING tier, so
    ``after_tool_callback`` reaches surface 4 (it early-returns when the
    call fired no drift). A loop note is pre-enqueued so the surface has
    something to render. Returns the final callback return value and the
    session.
    """
    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(run_id="r1")
    steerer.bind(sinks=[_ListSink()], planner=_StubPlanner())
    ctx = _ctx(session, steerer)
    plugin.set_active_context(ctx)
    _enqueue_loop_note(session)

    tool = _ToolStub("patch_file")
    args = {"path": "a.py", "diff": "x"}
    tool_ctx = _ToolCtxStub("inv-surface4", "worker")
    result = {"content": [{"type": "text", "text": "patched"}], "ok": True}
    returned: Any = None
    for _ in range(3):  # exact-repeat x3 -> WARNING loop drift
        returned = await plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_ctx, result=dict(result)
        )
    return returned, session


async def test_after_tool_callback_passes_result_through_unannotated_when_passive() -> None:
    """End-to-end negative control through ``after_tool_callback``.

    A repeated tool call under signal_channel="request_context" +
    observation_only=True (the branch default) must NOT annotate the tool
    result — the callback returns ``None`` so ADK keeps the real result
    verbatim and nothing reaches the model — yet the note is still
    consumed as a dry-run delivery for decision parity.
    """
    returned, session = await _drive_repeated_loop(_steerer())  # passive

    # Passive: no annotated replacement — never carries the reserved key.
    assert returned is None
    if isinstance(returned, dict):  # defensive
        assert "goldfive_observer_note" not in returned
    # Consumed as a dry-run delivery (mark_delivered ran for parity).
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is True


async def test_after_tool_callback_annotates_result_when_active() -> None:
    """End-to-end positive control through ``after_tool_callback``.

    The same drive under active mode (observation_only=False) DOES return
    a result carrying the ``goldfive_observer_note`` annotation, with the
    real result preserved verbatim.
    """
    returned, session = await _drive_repeated_loop(_steerer(active=True))

    assert isinstance(returned, dict)
    assert returned["ok"] is True  # real result preserved verbatim
    assert returned["goldfive_observer_note"].startswith("[goldfive observer:")
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is True
