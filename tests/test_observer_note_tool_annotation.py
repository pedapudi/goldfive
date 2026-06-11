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


def _steerer() -> DefaultSteerer:
    return DefaultSteerer(
        steering_config=SteeringConfig(signal_channel="request_context")
    )


async def test_annotation_appends_without_modifying_real_result() -> None:
    plugin = make_adk_plugin(host_agent_name="test_agent")
    session = Session(run_id="r1")
    steerer = _steerer()
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
