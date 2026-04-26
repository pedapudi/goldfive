"""Tests for ``declare_task_skipped`` / ``declare_task_not_needed``.

goldfive#271 Phase 3 — structural declaration tools. Observability-only:
the handlers emit a ``TaskDeclarationReceived`` event and NEVER mutate
plan state. The imperative ``report_task_*`` surface remains the only
path that drives the steerer; declarations are advisory signals queued
for the next refine to consume.

These tests pin:

1. Both handlers emit a ``task_declaration_received`` envelope with the
   right ``kind``, ``task_id``, ``reason``, and ``source_signal``.
2. Handlers NEVER mutate plan state (status, edges, tasks).
3. Idempotent on duplicate declarations of the same ``(kind, task_id)``
   pair — the second call is a no-op (no second event emitted, no
   recorded body rewritten).
4. The two tools are wired into ``BUILTIN_REPORTING_TOOLS`` and
   ``REPORTING_TOOL_NAMES``.
5. Distinct kinds on the same task each emit independently.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.reporting import (  # noqa: E402
    BUILTIN_REPORTING_TOOLS,
    DECLARATION_KIND_NOT_NEEDED,
    DECLARATION_KIND_SKIPPED,
    DECLARATION_KINDS,
    DECLARATIONS_KEY,
    REPORTING_TOOL_NAMES,
    ReportingToolSpec,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


def _plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="A"), Task(id="t2", title="B")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _fresh() -> tuple[DefaultSteerer, Session, ListSink]:
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=_plan(),
    )
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)
    return steerer, session, sink


def _tool(name: str) -> ReportingToolSpec:
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"builtin tool {name!r} missing")


def _declarations(events: list[Any]) -> list[Any]:
    """Return only the ``task_declaration_received`` dict envelopes."""
    return [
        e for e in events if isinstance(e, dict) and e.get("kind") == "task_declaration_received"
    ]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_declarations_in_canonical_names() -> None:
    """Both declaration tools appear in ``REPORTING_TOOL_NAMES``."""
    assert "declare_task_skipped" in REPORTING_TOOL_NAMES
    assert "declare_task_not_needed" in REPORTING_TOOL_NAMES


def test_declaration_kinds_vocabulary() -> None:
    """The ``DECLARATION_KINDS`` tuple holds the two known kinds."""
    assert DECLARATION_KINDS == (DECLARATION_KIND_SKIPPED, DECLARATION_KIND_NOT_NEEDED)
    assert DECLARATION_KIND_SKIPPED == "skipped"
    assert DECLARATION_KIND_NOT_NEEDED == "not_needed"


def test_declarations_in_builtin_tools() -> None:
    """Both declaration tool specs are wired into ``BUILTIN_REPORTING_TOOLS``."""
    skipped = _tool("declare_task_skipped")
    not_needed = _tool("declare_task_not_needed")
    assert callable(skipped.handler)
    assert callable(not_needed.handler)
    assert skipped.parameters.get("type") == "object"
    assert not_needed.parameters.get("type") == "object"
    # Both schemas require ``reason``.
    assert skipped.parameters.get("required") == ["reason"]
    assert not_needed.parameters.get("required") == ["reason"]
    # Neither requires ``task_id`` (defaulting from session pin is the
    # contract — same as report_task_*).
    assert "task_id" not in skipped.parameters.get("required", [])
    assert "task_id" not in not_needed.parameters.get("required", [])


# ---------------------------------------------------------------------------
# declare_task_skipped
# ---------------------------------------------------------------------------


async def test_declare_task_skipped_emits_event() -> None:
    """Handler emits a single ``task_declaration_received`` envelope."""
    steerer, session, sink = _fresh()
    out = await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "duplicate of upstream work"}, session, steerer
    )
    assert out == {"acknowledged": True}
    decls = _declarations(sink.events)
    assert len(decls) == 1
    payload = decls[0]["payload"]
    assert payload["kind"] == "skipped"
    assert payload["task_id"] == "t1"
    assert payload["reason"] == "duplicate of upstream work"
    assert payload["source_signal"] == "DECLARATION"


async def test_declare_task_skipped_does_not_mutate_plan() -> None:
    """Plan state is NEVER touched by the declaration handler."""
    steerer, session, sink = _fresh()
    plan_before = session.plan
    statuses_before = [t.status for t in session.plan.tasks]
    edges_before = list(session.plan.edges)
    await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "skip me"}, session, steerer
    )
    # Plan identity unchanged.
    assert session.plan is plan_before
    # Task statuses unchanged.
    assert [t.status for t in session.plan.tasks] == statuses_before
    # Edges unchanged.
    assert session.plan.edges == edges_before
    # No task transitioned.
    assert session.plan.tasks[0].status is TaskStatus.PENDING


async def test_declare_task_skipped_idempotent_on_duplicate() -> None:
    """Second declaration of the same kind+task is a no-op."""
    steerer, session, sink = _fresh()
    await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "first"}, session, steerer
    )
    out = await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "second"}, session, steerer
    )
    assert out == {"acknowledged": True, "idempotent": True}
    # Only ONE event emitted.
    decls = _declarations(sink.events)
    assert len(decls) == 1
    # The recorded body keeps the FIRST reason.
    assert decls[0]["payload"]["reason"] == "first"


async def test_declare_task_skipped_records_to_session_state() -> None:
    """Declaration is recorded on session.state under DECLARATIONS_KEY."""
    steerer, session, sink = _fresh()
    await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "noop"}, session, steerer
    )
    decls = session.state.get(DECLARATIONS_KEY)
    assert isinstance(decls, dict)
    body = decls.get("skipped:t1")
    assert body is not None
    assert body["kind"] == "skipped"
    assert body["task_id"] == "t1"
    assert body["reason"] == "noop"


async def test_declare_task_skipped_missing_task_id() -> None:
    """Missing task_id returns the canonical missing_task_id rejection."""
    steerer, session, sink = _fresh()
    out = await _tool("declare_task_skipped").handler({"reason": "no id"}, session, steerer)
    assert out["acknowledged"] is False
    assert out["error"] == "missing_task_id"
    assert out["tool"] == "declare_task_skipped"
    # No event emitted.
    assert _declarations(sink.events) == []


# ---------------------------------------------------------------------------
# declare_task_not_needed
# ---------------------------------------------------------------------------


async def test_declare_task_not_needed_emits_event() -> None:
    steerer, session, sink = _fresh()
    out = await _tool("declare_task_not_needed").handler(
        {"task_id": "t2", "reason": "satisfied by upstream"}, session, steerer
    )
    assert out == {"acknowledged": True}
    decls = _declarations(sink.events)
    assert len(decls) == 1
    payload = decls[0]["payload"]
    assert payload["kind"] == "not_needed"
    assert payload["task_id"] == "t2"
    assert payload["reason"] == "satisfied by upstream"
    assert payload["source_signal"] == "DECLARATION"


async def test_declare_task_not_needed_does_not_mutate_plan() -> None:
    """``declare_task_not_needed`` does NOT stamp NOT_NEEDED on the task.

    The reconciler / next refine consumes the declaration as a signal;
    the handler itself remains observability-only. Pin this contract
    so a future "let's just stamp it" refactor doesn't accidentally
    bypass the structural-steering machinery.
    """
    steerer, session, sink = _fresh()
    plan_before = session.plan
    await _tool("declare_task_not_needed").handler(
        {"task_id": "t2", "reason": "noop"}, session, steerer
    )
    assert session.plan is plan_before
    # Task t2 stayed PENDING — handler did NOT mark it NOT_NEEDED.
    t2 = next(t for t in session.plan.tasks if t.id == "t2")
    assert t2.status is TaskStatus.PENDING


async def test_declare_task_not_needed_idempotent() -> None:
    steerer, session, sink = _fresh()
    await _tool("declare_task_not_needed").handler(
        {"task_id": "t2", "reason": "first"}, session, steerer
    )
    out = await _tool("declare_task_not_needed").handler(
        {"task_id": "t2", "reason": "second"}, session, steerer
    )
    assert out == {"acknowledged": True, "idempotent": True}
    decls = _declarations(sink.events)
    assert len(decls) == 1


# ---------------------------------------------------------------------------
# Cross-kind
# ---------------------------------------------------------------------------


async def test_distinct_kinds_on_same_task_each_emit() -> None:
    """``skipped`` + ``not_needed`` on the same task are tracked separately."""
    steerer, session, sink = _fresh()
    await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "skip"}, session, steerer
    )
    await _tool("declare_task_not_needed").handler(
        {"task_id": "t1", "reason": "don't need"}, session, steerer
    )
    decls = _declarations(sink.events)
    assert len(decls) == 2
    kinds = {d["payload"]["kind"] for d in decls}
    assert kinds == {"skipped", "not_needed"}


async def test_distinct_tasks_each_emit_independently() -> None:
    """Same kind on two different task ids emits twice."""
    steerer, session, sink = _fresh()
    await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "first task"}, session, steerer
    )
    await _tool("declare_task_skipped").handler(
        {"task_id": "t2", "reason": "second task"}, session, steerer
    )
    decls = _declarations(sink.events)
    assert len(decls) == 2
    task_ids = {d["payload"]["task_id"] for d in decls}
    assert task_ids == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Defaulting from session.state pin
# ---------------------------------------------------------------------------


async def test_declare_task_skipped_defaults_from_pin() -> None:
    """Omitted task_id resolves from the goldfive.current_task_id pin.

    Mirrors the report_task_* defaulting contract (goldfive#191) — the
    declaration tools use the same precedence so an agent that drops
    the explicit arg still gets routed correctly.
    """
    from goldfive import orchestration_state as _ostate

    steerer, session, sink = _fresh()
    _ostate.write(session.state, _ostate.KEY_CURRENT_TASK_ID, "t1")
    out = await _tool("declare_task_skipped").handler({"reason": "implicit task"}, session, steerer)
    assert out == {"acknowledged": True}
    decls = _declarations(sink.events)
    assert len(decls) == 1
    assert decls[0]["payload"]["task_id"] == "t1"
