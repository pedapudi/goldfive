"""Required-field validation tests for canonical reporting tools.

The v15 cascade root cause: handlers ``_str``-coerced missing /
empty content fields to the empty string and forwarded the empty
payload onto the steerer, where it became drift detail like
``"new work under : : "`` — a semantically empty signal that the
planner correctly declined to act on, leaving the agent in a no-op
revision tool loop.

The fix: every handler validates each schema-``required`` field at
entry and returns a structured ``missing_required_field`` rejection
when a required field is missing, ``None``, or whitespace-only. The
steerer is NOT driven on rejection.

These tests pin:

1. Each handler with a non-empty schema ``required[]`` rejects calls
   with a missing / empty / whitespace-only required field.
2. The rejection shape is ``{"acknowledged": False, "error":
   "missing_required_field", "tool": ..., "field": ..., ...}`` so the
   LLM can self-correct on the next turn.
3. The steerer is NOT driven on rejection — no plan mutation, no
   sink events, no refine call.
4. A literal ``0`` / ``False`` is accepted (only absence/null/empty
   string is a violation).
5. Schemas with empty ``required[]`` (``report_task_started``,
   ``report_task_progress``) are unaffected — they keep working when
   called with only ``task_id``.

See goldfive#271 (v15 cascade root cause).
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


class StubPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return None


def _plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="A", status=TaskStatus.RUNNING),
            Task(id="t2", title="B"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _fresh() -> tuple[DefaultSteerer, Session, ListSink, StubPlanner]:
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=_plan(),
    )
    sink = ListSink()
    planner = StubPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    return steerer, session, sink, planner


def _tool(name: str) -> ReportingToolSpec:
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"missing builtin tool {name!r}")


def _assert_rejection(out: dict[str, Any], *, tool: str, field: str) -> None:
    """Pin the canonical rejection shape returned by ``_validate_required``."""
    assert out.get("acknowledged") is False, out
    assert out.get("error") == "missing_required_field", out
    assert out.get("tool") == tool, out
    assert out.get("field") == field, out
    assert out.get("reason") in {"missing", "null", "empty"}, out
    assert isinstance(out.get("expected"), dict), out
    assert isinstance(out.get("required"), list), out
    assert field in out["required"], out
    assert out.get("message"), out


# ---------------------------------------------------------------------------
# report_new_work_discovered — the canonical v15 reproducer
# ---------------------------------------------------------------------------


async def test_new_work_discovered_rejects_empty_parent_task_id() -> None:
    """The exact v15 cascade case: empty content fields propagate as
    drift detail like ``"new work under : : "`` — must reject."""
    steerer, session, sink, planner = _fresh()
    out = await _tool("report_new_work_discovered").handler(
        {
            "parent_task_id": "",
            "title": "dig deeper",
            "description": "validate source freshness",
            "assignee": "analyst",
        },
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_new_work_discovered", field="parent_task_id")
    # No drift: the steerer must NOT have been driven.
    assert planner.refine_calls == []
    assert sink.events == []


async def test_new_work_discovered_rejects_missing_title() -> None:
    steerer, session, sink, planner = _fresh()
    out = await _tool("report_new_work_discovered").handler(
        {"parent_task_id": "t1", "description": "validate source freshness"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_new_work_discovered", field="title")
    assert planner.refine_calls == []
    assert sink.events == []


async def test_new_work_discovered_rejects_whitespace_description() -> None:
    steerer, session, sink, planner = _fresh()
    out = await _tool("report_new_work_discovered").handler(
        {
            "parent_task_id": "t1",
            "title": "dig deeper",
            "description": "   \t\n   ",
        },
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_new_work_discovered", field="description")
    assert planner.refine_calls == []
    assert sink.events == []


async def test_new_work_discovered_rejects_null_description() -> None:
    steerer, session, sink, planner = _fresh()
    out = await _tool("report_new_work_discovered").handler(
        {
            "parent_task_id": "t1",
            "title": "dig deeper",
            "description": None,
        },
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_new_work_discovered", field="description")
    assert out.get("reason") == "null"
    assert planner.refine_calls == []
    assert sink.events == []


async def test_new_work_discovered_accepts_valid_payload() -> None:
    """Sanity: the happy path still drives the steerer."""
    steerer, session, _sink, planner = _fresh()
    out = await _tool("report_new_work_discovered").handler(
        {
            "parent_task_id": "t1",
            "title": "dig deeper",
            "description": "validate source freshness",
            "assignee": "analyst",
        },
        session,
        steerer,
    )
    assert out == {"acknowledged": True}
    assert len(planner.refine_calls) == 1


# ---------------------------------------------------------------------------
# report_task_completed — required: ["summary"]
# ---------------------------------------------------------------------------


async def test_task_completed_rejects_empty_summary() -> None:
    steerer, session, sink, _planner = _fresh()
    out = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": ""},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_task_completed", field="summary")
    # Plan state must be untouched.
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    assert sink.events == []


async def test_task_completed_rejects_missing_summary() -> None:
    steerer, session, sink, _planner = _fresh()
    out = await _tool("report_task_completed").handler(
        {"task_id": "t1"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_task_completed", field="summary")
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    assert sink.events == []


async def test_task_completed_validates_before_task_id_check() -> None:
    """A missing required content field is reported even if task_id is
    also missing — the schema violation is the actionable signal."""
    steerer, session, _sink, _planner = _fresh()
    out = await _tool("report_task_completed").handler(
        {},
        session,
        steerer,
    )
    # Required-field validation runs first, so the LLM sees the most
    # specific signal it can act on.
    assert out.get("acknowledged") is False
    assert out.get("error") in {"missing_required_field", "missing_task_id"}


# ---------------------------------------------------------------------------
# report_task_failed — required: ["reason"]
# ---------------------------------------------------------------------------


async def test_task_failed_rejects_empty_reason() -> None:
    steerer, session, sink, planner = _fresh()
    out = await _tool("report_task_failed").handler(
        {"task_id": "t1", "reason": "", "recoverable": True},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_task_failed", field="reason")
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    assert planner.refine_calls == []
    assert sink.events == []


async def test_task_failed_rejects_missing_reason() -> None:
    steerer, session, _sink, planner = _fresh()
    out = await _tool("report_task_failed").handler(
        {"task_id": "t1"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_task_failed", field="reason")
    assert planner.refine_calls == []


# ---------------------------------------------------------------------------
# report_task_blocked — required: ["blocker"]
# ---------------------------------------------------------------------------


async def test_task_blocked_rejects_empty_blocker() -> None:
    steerer, session, sink, planner = _fresh()
    out = await _tool("report_task_blocked").handler(
        {"task_id": "t1", "blocker": "", "needed": "anything"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_task_blocked", field="blocker")
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    assert planner.refine_calls == []
    assert sink.events == []


async def test_task_blocked_rejects_missing_blocker() -> None:
    steerer, session, _sink, planner = _fresh()
    out = await _tool("report_task_blocked").handler(
        {"task_id": "t1"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_task_blocked", field="blocker")
    assert planner.refine_calls == []


# ---------------------------------------------------------------------------
# report_plan_divergence — required: ["note"]
# ---------------------------------------------------------------------------


async def test_plan_divergence_rejects_empty_note() -> None:
    steerer, session, sink, planner = _fresh()
    out = await _tool("report_plan_divergence").handler(
        {"note": "", "suggested_action": "redo"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_plan_divergence", field="note")
    assert session.divergence_flag is False
    assert planner.refine_calls == []
    assert sink.events == []


async def test_plan_divergence_rejects_missing_note() -> None:
    steerer, session, _sink, planner = _fresh()
    out = await _tool("report_plan_divergence").handler({}, session, steerer)
    _assert_rejection(out, tool="report_plan_divergence", field="note")
    assert session.divergence_flag is False
    assert planner.refine_calls == []


# ---------------------------------------------------------------------------
# report_awaiting_approval — required: ["prompt"]
# ---------------------------------------------------------------------------


async def test_awaiting_approval_rejects_empty_prompt() -> None:
    steerer, session, sink, _planner = _fresh()
    out = await _tool("report_awaiting_approval").handler(
        {"task_id": "t1", "prompt": "", "timeout_ms": 0},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_awaiting_approval", field="prompt")
    # Task status must NOT have been driven to BLOCKED.
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    assert sink.events == []
    assert session.pending_approvals == {}


async def test_awaiting_approval_rejects_missing_prompt() -> None:
    steerer, session, _sink, _planner = _fresh()
    out = await _tool("report_awaiting_approval").handler(
        {"task_id": "t1"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="report_awaiting_approval", field="prompt")


# ---------------------------------------------------------------------------
# declare_task_skipped / declare_task_not_needed — required: ["reason"]
# ---------------------------------------------------------------------------


async def test_declare_task_skipped_rejects_empty_reason() -> None:
    steerer, session, sink, _planner = _fresh()
    out = await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": ""},
        session,
        steerer,
    )
    _assert_rejection(out, tool="declare_task_skipped", field="reason")
    # Plan state untouched + no declaration recorded.
    assert sink.events == []


async def test_declare_task_skipped_rejects_whitespace_reason() -> None:
    steerer, session, sink, _planner = _fresh()
    out = await _tool("declare_task_skipped").handler(
        {"task_id": "t1", "reason": "  "},
        session,
        steerer,
    )
    _assert_rejection(out, tool="declare_task_skipped", field="reason")
    assert sink.events == []


async def test_declare_task_not_needed_rejects_empty_reason() -> None:
    steerer, session, sink, _planner = _fresh()
    out = await _tool("declare_task_not_needed").handler(
        {"task_id": "t1", "reason": ""},
        session,
        steerer,
    )
    _assert_rejection(out, tool="declare_task_not_needed", field="reason")
    assert sink.events == []


async def test_declare_task_not_needed_rejects_missing_reason() -> None:
    steerer, session, _sink, _planner = _fresh()
    out = await _tool("declare_task_not_needed").handler(
        {"task_id": "t1"},
        session,
        steerer,
    )
    _assert_rejection(out, tool="declare_task_not_needed", field="reason")


# ---------------------------------------------------------------------------
# Schemas with empty required[] — must keep working unchanged
# ---------------------------------------------------------------------------


async def test_task_started_no_required_fields_accepts_minimal_call() -> None:
    """report_task_started has required=[] — only task_id (via pin or
    args) is needed. Empty optional fields must NOT trigger a
    rejection."""
    steerer, session, _sink, _planner = _fresh()
    # Reset t1 to PENDING so the started transition is legal.
    session.plan.tasks[0].status = TaskStatus.PENDING
    out = await _tool("report_task_started").handler({"task_id": "t1"}, session, steerer)
    assert out == {"acknowledged": True}
    assert session.plan.tasks[0].status is TaskStatus.RUNNING


async def test_task_progress_no_required_fields_accepts_minimal_call() -> None:
    """report_task_progress has required=[] and accepts a literal 0.0
    for fraction without rejection (numeric zero is a valid value)."""
    steerer, session, _sink, _planner = _fresh()
    out = await _tool("report_task_progress").handler(
        {"task_id": "t1", "fraction": 0.0}, session, steerer
    )
    assert out == {"acknowledged": True}


# ---------------------------------------------------------------------------
# Coverage: every reporting tool whose schema requires content fields
# rejects an empty payload. Acts as a contract test against the
# BUILTIN_REPORTING_TOOLS catalogue so a future tool gains validation
# automatically.
# ---------------------------------------------------------------------------


async def test_every_required_field_handler_rejects_empty_payload() -> None:
    """Iterate every spec; any tool with non-empty ``required[]`` must
    reject a call missing all required fields."""
    for spec in BUILTIN_REPORTING_TOOLS:
        required = (spec.parameters or {}).get("required") or []
        if not required:
            continue
        steerer, session, _sink, _planner = _fresh()
        # Provide task_id where the schema lists it as a property so
        # the rejection is specifically about the content field, not
        # about missing_task_id.
        args: dict[str, Any] = {}
        properties = (spec.parameters or {}).get("properties") or {}
        if "task_id" in properties:
            args["task_id"] = "t1"
        out = await spec.handler(args, session, steerer)
        assert out.get("acknowledged") is False, (spec.name, out)
        # The rejection MUST be the validation shape (not, say, an
        # invalid-transition shape that runs after validation).
        assert out.get("error") == "missing_required_field", (spec.name, out)
        assert out.get("field") in required, (spec.name, out)
