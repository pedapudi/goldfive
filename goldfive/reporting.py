from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.types import Session


# Framework-agnostic async handler signature.
# Handlers receive the tool-call arguments (already decoded to a dict),
# the live Session, and the Steerer, and return a JSON-serializable dict.
ReportingHandler = Callable[
    [dict[str, Any], "Session", "Steerer"],
    Awaitable[dict[str, Any]],
]


@dataclasses.dataclass
class ReportingToolSpec:
    """Framework-agnostic spec for a reporting tool.

    Adapters translate one of these into whatever tool representation their
    underlying framework expects (e.g., an ADK ``FunctionTool`` or a Claude
    Agent SDK tool definition). The canonical set of tool names is pinned in
    :data:`REPORTING_TOOL_NAMES`.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters
    handler: ReportingHandler


# The seven canonical reporting tool names. These are a stable contract: the
# adapter must surface tools with exactly these names so that the Steerer can
# interpret them uniformly across frameworks. Do not rename.
REPORTING_TOOL_NAMES: tuple[str, ...] = (
    "report_task_started",
    "report_task_progress",
    "report_task_completed",
    "report_task_failed",
    "report_task_blocked",
    "report_new_work_discovered",
    "report_plan_divergence",
)


# ---------------------------------------------------------------------------
# Built-in handlers
#
# Each handler mutates the live :class:`Session` via the :class:`Steerer`
# (for status transitions) or directly (for progress / notes). The steerer
# is responsible for fanning the corresponding event envelopes out to the
# sinks, so handlers stay small and framework-agnostic.
# ---------------------------------------------------------------------------


async def _report_task_started(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    from goldfive.types import TaskStatus

    task_id = str(args.get("task_id") or "")
    detail = str(args.get("detail") or "")
    if task_id:
        await steerer.transition(task_id, TaskStatus.RUNNING, detail=detail, session=session)
    return {"ok": True}


async def _report_task_progress(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    task_id = str(args.get("task_id") or "")
    fraction = float(args.get("fraction") or 0.0)
    detail = str(args.get("detail") or "")
    if task_id:
        session.task_progress[task_id] = max(0.0, min(1.0, fraction))
        if detail:
            session.agent_notes[task_id] = detail
        await steerer.observe(
            {"kind": "task_progress", "task_id": task_id, "fraction": fraction, "detail": detail},
            session,
        )
    return {"ok": True}


async def _report_task_completed(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    from goldfive.types import TaskStatus

    task_id = str(args.get("task_id") or "")
    summary = str(args.get("summary") or "")
    if task_id:
        if summary:
            session.completed_results[task_id] = summary
        await steerer.transition(task_id, TaskStatus.COMPLETED, detail=summary, session=session)
    return {"ok": True}


async def _report_task_failed(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    from goldfive.types import TaskStatus

    task_id = str(args.get("task_id") or "")
    reason = str(args.get("reason") or "")
    if task_id:
        await steerer.transition(task_id, TaskStatus.FAILED, detail=reason, session=session)
    return {"ok": True}


async def _report_task_blocked(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    from goldfive.types import TaskStatus

    task_id = str(args.get("task_id") or "")
    blocker = str(args.get("blocker") or "")
    if task_id:
        await steerer.transition(task_id, TaskStatus.BLOCKED, detail=blocker, session=session)
    return {"ok": True}


async def _report_new_work_discovered(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity

    parent = str(args.get("parent_task_id") or "")
    title = str(args.get("title") or "")
    description = str(args.get("description") or "")
    session.divergence_flag = True
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.INFO,
        detail=f"{title}: {description}",
        current_task_id=parent,
    )
    await steerer.observe({"kind": "drift", "drift": drift}, session)
    return {"ok": True}


async def _report_plan_divergence(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity

    note = str(args.get("note") or "")
    suggested = str(args.get("suggested_action") or "")
    session.divergence_flag = True
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail=f"{note} | suggested: {suggested}" if suggested else note,
        current_task_id=session.current_task_id,
    )
    await steerer.observe({"kind": "drift", "drift": drift}, session)
    return {"ok": True}


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


#: The seven canonical reporting tools, ready to hand to
#: :meth:`AgentAdapter.register_reporting_tools`. Immutable at the module
#: level; callers that want to customise descriptions should construct
#: their own :class:`ReportingToolSpec` list.
BUILTIN_REPORTING_TOOLS: list[ReportingToolSpec] = [
    ReportingToolSpec(
        name="report_task_started",
        description="Report that work on a task has started.",
        parameters=_schema(
            {
                "task_id": {"type": "string"},
                "detail": {"type": "string"},
            },
            ["task_id"],
        ),
        handler=_report_task_started,
    ),
    ReportingToolSpec(
        name="report_task_progress",
        description="Report incremental progress on a task (0-1 fraction).",
        parameters=_schema(
            {
                "task_id": {"type": "string"},
                "fraction": {"type": "number"},
                "detail": {"type": "string"},
            },
            ["task_id"],
        ),
        handler=_report_task_progress,
    ),
    ReportingToolSpec(
        name="report_task_completed",
        description="Report that a task has finished successfully.",
        parameters=_schema(
            {
                "task_id": {"type": "string"},
                "summary": {"type": "string"},
                "artifacts": {"type": "object"},
            },
            ["task_id", "summary"],
        ),
        handler=_report_task_completed,
    ),
    ReportingToolSpec(
        name="report_task_failed",
        description="Report that a task has failed.",
        parameters=_schema(
            {
                "task_id": {"type": "string"},
                "reason": {"type": "string"},
                "recoverable": {"type": "boolean"},
            },
            ["task_id", "reason"],
        ),
        handler=_report_task_failed,
    ),
    ReportingToolSpec(
        name="report_task_blocked",
        description="Report that a task is blocked on an external dependency.",
        parameters=_schema(
            {
                "task_id": {"type": "string"},
                "blocker": {"type": "string"},
                "needed": {"type": "string"},
            },
            ["task_id", "blocker"],
        ),
        handler=_report_task_blocked,
    ),
    ReportingToolSpec(
        name="report_new_work_discovered",
        description="Report a new task surfaced during execution.",
        parameters=_schema(
            {
                "parent_task_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "assignee": {"type": "string"},
            },
            ["parent_task_id", "title", "description"],
        ),
        handler=_report_new_work_discovered,
    ),
    ReportingToolSpec(
        name="report_plan_divergence",
        description="Report that the plan no longer matches reality.",
        parameters=_schema(
            {
                "note": {"type": "string"},
                "suggested_action": {"type": "string"},
            },
            ["note"],
        ),
        handler=_report_plan_divergence,
    ),
]


__all__ = [
    "BUILTIN_REPORTING_TOOLS",
    "REPORTING_TOOL_NAMES",
    "ReportingHandler",
    "ReportingToolSpec",
]
