"""Reporting-tool specs and handlers.

The seven canonical reporting tools — the agent-facing contract for
driving the plan's task state machine and signalling plan mutations.
Each :class:`ReportingToolSpec` pairs a stable tool name with a JSON-schema
parameters block and an async handler. Handlers receive the decoded
arguments, the live :class:`Session`, and the bound :class:`Steerer`, and
route the call into the steerer's transition / drift pipeline.

Adapters materialise these specs into whatever native tool shape their
framework wants (ADK ``FunctionTool``, Claude Agent SDK tool blocks, …).
"""

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
# Handler shims
# ---------------------------------------------------------------------------


_ACK: dict[str, Any] = {"acknowledged": True}


def _str(args: dict[str, Any], key: str, default: str = "") -> str:
    v = args.get(key, default)
    if v is None:
        return default
    return str(v)


def _float(args: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = args.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(args: dict[str, Any], key: str, default: bool = True) -> bool:
    v = args.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"true", "1", "yes"}
    return bool(v) if v is not None else default


async def _handle_task_started(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _str(args, "task_id")
    detail = _str(args, "detail")
    if task_id:
        await steerer.mark_task_running(task_id, session=session, detail=detail)
    return dict(_ACK)


async def _handle_task_progress(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _str(args, "task_id")
    fraction = _float(args, "fraction")
    detail = _str(args, "detail")
    if task_id:
        await steerer.mark_task_progress(
            task_id, session=session, fraction=fraction, detail=detail
        )
    return dict(_ACK)


async def _handle_task_completed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _str(args, "task_id")
    summary = _str(args, "summary")
    artifacts_raw = args.get("artifacts")
    artifacts = {
        str(k): str(v) for k, v in (artifacts_raw or {}).items()
    } if isinstance(artifacts_raw, dict) else {}
    if task_id:
        await steerer.mark_task_completed(
            task_id, session=session, summary=summary, artifacts=artifacts
        )
    return dict(_ACK)


async def _handle_task_failed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _str(args, "task_id")
    reason = _str(args, "reason")
    recoverable = _bool(args, "recoverable", default=True)
    if task_id:
        await steerer.mark_task_failed(
            task_id,
            session=session,
            reason=reason,
            recoverable=recoverable,
        )
    return dict(_ACK)


async def _handle_task_blocked(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _str(args, "task_id")
    blocker = _str(args, "blocker")
    needed = _str(args, "needed")
    if task_id:
        await steerer.mark_task_blocked(
            task_id, session=session, blocker=blocker, needed=needed
        )
    return dict(_ACK)


async def _handle_new_work_discovered(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    parent_task_id = _str(args, "parent_task_id")
    title = _str(args, "title")
    description = _str(args, "description")
    assignee = _str(args, "assignee")
    await steerer.report_new_work_discovered(
        session=session,
        parent_task_id=parent_task_id,
        title=title,
        description=description,
        assignee=assignee,
    )
    return dict(_ACK)


async def _handle_plan_divergence(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    note = _str(args, "note")
    suggested_action = _str(args, "suggested_action")
    await steerer.report_plan_divergence(
        session=session,
        note=note,
        suggested_action=suggested_action,
    )
    return dict(_ACK)


# ---------------------------------------------------------------------------
# Parameter schemas (JSON Schema draft-07 style)
# ---------------------------------------------------------------------------


def _object_schema(
    *, required: list[str], properties: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_SCHEMA_TASK_STARTED = _object_schema(
    required=["task_id"],
    properties={
        "task_id": {"type": "string"},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_PROGRESS = _object_schema(
    required=["task_id"],
    properties={
        "task_id": {"type": "string"},
        "fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_COMPLETED = _object_schema(
    required=["task_id", "summary"],
    properties={
        "task_id": {"type": "string"},
        "summary": {"type": "string"},
        "artifacts": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
)

_SCHEMA_TASK_FAILED = _object_schema(
    required=["task_id", "reason"],
    properties={
        "task_id": {"type": "string"},
        "reason": {"type": "string"},
        "recoverable": {"type": "boolean"},
    },
)

_SCHEMA_TASK_BLOCKED = _object_schema(
    required=["task_id", "blocker"],
    properties={
        "task_id": {"type": "string"},
        "blocker": {"type": "string"},
        "needed": {"type": "string"},
    },
)

_SCHEMA_NEW_WORK_DISCOVERED = _object_schema(
    required=["parent_task_id", "title", "description"],
    properties={
        "parent_task_id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "assignee": {"type": "string"},
    },
)

_SCHEMA_PLAN_DIVERGENCE = _object_schema(
    required=["note"],
    properties={
        "note": {"type": "string"},
        "suggested_action": {"type": "string"},
    },
)


# ---------------------------------------------------------------------------
# Built-in tool specs
# ---------------------------------------------------------------------------


BUILTIN_REPORTING_TOOLS: list[ReportingToolSpec] = [
    ReportingToolSpec(
        name="report_task_started",
        description=(
            "Report that you are beginning work on a planned task. Call this "
            "BEFORE doing the actual work so the framework knows which task "
            "is currently in progress."
        ),
        parameters=_SCHEMA_TASK_STARTED,
        handler=_handle_task_started,
    ),
    ReportingToolSpec(
        name="report_task_progress",
        description=(
            "Report mid-task progress. Optional — only call if the task has "
            "meaningful sub-steps. 'fraction' is a 0.0-1.0 hint of how far "
            "through the task you are."
        ),
        parameters=_SCHEMA_TASK_PROGRESS,
        handler=_handle_task_progress,
    ),
    ReportingToolSpec(
        name="report_task_completed",
        description=(
            "Report that you have completed a planned task successfully. "
            "Call this AFTER producing the final output. 'summary' describes "
            "the result in one or two sentences."
        ),
        parameters=_SCHEMA_TASK_COMPLETED,
        handler=_handle_task_completed,
    ),
    ReportingToolSpec(
        name="report_task_failed",
        description=(
            "Report that you were unable to complete a planned task. "
            "'recoverable=True' lets the plan route around this failure; "
            "'recoverable=False' means the whole workflow should probably stop."
        ),
        parameters=_SCHEMA_TASK_FAILED,
        handler=_handle_task_failed,
    ),
    ReportingToolSpec(
        name="report_task_blocked",
        description=(
            "Report that you cannot currently proceed with a task. Use this "
            "when an external blocker prevents progress. 'blocker' describes "
            "what is in the way; 'needed' describes what would unblock you."
        ),
        parameters=_SCHEMA_TASK_BLOCKED,
        handler=_handle_task_blocked,
    ),
    ReportingToolSpec(
        name="report_new_work_discovered",
        description=(
            "Report that you've discovered additional work the plan doesn't "
            "know about. The framework will ask the planner to add this task "
            "as a child of 'parent_task_id'."
        ),
        parameters=_SCHEMA_NEW_WORK_DISCOVERED,
        handler=_handle_new_work_discovered,
    ),
    ReportingToolSpec(
        name="report_plan_divergence",
        description=(
            "Report that the current plan no longer matches what needs to "
            "happen. The framework will trigger an explicit replan."
        ),
        parameters=_SCHEMA_PLAN_DIVERGENCE,
        handler=_handle_plan_divergence,
    ),
]


__all__ = [
    "ReportingHandler",
    "ReportingToolSpec",
    "REPORTING_TOOL_NAMES",
    "BUILTIN_REPORTING_TOOLS",
]
