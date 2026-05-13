"""JSON-schema parameter blocks for the canonical reporting tools.

These schemas are the over-the-wire contract: adapters embed them
verbatim into native tool-call definitions (ADK ``FunctionTool``
parameters, Claude Agent SDK tool input schemas, …) so the LLM sees a
stable shape for every reporting tool regardless of framework. Each
``_SCHEMA_*`` constant is intentionally a plain ``dict[str, Any]`` —
the marshaling layer is responsible for any adapter-specific
transformation (e.g. stripping ``additionalProperties`` for engines
that don't honour it).

NOTE: ``task_id`` is intentionally omitted from every schema's
``required`` list (goldfive#191). The adapter stamps
``goldfive.current_task_id`` onto session state at delegation time
so the handler can default from state when the model doesn't supply
the arg. Handlers still reject with the canonical
``missing_task_id`` shape when neither source resolves a value —
so strictness is enforced at the handler layer, not the schema.
"""

from __future__ import annotations

from typing import Any


def _object_schema(*, required: list[str], properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_SCHEMA_TASK_STARTED = _object_schema(
    required=[],
    properties={
        "task_id": {"type": "string"},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_PROGRESS = _object_schema(
    required=[],
    properties={
        "task_id": {"type": "string"},
        "fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_COMPLETED = _object_schema(
    required=["summary"],
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
    required=["reason"],
    properties={
        "task_id": {"type": "string"},
        "reason": {"type": "string"},
        "recoverable": {"type": "boolean"},
    },
)

_SCHEMA_TASK_BLOCKED = _object_schema(
    required=["blocker"],
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

_SCHEMA_AWAITING_APPROVAL = _object_schema(
    required=["prompt"],
    properties={
        "task_id": {"type": "string"},
        "prompt": {"type": "string"},
        "timeout_ms": {"type": "integer", "minimum": 0},
    },
)


_SCHEMA_DECLARE_TASK_SKIPPED = _object_schema(
    required=["reason"],
    properties={
        "task_id": {"type": "string"},
        "reason": {"type": "string"},
    },
)


_SCHEMA_DECLARE_TASK_NOT_NEEDED = _object_schema(
    required=["reason"],
    properties={
        "task_id": {"type": "string"},
        "reason": {"type": "string"},
    },
)


__all__ = [
    "_object_schema",
    "_SCHEMA_TASK_STARTED",
    "_SCHEMA_TASK_PROGRESS",
    "_SCHEMA_TASK_COMPLETED",
    "_SCHEMA_TASK_FAILED",
    "_SCHEMA_TASK_BLOCKED",
    "_SCHEMA_NEW_WORK_DISCOVERED",
    "_SCHEMA_PLAN_DIVERGENCE",
    "_SCHEMA_AWAITING_APPROVAL",
    "_SCHEMA_DECLARE_TASK_SKIPPED",
    "_SCHEMA_DECLARE_TASK_NOT_NEEDED",
]
