"""Shared helper for invoking reporting tool handlers from adapters.

Adapters typically receive a tool call from their SDK as ``(name, args)``
and need to locate the matching :class:`ReportingToolSpec` and call its
handler with ``(args, session, steerer)``. This helper centralises that
lookup so that every adapter (CallableAdapter, ADK, Claude) routes through
the same code path — keeping behaviour and error messages consistent.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from goldfive.reporting import ReportingToolSpec

if TYPE_CHECKING:  # pragma: no cover - type-only
    from goldfive.protocols import Steerer
    from goldfive.types import Session


def find_tool(
    tools: Iterable[ReportingToolSpec],
    name: str,
) -> ReportingToolSpec | None:
    """Return the first tool whose ``name`` matches, else ``None``."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def invoke_tool(
    tools: Iterable[ReportingToolSpec],
    name: str,
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    """Look up ``name`` in ``tools`` and invoke its handler.

    Raises :class:`KeyError` if no tool with that name is registered — this
    mirrors the behaviour a real SDK would exhibit when an agent hallucinates
    a tool name, and lets adapters surface a clean error to the agent.
    """
    tool = find_tool(tools, name)
    if tool is None:
        raise KeyError(f"unknown reporting tool: {name!r}")
    return await tool.handler(args, session, steerer)
