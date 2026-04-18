from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from goldfive.types import Session


@dataclasses.dataclass
class InvocationResult:
    """Result returned by ``AgentAdapter.invoke`` for a single task.

    ``text`` is the final assistant text, ``stop_reason`` is adapter-specific,
    ``error`` is populated if the invocation raised, and ``raw`` carries the
    adapter's native result object for debugging or downstream inspection.
    """

    task_id: str
    text: str = ""
    stop_reason: str = ""
    error: Optional[Exception] = None
    raw: Any = None


@dataclasses.dataclass
class ExecutionOutcome:
    """Final outcome of an ``Executor.run`` invocation.

    ``reason`` is populated when ``success`` is False to describe why the run
    terminated (e.g., unrecoverable task failure, user cancellation).
    """

    success: bool
    session: "Session"
    reason: str = ""
