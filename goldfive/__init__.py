"""goldfive — agent orchestration: stay on target."""

from __future__ import annotations

__version__ = "0.0.1"

# Core dataclasses / enums.
from goldfive.adapters import CallableAdapter
from goldfive.executors import ParallelDAGExecutor, SequentialExecutor

# Default component implementations.
from goldfive.goal_deriver import (
    LiteralGoalDeriver,
    LLMGoalDeriver,
    PassthroughGoalDeriver,
)
from goldfive.planner import LLMPlanner, PassthroughPlanner, StaticPlanner

# Protocols.
from goldfive.protocols import (
    AgentAdapter,
    EventSink,
    Executor,
    GoalDeriver,
    Planner,
    Steerer,
)

# Reporting tools.
from goldfive.reporting import (
    BUILTIN_REPORTING_TOOLS,
    REPORTING_TOOL_NAMES,
    ReportingHandler,
    ReportingToolSpec,
)

# Results.
from goldfive.results import ExecutionOutcome, InvocationResult

# The Runner — public entrypoint.
from goldfive.runner import Runner

# Sinks. ``InMemorySink`` is always available; ``LoggingSink`` and
# ``JSONLPersistenceSink`` depend on the optional ``proto`` extra and
# are re-exported lazily below when their imports succeed.
from goldfive.sinks.memory import InMemorySink

try:
    from goldfive.sinks.logging_sink import LoggingSink  # noqa: F401
except ImportError:  # pragma: no cover — proto extra not installed
    LoggingSink = None  # type: ignore[assignment]

try:
    from goldfive.sinks.persistence import (  # noqa: F401
        JSONLPersistenceSink,
        reconstruct_session,
        replay_from_jsonl,
    )
except ImportError:  # pragma: no cover — proto extra not installed
    JSONLPersistenceSink = None  # type: ignore[assignment]
    reconstruct_session = None  # type: ignore[assignment]
    replay_from_jsonl = None  # type: ignore[assignment]

from goldfive.steerer import DefaultSteerer
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

__all__ = [
    "BUILTIN_REPORTING_TOOLS",
    "REPORTING_TOOL_NAMES",
    "AgentAdapter",
    "CallableAdapter",
    "DefaultSteerer",
    "DriftEvent",
    "DriftKind",
    "DriftSeverity",
    "EventSink",
    "ExecutionOutcome",
    "Executor",
    "Goal",
    "GoalDeriver",
    "InMemorySink",
    "InvocationResult",
    "JSONLPersistenceSink",
    "LLMGoalDeriver",
    "LLMPlanner",
    "LiteralGoalDeriver",
    "LoggingSink",
    "ParallelDAGExecutor",
    "PassthroughGoalDeriver",
    "PassthroughPlanner",
    "Plan",
    "Planner",
    "ReportingHandler",
    "ReportingToolSpec",
    "Runner",
    "SequentialExecutor",
    "Session",
    "StaticPlanner",
    "Steerer",
    "Task",
    "TaskEdge",
    "TaskStatus",
    "__version__",
]
