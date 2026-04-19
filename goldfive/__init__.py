"""goldfive — agent orchestration: stay on target."""

from __future__ import annotations

__version__ = "0.1.0"

from goldfive.adapters.callable import CallableAdapter
from goldfive.control import (
    AckResult,
    ControlAck,
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.convenience import run, wrap
from goldfive.drift import classify_refusal, classify_stop_reason, classify_tool_error
from goldfive.executors.parallel import ParallelDAGExecutor
from goldfive.executors.sequential import SequentialExecutor
from goldfive.goal_deriver import (
    LiteralGoalDeriver,
    LLMGoalDeriver,
    PassthroughGoalDeriver,
)
from goldfive.planner import LLMPlanner, PassthroughPlanner, StaticPlanner
from goldfive.protocols import (
    AgentAdapter,
    EventSink,
    Executor,
    GoalDeriver,
    Planner,
    Steerer,
)
from goldfive.quickstart import quickstart
from goldfive.reporting import BUILTIN_REPORTING_TOOLS, ReportingToolSpec
from goldfive.results import ExecutionOutcome, InvocationResult
from goldfive.runner import Runner
from goldfive.sinks import (
    GRPCSink,
    InMemorySink,
    JSONLPersistenceSink,
    LoggingSink,
    SQLitePersistenceSink,
)
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
    "__version__",
    "AckResult",
    "AgentAdapter",
    "BUILTIN_REPORTING_TOOLS",
    "CallableAdapter",
    "ControlAck",
    "ControlChannel",
    "ControlKind",
    "ControlMessage",
    "DefaultSteerer",
    "DriftEvent",
    "DriftKind",
    "DriftSeverity",
    "EventSink",
    "ExecutionOutcome",
    "Executor",
    "GRPCSink",
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
    "ReportingToolSpec",
    "Runner",
    "SQLitePersistenceSink",
    "SequentialExecutor",
    "Session",
    "StaticPlanner",
    "Steerer",
    "Task",
    "TaskEdge",
    "TaskStatus",
    "classify_refusal",
    "classify_stop_reason",
    "classify_tool_error",
    "quickstart",
    "run",
    "wrap",
]
