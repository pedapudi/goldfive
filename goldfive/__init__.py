"""goldfive — agent orchestration: stay on target."""

from __future__ import annotations

__version__ = "0.1.0"

from goldfive import builtin_judges
from goldfive.adapters.callable import CallableAdapter
from goldfive.config import (
    EmbeddingConfig,
    GoalDriftConfig,
    ReasoningDriftConfig,
    RuntimeConfig,
    ToolLoopConfig,
)
from goldfive.control import (
    AckResult,
    ControlAck,
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.convenience import run, wrap
from goldfive.conversation import Conversation, TurnRecord
from goldfive.drift import classify_refusal, classify_stop_reason, classify_tool_error
from goldfive.executors.parallel import ParallelDAGExecutor
from goldfive.executors.sequential import SequentialExecutor
from goldfive.goal_deriver import (
    LiteralGoalDeriver,
    LLMGoalDeriver,
    PassthroughGoalDeriver,
)
from goldfive.judges import Judge, JudgeContext, JudgeVerdict
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
from goldfive.reporting import (
    BUILTIN_REPORTING_TOOLS,
    DRIFT_SELF_REPORTING_TOOL_NAMES,
    DRIFT_SELF_REPORTING_TOOLS,
    LIFECYCLE_REPORTING_TOOLS,
    ReportingToolSpec,
)
from goldfive.results import ExecutionOutcome, InvocationResult
from goldfive.runner import Runner
from goldfive.sinks import (
    GRPCSink,
    InMemorySink,
    JSONLPersistenceSink,
    LoggingSink,
    SQLitePersistenceSink,
)
from goldfive.steerer import DefaultSteerer, steering_is_active
from goldfive.types import (
    GOAL_SOURCE_USER_STEER,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    DriftSummary,
    Goal,
    ObservedAction,
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
    "builtin_judges",
    "DRIFT_SELF_REPORTING_TOOLS",
    "DRIFT_SELF_REPORTING_TOOL_NAMES",
    "LIFECYCLE_REPORTING_TOOLS",
    "ControlAck",
    "ControlChannel",
    "ControlKind",
    "ControlMessage",
    "Conversation",
    "DefaultSteerer",
    "DriftEvent",
    "DriftKind",
    "DriftSeverity",
    "DriftSummary",
    "EmbeddingConfig",
    "EventSink",
    "ExecutionOutcome",
    "Executor",
    "GOAL_SOURCE_USER_STEER",
    "GRPCSink",
    "Goal",
    "GoalDeriver",
    "GoalDriftConfig",
    "InMemorySink",
    "InvocationResult",
    "JSONLPersistenceSink",
    "Judge",
    "JudgeContext",
    "JudgeVerdict",
    "LLMGoalDeriver",
    "LLMPlanner",
    "LiteralGoalDeriver",
    "LoggingSink",
    "ObservedAction",
    "ParallelDAGExecutor",
    "PassthroughGoalDeriver",
    "PassthroughPlanner",
    "Plan",
    "Planner",
    "ReasoningDriftConfig",
    "ReportingToolSpec",
    "Runner",
    "RuntimeConfig",
    "SQLitePersistenceSink",
    "SequentialExecutor",
    "Session",
    "StaticPlanner",
    "Steerer",
    "Task",
    "TaskEdge",
    "TaskStatus",
    "ToolLoopConfig",
    "TurnRecord",
    "classify_refusal",
    "classify_stop_reason",
    "classify_tool_error",
    "quickstart",
    "run",
    "steering_is_active",
    "wrap",
]
