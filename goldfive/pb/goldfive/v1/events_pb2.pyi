import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from goldfive.v1 import types_pb2 as _types_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Event(_message.Message):
    __slots__ = ("event_id", "run_id", "sequence", "emitted_at", "run_started", "goal_derived", "plan_submitted", "plan_revised", "task_started", "task_progress", "task_completed", "task_failed", "task_blocked", "task_cancelled", "drift_detected", "run_completed", "run_aborted", "conversation_started", "conversation_ended", "approval_requested", "approval_granted", "approval_rejected")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    EMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    RUN_STARTED_FIELD_NUMBER: _ClassVar[int]
    GOAL_DERIVED_FIELD_NUMBER: _ClassVar[int]
    PLAN_SUBMITTED_FIELD_NUMBER: _ClassVar[int]
    PLAN_REVISED_FIELD_NUMBER: _ClassVar[int]
    TASK_STARTED_FIELD_NUMBER: _ClassVar[int]
    TASK_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    TASK_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    TASK_FAILED_FIELD_NUMBER: _ClassVar[int]
    TASK_BLOCKED_FIELD_NUMBER: _ClassVar[int]
    TASK_CANCELLED_FIELD_NUMBER: _ClassVar[int]
    DRIFT_DETECTED_FIELD_NUMBER: _ClassVar[int]
    RUN_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    RUN_ABORTED_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_STARTED_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ENDED_FIELD_NUMBER: _ClassVar[int]
    APPROVAL_REQUESTED_FIELD_NUMBER: _ClassVar[int]
    APPROVAL_GRANTED_FIELD_NUMBER: _ClassVar[int]
    APPROVAL_REJECTED_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    run_id: str
    sequence: int
    emitted_at: _timestamp_pb2.Timestamp
    run_started: RunStarted
    goal_derived: GoalDerived
    plan_submitted: PlanSubmitted
    plan_revised: PlanRevised
    task_started: TaskStarted
    task_progress: TaskProgress
    task_completed: TaskCompleted
    task_failed: TaskFailed
    task_blocked: TaskBlocked
    task_cancelled: TaskCancelled
    drift_detected: DriftDetected
    run_completed: RunCompleted
    run_aborted: RunAborted
    conversation_started: ConversationStarted
    conversation_ended: ConversationEnded
    approval_requested: ApprovalRequested
    approval_granted: ApprovalGranted
    approval_rejected: ApprovalRejected
    def __init__(self, event_id: _Optional[str] = ..., run_id: _Optional[str] = ..., sequence: _Optional[int] = ..., emitted_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., run_started: _Optional[_Union[RunStarted, _Mapping]] = ..., goal_derived: _Optional[_Union[GoalDerived, _Mapping]] = ..., plan_submitted: _Optional[_Union[PlanSubmitted, _Mapping]] = ..., plan_revised: _Optional[_Union[PlanRevised, _Mapping]] = ..., task_started: _Optional[_Union[TaskStarted, _Mapping]] = ..., task_progress: _Optional[_Union[TaskProgress, _Mapping]] = ..., task_completed: _Optional[_Union[TaskCompleted, _Mapping]] = ..., task_failed: _Optional[_Union[TaskFailed, _Mapping]] = ..., task_blocked: _Optional[_Union[TaskBlocked, _Mapping]] = ..., task_cancelled: _Optional[_Union[TaskCancelled, _Mapping]] = ..., drift_detected: _Optional[_Union[DriftDetected, _Mapping]] = ..., run_completed: _Optional[_Union[RunCompleted, _Mapping]] = ..., run_aborted: _Optional[_Union[RunAborted, _Mapping]] = ..., conversation_started: _Optional[_Union[ConversationStarted, _Mapping]] = ..., conversation_ended: _Optional[_Union[ConversationEnded, _Mapping]] = ..., approval_requested: _Optional[_Union[ApprovalRequested, _Mapping]] = ..., approval_granted: _Optional[_Union[ApprovalGranted, _Mapping]] = ..., approval_rejected: _Optional[_Union[ApprovalRejected, _Mapping]] = ...) -> None: ...

class RunStarted(_message.Message):
    __slots__ = ("run_id", "goal_summary", "started_at")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    GOAL_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    goal_summary: str
    started_at: _timestamp_pb2.Timestamp
    def __init__(self, run_id: _Optional[str] = ..., goal_summary: _Optional[str] = ..., started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GoalDerived(_message.Message):
    __slots__ = ("goals",)
    GOALS_FIELD_NUMBER: _ClassVar[int]
    goals: _containers.RepeatedCompositeFieldContainer[_types_pb2.Goal]
    def __init__(self, goals: _Optional[_Iterable[_Union[_types_pb2.Goal, _Mapping]]] = ...) -> None: ...

class PlanSubmitted(_message.Message):
    __slots__ = ("plan",)
    PLAN_FIELD_NUMBER: _ClassVar[int]
    plan: _types_pb2.Plan
    def __init__(self, plan: _Optional[_Union[_types_pb2.Plan, _Mapping]] = ...) -> None: ...

class PlanRevised(_message.Message):
    __slots__ = ("plan", "drift_kind", "severity", "reason", "revision_index")
    PLAN_FIELD_NUMBER: _ClassVar[int]
    DRIFT_KIND_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    REVISION_INDEX_FIELD_NUMBER: _ClassVar[int]
    plan: _types_pb2.Plan
    drift_kind: _types_pb2.DriftKind
    severity: _types_pb2.DriftSeverity
    reason: str
    revision_index: int
    def __init__(self, plan: _Optional[_Union[_types_pb2.Plan, _Mapping]] = ..., drift_kind: _Optional[_Union[_types_pb2.DriftKind, str]] = ..., severity: _Optional[_Union[_types_pb2.DriftSeverity, str]] = ..., reason: _Optional[str] = ..., revision_index: _Optional[int] = ...) -> None: ...

class TaskStarted(_message.Message):
    __slots__ = ("task_id", "detail")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    detail: str
    def __init__(self, task_id: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class TaskProgress(_message.Message):
    __slots__ = ("task_id", "fraction", "detail")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    FRACTION_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    fraction: float
    detail: str
    def __init__(self, task_id: _Optional[str] = ..., fraction: _Optional[float] = ..., detail: _Optional[str] = ...) -> None: ...

class TaskCompleted(_message.Message):
    __slots__ = ("task_id", "summary", "artifacts")
    class ArtifactsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    summary: str
    artifacts: _containers.ScalarMap[str, str]
    def __init__(self, task_id: _Optional[str] = ..., summary: _Optional[str] = ..., artifacts: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TaskFailed(_message.Message):
    __slots__ = ("task_id", "reason", "recoverable")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    RECOVERABLE_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    reason: str
    recoverable: bool
    def __init__(self, task_id: _Optional[str] = ..., reason: _Optional[str] = ..., recoverable: bool = ...) -> None: ...

class TaskBlocked(_message.Message):
    __slots__ = ("task_id", "blocker", "needed")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCKER_FIELD_NUMBER: _ClassVar[int]
    NEEDED_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    blocker: str
    needed: str
    def __init__(self, task_id: _Optional[str] = ..., blocker: _Optional[str] = ..., needed: _Optional[str] = ...) -> None: ...

class TaskCancelled(_message.Message):
    __slots__ = ("task_id", "reason")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    reason: str
    def __init__(self, task_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class DriftDetected(_message.Message):
    __slots__ = ("kind", "severity", "detail", "current_task_id", "current_agent_id")
    KIND_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    kind: _types_pb2.DriftKind
    severity: _types_pb2.DriftSeverity
    detail: str
    current_task_id: str
    current_agent_id: str
    def __init__(self, kind: _Optional[_Union[_types_pb2.DriftKind, str]] = ..., severity: _Optional[_Union[_types_pb2.DriftSeverity, str]] = ..., detail: _Optional[str] = ..., current_task_id: _Optional[str] = ..., current_agent_id: _Optional[str] = ...) -> None: ...

class RunCompleted(_message.Message):
    __slots__ = ("outcome_summary",)
    OUTCOME_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    outcome_summary: str
    def __init__(self, outcome_summary: _Optional[str] = ...) -> None: ...

class RunAborted(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class ConversationStarted(_message.Message):
    __slots__ = ("conversation_id", "started_at")
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    conversation_id: str
    started_at: _timestamp_pb2.Timestamp
    def __init__(self, conversation_id: _Optional[str] = ..., started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ConversationEnded(_message.Message):
    __slots__ = ("conversation_id", "turn_count", "reason")
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    TURN_COUNT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    conversation_id: str
    turn_count: int
    reason: str
    def __init__(self, conversation_id: _Optional[str] = ..., turn_count: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class ApprovalRequested(_message.Message):
    __slots__ = ("target_id", "kind", "prompt", "task_id", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    target_id: str
    kind: str
    prompt: str
    task_id: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, target_id: _Optional[str] = ..., kind: _Optional[str] = ..., prompt: _Optional[str] = ..., task_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ApprovalGranted(_message.Message):
    __slots__ = ("target_id", "detail")
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    target_id: str
    detail: str
    def __init__(self, target_id: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class ApprovalRejected(_message.Message):
    __slots__ = ("target_id", "detail")
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    target_id: str
    detail: str
    def __init__(self, target_id: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...
