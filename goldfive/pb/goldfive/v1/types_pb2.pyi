import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_STATUS_UNSPECIFIED: _ClassVar[TaskStatus]
    TASK_STATUS_PENDING: _ClassVar[TaskStatus]
    TASK_STATUS_RUNNING: _ClassVar[TaskStatus]
    TASK_STATUS_COMPLETED: _ClassVar[TaskStatus]
    TASK_STATUS_FAILED: _ClassVar[TaskStatus]
    TASK_STATUS_CANCELLED: _ClassVar[TaskStatus]
    TASK_STATUS_BLOCKED: _ClassVar[TaskStatus]

class DriftSeverity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DRIFT_SEVERITY_UNSPECIFIED: _ClassVar[DriftSeverity]
    DRIFT_SEVERITY_INFO: _ClassVar[DriftSeverity]
    DRIFT_SEVERITY_WARNING: _ClassVar[DriftSeverity]
    DRIFT_SEVERITY_CRITICAL: _ClassVar[DriftSeverity]

class DriftKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DRIFT_KIND_UNSPECIFIED: _ClassVar[DriftKind]
    DRIFT_KIND_TOOL_ERROR: _ClassVar[DriftKind]
    DRIFT_KIND_AGENT_REFUSAL: _ClassVar[DriftKind]
    DRIFT_KIND_NEW_WORK_DISCOVERED: _ClassVar[DriftKind]
    DRIFT_KIND_PLAN_DIVERGENCE: _ClassVar[DriftKind]
    DRIFT_KIND_USER_STEER: _ClassVar[DriftKind]
    DRIFT_KIND_USER_CANCEL: _ClassVar[DriftKind]
    DRIFT_KIND_TASK_FAILED_RECOVERABLE: _ClassVar[DriftKind]
    DRIFT_KIND_TASK_FAILED_FATAL: _ClassVar[DriftKind]
    DRIFT_KIND_CONTEXT_PRESSURE: _ClassVar[DriftKind]
    DRIFT_KIND_BLOCKED: _ClassVar[DriftKind]
    DRIFT_KIND_WRONG_AGENT: _ClassVar[DriftKind]
    DRIFT_KIND_AGENT_TRANSFER: _ClassVar[DriftKind]
    DRIFT_KIND_MODEL_REFUSAL: _ClassVar[DriftKind]
    DRIFT_KIND_STOPPED_EARLY: _ClassVar[DriftKind]
    DRIFT_KIND_TOO_MANY_STEPS: _ClassVar[DriftKind]
    DRIFT_KIND_GOAL_UNREACHABLE: _ClassVar[DriftKind]
    DRIFT_KIND_TASK_TIMEOUT: _ClassVar[DriftKind]
    DRIFT_KIND_REPEATED_FAILURE: _ClassVar[DriftKind]
    DRIFT_KIND_UNEXPECTED_OUTPUT: _ClassVar[DriftKind]
    DRIFT_KIND_SCHEMA_VIOLATION: _ClassVar[DriftKind]
    DRIFT_KIND_HALLUCINATION_SUSPECTED: _ClassVar[DriftKind]
    DRIFT_KIND_SAFETY_CONCERN: _ClassVar[DriftKind]
    DRIFT_KIND_RESOURCE_EXHAUSTED: _ClassVar[DriftKind]
    DRIFT_KIND_AMBIGUOUS_INTENT: _ClassVar[DriftKind]
    DRIFT_KIND_CUSTOM: _ClassVar[DriftKind]
    DRIFT_KIND_LOOPING_TOOL_CALL: _ClassVar[DriftKind]
    DRIFT_KIND_LOOPING_REASONING: _ClassVar[DriftKind]
    DRIFT_KIND_CONFUSION: _ClassVar[DriftKind]
    DRIFT_KIND_OFF_TOPIC: _ClassVar[DriftKind]
    DRIFT_KIND_INTENT_DIVERGENCE: _ClassVar[DriftKind]
    DRIFT_KIND_UNCERTAIN_PROGRESS: _ClassVar[DriftKind]
    DRIFT_KIND_SELF_REPORTED_STUCK: _ClassVar[DriftKind]
    DRIFT_KIND_REASONING_CLUSTER_TIGHTENING: _ClassVar[DriftKind]
    DRIFT_KIND_CONFABULATION_RISK: _ClassVar[DriftKind]
TASK_STATUS_UNSPECIFIED: TaskStatus
TASK_STATUS_PENDING: TaskStatus
TASK_STATUS_RUNNING: TaskStatus
TASK_STATUS_COMPLETED: TaskStatus
TASK_STATUS_FAILED: TaskStatus
TASK_STATUS_CANCELLED: TaskStatus
TASK_STATUS_BLOCKED: TaskStatus
DRIFT_SEVERITY_UNSPECIFIED: DriftSeverity
DRIFT_SEVERITY_INFO: DriftSeverity
DRIFT_SEVERITY_WARNING: DriftSeverity
DRIFT_SEVERITY_CRITICAL: DriftSeverity
DRIFT_KIND_UNSPECIFIED: DriftKind
DRIFT_KIND_TOOL_ERROR: DriftKind
DRIFT_KIND_AGENT_REFUSAL: DriftKind
DRIFT_KIND_NEW_WORK_DISCOVERED: DriftKind
DRIFT_KIND_PLAN_DIVERGENCE: DriftKind
DRIFT_KIND_USER_STEER: DriftKind
DRIFT_KIND_USER_CANCEL: DriftKind
DRIFT_KIND_TASK_FAILED_RECOVERABLE: DriftKind
DRIFT_KIND_TASK_FAILED_FATAL: DriftKind
DRIFT_KIND_CONTEXT_PRESSURE: DriftKind
DRIFT_KIND_BLOCKED: DriftKind
DRIFT_KIND_WRONG_AGENT: DriftKind
DRIFT_KIND_AGENT_TRANSFER: DriftKind
DRIFT_KIND_MODEL_REFUSAL: DriftKind
DRIFT_KIND_STOPPED_EARLY: DriftKind
DRIFT_KIND_TOO_MANY_STEPS: DriftKind
DRIFT_KIND_GOAL_UNREACHABLE: DriftKind
DRIFT_KIND_TASK_TIMEOUT: DriftKind
DRIFT_KIND_REPEATED_FAILURE: DriftKind
DRIFT_KIND_UNEXPECTED_OUTPUT: DriftKind
DRIFT_KIND_SCHEMA_VIOLATION: DriftKind
DRIFT_KIND_HALLUCINATION_SUSPECTED: DriftKind
DRIFT_KIND_SAFETY_CONCERN: DriftKind
DRIFT_KIND_RESOURCE_EXHAUSTED: DriftKind
DRIFT_KIND_AMBIGUOUS_INTENT: DriftKind
DRIFT_KIND_CUSTOM: DriftKind
DRIFT_KIND_LOOPING_TOOL_CALL: DriftKind
DRIFT_KIND_LOOPING_REASONING: DriftKind
DRIFT_KIND_CONFUSION: DriftKind
DRIFT_KIND_OFF_TOPIC: DriftKind
DRIFT_KIND_INTENT_DIVERGENCE: DriftKind
DRIFT_KIND_UNCERTAIN_PROGRESS: DriftKind
DRIFT_KIND_SELF_REPORTED_STUCK: DriftKind
DRIFT_KIND_REASONING_CLUSTER_TIGHTENING: DriftKind
DRIFT_KIND_CONFABULATION_RISK: DriftKind

class Goal(_message.Message):
    __slots__ = ("id", "summary", "metadata", "has_success_predicate")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    HAS_SUCCESS_PREDICATE_FIELD_NUMBER: _ClassVar[int]
    id: str
    summary: str
    metadata: _containers.ScalarMap[str, str]
    has_success_predicate: bool
    def __init__(self, id: _Optional[str] = ..., summary: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., has_success_predicate: bool = ...) -> None: ...

class TaskEdge(_message.Message):
    __slots__ = ("from_task_id", "to_task_id")
    FROM_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TO_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    from_task_id: str
    to_task_id: str
    def __init__(self, from_task_id: _Optional[str] = ..., to_task_id: _Optional[str] = ...) -> None: ...

class Task(_message.Message):
    __slots__ = ("id", "title", "description", "assignee_agent_id", "status", "predicted_start_ms", "predicted_duration_ms", "bound_span_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ASSIGNEE_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_START_MS_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    BOUND_SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    title: str
    description: str
    assignee_agent_id: str
    status: TaskStatus
    predicted_start_ms: int
    predicted_duration_ms: int
    bound_span_id: str
    def __init__(self, id: _Optional[str] = ..., title: _Optional[str] = ..., description: _Optional[str] = ..., assignee_agent_id: _Optional[str] = ..., status: _Optional[_Union[TaskStatus, str]] = ..., predicted_start_ms: _Optional[int] = ..., predicted_duration_ms: _Optional[int] = ..., bound_span_id: _Optional[str] = ...) -> None: ...

class Plan(_message.Message):
    __slots__ = ("id", "run_id", "goal_ids", "summary", "tasks", "edges", "revision_reason", "revision_kind", "revision_severity", "revision_index", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    GOAL_IDS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    REVISION_REASON_FIELD_NUMBER: _ClassVar[int]
    REVISION_KIND_FIELD_NUMBER: _ClassVar[int]
    REVISION_SEVERITY_FIELD_NUMBER: _ClassVar[int]
    REVISION_INDEX_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    run_id: str
    goal_ids: _containers.RepeatedScalarFieldContainer[str]
    summary: str
    tasks: _containers.RepeatedCompositeFieldContainer[Task]
    edges: _containers.RepeatedCompositeFieldContainer[TaskEdge]
    revision_reason: str
    revision_kind: DriftKind
    revision_severity: DriftSeverity
    revision_index: int
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., run_id: _Optional[str] = ..., goal_ids: _Optional[_Iterable[str]] = ..., summary: _Optional[str] = ..., tasks: _Optional[_Iterable[_Union[Task, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[TaskEdge, _Mapping]]] = ..., revision_reason: _Optional[str] = ..., revision_kind: _Optional[_Union[DriftKind, str]] = ..., revision_severity: _Optional[_Union[DriftSeverity, str]] = ..., revision_index: _Optional[int] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
