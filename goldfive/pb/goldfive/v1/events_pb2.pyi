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
    __slots__ = ("event_id", "run_id", "sequence", "emitted_at", "session_id", "run_started", "goal_derived", "plan_submitted", "plan_revised", "task_started", "task_progress", "task_completed", "task_failed", "task_blocked", "task_cancelled", "drift_detected", "run_completed", "run_aborted", "conversation_started", "conversation_ended", "approval_requested", "approval_granted", "approval_rejected", "agent_invocation_started", "agent_invocation_completed", "delegation_observed", "reasoning_judge_invoked", "goldfive_llm_call_start", "goldfive_llm_call_end", "invocation_cancelled", "task_transitioned", "task_transition_refused", "invocation_boundary_entered", "invocation_boundary_exited")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    EMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
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
    AGENT_INVOCATION_STARTED_FIELD_NUMBER: _ClassVar[int]
    AGENT_INVOCATION_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    DELEGATION_OBSERVED_FIELD_NUMBER: _ClassVar[int]
    REASONING_JUDGE_INVOKED_FIELD_NUMBER: _ClassVar[int]
    GOLDFIVE_LLM_CALL_START_FIELD_NUMBER: _ClassVar[int]
    GOLDFIVE_LLM_CALL_END_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_CANCELLED_FIELD_NUMBER: _ClassVar[int]
    TASK_TRANSITIONED_FIELD_NUMBER: _ClassVar[int]
    TASK_TRANSITION_REFUSED_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_BOUNDARY_ENTERED_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_BOUNDARY_EXITED_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    run_id: str
    sequence: int
    emitted_at: _timestamp_pb2.Timestamp
    session_id: str
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
    agent_invocation_started: AgentInvocationStarted
    agent_invocation_completed: AgentInvocationCompleted
    delegation_observed: DelegationObserved
    reasoning_judge_invoked: ReasoningJudgeInvoked
    goldfive_llm_call_start: GoldfiveLLMCallStart
    goldfive_llm_call_end: GoldfiveLLMCallEnd
    invocation_cancelled: InvocationCancelled
    task_transitioned: TaskTransitioned
    task_transition_refused: TaskTransitionRefused
    invocation_boundary_entered: InvocationBoundaryEntered
    invocation_boundary_exited: InvocationBoundaryExited
    def __init__(self, event_id: _Optional[str] = ..., run_id: _Optional[str] = ..., sequence: _Optional[int] = ..., emitted_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., session_id: _Optional[str] = ..., run_started: _Optional[_Union[RunStarted, _Mapping]] = ..., goal_derived: _Optional[_Union[GoalDerived, _Mapping]] = ..., plan_submitted: _Optional[_Union[PlanSubmitted, _Mapping]] = ..., plan_revised: _Optional[_Union[PlanRevised, _Mapping]] = ..., task_started: _Optional[_Union[TaskStarted, _Mapping]] = ..., task_progress: _Optional[_Union[TaskProgress, _Mapping]] = ..., task_completed: _Optional[_Union[TaskCompleted, _Mapping]] = ..., task_failed: _Optional[_Union[TaskFailed, _Mapping]] = ..., task_blocked: _Optional[_Union[TaskBlocked, _Mapping]] = ..., task_cancelled: _Optional[_Union[TaskCancelled, _Mapping]] = ..., drift_detected: _Optional[_Union[DriftDetected, _Mapping]] = ..., run_completed: _Optional[_Union[RunCompleted, _Mapping]] = ..., run_aborted: _Optional[_Union[RunAborted, _Mapping]] = ..., conversation_started: _Optional[_Union[ConversationStarted, _Mapping]] = ..., conversation_ended: _Optional[_Union[ConversationEnded, _Mapping]] = ..., approval_requested: _Optional[_Union[ApprovalRequested, _Mapping]] = ..., approval_granted: _Optional[_Union[ApprovalGranted, _Mapping]] = ..., approval_rejected: _Optional[_Union[ApprovalRejected, _Mapping]] = ..., agent_invocation_started: _Optional[_Union[AgentInvocationStarted, _Mapping]] = ..., agent_invocation_completed: _Optional[_Union[AgentInvocationCompleted, _Mapping]] = ..., delegation_observed: _Optional[_Union[DelegationObserved, _Mapping]] = ..., reasoning_judge_invoked: _Optional[_Union[ReasoningJudgeInvoked, _Mapping]] = ..., goldfive_llm_call_start: _Optional[_Union[GoldfiveLLMCallStart, _Mapping]] = ..., goldfive_llm_call_end: _Optional[_Union[GoldfiveLLMCallEnd, _Mapping]] = ..., invocation_cancelled: _Optional[_Union[InvocationCancelled, _Mapping]] = ..., task_transitioned: _Optional[_Union[TaskTransitioned, _Mapping]] = ..., task_transition_refused: _Optional[_Union[TaskTransitionRefused, _Mapping]] = ..., invocation_boundary_entered: _Optional[_Union[InvocationBoundaryEntered, _Mapping]] = ..., invocation_boundary_exited: _Optional[_Union[InvocationBoundaryExited, _Mapping]] = ...) -> None: ...

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

class PlanRevisionDiff(_message.Message):
    __slots__ = ("added_task_ids", "removed_task_ids", "modified_task_ids", "added_edges", "removed_edges")
    ADDED_TASK_IDS_FIELD_NUMBER: _ClassVar[int]
    REMOVED_TASK_IDS_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_TASK_IDS_FIELD_NUMBER: _ClassVar[int]
    ADDED_EDGES_FIELD_NUMBER: _ClassVar[int]
    REMOVED_EDGES_FIELD_NUMBER: _ClassVar[int]
    added_task_ids: _containers.RepeatedScalarFieldContainer[str]
    removed_task_ids: _containers.RepeatedScalarFieldContainer[str]
    modified_task_ids: _containers.RepeatedScalarFieldContainer[str]
    added_edges: _containers.RepeatedCompositeFieldContainer[_types_pb2.TaskEdge]
    removed_edges: _containers.RepeatedCompositeFieldContainer[_types_pb2.TaskEdge]
    def __init__(self, added_task_ids: _Optional[_Iterable[str]] = ..., removed_task_ids: _Optional[_Iterable[str]] = ..., modified_task_ids: _Optional[_Iterable[str]] = ..., added_edges: _Optional[_Iterable[_Union[_types_pb2.TaskEdge, _Mapping]]] = ..., removed_edges: _Optional[_Iterable[_Union[_types_pb2.TaskEdge, _Mapping]]] = ...) -> None: ...

class PlanRevised(_message.Message):
    __slots__ = ("plan", "drift_kind", "severity", "reason", "revision_index", "diff", "trigger_event_id", "refine_input_summary", "refine_output_summary", "target_agent_id", "dry_run")
    PLAN_FIELD_NUMBER: _ClassVar[int]
    DRIFT_KIND_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    REVISION_INDEX_FIELD_NUMBER: _ClassVar[int]
    DIFF_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    REFINE_INPUT_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    REFINE_OUTPUT_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TARGET_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    plan: _types_pb2.Plan
    drift_kind: _types_pb2.DriftKind
    severity: _types_pb2.DriftSeverity
    reason: str
    revision_index: int
    diff: PlanRevisionDiff
    trigger_event_id: str
    refine_input_summary: str
    refine_output_summary: str
    target_agent_id: str
    dry_run: bool
    def __init__(self, plan: _Optional[_Union[_types_pb2.Plan, _Mapping]] = ..., drift_kind: _Optional[_Union[_types_pb2.DriftKind, str]] = ..., severity: _Optional[_Union[_types_pb2.DriftSeverity, str]] = ..., reason: _Optional[str] = ..., revision_index: _Optional[int] = ..., diff: _Optional[_Union[PlanRevisionDiff, _Mapping]] = ..., trigger_event_id: _Optional[str] = ..., refine_input_summary: _Optional[str] = ..., refine_output_summary: _Optional[str] = ..., target_agent_id: _Optional[str] = ..., dry_run: bool = ...) -> None: ...

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
    __slots__ = ("kind", "severity", "detail", "current_task_id", "current_agent_id", "annotation_id", "id", "trigger_input", "authored_by", "suppressed_by_user_steer", "condition_id", "lifecycle", "prev_severity", "observed_revision_index")
    KIND_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    ANNOTATION_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_INPUT_FIELD_NUMBER: _ClassVar[int]
    AUTHORED_BY_FIELD_NUMBER: _ClassVar[int]
    SUPPRESSED_BY_USER_STEER_FIELD_NUMBER: _ClassVar[int]
    CONDITION_ID_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_FIELD_NUMBER: _ClassVar[int]
    PREV_SEVERITY_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_REVISION_INDEX_FIELD_NUMBER: _ClassVar[int]
    kind: _types_pb2.DriftKind
    severity: _types_pb2.DriftSeverity
    detail: str
    current_task_id: str
    current_agent_id: str
    annotation_id: str
    id: str
    trigger_input: str
    authored_by: str
    suppressed_by_user_steer: bool
    condition_id: str
    lifecycle: _types_pb2.DriftLifecycle
    prev_severity: _types_pb2.DriftSeverity
    observed_revision_index: int
    def __init__(self, kind: _Optional[_Union[_types_pb2.DriftKind, str]] = ..., severity: _Optional[_Union[_types_pb2.DriftSeverity, str]] = ..., detail: _Optional[str] = ..., current_task_id: _Optional[str] = ..., current_agent_id: _Optional[str] = ..., annotation_id: _Optional[str] = ..., id: _Optional[str] = ..., trigger_input: _Optional[str] = ..., authored_by: _Optional[str] = ..., suppressed_by_user_steer: bool = ..., condition_id: _Optional[str] = ..., lifecycle: _Optional[_Union[_types_pb2.DriftLifecycle, str]] = ..., prev_severity: _Optional[_Union[_types_pb2.DriftSeverity, str]] = ..., observed_revision_index: _Optional[int] = ...) -> None: ...

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

class AgentInvocationStarted(_message.Message):
    __slots__ = ("agent_name", "task_id", "invocation_id", "parent_invocation_id", "started_at")
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    agent_name: str
    task_id: str
    invocation_id: str
    parent_invocation_id: str
    started_at: _timestamp_pb2.Timestamp
    def __init__(self, agent_name: _Optional[str] = ..., task_id: _Optional[str] = ..., invocation_id: _Optional[str] = ..., parent_invocation_id: _Optional[str] = ..., started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class AgentInvocationCompleted(_message.Message):
    __slots__ = ("agent_name", "task_id", "invocation_id", "summary", "completed_at")
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    agent_name: str
    task_id: str
    invocation_id: str
    summary: str
    completed_at: _timestamp_pb2.Timestamp
    def __init__(self, agent_name: _Optional[str] = ..., task_id: _Optional[str] = ..., invocation_id: _Optional[str] = ..., summary: _Optional[str] = ..., completed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class InvocationBoundaryEntered(_message.Message):
    __slots__ = ("invocation_id", "agent_name", "task_id", "entered_at")
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ENTERED_AT_FIELD_NUMBER: _ClassVar[int]
    invocation_id: str
    agent_name: str
    task_id: str
    entered_at: _timestamp_pb2.Timestamp
    def __init__(self, invocation_id: _Optional[str] = ..., agent_name: _Optional[str] = ..., task_id: _Optional[str] = ..., entered_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class InvocationBoundaryExited(_message.Message):
    __slots__ = ("invocation_id", "agent_name", "task_id", "reason", "exited_at")
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    EXITED_AT_FIELD_NUMBER: _ClassVar[int]
    invocation_id: str
    agent_name: str
    task_id: str
    reason: str
    exited_at: _timestamp_pb2.Timestamp
    def __init__(self, invocation_id: _Optional[str] = ..., agent_name: _Optional[str] = ..., task_id: _Optional[str] = ..., reason: _Optional[str] = ..., exited_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class DelegationObserved(_message.Message):
    __slots__ = ("from_agent", "to_agent", "task_id", "invocation_id", "observed_at", "tool_args_json")
    FROM_AGENT_FIELD_NUMBER: _ClassVar[int]
    TO_AGENT_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    TOOL_ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    from_agent: str
    to_agent: str
    task_id: str
    invocation_id: str
    observed_at: _timestamp_pb2.Timestamp
    tool_args_json: str
    def __init__(self, from_agent: _Optional[str] = ..., to_agent: _Optional[str] = ..., task_id: _Optional[str] = ..., invocation_id: _Optional[str] = ..., observed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., tool_args_json: _Optional[str] = ...) -> None: ...

class ReasoningJudgeInvoked(_message.Message):
    __slots__ = ("run_id", "task_id", "subject_agent_id", "model", "elapsed_ms", "reasoning_input", "raw_response", "on_task", "severity", "reason", "classification")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    REASONING_INPUT_FIELD_NUMBER: _ClassVar[int]
    RAW_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ON_TASK_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    CLASSIFICATION_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    task_id: str
    subject_agent_id: str
    model: str
    elapsed_ms: int
    reasoning_input: str
    raw_response: str
    on_task: bool
    severity: str
    reason: str
    classification: str
    def __init__(self, run_id: _Optional[str] = ..., task_id: _Optional[str] = ..., subject_agent_id: _Optional[str] = ..., model: _Optional[str] = ..., elapsed_ms: _Optional[int] = ..., reasoning_input: _Optional[str] = ..., raw_response: _Optional[str] = ..., on_task: bool = ..., severity: _Optional[str] = ..., reason: _Optional[str] = ..., classification: _Optional[str] = ...) -> None: ...

class GoldfiveLLMCallStart(_message.Message):
    __slots__ = ("span_id", "name", "model", "task_id", "start_time_ns", "input_preview", "target_agent_id", "target_task_id")
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    INPUT_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    TARGET_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    span_id: str
    name: str
    model: str
    task_id: str
    start_time_ns: int
    input_preview: str
    target_agent_id: str
    target_task_id: str
    def __init__(self, span_id: _Optional[str] = ..., name: _Optional[str] = ..., model: _Optional[str] = ..., task_id: _Optional[str] = ..., start_time_ns: _Optional[int] = ..., input_preview: _Optional[str] = ..., target_agent_id: _Optional[str] = ..., target_task_id: _Optional[str] = ...) -> None: ...

class GoldfiveLLMCallEnd(_message.Message):
    __slots__ = ("span_id", "name", "end_time_ns", "status", "error", "input_preview", "output_preview", "target_agent_id", "target_task_id", "decision_summary")
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    INPUT_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    TARGET_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    DECISION_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    span_id: str
    name: str
    end_time_ns: int
    status: str
    error: str
    input_preview: str
    output_preview: str
    target_agent_id: str
    target_task_id: str
    decision_summary: str
    def __init__(self, span_id: _Optional[str] = ..., name: _Optional[str] = ..., end_time_ns: _Optional[int] = ..., status: _Optional[str] = ..., error: _Optional[str] = ..., input_preview: _Optional[str] = ..., output_preview: _Optional[str] = ..., target_agent_id: _Optional[str] = ..., target_task_id: _Optional[str] = ..., decision_summary: _Optional[str] = ...) -> None: ...

class InvocationCancelled(_message.Message):
    __slots__ = ("invocation_id", "agent_name", "reason", "severity", "drift_id", "drift_kind", "detail", "tool_name")
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    DRIFT_ID_FIELD_NUMBER: _ClassVar[int]
    DRIFT_KIND_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    TOOL_NAME_FIELD_NUMBER: _ClassVar[int]
    invocation_id: str
    agent_name: str
    reason: str
    severity: str
    drift_id: str
    drift_kind: str
    detail: str
    tool_name: str
    def __init__(self, invocation_id: _Optional[str] = ..., agent_name: _Optional[str] = ..., reason: _Optional[str] = ..., severity: _Optional[str] = ..., drift_id: _Optional[str] = ..., drift_kind: _Optional[str] = ..., detail: _Optional[str] = ..., tool_name: _Optional[str] = ...) -> None: ...

class TaskTransitioned(_message.Message):
    __slots__ = ("task_id", "from_status", "to_status", "source", "revision_stamp", "agent_name", "invocation_id")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    FROM_STATUS_FIELD_NUMBER: _ClassVar[int]
    TO_STATUS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    REVISION_STAMP_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    from_status: str
    to_status: str
    source: str
    revision_stamp: int
    agent_name: str
    invocation_id: str
    def __init__(self, task_id: _Optional[str] = ..., from_status: _Optional[str] = ..., to_status: _Optional[str] = ..., source: _Optional[str] = ..., revision_stamp: _Optional[int] = ..., agent_name: _Optional[str] = ..., invocation_id: _Optional[str] = ...) -> None: ...

class TaskTransitionRefused(_message.Message):
    __slots__ = ("task_id", "attempted_from", "attempted_to", "reason", "pin_revision", "current_revision", "agent_name", "invocation_id")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTED_FROM_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTED_TO_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    PIN_REVISION_FIELD_NUMBER: _ClassVar[int]
    CURRENT_REVISION_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempted_from: str
    attempted_to: str
    reason: str
    pin_revision: int
    current_revision: int
    agent_name: str
    invocation_id: str
    def __init__(self, task_id: _Optional[str] = ..., attempted_from: _Optional[str] = ..., attempted_to: _Optional[str] = ..., reason: _Optional[str] = ..., pin_revision: _Optional[int] = ..., current_revision: _Optional[int] = ..., agent_name: _Optional[str] = ..., invocation_id: _Optional[str] = ...) -> None: ...
