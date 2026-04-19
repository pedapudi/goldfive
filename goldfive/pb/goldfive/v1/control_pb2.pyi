import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ControlKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTROL_KIND_UNSPECIFIED: _ClassVar[ControlKind]
    CONTROL_KIND_PAUSE: _ClassVar[ControlKind]
    CONTROL_KIND_RESUME: _ClassVar[ControlKind]
    CONTROL_KIND_CANCEL: _ClassVar[ControlKind]
    CONTROL_KIND_REWIND_TO: _ClassVar[ControlKind]
    CONTROL_KIND_STEER: _ClassVar[ControlKind]
    CONTROL_KIND_APPROVE: _ClassVar[ControlKind]
    CONTROL_KIND_REJECT: _ClassVar[ControlKind]
    CONTROL_KIND_STATUS_QUERY: _ClassVar[ControlKind]
    CONTROL_KIND_INTERCEPT_TRANSFER: _ClassVar[ControlKind]
    CONTROL_KIND_INJECT_MESSAGE: _ClassVar[ControlKind]

class ControlAckResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTROL_ACK_RESULT_UNSPECIFIED: _ClassVar[ControlAckResult]
    CONTROL_ACK_RESULT_SUCCESS: _ClassVar[ControlAckResult]
    CONTROL_ACK_RESULT_FAILURE: _ClassVar[ControlAckResult]
    CONTROL_ACK_RESULT_UNSUPPORTED: _ClassVar[ControlAckResult]
CONTROL_KIND_UNSPECIFIED: ControlKind
CONTROL_KIND_PAUSE: ControlKind
CONTROL_KIND_RESUME: ControlKind
CONTROL_KIND_CANCEL: ControlKind
CONTROL_KIND_REWIND_TO: ControlKind
CONTROL_KIND_STEER: ControlKind
CONTROL_KIND_APPROVE: ControlKind
CONTROL_KIND_REJECT: ControlKind
CONTROL_KIND_STATUS_QUERY: ControlKind
CONTROL_KIND_INTERCEPT_TRANSFER: ControlKind
CONTROL_KIND_INJECT_MESSAGE: ControlKind
CONTROL_ACK_RESULT_UNSPECIFIED: ControlAckResult
CONTROL_ACK_RESULT_SUCCESS: ControlAckResult
CONTROL_ACK_RESULT_FAILURE: ControlAckResult
CONTROL_ACK_RESULT_UNSUPPORTED: ControlAckResult

class ControlTarget(_message.Message):
    __slots__ = ("agent_id", "task_id", "tool_call_id")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    task_id: str
    tool_call_id: str
    def __init__(self, agent_id: _Optional[str] = ..., task_id: _Optional[str] = ..., tool_call_id: _Optional[str] = ...) -> None: ...

class SteerPayload(_message.Message):
    __slots__ = ("note", "suggested_action")
    NOTE_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_ACTION_FIELD_NUMBER: _ClassVar[int]
    note: str
    suggested_action: str
    def __init__(self, note: _Optional[str] = ..., suggested_action: _Optional[str] = ...) -> None: ...

class RewindPayload(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    def __init__(self, task_id: _Optional[str] = ...) -> None: ...

class ApprovePayload(_message.Message):
    __slots__ = ("target_id", "detail")
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    target_id: str
    detail: str
    def __init__(self, target_id: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class RejectPayload(_message.Message):
    __slots__ = ("target_id", "detail")
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    target_id: str
    detail: str
    def __init__(self, target_id: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class InjectMessagePayload(_message.Message):
    __slots__ = ("role", "text")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    role: str
    text: str
    def __init__(self, role: _Optional[str] = ..., text: _Optional[str] = ...) -> None: ...

class ControlEvent(_message.Message):
    __slots__ = ("id", "issued_at", "target", "kind", "steer", "rewind", "approve", "reject", "inject_message")
    ID_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STEER_FIELD_NUMBER: _ClassVar[int]
    REWIND_FIELD_NUMBER: _ClassVar[int]
    APPROVE_FIELD_NUMBER: _ClassVar[int]
    REJECT_FIELD_NUMBER: _ClassVar[int]
    INJECT_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    id: str
    issued_at: _timestamp_pb2.Timestamp
    target: ControlTarget
    kind: ControlKind
    steer: SteerPayload
    rewind: RewindPayload
    approve: ApprovePayload
    reject: RejectPayload
    inject_message: InjectMessagePayload
    def __init__(self, id: _Optional[str] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., target: _Optional[_Union[ControlTarget, _Mapping]] = ..., kind: _Optional[_Union[ControlKind, str]] = ..., steer: _Optional[_Union[SteerPayload, _Mapping]] = ..., rewind: _Optional[_Union[RewindPayload, _Mapping]] = ..., approve: _Optional[_Union[ApprovePayload, _Mapping]] = ..., reject: _Optional[_Union[RejectPayload, _Mapping]] = ..., inject_message: _Optional[_Union[InjectMessagePayload, _Mapping]] = ...) -> None: ...

class ControlAck(_message.Message):
    __slots__ = ("control_id", "result", "detail", "acked_at")
    CONTROL_ID_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    ACKED_AT_FIELD_NUMBER: _ClassVar[int]
    control_id: str
    result: ControlAckResult
    detail: str
    acked_at: _timestamp_pb2.Timestamp
    def __init__(self, control_id: _Optional[str] = ..., result: _Optional[_Union[ControlAckResult, str]] = ..., detail: _Optional[str] = ..., acked_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
