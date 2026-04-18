from goldfive.v1 import events_pb2 as _events_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StreamEventsResponse(_message.Message):
    __slots__ = ("received", "error")
    RECEIVED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    received: int
    error: str
    def __init__(self, received: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...
