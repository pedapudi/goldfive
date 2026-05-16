"""Deterministic stand-in for the ``call_llm`` callable goldfive consumes.

Designed for offline replay in zicato's tests. A live ``call_llm`` is
an ``async (system, user, model) -> str`` callable that issues HTTP
requests to an LLM endpoint. :class:`CannedCallLLM` adopts the same
shape but returns the next pre-recorded response on each call, with
the call-site signature recorded for assertions.

Usage::

    transcript = ["judge response 1", "judge response 2"]
    call_llm = CannedCallLLM(transcript)
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
    )
    ...  # drive the run
    assert call_llm.calls[0].system == REASONING_DRIFT_SYSTEM_PROMPT

Exhaustion raises :class:`CannedCallLLMExhausted` rather than silently
returning the empty string — silent exhaustion routinely masks test
bugs ("why is the judge always returning on_task?").
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Iterable
from typing import Final

__all__ = [
    "CannedCallLLM",
    "CannedCallLLMCall",
    "CannedCallLLMExhausted",
]


class CannedCallLLMExhausted(RuntimeError):
    """Raised when the recorded transcript is exhausted on call N+1.

    Carries the ``call_index`` (0-based) of the call that exhausted the
    transcript so a debugging optimizer can locate the missing entry.
    """

    def __init__(self, call_index: int, transcript_length: int) -> None:
        super().__init__(
            f"CannedCallLLM transcript exhausted on call #{call_index} "
            f"(transcript length {transcript_length}); add a response to "
            "the transcript or stop driving the run"
        )
        self.call_index = call_index
        self.transcript_length = transcript_length


@dataclasses.dataclass(frozen=True)
class CannedCallLLMCall:
    """One captured invocation of :class:`CannedCallLLM`."""

    system: str
    user: str
    model: str


class CannedCallLLM:
    """An async ``(system, user, model) -> str`` shaped object backed by a list.

    Each call pops the head of the transcript and records the
    (system, user, model) tuple on :attr:`calls` for later
    assertions. When the transcript is exhausted the next call raises
    :class:`CannedCallLLMExhausted` — silent fall-through is a bug
    multiplier so we refuse to do it.

    Thread-safe (the in-process tuning loop may drive judges from
    different background tasks); the lock is short-lived.

    The class is intentionally a callable instance rather than a bare
    coroutine factory so optimizers can inspect / reset state between
    runs.
    """

    def __init__(self, transcript: Iterable[str]) -> None:
        self._transcript: list[str] = list(transcript)
        self._next_index: int = 0
        self._lock: Final[threading.RLock] = threading.RLock()
        self.calls: list[CannedCallLLMCall] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        """Return the next recorded response; record the call site."""
        with self._lock:
            if self._next_index >= len(self._transcript):
                raise CannedCallLLMExhausted(
                    call_index=self._next_index,
                    transcript_length=len(self._transcript),
                )
            response = self._transcript[self._next_index]
            self._next_index += 1
            self.calls.append(
                CannedCallLLMCall(system=system, user=user, model=model)
            )
            return response

    @property
    def remaining(self) -> int:
        """Number of responses left in the transcript."""
        with self._lock:
            return len(self._transcript) - self._next_index

    @property
    def call_count(self) -> int:
        """Number of times the canned LLM has been called so far."""
        with self._lock:
            return self._next_index

    def reset(self) -> None:
        """Rewind to the start of the transcript; clear :attr:`calls`.

        Useful for running the same harness across multiple trials in
        one process without re-instantiating.
        """
        with self._lock:
            self._next_index = 0
            self.calls.clear()
