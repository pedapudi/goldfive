"""Context manager + helpers for emitting goldfive-internal LLM call spans.

Wraps every goldfive-internal ``await call_llm(...)`` site (planner
refine / generate, goal-drift classifier, reasoning-drift judge,
goal-derive, reflective self-progress check) with a
``SpanStart``/``SpanEnd`` pair so harmonograf renders the work as a
proper span on the goldfive lane with accurate wall-clock duration.

Without this, goldfive's internal LLM work is invisible on the Gantt —
a live session can show a 3-4 minute mystery gap between two agent
spans while goldfive runs a refine + goal-drift judge back-to-back.

Wire shape (Option B — no new transport surface): two new Event oneof
cases ``goldfive_llm_call_start`` / ``goldfive_llm_call_end`` carrying
``GoldfiveLLMCallStart`` / ``GoldfiveLLMCallEnd`` payloads. They flow
through the same ``EventSink.emit(event_pb)`` path as every other
goldfive event. Harmonograf's frontend synthesizes a span row from
each matched pair (keyed by ``span_id``), identical to how it already
synthesizes spans from ``ReasoningJudgeInvoked`` events today.

Design notes
------------

* Failure to import the proto stubs (``proto`` extra not installed) is
  absorbed at runtime: the context manager yields normally and skips
  emission. An internal observability helper must never break the
  wrapped call.
* Sink failures are absorbed too — one broken sink cannot prevent the
  wrapped LLM call from running.
* On an exception from the wrapped body, a ``SpanEnd`` with
  ``status="failed"`` + truncated error text is emitted, then the
  exception is re-raised.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

log = logging.getLogger(__name__)

# Error text on ``GoldfiveLLMCallEnd.error`` is capped so a pathological
# traceback can't blow the event wire budget. Matches the truncation
# convention used by ``ReasoningJudgeInvoked``.
_MAX_ERROR_CHARS = 1024
_TRUNC_SUFFIX = " … [truncated]"


def _truncate(s: str, limit: int = _MAX_ERROR_CHARS) -> str:
    if not s:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - len(_TRUNC_SUFFIX))] + _TRUNC_SUFFIX


def _events_pb_module() -> Any | None:
    """Import the events proto module, returning ``None`` on failure.

    The ``proto`` extra is optional; callers running without it still
    build functional runners (tests frequently do). The span helper
    degrades to a no-op in that case.
    """
    try:
        from goldfive.pb.goldfive.v1 import events_pb2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - broad on purpose; proto extra may be missing
        return None
    return events_pb2


async def _emit_to_sinks(sinks: list[Any], event_pb: Any) -> None:
    """Fan ``event_pb`` out to every sink, absorbing per-sink failures."""
    for sink in sinks:
        try:
            await sink.emit(event_pb)
        except Exception as exc:  # noqa: BLE001 - observability must never break the run
            log.warning(
                "goldfive_llm_span: sink.emit raised %s; event dropped for this sink",
                exc,
            )


def _new_envelope(
    *,
    run_id: str,
    session_id: str,
    sequence_fn: Any | None,
) -> Any | None:
    """Build a fresh Event envelope via :func:`goldfive.events.new_event`.

    Returns ``None`` when the proto stubs aren't available (callers
    degrade to no-op emission). ``sequence_fn`` may be ``None`` when no
    session is in scope; the envelope then uses sequence 0.
    """
    try:
        from goldfive.events import new_event
    except Exception as exc:  # noqa: BLE001 - proto extra may be missing
        log.debug("goldfive_llm_span: new_event import failed (%s); skipping", exc)
        return None
    try:
        sequence = int(sequence_fn()) if callable(sequence_fn) else 0
        return new_event(run_id or "", sequence, session_id=session_id or "")
    except Exception as exc:  # noqa: BLE001
        log.warning("goldfive_llm_span: new_event() raised %s; skipping", exc)
        return None


@asynccontextmanager
async def goldfive_llm_span(
    *,
    sinks: list[Any],
    name: str,
    model: str,
    session_id: str = "",
    run_id: str = "",
    task_id: str = "",
    sequence_fn: Any | None = None,
) -> AsyncIterator[None]:
    """Emit ``GoldfiveLLMCallStart`` on enter, ``GoldfiveLLMCallEnd`` on exit.

    Use to wrap every goldfive-internal ``await call_llm(...)`` so the
    wrapped call shows up as a span on harmonograf's Gantt.

    Parameters
    ----------
    sinks:
        The EventSinks to fan the Start/End pair onto. An empty list is
        a valid no-op (the wrapped body still runs). Per-sink failures
        are absorbed so a broken sink cannot prevent the wrapped LLM
        call from running.
    name:
        Short symbolic label for the call site (e.g. ``"refine_steer"``,
        ``"judge_goal_drift"``, ``"plan_generate"``, ``"goal_derive"``,
        ``"reflective_check"``, ``"judge_reasoning"``).
    model:
        The LLM model the wrapped call dispatches to. Stamped onto the
        Start event for sinks that want to colour-code by model.
    session_id / run_id:
        Optional envelope-level session/run correlation. Forwarded to
        :func:`goldfive.events.new_event` so the events join the same
        stream as the rest of the session.
    task_id:
        The currently-bound task id, when available. Sinks use this to
        attribute the span to the driving task on a Gantt.
    sequence_fn:
        Optional callable returning the next per-run event sequence
        number (typically ``session.next_sequence``). Called twice per
        wrapped call — once for Start, once for End. Defaults to 0 when
        absent (non-session contexts).

    Notes
    -----

    On exception from the wrapped body, the ``SpanEnd`` is emitted with
    ``status="failed"`` + truncated error text, then the exception is
    re-raised so callers see identical behaviour to an unwrapped
    ``await call_llm(...)``.
    """
    span_id = uuid.uuid4().hex
    start_ns = time.time_ns()
    pb = _events_pb_module()
    if pb is not None and sinks:
        envelope = _new_envelope(
            run_id=run_id, session_id=session_id, sequence_fn=sequence_fn
        )
        if envelope is not None:
            try:
                envelope.goldfive_llm_call_start.span_id = span_id
                envelope.goldfive_llm_call_start.name = name or ""
                envelope.goldfive_llm_call_start.model = model or ""
                envelope.goldfive_llm_call_start.task_id = task_id or ""
                envelope.goldfive_llm_call_start.start_time_ns = int(start_ns)
                await _emit_to_sinks(sinks, envelope)
            except Exception as exc:  # noqa: BLE001 - observability must never break
                log.warning(
                    "goldfive_llm_span: failed to build/emit SpanStart for %r: %s",
                    name,
                    exc,
                )

    status = "completed"
    error_text = ""
    try:
        yield
    except BaseException as exc:
        status = "failed"
        error_text = _truncate(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        end_ns = time.time_ns()
        if pb is not None and sinks:
            envelope = _new_envelope(
                run_id=run_id, session_id=session_id, sequence_fn=sequence_fn
            )
            if envelope is not None:
                try:
                    envelope.goldfive_llm_call_end.span_id = span_id
                    envelope.goldfive_llm_call_end.name = name or ""
                    envelope.goldfive_llm_call_end.end_time_ns = int(end_ns)
                    envelope.goldfive_llm_call_end.status = status
                    envelope.goldfive_llm_call_end.error = error_text
                    await _emit_to_sinks(sinks, envelope)
                except Exception as exc:  # noqa: BLE001 - observability must never break
                    log.warning(
                        "goldfive_llm_span: failed to build/emit SpanEnd for %r: %s",
                        name,
                        exc,
                    )


__all__ = ["goldfive_llm_span"]
