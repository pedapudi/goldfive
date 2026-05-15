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

Decision-context fields (goldfive#decision-context)
---------------------------------------------------

In addition to the pairing fields, each span carries an
*input_preview* / *output_preview* / *target_agent_id* /
*target_task_id* / *decision_summary* payload so harmonograf can
render "what was goldfive doing here?" inline on the Gantt:

* ``input_preview`` (Start + End): a short human-readable summary of
  the data goldfive handed the LLM — drift detail + plan summary for
  a refine, activity block for the goal-drift judge, user_request for
  ``goal_derive``, reasoning text for ``judge_reasoning``.
* ``output_preview`` (End only): what goldfive produced from the
  call — new plan summary for refine, verdict + reason for judges,
  goals list for ``goal_derive``.
* ``target_agent_id`` / ``target_task_id`` (Start + End): the agent /
  task the call concerns. Empty for trajectory-level classifiers
  (``judge_goal_drift``) and initial-plan generation.
* ``decision_summary`` (End only): a one-line active-voice rendering
  of what goldfive DID ("refined plan in response to OFF_TOPIC drift
  on research_solar; produced research_solar_corrected assigned to
  research_agent"). Used by harmonograf as the span tooltip headline.

The Start-side fields are known when the call begins. The End-side
fields (``output_preview``, ``decision_summary``) are only known after
the wrapped body completes — callers populate them via the mutable
handle the context manager yields:

.. code-block:: python

    async with goldfive_llm_span(
        sinks=..., name="refine_steer", ...,
        input_preview="drift: USER_STEER/WARNING\\n...",
        target_agent_id="research_agent", target_task_id="research_solar",
    ) as span:
        result = await call_llm(...)
        span.output_preview = summarize_plan(result)
        span.decision_summary = "refined plan in response to ..."

The handle has two attributes — ``output_preview`` and
``decision_summary`` — both default to empty strings. On context-
manager exit, the helper reads those attributes off the handle and
stamps them into the ``GoldfiveLLMCallEnd`` event before emitting.
Callers that don't need the end-side context can ignore the yielded
handle entirely (``async with goldfive_llm_span(...):``) and both
fields default to the empty string.

Design notes
------------

* Failure to import the proto stubs (``proto`` extra not installed) is
  absorbed at runtime: the context manager yields normally and skips
  emission. An internal observability helper must never break the
  wrapped call.
* Sink failures are absorbed too — one broken sink cannot prevent the
  wrapped LLM call from running.
* On an exception from the wrapped body, a ``SpanEnd`` with
  ``status="failed"`` + truncated error text is emitted. The handle's
  ``output_preview`` / ``decision_summary`` are still read and stamped —
  callers that set them before the raise get retrospective context on
  the failure event; callers that don't see empty strings. The
  exception is re-raised.
* Special case (goldfive#266): when the wrapped body raises
  :class:`asyncio.CancelledError` AND the current task carries
  :data:`DRAIN_INITIATED_ATTR` (stamped by the steerer's drain path at
  run boundary), the ``SpanEnd`` is emitted with ``status="cancelled"``
  + a benign reason ("drained at run boundary") instead of
  ``status="failed"``. This distinguishes routine shutdown-initiated
  teardowns from genuine errors so harmonograf renders the span
  neutrally on the Gantt rather than red with a CancelledError stack.
  The cancel BEHAVIOUR is unchanged — only the observability axis.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# goldfive#266 — task-attribute marker that the steerer's drain path stamps
# onto in-flight background judge / drift tasks before issuing
# ``task.cancel()`` at run boundary. ``goldfive_llm_span`` reads this off
# ``asyncio.current_task()`` on its ``CancelledError`` path so the resulting
# ``GoldfiveLLMCallEnd`` reports ``status="cancelled"`` with a benign reason
# instead of ``status="failed"`` with a ``CancelledError`` traceback. The
# cancel BEHAVIOUR is unchanged — only the observability is corrected so
# operators stop seeing red ``judge_reasoning`` spans for routine
# shutdown-initiated teardowns. See ``_drain_background_set`` for the
# producer side. Public-ish (single underscore, exported through __all__)
# so the steerer can reference it without resorting to a magic string.
DRAIN_INITIATED_ATTR = "_goldfive_drain_initiated"

# Wire-level status string emitted when a drain-initiated cancel is
# detected. Distinct from ``"failed"`` (genuine error) and ``"completed"``
# (normal exit) so harmonograf can map it to ``SPAN_STATUS_CANCELLED``.
_STATUS_CANCELLED = "cancelled"
_STATUS_FAILED = "failed"
_STATUS_COMPLETED = "completed"
# Reason text stamped onto ``GoldfiveLLMCallEnd.error`` for drain-initiated
# cancels. Benign — operators reading the field see the lifecycle reason
# instead of a CancelledError traceback fragment.
_CANCEL_REASON_DRAIN = "drained at run boundary"

# Error text on ``GoldfiveLLMCallEnd.error`` is capped so a pathological
# traceback can't blow the event wire budget. Matches the truncation
# convention used by ``ReasoningJudgeInvoked``.
_MAX_ERROR_CHARS = 1024
# input_preview / output_preview cap on the wire. Matches the 4096-char
# ceiling already in use by ``ReasoningJudgeInvoked.reasoning_input`` so
# downstream sinks see one consistent truncation convention across
# goldfive observability payloads.
_MAX_PREVIEW_CHARS = 4096
# decision_summary is rendered inline as a Gantt tooltip headline; keep
# it short so the UI doesn't have to elide at render time.
_MAX_DECISION_SUMMARY_CHARS = 512
_TRUNC_SUFFIX = " … [truncated]"


def _truncate(s: str, limit: int = _MAX_ERROR_CHARS) -> str:
    if not isinstance(s, str) or not s:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - len(_TRUNC_SUFFIX))] + _TRUNC_SUFFIX


def _is_drain_initiated_cancel() -> bool:
    """Return True when the currently-running task was tagged by the
    steerer's drain path before its ``task.cancel()`` arrived.

    The steerer's :meth:`DefaultSteerer._drain_background_set` stamps
    :data:`DRAIN_INITIATED_ATTR` onto each background judge / drift task
    immediately before calling :meth:`task.cancel()`. Reading the
    attribute off :func:`asyncio.current_task` lets the span helper —
    which lives several frames inside the cancelled task — distinguish
    a lifecycle-initiated teardown from a genuine cancel without
    plumbing a flag through every classifier call signature.

    Returns False when no task is current (rare; only synchronous test
    harnesses) so the legacy ``failed`` branch wins by default — the
    cancel obviously didn't come from goldfive's drain in that case.
    """
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return False
    if task is None:
        return False
    return bool(getattr(task, DRAIN_INITIATED_ATTR, False))


@dataclass
class GoldfiveLLMSpanHandle:
    """Mutable handle yielded by :func:`goldfive_llm_span`.

    Callers set ``output_preview`` / ``decision_summary`` inside the
    ``async with`` body once the wrapped LLM call returns. The helper
    reads these off the handle on context-manager exit and stamps them
    into the ``GoldfiveLLMCallEnd`` event.

    Attributes default to empty strings so callers that don't need the
    end-side context can ignore the handle entirely.
    """

    output_preview: str = ""
    decision_summary: str = ""
    # Extra metadata slot for call sites that want to stash extra
    # debugging context. Not wired into the proto — lives on the handle
    # for convenience only.
    metadata: dict[str, Any] = field(default_factory=dict)


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
    input_preview: str = "",
    target_agent_id: str = "",
    target_task_id: str = "",
) -> AsyncIterator[GoldfiveLLMSpanHandle]:
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
        attribute the span to the driving task on a Gantt. Also
        mirrored onto ``target_task_id`` when the caller does not set
        ``target_task_id`` explicitly — that keeps pre-decision-context
        call sites emitting matching Start/End pairs without changes.
    sequence_fn:
        Optional callable returning the next per-run event sequence
        number (typically ``session.next_sequence``). Called twice per
        wrapped call — once for Start, once for End. Defaults to 0 when
        absent (non-session contexts).
    input_preview:
        Human-readable summary of what goldfive handed the LLM. Stamped
        verbatim onto both Start and End events (End echoes Start so
        sinks that lost Start can still render retrospective context).
        Truncated at ``_MAX_PREVIEW_CHARS`` (4096) with
        " … [truncated]".
    target_agent_id:
        Bare ADK name of the agent this call concerns. Empty for
        trajectory-level classifiers. Stamped on both Start and End.
    target_task_id:
        Task id this call concerns. Empty for non-task-scoped calls.
        When supplied it overrides ``task_id`` on the outgoing wire's
        ``target_task_id`` slot; when absent, ``task_id`` is mirrored
        into ``target_task_id`` for spec parity. Stamped on both Start
        and End.

    Yields
    ------
    GoldfiveLLMSpanHandle
        A mutable handle whose ``output_preview`` /
        ``decision_summary`` fields the caller sets inside the
        ``async with`` body. Those are stamped onto the
        ``GoldfiveLLMCallEnd`` event on context-manager exit. Callers
        that don't need the end-side context can ignore the yielded
        handle.

    Notes
    -----

    On exception from the wrapped body, the ``SpanEnd`` is emitted with
    ``status="failed"`` + truncated error text, then the exception is
    re-raised so callers see identical behaviour to an unwrapped
    ``await call_llm(...)``. The handle's ``output_preview`` /
    ``decision_summary`` are still read — a caller that partially
    populated them before the raise gets that context on the failure
    event; a caller that didn't populate them sees empty strings.

    goldfive#266 — drain-initiated CancelledError exception (the
    steerer's :meth:`drain_session_background_tasks` stamps
    :data:`DRAIN_INITIATED_ATTR` on the task before issuing
    ``task.cancel()``) is special-cased: ``status="cancelled"`` +
    a benign reason replaces the ``failed`` classification so live
    sessions don't render routine run-boundary teardowns of
    in-flight judge calls as red error spans. Cancel BEHAVIOUR is
    unchanged; this is an observability-only refinement.
    """
    from goldfive.runtime import seeded_uuid4

    span_id = seeded_uuid4().hex
    start_ns = time.time_ns()
    handle = GoldfiveLLMSpanHandle()

    # Pre-truncate the Start-side preview so the same clipped string is
    # echoed on both events. Mirror ``task_id`` into ``target_task_id``
    # when the caller didn't supply one — keeps every pre-decision-
    # context call site emitting a matching target_task_id without
    # change.
    effective_target_task = target_task_id or task_id or ""
    effective_input_preview = _truncate(input_preview, _MAX_PREVIEW_CHARS)
    effective_target_agent = target_agent_id or ""

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
                envelope.goldfive_llm_call_start.input_preview = (
                    effective_input_preview
                )
                envelope.goldfive_llm_call_start.target_agent_id = (
                    effective_target_agent
                )
                envelope.goldfive_llm_call_start.target_task_id = (
                    effective_target_task
                )
                await _emit_to_sinks(sinks, envelope)
            except Exception as exc:  # noqa: BLE001 - observability must never break
                log.warning(
                    "goldfive_llm_span: failed to build/emit SpanStart for %r: %s",
                    name,
                    exc,
                )

    status = _STATUS_COMPLETED
    error_text = ""
    try:
        yield handle
    except BaseException as exc:
        # goldfive#266 — when a CancelledError is raised AND the current
        # task carries the drain-initiated marker the steerer stamped on
        # before cancelling us, classify the span as ``cancelled`` rather
        # than ``failed``. Live sessions used to surface routine
        # run-boundary teardowns as red ``judge_reasoning`` spans with
        # CancelledError stack traces; this preserves the cancel
        # behaviour and only corrects the observability. Genuine
        # cancels from elsewhere (caller-driven asyncio.CancelledError,
        # USER_CANCEL flow) keep the ``failed`` classification so
        # operators can still spot real cancel-induced aborts.
        if isinstance(exc, asyncio.CancelledError) and _is_drain_initiated_cancel():
            status = _STATUS_CANCELLED
            error_text = _CANCEL_REASON_DRAIN
        else:
            status = _STATUS_FAILED
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
                    envelope.goldfive_llm_call_end.input_preview = (
                        effective_input_preview
                    )
                    envelope.goldfive_llm_call_end.output_preview = _truncate(
                        str(handle.output_preview or ""), _MAX_PREVIEW_CHARS
                    )
                    envelope.goldfive_llm_call_end.target_agent_id = (
                        effective_target_agent
                    )
                    envelope.goldfive_llm_call_end.target_task_id = (
                        effective_target_task
                    )
                    envelope.goldfive_llm_call_end.decision_summary = _truncate(
                        str(handle.decision_summary or ""),
                        _MAX_DECISION_SUMMARY_CHARS,
                    )
                    await _emit_to_sinks(sinks, envelope)
                except Exception as exc:  # noqa: BLE001 - observability must never break
                    log.warning(
                        "goldfive_llm_span: failed to build/emit SpanEnd for %r: %s",
                        name,
                        exc,
                    )


__all__ = ["DRAIN_INITIATED_ATTR", "GoldfiveLLMSpanHandle", "goldfive_llm_span"]
