"""Drift taxonomy and modular classifiers.

This module is the single *runtime* entry point for drift taxonomy lookups.
``DriftKind`` and ``DriftSeverity`` are re-exported from :mod:`goldfive.types`
so callers that only care about classification can import from one place.

The classifier helpers are deliberately simple, side-effect-free functions
that take a best-effort view of an upstream event (which can be anything —
an adapter-native object, a plain dict, a string) and return a
:class:`DriftEvent` when a signal is detected or ``None`` otherwise.

Heuristics here are ported from harmonograf's ``_AdkState.detect_drift``
and the tool-response classifier. ADK-specific bits (``function_call``
parts, ``session.state`` writes, gRPC submissions) are stripped; the
classifiers operate on framework-neutral shapes.
"""

from __future__ import annotations

from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity

__all__ = [
    "DriftKind",
    "DriftSeverity",
    "DriftEvent",
    "LLM_REFUSAL_MARKERS",
    "CONTEXT_PRESSURE_STOP_REASONS",
    "classify_tool_error",
    "classify_refusal",
    "classify_stop_reason",
]


# ---------------------------------------------------------------------------
# Marker tables (ported from harmonograf_client.adk)
# ---------------------------------------------------------------------------


# Substrings that indicate an LLM-level refusal in free-form text.
LLM_REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i won't",
    "i will not",
    "i'm unable",
    "i am unable",
    "i refuse",
    "i must decline",
    "i'm not able to",
    "i am not able to",
    "cannot assist",
    "can't help with",
)

# Stop-reason / finish-reason values that indicate the model hit a length
# cap, content filter, or other truncation event. Normalized to upper-case
# suffix (after the last ``.``).
CONTEXT_PRESSURE_STOP_REASONS: frozenset[str] = frozenset(
    {
        "MAX_TOKENS",
        "LENGTH",
        "MAX_OUTPUT_TOKENS",
        "TRUNCATED",
        "CONTENT_FILTER",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_marker(text: str, markers: tuple[str, ...]) -> str | None:
    """Return the first marker from ``markers`` present in lowercased ``text``."""
    if not text:
        return None
    lowered = text.lower()
    for m in markers:
        if m in lowered:
            return m
    return None


def _extract_text(event: Any) -> str:
    """Best-effort extraction of user-visible text from an opaque event.

    Recognises (in order):
      * plain ``str`` — returned as-is.
      * ``dict`` with a ``text`` or ``content`` key — value is stringified.
      * objects with a ``text`` attribute — the attribute is stringified.
    Otherwise returns ``""``.
    """
    if event is None:
        return ""
    if isinstance(event, str):
        return event
    if isinstance(event, dict):
        for key in ("text", "content", "message"):
            v = event.get(key)
            if isinstance(v, str) and v:
                return v
        return ""
    for attr in ("text", "content", "message"):
        v = getattr(event, attr, None)
        if isinstance(v, str) and v:
            return v
    return ""


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def classify_tool_error(event: Any) -> DriftEvent | None:
    """Return a ``DriftEvent`` of kind ``TOOL_ERROR`` if ``event`` looks
    like a tool-call result that failed.

    Recognises these shapes:
      * ``dict`` with a truthy ``error`` key
      * ``dict`` with ``status`` == ``"FAILED"`` / ``"ERROR"`` (case-insensitive)
      * ``dict`` with ``ok`` == ``False``
      * Objects exposing an ``error`` attribute that is truthy
    """
    if event is None:
        return None

    err_detail: str = ""
    tool_name: str = ""
    task_id: str = ""

    if isinstance(event, dict):
        # Accept harmonograf-ish and OpenAI-ish shapes.
        err = event.get("error")
        status = str(event.get("status", "") or "").upper()
        ok = event.get("ok")
        if err:
            err_detail = str(err)
        elif status in {"FAILED", "ERROR"}:
            err_detail = str(event.get("message", "") or status)
        elif ok is False:
            err_detail = str(event.get("message", "") or "tool returned ok=false")
        tool_name = str(event.get("tool", "") or event.get("name", "") or "")
        task_id = str(event.get("task_id", "") or "")
    else:
        err = getattr(event, "error", None)
        if err:
            err_detail = str(err)
        tool_name = str(getattr(event, "tool", "") or getattr(event, "name", "") or "")
        task_id = str(getattr(event, "task_id", "") or "")

    if not err_detail:
        return None

    detail = (
        f"tool {tool_name!r} errored: {err_detail}"
        if tool_name
        else f"tool error: {err_detail}"
    )
    return DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail=detail,
        current_task_id=task_id,
        raw=event,
    )


def classify_refusal(text: Any) -> DriftEvent | None:
    """Return a ``DriftEvent`` of kind ``AGENT_REFUSAL`` if the text
    contains any of :data:`LLM_REFUSAL_MARKERS`.

    ``text`` may be a raw string or any object from which
    :func:`_extract_text` can pull a text payload. Case-insensitive.
    """
    s = text if isinstance(text, str) else _extract_text(text)
    marker = _first_marker(s, LLM_REFUSAL_MARKERS)
    if marker is None:
        return None
    snippet = s[:140]
    return DriftEvent(
        kind=DriftKind.AGENT_REFUSAL,
        severity=DriftSeverity.WARNING,
        detail=f"refusal marker {marker!r}: {snippet!r}",
        raw=text,
    )


def classify_stop_reason(reason: Any) -> DriftEvent | None:
    """Return a ``DriftEvent`` when ``reason`` names a context-pressure
    stop reason (``MAX_TOKENS``, ``LENGTH``, ``TRUNCATED``, etc.).

    Accepts either a raw string, an enum-like object with a ``.name``
    attribute, or ``None``. The matched value is normalised to the
    final ``.``-delimited segment, upper-cased.
    """
    if reason is None:
        return None
    # Prefer enum-style .name if present; otherwise fall back to str().
    raw_name = getattr(reason, "name", None) or str(reason)
    normalised = raw_name.upper().rsplit(".", 1)[-1].strip()
    if not normalised:
        return None
    if normalised not in CONTEXT_PRESSURE_STOP_REASONS:
        return None
    return DriftEvent(
        kind=DriftKind.CONTEXT_PRESSURE,
        severity=DriftSeverity.WARNING,
        detail=f"response truncated (stop_reason={normalised})",
        raw=reason,
    )
