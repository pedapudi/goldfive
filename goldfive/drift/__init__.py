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
    "CONFABULATION_TRIGGER_KEYWORDS",
    "LLM_REFUSAL_MARKERS",
    "LLM_REFUSAL_MARKERS_INFO",
    "LLM_REFUSAL_MARKERS_WARNING",
    "LLM_REFUSAL_MARKERS_CRITICAL",
    "CONTEXT_PRESSURE_STOP_REASONS",
    "GOAL_DRIFT_CHECK_INTERVAL",
    "GOAL_DRIFT_IDLE_SECONDS",
    "classify_confabulation_risk",
    "classify_goal_drift",
    "classify_tool_error",
    "classify_refusal",
    "classify_stop_reason",
    "analyze_reasoning",
    "detect_confusion",
    "detect_intent_divergence",
    "detect_looping_reasoning",
    "detect_off_topic",
]


_REASONING_EXPORTS = frozenset(
    {
        "analyze_reasoning",
        "detect_confusion",
        "detect_intent_divergence",
        "detect_looping_reasoning",
        "detect_off_topic",
    }
)


_GOALS_EXPORTS = frozenset(
    {
        "classify_goal_drift",
        "GOAL_DRIFT_CHECK_INTERVAL",
        "GOAL_DRIFT_IDLE_SECONDS",
    }
)


def __getattr__(name: str) -> Any:
    """Lazy re-export of the reasoning-drift and goal-drift helpers.

    Defers the regex / optional-embedding imports in
    :mod:`goldfive.drift.reasoning` until first access, so
    ``from goldfive.drift import classify_tool_error`` stays cheap.
    The ``goals`` submodule is also lazy-loaded for parity.
    """
    if name in _REASONING_EXPORTS:
        from goldfive.drift import reasoning as _reasoning

        return getattr(_reasoning, name)
    if name in _GOALS_EXPORTS:
        from goldfive.drift import goals as _goals

        return getattr(_goals, name)
    raise AttributeError(f"module 'goldfive.drift' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Marker tables (ported from harmonograf_client.adk)
# ---------------------------------------------------------------------------


# Tiered refusal markers. ``classify_refusal`` scans the tiers in
# CRITICAL -> WARNING -> INFO order and returns a ``DriftEvent`` with
# the matching ``DriftSeverity``. First-match-wins: a substring match
# in a higher tier short-circuits scans of the lower tiers, so
# policy/safety refusals never get downgraded to hedging.
#
# INFO — hedging / deferral. The model is expressing low confidence
# but has not refused outright. Emitted as observational drift only;
# the default steerer does not trigger ``planner.refine`` for INFO.
LLM_REFUSAL_MARKERS_INFO: tuple[str, ...] = (
    "i may not be the best fit",
    "i think this might",
    "not particularly well suited",
    "i'm not confident",
    "i am not confident",
)

# WARNING — the model has stated it cannot or will not proceed but
# without invoking a safety policy. Historically the only severity for
# ``AGENT_REFUSAL``; still the most common tier. Triggers refine.
LLM_REFUSAL_MARKERS_WARNING: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i won't",
    "i will not",
    "i'm unable",
    "i am unable",
    "i refuse",
    "i'm not able to",
    "i am not able to",
    "can't help with",
    "beyond my capabilities",
    "outside my scope",
    "cannot proceed",
    "no viable approach",
    "unable to locate",
    "this is not something i can",
    "i was unable to",
)

# CRITICAL — policy / safety refusals. These usually mean the model
# will not produce the requested output no matter how the plan is
# refined; surface the highest severity so operators see the refusal
# clearly in sinks.
LLM_REFUSAL_MARKERS_CRITICAL: tuple[str, ...] = (
    "i must decline",
    "cannot assist with",
    "against my guidelines",
    "for safety reasons",
    "i will not proceed",
)

#: Deprecated: kept for back-compat with external callers that
#: imported the flat marker tuple. Prefer the tiered tables above
#: (:data:`LLM_REFUSAL_MARKERS_INFO`, :data:`LLM_REFUSAL_MARKERS_WARNING`,
#: :data:`LLM_REFUSAL_MARKERS_CRITICAL`) and scan via
#: :func:`classify_refusal`, which graduates severity per tier.
LLM_REFUSAL_MARKERS: tuple[str, ...] = (
    LLM_REFUSAL_MARKERS_CRITICAL + LLM_REFUSAL_MARKERS_WARNING + LLM_REFUSAL_MARKERS_INFO
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
        f"tool {tool_name!r} errored: {err_detail}" if tool_name else f"tool error: {err_detail}"
    )
    return DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail=detail,
        current_task_id=task_id,
        raw=event,
    )


def classify_refusal(text: Any) -> DriftEvent | None:
    """Return a ``DriftEvent`` of kind ``AGENT_REFUSAL`` with a severity
    graduated from the matching tier.

    Scans :data:`LLM_REFUSAL_MARKERS_CRITICAL`,
    :data:`LLM_REFUSAL_MARKERS_WARNING`, then
    :data:`LLM_REFUSAL_MARKERS_INFO` in that order and returns on the
    first match. This guarantees a policy/safety refusal is never
    downgraded to a WARNING or INFO just because the text also
    contains a hedging phrase. ``text`` may be a raw string or any
    object from which :func:`_extract_text` can pull a text payload.
    Case-insensitive.
    """
    s = text if isinstance(text, str) else _extract_text(text)
    if not s:
        return None
    for tier_markers, severity in (
        (LLM_REFUSAL_MARKERS_CRITICAL, DriftSeverity.CRITICAL),
        (LLM_REFUSAL_MARKERS_WARNING, DriftSeverity.WARNING),
        (LLM_REFUSAL_MARKERS_INFO, DriftSeverity.INFO),
    ):
        marker = _first_marker(s, tier_markers)
        if marker is None:
            continue
        snippet = s[:140]
        return DriftEvent(
            kind=DriftKind.AGENT_REFUSAL,
            severity=severity,
            detail=f"refusal marker {marker!r}: {snippet!r}",
            raw=text,
        )
    return None


# Keyword set for :func:`classify_confabulation_risk`. A task whose
# ``title`` or ``description`` contains any of these phrases (case-
# insensitive, substring match) is treated as "external-data-access
# shaped" and therefore suspicious when the invocation produces
# non-empty output with zero tool calls.
#
# The set is **conservative by design**: false positives here annoy
# operators because the drift surfaces in the UI on every clean run of
# a research-shaped task. Only include phrases that strongly imply the
# agent must go fetch / consult something external. Generic verbs
# ("write", "summarize", "format", "draft") are deliberately omitted —
# those often describe pure-synthesis work where zero tool calls is the
# expected shape.
#
# Add phrases here as empirical evidence accumulates; do not add whole-
# word verbs that routinely appear in synthesis prompts. The set is a
# module constant so tests can pin the contract explicitly.
CONFABULATION_TRIGGER_KEYWORDS: tuple[str, ...] = (
    "research",
    "gather",
    "look up",
    "lookup",
    "verify",
    "review",
    "fetch",
    "search",
    "analyze the file",
    "analyze the document",
    "check the",
    "find information about",
    "find information on",
    "read the file",
    "read the document",
    "investigate",
    "consult",
    "cross-reference",
    "cross reference",
)


def classify_confabulation_risk(
    *,
    task: Any,
    tool_call_count: int,
    output_text: str,
) -> DriftEvent | None:
    """Flag research / verification tasks that finished with zero tool calls.

    The cheap structural heuristic for confabulation risk (issue #128):
    if the task description reads like the agent was supposed to fetch,
    look up, or verify something external, but the invocation produced
    **non-empty output** without calling a single tool, that's the
    fishy pattern worth surfacing. The model may have fabricated the
    external observations wholesale.

    Returns a ``DriftEvent`` of kind
    :data:`~goldfive.types.DriftKind.CONFABULATION_RISK` at
    :data:`~goldfive.types.DriftSeverity.INFO` when:

    * The task's ``title`` or ``description`` contains any phrase in
      :data:`CONFABULATION_TRIGGER_KEYWORDS` (case-insensitive), AND
    * ``tool_call_count == 0``, AND
    * ``output_text`` is non-empty after stripping whitespace.

    Returns ``None`` otherwise. The severity is intentionally INFO —
    the steerer records the signal for the operator's UI but does not
    refine the plan on its own. A human can choose to cancel or let
    the run proceed.

    ``task`` may be a :class:`~goldfive.types.Task` or any duck-typed
    shape exposing ``title`` / ``description`` string attributes;
    missing attributes are treated as empty so adapter stubs that omit
    one field still work.
    """
    if task is None:
        return None
    if tool_call_count != 0:
        return None
    if not isinstance(output_text, str) or not output_text.strip():
        return None
    title = str(getattr(task, "title", "") or "")
    description = str(getattr(task, "description", "") or "")
    haystack = f"{title}\n{description}".lower()
    if not haystack.strip():
        return None
    matched: str | None = None
    for keyword in CONFABULATION_TRIGGER_KEYWORDS:
        if keyword in haystack:
            matched = keyword
            break
    if matched is None:
        return None
    task_id = str(getattr(task, "id", "") or "")
    assignee = str(getattr(task, "assignee_agent_id", "") or "")
    detail = (
        f"task {task_id!r} description implies external data access "
        f"(matched {matched!r}) but invocation produced output with "
        f"zero tool calls"
    )
    return DriftEvent(
        kind=DriftKind.CONFABULATION_RISK,
        severity=DriftSeverity.INFO,
        detail=detail,
        current_task_id=task_id,
        current_agent_id=assignee,
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
