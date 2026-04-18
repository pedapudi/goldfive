"""Drift classifiers (minimal stub until #6 lands).

Only the helpers consumed by the Claude adapter are implemented here;
issue #6 will replace this file with the full taxonomy.
"""

from __future__ import annotations

from goldfive.types import DriftEvent, DriftKind, DriftSeverity

# Claude Agent SDK stop-reason values that map to a goldfive drift kind.
# See https://docs.claude.com/en/api/messages streaming docs — the SDK
# surfaces stop_reason on ResultMessage / AssistantMessage verbatim.
_CLAUDE_STOP_REASON_MAP: dict[str, DriftKind] = {
    "max_turns": DriftKind.TOO_MANY_STEPS,
    "max_tokens": DriftKind.TOO_MANY_STEPS,
    "stop_sequence": DriftKind.STOPPED_EARLY,
    "end_turn": DriftKind.STOPPED_EARLY,
    "refusal": DriftKind.MODEL_REFUSAL,
    "pause_turn": DriftKind.STOPPED_EARLY,
}


def classify_stop_reason(
    stop_reason: str | None,
    *,
    current_task_id: str = "",
    current_agent_id: str = "",
    detail: str = "",
    raw: object = None,
) -> DriftEvent | None:
    """Map an adapter-reported ``stop_reason`` to a :class:`DriftEvent`.

    ``tool_use`` — the expected completion cause while the agent is
    mid-flight — is explicitly a non-drift signal and returns ``None``.
    Unknown reasons also return ``None`` so callers do not raise drift
    for benign stops.
    """

    if not stop_reason:
        return None
    if stop_reason == "tool_use":
        return None
    kind = _CLAUDE_STOP_REASON_MAP.get(stop_reason)
    if kind is None:
        return None
    severity = (
        DriftSeverity.CRITICAL
        if kind is DriftKind.MODEL_REFUSAL
        else DriftSeverity.WARNING
    )
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=detail or f"stop_reason={stop_reason}",
        current_task_id=current_task_id,
        current_agent_id=current_agent_id,
        raw=raw,
    )
