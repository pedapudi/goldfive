"""Detect an LLM call surface on an existing agent, without hard deps.

When a caller hands goldfive an ADK agent via :func:`goldfive.wrap`,
the convenience layer tries to reuse whatever model that agent is
already configured with so the planner and goal-deriver can talk to
the same endpoint without the caller writing a bespoke ``call_llm``.

This module exposes :func:`detect_llm` — inspect an agent and, if it
looks like an ADK ``BaseAgent`` with a usable ``.model`` attribute,
return a ``(call_llm, model_name)`` pair suitable for
:class:`LLMPlanner` and :class:`LLMGoalDeriver`. The builder it hands
the model to, :func:`goldfive._llm.make_default_adk_call_llm`, is
re-exported here for back-compat.

Neither helper hard-imports ``google.adk`` at module-import time. Both
tolerate missing deps and return ``None`` when detection fails so the
caller can fall back to a passthrough planner.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from goldfive._llm import make_default_adk_call_llm

log = logging.getLogger("goldfive.llm_detect")

CallLLM = Callable[[str, str, str], Awaitable[str]]


def _looks_like_adk_agent(agent: Any) -> bool:
    """Duck-type check for an ADK ``BaseAgent`` without importing ADK."""
    if agent is None:
        return False
    cls = type(agent)
    mro_names = {f"{c.__module__}.{c.__name__}" for c in cls.__mro__}
    for name in mro_names:
        if name.startswith("google.adk."):
            return True
    if hasattr(agent, "sub_agents") and hasattr(agent, "name"):
        return True
    return False


def _extract_adk_model(agent: Any) -> Any:
    """Return the first ``.model`` found on ``agent`` or a sub-agent.

    ADK coordinator trees often attach the model at the root; sub-agents
    inherit the same model by convention. We walk ``sub_agents``,
    ``inner_agent``, and ``tools[*].agent`` the same way the ADK
    adapter does in :func:`goldfive.adapters.adk.ADKAdapter.available_agents`.
    """
    seen: set[int] = set()
    stack: list[Any] = [agent]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        model = getattr(cur, "model", None)
        if model:
            return model
        for sub in getattr(cur, "sub_agents", None) or ():
            stack.append(sub)
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            stack.append(inner)
        for t in getattr(cur, "tools", None) or ():
            nested = getattr(t, "agent", None)
            if nested is not None:
                stack.append(nested)
    return None


def detect_llm(agent: Any) -> tuple[CallLLM, str] | None:
    """Detect an LLM surface on ``agent`` and return ``(call_llm, model)``.

    Currently only ADK agents are supported. Returns ``None`` for
    non-ADK shapes, for ADK agents without a resolvable model, when
    the ADK optional dep is not installed, **or for any unexpected
    exception during traversal**. The outer try/except is a
    belt-and-suspenders guard: `goldfive.wrap()` now calls this on
    every wrap regardless of whether ``planner`` / ``goal_deriver``
    are provided (the judges need the callable too), so a latent
    exception in a sub-agent's ``.model`` getter or a broken
    ``tools[*].agent`` reference would fault the whole wrap. Returning
    ``None`` on any failure keeps the degradation graceful; the
    caller logs a WARNING and falls back to unarmed judges.
    """
    try:
        if not _looks_like_adk_agent(agent):
            return None
        model = _extract_adk_model(agent)
        if model is None:
            return None
        call_llm = make_default_adk_call_llm(model)
        if call_llm is None:
            return None
        model_name = getattr(model, "model", None) or (model if isinstance(model, str) else "")
        return call_llm, str(model_name or "")
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "detect_llm: unexpected failure traversing %s (%s); returning None",
            type(agent).__name__,
            exc,
        )
        return None


__all__ = ["CallLLM", "detect_llm", "make_default_adk_call_llm"]
