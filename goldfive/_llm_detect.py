"""Detect an LLM call surface on an existing agent, without hard deps.

When a caller hands goldfive an ADK agent via :func:`goldfive.wrap`,
the convenience layer tries to reuse whatever model that agent is
already configured with so the planner and goal-deriver can talk to
the same endpoint without the caller writing a bespoke ``call_llm``.

This module exposes two helpers:

* :func:`detect_llm` — inspect an agent and, if it looks like an ADK
  ``BaseAgent`` with a usable ``.model`` attribute, return a
  ``(call_llm, model_name)`` pair suitable for :class:`LLMPlanner` and
  :class:`LLMGoalDeriver`.
* :func:`make_default_adk_call_llm` — build a ``call_llm(system, user,
  model) -> str`` coroutine that dispatches through ADK's
  ``LLMRegistry`` / ``BaseLlm.generate_content_async`` stream.

Neither helper hard-imports ``google.adk`` at module-import time. Both
tolerate missing deps and return ``None`` when detection fails so the
caller can fall back to a passthrough planner.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

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


def make_default_adk_call_llm(model: Any) -> CallLLM | None:
    """Return a ``call_llm(system, user, model) -> str`` backed by ADK.

    ``model`` may be a string alias (``"gpt-4o"``), a ``BaseLlm``
    instance (including ``LiteLlm``), or anything ``LLMRegistry`` can
    resolve. Returns ``None`` when ADK is not installed or the model
    cannot be resolved to a ``BaseLlm``.
    """
    try:
        from google.adk.models.base_llm import BaseLlm  # type: ignore
        from google.adk.models.llm_request import LlmRequest  # type: ignore
        from google.adk.models.registry import LLMRegistry  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except ImportError:
        log.debug("goldfive._llm_detect: google.adk not installed")
        return None

    if isinstance(model, BaseLlm):
        llm: Any = model
    elif isinstance(model, str) and model:
        try:
            llm_cls = LLMRegistry.new_llm(model)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            log.debug("goldfive._llm_detect: LLMRegistry.new_llm(%r) raised: %s", model, exc)
            return None
        llm = llm_cls
    else:
        return None

    async def _call_llm(system: str, user: str, model_str: str) -> str:
        _ = model_str  # ADK's BaseLlm is already model-bound
        # Pull the per-callsite cap (set by goldfive consumers via
        # :func:`goldfive._llm.call_llm_budget`). ``None`` from the var
        # falls back to ``DEFAULT_MAX_OUTPUT_TOKENS`` (4096) so an
        # unsupervised dispatch still has a finite ceiling. See
        # goldfive#271 follow-up — pre-fix evidence in demo-v8.log
        # showed unbounded calls reaching 9961 completion tokens (9.6
        # minutes wall) on a Qwen Q4 endpoint.
        from goldfive._llm import get_max_output_tokens

        max_output_tokens = get_max_output_tokens()
        req = LlmRequest(
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user)],
                ),
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_output_tokens,
            ),
        )
        chunks: list[str] = []
        async for resp in llm.generate_content_async(req, stream=False):
            content = getattr(resp, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or ():
                if getattr(part, "thought", False):
                    continue
                text = getattr(part, "text", "") or ""
                if text:
                    chunks.append(str(text))
        return "".join(chunks).strip()

    async def _close() -> None:
        # ADK's BaseLlm doesn't pin a uniform close protocol, but several
        # subclasses (LiteLlm, custom HTTP-backed adapters) wrap an
        # aiohttp / httpx client. Probe a few known attribute names and
        # await whichever exists. Silently no-op if nothing is found.
        for attr_name in ("aclose", "close"):
            target = getattr(llm, attr_name, None)
            if callable(target):
                try:
                    result = target()
                    if hasattr(result, "__await__"):
                        await result
                    return
                except Exception as exc:  # noqa: BLE001
                    log.debug("ADK call_llm.%s raised %s", attr_name, exc)
                    return
        # Some LiteLlm versions stash the client as ._client / .client.
        for client_attr in ("_client", "client"):
            client = getattr(llm, client_attr, None)
            if client is None:
                continue
            for attr_name in ("aclose", "close"):
                target = getattr(client, attr_name, None)
                if callable(target):
                    try:
                        result = target()
                        if hasattr(result, "__await__"):
                            await result
                        return
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "ADK call_llm.%s.%s raised %s",
                            client_attr,
                            attr_name,
                            exc,
                        )
                        return

    _call_llm.close = _close  # type: ignore[attr-defined]
    return _call_llm


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
        model_name = getattr(model, "model", None) or (
            model if isinstance(model, str) else ""
        )
        return call_llm, str(model_name or "")
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "detect_llm: unexpected failure traversing %s (%s); returning None",
            type(agent).__name__,
            exc,
        )
        return None


__all__ = ["CallLLM", "detect_llm", "make_default_adk_call_llm"]
