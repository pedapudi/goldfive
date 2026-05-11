"""Auto-detect the right :class:`AgentAdapter` for an arbitrary agent object.

Powers :func:`goldfive.wrap` / :func:`goldfive.run` by turning any of
the four common "agent" shapes goldfive supports into a concrete
:class:`~goldfive.protocols.AgentAdapter`:

* an existing :class:`AgentAdapter` (passed through verbatim),
* an ADK ``BaseAgent`` (wrapped in :class:`ADKAdapter`),
* a Claude SDK client factory (wrapped in :class:`ClaudeAgentSDKAdapter`),
* an async callable matching :data:`~goldfive.adapters.callable.AgentCallable`
  (wrapped in :class:`CallableAdapter`).

The detector avoids importing the ADK and Claude SDKs at module load
so users who only install the default extras never pay the import
cost. When detection is ambiguous the dispatch favours ADK over
callable — an object that quacks like both is almost always an ADK
agent (its ``sub_agents`` / ``.run_async`` coroutine methods make it
look callable) and ADK takes precedence.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from goldfive.adapters.callable import CallableAdapter
from goldfive.protocols import AgentAdapter


def _looks_like_adk_agent(agent: Any) -> bool:
    """Duck-type check for an ADK ``BaseAgent`` without importing ADK."""
    if agent is None:
        return False
    cls = type(agent)
    for base in cls.__mro__:
        qualified = f"{base.__module__}.{base.__name__}"
        if qualified.startswith("google.adk."):
            return True
    if hasattr(agent, "sub_agents") and hasattr(agent, "name"):
        return True
    return False


def is_adk_agent(agent: Any) -> bool:
    """Return True when ``agent`` looks like a Google ADK ``BaseAgent``.

    Public wrapper over the internal duck-type check. Used by
    :func:`goldfive.wrap` to decide whether to return a plain
    :class:`~goldfive.runner.Runner` or a polymorphic
    :class:`~goldfive.adapters.adk_wrap.GoldfiveADKAgent` that also
    satisfies the ``BaseAgent`` contract. Runs without importing ADK,
    so callers that never install the extra don't pay the import cost.
    """
    return _looks_like_adk_agent(agent)


def _looks_like_adk_runner(agent: Any) -> bool:
    """Duck-type check for an ADK ``Runner``."""
    return (
        agent is not None
        and callable(getattr(agent, "run_async", None))
        and getattr(agent, "agent", None) is not None
        and getattr(agent, "session_service", None) is not None
    )


def _looks_like_claude_client_factory(agent: Any) -> bool:
    """Heuristic for a Claude SDK ``ClientFactory`` callable.

    A client factory is a zero-arg callable returning a ``ClaudeSDKClient``.
    Without importing the SDK, inspect the signature: no required
    positional-or-keyword parameters. We also check the declared return
    annotation's module when available.
    """
    if not callable(agent):
        return False
    if inspect.isclass(agent):
        qualified = f"{agent.__module__}.{agent.__name__}"
        if "claude_agent_sdk" in qualified:
            return True
    try:
        sig = inspect.signature(agent)
    except (TypeError, ValueError):
        return False
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if required:
        return False
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return False
    ret_repr = getattr(ret, "__module__", "") + "." + getattr(ret, "__name__", "")
    return "claude_agent_sdk" in ret_repr or "ClaudeSDKClient" in str(ret)


def _looks_like_async_agent_callable(agent: Any) -> bool:
    """Check for an async ``(task, session, tools) -> InvocationResult`` callable."""
    if not callable(agent):
        return False
    if asyncio.iscoroutinefunction(agent):
        return True
    # Instance with an async ``__call__`` method.
    call_attr = inspect.getattr_static(type(agent), "__call__", None)
    return asyncio.iscoroutinefunction(call_attr)


def auto_adapter(
    agent: Any,
    *,
    plugins: list[Any] | None = None,
    llm_call_timeout_ms: int | None = None,
    agent_max_output_tokens: int | None = None,
) -> AgentAdapter:
    """Return a concrete :class:`AgentAdapter` for ``agent``.

    Dispatch order:

    1. If ``agent`` already implements :class:`AgentAdapter`, return it.
    2. If it looks like an ADK ``BaseAgent`` or ``Runner``, build an
       :class:`~goldfive.adapters.adk.ADKAdapter`. Raises
       :class:`ImportError` when the ``adk`` extra is missing.
    3. If it looks like a Claude SDK client factory, build a
       :class:`~goldfive.adapters.claude.ClaudeAgentSDKAdapter`. Raises
       :class:`ImportError` when the ``claude`` extra is missing.
    4. If it is an async callable, wrap it in
       :class:`~goldfive.adapters.callable.CallableAdapter`.
    5. Otherwise, raise :class:`TypeError` with a list of the shapes
       goldfive recognises.

    ``plugins`` is forwarded to :class:`ADKAdapter` when the dispatch
    target is an ADK agent/runner. It is ignored for the other shapes —
    there is no analogous plugin surface in the Claude SDK or callable
    adapters today. See goldfive#121.

    ``llm_call_timeout_ms`` (goldfive#271 follow-up) and
    ``agent_max_output_tokens`` (goldfive#256) forward to
    :class:`ADKAdapter` so the typed :class:`~goldfive.config.AgentConfig`
    threaded through :func:`goldfive.wrap` reaches the plugin's
    per-LLM-call structural caps. Ignored for non-ADK adapters (no
    analogous surface).
    """
    if isinstance(agent, AgentAdapter):
        return agent

    if _looks_like_adk_agent(agent) or _looks_like_adk_runner(agent):
        from goldfive.adapters.adk import ADKAdapter  # lazy: requires extra

        adk_kwargs: dict[str, Any] = {"plugins": plugins}
        if llm_call_timeout_ms is not None:
            adk_kwargs["llm_call_timeout_ms"] = int(llm_call_timeout_ms)
        if agent_max_output_tokens is not None:
            adk_kwargs["agent_max_output_tokens"] = int(agent_max_output_tokens)
        return ADKAdapter(agent, **adk_kwargs)

    if _looks_like_claude_client_factory(agent):
        from goldfive.adapters.claude import ClaudeAgentSDKAdapter  # lazy

        return ClaudeAgentSDKAdapter(client_factory=agent)

    if _looks_like_async_agent_callable(agent):
        return CallableAdapter(agent, available_agents=["default"])

    raise TypeError(
        "goldfive.wrap: could not pick an AgentAdapter for "
        f"{type(agent).__name__!s}. Supported shapes are:\n"
        "  - any object implementing goldfive.AgentAdapter,\n"
        "  - a google.adk.agents.BaseAgent (requires goldfive[adk]),\n"
        "  - a zero-arg factory returning claude_agent_sdk.ClaudeSDKClient "
        "(requires goldfive[claude]),\n"
        "  - an async callable (task, session, tools) -> InvocationResult.\n"
        "Pass an adapter directly with goldfive.Runner(agent=...) if "
        "you need a shape goldfive does not auto-detect."
    )


__all__ = ["auto_adapter", "is_adk_agent"]
