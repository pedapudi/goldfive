"""Claude Agent SDK adapter for goldfive's ``CallLLM`` contract.

Provides :func:`make_call_llm` — an async
``(system, prompt, model) -> str`` callable that satisfies the
``CallLLM`` contract. Routes through ``claude_agent_sdk.query``, which
uses the local ``claude`` CLI's login (Max plan, API key, etc.) — no
separate ``ANTHROPIC_API_KEY`` needed when the user is already
authenticated via ``claude /login``.

Drop in at:

* ``LLMPlanner(call_llm=...)``
* ``LLMGoalDeriver(call_llm=...)``
* ``goldfive.wrap(call_llm=...)`` (judge fallback per the precedence
  chain documented on :func:`goldfive.wrap`)

Usage::

    from goldfive import wrap, LLMPlanner, LLMGoalDeriver
    from goldfive.integrations.claude_sdk import make_call_llm

    call_llm = make_call_llm("claude-haiku-4-5")
    runner = wrap(
        agent_tree,
        planner=LLMPlanner(call_llm=call_llm, model="claude-haiku-4-5"),
        goal_deriver=LLMGoalDeriver(call_llm=call_llm, model="claude-haiku-4-5"),
        call_llm=call_llm,  # judges inherit this fallback
    )

Short, structured prompts (plan generation, goal extraction, drift
verdicts) run cleanly: claude-agent-sdk's internal agent loop does not
engage for one-shot structured outputs.

A companion ADK ``BaseLlm`` adapter for using Claude on subagents is in
development — see the PR thread for the observability/architecture
trade-offs being worked through.

The integration is gated on the optional ``claude-agent-sdk``
dependency. Install via ``uv pip install claude-agent-sdk`` (or add to
your project's extras) before importing this module.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable


def make_call_llm(
    default_model: str = "claude-haiku-4-5",
) -> Callable[[str, str, str], Awaitable[str]]:
    """Build an async ``(system, prompt, model) -> str`` callable.

    Goldfive plumbs the configured model through on every call. Empty/
    falsy ``model`` falls back to ``default_model``.

    Raises :class:`ImportError` (with an install hint) on first call if
    ``claude_agent_sdk`` is not importable. We import lazily so simply
    importing this module does not require the SDK to be installed.
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "goldfive.integrations.claude_sdk requires `claude-agent-sdk`. "
            "Install it with: uv pip install claude-agent-sdk"
        ) from e

    async def _call(system: str, prompt: str, model: str) -> str:
        opts = ClaudeAgentOptions(
            system_prompt=system or None,
            model=model or default_model,
            allowed_tools=[],
            max_turns=5,
        )
        chunks: list[str] = []
        async for msg in query(prompt=prompt, options=opts):
            for block in getattr(msg, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)
        return "".join(chunks)

    return _call
