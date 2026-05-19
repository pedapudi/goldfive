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

Quota / billing visibility
--------------------------

The callable bills against whichever auth the local ``claude`` CLI is
logged into — a Max plan in the typical setup. A single steered
goldfive run can issue multiple calls through this factory:

* one ``LLMPlanner`` call for the initial plan;
* one ``LLMPlanner.refine`` call per drift that produces a plan
  revision (rate-limited, but bursty runs can fire several);
* one ``LLMGoalDeriver`` call per ``run_started``;
* one ``judge_goal_drift`` call per task-boundary judge fire and per
  ``goal_drift_check_interval`` agent turn (rate-limited to ~10s
  between task-boundary fires);
* one ``judge_reasoning`` call per reasoning-judge fire.

Anecdotal: the ``presentation_agent`` example issues ~5–15 calls
through this factory on a clean run; failing runs (drift cascades) can
double that. Operators on Max should expect modest per-run quota burn
on Haiku; bumping to Sonnet/Opus or running tight loops in
production should plan for paid-tier billing instead.

A companion ADK ``BaseLlm`` adapter for using Claude on subagents is in
development — see the PR thread for the observability/architecture
trade-offs being worked through.

The integration is gated on the optional ``claude-agent-sdk``
dependency. Install via ``uv pip install claude-agent-sdk`` (or add to
your project's extras) before importing this module.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)


# Module-level so the test suite can assert on it without reaching
# into ``_call``'s closure.
_DEFAULT_MAX_TURNS = 5


def make_call_llm(
    default_model: str = "claude-haiku-4-5",
) -> Callable[[str, str, str], Awaitable[str]]:
    """Build an async ``(system, prompt, model) -> str`` callable.

    Goldfive plumbs the configured model through on every call. Falsy
    ``model`` (``""`` or ``None``) falls back to ``default_model`` —
    matches the rest of goldfive's call-site convention where an empty
    model string means "use whatever the integration picked." Callers
    that need to bypass the default (e.g. to assert a specific model
    string in tests) should pass a sentinel non-empty value instead.

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
        # Two SDK-isolation knobs (must agree with the BaseLlm adapter
        # in a follow-up PR):
        #
        # * ``setting_sources=[]`` — disables the SDK's default
        #   discovery chain (``~/.claude/settings.json``, project
        #   ``.claude/settings.json``, project ``CLAUDE.md``). Without
        #   it, operator-local Claude config leaks into every
        #   planner / judge prompt — e.g. a personal CLAUDE.md
        #   ("respond in YAML", "be terse") would silently corrupt the
        #   structured-JSON output goldfive parsers expect, surfacing
        #   downstream as "unparseable verdict" with no obvious cause.
        # * ``tools=[]`` — strips Claude Code's built-in tools
        #   (TodoWrite, Task, Read, Bash, …) from Claude's view, which
        #   keeps the internal agent loop dormant for these short
        #   structured prompts. ``allowed_tools=[]`` is NOT the right
        #   knob here: it only suppresses permission prompts; the tools
        #   stay visible to the model.
        opts = ClaudeAgentOptions(
            system_prompt=system or None,
            model=model or default_model,
            tools=[],
            setting_sources=[],
            max_turns=_DEFAULT_MAX_TURNS,
        )
        chunks: list[str] = []
        last_stop_reason: Any = None
        async for msg in query(prompt=prompt, options=opts):
            stop_reason = getattr(msg, "stop_reason", None)
            if stop_reason is not None:
                last_stop_reason = stop_reason
            for block in getattr(msg, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)

        if not chunks:
            # Silent zero-output is the classic "unparseable verdict"
            # diagnostic dead-end — downstream JSON parsers
            # (``LLMPlanner._parse_plan_response``,
            # ``judge_goal_drift._parse_verdict``) see ``""`` with no
            # breadcrumb pointing at the cause. Log loudly so operators
            # can correlate empty returns with model / turn settings.
            log.warning(
                "goldfive.integrations.claude_sdk.make_call_llm: "
                "claude-agent-sdk produced no text "
                "(model=%s, stop_reason=%s, max_turns=%d); "
                "downstream parsers will see an empty string",
                model or default_model,
                last_stop_reason,
                _DEFAULT_MAX_TURNS,
            )
        return "".join(chunks)

    return _call
