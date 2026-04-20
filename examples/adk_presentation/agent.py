"""ADK presentation reference example — goldfive wrapping a multi-subagent tree.

Ports harmonograf's ``presentation_agent`` demo to goldfive. The ADK tree
is unchanged: a ``coordinator_agent`` exposes four specialists as
``AgentTool`` s (researcher, web developer, reviewer, debugger). The
whole tree is wrapped with :class:`goldfive.adapters.adk.ADKAdapter`
and driven by a :class:`goldfive.Runner` with an ``LLMPlanner`` and
``LLMGoalDeriver``.

Two run modes:

* ``--mock`` — no network. Every ADK agent's model is replaced with an
  in-process ``_MockLlm`` that returns canned text, and the planner /
  goal-deriver use mock ``call_llm`` callables that emit deterministic
  JSON. Verifies goldfive wiring end-to-end without credentials.
* Default — uses ``OPENAI_API_KEY`` via the ``openai`` Python SDK for
  both planner / deriver and a ``LiteLLM`` model string (e.g.
  ``openai/gpt-4o-mini``) for the ADK subagents. Requires the
  ``openai`` and ``litellm`` packages.

Gated on the ``adk`` extra::

    uv pip install -e '.[adk]'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

try:
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools import AgentTool, FunctionTool
    from google.genai import types as genai_types
except ImportError as _adk_err:  # pragma: no cover
    raise SystemExit("install goldfive[adk] to run this example") from _adk_err

from goldfive import (
    InMemorySink,
    LLMGoalDeriver,
    LLMPlanner,
    Runner,
    SequentialExecutor,
)
from goldfive.adapters.adk import ADKAdapter
from goldfive.sinks import LoggingSink

log = logging.getLogger("examples.adk_presentation")


# ---------------------------------------------------------------------------
# ADK tools — mirror harmonograf's presentation_agent tool surface.
# ---------------------------------------------------------------------------


# All three file tools share one canonical directory so the writer, the
# reviewer, and the debugger always agree on which presentation they are
# operating on. The previous design threaded a ``topic`` string through
# each call and derived the directory name by slugifying it — but the
# writer, reviewer, and debugger each invented their own ``topic`` string
# from local context, and the slugs rarely matched. The reviewer would
# look in ``output/pandas/`` while the writer had saved to
# ``output/giant_pandas:_icons_of_conservation/``; ``read_presentation_files``
# returned error strings, the reviewer reported the task blocked, the
# planner refined, and the loop repeated. Root-caused during a live-run
# investigation; see ``docs/design/PLAN-LIFECYCLE.md`` §6.1 for the
# termination predicate this bug was exposing.
#
# Canonical path keeps the demo single-run-at-a-time (each run overwrites
# the prior). For multi-run isolation, a future enhancement can scope the
# path by ``session_id``.
_PRESENTATION_DIR = os.path.join(os.path.dirname(__file__), "output", "current")


def write_webpage(html_content: str, css_content: str, js_content: str) -> str:
    """Write the interactive webpage (HTML + CSS + JS) into the canonical
    presentation directory. Overwrites any prior files there.

    The reviewer and debugger read from the same canonical directory, so
    ``write_webpage`` does not accept a topic argument — the agent tree
    operates on exactly one presentation at a time per run.
    """
    os.makedirs(_PRESENTATION_DIR, exist_ok=True)
    try:
        with open(os.path.join(_PRESENTATION_DIR, "index.html"), "w") as f:
            f.write(html_content)
        with open(os.path.join(_PRESENTATION_DIR, "styles.css"), "w") as f:
            f.write(css_content)
        with open(os.path.join(_PRESENTATION_DIR, "script.js"), "w") as f:
            f.write(js_content)
        return f"Successfully wrote presentation to {_PRESENTATION_DIR}"
    except OSError as e:
        return f"Error writing file: {e}"


def read_presentation_files() -> dict[str, str]:
    """Read the current presentation files and return a name → content map.

    Reads from the canonical presentation directory that ``write_webpage``
    populated. No topic argument — writer and reader share one location so
    the reviewer cannot miss the file the writer just produced.

    If the writer hasn't run yet, each value carries a ``<not yet written>``
    marker so the reviewer can detect that state cleanly and report the
    task blocked (rather than silently "reviewing" an empty presentation).
    """
    files: dict[str, str] = {}
    for name in ("index.html", "styles.css", "script.js"):
        path = os.path.join(_PRESENTATION_DIR, name)
        try:
            with open(path) as f:
                files[name] = f.read()
        except FileNotFoundError:
            files[name] = "<not yet written>"
        except OSError as e:
            files[name] = f"<error reading {path}: {e}>"
    return files


def patch_file(filename: str, new_content: str) -> str:
    """Overwrite ``filename`` inside the canonical presentation directory.

    ``filename`` is the base name (e.g. ``index.html``); paths are always
    rooted at the canonical presentation directory so the debugger cannot
    write outside it or target a presentation the reviewer never saw.
    """
    safe_name = os.path.basename(filename)
    if not safe_name:
        return "Error: filename is empty"
    path = os.path.join(_PRESENTATION_DIR, safe_name)
    try:
        os.makedirs(_PRESENTATION_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully patched {path}"
    except OSError as e:
        return f"Error patching file: {e}"


write_webpage_tool = FunctionTool(write_webpage)
read_presentation_files_tool = FunctionTool(read_presentation_files)
patch_file_tool = FunctionTool(patch_file)


# ---------------------------------------------------------------------------
# Agent tree construction
# ---------------------------------------------------------------------------


def _build_agent_tree(model: str | BaseLlm) -> Any:
    """Build the coordinator + four specialist subagents and return the root."""
    research_agent = Agent(
        name="research_agent",
        model=model,
        instruction=(
            "You are a researcher. Gather key facts about the topic the user "
            "provides and synthesise them into concise bullet points suitable "
            "for a short presentation."
        ),
        description=("Agent that synthesises research bullet points for a topic."),
        tools=[],
    )

    web_developer_agent = Agent(
        name="web_developer_agent",
        model=model,
        instruction=(
            "You are an expert frontend developer. Given research notes, "
            "generate a single-page HTML slideshow plus CSS and JavaScript, "
            "then call the write_webpage tool to save the files. "
            "write_webpage takes ONLY (html_content, css_content, js_content) "
            "— there is no topic argument. All presentations are written "
            "to a shared canonical location that the reviewer and debugger "
            "read from."
        ),
        description=(
            "Frontend agent that produces and saves an interactive HTML/CSS/JS slideshow."
        ),
        tools=[write_webpage_tool],
    )

    reviewer_agent = Agent(
        name="reviewer_agent",
        model=model,
        instruction=(
            "You are a senior frontend reviewer. Call read_presentation_files() "
            "(no arguments — it reads from the canonical location the web "
            "developer wrote to) and return a structured list of issues "
            "(each with a severity of critical / major / minor) or an empty "
            "list if the output looks clean. If any file returns "
            "'<not yet written>', the web developer hasn't produced the "
            "presentation yet; report that as a blocker instead of reviewing "
            "an empty presentation."
        ),
        description=("Reviewer agent that critiques the generated presentation files."),
        tools=[read_presentation_files_tool],
    )

    debugger_agent = Agent(
        name="debugger_agent",
        model=model,
        instruction=(
            "You are a debugging agent. Given a list of issues flagged by "
            "the reviewer, call patch_file(filename, new_content) with "
            "the corrected contents for each affected file. "
            "filename is the base name only (index.html, styles.css, or "
            "script.js); paths are always rooted at the canonical "
            "presentation directory."
        ),
        description=("Debugger agent that patches presentation files flagged by the reviewer."),
        tools=[patch_file_tool],
    )

    coordinator_agent = Agent(
        name="coordinator_agent",
        model=model,
        instruction=(
            "You are the Coordinator. Drive a presentation workflow by "
            "delegating to research_agent, then web_developer_agent, then "
            "reviewer_agent, and finally debugger_agent if the reviewer "
            "flagged critical issues."
        ),
        description=(
            "Coordinator that delegates research / build / review / debug "
            "steps to specialist subagents."
        ),
        tools=[
            AgentTool(research_agent),
            AgentTool(web_developer_agent),
            AgentTool(reviewer_agent),
            AgentTool(debugger_agent),
        ],
    )
    return coordinator_agent


# ---------------------------------------------------------------------------
# Mock LLM — used by --mock mode so the demo runs without credentials.
# ---------------------------------------------------------------------------


class _MockLlm(BaseLlm):
    """Minimal BaseLlm that returns a single canned text response.

    Short-circuits ADK's model layer so the coordinator and every
    subagent produce a deterministic reply in ``--mock`` mode. The
    executor's auto-complete behaviour (goldfive #37) then marks each
    task COMPLETED based on the wrapped adapter returning cleanly.
    """

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"mock/.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        text = f"[mock:{self.model}] acknowledged task and deferred real work to a production run."
        yield LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=text)],
            ),
            partial=False,
            turn_complete=True,
        )


# ---------------------------------------------------------------------------
# Mock call_llm callables — used by LLMPlanner / LLMGoalDeriver under --mock.
# ---------------------------------------------------------------------------


def _make_mock_planner_llm(topic: str) -> Any:
    """Return an async ``call_llm`` that produces a canned presentation plan.

    Matches ``LLMPlanner``'s ``(system_prompt, user_prompt, model) -> str``
    signature and returns a plan with one task per specialist subagent so
    the executor emits ``TaskStarted`` / ``TaskCompleted`` for each.
    """

    plan_json = {
        "summary": f"Build a slideshow presentation on '{topic}'.",
        "tasks": [
            {
                "id": "research",
                "title": "Gather research bullet points on the topic",
                "description": "Summarise key facts about the topic.",
                "assignee_agent_id": "research_agent",
            },
            {
                "id": "build",
                "title": "Generate HTML/CSS/JS slideshow",
                "description": "Produce the presentation files and save them.",
                "assignee_agent_id": "web_developer_agent",
            },
            {
                "id": "review",
                "title": "Review the generated presentation",
                "description": "Critique the generated slideshow for issues.",
                "assignee_agent_id": "reviewer_agent",
            },
            {
                "id": "debug",
                "title": "Patch any critical issues the reviewer flagged",
                "description": "Apply fixes to the presentation files.",
                "assignee_agent_id": "debugger_agent",
            },
        ],
        "edges": [
            {"from_task_id": "research", "to_task_id": "build"},
            {"from_task_id": "build", "to_task_id": "review"},
            {"from_task_id": "review", "to_task_id": "debug"},
        ],
    }

    async def _call(system: str, prompt: str, model: str) -> str:
        return json.dumps(plan_json)

    return _call


def _make_mock_goal_llm(topic: str) -> Any:
    """Return an async ``call_llm`` that produces a single canned goal."""

    goals_json = {
        "goals": [
            {
                "id": "g1",
                "summary": f"Produce an interactive slideshow on '{topic}'.",
            }
        ]
    }

    async def _call(system: str, prompt: str, model: str) -> str:
        return json.dumps(goals_json)

    return _call


# ---------------------------------------------------------------------------
# Real call_llm — OpenAI via the official SDK.
# ---------------------------------------------------------------------------


def _make_openai_call_llm() -> Any:
    """Return an async ``call_llm`` backed by the OpenAI Python SDK.

    Used when ``--mock`` is not passed. Requires ``OPENAI_API_KEY`` and
    ``pip install openai``. Model identifiers default to
    ``gpt-4o-mini`` but can be overridden via
    ``GOLDFIVE_EXAMPLE_PLANNER_MODEL``.

    The returned callable carries an async ``close`` attribute that
    shuts down the underlying ``aiohttp`` session. Goldfive's
    :class:`Runner.close` discovers and awaits it via the duck-typed
    :class:`goldfive._llm.ClosableCallLLM` protocol, so callers don't
    need to register a separate close hook.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install openai to run this example without --mock") from exc

    client = AsyncOpenAI()

    async def _call(system: str, prompt: str, model: str) -> str:
        resolved = model or os.environ.get("GOLDFIVE_EXAMPLE_PLANNER_MODEL", "gpt-4o-mini")
        resp = await client.chat.completions.create(
            model=resolved,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    async def _close() -> None:
        await client.close()

    _call.close = _close  # type: ignore[attr-defined]
    return _call


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run(
    *,
    topic: str,
    mock: bool,
) -> None:
    """Drive one end-to-end presentation generation."""

    if mock:
        model: str | BaseLlm = _MockLlm(model="mock/presentation-agent")
        planner_call_llm = _make_mock_planner_llm(topic)
        goal_call_llm = _make_mock_goal_llm(topic)
        planner_model = "mock/planner"
        goal_model = "mock/goal-deriver"
    else:
        model = os.environ.get("GOLDFIVE_EXAMPLE_MODEL", "openai/gpt-4o-mini")
        openai_call_llm = _make_openai_call_llm()
        planner_call_llm = openai_call_llm
        goal_call_llm = openai_call_llm
        planner_model = os.environ.get("GOLDFIVE_EXAMPLE_PLANNER_MODEL", "gpt-4o-mini")
        goal_model = os.environ.get("GOLDFIVE_EXAMPLE_GOAL_MODEL", "gpt-4o-mini")

    root_agent = _build_agent_tree(model)
    adapter = ADKAdapter(root_agent)

    memory_sink = InMemorySink()
    logging_sink = LoggingSink()
    runner = Runner(
        agent=adapter,
        planner=LLMPlanner(call_llm=planner_call_llm, model=planner_model),
        executor=SequentialExecutor(max_plan_reinvocations=8),
        goal_deriver=LLMGoalDeriver(call_llm=goal_call_llm, model=goal_model),
        sinks=[memory_sink, logging_sink],
        max_plan_reinvocations=8,
    )

    outcome = await runner.run(f"Create a short interactive slideshow presentation on {topic}.")
    await runner.close()

    print(f"\nsuccess={outcome.success}  reason={outcome.reason!r}")
    print(f"events emitted: {len(memory_sink.events)}")
    for evt in memory_sink.events:
        if isinstance(evt, dict):
            kind = evt.get("kind", "?")
            run_id = evt.get("run_id", "")
            seq = evt.get("sequence", 0)
        else:
            kind = getattr(evt, "WhichOneof", lambda _: None)("payload") or "?"
            run_id = getattr(evt, "run_id", "")
            seq = getattr(evt, "sequence", 0)
        print(f"  seq={seq:>3}  run={run_id[:8]:<8}  {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="the history of the espresso machine",
        help="Topic for the slideshow presentation.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run without network calls — uses an in-process MockLlm for "
            "every ADK agent and canned JSON for the goldfive planner and "
            "goal deriver."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO-level logging from goldfive sinks.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )
    asyncio.run(run(topic=args.topic, mock=args.mock))


if __name__ == "__main__":
    main()
