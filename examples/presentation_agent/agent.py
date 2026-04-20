"""Full multi-agent ADK presentation tree, driven by goldfive.

This is the production-shaped sibling of ``examples/adk_presentation``.
Where ``adk_presentation`` shows the minimal *wrapping* pattern (one
``Agent``, goldfive plans + dispatches), this module shows the full
multi-agent tree a real app would ship:

    coordinator
      ├─ research_agent          — gathers facts for a topic
      ├─ web_developer_agent     — writes HTML/CSS/JS, saves via
      │                            ``write_webpage`` tool
      ├─ reviewer_agent          — reads files back via
      │                            ``read_presentation_files`` and
      │                            emits a structured critique
      └─ debugger_agent          — patches flagged issues via
                                   ``patch_file``

The tree is canonical: harmonograf's
``tests/reference_agents/presentation_agent/agent.py`` re-exports
exactly these agents so the harmonograf e2e suite and this example
share a single source of truth.

Run it three ways:

* ``uv run python examples/presentation_agent/agent.py --mock``
  — fully offline. Canned planner / goal deriver / ``_MockLlm``.
* ``OPENAI_API_KEY=sk-... uv run python examples/presentation_agent/agent.py --topic waffles``
  — live mode. Uses the ``openai`` SDK for planner + goal deriver and
  a LiteLLM string for the ADK subagents.
* ``adk web examples/presentation_agent``
  — module exposes ``app`` for the ADK web UI. Construction is lazy
  (PEP 562 ``__getattr__``) so importing the module offline does not
  blow up when ``OPENAI_API_KEY`` is absent.

Optional harmonograf telemetry: if ``HARMONOGRAF_SERVER`` is set and
``harmonograf_client`` is importable, the runner attaches a
``HarmonografSink`` and the ``App`` attaches a
``HarmonografTelemetryPlugin``. Neither is required — the example runs
without them.

Gated on the ``adk`` extra::

    uv pip install -e '.[adk]'
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
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

import goldfive
from goldfive import LLMGoalDeriver, LLMPlanner, SequentialExecutor
from goldfive.sinks import LoggingSink

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools — write / read / patch the generated presentation files.
# Output resolves under this module's directory to match harmonograf's
# reference tree so the two stay byte-compatible.
# ---------------------------------------------------------------------------


def write_webpage(
    topic: str, html_content: str, css_content: str, js_content: str
) -> str:
    """Write an interactive webpage (HTML, CSS, JS) under ``output/``."""
    try:
        topic_filename = topic.lower().replace(" ", "_").replace("/", "_")
        output_dir = os.path.join(os.path.dirname(__file__), "output", topic_filename)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "index.html"), "w") as f:
            f.write(html_content)
        with open(os.path.join(output_dir, "styles.css"), "w") as f:
            f.write(css_content)
        with open(os.path.join(output_dir, "script.js"), "w") as f:
            f.write(js_content)

        return f"Successfully created presentation on '{topic}' at {output_dir}"
    except OSError as e:
        return f"Error writing file: {e}"


def read_presentation_files(topic: str) -> dict[str, str]:
    """Read the generated presentation files and return name → contents."""
    topic_filename = topic.lower().replace(" ", "_").replace("/", "_")
    output_dir = os.path.join(os.path.dirname(__file__), "output", topic_filename)
    files: dict[str, str] = {}
    for name in ("index.html", "styles.css", "script.js"):
        path = os.path.join(output_dir, name)
        try:
            with open(path) as f:
                files[name] = f.read()
        except OSError as e:
            files[name] = f"<error reading {path}: {e}>"
    return files


def patch_file(path: str, new_content: str) -> str:
    """Overwrite ``path`` with ``new_content`` in place.

    Relative paths resolve against ``output/`` so the debugger cannot
    scribble outside the sandbox.
    """
    try:
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), "output", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully patched {path}"
    except OSError as e:
        return f"Error patching file: {e}"


write_webpage_tool = FunctionTool(write_webpage)
read_presentation_files_tool = FunctionTool(read_presentation_files)
patch_file_tool = FunctionTool(patch_file)


# ---------------------------------------------------------------------------
# Agent tree — coordinator + four specialists. Instruction text mirrors
# harmonograf's reference tree verbatim so the two modules describe the
# same behaviour.
# ---------------------------------------------------------------------------


def _build_agent_tree(model: Any) -> Agent:
    research_agent = Agent(
        name="research_agent",
        model=model,
        instruction=(
            "You are a researcher. Your goal is to gather information about "
            "the topic the user provides.\nThink step-by-step and provide a "
            "comprehensive synthesis of high-quality bullet points and facts "
            "that can be used to generate a presentation slideshow."
        ),
        description=(
            "An agent capable of deeply reasoning and synthesizing a given "
            "topic for presentation notes."
        ),
        tools=[],
    )

    web_developer_agent = Agent(
        name="web_developer_agent",
        model=model,
        instruction=(
            "You are an expert Frontend Web Developer. Your goal is to take "
            "research on a topic and generate a stunning, interactive, "
            "single-page presentation slideshow.\nGenerate beautiful semantic "
            "HTML structure, elegant CSS with modern design trends, "
            "animations, and transitions, and JavaScript for slideshow "
            "navigation (next/prev slides).\nThe HTML MUST include "
            '`<link rel="stylesheet" href="styles.css">` and '
            '`<script src="script.js"></script>` so the files are connected '
            "properly.\nRemember to output the absolute final HTML, CSS, and "
            "JS using the `write_webpage` tool! Do not just print the code "
            "out, you must invoke the tool once everything is ready."
        ),
        description=(
            "An expert frontend developer agent that generates interactive "
            "HTML, CSS, and JS slideshow presentations and saves them to disk."
        ),
        tools=[write_webpage_tool],
    )

    reviewer_agent = Agent(
        name="reviewer_agent",
        model=model,
        instruction=(
            "You are a senior frontend code reviewer. You will be given the "
            "topic of a presentation that ``web_developer_agent`` just "
            "generated. Call the ``read_presentation_files`` tool with the "
            "topic to fetch the generated HTML, CSS, and JS, then produce a "
            "structured critique as a list of issues. Each issue must "
            "include a short description and a severity of 'critical', "
            "'major', or 'minor'. If there are no issues, return an empty "
            "list and say so explicitly so the coordinator knows to skip "
            "debugging."
        ),
        description=(
            "A reviewer agent that reads the generated presentation files "
            "and produces a structured critique of issues and their severity."
        ),
        tools=[read_presentation_files_tool],
    )

    debugger_agent = Agent(
        name="debugger_agent",
        model=model,
        instruction=(
            "You are a debugging agent. You are invoked when "
            "``write_webpage`` failed or when ``reviewer_agent`` flagged "
            "critical issues in the generated presentation. Read the issues "
            "and their file paths, then call the ``patch_file`` tool with "
            "the full corrected content of each file that needs to change. "
            "Report which files you patched when you are done."
        ),
        description=(
            "A debugger agent that patches generated presentation files in "
            "place to resolve critical issues flagged by the reviewer or by "
            "a failing write_webpage call."
        ),
        tools=[patch_file_tool],
    )

    return Agent(
        name="coordinator_agent",
        model=model,
        instruction=(
            "You are the Coordinator Agent. Your task is to work with the "
            "user to pick a topic for an interactive slideshow "
            "presentation.\nFirst, get a topic from the user.\nSecond, "
            "transfer control to the 'research_agent' to gather "
            "comprehensive context and facts about the topic. Make sure to "
            "provide it with the topic!\nThird, after researching, transfer "
            "control to the 'web_developer_agent' and provide it with all "
            "the researched materials. Instruct it to generate and save the "
            "presentation codebase.\nFourth, transfer control to the "
            "'reviewer_agent' with the topic so it can read the generated "
            "files and produce a structured critique.\nFifth, if "
            "``write_webpage`` failed or the reviewer reported any critical "
            "issues, transfer control to the 'debugger_agent' with the "
            "reviewer's critique and have it patch the affected files. Skip "
            "this step when the reviewer reports no critical issues.\n"
            "Finally, report back to the user when the task is complete.\n"
            "Flow: research → web_developer → reviewer → (if critical "
            "issues) debugger → report."
        ),
        description=(
            "The main coordinator agent that drives the overall process of "
            "creating an interactive slideshow generation."
        ),
        tools=[
            AgentTool(research_agent),
            AgentTool(web_developer_agent),
            AgentTool(reviewer_agent),
            AgentTool(debugger_agent),
        ],
    )


_MODEL_NAME = os.environ.get("GOLDFIVE_EXAMPLE_MODEL", "openai/gpt-4o-mini")

# Build the real tree at import time so ``from ... import root_agent`` works
# without running the lazy ``app`` construction path.
root_agent = _build_agent_tree(_MODEL_NAME)


# ---------------------------------------------------------------------------
# Mock mode — deterministic, network-free demo path. Same shape as
# ``examples/adk_presentation`` so the two examples behave identically
# under ``--mock``.
# ---------------------------------------------------------------------------


class _MockLlm(BaseLlm):
    """Minimal ``BaseLlm`` that returns a one-line deterministic reply."""

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"mock/.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=f"[mock:{self.model}] task done.")],
            ),
            partial=False,
            turn_complete=True,
        )


def _mock_planner_call_llm(topic: str) -> Callable[[str, str, str], Any]:
    """Canned ``LLMPlanner.call_llm`` — one task per specialist subagent."""
    plan = {
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
        return json.dumps(plan)

    return _call


def _mock_goal_call_llm(topic: str) -> Callable[[str, str, str], Any]:
    """Canned ``LLMGoalDeriver.call_llm`` — one goal for the run."""
    goals = {
        "goals": [
            {
                "id": "g1",
                "summary": f"Produce an interactive slideshow on '{topic}'.",
            }
        ]
    }

    async def _call(system: str, prompt: str, model: str) -> str:
        return json.dumps(goals)

    return _call


# ---------------------------------------------------------------------------
# Live mode — openai-backed planner + goal deriver.
# ---------------------------------------------------------------------------


def _openai_call_llm() -> Callable[[str, str, str], Any]:
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pip install openai to run this example without --mock"
        ) from exc

    client = AsyncOpenAI()

    async def _call(system: str, prompt: str, model: str) -> str:
        resolved = model or os.environ.get(
            "GOLDFIVE_EXAMPLE_PLANNER_MODEL", "gpt-4o-mini"
        )
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
# Optional harmonograf telemetry wiring. Both the ``App`` plugin and the
# runner sink route through a single lazily-created Client so ``adk web``
# and the CLI driver share one connection. If the module or server is
# unreachable we fall through silently — harmonograf is an optional extra.
# ---------------------------------------------------------------------------


_DEFAULT_HARMONOGRAF_SERVER = "127.0.0.1:7531"

_CLIENT: Any | None = None
_ATEXIT_REGISTERED: bool = False


def _shutdown_client() -> None:
    global _CLIENT
    if _CLIENT is None:
        return
    try:
        _CLIENT.shutdown(flush_timeout=5.0)
    except Exception as e:  # noqa: BLE001 — atexit must never raise
        log.debug("harmonograf client shutdown raised: %s", e)
    _CLIENT = None


def _get_or_create_harmonograf_client() -> Any | None:
    """Return a shared harmonograf ``Client`` or ``None`` if unavailable.

    Gated on ``HARMONOGRAF_SERVER`` being set — otherwise this is an
    offline-friendly example and we don't try to reach out.
    """
    global _CLIENT, _ATEXIT_REGISTERED
    if _CLIENT is not None:
        return _CLIENT
    if "HARMONOGRAF_SERVER" not in os.environ:
        return None
    try:
        from harmonograf_client import Client
    except ImportError as e:
        log.warning(
            "harmonograf_client not installed (%s); running without telemetry", e
        )
        return None
    server_addr = os.environ.get("HARMONOGRAF_SERVER", _DEFAULT_HARMONOGRAF_SERVER)
    _CLIENT = Client(
        name="presentation_agent",
        server_addr=server_addr,
        framework="ADK",
        capabilities=["HUMAN_IN_LOOP", "STEERING"],
    )
    if not _ATEXIT_REGISTERED:
        atexit.register(_shutdown_client)
        _ATEXIT_REGISTERED = True
    log.info(
        "harmonograf: presentation_agent client → %s (agent_id=%s)",
        server_addr,
        _CLIENT.agent_id,
    )
    return _CLIENT


# ---------------------------------------------------------------------------
# ``adk web`` ``App`` export — lazy so that importing the module offline
# is side-effect free. ``adk web`` evaluates ``app`` on first use; tests
# can skip the attribute entirely.
# ---------------------------------------------------------------------------


_APP: Any | None = None


def _build_app() -> Any:
    """Construct the ``App`` whose root agent is ``goldfive.wrap(root_agent)``.

    Chooses planner / goal-deriver based on ``OPENAI_API_KEY``:

    * present — live mode, ``openai`` SDK behind ``LLMPlanner`` /
      ``LLMGoalDeriver``.
    * absent — mock mode so ``adk web examples/presentation_agent``
      loads offline without blowing up. Mock mode still exercises the
      full tree; the subagents just return deterministic stub text.

    Optionally attaches ``HarmonografTelemetryPlugin`` when
    ``HARMONOGRAF_SERVER`` is set and ``harmonograf_client`` is
    importable.
    """
    from google.adk.apps.app import App

    live = bool(os.environ.get("OPENAI_API_KEY"))
    topic = os.environ.get("GOLDFIVE_EXAMPLE_TOPIC", "waffles")

    if live:
        call_llm = _openai_call_llm()
        planner_model = os.environ.get(
            "GOLDFIVE_EXAMPLE_PLANNER_MODEL", "gpt-4o-mini"
        )
        tree = root_agent
    else:
        log.info(
            "presentation_agent: OPENAI_API_KEY unset; building App in mock mode "
            "so `adk web` can load offline."
        )
        call_llm = _mock_planner_call_llm(topic)
        planner_model = "mock/planner"
        tree = _build_agent_tree(_MockLlm(model="mock/presentation-agent"))

    planner = LLMPlanner(call_llm=call_llm, model=planner_model)
    goal_deriver = LLMGoalDeriver(call_llm=call_llm, model=planner_model)

    wrapped = goldfive.wrap(tree, planner=planner, goal_deriver=goal_deriver)

    plugins: list[Any] = []
    client = _get_or_create_harmonograf_client()
    if client is not None:
        try:
            from harmonograf_client import HarmonografTelemetryPlugin
        except ImportError as e:
            log.warning(
                "HarmonografTelemetryPlugin unavailable (%s); running without spans",
                e,
            )
        else:
            plugins.append(HarmonografTelemetryPlugin(client))

    return App(name="presentation_agent", root_agent=wrapped, plugins=plugins)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute — build ``app`` on first access."""
    global _APP
    if name == "app":
        if _APP is None:
            _APP = _build_app()
        return _APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# CLI driver — ``python examples/presentation_agent/agent.py [--mock] [--topic]``.
# Same shape as ``examples/adk_presentation/agent.py``.
# ---------------------------------------------------------------------------


async def _run(*, topic: str, mock: bool) -> Any:
    """Build a goldfive runner around the tree and execute one run.

    Returns the :class:`ExecutionOutcome`. Exposed so tests can call it
    directly (``asyncio.run(_run(topic=..., mock=True))``) without
    scraping argv.
    """
    if mock:
        agent_model: str | BaseLlm = _MockLlm(model="mock/presentation-agent")
        planner_call_llm = _mock_planner_call_llm(topic)
        goal_call_llm = _mock_goal_call_llm(topic)
        model_tag = "mock/planner"
    else:
        agent_model = os.environ.get("GOLDFIVE_EXAMPLE_MODEL", "openai/gpt-4o-mini")
        planner_call_llm = _openai_call_llm()
        goal_call_llm = planner_call_llm
        model_tag = os.environ.get("GOLDFIVE_EXAMPLE_PLANNER_MODEL", "gpt-4o-mini")

    tree = _build_agent_tree(agent_model)

    sinks: list[Any] = [LoggingSink()]
    client = _get_or_create_harmonograf_client()
    if client is not None:
        try:
            from harmonograf_client import HarmonografSink

            sinks.append(HarmonografSink(client))
        except ImportError as e:
            log.warning("HarmonografSink unavailable (%s)", e)

    runner = goldfive.wrap(
        tree,
        planner=LLMPlanner(call_llm=planner_call_llm, model=model_tag),
        goal_deriver=LLMGoalDeriver(call_llm=goal_call_llm, model=model_tag),
        executor=SequentialExecutor(max_task_invocations=8),
        sinks=sinks,
    )

    try:
        outcome = await runner.run(f"Make a short presentation about {topic}.")
        print(f"\nsuccess={outcome.success}  reason={outcome.reason!r}")
        if outcome.session.plan is not None:
            for task in outcome.session.plan.tasks:
                print(f"  {task.status.value:<10} {task.id}  {task.title}")
        return outcome
    finally:
        await runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="waffles", help="Presentation topic.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without network — canned planner + MockLlm.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="INFO-level logs from sinks."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )
    asyncio.run(_run(topic=args.topic, mock=args.mock))


if __name__ == "__main__":
    main()
