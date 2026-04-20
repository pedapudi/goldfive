"""Minimal goldfive + ADK example.

The whole point of goldfive is that *you bring an agent and goldfive
orchestrates around it*. This example is the smallest thing that
demonstrates that shape:

    agent  = <your ADK agent>                        # one ``BaseAgent``.
    runner = goldfive.wrap(agent, planner=...)       # goldfive plans + dispatches.
    await runner.run("make a presentation about waffles")

No hand-rolled coordinator, no subagent tree, no file tools wired up
across specialists. ``LLMPlanner`` receives the user prompt, emits a
task DAG, and the same agent handles each task in sequence. This is the
baseline a user should copy when they are integrating goldfive for the
first time.

For a realistic multi-agent reference — coordinator + research /
developer / reviewer / debugger subagents with real tools — see
``tests/reference_agents/presentation_agent/agent.py`` in
`harmonograf <https://github.com/pedapudi/harmonograf>`_. That module
exports ``root_agent`` plus a ``build_goldfive_runner`` helper that
assembles the same ``goldfive.Runner`` as this example, but with the
full four-specialist tree. The point of keeping the goldfive-side
example minimal is that a user reading this file learns *how to wrap*,
not *how to structure a multi-agent ADK tree* — those are orthogonal
concerns.

Two run modes:

* ``--mock`` — no network. Canned JSON for planner / goal-deriver and a
  single ``_MockLlm`` for the ADK agent itself. Good for CI.
* Default — uses ``OPENAI_API_KEY`` via the ``openai`` Python SDK for
  the planner + goal-deriver, and a LiteLLM model string for the ADK
  agent.

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
    from google.genai import types as genai_types
except ImportError as _adk_err:  # pragma: no cover
    raise SystemExit("install goldfive[adk] to run this example") from _adk_err

import goldfive
from goldfive import LLMGoalDeriver, LLMPlanner, SequentialExecutor
from goldfive.sinks import LoggingSink

# ---------------------------------------------------------------------------
# The agent. One ADK ``Agent``; goldfive's planner breaks the user prompt
# into tasks and invokes this same agent for each. No coordinator, no
# subagents. For a richer multi-agent example, see
# harmonograf/tests/reference_agents/presentation_agent/agent.py.
# ---------------------------------------------------------------------------


def _build_agent(model: str | BaseLlm) -> Any:
    return Agent(
        name="presenter",
        model=model,
        instruction=(
            "You are a helpful writing assistant working through one task "
            "at a time. Each message you receive is a single task from an "
            "orchestrator: complete *just that task*, write a short clear "
            "answer, and stop. Do not attempt to run the whole workflow "
            "in one turn — the orchestrator routes you through each step."
        ),
        description="Writing assistant for short presentation tasks.",
    )


# ---------------------------------------------------------------------------
# Mock mode — deterministic, network-free demo path for CI.
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


def _mock_planner_call_llm(topic: str):
    """Canned ``LLMPlanner.call_llm``: three sequential tasks on the agent."""
    plan = {
        "summary": f"Short presentation about {topic}.",
        "tasks": [
            {
                "id": "research",
                "title": f"Research key facts about {topic}",
                "assignee_agent_id": "presenter",
            },
            {
                "id": "outline",
                "title": "Draft a 5-slide outline from the research",
                "assignee_agent_id": "presenter",
            },
            {
                "id": "writeup",
                "title": "Expand each outline bullet into a slide-ready paragraph",
                "assignee_agent_id": "presenter",
            },
        ],
        "edges": [
            {"from_task_id": "research", "to_task_id": "outline"},
            {"from_task_id": "outline", "to_task_id": "writeup"},
        ],
    }

    async def _call(system: str, prompt: str, model: str) -> str:
        return json.dumps(plan)

    return _call


def _mock_goal_call_llm(topic: str):
    """Canned ``LLMGoalDeriver.call_llm``: one goal for the run."""
    goals = {"goals": [{"id": "g1", "summary": f"Produce a short presentation about {topic}."}]}

    async def _call(system: str, prompt: str, model: str) -> str:
        return json.dumps(goals)

    return _call


# ---------------------------------------------------------------------------
# Real mode — openai-backed planner + goal deriver.
# ---------------------------------------------------------------------------


def _openai_call_llm():
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
# Entry point. This is the whole "wrap an agent with goldfive" pattern.
# ---------------------------------------------------------------------------


async def run(*, topic: str, mock: bool) -> None:
    if mock:
        agent_model: str | BaseLlm = _MockLlm(model="mock/presenter")
        planner_call_llm = _mock_planner_call_llm(topic)
        goal_call_llm = _mock_goal_call_llm(topic)
        model_tag = "mock/planner"
    else:
        agent_model = os.environ.get("GOLDFIVE_EXAMPLE_MODEL", "openai/gpt-4o-mini")
        planner_call_llm = _openai_call_llm()
        goal_call_llm = planner_call_llm
        model_tag = os.environ.get("GOLDFIVE_EXAMPLE_PLANNER_MODEL", "gpt-4o-mini")

    # --- the whole wrap, in five lines. ---
    runner = goldfive.wrap(
        _build_agent(agent_model),
        planner=LLMPlanner(call_llm=planner_call_llm, model=model_tag),
        goal_deriver=LLMGoalDeriver(call_llm=goal_call_llm, model=model_tag),
        executor=SequentialExecutor(max_task_invocations=8),
        sinks=[LoggingSink()],
    )

    try:
        outcome = await runner.run(f"Make a short presentation about {topic}.")
        print(f"\nsuccess={outcome.success}  reason={outcome.reason!r}")
        if outcome.session.plan is not None:
            for task in outcome.session.plan.tasks:
                print(f"  {task.status.value:<10} {task.id}  {task.title}")
    finally:
        await runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="waffles", help="Presentation topic.")
    parser.add_argument(
        "--mock", action="store_true", help="Run without network — canned planner + MockLlm."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="INFO-level logs from sinks.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )
    asyncio.run(run(topic=args.topic, mock=args.mock))


if __name__ == "__main__":
    main()
