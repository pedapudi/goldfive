"""hello_callable — the simplest possible goldfive run.

Shows the one-line :func:`goldfive.run` convenience wrapper: hand it a
bare async callable that pretends to be an agent, hand it a user
prompt, and let goldfive pick every default (auto-adapter,
:class:`LiteralGoalDeriver`, :class:`PassthroughPlanner`,
:class:`SequentialExecutor`, :class:`DefaultSteerer`,
:class:`LoggingSink`).

To see the full orchestration loop — an LLM planner decomposing the
goal into tasks and the executor walking them — supply a
``call_llm=`` callable to :func:`goldfive.wrap`. Real agents usually
bring their own LLM via the ADK adapter path, in which case
:func:`goldfive.wrap` will reuse the agent's model automatically.

Run with::

    uv run python examples/hello_callable.py
"""

from __future__ import annotations

import asyncio
import json

import goldfive
from goldfive import (
    InMemorySink,
    InvocationResult,
    ReportingToolSpec,
    Session,
    Task,
)


async def greeter_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """Return a short canned reply for the task goldfive assigns."""
    _ = (session, tools)
    return InvocationResult(
        task_id=task.id,
        text=f"greeted: {task.title}",
    )


def _build_stub_call_llm():
    """Minimal deterministic ``call_llm`` so the demo needs no network.

    Real callers would drop in their own LLM binding (OpenAI, Anthropic,
    ADK's ``LLMRegistry``, etc.) — the contract is just
    ``async (system_prompt, user_prompt, model) -> str`` where the
    returned string is JSON matching the planner / goal-deriver schema.
    """
    plan_json = {
        "summary": "Greet the user.",
        "tasks": [
            {
                "id": "greet",
                "title": "Say hello to the user",
                "description": "Return a short greeting.",
                "assignee_agent_id": "default",
            }
        ],
        "edges": [],
    }
    goals_json = {"goals": [{"id": "g1", "summary": "Say hello"}]}
    responses = iter([json.dumps(goals_json), json.dumps(plan_json)])

    async def _call(system: str, user: str, model: str) -> str:
        _ = (system, user, model)
        try:
            return next(responses)
        except StopIteration:
            return json.dumps({})

    return _call


async def main() -> None:
    sink = InMemorySink()
    outcome = await goldfive.run(
        greeter_agent,
        "say hello to the user",
        sinks=[sink],
        call_llm=_build_stub_call_llm(),
        model="stub-model",
    )

    print(f"success={outcome.success}  reason={outcome.reason!r}")
    print(f"run_id={outcome.session.run_id}")
    print(f"goals={[g.summary for g in outcome.session.goals]}")
    print(f"{len(sink.events)} events:")
    for evt in sink.events:
        if isinstance(evt, dict):
            seq = evt.get("sequence", "?")
            kind = evt.get("kind", "?")
        else:
            seq = getattr(evt, "sequence", "?")
            kind = (
                evt.WhichOneof("payload")
                if hasattr(evt, "WhichOneof")
                else "?"
            ) or "?"
        print(f"  seq={seq:>3}  {kind}")


if __name__ == "__main__":
    asyncio.run(main())
