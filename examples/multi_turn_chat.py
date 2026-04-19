"""multi_turn_chat — a three-turn conversation on one Runner.

Shows :meth:`goldfive.Runner.run` called multiple times on the same
Runner instance. The Runner owns a :class:`goldfive.Conversation` that
persists cross-turn state — ``completed_results`` and goals from turn
1 are visible to the planner on turn 2, and turn 2's output is visible
on turn 3. This is what enables "actually, make it funnier" follow-up
phrasing to land without the caller stitching context by hand.

Run with::

    uv run python examples/multi_turn_chat.py

The example uses a scripted ``call_llm`` stub so it needs no network
credentials. Real callers drop in their own LLM binding and the
cross-turn behaviour is identical.
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


async def writer_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """Return a canned 'written' artefact for each task goldfive assigns."""
    _ = (session, tools)
    return InvocationResult(
        task_id=task.id,
        text=f"Draft produced for {task.title!r}: a limerick about cats.",
    )


def _build_scripted_call_llm():
    """Six scripted responses: (goals, plan) for each of three turns.

    Real integrations delegate to a real model; the point of the script
    is that the planner's *prompt* on turns 2 and 3 now contains a
    rendered summary of prior-turn completed_results and user inputs.
    """
    scripted = [
        # Turn 1 — goals, then plan.
        {"goals": [{"id": "g1", "summary": "Write a limerick about cats"}]},
        {
            "summary": "Draft a cat limerick",
            "tasks": [
                {
                    "id": "draft",
                    "title": "Draft limerick",
                    "description": "Write five rhyming lines about a cat.",
                    "assignee_agent_id": "writer",
                }
            ],
            "edges": [],
        },
        # Turn 2 — user says 'make it funnier'.
        {"goals": [{"id": "g2", "summary": "Make the prior limerick funnier"}]},
        {
            "summary": "Revise the existing limerick for humour",
            "tasks": [
                {
                    "id": "revise",
                    "title": "Punch up the limerick",
                    "description": "Rewrite the existing draft with stronger humour.",
                    "assignee_agent_id": "writer",
                }
            ],
            "edges": [],
        },
        # Turn 3 — user says 'now make it rhyme better'.
        {"goals": [{"id": "g3", "summary": "Tighten the rhymes"}]},
        {
            "summary": "Polish rhymes in the revised limerick",
            "tasks": [
                {
                    "id": "polish",
                    "title": "Polish rhymes",
                    "description": "Tighten the AABBA rhyme scheme.",
                    "assignee_agent_id": "writer",
                }
            ],
            "edges": [],
        },
    ]
    queue = iter(scripted)

    async def _call(system: str, user: str, model: str) -> str:
        _ = (system, user, model)
        try:
            return json.dumps(next(queue))
        except StopIteration:
            return "{}"

    return _call


async def main() -> None:
    sink = InMemorySink()
    runner = goldfive.wrap(
        writer_agent,
        sinks=[sink],
        call_llm=_build_scripted_call_llm(),
        model="scripted",
    )

    print(f"conversation_id: {runner.conversation_id}")
    print()

    for turn_number, user_input in enumerate(
        [
            "Write a limerick about cats",
            "Actually, make it funnier",
            "Now tighten the rhymes",
        ],
        start=1,
    ):
        print(f"--- turn {turn_number}: {user_input!r} ---")
        outcome = await runner.run(user_input)
        print(f"  success={outcome.success}")
        print(
            f"  session.run_id={outcome.session.run_id[:8]}..."
            f"  session.conversation_id={outcome.session.conversation_id[:8]}..."
        )
        print(f"  goals so far: {[g.summary for g in outcome.session.goals]}")
        print(
            f"  completed_results so far: "
            f"{list(outcome.session.completed_results.keys())}"
        )
        print()

    print(f"total turns on conversation: {len(runner.conversation.turns)}")
    print()
    print("Now resetting with runner.new_conversation() — the next run would")
    print("start with empty cross-turn state and a fresh conversation_id.")
    old_id = runner.conversation_id
    await runner.new_conversation()
    print(f"  old conversation_id={old_id[:8]}..., new={runner.conversation_id[:8]}...")

    await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
