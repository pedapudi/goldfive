"""live_steering — demo the goldfive :class:`ControlChannel` end-to-end.

Live steering lets an external caller (harmonograf UI, CLI, tests)
interrupt an in-flight goldfive run: pause, resume, cancel, steer the
plan mid-execution, or rewind to an earlier task. This demo wires up
the simplest shape — a 5-task plan driven by a canned callable agent
with 500ms task sleeps — and shows the full flow:

    1. Build a Runner with a :class:`ControlChannel` attached.
    2. Launch ``runner.run`` in a background task.
    3. Wait one second while the first task or two complete.
    4. Send a ``STEER`` :class:`ControlMessage` with a note describing
       the user's intent.
    5. The executor's mid-task race picks it up, cancels the in-flight
       task, and hands the STEER to the steerer — which translates it
       into a ``USER_STEER`` drift and calls ``planner.refine``.
    6. The planner (an :class:`LLMPlanner` fed a canned, offline
       ``call_llm``) produces a fresh 2-task plan that replaces the
       pending work.
    7. The demo prints every sink event it observed plus the final
       plan's revision index / kind so you can see the steering took
       effect.

Offline-safe: uses a deterministic stub for ``call_llm`` — nothing in
this script talks to a real LLM or the harmonograf server. Run with::

    uv run python examples/live_steering.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from goldfive import (
    CallableAdapter,
    DefaultSteerer,
    InMemorySink,
    InvocationResult,
    LLMPlanner,
    PassthroughGoalDeriver,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    Task,
)
from goldfive.control import ControlChannel, ControlKind, ControlMessage

# Two JSON plans the stubbed LLM hands back in order: the initial 5-task
# outline, then the 2-task "publish what we have" plan the STEER drives.
_INITIAL_PLAN_JSON = json.dumps(
    {
        "id": "demo-plan",
        "summary": "Research, outline, draft, review, publish.",
        "tasks": [
            {"id": tid, "title": tid.title(), "description": f"do {tid}",
             "assignee_agent_id": "writer"}
            for tid in ("research", "outline", "draft", "review", "publish")
        ],
        "edges": [
            {"from_task_id": a, "to_task_id": b}
            for a, b in zip(
                ("research", "outline", "draft", "review"),
                ("outline", "draft", "review", "publish"),
                strict=False,
            )
        ],
    }
)

_REFINED_PLAN_JSON = json.dumps(
    {
        "id": "demo-plan",
        "summary": "Cut the draft/review dance. Finalize and publish.",
        "tasks": [
            {"id": "finalize", "title": "Finalize",
             "description": "tighten what we have", "assignee_agent_id": "writer"},
            {"id": "publish_now", "title": "Publish Now",
             "description": "ship it", "assignee_agent_id": "writer"},
        ],
        "edges": [{"from_task_id": "finalize", "to_task_id": "publish_now"}],
    }
)


async def _slow_agent(
    task: Task, session: Session, tools: list[ReportingToolSpec]
) -> InvocationResult:
    """Sleep 500ms per task so STEER has a chance to land mid-task."""
    _ = (session, tools)
    await asyncio.sleep(0.5)
    return InvocationResult(task_id=task.id, text=f"done:{task.id}")


def _describe_event(evt: Any) -> str:
    """Return a compact ``seq=NN kind`` line for an event envelope."""
    if hasattr(evt, "WhichOneof"):
        kind = evt.WhichOneof("payload") or "?"
        seq = getattr(evt, "sequence", "?")
    elif isinstance(evt, dict):
        kind = str(evt.get("kind", "?"))
        seq = evt.get("sequence", "?")
    else:
        kind = "?"
        seq = "?"
    return f"  seq={seq:>3}  {kind}"


async def main() -> None:
    sink = InMemorySink()
    channel = ControlChannel()

    # The stubbed LLM hands back the initial plan on generate(), then the
    # refined plan on the USER_STEER-driven refine() call.
    responses = iter([_INITIAL_PLAN_JSON, _REFINED_PLAN_JSON])

    async def _call_llm(system: str, user: str, model: str) -> str:
        _ = (system, user, model)
        try:
            return next(responses)
        except StopIteration:
            return "{}"

    runner = Runner(
        agent=CallableAdapter(_slow_agent, available_agents=["writer"]),
        planner=LLMPlanner(call_llm=_call_llm, model="stub"),
        executor=SequentialExecutor(max_plan_reinvocations=10),
        goal_deriver=PassthroughGoalDeriver("live-steering-demo"),
        steerer=DefaultSteerer(),
        sinks=[sink],
        control=channel,
    )

    run_task = asyncio.create_task(runner.run("publish a post"))

    # Give the first task or two a chance to complete, then steer.
    await asyncio.sleep(1.0)
    await channel.send(
        ControlMessage(
            kind=ControlKind.STEER,
            payload={
                "note": "cut the draft/review dance — publish what we have",
            },
        )
    )

    outcome = await run_task
    await runner.close()

    print(f"run success={outcome.success}  reason={outcome.reason!r}")
    print(f"run_id={outcome.session.run_id}")
    final_plan = outcome.session.plan
    if final_plan is not None:
        print(
            f"final plan: id={final_plan.id!r} "
            f"revision_index={final_plan.revision_index} "
            f"revision_kind={final_plan.revision_kind!r} "
            f"reason={final_plan.revision_reason!r}"
        )
        print(f"final tasks: {[t.id for t in final_plan.tasks]}")
    print(f"\n{len(sink.events)} events emitted:")
    for evt in sink.events:
        print(_describe_event(evt))


if __name__ == "__main__":
    asyncio.run(main())
