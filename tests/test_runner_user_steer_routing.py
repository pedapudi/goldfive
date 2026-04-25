"""F1 / regression: turn-aware planning gate ``refine_existing`` branch
must route through the steerer.

Phase 2.X (goldfive#271 Gap 1):
``_last_plan`` is now stashed for the next turn even when the prior
turn aborted, and the gate's verdict is logged at INFO so operators
can grep ``/tmp/demo-validation.log`` for the actual classification.


PR #210 introduced the planning gate with three verdicts (``new_work``
/ ``conversational`` / ``refine_existing``). The ``refine_existing``
branch called ``planner.refine`` directly and treated the output as a
fresh plan via ``_emit_plan_submitted`` — bypassing every typed
observability hook from #258-#267 that lives on the steerer's
drift-handling pipeline.

This test pins the post-fix routing: a ``refine_existing`` verdict
synthesizes a ``DriftEvent(USER_STEER)`` and hands it to
``DefaultSteerer._handle_drift``, which:

* emits ``DriftDetected(USER_STEER)`` to sinks,
* applies user-steer state (sticky goals, processed_steer dedupe),
* dispatches ``planner.refine`` via the severity ladder,
* installs the revised plan via ``_apply_revision`` (preserving the
  prior plan's id and bumping ``revision_index``),
* emits ``PlanRevised`` (with ``RefineAttempted``/``RefineFailed``
  from #264, supersedes-coverage validator from #263, atomicity
  barrier from #264).

Today's E2E (2026-04-24) found this end-to-end: turn-2 steer
"forget X. tell me about Y." produced a brand-new plan with no
PlanRevised event, no USER_STEER drift, and silently dropped the
prior plan's constraints. This test is the unit-level pin for that
fix.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import (  # noqa: E402
    CallableAdapter,
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

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _kinds(events: list[Any]) -> list[str]:
    """Capitalize the proto oneof name into a CamelCase event name."""
    out: list[str] = []
    for e in events:
        try:
            name = e.WhichOneof("payload") or ""
        except Exception:  # noqa: BLE001
            name = ""
        out.append("".join(part.capitalize() for part in name.split("_")) if name else "")
    return out


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _planner_call_llm_factory(
    *,
    turn1_plan: str,
    turn2_verdict: str,
    refined_plan: str | None,
    on_synthesize: str | None = None,
):
    """Build an ``async call_llm`` that branches on the system prompt.

    Routes:

    * ``turn-classifier`` system prompt -> turn 2 returns
      ``turn2_verdict`` (turn 1 returns ``new_work``).
    * Synthesize-goal system prompt -> ``on_synthesize`` (or a sane
      default JSON shape).
    * Refine system prompt -> ``refined_plan``; ``None`` raises.
    * Anything else -> ``turn1_plan`` (the fresh-plan path).
    """
    async def _call_llm(system: str, user: str, model: str) -> str:
        _ = model
        # Turn 1 has no prior plan, so the Runner skips the gate
        # (line ~309 in runner.py: ``if self._last_plan is not None``).
        # Every "turn-classifier" call therefore corresponds to turn 2+
        # — return ``turn2_verdict`` deterministically.
        if "turn-classifier" in system:
            return json.dumps({"verdict": turn2_verdict, "reason": "follow-up"})
        # _SYNTHESIZE_GOAL_SYSTEM_PROMPT is invoked from
        # planner.synthesize_goal_from_steer; the user prompt prefix is
        # "STEERING DIRECTIVE:" — distinct from the refine_steer
        # _USER_STEER_SYSTEM_PROMPT which says "STEERING directive"
        # (lowercase "directive") and "STEERING NOTE" in the body.
        if "STEERING DIRECTIVE:" in user:
            return on_synthesize or json.dumps(
                {"goal": {"id": "g-steer", "summary": "y"}, "mode": "append"}
            )
        # USER_STEER refine system prompt uniquely contains
        # "STEERING directive" (lowercase d) and "in-flight plan".
        if "STEERING directive" in system or "in-flight plan" in system:
            if refined_plan is None:
                raise RuntimeError("refine forced to fail in test")
            return refined_plan
        # Generic refine system prompt (non-USER_STEER) contains "drift
        # event" — kept as a separate branch in case future tests need
        # it; today our refine_existing test only hits the USER_STEER
        # path.
        if "drift event" in system:
            if refined_plan is None:
                raise RuntimeError("refine forced to fail in test")
            return refined_plan
        # Default: a fresh-plan generate call.
        return turn1_plan

    return _call_llm


# ---------------------------------------------------------------------------
# 1. refine_existing routes through the steerer + emits DriftDetected
#    + emits PlanRevised (NOT PlanSubmitted)
# ---------------------------------------------------------------------------


async def test_refine_existing_routes_through_steerer_and_emits_plan_revised() -> None:
    """The fix: ``refine_existing`` verdict on turn 2 must emit
    ``DriftDetected(USER_STEER)`` and ``PlanRevised`` (with bumped
    ``revision_index`` and a stable ``plan_id``), and MUST NOT emit
    ``PlanSubmitted`` for the revised plan.
    """
    turn1_plan = json.dumps(
        {
            "id": "plan-original",
            "summary": "first plan",
            "tasks": [
                {"id": "t1", "title": "T1", "assignee_agent_id": "writer"},
                {"id": "t2", "title": "T2", "assignee_agent_id": "writer"},
            ],
        }
    )
    refined_plan = json.dumps(
        {
            "id": "plan-original",  # planner.refine reuses prior plan id
            "summary": "revised plan",
            "tasks": [
                {
                    "id": "t1",
                    "title": "T1",
                    "assignee_agent_id": "writer",
                    "status": "COMPLETED",
                },
                {"id": "t3", "title": "T3 (steer)", "assignee_agent_id": "writer"},
            ],
        }
    )

    planner = LLMPlanner(
        call_llm=_planner_call_llm_factory(
            turn1_plan=turn1_plan,
            turn2_verdict="refine_existing",
            refined_plan=refined_plan,
        ),
        model="stub",
    )
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    out1 = await runner.run("make a 2-slide presentation about solar")
    assert out1.success
    turn1_plan_id = out1.session.plan.id
    turn1_revision_index = out1.session.plan.revision_index
    turn1_end = len(sink.events)

    out2 = await runner.run("forget solar. tell me about wind power instead.")
    await runner.close()
    assert out2.success

    turn2_kinds = _kinds(sink.events[turn1_end:])

    # ------------------------------------------------------------------
    # F1 root-cause assertions
    # ------------------------------------------------------------------
    # The steerer's pipeline emits DriftDetected before refining.
    assert "DriftDetected" in turn2_kinds, (
        "USER_STEER drift must appear on the wire — the bug bypassed "
        "_emit_drift_detected by calling planner.refine directly."
    )
    # PlanRevised is the steerer's emit; PlanSubmitted is for fresh plans.
    assert "PlanRevised" in turn2_kinds, (
        "refine_existing branch must emit PlanRevised via the steerer; "
        "the bug emitted PlanSubmitted instead, fresh-minting plan_id."
    )
    assert "PlanSubmitted" not in turn2_kinds, (
        "PlanSubmitted is for fresh plans only — the refined plan is "
        "announced via PlanRevised, not PlanSubmitted."
    )

    # The drift detected this turn must be a USER_STEER.
    drift_kinds_seen: list[int] = []
    for e in sink.events[turn1_end:]:
        try:
            if e.WhichOneof("payload") == "drift_detected":
                drift_kinds_seen.append(e.drift_detected.kind)
        except Exception:  # noqa: BLE001
            continue
    # DriftKind.USER_STEER's int value — resolved against the proto for stability.
    from goldfive.pb.goldfive.v1 import types_pb2

    assert types_pb2.DRIFT_KIND_USER_STEER in drift_kinds_seen, (
        "expected USER_STEER among DriftDetected envelopes "
        f"(saw kinds={drift_kinds_seen!r})"
    )

    # ------------------------------------------------------------------
    # plan_id stable + revision_index bumped
    # ------------------------------------------------------------------
    assert out2.session.plan is not None
    assert out2.session.plan.id == turn1_plan_id, (
        "refine_existing must reuse the prior plan's id "
        f"(turn1_id={turn1_plan_id!r}, turn2_id={out2.session.plan.id!r}). "
        "The bug fresh-minted a new plan_id via _emit_plan_submitted."
    )
    assert out2.session.plan.revision_index == turn1_revision_index + 1, (
        "_apply_revision must bump revision_index by 1 — "
        f"prior={turn1_revision_index}, "
        f"revised={out2.session.plan.revision_index}. "
        "The bug returned the planner's freshly-minted plan with "
        "revision_index=0."
    )


# ---------------------------------------------------------------------------
# 2. Refine returning None on the steerer path falls back to planner.generate
# ---------------------------------------------------------------------------


async def test_refine_existing_falls_back_to_generate_on_refine_failure() -> None:
    """When the steerer's refine path fails (planner.refine raises),
    the runner falls through to ``planner.generate`` and emits
    ``PlanSubmitted`` for the fresh plan. No ``PlanRevised`` is
    emitted (the refine never succeeded).
    """
    turn1_plan = json.dumps(
        {
            "id": "plan-original",
            "summary": "first plan",
            "tasks": [
                {"id": "t1", "title": "T1", "assignee_agent_id": "writer"},
            ],
        }
    )
    fresh_plan_after_refine_fail = turn1_plan  # generate falls through

    planner = LLMPlanner(
        call_llm=_planner_call_llm_factory(
            turn1_plan=fresh_plan_after_refine_fail,
            turn2_verdict="refine_existing",
            refined_plan=None,  # forces refine to raise
        ),
        model="stub",
    )
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    out1 = await runner.run("turn one")
    assert out1.success
    turn1_end = len(sink.events)

    out2 = await runner.run("a steering follow-up")
    await runner.close()
    assert out2.success

    turn2_kinds = _kinds(sink.events[turn1_end:])

    # The steerer still got the drift (it ran _handle_drift even on a
    # refine that raised) — but no revision landed, so no PlanRevised
    # follow-up.
    assert "DriftDetected" in turn2_kinds, (
        "the steerer still emits DriftDetected before attempting refine"
    )
    assert "PlanRevised" not in turn2_kinds, (
        "refine raised — _emit_plan_revised must not fire when no "
        "revision was applied."
    )
    # Fall-through to planner.generate emitted a fresh plan event.
    assert "PlanSubmitted" in turn2_kinds, (
        "fall-back path must emit PlanSubmitted for the regenerated "
        "plan (the safe path always ships a plan)."
    )


# ---------------------------------------------------------------------------
# 3. Phase 2.X / Gap 1: prior plan stashed across an aborted turn so the
#    next turn's planner-gate can route through USER_STEER.
# ---------------------------------------------------------------------------


async def test_prior_plan_stashed_when_turn_aborts_so_next_steer_routes() -> None:
    """Regression for goldfive#271 Gap 1.

    Scenario from the validation E2E:

    1. Turn 1 builds a plan and starts executing it.
    2. Turn 1 aborts mid-flight (in this test: a task fails so the
       executor returns ``ExecutionOutcome(success=False)``).
    3. Turn 2 is a user steer ("forget X. tell me about Y.").

    Pre-Phase-2.X the runner only stashed ``_last_plan`` on
    ``outcome.success=True``, so the gate on turn 2 found
    ``_last_plan = None`` and short-circuited to ``new_work`` —
    silently dropping the steerer's USER_STEER pipeline. The fix
    stashes the plan whenever ``session.plan`` is non-None so the
    gate can refine against it on the next turn even if the prior
    turn was cancelled or failed.
    """
    turn1_plan = json.dumps(
        {
            "id": "plan-aborted",
            "summary": "first plan that aborts",
            "tasks": [
                {"id": "t1", "title": "T1", "assignee_agent_id": "writer"},
            ],
        }
    )
    refined_plan = json.dumps(
        {
            "id": "plan-aborted",
            "summary": "revised after steer",
            "tasks": [
                {
                    "id": "t1",
                    "title": "T1",
                    "assignee_agent_id": "writer",
                    "status": "FAILED",
                },
                {"id": "t2", "title": "T2 (steer)", "assignee_agent_id": "writer"},
            ],
        }
    )

    planner = LLMPlanner(
        call_llm=_planner_call_llm_factory(
            turn1_plan=turn1_plan,
            turn2_verdict="refine_existing",
            refined_plan=refined_plan,
        ),
        model="stub",
    )

    async def _failing_agent(
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult:
        _ = tools, session
        return InvocationResult(task_id=task.id, text="boom", success=False)

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_failing_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    out1 = await runner.run("make a 2-slide presentation about solar")
    # Turn 1 aborts (the agent returns success=False) — but a plan IS
    # installed on the session before the executor runs.
    assert not out1.success
    assert out1.session.plan is not None
    turn1_plan_id = out1.session.plan.id
    turn1_revision_index = out1.session.plan.revision_index
    turn1_end = len(sink.events)

    # The fix: ``_last_plan`` is stashed despite the abort.
    assert runner._last_plan is not None, (
        "Gap 1 regression: the runner must stash session.plan for the "
        "next turn's gate even when the turn aborted; otherwise the "
        "next user steer skips the USER_STEER pipeline."
    )
    assert runner._last_plan.id == turn1_plan_id

    # Turn 2: user steer should classify as refine_existing and route
    # through the steerer's USER_STEER pipeline.
    out2 = await runner.run("forget solar. tell me about wind power instead.")
    await runner.close()

    turn2_kinds = _kinds(sink.events[turn1_end:])
    assert "DriftDetected" in turn2_kinds, (
        "USER_STEER drift must appear on the wire even when the prior "
        "turn aborted — the prior plan is enough context for the gate."
    )
    assert "PlanRevised" in turn2_kinds, (
        "refine_existing routes through the steerer's USER_STEER "
        "pipeline; PlanRevised must fire with bumped revision_index."
    )

    # plan_id stable + revision_index bumped (same invariants as the
    # success-path test but exercised through the abort recovery).
    assert out2.session.plan is not None
    assert out2.session.plan.id == turn1_plan_id
    assert out2.session.plan.revision_index == turn1_revision_index + 1


# ---------------------------------------------------------------------------
# 4. Phase 2.X / Gap 1: gate verdict + USER_STEER routing log lines
# ---------------------------------------------------------------------------


async def test_gate_verdict_logs_at_info_for_operator_visibility(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The runner's planner-gate emits one INFO-level log line per turn
    capturing the verdict. A future operator debugging an E2E should
    be able to grep for ``Runner.run: gate verdict=`` and reconstruct
    the classification path that drove the turn.
    """
    turn1_plan = json.dumps(
        {
            "id": "plan-log",
            "summary": "first plan",
            "tasks": [
                {"id": "t1", "title": "T1", "assignee_agent_id": "writer"},
            ],
        }
    )
    refined_plan = json.dumps(
        {
            "id": "plan-log",
            "summary": "revised plan",
            "tasks": [
                {
                    "id": "t1",
                    "title": "T1",
                    "assignee_agent_id": "writer",
                    "status": "COMPLETED",
                },
                {"id": "t2", "title": "T2 (steer)", "assignee_agent_id": "writer"},
            ],
        }
    )
    planner = LLMPlanner(
        call_llm=_planner_call_llm_factory(
            turn1_plan=turn1_plan,
            turn2_verdict="refine_existing",
            refined_plan=refined_plan,
        ),
        model="stub",
    )
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    with caplog.at_level(logging.INFO, logger="goldfive.runner"):
        await runner.run("first turn")
        await runner.run("forget first. tell me about second instead.")
        await runner.close()

    messages = [rec.getMessage() for rec in caplog.records]

    # First turn: no prior plan → gate skipped log.
    assert any(
        "gate skipped (no prior plan or non-str input)" in m for m in messages
    ), f"expected gate-skipped log on first turn; got: {messages!r}"

    # Second turn: gate verdict logged with verdict + prior_plan_id +
    # user_input snippet.
    verdict_lines = [m for m in messages if "gate verdict=" in m]
    assert verdict_lines, (
        f"expected at least one gate verdict log; got: {messages!r}"
    )
    assert any("verdict=refine_existing" in m for m in verdict_lines), (
        f"expected refine_existing verdict in logs; got: {verdict_lines!r}"
    )

    # USER_STEER routing log fires when refine_existing routes through
    # the steerer.
    assert any(
        "routing refine_existing through steerer USER_STEER pipeline" in m
        for m in messages
    ), (
        "expected USER_STEER routing log; got: "
        f"{[m for m in messages if 'USER_STEER' in m]!r}"
    )

    # _last_plan stash log on a successful turn.
    assert any(
        "stashed prior plan for next turn's gate" in m for m in messages
    ), f"expected _last_plan stash log; got: {messages!r}"
