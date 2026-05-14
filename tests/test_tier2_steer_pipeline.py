"""Regression tests for the Tier 2 user-steer pipeline (F5+F6+F7+F9).

Closes goldfive#322 (Layer 2 + Layer 4) and references #277, #204.

The four fixes covered here:

* **F5 — Pivot routing.** When the planner's ``handle_turn`` flags
  ``replaces_prior=true`` on a steer like "forget X, do Y instead",
  the runner installs the result through
  :meth:`DefaultSteerer.install_initial_plan` (fresh plan_id,
  no Rule 6 binding) instead of
  :meth:`DefaultSteerer.install_revision_for_drift`.
* **F6 — Conversational dispatch path.** When ``handle_turn`` returns
  ``None`` on a real prior plan, the runner wraps the user_input with
  a directive that frames it as a follow-up question and asks the
  coordinator not to delegate. ``session._conversational_turn`` is
  set so a parallel adapter-plugin layer can tighten the tool surface.
* **F7 — handle_turn validator-feedback retry.** When the first LLM
  attempt produces a Rule 6-violating revision, ``handle_turn`` retries
  once with the validator's error message appended to the prompt
  (mirrors the ``_call_and_validate_refine`` pattern).
* **F9 — Goal id collision.** When the goal deriver mints a colliding
  ``g1`` on a follow-up turn, the runner renumbers it instead of
  silently dropping the new goal.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import (  # noqa: E402
    CallableAdapter,
    Goal,
    InMemorySink,
    InvocationResult,
    LLMPlanner,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers (model on tests/test_planner_handle_turn.py + test_runner_multi_turn.py)
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Returns canned responses, optionally per-call from a script.

    A list of responses lets a single LLM stub answer multiple
    handle_turn calls in sequence (one per turn).
    """

    def __init__(self, responses: str | list[str]) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        # Repeat the last response if the script is exhausted.
        i = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[i]


def _populated_session(*, plan_id: str = "plan-prior") -> Session:
    """Session with a one-completed-task prior plan."""
    s = Session(run_id="r-test")
    s.goals = [Goal(id="g1", summary="Make a 2-slide presentation about solar panels.")]
    s.plan = Plan(
        id=plan_id,
        run_id="r-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_panels",
                title="Research solar panels",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft_slides",
                title="Draft the slides",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[TaskEdge(from_task_id="research_panels", to_task_id="draft_slides")],
        summary="Make a 2-slide presentation about solar panels.",
        revision_index=1,
    )
    return s


def _plan_dict(
    *,
    plan_id: str = "plan-new",
    summary: str = "x",
    tasks: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": plan_id,
        "summary": summary,
        "tasks": tasks
        or [
            {
                "id": "t1",
                "title": "do the thing",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            }
        ],
        "edges": edges or [],
    }


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


# ---------------------------------------------------------------------------
# F5 — Pivot routing
# ---------------------------------------------------------------------------


async def test_f5_pivot_flag_mints_fresh_plan_id() -> None:
    """When the LLM emits ``replaces_prior: true``, the parser drops
    the prior plan id and mints a fresh one.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "user pivoted to a different artefact entirely",
                "replaces_prior": True,
                "plan": _plan_dict(plan_id="ignored", summary="haiku draft"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-stable")
    plan = await planner.handle_turn(
        user_input="forget the slides — write me a haiku about cats instead",
        session=session,
    )
    assert plan is not None
    # Pivot path: id is freshly minted, NOT inherited from the prior.
    assert plan.id != "plan-prior-stable"
    assert plan.id, "fresh plan id should be non-empty"
    # The pivot sentinel is set so the runner routes through
    # install_initial_plan instead of install_revision_for_drift.
    assert getattr(plan, "_goldfive_pivot", False) is True


async def test_f5_replaces_prior_false_preserves_plan_id() -> None:
    """When ``replaces_prior`` is False (or absent), the prior plan id
    is preserved verbatim — the existing revision path.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "additive constraint",
                "replaces_prior": False,
                "plan": _plan_dict(plan_id="ignored"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-stable")
    plan = await planner.handle_turn(
        user_input="make sure the slides are 2 slides max",
        session=session,
    )
    assert plan is not None
    assert plan.id == "plan-prior-stable"
    assert getattr(plan, "_goldfive_pivot", False) is False


async def test_f5_pivot_routes_through_install_initial_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: drive turn 1 (initial plan) then turn 2 with a pivot
    input. Assert turn 2 lands a NEW plan_id (Rule 6 NOT applied) by
    checking that ``install_initial_plan`` was called for both turns
    and ``install_revision_for_drift`` was never called.
    """
    plan_t1 = _plan_dict(
        plan_id="ignored-by-runner",
        summary="2-slide presentation about solar panels",
        tasks=[
            {
                "id": "research_panels",
                "title": "Research solar panels",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            },
            {
                "id": "draft_slides",
                "title": "Draft 2 slides about solar panels",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            },
        ],
        edges=[
            {"from_task_id": "research_panels", "to_task_id": "draft_slides"},
        ],
    )
    # Turn 2: pivot — drop the prior tasks entirely (which would
    # violate Rule 6 if routed as a revision), with replaces_prior=true.
    plan_t2_pivot = _plan_dict(
        plan_id="ignored-by-runner",
        summary="haiku about cats",
        tasks=[
            {
                "id": "write_haiku",
                "title": "Write a haiku about cats",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            }
        ],
        edges=[],
    )

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = model
        if (
            "warrants a plan change" in system
            or "PIVOT vs REVISION" in system
        ):
            # handle_turn call. Decide based on the user message.
            if "haiku" in user.lower() or "forget" in user.lower():
                return json.dumps(
                    {
                        "reasoning": "pivot to a different artefact",
                        "replaces_prior": True,
                        "plan": plan_t2_pivot,
                    }
                )
            return json.dumps(
                {
                    "reasoning": "first turn",
                    "replaces_prior": False,
                    "plan": plan_t1,
                }
            )
        # Plan-generate fall-through (first turn against empty seed).
        return json.dumps(plan_t1)

    planner = LLMPlanner(call_llm=planner_llm, model="stub")
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    # Spy on the steerer's install paths via ``steerer.plans`` (now
    # public surface per #410 facade cleanup; previous DefaultSteerer
    # subclass override is no longer available since the router shims
    # were removed).
    initial_calls: list[tuple[str, bool]] = []
    drift_calls: list[str] = []
    real_initial = runner.steerer.plans.install_initial_plan
    real_drift = runner.steerer.plans.install_revision_for_drift

    async def _spy_initial(
        *, session: Session, plan: Plan, is_pivot: bool = False
    ) -> bool:
        initial_calls.append((plan.id, is_pivot))
        return await real_initial(session=session, plan=plan, is_pivot=is_pivot)

    async def _spy_drift(*, session: Session, drift: Any, revised_plan: Plan) -> bool:
        drift_calls.append(revised_plan.id)
        return await real_drift(
            session=session, drift=drift, revised_plan=revised_plan
        )

    monkeypatch.setattr(runner.steerer.plans, "install_initial_plan", _spy_initial)
    monkeypatch.setattr(
        runner.steerer.plans, "install_revision_for_drift", _spy_drift
    )

    out1 = await runner.run("make a 2-slide presentation about solar panels")
    assert out1.success, out1.reason
    turn1_plan_id = out1.session.plan.id

    out2 = await runner.run("forget the slides — write a haiku about cats")
    await runner.close()
    assert out2.success, out2.reason
    turn2_plan_id = out2.session.plan.id

    # F5 invariant: pivot routed through install_initial_plan, NOT
    # install_revision_for_drift. Turn 1 also goes through
    # install_initial_plan (first-turn path), so we expect 2 calls.
    # Turn 1 is_pivot=False (genuine first turn); turn 2 is_pivot=True.
    assert len(initial_calls) == 2, (
        f"expected 2 install_initial_plan calls (turn 1 + pivot turn 2); "
        f"got {len(initial_calls)}"
    )
    assert initial_calls[0][1] is False, (
        f"turn 1 should be first-turn (is_pivot=False); got {initial_calls[0]}"
    )
    assert initial_calls[1][1] is True, (
        f"turn 2 should be flagged as pivot (is_pivot=True); got {initial_calls[1]}"
    )
    assert len(drift_calls) == 0, (
        f"pivot must NOT route through install_revision_for_drift; "
        f"got {len(drift_calls)} drift install(s)"
    )
    # Plan id changed across the pivot.
    assert turn2_plan_id != turn1_plan_id, (
        f"pivot should mint a fresh plan id; both turns got {turn1_plan_id}"
    )


# ---------------------------------------------------------------------------
# F6 — Conversational dispatch path
# ---------------------------------------------------------------------------


async def test_f6_conversational_turn_wraps_user_input_for_executor() -> None:
    """When handle_turn returns None on a real prior plan, the runner
    wraps the user_input with a CONVERSATIONAL FOLLOW-UP directive
    before handing it to the executor, AND sets
    ``session._conversational_turn = True``.
    """
    plan_t1 = _plan_dict(
        plan_id="ignored",
        summary="presentation about solar panels",
        tasks=[
            {
                "id": "draft_slides",
                "title": "Draft slides",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            }
        ],
    )

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = model
        if "warrants a plan change" in system or "PIVOT vs REVISION" in system:
            # Turn 2 conversational — return null plan.
            if "where" in user.lower():
                return json.dumps(
                    {
                        "reasoning": "factual question about prior",
                        "replaces_prior": False,
                        "plan": None,
                    }
                )
            return json.dumps(
                {"reasoning": "first turn", "replaces_prior": False, "plan": plan_t1}
            )
        return json.dumps(plan_t1)

    planner = LLMPlanner(call_llm=planner_llm, model="stub")
    sink = InMemorySink()

    # Capture user_inputs handed to the agent.
    seen_inputs: list[str] = []

    async def _capture_agent(
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult:
        _ = tools
        # SequentialExecutor passes user_input through invoke_passthrough
        # for the overlay path; the per-task invoke surface here only
        # sees the task. We instead capture via the executor's
        # passthrough invocation below — this branch covers the
        # legacy invocation surface so the test still exercises the
        # plan execution.
        return InvocationResult(task_id=task.id, text=f"done: {task.title}")

    runner = Runner(
        agent=CallableAdapter(_capture_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    # Wrap the executor.run to snapshot the user_input it receives.
    # Preserve the wrapped function's signature so the runner's
    # ``inspect.signature(executor.run)`` parameter probe (which gates
    # whether the runner passes ``user_input=`` at all) still sees a
    # ``user_input`` parameter.
    import functools

    real_executor_run = runner.executor.run
    captured_user_inputs: list[str] = []

    @functools.wraps(real_executor_run)
    async def _spy_run(*args: Any, **kwargs: Any) -> Any:
        captured_user_inputs.append(kwargs.get("user_input", ""))
        return await real_executor_run(*args, **kwargs)

    runner.executor.run = _spy_run  # type: ignore[method-assign]

    out1 = await runner.run("make a presentation about solar panels")
    assert out1.success, out1.reason

    out2 = await runner.run("where will the slides be saved?")
    await runner.close()
    assert out2.success, out2.reason

    # Turn 1: user_input is the raw user request (no wrapping).
    assert captured_user_inputs[0] == "make a presentation about solar panels"
    # Turn 2: user_input is wrapped with the conversational directive.
    wrapped = captured_user_inputs[1]
    assert "CONVERSATIONAL FOLLOW-UP" in wrapped, wrapped
    assert "Do NOT call any AgentTool" in wrapped, wrapped
    assert "where will the slides be saved?" in wrapped, wrapped
    # Plan summary is threaded through.
    assert "presentation about solar panels" in wrapped, wrapped

    # Session flag set so a parallel plugin layer can tighten the tool
    # surface for this turn.
    assert getattr(out2.session, "_conversational_turn", False) is True
    _ = seen_inputs  # silence unused — captured via _spy_run


# ---------------------------------------------------------------------------
# F7 — handle_turn validator-feedback retry
# ---------------------------------------------------------------------------


async def test_f7_handle_turn_retries_on_rule6_validation_failure() -> None:
    """First LLM attempt drops a terminal task (Rule 6 violation);
    second attempt preserves it. handle_turn must succeed via retry
    AND the second attempt's prompt must include the validator's error
    message.
    """
    # Attempt 1: drops the COMPLETED ``research_panels`` task →
    # Plan.validate(for_revision=True, prior=...) raises Rule 6.
    bad_plan = _plan_dict(
        plan_id="ignored",
        summary="solar flares",
        tasks=[
            {
                "id": "draft_slides",
                "title": "Draft slides about solar flares",
                "assignee_agent_id": "writer",
                "status": "COMPLETED",
            },
        ],
        edges=[],
    )
    # Attempt 2: preserves both terminal tasks AND their edge.
    good_plan = _plan_dict(
        plan_id="ignored",
        summary="2-slide presentation about solar flares",
        tasks=[
            {
                "id": "research_panels",
                "title": "Research solar panels",
                "assignee_agent_id": "writer",
                "status": "COMPLETED",
            },
            {
                "id": "draft_slides",
                "title": "Draft the slides",
                "assignee_agent_id": "writer",
                "status": "COMPLETED",
            },
            {
                "id": "research_flares",
                "title": "Research solar flares",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            },
        ],
        edges=[
            {"from_task_id": "research_panels", "to_task_id": "draft_slides"},
        ],
    )
    scripted = _ScriptedLLM(
        [
            json.dumps(
                {
                    "reasoning": "topic shift to flares",
                    "replaces_prior": False,
                    "plan": bad_plan,
                }
            ),
            json.dumps(
                {
                    "reasoning": "preserved terminal tasks this time",
                    "replaces_prior": False,
                    "plan": good_plan,
                }
            ),
        ]
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-id")
    plan = await planner.handle_turn(
        user_input="actually, also cover solar flares",
        session=session,
    )
    assert plan is not None, "retry should produce a valid plan"
    # Two LLM calls: first failed validation, second succeeded.
    assert len(scripted.calls) == 2, scripted.calls
    # Second call's user prompt carries the validator's error message.
    _sys2, user2, _model2 = scripted.calls[1]
    assert "PREVIOUS ATTEMPT FAILED" in user2, user2
    assert "terminal task" in user2 or "missing in revision" in user2, user2


async def test_f7_handle_turn_succeeds_on_first_attempt_when_valid() -> None:
    """No retry when the first attempt validates."""
    good_plan = _plan_dict(
        plan_id="ignored",
        summary="2-slide presentation about solar panels",
        tasks=[
            {
                "id": "research_panels",
                "title": "Research solar panels",
                "assignee_agent_id": "writer",
                "status": "COMPLETED",
            },
            {
                "id": "draft_slides",
                "title": "Draft the slides",
                "assignee_agent_id": "writer",
                "status": "COMPLETED",
            },
            {
                "id": "review",
                "title": "Review the slides",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            },
        ],
        edges=[
            {"from_task_id": "research_panels", "to_task_id": "draft_slides"},
        ],
    )
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "additive review task",
                "replaces_prior": False,
                "plan": good_plan,
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-id")
    plan = await planner.handle_turn(
        user_input="also have someone review the slides",
        session=session,
    )
    assert plan is not None
    assert len(scripted.calls) == 1, "no retry needed when validation passes"


# ---------------------------------------------------------------------------
# F9 — Goal id collision
# ---------------------------------------------------------------------------


class _FixedIdGoalDeriver:
    """Deriver that mints ``g1`` every turn — mirrors the LLM-deriver
    bug where the prompt elicits the same id on every call.
    """

    def __init__(self, summaries_by_input: dict[str, str]) -> None:
        self._map = summaries_by_input

    async def derive(
        self,
        user_input: str,
        *,
        context: Any = None,
    ) -> list[Goal]:
        summary = self._map.get(user_input, user_input)
        return [Goal(id="g1", summary=summary)]


async def test_f9_goal_id_collision_renumbers_instead_of_dropping() -> None:
    """Two consecutive turns whose deriver mints the same ``g1`` id
    must both end up in ``session.goals`` (the second renumbered to
    ``g2``).
    """
    plan_t1 = _plan_dict(
        plan_id="ignored",
        summary="presentation about solar panels",
        tasks=[
            {
                "id": "draft",
                "title": "Draft slides",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            }
        ],
    )
    # Turn 2 revision must preserve the COMPLETED ``draft`` from turn 1
    # (Rule 6) and add a delta task for the funnier tone.
    plan_t2 = _plan_dict(
        plan_id="ignored",
        summary="presentation about solar panels with funny tone",
        tasks=[
            {
                "id": "draft",
                "title": "Draft slides",
                "assignee_agent_id": "writer",
                "status": "COMPLETED",
            },
            {
                "id": "tone_pass",
                "title": "Rewrite slides with a funnier tone",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            },
        ],
        edges=[{"from_task_id": "draft", "to_task_id": "tone_pass"}],
    )

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = model
        if "warrants a plan change" in system or "PIVOT vs REVISION" in system:
            # Distinguish turn 1 (initial plan) from turn 2 (revision)
            # off the user prompt — turn 2's input contains "funnier".
            if "funnier" in user.lower():
                return json.dumps(
                    {
                        "reasoning": "tone tweak",
                        "replaces_prior": False,
                        "plan": plan_t2,
                    }
                )
            return json.dumps(
                {"reasoning": "first turn", "replaces_prior": False, "plan": plan_t1}
            )
        return json.dumps(plan_t1)

    planner = LLMPlanner(call_llm=planner_llm, model="stub")
    deriver = _FixedIdGoalDeriver(
        {
            "make a presentation about solar panels": (
                "Make a presentation about solar panels."
            ),
            "make it funnier": "Make the tone funnier.",
        }
    )
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=deriver,
        sinks=[sink],
    )

    out1 = await runner.run("make a presentation about solar panels")
    assert out1.success, out1.reason

    out2 = await runner.run("make it funnier")
    await runner.close()
    assert out2.success, out2.reason

    # F9 invariant: BOTH goals are in session.goals; the second was
    # renumbered to g2 instead of being silently dropped.
    summaries = [g.summary for g in out2.session.goals]
    assert len(out2.session.goals) == 2, (
        f"expected 2 goals after collision; got {len(out2.session.goals)}: "
        f"{summaries}"
    )
    assert "Make a presentation about solar panels." in summaries, summaries
    assert "Make the tone funnier." in summaries, summaries
    ids = sorted(g.id for g in out2.session.goals)
    assert ids == ["g1", "g2"], ids
