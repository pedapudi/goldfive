"""Strict-passive observation_only carve-outs for prompt shaping (goldfive#271).

``SteeringConfig.observation_only`` was introduced (#254) to give the
operator a "passive — observe, don't enforce" mode. Subsequent fixes
closed enforcement-side leaks (#260 abort path, #264 PAUSE_ESCALATE,
#267 supersedes-integration). Issue #271 closes the remaining gap:
**prompt-shaping**. Goldfive silently augmented the agent's prompt
context in three ways that survived under ``observation_only=True``,
making the operator's observation be of a goldfive-coached coordinator
rather than the raw one:

1. **Runner conversational-follow-up wrap**
   (:meth:`Runner._wrap_conversational_input`, runner.py ~1701).
   On a follow-up turn the runner rewrote the user's plain input into a
   ``[CONVERSATIONAL FOLLOW-UP — reuse prior plan, don't delegate]``
   directive. Under strict-passive that wrap is skipped: the executor
   receives the raw ``user_input``.

2. **ADK plugin ``before_model_callback`` ``system_instruction``
   injections** (:func:`_inject_goldfive_planner_instruction`,
   :func:`_inject_runtime_tools_hint` in
   :mod:`goldfive.adapters._adk_plugin`). Both append goldfive-shaped
   directive blocks to every LLM call's ``system_instruction``. Under
   strict-passive both are skipped at the callback gate; the
   coordinator's ``system_instruction`` is whatever ADK / the caller
   set it to, with no goldfive augmentation.

3. **Dynamic instruction resolver**
   (:func:`goldfive.adapters._adk_dynainst.make_dynamic_instruction`,
   installed at :func:`goldfive.wrap` time). The resolver replaces every
   wrapped ``LlmAgent.instruction`` with a closure that appends
   "Current assigned task: id/title/description" + any pending-correction
   block on every turn. Under strict-passive the resolver returns the
   ``original_instruction`` verbatim — the caller-supplied string the
   ``LlmAgent`` was constructed with.

Live reproduction context: typewriters / kikuchi session
``1c3602f8-3810-4158-ad8a-3c8f0b79dfdb`` (2026-05-12) showed the
conversational wrap firing under ``observation_only=True``. The audit
that produced this carve-out found two additional injection layers (R3
hint + GoldfivePlanner instruction) and the structural
dynamic-instruction resolver, all silent on observation_only.

Note on visible consequences: removing the conversational wrap may
cause a coordinator that previously got "do NOT delegate" coaching to
re-delegate on a follow-up turn. That's the diagnostic value of
strict-passive — the operator sees the raw behaviour and can address
it in their own coordinator prompt.
"""

from __future__ import annotations

import functools
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
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_runner(*, observation_only: bool) -> tuple[Runner, list[str]]:
    """Build a Runner whose executor.run captures the ``user_input`` it
    sees on each call. Returns ``(runner, captured_user_inputs)``.

    The script: turn 1 produces a real plan; turn 2 is a conversational
    follow-up (``handle_turn`` returns ``None``) so the runner's
    conversational-wrap path fires.
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
            # Turn 2 conversational — null plan to trigger the F6 path.
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

    # Build a steerer with the desired observation_only flag. The
    # autouse default-flip in tests/conftest.py drives the implicit
    # default to ``False`` for the suite; we explicitly pass the value
    # we want so the test's intent wins.
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(observation_only=observation_only)
    )

    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
        steerer=steerer,
    )

    real_executor_run = runner.executor.run
    captured: list[str] = []

    @functools.wraps(real_executor_run)
    async def _spy_run(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs.get("user_input", ""))
        return await real_executor_run(*args, **kwargs)

    runner.executor.run = _spy_run  # type: ignore[method-assign]
    return runner, captured


# ---------------------------------------------------------------------------
# Site 1 — runner conversational-follow-up wrap
# ---------------------------------------------------------------------------


async def test_conversational_wrap_skipped_under_observation_only() -> None:
    """Under ``observation_only=True`` the runner skips the
    ``[CONVERSATIONAL FOLLOW-UP]`` wrap on a follow-up turn — the
    executor sees the user's RAW input.

    Strict-passive contract: the operator observes whatever the
    coordinator chooses to do with the unmodified input. Without the
    wrap, a coordinator may re-delegate; that's the diagnostic value
    of strict-passive, not a regression.
    """
    runner, captured = _make_runner(observation_only=True)

    out1 = await runner.run("make a presentation about solar panels")
    assert out1.success, out1.reason

    out2 = await runner.run("where will the slides be saved?")
    await runner.close()
    assert out2.success, out2.reason

    # Turn 1: raw user input (no wrapping ever for a fresh-plan turn).
    assert captured[0] == "make a presentation about solar panels"

    # Turn 2: STILL raw — the conversational wrap is gated off under
    # observation_only=True. The wrap fingerprint
    # ``[CONVERSATIONAL FOLLOW-UP`` must NOT appear.
    assert captured[1] == "where will the slides be saved?", (
        f"observation_only=True must NOT wrap user_input on a "
        f"conversational follow-up turn; got: {captured[1]!r}"
    )
    assert "CONVERSATIONAL FOLLOW-UP" not in captured[1]
    assert "Do NOT call any AgentTool" not in captured[1]

    # The conversational-turn flag is still set — a downstream plugin
    # layer may still use it for OTHER purposes (e.g. emitting
    # diagnostic spans); the gate is on the WRAP, not the flag.
    assert getattr(out2.session, "_conversational_turn", False) is True


async def test_conversational_wrap_applied_under_observation_disabled() -> None:
    """Regression guard: under ``observation_only=False`` the runner
    STILL wraps the user_input on a follow-up turn — byte-identical to
    pre-#271 behaviour.
    """
    runner, captured = _make_runner(observation_only=False)

    out1 = await runner.run("make a presentation about solar panels")
    assert out1.success, out1.reason

    out2 = await runner.run("where will the slides be saved?")
    await runner.close()
    assert out2.success, out2.reason

    assert captured[0] == "make a presentation about solar panels"
    wrapped = captured[1]
    assert "CONVERSATIONAL FOLLOW-UP" in wrapped, wrapped
    assert "Do NOT call any AgentTool" in wrapped, wrapped
    assert "where will the slides be saved?" in wrapped, wrapped
    # Plan summary threaded through (existing F6 invariant).
    assert "presentation about solar panels" in wrapped, wrapped


async def test_conversational_wrap_log_emitted_under_observation_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The strict-passive carve-out emits an INFO-level skip log so
    operators can grep live logs to confirm the gate engaged.
    """
    import logging

    caplog.set_level(logging.INFO, logger="goldfive.runner")
    runner, _captured = _make_runner(observation_only=True)
    out1 = await runner.run("make a presentation about solar panels")
    assert out1.success
    out2 = await runner.run("where will the slides be saved?")
    await runner.close()
    assert out2.success

    skip_logs = [
        rec
        for rec in caplog.records
        if "observation_only=True" in rec.getMessage()
        and "SKIPPING conversational-follow-up wrap" in rec.getMessage()
    ]
    assert skip_logs, (
        f"expected an INFO log line announcing the skipped conversational "
        f"wrap; got {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Site 2 — ADK plugin ``before_model_callback`` system_instruction injections
# ---------------------------------------------------------------------------
#
# The two injection helpers live in goldfive.adapters._adk_plugin:
#   * ``_inject_goldfive_planner_instruction`` — appends
#     ``GoldfivePlanner.build_planning_instruction`` output
#   * ``_inject_runtime_tools_hint`` — appends "[GOLDFIVE PLAN-STATE HINT…]"
#     block listing pending agents
#
# Both fire inside ``_GoldfiveADKPlugin.before_model_callback``. The
# strict-passive gate is a single check on
# ``ctx.steerer._observation_only`` at the callback entry — the same
# shape DefaultSteerer._should_inject uses elsewhere. The unit test
# below drives the gate predicate directly (the full ADK callback
# requires google-adk + a live LlmRequest; the predicate is the
# behaviour-defining surface).


def test_observation_only_active_helper_returns_false_when_no_ctx() -> None:
    """The plugin's ``_observation_only_active`` helper degrades safely
    when no SessionContext is reachable — the pre-#271 paths (unit-test
    stubs that never call ``set_active_context``) keep working."""
    from goldfive.adapters._adk_plugin import _observation_only_active

    assert _observation_only_active(None) is False


def test_observation_only_active_helper_true_when_steerer_passive() -> None:
    """The plugin's ``_observation_only_active`` helper returns True
    when ``ctx.steerer._observation_only`` is True."""
    from goldfive.adapters._adk_plugin import _observation_only_active

    class _Ctx:
        steerer = DefaultSteerer(
            steering_config=SteeringConfig(observation_only=True)
        )

    assert _observation_only_active(_Ctx()) is True


def test_observation_only_active_helper_false_when_steerer_active() -> None:
    """The plugin's ``_observation_only_active`` helper returns False
    when ``ctx.steerer._observation_only`` is False (active steering)."""
    from goldfive.adapters._adk_plugin import _observation_only_active

    class _Ctx:
        steerer = DefaultSteerer(
            steering_config=SteeringConfig(observation_only=False)
        )

    assert _observation_only_active(_Ctx()) is False


# ---------------------------------------------------------------------------
# Site 3 — dynamic instruction resolver
# ---------------------------------------------------------------------------
#
# ``make_dynamic_instruction`` builds a resolver that ADK calls every
# turn to produce the LlmAgent's ``instruction`` string. Under
# observation_only the resolver must return the ``original_instruction``
# unchanged.


def _make_readonly_ctx_with_steerer(
    *, observation_only: bool, session: Session | None
) -> Any:
    """Build a fake ReadonlyContext that carries the goldfive
    SessionContext stash the resolver walks to.
    """
    from goldfive.adapters._adk_plugin import SessionContext

    steerer = DefaultSteerer(
        steering_config=SteeringConfig(observation_only=observation_only)
    )
    sess = session or Session(run_id="r-strict-passive-test")
    ctx_stash = SessionContext(
        session=sess,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coordinator",
    )

    class _ReadonlyCtx:
        # The resolver tries the live-run path (plugin_manager.plugins)
        # first, then falls back to ``state["goldfive._session_context"]``.
        # We use the fallback so this test is self-contained.
        state = {"goldfive._session_context": ctx_stash}
        _invocation_context = None

    return _ReadonlyCtx()


def test_dynamic_instruction_returns_original_under_observation_only() -> None:
    """Under ``observation_only=True`` the resolver returns the
    ``original_instruction`` verbatim — no "Current assigned task" block,
    no pending-correction block, no goldfive augmentation of any kind.
    """
    from goldfive.adapters._adk_dynainst import make_dynamic_instruction
    from goldfive.types import Goal, Plan, TaskStatus

    # Session with a real pinned task, so the non-strict path WOULD
    # produce a "Current assigned task: …" suffix.
    sess = Session(run_id="r-strict-passive-test")
    sess.goals = [Goal(id="g1", summary="Make a slide")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-strict-passive-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Draft slide",
                description="Write one slide about solar panels",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            )
        ],
        edges=[],
        summary="presentation",
        revision_index=1,
    )
    # Pin the current task so the resolver SHOULD render the augmented
    # block under the non-strict path — proving the gate is the only
    # thing suppressing it.
    from goldfive.orchestration_store import OrchestrationStore

    store = OrchestrationStore.for_session(sess)
    store.set_pin_current_task("t1", title="Draft slide")

    resolver = make_dynamic_instruction(
        original_instruction="You are a friendly slide-drafting agent.",
        agent_name="writer",
    )

    ctx = _make_readonly_ctx_with_steerer(observation_only=True, session=sess)
    out = resolver(ctx)
    assert out == "You are a friendly slide-drafting agent.", (
        f"observation_only=True must return the original instruction "
        f"verbatim; got: {out!r}"
    )
    # Negative checks against the augmented-path fingerprints.
    assert "Current assigned task" not in out
    assert "Draft slide" not in out


def test_dynamic_instruction_augments_under_observation_disabled() -> None:
    """Regression guard: under ``observation_only=False`` the resolver
    appends the "Current assigned task" block to the original
    instruction — byte-identical to pre-#271 behaviour.
    """
    from goldfive.adapters._adk_dynainst import make_dynamic_instruction
    from goldfive.orchestration_store import OrchestrationStore
    from goldfive.types import Goal, Plan, TaskStatus

    sess = Session(run_id="r-strict-passive-test")
    sess.goals = [Goal(id="g1", summary="Make a slide")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-strict-passive-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Draft slide",
                description="Write one slide about solar panels",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            )
        ],
        edges=[],
        summary="presentation",
        revision_index=1,
    )
    store = OrchestrationStore.for_session(sess)
    store.set_pin_current_task("t1", title="Draft slide")

    resolver = make_dynamic_instruction(
        original_instruction="You are a friendly slide-drafting agent.",
        agent_name="writer",
    )

    ctx = _make_readonly_ctx_with_steerer(observation_only=False, session=sess)
    out = resolver(ctx)
    # Active-steering path: the original is followed by the augmented
    # block. The exact format is owned by ``_compose_instruction`` —
    # we check for the structural fingerprints.
    assert out.startswith("You are a friendly slide-drafting agent."), out
    assert "Current assigned task:" in out, out
    assert "id: t1" in out, out
    assert "Draft slide" in out, out
