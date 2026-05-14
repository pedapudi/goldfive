"""observation_only carve-out for ``_emit_plan_revised`` side effects (goldfive#267).

:meth:`goldfive.steerer.DefaultSteerer._apply_revision` already gates the
``set_session_plan`` swap under ``observation_only=True``: when the gate
fires it returns ``(revised, was_installed=False)`` and the live
``session.plan`` is left alone. Live session ``5be95c62`` (2026-05-11)
surfaced a path that end-runs that gate.

A LOOPING_REASONING refine produced a corrective plan with
``supersedes: review_presentation -> re_review_presentation``. The
:meth:`_apply_revision` gate fired correctly. But the downstream
:meth:`_emit_plan_revised` call (with ``dry_run=True``) then:

* Computed the supersedes-integrated plan and called
  ``set_session_plan(session, revised)`` — bypassing the gate and
  swapping the live plan pointer to the would-have-been-installed
  revision.
* Re-pinned ``session.current_task_id`` onto the new corrective task
  (the goldfive#237 hook) — so the next ADK turn's pin handler picked
  it up as the live DAG-ready task and dispatched it.
* Stamped ``goldfive.pending_corrections.*`` keys in ``session.state``
  — so the agent's next-turn prompt would include the correction
  directive for a revision that was never installed.

End result: under ``observation_only=True``, a false-positive drift
still drove a full re-draft cycle. This bundle covers the three
mutation sites in :meth:`_emit_plan_revised`'s body that
:meth:`_apply_revision`'s gate did not close:

1. supersedes-integration :func:`goldfive.types.set_session_plan` swap;
2. :func:`goldfive._correction_injection.queue_corrections_for_revision`
   / :func:`clear_obsolete_corrections_on_revision` state writes;
3. :meth:`DefaultSteerer._repin_current_task_on_supersedes`.

Each is asserted under BOTH ``observation_only=False`` (regression guard
— behaviour preserved byte-identical for steering-enabled runs) and
``observation_only=True`` (carve-out fires; ``session`` and
``session.state`` left unchanged; ``PlanRevised`` event still emitted
with ``dry_run=True`` so operators see the preview).

A no-supersedes negative case is included as a guard that the
``_emit_plan_revised`` change does not regress the (already-working)
non-supersedes path.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import state_store as _ostate  # noqa: E402
from goldfive._correction_injection import (  # noqa: E402
    is_pending_correction_key,
)
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_correction_injection.py / test_observation_only.py
# shapes; kept local so this regression bundle stays self-contained).
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class _StubPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None


def _plan_revised_events(sink: ListSink) -> list[Any]:
    return [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "plan_revised"
    ]


def _base_plan() -> Plan:
    """Seed plan: research COMPLETED + review PENDING (mirrors the live
    session shape that triggered the bug — a COMPLETED upstream task and
    a PENDING downstream task that the refine wants to CORRECT).
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_presentation",
                title="Research presentation content",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="researcher",
            ),
            Task(
                id="review_presentation",
                title="Review presentation",
                status=TaskStatus.PENDING,
                assignee_agent_id="reviewer",
            ),
        ],
        edges=[
            TaskEdge(
                from_task_id="research_presentation",
                to_task_id="review_presentation",
            )
        ],
        revision_index=0,
    )


def _revised_plan_with_correct_supersedes() -> Plan:
    """Refine output: ``re_review_presentation`` supersedes
    ``review_presentation`` with ``supersedes_kind=CORRECT``.

    Shape matches the live LOOPING_REASONING refine on session
    ``5be95c62``: the COMPLETED upstream is preserved; the PENDING
    target is corrected by adding a new task linked via supersedes.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_presentation",
                title="Research presentation content",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="researcher",
            ),
            Task(
                id="review_presentation",
                title="Review presentation",
                status=TaskStatus.PENDING,
                assignee_agent_id="reviewer",
            ),
            Task(
                id="re_review_presentation",
                title="Re-review presentation (corrective)",
                status=TaskStatus.PENDING,
                assignee_agent_id="reviewer",
                supersedes="review_presentation",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[
            TaskEdge(
                from_task_id="research_presentation",
                to_task_id="review_presentation",
            )
        ],
        revision_index=1,
    )


def _revised_plan_without_supersedes() -> Plan:
    """Refine output that ADDS a task without any supersedes link.

    Exercises the negative path: under ``observation_only=True`` the
    ``_apply_revision`` gate already prevents installation; this just
    confirms ``_emit_plan_revised`` does not regress for plans without
    supersedes integration.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_presentation",
                title="Research presentation content",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="researcher",
            ),
            Task(
                id="review_presentation",
                title="Review presentation",
                status=TaskStatus.PENDING,
                assignee_agent_id="reviewer",
            ),
            Task(
                id="extra_followup",
                title="Newly added follow-up",
                status=TaskStatus.PENDING,
                assignee_agent_id="reviewer",
            ),
        ],
        edges=[
            TaskEdge(
                from_task_id="research_presentation",
                to_task_id="review_presentation",
            )
        ],
        revision_index=1,
    )


def _make_session(plan: Plan) -> Session:
    """Build a Session with the orchestration-state pin pre-set on the
    PENDING task that the refine wants to supersede. The repin hook
    reads ``session.current_task_id`` AND the state-dict pin; both must
    be left untouched under observation_only.
    """
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship the presentation")],
        plan=plan,
        current_task_id="review_presentation",
        state={_ostate.KEY_CURRENT_TASK_ID: "review_presentation"},
    )


def _drift_looping() -> DriftEvent:
    """A LOOPING_REASONING WARNING drift, matching the live trigger."""
    return DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="reasoning judge flagged a loop on review_presentation",
        current_task_id="review_presentation",
        current_agent_id="reviewer",
        authored_by="goldfive",
    )


async def _drive_emit(
    *,
    observation_only: bool,
    revised_plan: Plan,
) -> tuple[Session, ListSink, Plan, bool]:
    """Run the apply_revision -> emit_plan_revised pipeline and return
    enough state for the assertions to inspect both gate decisions.
    """
    cfg = SteeringConfig(observation_only=observation_only)
    steerer = DefaultSteerer(steering_config=cfg)
    sink = ListSink()
    planner = _StubPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session(_base_plan())
    prev_plan = session.plan
    drift = _drift_looping()

    # goldfive#247: _apply_revision returns ``(revised, was_installed)``.
    chosen, was_installed = steerer.plans._apply_revision(session, revised_plan, drift)
    await steerer.plans._emit_plan_revised(
        session,
        chosen,
        drift,
        prev_plan=prev_plan,
        attempt_id=None,
        dry_run=not was_installed,
    )
    return session, sink, prev_plan, was_installed


# ---------------------------------------------------------------------------
# 1. Primary case — observation_only=True + CORRECT-kind supersedes drift.
# ---------------------------------------------------------------------------


async def test_observation_only_supersedes_does_not_swap_session_plan() -> None:
    """Under ``observation_only=True``, the supersedes-integration
    ``set_session_plan`` swap inside ``_emit_plan_revised`` MUST NOT fire.

    The live session 5be95c62 (2026-05-11, 18:36:15) reproduction: a
    LOOPING_REASONING refine produced a CORRECT-kind supersedes plan;
    the ``_apply_revision`` gate correctly returned
    ``(revised, was_installed=False)``; but the
    supersedes-integration block then called
    ``set_session_plan(session, integrated)`` regardless, flipping the
    live pointer to the would-have-been-installed plan.
    """
    revised_plan = _revised_plan_with_correct_supersedes()
    session, sink, prev_plan, was_installed = await _drive_emit(
        observation_only=True,
        revised_plan=revised_plan,
    )

    # _apply_revision's gate fired — was_installed is False.
    assert was_installed is False, (
        "observation_only=True must gate the apply_revision install; got True"
    )

    # session.plan unchanged — same identity, same revision_index, same tasks.
    assert session.plan is prev_plan, (
        "session.plan must NOT be swapped under observation_only; "
        f"prev id={prev_plan.id} rev={prev_plan.revision_index} "
        f"got id={session.plan.id} rev={session.plan.revision_index}"
    )
    assert session.plan.revision_index == 0, (
        f"session.plan.revision_index must remain 0; got "
        f"{session.plan.revision_index}"
    )
    plan_task_ids = {t.id for t in session.plan.tasks}
    assert "re_review_presentation" not in plan_task_ids, (
        "the corrective task must NOT have leaked into session.plan; "
        f"tasks={plan_task_ids!r}"
    )

    # PlanRevised event STILL emitted (dry_run is observability, not silence).
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1, (
        f"PlanRevised must still fire so operators can preview the "
        f"would-have-applied plan; got {len(revised_events)} event(s)"
    )
    assert revised_events[0].plan_revised.dry_run is True, (
        "dry_run must be True so consumers can distinguish a preview "
        "from a real revision"
    )
    # The emit payload SHOULD show the integrated (re_review_presentation
    # included) plan so operators can preview what would have landed —
    # that's the whole point of dry_run=True observability.
    emitted_task_ids = {t.id for t in revised_events[0].plan_revised.plan.tasks}
    assert "re_review_presentation" in emitted_task_ids, (
        "PlanRevised payload should reflect the integrated would-have-"
        "applied plan (otherwise operators can't preview the refine); "
        f"got tasks={emitted_task_ids!r}"
    )


async def test_observation_only_supersedes_does_not_repin_current_task() -> None:
    """The goldfive#237 ``current_task_id`` repin MUST NOT fire under
    ``observation_only=True``. Without this carve-out, the next ADK turn
    finds the corrective task pinned and dispatches a full re-do cycle
    despite the gate having declined the install.
    """
    revised_plan = _revised_plan_with_correct_supersedes()
    session, _sink, prev_plan, _was_installed = await _drive_emit(
        observation_only=True,
        revised_plan=revised_plan,
    )

    # session.current_task_id unchanged.
    assert session.current_task_id == "review_presentation", (
        "session.current_task_id must NOT be repinned under "
        f"observation_only; got {session.current_task_id!r}"
    )

    # The orchestration-state pin (read by the reporting-handler fallback)
    # is also untouched.
    state_pin = session.state.get(_ostate.KEY_CURRENT_TASK_ID)
    assert state_pin == "review_presentation", (
        "session.state[KEY_CURRENT_TASK_ID] must NOT be repinned under "
        f"observation_only; got {state_pin!r}"
    )
    # Sanity: prev_plan reference is the live plan, confirming nothing
    # silently rebuilt the session pointer.
    assert session.plan is prev_plan


async def test_observation_only_supersedes_does_not_queue_corrections() -> None:
    """``goldfive.pending_corrections.*`` keys MUST NOT be stamped on
    ``session.state`` under ``observation_only=True``. The next-turn
    prompt resolver reads those keys and injects correction directives;
    a stamp here would end-run the gate by steering the agent on the
    next invocation.
    """
    revised_plan = _revised_plan_with_correct_supersedes()
    session, _sink, _prev_plan, _was_installed = await _drive_emit(
        observation_only=True,
        revised_plan=revised_plan,
    )

    correction_keys = [k for k in session.state if is_pending_correction_key(k)]
    assert correction_keys == [], (
        "observation_only must NOT write pending_corrections to state "
        f"(would inject corrective directives on next turn); got {correction_keys!r}"
    )


# ---------------------------------------------------------------------------
# 2. Regression guard — observation_only=False keeps the install path
#    byte-identical (this is the critical "do not break steering" check).
# ---------------------------------------------------------------------------


async def test_active_steering_supersedes_swaps_plan_and_repins() -> None:
    """Regression guard: with ``observation_only=False`` (the default for
    steering-enabled runs), every side effect MUST still fire. Without
    this, the goldfive#267 fix would silently break the active-steering
    install path.
    """
    revised_plan = _revised_plan_with_correct_supersedes()
    session, sink, prev_plan, was_installed = await _drive_emit(
        observation_only=False,
        revised_plan=revised_plan,
    )

    # _apply_revision's gate did NOT fire — was_installed is True.
    assert was_installed is True, (
        "observation_only=False must install the revision; got was_installed=False"
    )

    # session.plan was swapped to the integrated revision.
    assert session.plan is not prev_plan, (
        "session.plan must be swapped under active steering; "
        "the integration plan should replace the prior plan"
    )
    assert session.plan.revision_index == 1, (
        f"revision_index must be bumped to 1; got {session.plan.revision_index}"
    )
    plan_task_ids = {t.id for t in session.plan.tasks}
    assert "re_review_presentation" in plan_task_ids, (
        "the corrective task must be present on session.plan under "
        f"active steering; got tasks={plan_task_ids!r}"
    )

    # The goldfive#237 repin moved the pin onto the corrective successor.
    assert session.current_task_id == "re_review_presentation", (
        "session.current_task_id must be repinned onto the supersedes "
        f"successor; got {session.current_task_id!r}"
    )
    state_pin = session.state.get(_ostate.KEY_CURRENT_TASK_ID)
    assert state_pin == "re_review_presentation", (
        "session.state pin must be repinned onto the successor; "
        f"got {state_pin!r}"
    )

    # pending_corrections.* was stamped for the corrective task.
    correction_keys = [k for k in session.state if is_pending_correction_key(k)]
    assert correction_keys, (
        "active steering must queue corrections for CORRECT-kind supersedes; "
        f"got no keys (state={list(session.state.keys())!r})"
    )
    # The keyed agent is the reviewer (the corrective task's assignee).
    assert any("reviewer.re_review_presentation" in k for k in correction_keys), (
        f"correction key should be ``...reviewer.re_review_presentation``; "
        f"got {correction_keys!r}"
    )

    # PlanRevised event emitted with dry_run=False.
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False, (
        "dry_run must be False for a real revision install"
    )


# ---------------------------------------------------------------------------
# 3. Negative case — observation_only=True + no supersedes in the refine.
#    The ``_apply_revision`` gate already covers this path; assert the
#    ``_emit_plan_revised`` carve-out did not regress the non-supersedes
#    behaviour (no plan swap, no correction writes, no repin).
# ---------------------------------------------------------------------------


async def test_observation_only_no_supersedes_still_does_not_mutate_session() -> None:
    """No-supersedes refine under observation_only.

    The ``_apply_revision`` gate already prevents the install for plans
    without supersedes; this guard confirms the new
    ``_emit_plan_revised`` gating does not regress that path
    (e.g. by inadvertently calling ``set_session_plan`` on a plan with
    no supersedes-integration delta).
    """
    revised_plan = _revised_plan_without_supersedes()
    session, sink, prev_plan, was_installed = await _drive_emit(
        observation_only=True,
        revised_plan=revised_plan,
    )

    assert was_installed is False
    assert session.plan is prev_plan, (
        "session.plan must remain the prior plan (no supersedes "
        "integration, gate held)"
    )
    assert session.plan.revision_index == 0
    plan_task_ids = {t.id for t in session.plan.tasks}
    assert "extra_followup" not in plan_task_ids, (
        f"newly added task must NOT have leaked into session.plan; "
        f"tasks={plan_task_ids!r}"
    )
    assert session.current_task_id == "review_presentation"
    state_pin = session.state.get(_ostate.KEY_CURRENT_TASK_ID)
    assert state_pin == "review_presentation"

    correction_keys = [k for k in session.state if is_pending_correction_key(k)]
    assert correction_keys == []

    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is True
