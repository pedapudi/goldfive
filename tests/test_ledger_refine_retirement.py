"""Refine retirement in ledger plan mode (AGENCY-PRESERVATION.md PR 12).

In ledger mode the Plan is a ledger (OUTCOME deliverables + a DISCOVERED
trajectory record), not a forecast the agent is graded against — so the
drift-triggered forecast-repair refine has nothing to repair. Both
forecast-repair paths (the ladder ABSORB/CANCEL_REINVOKE ``planner.refine``
and the promotion ``refine_steer``) are gated on ``plan_mode``; in ledger
mode a goldfive-authored drift takes the ledger rung instead:

* looping kinds → force-FAIL the bound ledger task;
* everything else → the advisory observer note.

Refine survives for exactly three authors: USER_STEER (tested here),
``handle_turn`` replans, and descriptive absorption (separate dispatch
paths). ``_compose_instruction`` renders a ``[GOALS]`` block instead of
the task block for DISCOVERED pins. Forecast mode is byte-identical.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.control import ControlChannel, ControlKind  # noqa: E402
from goldfive.observer_note_queue import ObserverNoteQueue  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskKind,
    TaskStatus,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _RecordingPlanner:
    """Records refine / refine_steer calls; returns a trivial revision."""

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_steer_calls: list[dict[str, Any]] = []

    async def generate(self, *, goals: Any, available_agents: Any, context: Any = None) -> Any:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        plan = kwargs.get("plan")
        return plan  # echo — enough for _apply_revision to accept

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return kwargs.get("plan")

    @property
    def refined(self) -> bool:
        return bool(self.refine_calls or self.refine_steer_calls)


def _make_steerer(*, plan_mode: str) -> tuple[DefaultSteerer, _RecordingPlanner]:
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            plan_mode=plan_mode,
            threshold="warning",
        ),
    )
    planner = _RecordingPlanner()
    steerer.bind(sinks=[_ListSink()], planner=planner)
    return steerer, planner


def _ledger_plan() -> Plan:
    return Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(
            Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),
            Task(
                id="d1",
                title="writer: drafting",
                discovered=True,
                kind=TaskKind.DISCOVERED,
                status=TaskStatus.RUNNING,
            ),
        ),
        edges=(),
        revision_index=1,
    )


def _session() -> Session:
    return Session(
        run_id="r",
        goals=[Goal(id="g", summary="summarise the deck")],
        plan=_ledger_plan(),
    )


def _drift(kind: DriftKind, *, task_id: str = "d1", authored_by: str = "goldfive") -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=DriftSeverity.WARNING,
        detail="agent wandered",
        current_task_id=task_id,
        authored_by=authored_by,
    )


# ---------------------------------------------------------------------------
# Part 1 — drift-triggered forecast-repair refine retirement
# ---------------------------------------------------------------------------


async def test_ledger_goldfive_drift_no_refine_enqueues_note() -> None:
    steerer, planner = _make_steerer(plan_mode="ledger")
    session = _session()

    await steerer.drift.handle_drift(_drift(DriftKind.OFF_TOPIC), session)

    # No forecast-repair refine of either flavour.
    assert planner.refine_calls == []
    assert planner.refine_steer_calls == []
    # The ledger rung for a non-looping drift is the advisory note. Under
    # the default legacy signal_channel the SIGNAL note lands on
    # ``session.pending_nudges`` (the request_context surface uses the
    # ObserverNoteQueue); accept either so the test is channel-agnostic.
    noted = bool(getattr(session, "pending_nudges", None)) or bool(
        ObserverNoteQueue.for_session(session).pending()
    )
    assert noted, "a SIGNAL note should be enqueued in place of the refine"


async def test_forecast_goldfive_drift_still_refines() -> None:
    # Control: the SAME drift in forecast mode exercises a forecast-repair
    # refine (ladder ABSORB or promotion refine_steer) — proving the
    # ledger gate is what retires it.
    steerer, planner = _make_steerer(plan_mode="forecast")
    session = _session()

    await steerer.drift.handle_drift(_drift(DriftKind.OFF_TOPIC), session)

    assert planner.refined, "forecast mode must still run a forecast-repair refine"


async def test_ledger_looping_force_fails_bound_task() -> None:
    steerer, planner = _make_steerer(plan_mode="ledger")
    session = _session()

    await steerer.drift.handle_drift(
        _drift(DriftKind.LOOPING_TOOL_CALL, task_id="d1"), session
    )

    # No refine; the looping deterministic fallback reduces to force-FAIL.
    assert planner.refine_calls == []
    assert planner.refine_steer_calls == []
    d1 = next(t for t in session.plan.tasks if t.id == "d1")
    assert d1.status is TaskStatus.FAILED


async def test_ledger_user_steer_refine_survives() -> None:
    steerer, planner = _make_steerer(plan_mode="ledger")
    session = _session()

    await steerer.drift.handle_drift(
        _drift(DriftKind.USER_STEER, task_id="d1", authored_by="user"), session
    )

    # USER_STEER is one of the three surviving refine authors — it still
    # refines even in ledger mode.
    assert planner.refined, "USER_STEER refine must survive in ledger mode"


async def test_ledger_runaway_delegation_pauses_not_note() -> None:
    """RUNAWAY_DELEGATION (hard-safety) in ledger mode → PAUSE_ESCALATE.

    The forecast CANCEL_REINVOKE follow-on continues via the refine's
    revised plan (the GOLDFIVE_STEER restart carries replacement_task_ids
    from the post-refine plan — audit #402). In ledger mode there is no
    refine to produce that plan, so a note-replay would re-invoke the same
    coordinator on the same plan and likely re-trip the guardrail.
    Stop-and-ask is the safe follow-on — and it must be a CLEAN,
    observable pause (a GOLDFIVE_PAUSE_ESCALATE control + a
    HUMAN_INTERVENTION_REQUIRED drift), never a hang or a silent death.
    """
    steerer, planner = _make_steerer(plan_mode="ledger")
    channel = ControlChannel()
    steerer.bind_control_channel(channel)
    session = _session()

    drift = DriftEvent(
        kind=DriftKind.RUNAWAY_DELEGATION,
        severity=DriftSeverity.CRITICAL,
        detail="coordinator delegated past the cap",
        current_task_id="d1",
        authored_by="goldfive",
    )
    await steerer.drift.handle_drift(drift, session)

    # No forecast repair of either flavour.
    assert planner.refine_calls == []
    assert planner.refine_steer_calls == []
    # Did NOT force-fail the bound task (that's the looping rung, not the
    # hard-safety rung) and did NOT degrade to a mere advisory note.
    d1 = next(t for t in session.plan.tasks if t.id == "d1")
    assert d1.status is not TaskStatus.FAILED
    assert not getattr(session, "pending_nudges", None)
    # A clean, observable stop-and-ask: a GOLDFIVE_PAUSE_ESCALATE control
    # message was dispatched on the channel.
    drained: list[Any] = []
    inbox = channel._inbox  # noqa: SLF001 — test inspection
    while not inbox.empty():
        drained.append(inbox.get_nowait())
    kinds = [getattr(m, "kind", None) for m in drained]
    assert ControlKind.GOLDFIVE_PAUSE_ESCALATE in kinds, (
        f"expected a GOLDFIVE_PAUSE_ESCALATE control; got {kinds}"
    )


# ---------------------------------------------------------------------------
# Part 2 — [GOALS] block for DISCOVERED pins (_compose_instruction)
# ---------------------------------------------------------------------------


def _compose():
    from goldfive.adapters.adk_llm_instrumentation import _compose_instruction

    return _compose_instruction


def test_compose_discovered_pin_renders_goals_block() -> None:
    out = _compose()(
        original="SYS",
        task_id="d1",
        task_title="agent: work",
        task_description="did stuff",
        pending_correction="",
        task_kind="DISCOVERED",
        goals_block="  - summarise the deck",
    )
    assert "[GOALS]" in out
    assert "summarise the deck" in out
    assert "Current assigned task" not in out


def test_compose_forecast_pin_unchanged_task_block() -> None:
    # Default (no kind/goals) — the legacy task block, byte-for-byte.
    out = _compose()(
        original="SYS",
        task_id="f1",
        task_title="T",
        task_description="D",
        pending_correction="",
    )
    assert "Current assigned task:" in out
    assert "[GOALS]" not in out


def test_compose_outcome_pin_renders_task_block() -> None:
    out = _compose()(
        original="SYS",
        task_id="o1",
        task_title="Summary delivered",
        task_description="",
        pending_correction="",
        task_kind="OUTCOME",
        goals_block="  - g",
    )
    # Only DISCOVERED pins switch to [GOALS]; OUTCOME keeps the task block.
    assert "Current assigned task:" in out
    assert "[GOALS]" not in out


def test_compose_discovered_without_goals_falls_back_to_task_block() -> None:
    out = _compose()(
        original="SYS",
        task_id="d1",
        task_title="T",
        task_description="D",
        pending_correction="",
        task_kind="DISCOVERED",
        goals_block="",
    )
    assert "Current assigned task:" in out
    assert "[GOALS]" not in out


def test_compose_discovered_pin_keeps_pending_correction() -> None:
    out = _compose()(
        original="SYS",
        task_id="d1",
        task_title="T",
        task_description="D",
        pending_correction="[CORRECTION] do X",
        task_kind="DISCOVERED",
        goals_block="  - g",
    )
    assert "[GOALS]" in out
    assert "[CORRECTION] do X" in out


def test_task_kind_and_goals_helpers() -> None:
    from goldfive.adapters.adk_llm_instrumentation import (
        _goals_block_from_session,
        _task_kind_from_session,
    )

    plan = Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(
            Task(id="d1", title="disc", discovered=True, kind=TaskKind.DISCOVERED),
            Task(id="f1", title="fc"),
        ),
        edges=(),
    )
    sess = Session(
        run_id="r",
        goals=[Goal(id="g", summary="summarise"), Goal(id="g2", summary="")],
        plan=plan,
    )
    assert _task_kind_from_session(sess, "d1") == "DISCOVERED"
    assert _task_kind_from_session(sess, "f1") == "FORECAST"
    assert _task_kind_from_session(sess, "missing") == ""
    # Empty-summary goal is skipped.
    assert _goals_block_from_session(sess) == "  - summarise"
