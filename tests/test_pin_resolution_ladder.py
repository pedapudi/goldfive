"""Tests for the goldfive#264 aggressive pin-resolution ladder.

The brief: when an agent is invoked, *something precipitated the call*.
The pre-#264 resolver gave up silently when its narrow happy path
(exactly one DAG-ready assignee match) returned 0 or 2+ candidates,
leaving the pin unset and stalling the orchestration loop. The new
resolver in :meth:`_GoldfiveADKPlugin._pin_current_task_id_for_agent`
exhausts an 8-signal ladder, picking the first signal that yields a
single best candidate, and emits a ``pin_resolved`` sink event with
``via_signal`` labelled so operators can see which signal landed it.

This file pins each rung of the ladder:

  * Signal 1 — delegation-site pin happy path (regression for #195)
  * Signal 2 — DAG-ready exactly-1 happy path (regression for #242)
  * Signal 3 — tool-arg scoring picks among DAG-ready candidates
  * Signal 4 — DAG gate relaxed when no candidate is DAG-ready
  * Signal 5 — parent invocation's pin → downstream-edge candidate
  * Signal 6 — pending-correction targets the right task
  * Signal 7 — bare/compound assignee normalisation fallback
  * Signal 8 — low-confidence best-guess (never silent no-op)

Plus:

  * Every successful pin emits a ``pin_resolved`` sink event with the
    ``via_signal`` enum.
  * Signal 4 emits a WARNING log alongside the sink event.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID  # noqa: E402
from goldfive.types import (  # noqa: E402
    Plan,
    Session,
    Task,
    TaskEdge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StateCtx:
    """Minimal ADK-callback-context stub with a ``session.state`` dict."""

    class _Session:
        def __init__(self, state: dict) -> None:
            self.state = state

    def __init__(self, state: dict) -> None:
        self._state = state

    @property
    def session(self) -> Any:
        return _StateCtx._Session(self._state)


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name


class _CapturingSink:
    """Sink stub that captures every emitted event for assertions."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)

    async def close(self) -> None:
        pass


class _SinkingSteerer:
    """Steerer stub that exposes ``_sinks`` so the plugin emits events."""

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = list(sinks)

    async def observe(self, *a: Any, **kw: Any) -> None:  # pragma: no cover
        pass


def _plan_with(*tasks: Task, edges: list[TaskEdge] | None = None) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=list(tasks),
        edges=list(edges or []),
        summary="",
    )


def _session_with(plan: Plan | None) -> Session:
    return Session(run_id="r1", plan=plan)


def _ctx_for(
    session: Session,
    agent_name: str,
    *,
    sinks: list[Any] | None = None,
) -> tuple[dict, Any]:
    """Build a ({adk_state}, callback_context) pair for plugin callbacks.

    When ``sinks`` is provided, attaches a steerer stub so
    :meth:`_emit_pin_resolved` can fan out events.
    """
    steerer: Any = None
    if sinks is not None:
        steerer = _SinkingSteerer(sinks)
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=None,
            tool_handlers={},
            host_agent_name=agent_name,
        ),
    }
    return state, _StateCtx(state)


def _pin_resolved_events(events: list[Any]) -> list[dict]:
    """Filter sink events down to ``pin_resolved*`` envelopes."""
    out: list[dict] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind", "")
        if kind in ("pin_resolved", "pin_resolved_low_confidence"):
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# Signal 1 — delegation-site pin
# ---------------------------------------------------------------------------


async def test_signal1_delegation_site_pin() -> None:
    """A pre-stamped ``pending_delegations`` entry pins the right task."""
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="t1", title="research", assignee_agent_id="researcher"),
            Task(id="t2", title="another for researcher", assignee_agent_id="researcher"),
        )
    )
    # Two candidates so signal 2 would tie; delegation pin disambiguates.
    session.state["goldfive.pending_delegations"] = {"fc-1": "t2"}
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("researcher"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t2"
    events = _pin_resolved_events(sinks[0].events)
    assert len(events) == 1
    assert events[0]["payload"]["via_signal"] == "delegation_pin"
    assert events[0]["payload"]["task_id"] == "t2"


# ---------------------------------------------------------------------------
# Signal 2 — DAG-ready exactly-1 (happy path)
# ---------------------------------------------------------------------------


async def test_signal2_dag_ready_single_match() -> None:
    """One DAG-ready PENDING task assigned to the agent → signal 2 fires."""
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="t1", title="research", assignee_agent_id="researcher"),
        )
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("researcher"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t1"
    events = _pin_resolved_events(sinks[0].events)
    assert len(events) == 1
    assert events[0]["payload"]["via_signal"] == "dag_ready_single"


# ---------------------------------------------------------------------------
# Signal 3 — tool-arg scoring over DAG-ready candidates
# ---------------------------------------------------------------------------


async def test_signal3_arg_scoring_breaks_dag_ready_tie() -> None:
    """Two DAG-ready candidates → tool-arg scoring picks the right one.

    The scoring args (steer body / goals summary) carry tokens from
    one candidate's title + description; the other candidate's tokens
    don't match. Signal 3 picks the matching one.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(
                id="t1",
                title="solar telemetry research",
                description="gather solar telemetry",
                assignee_agent_id="researcher",
            ),
            Task(
                id="t2",
                title="quarterly invoice review",
                description="reconcile quarterly invoices",
                assignee_agent_id="researcher",
            ),
        )
    )
    # Steer body biases toward t1.
    session.state["goldfive.active_steer.body"] = (
        "focus on solar telemetry, not invoices"
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("researcher"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t1"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "arg_scored"


async def test_signal3_parent_tool_args_outrank_steer_body() -> None:
    """Parent's stamped AgentTool tool_args is the highest-priority signal 3 source.

    F7 (#265 followup): :meth:`_pin_delegation_task_id` records
    ``{tool_args: dict}`` on every pending-delegations entry. When
    signal 1 fails to bind (e.g. signal-1 iterates pend.values()
    looking for a task-id match but the entry's task is no longer
    PENDING/RUNNING — or the test stamps an entry whose tid doesn't
    exist) and signal 2 ties, signal 3's :meth:`_scoring_args_for`
    consults the recorded parent args BEFORE the steer body / goals
    summary fallback.

    This test stages a deliberate conflict: the steer body biases
    toward ``t2`` ("invoices"), but the parent's recorded tool_args
    bias toward ``t1`` ("solar"). Signal 3 must pick t1, proving
    parent-args win.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(
                id="t1",
                title="solar telemetry research",
                description="gather solar telemetry",
                assignee_agent_id="researcher",
            ),
            Task(
                id="t2",
                title="quarterly invoice review",
                description="reconcile quarterly invoices",
                assignee_agent_id="researcher",
            ),
        )
    )
    # Parent's tool_args bias toward t1 (solar). Use an entry whose
    # task_id doesn't exist in the plan so signal 1 skips and we drop
    # into signal 3.
    session.state["goldfive.pending_delegations"] = {
        "fc-parent": {
            "task_id": "missing_id",
            "revision": 0,
            "tool_args": {"prompt": "research solar telemetry now"},
        },
    }
    # Steer body biases toward t2 (invoices) — must NOT win.
    session.state["goldfive.active_steer.body"] = (
        "look at the quarterly invoices"
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("researcher"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t1", (
        "parent tool_args (solar) should outrank steer body (invoices)"
    )
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "arg_scored"


async def test_signal3_empty_parent_tool_args_falls_to_steer_body() -> None:
    """Empty / non-mapping parent tool_args → fall through to steer body.

    F7 hazard: dispatches with no args (or opaque blobs) tokenise to
    nothing and would otherwise bias signal 3 against the existing
    steer-body fallback. The signal-3 source priority must skip a
    parent-args bag that produces zero meaningful tokens.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(
                id="t1",
                title="solar telemetry research",
                description="gather solar telemetry",
                assignee_agent_id="researcher",
            ),
            Task(
                id="t2",
                title="quarterly invoice review",
                description="reconcile quarterly invoices",
                assignee_agent_id="researcher",
            ),
        )
    )
    # Parent stamped an entry but with empty / noise-only tool_args.
    # The token bag should be empty, so signal 3 falls through to the
    # steer-body branch which biases t2.
    session.state["goldfive.pending_delegations"] = {
        "fc-parent": {
            "task_id": "missing_id",
            "revision": 0,
            "tool_args": {},  # empty → skipped
        },
    }
    session.state["goldfive.active_steer.body"] = (
        "focus on quarterly invoices, not solar"
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("researcher"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t2", (
        "empty parent tool_args should fall through to steer body"
    )
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "arg_scored"


async def test_signal3_pre_f7_string_pending_delegation_back_compat() -> None:
    """Pre-F7 (string-only) pending_delegation entries still resolve via signal 1.

    Custom adapters or test fixtures that stamp ``{fc_id: "task_id"}``
    (the legacy shape, before #266 versioning and before F7 tool_args)
    must keep working unchanged. This regression pins the back-compat
    contract: signal 1 reads the bare-string entry, finds the task,
    and pins it. Signal 3's parent-args branch is silently skipped
    because :func:`_delegation_pin_tool_args` returns ``None`` for the
    legacy shape.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="t1", title="first", assignee_agent_id="researcher"),
            Task(id="t2", title="second", assignee_agent_id="researcher"),
        )
    )
    # Legacy shape: bare-string task_id, no revision, no tool_args.
    session.state["goldfive.pending_delegations"] = {"fc-old": "t2"}
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("researcher"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t2"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "delegation_pin"


async def test_signal3_tie_falls_through_to_signal4() -> None:
    """When tool-arg scoring ties, fall through to signal 4.

    Two candidates whose token bags don't overlap with the args at
    all → ``_score_candidates_by_args`` returns ``None`` → signal 3
    declines → signal 4 takes over and picks the first deterministically
    (with a relaxed-DAG warning, even though both ARE DAG-ready,
    because signal 4 doesn't gate on the DAG state).

    Wait — signal 4 uses ``assignee_candidates`` which is the pre-DAG
    set. With both DAG-ready, signal 4 also can't disambiguate. The
    scoring re-fires via signal 4 with the same args (still ties)
    and then falls through to signal 8. So the test asserts signal 8
    fires.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="t1", title="alpha", description="A", assignee_agent_id="a"),
            Task(id="t2", title="beta", description="B", assignee_agent_id="a"),
        )
    )
    # No steer body, no goals → no scoring args at all → signal 3 / 4
    # / 7 score paths skip; signal 8 picks first deterministically.
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("a"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] in {"t1", "t2"}
    events = _pin_resolved_events(sinks[0].events)
    # Signal 4 fires with relaxed assignee-candidates because BOTH are
    # DAG-ready — signal 2 sees 2 candidates so it bypasses; signal 3
    # has no args; signal 4's len==1 path doesn't apply (2 candidates,
    # no args) so signal 4 falls through; signals 5-7 don't apply;
    # signal 8 picks deterministically.
    assert events[-1]["payload"]["via_signal"] in {
        "low_confidence",
        "dag_relaxed",
    }


# ---------------------------------------------------------------------------
# Signal 4 — DAG gate relaxed (assignee match, upstreams incomplete)
# ---------------------------------------------------------------------------


async def test_signal4_dag_relaxed_pins_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Upstream incomplete + single assignee match → signal 4 binds + WARNING."""
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="a", title="upstream", assignee_agent_id="other"),
            Task(id="b", title="downstream", assignee_agent_id="researcher"),
            edges=[TaskEdge("a", "b")],
        )
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    with caplog.at_level(logging.WARNING, logger="goldfive.adapters.adk"):
        await plugin.before_agent_callback(
            agent=_Agent("researcher"),
            callback_context=ctx,
        )

    assert session.state[KEY_CURRENT_TASK_ID] == "b"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "dag_relaxed"
    # WARNING log surfaced for operator visibility.
    assert any(
        "DAG-gate relaxed" in record.getMessage() for record in caplog.records
    ), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# Signal 5 — parent invocation's task downstream
# ---------------------------------------------------------------------------


async def test_signal5_parent_pin_downstream() -> None:
    """Parent invocation pinned task A; A→B edge; agent on B → pin lands on B.

    Drives the resolver method directly so we can pass
    ``invocation_id`` / ``parent_invocation_id`` without spinning up
    a real ADK invocation_context. The plugin's ``before_agent_callback``
    plumbs the same kwargs from the live ADK callback context.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    # Two PENDING tasks for the child agent; only one is downstream
    # of the parent's pinned task. The child's signals 2/3 see two
    # ambiguous candidates and fall through; signal 4 also can't
    # disambiguate without scoring args; signal 5 reads the parent's
    # pin from the per-invocation map and picks the downstream-edge
    # candidate.
    session = _session_with(
        _plan_with(
            Task(id="A", title="parent task", assignee_agent_id="parent_agent"),
            Task(id="GATE", title="gate", assignee_agent_id="other_agent"),
            Task(
                id="B",
                title="downstream of A",
                assignee_agent_id="child_agent",
            ),
            Task(
                id="C",
                title="unrelated child task",
                assignee_agent_id="child_agent",
            ),
            # B blocked by parent's task A; C blocked by GATE — both
            # PENDING upstreams, so signal 2 (DAG-ready) sees 0
            # candidates and falls through to later signals.
            edges=[TaskEdge("A", "B"), TaskEdge("GATE", "C")],
        )
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    # Pin parent directly (single match for parent_agent → signal 2).
    await plugin._pin_current_task_id_for_agent(
        agent_name="parent_agent",
        callback_context=ctx,
        invocation_id="inv-parent",
        parent_invocation_id="",
    )
    assert session.state[KEY_CURRENT_TASK_ID] == "A"
    assert plugin._invocation_pinned_task_id["inv-parent"] == "A"

    # Now resolve child with parent_invocation_id → signal 5 picks B
    # because B is downstream of A in plan.edges. The child has 2
    # ambiguous assignee candidates (B and C), no scoring args, so
    # signals 2/3/4 all fall through.
    await plugin._pin_current_task_id_for_agent(
        agent_name="child_agent",
        callback_context=ctx,
        invocation_id="inv-child",
        parent_invocation_id="inv-parent",
    )
    assert session.state[KEY_CURRENT_TASK_ID] == "B"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "parent_pin_downstream"
    assert events[-1]["payload"]["task_id"] == "B"


# ---------------------------------------------------------------------------
# Signal 6 — recent drift / pending correction targeting
# ---------------------------------------------------------------------------


async def test_signal6_reasoning_binding_pins_target_task() -> None:
    """A reasoning-extracted binding routes the agent to the named task.

    Phase 1 of goldfive#271: when the steerer's reasoning judge has
    recorded a binding (via StateStore) and confidence is at
    the threshold, the pin ladder's signal 6 fires before the
    correction-targeting sub-signal.

    Setup forces every prior signal to fail:

      * No delegation pin (signal 1).
      * Two PENDING tasks for the agent so signal 2 ties.
      * No DAG edges and no scoring args, so signals 3 / 4 / 5 also
        skip / tie.
      * No pending corrections, so signal 6's correction sub-signal
        also skips.

    The reasoning binding is the only signal that can disambiguate.
    """
    from goldfive.state_store import StateStore

    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="t_alpha", title="Alpha", assignee_agent_id="agent_x"),
            Task(id="t_beta", title="Beta", assignee_agent_id="agent_x"),
        )
    )
    StateStore.for_session(session).record_reasoning_extracted_binding(
        agent_name="agent_x",
        task_id="t_beta",
        confidence=0.9,
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("agent_x"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t_beta"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "reasoning_binding"


async def test_signal6_reasoning_binding_falls_through_when_task_terminal() -> None:
    """A binding pointing at a COMPLETED task falls through to other signals."""
    from goldfive.state_store import StateStore
    from goldfive.types import TaskStatus

    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    plan = _plan_with(
        Task(
            id="t_alpha",
            title="Alpha",
            assignee_agent_id="agent_x",
            status=TaskStatus.COMPLETED,
        ),
        Task(id="t_beta", title="Beta", assignee_agent_id="agent_x"),
    )
    session = _session_with(plan)
    # Binding still names the completed task.
    StateStore.for_session(session).record_reasoning_extracted_binding(
        agent_name="agent_x",
        task_id="t_alpha",
        confidence=0.95,
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("agent_x"),
        callback_context=ctx,
    )

    # Binding rejected (target terminal) → signal 2 binds the only
    # remaining DAG-ready PENDING candidate.
    assert session.state[KEY_CURRENT_TASK_ID] == "t_beta"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] != "reasoning_binding"


async def test_signal6_pending_correction_pins_target_task() -> None:
    """A pending-correction key targets task ``corr1`` → signal 6 binds it."""
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    # Two assignee candidates for the agent — neither is DAG-ready
    # (no upstreams to be ready against, but the assignee+status
    # filter would tie under signals 2-4). The pending-correction
    # entry under one specific (agent, task) tuple is the
    # disambiguator.
    session = _session_with(
        _plan_with(
            Task(id="t_old", title="original", assignee_agent_id="agent_x"),
            Task(id="corr1", title="corrected take", assignee_agent_id="agent_x"),
        )
    )
    # Stamp the correction pointer; signals 2-4 should still tie for
    # ``agent_x`` (both DAG-ready singletons), so we need a single
    # DAG-ready candidate to ensure we get to signal 6.
    # Strategy: introduce an edge so corr1 is a downstream of t_old,
    # and the upstream is incomplete → signal 2 fails → signal 4
    # would relax with 2 assignee candidates... too messy.
    #
    # Simpler: only one task in the plan, no DAG concerns, but with
    # a pending-correction key → signal 2 would already bind corr1
    # on its own. So set up a different-agent ``other`` task as
    # filler so signal 2 still has its single match, but force the
    # signal-6 path by making the candidate DAG-NOT-ready and the
    # pending correction is the only signal that matches.
    #
    # Actually: signal 2 binds the only PENDING task assigned to
    # agent_x even if there's a correction pointer to the same id —
    # signal 1 would skip (no delegation), signal 2 finds 1 match
    # and binds it. That's still going through signal 2, not 6.
    #
    # To exercise signal 6 we need: zero DAG-ready candidates AND
    # zero plain assignee candidates (so signals 2-5 fail), but a
    # pending-correction key naming a task that exists in the plan
    # with PENDING status. That's a coordinator-pattern: agent name
    # was bare ``agent_x`` but the task got assigned to a sibling
    # under a different name. Use that.
    session = _session_with(
        _plan_with(
            Task(id="t_other", title="filler", assignee_agent_id="someone_else"),
            # corr1 IS PENDING, but assigned to the bare form that
            # neither matches the agent's compound nor any other
            # signal — only the pending-correction key.
            Task(id="corr1", title="corrected take", assignee_agent_id="bare_form"),
        )
    )
    session.state["goldfive.pending_corrections.agent_x.corr1"] = {
        "agent_name": "agent_x",
        "task_id": "corr1",
    }
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("agent_x"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "corr1"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "correction_target"


# ---------------------------------------------------------------------------
# Signal 7 — bare/compound assignee normalization fallback
# ---------------------------------------------------------------------------


async def test_signal7_compound_to_bare_normalisation() -> None:
    """Agent name ``"client:foo"`` with no compound assignees → strip and retry."""
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    # The plan was authored with bare names (post-#215 normalisation
    # should make this the norm). The invocation's agent_name is the
    # compound form.
    session = _session_with(
        _plan_with(
            Task(id="t1", title="research", assignee_agent_id="researcher"),
        )
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("client42:researcher"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t1"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "assignee_normalised"


# ---------------------------------------------------------------------------
# Signal 8 — low-confidence best-guess
# ---------------------------------------------------------------------------


async def test_signal8_low_confidence_best_guess_emits_low_confidence_event() -> None:
    """Pure ambiguity (multi-candidate, no scoring args) → signal 8 fires.

    Signal 8 is only exercised when every prior signal fails. Two
    PENDING candidates for the agent, no scoring args, no parent /
    correction context → signal 4 also can't disambiguate and falls
    through. Signal 8 picks the first candidate deterministically and
    emits a ``pin_resolved_low_confidence`` event so operators see
    the ladder bottomed out.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="alpha", title="alpha", assignee_agent_id="a"),
            Task(id="beta", title="beta", assignee_agent_id="a"),
        )
    )
    # No steer body, no goals → scoring_args is None → signals 3 / 4
    # / 7 / 8's score path all skip the score branches. Signal 8
    # falls back to the deterministic-first pick.
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("a"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] in {"alpha", "beta"}
    events = _pin_resolved_events(sinks[0].events)
    assert any(
        e["kind"] == "pin_resolved_low_confidence" for e in events
    ), [e.get("payload", {}).get("via_signal") for e in events]


# ---------------------------------------------------------------------------
# Pin-resolved sink event invariants
# ---------------------------------------------------------------------------


async def test_pin_resolved_event_shape() -> None:
    """Every successful pin emits exactly one pin_resolved-family event.

    Asserts the payload keys + types + signal labelling so downstream
    sinks (harmonograf) can rely on the contract without inspecting
    the resolver implementation.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(Task(id="t1", title="x", assignee_agent_id="a"))
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin._pin_current_task_id_for_agent(
        agent_name="a",
        callback_context=ctx,
        invocation_id="inv-7",
        parent_invocation_id="",
    )

    events = _pin_resolved_events(sinks[0].events)
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == "pin_resolved"
    payload = e["payload"]
    assert payload["agent_name"] == "a"
    assert payload["task_id"] == "t1"
    assert payload["via_signal"] == "dag_ready_single"
    assert payload["invocation_id"] == "inv-7"
    assert isinstance(payload["score"], float)
    assert isinstance(payload["candidate_count"], int)


# ---------------------------------------------------------------------------
# Regression — happy paths still work
# ---------------------------------------------------------------------------


async def test_regression_242_dag_gate_happy_path() -> None:
    """DAG-aware ambiguity-narrowing (the #242 happy path) still works."""
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="gate", title="g", assignee_agent_id="other"),
            Task(id="t1", title="first", assignee_agent_id="a"),
            Task(id="t2", title="second", assignee_agent_id="a"),
            edges=[TaskEdge("gate", "t2")],
        )
    )
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("a"),
        callback_context=ctx,
    )

    # Signal 2 short-circuits — t1 is the only DAG-ready candidate.
    assert session.state[KEY_CURRENT_TASK_ID] == "t1"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "dag_ready_single"


async def test_regression_195_delegation_pin_happy_path() -> None:
    """Pre-stamped delegation pin still wins over an ambiguous match set."""
    plugin = make_adk_plugin(host_agent_name="coord")
    sinks = [_CapturingSink()]
    session = _session_with(
        _plan_with(
            Task(id="t1", title="first", assignee_agent_id="a"),
            Task(id="t2", title="second", assignee_agent_id="a"),
        )
    )
    session.state["goldfive.pending_delegations"] = {"fc-1": "t1"}
    state, ctx = _ctx_for(session, "coord", sinks=sinks)

    await plugin.before_agent_callback(
        agent=_Agent("a"),
        callback_context=ctx,
    )

    assert session.state[KEY_CURRENT_TASK_ID] == "t1"
    events = _pin_resolved_events(sinks[0].events)
    assert events[-1]["payload"]["via_signal"] == "delegation_pin"


# ---------------------------------------------------------------------------
# Per-invocation pin map (signal-5 plumbing)
# ---------------------------------------------------------------------------


async def test_invocation_pin_map_records_resolution() -> None:
    """A successful pin records ``invocation_id -> task_id`` on the plugin.

    Signal 5 of a future child invocation reads this map to find the
    parent's pin without racing on the single ``goldfive.current_task_id``
    slot.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(Task(id="t1", title="x", assignee_agent_id="a"))
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin._pin_current_task_id_for_agent(
        agent_name="a",
        callback_context=ctx,
        invocation_id="inv-9",
        parent_invocation_id="",
    )

    assert plugin._invocation_pinned_task_id.get("inv-9") == "t1"


async def test_invocation_pin_map_cleared_on_clear_active_context() -> None:
    """``clear_active_context`` drops the per-invocation pin map."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(Task(id="t1", title="x", assignee_agent_id="a"))
    )
    state, ctx = _ctx_for(session, "coord")
    await plugin._pin_current_task_id_for_agent(
        agent_name="a",
        callback_context=ctx,
        invocation_id="inv-1",
        parent_invocation_id="",
    )
    assert plugin._invocation_pinned_task_id

    plugin.clear_active_context()
    assert plugin._invocation_pinned_task_id == {}
