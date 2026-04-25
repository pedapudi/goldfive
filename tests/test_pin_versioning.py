"""Tests for goldfive#266 pin versioning.

The brief: every pin write (current_task_id + pending_delegations)
carries the plan ``revision_index`` in effect at that moment. The
report-time classifier in :mod:`goldfive.reporting` consults the stamp
to distinguish:

* **Fresh pin** (pin_revision == current_revision) — proceed normally.
  A fresh pin pointing at a terminal task whose successor is
  REPLACE-kind still routes via the existing supersedes helper (LLM
  retried its own pin and the planner already repointed).
* **Stale REPLACE pin** (pin_revision < current_revision, successor is
  REPLACE-kind) — route to the successor (existing behaviour
  preserved for the most common refine path).
* **Stale CORRECT pin** (pin_revision < current_revision, successor is
  CORRECT-kind) — REFUSE. The old task's terminal state is historical
  fact; the correction is a separate work unit. ``acknowledged: True``
  is still returned to the LLM (no prompt-injection surface), but a
  ``TaskTransitionRefused`` proto sink event flags the refusal for
  operators (promoted from the original dict shape per the #262
  InvocationCancelled pattern).
* **Stale ambiguous pin** (pin_revision < current_revision, no
  supersedes successor) — REFUSE. Operator must disambiguate.

Plus:

* Pending-delegations entry shape evolved from ``{fc_id: task_id}``
  to ``{fc_id: {task_id, revision}}``. Readers tolerate both for
  back-compat.
* Stamp is written once at ``_stamp_current_task_id`` so all 8
  signals of the goldfive#264/#265 pin ladder pick it up uniformly.
* Concurrent refine: the report handler waits on the steerer's
  ``_wait_plan_stable`` barrier (#264) before reading
  ``plan.revision_index``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    _delegation_pin_revision,
    _delegation_pin_task_id,
    _resolve_pinned_task_id,
    make_adk_plugin,
)
from goldfive.adapters._adk_state_protocol import (  # noqa: E402
    KEY_CURRENT_TASK_ID,
    KEY_CURRENT_TASK_REVISION,
)
from goldfive.reporting import BUILTIN_REPORTING_TOOLS  # noqa: E402
from goldfive.types import (  # noqa: E402
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StateCtx:
    """Minimal ADK-callback-context stub with a ``session.state`` dict."""

    class _Session:
        def __init__(self, state: dict) -> None:
            self.state = state

    class _ToolCtx:
        def __init__(self, state: dict, fc_id: str) -> None:
            self._state = state
            self.function_call_id = fc_id

        @property
        def session(self) -> Any:
            return _StateCtx._Session(self._state)

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
    """Steerer stub that exposes ``_sinks`` so the plugin emits events.

    Optional ``wait_plan_stable_delay`` lets a test simulate a refine
    in flight: the helper sleeps before returning, which is the same
    semantic as a real DefaultSteerer holding the plan lock.
    """

    def __init__(
        self,
        sinks: list[Any],
        *,
        wait_plan_stable_delay: float = 0.0,
        on_wait: Any = None,
    ) -> None:
        self._sinks = list(sinks)
        self._wait_calls = 0
        self._wait_delay = wait_plan_stable_delay
        self._on_wait = on_wait

    async def observe(self, *a: Any, **kw: Any) -> None:  # pragma: no cover
        pass

    async def _wait_plan_stable(self, session: Any, *, timeout: float = 1.0) -> bool:
        self._wait_calls += 1
        if self._on_wait is not None:
            await self._on_wait(session)
        if self._wait_delay > 0:
            await asyncio.sleep(self._wait_delay)
        return True

    # No-op transitions so the report handlers can drive through.
    async def mark_task_running(self, *a: Any, **kw: Any) -> None:
        pass

    async def mark_task_progress(self, *a: Any, **kw: Any) -> None:
        pass

    async def mark_task_completed(self, *a: Any, **kw: Any) -> None:
        pass

    async def mark_task_failed(self, *a: Any, **kw: Any) -> None:
        pass

    async def mark_task_blocked(self, *a: Any, **kw: Any) -> None:
        pass


def _plan_with(
    *tasks: Task,
    edges: list[TaskEdge] | None = None,
    revision_index: int = 0,
) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=list(tasks),
        edges=list(edges or []),
        summary="",
        revision_index=revision_index,
    )


def _session_with(plan: Plan | None) -> Session:
    return Session(run_id="r1", plan=plan)


def _ctx_for(
    session: Session,
    agent_name: str,
    *,
    sinks: list[Any] | None = None,
) -> tuple[dict, Any]:
    """Build a ({adk_state}, callback_context) pair for plugin callbacks."""
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


def _tool(name: str):
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"builtin tool {name!r} missing")


def _refused_events(events: list[Any]) -> list[Any]:
    """Return ``TaskTransitionRefused`` proto envelopes from the sink stream.

    Promoted from the dict shape #266 originally shipped — the typed
    proto message is now the canonical wire shape (matches the
    ``InvocationCancelled`` promotion pattern from #262).
    """
    out: list[Any] = []
    for evt in events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") == "task_transition_refused":
                out.append(evt)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# 1. Pin stamp revision is written alongside task_id (current_task_id surface).
# ---------------------------------------------------------------------------


async def test_pin_stamp_writes_revision_alongside_task_id() -> None:
    """The pin ladder stamps ``current_task_revision`` on every signal."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="research", assignee_agent_id="researcher"),
            revision_index=3,
        )
    )
    state, ctx = _ctx_for(session, "coord", sinks=[_CapturingSink()])

    await plugin.before_agent_callback(
        agent=_Agent("researcher"),
        callback_context=ctx,
    )

    # Pin landed via signal 2 (DAG-ready exactly-1 happy path).
    assert state[KEY_CURRENT_TASK_ID] == "t1"
    # And the revision stamp followed it on the same surface.
    assert state[KEY_CURRENT_TASK_REVISION] == 3
    # Mirrored on goldfive's session.state too (the shared contract).
    assert session.state.get("goldfive.current_task_id") == "t1"
    assert session.state.get("goldfive.current_task_revision") == 3


# ---------------------------------------------------------------------------
# 2. Pending-delegation includes revision in the new dict shape.
# ---------------------------------------------------------------------------


async def test_pending_delegation_carries_revision() -> None:
    """``_pin_delegation_task_id`` writes the versioned dict shape."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="research solar", assignee_agent_id="researcher"),
            revision_index=2,
        )
    )
    state, _ = _ctx_for(session, "coord")
    tool_ctx = _StateCtx._ToolCtx(state, fc_id="fc-42")

    plugin._pin_delegation_task_id(  # type: ignore[attr-defined]
        ctx=state[SESSION_CONTEXT_STATE_KEY],
        tool_context=tool_ctx,
        to_agent="researcher",
        tool_args={"topic": "research solar"},
    )

    pend = session.state["goldfive.pending_delegations"]
    entry = pend["fc-42"]
    assert isinstance(entry, dict), "delegation entry must be the versioned shape"
    assert entry["task_id"] == "t1"
    assert entry["revision"] == 2

    # And the back-compat extractors reproduce the right values.
    assert _delegation_pin_task_id(entry) == "t1"
    assert _delegation_pin_revision(entry) == 2


# ---------------------------------------------------------------------------
# 3. Handler: pin_rev == current_rev → normal transition.
# ---------------------------------------------------------------------------


async def test_handler_fresh_pin_proceeds_normally() -> None:
    plan = _plan_with(
        Task(id="t1", title="research", assignee_agent_id="researcher"),
        revision_index=2,
    )
    session = _session_with(plan)
    session.state["goldfive.current_task_id"] = "t1"
    session.state["goldfive.current_task_revision"] = 2

    sinks = [_CapturingSink()]
    steerer = _SinkingSteerer(sinks)

    result = await _tool("report_task_started").handler(
        {"task_id": "t1", "detail": "starting"},
        session,
        steerer,
    )

    assert result == {"acknowledged": True}
    # No refusal events.
    assert _refused_events(sinks[0].events) == []


# ---------------------------------------------------------------------------
# 4. Handler: stale pin + REPLACE-supersedes successor → routes to new task.
# ---------------------------------------------------------------------------


async def test_handler_stale_replace_routes_to_successor() -> None:
    """Pre-#266 behaviour preserved for stale REPLACE-kind chains.

    Refine has replaced ``research_solar`` (now FAILED) with
    ``research_solar_v2`` (PENDING, supersedes=research_solar,
    REPLACE-kind). The agent's pin was set under revision 1; the
    plan is at revision 2 now. Handler should route the call onto
    ``research_solar_v2`` and proceed.
    """
    plan = _plan_with(
        Task(
            id="research_solar",
            title="solar v1",
            assignee_agent_id="researcher",
            status=TaskStatus.FAILED,
        ),
        Task(
            id="research_solar_v2",
            title="solar v2",
            assignee_agent_id="researcher",
            status=TaskStatus.PENDING,
            supersedes="research_solar",
            supersedes_kind=SupersessionKind.REPLACE,
        ),
        revision_index=2,
    )
    session = _session_with(plan)
    session.state["goldfive.current_task_id"] = "research_solar"
    session.state["goldfive.current_task_revision"] = 1  # stale

    sinks = [_CapturingSink()]
    steerer = _SinkingSteerer(sinks)

    # Capture the task_id the steerer sees.
    seen: dict[str, str] = {}

    async def _capture(task_id: str, **kw: Any) -> None:
        seen["task_id"] = task_id

    steerer.mark_task_running = _capture  # type: ignore[assignment]

    result = await _tool("report_task_started").handler(
        {"task_id": "research_solar"},
        session,
        steerer,
    )

    assert result == {"acknowledged": True}
    assert seen["task_id"] == "research_solar_v2", (
        "stale REPLACE pin should route to the REPLACE-kind successor"
    )
    assert _refused_events(sinks[0].events) == []


# ---------------------------------------------------------------------------
# 5. Handler: stale pin + CORRECT-supersedes successor → refuses.
# ---------------------------------------------------------------------------


async def test_handler_stale_correct_refuses_and_emits_event() -> None:
    """CORRECT-kind successor → refuse + sink event + ack-only response.

    The old task's terminal state is historical fact; the correction is
    a separate work unit. The LLM still sees ``acknowledged: True`` —
    surfacing the refusal as an error would create a prompt-injection
    surface. Operators see the refusal via ``task_transition_refused``.
    """
    plan = _plan_with(
        Task(
            id="research_solar",
            title="solar (history)",
            assignee_agent_id="researcher",
            status=TaskStatus.COMPLETED,
        ),
        Task(
            id="research_solar_corrected",
            title="solar with correction",
            assignee_agent_id="researcher",
            status=TaskStatus.PENDING,
            supersedes="research_solar",
            supersedes_kind=SupersessionKind.CORRECT,
        ),
        revision_index=2,
    )
    session = _session_with(plan)
    session.state["goldfive.current_task_id"] = "research_solar"
    session.state["goldfive.current_task_revision"] = 1  # stale

    sinks = [_CapturingSink()]
    steerer = _SinkingSteerer(sinks)

    seen_steerer_task_ids: list[str] = []

    async def _capture(task_id: str, **kw: Any) -> None:
        seen_steerer_task_ids.append(task_id)

    steerer.mark_task_completed = _capture  # type: ignore[assignment]

    result = await _tool("report_task_completed").handler(
        {"task_id": "research_solar", "summary": "done"},
        session,
        steerer,
    )

    # LLM sees an ack — no prompt-injection surface.
    assert result == {"acknowledged": True}
    # Steerer was never driven (the call refused).
    assert seen_steerer_task_ids == [], (
        "stale CORRECT pin must not transition the old task; "
        "its terminal state is historical fact"
    )
    # Operator sees the refusal as a typed proto sink event.
    refused = _refused_events(sinks[0].events)
    assert len(refused) == 1
    payload = refused[0].task_transition_refused
    assert payload.task_id == "research_solar"
    assert payload.reason == "stale_pin_correct_supersedes"
    assert payload.pin_revision == 1
    assert payload.current_revision == 2
    assert payload.attempted_to == TaskStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# 6. Handler: stale pin + no supersedes → refuses (ambiguity).
# ---------------------------------------------------------------------------


async def test_handler_stale_no_supersedes_refuses() -> None:
    plan = _plan_with(
        Task(
            id="orphan",
            title="orphaned by refine",
            assignee_agent_id="researcher",
            status=TaskStatus.RUNNING,
        ),
        revision_index=3,
    )
    session = _session_with(plan)
    session.state["goldfive.current_task_id"] = "orphan"
    session.state["goldfive.current_task_revision"] = 1  # stale, no successor

    sinks = [_CapturingSink()]
    steerer = _SinkingSteerer(sinks)

    seen_steerer_task_ids: list[str] = []

    async def _capture(task_id: str, **kw: Any) -> None:
        seen_steerer_task_ids.append(task_id)

    steerer.mark_task_progress = _capture  # type: ignore[assignment]

    result = await _tool("report_task_progress").handler(
        {"task_id": "orphan", "fraction": 0.5},
        session,
        steerer,
    )

    assert result == {"acknowledged": True}
    assert seen_steerer_task_ids == [], (
        "stale ambiguous pin must not drive the steerer"
    )
    refused = _refused_events(sinks[0].events)
    assert len(refused) == 1
    assert refused[0].task_transition_refused.reason == "stale_pin_no_supersedes"


# ---------------------------------------------------------------------------
# 7. Pending-delegation back-compat: legacy string entries read as revision 0.
# ---------------------------------------------------------------------------


def test_pending_delegations_back_compat_string_shape() -> None:
    """Legacy ``{fc_id: task_id_str}`` entries still read cleanly."""
    # Bare-string entry — the pre-#266 shape custom adapters may still
    # write. Both the readers and the back-compat extractors must
    # accept it without raising.
    raw_str = "research_solar"
    assert _delegation_pin_task_id(raw_str) == "research_solar"
    assert _delegation_pin_revision(raw_str) == 0

    # Dict entry — the new shape.
    raw_dict = {"task_id": "t2", "revision": 4}
    assert _delegation_pin_task_id(raw_dict) == "t2"
    assert _delegation_pin_revision(raw_dict) == 4

    # Malformed values fall through cleanly (don't raise).
    assert _delegation_pin_task_id(None) == ""
    assert _delegation_pin_revision({"task_id": "x"}) == 0  # missing rev

    # _resolve_pinned_task_id resolves both shapes.
    state_legacy: dict[str, Any] = {
        "goldfive.pending_delegations": {"fc_X": "task_legacy"},
    }
    state_new: dict[str, Any] = {
        "goldfive.pending_delegations": {
            "fc_X": {"task_id": "task_new", "revision": 7}
        },
    }
    tool_ctx_legacy = _StateCtx._ToolCtx(state_legacy, fc_id="fc_X")
    tool_ctx_new = _StateCtx._ToolCtx(state_new, fc_id="fc_X")
    assert _resolve_pinned_task_id(tool_context=tool_ctx_legacy) == "task_legacy"
    assert _resolve_pinned_task_id(tool_context=tool_ctx_new) == "task_new"


# ---------------------------------------------------------------------------
# 8. _wait_plan_stable: handler waits for concurrent refine to land.
# ---------------------------------------------------------------------------


async def test_handler_waits_for_plan_stable_during_concurrent_refine() -> None:
    """The handler invokes ``steerer._wait_plan_stable`` before classifying.

    Simulates a refine in flight: the steerer's barrier delays the
    handler until the refine has bumped ``revision_index``. Without
    the barrier the handler would read pre-revision state and
    misclassify a fresh pin as stale (or vice versa).
    """
    plan = _plan_with(
        Task(id="t1", title="task one", assignee_agent_id="researcher"),
        revision_index=0,
    )
    session = _session_with(plan)
    session.state["goldfive.current_task_id"] = "t1"
    session.state["goldfive.current_task_revision"] = 0

    sinks = [_CapturingSink()]

    async def _bump_during_wait(_session: Any) -> None:
        # Simulate the refine landing while the handler is parked at
        # the barrier — increments revision_index in place.
        plan.revision_index = 1

    steerer = _SinkingSteerer(sinks, on_wait=_bump_during_wait)

    result = await _tool("report_task_progress").handler(
        {"task_id": "t1", "fraction": 0.4},
        session,
        steerer,
    )

    assert steerer._wait_calls >= 1, (
        "handler must consult ``_wait_plan_stable`` before classifying"
    )
    # After the simulated refine landed mid-wait, the handler reads
    # plan.revision_index=1 and pin_revision=0 → stale. The pin task
    # has no successor → refuse. The LLM still sees ack-only.
    assert result == {"acknowledged": True}
    refused = _refused_events(sinks[0].events)
    assert len(refused) == 1
    # The refusal observed the post-refine revision.
    assert refused[0].task_transition_refused.current_revision == 1
    assert refused[0].task_transition_refused.pin_revision == 0


# ---------------------------------------------------------------------------
# Bonus — fresh pin on a terminal task with REPLACE successor still routes.
# ---------------------------------------------------------------------------


async def test_handler_fresh_pin_routes_through_replace_chain() -> None:
    """Regression: fresh pin + terminal task + REPLACE successor still routes.

    The classifier returns ``"match"`` and falls through to
    ``_reroute_if_superseded`` — a fresh stamp must NOT regress the
    pre-#266 retry-with-replaced-id path.
    """
    plan = _plan_with(
        Task(
            id="t_old",
            title="old",
            assignee_agent_id="researcher",
            status=TaskStatus.FAILED,
        ),
        Task(
            id="t_new",
            title="replacement",
            assignee_agent_id="researcher",
            status=TaskStatus.PENDING,
            supersedes="t_old",
            supersedes_kind=SupersessionKind.REPLACE,
        ),
        revision_index=2,
    )
    session = _session_with(plan)
    # Fresh pin (revision matches current).
    session.state["goldfive.current_task_id"] = "t_old"
    session.state["goldfive.current_task_revision"] = 2

    sinks = [_CapturingSink()]
    steerer = _SinkingSteerer(sinks)

    seen: dict[str, str] = {}

    async def _capture(task_id: str, **kw: Any) -> None:
        seen["task_id"] = task_id

    steerer.mark_task_running = _capture  # type: ignore[assignment]

    result = await _tool("report_task_started").handler(
        {"task_id": "t_old"},
        session,
        steerer,
    )
    assert result == {"acknowledged": True}
    assert seen["task_id"] == "t_new"
    assert _refused_events(sinks[0].events) == []
