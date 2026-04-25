"""Hidden-from-LLM ``task_id`` schema + delegation-site pinning (goldfive#241).

Pre-#241 the ``report_task_*`` tool schemas exposed ``task_id`` as
an optional parameter. Live evidence showed LLMs confused by the
optional-ness ("the function is still trying to use a task_id
parameter even though the schema says it doesn't require any
parameters") and abandoning the reporting protocol entirely.

#241 drops ``task_id`` from the LLM-visible schema. The plugin's
``before_tool_callback`` resolves the pin from session state and
either:

  * injects it into ``tool_args`` before the handler runs, or
  * short-circuits the dispatch with a bare ``{"acknowledged": True}``
    when neither the delegation-site pin nor the agent-turn pin
    resolves. When the current agent has PENDING/RUNNING candidates
    in the plan (so the pin SHOULD have worked), a ``DriftDetected``
    sink event also fires for operator observability — the tool
    response is still the bare ack so the LLM can't pattern-match
    on an error payload. See the #252 follow-up for the live
    prompt-injection evidence that motivated the silent-ack shape.

For coordinators that fire parallel AgentTool calls to the same
sub-agent on a single turn, the agent-turn pin is not unique
enough (all N parallel dispatches share the same agent_name). The
plugin now stamps per-``function_call_id`` pins at the delegation
site (``goldfive.pending_delegations``), resolved against
``tool_args`` keyword overlap when there are multiple candidates.

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    _PENDING_DELEGATIONS_KEY,
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    _score_candidates_by_args,
    _tokenize_for_matching,
    make_adk_plugin,
)
from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID  # noqa: E402
from goldfive.adapters.adk import (  # noqa: E402
    _apply_llm_signature,
    _build_ack_shim,
    _build_function_tool,
    _reporting_tool_signatures,
)
from goldfive.reporting import ReportingToolSpec  # noqa: E402
from goldfive.types import (  # noqa: E402
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers reused across tests
# ---------------------------------------------------------------------------


class _Ctx:
    """Minimal ADK-callback-context stub with a mutable state dict.

    Mirrors the fixture in tests/test_before_tool_task_id_injection.py
    so the plugin's ``_resolve_ctx`` → ``_session_state_from_callback``
    paths see the state we wire.
    """

    class _Session:
        def __init__(self, state: dict) -> None:
            self.state = state

    def __init__(self, state: dict, function_call_id: str = "") -> None:
        self._state = state
        self.function_call_id = function_call_id

    @property
    def session(self) -> Any:
        return _Ctx._Session(self._state)


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _AgentToolStub:
    """Stubbed AgentTool — duck-types as ``(agent=..., name=...)``."""

    def __init__(self, agent_name: str, tool_name: str = "") -> None:
        class _Agent:
            pass

        a = _Agent()
        a.name = agent_name
        self.agent = a
        self.name = tool_name or f"{agent_name}_tool"


class _SinkCapture:
    """Minimal sink stub — collects every event so tests can assert shape.

    ``goldfive.events.emit`` calls ``sink.emit(event_pb)``; the method
    name matches the ``ListSink`` pattern used elsewhere in the test
    suite (see ``tests/test_reporting.py``).
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _SinkingSteererStub:
    """Steerer stand-in that exposes ``_sinks`` so the plugin emits drift."""

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = list(sinks)

    async def observe(self, *a: Any, **kw: Any) -> None:  # pragma: no cover
        pass


def _make_plugin_with_handler(
    tool_name: str,
    plan: Plan | None = None,
    *,
    with_sink: bool = False,
):
    """Return (plugin, state_dict, captured, session, sink) wired for a reporting tool.

    When ``with_sink=True`` the returned ``SessionContext.steerer`` exposes a
    ``_sinks`` list so the plugin's ``DriftDetected`` emission path can
    observe a sink and write to it. Default is ``None`` steerer to keep
    back-compat with older call sites.
    """
    captured: list[dict[str, Any]] = []

    async def handler(args: Any, session: Any, steerer: Any) -> dict[str, Any]:
        captured.append(dict(args))
        return {"acknowledged": True, "echo": dict(args)}

    spec = ReportingToolSpec(
        name=tool_name,
        description="",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    plugin = make_adk_plugin(host_agent_name="coordinator")
    session_obj = Session(run_id="run-1", plan=plan)
    task = Task(id="t-ignored", title="x")
    sink: _SinkCapture | None = None
    steerer: Any | None = None
    if with_sink:
        sink = _SinkCapture()
        steerer = _SinkingSteererStub([sink])
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session_obj,
            steerer=steerer,
            task=task,
            tool_handlers={tool_name: handler},
            tools=[spec],
            host_agent_name="coordinator",
        )
    }
    return plugin, state, captured, session_obj, sink


# ---------------------------------------------------------------------------
# Schema visibility (task_id is NOT in the LLM-facing declaration)
# ---------------------------------------------------------------------------


def test_reporting_tool_signatures_omit_task_id() -> None:
    """Every built-in reporting tool's declared signature must not
    mention ``task_id`` — the LLM must never see it."""
    sigs = _reporting_tool_signatures()
    for tool_name, params in sigs.items():
        names = [p.name for p in params]
        assert "task_id" not in names, (
            f"task_id leaked into LLM-facing schema for {tool_name}: {names}"
        )


def test_function_tool_declaration_excludes_task_id() -> None:
    """Round-trip through ADK's FunctionTool — the declared parameters
    the LLM sees must not include ``task_id``."""
    for tool_name in (
        "report_task_started",
        "report_task_progress",
        "report_task_completed",
        "report_task_failed",
        "report_task_blocked",
    ):
        spec = ReportingToolSpec(
            name=tool_name,
            description="doc",
            parameters={"type": "object", "properties": {}},
            handler=lambda *a, **k: {"acknowledged": True},
        )
        ft = _build_function_tool(spec)
        decl = ft._get_declaration()
        # FunctionDeclaration may have None parameters (no args at all)
        # or a Schema with a properties map.
        params = getattr(decl, "parameters", None)
        if params is None:
            continue
        props = getattr(params, "properties", None) or {}
        assert "task_id" not in props, (
            f"task_id leaked into the FunctionDeclaration for {tool_name}: "
            f"{list(props.keys())}"
        )


def test_apply_llm_signature_noop_for_unknown_tool() -> None:
    """Custom tool names fall through to the legacy permissive
    signature — the injection path still covers them."""
    shim = _build_ack_shim("custom_tool", "doc")
    _apply_llm_signature(shim, "custom_tool")
    # No __signature__ attached when the name isn't one of the
    # built-ins.
    assert not hasattr(shim, "__signature__") or shim.__signature__ is None or (
        "task_id" not in [p.name for p in shim.__signature__.parameters.values()]
    )


# ---------------------------------------------------------------------------
# Injection + short-circuit behaviour
# ---------------------------------------------------------------------------


async def test_hidden_task_id_populates_from_state() -> None:
    """LLM calls ``report_task_started()`` with no task_id arg — the
    callback populates from ``goldfive.current_task_id`` on goldfive
    ``Session.state`` before the handler runs."""
    plugin, state, captured, session_obj, _ = _make_plugin_with_handler(
        "report_task_started"
    )
    session_obj.state[KEY_CURRENT_TASK_ID] = "t-pinned"

    args: dict[str, Any] = {"detail": "starting"}
    result = await plugin.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args=args,
        tool_context=_Ctx(state),
    )
    # Tool dispatch reached the handler — the args map was mutated
    # to carry the pinned id.
    assert result == {"acknowledged": True, "echo": {"task_id": "t-pinned", "detail": "starting"}}
    assert captured == [{"task_id": "t-pinned", "detail": "starting"}]


async def test_hidden_task_id_is_noop_when_state_empty() -> None:
    """No pin anywhere + no candidates in plan → bare silent ack.

    This replaces an earlier "fail with no_task_pinned error" contract
    (goldfive#250). The error variant crashed the LLM's reporting
    protocol when goldfive's refine moved an agent's assigned work to
    another agent (e.g. a coordinator whose own tasks were superseded
    into sub-agent tasks). Observed live: the LLM saw the error and
    reasoned "task started report didn't work, let me proceed directly
    with the research_agent call," bypassing the reporting contract
    entirely.

    The response must be EXACTLY ``{"acknowledged": True}`` — no
    ``detail`` / ``no_task_pinned`` keys (goldfive#250 follow-up). Tool
    responses go back to the LLM verbatim and any editorialising string
    is treated as actionable context: live research_agent paraphrased
    a ``detail`` string into its reasoning and proceeded with stale
    pre-refine instructions. Observability is preserved via an INFO
    log from the plugin.
    """
    # No plan attached — the agent has zero candidates, the
    # orchestration-only-turn branch applies.
    plugin, state, captured, _, _ = _make_plugin_with_handler("report_task_started")
    # Deliberately not setting KEY_CURRENT_TASK_ID or pending_delegations.

    args: dict[str, Any] = {"detail": "starting"}
    result = await plugin.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args=args,
        tool_context=_Ctx(state),
    )
    assert result == {"acknowledged": True}
    # Handler not invoked — captured stays empty (nothing to report on).
    assert captured == []


def _ambiguous_plan_for(agent_name: str = "coordinator") -> Plan:
    """Two PENDING tasks both assigned to ``agent_name`` → pin is
    ambiguous (>1 single-match candidates), but the agent DOES have
    work in the plan."""
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t-alpha",
                title="Alpha",
                description="First task",
                assignee_agent_id=agent_name,
                status=TaskStatus.PENDING,
            ),
            Task(
                id="t-beta",
                title="Beta",
                description="Second task",
                assignee_agent_id=agent_name,
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
        summary="Ambiguous",
    )


async def test_pin_unresolved_when_agent_has_candidates() -> None:
    """Agent has PENDING candidates but the pin couldn't resolve
    (ambiguous — >1 match) → the LLM-visible response is a bare
    ``{"acknowledged": True}`` (so the model can't pattern-match on an
    error payload) AND a ``DriftDetected`` sink event fires for operator
    observability. Pre-#252-followup the tool response carried an
    ``error: pin_unresolved`` payload; live research_agent read the
    payload as a reasoning cue and bypassed the reporting contract."""
    plan = _ambiguous_plan_for("coordinator")
    plugin, state, captured, _session, sink = _make_plugin_with_handler(
        "report_task_started", plan=plan, with_sink=True
    )
    # No KEY_CURRENT_TASK_ID / pending_delegations wired — resolution
    # fails despite the agent having candidates.

    args: dict[str, Any] = {"detail": "starting"}
    result = await plugin.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args=args,
        tool_context=_Ctx(state),
    )
    # LLM-visible response: bare ack. No error / detail / pin_unresolved
    # strings that the model could paraphrase into its reasoning.
    assert result == {"acknowledged": True}
    assert captured == []
    # Operator-visible response: a DriftDetected event with a
    # pin_unresolved reason prefix and the candidate task ids in detail.
    assert sink is not None
    assert len(sink.events) == 1, (
        f"expected exactly one sink event, got {len(sink.events)}"
    )
    evt = sink.events[0]
    # Protobuf ``Event`` with a ``drift_detected`` oneof branch.
    drift = evt.drift_detected
    assert drift.detail.startswith("pin_unresolved:"), (
        f"expected pin_unresolved prefix on drift detail, got {drift.detail!r}"
    )
    assert "report_task_started" in drift.detail
    assert "coordinator" in drift.detail
    # Candidate ids from the ambiguous plan must be present so operators
    # can see which tasks were eligible.
    assert "t-alpha" in drift.detail
    assert "t-beta" in drift.detail
    # Current-agent attribution for the UI.
    assert drift.current_agent_id == "coordinator"


def _dag_gated_plan_for(agent_name: str = "coordinator") -> Plan:
    """PENDING task assigned to ``agent_name`` but upstream is NOT
    COMPLETED. The agent has a candidate, but it's DAG-gated — the
    agent's turn shouldn't be happening yet, which is ALSO a stall
    worth surfacing rather than silencing (goldfive#250 follow-up)."""
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t-upstream",
                title="Upstream",
                description="Precursor work",
                assignee_agent_id="other_agent",
                status=TaskStatus.PENDING,  # NOT COMPLETED
            ),
            Task(
                id="t-gated",
                title="Gated",
                description="Downstream work",
                assignee_agent_id=agent_name,
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="t-upstream", to_task_id="t-gated")],
        summary="DAG-gated",
    )


async def test_pin_unresolved_when_candidate_is_dag_gated() -> None:
    """Agent has a PENDING candidate but it's upstream-gated. Even
    though the DAG gate would reject it for the pin, the agent still
    has work in the plan, so the pin failure is a stall worth
    surfacing via a ``DriftDetected`` sink event — but the tool response
    must still be a bare ack so the LLM can't read a ``pin_unresolved``
    error string."""
    plan = _dag_gated_plan_for("coordinator")
    plugin, state, captured, _session, sink = _make_plugin_with_handler(
        "report_task_started", plan=plan, with_sink=True
    )

    args: dict[str, Any] = {"detail": "starting"}
    result = await plugin.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args=args,
        tool_context=_Ctx(state),
    )
    assert result == {"acknowledged": True}
    assert captured == []
    assert sink is not None
    assert len(sink.events) == 1
    drift = sink.events[0].drift_detected
    assert drift.detail.startswith("pin_unresolved:")
    assert "t-gated" in drift.detail


_FORBIDDEN_LLM_KEYS = ("error", "detail", "no_task_pinned", "pin_unresolved")


def _assert_prompt_safe(response: Any) -> None:
    """The LLM-visible response from a pin-fail branch must be a bare
    ``{"acknowledged": True}`` with no editorialising keys or error-shaped
    payloads.

    Tool responses go back to the LLM verbatim. Any ``error``, ``detail``,
    ``no_task_pinned``, or ``pin_unresolved`` string is treated as
    actionable context (observed live twice: #250's ``detail`` leak and
    #252's ``error: pin_unresolved`` payload), and the model will bypass
    the reporting contract rather than retry. The fix is to keep the
    LLM-facing shape inert — operator visibility goes through the sink
    event channel instead.
    """
    assert isinstance(response, dict), f"expected dict response, got {type(response)}"
    for key in _FORBIDDEN_LLM_KEYS:
        assert key not in response, (
            f"pin-fail response leaked {key!r} to the LLM; this is a "
            f"prompt-injection surface. Full payload: {response}"
        )
    # Bare ack shape — no other keys.
    assert response == {"acknowledged": True}


async def test_pin_fail_responses_never_leak_error_keys_to_llm() -> None:
    """Invariant: every pin-fail code path in ``before_tool_callback``
    must return an LLM-visible payload that carries NONE of
    ``{error, detail, no_task_pinned, pin_unresolved}``.

    This is the single-sentence version of the #250 + #252-followup
    bug class: any named-problem string in the tool response becomes a
    prompt-injection surface. Covers both branches:

      * no-candidates (orchestration-only turn),
      * has-candidates (sink-driven drift emission path).
    """
    # Branch 1 — no candidates (no plan).
    plugin_a, state_a, _cap_a, _s_a, _sink_a = _make_plugin_with_handler(
        "report_task_started", plan=None, with_sink=True
    )
    result_a = await plugin_a.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args={"detail": "starting"},
        tool_context=_Ctx(state_a),
    )
    _assert_prompt_safe(result_a)

    # Branch 2 — has candidates, ambiguous pin.
    plan_b = _ambiguous_plan_for("coordinator")
    plugin_b, state_b, _cap_b, _s_b, _sink_b = _make_plugin_with_handler(
        "report_task_started", plan=plan_b, with_sink=True
    )
    result_b = await plugin_b.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args={"detail": "starting"},
        tool_context=_Ctx(state_b),
    )
    _assert_prompt_safe(result_b)

    # Branch 3 — has DAG-gated candidate.
    plan_c = _dag_gated_plan_for("coordinator")
    plugin_c, state_c, _cap_c, _s_c, _sink_c = _make_plugin_with_handler(
        "report_task_started", plan=plan_c, with_sink=True
    )
    result_c = await plugin_c.before_tool_callback(
        tool=_Tool("report_task_started"),
        tool_args={"detail": "starting"},
        tool_context=_Ctx(state_c),
    )
    _assert_prompt_safe(result_c)


# ---------------------------------------------------------------------------
# Delegation-site pinning (Item 3-bis)
# ---------------------------------------------------------------------------


def _parallel_plan() -> Plan:
    """Plan with two PENDING tasks both assigned to ``research_agent``.

    The coordinator fires two parallel AgentTool calls with different
    args (``topic=solar`` vs ``topic=wind``). Each call must resolve
    to the correctly-matching task at the delegation site.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar",
                description="Gather facts on solar power",
                assignee_agent_id="research_agent",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="research_wind",
                title="Research wind",
                description="Gather facts on wind power",
                assignee_agent_id="research_agent",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
        summary="Parallel research",
    )


def _gated_plan() -> Plan:
    """Plan where a PENDING task has an INCOMPLETE upstream predecessor.

    The DAG-aware candidate filter must exclude tasks whose dependency
    isn't COMPLETED yet — grafting onto them would be wrong even if
    the args match.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research",
                title="Research raw",
                description="Gather facts on solar power",
                assignee_agent_id="research_agent",
                status=TaskStatus.PENDING,  # upstream NOT COMPLETED
            ),
            Task(
                id="synthesise",
                title="Synthesise brief on solar",
                description="Compile solar briefing after research",
                assignee_agent_id="research_agent",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="research", to_task_id="synthesise")],
        summary="Gated",
    )


async def test_parallel_agenttool_resolves_by_args() -> None:
    """Two parallel AgentTool calls with distinguishing args each
    resolve to the matching plan task at the delegation site.

    Post-conditions:
      * ``goldfive.pending_delegations[fc_A]`` == ``research_solar``,
      * ``goldfive.pending_delegations[fc_B]`` == ``research_wind``.

    A subsequent ``report_task_completed()`` call inside the solar
    sub-invocation (fc_A) reads the solar pin and attributes the
    report correctly.
    """
    plan = _parallel_plan()
    plugin, state, _captured, session_obj, _sink = _make_plugin_with_handler(
        "report_task_completed", plan=plan
    )

    # Dispatch two parallel AgentTool calls with different args.
    tool_solar = _AgentToolStub("research_agent", tool_name="research_agent_tool")
    tool_wind = _AgentToolStub("research_agent", tool_name="research_agent_tool")

    ctx_solar = _Ctx(state, function_call_id="fc_solar")
    ctx_wind = _Ctx(state, function_call_id="fc_wind")
    await plugin.before_tool_callback(
        tool=tool_solar,
        tool_args={"topic": "solar power basics"},
        tool_context=ctx_solar,
    )
    await plugin.before_tool_callback(
        tool=tool_wind,
        tool_args={"topic": "wind turbines overview"},
        tool_context=ctx_wind,
    )

    pending = session_obj.state.get(_PENDING_DELEGATIONS_KEY) or {}
    # goldfive#266: entries evolved from ``{fc_id: task_id_str}`` to
    # ``{fc_id: {"task_id": str, "revision": int}}``. Normalise via the
    # back-compat extractor so the test asserts the contract, not the
    # storage shape.
    from goldfive.adapters._adk_plugin import _delegation_pin_task_id

    assert _delegation_pin_task_id(pending.get("fc_solar")) == "research_solar", pending
    assert _delegation_pin_task_id(pending.get("fc_wind")) == "research_wind", pending


async def test_parallel_agenttool_falls_through_on_ambiguous_args() -> None:
    """Parallel dispatches whose args don't distinguish candidates
    produce no delegation pin; sub-invocations fall back to the
    agent-turn pin path (which, for an ambiguous coordinator, ALSO
    leaves state unset — no guess)."""
    plan = _parallel_plan()
    plugin, state, _captured, session_obj, _sink = _make_plugin_with_handler(
        "report_task_completed", plan=plan
    )

    tool = _AgentToolStub("research_agent", tool_name="research_agent_tool")
    ctx = _Ctx(state, function_call_id="fc_ambig")
    # Args don't carry distinguishing keywords.
    await plugin.before_tool_callback(
        tool=tool,
        tool_args={"misc": "please proceed"},
        tool_context=ctx,
    )

    pending = session_obj.state.get(_PENDING_DELEGATIONS_KEY) or {}
    assert pending.get("fc_ambig") is None, (
        "Ambiguous args must NOT produce a delegation pin; fall "
        "through to the legacy path."
    )


async def test_dag_aware_candidate_filter() -> None:
    """A PENDING task whose upstream is not COMPLETED is NOT eligible
    to be the target of a delegation."""
    plan = _gated_plan()
    plugin, state, _captured, session_obj, _sink = _make_plugin_with_handler(
        "report_task_completed", plan=plan
    )

    tool = _AgentToolStub("research_agent", tool_name="research_agent_tool")
    ctx = _Ctx(state, function_call_id="fc_gated")
    # Args that would match BOTH tasks' descriptions ("solar").
    await plugin.before_tool_callback(
        tool=tool,
        tool_args={"topic": "solar briefing"},
        tool_context=ctx,
    )

    pending = session_obj.state.get(_PENDING_DELEGATIONS_KEY) or {}
    from goldfive.adapters._adk_plugin import _delegation_pin_task_id

    pinned = _delegation_pin_task_id(pending.get("fc_gated"))
    # Only "research" is upstream-clear (no predecessors); the
    # "synthesise" task is gated on "research". The pin must be on
    # "research" — NOT on "synthesise" even though its description
    # also matches "solar".
    assert pinned == "research", (
        f"DAG-aware filter should have excluded the gated 'synthesise' task; "
        f"got {pinned!r}"
    )


async def test_delegation_pin_beats_agent_turn_pin() -> None:
    """When both a delegation-site pin (via ``pending_delegations``)
    AND an agent-turn pin (``goldfive.current_task_id``) are
    available, the delegation pin wins — it's the more specific
    resolution."""
    plan = _parallel_plan()
    plugin, state, captured, session_obj, _sink = _make_plugin_with_handler(
        "report_task_completed", plan=plan
    )

    # Simulate: coordinator already stamped delegation pin for fc_A on
    # goldfive Session.state. Phase 2.1 (goldfive#271) — V4 readers
    # consult goldfive Session via the SessionContext stash.
    session_obj.state[_PENDING_DELEGATIONS_KEY] = {"fc_A": "research_solar"}
    # Agent-turn pin sits on a different task (stale from an earlier turn).
    session_obj.state[KEY_CURRENT_TASK_ID] = "research_wind"

    args: dict[str, Any] = {"summary": "done"}
    await plugin.before_tool_callback(
        tool=_Tool("report_task_completed"),
        tool_args=args,
        tool_context=_Ctx(state, function_call_id="fc_A"),
    )

    # The delegation-site pin wins — handler sees research_solar.
    assert captured == [{"task_id": "research_solar", "summary": "done"}]


# ---------------------------------------------------------------------------
# Scoring / tokenisation unit tests
# ---------------------------------------------------------------------------


def test_tokenize_for_matching_filters_short_tokens() -> None:
    """Tokens must be lowercase, alphanumeric, length ≥4."""
    toks = _tokenize_for_matching("Research Solar Power (2026 edition)")
    assert "research" in toks
    assert "solar" in toks
    assert "power" in toks
    # Short / noise tokens filtered.
    assert "in" not in toks
    # Year (4 digits) included — still ≥4.
    assert "2026" in toks


def test_score_candidates_by_args_picks_best_match() -> None:
    """Highest-overlap candidate wins; ties return None."""

    class _T:
        def __init__(self, title: str, description: str = "") -> None:
            self.title = title
            self.description = description

    solar = _T("Research solar power", "")
    wind = _T("Research wind turbines", "")
    choice = _score_candidates_by_args(
        [solar, wind],
        {"topic": "solar basics for beginners"},
    )
    assert choice is solar

    # Tie → None (args match neither uniquely).
    choice = _score_candidates_by_args(
        [solar, wind],
        {"note": "please proceed"},  # no matching tokens
    )
    assert choice is None
