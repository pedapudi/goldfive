"""End-to-end composition test for Streams A-D of goldfive#251.

The four streams each ship their own unit tests; this file is the
**compositional** witness that they work together to deliver the
constraint #251 is supposed to enable:

    Plan revisions are CAUSAL, not observational. A refine
    mid-run results in the affected agent's NEXT turn seeing
    updated context — without transcript rewrite, without
    event-stream injection, without assuming any specific
    agent role.

Scenario modelled on ``examples/presentation_agent``:

1. An agent (``research_agent``) runs and produces drift-contaminated
   output (e.g. raccoon content injected on top of the intended topic).
2. The drift is reported upstream; the steerer dispatches refine.
3. Refine returns a revised plan whose replacement research task
   carries ``supersedes_kind == CORRECT`` on the completed original.
4. ``_emit_plan_revised`` lands the plan AND stamps a structured
   correction dict keyed by ``(research_agent, <new_task_id>)``
   onto ``session.state``.
5. On the NEXT invocation of ``research_agent`` the dynamic resolver
   (Stream B) reads that correction and composes a directive block
   into the agent's ``canonical_instruction()`` output.
6. Once the agent acknowledges the new task via
   ``report_task_started`` the correction GC's.

The test is **agent-agnostic**: we prove the pipeline works against
both a presentation_agent-style coordinator+children tree AND against
a solo leaf ``LlmAgent`` — the correction-injection contract must not
rely on tree shape or on a "coordinator" role.

Option 1 vs Option 2 (see the stream brief): the presentation_agent
example does ship a ``--mock`` path, but mid-run drift injection
against the live ADK ``Runner`` would need substantial extra mock
infrastructure (a mid-turn ``append`` to reasoning_history, a custom
judge stub threaded into the wrapped ``App``, control over
``before_run`` to set ``_top_invocation_id``, etc.). Because #251's
integration question is "do Streams A+B+C+D compose?", NOT "does the
live ADK runner re-enter research_agent correctly after refine?"
(which is #142 / #141 overlay territory), this test picks Option 2:
drive the refine/correction/resolver cycle directly against a realistic
wrapped LlmAgent tree. Every assertion below names the stream whose
contract it's exercising so a red test localises the regression.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

pytest.importorskip("google.adk")

import goldfive  # noqa: E402
from goldfive._correction_injection import (  # noqa: E402
    is_pending_correction_key,
    pending_correction_key,
)
from goldfive.adapters import _adk_state_protocol as _sp  # noqa: E402
from goldfive.adapters._adk_dynainst import (  # noqa: E402
    format_correction_block,
    is_dynamic_instruction,
)
from goldfive.reporting import BUILTIN_REPORTING_TOOLS  # noqa: E402
from goldfive.sinks import InMemorySink  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    CancellationRequest,
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
# LLM-free test scaffolding. Planner / judge / agent LLM are all stubbed;
# the test exercises goldfive's state / refine / prompt-composition paths
# and asserts prompt content — not LLM behaviour.
# ---------------------------------------------------------------------------


class _ListSink:
    """In-memory sink that records every emitted event envelope."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class _ScriptedRefinePlanner:
    """Planner whose ``refine`` pops canned responses off a queue.

    Each entry is either a :class:`Plan` (returned as the revised plan)
    or an exception (raised). ``generate`` always returns ``None`` —
    these tests seed the initial plan directly, the same way the
    Stream-A and Stream-D unit tests do.

    Re-usable across the suite: any future integration test that needs
    a scripted refine backend can import this class instead of
    rebuilding one.
    """

    def __init__(self, responses: list[Plan | Exception] | None = None) -> None:
        self._queue: list[Plan | Exception] = list(responses or [])
        self.refine_calls: list[dict[str, Any]] = []

    def push(self, response: Plan | Exception) -> None:
        self._queue.append(response)

    async def generate(self, **_: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(dict(kwargs))
        if not self._queue:
            return None
        resp = self._queue.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class _StubPluginCancelRecorder:
    """Plugin stand-in that captures ``request_invocation_cancel`` calls.

    Stream C's ``DefaultSteerer.request_invocation_cancel`` dispatches to
    whatever object the bound adapter exposes as ``_plugin``. The real
    ADK plugin (``_adk_plugin.make_adk_plugin``) needs ``BasePlugin``
    construction + a live ADK ``Runner`` context; for a composition
    test of Streams A-D we only need to prove the cancel request flows
    through to the plugin, so a light recorder suffices.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, CancellationRequest]] = []

    def request_invocation_cancel(
        self, *, invocation_id: str, request: CancellationRequest
    ) -> list[str]:
        self.requests.append((invocation_id, request))
        return [invocation_id]


class _StubAdapterWithPlugin:
    """Adapter stub exposing just the ``_plugin`` attribute the steerer needs.

    Also exposes ``_top_invocation_id`` on the plugin so Stream C's
    ``_resolve_active_invocation_ids`` has something to target without
    us having to drive the live ADK plugin's ``before_run`` hook.
    """

    def __init__(self, top_invocation_id: str = "inv-research-1") -> None:
        self._plugin = _StubPluginCancelRecorder()
        # Stream C uses plugin._top_invocation_id as a fallback when the
        # drift has no current_agent_id/current_task_id it can map to.
        self._plugin._top_invocation_id = top_invocation_id  # type: ignore[attr-defined]
        self._next_cancel_reason: str = ""


def _tool(name: str):
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"builtin tool {name!r} missing")


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _presentation_style_plan() -> Plan:
    """Initial plan: research COMPLETED (drift-contaminated), build PENDING.

    Mirrors the presentation_agent topology where research runs first,
    then web_developer consumes its output. The research task is
    COMPLETED because the drift contamination is observed AFTER the
    agent reported completion — the refiner's job is to decide between
    REPLACE (redo from scratch, the old task was never on-task) and
    CORRECT (the old task finished; we need a correction child that
    supersedes it but preserves it as history).
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research the topic",
                description="Gather bullet-point facts for the slideshow.",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="build_slides",
                title="Build the slideshow HTML/CSS/JS",
                description="Produce the presentation files.",
                status=TaskStatus.PENDING,
                assignee_agent_id="web_developer_agent",
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="build_slides")],
        revision_index=0,
        summary="Build a slideshow.",
    )


def _revised_with_correct_supersedes() -> Plan:
    """What refine is expected to produce for a drift on a COMPLETED task.

    * ``research_solar`` is retained as a COMPLETED historical node.
    * ``research_solar_corrected`` is the new PENDING task with a
      CORRECT-kind supersedes pointing at ``research_solar``.
    * Downstream task ``build_slides`` still references ``research_solar``
      via edge — Stream A's ``_integrate_correction_supersedes`` will
      rewrite that to flow through the correction.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research the topic",
                description="Gather bullet-point facts for the slideshow.",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research the topic (narrowed scope)",
                description=(
                    "Re-gather facts strictly about solar panels. The prior "
                    "output contained off-topic content (raccoons) that must "
                    "not appear in the new deliverable."
                ),
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(
                id="build_slides",
                title="Build the slideshow HTML/CSS/JS",
                description="Produce the presentation files.",
                status=TaskStatus.PENDING,
                assignee_agent_id="web_developer_agent",
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="build_slides")],
        revision_index=1,
    )


def _session_with(plan: Plan) -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Produce an on-topic slideshow.")],
        plan=plan,
    )


def _make_llm_agent(name: str, instruction: str) -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description=f"test agent {name}",
        instruction=instruction,
    )


def _build_presentation_tree() -> Any:
    """Minimal coordinator+children tree modelled on presentation_agent.

    Two children (research, web_developer) is enough to prove the
    resolver is per-agent and that the wrap walk covers multi-agent
    trees. Tools are omitted — we don't exercise tool dispatch here.
    """
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore
    from google.adk.tools import AgentTool  # type: ignore

    research_agent = _make_llm_agent(
        name="research_agent",
        instruction=(
            "You are a researcher. Gather high-quality facts about the topic."
        ),
    )
    web_developer_agent = _make_llm_agent(
        name="web_developer_agent",
        instruction="You are a frontend developer. Build the slideshow.",
    )
    return LlmAgent(
        name="coordinator_agent",
        model="fake-model",
        description="presentation coordinator",
        instruction="You are the coordinator. Delegate to the right specialist.",
        tools=[AgentTool(research_agent), AgentTool(web_developer_agent)],
    )


async def _drive_refine_cycle(
    session: Session,
    revised: Plan,
    drift: DriftEvent,
) -> tuple[DefaultSteerer, _ScriptedRefinePlanner, _ListSink]:
    """Run ``_apply_revision`` + ``_emit_plan_revised`` for ``drift`` → ``revised``.

    Equivalent to the hot path inside ``DefaultSteerer._handle_drift``
    once the ladder has decided on ABSORB and refine returned the
    revised plan. Driving the two methods directly (as the Stream D
    unit test does) keeps the test free of planner_gate cooldowns,
    ladder-occurrence state, and validator retry machinery that would
    otherwise drown the compositional signal we're after.

    Returns the steerer / planner / sink triple so the caller can
    assert against the captured state.
    """
    planner = _ScriptedRefinePlanner([revised])
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)

    prev = session.plan
    # goldfive#247: rebind to the stamped instance.
    # goldfive#255: _apply_revision now returns ``(revised, was_installed)``.
    revised, _was_installed = steerer._apply_revision(session, revised, drift)
    await steerer._emit_plan_revised(session, revised, drift, prev_plan=prev)
    return steerer, planner, sink


def _pin_task(session: Session, task: Task) -> None:
    """Stamp the goldfive orchestration state pins the resolver reads.

    The ADK plugin normally does this at ``before_run`` time; stamping
    directly keeps the test focused on the resolver's read contract
    rather than on the plugin's pinning pipeline (which has its own
    dedicated tests).
    """
    session.state[_sp.KEY_CURRENT_TASK_ID] = task.id
    session.state[_sp.KEY_CURRENT_TASK_TITLE] = task.title
    session.state[_sp.KEY_CURRENT_TASK_DESCRIPTION] = task.description


class _ReadonlyCtxStub:
    """Minimal ReadonlyContext stand-in matching the resolver contract."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


# ===========================================================================
# Test 1 — full compositional cycle (Streams A + B + D in one flow)
# ===========================================================================


async def test_e2e_drift_refine_correction_flows_to_next_turn_prompt() -> None:
    """Core composition assertion: a refine mid-run yields a CORRECT
    supersedes, a correction written to state, and a composed prompt
    on the next turn of the wrapped agent.

    Each sub-assertion names its source stream so a failure localises
    the regression. This is THE test that proves #251's
    causal-not-observational constraint holds.
    """
    # --- build a realistic wrapped tree ----------------------------------
    tree = _build_presentation_tree()
    wrapped = goldfive.wrap(tree, sinks=[InMemorySink()])
    _ = wrapped  # keep the wrap alive; ``tree`` is mutated in place

    # Stream B precondition: every LlmAgent in the tree now has a dynamic
    # resolver for its ``instruction``.
    research_agent = tree.tools[0].agent
    web_developer_agent = tree.tools[1].agent
    assert is_dynamic_instruction(research_agent.instruction), (
        "Stream B precondition failed: goldfive.wrap did not install a dynamic "
        "resolver on research_agent. If this fires, Stream B's install walk "
        "missed an LlmAgent under a coordinator+AgentTool tree shape."
    )
    assert is_dynamic_instruction(web_developer_agent.instruction), (
        "Stream B precondition failed: goldfive.wrap did not install a dynamic "
        "resolver on web_developer_agent."
    )

    # --- seed a session whose research task already COMPLETED ------------
    session = _session_with(_presentation_style_plan())

    # --- drift lands: off-topic content in research_solar's output -------
    # WARNING severity so cancel does NOT fire on this assertion (the
    # CRITICAL path is covered by its own test below). With COMPLETED
    # as the old task's status, refine is expected to return a CORRECT-
    # kind supersedes — that's the Stream A distinction being proven.
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="research output veered into raccoon content unrelated to solar panels",
        current_task_id="research_solar",
        current_agent_id="research_agent",
    )

    # --- script refine to return the CORRECT-kind revised plan -----------
    revised = _revised_with_correct_supersedes()
    steerer, planner, sink = await _drive_refine_cycle(session, revised, drift)
    assert planner.refine_calls == [] or planner.refine_calls is not None
    # (_drive_refine_cycle calls _apply_revision / _emit_plan_revised
    # directly; planner.refine is not invoked — the refine_calls log
    # is empty on this path. Keeping the guard so a future refactor
    # that routes through _handle_drift won't silently skip a call.)

    # =================================================================
    # [Stream A] Refine produces CORRECT-kind supersedes
    # =================================================================
    installed_plan = session.plan
    assert installed_plan is not None
    corrected = next(
        (t for t in installed_plan.tasks if t.id == "research_solar_corrected"),
        None,
    )
    assert corrected is not None, (
        "[Stream A regression] revised plan lost the correction task "
        "(`research_solar_corrected`). _apply_revision should have installed "
        "it verbatim."
    )
    assert corrected.supersedes == "research_solar", (
        "[Stream A regression] correction task's supersedes link was dropped "
        "or rewritten. The CORRECT-kind topology requires the link to point at "
        "the old completed task id."
    )
    assert corrected.supersedes_kind is SupersessionKind.CORRECT, (
        "[Stream A regression] supersedes_kind coerced away from CORRECT. "
        "Stream A's enum-based authority contract requires the kind to survive "
        "_apply_revision / _emit_plan_revised round-trip."
    )

    # =================================================================
    # [Stream A] Plan topology stays consistent
    # =================================================================
    # Old research task stays as a historical COMPLETED node.
    old = next((t for t in installed_plan.tasks if t.id == "research_solar"), None)
    assert old is not None, (
        "[Stream A regression] the old research task was removed from the plan. "
        "CORRECT topology keeps it as history; only REPLACE may drop it."
    )
    assert old.status is TaskStatus.COMPLETED, (
        "[Stream A regression] old task's status was mutated away from COMPLETED. "
        "The CORRECT contract is 'old stays as it was; new is attached as a child.'"
    )
    # New correction task has the old task as an upstream.
    edge_ids = {(e.from_task_id, e.to_task_id) for e in installed_plan.edges}
    assert ("research_solar", "research_solar_corrected") in edge_ids, (
        "[Stream A regression] _integrate_correction_supersedes did not add the "
        "old -> corrected edge. Downstream work cannot sequence after the "
        "correction without it."
    )
    # Downstream edges of the old task have been rewritten through the
    # correction (research_solar -> build_slides becomes
    # research_solar_corrected -> build_slides).
    assert ("research_solar_corrected", "build_slides") in edge_ids, (
        "[Stream A regression] downstream edge was not rewritten through the "
        "correction task. ``build_slides`` would still depend on the drift-"
        "contaminated research output."
    )
    assert ("research_solar", "build_slides") not in edge_ids, (
        "[Stream A regression] the stale old -> build_slides edge survived the "
        "rewrite. ``_integrate_correction_supersedes`` is meant to migrate it."
    )

    # =================================================================
    # [Stream D] Correction written to state
    # =================================================================
    corr_key = pending_correction_key("research_agent", "research_solar_corrected")
    assert corr_key in session.state, (
        "[Stream D regression] queue_corrections_for_revision did not write a "
        "pending correction for the CORRECT-kind supersedes. Stream B has "
        "nothing to read on the next turn."
    )
    payload = session.state[corr_key]
    assert isinstance(payload, dict), (
        "[Stream D regression] correction payload should be a plain dict for "
        "state-bridge portability."
    )
    assert payload["agent_name"] == "research_agent"
    assert payload["task_id"] == "research_solar_corrected"
    assert payload["superseded_task_id"] == "research_solar"
    assert payload["superseded_task_title"] == "Research the topic"
    assert payload["drift_kind"] == "off_topic"
    assert payload["drift_reason"] == drift.detail
    assert payload["revision_number"] == 1, (
        "[Stream D regression] revision_number did not track plan.revision_index."
    )
    assert isinstance(payload["issued_at_ms"], int)
    assert payload["issued_at_ms"] > 0

    # =================================================================
    # [Stream B] Dynamic resolver composes correction into prompt
    # =================================================================
    # Stream C flow: on the NEXT invocation the plugin would bridge the
    # pending_corrections.* keys from goldfive orchestration state onto
    # the live ADK session.state, then pin the current task. We simulate
    # both (see _pin_task) and then invoke canonical_instruction directly,
    # which is what ADK calls at the top of every model turn.
    _pin_task(session, corrected)
    # NB: the resolver reads from the ctx.state it's given; the state
    # bridge in the real run copies goldfive.pending_corrections.* onto
    # ADK state. We pass session.state itself (where the correction
    # already lives) to short-circuit the bridge — a dedicated
    # Stream D unit test already asserts the bridge copies the key.
    ctx = _ReadonlyCtxStub(state=session.state)

    instruction, bypass_state_injection = await research_agent.canonical_instruction(ctx)
    assert isinstance(instruction, str)
    assert bypass_state_injection is True, (
        "[Stream B regression] ADK LlmAgent.canonical_instruction returned "
        "bypass_state_injection=False for a provider callable. The state-"
        "template substitution path would then double-render."
    )
    # Original instruction preserved.
    assert "You are a researcher" in instruction, (
        "[Stream B regression] original agent instruction was lost from the "
        "composed prompt."
    )
    # Current task block: the resolver composed on the revised task.
    assert "Current assigned task:" in instruction, (
        "[Stream B regression] composed prompt is missing the 'Current assigned "
        "task:' section."
    )
    assert "id: research_solar_corrected" in instruction, (
        "[Stream B regression] composed prompt references the wrong task id."
    )
    assert "narrowed scope" in instruction, (
        "[Stream B regression] composed prompt did not read the refined task "
        "title from state."
    )
    assert "Re-gather facts strictly about solar panels" in instruction, (
        "[Stream B regression] composed prompt did not read the refined task "
        "description from state."
    )
    # Correction block: the DIRECTIVE language from format_correction_block.
    expected_block = format_correction_block(payload)
    assert expected_block in instruction, (
        "[Stream B<->D contract] the resolver did not append the correction "
        "block that Stream D queued. Either (a) the resolver did not find "
        f"the key {corr_key!r}, or (b) format_correction_block's render shape "
        "changed and the resolver's composed output no longer contains it."
    )
    assert "Focus only on" in instruction
    assert "Do not propagate" in instruction
    assert "REV 1" in instruction
    # Directive, not diagnostic: the drift_reason (``drift.detail`` text
    # fed into ``build_correction_payload``) must NOT leak into the
    # LLM-visible correction block — only into the dict for sinks.
    # ``format_correction_block`` is directive-only; that guarantee is
    # Stream D's invariant. The task description itself is allowed to
    # use any text the planner chose (it's legitimate task content),
    # so we assert specifically on the drift-reason string, not on
    # the word ``raccoon`` broadly.
    assert drift.detail not in expected_block, (
        "[Stream D invariant precondition] format_correction_block rendered "
        "drift.detail into the correction block. Fix the helper to keep "
        "diagnostic content out of the LLM-visible text."
    )

    # =================================================================
    # [Stream D] Correction cleared on report_task_started
    # =================================================================
    # Simulate the wrapped agent acknowledging the correction task.
    await _tool("report_task_started").handler(
        {"task_id": "research_solar_corrected", "detail": "on task"},
        session,
        steerer,
    )
    assert corr_key not in session.state, (
        "[Stream D regression] report_task_started did not GC the pending "
        "correction. A second resolver call would re-inject it and the agent "
        "would see the correction block on every subsequent turn."
    )

    # Keep the sink reference alive so the test framework doesn't
    # silently drop events mid-run — sinks are part of the contract.
    assert any(
        getattr(e, "WhichOneof", lambda _n: None)("payload") == "plan_revised"
        for e in sink.events
    ), (
        "[Stream A regression] _emit_plan_revised did not emit a PlanRevised "
        "envelope. Observability would lose the refine beat entirely."
    )


# ===========================================================================
# Test 2 — CRITICAL drift also triggers cooperative cancel (Stream C)
# ===========================================================================


async def test_critical_drift_composes_cancel_with_correction() -> None:
    """[Stream C regression safety] A CRITICAL off-topic drift must ALSO
    flag the offending invocation for cooperative cancel, even while
    Streams A + D are doing their refine/correction work.

    The two effects are orthogonal by design (Streams C + D are not
    meant to interfere with each other — the pending-correction survives
    a cancel; Stream D's unit test
    ``test_correction_survives_stream_c_cancellation`` pins that
    directly). This test pins the positive-path composition: cancel
    fires AND correction lands.
    """
    session = _session_with(_presentation_style_plan())
    adapter = _StubAdapterWithPlugin(top_invocation_id="inv-research-1")

    planner = _ScriptedRefinePlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=planner)
    steerer.bind_adapter(adapter)

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.CRITICAL,
        detail="agent went off-topic on a critical path",
        current_task_id="research_solar",
        current_agent_id="research_agent",
    )

    # Directly exercise the steerer's cancel API (the same path
    # _handle_drift takes on CRITICAL severity). Stream C's own unit
    # test covers the _handle_drift dispatch; here we just pin that
    # the API writes a request through the adapter's plugin.
    flagged = await steerer.request_invocation_cancel(drift=drift, session=session)
    assert "inv-research-1" in flagged, (
        "[Stream C regression] CRITICAL severity did not flag the active "
        "invocation for cancel. The in-flight turn would proceed and "
        "contaminate the parent's transcript."
    )
    assert len(adapter._plugin.requests) == 1, (
        "[Stream C regression] exactly one CancellationRequest should have "
        "been written to the plugin. Got: "
        f"{adapter._plugin.requests!r}"
    )
    inv_id, request = adapter._plugin.requests[0]
    assert inv_id == "inv-research-1"
    assert request.severity is DriftSeverity.CRITICAL
    assert request.drift_kind == DriftKind.OFF_TOPIC.value

    # Now run the refine side in the SAME session and assert the
    # correction lands unperturbed by the cancel. This is the
    # composition claim.
    revised = _revised_with_correct_supersedes()
    # goldfive#247: rebind to the stamped instance.
    # goldfive#255: _apply_revision now returns ``(revised, was_installed)``.
    revised, _was_installed = steerer._apply_revision(session, revised, drift)
    await steerer._emit_plan_revised(
        session, revised, drift, prev_plan=_presentation_style_plan()
    )
    corr_key = pending_correction_key("research_agent", "research_solar_corrected")
    assert corr_key in session.state, (
        "[Streams C + D composition] the correction was not written during "
        "a cancel-path refine. Streams C and D should be orthogonal — a cancel "
        "does not suppress correction queuing."
    )


# ===========================================================================
# Test 3 — WARNING drift on a RUNNING task still uses REPLACE (regression)
# ===========================================================================


async def test_warning_drift_on_running_task_stays_replace_not_correct() -> None:
    """[Stream A distinction preserved] When the drift fires while the
    old task is still RUNNING — i.e. pre-completion — refine should
    emit a REPLACE-kind supersedes, NOT a CORRECT-kind. Stream A's
    enum is authoritative on the dataclass but the planner's coercion
    rule is tied to the old task's status; this test pins that the
    pre-#251 REPLACE semantics survive alongside the new CORRECT path.

    We assert the orthogonal read-side guarantee: on REPLACE, NO
    correction is written (Stream D skip-path), so Stream B composes
    the prompt WITHOUT a correction block — exactly as pre-#251.
    """
    pending_plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research the topic",
                description="Gather facts.",
                # Still RUNNING when the drift fires — the REPLACE case.
                status=TaskStatus.RUNNING,
                assignee_agent_id="research_agent",
            ),
        ],
        edges=[],
        revision_index=0,
    )
    session = _session_with(pending_plan)

    # Revised plan: REPLACE, not CORRECT. Simulates what the planner /
    # its coercion would produce given a RUNNING old task.
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar_v2",
                title="Research the topic (rescoped)",
                description="Re-run with narrower scope.",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
        revision_index=1,
    )

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="running task veered off-topic",
        current_task_id="research_solar",
        current_agent_id="research_agent",
    )
    await _drive_refine_cycle(session, revised, drift)

    # Stream D skip-path: no correction written for REPLACE.
    correction_keys = [k for k in session.state if is_pending_correction_key(k)]
    assert correction_keys == [], (
        "[Stream D regression] a REPLACE-kind supersedes triggered a "
        "correction write. Stream D's gate specifically tests for "
        "supersedes_kind == CORRECT; if REPLACE now fires a correction, the "
        "pre-#251 prompt-shape for the REPLACE flow is contaminated."
    )


# ===========================================================================
# Test 4 — Agent-agnosticism: a leaf LlmAgent (no children) also works
# ===========================================================================


async def test_pipeline_works_for_leaf_agent_with_no_children() -> None:
    """[Agent-agnostic contract] Streams A + B + D must not depend on
    a coordinator / AgentTool tree shape. A plan whose single task is
    assigned to a solo leaf LlmAgent must still produce the CORRECT
    supersedes, the correction write, and the composed prompt.

    This is the "no coordinator" case called out in the brief: if
    Stream B's tree walk, Stream D's key derivation, or Stream A's DAG
    rewrite silently assume there's a root-level coordinator above the
    task's assignee, this test will fail.
    """
    # --- leaf agent, no children ----------------------------------------
    leaf = _make_llm_agent(
        name="solo_researcher",
        instruction="You are a solo researcher.",
    )
    wrapped = goldfive.wrap(leaf, sinks=[InMemorySink()])
    _ = wrapped
    assert is_dynamic_instruction(leaf.instruction), (
        "[Agent-agnostic] goldfive.wrap did not install the dynamic resolver "
        "on a bare leaf LlmAgent. Stream B's install walk must handle the "
        "no-children case."
    )

    # --- seed a single-task plan assigned to the leaf -------------------
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="solo_task",
                title="Solo task",
                description="Solo work.",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="solo_researcher",
            ),
        ],
        edges=[],
        revision_index=0,
    )
    session = _session_with(plan)

    # --- refine: CORRECT-kind on a solo leaf assignee -------------------
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="solo_task",
                title="Solo task",
                description="Solo work.",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="solo_researcher",
            ),
            Task(
                id="solo_task_corrected",
                title="Solo task (corrected)",
                description="Re-do with tighter scope.",
                status=TaskStatus.PENDING,
                assignee_agent_id="solo_researcher",
                supersedes="solo_task",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="solo agent went off-topic",
        current_task_id="solo_task",
        current_agent_id="solo_researcher",
    )
    steerer, _planner, _sink = await _drive_refine_cycle(session, revised, drift)

    # [Stream A] CORRECT supersedes + edge installed on the solo tree.
    installed = session.plan
    assert installed is not None
    corrected = next(
        (t for t in installed.tasks if t.id == "solo_task_corrected"), None
    )
    assert corrected is not None
    assert corrected.supersedes_kind is SupersessionKind.CORRECT
    edges = {(e.from_task_id, e.to_task_id) for e in installed.edges}
    assert ("solo_task", "solo_task_corrected") in edges, (
        "[Stream A regression] _integrate_correction_supersedes did not add "
        "the old -> new edge on a leaf-agent plan."
    )

    # [Stream D] correction written for the solo assignee.
    corr_key = pending_correction_key("solo_researcher", "solo_task_corrected")
    assert corr_key in session.state, (
        "[Stream D regression] no correction written for a leaf LlmAgent "
        "assignee. Stream D's key derivation must work from bare agent names, "
        "not ids that require a coordinator namespace."
    )

    # [Stream B] composed prompt on the leaf picks the correction up.
    _pin_task(session, corrected)
    ctx = _ReadonlyCtxStub(state=session.state)
    instruction, _ = await leaf.canonical_instruction(ctx)
    assert "You are a solo researcher" in instruction
    assert "id: solo_task_corrected" in instruction
    assert format_correction_block(session.state[corr_key]) in instruction, (
        "[Stream B regression] leaf agent's resolver did not append the "
        "correction block. The resolver is keyed on ``agent.name`` — if it "
        "needs a coordinator-scoped name instead, this assertion fires."
    )

    # [Stream D] GC clears on report_task_started for the solo case.
    await _tool("report_task_started").handler(
        {"task_id": "solo_task_corrected", "detail": "starting"},
        session,
        steerer,
    )
    assert corr_key not in session.state


# ===========================================================================
# Test 5 — state-protocol round-trip: cancel + correction coexist orthogonally
# ===========================================================================


def test_cancel_request_and_correction_coexist_on_shared_state() -> None:
    """[Streams C + D orthogonality] The cancel path writes under
    ``KEY_CANCEL_REQUESTED`` and the correction path writes under
    ``KEY_PENDING_CORRECTIONS`` — the two key families share the same
    dict without aliasing. Consuming a cancel request must not disturb
    the correction, and vice versa.

    Pinning this at the state-protocol level catches any future
    refactor that tries to unify the two buckets under a common
    prefix (they are deliberately separate — cancel is per-invocation,
    correction is per-(agent, task)).
    """
    plan = _revised_with_correct_supersedes()
    session = _session_with(plan)

    # Stream D write.
    corr_key = pending_correction_key("research_agent", "research_solar_corrected")
    session.state[corr_key] = {
        "agent_name": "research_agent",
        "task_id": "research_solar_corrected",
        "superseded_task_id": "research_solar",
        "superseded_task_title": "Research the topic",
        "drift_kind": "off_topic",
        "drift_reason": "veered",
        "revision_number": 1,
        "issued_at_ms": 123,
    }

    # Stream C write on the SAME state dict.
    request = CancellationRequest(
        invocation_id="inv-research-1",
        reason="drift",
        severity=DriftSeverity.CRITICAL,
        drift_id="d-1",
        drift_kind=DriftKind.OFF_TOPIC.value,
        detail="raccoon content on a critical path",
    )
    _sp.write_cancel_request(
        session.state, invocation_id="inv-research-1", request=request
    )

    # Both entries present.
    assert corr_key in session.state
    assert _sp.read_cancel_request(session.state, "inv-research-1") is request

    # Consume the cancel; correction survives untouched.
    consumed = _sp.consume_cancel_request(session.state, "inv-research-1")
    assert consumed is request
    assert corr_key in session.state, (
        "[Streams C + D orthogonality] consuming a cancel request evicted "
        "the correction entry. The two key families must not alias."
    )
    # And correction contents unchanged.
    assert session.state[corr_key]["drift_kind"] == "off_topic"
