"""Regression tests for goldfive#244 reasoning-judge agent-tree awareness.

Empirical motivation (brussels-sprouts e2e session, OFF_TOPIC at 0:26):

* Plan task ``draft_slides`` had ``assignee_agent_id = coordinator_agent``
* The coordinator's job was to delegate to ``web_developer_agent``, a
  KNOWN sub-agent of ``coordinator_agent`` in the wrapped tree.
* The reasoning judge fired OFF_TOPIC with the verdict text "agent
  deviates from the plan by invoking an unlisted web_developer_agent
  instead of proceeding to the draft_slides task" — but
  ``web_developer_agent`` was structurally in the tree as a sub-agent
  the coordinator was allowed to invoke.
* Cascade: false-positive OFF_TOPIC at 0:26 → refine_steer dispatched
  → refine no-op (planner has nothing to fix) → iter-12-204 escalates
  to HUMAN_INTERVENTION_REQUIRED at 0:40.

Root cause: the judge's prompt sees the plan tasks (with their
``assignee_agent_id``) but does NOT see the agent tree's parent/child
relationships. It cannot tell the difference between "agent X is doing
arbitrary off-plan work" and "agent X is delegating to a known sub-
agent Y in pursuit of its assigned task".

Fix (goldfive#244): thread an optional ``available_agents`` keyword
through :func:`classify_reasoning_drift_with_focus`,
:func:`classify_reasoning_drift`, :func:`analyze_reasoning_with_focus`,
:func:`analyze_reasoning`, and the judge dispatch in
:meth:`DefaultSteerer.observe_reasoning`. When provided, the judge
prompt grows two new pieces of context:

1. An AGENT TREE section listing the wrapped agents and the parent →
   sub-agents edges.
2. A clarifying paragraph appended to the system prompt: legitimate
   coordinator → sub-agent delegation is ON-TASK execution, not a
   deviation.

The ``available_agents=None`` default produces the byte-identical
pre-#244 prompt so existing tests, classifications, and external
callers see no behavioural change.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import reasoning_judge as rjudge  # noqa: E402
from goldfive.drift.reasoning import (  # noqa: E402
    analyze_reasoning,
    analyze_reasoning_with_focus,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftKind,
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class NullPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None


def _capturing_call_llm(response: dict[str, Any]):
    """Async ``CallLLM`` stub that records (system, user, model) triples."""
    captured: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        captured.append((system, user, model))
        return json.dumps(response)

    _call_llm.calls = captured  # type: ignore[attr-defined]
    return _call_llm


async def _drain_judges(steerer: DefaultSteerer) -> None:
    pending = list(steerer._background_judges)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _brussels_sprouts_plan() -> Plan:
    """The plan shape that hit the OFF_TOPIC false positive at 0:26.

    ``draft_slides`` is assigned to ``coordinator_agent``; the
    coordinator's actual delegation target is ``web_developer_agent``,
    which is a sub-agent in the wrapped tree.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="draft_slides",
                title="Draft brussels-sprouts slides",
                description="Build a slide deck on brussels sprouts.",
                assignee_agent_id="coordinator_agent",
            ),
        ],
        edges=[],
    )


def _brussels_sprouts_tree() -> list[dict[str, Any]]:
    """Mirror the live ADKAdapter.available_agents_tree shape.

    Tree:
        presentation_orchestrated  (root)
        ├── coordinator_agent
        │   └── web_developer_agent
        ├── research_agent
        └── reviewer_agent
    """
    return [
        {
            "name": "presentation_orchestrated",
            "depth": 0,
            "parent": "",
            "role": "root",
            "kind": "SequentialAgent",
        },
        {
            "name": "coordinator_agent",
            "depth": 1,
            "parent": "presentation_orchestrated",
            "role": "intermediate",
            "kind": "LlmAgent",
        },
        {
            "name": "web_developer_agent",
            "depth": 2,
            "parent": "coordinator_agent",
            "role": "leaf",
            "kind": "LlmAgent",
        },
        {
            "name": "research_agent",
            "depth": 1,
            "parent": "presentation_orchestrated",
            "role": "leaf",
            "kind": "LlmAgent",
        },
        {
            "name": "reviewer_agent",
            "depth": 1,
            "parent": "presentation_orchestrated",
            "role": "leaf",
            "kind": "LlmAgent",
        },
    ]


def _brussels_sprouts_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a brussels-sprouts deck")],
        plan=_brussels_sprouts_plan(),
        current_task_id="draft_slides",
    )


# ---------------------------------------------------------------------------
# format_available_agents_block
# ---------------------------------------------------------------------------


def test_format_available_agents_block_renders_parent_child_edges() -> None:
    """Tree-shape input renders parent → sub-agents lines."""
    rendered = rjudge.format_available_agents_block(_brussels_sprouts_tree())
    assert "coordinator_agent" in rendered
    assert "web_developer_agent" in rendered
    # The "delegates to" suffix names the live sub-agent.
    assert "coordinator_agent" in rendered
    assert "delegates to: web_developer_agent" in rendered
    # Leaves render without a delegates-to suffix. Match on the
    # bullet's own line (starts with ``- research_agent``) so a parent
    # line that happens to mention research_agent in its delegates-to
    # list doesn't fool the assertion.
    research_lines = [
        line for line in rendered.splitlines() if line.startswith("- research_agent")
    ]
    assert research_lines and "delegates to:" not in research_lines[0]


def test_format_available_agents_block_handles_flat_string_list() -> None:
    """Legacy ``list[str]`` registries still render as a flat bullet list."""
    rendered = rjudge.format_available_agents_block(
        ["coordinator_agent", "web_developer_agent", "research_agent"]
    )
    assert "- coordinator_agent" in rendered
    assert "- web_developer_agent" in rendered
    # No edge data → no "delegates to" suffix.
    assert "delegates to:" not in rendered


def test_format_available_agents_block_empty_inputs_render_empty_string() -> None:
    """Falsy / unrecognised inputs short-circuit to ``""``.

    The classifier checks for the empty string and skips both the user
    prompt extension and the system prompt suffix — that's the
    byte-identical pre-#244 path.
    """
    assert rjudge.format_available_agents_block(None) == ""
    assert rjudge.format_available_agents_block([]) == ""
    assert rjudge.format_available_agents_block({}) == ""  # not a list/tuple


def test_format_available_agents_block_truncates_oversized_tree() -> None:
    """A pathologically large tree is truncated with a marker."""
    big_tree = [
        {"name": f"agent_{i}", "depth": 0, "parent": "", "role": "leaf"}
        for i in range(100)
    ]
    rendered = rjudge.format_available_agents_block(big_tree, max_chars=200)
    # Truncation marker present.
    assert "more agent" in rendered
    body_lines = [
        line for line in rendered.splitlines() if not line.startswith("...")
    ]
    # Bounded modulo a small slack for the truncation line.
    assert sum(len(line) for line in body_lines) <= 220


# ---------------------------------------------------------------------------
# classify_reasoning_drift_with_focus — agent_tree threading
# ---------------------------------------------------------------------------


async def test_classifier_default_none_path_renders_byte_identical_prompt() -> None:
    """When ``available_agents`` is ``None`` the prompt is byte-identical to
    the pre-#244 shape.

    Pins the back-compat invariant: callers / tests that don't pass
    ``available_agents`` see no system prompt suffix and no AGENT TREE
    section in the user prompt.
    """
    call_llm = _capturing_call_llm(
        {
            "classification": "on_task",
            "reason": "delegating per plan",
            "focused_task_id": "draft_slides",
            "focus_confidence": 0.9,
        }
    )
    await rjudge.classify_reasoning_drift_with_focus(
        reasoning="delegating slide drafting",
        task=Task(id="draft_slides", title="Draft slides"),
        goals=[Goal(id="g1", summary="Publish a deck")],
        plan=_brussels_sprouts_plan(),
        model="judge-model",
        call_llm=call_llm,
        current_task_id="draft_slides",
        current_agent_id="coordinator_agent",
        # available_agents intentionally omitted — defaults to None.
    )
    assert len(call_llm.calls) == 1
    system_sent, user_sent, _ = call_llm.calls[0]
    # System prompt is unchanged from the pinned constant.
    assert system_sent == rjudge.REASONING_DRIFT_SYSTEM_PROMPT
    # No AGENT TREE section in the user prompt.
    assert "AGENT TREE" not in user_sent
    # And no agent-aware annotation on the plan-tasks summary either —
    # a coordinator_agent / delegates-to suffix would tip its hand.
    assert "delegates to:" not in user_sent
    assert "[assignee=" not in user_sent


async def test_classifier_with_tree_appends_section_and_system_suffix() -> None:
    """When ``available_agents`` is provided the prompt grows two pieces of
    context: an AGENT TREE section and a clarifying system-prompt suffix.

    Pins the wiring: the operator-overridable
    :data:`AGENT_TREE_SYSTEM_PROMPT_SUFFIX` is appended to the system
    prompt; the user prompt carries the rendered tree under an
    "AGENT TREE" header so the LLM can correlate plan task assignees
    with the sub-agents they are allowed to invoke.
    """
    call_llm = _capturing_call_llm(
        {
            "classification": "on_task",
            "reason": "coordinator delegating to its known sub-agent",
            "focused_task_id": "draft_slides",
            "focus_confidence": 0.9,
        }
    )
    await rjudge.classify_reasoning_drift_with_focus(
        reasoning=(
            "I'll hand off the slide drafting to web_developer_agent so "
            "the deck gets built."
        ),
        task=Task(id="draft_slides", title="Draft slides"),
        goals=[Goal(id="g1", summary="Publish a deck")],
        plan=_brussels_sprouts_plan(),
        model="judge-model",
        call_llm=call_llm,
        current_task_id="draft_slides",
        current_agent_id="coordinator_agent",
        available_agents=_brussels_sprouts_tree(),
    )
    assert len(call_llm.calls) == 1
    system_sent, user_sent, _ = call_llm.calls[0]
    # System suffix appended.
    assert system_sent.endswith(rjudge.AGENT_TREE_SYSTEM_PROMPT_SUFFIX)
    assert system_sent.startswith(rjudge.REASONING_DRIFT_SYSTEM_PROMPT)
    # AGENT TREE section in the user prompt names the live edge.
    assert "AGENT TREE" in user_sent
    assert "coordinator_agent" in user_sent
    assert "delegates to: web_developer_agent" in user_sent
    # Plan-tasks summary now carries the assignee + delegates info too,
    # so the LLM sees the correlation between the task's assignee and
    # the sub-agents it may invoke.
    assert "[assignee=coordinator_agent" in user_sent


# ---------------------------------------------------------------------------
# Brussels-sprouts false-positive regression
# ---------------------------------------------------------------------------


async def test_subagent_delegation_with_tree_returns_on_task() -> None:
    """The brussels-sprouts false-positive cannot fire when the tree is provided.

    Reproduces the live OFF_TOPIC at 0:26: coordinator_agent's
    reasoning announces it will delegate to web_developer_agent for the
    ``draft_slides`` task. With the tree threaded into the prompt, the
    judge correctly returns on_task — there is no drift to refine and
    the cascade that ended in HUMAN_INTERVENTION_REQUIRED is broken at
    its root.

    The mock returns ``classification: on_task`` to model the
    post-prompt-improvement LLM behaviour. The point of this test is
    structural — pinning that the wiring delivers the tree to the
    prompt and that the verdict pipeline produces a no-drift result —
    not testing the LLM itself (we cannot run a real LLM in CI).
    """
    call_llm = _capturing_call_llm(
        {
            "classification": "on_task",
            "reason": (
                "coordinator_agent is delegating to its known sub-agent "
                "web_developer_agent per the AGENT TREE — this is normal "
                "delegation, not an off-plan deviation"
            ),
            "focused_task_id": "draft_slides",
            "focus_confidence": 0.95,
        }
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning=(
            "The plan task draft_slides is assigned to me. I'll "
            "delegate to web_developer_agent to actually build the "
            "slide content for the brussels-sprouts deck."
        ),
        task=Task(
            id="draft_slides",
            title="Draft brussels-sprouts slides",
            assignee_agent_id="coordinator_agent",
        ),
        goals=[Goal(id="g1", summary="Publish a brussels-sprouts deck")],
        plan=_brussels_sprouts_plan(),
        model="judge-model",
        call_llm=call_llm,
        current_task_id="draft_slides",
        current_agent_id="coordinator_agent",
        available_agents=_brussels_sprouts_tree(),
    )
    assert verdict.drift is None, (
        "coordinator → web_developer_agent delegation must NOT be flagged "
        "OFF_TOPIC when the AGENT TREE clearly lists web_developer_agent "
        "as a coordinator_agent sub-agent. Brussels-sprouts session at "
        "0:26 was the live regression."
    )
    assert verdict.classification == "on_task"
    # And the prompt actually carried the structural cue (sanity check
    # so a future refactor that drops the AGENT TREE section can't pass
    # this test by accident).
    _, user_sent, _ = call_llm.calls[0]
    assert "AGENT TREE" in user_sent
    assert "delegates to: web_developer_agent" in user_sent


async def test_invocation_of_unknown_agent_still_fires_off_topic() -> None:
    """Complement to the brussels-sprouts case: when the agent invokes
    something NOT in the tree, the judge still emits OFF_TOPIC.

    Confirms the prompt extension does not over-correct — only KNOWN
    sub-agents are blessed by the AGENT TREE section. Invoking
    ``acme_corp_agent`` (not in the tree) is still a deviation.
    """
    call_llm = _capturing_call_llm(
        {
            "classification": "erroneous_deviation",
            "severity": "warning",
            "reason": (
                "agent invoked acme_corp_agent which is not in the "
                "AGENT TREE — this is an off-plan deviation"
            ),
            "focused_task_id": "",
            "focus_confidence": 0.0,
        }
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning=(
            "I'll forget the slide deck and hand off to acme_corp_agent "
            "for some unrelated work."
        ),
        task=Task(
            id="draft_slides",
            title="Draft brussels-sprouts slides",
            assignee_agent_id="coordinator_agent",
        ),
        goals=[Goal(id="g1", summary="Publish a brussels-sprouts deck")],
        plan=_brussels_sprouts_plan(),
        model="judge-model",
        call_llm=call_llm,
        current_task_id="draft_slides",
        current_agent_id="coordinator_agent",
        available_agents=_brussels_sprouts_tree(),
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.classification == "erroneous_deviation"


# ---------------------------------------------------------------------------
# analyze_reasoning_with_focus / analyze_reasoning forward the tree
# ---------------------------------------------------------------------------


async def test_analyze_reasoning_with_focus_forwards_available_agents() -> None:
    """:func:`analyze_reasoning_with_focus` threads ``available_agents`` to
    the judge.

    Pins the wiring layer above the classifier: callers (e.g. the
    steerer's bg judge) can pass the tree and the prompt extension
    fires.
    """
    call_llm = _capturing_call_llm(
        {
            "classification": "on_task",
            "reason": "delegating",
            "focused_task_id": "draft_slides",
            "focus_confidence": 0.9,
        }
    )
    sink = ListSink()
    session = _brussels_sprouts_session()
    await analyze_reasoning_with_focus(
        "Delegating to web_developer_agent for slide rendering.",
        session,
        mode="judge",
        call_llm=call_llm,
        model="judge-model",
        sink=sink,
        agent_name="coordinator_agent",
        available_agents=_brussels_sprouts_tree(),
    )
    assert len(call_llm.calls) == 1
    _, user_sent, _ = call_llm.calls[0]
    assert "AGENT TREE" in user_sent
    assert "delegates to: web_developer_agent" in user_sent


async def test_analyze_reasoning_legacy_forwards_available_agents() -> None:
    """The legacy drift-only :func:`analyze_reasoning` also forwards the tree."""
    call_llm = _capturing_call_llm(
        {
            "classification": "on_task",
            "reason": "delegating",
        }
    )
    sink = ListSink()
    session = _brussels_sprouts_session()
    drift = await analyze_reasoning(
        "Delegating to web_developer_agent.",
        session,
        mode="judge",
        call_llm=call_llm,
        model="judge-model",
        sink=sink,
        agent_name="coordinator_agent",
        available_agents=_brussels_sprouts_tree(),
    )
    assert drift is None
    assert len(call_llm.calls) == 1
    _, user_sent, _ = call_llm.calls[0]
    assert "AGENT TREE" in user_sent


# ---------------------------------------------------------------------------
# DefaultSteerer.observe_reasoning resolves the tree from the adapter
# ---------------------------------------------------------------------------


class _StubAdapterWithTree:
    """Mimics ADKAdapter exposing ``available_agents_tree``.

    The steerer's :meth:`_resolve_available_agents` reads this property
    and forwards it into the judge call. Defining a tiny stub keeps the
    test free of an actual ADK runner.
    """

    @property
    def available_agents_tree(self) -> list[dict[str, Any]]:
        return _brussels_sprouts_tree()

    @property
    def available_agents(self) -> list[str]:
        return [e["name"] for e in _brussels_sprouts_tree()]


async def test_steerer_observe_reasoning_threads_available_agents() -> None:
    """End-to-end: :meth:`DefaultSteerer.observe_reasoning` reads
    ``adapter.available_agents_tree`` and threads it into the judge.

    The judge's user prompt then carries the AGENT TREE section so
    coordinator → sub-agent delegation is recognised as ON-TASK.
    """
    call_llm = _capturing_call_llm(
        {
            "classification": "on_task",
            "reason": (
                "coordinator delegating to a known sub-agent "
                "web_developer_agent per the AGENT TREE"
            ),
            "focused_task_id": "draft_slides",
            "focus_confidence": 0.9,
        }
    )
    sink = ListSink()
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="judge-model",
        reasoning_drift_mode="judge",
        reasoning_drift_rate_limit=1,
    )
    steerer.bind(sinks=[sink], planner=NullPlanner())
    steerer.bind_adapter(_StubAdapterWithTree())
    session = _brussels_sprouts_session()
    await steerer.drift.observe_reasoning(
        (
            "draft_slides is assigned to me. I'll delegate to "
            "web_developer_agent to build the actual slide content."
        ),
        session=session,
        agent_name="coordinator_agent",
    )
    await _drain_judges(steerer)
    # Verify the prompt the judge actually saw carried the AGENT TREE.
    assert len(call_llm.calls) == 1, (
        f"expected exactly one judge call; got {len(call_llm.calls)}"
    )
    _, user_sent, _ = call_llm.calls[0]
    assert "AGENT TREE" in user_sent, (
        "steerer must thread adapter.available_agents_tree into the judge "
        "prompt; a missing AGENT TREE section means the wiring regressed "
        "and brussels-sprouts-style false positives can return."
    )
    assert "delegates to: web_developer_agent" in user_sent


async def test_steerer_observe_reasoning_no_adapter_keeps_default_prompt() -> None:
    """When the steerer has no adapter, the judge prompt stays byte-identical
    to pre-#244.

    Locks in the back-compat invariant from the steerer's vantage point:
    legacy embed callers without an adapter (or with a custom adapter
    exposing neither ``available_agents_tree`` nor ``available_agents``)
    see exactly the same prompt they did before the fix.
    """
    call_llm = _capturing_call_llm(
        {
            "classification": "on_task",
            "reason": "ok",
            "focused_task_id": "draft_slides",
            "focus_confidence": 0.9,
        }
    )
    sink = ListSink()
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="judge-model",
        reasoning_drift_mode="judge",
        reasoning_drift_rate_limit=1,
    )
    # No adapter bound — the steerer's _resolve_available_agents should
    # return None and the judge prompt should look pre-#244.
    steerer.bind(sinks=[sink], planner=NullPlanner())
    session = _brussels_sprouts_session()
    await steerer.drift.observe_reasoning(
        "Drafting slides for the deck.",
        session=session,
        agent_name="coordinator_agent",
    )
    await _drain_judges(steerer)
    assert len(call_llm.calls) == 1
    system_sent, user_sent, _ = call_llm.calls[0]
    assert system_sent == rjudge.REASONING_DRIFT_SYSTEM_PROMPT
    assert "AGENT TREE" not in user_sent
