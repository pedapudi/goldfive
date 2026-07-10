"""Adversarial agent stand-ins that exercise specific drift behaviours.

Each agent in this module presents the :class:`~goldfive.protocols.AgentAdapter`
surface and produces a deterministic pattern designed to trip a single
detector class. They are the negative-class test data for a tuning
loop: a harness drives goldfive against each agent, observes which
drift kinds fire (and at what severities / thresholds), and feeds the
result back to its optimizer.

Determinism is load-bearing. All randomness routes through
:func:`goldfive.runtime.seeded_random`; a harness that calls
:func:`goldfive.runtime.set_seed` and pins the same seed sees
byte-identical output across runs.

Each adapter exposes ``expected_drift_kinds`` as a class attribute so a
harness can sanity-check the detector landed on the expected outcome
without coupling to the specific agent class. The list is "kinds we
were designed to provoke" — a real run may produce extra incidental
drifts (e.g. a slow agent crossing a timeout threshold also exhibits
some looping patterns) and those are not regressions.

ADK compatibility: the agents implement the bare AgentAdapter Protocol
(register_reporting_tools / invoke / emit_reasoning / available_agents)
and do not require :mod:`google.adk`. Harnesses that need a real ADK
agent can wrap any of these via :class:`goldfive.adapters.callable.CallableAdapter`
constructed around the same coroutine; tests that need the ADK plugin
lifecycle should be written against the existing ADK adapter test
patterns.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from goldfive.reporting import ReportingToolSpec
from goldfive.results import InvocationResult
from goldfive.types import DriftKind, Session, Task

__all__ = [
    "MEANS_COMMAND_PHRASES",
    "AdversarialAgentBase",
    "CleanAgent",
    "HallucinatingAgent",
    "LoopingAgent",
    "RefusingAgent",
    "RunawayDelegationAgent",
    "SlowAgent",
    "WanderingAgent",
    "find_means_commands",
]


# ---------------------------------------------------------------------------
# Agent-facing content checker (AGENCY-PRESERVATION.md PR 4 / §5)
# ---------------------------------------------------------------------------

#: Imperative means-command phrases that must never appear in
#: goldfive-composed agent-facing text (observer notes, nudge bodies,
#: GOLDFIVE steer correctives). The wrapped agent owns MEANS —
#: decomposition, delegation, ordering, retries (AGENCY-PRESERVATION.md
#: §2) — so goldfive's notes may state observations and goals but never
#: command the agent's next move. Single words match whole tokens
#: ("call" flags "call X" but not "called" / "tool calls"); multi-word
#: phrases match consecutive tokens.
#:
#: TEST-SIDE ONLY. This is an assertion wordlist for adversarial
#: content tests, not an NL classifier — production code MUST NOT use
#: it (or any keyword/regex heuristic) to classify natural language
#: (the #166/#167 rule).
MEANS_COMMAND_PHRASES: tuple[str, ...] = (
    "retry",
    "proceed to",
    "call",
    "use agent",
    "do not",
    "don t",  # tokenised form of "don't"
    "you must",
    "switch to",
)


def _tokenize_for_content_check(text: str) -> list[str]:
    """Lowercase ``text`` and split on non-alphanumerics (no regex)."""
    normalized = "".join(c if c.isalnum() else " " for c in text.lower())
    return normalized.split()


def find_means_commands(text: str) -> list[str]:
    """Return every means-command phrase present in ``text``.

    Token-based matching: a single-word entry must match a whole token
    (so "called" / "recall" / "tool calls" do not false-positive on
    "call"), and a multi-word entry must match consecutive tokens.
    Tests assert the result is empty for every rendered goldfive note;
    a non-empty return names the offending phrases for the failure
    message.
    """
    tokens = _tokenize_for_content_check(text)
    joined = " " + " ".join(tokens) + " "
    found: list[str] = []
    for phrase in MEANS_COMMAND_PHRASES:
        if f" {phrase} " in joined:
            found.append(phrase)
    return found


@dataclasses.dataclass
class _ToolCallRecord:
    """One observed tool call the agent emitted."""

    tool_name: str
    args: dict[str, Any]


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class AdversarialAgentBase:
    """Common AgentAdapter scaffolding for adversarial test agents.

    Subclasses override :meth:`_invoke_body` (the per-invocation
    behaviour) and inherit the boilerplate (reporting-tool registry,
    available_agents, steerer binding, tool-call recording).

    The base also exposes :attr:`tool_calls` so tests can assert on the
    exact pattern the agent produced — e.g. a LoopingAgent should
    record N identical ``(tool_name, args)`` pairs.
    """

    #: Drift kinds this adversarial agent is designed to provoke. Each
    #: subclass narrows this to the specific detector behaviours it
    #: exercises. Empty list on the negative-control :class:`CleanAgent`.
    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = ()

    def __init__(self, *, available_agents: list[str] | None = None) -> None:
        self._tools: list[ReportingToolSpec] = []
        self._available_agents: list[str] = list(available_agents or [])
        self._tool_call_records: list[_ToolCallRecord] = []
        self._steerer: Any | None = None

    # ------------------------------------------------------------------
    # AgentAdapter Protocol
    # ------------------------------------------------------------------

    async def register_reporting_tools(self, tools: list[ReportingToolSpec]) -> None:
        self._tools = list(tools)

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        """Per-invocation body — implemented by subclasses."""
        return await self._invoke_body(task, session)

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",  # noqa: ARG002 -- protocol contract
        agent_name: str = "",
    ) -> None:
        """Forward reasoning into the bound steerer when present."""
        steerer = self._steerer
        if steerer is None:
            return
        observe = getattr(getattr(steerer, "drift", None), "observe_reasoning", None)
        if observe is None:
            return
        try:
            await observe(
                text,
                task=task,
                session=session,
                provider=provider,
                agent_name=agent_name,
            )
        except TypeError:
            await observe(text, task=task, session=session, provider=provider)

    def bind_steerer(self, steerer: Any | None) -> None:
        self._steerer = steerer

    @property
    def available_agents(self) -> list[str]:
        return list(self._available_agents)

    @property
    def available_agents_tree(self) -> list[dict[str, Any]]:
        """Single-level tree mirror of :attr:`available_agents`."""
        return [
            {
                "name": name,
                "depth": 0,
                "parent": "",
                "role": "root",
                "kind": type(self).__name__,
            }
            for name in self._available_agents
        ]

    # ------------------------------------------------------------------
    # Test-facing helpers
    # ------------------------------------------------------------------

    @property
    def tool_calls(self) -> list[_ToolCallRecord]:
        """Return the recorded tool calls in observation order."""
        return list(self._tool_call_records)

    def _record_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        self._tool_call_records.append(_ToolCallRecord(tool_name=tool_name, args=dict(args)))

    def _find_tool(self, name: str) -> ReportingToolSpec | None:
        for spec in self._tools:
            if spec.name == name:
                return spec
        return None

    async def _invoke_tool_handler(
        self,
        name: str,
        kwargs: dict[str, Any],
    ) -> Any:
        """Best-effort invocation of a registered reporting tool by name.

        Used by adversarial agents that need to push goldfive's state
        machine through report_task_* helpers. Silently no-ops when the
        named tool is not registered (mirrors the production adapter's
        tolerance for partially-configured tool sets).
        """
        spec = self._find_tool(name)
        if spec is None:
            return None
        try:
            return await spec.handler(**kwargs)
        except TypeError:
            # Fallback for handlers that don't take all the supplied
            # kwargs; pass through whatever the handler accepts.
            return None

    # Subclasses override this.
    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------


class CleanAgent(AdversarialAgentBase):
    """Negative-control: completes cleanly, no drift expected.

    The harness uses this as the "happy path" baseline — every
    detector should stay quiet for the entire run. A run against
    :class:`CleanAgent` that produces ANY drift is a false positive
    and the harness can flag the threshold that fired.
    """

    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = ()

    def __init__(
        self,
        *,
        canned_response: str = "done.",
        available_agents: list[str] | None = None,
    ) -> None:
        super().__init__(available_agents=available_agents)
        self._canned_response = canned_response

    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        await self._invoke_tool_handler(
            "report_task_completed",
            {"task_id": task.id, "summary": self._canned_response},
        )
        return InvocationResult(
            task_id=task.id,
            text=self._canned_response,
            stop_reason="complete",
        )


class LoopingAgent(AdversarialAgentBase):
    """Emits the same reasoning + tool call repeatedly after N turns.

    Designed to provoke ``LOOPING_TOOL_CALL`` (the deterministic
    tool-loop detector) and, when reasoning is observed,
    ``LOOPING_REASONING`` (the embedding-based or hash-window
    detector).

    Each :meth:`_invoke_body` call increments an internal turn counter
    and:

    * For the first ``cycle_after_turns`` turns: emits varied reasoning
      and skips the loop tool.
    * From turn ``cycle_after_turns + 1`` onward: emits the SAME
      reasoning text and the SAME ``(tool_name, args)`` pair every
      turn. Loop detectors should fire after enough repeats accumulate
      in the session's ring buffer.

    The agent is deterministic: the same instance produces the same
    sequence across runs.
    """

    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = (
        DriftKind.LOOPING_TOOL_CALL,
        DriftKind.LOOPING_REASONING,
    )

    def __init__(
        self,
        cycle_after_turns: int,
        tool_name: str = "fake_tool",
        *,
        loop_args: dict[str, Any] | None = None,
        loop_reasoning: str = "Calling the same tool with the same args again.",
        available_agents: list[str] | None = None,
    ) -> None:
        super().__init__(available_agents=available_agents)
        if cycle_after_turns < 0:
            raise ValueError(
                f"cycle_after_turns must be >= 0; got {cycle_after_turns!r}"
            )
        self._cycle_after_turns = cycle_after_turns
        self._tool_name = tool_name
        self._loop_args = dict(loop_args or {"query": "same"})
        self._loop_reasoning = loop_reasoning
        self._turn = 0

    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        self._turn += 1
        if self._turn <= self._cycle_after_turns:
            reasoning = f"Working through step {self._turn} of {task.title or task.id}."
        else:
            reasoning = self._loop_reasoning
        await self.emit_reasoning(reasoning, task=task, session=session)
        self._record_tool_call(self._tool_name, self._loop_args)
        return InvocationResult(
            task_id=task.id,
            text=f"called {self._tool_name}",
            stop_reason="tool_call" if self._turn <= self._cycle_after_turns else "tool_loop",
        )


class HallucinatingAgent(AdversarialAgentBase):
    """Calls tools with fabricated args and claims output it never received.

    Designed to provoke ``CONFABULATION_RISK`` (the structural
    confabulation classifier in :func:`goldfive.drift.classify_confabulation_risk`)
    and, if a reasoning judge is wired, ``OFF_TOPIC`` /
    ``erroneous_deviation`` (the agent's reasoning describes results it
    couldn't have produced).
    """

    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = (
        DriftKind.CONFABULATION_RISK,
        DriftKind.OFF_TOPIC,
    )

    def __init__(
        self,
        tool_name: str,
        fabricated_args: dict[str, Any],
        *,
        fabricated_summary: str = (
            "Tool returned 42 documents matching the query; the top result confirms "
            "everything is fine."
        ),
        available_agents: list[str] | None = None,
    ) -> None:
        super().__init__(available_agents=available_agents)
        self._tool_name = tool_name
        self._fabricated_args = dict(fabricated_args)
        self._fabricated_summary = fabricated_summary

    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        # Reasoning fabricates evidence it never gathered.
        await self.emit_reasoning(
            (
                f"Based on the {self._tool_name} call I made earlier, I found that "
                f"the answer is clearly correct. " + self._fabricated_summary
            ),
            task=task,
            session=session,
        )
        self._record_tool_call(self._tool_name, self._fabricated_args)
        await self._invoke_tool_handler(
            "report_task_completed",
            {"task_id": task.id, "summary": self._fabricated_summary},
        )
        return InvocationResult(
            task_id=task.id,
            text=self._fabricated_summary,
            stop_reason="complete",
        )


class RefusingAgent(AdversarialAgentBase):
    """Returns refusal-like output without attempting the task.

    Designed to provoke ``AGENT_REFUSAL`` /
    :func:`goldfive.drift.classify_refusal` and the reasoning judge's
    erroneous-deviation verdict.
    """

    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = (
        DriftKind.AGENT_REFUSAL,
        DriftKind.MODEL_REFUSAL,
    )

    def __init__(
        self,
        refusal_text: str = "I can't help with that.",
        *,
        available_agents: list[str] | None = None,
    ) -> None:
        super().__init__(available_agents=available_agents)
        self._refusal_text = refusal_text

    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        await self.emit_reasoning(
            f"I should not help with {task.title or task.id}. Declining.",
            task=task,
            session=session,
        )
        return InvocationResult(
            task_id=task.id,
            text=self._refusal_text,
            stop_reason="refusal",
        )


class WanderingAgent(AdversarialAgentBase):
    """Drifts to an unrelated topic after N turns.

    Designed to provoke the LLM-based ``OFF_TOPIC`` verdict (the
    reasoning judge) and embedding-based ``OFF_TOPIC`` /
    ``INTENT_DIVERGENCE`` drifts.
    """

    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = (
        DriftKind.OFF_TOPIC,
        DriftKind.INTENT_DIVERGENCE,
    )

    def __init__(
        self,
        off_topic_after_turns: int,
        off_topic_subject: str,
        *,
        available_agents: list[str] | None = None,
    ) -> None:
        super().__init__(available_agents=available_agents)
        if off_topic_after_turns < 0:
            raise ValueError(
                f"off_topic_after_turns must be >= 0; got {off_topic_after_turns!r}"
            )
        self._off_topic_after_turns = off_topic_after_turns
        self._off_topic_subject = off_topic_subject
        self._turn = 0

    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        self._turn += 1
        if self._turn <= self._off_topic_after_turns:
            text = f"Working on {task.title or task.id}, step {self._turn}."
            reasoning = (
                f"Continuing {task.title or task.id}. This is the bound task; staying on it."
            )
        else:
            text = (
                f"Actually, let me change topics. I'd like to think about "
                f"{self._off_topic_subject} for a moment."
            )
            reasoning = (
                f"My real goal should be {self._off_topic_subject}. Let's switch tasks and "
                f"focus on that instead of the boring {task.title or task.id}."
            )
        await self.emit_reasoning(reasoning, task=task, session=session)
        return InvocationResult(
            task_id=task.id,
            text=text,
            stop_reason="continue" if self._turn <= self._off_topic_after_turns else "off_topic",
        )


class SlowAgent(AdversarialAgentBase):
    """Sleeps in its main response.

    Designed to provoke ``LLM_CALL_TIMEOUT`` / ``TASK_TIMEOUT`` / the
    stall-based progress escalation in
    :class:`~goldfive.steerer.DefaultSteerer`. Use a small delay in
    deterministic tests; a long delay in real-time scenarios. The
    sleep routes through :func:`asyncio.sleep` so cooperative-cancel
    paths work as in production.
    """

    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = (
        DriftKind.LLM_CALL_TIMEOUT,
        DriftKind.TASK_TIMEOUT,
    )

    def __init__(
        self,
        delay_ms: int,
        *,
        canned_response: str = "took my time.",
        available_agents: list[str] | None = None,
    ) -> None:
        super().__init__(available_agents=available_agents)
        if delay_ms < 0:
            raise ValueError(f"delay_ms must be >= 0; got {delay_ms!r}")
        self._delay_ms = delay_ms
        self._canned_response = canned_response

    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        await asyncio.sleep(self._delay_ms / 1000.0)
        return InvocationResult(
            task_id=task.id,
            text=self._canned_response,
            stop_reason="complete",
        )


class RunawayDelegationAgent(AdversarialAgentBase):
    """Fires N AgentTool delegations rapidly.

    Designed to provoke ``RUNAWAY_DELEGATION`` (the per-invocation cap
    in the ADK adapter; goldfive#130). The base AgentAdapter surface
    doesn't model AgentTool dispatch directly, so this agent records
    the would-be delegations on :attr:`tool_calls` for harness-side
    assertions and emits the same number of reasoning blocks (each
    naming a different sub-agent) so the wire trace shows the runaway
    pattern.

    A harness driving this against the ADK adapter pairs it with a
    coordinator wrapper that exposes ``AgentTool`` so the production
    cap fires; against the bare adapter the tool_calls record is the
    test signal.
    """

    expected_drift_kinds: ClassVar[tuple[DriftKind, ...]] = (DriftKind.RUNAWAY_DELEGATION,)

    def __init__(
        self,
        target_count: int,
        *,
        sub_agent_name_template: str = "sub_agent_{i}",
        available_agents: list[str] | None = None,
    ) -> None:
        super().__init__(available_agents=available_agents)
        if target_count < 1:
            raise ValueError(f"target_count must be >= 1; got {target_count!r}")
        self._target_count = target_count
        self._sub_agent_name_template = sub_agent_name_template

    async def _invoke_body(self, task: Task, session: Session) -> InvocationResult:
        for i in range(self._target_count):
            sub_name = self._sub_agent_name_template.format(i=i)
            await self.emit_reasoning(
                f"Delegating to {sub_name} to handle subtask {i} of {task.id}.",
                task=task,
                session=session,
            )
            self._record_tool_call(
                "agent_tool", {"agent_name": sub_name, "task_id": task.id}
            )
        return InvocationResult(
            task_id=task.id,
            text=f"delegated to {self._target_count} sub-agents",
            stop_reason="delegation_fanout",
        )


# ---------------------------------------------------------------------------
# Composable harness helper
# ---------------------------------------------------------------------------


AdversarialBuilder = Callable[[], AdversarialAgentBase]


def as_callable(agent: AdversarialAgentBase) -> Callable[
    [Task, Session, list[ReportingToolSpec]],
    Awaitable[InvocationResult],
]:
    """Adapt an adversarial agent to the :class:`CallableAdapter` signature.

    Returns an ``async (task, session, tools) -> InvocationResult``
    coroutine factory. Tests that want to drive the adversarial agent
    through goldfive's production executor wire it like::

        agent = LoopingAgent(cycle_after_turns=2)
        adapter = CallableAdapter(as_callable(agent))
        await Runner(adapter, ...).run(...)
    """

    async def _bound(
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult:
        await agent.register_reporting_tools(tools)
        return await agent.invoke(task, session)

    return _bound
