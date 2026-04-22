"""Unit tests for classifier helpers in :mod:`goldfive.drift`.

The reasoning-based detectors live in :mod:`goldfive.drift.reasoning`
and have their own coverage in ``test_drift_reasoning.py``. This module
pins the cheaper structural classifiers:

* :func:`goldfive.drift.classify_confabulation_risk` (issue #128) —
  flags research / verification tasks that finished with zero tool
  calls and non-empty output.
* :mod:`goldfive.drift.tool_loops` (issue #181) -- deterministic
  tool-call-loop detector wired into the ADK plugin's
  ``after_tool_callback``. The unit tests for the classifier live in
  ``tests/test_tool_loops.py``; here we pin the plugin-level
  end-to-end wiring.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.drift import (
    CONFABULATION_TRIGGER_KEYWORDS,
    classify_confabulation_risk,
)
from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Session, Task

# ---------------------------------------------------------------------------
# Keyword-set contract
# ---------------------------------------------------------------------------


def test_confabulation_trigger_keywords_is_tuple_of_strings() -> None:
    """Module constant must be a tuple of non-empty lowercase strings.

    Callers (tests, alternate detectors, future sinks) pin against
    this exact shape. Any regression that turns the set back into a
    list or mutable collection should fail here so we notice before
    downstream code copies the shape.
    """
    assert isinstance(CONFABULATION_TRIGGER_KEYWORDS, tuple)
    assert len(CONFABULATION_TRIGGER_KEYWORDS) > 0
    for kw in CONFABULATION_TRIGGER_KEYWORDS:
        assert isinstance(kw, str) and kw
        # Phrases are matched case-insensitively but stored lower-case
        # so the substring probe in classify_confabulation_risk skips
        # an extra ``.lower()`` per keyword.
        assert kw == kw.lower()


def test_confabulation_trigger_keywords_excludes_generic_verbs() -> None:
    """Conservative keyword set: no generic synthesis verbs.

    False positives are expensive — the drift surfaces on every clean
    run of a task whose description happens to contain one of these
    words. Synthesis verbs that commonly appear on tasks where zero
    tool calls is the EXPECTED shape must stay out.
    """
    forbidden = {"write", "summarize", "format", "draft", "compose", "edit", "create"}
    for kw in CONFABULATION_TRIGGER_KEYWORDS:
        assert kw not in forbidden, (
            f"keyword {kw!r} is a generic synthesis verb — its presence "
            f"would over-fire CONFABULATION_RISK on clean synthesis tasks"
        )


# ---------------------------------------------------------------------------
# classify_confabulation_risk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,keyword",
    [
        ("title", "research"),
        ("title", "verify"),
        ("title", "look up"),
        ("description", "review"),
        ("description", "fetch"),
        ("description", "cross-reference"),
    ],
)
def test_fires_when_keyword_matches_and_zero_tools(field: str, keyword: str) -> None:
    """Matching keyword + zero tool calls + non-empty output → INFO drift."""
    task = Task(
        id="t-research",
        title="Plain title" if field == "description" else f"Please {keyword} the latest docs",
        description=(
            f"Please {keyword} the latest docs" if field == "description" else "Plain description"
        ),
        assignee_agent_id="research_agent",
    )
    drift = classify_confabulation_risk(
        task=task,
        tool_call_count=0,
        output_text="Here is what I found about the topic...",
    )
    assert drift is not None
    assert drift.kind is DriftKind.CONFABULATION_RISK
    assert drift.severity is DriftSeverity.INFO
    assert drift.current_task_id == "t-research"
    assert drift.current_agent_id == "research_agent"
    # The detail should name the keyword that triggered it so operators
    # can see why the drift fired without re-scanning the task text.
    assert keyword in drift.detail


def test_case_insensitive_keyword_match() -> None:
    """Keyword matching must be case-insensitive.

    A task titled ``"RESEARCH the topic"`` or ``"Research"`` must
    trigger the same INFO drift as ``"research"``.
    """
    task = Task(
        id="t1",
        title="RESEARCH the latest papers on topic X",
        description="",
    )
    drift = classify_confabulation_risk(
        task=task,
        tool_call_count=0,
        output_text="According to my findings, ...",
    )
    assert drift is not None
    assert drift.kind is DriftKind.CONFABULATION_RISK


def test_no_fire_when_tools_were_called() -> None:
    """A research task that called tools is exactly the expected shape.

    The whole point of the detector is "research-shaped + zero tools"
    is fishy. One or more tool calls means the agent actually went to
    fetch external data, so no drift.
    """
    task = Task(id="t1", title="Research the latest papers", description="")
    drift = classify_confabulation_risk(
        task=task,
        tool_call_count=1,
        output_text="I found three relevant papers...",
    )
    assert drift is None

    # Many tool calls — still silent.
    drift_many = classify_confabulation_risk(
        task=task,
        tool_call_count=42,
        output_text="I found three relevant papers...",
    )
    assert drift_many is None


def test_no_fire_when_output_empty() -> None:
    """Zero output is not the confabulation pattern.

    An agent that produced nothing hasn't fabricated anything — some
    other drift (STOPPED_EARLY, AGENT_REFUSAL) should cover that case.
    Whitespace-only output is treated as empty.
    """
    task = Task(id="t1", title="Research the docs", description="")
    for empty in ("", "   ", "\n\t  \n"):
        drift = classify_confabulation_risk(
            task=task,
            tool_call_count=0,
            output_text=empty,
        )
        assert drift is None, f"should not fire for empty output {empty!r}"


def test_no_fire_when_no_keyword_match() -> None:
    """Non-research tasks are out of scope for this detector.

    "Format the slides" is pure synthesis — zero tool calls is the
    expected shape. The detector must stay silent so operators aren't
    spammed with INFO drifts on every well-behaved synthesis step.
    """
    synthesis_shapes = [
        Task(id="s1", title="Format the slides for the deck", description=""),
        Task(
            id="s2",
            title="Write a one-paragraph summary",
            description="From the provided data, write a summary.",
        ),
        Task(id="s3", title="Draft the opening line", description="Make it punchy."),
        Task(id="s4", title="", description="Refactor the code for clarity."),
    ]
    for task in synthesis_shapes:
        drift = classify_confabulation_risk(
            task=task,
            tool_call_count=0,
            output_text="Here is the output...",
        )
        assert drift is None, f"should not fire for synthesis task {task.title!r}"


def test_no_fire_when_task_is_none() -> None:
    """Tasks without a clear assignee / id fall through to no-op.

    This matches the "out of scope" contract in issue #128: we don't
    try to infer from context when there's no task at all.
    """
    drift = classify_confabulation_risk(
        task=None,
        tool_call_count=0,
        output_text="Some output.",
    )
    assert drift is None


def test_accepts_duck_typed_task() -> None:
    """The classifier must not require a full Task dataclass.

    Per issue #128 the ``task`` argument may be any duck-typed object
    exposing ``title`` / ``description`` — the ADK plugin or any
    alternate adapter should be free to pass a lightweight shim.
    """

    class _Shim:
        title = "Please research the quarterly numbers"
        description = ""
        id = "shim-1"
        assignee_agent_id = "analyst"

    drift = classify_confabulation_risk(
        task=_Shim(),
        tool_call_count=0,
        output_text="The quarterly numbers are ...",
    )
    assert drift is not None
    assert drift.kind is DriftKind.CONFABULATION_RISK
    assert drift.current_task_id == "shim-1"
    assert drift.current_agent_id == "analyst"


def test_missing_attributes_tolerated() -> None:
    """Missing ``title`` / ``description`` attrs are treated as empty.

    An adapter stub that only has one of the fields (or neither) must
    not raise. A shape with neither field can't match any keyword, so
    the result is None.
    """

    class _Bare:
        id = "b1"

    drift = classify_confabulation_risk(
        task=_Bare(),
        tool_call_count=0,
        output_text="some text",
    )
    assert drift is None


def test_detail_message_includes_task_id_and_keyword() -> None:
    """Detail must carry enough context for a human to triage the drift."""
    task = Task(
        id="t-abc-123",
        title="Verify the current exchange rates",
        description="",
    )
    drift = classify_confabulation_risk(
        task=task,
        tool_call_count=0,
        output_text="The rate is 1.23",
    )
    assert drift is not None
    assert "t-abc-123" in drift.detail
    assert "verify" in drift.detail
    assert "zero tool calls" in drift.detail


# ---------------------------------------------------------------------------
# Tool-loop drift detector: plugin wiring (goldfive#181)
# ---------------------------------------------------------------------------


class _RecordingToolLoopSteerer:
    """Steerer stub that captures every drift routed via ``_handle_drift``."""

    def __init__(self) -> None:
        self.drifts: list[DriftEvent] = []
        self.observations: list[Any] = []
        self._sinks: list[Any] = []

    async def observe(self, event: Any, session: Any) -> None:
        self.observations.append(event)

    async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
        self.drifts.append(drift)


def _plugin_with_tool_loop_ctx(task: Task, session: Session, steerer: Any):
    """Build an ADK plugin + state-context for tool-loop wiring tests.

    Importing deferred so the module stays import-clean when ``google.adk``
    isn't installed (the integration tests guard on ``importorskip``).
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=task,
            tool_handlers={},
            host_agent_name="test_agent",
        )
    }
    # Install the context on the plugin directly so _resolve_ctx
    # returns it without needing the ADK state-dict path.
    plugin.set_active_context(state[SESSION_CONTEXT_STATE_KEY])
    return plugin, state


class _ToolStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _InvCtxStub:
    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self.invocation_id = invocation_id
        self.agent = type("_A", (), {"name": agent_name})()


class _ToolCtxStub:
    """Minimal ADK tool_context shape exposing ``_invocation_context``."""

    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self._invocation_context = _InvCtxStub(invocation_id, agent_name)


async def test_plugin_after_tool_emits_tool_loop_drift() -> None:
    """Three identical arbitrary tool calls -> LOOPING_REASONING via the plugin."""
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-tl-1")
    task = Task(id="t-loop", title="Something", assignee_agent_id="worker")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool = _ToolStub("read_file")
    args = {"path": "doc.md"}
    tool_ctx = _ToolCtxStub("inv-loop-1", "test_agent")

    # First two calls should not fire.
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result={"ok": True}
    )
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result={"ok": True}
    )
    assert steerer.drifts == []

    # Third identical call trips mode 1.
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result={"ok": True}
    )
    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.WARNING
    assert drift.raw is not None
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("tool_name") == "read_file"
    assert drift.current_task_id == "t-loop"


async def test_plugin_progress_tool_resets_tool_loop_window() -> None:
    """A reporting-progress tool call clears the per-(invocation, agent) buffer."""
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-tl-2")
    task = Task(id="t-progress", title="Something", assignee_agent_id="worker")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool_ctx = _ToolCtxStub("inv-loop-2", "test_agent")
    tool = _ToolStub("read_file")
    args = {"path": "doc.md"}

    # Seed two identical calls.
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result={"ok": True}
    )
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result={"ok": True}
    )
    # Progress-reporting tool clears the window.
    progress_tool = _ToolStub("report_task_progress")
    await plugin.after_tool_callback(
        tool=progress_tool,
        tool_args={"task_id": "t-progress", "fraction": 0.5},
        tool_context=tool_ctx,
        result={"acknowledged": True},
    )
    # One more identical read_file call should NOT fire because the
    # buffer was cleared.
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result={"ok": True}
    )
    assert steerer.drifts == []


async def test_plugin_cross_agent_tool_loop_isolation() -> None:
    """Same tool, same args, same invocation but different sub-agents -> no drift."""
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-tl-3")
    task = Task(id="t-iso", title="Something", assignee_agent_id="coord")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool = _ToolStub("read_file")
    args = {"path": "same.md"}

    # Three identical calls but each attributed to a DIFFERENT sub-agent
    # (the ADK running_agent.name on ``_invocation_context``).
    for agent_name in ("researcher", "writer", "reviewer"):
        await plugin.after_tool_callback(
            tool=tool,
            tool_args=args,
            tool_context=_ToolCtxStub("inv-iso", agent_name),
            result={"ok": True},
        )
    assert steerer.drifts == []
