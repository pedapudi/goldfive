"""Unit tests for classifier helpers in :mod:`goldfive.drift`.

The reasoning-based detectors live in :mod:`goldfive.drift.reasoning`
and have their own coverage in ``test_drift_reasoning.py``. This module
pins the cheaper structural classifiers:

* :func:`goldfive.drift.classify_confabulation_risk` (issue #128) —
  flags research / verification tasks that finished with zero tool
  calls and non-empty output.
"""

from __future__ import annotations

import pytest

from goldfive.drift import (
    CONFABULATION_TRIGGER_KEYWORDS,
    classify_confabulation_risk,
)
from goldfive.types import DriftKind, DriftSeverity, Task

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
