"""Unit tests for :mod:`goldfive.drift.tool_loops` (goldfive#181).

Covers the eight contracts named in the PR plan:

1. Exact ``(tool_name, args_hash)`` repeat fires a WARNING.
2. Same-name (args-varying) repeat fires a WARNING.
3. Alternating A,B,A,B,A fires an INFO.
4. ``on_task_progress`` clears the window.
5. Cross-invocation buffers are isolated.
6. Cross-agent buffers are isolated.
7. Below-threshold calls do NOT fire.
8. Window slides correctly -- old distinct entries age out so three
   identical calls at the tail still trip mode 1.

Plus small helpers for the env-override config path and the
deterministic args-hashing helper.
"""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

import pytest

from goldfive.drift.tool_loops import (
    DEFAULT_ALTERNATING_THRESHOLD,
    DEFAULT_EXACT_THRESHOLD,
    DEFAULT_NAME_THRESHOLD,
    DEFAULT_WINDOW,
    ToolLoopTracker,
    _classify_tool_category,
    args_hash,
    load_thresholds_from_env,
)
from goldfive.types import DriftKind, DriftSeverity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(
    tracker: ToolLoopTracker,
    calls: list[tuple[str, dict[str, Any]]],
    *,
    invocation_id: str = "inv-1",
    agent_name: str = "agent-a",
    task_id: str = "t1",
) -> list[list[Any]]:
    """Drive ``tracker`` through ``calls`` returning per-call drifts."""
    out: list[list[Any]] = []
    for tool_name, args in calls:
        out.append(
            tracker.observe_tool_call(
                invocation_id=invocation_id,
                agent_name=agent_name,
                tool_name=tool_name,
                args=args,
                task_id=task_id,
            )
        )
    return out


# ---------------------------------------------------------------------------
# args_hash
# ---------------------------------------------------------------------------


def test_args_hash_is_order_insensitive() -> None:
    assert args_hash({"a": 1, "b": 2}) == args_hash({"b": 2, "a": 1})


def test_args_hash_differs_across_values() -> None:
    assert args_hash({"a": 1}) != args_hash({"a": 2})


def test_args_hash_copes_with_non_json_payloads() -> None:
    class Widget:
        def __repr__(self) -> str:
            return "<widget>"

    # Does not raise; returns an 8-char hex hash.
    h = args_hash({"w": Widget()})
    assert isinstance(h, str) and len(h) == 8


# ---------------------------------------------------------------------------
# load_thresholds_from_env
# ---------------------------------------------------------------------------


def test_load_thresholds_defaults_when_unset() -> None:
    # Strip any active env overrides before the call so the test is
    # deterministic regardless of the outer shell's environment.
    with mock.patch.dict(os.environ, {}, clear=False):
        for var in (
            "GOLDFIVE_TOOL_LOOP_WINDOW",
            "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD",
            "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD",
            "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD",
        ):
            os.environ.pop(var, None)
        thresholds = load_thresholds_from_env()
    assert thresholds == {
        "window": DEFAULT_WINDOW,
        "exact_threshold": DEFAULT_EXACT_THRESHOLD,
        "name_threshold": DEFAULT_NAME_THRESHOLD,
        "alternating_threshold": DEFAULT_ALTERNATING_THRESHOLD,
    }


def test_load_thresholds_respects_env_overrides() -> None:
    overrides = {
        "GOLDFIVE_TOOL_LOOP_WINDOW": "11",
        "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD": "4",
        "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD": "7",
        "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD": "6",
    }
    with mock.patch.dict(os.environ, overrides, clear=False):
        thresholds = load_thresholds_from_env()
    assert thresholds == {
        "window": 11,
        "exact_threshold": 4,
        "name_threshold": 7,
        "alternating_threshold": 6,
    }


def test_load_thresholds_rejects_bad_values() -> None:
    # Non-integer and non-positive overrides degrade to defaults.
    bad = {
        "GOLDFIVE_TOOL_LOOP_WINDOW": "not-an-int",
        "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD": "0",
        "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD": "-3",
        "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD": "  ",
    }
    with mock.patch.dict(os.environ, bad, clear=False):
        thresholds = load_thresholds_from_env()
    assert thresholds["window"] == DEFAULT_WINDOW
    assert thresholds["exact_threshold"] == DEFAULT_EXACT_THRESHOLD
    assert thresholds["name_threshold"] == DEFAULT_NAME_THRESHOLD
    assert thresholds["alternating_threshold"] == DEFAULT_ALTERNATING_THRESHOLD


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0},
        {"window": -1},
        {"exact_threshold": 0},
        {"name_threshold": 0},
        {"alternating_threshold": 0},
    ],
)
def test_constructor_rejects_non_positive(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ToolLoopTracker(**kwargs)


# ---------------------------------------------------------------------------
# Mode 1 -- exact repeat
# ---------------------------------------------------------------------------


def test_exact_repeat_fires_warning() -> None:
    """Mode 1: three identical ``(name, args)`` calls fire WARNING."""
    tracker = ToolLoopTracker()
    drifts_per_call = _seed(
        tracker,
        [
            ("read_file", {"path": "a.txt"}),
            ("read_file", {"path": "a.txt"}),
            ("read_file", {"path": "a.txt"}),
        ],
    )
    # First two calls below threshold.
    assert drifts_per_call[0] == []
    assert drifts_per_call[1] == []
    assert len(drifts_per_call[2]) == 1
    drift = drifts_per_call[2][0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.WARNING
    assert "tool_loop_exact" in drift.detail
    assert "read_file" in drift.detail
    assert drift.raw is not None
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("tool_name") == "read_file"
    assert drift.raw.get("count") == 3
    assert drift.current_task_id == "t1"
    assert drift.current_agent_id == "agent-a"


# ---------------------------------------------------------------------------
# Mode 2 -- name repeat (args-varying)
# ---------------------------------------------------------------------------


def test_name_repeat_fires_warning() -> None:
    """Mode 2: five same-name, distinct-args calls fire a WARNING."""
    tracker = ToolLoopTracker()
    drifts_per_call = _seed(
        tracker,
        [("read_file", {"path": f"p{i}.txt"}) for i in range(5)],
    )
    # Calls 1-4 below threshold. Call 5 fires mode 2 (and not
    # mode 1 because each args dict is distinct).
    for i in range(4):
        assert drifts_per_call[i] == [], f"unexpected drift at call {i}"
    assert len(drifts_per_call[4]) == 1
    drift = drifts_per_call[4][0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.WARNING
    assert "tool_loop_name" in drift.detail
    assert drift.raw is not None
    assert drift.raw.get("mode") == "name"
    assert drift.raw.get("tool_name") == "read_file"
    assert drift.raw.get("count") == 5


# ---------------------------------------------------------------------------
# Mode 3 -- alternating cycle
# ---------------------------------------------------------------------------


def test_alternating_pattern_fires_info() -> None:
    """Mode 3: A,B,A,B,A fires an INFO drift."""
    tracker = ToolLoopTracker()
    drifts_per_call = _seed(
        tracker,
        [
            ("A", {"x": 1}),
            ("B", {"y": 1}),
            ("A", {"x": 2}),
            ("B", {"y": 2}),
            ("A", {"x": 3}),
        ],
    )
    # No mode 1 (args differ), no mode 2 (max per-name count = 3 < 5),
    # mode 3 fires on the fifth call.
    assert drifts_per_call[4], "alternating pattern should fire on the 5th call"
    drift = drifts_per_call[4][0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.INFO
    assert "tool_loop_alternating" in drift.detail
    assert drift.raw is not None
    assert drift.raw.get("mode") == "alternating"
    assert set(drift.raw.get("tools", [])) == {"A", "B"}


def test_alternating_requires_two_distinct_tools() -> None:
    """A,A,A,A,A is mode 1, NOT mode 3 -- don't double-fire alternating."""
    tracker = ToolLoopTracker()
    drifts_per_call = _seed(
        tracker,
        [("A", {"x": i}) for i in range(5)],
    )
    # Mode 2 should fire by call 5 (5 calls to "A" with distinct args);
    # mode 3 must NOT also fire -- it would be a false positive.
    drifts = drifts_per_call[4]
    assert len(drifts) == 1
    assert drifts[0].raw.get("mode") == "name"


# ---------------------------------------------------------------------------
# Mode 4 -- progress reset
# ---------------------------------------------------------------------------


def test_task_progress_resets_window() -> None:
    """on_task_progress clears the per-(inv, agent) buffer."""
    tracker = ToolLoopTracker()
    _seed(
        tracker,
        [
            ("read_file", {"path": "a.txt"}),
            ("read_file", {"path": "a.txt"}),
        ],
    )
    # Reset, then send one more identical call -- should NOT trip
    # mode 1 because the buffer is empty.
    tracker.on_task_progress(invocation_id="inv-1", agent_name="agent-a")
    assert tracker.buffer_size(invocation_id="inv-1", agent_name="agent-a") == 0
    third = tracker.observe_tool_call(
        invocation_id="inv-1",
        agent_name="agent-a",
        tool_name="read_file",
        args={"path": "a.txt"},
        task_id="t1",
    )
    assert third == []


# ---------------------------------------------------------------------------
# Cross-invocation isolation
# ---------------------------------------------------------------------------


def test_cross_invocation_isolation() -> None:
    """Two invocations of the same tool don't share a buffer."""
    tracker = ToolLoopTracker()
    a1 = tracker.observe_tool_call(
        invocation_id="inv-1",
        agent_name="agent-a",
        tool_name="read_file",
        args={"path": "x"},
        task_id="t1",
    )
    a2 = tracker.observe_tool_call(
        invocation_id="inv-2",
        agent_name="agent-a",
        tool_name="read_file",
        args={"path": "x"},
        task_id="t1",
    )
    a3 = tracker.observe_tool_call(
        invocation_id="inv-3",
        agent_name="agent-a",
        tool_name="read_file",
        args={"path": "x"},
        task_id="t1",
    )
    # Three identical calls but on THREE different invocations --
    # each bucket has exactly one entry so no drift fires.
    assert a1 == a2 == a3 == []


# ---------------------------------------------------------------------------
# Cross-agent isolation
# ---------------------------------------------------------------------------


def test_cross_agent_isolation() -> None:
    """Same invocation, different agent_name -> independent buckets."""
    tracker = ToolLoopTracker()
    for agent in ("coord", "researcher", "writer"):
        tracker.observe_tool_call(
            invocation_id="inv-1",
            agent_name=agent,
            tool_name="read_file",
            args={"path": "x"},
            task_id="t1",
        )
    # Only one call per agent-bucket -- no drift.
    assert tracker.buffer_size(invocation_id="inv-1", agent_name="coord") == 1
    assert tracker.buffer_size(invocation_id="inv-1", agent_name="researcher") == 1
    assert tracker.buffer_size(invocation_id="inv-1", agent_name="writer") == 1


# ---------------------------------------------------------------------------
# Below-threshold
# ---------------------------------------------------------------------------


def test_below_threshold_no_drift() -> None:
    """Two identical calls with exact_threshold=3 -> no drift."""
    tracker = ToolLoopTracker(exact_threshold=3)
    drifts = _seed(
        tracker,
        [
            ("read_file", {"path": "x"}),
            ("read_file", {"path": "x"}),
        ],
    )
    assert drifts[0] == drifts[1] == []


# ---------------------------------------------------------------------------
# Window sliding
# ---------------------------------------------------------------------------


def test_window_slides() -> None:
    """After ``window`` old calls age out, a fresh burst at the tail fires mode 1.

    Setup: ``window=5, exact_threshold=3``. Seed 5 distinct calls,
    then 2 identical ``X`` calls (no fire yet -- window is
    ``[d, e, X, X]`` after the first 4 age-out slots? actually after
    7 calls the window is the last 5). Then a third identical ``X``
    fires mode 1 regardless of the aged-out distinct entries.
    """
    tracker = ToolLoopTracker(window=5, exact_threshold=3, name_threshold=10)
    # Fill the window with five distinct tool names.
    distinct = [(f"tool_{i}", {"i": i}) for i in range(5)]
    _seed(tracker, distinct)
    # Now seed three identical calls. By the third, the first three
    # distinct entries have aged out and the window contains
    # [tool_3, tool_4, X, X, X] (size=5, maxlen=5).
    x_calls = [("X", {"arg": 1})] * 3
    drifts_per_call = _seed(tracker, x_calls)
    assert drifts_per_call[0] == [] and drifts_per_call[1] == []
    # Third X call fires mode 1.
    assert len(drifts_per_call[2]) >= 1
    drift = drifts_per_call[2][0]
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("tool_name") == "X"
    assert drift.raw.get("count") == 3


# ---------------------------------------------------------------------------
# clear() wipes everything
# ---------------------------------------------------------------------------


def test_clear_wipes_all_buckets() -> None:
    tracker = ToolLoopTracker()
    for agent in ("a", "b"):
        for inv in ("i1", "i2"):
            tracker.observe_tool_call(
                invocation_id=inv,
                agent_name=agent,
                tool_name="read",
                args={},
                task_id="t",
            )
    tracker.clear()
    for agent in ("a", "b"):
        for inv in ("i1", "i2"):
            assert tracker.buffer_size(invocation_id=inv, agent_name=agent) == 0


# ---------------------------------------------------------------------------
# Suppression: mode 1 preempts mode 2 on the same tool
# ---------------------------------------------------------------------------


def test_exact_mode_preempts_name_mode_on_same_tool() -> None:
    """If the exact-signature count is also above the name threshold,
    the tracker emits one exact drift, not both."""
    tracker = ToolLoopTracker(exact_threshold=3, name_threshold=3)
    drifts_per_call = _seed(
        tracker,
        [("read_file", {"path": "same"})] * 3,
    )
    third = drifts_per_call[2]
    assert len(third) == 1
    assert third[0].raw.get("mode") == "exact"


# ---------------------------------------------------------------------------
# Graduated severity (goldfive#204) -- meta vs work categories
# ---------------------------------------------------------------------------


def test_classify_tool_category_meta_prefixes() -> None:
    """All ``report_task_*`` names and ``report_awaiting_approval`` are meta."""
    for name in (
        "report_task_started",
        "report_task_progress",
        "report_task_completed",
        "report_task_failed",
        "report_task_blocked",
        "report_awaiting_approval",
    ):
        assert _classify_tool_category(name) == "meta", name


def test_classify_tool_category_work_fallback() -> None:
    """Arbitrary tool names -- including the empty string -- classify as work."""
    for name in (
        "web_developer_agent",
        "research_agent",
        "patch_file",
        "read_file",
        "",
        "reporter",  # not a ``report_task_`` prefix -- work, not meta
    ):
        assert _classify_tool_category(name) == "work", name


def test_meta_tool_retries_fire_info_at_3_not_warning() -> None:
    """Meta tool (``report_task_completed``) x 3 fires INFO, not WARNING.

    Benign meta-tool retries should not mutate the plan -- INFO goes
    through the ladder at OBSERVE level.
    """
    tracker = ToolLoopTracker()
    drifts_per_call = _seed(
        tracker,
        [("report_task_completed", {"task_id": "t1"})] * 3,
    )
    # Below threshold on calls 1-2.
    assert drifts_per_call[0] == []
    assert drifts_per_call[1] == []
    # Call 3: INFO fires, not WARNING / CRITICAL.
    third = drifts_per_call[2]
    assert len(third) == 1, f"expected single INFO drift, got {third}"
    drift = third[0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.INFO, (
        f"meta retry at 3 should be INFO, got {drift.severity}"
    )
    assert drift.raw is not None
    assert drift.raw.get("category") == "meta"
    assert drift.raw.get("tier") == "info"
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("count") == 3


def test_meta_tool_retries_escalate_to_warning_at_6() -> None:
    """Meta tool x 6 identical calls escalates to WARNING."""
    # Default window (10) is large enough to hold 6 identical calls.
    tracker = ToolLoopTracker()
    drifts_per_call = _seed(
        tracker,
        [("report_task_completed", {"task_id": "t1"})] * 6,
    )
    # Calls 3-5 fire INFO (below WARNING threshold of 6).
    for i in (2, 3, 4):
        assert drifts_per_call[i], f"expected INFO at call {i + 1}"
        assert drifts_per_call[i][0].severity is DriftSeverity.INFO
    # Call 6: WARNING fires.
    sixth = drifts_per_call[5]
    assert len(sixth) == 1
    drift = sixth[0]
    assert drift.severity is DriftSeverity.WARNING
    assert drift.raw.get("category") == "meta"
    assert drift.raw.get("tier") == "warning"
    assert drift.raw.get("count") == 6


def test_work_tool_retries_fire_warning_at_3_as_before() -> None:
    """Backwards-compat: work tool x 3 exact calls still fires WARNING."""
    tracker = ToolLoopTracker()
    drifts_per_call = _seed(
        tracker,
        [("web_developer_agent", {"q": "hello"})] * 3,
    )
    assert drifts_per_call[0] == drifts_per_call[1] == []
    third = drifts_per_call[2]
    assert len(third) == 1
    drift = third[0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    # Pre-#204 behaviour: 3 identical work calls -> WARNING.
    assert drift.severity is DriftSeverity.WARNING
    assert drift.raw.get("category") == "work"
    assert drift.raw.get("tier") == "warning"
    assert drift.raw.get("mode") == "exact"


def test_critical_tool_loops_at_10_meta_or_6_work() -> None:
    """CRITICAL fires at the right per-category thresholds.

    * Meta exact: 10 identical ``report_task_*`` calls -> CRITICAL.
    * Work exact: 6 identical work-tool calls -> CRITICAL.
    """
    # --- Meta CRITICAL at 10 -------------------------------------------------
    meta_tracker = ToolLoopTracker()
    meta_drifts = _seed(
        meta_tracker,
        [("report_task_completed", {"task_id": "t1"})] * 10,
    )
    tenth = meta_drifts[9]
    assert len(tenth) == 1
    assert tenth[0].severity is DriftSeverity.CRITICAL
    assert tenth[0].raw.get("category") == "meta"
    assert tenth[0].raw.get("tier") == "critical"

    # --- Work CRITICAL at 6 --------------------------------------------------
    work_tracker = ToolLoopTracker()
    work_drifts = _seed(
        work_tracker,
        [("web_developer_agent", {"q": "hello"})] * 6,
    )
    sixth = work_drifts[5]
    assert len(sixth) == 1
    assert sixth[0].severity is DriftSeverity.CRITICAL
    assert sixth[0].raw.get("category") == "work"
    assert sixth[0].raw.get("tier") == "critical"


def test_work_name_tier_fires_critical_at_7() -> None:
    """Work tool, args varying, 7 same-name calls -> CRITICAL."""
    tracker = ToolLoopTracker()
    # 7 calls to the same work tool with distinct args -- no exact
    # signature hits, but the name axis trips CRITICAL at 7.
    drifts = _seed(
        tracker,
        [("web_developer_agent", {"q": f"q{i}"}) for i in range(7)],
    )
    # Call 5: name count = 5 -> WARNING tier (work name warning=5).
    assert drifts[4], "expected WARNING at call 5"
    assert drifts[4][0].severity is DriftSeverity.WARNING
    # Call 7: name count = 7 -> CRITICAL.
    seventh = drifts[6]
    assert len(seventh) == 1
    drift = seventh[0]
    assert drift.severity is DriftSeverity.CRITICAL
    assert drift.raw.get("category") == "work"
    assert drift.raw.get("tier") == "critical"
    assert drift.raw.get("mode") == "name"


def test_highest_severity_emitted_once() -> None:
    """At 10 identical meta calls, only CRITICAL fires -- not INFO/WARNING too."""
    tracker = ToolLoopTracker()
    drifts = _seed(
        tracker,
        [("report_task_completed", {"task_id": "t1"})] * 10,
    )
    tenth = drifts[9]
    # Exactly ONE drift, at CRITICAL -- not a cascade of INFO + WARNING + CRITICAL.
    assert len(tenth) == 1, f"expected single drift, got {len(tenth)}: {tenth}"
    assert tenth[0].severity is DriftSeverity.CRITICAL


def test_meta_at_3_does_not_fire_warning_or_critical() -> None:
    """Regression guard: meta x 3 MUST NOT fire WARNING (the original bug)."""
    tracker = ToolLoopTracker()
    drifts = _seed(
        tracker,
        [("report_task_completed", {"task_id": "t1"})] * 3,
    )
    severities = [d.severity for d in drifts[2]]
    assert DriftSeverity.WARNING not in severities
    assert DriftSeverity.CRITICAL not in severities
    assert severities == [DriftSeverity.INFO]


def test_mixed_meta_and_work_in_same_window() -> None:
    """Interleaved meta + work -- each tool classified independently.

    A window holding 3 identical meta calls AND 3 identical work calls
    should classify the work loop at WARNING (higher severity wins
    across tools) rather than stopping at the meta INFO.
    """
    tracker = ToolLoopTracker()
    # Interleave so both end up with 3 exact matches in the window.
    calls = [
        ("report_task_completed", {"task_id": "t1"}),
        ("web_developer_agent", {"q": "x"}),
        ("report_task_completed", {"task_id": "t1"}),
        ("web_developer_agent", {"q": "x"}),
        ("report_task_completed", {"task_id": "t1"}),
        ("web_developer_agent", {"q": "x"}),
    ]
    drifts = _seed(tracker, calls)
    last = drifts[-1]
    # The work tool matched WARNING (count=3 >= work warning exact=3);
    # meta only matched INFO (count=3). Highest severity wins -> WARNING.
    assert len(last) == 1
    assert last[0].severity is DriftSeverity.WARNING
    assert last[0].raw.get("category") == "work"
    assert last[0].raw.get("tool_name") == "web_developer_agent"


def test_legacy_exact_threshold_kwarg_overrides_work_warning() -> None:
    """The ``exact_threshold`` ctor kwarg continues to override work-WARNING.

    Env-override / programmatic callers that passed ``exact_threshold=4``
    pre-#204 should still see WARNING fire at 4 identical work calls,
    not 3.
    """
    tracker = ToolLoopTracker(exact_threshold=4, name_threshold=10)
    drifts = _seed(
        tracker,
        [("web_developer_agent", {"q": "x"})] * 3,
    )
    # At 3 calls: INFO fires (INFO exact=3 module constant, clamped to
    # the WARNING exact of 4 but not raised above). Since INFO exact=3
    # and WARNING exact=4, INFO matches but WARNING does not yet.
    assert drifts[2], "INFO should fire at 3"
    assert drifts[2][0].severity is DriftSeverity.INFO
    # At 4 calls: WARNING fires.
    drifts = _seed(
        tracker,
        [("web_developer_agent", {"q": "x"})],
    )
    assert drifts[0], "WARNING should fire at 4"
    assert drifts[0][0].severity is DriftSeverity.WARNING
