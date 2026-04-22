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
