"""Regression tests for the unified ``Session.recent_events`` buffer.

Goldfive#239. Merges the historical ``recent_agent_activity`` (fed to
the GOAL_DRIFT judge) and ``recent_tool_observations`` (fed to the
three-state reasoning judge) into a single discriminator-tagged buffer.

Pins:

* Both writer paths land entries on ``session.recent_events`` with the
  correct ``kind`` discriminator.
* Insertion order is preserved across writes regardless of which
  writer produced the entry (interleaved writes round-trip as one
  monotonic stream).
* The unified-buffer filter helper (:func:`filter_recent_events_by_kind`)
  recovers the pre-merge sub-buffers byte-identical to what the old
  ``recent_agent_activity`` / ``recent_tool_observations`` properties
  returned.
* Per-kind-class trim semantics: a flood of one kind cannot evict
  entries of the other kind (the legacy buffers had independent caps;
  the unified buffer preserves that with per-kind trimming at write
  time).
* The legacy cap knobs continue to bound their respective kind-classes.

This module is the regression net for the refactor; if any reader/
writer drifts off the unified contract, the asserts here fire.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    RECENT_EVENT_AGENT_ACTIVITY_KINDS,
    RECENT_EVENT_KIND_AGENT_COMPLETED,
    RECENT_EVENT_KIND_AGENT_STARTED,
    RECENT_EVENT_KIND_TOOL_OBSERVED,
    Session,
    filter_recent_events_by_kind,
)


def _make_session() -> Session:
    return Session(run_id="r-goldfive-239")


# ---------------------------------------------------------------------------
# Writers land on the unified buffer with the right discriminator
# ---------------------------------------------------------------------------


def test_note_agent_activity_writes_to_recent_events_with_kind() -> None:
    """``note_agent_activity`` lands an entry on ``recent_events`` with
    the supplied ``kind``."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.drift.note_agent_activity(
        session,
        kind=RECENT_EVENT_KIND_AGENT_STARTED,
        agent_name="worker",
        task_id="t1",
    )

    assert len(session.recent_events) == 1
    entry = session.recent_events[0]
    assert entry["kind"] == RECENT_EVENT_KIND_AGENT_STARTED
    assert entry["agent_name"] == "worker"
    assert entry["task_id"] == "t1"


def test_note_tool_observation_writes_to_recent_events_with_kind() -> None:
    """``note_tool_observation`` lands a ``tool_observed``-kind entry."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.drift.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t1",
        tool_name="web_search",
        args={"q": "foo"},
        result={"ok": True},
    )

    assert len(session.recent_events) == 1
    entry = session.recent_events[0]
    assert entry["kind"] == RECENT_EVENT_KIND_TOOL_OBSERVED
    assert entry["tool_name"] == "web_search"


# ---------------------------------------------------------------------------
# Ordering: interleaved writes preserve insertion order
# ---------------------------------------------------------------------------


def test_interleaved_writes_preserve_insertion_order() -> None:
    """Mixed agent-activity + tool-observation writes share one ordered
    stream — the canonical use case the merge enables."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.drift.note_agent_activity(
        session,
        kind=RECENT_EVENT_KIND_AGENT_STARTED,
        agent_name="worker",
        task_id="t1",
    )
    steerer.drift.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t1",
        tool_name="web_search",
        args={"q": "a"},
        result={"ok": True},
    )
    steerer.drift.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t1",
        tool_name="http_get",
        args={"url": "u"},
        result=None,
    )
    steerer.drift.note_agent_activity(
        session,
        kind=RECENT_EVENT_KIND_AGENT_COMPLETED,
        agent_name="worker",
        task_id="t1",
    )

    kinds_in_order = [e["kind"] for e in session.recent_events]
    assert kinds_in_order == [
        RECENT_EVENT_KIND_AGENT_STARTED,
        RECENT_EVENT_KIND_TOOL_OBSERVED,
        RECENT_EVENT_KIND_TOOL_OBSERVED,
        RECENT_EVENT_KIND_AGENT_COMPLETED,
    ]


# ---------------------------------------------------------------------------
# Filter helper recovers the pre-merge sub-buffers
# ---------------------------------------------------------------------------


def test_filter_recent_events_by_kind_recovers_tool_observations() -> None:
    """The ``tool_observed`` filter view matches the pre-merge buffer."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.drift.note_agent_activity(
        session,
        kind=RECENT_EVENT_KIND_AGENT_STARTED,
        agent_name="worker",
        task_id="t1",
    )
    steerer.drift.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t1",
        tool_name="web_search",
        args={"q": "a"},
        result={"ok": True},
    )

    tool_obs = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_KIND_TOOL_OBSERVED
    )
    assert len(tool_obs) == 1
    assert tool_obs[0]["tool_name"] == "web_search"


def test_filter_recent_events_by_kind_recovers_agent_activity() -> None:
    """The agent-activity filter view matches the pre-merge buffer."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.drift.note_agent_activity(
        session,
        kind=RECENT_EVENT_KIND_AGENT_STARTED,
        agent_name="worker",
        task_id="t1",
    )
    steerer.drift.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t1",
        tool_name="web_search",
        args={"q": "a"},
        result={"ok": True},
    )
    steerer.drift.note_agent_activity(
        session,
        kind=RECENT_EVENT_KIND_AGENT_COMPLETED,
        agent_name="worker",
        task_id="t1",
    )

    activity = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_AGENT_ACTIVITY_KINDS
    )
    assert len(activity) == 2
    assert [e["kind"] for e in activity] == [
        RECENT_EVENT_KIND_AGENT_STARTED,
        RECENT_EVENT_KIND_AGENT_COMPLETED,
    ]


def test_filter_returns_fresh_copy() -> None:
    """The filter helper returns a fresh list; mutations are isolated."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.drift.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args={},
        result={},
    )

    view = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_KIND_TOOL_OBSERVED
    )
    view.clear()
    # Underlying buffer untouched.
    assert len(session.recent_events) == 1


# ---------------------------------------------------------------------------
# Per-kind-class trim: floods of one kind do not evict the other
# ---------------------------------------------------------------------------


def test_tool_observation_flood_does_not_evict_agent_activity() -> None:
    """Push 50 tool observations after one agent_invocation_started.

    Without per-kind trimming the agent entry would be evicted; with
    per-kind trimming the agent-activity subset is preserved while the
    ``tool_observed`` subset is bounded by
    ``recent_tool_observations_max`` (default 16).
    """
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.drift.note_agent_activity(
        session,
        kind=RECENT_EVENT_KIND_AGENT_STARTED,
        agent_name="worker",
        task_id="t1",
    )
    for i in range(50):
        steerer.drift.note_tool_observation(
            session,
            agent_name="worker",
            task_id="t1",
            tool_name="tool",
            args={"i": i},
            result={"i": i},
        )

    # Agent-activity entry SURVIVED.
    activity = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_AGENT_ACTIVITY_KINDS
    )
    assert len(activity) == 1
    assert activity[0]["agent_name"] == "worker"

    # Tool observations capped at the default 16.
    tool_obs = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_KIND_TOOL_OBSERVED
    )
    assert len(tool_obs) == session.recent_tool_observations_max == 16


def test_agent_activity_flood_does_not_evict_tool_observations() -> None:
    """Inverse: 50 agent_invocation_started writes after a tool obs.

    The tool observation must survive; the agent-activity subset is
    capped at ``goal_drift_activity_window`` (default 10).
    """
    steerer = DefaultSteerer()  # default goal_drift_activity_window=10
    session = _make_session()

    steerer.drift.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t1",
        tool_name="web_search",
        args={"q": "the one true tool obs"},
        result={"ok": True},
    )
    for i in range(50):
        steerer.drift.note_agent_activity(
            session,
            kind=RECENT_EVENT_KIND_AGENT_STARTED,
            agent_name=f"a{i}",
            task_id="t1",
        )

    # Tool observation SURVIVED.
    tool_obs = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_KIND_TOOL_OBSERVED
    )
    assert len(tool_obs) == 1
    assert tool_obs[0]["tool_name"] == "web_search"

    # Agent-activity capped at the default window=10.
    activity = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_AGENT_ACTIVITY_KINDS
    )
    assert len(activity) == 10
    # Newest 10 retained (oldest dropped).
    assert [e["agent_name"] for e in activity] == [f"a{i}" for i in range(40, 50)]


def test_per_kind_caps_are_independent() -> None:
    """Both kind-classes can simultaneously sit at their respective caps.

    Pushes 25 entries of each kind. Expected: tool_observed capped at
    ``recent_tool_observations_max`` (16); agent-activity capped at
    ``goal_drift_activity_window`` (10 default). Total buffer size is
    bounded by the sum (10+16=26).
    """
    steerer = DefaultSteerer()
    session = _make_session()

    for i in range(25):
        steerer.drift.note_agent_activity(
            session,
            kind=RECENT_EVENT_KIND_AGENT_STARTED,
            agent_name=f"a{i}",
            task_id="t1",
        )
        steerer.drift.note_tool_observation(
            session,
            agent_name=f"a{i}",
            task_id="t1",
            tool_name="x",
            args={},
            result={},
        )

    activity = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_AGENT_ACTIVITY_KINDS
    )
    tool_obs = filter_recent_events_by_kind(
        session.recent_events, RECENT_EVENT_KIND_TOOL_OBSERVED
    )
    assert len(activity) == 10
    assert len(tool_obs) == 16
    # Buffer is bounded.
    assert len(session.recent_events) == 10 + 16


# ---------------------------------------------------------------------------
# Filter helper edge cases
# ---------------------------------------------------------------------------


def test_filter_handles_empty_buffer() -> None:
    """An empty buffer filters to an empty list."""
    assert filter_recent_events_by_kind([], RECENT_EVENT_KIND_TOOL_OBSERVED) == []
    assert filter_recent_events_by_kind(None, RECENT_EVENT_KIND_TOOL_OBSERVED) == []


def test_filter_handles_non_dict_entries() -> None:
    """Pathological non-dict entries are skipped, not raised on."""
    events = [
        {"kind": RECENT_EVENT_KIND_TOOL_OBSERVED, "tool_name": "x"},
        "not a dict",
        None,
        {"kind": "other"},
        {"kind": RECENT_EVENT_KIND_TOOL_OBSERVED, "tool_name": "y"},
    ]
    filtered = filter_recent_events_by_kind(events, RECENT_EVENT_KIND_TOOL_OBSERVED)
    assert [e["tool_name"] for e in filtered] == ["x", "y"]


def test_filter_accepts_string_kind() -> None:
    """One-kind filter accepts a bare string."""
    events = [
        {"kind": RECENT_EVENT_KIND_TOOL_OBSERVED, "tool_name": "x"},
        {"kind": RECENT_EVENT_KIND_AGENT_STARTED},
    ]
    filtered = filter_recent_events_by_kind(events, "tool_observed")
    assert len(filtered) == 1


def test_filter_accepts_multiple_kinds() -> None:
    """Multi-kind filter accepts a set/frozenset/tuple/list."""
    events = [
        {"kind": RECENT_EVENT_KIND_AGENT_STARTED},
        {"kind": RECENT_EVENT_KIND_TOOL_OBSERVED},
        {"kind": RECENT_EVENT_KIND_AGENT_COMPLETED},
    ]
    multi = filter_recent_events_by_kind(events, RECENT_EVENT_AGENT_ACTIVITY_KINDS)
    assert len(multi) == 2

    # Also accepts a plain list.
    multi_list = filter_recent_events_by_kind(
        events,
        [RECENT_EVENT_KIND_AGENT_STARTED, RECENT_EVENT_KIND_AGENT_COMPLETED],
    )
    assert len(multi_list) == 2


# ---------------------------------------------------------------------------
# Bare ``Session`` shape — back-compat for callers wiring sessions directly
# ---------------------------------------------------------------------------


def test_bare_session_has_empty_recent_events() -> None:
    """Bare ``Session(run_id=...)`` starts with an empty unified buffer."""
    session = Session(run_id="r")
    assert session.recent_events == []
    # The legacy cap knob is still on the dataclass for env compatibility.
    assert session.recent_tool_observations_max == 16


def test_two_sessions_have_independent_recent_events() -> None:
    """Mutable-default-list gotcha guard."""
    s1 = Session(run_id="r1")
    s2 = Session(run_id="r2")
    s1.recent_events.append({"kind": RECENT_EVENT_KIND_AGENT_STARTED})
    assert s2.recent_events == []
