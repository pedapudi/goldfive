"""Unit tests for ``Session.recent_tool_observations`` + steerer writer.

iter-10 PR 2. The buffer feeds the three-state reasoning judge (PR 3)
so it can distinguish a provoked deviation (the agent saw a tool error
or surprising result and pivoted) from an unprovoked one.

These tests pin the writer contract:

* :meth:`DefaultSteerer.note_tool_observation` appends one entry per
  call.
* The buffer is bounded by ``session.recent_tool_observations_max``
  with trim-on-write — newest wins, oldest dropped.
* The error path is detected from either an explicit ``error=`` kwarg
  (the ``on_tool_error_callback`` shape) or from a ``{"error": ...}``
  dict ``result`` (the acknowledged-failure shape).
* Internal failures are swallowed — observability must never break
  tool dispatch.
* Unhashable / unrepresentable args + results are repr-shaped and do
  not raise out of the writer.

PR 3 wires the judge prompt to read from this buffer; this PR ships
only the population path.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import Session  # noqa: E402


def _make_session() -> Session:
    return Session(run_id="r-iter10-pr2")


# ---------------------------------------------------------------------------
# Append + shape
# ---------------------------------------------------------------------------


def test_note_tool_observation_appends() -> None:
    """One call -> one entry with the full §3.1 shape."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="planner",
        task_id="t1",
        tool_name="web_search",
        args={"q": "raccoon facts"},
        result={"results": ["fact a", "fact b"]},
    )

    assert len(session.recent_tool_observations) == 1
    entry = session.recent_tool_observations[0]
    # Spec-mandated keys.
    assert set(entry.keys()) == {
        "ts_ms",
        "agent_name",
        "task_id",
        "tool_name",
        "args_preview",
        "result_preview",
        "is_error",
        "error_message",
    }
    assert isinstance(entry["ts_ms"], int)
    assert entry["agent_name"] == "planner"
    assert entry["task_id"] == "t1"
    assert entry["tool_name"] == "web_search"
    # repr-shaped previews.
    assert "raccoon facts" in entry["args_preview"]
    assert "fact a" in entry["result_preview"]
    assert entry["is_error"] is False
    assert entry["error_message"] == ""


def test_note_tool_observation_records_none_result_as_marker() -> None:
    """``result=None`` -> ``result_preview == "(none)"`` per §3.1."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="planner",
        task_id="t1",
        tool_name="report_task_started",
        args={"task_id": "t1"},
        result=None,
    )

    entry = session.recent_tool_observations[0]
    assert entry["result_preview"] == "(none)"
    # No error signal in this path — None is just an absent result.
    assert entry["is_error"] is False


# ---------------------------------------------------------------------------
# Trim-on-write
# ---------------------------------------------------------------------------


def test_note_tool_observation_trims_to_max() -> None:
    """``len(hist) > max`` after writes -> oldest dropped, newest retained.

    The default ``recent_tool_observations_max`` is 16. Push 21 calls
    and assert the buffer holds the most-recent 16 in order.
    """
    steerer = DefaultSteerer()
    session = _make_session()
    cap = session.recent_tool_observations_max
    assert cap == 16  # default per §3.1

    for i in range(cap + 5):
        steerer.note_tool_observation(
            session,
            agent_name=f"a{i}",
            task_id="t1",
            tool_name="some_tool",
            args={"i": i},
            result={"i": i},
        )

    hist = session.recent_tool_observations
    assert len(hist) == cap
    # Oldest five (a0..a4) dropped; a5..a20 retained, oldest first.
    assert [e["agent_name"] for e in hist] == [f"a{i}" for i in range(5, cap + 5)]


def test_note_tool_observation_respects_session_local_max_override() -> None:
    """A session may set a smaller ``recent_tool_observations_max``."""
    steerer = DefaultSteerer()
    session = _make_session()
    session.recent_tool_observations_max = 3

    for i in range(8):
        steerer.note_tool_observation(
            session,
            agent_name=f"a{i}",
            task_id="t1",
            tool_name="t",
            args={},
            result={},
        )

    assert len(session.recent_tool_observations) == 3
    assert [e["agent_name"] for e in session.recent_tool_observations] == ["a5", "a6", "a7"]


# ---------------------------------------------------------------------------
# Error detection
# ---------------------------------------------------------------------------


def test_note_tool_observation_records_error_path() -> None:
    """Explicit ``error=Exception(...)`` -> is_error=True, error_message set."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t2",
        tool_name="http_get",
        args={"url": "https://example.com"},
        result=None,
        error=Exception("boom"),
    )

    entry = session.recent_tool_observations[0]
    assert entry["is_error"] is True
    assert entry["error_message"] == "boom"


def test_note_tool_observation_records_error_dict_result() -> None:
    """Acknowledged-failure ``{"error": "..."}`` -> is_error=True."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t3",
        tool_name="report_task_started",
        args={"task_id": "bogus"},
        result={"acknowledged": False, "error": "tool failed: 503"},
    )

    entry = session.recent_tool_observations[0]
    assert entry["is_error"] is True
    assert entry["error_message"] == "tool failed: 503"


def test_note_tool_observation_record_error_string() -> None:
    """``error=`` may be a bare string (some adapters stringify upstream)."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="worker",
        task_id="t",
        tool_name="x",
        args={},
        result=None,
        error="upstream 500",
    )

    entry = session.recent_tool_observations[0]
    assert entry["is_error"] is True
    assert entry["error_message"] == "upstream 500"


def test_note_tool_observation_truncates_long_error_messages() -> None:
    """``error_message`` is bounded at 240 chars per §3.2."""
    steerer = DefaultSteerer()
    session = _make_session()
    long = "x" * 1000

    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args={},
        result=None,
        error=long,
    )

    entry = session.recent_tool_observations[0]
    assert len(entry["error_message"]) == 240


# ---------------------------------------------------------------------------
# Truncation: previews
# ---------------------------------------------------------------------------


def test_note_tool_observation_truncates_args_preview() -> None:
    """``args_preview`` clamped at 240 chars per §3.1."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args={"q": "a" * 500},
        result={"ok": True},
    )

    entry = session.recent_tool_observations[0]
    assert len(entry["args_preview"]) <= 240


def test_note_tool_observation_truncates_result_preview() -> None:
    """``result_preview`` clamped at 480 chars per §3.1."""
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args={},
        result={"data": "a" * 1000},
    )

    entry = session.recent_tool_observations[0]
    assert len(entry["result_preview"]) <= 480


# ---------------------------------------------------------------------------
# Robustness: unhashable / unrepresentable inputs
# ---------------------------------------------------------------------------


def test_note_tool_observation_handles_unhashable_args() -> None:
    """Args containing un-hashable but repr-able objects must not raise."""
    steerer = DefaultSteerer()
    session = _make_session()

    # ``{"x": object()}`` is unhashable but ``repr`` works.
    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args={"x": object()},
        result={"ok": True},
    )

    assert len(session.recent_tool_observations) == 1
    entry = session.recent_tool_observations[0]
    # repr of object() shape is "<object object at 0x...>" — sanity-check
    # that we ran repr at all.
    assert entry["args_preview"].startswith("{")


def test_note_tool_observation_handles_repr_failure_in_args() -> None:
    """An object whose ``repr`` raises -> placeholder, no propagation."""

    class _BadRepr:
        def __repr__(self) -> str:  # pragma: no cover - exercised via writer
            raise RuntimeError("nope")

    steerer = DefaultSteerer()
    session = _make_session()
    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args=_BadRepr(),
        result={"ok": True},
    )

    assert len(session.recent_tool_observations) == 1
    entry = session.recent_tool_observations[0]
    assert entry["args_preview"] == "(unrepresentable args)"


# ---------------------------------------------------------------------------
# Internal-error swallowing
# ---------------------------------------------------------------------------


def test_note_tool_observation_swallows_internal_errors() -> None:
    """A broken clock must not propagate out of the writer.

    Patches ``time.monotonic_ns`` (read inside the steerer module) to
    raise. The writer's broad try/except is the contract — observability
    code must never break tool dispatch.
    """
    steerer = DefaultSteerer()
    session = _make_session()

    with mock.patch("goldfive.steerer.time.monotonic_ns", side_effect=RuntimeError("clock dead")):
        # Must not raise.
        steerer.note_tool_observation(
            session,
            agent_name="w",
            task_id="t",
            tool_name="x",
            args={},
            result={},
        )

    # And: the buffer is left untouched (no half-written entry).
    assert session.recent_tool_observations == []


def test_note_tool_observation_zero_max_clamps_to_one() -> None:
    """A pathological ``max=0`` must not fast-loop trim or crash.

    The writer floors to 1 so the buffer always retains the most-recent
    entry — useful because ``0`` would silently drop everything.
    """
    steerer = DefaultSteerer()
    session = _make_session()
    session.recent_tool_observations_max = 0

    for i in range(3):
        steerer.note_tool_observation(
            session,
            agent_name=f"a{i}",
            task_id="t",
            tool_name="x",
            args={},
            result={},
        )

    assert len(session.recent_tool_observations) == 1
    assert session.recent_tool_observations[0]["agent_name"] == "a2"


# ---------------------------------------------------------------------------
# Session dataclass back-compat
# ---------------------------------------------------------------------------


def test_session_default_recent_tool_observations_is_empty_list() -> None:
    """Bare ``Session(run_id=...)`` constructions get an empty buffer."""
    session = Session(run_id="r")
    assert session.recent_tool_observations == []
    assert session.recent_tool_observations_max == 16


def test_two_sessions_have_independent_buffers() -> None:
    """The mutable-default-list classic gotcha must not bite us."""
    s1 = Session(run_id="r1")
    s2 = Session(run_id="r2")
    s1.recent_tool_observations.append({"sentinel": True})
    assert s2.recent_tool_observations == []


def test_note_tool_observation_per_task_filter_left_to_reader() -> None:
    """Writer captures every task; per-task filtering happens at READ time.

    Locked decision (§3.1): the buffer is global-per-session; the
    judge's prompt renderer in PR 3 filters by ``task_id``. This test
    pins that contract — observations from two distinct tasks both
    land in the same buffer.
    """
    steerer = DefaultSteerer()
    session = _make_session()

    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t1",
        tool_name="x",
        args={},
        result={},
    )
    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t2",
        tool_name="x",
        args={},
        result={},
    )

    task_ids = [e["task_id"] for e in session.recent_tool_observations]
    assert task_ids == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_note_tool_observation_ts_ms_is_monotonic() -> None:
    """``ts_ms`` is monotonic across writes (relative ordering only)."""
    steerer = DefaultSteerer()
    session = _make_session()

    for i in range(5):
        steerer.note_tool_observation(
            session,
            agent_name=f"a{i}",
            task_id="t",
            tool_name="x",
            args={},
            result={},
        )

    ts_values: list[int] = [int(e["ts_ms"]) for e in session.recent_tool_observations]
    assert ts_values == sorted(ts_values)


# ---------------------------------------------------------------------------
# Misc — defensive against odd ``Any`` inputs
# ---------------------------------------------------------------------------


def test_note_tool_observation_handles_none_args() -> None:
    """``args=None`` (some adapter edge cases) -> repr-shaped, no raise."""
    steerer = DefaultSteerer()
    session = _make_session()
    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args=None,
        result={"ok": True},
    )
    assert len(session.recent_tool_observations) == 1
    assert session.recent_tool_observations[0]["args_preview"] == "None"


def test_note_tool_observation_with_string_result() -> None:
    """Bare string ``result`` is repr-encoded, not flagged as error."""
    steerer = DefaultSteerer()
    session = _make_session()
    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args={},
        result="hello world",
    )
    entry = session.recent_tool_observations[0]
    assert "hello world" in entry["result_preview"]
    assert entry["is_error"] is False


# ---------------------------------------------------------------------------
# Explicit Any/typing pin so reviewers can see the writer accepts opaque
# types (it really does have to — tool args / results from arbitrary
# adapters are untyped).
# ---------------------------------------------------------------------------


def test_writer_accepts_arbitrary_any_types() -> None:
    arbitrary: Any = object()
    steerer = DefaultSteerer()
    session = _make_session()
    steerer.note_tool_observation(
        session,
        agent_name="w",
        task_id="t",
        tool_name="x",
        args=arbitrary,
        result=arbitrary,
    )
    assert len(session.recent_tool_observations) == 1
