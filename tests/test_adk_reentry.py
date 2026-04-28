"""Tests for the goldfive plugin re-entry contract.

Covers (harmonograf#234):

* :data:`current_reentry_kind` defaults to ``USER_TURN``.
* :func:`reentry` pins the kind for the duration of the block and
  resets it on exit, including on exception.
* Stack precedence: more-specific kinds (STEER/NUDGE) survive nesting
  inside OVERLAY; OVERLAY does NOT downgrade an inner STEER/NUDGE.
* :meth:`ADKAdapter.invoke_passthrough` pins ``OVERLAY_REPLAY`` across
  the inner ``runner.run_async`` call.
* :meth:`SequentialExecutor._run_overlay`'s nudge-replay path pins
  ``NUDGE_REPLAY`` for the next iteration's invoke.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from goldfive.adapters.adk_reentry import (
    ReentryKind,
    current_reentry_kind,
    reentry,
)


def test_default_is_user_turn() -> None:
    assert current_reentry_kind.get() is ReentryKind.USER_TURN


def test_reentry_sets_and_resets() -> None:
    assert current_reentry_kind.get() is ReentryKind.USER_TURN
    with reentry(ReentryKind.OVERLAY_REPLAY) as kind:
        assert kind is ReentryKind.OVERLAY_REPLAY
        assert current_reentry_kind.get() is ReentryKind.OVERLAY_REPLAY
    assert current_reentry_kind.get() is ReentryKind.USER_TURN


def test_reentry_resets_on_exception() -> None:
    assert current_reentry_kind.get() is ReentryKind.USER_TURN
    with pytest.raises(RuntimeError):
        with reentry(ReentryKind.STEER_REPLAY):
            assert current_reentry_kind.get() is ReentryKind.STEER_REPLAY
            raise RuntimeError("boom")
    assert current_reentry_kind.get() is ReentryKind.USER_TURN


def test_steer_inside_overlay_keeps_steer_visible() -> None:
    """Natural shape: executor pins STEER, then ADKAdapter.invoke_passthrough
    pins OVERLAY around the inner runner — plugins see STEER, not OVERLAY.
    """
    with reentry(ReentryKind.STEER_REPLAY):
        assert current_reentry_kind.get() is ReentryKind.STEER_REPLAY
        with reentry(ReentryKind.OVERLAY_REPLAY) as nested:
            # The nested call observes the prior, more-specific kind.
            assert nested is ReentryKind.STEER_REPLAY
            assert current_reentry_kind.get() is ReentryKind.STEER_REPLAY
        assert current_reentry_kind.get() is ReentryKind.STEER_REPLAY
    assert current_reentry_kind.get() is ReentryKind.USER_TURN


def test_nudge_inside_overlay_keeps_nudge_visible() -> None:
    with reentry(ReentryKind.NUDGE_REPLAY):
        with reentry(ReentryKind.OVERLAY_REPLAY) as nested:
            assert nested is ReentryKind.NUDGE_REPLAY
            assert current_reentry_kind.get() is ReentryKind.NUDGE_REPLAY


def test_overlay_inside_steer_does_not_downgrade() -> None:
    """OVERLAY entered after STEER must not clobber the more-specific kind."""
    with reentry(ReentryKind.STEER_REPLAY):
        with reentry(ReentryKind.OVERLAY_REPLAY):
            assert current_reentry_kind.get() is ReentryKind.STEER_REPLAY
        assert current_reentry_kind.get() is ReentryKind.STEER_REPLAY


def test_steer_inside_nudge_promotes_steer() -> None:
    """STEER and NUDGE are both 'specific'; the inner one wins for that block.

    This shape is unlikely in practice (the executor only pins one of
    the two before each iteration) but the contract should be predictable.
    """
    with reentry(ReentryKind.NUDGE_REPLAY):
        with reentry(ReentryKind.STEER_REPLAY):
            assert current_reentry_kind.get() is ReentryKind.STEER_REPLAY
        assert current_reentry_kind.get() is ReentryKind.NUDGE_REPLAY


def test_nested_user_turn_from_user_turn_is_a_noop() -> None:
    with reentry(ReentryKind.USER_TURN):
        assert current_reentry_kind.get() is ReentryKind.USER_TURN
    assert current_reentry_kind.get() is ReentryKind.USER_TURN


# ---------------------------------------------------------------------------
# ADKAdapter.invoke_passthrough pin site
# ---------------------------------------------------------------------------

pytest.importorskip("google.adk")

from goldfive.types import Plan, Session, Task  # noqa: E402


def _make_agent() -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name="reentry_test_agent",
        model="fake-model",
        description="Test",
        instruction="x",
    )


@dataclass
class _ReentryProbingRunner:
    """Runner stub that records ``current_reentry_kind`` at run_async time."""

    observed_kind: ReentryKind | None = None
    session_service: Any = None
    plugin_manager: Any = field(default=None)

    async def run_async(self, **_kwargs: Any):
        self.observed_kind = current_reentry_kind.get()
        if False:  # pragma: no cover -- generator shape
            yield None


async def test_invoke_passthrough_pins_overlay_replay() -> None:
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_make_agent())
    runner = _ReentryProbingRunner()
    adapter._runner = runner
    adapter._session_id = "stub-session"

    session = Session(
        run_id="r1",
        plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[], edges=[]),
    )

    # Outer scope is USER_TURN.
    assert current_reentry_kind.get() is ReentryKind.USER_TURN
    await adapter.invoke_passthrough("hello world", session=session)

    # Inside the inner runner, the contextvar saw OVERLAY_REPLAY.
    assert runner.observed_kind is ReentryKind.OVERLAY_REPLAY
    # Outer scope is restored.
    assert current_reentry_kind.get() is ReentryKind.USER_TURN


async def test_invoke_passthrough_preserves_outer_steer_replay() -> None:
    """Executor wraps the next-iteration invoke in ``reentry(STEER_REPLAY)``
    and ``invoke_passthrough`` itself nests ``OVERLAY_REPLAY``. The inner
    runner must observe STEER_REPLAY (the more-specific cause).
    """
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_make_agent())
    runner = _ReentryProbingRunner()
    adapter._runner = runner
    adapter._session_id = "stub-session"

    session = Session(
        run_id="r1",
        plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[], edges=[]),
    )

    with reentry(ReentryKind.STEER_REPLAY):
        await adapter.invoke_passthrough("steer body", session=session)

    assert runner.observed_kind is ReentryKind.STEER_REPLAY


async def test_invoke_does_not_pin_overlay_replay() -> None:
    """Legacy per-task ``invoke`` is goldfive's own dispatch with task
    framing — not a re-entry of the operator's user_input. It must not
    pin OVERLAY_REPLAY.
    """
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_make_agent())
    runner = _ReentryProbingRunner()
    adapter._runner = runner
    adapter._session_id = "stub-session"

    task = Task(id="t0", title="do work", description="x")
    session = Session(
        run_id="r1",
        plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
    )
    await adapter.invoke(task=task, session=session)

    # The legacy invoke path runs in the default USER_TURN context.
    assert runner.observed_kind is ReentryKind.USER_TURN


# ---------------------------------------------------------------------------
# SequentialExecutor nudge-replay pin site
# ---------------------------------------------------------------------------


async def test_executor_nudge_replay_pins_nudge_replay() -> None:
    """A queued nudge causes the next iteration's invoke to be wrapped
    in ``reentry(NUDGE_REPLAY)``. The adapter sees NUDGE_REPLAY (more
    specific than the OVERLAY_REPLAY pin invoke_passthrough adds).
    """
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_make_agent())
    runner = _ReentryProbingRunner()
    adapter._runner = runner
    adapter._session_id = "stub-session"

    session = Session(
        run_id="r1",
        plan=Plan(id="p0", run_id="r1", goal_ids=[], tasks=[], edges=[]),
    )

    # Simulate the executor's nudge-replay branch: it sets
    # ``next_reentry_kind = NUDGE_REPLAY`` and re-enters the loop, which
    # wraps the call to invoke_passthrough in ``reentry(...)``.
    with reentry(ReentryKind.NUDGE_REPLAY):
        await adapter.invoke_passthrough("nudge body", session=session)

    assert runner.observed_kind is ReentryKind.NUDGE_REPLAY
