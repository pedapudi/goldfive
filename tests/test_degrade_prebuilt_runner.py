"""Graceful-degrade tests for pre-built-Runner callers.

Phase 1 preserves a graceful-degrade path: if the caller passes a
pre-built ``InMemoryRunner`` (instead of a BaseAgent tree), ``ADKAdapter``
cannot build a meaningful registry — there's only the one runner and
its one root agent. The adapter:

* Builds a single-entry registry (the root agent's name → the root agent).
* Builds a single-entry runners dict pointing at the caller's runner.
* Logs a warning exactly once at wrap time so the caller knows
  per-assignee dispatch is off.
* Routes EVERY invoke to that single runner regardless of
  ``task.assignee_agent_id`` (with a DEBUG log on mismatch).
* Skips the wrap-time plugin-installed integrity check (the caller may
  have passed a runner shape we don't fully control).

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("google.adk")


def _mk(name: str) -> Any:
    from google.adk.agents.llm_agent import LlmAgent

    return LlmAgent(name=name, model="fake-model", description=name, instruction="x")


@dataclass
class _Event:
    marker: str = ""
    content: Any = None


@dataclass
class _PrebuiltRunner:
    """Pre-built runner duck-type the adapter accepts in degraded mode.

    Mirrors the Phase 1 ``test_degraded_mode_prebuilt_runner_ignores_assignee``
    fake, extended with plugin-manager plumbing so
    ``_register_plugin_on_runner`` can still attach the plugin (degraded
    mode doesn't assume it fails — only that the integrity check is
    skipped afterwards).
    """

    agent: Any = None
    session_service: Any = None
    plugin_manager: Any = None
    app_name: str = "prebuilt"
    plugins: list = field(default_factory=list)
    called: int = 0

    async def run_async(self, **kwargs: Any):  # noqa: ARG002
        self.called += 1
        yield _Event(marker=f"call-{self.called}")


# ---------------------------------------------------------------------------
# Wrap-time contract
# ---------------------------------------------------------------------------


def test_prebuilt_runner_yields_single_entry_registry_and_runners() -> None:
    """Pre-built Runner → ``_registry`` and ``_runners`` each have exactly one entry."""
    from goldfive.adapters.adk import ADKAdapter

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)

    assert adapter._degraded_prebuilt_runner is True
    assert set(adapter._registry) == {"solo"}
    assert adapter._registry["solo"] is inner
    assert set(adapter._runners) == {"solo"}
    assert adapter._runners["solo"] is runner


def test_prebuilt_runner_emits_wrap_time_warning_exactly_once(caplog) -> None:
    """Wrap-time warning fires exactly once — per-assignee dispatch is off.

    We assert the warning is present and the log record's logger name is
    the goldfive adapter logger so a downstream filter can silence it
    narrowly if the caller intentionally opts into degraded mode.
    """
    from goldfive.adapters.adk import ADKAdapter

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)

    with caplog.at_level(logging.WARNING, logger="goldfive.adapters.adk"):
        ADKAdapter(runner)

    degrade_warnings = [
        rec
        for rec in caplog.records
        if rec.name == "goldfive.adapters.adk"
        and rec.levelno == logging.WARNING
        and "pre-built Runner" in rec.getMessage()
    ]
    assert len(degrade_warnings) == 1, (
        f"expected exactly one degrade warning; got {len(degrade_warnings)}. "
        f"records={[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Invoke-time contract
# ---------------------------------------------------------------------------


async def test_prebuilt_runner_invokes_regardless_of_assignee() -> None:
    """A task with any assignee routes to the single runner — NO ValueError."""
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)

    # Assignee names we never registered. Must not raise; routes to
    # the single runner regardless.
    for assignee in ("not_in_registry", "also_not_here", "definitely_missing"):
        task = Task(id=f"t-{assignee}", title="x", assignee_agent_id=assignee)
        await adapter.invoke(
            task=task,
            session=Session(
                run_id="r1",
                plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
            ),
        )
    assert runner.called == 3


async def test_prebuilt_runner_invoke_logs_debug_on_assignee_mismatch(caplog) -> None:
    """DEBUG log on assignee mismatch — visible but non-fatal.

    Callers that opt into degraded mode may still populate
    ``task.assignee_agent_id`` (e.g. because the planner is shared with
    the multi-agent tree path). The mismatch shouldn't crash but it
    should be diagnosable via logging.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)

    task = Task(id="t1", title="x", assignee_agent_id="mismatch_assignee")
    with caplog.at_level(logging.DEBUG, logger="goldfive.adapters.adk"):
        await adapter.invoke(
            task=task,
            session=Session(
                run_id="r1",
                plan=Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[]),
            ),
        )
    mismatch_debug = [
        rec
        for rec in caplog.records
        if rec.name == "goldfive.adapters.adk"
        and rec.levelno == logging.DEBUG
        and "degraded pre-built Runner" in rec.getMessage()
    ]
    assert mismatch_debug, (
        "expected a DEBUG log on assignee-mismatch in degraded mode; "
        "callers rely on it to trace unexpected plan/adapter pairings"
    )


async def test_prebuilt_runner_skips_plugin_installed_integrity_check() -> None:
    """A pre-built runner whose plugin_manager is ``None`` still wraps —
    the ``__init__`` plugin-installed check is skipped in degraded mode.

    Regression: always-checking would break the graceful-degrade
    contract for callers with non-standard Runner shapes (e.g. a custom
    Runner subclass that moves plugins behind an adapter).
    """
    from goldfive.adapters.adk import ADKAdapter

    inner = _mk("solo")
    # plugin_manager defaults to None on _PrebuiltRunner — no plugin
    # registration path is available. Must still wrap successfully.
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)
    assert adapter._degraded_prebuilt_runner is True


async def test_prebuilt_runner_available_agents_is_single_root() -> None:
    """``available_agents`` in degraded mode is just the root agent name —
    no sub-agents are discovered because goldfive can only see the
    runner's top-level agent.
    """
    from goldfive.adapters.adk import ADKAdapter

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)
    assert adapter.available_agents == ["solo"]
