"""Graceful-degrade tests for pre-built-Runner callers.

When the caller passes a pre-built ``InMemoryRunner`` (instead of a
BaseAgent tree), ``ADKAdapter`` uses the caller-supplied runner
verbatim. Under the single-Runner model (goldfive#130) there is no
"dispatch" behaviour to degrade — the adapter always drives the one
runner — so the degraded path is thin:

* ``available_agents`` reports just the runner's root agent name
  (goldfive cannot see deeper without walking a tree it doesn't own).
* The wrap-time plugin-installed integrity check is skipped because
  the caller may have passed a runner shape we don't fully control.

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

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
    """Pre-built runner duck-type the adapter accepts in degraded mode."""

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


def test_prebuilt_runner_uses_caller_supplied_runner_verbatim() -> None:
    """Pre-built Runner → ``adapter._runner is the caller's runner``."""
    from goldfive.adapters.adk import ADKAdapter

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)

    assert adapter._degraded_prebuilt_runner is True
    assert adapter._runner is runner
    assert adapter._agent is inner


def test_prebuilt_runner_available_agents_is_single_root() -> None:
    """``available_agents`` in degraded mode is just the root agent name —
    goldfive cannot see deeper without walking a tree it doesn't own.
    """
    from goldfive.adapters.adk import ADKAdapter

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)
    assert adapter.available_agents == ["solo"]


# ---------------------------------------------------------------------------
# Invoke-time contract
# ---------------------------------------------------------------------------


async def test_prebuilt_runner_invokes_regardless_of_assignee() -> None:
    """A task with any assignee routes to the single runner — NO ValueError.

    Single-Runner model: the assignee is a hint carried on the task, not a
    routing key. Every task drives the one runner; delegation happens via
    ADK's native mechanisms inside the tree.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    inner = _mk("solo")
    runner = _PrebuiltRunner(agent=inner)
    adapter = ADKAdapter(runner)

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
