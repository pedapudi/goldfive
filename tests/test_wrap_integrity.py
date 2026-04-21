"""Wrap-time integrity check tests.

Phase 1 added two wrap-time integrity guards to ``ADKAdapter``:

1. The plugin must install on EVERY per-agent runner (checked at the
   tail of ``__init__`` — raises ``RuntimeError`` with the offending
   agent name).
2. After ``register_reporting_tools``, EVERY registry agent must carry
   the set of tool names we registered (checked at the tail of
   ``register_reporting_tools`` — raises ``RuntimeError`` listing the
   missing agents).

These tests exercise both guards.

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

pytest.importorskip("google.adk")


def _mk(name: str) -> Any:
    from google.adk.agents.llm_agent import LlmAgent

    return LlmAgent(name=name, model="fake-model", description=name, instruction="x")


# ---------------------------------------------------------------------------
# register_reporting_tools integrity check
# ---------------------------------------------------------------------------


async def test_register_reporting_tools_raises_if_agent_refuses_tool_assignment() -> None:
    """An agent whose ``tools`` setter raises must cause
    ``register_reporting_tools`` to fail with a message that names the
    offending agent.

    Regression: without the tail integrity check,
    ``_augment_subtree_with_reporting`` swallows the write failure (it
    logs at DEBUG and continues). That leaves the stubborn agent in the
    registry as a dispatchable target but without the reporting tools
    — so if goldfive routes a task to it, the agent cannot report
    terminal status and the executor spins to its max-invocations cap.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS

    # Stubborn agent refuses ``self.tools = ...`` but still exposes a
    # ``.tools`` list (so the traversal sees it) and a name (so it
    # lands in the registry).
    class _Stubborn:
        def __init__(self, name: str) -> None:
            self.name = name
            self.sub_agents: list[Any] = []
            self._tools: list[Any] = []

        @property
        def tools(self) -> list[Any]:
            return self._tools

        @tools.setter
        def tools(self, value: Any) -> None:
            raise RuntimeError("stubborn agent refuses tool writes")

    root = _mk("root")
    stubborn = _Stubborn("stubborn")
    root.sub_agents = [stubborn]

    adapter = ADKAdapter(root)
    with pytest.raises(RuntimeError) as excinfo:
        await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))
    msg = str(excinfo.value)
    # The message MUST name the offending agent so a developer can find it.
    assert "stubborn" in msg
    # And it MUST identify this as a reporting-tool coverage gap.
    assert "reporting" in msg.lower() or "tool" in msg.lower()


async def test_register_reporting_tools_lists_all_missing_agents() -> None:
    """When several agents fail augmentation, the error names ALL of them.

    A partial coverage gap is hidden if the error only names the first
    missing agent — reviewers fix one and ship, unaware of the others.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS

    class _Stubborn:
        def __init__(self, name: str) -> None:
            self.name = name
            self.sub_agents: list[Any] = []
            self._tools: list[Any] = []

        @property
        def tools(self) -> list[Any]:
            return self._tools

        @tools.setter
        def tools(self, value: Any) -> None:
            raise RuntimeError("stubborn")

    root = _mk("root")
    stubborn_a = _Stubborn("stubborn_a")
    stubborn_b = _Stubborn("stubborn_b")
    root.sub_agents = [stubborn_a, stubborn_b]

    adapter = ADKAdapter(root)
    with pytest.raises(RuntimeError) as excinfo:
        await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))
    msg = str(excinfo.value)
    assert "stubborn_a" in msg and "stubborn_b" in msg


async def test_register_reporting_tools_succeeds_when_every_agent_gets_tools() -> None:
    """Happy path: the integrity check is a pass-through when augmentation
    lands on every reachable agent.
    """
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS, REPORTING_TOOL_NAMES

    child = _mk("child")
    root = _mk("root")
    root.sub_agents = [child]

    adapter = ADKAdapter(root)
    # Must not raise.
    await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))

    # Post-condition: every reachable agent carries every name.
    for agent in (root, child):
        names = {
            getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", None)
            for t in getattr(agent, "tools", None) or ()
        }
        for expected in REPORTING_TOOL_NAMES:
            assert expected in names


async def test_register_reporting_tools_integrity_skipped_in_degraded_mode() -> None:
    """When the caller passes a pre-built Runner, the tail integrity
    check is skipped — the caller may have built a runner shape we
    don't fully control. No assumptions about sub-agent augmentation.
    """
    from google.adk.runners import InMemoryRunner

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS

    agent = _mk("solo")
    runner = InMemoryRunner(agent=agent, app_name="solo")
    adapter = ADKAdapter(runner)
    assert adapter._degraded_prebuilt_runner is True

    # Even though the single agent will still actually receive the
    # tools (augmentation is per-agent, not dependent on the runner
    # path), the integrity gate is unconditionally skipped in degraded
    # mode — regressing that to run would risk unexpected RuntimeErrors
    # for custom pre-built-runner callers.
    await adapter.register_reporting_tools(list(BUILTIN_REPORTING_TOOLS))


# ---------------------------------------------------------------------------
# __init__ plugin-installed integrity check
# ---------------------------------------------------------------------------


async def test_init_raises_when_plugin_cannot_install_on_a_runner() -> None:
    """If ``_register_plugin_on_runner`` silently fails for one agent,
    ``ADKAdapter.__init__`` must raise at wrap time — the plugin
    couldn't install so reporting callbacks, state writes, and
    observability would all be silently broken on that agent.

    We simulate the failure by monkey-patching
    ``_register_plugin_on_runner`` to return False for one specific
    runner (identified by its agent's name).
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters import adk as adk_module

    a = _mk("a")
    coord = _mk("coord")
    coord.tools = [AgentTool(a)]

    orig = adk_module._register_plugin_on_runner

    def _patched(runner: Any, plugin: Any) -> bool:
        # Simulate broken install for the runner whose agent is "a".
        if getattr(getattr(runner, "agent", None), "name", "") == "a":
            return False
        return orig(runner, plugin)

    adk_module._register_plugin_on_runner = _patched
    try:
        with pytest.raises(RuntimeError) as excinfo:
            adk_module.ADKAdapter(coord)
    finally:
        adk_module._register_plugin_on_runner = orig

    msg = str(excinfo.value)
    assert "'a'" in msg or "\"a\"" in msg
    assert "plugin" in msg.lower()


async def test_init_integrity_check_skipped_in_degraded_mode() -> None:
    """The ``__init__`` plugin-installed check is explicitly skipped in
    degraded mode (pre-built Runner) so callers with unconventional
    runner shapes aren't blocked. Regressing to always-check would
    break the graceful-degrade contract.
    """
    from dataclasses import field

    from goldfive.adapters.adk import ADKAdapter

    @dataclasses.dataclass
    class _NoPluginManagerRunner:
        agent: Any = None
        session_service: Any = None
        plugin_manager: Any = None
        app_name: str = "prebuilt"
        plugins: list = field(default_factory=list)

        async def run_async(self, **kwargs: Any):  # noqa: ARG002
            if False:  # pragma: no cover
                yield None

    inner = _mk("solo")
    runner = _NoPluginManagerRunner(agent=inner)
    # Must not raise even though plugin_manager is None — degraded mode
    # skips the integrity gate.
    adapter = ADKAdapter(runner)
    assert adapter._degraded_prebuilt_runner is True


async def test_init_succeeds_on_well_formed_tree() -> None:
    """Regression-guard: the happy path doesn't raise.

    Without this, a flaky refactor of the integrity check (e.g. stricter
    plugin-name comparison) could break wrap-time construction for
    every caller.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk("a")
    b = _mk("b")
    coord = _mk("coord")
    coord.sub_agents = [b]
    coord.tools = [AgentTool(a)]

    # Must not raise.
    adapter = ADKAdapter(coord)
    assert set(adapter._registry) == {"coord", "a", "b"}
