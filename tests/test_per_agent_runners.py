"""Per-agent runner wiring tests.

Phase 2 of feat/registry-dispatch-model: every registry agent gets its
own ``InMemoryRunner`` and they all share the SAME goldfive plugin
instance. That shared-instance contract is what lets ``set_active_context``
on the parent plugin make the same ctx visible to AgentTool sub-Runners
(since ADK's AgentTool propagates the parent's plugin_manager to the
sub-Runner).

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")


def _mk(name: str) -> Any:
    from google.adk.agents.llm_agent import LlmAgent

    return LlmAgent(name=name, model="fake-model", description=name, instruction="x")


def _plugin_names(runner: Any) -> list[str]:
    pm = getattr(runner, "plugin_manager", None)
    if pm is None:
        return []
    return [str(getattr(p, "name", "")) for p in getattr(pm, "plugins", []) or ()]


# ---------------------------------------------------------------------------
# Runner-per-registry-entry invariants
# ---------------------------------------------------------------------------


def test_one_runner_per_registry_entry_single_agent() -> None:
    """Single-agent wrap → exactly one runner in ``_runners``."""
    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(_mk("solo"))
    assert set(adapter._runners) == {"solo"}
    assert len(adapter._runners) == len(adapter._registry) == 1


def test_one_runner_per_registry_entry_tree() -> None:
    """Coordinator + two AgentTool specialists → three runners total."""
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk("a")
    b = _mk("b")
    coord = _mk("coord")
    coord.tools = [AgentTool(a), AgentTool(b)]

    adapter = ADKAdapter(coord)
    assert set(adapter._runners) == {"coord", "a", "b"}
    assert len(adapter._runners) == len(adapter._registry) == 3


def test_root_agent_runner_reuses_legacy_runner_attr() -> None:
    """The root agent's entry in ``_runners`` is the same object as the
    legacy ``adapter._runner`` attribute.

    The adapter intentionally aliases these so existing tests that
    monkey-patch ``adapter._runner`` still work. Regressing this would
    break test_adk_adapter.py's heal-path tests.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk("a")
    coord = _mk("coord")
    coord.tools = [AgentTool(a)]

    adapter = ADKAdapter(coord)
    assert adapter._runners["coord"] is adapter._runner


# ---------------------------------------------------------------------------
# Plugin-install invariants
# ---------------------------------------------------------------------------


def test_plugin_installed_exactly_once_per_runner() -> None:
    """Each runner carries EXACTLY ONE copy of the goldfive plugin —
    accidental double-registration would fire callbacks twice per event,
    producing duplicate DriftEvents / duplicate state-protocol writes /
    duplicate reporting-tool handlers.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk("a")
    b = _mk("b")
    coord = _mk("coord")
    coord.sub_agents = [b]
    coord.tools = [AgentTool(a)]

    adapter = ADKAdapter(coord)
    goldfive_plugin_name = adapter._plugin.name

    for agent_name, runner in adapter._runners.items():
        installed = _plugin_names(runner)
        count = installed.count(goldfive_plugin_name)
        assert count == 1, (
            f"runner for {agent_name!r} has {count} goldfive plugin(s); "
            f"plugins={installed}"
        )


def test_plugin_is_same_instance_across_runners() -> None:
    """All per-agent runners share the SAME plugin instance.

    This is the load-bearing invariant for state-protocol propagation:
    ``set_active_context`` on the plugin makes the same ctx visible to
    AgentTool sub-Runners because they read off the same ``_active_ctx``
    attribute. A regression that creates a fresh plugin per runner would
    silently break state-protocol propagation through AgentTool.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk("a")
    b = _mk("b")
    c = _mk("c")
    coord = _mk("coord")
    coord.sub_agents = [b]
    coord.tools = [AgentTool(a), AgentTool(c)]

    adapter = ADKAdapter(coord)

    plugins_seen: list[Any] = []
    for runner in adapter._runners.values():
        pm = getattr(runner, "plugin_manager", None)
        for p in getattr(pm, "plugins", []) or ():
            if getattr(p, "name", "") == adapter._plugin.name:
                plugins_seen.append(p)

    # Every runner contributed exactly one goldfive plugin, and all
    # references point at the same object.
    assert len(plugins_seen) == len(adapter._runners)
    first = plugins_seen[0]
    for other in plugins_seen[1:]:
        assert other is first, (
            "per-agent runners MUST share a single goldfive plugin instance "
            "— the state-protocol handoff via plugin.set_active_context "
            "depends on it."
        )


def test_per_agent_runner_agent_is_registry_agent_not_root() -> None:
    """Each per-agent runner's ``.agent`` attribute must be the registry
    agent it was built for — NOT the tree root.

    Regressing this (e.g. a bug that built every runner around the root
    agent) would silently make every dispatch actually drive the root's
    LLM with the root's tools, even though the adapter's ``_resolve_runner_for_task``
    picks the "right" runner dict entry. The asymmetry is invisible
    until the coordinator starts burning through its 500-call ceiling.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk("a")
    b = _mk("b")
    coord = _mk("coord")
    coord.tools = [AgentTool(a), AgentTool(b)]

    adapter = ADKAdapter(coord)
    for name, runner in adapter._runners.items():
        wrapped = getattr(runner, "agent", None)
        assert wrapped is adapter._registry[name], (
            f"runner for {name!r}.agent is {getattr(wrapped, 'name', '?')!r}, "
            f"expected the registry agent {name!r}"
        )


def test_per_agent_runner_has_its_own_app_name() -> None:
    """Each per-agent ``InMemoryRunner`` is built with its own ``app_name``
    (the agent's name). ADK looks sessions up by ``app_name`` — if every
    per-agent runner shared one, sessions would bleed between dispatch
    targets. See the "Session not found" dispatch regression note in
    :meth:`ADKAdapter._ensure_session_for`.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk("a")
    coord = _mk("coord")
    coord.tools = [AgentTool(a)]

    adapter = ADKAdapter(coord)
    # Non-root runner must have its own app_name.
    sub_runner = adapter._runners["a"]
    assert str(getattr(sub_runner, "app_name", "") or "") == "a"
    # Root runner carries the root agent's name.
    root_runner = adapter._runners["coord"]
    assert str(getattr(root_runner, "app_name", "") or "") == "coord"


# ---------------------------------------------------------------------------
# Degraded (pre-built runner) mode
# ---------------------------------------------------------------------------


def test_degraded_prebuilt_runner_yields_single_entry_registry() -> None:
    """When the caller passes a pre-built ``InMemoryRunner``, ``_runners``
    has exactly one entry pointing at that runner — no per-agent expansion."""
    from google.adk.runners import InMemoryRunner

    from goldfive.adapters.adk import ADKAdapter

    inner = _mk("solo_inner")
    prebuilt = InMemoryRunner(agent=inner, app_name="solo_inner")
    adapter = ADKAdapter(prebuilt)

    assert adapter._degraded_prebuilt_runner is True
    assert set(adapter._runners) == {"solo_inner"}
    assert adapter._runners["solo_inner"] is prebuilt
