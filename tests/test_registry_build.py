"""Registry construction tests for ADKAdapter.

Phase 2 of feat/registry-dispatch-model: the adapter builds a
``name -> agent`` registry at wrap time and uses it to dispatch tasks
per-assignee. These tests pin the registry's traversal contract so a
future refactor can't silently change what's reachable.

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

pytest.importorskip("google.adk")


def _mk_llm_agent(name: str) -> Any:
    """Build a bare ``LlmAgent`` for registry-shape tests."""
    from google.adk.agents.llm_agent import LlmAgent

    return LlmAgent(name=name, model="fake-model", description=name, instruction="x")


@dataclasses.dataclass
class _InnerWrapper:
    """Duck-typed wrapper agent that carries an ``inner_agent`` reference.

    ``LlmAgent`` is a pydantic model and rejects assigning arbitrary
    attributes (including ``inner_agent``), so the wrapper-shape test
    uses a plain dataclass that mimics the goldfive / harmonograf
    wrapper contract (``.name``, ``.inner_agent``, ``.sub_agents``,
    ``.tools``).
    """

    name: str
    inner_agent: Any = None
    sub_agents: list = dataclasses.field(default_factory=list)
    tools: list = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_registry_single_agent_no_children() -> None:
    """A leaf agent with no sub_agents / tools / inner_agent yields a
    single-entry registry."""
    from goldfive.adapters.adk import ADKAdapter

    agent = _mk_llm_agent("solo")
    adapter = ADKAdapter(agent)
    assert set(adapter._registry) == {"solo"}
    assert adapter._registry["solo"] is agent
    assert adapter.available_agents == ["solo"]


def test_registry_flat_sub_agents_tree() -> None:
    """A coordinator with ``sub_agents=[a, b, c]`` yields 4 registry entries."""
    from goldfive.adapters.adk import ADKAdapter

    a = _mk_llm_agent("a")
    b = _mk_llm_agent("b")
    c = _mk_llm_agent("c")
    coord = _mk_llm_agent("coord")
    coord.sub_agents = [a, b, c]

    adapter = ADKAdapter(coord)
    assert set(adapter._registry) == {"coord", "a", "b", "c"}
    # Values are references, not copies.
    assert adapter._registry["a"] is a
    assert adapter._registry["b"] is b
    assert adapter._registry["c"] is c


def test_registry_agent_tool_tree() -> None:
    """A coordinator whose ``tools`` list holds two AgentTools yields 3 registry entries."""
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    a = _mk_llm_agent("a")
    b = _mk_llm_agent("b")
    coord = _mk_llm_agent("coord")
    coord.tools = [AgentTool(a), AgentTool(b)]

    adapter = ADKAdapter(coord)
    assert set(adapter._registry) == {"coord", "a", "b"}
    assert adapter._registry["a"] is a
    assert adapter._registry["b"] is b


def test_registry_deeply_nested_tree() -> None:
    """``coord → AgentTool(researcher) → sub_agents=[web, academic]``.

    All four names land in the registry — the traversal follows
    AgentTool.agent then sub_agents on the wrapped agent.
    """
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    web = _mk_llm_agent("web_searcher")
    academic = _mk_llm_agent("academic_searcher")
    researcher = _mk_llm_agent("researcher")
    researcher.sub_agents = [web, academic]
    coord = _mk_llm_agent("coord")
    coord.tools = [AgentTool(researcher)]

    adapter = ADKAdapter(coord)
    assert set(adapter._registry) == {
        "coord",
        "researcher",
        "web_searcher",
        "academic_searcher",
    }
    assert adapter._registry["web_searcher"] is web
    assert adapter._registry["academic_searcher"] is academic


def test_registry_inner_agent_wrapper_includes_both() -> None:
    """A wrapper agent with ``inner_agent`` exposes BOTH names in the registry.

    This pins the current behavior — :func:`_build_agent_registry`
    traverses the ``inner_agent`` edge without collapsing the wrapper.
    Both agents get runners and become dispatchable; callers that want
    the inner-only dispatch must use the inner agent's name explicitly.
    """
    from goldfive.adapters.adk import _build_agent_registry

    inner = _InnerWrapper(name="real_agent")
    outer = _InnerWrapper(name="wrapper", inner_agent=inner)

    registry = _build_agent_registry(outer)
    assert set(registry) == {"wrapper", "real_agent"}
    assert registry["wrapper"] is outer
    assert registry["real_agent"] is inner


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_registry_raises_on_duplicate_name_sub_agents() -> None:
    """Two agents with the same name under ``sub_agents`` must raise."""
    from goldfive.adapters.adk import ADKAdapter

    dup1 = _mk_llm_agent("duplicate")
    dup2 = _mk_llm_agent("duplicate")
    root = _mk_llm_agent("root")
    root.sub_agents = [dup1, dup2]

    with pytest.raises(ValueError, match="duplicate agent name"):
        ADKAdapter(root)


def test_registry_raises_on_duplicate_name_agent_tools() -> None:
    """Two AgentTools sharing a name also raise — the dispatch layer
    cannot disambiguate."""
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    dup1 = _mk_llm_agent("duplicate")
    dup2 = _mk_llm_agent("duplicate")
    root = _mk_llm_agent("root")
    root.tools = [AgentTool(dup1), AgentTool(dup2)]

    with pytest.raises(ValueError) as excinfo:
        ADKAdapter(root)
    # The error names the colliding agent so a planner bug is
    # diagnosable without a debugger.
    assert "duplicate" in str(excinfo.value)


def test_registry_raises_on_empty_name() -> None:
    """An agent with ``name=""`` cannot be dispatched to — raise."""
    from goldfive.adapters.adk import _build_agent_registry

    bad = _InnerWrapper(name="")
    with pytest.raises(ValueError, match="no usable name"):
        _build_agent_registry(bad)


def test_registry_raises_on_none_name() -> None:
    """An agent whose ``name`` attribute is ``None`` also raises."""
    from goldfive.adapters.adk import _build_agent_registry

    @dataclasses.dataclass
    class _NoName:
        name: Any = None
        sub_agents: list = dataclasses.field(default_factory=list)
        tools: list = dataclasses.field(default_factory=list)

    with pytest.raises(ValueError, match="no usable name"):
        _build_agent_registry(_NoName())


def test_registry_shape_reflected_in_available_agents() -> None:
    """``available_agents`` is the sorted registry keys — pinned public API."""
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    b = _mk_llm_agent("bravo")
    a = _mk_llm_agent("alpha")
    coord = _mk_llm_agent("zulu")
    coord.tools = [AgentTool(a), AgentTool(b)]

    adapter = ADKAdapter(coord)
    # Sorted for deterministic planner output.
    assert adapter.available_agents == ["alpha", "bravo", "zulu"]
