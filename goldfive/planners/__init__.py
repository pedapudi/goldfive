"""Goldfive ADK ``BasePlanner`` subclasses.

This is the structural steering layer (goldfive#153). The module
:mod:`~goldfive.planners.goldfive_planner` ships
:class:`~goldfive.planners.goldfive_planner.GoldfivePlanner` — an ADK
``BasePlanner`` subclass that injects a tree-agnostic orchestration
context block on every LLM call and performs structural
post-processing on the response.

The planner is auto-attached to every ``LlmAgent`` reachable from the
wrapped root by :func:`goldfive.wrap`. Agents carrying an explicit
``_goldfive_planner_opt_out = True`` marker are skipped. When a user
has already set a planner on an agent, :class:`GoldfivePlanner` wraps
it via ``user_planner=`` and delegates for composition — goldfive's
instruction is appended to the user planner's, and the user planner's
response post-processing runs after goldfive's structural filters.

Kept as a package (``goldfive.planners``) rather than a flat module so
future work (#154 goal-aware refine, any subsequent planner family)
can drop siblings here without reshuffling imports.
"""

from __future__ import annotations

try:
    from goldfive.planners.goldfive_planner import GoldfivePlanner
except ImportError:  # pragma: no cover — optional dep (ADK) missing
    GoldfivePlanner = None  # type: ignore[assignment,misc]

__all__ = ["GoldfivePlanner"]
