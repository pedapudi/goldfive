"""Full multi-agent ADK presentation reference example.

See ``agent.py`` for the coordinator + research + web_developer +
reviewer + debugger tree and the goldfive wrapping.

Re-exports:

* ``root_agent`` — the plain ADK coordinator ``BaseAgent`` (pre-wrap).
* ``_build_agent_tree`` — factory that constructs the five-agent tree
  for a given model. Used by tests to swap in a mock ``BaseLlm``.
* ``app`` — lazy PEP 562 attribute that builds the ``adk web`` ``App``
  with ``goldfive.wrap(root_agent, ...)`` as the root agent. Import is
  side-effect free; ``app`` construction runs on first access.
"""

from __future__ import annotations

from typing import Any

from examples.presentation_agent.agent import _build_agent_tree, root_agent


def __getattr__(name: str) -> Any:
    if name == "app":
        from examples.presentation_agent.agent import app as _app

        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["_build_agent_tree", "root_agent", "app"]
