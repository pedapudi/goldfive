"""Agent adapter implementations.

Each adapter bridges an external agent SDK (or a plain async callable) to
the :class:`goldfive.protocols.AgentAdapter` protocol. The CallableAdapter
is the reference implementation — see :mod:`goldfive.adapters.callable`.

Adapters for frameworks with heavy optional dependencies (e.g. ADK in
:mod:`goldfive.adapters.adk`) are *not* imported here — importing a
submodule requires the corresponding framework to be installed
(``pip install goldfive[adk]`` for :mod:`goldfive.adapters.adk`). The
per-module import guard raises :class:`ImportError` with an install
hint when the framework is missing.
"""

from __future__ import annotations

from goldfive.adapters.callable import CallableAdapter

__all__ = ["CallableAdapter"]
