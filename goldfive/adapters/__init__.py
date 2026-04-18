"""Agent adapter implementations.

Each adapter bridges an external agent SDK (or a plain async callable) to
the :class:`goldfive.protocols.AgentAdapter` protocol. The CallableAdapter
is the reference implementation — see :mod:`goldfive.adapters.callable`.
"""

from __future__ import annotations

from goldfive.adapters.callable import CallableAdapter

__all__ = ["CallableAdapter"]
