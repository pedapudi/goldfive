"""Built-in executors.

Each executor drives a :class:`Plan` to completion against an
:class:`AgentAdapter`. See ``INTERFACE_SPEC.md`` and issue #10
(``ParallelDAGExecutor``) for the contract.
"""

from __future__ import annotations

from goldfive.executors.parallel import ParallelDAGExecutor

__all__ = ["ParallelDAGExecutor"]
