"""Built-in executors.

Each executor drives a :class:`Plan` to completion against an
:class:`AgentAdapter`. See ``INTERFACE_SPEC.md`` and issues #9 / #10
for the contracts pinned here.
"""

from __future__ import annotations

from goldfive.executors.parallel import ParallelDAGExecutor
from goldfive.executors.sequential import SequentialExecutor

__all__ = ["ParallelDAGExecutor", "SequentialExecutor"]
