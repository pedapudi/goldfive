"""Built-in executors.

Each executor drives a :class:`Plan` to completion against an
:class:`AgentAdapter`. See ``INTERFACE_SPEC.md`` for the contract.
"""

from __future__ import annotations

from goldfive.executors.parallel import ParallelDAGExecutor
from goldfive.executors.sequential import SequentialExecutor, build_task_nudge

__all__ = ["ParallelDAGExecutor", "SequentialExecutor", "build_task_nudge"]
