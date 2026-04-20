"""Shared typing + lifecycle helpers for the planner / goal-deriver LLM callable.

Goldfive accepts an opaque ``call_llm(system, user, model) -> str`` async
callable for both :class:`LLMPlanner` and :class:`LLMGoalDeriver`. That
keeps the framework decoupled from any specific SDK, but it also leaves
resource cleanup ambiguous — the most common pattern (an OpenAI
``AsyncClient`` whose ``aiohttp.ClientSession`` lives until garbage
collection) leaks at process exit.

This module standardises a duck-typed close protocol:

* :class:`CallLLM` — a structural :class:`typing.Protocol` describing
  the call signature. Pure documentation; runtime acceptance has always
  been "anything callable with the right shape".
* :class:`ClosableCallLLM` — extends :class:`CallLLM` with an optional
  async ``close()``. Callables that own a network session implement
  ``close``; bare lambdas don't, and that's fine — the runtime probes
  via ``getattr(call_llm, "close", None)``.
* :func:`maybe_close_call_llm` — utility used by :class:`Runner.close`
  to fire the optional ``close`` if present, swallowing exceptions so a
  misbehaving teardown can't hang the process.

There is no breaking change for existing callers: existing call_llm
callables continue to work because they just don't have a ``close``
attribute and the helper short-circuits.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("goldfive.llm")


@runtime_checkable
class CallLLM(Protocol):
    """Async callable shape: ``(system, user, model) -> str``.

    Used by :class:`~goldfive.planner.LLMPlanner` and
    :class:`~goldfive.goal_deriver.LLMGoalDeriver`. The ``model`` argument
    may be empty when the callable is already model-bound.
    """

    async def __call__(self, system: str, user: str, model: str) -> str: ...


@runtime_checkable
class ClosableCallLLM(CallLLM, Protocol):
    """Optional extension: a ``call_llm`` that owns network resources.

    Implementations should define an async ``close()`` that releases the
    underlying HTTP session (e.g. ``await openai_client.close()``).
    Goldfive's :class:`Runner.close` will await it automatically.
    """

    async def close(self) -> None: ...


async def maybe_close_call_llm(call_llm: Any, *, label: str = "call_llm") -> None:
    """Await ``call_llm.close()`` if it exists. Swallow exceptions.

    Returns immediately when ``call_llm`` is ``None`` or has no
    ``close`` attribute. Logs and discards any exception raised by
    ``close`` — Runner teardown must remain robust under partial
    initialisation.
    """
    if call_llm is None:
        return
    closer = getattr(call_llm, "close", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        log.warning("%s.close() raised %s; ignored", label, exc)


__all__ = ["CallLLM", "ClosableCallLLM", "maybe_close_call_llm"]
