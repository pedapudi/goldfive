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

Per-call ``max_output_tokens`` budget (goldfive#271 follow-up)
--------------------------------------------------------------

The ``call_llm`` signature is opaque ``(system, user, model) -> str`` —
adding a ``max_tokens`` parameter would be a breaking change for
user-supplied callables. Instead, goldfive's own consumers (planner,
goal_deriver, judges, reflective check) set a per-callsite cap via
:data:`MAX_OUTPUT_TOKENS_VAR` (a :class:`contextvars.ContextVar`)
immediately before ``await call_llm(...)``. The default ADK / OpenAI
builders in :mod:`goldfive._llm_detect` and :mod:`goldfive.convenience`
read the var and forward it as ``max_output_tokens`` /
``max_completion_tokens`` on the underlying client call.

User-supplied ``call_llm`` callables can opt in by reading
:func:`get_max_output_tokens` themselves. They are not required to —
the only effect of ignoring the var is that the LLM continues to emit
to its natural stop, the very behaviour that caused 9.6-minute /
5.3-minute calls in goldfive#271 evidence. Setting the cap on the
default builders restores sane wall-clock budgets without touching
caller code.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("goldfive.llm")


# ---------------------------------------------------------------------------
# Per-callsite ``max_output_tokens`` budget (goldfive#271 follow-up)
# ---------------------------------------------------------------------------

# Default cap when no consumer-specific override is in effect. 4096 is
# generous enough for the largest goldfive-internal call (refine /
# generate plan), while still bounding wall-clock at typical Q4 tps
# (~17 tok/sec → ~4 minutes worst case). Worst-case before this var:
# 9.6 minutes (9961 tokens, demo-v8.log).
DEFAULT_MAX_OUTPUT_TOKENS: int = 4096

#: ContextVar carrying the per-callsite cap. ``None`` means "no
#: explicit cap" — the default ADK / OpenAI builder falls back to
#: :data:`DEFAULT_MAX_OUTPUT_TOKENS`. User-supplied ``call_llm``
#: callables may inspect this via :func:`get_max_output_tokens`.
MAX_OUTPUT_TOKENS_VAR: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "goldfive_call_llm_max_output_tokens", default=None
)


def get_max_output_tokens() -> int:
    """Return the per-callsite cap, or :data:`DEFAULT_MAX_OUTPUT_TOKENS`.

    Always returns a positive int. Used by the default ADK / OpenAI
    builders inside the ``call_llm`` body so the underlying SDK call
    receives a finite cap on every dispatch.
    """
    cap = MAX_OUTPUT_TOKENS_VAR.get()
    if cap is None or cap <= 0:
        return DEFAULT_MAX_OUTPUT_TOKENS
    return int(cap)


@contextmanager
def call_llm_budget(max_output_tokens: int | None) -> Iterator[None]:
    """Set :data:`MAX_OUTPUT_TOKENS_VAR` for the duration of the with-block.

    Used by goldfive consumers (planner / goal_deriver / judges /
    reflective check) to bind a per-callsite cap around
    ``await call_llm(...)``. ``None`` resets to no-cap (default applied
    by the builders). Restores the prior value on exit even if the body
    raises.

    Sizing note (Qwen 3.5 thinking models)
    --------------------------------------
    Qwen 3.5 thinking models combine ``<think>`` reasoning and the final
    answer under a single ``max_output_tokens`` ceiling. A judge prompt
    that returns a ~100-300 token JSON verdict still requires several
    thousand tokens of reasoning headroom on the 35B variant; capping
    at 2048 produced empty (``raw=''``) responses on v16 because the
    model exhausted its budget inside the think block before emitting
    a single JSON byte. Goldfive's consumer caps therefore budget 16k
    (judges / reflective check / planner) or 8k (goal deriver) to
    leave ample room for both the think prelude and the structured
    answer. The wall-clock backstop lives in
    :data:`goldfive.adapters._adk_plugin.DEFAULT_LLM_CALL_TIMEOUT_MS`.
    """
    token = MAX_OUTPUT_TOKENS_VAR.set(max_output_tokens)
    try:
        yield
    finally:
        MAX_OUTPUT_TOKENS_VAR.reset(token)


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


__all__ = [
    "CallLLM",
    "ClosableCallLLM",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "MAX_OUTPUT_TOKENS_VAR",
    "call_llm_budget",
    "get_max_output_tokens",
    "maybe_close_call_llm",
]
