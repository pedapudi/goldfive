"""Markdown-backed prompt catalog for goldfive's steering surface.

Each prompt body that drives a goldfive judge / refine / goal-derive call
is mirrored as a markdown file under
``goldfive/optimization/prompts/``. The markdown file carries an
optimizer-readable header describing the prompt's role and required
placeholders, followed by a ``---`` separator, followed by the prompt
body verbatim.

This module exposes:

* :func:`load` — read the prompt body for a name (cached).
* :func:`bind` — install an optimizer-supplied override that subsequent
  :func:`load` calls return instead of the on-disk text. Useful for
  in-process A/B during optimization without touching the markdown
  files.
* :func:`reset` — drop overrides + the in-memory cache for a name (or
  for every name when called without args). Tests use this between
  cases to recover a clean slate.
* :func:`available_prompts` — list the prompt names this package ships
  with.

The Python module-level constants that the drift / planner / goal-deriver
modules read at runtime are NOT replaced by these markdown files
(retaining byte-identical semantics for the existing test corpus). The
markdown copy is the optimizer-facing artifact; the Python attribute is
the runtime source of truth. Optimizers that want to mutate runtime
behaviour also patch the Python attribute (the manifest names the path).

Loader contract
---------------
The file format is::

    # Title line
    ...optional header text describing the prompt...
    Required placeholders: ...

    ---
    <prompt body>

:func:`load` returns the bytes between the first ``\\n---\\n`` separator
and EOF, minus exactly one trailing newline if present. The convention
when AUTHORING a prompt file is to write the body byte-for-byte and
then append a single ``\\n`` (the editor's customary trailing newline)
— :func:`load` strips that one newline so the returned string is
byte-identical to the original Python constant.
"""

from __future__ import annotations

import importlib.resources as _resources
import threading
from collections.abc import Iterable
from typing import Final

__all__ = [
    "available_prompts",
    "bind",
    "load",
    "reset",
    "PromptNotFound",
]


_LOCK: Final[threading.RLock] = threading.RLock()
_CACHE: dict[str, str] = {}
_OVERRIDES: dict[str, str] = {}

# Source-of-truth list of prompts shipped under
# ``goldfive/optimization/prompts/``. Adding a new prompt: drop the
# markdown file in the directory and append the bare name here. Keeping
# this list explicit (rather than scanning the directory) makes the
# catalog deterministic across installation layouts (wheel install,
# editable install, zipapp) and lets the manifest validator confirm
# every referenced prompt name resolves without performing filesystem
# discovery.
_PROMPT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "reasoning_judge_system",
        "reasoning_judge_user",
        "reasoning_judge_agent_tree_suffix",
        "goal_drift_system",
        "goal_drift_user",
        "reflective_check_system",
        "reflective_check_user",
        "goal_derive_system",
        "plan_generate_system",
        "looping_tool_call_system",
        "user_steer_system",
        "plan_divergence_system",
        "refine_system",
        # manifest-and-decision-telemetry: plan-template fragments
        # embedded in the four refine prompts above. Tuning here moves
        # supersession precision / recall without touching drift-
        # specific prompts.
        "plan_template_supersession_invariant",
        "plan_template_supersession_examples",
        "plan_template_refinement_guidance",
    }
)


class PromptNotFound(KeyError):
    """Raised by :func:`load` when ``name`` does not match a shipped prompt."""


def available_prompts() -> tuple[str, ...]:
    """Return the sorted tuple of prompt names this package ships with."""
    return tuple(sorted(_PROMPT_NAMES))


def _read_disk(name: str) -> str:
    """Read ``name``.md from the package and return the prompt body.

    Strips everything up to and including the first ``\\n---\\n``
    separator. Strips exactly one trailing ``\\n`` from the remaining
    bytes when present.
    """
    try:
        text = _resources.files(__package__).joinpath(
            f"prompts/{name}.md"
        ).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptNotFound(name) from exc
    sep = "\n---\n"
    idx = text.find(sep)
    if idx == -1:
        # Tolerate header-less files: the whole file is the prompt body.
        body = text
    else:
        body = text[idx + len(sep) :]
    if body.endswith("\n"):
        body = body[:-1]
    return body


def load(name: str) -> str:
    """Return the prompt body for ``name`` (cached, override-aware).

    Lookup precedence:

    1. An in-process override installed via :func:`bind` — wins
       unconditionally. Useful for A/B tests + in-process optimization.
    2. The cached on-disk parse from a previous :func:`load`.
    3. A fresh on-disk parse.

    Raises :class:`PromptNotFound` when ``name`` is not in
    :func:`available_prompts`.
    """
    if name not in _PROMPT_NAMES:
        raise PromptNotFound(name)
    with _LOCK:
        if name in _OVERRIDES:
            return _OVERRIDES[name]
        cached = _CACHE.get(name)
        if cached is not None:
            return cached
        body = _read_disk(name)
        _CACHE[name] = body
        return body


def bind(name: str, value: str) -> None:
    """Install an in-process override for ``name``.

    Subsequent :func:`load` calls return ``value`` directly without
    consulting the on-disk file. Designed for the manifest-driven
    mutation flow: an optimizer proposes a new prompt body, validates
    it against the manifest, and then installs it via :func:`bind` so
    every consumer that routes through :func:`load` sees the new text
    without restarting the process.

    Note: the Python module-level prompt constants (e.g.
    ``REASONING_DRIFT_SYSTEM_PROMPT``) are NOT mutated by this call.
    The runtime drift / planner / goal-deriver paths still read those
    constants directly. Optimizers that need to swing the live
    behaviour also patch the constant (the manifest names the
    attribute path).
    """
    if name not in _PROMPT_NAMES:
        raise PromptNotFound(name)
    with _LOCK:
        _OVERRIDES[name] = value


def reset(name: str | None = None) -> None:
    """Drop the in-process override (and cache entry) for ``name``.

    When ``name`` is ``None``, drop EVERY override and cache entry —
    the next :func:`load` call re-reads from disk. Use between test
    cases that mutate the catalog so each case starts from a clean
    slate.
    """
    with _LOCK:
        if name is None:
            _OVERRIDES.clear()
            _CACHE.clear()
            return
        _OVERRIDES.pop(name, None)
        _CACHE.pop(name, None)


def _iter_names() -> Iterable[str]:
    """Iterator over the shipped prompt names. Sugar for the manifest tests."""
    return iter(sorted(_PROMPT_NAMES))
