"""Goal derivation strategies.

Goldfive introduces an explicit ``GoalDeriver`` abstraction on top of the
harmonograf design. The deriver converts a user's free-form request into a
concrete ``list[Goal]``. Plans are then generated from goals, and drift
detection is measured against those same goals — so clean, explicit goals are
the foundation for every downstream decision.

Three variants are provided:

* :class:`PassthroughGoalDeriver` — for callers that already know the goals.
  Accepts a single summary string, a list of strings, or a list of ``Goal``
  objects and returns them verbatim from :meth:`derive`.
* :class:`LiteralGoalDeriver` — trivial wrapper that turns the incoming
  ``user_input`` string into a single ``Goal``. Handy for tests and simple
  CLIs where no LLM is desired.
* :class:`LLMGoalDeriver` — asks an LLM to extract one-or-more goals in JSON,
  parses the response, and falls back to a single passthrough goal on error.

All variants implement the ``GoalDeriver`` protocol (issue #5). The protocol is
structural, so explicit inheritance is not required; each class exposes the
expected ``async def derive(...)`` signature.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from goldfive.types import Goal

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Default system prompt for the LLM-backed deriver.
#
# Design notes:
# * The prompt is intentionally short. Agents that call this module have their
#   own, richer system prompts; the deriver's job is narrow: turn the raw user
#   request into structured goals. We keep guidance here focused on SHAPE
#   (JSON schema) rather than content.
# * We ask for one or more goals. A request may legitimately contain several
#   independent outcomes ("ship the feature AND write the blog post"); the
#   planner is responsible for ordering them.
# * The ``"id"`` values are authored by the LLM. We do NOT enforce a specific
#   scheme (e.g. ``"g1"``, ``"g2"``); the only requirement is that they be
#   unique within a response. Callers that need stable IDs can normalise them
#   after parsing.
# * We explicitly ask for a bare JSON object (no Markdown fences), but still
#   apply fence-stripping on parse to be robust to models that ignore that
#   instruction.
# -----------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You extract explicit goals from a user's request.

Return a JSON object of the form:
{"goals": [{"id": "g1", "summary": "..."}, ...]}

Rules:
- Produce one or more goals. Prefer a single goal unless the user has clearly
  asked for multiple, independent outcomes.
- Each ``summary`` should describe what "done" looks like, in one sentence.
- Each ``id`` must be unique within the response (e.g. "g1", "g2", ...).
- Respond with JSON only — no prose, no Markdown fences.
""".strip()


# -----------------------------------------------------------------------------
# Fence-stripping helper. Mirrors the convention used by ``LLMPlanner`` so the
# behaviour is consistent across modules.
# -----------------------------------------------------------------------------
_FENCE_RE = re.compile(
    r"""
    ^\s*                    # leading whitespace
    ```(?:[a-zA-Z0-9_+-]*)? # opening fence with optional language tag
    \s*\n                   # newline after fence
    (?P<body>.*?)           # the actual content (non-greedy)
    \n\s*```                # closing fence
    \s*$                    # trailing whitespace
    """,
    re.DOTALL | re.VERBOSE,
)


def _strip_fences(text: str) -> str:
    """Strip a single outer Markdown code fence, if present.

    Handles ``` with or without a language tag (``json``, ``JSON``, etc.).
    If no fence is present the text is returned unchanged (trimmed).
    """
    match = _FENCE_RE.match(text)
    if match is not None:
        return match.group("body").strip()
    return text.strip()


# -----------------------------------------------------------------------------
# PassthroughGoalDeriver
# -----------------------------------------------------------------------------
class PassthroughGoalDeriver:
    """Returns a pre-configured list of goals, ignoring ``user_input``.

    Accepts any of:
    * A single summary string — wrapped as ``[Goal(id="g1", summary=...)]``.
    * A list of summary strings — each wrapped as ``Goal(id="gN", summary=...)``.
    * A list of :class:`Goal` objects — returned verbatim.

    Useful when the caller has already decided what the goals are (e.g. a
    programmatic invocation or a test fixture) and only wants to use the rest
    of the goldfive pipeline (planning, execution, drift detection).
    """

    def __init__(self, goals: str | list[str] | list[Goal]) -> None:
        self._goals: list[Goal] = _coerce_goals(goals)

    async def derive(
        self,
        user_input: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[Goal]:
        """Return the pre-configured goals. ``user_input`` is ignored."""
        # Defensive copy so callers cannot mutate the internal list.
        return list(self._goals)


def _coerce_goals(goals: str | list[str] | list[Goal]) -> list[Goal]:
    """Normalise the constructor argument into a ``list[Goal]``."""
    if isinstance(goals, str):
        if not goals.strip():
            raise ValueError("PassthroughGoalDeriver: empty goal summary string")
        return [Goal(id="g1", summary=goals)]

    if not isinstance(goals, list):
        raise TypeError(
            f"PassthroughGoalDeriver: expected str | list[str] | list[Goal], "
            f"got {type(goals).__name__}"
        )

    if len(goals) == 0:
        raise ValueError("PassthroughGoalDeriver: empty goals list")

    out: list[Goal] = []
    for i, item in enumerate(goals, start=1):
        if isinstance(item, Goal):
            out.append(item)
        elif isinstance(item, str):
            if not item.strip():
                raise ValueError(
                    f"PassthroughGoalDeriver: empty summary string at index {i - 1}"
                )
            out.append(Goal(id=f"g{i}", summary=item))
        else:
            raise TypeError(
                f"PassthroughGoalDeriver: list items must be str or Goal, "
                f"got {type(item).__name__} at index {i - 1}"
            )
    return out


# -----------------------------------------------------------------------------
# LiteralGoalDeriver
# -----------------------------------------------------------------------------
class LiteralGoalDeriver:
    """Wraps ``user_input`` as a single goal, verbatim.

    The simplest non-trivial deriver: if the caller hands us a non-empty
    string, we emit ``[Goal(id="g1", summary=user_input)]``. If the string is
    empty or whitespace-only we raise ``ValueError``.

    Unlike :class:`PassthroughGoalDeriver`, this deriver is configured with
    nothing — it always reflects the ``user_input`` passed to :meth:`derive`.
    """

    async def derive(
        self,
        user_input: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[Goal]:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("LiteralGoalDeriver: user_input must be a non-empty string")
        return [Goal(id="g1", summary=user_input)]


# -----------------------------------------------------------------------------
# LLMGoalDeriver
# -----------------------------------------------------------------------------
class LLMGoalDeriver:
    """Uses an LLM to extract explicit goals from free-form user input.

    The LLM is prompted to return JSON of the shape
    ``{"goals": [{"id": "...", "summary": "..."}, ...]}``. On any failure
    (network error, non-JSON response, wrong schema) we fall back to a single
    passthrough goal: ``[Goal(id="g1", summary=user_input)]``. The fallback
    is logged at WARNING level so the caller can spot misconfiguration.

    Args:
        call_llm: Async callable ``(system, prompt, model) -> str`` that
            performs the LLM call and returns the assistant text. Matches the
            signature used by :class:`LLMPlanner` (issue #7) so the same
            adapter helper can be reused.
        model: Model identifier passed to ``call_llm``. Defaults to the empty
            string, which lets the adapter pick its own default.
        system_prompt: Optional override for the default system prompt. The
            default is :data:`DEFAULT_SYSTEM_PROMPT`.
    """

    def __init__(
        self,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str = "",
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._call_llm = call_llm
        self._model = model
        self._system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    async def derive(
        self,
        user_input: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[Goal]:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("LLMGoalDeriver: user_input must be a non-empty string")

        prompt = self._build_prompt(user_input, context)

        try:
            raw = await self._call_llm(self._system_prompt, prompt, self._model)
        except Exception as e:  # pragma: no cover - exercised via tests w/ stub
            logger.warning(
                "LLMGoalDeriver: call_llm raised %s; falling back to passthrough goal", e
            )
            return [Goal(id="g1", summary=user_input)]

        try:
            goals = _parse_goals_response(raw)
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "LLMGoalDeriver: could not parse LLM response (%s); "
                "falling back to passthrough goal",
                e,
            )
            return [Goal(id="g1", summary=user_input)]

        if not goals:
            logger.warning(
                "LLMGoalDeriver: LLM returned zero goals; falling back to passthrough goal"
            )
            return [Goal(id="g1", summary=user_input)]

        return goals

    # ---- internals --------------------------------------------------------

    def _build_prompt(
        self,
        user_input: str,
        context: Mapping[str, Any] | None,
    ) -> str:
        """Build the user-side prompt. Context, if present, is appended as a
        short JSON block so the LLM can condition on it (e.g. prior goals)."""
        parts = [f"User request:\n{user_input}"]
        if context:
            try:
                ctx_json = json.dumps(dict(context), default=str, indent=2)
            except (TypeError, ValueError):
                ctx_json = str(context)
            parts.append(f"Context:\n{ctx_json}")
        parts.append(
            'Return JSON: {"goals": [{"id": "g1", "summary": "..."}, ...]}.'
        )
        return "\n\n".join(parts)


def _parse_goals_response(raw: str) -> list[Goal]:
    """Parse an LLM response into a list of ``Goal`` objects.

    Applies fence-stripping before JSON parsing. Raises ``ValueError`` if the
    response does not match the expected schema.
    """
    if not isinstance(raw, str):
        raise ValueError(f"expected str response, got {type(raw).__name__}")

    stripped = _strip_fences(raw)
    if not stripped:
        raise ValueError("empty response")

    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")

    goals_raw = data.get("goals")
    if not isinstance(goals_raw, list):
        raise ValueError('missing or non-list "goals" field')

    goals: list[Goal] = []
    for i, item in enumerate(goals_raw):
        if not isinstance(item, dict):
            raise ValueError(f"goals[{i}] is not an object")
        summary = item.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f'goals[{i}] missing non-empty "summary"')
        gid = item.get("id")
        if not isinstance(gid, str) or not gid.strip():
            # Auto-assign an id when the LLM forgot one. This is less harsh
            # than rejecting the whole response.
            gid = f"g{i + 1}"
        metadata_raw = item.get("metadata", {})
        metadata: dict[str, str]
        if isinstance(metadata_raw, dict):
            # Coerce values to str for type stability. metadata is declared as
            # ``dict[str, str]`` on the dataclass.
            metadata = {str(k): str(v) for k, v in metadata_raw.items()}
        else:
            metadata = {}
        goals.append(Goal(id=gid, summary=summary.strip(), metadata=metadata))

    return goals


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LLMGoalDeriver",
    "LiteralGoalDeriver",
    "PassthroughGoalDeriver",
]
