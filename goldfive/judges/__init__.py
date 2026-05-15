"""Pluggable judges — the public extension point for goldfive verdicts.

Historically goldfive shipped a closed set of drift detectors hardcoded
into the steerer (reasoning-drift, goal-drift, tool-loops, refusal,
context-pressure). That works for the "stay-on-target" mission but
locks downstream consumers — optimizers, evaluators, cost / latency
trackers — to goldfive's universal drift taxonomy.

This module exposes a :class:`Judge` extension point so operators can
register agent-specific quality signals alongside (or instead of) the
built-in drift detectors. Judges return :class:`JudgeVerdict` payloads
in one of four flavours:

* **drift** — back-compat with the existing detectors. The verdict
  fires both a :class:`DriftDetected` envelope (as before) AND the new
  :class:`JudgementEmitted` envelope so consumers can join on
  ``judge_name`` without parsing drift kinds.
* **rubric** — a multi-dimensional score (overall + per-dimension)
  for human-style grading: format adherence, slide structure,
  evidence quality, etc.
* **boolean** — a pass / fail verdict for hard contracts (e.g. "did
  the agent stay under the cost budget?").
* **numeric** — a single named metric (latency_ms, cost_usd,
  tokens_out) that consumers can chart over time without inventing
  a side-channel.

Operators wire a custom judge list via :func:`goldfive.wrap`::

    runner = goldfive.wrap(
        agent,
        judges=[
            goldfive.builtin_judges.reasoning_drift(),
            goldfive.builtin_judges.goal_drift(),
            MyCustomLengthJudge(),
        ],
    )

The built-in judges in :mod:`goldfive.builtin_judges` are thin wrappers
around the existing detectors. They emit ``DriftDetected`` exactly as
the pre-judges code path did (so the v0.1 wire contract is preserved)
and additionally publish ``JudgementEmitted`` with
``verdict_kind = "drift"``.
"""

from __future__ import annotations

from goldfive.judges.base import Judge, JudgeContext, JudgeVerdict

__all__ = [
    "Judge",
    "JudgeContext",
    "JudgeVerdict",
]
