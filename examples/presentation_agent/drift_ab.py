"""Drift-fidelity A/B harness — native subagent routing vs ``ClaudeAgentSDKLlm``.

Why this exists
---------------
``ClaudeAgentSDKLlm`` (PR #383) reaches a Claude model through a
*text-replay* transport: on every ADK turn it spins up a fresh
``ClaudeSDKClient`` and re-renders the prior conversation as a
descriptive text transcript. A reviewer's standing concern is that this
representation might *manufacture* the very drift goldfive exists to
detect — e.g. a serialised transcript could read as looping/off-topic to
the reasoning judge even when the underlying agent behaviour is fine.

The only honest way to answer that is an apples-to-apples A/B: run the
**same tree** on the **same task** twice, holding everything constant
except the subagent's transport, and compare the *profile of
``DriftDetected`` events* (count and kind) the two arms emit. If the
adapter's profile is not materially worse than the native baseline, the
text-replay transport is not inventing drift.

What this isolates (and what it does not)
-----------------------------------------
The single variable under test is the **subagent transport**:

* ``adapter`` arm — subagent model is ``ClaudeAgentSDKLlm(model=...)``
  (text-replay through claude-agent-sdk).
* ``native`` arm — subagent model is a plain model string handed to
  ADK's own routing (LiteLLM / Gemini message-array transport, no
  text replay).

Everything else is held identical across both arms:

* the agent tree (``_build_agent_tree``),
* the planner / goal-deriver / judge ``call_llm`` (a single shared
  ``make_call_llm`` callable, so the *judges that mint drift* are byte
  for byte the same in both arms),
* the executor, the task, and the number of repetitions.

The **most rigorous** baseline points the native arm at the *same Claude
model* via LiteLLM (``GOLDFIVE_AB_NATIVE_MODEL=anthropic/claude-haiku-4-5``
with ``ANTHROPIC_API_KEY`` set). Then the model weights are identical and
*only* the transport differs — any drift delta is attributable to
text-replay alone. Pointing the native arm at a different model
(e.g. ``gemini-2.5-flash``) answers a looser question ("is the adapter's
drift profile in the same ballpark as a healthy native run?") and
conflates transport with model behaviour; use it only as a sanity check.

Single runs are noisy: drift detection is stochastic (LLM judges, real
tool calls). Run several repetitions per arm (``GOLDFIVE_AB_RUNS``,
default 3) and compare *aggregate* profiles, not one-off runs.

This harness needs the live environment
---------------------------------------
Both arms make real LLM calls. The adapter arm needs claude-agent-sdk
auth (Max/OAuth); the native arm needs whatever its model string
resolves to (an ``ANTHROPIC_API_KEY`` for LiteLLM Claude, a
``GOOGLE_API_KEY`` for Gemini, etc.). It cannot run in CI or an offline
sandbox — run it locally and paste the printed table into the PR.

Usage
-----
::

    # Rigorous: same Claude model both ways, only transport differs.
    ANTHROPIC_API_KEY=sk-ant-... \\
    GOLDFIVE_AB_NATIVE_MODEL=anthropic/claude-haiku-4-5 \\
    GOLDFIVE_AB_ADAPTER_MODEL=claude-haiku-4-5 \\
    GOLDFIVE_AB_RUNS=3 \\
    uv run --extra adk python examples/presentation_agent/drift_ab.py --topic waffles

    # Looser sanity check: native Gemini vs adapter Claude.
    GOOGLE_API_KEY=... GOLDFIVE_AB_NATIVE_MODEL=gemini-2.5-flash \\
    uv run --extra adk python examples/presentation_agent/drift_ab.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter
from typing import Any

import goldfive
from goldfive import LLMGoalDeriver, LLMPlanner, SequentialExecutor

# Reuse the canonical tree + the SDK integration from the example module
# so the A/B exercises exactly the production-shaped agents, not a toy.
from examples.presentation_agent.agent import _build_agent_tree
from goldfive.integrations.claude_sdk import ClaudeAgentSDKLlm, make_call_llm


# ---------------------------------------------------------------------------
# Drift-counting sink
# ---------------------------------------------------------------------------


class _DriftCountingSink:
    """EventSink that tallies ``DriftDetected`` events by kind name.

    Sinks receive pb ``Event`` messages (see ``goldfive.protocols`` and
    ``LoggingSink``). We only care about the ``drift_detected`` payload;
    every other event is counted in ``total_events`` for context and
    otherwise ignored.
    """

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.total_events = 0
        self._drift_kind_name = _make_drift_kind_namer()

    async def emit(self, event: Any) -> None:
        self.total_events += 1
        # pb Event: a oneof named "payload"; drift events set
        # ``drift_detected``. Guard with getattr/WhichOneof so a dataclass
        # event (if a future sink path hands one over) degrades instead of
        # raising.
        which = None
        if hasattr(event, "WhichOneof"):
            which = event.WhichOneof("payload")
        if which == "drift_detected":
            kind_val = event.drift_detected.kind
            self.kinds[self._drift_kind_name(kind_val)] += 1
        elif hasattr(event, "drift_detected") and getattr(
            event.drift_detected, "kind", None
        ):
            kind_val = event.drift_detected.kind
            self.kinds[self._drift_kind_name(kind_val)] += 1

    async def close(self) -> None:
        return None


def _make_drift_kind_namer():
    """Return a fn mapping a DriftKind enum value to its short name.

    Tries the pb enum's ``DriftKind.Name`` first (strips the
    ``DRIFT_KIND_`` prefix); falls back to ``str(value)`` so the harness
    never crashes on an unmapped kind.
    """
    try:
        from goldfive.pb.goldfive.v1 import types_pb2 as pb  # type: ignore

        def namer(value: Any) -> str:
            try:
                name = pb.DriftKind.Name(int(value))
            except Exception:
                return str(value)
            return name[len("DRIFT_KIND_") :] if name.startswith("DRIFT_KIND_") else name

        return namer
    except Exception:
        return lambda value: str(value)


# ---------------------------------------------------------------------------
# One arm = N repetitions of (build tree → wrap → run), drift tallied
# ---------------------------------------------------------------------------


async def _run_arm(
    *,
    arm: str,
    subagent_model: Any,
    topic: str,
    runs: int,
    shared_call_llm: Any,
    planner_model_tag: str,
) -> _DriftCountingSink:
    """Run one arm ``runs`` times, accumulating drift counts in one sink.

    The judges/planner/goal-deriver all route through ``shared_call_llm``
    so the drift-minting machinery is identical across arms — only
    ``subagent_model`` (the transport under test) differs.
    """
    sink = _DriftCountingSink()
    for i in range(runs):
        tree = _build_agent_tree(subagent_model)
        runner = goldfive.wrap(
            tree,
            planner=LLMPlanner(call_llm=shared_call_llm, model=planner_model_tag),
            goal_deriver=LLMGoalDeriver(
                call_llm=shared_call_llm, model=planner_model_tag
            ),
            executor=SequentialExecutor(max_task_invocations=8),
            sinks=[sink],
            # Route judges (reasoning, goal-drift) through the same callable
            # in BOTH arms so drift detection is held constant.
            call_llm=shared_call_llm,
        )
        try:
            outcome = await runner.run(f"Make a short presentation about {topic}.")
            print(
                f"  [{arm}] run {i + 1}/{runs}: "
                f"success={outcome.success} reason={outcome.reason!r}"
            )
        finally:
            await runner.close()
    return sink


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(
    native: _DriftCountingSink,
    adapter: _DriftCountingSink,
    *,
    runs: int,
) -> None:
    all_kinds = sorted(set(native.kinds) | set(adapter.kinds))
    print("\n" + "=" * 64)
    print(f"DRIFT-FIDELITY A/B  ({runs} run(s) per arm)")
    print("=" * 64)
    print(f"{'DriftKind':<34}{'native':>10}{'adapter':>10}{'Δ':>8}")
    print("-" * 64)
    if not all_kinds:
        print("  (no DriftDetected events in either arm)")
    for kind in all_kinds:
        n = native.kinds.get(kind, 0)
        a = adapter.kinds.get(kind, 0)
        print(f"{kind:<34}{n:>10}{a:>10}{a - n:>+8}")
    print("-" * 64)
    nt, at = sum(native.kinds.values()), sum(adapter.kinds.values())
    print(f"{'TOTAL drift events':<34}{nt:>10}{at:>10}{at - nt:>+8}")
    print(
        f"{'events seen (all kinds)':<34}"
        f"{native.total_events:>10}{adapter.total_events:>10}"
    )
    print("=" * 64)
    # A plain-language verdict the operator can paste into the PR. The bar
    # the reviewer set: the adapter's drift profile must not be
    # *materially worse* than native.
    delta = at - nt
    per_run = delta / runs if runs else 0.0
    print(
        "\nVerdict heuristic: the adapter adds "
        f"{delta:+d} total drift events vs native "
        f"({per_run:+.1f} per run). Inspect the per-kind rows above — a "
        "spike concentrated in LOOPING_REASONING / OFF_TOPIC / "
        "CONFABULATION_RISK is the signature of the text-replay transport "
        "manufacturing drift; a roughly flat profile means it is not.\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _main_async(topic: str, runs: int) -> None:
    native_model = os.environ.get("GOLDFIVE_AB_NATIVE_MODEL", "anthropic/claude-haiku-4-5")
    adapter_model = os.environ.get("GOLDFIVE_AB_ADAPTER_MODEL", "claude-haiku-4-5")
    planner_model = os.environ.get("GOLDFIVE_AB_PLANNER_MODEL", "claude-haiku-4-5")

    # One shared planner/goal/judge callable, used by BOTH arms so the
    # drift-minting path is constant and the subagent transport is the
    # sole independent variable.
    shared_call_llm = make_call_llm(planner_model)

    print(
        f"native arm subagent model : {native_model!r} (ADK native routing)\n"
        f"adapter arm subagent model: {adapter_model!r} (ClaudeAgentSDKLlm text-replay)\n"
        f"shared planner/judge model: {planner_model!r}\n"
        f"topic={topic!r}  runs/arm={runs}\n"
    )

    print("Running NATIVE arm ...")
    native = await _run_arm(
        arm="native",
        subagent_model=native_model,
        topic=topic,
        runs=runs,
        shared_call_llm=shared_call_llm,
        planner_model_tag=planner_model,
    )

    print("Running ADAPTER arm ...")
    adapter = await _run_arm(
        arm="adapter",
        subagent_model=ClaudeAgentSDKLlm(model=adapter_model),
        topic=topic,
        runs=runs,
        shared_call_llm=shared_call_llm,
        planner_model_tag=planner_model,
    )

    _print_report(native, adapter, runs=runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="waffles", help="Presentation topic.")
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("GOLDFIVE_AB_RUNS", "3")),
        help="Repetitions per arm (drift is stochastic; average several).",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args.topic, args.runs))


if __name__ == "__main__":
    main()
