"""Diagnostic for the all-thought-no-answer LLM-call failure mode
(goldfive#271 follow-up to #311).

Pre-fix: ``_call_llm`` silently dropped every ``thought=True`` part and
returned ``"".join(answer_parts).strip()``. When all parts were thought
parts, the function returned ``""`` — indistinguishable from "the model
truly produced empty output" or "the network ate the response". Two
days were lost to that ambiguity in the v16 / Qwen 35B investigation.

Post-fix:

1. The default ADK builder counts ``thought=True`` vs answer parts on
   every dispatch and stashes the counts on the closure
   (``_call_llm.last_thought_count`` / ``last_answer_count``).
2. When the answer is empty AND there were thought parts, the builder
   logs at INFO with a diagnostic message naming the failure shape so
   operators can distinguish "model spent its budget thinking" from
   "real empty response".
3. The judge call sites (reasoning_judge, classify_goal_drift,
   reflective check) read the stashed counts when they fail to parse
   the response and surface ``"empty answer (N thought parts)"`` on the
   span ``output_preview`` rather than the indistinguishable ``raw=''``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_adk_builder_logs_diagnostic_when_all_thoughts_no_answer(caplog):
    """ADK ``_call_llm`` logs INFO with thought/answer part counts when
    the final concatenated answer is empty but ``thought=True`` parts
    were emitted. This is the exact shape that produced ``raw=''`` on
    Qwen 35B + a 2048 cap pre-#311."""
    pytest.importorskip("google.adk")
    pytest.importorskip("google.genai")

    from google.adk.models.base_llm import BaseLlm  # type: ignore[import-not-found]
    from google.adk.models.llm_response import (  # type: ignore[import-not-found]
        LlmResponse,
    )
    from google.genai import types as genai_types  # type: ignore[import-not-found]

    from goldfive._llm_detect import make_default_adk_call_llm

    class _AllThoughtNoAnswerLLM(BaseLlm):
        async def generate_content_async(self, req, stream=False):  # type: ignore[override]
            # Three thought parts, zero answer parts — the v16 / Qwen
            # 35B failure shape.
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(text="thinking step 1", thought=True),
                        genai_types.Part(text="thinking step 2", thought=True),
                        genai_types.Part(text="thinking step 3", thought=True),
                    ],
                ),
            )

    stub = _AllThoughtNoAnswerLLM(model="stub")
    call_llm = make_default_adk_call_llm(stub)
    assert call_llm is not None

    with caplog.at_level(logging.INFO, logger="goldfive.llm_detect"):
        out = await call_llm("system", "user", "stub")

    assert out == "", "all-thought response must produce empty answer"
    # Part counts stashed on the closure for the caller.
    assert getattr(call_llm, "last_thought_count", None) == 3
    assert getattr(call_llm, "last_answer_count", None) == 0
    # Diagnostic INFO log fired.
    matching = [
        rec
        for rec in caplog.records
        if "thought part" in rec.message and "answer text empty" in rec.message
    ]
    assert matching, (
        f"expected the all-thought-no-answer diagnostic; got log records: "
        f"{[r.message for r in caplog.records]}"
    )
    msg = matching[0].message
    assert "3 thought" in msg


@pytest.mark.asyncio
async def test_adk_builder_no_diagnostic_on_normal_response(caplog):
    """When the model returns a real answer, no diagnostic fires."""
    pytest.importorskip("google.adk")
    pytest.importorskip("google.genai")

    from google.adk.models.base_llm import BaseLlm  # type: ignore[import-not-found]
    from google.adk.models.llm_response import (  # type: ignore[import-not-found]
        LlmResponse,
    )
    from google.genai import types as genai_types  # type: ignore[import-not-found]

    from goldfive._llm_detect import make_default_adk_call_llm

    class _NormalLLM(BaseLlm):
        async def generate_content_async(self, req, stream=False):  # type: ignore[override]
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(text="thinking", thought=True),
                        genai_types.Part(text='{"on_task": true}'),
                    ],
                ),
            )

    stub = _NormalLLM(model="stub")
    call_llm = make_default_adk_call_llm(stub)
    assert call_llm is not None

    with caplog.at_level(logging.INFO, logger="goldfive.llm_detect"):
        out = await call_llm("system", "user", "stub")

    assert out == '{"on_task": true}'
    assert getattr(call_llm, "last_thought_count", None) == 1
    assert getattr(call_llm, "last_answer_count", None) == 1
    # No diagnostic fired.
    diag = [rec for rec in caplog.records if "answer text empty" in rec.message]
    assert not diag


@pytest.mark.asyncio
async def test_reasoning_judge_surfaces_diagnostic_in_span():
    """``classify_reasoning_drift`` records an "empty answer (N thought
    parts)" output_preview on the span when the call_llm returned ``""``
    and stashed a positive thought count. Replaces the previous
    indistinguishable ``raw=''`` shape."""
    from goldfive.drift.reasoning_judge import classify_reasoning_drift
    from goldfive.protocols import EventSink
    from goldfive.types import Task

    # Stub call_llm returning empty AND advertising 3 thought parts via
    # the closure-attached attribute (the same shape the default ADK
    # builder produces).
    async def stub_call_llm(system: str, user: str, model: str) -> str:
        return ""

    stub_call_llm.last_thought_count = 3  # type: ignore[attr-defined]
    stub_call_llm.last_answer_count = 0  # type: ignore[attr-defined]

    captured: list[Any] = []

    class _CaptureSink(EventSink):
        async def emit(self, event: Any) -> None:
            captured.append(event)

        async def close(self) -> None:
            return None

    sink = _CaptureSink()

    drift = await classify_reasoning_drift(
        reasoning="some reasoning to judge",
        task=Task(id="t1", title="Ship the feature"),
        goals=None,
        model="x",
        call_llm=stub_call_llm,
        sink=sink,
        run_id="r",
        session_id="s",
    )
    assert drift is None

    # Find the LLM span End event and check its output_preview
    # mentions the thought-part diagnostic. Span emission goes through
    # ``goldfive_llm_span``; the End event payload is the
    # ``goldfive_llm_call_end`` oneof case.
    end_previews: list[str] = []
    for evt in captured:
        # WhichOneof returns the active oneof field name; check directly
        # by looking for the field on the proto message. ``HasField``
        # works for oneof cases.
        try:
            if evt.HasField("goldfive_llm_call_end"):
                end_previews.append(evt.goldfive_llm_call_end.output_preview)
        except (AttributeError, ValueError):
            continue
    diag_previews = [p for p in end_previews if "thought part" in p]
    assert diag_previews, (
        f"expected an 'empty answer (N thought parts)' span preview; "
        f"saw output_previews: {end_previews}"
    )
    assert "3" in diag_previews[0]
