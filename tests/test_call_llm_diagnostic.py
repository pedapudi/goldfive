"""Diagnostic for the all-thought-no-answer LLM-call failure mode
(goldfive#271 follow-up to #311).

Pre-fix: ``_call_llm`` silently dropped every ``thought=True`` part and
returned ``"".join(answer_parts).strip()``. When all parts were thought
parts, the function returned ``""`` — indistinguishable from "the model
truly produced empty output" or "the network ate the response". Two
days were lost to that ambiguity in the v16 / Qwen 35B investigation.

Post-fix (one-llm-call-module refactor):

1. The default builders count ``thought=True`` vs answer parts on every
   dispatch and record them into the per-call
   :class:`goldfive._llm.LlmCallDiagnostics` object installed by the
   consumer via :func:`goldfive._llm.llm_call_diagnostics`. The counts
   previously travelled as attributes mutated on the shared callable —
   last-writer-wins under concurrent background judges.
2. When the answer is empty AND there were thought parts, the builder
   logs at INFO with a diagnostic message naming the failure shape so
   operators can distinguish "model spent its budget thinking" from
   "real empty response".
3. The judge call sites (reasoning_judge, classify_goal_drift,
   reflective check) read the recorded counts when they fail to parse
   the response and surface ``"empty answer (N thought parts)"`` on the
   span ``output_preview`` rather than the indistinguishable ``raw=''``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from goldfive._llm import llm_call_diagnostics, record_llm_call_diagnostics


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

    with caplog.at_level(logging.INFO, logger="goldfive"):
        with llm_call_diagnostics() as diag:
            out = await call_llm("system", "user", "stub")

    assert out == "", "all-thought response must produce empty answer"
    # Part counts recorded into the per-call diagnostics object.
    assert diag.thought_count == 3
    assert diag.answer_count == 0
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

    with caplog.at_level(logging.INFO, logger="goldfive"):
        with llm_call_diagnostics() as diag:
            out = await call_llm("system", "user", "stub")

    assert out == '{"on_task": true}'
    assert diag.thought_count == 1
    assert diag.answer_count == 1
    # No diagnostic fired.
    diag_records = [rec for rec in caplog.records if "answer text empty" in rec.message]
    assert not diag_records


@pytest.mark.asyncio
async def test_openai_builder_records_diagnostic_when_all_reasoning_no_answer(caplog):
    """The OpenAI-compatible builder reports ``reasoning_content`` with
    empty ``content`` as the all-thought-no-answer shape (0/1 sentinel
    counts) through the same diagnostics channel as the ADK builder."""
    pytest.importorskip("openai")
    from goldfive._llm import make_default_openai_call_llm
    from goldfive.config import JudgeConfig

    built = make_default_openai_call_llm(
        JudgeConfig(base_url="http://stub-judge.invalid", model="stub-judge")
    )
    assert built is not None
    call_llm, _model = built

    fake_message = MagicMock(content="")
    type(fake_message).reasoning_content = "chain of thought " * 20  # type: ignore[attr-defined]
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]

    async def fake_create(**kwargs: Any) -> Any:
        return fake_response

    client_cell = None
    for c in call_llm.__closure__ or ():
        if hasattr(c.cell_contents, "chat"):
            client_cell = c
            break
    assert client_cell is not None
    client_cell.cell_contents.chat.completions.create = fake_create

    with caplog.at_level(logging.INFO, logger="goldfive"):
        with llm_call_diagnostics() as diag:
            out = await call_llm("system", "user", "stub-judge")

    assert out == ""
    assert diag.thought_count == 1
    assert diag.answer_count == 0
    matching = [
        rec
        for rec in caplog.records
        if "thought part" in rec.message and "answer text empty" in rec.message
    ]
    assert matching, (
        f"expected the all-thought-no-answer diagnostic; got log records: "
        f"{[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_diagnostics_isolated_across_concurrent_calls():
    """Three concurrent judge-shaped dispatches each observe their own
    counts — the race the old closure-attribute side channel had once
    Wave-2's semaphore allowed up to 3 background judges in flight."""
    started = asyncio.Event()

    async def fake_call_llm(thought_count: int) -> str:
        # Simulate the consolidated builder: record after a suspension
        # point so the three calls fully interleave.
        started.set()
        await asyncio.sleep(0.01 * thought_count)
        record_llm_call_diagnostics(thought_count=thought_count, answer_count=0)
        await asyncio.sleep(0.01)
        return ""

    async def judge_dispatch(thought_count: int) -> tuple[int, int]:
        # Each consumer installs its own per-call diagnostics object,
        # exactly like the judge / reflective-check call sites.
        with llm_call_diagnostics() as diag:
            await fake_call_llm(thought_count)
        return diag.thought_count, diag.answer_count

    results = await asyncio.gather(judge_dispatch(1), judge_dispatch(2), judge_dispatch(3))
    assert results == [(1, 0), (2, 0), (3, 0)]


def test_record_is_noop_without_installed_diagnostics():
    """Recording outside ``llm_call_diagnostics()`` must not raise —
    diagnostics are optional and absent for operator-supplied callables."""
    record_llm_call_diagnostics(thought_count=5, answer_count=1)


@pytest.mark.asyncio
async def test_reasoning_judge_surfaces_diagnostic_in_span():
    """``classify_reasoning_drift`` records an "empty answer (N thought
    parts)" output_preview on the span when the call_llm returned ``""``
    and recorded a positive thought count. Replaces the previous
    indistinguishable ``raw=''`` shape."""
    from goldfive.drift.reasoning_judge import classify_reasoning_drift
    from goldfive.protocols import EventSink
    from goldfive.types import Task

    # Stub call_llm returning empty AND recording 3 thought parts via
    # the per-call diagnostics channel (the same shape the default
    # builders produce).
    async def stub_call_llm(system: str, user: str, model: str) -> str:
        record_llm_call_diagnostics(thought_count=3, answer_count=0)
        return ""

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
