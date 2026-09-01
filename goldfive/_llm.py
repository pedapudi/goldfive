"""The one internal LLM-call module: typing, lifecycle, budget, and builders.

Goldfive accepts an opaque ``call_llm(system, user, model) -> str`` async
callable for both :class:`LLMPlanner` and :class:`LLMGoalDeriver`. That
keeps the framework decoupled from any specific SDK, but it also leaves
resource cleanup ambiguous — the most common pattern (an OpenAI
``AsyncClient`` whose ``aiohttp.ClientSession`` lives until garbage
collection) leaks at process exit.

Besides the protocols and ContextVar plumbing, this module owns the two
default ``call_llm`` builders (:func:`make_default_adk_call_llm` for ADK
trees, :func:`make_default_openai_call_llm` for a dedicated
:class:`~goldfive.config.JudgeConfig` endpoint), the model-capability
table for vendor thinking-disable conventions, and the per-dispatch
diagnostics channel (:func:`llm_call_diagnostics`). They previously
lived as two divergent copies in :mod:`goldfive._llm_detect` and
:mod:`goldfive.convenience`; those modules keep thin re-export shims.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from goldfive.config import JudgeConfig

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


# ---------------------------------------------------------------------------
# Per-callsite "disable thinking" signal (goldfive#271 follow-up to #311)
# ---------------------------------------------------------------------------
#
# Goldfive's judges / goal_deriver / planner refines ask small JSON-shaped
# questions ("is this on-task?", "is the trajectory progressing?", "extract
# the goal"). Running them through Qwen 3.5 / Gemini "thinking" mode wastes
# the entire ``max_output_tokens`` ceiling on internal ``<think>`` reasoning
# and leaves nothing for the JSON answer — the v16 / Qwen 35B failure mode
# fixed by #311's 16k cap was the *symptom*; the *cause* is that judge
# dispatches don't need thinking at all.
#
# Thinking mode is the user's model behaviour for their own agents
# (coordinator / research / web_developer / ...) — we do NOT change that.
# This ContextVar narrowly scopes "disable thinking" to goldfive's internal
# meta-cognition dispatches. The default ADK / OpenAI builders read it and
# attach the SDK-specific opt-out (``ThinkingConfig(thinking_budget=0)`` for
# google.genai; ``extra_body={"enable_thinking": False}`` for Qwen-via-litellm
# / OpenAI-compatible endpoints).

#: ContextVar carrying the per-callsite disable-thinking flag. ``None`` /
#: ``False`` means "use the model's natural thinking behaviour" — the
#: default. ``True`` means "ask the SDK to suppress thinking for this
#: dispatch".
THINKING_DISABLED_VAR: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "goldfive_call_llm_thinking_disabled", default=None
)


def get_thinking_disabled() -> bool:
    """Return whether the per-callsite disable-thinking signal is set.

    ``False`` by default — agent-side LLM calls (coordinator / research /
    web_developer / etc.) keep their natural thinking behaviour. The
    judges / goal_deriver / planner refine call sites enter
    :func:`call_llm_thinking_disabled` to flip this on for the duration
    of a single ``await call_llm(...)``.
    """
    flag = THINKING_DISABLED_VAR.get()
    return bool(flag)


@contextmanager
def call_llm_thinking_disabled() -> Iterator[None]:
    """Disable thinking-mode on the ``call_llm`` dispatch inside the with-block.

    Used by goldfive consumers that ask small JSON-shaped questions
    (judges / goal_deriver / planner refine / reflective check) to avoid
    burning the ``max_output_tokens`` budget on ``<think>`` reasoning
    that nobody reads. Restores the prior value on exit even if the body
    raises.

    Reads + writes :data:`THINKING_DISABLED_VAR`. The default ADK and
    OpenAI builders inspect ``get_thinking_disabled()`` and attach the
    SDK-specific opt-out:

    * ``google.genai.types.ThinkingConfig(include_thoughts=False,
      thinking_budget=0)`` on ``GenerateContentConfig`` for ADK / Gemini.
    * ``extra_body={"enable_thinking": False}`` on
      ``client.chat.completions.create`` for Qwen via litellm / OpenAI-
      compatible endpoints. Vendors that don't recognise the field
      ignore it (we also include ``/no_think`` in the system prompt as a
      Qwen-prompt-level fallback when the SDK shape doesn't accept the
      kwarg).

    User-supplied ``call_llm`` callables can opt in by reading
    :func:`get_thinking_disabled` themselves; they are not required to.
    The cost of ignoring it is "the model thinks anyway" — exactly the
    pre-fix behaviour that wasted the token budget in v16 evidence.
    """
    token = THINKING_DISABLED_VAR.set(True)
    try:
        yield
    finally:
        THINKING_DISABLED_VAR.reset(token)


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


# ---------------------------------------------------------------------------
# Model-capability table: vendor thinking-disable conventions
# ---------------------------------------------------------------------------
#
# How a "disable thinking" request is expressed on the wire is a vendor
# convention, not a property of the transport. The genai
# ``ThinkingConfig(include_thoughts=False, thinking_budget=0)`` opt-out is
# first-class SDK surface on the ADK path and is applied there for every
# model (unchanged behaviour). The two Qwen-specific hacks — the
# ``enable_thinking`` extra-body field and the ``/no_think`` prompt
# prefix — ride the OpenAI-compatible wire format and are meaningful only
# to the Qwen / litellm family, so they are keyed off the model name
# here. This is a lookup table of vendor conventions (configuration),
# not NL classification.


@dataclass(frozen=True)
class ThinkingDisableCaps:
    """Vendor-convention knobs available for suppressing thinking."""

    #: ``extra_body={"enable_thinking": False}`` on
    #: ``chat.completions.create`` (Qwen-via-litellm / OpenAI-compatible).
    openai_enable_thinking_field: bool = False
    #: ``/no_think`` prepended to the system prompt (Qwen prompt-level
    #: toggle; fallback for endpoints that drop unknown request fields).
    no_think_prompt_prefix: bool = False


_NO_VENDOR_THINKING_CAPS = ThinkingDisableCaps()

#: ``(model-name substring, caps)`` pairs, matched case-insensitively in
#: order. Models that match no entry get NO vendor hacks — the genai
#: ``ThinkingConfig`` opt-out on the ADK path still applies regardless.
THINKING_DISABLE_CAPABILITIES: tuple[tuple[str, ThinkingDisableCaps], ...] = (
    (
        "qwen",
        ThinkingDisableCaps(openai_enable_thinking_field=True, no_think_prompt_prefix=True),
    ),
)


def thinking_disable_caps(model_name: str) -> ThinkingDisableCaps:
    """Return the vendor thinking-disable conventions for ``model_name``.

    Matches :data:`THINKING_DISABLE_CAPABILITIES` by lowercase substring
    so litellm-prefixed names (``"openai/Qwen3-32B"``,
    ``"hosted_vllm/Qwen/Qwen3-32B"``) route to the same family. Unknown
    models get the empty caps — no vendor hacks.
    """
    lowered = (model_name or "").lower()
    for marker, caps in THINKING_DISABLE_CAPABILITIES:
        if marker in lowered:
            return caps
    return _NO_VENDOR_THINKING_CAPS


# ---------------------------------------------------------------------------
# Per-dispatch diagnostics (goldfive#271 follow-up to #311)
# ---------------------------------------------------------------------------
#
# When a judge / reflective-check response fails to parse, the call site
# wants to distinguish "the model spent its budget thinking and emitted
# no answer" from "the model returned garbage". The default builders
# below count thought vs answer parts on every dispatch. The counts used
# to be smuggled as attributes mutated on the shared callable
# (``call_llm.last_thought_count``) — last-writer-wins once concurrent
# background judges dispatch through the same closure. They now travel
# through a ContextVar-bound per-call object: each consumer installs a
# fresh :class:`LlmCallDiagnostics` via :func:`llm_call_diagnostics`
# around its own ``await call_llm(...)``, so concurrent tasks cannot
# observe each other's counts. Diagnostics are optional — user-supplied
# callables that never call :func:`record_llm_call_diagnostics` simply
# leave the counts at zero.


@dataclass
class LlmCallDiagnostics:
    """Per-dispatch part counts recorded by goldfive's default builders.

    ``thought_count`` counts real ``thought=True`` parts on the ADK
    path; the OpenAI-compatible path reports the presence of
    ``reasoning_content`` as a 0/1 sentinel. ``answer_count`` counts
    non-empty answer parts (0/1 sentinel on the OpenAI path).
    """

    thought_count: int = 0
    answer_count: int = 0


#: ContextVar carrying the per-call diagnostics object, or ``None`` when
#: no consumer asked for diagnostics (user-supplied dispatch paths).
LLM_CALL_DIAGNOSTICS_VAR: contextvars.ContextVar[LlmCallDiagnostics | None] = (
    contextvars.ContextVar("goldfive_call_llm_diagnostics", default=None)
)


@contextmanager
def llm_call_diagnostics() -> Iterator[LlmCallDiagnostics]:
    """Install a fresh diagnostics object for the dispatch inside the block.

    Yields the object; the caller reads its counts after ``await
    call_llm(...)`` returns (the object outlives the with-block). Resets
    the var on exit even if the body raises, so a failed dispatch cannot
    leak stale counts into a sibling call in the same context.
    """
    diag = LlmCallDiagnostics()
    token = LLM_CALL_DIAGNOSTICS_VAR.set(diag)
    try:
        yield diag
    finally:
        LLM_CALL_DIAGNOSTICS_VAR.reset(token)


def record_llm_call_diagnostics(*, thought_count: int, answer_count: int) -> None:
    """Record part counts into the current dispatch's diagnostics object.

    No-op when no consumer installed one via
    :func:`llm_call_diagnostics` — recording is strictly optional, and
    user-supplied ``call_llm`` callables are not expected to call this.
    """
    diag = LLM_CALL_DIAGNOSTICS_VAR.get()
    if diag is None:
        return
    diag.thought_count = thought_count
    diag.answer_count = answer_count


def _note_dispatch_result(
    *, transport: str, result: str, thought_count: int, answer_count: int
) -> None:
    """Record diagnostics and log the all-thought-no-answer failure shape.

    Shared by both default builders so success and failure expose the
    same observable shape (goldfive#271 follow-up to #311: pre-fix, an
    all-thought response returned an indistinguishable ``raw=''`` that
    cost two days of misdiagnosis on v16 / Qwen 35B).
    """
    record_llm_call_diagnostics(thought_count=thought_count, answer_count=answer_count)
    if not result and thought_count > 0:
        log.info(
            "goldfive._llm.%s: model returned %d thought part(s), %d answer "
            "part(s), answer text empty — check thinking-mode config or "
            "max_output_tokens (the model spent its budget thinking and "
            "emitted no answer). Goldfive's judges should run with "
            "call_llm_thinking_disabled() entered.",
            transport,
            thought_count,
            answer_count,
        )


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
    underlying HTTP session (e.g. ``await openai_client.close()``). A caller
    that constructs or supplies the callable retains ownership unless the
    accepting API explicitly documents otherwise.
    """

    async def close(self) -> None: ...


async def maybe_close_call_llm(call_llm: Any, *, label: str = "call_llm") -> None:
    """Release resources owned by ``call_llm`` when it exposes ``close()``.

    Returns immediately when ``call_llm`` is ``None`` or has no
    ``close`` attribute. Logs and discards any exception raised by
    ``close`` so cleanup remains robust under partial initialisation.
    Callers that create a callable with
    :func:`make_default_openai_call_llm` should await this helper once
    after the last dispatch.
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


# ---------------------------------------------------------------------------
# Default ``call_llm`` builders (ADK tree LLM / OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


async def _probe_close(target: Any, *, label: str) -> None:
    """Close ``target``'s network resources via duck-typed probing.

    Probes ``aclose`` / ``close`` on ``target`` itself, then on a nested
    ``._client`` / ``.client`` (LiteLlm and friends stash the HTTP
    session there). Awaits the first hit; silently no-ops when nothing
    is found. Exceptions are logged and swallowed — teardown must not
    raise.
    """
    candidates: list[tuple[str, Any]] = [(label, target)]
    for client_attr in ("_client", "client"):
        client = getattr(target, client_attr, None)
        if client is not None:
            candidates.append((f"{label}.{client_attr}", client))
    for name, obj in candidates:
        for attr_name in ("aclose", "close"):
            closer = getattr(obj, attr_name, None)
            if callable(closer):
                try:
                    result = closer()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:  # noqa: BLE001 - cleanup must not raise
                    log.debug("%s.%s raised %s", name, attr_name, exc)
                return


def make_default_adk_call_llm(model: Any) -> CallLLM | None:
    """Return a ``call_llm(system, user, model) -> str`` backed by ADK.

    ``model`` may be a string alias (``"gpt-4o"``), a ``BaseLlm``
    instance (including ``LiteLlm``), or anything ``LLMRegistry`` can
    resolve. Returns ``None`` when ADK is not installed or the model
    cannot be resolved to a ``BaseLlm``.
    """
    try:
        from google.adk.models.base_llm import BaseLlm  # type: ignore
        from google.adk.models.llm_request import LlmRequest  # type: ignore
        from google.adk.models.registry import LLMRegistry  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except ImportError:
        log.debug("goldfive._llm: google.adk not installed")
        return None

    if isinstance(model, BaseLlm):
        llm: Any = model
    elif isinstance(model, str) and model:
        try:
            llm_cls = LLMRegistry.new_llm(model)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            log.debug("goldfive._llm: LLMRegistry.new_llm(%r) raised: %s", model, exc)
            return None
        llm = llm_cls
    else:
        return None

    async def _call_llm(system: str, user: str, model_str: str) -> str:
        _ = model_str  # ADK's BaseLlm is already model-bound
        # Per-callsite cap (see :func:`call_llm_budget`); ``None`` falls
        # back to DEFAULT_MAX_OUTPUT_TOKENS so an unsupervised dispatch
        # still has a finite ceiling (goldfive#271: pre-fix evidence in
        # demo-v8.log showed unbounded calls reaching 9961 completion
        # tokens / 9.6 minutes wall on a Qwen Q4 endpoint).
        max_output_tokens = get_max_output_tokens()
        # Per-callsite disable-thinking signal (goldfive#271 follow-up
        # to #311). The genai ``ThinkingConfig`` opt-out is first-class
        # SDK surface on this path and applies to every model —
        # without it, the 16k cap from #311 is spent inside ``<think>``
        # and the JSON answer comes back truncated. The Qwen-only
        # prompt/extra-body hacks live in the OpenAI builder, gated on
        # :func:`thinking_disable_caps`.
        thinking_config: Any = None
        if get_thinking_disabled():
            try:
                thinking_config = genai_types.ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=0,
                )
            except Exception as exc:  # noqa: BLE001
                # Older google.genai shapes may not expose ThinkingConfig;
                # the model just keeps thinking, as before this fix.
                log.debug(
                    "goldfive._llm: ThinkingConfig unavailable (%s); "
                    "continuing without thinking-disabled hint",
                    exc,
                )
                thinking_config = None
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_output_tokens,
        }
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        req = LlmRequest(
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user)],
                ),
            ],
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        chunks: list[str] = []
        thought_part_count = 0
        answer_part_count = 0
        async for resp in llm.generate_content_async(req, stream=False):
            content = getattr(resp, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or ():
                if getattr(part, "thought", False):
                    thought_part_count += 1
                    continue
                text = getattr(part, "text", "") or ""
                if text:
                    answer_part_count += 1
                    chunks.append(str(text))
        result = "".join(chunks).strip()
        _note_dispatch_result(
            transport="adk_call_llm",
            result=result,
            thought_count=thought_part_count,
            answer_count=answer_part_count,
        )
        return result

    async def _close() -> None:
        await _probe_close(llm, label="adk_call_llm")

    _call_llm.close = _close  # type: ignore[attr-defined]
    return cast(CallLLM, _call_llm)


def make_default_openai_call_llm(
    config: JudgeConfig,
) -> tuple[ClosableCallLLM, str] | None:
    """Construct an OpenAI-compatible ``CallLLM`` from a :class:`JudgeConfig`.

    Returns ``(call_llm, model)`` or ``None`` when the ``openai``
    package is not importable or the client cannot be built. The returned
    callable exposes a ``close`` coroutine. Ownership transfers to the
    caller, which should await :func:`maybe_close_call_llm` once after its
    last dispatch. Passing the callable to ``goldfive.wrap`` as
    ``judge_call_llm`` does not transfer ownership to the resulting Runner.

    Design parallels :class:`goldfive.drift._embed._OpenAIEmbeddingBackend`
    — we intentionally tolerate missing / placeholder ``api_key`` so
    llama.cpp / Ollama endpoints "just work" (they don't check the
    header).
    """
    base_url = (config.base_url or "").strip()
    if not base_url:
        return None
    try:
        from openai import AsyncOpenAI  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "goldfive._llm: openai SDK not importable for JudgeConfig (base_url=%r): %s",
            base_url,
            exc,
        )
        return None
    timeout_s = max(0.1, config.timeout_ms / 1000.0)
    try:
        client: Any = AsyncOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=config.api_key or "not-needed",
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "goldfive._llm: AsyncOpenAI client construction failed for "
            "JudgeConfig (base_url=%r): %s",
            base_url,
            exc,
        )
        return None

    model_name = config.model or ""

    async def _call_llm(system: str, user: str, model_str: str) -> str:
        # Prefer the model argument supplied by the caller (matches the
        # contract used by :class:`~goldfive.planner.LLMPlanner`); fall
        # back to the config model when the caller passes the empty
        # string. An empty-string model is tolerated by llama.cpp /
        # Ollama even against an OpenAI endpoint that requires one,
        # because we only hit endpoints the operator configured.
        effective_model = model_str or model_name
        # Per-callsite disable-thinking signal (goldfive#271 follow-up
        # to #311). The ``enable_thinking`` extra-body field and the
        # ``/no_think`` prompt prefix are Qwen / litellm conventions —
        # applied only when the model matches that family (see
        # :func:`thinking_disable_caps`). Other vendors get no hacks on
        # this wire format.
        effective_system = system
        extra_body: dict[str, Any] = {}
        if get_thinking_disabled():
            caps = thinking_disable_caps(effective_model)
            if caps.openai_enable_thinking_field:
                extra_body["enable_thinking"] = False
            if caps.no_think_prompt_prefix and "/no_think" not in (system or ""):
                effective_system = f"/no_think\n{system}" if system else "/no_think"

        create_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user},
            ],
            # Per-callsite cap (see :func:`call_llm_budget`). Pre-fix:
            # unbounded → 9961-token responses (goldfive#271 demo-v8.log).
            "max_tokens": get_max_output_tokens(),
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        try:
            resp = await client.chat.completions.create(**create_kwargs)
        except TypeError as exc:
            # Older OpenAI client versions don't accept ``extra_body``.
            # Retry without it — the ``/no_think`` system-prompt prefix
            # still does its job for Qwen. Other TypeErrors are real
            # failures; fall through.
            if "extra_body" not in create_kwargs:
                raise
            log.debug(
                "goldfive._llm: AsyncOpenAI rejected extra_body=%r (%s); retrying without it",
                extra_body,
                exc,
            )
            create_kwargs.pop("extra_body", None)
            resp = await client.chat.completions.create(**create_kwargs)
        try:
            content = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return ""
        # Qwen-via-litellm returns reasoning text on a sibling field
        # (``reasoning_content``); when ``content == ""`` but reasoning
        # is present, the model spent its budget thinking and produced
        # no answer — the OpenAI-compatible analogue of the ADK
        # all-thought-no-answer shape, reported as 0/1 sentinels.
        result = str(content)
        try:
            reasoning_content = getattr(resp.choices[0].message, "reasoning_content", "") or ""
        except Exception:  # noqa: BLE001
            reasoning_content = ""
        _note_dispatch_result(
            transport="openai_call_llm",
            result=result,
            thought_count=1 if reasoning_content else 0,
            answer_count=1 if result else 0,
        )
        return result

    async def _close() -> None:
        await _probe_close(client, label="openai_call_llm")

    _call_llm.close = _close  # type: ignore[attr-defined]
    return cast(ClosableCallLLM, _call_llm), model_name


__all__ = [
    "CallLLM",
    "ClosableCallLLM",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "LLM_CALL_DIAGNOSTICS_VAR",
    "LlmCallDiagnostics",
    "MAX_OUTPUT_TOKENS_VAR",
    "THINKING_DISABLED_VAR",
    "THINKING_DISABLE_CAPABILITIES",
    "ThinkingDisableCaps",
    "call_llm_budget",
    "call_llm_thinking_disabled",
    "get_max_output_tokens",
    "get_thinking_disabled",
    "llm_call_diagnostics",
    "make_default_adk_call_llm",
    "make_default_openai_call_llm",
    "maybe_close_call_llm",
    "record_llm_call_diagnostics",
    "thinking_disable_caps",
]
