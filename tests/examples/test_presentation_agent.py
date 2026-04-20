"""Smoke tests for ``examples/presentation_agent``.

The ``presentation_agent`` example is the canonical multi-agent ADK
tree that harmonograf re-exports. These tests pin the three things
downstream callers depend on:

1. Module imports cleanly and re-exports the expected names.
2. ``_build_agent_tree(model)`` builds a coordinator whose
   ``AgentTool`` children are the four expected specialists.
3. ``_run(topic=..., mock=True)`` completes offline with
   ``outcome.success is True``.
4. The lazy ``app`` attribute builds a valid ADK ``App`` whose
   ``root_agent`` is a :class:`BaseAgent` subclass — the contract
   ``adk web`` requires.

A one-liner regression test for the ``adk_web_wrapped`` hyphen fix
lives in :func:`test_adk_web_wrapped_app_name_is_identifier` — prior
to the fix the module raised a pydantic ``ValidationError`` at import
because ``goldfive-wrapped-demo`` is not a valid identifier.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("google.adk")


def test_module_exports() -> None:
    from examples.presentation_agent import _build_agent_tree, root_agent

    assert root_agent is not None
    assert callable(_build_agent_tree)


def test_build_agent_tree_has_four_specialists() -> None:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _Mock(BaseLlm):
        @classmethod
        def supported_models(cls):  # type: ignore[override]
            return [r"mock/.*"]

        async def generate_content_async(  # type: ignore[override]
            self, llm_request: LlmRequest, stream: bool = False
        ):
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                partial=False,
                turn_complete=True,
            )

    from examples.presentation_agent import _build_agent_tree

    coordinator = _build_agent_tree(_Mock(model="mock/test"))
    assert coordinator.name == "coordinator_agent"

    subagent_names = {tool.agent.name for tool in coordinator.tools}
    assert subagent_names == {
        "research_agent",
        "web_developer_agent",
        "reviewer_agent",
        "debugger_agent",
    }


def test_mock_run_completes_successfully() -> None:
    from examples.presentation_agent.agent import _run

    outcome = asyncio.run(_run(topic="testtopic", mock=True))
    assert outcome.success is True
    assert outcome.session.plan is not None
    task_ids = {t.id for t in outcome.session.plan.tasks}
    assert task_ids == {"research", "build", "review", "debug"}


def test_app_is_valid_adk_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """``app`` must load offline and present a ``BaseAgent`` root."""
    # Force mock-mode App construction — no OPENAI_API_KEY, no harmonograf.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HARMONOGRAF_SERVER", raising=False)

    # Reset the cached app so we build a fresh one under the scrubbed env.
    from examples.presentation_agent import agent as agent_mod

    agent_mod._APP = None

    from google.adk.agents import BaseAgent

    from examples.presentation_agent import app

    assert app.name == "presentation_agent"
    assert isinstance(app.root_agent, BaseAgent)


def test_adk_web_wrapped_app_name_is_identifier() -> None:
    """Regression for the hyphen-in-App-name pydantic failure."""
    from examples.adk_web_wrapped import app

    assert app.name == "goldfive_wrapped_demo"
    assert app.name.isidentifier()
