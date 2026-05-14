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


def test_app_is_valid_adk_app(goldfive_examples_env) -> None:
    """``app`` must load offline and present a ``BaseAgent`` root."""
    # Force mock-mode App construction — no OPENAI_API_KEY, no harmonograf.
    # The fixture's ``clear()`` pre-step already unset both variables in
    # the calling environment, so the explicit unsets below are belt-
    # and-braces for readers; either is sufficient.
    goldfive_examples_env.unset("openai_api_key", "harmonograf_server")

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


# ---------------------------------------------------------------------------
# Regression tests for the debugger-cascade fix (#416 / #417 / #418).
# The cherry-tree e2e session 2d27ff4a (2026-05-13) hit a 20+ tool-call
# cascade caused by:
#   #416 — find_presentation_files returned ``candidates`` on miss; the
#          LLM treated it as a list of actionable alternative topics.
#   #417 — output/ was process-wide so the candidate list leaked stale
#          directories from prior sessions.
#   #418 — the debugger instruction lacked the reviewer's anti-loop
#          clause so it kept retrying with guessed topics.
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stand-in for ADK ``Session`` exposing only ``.id``."""

    def __init__(self, sid: str) -> None:
        self.id = sid


class _FakeToolContext:
    """Stand-in for ADK ``ToolContext`` exposing only ``.session.id``.

    The agent's ``_session_output_dir`` only reads ``tool_context.session.id``;
    nothing else needs to be wired for these tests.
    """

    def __init__(self, sid: str) -> None:
        self.session = _FakeSession(sid)


def test_find_presentation_files_miss_returns_no_candidates(
    tmp_path, monkeypatch
) -> None:
    """#416 regression: a miss must NOT include ``candidates``.

    The LLM was treating ``candidates`` as actionable alternative topics
    and looping. Reading the field by accident from a miss response is
    the footgun we are closing.
    """
    from examples.presentation_agent import agent as agent_mod

    # Redirect output/ to a tmp_path so this test never touches the repo.
    monkeypatch.setattr(
        agent_mod, "__file__", str(tmp_path / "agent.py")
    )

    ctx = _FakeToolContext(sid="session-empty")

    result = agent_mod.find_presentation_files("cherry trees", tool_context=ctx)
    assert result == {"found": False}, (
        f"Expected bare {{found: False}}, got {result!r}. "
        "If you re-introduce a candidates list the LLM will retry with "
        "alternative topics — see issue #416."
    )
    assert "candidates" not in result


def test_find_presentation_files_miss_with_existing_dirs_still_no_candidates(
    tmp_path, monkeypatch
) -> None:
    """#416 / #417 regression: even when the session dir exists and has
    sibling directories from previous topics in the same session, a miss
    must not echo their names back to the model."""
    from examples.presentation_agent import agent as agent_mod

    fake_module_dir = tmp_path / "presentation_agent"
    fake_module_dir.mkdir()
    monkeypatch.setattr(
        agent_mod, "__file__", str(fake_module_dir / "agent.py")
    )

    ctx = _FakeToolContext(sid="session-with-prior-work")
    base_dir = fake_module_dir / "output" / "session-with-prior-work"
    base_dir.mkdir(parents=True)
    (base_dir / "pothos_plants_presentation").mkdir()
    (base_dir / "nand_flash_memory_presentation").mkdir()

    result = agent_mod.find_presentation_files("cherry trees", tool_context=ctx)
    assert result == {"found": False}
    assert "candidates" not in result
    # Sanity: the directories really do exist, so the old code WOULD have
    # returned them — this test wouldn't be load-bearing otherwise.
    assert any(base_dir.iterdir())


def test_per_session_output_dir_isolation(tmp_path, monkeypatch) -> None:
    """#417 regression: two distinct session_ids must not see each other's
    presentation files via ``find_presentation_files``."""
    from examples.presentation_agent import agent as agent_mod

    fake_module_dir = tmp_path / "presentation_agent"
    fake_module_dir.mkdir()
    monkeypatch.setattr(
        agent_mod, "__file__", str(fake_module_dir / "agent.py")
    )

    ctx_a = _FakeToolContext(sid="session-A")
    ctx_b = _FakeToolContext(sid="session-B")

    # Session A writes a cherry-trees presentation.
    write_a = agent_mod.write_webpage(
        topic="cherry trees",
        html_content="<html>A</html>",
        css_content="body{}",
        js_content="//A",
        tool_context=ctx_a,
    )
    assert "Successfully" in write_a, write_a

    # Session B writes a pothos presentation.
    write_b = agent_mod.write_webpage(
        topic="pothos plants",
        html_content="<html>B</html>",
        css_content="body{}",
        js_content="//B",
        tool_context=ctx_b,
    )
    assert "Successfully" in write_b, write_b

    # Session A asking for pothos must NOT find session B's directory.
    miss_b_from_a = agent_mod.find_presentation_files(
        "pothos plants", tool_context=ctx_a
    )
    assert miss_b_from_a == {"found": False}, miss_b_from_a

    # Session B asking for cherry trees must NOT find session A's directory.
    miss_a_from_b = agent_mod.find_presentation_files(
        "cherry trees", tool_context=ctx_b
    )
    assert miss_a_from_b == {"found": False}, miss_a_from_b

    # Each session CAN find its own work.
    hit_a = agent_mod.find_presentation_files("cherry trees", tool_context=ctx_a)
    assert hit_a["found"] is True
    assert hit_a["files"]["index.html"] == "<html>A</html>"

    hit_b = agent_mod.find_presentation_files("pothos plants", tool_context=ctx_b)
    assert hit_b["found"] is True
    assert hit_b["files"]["index.html"] == "<html>B</html>"


def test_read_presentation_files_respects_session_scope(
    tmp_path, monkeypatch
) -> None:
    """#417 regression: ``read_presentation_files`` reads from the
    session-scoped directory; cross-session reads must miss."""
    from examples.presentation_agent import agent as agent_mod

    fake_module_dir = tmp_path / "presentation_agent"
    fake_module_dir.mkdir()
    monkeypatch.setattr(
        agent_mod, "__file__", str(fake_module_dir / "agent.py")
    )

    ctx_writer = _FakeToolContext(sid="writer-session")
    ctx_reader = _FakeToolContext(sid="reader-session")

    agent_mod.write_webpage(
        topic="waffles",
        html_content="<html>waffles</html>",
        css_content="body{}",
        js_content="//w",
        tool_context=ctx_writer,
    )

    # Same session reads back ok.
    same = agent_mod.read_presentation_files("waffles", tool_context=ctx_writer)
    assert same["index.html"] == "<html>waffles</html>"

    # Different session sees error markers, not the writer's content.
    cross = agent_mod.read_presentation_files("waffles", tool_context=ctx_reader)
    assert "<error reading" in cross["index.html"]


def test_patch_file_respects_session_scope(tmp_path, monkeypatch) -> None:
    """#417 regression: ``patch_file`` with a relative path resolves under
    the session-scoped output directory, not the bare ``output/`` root."""
    from examples.presentation_agent import agent as agent_mod

    fake_module_dir = tmp_path / "presentation_agent"
    fake_module_dir.mkdir()
    monkeypatch.setattr(
        agent_mod, "__file__", str(fake_module_dir / "agent.py")
    )

    ctx = _FakeToolContext(sid="patch-session")
    result = agent_mod.patch_file(
        path="waffles/index.html",
        new_content="<html>patched</html>",
        tool_context=ctx,
    )
    assert "Successfully patched" in result, result

    written_path = (
        fake_module_dir / "output" / "patch-session" / "waffles" / "index.html"
    )
    assert written_path.exists()
    assert written_path.read_text() == "<html>patched</html>"

    # The bare output/ root must NOT contain a sibling 'waffles' directory.
    bare_path = fake_module_dir / "output" / "waffles" / "index.html"
    assert not bare_path.exists()


def test_debugger_instruction_has_anti_loop_clause() -> None:
    """#418 regression: the debugger instruction must explicitly forbid
    retrying ``find_presentation_files`` with an alternative topic, mirroring
    the reviewer's anti-loop prohibition."""
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
    debugger = next(
        t.agent for t in coordinator.tools if t.agent.name == "debugger_agent"
    )
    instr = debugger.instruction
    # The clause must mention BOTH ``find_presentation_files`` and the
    # do-not-retry-with-alternative-topic prohibition.
    assert "find_presentation_files" in instr
    assert "do NOT" in instr or "do not" in instr.lower()
    assert "alternative topic" in instr or "alternative" in instr
    assert "loop" in instr


def test_list_presentation_directory_not_registered_to_debugger() -> None:
    """``list_presentation_directory`` is operator-only — exposing it to
    the debugger would recreate the #416 footgun. Pin that it's not on
    the debugger's tool list."""
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
    debugger = next(
        t.agent for t in coordinator.tools if t.agent.name == "debugger_agent"
    )
    tool_names = {
        getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", "")
        for t in debugger.tools
    }
    assert "list_presentation_directory" not in tool_names
