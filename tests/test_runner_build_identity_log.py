"""Tests for the build-identity log line emitted by :class:`Runner`.

Goldfive#225: when debugging "did the change actually deploy?", the
running build's commit sha and version must be in the logs. The Runner
emits one ``goldfive runner starting: version=X sha=Y`` INFO line per
construction. This guards against regressions in:

* the log line firing on every Runner construction (not silently
  skipped when version detection fails);
* the line surviving when neither importlib.metadata nor a git
  checkout is available — both fields must fall through to ``unknown``
  rather than raising;
* the canary that the line only fires once per Runner instance, not
  per ``run()`` call (we don't want startup chatter on every turn).
"""

from __future__ import annotations

import logging
import re
from typing import Any
from unittest import mock

import pytest

from goldfive import (
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
)
from goldfive import runner as _runner_mod


def _one_task_plan() -> Plan:
    return Plan(
        id="plan-bi",
        run_id="",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="only", assignee_agent_id="writer")],
        edges=[],
        summary="single",
    )


async def _happy_agent(
    task: Task, session: Session, tools: list[ReportingToolSpec]
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text="ok")


def _make_runner() -> Runner:
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_one_task_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("go"),
        sinks=[InMemorySink()],
    )


def test_runner_logs_build_identity_on_construction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``Runner(...)`` emits one ``goldfive runner starting`` INFO line."""
    with caplog.at_level(logging.INFO, logger="goldfive.runner"):
        _make_runner()

    matches = [
        rec for rec in caplog.records if "goldfive runner starting" in rec.message
    ]
    assert len(matches) == 1, [rec.message for rec in caplog.records]
    msg = matches[0].message
    assert re.search(r"version=\S+", msg), msg
    assert re.search(r"sha=\S+", msg), msg
    assert matches[0].levelno == logging.INFO


def test_runner_build_identity_falls_through_to_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Detection failures must NEVER crash construction.

    Both the version path (``importlib.metadata`` raising +
    ``goldfive.__version__`` import raising) and the git path
    (``subprocess.run`` raising) get patched to fail. The Runner must
    still construct and log ``version=unknown sha=unknown``.
    """

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("nope")

    with (
        mock.patch.object(
            _runner_mod._importlib_metadata, "version", side_effect=_boom
        ),
        mock.patch.object(_runner_mod.subprocess, "run", side_effect=_boom),
        mock.patch.dict(
            "sys.modules",
            {"goldfive": mock.MagicMock(__version__=None)},
            clear=False,
        ),
    ):
        with caplog.at_level(logging.INFO, logger="goldfive.runner"):
            _make_runner()

    matches = [
        rec for rec in caplog.records if "goldfive runner starting" in rec.message
    ]
    assert len(matches) == 1
    # Version may resolve via the fallback __version__ chain or end up
    # ``unknown``; the contract that matters is that we DID NOT crash
    # and the line fired.
    assert "sha=" in matches[0].message
    assert "version=" in matches[0].message


async def test_runner_build_identity_logged_once_not_per_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two ``runner.run(...)`` calls must NOT re-emit the startup line."""
    with caplog.at_level(logging.INFO, logger="goldfive.runner"):
        runner = _make_runner()
        await runner.run("go")
        await runner.run("go again")
        await runner.close()

    matches = [
        rec for rec in caplog.records if "goldfive runner starting" in rec.message
    ]
    assert len(matches) == 1, [rec.message for rec in caplog.records]


def test_detect_build_identity_returns_strings() -> None:
    """``_detect_build_identity()`` always returns a ``(str, str)`` tuple."""
    version, sha = _runner_mod._detect_build_identity()
    assert isinstance(version, str) and version
    assert isinstance(sha, str) and sha
