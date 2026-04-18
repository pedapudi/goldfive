"""Unit tests for the built-in EventSinks."""

from __future__ import annotations

import logging

import pytest

from goldfive.pb.goldfive.v1 import events_pb2, types_pb2
from goldfive.sinks import (
    InMemorySink,
    JSONLPersistenceSink,
    LoggingSink,
    reconstruct_session,
    replay_from_jsonl,
)
from goldfive.types import TaskStatus


def _make_event(seq: int, run_id: str = "run-1") -> events_pb2.Event:
    """Build a minimal valid Event with a TaskStarted payload."""
    evt = events_pb2.Event(event_id=f"evt-{seq}", run_id=run_id, sequence=seq)
    evt.task_started.task_id = f"task-{seq}"
    evt.task_started.detail = f"starting {seq}"
    return evt


# ---------------------------------------------------------------------------
# InMemorySink
# ---------------------------------------------------------------------------


async def test_in_memory_sink_collects_events() -> None:
    sink = InMemorySink()
    e1 = _make_event(0)
    e2 = _make_event(1)
    await sink.emit(e1)
    await sink.emit(e2)
    assert sink.events == [e1, e2]
    # close is a no-op and does not clear the list
    await sink.close()
    assert sink.events == [e1, e2]


async def test_in_memory_sink_events_property_is_live() -> None:
    sink = InMemorySink()
    snapshot_before = sink.events
    await sink.emit(_make_event(0))
    # Same list object, not a copy — callers see the append immediately.
    assert sink.events is snapshot_before
    assert len(snapshot_before) == 1


# ---------------------------------------------------------------------------
# LoggingSink
# ---------------------------------------------------------------------------


async def test_logging_sink_writes_one_line_per_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.goldfive.logging_sink")
    logger.setLevel(logging.DEBUG)
    sink = LoggingSink(logger=logger, level=logging.INFO)
    with caplog.at_level(logging.INFO, logger=logger.name):
        await sink.emit(_make_event(0))
        await sink.emit(_make_event(1))
    records = [r for r in caplog.records if r.name == logger.name]
    assert len(records) == 2
    # Each log message is a single line with no embedded newlines
    for r in records:
        assert "\n" not in r.getMessage()
    # Proto field names are preserved (snake_case), not JSON camelCase.
    assert "task_started" in records[0].getMessage()
    assert "task_id" in records[0].getMessage()
    assert "task-0" in records[0].getMessage()
    assert "task-1" in records[1].getMessage()


async def test_logging_sink_respects_custom_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.goldfive.logging_sink_level")
    logger.setLevel(logging.DEBUG)
    sink = LoggingSink(logger=logger, level=logging.WARNING)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        await sink.emit(_make_event(0))
    records = [r for r in caplog.records if r.name == logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING


# ---------------------------------------------------------------------------
# JSONLPersistenceSink
# ---------------------------------------------------------------------------


async def test_jsonl_persistence_sink_roundtrip(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    sink = JSONLPersistenceSink(path)
    events = [_make_event(i) for i in range(5)]
    for e in events:
        await sink.emit(e)
    await sink.close()

    # File has one line per event.
    contents = path.read_text(encoding="utf-8")
    lines = [ln for ln in contents.splitlines() if ln]
    assert len(lines) == 5
    for ln in lines:
        assert "\n" not in ln  # each line is a single JSON doc

    # Replay produces equal Event messages.
    replayed = replay_from_jsonl(path)
    assert len(replayed) == len(events)
    for original, round_tripped in zip(events, replayed, strict=True):
        assert original == round_tripped


async def test_jsonl_persistence_sink_append_mode(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    # First session: write two events and close.
    sink1 = JSONLPersistenceSink(path, mode="append")
    await sink1.emit(_make_event(0))
    await sink1.emit(_make_event(1))
    await sink1.close()
    # Second session: append two more events.
    sink2 = JSONLPersistenceSink(path, mode="append")
    await sink2.emit(_make_event(2))
    await sink2.emit(_make_event(3))
    await sink2.close()

    replayed = replay_from_jsonl(path)
    assert [e.sequence for e in replayed] == [0, 1, 2, 3]


async def test_jsonl_persistence_sink_write_mode_truncates(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    sink1 = JSONLPersistenceSink(path, mode="write")
    await sink1.emit(_make_event(0))
    await sink1.close()
    sink2 = JSONLPersistenceSink(path, mode="write")
    await sink2.emit(_make_event(9))
    await sink2.close()
    replayed = replay_from_jsonl(path)
    assert [e.sequence for e in replayed] == [9]


async def test_jsonl_persistence_sink_concurrent_emit_does_not_corrupt(
    tmp_path,
) -> None:
    import asyncio as _asyncio

    path = tmp_path / "concurrent.jsonl"
    sink = JSONLPersistenceSink(path)
    n = 50
    events = [_make_event(i) for i in range(n)]
    await _asyncio.gather(*(sink.emit(e) for e in events))
    await sink.close()

    replayed = replay_from_jsonl(path)
    assert len(replayed) == n
    # Every emitted event round-trips; order may vary under concurrency
    # but no line is corrupted.
    replayed_ids = sorted(e.event_id for e in replayed)
    expected_ids = sorted(e.event_id for e in events)
    assert replayed_ids == expected_ids


def test_jsonl_persistence_sink_rejects_bad_mode(tmp_path) -> None:
    with pytest.raises(ValueError):
        JSONLPersistenceSink(tmp_path / "x.jsonl", mode="nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reconstruct_session
# ---------------------------------------------------------------------------


def _plan_pb(run_id: str = "run-rec") -> types_pb2.Plan:
    plan = types_pb2.Plan(id="plan-0", run_id=run_id, summary="mini run")
    for tid, title in (("t1", "one"), ("t2", "two"), ("t3", "three")):
        plan.tasks.add(
            id=tid,
            title=title,
            status=types_pb2.TASK_STATUS_PENDING,
        )
    return plan


async def test_reconstruct_session_from_mini_run(tmp_path) -> None:
    path = tmp_path / "mini.jsonl"
    sink = JSONLPersistenceSink(path)
    run_id = "run-rec"

    seq = 0

    def _env() -> events_pb2.Event:
        nonlocal seq
        e = events_pb2.Event(event_id=f"e-{seq}", run_id=run_id, sequence=seq)
        seq += 1
        return e

    # RunStarted
    e = _env()
    e.run_started.run_id = run_id
    e.run_started.goal_summary = "mini"
    await sink.emit(e)

    # PlanSubmitted
    e = _env()
    e.plan_submitted.plan.CopyFrom(_plan_pb(run_id))
    await sink.emit(e)

    # t1: started then completed
    e = _env()
    e.task_started.task_id = "t1"
    await sink.emit(e)

    e = _env()
    e.task_completed.task_id = "t1"
    e.task_completed.summary = "done-1"
    await sink.emit(e)

    # t2: started then failed
    e = _env()
    e.task_started.task_id = "t2"
    await sink.emit(e)

    e = _env()
    e.task_failed.task_id = "t2"
    e.task_failed.reason = "boom"
    e.task_failed.recoverable = True
    await sink.emit(e)

    # t3: blocked
    e = _env()
    e.task_blocked.task_id = "t3"
    e.task_blocked.blocker = "waiting on human"
    await sink.emit(e)

    await sink.close()

    replayed = replay_from_jsonl(path)
    session = reconstruct_session(replayed)

    assert session.run_id == run_id
    assert session.plan is not None
    statuses = {t.id: t.status for t in session.plan.tasks}
    assert statuses == {
        "t1": TaskStatus.COMPLETED,
        "t2": TaskStatus.FAILED,
        "t3": TaskStatus.BLOCKED,
    }
    assert session.completed_results == {"t1": "done-1"}
    # sequence counter resumes past the highest seen sequence
    assert session._next_sequence == seq


async def test_reconstruct_session_plan_revision_uses_latest(tmp_path) -> None:
    path = tmp_path / "revised.jsonl"
    sink = JSONLPersistenceSink(path)
    run_id = "run-rev"

    # Initial plan with two tasks, t1 already completed.
    e0 = events_pb2.Event(event_id="e0", run_id=run_id, sequence=0)
    e0.plan_submitted.plan.CopyFrom(_plan_pb(run_id))
    await sink.emit(e0)

    e1 = events_pb2.Event(event_id="e1", run_id=run_id, sequence=1)
    e1.task_completed.task_id = "t1"
    e1.task_completed.summary = "ok"
    await sink.emit(e1)

    # Revised plan: drop t3, keep t1/t2. Replay should use the revised
    # plan's tasks (so no "t3" in statuses).
    revised = types_pb2.Plan(id="plan-1", run_id=run_id, summary="revised")
    revised.tasks.add(id="t1", title="one", status=types_pb2.TASK_STATUS_PENDING)
    revised.tasks.add(id="t2", title="two", status=types_pb2.TASK_STATUS_PENDING)
    revised.revision_index = 1
    e2 = events_pb2.Event(event_id="e2", run_id=run_id, sequence=2)
    e2.plan_revised.plan.CopyFrom(revised)
    e2.plan_revised.revision_index = 1
    await sink.emit(e2)

    await sink.close()

    replayed = replay_from_jsonl(path)
    session = reconstruct_session(replayed)
    assert session.plan is not None
    assert [t.id for t in session.plan.tasks] == ["t1", "t2"]
    # t1's completion event fired before the revision; since the revised
    # plan replaced the task list, t1 on the new plan is PENDING again
    # (the framework re-applies completion in a later event if the
    # task carries over). completed_results still remembers the summary.
    assert session.completed_results == {"t1": "ok"}
