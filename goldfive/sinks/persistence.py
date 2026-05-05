"""JSONL persistence sink and replay/reconstruction helpers.

:class:`JSONLPersistenceSink` writes each event as one JSON document per
line (newline-delimited JSON, a.k.a. JSONL). The format is proto-
canonical: a full ``MessageToJson`` of the event with keys sorted so
byte-level diffs stay stable across runs.

Crash recovery is a three-step dance driven from the Runner (issue #15):

1. Load the JSONL with :func:`replay_from_jsonl` — returns the parsed
   ``Event`` messages in emit order.
2. Call :func:`reconstruct_session` on that list — returns a best-effort
   :class:`goldfive.types.Session` reflecting the latest plan and task
   statuses.
3. Hand the session back to ``Runner.resume()`` (not implemented in this
   PR; tracked in the Runner issue).

The sink is async-safe: a single :class:`asyncio.Lock` serialises
writes so concurrent ``emit`` coroutines cannot interleave bytes. The
write itself is a synchronous ``file.write`` + ``flush`` inside the
lock, which is fine for the file sizes goldfive runs produce and keeps
the dependency surface to stdlib only.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from google.protobuf.json_format import MessageToJson, Parse

if TYPE_CHECKING:
    # Importing the generated proto module at module load would force
    # every goldfive user to have the `proto` extra installed. Defer it
    # to first use; the functions that need it import locally.
    from goldfive.pb.goldfive.v1 import events_pb2 as _events_pb2  # noqa: F401


def _events_module() -> Any:
    """Lazy-import the generated ``events_pb2`` module."""
    try:
        from goldfive.pb.goldfive.v1 import events_pb2
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by tests
        raise ModuleNotFoundError(
            "goldfive protobuf stubs not available; generate them via "
            "`make proto` (requires the `proto` optional-dependency group) "
            "or install the package with the `proto` extra. See issue #3."
        ) from exc
    return events_pb2


class JSONLPersistenceSink:
    """EventSink that appends events to a JSONL file.

    Parameters
    ----------
    path:
        File path to write to. Parent directories are created on first
        emit if they do not already exist.
    mode:
        ``"append"`` (default) opens the file with ``"a"`` so reruns
        extend the existing log. ``"write"`` opens with ``"w"`` and
        truncates — useful for deterministic tests and for callers that
        manage one-file-per-run themselves.

    Notes
    -----
    The file handle is opened lazily on the first ``emit`` so constructing
    the sink is side-effect-free. ``close`` flushes and closes the handle;
    subsequent emits reopen it.
    """

    def __init__(
        self,
        path: str | Path,
        mode: Literal["append", "write"] = "append",
    ) -> None:
        self._path = Path(path)
        if mode not in ("append", "write"):
            raise ValueError(f"mode must be 'append' or 'write', got {mode!r}")
        self._mode = mode
        self._handle: Any = None
        # Created lazily so the sink can be built off-loop and attached
        # to an event loop later (asyncio.Lock binds to the running loop
        # at first await).
        self._lock = asyncio.Lock()

    def _open(self) -> None:
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        flag = "a" if self._mode == "append" else "w"
        # Line-buffered text mode so each write lands on disk promptly;
        # the explicit flush inside ``emit`` backs this up for buffered
        # configurations (e.g. when stdout is redirected into the handle).
        self._handle = open(self._path, flag, buffering=1, encoding="utf-8")

    async def emit(self, event: Any) -> None:
        """Serialise ``event`` to one JSON line and append it to the file.

        Proto messages are serialised with ``MessageToJson(event,
        sort_keys=True, indent=None)`` for byte-stable output. Non-proto
        events (dicts or objects exposing ``to_dict``) fall through to
        ``json.dumps`` so the sink accepts both shapes. Concurrent callers
        are serialised by an :class:`asyncio.Lock` so lines never interleave.
        """
        if hasattr(event, "DESCRIPTOR"):
            line = MessageToJson(event, sort_keys=True, indent=None)
        elif hasattr(event, "to_dict"):
            line = json.dumps(event.to_dict(), sort_keys=True)
        elif isinstance(event, dict):
            line = json.dumps(event, sort_keys=True, default=str)
        else:
            # Last-resort: reflect public attributes.
            payload = {k: v for k, v in vars(event).items() if not k.startswith("_")}
            line = json.dumps(payload, sort_keys=True, default=str)
        async with self._lock:
            self._open()
            assert self._handle is not None
            self._handle.write(line + "\n")
            self._handle.flush()

    async def close(self) -> None:
        """Flush and close the file handle. Safe to call repeatedly."""
        async with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                finally:
                    self._handle.close()
                    self._handle = None


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def replay_from_jsonl(path: str | Path) -> list[Any]:
    """Read ``path`` and return a list of parsed ``Event`` messages.

    Each non-empty line is parsed via ``google.protobuf.json_format.Parse``
    into a fresh ``Event``. Blank lines are skipped. Malformed lines
    propagate the parser's exception; callers that want best-effort
    replay can wrap the call and filter by line number.
    """
    pb = _events_module()
    p = Path(path)
    events: list[Any] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = pb.Event()
            Parse(line, msg)
            events.append(msg)
    return events


def reconstruct_session(events: list[Any]) -> Any:
    """Rebuild a :class:`goldfive.types.Session` from a list of events.

    Best-effort: walks events in order, applying the side-effects a real
    run would have produced:

    * ``RunStarted`` seeds ``run_id`` and ``started_at_ms``.
    * ``GoalDerived`` populates ``session.goals``.
    * ``PlanSubmitted`` / ``PlanRevised`` replace ``session.plan``. The
      latest plan wins.
    * ``TaskStarted`` / ``TaskCompleted`` / ``TaskFailed`` /
      ``TaskBlocked`` / ``TaskCancelled`` update the matching task's
      ``status`` on the current plan (if present).
    * ``TaskCompleted.summary`` is written into
      ``session.completed_results``.
    * ``TaskProgress`` writes ``fraction`` into ``session.task_progress``
      and ``detail`` into ``session.agent_notes``.
    * ``session._next_sequence`` advances past the highest seen sequence
      so resumed runs keep the counter monotonic.

    The reconstruction is intentionally forgiving — unknown task ids are
    ignored rather than raised, because a later plan revision may have
    dropped a task that earlier events referenced. Callers that need
    strict validation should walk the events themselves.
    """
    # Local imports: we do not want this module to hard-require the
    # dataclasses package at import time (it pulls ``goldfive.types``,
    # which is fine, but keeping the import local matches the lazy proto
    # import above).
    from goldfive.conv import from_pb_goal, from_pb_plan
    from goldfive.types import (
        Session,
        TaskStatus,
        channel_processor_active,
        set_session_plan,
        with_task_status,
    )

    session = Session(run_id="")

    # Track the maximum sequence we have seen so the counter can resume.
    max_seq = -1

    def _update_task_status(task_id: str, status: TaskStatus) -> None:
        # goldfive#247: Plan + Task are frozen — derive a new Plan via
        # :func:`with_task_status` and swap it onto the session under
        # :func:`channel_processor_active`. ``reconstruct_session`` is a
        # crash-recovery synthetic writer (no live channel processor),
        # but it IS the sole writer for the duration of replay so the
        # contextvar is the right scope.
        if session.plan is None or not task_id:
            return
        # Fast path: skip if no task with this id is in the plan.
        if not any(t.id == task_id for t in session.plan.tasks):
            return
        with channel_processor_active():
            set_session_plan(session, with_task_status(session.plan, task_id, status))

    for evt in events:
        seq = int(getattr(evt, "sequence", 0) or 0)
        if seq > max_seq:
            max_seq = seq
        payload = evt.WhichOneof("payload")
        if payload is None:
            continue
        if payload == "run_started":
            rs = evt.run_started
            session.run_id = rs.run_id or evt.run_id or session.run_id
            if rs.HasField("started_at"):
                session.started_at_ms = (
                    rs.started_at.seconds * 1000 + rs.started_at.nanos // 1_000_000
                )
        elif payload == "goal_derived":
            session.goals = [from_pb_goal(g) for g in evt.goal_derived.goals]
        elif payload == "plan_submitted":
            with channel_processor_active():
                set_session_plan(session, from_pb_plan(evt.plan_submitted.plan))
        elif payload == "plan_revised":
            with channel_processor_active():
                set_session_plan(session, from_pb_plan(evt.plan_revised.plan))
        elif payload == "task_started":
            ts = evt.task_started
            session.current_task_id = ts.task_id
            _update_task_status(ts.task_id, TaskStatus.RUNNING)
        elif payload == "task_progress":
            tp = evt.task_progress
            if tp.task_id:
                session.task_progress[tp.task_id] = float(tp.fraction)
                if tp.detail:
                    session.agent_notes[tp.task_id] = tp.detail
        elif payload == "task_completed":
            tc = evt.task_completed
            _update_task_status(tc.task_id, TaskStatus.COMPLETED)
            if tc.task_id:
                session.completed_results[tc.task_id] = tc.summary
            if session.current_task_id == tc.task_id:
                session.current_task_id = ""
        elif payload == "task_failed":
            tf = evt.task_failed
            _update_task_status(tf.task_id, TaskStatus.FAILED)
            if session.current_task_id == tf.task_id:
                session.current_task_id = ""
        elif payload == "task_blocked":
            tb = evt.task_blocked
            _update_task_status(tb.task_id, TaskStatus.BLOCKED)
        elif payload == "task_cancelled":
            tc2 = evt.task_cancelled
            _update_task_status(tc2.task_id, TaskStatus.CANCELLED)
            if session.current_task_id == tc2.task_id:
                session.current_task_id = ""
        elif payload == "drift_detected":
            session.divergence_flag = True
        # run_completed / run_aborted are terminal markers; the caller
        # decides whether to resume. We leave session state as-is.

    if max_seq >= 0:
        session._next_sequence = max_seq + 1
    return session
