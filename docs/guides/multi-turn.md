# Multi-turn conversations

Most agent UIs are conversations, not one-shot runs: a user asks for
something, sees the result, and follows up — *"actually, make it
funnier"*, *"now tighten the rhymes"*. goldfive's `Runner` supports
this out of the box via a `Conversation` that persists cross-turn
state so the planner can see what the previous turns produced.

This guide covers how the `Conversation` works, what each turn sees,
how to reset it, and what lifecycle events sinks observe.

Related: [Getting started](getting-started.md),
[Goals and plans](goals-and-plans.md),
[EVENT-MODEL.md](../design/EVENT-MODEL.md).

## The shapes

Every `Runner` owns a `Conversation`. On each `runner.run(user_input)`
call the Runner seeds a fresh `Session` from the Conversation, runs the
usual derive → plan → execute loop, then folds the outcome back into
the Conversation so the next turn starts from the updated state.

```python
@dataclasses.dataclass
class Conversation:
    id: str                                # stable across turns
    started_at_ms: int
    goals: list[Goal]                      # accumulated, deduped by id
    completed_results: dict[str, str]      # task_id -> summary, cross-turn
    turns: list[TurnRecord]                # one record per completed turn


@dataclasses.dataclass
class TurnRecord:
    run_id: str
    user_input_summary: str
    plan_summary: str
    outcome_success: bool
    outcome_reason: str
    completed_task_ids: list[str]
    started_at_ms: int
    ended_at_ms: int
```

Single-turn callers never need to touch either class — the Runner
constructs a fresh `Conversation` on `__init__` and the surface of
`runner.run()` is unchanged.

## The minimal multi-turn flow

```python
import goldfive

runner = goldfive.wrap(my_agent)   # owns a fresh Conversation

out1 = await runner.run("Write a limerick about cats")
out2 = await runner.run("Actually, make it funnier")
out3 = await runner.run("Now tighten the rhymes")

# out2's session was seeded with turn 1's completed_results; the
# planner's prompt for turn 2 contains a rendered summary of turn 1's
# output. Turn 3 sees both turn 1 and turn 2. No caller plumbing.
```

Every turn still emits its own `RunStarted` → … → `RunCompleted` event
sequence; the `conversation_id` field on the `Session` (and on the
`ConversationStarted` lifecycle event) is what ties them together for
observers.

See [examples/multi_turn_chat.py](../../examples/multi_turn_chat.py) for
a full runnable demo that uses a scripted LLM stub.

## What each turn sees

When turn `N` runs, the `Session` the executor receives starts with:

| Field                    | Seeded from                    |
|--------------------------|--------------------------------|
| `run_id`                 | fresh UUID (per turn)          |
| `conversation_id`        | `conversation.id` (stable)     |
| `goals`                  | `conversation.goals` (copy)    |
| `completed_results`      | `conversation.completed_results` (copy) |
| `plan`                   | `None` — planned per turn      |
| `history`                | `[]` (v1)                      |

The planner additionally receives a `context` dict that includes:

- `conversation_id`
- `turn_index` — how many turns have already finished
- `prior_completed_results` — the full cross-turn result map
- `prior_turns` — the last 3 `TurnRecord`s (windowed to keep prompts
  bounded)

`LLMPlanner.generate()` renders these into a human-readable
"Prior-turn context" block in its prompt so the LLM can reason about
follow-up intent without the caller doing anything. If you're writing
a custom planner, read the same keys off your `context: Mapping[str,
Any]` parameter.

## Resetting: `runner.new_conversation()`

When the current conversation has ended and the next `runner.run()`
should start from a blank slate:

```python
await runner.new_conversation()
```

Effects:

- The outgoing Conversation emits a terminal `ConversationEnded` event.
- A fresh `Conversation` is installed. `runner.conversation_id` now
  returns a different value.
- The next `runner.run()` emits a new `ConversationStarted` event
  before its `RunStarted`.

Use this whenever a follow-up should *not* carry prior context: the
user switched tasks, a new support session started, a test needs
isolation between cases.

## Injecting a pre-built Conversation

For persistence-aware hosts (e.g. rehydrating a chat session from
storage), construct a `Conversation` directly and pass it to the
`Runner`:

```python
conv = goldfive.Conversation.new()
conv.goals = [goldfive.Goal(id="g1", summary="Original ask")]
conv.completed_results = {"prior_task": "prior output"}

runner = goldfive.Runner(
    agent=my_adapter,
    planner=my_planner,
    executor=my_executor,
    conversation=conv,
)
```

The Runner's first `run()` will see that pre-loaded state exactly as
if it had been produced by an earlier turn on the same Runner.

## Observability

Phase 3 adds two event payloads to the `Event` envelope:

- `ConversationStarted` — emitted once per Conversation on the first
  turn that uses it. Carries `conversation_id` and `started_at`.
- `ConversationEnded` — emitted when `new_conversation()` is called or
  when `runner.close()` runs after a used Conversation. Carries
  `conversation_id`, `turn_count`, and a short `reason` string.

Both ride the existing `Event` envelope (proto regenerates via
`make proto`) so every sink — `LoggingSink`, `JSONLPersistenceSink`,
`SQLitePersistenceSink`, `GRPCSink` — picks them up with no code
change.

Note: the envelope's `run_id` on a `ConversationStarted`/`Ended` is
the anchoring turn's run_id (first turn / last turn respectively). The
stable conversation identifier is inside the payload.

## What's *not* in Phase 3

- **Cross-turn plan lineage.** Each turn still plans from scratch;
  v2 will expose the prior plan DAG to the planner so it can revise
  rather than replace. Today the LLM sees prior `completed_results` as
  text context instead.
- **Persistence across process restarts.** A Runner's Conversation
  lives in memory. Use `JSONLPersistenceSink` to log the event stream;
  `Runner.resume()` can replay it, but rehydrating a live Conversation
  from a log is a Phase 3.5 follow-up.
- **Multi-user conversation routing.** One Runner, one Conversation.
  Multi-user systems should build a per-user Runner pool.

## Overlay model + multi-turn under adk-web

Under `adk web`, each user turn in the UI is a fresh
`GoldfiveADKAgent._run_async_impl` call, which translates to one
`runner.run(user_text, session_id=outer_sid)` pass. The outer
adk-web session id (`ctx.session.id`) is stable across turns on the
same URL, so:

- Goldfive Session.id stays constant (pinned to outer id,
  goldfive#161) — one harmonograf session row carries every turn.
- Goldfive Conversation state accumulates normally: turn N sees
  turn N-1's `completed_results` rendered into the planner prompt.
- Each turn's plan is independent — different `plan.id`,
  `revision_index` restarts at 0 — but harmonograf's UI shows them
  all on the same session timeline.

Overlay STEER mid-turn is an **in-turn restart** (the executor
cancels the in-flight invoke and re-dispatches with the corrective
message composed by the steerer). It does not create a new turn
from the user-facing perspective — the harmonograf UI sees one
continuous span with a PlanRevised marker partway through.

## Summary

| Question                                     | Answer                                            |
|----------------------------------------------|---------------------------------------------------|
| Do I need to change anything for single-turn?| No — the default Conversation is invisible.       |
| Does turn 2 see turn 1's results?            | Yes, on `session.completed_results`.              |
| Does turn 2's planner see turn 1's output?   | Yes, via rendered context in the prompt.          |
| How do I reset state?                        | `await runner.new_conversation()`.                |
| How do I check the current id?               | `runner.conversation_id`.                         |
| What lifecycle events fire?                  | `ConversationStarted` / `ConversationEnded`.      |
