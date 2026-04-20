---
name: agents-index
description: Index of agent-facing skills for goldfive — what's here and when to read which.
---

# `.agents/` — agent-facing skills

Canonical "how do I X" guides for agents (Claude Code sessions, internal
tools, IDE integrations) that need to use, extend, or diagnose goldfive.
Each skill is a short, focused markdown file whose snippets run against
current `main`.

These are distinct from `docs/`. `docs/` is human prose organised for
learning and reference. `.agents/` is terse, task-shaped, and optimised
for a sibling agent that wants the shortest correct path from intent to
action.

## When to read which

| I want to… | Read |
|---|---|
| wrap my existing agent with goldfive | [use-goldfive.md](use-goldfive.md) |
| add a feature, adapter, or sink to goldfive | [develop-goldfive.md](develop-goldfive.md) |
| diagnose a broken run | [debug-goldfive.md](debug-goldfive.md) |
| understand or implement an `AgentAdapter` | [adapters.md](adapters.md) |
| understand or implement an `EventSink` | [sinks.md](sinks.md) |
| know what events exist and how to emit one | [events.md](events.md) |
| write a goldfive test | [testing.md](testing.md) |
| cut a release | [release.md](release.md) |
| add a new `AgentAdapter` wrapping a new framework | [how-to-add-a-new-adapter.md](how-to-add-a-new-adapter.md) |
| debug the "guards defined but not firing" class of bug | [how-to-debug-a-filler-loop.md](how-to-debug-a-filler-loop.md) |
| add a new `DriftKind` | [how-to-add-a-drift-kind.md](how-to-add-a-drift-kind.md) |
| drive a goldfive end-to-end run with harmonograf locally | [how-to-run-the-e2e.md](how-to-run-the-e2e.md) |

## How these get referenced

Other agents either load the file directly (`cat .agents/use-goldfive.md`)
or reference it by path from their own instructions. Keep each file
self-contained so a reader arriving cold can complete the task without
cross-loading the rest of the folder.

## Conventions

- Every snippet runs against current `main`. If you change goldfive's
  public surface, update the affected skill in the same PR.
- No emojis. Terse voice. Files under 200 lines.
- Prefer pointing at `docs/` for long-form reference over duplicating
  prose here.
