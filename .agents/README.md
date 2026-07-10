---
name: agents-index
description: Index of agent-facing skills for goldfive — what's here and when to read which.
---

# `.agents/` — agent-facing skills

Canonical "how do I X" guides for agents (coding-assistant sessions,
internal tools, IDE integrations) that need to use, extend, or diagnose
goldfive.
Each skill is a short, focused markdown file whose snippets run against
current `main`.

These are distinct from `docs/`. `docs/` is human prose organised for
learning and reference. `.agents/` is terse, task-shaped, and optimised
for a sibling agent that wants the shortest correct path from intent to
action.

## Deep reference: `docs/dev-guide/`

These skills point; the guide explains. When a task needs more than the
shortest path — the *why*, the blast radius, the invariant that will bite
you — go to [`docs/dev-guide/`](../docs/dev-guide/), a 17-chapter
developer's guide grounded in current `main`. Start at
[`docs/dev-guide/00-index.md`](../docs/dev-guide/00-index.md) for the
routing table (task → chapters).

**Read-first rule:** before any nontrivial edit to goldfive itself, read
[`17-invariants-hazards-history.md`](../docs/dev-guide/17-invariants-hazards-history.md)
(the six hard invariants, the Protected List, the Deferred-Work Register,
the hazard catalog, the pre-PR checklist). For common extensions, follow
the matching recipe in
[`16-recipes.md`](../docs/dev-guide/16-recipes.md) rather than
reconstructing the steps.

Routing hints (skill → guide chapter): use/wrap → user guides; develop →
16 + 17; debug → 09/12/08; adapters → 06 + Recipe 10; sinks/events → 12 +
Recipes 6-7; testing → 15; add-a-drift-kind → Recipe 1 (07/08, 09);
filler-loop → 07/09/12.

## When to read which

| I want to… | Read |
|---|---|
| wrap my existing agent with goldfive | [use-goldfive.md](use-goldfive.md) |
| add a feature, adapter, or sink to goldfive | [develop-goldfive.md](develop-goldfive.md) |
| diagnose a broken run (overlay + ladder + session id) | [debug-goldfive.md](debug-goldfive.md) |
| understand or implement an `AgentAdapter` (incl. overlay `invoke_passthrough`) | [adapters.md](adapters.md) |
| understand or implement an `EventSink` (incl. per-event session_id routing) | [sinks.md](sinks.md) |
| know what events exist and how to emit one | [events.md](events.md) |
| write a goldfive test | [testing.md](testing.md) |
| cut a release | [release.md](release.md) |
| add a new `AgentAdapter` wrapping a new framework | [how-to-add-a-new-adapter.md](how-to-add-a-new-adapter.md) |
| debug a filler loop (tool-loop detector, intervention ladder) | [how-to-debug-a-filler-loop.md](how-to-debug-a-filler-loop.md) |
| add a new `DriftKind` (proto reservations included) | [how-to-add-a-drift-kind.md](how-to-add-a-drift-kind.md) |
| drive a goldfive end-to-end run with harmonograf + adk-web + kikuchi + steer.py | [how-to-run-the-e2e.md](how-to-run-the-e2e.md) |

The set is stable. If you think you need a new skill, first check
whether one of the existing ones should be extended instead — skills
that overlap are worse than skills that cross-reference.

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
