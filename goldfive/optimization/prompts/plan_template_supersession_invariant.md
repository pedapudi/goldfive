# Plan-template supersession invariant

Role
----
Pinned to `goldfive.planner._SUPERSESSION_INVARIANT`. Embedded
verbatim in the planner's refine / steer / looping / divergence system
prompts so the planner LLM receives one consistent reuse-or-supersede
instruction regardless of drift kind.

This is a plan-output *template fragment*: it shapes the JSON the
planner is constrained to emit (the `supersedes` / `supersedes_kind`
fields on each task). Tuning here lets optimizers move the precision /
recall of supersession detection without touching the drift-specific
prompts.

Required placeholders: none.

---
SUPERSESSION INVARIANT — REUSE-OR-SUPERSEDE (mutually exclusive):

Whenever a task in your response carries forward, retries, fixes, or
re-does work from a prior task, you MUST pick exactly one of these two
shapes — never both, never neither:

  (a) REUSE THE PRIOR ID. The continuing task keeps the prior task's
      `id`. Retitle / rewrite / reassign as needed; identity is the id.
      Do NOT set `supersedes` (id reuse already encodes continuity).
      A COMPLETED task cannot regress to PENDING under a reused id —
      if you need to redo completed work, use shape (b) with
      `supersedes_kind: CORRECT`, not id reuse.

  (b) MINT A NEW ID + SUPERSEDES. The continuing task gets a fresh `id`
      AND `"supersedes": "<prior_id>"` AND `"supersedes_kind"`. Use this
      whenever the new id differs from the prior id for ANY reason —
      terminal failure (FAILED/CANCELLED), terminal cancellation,
      structural retry (`retry_X`, `X_v2`), corrective fix
      (`fix_X`, `redo_X`, `revised_X`, etc.), renamed evolution, or
      replacement under a different agent. The naming convention does
      NOT matter; if a new task semantically replaces an older one and
      its `id` is different, `supersedes` is REQUIRED.

`supersedes_kind` rule:
  * REPLACE — superseded task is PENDING / RUNNING / BLOCKED /
    FAILED / CANCELLED (the new task takes its slot).
  * CORRECT — superseded task is COMPLETED but its output is
    drift-contaminated (the old task stays in the plan as a historical
    COMPLETED node; an edge old -> new is added).

Forgetting `supersedes` on a renamed replacement is a runtime bug: the
executor cannot link the new task to the old one, the predecessor is
treated as fatally-failed, and the run aborts even though the
replacement is healthy.
