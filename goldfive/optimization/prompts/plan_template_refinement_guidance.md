# Plan-template refinement guidance

Role
----
Pinned to `goldfive.planner._REFINEMENT_GUIDANCE_BLOCK`. Embedded in
the planner's refine system prompts as the per-task default-pattern
guidance — "small drift means swap the drifted task with a corrected
variant; reshape only when the work itself needs it".

This is a plan-output *template fragment*: it shapes the JSON the
planner is constrained to emit. Tuning here lets optimizers move the
planner toward more conservative or more aggressive plan reshaping
without touching any drift-specific prompt.

Required placeholders: none.

---
REFINEMENT GUIDANCE:
- The drift you're correcting is usually "small" — a single agent produced off-topic or flawed output on a task it's otherwise capable of. Default pattern: replace the drifted task with a corrected variant and populate `supersedes: <old_task_id>` on the replacement. Leave `assignee_agent_id` empty on the replacement — the framework populates it observationally (goldfive#252). Preserve the surrounding DAG structure (edges, sibling tasks, stage count).
- Only reshape the plan (collapse stages, drop tasks) when the drift indicates the work itself needs restructuring — repeated failures on the same task, tool errors that can't be recovered from, or a pattern the prior shape of the work has already failed at.
- Do NOT collapse a multi-stage plan to a single task unless the user request genuinely warrants it.
- The `supersedes` field is required on every replacement — it's how runtime routing re-pins reports from the old task to the new one.
- The `supersedes_kind` field MUST accompany `supersedes`:
  * REPLACE when the superseded task was PENDING / RUNNING / FAILED / CANCELLED (the typical retry).
  * CORRECT when the superseded task is already COMPLETED but its output is drift-contaminated (the agent wandered off-topic yet still signalled completion). CORRECT keeps the old task in the plan as a historical COMPLETED node and adds an edge old -> new so downstream work flows through the correction.
