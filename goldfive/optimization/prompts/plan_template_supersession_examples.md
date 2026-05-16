# Plan-template supersession examples

Role
----
Pinned to `goldfive.planner._SUPERSESSION_EXAMPLES`. Few-shot examples
paired with the supersession invariant (positive REPLACE + negative
EVOLUTION) to anchor the mutual-exclusivity framing. Without the
negative example, the LLM tends to over-apply `supersedes` on
legitimate id-reuse-with-evolved-title cases.

This is a plan-output *template fragment*: it shapes the JSON the
planner is constrained to emit. Tuning the examples lets optimizers
shift the planner toward more or less aggressive supersession.

Required placeholders: none.

---
EXAMPLES:

  Reused id (evolution; no supersedes):
    Prior: {"id": "research_solar", "title": "Research solar panels"}
    New:   {"id": "research_solar", "title": "Research solar + battery
            costs"}
    No supersedes — id reuse already encodes that this is the same
    logical step.

  New id with supersedes (replacement; ANY name shape):
    Prior: {"id": "review_slides", "status": "FAILED"}
    New:   {"id": "fix_review_slides", "title": "Re-do slide review
            with cleaner outline", "supersedes": "review_slides",
            "supersedes_kind": "REPLACE"}
    The id `fix_review_slides` is fresh; supersedes is REQUIRED. The
    same applies to `redo_review_slides`, `review_slides_again`,
    `slide_review_2`, etc.
