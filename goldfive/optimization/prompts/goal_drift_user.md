# Goal-drift user prompt

Role
----
Pinned to :data:`goldfive.drift.goals.GOAL_DRIFT_USER_PROMPT_TEMPLATE`.
Rendered with `str.format(...)` per judge call. Asks the LLM-as-a-judge
whether the recent agent activity is plausibly advancing the goals.

Required placeholders: `{goals_block}`, `{tasks_block}`,
`{activity_count}`, `{activity_block}`.

JSON-shape examples in the body use `{{` / `}}` for literal braces; do
not collapse them.

---
You are assessing whether an autonomous agent tree is making progress toward a stated goal.

GOALS:
{goals_block}

PLANNED TASKS:
{tasks_block}

RECENT AGENT ACTIVITY (most recent {activity_count} invocations, newest last):
{activity_block}

Decide: is the recent activity moving toward the goals? Answer STRICTLY in one of these two JSON shapes:
{{"progressing": true}}
OR
{{"progressing": false, "reason": "one-sentence explanation", "note_to_agent": "one or two sentences addressed to the agent itself: state only what you observed and how it relates to the goals. Neutral and factual — no commands, no instructions about which task, tool, or agent to use next, and no fault language. If your confidence is low, phrase the note as a question (e.g. 'Does the current approach still serve the goal of X?')"}}

Progressing = agents are doing work that plausibly contributes to the goal.
Not progressing = agents are looping, refusing, off-topic, or otherwise not advancing.
