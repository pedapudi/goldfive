# Reflective self-progress check user prompt

Role
----
Pinned to
:data:`goldfive.drift_observer.DriftObserver.REFLECTIVE_USER_PROMPT_TEMPLATE`.
Rendered with `str.format(...)` once per reflective check.

Required placeholders: `{task_id}`, `{task_title}`, `{task_description}`,
`{window}`, `{tool_call_summary}`, `{reasoning_summary}`.

JSON-shape examples in the body use `{{` / `}}` for literal braces; do
not collapse them.

---
You are assessing your own progress on a task.

CURRENT TASK:
id: {task_id}
title: {task_title}
description: {task_description}

WHAT YOU HAVE DONE IN THE LAST {window} LLM TURNS (summarized):
- recent tool calls: {tool_call_summary}
- recent reasoning (last 3 blocks): {reasoning_summary}

Q: Are you making forward progress on the task? Reply with a single JSON object:
{{"making_progress": true|false, "confidence": 0.0-1.0, "reason": "one-sentence explanation"}}
