"""System-prompt rendering for :class:`goldfive.adapters.claude.ClaudeAgentSDKAdapter`.

The Claude Agent SDK has no shared ``session.state`` dict like Google ADK.
To give the agent the same live context ADK gets via callbacks, goldfive
re-renders a system prompt for every ``client.query(...)`` call. The
template is a plain ``str.format(...)`` template — no Jinja, no extra
dependencies — so it is trivially overrideable by callers.

Template placeholders (all four are required):

``{goal_block}``
    Bulleted list of goal summaries (or ``"(no goals declared)"``).

``{task_block}``
    The current task's id, title, and description.

``{plan_summary}``
    One-line summary of the active plan (may be empty).

``{completed_block}``
    Bulleted list of previously completed task ids and their result
    summaries (or ``"(none)"``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from goldfive.types import Goal, Task

# --------------------------------------------------------------------------- #
# Default template
# --------------------------------------------------------------------------- #

#: Default system-prompt template used when the adapter caller does not
#: supply one. Exported for documentation and override convenience — copy,
#: edit, and pass the result to ``ClaudeAgentSDKAdapter(system_prompt_template=...)``.
DEFAULT_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a goldfive-orchestrated agent. Stay on target.

# Goals
{goal_block}

# Active plan
{plan_summary}

# Already completed
{completed_block}

# Your current task
{task_block}

# How to report progress
You MUST call the goldfive reporting tools to keep the orchestrator
informed. At minimum:

  - Call `report_task_started` as soon as you begin work on the task.
  - Call `report_task_progress` with a fraction (0.0-1.0) for long steps.
  - When you finish, call exactly one of:
      * `report_task_completed(task_id, summary, artifacts=...)`
      * `report_task_failed(task_id, reason, recoverable=...)`
      * `report_task_blocked(task_id, blocker, needed=...)`
  - If you discover new work that is NOT part of your current task,
    call `report_new_work_discovered` instead of silently doing it.
  - If the plan no longer makes sense, call `report_plan_divergence`.

Do not ask the user clarifying questions — the orchestrator is the only
caller. Stay focused on the current task."""


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _format_goal_block(goals: Iterable[Goal]) -> str:
    items = list(goals)
    if not items:
        return "(no goals declared)"
    lines: list[str] = []
    for g in items:
        lines.append(f"- {g.id}: {g.summary}")
    return "\n".join(lines)


def _format_task_block(task: Task) -> str:
    description = task.description or "(no description)"
    return (
        f"id: {task.id}\n"
        f"title: {task.title}\n"
        f"description: {description}"
    )


def _format_completed_block(completed: Mapping[str, str]) -> str:
    if not completed:
        return "(none)"
    lines: list[str] = []
    for task_id, summary in completed.items():
        summary_text = summary or "(no summary)"
        lines.append(f"- {task_id}: {summary_text}")
    return "\n".join(lines)


def render_system_prompt(
    template: str | None,
    *,
    task: Task,
    goals: Iterable[Goal],
    plan_summary: str,
    completed: Mapping[str, str],
) -> str:
    """Render the adapter's system prompt.

    Passing ``template=None`` selects :data:`DEFAULT_SYSTEM_PROMPT_TEMPLATE`.
    The template must contain the four placeholders ``{goal_block}``,
    ``{task_block}``, ``{plan_summary}``, ``{completed_block}``; a
    ``KeyError`` from ``str.format`` surfaces unchanged so template bugs
    are noisy instead of silent.
    """

    body = template if template is not None else DEFAULT_SYSTEM_PROMPT_TEMPLATE
    return body.format(
        goal_block=_format_goal_block(goals),
        task_block=_format_task_block(task),
        plan_summary=plan_summary or "(no active plan)",
        completed_block=_format_completed_block(completed),
    )
