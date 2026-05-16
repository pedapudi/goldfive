# Reasoning-judge agent-tree system-prompt suffix

Role
----
Pinned to
:data:`goldfive.drift.reasoning_judge.AGENT_TREE_SYSTEM_PROMPT_SUFFIX`.
Appended to the reasoning-judge system prompt when the caller passes
`available_agents=...`, so the judge does not treat sub-agent delegation
as off-plan drift.

Required placeholders: none.

Body starts with a leading space; the suffix is concatenated directly
onto the base system prompt without an intervening separator.

---
 The user prompt may include an AGENT TREE section listing the wrapped agents and their parent/child relationships. If the agent in question invokes a known sub-agent (per that tree) to perform its assigned task, treat that as ON-TASK execution of the bound task — delegation is normal coordinator behaviour and is NOT a deviation. Mark a deviation only when the agent invokes something not in the tree, or when its reasoning wanders semantically away from the bound task and goals.
