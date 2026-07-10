"""Guard: ``TERMINAL_TASK_STATUSES`` has exactly one definition.

``goldfive.types.TERMINAL_TASK_STATUSES`` is documented as the single
source of truth for terminal task statuses (TASK-LIFECYCLE.md §7.1).
Divergent copies have caused real bugs — the parallel executor's local
frozenset omitted ``NOT_NEEDED``, so reconciler-stamped tasks were
still scheduled for an LLM turn. This test walks the package AST and
rejects any assignment that rebuilds the set instead of aliasing the
canonical one.
"""

from __future__ import annotations

import ast
import pathlib

_CANONICAL = "TERMINAL_TASK_STATUSES"
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent / "goldfive"


def _is_alias_of_canonical(value: ast.expr | None) -> bool:
    """True when the assigned value merely references the canonical set."""
    if isinstance(value, ast.Name):
        return value.id == _CANONICAL
    if isinstance(value, ast.Attribute):
        return value.attr == _CANONICAL
    return False


def test_no_module_redefines_terminal_task_statuses() -> None:
    offenders: list[str] = []
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(_PKG_ROOT)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            for target in targets:
                if not isinstance(target, ast.Name) or _CANONICAL not in target.id:
                    continue
                # The one real definition lives in goldfive/types.py.
                if str(rel) == "types.py" and target.id == _CANONICAL:
                    continue
                # Module-local aliases of the canonical set are fine.
                if _is_alias_of_canonical(value):
                    continue
                offenders.append(f"{rel}:{node.lineno}: {target.id} redefined")
    assert not offenders, (
        "TERMINAL_TASK_STATUSES must only be defined in goldfive/types.py "
        "(import or alias it elsewhere):\n  " + "\n  ".join(offenders)
    )
