"""Helper that reports whether goldfive proto stubs are importable.

The stubs live at ``goldfive/pb/goldfive/v1/*_pb2.py`` and are regenerated
by ``make proto`` from the ``.proto`` sources under ``proto/``. They are
committed to the repo, so under normal conditions ``ensure_pb_available``
returns True on the first check.

The function exists because earlier in the project's life the stubs were
generated on demand before issue #3 landed; callers still import it to
skip tests gracefully if a future refactor leaves the stubs unavailable
(e.g. running from a dirty worktree after ``make clean`` removes them).
"""

from __future__ import annotations

import importlib.util


def _pb_available() -> bool:
    try:
        return importlib.util.find_spec("goldfive.pb.goldfive.v1.events_pb2") is not None
    except (ModuleNotFoundError, ImportError):
        return False


def ensure_pb_available() -> bool:
    """Return True iff ``goldfive.pb.goldfive.v1.*`` is importable."""
    return _pb_available()
