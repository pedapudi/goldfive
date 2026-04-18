"""Generate goldfive proto stubs on demand for steerer / reporting tests.

This is imported (and invoked) only by the test modules that actually need
the stubs. It intentionally lives outside ``conftest.py`` so modules with
their own skip logic (``tests/test_conv.py``) are not forced to see the
generated stubs — their skip predicate re-runs at their import time, and we
don't want to flip their behaviour from "skipped" to "fail due to bugs in
the sibling PR this PR consumes".

When issue #3 lands and ``goldfive/pb`` is populated by ``make proto``,
this helper is a no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys


def _pb_available() -> bool:
    try:
        return (
            importlib.util.find_spec("goldfive.pb.goldfive.v1.events_pb2") is not None
        )
    except (ModuleNotFoundError, ImportError):
        return False


def ensure_pb_available() -> bool:
    """Return True iff ``goldfive.pb.goldfive.v1.*`` is importable.

    Attempts one on-demand compilation of the vendored ``.proto`` sources
    under ``tests/_proto/`` if ``grpc_tools.protoc`` is installed.
    """
    if _pb_available():
        return True

    try:
        from grpc_tools import protoc  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return False

    tests_dir = pathlib.Path(__file__).resolve().parent
    proto_root = tests_dir / "_proto"
    if not proto_root.exists():
        return False

    repo_root = tests_dir.parent
    out_dir = repo_root / "goldfive" / "pb"
    out_dir.mkdir(parents=True, exist_ok=True)
    # protoc emits absolute imports of the form ``from goldfive.v1 import
    # types_pb2`` inside the generated modules. Because the outer package is
    # also named ``goldfive``, those imports would normally fail: we extend
    # the outer package's ``__path__`` from this inner ``__init__.py`` so
    # ``goldfive.v1`` resolves to ``goldfive/pb/goldfive/v1/``.
    init_body = (
        "from __future__ import annotations\n"
        "import os as _os\n"
        "import sys as _sys\n"
        "_inner = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'goldfive')\n"
        "_outer = _sys.modules.get('goldfive')\n"
        "if _outer is not None and hasattr(_outer, '__path__'):\n"
        "    if _inner not in list(_outer.__path__):\n"
        "        _outer.__path__.append(_inner)\n"
    )
    (out_dir / "__init__.py").write_text(init_body)
    (out_dir / "goldfive").mkdir(parents=True, exist_ok=True)
    (out_dir / "goldfive" / "__init__.py").touch(exist_ok=True)
    (out_dir / "goldfive" / "v1").mkdir(parents=True, exist_ok=True)
    (out_dir / "goldfive" / "v1" / "__init__.py").touch(exist_ok=True)

    proto_files = list(proto_root.rglob("*.proto"))
    if not proto_files:
        return False

    grpc_tools_pkg = pathlib.Path(protoc.__file__).resolve().parent
    well_known = grpc_tools_pkg / "_proto"

    args = [
        "grpc_tools.protoc",
        f"--proto_path={proto_root}",
        f"--proto_path={well_known}",
        f"--python_out={out_dir}",
        f"--pyi_out={out_dir}",
        *[str(p) for p in proto_files],
    ]
    rc = protoc.main(args)
    if rc != 0:
        return False

    importlib.invalidate_caches()
    # Force ``goldfive`` namespace re-scan so the new subpackage is picked up.
    if "goldfive" in sys.modules:
        importlib.reload(sys.modules["goldfive"])
    return _pb_available()
