"""Namespace wrapper for generated goldfive proto stubs.

``grpc_tools.protoc`` emits absolute imports like
``from goldfive.v1 import types_pb2`` inside the generated ``*_pb2.py``
modules because the proto ``package goldfive.v1;`` declaration drives the
import path. But our Python package is *also* named ``goldfive``, so those
imports would normally resolve against the outer package and fail.

To bridge this we extend the outer ``goldfive`` package's ``__path__`` to
also include the inner ``goldfive/pb/goldfive/`` directory, so
``goldfive.v1`` resolves to ``goldfive/pb/goldfive/v1/`` at import time.
This is a namespace-package trick, not sys.path manipulation — the outer
package retains its own modules, and we just graft the generated subtree
onto it.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_inner = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "goldfive")
_outer = _sys.modules.get("goldfive")
if _outer is not None and hasattr(_outer, "__path__"):
    if _inner not in list(_outer.__path__):
        _outer.__path__.append(_inner)
