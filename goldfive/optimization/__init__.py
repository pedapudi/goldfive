"""Optimization surface for goldfive (goldfive#zicato-optimization-surface).

This package exposes the knobs downstream optimizers
(e.g. ``pedapudi/zicato``) mutate when tuning goldfive's steering
behaviour, in two shapes:

* :mod:`goldfive.optimization.manifest` — a typed inventory of every
  mutable knob (prompt template, numeric threshold) on the steering
  path. Optimizers read the manifest to discover what they can change
  and what the constraints on a proposed change are.
* :mod:`goldfive.optimization.prompts` — a markdown-backed catalog of
  the canonical prompt bodies plus a ``load(name)`` accessor that
  caches the parsed body in-process. The Python attributes on the
  drift / planner / goal-deriver modules remain the runtime source of
  truth; the markdown files are the optimizer-facing copy.

The package is intentionally additive: existing imports and behaviour
do not change. Nothing inside :mod:`goldfive.steerer`,
:mod:`goldfive.drift_observer`, :mod:`goldfive.planner`,
:mod:`goldfive.drift`, or :mod:`goldfive.goal_deriver` shifts onto this
surface — this package only describes those existing surfaces in a
machine-readable form.
"""

from __future__ import annotations

from goldfive.optimization.manifest import (
    Manifest,
    ManifestLoadError,
    Mutation,
    ValidationError,
)
from goldfive.optimization.prompts import (
    PromptNotFound,
    available_prompts,
)
from goldfive.optimization.prompts import (
    bind as bind_prompt,
)
from goldfive.optimization.prompts import (
    load as load_prompt,
)
from goldfive.optimization.prompts import (
    reset as reset_prompts,
)

__all__ = [
    "Manifest",
    "ManifestLoadError",
    "Mutation",
    "PromptNotFound",
    "ValidationError",
    "available_prompts",
    "bind_prompt",
    "load_prompt",
    "reset_prompts",
]
