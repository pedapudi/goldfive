"""Optimization-manifest loader, validator, and dataclasses.

The manifest (``manifest.toml`` in this package) is the source-of-truth
inventory of every prompt + numeric knob downstream optimizers (e.g.
``pedapudi/zicato``) may mutate on goldfive's steering path.

This module turns that TOML into typed :class:`Mutation` records,
exposes a :class:`Manifest` container, and provides
:meth:`Manifest.validate` which checks proposed updates against the
per-entry constraints (required placeholders for prompts; numeric
range / type for thresholds).

The loader is deliberately small and dependency-free beyond the
standard library: it reads with :mod:`tomllib` (Python 3.11+).
"""

from __future__ import annotations

import dataclasses
import importlib.resources as _resources
import re
import tomllib
from collections.abc import Iterable, Mapping
from typing import Any, Literal

__all__ = [
    "Manifest",
    "ManifestLoadError",
    "Mutation",
    "ValidationError",
]


_VALID_KINDS: frozenset[str] = frozenset({"prompt", "numeric"})
_VALID_NUMERIC_TYPES: frozenset[str] = frozenset({"float", "int"})

# Module-attribute reference syntax: ``goldfive.foo.bar:NAME`` (module
# path before ``:``, attribute path — supporting one level of dotted
# nesting for classes — after). Accepts e.g. ``goldfive.drift.goals:GOAL_DRIFT_SYSTEM_PROMPT``
# or ``goldfive.drift_observer:DriftObserver.REFLECTIVE_SYSTEM_PROMPT``.
_PYTHON_ATTR_RE = re.compile(
    r"^[a-zA-Z_][\w.]*:[a-zA-Z_][\w.]*$"
)

# Source path syntax: either a markdown prompt path or a
# ``<module>.py:<ATTR>`` shorthand pointing at the live Python attribute.
_SOURCE_PROMPT_RE = re.compile(
    r"^goldfive/optimization/prompts/[A-Za-z0-9_]+\.md$"
)
_SOURCE_PYTHON_RE = re.compile(
    r"^goldfive(?:/[A-Za-z0-9_]+)+\.py:[a-zA-Z_][\w.]*$"
)


class ManifestLoadError(RuntimeError):
    """Raised when the manifest TOML is malformed or fails self-validation.

    Distinct from :class:`ValidationError` — :class:`ValidationError`
    instances describe a problem with a PROPOSED update, not with the
    manifest itself. A manifest that does not parse or contains an
    invalid entry is a packaging bug and raises eagerly.
    """


@dataclasses.dataclass(frozen=True)
class ValidationError:
    """One validation failure produced by :meth:`Manifest.validate`.

    Per-entry rather than aggregate: a single :meth:`Manifest.validate`
    call returns a list of errors so the optimizer can surface every
    rejected mutation in a batch, not just the first.

    ``mutation_id`` and ``code`` are short machine-readable strings;
    ``message`` is the human-readable explanation.
    """

    mutation_id: str
    code: str
    message: str


@dataclasses.dataclass(frozen=True)
class Mutation:
    """One mutation knob exposed by the manifest.

    Fields
    ------
    id:
        Stable short identifier (e.g. ``reasoning_judge_system_prompt``).
        Manifest-unique.
    kind:
        ``"prompt"`` or ``"numeric"``.
    source:
        Path-shaped string locating the canonical text for this knob —
        either a markdown file under
        ``goldfive/optimization/prompts/`` (for prompts) or a
        ``<module>.py:<ATTR>`` pointer (for numerics).
    python_attr:
        Dotted-path indirection to the live Python attribute the
        optimizer mutates with ``setattr``. Format:
        ``<module>:<ATTR>`` or ``<module>:<Class>.<ATTR>``.
    description:
        One-sentence summary of what the knob controls.
    required_placeholders:
        For ``kind="prompt"`` mutations: the literal placeholder
        substrings the new prompt body MUST contain (``{name}`` form).
        :meth:`Manifest.validate` rejects updates that drop any of
        these. Empty list for prompts whose body has no
        ``.format(...)`` placeholders (most system prompts).
    type:
        For ``kind="numeric"`` mutations: ``"float"`` or ``"int"``.
        Empty for prompts.
    range:
        For ``kind="numeric"`` mutations: ``(low, high)`` inclusive
        bounds the new value MUST land in. Empty for prompts.
    default:
        The repo's current value at manifest authorship time. Used by
        the test suite to catch drift between manifest and code.
    tags:
        Free-form labels for grouping (e.g. ``"judge"``, ``"reasoning"``).
    """

    id: str
    kind: Literal["prompt", "numeric"]
    source: str
    python_attr: str
    description: str
    required_placeholders: tuple[str, ...] = ()
    type: str = ""
    range: tuple[float, float] | tuple[()] = ()
    default: Any = None
    tags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Manifest:
    """The parsed manifest — a frozen ordered tuple of :class:`Mutation`.

    Use :meth:`load` to construct from the package's bundled TOML, or
    :meth:`from_text` for in-memory tests. :meth:`validate` checks a
    batch of proposed updates against per-entry constraints. :meth:`get`
    looks up a single mutation by id.
    """

    mutations: tuple[Mutation, ...]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> Manifest:
        """Load the bundled ``manifest.toml`` shipped with the package."""
        text = _resources.files(__package__).joinpath("manifest.toml").read_text(
            encoding="utf-8"
        )
        return cls.from_text(text)

    @classmethod
    def from_text(cls, text: str) -> Manifest:
        """Parse ``text`` (TOML) into a :class:`Manifest`.

        Raises :class:`ManifestLoadError` when the TOML is malformed or
        any entry fails structural validation (required fields missing,
        unknown kind, range malformed, etc.). Entry-level errors are
        eager because a broken manifest is a packaging bug — surfacing
        them at load time is preferable to silently dropping an entry.
        """
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ManifestLoadError(f"manifest TOML parse failed: {exc}") from exc
        entries = raw.get("mutation", [])
        if not isinstance(entries, list):
            raise ManifestLoadError(
                "manifest: top-level [[mutation]] array missing or malformed"
            )
        mutations: list[Mutation] = []
        seen: set[str] = set()
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ManifestLoadError(
                    f"manifest: entry #{i} is not a table: {entry!r}"
                )
            mut = _parse_entry(entry, position=i)
            if mut.id in seen:
                raise ManifestLoadError(
                    f"manifest: duplicate mutation id {mut.id!r}"
                )
            seen.add(mut.id)
            mutations.append(mut)
        return cls(mutations=tuple(mutations))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, mutation_id: str) -> Mutation:
        """Return the mutation with id ``mutation_id``.

        Raises :class:`KeyError` when no mutation matches.
        """
        for mut in self.mutations:
            if mut.id == mutation_id:
                return mut
        raise KeyError(mutation_id)

    def ids(self) -> tuple[str, ...]:
        """Return the ordered tuple of mutation ids."""
        return tuple(m.id for m in self.mutations)

    def filter_by_tag(self, tag: str) -> tuple[Mutation, ...]:
        """Return the subset of mutations carrying ``tag``."""
        return tuple(m for m in self.mutations if tag in m.tags)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        updates: Mapping[str, str | int | float],
    ) -> list[ValidationError]:
        """Validate a batch of proposed ``{mutation_id: new_value}`` updates.

        Returns a list of :class:`ValidationError` — empty when every
        update passes. Per-entry checks:

        * Unknown ``mutation_id`` → ``code="unknown_id"``.
        * ``kind="prompt"``:
            - Value must be a ``str`` → ``code="type"``.
            - Every entry in ``required_placeholders`` must appear in
              the new body verbatim → ``code="missing_placeholder"``.
            - The new body must not collapse to whitespace →
              ``code="empty_body"``.
        * ``kind="numeric"``:
            - For ``type="float"``: value must be ``int | float`` →
              ``code="type"``; must lie within ``range`` →
              ``code="out_of_range"``.
            - For ``type="int"``: value must be ``int`` (or a ``float``
              that is integer-valued) → ``code="type"``; must lie
              within ``range`` → ``code="out_of_range"``.
        """
        errors: list[ValidationError] = []
        for mutation_id, value in updates.items():
            try:
                mut = self.get(mutation_id)
            except KeyError:
                errors.append(
                    ValidationError(
                        mutation_id=mutation_id,
                        code="unknown_id",
                        message=(
                            f"no mutation with id {mutation_id!r} in the manifest"
                        ),
                    )
                )
                continue
            if mut.kind == "prompt":
                errors.extend(_validate_prompt(mut, value))
            else:  # numeric
                errors.extend(_validate_numeric(mut, value))
        return errors

    def __iter__(self) -> Iterable[Mutation]:
        """Iterate mutations in declaration order."""
        return iter(self.mutations)

    def __len__(self) -> int:
        return len(self.mutations)


# ---------------------------------------------------------------------------
# Per-entry parsers + validators
# ---------------------------------------------------------------------------


def _parse_entry(entry: Mapping[str, Any], *, position: int) -> Mutation:
    def _require(key: str) -> Any:
        if key not in entry:
            raise ManifestLoadError(
                f"manifest: entry #{position} missing required field {key!r}"
            )
        return entry[key]

    mutation_id = _require("id")
    if not isinstance(mutation_id, str) or not mutation_id:
        raise ManifestLoadError(
            f"manifest: entry #{position} has non-string id {mutation_id!r}"
        )
    kind = _require("kind")
    if kind not in _VALID_KINDS:
        raise ManifestLoadError(
            f"manifest: entry {mutation_id!r} has unknown kind {kind!r} "
            f"(expected one of {sorted(_VALID_KINDS)!r})"
        )
    source = _require("source")
    if not isinstance(source, str):
        raise ManifestLoadError(
            f"manifest: entry {mutation_id!r} has non-string source"
        )
    if kind == "prompt":
        if not _SOURCE_PROMPT_RE.match(source):
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} prompt source path "
                f"{source!r} must point at a markdown file under "
                "goldfive/optimization/prompts/"
            )
    else:
        if not _SOURCE_PYTHON_RE.match(source):
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} numeric source path "
                f"{source!r} must be a goldfive/<module>.py:<ATTR> pointer"
            )
    python_attr = _require("python_attr")
    if not isinstance(python_attr, str) or not _PYTHON_ATTR_RE.match(python_attr):
        raise ManifestLoadError(
            f"manifest: entry {mutation_id!r} python_attr {python_attr!r} "
            "must match '<module>:<ATTR>' (with optional dotted attribute path)"
        )
    description = entry.get("description", "")
    if not isinstance(description, str):
        raise ManifestLoadError(
            f"manifest: entry {mutation_id!r} description must be a string"
        )

    required_placeholders: tuple[str, ...] = ()
    if kind == "prompt":
        raw_ph = entry.get("required_placeholders", [])
        if not isinstance(raw_ph, list) or not all(
            isinstance(p, str) for p in raw_ph
        ):
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} required_placeholders "
                "must be a list of strings"
            )
        required_placeholders = tuple(raw_ph)

    numeric_type = ""
    numeric_range: tuple[float, float] | tuple[()] = ()
    default: Any = entry.get("default")
    if kind == "numeric":
        numeric_type = entry.get("type", "")
        if numeric_type not in _VALID_NUMERIC_TYPES:
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} numeric type "
                f"{numeric_type!r} must be one of {sorted(_VALID_NUMERIC_TYPES)!r}"
            )
        raw_range = entry.get("range")
        if (
            not isinstance(raw_range, list)
            or len(raw_range) != 2
            or not all(isinstance(v, (int, float)) for v in raw_range)
        ):
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} range must be a 2-element "
                f"[low, high] list of numbers; got {raw_range!r}"
            )
        low, high = float(raw_range[0]), float(raw_range[1])
        if low > high:
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} range low={low} > high={high}"
            )
        numeric_range = (low, high)
        if default is None:
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} numeric mutations must "
                "carry a 'default' field"
            )
        if not isinstance(default, (int, float)):
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} numeric default must be a number"
            )
        if not (low <= float(default) <= high):
            raise ManifestLoadError(
                f"manifest: entry {mutation_id!r} default {default} lies "
                f"outside declared range [{low}, {high}]"
            )

    tags_raw = entry.get("tags", [])
    if not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw):
        raise ManifestLoadError(
            f"manifest: entry {mutation_id!r} tags must be a list of strings"
        )

    return Mutation(
        id=mutation_id,
        kind=kind,  # type: ignore[arg-type]
        source=source,
        python_attr=python_attr,
        description=description,
        required_placeholders=required_placeholders,
        type=numeric_type,
        range=numeric_range,
        default=default,
        tags=tuple(tags_raw),
    )


def _validate_prompt(mut: Mutation, value: Any) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not isinstance(value, str):
        errors.append(
            ValidationError(
                mutation_id=mut.id,
                code="type",
                message=(
                    f"prompt mutation expects a string body; got {type(value).__name__}"
                ),
            )
        )
        return errors
    if not value.strip():
        errors.append(
            ValidationError(
                mutation_id=mut.id,
                code="empty_body",
                message="prompt body is empty / whitespace-only",
            )
        )
        # Don't bother checking placeholders on an empty body.
        return errors
    for placeholder in mut.required_placeholders:
        if placeholder not in value:
            errors.append(
                ValidationError(
                    mutation_id=mut.id,
                    code="missing_placeholder",
                    message=(
                        f"required placeholder {placeholder!r} not present in "
                        "proposed prompt body"
                    ),
                )
            )
    return errors


def _validate_numeric(mut: Mutation, value: Any) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; refuse it explicitly so an
        # optimizer that sends ``True`` for an int knob does not pass.
        errors.append(
            ValidationError(
                mutation_id=mut.id,
                code="type",
                message="numeric mutation rejected boolean value",
            )
        )
        return errors
    if mut.type == "int":
        if isinstance(value, int):
            numeric_value = float(value)
        elif isinstance(value, float) and value.is_integer():
            numeric_value = value
        else:
            errors.append(
                ValidationError(
                    mutation_id=mut.id,
                    code="type",
                    message=(
                        f"int mutation expects an integer; got {type(value).__name__} "
                        f"value {value!r}"
                    ),
                )
            )
            return errors
    elif mut.type == "float":
        if not isinstance(value, (int, float)):
            errors.append(
                ValidationError(
                    mutation_id=mut.id,
                    code="type",
                    message=(
                        f"float mutation expects a number; got {type(value).__name__}"
                    ),
                )
            )
            return errors
        numeric_value = float(value)
    else:  # pragma: no cover — guarded at load time
        errors.append(
            ValidationError(
                mutation_id=mut.id,
                code="type",
                message=f"unknown numeric type {mut.type!r} on manifest entry",
            )
        )
        return errors
    if not mut.range:  # pragma: no cover — guarded at load time
        return errors
    low, high = mut.range
    if not (low <= numeric_value <= high):
        errors.append(
            ValidationError(
                mutation_id=mut.id,
                code="out_of_range",
                message=(
                    f"value {value!r} outside declared range [{low}, {high}]"
                ),
            )
        )
    return errors
