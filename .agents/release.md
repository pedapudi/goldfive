---
name: release
description: Cut a goldfive release — version bumps, changelog, proto regeneration, tag + push.
applies-when: ["cut a release", "bump version", "tag goldfive", "publish"]
---

# Release

Releases are manual in v0.1 — there's no automated publishing pipeline
yet. The process is short.

## Steps

1. **Pick the new version.** SemVer. Alpha line so far (`0.1.0`).
2. **Bump it in two places.**
   - `pyproject.toml` → `[project] version = "X.Y.Z"`.
   - `goldfive/__init__.py` → `__version__ = "X.Y.Z"`.
3. **Regenerate proto if `proto/*.proto` changed since the last
   release.**
   ```bash
   make proto
   ```
   Commit any drift under `goldfive/pb/`.
4. **Update `CHANGELOG.md`.** Add a new top section with the ISO date
   and group entries under `### Added`, `### Changed`, `### Fixed`.
   The existing `0.1.0` entry is the format reference.
5. **Confirm green.**
   ```bash
   uv run pytest -q
   uv run ruff check .
   ```
6. **Commit on a release branch, PR, review, merge `--admin --squash`.**
   Title: `Release vX.Y.Z`.
7. **Tag and push.**
   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

Tagging is not automated yet. A future PR will wire a GitHub Actions
release workflow (see issue tracker).

## Pre-flight checklist

- [ ] `pyproject.toml` version matches `goldfive/__init__.py` `__version__`.
- [ ] `CHANGELOG.md` has a dated entry for the new version.
- [ ] `make proto` was re-run if `proto/` changed.
- [ ] `uv run pytest -q` passes on a clean checkout.
- [ ] `uv run ruff check .` passes.
- [ ] README `Install` / `Hello goldfive` snippets still work.
- [ ] `docs/reference/api.md` re-export list matches `goldfive.__all__`.

## Quick reference

```bash
# version bumps
sed -i 's/^version = ".*"/version = "X.Y.Z"/' pyproject.toml
sed -i 's/^__version__ = ".*"/__version__ = "X.Y.Z"/' goldfive/__init__.py

# verify
grep ^version pyproject.toml
grep __version__ goldfive/__init__.py

# pre-flight
make proto && uv run pytest -q && uv run ruff check .

# tag
git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
```

## Common pitfalls

- Bumping one version string and not the other → PyPI metadata
  disagrees with `__version__`.
- Forgetting `make proto` after editing `proto/*.proto` → users of the
  new wheel see stale stubs.
- CHANGELOG entry missing a PR reference → harder to trace back what
  changed. Link the PR number (`#66`) for every bullet.
- Tagging before the release PR merges → tag points at a commit that
  isn't on `main`.

## Related

- [develop-goldfive.md](develop-goldfive.md) — dev loop and merge flow.
- [docs/performance.md](../docs/performance.md) — perf baseline to re-run before major releases.
- `CHANGELOG.md` — prior release entries for format reference.
