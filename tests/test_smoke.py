"""Smoke test: the package imports and exposes a version string."""

import goldfive


def test_version_is_set() -> None:
    assert goldfive.__version__
    assert isinstance(goldfive.__version__, str)


def test_version_matches_pyproject() -> None:
    assert goldfive.__version__ == "0.1.0"
