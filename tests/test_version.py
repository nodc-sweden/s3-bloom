"""The package version is declared in three places that must agree."""

from __future__ import annotations

import tomllib
from pathlib import Path

import s3bloom

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _lockfile_version() -> str:
    with (PROJECT_ROOT / "uv.lock").open("rb") as fh:
        packages = tomllib.load(fh)["package"]
    return next(p["version"] for p in packages if p["name"] == "s3bloom")


class TestVersion:
    def test_package_matches_pyproject(self):
        assert s3bloom.__version__ == _pyproject_version()

    def test_lockfile_matches_pyproject(self):
        assert _lockfile_version() == _pyproject_version()

    def test_is_at_least_0_2(self):
        # Outputs are linear from 0.2 on; downstream scripts rely on
        # "s3bloom >= 0.2" meaning exactly that.
        major, minor = (int(part) for part in s3bloom.__version__.split(".")[:2])
        assert (major, minor) >= (0, 2)
