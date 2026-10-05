"""Provenance stamps recorded with every run: package versions and git hash."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

from qm9dipole import REPO_ROOT

#: Distributions whose versions are recorded with every results file.
TRACKED_PACKAGES: tuple[str, ...] = (
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "pyarrow",
    "matplotlib",
    "pyyaml",
    "pytest",
)


def package_versions() -> dict[str, str]:
    """Return {distribution: version} for Python and every tracked package."""
    out = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "MISSING"
    return out


def git_hash(short: bool = False) -> str:
    """Return the current commit hash, with "-dirty" appended if the tree has changes.

    Returns "nocommit" before the first commit and "nogit" outside a repository.
    """
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)

    head = run("rev-parse", "--short" if short else "--verify", "HEAD")
    if head.returncode != 0:
        return "nogit" if run("rev-parse", "--git-dir").returncode != 0 else "nocommit"
    dirty = run("status", "--porcelain").stdout.strip()
    return head.stdout.strip() + ("-dirty" if dirty else "")


def config_hash(config: dict) -> str:
    """Short, order-independent hash of a config dict (12 hex chars of SHA-256)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
