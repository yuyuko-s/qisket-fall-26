"""Provenance stamps recorded with every run: package versions and git hash."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

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


# --------------------------------------------------------------------------------------
# Environment check, run as the first cell of every notebook.
# --------------------------------------------------------------------------------------

class ProjectEnvironmentError(RuntimeError):
    """The running Python is not a usable project environment (see README.md, Setup)."""


def pinned_versions(requirements: Path | None = None) -> dict[str, str]:
    """{distribution: version} for every `name==version` line of requirements.txt."""
    requirements = requirements or REPO_ROOT / "requirements.txt"
    pins = {}
    for line in requirements.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, pinned = line.split("==", 1)
            pins[name.strip()] = pinned.strip()
    return pins


def check_environment(requirements: Path | None = None, python: str = "3.12") -> str:
    """Check that this is the project environment and return a one-line summary.

    Raises ProjectEnvironmentError if the package is not an editable install of this
    repository or a pinned package is missing. Version differences only warn: the code
    still runs, but numbers may differ slightly from the reference results.
    """
    if not (REPO_ROOT / "pyproject.toml").exists():
        raise ProjectEnvironmentError(
            f"qm9dipole is installed from {REPO_ROOT}, not from a repository checkout. Install it "
            "in editable mode from the repo root: pip install -r requirements.txt"
        )
    pins = pinned_versions(requirements)
    missing, differ = [], []
    for name, pinned in pins.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        if installed != pinned:
            differ.append(f"{name} {installed} (pinned {pinned})")
    if missing:
        raise ProjectEnvironmentError(
            f"missing packages {missing}: this kernel is not the project environment. "
            "Follow README.md (Setup) and select that environment as the notebook kernel."
        )
    py = platform.python_version()
    if not py.startswith(python + "."):
        differ.append(f"Python {py} (reference {python})")
    if differ:
        warnings.warn(
            "environment differs from requirements.txt; results may differ slightly: "
            + "; ".join(differ),
            stacklevel=2,
        )
    return (f"environment OK: Python {py}, {len(pins)} pinned packages checked, "
            f"{len(differ)} differ; repo {REPO_ROOT}")
