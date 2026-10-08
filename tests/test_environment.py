"""Environment smoke tests: the stack imports, and Qiskit behaves the way the quantum models assume."""

import numpy as np
import pytest

from qm9dipole import REPO_ROOT
from qm9dipole.provenance import (
    ProjectEnvironmentError, check_environment, package_versions, pinned_versions,
)


def test_tracked_packages_installed():
    missing = [k for k, v in package_versions().items() if v == "MISSING"]
    assert not missing, f"missing packages: {missing}"


def test_qiskit_is_2x():
    import qiskit

    assert qiskit.__version__.startswith("2.")


def test_statevector_self_fidelity_is_one():
    # The fidelity kernel is k(x, x') = |<psi(x)|psi(x')>|^2, so k(x, x) must be 1.
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.rz(0.7, 0)
    qc.cx(0, 1)
    psi = Statevector(qc)
    assert abs(psi.inner(psi)) ** 2 == pytest.approx(1.0)


def test_aer_sampler_runs():
    # P(all zeros) estimated from shots is how a device measures kernel values.
    from qiskit import QuantumCircuit
    from qiskit_aer.primitives import SamplerV2

    qc = QuantumCircuit(2)
    qc.measure_all()
    counts = SamplerV2(seed=0).run([qc], shots=200).result()[0].data.meas.get_counts()
    assert counts == {"00": 200}
    assert np.isclose(counts["00"] / 200, 1.0)


def test_git_hash_ignores_outputs_but_not_inputs(tmp_path):
    import subprocess

    from qm9dipole.provenance import git_hash

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src.py").write_text("x = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.csv").write_text("a\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert not git_hash(repo=tmp_path).endswith("-dirty")
    (tmp_path / "results" / "r.csv").write_text("b\n")       # a run rewrote an output
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "f.png").write_text("png")       # and added a new one
    assert not git_hash(repo=tmp_path).endswith("-dirty")
    (tmp_path / "new_module.py").write_text("y = 2\n")       # untracked code is an input
    assert git_hash(repo=tmp_path).endswith("-dirty")
    (tmp_path / "new_module.py").unlink()
    (tmp_path / "src.py").write_text("x = 2\n")              # so is an edit to tracked code
    assert git_hash(repo=tmp_path).endswith("-dirty")


def test_save_result_writes_csv_and_provenance(tmp_path):
    import json

    import pandas as pd

    from qm9dipole.provenance import save_result

    path = save_result(pd.DataFrame({"a": [1, 2]}), "demo", out_dir=tmp_path, seed=7)
    assert pd.read_csv(path)["a"].tolist() == [1, 2]
    meta = json.loads((tmp_path / "demo.meta.json").read_text())
    assert meta["seed"] == 7 and "git_hash" in meta and meta["versions"]["qiskit"].startswith("2.")


# --- check_environment ---------------------------------------------------------------

def test_pinned_versions_reads_requirements():
    pins = pinned_versions()
    assert pins["qiskit"].startswith("2.")
    assert "-e ." not in pins and all("#" not in k for k in pins)


def test_check_environment_accepts_this_environment():
    # Version differences only warn, so this passes in any complete project environment.
    assert check_environment(workdir=REPO_ROOT / "notebooks").startswith("environment OK")


def test_check_environment_rejects_notebook_from_another_checkout(tmp_path):
    # The package in this kernel belongs to REPO_ROOT; a notebook elsewhere must not use it.
    other_checkout = tmp_path / "other-clone" / "notebooks"
    other_checkout.mkdir(parents=True)
    with pytest.raises(ProjectEnvironmentError, match="belongs to the checkout"):
        check_environment(workdir=other_checkout)


def test_check_environment_rejects_missing_package(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.0\ndefinitely-not-installed-pkg==1.0\n")
    with pytest.raises(ProjectEnvironmentError, match="definitely-not-installed-pkg"):
        check_environment(req, workdir=REPO_ROOT)


def test_check_environment_warns_on_version_mismatch(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==0.0.1  # deliberately wrong\n")
    with pytest.warns(UserWarning, match="numpy"):
        check_environment(req, workdir=REPO_ROOT)
