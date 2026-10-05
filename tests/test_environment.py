"""M0 smoke tests: the stack imports, and Qiskit behaves the way later milestones assume."""

import numpy as np
import pytest

from qm9dipole.provenance import package_versions


def test_tracked_packages_installed():
    missing = [k for k, v in package_versions().items() if v == "MISSING"]
    assert not missing, f"missing packages: {missing}"


def test_qiskit_is_2x():
    import qiskit

    assert qiskit.__version__.startswith("2.")


def test_statevector_self_fidelity_is_one():
    # The fidelity kernel in M4 is k(x, x') = |<psi(x)|psi(x')>|^2, so k(x, x) must be 1.
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.rz(0.7, 0)
    qc.cx(0, 1)
    psi = Statevector(qc)
    assert abs(psi.inner(psi)) ** 2 == pytest.approx(1.0)


def test_aer_sampler_runs():
    # P(all zeros) estimated from shots is how M5 measures kernel values.
    from qiskit import QuantumCircuit
    from qiskit_aer.primitives import SamplerV2

    qc = QuantumCircuit(2)
    qc.measure_all()
    counts = SamplerV2(seed=0).run([qc], shots=200).result()[0].data.meas.get_counts()
    assert counts == {"00": 200}
    assert np.isclose(counts["00"] / 200, 1.0)
