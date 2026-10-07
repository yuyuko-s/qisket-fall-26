"""Finite shots, noisy simulation (noise.py) and resource accounting (cost.py), on local
simulators only."""

import numpy as np
import pytest

from qm9dipole import cost, noise
from qm9dipole.models.quantum import build_encoding_circuit
from qm9dipole.models.quantum_kernel import FidelityKernel, bloch_vectors
from qm9dipole.models.qsim import BatchedCircuit, fidelity_kernel


def test_shot_kernel_is_unbiased_with_binomial_spread():
    K = np.array([[1.0, 0.3], [0.3, 1.0]])
    rng = np.random.default_rng(0)
    draws = np.stack([noise.shot_kernel(K, 1000, rng) for _ in range(4000)])
    assert draws[:, 0, 1].mean() == pytest.approx(0.3, abs=0.002)
    assert draws[:, 0, 1].std() == pytest.approx(np.sqrt(0.3 * 0.7 / 1000), rel=0.05)
    assert (draws[:, 0, 0] == 1.0).all()


def test_shot_expectations_are_unbiased_and_bounded():
    E = np.array([-0.8, 0.0, 0.5, 1.0])
    rng = np.random.default_rng(1)
    draws = np.stack([noise.shot_expectations(E, 500, rng) for _ in range(3000)])
    np.testing.assert_allclose(draws.mean(axis=0), E, atol=0.005)
    assert np.abs(draws).max() <= 1.0


def test_repair_kernel_gives_a_valid_kernel():
    K = noise.shot_kernel(np.full((30, 30), 0.02) + 0.98 * np.eye(30), 50, np.random.default_rng(2))
    R = noise.repair_kernel(K)
    np.testing.assert_allclose(R, R.T, atol=1e-15)
    assert np.linalg.eigvalsh(R).min() > -1e-10


def test_ideal_overlap_circuits_agree_with_exact_kernel_values():
    """PLAN §7.5 validation of the fast path: z-scores of Aer shot estimates."""
    enc = build_encoding_circuit(3, 3, encoding="zz", reps=2)
    rng = np.random.default_rng(3)
    A, B = 0.4 * rng.normal(size=(25, 3)), 0.4 * rng.normal(size=(25, 3))
    bc = BatchedCircuit(enc)
    exact = np.abs(np.einsum("ij,ij->i", bc.statevectors(A).conj(), bc.statevectors(B))) ** 2
    shots = 4000
    est = noise.sample_overlaps(enc, A, B, shots, seed=5)
    z = (est - exact) / np.sqrt(np.clip(exact * (1 - exact), 1e-4, None) / shots)
    assert np.abs(z).max() < 4.5


def test_ideal_basis_circuits_agree_with_exact_bloch_vectors():
    enc = build_encoding_circuit(3, 3, encoding="ry_rz", reps=1)
    A = np.random.default_rng(4).normal(size=(6, 3))
    exact = bloch_vectors(BatchedCircuit(enc).statevectors(A))
    est = noise.sample_bloch(enc, A, 20000, seed=6)
    assert np.abs(est - exact).max() < 4.5 * np.sqrt(1 / 20000)
    par = noise.sample_bloch(enc, A, 20000, seed=6, n_jobs=2, chunk=4)  # chunked rows keep their order
    assert par.shape == exact.shape and np.abs(par - exact).max() < 4.5 * np.sqrt(1 / 20000)
    z_only = noise.sample_bloch(enc, A, 20000, seed=6, bases="Z")
    assert z_only.shape == (6, 3) and np.abs(z_only - exact[:, 6:]).max() < 4.5 * np.sqrt(1 / 20000)


def test_noisy_overlaps_on_a_fake_backend_lose_fidelity():
    from qiskit_ibm_runtime.fake_provider import FakeFez

    enc = build_encoding_circuit(2, 2, encoding="zz", reps=2)
    A = 0.5 * np.random.default_rng(5).normal(size=(4, 2))
    est = noise.sample_overlaps(enc, A, A, 2000, backend=FakeFez(), seed=7)  # k(x, x) = 1 exactly
    assert ((est > 0.5) & (est < 1.0)).all()


def test_cost_accounting():
    enc = build_encoding_circuit(4, 4, encoding="zz", reps=2)
    c = cost.gate_counts(enc)
    assert c["qubits"] == 4 and c["two_qubit_gates"] == 2 * 2 * 3  # 3 pairs, CX-P-CX, 2 reps
    assert cost.circuits_needed("fidelity", 100, 10) == (4950, 1000)
    assert cost.circuits_needed("projected", 100, 10) == (300, 30)
    assert cost.qpu_seconds(10, 1000, 0.0) == pytest.approx(10 * 1000 * (cost.MEASURE_S + cost.REP_DELAY_S))


def test_transpiled_costs_on_heron():
    from qiskit_ibm_runtime.fake_provider import FakeFez

    t = cost.transpiled_costs(noise.overlap_circuit(build_encoding_circuit(3, 3, encoding="zz", reps=2)), FakeFez())
    assert t["qubits"] >= 3 and t["two_qubit_gates"] > 0 and t["duration_s"] > 0


def test_depolarizing_mitigation_inverts_a_depolarizing_channel():
    rng = np.random.default_rng(9)
    K = rng.uniform(0, 1, size=(4, 6))
    lam, n = np.array([0.9, 0.5, 0.7, 0.3]), 5
    noisy = lam[:, None] * K + (1 - lam[:, None]) / 2**n
    survival = lam + (1 - lam) / 2**n
    np.testing.assert_allclose(noise.depolarizing_mitigation(noisy, survival, n), K, atol=1e-12)


def test_parallel_overlap_sampling_matches_serial_layout():
    enc = build_encoding_circuit(2, 2, encoding="ry", reps=1)
    A = np.random.default_rng(10).normal(size=(9, 2))
    est = noise.sample_overlaps(enc, A, A, 200, seed=1, n_jobs=2, chunk=4)
    assert est.shape == (9,) and np.allclose(est, 1.0)  # k(x, x) = 1 without noise
