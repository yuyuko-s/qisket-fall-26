"""Batched exact simulation (models/qsim.py) against Qiskit's own Statevector."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from qm9dipole.models import qsim
from qm9dipole.models.qsim import BatchedCircuit, fidelity_kernel, pauli_expectations, zz_expectations
from qm9dipole.models.quantum import build_encoding_circuit


def _reference(circuit, A):
    return np.stack([Statevector(circuit.assign_parameters(row)).data for row in A])


def _every_gate_circuit():
    x = ParameterVector("x", 4)
    c = QuantumCircuit(3, global_phase=0.3 * x[3])
    c.h(0); c.x(1); c.y(2); c.z(0); c.s(1); c.sdg(2); c.t(0); c.tdg(1); c.sx(2)
    c.rx(x[0], 0); c.ry(2 * x[1] - 0.4, 1); c.rz(x[2] * x[3], 2); c.p(np.pi - x[0], 1)
    c.u(x[0], x[1] + 1.0, -x[2], 2)
    c.cx(0, 1); c.cz(1, 2); c.cp(x[3], 0, 2); c.rzz(x[0] * x[1], 1, 2); c.swap(0, 2)
    c.cx(0, 2); c.p(2 * (np.pi - x[1]) * (np.pi - x[2]), 2); c.cx(0, 2)   # parity-phase pattern
    c.cx(1, 0); c.rz(x[3], 0); c.cx(1, 0)
    c.barrier(); c.id(0)
    c.cx(2, 1); c.p(x[0], 0); c.cx(2, 1)  # not the pattern (phase on another qubit)
    return c


def test_every_supported_gate_matches_qiskit_with_global_phase():
    c = _every_gate_circuit()
    A = np.random.default_rng(0).normal(size=(7, 4))
    np.testing.assert_allclose(BatchedCircuit(c).statevectors(A), _reference(c, A), atol=1e-12)


@pytest.mark.parametrize("encoding, n_features, n_qubits, reps",
                         [("zz", 5, 5, 2), ("zz", 7, 3, 3), ("ry", 6, 6, 1), ("ry", 9, 4, 2),
                          ("ry_rz", 6, 3, 2), ("ry_rz", 5, 2, 1)])
def test_team_encoders_match_qiskit(encoding, n_features, n_qubits, reps):
    c = build_encoding_circuit(n_features, n_qubits, encoding=encoding, reps=reps)
    A = np.random.default_rng(1).normal(scale=1.5, size=(9, n_features))
    np.testing.assert_allclose(BatchedCircuit(c).statevectors(A), _reference(c, A), atol=1e-12)


def test_chunking_gives_identical_states(monkeypatch):
    c = build_encoding_circuit(4, 4, encoding="zz", reps=2)
    A = np.random.default_rng(2).normal(size=(37, 4))
    whole = BatchedCircuit(c).statevectors(A)
    monkeypatch.setattr(qsim, "CHUNK_AMPLITUDES", 16 * 5)  # 5 rows per chunk
    np.testing.assert_array_equal(BatchedCircuit(c).statevectors(A), whole)


def test_pauli_and_zz_expectations_match_qiskit():
    c = _every_gate_circuit()
    A = np.random.default_rng(3).normal(size=(5, 4))
    psi = BatchedCircuit(c).statevectors(A)
    ev = pauli_expectations(psi)
    zz = zz_expectations(psi, [(0, 1), (0, 2)])
    for b, row in enumerate(A):
        sv = Statevector(c.assign_parameters(row))
        for q in range(3):
            for P in "XYZ":
                op = SparsePauliOp.from_sparse_list([(P, [q], 1.0)], num_qubits=3)
                assert ev[P][b, q] == pytest.approx(sv.expectation_value(op).real, abs=1e-12)
        for k, (a, bb) in enumerate([(0, 1), (0, 2)]):
            op = SparsePauliOp.from_sparse_list([("ZZ", [a, bb], 1.0)], num_qubits=3)
            assert zz[b, k] == pytest.approx(sv.expectation_value(op).real, abs=1e-12)


def test_fidelity_kernel_is_a_valid_kernel():
    c = build_encoding_circuit(5, 5, encoding="zz", reps=2)
    psi = BatchedCircuit(c).statevectors(np.random.default_rng(4).normal(size=(30, 5)))
    K = fidelity_kernel(psi)
    np.testing.assert_allclose(K, K.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert (K >= 0).all() and (K <= 1 + 1e-12).all()
    # against the textbook definition for one pair
    assert K[3, 7] == pytest.approx(abs(np.vdot(psi[3], psi[7])) ** 2, abs=1e-14)


def test_unsupported_gate_and_non_polynomial_angle_raise():
    c = QuantumCircuit(2)
    c.ch(0, 1)
    with pytest.raises(NotImplementedError, match="ch"):
        BatchedCircuit(c)
    a = Parameter("a")
    c = QuantumCircuit(1)
    c.rz(a.sin(), 0)
    with pytest.raises(NotImplementedError, match="polynomial"):
        BatchedCircuit(c)


def test_wrong_parameter_width_raises():
    c = build_encoding_circuit(3, 3, encoding="ry", reps=1)
    with pytest.raises(ValueError, match="parameters"):
        BatchedCircuit(c).statevectors(np.zeros((2, 4)))


# --- The team's quantum-feature model with the batched simulator --------------------------

@pytest.mark.parametrize("encoding, n_features, n_qubits", [("zz", 4, 4), ("zz", 7, 3), ("ry", 5, 5),
                                                          ("ry", 7, 3), ("ry_rz", 6, 3), ("ry_rz", 3, 5)])
def test_batched_method_matches_the_statevector_estimator(encoding, n_features, n_qubits):
    from qm9dipole.models.quantum import QuantumFeatureTransformer, QuantumRidgeRegressor

    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(12, n_features)), rng.uniform(0, 5, size=12)
    params = dict(n_qubits=n_qubits, encoding=encoding, reps=2, gamma=0.6)
    ref = QuantumFeatureTransformer(**params, simulation_method="statevector").fit(X).transform(X)
    fast = QuantumFeatureTransformer(**params, simulation_method="batched").fit(X).transform(X)
    np.testing.assert_allclose(fast, ref, atol=1e-12)
    a = QuantumRidgeRegressor(**params, alpha=0.3, clip_negative=False, simulation_method="statevector").fit(X, y)
    b = QuantumRidgeRegressor(**params, alpha=0.3, clip_negative=False, simulation_method="batched").fit(X, y)
    np.testing.assert_allclose(b.predict(X[::-1]), a.predict(X[::-1]), atol=1e-10)


def test_batched_method_clones_and_grid_searches():
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    from qm9dipole.models.quantum import QuantumRidgeRegressor

    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(30, 4)), rng.uniform(0, 5, size=30)
    est = QuantumRidgeRegressor(simulation_method="batched", clip_negative=False)
    assert clone(est).get_params()["simulation_method"] == "batched"
    search = GridSearchCV(est, {"gamma": [0.2, 0.8], "alpha": [0.1, 1.0]}, cv=3).fit(X, y)
    assert np.isfinite(search.best_score_)
