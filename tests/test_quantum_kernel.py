"""Quantum kernel ridge (models/quantum_kernel.py): valid kernels, the shared tuning protocol,
and invariance of quantum predictions under rotation, translation and relabeling."""

import json

import numpy as np
import pandas as pd
import pytest
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from qm9dipole.descriptors import MODEL_DESCRIPTORS, feature_frame
from qm9dipole.invariance import TRANSFORMS
from qm9dipole.models import classical
from qm9dipole.models.fitters import TabularFitter
from qm9dipole.models.quantum_kernel import (
    FidelityKernel, KernelRidgeFitter, ProjectedKernel, RBFKernel, ShotNoisyKernel, bloch_vectors, kernel_spectrum,
    kernel_target_alignment, offdiag_stats, quantum_gamma_grid,
)
from qm9dipole.models.qsim import BatchedCircuit


@pytest.fixture(scope="module")
def frame():
    """200 'molecules', 8 features, y a smooth positive function of 3 of them."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = 3 + np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - 0.4 * X[:, 2] + 0.05 * rng.normal(size=200)
    ids = np.arange(5000, 5200)
    return pd.DataFrame(X, index=ids, columns=[f"f{k}" for k in range(8)]), pd.Series(np.clip(y, 0.05, None), index=ids)


# --- The kernels --------------------------------------------------------------------------

@pytest.mark.parametrize("encoding", ["zz", "ry", "ry_rz"])
def test_fidelity_kernel_is_symmetric_unit_diagonal_and_psd(encoding):
    X = np.random.default_rng(1).normal(size=(40, 6))
    kern = FidelityKernel(encoding=encoding)
    E = kern.embed(X, {"gamma": 0.5})
    K = kern.gram(E, E, {})
    np.testing.assert_allclose(K, K.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert kern.qubits(6) == (3 if encoding == "ry_rz" else 6)


def test_fidelity_matches_qiskit_statevectors():
    X = np.random.default_rng(2).normal(size=(5, 4))
    kern = FidelityKernel("zz")
    circuit = kern.circuit(4)
    K = kern.gram(kern.embed(X, {"gamma": 0.8}), kern.embed(X, {"gamma": 0.8}), {})
    sv = [Statevector(circuit.assign_parameters(0.8 * x)) for x in X]
    for i in range(5):
        for j in range(5):
            assert K[i, j] == pytest.approx(abs(sv[i].inner(sv[j])) ** 2, abs=1e-12)


def test_unentangled_encoding_is_the_classical_cosine_kernel():
    # ry with one upload and no CZ is a product state: its fidelity is Π cos²(γΔx/2), a
    # classical kernel. That makes it the "no entanglement" control.
    X = np.random.default_rng(3).normal(size=(12, 5))
    kern = FidelityKernel("ry", reps=1)
    K = kern.gram(kern.embed(X, {"gamma": 0.7}), kern.embed(X, {"gamma": 0.7}), {})
    expected = np.prod(np.cos(0.7 * (X[:, None, :] - X[None, :, :]) / 2) ** 2, axis=-1)
    np.testing.assert_allclose(K, expected, atol=1e-12)


def test_quantum_gamma_grid_matches_the_rbf_grid_in_the_small_angle_limit():
    for d in (4, 10, 16):
        np.testing.assert_allclose(np.array(quantum_gamma_grid(d)) ** 2 / 4, classical.rbf_grid(d)["model__gamma"])
    from qm9dipole.models.quantum_kernel import KRR_ALPHAS
    assert list(KRR_ALPHAS) == classical.rbf_grid(10)["model__alpha"]


def test_projected_kernel_matches_reduced_density_matrices():
    X = np.random.default_rng(4).normal(size=(3, 4))
    kern = ProjectedKernel("zz")
    B = kern.embed(X, {"gamma": 0.9})
    assert B.shape == (3, 12) and (np.linalg.norm(B.reshape(3, 3, 4), axis=1) <= 1 + 1e-12).all()
    gp = 0.37
    K = kern.gram(B, B, {"gamma_p": gp})
    circuit = kern.circuit(4)
    rho = [[partial_trace(DensityMatrix(Statevector(circuit.assign_parameters(0.9 * x))), [r for r in range(4) if r != q]).data
            for q in range(4)] for x in X]
    for i in range(3):
        for j in range(3):
            dist = sum(np.linalg.norm(rho[i][q] - rho[j][q]) ** 2 for q in range(4))
            assert K[i, j] == pytest.approx(np.exp(-gp * dist), abs=1e-12)


def test_diagnostics_on_extreme_kernels():
    assert offdiag_stats(np.eye(5)) == (0.0, 0.0)
    assert offdiag_stats(np.ones((5, 5))) == (1.0, 0.0)
    y = np.array([1.0, 2.0, 4.0, 0.5])
    yc = y - y.mean()
    assert kernel_target_alignment(np.outer(yc, yc), y) == pytest.approx(1.0)
    lam = kernel_spectrum(np.eye(6))
    np.testing.assert_allclose(lam, 1 / 6)


# --- The fitter -------------------------------------------------------------------------

@pytest.mark.parametrize("cv_max_n", [1000, 50])  # 5-fold CV, and the inner validation split
def test_rbf_through_the_kernel_fitter_reproduces_tabular_fitter(frame, cv_max_n):
    """The fairness check: one tuning protocol for every kernel."""
    F, y = frame
    train, new = F.index[:120], F.index[150:]
    kwargs = dict(scaling="yeo_johnson", target="sqrt", reduction=("pls", 4))
    est, grid = classical.build("rbf_krr", seed=1, **kwargs)
    ref = TabularFitter(F, est, grid, seed=1, cv_max_n=cv_max_n, n_jobs=1).fit(train, y.loc[train])
    ours = KernelRidgeFitter(F, RBFKernel(), seed=1, cv_max_n=cv_max_n, n_jobs=1, **kwargs).fit(train, y.loc[train])
    assert ours.n_candidates_ == len(grid["regressor__model__alpha"]) * len(grid["regressor__model__gamma"])
    assert ours.alpha_ == ref.params_["regressor__model__alpha"]
    assert ours.kernel_setting_["gamma"] == pytest.approx(ref.params_["regressor__model__gamma"])
    assert ours.tuning_mae_ == pytest.approx(ref.tuning_mae_, rel=1e-8)
    np.testing.assert_allclose(ours.predict(new), ref.predict(new), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("kernel", [FidelityKernel("zz"), FidelityKernel("ry_rz"), ProjectedKernel("zz")],
                         ids=["fidelity-zz", "fidelity-ry_rz", "projected-zz"])
def test_quantum_kernel_fitter_end_to_end(frame, kernel):
    F, y = frame
    train, new = F.index[:80], F.index[150:]
    fit = KernelRidgeFitter(F, kernel, seed=0, reduction=("pca", 4), n_jobs=1).fit(train, y.loc[train])
    pred = fit.predict(new)
    assert pred.shape == (len(new),) and np.isfinite(pred).all() and (pred >= 0).all()
    mean_mae = np.abs(y.loc[train].mean() - y.loc[new]).mean()
    assert np.abs(pred - y.loc[new]).mean() < mean_mae  # it learned something
    d = fit.describe()
    json.dumps(d)
    assert d["n_qubits"] == kernel.qubits(4) and d["tuning"] == "5-fold CV"
    assert 1.0 <= d["effective_df"] <= len(train)
    assert 0.0 <= d["offdiag_mean"] <= 1.0
    np.testing.assert_array_equal(fit.predict(new), pred)  # deterministic


def _molecules(rng, n):
    out = []
    for _ in range(n):
        k = int(rng.integers(4, 10))
        Z = rng.choice([1, 1, 6, 6, 7, 8, 9], size=k)
        R = []
        while len(R) < k:
            p = rng.uniform(-2.5, 2.5, size=3)
            if all(np.linalg.norm(p - q) > 0.95 for q in R):
                R.append(p)
        out.append((Z, np.array(R)))
    return out


def _features(mols, names):
    table = pd.DataFrame({"id": np.arange(len(mols)), "Z": [m[0] for m in mols], "R": [m[1] for m in mols]})
    return pd.concat([feature_frame(table, name) for name in names], axis=1)


def test_quantum_predictions_are_invariant_and_the_negative_control_is_not():
    """End-to-end invariance: descriptors from moved molecules -> the fitted quantum model."""
    rng = np.random.default_rng(7)
    mols = _molecules(rng, 70)
    y = pd.Series(rng.uniform(0.5, 5.0, size=70))
    for names, invariant in [(MODEL_DESCRIPTORS, True), (("raw_coordinates",), False)]:
        F = _features(mols, names)
        F = F.loc[:, F.std() > 0]
        fit = KernelRidgeFitter(F, FidelityKernel("zz"), seed=0, scaling="standard", reduction=("pca", 4),
                                n_jobs=1).fit(F.index[:55], y.iloc[:55])
        test = mols[55:]
        base = fit.predict_X(F.iloc[55:].to_numpy())
        for name, tf in TRANSFORMS.items():
            moved = [tf(Z, R, np.random.default_rng([7, len(name)])) for Z, R in test]
            change = np.abs(fit.predict_X(_features(moved, names)[F.columns].to_numpy()) - base).max()
            if invariant:
                assert change < 1e-8, (name, change)
            else:
                assert change > 1e-3, (name, change)


def test_team_zz_readout_has_a_classical_closed_form():
    from qm9dipole.models.quantum import QuantumFeatureTransformer
    from qm9dipole.models.quantum_kernel import zz_readout_closed_form

    X = np.random.default_rng(8).normal(size=(20, 6))
    feats = QuantumFeatureTransformer(gamma=0.45, reps=2, simulation_method="batched").fit(X).transform(X)
    np.testing.assert_allclose(zz_readout_closed_form(0.45 * X), feats, atol=1e-12)


def test_quantum_ridge_build_runs_in_the_tabular_harness(frame):
    from qm9dipole.models.quantum_kernel import quantum_ridge_build

    F, y = frame
    est, grid = quantum_ridge_build(0, "yeo_johnson", "sqrt", ("pca", 4), F.shape[1])
    assert est.regressor[-1].clip_negative is False and est.regressor[-1].simulation_method == "batched"
    small = {k: v[::4] for k, v in grid.items()}
    fit = TabularFitter(F, est, small, seed=0, n_jobs=1).fit(F.index[:60], y.loc[:].iloc[:60])
    pred = fit.predict(F.index[150:])
    assert np.isfinite(pred).all() and (pred >= 0).all()


def test_chunked_prediction_matches_one_block(frame):
    F, y = frame
    fit = KernelRidgeFitter(F, FidelityKernel("zz"), seed=0, reduction=("pca", 3), n_jobs=1).fit(F.index[:60], y.iloc[:60])
    X = F.iloc[100:].to_numpy()
    np.testing.assert_allclose(fit.predict_X(X, chunk=7), fit.predict_X(X, chunk=1000), rtol=1e-12)


# --- Shot-aware kernels ---------------------------------------------------------------------

def test_shot_noisy_fidelity_kernel_is_a_valid_reproducible_estimate():
    X = np.random.default_rng(7).normal(size=(30, 4))
    base = FidelityKernel("zz")
    kern = ShotNoisyKernel(base, 200, seed=1)
    E = kern.embed(X, {"gamma": 0.4})
    np.testing.assert_array_equal(E, base.embed(X, {"gamma": 0.4}))  # the states themselves are exact
    exact = base.gram(E, E, {})
    K = kern.gram(E, E, {})
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    np.testing.assert_array_equal(K, kern.gram(E, E, {}))  # deterministic in (seed, inputs)
    assert not np.array_equal(K, ShotNoisyKernel(base, 200, seed=2).gram(E, E, {}))
    Kv = kern.gram(E[:7], E, {})
    np.testing.assert_allclose(Kv * 200, np.round(Kv * 200), atol=1e-9)  # counts over 200 shots
    assert 0.005 < np.abs(Kv - exact[:7]).mean() < 0.05  # about sqrt(k(1-k)/200)
    many = ShotNoisyKernel(base, 10**9, seed=1).gram(E[:7], E, {})
    np.testing.assert_allclose(many, exact[:7], atol=2e-4)


def test_shot_noisy_projected_kernel_measures_bloch_vectors():
    X = np.random.default_rng(8).normal(size=(20, 4))
    base = ProjectedKernel("zz")
    kern = ShotNoisyKernel(base, 100, seed=0)
    B = kern.embed(X, {"gamma": 0.5})
    exact = base.embed(X, {"gamma": 0.5})
    assert B.shape == exact.shape and np.all(np.abs(B) <= 1)
    np.testing.assert_allclose((B + 1) * 50, np.round((B + 1) * 50), atol=1e-9)
    assert 0.01 < np.abs(B - exact).mean() < 0.2
    np.testing.assert_allclose(kern.gram(B, B, {"gamma_p": 0.3}), base.gram(B, B, {"gamma_p": 0.3}))
    with pytest.raises(TypeError):
        ShotNoisyKernel(RBFKernel(), 100)


@pytest.mark.parametrize("make", [lambda: FidelityKernel("zz", gammas=[0.3, 1.0]),
                                  lambda: ProjectedKernel("zz", gammas=[0.3, 1.0], gammas_p=[0.1, 1.0])])
def test_shot_aware_fitter_tunes_with_noise_and_approaches_exact(frame, make):
    X, y = frame
    tr, te = X.index[:120], X.index[120:]
    exact = KernelRidgeFitter(X, make(), 0, scaling="standard", target="sqrt", reduction=("pca", 4)).fit(tr, y.loc[tr])
    huge = KernelRidgeFitter(X, ShotNoisyKernel(make(), 10**9), 0, scaling="standard", target="sqrt",
                             reduction=("pca", 4)).fit(tr, y.loc[tr])
    assert huge.describe()["params"] == exact.describe()["params"]
    np.testing.assert_allclose(huge.predict(te), exact.predict(te), atol=5e-3)
    few = KernelRidgeFitter(X, ShotNoisyKernel(make(), 50), 0, scaling="standard", target="sqrt",
                            reduction=("pca", 4)).fit(tr, y.loc[tr])
    assert few.tuning_mae_ >= exact.tuning_mae_ - 0.05 and np.isfinite(few.predict(te)).all()
    assert few.config["kernel"]["shots"] == 50 and "shots" not in exact.config["kernel"]
