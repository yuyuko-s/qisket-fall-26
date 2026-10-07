"""Latent-charge model tests (models/charge.py): gradients, symmetry, and that it can learn.

The gradients are hand-written, so they are checked against central finite differences in
float64 for every parameter tensor. Invariance is checked end to end: the same fitted model
must give the same |μ| for a rotated, translated or relabeled molecule.
"""

import numpy as np
import pandas as pd
import pytest

from qm9dipole.atoms import build_atom_data
from qm9dipole.invariance import permute, rotate, translate
from qm9dipole.models.charge import ChargeEnsemble, LatentChargeModel

from test_invariance import random_molecule


def _table(molecules):
    return pd.DataFrame({"id": np.arange(1, len(molecules) + 1), "Z": [m[0] for m in molecules],
                         "R": [m[1] for m in molecules]})


def _organic_like(rng, n):
    """A chain of heavy atoms ~1.45 Å apart with H atoms ~1.05 Å off each: bonded, connected,
    and far from the degenerate geometries of fully random placements."""
    heavy = rng.choice([6, 6, 6, 7, 8], size=n)
    R, Z = [np.zeros(3)], [heavy[0]]
    for z in heavy[1:]:
        while True:
            step = rng.normal(size=3)
            p = R[-1] + 1.45 * step / np.linalg.norm(step)
            if all(np.linalg.norm(p - q) > 2.0 for q in R[:-1]):
                break
        R.append(p)
        Z.append(z)
    for k in range(n):
        for _ in range(rng.integers(1, 3)):
            for _ in range(50):
                d = rng.normal(size=3)
                h = R[k] + 1.05 * d / np.linalg.norm(d)
                if all(np.linalg.norm(h - q) > 1.6 for i, q in enumerate(R) if i != k):
                    R.append(h)
                    Z.append(1)
                    break
    return np.array(Z), np.array(R)


@pytest.fixture(scope="module")
def molecules():
    rng = np.random.default_rng(0)
    return [_organic_like(rng, int(rng.integers(2, 8))) for _ in range(60)]


@pytest.fixture(scope="module")
def data(molecules):
    return build_atom_data(_table(molecules), n_jobs=1)[0]


@pytest.mark.parametrize("hidden, mp, dipoles", [((), 0, False), ((), 0, True), ((7, 5), 0, False),
                                                  ((7, 5), 0, True), ((6,), 1, True), ((7, 5), 2, True)])
@pytest.mark.parametrize("loss", ["mse", "huber"])
def test_gradients_match_finite_differences(data, hidden, mp, dipoles, loss):
    rng = np.random.default_rng(1)
    m = LatentChargeModel(hidden=hidden, message_passing=mp, atomic_dipoles=dipoles, loss=loss, dtype=np.float64,
                          huber_delta=0.5)
    m.mean_, m.scale_ = data.X.mean(0), np.where(data.X.std(0) > 1e-6, data.X.std(0), 1.0)
    Xs = m._standardize(data.X)
    p = m._init_params(data.X.shape[1], rng)
    p = {k: v + rng.normal(0, 0.2, v.shape) for k, v in p.items()}  # away from the tiny initial output layer
    k = np.arange(12)
    b = m._batch(data, Xs, k)
    y = rng.uniform(0.5, 4.0, len(k))

    def loss_of(params):
        return m._loss_grad(m._forward(params, b)[0], y)[0]

    pred, cache = m._forward(p, b)
    _, dpred = m._loss_grad(pred, y)
    grads = m._backward(p, b, cache, dpred)
    assert set(grads) == set(p)
    h = 1e-6
    for name, value in p.items():
        for idx in rng.choice(value.size, size=min(6, value.size), replace=False):
            up, down = {k: v.copy() for k, v in p.items()}, {k: v.copy() for k, v in p.items()}
            up[name].flat[idx] += h
            down[name].flat[idx] -= h
            numeric = (loss_of(up) - loss_of(down)) / (2 * h)
            assert grads[name].flat[idx] == pytest.approx(numeric, rel=1e-4, abs=1e-7), (name, idx)


@pytest.mark.parametrize("transform", [rotate, translate, permute])
def test_predictions_are_invariant(molecules, transform):
    rng = np.random.default_rng(2)
    table = _table(molecules)
    y = rng.uniform(0.5, 4.0, len(table))
    base_data = build_atom_data(table, n_jobs=1, dtype=np.float64)[0]
    model = LatentChargeModel(hidden=(8,), message_passing=2, atomic_dipoles=True, max_epochs=3, min_steps=20,
                              dtype=np.float64).fit(base_data, y)
    moved = _table([transform(z, r, rng) for z, r in molecules])
    np.testing.assert_allclose(model.predict(build_atom_data(moved, n_jobs=1, dtype=np.float64)[0]), model.predict(base_data),
                               rtol=0, atol=1e-9)


def test_negative_control_raw_frame_dipole_is_not_rotation_invariant(data):
    # Sanity check that the test above can fail: a "dipole" built from uncentred raw
    # coordinates of each atom (no neutrality, no centring) changes under translation.
    Z, R = random_molecule(np.random.default_rng(3), 6)
    q = np.where(Z == 1, 0.2, -0.1)
    moved = translate(Z, R, np.random.default_rng(4))[1]
    assert abs(np.linalg.norm(q @ R) - np.linalg.norm(q @ moved)) > 1e-3


def test_learns_known_point_charges(molecules):
    # Ground truth: fixed charges per element, made neutral; |μ| = 4.803 |Σ q c| in debye.
    # The linear charge model sees the element one-hot, so it can represent this exactly.
    charge = {1: 0.15, 6: -0.05, 7: -0.35, 8: -0.45}
    rng = np.random.default_rng(5)
    mols = molecules + [_organic_like(rng, int(rng.integers(2, 8))) for _ in range(240)]
    table = _table(mols)
    y = []
    for Z, R in mols:
        q = np.array([charge[int(z)] for z in Z])
        q -= q.mean()
        y.append(4.80320 * np.linalg.norm(q @ (R - R.mean(0))))
    y = np.array(y)
    data = build_atom_data(table, n_jobs=1)[0]
    train, test = np.arange(240), np.arange(240, len(mols))
    # Patient early stopping matters: at lr 1e-2 with patience 60 it stalls on a plateau
    # at ~10% of the baseline error; these settings reach ~1.4%.
    model = LatentChargeModel(hidden=(), atomic_dipoles=False, loss="mse", lr=3e-3, weight_decay=0.0,
                              max_epochs=1500, patience=200, lr_patience=50, seed=0)
    model.fit(data.subset(table["id"].to_numpy()[train]), y[train])
    pred = model.predict(data.subset(table["id"].to_numpy()[test]))
    assert np.abs(pred - y[test]).mean() < 0.03 * np.abs(y[test] - y[test].mean()).mean()


def test_subset_keeps_molecules_intact(data):
    ids = data.ids[[5, 0, 17]]
    sub = data.subset(ids)
    assert sub.n_molecules == 3 and np.array_equal(sub.ids, ids)
    for k, i in enumerate(ids):
        r = data.rows([i])[0]
        np.testing.assert_array_equal(sub.X[sub.offsets[k]:sub.offsets[k + 1]], data.X[data.offsets[r]:data.offsets[r + 1]])
        e_sub = sub.edges[sub.edge_offsets[k]:sub.edge_offsets[k + 1]] - sub.offsets[k]
        e_full = data.edges[data.edge_offsets[r]:data.edge_offsets[r + 1]] - data.offsets[r]
        np.testing.assert_array_equal(e_sub, e_full)
    assert (np.bincount(np.repeat(np.arange(3), sub.n_atoms), minlength=3) == sub.n_atoms).all()


def test_centred_positions_sum_to_zero(data):
    sums = np.add.reduceat(data.pos.astype(np.float64), data.offsets[:-1], axis=0)
    np.testing.assert_allclose(sums, 0, atol=1e-5)


def test_charges_sum_to_zero_and_reproduce_the_dipole(data):
    rng = np.random.default_rng(6)
    y = rng.uniform(0.5, 4.0, data.n_molecules)
    model = LatentChargeModel(hidden=(8,), atomic_dipoles=False, max_epochs=2, min_steps=10, dtype=np.float64).fit(data, y)
    q = model.charges(data)
    sums = np.add.reduceat(q, data.offsets[:-1])
    np.testing.assert_allclose(sums, 0, atol=1e-10)
    vec = np.add.reduceat(q[:, None] * data.pos, data.offsets[:-1], axis=0) * 4.80320
    np.testing.assert_allclose(np.linalg.norm(vec, axis=1), model.predict(data), rtol=1e-6, atol=1e-5)


def test_ensemble_averages_magnitudes(data):
    y = np.random.default_rng(7).uniform(0.5, 4.0, data.n_molecules)
    ens = ChargeEnsemble(n_models=2, hidden=(4,), max_epochs=2, min_steps=10).fit(data, y)
    expected = np.mean([m.predict(data) for m in ens.members_], axis=0)
    np.testing.assert_allclose(ens.predict(data), expected)
