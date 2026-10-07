"""Per-atom environment tests (atoms.py): definitions on molecules with known answers, and
the symmetry every per-atom row must have (invariant to rotation and translation, permuted
with the atoms). Molecule-level blocks are covered by tests/test_invariance.py through
descriptors.MODEL_DESCRIPTORS.
"""

import numpy as np
import pandas as pd
import pytest

from qm9dipole.atoms import (
    ATOM_FEATURE_NAMES, GROUP_NAMES, QEQ_NAMES, RDF_NAMES, build_atom_data, describe, qeq_charges,
)
from qm9dipole.invariance import permute, rotate, translate

from test_invariance import random_molecule

F = {name: k for k, name in enumerate(ATOM_FEATURE_NAMES)}
G = {name: k for k, name in enumerate(GROUP_NAMES)}


def tetrahedral(center, bond, n=4):
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)[:n] / np.sqrt(3)
    return center + bond * v


def ring(n_atoms, bond):
    """Heavy-atom ring (no hydrogens), regular polygon of side `bond` Å."""
    radius = bond / (2 * np.sin(np.pi / n_atoms))
    a = 2 * np.pi * np.arange(n_atoms) / n_atoms
    return np.full(n_atoms, 6), np.column_stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n_atoms)])


WATER = (np.array([8, 1, 1]), np.array([[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]]))
AMMONIUM = (np.array([7, 1, 1, 1, 1]), np.vstack([np.zeros(3), tetrahedral(np.zeros(3), 1.03)]))
# Ammonia: N 0.38 Å above the plane of its three H atoms.
AMMONIA = (np.array([7, 1, 1, 1]), np.array([[0, 0, 0.38], [0.94, 0, 0], [-0.47, 0.814, 0], [-0.47, -0.814, 0]]))
CO2 = (np.array([6, 8, 8]), np.array([[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]))


def glycine_zwitterion():
    """⁺H₃N–CH₂–COO⁻, idealized: ammonium N, sp3 C, carboxylate C with two O at 1.26 Å."""
    N, Ca, Cc = np.array([0.0, 0, 0]), np.array([1.49, 0, 0]), np.array([2.0, 1.43, 0])
    O1, O2 = Cc + [1.20, 0.35, 0], Cc + [-0.40, 1.19, 0]
    H_N = N + 1.03 * np.array([[-0.33, 0.94, 0], [-0.33, -0.47, 0.82], [-0.33, -0.47, -0.82]])
    H_C = Ca + 1.09 * np.array([[0.33, -0.47, 0.82], [0.33, -0.47, -0.82]])
    Z = np.array([7, 6, 6, 8, 8, 1, 1, 1, 1, 1])
    return Z, np.vstack([N, Ca, Cc, O1, O2, H_N, H_C])


def test_feature_names_match_columns():
    env = describe(*WATER)
    assert env.features.shape == (3, len(ATOM_FEATURE_NAMES))
    assert len(set(ATOM_FEATURE_NAMES)) == len(ATOM_FEATURE_NAMES)
    assert env.groups.shape == (len(GROUP_NAMES),) and env.qeq.shape == (len(QEQ_NAMES),)
    assert env.rdf.shape == (len(RDF_NAMES),)


def test_water_graph_and_charges():
    env = describe(*WATER)
    x = env.features
    assert x[:, F["is_O"]].tolist() == [1, 0, 0] and x[:, F["is_H"]].tolist() == [0, 1, 1]
    assert x[:, F["degree"]].tolist() == [2, 1, 1]
    assert x[0, F["bonded_H"]] == 2 and x[1, F["bonded_O"]] == 1 and x[1, F["shell2_H"]] == 1
    q = env.qeq_charges
    assert q.sum() == pytest.approx(0, abs=1e-12)
    assert q[0] < 0 < q[1] and q[1] == pytest.approx(q[2])  # O pulls charge; symmetric H
    assert env.qeq[0] > 0.5  # a clearly polar molecule
    assert (x[:, F["smallest_ring"]] == 0).all()


def test_symmetric_molecule_has_zero_qeq_dipole():
    assert describe(*CO2).qeq[0] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_ring_sizes(n):
    Z, R = ring(n, 1.45)
    env = describe(Z, R)
    assert (env.features[:, F["smallest_ring"]] == n).all()
    key = f"ring{n}_atoms" if n < 7 else "ring7plus_atoms"
    assert env.groups[G[key]] == n


def test_chain_has_no_ring():
    Z = np.array([6, 6, 6])
    R = np.array([[0, 0, 0], [1.52, 0, 0], [2.03, 1.43, 0]])
    assert (describe(Z, R).features[:, F["smallest_ring"]] == 0).all()


def test_ammonium_and_amine_geometry():
    nh4 = describe(*AMMONIUM)
    assert nh4.features[0, F["ammonium_N"]] == 1 and nh4.groups[G["n_ammonium"]] == 1
    nh3 = describe(*AMMONIA)
    assert nh3.features[0, F["ammonium_N"]] == 0
    assert nh3.features[0, F["pyramid_height"]] == pytest.approx(0.38, abs=1e-9)
    assert nh3.groups[G["n_N3_pyramidal"]] == 1 and nh3.groups[G["n_N3_planar"]] == 0
    flat = (AMMONIA[0], AMMONIA[1] * np.array([1, 1, 0]))  # N pushed into the H plane
    assert describe(*flat).groups[G["n_N3_planar"]] == 1


def test_glycine_zwitterion_is_recognized():
    env = describe(*glycine_zwitterion())
    x = env.features
    assert x[2, F["terminal_O_neighbours"]] == 2  # the carboxylate carbon
    assert env.groups[G["n_carboxylate"]] == 1 and env.groups[G["n_ammonium"]] == 1
    assert env.groups[G["zwitterion"]] == 1
    assert env.qeq[0] > describe(*WATER).qeq[0]  # charges far apart: a large dipole


def test_qeq_charges_are_neutral_on_random_molecules():
    rng = np.random.default_rng(0)
    for _ in range(20):
        Z, R = random_molecule(rng, int(rng.integers(3, 29)))
        env = describe(Z, R)
        assert env.qeq_charges.sum() == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("transform", [rotate, translate])
def test_per_atom_rows_are_invariant(transform):
    rng = np.random.default_rng(1)
    for _ in range(30):
        Z, R = random_molecule(rng, int(rng.integers(3, 29)))
        a = describe(Z, R).features
        b = describe(*transform(Z, R, rng)).features
        assert np.abs(a - b).max() <= 1e-8


def test_per_atom_rows_permute_with_the_atoms():
    rng = np.random.default_rng(2)
    for _ in range(30):
        Z, R = random_molecule(rng, int(rng.integers(3, 29)))
        p = rng.permutation(len(Z))
        assert np.abs(describe(Z[p], R[p]).features - describe(Z, R).features[p]).max() <= 1e-8


def test_build_atom_data_layout():
    mols = [WATER, AMMONIA, glycine_zwitterion()]
    table = pd.DataFrame({"id": [10, 20, 30], "Z": [m[0] for m in mols], "R": [m[1] for m in mols]})
    data, blocks = build_atom_data(table, n_jobs=1)
    assert data.n_atoms.tolist() == [3, 4, 10] and list(blocks.index) == [10, 20, 30]
    assert data.X.shape == (17, len(ATOM_FEATURE_NAMES)) and data.X.dtype == np.float32
    for k in range(3):
        e = data.edges[data.edge_offsets[k]:data.edge_offsets[k + 1]]
        assert ((e >= data.offsets[k]) & (e < data.offsets[k + 1])).all()  # edges stay in their molecule
    np.testing.assert_allclose(np.linalg.norm(data.edge_vectors, axis=1), 1, atol=1e-6)
    assert len(data.edges) == 2 * (2 + 3 + 9)  # directed bonds: water 2, ammonia 3, glycine 9
    np.testing.assert_allclose(blocks.loc[30, list(QEQ_NAMES)], describe(*glycine_zwitterion()).qeq, rtol=1e-12)


def test_qeq_charges_function_matches_describe():
    Z, R = glycine_zwitterion()
    D = np.linalg.norm(R[:, None] - R[None], axis=-1)
    from qm9dipole.atoms import _INDEX
    np.testing.assert_allclose(qeq_charges(_INDEX[Z], D), describe(Z, R).qeq_charges)
