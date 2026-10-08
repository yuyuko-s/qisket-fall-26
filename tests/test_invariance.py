"""Descriptor and invariance tests.

Model descriptors must not change under rotation, translation or atom relabeling; the
raw-coordinate negative control must change under all three, proving the tests can fail.
"""

import numpy as np
import pandas as pd
import pytest

from qm9dipole.data import QM9_PARQUET
from qm9dipole.descriptors import (
    ANGSTROM_TO_BOHR, MAX_ATOMS, MODEL_DESCRIPTORS, cm_spectrum, composition, coulomb_matrix,
    featurize, raw_coordinates,
)
from qm9dipole.invariance import ATOL, TRANSFORMS, invariance_table, max_change

METHANE_Z = np.array([6, 1, 1, 1, 1])
METHANE_R = np.array([
    [-0.0126981359, 1.0858041578, 0.0080009958],
    [0.002150416, -0.0060313176, 0.0019761204],
    [1.0117308433, 1.4637511618, 0.0002765748],
    [-0.540815069, 1.4475266138, -0.8766437152],
    [-0.5238136345, 1.4379326443, 0.9063972942],
])


def random_molecule(rng, n_atoms):
    """Atoms of QM9 elements placed one by one, at least 0.9 Å apart."""
    Z = rng.choice([1, 6, 7, 8, 9], size=n_atoms)
    side = 2.0 * n_atoms ** (1 / 3)
    R = []
    while len(R) < n_atoms:
        p = rng.uniform(-side, side, size=3)
        if all(np.linalg.norm(p - q) > 0.9 for q in R):
            R.append(p)
    return Z, np.array(R)


@pytest.fixture(scope="module")
def synthetic_molecules():
    rng = np.random.default_rng(0)
    return [random_molecule(rng, int(rng.integers(3, MAX_ATOMS + 1))) for _ in range(100)]


# --- Descriptor definitions ----------------------------------------------------------

def test_composition_counts_c_h_n_o_f():
    assert composition(METHANE_Z).tolist() == [1, 4, 0, 0, 0]
    assert composition(np.array([8, 1, 1, 7, 9, 9])).tolist() == [0, 2, 1, 1, 2]


def test_coulomb_matrix_of_h2_by_hand():
    Z, R = np.array([1, 1]), np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    off = 1.0 / (0.74 * ANGSTROM_TO_BOHR)  # Z_i Z_j / distance in bohr
    np.testing.assert_allclose(coulomb_matrix(Z, R), [[0.5, off], [off, 0.5]])
    spectrum = cm_spectrum(Z, R)
    np.testing.assert_allclose(spectrum[:2], [0.5 + off, 0.5 - off])  # sorted by |λ|
    assert spectrum.shape == (MAX_ATOMS,) and not spectrum[2:].any()  # zero-padded


def test_coulomb_diagonal_is_half_z_to_the_2_4():
    assert coulomb_matrix(METHANE_Z, METHANE_R)[0, 0] == pytest.approx(0.5 * 6**2.4)


def test_shapes_and_featurize():
    table = pd.DataFrame({"Z": [METHANE_Z, METHANE_Z[:3]], "R": [METHANE_R, METHANE_R[:3]]})
    assert featurize(table, "composition").shape == (2, 5)
    assert featurize(table, "cm_spectrum").shape == (2, MAX_ATOMS)
    assert featurize(table, "raw_coordinates").shape == (2, 3 * MAX_ATOMS)
    assert raw_coordinates(METHANE_Z, METHANE_R)[:3].tolist() == METHANE_R[0].tolist()


# --- Invariance, synthetic molecules (no data needed) --------------------------------

@pytest.mark.parametrize("transform", list(TRANSFORMS))
@pytest.mark.parametrize("descriptor", MODEL_DESCRIPTORS)
def test_model_descriptors_are_invariant(descriptor, transform, synthetic_molecules):
    change = max_change(descriptor, transform, synthetic_molecules, np.random.default_rng(1))
    assert change.max() <= ATOL, f"{descriptor} changed by {change.max():.2e} under {transform}"


@pytest.mark.parametrize("transform", list(TRANSFORMS))
def test_negative_control_fails_every_check(transform, synthetic_molecules):
    # If this passed, the invariance tests above could not detect a broken descriptor.
    change = max_change("raw_coordinates", transform, synthetic_molecules, np.random.default_rng(1))
    assert change.max() > 1e-3


def test_invariance_table_pattern(synthetic_molecules):
    table = invariance_table(synthetic_molecules, seed=0)
    expected = {d: d in MODEL_DESCRIPTORS for d in table["descriptor"]}
    assert (table["invariant"] == table["descriptor"].map(expected)).all()


# --- Invariance, real QM9 molecules ----------------------------------------------------

@pytest.mark.skipif(not QM9_PARQUET.exists(), reason="data/processed/qm9.parquet not built")
def test_invariance_on_100_real_molecules():
    from qm9dipole.data import load_qm9_table

    table = load_qm9_table().sample(100, random_state=0)
    molecules = list(zip(table["Z"], table["R"]))
    result = invariance_table(molecules, seed=0)
    for row in result.itertuples():
        assert row.invariant == (row.descriptor in MODEL_DESCRIPTORS), row
