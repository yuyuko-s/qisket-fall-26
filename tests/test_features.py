"""Engineered-feature tests (features.py).

Rotation, translation and relabeling invariance of the full engineered vector is tested in
tests/test_invariance.py, because "engineered" is in MODEL_DESCRIPTORS. These tests check
the definitions on molecules whose answers are known by hand.
"""

import numpy as np
import pytest

from qm9dipole.data import QM9_PARQUET
from qm9dipole.features import (
    BOND_TYPES, ELECTRONEGATIVITY, ENGINEERED_NAMES, bond_counts, bond_dipoles, bonds,
    electronegativity_dipole, engineered, geometry, heteroatom_offset, valence_ok,
)

HOH_ANGLE = np.deg2rad(104.5)
WATER_Z = np.array([8, 1, 1])
WATER_R = np.array([[0.0, 0.0, 0.0],
                    [0.958 * np.sin(HOH_ANGLE / 2), 0.958 * np.cos(HOH_ANGLE / 2), 0.0],
                    [-0.958 * np.sin(HOH_ANGLE / 2), 0.958 * np.cos(HOH_ANGLE / 2), 0.0]])
CO2_Z = np.array([6, 8, 8])
CO2_R = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
HF_Z = np.array([1, 9])
HF_R = np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]])


def methane(bond: float = 1.09) -> tuple[np.ndarray, np.ndarray]:
    """Ideal tetrahedral CH4, carbon at the origin."""
    h = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3) * bond
    return np.array([6, 1, 1, 1, 1]), np.vstack([np.zeros(3), h])


def named_counts(Z, R):
    counts = bond_counts(Z, bonds(Z, R))
    return {t: c for t, c in zip(BOND_TYPES, counts) if c}


# --- Bonds -----------------------------------------------------------------------------

def test_bond_types_cover_every_element_pair_once():
    assert len(BOND_TYPES) == 15 and len(set(BOND_TYPES)) == 15
    assert BOND_TYPES[:5] == ("C-C", "C-H", "C-N", "C-O", "C-F")


def test_bonds_of_small_molecules():
    assert named_counts(WATER_Z, WATER_R) == {"H-O": 2}
    assert named_counts(*methane()) == {"C-H": 4}  # H···H at 1.78 Å is not a bond
    assert named_counts(CO2_Z, CO2_R) == {"C-O": 2}  # O···O at 2.32 Å is not a bond


def test_valence_check():
    assert valence_ok(WATER_Z, bonds(WATER_Z, WATER_R))
    assert not valence_ok(WATER_Z, np.array([[0, 1]]))  # second H left unbonded
    assert not valence_ok(np.array([1, 1, 1]), np.array([[0, 1], [0, 2]]))  # H with 2 bonds


def test_bond_counts_do_not_depend_on_atom_order():
    Z, R = methane()
    p = np.array([3, 0, 4, 1, 2])
    np.testing.assert_array_equal(bond_counts(Z[p], bonds(Z[p], R[p])), bond_counts(Z, bonds(Z, R)))


# --- Bond dipoles ----------------------------------------------------------------------

def test_bond_dipoles_cancel_in_symmetric_molecules():
    dchi_co = ELECTRONEGATIVITY[8] - ELECTRONEGATIVITY[6]
    total, vector = bond_dipoles(CO2_Z, CO2_R, bonds(CO2_Z, CO2_R))
    assert total == pytest.approx(2 * dchi_co) and vector == pytest.approx(0.0, abs=1e-12)
    Z, R = methane()
    total, vector = bond_dipoles(Z, R, bonds(Z, R))
    assert total == pytest.approx(4 * (ELECTRONEGATIVITY[6] - ELECTRONEGATIVITY[1]))
    assert vector == pytest.approx(0.0, abs=1e-12)


def test_bond_dipole_of_water_by_hand():
    # Two O–H bond dipoles of strength Δχ at the H–O–H angle θ add to 2·Δχ·cos(θ/2).
    dchi = ELECTRONEGATIVITY[8] - ELECTRONEGATIVITY[1]
    total, vector = bond_dipoles(WATER_Z, WATER_R, bonds(WATER_Z, WATER_R))
    assert total == pytest.approx(2 * dchi)
    assert vector == pytest.approx(2 * dchi * np.cos(HOH_ANGLE / 2))


def test_no_bonds_gives_zero_polarity():
    Z, R = np.array([1, 1]), np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert bond_dipoles(Z, R, bonds(Z, R)) == (0.0, 0.0)


# --- Geometry and heteroatom offset ----------------------------------------------------

def test_geometry_of_a_linear_molecule():
    rg, m1, m2, m3, d_max = geometry(CO2_R)
    assert rg == pytest.approx(np.sqrt(2 * 1.16**2 / 3))
    assert m1 == pytest.approx(rg**2)  # all the spread lies along the molecular axis
    assert m2 == pytest.approx(0.0, abs=1e-12) and m3 == pytest.approx(0.0, abs=1e-12)
    assert d_max == pytest.approx(2.32)


def test_geometry_of_a_planar_molecule_is_rotation_stable():
    # A 3-atom molecule is planar: its third moment is exactly 0, and must stay ~0 (not
    # ~1e-8, which a square root of floating-point noise would give) after any rotation.
    from scipy.spatial.transform import Rotation

    R = WATER_R @ Rotation.random(random_state=3).as_matrix().T
    np.testing.assert_allclose(geometry(R), geometry(WATER_R), atol=1e-12)


def test_heteroatom_offset():
    assert heteroatom_offset(*methane()) == 0.0  # no N, O or F
    assert heteroatom_offset(HF_Z, HF_R) == pytest.approx(0.46)  # F at 0.92, centroid at 0.46
    assert heteroatom_offset(CO2_Z, CO2_R) == pytest.approx(0.0, abs=1e-12)  # O centroid = C


# --- Electronegativity point-charge dipole ---------------------------------------------

def test_en_dipole_vanishes_for_symmetric_molecules():
    assert electronegativity_dipole(CO2_Z, CO2_R) == pytest.approx(0.0, abs=1e-12)
    assert electronegativity_dipole(*methane()) == pytest.approx(0.0, abs=1e-12)


def test_en_dipole_is_positive_for_polar_molecules():
    assert electronegativity_dipole(WATER_Z, WATER_R) > 0
    assert electronegativity_dipole(HF_Z, HF_R) > 0


def test_en_dipole_scales_with_bond_length():
    # A dipole is charge × distance: stretching HF twofold must double it.
    assert electronegativity_dipole(HF_Z, 2 * HF_R) == pytest.approx(2 * electronegativity_dipole(HF_Z, HF_R))


def test_en_dipole_does_not_depend_on_the_origin():
    # Fails unless the pseudo-charges sum to zero: the dipole of a net charge depends on the
    # origin it is measured from.
    shift = np.array([5.0, -3.0, 2.0])
    assert electronegativity_dipole(WATER_Z, WATER_R + shift) == pytest.approx(
        electronegativity_dipole(WATER_Z, WATER_R))


# --- Bond orders, rings, inertia -------------------------------------------------------

def named_orders(Z, R):
    from qm9dipole.features import BOND_ORDER_RULES, bond_order_counts

    counts = bond_order_counts(Z, R, bonds(Z, R))
    return {name: c for (name, *_), c in zip(BOND_ORDER_RULES, counts) if c}


def test_bond_orders_from_lengths():
    assert named_orders(CO2_Z, CO2_R) == {"double_C-O": 2}  # C=O at 1.16 Å
    hcn_Z, hcn_R = np.array([1, 6, 7]), np.array([[-1.07, 0, 0], [0.0, 0, 0], [1.16, 0, 0]])
    assert named_orders(hcn_Z, hcn_R) == {"triple_C-N": 1}  # nitrile
    methanol_CO = np.array([[0.0, 0, 0], [1.43, 0, 0]])
    assert named_orders(np.array([6, 8]), methanol_CO) == {}  # a single C–O bond


def test_ring_count():
    from qm9dipole.features import n_rings

    triangle = np.array([[0.0, 0, 0], [1.5, 0, 0], [0.75, 1.3, 0]])  # cyclopropane skeleton
    assert n_rings(3, bonds(np.array([6, 6, 6]), triangle)) == 1
    assert n_rings(5, bonds(*methane())) == 0


def test_inertia_moments_of_co2():
    from qm9dipole.features import ISOTOPE_MASS, inertia_moments

    I = inertia_moments(CO2_Z, CO2_R)
    perpendicular = 2 * ISOTOPE_MASS[8] * 1.16**2
    np.testing.assert_allclose(I, [0.0, perpendicular, perpendicular], atol=1e-9)


def test_exploration_only_features_are_not_descriptors():
    from qm9dipole.descriptors import DESCRIPTORS
    from qm9dipole.features import charge_features, frequency_features

    q = np.array([-0.8, 0.4, 0.4])
    out = charge_features(q, WATER_R)
    expected = np.linalg.norm(q @ WATER_R) * 4.80320  # e·Å → D
    np.testing.assert_allclose(out, [expected, 0.8, 0.8])
    np.testing.assert_allclose(frequency_features(np.array([400.0, 1600.0, 3700.0])),
                               [400.0, 3700.0, 1900.0, 1.0, 1.0])
    assert not {"charge_features", "frequency_features"} & set(DESCRIPTORS)


# --- Full vector -----------------------------------------------------------------------

def test_engineered_vector_matches_names():
    x = engineered(WATER_Z, WATER_R)
    assert x.shape == (len(ENGINEERED_NAMES),) and np.isfinite(x).all()
    named = dict(zip(ENGINEERED_NAMES, x))
    assert named["bonds_H-O"] == 2 and named["bonds_C-H"] == 0


@pytest.mark.skipif(not QM9_PARQUET.exists(), reason="data/processed/qm9.parquet not built")
def test_inferred_bonds_respect_valence_on_real_molecules():
    from qm9dipole.data import load_qm9_table

    max_valence = {1: 1, 6: 4, 7: 4, 8: 2, 9: 1}
    for Z, R in load_qm9_table().sample(2000, random_state=0)[["Z", "R"]].itertuples(index=False):
        degree = np.bincount(bonds(Z, R).ravel(), minlength=len(Z))
        assert (degree[Z == 1] == 1).all(), "every hydrogen has exactly one bond"
        assert all(1 <= d <= max_valence[int(z)] for z, d in zip(Z, degree))
