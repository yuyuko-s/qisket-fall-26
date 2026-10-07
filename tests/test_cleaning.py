"""Cleaning-rule tests (cleaning.py)."""

import numpy as np
import pandas as pd
import pytest

from qm9dipole.cleaning import (
    clean_frequencies, duplicate_groups, is_linear, moments_from_rotational_constants,
    redundant_columns, vibrational_modes, zero_variance,
)
from qm9dipole.data import QM9_PARQUET
from qm9dipole.features import inertia_moments


def test_doubled_frequency_list_keeps_first_copy():
    f = np.arange(1.0, 10.0)  # 9 modes: methane (n = 5, 3n − 6)
    np.testing.assert_array_equal(clean_frequencies(np.concatenate([f, f + 0.01]), 5), f)
    np.testing.assert_array_equal(clean_frequencies(f, 5), f)  # already clean
    assert vibrational_modes(5, linear=False) == 9 and vibrational_modes(3, linear=True) == 4


def test_linear_molecules_are_recognized_by_mode_count():
    assert is_linear(np.ones(4), n_atoms=3)       # CO2: 3·3 − 5
    assert not is_linear(np.ones(3), n_atoms=3)   # H2O: 3·3 − 6


def test_moments_from_rotational_constants_handle_the_linear_sentinel():
    I = moments_from_rotational_constants(np.array([0.0, 10.0, 505.379009]))
    np.testing.assert_allclose(I, [0.0, 50.5379009, 1.0])


def test_duplicate_groups():
    spectra = np.array([[1.000, 2.0], [1.001, 2.0], [1.2, 2.0], [1.0, 2.0]])
    groups = duplicate_groups(np.array([10, 11, 12, 13]), ["CH4", "CH4", "CH4", "H2O"], spectra)
    assert groups.to_dict() == {10: 10, 11: 10}  # 13 has the same spectrum but another formula


def test_zero_variance_and_redundancy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    frame = pd.DataFrame({"a": a, "a_cubed": a**3, "b": rng.normal(size=200), "const": 1.0})
    assert zero_variance(frame) == ["const"]
    red = redundant_columns(frame.drop(columns="const"), threshold=0.99)
    assert red[["dropped", "kept"]].values.tolist() == [["a_cubed", "a"]]  # monotone: ρ = 1
    red = redundant_columns(frame.drop(columns="const"), 0.99, priority=["a_cubed", "a", "b"])
    assert red["dropped"].tolist() == ["a"]  # priority decides which survives


def test_redundancy_is_not_fooled_by_shared_zeros():
    # Two sparse columns: zero on the same 95% of rows, unrelated where they are non-zero.
    # Ties at zero make ρ over all rows ≥ 0.99; on the active rows they are independent.
    rng = np.random.default_rng(1)
    active = rng.random(4000) < 0.05
    u = np.where(active, rng.uniform(1, 2, 4000), 0.0)
    v = np.where(active, rng.uniform(1, 2, 4000), 0.0)
    frame = pd.DataFrame({"u": u, "v": v, "u_twice": 2 * u})
    assert frame.corr(method="spearman").loc["u", "v"] >= 0.99
    red = redundant_columns(frame, 0.99)
    assert red["dropped"].tolist() == ["u_twice"]  # the true duplicate still goes; v stays
    assert red["dropped"].tolist() != redundant_columns(frame, 0.99, active_only=False)["dropped"].tolist()


def test_identical_indicators_are_redundant_but_an_indicator_of_a_count_is_not():
    rng = np.random.default_rng(2)
    flag = (rng.random(3000) < 0.05).astype(float)
    count = flag * rng.integers(1, 4, 3000)  # non-zero exactly where flag is, but varies there
    frame = pd.DataFrame({"flag": flag, "same_flag": flag.copy(), "count": count})
    red = redundant_columns(frame, 0.99)
    assert red["dropped"].tolist() == ["same_flag"]


@pytest.mark.skipif(not QM9_PARQUET.exists(), reason="data/processed/qm9.parquet not built")
def test_qm9_rotational_constants_are_functions_of_the_geometry():
    # A, B, C (GHz) from QM9 equal 505.379/I for the moments computed from Z and R alone.
    from qm9dipole.data import load_qm9_table

    sample = load_qm9_table().sample(300, random_state=1)
    for row in sample.itertuples():
        from_geometry = inertia_moments(row.Z, row.R)
        from_qm9 = np.sort(moments_from_rotational_constants(np.array([row.A, row.B, row.C])))
        np.testing.assert_allclose(from_geometry, from_qm9, rtol=1e-4, atol=1e-6)
