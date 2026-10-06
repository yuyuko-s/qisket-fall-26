"""Exploration-chain plumbing tests (explore.py)."""

import pandas as pd

from qm9dipole.explore import EXPLORATION_ONLY_SETS, feature_sets


def test_feature_sets_respect_blocks_and_legality():
    catalog = pd.DataFrame([
        ("n_C", "composition", True, True), ("cm_01", "cm", True, True),
        ("bond_dipole_norm", "engineered", True, True), ("bonds_F-F", "engineered", True, False),
        ("alpha", "dft_property", False, True), ("charge_dipole_D", "charge", False, True),
        ("freq_lowest", "frequency", False, True),
    ], columns=["feature", "block", "legal", "kept"])
    sets = feature_sets(catalog)
    assert sets["structure"] == ["n_C", "bond_dipole_norm"]  # dropped features never return
    assert sets["cm+structure"] == ["cm_01", "n_C", "bond_dipole_norm"]
    legal = dict(zip(catalog["feature"], catalog["legal"]))
    for name, cols in sets.items():
        uses_dft = not all(legal[c] for c in cols)
        assert uses_dft == (name in EXPLORATION_ONLY_SETS), name
