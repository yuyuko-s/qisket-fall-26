"""Exploration-chain plumbing tests (explore.py)."""

import numpy as np
import pandas as pd

from qm9dipole.explore import EXPLORATION_ONLY_SETS, atom_feature_names, development_ids, feature_sets


def _catalog():
    return pd.DataFrame([
        ("n_C", "composition", "molecule", True, True), ("cm_01", "cm", "molecule", True, True),
        ("bond_dipole_norm", "engineered", "molecule", True, True),
        ("bonds_F-F", "engineered", "molecule", True, False), ("n_ammonium", "groups", "molecule", True, True),
        ("qeq_dipole_D", "qeq", "molecule", True, True), ("rdf_CC_0.90", "rdf", "molecule", True, True),
        ("alpha", "dft_property", "molecule", False, True), ("charge_dipole_D", "charge", "molecule", False, True),
        ("freq_lowest", "frequency", "molecule", False, True),
        ("is_C", "atom", "atom", True, True), ("degree", "atom", "atom", True, False), ("is_H", "atom", "atom", True, True),
    ], columns=["feature", "block", "level", "legal", "kept"])


def test_feature_sets_respect_blocks_and_legality():
    catalog = _catalog()
    sets = feature_sets(catalog)
    assert sets["structure"] == ["n_C", "bond_dipole_norm"]  # dropped features never return
    assert sets["structure+"] == ["n_C", "bond_dipole_norm", "n_ammonium", "qeq_dipole_D"]
    assert sets["all_legal"] == [*sets["structure+"], "rdf_CC_0.90", "cm_01"]
    assert not any("is_" in c for cols in sets.values() for c in cols)  # atom features stay out
    legal = dict(zip(catalog["feature"], catalog["legal"]))
    for name, cols in sets.items():
        uses_dft = not all(legal[c] for c in cols)
        assert uses_dft == (name in EXPLORATION_ONLY_SETS), name


def test_atom_feature_names_keep_descriptor_order():
    assert atom_feature_names(_catalog()) == ["is_C", "is_H"]


def test_development_ids_never_include_tests():
    class S:
        pool, dev, dev_unseen = np.array([5, 1]), np.array([9]), np.array([3])
        test_unseen, test_familiar = np.array([2]), np.array([4])

    ids = development_ids(S)
    assert ids.tolist() == [1, 3, 5, 9]
    assert not set(ids) & {2, 4}
