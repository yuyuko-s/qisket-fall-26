"""EDA helper tests (eda.py): the integrity checks count what they claim to."""

import numpy as np
import pandas as pd

from qm9dipole.data import TABLE_COLUMNS, parse_xyz
from qm9dipole.eda import DATA_DICTIONARY, integrity_checks, per_molecule_summaries
from test_parser import METHANE


def test_dictionary_covers_every_column():
    assert [row[0] for row in DATA_DICTIONARY] == list(TABLE_COLUMNS)


def test_integrity_checks_count_violations():
    good = parse_xyz(METHANE)
    doubled = parse_xyz(METHANE.replace("gdb 1", "gdb 2"))
    doubled["freqs"] = np.concatenate([doubled["freqs"], doubled["freqs"]])
    broken = parse_xyz(METHANE.replace("gdb 1", "gdb 3"))
    broken["gap"] += 0.01
    checks = integrity_checks(pd.DataFrame([good, doubled, broken])).set_index("check")["violations"]
    assert checks["frequencies: 3n − 6 listed (3n − 5 if linear)"] == 1
    assert checks["gap = lumo − homo (values are rounded to 1e-4 Ha)"] == 1
    assert checks["rotational constants = 505.379 GHz·amu·Å² / I(Z, R)"] == 0
    assert checks.drop(["frequencies: 3n − 6 listed (3n − 5 if linear)",
                        "gap = lumo − homo (values are rounded to 1e-4 Ha)"]).sum() == 0


def test_per_molecule_summaries():
    s = per_molecule_summaries(pd.DataFrame([parse_xyz(METHANE)])).iloc[0]
    assert (s["n_C"], s["n_H"], s["n_freqs_listed"], s["n_freqs_expected"]) == (1, 4, 9, 9)


def test_natural_features_are_the_fields_qm9_gives_directly():
    from qm9dipole.eda import DFT, FROM_ZR, NATURAL_FEATURES, natural_features, natural_role

    m = parse_xyz(METHANE)
    nat = natural_features(pd.DataFrame([m]))
    assert list(nat.columns) == list(NATURAL_FEATURES) and len(NATURAL_FEATURES) == 21
    assert "mu" not in nat.columns  # the target is never a feature
    row = nat.loc[1]
    assert (row["n_atoms"], row["n_C"], row["n_H"], row["A"]) == (5, 1, 4, m["A"])
    assert natural_role("n_O") == FROM_ZR and natural_role("B") == FROM_ZR and natural_role("alpha") == DFT
