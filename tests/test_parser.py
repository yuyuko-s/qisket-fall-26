"""Parser tests: inline QM9 records, plus a few known molecules read from the real tarball."""

import itertools

import numpy as np
import pytest

from qm9dipole.data import (
    HEADLINE_INPUTS, TABLE_COLUMNS, TARBALL, TARGET, hill_formula, iter_tarball,
    load_qm9_table, parse_xyz, qm9_table_is_current, save_qm9_table,
)

# Molecule 1 (methane), verbatim from the tarball.
METHANE = (
    "5\n"
    "gdb 1\t157.7118\t157.70997\t157.70699\t0.\t13.21\t-0.3877\t0.1171\t0.5048\t35.3641\t"
    "0.044749\t-40.47893\t-40.476062\t-40.475117\t-40.498597\t6.469\t\n"
    "C\t-0.0126981359\t 1.0858041578\t 0.0080009958\t-0.535689\n"
    "H\t 0.002150416\t-0.0060313176\t 0.0019761204\t 0.133921\n"
    "H\t 1.0117308433\t 1.4637511618\t 0.0002765748\t 0.133922\n"
    "H\t-0.540815069\t 1.4475266138\t-0.8766437152\t 0.133923\n"
    "H\t-0.5238136345\t 1.4379326443\t 0.9063972942\t 0.133923\n"
    "1341.307\t1341.3284\t1341.365\t1562.6731\t1562.7453\t3038.3205\t3151.6034\t3151.6788\t3151.7078\n"
    "C\tC\t\n"
    "InChI=1S/CH4/h1H4\tInChI=1S/CH4/h1H4\n"
)

needs_tarball = pytest.mark.skipif(not TARBALL.exists(), reason="QM9 tarball not downloaded")


def test_parses_methane():
    m = parse_xyz(METHANE)
    assert m["id"] == 1
    assert m["formula"] == "CH4"
    assert (m["n_atoms"], m["n_heavy"]) == (5, 1)
    assert m["Z"].tolist() == [6, 1, 1, 1, 1]
    assert m["R"].shape == (5, 3)
    assert m["R"][0] == pytest.approx([-0.0126981359, 1.0858041578, 0.0080009958])
    assert m["mu"] == 0.0
    assert m["smiles"] == "C"


def test_keeps_every_field_of_the_record():
    m = parse_xyz(METHANE)
    assert set(m) == set(TABLE_COLUMNS)
    # Frequencies: 3·5 − 6 = 9 vibrational modes for non-linear methane, in cm⁻¹.
    assert m["freqs"].shape == (9,) and m["freqs"][0] == pytest.approx(1341.307)
    assert (m["smiles_gdb"], m["smiles"]) == ("C", "C")
    assert m["inchi"] == m["inchi_gdb"] == "InChI=1S/CH4/h1H4"


def test_keeps_charges_and_all_properties_for_exploration():
    m = parse_xyz(METHANE)
    assert m["q"] == pytest.approx([-0.535689, 0.133921, 0.133922, 0.133923, 0.133923])
    assert abs(m["q"].sum()) < 1e-5  # neutral molecule
    assert -0.535689 not in m["R"]  # charges are not folded into the coordinates
    # Readme order: A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
    assert (m["A"], m["mu"], m["alpha"], m["homo"], m["U0"], m["Cv"]) == pytest.approx(
        (157.7118, 0.0, 13.21, -0.3877, -40.47893, 6.469)
    )


def test_headline_inputs_are_the_briefs():
    # The brief's pipeline is composition + geometry -> model -> |mu|.
    assert HEADLINE_INPUTS == ("Z", "R") and TARGET == "mu"


def test_fortran_exponent():
    text = METHANE.replace("-0.0126981359", "2.1997*^-6")
    assert parse_xyz(text)["R"][0, 0] == pytest.approx(2.1997e-6)


def test_rejects_malformed_property_line():
    with pytest.raises(ValueError, match="property line"):
        parse_xyz(METHANE.replace("gdb 1", "xyz 1"))


@pytest.mark.parametrize("elements, expected", [
    (["C", "H", "H", "H", "H"], "CH4"),
    (["O", "C", "H", "H"], "CH2O"),
    (["H", "H", "O"], "H2O"),
    (["N", "H", "H", "H"], "H3N"),
    (["F", "C", "C", "N", "H", "O"], "C2HFNO"),
])
def test_hill_formula(elements, expected):
    assert hill_formula(elements) == expected


def test_hill_formula_rejects_unknown_elements():
    with pytest.raises(ValueError):
        hill_formula(["C", "S"])


def test_parquet_round_trip(tmp_path):
    import pandas as pd

    water_like = parse_xyz(METHANE.replace("gdb 1", "gdb 2"))
    table = pd.DataFrame([parse_xyz(METHANE), water_like])
    path = tmp_path / "t.parquet"
    save_qm9_table(table, path)
    back = load_qm9_table(path)
    assert back["id"].tolist() == [1, 2]
    for a, b in zip(table["R"], back["R"]):
        np.testing.assert_array_equal(a, b)  # bit-exact, shape (n, 3) restored
    for col in ("q", "freqs"):
        for a, b in zip(table[col], back[col]):
            np.testing.assert_array_equal(a, b)
    assert back["Z"][0].dtype == np.int8
    assert qm9_table_is_current(path)


def test_table_from_older_parser_is_stale(tmp_path):
    import pandas as pd

    table = pd.DataFrame([parse_xyz(METHANE)]).drop(columns=["freqs", "inchi"])
    path = tmp_path / "old.parquet"
    save_qm9_table(table, path)
    assert not qm9_table_is_current(path)
    assert not qm9_table_is_current(tmp_path / "missing.parquet")


@needs_tarball
def test_first_molecules_from_tarball():
    first = [parse_xyz(text) for _, text in itertools.islice(iter_tarball(), 3)]
    assert [m["formula"] for m in first] == ["CH4", "H3N", "H2O"]
    assert [len(m["freqs"]) for m in first] == [9, 6, 3]  # 3n − 6 vibrational modes
    assert first[0]["mu"] == 0.0           # methane: bond dipoles cancel by symmetry
    assert 1.0 < first[1]["mu"] < 2.5      # ammonia is polar
    assert 1.0 < first[2]["mu"] < 2.5      # water is polar (experiment ≈ 1.85 D)


@needs_tarball
def test_fortran_exponent_in_real_file():
    # Molecule 212 is the first with a "*^" coordinate: C  2.1997*^-6  1.4462618059 ...
    for name, text in iter_tarball():
        if name == "dsgdb9nsd_000212.xyz":
            m = parse_xyz(text)
            assert m["R"][0] == pytest.approx([2.1997e-6, 1.4462618059, 0.0098312216])
            return
    pytest.fail("molecule 212 not found")
