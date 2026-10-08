"""EDA helpers (notebooks/explore_01): a data dictionary of every QM9 field, integrity
checks, and per-molecule summaries of the array-valued fields.

Statistics should be computed on training-pool rows only; `integrity_checks` and the
dictionary are label-free and may be run on any rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qm9dipole.cleaning import moments_from_rotational_constants
from qm9dipole.descriptors import COMPOSITION_ELEMENTS, composition
from qm9dipole.features import bonds, inertia_moments

# Roles: what each field may be used for (docs/EVALUATION_RULES.md rule 1).
HEADLINE = "headline input"
TARGET_ROLE = "target"
FROM_ZR = "function of Z and R (headline-legal)"
DFT = "DFT output (exploration only)"
ID = "identifier / bookkeeping"

#: Every column of the parsed table: (column, per, unit, meaning, role).
DATA_DICTIONARY: tuple[tuple[str, str, str, str, str], ...] = (
    ("id", "molecule", "", "QM9 index (GDB-17 numbering)", ID),
    ("formula", "molecule", "", "element counts, e.g. C2H6O (from Z)", FROM_ZR),
    ("n_atoms", "molecule", "", "number of atoms (from Z)", FROM_ZR),
    ("n_heavy", "molecule", "", "number of C, N, O, F atoms (from Z)", FROM_ZR),
    ("Z", "atom", "", "atomic number", HEADLINE),
    ("R", "atom", "Å", "Cartesian coordinates of the B3LYP-optimized geometry", HEADLINE),
    ("q", "atom", "e", "Mulliken partial charge", DFT),
    ("A", "molecule", "GHz", "rotational constant (largest)", FROM_ZR),
    ("B", "molecule", "GHz", "rotational constant", FROM_ZR),
    ("C", "molecule", "GHz", "rotational constant (smallest)", FROM_ZR),
    ("mu", "molecule", "D", "dipole moment magnitude |μ|", TARGET_ROLE),
    ("alpha", "molecule", "bohr³", "isotropic polarizability", DFT),
    ("homo", "molecule", "Ha", "highest occupied orbital energy", DFT),
    ("lumo", "molecule", "Ha", "lowest unoccupied orbital energy", DFT),
    ("gap", "molecule", "Ha", "lumo − homo", DFT),
    ("r2", "molecule", "bohr²", "electronic spatial extent ⟨R²⟩", DFT),
    ("zpve", "molecule", "Ha", "zero-point vibrational energy", DFT),
    ("U0", "molecule", "Ha", "internal energy at 0 K", DFT),
    ("U", "molecule", "Ha", "internal energy at 298.15 K", DFT),
    ("H", "molecule", "Ha", "enthalpy at 298.15 K", DFT),
    ("G", "molecule", "Ha", "free energy at 298.15 K", DFT),
    ("Cv", "molecule", "cal/(mol·K)", "heat capacity at 298.15 K", DFT),
    ("freqs", "mode", "cm⁻¹", "harmonic vibrational frequencies (3n − 6 modes)", DFT),
    ("smiles", "molecule", "", "SMILES of the relaxed geometry", ID),
    ("smiles_gdb", "molecule", "", "SMILES of the GDB-17 input structure", ID),
    ("inchi", "molecule", "", "InChI of the relaxed geometry", ID),
    ("inchi_gdb", "molecule", "", "InChI of the GDB-17 input structure", ID),
)

#: The 15 scalar properties of line 2, plus the molecule-level counts.
SCALAR_PROPERTIES: tuple[str, ...] = (
    "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv",
)


#: Numbers QM9 provides directly, one per molecule, usable as features without engineering:
#: the counts read off the atom list and the 14 computed properties besides μ. Per-atom
#: arrays (coordinates, charges, frequencies) need engineering to become fixed-length
#: features, so they are not "natural" here.
NATURAL_FEATURES: tuple[str, ...] = (
    "n_atoms", "n_heavy", *(f"n_{e}" for e in COMPOSITION_ELEMENTS),
    *(p for p in SCALAR_PROPERTIES if p != "mu"),
)


def natural_features(table: pd.DataFrame) -> pd.DataFrame:
    """NATURAL_FEATURES for the rows of the parsed table, indexed by molecule id."""
    counts = np.stack([composition(z) for z in table["Z"]])
    out = table[["n_atoms", "n_heavy", *(p for p in SCALAR_PROPERTIES if p != "mu")]].copy()
    for k, e in enumerate(COMPOSITION_ELEMENTS):
        out[f"n_{e}"] = counts[:, k]
    out.index = pd.Index(table["id"].to_numpy(), name="id")
    return out[list(NATURAL_FEATURES)]


def natural_role(feature: str) -> str:
    """Role of a natural feature (docs/EVALUATION_RULES.md rule 1): counts and the rotational constants are
    functions of Z and R; the electronic and thermal properties are DFT outputs."""
    roles = {row[0]: row[4] for row in DATA_DICTIONARY}
    return FROM_ZR if feature.startswith("n_") else roles[feature]


def data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(DATA_DICTIONARY, columns=["column", "per", "unit", "meaning", "role"])


def per_molecule_summaries(table: pd.DataFrame) -> pd.DataFrame:
    """Scalar summaries of the array-valued fields, one row per molecule (indexed by id):
    composition counts (from Z), centroid distance from the origin (from R), charge
    statistics (from q) and frequency-list statistics (from freqs, as listed in the file)."""
    rows = []
    for z, r, q, f, n in zip(table["Z"], table["R"], table["q"], table["freqs"], table["n_atoms"]):
        rows.append({
            **{f"n_{e}": c for e, c in zip(COMPOSITION_ELEMENTS, composition(z))},
            "centroid_offset": float(np.linalg.norm(r.mean(axis=0))),
            "q_sum": float(q.sum()), "q_max_abs": float(np.abs(q).max()),
            "n_freqs_listed": len(f), "n_freqs_expected": 3 * n - 6,
            "freq_min": float(f.min()), "freq_max": float(f.max()),
        })
    return pd.DataFrame(rows, index=pd.Index(table["id"].to_numpy(), name="id"))


def integrity_checks(table: pd.DataFrame) -> pd.DataFrame:
    """Domain rules every record should satisfy, with violation counts and the largest
    deviation. Label-free: none of these looks at μ except the μ ≥ 0 check."""
    t = table
    summ = per_molecule_summaries(t)
    moments = np.array([inertia_moments(z, r) for z, r in zip(t["Z"], t["R"])])
    from_qm9 = np.sort(moments_from_rotational_constants(t[["A", "B", "C"]].to_numpy()), axis=1)
    rot_dev = np.abs(moments - from_qm9).max(axis=1) / np.maximum(moments.max(axis=1), 1e-9)
    n_bonds_ok = np.array([len(bonds(z, r)) >= len(z) - 1 for z, r in zip(t["Z"], t["R"])])
    listed, n = summ["n_freqs_listed"].to_numpy(), t["n_atoms"].to_numpy()
    linear = (listed == 3 * n - 5) | (listed == 2 * (3 * n - 5))  # row by row, not set membership

    def row(check, ok, deviation=np.nan, note=""):
        ok = np.asarray(ok, dtype=bool)
        return {"check": check, "checked": int(ok.size), "violations": int((~ok).sum()),
                "max deviation": deviation, "note": note}

    gap_dev = (t["gap"] - (t["lumo"] - t["homo"])).abs()
    h_minus_u = t["H"] - t["U"]
    return pd.DataFrame([
        row("len(Z) = len(R) = len(q) = n_atoms",
            [len(z) == len(r) == len(q) == n for z, r, q, n in zip(t["Z"], t["R"], t["q"], t["n_atoms"])]),
        row("no missing scalar values", ~t[list(SCALAR_PROPERTIES)].isna().any(axis=1)),
        row("neutral: |Σ q| < 1e-4 e", summ["q_sum"].abs() < 1e-4, summ["q_sum"].abs().max()),
        row("μ ≥ 0", t["mu"] >= 0),
        row("gap = lumo − homo (values are rounded to 1e-4 Ha)", gap_dev < 1.5e-4, gap_dev.max()),
        row("H − U = RT at 298.15 K (0.000944 Ha)", (h_minus_u - 0.000944).abs() < 3e-6,
            (h_minus_u - 0.000944).abs().max()),
        row("U0 < U and G < H", (t["U0"] < t["U"]) & (t["G"] < t["H"])),
        row("A ≥ B ≥ C, or A = 0 (linear)", ((t["A"] >= t["B"]) | (t["A"] == 0)) & (t["B"] >= t["C"])),
        row("rotational constants = 505.379 GHz·amu·Å² / I(Z, R)", rot_dev < 1e-4, float(rot_dev.max()),
            "relative; I from isotope masses and R"),
        row("frequencies: 3n − 6 listed (3n − 5 if linear)",
            (listed == 3 * n - 6) | (listed == 3 * n - 5),
            note="exactly twice that is a doubled list in the raw file"),
        row("no imaginary (negative) frequency", summ["freq_min"] > 0, note="one would mean a saddle point"),
        row("bond graph has ≥ n − 1 bonds (connected)", n_bonds_ok),
        row("linear molecules store A = 0", ~linear | (t["A"].to_numpy() == 0),
            note=f"{int(linear.sum())} linear molecules"),
    ])


def bond_lengths(table: pd.DataFrame) -> pd.DataFrame:
    """Every inferred bond as (pair, length in Å), pair in composition element order."""
    sym = dict(zip((6, 1, 7, 8, 9), COMPOSITION_ELEMENTS))
    order = list(COMPOSITION_ELEMENTS)
    rows = []
    for z, r in zip(table["Z"], table["R"]):
        for i, j in bonds(z, r):
            a, b = sorted((sym[int(z[i])], sym[int(z[j])]), key=order.index)
            rows.append((f"{a}-{b}", float(np.linalg.norm(r[i] - r[j]))))
    return pd.DataFrame(rows, columns=["pair", "length"])

