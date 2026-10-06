"""Data cleaning for the exploration chain (notebooks/explore_02), motivated by the EDA
(notebooks/explore_01). Rules run on the training pool only and never read a test set.

Record level (QM9 quirks found in the EDA):
- frequency lists printed twice: 433 raw records list exactly 2 × (3n − 6) frequencies;
  `clean_frequencies` keeps the first copy;
- linear molecules: QM9 stores rotational constant A = 0 in place of infinity;
  `moments_from_rotational_constants` gives the finite moments of inertia instead;
- geometric duplicates: a few molecules are listed twice, with identical geometry and μ.
  `duplicate_groups` finds them from Z and R alone (formula + rounded Coulomb spectrum).
  InChI cannot be used for this: standard InChI merges tautomers, which are different
  molecules with different dipoles.

Feature level, unsupervised (a rule never sees μ, so applying it to the whole pool cannot
leak labels into small-N learning curves):
- `zero_variance`: columns that never change carry no information;
- `redundant_columns`: of two columns with |Spearman ρ| ≥ a threshold (the same information
  up to a monotone transform), the one later in priority order is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qm9dipole.features import ROTATIONAL_GHZ_AMU_A2


def vibrational_modes(n_atoms: int, linear: bool) -> int:
    """3n − 5 modes for a linear molecule, 3n − 6 otherwise."""
    return 3 * n_atoms - (5 if linear else 6)


def clean_frequencies(freqs: np.ndarray, n_atoms: int) -> np.ndarray:
    """The frequency list with a doubled copy removed (cm⁻¹).

    A record listing exactly twice the expected number of modes keeps its first half;
    anything else is returned unchanged. 2(3n − 6) never equals 3n − 5, so a doubled list
    cannot be mistaken for a linear molecule's.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    for modes in (vibrational_modes(n_atoms, False), vibrational_modes(n_atoms, True)):
        if len(freqs) == 2 * modes:
            return freqs[:modes]
    return freqs


def is_linear(clean_freqs: np.ndarray, n_atoms: int) -> bool:
    """True if a cleaned spectrum has 3n − 5 modes (a linear molecule)."""
    return len(clean_freqs) == vibrational_modes(n_atoms, True)


def moments_from_rotational_constants(abc: np.ndarray) -> np.ndarray:
    """Moments of inertia (amu·Å²) from rotational constants (GHz); A = 0 (QM9's stand-in
    for an infinite constant, linear molecules) maps to a zero moment."""
    abc = np.asarray(abc, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(abc > 0, ROTATIONAL_GHZ_AMU_A2 / np.where(abc > 0, abc, 1.0), 0.0)


def duplicate_groups(ids: np.ndarray, formulas: pd.Series | np.ndarray, spectra: np.ndarray,
                     decimals: int = 2) -> pd.Series:
    """Group label for every molecule that shares its geometry with another, by ID.

    Key: formula + Coulomb-matrix spectrum rounded to `decimals` (units of the spectrum).
    Only molecules in a group of ≥ 2 appear in the result; the label is the group's
    smallest ID.
    """
    keys = pd.Series([f"{f}|" + ",".join(f"{v:.{decimals}f}" for v in row)
                      for f, row in zip(np.asarray(formulas), spectra)], index=np.asarray(ids))
    dup = keys[keys.duplicated(keep=False)]
    first = dup.groupby(dup).transform(lambda s: s.index.min())
    return first.rename("duplicate_of").sort_index()


def duplicate_extras(ids: np.ndarray, formulas: pd.Series | np.ndarray, spectra: np.ndarray,
                     decimals: int = 2) -> frozenset[int]:
    """IDs to exclude so that every duplicate group keeps exactly one molecule, its smallest
    ID. Dropping the extra copies before splitting means no molecule can sit in two sets
    (exploration X2; DECISIONS.md)."""
    group = duplicate_groups(ids, formulas, spectra, decimals)
    return frozenset(int(i) for i, first in group.items() if i != first)


def zero_variance(frame: pd.DataFrame) -> list[str]:
    """Columns with a single distinct value."""
    return [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1]


def redundant_columns(frame: pd.DataFrame, threshold: float,
                      priority: list[str] | None = None) -> pd.DataFrame:
    """Greedy redundancy filter: walk columns in priority order and keep a column unless its
    |Spearman ρ| with an already kept column is ≥ `threshold`.

    Returns one row per dropped column: (dropped, kept, rho). Constant columns should be
    removed first (their correlation is undefined).
    """
    order = list(priority or frame.columns)
    rho = frame[order].corr(method="spearman").abs()
    kept, rows = [], []
    for col in order:
        match = next((k for k in kept if rho.loc[col, k] >= threshold), None)
        if match is None:
            kept.append(col)
        else:
            rows.append({"dropped": col, "kept": match, "rho": float(rho.loc[col, match])})
    return pd.DataFrame(rows, columns=["dropped", "kept", "rho"])
