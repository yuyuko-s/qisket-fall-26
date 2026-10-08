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
    (exploration X2)."""
    group = duplicate_groups(ids, formulas, spectra, decimals)
    return frozenset(int(i) for i, first in group.items() if i != first)


def zero_variance(frame: pd.DataFrame) -> list[str]:
    """Columns with a single distinct value."""
    return [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1]


def redundant_columns(frame: pd.DataFrame, threshold: float, priority: list[str] | None = None,
                      active_only: bool = True, min_active: int = 30) -> pd.DataFrame:
    """Greedy redundancy filter: walk columns in priority order and keep a column unless its
    |Spearman ρ| with an already kept column is ≥ `threshold`.

    With `active_only` (exploration X2), a pair must also reach the threshold on the rows where
    either column is non-zero (when there are at least `min_active` such rows). Zero-inflated
    columns, such as counts of a rare neighbour, share long runs of tied zeros, which alone
    push ρ over all rows towards 1 even when their non-zero values differ; the second test
    keeps only pairs that are monotone where they actually vary.

    Returns one row per dropped column: (dropped, kept, rho, rho_active). Constant columns
    should be removed first (their correlation is undefined).
    """
    order = list(priority or frame.columns)
    rho = frame[order].corr(method="spearman").abs()
    nonzero = frame[order].to_numpy() != 0
    col_index = {c: i for i, c in enumerate(order)}
    kept, rows = [], []
    for col in order:
        match, rho_active = None, np.nan
        for k in kept:
            if rho.loc[col, k] < threshold:
                continue
            if active_only:
                active = nonzero[:, col_index[col]] | nonzero[:, col_index[k]]
                if min_active <= active.sum() < len(active):
                    pair = frame.loc[active, [col, k]]
                    if (pair.nunique() == 1).all():
                        rho_active = 1.0  # both constant where active: they mark the same rows
                    else:
                        rho_active = abs(pair.corr(method="spearman").iloc[0, 1])
                    if not rho_active >= threshold:  # NaN: one varies where the other is constant
                        continue
            match = k
            break
        if match is None:
            kept.append(col)
        else:
            rows.append({"dropped": col, "kept": match, "rho": float(rho.loc[col, match]), "rho_active": rho_active})
    return pd.DataFrame(rows, columns=["dropped", "kept", "rho", "rho_active"])
