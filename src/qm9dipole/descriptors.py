"""Molecular descriptors (M2, PLAN §5). Every descriptor is a function of (Z, R) only.

- composition: counts of C, H, N, O, F (5 numbers). Invariant by construction, but blind to
  geometry, so it cannot tell isomers apart.
- cm_spectrum: eigenvalues of the Coulomb matrix (Rupp et al. 2012), sorted by decreasing
  magnitude and zero-padded to 29. The matrix depends only on atomic numbers and interatomic
  distances, so its spectrum is invariant to rotation and translation; eigenvalues do not
  depend on the order of the atoms, so it is also invariant to relabeling.
- raw_coordinates: flattened, zero-padded coordinates. Deliberately NOT invariant: it is the
  negative control that the invariance tests must catch (PLAN §8.3).
- engineered: physics-motivated geometry, bond and polarity features (exploration; defined
  in `qm9dipole.features`, names in `features.ENGINEERED_NAMES`).

Fitted transforms (standardization, PCA for the 8-qubit "compressed" variant) are not here:
they must be fit on each training set, so they live in the model pipelines (M3+).

Units: R in Å; Coulomb-matrix distances in bohr.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from qm9dipole.features import ENGINEERED_NAMES, engineered

ANGSTROM_TO_BOHR = 1.8897261
MAX_ATOMS = 29  # largest QM9 molecule, hydrogens included

#: Element order of the composition vector (PLAN §5): C, H, N, O, F.
COMPOSITION_ELEMENTS: tuple[str, ...] = ("C", "H", "N", "O", "F")
_COMPOSITION_Z = np.array([6, 1, 7, 8, 9])


def composition(Z: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """Counts of C, H, N, O, F. `R` is accepted for a uniform signature and ignored."""
    return (np.asarray(Z)[:, None] == _COMPOSITION_Z[None, :]).sum(axis=0).astype(np.float64)


def coulomb_matrix(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Coulomb matrix: M_ii = 0.5·Z_i^2.4, M_ij = Z_i·Z_j / |R_i − R_j| (distances in bohr)."""
    Z = np.asarray(Z, dtype=np.float64)
    R_bohr = np.asarray(R, dtype=np.float64) * ANGSTROM_TO_BOHR
    dist = np.linalg.norm(R_bohr[:, None, :] - R_bohr[None, :, :], axis=-1)
    np.fill_diagonal(dist, 1.0)  # placeholder; the diagonal is overwritten below
    M = np.outer(Z, Z) / dist
    np.fill_diagonal(M, 0.5 * Z**2.4)
    return M


def cm_spectrum(Z: np.ndarray, R: np.ndarray, size: int = MAX_ATOMS) -> np.ndarray:
    """Coulomb-matrix eigenvalues sorted by decreasing |λ|, zero-padded to `size`.

    Ties in |λ| (equal up to floating-point noise) are broken by λ itself, so the order
    cannot flip between, e.g., +a and −a when atoms are relabeled.
    """
    lam = np.linalg.eigvalsh(coulomb_matrix(Z, R))
    order = np.lexsort((-lam, -np.round(np.abs(lam), 8)))  # primary key last: |λ| descending
    out = np.zeros(size)
    out[: len(lam)] = lam[order]
    return out


def raw_coordinates(Z: np.ndarray, R: np.ndarray, size: int = MAX_ATOMS) -> np.ndarray:
    """Negative control: coordinates flattened in atom order and zero-padded to 3·`size`."""
    out = np.zeros(3 * size)
    flat = np.asarray(R, dtype=np.float64).ravel()
    out[: len(flat)] = flat
    return out


#: Descriptors by name. Each maps (Z, R) to a fixed-length vector.
DESCRIPTORS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "composition": composition,
    "cm_spectrum": cm_spectrum,
    "raw_coordinates": raw_coordinates,
    "engineered": engineered,
}

#: The descriptors models may use. raw_coordinates exists only as the negative control.
MODEL_DESCRIPTORS: tuple[str, ...] = ("composition", "cm_spectrum", "engineered")


def featurize(table: pd.DataFrame, name: str) -> np.ndarray:
    """Descriptor matrix (n_molecules × d) for the rows of `table`, in row order."""
    fn = DESCRIPTORS[name]
    return np.stack([fn(z, r) for z, r in zip(table["Z"], table["R"])])


def feature_names(name: str) -> list[str]:
    """Column names of descriptor `name`, in vector order."""
    match name:
        case "composition":
            return [f"n_{e}" for e in COMPOSITION_ELEMENTS]
        case "cm_spectrum":
            return [f"cm_{k:02d}" for k in range(1, MAX_ATOMS + 1)]
        case "raw_coordinates":
            return [f"{axis}_{k:02d}" for k in range(1, MAX_ATOMS + 1) for axis in "xyz"]
        case "engineered":
            return list(ENGINEERED_NAMES)
    raise KeyError(f"unknown descriptor {name!r}")


def feature_frame(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """`featurize` as a DataFrame indexed by molecule ID, with named columns."""
    return pd.DataFrame(featurize(table, name), columns=feature_names(name),
                        index=pd.Index(table["id"].to_numpy(), name="id"))
