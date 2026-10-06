"""Invariance checks (M2, PLAN §8): does a descriptor change when the molecule is rotated,
translated or has its atoms relabeled? A physically sensible |μ| predictor must not.

Each transform maps (Z, R, rng) to a transformed (Z, R) describing the same molecule.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from qm9dipole.descriptors import DESCRIPTORS

#: Default tolerance (PLAN §8.1). Double-precision noise in these descriptors is ~1e-12.
ATOL = 1e-8


def rotate(Z: np.ndarray, R: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly random 3D rotation about the origin."""
    return Z, R @ Rotation.random(random_state=rng).as_matrix().T


def translate(Z: np.ndarray, R: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random rigid shift of up to 10 Å along each axis."""
    return Z, R + rng.uniform(-10.0, 10.0, size=3)


def permute(Z: np.ndarray, R: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random relabeling: Z and R are permuted together, so the molecule is unchanged."""
    p = rng.permutation(len(Z))
    return Z[p], R[p]


TRANSFORMS: dict[str, Callable] = {"rotation": rotate, "translation": translate, "permutation": permute}


def max_change(descriptor: str, transform: str, molecules: list[tuple[np.ndarray, np.ndarray]],
               rng: np.random.Generator) -> np.ndarray:
    """Max |descriptor(transformed) − descriptor(original)| for each molecule."""
    fn, tf = DESCRIPTORS[descriptor], TRANSFORMS[transform]
    return np.array([np.abs(fn(*tf(Z, R, rng)) - fn(Z, R)).max() for Z, R in molecules])


def invariance_table(molecules: list[tuple[np.ndarray, np.ndarray]], seed: int,
                     descriptors: tuple[str, ...] = tuple(DESCRIPTORS), atol: float = ATOL) -> pd.DataFrame:
    """One row per (descriptor, transform): mean and max change, and whether max ≤ atol.

    Each (descriptor, transform) pair gets its own generator seeded from `seed`, so a row
    does not depend on which other rows were computed.
    """
    rows = []
    for i, d in enumerate(descriptors):
        for j, t in enumerate(TRANSFORMS):
            rng = np.random.default_rng([seed, i, j])
            change = max_change(d, t, molecules, rng)
            rows.append({"descriptor": d, "transform": t, "n_molecules": len(molecules),
                         "mean_change": change.mean(), "max_change": change.max(),
                         "invariant": bool(change.max() <= atol)})
    return pd.DataFrame(rows)
