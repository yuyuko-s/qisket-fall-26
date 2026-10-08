"""Engineered, physics-motivated descriptors for |μ| (functions of Z and R).

Every feature is a function of (Z, R) alone plus fixed tabulated constants (covalent radii,
Pauling electronegativities, isotope masses), never another QM9 property, so these are headline-legal
inputs. Each is invariant to rotation, translation and atom relabeling; they are registered
in `descriptors.MODEL_DESCRIPTORS`, so the descriptor invariance tests cover them.

Groups (names in ENGINEERED_NAMES):
- geometry: radius of gyration, the three eigenvalues (moments) of the gyration tensor,
  and the largest interatomic distance. Size and shape set how far charge *can* be
  separated.
- polarity: bonds are inferred from distances (d < BOND_SCALE · (r_i + r_j)). Each bond is
  given a dipole along its direction with strength Δχ, the electronegativity difference:
  `bond_polarity_total` = Σ |Δχ| (how polar the bonds are), and `bond_dipole_norm` =
  |Σ Δχ·û| (how much of that survives geometric cancellation, the textbook vector-sum
  picture). `en_point_dipole` is the same idea with atom-centred pseudo-charges, and
  `heteroatom_offset` is how far the N/O/F centroid sits from the molecular centroid.
- bond counts by element pair (C-C, C-H, ..., F-F), in composition element order.
- bond orders: multiple bonds told apart by length (cutoffs at the valleys of the training
  pool's bond-length histograms, explore_01): C≡C, C≡N (nitrile), C=N or aromatic C–N,
  C=O (carbonyl), N=O. Nitriles and carbonyls are among the most polar groups.
- rings: the cyclomatic number of the bond graph (bonds − atoms + connected parts).
- inertia: the three principal moments of inertia (amu·Å², isotope masses). QM9's rotational
  constants A, B, C are exactly 505.379 GHz·amu·Å² / I (explore_01 verifies this), so they
  are functions of Z and R too; moments stay finite for linear molecules, where QM9 stores
  A = 0 in place of infinity.

Exploration-only summaries of the DFT outputs (`charge_features`, `frequency_features`) are
at the end of this module. They use the Mulliken charges or the vibrational frequencies, so
they are NOT descriptors and are never registered in `descriptors.DESCRIPTORS`.

Units: lengths in Å; electronegativity in Pauling units, so the dipole proxies are in
Pauling·Å. They track μ but are not calibrated to debye; the models learn that mapping.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

#: Single-bond covalent radii in Å (Cordero et al., Dalton Trans. 2008; sp3 carbon).
COVALENT_RADII: dict[int, float] = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
#: Pauling electronegativities.
ELECTRONEGATIVITY: dict[int, float] = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98}
#: Atoms i, j are bonded if |R_i − R_j| < BOND_SCALE · (r_i + r_j). In QM9, bonded and
#: non-bonded distances separate near 1.15× for every element pair; the tightest is C–C,
#: where cross-ring C···C pairs in strained cages start at ~1.75 Å (1.15 × 1.52). At 1.2,
#: 1.3% of molecules got a 5-bonded carbon. At 1.15, no atom in the 8,438-molecule working
#: pool exceeds its valence (H 1, C 4, N 4, O 2, F 1) or is left unbonded; vs. 1.12, it
#: adds 15 long N–O/C–N bonds (1.54–1.65 Å) in strained rings.
BOND_SCALE = 1.15

_ELEMENTS = ("C", "H", "N", "O", "F")  # composition order (descriptors.COMPOSITION_ELEMENTS)
_ELEMENT_INDEX = {6: 0, 1: 1, 7: 2, 8: 3, 9: 4}
_PAIRS = list(combinations_with_replacement(range(len(_ELEMENTS)), 2))
_PAIR_INDEX = {pair: k for k, pair in enumerate(_PAIRS)}
_HETEROATOMS = (7, 8, 9)

BOND_TYPES: tuple[str, ...] = tuple(f"{_ELEMENTS[a]}-{_ELEMENTS[b]}" for a, b in _PAIRS)

#: Multiple bonds by length: (name, sorted atomic-number pair, lower Å, upper Å). Each upper
#: cutoff sits in an empty valley of the training pool's bond-length histogram (explore_01):
#: C≡C 1.18–1.22 | gap 1.22–1.32; C≡N 1.14–1.16 | gap 1.18–1.24; C–N double/aromatic up to
#: a dip at 1.42–1.44 before single bonds (~1.47); C=O 1.18–1.24 | gap 1.24–1.30;
#: N=O 1.20–1.24 | gap 1.26–1.32. C–C double vs single has no valley (aromatic bonds fill
#: it), so no cutoff is invented there.
BOND_ORDER_RULES: tuple[tuple[str, tuple[int, int], float, float], ...] = (
    ("triple_C-C", (6, 6), 0.0, 1.26),
    ("triple_C-N", (6, 7), 0.0, 1.21),
    ("multiple_C-N", (6, 7), 1.21, 1.43),
    ("double_C-O", (6, 8), 0.0, 1.27),
    ("double_N-O", (7, 8), 0.0, 1.28),
)

#: Isotope masses (amu) of 1H, 12C, 14N, 16O, 19F, as used for QM9's rotational constants.
ISOTOPE_MASS: dict[int, float] = {1: 1.00782503, 6: 12.0, 7: 14.0030740, 8: 15.9949146, 9: 18.9984032}
#: Rotational constant (GHz) = ROTATIONAL_GHZ_AMU_A2 / moment of inertia (amu·Å²).
ROTATIONAL_GHZ_AMU_A2 = 505.379009

ENGINEERED_NAMES: tuple[str, ...] = (
    "radius_of_gyration", "gyration_moment_1", "gyration_moment_2", "gyration_moment_3",
    "max_distance", "heteroatom_offset", "bond_polarity_total", "bond_dipole_norm",
    "en_point_dipole", *(f"bonds_{t}" for t in BOND_TYPES),
    *(name for name, *_ in BOND_ORDER_RULES), "n_rings",
    "inertia_moment_1", "inertia_moment_2", "inertia_moment_3",
)


def _lookup(table: dict[int, float], Z: np.ndarray) -> np.ndarray:
    return np.array([table[int(z)] for z in Z], dtype=np.float64)


def bonds(Z: np.ndarray, R: np.ndarray, scale: float = BOND_SCALE) -> np.ndarray:
    """Bonded atom pairs (n_bonds × 2, i < j), inferred from distances in Å."""
    r = _lookup(COVALENT_RADII, Z)
    R = np.asarray(R, dtype=np.float64)
    i, j = np.triu_indices(len(r), k=1)
    d = np.linalg.norm(R[i] - R[j], axis=1)
    keep = d < scale * (r[i] + r[j])
    return np.column_stack([i[keep], j[keep]])


#: Largest usual number of bonds per element (N up to 4, as in ammonium-like zwitterions).
MAX_VALENCE: dict[int, int] = {1: 1, 6: 4, 7: 4, 8: 2, 9: 1}


def valence_ok(Z: np.ndarray, pairs: np.ndarray) -> bool:
    """True if every atom has at least one bond and no more than MAX_VALENCE: a sanity
    check that the inferred bonds describe one connected-looking molecule."""
    degree = np.bincount(np.asarray(pairs, dtype=int).ravel(), minlength=len(Z))
    return all(1 <= d <= MAX_VALENCE[int(z)] for z, d in zip(Z, degree))


def bond_counts(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Number of bonds of each type in BOND_TYPES."""
    out = np.zeros(len(BOND_TYPES))
    for i, j in pairs:
        a, b = sorted((_ELEMENT_INDEX[int(Z[i])], _ELEMENT_INDEX[int(Z[j])]))
        out[_PAIR_INDEX[(a, b)]] += 1
    return out


def bond_dipoles(Z: np.ndarray, R: np.ndarray, pairs: np.ndarray) -> tuple[float, float]:
    """(Σ |Δχ|, |Σ Δχ·û|) over bonds; û is the unit bond vector from atom i to atom j.

    Swapping i and j flips both Δχ and û, so each term, and the sum, does not depend on
    the order of the atoms within a bond.
    """
    if len(pairs) == 0:
        return 0.0, 0.0
    chi = _lookup(ELECTRONEGATIVITY, Z)
    i, j = pairs.T
    v = np.asarray(R, dtype=np.float64)[j] - np.asarray(R, dtype=np.float64)[i]
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    dchi = chi[j] - chi[i]
    return float(np.abs(dchi).sum()), float(np.linalg.norm(dchi @ u))


def electronegativity_dipole(Z: np.ndarray, R: np.ndarray) -> float:
    """|Σᵢ qᵢ Rᵢ| for atom-centred pseudo-charges qᵢ derived from Pauling electronegativity.

    A crude stand-in for the point-charge dipole |Σ qᵢ rᵢ| that the Mulliken charges give,
    built only from Z and R. Pauling·Å; must be invariant to rotation,
    translation and atom relabeling (tests/test_features.py).
    """
    chi = _lookup(ELECTRONEGATIVITY, Z)
    R = np.asarray(R, dtype=np.float64)
    # Pseudo-charge = electronegativity relative to the molecule's mean, so the charges sum
    # to zero like a neutral molecule's. Then Σ qᵢ(Rᵢ + t) = Σ qᵢRᵢ + (Σ qᵢ)t = Σ qᵢRᵢ: the
    # dipole no longer depends on the origin. The sign convention cannot matter for a norm.
    q = chi - chi.mean()
    return float(np.linalg.norm(q @ R))


def geometry(R: np.ndarray) -> np.ndarray:
    """[radius of gyration (Å), gyration-tensor eigenvalues (Å², descending), largest
    interatomic distance (Å)].

    The eigenvalues are kept as moments (Å²), not square-rooted: a planar molecule has a
    zero eigenvalue that floating point returns as ~1e-16, and √ would amplify that noise to
    ~1e-8, enough to break rotation invariance.
    """
    R = np.asarray(R, dtype=np.float64)
    X = R - R.mean(axis=0)
    moments = np.clip(np.linalg.eigvalsh(X.T @ X / len(X))[::-1], 0.0, None)
    i, j = np.triu_indices(len(R), k=1)
    d_max = np.linalg.norm(R[i] - R[j], axis=1).max() if len(i) else 0.0
    return np.array([np.sqrt((X**2).sum(axis=1).mean()), *moments, d_max])


def heteroatom_offset(Z: np.ndarray, R: np.ndarray) -> float:
    """Distance (Å) between the centroid of the N, O and F atoms and the molecular centroid
    (0 if there are none)."""
    R = np.asarray(R, dtype=np.float64)
    het = np.isin(Z, _HETEROATOMS)
    return float(np.linalg.norm(R[het].mean(axis=0) - R.mean(axis=0))) if het.any() else 0.0


def bond_order_counts(Z: np.ndarray, R: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Number of bonds matching each BOND_ORDER_RULES entry (by element pair and length)."""
    out = np.zeros(len(BOND_ORDER_RULES))
    R = np.asarray(R, dtype=np.float64)
    for i, j in pairs:
        pair = tuple(sorted((int(Z[i]), int(Z[j]))))
        d = np.linalg.norm(R[i] - R[j])
        for k, (_, rule_pair, lo, hi) in enumerate(BOND_ORDER_RULES):
            if pair == rule_pair and lo <= d < hi:
                out[k] += 1
    return out


def n_rings(n_atoms: int, pairs: np.ndarray) -> int:
    """Independent rings: the cyclomatic number bonds − atoms + connected components."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_atoms, n_atoms))
    n_parts, _ = connected_components(graph, directed=False)
    return int(len(pairs) - n_atoms + n_parts)


def inertia_moments(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Principal moments of inertia (amu·Å², ascending) about the centre of mass.

    Eigenvalues are used directly (no square root), so a zero moment, as for a linear
    molecule, stays at floating-point zero under rotation.
    """
    m = _lookup(ISOTOPE_MASS, Z)
    R = np.asarray(R, dtype=np.float64)
    X = R - (m[:, None] * R).sum(axis=0) / m.sum()
    second = np.einsum("i,ij,ik->jk", m, X, X)
    tensor = np.trace(second) * np.eye(3) - second
    return np.clip(np.linalg.eigvalsh(tensor), 0.0, None)


def engineered(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """All engineered features, in ENGINEERED_NAMES order."""
    pairs = bonds(Z, R)
    polarity, dipole = bond_dipoles(Z, R, pairs)
    return np.concatenate([
        geometry(R),
        [heteroatom_offset(Z, R), polarity, dipole, electronegativity_dipole(Z, R)],
        bond_counts(Z, pairs),
        bond_order_counts(Z, R, pairs),
        [n_rings(len(Z), pairs)],
        inertia_moments(Z, R),
    ])


# --------------------------------------------------------------------------------------
# Exploration-only features: they read DFT outputs (Mulliken charges, vibrational
# frequencies) from the same calculation as μ, so they may only appear in results that are
# labeled with the inputs used (headline models use only Z and R). Not descriptors.
# --------------------------------------------------------------------------------------

#: 1 e·Å in debye.
E_ANGSTROM_TO_DEBYE = 4.80320

CHARGE_FEATURE_NAMES: tuple[str, ...] = ("charge_dipole_D", "max_abs_charge", "charge_separation")
FREQUENCY_FEATURE_NAMES: tuple[str, ...] = (
    "freq_lowest", "freq_highest", "freq_mean", "n_modes_above_2800", "n_modes_below_500",
)


def charge_features(q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """[|Σ qᵢ Rᵢ| in debye (the point-charge dipole), max |qᵢ| (e), Σ|qᵢ|/2 (e)].

    Mulliken charges of a neutral molecule sum to ~0 (|Σq| ≤ 6e-6 in QM9), so the
    point-charge dipole does not depend on the origin.
    """
    q = np.asarray(q, dtype=np.float64)
    dipole = np.linalg.norm(q @ np.asarray(R, dtype=np.float64)) * E_ANGSTROM_TO_DEBYE
    return np.array([dipole, np.abs(q).max(), np.abs(q).sum() / 2])


def frequency_features(freqs: np.ndarray) -> np.ndarray:
    """Summary of a cleaned harmonic spectrum (cm⁻¹): lowest, highest and mean mode, X–H
    stretches (above 2800 cm⁻¹) and soft modes (below 500 cm⁻¹)."""
    f = np.asarray(freqs, dtype=np.float64)
    return np.array([f.min(), f.max(), f.mean(), (f > 2800).sum(), (f < 500).sum()])
