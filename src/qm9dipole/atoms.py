"""Per-atom chemical environments from Z and R, and their molecule-level aggregates
(exploration X2, Track A).

A dipole is a vector sum, μ = Σᵢ qᵢ rᵢ, so the natural unit of a dipole model is the atom:
predict each atom's charge from its surroundings and let geometry add the contributions up
(`models/charge.py`). This module describes each atom's surroundings. Everything is a function
of (Z, R) plus tabulated constants (covalent radii, Pauling electronegativities, the QEq
parameters below), never another QM9 property, so every feature is headline-legal.

Symmetry: each per-atom feature row is invariant to rotation and translation, and relabeling
the atoms permutes the rows (tests/test_atoms.py). Molecule-level aggregates (sums over atoms)
are therefore invariant to all three, and are registered in `descriptors.MODEL_DESCRIPTORS`.

Per-atom features (ATOM_FEATURE_NAMES), by group:
- identity: one-hot element (C, H, N, O, F);
- bond graph (bonds as in `features.bonds`): degree; bonded neighbours by element; multiple
  bonds by type, read from bond lengths (`features.BOND_ORDER_RULES`); atoms two and three
  bonds away, by element; the smallest ring through the atom (0 if none);
- local geometry: pyramidal height of 3-coordinated atoms (planar amide/aromatic N ≈ 0,
  amine N ≈ 0.35 Å); |Σ û| over bonds (how one-sided the first shell is); |Σ Δχ·û| (the
  local bond-dipole sum); mean and smallest bond-angle cosines;
- radial environment: Behler-style radial symmetry functions, Σⱼ exp(−(rᵢⱼ − μₖ)²/2σ²)·f_c(rᵢⱼ)
  per neighbour element and centre μₖ, with a cosine cutoff f_c at 5 Å;
- directional environment: |Σⱼ w_c(rᵢⱼ) ûᵢⱼ| per neighbour element and radial shell c, the
  norm of a local "vector moment". Directional information is what sorted eigenvalues and
  counts discard, and what decides whether bond dipoles add or cancel;
- electrostatics: the atom's charge from charge equilibration (QEq), the potential the other
  QEq charges put on it, and Σⱼ Zⱼ/rᵢⱼ;
- charged groups: ammonium-like N (4 bonds), the number of terminal O neighbours (two on one
  C mark a carboxylate, two on one N a nitro-like group), and the distance from the centroid.

Molecule-level blocks: `groups` (atom types, ring atoms, charged groups, zwitterion flag),
`qeq` (QEq dipole and charge spread) and `rdf` (a radial distribution function per element
pair, Σ of the radial features). Units: Å, e, eV, debye.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import shortest_path

from qm9dipole.features import BOND_ORDER_RULES, BOND_SCALE, COVALENT_RADII, ELECTRONEGATIVITY

ELEMENTS: tuple[str, ...] = ("C", "H", "N", "O", "F")  # composition order (descriptors)
_ELEMENT_Z = np.array([6, 1, 7, 8, 9])
_INDEX = np.full(10, -1)
_INDEX[_ELEMENT_Z] = np.arange(len(_ELEMENT_Z))
_RADIUS = np.array([COVALENT_RADII[z] for z in _ELEMENT_Z])
_CHI = np.array([ELECTRONEGATIVITY[z] for z in _ELEMENT_Z])

#: QEq parameters (Rappé & Goddard, J. Phys. Chem. 95, 3358 (1991)), in ELEMENTS order:
#: electronegativity χ and idempotential J, eV.
QEQ_CHI = np.array([5.343, 4.528, 6.899, 8.741, 10.874])
QEQ_J = np.array([10.126, 13.890, 11.760, 13.364, 14.948])
#: e²/(4πε₀) in eV·Å.
COULOMB_EV_A = 14.399645
#: 1 e·Å in debye.
E_ANGSTROM_TO_DEBYE = 4.80320

CUTOFF = 5.0  # Å, cosine cutoff of the environment functions
RADIAL_CENTERS = np.linspace(0.9, 4.7, 8)  # Å
RADIAL_WIDTH = 0.4  # Å
MOMENT_CENTERS = np.array([1.25, 2.5, 3.75])  # Å: bonded, geminal, farther shells
MOMENT_WIDTH = 0.5  # Å
PLANAR_N_HEIGHT = 0.15  # Å: a 3-coordinated N below this is planar (amide, aromatic)

_ORDER_NAMES = [name for name, *_ in BOND_ORDER_RULES]
ATOM_FEATURE_NAMES: tuple[str, ...] = (
    *(f"is_{e}" for e in ELEMENTS), "degree", *(f"bonded_{e}" for e in ELEMENTS),
    *(f"bond_{name}" for name in _ORDER_NAMES),
    *(f"shell2_{e}" for e in ELEMENTS), *(f"shell3_{e}" for e in ELEMENTS), "smallest_ring",
    "pyramid_height", "bond_vector_asymmetry", "local_bond_dipole", "angle_cos_mean", "angle_cos_min",
    *(f"radial_{e}_{c:.2f}" for e in ELEMENTS for c in RADIAL_CENTERS),
    *(f"moment_{e}_{c:.2f}" for e in ELEMENTS for c in MOMENT_CENTERS),
    "qeq_charge", "qeq_potential", "nuclear_potential",
    "ammonium_N", "terminal_O_neighbours", "centroid_distance",
)

_PAIRS = [(a, b) for a in range(len(ELEMENTS)) for b in range(a, len(ELEMENTS))]
GROUP_NAMES: tuple[str, ...] = (
    *(f"atoms_C{d}" for d in (1, 2, 3, 4)), *(f"atoms_N{d}" for d in (1, 2, 3, 4)), "atoms_O1", "atoms_O2",
    *(f"ring{s}_atoms" for s in ("3", "4", "5", "6", "7plus")),
    "n_ammonium", "n_carboxylate", "n_nitro_like", "zwitterion", "n_N3_planar", "n_N3_pyramidal",
)
QEQ_NAMES: tuple[str, ...] = ("qeq_dipole_D", "qeq_max_abs_charge", "qeq_charge_separation")
RDF_NAMES: tuple[str, ...] = tuple(f"rdf_{ELEMENTS[a]}{ELEMENTS[b]}_{c:.2f}"
                                   for a, b in _PAIRS for c in RADIAL_CENTERS)


@dataclass
class Environment:
    """Everything `describe` derives from one molecule (n atoms)."""

    element: np.ndarray  # (n,) index into ELEMENTS
    bonds: np.ndarray  # (n, n) bool adjacency
    features: np.ndarray  # (n, len(ATOM_FEATURE_NAMES))
    qeq_charges: np.ndarray  # (n,) e
    groups: np.ndarray  # (len(GROUP_NAMES),)
    qeq: np.ndarray  # (len(QEQ_NAMES),)
    rdf: np.ndarray  # (len(RDF_NAMES),)


def qeq_matrix(element: np.ndarray, D: np.ndarray) -> np.ndarray:
    """QEq hardness matrix (eV): J on the diagonal, shielded Coulomb off it.

    Ohno–Klopman shielding, k/√(r² + a²) with a = k(1/Jᵢ + 1/Jⱼ)/2, joins the bare Coulomb
    law k/r at long range to Jᵢ at r = 0, which keeps the matrix positive definite at bonded
    distances where 1/r alone would not be.
    """
    J = QEQ_J[element]
    a = 0.5 * COULOMB_EV_A * (1.0 / J[:, None] + 1.0 / J[None, :])
    return COULOMB_EV_A / np.sqrt(D**2 + a**2)


def qeq_charges(element: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Charge-equilibration charges (e) of a neutral molecule: minimize
    Σ χᵢqᵢ + ½ Σᵢⱼ Hᵢⱼqᵢqⱼ subject to Σ qᵢ = 0 (one linear solve)."""
    n = len(element)
    if n == 1:
        return np.zeros(1)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = qeq_matrix(element, D)
    M[:n, n] = -1.0  # Lagrange multiplier: the equalized electronegativity
    M[n, :n] = 1.0
    rhs = np.concatenate([-QEQ_CHI[element], [0.0]])
    return np.linalg.solve(M, rhs)[:n]


def smallest_rings(bonds: np.ndarray, heavy: np.ndarray) -> np.ndarray:
    """Size of the smallest ring through each atom (0 if none).

    Rings run through heavy atoms only (H and F have one bond). The smallest ring through a
    bond (i, j) is 1 + the shortest i → j path that avoids that bond; the smallest ring
    through an atom is the minimum over its bonds.
    """
    n = len(bonds)
    adj = [[j for j in np.flatnonzero(bonds[i]) if heavy[j]] if heavy[i] else [] for i in range(n)]
    best = np.zeros(n, dtype=int)
    for i in range(n):
        for j in adj[i]:
            if j < i or len(adj[i]) < 2 or len(adj[j]) < 2:
                continue
            dist, queue = {i: 0}, deque([i])
            while queue and j not in dist:
                a = queue.popleft()
                for b in adj[a]:
                    if b not in dist and not (a == i and b == j):
                        dist[b] = dist[a] + 1
                        queue.append(b)
            if j in dist:
                size = dist[j] + 1
                for k in (i, j):
                    best[k] = size if best[k] == 0 else min(best[k], size)
    return best


def describe(Z: np.ndarray, R: np.ndarray) -> Environment:
    """Per-atom features and molecule-level blocks of one molecule (Z, R in Å)."""
    Z = np.asarray(Z, dtype=int)
    R = np.asarray(R, dtype=np.float64)
    n = len(Z)
    e = _INDEX[Z]
    if (e < 0).any():
        raise ValueError(f"unsupported elements {sorted(set(Z[e < 0]))}")
    E = np.eye(len(ELEMENTS))[e]

    diff = R[None, :, :] - R[:, None, :]  # diff[i, j] = R_j − R_i
    D = np.sqrt((diff**2).sum(-1))
    off = ~np.eye(n, dtype=bool)
    U = np.where(off[..., None], diff / np.where(off, D, 1.0)[..., None], 0.0)  # unit i → j
    r = _RADIUS[e]
    B = off & (D < BOND_SCALE * (r[:, None] + r[None, :]))
    Bf = B.astype(np.float64)
    deg = Bf.sum(1)

    # Bond graph.
    orders = np.column_stack([
        (B & (((Z[:, None] == za) & (Z[None, :] == zb)) | ((Z[:, None] == zb) & (Z[None, :] == za)))
         & (D >= lo) & (D < hi)).sum(1)
        for _, (za, zb), lo, hi in BOND_ORDER_RULES
    ])
    G = shortest_path(Bf, unweighted=True, directed=False)
    shell2, shell3 = (G == 2).astype(np.float64) @ E, (G == 3).astype(np.float64) @ E
    ring = smallest_rings(B, Z > 1)

    # Local geometry.
    pyramid = np.zeros(n)
    for i in np.flatnonzero(deg == 3):
        p = R[B[i]]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(normal)
        if length > 1e-12:
            pyramid[i] = abs(normal @ (R[i] - p[0])) / length
    chi = _CHI[e]
    asymmetry = np.linalg.norm(np.einsum("ij,ijx->ix", Bf, U), axis=1)
    local_dipole = np.linalg.norm(np.einsum("ij,ijx->ix", (chi[None, :] - chi[:, None]) * Bf, U), axis=1)
    cos = np.einsum("ijx,ikx->ijk", U, U)
    pairs = B[:, :, None] & B[:, None, :] & np.triu(np.ones((n, n), dtype=bool), 1)[None]
    n_pairs = pairs.sum((1, 2))
    has = n_pairs > 0
    cos_mean = np.where(has, (cos * pairs).sum((1, 2)) / np.maximum(n_pairs, 1), 0.0)
    cos_min = np.where(has, np.where(pairs, cos, np.inf).min((1, 2), initial=np.inf), 0.0)

    # Radial and directional environment.
    fc = np.where(off & (D < CUTOFF), 0.5 * (np.cos(np.pi * D / CUTOFF) + 1.0), 0.0)
    W = np.exp(-((D[..., None] - RADIAL_CENTERS) ** 2) / (2 * RADIAL_WIDTH**2)) * fc[..., None]
    radial = np.einsum("ijk,je->iek", W, E)  # (n, element, centre)
    Wm = np.exp(-((D[..., None] - MOMENT_CENTERS) ** 2) / (2 * MOMENT_WIDTH**2)) * fc[..., None]
    moments = np.linalg.norm(np.einsum("ijc,je,ijx->iecx", Wm, E, U, optimize=True), axis=-1)

    # Electrostatics.
    q = qeq_charges(e, D)
    H = qeq_matrix(e, D)
    potential = np.where(off, H, 0.0) @ q
    nuclear = np.where(off, Z[None, :] / np.where(off, D, 1.0), 0.0).sum(1)

    # Charged groups.
    terminal_O = (Z == 8) & (deg == 1)
    terminal_O_nb = Bf @ terminal_O
    ammonium = (Z == 7) & (deg == 4)
    centroid = np.linalg.norm(R - R.mean(axis=0), axis=1)

    features = np.column_stack([
        E, deg, Bf @ E, orders, shell2, shell3, ring,
        pyramid, asymmetry, local_dipole, cos_mean, cos_min,
        radial.reshape(n, -1), moments.reshape(n, -1),
        q, potential, nuclear, ammonium, terminal_O_nb, centroid,
    ])

    # Molecule-level blocks.
    def count(mask) -> float:
        return float(np.count_nonzero(mask))

    carboxylate = (Z == 6) & (deg == 3) & (terminal_O_nb >= 2)
    nitro = (Z == 7) & (terminal_O_nb >= 2)
    n3 = (Z == 7) & (deg == 3)
    groups = np.array([
        *(count((Z == 6) & (deg == d)) for d in (1, 2, 3, 4)),
        *(count((Z == 7) & (deg == d)) for d in (1, 2, 3, 4)),
        count((Z == 8) & (deg == 1)), count((Z == 8) & (deg == 2)),
        count(ring == 3), count(ring == 4), count(ring == 5), count(ring == 6), count(ring >= 7),
        count(ammonium), count(carboxylate), count(nitro),
        float(ammonium.any() and carboxylate.any()),
        count(n3 & (pyramid < PLANAR_N_HEIGHT)), count(n3 & (pyramid >= PLANAR_N_HEIGHT)),
    ])
    qeq = np.array([np.linalg.norm(q @ R) * E_ANGSTROM_TO_DEBYE, np.abs(q).max(), np.abs(q).sum() / 2])
    by_element = np.einsum("ie,ibk->ebk", E, radial)  # Σ over centre atoms of element a
    rdf = np.concatenate([by_element[a, b] for a, b in _PAIRS])
    return Environment(element=e, bonds=B, features=features, qeq_charges=q, groups=groups, qeq=qeq, rdf=rdf)


# Descriptor functions (Z, R) -> vector, for `descriptors.DESCRIPTORS` and the invariance tests.

def groups(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Atom types, ring atoms and charged groups (GROUP_NAMES)."""
    return describe(Z, R).groups


def qeq(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """QEq dipole (D), largest |charge| and charge separation (e) (QEQ_NAMES)."""
    return describe(Z, R).qeq


def rdf(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Radial distribution per element pair (RDF_NAMES)."""
    return describe(Z, R).rdf


# --------------------------------------------------------------------------------------
# Many molecules: a ragged atom table for the latent-charge model, built in parallel.
# --------------------------------------------------------------------------------------

@dataclass
class AtomData:
    """Atoms of many molecules, stored molecule by molecule (molecule k owns atom rows
    offsets[k]:offsets[k+1] and edge rows edge_offsets[k]:edge_offsets[k+1]).

    `pos` are coordinates centred on each molecule's unweighted centroid (Å), so Σᵢ posᵢ = 0
    per molecule; `edges` are directed bonded pairs (i, j) as global atom rows, with unit
    vectors i → j in `edge_vectors`.
    """

    ids: np.ndarray  # (m,) QM9 molecule IDs
    offsets: np.ndarray  # (m + 1,)
    X: np.ndarray  # (n_atoms, d) float32 per-atom features
    pos: np.ndarray  # (n_atoms, 3) float32, Å
    element: np.ndarray  # (n_atoms,) int8, index into ELEMENTS
    edge_offsets: np.ndarray  # (m + 1,)
    edges: np.ndarray  # (n_edges, 2) int64
    edge_vectors: np.ndarray  # (n_edges, 3) float32
    feature_names: tuple[str, ...] = ATOM_FEATURE_NAMES

    @property
    def n_molecules(self) -> int:
        return len(self.ids)

    @property
    def n_atoms(self) -> np.ndarray:
        return np.diff(self.offsets)

    def rows(self, ids: np.ndarray) -> np.ndarray:
        """Molecule positions of `ids` (raises KeyError for unknown IDs)."""
        lookup = getattr(self, "_lookup", None)
        if lookup is None:
            lookup = self._lookup = {int(i): k for k, i in enumerate(self.ids)}
        return np.array([lookup[int(i)] for i in ids], dtype=np.int64)

    def subset(self, ids: np.ndarray) -> "AtomData":
        """The molecules `ids`, in that order, as a new contiguous AtomData."""
        k = self.rows(ids)
        atom_idx, counts = _ranges(self.offsets, k)
        edge_idx, ecounts = _ranges(self.edge_offsets, k)
        new_offsets = np.concatenate([[0], np.cumsum(counts)])
        shift = np.repeat(new_offsets[:-1] - self.offsets[k], ecounts)
        return AtomData(
            ids=np.asarray(ids).copy(), offsets=new_offsets, X=self.X[atom_idx], pos=self.pos[atom_idx],
            element=self.element[atom_idx], edge_offsets=np.concatenate([[0], np.cumsum(ecounts)]),
            edges=self.edges[edge_idx] + shift[:, None], edge_vectors=self.edge_vectors[edge_idx],
            feature_names=self.feature_names,
        )

    def select_features(self, names: list[str] | tuple[str, ...]) -> "AtomData":
        """The same atoms with only the feature columns `names`, in that order."""
        cols = [self.feature_names.index(c) for c in names]
        return AtomData(self.ids, self.offsets, self.X[:, cols], self.pos, self.element, self.edge_offsets,
                        self.edges, self.edge_vectors, tuple(names))


def _ranges(offsets: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated row ranges offsets[k]:offsets[k+1], and their lengths."""
    starts, counts = offsets[k], offsets[k + 1] - offsets[k]
    if counts.sum() == 0:
        return np.zeros(0, dtype=np.int64), counts
    begin = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return np.repeat(starts - begin, counts) + np.arange(counts.sum()), counts


def _describe_chunk(Zs, Rs, dtype=np.float32):
    out = []
    for Z, R in zip(Zs, Rs):
        env = describe(Z, R)
        R = np.asarray(R, dtype=np.float64)
        i, j = np.nonzero(env.bonds)
        v = R[j] - R[i]
        out.append((env.features.astype(dtype), (R - R.mean(axis=0)).astype(dtype),
                    env.element.astype(np.int8), np.column_stack([i, j]),
                    (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(dtype),
                    np.concatenate([env.groups, env.qeq, env.rdf])))
    return out


def build_atom_data(table, n_jobs: int = -1, chunk: int = 2000, dtype=np.float32):
    """(AtomData, molecule-level blocks) for every row of `table` (columns id, Z, R), in row
    order. The blocks are a DataFrame indexed by ID with GROUP_NAMES + QEQ_NAMES + RDF_NAMES.
    About 1.7 ms per molecule per process. Per-atom arrays are stored as `dtype` (float32
    halves the memory; float64 for exact symmetry tests)."""
    import pandas as pd
    from joblib import Parallel, delayed

    Zs, Rs = table["Z"].to_list(), table["R"].to_list()
    parts = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_describe_chunk)(Zs[s:s + chunk], Rs[s:s + chunk], dtype) for s in range(0, len(Zs), chunk))
    mols = [m for part in parts for m in part]
    counts = np.array([len(m[0]) for m in mols])
    ecounts = np.array([len(m[3]) for m in mols])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    edge_offsets = np.concatenate([[0], np.cumsum(ecounts)])
    edges = np.concatenate([m[3] for m in mols]) + np.repeat(offsets[:-1], ecounts)[:, None]
    data = AtomData(
        ids=table["id"].to_numpy().copy(), offsets=offsets, X=np.concatenate([m[0] for m in mols]),
        pos=np.concatenate([m[1] for m in mols]), element=np.concatenate([m[2] for m in mols]),
        edge_offsets=edge_offsets, edges=edges.astype(np.int64),
        edge_vectors=np.concatenate([m[4] for m in mols]),
    )
    blocks = pd.DataFrame(np.stack([m[5] for m in mols]), columns=[*GROUP_NAMES, *QEQ_NAMES, *RDF_NAMES],
                          index=pd.Index(data.ids, name="id"))
    return data, blocks
