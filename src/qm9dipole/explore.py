"""Plumbing for the exploration chain (notebooks/explore_0*): where each step writes its
output, how the next step loads it, and the feature sets every model notebook uses.

    01 EDA ─▶ results/explore01_recommendations.csv
    02 cleaning + features ─▶ data/processed/explore_features.parquet (regenerated, gitignored)
                              results/explore02_feature_catalog.csv (committed)
    03 scaling + dimension ─▶ results/explore03_preprocessing.json (committed)
    04 models ─▶ results/explore04_*.csv (learning curves, Track A and Track B)
    05 generalization ─▶ results/explore05_*.csv

Exploration X2 (DECISIONS.md, 2026-10-06) works on the **development universe**: the training
pool plus the two development sets (`dev`, `dev_unseen`), never a test set. Per-atom features
are not stored (about 0.7 GB); `load_atoms` recomputes them from Z and R in about a minute and
keeps the columns notebook 02 kept.

Keeping the definitions here means a change upstream (a dropped feature, a new scaling
choice) reaches every later notebook without editing them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from qm9dipole import REPO_ROOT
from qm9dipole.data import PROCESSED_DIR
from qm9dipole.provenance import results_file

CONFIG_PATH = REPO_ROOT / "configs" / "explore.yaml"
FEATURES_PATH = PROCESSED_DIR / "explore_features.parquet"
CATALOG_NAME = "explore02_feature_catalog"
PREPROCESSING_PATH = results_file("explore03_preprocessing.json")

#: Molecule-level feature blocks, as written by explore_02. Legal blocks are functions of Z and R.
LEGAL_BLOCKS: tuple[str, ...] = ("composition", "cm", "engineered", "groups", "qeq", "rdf")
DFT_BLOCKS: tuple[str, ...] = ("dft_property", "charge", "frequency")
#: Per-atom features (atoms.ATOM_FEATURE_NAMES) are catalogued with this block name.
ATOM_BLOCK = "atom"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def development_ids(splits) -> np.ndarray:
    """Sorted IDs of the training pool and both development sets: every molecule the
    exploration chain may read. No test molecule is among them."""
    return np.sort(np.concatenate([splits.pool, splits.dev, splits.dev_unseen]))


def load_features() -> pd.DataFrame:
    """The cleaned molecule-level feature table (one row per development-universe molecule,
    indexed by id), with the target `mu` and bookkeeping columns `formula`, `n_heavy` and
    `subset` (pool, dev or dev_unseen)."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH.name} not found: run notebooks/explore_02_cleaning_and_features.ipynb")
    return pd.read_parquet(FEATURES_PATH)


def load_catalog() -> pd.DataFrame:
    """One row per feature: block, level (molecule or atom), legal, kept, reason, description."""
    path = results_file(f"{CATALOG_NAME}.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run notebooks/explore_02_cleaning_and_features.ipynb")
    return pd.read_csv(path)


def feature_sets(catalog: pd.DataFrame) -> dict[str, list[str]]:
    """Named molecule-level feature sets built from the kept features of the catalog.

    Headline-legal (functions of Z and R):
    - composition: atom counts (the brief's composition-only ablation);
    - cm: the Coulomb-matrix spectrum (PLAN §5);
    - structure: composition + X1's engineered features (X1's best set, for continuity);
    - structure+: structure + charged groups and atom types + the QEq dipole (X2);
    - all_legal: structure+ + the per-element-pair radial distribution + cm (X2's richest).
    Exploration-only, labeled as such in every result: structure+dft (structure plus every
    kept DFT-derived feature).
    """
    kept = catalog[catalog["kept"] & (catalog.get("level", "molecule") == "molecule")]
    block = {b: kept.loc[kept["block"] == b, "feature"].tolist() for b in (*LEGAL_BLOCKS, *DFT_BLOCKS)}
    structure = block["composition"] + block["engineered"]
    structure_plus = structure + block["groups"] + block["qeq"]
    dft = block["dft_property"] + block["charge"] + block["frequency"]
    return {
        "composition": block["composition"],
        "cm": block["cm"],
        "structure": structure,
        "structure+": structure_plus,
        "all_legal": structure_plus + block["rdf"] + block["cm"],
        "structure+dft": structure + dft,
    }


#: Feature sets that use DFT outputs; any result with them is exploration-only.
EXPLORATION_ONLY_SETS: tuple[str, ...] = ("structure+dft",)


def atom_feature_names(catalog: pd.DataFrame) -> list[str]:
    """The per-atom features notebook 02 kept, in atoms.ATOM_FEATURE_NAMES order."""
    from qm9dipole.atoms import ATOM_FEATURE_NAMES

    kept = set(catalog.loc[(catalog["block"] == ATOM_BLOCK) & catalog["kept"], "feature"])
    return [f for f in ATOM_FEATURE_NAMES if f in kept]


def load_atoms(table: pd.DataFrame, ids: np.ndarray, catalog: pd.DataFrame | None = None, n_jobs: int = -1):
    """AtomData (atoms.py) for the molecules `ids`, computed from Z and R, with only the kept
    per-atom feature columns. `table` is the parsed QM9 table (columns id, Z, R)."""
    from qm9dipole.atoms import build_atom_data

    rows = table.set_index("id").loc[ids, ["Z", "R"]].reset_index()
    data, _ = build_atom_data(rows, n_jobs=n_jobs)
    return data.select_features(atom_feature_names(catalog if catalog is not None else load_catalog()))


def load_preprocessing() -> dict:
    """The scaling, target transform and Track B reduction chosen by explore_03 (and evidence)."""
    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(f"{PREPROCESSING_PATH.name} not found: run notebooks/explore_03_standardization_and_dimension.ipynb")
    return json.loads(PREPROCESSING_PATH.read_text())
