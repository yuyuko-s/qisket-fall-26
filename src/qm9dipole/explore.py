"""Plumbing for the exploration chain (notebooks/explore_0*): where each step writes its
output, how the next step loads it, and the feature sets every model notebook uses.

    01 EDA ─▶ results/explore01_recommendations.csv
    02 cleaning + features ─▶ data/processed/explore_features.parquet (regenerated, gitignored)
                              results/explore02_feature_catalog.csv (committed)
    03 scaling + dimension ─▶ results/explore03_preprocessing.json (committed)
    04 models ─▶ results/explore04_classical_cv.csv
    05 generalization ─▶ results/explore05_*.csv

Keeping the definitions here means a change upstream (a dropped feature, a new scaling
choice) reaches every later notebook without editing them.
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from qm9dipole import REPO_ROOT
from qm9dipole.data import PROCESSED_DIR
from qm9dipole.provenance import RESULTS_DIR

CONFIG_PATH = REPO_ROOT / "configs" / "explore.yaml"
FEATURES_PATH = PROCESSED_DIR / "explore_features.parquet"
CATALOG_NAME = "explore02_feature_catalog"
PREPROCESSING_PATH = RESULTS_DIR / "explore03_preprocessing.json"

#: Feature blocks, as written by explore_02. Legal blocks are functions of Z and R only.
LEGAL_BLOCKS: tuple[str, ...] = ("composition", "cm", "engineered")
DFT_BLOCKS: tuple[str, ...] = ("dft_property", "charge", "frequency")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_features() -> pd.DataFrame:
    """The cleaned feature table (one row per training-pool molecule, indexed by id), with
    the target `mu` and bookkeeping columns `formula`, `n_heavy` and `pool_analysis`."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH.name} not found: run notebooks/explore_02_cleaning_and_features.ipynb")
    return pd.read_parquet(FEATURES_PATH)


def load_catalog() -> pd.DataFrame:
    """One row per feature: block, legal (bool), kept (bool), reason, description."""
    path = RESULTS_DIR / f"{CATALOG_NAME}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run notebooks/explore_02_cleaning_and_features.ipynb")
    return pd.read_csv(path)


def feature_sets(catalog: pd.DataFrame) -> dict[str, list[str]]:
    """Named feature sets built from the kept features of the catalog.

    Headline-legal (functions of Z and R): composition; cm (Coulomb spectrum); structure
    (composition + engineered); cm+structure. Exploration-only, labeled as such in every
    result: structure+dft (structure plus every kept DFT-derived feature).
    """
    kept = catalog[catalog["kept"]]
    block = {b: kept.loc[kept["block"] == b, "feature"].tolist() for b in (*LEGAL_BLOCKS, *DFT_BLOCKS)}
    structure = block["composition"] + block["engineered"]
    dft = block["dft_property"] + block["charge"] + block["frequency"]
    return {
        "composition": block["composition"],
        "cm": block["cm"],
        "structure": structure,
        "cm+structure": block["cm"] + structure,
        "structure+dft": structure + dft,
    }


#: Feature sets that use DFT outputs; any result with them is exploration-only.
EXPLORATION_ONLY_SETS: tuple[str, ...] = ("structure+dft",)


def load_preprocessing() -> dict:
    """The scaling and target transform chosen by explore_03 (and its evidence)."""
    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(f"{PREPROCESSING_PATH.name} not found: run notebooks/explore_03_standardization_and_dimension.ipynb")
    return json.loads(PREPROCESSING_PATH.read_text())
