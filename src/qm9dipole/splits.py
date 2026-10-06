"""Working pool, formula holdout, anchors and nested training sets (PLAN §3.5, §4).

All randomness comes from the seeds in configs/splits.yaml:
- the split seed fixes the working pool and both test sets. The three draws (pool, unseen,
  familiar) use independent streams spawned from one SeedSequence, so changing one step's
  draws never shifts another's;
- each training seed s fixes that seed's anchors and fill order.

Splits are saved as QM9 molecule IDs (CLAUDE.md rule 7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qm9dipole import REPO_ROOT
from qm9dipole.provenance import config_hash, git_hash

SPLITS_DIR = REPO_ROOT / "splits"
SPLITS_CONFIG = REPO_ROOT / "configs" / "splits.yaml"


class SplitCheckError(AssertionError):
    """A split invariant is violated. The message starts with its PLAN §4 label, e.g. "(c)"."""


@dataclass
class Splits:
    config: dict
    pool: np.ndarray  # working-pool IDs (PLAN §3.5)
    unseen_formulas: list[str]
    familiar_formulas: list[str]
    test_unseen: np.ndarray
    test_familiar: np.ndarray
    anchors: dict[int, np.ndarray]  # training seed -> anchor IDs, one per familiar formula
    train: dict[int, dict[int, np.ndarray]]  # training seed -> N -> sorted IDs

    def all_id_arrays(self) -> dict[str, np.ndarray]:
        arrays = {"pool": self.pool, "test_unseen": self.test_unseen,
                  "test_familiar": self.test_familiar}
        for seed, by_n in self.train.items():
            arrays |= {f"S_({seed},{n})": ids for n, ids in by_n.items()}
        return arrays


def load_config(path: Path = SPLITS_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def working_pool(table: pd.DataFrame, max_per_formula: int, rng: np.random.Generator) -> np.ndarray:
    """IDs of at most `max_per_formula` molecules per formula, sampled uniformly within each."""
    keep = []
    for _, ids in table.sort_values("id").groupby("formula", sort=True)["id"]:
        ids = ids.to_numpy()
        if len(ids) > max_per_formula:
            ids = rng.choice(ids, max_per_formula, replace=False)
        keep.append(ids)
    return np.sort(np.concatenate(keep))


def choose_unseen_formulas(n_heavy: pd.Series, frac: float, rng: np.random.Generator) -> list[str]:
    """Stratified draw: round(frac × count) formulas from each heavy-atom-count stratum.

    `n_heavy` maps formula -> heavy-atom count. Strata too small to round up to one formula
    contribute none.
    """
    chosen: list[str] = []
    n_heavy = n_heavy.sort_index()
    for _, stratum in n_heavy.groupby(n_heavy):
        k = int(np.floor(frac * len(stratum) + 0.5))
        chosen += rng.choice(stratum.index.to_numpy(), k, replace=False).tolist()
    return sorted(chosen)


def build_splits(table: pd.DataFrame, cfg: dict) -> Splits:
    """Build every split from the post-exclusion table (columns id, formula, n_heavy).

    Procedure (PLAN §4): pool -> unseen formulas and their test molecules -> familiar
    formulas and their test molecules -> training pool P -> per seed, anchors plus a random
    fill order, cut at each N. Training sets are nested by construction.
    """
    sizes = sorted(cfg["train_sizes"])
    n_fam, per_fam = cfg["n_familiar_formulas"], cfg["familiar_per_formula"]
    if n_fam >= sizes[0]:
        raise ValueError(f"n_familiar_formulas ({n_fam}) must be < smallest N ({sizes[0]})")
    if cfg["familiar_min_pool"] <= per_fam:
        raise ValueError("familiar_min_pool must leave at least one molecule for the anchor")

    rng_pool, rng_unseen, rng_familiar = (
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg["split_seed"]).spawn(3)
    )

    pool_ids = working_pool(table, cfg["max_per_formula"], rng_pool)
    pool = table[table["id"].isin(pool_ids)].sort_values("id")
    by_formula = {f: ids.to_numpy() for f, ids in pool.groupby("formula", sort=True)["id"]}

    # 1. Unseen formulas, and up to unseen_per_formula test molecules from each.
    unseen = choose_unseen_formulas(
        pool.groupby("formula")["n_heavy"].first(), cfg["unseen_formula_frac"], rng_unseen
    )
    k = cfg["unseen_per_formula"]
    test_unseen = np.concatenate(
        [rng_unseen.choice(by_formula[f], min(k, len(by_formula[f])), replace=False) for f in unseen]
    )
    if len(test_unseen) > cfg["unseen_max_total"]:
        test_unseen = rng_unseen.choice(test_unseen, cfg["unseen_max_total"], replace=False)

    # 2. Familiar formulas: not unseen, with enough pool molecules for the test set + an anchor.
    unseen_set = set(unseen)
    eligible = [f for f, ids in by_formula.items()
                if f not in unseen_set and len(ids) >= cfg["familiar_min_pool"]]
    if len(eligible) < n_fam:
        raise ValueError(f"only {len(eligible)} formulas eligible as familiar; need {n_fam}")
    familiar = sorted(rng_familiar.choice(eligible, n_fam, replace=False).tolist())
    test_familiar = np.concatenate(
        [rng_familiar.choice(by_formula[f], per_fam, replace=False) for f in familiar]
    )

    # 3. Training pool P: no unseen formula, no test molecule.
    train_pool = pool[~pool["formula"].isin(unseen_set) & ~pool["id"].isin(test_familiar)]
    P = train_pool["id"].to_numpy()
    if len(P) < sizes[-1]:
        raise ValueError(f"training pool has {len(P)} molecules; largest N is {sizes[-1]}")

    # 4. Per training seed: anchors A_s, fill order π_s, S_{s,N} = A_s ∪ π_s[:N − |A_s|].
    anchors, train = {}, {}
    for s in cfg["train_seeds"]:
        rng = np.random.default_rng(np.random.SeedSequence([cfg["split_seed"], s]))
        a = np.array([rng.choice(np.setdiff1d(by_formula[f], test_familiar)) for f in familiar])
        fill = rng.permutation(np.setdiff1d(P, a))
        anchors[s] = np.sort(a)
        train[s] = {n: np.sort(np.concatenate([a, fill[: n - len(a)]])) for n in sizes}

    return Splits(
        config=dict(cfg), pool=pool_ids, unseen_formulas=unseen, familiar_formulas=familiar,
        test_unseen=np.sort(test_unseen), test_familiar=np.sort(test_familiar),
        anchors=anchors, train=train,
    )


def training_pool(s: Splits, table: pd.DataFrame) -> np.ndarray:
    """Sorted IDs of the training pool P: pool molecules whose formula is not unseen and
    which are not in the familiar test set. Every training set is drawn from P, so analyses
    restricted to P never touch test molecules."""
    pool = table[table["id"].isin(s.pool)]
    keep = ~pool["formula"].isin(s.unseen_formulas) & ~pool["id"].isin(s.test_familiar)
    return np.sort(pool.loc[keep, "id"].to_numpy())


def check_splits(s: Splits, table: pd.DataFrame, excluded_ids: set[int] | frozenset[int]) -> None:
    """Raise SplitCheckError unless PLAN §4 asserts (a)–(g) hold, plus (h) test-set formulas.

    `table` must cover every ID in the splits (pass the full parsed table) and supply its
    formula. Checks run in label order, so the first failure is reported.
    """
    def fail(msg: str) -> None:
        raise SplitCheckError(msg)

    formula = table.set_index("id")["formula"]
    arrays = s.all_id_arrays()
    for name, ids in arrays.items():
        if unknown := set(ids) - set(formula.index):
            fail(f"(id) {name} has {len(unknown)} IDs not in the QM9 table")

    sizes = sorted(s.config["train_sizes"])
    tests = set(s.test_unseen) | set(s.test_familiar)
    unseen, familiar = set(s.unseen_formulas), set(s.familiar_formulas)
    trains = [(seed, n, ids) for seed, by_n in s.train.items() for n, ids in by_n.items()]

    # (a) training and test sets are disjoint (and the two test sets from each other)
    if set(s.test_unseen) & set(s.test_familiar):
        fail("(a) the unseen and familiar test sets overlap")
    for seed, n, ids in trains:
        if shared := set(ids) & tests:
            fail(f"(a) S_({seed},{n}) shares {len(shared)} IDs with the test sets")
    # (b) no unseen formula appears in any training set
    for seed, n, ids in trains:
        if leaked := set(formula.loc[ids]) & unseen:
            fail(f"(b) S_({seed},{n}) contains unseen formulas {sorted(leaked)[:5]}")
    # (c) every familiar formula is in S_{s,N1}
    for seed, by_n in s.train.items():
        if missing := familiar - set(formula.loc[by_n[sizes[0]]]):
            fail(f"(c) S_({seed},{sizes[0]}) is missing familiar formulas {sorted(missing)[:5]}")
    # (d) S_{s,N1} ⊂ S_{s,N2} ⊂ S_{s,N3}
    for seed, by_n in s.train.items():
        for small, big in zip(sizes, sizes[1:]):
            if not set(by_n[small]) <= set(by_n[big]):
                fail(f"(d) S_({seed},{small}) is not a subset of S_({seed},{big})")
    # (e) sizes are exact
    if set(s.train) != set(s.config["train_seeds"]):
        fail(f"(e) training seeds {sorted(s.train)} != config {s.config['train_seeds']}")
    for seed, by_n in s.train.items():
        if sorted(by_n) != sizes:
            fail(f"(e) seed {seed} has sizes {sorted(by_n)}, expected {sizes}")
    for seed, n, ids in trains:
        if len(ids) != n:
            fail(f"(e) S_({seed},{n}) has {len(ids)} IDs")
    if len(s.test_familiar) != len(familiar) * s.config["familiar_per_formula"]:
        fail(f"(e) familiar test set has {len(s.test_familiar)} IDs")
    if len(s.test_unseen) > s.config["unseen_max_total"]:
        fail(f"(e) unseen test set has {len(s.test_unseen)} IDs, above the cap")
    # (f) IDs are unique within every set
    for name, ids in arrays.items():
        if len(set(ids)) != len(ids):
            fail(f"(f) {name} contains duplicate IDs")
    # (g) no excluded ID appears anywhere
    for name, ids in arrays.items():
        if bad := set(ids) & set(excluded_ids):
            fail(f"(g) {name} contains {len(bad)} excluded IDs")
    # (h) test sets draw from the right formulas
    if unseen & familiar:
        fail("(h) a formula is both unseen and familiar")
    if not set(formula.loc[s.test_unseen]) <= unseen:
        fail("(h) the unseen test set contains a non-unseen formula")
    if set(formula.loc[s.test_familiar]) != familiar:
        fail("(h) familiar test formulas differ from the familiar formula list")


# --------------------------------------------------------------------------------------
# Saving and loading. One JSON file per set, each stamped with seeds and hashes.
# --------------------------------------------------------------------------------------

def _ids(a: np.ndarray) -> list[int]:
    return [int(x) for x in a]


def save_splits(s: Splits, out_dir: Path = SPLITS_DIR, extra_meta: dict | None = None) -> list[Path]:
    """Write meta.json, pool.json, test_*.json and train_s{seed}_n{N}.json.

    Existing train_s*_n*.json files are removed first, so sizes or seeds dropped from the
    config cannot leave stale files behind.
    """
    # Stamp before touching any file: deleting committed split files would make the
    # working tree dirty, so git_hash() would always report "-dirty".
    stamp = {"split_seed": s.config["split_seed"], "config_hash": config_hash(s.config),
             "git_hash": git_hash()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("train_s*_n*.json"):
        old.unlink()
    written = []

    def write(name: str, payload: dict) -> None:
        path = out_dir / name
        path.write_text(json.dumps({**stamp, **payload}, indent=1) + "\n")
        written.append(path)

    write("meta.json", {
        "config": s.config,
        "unseen_formulas": s.unseen_formulas,
        "familiar_formulas": s.familiar_formulas,
        "counts": {"pool": len(s.pool), "test_unseen": len(s.test_unseen),
                   "test_familiar": len(s.test_familiar)},
        **(extra_meta or {}),
    })
    write("pool.json", {"kind": "pool", "ids": _ids(s.pool)})
    write("test_unseen.json", {"kind": "test_unseen", "ids": _ids(s.test_unseen)})
    write("test_familiar.json", {"kind": "test_familiar", "ids": _ids(s.test_familiar)})
    for seed, by_n in s.train.items():
        for n, ids in by_n.items():
            write(f"train_s{seed}_n{n}.json", {"kind": "train", "seed": seed, "n": n,
                                               "anchor_ids": _ids(s.anchors[seed]), "ids": _ids(ids)})
    return written


def load_splits(split_dir: Path = SPLITS_DIR) -> Splits:
    """Load splits saved by save_splits."""
    def read(name: str) -> dict:
        return json.loads((split_dir / name).read_text())

    meta = read("meta.json")
    cfg = meta["config"]
    anchors, train = {}, {}
    for seed in cfg["train_seeds"]:
        train[seed] = {}
        for n in sorted(cfg["train_sizes"]):
            d = read(f"train_s{seed}_n{n}.json")
            train[seed][n] = np.array(d["ids"])
            anchors[seed] = np.array(d["anchor_ids"])
    return Splits(
        config=cfg, pool=np.array(read("pool.json")["ids"]),
        unseen_formulas=meta["unseen_formulas"], familiar_formulas=meta["familiar_formulas"],
        test_unseen=np.array(read("test_unseen.json")["ids"]),
        test_familiar=np.array(read("test_familiar.json")["ids"]),
        anchors=anchors, train=train,
    )
