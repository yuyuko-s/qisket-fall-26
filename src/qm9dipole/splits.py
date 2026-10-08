"""Formula holdouts, test and development sets, anchors and nested training sets.

Design (exploration X2, 2026-10-06; it replaces an earlier 25-molecules-per-formula pool
and its small test sets). From the kept molecules (exclusions and geometric duplicates
removed):

1. **Unseen-formula test set** `test_unseen`: every molecule of a stratified 15% of formulas
   (by heavy-atom count). The brief's "formulas absent from training".
2. **Development unseen formulas** `dev_unseen`: every molecule of a further stratified draw
   of formulas. They estimate new-formula error during development, so the test holdout is
   scored only by `scripts/final_eval.py`.
3. **Familiar test set** `test_familiar`: a random sample of the remaining molecules, each
   formula keeping at least one molecule in the training pool.
4. **Development set** `dev`: a random sample of what is left, under the same rule. Learning
   curves and diagnostics are scored on it, never on a test set.
5. **Training pool** `pool` (P): everything else.
6. **Quantum subsets** (inference costs n_test × N circuits): `test_unseen_q`, at most a few
   molecules per unseen formula; `test_familiar_q`, a few molecules from each of a few
   familiar formulas, which are **anchored**: each seed puts one molecule of every such
   formula first in its fill order, so they are in every training set.
7. **Nested training sets**: per training seed s, an order of P (anchors, then a random
   permutation of the rest); S_(s,N) is its first N molecules, for every N in the config and,
   optionally, N = |P| (the full pool, the same set for every seed).

All randomness comes from the seeds in configs/splits.yaml. The split seed spawns one
independent stream per draw, so changing one draw never shifts another; stream 1 (unseen
formulas) is the stream X1 used, so the unseen formulas are the same 93 as before. Each
training seed s uses SeedSequence([split_seed, s]), so test sets cannot depend on it.

Splits are saved as QM9 molecule IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qm9dipole import REPO_ROOT
from qm9dipole.provenance import config_hash, git_hash

SPLITS_DIR = REPO_ROOT / "splits"
SPLITS_CONFIG = REPO_ROOT / "configs" / "splits.yaml"

#: Independent random streams spawned from the split seed (index = spawn key). Stream 0 drew
#: X1's 25-per-formula working pool and is retired; stream 1 is kept for the unseen formulas.
STREAMS: tuple[str, ...] = ("retired_working_pool", "unseen", "familiar", "dev_unseen", "dev", "quantum")

#: The molecule sets every Splits object holds, besides the training sets.
SET_NAMES: tuple[str, ...] = (
    "pool", "test_unseen", "test_familiar", "dev", "dev_unseen", "test_unseen_q", "test_familiar_q",
)


class SplitCheckError(AssertionError):
    """A split invariant is violated. The message starts with its label, e.g. "(c)"."""


@dataclass
class Splits:
    config: dict
    unseen_formulas: list[str]  # test holdout
    dev_unseen_formulas: list[str]  # development holdout
    familiar_formulas: list[str]  # anchored formulas of the quantum familiar subset
    pool: np.ndarray  # training pool P
    test_unseen: np.ndarray
    test_familiar: np.ndarray
    dev: np.ndarray
    dev_unseen: np.ndarray
    test_unseen_q: np.ndarray
    test_familiar_q: np.ndarray
    anchors: dict[int, np.ndarray]  # training seed -> anchor IDs, one per familiar formula
    order: dict[int, np.ndarray]  # training seed -> P in fill order (anchors first)
    train: dict[int, dict[int, np.ndarray]] = field(init=False)  # seed -> N -> sorted IDs

    def __post_init__(self) -> None:
        self.train = {s: {n: np.sort(o[:n]) for n in self.sizes} for s, o in self.order.items()}

    @property
    def sizes(self) -> list[int]:
        """Training-set sizes, ascending; the last is |P| if the config asks for the full pool."""
        sizes = sorted(self.config["train_sizes"])
        if self.config.get("train_full_pool") and len(self.pool) > sizes[-1]:
            sizes.append(len(self.pool))
        return sizes

    @property
    def track_b_sizes(self) -> list[int]:
        """The small sizes of the quantum-comparable learning curve (Track B)."""
        return sorted(self.config["track_b_sizes"])

    def sets(self) -> dict[str, np.ndarray]:
        """Every non-training set by name (SET_NAMES)."""
        return {name: getattr(self, name) for name in SET_NAMES}

    def all_id_arrays(self) -> dict[str, np.ndarray]:
        arrays = self.sets()
        for seed, by_n in self.train.items():
            arrays |= {f"S_({seed},{n})": ids for n, ids in by_n.items()}
        return arrays


def load_config(path: Path = SPLITS_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def choose_unseen_formulas(n_heavy: pd.Series, frac: float, rng: np.random.Generator) -> list[str]:
    """Stratified draw: round-half-up(frac × count) formulas from each heavy-atom stratum.

    `n_heavy` maps formula -> heavy-atom count. Strata too small to round up to one formula
    contribute none.
    """
    chosen: list[str] = []
    n_heavy = n_heavy.sort_index()
    for _, stratum in n_heavy.groupby(n_heavy):
        k = _round_half_up(frac * len(stratum))
        chosen += rng.choice(stratum.index.to_numpy(), k, replace=False).tolist()
    return sorted(chosen)


def sample_keeping_one(ids: np.ndarray, formulas: pd.Series, k: int,
                       rng: np.random.Generator) -> np.ndarray:
    """A uniform random sample of `k` of `ids` in which no formula is taken whole.

    Every formula keeps at least one molecule outside the sample, so a sampled molecule's
    formula stays in the training pool ("familiar"). Formulas the first draw took whole give
    one molecule back, and the sample is topped up from the remaining eligible molecules.
    `formulas` maps ID -> formula.
    """
    ids = np.sort(np.asarray(ids))
    f = formulas.loc[ids].to_numpy()
    perm = rng.permutation(len(ids))
    room = (pd.Series(f).value_counts() - 1).to_dict()  # molecules each formula may give up
    taken, used = [], {}
    for i in perm:
        if len(taken) == k:
            break
        if used.get(f[i], 0) < room[f[i]]:
            taken.append(ids[i])
            used[f[i]] = used.get(f[i], 0) + 1
    if len(taken) < k:
        raise ValueError(f"only {len(taken)} molecules can be sampled without emptying a formula; need {k}")
    return np.sort(np.array(taken))


def build_splits(table: pd.DataFrame, cfg: dict) -> Splits:
    """Build every split from the kept table (columns id, formula, n_heavy), after exclusions
    and duplicate removal. Steps 1–7 of the module docstring, in order."""
    sizes = sorted(cfg["train_sizes"])
    n_fam, per_fam = cfg["n_familiar_formulas"], cfg["familiar_per_formula"]
    if n_fam >= sizes[0]:
        raise ValueError(f"n_familiar_formulas ({n_fam}) must be < smallest N ({sizes[0]})")
    if not set(cfg["track_b_sizes"]) <= set(sizes):
        raise ValueError("track_b_sizes must be among train_sizes")
    rng = dict(zip(STREAMS, (np.random.default_rng(s)
                             for s in np.random.SeedSequence(cfg["split_seed"]).spawn(len(STREAMS)))))

    table = table.sort_values("id")
    formula = table.set_index("id")["formula"]
    n_heavy = table.groupby("formula")["n_heavy"].first()

    # 1-2. Whole-formula holdouts: the test one, then the development one from the rest.
    unseen = choose_unseen_formulas(n_heavy, cfg["unseen_formula_frac"], rng["unseen"])
    rest = n_heavy.drop(unseen)
    dev_unseen_f = choose_unseen_formulas(rest, cfg["dev_unseen_formula_frac"], rng["dev_unseen"])
    test_unseen = table.loc[table["formula"].isin(unseen), "id"].to_numpy()
    dev_unseen = table.loc[table["formula"].isin(dev_unseen_f), "id"].to_numpy()

    # 3-5. Random molecules from the remaining formulas; no formula is emptied.
    remaining = table.loc[~table["formula"].isin([*unseen, *dev_unseen_f]), "id"].to_numpy()
    k_fam = _round_half_up(cfg["familiar_test_frac"] * len(remaining))
    test_familiar = sample_keeping_one(remaining, formula, k_fam, rng["familiar"])
    left = np.setdiff1d(remaining, test_familiar)
    dev = sample_keeping_one(left, formula, cfg["dev_size"], rng["dev"])
    pool = np.setdiff1d(left, dev)
    if len(pool) < sizes[-1]:
        raise ValueError(f"training pool has {len(pool)} molecules; largest N is {sizes[-1]}")

    # 6. Quantum subsets: a few molecules per unseen formula; anchored familiar formulas.
    rq = rng["quantum"]
    by_f = pd.Series(test_unseen).groupby(formula.loc[test_unseen].to_numpy())
    test_unseen_q = np.concatenate([rq.choice(g.to_numpy(), min(cfg["quantum_unseen_per_formula"], len(g)),
                                              replace=False) for _, g in by_f])
    if len(test_unseen_q) > cfg["quantum_unseen_max_total"]:
        test_unseen_q = rq.choice(test_unseen_q, cfg["quantum_unseen_max_total"], replace=False)
    fam_counts = formula.loc[test_familiar].value_counts().sort_index()
    eligible = fam_counts[fam_counts >= per_fam].index.to_numpy()
    if len(eligible) < n_fam:
        raise ValueError(f"only {len(eligible)} formulas have {per_fam} familiar test molecules; need {n_fam}")
    familiar = sorted(rq.choice(eligible, n_fam, replace=False).tolist())
    fam_by_f = pd.Series(test_familiar).groupby(formula.loc[test_familiar].to_numpy())
    test_familiar_q = np.concatenate([rq.choice(fam_by_f.get_group(f).to_numpy(), per_fam, replace=False)
                                      for f in familiar])

    # 7. Per training seed: anchors first, then a random permutation of the rest of P.
    pool_by_f = pd.Series(pool).groupby(formula.loc[pool].to_numpy())
    anchors, order = {}, {}
    for s in cfg["train_seeds"]:
        r = np.random.default_rng(np.random.SeedSequence([cfg["split_seed"], s]))
        a = np.sort(np.array([r.choice(pool_by_f.get_group(f).to_numpy()) for f in familiar]))
        anchors[s] = a
        order[s] = np.concatenate([a, r.permutation(np.setdiff1d(pool, a))])

    return Splits(
        config=dict(cfg), unseen_formulas=unseen, dev_unseen_formulas=dev_unseen_f,
        familiar_formulas=familiar, pool=np.sort(pool), test_unseen=np.sort(test_unseen),
        test_familiar=np.sort(test_familiar), dev=np.sort(dev), dev_unseen=np.sort(dev_unseen),
        test_unseen_q=np.sort(test_unseen_q), test_familiar_q=np.sort(test_familiar_q),
        anchors=anchors, order=order,
    )


def training_pool(s: Splits, table: pd.DataFrame | None = None) -> np.ndarray:
    """Sorted IDs of the training pool P. Every training set is drawn from P, so analyses
    restricted to P never touch a test or development molecule. (`table` is accepted for
    compatibility with callers written for X1 and ignored.)"""
    return s.pool


def check_splits(s: Splits, table: pd.DataFrame, excluded_ids: set[int] | frozenset[int]) -> None:
    """Raise SplitCheckError unless every split invariant holds. Labels:

    (a) disjoint: P, both test sets and both development sets; training sets inside P; the
        quantum subsets inside their test sets;
    (b) no held-out formula (test or development) in P, the familiar test set or dev;
    (c) every anchored familiar formula is in S_(s,N1);
    (d) nested: S_(s,N1) ⊂ S_(s,N2) ⊂ …;
    (e) sizes are exact;
    (f) IDs are unique within every set;
    (g) no excluded ID appears anywhere;
    (h) each set draws from the right formulas, and the holdouts are complete;
    (i) every kept molecule is in exactly one of P, the test sets and the development sets.

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

    main = {k: set(getattr(s, k)) for k in ("pool", "test_unseen", "test_familiar", "dev", "dev_unseen")}
    unseen, dev_unseen_f, familiar = set(s.unseen_formulas), set(s.dev_unseen_formulas), set(s.familiar_formulas)
    trains = [(seed, n, ids) for seed, by_n in s.train.items() for n, ids in by_n.items()]
    sizes = s.sizes

    # (a) disjointness and containment
    names = list(main)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if shared := main[a] & main[b]:
                fail(f"(a) {a} and {b} share {len(shared)} IDs")
    for seed, n, ids in trains:
        if outside := set(ids) - main["pool"]:
            fail(f"(a) S_({seed},{n}) has {len(outside)} IDs outside the training pool")
    for sub, parent in (("test_unseen_q", "test_unseen"), ("test_familiar_q", "test_familiar")):
        if not set(getattr(s, sub)) <= main[parent]:
            fail(f"(a) {sub} is not a subset of {parent}")
    # (b) held-out formulas stay out of training (and of the familiar-formula sets)
    if unseen & dev_unseen_f:
        fail("(b) a formula is in both the test and the development holdout")
    for name in ("pool", "test_familiar", "dev"):
        if leaked := set(formula.loc[list(main[name])]) & (unseen | dev_unseen_f):
            fail(f"(b) {name} contains held-out formulas {sorted(leaked)[:5]}")
    # (c) anchored formulas are in the smallest training set
    for seed, by_n in s.train.items():
        if missing := familiar - set(formula.loc[by_n[sizes[0]]]):
            fail(f"(c) S_({seed},{sizes[0]}) is missing familiar formulas {sorted(missing)[:5]}")
    # (d) nesting
    for seed, by_n in s.train.items():
        for small, big in zip(sizes, sizes[1:]):
            if not set(by_n[small]) <= set(by_n[big]):
                fail(f"(d) S_({seed},{small}) is not a subset of S_({seed},{big})")
    # (e) sizes
    if set(s.train) != set(s.config["train_seeds"]):
        fail(f"(e) training seeds {sorted(s.train)} != config {s.config['train_seeds']}")
    for seed, by_n in s.train.items():
        if sorted(by_n) != sizes:
            fail(f"(e) seed {seed} has sizes {sorted(by_n)}, expected {sizes}")
    for seed, n, ids in trains:
        if len(ids) != n:
            fail(f"(e) S_({seed},{n}) has {len(ids)} IDs")
    if len(s.dev) != s.config["dev_size"]:
        fail(f"(e) dev has {len(s.dev)} IDs, expected {s.config['dev_size']}")
    if len(s.test_familiar_q) != len(familiar) * s.config["familiar_per_formula"]:
        fail(f"(e) test_familiar_q has {len(s.test_familiar_q)} IDs")
    if len(s.test_unseen_q) > s.config["quantum_unseen_max_total"]:
        fail(f"(e) test_unseen_q has {len(s.test_unseen_q)} IDs, above the cap")
    # (f) uniqueness
    for name, ids in arrays.items():
        if len(set(ids)) != len(ids):
            fail(f"(f) {name} contains duplicate IDs")
    # (g) exclusions
    for name, ids in arrays.items():
        if bad := set(ids) & set(excluded_ids):
            fail(f"(g) {name} contains {len(bad)} excluded IDs")
    # (h) formulas: complete holdouts; familiar sets represented in P; anchors as listed
    kept = table[~table["id"].isin(excluded_ids)]
    if main["test_unseen"] != set(kept.loc[kept["formula"].isin(unseen), "id"]):
        fail("(h) test_unseen is not exactly the kept molecules of the unseen formulas")
    if main["dev_unseen"] != set(kept.loc[kept["formula"].isin(dev_unseen_f), "id"]):
        fail("(h) dev_unseen is not exactly the kept molecules of the development holdout formulas")
    pool_formulas = set(formula.loc[list(main["pool"])])
    for name in ("test_familiar", "dev"):
        if absent := set(formula.loc[list(main[name])]) - pool_formulas:
            fail(f"(h) {name} has formulas absent from the training pool {sorted(absent)[:5]}")
    if set(formula.loc[s.test_familiar_q]) != familiar:
        fail("(h) test_familiar_q formulas differ from the familiar formula list")
    for seed, a in s.anchors.items():
        if sorted(formula.loc[a]) != sorted(familiar) or not np.array_equal(np.sort(s.order[seed][:len(a)]), a):
            fail(f"(h) seed {seed}: anchors are not one per familiar formula at the head of the order")
    # (i) coverage
    covered = set().union(*main.values())
    if missing := set(kept["id"]) - covered:
        fail(f"(i) {len(missing)} kept molecules are in no set")


# --------------------------------------------------------------------------------------
# Saving and loading. One JSON file per set, each stamped with seeds and hashes; ID lists
# are written on one line to keep the files small.
# --------------------------------------------------------------------------------------

def _ids(a: np.ndarray) -> list[int]:
    return [int(x) for x in a]


def save_splits(s: Splits, out_dir: Path = SPLITS_DIR, extra_meta: dict | None = None) -> list[Path]:
    """Write meta.json, one {set}.json per set in SET_NAMES, and train_order_s{seed}.json.

    Existing .json files are removed first, so files from an older design (e.g. X1's
    train_s{seed}_n{N}.json) cannot be left behind.
    """
    # Stamp before touching any file: deleting committed split files would make the
    # working tree dirty, so git_hash() would always report "-dirty".
    stamp = {"split_seed": s.config["split_seed"], "config_hash": config_hash(s.config),
             "git_hash": git_hash()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    written = []

    def write(name: str, payload: dict, ids: np.ndarray | None = None) -> None:
        path = out_dir / name
        head = json.dumps({**stamp, **payload}, indent=1)
        if ids is not None:  # a long one-line list keeps the file small and diffs readable
            head = head[:-2] + ',\n "ids": ' + json.dumps(_ids(ids)) + "\n}"
        path.write_text(head + "\n")
        written.append(path)

    counts = {name: len(ids) for name, ids in s.sets().items()}
    write("meta.json", {
        "config": s.config,
        "unseen_formulas": s.unseen_formulas,
        "dev_unseen_formulas": s.dev_unseen_formulas,
        "familiar_formulas": s.familiar_formulas,
        "counts": counts,
        "train_sizes": s.sizes,
        **(extra_meta or {}),
    })
    for name, ids in s.sets().items():
        write(f"{name}.json", {"kind": name, "n": len(ids)}, ids)
    for seed, o in s.order.items():
        write(f"train_order_s{seed}.json",
              {"kind": "train_order", "seed": seed, "n": len(o), "anchor_ids": _ids(s.anchors[seed]),
               "note": "S_(seed,N) is the first N IDs of this order"}, o)
    return written


def load_splits(split_dir: Path = SPLITS_DIR) -> Splits:
    """Load splits saved by save_splits."""
    def read(name: str) -> dict:
        return json.loads((split_dir / name).read_text())

    meta = read("meta.json")
    cfg = meta["config"]
    sets = {name: np.array(read(f"{name}.json")["ids"]) for name in SET_NAMES}
    anchors, order = {}, {}
    for seed in cfg["train_seeds"]:
        d = read(f"train_order_s{seed}.json")
        order[seed] = np.array(d["ids"])
        anchors[seed] = np.array(d["anchor_ids"])
    return Splits(
        config=cfg, unseen_formulas=meta["unseen_formulas"], dev_unseen_formulas=meta["dev_unseen_formulas"],
        familiar_formulas=meta["familiar_formulas"], anchors=anchors, order=order, **sets,
    )
