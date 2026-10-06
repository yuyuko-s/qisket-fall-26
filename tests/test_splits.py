"""Split builder tests on a synthetic dataset, plus a check of the saved splits/ on real data.

Each split invariant gets a test that breaks exactly that invariant and expects check_splits to
fail with its label: the split checks' equivalent of the invariance negative control.
"""

import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from qm9dipole.provenance import config_hash
from qm9dipole.splits import (
    SET_NAMES, SPLITS_DIR, SplitCheckError, build_splits, check_splits, choose_unseen_formulas,
    load_config, load_splits, sample_keeping_one, save_splits, training_pool,
)

CFG = {
    "exclude_readme_flagged": True,
    "exclude_geometric_duplicates": True,
    "duplicate_cm_decimals": 2,
    "split_seed": 7,
    "unseen_formula_frac": 0.15,
    "dev_unseen_formula_frac": 0.05,
    "familiar_test_frac": 0.10,
    "dev_size": 150,
    "quantum_unseen_per_formula": 5,
    "quantum_unseen_max_total": 60,
    "n_familiar_formulas": 10,
    "familiar_per_formula": 2,
    "train_sizes": [50, 150, 400],
    "train_full_pool": True,
    "train_seeds": [0, 1, 2],
    "track_b_sizes": [50, 150],
}
N1, N2, N3 = CFG["train_sizes"]


@pytest.fixture(scope="module")
def table():
    """150 formulas with 1–59 molecules each, heavy-atom counts 1–9."""
    rng = np.random.default_rng(0)
    rows, next_id = [], 1
    for i in range(150):
        for _ in range(int(rng.integers(1, 60))):
            rows.append({"id": next_id, "formula": f"X{i}", "n_heavy": 1 + i % 9})
            next_id += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def splits(table):
    return build_splits(table, CFG)


def test_built_splits_pass_all_checks(splits, table):
    check_splits(splits, table, excluded_ids=set())


def test_build_is_deterministic(splits, table):
    again = build_splits(table, CFG)
    assert again.unseen_formulas == splits.unseen_formulas
    assert again.dev_unseen_formulas == splits.dev_unseen_formulas
    assert again.familiar_formulas == splits.familiar_formulas
    for name, ids in splits.all_id_arrays().items():
        np.testing.assert_array_equal(again.all_id_arrays()[name], ids)


def test_training_seeds_give_different_sets(splits):
    assert not np.array_equal(splits.train[0][N3], splits.train[1][N3])
    assert not np.array_equal(splits.anchors[0], splits.anchors[1])


def test_full_pool_is_the_last_size_and_the_same_for_every_seed(splits):
    full = len(splits.pool)
    assert splits.sizes == [*CFG["train_sizes"], full]
    for by_n in splits.train.values():
        np.testing.assert_array_equal(by_n[full], splits.pool)


def test_held_out_and_development_sets_do_not_depend_on_training_seeds(splits, table):
    other = build_splits(table, {**CFG, "train_seeds": [5, 6]})
    for name in SET_NAMES:
        np.testing.assert_array_equal(getattr(other, name), getattr(splits, name))


def test_unseen_formulas_use_split_stream_1(splits, table):
    # The unseen draw keeps X1's stream, so a design change elsewhere cannot move it.
    rng = np.random.default_rng(np.random.SeedSequence(CFG["split_seed"]).spawn(2)[1])
    n_heavy = table.groupby("formula")["n_heavy"].first()
    assert choose_unseen_formulas(n_heavy, CFG["unseen_formula_frac"], rng) == splits.unseen_formulas


def test_anchors_head_every_order_one_per_familiar_formula(splits, table):
    formula = table.set_index("id")["formula"]
    for seed, a in splits.anchors.items():
        assert sorted(formula.loc[a]) == splits.familiar_formulas
        np.testing.assert_array_equal(np.sort(splits.order[seed][: len(a)]), a)


def test_training_pool_holds_every_training_set_and_nothing_held_out(splits):
    P = set(training_pool(splits))
    held_out = set().union(*(set(getattr(splits, n)) for n in SET_NAMES if n != "pool"))
    assert not P & held_out
    assert all(set(ids) <= P for by_n in splits.train.values() for ids in by_n.values())


def test_sample_keeping_one_never_empties_a_formula():
    formulas = pd.Series(["A"] * 3 + ["B"] * 1 + ["C"] * 6, index=np.arange(10))
    for seed in range(20):
        taken = sample_keeping_one(formulas.index.to_numpy(), formulas, 7, np.random.default_rng(seed))
        assert len(taken) == 7  # the most possible: A gives 2, B none, C 5
        left = formulas.drop(taken)
        assert set(left) == {"A", "B", "C"}
    with pytest.raises(ValueError, match="without emptying"):
        sample_keeping_one(formulas.index.to_numpy(), formulas, 8, np.random.default_rng(0))


def test_rejects_more_familiar_formulas_than_smallest_n(table):
    with pytest.raises(ValueError, match="smallest N"):
        build_splits(table, {**CFG, "n_familiar_formulas": N1})


def test_save_load_round_trip_and_stale_files_removed(splits, tmp_path):
    (tmp_path / "train_s0_n100.json").write_text("{}")  # an X1-era file
    save_splits(splits, tmp_path)
    assert not (tmp_path / "train_s0_n100.json").exists()
    back = load_splits(tmp_path)
    assert back.config == splits.config
    for name, ids in splits.all_id_arrays().items():
        np.testing.assert_array_equal(back.all_id_arrays()[name], ids)
    for seed in CFG["train_seeds"]:
        np.testing.assert_array_equal(back.anchors[seed], splits.anchors[seed])
    assert json.loads((tmp_path / "dev.json").read_text())["n"] == CFG["dev_size"]


def test_save_stamps_git_hash_before_touching_files(splits, tmp_path, monkeypatch):
    # Regression: deleting the old files before stamping made every stamp "-dirty".
    import qm9dipole.splits as splits_module

    save_splits(splits, tmp_path)
    n_files = len(list(tmp_path.glob("*.json")))

    def fake_git_hash():
        assert len(list(tmp_path.glob("*.json"))) == n_files
        return "abc123"

    monkeypatch.setattr(splits_module, "git_hash", fake_git_hash)
    save_splits(splits, tmp_path)
    assert '"git_hash": "abc123"' in (tmp_path / "train_order_s0.json").read_text()


# --- Negative controls: break one invariant at a time --------------------------------

def _top_only(s):
    """IDs in S_(0,N3) but not S_(0,N2): editable without affecting nesting or anchors."""
    return np.setdiff1d(s.train[0][N3], s.train[0][N2])


def _replace(ids, old, new):
    out = ids.copy()
    out[out == old] = new
    return np.sort(out)


def break_a(s, table):  # a familiar test molecule leaks into training
    s.train[0][N3] = _replace(s.train[0][N3], _top_only(s)[0], s.test_familiar[0])
    return table


def break_b(s, table):  # a pool molecule turns out to have a held-out formula
    t = table.copy()
    t.loc[t["id"] == s.pool[0], "formula"] = s.unseen_formulas[0]
    return t


def break_c(s, table):  # a familiar formula vanishes from the smallest set
    f_ids = table.loc[table["formula"] == s.familiar_formulas[0], "id"]
    s.train[0][N1] = np.setdiff1d(s.train[0][N1], f_ids)
    return table


def break_d(s, table):  # S_N1 gains a molecule that S_N3 lacks
    outside = np.setdiff1d(s.pool, s.train[0][N3])[0]
    victim = np.setdiff1d(s.train[0][N1], s.anchors[0])[0]
    s.train[0][N1] = _replace(s.train[0][N1], victim, outside)
    return table


def break_e(s, table):  # wrong size
    s.train[0][N3] = np.setdiff1d(s.train[0][N3], _top_only(s)[:1])
    return table


def break_f(s, table):  # duplicate ID
    top = _top_only(s)
    s.train[0][N3] = _replace(s.train[0][N3], top[0], top[1])
    return table


def break_h(s, table):  # an unseen-formula molecule is dropped from its holdout
    victim = np.setdiff1d(s.test_unseen, s.test_unseen_q)[0]  # not in the quantum subset: (a) holds
    s.test_unseen = np.setdiff1d(s.test_unseen, [victim])
    return table


def break_i(s, table):  # a kept molecule is in no set
    row = table[table["id"] == s.pool[0]].assign(id=table["id"].max() + 1)
    return pd.concat([table, row], ignore_index=True)


@pytest.mark.parametrize("label, breaker", [
    ("(a)", break_a), ("(b)", break_b), ("(c)", break_c), ("(d)", break_d),
    ("(e)", break_e), ("(f)", break_f), ("(h)", break_h), ("(i)", break_i),
])
def test_checks_catch_each_violation(splits, table, label, breaker):
    s = copy.deepcopy(splits)
    t = breaker(s, table)
    with pytest.raises(SplitCheckError, match=f"^{re.escape(label)}"):
        check_splits(s, t, excluded_ids=set())


def test_checks_catch_excluded_ids(splits, table):
    with pytest.raises(SplitCheckError, match=r"^\(g\)"):
        check_splits(splits, table, excluded_ids={int(splits.train[0][N3][0])})


# --- The real, saved splits ----------------------------------------------------------

def _saved_exclusions(cfg):
    from qm9dipole.data import exclusion_rules

    rules = exclusion_rules(cfg["exclude_readme_flagged"])
    meta = json.loads((SPLITS_DIR / "meta.json").read_text())
    return frozenset().union(*rules.values(), meta["excluded_duplicates"])


@pytest.mark.skipif(not (SPLITS_DIR / "meta.json").exists(), reason="splits/ not built yet")
def test_saved_splits_are_current_and_valid():
    from qm9dipole.data import QM9_PARQUET

    s = load_splits()
    cfg = load_config()
    assert s.config == cfg, "configs/splits.yaml changed since splits/ was built; rebuild it"
    assert (SPLITS_DIR / "meta.json").read_text().count(config_hash(cfg)) >= 1
    if not QM9_PARQUET.exists():
        pytest.skip("data/processed/qm9.parquet not built")
    table = pd.read_parquet(QM9_PARQUET, columns=["id", "formula"])
    check_splits(s, table, _saved_exclusions(cfg))


@pytest.mark.skipif(not (SPLITS_DIR / "meta.json").exists(), reason="splits/ not built yet")
def test_no_duplicate_molecule_anywhere_in_saved_splits():
    # QM9 lists a few molecules twice (identical geometry and μ; explore_01). The split build
    # keeps one copy of each, so no two saved IDs, in any set, may share a geometry. Recomputed
    # here from geometry alone (no label is read), independently of the build.
    from qm9dipole.cleaning import duplicate_groups
    from qm9dipole.data import QM9_PARQUET, load_qm9_table
    from qm9dipole.descriptors import cm_spectrum

    if not QM9_PARQUET.exists():
        pytest.skip("data/processed/qm9.parquet not built")
    s = load_splits()
    every = np.unique(np.concatenate(list(s.sets().values())))
    t = load_qm9_table().set_index("id").loc[every]
    spectra = np.stack([cm_spectrum(z, r) for z, r in zip(t["Z"], t["R"])])
    group = duplicate_groups(t.index.to_numpy(), t["formula"], spectra, s.config["duplicate_cm_decimals"])
    assert group.empty, f"{len(group)} saved molecules share a geometry with another"
