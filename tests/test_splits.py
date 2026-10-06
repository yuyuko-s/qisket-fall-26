"""Split builder tests on a synthetic dataset, plus a check of the saved splits/ on real data.

Each PLAN §4 assert gets a test that breaks exactly that invariant and expects check_splits to
fail with its label: the split checks' equivalent of the invariance negative control.
"""

import copy
import re

import numpy as np
import pandas as pd
import pytest

from qm9dipole.provenance import config_hash
from qm9dipole.splits import (
    SPLITS_DIR, SplitCheckError, build_splits, check_splits, load_config, load_splits,
    save_splits, training_pool,
)

CFG = {
    "exclude_readme_flagged": True,
    "split_seed": 7,
    "max_per_formula": 25,
    "unseen_formula_frac": 0.15,
    "unseen_per_formula": 5,
    "unseen_max_total": 600,
    "n_familiar_formulas": 10,
    "familiar_per_formula": 2,
    "familiar_min_pool": 3,
    "train_sizes": [50, 150, 400],
    "train_seeds": [0, 1, 2],
}
N1, N2, N3 = CFG["train_sizes"]


@pytest.fixture(scope="module")
def table():
    """150 formulas with 1–39 molecules each, heavy-atom counts 1–9."""
    rng = np.random.default_rng(0)
    rows, next_id = [], 1
    for i in range(150):
        for _ in range(int(rng.integers(1, 40))):
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
    assert again.familiar_formulas == splits.familiar_formulas
    for name, ids in splits.all_id_arrays().items():
        np.testing.assert_array_equal(again.all_id_arrays()[name], ids)


def test_training_seeds_give_different_sets(splits):
    assert not np.array_equal(splits.train[0][N3], splits.train[1][N3])
    assert not np.array_equal(splits.anchors[0], splits.anchors[1])


def test_test_sets_do_not_depend_on_training_seeds(splits, table):
    other = build_splits(table, {**CFG, "train_seeds": [5, 6]})
    np.testing.assert_array_equal(other.test_unseen, splits.test_unseen)
    np.testing.assert_array_equal(other.test_familiar, splits.test_familiar)


def test_one_anchor_per_familiar_formula(splits, table):
    formula = table.set_index("id")["formula"]
    for a in splits.anchors.values():
        assert sorted(formula.loc[a]) == splits.familiar_formulas


def test_training_pool_holds_every_training_set_and_no_test_molecule(splits, table):
    P = set(training_pool(splits, table))
    assert not P & (set(splits.test_unseen) | set(splits.test_familiar))
    assert all(set(ids) <= P for by_n in splits.train.values() for ids in by_n.values())


def test_rejects_more_familiar_formulas_than_smallest_n(table):
    with pytest.raises(ValueError, match="smallest N"):
        build_splits(table, {**CFG, "n_familiar_formulas": N1})


def test_save_load_round_trip(splits, tmp_path):
    save_splits(splits, tmp_path)
    back = load_splits(tmp_path)
    assert back.config == splits.config
    for name, ids in splits.all_id_arrays().items():
        np.testing.assert_array_equal(back.all_id_arrays()[name], ids)
    for seed in CFG["train_seeds"]:
        np.testing.assert_array_equal(back.anchors[seed], splits.anchors[seed])


def test_save_stamps_git_hash_before_touching_files(splits, tmp_path, monkeypatch):
    # Regression: deleting the old train files before stamping made every stamp "-dirty".
    import qm9dipole.splits as splits_module

    save_splits(splits, tmp_path)
    n_train_files = len(list(tmp_path.glob("train_s*_n*.json")))

    def fake_git_hash():
        assert len(list(tmp_path.glob("train_s*_n*.json"))) == n_train_files
        return "abc123"

    monkeypatch.setattr(splits_module, "git_hash", fake_git_hash)
    save_splits(splits, tmp_path)
    assert '"git_hash": "abc123"' in (tmp_path / "train_s0_n50.json").read_text()


# --- Negative controls: break one invariant at a time --------------------------------

def _top_only(s):
    """IDs in S_(0,N3) but not S_(0,N2): editable without affecting nesting or anchors."""
    return np.setdiff1d(s.train[0][N3], s.train[0][N2])


def _replace(ids, old, new):
    out = ids.copy()
    out[out == old] = new
    return np.sort(out)


def _training_pool(s, table):
    unseen_ids = table.loc[table["formula"].isin(s.unseen_formulas), "id"]
    return np.setdiff1d(s.pool, np.concatenate([unseen_ids, s.test_familiar]))


def break_a(s, table):  # a familiar test molecule leaks into training
    s.train[0][N3] = _replace(s.train[0][N3], _top_only(s)[0], s.test_familiar[0])


def break_b(s, table):  # an unseen-formula molecule (not itself a test molecule) leaks in
    unseen_pool = table.loc[table["formula"].isin(s.unseen_formulas) & table["id"].isin(s.pool), "id"]
    leftover = np.setdiff1d(unseen_pool, s.test_unseen)[0]
    s.train[0][N3] = _replace(s.train[0][N3], _top_only(s)[0], leftover)


def break_c(s, table):  # a familiar formula vanishes from the smallest set
    f_ids = table.loc[table["formula"] == s.familiar_formulas[0], "id"]
    s.train[0][N1] = np.setdiff1d(s.train[0][N1], f_ids)


def break_d(s, table):  # S_N1 gains a molecule that S_N3 lacks
    outside = np.setdiff1d(_training_pool(s, table), s.train[0][N3])[0]
    victim = np.setdiff1d(s.train[0][N1], s.anchors[0])[0]
    s.train[0][N1] = _replace(s.train[0][N1], victim, outside)


def break_e(s, table):  # wrong size
    s.train[0][N3] = np.setdiff1d(s.train[0][N3], _top_only(s)[:1])


def break_f(s, table):  # duplicate ID
    top = _top_only(s)
    s.train[0][N3] = _replace(s.train[0][N3], top[0], top[1])


@pytest.mark.parametrize("label, breaker", [
    ("(a)", break_a), ("(b)", break_b), ("(c)", break_c),
    ("(d)", break_d), ("(e)", break_e), ("(f)", break_f),
])
def test_checks_catch_each_violation(splits, table, label, breaker):
    s = copy.deepcopy(splits)
    breaker(s, table)
    with pytest.raises(SplitCheckError, match=f"^{re.escape(label)}"):
        check_splits(s, table, excluded_ids=set())


def test_checks_catch_excluded_ids(splits, table):
    with pytest.raises(SplitCheckError, match=r"^\(g\)"):
        check_splits(splits, table, excluded_ids={int(splits.train[0][N3][0])})


# --- The real, saved splits ----------------------------------------------------------

@pytest.mark.skipif(not (SPLITS_DIR / "meta.json").exists(), reason="splits/ not built yet")
def test_saved_splits_are_current_and_valid():
    from qm9dipole.data import QM9_PARQUET, exclusion_rules

    s = load_splits()
    cfg = load_config()
    assert s.config == cfg, "configs/splits.yaml changed since splits/ was built; rebuild it"
    assert (SPLITS_DIR / "meta.json").read_text().count(config_hash(cfg)) >= 1
    if not QM9_PARQUET.exists():
        pytest.skip("data/processed/qm9.parquet not built")
    table = pd.read_parquet(QM9_PARQUET, columns=["id", "formula"])
    rules = exclusion_rules(cfg["exclude_readme_flagged"])
    excluded = frozenset().union(*rules.values())
    check_splits(s, table, excluded)
