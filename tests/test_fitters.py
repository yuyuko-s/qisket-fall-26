"""Exploration X2 harness tests: inner validation splits, supervised reductions in the shared
CV path, the fitters (models/fitters.py) and the development-set learning curve."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from qm9dipole.evaluate import clip_predictions, dev_curve, holdout_split, kfold, scores, tune
from qm9dipole.models import classical
from qm9dipole.models.fitters import ChargeFitter, MeanFitter, TabularFitter, XGBFitter, sample_params, XGB_SPACE


@pytest.fixture(scope="module")
def frame():
    """300 'molecules' (IDs 1000–1299), 12 features, y = smooth function of 3 of them, > 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    y = 3 + np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - 0.4 * X[:, 2] + 0.05 * rng.normal(size=300)
    ids = np.arange(1000, 1300)
    return (pd.DataFrame(X, index=ids, columns=[f"f{k}" for k in range(12)]),
            pd.Series(np.clip(y, 0.01, None), index=ids))


def test_holdout_split_is_disjoint_bounded_and_reproducible():
    tr, va = holdout_split(1000, seed=3)
    assert len(va) == 100 and len(tr) == 900 and not set(tr) & set(va)
    np.testing.assert_array_equal(np.sort(np.r_[tr, va]), np.arange(1000))
    np.testing.assert_array_equal(holdout_split(1000, seed=3)[1], va)
    assert not np.array_equal(holdout_split(1000, seed=4)[1], va)
    assert len(holdout_split(50, seed=0)[1]) == 10  # at least min_val
    assert len(holdout_split(200_000, seed=0)[1]) == 5000  # at most max_val


def test_tune_on_one_holdout_matches_a_manual_fit(frame):
    X, y = frame[0].to_numpy(), frame[1].to_numpy()
    est, grid = classical.build("ridge", 0)
    tr, va = holdout_split(len(y), seed=1)
    with pytest.raises(ValueError, match="partition"):
        tune(est, grid, X, y, [(tr, va)])
    r = tune(est, grid, X, y, [(tr, va)], require_partition=False)
    assert np.isnan(r.oof[tr]).all() and not np.isnan(r.oof[va]).any()
    manual = clip_predictions(clone(est).set_params(**r.params).fit(X[tr], y[tr]).predict(X[va]))
    np.testing.assert_allclose(r.oof[va], manual, rtol=1e-10)
    assert r.mae == pytest.approx(np.abs(manual - y[va]).mean())


@pytest.mark.parametrize("method", ["pca", "pls", "kbest"])
def test_supervised_reductions_in_the_shared_path_match_full_refits(frame, method):
    X, y = frame[0].to_numpy(), frame[1].to_numpy()
    est, grid = classical.build("rbf_krr", 0, scaling="standard", target="sqrt", reduction=(method, 4))
    grid = {k: v[::3] for k, v in grid.items()}
    folds = kfold(len(y), 0)
    r = tune(est, grid, X, y, folds)
    for c, params in enumerate(ParameterGrid(grid)):
        oof = np.empty(len(y))
        for tr, va in folds:
            oof[va] = clone(est).set_params(**params).fit(X[tr], y[tr]).predict(X[va])
        assert r.candidate_mae[c] == pytest.approx(np.abs(clip_predictions(oof) - y).mean(), rel=1e-8)


def test_reduction_outputs_k_standardized_inputs(frame):
    X, y = frame[0].to_numpy(), frame[1].to_numpy()
    for method in ("pca", "pls", "kbest"):
        est, _ = classical.build("ridge", 0, reduction=(method, 5))
        est.fit(X, y)
        Z = est.regressor_[:-1].transform(X)
        assert Z.shape == (300, 5)
        np.testing.assert_allclose(Z.std(axis=0), 1, rtol=1e-6)


def test_rbf_grid_scales_with_dimension():
    a, b = classical.rbf_grid(10)["model__gamma"], classical.rbf_grid(100)["model__gamma"]
    np.testing.assert_allclose(np.array(a) / 10, b)
    _, grid = classical.build("rbf_krr", 0, reduction=("pca", 8))
    assert grid["regressor__model__gamma"] == classical.rbf_grid(8)["model__gamma"]


def test_scores():
    y, p = np.array([1.0, 2.0, 10.0]), np.array([1.5, 2.0, 8.0])
    s = scores(y, p)
    assert s["mae_D"] == pytest.approx(2.5 / 3) and s["bias_D"] == pytest.approx(-1.5 / 3)
    assert s["rmse_below_tail_D"] == pytest.approx(np.sqrt(0.125))  # the 10 D molecule is excluded
    assert s["r2"] == pytest.approx(1 - np.mean((p - y) ** 2) / y.var())


def test_tabular_fitter_tunes_by_cv_or_holdout(frame):
    X, y = frame
    est, grid = classical.build("ridge", 0)
    small = TabularFitter(X, est, grid, seed=0, cv_max_n=1000).fit(X.index[:200], y.iloc[:200])
    assert small.describe()["tuning"] == "5-fold CV"
    big = TabularFitter(X, est, grid, seed=0, cv_max_n=100).fit(X.index[:200], y.iloc[:200])
    assert big.describe()["tuning"] == "holdout"
    p = big.predict(X.index[200:])
    assert p.shape == (100,) and (p >= 0).all()


def test_xgb_fitter_beats_the_mean_and_is_reproducible(frame):
    X, y = frame
    tr, te = X.index[:240], X.index[240:]
    a = XGBFitter(X, seed=0, n_iter=4, max_trees=300, early_stopping=20, n_jobs=1).fit(tr, y.loc[tr])
    b = XGBFitter(X, seed=0, n_iter=4, max_trees=300, early_stopping=20, n_jobs=1).fit(tr, y.loc[tr])
    np.testing.assert_allclose(a.predict(te), b.predict(te))
    assert np.abs(a.predict(te) - y.loc[te]).mean() < 0.8 * np.abs(y.loc[tr].mean() - y.loc[te]).mean()
    d = a.describe()
    assert d["n_trees"] >= 1 and d["n_trials"] == 4 and d["tuning"] == "5-fold CV"


def test_sample_params_respects_the_space():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample_params(XGB_SPACE, rng)
        for name, kind, lo, hi in XGB_SPACE:
            assert lo <= p[name] <= hi
        assert isinstance(p["max_depth"], int)


def test_charge_fitter_runs_on_atom_data():
    from qm9dipole.atoms import build_atom_data
    from test_charge_model import _organic_like

    rng = np.random.default_rng(1)
    mols = [_organic_like(rng, int(rng.integers(2, 7))) for _ in range(40)]
    table = pd.DataFrame({"id": np.arange(1, 41), "Z": [m[0] for m in mols], "R": [m[1] for m in mols]})
    atoms = build_atom_data(table, n_jobs=1)[0]
    y = pd.Series(rng.uniform(0.5, 4, 40), index=table["id"])
    f = ChargeFitter(atoms, {"hidden": (8,), "message_passing": 1, "max_epochs": 3, "min_steps": 10}, seed=0, n_models=2)
    f.fit(y.index[:30], y.iloc[:30])
    p = f.predict(y.index[30:])
    assert p.shape == (10,) and (p >= 0).all()
    assert len(f.describe()["epochs"]) == 2 and f.describe()["n_parameters"] > 0


def test_dev_curve_rows_and_cache(frame, tmp_path):
    X, y = frame
    train_sets = {0: {50: X.index[:50].to_numpy(), 150: X.index[:150].to_numpy()}}
    eval_sets = {"dev": X.index[200:250].to_numpy(), "dev_unseen": X.index[250:].to_numpy()}
    est, grid = classical.build("ridge", 0)

    def make(seed, n):
        return {"mean": MeanFitter(), "ridge|all features": TabularFitter(X, est, grid, seed=seed)}

    rows, preds = dev_curve(make, train_sets, eval_sets, y, cache_dir=tmp_path, progress=False)
    assert len(rows) == 2 * 2 * 2 and set(rows["eval_set"]) == {"dev", "dev_unseen"}
    mean_row = rows[(rows["model"] == "mean") & (rows["n_train"] == 50) & (rows["eval_set"] == "dev")].iloc[0]
    assert mean_row["mae_D"] == pytest.approx(np.abs(y.iloc[:50].mean() - y.loc[eval_sets["dev"]]).mean())
    assert preds[(0, 150, "ridge|all features")]["dev"].index.equals(pd.Index(eval_sets["dev"]))
    assert not any("|" in p.name or " " in p.name for p in tmp_path.glob("*.pkl"))  # names safe on Windows
    assert len(list(tmp_path.glob("*.pkl"))) == 4
    again, _ = dev_curve(make, train_sets, eval_sets, y, cache_dir=tmp_path, progress=False)
    pd.testing.assert_frame_equal(rows.drop(columns="fit_seconds"), again.drop(columns="fit_seconds"))


def test_tabular_cache_key_has_no_memory_addresses(frame):
    """The √μ target's FunctionTransformer prints "<function … at 0x…>"; the cache key must not,
    or every new process misses the cache."""
    import subprocess
    import sys

    est, grid = classical.build("rbf_krr", 0, target="sqrt", scaling="yeo_johnson", reduction=("pls", 3))
    assert " at 0x" in repr(est)  # the raw repr would break the key
    fitter = TabularFitter(frame[0], est, grid, seed=0)
    assert " at 0x" not in fitter.config["estimator"]
    from qm9dipole.provenance import config_hash

    here = config_hash(fitter.config)
    code = ("import pandas as pd, numpy as np\n"
            "from qm9dipole.models import classical\nfrom qm9dipole.models.fitters import TabularFitter\n"
            "from qm9dipole.provenance import config_hash\n"
            "f = pd.DataFrame(np.zeros((3, 12)), columns=[f'f{k}' for k in range(12)])\n"
            "e, g = classical.build('rbf_krr', 0, target='sqrt', scaling='yeo_johnson', reduction=('pls', 3))\n"
            "print(config_hash(TabularFitter(f, e, g, seed=0).config))")
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert other == here  # the same key in a fresh process
