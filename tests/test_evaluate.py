"""CV harness and classical-model tests (evaluate.py, models/classical.py)."""

import numpy as np
import pandas as pd
import pytest

from qm9dipole.evaluate import (
    clip_predictions, group_kfold, kfold, learning_curve, mae, permutation_importance_cv, rmse,
    tune,
)
from qm9dipole.models import classical


@pytest.fixture(scope="module")
def linear_data():
    """y = 3·x0 − 2·x1 + 20 + small noise; x2 is pure noise. The offset keeps y > 0, like
    |μ|, so clipping predictions at 0 never matters here."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 20 + 0.1 * rng.normal(size=120)
    assert y.min() > 0
    return X, y


# --- Metrics and folds -----------------------------------------------------------------

def test_metrics_and_clipping():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 2.0])
    assert mae(y, p) == pytest.approx(1.0) and rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    np.testing.assert_array_equal(clip_predictions(np.array([-0.5, 0.0, 2.0])), [0.0, 0.0, 2.0])


def test_kfold_partitions_and_is_reproducible():
    folds = kfold(23, seed=1)
    assert len(folds) == 5
    np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(23))
    for train, val in folds:
        assert not set(train) & set(val)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, kfold(23, seed=1)))
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(folds, kfold(23, seed=2)))


def test_group_kfold_never_splits_a_group():
    groups = np.repeat(np.arange(12), 3)  # 12 "formulas" with 3 molecules each
    for train, val in group_kfold(groups, seed=0):
        assert not set(groups[train]) & set(groups[val])


# --- tune --------------------------------------------------------------------------------

def test_tune_picks_small_alpha_for_a_clean_linear_signal(linear_data):
    X, y = linear_data
    pipe, grid = classical.build("ridge", seed=0)
    r = tune(pipe, grid, X, y, kfold(len(y), 0), n_jobs=1)
    assert r.params["regressor__model__alpha"] <= 1.0
    assert r.mae < 0.2 and r.oof.shape == y.shape
    assert r.mae == pytest.approx(mae(y, r.oof)) and r.rmse == pytest.approx(rmse(y, r.oof))
    assert len(r.candidate_mae) == len(grid["regressor__model__alpha"])


def test_shared_feature_steps_give_the_same_results_as_full_refits(linear_data):
    # tune fits scaling + PCA once per fold and shares them across candidates; refitting the
    # whole estimator for every candidate must give identical out-of-fold errors.
    from sklearn.base import clone

    X, y = linear_data
    folds = kfold(len(y), 0)
    est, grid = classical.build("ridge", seed=0, compressed=True, n_components=2, scaling="yeo_johnson")
    grid = {k: v[::4] for k, v in grid.items()}
    r = tune(est, grid, X, y, folds, n_jobs=1)
    for c, params in enumerate(r.candidates):
        oof = np.empty_like(y)
        for train, val in folds:
            oof[val] = clone(est).set_params(**params).fit(X[train], y[train]).predict(X[val])
        assert r.candidate_mae[c] == pytest.approx(mae(y, clip_predictions(oof)), rel=1e-10)


def test_tune_mean_model_matches_hand_computed_cv(linear_data):
    X, y = linear_data
    folds = kfold(len(y), 0)
    expected = np.empty_like(y)
    for train, val in folds:
        expected[val] = y[train].mean()
    r = tune(*classical.build("mean", seed=0), X, y, folds, n_jobs=1)
    np.testing.assert_allclose(r.oof, clip_predictions(expected))


def test_tune_rejects_folds_that_do_not_partition(linear_data):
    X, y = linear_data
    folds = kfold(len(y), 0)[:4]  # one fold missing: some samples never validated
    with pytest.raises(ValueError, match="partition"):
        tune(*classical.build("mean", seed=0), X, y, folds, n_jobs=1)


def test_predictions_are_clipped_at_zero():
    # A linear fit to y = |x| extrapolates below zero; clipped predictions cannot.
    x = np.linspace(-3, 3, 60)[:, None]
    r = tune(*classical.build("linear", seed=0), x, np.abs(x[:, 0]) - 1.2, kfold(60, 0), n_jobs=1)
    assert (r.oof >= 0).all()


# --- Models ------------------------------------------------------------------------------

@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("name", classical.MODELS)
def test_every_model_fits_and_predicts(name, compressed):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(40, 10)), rng.normal(size=40)
    pipe, grid = classical.build(name, seed=0, compressed=compressed)
    assert ("pca" in pipe.regressor.named_steps) == compressed
    params = {k: v[0] for k, v in grid.items()}
    pred = pipe.set_params(**params).fit(X, y).predict(X)
    assert pred.shape == (40,) and np.isfinite(pred).all()


def test_compressed_pipeline_fits_pca_on_training_rows_only():
    # PCA inside the pipeline sees only the rows it is fit on (docs/EVALUATION_RULES.md rule 3).
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 12))
    pipe, _ = classical.build("ridge", seed=0, compressed=True)
    pipe.fit(X[:30], rng.normal(size=30))
    np.testing.assert_allclose(pipe.regressor_.named_steps["scale"].mean_, X[:30].mean(axis=0))
    assert pipe.regressor_.named_steps["pca"].n_components_ == classical.N_COMPONENTS


def test_tree_models_are_reproducible_for_a_seed():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(60, 5)), rng.normal(size=60)
    for name in ("rf", "xgb"):
        a = classical.build(name, seed=7)[0].fit(X, y).predict(X)
        b = classical.build(name, seed=7)[0].fit(X, y).predict(X)
        np.testing.assert_array_equal(a, b)


# --- learning_curve and permutation importance ----------------------------------------

def test_learning_curve_long_format(linear_data):
    X, y = linear_data
    ids = np.arange(1000, 1000 + len(y))
    features = {"lin": pd.DataFrame(X, index=ids)}
    train_sets = {0: {40: ids[:40], 80: ids[:80]}, 1: {40: ids[40:80], 80: ids[40:120]}}

    def make_specs(seed, fname):
        return classical.specs(seed, models=("mean", "ridge"))

    df, oof = learning_curve(make_specs, features, pd.Series(y, index=ids), train_sets,
                             n_jobs=1, progress=False)
    assert len(df) == 2 * 2 * 2 and (df["eval_set"] == "cv").all()
    assert set(df["n_train"]) == {40, 80} and (df["n_eval"] == df["n_train"]).all()
    ridge, mean = (df[df["model"] == m].set_index(["seed", "n_train"])["mae_D"] for m in ("ridge", "mean"))
    assert (ridge < mean).all()
    assert oof[(1, 80, "ridge", "lin")].index.equals(pd.Index(train_sets[1][80]))


def test_permutation_importance_finds_the_informative_features(linear_data):
    X, y = linear_data
    pipe, _ = classical.build("ridge", seed=0)
    imp = permutation_importance_cv(pipe.set_params(regressor__model__alpha=1e-3), X, y, kfold(len(y), 0),
                                    np.random.default_rng(0), n_repeats=3)
    assert imp[0] > imp[1] > 10 * abs(imp[2])  # |3| > |−2| > noise column
