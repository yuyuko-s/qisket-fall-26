"""Scaling and target-transform tests (preprocess.py), and their use in the models."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from qm9dipole.evaluate import kfold, tune
from qm9dipole.models import classical
from qm9dipole.preprocess import SCALINGS, TARGETS, SkewedLog, scaling_steps, target_transformer


@pytest.fixture(scope="module")
def skewed_data():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.lognormal(0, 1.5, 300), rng.normal(size=300), np.zeros(300),
                         rng.integers(0, 3, 300)])  # skewed, symmetric, constant, discrete
    return X


@pytest.mark.parametrize("name", SCALINGS)
def test_every_scaling_is_finite_and_fit_on_training_rows(name, skewed_data):
    pipe = Pipeline(scaling_steps(name)).fit(skewed_data[:200])
    out = pipe.transform(skewed_data[200:])
    assert out.shape == (100, 4) and np.isfinite(out).all()


def test_skewed_log_picks_columns_from_training_data(skewed_data):
    t = SkewedLog(threshold=1.0).fit(skewed_data)
    assert t.columns_.tolist() == [0]  # only the lognormal column is skewed
    out = t.transform(skewed_data)
    np.testing.assert_allclose(out[:, 0], np.log1p(skewed_data[:, 0]))
    np.testing.assert_array_equal(out[:, 1:], skewed_data[:, 1:])
    neg = SkewedLog(0.0).fit(-skewed_data[:, :1]).transform(-skewed_data[:, :1])
    np.testing.assert_allclose(neg[:, 0], -np.log1p(skewed_data[:, 0]))  # signed log


@pytest.mark.parametrize("name", TARGETS)
def test_target_transforms_invert_exactly_on_nonnegative_mu(name):
    mu = np.array([[0.0], [0.4], [1.85], [29.6]])
    t = target_transformer(name).fit(mu)
    np.testing.assert_allclose(t.inverse_transform(t.transform(mu)), mu, atol=1e-12)


def test_sqrt_inverse_maps_negative_predictions_to_zero():
    t = target_transformer("sqrt").fit(np.array([[1.0], [4.0], [9.0]]))
    z_below_zero = t.transform(np.array([[0.0]])) - 5.0  # far below √μ = 0
    assert t.inverse_transform(z_below_zero)[0, 0] == 0.0


def test_kernel_ridge_falls_back_to_the_training_mean_far_from_data():
    # Kernel ridge has no intercept: away from every training point its prediction decays
    # to 0 in the target space. Standardizing μ makes that the training mean, not 0 D.
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(80, 3)), 20 + rng.normal(size=80)
    far = np.full((1, 3), 50.0)
    est, _ = classical.build("rbf_krr", seed=0)
    pred = est.set_params(regressor__model__gamma=1.0, regressor__model__alpha=0.1).fit(X, y).predict(far)
    assert pred[0] == pytest.approx(y.mean(), abs=1e-6)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("scaling", SCALINGS)
def test_models_accept_every_scaling_and_target(scaling, target, skewed_data):
    y = 2.0 + skewed_data[:, 0] / 5
    est, grid = classical.build("ridge", seed=0, compressed=False, scaling=scaling, target=target)
    r = tune(est, {k: v[:3] for k, v in grid.items()}, skewed_data, y, kfold(len(y), 0), n_jobs=1)
    assert np.isfinite(r.mae) and (r.oof >= 0).all()


def _near_empty_column_data():
    """A training set with one column that is non-zero in a single row (1e-12), like the
    radial-distribution bins at physically empty distances that blew up ridge in explore_03,
    and a new molecule with 1e-5 in that column (z-score ~1e8 without the guard)."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(240, 4))
    X[:, 3] = 0.0
    X[17, 3] = 1e-12
    y = 3.0 + X[:, 0] + 0.3 * rng.normal(size=240)
    return X, y, np.array([[0.1, -0.2, 0.3, 1e-5]])


@pytest.mark.parametrize("scaling", ["standard", "log_standard"])
def test_z_clip_stops_ridge_extrapolating_a_near_empty_column(scaling):
    from sklearn.base import clone

    X, y, new = _near_empty_column_data()
    est, _ = classical.build("ridge", seed=0, scaling=scaling, target="standard")
    est.set_params(regressor__model__alpha=1e-3)
    assert [name for name, _ in est.regressor.steps][-2] == "clip"
    guarded = est.fit(X, y).predict(new)[0]
    assert 0 < guarded < 2 * y.max()
    unguarded_steps = [s for s in est.regressor.steps if s[0] != "clip"]
    unguarded = clone(est).set_params(regressor=Pipeline(unguarded_steps)).fit(X, y).predict(new)[0]
    assert abs(unguarded) > 1e3  # the failure the guard exists for


def test_z_clip_leaves_ordinary_columns_unchanged():
    from sklearn.preprocessing import StandardScaler

    X = np.random.default_rng(4).normal(size=(500, 6))
    np.testing.assert_array_equal(Pipeline(scaling_steps("standard")).fit_transform(X),
                                  StandardScaler().fit_transform(X))


def test_z_clip_stops_ridge_extrapolating_a_steep_yeo_johnson_curve():
    """A left-skewed training column makes Yeo-Johnson fit a steep power curve (λ ≈ 10); a new
    molecule above the training range then scores z ≈ 4,000, as small molecules' padded
    Coulomb-matrix eigenvalues did on dev_unseen."""
    from sklearn.base import clone

    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 4))
    X[:, 3] = 20 - rng.lognormal(0, 0.8, 100)
    y = 3.0 + X[:, 0] + 0.5 * (X[:, 3] - X[:, 3].mean()) + 0.2 * rng.normal(size=100)
    new = np.array([[0.1, -0.2, 0.3, 40.0]])
    est, _ = classical.build("ridge", seed=0, scaling="yeo_johnson", target="standard")
    est.set_params(regressor__model__alpha=1.0)
    guarded = est.fit(X, y).predict(new)[0]
    unguarded = clone(est).set_params(regressor=Pipeline([s for s in est.regressor.steps if s[0] != "clip"]))
    unguarded = unguarded.fit(X, y).predict(new)[0]
    assert 0 < guarded < 20
    assert unguarded > 1000
