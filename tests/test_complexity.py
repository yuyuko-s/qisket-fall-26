"""Effective-dimension and effective-degrees-of-freedom tests (complexity.py)."""

import numpy as np
import pytest

from qm9dipole.complexity import (
    components_for_share, condition_number, kernel_ridge_df, monte_carlo_df, participation_ratio,
    ridge_df, smoother_df, vif,
)
from qm9dipole.models import classical


def test_participation_ratio_and_components():
    assert participation_ratio(np.ones(5)) == pytest.approx(5.0)
    assert participation_ratio([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert components_for_share([5.0, 3.0, 1.0, 1.0], 0.8) == 2
    assert components_for_share([5.0, 3.0, 1.0, 1.0], 0.9) == 3
    assert condition_number([4.0, 1.0]) == pytest.approx(2.0)
    assert condition_number([4.0, 0.0]) == np.inf


def test_vif_flags_collinearity():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    v = vif(np.column_stack([a, b, a + b + 0.01 * rng.normal(size=500)]))
    assert v.iloc[2] > 100 and v.iloc[0] > 100  # each is nearly a combination of the others
    assert vif(np.column_stack([a, b])).max() < 1.1


def test_ridge_df_limits():
    X = np.random.default_rng(1).normal(size=(50, 4))
    assert ridge_df(X, 1e-12) == pytest.approx(5.0)    # OLS: 4 slopes + intercept
    assert ridge_df(X, 1e12) == pytest.approx(1.0)     # only the intercept survives


def test_kernel_ridge_df_limits():
    X = np.random.default_rng(2).normal(size=(30, 2))
    from sklearn.metrics.pairwise import rbf_kernel

    K = rbf_kernel(X, gamma=0.5)
    assert kernel_ridge_df(K, 1e-10) == pytest.approx(30.0, abs=1e-3)  # interpolates
    assert kernel_ridge_df(K, 1e10) == pytest.approx(1.0, abs=1e-6)    # just the mean


@pytest.mark.parametrize("name, params", [
    ("mean", {}), ("linear", {}), ("ridge", {"regressor__model__alpha": 3.0}),
    ("rbf_krr", {"regressor__model__alpha": 0.1, "regressor__model__gamma": 0.2}),
])
def test_smoother_df_matches_numerical_trace(name, params):
    # tr(S) = Σᵢ ∂ŷᵢ/∂yᵢ: nudge each label and watch its own prediction move.
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(40, 3)), 5 + rng.normal(size=40)
    est = classical.build(name, seed=0)[0].set_params(**params)
    base = est.fit(X, y).predict(X)
    exact = smoother_df(est, X)
    numeric, eps = 0.0, 1e-4
    for i in range(len(y)):
        y2 = y.copy()
        y2[i] += eps
        numeric += (classical.build(name, seed=0)[0].set_params(**params).fit(X, y2).predict(X)[i] - base[i]) / eps
    assert exact == pytest.approx(numeric, rel=1e-3)


def test_effective_df_picks_the_right_method():
    from qm9dipole.complexity import effective_df

    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(60, 3)), 4 + rng.normal(size=60)
    ridge = classical.build("ridge", seed=0)[0].set_params(regressor__model__alpha=1.0)
    df, se, method = effective_df(ridge, X, y, rng, n_jobs=1)
    assert method == "exact tr(S)" and se == 0.0 and 1 < df < 4
    # A linear smoother's hat matrix depends on X and α only, so a √μ target leaves df unchanged.
    sqrt_ridge = classical.build("ridge", seed=0, target="sqrt")[0].set_params(regressor__model__alpha=1.0)
    assert effective_df(sqrt_ridge, X, y, rng, n_jobs=1) == (df, 0.0, "exact tr(S)")
    # Trees: Monte Carlo in the transformed space, where label noise cannot make √μ undefined.
    y_small = np.abs(rng.normal(size=60)) * 0.05  # perturbing these in debye would go negative
    rf = classical.build("rf", seed=0, target="sqrt")[0]
    df_rf, se_rf, method_rf = effective_df(rf, X, y_small, rng, n_rep=8, n_jobs=1)
    assert method_rf == "Monte Carlo (Ye)" and np.isfinite(df_rf) and se_rf > 0


def test_monte_carlo_df_agrees_with_exact_ridge():
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(200, 6)), 3 + rng.normal(size=200)
    est = classical.build("ridge", seed=0)[0].set_params(regressor__model__alpha=10.0)
    exact = smoother_df(est.fit(X, y), X)
    estimate, se = monte_carlo_df(est, X, y, np.random.default_rng(5), n_rep=40, n_jobs=1)
    assert abs(estimate - exact) < 4 * se + 0.5
    assert smoother_df(classical.build("rf", seed=0)[0].fit(X, y), X) is None  # not a smoother
