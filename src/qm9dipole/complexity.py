"""Effective dimension of the inputs and effective number of parameters of fitted models.

Inputs (explore_03): how many independent directions do the standardized features span?
- participation ratio PR = (Σλ)² / Σλ² of the covariance eigenvalues: d for d equally
  strong directions, 1 if one direction dominates;
- components needed for a share of the variance; condition number √(λ_max/λ_min);
- variance inflation factors VIF_j = 1 / (1 − R²_j), R²_j from regressing feature j on the
  others: above ~10, a linear model cannot separate feature j's effect from the rest.

Fitted models (explore_04), following Hastie, Tibshirani & Friedman, ESL §7.6: a model
that fits ŷ = S y is a linear smoother, and its effective number of parameters is
df = tr(S) = Σᵢ ∂ŷᵢ/∂yᵢ, the total self-influence of the training labels:
- mean: 1; OLS: 1 + rank of the centred design;
- ridge (with intercept): 1 + Σⱼ dⱼ² / (dⱼ² + α), d = singular values of the centred design;
- kernel ridge on a centred target (no intercept): S = K(K+αI)⁻¹(I − J) + J, J = 11ᵀ/n,
  so df = 1 + tr(K(K+αI)⁻¹) − 1ᵀK(K+αI)⁻¹1 / n. The same formula will apply to the
  quantum kernel in M4.
For models that are not linear smoothers (random forest, boosting), df is estimated with
Ye's (1998) generalized degrees of freedom: perturb y by small Gaussian noise δ, refit, and
sum the per-sample slopes of ŷᵢ on δᵢ, df ≈ Σᵢ cov(ŷᵢ, δᵢ) / τ².
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics.pairwise import rbf_kernel


# --- Effective dimension of the inputs --------------------------------------------------

def participation_ratio(eigenvalues: np.ndarray) -> float:
    """(Σλ)² / Σλ²: the number of equally strong directions with the same spread."""
    lam = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    return float(lam.sum() ** 2 / (lam**2).sum())


def components_for_share(eigenvalues: np.ndarray, share: float) -> int:
    """Smallest k whose top-k eigenvalues hold at least `share` of the total."""
    lam = np.sort(np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None))[::-1]
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), share - 1e-12) + 1)


def condition_number(eigenvalues: np.ndarray, floor: float = 1e-12) -> float:
    """√(λ_max / λ_min) of the covariance, i.e. the design matrix's condition number;
    eigenvalues below `floor` × λ_max count as exact collinearity (infinite)."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    lo, hi = lam.min(), lam.max()
    return float("inf") if lo <= floor * hi else float(np.sqrt(hi / lo))


def vif(X: np.ndarray, names: list[str] | None = None) -> pd.Series:
    """Variance inflation factor of every column of standardized X (inf if exactly
    collinear), from the diagonal of the inverse correlation matrix."""
    corr = np.corrcoef(np.asarray(X, dtype=np.float64), rowvar=False)
    try:
        inv = np.linalg.inv(corr)
        values = np.diag(inv)
        values = np.where(values < 0, np.inf, values)  # numerically singular
    except np.linalg.LinAlgError:
        values = np.full(corr.shape[0], np.inf)
    return pd.Series(values, index=names, name="VIF")


# --- Effective number of parameters of fitted models ------------------------------------

def ridge_df(Xt: np.ndarray, alpha: float) -> float:
    """df of ridge with an unpenalized intercept on design Xt: 1 + Σ d²/(d² + α)."""
    Xc = np.asarray(Xt, dtype=np.float64) - np.mean(Xt, axis=0)
    d = np.linalg.svd(Xc, compute_uv=False)
    return float(1 + np.sum(d**2 / (d**2 + alpha)))


def kernel_ridge_df(K: np.ndarray, alpha: float) -> float:
    """df of kernel ridge (no intercept) fit to a centred target, with the mean added
    back: tr(H) for H = K(K+αI)⁻¹(I − J) + J."""
    n = len(K)
    A = np.linalg.solve(K + alpha * np.eye(n), K).T  # K(K+αI)⁻¹; K symmetric
    return float(1 + np.trace(A) - A.sum() / n)


def smoother_df(fitted: BaseEstimator, X: np.ndarray) -> float | None:
    """Exact df of a fitted estimator (a `models.classical` TransformedTargetRegressor, or its
    feature pipeline) if its final model is a linear smoother (mean, OLS, ridge, RBF kernel
    ridge), else None.

    The feature steps (scaling, PCA) do not depend on y, so given X the map from the
    (transformed, standardized) target to its fit is linear, with a hat matrix that depends
    on X and the hyperparameters only, not on y.
    """
    pipe = fitted.regressor_ if isinstance(fitted, TransformedTargetRegressor) else fitted
    model = pipe[-1]
    Xt = pipe[:-1].transform(X) if len(pipe) > 1 else np.asarray(X, dtype=np.float64)
    if isinstance(model, DummyRegressor):
        return 1.0
    if isinstance(model, LinearRegression):
        Xc = Xt - Xt.mean(axis=0)
        return float(1 + np.linalg.matrix_rank(Xc))
    if isinstance(model, Ridge):
        return ridge_df(Xt, model.alpha)
    if isinstance(model, KernelRidge) and model.kernel == "rbf":
        return kernel_ridge_df(rbf_kernel(Xt, gamma=model.gamma), model.alpha)
    return None


def effective_df(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                 n_rep: int = 20, n_jobs: int = -1) -> tuple[float, float, str]:
    """(df, standard error, method) of `estimator` fit to (X, y), in the space the model is
    fit in: the transformed target z = g(y) of a TransformedTargetRegressor.

    Measuring in z-space keeps every model comparable and every perturbation valid (adding
    noise to μ itself could push it below 0, where √μ is undefined). Exact tr(S) when the
    final model is a linear smoother, whose hat matrix does not depend on the target at all;
    otherwise Ye's Monte-Carlo estimate on z.
    """
    y = np.asarray(y, dtype=np.float64)
    if isinstance(estimator, TransformedTargetRegressor):
        z = clone(estimator.transformer).fit_transform(y.reshape(-1, 1)).ravel()
        regressor = clone(estimator.regressor)
    else:
        z, regressor = y, clone(estimator)
    exact = smoother_df(clone(regressor).fit(X, z), X)
    if exact is not None:
        return exact, 0.0, "exact tr(S)"
    df, se = monte_carlo_df(regressor, X, z, rng, n_rep=n_rep, n_jobs=n_jobs)
    return df, se, "Monte Carlo (Ye)"


def _fit_predict_perturbed(estimator, X, y, delta):
    return clone(estimator).fit(X, y + delta).predict(X)


def monte_carlo_df(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   rng: np.random.Generator, n_rep: int = 20, tau: float | None = None,
                   n_jobs: int = -1) -> tuple[float, float]:
    """Ye's generalized degrees of freedom: (df estimate, standard error).

    Fits `estimator` to y + δ for `n_rep` draws δ ~ N(0, τ²I) (τ defaults to 0.5·sd(y))
    and estimates df = Σᵢ cov(ŷᵢ, δᵢ)/τ². The estimator's own randomness should be fixed
    (random_state), so δ is the only thing that changes between fits. The standard error
    comes from splitting the sum over samples into 10 blocks.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    tau = float(tau if tau is not None else 0.5 * y.std())
    deltas = rng.normal(0.0, tau, size=(n_rep, len(y)))
    preds = np.array(Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_fit_predict_perturbed)(estimator, X, y, d) for d in deltas))
    dc, pc = deltas - deltas.mean(axis=0), preds - preds.mean(axis=0)
    per_sample = (dc * pc).sum(axis=0) / ((n_rep - 1) * tau**2)  # slope of ŷᵢ on δᵢ
    blocks = np.array([b.sum() for b in np.array_split(per_sample, 10)])
    return float(per_sample.sum()), float(np.sqrt(10) * blocks.std(ddof=1))
