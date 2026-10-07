"""Classical regressors as scikit-learn pipelines with their CV grids (PLAN §6).

Every model is TransformedTargetRegressor(target transform, Pipeline(scaling → [PCA(k)] →
regressor)). Scaling and target options are in `preprocess` (explore_03 chooses them by
CV). Because every fitted step sits inside the estimator, cloning and fitting it on a fold's
training part refits all of them there: the compressed variant and the target transform are
fit on each training set (and fold) only (CLAUDE.md rule 3). Hyperparameter names carry the
wrapper's prefix, e.g. "regressor__model__alpha".

Models:
- mean:    the training-set mean (PLAN §6.1);
- linear:  ordinary least squares, no regularization;
- ridge:   L2-penalized least squares;
- rbf_krr: RBF kernel ridge (PLAN §6.2), the classical counterpart of the quantum kernel
           model, which has the same form with the RBF kernel swapped for a circuit fidelity;
- rf:      random forest (PLAN §6.3);
- xgb:     gradient-boosted trees (XGBoost).

Targets: μ in debye, transformed by `target` (default: standardized, which gives kernel
ridge, a model without an intercept, the training mean as its baseline) and inverted after
prediction, so predictions are in debye.

Reductions (exploration X2, Track B): `reduction=(method, k)` inserts, after scaling, one of
- "pca":   the first k principal components (unsupervised);
- "pls":   k partial-least-squares components, directions chosen for covariance with the
           (transformed) target, so supervised;
- "kbest": the k features with the highest mutual information with the target (supervised);
then re-standardizes the k outputs, so every input of a matched classical model sees the
same numbers a k-qubit angle encoding would. All are fit inside the pipeline, on each
training set and fold only.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from qm9dipole.preprocess import scaling_steps, target_transformer

#: Components kept by the "compressed" variant: one per qubit of the 8-qubit feature map.
N_COMPONENTS = 8

MODELS: tuple[str, ...] = ("mean", "linear", "ridge", "rbf_krr", "rf", "xgb")

#: Hyperparameter grids, searched by CV MAE (keys address the pipeline's "model" step).
GRIDS: dict[str, dict[str, list]] = {
    "mean": {},
    "linear": {},
    "ridge": {"model__alpha": np.logspace(-4, 3, 15).tolist()},
    "rbf_krr": {"model__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "model__gamma": np.logspace(-3, 1, 9).tolist()},
    "rf": {"model__max_features": [1.0, 0.5, "sqrt"], "model__min_samples_leaf": [1, 3, 5]},
    "xgb": {"model__max_depth": [3, 5, 7], "model__learning_rate": [0.03, 0.1],
            "model__min_child_weight": [1, 5]},
}


#: Reduction methods for the compressed (quantum-comparable) inputs.
REDUCTIONS: tuple[str, ...] = ("pca", "pls", "kbest")


class PLSScores(TransformerMixin, BaseEstimator):
    """Partial-least-squares x-scores as a pipeline step (sklearn's PLSRegression returns a
    tuple from fit_transform, which a Pipeline cannot pass on)."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit(self, X, y):
        k = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pls_ = PLSRegression(n_components=k, scale=False).fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return self.pls_.transform(X)


def reduction_steps(method: str, k: int) -> list[tuple[str, object]]:
    """Pipeline steps reducing scaled features to k inputs (see REDUCTIONS), re-standardized."""
    match method:
        case "pca":
            step = PCA(n_components=k)
        case "pls":
            step = PLSScores(n_components=k)
        case "kbest":
            step = SelectKBest(partial(mutual_info_regression, random_state=0), k=k)
        case _:
            raise KeyError(f"unknown reduction {method!r}; choose from {REDUCTIONS}")
    return [("reduce", step), ("rescale", StandardScaler())]


#: Kernel-ridge penalties for the dimension-scaled grid (shared with the quantum kernels).
KRR_ALPHAS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def rbf_grid(n_features: int) -> dict[str, list]:
    """Kernel-ridge grid whose RBF bandwidths scale with the input dimension: γ = c / d for
    c in logspace(−3, 2, 11), α in KRR_ALPHAS. Squared distances between standardized points
    grow like 2d, so a fixed γ grid suits only one dimension (X1's grid assumed tens of
    features). Widened on 2026-10-07 at the smooth end (c from 1e-2 down to 1e-3, α up to 10)
    together with the quantum kernels' grids, after optima at both edges (DECISIONS.md)."""
    return {"model__alpha": list(KRR_ALPHAS),
            "model__gamma": (np.logspace(-3, 2, 11) / max(n_features, 1)).tolist()}


def _regressor(name: str, seed: int):
    # Tree models run single-threaded: the CV harness parallelizes over (candidate, fold).
    match name:
        case "mean":
            return DummyRegressor(strategy="mean")
        case "linear":
            return LinearRegression()
        case "ridge":
            return Ridge()
        case "rbf_krr":
            return KernelRidge(kernel="rbf")
        case "rf":
            return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1)
        case "xgb":
            return XGBRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", random_state=seed, n_jobs=1)
    raise KeyError(f"unknown model {name!r}; choose from {MODELS}")


def build(name: str, seed: int, compressed: bool = False, n_components: int = N_COMPONENTS,
          scaling: str = "standard", target: str = "standard", reduction: tuple[str, int] | None = None,
          n_features: int | None = None) -> tuple[TransformedTargetRegressor, dict]:
    """(estimator, grid) for model `name`; `seed` fixes the tree models' randomness.

    The estimator's `regressor` is the feature pipeline (steps "scale"/"log"/..., optional
    "pca" (X1's compressed variant) or "reduce" + "rescale" (`reduction`), then "model"); after
    fitting, the fitted copy is `regressor_`. With `n_features` (or a reduction), kernel ridge
    uses the dimension-scaled bandwidth grid `rbf_grid`.
    """
    steps = [*scaling_steps(scaling)]
    if compressed:
        steps.append(("pca", PCA(n_components=n_components)))
    if reduction is not None:
        steps += reduction_steps(*reduction)
    steps.append(("model", _regressor(name, seed)))
    estimator = TransformedTargetRegressor(regressor=Pipeline(steps),
                                           transformer=target_transformer(target), check_inverse=False)
    grid = GRIDS[name]
    d = reduction[1] if reduction is not None else n_features
    if name == "rbf_krr" and d is not None:
        grid = rbf_grid(d)
    return estimator, {f"regressor__{k}": v for k, v in grid.items()}


def specs(seed: int, compressed: bool = False, models: tuple[str, ...] = MODELS,
          scaling: str = "standard", target: str = "standard") -> dict[str, tuple[TransformedTargetRegressor, dict]]:
    """{model name: (estimator, grid)} for `evaluate.learning_curve`."""
    return {m: build(m, seed, compressed, scaling=scaling, target=target) for m in models}
