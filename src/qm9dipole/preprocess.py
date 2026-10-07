"""Feature scaling and target transforms, as pipeline steps (explore_03 compares them).

Every option is an sklearn transformer placed inside the model pipeline, so it is fit on each
training set or fold only (CLAUDE.md rule 3), including choices that depend on the data,
such as which columns count as skewed.

Feature scaling (SCALINGS):
- standard:        z-score (mean 0, sd 1). Sensitive to outliers: one extreme value
                   stretches the scale for everyone.
- robust:          median and inter-quartile range. Outliers do not set the scale, but they
                   stay extreme after it.
- log_standard:    signed log1p on the columns whose training skewness exceeds a threshold,
                   then z-score. Tames long right tails (sizes, moments of inertia).
- yeo_johnson:     per-column power transform fit by maximum likelihood towards normality,
                   then z-score.
- quantile_normal: rank-based map to a standard normal; bounded (≈ ±5.2) and insensitive to
                   outliers, but it discards distances between values.

Guard (every unbounded scaling: standard, robust, log_standard, yeo_johnson): scaled values
are clipped to ±Z_CLIP. Two failures made it necessary, both from new molecules far outside a
training set's range, which a linear model extrapolates:
- a column that is almost constant in a training fold gets a tiny standard deviation. In
  explore_03, radial-distribution bins at physically empty distances (an O–F pair 0.9 Å apart)
  held only the tails of Gaussians, ~1e-11 to 1e-5; with 1–3 non-zero training rows their sd
  was ~1e-12, a validation molecule scored z ≈ 10⁷, and ridge predicted up to 10¹⁹ D;
- a fitted Yeo-Johnson power curve extrapolates steeply: a small molecule's padded
  Coulomb-matrix eigenvalue, outside the training range, scored z ≈ −10⁴ and ridge predicted
  1.3 × 10⁵ D on dev_unseen (S_(0,100), PLS(10)).
Clipping is stateless (a fixed bound, nothing fit), and |z| > 10 cannot occur for more than 1%
of any column's training values (Chebyshev), so it acts only on such outliers. Kernel models
are bounded anyway (an RBF or quantum kernel sends a far-away molecule to the mean); the clip
matters most for linear models.

Target transforms (TARGETS), applied to μ before the model and inverted afterwards, so
predictions stay in debye (CLAUDE.md rule 8):
- standard: centre and scale μ. Linear, so it changes nothing for models with an intercept,
            but kernel ridge has no intercept and would otherwise shrink toward 0 D, not
            toward the mean.
- sqrt:     √μ, then standard. Compresses the long right tail of |μ|.
- log1p:    log(1 + μ), then standard. Compresses it more.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler,
)

SCALINGS: tuple[str, ...] = ("standard", "robust", "log_standard", "yeo_johnson", "quantile_normal")
TARGETS: tuple[str, ...] = ("standard", "sqrt", "log1p")

#: Bound on scaled values after every unbounded scaling (see the module docstring).
Z_CLIP = 10.0


class ClipScaled(TransformerMixin, BaseEstimator):
    """Clip already-scaled features to [−bound, bound]. Stateless: nothing is fit."""

    def __init__(self, bound: float = Z_CLIP):
        self.bound = bound

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float64), -self.bound, self.bound)


class SkewedLog(TransformerMixin, BaseEstimator):
    """Signed log1p, sign(x)·log(1 + |x|), on the columns whose skewness in the training
    data exceeds `threshold` in magnitude; other columns pass through."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            s = np.nan_to_num(skew(X, axis=0), nan=0.0)  # constant column: skew undefined -> 0
        self.columns_ = np.flatnonzero(np.abs(s) > self.threshold)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        c = self.columns_
        X[:, c] = np.sign(X[:, c]) * np.log1p(np.abs(X[:, c]))
        return X


class AdaptiveQuantile(TransformerMixin, BaseEstimator):
    """QuantileTransformer to a normal, with n_quantiles capped at the training-set size
    (a fixed value larger than a small fold warns and is reset anyway)."""

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.qt_ = QuantileTransformer(n_quantiles=min(self.n_quantiles, len(X)),
                                       output_distribution="normal", subsample=None).fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return self.qt_.transform(np.asarray(X, dtype=np.float64))


def scaling_steps(name: str, skew_threshold: float = 1.0) -> list[tuple[str, object]]:
    """Pipeline steps for feature scaling `name` (see SCALINGS)."""
    match name:
        case "standard":
            return [("scale", StandardScaler()), ("clip", ClipScaled())]
        case "robust":
            return [("scale", RobustScaler()), ("clip", ClipScaled())]
        case "log_standard":
            return [("log", SkewedLog(skew_threshold)), ("scale", StandardScaler()), ("clip", ClipScaled())]
        case "yeo_johnson":
            return [("power", PowerTransformer(method="yeo-johnson", standardize=True)), ("clip", ClipScaled())]
        case "quantile_normal":
            return [("quantile", AdaptiveQuantile())]
    raise KeyError(f"unknown scaling {name!r}; choose from {SCALINGS}")


def _square_nonnegative(z):
    return np.clip(z, 0.0, None) ** 2


def target_transformer(name: str) -> Pipeline:
    """Invertible transformer for μ (debye) named `name` (see TARGETS).

    Module-level functions (not lambdas) keep it picklable for parallel CV. The inverse of
    sqrt clips at 0 before squaring, so a negative prediction in √μ space maps to 0 D rather
    than to a positive value.
    """
    match name:
        case "standard":
            steps = []
        case "sqrt":
            steps = [("f", FunctionTransformer(np.sqrt, inverse_func=_square_nonnegative,
                                               check_inverse=False))]
        case "log1p":
            steps = [("f", FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False))]
        case _:
            raise KeyError(f"unknown target transform {name!r}; choose from {TARGETS}")
    return Pipeline([*steps, ("scale", StandardScaler())])
