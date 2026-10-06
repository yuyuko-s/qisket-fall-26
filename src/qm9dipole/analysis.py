"""Exploratory statistics for the classical analysis (notebooks/explore_*.ipynb).

Nothing here reads a test set: callers pass training-pool data (or a training set) only.
Units: μ and every error in debye.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qm9dipole.evaluate import Folds, tune
from qm9dipole.models import classical


def formula_variance(y: pd.Series, formula: pd.Series, min_count: int = 5) -> dict[str, float]:
    """How much of var(μ) the molecular formula explains (one-way ANOVA by formula).

    Uses formulas with ≥ `min_count` molecules, so each formula mean is reasonably estimated.
    - eta2: SS_between / SS_total, the in-sample share of variance explained by the formula
      mean (biased upward, since every mean is fit to its own molecules);
    - omega2: the less biased estimate (SS_b − (k−1)·MS_w) / (SS_t + MS_w);
    - within_std: pooled standard deviation of μ among isomers of one formula;
    - within_mae: mean |μ − its formula's mean|, an in-sample floor for any model that sees
      composition only (it cannot tell isomers apart).
    """
    keep = formula.map(formula.value_counts()) >= min_count
    y, formula = y[keep].astype(float), formula[keep]
    means = y.groupby(formula).transform("mean")
    n, k = len(y), formula.nunique()
    ss_total = float(((y - y.mean()) ** 2).sum())
    ss_within = float(((y - means) ** 2).sum())
    ms_within = ss_within / (n - k)
    return {
        "n_molecules": n, "n_formulas": k,
        "eta2": 1 - ss_within / ss_total,
        "omega2": (ss_total - ss_within - (k - 1) * ms_within) / (ss_total + ms_within),
        "total_std": float(y.std(ddof=1)), "within_std": float(np.sqrt(ms_within)),
        "within_mae": float((y - means).abs().mean()),
    }


def pca_cv_curve(X: np.ndarray, y: np.ndarray, ks: list[int], folds: Folds, model: str = "ridge",
                 seed: int = 0, n_jobs: int = -1, scaling: str = "standard",
                 target: str = "standard") -> pd.DataFrame:
    """CV error of `model` on the first k principal components of scaled X, per k.

    Scaling and PCA sit inside the pipeline, so they are refit on every fold's training
    part. A final row with k = X.shape[1] and `pca=False` gives the uncompressed reference.
    """
    rows = []
    opts = {"scaling": scaling, "target": target}
    for k in ks:
        r = tune(*classical.build(model, seed, compressed=True, n_components=k, **opts), X, y, folds, n_jobs)
        rows.append({"k": k, "pca": True, "mae_D": r.mae, "rmse_D": r.rmse, "params": r.params})
    r = tune(*classical.build(model, seed, **opts), X, y, folds, n_jobs)
    rows.append({"k": X.shape[1], "pca": False, "mae_D": r.mae, "rmse_D": r.rmse, "params": r.params})
    return pd.DataFrame(rows)


def summarize_curves(results: pd.DataFrame, metric: str = "mae_D",
                     by: tuple[str, ...] = ("model", "features", "n_train")) -> pd.DataFrame:
    """Mean and standard deviation of `metric` over seeds, one row per `by` group."""
    g = results.groupby(list(by))[metric]
    return g.agg(mean="mean", std="std", n_seeds="count").reset_index()


def loglog_slope(n: np.ndarray, err: np.ndarray) -> float:
    """Slope of log(error) against log(N): the learning-curve exponent (≈ −0.5 is typical
    for kernel models on smooth targets; 0 means more labels do not help)."""
    return float(np.polyfit(np.log(n), np.log(err), 1)[0])


def error_by_group(y: pd.Series, pred: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """n, MAE, RMSE and bias (mean of pred − y) of predictions within each group."""
    df = pd.DataFrame({"err": pred - y, "group": groups})
    out = df.groupby("group", observed=True)["err"].agg(
        n="count", mae_D=lambda e: e.abs().mean(), rmse_D=lambda e: np.sqrt((e**2).mean()),
        bias_D="mean")
    return out.reset_index()
