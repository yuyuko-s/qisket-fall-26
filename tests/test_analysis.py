"""Exploratory-statistics tests (analysis.py) and named feature frames (descriptors.py)."""

import numpy as np
import pandas as pd
import pytest

from qm9dipole.analysis import (
    error_by_group, formula_variance, loglog_slope, pca_cv_curve, summarize_curves,
)
from qm9dipole.descriptors import feature_frame, feature_names
from qm9dipole.evaluate import kfold


def test_formula_variance_by_hand():
    # Formula A: μ = 1, 3 (mean 2); formula B: μ = 5, 7 (mean 6); C has 1 molecule (dropped).
    y = pd.Series([1.0, 3.0, 5.0, 7.0, 100.0])
    formula = pd.Series(["A", "A", "B", "B", "C"])
    out = formula_variance(y, formula, min_count=2)
    assert out["n_molecules"] == 4 and out["n_formulas"] == 2
    # SS_total = 9+1+1+9 = 20, SS_within = 1+1+1+1 = 4.
    assert out["eta2"] == pytest.approx(1 - 4 / 20)
    assert out["within_std"] == pytest.approx(np.sqrt(4 / 2)) and out["within_mae"] == pytest.approx(1.0)
    assert out["omega2"] < out["eta2"]


def test_formula_variance_is_zero_when_formulas_do_not_matter():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=4000))
    formula = pd.Series(rng.choice(list("ABCDEFGH"), size=4000))
    assert abs(formula_variance(y, formula)["omega2"]) < 0.01


def test_loglog_slope_recovers_a_power_law():
    n = np.array([100, 300, 1000])
    assert loglog_slope(n, 3.0 * n**-0.5) == pytest.approx(-0.5)


def test_summarize_curves_over_seeds():
    df = pd.DataFrame({"model": "m", "features": "f", "n_train": 100, "seed": [0, 1, 2],
                       "mae_D": [1.0, 2.0, 3.0]})
    row = summarize_curves(df).iloc[0]
    assert row["mean"] == 2.0 and row["std"] == 1.0 and row["n_seeds"] == 3


def test_error_by_group():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    pred = pd.Series([2.0, 2.0, 1.0, 4.0])
    out = error_by_group(y, pred, pd.Series(["a", "a", "b", "b"])).set_index("group")
    assert out.loc["a", "mae_D"] == 0.5 and out.loc["a", "bias_D"] == 0.5
    assert out.loc["b", "rmse_D"] == pytest.approx(np.sqrt(2.0)) and out.loc["b", "n"] == 2


def test_pca_cv_curve_improves_with_components():
    # y depends on all 6 directions, so 1 component cannot match the full set.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 6))
    y = X @ np.arange(1, 7) + 30
    curve = pca_cv_curve(X, y, ks=[1, 3], folds=kfold(150, 0), n_jobs=1)
    assert list(curve["k"]) == [1, 3, 6] and list(curve["pca"]) == [True, True, False]
    assert curve["mae_D"].iloc[0] > curve["mae_D"].iloc[-1]


def test_feature_frame_is_indexed_by_id_with_names():
    Z = np.array([6, 1, 1, 1, 1])
    R = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                  [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]], dtype=float)
    table = pd.DataFrame({"id": [7, 9], "Z": [Z, Z], "R": [R, R + 1.0]})
    frame = feature_frame(table, "composition")
    assert list(frame.index) == [7, 9] and list(frame.columns) == feature_names("composition")
    assert frame.loc[9, "n_H"] == 4
    assert feature_frame(table, "cm_spectrum").shape == (2, 29)


def test_rank_features_puts_the_informative_feature_first():
    from qm9dipole.analysis import rank_features

    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    X = pd.DataFrame({"noise": rng.normal(size=2000), "signal": x, "weak": x + 3 * rng.normal(size=2000),
                      "constant": 1.0})
    y = pd.Series(5 - 2 * x + 0.1 * rng.normal(size=2000))
    ranked = rank_features(X, y, n_jobs=1)
    assert ranked.index.tolist() == ["signal", "weak", "noise"]  # constant dropped: no information
    assert ranked["rank"].tolist() == [1, 2, 3]
    assert ranked.loc["signal", "Spearman ρ"] < -0.99  # direction: y falls as x rises
