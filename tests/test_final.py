"""Test-set evaluation helpers (final.py): the run log, features for molecules outside the
development universe, and finite-shot scoring of fitted quantum models."""

import numpy as np
import pandas as pd
import pytest

from qm9dipole.data import QM9_PARQUET
from qm9dipole.explore import FEATURES_PATH
from qm9dipole.final import evaluation_frame, legal_features, log_run, next_run_number, shot_scores
from qm9dipole.models.fitters import TabularFitter
from qm9dipole.models.quantum_kernel import FidelityKernel, KernelRidgeFitter, ProjectedKernel, quantum_ridge_build


def test_run_log_numbers_runs_and_appends(tmp_path):
    log = tmp_path / "test_runs.csv"
    assert next_run_number(log) == 1
    log_run({"run": 1, "commit": "abc", "note": "first"}, log)
    log_run({"run": 2, "commit": "def", "note": "second, with a, comma"}, log)
    runs = pd.read_csv(log)
    assert runs["run"].tolist() == [1, 2] and runs["note"].iloc[1] == "second, with a, comma"
    assert next_run_number(log) == 3


needs_data = pytest.mark.skipif(not (QM9_PARQUET.exists() and FEATURES_PATH.exists()),
                                reason="qm9.parquet or explore_features.parquet not built")


@pytest.fixture(scope="module")
def stored():
    from qm9dipole.data import load_qm9_table
    from qm9dipole.explore import feature_sets, load_catalog, load_features

    features = load_features()
    return features, load_qm9_table(), feature_sets(load_catalog())["all_legal"]


@needs_data
def test_recomputed_features_match_the_stored_table(stored):
    # Test molecules get their features from the same descriptor code as the stored table;
    # recomputing development molecules must reproduce it (float32 storage precision).
    features, table, cols = stored
    ids = np.random.default_rng(0).choice(features.index.to_numpy(), 25, replace=False)
    new = legal_features(table, ids, cols, n_jobs=1)
    assert list(new.columns) == cols and new.dtypes.eq(np.float64).all()
    np.testing.assert_allclose(new.loc[ids].to_numpy(), features.loc[ids, cols].to_numpy(np.float64), rtol=1e-5, atol=1e-4)


@needs_data
def test_evaluation_frame_adds_only_missing_molecules(stored):
    features, table, cols = stored
    outside = np.setdiff1d(table["id"].to_numpy(), features.index.to_numpy())[:7]
    inside = features.index.to_numpy()[:5]
    frame = evaluation_frame(features.iloc[:50], table, np.r_[inside, outside], cols, n_jobs=1)
    assert len(frame) == 50 + 7 and frame.index.is_unique
    pd.testing.assert_frame_equal(frame.iloc[:50], features.iloc[:50][cols].astype(np.float64))
    assert np.isfinite(frame.loc[outside].to_numpy()).all()


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = np.clip(2 + np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2, 0.05, None)
    ids = np.arange(100, 220)
    frame = pd.DataFrame(X, index=ids, columns=list("abcd"))
    return frame, pd.Series(y, index=ids), ids[:80], ids[80:]


def _eval(frame, y, ids):
    return {"held": frame.loc[ids].to_numpy()}, {"held": y.loc[ids].to_numpy()}


@pytest.mark.parametrize("kind", ["fidelity", "projected"])
def test_shot_scores_converge_to_exact_with_many_shots(toy, kind):
    frame, y, tr, te = toy
    kern = FidelityKernel("zz", 2, gammas=[0.5]) if kind == "fidelity" else ProjectedKernel("zz", 2, gammas=[0.5], gammas_p=[0.5])
    f = KernelRidgeFitter(frame, kern, 0, scaling="standard", target="sqrt", alphas=(0.1,)).fit(tr, y.loc[tr])
    Xe, ye = _eval(frame, y, te)
    rows = pd.DataFrame(shot_scores(f, frame.loc[tr].to_numpy(), y.loc[tr].to_numpy(), Xe, ye, [10**9], 1, 0, kind))
    exact = rows[rows["mode"] == "exact"]["mae_D"].iloc[0]
    assert exact == pytest.approx(np.mean(np.abs(f.predict(te) - y.loc[te].to_numpy())), abs=1e-9)
    assert np.allclose(rows["mae_D"], exact, atol=2e-3)
    few = pd.DataFrame(shot_scores(f, frame.loc[tr].to_numpy(), y.loc[tr].to_numpy(), Xe, ye, [20], 1, 0, kind))
    assert (few.loc[few["mode"] != "exact", "mae_D"] != exact).all()  # 20 shots must change something


def test_shot_scores_of_the_team_quantum_ridge(toy):
    frame, y, tr, te = toy
    est, grid = quantum_ridge_build(0, "standard", "sqrt", None, 4)
    f = TabularFitter(frame, est, {k: [v[len(v) // 2]] for k, v in grid.items()}, 0).fit(tr, y.loc[tr])
    Xe, ye = _eval(frame, y, te)
    rows = pd.DataFrame(shot_scores(f, frame.loc[tr].to_numpy(), y.loc[tr].to_numpy(), Xe, ye, [10**9], 1, 0, "z_readout"))
    exact = rows[rows["mode"] == "exact"]["mae_D"].iloc[0]
    assert exact == pytest.approx(np.mean(np.abs(f.predict(te) - y.loc[te].to_numpy())), abs=1e-9)
    assert np.allclose(rows["mae_D"], exact, atol=2e-3)
    with pytest.raises(KeyError):
        shot_scores(f, frame.loc[tr].to_numpy(), y.loc[tr].to_numpy(), Xe, ye, [10], 1, 0, "unknown")
