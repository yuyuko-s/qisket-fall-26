"""Metrics, cross-validation and the learning-curve harnesses (PLAN §6, §9; used from M3 on).

Nothing here reads a test set (CLAUDE.md rule 2). Callers pass training data only, and every
number this module returns is a cross-validation estimate inside it, or (exploration X2,
`dev_curve`) a score on a development set held out from training:
- hyperparameters are chosen by CV MAE, with the same folds for every model (PLAN §4 note);
- fitted transforms (scaler, PCA) live inside the model pipeline, so `clone` + `fit` refits
  them on each fold's training part (rule 3);
- negative predictions are clipped at 0 before scoring, since |μ| ≥ 0 (PLAN §6.5 default).

The best-of-grid CV MAE is slightly optimistic (the grid was chosen to minimize it), more so
for larger grids. The single final evaluation on the test sets (M7) is the unbiased number.

Units: μ and every error in debye.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupKFold, KFold, ParameterGrid

Folds = list[tuple[np.ndarray, np.ndarray]]

# Parallel jobs use max_nbytes=None: the arrays are small, and joblib's memory-mapped
# temporary folders only produce clean-up errors from its resource tracker on Windows.


def clip_predictions(pred: np.ndarray) -> np.ndarray:
    """Clip predictions at 0 D (PLAN §6.5): a dipole magnitude cannot be negative."""
    return np.clip(pred, 0.0, None)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error, debye."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    """Root-mean-square error, debye."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def kfold(n: int, seed: int, n_splits: int = 5) -> Folds:
    """PLAN §4 default folds: KFold(5, shuffle=True, random_state=seed) over n samples."""
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n)))


def holdout_split(n: int, seed: int, val_frac: float = 0.1, max_val: int = 5000,
                  min_val: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(train, validation) positions for an inner validation split of n training samples:
    round(val_frac·n) samples, at least `min_val` and at most `max_val`, drawn with `seed`."""
    n_val = int(min(max_val, max(min_val, round(val_frac * n))))
    perm = np.random.default_rng([seed, 7919]).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def group_kfold(groups: np.ndarray, seed: int, n_splits: int = 5) -> Folds:
    """Folds that never split a group, e.g. a formula: every validation molecule's formula
    is absent from that fold's training part, as for the unseen-formula test set."""
    groups = np.asarray(groups)
    cv = GroupKFold(n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.arange(len(groups)), groups=groups))


@dataclass
class CVResult:
    """Cross-validated performance of one estimator at its CV-best hyperparameters."""

    params: dict
    mae: float
    rmse: float
    oof: np.ndarray  # clipped out-of-fold predictions at `params`, one per sample (debye; NaN if never validated)
    fold_mae: np.ndarray  # MAE of each fold at `params`
    candidate_mae: np.ndarray = field(repr=False)  # pooled CV MAE of every grid candidate
    candidates: list[dict] = field(repr=False)
    seconds: float = 0.0


def _fit_predict(estimator: BaseEstimator, params: dict, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray) -> np.ndarray:
    model = clone(estimator).set_params(**params)
    model.fit(X_train, y_train)
    return model.predict(X_val)


def _split_feature_steps(estimator: BaseEstimator, candidates: list[dict]):
    """(feature steps, estimator with only the final model) if every candidate tunes only
    the final "model" step of a TransformedTargetRegressor(Pipeline); else None.

    The feature steps (scaling, PCA) never see y and are unaffected by the candidates, so
    fitting them once per fold gives exactly the same predictions as refitting them inside
    every candidate, at a fraction of the cost (a Yeo-Johnson fit takes ~2 s).
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    if not isinstance(estimator, TransformedTargetRegressor):
        return None
    pipe = estimator.regressor
    if not isinstance(pipe, Pipeline) or len(pipe) < 2 or pipe.steps[-1][0] != "model":
        return None
    if not all(key.startswith("regressor__model__") for c in candidates for key in c):
        return None
    head = clone(estimator).set_params(regressor=Pipeline([pipe.steps[-1]]))
    return pipe[:-1], head


def _fit_transform_fold(steps, estimator, X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray):
    """Fit the feature steps on a fold's training part, with the target they would see inside
    the TransformedTargetRegressor (its transformer fit on the same part), so supervised steps
    such as partial least squares behave exactly as in a full refit."""
    y_train = y[train].reshape(-1, 1)
    transformer = getattr(estimator, "transformer", None)
    if transformer is not None:
        y_train = clone(transformer).fit(y_train).transform(y_train)
    fitted = clone(steps).fit(X[train], np.asarray(y_train).ravel())
    return fitted.transform(X[train]), fitted.transform(X[val])


def tune(estimator: BaseEstimator, grid: dict | list[dict] | None, X: np.ndarray, y: np.ndarray,
         folds: Folds, n_jobs: int = -1, require_partition: bool = True) -> CVResult:
    """Grid search by CV MAE. Every candidate sees the same folds; folds must partition X
    unless `require_partition` is False (e.g. one inner validation split, `holdout_split`),
    in which case only the validated samples are scored.

    Each (candidate, fold) fit runs as one parallel job. When the candidates only change the
    final model, the feature steps are fit once per fold (on that fold's training part, as
    always) and shared by all candidates. Out-of-fold predictions are kept for every
    candidate, so the winner's MAE and RMSE come from pooled out-of-fold predictions (each
    sample predicted exactly once) with no refit.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    covered = np.sort(np.concatenate([val for _, val in folds]))
    if len(np.unique(covered)) != len(covered):
        raise ValueError("a sample is validated more than once")
    if require_partition and not np.array_equal(covered, np.arange(len(y))):
        raise ValueError("folds must partition the samples (each validated exactly once)")
    candidates = list(ParameterGrid(grid)) if grid else [{}]

    start = time.perf_counter()
    split = _split_feature_steps(estimator, candidates)
    if split is None:
        fold_data = [(X[train], X[val]) for train, val in folds]
        model = estimator
    else:
        steps, model = split
        fold_data = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_fit_transform_fold)(steps, estimator, X, y, train, val) for train, val in folds)
    jobs = [(c, f) for c in range(len(candidates)) for f in range(len(folds))]
    preds = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_fit_predict)(model, candidates[c], fold_data[f][0], y[folds[f][0]], fold_data[f][1])
        for c, f in jobs
    )
    jobs = [(c, folds[f][0], folds[f][1]) for c, f in jobs]
    oof = np.full((len(candidates), len(y)), np.nan)
    for (c, _, val), pred in zip(jobs, preds):
        oof[c, val] = pred
    oof = clip_predictions(oof)

    candidate_mae = np.abs(oof[:, covered] - y[covered]).mean(axis=1)
    best = int(np.argmin(candidate_mae))  # ties -> first candidate, so the choice is stable
    return CVResult(
        params=candidates[best], mae=float(candidate_mae[best]), rmse=rmse(y[covered], oof[best, covered]),
        oof=oof[best], fold_mae=np.array([mae(y[val], oof[best, val]) for _, val in folds]),
        candidate_mae=candidate_mae, candidates=candidates,
        seconds=time.perf_counter() - start,
    )


def _jsonable(params: dict) -> str:
    return json.dumps({k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()},
                      sort_keys=True)


SpecFactory = Callable[[int, str], dict[str, tuple[BaseEstimator, dict | None]]]
FoldFactory = Callable[[np.ndarray, int], Folds]


def _default_folds(ids: np.ndarray, seed: int) -> Folds:
    return kfold(len(ids), seed)


def learning_curve(
    make_specs: SpecFactory,
    features: dict[str, pd.DataFrame],
    y: pd.Series,
    train_sets: dict[int, dict[int, np.ndarray]],
    make_folds: FoldFactory = _default_folds,
    eval_set: str = "cv",
    n_jobs: int = -1,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """CV-only learning curves over nested training sets (PLAN §9.2 long format).

    make_specs: (seed, feature-set name) -> {model name: (pipeline, grid)}. The seed fixes
                model randomness; the feature-set name can select a variant (e.g. PCA).
    features:   {feature-set name: DataFrame indexed by molecule ID}; must cover every ID
                in `train_sets`.
    y:          μ in debye, indexed by molecule ID.
    train_sets: {seed: {N: training IDs}}, e.g. `Splits.train`.
    make_folds: (training IDs, seed) -> folds. Default: KFold(5, shuffle, random_state=seed),
                the PLAN §4 default. Pass formula-grouped folds to mimic unseen formulas.
    eval_set:   label written to the `eval_set` column, e.g. "cv" or "cv_by_formula".

    For each (seed, N), one set of folds is shared by every model and feature set (paired
    comparison). Returns (results, out-of-fold predictions keyed by (seed, N, model,
    features)).
    """
    rows, oof = [], {}
    for seed, by_n in train_sets.items():
        for n, ids in sorted(by_n.items()):
            ids = np.asarray(ids)
            folds = make_folds(ids, seed)
            y_s = y.loc[ids].to_numpy()
            for fname, F in features.items():
                X = F.loc[ids].to_numpy()
                scores = {}
                for mname, (estimator, grid) in make_specs(seed, fname).items():
                    r = tune(estimator, grid, X, y_s, folds, n_jobs=n_jobs)
                    oof[(seed, n, mname, fname)] = pd.Series(r.oof, index=ids)
                    scores[mname] = (r.mae, r.seconds)
                    rows.append({
                        "seed": seed, "n_train": n, "model": mname, "features": fname,
                        "eval_set": eval_set, "mae_D": r.mae, "rmse_D": r.rmse,
                        "mae_fold_std_D": float(r.fold_mae.std(ddof=1)), "n_eval": len(ids),
                        "n_candidates": len(r.candidates), "hyperparams": _jsonable(r.params),
                        "seconds": round(r.seconds, 2),
                    })
                if progress:
                    best = min(scores, key=lambda m: scores[m][0])
                    print(f"seed {seed}  N={n:<5} {fname:<22} best {best:<8} MAE {scores[best][0]:.3f} D  "
                          f"({sum(s for _, s in scores.values()):5.1f} s for {len(scores)} models)", flush=True)
    return pd.DataFrame(rows), oof


def permutation_importance_cv(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                              folds: Folds, rng: np.random.Generator,
                              n_repeats: int = 5) -> np.ndarray:
    """Mean rise in validation MAE (debye) when one feature column is shuffled.

    The model is fit on each fold's training part and scored on its validation part, so the
    importances measure what the model uses on molecules it has not seen. Returns an array
    of shape (n_features,), averaged over folds and repeats.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = np.zeros(X.shape[1])
    for train, val in folds:
        model = clone(estimator).fit(X[train], y[train])
        base = mae(y[val], clip_predictions(model.predict(X[val])))
        for j in range(X.shape[1]):
            rises = []
            for _ in range(n_repeats):
                Xv = X[val].copy()
                Xv[:, j] = rng.permutation(Xv[:, j])
                rises.append(mae(y[val], clip_predictions(model.predict(Xv))) - base)
            out[j] += np.mean(rises)
    return out / len(folds)


# --------------------------------------------------------------------------------------
# Exploration X2: fit on a training set, score on development sets (never a test set).
# --------------------------------------------------------------------------------------

#: |μ| at and above this (debye) is the zwitterion tail that dominates RMSE (explore_01).
TAIL_D = 9.0


def scores(y: np.ndarray, pred: np.ndarray, tail: float = TAIL_D) -> dict[str, float]:
    """MAE, RMSE, RMSE below the |μ| tail, R² (against the evaluation set's own mean) and bias
    (mean of pred − y), all in debye except R²."""
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    err = pred - y
    below = y < tail
    return {"mae_D": float(np.abs(err).mean()), "rmse_D": float(np.sqrt((err**2).mean())),
            "rmse_below_tail_D": float(np.sqrt((err[below] ** 2).mean())),
            "r2": float(1 - (err**2).mean() / y.var()), "bias_D": float(err.mean())}


def dev_curve(make_fitters: Callable[[int, int], dict], train_sets: dict[int, dict[int, np.ndarray]],
              eval_sets: dict[str, np.ndarray], y: pd.Series, sizes: list[int] | None = None,
              seeds: list[int] | None = None, cache_dir=None, progress: bool = True) -> tuple[pd.DataFrame, dict]:
    """Learning curves scored on development sets (exploration X2, Track A).

    make_fitters: (seed, N) -> {model name: fitter} (models/fitters.py); each fitter tunes inside
                  the training IDs it is given and nowhere else.
    train_sets:   {seed: {N: training IDs}}, e.g. `Splits.train`.
    eval_sets:    {name: IDs}, e.g. {"dev": splits.dev, "dev_unseen": splits.dev_unseen}. Callers
                  must never pass a test set.
    cache_dir:    optional folder for finished (seed, N, model) runs, keyed by the fitter's config
                  hash; pass a folder specific to the code version (e.g. one per git commit) so a
                  long run can resume after an interruption with identical results.

    Returns (one row per seed × N × model × evaluation set, predictions keyed by (seed, N, model)
    then evaluation-set name, as Series indexed by ID).
    """
    import pickle
    from pathlib import Path

    from qm9dipole.provenance import config_hash

    rows, predictions = [], {}
    for seed in (seeds if seeds is not None else list(train_sets)):
        by_n = train_sets[seed]
        for n in sorted(by_n):
            if sizes is not None and n not in sizes:
                continue
            ids = np.asarray(by_n[n])
            y_train = y.loc[ids].to_numpy()
            for name, fitter in make_fitters(seed, n).items():
                path = None
                if cache_dir is not None:
                    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", name)  # e.g. "ridge|all_legal": | is illegal on Windows
                    path = Path(cache_dir) / f"{safe}_s{seed}_n{n}_{config_hash(fitter.config)}.pkl"
                if path is not None and path.exists():
                    record = pickle.loads(path.read_bytes())
                else:
                    start = time.perf_counter()
                    fitter.fit(ids, y_train)
                    fit_seconds = time.perf_counter() - start
                    record = {"describe": fitter.describe(), "fit_seconds": fit_seconds,
                              "pred": {k: np.asarray(fitter.predict(v)) for k, v in eval_sets.items()}}
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(pickle.dumps(record))
                predictions[(seed, n, name)] = {k: pd.Series(record["pred"][k], index=np.asarray(v))
                                                for k, v in eval_sets.items()}
                for set_name, eval_ids in eval_sets.items():
                    rows.append({"seed": seed, "n_train": n, "model": name, "eval_set": set_name,
                                 **scores(y.loc[eval_ids].to_numpy(), record["pred"][set_name]),
                                 "n_eval": len(eval_ids), "fit_seconds": round(record["fit_seconds"], 1),
                                 "details": json.dumps(record["describe"], default=str)})
                if progress:
                    first = next(iter(eval_sets))
                    mae_first = rows[-len(eval_sets)]["mae_D"]
                    print(f"seed {seed}  N={n:<6} {name:<16} {first} MAE {mae_first:.3f} D  "
                          f"({record['fit_seconds']:.0f} s)", flush=True)
    return pd.DataFrame(rows), predictions
