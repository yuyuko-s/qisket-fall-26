"""Models that train on a set of molecule IDs and predict any other IDs (exploration X2).

Every fitter has the same small interface, so one learning-curve harness
(`evaluate.dev_curve`) can drive a mean baseline, a tuned XGBoost and the latent-charge network
alike:

    fitter.fit(ids, y)      # train on these molecules (y = |μ| in debye); tune inside them only
    fitter.predict(ids)     # |μ| in debye, clipped at 0
    fitter.describe()       # what was chosen (hyperparameters, epochs, parameter counts, ...)
    fitter.config           # the settings that define it (JSON-able; part of the cache key)

Tuning never looks outside the training IDs (docs/EVALUATION_RULES.md rule 2): 5-fold CV for small training
sets (the X1 / Track B protocol), an inner validation split for large ones. Every fitted
transform lives inside the estimator, so it is refit on each training set and fold (rule 3).
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid

from qm9dipole.evaluate import clip_predictions, holdout_split, kfold, tune


def _json(d: dict) -> dict:
    return json.loads(json.dumps(d, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))


def stable_repr(estimator: BaseEstimator) -> str:
    """repr(estimator) without memory addresses, for cache keys. A FunctionTransformer prints its
    functions as "<function f at 0x…>", whose address changes in every process, so the raw repr
    of a √μ-target pipeline gave a different cache key in every run."""
    return re.sub(r" at 0x[0-9A-Fa-f]+", "", repr(estimator))


class MeanFitter:
    """Predict the training-set mean (PLAN §6.1)."""

    config = {"kind": "mean"}

    def fit(self, ids, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, ids):
        return np.full(len(ids), self.mean_)

    def describe(self) -> dict:
        return {"mean_D": self.mean_}


class TabularFitter:
    """A scikit-learn estimator on a molecule-level feature frame (indexed by ID).

    Hyperparameters are chosen from `grid` by 5-fold CV MAE (`evaluate.tune`, shared folds
    seeded by `seed`) when the training set has at most `cv_max_n` molecules, and by an inner
    validation split otherwise; the winner is then refit on the whole training set.
    """

    def __init__(self, frame: pd.DataFrame, estimator: BaseEstimator, grid: dict | None, seed: int,
                 cv_max_n: int = 1000, val_frac: float = 0.1, max_val: int = 5000, n_jobs: int = -1,
                 label: str = ""):
        self.frame, self.estimator, self.grid, self.seed = frame, estimator, grid, seed
        self.cv_max_n, self.val_frac, self.max_val, self.n_jobs = cv_max_n, val_frac, max_val, n_jobs
        self.config = _json({"kind": "tabular", "label": label, "features": list(frame.columns),
                             "estimator": stable_repr(estimator), "grid": grid, "seed": seed, "cv_max_n": cv_max_n,
                             "val_frac": val_frac, "max_val": max_val})

    def _X(self, ids) -> np.ndarray:
        return self.frame.loc[ids].to_numpy(dtype=np.float64)

    def fit(self, ids, y):
        X, y = self._X(ids), np.asarray(y, dtype=np.float64)
        n_candidates = len(ParameterGrid(self.grid)) if self.grid else 1
        if n_candidates == 1:
            self.params_, self.tuning_ = (list(ParameterGrid(self.grid))[0] if self.grid else {}), "none"
        else:
            if len(y) <= self.cv_max_n:
                folds, self.tuning_ = kfold(len(y), self.seed), "5-fold CV"
            else:
                folds, self.tuning_ = [holdout_split(len(y), self.seed, self.val_frac, self.max_val)], "holdout"
            r = tune(self.estimator, self.grid, X, y, folds, n_jobs=self.n_jobs, require_partition=len(folds) > 1)
            self.params_, self.tuning_mae_ = r.params, r.mae
        self.model_ = clone(self.estimator).set_params(**self.params_).fit(X, y)
        return self

    def predict(self, ids):
        return clip_predictions(self.model_.predict(self._X(ids)))

    def describe(self) -> dict:
        out = {"params": _json(self.params_), "tuning": self.tuning_}
        if hasattr(self, "tuning_mae_"):
            out["tuning_mae_D"] = self.tuning_mae_
        return out


#: Randomized-search space for gradient-boosted trees: (name, kind, low, high).
XGB_SPACE: tuple[tuple[str, str, float, float], ...] = (
    ("max_depth", "int", 3, 10),
    ("learning_rate", "log", 0.01, 0.2),
    ("min_child_weight", "log", 1.0, 50.0),
    ("subsample", "uniform", 0.5, 1.0),
    ("colsample_bytree", "uniform", 0.3, 1.0),
    ("reg_lambda", "log", 1e-2, 100.0),
    ("reg_alpha", "log", 1e-3, 10.0),
)


def sample_params(space, rng: np.random.Generator) -> dict:
    out = {}
    for name, kind, lo, hi in space:
        if kind == "int":
            out[name] = int(rng.integers(lo, hi + 1))
        elif kind == "log":
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[name] = float(rng.uniform(lo, hi))
    return out


class XGBFitter:
    """XGBoost with a randomized hyperparameter search and early stopping, inside the training set.

    Each of `n_iter` sampled settings is trained on every inner fold (5-fold CV up to `cv_max_n`
    molecules, one inner validation split above) with early stopping on that fold's validation
    part; the setting with the lowest pooled validation MAE (debye) wins. The final model is
    refit on the whole training set with the mean best number of trees, scaled up by the ratio
    of the full size to the inner training size. The target is transformed with
    `preprocess.target_transformer(target)` (fit on the training labels) and inverted, so
    predictions are in debye. Trees are invariant to monotone feature scaling, so features are
    used as they are.
    """

    def __init__(self, frame: pd.DataFrame, seed: int, n_iter: int = 20, target: str = "standard",
                 cv_max_n: int = 1000, val_frac: float = 0.1, max_val: int = 5000, max_trees: int = 4000,
                 early_stopping: int = 100, n_jobs: int = -1, space=XGB_SPACE, label: str = ""):
        self.frame, self.seed, self.n_iter, self.target = frame, seed, n_iter, target
        self.cv_max_n, self.val_frac, self.max_val = cv_max_n, val_frac, max_val
        self.max_trees, self.early_stopping, self.n_jobs, self.space = max_trees, early_stopping, n_jobs, space
        self.config = _json({"kind": "xgb", "label": label, "features": list(frame.columns), "seed": seed,
                             "n_iter": n_iter, "target": target, "cv_max_n": cv_max_n, "val_frac": val_frac,
                             "max_val": max_val, "max_trees": max_trees, "early_stopping": early_stopping,
                             "space": [list(s) for s in space]})

    def _regressor(self, n_estimators: int, early: bool, params: dict):
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=n_estimators, tree_method="hist", random_state=self.seed,
                            n_jobs=self.n_jobs, early_stopping_rounds=self.early_stopping if early else None,
                            eval_metric="rmse", **params)

    def fit(self, ids, y):
        from qm9dipole.preprocess import target_transformer

        start = time.perf_counter()
        X, y = self.frame.loc[ids].to_numpy(dtype=np.float32), np.asarray(y, dtype=np.float64)
        if len(y) <= self.cv_max_n:
            folds, self.tuning_ = kfold(len(y), self.seed), "5-fold CV"
        else:
            folds, self.tuning_ = [holdout_split(len(y), self.seed, self.val_frac, self.max_val)], "holdout"
        rng = np.random.default_rng(self.seed)
        trials = []
        for _ in range(self.n_iter):
            params = sample_params(self.space, rng)
            pred, iters = np.full(len(y), np.nan), []
            for tr, va in folds:
                tt = target_transformer(self.target).fit(y[tr, None])
                z = tt.transform(y[:, None]).ravel()
                m = self._regressor(self.max_trees, True, params)
                m.fit(X[tr], z[tr], eval_set=[(X[va], z[va])], verbose=False)
                iters.append(m.best_iteration + 1)
                p = m.predict(X[va], iteration_range=(0, m.best_iteration + 1))
                pred[va] = tt.inverse_transform(p[:, None]).ravel()
            val = ~np.isnan(pred)
            mae = float(np.abs(clip_predictions(pred[val]) - y[val]).mean())
            trials.append({"params": params, "mae_D": mae, "trees": float(np.mean(iters))})
        best = min(trials, key=lambda t: t["mae_D"])
        n_inner = np.mean([len(tr) for tr, _ in folds])
        self.params_ = best["params"]
        self.n_trees_ = int(np.ceil(best["trees"] * len(y) / n_inner))
        self.tuning_mae_ = best["mae_D"]
        self.trials_ = trials
        self.tt_ = target_transformer(self.target).fit(y[:, None])
        self.model_ = self._regressor(self.n_trees_, False, self.params_)
        self.model_.fit(X, self.tt_.transform(y[:, None]).ravel(), verbose=False)
        self.seconds_ = time.perf_counter() - start
        return self

    def predict(self, ids):
        p = self.model_.predict(self.frame.loc[ids].to_numpy(dtype=np.float32))
        return clip_predictions(self.tt_.inverse_transform(p[:, None]).ravel())

    def describe(self) -> dict:
        return {"params": _json(self.params_), "n_trees": self.n_trees_, "tuning": self.tuning_,
                "tuning_mae_D": self.tuning_mae_, "n_trials": len(self.trials_)}


class ChargeFitter:
    """The latent-charge network (models/charge.py) on per-atom features (an atoms.AtomData
    that covers every ID it will see). Early stopping uses the model's own inner validation
    split of the training molecules. `n_models` > 1 averages an ensemble's magnitudes."""

    def __init__(self, atoms, params: dict, seed: int, n_models: int = 1, label: str = ""):
        self.atoms, self.params, self.seed, self.n_models = atoms, dict(params), seed, n_models
        self._cache: dict = {}
        self.config = _json({"kind": "charge", "label": label, "features": list(atoms.feature_names),
                             "params": self.params, "seed": seed, "n_models": n_models})

    def _subset(self, ids):
        ids = np.asarray(ids)
        key = (len(ids), hash(ids.tobytes()))
        if key not in self._cache:
            if len(self._cache) > 4:
                self._cache.clear()
            self._cache[key] = self.atoms.subset(ids)
        return self._cache[key]

    def fit(self, ids, y):
        from qm9dipole.models.charge import LatentChargeModel

        data = self.atoms.subset(np.asarray(ids))
        self.members_ = [LatentChargeModel(seed=self.seed * 1000 + k, **self.params).fit(data, np.asarray(y))
                         for k in range(self.n_models)]
        return self

    def predict(self, ids):
        data = self._subset(ids)
        return clip_predictions(np.mean([m.predict(data) for m in self.members_], axis=0))

    def describe(self) -> dict:
        m = self.members_[0]
        return {"params": _json({k: v for k, v in m.get_params().items()}), "n_parameters": m.n_parameters,
                "epochs": [x.log_.epochs for x in self.members_], "best_epoch": [x.log_.best_epoch for x in self.members_],
                "inner_val_mae_D": [x.log_.best_val_mae for x in self.members_],
                "train_seconds": [round(x.log_.seconds, 1) for x in self.members_]}
