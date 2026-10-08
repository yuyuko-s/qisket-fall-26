"""Test-set evaluation: features for the test molecules, the model roster of
`configs/test_eval.yaml`, finite-shot and noisy inference on the quantum test subsets, and the
log of test runs. The driver is `scripts/final_eval.py`; `notebooks/07_quantum_feature_ridge.ipynb`
presents a run.

The team iterates after test runs (the team's decision; every run is logged): change the config or the code, describe
the change in the config's `note`, run again. Every run gets the next run number, its results
are written as `results/test_runNN_<section>.csv` (with provenance sidecars), and one row is
appended to `results/test_runs.csv`. Inside a run, every model still tunes on its training set
only, exactly as in the development notebooks (04, 06, 07), and also predicts the development
sets, so a run can be checked against those notebooks and its reference models can be chosen
on `dev` as before.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qm9dipole import REPO_ROOT
from qm9dipole.descriptors import MODEL_DESCRIPTORS, feature_frame
from qm9dipole.evaluate import clip_predictions, scores

CONFIG_PATH = REPO_ROOT / "configs" / "test_eval.yaml"
RUN_LOG = REPO_ROOT / "results" / "test_runs.csv"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


# --- Features for molecules outside the development universe ----------------------------

def legal_features(table: pd.DataFrame, ids: np.ndarray, columns: list[str], n_jobs: int = -1) -> pd.DataFrame:
    """The headline-legal molecule-level features `columns` for molecules `ids`, computed from Z
    and R with the same descriptor functions as notebook 02 and rounded through float32, the
    precision `explore_features.parquet` stores; float64 frame indexed by id.

    `table` is the parsed QM9 table (columns id, Z, R)."""
    rows = table.set_index("id").loc[np.asarray(ids), ["Z", "R"]].reset_index()
    blocks = [feature_frame(rows, name, n_jobs=n_jobs) for name in MODEL_DESCRIPTORS]
    frame = pd.concat(blocks, axis=1)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"features not produced by the descriptors: {missing[:5]}")
    return frame[columns].astype(np.float32).astype(np.float64)


def evaluation_frame(dev_features: pd.DataFrame, table: pd.DataFrame, extra_ids: np.ndarray,
                     columns: list[str], n_jobs: int = -1) -> pd.DataFrame:
    """One float64 feature frame for the development universe (from the stored table) and the
    `extra_ids` (computed), indexed by id."""
    extra = np.setdiff1d(np.asarray(extra_ids), dev_features.index.to_numpy())
    parts = [dev_features[columns].astype(np.float64)]
    if len(extra):
        parts.append(legal_features(table, extra, columns, n_jobs=n_jobs))
    return pd.concat(parts)


# --- The run log ----------------------------------------------------------------------------

def next_run_number(log: Path = RUN_LOG) -> int:
    if not Path(log).exists():
        return 1
    runs = pd.read_csv(log)
    return int(runs["run"].max()) + 1 if len(runs) else 1


def log_run(record: dict, log: Path = RUN_LOG) -> None:
    """Append one row to the test-run log (created with a header if missing)."""
    log = Path(log)
    row = pd.DataFrame([record])
    if log.exists():
        row = pd.concat([pd.read_csv(log), row], ignore_index=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(log, index=False)


# --- Finite shots and noise with fixed settings ----------------------------------------------

def shot_scores(fitter, X_train: np.ndarray, y_train: np.ndarray, X_eval: dict[str, np.ndarray],
                y_eval: dict[str, np.ndarray], shots: list[int], reps: int, seed: int, kind: str) -> list[dict]:
    """Scores of a fitted quantum model on each evaluation set, exactly and from finite shots.

    kind: "fidelity" (a fitted KernelRidgeFitter with a fidelity kernel: the test kernel from
    shots, then also the training kernel), "projected" (KernelRidgeFitter with a projected kernel:
    every Bloch coordinate of training and evaluation molecules from shots, model refit) or
    "z_readout" (a fitted TabularFitter around the team's QuantumRidgeRegressor: every ⟨Z⟩ from
    shots, ridge refit). Shot noise is drawn exactly as an ideal device would return it
    (`noise.shot_kernel`, `noise.shot_expectations`). X_train, y_train: the raw training features
    and labels (debye) the fitter was trained on.
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge

    from qm9dipole.noise import repair_kernel, shot_expectations, shot_kernel

    rows = []

    def add(mode, S, r, ev, pred):
        rows.append({"mode": mode, "shots": S, "rep": r, "eval_set": ev, **scores(y_eval[ev], pred)})

    if kind == "fidelity":
        Ktr = fitter.kernel.gram(fitter.train_embedding_, fitter.train_embedding_, fitter.kernel_setting_)
        Kev = {ev: fitter.gram_to_train(X) for ev, X in X_eval.items()}
        for ev in X_eval:
            add("exact", np.inf, 0, ev, fitter.predict_from_gram(Kev[ev]))
        for S in shots:
            for r in range(reps):
                rng = np.random.default_rng([seed, S, r])
                noisy = KernelRidge(kernel="precomputed", alpha=fitter.alpha_).fit(repair_kernel(shot_kernel(Ktr, S, rng)),
                                                                                  fitter.z_train_)
                for ev in X_eval:
                    Kh = shot_kernel(Kev[ev], S, rng)
                    add("test kernel from shots", S, r, ev, fitter.predict_from_gram(Kh))
                    add("training and test kernels from shots", S, r, ev, fitter.predict_from_gram(Kh, noisy))
    elif kind == "projected":
        Btr, ks = fitter.train_embedding_, fitter.kernel_setting_
        Bev = {ev: fitter.embed_X(X) for ev, X in X_eval.items()}
        for ev in X_eval:
            add("exact", np.inf, 0, ev, fitter.predict_from_gram(fitter.kernel.gram(Bev[ev], Btr, ks)))
        for S in shots:
            for r in range(reps):
                rng = np.random.default_rng([seed, S, r, 1])
                Bh = shot_expectations(Btr, S, rng)
                model = KernelRidge(kernel="precomputed", alpha=fitter.alpha_).fit(fitter.kernel.gram(Bh, Bh, ks), fitter.z_train_)
                for ev in X_eval:
                    Kh = fitter.kernel.gram(shot_expectations(Bev[ev], S, rng), Bh, ks)
                    add("features from shots", S, r, ev, fitter.predict_from_gram(Kh, model))
    elif kind == "z_readout":
        est = fitter.model_
        prefix, qr = est.regressor_[:-1], est.regressor_[-1]
        Ftr = qr.quantum_features(prefix.transform(X_train))
        Fev = {ev: qr.quantum_features(prefix.transform(X)) for ev, X in X_eval.items()}
        z_tr = est.transformer_.transform(np.asarray(y_train, dtype=np.float64)[:, None]).ravel()
        alpha = fitter.params_["regressor__model__alpha"]

        def inv(z):
            return clip_predictions(est.transformer_.inverse_transform(z[:, None]).ravel())

        for ev, X in X_eval.items():
            add("exact", np.inf, 0, ev, clip_predictions(est.predict(X)))
        for S in shots:
            for r in range(reps):
                rng = np.random.default_rng([seed, S, r, 2])
                ridge = Ridge(alpha=alpha).fit(shot_expectations(Ftr, S, rng), z_tr)
                for ev in X_eval:
                    add("features from shots", S, r, ev, inv(ridge.predict(shot_expectations(Fev[ev], S, rng))))
    else:
        raise KeyError(kind)
    return rows


def noisy_fidelity_scores(fitter, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, shots: int,
                          backend: str, seed: int = 0, n_jobs: int = -1) -> tuple[list[dict], dict]:
    """A fitted fidelity-kernel model's test predictions with every test-kernel entry measured on
    a noisy simulator (`backend`'s noise model on Aer, locally), against exact and shots-only
    predictions, plus the one-circuit-per-molecule depolarizing correction. Returns score rows
    and the kernels (exact, noisy, mitigated, survival)."""
    from qm9dipole.noise import depolarizing_mitigation, sample_overlaps, shot_kernel

    g = fitter.embed_setting_["gamma"]
    d = fitter.n_inputs_
    enc = fitter.kernel.circuit(d)
    At = g * fitter.transform_inputs(X_test)
    Atr = g * fitter.transform_inputs(X_train)
    K_exact = fitter.gram_to_train(X_test)
    K_noisy = sample_overlaps(enc, np.repeat(At, len(Atr), axis=0), np.tile(Atr, (len(At), 1)), shots,
                              backend=backend, seed=seed, n_jobs=n_jobs, chunk=25).reshape(K_exact.shape)
    survival = sample_overlaps(enc, At, At, shots, backend=backend, seed=seed + 1, n_jobs=n_jobs, chunk=5)
    K_mit = depolarizing_mitigation(K_noisy, survival, enc.num_qubits)
    K_shots = shot_kernel(K_exact, shots, np.random.default_rng(seed + 2))
    rows = [{"mode": mode, **scores(y_test, fitter.predict_from_gram(K))}
            for mode, K in (("exact", K_exact), ("shots only", K_shots), ("noisy", K_noisy), ("noisy, mitigated", K_mit))]
    return rows, {"exact": K_exact, "noisy": K_noisy, "mitigated": K_mit, "survival": survival}


def feature_model(fitter, kind: str, X_train: np.ndarray, y_train: np.ndarray):
    """What it takes to run a fitted feature-based quantum model on a device: kind "projected" (a
    KernelRidgeFitter with a projected kernel: X, Y and Z bases) or "z_readout" (a TabularFitter around
    the team's QuantumRidgeRegressor: Z basis). Returns a namespace with
    - encoder: the parameterized encoding circuit U(x) (parameter vector "x");
    - bases: the measurement bases, in the feature layout's order;
    - angles(X): the circuit parameters (γ·scaled inputs) for raw feature rows X;
    - exact_train, exact(X): the exact features of the training rows and of any raw rows;
    - predict(F_train, F_test): debye predictions after refitting the model's readout (kernel ridge or
      ridge, at its tuned penalty) on the given training features, e.g. ones measured with shots or on
      hardware.
    """
    from types import SimpleNamespace

    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge

    Xtr = np.asarray(X_train, dtype=np.float64)
    if kind == "projected":
        g, d, ks = fitter.embed_setting_["gamma"], fitter.n_inputs_, fitter.kernel_setting_

        def predict(F_tr, F_te):
            m = KernelRidge(kernel="precomputed", alpha=fitter.alpha_).fit(fitter.kernel.gram(F_tr, F_tr, ks), fitter.z_train_)
            return fitter.predict_from_gram(fitter.kernel.gram(F_te, F_tr, ks), m)

        return SimpleNamespace(encoder=fitter.kernel.circuit(d), bases="XYZ",
                               angles=lambda X: g * fitter.transform_inputs(np.asarray(X, dtype=np.float64)),
                               exact_train=fitter.train_embedding_, exact=fitter.embed_X, predict=predict)
    if kind == "z_readout":
        est = fitter.model_
        prefix, qr = est.regressor_[:-1], est.regressor_[-1]
        scaler, gq = qr.pipeline_[0], qr.pipeline_[1].gamma_
        z_tr = est.transformer_.transform(np.asarray(y_train, dtype=np.float64)[:, None]).ravel()

        def predict(F_tr, F_te):
            ridge = Ridge(alpha=fitter.params_["regressor__model__alpha"]).fit(F_tr, z_tr)
            return clip_predictions(est.transformer_.inverse_transform(ridge.predict(F_te)[:, None]).ravel())

        return SimpleNamespace(encoder=qr.pipeline_[1].circuit_, bases="Z",
                               angles=lambda X: gq * scaler.transform(prefix.transform(np.asarray(X, dtype=np.float64))),
                               exact_train=qr.quantum_features(prefix.transform(Xtr)),
                               exact=lambda X: qr.quantum_features(prefix.transform(np.asarray(X, dtype=np.float64))),
                               predict=predict)
    raise KeyError(kind)


def noisy_feature_scores(fitter, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                         shots: int, backend: str, kind: str, seed: int = 0, n_jobs: int = -1) -> list[dict]:
    """End-to-end noisy runs of the projected kernel (kind "projected": X, Y, Z bases) or the
    team's ⟨Z⟩ ridge (kind "z_readout": Z basis): training and test features measured on the
    noisy simulator, model refit; with exact and shots-only references."""
    from qm9dipole.noise import sample_bloch, shot_expectations

    rng = np.random.default_rng(seed + 3)
    fm = feature_model(fitter, kind, X_train, y_train)
    exact = (fm.exact_train, fm.exact(X_test))
    noisy = (sample_bloch(fm.encoder, fm.angles(X_train), shots, backend=backend, seed=seed, bases=fm.bases,
                          n_jobs=n_jobs, chunk=10),
             sample_bloch(fm.encoder, fm.angles(X_test), shots, backend=backend, seed=seed + 100, bases=fm.bases,
                          n_jobs=n_jobs, chunk=10))
    shot = (shot_expectations(exact[0], shots, rng), shot_expectations(exact[1], shots, rng))
    return [{"mode": mode, **scores(y_test, fm.predict(*F))} for mode, F in (("exact", exact), ("shots only", shot), ("noisy", noisy))]


def timed(label: str):
    """Context manager printing how long a section took."""
    class _T:
        def __enter__(self):
            self.t = time.perf_counter()
            print(f"[{time.strftime('%H:%M')}] {label} ...", flush=True)
            return self

        def __exit__(self, *exc):
            print(f"[{time.strftime('%H:%M')}] {label}: {(time.perf_counter() - self.t) / 60:.1f} min", flush=True)
    return _T()


def jsonable(d) -> str:
    return json.dumps(d, default=str, sort_keys=True)
