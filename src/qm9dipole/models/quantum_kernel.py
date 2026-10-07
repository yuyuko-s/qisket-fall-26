"""Quantum kernel ridge regression (M4), simulated exactly: fidelity and projected quantum kernels.

Concepts, for a team that knows ML but is new to quantum computing:
- **Encoding (feature map).** A fixed circuit U(x) whose rotation angles are a molecule's k
  standardized inputs times an angle scale γ (`models/quantum.py`, `build_encoding_circuit`).
  Run on |0…0⟩, it prepares an n-qubit state |ψ(x)⟩: a unit vector of 2ⁿ complex amplitudes.
- **Fidelity kernel.** k(x, x′) = |⟨ψ(x)|ψ(x′)⟩|², the squared overlap of two prepared states.
  On hardware it is estimated by running U(x′) then U(x)† and counting how often every qubit
  reads 0 (PLAN §7.5). It is a valid kernel (symmetric, positive semidefinite, 1 on the
  diagonal), so kernel ridge regression uses it exactly as it uses the RBF kernel.
- **γ plays the role of the RBF bandwidth.** Small γ: every state is nearly |ψ(0)⟩, K ≈ all
  ones, and the model is nearly linear. Large γ: states become nearly orthogonal and K ≈ the
  identity (**exponential concentration**: off-diagonal entries shrink toward ~2⁻ⁿ), so the
  model can only memorize. Useful γ lies in between; `offdiag_stats` measures where a kernel is.
- **Projected quantum kernel** (Huang et al., Nat. Commun. 12, 2631, 2021). Compare each qubit's
  reduced state (its Bloch vector ⟨X⟩, ⟨Y⟩, ⟨Z⟩) instead of whole 2ⁿ-dimensional states:
  k(x, x′) = exp(−γ_p Σ_q ‖ρ_q(x) − ρ_q(x′)‖²_F). It needs 3 measured circuits per molecule
  instead of one per *pair*, and concentrates less.
- **Exact simulation.** Everything here uses exact statevectors (`models/qsim.py`): the values a
  noiseless quantum computer would give with infinitely many measurements. `noise.py` adds
  finite shots and hardware noise.

Fairness (DECISIONS.md, 2026-10-07): `KernelRidgeFitter` tunes any kernel with the protocol the
classical models use (`fitters.TabularFitter` + `evaluate.tune`): the same folds, the pooled
out-of-fold MAE in debye after inverting the target transform and clipping at 0, ties to the
first candidate, and a refit on the whole training set. With `RBFKernel` it reproduces
TabularFitter's RBF kernel ridge exactly (tests/test_quantum_kernel.py), so a difference
between a quantum kernel and RBF kernel ridge comes from the kernel, not from the tuning code.
It is faster for quantum kernels because the statevectors depend on γ but not on the ridge
penalty α: they are computed once per (fold, γ), and every α is solved from one
eigendecomposition of that fold's kernel.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.pipeline import Pipeline

from qm9dipole.complexity import kernel_ridge_df, participation_ratio
from qm9dipole.evaluate import clip_predictions, holdout_split, kfold
from qm9dipole.models.classical import KRR_ALPHAS, rbf_grid, reduction_steps
from qm9dipole.models.qsim import BatchedCircuit, fidelity_kernel, pauli_expectations
from qm9dipole.models.quantum import build_encoding_circuit
from qm9dipole.preprocess import scaling_steps, target_transformer

def quantum_gamma_grid(d: int) -> list[float]:
    """Angle scales γ = (2/√d)·c, c in logspace(−1.5, 1, 11), for d standardized inputs.

    Matched to the RBF grid (`classical.rbf_grid`, ridge penalties `classical.KRR_ALPHAS`):
    for small angles a product-state encoding has fidelity Π cos²(γΔxᵢ/2) ≈ exp(−γ²‖Δx‖²/4),
    an RBF kernel with bandwidth γ²/4. The RBF grid c′/d, c′ in logspace(−3, 2, 11), therefore
    corresponds to γ = 2√(c′/d), which is this grid (tested).
    """
    return (2.0 / np.sqrt(d) * np.logspace(-1.5, 1, 11)).tolist()


def default_qubits(encoding: str, d: int) -> int:
    """Qubits for d inputs: one per input, or one per two inputs for ry_rz (two angles per qubit)."""
    return int(np.ceil(d / 2)) if encoding == "ry_rz" else d


# --- Kernels: an embedding (with its own settings) and a Gram function --------------------

_CIRCUITS: dict[tuple, BatchedCircuit] = {}  # per process, so workers compile each circuit once


def _batched(encoding: str, reps: int, n_qubits: int, d: int) -> BatchedCircuit:
    key = (encoding, reps, n_qubits, d)
    if key not in _CIRCUITS:
        circuit = build_encoding_circuit(d, n_qubits, encoding=encoding, reps=reps)
        if [p.index for p in circuit.parameters] != list(range(d)):
            raise RuntimeError("circuit parameters are not in input order")
        _CIRCUITS[key] = BatchedCircuit(circuit)
    return _CIRCUITS[key]


class FidelityKernel:
    """k(x, x′) = |⟨ψ(x)|ψ(x′)⟩|² for the team encoder with angles γ·x; γ is tuned."""

    def __init__(self, encoding: str = "zz", reps: int = 2, n_qubits: int | None = None,
                 gammas: list[float] | None = None):
        self.encoding, self.reps, self.n_qubits, self.gammas = encoding, reps, n_qubits, gammas

    @property
    def config(self) -> dict:
        return {"kernel": "fidelity", "encoding": self.encoding, "reps": self.reps,
                "n_qubits": self.n_qubits, "gammas": self.gammas}

    def qubits(self, d: int) -> int:
        return self.n_qubits or default_qubits(self.encoding, d)

    def circuit(self, d: int):
        return build_encoding_circuit(d, self.qubits(d), encoding=self.encoding, reps=self.reps)

    def embed_grid(self, d: int) -> list[dict]:
        return [{"gamma": float(g)} for g in (self.gammas or quantum_gamma_grid(d))]

    def kernel_grid(self, d: int) -> list[dict]:
        return [{}]

    def embed(self, X: np.ndarray, setting: dict) -> np.ndarray:
        d = X.shape[1]
        return _batched(self.encoding, self.reps, self.qubits(d), d).statevectors(setting["gamma"] * X)

    def gram(self, A: np.ndarray, B: np.ndarray, setting: dict) -> np.ndarray:
        return fidelity_kernel(A, B)


def bloch_vectors(states: np.ndarray) -> np.ndarray:
    """(⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩ for every qubit q), batch × 3n: each qubit's reduced state."""
    ev = pauli_expectations(states)
    return np.hstack([ev["X"], ev["Y"], ev["Z"]])


class ProjectedKernel(FidelityKernel):
    """Projected quantum kernel: exp(−γ_p Σ_q ‖ρ_q(x) − ρ_q(x′)‖²_F) on each qubit's reduced state.

    ‖ρ − ρ′‖²_F = ½‖r − r′‖² for Bloch vectors r, so it is an RBF kernel on the 3n Bloch
    coordinates with bandwidth γ_p/2. Tuned: the encoding's angle scale γ (every other value of
    the fidelity grid: 6) and γ_p = c/n, c in logspace(−1, 2, 7) (Σ_q ‖Δr_q‖² is of order n).
    """

    def __init__(self, encoding: str = "zz", reps: int = 2, n_qubits: int | None = None,
                 gammas: list[float] | None = None, gammas_p: list[float] | None = None):
        super().__init__(encoding, reps, n_qubits, gammas)
        self.gammas_p = gammas_p

    @property
    def config(self) -> dict:
        return {**super().config, "kernel": "projected", "gammas_p": self.gammas_p}

    def embed_grid(self, d: int) -> list[dict]:
        return [{"gamma": float(g)} for g in (self.gammas or quantum_gamma_grid(d)[::2])]

    def kernel_grid(self, d: int) -> list[dict]:
        n = self.qubits(d)
        return [{"gamma_p": float(g)} for g in (self.gammas_p or (np.logspace(-1, 2, 7) / n).tolist())]

    def embed(self, X: np.ndarray, setting: dict) -> np.ndarray:
        return bloch_vectors(super().embed(X, setting))

    def gram(self, A: np.ndarray, B: np.ndarray, setting: dict) -> np.ndarray:
        return np.exp(-0.5 * setting["gamma_p"] * euclidean_distances(A, B, squared=True))


class RBFKernel:
    """The classical RBF kernel through the same code path (the fairness check)."""

    config = {"kernel": "rbf"}

    def embed_grid(self, d: int) -> list[dict]:
        return [{}]

    def kernel_grid(self, d: int) -> list[dict]:
        return [{"gamma": float(g)} for g in rbf_grid(d)["model__gamma"]]

    def embed(self, X: np.ndarray, setting: dict) -> np.ndarray:
        return X

    def gram(self, A: np.ndarray, B: np.ndarray, setting: dict) -> np.ndarray:
        return rbf_kernel(A, B, gamma=setting["gamma"])


def zz_readout_closed_form(angles: np.ndarray) -> np.ndarray:
    """⟨Z_k⟩ of the two-pass linear ZZ encoder (`build_encoding_circuit(d, d, "zz", reps=2)`) at
    angles a = γ·x, without simulating anything:

        ⟨Z_k⟩ = cos(2a_k) · Π_{j ∈ neighbours(k)} cos(2(π − a_k)(π − a_j)).

    The last pass's phase gates commute with Z, so ⟨Z_k⟩ = ⟨X_k⟩ after one pass, and averaging
    over the uniform superposition factorizes. So the teammate's quantum-feature ridge is ridge
    regression on this fixed classical feature map: a "dequantized" twin (tests check equality).
    """
    a = np.atleast_2d(np.asarray(angles, dtype=np.float64))
    out = np.cos(2 * a)
    pair = np.cos(2 * (np.pi - a[:, :-1]) * (np.pi - a[:, 1:]))
    out[:, :-1] *= pair
    out[:, 1:] *= pair
    return out


# --- Diagnostics ---------------------------------------------------------------------------

def offdiag_stats(K: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of the off-diagonal entries of a square kernel matrix.

    Concentration check (PLAN §7.3): sd < 1e-3 means the kernel barely distinguishes molecules
    (all ~1: γ too small; all ~0: γ too large, states nearly orthogonal)."""
    off = K[~np.eye(len(K), dtype=bool)]
    return float(off.mean()), float(off.std())


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centred kernel–target alignment ⟨K_c, y_c y_cᵀ⟩_F / (‖K_c‖_F ‖y_c‖²), in [−1, 1].

    How well the kernel's notion of similarity lines up with the target's (Cortes et al. 2012).
    """
    n = len(K)
    H = np.eye(n) - 1.0 / n
    Kc = H @ K @ H
    yc = np.asarray(y, dtype=np.float64) - np.mean(y)
    return float(yc @ Kc @ yc / (np.linalg.norm(Kc) * (yc @ yc)))


def kernel_spectrum(K: np.ndarray) -> np.ndarray:
    """Eigenvalues of K / n, largest first (they sum to the mean diagonal, 1 for a fidelity kernel)."""
    return np.sort(np.clip(np.linalg.eigvalsh(K), 0.0, None))[::-1] / len(K)


# --- The fitter ---------------------------------------------------------------------------

def feature_prefix(scaling: str, reduction: tuple[str, int] | None) -> Pipeline:
    """The classical Track B input steps (scaling, then reduction + re-standardization), so a
    kernel model sees exactly the numbers `classical.build(..., reduction=...)` feeds its model."""
    steps = [*scaling_steps(scaling)]
    if reduction is not None:
        steps += reduction_steps(*reduction)
    return Pipeline(steps)


def quantum_ridge_build(seed: int, scaling: str, target: str, reduction: tuple[str, int] | None,
                        n_features: int, encoding: str = "zz", reps: int = 2, n_qubits: int | None = None):
    """(estimator, grid) for the teammate's quantum-feature ridge (`models/quantum.py`) inside the
    classical Track B pipeline, for `fitters.TabularFitter`: the same input steps and target
    transform as `classical.build`, then QuantumRidgeRegressor (exact batched simulation, no
    clipping in the transformed target space). Grid: the classical ridge α grid × the angle
    scales `quantum_gamma_grid`. `seed` is unused (the model is deterministic); it keeps the
    signature of `classical.build`."""
    from sklearn.compose import TransformedTargetRegressor

    from qm9dipole.models.classical import GRIDS
    from qm9dipole.models.quantum import QuantumRidgeRegressor

    d = reduction[1] if reduction is not None else n_features
    model = QuantumRidgeRegressor(encoding=encoding, reps=reps, n_qubits=n_qubits or default_qubits(encoding, d),
                                  simulation_method="batched", clip_negative=False)
    est = TransformedTargetRegressor(regressor=Pipeline([*feature_prefix(scaling, reduction).steps, ("model", model)]),
                                     transformer=target_transformer(target), check_inverse=False)
    grid = {"regressor__model__alpha": GRIDS["ridge"]["model__alpha"], "regressor__model__gamma": quantum_gamma_grid(d)}
    return est, grid


def _fit_prefix(prefix: Pipeline, target: str, X: np.ndarray, y: np.ndarray, train: np.ndarray,
                val: np.ndarray | None):
    """Fit the input steps and the target transform on the training rows (as inside a
    TransformedTargetRegressor: supervised steps such as PLS see the transformed target)."""
    tt = target_transformer(target).fit(y[train, None])
    z = tt.transform(y[train, None]).ravel()
    fitted = clone(prefix).fit(X[train], z)
    return fitted, tt, z, fitted.transform(X[train]), (fitted.transform(X[val]) if val is not None else None)


def _fold_job(kernel, fold, embed_setting, kernel_grid, alphas):
    """Out-of-fold predictions (debye, clipped) for every (kernel setting, α) of one fold and
    one embedding setting: array (len(kernel_grid), len(alphas), n_val)."""
    Xtr, Xva, z, tt = fold
    Etr, Eva = kernel.embed(Xtr, embed_setting), kernel.embed(Xva, embed_setting)
    out = np.empty((len(kernel_grid), len(alphas), len(Xva)))
    for i, ks in enumerate(kernel_grid):
        K, Kv = kernel.gram(Etr, Etr, ks), kernel.gram(Eva, Etr, ks)
        lam, V = np.linalg.eigh((K + K.T) / 2)
        P, w = Kv @ V, V.T @ z
        for j, a in enumerate(alphas):
            zp = P @ (w / (lam + a))
            out[i, j] = clip_predictions(tt.inverse_transform(zp[:, None]).ravel())
    return out


class KernelRidgeFitter:
    """Kernel ridge regression with any kernel (fidelity, projected, RBF) for the X2 harness
    (`evaluate.dev_curve`), tuned exactly as `fitters.TabularFitter` tunes RBF kernel ridge.

    frame:     molecule-level features indexed by ID (e.g. the all_legal columns).
    kernel:    a kernel object (FidelityKernel, ProjectedKernel, RBFKernel).
    scaling, reduction, target: the classical input steps and target transform
               (`feature_prefix`, `preprocess.target_transformer`), fit inside every fold.
    alphas:    ridge penalties (default KRR_ALPHAS, the RBF grid's).

    Tuning: 5-fold CV (`evaluate.kfold`, seeded by `seed`) up to `cv_max_n` training molecules,
    one inner validation split above; candidates are (α, embedding setting, kernel setting) in
    that nesting order, scored by pooled out-of-fold MAE in debye. The winner is refit on all
    training molecules. Predictions are in debye, clipped at 0.
    """

    def __init__(self, frame: pd.DataFrame, kernel, seed: int, scaling: str = "yeo_johnson",
                 target: str = "sqrt", reduction: tuple[str, int] | None = None,
                 alphas: tuple[float, ...] = KRR_ALPHAS, cv_max_n: int = 1000, val_frac: float = 0.1,
                 max_val: int = 5000, n_jobs: int = -1, diagnostics_max_n: int = 3000, label: str = ""):
        self.frame, self.kernel, self.seed = frame, kernel, seed
        self.scaling, self.target, self.reduction, self.alphas = scaling, target, reduction, tuple(alphas)
        self.cv_max_n, self.val_frac, self.max_val, self.n_jobs = cv_max_n, val_frac, max_val, n_jobs
        self.diagnostics_max_n = diagnostics_max_n
        self.prefix = feature_prefix(scaling, reduction)
        self.config = json.loads(json.dumps(
            {"kind": "kernel_ridge", "label": label, "features": list(frame.columns), "kernel": kernel.config,
             "scaling": scaling, "target": target, "reduction": list(reduction) if reduction else None,
             "alphas": list(self.alphas), "seed": seed, "cv_max_n": cv_max_n, "val_frac": val_frac,
             "max_val": max_val}))

    def _X(self, ids) -> np.ndarray:
        return self.frame.loc[ids].to_numpy(dtype=np.float64)

    def candidates(self, d: int) -> list[tuple[float, dict, dict]]:
        return [(a, e, k) for a in self.alphas for e in self.kernel.embed_grid(d) for k in self.kernel.kernel_grid(d)]

    def fit(self, ids, y):
        start = time.perf_counter()
        X, y = self._X(ids), np.asarray(y, dtype=np.float64)
        n = len(y)
        if n <= self.cv_max_n:
            folds, self.tuning_ = kfold(n, self.seed), "5-fold CV"
        else:
            folds, self.tuning_ = [holdout_split(n, self.seed, self.val_frac, self.max_val)], "holdout"
        prepared = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(_fit_prefix)(self.prefix, self.target, X, y, tr, va) for tr, va in folds)
        fold_data = [(Xtr, Xva, z, tt) for _, tt, z, Xtr, Xva in prepared]
        d = fold_data[0][0].shape[1]
        self.n_inputs_ = d
        self.n_qubits_ = self.kernel.qubits(d) if hasattr(self.kernel, "qubits") else None
        egrid, kgrid = self.kernel.embed_grid(d), self.kernel.kernel_grid(d)
        jobs = [(f, e) for f in range(len(folds)) for e in range(len(egrid))]
        preds = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(_fold_job)(self.kernel, fold_data[f], egrid[e], kgrid, self.alphas) for f, e in jobs)
        # oof[α, e, k, sample]: the candidate order of `candidates` (α outermost)
        oof = np.full((len(self.alphas), len(egrid), len(kgrid), n), np.nan)
        for (f, e), p in zip(jobs, preds):
            oof[:, e, :, folds[f][1]] = np.transpose(p, (2, 1, 0))
        covered = np.sort(np.concatenate([va for _, va in folds]))
        mae = np.abs(oof[..., covered] - y[covered]).mean(axis=-1)
        a, e, k = np.unravel_index(int(np.argmin(mae.ravel())), mae.shape)  # ties -> first candidate
        self.alpha_, self.embed_setting_, self.kernel_setting_ = self.alphas[a], egrid[e], kgrid[k]
        self.tuning_mae_ = float(mae[a, e, k])
        self.n_candidates_ = int(mae.size)
        self.grid_edge_ = [name for name, idx, size in (("alpha", a, len(self.alphas)), ("embedding", e, len(egrid)),
                                                        ("kernel", k, len(kgrid))) if size > 1 and idx in (0, size - 1)]
        self.cv_mae_surface_ = mae
        # Refit on the whole training set.
        self.prefix_, self.tt_, z, Xt, _ = _fit_prefix(self.prefix, self.target, X, y, np.arange(n), None)
        self.z_train_ = z
        self.train_embedding_ = self.kernel.embed(Xt, self.embed_setting_)
        K = self.kernel.gram(self.train_embedding_, self.train_embedding_, self.kernel_setting_)
        self.model_ = KernelRidge(kernel="precomputed", alpha=self.alpha_).fit(K, z)
        self.diagnostics_ = {}
        if n <= self.diagnostics_max_n:
            mean, sd = offdiag_stats(K)
            self.diagnostics_ = {"offdiag_mean": mean, "offdiag_sd": sd, "alignment": kernel_target_alignment(K, z),
                                 "effective_df": kernel_ridge_df(K, self.alpha_),
                                 "kernel_participation_ratio": participation_ratio(kernel_spectrum(K))}
        self.seconds_ = time.perf_counter() - start
        return self

    def transform_inputs(self, X: np.ndarray) -> np.ndarray:
        """The fitted input steps applied to raw features X (what the encoder sees, before γ)."""
        return self.prefix_.transform(np.asarray(X, dtype=np.float64))

    def embed_X(self, X: np.ndarray) -> np.ndarray:
        """The chosen embedding (statevectors, Bloch vectors, …) of raw feature rows X."""
        return self.kernel.embed(self.transform_inputs(X), self.embed_setting_)

    def gram_to_train(self, X: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Kernel matrix between raw feature rows X and the training molecules (len(X) × N)."""
        Xt = self.transform_inputs(X)
        return np.vstack([self.kernel.gram(self.kernel.embed(Xt[s:s + chunk], self.embed_setting_),
                                           self.train_embedding_, self.kernel_setting_)
                          for s in range(0, len(Xt), chunk)])

    def predict_from_gram(self, Kv: np.ndarray, model: KernelRidge | None = None) -> np.ndarray:
        """Predictions (debye, clipped at 0) from a given kernel matrix to the training set, e.g.
        one estimated with finite shots; `model` defaults to the fitted kernel ridge."""
        z = (model or self.model_).predict(Kv)
        return clip_predictions(self.tt_.inverse_transform(z[:, None]).ravel())

    def predict_X(self, X: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Predictions (debye, clipped at 0) for a raw feature matrix in the frame's column order,
        `chunk` rows at a time (16-qubit statevectors take 1 MB per molecule)."""
        Xt = self.transform_inputs(X)
        z = np.empty(len(Xt))
        for s in range(0, len(Xt), chunk):
            E = self.kernel.embed(Xt[s:s + chunk], self.embed_setting_)
            z[s:s + chunk] = self.model_.predict(self.kernel.gram(E, self.train_embedding_, self.kernel_setting_))
        return clip_predictions(self.tt_.inverse_transform(z[:, None]).ravel())

    def predict(self, ids):
        return self.predict_X(self._X(ids))

    def describe(self) -> dict:
        return {"params": {"alpha": self.alpha_, **self.embed_setting_, **self.kernel_setting_},
                "tuning": self.tuning_, "tuning_mae_D": self.tuning_mae_, "n_candidates": self.n_candidates_,
                "grid_edge": self.grid_edge_, "n_inputs": self.n_inputs_, "n_qubits": self.n_qubits_,
                "fit_seconds": round(self.seconds_, 2), **self.diagnostics_}
