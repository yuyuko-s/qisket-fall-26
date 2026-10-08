"""Training pipeline and diagnostics for the active quantum-feature ⟨Z⟩ ridge model.

Kept separate from the archived fidelity and projected kernel regressors.
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from qm9dipole.models.classical import reduction_steps
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
