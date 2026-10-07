"""Fixed Qiskit quantum features with a classical linear ridge readout.

The encoder prepares a state for each molecule; per-qubit Z expectations are
its dimensionless features. Ridge learns their linear mapping to debye labels.
This is a linear projected quantum-feature model, not a fidelity kernel or VQR.
Execution is exact, local statevector simulation only: no IBM clients or jobs.
Inputs must already be fixed-width descriptors; molecular invariance and any
upstream PCA are the caller's responsibility, including fold-local PCA fitting.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _integer_parameter(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _real_parameter(value: float, name: str, *, allow_zero: bool = False) -> float:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and np.isfinite(value)
        and (value >= 0 if allow_zero else value > 0)
    )
    if not valid:
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {bound} real number.")
    return float(value)


class QuantumFeatureTransformer(TransformerMixin, BaseEstimator):
    """Map circuit-angle inputs to per-qubit Z expectations in [-1, 1].

    One qubit is used per input column (at least two). Prefer 4–8 columns for
    this project: statevector memory grows exponentially with the qubit count.
    ``fit`` records the width and constructs the encoder; it never uses labels.
    ``transform`` binds ``gamma * X`` as angles in radians, preserving row order.
    Use training-fitted scaling upstream, as QuantumRidgeRegressor does.

    The Qiskit ZZ feature map encodes inputs through phases and entanglement.
    At least two repetitions are required: one layer leaves all Z expectations
    zero, whereas the next layer's Hadamards convert phases into readout signal.
    All local Z observables commute, so hardware could share one measurement
    basis per input. This implementation evaluates them exactly on a simulator.
    """

    def __init__(
        self, *, reps: int = 2, gamma: float = 1.0, batch_size: int = 128
    ) -> None:
        self.reps = reps
        self.gamma = gamma
        self.batch_size = batch_size

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Construct the encoder for a finite (n_samples, n_features) array."""
        reps = _integer_parameter(self.reps, "reps", 2)
        gamma = _real_parameter(self.gamma, "gamma")
        batch_size = _integer_parameter(self.batch_size, "batch_size", 1)
        matrix: NDArray[np.float64] = check_array(X, dtype=np.float64)
        n_features = matrix.shape[1]
        if n_features < 2:
            raise ValueError("The ZZ encoder requires at least 2 input features.")

        circuit = zz_feature_map(
            feature_dimension=n_features, reps=reps, entanglement="linear"
        )
        # Explicit qubit indices avoid Qiskit's little-endian bitstring ambiguity.
        observables = [
            SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_features)
            for i in range(n_features)
        ]
        self.circuit_ = circuit
        self.observables_ = observables
        self.estimator_ = StatevectorEstimator(default_precision=0.0)
        self.n_features_in_ = n_features
        self.gamma_ = gamma
        self.batch_size_ = batch_size
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return dimensionless expectations for inputs with the fitted width.

        Precision zero means exact expectations, not finite-shot measurements
        or Gaussian sampling noise. Batching limits primitive request sizes.
        """
        check_is_fitted(self, ["circuit_", "observables_", "estimator_"])
        matrix: NDArray[np.float64] = check_array(X, dtype=np.float64)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features; expected {self.n_features_in_}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            angles = self.gamma_ * matrix
        if not np.isfinite(angles).all():
            raise ValueError("gamma * X must contain finite circuit angles.")

        features = np.empty(matrix.shape, dtype=np.float64)
        for start in range(0, len(matrix), self.batch_size_):
            stop = min(start + self.batch_size_, len(matrix))
            results = self.estimator_.run(
                [(self.circuit_, self.observables_, row) for row in angles[start:stop]],
                precision=0.0,
            ).result()
            features[start:stop] = np.stack(
                [results[i].data["evs"] for i in range(len(results))]
            )
        return features

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Name outputs by the qubit whose Z expectation they contain."""
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if names.ndim != 1 or len(names) != self.n_features_in_:
                raise ValueError("input_features must name every input feature.")
        return np.asarray(
            [f"z_expectation_{i}" for i in range(self.n_features_in_)],
            dtype=object,
        )


class QuantumRidgeRegressor(RegressorMixin, BaseEstimator):
    """Training-fitted scaling → fixed quantum features → linear ridge.

    ``X`` is a finite (n_molecules, n_features) descriptor/PCA-score matrix;
    ``y`` is a one-dimensional array of dipole magnitudes in debye. The readout
    is fitted directly on these labels, so predictions are already in debye,
    not bounded to the expectation range [-1, 1]. By default negative magnitude
    predictions are clipped at zero, following PLAN §6.5.

    ``alpha`` controls ridge regularization, ``gamma`` scales standardized
    inputs into circuit angles, and ``reps`` controls encoder repetitions.
    Defaults are illustrative, not tuned. The estimator is cloneable for
    GridSearchCV; each fit builds a fresh scaler and readout. Put any learned
    descriptor preprocessing/PCA outside this estimator in the same sklearn
    Pipeline so it is also fitted inside each training/CV fold.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        reps: int = 2,
        gamma: float = 1.0,
        batch_size: int = 128,
        clip_negative: bool = True,
    ) -> None:
        self.alpha = alpha
        self.reps = reps
        self.gamma = gamma
        self.batch_size = batch_size
        self.clip_negative = clip_negative

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit all internal transforms/readout using only X and debye labels y."""
        alpha = _real_parameter(self.alpha, "alpha", allow_zero=True)
        if not isinstance(self.clip_negative, (bool, np.bool_)):
            raise ValueError("clip_negative must be a boolean.")
        if np.asarray(y).ndim != 1:
            raise ValueError("y must be a one-dimensional array of debye labels.")
        matrix: NDArray[np.float64]
        labels: NDArray[np.float64]
        matrix, labels = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        labels = np.asarray(labels, dtype=np.float64)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "quantum",
                    QuantumFeatureTransformer(
                        reps=self.reps,
                        gamma=self.gamma,
                        batch_size=self.batch_size,
                    ),
                ),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        pipeline.fit(matrix, labels)
        self.pipeline_ = pipeline
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict dipole magnitudes in debye without refitting any transform."""
        check_is_fitted(self, "pipeline_")
        predictions = self.pipeline_.predict(X)
        return np.maximum(predictions, 0.0) if self.clip_negative else predictions

    def quantum_features(self, X: ArrayLike) -> NDArray[np.float64]:
        """Apply fitted input scaling and return dimensionless quantum features."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_[:-1].transform(X)
