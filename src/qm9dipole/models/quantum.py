"""Fixed Qiskit quantum features with a classical linear ridge readout.

The encoder prepares a state for each molecule; per-qubit Z expectations are
its dimensionless features. Ridge learns their linear mapping to debye labels.
This is a linear projected quantum-feature model, not a fidelity kernel or VQR.
Execution is local simulation only (statevector or MPS): no IBM clients or jobs.
Input columns and qubits are independent; encoders re-upload inputs in batches
according to a capacity-based mapping rather than qubit-count-specific cases.
Inputs must already be fixed-width descriptors; molecular invariance and any
upstream PCA are the caller's responsibility, including fold-local PCA fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXGate, RYGate, RZGate, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
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


# Adding an angle encoding requires an axis tuple, not feature/qubit-count cases.
ENCODING_AXES: dict[str, tuple[str, ...]] = {
    "zz": ("z",),
    "ry": ("ry",),
    "ry_rz": ("ry", "rz"),
}
MIN_REPS: dict[str, int] = {"zz": 2, "ry": 1, "ry_rz": 1}
MIN_QUBITS: dict[str, int] = {"zz": 2, "ry": 1, "ry_rz": 1}
ROTATION_GATES = {"ry": RYGate, "rz": RZGate}
MIXING_ANGLE = np.pi / 4
SIMULATION_METHODS = ("statevector", "matrix_product_state")


@dataclass(frozen=True)
class FeaturePlacement:
    """Location of an input in the circuit, with all indices zero-based."""

    feature_index: int
    qubit_index: int
    upload_layer: int
    rotation_index: int


def feature_qubit_mapping(
    n_features: int, n_qubits: int, rotations_per_qubit: int = 1
) -> tuple[FeaturePlacement, ...]:
    """Pack inputs into qubit/rotation slots and then further upload layers.

    Capacity per layer is n_qubits * rotations_per_qubit. Each feature occurs
    exactly once in this mapping; repeated passes reuse these placements.
    Partial layers need no special cases: the circuit fills unused slots with 0.
    """
    n_features = _integer_parameter(n_features, "n_features", 1)
    n_qubits = _integer_parameter(n_qubits, "n_qubits", 1)
    rotations_per_qubit = _integer_parameter(
        rotations_per_qubit, "rotations_per_qubit", 1
    )
    capacity = n_qubits * rotations_per_qubit
    return tuple(
        FeaturePlacement(
            feature_index=i,
            qubit_index=(i % capacity) // rotations_per_qubit,
            upload_layer=i // capacity,
            rotation_index=i % rotations_per_qubit,
        )
        for i in range(n_features)
    )


def _encoding_axes(encoding: str) -> tuple[str, ...]:
    if not isinstance(encoding, str) or encoding not in ENCODING_AXES:
        raise ValueError(f"encoding must be one of {tuple(ENCODING_AXES)}.")
    return ENCODING_AXES[encoding]


def build_encoding_circuit(
    n_features: int, n_qubits: int, *, encoding: str = "zz", reps: int = 2
) -> QuantumCircuit:
    """Build a fixed data encoder using generic feature-to-qubit placement.

    All input parameters are dimensionless angles in radians. ZZ uses Qiskit's
    built-in phase map; angle encoders use library rotation gates. No parameter
    is a trainable weight. reps repeats complete passes over the input layers.
    """
    axes = _encoding_axes(encoding)
    n_qubits = _integer_parameter(n_qubits, "n_qubits", MIN_QUBITS[encoding])
    reps = _integer_parameter(reps, "reps", MIN_REPS[encoding])
    mapping = feature_qubit_mapping(n_features, n_qubits, len(axes))
    n_layers = mapping[-1].upload_layer + 1
    parameters = ParameterVector("x", len(mapping))
    slots = np.zeros((n_layers, n_qubits, len(axes)), dtype=object)
    for placement in mapping:
        slots[
            placement.upload_layer, placement.qubit_index, placement.rotation_index
        ] = parameters[placement.feature_index]

    circuit = QuantumCircuit(n_qubits, name=f"{encoding}_encoding")
    if encoding == "zz":
        block = zz_feature_map(
            n_qubits, reps=1, entanglement="linear", parameter_prefix="block"
        )
        for step in range(reps * n_layers):
            values = slots[step % n_layers, :, 0]
            bound = block.assign_parameters(dict(zip(block.parameters, values)))
            circuit.compose(bound, inplace=True)
    else:
        n_steps = reps * n_layers
        for step in range(n_steps):
            layer = slots[step % n_layers]
            for qubit in range(n_qubits):
                for rotation, axis in enumerate(axes):
                    circuit.append(
                        ROTATION_GATES[axis](layer[qubit, rotation]), [qubit]
                    )
                # A noncommuting mixer makes phase inputs visible to Z readout.
                circuit.append(RXGate(MIXING_ANGLE), [qubit])
            if step + 1 < n_steps:
                # CZs commute: even/odd scheduling avoids unnecessary chain depth.
                for parity in range(2):
                    for qubit in range(parity, n_qubits - 1, 2):
                        circuit.cz(qubit, qubit + 1)
    return circuit


def _local_estimator(method: str) -> StatevectorEstimator | AerEstimatorV2:
    if not isinstance(method, str) or method not in SIMULATION_METHODS:
        raise ValueError(f"simulation_method must be one of {SIMULATION_METHODS}.")
    if method == "statevector":
        return StatevectorEstimator(default_precision=0.0)
    return AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": {
                "method": "matrix_product_state",
                "matrix_product_state_truncation_threshold": 0.0,
                "max_parallel_threads": 1,
            },
        }
    )


class QuantumFeatureTransformer(TransformerMixin, BaseEstimator):
    """Map circuit-angle inputs to per-qubit Z expectations in [-1, 1].

    ``n_qubits`` is independent of input width. Each encoding defines rotation
    slots per qubit; excess inputs use further upload layers and unused slots
    are zero-padded. Default ``n_qubits=None, encoding='zz'`` retains the original
    one-column-per-qubit circuit. ``ry`` and ``ry_rz`` are alternative angle maps.
    ``fit`` records the width and constructs the encoder; it never uses labels.
    ``transform`` binds ``gamma * X`` as angles in radians, preserving row order.
    Use training-fitted scaling upstream, as QuantumRidgeRegressor does.

    ZZ needs at least two passes to avoid constant Z readout. Angle maps use a
    fixed noncommuting mixer after each upload to expose phase inputs. All local
    Z observables share a measurement basis. Simulation is exact up to numerical
    precision; MPS is useful for wider, shallow nearest-neighbor circuits, not
    a guarantee of efficient simulation for arbitrary entanglement/depth.
    """

    def __init__(
        self,
        *,
        reps: int = 2,
        gamma: float = 1.0,
        batch_size: int = 128,
        n_qubits: int | None = None,
        encoding: str = "zz",
        simulation_method: str = "statevector",
    ) -> None:
        self.reps = reps
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.simulation_method = simulation_method

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Construct the encoder for a finite (n_samples, n_features) array."""
        axes = _encoding_axes(self.encoding)
        reps = _integer_parameter(self.reps, "reps", MIN_REPS[self.encoding])
        gamma = _real_parameter(self.gamma, "gamma")
        batch_size = _integer_parameter(self.batch_size, "batch_size", 1)
        matrix: NDArray[np.float64] = check_array(X, dtype=np.float64)
        n_features = matrix.shape[1]
        n_qubits = _integer_parameter(
            n_features if self.n_qubits is None else self.n_qubits,
            "n_qubits",
            MIN_QUBITS[self.encoding],
        )
        mapping = feature_qubit_mapping(n_features, n_qubits, len(axes))
        circuit = build_encoding_circuit(
            n_features, n_qubits, encoding=self.encoding, reps=reps
        )
        estimator = _local_estimator(self.simulation_method)
        # Explicit qubit indices avoid Qiskit's little-endian bitstring ambiguity.
        observables = [
            SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits)
        ]
        self.circuit_ = circuit
        self.observables_ = observables
        self.estimator_ = estimator
        self.n_features_in_ = n_features
        self.n_qubits_ = n_qubits
        self.n_features_out_ = n_qubits
        self.feature_mapping_ = mapping
        self.n_upload_layers_ = mapping[-1].upload_layer + 1
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

        features = np.empty((len(matrix), self.n_qubits_), dtype=np.float64)
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
            [f"z_expectation_{i}" for i in range(self.n_qubits_)],
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
    inputs into circuit angles, and ``reps`` repeats input-upload passes.
    ``n_qubits`` and ``encoding`` determine the generic packing of inputs;
    ``simulation_method='matrix_product_state'`` avoids a full statevector
    for the wider shallow-circuit experiments. Output feature count is the
    qubit count, not necessarily the original descriptor width.
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
        n_qubits: int | None = None,
        encoding: str = "zz",
        simulation_method: str = "statevector",
    ) -> None:
        self.alpha = alpha
        self.reps = reps
        self.gamma = gamma
        self.batch_size = batch_size
        self.clip_negative = clip_negative
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.simulation_method = simulation_method

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
                        n_qubits=self.n_qubits,
                        encoding=self.encoding,
                        simulation_method=self.simulation_method,
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
