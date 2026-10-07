"""Generic input packing and exact local encoders on synthetic arrays only."""

from dataclasses import fields, is_dataclass

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import EstimatorPub
from qiskit.quantum_info import Statevector
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from qm9dipole.models.quantum import (
    ENCODING_AXES,
    MIN_QUBITS,
    MIN_REPS,
    MIXING_ANGLE,
    FeaturePlacement,
    QuantumFeatureTransformer,
    QuantumRidgeRegressor,
    build_encoding_circuit,
    feature_qubit_mapping,
)

ENCODINGS = ("zz", "ry", "ry_rz")
SIMULATION_METHODS = ("statevector", "matrix_product_state")
INVALID_COUNTS = (
    True,
    False,
    np.bool_(True),
    np.bool_(False),
    0,
    -1,
    2.0,
    1.5,
    np.nan,
    np.inf,
    "2",
    None,
)


def _manual_circuit(
    row: np.ndarray, *, n_qubits: int, encoding: str, reps: int, gamma: float = 1.0
) -> QuantumCircuit:
    """Independent numeric reference, used only with at most five qubits."""
    axes = {"zz": ("z",), "ry": ("ry",), "ry_rz": ("ry", "rz")}[encoding]
    values = gamma * np.asarray(row, dtype=np.float64)
    capacity = n_qubits * len(axes)
    n_layers = (len(values) + capacity - 1) // capacity
    padded = np.zeros(n_layers * capacity)
    padded[: len(values)] = values
    layers = padded.reshape(n_layers, n_qubits, len(axes))
    circuit = QuantumCircuit(n_qubits)
    steps_left = reps * n_layers
    for _ in range(reps):
        for layer in layers:
            if encoding == "zz":
                circuit.h(range(n_qubits))
                for qubit in range(n_qubits):
                    circuit.p(2 * layer[qubit, 0], qubit)
                for qubit in range(n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
                    circuit.p(
                        2 * (np.pi - layer[qubit, 0]) * (np.pi - layer[qubit + 1, 0]),
                        qubit + 1,
                    )
                    circuit.cx(qubit, qubit + 1)
            else:
                for qubit in range(n_qubits):
                    circuit.ry(layer[qubit, 0], qubit)
                    if encoding == "ry_rz":
                        circuit.rz(layer[qubit, 1], qubit)
                    # RX mixes phase information into the subsequent Z readout.
                    circuit.rx(np.pi / 4, qubit)
                if steps_left > 1:
                    for parity in (0, 1):
                        for qubit in range(parity, n_qubits - 1, 2):
                            circuit.cz(qubit, qubit + 1)
            steps_left -= 1
    return circuit


def _manual_expectations(
    X: np.ndarray, *, n_qubits: int, encoding: str, reps: int, gamma: float
) -> np.ndarray:
    """Reference Z features without the production factory, mapping or estimator."""
    basis_indices = np.arange(2**n_qubits)
    expectations = []
    for row in X:
        circuit = _manual_circuit(
            row, n_qubits=n_qubits, encoding=encoding, reps=reps, gamma=gamma
        )
        probabilities = Statevector.from_instruction(circuit).probabilities()
        expectations.append(
            [
                probabilities @ (1 - 2 * ((basis_indices >> qubit) & 1))
                for qubit in range(n_qubits)
            ]
        )
    return np.asarray(expectations, dtype=np.float64)


@pytest.fixture
def packed_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Eight synthetic rows; all seven columns need training-fitted scaling."""
    rng = np.random.default_rng(321)
    X = rng.normal(size=(8, 7)) * np.linspace(0.3, 1.3, 7) + 2 * np.arange(7)
    y = 2.5 + 0.4 * X[:, 0] + 0.1 * X[:, 1]
    return X, y


def test_public_encoding_constants_and_placement_fields():
    assert ENCODING_AXES == {"zz": ("z",), "ry": ("ry",), "ry_rz": ("ry", "rz")}
    assert MIN_REPS == {"zz": 2, "ry": 1, "ry_rz": 1}
    assert MIN_QUBITS == {"zz": 2, "ry": 1, "ry_rz": 1}
    assert MIXING_ANGLE == pytest.approx(np.pi / 4, abs=1e-15)
    assert is_dataclass(FeaturePlacement)
    assert [field.name for field in fields(FeaturePlacement)] == [
        "feature_index",
        "qubit_index",
        "upload_layer",
        "rotation_index",
    ]


@pytest.mark.parametrize(
    "n_features, n_qubits, rotations_per_qubit",
    [
        (1, 1, 1),
        (2, 7, 3),
        (5, 3, 1),
        (12, 4, 3),
        (22, 5, 2),
        (23, 5, 2),
        (31, 7, 3),
        (24, 8, 1),
        (24, 16, 1),
        (24, 24, 1),
        (np.int64(23), np.int32(5), np.int64(2)),
    ],
)
def test_mapping_uses_capacity_formula_without_missing_or_duplicate_inputs(
    n_features, n_qubits, rotations_per_qubit
):
    mapping = feature_qubit_mapping(n_features, n_qubits, rotations_per_qubit)
    capacity = n_qubits * rotations_per_qubit
    assert isinstance(mapping, tuple)
    assert len(mapping) == n_features
    assert all(isinstance(placement, FeaturePlacement) for placement in mapping)
    assert [placement.feature_index for placement in mapping] == list(range(n_features))
    actual = [
        (p.feature_index, p.qubit_index, p.upload_layer, p.rotation_index)
        for p in mapping
    ]
    expected = [
        (
            i,
            (i % capacity) // rotations_per_qubit,
            i // capacity,
            i % rotations_per_qubit,
        )
        for i in range(n_features)
    ]
    assert actual == expected
    slots = {(p.upload_layer, p.qubit_index, p.rotation_index) for p in mapping}
    assert len(slots) == n_features
    assert (
        max(p.upload_layer for p in mapping) + 1
        == (n_features + capacity - 1) // capacity
    )


def test_mapping_default_has_one_rotation_slot_per_qubit():
    assert feature_qubit_mapping(11, 4) == feature_qubit_mapping(11, 4, 1)


@pytest.mark.parametrize("parameter", ["n_features", "n_qubits", "rotations_per_qubit"])
@pytest.mark.parametrize("value", INVALID_COUNTS)
def test_mapping_rejects_non_positive_or_non_integral_counts(parameter, value):
    parameters = {"n_features": 7, "n_qubits": 3, "rotations_per_qubit": 2}
    parameters[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        feature_qubit_mapping(**parameters)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("n_features, n_qubits", [(1, 2), (7, 3), (23, 5), (3, 5)])
def test_factory_preserves_every_independent_input_parameter(
    encoding, n_features, n_qubits
):
    circuit = build_encoding_circuit(n_features, n_qubits, encoding=encoding)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == n_qubits
    assert circuit.num_parameters == n_features
    parameters = list(circuit.parameters)
    assert [parameter.name for parameter in parameters] == [
        f"x[{i}]" for i in range(n_features)
    ]
    assert [parameter.index for parameter in parameters] == list(range(n_features))
    assert all(
        isinstance(parameter.vector, ParameterVector) for parameter in parameters
    )
    assert all(parameter.vector.name == "x" for parameter in parameters)
    assert all(len(parameter.vector) == n_features for parameter in parameters)
    assert len(set(parameters)) == n_features
    bound = circuit.assign_parameters(np.linspace(0.1, 0.9, n_features))
    assert bound.num_parameters == 0


@pytest.mark.parametrize("parameter", ["n_features", "n_qubits", "reps"])
@pytest.mark.parametrize("value", INVALID_COUNTS)
def test_factory_validates_integer_counts(parameter, value):
    parameters = {"n_features": 7, "n_qubits": 3, "reps": 2, "encoding": "ry_rz"}
    parameters[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        build_encoding_circuit(**parameters)


@pytest.mark.parametrize("encoding", [None, True, "", "rx", "RY", ["ry"]])
def test_factory_rejects_unknown_encodings(encoding):
    with pytest.raises(ValueError, match="encoding"):
        build_encoding_circuit(7, 3, encoding=encoding)


@pytest.mark.parametrize(
    "n_qubits, reps, parameter", [(1, 2, "n_qubits"), (2, 1, "reps")]
)
def test_zz_factory_requires_two_qubits_and_two_passes(n_qubits, reps, parameter):
    with pytest.raises(ValueError, match=parameter):
        build_encoding_circuit(7, n_qubits, encoding="zz", reps=reps)


def test_factory_accepts_numpy_integer_counts():
    circuit = build_encoding_circuit(
        np.int64(7), np.int32(3), encoding="ry_rz", reps=np.int64(1)
    )
    assert circuit.num_qubits == 3
    assert circuit.num_parameters == 7


@pytest.mark.parametrize(
    "encoding, reps",
    [("zz", 2), ("zz", 3), ("ry", 1), ("ry", 3), ("ry_rz", 1), ("ry_rz", 3)],
)
@pytest.mark.parametrize("n_features, n_qubits", [(7, 3), (3, 5)])
def test_factory_matches_independent_numeric_circuit_with_padding_and_full_passes(
    encoding, reps, n_features, n_qubits
):
    row = np.random.default_rng(77).uniform(-0.8, 1.2, size=n_features)
    circuit = build_encoding_circuit(n_features, n_qubits, encoding=encoding, reps=reps)
    reference = _manual_circuit(row, n_qubits=n_qubits, encoding=encoding, reps=reps)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit.assign_parameters(row)).data,
        Statevector.from_instruction(reference).data,
        atol=1e-12,
        rtol=1e-12,
    )


def test_factory_defaults_to_two_pass_zz():
    row = np.random.default_rng(11).uniform(-0.7, 0.9, size=7)
    circuit = build_encoding_circuit(7, 3)
    reference = _manual_circuit(row, n_qubits=3, encoding="zz", reps=2)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit.assign_parameters(row)).data,
        Statevector.from_instruction(reference).data,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("encoding", ["ry", "ry_rz"])
@pytest.mark.parametrize("n_qubits", [3, 4, 5])
def test_angle_axes_zero_padding_mixers_and_even_odd_cz_schedule(encoding, n_qubits):
    n_features, reps = 7, 2
    axes = ENCODING_AXES[encoding]
    capacity = n_qubits * len(axes)
    n_layers = (n_features + capacity - 1) // capacity
    circuit = build_encoding_circuit(n_features, n_qubits, encoding=encoding, reps=reps)
    parameters = list(circuit.parameters)
    expected = []
    for step in range(reps * n_layers):
        for qubit in range(n_qubits):
            for rotation, axis in enumerate(axes):
                index = (step % n_layers) * capacity + qubit * len(axes) + rotation
                angle = parameters[index] if index < n_features else 0.0
                expected.append((axis, (qubit,), (angle,)))
            expected.append(("rx", (qubit,), (np.pi / 4,)))
        if step + 1 < reps * n_layers:
            for parity in (0, 1):
                for qubit in range(parity, n_qubits - 1, 2):
                    expected.append(("cz", (qubit, qubit + 1), ()))
    actual = [
        (
            instruction.operation.name,
            tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
            tuple(instruction.operation.params),
        )
        for instruction in circuit.data
    ]
    # A final CZ cannot change any local Z expectation, so it must be omitted.
    assert actual == expected


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("simulation_method", SIMULATION_METHODS)
@pytest.mark.parametrize("n_features, n_qubits", [(7, 3), (3, 5), (1, 2)])
def test_packed_and_spare_qubit_features_match_independent_z_reference(
    encoding, simulation_method, n_features, n_qubits
):
    X = np.random.default_rng(29).uniform(-0.9, 1.1, size=(3, n_features))
    transformer = QuantumFeatureTransformer(
        n_qubits=n_qubits,
        encoding=encoding,
        simulation_method=simulation_method,
        reps=2,
        gamma=0.37,
        batch_size=2,
    ).fit(X)
    capacity = n_qubits * len(ENCODING_AXES[encoding])
    assert transformer.n_features_in_ == n_features
    assert transformer.n_qubits_ == transformer.n_features_out_ == n_qubits
    assert transformer.n_upload_layers_ == (n_features + capacity - 1) // capacity
    assert transformer.feature_mapping_ == feature_qubit_mapping(
        n_features, n_qubits, len(ENCODING_AXES[encoding])
    )
    assert transformer.circuit_.num_qubits == n_qubits
    assert transformer.circuit_.num_parameters == n_features
    assert len(transformer.observables_) == n_qubits
    for qubit, observable in enumerate(transformer.observables_):
        label = "I" * (n_qubits - qubit - 1) + "Z" + "I" * qubit
        assert observable.to_list() == [(label, 1.0)]
    if simulation_method == "statevector":
        assert isinstance(transformer.estimator_, StatevectorEstimator)
        assert transformer.estimator_.default_precision == 0.0
    else:
        assert isinstance(transformer.estimator_, AerEstimatorV2)
        options = transformer.estimator_.options
        assert options.default_precision == 0.0
        assert options.backend_options["method"] == "matrix_product_state"
        assert (
            options.backend_options["matrix_product_state_truncation_threshold"] == 0.0
        )
    actual = transformer.transform(X)
    expected = _manual_expectations(
        X, n_qubits=n_qubits, encoding=encoding, reps=2, gamma=0.37
    )
    tolerance = 1e-9 if simulation_method == "matrix_product_state" else 1e-12
    assert actual.shape == (len(X), n_qubits)
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual) <= 1 + tolerance)
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("n_features, n_qubits", [(7, 3), (3, 5)])
def test_output_names_use_qubit_count_but_validate_input_width(
    encoding, n_features, n_qubits
):
    transformer = QuantumFeatureTransformer(n_qubits=n_qubits, encoding=encoding).fit(
        np.ones((2, n_features))
    )
    expected = [f"z_expectation_{qubit}" for qubit in range(n_qubits)]
    names = [f"input_{i}" for i in range(n_features)]
    np.testing.assert_array_equal(transformer.get_feature_names_out(), expected)
    np.testing.assert_array_equal(transformer.get_feature_names_out(names), expected)
    for invalid_names in (names[:-1], names + ["extra"], [names], "input"):
        with pytest.raises(ValueError):
            transformer.get_feature_names_out(invalid_names)
    with pytest.raises(ValueError):
        transformer.get_feature_names_out([f"input_{i}" for i in range(n_qubits)])
    with pytest.raises(ValueError, match="features"):
        transformer.transform(np.ones((2, n_qubits)))


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("simulation_method", SIMULATION_METHODS)
def test_packed_batches_are_exact_deterministic_ordered_and_non_mutating(
    encoding, simulation_method, monkeypatch
):
    X = np.random.default_rng(17).uniform(-0.6, 0.9, size=(5, 7))
    original = X.copy()
    parameters = dict(
        n_qubits=3,
        encoding=encoding,
        simulation_method=simulation_method,
        reps=2,
        gamma=0.41,
    )
    transformer = QuantumFeatureTransformer(**parameters, batch_size=2).fit(X)
    run = transformer.estimator_.run
    batch_lengths, precisions = [], []

    def record_run(pubs, *, precision=None):
        pubs = list(pubs)
        coerced = [EstimatorPub.coerce(pub, precision=precision) for pub in pubs]
        batch_lengths.append(
            sum(int(np.prod(pub.parameter_values.shape)) for pub in coerced)
        )
        precisions.extend(pub.precision for pub in coerced)
        return run(pubs, precision=precision)

    monkeypatch.setattr(transformer.estimator_, "run", record_run)
    actual = transformer.transform(X)
    assert batch_lengths == [2, 2, 1]
    assert precisions and all(precision == 0.0 for precision in precisions)
    np.testing.assert_array_equal(transformer.transform(X), actual)
    tolerance = 1e-9 if simulation_method == "matrix_product_state" else 1e-12
    order = np.array([3, 0, 4, 1, 2])
    np.testing.assert_allclose(
        transformer.transform(X[order]), actual[order], atol=tolerance, rtol=tolerance
    )
    other = QuantumFeatureTransformer(**parameters, batch_size=10).fit(X)
    np.testing.assert_allclose(
        other.transform(X), actual, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("n_qubits", [8, 16, 24])
def test_24_input_output_width_uses_requested_qubits_with_mps(encoding, n_qubits):
    # Wider circuits never allocate a full 16/24-qubit statevector in this suite.
    X = np.random.default_rng(81).uniform(-0.8, 0.9, size=(2, 24))
    transformer = QuantumFeatureTransformer(
        n_qubits=n_qubits,
        encoding=encoding,
        simulation_method="matrix_product_state",
        gamma=0.31,
        batch_size=1,
    ).fit(X)
    actual = transformer.transform(X)
    capacity = n_qubits * len(ENCODING_AXES[encoding])
    assert transformer.n_features_in_ == 24
    assert transformer.n_qubits_ == transformer.n_features_out_ == n_qubits
    assert transformer.n_upload_layers_ == (24 + capacity - 1) // capacity
    assert len(transformer.feature_mapping_) == 24
    assert isinstance(transformer.estimator_, AerEstimatorV2)
    assert actual.shape == (2, n_qubits)
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual) <= 1 + 1e-9)
    assert len(transformer.get_feature_names_out()) == n_qubits


@pytest.mark.parametrize("encoding", ["ry", "ry_rz"])
@pytest.mark.parametrize("n_qubits", [None, 1, np.int64(1)])
def test_angle_encodings_allow_one_input_one_qubit_and_one_pass(encoding, n_qubits):
    X = np.array([[0.2], [-0.6], [0.9]])
    parameters = dict(n_qubits=n_qubits, encoding=encoding, reps=1, gamma=0.4)
    transformer = QuantumFeatureTransformer(**parameters).fit(X)
    assert transformer.n_features_in_ == transformer.n_qubits_ == 1
    assert transformer.n_features_out_ == transformer.n_upload_layers_ == 1
    np.testing.assert_allclose(
        transformer.transform(X),
        _manual_expectations(X, n_qubits=1, encoding=encoding, reps=1, gamma=0.4),
        atol=1e-12,
        rtol=1e-12,
    )
    model = QuantumRidgeRegressor(**parameters).fit(X, [2.75, 2.75, 2.75])
    assert model.quantum_features(X).shape == (3, 1)
    np.testing.assert_allclose(model.predict(X), 2.75, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "estimator_type", [QuantumFeatureTransformer, QuantumRidgeRegressor]
)
@pytest.mark.parametrize(
    "parameter, value",
    [("n_qubits", value) for value in INVALID_COUNTS if value is not None]
    + [("n_qubits", 1)]
    + [("encoding", value) for value in (None, True, "", "rx", "RY", ["ry"])]
    + [
        ("simulation_method", value)
        for value in (
            None,
            True,
            "",
            "mps",
            "density_matrix",
            ["statevector"],
        )
    ],
)
def test_invalid_new_estimator_configuration_raises_value_error(
    estimator_type, parameter, value
):
    X = [[0.1, 0.4, -0.3], [0.7, -0.2, 0.5]]
    with pytest.raises(ValueError, match=parameter):
        estimator_type(**{parameter: value}).fit(X, [1.0, 2.0])


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_every_input_changes_at_least_one_z_readout_at_generic_angles(encoding):
    n_features = 7
    point = np.random.default_rng(731).uniform(0.2, 1.1, size=n_features)
    X = np.vstack([point, point + 0.173 * np.eye(n_features)])
    transformer = QuantumFeatureTransformer(
        n_qubits=3, encoding=encoding, gamma=0.7, batch_size=len(X)
    ).fit(X)
    features = transformer.transform(X)
    changes = np.max(np.abs(features[1:] - features[0]), axis=1)
    assert np.all(changes > 1e-6), (
        f"{encoding} discarded inputs {np.flatnonzero(changes <= 1e-6).tolist()}; "
        f"per-input changes: {changes}"
    )


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_packed_refit_rebuilds_mapping_for_a_new_input_width(encoding):
    transformer = QuantumFeatureTransformer(n_qubits=3, encoding=encoding, gamma=0.6)
    transformer.fit(np.random.default_rng(5).normal(size=(3, 7)))
    previous_circuit = transformer.circuit_
    X = np.random.default_rng(6).normal(size=(3, 2))
    assert transformer.fit(X) is transformer
    assert transformer.circuit_ is not previous_circuit
    assert transformer.n_features_in_ == 2
    assert transformer.n_qubits_ == transformer.n_features_out_ == 3
    assert transformer.n_upload_layers_ == 1
    assert transformer.feature_mapping_ == feature_qubit_mapping(
        2, 3, len(ENCODING_AXES[encoding])
    )
    np.testing.assert_allclose(
        transformer.transform(X),
        _manual_expectations(X, n_qubits=3, encoding=encoding, reps=2, gamma=0.6),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("simulation_method", SIMULATION_METHODS)
def test_packed_ridge_matches_manual_pipeline_and_keeps_scaling_training_only(
    packed_training_data, encoding, simulation_method
):
    X, y = packed_training_data
    X_validation = X[:2] + np.linspace(9.0, 15.0, X.shape[1])
    originals = (X.copy(), y.copy(), X_validation.copy())
    parameters = dict(
        n_qubits=3,
        encoding=encoding,
        simulation_method=simulation_method,
        reps=2,
        gamma=0.37,
        batch_size=3,
    )
    model = QuantumRidgeRegressor(**parameters, alpha=0.6, clip_negative=False)
    assert model.fit(X, y) is model
    assert list(model.pipeline_.named_steps) == ["scaler", "quantum", "ridge"]
    scaler = model.pipeline_.named_steps["scaler"]
    quantum = model.pipeline_.named_steps["quantum"]
    ridge = model.pipeline_.named_steps["ridge"]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(quantum, QuantumFeatureTransformer)
    assert isinstance(ridge, Ridge)
    assert quantum.get_params() == parameters
    assert model.n_features_in_ == scaler.n_features_in_ == quantum.n_features_in_ == 7
    assert quantum.n_qubits_ == quantum.n_features_out_ == ridge.n_features_in_ == 3
    assert ridge.alpha == 0.6
    before = {
        name: np.array(getattr(scaler, name), copy=True)
        for name in ("mean_", "var_", "scale_", "n_samples_seen_")
    }
    reference_scaler = StandardScaler().fit(X)
    np.testing.assert_allclose(
        scaler.mean_, reference_scaler.mean_, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        scaler.scale_, reference_scaler.scale_, atol=1e-12, rtol=1e-12
    )
    train_features = _manual_expectations(
        reference_scaler.transform(X), n_qubits=3, encoding=encoding, reps=2, gamma=0.37
    )
    validation_features = _manual_expectations(
        reference_scaler.transform(X_validation),
        n_qubits=3,
        encoding=encoding,
        reps=2,
        gamma=0.37,
    )
    reference_ridge = Ridge(alpha=0.6).fit(train_features, y)
    tolerance = 1e-9 if simulation_method == "matrix_product_state" else 1e-12
    np.testing.assert_allclose(
        model.quantum_features(X), train_features, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        model.quantum_features(X_validation),
        validation_features,
        atol=tolerance,
        rtol=tolerance,
    )
    actual = model.predict(X_validation)
    assert actual.shape == (len(X_validation),)
    np.testing.assert_allclose(
        actual,
        reference_ridge.predict(validation_features),
        atol=tolerance,
        rtol=tolerance,
    )
    assert model.pipeline_.named_steps["scaler"] is scaler
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(scaler, name), value)
    for array, original in zip((X, y, X_validation), originals):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize(
    "estimator_type", [QuantumFeatureTransformer, QuantumRidgeRegressor]
)
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_sklearn_clone_preserves_packed_parameters_not_fitted_state(
    packed_training_data, estimator_type, encoding
):
    X, y = packed_training_data
    parameters = dict(
        n_qubits=3,
        encoding=encoding,
        simulation_method="matrix_product_state",
        reps=2,
        gamma=0.4,
        batch_size=3,
    )
    if estimator_type is QuantumRidgeRegressor:
        parameters.update(alpha=0.2, clip_negative=False)
    estimator = estimator_type(**parameters).fit(X, y)
    cloned = clone(estimator)
    assert cloned is not estimator
    assert cloned.get_params() == parameters
    for attribute in (
        "n_features_in_",
        "n_features_out_",
        "n_qubits_",
        "feature_mapping_",
        "n_upload_layers_",
        "circuit_",
        "estimator_",
        "pipeline_",
    ):
        assert not hasattr(cloned, attribute)
    cloned.fit(X, y)
    method = "transform" if estimator_type is QuantumFeatureTransformer else "predict"
    np.testing.assert_allclose(
        getattr(cloned, method)(X[:2]),
        getattr(estimator, method)(X[:2]),
        atol=1e-9,
        rtol=1e-9,
    )


def test_packed_model_cross_validation_fits_scalers_on_training_folds_only(
    packed_training_data, monkeypatch
):
    X, y = packed_training_data
    cv = KFold(n_splits=2)
    fold_inputs = [X[train_indices] for train_indices, _ in cv.split(X, y)]
    fit_inputs = []
    fit = StandardScaler.fit

    def record_fit(self, X, y=None, **kwargs):
        fit_inputs.append(np.array(X, copy=True))
        return fit(self, X, y=y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    estimator = QuantumRidgeRegressor(
        n_qubits=3,
        encoding="ry_rz",
        reps=1,
        simulation_method="matrix_product_state",
        batch_size=4,
        clip_negative=False,
    )
    search = GridSearchCV(
        estimator,
        {"alpha": [0.1, 1.0]},
        cv=cv,
        scoring="neg_mean_absolute_error",
        refit=False,
        n_jobs=1,
        error_score="raise",
    ).fit(X, y)
    assert not hasattr(estimator, "pipeline_")
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert len(fit_inputs) == 2 * 2
    for fold_X in fold_inputs:
        assert sum(np.array_equal(fit_X, fold_X) for fit_X in fit_inputs) == 2
