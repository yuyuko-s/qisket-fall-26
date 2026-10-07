"""Simulator-only quantum model contracts, using small synthetic training/CV arrays."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import EstimatorPub
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qm9dipole.models.quantum import QuantumFeatureTransformer, QuantumRidgeRegressor


def _reference_expectations(
    X: np.ndarray,
    *,
    reps: int = 2,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute exact Z expectations without the module's circuit or estimator."""
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]
    basis_indices = np.arange(2**n_features)
    expectations = []
    for row in X:
        angles = gamma * row
        circuit = QuantumCircuit(n_features)
        # ZZFeatureMap uses P(2*x) and nearest-neighbour ZZ phases, repeated after H.
        for _ in range(reps):
            circuit.h(range(n_features))
            for qubit, angle in enumerate(angles):
                circuit.p(2 * angle, qubit)
            for qubit in range(n_features - 1):
                circuit.cx(qubit, qubit + 1)
                circuit.p(
                    2 * (np.pi - angles[qubit]) * (np.pi - angles[qubit + 1]),
                    qubit + 1,
                )
                circuit.cx(qubit, qubit + 1)
        probabilities = Statevector.from_instruction(circuit).probabilities()
        # Qiskit qubit 0 is the least-significant bit (rightmost Pauli label).
        expectations.append(
            [
                probabilities @ (1 - 2 * ((basis_indices >> qubit) & 1))
                for qubit in range(n_features)
            ]
        )
    return np.asarray(expectations, dtype=np.float64)


@pytest.fixture
def training_data() -> tuple[np.ndarray, np.ndarray]:
    """Twelve synthetic training rows with nonzero means; targets are in debye."""
    rng = np.random.default_rng(21)
    X = rng.normal(size=(12, 4)) * [0.4, 1.2, 2.5, 0.8] + [3.0, -2.0, 1.5, 8.0]
    y = 2.5 + 0.3 * X[:, 0] + 0.1 * X[:, 1]
    return X, y


def test_transformer_defaults_and_fit_contract(training_data):
    X, _ = training_data
    transformer = QuantumFeatureTransformer()
    assert isinstance(transformer, TransformerMixin) and isinstance(
        transformer, BaseEstimator
    )
    assert transformer.get_params() == {
        "reps": 2,
        "gamma": 1.0,
        "batch_size": 128,
        "n_qubits": None,
        "encoding": "zz",
        "simulation_method": "statevector",
    }
    assert transformer.fit(X) is transformer
    assert transformer.n_features_in_ == 4
    assert transformer.n_qubits_ == transformer.n_features_out_ == 4
    assert transformer.n_upload_layers_ == 1
    assert transformer.circuit_.num_qubits == 4
    assert transformer.circuit_.num_parameters == 4
    assert isinstance(transformer.estimator_, StatevectorEstimator)
    assert len(transformer.observables_) == 4
    for qubit, observable in enumerate(transformer.observables_):
        assert isinstance(observable, SparsePauliOp)
        label = "I" * (3 - qubit) + "Z" + "I" * qubit
        assert observable.to_list() == [(label, 1.0)]
    expected_circuit = zz_feature_map(4, reps=2, entanglement="linear")
    np.testing.assert_allclose(
        Statevector.from_instruction(transformer.circuit_.assign_parameters(X[0])).data,
        Statevector.from_instruction(expected_circuit.assign_parameters(X[0])).data,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("n_features, reps, gamma", [(4, 2, 0.37), (8, 3, 0.83)])
def test_exact_features_qubit_order_gamma_and_nonconstant_signal(
    n_features, reps, gamma
):
    rng = np.random.default_rng(42)
    X = rng.uniform(-1.1, 1.3, size=(5, n_features))
    transformer = QuantumFeatureTransformer(reps=reps, gamma=gamma, batch_size=2).fit(X)
    actual = transformer.transform(X)
    expected = _reference_expectations(X, reps=reps, gamma=gamma)
    assert actual.shape == X.shape
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual) <= 1 + 1e-12)
    assert np.all(np.ptp(actual, axis=0) > 1e-3)
    assert not np.allclose(expected, expected[:, ::-1]), (
        "fixture must detect reversed qubits"
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_numeric_inputs_are_converted_to_float64(dtype):
    X = np.arange(6).reshape(3, 2).astype(dtype)
    transformer = QuantumFeatureTransformer(gamma=0.3).fit(X.tolist())
    actual = transformer.transform(X)
    assert actual.dtype == np.float64
    assert actual.shape == (3, 2)
    np.testing.assert_allclose(
        actual,
        _reference_expectations(X, gamma=0.3),
        atol=1e-12,
        rtol=1e-12,
    )


def test_batches_are_exact_deterministic_ordered_and_do_not_mutate_input(
    training_data, monkeypatch
):
    X, _ = training_data
    X = X[:7].copy()
    original = X.copy()
    transformer = QuantumFeatureTransformer(reps=3, gamma=0.41, batch_size=3).fit(X)
    run = transformer.estimator_.run
    batch_lengths, precisions = [], []

    def record_run(pubs, *, precision=None):
        pubs = list(pubs)
        effective_precision = (
            transformer.estimator_.default_precision if precision is None else precision
        )
        coerced = [
            EstimatorPub.coerce(pub, precision=effective_precision) for pub in pubs
        ]
        batch_lengths.append(
            sum(int(np.prod(pub.parameter_values.shape)) for pub in coerced)
        )
        precisions.extend(pub.precision for pub in coerced)
        return run(pubs, precision=precision)

    monkeypatch.setattr(transformer.estimator_, "run", record_run)
    actual = transformer.transform(X)
    assert batch_lengths == [3, 3, 1]
    assert precisions and all(precision == 0 for precision in precisions)
    np.testing.assert_array_equal(transformer.transform(X), actual)
    order = np.random.default_rng(9).permutation(len(X))
    np.testing.assert_array_equal(transformer.transform(X[order]), actual[order])
    for batch_size in (1, 20):
        other = QuantumFeatureTransformer(
            reps=3, gamma=0.41, batch_size=batch_size
        ).fit(X)
        np.testing.assert_array_equal(other.transform(X), actual)
    np.testing.assert_array_equal(X, original)


def test_feature_names_and_optional_name_length(training_data):
    X, _ = training_data
    transformer = QuantumFeatureTransformer().fit(X)
    expected = [f"z_expectation_{qubit}" for qubit in range(4)]
    np.testing.assert_array_equal(transformer.get_feature_names_out(), expected)
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(["a", "b", "c", "d"]), expected
    )
    for names in (["a", "b", "c"], ["a", "b", "c", "d", "e"]):
        with pytest.raises(ValueError):
            transformer.get_feature_names_out(names)


@pytest.mark.parametrize(
    "estimator_type, method",
    [
        (QuantumFeatureTransformer, "transform"),
        (QuantumFeatureTransformer, "get_feature_names_out"),
        (QuantumRidgeRegressor, "predict"),
        (QuantumRidgeRegressor, "quantum_features"),
    ],
)
def test_unfitted_operations_raise_not_fitted(estimator_type, method):
    argument = None if method == "get_feature_names_out" else np.ones((2, 4))
    with pytest.raises(NotFittedError):
        getattr(estimator_type(), method)(argument)


@pytest.mark.parametrize(
    "estimator_type, method",
    [
        (QuantumFeatureTransformer, "transform"),
        (QuantumRidgeRegressor, "predict"),
        (QuantumRidgeRegressor, "quantum_features"),
    ],
)
@pytest.mark.parametrize("width", [2, 5])
def test_mismatched_width_mentions_features(
    training_data, estimator_type, method, width
):
    X, y = training_data
    estimator = estimator_type().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        getattr(estimator, method)(np.ones((2, width)))


@pytest.mark.parametrize(
    "estimator_type", [QuantumFeatureTransformer, QuantumRidgeRegressor]
)
@pytest.mark.parametrize(
    "parameter, value",
    [
        ("reps", 1),
        ("reps", 0),
        ("reps", -2),
        ("reps", 2.0),
        ("reps", 2.5),
        ("reps", True),
        ("reps", False),
        ("reps", np.bool_(True)),
        ("reps", "2"),
        ("reps", None),
        ("gamma", 0.0),
        ("gamma", -0.5),
        ("gamma", np.nan),
        ("gamma", np.inf),
        ("gamma", -np.inf),
        ("gamma", 1 + 0.5j),
        ("gamma", "1"),
        ("gamma", None),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", 2.0),
        ("batch_size", 1.5),
        ("batch_size", True),
        ("batch_size", False),
        ("batch_size", np.bool_(False)),
        ("batch_size", "2"),
        ("batch_size", None),
    ],
)
def test_invalid_shared_hyperparameters_raise_value_error(
    estimator_type, parameter, value
):
    with pytest.raises(ValueError):
        estimator_type(**{parameter: value}).fit([[0.1, 0.4], [0.7, -0.2]], [1.0, 2.0])


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("alpha", -0.1),
        ("alpha", np.nan),
        ("alpha", np.inf),
        ("alpha", -np.inf),
        ("alpha", 1 + 0.5j),
        ("alpha", "1"),
        ("alpha", None),
        ("clip_negative", 0),
        ("clip_negative", 1),
        ("clip_negative", "false"),
        ("clip_negative", None),
    ],
)
def test_invalid_regressor_hyperparameters_raise_value_error(parameter, value):
    with pytest.raises(ValueError):
        QuantumRidgeRegressor(**{parameter: value}).fit(
            [[0.1, 0.4], [0.7, -0.2]], [1.0, 2.0]
        )


@pytest.mark.parametrize(
    "estimator_type", [QuantumFeatureTransformer, QuantumRidgeRegressor]
)
def test_numpy_integral_and_real_hyperparameters_are_accepted(estimator_type):
    estimator = estimator_type(
        reps=np.int64(2), gamma=np.float64(0.5), batch_size=np.int64(1)
    )
    assert estimator.fit([[0.1, 0.4], [0.7, -0.2]], [1.0, 2.0]) is estimator


@pytest.mark.parametrize(
    "estimator_type", [QuantumFeatureTransformer, QuantumRidgeRegressor]
)
@pytest.mark.parametrize(
    "X",
    [
        pytest.param(np.array(1.0), id="scalar"),
        pytest.param(np.ones(3), id="one-dimensional"),
        pytest.param(np.ones((3, 2, 1)), id="three-dimensional"),
        pytest.param(np.ones((3, 1)), id="one-feature"),
        pytest.param(np.empty((3, 0)), id="no-features"),
        pytest.param(np.empty((0, 2)), id="no-samples"),
        pytest.param([[1, 2], [3], [4, 5]], id="ragged"),
        pytest.param([["bad", 1], [2, 3], [4, 5]], id="nonnumeric"),
        pytest.param([[np.nan, 1], [2, 3], [4, 5]], id="nan"),
        pytest.param([[np.inf, 1], [2, 3], [4, 5]], id="positive-inf"),
        pytest.param([[-np.inf, 1], [2, 3], [4, 5]], id="negative-inf"),
        pytest.param(np.full((3, 2), 1 + 0.5j), id="complex"),
    ],
)
def test_invalid_features_rejected_at_fit_and_inference(estimator_type, X):
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        estimator_type().fit(X, y)
    estimator = estimator_type().fit([[0.1, 0.4], [0.7, -0.2], [0.3, 0.8]], y)
    methods = (
        ("transform",)
        if estimator_type is QuantumFeatureTransformer
        else ("predict", "quantum_features")
    )
    for method in methods:
        with pytest.raises(ValueError):
            getattr(estimator, method)(X)


@pytest.mark.parametrize(
    "y",
    [
        pytest.param(None, id="missing"),
        pytest.param(np.array(2.0), id="scalar"),
        pytest.param(np.ones(3), id="wrong-length"),
        pytest.param(np.ones((4, 1)), id="column-vector"),
        pytest.param(np.ones((4, 2)), id="multiple-targets"),
        pytest.param([1.0, 2.0, np.nan, 4.0], id="nan"),
        pytest.param([1.0, 2.0, np.inf, 4.0], id="positive-inf"),
        pytest.param([1.0, 2.0, -np.inf, 4.0], id="negative-inf"),
        pytest.param([1.0, 2.0, "bad", 4.0], id="nonnumeric"),
        pytest.param(np.full(4, 1 + 0.5j), id="complex"),
    ],
)
def test_invalid_targets_raise_value_error(y):
    with pytest.raises(ValueError):
        QuantumRidgeRegressor().fit(np.arange(8).reshape(4, 2), y)


@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_regression_matches_independent_scaler_quantum_features_and_ridge(
    training_data, alpha
):
    X, y = training_data
    X_validation = np.random.default_rng(8).normal(size=(3, 4)) + [3.0, -2.0, 1.5, 8.0]
    model = QuantumRidgeRegressor(
        alpha=alpha, reps=3, gamma=0.43, batch_size=5, clip_negative=False
    )
    assert model.fit(X, y) is model
    assert isinstance(model, RegressorMixin) and isinstance(model, BaseEstimator)
    assert isinstance(model.pipeline_, Pipeline)
    assert list(model.pipeline_.named_steps) == ["scaler", "quantum", "ridge"]
    assert isinstance(model.pipeline_.named_steps["scaler"], StandardScaler)
    quantum = model.pipeline_.named_steps["quantum"]
    assert isinstance(quantum, QuantumFeatureTransformer)
    assert (quantum.reps, quantum.gamma, quantum.batch_size) == (3, 0.43, 5)
    assert quantum.circuit_.num_qubits == model.n_features_in_ == X.shape[1]
    assert isinstance(model.pipeline_.named_steps["ridge"], Ridge)
    assert model.pipeline_.named_steps["ridge"].alpha == alpha

    scaler = StandardScaler().fit(X)
    train_features = _reference_expectations(scaler.transform(X), reps=3, gamma=0.43)
    validation_features = _reference_expectations(
        scaler.transform(X_validation), reps=3, gamma=0.43
    )
    ridge = Ridge(alpha=alpha).fit(train_features, y)
    actual = model.predict(X_validation)
    assert actual.shape == (len(X_validation),)
    np.testing.assert_allclose(
        actual, ridge.predict(validation_features), atol=1e-11, rtol=1e-10
    )
    np.testing.assert_allclose(
        model.quantum_features(X_validation),
        validation_features,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        model.quantum_features(X_validation),
        model.pipeline_[:-1].transform(X_validation),
    )


def test_validation_predictions_preserve_training_only_scaling(training_data):
    X, y = training_data
    X_validation = X[:3] + [12.0, -18.0, 25.0, -9.0]
    original_X, original_y, original_validation = (
        X.copy(),
        y.copy(),
        X_validation.copy(),
    )
    model = QuantumRidgeRegressor(gamma=0.37).fit(X, y)
    scaler = model.pipeline_.named_steps["scaler"]
    reference_scaler = StandardScaler().fit(X)
    before = {
        name: getattr(scaler, name).copy() for name in ("mean_", "var_", "scale_")
    }
    n_seen = np.array(scaler.n_samples_seen_, copy=True)
    np.testing.assert_allclose(scaler.mean_, reference_scaler.mean_)
    np.testing.assert_allclose(scaler.scale_, reference_scaler.scale_)
    expected = _reference_expectations(
        reference_scaler.transform(X_validation), gamma=0.37
    )
    model.predict(X_validation)
    np.testing.assert_allclose(
        model.quantum_features(X_validation), expected, atol=1e-12, rtol=1e-12
    )
    assert model.pipeline_.named_steps["scaler"] is scaler
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(scaler, name), value)
    np.testing.assert_array_equal(scaler.n_samples_seen_, n_seen)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(X_validation, original_validation)


@pytest.mark.parametrize("label_debye", [0.0, 4.75])
def test_constant_labels_stay_in_debye(training_data, label_debye):
    X, _ = training_data
    model = QuantumRidgeRegressor().fit(X, np.full(len(X), label_debye))
    predictions = model.predict(X[:3] + 7.0)
    assert predictions.shape == (3,)
    np.testing.assert_allclose(predictions, label_debye, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("clip_negative", [False, True])
def test_negative_prediction_clipping_policy(training_data, monkeypatch, clip_negative):
    X, y = training_data
    model = QuantumRidgeRegressor(clip_negative=clip_negative).fit(X, y)
    raw_predictions = np.array([-1.25, 0.0, 2.75])
    original = raw_predictions.copy()
    monkeypatch.setattr(
        model.pipeline_.named_steps["ridge"],
        "predict",
        lambda features: raw_predictions,
    )
    expected = np.maximum(original, 0) if clip_negative else original
    np.testing.assert_array_equal(model.predict(X[:3]), expected)
    np.testing.assert_array_equal(raw_predictions, original)


def test_transformer_refit_rebuilds_for_new_width(training_data):
    X, _ = training_data
    transformer = QuantumFeatureTransformer(gamma=0.6).fit(X)
    previous_circuit = transformer.circuit_
    new_X = np.random.default_rng(5).normal(size=(4, 2))
    assert transformer.fit(new_X) is transformer
    assert transformer.n_features_in_ == transformer.circuit_.num_qubits == 2
    assert transformer.circuit_ is not previous_circuit
    assert len(transformer.observables_) == 2
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(), ["z_expectation_0", "z_expectation_1"]
    )
    np.testing.assert_allclose(
        transformer.transform(new_X),
        _reference_expectations(new_X, gamma=0.6),
        atol=1e-12,
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="features"):
        transformer.transform(X)


def test_regressor_refit_uses_fresh_pipeline_and_new_training_scaler(training_data):
    X, y = training_data
    model = QuantumRidgeRegressor(gamma=0.6).fit(X, y)
    previous_pipeline = model.pipeline_
    previous_mean = previous_pipeline.named_steps["scaler"].mean_.copy()
    new_X = np.random.default_rng(5).normal(size=(6, 2)) * [2.0, 0.5] + [20.0, -8.0]
    new_y = np.linspace(1.0, 3.0, len(new_X))
    assert model.fit(new_X, new_y) is model
    assert model.pipeline_ is not previous_pipeline
    for name in ("scaler", "quantum", "ridge"):
        assert (
            model.pipeline_.named_steps[name] is not previous_pipeline.named_steps[name]
        )
    assert (
        model.n_features_in_
        == model.pipeline_.named_steps["quantum"].n_features_in_
        == 2
    )
    np.testing.assert_allclose(
        model.pipeline_.named_steps["scaler"].mean_, new_X.mean(axis=0)
    )
    np.testing.assert_allclose(
        model.pipeline_.named_steps["scaler"].scale_, new_X.std(axis=0)
    )
    np.testing.assert_array_equal(
        previous_pipeline.named_steps["scaler"].mean_, previous_mean
    )
    assert previous_pipeline.named_steps["quantum"].n_features_in_ == 4
    assert model.quantum_features(new_X).shape == new_X.shape
    assert model.predict(new_X).shape == (len(new_X),)
    with pytest.raises(ValueError, match="features"):
        model.predict(X)


@pytest.mark.parametrize(
    "estimator_type", [QuantumFeatureTransformer, QuantumRidgeRegressor]
)
def test_sklearn_clone_preserves_parameters_but_not_fitted_state(
    training_data, estimator_type
):
    X, y = training_data
    parameters = {
        "reps": 3,
        "gamma": 0.4,
        "batch_size": 3,
        "n_qubits": None,
        "encoding": "zz",
        "simulation_method": "statevector",
    }
    if estimator_type is QuantumRidgeRegressor:
        parameters.update(alpha=0.2, clip_negative=False)
    estimator = estimator_type(**parameters).fit(X, y)
    cloned = clone(estimator)
    assert cloned is not estimator
    assert cloned.get_params() == parameters
    assert not hasattr(cloned, "n_features_in_")
    assert not hasattr(cloned, "pipeline_")
    assert not hasattr(cloned, "circuit_")
    cloned.fit(X, y)
    method = "transform" if estimator_type is QuantumFeatureTransformer else "predict"
    np.testing.assert_array_equal(
        getattr(cloned, method)(X[:2]), getattr(estimator, method)(X[:2])
    )


def test_regressor_defaults_and_optional_external_pca(training_data):
    X, y = training_data
    model = QuantumRidgeRegressor()
    assert model.get_params() == {
        "alpha": 1.0,
        "reps": 2,
        "gamma": 1.0,
        "batch_size": 128,
        "clip_negative": True,
        "n_qubits": None,
        "encoding": "zz",
        "simulation_method": "statevector",
    }
    pipeline = Pipeline(
        [("pca", PCA(n_components=2, svd_solver="full")), ("regressor", model)]
    )
    pipeline.fit(X, y)
    assert pipeline.n_features_in_ == 4
    assert (
        model.n_features_in_
        == model.pipeline_.named_steps["quantum"].circuit_.num_qubits
        == 2
    )
    assert list(model.pipeline_.named_steps) == ["scaler", "quantum", "ridge"]
    assert pipeline.predict(X[:3]).shape == (3,)


def test_small_grid_search_clones_and_fits_scalers_only_on_training_folds(
    training_data, monkeypatch
):
    X, y = training_data
    cv = KFold(n_splits=2)
    fold_inputs = [X[train_indices] for train_indices, _ in cv.split(X, y)]
    fit_inputs = []
    fit = StandardScaler.fit

    def record_fit(self, X, y=None, **kwargs):
        fit_inputs.append(np.array(X, copy=True))
        return fit(self, X, y=y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    estimator = QuantumRidgeRegressor(batch_size=4, clip_negative=False)
    search = GridSearchCV(
        estimator,
        {"alpha": [0.1, 1.0], "gamma": [0.3, 0.7]},
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        error_score="raise",
    ).fit(X, y)
    assert not hasattr(estimator, "pipeline_")
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert len(fit_inputs) == 4 * 2 + 1
    for fold_X in fold_inputs:
        assert sum(np.array_equal(fit_X, fold_X) for fit_X in fit_inputs[:-1]) == 4
    np.testing.assert_array_equal(fit_inputs[-1], X)
    np.testing.assert_allclose(
        search.best_estimator_.pipeline_.named_steps["scaler"].mean_, X.mean(axis=0)
    )
    assert search.predict(X[:3]).shape == (3,)
