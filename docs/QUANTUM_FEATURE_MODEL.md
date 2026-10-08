# Quantum-feature regression model (`QuantumRidgeRegressor`)

The team's quantum-feature ridge model (`src/qm9dipole/models/quantum.py`; tests `tests/test_quantum.py`, `tests/test_quantum_encoding.py`). The synthetic demo notebook described at the end was archived and is not part of the submission. This page is the model's API documentation, moved here unchanged from the README; how the model performed is in the README's Results section.

`src/qm9dipole/models/quantum.py` implements a fixed Qiskit data encoder with a classical
linear ridge readout. `QuantumRidgeRegressor` fits `StandardScaler` on training inputs,
binds `gamma` times the standardized scores as circuit data, measures one local Z expectation
per qubit, and fits `Ridge` on debye labels. There are no variational gate weights.
The circuit needs fixed-width inputs; it does not make raw XYZ inputs invariant.

The legacy default remains one input column per qubit with `n_qubits=None`, `encoding="zz"`
and `simulation_method="statevector"`; 4–8 columns are a small starting example. The unchanged
readout/scaling defaults are `alpha=1.0`, `gamma=1.0`, `reps=2`, `batch_size=128` and
`clip_negative=True`. These are illustrative, not selected hyperparameters.

An explicit `n_qubits` is independent of input width. For example, 24 columns can use 8 qubits
without first requiring PCA:

```python
from qm9dipole.models.quantum import QuantumRidgeRegressor

# X_train and X_val have 24 fixed-width descriptor columns.
model = QuantumRidgeRegressor(
    n_qubits=8,
    encoding="ry_rz",
    simulation_method="matrix_product_state",
)
model.fit(X_train, y_train)          # y_train: dipole magnitude in debye
predictions = model.predict(X_val)  # debye; negative predictions clipped at 0
features = model.quantum_features(X_val)
assert features.shape == (len(X_val), 8)
encoder = model.pipeline_.named_steps["quantum"]
assert encoder.n_upload_layers_ == 2  # ceil(24 / (8 qubits * 2 angles))
```

Each encoding defines a constant ordered axis tuple with `A` input-angle slots per qubit per
upload. Capacity is `Q * A` for `Q` qubits. Zero-based input `i` maps to
`upload = i // (Q * A)`, `qubit = (i % (Q * A)) // A`, `rotation = i % A`.
Unused slots are zero-filled; `n_upload_layers_ = ceil(n_inputs / (Q * A))` counts uploads
per complete input pass, and `reps` repeats those complete passes.

- `encoding="zz"`: one phase input per qubit/upload (`A=1`), using the built-in Qiskit
  `zz_feature_map` with linear entanglement and `reps=1` blocks. The complete encoder needs
  `reps >= 2` to avoid constant Z readout, and at least two qubits.
- `encoding="ry"`: one RY input angle per qubit/upload (`A=1`).
- `encoding="ry_rz"`: ordered RY then RZ input angles (`A=2`). Both angle encoders use library
  gates, a fixed `RX(pi/4)` mixer after each upload to expose phase inputs to Z readout, and
  neighboring CZ gates between steps. They allow `reps >= 1` and at least one qubit.

`quantum_features(X)` returns dimensionless expectations with shape `(n_samples, Q)`, not
necessarily the input width. Changing the encoding/qubit count changes the model; this is not
a lossless reshaping of the inputs. The fitted ridge readout maps expectations to physical
units. This is a linear projected quantum-feature model, not the pairwise fidelity kernel.

Any tuning uses training-only CV and refits all learned transforms in each fold. Upstream PCA
is optional; if used, put it **before this model in the same sklearn `Pipeline`**, not in
globally fitted scores supplied to CV. There is no arbitrary division by 255: that scale is
appropriate only for known 8-bit pixel inputs, not generic molecular descriptors.

Both simulation options are local only: `statevector` uses `StatevectorEstimator`, while
`matrix_product_state` uses Aer exact expectations with truncation threshold zero. Neither
uses finite shots or a noise model here. Matrix product state (MPS) simulation can be efficient
for shallow nearest-neighbor structure, not arbitrary deep or highly entangled circuits.
No IBM services are connected; a hardware adapter remains separate work.

The model's tests:

```bash
pytest -q tests/test_quantum.py tests/test_quantum_encoding.py
```

The archived synthetic demo notebook ran the same 24-input arrays across all nine combinations of 8/16/24 qubits
and ZZ/RY/RY-RZ encoding with MPS. It reported a training-mean baseline, validation MAE/RMSE, dimensions, logical
circuit cost and local elapsed time without tuning or selecting a final model. It does not
access QM9 test sets or hardware, complete M4, or commit the final hardware architecture.
Nothing in this prototype demonstrates quantum advantage.
