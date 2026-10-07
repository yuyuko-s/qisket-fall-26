"""Finite-shot and noisy inference for the quantum models (M5; local simulators only).

Concepts, for the team:
- **Shots.** A quantum computer never returns an exact kernel value. The overlap circuit
  C(x, x′) = U(x′) followed by U(x)† is run S times ("shots") and every qubit is measured each
  time; the fraction of all-zero outcomes estimates k(x, x′) = |⟨ψ(x)|ψ(x′)⟩|². With an ideal
  device it is Binomial(S, k)/S, with standard error √(k(1−k)/S): small kernel values, the
  typical ones for many qubits, are hard to resolve with few shots.
- **Expectation values** (the projected kernel's Bloch vectors, the teammate's ⟨Z⟩ features)
  are estimated the same way: measuring Pauli P gives ±1, so ⟨P⟩ ≈ 2·Binomial(S, (1+⟨P⟩)/2)/S − 1.
  X and Y are measured by rotating each qubit into the Z basis first (H, or S†·H).
- **Hardware noise.** Real gates are imperfect and qubits lose coherence; readout flips bits.
  `AerSimulator` with a noise model taken from a fake IBM backend (a snapshot of a real
  device's calibrated error rates; runs locally) simulates this. The circuit must first be
  **transpiled**: rewritten into the device's native gates (rz, sx, x, cz on Heron) and mapped
  onto physically connected qubits, which adds gates.

The fast path (`shot_kernel`, `shot_expectations`) draws exactly the distribution an ideal
device produces, from the exact values; `sample_overlaps` / `sample_bloch` run real circuits
on Aer, ideal or noisy, to validate it and to measure the effect of noise (PLAN §7.5–7.6).
Nothing here contacts IBM hardware.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# --- Fast path: exact values -> what an ideal device with S shots would report ----------------

def shot_kernel(K: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Each entry replaced by Binomial(shots, k)/shots (k clipped to [0, 1])."""
    return rng.binomial(shots, np.clip(K, 0.0, 1.0)) / shots


def shot_expectations(E: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Pauli expectations in [−1, 1] estimated from `shots` ±1 outcomes each."""
    p = np.clip((1.0 + np.asarray(E)) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(shots, p) / shots - 1.0


def repair_kernel(K: np.ndarray) -> np.ndarray:
    """Make a shot-estimated square kernel usable for training (PLAN §7.5): symmetrize, set the
    diagonal to 1 (k(x, x) = 1 exactly), and clip negative eigenvalues to 0."""
    K = (K + K.T) / 2
    np.fill_diagonal(K, 1.0)
    lam, V = np.linalg.eigh(K)
    R = (V * np.clip(lam, 0.0, None)) @ V.T
    return (R + R.T) / 2


# --- Circuits ----------------------------------------------------------------------------------

def overlap_circuit(encoder: QuantumCircuit) -> QuantumCircuit:
    """C(a, b) = U(b) then U(a)†, measured on every qubit: P(all zeros) = |⟨ψ(a)|ψ(b)⟩|².

    Parameters: vectors "a" and "b" of the encoder's width (the angles γ·x of two molecules).
    """
    k = encoder.num_parameters
    a, b = ParameterVector("a", k), ParameterVector("b", k)
    c = encoder.assign_parameters(b).compose(encoder.assign_parameters(a).inverse())
    c.measure_all()
    return c


def basis_circuits(encoder: QuantumCircuit) -> dict[str, QuantumCircuit]:
    """U(x) followed by a measurement of every qubit in the X, Y or Z basis."""
    out = {}
    for basis in "XYZ":
        c = encoder.copy()
        for q in range(c.num_qubits):
            if basis == "X":
                c.h(q)
            elif basis == "Y":
                c.sdg(q)
                c.h(q)
        c.measure_all()
        out[basis] = c
    return out


def _values(circuit: QuantumCircuit, **vectors: np.ndarray) -> np.ndarray:
    """Parameter rows in `circuit.parameters` order from named vectors (rows × width each)."""
    cols = [vectors[p.vector.name][:, p.index] for p in circuit.parameters]
    return np.column_stack(cols)


def resolve_backend(backend):
    """A fake backend instance from its class name (e.g. "FakeFez"), or the object itself.
    Names are cheap to send to worker processes; the fake backends run locally."""
    if isinstance(backend, str):
        from qiskit_ibm_runtime import fake_provider

        return getattr(fake_provider, backend)()
    return backend


#: Noisy runs up to this many qubits evolve a density matrix (4ⁿ entries) once and then sample all
#: shots from it: ~4x faster than Aer's default of one noisy statevector trajectory per shot.
DENSITY_MATRIX_MAX_QUBITS = 12


def _sampler(backend=None, seed: int = 0, threads: int = 0, n_qubits: int = 0):
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.primitives import SamplerV2

    options = {"backend_options": {"max_parallel_threads": threads}}
    if backend is not None:
        options["backend_options"]["noise_model"] = NoiseModel.from_backend(backend)
        if n_qubits <= DENSITY_MATRIX_MAX_QUBITS:
            options["backend_options"]["method"] = "density_matrix"
    return SamplerV2(seed=seed, options=options)


def isa_circuit(circuit: QuantumCircuit, backend, optimization_level: int = 2, seed: int = 0) -> QuantumCircuit:
    """Transpile for `backend` (native gates and connectivity), as hardware would require."""
    from qiskit.transpiler import generate_preset_pass_manager

    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=resolve_backend(backend),
                                      seed_transpiler=seed)
    return pm.run(circuit)


def _overlap_chunk(encoder, A, B, shots, backend, seed, threads):
    backend = resolve_backend(backend)
    circuit = overlap_circuit(encoder)
    if backend is not None:
        circuit = isa_circuit(circuit, backend, seed=0)
    sampler = _sampler(backend, seed, threads, encoder.num_qubits)
    zero = "0" * encoder.num_qubits
    data = sampler.run([(circuit, _values(circuit, a=A, b=B))], shots=shots).result()[0].data.meas
    return np.array([data.get_counts(i).get(zero, 0) / shots for i in range(len(A))])


def sample_overlaps(encoder: QuantumCircuit, A: np.ndarray, B: np.ndarray, shots: int, backend=None,
                    seed: int = 0, n_jobs: int = 1, chunk: int = 250) -> np.ndarray:
    """Shot estimates of |⟨ψ(a_i)|ψ(b_i)⟩|² for paired rows of angle matrices A and B, from the
    overlap circuit on AerSimulator: ideal if `backend` is None, else with the noise model of
    `backend` (an object or a fake-backend class name) after transpiling for it. With n_jobs > 1,
    chunks of pairs run in separate processes, single-threaded each. Returns an array of len(A)."""
    from joblib import Parallel, delayed

    starts = range(0, len(A), chunk)
    threads = 1 if n_jobs != 1 else 0
    parts = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_overlap_chunk)(encoder, A[s:s + chunk], B[s:s + chunk], shots, backend, seed + i, threads)
        for i, s in enumerate(starts))
    return np.concatenate(parts)


def _basis_chunk(encoder, A, basis, shots, backend, seed, threads):
    backend = resolve_backend(backend)
    circuit = basis_circuits(encoder)[basis]
    if backend is not None:
        circuit = isa_circuit(circuit, backend, seed=0)
    bits = _sampler(backend, seed, threads, encoder.num_qubits).run([(circuit, _values(circuit, x=A))], shots=shots).result()[0].data.meas
    return 1.0 - 2.0 * bits.to_bool_array(order="little").mean(axis=1)  # (rows, n): qubit q is column q


def sample_bloch(encoder: QuantumCircuit, A: np.ndarray, shots: int, backend=None, seed: int = 0,
                 bases: str = "XYZ", n_jobs: int = 1, chunk: int = 25) -> np.ndarray:
    """Shot estimates of every qubit's ⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩ (blocks in the order of `bases`; with
    "XYZ" the layout of `quantum_kernel.bloch_vectors`), one measured circuit per row and basis.
    With n_jobs > 1, chunks of rows run in separate processes, single-threaded each."""
    from joblib import Parallel, delayed

    threads = 1 if n_jobs != 1 else 0
    jobs = [(basis, s) for basis in bases for s in range(0, len(A), chunk)]
    parts = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_basis_chunk)(encoder, A[s:s + chunk], basis, shots, backend, seed + 1000 * i + s, threads)
        for i, (basis, s) in enumerate(jobs))
    blocks = {b: np.vstack([p for (bb, _), p in zip(jobs, parts) if bb == b]) for b in bases}
    return np.hstack([blocks[b] for b in bases])


def depolarizing_mitigation(K_noisy: np.ndarray, survival: np.ndarray, n_qubits: int) -> np.ndarray:
    """Undo a global depolarizing channel row by row.

    If noise turns each prepared state ρ into λρ + (1 − λ)I/2ⁿ, a measured overlap becomes
    k̂ = λk + (1 − λ)/2ⁿ. The overlap of a molecule with itself (k = 1 exactly) measures
    λ = (p − 2⁻ⁿ)/(1 − 2⁻ⁿ) from its survival probability p, at the cost of one extra circuit per
    molecule; each row i is then mapped back with that molecule's λ_i, and clipped to [0, 1].
    An approximation: real noise is not exactly depolarizing.
    """
    floor = 2.0 ** -n_qubits
    lam = np.clip((np.asarray(survival) - floor) / (1 - floor), 1e-3, 1.0)
    return np.clip((K_noisy - (1 - lam[:, None]) * floor) / lam[:, None], 0.0, 1.0)
