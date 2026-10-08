"""Running the quantum models on an IBM quantum computer (optional). Driver: `scripts/hardware_run.py`.

Nothing here submits a job by itself: `submit` is called only by the driver with `--submit`, after the
plan (circuits, shots, estimated QPU time) has been shown and confirmed (the team approves every hardware job in advance). The IBM
account is read from the local Qiskit config by name (`QiskitRuntimeService(name=...)`); no token is
ever written anywhere.

Concepts, for the team:
- **Only the circuits run on the device.** Input scaling, PLS, the kernel-ridge or ridge readout and
  the mapping to debye run on this computer, exactly as in simulation.
- **Which models fit a QPU budget.** The fidelity kernel needs one circuit per *pair* of molecules
  (N(N−1)/2 to train, N per new molecule): ~5,000 circuits even at N = 100. The projected kernel needs
  3 circuits per *molecule* (X, Y and Z bases) and the team's ⟨Z⟩ ridge 1. So the hardware run measures
  those two end to end (every training and test molecule on the device, readout refit on the device's
  features), plus a small block of fidelity-kernel entries, compared with their exact values.
- **ISA circuits and PUBs.** A device runs only its native gates on physically connected qubits. Each
  circuit template is transpiled once for the device with its parameters left free; one PUB (primitive
  unified bloc) is that circuit plus a table of parameter values, one row per molecule (or pair).
- **One job, capped.** All PUBs go to `SamplerV2` as a single job with `max_execution_time` set, so the
  device can never spend more than the approved QPU time on it.
- **Local testing.** With a fake backend (`FakeQuebec`: a calibration snapshot of ibm_quebec, the
  device of the PINQ² hackathon account) the same circuits run on Aer on this computer with that
  device's noise model; with no backend, on an ideal simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qm9dipole.evaluate import scores
from qm9dipole.final import feature_model

PARTS = ("qridge",)  # active presentation; legacy parts remain for archived runs
KINDS = {"pqk": "projected", "qridge": "z_readout"}
#: The fake backend that stands in for a real device when planning without connecting to IBM.
FAKE_FOR = {"ibm_quebec": "FakeQuebec", "ibm_fez": "FakeFez", "ibm_marrakesh": "FakeMarrakesh"}


@dataclass
class Block:
    """One PUB: an (ISA) circuit and its parameter rows."""
    part: str                    # pqk | qridge | kernel
    basis: str                   # X, Y, Z (features) or "overlap" (kernel entries)
    circuit: object              # QuantumCircuit, transpiled for the backend (or logical if none)
    values: np.ndarray           # (rows, parameters)
    meta: dict = field(default_factory=dict)  # row layout: n_train, n_test (features) or pairs (kernel)

    @property
    def rows(self) -> int:
        return len(self.values)


def select_test_molecules(subsets: dict[str, np.ndarray], per: int) -> tuple[np.ndarray, np.ndarray]:
    """`per` molecules from each subset, drawn as in `scripts/final_eval.py`'s noise section (so a
    hardware run and test run 1's noisy simulation use the same molecules). Returns (ids, subset name
    per id)."""
    ids = [np.sort(np.random.default_rng([2027, i]).choice(v, per, replace=False)) for i, v in enumerate(subsets.values())]
    return np.concatenate(ids), np.concatenate([[k] * per for k in subsets])


def settings_from_run(quantum: pd.DataFrame, seed: int, n: int, inputs: str = "pls10") -> dict[str, dict]:
    """The settings each quantum model chose by CV in a test run's quantum section (seed, N)."""
    import json

    out = {}
    for model in ("qkrr", "pqk", "qridge"):
        row = quantum[(quantum["model"] == model) & (quantum["inputs"] == inputs) & (quantum["seed"] == seed)
                      & (quantum["n_train"] == n)]
        if len(row):
            out[model] = json.loads(row["details"].iloc[0])["params"]
    return out


def fit_models(frame: pd.DataFrame, y: pd.Series, ids: np.ndarray, settings: dict[str, dict], seed: int,
               scaling: str, target: str, reduction: tuple[str, int], cv_max_n: int = 1000) -> dict:
    """Refit qkrr (ZZ fidelity kernel), pqk and qridge on training IDs at fixed `settings` (no search),
    exactly as `scripts/final_eval.py` refits them for its shots and noise sections."""
    from qm9dipole.models.fitters import TabularFitter
    from qm9dipole.models.quantum_readout import quantum_ridge_build

    out = {}
    for model, p in settings.items():
        if model == "qridge":
            est, _ = quantum_ridge_build(seed, scaling, target, reduction, frame.shape[1])
            f = TabularFitter(frame, est, {k: [v] for k, v in p.items()}, seed, cv_max_n=cv_max_n)
        else:
            from qm9dipole.archive.quantum_kernel import FidelityKernel, KernelRidgeFitter, ProjectedKernel

            kern = (ProjectedKernel("zz", 2, gammas=[p["gamma"]], gammas_p=[p["gamma_p"]]) if model == "pqk"
                    else FidelityKernel("zz", 2, gammas=[p["gamma"]]))
            f = KernelRidgeFitter(frame, kern, seed, scaling=scaling, target=target, reduction=reduction,
                                  alphas=(p["alpha"],), cv_max_n=cv_max_n)
        out[model] = f.fit(ids, y.loc[ids])
    return out


def build_blocks(models: dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, parts: list[str],
                 backend=None, kernel_block: tuple[int, int] = (10, 5)) -> list[Block]:
    """The PUBs of a hardware run. Features (pqk: X, Y, Z; qridge: Z): one row per training molecule,
    then one per test molecule. Kernel ("kernel", from qkrr): the first kernel_block[0] test molecules
    against the first kernel_block[1] training molecules, then each of those test molecules with itself
    (its survival probability, for `noise.depolarizing_mitigation`). Circuits are transpiled for
    `backend` (an object or a fake-backend name) once each; None keeps the logical circuits."""
    from qm9dipole.noise import _values, basis_circuits, isa_circuit, overlap_circuit

    def prepare(c):
        return isa_circuit(c, backend, seed=0) if backend is not None else c

    blocks = []
    for part in parts:
        if part in KINDS:
            fm = feature_model(models[part], KINDS[part], X_train, y_train)
            A = np.vstack([fm.angles(X_train), fm.angles(X_test)])
            for basis in fm.bases:
                c = prepare(basis_circuits(fm.encoder)[basis])
                blocks.append(Block(part, basis, c, _values(c, x=A), {"n_train": len(X_train), "n_test": len(X_test),
                                                                      "logical_qubits": fm.encoder.num_qubits}))
        elif part == "kernel":
            f = models["qkrr"]
            g = f.embed_setting_["gamma"]
            kt, ktr = min(kernel_block[0], len(X_test)), min(kernel_block[1], len(X_train))
            At, Atr = g * f.transform_inputs(X_test[:kt]), g * f.transform_inputs(X_train[:ktr])
            a = np.vstack([np.repeat(At, ktr, axis=0), At])
            b = np.vstack([np.tile(Atr, (kt, 1)), At])
            enc = f.kernel.circuit(f.n_inputs_)
            c = prepare(overlap_circuit(enc))
            blocks.append(Block(part, "overlap", c, _values(c, a=a, b=b), {"n_test": kt, "n_train": ktr,
                                                                           "logical_qubits": enc.num_qubits}))
        else:
            raise KeyError(part)
    return blocks


def plan(blocks: list[Block], shots: int, backend=None) -> pd.DataFrame:
    """One row per PUB: circuits, transpiled size and duration, and a rough QPU time (`cost.qpu_seconds`:
    circuits × shots × (duration + readout + repetition delay); job overheads not included)."""
    from qm9dipole.cost import gate_counts, qpu_seconds
    from qm9dipole.noise import resolve_backend

    be = resolve_backend(backend) if backend is not None else None
    rows = []
    for b in blocks:
        g = gate_counts(b.circuit)
        try:
            dur = float(b.circuit.estimate_duration(be.target, unit="s")) if be is not None else float("nan")
        except Exception:
            dur = float("nan")
        rows.append({"part": b.part, "basis": b.basis, "circuits": b.rows, "qubits": g["qubits"], "depth": g["depth"],
                     "two_qubit_gates": g["two_qubit_gates"], "duration_us": dur * 1e6, "shots": shots,
                     "qpu_seconds": qpu_seconds(b.rows, shots, 0.0 if np.isnan(dur) else dur)})
    return pd.DataFrame(rows)


def run_local(blocks: list[Block], shots: int, backend=None, seed: int = 0):
    """Run the PUBs on Aer on this computer: ideal (backend None) or with a fake backend's noise model."""
    from qm9dipole.noise import _sampler, resolve_backend

    be = resolve_backend(backend) if backend is not None else None
    # Transpiled circuits span the whole device; Aer drops the idle qubits, so the logical width decides
    # whether a noisy run can evolve a density matrix (`noise.DENSITY_MATRIX_MAX_QUBITS`).
    sampler = _sampler(be, seed, 0, max(b.meta["logical_qubits"] for b in blocks))
    return sampler.run([(b.circuit, b.values) for b in blocks], shots=shots).result()


def submit(blocks: list[Block], shots: int, backend, max_seconds: int):
    """Submit every PUB as one SamplerV2 job on a real device (`backend` from QiskitRuntimeService),
    capped at `max_seconds` of QPU time. Returns the job (its ID is `job.job_id()`)."""
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    sampler.options.max_execution_time = int(max_seconds)
    return sampler.run([(b.circuit, b.values) for b in blocks], shots=shots)


def parse(result, blocks: list[Block]) -> dict[str, np.ndarray]:
    """Measured features from a SamplerV2 result, in the layouts the models use: pqk → (rows, 3n) Bloch
    coordinates (X block, Y block, Z block, as `quantum_kernel.bloch_vectors`); qridge → (rows, n) ⟨Z⟩;
    kernel → the all-zeros frequency of each overlap circuit. Measured bit q is logical qubit q (the
    `measure_all` register survives transpilation in logical order)."""
    feats: dict[str, list] = {}
    for b, pub in zip(blocks, result):
        bits = pub.data.meas.to_bool_array(order="little")  # (rows, shots, n)
        if b.basis == "overlap":
            feats.setdefault(b.part, []).append((~bits.any(axis=-1)).mean(axis=1))
        else:
            feats.setdefault(b.part, []).append(1.0 - 2.0 * bits.mean(axis=1))
    return {p: np.hstack(v) if p != "kernel" else v[0] for p, v in feats.items()}


def score(models: dict, measured: dict[str, np.ndarray], X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray, y_test: np.ndarray, subset: np.ndarray, shots: int, device: str,
          kernel_block: tuple[int, int] = (10, 5), seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores (debye) of the feature models with exact, shots-only (ideal device, same shots) and
    measured features, overall and per test subset; and the kernel block's measured entries against
    exact values (raw and with the depolarizing correction)."""
    from qm9dipole.noise import depolarizing_mitigation, shot_expectations

    rng = np.random.default_rng(seed + 7)
    rows = []
    n_tr = len(X_train)
    for part in (p for p in KINDS if p in measured):
        fm = feature_model(models[part], KINDS[part], X_train, y_train)
        exact = (fm.exact_train, fm.exact(X_test))
        F = measured[part]
        variants = {"exact": exact, f"ideal device, {shots:,} shots": tuple(shot_expectations(e, shots, rng) for e in exact),
                    device: (F[:n_tr], F[n_tr:])}
        for mode, (Ftr, Fte) in variants.items():
            pred = fm.predict(Ftr, Fte)
            for name, mask in (("both", np.ones(len(y_test), bool)), *((s, subset == s) for s in np.unique(subset))):
                rows.append({"model": part, "mode": mode, "test_subset": name, "n_test": int(mask.sum()),
                             **scores(y_test[mask], pred[mask])})
    kernel = pd.DataFrame()
    if "kernel" in measured:
        f = models["qkrr"]
        kt, ktr = min(kernel_block[0], len(X_test)), min(kernel_block[1], len(X_train))
        exact = f.gram_to_train(X_test[:kt])[:, :ktr]
        k = measured["kernel"]
        K, survival = k[:kt * ktr].reshape(kt, ktr), k[kt * ktr:]
        mitigated = depolarizing_mitigation(K, survival, f.kernel.qubits(f.n_inputs_))
        kernel = pd.DataFrame({"test_row": np.repeat(np.arange(kt), ktr), "train_row": np.tile(np.arange(ktr), kt),
                               "exact": exact.ravel(), "measured": K.ravel(), "mitigated": mitigated.ravel(),
                               "survival": np.repeat(survival, ktr)})
    return pd.DataFrame(rows), kernel
