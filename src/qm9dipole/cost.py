"""Quantum resource accounting: what each quantum model would cost on IBM hardware.

For a circuit: qubits, depth and gate counts before transpiling (the logical circuit) and
after transpiling for a fake IBM Heron backend (native gates rz, sx, x and the two-qubit cz,
on the device's heavy-hex connectivity), plus the transpiled circuit's duration from the
backend's calibrated gate times. For a model: how many distinct circuits training and
prediction need, and a rough QPU-time estimate.

Circuit counts (N training molecules, one new molecule):
- fidelity kernel: one overlap circuit per pair: N(N−1)/2 to train (k(x, x) = 1 needs no
  circuit), N per new molecule;
- projected kernel: three circuits per molecule (X, Y, Z bases): 3N to train, 3 per molecule;
- quantum-feature ridge (⟨Z⟩ readout): one circuit per molecule: N to train, 1 per molecule.

QPU time ≈ circuits × shots × (circuit duration + measurement + repetition delay). The delay
between shots (~250 µs by default on IBM devices) dominates for these short circuits. It
ignores job and compilation overheads, so treat it as a lower bound.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

TWO_QUBIT = ("cx", "cz", "ecr", "rzz", "cp", "swap", "iswap")
#: Repetition delay between shots (s); IBM's usual default. A rough constant, stated as such.
REP_DELAY_S = 250e-6
#: Readout + reset per shot (s), a rough figure for Heron devices.
MEASURE_S = 2e-6


def gate_counts(circuit: QuantumCircuit) -> dict:
    """Qubits, depth, two-qubit and single-qubit gate counts (barriers and measurements excluded)."""
    ops = circuit.count_ops()
    two = sum(v for k, v in ops.items() if k in TWO_QUBIT)
    one = sum(v for k, v in ops.items() if k not in TWO_QUBIT and k not in ("barrier", "measure", "reset", "delay"))
    active = {circuit.find_bit(q).index for ins in circuit.data if ins.operation.name != "barrier" for q in ins.qubits}
    return {"qubits": len(active), "depth": circuit.depth(lambda ins: ins.operation.name not in ("barrier",)),
            "two_qubit_gates": two, "one_qubit_gates": one,
            "two_qubit_depth": circuit.depth(lambda ins: ins.operation.num_qubits == 2 and ins.operation.name != "barrier")}


def transpiled_costs(circuit: QuantumCircuit, backend, optimization_level: int = 2, seed: int = 0) -> dict:
    """`gate_counts` of the circuit transpiled for `backend` (an object or a fake-backend class name),
    plus its duration (s)."""
    from qm9dipole.noise import isa_circuit, resolve_backend

    backend = resolve_backend(backend)
    isa = isa_circuit(circuit, backend, optimization_level=optimization_level, seed=seed)
    out = gate_counts(isa)
    try:
        out["duration_s"] = float(isa.estimate_duration(backend.target, unit="s"))
    except Exception:  # durations missing for some gate
        out["duration_s"] = float("nan")
    return out


def circuits_needed(model: str, n_train: int, n_new: int) -> tuple[int, int]:
    """(circuits to train, circuits to predict n_new molecules) for a quantum model."""
    match model:
        case "fidelity":
            return n_train * (n_train - 1) // 2, n_train * n_new
        case "projected":
            return 3 * n_train, 3 * n_new
        case "z_readout":
            return n_train, n_new
    raise KeyError(model)


def qpu_seconds(n_circuits: int, shots: int, duration_s: float) -> float:
    """Rough QPU time: circuits × shots × (duration + measurement + repetition delay)."""
    return n_circuits * shots * (duration_s + MEASURE_S + REP_DELAY_S)
