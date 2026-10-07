"""Exact statevectors of one parameterized Qiskit circuit for many inputs at once (simulator only).

Quantum kernel and projected-feature models need one statevector per molecule, for thousands
of molecules and several angle scales. `qiskit.quantum_info.Statevector` evolves one bound
circuit at a time (2–6 ms per molecule for the 10-qubit encoders of `models/quantum.py`,
mostly Python overhead per gate). `BatchedCircuit` reads a Qiskit circuit once and then
evolves a whole batch of parameter rows together: every gate acts on a (batch, 2ⁿ) complex
array, with its angle evaluated per row. The circuit is still the Qiskit object the team
builds, draws and transpiles; only the arithmetic is batched. Statevectors equal
`Statevector(circuit.assign_parameters(row))` to ~1e-13, global phase included
(tests/test_qsim.py).

Conventions follow Qiskit: qubit q is bit q of the basis-state index (little-endian), and
parameter column j is `circuit.parameters[j]`, the order `assign_parameters` uses.

Supported gates: id, h, x, y, z, s, sdg, t, tdg, sx, rx, ry, rz, p, u, cx, cz, cp, rzz, swap
(barriers are skipped); anything else raises NotImplementedError. A gate angle may be any
polynomial of degree ≤ 2 in the parameters: Qiskit's ZZ feature map uses 2·(π − x_i)(π − x_j).

Three standard simulator shortcuts, none of which changes the result:
- consecutive single-qubit gates on a qubit are multiplied into one 2×2 matrix per row;
- until the first two-qubit gate, the state is kept as a product of single-qubit states;
- diagonal (phase) gates are applied in blocks: exp(i·Σ_g θ_g(x)·pattern_g) as one
  (batch × gates) @ (gates × 2ⁿ) product. The pattern CX(c, t)·P(θ)_t·CX(c, t), which the ZZ
  feature map uses for every pair, is the phase θ·(b_c ⊕ b_t) and counts as one phase gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression

#: Amplitudes per chunk when evolving a batch (complex128: 2**23 amplitudes = 128 MB).
CHUNK_AMPLITUDES = 2**23


@dataclass
class Angle:
    """A gate angle (radians) as a polynomial of degree ≤ 2 in the circuit's parameter columns."""

    const: float = 0.0
    lin_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    lin_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    quad_a: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    quad_b: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    quad_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __call__(self, A: np.ndarray) -> np.ndarray:
        """Angle for every row of the parameter matrix A (batch × n_parameters)."""
        out = np.full(len(A), self.const, dtype=np.float64)
        if len(self.lin_cols):
            out += A[:, self.lin_cols] @ self.lin_coef
        if len(self.quad_a):
            out += (A[:, self.quad_a] * A[:, self.quad_b]) @ self.quad_coef
        return out


def _number(expr) -> float:
    value = complex(expr.numeric() if isinstance(expr, ParameterExpression) else expr)
    if abs(value.imag) > 1e-12:
        raise NotImplementedError(f"complex gate angle {value}")
    return value.real


def angle_polynomial(value, index: dict) -> Angle:
    """Express a gate parameter (a number or a ParameterExpression) as an `Angle`.

    The coefficients of the degree-2 polynomial are recovered by least squares from bound
    evaluations at random points and must reproduce every evaluation to 1e-9 (relative); an
    expression that is not such a polynomial (e.g. sin(x)) raises NotImplementedError.
    """
    if not isinstance(value, ParameterExpression) or not value.parameters:
        return Angle(const=_number(value))
    params = sorted(value.parameters, key=lambda p: index[p])
    m = len(params)
    pairs = [(a, b) for a in range(m) for b in range(a, m)]
    n_terms = 1 + m + len(pairs)
    points = np.random.default_rng(20261007).uniform(-2.0, 2.0, size=(n_terms + 6, m))
    vals = np.array([_number(value.bind(dict(zip(params, map(float, row))))) for row in points])
    design = np.column_stack([np.ones(len(points)), points,
                              *[points[:, a] * points[:, b] for a, b in pairs]])
    coef = np.linalg.lstsq(design, vals, rcond=None)[0]
    scale = 1.0 + np.abs(vals).max()
    if np.abs(design @ coef - vals).max() > 1e-9 * scale:
        raise NotImplementedError(f"gate angle {value} is not a polynomial of degree <= 2")
    coef[np.abs(coef) < 1e-12 * scale] = 0.0
    cols = np.array([index[p] for p in params])
    lin = np.flatnonzero(coef[1:1 + m])
    quad = np.flatnonzero(coef[1 + m:])
    return Angle(const=float(coef[0]), lin_cols=cols[lin], lin_coef=coef[1 + lin],
                 quad_a=cols[[pairs[k][0] for k in quad]].astype(int),
                 quad_b=cols[[pairs[k][1] for k in quad]].astype(int), quad_coef=coef[1 + m + quad])


# Single-qubit gates as batched 2x2 matrices of their angles (each angle an array of shape (B,)).
_SQ2 = 1 / np.sqrt(2)
_CONST = {
    "h": np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=np.complex128),
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "sx": np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=np.complex128),
}
_FIXED_PHASE = {"z": np.pi, "s": np.pi / 2, "sdg": -np.pi / 2, "t": np.pi / 4, "tdg": -np.pi / 4}
ONE_QUBIT = {*_CONST, *_FIXED_PHASE, "rx", "ry", "rz", "p", "u", "u3"}


def _matrix(name: str, th: list[np.ndarray], B: int) -> np.ndarray:
    """(B, 2, 2) complex matrices of a single-qubit gate."""
    if name in _CONST:
        return np.broadcast_to(_CONST[name], (B, 2, 2))
    M = np.zeros((B, 2, 2), dtype=np.complex128)
    if name in _FIXED_PHASE or name == "p":
        lam = np.full(B, _FIXED_PHASE[name]) if name in _FIXED_PHASE else th[0]
        M[:, 0, 0], M[:, 1, 1] = 1.0, np.exp(1j * lam)
    elif name == "rz":
        M[:, 0, 0], M[:, 1, 1] = np.exp(-0.5j * th[0]), np.exp(0.5j * th[0])
    elif name in ("rx", "ry"):
        c, s = np.cos(th[0] / 2), np.sin(th[0] / 2)
        M[:, 0, 0], M[:, 1, 1] = c, c
        M[:, 0, 1], M[:, 1, 0] = (-1j * s, -1j * s) if name == "rx" else (-s, s)
    else:  # u / u3
        t, phi, lam = th
        c, s = np.cos(t / 2), np.sin(t / 2)
        M[:, 0, 0], M[:, 0, 1] = c, -np.exp(1j * lam) * s
        M[:, 1, 0], M[:, 1, 1] = np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c
    return M


def _apply_1q(psi: np.ndarray, n: int, q: int, M: np.ndarray) -> np.ndarray:
    """Apply batched 2x2 matrices M (B, 2, 2) to qubit q of statevectors psi (B, 2**n)."""
    B = psi.shape[0]
    v = psi.reshape(B, 2 ** (n - q - 1), 2, 2**q)
    a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
    m = M[:, :, :, None, None]
    out = np.empty_like(v)
    np.multiply(m[:, 0, 0], a0, out=out[:, :, 0, :])
    out[:, :, 0, :] += m[:, 0, 1] * a1
    np.multiply(m[:, 1, 0], a0, out=out[:, :, 1, :])
    out[:, :, 1, :] += m[:, 1, 1] * a1
    return out.reshape(B, 2**n)


class BatchedCircuit:
    """A Qiskit circuit compiled for batched exact simulation (see the module docstring).

    `statevectors(A)` returns the (len(A), 2ⁿ) statevectors for parameter rows A, starting
    from |0…0⟩.
    """

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.n_qubits = n = circuit.num_qubits
        self.parameters = list(circuit.parameters)
        index = {p: j for j, p in enumerate(self.parameters)}
        idx = np.arange(2**n)
        bits = [((idx >> q) & 1).astype(np.float64) for q in range(n)]
        qubit = {bit: i for i, bit in enumerate(circuit.qubits)}
        ops = [(ins.operation.name, [qubit[b] for b in ins.qubits], list(ins.operation.params))
               for ins in circuit.data if ins.operation.name not in ("barrier", "id")]
        # Compiled ops: ("1q", q, name, [Angle]) | ("diag", qubits, Angle, pattern) | ("perm", qubits, perm)
        self._ops: list[tuple] = []
        k = 0
        while k < len(ops):
            name, qs, params = ops[k]
            # CX(c,t) · P(θ)_t or RZ(θ)_t · CX(c,t) is a parity phase θ·(b_c ⊕ b_t) (minus θ/2 for RZ).
            if (name == "cx" and k + 2 < len(ops) and ops[k + 2][0] == "cx" and ops[k + 2][1] == qs
                    and ops[k + 1][0] in ("p", "rz") and ops[k + 1][1] == [qs[1]]):
                parity = np.abs(bits[qs[0]] - bits[qs[1]])
                pattern = parity if ops[k + 1][0] == "p" else parity - 0.5
                self._ops.append(("diag", tuple(qs), angle_polynomial(ops[k + 1][2][0], index), pattern))
                k += 3
                continue
            if name in ONE_QUBIT:
                self._ops.append(("1q", qs[0], name, [angle_polynomial(p, index) for p in params]))
            elif name == "cz":
                self._ops.append(("diag", tuple(qs), Angle(const=np.pi), bits[qs[0]] * bits[qs[1]]))
            elif name == "cp":
                self._ops.append(("diag", tuple(qs), angle_polynomial(params[0], index), bits[qs[0]] * bits[qs[1]]))
            elif name == "rzz":
                self._ops.append(("diag", tuple(qs), angle_polynomial(params[0], index),
                                  np.abs(bits[qs[0]] - bits[qs[1]]) - 0.5))
            elif name == "cx":
                c, t = qs
                self._ops.append(("perm", tuple(qs), np.where((idx >> c) & 1, idx ^ (1 << t), idx)))
            elif name == "swap":
                a, b = qs
                differ = ((idx >> a) & 1) != ((idx >> b) & 1)
                self._ops.append(("perm", tuple(qs), np.where(differ, idx ^ ((1 << a) | (1 << b)), idx)))
            else:
                raise NotImplementedError(f"gate {name!r} is not supported by BatchedCircuit")
            k += 1
        self._global_phase = angle_polynomial(circuit.global_phase, index)

    def _evolve(self, A: np.ndarray) -> np.ndarray:
        B, n = len(A), self.n_qubits
        pending: dict[int, np.ndarray] = {}  # qubit -> fused (B, 2, 2) matrix not yet applied
        product = np.zeros((n, B, 2), dtype=np.complex128)  # per-qubit states while unentangled
        product[:, :, 0] = 1.0
        state = {"psi": None}  # full statevectors once a multi-qubit gate has acted
        diag_angles: list[np.ndarray] = []
        diag_patterns: list[np.ndarray] = []
        touched: set[int] = set()  # qubits of the phase gates waiting in diag_*

        def flush_diag():
            if diag_angles:
                state["psi"] *= np.exp(1j * (np.column_stack(diag_angles) @ np.stack(diag_patterns)))
                diag_angles.clear()
                diag_patterns.clear()
                touched.clear()

        def apply_pending(qubits):
            for q in qubits:
                M = pending.pop(q, None)
                if M is None:
                    continue
                if q in touched:  # phase gates on q came first: apply them before M
                    flush_diag()
                state["psi"] = _apply_1q(state["psi"], n, q, M)

        def entangle():
            for q, M in pending.items():
                product[q] = np.einsum("bij,bj->bi", M, product[q])
            pending.clear()
            psi = product[n - 1]
            for q in range(n - 2, -1, -1):  # qubit 0 is the least significant bit
                psi = (psi[:, :, None] * product[q][:, None, :]).reshape(B, -1)
            state["psi"] = np.ascontiguousarray(psi)

        for op in self._ops:
            kind = op[0]
            if kind == "1q":
                _, q, name, fns = op
                M = _matrix(name, [f(A) for f in fns], B)
                pending[q] = M if q not in pending else np.einsum("bij,bjk->bik", M, pending[q])
                continue
            if state["psi"] is None:
                entangle()
            qubits = op[1]
            apply_pending(qubits)
            if kind == "diag":
                diag_angles.append(op[2](A))
                diag_patterns.append(op[3])
                touched.update(qubits)
            else:  # a permutation (cx, swap) does not commute with pending phases
                flush_diag()
                state["psi"] = state["psi"][:, op[2]]
        if state["psi"] is None:
            entangle()
        apply_pending(list(pending))
        flush_diag()
        psi = state["psi"]
        gp = self._global_phase(A)
        if np.any(gp != 0):
            psi *= np.exp(1j * gp)[:, None]
        return psi

    def statevectors(self, A: np.ndarray) -> np.ndarray:
        """Statevectors (len(A) × 2ⁿ, complex128) for parameter rows A (len(A) × n_parameters)."""
        A = np.atleast_2d(np.asarray(A, dtype=np.float64))
        if A.shape[1] != len(self.parameters):
            raise ValueError(f"A has {A.shape[1]} columns; the circuit has {len(self.parameters)} parameters")
        chunk = max(1, CHUNK_AMPLITUDES // 2**self.n_qubits)
        if len(A) <= chunk:
            return self._evolve(A)
        return np.concatenate([self._evolve(A[s:s + chunk]) for s in range(0, len(A), chunk)])


# --- Quantities computed from statevectors ---------------------------------------------

def fidelity_kernel(psi_a: np.ndarray, psi_b: np.ndarray | None = None) -> np.ndarray:
    """K_ij = |⟨ψ_a,i|ψ_b,j⟩|², the state-overlap (fidelity) kernel; psi_b defaults to psi_a."""
    psi_b = psi_a if psi_b is None else psi_b
    return np.abs(psi_a.conj() @ psi_b.T) ** 2


def pauli_expectations(psi: np.ndarray) -> dict[str, np.ndarray]:
    """Single-qubit expectations ⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩ (each batch × n_qubits) of statevectors psi.

    They are the Bloch vectors of each qubit's reduced state: ρ_q = (I + ⟨X⟩X + ⟨Y⟩Y + ⟨Z⟩Z)/2.
    """
    B, dim = psi.shape
    n = int(np.log2(dim))
    out = {k: np.empty((B, n)) for k in "XYZ"}
    for q in range(n):
        v = psi.reshape(B, 2 ** (n - q - 1), 2, 2**q)
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        rho01 = np.einsum("blr,blr->b", a0, a1.conj())
        out["X"][:, q] = 2 * rho01.real
        out["Y"][:, q] = -2 * rho01.imag
        out["Z"][:, q] = (np.abs(a0) ** 2).sum(axis=(1, 2)) - (np.abs(a1) ** 2).sum(axis=(1, 2))
    return out


def zz_expectations(psi: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """⟨Z_a Z_b⟩ for each qubit pair (batch × len(pairs))."""
    B, dim = psi.shape
    idx = np.arange(dim)
    signs = np.stack([(1 - 2 * ((idx >> a) & 1)) * (1 - 2 * ((idx >> b) & 1)) for a, b in pairs], axis=1)
    return (np.abs(psi) ** 2) @ signs.astype(np.float64)
