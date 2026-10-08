"""Latent-charge dipole model (exploration X2, Track A; PLAN §11 "physics-informed charge
model"). numpy only, with hand-written gradients (checked against finite differences in
tests/test_charge_model.py).

The model builds the dipole **vector** the way physics does, then takes its length:

    μ̂ = Σᵢ qᵢ cᵢ  +  Σ_(i→j bonded) sᵢⱼ ûᵢⱼ,        |μ̂| in debye,

- qᵢ = f(xᵢ): a small multilayer perceptron of atom i's environment features
  (`atoms.ATOM_FEATURE_NAMES`), plus a learned bias per element. Optional message passing
  (`message_passing` rounds) lets bonded atoms exchange learned features first, so a charge
  can depend on the molecule beyond the fixed features' reach: after the first layer,
  hᵢ ← hᵢ + SiLU(W hᵢ + M Σ_(j bonded to i) hⱼ + b), a residual graph-network update;
- cᵢ = rᵢ − r̄: coordinates centred on the molecule's unweighted centroid;
- sᵢⱼ (optional, `atomic_dipoles`): a scalar read from atom i's last hidden layer for the
  element of its bonded neighbour j, times the bond's unit vector ûᵢⱼ. This lets an atom
  carry a dipole of its own (lone pairs, polarization), which point charges cannot.

Symmetry, by construction: rotations rotate every cᵢ and ûᵢⱼ but leave qᵢ and sᵢⱼ unchanged
(they come from invariant features), so μ̂ rotates and |μ̂| is unchanged; relabeling permutes
the terms of the sums. Neutrality comes free: Σᵢ cᵢ = 0, so adding a constant to every charge
changes nothing, which makes the model identical to one whose charges are forced to sum to
zero, and its output independent of the origin (translation).

It is trained on |μ| alone. QM9 provides no dipole direction, and a global sign flip of all
charges gives the same |μ̂|, so the learned charges are defined up to that sign. Ensembles
therefore average magnitudes, never vectors.

Units: features standardized on the training atoms (docs/EVALUATION_RULES.md rule 3); charges come out in e
when positions are in Å, since |μ̂| = 4.803 D/(e·Å) × ‖μ̂‖.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit

from qm9dipole.atoms import ELEMENTS, E_ANGSTROM_TO_DEBYE, AtomData, _ranges

_N_ELEMENTS = len(ELEMENTS)
_EPS = 1e-12  # (e·Å)²: keeps ‖μ̂‖ differentiable at zero (adds ≤ 5e-6 D)


@dataclass
class _Batch:
    X: np.ndarray  # standardized features of the batch atoms
    pos: np.ndarray
    element: np.ndarray
    mol: np.ndarray  # batch molecule of each atom
    n_mol: int
    ei: np.ndarray  # edges as batch atom rows
    ej: np.ndarray
    u: np.ndarray
    A: object = None  # sparse bonded adjacency (batch atoms), for message passing


@dataclass
class TrainingLog:
    epochs: int = 0
    steps: int = 0
    best_epoch: int = 0
    best_val_mae: float = np.inf
    seconds: float = 0.0
    history: list[tuple[int, float, float, float]] = field(default_factory=list)  # epoch, lr, train loss, val MAE


class LatentChargeModel:
    """|μ| from per-atom charges (and optional bond-directed atomic dipoles).

    Parameters
    ----------
    hidden: sizes of the hidden layers (SiLU). () gives a linear charge model, qᵢ = w·xᵢ + b.
    message_passing: residual message-passing rounds over the bond graph, applied after the
        first hidden layer (needs at least one hidden layer).
    atomic_dipoles: add the Σ sᵢⱼ ûᵢⱼ term.
    loss: "mae", "mse" or "huber" (on |μ| in debye).
    weight_decay: decoupled (AdamW) decay of the weight matrices per step, relative to lr.
    val_frac, max_val: inner validation split of the training molecules, used for early
        stopping and learning-rate decay only. Never a test set.
    max_epochs, min_steps, max_steps: an epoch is one pass over the training molecules; small
        training sets get at least `min_steps` updates, large ones at most `max_steps` (whole
        epochs), since one epoch of a large set is already many updates.
    patience: stop after this many epochs without a better validation MAE; lr_patience: halve
        the learning rate after this many; training also stops below min_lr.
    """

    def __init__(self, hidden: tuple[int, ...] = (128, 64), message_passing: int = 0, atomic_dipoles: bool = True,
                 loss: str = "mae",
                 lr: float = 2e-3, weight_decay: float = 1e-4, batch_size: int = 64, max_epochs: int = 300,
                 min_steps: int = 4000, max_steps: int = 120_000, patience: int = 30, lr_patience: int = 10, min_lr: float = 1e-5,
                 val_frac: float = 0.1, max_val: int = 5000, huber_delta: float = 1.0, seed: int = 0,
                 dtype=np.float32, verbose: bool = False):
        self.hidden = tuple(hidden)
        self.message_passing = message_passing
        self.atomic_dipoles = atomic_dipoles
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.patience = patience
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.val_frac = val_frac
        self.max_val = max_val
        self.huber_delta = huber_delta
        self.seed = seed
        self.dtype = dtype
        self.verbose = verbose

    def get_params(self) -> dict:
        keys = ("hidden", "message_passing", "atomic_dipoles", "loss", "lr", "weight_decay", "batch_size", "max_epochs", "min_steps",
                "max_steps", "patience", "lr_patience", "min_lr", "val_frac", "max_val", "huber_delta", "seed")
        return {k: getattr(self, k) for k in keys}

    def set_params(self, **params) -> "LatentChargeModel":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"unknown parameter {k!r}")
            setattr(self, k, tuple(v) if k == "hidden" else v)
        return self

    # --- parameters ------------------------------------------------------------------

    def _init_params(self, d: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.message_passing and not self.hidden:
            raise ValueError("message passing needs at least one hidden layer")
        p, fan_in = {}, d
        for k, h in enumerate(self.hidden):
            p[f"W{k}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, h))
            p[f"b{k}"] = np.zeros(h)
            fan_in = h
            if k == 0:  # residual message-passing rounds start near the identity
                for r in range(self.message_passing):
                    p[f"Wm{r}"] = rng.normal(0.0, np.sqrt(0.5 / h), (h, h))
                    p[f"Mm{r}"] = rng.normal(0.0, np.sqrt(0.5 / h), (h, h))
                    p[f"bm{r}"] = np.zeros(h)
        p["wq"] = rng.normal(0.0, 0.01, fan_in)
        p["bq"] = np.zeros(_N_ELEMENTS)
        if self.atomic_dipoles:
            p["V"] = rng.normal(0.0, 0.01, (fan_in, _N_ELEMENTS))
            p["c"] = np.zeros(_N_ELEMENTS)
        return {k: v.astype(self.dtype) for k, v in p.items()}

    @property
    def n_parameters(self) -> int:
        return int(sum(v.size for v in self.params_.values()))

    # --- forward and backward ----------------------------------------------------------

    def _forward(self, p: dict, b: _Batch):
        ops = []  # what the backward pass needs, layer by layer
        a = b.X
        for k in range(len(self.hidden)):
            z = a @ p[f"W{k}"] + p[f"b{k}"]
            s = expit(z)
            ops.append(("dense", k, a, z, s))
            a = z * s  # SiLU
            if k == 0:
                for r in range(self.message_passing):
                    m = b.A @ a  # sum over bonded neighbours
                    z = a @ p[f"Wm{r}"] + m @ p[f"Mm{r}"] + p[f"bm{r}"]
                    s = expit(z)
                    ops.append(("mp", r, a, m, z, s))
                    a = a + z * s
        q = a @ p["wq"] + p["bq"][b.element]
        w = q[:, None] * b.pos
        mu = np.column_stack([np.bincount(b.mol, w[:, x], minlength=b.n_mol) for x in range(3)])
        cij = None
        if self.atomic_dipoles:
            S = a @ p["V"] + p["c"]
            cij = S[b.ei, b.element[b.ej]]
            wd = cij[:, None] * b.u
            me = b.mol[b.ei]
            mu += np.column_stack([np.bincount(me, wd[:, x], minlength=b.n_mol) for x in range(3)])
        norm = np.sqrt((mu**2).sum(1) + _EPS)
        return E_ANGSTROM_TO_DEBYE * norm, (ops, a, q, cij, mu, norm)

    def _backward(self, p: dict, b: _Batch, cache, dpred: np.ndarray) -> dict[str, np.ndarray]:
        ops, a, _, _, mu, norm = cache
        dt = self.dtype
        dmu = ((E_ANGSTROM_TO_DEBYE * dpred / norm)[:, None] * mu).astype(dt)  # (n_mol, 3)
        g = {}
        dq = (dmu[b.mol] * b.pos).sum(1)
        g["wq"] = a.T @ dq
        g["bq"] = np.bincount(b.element, dq, minlength=_N_ELEMENTS).astype(dt)
        da = np.outer(dq, p["wq"])
        if self.atomic_dipoles:
            dc = (dmu[b.mol[b.ei]] * b.u).sum(1)
            dS = np.bincount(b.ei * _N_ELEMENTS + b.element[b.ej], dc,
                             minlength=len(a) * _N_ELEMENTS).reshape(len(a), _N_ELEMENTS).astype(dt)
            g["V"] = a.T @ dS
            g["c"] = dS.sum(0)
            da = da + dS @ p["V"].T
        for op in reversed(ops):
            if op[0] == "dense":
                _, k, a_in, z, s = op
                dz = da * (s * (1.0 + z * (1.0 - s)))  # derivative of SiLU
                g[f"W{k}"] = a_in.T @ dz
                g[f"b{k}"] = dz.sum(0)
                if k:
                    da = dz @ p[f"W{k}"].T
            else:
                _, r, a_in, m, z, s = op
                dz = da * (s * (1.0 + z * (1.0 - s)))
                g[f"Wm{r}"] = a_in.T @ dz
                g[f"Mm{r}"] = m.T @ dz
                g[f"bm{r}"] = dz.sum(0)
                da = da + dz @ p[f"Wm{r}"].T + b.A.T @ (dz @ p[f"Mm{r}"].T)
        return g

    def _loss_grad(self, pred: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        r = pred - y
        n = len(r)
        if self.loss == "mae":
            return float(np.abs(r).mean()), np.sign(r) / n
        if self.loss == "mse":
            return float((r**2).mean()), 2.0 * r / n
        if self.loss == "huber":
            d = self.huber_delta
            small = np.abs(r) <= d
            return (float(np.where(small, 0.5 * r**2, d * (np.abs(r) - 0.5 * d)).mean()),
                    np.clip(r, -d, d) / n)
        raise ValueError(f"unknown loss {self.loss!r}")

    # --- batches ---------------------------------------------------------------------

    def _batch(self, data: AtomData, Xs: np.ndarray, k: np.ndarray) -> _Batch:
        """Molecules k of `data` (positions), with standardized features Xs."""
        atom_idx, counts = _ranges(data.offsets, k)
        edge_idx, ecounts = _ranges(data.edge_offsets, k)
        begin = np.concatenate([[0], np.cumsum(counts)[:-1]])
        shift = np.repeat(begin - data.offsets[k], ecounts)
        e = data.edges[edge_idx]
        ei, ej = e[:, 0] + shift, e[:, 1] + shift
        A = None
        if self.message_passing:
            n = int(counts.sum())
            A = csr_matrix((np.ones(len(ei), dtype=self.dtype), (ei, ej)), shape=(n, n))
        return _Batch(X=Xs[atom_idx], pos=data.pos[atom_idx].astype(self.dtype), element=data.element[atom_idx],
                      mol=np.repeat(np.arange(len(k)), counts), n_mol=len(k), ei=ei, ej=ej,
                      u=data.edge_vectors[edge_idx].astype(self.dtype), A=A)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(self.dtype)

    def _predict_rows(self, p: dict, data: AtomData, Xs: np.ndarray, k: np.ndarray, chunk: int = 4096) -> np.ndarray:
        out = [self._forward(p, self._batch(data, Xs, k[s:s + chunk]))[0] for s in range(0, len(k), chunk)]
        return np.concatenate(out) if out else np.zeros(0)

    # --- training --------------------------------------------------------------------

    def fit(self, data: AtomData, y: np.ndarray) -> "LatentChargeModel":
        """Train on every molecule of `data`, with targets y = |μ| (debye) in molecule order."""
        start = time.perf_counter()
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        n = data.n_molecules
        n_val = int(min(self.max_val, max(10, round(self.val_frac * n)))) if n >= 20 else 0
        perm = rng.permutation(n)
        val, train = np.sort(perm[:n_val]), perm[n_val:]

        # Feature standardization on the training atoms only (rule 3).
        train_atoms, _ = _ranges(data.offsets, np.sort(train))
        self.mean_ = data.X[train_atoms].mean(axis=0)
        sd = data.X[train_atoms].std(axis=0)
        self.scale_ = np.where(sd > 1e-6, sd, 1.0)
        Xs = self._standardize(data.X)

        p = self._init_params(data.X.shape[1], rng)
        m1 = {k: np.zeros_like(v) for k, v in p.items()}
        m2 = {k: np.zeros_like(v) for k, v in p.items()}
        decay = {k for k in p if k.startswith(("W", "M")) or k in ("wq", "V")}
        b1, b2 = 0.9, 0.999
        lr = self.lr
        steps_per_epoch = max(1, int(np.ceil(len(train) / self.batch_size)))
        epochs = max(self.max_epochs, int(np.ceil(self.min_steps / steps_per_epoch))) if n_val else \
            int(np.ceil(self.min_steps / steps_per_epoch))
        epochs = max(1, min(epochs, self.max_steps // steps_per_epoch))
        log = TrainingLog()
        best_p, best_mae, since_best, since_lr = {k: v.copy() for k, v in p.items()}, np.inf, 0, 0
        t = 0
        for epoch in range(1, epochs + 1):
            order = rng.permutation(train)
            losses = []
            for s in range(0, len(order), self.batch_size):
                k = order[s:s + self.batch_size]
                b = self._batch(data, Xs, k)
                pred, cache = self._forward(p, b)
                loss, dpred = self._loss_grad(pred, y[k])
                g = self._backward(p, b, cache, dpred)
                t += 1
                for name, grad in g.items():
                    m1[name] = b1 * m1[name] + (1 - b1) * grad
                    m2[name] = b2 * m2[name] + (1 - b2) * grad * grad
                    step = lr * (m1[name] / (1 - b1**t)) / (np.sqrt(m2[name] / (1 - b2**t)) + 1e-8)
                    if name in decay:
                        p[name] *= 1 - lr * self.weight_decay
                    p[name] -= step.astype(self.dtype)
                losses.append(loss)
            log.epochs, log.steps = epoch, t
            if not n_val:
                continue
            val_mae = float(np.abs(self._predict_rows(p, data, Xs, val) - y[val]).mean())
            log.history.append((epoch, lr, float(np.mean(losses)), val_mae))
            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  epoch {epoch:4d}  lr {lr:.1e}  train loss {np.mean(losses):.4f}  val MAE {val_mae:.4f} D")
            if val_mae < best_mae - 1e-5:
                best_mae, best_p, since_best, since_lr = val_mae, {k: v.copy() for k, v in p.items()}, 0, 0
                log.best_epoch, log.best_val_mae = epoch, val_mae
            else:
                since_best += 1
                since_lr += 1
                if since_lr >= self.lr_patience:
                    lr, since_lr = lr * 0.5, 0
                if since_best >= self.patience or lr < self.min_lr:
                    break
        self.params_ = best_p if n_val else p
        log.seconds = time.perf_counter() - start
        self.log_ = log
        return self

    # --- prediction ------------------------------------------------------------------

    def predict(self, data: AtomData) -> np.ndarray:
        """|μ| in debye for every molecule of `data`, in molecule order."""
        return self._predict_rows(self.params_, data, self._standardize(data.X), np.arange(data.n_molecules))

    def dipole_vectors(self, data: AtomData) -> np.ndarray:
        """Predicted dipole vectors (debye), in each molecule's stored frame (up to the global
        sign convention the model happened to learn)."""
        Xs = self._standardize(data.X)
        out = []
        for s in range(0, data.n_molecules, 4096):
            k = np.arange(s, min(s + 4096, data.n_molecules))
            _, cache = self._forward(self.params_, self._batch(data, Xs, k))
            out.append(cache[4] * E_ANGSTROM_TO_DEBYE)
        return np.concatenate(out)

    def charges(self, data: AtomData) -> np.ndarray:
        """Per-atom charges (e), shifted to sum to zero per molecule (the shift leaves μ̂
        unchanged), in atom order."""
        Xs = self._standardize(data.X)
        k = np.arange(data.n_molecules)
        b = self._batch(data, Xs, k)
        _, cache = self._forward(self.params_, b)
        q = cache[2].astype(np.float64)
        mean = np.bincount(b.mol, q, minlength=b.n_mol) / np.bincount(b.mol, minlength=b.n_mol)
        return q - mean[b.mol]


class ChargeEnsemble:
    """Average of `n_models` LatentChargeModels with different seeds (magnitudes, not vectors:
    each member's charges carry an arbitrary global sign)."""

    def __init__(self, n_models: int = 3, seed: int = 0, **params):
        self.n_models = n_models
        self.seed = seed
        self.params = params

    def fit(self, data: AtomData, y: np.ndarray) -> "ChargeEnsemble":
        self.members_ = [LatentChargeModel(seed=self.seed * 1000 + k, **self.params).fit(data, y)
                         for k in range(self.n_models)]
        return self

    def predict(self, data: AtomData) -> np.ndarray:
        return np.mean([m.predict(data) for m in self.members_], axis=0)
