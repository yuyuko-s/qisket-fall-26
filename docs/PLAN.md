# PLAN — QM9 dipole magnitude from fewer labels (quantum kernel regression)

How to use this document:
- Sections 1–8 are the design. Section 9 holds the experiment grid, output schema and
  milestones with acceptance criteria. Section 10 lists known gotchas.
- Values marked **default** may be changed. Log every change in `docs/DECISIONS.md`, with the
  reason, *before* looking at any test-set result.
- If anything here conflicts with `docs/BRIEF.md`, the brief wins.

---

## 1. Background

### 1.1 Dipole moment

- Electrons are not shared evenly in most molecules: O, N and F pull electron density toward
  themselves. The dipole moment measures the resulting charge separation.
- In a point-charge picture it is the vector μ = Σᵢ qᵢ rᵢ. The prompt's target is its
  magnitude |μ|, in debye (D). 1 D ≈ 0.208 e·Å.
- Reference points: water is about 1.85 D. CO₂ is 0 D, because its two bond dipoles cancel by
  symmetry.
- Geometry matters, not just composition. Ethanol and dimethyl ether are both C₂H₆O, with
  experimental dipoles of about 1.7 D and 1.3 D. This is why composition-only is the natural
  ablation.

A physically sensible |μ| predictor is invariant to:
- **rotation** — the vector rotates, its length does not;
- **translation** — for a neutral molecule, μ does not depend on the origin;
- **atom relabeling** — atom order is arbitrary bookkeeping.

### 1.2 QM9

- 133,885 molecules made of C, H, N, O and F, with up to 9 heavy atoms (29 atoms maximum
  including hydrogens).
- Geometries are DFT-relaxed and properties are computed at the B3LYP/6-31G(2df,p) level.
- These are computed reference values, not experimental measurements. Molecules come from
  GDB-17 (Ruddigkeit et al. 2012).
- Label: `mu`, in debye, taken from the property line of each file.
- No new electronic-structure calculations are needed.

### 1.3 Geometry provenance caveat (must appear in the writeup)

The inputs are DFT-optimized geometries. Getting one requires a geometry optimization: many
self-consistent-field (SCF) cycles plus force evaluations. That usually costs far more than
computing the dipole at the final geometry, which needs a single SCF.

So "fewer labels" only saves real compute if geometries come from somewhere cheap (a force
field or semi-empirical method). That is out of scope here. State it plainly as an assumption.

---

## 2. Requirements traceability

| ID | Requirement | Milestone | Artifact |
|---|---|---|---|
| R1 | Small documented QM9 subset with formula diversity; molecule IDs; documented exclusions | M1 | `splits/*.json`, `docs/DATA.md` |
| R2 | Unseen-formula test set (whole formulas held out) | M1 | `splits/test_unseen.json` |
| R3 | Familiar-formula test set: unseen molecules whose formulas are in training | M1 | `splits/test_familiar.json` |
| R4 | Familiar formulas represented at every nested size | M1 | asserts and split report |
| R5 | ≥3 nested training sizes × ≥3 seeds | M1, M3 | `splits/train_s{seed}_n{N}.json` |
| R6 | Baselines: mean, RBF, tree | M3 | results CSV |
| R7 | Team-built quantum regressor at every size and seed | M4 | results CSV |
| R8 | Matched compressed and uncompressed inputs | M3, M4 | `input_variant` column |
| R9 | MAE and RMSE in debye | M3+ | results CSV |
| R10 | Representation ablation (composition-only vs. composition + geometry) | M6 | ablation figure |
| R11 | Invariance checks: rotation, translation, relabeling | M2, M7 | tests and invariance table |
| R12 | Finite-shot and noisy inference | M5 | shots/noise figure |
| R13 | Quantum cost | M6 | cost table |
| R14 | No test-set selection; transforms fit on training only | all | protocol and `final_eval.py` |
| R15 | Bounded circuit outputs mapped to debye | M4 | code and writeup note |
| R16 | Geometry provenance and cost stated | M7 | writeup |

---

## 3. Data (`src/qm9dipole/data.py`)

### 3.1 Source

- The figshare collection "Quantum chemistry structures and properties of 134 kilo molecules"
  (Ramakrishnan et al. 2014), linked from https://quantum-machine.org/datasets/.
- Files:
  - `dsgdb9nsd.xyz.tar.bz2` — 133,885 per-molecule `.xyz` files (figshare file id 3195389).
  - `uncharacterized.txt` — the 3,054 molecules that failed the authors' geometry consistency
    check.
  - `readme.txt`.
- Download to `data/raw/` (gitignored) and record SHA-256 checksums in `docs/DATA.md`.
- Alternative: PyTorch Geometric's QM9 already drops the 3,054 molecules, and its target 0 is
  μ. It is a heavier dependency, so prefer the raw files.

### 3.2 File format (verify against `readme.txt`)

- **Line 1:** number of atoms `na`.
- **Line 2:** tab-separated properties, in this order: tag `gdb`, index, A, B, C, **mu (D)**,
  alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv.
- **Lines 3 … na+2:** element, x, y, z (Å), Mulliken charge (e). The charge column is
  **never** used as an input.
- After the atoms: a harmonic-frequencies line, a SMILES line (GDB and relaxed), and an
  InChI line.
- Some floats use a Fortran-style exponent such as `1.23*^-6`. Replace `*^` with `e` before
  calling `float()`.

### 3.3 Parsed table

Save to `data/processed/qm9.parquet` (or `.npz`) with these columns:
- `id` (int, QM9 index)
- `formula` (Hill order: C, H, then the rest alphabetically)
- `n_atoms`, `n_heavy`
- `Z` (int array), `R` (float array, Å)
- `mu` (float, D)
- `smiles` (for reference and plots only — never a feature)

### 3.4 Exclusions

Record each rule in `docs/DATA.md` with its count and reason:
- the 3,054 uncharacterized molecules (**default:** exclude);
- the readme also flags a few other molecules (for example, two that converged to saddle
  points) — decide whether to exclude them and document the choice;
- parse failures (expected: zero; log any).

### 3.5 Working subset

Build a documented, manageable, formula-diverse pool:
- **Default:** from the remaining molecules, keep at most 25 per formula, sampled randomly
  within each formula using the split seed.
- Report the number of formulas and molecules, the heavy-atom-count distribution, and a μ
  histogram.

---

## 4. Splits (`src/qm9dipole/splits.py`)

Seeds: a fixed **split seed** (**default** 2026) defines the test sets. **Training seeds**
s ∈ {0, 1, 2} define the training sets (add 3 and 4 if cheap).

| Parameter | Default |
|---|---|
| `unseen_formula_frac` | 0.15, stratified by `n_heavy` |
| unseen test | ≤5 molecules per unseen formula, ≤600 total |
| `n_familiar_formulas` | 40, chosen from non-unseen formulas with ≥3 pool molecules (must be < N1) |
| familiar test | 2 molecules per familiar formula (80 total) |
| training sizes N | [100, 300, 1000]; optional classical-only extension [3000] |

Procedure:
1. Let F be the formulas in the working pool. Pick F_unseen with a stratified random draw
   (split seed). The unseen test set is sampled from those formulas.
2. Pick the familiar formulas from F \ F_unseen. Sample 2 molecules per familiar formula into
   the familiar test set (split seed).
3. The training pool P is every pool molecule whose formula is not in F_unseen and which is
   not in the familiar test set.
4. For each training seed s:
   - **Anchors** A_s: one random molecule from P for each familiar formula.
   - **Fill order** π_s: a random permutation of P \ A_s.
   - **Training sets:** S_{s,N} = A_s ∪ π_s[: N − |A_s|] for every N. This makes them nested
     by construction.
5. Save JSON files containing the IDs plus metadata (seed, config hash, counts).

Asserts — all must pass, as unit tests and at build time:
- (a) training and test sets are disjoint;
- (b) no unseen formula appears in any training set;
- (c) every familiar formula is in S_{s,N1};
- (d) S_{s,N1} ⊂ S_{s,N2} ⊂ S_{s,N3};
- (e) sizes are exact;
- (f) IDs are unique;
- (g) no excluded ID appears anywhere.

Notes:
- At N1 = 100, anchors are 40% of the set, which skews its formula mix. Document this, and
  lower `n_familiar_formulas` if it matters.
- Cross-validation inside each training set: **default** `KFold(5, shuffle=True,
  random_state=s)`, with the same folds for every model. An optional, documented alternative
  is `GroupKFold` by formula, which tunes for unseen-formula generalization.

---

## 5. Representations (`src/qm9dipole/descriptors.py`)

1. **Composition (5-dim):** counts of [C, H, N, O, F]. Trivially invariant, but cannot tell
   isomers apart.
2. **Coulomb-matrix spectrum (29-dim)** (Rupp et al. 2012):
   - Diagonal entries M_ii = 0.5·Z_i^2.4. Off-diagonal entries M_ij = Z_i·Z_j / |R_i − R_j|.
   - Use distances in bohr (1 Å = 1.8897261 bohr).
   - Take the eigenvalues, sort by decreasing |λ|, and zero-pad to 29.
   - It depends only on distances, so it is invariant to rotation and translation. The
     spectrum is also invariant to atom permutation.
3. **Variants:**
   - `full` — the 29-dim spectrum (classical models only);
   - `compressed` — PCA(k) on the standardized spectrum, **default** k = 8 = number of
     qubits, fit on S_{s,N} only;
   - `composition` — 5-dim, used by classical models and by the quantum model on 5 qubits;
   - optional: composition combined with the compressed spectrum.
4. **Scaling:** fit `StandardScaler` on the training set. Quantum rotation angles are
   γ · (standardized feature), with γ tuned by CV.

---

## 6. Models

Shared protocol, for every (seed, N, representation, variant):
- fit all transforms on S only;
- run 5-fold CV inside S (the same folds for every model) and choose hyperparameters by CV
  MAE;
- refit on all of S, then predict.

### 6.1 Mean baseline

Predict the training-set mean.

### 6.2 RBF kernel ridge

`sklearn.kernel_ridge.KernelRidge(kernel="rbf")` on standardized inputs.
- alpha ∈ {1e-6, 1e-5, …, 1}
- gamma ∈ logspace(−3, 1, 9)

### 6.3 Tree model

`RandomForestRegressor` with n_estimators = 500.
- max_features ∈ {1.0, 0.5, "sqrt"}
- min_samples_leaf ∈ {1, 3, 5}

### 6.4 Quantum kernel ridge regression (QKRR) — the team's quantum regressor

`KernelRidge(kernel="precomputed")` with the quantum fidelity kernel from §7.
- alpha: the same grid as RBF
- γ ∈ {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}
- optional: reps ∈ {1, 2}

### 6.5 Targets

- Raw debye; optionally standardize with the training mean and standard deviation, then
  invert after prediction.
- Negative predictions: decide the policy before the final evaluation and apply it to every
  model equally. **Default:** clip at 0, since |μ| ≥ 0. Record the choice in DECISIONS.md.

---

## 7. Quantum model details (`src/qm9dipole/models/quantum_kernel.py`)

### 7.1 Feature map

**Default:** an IQP-style circuit on n = k qubits arranged as a linear chain. Repeat
r = 2 times:
- H on every qubit;
- RZ(γ·xᵢ) on qubit i;
- RZZ(γ²·xᵢ·xᵢ₊₁) on each nearest-neighbour pair (i, i+1).

Build it once as a parameterized circuit (`ParameterVector`) and bind values per molecule.
Nearest-neighbour entanglers keep transpiled depth low on IBM qubit layouts. Qiskit's
ZZ-style feature map is an acceptable alternative; document it if used.

### 7.2 Exact kernel

- ψ(x) is the `Statevector` of the bound circuit.
- Stack the N statevectors as rows of Ψ (N × 2ⁿ). Then K = |Ψ Ψ†|², taken elementwise.
- Optionally cross-check against qiskit-machine-learning's statevector fidelity kernel.

### 7.3 Concentration diagnostics

For each γ, log the mean and standard deviation of the off-diagonal training-kernel entries,
and flag the run if the standard deviation is below 1e-3.
- If γ is too large, entries collapse toward ~2⁻ⁿ (states nearly orthogonal).
- If γ is too small, entries collapse toward 1 (all states nearly identical).

### 7.4 Units

The KRR prediction ŷ(x) = Σᵢ αᵢ k(x, xᵢ) uses α fit on debye labels, so its output is
already in debye. If a variational regressor is ever added, map its ⟨Z⟩ ∈ [−1, 1] to debye
with an affine transform fit on training labels.

### 7.5 Finite-shot kernel estimation

- The circuit C(x, x′) applies U(x), then U(x′)†, then measures all qubits. The kernel value
  is k = P(all zeros).
- With S shots, the estimate behaves as k̂ ~ Binomial(S, k) / S.
- **Fast path:** draw binomial samples from the exact k. For ideal hardware this is
  statistically equivalent.
- **Validation:** for about 50 pairs, run C(x, x′) on `AerSimulator` through a Sampler with S
  shots and confirm agreement (check the z-scores).
- **Default setup:**
  - training kernel exact;
  - test kernel (n_test × N) estimated with S ∈ {100, 1000, 10000};
  - 5 shot-noise repetitions per setting.
- **Variant:** a shot-noisy training kernel as well. Symmetrize it, set the diagonal to 1, and
  clip negative eigenvalues, since a shot-estimated kernel matrix can lose positive
  semidefiniteness.

### 7.6 Noisy inference

- Simulator: `AerSimulator.from_backend(<Heron-class fake backend from
  qiskit_ibm_runtime.fake_provider>)`.
- Transpile C(x, x′) for that backend; primitives require backend-native (ISA) circuits.
- Run on a subset: 50 molecules per test set, N ∈ {100, 300}, S = 1000.
- Report how much error rises compared with exact and shot-only inference.

### 7.7 Projected quantum kernel (optional, M8)

- Features: f(x) = (⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩) for every qubit q — 3n numbers from one prepared
  state, measured in 3 bases.
- Model: RBF kernel ridge on f.
- Circuit count grows like N, compared with ~N²/2 for the fidelity kernel.
- Less prone to concentration (Huang et al. 2021). Cheapest option for a real-QPU demo.

### 7.8 Quantum cost accounting (`src/qm9dipole/cost.py`)

For the target backend, report:
- number of qubits;
- depth and two-qubit-gate count of U(x) and C(x, x′), before and after transpilation;
- circuits needed:
  - fidelity kernel: N(N−1)/2 for training, N per test molecule;
  - projected kernel: 3N for training, 3 per test molecule;
- shots per circuit and total shots;
- QPU time: measured if a real job runs; otherwise give a stated rough estimate.

Tabulate for N ∈ {100, 300, 1000}.

---

## 8. Invariance checks

1. **Unit tests** (`tests/test_invariance.py`): for about 100 random molecules, check that
   descriptor(transformed) = descriptor(original) to atol 1e-8 under:
   - random rotations (`scipy.spatial.transform.Rotation.random`, seeded);
   - random translations;
   - random permutations (permute `Z` and `R` together).
2. **End-to-end:** for every trained model (at least one N per seed), predict on transformed
   test molecules and report max and mean |Δμ̂| in debye. For finite-shot runs, use the same
   shot RNG, or show that the differences sit below shot noise.
3. **Negative control:** a deliberately non-invariant descriptor (flattened, zero-padded raw
   coordinates) must **fail** these checks. This shows the tests can actually catch errors.

Output: `results/invariance.csv`, plus a table in the writeup.

---

## 9. Experiments, outputs, milestones

### 9.1 Grid

For each seed in {0, 1, 2} and N in {100, 300, 1000}:

| Model | Inputs |
|---|---|
| mean | — |
| RBF-KRR | CM `full`, CM `compressed`, `composition` |
| RF | CM `full`, CM `compressed`, `composition` |
| QKRR | CM `compressed` (8 qubits), `composition` (5 qubits) |

Evaluate every combination on the familiar and unseen test sets (final evaluation only).

### 9.2 Results schema (long-format CSV)

Columns: `run_id`, `git_hash`, `config_hash`, `seed`, `n_train`, `model`, `representation`
(composition | cm), `input_variant` (full | compressed), `eval_set` (cv | familiar | unseen),
`mae_D`, `rmse_D`, `n_eval`, `shots` (empty for exact), `noise` (none | fake backend name),
`hyperparams` (JSON).

### 9.3 Figures and tables

1. Learning curves:
   - MAE vs. N on log–log axes, mean ± std over seeds;
   - one panel per test set (familiar, unseen), one line per model and variant;
   - the same plot for RMSE (appendix).
2. Ablation: composition-only vs. Coulomb-matrix spectrum, per model and N.
3. QKRR error vs. shots (exact, 10⁴, 10³, 10²), with the noisy-simulation points.
4. Tables:
   - main results at N = 1000;
   - paired differences (QKRR − RBF-compressed) per seed;
   - invariance results;
   - quantum cost.
5. Optional: kernel-concentration plot.

With 3 seeds, report means ± std and paired differences. Avoid overclaiming significance.

### 9.4 Milestones and acceptance criteria

- **M0 — environment and data**
  - Install the stack and pin versions; `pytest` runs.
  - QM9 tarball, `uncharacterized.txt` and the readme are in `data/raw/`, with checksums
    logged.
- **M1 — parser, exclusions and splits**
  - Parsed table saved; parser tested on a few known molecules.
  - Exclusions applied and documented.
  - `build_splits.py` writes all split files, every assert in §4 passes, and a split report
    (formula counts and sizes) is printed.
- **M2 — descriptors and invariance**
  - Composition and Coulomb-matrix-spectrum descriptors implemented.
  - Invariance tests pass; the negative control fails.
- **M3 — classical harness**
  - `run_dev.py` produces CV-only learning curves for the mean, RBF and RF models across all
    variants, seeds and N.
  - No test-set evaluation yet.
- **M4 — QKRR**
  - Feature map and exact kernel implemented, with a unit test (kernel symmetric,
    diagonal = 1, positive semidefinite).
  - γ and alpha chosen by CV; concentration diagnostics logged; plugged into the harness.
- **M5 — shots and noise**
  - Binomial shot model validated against real Sampler runs on `AerSimulator`.
  - Shot sweep and noisy-backend subset results produced.
- **M6 — ablation and cost**
  - Composition-only results for all models.
  - Cost table generated by `cost.py`.
- **M7 — final**
  - Configs frozen in `configs/frozen.yaml` (hash recorded).
  - `final_eval.py` run once, producing `results/final_*.csv`.
  - Figures generated.
  - Writeup (`README.md`) states the honest conclusion, the geometry provenance caveat, the
    exclusions, and any post-hoc runs.
- **M8 — optional**
  - Projected quantum kernel.
  - A small real-QPU inference demo (ask first; state circuits, shots and estimated QPU
    time).
  - Extensions from §11.

Any rerun made after seeing test results must be labeled post-hoc in DECISIONS.md and in the
writeup.

---

## 10. Gotchas

- **Leakage:**
  - Mulliken charges and every QM9 property other than μ are off-limits as inputs.
  - Fitting the scaler, PCA or target scaling on anything beyond the current training set
    also leaks.
- **Anchors limit the familiar set:** the number of familiar formulas must be below N1.
- **Parsing:** convert the Fortran `*^` exponent before calling `float()`.
- **Kernel matrices:**
  - Shot-estimated kernels may not be positive semidefinite; use the projection in §7.5.
  - Bandwidth γ can be too large or too small (see §7.3). Choose it only by CV.
- **Hardware execution:**
  - Transpile to the backend's native gate set before running on fake-backend noise models or
    real hardware.
  - Qiskit uses little-endian bit order. It doesn't affect P(all zeros), but matters for any
    other bitstring.
- **Units:** coordinates in Å (convert to bohr for the Coulomb matrix), labels in debye.
  Never mix them silently.
- **Clipping:** decide the clipping policy for negative predictions in advance.
- **Reproducibility:** record versions, seeds, config hash and git hash in every results
  file.

---

## 11. Possible extensions (after M7)

- **Physics-informed charge model:**
  - A quantum model predicts a partial charge per atom from its local environment.
  - Charges are constrained to sum to zero, and the model outputs |Σᵢ qᵢ rᵢ|.
  - Invariance comes by construction, and it also yields the dipole vector.
- **Learning-curve theory:** use kernel eigenvalue spectra and kernel–target alignment to
  explain why one kernel learns faster per label.
- **ML-based error mitigation** for the noisy-inference results.
- **Trainable quantum kernel:** optimize the feature map for kernel–target alignment.

---

## 12. References

- Ramakrishnan, Dral, Rupp, von Lilienfeld, "Quantum chemistry structures and properties of
  134 kilo molecules," *Scientific Data* 1, 140022 (2014).
- Ruddigkeit, van Deursen, Blum, Reymond, "Enumeration of 166 billion organic small
  molecules in the chemical universe database GDB-17," *J. Chem. Inf. Model.* (2012).
- Rupp, Tkatchenko, Müller, von Lilienfeld, "Fast and accurate modeling of molecular
  atomization energies with machine learning," *Phys. Rev. Lett.* 108, 058301 (2012) —
  Coulomb matrix.
- Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," *Nature* 567,
  209 (2019) — quantum kernels.
- Huang et al., "Power of data in quantum machine learning," *Nature Communications* 12,
  2631 (2021) — projected quantum kernels.
- Thanasilp, Wang, Cerezo, Holmes, "Exponential concentration in quantum kernel methods,"
  *Nature Communications* (2024).
