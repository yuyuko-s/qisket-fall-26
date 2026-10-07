# Overnight report, 2026-10-07: classical analysis finished, quantum models built and compared

All numbers are on the **development sets** (`dev`: 5,000 molecules with familiar formulas; `dev_unseen`: 4,183
molecules of 26 formulas absent from every training set). **No test set was read.** Quantum models ran on local
simulators only; nothing was sent to IBM hardware. Every design choice is logged in `docs/DECISIONS.md` (rows dated
2026-10-07), most of them before the numbers they concern existed.

## 1. Headline

**The team's quantum kernel regressor works and is invariant, but it ties its classical twin and is far from the best
classical model, and on hardware it would be both unaffordable and destroyed by shot noise.**

| Development MAE (D), mean over 3 seeds, `dev` / `dev_unseen` | N = 100 | N = 300 | N = 1,000 |
|---|---|---|---|
| quantum kernel ridge, ZZ map, 10 qubits (headline; chosen by CV) | 0.745 / 1.003 | 0.581 / 0.834 | 0.499 / 0.734 |
| matched classical twin: RBF kernel ridge on the same 10 inputs | 0.738 / 0.976 | 0.582 / 0.837 | 0.497 / 0.735 |
| best Track B model (≤ 1,000 labels, any input; chosen by CV) | 0.694 / 0.967 | 0.554 / 0.832 | 0.463 / 0.747 |
| best Track A model at that N (any legal input and model) | 0.672 / 0.993 | 0.561 / 0.859 | 0.281 / 0.496 |
| best Track A model with all 99,198 labels (charge network) | 0.042 / 0.066 | | |
| mean predictor | 1.198 / 1.304 | 1.185 / 1.286 | 1.181 / 1.283 |

- Quantum − twin, paired per seed on `dev`: +0.007 ± 0.011, −0.001 ± 0.004, +0.002 ± 0.002 D. A product-state control
  with no entanglement (a classical cosine kernel) ties as well (+0.000 ± 0.001 D at N = 1000).
- At 1,000 shots per kernel entry the headline model's error is **12.2 D** (2.5 D at 10,000 shots); training it on
  1,000 molecules and predicting the development sets would take ~700 QPU-hours (~4,200 months of the Open Plan).
- The projected quantum kernel and the team's ⟨Z⟩ ridge survive shots (0.56 and 0.58 D at 1,000 shots, N = 1000)
  and cost 0.7–2.2 QPU-hours: the only candidates for a real-hardware demonstration, if the team wants one.

## 2. What ran

| Notebook | Commit | What it produced |
|---|---|---|
| `explore_04_classical_models` | `ee8147a` | Track A learning curves to 99,198 labels; Track B baselines; effective df |
| `explore_05_generalization_and_diagnostics` | `ee8147a` | new formulas, error structure, importance, charge-network diagnostics, invariance |
| `explore_06_quantum_vs_classical` | `454e908` | quantum vs classical on identical inputs; qubit count, inputs, kernel diagnostics, invariance (fits computed at `09a6a30`, reused from the cache; the last commits only changed figure legends) |
| `explore_07_shots_noise_cost` | `09a6a30` | cost on IBM Heron, finite shots, FakeFez noise |

explore_03 was not re-run (its choices stand; see DECISIONS.md); its summary was written from its existing outputs.

## 3. What broke, and how it was fixed

1. **Ridge blow-ups (the handoff's §4), two mechanisms.** (a) Radial-distribution bins at physically empty distances
   (an O–F pair 0.9 Å apart) hold only Gaussian tails; with 1–3 non-zero training rows their sd is ~1e-12 and a new
   molecule's z-score reaches 10⁷. (b) A fitted Yeo-Johnson curve extrapolates steeply for molecules outside the
   training range (z ≈ −10⁴ for a small molecule's padded Coulomb eigenvalue; ridge predicted 1.3 × 10⁵ D on
   dev_unseen). **Fix:** every unbounded scaling clips scaled values at ±10 (`preprocess.ClipScaled`), with
   regression tests for both. The first explore_04 run (started before the fix) confirmed the damage: unguarded Track A
   ridge had a mean development MAE of 17,899 D at N = 100. That run was stopped after Track A (with the user's OK) and
   explore_04 re-run at the fixed commit, reusing every cached fit the fix did not touch.
2. **Cache keys contained memory addresses.** `TabularFitter` hashed `repr(estimator)`, which for the √|μ| target
   prints `<function … at 0x…>`; every Track B fit missed the cache in a new process. Fixed (`fitters.stable_repr`,
   cross-process test). Results were never affected.
3. **Two reporting flaws in notebook 06, caught before its final run:** the "best Track A" reference used the
   N = 1000 winner (the charge network) at every N, although it is poor at N = 100 (1.151 D vs XGBoost 0.672 D), which
   would have shown the quantum model "beating" Track A there; and the product-state control (a classical kernel)
   was eligible as the headline quantum model. Both fixed; references are now chosen per N.
4. **float32 features.** The feature table stores float32; a direct refit passed them to scikit-learn, which then
   computed Yeo-Johnson/PLS in float32 (up to 0.024 D difference). The notebooks now cast to float64, as the harness
   already did.
5. Smaller: Aer's density-matrix method for noisy runs (4× faster than per-shot trajectories); a widened kernel grid
   (both families alike) after optima sat at the grid edges in a smoke run; 16 qubits limited to seed 0 at N ≤ 300
   (compute).

## 4. Classical results (explore_04, explore_05)

Development MAE in debye, mean over 3 seeds (`dev` / `dev_unseen`):

| Training labels N | 100 | 300 | 1,000 | 10,000 | 99,198 (full pool) |
|---|---|---|---|---|---|
| mean predictor | 1.198 / 1.304 | 1.185 / 1.286 | 1.181 / 1.283 | 1.176 / 1.286 | 1.176 / 1.285 |
| ridge, all 188 features (Track A) | 0.700 / 0.969 | 0.589 / 0.862 | 0.499 / 0.738 | 0.454 / 0.696 | 0.446 / 0.687 |
| RBF kernel ridge, all features (Track A) | 0.703 / 0.988 | **0.561** / 0.859 | 0.458 / 0.722 | 0.331 / 0.570 | (O(N³): not run) |
| XGBoost (Track A) | **0.672** / 0.993 | 0.575 / 0.844 | 0.470 / 0.724 | 0.337 / 0.577 | 0.226 / 0.438 |
| latent-charge network (Track A) | 1.151 / 1.410 | 0.704 / 1.009 | **0.281 / 0.496** | **0.097 / 0.160** | **0.042 / 0.066** |
| best Track B (≤ 1000, chosen by CV) | 0.694 / 0.967 | 0.554 / 0.832 | 0.463 / 0.747 | | |

- **Best model from Z and R:** the latent-charge network (three message-passing rounds, bond-directed atomic dipoles):
  0.042 D MAE, RMSE 0.079 D, R² 0.997 on `dev` with all labels; 0.066 D on formulas never seen in training. It is the
  only model whose error keeps falling steeply with data (log–log slope −0.45 above N = 1000). Below N = 1000 it is
  unreliable; XGBoost (N = 100) and RBF kernel ridge (N = 300) are best there.
- **Composition alone** (the brief's ablation) is barely better than the mean (0.986 D at N = 1000).
- **What predicts |μ|:** dipole estimates computed from the geometry (QEq charge-equilibration dipole, bond-dipole
  sum), then nitrile and carbonyl groups. The charge network's learned charges correlate with DFT Mulliken charges at
  r = 0.65, although it never saw a charge (exploration).
- **Errors** are flat across |μ| up to 6 D and concentrated in the rare zwitterions (≥ 9 D). The new-formula penalty is
  ~0.24 D for the feature-based models at any N, but only 0.024 D for the charge network on the full pool; part of the
  dev vs dev_unseen gap is the 26-formula holdout being harder (formula-grouped CV costs only 0.01–0.02 D).
- Invariance holds end to end for the charge network (≤ 1e-6 D) and fails for the raw-coordinate negative control.

## 5. Quantum results (explore_06, explore_07)

**Models built** (`models/quantum_kernel.py`, `models/qsim.py`; simulated exactly): quantum kernel ridge with the
team's three encoders (ZZ map; RY + CZ; RY-RZ with two inputs per qubit), a product-state control without
entanglement, the projected quantum kernel (Huang et al. 2021), and the team's quantum-feature ridge (unchanged model,
now with a 10×-faster exact simulation path). All are tuned by the same protocol as the classical models; with the RBF
kernel plugged in, the quantum tuner reproduces the classical harness exactly (unit-tested), and the classical twins
reproduce notebook 04's Track B rows to 1e-16 D.

**Comparison (explore_06):**
- Every quantum kernel is within 0.013 D of RBF kernel ridge on the same inputs at every N, and within 0.011 D at every
  qubit count k = 4 … 12 (16 qubits at seed 0, N = 300: 0.591 vs 0.588 D). Error falls with k up to 12 for both.
- Inputs: PLS beats top-k by mutual information and PCA for both families; composition alone (5 qubits) is barely
  better than the mean (0.986 D), for both.
- Why they tie: CV picks the smallest angle scale, where the quantum kernel behaves like a smooth classical kernel
  (same effective df, 71.4 vs 71.7; similar spectra). At larger angles the fidelity kernel concentrates exponentially
  with qubit count (mean off-diagonal entry at γ = 1: 0.076 on 4 qubits, 0.0014 on 10, ≈ 0 on 16) and the
  kernel-target alignment collapses. The projected kernel does not concentrate but is no more accurate.
- The team's quantum-feature ridge (0.547 D at N = 1000) is worse than plain ridge on the same inputs (0.514 D); its
  features equal a classical closed form to 3e-14.
- Familiar vs new formulas: the same penalty for quantum and classical (+0.235 vs +0.239 D at N = 1000).
- End-to-end invariance holds (≤ 4e-13 D); the raw-coordinate negative control fails (1.1–1.3 D).
- On a simulator the quantum kernel follows RBF kernel ridge to N = 10,000 (0.447 vs 0.439 D, seed 0).

**Cost, shots, noise (explore_07):** one 10-qubit fidelity-kernel entry is a 206-deep circuit with 70 CZ gates on
IBM Heron. The binomial shot model was validated against Aer's sampler (z-scores mean −0.01, sd 1.00). Finite shots
break QKRR at N = 1000 (88.6 / 12.2 / 2.5 D at 100 / 1,000 / 10,000 shots): the small-angle regime CV chose puts the
signal in tiny kernel differences that shot noise swamps. FakeFez noise keeps 73% of an overlap circuit's
state, shrinks kernel entries uniformly (a one-circuit-per-molecule correction undoes it), and on a 30-molecule,
N = 100 subset barely changes predictions.

## 6. Committed and pushed

Everything is on `origin/main` (GitHub `yuyuko-s/qisket-fall-26`), by fast-forward only; no other branch, no force:
- `58e1b2e`: the scaling guard, the widened kernel grids and the pre-registered quantum design (DECISIONS.md);
- `ee8147a`: the quantum code (batched simulator, quantum kernels, shots, noise, cost) and notebooks 06/07;
- `4cb3aa2`, `ea068f0`, `09a6a30`: the fixes of §3 (cache keys, reference models, control eligibility, float64);
- `8b5125c`: explore_04/05 outputs and the summaries of 03–05 (the Phase 1 checkpoint, pushed first);
- `90ac8ad`, `454e908`: notebook 06 figure legends; `674bcc9`: explore_06/07 outputs and summaries;
- the commit holding this report, with README.md and CLAUDE.md updated.

Tests: 656 pass (`pytest -q`). Every results file's `.meta.json` names the commit it ran at; none is dirty.

## 7. Open items and decisions for the team

1. **M7 is next:** freeze the configurations (`configs/frozen.yaml`) and write `scripts/final_eval.py`, which reads
   the test sets exactly once. No test-set number exists yet. Natural headline candidates: the charge network (Track
   A), RBF kernel ridge on PLS(10) and QKRR (ZZ map) (Track B / quantum), composition only (ablation).
2. **A real-QPU demonstration is a team decision** (CLAUDE.md rule 9). If wanted, the projected kernel or the team's
   ⟨Z⟩ ridge at N ≤ 300 fit a small budget (notebook 07's cost table gives circuits, shots and QPU time); the
   fidelity kernel does not.
3. **Shot-aware tuning of QKRR** (choosing α and γ with shot noise in the cross-validation) might recover some of the
   accuracy lost to shots; not tried.
4. **Root cause of failure 1 in §3:** drop the physically empty radial-distribution bins in explore_02. The clip
   guards against it, but the bins carry no information.
5. **The charge network is unstable below N = 1000** (one seed failed at N = 300) and the linear charge model across
   seeds at several sizes; an ensemble or longer patience could help before M7.
6. **Housekeeping for the user:** close the VS Code tab of `explore_04` without saving (its kernel was stopped; the
   committed version is the re-run). `git stash@{0}` (an old notebook-00 kernel-name change) was left untouched. The
   team's `models/quantum.py` gained a `simulation_method="batched"` option and a docstring note on `clip_negative`;
   its defaults and other paths are unchanged (DECISIONS.md, for the teammate).
7. **Reproducing without recomputing:** the fits cached tonight were copied into `data/processed/cache/` under the
   final commit's hash, and `data/processed/explore04_dev_predictions.parquet` into `data/processed/` (both gitignored,
   local to this machine). Re-running notebook 04 at that commit reuses Track A's expensive fits (about 7.5 h of
   compute) and recomputes Track B and the effective df (~1.5 h); notebook 06 reuses every fit.
