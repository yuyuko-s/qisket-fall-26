# Results

Every table the project produced, grouped by the question it answers. Each `.csv` has a `.meta.json` file next to
it that records the commit, the configuration and the package versions it was made with. Errors are in debye (D).
`test_run*` files come from the test sets; `explore*` files come from the development sets only (`dev`,
`dev_unseen`). The run log, [test_runs.csv](test_runs.csv), has one row per test run: what changed, the commit and
the config hash. Figures are grouped the same way in [../figures/](../figures/). The code finds every file by its
name (`qm9dipole.provenance.results_file`).

## [comparison/](comparison/): quantum models vs classical models of the same size

Start here. Quantum kernel ridge (three encoders), a no-entanglement control, the projected quantum kernel and the
team's ⟨Z⟩ ridge, each against classical models with the same inputs, training sets and tuning budget (N ≤ 1,000).

| File | What it holds |
|---|---|
| `test_run01_headline.csv` | **The headline table**: the quantum model chosen by CV vs its RBF twin (paired difference), the best quantum-comparable classical model, the best classical model at that N, and the mean baseline; per test set and N |
| `test_run01_quantum.csv` | Every quantum and same-size classical model × input set × N × seed: MAE, RMSE, R² and bias on both test sets and both development sets, with the CV score and the settings chosen |
| `test_run02_quantum.csv` | Run 2 (post-hoc): shot-aware quantum kernels at 100 / 1,000 / 10,000 shots, the exact models and the RBF twins again |
| `test_run02_shot_aware.csv` | The same shot budget, tuned for exact kernels (run 1) vs with the shots inside cross-validation (run 2) |
| `explore06_main.csv` | Development sets: every quantum model vs its classical twin on the same 10 PLS inputs |
| `explore06_headline.csv` | Development sets: the headline comparison as it looked before any test run |
| `explore06_paired.csv` | Paired per-seed differences, quantum minus RBF kernel ridge on the same inputs |
| `explore06_qubits.csv` | Error vs number of inputs/qubits (k = 4–16), quantum vs classical |
| `explore06_inputs.csv` | Input variants (PLS, PCA, top-k, composition only), quantum vs classical |
| `explore06_extended.csv` | QKRR vs RBF kernel ridge up to N = 10,000 (simulator) |
| `explore06_kernel_diagnostics.csv`, `explore06_concentration.csv` | Kernel spectra, effective parameters, kernel-target alignment; exponential concentration vs qubits and angle scale |
| `explore06_invariance.csv` | End-to-end rotation, translation and relabeling test of the quantum model, with a negative control |
| `explore04_track_b.csv`, `explore04_quantum_baselines.csv` | The quantum-comparable classical baselines (Track B: mean, ridge, RBF kernel ridge, random forest, XGBoost on three input sets) |

## [hardware/](hardware/): finite shots, noise, cost and IBM hardware

| File | What it holds |
|---|---|
| `hardware_ibm_quebec_20261007-1925_scores.csv` | **Test run 4, IBM `ibm_quebec`**: the team's ⟨Z⟩ ridge on the device vs exact and ideal-device simulation |
| `hardware_ibm_quebec_20261007-1925_ibm_record.json` | IBM's record of the job: backend, timestamps, execution span, 45 s of QPU time charged |
| `hardware_ibm_quebec_20261007-1925_raw_bits.npz`, `..._features.npz` | The device's raw shot outcomes (160 circuits × 1,000 shots) and the ⟨Z⟩ values computed from them |
| `hardware_ibm_quebec_20261007-1925_job.json`, `..._plan.csv` | The submission record (molecules, settings) and the circuit plan |
| `hardware_FakeQuebec_20261007-2019_*` | Test run 3: the same circuits (plus the projected kernel and a fidelity-kernel block) on the FakeQuebec noise model |
| `test_run01_shots.csv`, `test_run01_shots_summary.csv` | Test run 1: MAE vs shots (100 / 1,000 / 10,000, 5 repetitions) for QKRR, the projected kernel and the ⟨Z⟩ ridge |
| `test_run01_noise.csv`, `test_run01_noise_kernels.npz` | Test run 1: FakeFez noise on 60 test molecules (exact, shots only, noisy, corrected) and the kernels |
| `explore07_circuit_costs.csv`, `explore07_model_costs.csv` | Circuit sizes transpiled for IBM Heron; circuits and QPU time per model and N |
| `explore07_shots.csv`, `explore07_shot_validation.csv`, `explore07_noise.csv` | Development sets: shots, the shot model checked against Aer, FakeFez noise |

## [classical/](classical/): the most accurate classical models (Track A)

| File | What it holds |
|---|---|
| `test_run01_track_a.csv` | Test run 1: ridge, RBF kernel ridge, XGBoost and the latent-charge network from N = 100 to 99,198 |
| `explore04_track_a.csv` | The same on the development sets (3 seeds at every N) |
| `explore04_charge_selection.csv` | Architecture search for the charge network (inner validation split) |
| `explore04_effective_df.csv` | Effective number of parameters of each model |
| `explore05_*.csv` | Formula-grouped CV, permutation importance, end-to-end invariance of the charge network |

## [development/](development/): data, features and preprocessing

| File | What it holds |
|---|---|
| `explore01_recommendations.csv` | Field-by-field data analysis: what to keep and how to treat it |
| `explore02_feature_catalog.csv` | Every feature: block, level (molecule or atom), whether it is a function of Z and R, kept or dropped and why |
| `explore03_preprocessing.json` | The chosen scaling, target transform and the PLS(10) reduction for the quantum-comparable track |
| `explore03_scaling.csv`, `explore03_target.csv`, `explore03_compression_cost.csv` | The evidence for those choices |
| `m2_descriptor_invariance.csv` | Rotation, translation and relabeling test of every descriptor, with a negative control |
