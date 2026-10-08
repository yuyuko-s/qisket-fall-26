# QM9 dipoles from fewer labels — Qiskit Fall Fest 2026 hackathon

## What this project is

Hackathon prompt 07/08 (QML · Advanced), Qiskit Fall Fest 2026, University of Ottawa.
Learn the dipole-moment magnitude |μ| (debye) of small organic molecules from QM9 with a
**team-built quantum regression model**, and compare it fairly against classical models as
the number of training labels grows.

Deliverable ("SHOW"): prediction error vs. label count, generalization to familiar and unseen
molecular formulas, and quantum cost.

Beating classical ML is **not** required. Fair comparisons and an honest conclusion are.

## Two tracks: accuracy first (exploration X2, 2026-10-06)

The user's directive, which overrides PLAN.md defaults wherever they conflict:
- **Track A (accurate) comes first:** the most accurate and robust classical model that can be
  built from Z and R. Splits, training size (up to the full training pool), features, scaling,
  dimensionality, hyperparameters and model family are chosen for accuracy, with no regard for
  what a quantum computer can handle.
- **Track B (quantum-comparable) comes second:** the same pipeline scaled down (N ≤ 1000, a few
  inputs chosen by CV) to give matched baselines for the quantum model.
- Report every quantum result against **both**. Never present Track B as the best classical can do.

PLAN.md values that were set for the quantum model (N ≤ 1000, 25 molecules per formula, small
test sets, k = 8 inputs, the brief's minimum model list, small fixed grids) are Track B
settings at most. The evaluation rules below still apply to both tracks.

- Full requirements: `docs/BRIEF.md` (imported below — the brief wins any conflict).
- Design, defaults and milestone acceptance criteria: `docs/PLAN.md`. Read the relevant
  section before starting each milestone; where it conflicts with the two-track directive
  above, the directive wins.
- Log every design change in `docs/DECISIONS.md` (date, change, reason). Changes made after
  a test run are allowed; label them post-hoc with the test run they followed (rule 2).

@docs/BRIEF.md

## Team context

- The team is new to quantum computing (a few days of intro lessons) but comfortable with
  linear algebra, statistics and ML. When you introduce a quantum-specific concept or choice
  (feature map, fidelity kernel, shots, transpilation, noise model), explain it briefly in a
  comment or in your end-of-task summary.
- All hackathon submissions must be built with Qiskit.
- Real QPU time is scarce (IBM Open Plan ≈ 10 min per rolling 28 days unless organizers
  provide more). Everything is developed on simulators.

## Evaluation rules (these keep the headline comparison fair)

1. **Headline results use the brief's inputs: atomic numbers and 3D coordinates.** The other
   QM9 properties (Mulliken charges, polarizability, orbital energies and so on) stay in the
   parsed table and are open for exploration (see below). Any result that uses them is
   reported separately and names the properties it used. They come from the same DFT
   calculation as μ, so they answer a different question from the brief's.
2. **Inside a run, tuning uses training data only** — hyperparameters, feature choices, model
   selection and early stopping come from cross-validation or a validation split inside the
   training data, never from the test sets. The development sets in `splits/` (`dev`,
   `dev_unseen`) are training data held out for development; the test sets (`test_*`) are not.
   Test metrics come from `scripts/final_eval.py`. **Iterating after seeing test results is
   allowed** (the team's decision, 2026-10-07): look at a test run, change models or
   settings, and run again. Keep it traceable: every test run is logged in
   `results/test_runs.csv` (run number, commit, config, what changed), and changes made after
   a test run are labeled post-hoc in DECISIONS.md, so the writeup can say how the test sets
   were used. (The brief's full requirements include "Final test results must not select
   settings"; the writeup should state the number of test runs and what changed between them.)
3. **Every fitted transform is fit on the current training set only** (standardization, PCA,
   target scaling, bandwidth heuristics) and refit for every (training size, seed).
4. **Unseen-formula molecules never appear** in any training set or CV fold (neither the test
   holdout formulas nor the development holdout formulas).
5. **Every formula of the quantum familiar test subset has an anchor** (≥1 molecule) in the
   smallest training set, and therefore in all nested sets. For the large familiar test set,
   "familiar at size N" means the formula is in that training set.
6. **Nested training sets** (N1 ⊂ N2 ⊂ … ⊂ full pool), ≥3 sizes × ≥3 seeds, identical subsets
   for every model (paired comparison).
7. **Splits are saved as QM9 molecule IDs** with the seed and config that produced them.
   Exclusions are documented with counts and reasons in `docs/DATA.md`.
8. **Predictions are in debye.** Any bounded circuit output (e.g. ⟨Z⟩ ∈ [−1, 1]) is mapped to
   debye with a transform fit on training labels only.
9. **Never submit jobs to real IBM hardware** without asking first and stating the number of
   circuits, shots, and estimated QPU time.
10. **Invariance tests are required**: rotation, translation and atom relabeling, plus a
    negative control that must fail.

## Exploration

The rules above protect one thing: the headline quantum-vs-classical comparison. Everything
else is open, and what we try and learn is part of the presentation. Ideas worth trying:
- other representations, feature maps, kernels and models;
- the extra QM9 properties, for example:
  - Mulliken charges as an auxiliary target, or as a physics reference: a point-charge
    estimate |Σ qᵢ rᵢ| already tracks μ closely;
  - "what if we knew the charges?" models that show how much of μ the charges explain;
- the extensions in PLAN §11.

Keep exploratory work in its own clearly named notebooks, label its results "exploratory"
with the inputs used, and record what was tried and what was learned (including dead ends)
in `docs/DECISIONS.md`. Exploration may use cross-validation freely; it works on the
training and development sets, and test sets are scored by `scripts/final_eval.py` (rule 2).

## Stack

- Python 3.12 (`requires-python >= 3.12`: the pinned numpy 2.5 needs it)
- `qiskit` 2.x, `qiskit-aer`, `qiskit-ibm-runtime` (fake backends for noise models; optional
  QPU run)
- `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `pytest`
- `qiskit-machine-learning` is optional. Check compatibility with the installed Qiskit before
  depending on it; the fidelity kernel is simple to compute from
  `qiskit.quantum_info.Statevector` directly (see PLAN §7).
- Exact pins are in `requirements.txt`; `pyproject.toml` holds compatible ranges. `pyarrow`
  is included for parquet.
- 8-qubit statevector simulation is cheap; an ordinary CPU is enough.
- **Environment:** the conda env `qm9dipole`: `conda env create -f environment.yml` makes
  Python 3.12, then `pip install -r requirements.txt` *after activation* installs the pins
  (its last line `-e .` installs this package in editable mode). `environment.yml` sets
  `PYTHONNOUSERSITE=1` on activation; set it yourself when calling the env's Python without
  activating. Setup instructions are in `README.md`.

## Working style

- **Notebooks are the drivers**: one per milestone in `notebooks/`, named `NN_topic.ipynb`.
  All logic lives in `src/qm9dipole/` and is tested with pytest; notebooks import it and stay
  thin. Every notebook must run top to bottom and be safe to re-run.
- Notebooks must keep the generic kernelspec `python3` (never a machine-specific kernel name),
  and their first code cell is the environment check
  (`qm9dipole.provenance.check_environment()`).
- The exceptions are `scripts/final_eval.py`, a script so that each test run is a single,
  logged unit (rule 2; `notebooks/07_test_comparison.ipynb` presents its results), and
  `scripts/hardware_run.py`, which sends one capped job to an IBM device only with `--submit` (rule 9).
- `data/raw/dsgdb9nsd.xyz.tar.bz2` is not extracted (it would create 133,885 small files).
  Read it with `tarfile` in streaming mode; it is parsed once into `data/processed/qm9.parquet`.

## Repo layout

```
CLAUDE.md, README.md (setup and reproduction for teammates and judges)
environment.yml, requirements.txt, pyproject.toml
docs/        BRIEF.md, PLAN.md, DATA.md, DECISIONS.md, prompt_slide.png
configs/     splits.yaml, explore.yaml (exploration-chain thresholds), test_eval.yaml (models of each test run)
data/raw/    QM9 downloads (gitignored)
data/processed/  qm9.parquet: parsed table (gitignored; regenerated by notebook 01)
splits/      meta.json, pool.json, test_{unseen,familiar}.json, train_s{seed}_n{N}.json (committed)
notebooks/   00_setup_and_data, 01_parse_and_splits, 02_descriptors_invariance, ... (one per milestone)
             explore_NN_topic: exploratory notebooks (labeled; never read a test set)
src/qm9dipole/
  provenance.py    package versions, git hash, environment check
  data.py          download (M0), parse .xyz (all fields kept), exclusions (M1)
  splits.py        formula holdout, anchors, nested training sets, training pool
  descriptors.py   composition counts, Coulomb-matrix spectrum, raw-coordinate negative control,
                   registry (DESCRIPTORS / MODEL_DESCRIPTORS), named feature frames
  features.py      engineered descriptors (geometry, bonds, bond orders, rings, inertia, dipole
                   proxies) + exploration-only charge/frequency summaries (not descriptors)
  eda.py           data dictionary of every field, integrity checks
  cleaning.py      record- and feature-level cleaning rules (unsupervised)
  preprocess.py    feature scalings and target transforms, as pipeline steps
  complexity.py    effective dimension of inputs; effective number of parameters (df) of models
  explore.py       exploration-chain plumbing: artifacts, loaders, feature sets
  invariance.py    rotation / translation / relabeling checks and the invariance table
  models/classical.py  mean, linear, ridge, RBF-KRR, RF, XGBoost pipelines and CV grids
  models/quantum.py  the team's encoders (build_encoding_circuit) + quantum-feature ridge (simulator only)
  models/qsim.py   batched exact statevectors of a Qiskit circuit (checked against Statevector)
  models/quantum_kernel.py  fidelity and projected quantum kernels, KernelRidgeFitter (the harness
                   protocol for any kernel), kernel diagnostics
  models/fitters.py  harness fitters (mean, tabular, XGBoost, latent-charge network)
  models/charge.py  latent-charge network (numpy)
  noise.py         finite shots (exact binomial), Aer ideal and fake-backend noisy sampling, mitigation
  cost.py          quantum resource accounting (gate counts, transpiled costs, circuits, QPU time)
  evaluate.py      metrics, CV (tune), learning-curve harness, permutation importance
  final.py         test-set evaluation helpers: test-molecule features, shots/noise scoring, run log
  hardware.py      device runs: PUBs (ISA circuits + parameter rows), plan/QPU time, submit, parse, score
  analysis.py      exploratory statistics (variance by formula, PCA-vs-k curves, error breakdowns)
  plots.py         shared figure style; one fixed color per model
scripts/     final_eval.py (test runs), hardware_run.py (IBM device runs; rule 9)
tests/       test_environment.py, test_data_download.py, test_parser.py, test_splits.py,
             test_invariance.py, test_features.py, test_evaluate.py, test_analysis.py,
             test_eda.py, test_cleaning.py, test_preprocess.py, test_complexity.py,
             test_quantum.py, test_quantum_encoding.py (teammate), test_qsim.py,
             test_quantum_kernel.py, test_noise_cost.py, test_final.py, test_hardware.py, ...
results/     CSV outputs, each with a .meta.json provenance sidecar (provenance.save_result)
figures/
```

## Commands (keep this section current as notebooks and scripts land)

```
conda env create -f environment.yml && conda activate qm9dipole
pip install -r requirements.txt     # only after activation (PYTHONNOUSERSITE; see environment.yml)
pip check && pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_and_data.ipynb   # M0: download + checks
jupyter nbconvert --to notebook --execute --inplace notebooks/01_parse_and_splits.ipynb # M1: parquet + splits/
jupyter nbconvert --to notebook --execute --inplace notebooks/02_descriptors_invariance.ipynb # M2: invariance table
# The exploration chain (X2), run in order; each reads the previous one's output. Long ones cache finished fits per
# commit in data/processed/cache/ (resumable). Times on an 8-core CPU, from scratch:
# 01 EDA (~30 s), 02 features (~2 min), 03 scaling + dimension (~1.5 h), 04 Track A + B + df (~9 h; Track A is the
# charge network and XGBoost on 99k molecules), 05 (~30 min), 06 quantum vs classical (~1.5 h), 07 shots/noise/cost (~50 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/explore_01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/explore_02_cleaning_and_features.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_03_standardization_and_dimension.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_04_classical_models.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_05_generalization_and_diagnostics.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_06_quantum_vs_classical.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_07_shots_noise_cost.ipynb
# QM9DIPOLE_SMOKE=1 makes explore_06/07 run one seed at N = 100 with small axes (minutes; development only)
jupyter nbconvert --to notebook --execute --inplace notebooks/02_quantum_regression.ipynb # synthetic quantum prototype, not M4 completion
# M7 test runs (simulators only): --dry-run scores the development sets only; --smoke is a minutes-long check
python scripts/final_eval.py --dry-run --smoke
python scripts/final_eval.py                 # a logged test run (~3-4 h); refuses a dirty tree
jupyter nbconvert --to notebook --execute --inplace notebooks/07_test_comparison.ipynb  # presents the latest run
# IBM device runs (rule 9: ask first). Default backend FakeQuebec = local Aer with ibm_quebec's noise model
python scripts/hardware_run.py --dry-run                      # local, development molecules
python scripts/hardware_run.py --backend ibm_quebec           # plan only: circuits, shots, QPU time
python scripts/hardware_run.py --backend ibm_quebec --submit  # one job, capped by --max-seconds (default 300)
python scripts/hardware_run.py --retrieve <job id>
```

## Conventions

- Pass `numpy.random.Generator` objects explicitly; no global random state.
- Type hints; docstrings state units (Å for coordinates, debye for μ).
- Results are long-format CSV (schema in PLAN §9.2). Every run records config hash, git
  hash, seed, and package versions.
- Small, tested functions. Any change to splits or descriptors needs passing tests.

## Milestones (details and acceptance criteria in PLAN §9.4)

| ID | Milestone | Status |
|---|---|---|
| M0 | Environment setup and QM9 download | done 2026-10-04 (`notebooks/00_setup_and_data.ipynb`) |
| M1 | Parser, exclusions and split builder (with asserts) | done 2026-10-05 (`notebooks/01_parse_and_splits.ipynb`); open: anchor-skew review (DECISIONS.md) |
| M2 | Descriptors and invariance tests (including the negative control) | done 2026-10-05 (`notebooks/02_descriptors_invariance.ipynb`) |
| X1 | Exploration: classical statistical analysis before any quantum work (EDA → cleaning → features → scaling → models, `notebooks/explore_01..05`) | done 2026-10-06; superseded by X2 |
| X2 | Exploration: classical rebuild for accuracy (Track A), then quantum-comparable baselines (Track B); new splits, per-atom features, latent-charge model, learning curves to the full pool (`notebooks/01`, `explore_01..05` rerun) | done 2026-10-07 (summary: `docs/OVERNIGHT_REPORT.md`) |
| M3 | Classical baselines and learning-curve harness (CV metrics only) | done 2026-10-07 as X2 Track B (`explore_04`: CV and development-set scores) |
| M4 | Quantum kernel ridge regression (statevector) | done 2026-10-07 on development sets (`explore_06`; `models/quantum_kernel.py`) |
| M5 | Finite-shot and noisy inference | done 2026-10-07 (`explore_07`: binomial model validated on Aer; shot sweep; FakeFez subset) |
| M6 | Representation ablation and quantum cost table | done 2026-10-07 (composition only in `explore_04`/`06`; cost table in `explore_07`) |
| M7 | Test-set evaluation, figures, writeup | done 2026-10-07: test runs 1 (development settings), 2 (post-hoc: shot-aware kernels), 3 (hardware rehearsal on FakeQuebec), 4 (ibm_quebec); `scripts/final_eval.py`, `notebooks/07_test_comparison.ipynb`, README results |
| M8 | Optional: projected quantum kernel, small real-QPU run (ask first), extensions | done 2026-10-07: projected kernel; ⟨Z⟩ ridge on ibm_quebec (job `db3da0kvf2bc73csuk60`, 160 circuits × 1,000 shots, 45 s QPU; 0.778 D vs 0.794 exact; test run 4) |

M0–M7 are all required by the full brief. Work one milestone at a time. After each, summarize
what was done, what passed, and what the user should check.
