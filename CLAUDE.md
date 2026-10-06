# QM9 dipoles from fewer labels — Qiskit Fall Fest 2026 hackathon

## What this project is

Hackathon prompt 07/08 (QML · Advanced), Qiskit Fall Fest 2026, University of Ottawa.
Learn the dipole-moment magnitude |μ| (debye) of small organic molecules from QM9 with a
**team-built quantum regression model**, and compare it fairly against classical models as
the number of training labels grows.

Deliverable ("SHOW"): prediction error vs. label count, generalization to familiar and unseen
molecular formulas, and quantum cost.

Beating classical ML is **not** required. Fair comparisons and an honest conclusion are.

- Full requirements: `docs/BRIEF.md` (imported below — the brief wins any conflict).
- Design, defaults and milestone acceptance criteria: `docs/PLAN.md`. Read the relevant
  section before starting each milestone.
- Log every design change in `docs/DECISIONS.md` (date, change, reason). Decisions must be
  made before looking at test-set results.

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
2. **Test sets are never used for any choice** — no hyperparameters, feature choices, model
   selection or early stopping. Tune with cross-validation inside the training set only.
   Test metrics are produced once, by `scripts/final_eval.py`, after configs are frozen.
3. **Every fitted transform is fit on the current training set only** (standardization, PCA,
   target scaling, bandwidth heuristics) and refit for every (training size, seed).
4. **Unseen-formula molecules never appear** in any training set or CV fold.
5. **Every familiar test formula has an anchor** (≥1 molecule) in the smallest training set,
   and therefore in all nested sets.
6. **Nested training sets** (N1 ⊂ N2 ⊂ N3), ≥3 sizes × ≥3 seeds, identical subsets for every
   model (paired comparison).
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
in `docs/DECISIONS.md`. Exploration may use cross-validation freely. Like everything else, it
never selects anything on the test sets.

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
- The one exception is `scripts/final_eval.py`, which stays a script so it runs once, as a
  single unit (rule 2).
- `data/raw/dsgdb9nsd.xyz.tar.bz2` is not extracted (it would create 133,885 small files).
  Read it with `tarfile` in streaming mode; it is parsed once into `data/processed/qm9.parquet`.

## Repo layout

```
CLAUDE.md, README.md (setup and reproduction for teammates and judges)
environment.yml, requirements.txt, pyproject.toml
docs/        BRIEF.md, PLAN.md, DATA.md, DECISIONS.md, prompt_slide.png
configs/     splits.yaml, explore.yaml (exploration-chain thresholds), dev.yaml, frozen.yaml
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
  models/quantum_kernel.py
  noise.py         finite-shot and noisy inference
  cost.py          quantum resource accounting
  evaluate.py      metrics, CV (tune), learning-curve harness, permutation importance
  analysis.py      exploratory statistics (variance by formula, PCA-vs-k curves, error breakdowns)
  plots.py         shared figure style; one fixed color per model
scripts/     final_eval.py only
tests/       test_environment.py, test_data_download.py, test_parser.py, test_splits.py,
             test_invariance.py, test_features.py, test_evaluate.py, test_analysis.py,
             test_eda.py, test_cleaning.py, test_preprocess.py, test_complexity.py,
             test_quantum_kernel.py
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
# Exploration X1 (classical statistical analysis), a chain: run in order; each reads the previous one's output.
# 01 EDA (~30 s), 02 cleaning + features (~20 s), 03 scaling + dimension (~40 min), 04 models + df (~40 min), 05 (~10 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/explore_01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/explore_02_cleaning_and_features.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_03_standardization_and_dimension.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_04_classical_models.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/explore_05_generalization_and_diagnostics.ipynb
python scripts/final_eval.py --config configs/frozen.yaml  # M7: run once, at the end (not yet written)
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
| X1 | Exploration: classical statistical analysis before any quantum work (EDA → cleaning → features → scaling → models, `notebooks/explore_01..05`) | done 2026-10-06 |
| M3 | Classical baselines and learning-curve harness (CV metrics only) | |
| M4 | Quantum kernel ridge regression (statevector) | |
| M5 | Finite-shot and noisy inference | |
| M6 | Representation ablation and quantum cost table | |
| M7 | Frozen configs, single final evaluation, figures, writeup | |
| M8 | Optional: projected quantum kernel, small real-QPU run (ask first), extensions | |

M0–M7 are all required by the full brief. Work one milestone at a time. After each, summarize
what was done, what passed, and what the user should check.
