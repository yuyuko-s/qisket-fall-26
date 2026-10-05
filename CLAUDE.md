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

## Non-negotiable rules (judging hinges on these)

1. **Inputs are only atomic numbers and 3D coordinates.** Never use Mulliken charges or any
   other QM9 property as a feature — they come from the same DFT calculation as the label.
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

## Stack

- Python 3.11+
- `qiskit` 2.x, `qiskit-aer`, `qiskit-ibm-runtime` (fake backends for noise models; optional
  QPU run)
- `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `pytest`
- `qiskit-machine-learning` is optional. Check compatibility with the installed Qiskit before
  depending on it; the fidelity kernel is simple to compute from
  `qiskit.quantum_info.Statevector` directly (see PLAN §7).
- Exact pins are in `requirements.txt`; `pyproject.toml` holds compatible ranges. `pyarrow`
  is included for parquet.
- 8-qubit statevector simulation is cheap; a laptop CPU is enough.
- **Environment:** the conda env `qm9dipole`: `conda env create -f environment.yml` makes
  Python 3.12, then `pip install -r requirements.txt` *after activation* installs the pins
  (its last line `-e .` installs this package in editable mode). The older env `qff26` has the same pins and also works.
  `PYTHONNOUSERSITE=1` must be set: `environment.yml` sets it on activation, and when calling an
  env's python.exe directly without activating, set it yourself. Setup for others is in `README.md`.

## Working style

- **Notebooks are the drivers**: one per milestone in `notebooks/`, named `NN_topic.ipynb`.
  All logic lives in `src/qm9dipole/` and is tested with pytest; notebooks import it and stay
  thin. Every notebook must run top to bottom and be safe to re-run.
- Notebooks must keep the generic kernelspec `python3` (never a machine-specific kernel name),
  and their first code cell is the environment check
  (`qm9dipole.provenance.check_environment()`).
- The one exception is `scripts/final_eval.py`, which stays a script so it runs once, as a
  single unit (rule 2).
- `data/raw/dsgdb9nsd.xyz.tar.bz2` is **never extracted** (OneDrive sync). Read it with
  `tarfile` in streaming mode.

## Repo layout

```
CLAUDE.md, README.md (setup and reproduction for teammates and judges)
environment.yml, requirements.txt, pyproject.toml
docs/        BRIEF.md, PLAN.md, DATA.md, DECISIONS.md, prompt_slide.png
configs/     splits.yaml, dev.yaml, frozen.yaml
data/raw/    QM9 downloads (gitignored)
data/processed/  qm9.parquet: parsed table (gitignored; regenerated by notebook 01)
splits/      meta.json, pool.json, test_{unseen,familiar}.json, train_s{seed}_n{N}.json (committed)
notebooks/   00_setup_and_data.ipynb, 01_parse_and_splits.ipynb, one per milestone
src/qm9dipole/
  provenance.py    package versions, git hash
  data.py          download (M0), parse .xyz, exclusions (M1)
  splits.py        formula holdout, anchors, nested training sets
  descriptors.py   composition counts, Coulomb-matrix spectrum
  models/classical.py
  models/quantum_kernel.py
  noise.py         finite-shot and noisy inference
  cost.py          quantum resource accounting
  evaluate.py      metrics, CV, learning-curve harness
scripts/     final_eval.py only
tests/       test_environment.py, test_data_download.py, test_parser.py, test_splits.py,
             test_invariance.py, test_quantum_kernel.py
results/     CSV outputs
figures/
Handoff files/   original handoff bundle (gitignored archive; docs/ is canonical)
```

## Commands (keep this section current as notebooks and scripts land)

```
conda env create -f environment.yml && conda activate qm9dipole
pip install -r requirements.txt     # only after activation (PYTHONNOUSERSITE; see environment.yml)
pip check && pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_and_data.ipynb   # M0: download + checks
jupyter nbconvert --to notebook --execute --inplace notebooks/01_parse_and_splits.ipynb # M1: parquet + splits/
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
| M2 | Descriptors and invariance tests (including the negative control) | |
| M3 | Classical baselines and learning-curve harness (CV metrics only) | |
| M4 | Quantum kernel ridge regression (statevector) | |
| M5 | Finite-shot and noisy inference | |
| M6 | Representation ablation and quantum cost table | |
| M7 | Frozen configs, single final evaluation, figures, writeup | |
| M8 | Optional: projected quantum kernel, small real-QPU run (ask first), extensions | |

M0–M7 are all required by the full brief. Work one milestone at a time. After each, summarize
what was done, what passed, and what the user should check.
