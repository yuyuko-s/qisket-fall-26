# QM9 dipoles from fewer labels

Qiskit Fall Fest 2026 (University of Ottawa), hackathon prompt 07 (QML, advanced).

We predict the **dipole-moment magnitude |μ| (debye)** of small organic molecules from QM9 with
a team-built **quantum kernel ridge regressor**. We compare it fairly against classical models as
the number of training labels grows, on molecules whose formulas were seen in training
(*familiar*) and on formulas never seen (*unseen*). Beating classical ML is not the goal; a fair
comparison and an honest conclusion are.

## How we work

**The headline comparison is strict.** It uses only atom types and 3D coordinates, as the brief
asks:
- every model sees the same nested training sets (100 ⊂ 300 ⊂ 1000 molecules, 3 seeds);
- settings are tuned by cross-validation inside the training set only;
- the test sets are evaluated exactly once, at the end;
- every representation is tested for invariance to rotation, translation and atom reordering.

**Around it, we explore freely.** QM9 also records each atom's partial charge and 14 other
computed properties. A dipole is roughly a charge-weighted sum of atom positions, so the
charges alone already track |μ| closely. We use these extras to ask questions such as "how much
of the dipole do the charges explain?" and "does predicting charges first help a quantum
model?". Results that use them are labeled and reported separately, because those properties
come from the same quantum-chemistry calculation as the answer. Decisions, experiments and
dead ends are logged in `docs/DECISIONS.md`.

## Status

| Milestone | | Notebook |
|---|---|---|
| M0 Environment and QM9 download | done | `notebooks/00_setup_and_data.ipynb` |
| M1 Parser, exclusions, splits | done | `notebooks/01_parse_and_splits.ipynb` |
| M2 Descriptors and invariance tests | done | `notebooks/02_descriptors_invariance.ipynb` |
| X1 Classical statistical analysis (exploration): EDA → cleaning and features → scaling and effective dimension → models and effective parameters → new-formula generalization | done | `notebooks/explore_01` … `explore_05` |
| M3 Classical baselines (CV only) | next | |
| M4 Quantum kernel ridge regression | | |
| M5 Finite-shot and noisy inference | | |
| M6 Representation ablation, quantum cost | | |
| M7 Final evaluation, figures, writeup | | |

## Setup

You need Git and either conda or Python 3.12. Every command below runs from the repository
root.

**Option A: conda**

```bash
conda env create -f environment.yml
conda activate qm9dipole
pip install -r requirements.txt
```

**Option B: plain Python.** This needs Python **3.12** installed. Create the venv with 3.12
*explicitly*, because a bare `python` may point to a different version:

```bash
py -3.12 -m venv .venv            # Windows (Python launcher)
python3.12 -m venv .venv          # macOS / Linux

.venv\Scripts\activate            # Windows: PowerShell or cmd
source .venv/Scripts/activate     # Windows: Git Bash
source .venv/bin/activate         # macOS / Linux

python --version                  # must print Python 3.12.x
pip install -r requirements.txt
```

If `python --version` shows another version, delete `.venv` and recreate it with a 3.12
interpreter.

Both options install the exact package versions in `requirements.txt`, plus this project's
`qm9dipole` package in editable mode. The notebooks import it. With conda, run `pip install`
*after* `conda activate`: activation hides any packages in your per-user Python folder, which
would otherwise make pip skip dependencies (see the comment in `environment.yml`).

**Troubleshooting**
- `No matching distribution found for numpy==2.5.3`: the environment is not Python 3.12. The
  pinned numpy needs 3.12, so recreate the venv as above.
- Windows, `OSError: [Errno 2] No such file or directory` with a very long path during
  `pip install`: the path exceeds Windows' 260-character limit. Clone into a shorter folder,
  or enable long paths (*Local Group Policy → Enable Win32 long paths*, or the registry value
  `LongPathsEnabled`).
- The first notebook cell says the package "belongs to the checkout at …": the selected kernel
  is an environment set up for a different copy of the project. Select this copy's environment.

**Choosing the notebook kernel.** The notebooks use the standard `python3` kernel, which means
"the Python of the environment Jupyter runs in":
- **JupyterLab:** with the environment activated, run `jupyter lab` and keep the default kernel.
- **VS Code:** open a notebook, click the kernel picker, choose *Python Environments*, then pick
  `qm9dipole` (conda) or `.venv`.

The first cell of every notebook checks the environment. It stops with instructions if the
package is missing, and warns if installed versions differ from `requirements.txt`.

**Check the install:** `pip check` should report no broken requirements, and `pytest -q`
should report all tests passing.

## Reproduce

Run the notebooks in order. Each runs top to bottom and is safe to re-run.

| Notebook | What it does | Time |
|---|---|---|
| `00_setup_and_data.ipynb` | Downloads QM9 (86 MB) into `data/raw/` and verifies it against the published checksums | ~1 min |
| `01_parse_and_splits.ipynb` | Parses all 133,885 molecules into `data/processed/qm9.parquet`, applies exclusions, rebuilds `splits/` | ~1 min |
| `02_descriptors_invariance.ipynb` | Builds the descriptors, checks invariance to rotation, translation and atom reordering (with a negative control), writes `results/m2_descriptor_invariance.csv` | ~10 s |

Splits are deterministic. Rebuilding them reproduces the committed `splits/*.json` molecule IDs
exactly. Only the `git_hash` stamp changes.

**Exploration X1**, a classical statistical analysis done before any quantum model, is a chain.
Each notebook reads the previous one's output (`src/qm9dipole/explore.py`), so run them in order.
Its thresholds live in `configs/explore.yaml`. It uses the training pool and training sets only.

| Notebook | What it does | Time |
|---|---|---|
| `explore_01_eda.ipynb` | Exploratory data analysis of every QM9 field: roles (headline-legal vs DFT output), integrity, duplicates, distributions, information about \|μ\|, redundancy, outliers; writes recommendations | ~30 s |
| `explore_02_cleaning_and_features.ipynb` | Acts on the EDA: record- and feature-level cleaning, engineered features (bond dipoles, bond orders, rings, inertia), feature catalog | ~20 s |
| `explore_03_standardization_and_dimension.ipynb` | Chooses feature scaling and target transform by CV; PCA and the cost of 8-component compression; effective dimension of the inputs | ~40 min |
| `explore_04_classical_models.ipynb` | Mean, linear, ridge, RBF kernel ridge, random forest, XGBoost learning curves; effective number of parameters of each fitted model | ~40 min |
| `explore_05_generalization_and_diagnostics.ipynb` | Formula-grouped CV (new formulas), error structure, permutation importance | ~10 min |

To run everything without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_and_data.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_parse_and_splits.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_descriptors_invariance.ipynb
pytest -q
# exploration X1, in order (long ones need no cell timeout):
for nb in explore_01_eda explore_02_cleaning_and_features explore_03_standardization_and_dimension           explore_04_classical_models explore_05_generalization_and_diagnostics; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/$nb.ipynb
done
```

The final test-set evaluation (M7) will be a single script, `scripts/final_eval.py`, run once on
frozen settings. Test sets are never used to choose anything before that.

## Quantum-feature regression prototype

`src/qm9dipole/models/quantum.py` implements a fixed Qiskit encoder with a classical
linear ridge readout. `QuantumRidgeRegressor` fits input standardization on training data,
encodes each input column on one qubit with `zz_feature_map`, measures per-qubit Z
expectations using the exact local `StatevectorEstimator`, and fits `Ridge` on debye labels.
Use 4–8 fixed-width input features; the circuit does not itself make raw XYZ inputs invariant.

```python
from qm9dipole.models.quantum import QuantumRidgeRegressor

model = QuantumRidgeRegressor(alpha=1.0, gamma=1.0, reps=2)
model.fit(X_train, y_train)           # y_train: dipole magnitude in debye
predictions = model.predict(X_val)   # debye; negative predictions clipped at 0
features = model.quantum_features(X_val)
```

Defaults are illustrative, not selected hyperparameters. Tune `alpha`, `gamma` and optionally
`reps` with training-only CV. If PCA is learned from descriptors, put it **before this model in
the same sklearn `Pipeline`**, rather than supplying globally fitted PCA scores to CV.
The measured features are dimensionless; the fitted ridge readout maps them to physical units.
This is a linear projected quantum-feature model, not the pairwise fidelity kernel.

Run the synthetic smoke notebook (no QM9 evaluation or hardware access):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_quantum_regression.ipynb
pytest -q tests/test_quantum.py
```

This prototype does not complete the molecular benchmark or M4. At 4–8 qubits, exact CPU
simulation is inexpensive; using quantum features is not a claim of quantum speedup.

## Data

QM9: Ramakrishnan, Dral, Rupp, von Lilienfeld, *Scientific Data* 1, 140022 (2014), from GDB-17
(Ruddigkeit et al. 2012), <https://quantum-machine.org/datasets/>. The data is not redistributed
here: notebook 00 downloads it from figshare. `docs/DATA.md` documents the checksums, the 3,062
excluded molecules (with reasons), the working pool and the splits.

## Repository layout

```
notebooks/      one notebook per milestone (the entry points); explore_* for exploration X1
src/qm9dipole/  all logic: data.py (download, parsing, exclusions), splits.py, descriptors.py,
                features.py, invariance.py, provenance.py; eda.py, cleaning.py, preprocess.py,
                complexity.py, evaluate.py, analysis.py, explore.py, plots.py, models/
                (models/quantum.py: fixed quantum features + linear ridge prototype)
tests/          pytest suite, including negative controls for every split check and invariance test
results/        result tables (CSV), each with a .meta.json provenance file
configs/        splits.yaml (seeds and sizes; its hash is stamped into every split file),
                explore.yaml (exploration thresholds)
splits/         saved molecule-ID lists: test sets and nested training sets
docs/           BRIEF.md (requirements), PLAN.md (design), DATA.md, DECISIONS.md (change log)
figures/        generated figures
data/           raw download and parsed table (not committed)
```

`CLAUDE.md` holds the working rules for the AI coding assistant used on this project. Its
"Evaluation rules" and "Exploration" sections are also the most detailed statement of the
protocol summarized under "How we work".

## Hardware

Everything runs on simulators. Real IBM Quantum hardware is used only by explicit team decision,
with the circuits, shots and estimated QPU time stated beforehand.
