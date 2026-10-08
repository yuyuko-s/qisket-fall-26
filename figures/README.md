# Figures

Grouped like [../results/](../results/README.md). `test_run*` figures are made by `notebooks/07_test_comparison.ipynb`
from the test sets; `explore*` figures by the development notebooks.

| Folder | Key figures |
|---|---|
| [comparison/](comparison/): quantum vs same-size classical | `test_run01_learning_curves.png` (**headline**: test error vs training labels), `test_run02_shot_aware.png` (shot-aware quantum models), `explore06_learning_curves.png`, `explore06_qubits.png` (error vs qubits), `explore06_extended.png` (to N = 10,000), `explore06_concentration.png`, `explore06_spectra.png`, `explore06_models_at_nmax.png` |
| [hardware/](hardware/): shots, noise, cost, IBM hardware | `hardware_comparison.png` (**exact vs simulated noise vs `ibm_quebec`**), `test_run01_shots.png` (error vs shots), `explore07_qpu_time.png`, `explore07_shots.png`, `explore07_noise.png` |
| [classical/](classical/): most accurate classical models | `test_run01_track_a.png` (test learning curves to 99,198 labels), `explore04_learning_curves.png`, `explore05_*.png` (parity, learned charges, new-formula gap, feature importance) |
| [development/](development/): data and features | `m1_pool_overview.png`, `explore01_*.png` (data analysis), `explore02_engineered_information.png`, `explore03_*.png` |
