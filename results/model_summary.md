# Model summary: quantum models and their same-size classical counterparts

For building the presentation slides. Every number below is read from the committed test-run results (the source file
is named under each table) by a script, so it matches `notebooks/07_test_comparison.ipynb` exactly. A CSV with the same
numbers, one row per model × setting × test set, is next to this file: [model_summary.csv](model_summary.csv).

**How to read the numbers**
- **MAE and RMSE** are in debye (D); lower is better. MAE is the primary metric (rare zwitterions up to 29.6 D inflate
  RMSE). For scale: predicting the training-set mean gives MAE ≈ 1.14 D (familiar) and ≈ 1.25 D (unseen); a typical
  molecule's |μ| is 2.5–2.7 D.
- **Familiar** = `test_familiar` (5,484 molecules whose formulas appear in training). **Unseen** = `test_unseen`
  (16,818 molecules from 93 formulas absent from all training and development data).
- **"0.492 ± 0.008"** = mean ± standard deviation over 3 seeds (3 different random training sets of the same size).
- **Training data:** nested training sets of N = 100 ⊂ 300 ⊂ 1,000 molecules (identical for every model, so comparisons
  are paired); every setting is tuned by 5-fold cross-validation inside the training set.
- **Simulation:** unless marked otherwise, quantum models are simulated exactly (infinitely many measurements, no
  noise). Sections A6–A9 add finite shots, simulated hardware noise and the IBM hardware run.

**Input feature sets** (all computed from atomic numbers Z and 3D coordinates R only)

| Name | What it is | Size → qubits |
|---|---|---|
| `pls10` | 10 PLS components of the 188 features below: weighted combinations chosen to predict √\|μ\|, refit on each training set (Yeo-Johnson scaling first) | 10 inputs → 10 qubits (5 for the RY-RZ encoder) |
| `composition` | element counts: C, H, N, O, F (the brief's composition-only ablation) | 5 inputs → 5 qubits |
| `all_legal` | all 188 features: radial distributions (106), Coulomb-matrix eigenvalues (29), geometry, bond and ring features (26), functional groups (19), composition (5), charge-equilibration dipole (3) | 188 (classical only) |

All models in parts A and B share the same preprocessing: Yeo-Johnson feature scaling (clipped at ±10), a √|μ| target
transform (predictions mapped back to debye) and identical CV folds.

---

## Part A: Quantum models

**Encoders.** Every quantum model starts from the team's Qiskit encoding circuit U(x): the inputs, multiplied by a
tuned angle scale γ, become rotation angles; running U(x) on |0…0⟩ prepares a quantum state |ψ(x)⟩. Nothing in the
circuit is trained. On IBM Heron hardware one 10-qubit encoding is ~109 gates deep with 36 two-qubit (CZ) gates; a
fidelity-kernel entry (U(x′) then U(x)†) is ~206 deep with 70 CZ gates.


### A1. Quantum kernel ridge regression, ZZ feature map (the headline quantum model)

Encodes the inputs with Qiskit's ZZ feature map (Hadamards, single-qubit phases and two-qubit ZZ phases between neighbouring qubits, two passes). The **kernel** between two molecules is the overlap of their quantum states, k(x, x′) = |⟨ψ(x)|ψ(x′)⟩|², which a device estimates as the probability of reading all zeros after running U(x′) then U(x)†. Kernel ridge regression on that kernel predicts the dipole. Tuned by CV: γ and the ridge penalty α. Chosen as the headline quantum model by CV at N = 1,000. **Cost:** one circuit per pair of molecules.

**Inputs `pls10` (10 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.735 ± 0.062 | 1.062 ± 0.051 | 0.689 ± 0.045 | 1.009 ± 0.062 |
| 300 | 0.578 ± 0.011 | 0.859 ± 0.028 | 0.525 ± 0.003 | 0.810 ± 0.027 |
| 1,000 | 0.492 ± 0.008 | 0.722 ± 0.014 | 0.437 ± 0.010 | 0.691 ± 0.017 |

**Inputs `composition` (5 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.995 ± 0.030 | 1.346 ± 0.030 | 0.879 ± 0.053 | 1.238 ± 0.020 |
| 300 | 0.980 ± 0.039 | 1.329 ± 0.057 | 0.860 ± 0.033 | 1.226 ± 0.043 |
| 1,000 | 0.942 ± 0.011 | 1.281 ± 0.016 | 0.805 ± 0.005 | 1.170 ± 0.007 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### A2. Quantum kernel ridge regression, RY + CZ encoder

Same kernel ridge model with a different encoder: one RY rotation per input, a fixed RX(π/4) mixer, and a ladder of CZ gates between neighbouring qubits, two passes.

**Inputs `pls10` (10 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.739 ± 0.068 | 1.069 ± 0.063 | 0.676 ± 0.074 | 1.004 ± 0.082 |
| 300 | 0.585 ± 0.012 | 0.866 ± 0.033 | 0.533 ± 0.025 | 0.816 ± 0.040 |
| 1,000 | 0.493 ± 0.009 | 0.730 ± 0.017 | 0.433 ± 0.007 | 0.690 ± 0.011 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### A3. Quantum kernel ridge regression, RY-RZ encoder

Same model; each qubit takes two inputs (an RY and an RZ angle), so the 10 inputs fit on 5 qubits.

**Inputs `pls10` (5 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.708 ± 0.056 | 1.031 ± 0.068 | 0.660 ± 0.046 | 0.990 ± 0.088 |
| 300 | 0.586 ± 0.002 | 0.874 ± 0.035 | 0.527 ± 0.006 | 0.823 ± 0.025 |
| 1,000 | 0.499 ± 0.006 | 0.736 ± 0.005 | 0.441 ± 0.013 | 0.700 ± 0.032 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### A4. Product-state kernel (control: no entanglement)

RY encoder with one pass and **no entangling gates**: every qubit evolves independently and the kernel factorizes into a product of cos² terms, a classical kernel written as a circuit. It is the control for entanglement: if it ties the entangled kernels, entanglement adds nothing measurable.

**Inputs `pls10` (10 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.732 ± 0.075 | 1.059 ± 0.065 | 0.673 ± 0.068 | 0.996 ± 0.079 |
| 300 | 0.578 ± 0.011 | 0.860 ± 0.028 | 0.519 ± 0.008 | 0.806 ± 0.029 |
| 1,000 | 0.490 ± 0.009 | 0.720 ± 0.016 | 0.433 ± 0.009 | 0.686 ± 0.017 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### A5. Projected quantum kernel

Same ZZ encoding, but instead of comparing whole states it measures each qubit's Bloch vector (⟨X⟩, ⟨Y⟩, ⟨Z⟩: 30 numbers per molecule) and applies a classical RBF kernel to them, then kernel ridge (Huang et al. 2021). Tuned: γ, the RBF width and α. **Cost:** 3 circuits per molecule instead of one per pair.

**Inputs `pls10` (10 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.724 ± 0.057 | 1.051 ± 0.047 | 0.681 ± 0.045 | 0.997 ± 0.060 |
| 300 | 0.578 ± 0.013 | 0.866 ± 0.036 | 0.527 ± 0.012 | 0.818 ± 0.030 |
| 1,000 | 0.493 ± 0.007 | 0.724 ± 0.016 | 0.440 ± 0.016 | 0.699 ± 0.025 |

**Inputs `composition` (5 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 1.009 ± 0.040 | 1.358 ± 0.018 | 0.917 ± 0.100 | 1.262 ± 0.044 |
| 300 | 0.974 ± 0.022 | 1.322 ± 0.035 | 0.857 ± 0.026 | 1.217 ± 0.018 |
| 1,000 | 0.941 ± 0.009 | 1.285 ± 0.013 | 0.805 ± 0.005 | 1.178 ± 0.003 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### A6. Quantum-feature ⟨Z⟩ ridge (the team's original model)

Same ZZ encoding; measures ⟨Z⟩ on each of the 10 qubits (10 numbers per molecule) and fits linear ridge regression on those quantum features. No trainable gates. **Cost:** 1 circuit per molecule. This is the model that ran on IBM hardware (A9).

**Inputs `pls10` (10 qubits), exact simulation, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.731 ± 0.072 | 1.074 ± 0.077 | 0.699 ± 0.050 | 1.022 ± 0.088 |
| 300 | 0.597 ± 0.010 | 0.883 ± 0.047 | 0.556 ± 0.016 | 0.846 ± 0.050 |
| 1,000 | 0.545 ± 0.011 | 0.781 ± 0.020 | 0.493 ± 0.021 | 0.748 ± 0.013 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### A7. Shot-aware quantum kernels (test run 2, designed after run 1: post-hoc)

On a device every kernel entry (or Bloch coordinate) is estimated from S measurements ("shots"). Tuned on exact
values, the fidelity kernel collapses under shot noise (A8). The shot-aware versions use S-shot estimates **everywhere**:
in cross-validation, in the final fit and in prediction, so CV picks settings that work under that noise (it chooses
10-100× stronger regularization). Same encoders, inputs (`pls10`, 10 qubits) and training sets as A1 and A5.

**Quantum kernel ridge, ZZ map, shot-aware at 100 shots per circuit:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.767 ± 0.065 | 1.091 ± 0.043 | 0.731 ± 0.072 | 1.036 ± 0.058 |
| 300 | 0.665 ± 0.024 | 0.955 ± 0.040 | 0.629 ± 0.016 | 0.917 ± 0.041 |
| 1,000 | 0.577 ± 0.012 | 0.865 ± 0.012 | 0.537 ± 0.027 | 0.815 ± 0.027 |

**Quantum kernel ridge, ZZ map, shot-aware at 1,000 shots per circuit:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.765 ± 0.107 | 1.094 ± 0.099 | 0.714 ± 0.079 | 1.029 ± 0.088 |
| 300 | 0.605 ± 0.013 | 0.898 ± 0.033 | 0.572 ± 0.026 | 0.858 ± 0.048 |
| 1,000 | 0.544 ± 0.003 | 0.796 ± 0.031 | 0.497 ± 0.024 | 0.759 ± 0.053 |

**Quantum kernel ridge, ZZ map, shot-aware at 10,000 shots per circuit:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.740 ± 0.070 | 1.068 ± 0.060 | 0.693 ± 0.051 | 1.013 ± 0.068 |
| 300 | 0.591 ± 0.015 | 0.881 ± 0.015 | 0.547 ± 0.000 | 0.838 ± 0.014 |
| 1,000 | 0.513 ± 0.006 | 0.755 ± 0.006 | 0.464 ± 0.017 | 0.723 ± 0.037 |

**Projected quantum kernel, shot-aware at 100 shots per circuit:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.781 ± 0.063 | 1.103 ± 0.047 | 0.756 ± 0.061 | 1.062 ± 0.068 |
| 300 | 0.668 ± 0.014 | 0.965 ± 0.023 | 0.614 ± 0.029 | 0.913 ± 0.037 |
| 1,000 | 0.605 ± 0.014 | 0.904 ± 0.022 | 0.551 ± 0.028 | 0.856 ± 0.029 |

**Projected quantum kernel, shot-aware at 1,000 shots per circuit:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.768 ± 0.070 | 1.098 ± 0.061 | 0.724 ± 0.053 | 1.045 ± 0.064 |
| 300 | 0.617 ± 0.011 | 0.900 ± 0.028 | 0.574 ± 0.012 | 0.865 ± 0.027 |
| 1,000 | 0.544 ± 0.001 | 0.783 ± 0.017 | 0.495 ± 0.024 | 0.754 ± 0.046 |

*Source: `results/comparison/test_run02_quantum.csv`. For comparison, the exact models are in A1 and A5.*


### A8. Finite shots on models tuned for exact simulation (test run 1)

The exact-tuned models of A1, A5 and A6 run with S shots per circuit, as a device would (ideal device, no noise),
on the quantum test subsets (80 familiar + 370 unseen molecules; the full test sets would need too many circuits).
Fidelity kernel: both the training and the test kernel estimated from shots. Projected kernel and ⟨Z⟩ ridge: every
measured feature from shots, readout refit. Mean over 3 seeds × 5 repetitions.

| Model | N | Exact | 100 shots | 1,000 shots | 10,000 shots |
|---|---|---|---|---|---|
| Quantum kernel ridge (A1) | 100 | 0.795 / 0.955 | 1.087 / 1.180 | 0.848 / 0.996 | 0.795 / 0.959 |
| Quantum kernel ridge (A1) | 300 | 0.724 / 0.821 | 2.824 / 3.126 | 1.069 / 1.242 | 0.782 / 0.879 |
| Quantum kernel ridge (A1) | 1,000 | 0.623 / 0.704 | 64.415 / 77.499 | 10.466 / 10.678 | 2.155 / 2.592 |
| Projected quantum kernel (A5) | 100 | 0.787 / 0.963 | 0.946 / 1.059 | 0.814 / 0.980 | 0.790 / 0.965 |
| Projected quantum kernel (A5) | 300 | 0.714 / 0.821 | 0.881 / 0.942 | 0.762 / 0.857 | 0.717 / 0.822 |
| Projected quantum kernel (A5) | 1,000 | 0.613 / 0.742 | 0.827 / 0.925 | 0.658 / 0.761 | 0.628 / 0.750 |
| ⟨Z⟩ ridge (A6) | 100 | 0.789 / 1.013 | 0.869 / 1.035 | 0.809 / 1.015 | 0.789 / 1.013 |
| ⟨Z⟩ ridge (A6) | 300 | 0.722 / 0.881 | 0.836 / 0.916 | 0.742 / 0.879 | 0.726 / 0.883 |
| ⟨Z⟩ ridge (A6) | 1,000 | 0.625 / 0.749 | 0.779 / 0.869 | 0.640 / 0.770 | 0.624 / 0.749 |
| *RBF kernel ridge, classical twin (B3)* | 100 | 0.776 / 0.917 | no shots needed | | |
| *RBF kernel ridge, classical twin (B3)* | 300 | 0.713 / 0.829 | no shots needed | | |
| *RBF kernel ridge, classical twin (B3)* | 1,000 | 0.624 / 0.699 | no shots needed | | |

Cells: MAE familiar / unseen (quantum test subsets). RMSE is in the CSV. *Sources: `results/hardware/test_run01_shots.csv`; the classical twin from `results/comparison/test_run02_quantum.csv`.*


### A9. Simulated hardware noise and the real IBM device

Trained on one training set of N = 100 (seed 0) and tested on 60 molecules (30 from each quantum test subset), 1,000
shots per circuit. Circuits transpiled for IBM hardware and run on Aer with the noise models of IBM's fake backends:
FakeFez (an IBM Heron r2 processor; test run 1) and FakeQuebec (IBM's fake backend for `ibm_quebec`; test run 3).
The ⟨Z⟩ ridge also ran on the real **IBM `ibm_quebec`** (test run 4: 160 circuits × 1,000 shots, 45 s of QPU time,
job `db3da0kvf2bc73csuk60`), with its readout refit on the device's own measurements. Projected kernel and ⟨Z⟩ ridge
run end to end (training and test circuits measured); the fidelity kernel's test-kernel entries are measured
(training kernel exact), with an optional depolarizing correction.

| Model | Exact | Ideal device, 1,000 shots | FakeFez noise | FakeQuebec noise | **IBM `ibm_quebec`** |
|---|---|---|---|---|---|
| ⟨Z⟩ ridge (team model) | 0.794 (1.158) | 0.816 (1.178) | 0.812 (1.200) | 0.798 (1.162) | 0.778 (1.175) |
| Projected quantum kernel | 0.816 (1.155) | 0.858 (1.200) | 0.794 (1.152) | 0.776 (1.114) | — |
| Quantum kernel ridge | 0.833 (1.193) | 0.833 (1.197) | 0.919 (1.273) (corrected: 0.854) | — | — |

Cells: MAE (RMSE) in D on the 60 molecules. On the device, the ⟨Z⟩ ridge scores 0.792 D on the 30 familiar and
0.764 D on the 30 unseen molecules. With 60 molecules each MAE is uncertain by about ±0.1 D, so the device result
means "as good as exact simulation", not better. *Sources: `results/hardware/test_run01_noise.csv`,
`results/hardware/hardware_FakeQuebec_20261007-2019_scores.csv`, `results/hardware/hardware_ibm_quebec_20261007-1925_scores.csv`
(and IBM's job record `..._ibm_record.json`).*

---

## Part B: Classical models of the same size (Track B)

The quantum models' classical counterparts: same training sets (N = 100, 300, 1,000 × 3 seeds), same preprocessing,
same CV folds and tuning budget, and, on `pls10`, exactly the same 10 inputs the quantum models encode. No quantum
hardware is involved.


### B1. Mean predictor

Predicts the training-set mean |μ| for every molecule: the floor any model must beat.

**Test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 1.156 ± 0.053 | 1.483 ± 0.029 | 1.266 ± 0.127 | 1.534 ± 0.114 |
| 300 | 1.142 ± 0.022 | 1.462 ± 0.009 | 1.260 ± 0.060 | 1.521 ± 0.053 |
| 1,000 | 1.138 ± 0.011 | 1.457 ± 0.004 | 1.254 ± 0.032 | 1.514 ± 0.028 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### B2. Ridge regression

Linear regression with an L2 penalty; α by CV. On `pls10` it is the linear counterpart of the ⟨Z⟩ ridge (A6): same inputs, no quantum circuit.

**Inputs `pls10`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.730 ± 0.061 | 1.058 ± 0.055 | 0.676 ± 0.075 | 0.983 ± 0.067 |
| 300 | 0.588 ± 0.016 | 0.874 ± 0.051 | 0.533 ± 0.004 | 0.813 ± 0.027 |
| 1,000 | 0.509 ± 0.009 | 0.742 ± 0.027 | 0.452 ± 0.004 | 0.699 ± 0.012 |

**Inputs `all_legal`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.666 ± 0.053 | 0.987 ± 0.050 | 0.608 ± 0.050 | 0.914 ± 0.056 |
| 300 | 0.567 ± 0.012 | 0.859 ± 0.052 | 0.513 ± 0.014 | 0.808 ± 0.031 |
| 1,000 | 0.490 ± 0.014 | 0.727 ± 0.045 | 0.437 ± 0.003 | 0.695 ± 0.031 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### B3. RBF kernel ridge regression (the matched classical twin)

Kernel ridge regression with the Gaussian (RBF) kernel exp(−γ‖x − x′‖²); γ and α by CV from a grid of the same size as the quantum kernels' (88 candidates), through the same tuning code. On `pls10` it is the **matched twin of every quantum kernel model (A1–A5)**: the only difference is the kernel. On all 188 features it is the best quantum-comparable classical model by CV.

**Inputs `pls10`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.730 ± 0.075 | 1.055 ± 0.065 | 0.666 ± 0.074 | 0.988 ± 0.087 |
| 300 | 0.578 ± 0.010 | 0.861 ± 0.028 | 0.520 ± 0.010 | 0.807 ± 0.029 |
| 1,000 | 0.490 ± 0.009 | 0.720 ± 0.016 | 0.433 ± 0.010 | 0.686 ± 0.017 |

**Inputs `all_legal`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.662 ± 0.040 | 0.982 ± 0.032 | 0.603 ± 0.044 | 0.922 ± 0.047 |
| 300 | 0.549 ± 0.012 | 0.838 ± 0.026 | 0.499 ± 0.024 | 0.790 ± 0.024 |
| 1,000 | 0.453 ± 0.009 | 0.685 ± 0.020 | 0.420 ± 0.015 | 0.679 ± 0.030 |

**Inputs `composition`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.983 ± 0.023 | 1.332 ± 0.035 | 0.865 ± 0.013 | 1.235 ± 0.027 |
| 300 | 0.975 ± 0.028 | 1.313 ± 0.038 | 0.846 ± 0.021 | 1.201 ± 0.016 |
| 1,000 | 0.942 ± 0.011 | 1.279 ± 0.020 | 0.807 ± 0.009 | 1.167 ± 0.012 |

*Source: `results/comparison/test_run01_quantum.csv`.*


### B4. Random forest

An ensemble of regression trees (the brief's tree baseline); tree settings by CV.

**Inputs `pls10`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.747 ± 0.036 | 1.079 ± 0.011 | 0.698 ± 0.085 | 1.028 ± 0.061 |
| 300 | 0.632 ± 0.003 | 0.931 ± 0.004 | 0.594 ± 0.021 | 0.893 ± 0.013 |
| 1,000 | 0.535 ± 0.013 | 0.793 ± 0.024 | 0.493 ± 0.027 | 0.768 ± 0.023 |

**Inputs `all_legal`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 0.669 ± 0.025 | 0.984 ± 0.047 | 0.580 ± 0.022 | 0.895 ± 0.042 |
| 300 | 0.580 ± 0.008 | 0.885 ± 0.005 | 0.495 ± 0.007 | 0.807 ± 0.010 |
| 1,000 | 0.489 ± 0.001 | 0.730 ± 0.015 | 0.412 ± 0.003 | 0.691 ± 0.008 |

**Inputs `composition`, test run 1:**

| Training molecules N | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|
| 100 | 1.015 ± 0.015 | 1.358 ± 0.022 | 0.920 ± 0.040 | 1.258 ± 0.017 |
| 300 | 1.000 ± 0.019 | 1.338 ± 0.035 | 0.875 ± 0.009 | 1.224 ± 0.008 |
| 1,000 | 0.959 ± 0.008 | 1.299 ± 0.016 | 0.825 ± 0.014 | 1.196 ± 0.014 |

*Source: `results/comparison/test_run01_quantum.csv`.*


---

## Part C: For context, the most accurate classical models (Track A)

Not size-matched counterparts: these use every feature (or per-atom features) with their own accuracy-tuned
preprocessing, and train on up to the full 99,198-molecule pool (N = 99,198 with one seed, for cost). They show how
far classical methods go without the quantum models' limits.


### C1. Ridge regression, all 188 features

Linear model with an L2 penalty (log-standard scaling, standardized target).

| Training molecules N | Seeds | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|---|
| 100 | 3 | 0.687 ± 0.065 | 0.999 ± 0.065 | 0.648 ± 0.075 | 0.958 ± 0.100 |
| 300 | 3 | 0.579 ± 0.013 | 0.854 ± 0.045 | 0.545 ± 0.031 | 0.849 ± 0.071 |
| 1,000 | 3 | 0.491 ± 0.008 | 0.712 ± 0.019 | 0.444 ± 0.022 | 0.693 ± 0.032 |
| 3,000 | 3 | 0.461 ± 0.003 | 0.684 ± 0.013 | 0.417 ± 0.009 | 0.661 ± 0.021 |
| 10,000 | 3 | 0.444 ± 0.006 | 0.650 ± 0.017 | 0.401 ± 0.008 | 0.626 ± 0.012 |
| 99,198 | 1 | 0.436 | 0.629 | 0.393 | 0.605 |


### C2. RBF kernel ridge, all 188 features

Gaussian-kernel ridge regression; run up to N = 10,000 (its cost grows as N³).

| Training molecules N | Seeds | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|---|
| 100 | 3 | 0.689 ± 0.054 | 1.006 ± 0.048 | 0.641 ± 0.071 | 0.963 ± 0.086 |
| 300 | 3 | 0.550 ± 0.018 | 0.828 ± 0.042 | 0.514 ± 0.026 | 0.822 ± 0.068 |
| 1,000 | 3 | 0.447 ± 0.001 | 0.669 ± 0.005 | 0.414 ± 0.018 | 0.672 ± 0.028 |
| 3,000 | 3 | 0.391 ± 0.007 | 0.598 ± 0.013 | 0.375 ± 0.013 | 0.614 ± 0.014 |
| 10,000 | 3 | 0.326 ± 0.002 | 0.499 ± 0.004 | 0.325 ± 0.010 | 0.535 ± 0.008 |


### C3. XGBoost, all 188 features

Gradient-boosted decision trees; randomized hyperparameter search with early stopping.

| Training molecules N | Seeds | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|---|
| 100 | 3 | 0.653 ± 0.022 | 0.973 ± 0.006 | 0.604 ± 0.067 | 0.933 ± 0.077 |
| 300 | 3 | 0.558 ± 0.002 | 0.860 ± 0.016 | 0.497 ± 0.018 | 0.814 ± 0.038 |
| 1,000 | 3 | 0.464 ± 0.010 | 0.699 ± 0.037 | 0.420 ± 0.021 | 0.695 ± 0.053 |
| 3,000 | 3 | 0.392 ± 0.010 | 0.603 ± 0.018 | 0.354 ± 0.007 | 0.614 ± 0.016 |
| 10,000 | 3 | 0.329 ± 0.003 | 0.507 ± 0.006 | 0.312 ± 0.009 | 0.542 ± 0.017 |
| 99,198 | 1 | 0.225 | 0.350 | 0.230 | 0.399 |


### C4. Latent-charge network (per-atom features)

A small neural network predicts a charge and a small dipole for every atom from its environment (three rounds of message passing along bonds), and outputs |Σ qᵢ rᵢ + atomic dipoles|: the dipole built the way physics builds it, invariant by construction. The best model from N = 1,000 on; unstable below that.

| Training molecules N | Seeds | Familiar MAE | Familiar RMSE | Unseen MAE | Unseen RMSE |
|---|---|---|---|---|---|
| 100 | 3 | 1.140 ± 0.104 | 1.976 ± 0.538 | 1.083 ± 0.043 | 1.998 ± 0.701 |
| 300 | 3 | 0.689 ± 0.370 | 1.223 ± 0.724 | 0.687 ± 0.404 | 1.391 ± 0.961 |
| 1,000 | 3 | 0.280 ± 0.007 | 0.503 ± 0.028 | 0.276 ± 0.017 | 0.537 ± 0.044 |
| 3,000 | 3 | 0.230 ± 0.093 | 0.484 ± 0.160 | 0.206 ± 0.061 | 0.468 ± 0.135 |
| 10,000 | 3 | 0.095 ± 0.004 | 0.170 ± 0.010 | 0.090 ± 0.002 | 0.196 ± 0.010 |
| 99,198 | 1 | 0.040 | 0.069 | 0.038 | 0.084 |

*Source: `results/classical/test_run01_track_a.csv`.*


---

## At a glance: N = 1,000, test MAE in D (familiar / unseen), mean over 3 seeds

| Quantum model | Inputs, qubits | Quantum | Classical counterpart (same inputs, data, tuning) | Classical |
|---|---|---|---|---|
| QKRR, ZZ map (headline) | `pls10`, 10 | 0.492 / 0.437 | RBF kernel ridge on `pls10` | 0.490 / 0.433 |
| QKRR, RY + CZ | `pls10`, 10 | 0.493 / 0.433 | RBF kernel ridge on `pls10` | 0.490 / 0.433 |
| QKRR, RY-RZ | `pls10`, 5 | 0.499 / 0.441 | RBF kernel ridge on `pls10` | 0.490 / 0.433 |
| Product-state kernel (control) | `pls10`, 10 | 0.490 / 0.433 | RBF kernel ridge on `pls10` | 0.490 / 0.433 |
| Projected quantum kernel | `pls10`, 10 | 0.493 / 0.440 | RBF kernel ridge on `pls10` | 0.490 / 0.433 |
| ⟨Z⟩ ridge (team model) | `pls10`, 10 | 0.545 / 0.493 | Ridge regression on `pls10` | 0.509 / 0.452 |
| QKRR, ZZ map | `composition`, 5 | 0.942 / 0.805 | RBF kernel ridge on `composition` | 0.942 / 0.807 |
| Projected quantum kernel | `composition`, 5 | 0.941 / 0.805 | RBF kernel ridge on `composition` | 0.942 / 0.807 |
| QKRR, shot-aware at 1,000 shots (run 2) | `pls10`, 10 | 0.544 / 0.497 | RBF kernel ridge on `pls10` | 0.490 / 0.433 |
| Projected kernel, shot-aware at 1,000 shots (run 2) | `pls10`, 10 | 0.544 / 0.495 | RBF kernel ridge on `pls10` | 0.490 / 0.433 |

For scale at N = 1,000: best quantum-comparable classical (RBF kernel ridge on all 188 features) 0.453 / 0.420; most accurate
classical (latent-charge network) 0.280 / 0.276; predicting the mean 1.138 / 1.254. With all 99,198 training molecules the charge network
reaches 0.040 / 0.038.

