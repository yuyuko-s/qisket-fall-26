# Archived quantum comparisons

The active presentation uses **quantum-feature ⟨Z⟩ ridge** (`qridge`), implemented in
[quantum.py](../../src/qm9dipole/models/quantum.py). The following five quantum models are archived:

| Model name | Model |
|---|---|
| `qkrr` | Fidelity kernel ridge, ZZ encoding |
| `qkrr_ry` | Fidelity kernel ridge, RY + CZ encoding |
| `qkrr_ryrz` | Fidelity kernel ridge, RY-RZ encoding |
| `qkrr_product` | Product-state kernel, no entanglement |
| `pqk` | Projected quantum kernel |

Their implementation is [quantum_kernel.py](../../src/qm9dipole/archive/quantum_kernel.py),
importable as `qm9dipole.archive.quantum_kernel`. Shared ⟨Z⟩ readout helpers live in the active
[quantum_readout.py](../../src/qm9dipole/models/quantum_readout.py); the archive re-exports them
for historical reproduction.

- [docs/](docs/): archived reference documentation and decision history.
- [Previous writeup](README_previous.md): original multi-model results and discussion.
- [notebooks/](notebooks/): broader development comparisons, shots/noise study and original test presentation.
- [configs/](configs/): original run-1 and run-2 configurations.
- [scripts/](scripts/): historical evaluation and hardware drivers.
- [tests/](tests/): kernel-specific tests, excluded from the default pytest testpaths.

Run historical scripts from the repository root, using the installed project environment:

```bash
python archive/quantum_models/scripts/final_eval.py --config archive/quantum_models/configs/test_eval_run1.yaml --dry-run --smoke
python archive/quantum_models/scripts/hardware_run.py --backend ibm_quebec
pytest archive/quantum_models/tests/test_quantum_kernel.py
```

Archived notebooks still read the original files in `results/` and `figures/`. Those historical artifacts,
including mixed tables containing ⟨Z⟩ ridge, remain at their original paths so provenance and recorded hardware
jobs remain traceable. The active presentation filters the tables to `qridge` and classical baselines.
Historical decision logs describe the scope at the time; they are not the active model roster.
