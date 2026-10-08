# Evaluation rules

The protocol this project follows: the two tracks, the rules that keep the quantum-vs-classical comparison fair, how exploration is kept separate, and coding conventions. Code and notebooks cite these as "docs/EVALUATION_RULES.md rule N". How each rule was applied is logged in [DECISIONS.md](DECISIONS.md).

## Two tracks (decided 2026-10-06)

- **Track A (accurate):** the most accurate and robust classical model that can be built from Z and R. Training
  size (up to the full training pool), features, scaling, dimensionality, hyperparameters and model family are
  chosen for accuracy, with no regard for what a quantum computer can handle.
- **Track B (quantum-comparable):** the same pipeline scaled down (N ≤ 1000, a few inputs chosen by CV) to give
  matched baselines for the quantum models.
- Every quantum result is reported against **both**; Track B is never presented as the best classical can do.
- The requirements are in [BRIEF.md](BRIEF.md) (the brief wins any conflict). Every design change is logged in
  [DECISIONS.md](DECISIONS.md) (date, change, reason); changes made after a test run are labeled post-hoc with the
  test run they followed (rule 2).

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
else is open, and what we try and learn is part of the presentation. Ideas that were open:
- other representations, feature maps, kernels and models;
- the extra QM9 properties, for example:
  - Mulliken charges as an auxiliary target, or as a physics reference: a point-charge
    estimate |Σ qᵢ rᵢ| already tracks μ closely;
  - "what if we knew the charges?" models that show how much of μ the charges explain;
- extensions such as a projected quantum kernel or a small real-QPU run.

Keep exploratory work in its own clearly named notebooks, label its results "exploratory"
with the inputs used, and record what was tried and what was learned (including dead ends)
in `docs/DECISIONS.md`. Exploration may use cross-validation freely; it works on the
training and development sets, and test sets are scored by `scripts/final_eval.py` (rule 2).

## Conventions

- Pass `numpy.random.Generator` objects explicitly; no global random state.
- Type hints; docstrings state units (Å for coordinates, debye for μ).
- Results are long-format CSV, grouped in `results/` by question (`results/README.md`). Every run
  records config hash, git hash, seed, and package versions.
- Small, tested functions. Any change to splits or descriptors needs passing tests.
