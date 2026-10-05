# Hackathon brief — Prompt 07/08 · QML · Advanced

Source: `prompts/hackathon/07_qml_molecular_properties/PROMPT.md` (Qiskit Fall Fest 2026,
University of Ottawa). Slide image: `docs/prompt_slide.png`.

## Slide

**Learn molecular dipoles from fewer labels.**
How much reference data does a model need to predict molecular polarity?

What to build:
- Use a small QM9 subset to predict dipole-moment magnitude.
- Compare a quantum regressor with classical models at several data sizes.
- Test familiar and unseen formulas; check that rotated inputs agree.

Pipeline: composition + geometry → quantum regression model → dipole magnitude in debye.

**SHOW:** prediction error versus label count, generalization, and quantum cost.

## Presenter notes

- Explain dipole magnitude as a measure of molecular charge separation.
- QM9 targets and geometries are computed reference data, not experimental measurements.
- No new electronic-structure calculations are required.
- QM9 is a public dataset of 133,885 small organic molecules containing C, H, O, N and F,
  with up to nine non-hydrogen atoms. It supplies atomic identities, optimized 3D coordinates
  and calculated molecular properties.
- Here the label is dipole-moment magnitude in debye. The quantum component is the learning
  model built by the team.
- Suggested introduction: use existing molecules with known calculated dipoles to train a
  predictor. Ask how many labeled molecules it needs, and whether it works on molecular
  formulas absent from training.

## Likely questions (organizer answers)

- **Must we use the whole dataset?** No. Use a documented, manageable subset with enough
  formula diversity.
- **What does "fewer labels" mean?** Training examples with known target values. Compare
  error at at least three nested sample sizes, with at least three seeds.
- **Is formula alone enough?** The core asks for composition and geometry. Composition-only
  is a useful comparison.
- **Why rotate the molecule?** Its dipole vector rotates, but its magnitude should stay the
  same.
- **Must we beat classical ML?** No. Fair comparisons and an honest conclusion are required.
- **Can we compute new labels?** Not required. Use the supplied reference values.

## Data handling requirements

- Use molecule identifiers and documented exclusions.
- Reserve unseen formulas, and keep unseen molecules separate from represented formulas.
- Keep familiar test formulas represented at every nested training size.
- Fit transformations using only the allowed training data.

## Physical requirements

- A physically suitable magnitude prediction is invariant to atom relabeling, translation and
  rotation. **Do all three checks**; the slide highlights rotation as an accessible example.
- Optimized geometry is an assumed available input, with associated provenance and cost.

## Full brief requirements

- Baselines: mean, RBF and tree models.
- Matched compressed and uncompressed inputs.
- At least three nested training sizes and three seeds.
- MAE and RMSE in debye.
- One representation ablation.
- Finite-shot and noisy inference.
- Final test results must not select settings.
- Map any bounded circuit expectation back to physical units.

This is an advanced direction. A small subset is encouraged, but the reveal slide does not
silently remove requirements from the full brief.

## Data source

QM9: Ramakrishnan et al., *Scientific Data* 1, 140022 (2014), with GDB-17 provenance from
Ruddigkeit et al. (2012). https://quantum-machine.org/datasets/
