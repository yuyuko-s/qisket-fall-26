# Data

## Source

QM9: Ramakrishnan, Dral, Rupp, von Lilienfeld, *Scientific Data* 1, 140022 (2014), figshare
collection 978904 ("Quantum chemistry structures and properties of 134 kilo molecules").
Molecules from GDB-17 (Ruddigkeit et al. 2012). Geometries and properties at
B3LYP/6-31G(2df,p). These are computed reference values, not experimental measurements.

Downloaded 2026-10-04 by `qm9dipole.data.fetch_all()` (run from `notebooks/00_setup_and_data.ipynb`).

## Raw files (`data/raw/`, gitignored)

Each download is verified against the size and MD5 that figshare publishes before it is kept.

| File | figshare file id | Bytes | MD5 (figshare, verified) | SHA-256 |
|---|---|---|---|---|
| `dsgdb9nsd.xyz.tar.bz2` | 3195389 | 86,144,227 | `ad1ebd51ee7f5b3a6e32e974e5d54012` | `3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c` |
| `uncharacterized.txt` | 3195404 | 486,752 | `a361887bacb427b8a0ce7903d92a53b4` | `3aa5115d540b356de94791d4a74c3bf1ed91c469ecf52a4f5d7cc0506fe02e24` |
| `readme.txt` | 3195392 | 5,198 | `c6581a03f673746528c57acfc6b79679` | `0ee83fc21faa9527ebe89fad0ce8f2bd2da1cbd35a3c7edc199f4ad21d31d97a` |

Observations from M0:
- The tarball holds 133,885 members, with no directory prefix (`dsgdb9nsd_000001.xyz` …
  `dsgdb9nsd_133885.xyz`). It is read in place and never extracted (DECISIONS.md, 2026-10-04).
- The readme confirms the property order in PLAN §3.2: μ is property 6, in debye.
- Line 2 starts `gdb 1` with a space, followed by tab-separated values, so split it on any
  whitespace.
- Molecule 1 (methane) has μ = 0, as expected from its tetrahedral symmetry.

## Parsed table (M1)

`data/processed/qm9.parquet` (gitignored), built by `notebooks/01_parse_and_splits.ipynb`:
- 133,885 molecules parsed, **0 parse failures**, 621 distinct formulas;
- every field of every record is kept, with these roles:

  | Role | Columns |
  |---|---|
  | Headline inputs | `Z` (atomic numbers), `R` (Å; stored flattened, restored to (n, 3) by `load_qm9_table`) |
  | Target | `mu` (debye) |
  | Functions of Z and R (headline-legal) | `A`, `B`, `C` (rotational constants, GHz): equal to 505.379 GHz·amu·Å² / I for the moments of inertia of the geometry (verified to 1.2e-5, `explore_01`) |
  | Exploration only, reported separately (docs/EVALUATION_RULES.md rule 1) | `q` (Mulliken charges, e); `alpha`, `homo`, `lumo`, `gap`, `r2`, `zpve`, `U0`, `U`, `H`, `G`, `Cv` (units in `readme.txt`); `freqs` (harmonic frequencies, cm⁻¹, as listed in the file) |
  | Bookkeeping | `id`, `formula`, `n_atoms`, `n_heavy`, `smiles` and `inchi` (relaxed geometry), `smiles_gdb` and `inchi_gdb` (GDB-17 input) |

- 176 coordinates use the Fortran `*^` exponent (first: molecule 212), and all of them parse.

Spot checks: methane μ = 0 D, ammonia 1.6256 D, water 1.8511 D (experiment ≈ 1.85 D).

## Exclusions (applied in M1)

Rules are applied in order. "Removed" counts only IDs not already removed by an earlier rule.

| Rule | IDs | Listed | Removed | Reason |
|---|---|---|---|---|
| `uncharacterized.txt` | listed in file | 3,054 | 3,054 | The relaxed B3LYP geometry corresponds to a different SMILES than the GDB-17 input, so the structure–label pairing is suspect |
| readme: difficult to converge | 21725, 87037, 59827, 117523, 128113, 129053, 129152, 129158, 130535, 6620, 59818 | 11 | 8 | 6620 and 59818 converged to saddle points (not true minima); six converged only at a low threshold; three (21725, 87037, 117523) are already in `uncharacterized.txt` |
| parse failures | — | 0 | 0 | — |
| geometric duplicate (extra copy), X2 | same formula and Coulomb spectrum to 0.01 (identical geometry and \|μ\|); keep the smallest ID | 140 | 140 | The same molecule listed twice (133 groups). Dropping the copies before splitting means no molecule can sit in two sets or two CV folds. Found by geometry, not InChI (InChI merges tautomers). The IDs are in `splits/meta.json` (`excluded_duplicates`) |
| **Total** | | | **3,202** | **130,683 molecules kept**, 616 formulas (5 formulas existed only among excluded molecules) |

## Working pool (M1 as redesigned in exploration X2)

X1 kept at most 25 molecules per formula (8,438 molecules), as PLAN §3.5 asked. **X2 removed the
cap** (DECISIONS.md, 2026-10-06): QM9 is very unevenly spread over formulas (the 20 largest of 616
hold 53% of the molecules; the median formula has 12), so the cap discarded 94% of the data and
changed its distribution. Every kept molecule is now assigned to exactly one set below.

## Splits (exploration X2 design, `src/qm9dipole/splits.py`)

Config: `configs/splits.yaml` (config_hash `b6dc147c9647`). Files are in `splits/`; built by
`notebooks/01_parse_and_splits.ipynb`.

| Set | File | Molecules | Formulas | Use |
|---|---|---|---|---|
| Unseen-formula test | `test_unseen.json` | 16,818 | 93 (15%, stratified by heavy-atom count; the same formulas as X1) | final evaluation only (`scripts/final_eval.py`) |
| Familiar test | `test_familiar.json` | 5,484 | 257 | final evaluation only |
| Development | `dev.json` | 5,000 | 257 | learning curves, model comparison, diagnostics |
| Development holdout (unseen formulas) | `dev_unseen.json` | 4,183 | 26 | new-formula error during development |
| Training pool P | `pool.json` | 99,198 | 497 | training |
| Quantum unseen subset | `test_unseen_q.json` | 370 | 93 (≤ 5 each) | quantum inference (costs n_test × N circuits) |
| Quantum familiar subset | `test_familiar_q.json` | 80 | 20 anchored formulas × 4 | quantum inference |

- Whole formulas form both unseen holdouts; the familiar test and development sets are random
  molecules of the remaining formulas, sampled so that every formula keeps molecules in P.
- **Training sets** S_(s,N), N = 100, 300, 1000, 3000, 10000, 30000 and 99,198 (the whole pool),
  seeds 0, 1, 2: `train_order_s{seed}.json` lists P in fill order, and S_(s,N) is its first N IDs,
  so the sets are nested by construction. The 20 anchored familiar formulas come first, so they are
  in every set. N = 100, 300, 1000 are the quantum-comparable (Track B) sizes.
- **Anchor skew:** the anchored formulas make up 15% of P but 27–40% of S_(s,100), 21–22% of
  S_(s,300) and 17–18% of S_(s,1000) (X1: 9% of P, 43–50% of S_(s,100)).
- Every split invariant (disjoint sets, held-out formulas absent from training, anchors, nesting,
  exact sizes, unique IDs, no excluded ID, complete holdouts, every kept molecule in exactly one
  set) is checked at build time and by `tests/test_splits.py`, each with a negative control.

## Data quality (exploration X2, `notebooks/explore_01_eda.ipynb`)

Checked on the training pool P (99,198 molecules).

| Check | Result |
|---|---|
| missing values, array lengths, neutral charges (\|Σq\| ≤ 6e-6 e), μ ≥ 0 | all pass |
| gap = lumo − homo; H − U = RT; U0 < U; G < H; A ≥ B ≥ C | all pass (gap to the file's 1e-4 Ha rounding) |
| imaginary (negative) frequencies | none (5 in QM9, all excluded: 6620, 59818, 87037, 117523, 129158) |
| frequency lists printed twice (2 × (3n − 6) values, identical halves) | 255 in P (433 in QM9); cleaning keeps the first copy |
| linear molecules (3n − 5 modes; QM9 stores A = 0 for the infinite constant) | 5, all in P |
| geometric duplicates | none left (removed before splitting) |
| same InChI | 1,824 groups in P, all different geometries (tautomers, \|μ\| up to 8.5 D apart): standard InChI is not an identity test here |
| stored coordinate frames | arbitrary: centroids up to 6.7 Å from the origin, 1.6% aligned with the principal axes |
| target | mean 2.71 D, median 2.52 D; 41 exact zeros; 223 molecules (0.22%) at ≥ 9 D, up to 29.6 D (zwitterions, e.g. ID 123126 `[NH3]CCCCCC(=O)[O]`). Kept: valid computed values, but they weigh heavily on RMSE |
