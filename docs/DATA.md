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
  | Exploration only, reported separately (CLAUDE.md rule 1) | `q` (Mulliken charges, e); `alpha`, `homo`, `lumo`, `gap`, `r2`, `zpve`, `U0`, `U`, `H`, `G`, `Cv` (units in `readme.txt`); `freqs` (harmonic frequencies, cm⁻¹, as listed in the file) |
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
| **Total** | | | **3,062** | **130,823 molecules kept**, 616 formulas (5 formulas existed only among excluded molecules) |

## Working pool (M1, PLAN §3.5)

At most 25 molecules per formula, sampled uniformly within each formula with the split seed (2026).
The IDs are in `splits/pool.json`.

- **8,438 molecules, 616 formulas.** Median 12 molecules per formula; 240 formulas are
  capped at 25, and 144 have fewer than 3 (these cannot be familiar formulas).
- Heavy-atom counts:

  | heavy atoms | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|
  | formulas | 3 | 5 | 7 | 20 | 31 | 64 | 92 | 162 | 232 |
  | molecules | 3 | 5 | 9 | 31 | 127 | 521 | 1,265 | 2,371 | 4,106 |

- μ: median about 2.5 D. 422 molecules are below 0.25 D: near-symmetric molecules, 27 of them
  exactly 0, such as methane, acetylene and ethane. The tail is long: 34 molecules are at or
  above 9 D, up to 29.6 D. The largest dipoles are **zwitterions** (for example ID 123126,
  `[NH3]CCCCCC(=O)[O]`, 29.6 D), with a full charge separated across the molecule. They are real
  computed values and are kept, but they weigh heavily on RMSE, which is worth remembering when
  comparing MAE and RMSE.
- Figure: `figures/m1_pool_overview.png`.

## Splits (M1, PLAN §4)

Config: `configs/splits.yaml` (config_hash `da596c9e207c`). Files are in `splits/`.

| Set | Size | Notes |
|---|---|---|
| Unseen-formula test | 370 molecules | 93 of 616 formulas (15%, stratified by heavy-atom count), ≤5 molecules each |
| Familiar-formula test | 80 molecules | 40 formulas × 2 |
| Training pool P | 7,131 molecules | pool minus unseen formulas and the familiar test set |
| Training sets S_(s,N) | N = 100, 300, 1000 | seeds 0, 1, 2; nested; each contains one anchor per familiar formula |

Anchor skew: familiar formulas make up 9% of P but 43–50% of S_(s,100), 20–23% of S_(s,300),
and 12–13% of S_(s,1000). See notebook 01 §5.

## Data quality (exploration X1, `notebooks/explore_01_eda.ipynb`)

Checked on the training pool P (7,131 molecules); the counts in brackets are for all 133,885 records.

| Check | Result |
|---|---|
| missing values, array lengths, neutral charges (\|Σq\| ≤ 6e-6 e), μ ≥ 0 | all pass |
| gap = lumo − homo; H − U = RT; U0 < U; G < H; A ≥ B ≥ C | all pass (gap to the file's 1e-4 Ha rounding) |
| imaginary (negative) frequencies | none in P (5 in QM9, all already excluded: 6620, 59818, 87037, 117523, 129158) |
| frequency lists printed twice (2 × (3n − 6) values, identical halves) | 25 in P (433); cleaning keeps the first copy |
| linear molecules (3n − 5 modes; QM9 stores A = 0 for the infinite constant) | 5, all in P |
| geometric duplicates (same formula and Coulomb spectrum to 0.01; \|μ\| equal within 0.003 D) | 12 pairs in P (133 groups, 273 molecules among the kept 130,823). No training set holds both twins and none shares a molecule with a test set (unit-tested), so the splits are unchanged; pool-wide analyses keep one twin per pair |
| same InChI | 171 groups in P, but 159 are tautomers (different molecules, \|μ\| up to 6.3 D apart): standard InChI is not an identity test here |
| stored coordinate frames | arbitrary: centroids up to 6.8 Å from the origin, 6.9% aligned with the principal axes |

