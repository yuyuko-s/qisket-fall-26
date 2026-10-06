"""QM9 raw data: download and integrity checks (M0), parsing and exclusions (M1).

Source: Ramakrishnan et al., Scientific Data 1, 140022 (2014), figshare collection 978904.
The 133,885-molecule tarball is streamed with `tarfile` rather than extracted. It is parsed
once into data/processed/qm9.parquet, which every later milestone loads instead of the raw
files.

The table keeps every field of every record. Headline models use only HEADLINE_INPUTS
(atomic numbers and coordinates) to predict TARGET; the Mulliken charges and the other
properties are kept for clearly labeled exploration (CLAUDE.md, "Exploration").

Units: coordinates R in Å, dipole magnitude mu in debye, charges q in e; other properties as
in readme.txt.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qm9dipole import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
QM9_PARQUET = PROCESSED_DIR / "qm9.parquet"

_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class RawFile:
    """One file of the figshare collection, with the size and MD5 that figshare publishes."""

    name: str
    figshare_id: int
    size: int  # bytes
    md5: str

    @property
    def url(self) -> str:
        return f"https://ndownloader.figshare.com/files/{self.figshare_id}"


#: The three files PLAN §3.1 needs. Sizes and MD5s are from api.figshare.com (2026-10-04).
QM9_FILES: tuple[RawFile, ...] = (
    RawFile("dsgdb9nsd.xyz.tar.bz2", 3195389, 86_144_227, "ad1ebd51ee7f5b3a6e32e974e5d54012"),
    RawFile("uncharacterized.txt", 3195404, 486_752, "a361887bacb427b8a0ce7903d92a53b4"),
    RawFile("readme.txt", 3195392, 5_198, "c6581a03f673746528c57acfc6b79679"),
)

TARBALL = RAW_DIR / QM9_FILES[0].name


class ChecksumError(RuntimeError):
    """A file's size or MD5 does not match what the source publishes."""


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hex digest of a file, read in chunks so large files never sit in memory."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, raw: RawFile) -> None:
    """Raise ChecksumError unless the file's size and MD5 match the published values."""
    size = path.stat().st_size
    if size != raw.size:
        raise ChecksumError(f"{path.name}: size {size} != expected {raw.size}")
    md5 = file_digest(path, "md5")
    if md5 != raw.md5:
        raise ChecksumError(f"{path.name}: md5 {md5} != expected {raw.md5}")


def fetch(raw: RawFile, dest_dir: Path = RAW_DIR, url: str | None = None) -> Path:
    """Download one file into dest_dir and verify it. Returns the local path.

    The download goes to "<name>.part" and is renamed only after it verifies, so a file
    with the final name was complete and correct when it was written. `url` overrides
    the figshare URL (tests use file:// URLs).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / raw.name

    if dest.exists():
        # Re-verify on every run (~0.3 s for the tarball): a file that verified when written
        # can still change later (file-sync conflicts, a copied-in file, a manual edit). On a
        # mismatch, fail loudly instead of re-downloading, so the cause gets noticed.
        try:
            verify(dest, raw)
        except ChecksumError as e:
            raise ChecksumError(
                f"{e}. The existing file {dest} is corrupt or was modified; "
                "inspect it, delete it, and re-run to download a fresh copy."
            ) from e
        return dest

    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url or raw.url, headers={"User-Agent": "qm9dipole/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out, _CHUNK)
        verify(part, raw)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


def fetch_all(dest_dir: Path = RAW_DIR) -> list[dict[str, object]]:
    """Fetch every QM9 file and return manifest rows for docs/DATA.md."""
    rows = []
    for raw in QM9_FILES:
        path = fetch(raw, dest_dir)
        rows.append({
            "file": raw.name,
            "figshare_id": raw.figshare_id,
            "bytes": path.stat().st_size,
            "md5": raw.md5,
            "sha256": file_digest(path, "sha256"),
        })
    return rows


# --------------------------------------------------------------------------------------
# Parsing (M1). File format: data/raw/readme.txt, "Format"; PLAN §3.2.
# --------------------------------------------------------------------------------------

ELEMENT_Z: dict[str, int] = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

#: Formula element order (PLAN §3.3): C, H, then the rest alphabetically.
FORMULA_ORDER: tuple[str, ...] = ("C", "H", "F", "N", "O")

#: The 15 properties on line 2 after "gdb" and the index, in readme order and with readme
#: names: rotational constants A, B, C (GHz); dipole mu (D); polarizability alpha (bohr³);
#: homo, lumo, gap (Hartree); r2 (bohr²); zpve, U0, U, H, G (Hartree); Cv (cal/mol K).
PROPERTY_NAMES: tuple[str, ...] = (
    "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv",
)
_N_PROPERTY_TOKENS = 2 + len(PROPERTY_NAMES)

#: What the headline models may use (the brief: composition + geometry) and what they predict.
HEADLINE_INPUTS: tuple[str, ...] = ("Z", "R")
TARGET = "mu"

#: Every column of the parsed table, i.e. every field of a QM9 record plus bookkeeping.
TABLE_COLUMNS: tuple[str, ...] = (
    "id", "formula", "n_atoms", "n_heavy", "Z", "R", "q", *PROPERTY_NAMES,
    "freqs", "smiles", "smiles_gdb", "inchi", "inchi_gdb",
)


def _float(token: str) -> float:
    """Parse a QM9 float. A few coordinates use a Fortran exponent, e.g. "2.1997*^-6"."""
    return float(token.replace("*^", "e"))


def hill_formula(elements: list[str]) -> str:
    """Formula string such as "C2H6O" (count 1 omitted), in FORMULA_ORDER."""
    counts = Counter(elements)
    unknown = set(counts) - set(FORMULA_ORDER)
    if unknown:
        raise ValueError(f"unexpected elements {sorted(unknown)}")
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in FORMULA_ORDER if counts[e])


def parse_xyz(text: str) -> dict:
    """Parse one QM9 .xyz record, keeping every field.

    Returns TABLE_COLUMNS: id, formula, n_atoms, n_heavy, Z (int8 array), R (float array
    (n_atoms, 3), Å), q (Mulliken charge per atom, e), the 15 PROPERTY_NAMES (mu in debye),
    freqs (harmonic vibrational frequencies, cm⁻¹, exactly as listed in the file), smiles
    and inchi (of the relaxed geometry) and smiles_gdb and inchi_gdb (of the GDB-17 input).
    Values are kept as found; cleaning (e.g. of doubled frequency lists) is a later step.

    Headline models use only HEADLINE_INPUTS. q, freqs and the other properties come from
    the same DFT calculation as mu, so they are for labeled exploration (CLAUDE.md, rule 1).
    """
    lines = text.splitlines()
    n_atoms = int(lines[0])
    if len(lines) < n_atoms + 5:
        raise ValueError(f"expected >= {n_atoms + 5} lines, got {len(lines)}")

    props = lines[1].split()  # "gdb 1" is space-separated, the rest tab-separated
    if props[0] != "gdb" or len(props) != _N_PROPERTY_TOKENS:
        raise ValueError(f"bad property line: {lines[1][:60]!r}")

    atoms = [line.split() for line in lines[2 : 2 + n_atoms]]
    if any(len(a) != 5 for a in atoms):
        raise ValueError("atom line without 5 columns (element, x, y, z, charge)")
    elements = [a[0] for a in atoms]
    Z = np.array([ELEMENT_Z[e] for e in elements], dtype=np.int8)
    R = np.array([[_float(x) for x in a[1:4]] for a in atoms], dtype=np.float64)
    q = np.array([_float(a[4]) for a in atoms], dtype=np.float64)

    freqs = np.array([_float(tok) for tok in lines[n_atoms + 2].split()], dtype=np.float64)
    smiles_gdb, smiles_relaxed = lines[n_atoms + 3].split()
    inchi_gdb, inchi_relaxed = lines[n_atoms + 4].split()
    return {
        "id": int(props[1]),
        "formula": hill_formula(elements),
        "n_atoms": n_atoms,
        "n_heavy": int((Z > 1).sum()),
        "Z": Z,
        "R": R,
        "q": q,
        **{name: _float(tok) for name, tok in zip(PROPERTY_NAMES, props[2:])},
        "freqs": freqs,
        "smiles": smiles_relaxed,
        "smiles_gdb": smiles_gdb,
        "inchi": inchi_relaxed,
        "inchi_gdb": inchi_gdb,
    }


def iter_tarball(path: Path = TARBALL) -> Iterator[tuple[str, str]]:
    """Yield (member name, text) for every .xyz file, streaming; nothing touches the disk."""
    with tarfile.open(path, "r:bz2") as tar:
        for member in tar:
            if member.isfile():
                yield member.name, tar.extractfile(member).read().decode()


def build_qm9_table(path: Path = TARBALL) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Parse every molecule. Returns (table sorted by id, [(member name, error), ...])."""
    rows, failures = [], []
    for name, text in iter_tarball(path):
        try:
            row = parse_xyz(text)
            file_id = int(name.removesuffix(".xyz").split("_")[-1])
            if row["id"] != file_id:
                raise ValueError(f"index {row['id']} != file number {file_id}")
            rows.append(row)
        except Exception as e:  # logged and reported, never silently dropped
            failures.append((name, f"{type(e).__name__}: {e}"))
    table = pd.DataFrame(rows).sort_values("id", ignore_index=True)
    return table, failures


def save_qm9_table(table: pd.DataFrame, path: Path = QM9_PARQUET) -> None:
    """Write the table to parquet. R is flattened to 3*n_atoms floats (parquet lists are 1-D)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    out["R"] = [r.ravel() for r in out["R"]]
    out.to_parquet(path, index=False)


def load_qm9_table(path: Path = QM9_PARQUET) -> pd.DataFrame:
    """Load the parsed table: Z as int8 arrays, R as (n_atoms, 3) arrays in Å, q in e,
    freqs in cm⁻¹."""
    table = pd.read_parquet(path)
    table["Z"] = [np.asarray(z, dtype=np.int8) for z in table["Z"]]
    table["R"] = [np.asarray(r, dtype=np.float64).reshape(-1, 3) for r in table["R"]]
    for col in ("q", "freqs"):
        table[col] = [np.asarray(v, dtype=np.float64) for v in table[col]]
    return table


def qm9_table_is_current(path: Path = QM9_PARQUET) -> bool:
    """True if the parquet file exists and has exactly the columns the current parser writes.

    A table built by an older parser (e.g. before the frequencies and InChIs were kept) is
    stale and must be rebuilt.
    """
    if not path.exists():
        return False
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names) == set(TABLE_COLUMNS)


# --------------------------------------------------------------------------------------
# Exclusions (M1). Counts and reasons are documented in docs/DATA.md.
# --------------------------------------------------------------------------------------

#: readme.txt "Notes": the 11 molecules whose geometries were difficult to converge.
#: 6620 and 59818 converged to saddle points; six others only at a low threshold.
README_FLAGGED_IDS: frozenset[int] = frozenset(
    {21725, 87037, 59827, 117523, 128113, 129053, 129152, 129158, 130535, 6620, 59818}
)


def read_uncharacterized(path: Path = RAW_DIR / "uncharacterized.txt") -> frozenset[int]:
    """IDs of the 3,054 molecules that failed the authors' geometry consistency check."""
    ids = {
        int(tok[0])
        for line in path.read_text().splitlines()
        if (tok := line.split()) and tok[0].isdigit()
    }
    return frozenset(ids)


def exclusion_rules(include_readme_flagged: bool = True) -> dict[str, frozenset[int]]:
    """Exclusion rules in the order they are applied: {rule name: IDs}."""
    rules = {"uncharacterized.txt": read_uncharacterized()}
    if include_readme_flagged:
        rules["readme: difficult to converge"] = README_FLAGGED_IDS
    return rules


def apply_exclusions(
    table: pd.DataFrame, rules: dict[str, frozenset[int]]
) -> tuple[pd.DataFrame, pd.DataFrame, frozenset[int]]:
    """Drop excluded molecules.

    Returns (kept table, report with one row per rule, set of all excluded IDs). The report
    counts IDs listed by each rule and how many it newly removed, since rules overlap.
    """
    excluded: set[int] = set()
    report = []
    for rule, ids in rules.items():
        new = (ids & set(table["id"])) - excluded
        report.append({"rule": rule, "listed": len(ids), "newly_removed": len(new)})
        excluded |= new
    kept = table[~table["id"].isin(excluded)].reset_index(drop=True)
    return kept, pd.DataFrame(report), frozenset(excluded)
