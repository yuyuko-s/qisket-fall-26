"""Run the quantum models' circuits on an IBM quantum computer (or locally on its noise model).

    python scripts/hardware_run.py --dry-run                     # local, FakeQuebec noise, development molecules
    python scripts/hardware_run.py                               # local, FakeQuebec noise, the quantum test molecules
    python scripts/hardware_run.py --backend ibm_quebec          # plan only: circuits, shots, QPU time; submits nothing
    python scripts/hardware_run.py --backend ibm_quebec --submit # submit ONE job (shows the plan, asks to confirm)
    python scripts/hardware_run.py --retrieve <job id>           # fetch a submitted job's results and score them

What runs (`qm9dipole.hardware`): the projected quantum kernel (X, Y, Z bases) and the team's ⟨Z⟩ ridge
(Z basis) end to end — every training molecule of S_(seed, N) and every test molecule measured, readout
refit on the measured features — and a small block of fidelity-kernel entries (with each test molecule's
overlap with itself, for the depolarizing correction). Models use the settings their CV chose in a test
run (default: run 1). The test molecules are those of test run 1's noise section (30 per quantum test
subset), so the device, the FakeFez/FakeQuebec noise models and exact simulation compare directly.

Safety (the team approves every hardware job in advance): a real device is contacted only with --submit (after a typed confirmation of
the plan, or --yes once the team has approved it in advance) or --retrieve. The job is capped at
--max-seconds of QPU time. The IBM account is the one saved locally under --account (default "pinq2");
no token appears in this repository. Runs that score test molecules are logged in results/test_runs.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd

from qm9dipole import REPO_ROOT

from qm9dipole.data import PROCESSED_DIR, load_qm9_table
from qm9dipole.explore import feature_sets, load_catalog, load_features, load_preprocessing
from qm9dipole.final import evaluation_frame, log_run, next_run_number, timed
from qm9dipole.hardware import (FAKE_FOR, PARTS, build_blocks, fit_models, parse, plan, run_local, score,
                                select_test_molecules, settings_from_run, submit)
from qm9dipole.provenance import RESULTS_DIR, config_hash, git_hash, results_file, save_result
from qm9dipole.splits import load_splits

OUT = RESULTS_DIR / "hardware"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", default="FakeQuebec", help="a fake backend (local Aer) or a device name such as ibm_quebec")
    ap.add_argument("--account", default="pinq2", help="saved QiskitRuntimeService account name")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-subset", type=int, default=30)
    ap.add_argument("--kernel-block", default="10x5", help="test x training molecules of fidelity-kernel entries")
    ap.add_argument("--max-seconds", type=int, default=300, help="QPU-time cap of the job (refuses a larger plan)")
    ap.add_argument("--settings", default="results/comparison/test_run01_quantum.csv",
                    help="a test run's quantum-section CSV (relative to the repository): the settings each model chose by CV")
    ap.add_argument("--dry-run", action="store_true", help="development molecules instead of test molecules; not logged")
    ap.add_argument("--submit", action="store_true", help="submit to the real device named by --backend")
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmation (only with the team's approval)")
    ap.add_argument("--no-wait", action="store_true", help="exit after submitting; fetch the result later with --retrieve")
    ap.add_argument("--retrieve", metavar="JOB_ID")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    job_record = None
    if args.retrieve:  # restore the submitted run's arguments
        matches = [p for p in OUT.glob("*_job.json") if json.loads(p.read_text())["job_id"] == args.retrieve]
        if not matches:
            print(f"no record of job {args.retrieve} in {OUT}")
            return 1
        job_record = json.loads(matches[0].read_text())
        for k, v in job_record["args"].items():
            if k not in ("retrieve", "submit", "yes", "no_wait"):
                setattr(args, k, v)
    parts = [p for p in args.parts.split(",") if p]
    kb = tuple(int(v) for v in args.kernel_block.lower().split("x"))
    device = not args.backend.startswith("Fake")
    if args.submit and not device:
        ap.error("--submit needs a real device name for --backend (e.g. ibm_quebec)")
    commit = git_hash(short=True)
    if commit.endswith("-dirty") and not (args.allow_dirty or args.dry_run):
        print("The working tree has uncommitted changes: commit them first (or pass --allow-dirty).")
        return 1

    # --- Data, molecules and models (all local) ---------------------------------------------------
    splits = load_splits()
    features = load_features()
    SETS = feature_sets(load_catalog())
    PRE = load_preprocessing()
    SC_B, TG_B = PRE["scaling"]["track_b"], PRE["target"]["track_b"]
    RED = (PRE["reduction"]["method"], int(PRE["reduction"]["k"]))
    if args.dry_run:  # the development stand-ins of `final_eval.py --dry-run`
        rng = np.random.default_rng(2027)
        subsets = {"dev_q": np.sort(rng.choice(splits.dev, len(splits.test_familiar_q), replace=False)),
                   "dev_unseen_q": np.sort(rng.choice(splits.dev_unseen, len(splits.test_unseen_q), replace=False))}
    else:
        subsets = {"test_familiar_q": splits.test_familiar_q, "test_unseen_q": splits.test_unseen_q}
    test_ids, which = select_test_molecules(subsets, args.per_subset)
    train_ids = splits.train[args.seed][args.n_train]
    table = load_qm9_table()
    XA = evaluation_frame(features, table, test_ids, SETS["all_legal"])
    mu = table.set_index("id")["mu"]
    y = pd.concat([features["mu"].astype(np.float64), mu.loc[np.setdiff1d(XA.index, features.index)]])
    needed = {"qkrr" if p == "kernel" else p for p in parts}  # the kernel block uses the fidelity-kernel model
    settings_csv = REPO_ROOT / args.settings if not Path(args.settings).is_absolute() else Path(args.settings)
    if not settings_csv.exists():  # a path recorded before results/ was grouped (results/README.md)
        settings_csv = results_file(settings_csv.name)
    settings = {k: v for k, v in settings_from_run(pd.read_csv(settings_csv), args.seed, args.n_train).items() if k in needed}
    missing = needed - set(settings)
    if missing:
        print(f"no settings for {sorted(missing)} at seed {args.seed}, N = {args.n_train} in {args.settings}")
        return 1
    with timed(f"refit {sorted(settings)} on S_({args.seed},{args.n_train}) at the settings of {args.settings}"):
        models = fit_models(XA, y, train_ids, settings, args.seed, SC_B, TG_B, RED)
    Xtr, ytr = XA.loc[train_ids].to_numpy(), y.loc[train_ids].to_numpy()
    Xte, yte = XA.loc[test_ids].to_numpy(), y.loc[test_ids].to_numpy()

    # --- Backend and circuits ---------------------------------------------------------------------
    real = None
    if args.submit or args.retrieve:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(name=args.account)
        if args.retrieve:
            target = None  # the PUB layout is all parsing needs
        else:
            real = service.backend(args.backend)
            target = real
    else:
        target = FAKE_FOR.get(args.backend, args.backend)  # plan a device on its calibration snapshot
    with timed(f"build and transpile the circuits ({'logical only' if target is None else getattr(target, 'name', target)})"):
        blocks = build_blocks(models, Xtr, ytr, Xte, parts, backend=target, kernel_block=kb)
    tag = job_record["tag"] if job_record else f"hardware_{args.backend}_{time.strftime('%Y%m%d-%H%M')}" + ("_dry" if args.dry_run else "")
    out_dir = PROCESSED_DIR / "hardware_dry_run" if args.dry_run else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.retrieve:
        p = plan(blocks, args.shots, target)
        total = float(p["qpu_seconds"].sum())
        pd.set_option("display.width", 160)
        print(p.round({"duration_us": 2, "qpu_seconds": 1}).to_string(index=False))
        print(f"total: {int(p['circuits'].sum()):,} circuits × {args.shots:,} shots = {int(p['circuits'].sum()) * args.shots:,} "
              f"shots; estimated QPU time ≈ {total:.0f} s ({total / 60:.1f} min; a rough lower bound, see cost.qpu_seconds)")
        p.to_csv(out_dir / f"{tag}_plan.csv", index=False)
        if device and not args.submit:
            print(f"Plan only (on {FAKE_FOR.get(args.backend, args.backend)}): nothing was submitted. "
                  f"Add --submit to send this to {args.backend}.")
            return 0

    # --- Measure ------------------------------------------------------------------------------------
    if args.submit:
        if total > args.max_seconds:
            print(f"Refused: the plan needs ≈ {total:.0f} s, above --max-seconds {args.max_seconds}.")
            return 1
        prompt = (f"Submit {int(p['circuits'].sum()):,} circuits × {args.shots:,} shots to {args.backend} "
                  f"(≈ {total:.0f} s QPU, capped at {args.max_seconds} s)? Type 'submit' to confirm: ")
        if not args.yes and input(prompt).strip().lower() != "submit":
            print("Not submitted.")
            return 1
        job = submit(blocks, args.shots, real, max(args.max_seconds, 60))
        job_record = {"job_id": job.job_id(), "tag": tag, "backend": args.backend, "submitted": time.strftime("%Y-%m-%d %H:%M"),
                      "commit": commit, "args": {k: v for k, v in vars(args).items() if k not in ("submit", "yes", "retrieve", "no_wait")},
                      "circuits": int(p["circuits"].sum()), "estimated_qpu_seconds": total,
                      "test_ids": test_ids.tolist(), "subset": which.tolist(), "settings": settings}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{tag}_job.json").write_text(json.dumps(job_record, indent=1))
        if args.no_wait:
            print(f"submitted job {job.job_id()} (record: results/hardware/{tag}_job.json). "
                  f"Fetch and score it later: python scripts/hardware_run.py --retrieve {job.job_id()}")
            return 0
        print(f"submitted job {job.job_id()} (record: results/hardware/{tag}_job.json); waiting for the result. "
              f"If interrupted: python scripts/hardware_run.py --retrieve {job.job_id()}")
        result = job.result()
    elif args.retrieve:
        job = service.job(args.retrieve)
        print(f"job {args.retrieve}: {job.status()}")
        result = job.result()
    else:
        with timed(f"run {sum(b.rows for b in blocks):,} circuits locally with the {args.backend} noise model"):
            result = run_local(blocks, args.shots, backend=args.backend, seed=args.seed)

    # --- Score --------------------------------------------------------------------------------------
    measured = parse(result, blocks)
    label = args.backend if device else f"{args.backend} noise model (local Aer)"
    sc, kern = score(models, measured, Xtr, ytr, Xte, yte, which, args.shots, label, kernel_block=kb, seed=args.seed)
    run = 0 if args.dry_run else next_run_number()
    meta = dict(run=run, backend=args.backend, device=device, job_id=job_record["job_id"] if job_record else None,
                shots=args.shots, seed=args.seed, n_train=args.n_train, per_subset=args.per_subset, parts=parts,
                kernel_block=list(kb), settings_from=args.settings, settings=settings, dry_run=args.dry_run,
                label="quantum circuits measured on " + label + "; classical readout refit locally; Z, R only")
    save_result(sc, f"{tag}_scores", out_dir=out_dir, **meta)
    if len(kern):
        save_result(kern, f"{tag}_kernel", out_dir=out_dir, **meta)
    np.savez(out_dir / f"{tag}_features.npz", test_ids=test_ids, subset=which, train_ids=train_ids,
             **{k: v for k, v in measured.items()})
    show = sc[sc["test_subset"] == "both"].set_index(["model", "mode"])[["mae_D", "rmse_D", "r2"]]
    print(show.round(3).to_string())
    if len(kern):
        err = (kern["measured"] - kern["exact"]).abs().mean(), (kern["mitigated"] - kern["exact"]).abs().mean()
        print(f"fidelity-kernel entries: mean |measured − exact| = {err[0]:.3f}, after the depolarizing correction "
              f"{err[1]:.3f}; mean survival {kern['survival'].mean():.3f}")
    if not args.dry_run:
        log_run({"run": run, "started": job_record["submitted"] if job_record else time.strftime("%Y-%m-%d %H:%M"),
                 "finished": time.strftime("%Y-%m-%d %H:%M"), "commit": job_record["commit"] if job_record else commit,
                 "config_hash": config_hash(meta), "sections": f"hardware:{args.backend}", "smoke": False,
                 "note": f"{'device' if device else 'local noise model'} run of {', '.join(parts)} on {args.backend}: "
                         f"S_({args.seed},{args.n_train}), {args.per_subset} molecules per quantum test subset, "
                         f"{args.shots:,} shots; settings from {args.settings.replace(chr(92), '/').split('/')[-1]}"})
        print(f"logged run {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
