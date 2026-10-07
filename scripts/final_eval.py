"""Test-set evaluation of the quantum and classical models (M7).

    python scripts/final_eval.py                      # a test run with configs/test_eval.yaml
    python scripts/final_eval.py --dry-run            # the same pipeline, scored on the development sets only
    python scripts/final_eval.py --dry-run --smoke    # one seed, N = 100, small subsets: checks the script in minutes
    python scripts/final_eval.py --sections quantum,shots   # only some sections (quantum, track_a, shots, noise)

Every model is trained and tuned exactly as in the development notebooks (explore_04 for the
classical tracks, explore_06/07 for the quantum models): on the saved training sets, with
cross-validation inside each training set only. Each fitted model then predicts the test sets
and the development sets; the development scores reproduce those notebooks and are used, as
before, to name the best Track A model at each size.

The team may iterate after a test run (CLAUDE.md rule 2). Each run gets the next run number,
writes results/test_runNN_<section>.csv (with .meta.json provenance) and appends a row to
results/test_runs.csv with the commit and the config's `note` (what changed since the last run).
Runs from a working tree with uncommitted code changes are refused unless --allow-dirty.
Quantum models are simulated locally; nothing is sent to IBM hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from qm9dipole.data import PROCESSED_DIR, load_qm9_table
from qm9dipole.evaluate import dev_curve
from qm9dipole.explore import feature_sets, load_atoms, load_catalog, load_config, load_features, load_preprocessing
from qm9dipole.final import (CONFIG_PATH, RUN_LOG, evaluation_frame, load_config as load_eval_config, log_run,
                             next_run_number, noisy_feature_scores, noisy_fidelity_scores, shot_scores, timed)
from qm9dipole.models import classical
from qm9dipole.models.fitters import ChargeFitter, MeanFitter, TabularFitter, XGBFitter
from qm9dipole.models.quantum_kernel import FidelityKernel, KernelRidgeFitter, ProjectedKernel, quantum_ridge_build
from qm9dipole.provenance import RESULTS_DIR, config_hash, git_hash, save_result
from qm9dipole.splits import load_splits

KERNELS = {"qkrr": ("fidelity", "zz", 2), "qkrr_ry": ("fidelity", "ry", 2), "qkrr_ryrz": ("fidelity", "ry_rz", 2),
           "qkrr_product": ("fidelity", "ry", 1), "pqk": ("projected", "zz", 2)}
SECTIONS = ("quantum", "track_a", "shots", "noise")


def kernel_for(model: str, gammas=None, gammas_p=None):
    kind, enc, reps = KERNELS[model]
    if kind == "projected":
        return ProjectedKernel(enc, reps, gammas=gammas, gammas_p=gammas_p)
    return FidelityKernel(enc, reps, gammas=gammas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--dry-run", action="store_true", help="score the development sets only (no test set is read)")
    ap.add_argument("--smoke", action="store_true", help="one seed, N = 100, small subsets")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    sections = [s for s in args.sections.split(",") if s]
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        ap.error(f"unknown sections {unknown}")

    cfg = load_eval_config(args.config)
    commit = git_hash(short=True)
    if commit.endswith("-dirty") and not (args.allow_dirty or args.dry_run):
        print("The working tree has uncommitted changes: commit them first (or pass --allow-dirty).")
        return 1
    started = time.strftime("%Y-%m-%d %H:%M")
    run = 0 if args.dry_run else next_run_number()
    tag = "dryrun" if args.dry_run else f"test_run{run:02d}"
    out_dir = PROCESSED_DIR / "final_eval_dry_run" if args.dry_run else RESULTS_DIR
    cache = PROCESSED_DIR / "cache" / ("final_eval_dry" if args.dry_run else "final_eval") / commit
    print(f"{tag}: commit {commit}, config {config_hash(cfg)}, sections {sections}" + (" (smoke)" if args.smoke else ""))
    print(f"note: {cfg['note']}")

    # --- Data -------------------------------------------------------------------------------
    splits = load_splits()
    features = load_features()
    catalog = load_catalog()
    SETS = feature_sets(catalog)
    PRE = load_preprocessing()
    SC_A, SC_B = PRE["scaling"]["track_a"], PRE["scaling"]["track_b"]
    TG_A, TG_B = PRE["target"]["track_a"], PRE["target"]["track_b"]
    rng = np.random.default_rng(2027)
    eval_ids = {"dev": splits.dev, "dev_unseen": splits.dev_unseen}
    if args.dry_run:  # stand-ins for the quantum test subsets, drawn from the development sets
        q_ids = {"dev_q": np.sort(rng.choice(splits.dev, len(splits.test_familiar_q), replace=False)),
                 "dev_unseen_q": np.sort(rng.choice(splits.dev_unseen, len(splits.test_unseen_q), replace=False))}
    else:
        eval_ids = {"test_familiar": splits.test_familiar, "test_unseen": splits.test_unseen, **eval_ids}
        q_ids = {"test_familiar_q": splits.test_familiar_q, "test_unseen_q": splits.test_unseen_q}
    seeds = [0] if args.smoke else list(cfg["seeds"])

    table = load_qm9_table()
    every_eval = np.unique(np.concatenate([*eval_ids.values(), *q_ids.values()]))
    with timed(f"features for {len(np.setdiff1d(every_eval, features.index)):,} molecules outside the development universe"):
        XA = evaluation_frame(features, table, every_eval, SETS["all_legal"])
    frames = {"all_legal": XA, "composition": XA[SETS["composition"]]}
    mu = table.set_index("id")["mu"]
    y = pd.concat([features["mu"].astype(np.float64), mu.loc[np.setdiff1d(XA.index, features.index)]])
    meta = dict(run=run, note=cfg["note"], config_hash=config_hash(cfg), dry_run=args.dry_run, smoke=args.smoke,
                eval_sets={k: len(v) for k, v in {**eval_ids, **q_ids}.items()})

    def save(df, section, **extra):
        df = df.assign(run=run)
        save_result(df, f"{tag}_{section}", out_dir=out_dir, **meta, **extra,
                    label=f"{'dry run on development sets' if args.dry_run else 'test-set evaluation'}; Z, R only; "
                          "trained and tuned on the saved training sets only")
        return df

    qcfg = cfg["quantum"]
    q_sizes = [100] if args.smoke else list(qcfg["sizes"])
    inputs = {name: (frames[fs], tuple(red) if red else None) for name, (fs, red) in qcfg["inputs"].items()}

    # --- Section 1: quantum models and their classical comparisons (N <= 1000) -----------------
    quantum = None
    if "quantum" in sections or "shots" in sections or "noise" in sections:
        def make_quantum(seed, n):
            out = {}
            for model, ins in qcfg["models"].items():
                for name in ins:
                    if model == "mean":
                        out["mean"] = MeanFitter()
                        continue
                    frame, red = inputs[name]
                    key = f"{model}|{name}"
                    if model in KERNELS:
                        out[key] = KernelRidgeFitter(frame, kernel_for(model), seed, scaling=SC_B, target=TG_B,
                                                     reduction=red, cv_max_n=max(q_sizes))
                    elif model == "qridge":
                        est, grid = quantum_ridge_build(seed, SC_B, TG_B, red, frame.shape[1])
                        out[key] = TabularFitter(frame, est, grid, seed, cv_max_n=max(q_sizes))
                    else:
                        est, grid = classical.build(model, seed, scaling=SC_B, target=TG_B, reduction=red,
                                                    n_features=frame.shape[1])
                        out[key] = TabularFitter(frame, est, grid, seed, cv_max_n=max(q_sizes))
            return out

        sets = {s: {n: splits.train[s][n] for n in q_sizes} for s in seeds}
        with timed("quantum section"):
            quantum, _ = dev_curve(make_quantum, sets, eval_ids, y, cache_dir=cache, progress=True)
        quantum[["model", "inputs"]] = quantum["model"].str.split("|", expand=True).reindex(columns=[0, 1])
        quantum["inputs"] = quantum["inputs"].fillna("—")
        quantum["cv_mae_D"] = quantum["details"].map(lambda d: json.loads(d).get("tuning_mae_D", np.nan))
        if "quantum" in sections:
            quantum = save(quantum, "quantum", scaling=SC_B, target=TG_B, sizes=q_sizes, seeds=seeds)

    def chosen(model, seed, n):
        row = quantum[(quantum["model"] == model) & (quantum["inputs"] == "pls10") & (quantum["seed"] == seed)
                      & (quantum["n_train"] == n)]
        return json.loads(row["details"].iloc[0])["params"]

    def fixed_fit(model, seed, n, ids):
        """Refit a model of the quantum section at the settings it chose there (no search)."""
        frame, red = inputs["pls10"]
        p = chosen(model, seed, n)
        if model == "qridge":
            est, _ = quantum_ridge_build(seed, SC_B, TG_B, red, frame.shape[1])
            return TabularFitter(frame, est, {k: [v] for k, v in p.items()}, seed, cv_max_n=max(q_sizes)).fit(ids, y.loc[ids])
        kern = kernel_for(model, gammas=[p["gamma"]], gammas_p=[p["gamma_p"]] if "gamma_p" in p else None)
        return KernelRidgeFitter(frame, kern, seed, scaling=SC_B, target=TG_B, reduction=red, alphas=(p["alpha"],),
                                 cv_max_n=max(q_sizes)).fit(ids, y.loc[ids])

    # --- Section 2: finite shots on the quantum test subsets ------------------------------------
    if "shots" in sections:
        scfg = cfg["shots"]
        kinds = {"qkrr": "fidelity", "pqk": "projected", "qridge": "z_readout"}
        rows = []
        Xq = {k: XA.loc[v].to_numpy() for k, v in q_ids.items()}
        yq = {k: y.loc[v].to_numpy() for k, v in q_ids.items()}
        with timed("shots section"):
            for seed in seeds:
                for n in ([100] if args.smoke else scfg["sizes"]):
                    ids = splits.train[seed][n]
                    for model in scfg["models"]:
                        f = fixed_fit(model, seed, n, ids)
                        for r in shot_scores(f, XA.loc[ids].to_numpy(), y.loc[ids].to_numpy(), Xq, yq, scfg["shots"],
                                             2 if args.smoke else scfg["reps"], seed * 10_000 + n, kinds[model]):
                            rows.append({"model": model, "seed": seed, "n_train": n, **r})
        save(pd.DataFrame(rows), "shots", shots=scfg["shots"], reps=scfg["reps"])

    # --- Section 3: hardware noise (fake-backend noise model on local Aer) -----------------------
    if "noise" in sections:
        ncfg = cfg["noise"]
        per = 5 if args.smoke else ncfg["per_subset"]
        test_ids = np.concatenate([np.sort(np.random.default_rng([2027, i]).choice(v, per, replace=False))
                                   for i, v in enumerate(q_ids.values())])
        which = np.concatenate([[k] * per for k in q_ids])
        ids = splits.train[ncfg["seed"]][ncfg["n_train"]]
        Xtr, ytr, Xte, yte = XA.loc[ids].to_numpy(), y.loc[ids].to_numpy(), XA.loc[test_ids].to_numpy(), y.loc[test_ids].to_numpy()
        rows = []
        with timed(f"noise section ({len(test_ids)} test molecules, {ncfg['backend']})"):
            f = fixed_fit("qkrr", ncfg["seed"], ncfg["n_train"], ids)
            fid_rows, kernels = noisy_fidelity_scores(f, Xtr, Xte, yte, ncfg["shots"], ncfg["backend"])
            rows += [{"model": "qkrr", **r} for r in fid_rows]
            for model, kind in (("pqk", "projected"), ("qridge", "z_readout")):
                g = fixed_fit(model, ncfg["seed"], ncfg["n_train"], ids)
                rows += [{"model": model, **r} for r in noisy_feature_scores(g, Xtr, ytr, Xte, yte, ncfg["shots"],
                                                                             ncfg["backend"], kind)]
        noise = pd.DataFrame(rows)
        save(noise, "noise", backend=ncfg["backend"], shots=ncfg["shots"], n_train=ncfg["n_train"], seed=ncfg["seed"],
             test_molecules={k: per for k in q_ids}, survival_mean=float(kernels["survival"].mean()))
        np.savez(out_dir / f"{tag}_noise_kernels.npz", ids=test_ids, subset=which, **kernels)

    # --- Section 4: the most accurate classical models (Track A) ---------------------------------
    if "track_a" in sections:
        acfg, xcfg = cfg["track_a"], load_config()
        TA, CC = xcfg["track_a"], xcfg["charge_model"]
        chosen_net = json.loads((RESULTS_DIR / "explore04_charge_selection.meta.json").read_text())["chosen"]
        p_net = {**CC["common"], **CC["candidates"][chosen_net]}
        p_net["hidden"] = tuple(p_net["hidden"])
        a_sizes = [100] if args.smoke else list(acfg["sizes"])
        full = int(splits.sizes[-1])
        with timed("per-atom features for the charge network"):
            atoms = load_atoms(table, np.unique(np.concatenate([splits.pool, every_eval])), catalog)

        def make_track_a(seed, n):
            out = {}
            for m in acfg["models"]:
                if m == "mean":
                    out[m] = MeanFitter()
                elif m in ("ridge", "rbf_krr"):
                    if m == "rbf_krr" and n > acfg["rbf_krr_max_n"]:
                        continue
                    est, grid = classical.build(m, seed, scaling=SC_A, target=TG_A, n_features=XA.shape[1])
                    out[m] = TabularFitter(XA, est, grid, seed)
                elif m == "xgb":
                    out[m] = XGBFitter(XA, seed, n_iter=TA["xgb_iter"]["small" if n <= TA["xgb_large_n"] else "large"],
                                       target=TG_A)
                elif m == "charge_net":
                    out[m] = ChargeFitter(atoms, p_net, seed)
            return out

        sets = {s: {n: splits.train[s][n] for n in a_sizes} for s in seeds}
        if not args.smoke:
            for s in acfg["full_pool_seeds"]:
                sets.setdefault(s, {})[full] = splits.train[s][full]
        with timed("track A section"):
            track_a, _ = dev_curve(make_track_a, sets, eval_ids, y, cache_dir=cache, progress=True)
        save(track_a, "track_a", scaling=SC_A, target=TG_A, charge_net=chosen_net, sizes=a_sizes,
             full_pool_seeds=acfg["full_pool_seeds"])

    if not args.dry_run:
        log_run({"run": run, "started": started, "finished": time.strftime("%Y-%m-%d %H:%M"), "commit": commit,
                 "config_hash": config_hash(cfg), "sections": ",".join(sections), "smoke": args.smoke,
                 "note": cfg["note"]})
        print(f"logged run {run} in {RUN_LOG.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
