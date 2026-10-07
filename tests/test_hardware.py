"""Hardware-run plumbing (hardware.py), tested on local simulators only: PUB construction, parsing
of sampler results back into model features, scoring, and transpilation for a fake device."""

import json

import numpy as np
import pandas as pd
import pytest

from qm9dipole.hardware import (build_blocks, fit_models, parse, plan, run_local, score, select_test_molecules,
                                settings_from_run)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(70, 4))
    y = np.clip(2 + np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2, 0.05, None)
    ids = np.arange(300, 370)
    frame = pd.DataFrame(X, index=ids, columns=list("abcd"))
    settings = {"qkrr": {"alpha": 0.1, "gamma": 0.4}, "pqk": {"alpha": 0.1, "gamma": 0.4, "gamma_p": 0.5},
                "qridge": {"regressor__model__alpha": 1.0, "regressor__model__gamma": 0.4}}
    tr, te = ids[:50], ids[50:]
    models = fit_models(frame, pd.Series(y, index=ids), tr, settings, 0, "standard", "sqrt", ("pca", 4))
    return frame, pd.Series(y, index=ids), tr, te, models


def test_select_test_molecules_matches_the_noise_section():
    subsets = {"a": np.arange(100, 180), "b": np.arange(500, 870)}
    ids, which = select_test_molecules(subsets, 5)
    expect = np.sort(np.random.default_rng([2027, 1]).choice(subsets["b"], 5, replace=False))
    np.testing.assert_array_equal(ids[5:], expect)
    assert list(which) == ["a"] * 5 + ["b"] * 5
    np.testing.assert_array_equal(select_test_molecules(subsets, 5)[0], ids)


def test_settings_from_run_reads_the_chosen_parameters():
    q = pd.DataFrame({"model": ["qkrr", "pqk", "rbf_krr"], "inputs": ["pls10"] * 3, "seed": [0] * 3, "n_train": [100] * 3,
                      "details": [json.dumps({"params": {"alpha": 0.1, "gamma": 0.02}}),
                                  json.dumps({"params": {"alpha": 1.0, "gamma": 0.06, "gamma_p": 0.1}}), "{}"]})
    s = settings_from_run(q, 0, 100)
    assert s == {"qkrr": {"alpha": 0.1, "gamma": 0.02}, "pqk": {"alpha": 1.0, "gamma": 0.06, "gamma_p": 0.1}}


def test_ideal_sampling_reproduces_exact_features_kernel_and_scores(toy):
    frame, y, tr, te, models = toy
    Xtr, Xte, ytr, yte = frame.loc[tr].to_numpy(), frame.loc[te].to_numpy(), y.loc[tr].to_numpy(), y.loc[te].to_numpy()
    blocks = build_blocks(models, Xtr, ytr, Xte, ["pqk", "qridge", "kernel"], backend=None, kernel_block=(4, 3))
    assert [(b.part, b.basis, b.rows) for b in blocks] == [("pqk", "X", 70), ("pqk", "Y", 70), ("pqk", "Z", 70),
                                                         ("qridge", "Z", 70), ("kernel", "overlap", 4 * 3 + 4)]
    shots = 40_000
    measured = parse(run_local(blocks, shots, backend=None, seed=1), blocks)
    from qm9dipole.final import feature_model

    for part, kind in (("pqk", "projected"), ("qridge", "z_readout")):
        fm = feature_model(models[part], kind, Xtr, ytr)
        exact = np.vstack([fm.exact_train, fm.exact(Xte)])
        assert measured[part].shape == exact.shape
        assert np.abs(measured[part] - exact).max() < 0.03
    exact_k = models["qkrr"].gram_to_train(Xte[:4])[:, :3].ravel()
    np.testing.assert_allclose(measured["kernel"][:12], exact_k, atol=0.02)
    np.testing.assert_allclose(measured["kernel"][12:], 1.0)  # an ideal device: every state overlaps itself fully

    sc, kern = score(models, measured, Xtr, ytr, Xte, yte, np.array(["f"] * 10 + ["u"] * 10), shots, "ideal Aer",
                     kernel_block=(4, 3))
    both = sc[sc["test_subset"] == "both"].set_index(["model", "mode"])["mae_D"]
    for part in ("pqk", "qridge"):
        assert both[(part, "exact")] == pytest.approx(np.abs(models[part].predict(te) - yte).mean(), abs=1e-9)
        assert both[(part, "ideal Aer")] == pytest.approx(both[(part, "exact")], abs=0.03)
    assert set(sc["test_subset"]) == {"both", "f", "u"}
    assert len(kern) == 12 and np.allclose(kern["mitigated"], kern["measured"], atol=1e-9)  # survival 1: no correction


def test_transpiled_for_a_fake_device_and_sampled_with_its_noise(toy):
    frame, y, tr, te, models = toy
    Xtr, Xte, ytr = frame.loc[tr].to_numpy()[:6], frame.loc[te].to_numpy()[:4], y.loc[tr].to_numpy()[:6]
    blocks = build_blocks(models, Xtr, ytr, Xte, ["qridge", "kernel"], backend="FakeQuebec", kernel_block=(2, 2))
    p = plan(blocks, 1000, "FakeQuebec")
    assert (p["two_qubit_gates"] > 0).all() and (p["duration_us"] > 0).all() and (p["qubits"] == 4).all()
    assert p["circuits"].tolist() == [10, 2 * 2 + 2]
    measured = parse(run_local(blocks, 2000, backend="FakeQuebec", seed=0), blocks)
    assert measured["qridge"].shape == (10, 4) and np.all(np.abs(measured["qridge"]) <= 1)
    surv = measured["kernel"][4:]
    assert np.all((surv > 0.5) & (surv < 1.0))  # noise: a state no longer overlaps itself perfectly


def test_runtime_sampler_accepts_the_pubs(toy):
    # The submission code path (qiskit_ibm_runtime.SamplerV2, one job, QPU-time cap) in Qiskit Runtime's local
    # testing mode: a fake device as the mode runs the job on Aer here; nothing is sent to IBM.
    from qiskit_ibm_runtime.fake_provider import FakeQuebec

    from qm9dipole.hardware import submit

    frame, y, tr, te, models = toy
    be = FakeQuebec()
    blocks = build_blocks(models, frame.loc[tr].to_numpy()[:3], y.loc[tr].to_numpy()[:3], frame.loc[te].to_numpy()[:2],
                          ["qridge"], backend=be)
    job = submit(blocks, 200, be, 60)
    measured = parse(job.result(), blocks)
    assert measured["qridge"].shape == (5, 4) and np.all(np.abs(measured["qridge"]) <= 1)
