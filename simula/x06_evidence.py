"""x06, evidence judge: does Jev read deploy-gate and benchmark evidence better than the standard tests?

(a) Ship/no-ship readouts (Fable E7). 200 synthetic A/B readouts built from Oct 29-30 rows. Each arm draws 8,000 real
    impressions by side; the candidate arm is oversampled site-side (site CTR is higher) by a random margin, and a
    per-side click-rate shift delta in {-1, -0.5, 0, +0.5, +1} pp is injected into it. Truth: ship iff delta > 0.
    Jev answers a Noul from the per-side table (arm `sides`) and from overall rows plus a mix note (arm `mix`).
    Controls: naive pooled z on overall CTR, and the CMH statistic stratified by side. Metric: AUROC.
    Pre-declared win: AUROC(Jev sides) >= AUROC(CMH) and AUROC(Jev sides) - AUROC(naive) interval above 0.
(b) Benchmark on trial (Astra 5). Panels of matched pairs (a clicked and an unclicked impression with different
    creatives in one exact context, `rank_pairs.build_pairs`) over Oct 29-30 rows, with clicks simulated per row.
    Genuine panels: clicks from B's calibrated pCTR, so candidates matter. False panels: clicks from a request-only
    LightGBM fitted on the refit window, so any ranking gain is nuisance. Only panels where B's accuracy exceeds 50%
    become claims. Control: exact-request matching plus a fixed checklist (>= 10 request-identical pairs, B above
    50% on them, B above the request-only ranker). Pre-declared win: at the checklist's retention of genuine
    panels, Jev rejects a larger share of false panels, interval above 0.

Intervals resample readouts or panels (tasks), not impressions. Run:
  uv run python -m simula.x06_evidence states
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x06_evidence report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from simula import train
from simula.data import add_flags, load, split
from simula.features import CONTRACT, fit_encoder, transform
from simula.rank_pairs import CONTEXT, _auc_diff, _scored, build_pairs, read_answers

WORK = Path.home() / "simula-local/rank-jev-work/exp-x-evidence"
ANSWERS = Path("reports/x06_jev_answers.jsonl")
DELTAS = [-0.01, -0.005, 0.0, 0.005, 0.01]
N_ARM, N_READOUTS, N_PANELS, N_RECHECK = 8000, 200, 150, 50
REQUEST = CONTRACT["request"] + CONTRACT["character"]
SHIP = {"type": "noul", "criteria": None, "instructions": "The candidate arm has a truly higher click rate than the "
        "control arm on comparable traffic, so it should ship."}
USE = {"type": "choice", "instructions": "How should this comparison be used for the stated claim?", "criteria": {
    "predictive_diagnostic": "it supports the claim that the ranker orders candidate ads",
    "rematch": "the apparent gain may come from request differences between paired impressions; re-match first",
    "unidentifiable": "this data cannot separate a candidate effect from request differences"}}


def readout(pools, delta, rng):
    """One synthetic readout from real click arrays by side. Returns {(arm, side): (impressions, clicks)}."""
    share_c = rng.uniform(0.25, 0.55)
    share = {"control": share_c, "candidate": share_c + rng.uniform(0, 0.35)}
    cells = {}
    for side, pool in pools.items():
        n = {arm: round(N_ARM * (s if side == "site" else 1 - s)) for arm, s in share.items()}
        idx = rng.choice(len(pool), n["control"] + n["candidate"], replace=False)
        control, candidate = pool[idx[:n["control"]]], pool[idx[n["control"]:]].copy()
        base = pool.mean()  # flip probabilities that shift this side's expected CTR by exactly delta
        if delta > 0:
            candidate |= rng.random(len(candidate)) < delta / (1 - base)
        elif delta < 0:
            candidate &= rng.random(len(candidate)) >= -delta / base
        cells[("control", side)] = (len(control), int(control.sum()))
        cells[("candidate", side)] = (len(candidate), int(candidate.sum()))
    return cells


def _overall(cells, arm):
    return tuple(sum(v[i] for (a, _), v in cells.items() if a == arm) for i in (0, 1))


def naive_z(cells):
    """Pooled two-proportion z of candidate minus control on overall CTR."""
    (n1, k1), (n0, k0) = _overall(cells, "candidate"), _overall(cells, "control")
    p = (k1 + k0) / (n1 + n0)
    return (k1 / n1 - k0 / n0) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))


def cmh_z(cells):
    """Signed Cochran-Mantel-Haenszel statistic for candidate clicks, stratified by side."""
    num = var = 0.0
    for side in {s for _, s in cells}:
        (n1, k1), (n0, k0) = cells[("candidate", side)], cells[("control", side)]
        n, m = n1 + n0, k1 + k0
        num += k1 - n1 * m / n
        var += n1 * n0 * m * (n - m) / (n * n * (n - 1))
    return num / np.sqrt(var)


def readout_states(cells, rng):
    """The two Jev states for one readout, rows in random order: per-side table, and overall rows plus a mix note."""
    rows = [{"arm": a, "traffic": s, "impressions": n, "clicks": k} for (a, s), (n, k) in cells.items()]
    overall = [{"arm": a, "traffic": "all", "impressions": n, "clicks": k}
               for a in ("control", "candidate") for n, k in [_overall(cells, a)]]
    site = {a: cells[(a, "site")][0] / _overall(cells, a)[0] for a in ("control", "candidate")}
    sides = [rows[i] for i in rng.permutation(4)] + [overall[i] for i in rng.permutation(2)]
    mix = {"rows": [overall[i] for i in rng.permutation(2)],
           "note": f"The arms' traffic mix differs: site-side traffic is {site['control']:.0%} of control and "
                   f"{site['candidate']:.0%} of candidate impressions."}
    return {"rows": sides}, mix


def readouts(test, seed=0):
    """The 200 readouts, cycling over DELTAS. Returns a list of (delta, cells, sides_state, mix_state)."""
    rng = np.random.default_rng(seed)
    pools = {s: test.loc[test["is_site_side"].eq(s == "site"), "click"].to_numpy().astype(bool) for s in ("site", "app")}
    out = []
    for i in range(N_READOUTS):
        delta = DELTAS[i % len(DELTAS)]
        cells = readout(pools, delta, rng)
        out.append((delta, cells, *readout_states(cells, rng)))
    return out


def request_rates(refit, calibration, rounds, seed=0):
    """Request-only LightGBM on REQUEST fields (300k-row refit subsample), calibrated as B is (half shift, Oct 28)."""
    sub = refit.sample(min(len(refit), 300_000), random_state=seed)
    encoder = fit_encoder(sub, REQUEST)
    booster = train.refit(sub, REQUEST, encoder, rounds)
    delta = train.calibrate(booster.predict(transform(calibration, encoder)), calibration["click"], 0.5)
    return lambda df: train.apply_calibration(booster.predict(transform(df, encoder)), delta)


def panel_stats(rows, pairs):
    """Label-free evidence about one panel. rows carry pB, pR, req_key and REQUEST; pairs come from build_pairs."""
    c = rows.loc[np.where(pairs["clicked"].eq("X1"), pairs["x1"], pairs["x2"])].reset_index(drop=True)
    u = rows.loc[np.where(pairs["clicked"].eq("X1"), pairs["x2"], pairs["x1"])].reset_index(drop=True)
    b, r = _scored(c["pB"].to_numpy(), u["pB"].to_numpy()), _scored(c["pR"].to_numpy(), u["pR"].to_numpy())
    exact = (c["req_key"] == u["req_key"]).to_numpy()
    differ = {f: round(float((c[f].astype(str) != u[f].astype(str)).mean()), 3)
              for f in REQUEST if f not in CONTEXT}
    return {"pairs": len(pairs), "ranker_accuracy": round(float(b.mean()), 4),
            "request_only_ranker_accuracy": round(float(r.mean()), 4),
            "request_identical_pairs": {"pairs": int(exact.sum()),
                                        "ranker_accuracy": round(float(b[exact].mean()), 4) if exact.any() else None},
            "share_of_pairs_differing": {f: v for f, v in differ.items() if v > 0}}


def checklist(s):
    """Fixed validity checklist after exact-request rematching: True keeps the panel as a predictive diagnostic."""
    exact = s["request_identical_pairs"]
    return bool(exact["pairs"] >= 10 and exact["ranker_accuracy"] > 0.5
                and s["ranker_accuracy"] > s["request_only_ranker_accuracy"])


def panel_state(s):
    """Jev state for one panel. Wording and provenance are identical for genuine and false panels."""
    return {"claim": f"Ranker R scores the clicked ad above the unclicked ad in {s['ranker_accuracy']:.1%} of "
                     f"{s['pairs']} matched pairs (a coin scores 50%). Claim: R ranks candidate ads.",
            "matching_keys": CONTEXT, "contract": {"request": REQUEST, "candidate": CONTRACT["candidate"]},
            "outcome_provenance": "click labels simulated per impression from a rate model fitted before this window",
            "request_only_ranker": "a model fitted on request fields only, scored on the same pairs", **s}


def panels(test, p_b, p_r, seed=1):
    """N_PANELS genuine and N_PANELS false panels whose B accuracy exceeds 50%. Returns [(world, stats)]."""
    rows = test.assign(pB=p_b, pR=p_r, req_key=pd.util.hash_pandas_object(test[REQUEST].astype(str), index=False))
    rows = rows[rows.groupby(CONTEXT)["C14"].transform("nunique").gt(1)]
    rows["group"] = rows.groupby(CONTEXT).ngroup()
    rng, out, kept = np.random.default_rng(seed), [], {"genuine": 0, "false": 0}
    while min(kept.values()) < N_PANELS:
        world = "genuine" if kept["genuine"] <= kept["false"] else "false"
        groups = rng.choice(rows["group"].max() + 1, rng.integers(300, 1500), replace=False)
        sub = rows[rows["group"].isin(groups)].copy()
        sub["click"] = (rng.random(len(sub)) < sub["pB" if world == "genuine" else "pR"]).astype(int)
        s = panel_stats(sub, build_pairs(sub))
        if s["ranker_accuracy"] > 0.5:
            out.append((world, s))
            kept[world] += 1
    return out


def prepare(data_dir):
    """Readouts and panels; fits the request-only model once."""
    df = add_flags(load(data_dir))
    refit, calibration, test = split(df, "refit"), split(df, "calibrate"), split(df, "test")
    rates = WORK / "x06_rates.npz"  # cached so the report step rebuilds the exact panels the states came from
    if not rates.exists():
        bundle = train.load_bundle("bundle/B")
        p_r = request_rates(refit, calibration, bundle[2]["best_iteration"])(test)
        WORK.mkdir(parents=True, exist_ok=True)
        np.savez(rates, p_b=train.predict(bundle, test), p_r=p_r)
    cached = np.load(rates)
    return readouts(test), panels(test, cached["p_b"], cached["p_r"])


def write_states(reads, pans, path):
    """Jev input: both readout arms, a reshuffled sides recheck, each panel, and a reordered-options recheck."""
    rng = np.random.default_rng(2)
    with open(path, "w") as f:
        def put(key, state, q):
            f.write(json.dumps({"id": key, "state": state, "questions": q}) + "\n")
        for i, (_, cells, sides, mix) in enumerate(reads):
            put(f"sides:{i}", sides, {"ship": SHIP})
            put(f"mix:{i}", mix, {"ship": SHIP})
        for i in range(N_RECHECK):
            put(f"resides:{i}", readout_states(reads[i][1], rng)[0], {"ship": SHIP})
        reordered = {**USE, "criteria": dict(reversed(USE["criteria"].items()))}
        for i, (_, s) in enumerate(pans):
            put(f"panel:{i}", panel_state(s), {"use": USE})
            if i < N_RECHECK:
                put(f"repanel:{i}", panel_state(s), {"use": reordered})


def _at_retention(score, genuine, retention):
    """Share of false panels rejected when the score keeps the top `retention` share of genuine panels."""
    cut = np.quantile(score[genuine], 1 - retention, method="lower") if retention > 0 else np.inf
    return float(np.mean(score[~genuine] < cut))


def _boot(fn, n, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(n, size=n)) for _ in range(n_boot)]
    return [float(fn(np.arange(n))), *map(float, np.percentile(vals, [2.5, 97.5]))]


def evaluate(reads, pans, answers):
    """AUROCs and paired differences for (a); rejection at matched retention for (b); Jev stability checks."""
    y = np.array([d > 0 for d, *_ in reads])
    arms = {"naive_z": np.array([naive_z(c) for _, c, *_ in reads]), "cmh_z": np.array([cmh_z(c) for _, c, *_ in reads]),
            "jev_sides": np.array([answers[f"sides:{i}"]["answers"]["ship"]["noul"] for i in range(len(reads))]),
            "jev_mix": np.array([answers[f"mix:{i}"]["answers"]["ship"]["noul"] for i in range(len(reads))])}
    nonzero = np.array([d != 0 for d, *_ in reads])
    idx = np.arange(len(reads))
    a = {"auroc": {k: float(roc_auc_score(y, v)) for k, v in arms.items()},
         "auroc_nonzero_delta": {k: float(roc_auc_score(y[nonzero], v[nonzero])) for k, v in arms.items()},
         "diffs": {f"{x}_minus_{z}": _auc_diff(idx, y, arms[x], arms[z], n_boot=2000)
                   for x, z in [("jev_sides", "cmh_z"), ("jev_sides", "naive_z"), ("jev_mix", "naive_z"),
                                ("cmh_z", "naive_z")]},
         "mean_noul_by_delta": {str(d): {k: float(arms[k][[r[0] == d for r in reads]].mean())
                                         for k in ("jev_sides", "jev_mix")} for d in DELTAS},
         "reshuffle_mean_abs_change": float(np.mean([abs(answers[f"resides:{i}"]["answers"]["ship"]["noul"]
                                                         - arms["jev_sides"][i]) for i in range(N_RECHECK)]))}
    a["win"] = bool(a["auroc"]["jev_sides"] >= a["auroc"]["cmh_z"] and a["diffs"]["jev_sides_minus_naive_z"][1] > 0)

    genuine = np.array([w == "genuine" for w, _ in pans])
    use = [answers[f"panel:{i}"]["answers"]["use"] for i in range(len(pans))]
    jev = np.array([u["probabilities"]["predictive_diagnostic"] for u in use])
    keep = np.array([checklist(s) for _, s in pans])
    exact_z = np.array([(s["request_identical_pairs"]["ranker_accuracy"] - 0.5) * 2 * np.sqrt(s["request_identical_pairs"]["pairs"])
                        if s["request_identical_pairs"]["pairs"] else -9.0 for _, s in pans])
    gap = np.array([s["ranker_accuracy"] - s["request_only_ranker_accuracy"] for _, s in pans])
    retention = float(keep[genuine].mean())

    def diff(i):
        g = genuine[i]
        return _at_retention(jev[i], g, keep[i][g].mean()) - float(np.mean(~keep[i][~g]))
    reorder = [answers[f"repanel:{i}"]["answers"]["use"] for i in range(N_RECHECK)]
    b = {"panels": {"genuine": int(genuine.sum()), "false": int((~genuine).sum())},
         "checklist": {"retention_genuine": retention, "rejection_false": float(np.mean(~keep[~genuine]))},
         "jev_at_checklist_retention": {"rejection_false": _at_retention(jev, genuine, retention)},
         "jev_rejection_minus_checklist": _boot(diff, len(pans)),
         "jev_argmax": {w: pd.Series([u["choice"] for u, g in zip(use, genuine) if g == (w == "genuine")])
                        .value_counts().to_dict() for w in ("genuine", "false")},
         "auroc_genuine": {"jev": float(roc_auc_score(genuine, jev)), "exact_request_z": float(roc_auc_score(genuine, exact_z)),
                           "accuracy_gap_vs_request_only": float(roc_auc_score(genuine, gap))},
         "reorder_same_choice_share": float(np.mean([r["choice"] == u["choice"] for r, u in zip(reorder, use)])),
         "medians": {w: {k: float(np.median([s[k] for ww, s in pans if ww == w]))
                         for k in ("pairs", "ranker_accuracy", "request_only_ranker_accuracy")}
                     | {"request_identical_pairs": float(np.median([s["request_identical_pairs"]["pairs"]
                                                                     for ww, s in pans if ww == w]))}
                     for w in ("genuine", "false")}}
    b["win"] = bool(b["jev_rejection_minus_checklist"][1] > 0)
    calls = list(answers.values())
    return {"readouts": a, "panels": b,
            "jev": {"calls": len(calls), "attempts": int(sum(c["attempts"] for c in calls)),
                    "models": sorted({c["model"] for c in calls}), "input_tokens": int(sum(c["tokens_in"] for c in calls)),
                    "call_seconds_median": float(np.median([c["seconds"] for c in calls]))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    reads, pans = prepare(args.data)
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        write_states(reads, pans, WORK / "x06_states.jsonl")
        print("wrote", WORK / "x06_states.jsonl")
    else:
        out = evaluate(reads, pans, read_answers(ANSWERS))
        Path("reports/x06_evidence_judge.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps(out, indent=1, default=float))
