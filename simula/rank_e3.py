"""Experiment 3, cold start: when both creatives are new, can Jev rank them from the stats of their attribute values?

Reuses the pair harness in `simula.rank_pairs`. Pairs are the harness pairs whose two C14s are both absent from the
refit window. Jev sees the context and, per candidate, each attribute value with its refit impressions and smoothed
CTR. No pCTR, no creative history, no C14 and no C17 (it behaves like a creative id). Run:
  uv run python -m simula.rank_e3 states            # Jev input JSONL, written outside the checkout
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.rank_e3 report            # reports/rank_e3.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from simula import baselines, train
from simula.data import add_flags, load, split
from simula.jev_ctr import bootstrap_diff
from simula.rank_pairs import (_mcnemar, _scored, add_history, build_pairs, ceiling, history, read_answers,
                               state as pair_state)

ATTRIBUTES = ["banner_pos", "C15", "C16", "C18", "C19", "C20", "C21"]
CELLS = ["genre_x_C18", "app_category_x_C18"]
WORK = Path.home() / "simula-local/rank-jev-work/exp-rank-e3"
ANSWERS = Path("reports/rank_e3_jev_answers.jsonl")
PICK = {"type": "choice", "instructions": "Which candidate ad is the user more likely to click, judging only from how "
        "its attributes have performed before?",
        "criteria": {"X1": "candidate X1", "X2": "candidate X2", "tie": "no real difference"}}
N_DUP = 100


def both_unseen(pairs, rows):
    """The pairs whose two creatives have no refit impressions."""
    unseen = rows["C14_impressions"].eq(0)
    return pairs[unseen.loc[pairs["x1"]].to_numpy() & unseen.loc[pairs["x2"]].to_numpy()].reset_index(drop=True)


def _candidate(r):
    stats = {k: {"value": str(r[k]), "impressions": int(r[f"{k}_impressions"]),
                 "smoothed_ctr": round(float(r[f"{k}_smoothed_ctr"]), 4)} for k in ATTRIBUTES}
    cells = {k: {"impressions": int(r[f"{k}_impressions"]), "smoothed_ctr": round(float(r[f"{k}_smoothed_ctr"]), 4)}
             for k in CELLS}
    return {"slot": int(r["banner_pos"]), **stats, **cells}


def state(pair):
    """Jev state for a (row_x1, row_x2) pair: the harness context plus attribute stats. No pCTR, C14, C17 or click."""
    r1, r2 = pair
    return {"context": pair_state(pair, with_pctr=False)["context"],
            "candidates": {"X1": _candidate(r1), "X2": _candidate(r2)}}


def count_weighted_mean(candidate, prior):
    """Impressions-weighted mean of a state candidate's attribute smoothed CTRs; the prior if every count is 0."""
    stats = [candidate[k] for k in ATTRIBUTES + CELLS]
    n = sum(s["impressions"] for s in stats)
    return sum(s["impressions"] * s["smoothed_ctr"] for s in stats) / n if n else prior


def prepare(data_dir):
    """Both-unseen pairs, and their rows annotated with pCTR, lookup, creative and attribute history."""
    df = add_flags(load(data_dir))
    refit, test = split(df, "refit"), split(df, "test")
    hist = history(refit)
    pairs = build_pairs(test)
    rows = add_history(test.loc[np.union1d(pairs["x1"], pairs["x2"])].copy(), hist, ["C14", *CELLS, *ATTRIBUTES])
    pairs = both_unseen(pairs, rows)
    print(f"{len(pairs)} both-unseen pairs from {pairs['context_group'].nunique()} groups")
    rows = rows.loc[np.union1d(pairs["x1"], pairs["x2"])].copy()
    rows["pctr"] = train.predict(train.load_bundle("bundle/B"), rows)
    rows["lookup"] = baselines.predict(baselines.lookup(refit, ["app_id", "C18"]), rows)
    return pairs, rows, hist["prior"]


def dup_sample(rows):
    """The fixed 100 real candidates each duplicated under both labels."""
    return np.sort(np.random.default_rng(0).choice(rows.index.to_numpy(), N_DUP, replace=False))


def write_states(pairs, rows, path):
    """Write the Jev input JSONL: each pair, each pair swapped, and the identical-evidence pairs."""
    q = {"pick": PICK}
    with open(path, "w") as f:
        for p in pairs.itertuples():
            r1, r2 = rows.loc[p.x1], rows.loc[p.x2]
            for key, pair in ((f"pair:{p.pair}", (r1, r2)), (f"swap:{p.pair}", (r2, r1))):
                f.write(json.dumps({"id": key, "state": state(pair), "questions": q}) + "\n")
        for i in dup_sample(rows):
            f.write(json.dumps({"id": f"dup:{i}", "state": state((rows.loc[i], rows.loc[i])), "questions": q}) + "\n")


def evaluate(pairs, rows, prior, answers):
    """Accuracy of every arm on the both-unseen pairs, each vs B and vs the count-weighted mean, plus Jev's checks."""
    c, u = (rows.loc[np.where(pairs["clicked"].eq(x), pairs["x1"], pairs["x2"])].reset_index(drop=True)
            for x in ("X1", "X2"))
    groups = pairs["character_id"].to_numpy()
    cwm = {name: np.array([count_weighted_mean(_candidate(r), prior) for _, r in side.iterrows()])
           for name, side in (("c", c), ("u", u))}
    b = _scored(c["pctr"].to_numpy(), u["pctr"].to_numpy())
    pick = np.array([answers[f"pair:{p}"]["answers"]["pick"]["choice"] for p in pairs["pair"]])
    arms = {"coin": np.full(len(pairs), 0.5), "B": b,
            "lookup": _scored(c["lookup"].to_numpy(), u["lookup"].to_numpy()),
            "count_weighted_mean": _scored(cwm["c"], cwm["u"]),
            "jev": np.where(pick == "tie", b, (pick == pairs["clicked"]).astype(float))}
    table = []
    for name, score in arms.items():
        row = {"arm": name, "acc": float(score.mean()), "acc_ci": bootstrap_diff(groups, score, np.zeros(len(score)))[1:]}
        for ref in ("B", "count_weighted_mean"):
            if name != ref:
                row[f"diff_vs_{ref}"] = bootstrap_diff(groups, score, arms[ref])
                row[f"mcnemar_vs_{ref}"] = _mcnemar(score, arms[ref]) if name != "coin" else None
        table.append(row)

    flip = {"X1": "X2", "X2": "X1", "tie": "tie"}
    swapped = np.array([flip[answers[f"swap:{p}"]["answers"]["pick"]["choice"]] for p in pairs["pair"]])
    decided = pick != "tie"
    dup = [answers[f"dup:{i}"]["answers"]["pick"] for i in dup_sample(rows)]
    calls = list(answers.values())
    return {
        "pairs": len(pairs), "groups": int(pairs["context_group"].nunique()),
        "characters": int(pairs["character_id"].nunique()),
        "b_implied_benchmark": ceiling(c["pctr"].to_numpy(), u["pctr"].to_numpy()),
        "identical_evidence_pairs": int(sum(_candidate(a) == _candidate(b) for (_, a), (_, b) in
                                            zip(c.iterrows(), u.iterrows()))),
        "jev_ties": int((pick == "tie").sum()), "b_exact_ties": int((b == 0.5).sum()),
        "arms": table,
        "jev_non_tie": {"pairs": int(decided.sum()), "jev_acc": float(arms["jev"][decided].mean()),
                        "b_acc": float(b[decided].mean())},
        "order_check": {"pairs": len(pairs), "same_choice_share": float(np.mean(swapped == pick)),
                        "same_ad_share_both_non_tie": float(np.mean((swapped == pick)[decided & (swapped != "tie")])),
                        "ties_original": int((pick == "tie").sum()), "ties_swapped": int((swapped == "tie").sum())},
        "identical_evidence": {"pairs": len(dup), "non_tie_share": float(np.mean([d["choice"] != "tie" for d in dup])),
                               "mean_abs_p_x1_minus_p_x2": float(np.mean(
                                   [abs(d["probabilities"]["X1"] - d["probabilities"]["X2"]) for d in dup]))},
        "jev": {"calls": len(calls), "attempts": int(sum(a["attempts"] for a in calls)), "models": sorted({a["model"] for a in calls}),
                "input_tokens": int(sum(a["tokens_in"] for a in calls)),
                "call_seconds_sum": float(sum(a["seconds"] for a in calls)),
                "call_seconds_median": float(np.median([a["seconds"] for a in calls]))},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    pairs, rows, prior = prepare(args.data)
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        write_states(pairs, rows, WORK / "states.jsonl")
        print("wrote", WORK / "states.jsonl")
    else:
        out = evaluate(pairs, rows, prior, read_answers(ANSWERS))
        Path("reports/rank_e3.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps(out, indent=1, default=float))
