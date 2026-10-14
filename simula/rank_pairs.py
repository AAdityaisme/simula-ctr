"""Pair harness for the rank-with-Jev experiments, and experiments 1 (Jev picks the ad) and 2 (Jev as a gate).

A pair is a clicked and an unclicked impression with different creatives (C14) in one exact context. Each row
carries B's calibrated pCTR, the lookup(app_id, C18) baseline and refit-window history. Run:
  uv run python -m simula.rank_pairs states            # Jev input JSONL, written outside the checkout
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.rank_pairs report            # reports/rank_e12.{md,json}
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from simula import baselines, train
from simula.data import add_flags, load, split
from simula.jev_ctr import bootstrap_diff

M = 20
CONTEXT = ["hour", "character_id", "app_id", "site_id", "device_type"]
C_FIELDS = ["C15", "C16", "C17", "C18", "C19", "C20", "C21"]
HISTORY_KEYS = {"C14": ["C14"], "genre_x_C18": ["genre", "C18"], "app_category_x_C18": ["app_category", "C18"],
                **{c: [c] for c in ["banner_pos", "C15", "C16", "C18", "C19", "C20", "C21"]}}
STATE_KEYS = ("C14", "genre_x_C18", "app_category_x_C18")
WORK = Path.home() / "simula-local/rank-jev-work/exp-rank-e12"
ANSWERS = Path("reports/rank_e12_jev_answers.jsonl")
PICK = {"type": "choice", "instructions": "Which candidate ad is the user more likely to click? Use the evidence in "
        "the state.", "criteria": {"X1": "candidate X1", "X2": "candidate X2"}}
GATE = {"type": "noul", "instructions": "The candidate with the higher model_pctr is the one the user clicks.",
        "criteria": None}
N_SWAP = 300
NEAR_TIE = 0.01


def history(refit):
    """Refit-window impressions, clicks and m-estimate CTR (m=20, around the refit CTR) for each history key."""
    prior = float(refit["click"].mean())
    tables = {}
    for name, cols in HISTORY_KEYS.items():
        g = refit.groupby(cols)["click"].agg(clicks="sum", impressions="count")
        g["smoothed_ctr"] = (g["clicks"] + M * prior) / (g["impressions"] + M)
        tables[name] = g
    return {"prior": prior, "tables": tables}


def add_history(rows, hist, keys=STATE_KEYS):
    """Join <key>_impressions, _clicks and _smoothed_ctr onto rows. An unseen key gets 0 and the prior."""
    out = rows
    for name in keys:
        out = out.join(hist["tables"][name].add_prefix(f"{name}_"), on=HISTORY_KEYS[name])
        out[[f"{name}_impressions", f"{name}_clicks"]] = (
            out[[f"{name}_impressions", f"{name}_clicks"]].fillna(0).astype(int))
        out[f"{name}_smoothed_ctr"] = out[f"{name}_smoothed_ctr"].fillna(hist["prior"])
    return out


def _x1_first(id_a, id_b):
    """Label-free display order: a sha256 of the sorted id pair decides which id is X1. Returns (pair id, x1 id)."""
    lo, hi = sorted((id_a, id_b))
    digest = hashlib.sha256(f"{lo}|{hi}".encode()).hexdigest()
    return digest[:16], lo if int(digest[-1], 16) % 2 == 0 else hi


def build_pairs(test):
    """Every (clicked, unclicked) pair with different C14 inside one exact context.

    Returns one row per pair: pair, x1 and x2 (row labels into test), clicked ("X1"/"X2") and character_id.
    """
    g = test.groupby(CONTEXT)
    keep = g["click"].transform("min").eq(0) & g["click"].transform("max").eq(1) & g["C14"].transform("nunique").gt(1)
    rows = test.loc[keep, CONTEXT + ["id", "C14", "click"]].reset_index(names="row")
    m = rows[rows["click"].eq(1)].merge(rows[rows["click"].eq(0)], on=CONTEXT, suffixes=("_c", "_u"))
    m = m[m["C14_c"].ne(m["C14_u"])]
    order = [_x1_first(a, b) for a, b in zip(m["id_c"], m["id_u"])]
    clicked_first = np.array([x1 == c for (_, x1), c in zip(order, m["id_c"])], dtype=bool)
    return pd.DataFrame({
        "pair": [p for p, _ in order],
        "x1": np.where(clicked_first, m["row_c"], m["row_u"]),
        "x2": np.where(clicked_first, m["row_u"], m["row_c"]),
        "clicked": np.where(clicked_first, "X1", "X2"),
        "character_id": m["character_id"].to_numpy(),
        "context_group": m.groupby(CONTEXT).ngroup().to_numpy(),
    })


def _history(r, key, clicks=False):
    out = {"impressions": int(r[f"{key}_impressions"])}
    if clicks:
        out["clicks"] = int(r[f"{key}_clicks"])
    return {**out, "smoothed_ctr": round(float(r[f"{key}_smoothed_ctr"]), 4)}


def _candidate(r, with_pctr):
    c = {"slot": int(r["banner_pos"]), **{k: str(r[k]) for k in C_FIELDS},
         "creative_history": _history(r, "C14", clicks=True),
         "genre_x_C18": _history(r, "genre_x_C18"),
         "app_category_x_C18": _history(r, "app_category_x_C18")}
    if with_pctr:
        c["model_pctr"] = round(float(r["pctr"]), 4)
    return c


def state(pair, with_pctr):
    """Jev state for a (row_x1, row_x2) pair of annotated rows. Never contains the click or C14's raw value."""
    r1, r2 = pair
    context = {"character_genre": r1["genre"], "character_description": r1["character_description"],
               "app_category": r1["app_category"], "site_category": r1["site_category"],
               "hour_of_day": int(r1["hour_of_day"]), "device_type": int(r1["device_type"])}
    return {"context": context, "candidates": {"X1": _candidate(r1, with_pctr), "X2": _candidate(r2, with_pctr)}}


def ceiling(p_a, p_b):
    """Pairwise accuracy if the pCTRs were the truth: P(the higher one is the click | exactly one click)."""
    hi, lo = np.maximum(p_a, p_b), np.minimum(p_a, p_b)
    return float(np.mean(hi * (1 - lo) / (hi * (1 - lo) + lo * (1 - hi))))


def prepare(data_dir):
    """Load the data, build the pairs and annotate each pair row with pCTR, lookup and history."""
    df = add_flags(load(data_dir))
    refit, test = split(df, "refit"), split(df, "test")
    pairs = build_pairs(test)
    print(f"{len(pairs)} pairs from {pairs['context_group'].nunique()} groups")
    rows = test.loc[np.union1d(pairs["x1"], pairs["x2"])].copy()
    rows["pctr"] = train.predict(train.load_bundle("bundle/B"), rows)
    rows["lookup"] = baselines.predict(baselines.lookup(refit, ["app_id", "C18"]), rows)
    return pairs, add_history(rows, history(refit))


def swap_sample(pairs):
    """The fixed 300 pairs asked again with X1 and X2 swapped."""
    return np.sort(np.random.default_rng(0).choice(len(pairs), N_SWAP, replace=False))


def write_states(pairs, rows, path):
    """Write the Jev input JSONL: with pCTR (pick, gate), without pCTR (pick), and the swapped sample (pick)."""
    with open(path, "w") as f:
        for p in pairs.itertuples():
            r1, r2 = rows.loc[p.x1], rows.loc[p.x2]
            items = [(f"with:{p.pair}", (r1, r2), True, {"pick": PICK, "gate": GATE}),
                     (f"without:{p.pair}", (r1, r2), False, {"pick": PICK})]
            for key, pair, with_pctr, questions in items:
                f.write(json.dumps({"id": key, "state": state(pair, with_pctr), "questions": questions}) + "\n")
        for i in swap_sample(pairs):
            p = pairs.iloc[i]
            f.write(json.dumps({"id": f"swap:{p.pair}", "state": state((rows.loc[p.x2], rows.loc[p.x1]), True),
                                "questions": {"pick": PICK}}) + "\n")


def _scored(a, b):
    """1 if a > b, 0.5 on a tie, 0 otherwise."""
    return np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))


def _mcnemar(arm, ref):
    wins, losses = int(((arm == 1) & (ref == 0)).sum()), int(((arm == 0) & (ref == 1)).sum())
    return wins, losses, float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0


def _auc_diff(groups, y, a, b, n_boot=1000, seed=0):
    """Character-cluster bootstrap of AUROC(a) - AUROC(b). Returns (diff, lo, hi)."""
    codes, _ = pd.factorize(groups)
    members = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(len(members), size=len(members))])
        boots.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return roc_auc_score(y, a) - roc_auc_score(y, b), *map(float, np.percentile(boots, [2.5, 97.5]))


def evaluate(pairs, rows, answers):
    """Experiments 1 and 2 from the pairs, annotated rows and Jev answers (id -> output line)."""
    c, u = (rows.loc[np.where(pairs["clicked"].eq(x), pairs["x1"], pairs["x2"])].reset_index(drop=True)
            for x in ("X1", "X2"))
    groups = pairs["character_id"].to_numpy()
    pick = {k: np.array([answers[f"{k}:{p}"]["answers"]["pick"]["choice"] for p in pairs["pair"]])
            for k in ("with", "without")}
    arms = {"coin": np.full(len(pairs), 0.5),
            "B": _scored(c["pctr"].to_numpy(), u["pctr"].to_numpy()),
            "lookup": _scored(c["lookup"].to_numpy(), u["lookup"].to_numpy()),
            "creative_history": _scored(c["C14_smoothed_ctr"].to_numpy(), u["C14_smoothed_ctr"].to_numpy()),
            "jev_with_pctr": (pick["with"] == pairs["clicked"]).astype(float),
            "jev_without_pctr": (pick["without"] == pairs["clicked"]).astype(float)}
    seen_c, seen_u = c["C14_impressions"].gt(0).to_numpy(), u["C14_impressions"].gt(0).to_numpy()
    margin = np.abs(c["pctr"].to_numpy() - u["pctr"].to_numpy())
    subsets = {"all": np.ones(len(pairs), bool), "both_seen": seen_c & seen_u, "both_unseen": ~seen_c & ~seen_u,
               "mixed": seen_c != seen_u, "near_tie": margin < NEAR_TIE}
    exp1 = []
    for name, score in arms.items():
        row = {"arm": name, **{f"acc_{s}": float(score[m].mean()) for s, m in subsets.items()},
               "acc_ci": bootstrap_diff(groups, score, np.zeros(len(score)))[1:]}
        if name != "B":
            row["diff_vs_B"] = bootstrap_diff(groups, score, arms["B"])
            row["mcnemar_wins_losses_p"] = _mcnemar(score, arms["B"]) if name != "coin" else None
        exp1.append(row)
    swap = swap_sample(pairs)
    same = [answers[f"swap:{p}"]["answers"]["pick"]["choice"] != pick["with"][i]
            for i, p in zip(swap, pairs["pair"].iloc[swap])]

    # Experiment 2: on pairs with a nonzero margin, which score ranks B-correct pairs above B-wrong pairs?
    m = margin > 0
    y = arms["B"][m].astype(int)
    hi_is_c = c["pctr"].to_numpy() > u["pctr"].to_numpy()
    imp_c, imp_u = c["C14_impressions"].to_numpy(), u["C14_impressions"].to_numpy()
    X = np.column_stack([margin, seen_c & seen_u, ~seen_c & ~seen_u,
                         np.log1p(np.where(hi_is_c, imp_c, imp_u)), np.log1p(np.where(hi_is_c, imp_u, imp_c)),
                         np.abs(c["lookup"].to_numpy() - u["lookup"].to_numpy())])[m]
    logistic = cross_val_predict(LogisticRegression(max_iter=1000), X, y, groups=groups[m], cv=GroupKFold(5),
                                 method="predict_proba")[:, 1]
    gates = {"margin": margin[m], "logistic": logistic,
             "jev_gate": np.array([answers[f"with:{p}"]["answers"]["gate"]["noul"] for p in pairs["pair"][m]])}
    tiebreak = np.random.default_rng(1).random(m.sum())
    selective = {name: {f"{q}%": float(y[np.lexsort((tiebreak, -s))[:int(np.ceil(q / 100 * len(y)))]].mean())
                        for q in (100, 80, 60, 40, 20)} for name, s in gates.items()}
    calls = [answers[k] for k in answers]
    return {
        "pairs": len(pairs), "groups": int(pairs["context_group"].nunique()), "characters": int(pairs["character_id"].nunique()),
        "subset_sizes": {s: int(mask.sum()) for s, mask in subsets.items()},
        "ceiling": {s: ceiling(c["pctr"].to_numpy()[mask], u["pctr"].to_numpy()[mask]) for s, mask in subsets.items()},
        "experiment_1": exp1,
        "order_robustness": {"pairs": len(same), "same_ad_share": float(np.mean(same))},
        "experiment_2": {"pairs": int(m.sum()), "b_accuracy": float(y.mean()),
                         "auroc": {name: float(roc_auc_score(y, s)) for name, s in gates.items()},
                         "jev_minus_logistic": _auc_diff(groups[m], y, gates["jev_gate"], logistic),
                         "selective_accuracy": selective},
        "jev": {"calls": len(calls), "models": sorted({a["model"] for a in calls}),
                "input_tokens": int(sum(a["tokens_in"] for a in calls)),
                "call_seconds_sum": float(sum(a["seconds"] for a in calls)),
                "call_seconds_median": float(np.median([a["seconds"] for a in calls]))},
    }


def read_answers(path):
    """Successful Jev lines keyed by id; the last line per id wins."""
    answers = {}
    for row in map(json.loads, Path(path).read_text().splitlines()):
        answers[row["id"]] = row
    bad = [k for k, row in answers.items() if "error" in row]
    assert not bad, f"{len(bad)} ids still errored, rerun jev_ask.py: {bad[:3]}"
    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    pairs, rows = prepare(args.data)
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        write_states(pairs, rows, WORK / "states.jsonl")
        print("wrote", WORK / "states.jsonl")
    else:
        out = evaluate(pairs, rows, read_answers(ANSWERS))
        Path("reports/rank_e12.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps(out, indent=1, default=float))
