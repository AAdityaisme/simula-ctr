"""Experiment 4: can a per-cell CTR prior that Jev writes offline help brand-new creatives, at the cost of a dict
lookup in serving? A cell is (genre, C18, app_category). Model B routes an unseen C14 to its missing branch; here
its pCTR is blended toward the cell prior, less so as the creative gathers refit impressions.

  uv run python -m simula.jev_prior states   # writes Jev input JSONL outside the checkout
  (run jev_ask.py on it, answers to reports/rank_e4_jev_answers.jsonl)
  uv run python -m simula.jev_prior          # evaluates B, B + table prior, B + Jev prior
"""
import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from simula.data import add_flags, load, split
from simula.jev_ctr import _row_loss, bootstrap_diff
from simula.train import predict

CELL = ["genre", "C18", "app_category"]
NEIGHBOURS = {"genre_x_C18": ["genre", "C18"], "app_category_x_C18": ["app_category", "C18"],
              "genre": ["genre"], "C18": ["C18"]}
RUNGS = ["under 5%", "5-10%", "10-15%", "15-25%", "over 25%"]
MIDPOINTS = np.array([0.025, 0.075, 0.125, 0.20, 0.30])
QUESTIONS = {"ctr": {"type": "score", "criteria": RUNGS, "instructions":
                     "What click-through rate should we expect for ads in this cell? "
                     "Use the cell's history and its neighbours."}}
KS, BS = (50, 500), (0.1, 0.25, 0.5)
WORK = Path.home() / "simula-local/rank-jev-work/exp-rank-e4"
ANSWERS = Path("reports/rank_e4_jev_answers.jsonl")


def _counts(rows, cols):
    grouped = rows.groupby(cols, dropna=False)["click"].agg(["count", "sum"])
    return {(k if isinstance(k, tuple) else (k,)): (int(n), int(c)) for k, (n, c) in grouped.iterrows()}


def count_tables(refit):
    """Impression and click counts per cell and per neighbour level, from refit-window rows only."""
    return {"cell": _counts(refit, CELL), **{name: _counts(refit, cols) for name, cols in NEIGHBOURS.items()},
            "global_ctr": float(refit["click"].mean())}


def cell_state(cell, tables):
    """The JSON object Jev reads for one (genre, C18, app_category) cell."""
    genre, c18, app = cell
    keys = {"genre_x_C18": (genre, c18), "app_category_x_C18": (app, c18), "genre": (genre,), "C18": (c18,)}

    def ic(counts):
        return {"impressions": counts[0], "clicks": counts[1]}

    return {"cell": {"genre": genre, "C18": str(c18), "app_category": app},
            "cell_history": ic(tables["cell"][cell]),
            "neighbours": {name: ic(tables[name][key]) for name, key in keys.items()},
            "global_ctr": round(tables["global_ctr"], 4)}


def table_prior(cell, tables, m=20):
    """Hierarchical m-estimate: cell -> genre x C18 -> C18 -> global CTR, m pseudo-impressions per level."""
    genre, c18, _ = cell
    p = tables["global_ctr"]
    for n, c in (tables["C18"][(c18,)], tables["genre_x_C18"][(genre, c18)], tables["cell"][cell]):
        p = (c + m * p) / (n + m)
    return p


def jev_prior(answer):
    """Expected CTR under Jev's rung probabilities, normalized by their sum (rounding makes some sum to 0.99)."""
    probs = answer["answers"]["ctr"]["probabilities"]
    return float(sum(p * MIDPOINTS[int(i)] for i, p in probs.items()) / sum(probs.values()))


def blend(p_b, prior, n_c14, k, b):
    """p = (1 - b(1-w)) pB + b(1-w) prior, w = n/(n+k). A missing prior (NaN) leaves pB."""
    shift = b * (1 - n_c14 / (n_c14 + k))
    p = (1 - shift) * p_b + shift * prior
    return np.where(np.isnan(prior), p_b, p)


def write_states(tables, path, cells=None):
    """One jev_ask.py input line per cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for cell in cells or sorted(tables["cell"], key=str):
            f.write(json.dumps({"id": json.dumps([cell[0], str(cell[1]), cell[2]]), "state": cell_state(cell, tables),
                                "questions": QUESTIONS}) + "\n")


def read_answers(path):
    """{cell: prior} from a jev_ask.py output file. A repeated id keeps its last line; error lines are dropped."""
    last = {row["id"]: row for row in map(json.loads, Path(path).read_text().splitlines())}
    out = {}
    for key, row in last.items():
        if "answers" in row:
            genre, c18, app = json.loads(key)
            out[(genre, int(c18), app)] = jev_prior(row)
    return out


def _lookup(df, priors):
    return np.array([priors.get(key, np.nan) for key in df[CELL].itertuples(index=False, name=None)])


def _bench(priors, n_c14, rows, k, b, reps=1000):
    """Median microseconds for prior lookup + blend over one 50-candidate request."""
    cands = rows[CELL + ["C14"]].head(50).itertuples(index=False, name=None)
    cands = [((g, c18, app), c14) for g, c18, app, c14 in cands]
    p_b = list(np.linspace(0.1, 0.2, len(cands)))
    times = []
    for _ in range(reps):
        t = time.perf_counter_ns()
        out = []
        for ((cell, c14), p) in zip(cands, p_b):
            prior = priors.get(cell)
            if prior is None:
                out.append(p)
                continue
            n = n_c14.get(c14, 0)
            s = b * (1 - n / (n + k))
            out.append((1 - s) * p + s * prior)
        times.append(time.perf_counter_ns() - t)
    return float(np.median(times) / 1000)


def _metrics(y, p):
    return {"log_loss": float(log_loss(y, p)), "auc": float(roc_auc_score(y, p))}


def run(data_dir, answers=ANSWERS, recheck=None):
    df = add_flags(load(data_dir))
    refit, cal, test = (split(df, w) for w in ("refit", "calibrate", "test"))
    tables = count_tables(refit)
    sizes = pd.Series([n for n, _ in tables["cell"].values()])
    print(f"cells: {len(sizes)}, median rows {sizes.median():.0f}, under 50: {(sizes < 50).mean():.1%}")
    n_c14 = refit["C14"].value_counts().to_dict()
    priors = {"table": {cell: table_prior(cell, tables) for cell in tables["cell"]}}
    if Path(answers).exists():
        priors["jev"] = read_answers(answers)

    def frame(rows):
        return {"y": rows["click"].to_numpy(), "p_b": predict("bundle/B", rows),
                "n": rows["C14"].map(n_c14).fillna(0).to_numpy(float),
                "prior": {name: _lookup(rows, pr) for name, pr in priors.items()}}

    c, t = frame(cal), frame(test)
    unseen = t["n"] == 0
    chosen, preds = {}, {"B": t["p_b"]}
    for name in priors:  # k and b picked on Oct 28 only
        grid = {(k, b): log_loss(c["y"], blend(c["p_b"], c["prior"][name], c["n"], k, b)) for k, b in product(KS, BS)}
        k, b = min(grid, key=grid.get)
        chosen[name] = {"k": k, "b": b, "oct28_log_loss": float(grid[(k, b)]),
                        "oct28_grid": {f"k={kk},b={bb}": float(v) for (kk, bb), v in grid.items()}}
        preds[name] = blend(t["p_b"], t["prior"][name], t["n"], k, b)

    y = t["y"]
    arms = {name: {"all": _metrics(y, p), "unseen_c14": _metrics(y[unseen], p[unseen]),
                   "mean_p_unseen": float(p[unseen].mean())} for name, p in preds.items()}
    loss = {name: _row_loss(y, p) for name, p in preds.items()}
    pairs = [("jev", "table"), ("jev", "B"), ("table", "B")]
    pairs = [(a, b) for a, b in pairs if a in preds and b in preds]
    chars = test["character_id"].to_numpy()
    comparisons = [{"candidate": a, "baseline": b, "slice": s,
                    **dict(zip(("diff", "ci_lo", "ci_hi"), bootstrap_diff(chars[m], loss[a][m], loss[b][m])))}
                   for a, b in pairs for s, m in (("all", np.ones_like(unseen)), ("unseen_c14", unseen))]
    out = {"cells": len(sizes), "cell_median_rows": float(sizes.median()), "cells_under_50": float((sizes < 50).mean()),
           "test_rows": len(test), "unseen_c14_rows": int(unseen.sum()), "unseen_c14_mean_click": float(y[unseen].mean()),
           "test_rows_without_cell": {name: int(np.isnan(v).sum()) for name, v in t["prior"].items()},
           "chosen": chosen, "arms": arms, "comparisons": comparisons,
           "latency_us_50_candidates": {name: _bench(priors[name], n_c14, test, chosen[name]["k"], chosen[name]["b"])
                                        for name in priors}}
    if "jev" in priors:
        out["prior_summary"] = {name: pd.Series(list(pr.values())).describe().to_dict() for name, pr in priors.items()}
        common = [cell for cell in priors["jev"] if cell in priors["table"]]
        out["jev_table_prior_corr"] = float(np.corrcoef([priors["jev"][x] for x in common],
                                                        [priors["table"][x] for x in common])[0, 1])
    if recheck and Path(recheck).exists():
        again = read_answers(recheck)
        out["determinism"] = {"cells": len(again), "mean_abs_change": float(np.mean(
            [abs(again[x] - priors["jev"][x]) for x in again if x in priors["jev"]]))}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", nargs="?", default="eval", choices=["states", "eval"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    if args.step == "states":
        tables = count_tables(split(add_flags(load(args.data)), "refit"))
        write_states(tables, WORK / "states.jsonl")
        cells = sorted(tables["cell"], key=str)
        rng = np.random.default_rng(0)
        write_states(tables, WORK / "recheck_states.jsonl",
                     [cells[i] for i in sorted(rng.choice(len(cells), 100, replace=False))])
        print(f"{len(cells)} cells -> {WORK}")
    else:
        out = run(args.data, recheck=WORK / "recheck_answers.jsonl")
        Path("reports/rank_e4.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps({k: out[k] for k in ("chosen", "arms", "comparisons", "latency_us_50_candidates")}, indent=1))
