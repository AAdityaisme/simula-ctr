"""x11: confidence as a surprise forecaster for the exploration budget. For each creative (C14) unseen in refit with
>= 100 test rows, Jev scores how far its realized CTR will land from the e4 table prior, reading e3's attribute
statistics for the creative's first test row. Exploration would spend more on the creatives it flags; offline only
the forecast is tested.

Target: |realized CTR - mean e4 table prior over the creative's test rows|. Jev's forecast is the expected rung
index under its probabilities. Controls, from the same state: the standard deviation of the nine smoothed CTRs,
-log(1 + cell n) for the e4 cell (genre, C18, app_category), and their rank sum; the best of the three is the bar.
Realized noise shrinks with the creative's test rows, so the headline is the partial Spearman given log(test rows).
Pre-declared win (Fable E6): Jev's partial Spearman >= 0.3 and above the best control's. Run:
  uv run python -m simula.x11_surprise states   # Jev input JSONL outside the checkout (asked twice)
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x11_surprise report   # reports/x11_surprise.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from simula.data import add_flags, load, split
from simula.jev_prior import CELL, count_tables, table_prior
from simula.rank_e3 import ATTRIBUTES, CELLS, _candidate
from simula.rank_pairs import add_history, history

MIN_ROWS = 100
RUNGS = ["within 1 point of it", "1 to 3 points away", "3 to 6 points away", "6 to 10 points away",
         "more than 10 points away"]
QUESTION = {"uncertainty": {"type": "score", "criteria": RUNGS, "instructions":
    "This ad creative is new. Its attributes' history suggests a click-through rate. How far could its real "
    "click-through rate end up from that suggestion, in percentage points? Use the evidence in the state."}}
WORK = Path.home() / "simula-local/rank-jev-work/exp-x-novelty/x11"
ANSWERS = Path("reports/x11_surprise_jev_answers.jsonl")


def creatives(data_dir):
    """One row per qualifying creative: its first test row's attribute stats, cell n, test rows and the target."""
    df = add_flags(load(data_dir))
    refit, test = split(df, "refit"), split(df, "test")
    rows = test[~test["C14"].isin(set(refit["C14"]))].sort_values("hour", kind="stable")
    rows = rows[rows.groupby("C14")["C14"].transform("size").ge(MIN_ROWS)].copy()
    tables = count_tables(refit)
    keys = list(rows[CELL].itertuples(index=False, name=None))
    rows["prior"] = [table_prior(k, tables) if k in tables["cell"] else tables["global_ctr"] for k in keys]
    rows["cell_n"] = [tables["cell"].get(k, (0, 0))[0] for k in keys]
    rows = add_history(rows, history(refit), [*CELLS, *ATTRIBUTES])
    first = rows.drop_duplicates("C14").set_index("C14")
    out = rows.groupby("C14").agg(test_rows=("click", "size"), ctr=("click", "mean"), prior=("prior", "mean"))
    out["target"] = (out["ctr"] - out["prior"]).abs()
    out["cell_n"] = first["cell_n"]
    out["state"] = [state(first.loc[c]) for c in out.index]
    return out


def state(r):
    """e3's attribute-stats candidate state plus the e4 cell's refit impressions. No pCTR, C14, C17 or click."""
    return {"candidate": _candidate(r), "cell_genre_x_C18_x_app_category": {"impressions": int(r["cell_n"])}}


def controls(states):
    """Non-Jev forecasts from the same states: higher means more expected surprise."""
    sd = np.array([np.std([s["candidate"][k]["smoothed_ctr"] for k in ATTRIBUTES + CELLS]) for s in states])
    thin = -np.log1p([s["cell_genre_x_C18_x_app_category"]["impressions"] for s in states])
    return {"attribute_sd": sd, "cell_thinness": thin, "rank_sum": rankdata(sd) + rankdata(thin)}


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the linear effect of rank(z) from each rank."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    resid = [r - np.polyval(np.polyfit(rz, r, 1), rz) for r in (rx, ry)]
    return float(np.corrcoef(*resid)[0, 1])


def expected_rung(answer):
    """Expected rung index (0-4) under Jev's probabilities, normalized by their sum."""
    probs = answer["answers"]["uncertainty"]["probabilities"]
    return sum(p * int(i) for i, p in probs.items()) / sum(probs.values())


def read_answers(path, prefix):
    last = {row["id"]: row for row in map(json.loads, Path(path).read_text().splitlines())}
    return {int(i.split(":")[1]): row for i, row in last.items() if i.startswith(prefix) and "answers" in row}


def evaluate(table, answers, n_boot=1000):
    first, again = read_answers(answers, "a:"), read_answers(answers, "b:")
    jev = np.array([expected_rung(first[c]) for c in table.index])
    arms = {"jev": jev, **controls(list(table["state"]))}
    y, z = table["target"].to_numpy(), np.log(table["test_rows"].to_numpy())
    stats = {name: {"spearman": float(spearmanr(x, y).statistic), "partial_spearman_given_log_rows":
                    partial_spearman(x, y, z)} for name, x in arms.items()}
    best = max((n for n in arms if n != "jev"), key=lambda n: stats[n]["partial_spearman_given_log_rows"])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):  # resample creatives
        i = rng.integers(0, len(y), len(y))
        j, b = (partial_spearman(arms[n][i], y[i], z[i]) for n in ("jev", best))
        boots.append((j, j - b))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    jp = stats["jev"]["partial_spearman_given_log_rows"]
    calls = list(first.values()) + list(again.values())
    rechecked = [c for c in table.index if c in again]
    return {"creatives": len(table), "test_rows": int(table["test_rows"].sum()),
            "target_summary": table["target"].describe().to_dict(),
            "target_vs_log_rows_spearman": float(spearmanr(y, z).statistic),
            "arms": stats, "best_control": best,
            "jev_partial_ci": [float(lo[0]), float(hi[0])],
            "jev_minus_best_partial": float(jp - stats[best]["partial_spearman_given_log_rows"]),
            "jev_minus_best_partial_ci": [float(lo[1]), float(hi[1])],
            "win": bool(jp >= 0.3 and jp > stats[best]["partial_spearman_given_log_rows"]),
            "jev_rung_counts": pd.Series([first[c]["answers"]["uncertainty"]["score"] for c in table.index])
                                 .value_counts().sort_index().to_dict(),
            "recheck": {"creatives": len(rechecked), "mean_abs_change_expected_rung": float(np.mean(
                [abs(expected_rung(first[c]) - expected_rung(again[c])) for c in rechecked]))},
            "jev": {"calls": len(calls), "input_tokens": int(sum(c["tokens_in"] for c in calls)),
                    "models": sorted({c["model"] for c in calls})}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    table = creatives(args.data)
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        with (WORK / "states.jsonl").open("w") as f:
            for ask in ("a", "b"):  # b re-asks every state once, for stability
                for c, s in table["state"].items():
                    f.write(json.dumps({"id": f"{ask}:{c}", "state": s, "questions": QUESTION}) + "\n")
        print(f"{len(table)} creatives, {table['test_rows'].sum()} rows -> {WORK}")
    else:
        out = evaluate(table, ANSWERS)
        Path("reports/x11_surprise.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({k: out[k] for k in out if k != "target_summary"}, indent=1, default=float))
