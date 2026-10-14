"""x10: sequential graduation. Oct 29-30 is replayed hour by hour over the creatives (C14) unseen in refit. When a
creative's own impressions in strictly earlier hours first reach a checkpoint n in {0, 10, 25, 50, 100, 250}, Jev
picks the source that scores its rows until the next checkpoint: the e4 cell prior, e4's fixed blend, B, or the
creative's running CTR. Each source is recomputed every hour from strictly earlier hours, so no row sees a click
from its own hour.

Controls: e4's fixed blend (k=500, b=0.1; n=0 for every unseen creative, so 0.9 pB + 0.1 prior) and a Beta posterior
(K + 20 prior) / (N + 20) around the e4 table prior. Pre-declared win (Fable E5): Jev's policy beats the Beta
posterior by >= 0.0005 log loss on the unseen-C14 rows, character-bootstrap interval excluding 0. Run:
  uv run python -m simula.x10_graduation states   # Jev input JSONL outside the checkout
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x10_graduation report   # reports/x10_graduation.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simula.data import add_flags, load, split
from simula.jev_ctr import _row_loss, bootstrap_diff
from simula.jev_prior import CELL, blend, count_tables, table_prior
from simula.rank_e3 import ATTRIBUTES, CELLS, _candidate
from simula.rank_pairs import add_history, history
from simula.train import predict

CHECKPOINTS = np.array([0, 10, 25, 50, 100, 250])
SOURCES = ["prior", "blend", "B", "running_ctr"]
M, K_E4, B_E4 = 20, 500, 0.1
QUESTION = {"source": {"type": "choice", "instructions":
    "This ad creative is new: it had no impressions before this period. Which estimate should score its next "
    "impressions, until it reaches the next checkpoint? Use the evidence in the state.",
    "criteria": {"prior": "the cell prior from similar ads' history",
                 "blend": "the click model, nudged slightly toward the cell prior",
                 "B": "the click model's prediction",
                 "running_ctr": "this creative's own click rate so far"}}}
WORK = Path.home() / "simula-local/rank-jev-work/exp-x-novelty/x10"
ANSWERS = Path("reports/x10_graduation_jev_answers.jsonl")


def running(rows):
    """Per (C14, hour): impressions N, clicks K and hours seen in strictly earlier hours, plus the sums of pB and
    prior over those earlier rows and the means over the hour's own rows (features only)."""
    g = (rows.groupby(["C14", "hour"]).agg(n=("click", "size"), k=("click", "sum"), pb=("p_b", "sum"),
                                           pr=("prior", "sum")).reset_index().sort_values(["C14", "hour"]))
    by = g.groupby("C14")
    for col, out in (("n", "N"), ("k", "K"), ("pb", "pb_before"), ("pr", "prior_before")):
        g[out] = by[col].cumsum() - g[col]
    g["hours_seen"] = by.cumcount()
    g["pb_now"], g["prior_now"] = g["pb"] / g["n"], g["pr"] / g["n"]
    g["level"] = CHECKPOINTS[np.searchsorted(CHECKPOINTS, g["N"], side="right") - 1]
    return g.drop(columns=["n", "k", "pb", "pr"])


def sources(rows):
    """Every source's prediction per row, from rows annotated with p_b, prior, N and K."""
    n, k, prior = rows["N"].to_numpy(float), rows["K"].to_numpy(float), rows["prior"].to_numpy()
    return {"prior": prior, "blend": blend(rows["p_b"].to_numpy(), prior, np.zeros(len(rows)), K_E4, B_E4),
            "B": rows["p_b"].to_numpy(), "running_ctr": (k + 0.5) / (n + 1), "beta": (k + M * prior) / (n + M)}


def prepare(data_dir):
    """Unseen-C14 test rows in time order with pB, the e4 table prior, running counts and attribute history."""
    df = add_flags(load(data_dir))
    refit, test = split(df, "refit"), split(df, "test")
    rows = test[~test["C14"].isin(set(refit["C14"]))].sort_values("hour", kind="stable").copy()
    tables = count_tables(refit)
    priors = {cell: table_prior(cell, tables) for cell in tables["cell"]}
    rows["prior"] = [priors.get(c, tables["global_ctr"]) for c in rows[CELL].itertuples(index=False, name=None)]
    rows["p_b"] = predict("bundle/B", rows)
    rows = add_history(rows, history(refit), [*CELLS, *ATTRIBUTES])
    run = running(rows)
    return rows.join(run.set_index(["C14", "hour"]), on=["C14", "hour"])


def checkpoints(rows):
    """One row per (creative, checkpoint): the first row of its first hour at that level (rows are in time order)."""
    return rows.drop_duplicates(["C14", "level"]).reset_index(drop=True)


def state(r):
    """Jev's state at one checkpoint: the creative's own earlier record and what each source would say now."""
    n = int(r["N"])
    base = "earlier impressions" if n else "this hour's impressions (no clicks seen yet)"
    pb = r["pb_before"] / n if n else r["pb_now"]
    prior = r["prior_before"] / n if n else r["prior_now"]
    later = CHECKPOINTS[CHECKPOINTS > n]
    return {"checkpoint": int(r["level"]), "next_checkpoint": int(later[0]) if len(later) else None,
            "creative_so_far": {"impressions": n, "clicks": int(r["K"]), "hours_seen": int(r["hours_seen"])},
            "estimates": {"averaged_over": base, "prior": round(float(prior), 4),
                          "blend": round(float(0.9 * pb + 0.1 * prior), 4), "B": round(float(pb), 4),
                          "running_ctr": round(float((r["K"] + 0.5) / (n + 1)), 4) if n else None},
            "attributes": _candidate(r)}


def write_states(points, path):
    """One jev_ask.py input line per (creative, checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for _, r in points.iterrows():
            f.write(json.dumps({"id": f"{r['C14']}:{int(r['level'])}", "state": state(r), "questions": QUESTION})
                    + "\n")


def read_answers(path):
    """{(C14, level): source}; a repeated id keeps its last line, error lines are dropped."""
    last = {row["id"]: row for row in map(json.loads, Path(path).read_text().splitlines())}
    return {(int(i.split(":")[0]), int(i.split(":")[1])): row["answers"]["source"]["choice"]
            for i, row in last.items() if "answers" in row}, list(last.values())


def policy(rows, choices):
    """Jev's arm: each row takes the source Jev chose at its creative's current checkpoint."""
    src = sources(rows)
    picked = np.array([choices[k] for k in zip(rows["C14"], rows["level"])])
    return np.select([picked == s for s in SOURCES], [src[s] for s in SOURCES]), picked


def evaluate(rows, answers):
    choices, calls = read_answers(answers)
    preds = sources(rows)
    preds["jev"], picked = policy(rows, choices)
    y, chars = rows["click"].to_numpy(), rows["character_id"].to_numpy()
    loss = {name: _row_loss(y, p) for name, p in preds.items()}
    comparisons = {f"jev_vs_{b}": dict(zip(("diff", "ci_lo", "ci_hi"), bootstrap_diff(chars, loss["jev"], loss[b])))
                   for b in ("beta", "blend", "B")}
    comparisons["beta_vs_blend"] = dict(zip(("diff", "ci_lo", "ci_hi"),
                                            bootstrap_diff(chars, loss["beta"], loss["blend"])))
    by_level = []
    for level in CHECKPOINTS:
        m = rows["level"].eq(level).to_numpy()
        by_level.append({"checkpoint": int(level), "rows": int(m.sum()), "creatives": int(rows.loc[m, "C14"].nunique()),
                         "ctr": float(y[m].mean()),
                         **{f"loss_{name}": float(loss[name][m].mean()) for name in preds},
                         "jev_choices": pd.Series(picked[m]).value_counts().to_dict()})
    ok = [c for c in calls if "answers" in c]
    point_choices = pd.Series(list(choices.values()))
    jc = comparisons["jev_vs_beta"]
    return {"rows": len(rows), "creatives": int(rows["C14"].nunique()), "ctr": float(y.mean()),
            "checkpoint_calls": len(choices), "log_loss": {name: float(v.mean()) for name, v in loss.items()},
            "comparisons": comparisons, "by_checkpoint": by_level,
            "jev_choices_per_checkpoint_call": point_choices.value_counts().to_dict(),
            "win": bool(jc["diff"] <= -0.0005 and jc["ci_hi"] < 0),
            "jev": {"calls": len(calls), "errors": len(calls) - len(ok), "attempts": int(sum(c["attempts"] for c in calls)),
                    "models": sorted({c["model"] for c in ok}), "input_tokens": int(sum(c["tokens_in"] for c in ok)),
                    "call_seconds_median": float(np.median([c["seconds"] for c in ok]))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    rows = prepare(args.data)
    if args.step == "states":
        points = checkpoints(rows)
        write_states(points, WORK / "states.jsonl")
        print(f"{len(rows)} rows, {rows['C14'].nunique()} creatives, {len(points)} checkpoint calls -> {WORK}")
        print(points["level"].value_counts().sort_index().to_dict())
    else:
        out = evaluate(rows, ANSWERS)
        Path("reports/x10_graduation.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({k: out[k] for k in ("log_loss", "comparisons", "jev_choices_per_checkpoint_call", "win")},
                         indent=1))
