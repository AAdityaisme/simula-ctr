"""x01: audit the shipped fatigue discount, with Jev as a world-knowledge prior (Fable E8).

k = 1 + earlier impressions of the same C14, the same day, to the same (device_ip, device_model), counting only
strictly earlier hours. The refit window (Oct 21-27) gives the fitted relative-CTR curve over k in {1, 2, 3, 4+}.
Jev proposes its curve from the assignment's goal-2 paragraph and the schema alone, before seeing any data.
Arms on test multiply B's calibrated pCTR by each curve; one more arm retrains B with the counter. Run:
  uv run python -m simula.x01_fatigue states       # Jev input JSONL, no data in it
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x01_fatigue report       # reports/x01_fatigue.json

Pre-registered: Jev's ratio per bucket is the rung-midpoint expectation, averaged over 3 asks. Arms: B, and B times the
shipped, Jev and fitted curves (raw product is primary; a refit Oct 28 intercept is a sensitivity), plus B_sub and
B_sub_plus_counter retrained on the same 300,000-row subsamples. Win for Jev: every ratio within 0.05 of the fitted
one, and B_x_jev beats B_x_shipped on repeat-slice (k >= 2) log loss with the character-bootstrap interval below 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from simula import train
from simula.data import add_flags, load, split
from simula.features import CONTRACT, FEATURE_SETS, fit_encoder, transform
from simula.jev_ctr import _row_loss, bootstrap_diff
from simula.rank_pairs import _auc_diff

FEATURE = "prior_impressions_today"
# ponytail: extends the shared contract at import instead of editing features.py on this experiment branch
CONTRACT["numeric"] = CONTRACT["numeric"] + [FEATURE]
ASSIGNMENT = Path.home() / "Desktop/Simula/ml-takehome/ASSIGNMENT.md"
WORK = Path.home() / "simula-local/rank-jev-work/exp-x-fatigue-stack"
ANSWERS = Path("reports/x01_fatigue_jev_answers.jsonl")
BUCKETS = (2, 3, 4)  # 4 means 4 or more; k=1 is the reference, ratio 1 by definition
RUNGS = [f"{r} times the first-impression CTR" for r in ("below 0.45", "0.45 to 0.6", "0.6 to 0.75", "0.75 to 0.9",
                                                         "0.9 or more")]
MIDPOINTS = np.array([0.375, 0.525, 0.675, 0.825, 0.95])
ORDINAL = {2: "2nd", 3: "3rd", 4: "4th or later"}
QUESTIONS = {f"ratio_k{k}": {"type": "score", "criteria": RUNGS, "instructions":
             f"Consider the {ORDINAL[k]} time the same device is shown the same ad creative (C14) on the same day. "
             "What is its click-through rate relative to the click-through rate of that device's first impression "
             "of that creative that day?"} for k in BUCKETS}
N_ASKS = 3
SHRINK = 0.5  # bundle/B's calibration recipe: one day (Oct 28), half shift
N_SUB = 300_000
TOLERANCE = 0.05


def repeat_count(df):
    """k per row: 1 + same-day impressions of the same C14 to the same (device_ip, device_model) in earlier hours.

    Rows in the same hour have no order, so they never count each other."""
    key = ["day", "device_ip", "device_model", "C14"]
    per_hour = df.groupby(key + ["hour"], dropna=False).size().rename("n")
    earlier = (per_hour.groupby(level=key, dropna=False).cumsum() - per_hour).rename("earlier")
    return 1 + df.join(earlier, on=key + ["hour"])["earlier"].astype(int)


def bucket(k):
    """k capped at 4, the 4+ bucket."""
    return np.minimum(np.asarray(k), 4)


def fitted_curve(rows):
    """CTR at each k bucket divided by CTR at k=1, with the counts behind it."""
    g = rows.groupby(bucket(rows["k"]))["click"].agg(["mean", "size"])
    return {int(b): {"ratio": float(g.loc[b, "mean"] / g.loc[1, "mean"]), "ctr": float(g.loc[b, "mean"]),
                     "rows": int(g.loc[b, "size"])} for b in g.index}


def shipped_multiplier(k):
    """`simula/rank.py`: utility = pCTR / (1 + 0.5 x times_seen), times_seen = k - 1, uncapped."""
    return 1 / (1 + 0.5 * (np.asarray(k) - 1))


def curve_multiplier(k, ratios):
    """Multiplier from a {2: r2, 3: r3, 4: r4plus} curve; k=1 gets 1."""
    return np.select([bucket(k) == b for b in BUCKETS], [ratios[b] for b in BUCKETS], 1.0)


def jev_curve(answers):
    """Expected ratio per bucket under Jev's rung probabilities (rung midpoints), averaged over the asks."""
    def expected(a):
        probs = a["probabilities"]
        return sum(p * MIDPOINTS[int(i)] for i, p in probs.items()) / sum(probs.values())

    return {k: float(np.mean([expected(a["answers"][f"ratio_k{k}"]) for a in answers])) for k in BUCKETS}


def jev_state(assignment_text):
    """The goal-2 paragraph and the CSV schema, verbatim, plus the definition of k. No data."""
    goal2 = assignment_text.split("#### 2. Candidate Ranking")[1].split("#### 3.")[0].strip()
    schema = assignment_text.split("### What you get")[1].split("### Goals")[0].strip()
    return {"assignment_goal_2_candidate_ranking": goal2, "data_schema": schema,
            "definition": "k counts how many times a device (device_ip plus device_model) has been shown the same ad "
                          "creative (C14) so far that day, including the current impression; k=1 is the first."}


def retrain(df, features, seed=0):
    """B's recipe on a subsample: early stopping on Oct 27, refit on Oct 21-27, half intercept shift on Oct 28."""
    dev_fit, select, refit_rows, cal, test = (split(df, w) for w in ("dev_fit", "select", "refit", "calibrate", "test"))
    dev_fit = dev_fit.sample(min(N_SUB, len(dev_fit)), random_state=seed)
    refit_rows = refit_rows.sample(min(N_SUB, len(refit_rows)), random_state=seed)
    encoder = fit_encoder(dev_fit, features)
    _, rounds = train.fit(dev_fit, select, features, encoder)
    encoder = fit_encoder(refit_rows, features)
    booster = train.refit(refit_rows, features, encoder, rounds)
    delta = train.calibrate(booster.predict(transform(cal, encoder)), cal["click"], SHRINK)
    return train.apply_calibration(booster.predict(transform(test, encoder)), delta), int(rounds)


def recalibrated(p_cal, y_cal, p_test):
    """Sensitivity arm: refit the half intercept shift on Oct 28 after the multiplier, as B's recipe does."""
    return train.apply_calibration(p_test, train.calibrate(p_cal, y_cal, SHRINK))


def prepare(data_dir):
    """All rows with flags and k."""
    df = add_flags(load(data_dir))
    df["k"] = repeat_count(df)
    df[FEATURE] = df["k"] - 1
    return df


def evaluate(df, answers):
    """Curves, every arm's test metrics, and intervals against B and between the curves."""
    refit, cal, test = (split(df, w) for w in ("refit", "calibrate", "test"))
    fitted = fitted_curve(refit)
    curves = {"jev": jev_curve(answers), "fitted": {k: fitted[k]["ratio"] for k in BUCKETS}}
    bundle = train.load_bundle("bundle/B")
    p_b, p_b_cal = train.predict(bundle, test), train.predict(bundle, cal)
    mult = {"shipped": shipped_multiplier, **{n: (lambda k, c=c: curve_multiplier(k, c)) for n, c in curves.items()}}
    arms = {"B": p_b, **{f"B_x_{n}": p_b * m(test["k"]) for n, m in mult.items()}}
    arms.update({f"B_x_{n}_recal": recalibrated(p_b_cal * m(cal["k"]), cal["click"], p_b * m(test["k"]))
                 for n, m in mult.items()})
    rounds = {}
    for name, features in (("B_sub", FEATURE_SETS["B"]), ("B_sub_plus_counter", FEATURE_SETS["B"] + [FEATURE])):
        arms[name], rounds[name] = retrain(df, features)
        print(name, rounds[name], flush=True)

    y, groups = test["click"].to_numpy(), test["character_id"].to_numpy()
    repeat = test["k"].ge(2).to_numpy()
    slices = {"all": np.ones(len(test), bool), "repeat": repeat}
    losses = {n: _row_loss(y, p) for n, p in arms.items()}
    table = [{"arm": n, **{f"{s}_{m}": f(y[mask], p[mask]) for s, mask in slices.items()
                           for m, f in (("log_loss", log_loss), ("auc", roc_auc_score), ("mean_p", lambda _, q: q.mean()))}}
             for n, p in arms.items()]
    pairs = [(n, "B") for n in arms if n != "B"] + [("B_x_jev", "B_x_shipped"), ("B_x_jev_recal", "B_x_shipped_recal"),
                                                     ("B_sub_plus_counter", "B_sub"), ("B_x_fitted", "B_x_shipped")]
    diffs = []
    for a, b in pairs:
        for s, mask in slices.items():
            row = {"a": a, "b": b, "slice": s,
                   "log_loss_diff": bootstrap_diff(groups[mask], losses[a][mask], losses[b][mask])}
            if not a.endswith("_recal"):  # an intercept shift leaves AUC unchanged
                row["auc_diff"] = _auc_diff(groups[mask], y[mask], arms[a][mask], arms[b][mask])
            diffs.append(row)
            print(row, flush=True)
    jev_vs_shipped = next(d for d in diffs if (d["a"], d["b"], d["slice"]) == ("B_x_jev", "B_x_shipped", "repeat"))
    within = {k: abs(curves["jev"][k] - curves["fitted"][k]) <= TOLERANCE for k in BUCKETS}
    calls = list(answers)
    return {
        "k_definition": "1 + same-day same-C14 impressions to the same (device_ip, device_model) in strictly earlier hours",
        "fitted_curve_refit": fitted, "test_curve": fitted_curve(test), "calibrate_curve": fitted_curve(cal),
        "curves": {**curves, "shipped": {k: float(shipped_multiplier(k)) for k in BUCKETS}},
        "repeat_share": {"refit": float(refit["k"].ge(2).mean()), "test": float(repeat.mean())},
        "test_rows": len(test), "repeat_rows": int(repeat.sum()), "retrain_rounds": rounds, "subsample_rows": N_SUB,
        "arms": table, "diffs": diffs,
        "win": {"ratios_within_0.05": {str(k): bool(v) for k, v in within.items()},
                "jev_beats_shipped_repeat": jev_vs_shipped["log_loss_diff"][2] < 0,
                "verdict": "win" if all(within.values()) and jev_vs_shipped["log_loss_diff"][2] < 0 else "null"},
        "jev": {"calls": len(calls), "models": sorted({a["model"] for a in calls}),
                "input_tokens": int(sum(a["tokens_in"] for a in calls)),
                "call_seconds_sum": float(sum(a["seconds"] for a in calls)),
                "raw": [{k: a["answers"][f"ratio_k{k}"] for k in BUCKETS} for a in calls]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        state = jev_state(ASSIGNMENT.read_text())
        with open(WORK / "x01_states.jsonl", "w") as f:
            for i in range(1, N_ASKS + 1):
                f.write(json.dumps({"id": f"ask:{i}", "state": state, "questions": QUESTIONS}) + "\n")
        print("wrote", WORK / "x01_states.jsonl")
    else:
        answers = list({a["id"]: a for a in map(json.loads, ANSWERS.read_text().splitlines())}.values())
        assert not any("error" in a for a in answers), "rerun jev_ask.py"
        out = evaluate(prepare(args.data), answers)
        Path("reports/x01_fatigue.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(pd.DataFrame(out["arms"]).round(5).to_string(index=False))
