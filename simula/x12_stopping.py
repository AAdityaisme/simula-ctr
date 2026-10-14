"""x12 (Fable E3): can Jev read a validation curve and pick a better tree count than argmin for a drifted window?

Pre-registered before any test outcome:
- Curves: 10 seeds x feature sets A/B/C. Each seed draws 300,000 Oct 21-26 rows and fits 400 trees with that seed,
  recording train and Oct 27 loss per tree.
- Options, all read off the Oct 27 curve: argmin; argmin x0.8, x0.9, x1.1, x1.2; one_se (fewest trees within one
  standard error of the minimum, SE of the per-row loss at argmin); plateau_start (fewest trees after which the next
  50 trees improve Oct 27 loss by less than 0.0001); max (400).
- Jev state: the curve every 5 trees, the option table and a drift table (train / Oct 27 / Oct 28: rows, CTR,
  site-side, C20=-1, null-device and unseen-C14 shares). No test row enters a state.
- Jev's pick sets the tree count of a refit on 300,000 Oct 21-27 rows (same seed), calibrated on Oct 28 with B's
  half intercept shift and scored on Oct 29-30. Controls: argmin, one_se, 0.8 x argmin.
- Win: Jev's mean test loss over the 30 curves is at least 0.0003 below argmin, with a paired character-cluster
  interval excluding 0. Also reported: how often Jev picks argmin, and order stability with the options reversed.
Run: OMP_NUM_THREADS=3 uv run python -m simula.x12_stopping
"""
import json
import sys
from pathlib import Path

import numpy as np

from simula import xtrain
from simula.data import split
from simula.features import FEATURE_SETS

SEEDS = range(10)
TREES = 400
FACTORS = {"minus_20": 0.8, "minus_10": 0.9, "plus_10": 1.1, "plus_20": 1.2}
LABELS = {"argmin": "the tree count with the lowest Oct 27 validation loss",
          "minus_20": "20% fewer trees than argmin", "minus_10": "10% fewer trees than argmin",
          "plus_10": "10% more trees than argmin", "plus_20": "20% more trees than argmin",
          "one_se": "the fewest trees within one standard error of the minimum validation loss",
          "plateau_start": "the fewest trees after which the next 50 trees improve validation loss by under 0.0001",
          "max": "every tree fitted"}
INSTRUCTIONS = ("A click model was fit on Oct 21-26 and validated on Oct 27. It will now be refit on Oct 21-27 and "
                "serve Oct 29-30, after an intercept recalibration on Oct 28. The drift table shows how traffic "
                "moved between windows. Which tree count should the refit use to have the lowest log loss on the "
                "serving window?")


def options(valid, se):
    """Tree counts for each option, from a per-tree validation loss curve and the SE of the loss at its minimum."""
    best = int(np.argmin(valid)) + 1
    gains = np.array([valid[k] - valid[k:k + 50].min() for k in range(len(valid))])
    return {"argmin": best, **{name: max(1, round(best * f)) for name, f in FACTORS.items()},
            "one_se": int(np.argmax(valid <= valid.min() + se)) + 1,
            "plateau_start": int(np.argmax(gains < 1e-4)) + 1, "max": len(valid)}


def _mix(df, seen):
    return {"rows": len(df), "ctr": round(float(df["click"].mean()), 4),
            "site_side_share": round(float(df["is_site_side"].mean()), 3),
            "c20_minus1_share": round(float(df["c20_is_sentinel"].mean()), 3),
            "null_device_share": round(float(df["is_null_device"].mean()), 3),
            "unseen_c14_share": round(float((~df["C14"].isin(seen)).mean()), 3)}


def state(curve, opts, drift):
    """Jev state: the curve every 5 trees, the options with their tree counts, and the drift table."""
    every5 = range(4, len(curve["valid"]), 5)
    return {"loss_curve_every_5_trees": [{"trees": k + 1, "train_loss": round(float(curve["train"][k]), 5),
                                          "validation_loss": round(float(curve["valid"][k]), 5)} for k in every5],
            "options": {name: {"trees": n, "validation_loss": round(float(curve["valid"][n - 1]), 5)}
                        for name, n in opts.items() if n <= len(curve["valid"])} |
                       {name: {"trees": n, "validation_loss": "beyond the fitted curve"}
                        for name, n in opts.items() if n > len(curve["valid"])},
            "drift": drift}


def question(opts, reverse=False):
    """The Choice question; option text carries the tree count."""
    names = list(LABELS)[::-1] if reverse else list(LABELS)
    return {"tree_count": {"type": "choice", "instructions": INSTRUCTIONS,
                           "criteria": {n: f"{LABELS[n]} ({opts[n]} trees)" for n in names}}}


def curve_items(df, keys):
    """Fit the Oct 27 curve for each (feature set, seed) and return the curve records and Jev input items."""
    dev, select, cal = (split(df, w) for w in ("dev_fit", "select", "calibrate"))
    curves, items = [], []
    for fs, seed in keys:
        train = xtrain.subsample(dev, seed)
        model = xtrain.fit(train, FEATURE_SETS[fs], TREES, seed, valid=select)
        best = int(np.argmin(model[2]["valid"])) + 1
        loss = xtrain.row_loss(select["click"], xtrain.raw(model[:2], select, best))
        opts = options(model[2]["valid"], loss.std() / np.sqrt(len(loss)))
        seen = set(train["C14"])
        drift = {"train_oct21_26": _mix(train, seen), "validation_oct27": _mix(select, seen),
                 "calibration_oct28": _mix(cal, seen)}
        key = f"{fs}:{seed}"
        curves.append({"key": key, "fs": fs, "seed": seed, "options": opts})
        s = state(model[2], opts, drift)
        items += [{"id": key, "state": s, "questions": question(opts)},
                  {"id": f"rev:{key}", "state": s, "questions": question(opts, reverse=True)}]
        print(key, opts, flush=True)
    return curves, items


def run():
    df = xtrain.frame()
    refit, cal, test = (split(df, w) for w in ("refit", "calibrate", "test"))
    y, groups = test["click"].to_numpy(), test["character_id"].to_numpy()
    curves, items = curve_items(df, [(fs, seed) for fs in "ABC" for seed in SEEDS])
    xtrain.WORK.mkdir(parents=True, exist_ok=True)
    xtrain.WORK.joinpath("x12_example_state.json").write_text(json.dumps(items[0], indent=1))
    answers = xtrain.ask_jev(items, "x12")

    arms = ["jev", *LABELS]
    losses = {a: np.zeros(len(test)) for a in arms}
    per_curve = []
    for c in curves:
        pick = answers[c["key"]]["answers"]["tree_count"]["choice"]
        c["jev_pick"], c["jev_pick_reversed"] = pick, answers[f"rev:{c['key']}"]["answers"]["tree_count"]["choice"]
        model = xtrain.fit(xtrain.subsample(refit, c["seed"]), FEATURE_SETS[c["fs"]], max(c["options"].values()),
                           c["seed"])[:2]
        by_trees = {n: xtrain.row_loss(y, xtrain.calibrated(model, cal, test, n)) for n in set(c["options"].values())}
        row = {**c, "test_log_loss": {name: float(by_trees[n].mean()) for name, n in c["options"].items()}}
        row["test_log_loss"]["jev"] = row["test_log_loss"][pick]
        for name, n in c["options"].items():
            losses[name] += by_trees[n] / len(curves)
        losses["jev"] += by_trees[c["options"][pick]] / len(curves)
        per_curve.append(row)
        print(c["key"], pick, {k: round(v, 5) for k, v in row["test_log_loss"].items()}, flush=True)

    def seed_ci(a, b):
        d = np.array([r["test_log_loss"][a] - r["test_log_loss"][b] for r in per_curve])
        boots = np.random.default_rng(0).choice(d, (2000, len(d))).mean(axis=1)
        return [float(d.mean()), *map(float, np.percentile(boots, [2.5, 97.5]))]

    picks = [r["jev_pick"] for r in per_curve]
    return {
        "curves": len(per_curve), "test_rows": len(test),
        "mean_test_log_loss": {a: float(losses[a].mean()) for a in arms},
        "comparisons": [{"candidate": a, "baseline": b, "char_ci": xtrain.bootstrap_diff(groups, losses[a], losses[b]),
                         "curve_ci": seed_ci(a, b)}
                        for a, b in [("jev", "argmin"), ("jev", "one_se"), ("jev", "minus_20"),
                                     ("one_se", "argmin"), ("minus_20", "argmin")]],
        "jev_pick_counts": {p: picks.count(p) for p in sorted(set(picks))},
        "jev_picks_argmin_tree_count": sum(r["options"][r["jev_pick"]] == r["options"]["argmin"] for r in per_curve),
        "order_stability": float(np.mean([r["jev_pick"] == r["jev_pick_reversed"] for r in per_curve])),
        "jev": xtrain.jev_cost({k: v for k, v in answers.items()}),
        "per_curve": per_curve,
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["smoke"]:
        _, items = curve_items(xtrain.frame(), [("B", 0), ("A", 1), ("C", 2)])
        for k, a in xtrain.ask_jev(items[::2], "x12_smoke").items():
            print(k, a["tokens_in"], a["answers"])
        sys.exit()
    out = run()
    Path("reports/x12_stopping.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_curve"}, indent=1, default=float))
