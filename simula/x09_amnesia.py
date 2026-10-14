"""x09: teach B coordinated amnesia. Jev picks which identifier block (C14, C17, C21) to hide together for 10% of
training creatives, so B rehearses brand-new creatives, whose C14 and usually C17 are both unseen at serving.

Every scheme refits B on its recipe (dev fit Oct 21-26, early stopping on Oct 27, refit Oct 21-27 with that tree
count, half intercept shift on Oct 28), each fit on the same 300,000-row subsample.

Controls, each searching 4 options: ordinary B; independent per-field masking and coordinated (dependency) masking,
each picked on Oct 27 as the lowest log loss on rows whose C14 is new to Oct 21-26, among options whose overall
Oct 27 loss is within 0.0005 of no masking. Jev's pick is the option with the highest mean probability over the
canonical call and 20 option-shuffled calls. Pre-declared win (Astra 2): Jev's scheme beats the strongest control
(lowest test unseen-C14 loss) by >= 0.001 unseen-C14 log loss, character-bootstrap interval below 0, and overall
test loss no more than 0.0005 worse than ordinary B. Run:
  uv run python -m simula.x09_amnesia states   # one dev fit, then Jev input JSONL outside the checkout
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x09_amnesia run      # 12 fits, reports/x09_amnesia.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from simula.data import add_flags, load, split
from simula.features import FEATURE_SETS, fit_encoder, transform
from simula.jev_ctr import _row_loss, bootstrap_diff
from simula.train import apply_calibration, calibrate, fit, predict, refit

FIELDS = ["C14", "C17", "C21"]
OPTIONS = {"none": [], "C14": ["C14"], "C14+C17": ["C14", "C17"], "C14+C17+C21": ["C14", "C17", "C21"]}
# (fields, coordinated). Independent C14 alone is the coordinated C14 scheme, so both controls search 4 options.
SCHEMES = {"none": ([], True), "coord:C14": (["C14"], True), "coord:C14+C17": (["C14", "C17"], True),
           "coord:C14+C17+C21": (FIELDS, True), "indep:C14+C17": (["C14", "C17"], False),
           "indep:C14+C17+C21": (FIELDS, False)}
SEARCH = {"coord": ["none", "coord:C14", "coord:C14+C17", "coord:C14+C17+C21"],
          "indep": ["none", "coord:C14", "indep:C14+C17", "indep:C14+C17+C21"]}
JEV_SCHEME = {"none": "none", "C14": "coord:C14", "C14+C17": "coord:C14+C17", "C14+C17+C21": "coord:C14+C17+C21"}
RATE, N_ROWS, SHRINK, OVERALL_SLACK = 0.10, 300_000, 0.5, 0.0005
B = FEATURE_SETS["B"]
QUESTION = {"block": {"type": "choice", "instructions":
    "A click model is retrained with some creatives hidden: for 10% of training creatives, the chosen identifiers "
    "are set to missing on all of their rows, the way identifiers never seen in training are missing at serving. "
    "Which identifiers should be hidden together, so the model learns to score brand-new creatives without hurting "
    "the others?",
    "criteria": {"none": "hide nothing; train as usual", "C14": "hide C14 only",
                 "C14+C17": "hide C14 and C17 together", "C14+C17+C21": "hide C14, C17 and C21 together"}}}
N_SHUFFLED = 20
WORK = Path.home() / "simula-local/rank-jev-work/exp-x-novelty/x09"
ANSWERS = Path("reports/x09_amnesia_jev_answers.jsonl")


def _hit(values, salt):
    """Deterministic 10% draw per value: the same creative is drawn in every window and every scheme."""
    draws = {v: int(hashlib.sha256(f"{salt}|{v}".encode()).hexdigest()[:8], 16) % 1000 < RATE * 1000
             for v in pd.unique(values)}
    return values.map(draws).to_numpy(bool)


def mask(rows, fields, coordinated):
    """Set `fields` to missing on the rows of 10% of creatives (C14): one draw per creative shared by every field
    (coordinated), or one draw per creative and field (independent)."""
    out = rows.copy()
    for f in fields:
        hit = _hit(rows["C14"], "C14" if coordinated else f)
        out[f] = out[f].astype("string")
        out.loc[hit, f] = pd.NA
    return out


def windows(df):
    """Subsampled dev-fit and refit rows, plus the untouched select, calibrate and test windows."""
    sub = lambda w: split(df, w).sample(N_ROWS, random_state=0)  # noqa: E731
    return {"dev": sub("dev_fit"), "refit": sub("refit"),
            **{w: split(df, w) for w in ("select", "calibrate", "test")}}


def dev_fit(w, fields, coordinated):
    """Dev fit with early stopping on Oct 27. Returns Oct 27 raw predictions and the tree count."""
    dev = mask(w["dev"], fields, coordinated)
    encoder = fit_encoder(dev, B)
    booster, rounds = fit(dev, w["select"], B, encoder)
    return booster.predict(transform(w["select"], encoder)), rounds


def run_scheme(w, fields, coordinated):
    """Fit one masking scheme through B's recipe. Returns Oct 27 raw predictions, test predictions and trees."""
    select_p, rounds = dev_fit(w, fields, coordinated)
    rows = mask(w["refit"], fields, coordinated)
    encoder = fit_encoder(rows, B)
    booster = refit(rows, B, encoder, rounds)
    cal = w["calibrate"]
    delta = calibrate(booster.predict(transform(cal, encoder)), cal["click"], SHRINK)
    return select_p, apply_calibration(booster.predict(transform(w["test"], encoder)), delta), int(rounds)


def _new(rows, earlier, f):
    return ~rows[f].isin(set(earlier[f]))


def pattern(rows, earlier):
    """Which of C14, C17, C21 are unseen in `earlier`, as a label per row."""
    new = {f: _new(rows, earlier, f) for f in FIELDS}
    out = pd.Series("other", index=rows.index)
    out[~new["C14"] & ~new["C17"] & ~new["C21"]] = "all seen"
    out[new["C14"] & ~new["C17"] & ~new["C21"]] = "C14 new only"
    out[new["C14"] & new["C17"] & ~new["C21"]] = "C14+C17 new"
    out[new["C14"] & new["C17"] & new["C21"]] = "C14+C17+C21 new"
    return out


def state(df, dev_select_p):
    """Jev's state: training-window statistics only (Oct 21-27) and the unmasked dev fit's Oct 27 errors."""
    rows = split(df, "refit")
    first = {}
    for day in sorted(rows["day"].unique())[1:]:
        today, before = rows[rows["day"].eq(day)], rows[rows["day"].lt(day)]
        new = {f: _new(today, before, f) for f in FIELDS}
        first[f"20{day}"] = {"impressions": len(today),
                             **{f"share_new_{f}": round(float(new[f].mean()), 4) for f in FIELDS},
                             "ctr_new_C14": round(float(today.loc[new["C14"], "click"].mean()), 4),
                             "ctr_seen_C14": round(float(today.loc[~new["C14"], "click"].mean()), 4)}
    later = rows[rows["day"].gt(rows["day"].min())]
    new = pd.concat([pd.DataFrame({f: _new(r, rows[rows["day"].lt(d)], f) for f in FIELDS})
                     for d, r in later.groupby("day")])
    co = {f"given_new_{a}": {f"share_new_{b}": round(float(new.loc[new[a], b].mean()), 4) for b in FIELDS if b != a}
          for a in FIELDS}
    per = lambda a, b: round(float(rows.groupby(a)[b].nunique().eq(1).mean()), 4)  # noqa: E731
    dependency = {"distinct_values": {f: int(rows[f].nunique()) for f in FIELDS},
                  "share_C14_with_one_C17": per("C14", "C17"), "share_C14_with_one_C21": per("C14", "C21"),
                  "share_C17_with_one_C21": per("C17", "C21")}
    select = split(df, "select")
    labels = pattern(select, split(df, "dev_fit"))
    y = select["click"].to_numpy()
    forward = {name: {"impressions": int(m.sum()), "ctr": round(float(y[m].mean()), 4),
                      "mean_prediction": round(float(dev_select_p[m].mean()), 4),
                      "log_loss": round(float(log_loss(y[m], dev_select_p[m], labels=[0, 1])), 4)}
               for name in ["all seen", "C14 new only", "C14+C17 new", "C14+C17+C21 new", "other"]
               if (m := labels.eq(name).to_numpy()).sum()}
    return {"task": "Choose which creative identifiers to hide together during training.",
            "fields": {"C14": "creative id", "C17": "anonymized ad field", "C21": "anonymized ad field"},
            "first_seen_by_day": first, "co_novelty": co, "dependency": dependency,
            "forward_validation_oct27_by_unseen_fields": forward}


def write_states(st, path):
    """The canonical question and N_SHUFFLED copies with the options in a shuffled order."""
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    criteria = QUESTION["block"]["criteria"]
    with path.open("w") as f:
        for i in range(N_SHUFFLED + 1):
            keys = list(criteria) if i == 0 else list(rng.permutation(list(criteria)))
            q = {"block": {**QUESTION["block"], "criteria": {k: criteria[k] for k in keys}}}
            f.write(json.dumps({"id": f"order:{i}", "state": st, "questions": q}) + "\n")


def jev_pick(path):
    """The option with the highest mean probability over every call, and how often each call's choice differs."""
    rows = [r for r in map(json.loads, Path(path).read_text().splitlines()) if "answers" in r]
    probs = pd.DataFrame([r["answers"]["block"]["probabilities"] for r in rows]).fillna(0)
    choices = pd.Series([r["answers"]["block"]["choice"] for r in rows])
    pick = probs.mean().idxmax()
    return pick, {"calls": len(rows), "mean_probabilities": probs.mean().round(4).to_dict(),
                  "choice_counts": choices.value_counts().to_dict(), "share_differing_from_pick": float(
                      choices.ne(pick).mean()), "tokens_in": int(sum(r["tokens_in"] for r in rows)),
                  "model": sorted({r["model"] for r in rows})}


def choose(scheme_names, val, base):
    """Validation pick: lowest Oct 27 new-C14 log loss among schemes whose overall Oct 27 loss is within the slack."""
    ok = [s for s in scheme_names if val[s]["all"] <= val[base]["all"] + OVERALL_SLACK]
    return min(ok, key=lambda s: val[s]["new_C14"])


def run(data_dir, answers=ANSWERS):
    df = add_flags(load(data_dir))
    w = windows(df)
    test, select = w["test"], w["select"]
    unseen = (~test["C14"].isin(set(split(df, "refit")["C14"]))).to_numpy()
    new27 = (~select["C14"].isin(set(split(df, "dev_fit")["C14"]))).to_numpy()
    y, y27 = test["click"].to_numpy(), select["click"].to_numpy()
    pick, jev = jev_pick(answers)
    val, preds, trees = {}, {}, {}
    for name, (fields, coordinated) in SCHEMES.items():
        p27, preds[name], trees[name] = run_scheme(w, fields, coordinated)
        val[name] = {"all": float(log_loss(y27, p27)), "new_C14": float(log_loss(y27[new27], p27[new27]))}
        print(name, val[name], trees[name], flush=True)
    controls = {"ordinary_B": "none", "indep_validation": choose(SEARCH["indep"], val, "none"),
                "coord_validation": choose(SEARCH["coord"], val, "none")}
    loss = {s: _row_loss(y, p) for s, p in preds.items()}
    test_loss = {s: {"all": float(loss[s].mean()), "unseen_C14": float(loss[s][unseen].mean()),
                     "seen_C14": float(loss[s][~unseen].mean())} for s in SCHEMES}
    chars = test["character_id"].to_numpy()

    def diff(a, b, m):
        return dict(zip(("diff", "ci_lo", "ci_hi"), bootstrap_diff(chars[m], loss[a][m], loss[b][m])))

    everything = np.ones(len(y), bool)
    vs_b = {s: {"unseen_C14": diff(s, "none", unseen), "all": diff(s, "none", everything)} for s in SCHEMES}
    jev_scheme = JEV_SCHEME[pick]
    strongest = min(set(controls.values()), key=lambda s: test_loss[s]["unseen_C14"])
    gate = {"jev_scheme": jev_scheme, "strongest_control": strongest,
            "unseen_vs_strongest": diff(jev_scheme, strongest, unseen), "all_vs_ordinary_B": vs_b[jev_scheme]["all"]}
    u, a = gate["unseen_vs_strongest"], gate["all_vs_ordinary_B"]
    gate["win"] = bool(u["diff"] <= -0.001 and u["ci_hi"] < 0 and a["diff"] <= OVERALL_SLACK)
    shipped = predict("bundle/B", test)
    return {"rows": {"dev_fit": N_ROWS, "refit": N_ROWS, "test": len(test), "unseen_C14": int(unseen.sum()),
                     "oct27_new_C14": int(new27.sum())},
            "masked_share_of_refit_rows": float(np.mean(_hit(w["refit"]["C14"], "C14"))),
            "jev": {"pick": pick, **jev}, "controls": controls, "trees": trees, "validation_oct27": val,
            "test": test_loss, "vs_ordinary_B": vs_b, "gate": gate,
            "shipped_B_full_refit": {"all": float(log_loss(y, shipped)),
                                     "unseen_C14": float(log_loss(y[unseen], shipped[unseen]))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "run"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    if args.step == "states":
        df = add_flags(load(args.data))
        w = windows(df)
        select_p, _ = dev_fit(w, [], True)
        st = state(df, select_p)
        write_states(st, WORK / "states.jsonl")
        print(json.dumps(st, indent=1))
    else:
        out = run(args.data)
        Path("reports/x09_amnesia.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps({k: out[k] for k in ("jev", "controls", "test", "gate")}, indent=1))
