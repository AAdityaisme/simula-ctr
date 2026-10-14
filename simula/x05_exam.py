"""x05, choose the exam before choosing the model (Astra 7): can Jev pick the validation split that selects B's
training recipe best for the following days?

Two rolling origins, both before the test window: pool Oct 21-24 deploying on Oct 25-26, and pool Oct 21-26 deploying
on Oct 27-28. Each pool gives four equal-sized splits (chronological, held-out C17, held-out character, held-out app).
Three recipes (B, B with 15 leaves, B with 10% joint C14+C17+C21 masking) are fit on each split's fit part:
3 x 4 x 2 = 24 fits of at most 300,000 rows and a fixed 183 trees (bundle B's count), so no fit uses its validation
rows. Jev picks the split from churn, overlap, field roles and sizes before any loss exists; it never sees a click.
Run:
  uv run python -m simula.x05_exam states
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x05_exam run
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from simula.data import add_flags, load
from simula.features import FEATURE_SETS, fit_encoder, transform
from simula.jev_ctr import _row_loss, bootstrap_diff
from simula.rank_pairs import read_answers
from simula.train import PARAMS
from simula.x_judge import DATA, N_SUB, WORK, subsample

ANSWERS = Path("reports/x05_choose_exam_jev_answers.jsonl")
ORIGINS = {"o1": ([141021, 141022, 141023, 141024], [141025, 141026]),
           "o2": ([141021, 141022, 141023, 141024, 141025, 141026], [141027, 141028])}
TREES, VALID_CAP = 183, 100_000
FIELDS = {"heldout_C17": "C17", "heldout_character": "character_id", "heldout_app": "app_id"}
SPLITS = {"chronological": "Validate on the pool's last day; fit on the earlier days.",
          "heldout_C17": "Validate on rows of a random set of C17 values (an ad attribute that groups creatives); fit "
                         "on rows of all other C17 values, all days.",
          "heldout_character": "Validate on rows of a random set of characters; fit on rows of all other characters.",
          "heldout_app": "Validate on rows of a random set of host apps; fit on rows of all other apps."}
RECIPES = {"B": "LightGBM model B as shipped (31 leaves).",
           "B_15_leaves": "Model B with 15 leaves per tree.",
           "B_joint_id_mask": "Model B trained with C14, C17 and C21 blanked together on the rows of a random 10% of "
                              "creatives (C14), to rehearse new creatives."}
QUESTION = {"exam": {"type": "choice", "criteria": SPLITS,
                     "instructions": "Three training recipes will be compared on one validation split, and the one "
                                     "with the lowest validation log loss is deployed on the next two days. Which "
                                     "split's validation loss best predicts which recipe will do best after deployment?"}}
ROLES = {"C14": "creative id", "C17": "ad attribute that groups creatives", "character_id": "character (AI persona)",
         "app_id": "host app; ecad2386 marks site-side traffic with no app"}


def make_splits(pool, seed=0):
    """Four (fit, valid) pairs of equal sizes from one pool. Held-out values are drawn at random, skipping values
    bigger than half the target and the no-app placeholder, until they cover one day's worth of rows."""
    last = pool["day"].max()
    target = int(pool["day"].eq(last).sum())
    raw = {"chronological": (pool[pool["day"] < last], pool[pool["day"] == last])}
    for name, field in FIELDS.items():
        counts = pool[field].value_counts()
        counts = counts[(counts <= target / 2) & (counts.index != "ecad2386")]
        values = counts.sample(frac=1, random_state=seed)
        held = set(values.index[: int(np.searchsorted(values.cumsum().to_numpy(), target)) + 1])
        mask = pool[field].isin(held)
        raw[name] = (pool[~mask], pool[mask])
    n_fit = min(N_SUB, *(len(f) for f, _ in raw.values()))
    n_valid = min(VALID_CAP, *(len(v) for _, v in raw.values()))
    return {k: (subsample(f, n_fit, seed), subsample(v, n_valid, seed)) for k, (f, v) in raw.items()}


def novelty(rows, seen_rows, fields=("C14", "C17", "character_id", "app_id")):
    """Share of `rows` whose value of each field never appears in `seen_rows`."""
    return {f: round(float((~rows[f].isin(set(seen_rows[f]))).mean()), 4) for f in fields}


def state(pool, splits):
    """Jev state for one origin: pool, churn, roles, recipes and each split's sizes and overlap. No clicks, no loss."""
    last = pool["day"].max()
    return {"pool_days": sorted(int(d) for d in pool["day"].unique()), "pool_rows": len(pool),
            "deployment": "the model is deployed on the two days after the pool",
            "pool_last_day_churn": novelty(pool[pool["day"] == last], pool[pool["day"] < last]),
            "field_roles": ROLES, "recipes": RECIPES,
            "splits": {k: {"how": SPLITS[k], "fit_rows": len(f), "validation_rows": len(v),
                           "validation_novelty_vs_fit": novelty(v, f)} for k, (f, v) in splits.items()}}


def pools(df):
    return {o: (df[df["day"].isin(p)], df[df["day"].isin(later)]) for o, (p, later) in ORIGINS.items()}


def write_states(df, path):
    orders = {"": SPLITS, "_rev": dict(reversed(SPLITS.items())),
              "_rot": dict(list(SPLITS.items())[2:] + list(SPLITS.items())[:2])}
    with open(path, "w") as f:
        for o, (pool, _) in pools(df).items():
            s = state(pool, make_splits(pool))
            for tag, crit in orders.items():
                q = {"exam": {**QUESTION["exam"], "criteria": crit}}
                f.write(json.dumps({"id": f"{o}{tag}", "state": s, "questions": q}) + "\n")


def novelty_matching(s):
    """Deterministic control: the held-out split for the field with the highest pool churn (C17, character, app)."""
    churn = s["pool_last_day_churn"]
    return max(FIELDS, key=lambda k: churn[FIELDS[k]])


def joint_mask(rows, share=0.1, seed=0):
    """Blank C14, C17 and C21 together on the rows of a random `share` of creatives."""
    values = rows["C14"].unique()
    hit = set(np.random.default_rng(seed).choice(values, int(share * len(values)), replace=False))
    out = rows.copy()
    out.loc[out["C14"].isin(hit), ["C14", "C17", "C21"]] = np.nan
    return out


def fit_recipe(name, rows):
    """One fixed-length fit of a recipe; returns a predict function."""
    rows = joint_mask(rows) if name == "B_joint_id_mask" else rows
    params = {**PARAMS, "num_leaves": 15} if name == "B_15_leaves" else PARAMS
    encoder = fit_encoder(rows, FEATURE_SETS["B"])
    data = lgb.Dataset(transform(rows, encoder), label=rows["click"], categorical_feature=list(encoder["categories"]))
    booster = lgb.train(params, data, num_boost_round=TREES)
    return lambda x: booster.predict(transform(x, encoder))


def regret(table, pick):
    """Regrets of the recipe a split selects. `table` rows: split, recipe, valid_loss, later_loss.

    recipe regret (headline): the selected recipe's later loss averaged over its four fits, minus the best recipe's.
    fit regret (Astra literal): the later loss of the selected fit itself, minus the best of all 12 fits.
    """
    sel = table[table["split"] == pick].sort_values("valid_loss").iloc[0]
    by_recipe = table.groupby("recipe")["later_loss"].mean()
    return {"split": pick, "recipe": sel["recipe"],
            "recipe_regret": float(by_recipe[sel["recipe"]] - by_recipe.min()),
            "fit_regret": float(sel["later_loss"] - table["later_loss"].min())}


def run(df, answers):
    out = {}
    for o, (pool, later) in pools(df).items():
        splits = make_splits(pool)
        s = state(pool, splits)
        y, losses, rows = later["click"].to_numpy(), {}, []
        for split_name, (fit_rows, valid) in splits.items():
            for recipe in RECIPES:
                predict = fit_recipe(recipe, fit_rows)
                losses[(split_name, recipe)] = _row_loss(y, predict(later))
                rows.append({"split": split_name, "recipe": recipe,
                             "valid_loss": float(_row_loss(valid["click"].to_numpy(), predict(valid)).mean()),
                             "later_loss": float(losses[(split_name, recipe)].mean())})
                print(o, rows[-1], flush=True)
        table = pd.DataFrame(rows)
        picks = {"jev": answers[o]["answers"]["exam"]["choice"], "chronological": "chronological",
                 "novelty_matching": novelty_matching(s)}
        sel = {k: regret(table, v) for k, v in picks.items()}
        recipe_loss = {r: np.mean([losses[(sp, r)] for sp in SPLITS], axis=0) for r in RECIPES}
        intervals = {f"jev_minus_{k}": dict(zip(("diff", "ci_lo", "ci_hi"), bootstrap_diff(
            later["character_id"], recipe_loss[sel["jev"]["recipe"]], recipe_loss[sel[k]["recipe"]])))
            for k in ("chronological", "novelty_matching")}
        out[o] = {"later_rows": len(y), "state": s, "fits": rows, "selections": sel, "intervals": intervals,
                  "jev_orders": {t: answers[f"{o}{t}"]["answers"]["exam"] for t in ("", "_rev", "_rot")}}
    mean = {k: float(np.mean([out[o]["selections"][k]["recipe_regret"] for o in ORIGINS]))
            for k in ("jev", "chronological", "novelty_matching")}
    calls = list(answers.values())
    return {"origins": out, "mean_recipe_regret": mean,
            "mean_fit_regret": {k: float(np.mean([out[o]["selections"][k]["fit_regret"] for o in ORIGINS]))
                                for k in mean},
            "win": bool(mean["jev"] < min(mean["chronological"], mean["novelty_matching"])),
            "jev": {"calls": len(calls), "models": sorted({a["model"] for a in calls}),
                    "input_tokens": int(sum(a["tokens_in"] for a in calls)),
                    "call_seconds_sum": float(sum(a["seconds"] for a in calls))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "run"])
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()
    data = add_flags(load(args.data))
    data = data[data["day"] <= 141028]  # the test window is never read here
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        write_states(data, WORK / "x05_states.jsonl")
        print("wrote", WORK / "x05_states.jsonl")
    else:
        result = run(data, read_answers(ANSWERS))
        Path("reports/x05_choose_exam.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "origins"}, indent=1, default=float))
        for o in ORIGINS:
            print(o, json.dumps(result["origins"][o]["selections"]), json.dumps(result["origins"][o]["intervals"]))
