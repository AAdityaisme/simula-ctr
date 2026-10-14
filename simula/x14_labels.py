"""x14 (Astra 3): spend a label budget. Does Jev choose which rows B learns from better than acquisition rules?

Pre-registered before any outcome-bearing run:
- Pool: Oct 21-26. Each seed labels 50,000 random pool rows, then allocates 20,000 more. Every model is B's recipe
  fit on the labeled rows with that seed, trees by early stopping on Oct 27 (up to 400), B's half intercept shift on
  Oct 28, scored on Oct 29-30.
- Cohorts (40): genre (10) x surface (site/app) x creative frequency (C14 with fewer than 20 rows in the first
  50,000 labels, "rare_ad", or more, "common_ad"). A smoke run split on seen/unseen C14 instead: unseen cohorts held
  about 100 rows each, so the split was changed before any outcome-bearing run.
  Per cohort, Jev sees: unlabeled rows, labeled rows, coverage, the share of unlabeled rows with a C14, app or site
  unseen in the labels, mean entropy of the 50,000-row model, and mean disagreement (sd across three bootstrap models
  of the same 50,000 rows). No unlabeled click.
- Jev: 20 sequential quotas of 1,000. Each quota is a Choice over the 40 cohorts, with the state updated for labels
  allocated so far; the quota is drawn at random inside the chosen cohort (all of it if fewer remain). Each call is
  repeated with the options reversed to measure order stability. Astra's design also offered "stop"; in the smoke run
  (3 states, no outcome) Jev chose stop on the first quota in all three seeds, which leaves the budget unspent, so
  stop was dropped to keep the comparison at an equal 20,000 labels.
- Controls with the same 20,000: random rows; uncertainty (top entropy rows); uncertainty plus diversity (per-cohort
  quota proportional to unlabeled rows x mean entropy, top-entropy rows inside each cohort).
- Win: Jev's test log loss is below every control in each of three seeds. This tests sample efficiency only, not
  real labeling cost: every "new" label already existed.
Run: OMP_NUM_THREADS=3 uv run python -m simula.x14_labels [smoke]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simula import xtrain
from simula.data import split
from simula.features import FEATURE_SETS

SEEDS = (0, 1, 2)
START, BUDGET, QUOTA = 50_000, 20_000, 1_000
RARE = 20
TREES = 400
ARMS = ("jev", "random", "uncertainty", "uncertainty_diversity")
INSTRUCTIONS = ("A click model has 50,000 labeled training rows. Labels are bought in quotas of 1,000 rows drawn "
                "at random from one cohort. Which cohort should get the next quota so the retrained model predicts "
                "clicks on future traffic best?")


def entropy(p):
    """Binary entropy in bits."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def largest_remainder(weights, total):
    """Integer allocation of `total` proportional to `weights`."""
    w = np.asarray(weights, dtype=float)
    exact = total * w / w.sum()
    out = np.floor(exact).astype(int)
    out[np.argsort(exact - out)[::-1][:total - out.sum()]] += 1
    return out


def cohorts(pool, labeled_c14):
    """Cohort label per row: genre x surface x C14 frequency in the first labels."""
    side = np.where(pool["is_site_side"], "site", "app")
    novelty = np.where(pool["C14"].map(labeled_c14).fillna(0).ge(RARE), "common_ad", "rare_ad")
    return pool["genre"].astype(str) + "|" + side + "|" + novelty


def best_fit(train, valid, seed):
    """B's recipe fit with early stopping on `valid`; returns (model, trees)."""
    booster, encoder, curve = xtrain.fit(train, FEATURE_SETS["B"], TREES, seed, valid=valid)
    return (booster, encoder), int(np.argmin(curve["valid"])) + 1


def state(table, spent):
    """Jev state: per-cohort stats with labels allocated so far."""
    return {"labels_spent_of_20000": spent,
            "cohorts": {cid: {"genre": r["genre"], "surface": r["surface"], "creative": r["creative"],
                              "unlabeled_rows": int(r["unlabeled"]), "labeled_rows": int(r["labeled"]),
                              "coverage": round(r["labeled"] / (r["labeled"] + r["unlabeled"]), 4),
                              "novel_id_share": round(r["novel"], 3), "mean_entropy_bits": round(r["entropy"], 4),
                              "mean_disagreement": round(r["disagreement"], 4)} for cid, r in table.items()}}


def question(table, reverse=False):
    """Choice over the cohort ids."""
    criteria = {cid: f"{r['genre']} characters, {r['surface']} side, {r['creative'].replace('_', ' ')}s"
                for cid, r in table.items()}
    names = list(criteria)[::-1] if reverse else list(criteria)
    return {"cohort": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {n: criteria[n] for n in names}}}


def run(smoke=False):
    df = xtrain.frame()
    pool_all, select, cal, test = (split(df, w) for w in ("dev_fit", "select", "calibrate", "test"))
    y, groups = test["click"].to_numpy(), test["character_id"].to_numpy()
    seeds = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        start = pool_all.sample(START, random_state=seed)
        pool = pool_all.drop(start.index)
        m0, trees0 = best_fit(start, select, seed)
        p0 = xtrain.raw(m0, pool, trees0)
        bags = [xtrain.fit(start.sample(START, replace=True, random_state=100 + b), FEATURE_SETS["B"], trees0,
                           seed)[:2] for b in range(3)]
        disagreement = np.std([xtrain.raw(m, pool) for m in bags], axis=0)
        ent = entropy(p0)
        counts = start["C14"].value_counts()
        cohort = cohorts(pool, counts)
        novel = ~pool["C14"].isin(set(start["C14"])) | ~pool["app_id"].isin(set(start["app_id"])) | \
            ~pool["site_id"].isin(set(start["site_id"]))
        stats = pool.assign(cohort=cohort.to_numpy(), entropy=ent, disagreement=disagreement, novel=novel)
        stats = stats.groupby("cohort").agg(unlabeled=("click", "size"), entropy=("entropy", "mean"),
                                            disagreement=("disagreement", "mean"), novel=("novel", "mean"))
        start_cohort = cohorts(start, counts).value_counts()
        table = {f"c{i:02d}": {"key": k, "genre": k.split("|")[0], "surface": k.split("|")[1],
                               "creative": k.split("|")[2], "labeled": int(start_cohort.get(k, 0)), **r}
                 for i, (k, r) in enumerate(stats.sort_index().iterrows(), 1)}
        seeds[seed] = dict(rng=rng, start=start, pool=pool, cohort=cohort, ent=ent, table=table,
                           available={cid: list(np.flatnonzero(cohort.to_numpy() == r["key"]))
                                      for cid, r in table.items()},
                           jev_rows=[], picks=[])
        print("seed", seed, "start trees", trees0, "cohorts", len(table), flush=True)

    xtrain.WORK.mkdir(parents=True, exist_ok=True)
    answers = {}
    for q in range(BUDGET // QUOTA):
        items = []
        for seed, s in seeds.items():
            st = state(s["table"], sum(len(r) for r in s["jev_rows"]))
            items += [{"id": f"{seed}:{q}", "state": st, "questions": question(s["table"])},
                      {"id": f"rev:{seed}:{q}", "state": st, "questions": question(s["table"], reverse=True)}]
        if smoke:
            return {k: a["answers"] for k, a in xtrain.ask_jev(items[::2], "x14_smoke").items()}
        if q == 0:
            xtrain.WORK.joinpath("x14_example_state.json").write_text(json.dumps(items[0], indent=1))
        answers.update(xtrain.ask_jev(items, "x14"))
        for seed, s in seeds.items():
            pick = answers[f"{seed}:{q}"]["answers"]["cohort"]["choice"]
            s["picks"].append({"quota": q, "pick": pick, "confidence": answers[f"{seed}:{q}"]["answers"]["cohort"]["confidence"],
                               "reversed_pick": answers[f"rev:{seed}:{q}"]["answers"]["cohort"]["choice"]})
            avail = s["available"][pick]
            take = s["rng"].choice(len(avail), min(QUOTA, len(avail)), replace=False)
            chosen = [avail[i] for i in take]
            taken = set(take)
            s["available"][pick] = [a for i, a in enumerate(avail) if i not in taken]
            s["jev_rows"].append(chosen)
            s["table"][pick]["labeled"] += len(chosen)
            s["table"][pick]["unlabeled"] -= len(chosen)
        print("quota", q, {seed: s["picks"][-1]["pick"] for seed, s in seeds.items() if s["picks"]}, flush=True)

    per_seed = {}
    for seed, s in seeds.items():
        pool, ent, cohort = s["pool"], s["ent"], s["cohort"].to_numpy()
        keys = [r["key"] for r in s["table"].values()]
        n = np.array([(cohort == k).sum() for k in keys])
        quota = largest_remainder(n * np.array([ent[cohort == k].mean() for k in keys]), BUDGET)
        diverse = np.concatenate([np.flatnonzero(cohort == k)[np.argsort(-ent[cohort == k])[:q]]
                                  for k, q in zip(keys, quota)])
        picks = {"jev": np.concatenate(s["jev_rows"]) if s["jev_rows"] else np.array([], dtype=int),
                 "random": np.random.default_rng(1000 + seed).choice(len(pool), BUDGET, replace=False),
                 "uncertainty": np.argsort(-ent)[:BUDGET], "uncertainty_diversity": diverse}
        losses, row = {}, {"jev_labels_spent": int(len(picks["jev"])), "jev_picks": s["picks"]}
        for arm in ARMS:
            train = pd.concat([s["start"], pool.iloc[picks[arm]]])
            model, trees = best_fit(train, select, seed)
            losses[arm] = xtrain.row_loss(y, xtrain.calibrated(model, cal, test, trees))
            row[arm] = {"labels": len(picks[arm]), "trees": trees, "test_log_loss": float(losses[arm].mean())}
            print(seed, arm, row[arm], flush=True)
        row["jev_vs"] = {arm: xtrain.bootstrap_diff(groups, losses["jev"], losses[arm]) for arm in ARMS[1:]}
        per_seed[seed] = row
    wins = [all(per_seed[s]["jev"]["test_log_loss"] < per_seed[s][a]["test_log_loss"] for a in ARMS[1:])
            for s in SEEDS]
    return {"per_seed": per_seed, "jev_beats_every_control": wins, "win": all(wins),
            "order_stability": float(np.mean([p["pick"] == p["reversed_pick"]
                                              for s in seeds.values() for p in s["picks"]])),
            "jev": xtrain.jev_cost(answers)}


if __name__ == "__main__":
    if sys.argv[1:] == ["smoke"]:
        print(json.dumps(run(smoke=True), indent=1))
        sys.exit()
    out = run()
    Path("reports/x14_labels.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1, default=float))
