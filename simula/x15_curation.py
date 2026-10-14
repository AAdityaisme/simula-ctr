"""x15 (Fable E4): training-data curation. Does excluding the device-days Jev judges automated help B?

Pre-registered before any outcome-bearing run:
- Profiles: every non-sentinel (device_id, day) in the refit window (Oct 21-27) with at least 5 impressions (371).
  Fable's 581 counted all ten days; only refit rows can be excluded, so only refit device-days are judged.
- Profile: impressions, clicks, CTR, hours active, distinct creatives (C14), apps and sites, device_model and
  site-side share.
- Jev: Noul "exclude_as_automated"; a profile is flagged when the Noul is at least 0.5.
- Controls: none excluded (B); Fable's rule (impressions >= 20 or CTR >= 0.5); an isolation forest on the numeric
  profile fields with contamination equal to the rule's flagged share (the same exclusion budget), seed 0.
- Each arm drops its flagged device-days' rows from the full refit window (no subsample: a fit takes seconds), refits
  B's recipe for B's 183 trees with three LightGBM seeds, is recalibrated on Oct 28 with B's half intercept shift and
  scored on Oct 29-30. Row losses are averaged over the seeds.
- Clean slice: test rows whose (device_ip, device_model) has at most 2 impressions that day.
- Win: Jev beats the rule on the clean slice, paired character-cluster interval excluding 0, every arm recalibrated.
  Also reported: kappa(Jev, rule), all-test loss, rows excluded.
Run: OMP_NUM_THREADS=3 uv run python -m simula.x15_curation [smoke | noise]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import cohen_kappa_score

from simula import xtrain
from simula.data import split
from simula.features import FEATURE_SETS

SEEDS = (0, 1, 2)
TREES = 183  # bundle/B best_iteration
ARMS = ("none", "rule", "iforest", "jev")
NUMERIC = ["impressions", "clicks", "ctr", "hours_active", "distinct_creatives", "distinct_apps", "distinct_sites",
           "site_side_share"]
EXCLUDE = {"exclude_as_automated": {
    "type": "noul", "criteria": None,
    "instructions": "This device-day is automated traffic, such as a click farm or bot, and its impressions should be "
                    "excluded from the click model's training data."}}


def profiles(refit):
    """One row per non-sentinel (device_id, day) with at least 5 refit impressions."""
    rows = refit[~refit["is_null_device"]]
    g = rows.groupby(["device_id", "day"])
    out = g.agg(impressions=("click", "size"), clicks=("click", "sum"), hours_active=("hour", "nunique"),
                distinct_creatives=("C14", "nunique"), distinct_apps=("app_id", "nunique"),
                distinct_sites=("site_id", "nunique"), device_model=("device_model", "first"),
                site_side_share=("is_site_side", "mean")).reset_index()
    out = out[out["impressions"].ge(5)].reset_index(drop=True)
    out["ctr"] = out["clicks"] / out["impressions"]
    return out


def rule(p):
    """Fable's rule: 20+ impressions or CTR of at least 0.5."""
    return (p["impressions"].ge(20) | p["ctr"].ge(0.5)).to_numpy()


def iforest(p, share):
    """Isolation forest on the numeric profile fields, flagging `share` of profiles."""
    x = p[NUMERIC].assign(impressions=np.log(p["impressions"]), clicks=np.log1p(p["clicks"])).to_numpy()
    return IsolationForest(contamination=share, random_state=0).fit_predict(x) == -1


def state(r):
    """Jev state for one profile."""
    return {"impressions": int(r.impressions), "clicks": int(r.clicks), "ctr": round(float(r.ctr), 3),
            "hours_active_of_24": int(r.hours_active), "distinct_creatives": int(r.distinct_creatives),
            "distinct_apps": int(r.distinct_apps), "distinct_sites": int(r.distinct_sites),
            "device_model": r.device_model, "site_side_share": round(float(r.site_side_share), 2)}


def items(p):
    """Jev input lines, one per profile."""
    return [{"id": f"{r.device_id}:{r.day}", "state": state(r), "questions": EXCLUDE} for r in p.itertuples()]


def clean_slice(test):
    """Test rows whose (device_ip, device_model) has at most 2 impressions that day."""
    return test.groupby(["device_ip", "device_model", "day"])["click"].transform("size").le(2).to_numpy()


def run():
    df = xtrain.frame()
    refit, cal, test = (split(df, w) for w in ("refit", "calibrate", "test"))
    prof = profiles(refit)
    answers = xtrain.ask_jev(items(prof), "x15")
    noul = np.array([answers[i["id"]]["answers"]["exclude_as_automated"]["noul"] for i in items(prof)])
    flags = {"none": np.zeros(len(prof), bool), "rule": rule(prof)}
    flags["iforest"] = iforest(prof, flags["rule"].mean())
    flags["jev"] = noul >= 0.5

    key = refit["device_id"].astype(str) + ":" + refit["day"].astype(str)
    prof_key = prof["device_id"].astype(str) + ":" + prof["day"].astype(str)
    y, groups, clean = test["click"].to_numpy(), test["character_id"].to_numpy(), clean_slice(test)
    losses, arms = {}, {}
    for arm in ARMS:
        train = refit[~key.isin(set(prof_key[flags[arm]]))]
        losses[arm] = np.mean([xtrain.row_loss(y, xtrain.calibrated(
            xtrain.fit(train, FEATURE_SETS["B"], TREES, seed)[:2], cal, test)) for seed in SEEDS], axis=0)
        arms[arm] = {"profiles_flagged": int(flags[arm].sum()), "rows_excluded": len(refit) - len(train),
                     "test_log_loss": float(losses[arm].mean()), "clean_log_loss": float(losses[arm][clean].mean()),
                     "dirty_log_loss": float(losses[arm][~clean].mean())}
        print(arm, arms[arm], flush=True)
    comparisons = [{"candidate": a, "baseline": b, "slice": s,
                    "char_ci": xtrain.bootstrap_diff(groups[m], losses[a][m], losses[b][m])}
                   for a, b in [("jev", "rule"), ("jev", "none"), ("jev", "iforest"), ("rule", "none"),
                                ("iforest", "none")]
                   for s, m in (("clean", clean), ("all", np.ones(len(test), bool)))]
    return {"profiles": len(prof), "profile_rows": int(prof["impressions"].sum()), "refit_rows": len(refit),
            "test_rows": len(test), "clean_rows": int(clean.sum()), "arms": arms, "comparisons": comparisons,
            "kappa": {"jev_rule": cohen_kappa_score(flags["jev"], flags["rule"]),
                      "jev_iforest": cohen_kappa_score(flags["jev"], flags["iforest"]),
                      "iforest_rule": cohen_kappa_score(flags["iforest"], flags["rule"])},
            "jev_noul_quantiles": dict(zip(("min", "p25", "p50", "p75", "max"),
                                           map(float, np.percentile(noul, [0, 25, 50, 75, 100])))),
            "jev_flagged_profile_means": prof[flags["jev"]][NUMERIC].mean().round(3).to_dict(),
            "rule_flagged_profile_means": prof[flags["rule"]][NUMERIC].mean().round(3).to_dict(),
            "jev": xtrain.jev_cost(answers)}


def noise():
    """Post-hoc, after the pre-registered run: per-seed losses, extra seeds for no exclusion, and a placebo that drops
    as many random non-profile refit rows as the rule does. Separates exclusion effects from refit noise."""
    df = xtrain.frame()
    refit, cal, test = (split(df, w) for w in ("refit", "calibrate", "test"))
    prof = profiles(refit)
    answers = xtrain.ask_jev(items(prof), "x15")
    flags = {"rule": rule(prof), "jev": np.array([answers[i["id"]]["answers"]["exclude_as_automated"]["noul"]
                                                   for i in items(prof)]) >= 0.5}
    key = refit["device_id"].astype(str) + ":" + refit["day"].astype(str)
    prof_key = prof["device_id"].astype(str) + ":" + prof["day"].astype(str)
    drop = {arm: key.isin(set(prof_key[f])) for arm, f in flags.items()}
    placebo = refit.index[~key.isin(set(prof_key))].to_series().sample(int(drop["rule"].sum()), random_state=0)
    drop["placebo"], drop["none"] = refit.index.isin(placebo), np.zeros(len(refit), bool)
    y, clean = test["click"].to_numpy(), clean_slice(test)
    out = {}
    for arm, seeds in (("none", range(6)), ("placebo", SEEDS), ("rule", SEEDS), ("jev", SEEDS)):
        for seed in seeds:
            loss = xtrain.row_loss(y, xtrain.calibrated(
                xtrain.fit(refit[~drop[arm]], FEATURE_SETS["B"], TREES, seed)[:2], cal, test))
            out[f"{arm}:{seed}"] = {"rows_dropped": int(drop[arm].sum()), "test": float(loss.mean()),
                                    "clean": float(loss[clean].mean())}
            print(arm, seed, out[f"{arm}:{seed}"], flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["noise"]:
        Path("reports/x15_noise.json").write_text(json.dumps(noise(), indent=1) + "\n")
        sys.exit()
    if sys.argv[1:] == ["smoke"]:
        prof = profiles(split(xtrain.frame(), "refit"))
        print(len(prof), "profiles")
        for k, a in xtrain.ask_jev(items(prof.sample(3, random_state=0)), "x15_smoke").items():
            print(k, a["tokens_in"], a["answers"])
        sys.exit()
    out = run()
    Path("reports/x15_curation.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))
