"""x13 (Astra 4): make retraining a scarce action. Pilot.

Pre-registered before any outcome-bearing run:
- Replay Oct 26-30. The starting model is B's recipe fit on 300,000 Oct 21-25 rows, 110 trees (near the 300,000-row
  Oct 27 argmin of the x12 smoke curves), intercept 0. Oct 26 is scored with it for every policy.
- At midnight before each of Oct 27, 28, 29 and 30 a policy picks one action using labels through the previous day:
  hold; intercept_update (B's half intercept shift fitted on the previous day); refit_recent_three_days (110 trees
  on the previous three days, at most 300,000 rows, intercept 0). At most two refits per replay.
- Policies, all under the same two-refit budget:
  - jev: Jev's Choice over the three actions, from the state below. A refit pick with no budget left counts as hold.
  - scheduled: refit before Oct 27 and Oct 29, intercept_update otherwise.
  - ewma: g = logit(actual CTR) - logit(mean prediction) on the previous day; e = 0.5 g + 0.5 e (reset to 0 after a
    refit); refit when |e| > 0.10 and budget remains, else intercept_update.
  - hour_z: z_h = (clicks - sum p) / sqrt(sum p(1-p)) per hour of the previous day, Z = sum z_h / sqrt(hours);
    refit when |Z| > 3 and budget remains, intercept_update when |Z| > 2, else hold.
  - References with no refits: always_hold, always_intercept.
- Jev state: the current model's training window, intercept and refits left; for each out-of-time day since its
  training ended (last three at most): rows, actual and mean predicted CTR, log loss, predicted vs actual by 6-hour
  block, and the site-side, C20=-1, null-device and unseen-C14 shares; the training-window mix; past actions.
  The replay feeds prior-day labels into each decision by design, so the Oct 30 decision sees Oct 29 labels. No
  threshold, rule or wording was tuned on any outcome.
- Score: pooled log loss on Oct 27-29, and the partial Oct 30 separately. Three seeds (subsample and LightGBM seed).
- Win: jev has lower pooled Oct 27-29 loss than scheduled, ewma and hour_z, with no extra refits. Twelve decisions in
  all, so it is a pilot. Intervals: paired character-cluster bootstrap over the pooled rows of the three seeds.
Run: OMP_NUM_THREADS=3 uv run python -m simula.x13_retrain [smoke]
"""
import json
import sys
from pathlib import Path

import numpy as np

from simula import xtrain
from simula.features import FEATURE_SETS
from simula.train import calibrate

SEEDS = (0, 1, 2)
TREES = 110
DECIDE = (141027, 141028, 141029, 141030)
SCORE = (141027, 141028, 141029)
BUDGET = 2
POLICIES = ("jev", "scheduled", "ewma", "hour_z", "always_hold", "always_intercept")
CRITERIA = {"hold": "keep the current model and intercept unchanged",
            "intercept_update": "keep the trees, refit only the intercept on yesterday's labels",
            "refit_recent_three_days": "retrain the trees on the last three labeled days (uses one refit)"}
INSTRUCTIONS = ("It is midnight. A click model serves ads all of tomorrow. Labels are in through yesterday. Refits "
                "are scarce: at most two in this replay. Which action keeps tomorrow's log loss lowest?")
BLOCKS = {"00-05": range(0, 6), "06-11": range(6, 12), "12-17": range(12, 18), "18-23": range(18, 24)}


def _logit(p):
    return float(np.log(p / (1 - p)))


def ewma_step(e, gap):
    """One EWMA update of the daily logit gap."""
    return 0.5 * gap + 0.5 * e


def hour_z(hours, y, p):
    """Hour-standardized residual: per-hour binomial z of clicks vs predictions, summed over sqrt(hours)."""
    z = [(y[m].sum() - p[m].sum()) / np.sqrt((p[m] * (1 - p[m])).sum()) for h in np.unique(hours) for m in [hours == h]]
    return float(np.sum(z) / np.sqrt(len(z)))


def rule_action(policy, day, t, prev):
    """The control policies' action for trajectory t at decision `day`, given the previous day's summary."""
    can_refit = t["refits"] < BUDGET
    if policy == "always_hold":
        return "hold"
    if policy == "always_intercept":
        return "intercept_update"
    if policy == "scheduled":
        return "refit_recent_three_days" if day in (141027, 141029) else "intercept_update"
    if policy == "ewma":
        t["ewma"] = ewma_step(t["ewma"], prev["logit_gap"])
        return "refit_recent_three_days" if abs(t["ewma"]) > 0.10 and can_refit else "intercept_update"
    if policy == "hour_z":
        z = prev["hour_z"]
        return "refit_recent_three_days" if abs(z) > 3 and can_refit else "intercept_update" if abs(z) > 2 else "hold"
    raise ValueError(policy)


def _mix(df, seen):
    return {"site_side_share": round(float(df["is_site_side"].mean()), 3),
            "c20_minus1_share": round(float(df["c20_is_sentinel"].mean()), 3),
            "null_device_share": round(float(df["is_null_device"].mean()), 3),
            "unseen_c14_share": round(float((~df["C14"].isin(seen)).mean()), 3)}


def day_summary(df, p, seen):
    """What a policy may know about one labeled out-of-time day under the current model."""
    y, hours = df["click"].to_numpy(), df["hour_of_day"].to_numpy()
    blocks = {b: {"actual_ctr": round(float(y[m].mean()), 4), "mean_predicted_ctr": round(float(p[m].mean()), 4)}
              for b, hs in BLOCKS.items() if (m := np.isin(hours, list(hs))).any()}
    return {"labeled_rows": len(df), "actual_ctr": round(float(y.mean()), 4),
            "mean_predicted_ctr": round(float(p.mean()), 4),
            "log_loss": round(float(xtrain.row_loss(y, p).mean()), 5), "by_hour_block": blocks, **_mix(df, seen),
            "logit_gap": _logit(y.mean()) - _logit(p.mean()), "hour_z": hour_z(hours, y, p)}


def _label(day):
    return f"Oct {day % 100}"


def state(t, day, summaries, models):
    """Jev state for trajectory t at decision `day`."""
    m = models[t["model"]]
    recent = [d for d in summaries if d > m["last_day"]][-3:]
    return {"decision": f"midnight before {_label(day)}; the action applies to all of {_label(day)}",
            "current_model": {"trained_on": m["window"], "intercept_shift": round(t["delta"], 4)},
            "refits_left": BUDGET - t["refits"], "decisions_left_including_this": len([d for d in DECIDE if d >= day]),
            "training_window": m["mix"],
            "recent_out_of_time_days": {_label(d): {k: v for k, v in summaries[d].items()
                                                    if k not in ("logit_gap", "hour_z")} for d in recent},
            "past_actions": {_label(d): a for d, a in t["actions"].items()}}


def question(reverse=False):
    """The Choice question."""
    names = list(CRITERIA)[::-1] if reverse else list(CRITERIA)
    return {"action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {n: CRITERIA[n] for n in names}}}


def run(smoke=False):
    df = xtrain.frame()
    days = {d: df[df["day"].eq(d)] for d in range(141021, 141031)}
    features = FEATURE_SETS["B"]
    models, raw = {}, {}

    def model(seed, key):
        if (seed, key) not in models:
            window = list(range(141021, 141026)) if key == "init" else [key - 3, key - 2, key - 1]
            train = xtrain.subsample(df[df["day"].isin(window)], seed)
            models[(seed, key)] = {"fit": xtrain.fit(train, features, TREES, seed)[:2], "last_day": window[-1],
                                   "window": f"{_label(window[0])}-{window[-1] % 100}", "seen": set(train["C14"]),
                                   "mix": {"ctr": round(float(train["click"].mean()), 4),
                                           **_mix(train, set(train["C14"]))}}
            print("fit", seed, key, flush=True)
        return models[(seed, key)]

    def predict(t, day):
        key = (t["seed"], t["model"], day)
        if key not in raw:
            raw[key] = xtrain.raw(model(t["seed"], t["model"])["fit"], days[day])
        return xtrain.apply_calibration(raw[key], t["delta"])

    trajs = [{"policy": p, "seed": s, "model": "init", "delta": 0.0, "refits": 0, "ewma": 0.0, "actions": {},
              "summaries": {}, "loss": {}, "budget_violations": 0} for p in POLICIES for s in SEEDS]
    jev_answers = {}
    for day in DECIDE:
        for t in trajs:
            prev = day - 1
            m = model(t["seed"], t["model"])
            t["summaries"][prev] = day_summary(days[prev], predict(t, prev), m["seen"])
        items = []
        for t in trajs:
            if t["policy"] == "jev":
                s = state(t, day, t["summaries"], {t["model"]: model(t["seed"], t["model"])})
                key = f"{day}:{t['seed']}:{','.join(t['actions'].values())}"
                items += [{"id": key, "state": s, "questions": question()},
                          {"id": f"rev:{key}", "state": s, "questions": question(reverse=True)}]
                t["jev_id"] = key
        if smoke:
            return {k: a["answers"] for k, a in xtrain.ask_jev(items[::2], "x13_smoke").items()}
        if day == DECIDE[0]:
            xtrain.WORK.joinpath("x13_example_state.json").write_text(json.dumps(items[0], indent=1))
        jev_answers.update(xtrain.ask_jev(items, "x13"))
        for t in trajs:
            prev = t["summaries"][day - 1]
            if t["policy"] == "jev":
                a = jev_answers[t["jev_id"]]["answers"]["action"]
                action = a["choice"]
                t.setdefault("jev_detail", {})[day] = {
                    "choice": action, "probabilities": a["probabilities"],
                    "reversed_choice": jev_answers[f"rev:{t['jev_id']}"]["answers"]["action"]["choice"]}
                if action == "refit_recent_three_days" and t["refits"] >= BUDGET:
                    t["budget_violations"] += 1
                    action = "hold"
            else:
                action = rule_action(t["policy"], day, t, prev)
            if action == "intercept_update":
                y = days[day - 1]["click"]
                t["delta"] = calibrate(raw[(t["seed"], t["model"], day - 1)], y, xtrain.SHRINK)
            elif action == "refit_recent_three_days":
                t.update(model=day, delta=0.0, refits=t["refits"] + 1, ewma=0.0)
            t["actions"][day] = action
            t["loss"][day] = xtrain.row_loss(days[day]["click"], predict(t, day))
            print(t["policy"], t["seed"], _label(day), action, round(float(t["loss"][day].mean()), 5), flush=True)

    groups = {span: np.concatenate([days[d]["character_id"].to_numpy() for d in span] * len(SEEDS))
              for span in (SCORE, (141030,))}

    def pooled(policy, span):
        return np.concatenate([t["loss"][d] for s in SEEDS for t in trajs if t["policy"] == policy and t["seed"] == s
                               for d in span])

    out = {"policies": {}, "comparisons": [], "jev": xtrain.jev_cost(jev_answers)}
    for p in POLICIES:
        mine = [t for t in trajs if t["policy"] == p]
        out["policies"][p] = {
            "oct27_29_log_loss": float(pooled(p, SCORE).mean()), "oct30_log_loss": float(pooled(p, (141030,)).mean()),
            "per_seed": {t["seed"]: {"actions": {_label(d): a for d, a in t["actions"].items()}, "refits": t["refits"],
                                     "oct27_29": float(np.concatenate([t["loss"][d] for d in SCORE]).mean()),
                                     "oct30": float(t["loss"][141030].mean()),
                                     **({"jev": {_label(d): v for d, v in t["jev_detail"].items()},
                                         "budget_violations": t["budget_violations"]} if p == "jev" else {})}
                         for t in mine}}
    for base in POLICIES[1:]:
        for span, name in ((SCORE, "oct27_29"), ((141030,), "oct30")):
            out["comparisons"].append({"candidate": "jev", "baseline": base, "span": name,
                                       "char_ci": xtrain.bootstrap_diff(groups[span], pooled("jev", span),
                                                                        pooled(base, span))})
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["smoke"]:
        print(json.dumps(run(smoke=True), indent=1))
        sys.exit()
    out = run()
    Path("reports/x13_retrain.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=float))
