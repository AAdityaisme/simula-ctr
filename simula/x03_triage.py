"""x03, feature triage from definitions alone (Fable E1): can Jev tell which candidate features will help B?

Jev reads an assignment excerpt, B's feature list and one feature card (name, one-line definition, availability
note). It never sees a row or a statistic. Each of the 12 zoo features is then tested by fitting B + f on
train.py's recipe (early stopping on Oct 27, refit through Oct 27, calibrate on Oct 28, test Oct 29-30), with both
fitting windows subsampled to 300,000 rows. v2 (after the red team on 1d613cd): every arm under 5 seeds, the
target is the mean gain over seeds for admissible features only, and availability is scored separately. Run:
  uv run python -m simula.x03_triage states     # Jev input JSONL, outside the checkout
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x03_triage run        # reports/x03_feature_triage_v2.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from simula.data import add_flags, load
from simula.features import FEATURE_SETS
from simula.jev_ctr import _row_loss, bootstrap_diff
from simula.rank_pairs import read_answers
from simula.x_judge import (DATA, WORK, b_recipe, day_bounds, extend_numeric, hour_index, range_sum, trailing_ctr,
                             windows)

ANSWERS = Path("reports/x03_feature_triage_jev_answers.jsonl")
REPEATS, N_RANDOM, M = 3, 20, 20
EARLIER_TODAY = "Counted from impressions in strictly earlier hours of the same UTC day; rows in the same hour are " \
                "not counted because they have no order."
# (name, definition, availability note, available at decision time by construction)
ZOO = [
    ("prior_impressions_today_any_ad", "Number of earlier impressions served today to the same device "
     "(same device_ip and device_model), for any ad.", EARLIER_TODAY, True),
    ("prior_impressions_today_this_creative", "Number of earlier impressions of this same creative (C14) served "
     "today to the same device (same device_ip and device_model).", EARLIER_TODAY, True),
    ("hours_since_creative_first_seen", "Hours since this creative (C14) first appeared in the impression log.",
     "Computed from the log up to this impression's hour; the log starts Oct 21.", True),
    ("ip_volume_today", "Number of impressions from the same device_ip today, any device model, any ad.",
     EARLIER_TODAY, True),
    ("device_model_rarity", "How common this device_model is: its number of impressions in all earlier hours of "
     "the log.", "Counted from strictly earlier hours of the log.", True),
    ("c20_is_minus_one", "1 if the anonymized ad attribute C20 equals -1 (its most common value), else 0.",
     "Read from the candidate ad at serving time.", True),
    ("conversation_turn", "Message index in the chat session when the ad was rendered (1, 2, ...).",
     "Taken from the impression record.", True),
    ("turn_share_of_session", "conversation_turn divided by session_msg_count, where session_msg_count is the "
     "total number of messages in the session.", "Both values taken from the impression record.", False),
    ("safety_tier_ordinal", "The character's safety tier as an ordered number: sfw=0, suggestive=1, mature=2.",
     "From the character table.", True),
    ("character_trailing_ctr", "The character's click-through rate over all earlier days, smoothed toward the "
     "overall rate with 20 pseudo-impressions.", "Uses clicks from strictly earlier days only.", True),
    ("num_interactions", "Aggregated number of user chats with the character.",
     "One value per character in the character table export; the table has no snapshot time.", False),
    ("session_msg_count", "Total messages in the chat session.", "Taken from the impression record.", False),
]
NAMES = [z[0] for z in ZOO]
ADMISSIBLE = [z[0] for z in ZOO if z[3]]  # v2: only these are ranked; availability is scored on its own
SEEDS = [20260920 + s for s in range(5)]
COLUMNS = [f"zoo_{n}" for n in NAMES]
TASK = ("Simula serves contextual ads inside AI companion apps. Every impression is tied to a character (the AI "
        "persona the user is chatting with) and the conversation state at the moment of impression. Goal: predict "
        "P(click) for each impression so candidate ads can be ranked. Model B is a LightGBM click model trained on "
        "Oct 21-27 and scored on Oct 29-30. A proposed feature is added to B and B is refit the same way; the "
        "question is how much the feature changes B's log loss on the later days.")
B_FEATURES = ("request: hour_of_day, site_id, site_domain, site_category, app_id, app_domain, app_category, "
              "device_model, device_type, device_conn_type, is_null_device, is_site_side, C1; candidate ad: "
              "banner_pos, C14-C21 (anonymized creative and advertiser attributes); character: safety_tier, "
              "creator_type, genre (name prefix), character_age_days")
QUESTIONS = {
    "expected_gain": {"type": "score", "criteria": ["hurts", "none", "small", "medium", "large"],
                      "instructions": "If this feature is added to model B, how much will it improve B's log loss "
                                      "on the later days? hurts = makes it worse, none = no change."},
    "available_at_decision_time": {"type": "noul", "criteria": None,
                                   "instructions": "Is this feature's value known at the moment the ad is chosen, "
                                                   "before anything that happens after the impression?"},
}
extend_numeric(COLUMNS)


def add_zoo(df):
    """Add the 12 zoo columns. Click-derived values use no test-window (Oct 29-30) labels."""
    t = hour_index(df["hour"])
    day_lo, _ = day_bounds(df)
    today = lambda keys: range_sum(df, keys, day_lo, t - 1)  # noqa: E731
    first_seen = df.assign(_t=t).groupby("C14")["_t"].transform("min")
    zoo = {
        "prior_impressions_today_any_ad": today(["device_ip", "device_model"]),
        "prior_impressions_today_this_creative": today(["device_ip", "device_model", "C14"]),
        "hours_since_creative_first_seen": t - first_seen.to_numpy(),
        "ip_volume_today": today(["device_ip"]),
        "device_model_rarity": range_sum(df, ["device_model"], -np.inf, t - 1),
        "c20_is_minus_one": df["c20_is_sentinel"].astype(float),
        "conversation_turn": df["conversation_turn"].astype(float),
        "turn_share_of_session": df["conversation_turn"] / df["session_msg_count"],
        "safety_tier_ordinal": df["safety_tier"].map({"sfw": 0, "suggestive": 1, "mature": 2}).astype(float),
        # v2: prior = overall CTR of earlier days, not of all of Oct 21-28 (red team, 1d613cd)
        "character_trailing_ctr": trailing_ctr(df, ["character_id"], M, df["day"] < 141029),
        "num_interactions": df["num_interactions"].astype(float),
        "session_msg_count": df["session_msg_count"].astype(float),
    }
    return df.assign(**{f"zoo_{k}": np.asarray(v, dtype=float) for k, v in zoo.items()})


def state(card):
    """Jev state for one zoo card: task, B's features and the card. No rows, no statistics."""
    name, definition, availability, _ = card
    return {"task": TASK, "model_b_features": B_FEATURES,
            "proposed_feature": {"name": name, "definition": definition, "availability": availability}}


def write_states(path):
    with open(path, "w") as f:
        for card in ZOO:
            for r in range(REPEATS):
                f.write(json.dumps({"id": f"{card[0]}:{r}", "state": state(card), "questions": QUESTIONS}) + "\n")


def jev_ranking(answers):
    """Mean expected rung over the repeats per feature, the share answering 'available', and the top 3.

    Top 3 = highest mean score among ADMISSIBLE features; ties keep ZOO order (pre-registered, v2).
    """
    score = {n: float(np.mean([answers[f"{n}:{r}"]["answers"]["expected_gain"]["score"] for r in range(REPEATS)]))
             for n in NAMES}
    available = {n: float(np.mean([answers[f"{n}:{r}"]["answers"]["available_at_decision_time"]["noul"]
                                   for r in range(REPEATS)])) for n in NAMES}
    top3 = sorted(ADMISSIBLE, key=lambda n: -score[n])[:3]
    return score, available, top3


def random_draws(seed=0):
    """The 20 pre-registered random 3-feature sets, drawn from the admissible features (v2)."""
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(ADMISSIBLE, 3, replace=False).tolist(), key=NAMES.index) for _ in range(N_RANDOM)]


def spearman_with_ci(x, y, n_boot=2000, seed=0):
    """Spearman rho, exact-enough permutation p (10,000 shuffles) and a bootstrap-over-features 95% interval."""
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(10_000)])
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            boots.append(spearmanr(x[i], y[i]).statistic)
    return {"rho": float(rho), "p_one_sided": float(np.mean(perm >= rho)),
            "ci": [float(v) for v in np.percentile(boots, [2.5, 97.5])]}


def run(data_dir, answers):
    """v2: every arm refit under the 5 SEEDS. A feature's gain is its mean over seeds of B's test log loss minus
    B + f's, with a t-interval over seeds; only admissible features enter the ranking and the set arms."""
    df = add_zoo(add_flags(load(data_dir)))
    w = windows(df)
    y, chars = w["test"]["click"].to_numpy(), w["test"]["character_id"]
    score, available, top3 = jev_ranking(answers)
    seed_loss, row_loss, trees, splits = {}, {}, {}, {}

    def arm(name, feats):
        for s in SEEDS:
            r = b_recipe(w, [f"zoo_{f}" for f in feats], seed=s)
            loss = _row_loss(y, r["p"])
            seed_loss[(name, s)], trees[(name, s)], splits[(name, s)] = float(loss.mean()), r["trees"], r["splits"]
            row_loss[name] = row_loss.get(name, 0) + loss / len(SEEDS)
            print(name, feats, s, r["trees"], round(loss.mean(), 6), r["splits"], flush=True)

    def gains(name, ref="B"):
        return [seed_loss[(ref, s)] - seed_loss[(name, s)] for s in SEEDS]

    arm("B", [])
    for n in ADMISSIBLE:
        arm(n, [n])
    singles = [{"feature": n, "jev_score": score[n], "jev_available_share": available[n],
                "seed_gains": gains(n), "gain": float(np.mean(gains(n))), "seed_ci": seed_interval(gains(n)),
                "trees_per_seed": [trees[(n, s)] for s in SEEDS],
                "splits_per_seed": [splits[(n, s)][f"zoo_{n}"] for s in SEEDS],
                "char_ci_seed_mean": list(bootstrap_diff(chars, row_loss["B"], row_loss[n]))[1:]}
               for n in ADMISSIBLE]
    oracle3 = [s["feature"] for s in sorted(singles, key=lambda s: -s["gain"])][:3]
    arm("jev_top3", top3)
    arm("oracle_top3", oracle3)
    draws = random_draws()
    for d, feats in enumerate(draws):
        arm(f"random3_{d}", feats)
    random_names = [f"random3_{d}" for d in range(N_RANDOM)]
    jev_minus_random = [seed_loss[("jev_top3", s)] - np.mean([seed_loss[(k, s)] for k in random_names])
                        for s in SEEDS]
    sets = [{"arm": k, "gain_vs_B": float(np.mean(gains(k))), "seed_ci": seed_interval(gains(k))}
            for k in ["jev_top3", "oracle_top3"] + random_names]
    rho = spearman_with_ci([score[n] for n in ADMISSIBLE], [s["gain"] for s in singles])
    seed_ci = seed_interval(jev_minus_random)
    calls = list(answers.values())
    return {
        "seeds": SEEDS, "test_rows": len(y), "fit_rows": {k: len(w[k]) for k in ("dev_fit", "refit")},
        "admissible": ADMISSIBLE, "singles": singles, "sets": sets, "jev_top3": top3, "oracle_top3": oracle3,
        "random_draws": draws, "B_seed_losses": [seed_loss[("B", s)] for s in SEEDS],
        "spearman_jev_vs_gain": rho,
        "jev_top3_minus_random3_loss": {"seed_diffs": jev_minus_random, "mean": float(np.mean(jev_minus_random)),
                                        "seed_ci": seed_ci,
                                        "char_ci_seed_mean": list(bootstrap_diff(
                                            chars, row_loss["jev_top3"],
                                            np.mean([row_loss[k] for k in random_names], axis=0)))},
        "jev_top3_rank_among_random": int(sum(np.mean([seed_loss[(k, s)] for s in SEEDS])
                                              < np.mean([seed_loss[("jev_top3", s)] for s in SEEDS])
                                              for k in random_names)),
        "availability": {"accuracy": float(np.mean([(available[n] >= 0.5) == z[3] for n, z in zip(NAMES, ZOO)])),
                         "share_available": available},
        "win": bool(rho["rho"] >= 0.5 and seed_ci[1] < 0),
        "jev": {"calls": len(calls), "models": sorted({a["model"] for a in calls}),
                "input_tokens": int(sum(a["tokens_in"] for a in calls))},
    }


def seed_interval(gains):
    """Mean and 95% t-interval over seeds; the character bootstrap cannot see seed-to-seed variance."""
    from scipy.stats import t
    g = np.asarray(gains)
    half = t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
    return [float(g.mean() - half), float(g.mean() + half)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "run"])
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        write_states(WORK / "x03_states.jsonl")
        print("wrote", WORK / "x03_states.jsonl")
    else:
        out = run(args.data, read_answers(ANSWERS))
        Path("reports/x03_feature_triage_v2.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k not in ("sets", "random_draws")}, indent=1, default=float))
