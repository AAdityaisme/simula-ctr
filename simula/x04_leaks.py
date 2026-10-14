"""x04, leak and availability auditor (Fable E2 merged with Astra 1): can Jev veto time travel from a card?

Cards: every real column of the two CSVs, plus constructed recipe cards whose truth is fixed by construction (Fable's
three plants, Astra's future aggregates and nested aliases, legal lagged versions, identity recipes). Each card is
materialized on the refit window (Oct 21-27) only, to compute Fable's four statistics; no test-window row is read.
Jev answers one Choice per card, in two arms: names shown and names masked (the field renamed f07, its definition
kept). Controls: Fable's MI-outlier rule, and Astra's denylist (the policy-barred part of CONTRACT["never"]) plus a
timestamp/lineage rule. v2 (after the red team on 1d613cd): three legal cards rebuilt from decision-time values only,
and one admission policy for the gold labels and the denylist.
Run:
  uv run python -m simula.x04_leaks states
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x04_leaks report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score

from simula.data import add_flags, load, split
from simula.features import CONTRACT
from simula.rank_pairs import read_answers
from simula.x_judge import DATA, WORK, day_bounds, frozen_decile, hour_index, range_sum, trailing_ctr

ANSWERS = Path("reports/x04_leak_auditor_v2_jev_answers.jsonl")
STATS = WORK / "x04_card_stats_v2.json"
LABELS = {"safe": "may be used: known at the moment the ad is chosen and not derived from outcomes that are not yet "
                  "known",
          "leaky": "derived from click outcomes (this impression's or others') that are not yet known when the ad "
                   "is chosen",
          "unavailable_at_decision_time": "uses information that only exists after the ad is chosen (later events, "
                                          "session or export totals), though not click outcomes",
          "identity_restricted": "identifies or tracks an individual user or device; barred by the identity policy"}
QUESTION = {"admission": {"type": "choice", "criteria": LABELS,
                          "instructions": "May this field be used as an input to the click model, which predicts "
                                          "P(click) at the moment the ad is chosen? Pick the one label that fits."}}
CONTEXT = ("Simula serves ads inside AI companion apps. A click model predicts P(click) for an impression at the "
           "moment the ad is chosen, from fields known at that moment. The log has one row per impression, with an "
           "hour (UTC, YYMMDDHH) and a click outcome. Click outcomes for a day become available at the end of that "
           "day. Each impression joins to a character table. Statistics on the card are computed on Oct 21-27.")

# real columns: (name, definition from the assignment schema, truth)
REAL = [
    ("id", "Impression ID.", "safe"), ("hour", "YYMMDDHH, UTC timestamp when the ad was rendered.", "safe"),
    ("click", "Whether the user clicked the ad, 0 or 1.", "leaky"),
    ("banner_pos", "Slot position in the host surface.", "safe"),
    ("site_id", "Publisher surface id.", "safe"), ("site_domain", "Publisher surface domain.", "safe"),
    ("site_category", "Publisher surface category.", "safe"), ("app_id", "Host AI app id.", "safe"),
    ("app_domain", "Host AI app domain.", "safe"), ("app_category", "Host AI app category.", "safe"),
    ("device_id", "Device identifier.", "identity_restricted"),
    ("device_ip", "Device IP address (hashed).", "identity_restricted"),
    ("device_model", "Device model.", "safe"), ("device_type", "Device type.", "safe"),
    ("device_conn_type", "Device connection type.", "safe"),
    *[(c, "Anonymized categorical feature; assume it may represent creative or advertiser attributes.", "safe")
      for c in ["C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]],
    ("character_id", "The character (AI persona) the user is chatting with; joins to the character table.", "safe"),
    ("conversation_turn", "Message index in the session when the ad was rendered (1, 2, ...).", "safe"),
    ("session_msg_count", "Total messages in the session.", "unavailable_at_decision_time"),
    ("character_name", "Character table: short display name.", "safe"),
    ("character_description", "Character table: one-line description of the character's persona.", "safe"),
    ("safety_tier", "Character table: sfw / suggestive / mature.", "safe"),
    ("creator_type", "Character table: official / community.", "safe"),
    ("num_interactions", "Character table: aggregated number of user chats with this character.",
     "unavailable_at_decision_time"),
    ("created_at", "Character table: when the character was first published.", "safe"),
]
KNOWN_LEAKS = ["session_msg_count", "num_interactions"]
PLANTS = ["click_plus_noise", "click_shuffled_within_day", "device_impressions_whole_day"]


def _ctr(df, keys, lo, hi):
    clicks, shown = range_sum(df, keys, lo, hi, "click"), range_sum(df, keys, lo, hi)
    return np.where(shown > 0, clicks / np.maximum(shown, 1), np.nan)


def _decile(x):
    return pd.Series(x).rank(pct=True).mul(10).clip(upper=9.999).fillna(-1).astype(int).to_numpy()


def constructed(df):
    """The constructed cards: {name: (definition, sources, window, truth, values)}. `df` is the refit window.

    `window` is the lineage metadata the rule control reads: which rows the recipe aggregates, relative to the
    impression (row, earlier_days, earlier_hours_today, earlier_hours, previous_hour, same_hour, same_day, next_hours,
    whole_log, static).
    """
    t = hour_index(df["hour"])
    lo, hi = day_bounds(df)
    ALL = (-np.inf, np.inf)
    rng = np.random.default_rng(0)
    shuffled = df.groupby("day")["click"].transform(lambda s: rng.permutation(s.to_numpy()))
    turn, total = df["conversation_turn"].astype(float), df["session_msg_count"].astype(float)
    inter = df["num_interactions"].astype(float)
    char_quartile = df["character_id"].map(
        df.drop_duplicates("character_id").set_index("character_id")["num_interactions"].rank(pct=True)
        .mul(4).clip(upper=3.999).astype(int))
    age_weeks = (pd.to_datetime(df["day"].astype(str), format="%y%m%d") - df["created_at"].dt.normalize()).dt.days // 7
    c14_prev7 = _ctr(df, ["C14"], lo - 168, lo - 1)
    cards = {
        # Fable's plants
        "click_plus_noise": ("click plus Gaussian noise with standard deviation 0.5.", ["click"], "row", "leaky",
                             df["click"] + rng.normal(0, 0.5, len(df))),
        "click_shuffled_within_day": ("The click column randomly permuted among all rows of the same calendar day.",
                                      ["click"], "same_day", "leaky", shuffled),
        "device_impressions_whole_day": ("Number of impressions from the same device_ip and device_model over the "
                                         "whole calendar day of the impression.", ["device_ip", "device_model"],
                                         "same_day", "unavailable_at_decision_time",
                                         range_sum(df, ["device_ip", "device_model"], lo, hi)),
        # future aggregates of clicks
        "creative_ctr_same_day": ("Clicks divided by impressions for this C14 over all rows of the same calendar "
                                  "day.", ["click", "C14"], "same_day", "leaky", _ctr(df, ["C14"], lo, hi)),
        "creative_ctr_next_24h": ("Clicks divided by impressions for this C14 over the 24 hours after this "
                                  "impression's hour.", ["click", "C14"], "next_hours", "leaky",
                                  _ctr(df, ["C14"], t + 1, t + 24)),
        "character_ctr_whole_log": ("Clicks divided by impressions for this character_id over the whole log.",
                                    ["click", "character_id"], "whole_log", "leaky", _ctr(df, ["character_id"], *ALL)),
        "app_ctr_same_hour": ("Clicks divided by impressions for this app_id within the same hour, this impression "
                              "included.", ["click", "app_id"], "same_hour", "leaky", _ctr(df, ["app_id"], t, t)),
        "c14_mean_click_all_rows": ("Mean click of all rows with this C14 in the log, this impression included.",
                                    ["click", "C14"], "whole_log", "leaky", _ctr(df, ["C14"], *ALL)),
        "creative_ctr_earlier_today": ("Clicks divided by impressions for this C14 over earlier hours of the same "
                                       "day.", ["click", "C14"], "earlier_hours_today", "leaky",
                                       _ctr(df, ["C14"], lo, t - 1)),
        "app_ctr_previous_hour": ("Clicks divided by impressions for this app_id in the previous hour.",
                                  ["click", "app_id"], "previous_hour", "leaky", _ctr(df, ["app_id"], t - 1, t - 1)),
        # nested aliases of leaks
        "creative_same_day_ctr_decile": ("Decile (0-9) of: clicks divided by impressions for this C14 over the same "
                                         "calendar day.", ["click", "C14"], "same_day", "leaky",
                                         _decile(_ctr(df, ["C14"], lo, hi))),
        "site_rank_by_whole_log_ctr": ("Rank of this site_id among the sites active in this hour, ordered by "
                                       "that site's clicks per impression over the whole log.",
                                       ["click", "site_id"], "whole_log", "leaky",
                                       pd.Series(_ctr(df, ["site_id"], *ALL)).groupby(t).rank().to_numpy()),
        "log_device_model_clicks_next_6h": ("log(1 + clicks) for this device_model over the next 6 hours.",
                                            ["click", "device_model"], "next_hours", "leaky",
                                            np.log1p(range_sum(df, ["device_model"], t + 1, t + 6, "click"))),
        # future aggregates without clicks
        "creative_impressions_whole_day": ("Impressions of this C14 over the whole calendar day.", ["C14"],
                                           "same_day", "unavailable_at_decision_time", range_sum(df, ["C14"], lo, hi)),
        "hours_until_creative_last_seen": ("Hours from this impression until this C14's last appearance in the log.",
                                           ["C14", "hour"], "whole_log", "unavailable_at_decision_time",
                                           pd.Series(t).groupby(df["C14"].to_numpy()).transform("max").to_numpy() - t),
        "app_impressions_next_6h": ("Impressions for this app_id in the 6 hours after this impression's hour.",
                                    ["app_id"], "next_hours", "unavailable_at_decision_time",
                                    range_sum(df, ["app_id"], t + 1, t + 6)),
        "character_impressions_whole_log": ("Impressions for this character_id over the whole log.",
                                            ["character_id"], "whole_log", "unavailable_at_decision_time",
                                            range_sum(df, ["character_id"], *ALL)),
        "site_distinct_creatives_whole_day": ("Number of distinct C14 values shown on this site_id over the whole "
                                              "calendar day.", ["site_id", "C14"], "same_day",
                                              "unavailable_at_decision_time",
                                              df.groupby(["site_id", "day"])["C14"].transform("nunique")),
        # nested aliases of the two real leaks
        "turn_share_of_session": ("conversation_turn divided by session_msg_count.",
                                  ["conversation_turn", "session_msg_count"], "row", "unavailable_at_decision_time",
                                  turn / total),
        "remaining_messages": ("session_msg_count minus conversation_turn.", ["session_msg_count", "conversation_turn"],
                               "row", "unavailable_at_decision_time", total - turn),
        "session_length_decile": ("Decile (0-9) of session_msg_count.", ["session_msg_count"], "row",
                                  "unavailable_at_decision_time", _decile(total)),
        "log_num_interactions": ("log(1 + num_interactions).", ["num_interactions"], "static",
                                 "unavailable_at_decision_time", np.log1p(inter)),
        "character_popularity_quartile": ("Quartile (0-3) of the character's num_interactions among all characters.",
                                          ["num_interactions"], "static", "unavailable_at_decision_time",
                                          char_quartile),
        # legal lagged versions and legal transforms
        "creative_ctr_previous_day": ("Clicks divided by impressions for this C14 on the previous calendar day.",
                                      ["click", "C14"], "earlier_days", "safe", _ctr(df, ["C14"], lo - 24, lo - 1)),
        # v2: the smoothing prior is the overall CTR of earlier days, not of the whole window (red team, 1d613cd)
        "creative_ctr_earlier_days": ("Clicks divided by impressions for this C14 over all earlier days, smoothed "
                                      "with 20 pseudo-impressions toward the overall click rate of all earlier days.",
                                      ["click", "C14"], "earlier_days", "safe", trailing_ctr(df, ["C14"], 20)),
        "character_ctr_earlier_days": ("Clicks divided by impressions for this character_id over all earlier days, "
                                       "smoothed with 20 pseudo-impressions toward the overall click rate of all "
                                       "earlier days.", ["click", "character_id"], "earlier_days", "safe",
                                       trailing_ctr(df, ["character_id"], 20)),
        "app_ctr_previous_day": ("Clicks divided by impressions for this app_id on the previous calendar day.",
                                 ["click", "app_id"], "earlier_days", "safe", _ctr(df, ["app_id"], lo - 24, lo - 1)),
        # v2: decile cutoffs come from earlier days' rows, not from ranking the whole window
        "creative_ctr_previous_7_days_decile": ("Decile (0-9) of: clicks divided by impressions for this C14 over the "
                                                "previous 7 calendar days, with the decile cutoffs taken from rows of "
                                                "earlier days only.", ["click", "C14"], "earlier_days", "safe",
                                                frozen_decile(c14_prev7, df["day"])),
        "creative_impressions_earlier_today": ("Impressions of this C14 in earlier hours of the same day.", ["C14"],
                                               "earlier_hours_today", "safe", range_sum(df, ["C14"], lo, t - 1)),
        "device_model_impressions_earlier": ("Impressions with this device_model over all earlier hours of the log.",
                                             ["device_model"], "earlier_hours", "safe",
                                             range_sum(df, ["device_model"], -np.inf, t - 1)),
        "hours_since_creative_first_seen": ("Hours since this C14 first appeared in the log.", ["C14", "hour"],
                                            "earlier_hours", "safe",
                                            t - pd.Series(t).groupby(df["C14"].to_numpy()).transform("min").to_numpy()),
        "site_impressions_previous_hour": ("Impressions for this site_id in the previous hour.", ["site_id"],
                                           "previous_hour", "safe", range_sum(df, ["site_id"], t - 1, t - 1)),
        "app_impressions_previous_day": ("Impressions for this app_id on the previous calendar day.", ["app_id"],
                                         "earlier_days", "safe", range_sum(df, ["app_id"], lo - 24, lo - 1)),
        "log_conversation_turn": ("log(1 + conversation_turn).", ["conversation_turn"], "row", "safe",
                                  np.log1p(turn)),
        "conversation_turn_bucket": ("conversation_turn bucketed: 1, 2-3, 4-7, 8 or more.", ["conversation_turn"],
                                     "row", "safe", np.digitize(turn, [2, 4, 8])),
        "character_age_weeks": ("Weeks between the character's created_at and the impression's day.",
                                ["created_at", "hour"], "row", "safe", age_weeks),
        "genre_from_name": ("The prefix of character_name before the first underscore.", ["character_name"],
                            "static", "safe", df["character_name"].str.split("_", n=1).str[0]),
        "hour_x_app_category": ("hour_of_day combined with app_category as one category.", ["hour", "app_category"],
                                "row", "safe", (df["hour"] % 100).astype(str) + "_" + df["app_category"]),
        "c14_x_c17": ("C14 and C17 combined as one category.", ["C14", "C17"], "row", "safe",
                      df["C14"].astype(str) + "_" + df["C17"].astype(str)),
        # identity
        "device_ip_prefix": ("The first 4 characters of the hashed device_ip.", ["device_ip"], "row",
                             "identity_restricted", df["device_ip"].str[:4]),
        "device_id_seen_earlier_day": ("1 if this device_id appeared on an earlier day, else 0.", ["device_id"],
                                       "earlier_days", "identity_restricted",
                                       (range_sum(df, ["device_id"], -np.inf, lo - 1) > 0).astype(int)),
        # Fable's probe; its truth is a policy call, so it is reported but not scored
        "ip_impressions_earlier_today": ("Impressions from the same device_ip in earlier hours of the same day.",
                                         ["device_ip"], "earlier_hours_today", "probe",
                                         range_sum(df, ["device_ip"], lo, t - 1)),
    }
    return cards


def _mi(values, y):
    """Mutual information (nats) with click: numeric values in 32 quantile bins, categories as the top 1000."""
    s = pd.Series(np.asarray(values)).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 32:
        codes = pd.qcut(s.rank(method="first"), 32, labels=False).fillna(-1)
    else:
        top = s.astype(str).value_counts().index[:1000]
        codes = s.astype(str).where(s.astype(str).isin(top), "_other")
    return float(mutual_info_score(y, codes))


def card_stats(values, df, char_count):
    """Fable's four statistics on the refit window: MI with click, Spearman with the character's impression count,
    share of rows above conversation_turn, cardinality. The last three are null where they do not apply."""
    s = pd.Series(np.asarray(values))
    numeric = pd.api.types.is_numeric_dtype(s)
    ok = s.notna().to_numpy()
    return {"mi_with_click": round(_mi(values, df["click"].to_numpy()), 5),
            "spearman_with_character_impressions": round(float(spearmanr(s[ok], char_count[ok]).statistic), 3)
            if numeric and s[ok].nunique() > 1 else None,
            "share_above_conversation_turn": round(float((s > df["conversation_turn"].to_numpy()).mean()), 3)
            if numeric else None,
            "cardinality": int(s.nunique())}


def build_cards(data_dir):
    """Every card with definition, metadata, truth and statistics, computed on the refit window only."""
    df = add_flags(load(data_dir))
    df = split(df, "refit").reset_index(drop=True)
    char_count = df.groupby("character_id")["id"].transform("size").to_numpy()
    cards = [{"name": n, "definition": d, "sources": [n], "window": "row", "truth": tr, "kind": "real",
              "stats": card_stats(df[n], df, char_count)} for n, d, tr in REAL]
    for n, (d, src, win, tr, values) in constructed(df).items():
        cards.append({"name": n, "definition": d, "sources": src, "window": win, "truth": tr, "kind": "constructed",
                      "stats": card_stats(values, df, char_count)})
    return cards


def masked_names(cards):
    """f01, f02, ... in a fixed shuffled order, so a masked name says nothing about the card's position or kind."""
    order = np.random.default_rng(7).permutation(len(cards))
    return {cards[i]["name"]: f"f{k + 1:02d}" for k, i in enumerate(order)}


def state(card, name):
    """Jev state for a card under a shown or masked name. Truth and lineage metadata are never in it."""
    return {"context": CONTEXT, "field": {"name": name, "definition": card["definition"], "statistics": card["stats"]}}


def write_states(cards, path):
    masks = masked_names(cards)
    reversed_q = {"admission": {**QUESTION["admission"], "criteria": dict(reversed(LABELS.items()))}}
    with open(path, "w") as f:
        for c in cards:
            f.write(json.dumps({"id": f"shown:{c['name']}", "state": state(c, c["name"]), "questions": QUESTION}) + "\n")
            f.write(json.dumps({"id": f"masked:{c['name']}", "state": state(c, masks[c["name"]]),
                                "questions": QUESTION}) + "\n")
            # Astra's shuffled-option check, on the masked arm
            f.write(json.dumps({"id": f"masked_rev:{c['name']}", "state": state(c, masks[c["name"]]),
                                "questions": reversed_q}) + "\n")


def mi_outlier_rule(cards):
    """Fable's control: reject a card whose MI with click is an outlier (modified z-score > 3.5 across all cards)."""
    mi = np.array([c["stats"]["mi_with_click"] for c in cards])
    mad = np.median(np.abs(mi - np.median(mi)))
    z = 0.6745 * (mi - np.median(mi)) / mad
    return {c["name"]: "leaky" if zi > 3.5 else "safe" for c, zi in zip(cards, z)}


FUTURE = {"same_hour", "same_day", "next_hours", "whole_log"}
# v2 admission policy, shared by the gold labels and the denylist control: a field may be used iff it is known when
# the ad is chosen, is not derived from clicks not yet known, and is not barred by the identity policy. The never-list
# also holds modelling choices this policy admits (id, character_name, character_description, created_at,
# conversation_turn), so the denylist control reads only the entries the policy bars.
POLICY_BARRED = {"click", "session_msg_count", "num_interactions", "device_id", "device_ip"}
assert POLICY_BARRED <= set(CONTRACT["never"])
LABEL_SAFE = {"earlier_days"}  # clicks for a day become available at its end


def lineage_rule(cards, masked, denylist=True):
    """Astra's control: the denylist (the never-list entries the admission policy bars) plus a timestamp/lineage rule.

    Denylist: a card whose name, or one of whose source columns, is on that list is rejected (device ids as
    identity_restricted). In the masked arm a real column's own name is hidden, so the denylist cannot see it.
    Timestamp: a recipe aggregating rows at or after the impression is unavailable; one reading click from rows whose
    labels are not yet available is leaky. With denylist=False only the timestamp part runs (an extra control).
    """
    never = POLICY_BARRED - {"click"} if denylist else set()  # label use is the timestamp part's job
    out = {}
    for c in cards:
        seen = [] if masked and c["kind"] == "real" else c["sources"]
        if denylist and {"device_id", "device_ip"} & set(seen):
            out[c["name"]] = "identity_restricted"
        elif "click" in seen and c["window"] not in LABEL_SAFE:
            out[c["name"]] = "leaky"
        elif c["window"] in FUTURE or never & set(seen):
            out[c["name"]] = "unavailable_at_decision_time"
        else:
            out[c["name"]] = "safe"
    return out


CONTROLS = ["mi_outlier", "denylist_lineage", "lineage_only", "union_mi_denylist_lineage"]


def controls(cards, masked):
    """The pre-registered controls' labels for one arm."""
    rule, mi = lineage_rule(cards, masked), mi_outlier_rule(cards)
    return {"mi_outlier": mi, "denylist_lineage": rule, "lineage_only": lineage_rule(cards, masked, denylist=False),
            "union_mi_denylist_lineage": {k: rule[k] if rule[k] != "safe" else mi[k] for k in rule}}


def score(cards, pred):
    """Leak recall, false rejection of legal cards, plants and known leaks caught, identity and 4-way accuracy."""
    leak = [c for c in cards if c["truth"] in ("leaky", "unavailable_at_decision_time")]
    legal = [c for c in cards if c["truth"] == "safe"]
    ident = [c for c in cards if c["truth"] == "identity_restricted"]
    scored = leak + legal + ident
    return {"leak_recall": float(np.mean([pred[c["name"]] != "safe" for c in leak])), "leak_cards": len(leak),
            "false_rejection": float(np.mean([pred[c["name"]] != "safe" for c in legal])), "legal_cards": len(legal),
            "missed_leaks": [c["name"] for c in leak if pred[c["name"]] == "safe"],
            "rejected_legal": [c["name"] for c in legal if pred[c["name"]] != "safe"],
            "plants_caught": all(pred[p] != "safe" for p in PLANTS if p in pred),
            "known_leaks_caught": all(pred[k] != "safe" for k in KNOWN_LEAKS if k in pred),
            "identity_rejected": float(np.mean([pred[c["name"]] != "safe" for c in ident])),
            "four_way_accuracy": float(np.mean([pred[c["name"]] == c["truth"] for c in scored]))}


def card_bootstrap(cards, pred_a, pred_b, n_boot=2000, seed=0):
    """Paired bootstrap over cards of (a - b) in leak recall and in false rejection: (diff, lo, hi) for each."""
    out = {}
    for key, truths in (("leak_recall", ("leaky", "unavailable_at_decision_time")), ("false_rejection", ("safe",))):
        names = [c["name"] for c in cards if c["truth"] in truths]
        d = np.array([(pred_a[n] != "safe") - (pred_b[n] != "safe") for n in names], dtype=float)
        boots = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(axis=1)
        out[key] = (float(d.mean()), *map(float, np.percentile(boots, [2.5, 97.5])))
    return out


def report(cards, answers):
    jev = {arm: {c["name"]: answers[f"{arm}:{c['name']}"]["answers"]["admission"]["choice"] for c in cards}
           for arm in ("shown", "masked", "masked_rev")}
    arms = {}
    for masked in (False, True):
        arm = "masked" if masked else "shown"
        arms[arm] = {"jev": score(cards, jev[arm]), **{k: score(cards, v) for k, v in controls(cards, masked).items()}}
    m = arms["masked"]
    paired = {k: card_bootstrap(cards, jev["masked"], v) for k, v in controls(cards, True).items()}
    eligible = [m[k]["leak_recall"] for k in CONTROLS if m[k]["false_rejection"] <= 0.02]
    best_control = max(eligible, default=0.0)
    calls = list(answers.values())
    return {
        "cards": len(cards), "by_truth": pd.Series([c["truth"] for c in cards]).value_counts().to_dict(),
        "arms": arms, "best_eligible_control_recall_masked": best_control, "masked_jev_minus_control": paired,
        "win": bool(m["jev"]["plants_caught"] and m["jev"]["known_leaks_caught"]
                    and m["jev"]["leak_recall"] > best_control and m["jev"]["false_rejection"] <= 0.02),
        "shuffled_option_agreement": float(np.mean([jev["masked"][c["name"]] == jev["masked_rev"][c["name"]]
                                                    for c in cards])),
        "shown_masked_agreement": float(np.mean([jev["shown"][c["name"]] == jev["masked"][c["name"]] for c in cards])),
        "probe": {arm: jev[arm]["ip_impressions_earlier_today"] for arm in jev},
        "answers": [{"name": c["name"], "kind": c["kind"], "truth": c["truth"], "window": c["window"],
                     **{arm: jev[arm][c["name"]] for arm in jev},
                     **{f"{k}_masked": v[c["name"]] for k, v in controls(cards, True).items()},
                     "mi_with_click": c["stats"]["mi_with_click"]} for c in cards],
        "jev": {"calls": len(calls), "models": sorted({a["model"] for a in calls}),
                "input_tokens": int(sum(a["tokens_in"] for a in calls)),
                "call_seconds_sum": float(sum(a["seconds"] for a in calls))},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.step == "states":
        cards = build_cards(args.data)
        STATS.write_text(json.dumps(cards, indent=1) + "\n")
        write_states(cards, WORK / "x04_v2_states.jsonl")
        print(len(cards), "cards; wrote", WORK / "x04_v2_states.jsonl")
    else:
        out = report(json.loads(STATS.read_text()), read_answers(ANSWERS))
        Path("reports/x04_leak_auditor_v2.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k != "answers"}, indent=1, default=float))
