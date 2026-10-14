import json

import numpy as np
import pandas as pd

from simula import x03_triage as x03, x04_leaks as x04, x05_exam as x05, x_judge
from simula.x_judge import day_bounds, hour_index, range_sum

TOY = pd.DataFrame({"hour": [14102100, 14102100, 14102101, 14102105, 14102200, 14102201],
                    "k": ["a", "a", "a", "b", "a", "a"], "click": [1, 0, 1, 1, 0, 1]})


def test_range_sum_windows():
    t, (lo, hi) = hour_index(TOY["hour"]), day_bounds(TOY)
    assert range_sum(TOY, ["k"], lo, t - 1).tolist() == [0, 0, 2, 0, 0, 1]  # same hour has no order
    assert range_sum(TOY, ["k"], lo, hi).tolist() == [3, 3, 3, 1, 2, 2]
    assert range_sum(TOY, ["k"], -np.inf, lo - 1, "click").tolist() == [0, 0, 0, 0, 2, 2]
    assert range_sum(TOY, ["k"], t + 1, t + 24).tolist() == [2, 2, 2, 0, 1, 0]


def test_x03_state_has_no_rows_and_top3_breaks_ties_in_zoo_order():
    assert set(x03.state(x03.ZOO[0])["proposed_feature"]) == {"name", "definition", "availability"}
    answers = {f"{n}:{r}": {"answers": {"expected_gain": {"score": 2.0 if i in (4, 9, 10, 11) else 1.0},
                                        "available_at_decision_time": {"noul": True}}}
               for i, n in enumerate(x03.NAMES) for r in range(x03.REPEATS)}
    _, _, top3 = x03.jev_ranking(answers)
    assert top3 == [x03.NAMES[4], x03.NAMES[9], x03.NAMES[0]]  # v2: inadmissible 10 and 11 are not ranked
    draws = x03.random_draws()
    assert draws == x03.random_draws() and len(draws) == 20 and all(len(set(d)) == 3 for d in draws)
    assert set(sum(draws, [])) <= set(x03.ADMISSIBLE)


def _card(name, truth, window="row", sources=None, kind="constructed", mi=0.001):
    return {"name": name, "definition": "d", "sources": sources or [name], "window": window, "truth": truth,
            "kind": kind, "stats": {"mi_with_click": mi}}


def test_x04_lineage_rule_and_masking():
    cards = [_card("session_msg_count", "unavailable_at_decision_time", kind="real"),
             _card("log_turn", "safe", sources=["conversation_turn"]),
             _card("ctr_prev_day", "safe", "earlier_days", ["click", "C14"]),
             _card("ctr_earlier_today", "leaky", "earlier_hours_today", ["click", "C14"]),
             _card("imps_whole_day", "unavailable_at_decision_time", "same_day", ["C14"])]
    shown, masked = x04.lineage_rule(cards, False), x04.lineage_rule(cards, True)
    assert shown["session_msg_count"] != "safe" and masked["session_msg_count"] == "safe"  # name hidden
    assert shown["log_turn"] == "safe"  # v2: conversation_turn is a modelling exclusion, not an admission bar
    assert shown["ctr_prev_day"] == "safe" and shown["ctr_earlier_today"] == "leaky"
    assert shown["imps_whole_day"] == "unavailable_at_decision_time"
    s = x04.score(cards, masked)
    assert s["leak_recall"] == 2 / 3 and s["false_rejection"] == 0 and s["missed_leaks"] == ["session_msg_count"]


def test_x04_state_hides_truth_and_mi_rule_flags_outlier():
    cards = [_card(f"c{i}", "safe", mi=0.001 + 0.0001 * i) for i in range(20)] + [_card("plant", "leaky", mi=0.4)]
    assert x04.mi_outlier_rule(cards) == {**{f"c{i}": "safe" for i in range(20)}, "plant": "leaky"}
    text = json.dumps(x04.state(cards[-1], "f07"))
    assert "plant" not in text and "leaky" not in text and "sources" not in text and "window" not in text


def _pool(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"day": rng.choice([141021, 141022, 141023], n), "C14": rng.integers(0, 200, n).astype(str),
                         "C17": rng.integers(0, 80, n).astype(str), "C21": "1",
                         "character_id": rng.integers(0, 300, n).astype(str),
                         "app_id": rng.choice(["ecad2386"] + [f"a{i}" for i in range(60)], n)})


def test_x05_splits_are_equal_sized_and_held_out_values_disjoint():
    splits = x05.make_splits(_pool())
    assert len({len(f) for f, _ in splits.values()}) == 1 and len({len(v) for _, v in splits.values()}) == 1
    for name, field in x05.FIELDS.items():
        fit_rows, valid = splits[name]
        assert not set(fit_rows[field]) & set(valid[field])
    assert "ecad2386" not in set(splits["heldout_app"][1]["app_id"])
    masked = x05.joint_mask(_pool())
    blank = masked["C14"].isna()
    assert blank.any() and masked.loc[blank, ["C17", "C21"]].isna().all().all()


def test_x05_regret_and_novelty_matching():
    table = pd.DataFrame({"split": ["chronological"] * 2 + ["heldout_C17"] * 2, "recipe": ["B", "L15"] * 2,
                          "valid_loss": [0.40, 0.41, 0.50, 0.49], "later_loss": [0.45, 0.44, 0.43, 0.42]})
    assert x05.regret(table, "chronological")["recipe"] == "B"
    assert abs(x05.regret(table, "chronological")["recipe_regret"] - 0.01) < 1e-12  # 0.44 vs 0.43 mean
    assert x05.regret(table, "heldout_C17")["recipe_regret"] == 0.0
    assert abs(x05.regret(table, "heldout_C17")["fit_regret"]) < 1e-12
    churn = {"pool_last_day_churn": {"C14": 0.4, "C17": 0.3, "character_id": 0.01, "app_id": 0.005}}
    assert x05.novelty_matching(churn) == "heldout_C17"


def test_trailing_ctr_and_frozen_decile_ignore_later_clicks():
    rng = np.random.default_rng(1)
    day = rng.choice([141021, 141022, 141023, 141024], 400)
    df = pd.DataFrame({"hour": day * 100 + rng.integers(0, 24, 400), "day": day, "k": rng.choice(["a", "b"], 400),
                       "click": rng.integers(0, 2, 400)})
    later = df.assign(click=np.where(df["day"] >= 141023, 1 - df["click"], df["click"]))
    early = (df["day"] <= 141022).to_numpy()
    a, b = x_judge.trailing_ctr(df, ["k"], 20), x_judge.trailing_ctr(later, ["k"], 20)
    assert np.isnan(a[day == 141021]).all() and np.allclose(a[early & (day > 141021)], b[early & (day > 141021)])
    assert not np.allclose(a[day == 141024], b[day == 141024])  # Oct 24 rows do see the changed Oct 23 clicks
    masked = x_judge.trailing_ctr(df, ["k"], 20, df["day"] < 141023)
    unlabelled_ignored = x_judge.trailing_ctr(later, ["k"], 20, df["day"] < 141023)
    assert np.allclose(masked, unlabelled_ignored, equal_nan=True)
    x = rng.random(400)
    d1, d2 = x_judge.frozen_decile(x, day), x_judge.frozen_decile(np.where(day >= 141023, 1 - x, x), day)
    assert (d1[day == 141021] == -1).all() and (d1[day <= 141022] == d2[day <= 141022]).all()
    assert set(d1[day > 141021]) <= set(range(10))
