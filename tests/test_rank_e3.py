import json

import pandas as pd

from simula.rank_e3 import both_unseen, count_weighted_mean, state
from simula.rank_pairs import add_history, build_pairs, history


def _frame(rows):
    df = pd.DataFrame(rows, columns=["id", "hour", "character_id", "app_id", "site_id", "device_type", "C14", "click"])
    df["banner_pos"], df["genre"], df["character_description"] = 0, "romance", "A warm companion."
    df["app_category"], df["site_category"], df["hour_of_day"] = "appcat", "sitecat", df["hour"] % 100
    for i, c in enumerate(["C15", "C16", "C17", "C18", "C19", "C20", "C21"]):
        df[c] = 10 + i
    df["C17"] = 99917
    return df


CTX = (14102901, "ch1", "app", "site", 1)
TEST = _frame([
    # one group, three creatives: 777001 was seen in refit, 777002 and 777003 were not
    ("a", *CTX, 777001, 1), ("b", *CTX, 777002, 1), ("c", *CTX, 777003, 0), ("d", *CTX, 777001, 0),
])
REFIT = _frame([("r1", 14102701, "ch1", "app", "site", 1, 777001, 1), ("r2", 14102701, "ch9", "x", "y", 1, 555, 0)])


def _rows():
    keys = ["C14", "genre_x_C18", "app_category_x_C18", "banner_pos", "C15", "C16", "C18", "C19", "C20", "C21"]
    return add_history(TEST, history(REFIT), keys)


def test_both_unseen_keeps_only_pairs_of_two_new_creatives():
    rows = _rows()
    pairs = both_unseen(build_pairs(TEST), rows)
    assert [frozenset(TEST.loc[[p.x1, p.x2], "id"]) for p in pairs.itertuples()] == [frozenset("bc")]


def test_state_has_no_c17_creative_id_or_click():
    rows = _rows()
    for p in build_pairs(TEST).itertuples():
        s = state((rows.loc[p.x1], rows.loc[p.x2]))
        text = json.dumps(s)
        assert "C17" not in text and "99917" not in text and "777" not in text and "click" not in text
        assert "model_pctr" not in text and "creative_history" not in text
        assert s["candidates"]["X1"]["C15"] == {"value": "10", "impressions": 2, "smoothed_ctr": 0.5}


def test_count_weighted_mean():
    keys = ["banner_pos", "C15", "C16", "C18", "C19", "C20", "C21", "genre_x_C18", "app_category_x_C18"]
    cand = {k: {"impressions": 0, "smoothed_ctr": 0.9} for k in keys}
    assert count_weighted_mean(cand, prior=0.17) == 0.17
    cand["C15"] = {"value": "320", "impressions": 300, "smoothed_ctr": 0.1}
    cand["genre_x_C18"] = {"impressions": 100, "smoothed_ctr": 0.5}
    assert abs(count_weighted_mean(cand, prior=0.17) - (300 * 0.1 + 100 * 0.5) / 400) < 1e-12
