import json

import numpy as np
import pandas as pd

from simula.rank_pairs import add_history, build_pairs, ceiling, history, state


def _frame(rows):
    cols = ["id", "hour", "character_id", "app_id", "site_id", "device_type", "C14", "click"]
    df = pd.DataFrame(rows, columns=cols)
    df["banner_pos"], df["genre"], df["character_description"] = 0, "romance", "A warm companion."
    df["app_category"], df["site_category"], df["hour_of_day"] = "appcat", "sitecat", df["hour"] % 100
    for i, c in enumerate(["C15", "C16", "C17", "C18", "C19", "C20", "C21"]):
        df[c] = 10 + i
    return df


TEST = _frame([
    # group 1: two clicks, two non-clicks, two creatives -> pairs (a,d) and (b,c); (a,c) and (b,d) share a C14
    ("a", 14102901, "ch1", "app", "site", 1, 777001, 1), ("b", 14102901, "ch1", "app", "site", 1, 777002, 1),
    ("c", 14102901, "ch1", "app", "site", 1, 777001, 0), ("d", 14102901, "ch1", "app", "site", 1, 777002, 0),
    # group 2: no click -> no pairs
    ("e", 14102901, "ch2", "app", "site", 1, 777003, 0), ("f", 14102901, "ch2", "app", "site", 1, 777004, 0),
    # group 3: mixed labels but one creative -> no pairs
    ("g", 14102902, "ch1", "app", "site", 1, 777005, 1), ("h", 14102902, "ch1", "app", "site", 1, 777005, 0),
])


def test_pairs_are_mixed_label_different_creative_and_order_ignores_the_click():
    pairs = build_pairs(TEST)
    ids = {frozenset(TEST.loc[[x1, x2], "id"]) for x1, x2 in zip(pairs["x1"], pairs["x2"])}
    assert ids == {frozenset("ad"), frozenset("bc")}
    for p in pairs.itertuples():
        clicked = p.x1 if p.clicked == "X1" else p.x2
        assert TEST.loc[clicked, "click"] == 1 and TEST.loc[p.x1, "C14"] != TEST.loc[p.x2, "C14"]
    flipped = TEST.assign(click=1 - TEST["click"])
    again = build_pairs(flipped).set_index("pair")
    for p in pairs.itertuples():
        assert (again.loc[p.pair, "x1"], again.loc[p.pair, "x2"]) == (p.x1, p.x2)
        assert again.loc[p.pair, "clicked"] != p.clicked


def test_ceiling():
    assert ceiling(np.array([0.3]), np.array([0.3])) == 0.5
    p_hi, p_lo = 0.4, 0.1
    expected = p_hi * (1 - p_lo) / (p_hi * (1 - p_lo) + p_lo * (1 - p_hi))
    assert np.isclose(ceiling(np.array([p_lo, 0.2]), np.array([p_hi, 0.2])), (expected + 0.5) / 2)


def test_state_has_no_click_or_creative_id():
    refit = TEST.assign(hour=14102701)
    rows = add_history(TEST, history(refit)).assign(pctr=0.123)
    for p in build_pairs(TEST).itertuples():
        for with_pctr in (True, False):
            s = state((rows.loc[p.x1], rows.loc[p.x2]), with_pctr)
            text = json.dumps(s)
            assert "click\"" not in text.replace("clicks\"", "") and "777" not in text
            assert ("model_pctr" in text) == with_pctr
            assert s["candidates"]["X1"]["creative_history"]["impressions"] == 2
