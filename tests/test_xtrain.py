import json

import numpy as np
import pandas as pd
import pytest

from simula import xtrain
from simula.x12_stopping import options
from simula.x13_retrain import hour_z, rule_action
from simula.x14_labels import largest_remainder
from simula.x15_curation import clean_slice, profiles, rule, run


def test_subsample_caps_rows_and_is_seeded():
    df = pd.DataFrame({"a": range(10)})
    assert len(xtrain.subsample(df, 0, cap=4)) == 4
    assert xtrain.subsample(df, 0, cap=4).index.tolist() == xtrain.subsample(df, 0, cap=4).index.tolist()
    assert xtrain.subsample(df, 0, cap=20) is df


def test_stopping_options_from_curve():
    valid = np.r_[np.linspace(0.50, 0.40, 100), np.full(100, 0.40) + np.linspace(0, 0.01, 100)]
    valid[99] = 0.399
    opts = options(valid, se=0.005)
    assert opts["argmin"] == 100 and opts["minus_20"] == 80 and opts["plus_10"] == 110 and opts["max"] == 200
    assert opts["one_se"] == int(np.argmax(valid <= 0.404)) + 1 < 100
    assert opts["plateau_start"] <= 100


def test_retrain_rules_respect_budget():
    t = {"refits": 2, "ewma": 0.0}
    assert rule_action("ewma", 141028, t, {"logit_gap": 1.0}) == "intercept_update"
    t = {"refits": 0, "ewma": 0.15}
    assert rule_action("ewma", 141028, t, {"logit_gap": 0.1}) == "refit_recent_three_days"
    assert rule_action("hour_z", 141028, {"refits": 0}, {"hour_z": 1.0}) == "hold"
    assert rule_action("hour_z", 141028, {"refits": 2}, {"hour_z": 5.0}) == "intercept_update"


def test_hour_z_is_zero_when_calibrated_and_signed():
    hours, p = np.repeat([0, 1], 1000), np.full(2000, 0.2)
    y = np.tile(np.r_[np.ones(200), np.zeros(800)], 2)
    assert abs(hour_z(hours, y, p)) < 1e-9
    assert hour_z(hours, y, np.full(2000, 0.1)) > 3


def test_largest_remainder_sums_to_total():
    out = largest_remainder([1, 1, 1], 20_000)
    assert out.sum() == 20_000 and out.max() - out.min() <= 1


def test_curation_profiles_rule_and_clean_slice():
    rows = [("d1", 141027, h % 24, 1, "ipA") for h in range(25)] + [("d2", 141027, 3, 0, "ipB")] * 6 + \
           [("d3", 141027, 5, 1, "ipC")] * 4 + [("a99f214a", 141027, 1, 0, "ipD")] * 9
    df = pd.DataFrame(rows, columns=["device_id", "day", "hour", "click", "device_ip"])
    df = df.assign(is_null_device=df["device_id"].eq("a99f214a"), C14="c", app_id="a", site_id="s",
                   device_model="m", is_site_side=True)
    p = profiles(df)
    assert p["device_id"].tolist() == ["d1", "d2"]
    assert rule(p).tolist() == [True, False]
    assert clean_slice(df).tolist() == [False] * 25 + [False] * 6 + [False] * 4 + [False] * 9
    assert clean_slice(df.iloc[[0, 25, 26, 31]]).tolist() == [True, True, True, True]


@pytest.mark.skipif(not xtrain.DATA.exists(), reason="needs the take-home export")
def test_x15_replay_reproduces_published_losses():
    published = json.loads(open("reports/x15_curation.json").read())["arms"]
    replayed = run(replay=True)["arms"]
    for arm, row in published.items():
        assert replayed[arm]["profiles_flagged"] == row["profiles_flagged"]
        assert replayed[arm]["rows_excluded"] == row["rows_excluded"]
        for k in ("test_log_loss", "clean_log_loss"):
            assert replayed[arm][k] == pytest.approx(row[k], rel=1e-9)
