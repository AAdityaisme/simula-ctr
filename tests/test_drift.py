import numpy as np
import pytest

from simula.data import add_flags, load, split
from simula.drift import adapt, daily_table, hour_matched, psi
from simula.features import FEATURE_SETS, fit_encoder
from simula.train import calibrate, fit, predict


@pytest.fixture(scope="module")
def fitted_smoke():
    df = add_flags(load("fixtures"))
    train = split(df, "dev_fit")
    valid = split(df, "select")
    encoder = fit_encoder(train, FEATURE_SETS["C"])
    encoder["feature_set"] = "C"
    booster, best_iteration = fit(train, valid, FEATURE_SETS["C"], encoder)
    meta = {
        "best_iteration": best_iteration,
        "calibration": {"recipe": "test", "days": 1, "shrink": 1.0, "delta": 0.0},
    }
    return df, (booster, encoder, meta)


def test_daily_table_contract(fitted_smoke):
    df, bundle = fitted_smoke
    table = daily_table(df, bundle)
    assert table["day"].tolist() == sorted(df["day"].unique())
    assert table["rows"].sum() == len(df)
    assert np.allclose(table["ctr"], table["clicks"] / table["rows"])
    assert table.loc[table["partial"], "day"].tolist() == [141030]
    assert table.loc[table["day"].le(141027), "unseen_c14_share"].eq(0).all()


def test_hour_matched_uses_only_hours_zero_through_five(fitted_smoke):
    df, _ = fitted_smoke
    table = hour_matched(df)
    early = df["hour_of_day"].between(0, 5)
    expected_pre = (early & df["day"].between(141021, 141028)).sum()
    expected_test = (early & df["day"].between(141029, 141030)).sum()
    assert table.loc["Oct21-28 h00-05", "rows"] == expected_pre
    assert table.loc["Oct29-30 h00-05", "rows"] == expected_test


def test_psi_fixed_other_bucket():
    reference = np.array(["a", "a", "b"])
    same = np.array(["a", "a", "b"])
    unseen = np.array(["a", "new", "new"])
    import pandas as pd
    reference = pd.DataFrame({"value": reference})
    assert psi(reference, pd.DataFrame({"value": same}), "value") == pytest.approx(0)
    shifted = psi(reference, pd.DataFrame({"value": unseen}), "value")
    assert np.isfinite(shifted) and shifted > 0


def test_adapt_direction_and_shrink(fitted_smoke):
    df, bundle = fitted_smoke
    full = adapt(df, bundle, 1.0)
    day29 = df.loc[df["day"].eq(141029)]
    p29 = predict(bundle, day29, calibrated=True)
    assert full["delta"] == pytest.approx(calibrate(p29, day29["click"], 1.0))
    frozen = full["frozen"][141030]["mean_pred"]
    adapted = full["adapted"][141030]["mean_pred"]
    target = full["frozen"][141029]["mean_actual"]
    assert np.sign(adapted - frozen) == np.sign(target - full["frozen"][141029]["mean_pred"])
    zero = adapt(df, bundle, 0.0)
    assert zero["adapted"][141030] == pytest.approx(zero["frozen"][141030])


def test_adapt_delta_does_not_read_day_30_labels(fitted_smoke):
    df, bundle = fitted_smoke
    before = adapt(df, bundle, 1.0)["delta"]
    flipped = df.copy()
    day30 = flipped["day"].eq(141030)
    flipped.loc[day30, "click"] = 1 - flipped.loc[day30, "click"]
    assert adapt(flipped, bundle, 1.0)["delta"] == before
