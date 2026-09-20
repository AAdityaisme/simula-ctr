import numpy as np
import pandas as pd

from simula.baselines import constant, lookup, predict
from simula.data import add_flags, load, split


def smoke_data():
    return add_flags(load("fixtures"))


def test_join_has_no_unmatched_characters():
    df = smoke_data()
    assert df["character_name"].notna().all()


def test_windows_partition_smoke_rows():
    df = smoke_data()
    dev_fit = split(df, "dev_fit")
    select = split(df, "select")
    refit = split(df, "refit")
    calibrate = split(df, "calibrate")
    test = split(df, "test")

    expected = {
        "dev_fit": 3000,
        "select": 500,
        "refit": 3500,
        "calibrate": 500,
        "test": 1000,
    }
    actual = {
        "dev_fit": len(dev_fit),
        "select": len(select),
        "refit": len(refit),
        "calibrate": len(calibrate),
        "test": len(test),
    }
    assert actual == expected
    assert set(dev_fit.index).isdisjoint(select.index)
    assert set(dev_fit.index) | set(select.index) == set(refit.index)
    assert set(refit.index).isdisjoint(calibrate.index)
    assert set(refit.index).isdisjoint(test.index)
    assert set(calibrate.index).isdisjoint(test.index)
    assert set(refit.index) | set(calibrate.index) | set(test.index) == set(df.index)


def test_site_side_is_exact_placeholder_complement():
    df = smoke_data()
    assert (df["is_site_side"] ^ df["site_id"].eq("85f751fd")).all()


def test_unseen_lookup_uses_finite_prior():
    train = split(smoke_data(), "refit")
    fitted = lookup(train, ["C18"])
    unseen = pd.DataFrame({"C18": ["never-seen"]})
    prediction = predict(fitted, unseen)
    assert np.isfinite(prediction).all()
    assert prediction[0] == fitted["prior"]


def test_all_baseline_predictions_are_probabilities():
    df = smoke_data()
    train = split(df, "refit")
    test = split(df, "test")
    fitted = [
        constant(train),
        lookup(train, ["C18"]),
        lookup(train, ["app_id", "C18"]),
        lookup(train, ["hour_of_day", "C18"]),
    ]
    for baseline in fitted:
        probabilities = predict(baseline, test)
        assert ((probabilities > 0) & (probabilities < 1)).all()
