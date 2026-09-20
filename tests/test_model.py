import numpy as np
import pytest

from simula.data import add_flags, load, split
from simula.features import CONTRACT, FEATURE_SETS, fit_encoder, transform
from simula.train import (
    _save_bundle,
    apply_calibration,
    calibrate,
    fit,
    load_bundle,
    predict,
)


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
    return df, booster, encoder, meta


def unseen_row(df):
    row = split(df, "test").iloc[[0]].copy()
    for feature in ("C14", "character_id", "site_id", "app_id"):
        row[feature] = f"never-seen-{feature}"
    return row


def test_unseen_categories_produce_finite_probability(fitted_smoke):
    df, booster, encoder, meta = fitted_smoke
    probability = predict((booster, encoder, meta), unseen_row(df), calibrated=False)
    assert np.isfinite(probability).all()
    assert ((probability > 0) & (probability < 1)).all()


def test_bundle_round_trip_is_exact(fitted_smoke, tmp_path):
    df, booster, encoder, meta = fitted_smoke
    rows = split(df, "test").head(25)
    before = predict((booster, encoder, meta), rows)
    _save_bundle(tmp_path, booster, encoder, meta)
    after = predict(load_bundle(tmp_path), rows)
    assert np.array_equal(before, after)


def test_feature_contract_is_safe_and_present():
    df = add_flags(load("fixtures"))
    for features in FEATURE_SETS.values():
        assert not set(features) & set(CONTRACT["never"])
        assert set(features) <= set(df.columns)


def test_full_calibration_matches_observed_mean():
    df = add_flags(load("fixtures"))
    calibration = split(df, "calibrate")
    y = calibration["click"].to_numpy()
    p = np.full(len(y), split(df, "refit")["click"].mean())
    delta = calibrate(p, y, shrink=1.0)
    assert np.mean(apply_calibration(p, delta)) == pytest.approx(np.mean(y), abs=1e-3)


def test_transform_uses_encoder_order_regardless_of_input_order():
    train = split(add_flags(load("fixtures")), "dev_fit")
    encoder = fit_encoder(train, FEATURE_SETS["B"])
    shuffled = train[list(reversed(train.columns))]
    assert list(transform(shuffled, encoder).columns) == encoder["features"]
