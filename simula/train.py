import json
import platform
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss

from simula.data import WINDOWS, split
from simula.features import FEATURE_SETS, fit_encoder, transform


PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_per_group=100,
    cat_smooth=10,
    verbose=-1,
    seed=20260920,
)


def fit(train, valid, features, encoder):
    """Fit with early stopping and return the selected tree count."""
    categorical = list(encoder["categories"])
    training = lgb.Dataset(
        transform(train, encoder), label=train["click"], categorical_feature=categorical
    )
    validation = lgb.Dataset(
        transform(valid, encoder), label=valid["click"], categorical_feature=categorical
    )
    booster = lgb.train(
        PARAMS,
        training,
        num_boost_round=2000,
        valid_sets=[validation],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    booster._simula_encoder = encoder
    return booster, booster.best_iteration


def refit(train, features, encoder, n_rounds):
    """Refit a model for a fixed number of rounds."""
    dataset = lgb.Dataset(
        transform(train, encoder),
        label=train["click"],
        categorical_feature=list(encoder["categories"]),
    )
    booster = lgb.train(PARAMS, dataset, num_boost_round=int(n_rounds))
    booster._simula_encoder = encoder
    return booster


def calibrate(p_cal, y_cal, shrink):
    """Fit a shrunk intercept shift on prediction logits.

    Solves for the shift that makes the mean calibrated probability equal the
    observed click rate. Averaging logits instead would bias the shift upward,
    because logit is concave below 0.5.
    """
    epsilon = np.finfo(float).eps
    logits = _logit(np.clip(np.asarray(p_cal, dtype=float), epsilon, 1 - epsilon))
    target = float(np.mean(y_cal))
    low, high = -10.0, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        if _sigmoid(logits + mid).mean() < target:
            low = mid
        else:
            high = mid
    return float(shrink * mid)


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def apply_calibration(p, delta):
    """Apply an intercept shift and return calibrated probabilities."""
    epsilon = np.finfo(float).eps
    p = np.clip(np.asarray(p, dtype=float), epsilon, 1 - epsilon)
    logits = np.log(p / (1 - p)) + delta
    return 1 / (1 + np.exp(-logits))


def choose_calibration(dev_booster, df):
    """Choose the calibration window and shrinkage using October 27 only."""
    encoder = dev_booster._simula_encoder
    select = split(df, "select")
    recipes = {
        "one_day_full": (df["day"].eq(141026), 1, 1.0),
        "one_day_half": (df["day"].eq(141026), 1, 0.5),
        "two_day_full": (df["day"].isin([141025, 141026]), 2, 1.0),
    }
    select_raw = dev_booster.predict(transform(select, encoder))
    losses = {}
    for name, (mask, _, shrink) in recipes.items():
        calibration = df.loc[mask]
        p_cal = dev_booster.predict(transform(calibration, encoder))
        delta = calibrate(p_cal, calibration["click"], shrink)
        losses[name] = float(log_loss(select["click"], apply_calibration(select_raw, delta)))
    winner = min(losses, key=losses.get)
    _, days, shrink = recipes[winner]
    return {"recipe": winner, "days": days, "shrink": shrink, "dev_log_loss": losses}


def _save_bundle(path, booster, encoder, meta):
    path.mkdir(parents=True, exist_ok=True)
    booster.save_model(path / "model.txt")
    path.joinpath("encoder.json").write_text(json.dumps(encoder, indent=2) + "\n")
    path.joinpath("meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_bundle(directory):
    """Load a fitted model, encoder, and metadata bundle."""
    directory = Path(directory)
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    encoder = json.loads(directory.joinpath("encoder.json").read_text())
    meta = json.loads(directory.joinpath("meta.json").read_text())
    return booster, encoder, meta


def predict(bundle, df, calibrated=True):
    """Predict from a loaded bundle, optionally applying its calibration."""
    if isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle)
    booster, encoder, meta = bundle
    probabilities = booster.predict(transform(df, encoder))
    if calibrated:
        probabilities = apply_calibration(probabilities, meta["calibration"]["delta"])
    return np.asarray(probabilities, dtype=float)


def _metadata(df, best_iteration, choice, delta, select_loss):
    libraries = {
        name: version(name) for name in ("lightgbm", "pandas", "scikit-learn")
    }
    libraries["python"] = platform.python_version()
    return {
        "params": PARAMS,
        "best_iteration": int(best_iteration),
        "select_log_loss": float(select_loss),
        "calibration": {**choice, "delta": float(delta)},
        "window_cutoffs": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "versions": libraries,
        "data_row_counts": {"all": len(df), **{name: len(split(df, name)) for name in WINDOWS}},
    }


def train_all(df, out_dir):
    """Train all feature sets, save round-tripped bundles, and select a model."""
    out_dir = Path(out_dir)
    dev_fit, select, refit_rows, test = (split(df, name) for name in ("dev_fit", "select", "refit", "test"))
    development = {}
    for name, features in FEATURE_SETS.items():
        encoder = fit_encoder(dev_fit, features)
        encoder["feature_set"] = name
        booster, best_iteration = fit(dev_fit, select, features, encoder)
        select_raw = booster.predict(transform(select, encoder))
        development[name] = {
            "best_iteration": best_iteration,
            "select_log_loss": float(log_loss(select["click"], select_raw)),
            "booster": booster,
        }

    choice = choose_calibration(development["B"]["booster"], df)
    calibration = df.loc[df["day"].isin([141028] if choice["days"] == 1 else [141027, 141028])]
    results = {}
    for name, features in FEATURE_SETS.items():
        selected_rounds = development[name]["best_iteration"]
        if len(df) <= 5_000:
            selected_rounds = max(20, selected_rounds)
        encoder = fit_encoder(refit_rows, features)
        encoder["feature_set"] = name
        booster = refit(refit_rows, features, encoder, selected_rounds)
        p_cal = booster.predict(transform(calibration, encoder))
        delta = calibrate(p_cal, calibration["click"], choice["shrink"])
        raw = booster.predict(transform(test, encoder))
        calibrated = apply_calibration(raw, delta)
        meta = _metadata(df, selected_rounds, choice, delta, development[name]["select_log_loss"])
        path = out_dir / name
        _save_bundle(path, booster, encoder, meta)
        loaded = load_bundle(path)
        if not np.array_equal(raw, predict(loaded, test, calibrated=False)):
            raise AssertionError(f"{name} raw predictions changed after save/load")
        if not np.array_equal(calibrated, predict(loaded, test, calibrated=True)):
            raise AssertionError(f"{name} calibrated predictions changed after save/load")
        results[name] = {
            "select_log_loss": development[name]["select_log_loss"],
            "best_iteration": int(selected_rounds),
            "calibration_delta": float(delta),
        }

    selected = min(results, key=lambda name: results[name]["select_log_loss"])
    selection = {"feature_set": selected, "select_log_loss": results[selected]["select_log_loss"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("selected.json").write_text(json.dumps(selection, indent=2) + "\n")
    return {"models": results, "selected": selection, "calibration": choice}
