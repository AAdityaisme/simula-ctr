"""Shared machinery for batch X worker B (x03 feature triage, x04 leak auditor, x05 choose the exam).

Jev reads definitions and process descriptions here, never rows. This module holds the time-window counting used to
build features and recipe cards, and the B fitting recipe from `simula/train.py` on subsampled windows.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from simula.data import split
from simula.features import CONTRACT, FEATURE_SETS, fit_encoder, transform
from simula.jev_ctr import _row_loss
from simula.train import PARAMS, apply_calibration, calibrate, fit, refit

DATA = Path.home() / "Desktop/Simula/ml-takehome/data"
WORK = Path.home() / "simula-local/rank-jev-work/exp-x-judge"
N_SUB = 300_000
SHRINK = 0.5  # bundle B's calibration recipe (one_day_half), fixed rather than re-chosen per arm


def hour_index(hour):
    """Hours since 2014-10-21 00:00 for YYMMDDHH integers."""
    hour = np.asarray(hour)
    return (hour // 100 % 100 - 21) * 24 + hour % 100


def range_sum(df, keys, lo, hi, value=None):
    """Per row: the sum of `value` (or the row count) over rows with the same keys whose hour index lies in [lo, hi].

    `lo` and `hi` are per-row hour indices (arrays or scalars, inclusive); hi < lo gives 0. Only rows inside `df`
    are counted, so the caller decides which window may be seen.
    """
    h = hour_index(df["hour"])
    table = df[keys].assign(_h=h, _v=1.0 if value is None else df[value].astype(float))
    table = table.groupby(keys + ["_h"], sort=False)["_v"].sum().reset_index().sort_values("_h")
    table["_c"] = table.groupby(keys, sort=False)["_v"].cumsum()
    n = len(df)
    lo = np.broadcast_to(np.asarray(lo, dtype=float), n)
    hi = np.broadcast_to(np.asarray(hi, dtype=float), n)

    def cum_at(x):
        query = df[keys].reset_index(drop=True).assign(_h=np.floor(x), _row=np.arange(n))
        query = query.replace({"_h": {np.inf: 1e9, -np.inf: -1e9}}).sort_values("_h")
        found = pd.merge_asof(query, table[keys + ["_h", "_c"]].astype({"_h": float}), on="_h", by=keys,
                              direction="backward")
        return found.sort_values("_row")["_c"].fillna(0).to_numpy()

    return np.where(hi >= lo, cum_at(hi) - cum_at(lo - 1), 0.0)


def day_bounds(df):
    """First and last hour index of each row's calendar day."""
    start = (np.asarray(df["hour"]) // 100 % 100 - 21) * 24
    return start, start + 23


def extend_numeric(columns):
    """Treat new columns as numeric model features (as `simula.jev_ctr` does), without editing features.py."""
    CONTRACT["numeric"] = CONTRACT["numeric"] + [c for c in columns if c not in CONTRACT["numeric"]]


def subsample(df, n=N_SUB, seed=0):
    """At most n rows, the same rows for every arm that shares the seed."""
    return df if len(df) <= n else df.sample(n, random_state=seed).sort_index()


def windows(df):
    """The train.py windows, with the two fitting windows subsampled to N_SUB rows."""
    return {"dev_fit": subsample(split(df, "dev_fit")), "select": split(df, "select"),
            "refit": subsample(split(df, "refit")), "calibrate": split(df, "calibrate"), "test": split(df, "test")}


def b_recipe(w, extra=(), seed=None):
    """train.py's recipe for B + extra: early stopping on Oct 27, refit through Oct 27, calibrate on Oct 28.

    Returns tree count, Oct 27 log loss of the development fit, calibrated test probabilities and the refit's split
    count for each extra feature. `seed` overrides PARAMS["seed"] for this call only.
    """
    if seed is not None:
        # ponytail: train.fit/refit read the module-level PARAMS, so the seed is swapped in place and restored
        old, PARAMS["seed"] = PARAMS["seed"], seed
        try:
            return b_recipe(w, extra)
        finally:
            PARAMS["seed"] = old
    features = FEATURE_SETS["B"] + list(extra)
    encoder = fit_encoder(w["dev_fit"], features)
    dev, rounds = fit(w["dev_fit"], w["select"], features, encoder)
    select_loss = _row_loss(w["select"]["click"].to_numpy(), dev.predict(transform(w["select"], encoder))).mean()
    encoder = fit_encoder(w["refit"], features)
    booster = refit(w["refit"], features, encoder, rounds)
    cal = w["calibrate"]
    delta = calibrate(booster.predict(transform(cal, encoder)), cal["click"], SHRINK)
    p = apply_calibration(booster.predict(transform(w["test"], encoder)), delta)
    splits = dict(zip(booster.feature_name(), booster.feature_importance("split").tolist()))
    return {"trees": int(rounds), "select_log_loss": float(select_loss), "p": p,
            "splits": {f: splits[f] for f in extra}}
