"""Shared machinery for the batch-X training-loop experiments (x12-x15): seeded, subsampled B-recipe fits,
calibration, per-row losses and the Jev round trip. Every fit uses B's params with its own seed; set
OMP_NUM_THREADS=3 before running so LightGBM uses three threads.
"""
import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np

from simula.data import add_flags, load
from simula.features import fit_encoder, transform
from simula.jev_ctr import _row_loss, bootstrap_diff  # noqa: F401 - re-exported for the experiment modules
from simula.rank_pairs import read_answers
from simula.train import PARAMS, apply_calibration, calibrate

DATA = Path.home() / "Desktop/Simula/ml-takehome/data"
WORK = Path.home() / "simula-local/rank-jev-work/exp-x-training-loop"
CAP = 300_000
SHRINK = 0.5  # B's calibration recipe (one_day_half), from bundle/B/meta.json
JEV_CMD = ["uv", "run", "--no-project", "--with", "typesafe-sdk==0.7.2", "--with", "python-dotenv", "python",
           "jev_ask.py"]


def frame(data_dir=DATA):
    """The joined impressions with flags."""
    return add_flags(load(data_dir))


def subsample(df, seed, cap=CAP):
    """At most `cap` rows, drawn without replacement with `seed`."""
    return df if len(df) <= cap else df.sample(cap, random_state=seed)


def fit(train, features, rounds, seed, valid=None):
    """Fit B's recipe for a fixed number of rounds with its own seed.

    Returns (booster, encoder, curve). With `valid`, curve holds the per-tree train and valid log loss.
    """
    encoder = fit_encoder(train, features)
    categorical = list(encoder["categories"])
    data = lgb.Dataset(transform(train, encoder), label=train["click"], categorical_feature=categorical)
    sets, names, curve = [], [], {}
    if valid is not None:
        sets = [data, lgb.Dataset(transform(valid, encoder), label=valid["click"], categorical_feature=categorical,
                                  reference=data)]
        names = ["train", "valid"]
    booster = lgb.train({**PARAMS, "seed": seed, "metric": "binary_logloss"}, data, num_boost_round=int(rounds),
                        valid_sets=sets, valid_names=names, callbacks=[lgb.record_evaluation(curve)] if sets else [])
    return booster, encoder, {k: np.asarray(v["binary_logloss"]) for k, v in curve.items()}


def raw(model, df, trees=None):
    """Uncalibrated predictions of (booster, encoder), optionally truncated to the first `trees` trees."""
    booster, encoder = model
    return booster.predict(transform(df, encoder), num_iteration=trees)


def calibrated(model, cal, df, trees=None, shrink=SHRINK):
    """Predictions on `df` after B's shrunk intercept shift fitted on `cal`."""
    return apply_calibration(raw(model, df, trees), calibrate(raw(model, cal, trees), cal["click"], shrink))


def row_loss(y, p):
    """Per-row log loss."""
    return _row_loss(np.asarray(y, dtype=float), np.asarray(p, dtype=float))


def ask_jev(items, name):
    """Write Jev input items to WORK/<name>.in.jsonl, run jev_ask.py (resumable) and return answers by id."""
    WORK.mkdir(parents=True, exist_ok=True)
    src, dst = WORK / f"{name}.in.jsonl", WORK / f"{name}.out.jsonl"
    src.write_text("".join(json.dumps(item) + "\n" for item in items))
    subprocess.run([*JEV_CMD, str(src), str(dst)], check=True)
    return read_answers(dst)


def jev_cost(answers):
    """Calls, attempts, models, input tokens, $ and seconds for a set of answer lines."""
    rows = list(answers.values())
    tokens = int(sum(r["tokens_in"] for r in rows))
    return {"calls": len(rows), "attempts": int(sum(r["attempts"] for r in rows)),
            "models": sorted({r["model"] for r in rows}), "input_tokens": tokens, "usd": tokens * 0.042e-6,
            "call_seconds_sum": float(sum(r["seconds"] for r in rows))}
