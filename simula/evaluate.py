import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from simula.data import split


def metrics(y, p):
    """Compute discrimination, probability, and volume metrics."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p):
        raise ValueError("labels and predictions must have the same length")
    if not np.isfinite(p).all():
        raise ValueError("predictions must be finite")
    auc = float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))
    return {
        "roc_auc": auc,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "n": int(len(y)),
        "clicks": int(y.sum()),
        "mean_pred": float(p.mean()),
        "mean_actual": float(y.mean()),
    }


def slices(df, p):
    """Evaluate test predictions by day and refit-window novelty."""
    test = split(df, "test")
    p = np.asarray(p, dtype=float)
    if len(test) != len(p):
        raise ValueError("predictions must align with the test window")

    refit = split(df, "refit")
    c14_seen = test["C14"].isin(refit["C14"])
    character_seen = test["character_id"].isin(refit["character_id"])
    masks = {
        "all_test": np.ones(len(test), dtype=bool),
        "day_141029": test["day"].eq(141029).to_numpy(),
        "day_141030": test["day"].eq(141030).to_numpy(),
        "C14_seen": c14_seen.to_numpy(),
        "C14_unseen": (~c14_seen).to_numpy(),
        "character_id_seen": character_seen.to_numpy(),
        "character_id_unseen": (~character_seen).to_numpy(),
    }

    rows = []
    y = test["click"].to_numpy()
    for name, mask in masks.items():
        row = {"slice": name}
        row.update(metrics(y[mask], p[mask]))
        rows.append(row)
    return pd.DataFrame(rows)
