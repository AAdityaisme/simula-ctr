from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from simula.data import split
from simula.features import CONTRACT

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metrics(y, p):
    """Compute discrimination, probability, and volume metrics."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p):
        raise ValueError("labels and predictions must have the same length")
    if not np.isfinite(p).all():
        raise ValueError("predictions must be finite")
    if not len(y):
        return {
            "roc_auc": float("nan"),
            "log_loss": float("nan"),
            "n": 0,
            "clicks": 0,
            "mean_pred": float("nan"),
            "mean_actual": float("nan"),
        }
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
    """Evaluate test predictions by time, novelty, app category, and safety."""
    test = split(df, "test")
    p = np.asarray(p, dtype=float)
    if len(test) != len(p):
        raise ValueError("predictions must align with the test window")

    refit = split(df, "refit")
    c14_seen = test["C14"].isin(refit["C14"])
    c17_seen = test["C17"].isin(refit["C17"])
    character_seen = test["character_id"].isin(refit["character_id"])
    c14_frequency = test["C14"].map(refit["C14"].value_counts()).fillna(0)
    character_frequency = test["character_id"].map(
        refit["character_id"].value_counts()
    ).fillna(0)
    masks = {
        "all_test": np.ones(len(test), dtype=bool),
        "day_141029": test["day"].eq(141029).to_numpy(),
        "day_141030": test["day"].eq(141030).to_numpy(),
        "C14_seen": c14_seen.to_numpy(),
        "C14_unseen": (~c14_seen).to_numpy(),
        "C14_unseen_C17_seen": ((~c14_seen) & c17_seen).to_numpy(),
        "C14_unseen_C17_unseen": ((~c14_seen) & (~c17_seen)).to_numpy(),
        "C14_rare": c14_frequency.between(1, 49).to_numpy(),
        "C14_common": c14_frequency.ge(50).to_numpy(),
        "character_id_seen": character_seen.to_numpy(),
        "character_id_unseen": (~character_seen).to_numpy(),
        "character_id_rare": character_frequency.between(1, 49).to_numpy(),
        "character_id_common": character_frequency.ge(50).to_numpy(),
        "in_window_created_characters": test["created_at"].ge("2014-10-21").to_numpy(),
    }
    for value in test["app_category"].value_counts().head(5).index:
        masks[f"app_category_{value}"] = test["app_category"].eq(value).to_numpy()
    for value in sorted(test["safety_tier"].dropna().unique(), key=str):
        masks[f"safety_tier_{value}"] = test["safety_tier"].eq(value).to_numpy()

    rows = []
    y = test["click"].to_numpy()
    for name, mask in masks.items():
        row = {"slice": name}
        row.update(metrics(y[mask], p[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(y, p, bins=10):
    """Summarize calibration in equal-width probability bins."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p):
        raise ValueError("labels and predictions must have the same length")
    if bins < 1:
        raise ValueError("bins must be positive")
    assignments = np.minimum((np.clip(p, 0, 1) * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = assignments == index
        rows.append(
            {
                "bin": index,
                "n": int(mask.sum()),
                "mean_pred": float(p[mask].mean()) if mask.any() else float("nan"),
                "mean_actual": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(tables, path):
    """Plot reliability lines with the number of rows in each populated bin."""
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    for table_index, (name, table) in enumerate(tables.items()):
        populated = table.loc[table["n"].gt(0)]
        line = axis.plot(
            populated["mean_pred"], populated["mean_actual"], marker="o", label=name
        )[0]
        for row in populated.itertuples(index=False):
            right_edge = row.mean_pred > 0.85
            axis.annotate(
                f"n={row.n}",
                (row.mean_pred, row.mean_actual),
                xytext=((-4 if right_edge else 4), (7 if table_index == 0 else -12)),
                textcoords="offset points",
                fontsize=7,
                color=line.get_color(),
                ha="right" if right_edge else "left",
            )
    axis.set(xlabel="Mean predicted CTR", ylabel="Mean actual CTR", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def rare_table(train, test, features):
    """Return the test share below 50 training occurrences by categorical feature."""
    rows = []
    for feature in features:
        if feature in CONTRACT["numeric"]:
            continue
        counts = train[feature].astype("string").value_counts()
        frequency = test[feature].astype("string").map(counts).fillna(0)
        rows.append({"feature": feature, "share": float(frequency.lt(50).mean())})
    return pd.DataFrame(rows).sort_values("share", ascending=False, ignore_index=True)
