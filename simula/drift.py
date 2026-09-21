from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from simula.data import split
from simula.evaluate import metrics
from simula.train import apply_calibration, calibrate, predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ADAPT = dict(shrink=0.5, label_available="next day 00:00")
PSI_COLUMNS = ["app_category", "site_category", "device_type", "C20"]


def daily_table(df, bundle):
    """Summarize daily mix, calibrated predictions, and error."""
    refit_c14 = set(split(df, "refit")["C14"])
    probabilities = predict(bundle, df, calibrated=True)
    rows = []
    for day, positions in df.groupby("day", sort=True).indices.items():
        daily = df.iloc[positions]
        summary = metrics(daily["click"], probabilities[positions])
        rows.append({
            "day": int(day),
            "rows": int(len(daily)),
            "clicks": int(daily["click"].sum()),
            "ctr": float(daily["click"].mean()),
            "site_side_share": float(daily["is_site_side"].mean()),
            "unseen_c14_share": float((~daily["C14"].isin(refit_c14)).mean()),
            "c20_sentinel_share": float(daily["c20_is_sentinel"].mean()),
            "mean_pred": summary["mean_pred"],
            "log_loss": summary["log_loss"],
            "partial": bool(day == 141030),
        })
    return pd.DataFrame(rows)


def hour_matched(df):
    """Compare pre-test and test CTR with and without matched hours."""
    rows = []
    for label, days in (("Oct21-28", range(141021, 141029)), ("Oct29-30", range(141029, 141031))):
        period = df.loc[df["day"].isin(days)]
        for suffix, sample in (("h00-05", period.loc[period["hour_of_day"].between(0, 5)]),
                               ("all", period)):
            rows.append({
                "period": f"{label} {suffix}",
                "rows": int(len(sample)),
                "clicks": int(sample["click"].sum()),
                "ctr": float(sample["click"].mean()),
            })
    order = ["Oct21-28 h00-05", "Oct29-30 h00-05", "Oct21-28 all", "Oct29-30 all"]
    return pd.DataFrame(rows).set_index("period").loc[order]


def psi(reference, current, column, bins=None):
    """Compute PSI using reference-fixed buckets and an unseen bucket."""
    def bucket(series):
        values = pd.cut(series, bins=bins, include_lowest=True) if bins is not None else series
        values = values.astype("string")
        return ("value:" + values).fillna("missing:")

    reference_values = bucket(reference[column])
    current_values = bucket(current[column])
    categories = reference_values.unique().tolist()
    current_values = current_values.where(current_values.isin(categories), "other:")
    total = 0.0
    for category in categories + ["other:"]:
        reference_share = float(reference_values.eq(category).mean())
        current_share = float(current_values.eq(category).mean())
        total += (current_share - reference_share) * np.log(
            (current_share + 1e-6) / (reference_share + 1e-6)
        )
    return float(total)


def psi_table(df):
    """Compare each test day's monitored columns with the refit window."""
    reference = split(df, "refit")
    return pd.DataFrame([
        {"column": column, "day": day,
         "psi": psi(reference, df.loc[df["day"].eq(day)], column)}
        for column in PSI_COLUMNS for day in (141029, 141030)
    ])


def adapt(df, bundle, shrink):
    """Predict each test day, then update the intercept from day 141029."""
    day29 = df.loc[df["day"].eq(141029)]
    day30 = df.loc[df["day"].eq(141030)]
    p29 = predict(bundle, day29, calibrated=True)
    p30 = predict(bundle, day30, calibrated=True)
    delta = calibrate(p29, day29["click"], shrink)
    p30_adapted = apply_calibration(p30, delta)

    def result(rows, probabilities):
        values = metrics(rows["click"], probabilities)
        return {key: values[key] for key in ("log_loss", "mean_pred", "mean_actual")} | {
            "rows": values["n"]
        }

    return {
        "delta": float(delta),
        "shrink": float(shrink),
        "frozen": {141029: result(day29, p29), 141030: result(day30, p30)},
        "adapted": {141030: result(day30, p30_adapted)},
    }


def plot_drift(daily, path):
    """Plot daily CTR and monitored composition shares."""
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(daily["day"], daily["ctr"], marker="o", label="CTR")
    partial = daily["partial"]
    top.plot(daily.loc[partial, "day"], daily.loc[partial, "ctr"], linestyle="none",
             marker="o", markerfacecolor="white", markeredgecolor="C0", markersize=7)
    top.set_ylabel("CTR")
    top.grid(alpha=0.2)
    for column, label in (("site_side_share", "site side"),
                          ("unseen_c14_share", "unseen C14"),
                          ("c20_sentinel_share", "C20=-1")):
        bottom.plot(daily["day"], daily[column], marker="o", label=label)
    bottom.set(xlabel="Day", ylabel="Share")
    bottom.ticklabel_format(axis="x", style="plain", useOffset=False)
    bottom.legend()
    bottom.grid(alpha=0.2)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
