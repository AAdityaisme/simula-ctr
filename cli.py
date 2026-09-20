import argparse
import json
from pathlib import Path

import pandas as pd

from simula.baselines import constant, lookup, predict
from simula.data import WINDOWS, add_flags, load, split
from simula.evaluate import slices


BASELINES = (
    ("constant", lambda train: constant(train)),
    ("lookup_C18", lambda train: lookup(train, ["C18"])),
    ("lookup_app_id_C18", lambda train: lookup(train, ["app_id", "C18"])),
    ("lookup_hour_of_day_C18", lambda train: lookup(train, ["hour_of_day", "C18"])),
)


def _markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def evaluate_baselines(data_dir, smoke=False):
    """Fit refit-window baselines, evaluate the test window, and write reports."""
    source = Path("fixtures") if smoke else Path(data_dir)
    df = add_flags(load(source))
    counts = {name: len(split(df, name)) for name in WINDOWS}
    print("window counts:", " ".join(f"{name}={count}" for name, count in counts.items()))

    train = split(df, "refit")
    test = split(df, "test")
    results = []
    for name, fit in BASELINES:
        fitted = fit(train)
        table = slices(df, predict(fitted, test))
        table.insert(0, "baseline", name)
        results.append(table)
        print(f"\n{name}")
        print(table.drop(columns="baseline").to_string(index=False))

    combined = pd.concat(results, ignore_index=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    reports_dir.joinpath("baselines.md").write_text(
        "# Baseline evaluation\n\n" + _markdown_table(combined) + "\n",
        encoding="utf-8",
    )
    records = combined.where(pd.notna(combined), None).to_dict(orient="records")
    reports_dir.joinpath("baselines.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Simula CTR utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate held-out predictions")
    evaluate.add_argument("--baseline", action="store_true", required=True)
    evaluate.add_argument("--data-dir", default="data")
    evaluate.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate_baselines(args.data_dir, args.smoke)


if __name__ == "__main__":
    main()
