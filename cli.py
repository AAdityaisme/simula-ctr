import argparse
import json
from pathlib import Path

import pandas as pd

from simula.baselines import constant, lookup, predict as baseline_predict
from simula.data import WINDOWS, add_flags, load, split
from simula.evaluate import plot_reliability, rare_table, reliability, slices
from simula.features import FEATURE_SETS
from simula.train import load_bundle, predict as model_predict, train_all


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
        table = slices(df, baseline_predict(fitted, test))
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
    records = combined.astype(object).where(pd.notna(combined), None).to_dict(orient="records")
    reports_dir.joinpath("baselines.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train_models(data_dir, out_dir, smoke=False):
    """Train all model feature sets and write a compact training report."""
    source = Path("fixtures") if smoke else Path(data_dir)
    result = train_all(add_flags(load(source)), out_dir)
    rows = []
    for name, model in result["models"].items():
        row = {
            "feature_set": name,
            "select_log_loss": model["select_log_loss"],
            "best_iteration": model["best_iteration"],
            "calibration_recipe": result["calibration"]["recipe"],
            "calibration_delta": model["calibration_delta"],
        }
        rows.append(row)
        print(
            f"{name}: select_log_loss={model['select_log_loss']:.6f} "
            f"best_iteration={model['best_iteration']} "
            f"calibration={result['calibration']['recipe']}"
        )
    print(f"selected: {result['selected']['feature_set']}")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    reports_dir.joinpath("train.md").write_text(
        "# Model training\n\n" + _markdown_table(pd.DataFrame(rows)) + "\n",
        encoding="utf-8",
    )


def _model_table(df, raw, calibrated, feature_set):
    raw_table = slices(df, raw).add_prefix("raw_").rename(columns={"raw_slice": "slice"})
    calibrated_table = slices(df, calibrated).add_prefix("calibrated_").rename(
        columns={"calibrated_slice": "slice"}
    )
    table = raw_table.merge(calibrated_table, on="slice")
    table.insert(0, "feature_set", feature_set)
    return table


def evaluate_models(data_dir, model_dir, smoke=False):
    """Evaluate raw and calibrated model bundles and write model reports."""
    source = Path("fixtures") if smoke else Path(data_dir)
    df = add_flags(load(source))
    test = split(df, "test")
    model_dir = Path(model_dir)
    selected = json.loads(model_dir.joinpath("selected.json").read_text())["feature_set"]
    results = []
    selected_predictions = None
    for name in FEATURE_SETS:
        bundle = load_bundle(model_dir / name)
        raw = model_predict(bundle, test, calibrated=False)
        calibrated = model_predict(bundle, test, calibrated=True)
        table = _model_table(df, raw, calibrated, name)
        results.append(table)
        shown = table.loc[
            table["slice"].isin(["all_test", "day_141029", "day_141030", "C14_seen", "C14_unseen"])
        ]
        print(f"\nfeature set {name}")
        print(shown.to_string(index=False))
        if name == selected:
            selected_predictions = (raw, calibrated)

    combined = pd.concat(results, ignore_index=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    reports_dir.joinpath("model_eval.md").write_text(
        "# Model evaluation\n\n" + _markdown_table(combined) + "\n",
        encoding="utf-8",
    )
    records = combined.astype(object).where(pd.notna(combined), None).to_dict(orient="records")
    reports_dir.joinpath("model_eval.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    raw, calibrated = selected_predictions
    plot_reliability(
        {
            "raw": reliability(test["click"], raw),
            "calibrated": reliability(test["click"], calibrated),
        },
        reports_dir / "reliability.png",
    )
    rarity = rare_table(split(df, "refit"), test, FEATURE_SETS["C"])
    reports_dir.joinpath("rare_table.md").write_text(
        "# Rare categorical values\n\n" + _markdown_table(rarity) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Simula CTR utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train LightGBM model bundles")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--out", default="bundle")
    train.add_argument("--smoke", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="evaluate held-out predictions")
    mode = evaluate.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--model")
    evaluate.add_argument("--data-dir", default="data")
    evaluate.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.command == "train":
        train_models(args.data_dir, args.out, args.smoke)
    elif args.baseline:
        evaluate_baselines(args.data_dir, args.smoke)
    else:
        evaluate_models(args.data_dir, args.model, args.smoke)


if __name__ == "__main__":
    main()
