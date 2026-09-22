import argparse
import copy
import platform
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simula.baselines import constant, lookup, predict as baseline_predict
from simula.data import WINDOWS, add_flags, load, load_characters, split
from simula.drift import ADAPT, adapt, daily_table, hour_matched, plot_drift, psi_table
from simula.evaluate import plot_reliability, rare_table, reliability, slices
from simula.features import FEATURE_SETS
from simula.rank import rank as rank_candidates
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


def benchmark_rank(payloads, bundle, characters, rounds):
    """Time rank() per request on warm artifacts and write reports/latency.md."""
    import time

    payloads = list(payloads)
    base = next((p for p in payloads if p.get("exposure")), payloads[0])
    for size in (1, 10, 50):
        template = next(c for c in base["candidates"] if c["content_tier"] == "sfw")
        synthetic = {**copy.deepcopy(base), "id": f"synthetic_{size}_candidates"}
        synthetic["candidates"] = [
            {**template, "candidate_id": f"c{i}", "banner_pos": i % 2} for i in range(size)
        ]
        payloads.append(synthetic)
    for payload in payloads:
        rank_candidates(payload, bundle, characters)  # warm-up
    timings = {payload["id"]: [] for payload in payloads}
    for _ in range(rounds):
        for payload in payloads:
            start = time.perf_counter()
            rank_candidates(payload, bundle, characters)
            timings[payload["id"]].append((time.perf_counter() - start) * 1000)
    rows = [
        {
            "request": name, "candidates": len(next(p for p in payloads if p["id"] == name)["candidates"]),
            "rounds": rounds, "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
        }
        for name, values in timings.items()
    ]
    table = pd.DataFrame(rows)
    print(f"\nlatency (in-process rank(), warm bundle, one request at a time, {platform.machine()} python {platform.python_version()})")
    print(table.to_string(index=False))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latency.md").write_text(
        "# Rank latency\n\nIn-process `rank()` on a warm bundle and character table, one request "
        "at a time, no network, no serialization, single core, after one warm-up call. "
        f"Machine: {platform.machine()}, python {platform.python_version()}. "
        "This is the model-plus-policy cost inside a 50 ms budget, not an end-to-end p99.\n\n"
        + _markdown_table(table) + "\n",
        encoding="utf-8",
    )


def rank_requests(requests_path, model_dir, data_dir, out_path, smoke=False, bench=0):
    """Rank fixture requests with the selected bundle and write full decisions."""
    source = Path("fixtures") if smoke else Path(data_dir)
    model_dir = Path(model_dir)
    selected = json.loads(model_dir.joinpath("selected.json").read_text())["feature_set"]
    bundle = load_bundle(model_dir / selected)
    characters = load_characters(source)
    payloads = json.loads(Path(requests_path).read_text())
    if bench:
        benchmark_rank(payloads, bundle, characters, bench)
    results = []
    for payload in payloads:
        result = {"id": payload["id"], **rank_candidates(payload, bundle, characters)}
        results.append(result)
        print(
            f"\n{payload['id']}: selected={result['selected_candidate_id']} "
            f"no_fill={result['no_fill']} indistinguishable={result['indistinguishable']} "
            f"degraded={result['degraded']}"
        )
        table = pd.DataFrame(result["candidates"])[
            [
                "candidate_id", "rank", "eligible", "exclusion_reason", "banner_pos",
                "C14", "calibrated_pctr", "utility", "in_exploration_set",
                "selection_probability", "is_unseen_C14",
            ]
        ].rename(
            columns={
                "in_exploration_set": "in_E",
                "selection_probability": "p_select",
                "is_unseen_C14": "unseen_C14",
            }
        )
        print(table.to_string(index=False))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


def drift_report(data_dir, model_dir, smoke=False):
    """Generate daily drift and predict-then-update adaptation reports."""
    source = Path("fixtures") if smoke else Path(data_dir)
    df = add_flags(load(source))
    model_dir = Path(model_dir)
    selected = json.loads(model_dir.joinpath("selected.json").read_text())["feature_set"]
    bundle = load_bundle(model_dir / selected)
    daily = daily_table(df, bundle)
    matched = hour_matched(df)
    matched_report = matched.rename_axis("period").reset_index()
    stability = psi_table(df)
    adaptation = adapt(df, bundle, ADAPT["shrink"])
    adaptation_rows = []
    for state in ("frozen", "adapted"):
        for day, values in adaptation[state].items():
            adaptation_rows.append({
                "state": state, "day": day, **values,
                "delta": adaptation["delta"], "shrink": adaptation["shrink"],
            })
    adaptation_table = pd.DataFrame(adaptation_rows)

    print("\ndaily")
    print(daily.to_string(index=False))
    print("\nhour matched")
    print(matched_report.to_string(index=False))
    print("\nPSI")
    print(stability.to_string(index=False))
    print(f"\nadaptation: delta={adaptation['delta']:.6f} shrink={adaptation['shrink']:.6f}")
    print(adaptation_table.to_string(index=False))

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    header = (
        "# Drift monitoring and intercept adaptation\n\n"
        "Oct 21–27 predictions are model-in-sample and Oct 28 is calibration-in-sample; "
        "they show error moving with the mix and are not held-out metrics. Labels for day d are assumed available at "
        "00:00 on day d+1 because the data has no availability timestamps.\n\n"
        "Unseen-C14 share is zero by construction during the Oct 21–27 refit window. "
        "The prototype makes one update on the partial Oct 30 day. Its monotonic intercept "
        "shift changes probability levels, never candidate order.\n\n"
    )
    sections = (
        ("Daily", daily), ("Hour-matched CTR", matched_report),
        ("Population stability index", stability), ("Adaptation", adaptation_table),
    )
    body = "\n\n".join(f"## {name}\n\n{_markdown_table(table)}" for name, table in sections)
    reports_dir.joinpath("drift.md").write_text(header + body + "\n", encoding="utf-8")
    records = {
        "daily": daily.to_dict(orient="records"),
        "hour_matched": matched_report.to_dict(orient="records"),
        "psi": stability.to_dict(orient="records"),
        "adapt": adaptation,
    }
    reports_dir.joinpath("drift.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_drift(daily, reports_dir / "drift.png")


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
    rank_parser = subparsers.add_parser("rank", help="rank candidate sets")
    rank_parser.add_argument("--requests", required=True)
    rank_parser.add_argument("--model", required=True)
    rank_parser.add_argument("--data-dir", default="data")
    rank_parser.add_argument("--smoke", action="store_true")
    rank_parser.add_argument("--out", default="reports/rank_sample.json")
    rank_parser.add_argument("--bench", type=int, default=0, help="time rank() this many rounds")
    drift_parser = subparsers.add_parser("drift", help="report drift and daily adaptation")
    drift_parser.add_argument("--model", required=True)
    drift_parser.add_argument("--data-dir", default="data")
    drift_parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.command == "train":
        train_models(args.data_dir, args.out, args.smoke)
    elif args.command == "rank":
        rank_requests(args.requests, args.model, args.data_dir, args.out, args.smoke, args.bench)
    elif args.command == "drift":
        drift_report(args.data_dir, args.model, args.smoke)
    elif args.baseline:
        evaluate_baselines(args.data_dir, args.smoke)
    else:
        evaluate_models(args.data_dir, args.model, args.smoke)


if __name__ == "__main__":
    main()
