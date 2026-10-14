"""x08, diagnose a broken pipeline before retraining it (Astra 8): does Jev pick repairs that restore B's log loss?

Incidents are 3-hour serving batches of Oct 29-30 rows. Each is clean (a temporal-shift control) or has a fault
injected into an in-memory copy on a random share f ~ U(0.1, 1) of rows: `join` gives rows another character's
metadata; `sentinel` writes C20's "-1" as "-1.0" and the null-device hash in upper case; `both` does both. 40 of each.
Each policy takes two steps, each one of restore_join, restore_missing_encoding, retrain (switch to a cached B refit
on Oct 22-28) or hold; repairs apply before B scores. States are label-free batch statistics against Oct 28 and never
name the fault. Controls: fixed invariant checks, and a depth-3 tree trained on 200 incidents from Oct 25-27 blocks
with separate seeds. Pre-declared win: Jev restores a larger share of lost log loss than both controls, paired
bootstrap interval over incidents above 0, at a false-repair rate no higher than each control's. v2 lists the options
in a seeded random order per call, rechecks every fourth incident with that order reversed, and also counts every
unnecessary restore execution. Run:
  uv run python -m simula.x08_repair states1     # step-1 Jev input
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x08_repair states2     # step-2 Jev input, after Jev's step-1 repairs; then jev_ask again
  uv run python -m simula.x08_repair report
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

from simula import train
from simula.data import add_flags, load, load_characters
from simula.features import FEATURE_SETS, fit_encoder, transform
from simula.rank_pairs import read_answers
from simula.x06_evidence import WORK, _boot, option_orders, order_sensitivity, ordered

ANSWERS = Path("reports/x08_v2_jev_answers.jsonl")
FAULTS = ["none", "join", "sentinel", "both"]
META = ["safety_tier", "creator_type", "genre", "created_at"]
ENCODED = ["C20", "device_id", "is_null_device"]
ACTIONS = ["restore_join", "restore_missing_encoding", "retrain", "hold"]
N_PER_FAULT, N_TRAIN = 40, 200
NOTES = ("Serving batch for model B. Character metadata is joined on character_id from a registry with one row per "
         "character. C20 uses -1 as a sentinel value. device_id uses one repeated hash for a missing id. "
         "Reference is Oct 28 traffic through the same pipeline.")
INTERVENTION = {"type": "choice", "instructions": "What intervention fits this incident?", "criteria": {
    "restore_join": "re-run the character metadata join from the source registry",
    "restore_missing_encoding": "re-apply the raw encoding of sentinel and missing values",
    "retrain": "switch to the model refit on more recent days",
    "hold": "change nothing"}}


def blocks(df, days):
    """3-hour blocks of the given days, as lists of row labels."""
    rows = df[df["day"].isin(days)]
    return [g.index for _, g in rows.groupby(rows["day"] * 10 + rows["hour_of_day"] // 3)]


def inject(batch, fault, share, chars, rng):
    """In-memory faulty copy of a clean batch."""
    out = batch.copy()
    if fault in ("join", "both"):
        hit = rng.random(len(out)) < share
        donor = chars.iloc[rng.integers(len(chars), size=int(hit.sum()))]
        out.loc[hit, "safety_tier"] = donor["safety_tier"].to_numpy()
        out.loc[hit, "creator_type"] = donor["creator_type"].to_numpy()
        out.loc[hit, "genre"] = donor["character_name"].str.split("_", n=1).str[0].to_numpy()
        out.loc[hit, "created_at"] = donor["created_at"].to_numpy()
        day = pd.to_datetime(out.loc[hit, "day"].astype(str), format="%y%m%d")
        out.loc[hit, "character_age_days"] = (day - pd.to_datetime(out.loc[hit, "created_at"]).dt.normalize()).dt.days
    if fault in ("sentinel", "both"):
        hit = rng.random(len(out)) < share
        out.loc[hit & out["C20"].eq("-1"), "C20"] = "-1.0"
        null = hit & out["is_null_device"]
        out.loc[null, "device_id"] = out.loc[null, "device_id"].str.upper()
        out.loc[null, "is_null_device"] = False
    return out


def repair(batch, clean, action):
    """Apply one restore action by re-deriving its columns from the clean source."""
    cols = {"restore_join": META + ["character_age_days"], "restore_missing_encoding": ENCODED}.get(action, [])
    out = batch.copy()
    out[cols] = clean[cols]
    return out


def stats(batch, model, ref=None):
    """Label-free incident statistics; with `ref`, each is reported next to the reference value."""
    encoder = model[1]
    ids = batch.groupby("character_id")
    multi = ids[META].nunique().max(axis=1).gt(1)[ids.size().ge(2)]
    s = {"rows": len(batch),
         "share_of_repeat_characters_with_several_metadata_rows": float(multi.mean()),
         "share_C20_sentinel": float(batch["C20"].eq("-1").mean()),
         "share_null_device": float(batch["is_null_device"].mean()),
         "share_C20_unseen_by_model": float((~batch["C20"].isin(encoder["categories"]["C20"])).mean()),
         "share_C14_unseen_by_model": float((~batch["C14"].astype(str).isin(encoder["categories"]["C14"])).mean()),
         "share_site_side": float(batch["is_site_side"].mean()),
         "mean_predicted_ctr": float(train.predict(model, batch).mean())}
    if ref is None:
        return s
    return {k: {"batch": round(v, 4), "reference": round(ref[k], 4)} for k, v in s.items() if k != "rows"} | \
        {"rows": len(batch)}


def features(s, done):
    """Numeric tree/rule input: batch minus reference for each statistic, plus which repairs are already applied."""
    return [s[k]["batch"] - s[k]["reference"] for k in sorted(s) if k != "rows"] + \
        [float("restore_join" in done), float("restore_missing_encoding" in done)]


def state(s, done):
    """Jev state for one step of one incident."""
    return {"notes": NOTES, "statistics": s, "actions_already_taken": list(done)}


def invariants(s, done):
    """Fixed invariant checks: a registry join can never give one id two metadata rows; encodings never go unseen."""
    if s["share_of_repeat_characters_with_several_metadata_rows"]["batch"] > 0 and "restore_join" not in done:
        return "restore_join"
    unseen = s["share_C20_unseen_by_model"]
    null = s["share_null_device"]
    if ("restore_missing_encoding" not in done and (unseen["batch"] > unseen["reference"] + 0.01
                                                     or abs(null["batch"] - null["reference"]) > 0.05)):
        return "restore_missing_encoding"
    return "hold"


def correct(fault, done):
    """The repair a perfect diagnostician takes next: the join first, then the encoding, then hold."""
    for action, faults in (("restore_join", ("join", "both")), ("restore_missing_encoding", ("sentinel", "both"))):
        if fault in faults and action not in done:
            return action
    return "hold"


def unnecessary(actions, fault):
    """Restore executions that fix nothing: that fault was never injected, or an earlier step already restored it."""
    needed = {"restore_join": fault in ("join", "both"), "restore_missing_encoding": fault in ("sentinel", "both")}
    return sum(a in needed and (not needed[a] or a in actions[:k]) for k, a in enumerate(actions))


def recent_model(df, rounds, seed=0):
    """B refit on Oct 22-28 (300k-row subsample, B's tree count), calibrated like B on Oct 28 (in-sample). Cached."""
    path = WORK / "x08_recent"
    calibration = df[df["day"].eq(141028)]
    if not (path / "model.txt").exists():
        rows = df[df["day"].between(141022, 141028)].sample(300_000, random_state=seed)
        encoder = fit_encoder(rows, FEATURE_SETS["B"])
        booster = train.refit(rows, FEATURE_SETS["B"], encoder, rounds)
        delta = train.calibrate(booster.predict(transform(calibration, encoder)), calibration["click"], 0.5)
        train._save_bundle(path, booster, encoder, {"calibration": {"delta": delta}})
    return train.load_bundle(path)


def incidents(df, days, n_per_fault, seed):
    """[(fault, share, block rows)]: every block gets each fault in turn."""
    rng, bs = np.random.default_rng(seed), blocks(df, days)
    return [(FAULTS[i % 4], float(rng.uniform(0.1, 1.0)), bs[(i // 4) % len(bs)]) for i in range(4 * n_per_fault)]


def prepare(data_dir):
    """Load data and models, build eval and tree-training incidents, fit the tree."""
    r = SimpleNamespace()
    r.df = add_flags(load(data_dir))
    r.chars = load_characters(data_dir)
    r.old = train.load_bundle("bundle/B")
    r.recent = recent_model(r.df, r.old[2]["best_iteration"])
    r.ref = stats(r.df[r.df["day"].eq(141028)], r.old)
    r.eval = incidents(r.df, [141029, 141030], N_PER_FAULT, seed=8)
    xs, ys = [], []
    rng = np.random.default_rng(80)  # tree seeds, separate from the eval seeds
    for fault, share, idx in incidents(r.df, [141025, 141026, 141027], N_TRAIN // 4, seed=81):
        clean = r.df.loc[idx]
        batch, done = inject(clean, fault, share, r.chars, rng), []
        for _ in range(2):
            s, label = stats(batch, r.old, r.ref), correct(fault, done)
            xs.append(features(s, done))
            ys.append(label)
            batch, done = repair(batch, clean, label), done + [label]
    r.tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(xs, ys)
    return r


def faulty(r, i):
    """The clean and faulty batch of eval incident i, rebuilt deterministically from its own seed."""
    fault, share, idx = r.eval[i]
    clean = r.df.loc[idx]
    return clean, inject(clean, fault, share, r.chars, np.random.default_rng(1000 + i))


def step(r, batch, clean, model, action):
    """Apply one action. Returns (batch, model)."""
    return (batch, r.recent) if action == "retrain" else (repair(batch, clean, action), model)


def run_policy(r, choose):
    """Two steps of `choose(incident, step, stats, done) -> action` on every eval incident. Returns per-incident rows."""
    out = []
    for i in range(len(r.eval)):
        clean, batch = faulty(r, i)
        model, done = r.old, []
        for k in (1, 2):
            action = choose(i, k, stats(batch, model, r.ref), done)
            batch, model = step(r, batch, clean, model, action)
            done.append(action)
        out.append({"actions": done, "final_loss": log_loss(clean["click"], train.predict(model, batch), labels=[0, 1])})
    return out


def write_step(r, k, path, step1=None):
    """Jev input for step k, options in a seeded order per incident. Step 2 replays Jev's step-1 answers."""
    orders = option_orders(INTERVENTION, len(r.eval), seed=80 + k)
    with open(path, "w") as f:
        for i in range(len(r.eval)):
            clean, batch = faulty(r, i)
            model, done = r.old, []
            if k == 2:
                a = step1[f"s1:{i}"]["answers"]["intervention"]["choice"]
                batch, model = step(r, batch, clean, model, a)
                done = [a]
            s = state(stats(batch, model, r.ref), done)
            f.write(json.dumps({"id": f"s{k}:{i}", "state": s,
                                "questions": {"intervention": ordered(INTERVENTION, orders[i])}}) + "\n")
            if i % 4 == 0:  # 40 rechecks per step, one per block and fault cycle
                f.write(json.dumps({"id": f"r{k}:{i}", "state": s,
                                    "questions": {"intervention": ordered(INTERVENTION, orders[i][::-1])}}) + "\n")


def evaluate(r, answers):
    """Lost and restored log loss and false repairs for each policy; Jev minus each control, bootstrap over incidents."""
    faults = np.array([f for f, _, _ in r.eval])
    base = []
    for i in range(len(r.eval)):
        clean, batch = faulty(r, i)
        y = clean["click"]
        base.append((log_loss(y, train.predict(r.old, clean), labels=[0, 1]),
                     log_loss(y, train.predict(r.old, batch), labels=[0, 1])))
    clean_loss, faulty_loss = map(np.array, zip(*base))
    policies = {
        "hold": lambda i, k, s, done: "hold",
        "invariants": lambda i, k, s, done: invariants(s, done),
        "tree": lambda i, k, s, done: r.tree.predict([features(s, done)])[0],
        "oracle": lambda i, k, s, done: correct(faults[i], done),
        "jev": lambda i, k, s, done: answers[f"s{k}:{i}"]["answers"]["intervention"]["choice"]}
    broken = faults != "none"
    lost = faulty_loss - clean_loss
    table, restored = {}, {}
    for name, choose in policies.items():
        rows = run_policy(r, choose)
        final = np.array([x["final_loss"] for x in rows])
        restored[name] = faulty_loss - final
        false = np.array([any(a == "restore_join" and f not in ("join", "both")
                              or a == "restore_missing_encoding" and f not in ("sentinel", "both") for a in x["actions"])
                          for x, f in zip(rows, faults)])
        table[name] = {"restored_share": float(restored[name][broken].sum() / lost[broken].sum()),
                       "restored_share_by_fault": {f: float(restored[name][faults == f].sum() / lost[faults == f].sum())
                                                   for f in FAULTS[1:]},
                       "false_repair_rate": float(false.mean()),
                       "unnecessary_restore_executions": int(sum(unnecessary(x["actions"], f) for x, f in zip(rows, faults))),
                       "incidents_with_an_unnecessary_restore": float(np.mean([unnecessary(x["actions"], f) > 0
                                                                               for x, f in zip(rows, faults)])),
                       "sequences": pd.Series([" > ".join(x["actions"]) for x in rows]).groupby(faults).value_counts()
                       .unstack(fill_value=0).to_dict(orient="index"),
                       "retrain_rate": float(np.mean(["retrain" in x["actions"] for x in rows])),
                       "loss_change_on_clean_incidents": float(-restored[name][~broken].mean()),
                       "first_actions": pd.Series([x["actions"][0] for x in rows]).groupby(faults).value_counts()
                       .unstack(fill_value=0).to_dict(orient="index")}
    idx = np.flatnonzero(broken)

    def share_diff(name):
        return lambda b: (restored["jev"][idx[b]].sum() - restored[name][idx[b]].sum()) / lost[idx[b]].sum()
    diffs = {name: _boot(share_diff(name), len(idx)) for name in ("invariants", "tree")}
    win = all(diffs[n][1] > 0 and table["jev"]["false_repair_rate"] <= table[n]["false_repair_rate"] for n in diffs)
    order = {f"step{k}": order_sensitivity(
        [answers[f"s{k}:{i}"]["answers"]["intervention"] for i in range(len(r.eval))],
        {int(key.split(":")[1]): a["answers"]["intervention"] for key, a in answers.items() if key.startswith(f"r{k}:")},
        option_orders(INTERVENTION, len(r.eval), seed=80 + k)) for k in (1, 2)}
    calls = list(answers.values())
    return {"option_order": order, "incidents": {f: int((faults == f).sum()) for f in FAULTS},
            "mean_lost_log_loss_by_fault": {f: float(lost[faults == f].mean()) for f in FAULTS},
            "policies": table, "jev_restored_share_minus": diffs, "win": bool(win),
            "jev": {"calls": len(calls), "attempts": int(sum(c["attempts"] for c in calls)),
                    "models": sorted({c["model"] for c in calls}), "input_tokens": int(sum(c["tokens_in"] for c in calls)),
                    "call_seconds_median": float(np.median([c["seconds"] for c in calls]))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states1", "states2", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    run = prepare(args.data)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.step == "states1":
        write_step(run, 1, WORK / "x08_v2_states1.jsonl")
    elif args.step == "states2":
        write_step(run, 2, WORK / "x08_v2_states2.jsonl", read_answers(ANSWERS))
    else:
        out = evaluate(run, read_answers(ANSWERS))
        Path("reports/x08_pipeline_repair_v2.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps(out, indent=1, default=float))
