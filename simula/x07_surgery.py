"""x07, approve counterfactual surgery (Astra 6): can Jev tell a coherent one-field edit of an ad from a broken one?

Constraints are learned on the refit window only: X -> Y is a dependency when every refit value of X has exactly one
Y. Items are one-field edits of a held-out (Oct 29-30) ad tuple (banner_pos, C14-C21):
- real (300): the edit turns one observed test tuple into another observed test tuple, 60 per field over the five
  fields that ever differ alone (banner_pos, C14, C18, C19, C20);
- broken (300): the edit swaps a dependent field (C15, C16, C17 or C21) for another refit value, in three families of
  100: C14 seen in refit; C14 new but C17 seen (only C17 -> C21 can bind); C14 and C17 both new. Every broken result
  is asserted absent from the full data.
Jev answers Choice `operation`; its score is P(single_field). Controls: exact-tuple lookup (accept iff the result was
seen in refit) and empirical dependency closure (accept iff no learned dependency with a refit-seen determinant is
violated). Pre-declared win: at closure's detection rate on broken items, Jev accepts a larger share of real items,
bootstrap interval over items above 0. Run:
  uv run python -m simula.x07_surgery states
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x07_surgery report
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from simula.data import add_flags, load, split
from simula.rank_pairs import read_answers
from simula.x06_evidence import WORK, _boot

ANSWERS = Path("reports/x07_jev_answers.jsonl")
FIELDS = ["banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
REAL_FIELDS, DEPENDENT = ["banner_pos", "C14", "C18", "C19", "C20"], ["C15", "C16", "C17", "C21"]
N_REAL, N_BROKEN, N_RECHECK = 60, 100, 50
NOTES = "Anonymized categorical ad attributes of one ad slot. C14 is the creative id by convention; the others are " \
        "undocumented."
OPERATION = {"type": "choice", "instructions": "Which operation does the evidence support for this proposed edit?",
             "criteria": {"single_field": "the edit alone gives a coherent ad",
                          "dependency_closure": "the edit is coherent only if dependent fields change with it",
                          "reject": "no coherent ad has this combination",
                          "unknown": "the evidence cannot tell"}}


def dependencies(tuples):
    """Every (X, Y) where each value of X in `tuples` has exactly one value of Y."""
    return [(x, y) for x, y in itertools.permutations(FIELDS, 2) if tuples.groupby(x)[y].nunique().max() == 1]


def closure_ok(result, mapping):
    """True when no dependency whose determinant value was seen maps it to a different dependent value."""
    return all(result[y] == m[result[x]] for (x, y), m in mapping.items() if result[x] in m)


def items(refit_tuples, test_tuples, all_tuples, seed=0):
    """Real and broken one-field edits: [(kind, family, original dict, field, new value)]."""
    rng = np.random.default_rng(seed)
    out = []
    arr = test_tuples[FIELDS].to_numpy()
    for j, f in enumerate(REAL_FIELDS):
        k = FIELDS.index(f)
        rest = pd.Series([tuple(np.delete(r, k)) for r in arr])
        pairs = [(a, b) for idx in rest.groupby(rest).indices.values() if len(idx) > 1
                 for a, b in itertools.permutations(idx, 2)]
        for i in rng.choice(len(pairs), min(N_REAL, len(pairs)), replace=False):
            a, b = pairs[i]
            out.append(("real", f, dict(zip(FIELDS, arr[a])), f, arr[b][k]))
    seen = {f: set(refit_tuples[f]) for f in ("C14", "C17")}
    families = {"c14_seen": test_tuples["C14"].isin(seen["C14"]),
                "c17_only": ~test_tuples["C14"].isin(seen["C14"]) & test_tuples["C17"].isin(seen["C17"]),
                "both_new": ~test_tuples["C14"].isin(seen["C14"]) & ~test_tuples["C17"].isin(seen["C17"])}
    observed = set(map(tuple, all_tuples[FIELDS].to_numpy()))
    for family, mask in families.items():
        pool, n = test_tuples[mask][FIELDS].to_numpy(), 0
        while n < N_BROKEN:
            row = dict(zip(FIELDS, pool[rng.integers(len(pool))]))
            f = "C21" if family == "c17_only" else DEPENDENT[rng.integers(len(DEPENDENT))]
            value = refit_tuples[f].iloc[rng.integers(len(refit_tuples))]
            if value != row[f] and tuple({**row, f: value}.values()) not in observed:
                out.append(("broken", family, row, f, value))
                n += 1
    return out


def evidence(item, refit_tuples, deps, counts):
    """Jev state for one edit: refit counts for the result, its values, its pairs and its dependencies."""
    _, _, row, f, value = item
    result = {**row, f: value}
    rel = [(x, y) for x, y in deps if f in (x, y)]
    return {"field_notes": NOTES,
            "training_dependencies": [f"each {x} value had exactly one {y} value" for x, y in deps],
            "original": row, "proposal": {"field": f, "new_value": value},
            "result_training_impressions": counts["tuple"].get(tuple(result.values()), 0),
            "value_training_impressions": {k: counts[k].get(v, 0) for k, v in result.items()},
            "pairs_with_edited_field": {k: counts[(f, k)].get(value, {}).get(v, 0) for k, v in result.items() if k != f},
            "dependency_evidence": [{"rule": f"{x} -> {y}", "determinant": result[x],
                                     "determinant_training_impressions": counts[x].get(result[x], 0),
                                     "dependent_values_in_training": counts[(x, y)].get(result[x], {}),
                                     "proposed_dependent": result[y]} for x, y in rel]}


def _counts(refit):
    t = refit[FIELDS].astype(str)
    counts = {"tuple": t.value_counts().to_dict(), **{f: t[f].value_counts().to_dict() for f in FIELDS}}
    for x, y in itertools.permutations(FIELDS, 2):  # {x value: {y value: refit rows}}
        sizes = t.groupby([x, y]).size()
        counts[(x, y)] = {xv: g.droplevel(0).to_dict() for xv, g in sizes.groupby(level=0)}
    return counts


def prepare(data_dir):
    """Refit dependencies and counts, and the 600 edits."""
    df = add_flags(load(data_dir))
    t = df[FIELDS + ["hour"]].astype(str).assign(hour=df["hour"])
    refit, test = split(t, "refit"), split(t, "test")
    refit_tuples = refit[FIELDS].drop_duplicates()
    deps = dependencies(refit_tuples)
    return deps, refit, items(refit_tuples, test[FIELDS].drop_duplicates(), t[FIELDS].drop_duplicates())


def write_states(deps, refit, its, path):
    """Jev input: every edit, plus the first N_RECHECK edits of each kind with the options reversed."""
    counts = _counts(refit)
    reordered = {**OPERATION, "criteria": dict(reversed(OPERATION["criteria"].items()))}
    with open(path, "w") as f:
        for i, item in enumerate(its):
            s = evidence(item, refit, deps, counts)
            f.write(json.dumps({"id": f"edit:{i}", "state": s, "questions": {"operation": OPERATION}}) + "\n")
            if i % 12 == 0:  # 50 rechecks spread over both kinds
                f.write(json.dumps({"id": f"reorder:{i}", "state": s, "questions": {"operation": reordered}}) + "\n")


def accept_at_detection(score, real, detection):
    """Best share of real items accepted (score >= t) by any threshold t that rejects >= `detection` of broken."""
    best = 0.0
    for t in np.append(np.unique(score), np.inf):
        if np.mean(score[~real] < t) >= detection - 1e-12:
            best = max(best, float(np.mean(score[real] >= t)))
    return best


def evaluate(deps, refit, its, answers):
    """Acceptance and detection of each arm, Jev at each control's detection rate, per-family detection."""
    refit_tuples = refit[FIELDS].drop_duplicates()
    seen = set(map(tuple, refit_tuples.to_numpy()))
    mapping = {(x, y): dict(zip(refit_tuples[x], refit_tuples[y])) for x, y in deps}
    results = [{**row, f: v} for _, _, row, f, v in its]
    real = np.array([k == "real" for k, *_ in its])
    family = np.array([fam for _, fam, *_ in its])
    ops = [answers[f"edit:{i}"]["answers"]["operation"] for i in range(len(its))]
    jev = np.array([o["probabilities"]["single_field"] for o in ops])
    controls = {"exact_tuple": np.array([tuple(r.values()) in seen for r in results]),
                "closure": np.array([closure_ok(r, mapping) for r in results])}
    table = {}
    for name, accept in {**controls, "jev_argmax_single_field": np.array([o["choice"] == "single_field" for o in ops])}.items():
        table[name] = {"accept_real": float(accept[real].mean()), "detect_broken": float(1 - accept[~real].mean()),
                       "detect_by_family": {fam: float(1 - accept[family == fam].mean()) for fam in np.unique(family[~real])},
                       "accept_by_field": {fam: float(accept[family == fam].mean()) for fam in REAL_FIELDS}}
    matched = {}
    for name, accept in controls.items():
        d = 1 - accept[~real].mean()

        def diff(i, accept=accept, d=d):
            return accept_at_detection(jev[i], real[i], 1 - accept[i][~real[i]].mean()) - accept[i][real[i]].mean()
        matched[name] = {"detection": float(d), "jev_accept_real": accept_at_detection(jev, real, d),
                         "control_accept_real": float(accept[real].mean()), "jev_minus_control": _boot(diff, len(its))}
    rechecks = [k for k in answers if k.startswith("reorder:")]
    calls = list(answers.values())
    return {"dependencies": [f"{x} -> {y}" for x, y in deps], "items": {"real": int(real.sum()), "broken": int((~real).sum())},
            "arms": table, "jev_at_matched_detection": matched, "jev_auroc_real_vs_broken": float(roc_auc_score(real, jev)),
            "jev_choice_counts": {k: pd.Series([o["choice"] for o, r in zip(ops, real) if r == (k == "real")]).value_counts().to_dict()
                                  for k in ("real", "broken")},
            "reorder_same_choice_share": float(np.mean([answers[k]["answers"]["operation"]["choice"] ==
                                                        ops[int(k.split(":")[1])]["choice"] for k in rechecks])),
            "win": bool(matched["closure"]["jev_minus_control"][1] > 0),
            "jev": {"calls": len(calls), "attempts": int(sum(c["attempts"] for c in calls)),
                    "models": sorted({c["model"] for c in calls}), "input_tokens": int(sum(c["tokens_in"] for c in calls)),
                    "call_seconds_median": float(np.median([c["seconds"] for c in calls]))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    deps, refit, its = prepare(args.data)
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        write_states(deps, refit, its, WORK / "x07_states.jsonl")
        print("dependencies:", deps, "items:", len(its), "wrote", WORK / "x07_states.jsonl")
    else:
        out = evaluate(deps, refit, its, read_answers(ANSWERS))
        Path("reports/x07_counterfactual_surgery.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps(out, indent=1, default=float))
