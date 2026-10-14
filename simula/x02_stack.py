"""x02: learned stacking of B and Jev on exact-context pairs (the pending "experiment 5").

Fit day: the `rank_pairs` pairs built on Oct 28 (calibrate window), each asked the e12 `pick` question with pCTR in
the state. Test day: the 1,735 e12 test pairs, reusing their committed `with:` answers; no new test calls. Run:
  uv run python -m simula.x02_stack states        # Oct 28 Jev input JSONL, written outside the checkout
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
  uv run python -m simula.x02_stack report        # reports/x02_stack.json

Pre-registered: logistic regressions fit on Oct 28 pairs only, unpenalized, predicting "X1 is the clicked one".
S0 = [logit pCTR_X1 - logit pCTR_X2, |pCTR_X1 - pCTR_X2|, X1 seen in refit, X2 seen in refit]; S1 = S0 + Jev's
P(X1) - 0.5. B alone is p1(1-p2) / (p1(1-p2) + p2(1-p1)). Win: S1 beats S0 on test pair log loss, with the
character-bootstrap interval of S1 - S0 below 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from simula import train
from simula.data import add_flags, load, split
from simula.jev_ctr import _row_loss, bootstrap_diff
from simula.rank_pairs import PICK, add_history, build_pairs, history, read_answers, state

WORK = Path.home() / "simula-local/rank-jev-work/exp-x-fatigue-stack"
ANSWERS = Path("reports/x02_stack_jev_answers.jsonl")
E12 = Path("reports/rank_e12_jev_answers.jsonl")
S0 = ["logit_diff", "abs_margin", "x1_seen", "x2_seen"]


def annotate(df, window, bundle, hist):
    """Pairs in one window, and their rows with B's calibrated pCTR and refit history."""
    rows = split(df, window)
    pairs = build_pairs(rows)
    rows = add_history(rows.loc[np.union1d(pairs["x1"], pairs["x2"])].copy(), hist)
    rows["pctr"] = train.predict(bundle, rows)
    return pairs, rows


def _logit(p):
    return np.log(p / (1 - p))


def b_pair_prob(p1, p2):
    """B alone: P(X1 is the click | exactly one click), treating the pCTRs as true rates."""
    return p1 * (1 - p2) / (p1 * (1 - p2) + p2 * (1 - p1))


def jev_p_x1(answers, prefix, pairs):
    """Jev's normalized P(X1) for each pair from answers keyed `<prefix>:<pair>`."""
    probs = [answers[f"{prefix}:{p}"]["answers"]["pick"]["probabilities"] for p in pairs["pair"]]
    return np.array([p["X1"] / (p["X1"] + p["X2"]) for p in probs])


def design(pairs, rows, p_jev):
    """S1's columns in order (S0 is the first four), and y = X1 is the clicked one."""
    p1, p2 = rows.loc[pairs["x1"], "pctr"].to_numpy(), rows.loc[pairs["x2"], "pctr"].to_numpy()
    X = np.column_stack([_logit(p1) - _logit(p2), np.abs(p1 - p2),
                         rows.loc[pairs["x1"], "C14_impressions"].gt(0).to_numpy(),
                         rows.loc[pairs["x2"], "C14_impressions"].gt(0).to_numpy(), p_jev - 0.5]).astype(float)
    return X, pairs["clicked"].eq("X1").to_numpy().astype(int), b_pair_prob(p1, p2)


def fit(X, y):
    """Unpenalized logistic regression, with coefficient standard errors from the inverse Fisher information."""
    model = LogisticRegression(C=np.inf, max_iter=1000).fit(X, y)
    Z = np.column_stack([np.ones(len(X)), X])
    p = model.predict_proba(X)[:, 1]
    se = np.sqrt(np.diag(np.linalg.inv(Z.T @ (Z * (p * (1 - p))[:, None]))))
    return model, {"coef": [float(model.intercept_[0]), *map(float, model.coef_[0])], "se": list(map(float, se))}


def _accuracy(prob, y):
    return np.where(prob == 0.5, 0.5, ((prob > 0.5) == y).astype(float))


def evaluate(fit_pairs, fit_rows, fit_answers, test_pairs, test_rows, test_answers):
    """Fit S0 and S1 on the fit-day pairs; score B, S0 and S1 on the test pairs."""
    X_fit, y_fit, b_fit = design(fit_pairs, fit_rows, jev_p_x1(fit_answers, "fit", fit_pairs))
    X_test, y_test, b_test = design(test_pairs, test_rows, jev_p_x1(test_answers, "with", test_pairs))
    s0, s0_coef = fit(X_fit[:, :4], y_fit)
    s1, s1_coef = fit(X_fit, y_fit)
    probs = {"B": b_test, "S0": s0.predict_proba(X_test[:, :4])[:, 1], "S1": s1.predict_proba(X_test)[:, 1],
             "jev_alone": X_test[:, 4] + 0.5}
    groups = test_pairs["character_id"].to_numpy()
    loss = {n: _row_loss(y_test, p) for n, p in probs.items()}
    acc = {n: _accuracy(p, y_test) for n, p in probs.items()}
    zeros = np.zeros(len(y_test))
    arms = [{"arm": n, "pair_log_loss": bootstrap_diff(groups, loss[n], zeros),
             "accuracy": bootstrap_diff(groups, acc[n], zeros)} for n in probs]
    diffs = [{"a": a, "b": b, "log_loss": bootstrap_diff(groups, loss[a], loss[b]),
              "accuracy": bootstrap_diff(groups, acc[a], acc[b])}
             for a, b in (("S1", "S0"), ("S0", "B"), ("S1", "B"), ("jev_alone", "B"))]
    s1_vs_s0 = diffs[0]["log_loss"]
    calls = [fit_answers[f"fit:{p}"] for p in fit_pairs["pair"]]
    return {
        "fit_day": {"pairs": len(fit_pairs), "groups": int(fit_pairs["context_group"].nunique()),
                    "characters": int(fit_pairs["character_id"].nunique()), "x1_clicked_share": float(y_fit.mean()),
                    "in_sample_pair_log_loss": {"B": float(_row_loss(y_fit, b_fit).mean()),
                                                "S0": float(_row_loss(y_fit, s0.predict_proba(X_fit[:, :4])[:, 1]).mean()),
                                                "S1": float(_row_loss(y_fit, s1.predict_proba(X_fit)[:, 1]).mean())}},
        "test": {"pairs": len(test_pairs), "characters": int(test_pairs["character_id"].nunique())},
        "jev_follows_b": {day: float(np.mean(np.sign(X[:, 4]) == np.sign(X[:, 0]))) for day, X in
                          (("fit", X_fit[X_fit[:, 4] != 0]), ("test", X_test[X_test[:, 4] != 0]))},
        "features": S0 + ["jev_p_x1_minus_half"],
        "S0": s0_coef, "S1": s1_coef, "jev_coef": {"coef": s1_coef["coef"][-1], "se": s1_coef["se"][-1]},
        "arms": arms, "diffs": diffs,
        "win": {"s1_minus_s0_log_loss": s1_vs_s0, "verdict": "win" if s1_vs_s0[2] < 0 else
                "loss" if s1_vs_s0[1] > 0 else "null"},
        "jev_fit_day": {"calls": len(calls), "attempts": int(sum(a["attempts"] for a in calls)),
                        "models": sorted({a["model"] for a in calls}),
                        "input_tokens": int(sum(a["tokens_in"] for a in calls)),
                        "call_seconds_sum": float(sum(a["seconds"] for a in calls)),
                        "call_seconds_median": float(np.median([a["seconds"] for a in calls]))},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    df = add_flags(load(args.data))
    hist, bundle = history(split(df, "refit")), train.load_bundle("bundle/B")
    fit_pairs, fit_rows = annotate(df, "calibrate", bundle, hist)
    print(f"Oct 28: {len(fit_pairs)} pairs from {fit_pairs['context_group'].nunique()} groups")
    if args.step == "states":
        WORK.mkdir(parents=True, exist_ok=True)
        with open(WORK / "x02_states.jsonl", "w") as f:
            for p in fit_pairs.itertuples():
                s = state((fit_rows.loc[p.x1], fit_rows.loc[p.x2]), with_pctr=True)
                f.write(json.dumps({"id": f"fit:{p.pair}", "state": s, "questions": {"pick": PICK}}) + "\n")
        print("wrote", WORK / "x02_states.jsonl")
    else:
        test_pairs, test_rows = annotate(df, "test", bundle, hist)
        out = evaluate(fit_pairs, fit_rows, read_answers(ANSWERS), test_pairs, test_rows, read_answers(E12))
        Path("reports/x02_stack.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps(out, indent=1, default=float))
