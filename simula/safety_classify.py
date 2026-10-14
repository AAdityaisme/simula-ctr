"""Safety-tier classifier for new characters: majority, LLM, LightGBM, LightGBM on LLM labels.

Run: uv run python -m simula.safety_classify [--characters PATH] [--relabel]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import cohen_kappa_score

TIERS = ["sfw", "suggestive", "mature"]
CACHE = Path("reports/ad_classify_llm_labels.json")
PROMPT = """You label role-play characters for an ad-safety gate. Tiers:
- sfw: fine next to any ad, no romantic, sexual or dark-adult tone.
- suggestive: flirtatious, romantic, dominant or edgy undertones, nothing explicit.
- mature: adult-oriented; sexual, intense or dark themes an advertiser would avoid.
Reply with ONLY a JSON object mapping each id to one tier, e.g. {"a1": "sfw"}.

"""


def split_by_date(chars, test_frac=0.2):
    """Oldest (1 - test_frac) of characters to train, newest to test."""
    chars = chars.sort_values(["created_at", "character_id"]).reset_index(drop=True)
    cut = int(len(chars) * (1 - test_frac))
    return chars.iloc[:cut], chars.iloc[cut:]


def _matrix(vectorizer, frame):
    words = vectorizer.transform(frame["character_description"])
    extra = np.column_stack([
        (frame["creator_type"] == "official").astype(float),
        np.log1p(frame["num_interactions"].astype(float)),
    ])
    return hstack([words, csr_matrix(extra)]).tocsr()


def fit(train, labels):
    """Word-count + creator_type + num_interactions features into a multiclass LightGBM."""
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=2).fit(train["character_description"])
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               min_child_samples=5, verbose=-1, random_state=0)
    model.fit(_matrix(vectorizer, train), np.array([TIERS.index(t) for t in labels]))
    return vectorizer, model


def predict(fitted, frame):
    vectorizer, model = fitted
    return [TIERS[i] for i in model.predict(_matrix(vectorizer, frame))]


def metrics(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    out = {"n": len(true), "accuracy": float((true == pred).mean()),
           "kappa": float(cohen_kappa_score(true, pred, labels=TIERS))}
    for tier in TIERS:
        mask = true == tier
        out[f"recall_{tier}"] = float((pred[mask] == tier).mean()) if mask.any() else float("nan")
    return out


def llm_label(frame, model="claude-haiku-5-5", batch=25):
    """Label descriptions with headless Claude; returns (labels by id, total cost usd, seconds, calls)."""
    labels, cost, start = {}, 0.0, time.monotonic()
    rows = frame[["character_id", "character_description"]].values.tolist()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        body = "\n".join(f"{cid}: {desc}" for cid, desc in chunk)
        done = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json"],
                              input=PROMPT + body, capture_output=True, text=True, check=True)
        envelope = json.loads(done.stdout)
        cost += envelope.get("total_cost_usd", 0.0)
        text = envelope["result"]
        answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
        for cid, _ in chunk:
            tier = str(answer.get(cid, "")).strip().lower()
            # ponytail: an unparsed row fails closed to the strictest tier, same as rank.py
            labels[cid] = tier if tier in TIERS else "mature"
    return labels, cost, time.monotonic() - start, -(-len(rows) // batch)


def run(characters, relabel=False, seed=0):
    chars = pd.read_csv(characters, dtype={"character_id": str})
    train, test = split_by_date(chars)
    llm_test = test.sample(100, random_state=seed)
    llm_train = train.sample(200, random_state=seed)

    if relabel or not CACHE.exists():
        labels, cost, seconds, calls = llm_label(pd.concat([llm_test, llm_train]))
        CACHE.write_text(json.dumps({"model": "claude-haiku-5-5", "cost_usd": cost, "seconds": seconds,
                                     "calls": calls, "prompt": PROMPT, "labels": labels}, indent=1))
    cache = json.loads(CACHE.read_text())
    llm = cache["labels"]

    majority = train["safety_tier"].mode()[0]
    full = fit(train, train["safety_tier"])
    small_true = fit(llm_train, llm_train["safety_tier"])
    small_llm = fit(llm_train, [llm[c] for c in llm_train["character_id"]])
    arms = {
        "majority": lambda f: [majority] * len(f),
        "lgbm_true_4000": lambda f: predict(full, f),
        "lgbm_true_200": lambda f: predict(small_true, f),
        "lgbm_llm_200": lambda f: predict(small_llm, f),
        "llm_haiku": lambda f: [llm[c] for c in f["character_id"]],
    }
    rows = []
    for name, arm in arms.items():
        for scope, frame in (("test_1000", test), ("llm_subset_100", llm_test)):
            if name == "llm_haiku" and scope == "test_1000":
                continue
            rows.append({"arm": name, "scope": scope, **metrics(frame["safety_tier"], arm(frame))})
    llm_vs_true_train = metrics(llm_train["safety_tier"], [llm[c] for c in llm_train["character_id"]])
    return {"split": {"train": [str(train.created_at.min()), str(train.created_at.max()), len(train)],
                      "test": [str(test.created_at.min()), str(test.created_at.max()), len(test)]},
            "results": pd.DataFrame(rows), "llm_on_train_200": llm_vs_true_train,
            "llm_cost": {k: cache[k] for k in ("model", "cost_usd", "seconds", "calls")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--characters", default=str(Path.home() / "Desktop/Simula/ml-takehome/data/characters.csv"))
    parser.add_argument("--relabel", action="store_true")
    args = parser.parse_args()
    out = run(args.characters, args.relabel)
    pd.set_option("display.width", 200)
    print(out["split"])
    print(out["results"].round(3).to_string(index=False))
    print("llm vs true on its 200 train rows:", {k: round(v, 3) for k, v in out["llm_on_train_200"].items()})
    print(out["llm_cost"])
