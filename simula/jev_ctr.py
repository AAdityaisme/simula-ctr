"""Does a text classifier (Jev) reading the character description help the CTR model?

Arms differ only in character features; everything else is model B's recipe (windows, params, early stopping,
calibration). Run after jev_annotate.py: uv run python -m simula.jev_ctr [--data DIR]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from simula.data import add_flags, load, split
from simula.features import CONTRACT, FEATURE_SETS, fit_encoder, transform
from simula.train import apply_calibration, calibrate, choose_calibration, fit, refit

JEV = Path("reports/jev_characters.csv")
GENRES = ["anime", "comedic", "fantasy", "historical", "horror", "mentor", "mystery", "romance", "sci", "slice"]
JEV_FEATURES = [f"jev_genre_{g}" for g in GENRES] + ["jev_romance", "jev_darkness", "jev_warmth", "jev_brand_safe"]
TEXT_FEATURES = ["frame", "trait_a", "trait_b"]
# ponytail: extends the shared contract at import instead of editing features.py on this experiment branch
CONTRACT["numeric"] = CONTRACT["numeric"] + JEV_FEATURES

B = FEATURE_SETS["B"]
NO_GENRE = [f for f in B if f != "genre"]
ARMS = {  # B first: the calibration recipe is chosen on its dev booster, as train_all does
    "B_submitted": B,
    "A_no_character": FEATURE_SETS["A"],
    "B_minus_genre": NO_GENRE,
    "B_minus_genre_plus_text": NO_GENRE + TEXT_FEATURES,
    "B_minus_genre_plus_jev": NO_GENRE + JEV_FEATURES,
    "B_plus_jev": B + JEV_FEATURES,
}
COMPARISONS = [  # (candidate, baseline): negative difference = candidate has lower log loss
    ("B_minus_genre_plus_jev", "B_minus_genre"),
    ("B_minus_genre_plus_jev", "B_submitted"),
    ("B_minus_genre_plus_jev", "B_minus_genre_plus_text"),
    ("B_plus_jev", "B_submitted"),
]


def text_slots(descriptions):
    """Split each templated description into its sentence frame and its two trait adjectives."""
    vocab = sorted(descriptions.str.extract(r"^An? ([\w-]+)")[0].dropna().unique())
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, vocab)) + r")\b")
    traits = descriptions.map(pattern.findall)
    return pd.DataFrame({
        "frame": descriptions.map(lambda s: re.sub(r"^An ", "A ", pattern.sub("_", s))),
        "trait_a": traits.str[0],
        "trait_b": traits.str[1],
    }, index=descriptions.index)


def bootstrap_diff(groups, loss_a, loss_b, n_boot=1000, seed=0):
    """Paired cluster bootstrap of mean(loss_a - loss_b), resampling whole characters. Returns (diff, lo, hi)."""
    codes, _ = pd.factorize(groups)
    diff_sum = np.bincount(codes, weights=loss_a - loss_b)
    rows = np.bincount(codes)
    weights = np.random.default_rng(seed).multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=n_boot)
    boots = (weights @ diff_sum) / (weights @ rows)
    return float(diff_sum.sum() / rows.sum()), *map(float, np.percentile(boots, [2.5, 97.5]))


def _row_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(data_dir):
    df = add_flags(load(data_dir))
    df = df.join(text_slots(df["character_description"]))
    df = df.merge(pd.read_csv(JEV, dtype={"character_id": str})[["character_id"] + JEV_FEATURES],
                  on="character_id", how="left", validate="many_to_one")
    dev_fit, select, refit_rows, test = (split(df, w) for w in ("dev_fit", "select", "refit", "test"))
    unseen = ~test["character_id"].isin(set(refit_rows["character_id"]))

    rows, losses, choice = [], {}, None
    for name, features in ARMS.items():
        encoder = fit_encoder(dev_fit, features)
        dev_booster, rounds = fit(dev_fit, select, features, encoder)
        if choice is None:
            choice = choose_calibration(dev_booster, df)
            calibration = df.loc[df["day"].isin([141028] if choice["days"] == 1 else [141027, 141028])]
        encoder = fit_encoder(refit_rows, features)
        booster = refit(refit_rows, features, encoder, rounds)
        delta = calibrate(booster.predict(transform(calibration, encoder)), calibration["click"], choice["shrink"])
        p = apply_calibration(booster.predict(transform(test, encoder)), delta)
        y = test["click"].to_numpy()
        losses[name] = _row_loss(y, p)
        rows.append({"arm": name, "trees": int(rounds), "test_log_loss": log_loss(y, p), "test_auc": roc_auc_score(y, p),
                     "unseen_char_log_loss": log_loss(y[unseen], p[unseen]),
                     "unseen_char_auc": roc_auc_score(y[unseen], p[unseen])})
        print(rows[-1], flush=True)
    diffs = [{"candidate": a, "baseline": b,
              **dict(zip(("diff", "ci_lo", "ci_hi"), bootstrap_diff(test["character_id"], losses[a], losses[b])))}
             for a, b in COMPARISONS]
    return {"calibration": choice["recipe"], "test_rows": len(test), "unseen_char_rows": int(unseen.sum()),
            "arms": rows, "comparisons": diffs}


def diagnostics(characters):
    """How Jev's answers line up with the name-prefix genre and the given safety tier (not a CTR result)."""
    from sklearn.metrics import cohen_kappa_score

    chars = pd.read_csv(characters, dtype={"character_id": str}).merge(pd.read_csv(JEV, dtype={"character_id": str}))
    genre = chars["character_name"].str.split("_", n=1).str[0]
    jev_genre = chars[[f"jev_genre_{g}" for g in GENRES]].idxmax(axis=1).str.removeprefix("jev_genre_")
    tiers = ["sfw", "suggestive", "mature"]
    jev_tier = chars[[f"jev_tier_{t}" for t in tiers]].idxmax(axis=1).str.removeprefix("jev_tier_")
    return {"genre_accuracy": float((jev_genre == genre).mean()),
            "tier_accuracy": float((jev_tier == chars["safety_tier"]).mean()),
            "tier_kappa": float(cohen_kappa_score(chars["safety_tier"], jev_tier)),
            "jev_tier_counts": jev_tier.value_counts().to_dict(),
            "true_tier_counts": chars["safety_tier"].value_counts().to_dict()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path.home() / "Desktop/Simula/ml-takehome/data"))
    args = parser.parse_args()
    out = {"diagnostics": diagnostics(Path(args.data) / "characters.csv"), **run(args.data)}
    Path("reports/jev_ctr.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["diagnostics"], indent=1))
    print(pd.DataFrame(out["arms"]).round(6).to_string(index=False))
    print(pd.DataFrame(out["comparisons"]).to_string(index=False))
