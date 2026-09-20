import numpy as np


def constant(train):
    """Fit a constant click-rate predictor."""
    return {"kind": "constant", "prior": float(train["click"].mean())}


def lookup(train, cols, m=20):
    """Fit an m-estimate click-rate lookup over one or more columns."""
    cols = list(cols)
    if not cols:
        raise ValueError("lookup needs at least one column")
    prior = float(train["click"].mean())
    grouped = train.groupby(cols, dropna=False)["click"].agg(["sum", "count"])
    probabilities = (grouped["sum"] + m * prior) / (grouped["count"] + m)

    keys = probabilities.index.tolist()
    if len(cols) == 1:
        keys = [(key,) for key in keys]
    rates = dict(zip(keys, probabilities.to_numpy(dtype=float), strict=True))
    return {
        "kind": "lookup",
        "cols": cols,
        "m": m,
        "prior": prior,
        "rates": rates,
    }


def predict(fitted, df):
    """Apply a fitted baseline and return a probability array."""
    if fitted["kind"] == "constant":
        return np.full(len(df), fitted["prior"], dtype=float)
    if fitted["kind"] != "lookup":
        raise ValueError(f"Unknown baseline kind {fitted['kind']!r}")

    cols = fitted["cols"]
    keys = df[cols].itertuples(index=False, name=None)
    rates = fitted["rates"]
    prior = fitted["prior"]
    return np.fromiter((rates.get(key, prior) for key in keys), dtype=float, count=len(df))
