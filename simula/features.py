import pandas as pd


CONTRACT = {
    "request": [
        "hour_of_day", "site_id", "site_domain", "site_category", "app_id",
        "app_domain", "app_category", "device_model", "device_type",
        "device_conn_type", "is_null_device", "is_site_side", "C1",
    ],
    "candidate": ["banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"],
    "character": ["safety_tier", "creator_type", "genre", "character_age_days"],
    "identity": ["character_id"],
    "numeric": ["hour_of_day", "character_age_days", "is_null_device", "is_site_side"],
    "never": [
        "id", "click", "session_msg_count", "num_interactions", "device_id",
        "device_ip", "character_name", "character_description", "created_at",
        "conversation_turn",
    ],
}

FEATURE_SETS = {
    "A": CONTRACT["request"] + CONTRACT["candidate"],
    "B": CONTRACT["request"] + CONTRACT["candidate"] + CONTRACT["character"],
    "C": CONTRACT["request"] + CONTRACT["candidate"] + CONTRACT["character"] + CONTRACT["identity"],
}

for _name, _features in FEATURE_SETS.items():
    assert not set(_features) & set(CONTRACT["never"]), f"forbidden feature in set {_name}"


def _string_counts(series):
    counts = series.dropna().astype(str).value_counts()
    return {value: int(count) for value, count in counts.items()}


def fit_encoder(train, features):
    """Fit categorical dictionaries and novelty counts on one training window."""
    features = list(features)
    forbidden = set(features) & set(CONTRACT["never"])
    if forbidden:
        raise ValueError(f"forbidden model features: {sorted(forbidden)}")
    missing = set(features) - set(train.columns)
    if missing:
        raise ValueError(f"missing model features: {sorted(missing)}")

    categories = {}
    for feature in features:
        if feature not in CONTRACT["numeric"]:
            categories[feature] = sorted(train[feature].dropna().astype(str).unique().tolist())
    return {
        "features": features,
        "categories": categories,
        "counts": {
            feature: _string_counts(train[feature])
            for feature in ("C14", "character_id")
        },
    }


def transform(df, encoder):
    """Apply fitted categories and return model columns in their fitted order."""
    result = pd.DataFrame(index=df.index)
    for feature in encoder["features"]:
        if feature in CONTRACT["numeric"]:
            result[feature] = pd.to_numeric(df[feature], errors="coerce").astype(float)
        else:
            categories = encoder["categories"][feature]
            values = df[feature].astype("string")
            values = values.where(values.isin(categories))
            result[feature] = pd.Categorical(
                values, categories=categories
            )
    return result


def novelty_flags(df, encoder):
    """Mark unseen and seen-but-rare candidate and character identifiers."""
    result = pd.DataFrame(index=df.index)
    for feature in ("C14", "character_id"):
        counts = encoder["counts"][feature]
        frequency = df[feature].astype("string").map(counts).fillna(0).astype(int)
        result[f"is_unseen_{feature}"] = frequency.eq(0)
        result[f"is_rare_{feature}"] = frequency.between(1, 49)
    return result
