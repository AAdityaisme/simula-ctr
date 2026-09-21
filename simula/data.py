from pathlib import Path

import pandas as pd


HASHED_COLUMNS = (
    "id",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "character_id",
)

WINDOWS = {
    "dev_fit": (14102100, 14102700),
    "select": (14102700, 14102800),
    "refit": (14102100, 14102800),
    "calibrate": (14102800, 14102900),
    "test": (14102900, None),
}


def _csv_paths(data_dir):
    data_dir = Path(data_dir)
    regular = (data_dir / "impressions.csv", data_dir / "characters.csv")
    smoke = (
        data_dir / "smoke_impressions.csv",
        data_dir / "smoke_characters.csv",
    )
    if all(path.exists() for path in regular):
        return regular
    if all(path.exists() for path in smoke):
        return smoke
    raise FileNotFoundError(f"Could not find an impressions/characters CSV pair in {data_dir}")


def load(data_dir):
    """Read, validate, and join impression and character data."""
    impressions_path, characters_path = _csv_paths(data_dir)
    dtypes = {column: str for column in HASHED_COLUMNS}
    dtypes.update({"click": "int8", "hour": "int64", "C20": str})
    impressions = pd.read_csv(impressions_path, dtype=dtypes)
    characters = pd.read_csv(
        characters_path,
        dtype={"character_id": str},
        parse_dates=["created_at"],
    )

    assert not impressions["id"].duplicated().any(), "duplicate impression ids"
    assert not characters["character_id"].duplicated().any(), "duplicate character ids"
    joined = impressions.merge(
        characters,
        on="character_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = joined["_merge"].ne("both").sum()
    assert unmatched == 0, f"{unmatched} impressions have no matching character"
    return joined.drop(columns="_merge")


def load_characters(data_dir):
    """Read only the character table with stable identifier and date types."""
    _, characters_path = _csv_paths(data_dir)
    characters = pd.read_csv(
        characters_path,
        dtype={"character_id": str},
        parse_dates=["created_at"],
    )
    assert not characters["character_id"].duplicated().any(), "duplicate character ids"
    return characters


def add_flags(df):
    """Add time, surface, sentinel, genre, and character-age columns."""
    result = df.copy()
    result["hour_of_day"] = result["hour"] % 100
    result["day"] = result["hour"] // 100
    result["is_null_device"] = result["device_id"].eq("a99f214a")

    app_placeholder = result["app_id"].eq("ecad2386")
    site_placeholder = result["site_id"].eq("85f751fd")
    assert app_placeholder.eq(~site_placeholder).all(), "site/app placeholders are not complements"
    result["is_site_side"] = app_placeholder
    result["c20_is_sentinel"] = result["C20"].eq("-1")
    result["genre"] = result["character_name"].str.split("_", n=1).str[0]

    impression_date = pd.to_datetime(
        result["day"].astype(str), format="%y%m%d", errors="raise"
    )
    created_date = pd.to_datetime(result["created_at"], errors="raise").dt.normalize()
    result["character_age_days"] = (impression_date - created_date).dt.days
    return result


def split(df, name):
    """Return one named chronological window."""
    if name not in WINDOWS:
        raise ValueError(f"Unknown window {name!r}; choose from {', '.join(WINDOWS)}")
    start, end = WINDOWS[name]
    mask = df["hour"].ge(start)
    if end is not None:
        mask &= df["hour"].lt(end)
    return df.loc[mask].copy()
