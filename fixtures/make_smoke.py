from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HASHED_COLUMNS = [
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
]


def main():
    """Create a deterministic 5,000-row, ten-day smoke fixture."""
    dtypes = {column: str for column in HASHED_COLUMNS}
    dtypes.update({"click": "int8", "hour": "int64", "C20": str})
    impressions = pd.read_csv(ROOT / "data/impressions.csv", dtype=dtypes)
    impressions["_day"] = impressions["hour"] // 100
    daily_counts = impressions.groupby("_day").size()
    assert len(daily_counts) == 10 and daily_counts.ge(500).all()

    sampled = impressions.groupby("_day", group_keys=False).sample(
        n=500, random_state=20260920
    )
    sampled = sampled.drop(columns="_day").sort_values(["hour", "id"])

    characters = pd.read_csv(
        ROOT / "data/characters.csv", dtype={"character_id": str}
    )
    matching = characters[characters["character_id"].isin(sampled["character_id"])]
    assert sampled["character_id"].isin(matching["character_id"]).all()

    sampled.to_csv(ROOT / "fixtures/smoke_impressions.csv", index=False)
    matching.to_csv(ROOT / "fixtures/smoke_characters.csv", index=False)
    print(f"wrote {len(sampled)} impressions and {len(matching)} characters")


if __name__ == "__main__":
    main()
