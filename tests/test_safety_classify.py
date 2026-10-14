import pandas as pd

from simula.safety_classify import fit, metrics, predict, split_by_date


def test_split_metrics_and_learnable_signal():
    words = {"sfw": "kind gardener", "suggestive": "flirty bartender", "mature": "explicit nightclub"}
    rows = [{"character_id": f"c{i}", "character_description": f"A {words[t]} number {i % 7}",
             "safety_tier": t, "creator_type": "community", "num_interactions": 100,
             "created_at": f"2014-{1 + i % 9:02d}-01"}
            for i, t in enumerate(["sfw", "suggestive", "mature"] * 40)]
    train, test = split_by_date(pd.DataFrame(rows))
    assert train["created_at"].max() <= test["created_at"].min() and len(test) == 24

    m = metrics(test["safety_tier"], predict(fit(train, train["safety_tier"]), test))
    assert m["accuracy"] == 1.0 and m["recall_mature"] == 1.0 and m["kappa"] == 1.0

    always_sfw = metrics(["sfw", "mature", "suggestive"], ["sfw"] * 3)
    assert always_sfw["kappa"] == 0.0 and always_sfw["recall_mature"] == 0.0
