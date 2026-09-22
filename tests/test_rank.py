import copy
import json
from collections import Counter

import numpy as np
import pytest

import simula.rank as rank_module
from simula.data import add_flags, load, load_characters, split
from simula.features import FEATURE_SETS, fit_encoder
from simula.rank import POLICY, rank
from simula.train import fit


@pytest.fixture(scope="module")
def fitted_smoke():
    df = add_flags(load("fixtures"))
    train = split(df, "dev_fit")
    valid = split(df, "select")
    encoder = fit_encoder(train, FEATURE_SETS["C"])
    encoder["feature_set"] = "C"
    booster, best_iteration = fit(train, valid, FEATURE_SETS["C"], encoder)
    meta = {
        "best_iteration": best_iteration,
        "calibration": {"recipe": "test", "days": 1, "shrink": 1.0, "delta": 0.0},
    }
    return (booster, encoder, meta), load_characters("fixtures")


@pytest.fixture(scope="module")
def requests():
    payloads = json.loads(open("fixtures/rank_requests.json").read())
    return {payload["id"]: payload for payload in payloads}


def by_id(result):
    return {candidate["candidate_id"]: candidate for candidate in result["candidates"]}


def test_probabilities_match_closed_form(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    result = rank(requests["shared_creative_two_slots"], bundle, characters)
    eligible = [candidate for candidate in result["candidates"] if candidate["eligible"]]
    best = min(eligible, key=lambda row: (-row["utility"], row["candidate_id"]))
    exploration = [
        candidate for candidate in eligible
        if candidate["utility"] >= best["utility"] - POLICY["score_gap"]
    ]
    assert sum(row["selection_probability"] for row in result["candidates"]) == pytest.approx(1)
    for candidate in eligible:
        expected = (1 - POLICY["epsilon"]) * (candidate is best)
        if candidate in exploration:
            expected += POLICY["epsilon"] / len(exploration)
        assert candidate["selection_probability"] == pytest.approx(expected)
        if candidate not in exploration:
            assert candidate["selection_probability"] == 0


def test_empirical_selection_matches_probability(fitted_smoke, requests, monkeypatch):
    bundle, characters = fitted_smoke
    payload = copy.deepcopy(requests["fixed_slot_with_gate"])
    score = rank_module._score
    cached = {}

    def fixed_score(rows, fitted):
        if "value" not in cached:
            cached["value"] = score(rows, fitted)
        return cached["value"]

    monkeypatch.setattr(rank_module, "_score", fixed_score)
    expected = by_id(rank(payload, bundle, characters))
    selected = Counter()
    for seed in range(2000):
        payload["seed"] = seed
        selected[rank(payload, bundle, characters)["selected_candidate_id"]] += 1
    for candidate_id, candidate in expected.items():
        assert selected[candidate_id] / 2000 == pytest.approx(
            candidate["selection_probability"], abs=0.03
        )


def test_candidate_cannot_override_request_field(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    payload = copy.deepcopy(requests["shared_creative_two_slots"])
    payload["candidates"][0]["device_type"] = 2
    with pytest.raises(ValueError, match="device_type"):
        rank(payload, bundle, characters)


def test_publisher_and_character_tier_ceilings(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    result = rank(requests["fixed_slot_with_gate"], bundle, characters)
    gated = by_id(result)["fixed-gated"]
    assert gated["eligible"] is False
    assert gated["exclusion_reason"] == "content_tier_above_ceiling"
    assert gated["rank"] is None
    assert gated["selection_probability"] == 0
    assert result["ceiling"]["effective"] == "sfw"

    payload = copy.deepcopy(requests["shared_creative_two_slots"])
    mature_id = characters.loc[characters["safety_tier"].eq("mature"), "character_id"].iloc[0]
    payload["request"]["character_id"] = mature_id
    payload["publisher"]["max_content_tier"] = "suggestive"
    payload["candidates"] = [payload["candidates"][0]]
    payload["candidates"][0]["content_tier"] = "mature"
    result = rank(payload, bundle, characters)
    assert result["candidates"][0]["eligible"] is False
    assert result["ceiling"]["effective"] == "suggestive"


def test_shared_creative_in_two_slots_is_scored_twice(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    result = rank(requests["shared_creative_two_slots"], bundle, characters)
    shared = [row for row in result["candidates"] if row["candidate_id"].startswith("shared-slot")]
    assert len({row["C14"] for row in shared}) == 1
    assert {row["banner_pos"] for row in shared} == {0, 1}
    assert all(np.isfinite(row["calibrated_pctr"]) for row in shared)
    ranks = [row["rank"] for row in result["candidates"] if row["eligible"]]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_exposure_discount_and_missing_state_fallback(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    payload = copy.deepcopy(requests["fixed_slot_with_gate"])
    present = rank(payload, bundle, characters)
    exposed = by_id(present)["fixed-high-exposed"]
    assert exposed["utility"] < exposed["calibrated_pctr"]
    assert present["degraded"]["exposure_state_unavailable"] is False

    del payload["exposure"]
    absent = rank(payload, bundle, characters)
    assert all(
        row["utility"] == row["calibrated_pctr"]
        for row in absent["candidates"] if row["eligible"]
    )
    assert absent["degraded"]["exposure_state_unavailable"] is True


def test_all_unseen_identical_uses_tie_break(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    result = rank(requests["all_unseen_identical"], bundle, characters)
    assert result["indistinguishable"] is True
    assert result["degraded"]["character_metadata_missing"] is True
    assert result["ceiling"]["effective"] == "sfw"
    assert all(row["is_unseen_C14"] is True for row in result["candidates"])
    assert result["selected_candidate_id"] == "unseen-a"


def test_negative_exposure_count_is_rejected(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    payload = copy.deepcopy(requests["fixed_slot_with_gate"])
    payload["exposure"]["counts"] = {"22163": -2}
    with pytest.raises(ValueError, match="non-negative"):
        rank(payload, bundle, characters)


def test_no_fill(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    result = rank(requests["no_fill"], bundle, characters)
    assert result["selected_candidate_id"] is None
    assert result["no_fill"] is True
    assert result["reason"] == "no_fill"


def test_malformed_payloads_are_rejected(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    base = requests["shared_creative_two_slots"]
    null_id = copy.deepcopy(base)
    null_id["candidates"][0]["candidate_id"] = None
    with pytest.raises(ValueError, match="missing candidate_id"):
        rank(null_id, bundle, characters)
    no_seed = copy.deepcopy(base)
    del no_seed["seed"]
    with pytest.raises(ValueError, match="seed"):
        rank(no_seed, bundle, characters)
    too_many = copy.deepcopy(base)
    too_many["candidates"] = [
        {**base["candidates"][0], "candidate_id": f"c{i}"} for i in range(51)
    ]
    with pytest.raises(ValueError, match="more than 50"):
        rank(too_many, bundle, characters)


def test_null_device_id_is_unavailable(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    scores = []
    for value in (None, "a99f214a", "missing"):
        payload = copy.deepcopy(requests["shared_creative_two_slots"])
        if value == "missing":
            del payload["request"]["device_id"]
        else:
            payload["request"]["device_id"] = value
        result = rank(payload, bundle, characters)
        scores.append([row["calibrated_pctr"] for row in result["candidates"]])
    assert scores[0] == scores[1] == scores[2]
    for value, expected in ((None, True), ("a99f214a", True), ("0f7c61dc", False)):
        payload = copy.deepcopy(requests["shared_creative_two_slots"])
        payload["request"]["device_id"] = value
        rows = rank_module._assemble(payload, characters, [True, True, True])
        assert rows["is_null_device"].eq(expected).all()


def test_malformed_exposure_is_rejected(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    base = requests["fixed_slot_with_gate"]
    for counts in ({"20345": "3"}, {"20345": float("nan")}, {"20345": True}):
        payload = copy.deepcopy(base)
        payload["exposure"]["counts"] = counts
        with pytest.raises(ValueError, match="finite non-negative"):
            rank(payload, bundle, characters)
    payload = copy.deepcopy(base)
    payload["exposure"] = None
    assert rank(payload, bundle, characters)["degraded"]["exposure_state_unavailable"] is True


def test_indistinguishable_requires_one_distinct_row(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    payload = copy.deepcopy(requests["all_unseen_identical"])
    payload["candidates"][2]["C21"] = 999
    assert rank(payload, bundle, characters)["indistinguishable"] is False


def test_empty_exposure_state_is_unavailable(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    payload = copy.deepcopy(requests["fixed_slot_with_gate"])
    payload["exposure"] = {}
    result = rank(payload, bundle, characters)
    assert result["degraded"]["exposure_state_unavailable"] is True
    for row in result["candidates"]:
        if row["eligible"]:
            assert row["utility"] == row["calibrated_pctr"]


def test_fatigue_is_keyed_by_creative(fitted_smoke, requests):
    bundle, characters = fitted_smoke
    payload = copy.deepcopy(requests["shared_creative_two_slots"])
    creative = str(payload["candidates"][0]["C14"])
    payload["exposure"] = {"key": "demo", "window": "24h", "counts": {creative: 2}}
    rows = {row["candidate_id"]: row for row in rank(payload, bundle, characters)["candidates"]}
    for name in ("shared-slot-0", "shared-slot-1"):
        assert rows[name]["utility"] < rows[name]["calibrated_pctr"]
    assert rows["different-creative"]["utility"] == rows["different-creative"]["calibrated_pctr"]
