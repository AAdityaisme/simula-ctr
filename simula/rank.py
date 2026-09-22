import numpy as np
import pandas as pd

from simula.data import add_flags
from simula.features import CONTRACT, novelty_flags, transform
from simula.train import apply_calibration


POLICY = dict(
    version="policy-v1", epsilon=0.05, score_gap=0.01, fatigue_k=0.5,
    tie_break="lowest candidate_id", max_candidates=50,
)
TIERS = ("sfw", "suggestive", "mature")
REQUEST_FIELDS = [
    "hour", "character_id", "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category", "device_id", "device_ip",
    "device_model", "device_type", "device_conn_type", "C1",
]
OPTIONAL_REQUEST_FIELDS = ["conversation_turn", "session_msg_count"]
CANDIDATE_FIELDS = ["candidate_id", "content_tier"] + CONTRACT["candidate"]


def validate(payload, policy=POLICY):
    """Validate ownership, identity, tier, and surface invariants."""
    request = payload.get("request", {})
    for field in REQUEST_FIELDS:
        if field not in request and field != "device_id":
            raise ValueError(f"missing request field: {field}")
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError("candidate list is empty")
    if len(candidates) > policy["max_candidates"]:
        raise ValueError(f"more than {policy['max_candidates']} candidates")
    if not isinstance(payload.get("seed"), int):
        raise ValueError("missing integer seed")
    request_owned = set(REQUEST_FIELDS + OPTIONAL_REQUEST_FIELDS)
    required = ("candidate_id", "banner_pos", "C14", "content_tier")
    ids = []
    for index, candidate in enumerate(candidates):
        conflicts = sorted(request_owned & set(candidate))
        if conflicts:
            raise ValueError(f"candidate {index} carries request field: {conflicts[0]}")
        for field in required:
            if candidate.get(field) is None:
                raise ValueError(f"candidate {index} missing {field}")
        if not isinstance(candidate["candidate_id"], str):
            raise ValueError(f"candidate {index} candidate_id must be a string")
        if candidate["content_tier"] not in TIERS:
            raise ValueError(f"unknown tier: {candidate['content_tier']}")
        ids.append(candidate["candidate_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate_id")
    publisher_tier = payload.get("publisher", {}).get("max_content_tier")
    if publisher_tier not in TIERS:
        raise ValueError(f"unknown tier: {publisher_tier}")
    counts = payload.get("exposure", {}).get("counts") or {}
    if any(count < 0 for count in counts.values()):
        raise ValueError("exposure counts must be non-negative")
    app_placeholder = request["app_id"] == "ecad2386"
    site_placeholder = request["site_id"] == "85f751fd"
    if app_placeholder == site_placeholder:
        raise ValueError("request must have exactly one site/app placeholder")


def _eligibility(payload, characters):
    request = payload["request"]
    publisher = payload["publisher"]["max_content_tier"]
    matched = characters.loc[characters["character_id"].eq(request["character_id"])]
    character = None if matched.empty else matched.iloc[0]["safety_tier"]
    if character is not None and character not in TIERS:
        raise ValueError(f"unknown tier: {character}")
    # unknown character metadata fails closed: strictest tier until it lands
    ceiling = min(TIERS.index(publisher), TIERS.index(character) if character is not None else 0)
    effective = TIERS[ceiling]
    eligible = [TIERS.index(row["content_tier"]) <= ceiling for row in payload["candidates"]]
    return eligible, {
        "publisher": publisher,
        "character": character,
        "effective": effective,
    }, matched.empty


def _assemble(payload, characters, eligible):
    request = {field: payload["request"].get(field) for field in REQUEST_FIELDS}
    records = []
    for candidate, keep in zip(payload["candidates"], eligible):
        if keep:
            records.append({**request, **{field: candidate.get(field) for field in CANDIDATE_FIELDS}})
    rows = pd.DataFrame(records)
    rows = rows.merge(characters, on="character_id", how="left", validate="many_to_one")
    rows["character_name"] = rows["character_name"].astype("string")
    return add_flags(rows)


def _score(rows, bundle):
    booster, encoder, meta = bundle
    model_rows = transform(rows, encoder)
    raw = np.asarray(booster.predict(model_rows), dtype=float)
    calibrated = apply_calibration(raw, meta["calibration"]["delta"])
    flags = novelty_flags(rows, encoder)
    indistinguishable = len(rows) > 1 and len(model_rows.drop_duplicates()) == 1
    return raw, calibrated, flags, indistinguishable


def _exposure_counts(payload):
    counts = payload.get("exposure", {}).get("counts")
    return counts if isinstance(counts, dict) else None


def _utilities(payload, creatives, calibrated, policy):
    counts = _exposure_counts(payload)
    if counts is None:
        return np.asarray(calibrated, dtype=float)
    # keyed by creative, not candidate: one creative in two slots shares its history
    fatigue = np.array([counts.get(str(creative), 0) for creative in creatives], dtype=float)
    return calibrated / (1 + policy["fatigue_k"] * fatigue)


def _select(ids, utilities, seed, policy):
    order = sorted(range(len(ids)), key=lambda index: (-utilities[index], str(ids[index])))
    best = order[0]
    exploration = [
        index for index in order
        if utilities[index] >= utilities[best] - policy["score_gap"]
    ]
    probabilities = np.zeros(len(ids), dtype=float)
    probabilities[best] += 1 - policy["epsilon"]
    probabilities[exploration] += policy["epsilon"] / len(exploration)
    rng = np.random.default_rng(seed)
    selected = best if rng.random() < 1 - policy["epsilon"] else int(rng.choice(exploration))
    assert np.isclose(probabilities.sum(), 1.0)
    return order, set(exploration), probabilities, selected


def _excluded(candidate):
    return {
        "candidate_id": candidate["candidate_id"], "rank": None, "eligible": False,
        "exclusion_reason": "content_tier_above_ceiling", "banner_pos": candidate["banner_pos"],
        "C14": candidate["C14"], "raw_pctr": None, "calibrated_pctr": None,
        "utility": None, "in_exploration_set": False, "selection_probability": 0.0,
        "is_unseen_C14": None, "is_rare_C14": None,
        "is_unseen_character_id": None, "is_rare_character_id": None,
    }


def rank(payload, bundle, characters, policy=POLICY):
    """Rank and select a bounded candidate set with eligibility and exploration."""
    validate(payload, policy)
    eligible, ceiling, missing_character = _eligibility(payload, characters)
    kept = [index for index, value in enumerate(eligible) if value]
    candidate_rows = []
    selected_id = None
    indistinguishable = False
    if kept:
        rows = _assemble(payload, characters, eligible)
        raw, calibrated, flags, indistinguishable = _score(rows, bundle)
        ids = [payload["candidates"][index]["candidate_id"] for index in kept]
        creatives = [payload["candidates"][index]["C14"] for index in kept]
        utilities = _utilities(payload, creatives, calibrated, policy)
        order, exploration, probabilities, selected = _select(
            ids, utilities, payload["seed"], policy
        )
        selected_id = ids[selected]
        for rank_index, scored_index in enumerate(order, start=1):
            original = kept[scored_index]
            candidate = payload["candidates"][original]
            candidate_rows.append({
                "candidate_id": ids[scored_index], "rank": rank_index, "eligible": True,
                "exclusion_reason": None, "banner_pos": candidate["banner_pos"], "C14": candidate["C14"],
                "raw_pctr": float(raw[scored_index]), "calibrated_pctr": float(calibrated[scored_index]),
                "utility": float(utilities[scored_index]),
                "in_exploration_set": scored_index in exploration,
                "selection_probability": float(probabilities[scored_index]),
                **{column: bool(flags.iloc[scored_index][column]) for column in flags.columns},
            })
    candidate_rows.extend(
        _excluded(candidate) for candidate, keep in zip(payload["candidates"], eligible) if not keep
    )
    return {
        "selected_candidate_id": selected_id, "no_fill": not kept,
        "reason": "no_fill" if not kept else None, "indistinguishable": indistinguishable,
        "ceiling": ceiling,
        "degraded": {
            "exposure_state_unavailable": _exposure_counts(payload) is None,
            "character_metadata_missing": missing_character,
        },
        "versions": {
            "feature_set": bundle[1]["feature_set"],
            "calibration": bundle[2]["calibration"]["recipe"], "policy": policy["version"],
        },
        "echo": {field: payload["request"].get(field) for field in OPTIONAL_REQUEST_FIELDS},
        "candidates": candidate_rows,
    }
