import numpy as np
import pandas as pd

from simula.x06_evidence import checklist, cmh_z, naive_z, readout
from simula.x07_surgery import FIELDS, accept_at_detection, closure_ok, dependencies, items
from simula.x08_repair import correct, inject, invariants, repair, unnecessary


def test_mix_shift_fools_naive_but_not_cmh():
    # same per-side CTR in both arms, candidate heavy on the high-CTR side
    cells = {("control", "site"): (2000, 400), ("control", "app"): (6000, 600),
             ("candidate", "site"): (6000, 1200), ("candidate", "app"): (2000, 200)}
    assert naive_z(cells) > 5
    assert abs(cmh_z(cells)) < 1e-9


def test_injected_delta_moves_each_side_by_delta():
    rng = np.random.default_rng(0)
    pools = {"site": rng.random(200_000) < 0.2, "app": rng.random(200_000) < 0.1}
    for delta in (-0.01, 0.01):
        diffs = []
        for _ in range(40):
            cells = readout(pools, delta, rng)
            diffs.append([cells[("candidate", s)][1] / cells[("candidate", s)][0]
                          - cells[("control", s)][1] / cells[("control", s)][0] for s in ("site", "app")])
        assert np.allclose(np.mean(diffs, axis=0), delta, atol=0.003)


def test_checklist_needs_request_identical_support_and_a_gap():
    s = {"ranker_accuracy": 0.55, "request_only_ranker_accuracy": 0.52,
         "request_identical_pairs": {"pairs": 12, "ranker_accuracy": 0.6}}
    assert checklist(s)
    assert not checklist({**s, "request_identical_pairs": {"pairs": 9, "ranker_accuracy": 0.6}})
    assert not checklist({**s, "request_only_ranker_accuracy": 0.56})


def test_dependencies_and_closure():
    rows = [dict(zip(FIELDS, ["0", c14, "320", "50", c17, "0", "35", c20, c21]))
            for c14, c17, c20, c21 in [("a", "p", "1", "x"), ("a", "p", "2", "x"), ("b", "p", "1", "x"), ("c", "q", "1", "y")]]
    tuples = pd.DataFrame(rows)
    deps = dependencies(tuples)
    assert ("C14", "C17") in deps and ("C17", "C21") in deps and ("C14", "C20") not in deps
    mapping = {(x, y): dict(zip(tuples[x], tuples[y])) for x, y in deps}
    assert closure_ok({**rows[0], "C14": "new", "C20": "9"}, mapping)  # unseen creative, nothing binds it
    assert not closure_ok({**rows[0], "C21": "y"}, mapping)  # a -> p -> x is broken
    assert not closure_ok({**rows[0], "C14": "new", "C21": "y"}, mapping)  # C17 p still binds C21


def test_accept_at_detection():
    score = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.3])
    real = np.array([True, True, True, False, False, False])
    assert accept_at_detection(score, real, 2 / 3) == 2 / 3  # t = 0.8 rejects 0.1 and 0.3
    assert accept_at_detection(score, real, 1.0) == 1 / 3  # only t = 0.9 rejects all broken


def test_inject_then_repair_restores_the_clean_batch():
    batch = pd.DataFrame({"character_id": ["a", "a", "b"], "safety_tier": ["sfw"] * 3, "creator_type": ["user"] * 3,
                          "genre": ["romance"] * 3, "created_at": pd.to_datetime(["2014-10-01"] * 3),
                          "character_age_days": [28] * 3, "day": [141029] * 3, "C20": ["-1", "100", "-1"],
                          "device_id": ["a99f214a", "x", "a99f214a"], "is_null_device": [True, False, True]})
    chars = pd.DataFrame({"character_id": ["z"], "safety_tier": ["mature"], "creator_type": ["studio"],
                          "character_name": ["horror_x"], "created_at": pd.to_datetime(["2014-10-20"])})
    bad = inject(batch, "both", 1.0, chars, np.random.default_rng(0))
    assert bad["C20"].tolist() == ["-1.0", "100", "-1.0"] and not bad["is_null_device"].any()
    assert set(bad["genre"]) == {"horror"} and bad["character_age_days"].eq(9).all()
    fixed = repair(repair(bad, batch, "restore_join"), batch, "restore_missing_encoding")
    pd.testing.assert_frame_equal(fixed, batch, check_dtype=False)


def test_invariants_and_correct_repair_order():
    def s(multi=0.0, unseen=0.0, null=0.8):
        return {"share_of_repeat_characters_with_several_metadata_rows": {"batch": multi, "reference": 0.0},
                "share_C20_unseen_by_model": {"batch": unseen, "reference": 0.0},
                "share_null_device": {"batch": null, "reference": 0.8}}
    assert invariants(s(), []) == "hold"
    assert invariants(s(multi=0.2, unseen=0.3), []) == "restore_join"
    assert invariants(s(multi=0.2, unseen=0.3), ["restore_join"]) == "restore_missing_encoding"
    assert invariants(s(null=0.7), []) == "restore_missing_encoding"
    assert correct("both", []) == "restore_join"
    assert correct("both", ["restore_join"]) == "restore_missing_encoding"
    assert correct("none", ["hold"]) == "hold"


def test_c14_items_share_field_originals_and_values():
    def t(c14, c17, c20):
        return dict(zip(FIELDS, ["0", c14, "320", "50", c17, "0", "35", c20, "x"]))
    refit = pd.DataFrame([t(c14, c17, c20) for c14, c17 in [("a", "p"), ("b", "q"), ("c", "p")] for c20 in "13"])
    test = pd.DataFrame([t("a", "p", "2"), t("c", "p", "2"), t("b", "q", "2"), t("n", "q", "2")])
    deps = dependencies(refit)
    mapping = {(x, y): dict(zip(refit[x], refit[y])) for x, y in deps}
    observed = set(map(tuple, test.to_numpy()))
    its = items(refit, test, pd.concat([refit, test]), deps, seed=0, n=2)
    assert [i[0] for i in its] == ["real"] * 2 + ["broken"] * 2 and {i[3] for i in its} == {"C14"}
    for kind, _, row, f, value in its:
        result = {**row, f: value}
        assert row["C14"] in "acn" and value in "abc"  # originals and new values both come from the real pairs
        assert (tuple(result.values()) in observed) == closure_ok(result, mapping) == (kind == "real")


def test_unnecessary_counts_absent_and_repeated_restores():
    assert unnecessary(["restore_missing_encoding", "restore_missing_encoding"], "none") == 2
    assert unnecessary(["restore_join", "restore_join"], "join") == 1
    assert unnecessary(["restore_join", "restore_missing_encoding"], "both") == 0
    assert unnecessary(["hold", "retrain"], "none") == 0
