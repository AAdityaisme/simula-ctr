import numpy as np
import pandas as pd

from simula.x06_evidence import checklist, cmh_z, naive_z, readout
from simula.x07_surgery import FIELDS, accept_at_detection, closure_ok, dependencies


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
