import numpy as np

from simula.x06_evidence import checklist, cmh_z, naive_z, readout


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
