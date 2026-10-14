import numpy as np
import pandas as pd

from simula.jev_ctr import bootstrap_diff, text_slots


def test_text_slots_split_frame_and_traits():
    slots = text_slots(pd.Series([
        "A shy sleuth haunted by a case, witty and patient.",
        "An artistic sleuth haunted by a case, shy and patient.",
        "A witty ghost, artistic at night.",
    ]))
    assert slots["frame"].tolist()[:2] == ["A _ sleuth haunted by a case, _ and patient."] * 2
    assert slots[["trait_a", "trait_b"]].values.tolist() == [["shy", "witty"], ["artistic", "shy"], ["witty", "artistic"]]


def test_bootstrap_diff_sign_and_zero():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(200), 50)
    base = rng.random(groups.size)
    diff, lo, hi = bootstrap_diff(groups, base, base)
    assert diff == lo == hi == 0.0
    diff, lo, hi = bootstrap_diff(groups, base + 0.01, base)
    assert np.isclose(diff, 0.01) and lo > 0
