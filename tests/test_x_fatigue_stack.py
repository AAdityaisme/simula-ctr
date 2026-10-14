import numpy as np
import pandas as pd

from simula.x01_fatigue import curve_multiplier, jev_curve, repeat_count, shipped_multiplier


def test_repeat_count_counts_only_earlier_hours_of_the_same_creative_device_and_day():
    rows = [  # day, hour, ip, model, C14
        (141029, 14102901, "ip", "m", "c"), (141029, 14102901, "ip", "m", "c"),  # same hour: no order, both k=1
        (141029, 14102903, "ip", "m", "c"), (141029, 14102905, "ip", "m", "c"),  # k=3, k=4
        (141029, 14102905, "ip", "m", "other"), (141029, 14102905, "ip", "m2", "c"),  # new creative / device
        (141030, 14103001, "ip", "m", "c"),  # new day
    ]
    df = pd.DataFrame(rows, columns=["day", "hour", "device_ip", "device_model", "C14"], index=list("abcdefg"))
    assert repeat_count(df).to_dict() == dict(zip("abcdefg", [1, 1, 3, 4, 1, 1, 1]))


def test_multipliers():
    k = np.array([1, 2, 3, 4, 7])
    assert np.allclose(shipped_multiplier(k), [1, 2 / 3, 0.5, 0.4, 0.25])
    assert np.allclose(curve_multiplier(k, {2: 0.8, 3: 0.7, 4: 0.6}), [1, 0.8, 0.7, 0.6, 0.6])


def test_jev_curve_is_the_mean_expected_rung_midpoint():
    def ask(p):
        return {"answers": {f"ratio_k{k}": {"probabilities": p} for k in (2, 3, 4)}}

    curve = jev_curve([ask({"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}),
                       ask({"0": 0.0, "1": 0.0, "2": 0.5, "3": 0.0, "4": 0.5})])
    assert np.isclose(curve[2], (0.825 + (0.675 + 0.95) / 2) / 2)


def test_b_pair_prob_and_unpenalized_fit():
    from simula.x02_stack import b_pair_prob, fit

    assert b_pair_prob(0.2, 0.2) == 0.5
    assert np.isclose(b_pair_prob(0.3, 0.1) + b_pair_prob(0.1, 0.3), 1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 1))
    y = (rng.random(20_000) < 1 / (1 + np.exp(-(0.5 + 2 * X[:, 0])))).astype(int)
    _, coef = fit(X, y)
    assert abs(coef["coef"][1] - 2) < 4 * coef["se"][1] and 0 < coef["se"][1] < 0.1
