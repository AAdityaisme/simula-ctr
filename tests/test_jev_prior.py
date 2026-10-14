import numpy as np
import pandas as pd

from simula.jev_prior import blend, cell_state, count_tables, jev_prior, table_prior


def _refit():
    # genre x C18 ("horror", 1) is 50% CTR over 200 rows; the thin cell has 2 rows, both clicks
    rows = [("horror", 1, "app_a", i % 2) for i in range(200)] + [("horror", 1, "app_b", 1)] * 2
    rows += [("romance", 0, "app_a", int(i % 10 == 0)) for i in range(800)]
    return pd.DataFrame(rows, columns=["genre", "C18", "app_category", "click"])


def test_thin_cell_shrinks_toward_parent():
    tables = count_tables(_refit())
    thin = table_prior(("horror", 1, "app_b"), tables)
    assert 0.5 < thin < 0.6  # raw cell CTR is 1.0; parent genre x C18 is ~0.5
    thick = table_prior(("horror", 1, "app_a"), tables)
    assert abs(thick - 0.5) < 0.01
    state = cell_state(("horror", 1, "app_b"), tables)
    assert state["cell_history"] == {"impressions": 2, "clicks": 2}
    assert state["neighbours"]["genre_x_C18"] == {"impressions": 202, "clicks": 102}


def test_blend_ends():
    p_b, prior = np.array([0.1, 0.1, 0.1]), np.array([0.3, 0.3, np.nan])
    fresh = blend(p_b, prior, np.zeros(3), k=50, b=0.5)
    assert np.allclose(fresh, [0.2, 0.2, 0.1])  # w=0 -> half-way mix; missing cell -> pB
    old = blend(p_b, prior, np.full(3, 1e9), k=50, b=0.5)
    assert np.allclose(old, p_b)


def test_jev_prior_expected_midpoint():
    answer = {"answers": {"ctr": {"probabilities": {"0": 0.5, "4": 0.5}}}}
    assert np.isclose(jev_prior(answer), (0.025 + 0.30) / 2)
