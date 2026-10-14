import numpy as np
import pandas as pd

from simula.x09_amnesia import FIELDS, mask


def _creatives(n=4000):
    c14 = np.repeat(np.arange(n), 3)
    return pd.DataFrame({"C14": c14, "C17": c14 // 4, "C21": c14 % 7, "click": 0})


def test_coordinated_mask_hides_the_block_together_per_creative():
    rows = _creatives()
    out = mask(rows, FIELDS, coordinated=True)
    hidden = out[FIELDS].isna()
    assert hidden.nunique(axis=1).eq(1).all()  # all three fields hidden together or not at all
    per_creative = hidden["C14"].groupby(rows["C14"]).agg(["min", "max"])
    assert per_creative["min"].eq(per_creative["max"]).all()  # every row of a creative shares its draw
    assert 0.08 < per_creative["max"].mean() < 0.12
    assert out.loc[~hidden["C14"], "C14"].eq(rows.loc[~hidden["C14"], "C14"].astype(str)).all()


def test_independent_mask_draws_each_field_separately():
    rows = _creatives()
    hidden = mask(rows, FIELDS, coordinated=False)[FIELDS].isna()
    assert hidden.nunique(axis=1).gt(1).mean() > 0.1
    assert hidden["C14"].equals(mask(rows, ["C14"], coordinated=True)["C14"].isna())  # same C14 draw in both modes
    assert mask(rows, [], coordinated=True).equals(rows)
