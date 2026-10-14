# x11: can Jev forecast which new creatives will surprise us?

Branch `exp/x-novelty`, 2026-10-14. Batch X, worker D. Design: Fable's E6 ("confidence as a surprise forecaster"). Code: `simula/x11_surprise.py`.

## Question

Exploration budget is best spent on the new creatives whose real click rate will land far from what we expect. e3 asked Jev which of two ads is better, and it could not tell. This asks a different question: from the same attribute statistics, can Jev tell which new creative will surprise the prior? If it can, its score could weight ε in `rank()`'s near-tie set. Offline, only the forecast is tested.

**Result: null.** Jev's forecast has a Spearman correlation of 0.06 with realized surprise once creative size is held fixed (95% interval −0.15 to 0.28). The best arithmetic control scores 0.08. Neither shows useful forecasting in this sample.

## Setup

- **Creatives.** Every creative (C14) absent from the refit window (Oct 21–27) with at least 100 rows on Oct 29–30. That is 87 creatives and 48,227 rows. Fable counted the same 87.
- **Target.** |realized test CTR − e4 table prior|. The prior is averaged over the creative's own test rows, because each row's cell (genre, C18, app_category) depends on its context. The median target is 0.049, the mean 0.050 and the maximum 0.161. The binomial noise at the median size (191 rows) is about 0.027, so roughly half the typical surprise is sampling noise.
- **Jev's state.** e3's attribute-statistics candidate state, read from the creative's first test row:
  - its slot;
  - each of `banner_pos`, C15, C16, C18, C19, C20 and C21 as a value with its refit impressions and smoothed CTR (m = 20);
  - the `genre_x_C18` and `app_category_x_C18` cells.

  One field is added: the refit impressions of the e4 cell, so that Jev sees what the cell-n control sees. The state has no pCTR, C14, C17, click or character description. Example, cut:

  ```json
  {"candidate": {"slot": 0, "C18": {"value": "3", "impressions": 258670, "smoothed_ctr": 0.1615},
                 "C20": {"value": "-1", "impressions": 342812, "smoothed_ctr": 0.2069},
                 "genre_x_C18": {"impressions": 25465, "smoothed_ctr": 0.156}, "...": "..."},
   "cell_genre_x_C18_x_app_category": {"impressions": 1062}}
  ```

- **Jev's question.** One Score, `uncertainty`: "This ad creative is new. Its attributes' history suggests a click-through rate. How far could its real click-through rate end up from that suggestion, in percentage points? Use the evidence in the state." The rungs are: within 1 point, 1–3, 3–6, 6–10, more than 10 points away. The forecast is the expected rung index under Jev's probabilities.
- **Controls**, from the same state. The best of the three is the bar.
  - **Attribute sd**: the standard deviation of the nine smoothed CTRs.
  - **Cell thinness**: −log(1 + e4 cell n).
  - **Rank sum** of the two.
- **Covariate.** Realized noise shrinks as a creative gets more rows, so Fable's trap note asks for size as a covariate. The headline is the partial Spearman given log(test rows): ranks are residualized on rank(log rows), then correlated.
- **Pre-declared win** (Fable). Jev's partial Spearman ≥ 0.3 and above the best control's.
- **Intervals.** 1,000 bootstrap resamples of creatives. These are tasks, not impressions.
- **Order.** The question, state, controls and win were committed (`prereg:` commit 1cb8665) before any answer was scored. After the answers arrived and before the evaluation, one descriptive field changed: the rung histogram now counts each answer's most likely rung. Jev's `score` field is continuous, so counting it directly was meaningless. No metric changed.
- **Cost.** 177 calls (3 smoke, 87 states asked twice), 0 errors, 2 s of wall time, 138,545 input tokens, **$0.0058**. `jev-latest` resolved to `jev-1.13.0`. The answers are in `reports/x11_surprise_jev_answers.jsonl`.

## Results

| Forecast | Spearman with surprise | Partial Spearman given log(test rows) |
| --- | --- | --- |
| **Jev uncertainty** | 0.082 | **0.062** (95% CI −0.146 to 0.278) |
| Attribute sd (best control) | 0.152 | 0.084 |
| Cell thinness | 0.011 | −0.008 |
| Rank sum of the two | 0.095 | 0.039 |
| Jev − best control | | −0.022 (95% CI −0.139 to +0.096) |

- **Win: no.** 0.062 is far below 0.3 and below the attribute-sd control.
- **Jev's spread is narrow.** Its expected rung runs from 1.7 to 3.3, with a median of 2.4. Its most likely rung was "3 to 6 points" for 32 creatives, "6 to 10" for 38 and "over 10" for 17. Re-asking all 87 moved the expected rung by 0.07 on average, so the answers are stable but nearly flat.
- **Size runs the other way.** Surprise rises with size: the Spearman correlation of surprise with log(test rows) is +0.40. The 16 creatives with 1,000 or more rows have a median surprise of 0.066, against 0.043 for the rest. The prior misses most on the big new creatives, where noise is smallest. That is real miscalibration of the prior, not noise, and the attribute statistics do not see it coming.

## What it means for ranking

- **Do not weight exploration by Jev's uncertainty.** This sample did not show useful forecasting, and Jev did not beat the arithmetic control (the spread of attribute CTRs). The interval, −0.146 to +0.278, rules out the 0.3 bar but allows a modest positive association.
- **Spend exploration on size instead.** The creatives the prior misses most are the large ones, the ones that will carry the most traffic. That argues for the e4 table prior giving way to B and to fast per-creative correction as volume arrives (see x10), not for a judge reading attribute tables.

## Limits

- The test window (Oct 29–30) was inspected in earlier experiments, so everything here is exploratory.
- There are 87 creatives. With this n, a true Spearman of 0.3 would usually be detectable, but small effects would not.
- Each creative's attributes are read from its first test row. C18, C19, C20 and slot can vary across a creative's rows.
- The target is a forecast-quality measure on logged impressions. Nothing here is a causal CTR lift.

Full suite: 47 passed in 9.61s at 3b809c9
