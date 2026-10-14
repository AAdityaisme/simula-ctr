# Cold start: can Jev rank two new ads from the stats of their attributes?

Branch `exp/rank-e3`, 2026-10-14. Experiment 3 of the rank-with-Jev lane. Code: `simula/rank_e3.py`, which reuses the pair harness in `simula/rank_pairs.py`.

## Question

New creatives are 42.9% of test traffic. When both ads in a pair are new, B is close to a coin (53.7% on these pairs in `rank_e12`). Can Jev pick the clicked ad from how its attribute values performed before, better than B and better than plain arithmetic on the same numbers?

## Setup

- **Pairs.** These are the `rank_e12` exact-context pairs whose two creatives (C14) both have no impressions in the refit window (Oct 21–27). That leaves 424 pairs from 327 groups and 265 characters. The count matches the spec.
- **Identical evidence.** In 154 of the 424 pairs (36%), the two ads have exactly the same attribute values. They differ only in C14 and C17, which Jev does not see. On these pairs, no attribute-based arm can tell the ads apart.
- **What Jev sees.** It gets the harness context: character genre and description, app and site category, hour and device type. For each candidate it gets `slot` and, for each of `banner_pos`, C15, C16, C18, C19, C20 and C21, the value as a string with its refit `{impressions, smoothed_ctr}`. It also gets the `genre_x_C18` and `app_category_x_C18` cells. Smoothing is the harness m-estimate, with m=20 around the refit CTR.
  - It gets no pCTR, no creative history, no C14 and no click.
  - It gets no C17 either. C17 acts like a creative id: Fable found it seen in refit on only 2.1% of unseen-C14 rows.
- **Arms.**
  - **Coin.**
  - **B**: the higher calibrated pCTR from `bundle/B`.
  - **Lookup**: `app_id × C18`, fit on refit.
  - **Count-weighted mean**: the impressions-weighted average of the candidate's nine attribute smoothed CTRs. It uses the numbers Jev sees, read from the state itself.
  - **Jev**: when it answers "tie", the pair goes to B.
  - B, lookup and the count-weighted mean score 0.5 on an exact tie.
- **Significance.**
  - The 95% intervals come from a paired bootstrap that resamples whole characters, 1,000 times (`bootstrap_diff`).
  - McNemar is exact and counts only decisive pairs, where neither arm scores 0.5. The table shows the denominator.
- **B-implied benchmark.** This is the e12 formula: the mean of p_hi(1−p_lo) / (p_hi(1−p_lo) + p_lo(1−p_hi)) over B's pCTRs. It is what B would score if its pCTRs were the true rates. It is not an upper bound for other rankers.

### Jev

`jev-latest` resolved to `jev-1.13.0`. The question and state format were written in code before any outcome by arm was seen, and were not changed. This is the only version.

- `pick` is a Choice: "Which candidate ad is the user more likely to click, judging only from how its attributes have performed before?" The options are `{"X1": "candidate X1", "X2": "candidate X2", "tie": "no real difference"}`.
- **Calls.**
  - One call for each of the 424 pairs.
  - The same 424 pairs with X1 and X2 swapped.
  - 100 identical-evidence pairs. Each is a real candidate from these pairs, duplicated under both labels and drawn with a fixed seed.
- **Cost.** 948 calls in 948 attempts with 0 errors. They took 14 s of wall time on 8 threads, with a median of 0.105 s per call. They used 1,157,641 input tokens, or $0.049 at $0.042 per million. The answers are in `reports/rank_e3_jev_answers.jsonl`. States are not committed.

One example state. The description comes from the private dataset and is omitted here; Jev saw the full text. X2 is shown short because it is identical to X1: this is one of the 154 identical-evidence pairs.

```json
{"context": {"character_genre": "sci", "character_description": "<omitted>", "app_category": "0f2161f8",
             "site_category": "50e219e0", "hour_of_day": 2, "device_type": 1},
 "candidates": {
  "X1": {"slot": 0,
         "banner_pos": {"value": "0", "impressions": 531339, "smoothed_ctr": 0.1795},
         "C15": {"value": "320", "impressions": 679433, "smoothed_ctr": 0.1733},
         "C16": {"value": "50", "impressions": 684947, "smoothed_ctr": 0.1732},
         "C18": {"value": "0", "impressions": 297662, "smoothed_ctr": 0.1753},
         "C19": {"value": "35", "impressions": 231351, "smoothed_ctr": 0.1774},
         "C20": {"value": "100176", "impressions": 5437, "smoothed_ctr": 0.1064},
         "C21": {"value": "221", "impressions": 56320, "smoothed_ctr": 0.2023},
         "genre_x_C18": {"impressions": 27320, "smoothed_ctr": 0.1653},
         "app_category_x_C18": {"impressions": 41466, "smoothed_ctr": 0.1476}},
  "X2": "<same as X1>"}}
```

## Results

Pairwise accuracy is the share of the 424 pairs where an arm picks the clicked ad.

| Arm | Accuracy % (95% CI) | vs B, pp (95% CI) | McNemar vs B: wins / losses of decisive, p | vs count-weighted mean, pp (95% CI) | McNemar vs CWM: wins / losses of decisive, p |
| --- | --- | --- | --- | --- | --- |
| B-implied benchmark | 55.9 | | | | |
| Coin | 50.0 | −3.7 (−8.6 to +1.3) | | +2.8 (−1.2 to +7.2) | |
| **B** | **53.7 (48.7 to 58.6)** | | | +6.5 (−0.1 to +13.1) | 72 / 53 of 269, p=0.11 |
| Lookup (app_id × C18) | 52.2 (50.1 to 54.5) | −1.4 (−7.1 to +3.7) | 24 / 15 of 74, p=0.20 | +5.1 (0.0 to +10.1) | 32 / 21 of 75, p=0.17 |
| Count-weighted mean | 47.2 (42.8 to 51.2) | −6.5 (−13.1 to +0.1) | 53 / 72 of 269, p=0.11 | | |
| **Jev (tie → B)** | **51.7 (46.4 to 56.5)** | **−2.0 (−6.5 to +2.3)** | 38 / 46 of 411, p=0.45 | +4.5 (−1.6 to +10.0) | 53 / 42 of 270, p=0.30 |

**Ties.** Jev answered "tie" on 213 pairs. These include all 154 identical-evidence pairs. On those 213 pairs it hands the decision to B, so it can only differ from B on the other 211. On those 211 pairs, Jev is right 47.9% of the time and B is right 51.9%.

**Identical-evidence check.** On 100 synthetic pairs with the same candidate under both labels, Jev answered "tie" every time (0% non-tie picks). The mean |p(X1) − p(X2)| was 0.004. That is the ideal behavior.

**Order check.** With X1 and X2 swapped, Jev gives the same answer on 69.3% of the 424 pairs. Here "same" means the same underlying ad, or a tie both times. It ties 213 times in each order. On the pairs where it commits to an ad both times, it picks the same ad only 61.3% of the time. So when Jev does pick, the order of the candidates drives the choice nearly as much as the evidence does.

**Pre-declared win: not met.** The win needed Jev to beat B by at least 5 points and to beat the count-weighted mean, each with an interval that excludes 0. Jev is 2.0 points *below* B, and its interval against the count-weighted mean includes 0. The test can resolve the main claim: the interval against B tops out at +2.3, so a 5-point gain is ruled out. It cannot tell whether Jev is slightly worse than B or level with it.

## What it means for ranking

- **Jev does not solve cold start.** On two new ads, it does no better than B, and its non-tie picks are a coin at best (47.9%).
- **It is well behaved but adds nothing.** It recognizes identical evidence and declines to pick. That is correct, and it is the only part of its behavior that helps. Since a tie goes to B, the result is B's answer.
- **The attribute stats carry almost no pair signal.** The count-weighted mean, built from the same numbers, scores 47.2%, below a coin. The high-count fields dominate the weights, and both ads nearly always share them. Across the 270 pairs that differ at all, `banner_pos` and C16 never differ, and C15 differs once. The differences sit mostly in C20 (192 pairs) and C21 (155), with C18 and its two cells in 75 and C19 in 57. Neither the arithmetic nor Jev turns those differences into a correct pick.
- **B is close to its own benchmark here.** B scores 53.7% against a B-implied 55.9%. A third of the pairs are identical on every attribute the rankers see. So little is left for any attribute-based ranker.
- **Keep `rank()` as it is.** For new creatives, B (with the lookup as a weaker fallback) remains the right default. A Jev call does not earn a place in the serving path.

## Limits

- The Oct 29–30 test window was already inspected while building B, so everything here is exploratory.
- A pair is two impressions in one context, not a logged auction. The user never chose between these two ads. Nothing here is a causal CTR lift.
- There are 424 pairs from 265 characters, and pairs in one group share rows. The intervals are about ±5 points. That rules out a 5-point gain over B, but it cannot separate smaller differences.
- The refit history ends on Oct 27 and B is calibrated on Oct 28, so no test click enters a state or a score.
- The question was asked once, in one wording. A different rubric might do better, but tuning it on this window would overfit the test.

Full suite: 42 passed in 8.90s at d115f6a
