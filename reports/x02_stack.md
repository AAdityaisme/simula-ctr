# x02: does a learned stack of B and Jev beat B's own stack on ad pairs?

Branch `exp/x-fatigue-stack`, 2026-10-14. Batch X, worker A, experiment x02 (the pending "experiment 5"). Code: `simula/x02_stack.py`, which reuses the pair harness in `simula/rank_pairs.py`.

## Question

In `rank_e12`, Jev picked the clicked ad no better than B. That put Jev *in place of* B. Here Jev sits *beside* B. A logistic stacker gets B's pCTR gap plus Jev's probability. Does Jev's probability add anything that B's numbers do not already carry?

## Setup

- **Fit day: Oct 28** (`split(df, "calibrate")`). Pairs are built exactly as `rank_pairs.build_pairs` builds them on test: a clicked and an unclicked impression with different creatives (C14) in one exact context (hour, character, app, site, device type). That gives 2,668 pairs from 2,028 groups and 907 characters. X1 is the clicked ad in 49.3% of them.
- **Test: the 1,735 e12 pairs** (Oct 29–30, 710 characters). Jev's test answers are the committed e12 `with:` answers in `reports/rank_e12_jev_answers.jsonl`. No new test calls were made.
- **Per row.** B's calibrated pCTR from `bundle/B`, and refit-window (Oct 21–27) history from `rank_pairs.add_history`. "Seen" means the creative has refit impressions. The fit day and the test day both use refit history, so neither day's own clicks reach a state.
- **Stackers.** Unpenalized logistic regressions, fit on Oct 28 pairs only, predicting "X1 is the clicked one":
  - **S0:** [logit pCTR_X1 − logit pCTR_X2, |pCTR_X1 − pCTR_X2|, X1 seen, X2 seen];
  - **S1:** S0 plus Jev's P(X1) − 0.5, with P(X1) normalized over X1 and X2.
- **B alone** is the pair probability p1(1−p2) / (p1(1−p2) + p2(1−p1)), treating B's pCTRs as true rates.
- **Metrics.** Pair log loss and accuracy on the test pairs. A prediction of exactly 0.5 scores 0.5. Intervals are a paired bootstrap that resamples whole characters, 1,000 times (`bootstrap_diff`). Coefficient standard errors come from the inverse Fisher information.
- **Pre-registered** in commit `5cba215` (`prereg:`) before the fit-day calls and before any test scoring. One descriptive check was added after: how often Jev's direction agrees with B's. It changes no arm.
- **Pre-declared win:** S1 beats S0 on test pair log loss, with the interval excluding 0.

### Jev

`jev-latest` resolved to `jev-1.13.0`. The question and state are the e12 ones, reused unchanged: `rank_pairs.PICK` and `rank_pairs.state(..., with_pctr=True)`.

- `pick`, a Choice: "Which candidate ad is the user more likely to click? Use the evidence in the state." Options `{"X1": "candidate X1", "X2": "candidate X2"}`.
- **Calls.** A 3-pair smoke test, then one call per Oct 28 pair: 2,668 calls in 2,668 attempts, 0 errors, 27 s of wall time on 8 threads, median 0.074 s per call. They used 2,299,574 input tokens, or $0.097 at $0.042 per million (the smoke test added 2,590 tokens). The answers are in `reports/x02_stack_jev_answers.jsonl`. States are not committed.

One fit-day example state. The description comes from the private dataset and is omitted; Jev saw the full text.

```json
{"context": {"character_genre": "horror", "character_description": "<omitted>", "app_category": "07d7df22",
             "site_category": "28905ebd", "hour_of_day": 0, "device_type": 1},
 "candidates": {
  "X1": {"slot": 0, "C15": "320", "C16": "50", "C17": "1722", "C18": "0", "C19": "35", "C20": "-1", "C21": "79",
         "creative_history": {"impressions": 11828, "clicks": 2517, "smoothed_ctr": 0.2128},
         "genre_x_C18": {"impressions": 25965, "smoothed_ctr": 0.2013},
         "app_category_x_C18": {"impressions": 238185, "smoothed_ctr": 0.187}, "model_pctr": 0.2657},
  "X2": {"slot": 0, "C15": "320", "C16": "50", "C17": "1722", "C18": "0", "C19": "35", "C20": "-1", "C21": "79",
         "creative_history": {"impressions": 11837, "clicks": 2505, "smoothed_ctr": 0.2116},
         "genre_x_C18": {"impressions": 25965, "smoothed_ctr": 0.2013},
         "app_category_x_C18": {"impressions": 238185, "smoothed_ctr": 0.187}, "model_pctr": 0.2383}}}
```

## Results

Test pairs (1,735). Lower log loss is better.

| Arm | Pair log loss (95% CI) | Accuracy % (95% CI) |
| --- | --- | --- |
| B alone | 0.6824 (0.6723 to 0.6918) | 55.2 (52.8 to 57.7) |
| S0: B's numbers, stacked | 0.6789 (0.6720 to 0.6857) | 55.2 (52.7 to 57.9) |
| **S1: S0 + Jev** | **0.6791 (0.6721 to 0.6860)** | **55.5 (53.0 to 58.1)** |
| Jev alone (its P(X1)) | 2.7041 (2.3531 to 3.0469) | 54.5 (51.9 to 57.0) |

| Comparison | Log loss diff (95% CI) | Accuracy diff, pp (95% CI) |
| --- | --- | --- |
| **S1 − S0 (the pre-declared test)** | **+0.00016 (−0.00010 to +0.00039)** | +0.29 (−0.34 to +0.88) |
| S0 − B | −0.0035 (−0.0080 to +0.0009) | 0.00 (−1.98 to +1.90) |
| S1 − B | −0.0033 (−0.0079 to +0.0012) | +0.29 (−1.54 to +2.12) |
| Jev alone − B | +2.02 (+1.68 to +2.36) | −0.75 (−2.23 to +0.77) |

Fitted S1 coefficients (standard error):

| Term | S0 | S1 |
| --- | --- | --- |
| intercept | 0.074 (0.079) | 0.075 (0.079) |
| logit pCTR gap | 0.585 (0.103) | 0.564 (0.133) |
| \|pCTR gap\| | 0.000 (0.854) | 0.001 (0.854) |
| X1 seen | 0.094 (0.094) | 0.087 (0.097) |
| X2 seen | −0.265 (0.094) | −0.260 (0.097) |
| **Jev P(X1) − 0.5** | | **0.033 (0.133)** |

**Verdict: null.** S1 does not beat S0. The interval for S1 − S0 includes 0 and its point estimate is slightly worse. The Jev coefficient is 0.033 with a standard error of 0.133: indistinguishable from 0.

Why: with pCTR in its state, Jev mostly repeats B. Its direction agrees with B's on 86.7% of fit-day pairs and 87.8% of test pairs (pairs where Jev is not exactly 0.5). Once the stacker has B's gap, Jev's probability carries almost nothing new. On its own Jev is also badly overconfident: it often answers 0.99 or 1.0, so its raw log loss is 2.70, four times B's.

## What it means for ranking

Stacking Jev's pick on top of B does not help, at least not when Jev reads B's pCTR. The stacker's real finding is about B: its pair odds are too sharp. The fitted weight on the logit gap is 0.585, not 1, so a shrunk B scores better log loss than raw B (−0.0035, interval just crossing 0). That is a calibration point about B on near-identical pairs, not a reason to use Jev.

## Limits

- The test window was already inspected in earlier experiments. Everything here is exploratory.
- B's calibration intercept was fit on Oct 28, so B's pCTR on the fit day is in-sample for that intercept. The trees were fit on Oct 21–27 and are out of sample on Oct 28.
- The X1/X2 order is random, so the "X1 seen" and "X2 seen" weights should be equal and opposite in expectation. They are not (0.09 vs −0.26), which is a sign of noise in 2,668 pairs. A symmetric parameterization would be cleaner; it was not pre-registered, so it is not reported as a result.
- Pairs are two impressions in one context, not a logged auction. Neither stacker's accuracy is a causal CTR lift.
- Jev's test answers come from the e12 run, which counted successful attempts only.

Full suite: 46 passed in 10.19s at 12078fb
