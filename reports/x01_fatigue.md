# x01: is the shipped fatigue discount right, and can Jev guess the curve without data?

Branch `exp/x-fatigue-stack`, 2026-10-14. Batch X, worker A, experiment x01 (Fable E8). Code: `simula/x01_fatigue.py`.

## Question

`rank()` multiplies pCTR by 1 / (1 + 0.5 × times_seen) when the caller sends exposure counts. Is that the right shape? And can Jev propose a better curve from the assignment's wording alone, before seeing any data?

## Setup

- **Repeat count k.** For each row, k = 1 + the impressions of the same creative (C14), on the same `day`, to the same (`device_ip`, `device_model`), in **strictly earlier hours**. Rows in the same hour have no order, so they never count each other. k=1 is a first exposure. Buckets are 1, 2, 3 and 4+.
- **My definition is stricter than the orchestrator's check.** That check ordered ties inside an hour arbitrarily and found 11.4% repeats on test. With earlier hours only, repeats (k ≥ 2) are **6.4% of test rows** (8,195 of 127,406) and 6.3% of refit rows.
- **Fitted curve, the truth for Jev.** CTR at each bucket divided by CTR at k=1, on the refit window only (Oct 21–27).
- **Jev curve.** Written before Jev or I looked at any data through this code. The state is the assignment's goal-2 paragraph and the CSV schema, verbatim from `ASSIGNMENT.md`, plus one sentence defining k. Jev answers one Score per bucket. A Score becomes a ratio as the expected rung midpoint (0.375, 0.525, 0.675, 0.825, 0.95). The state was asked 3 times and averaged.
- **Shipped discount.** 1 / (1 + 0.5 × (k − 1)), uncapped, as in `simula/rank.py`.
- **Arms on test (Oct 29–30).**
  - B: `bundle/B`'s calibrated pCTR.
  - B × shipped, B × Jev curve, B × fitted curve. The raw product is the primary arm. A sensitivity arm refits B's half intercept shift on Oct 28 after the multiplier.
  - **B_sub** and **B_sub + counter:** B's recipe (early stopping on Oct 27, refit on Oct 21–27, half intercept shift on Oct 28) on the same 300,000-row random subsamples of dev-fit and refit, without and with a `prior_impressions_today` = k − 1 feature. The subsample follows the shared-machine rule, so B_sub is the fair control for the counter, not B.
- **Metrics.** Log loss and AUC on all test rows and on the repeat slice (k ≥ 2). Intervals are a paired bootstrap that resamples whole characters, 1,000 times: `bootstrap_diff` for log loss and the `rank_pairs._auc_diff` cluster bootstrap for AUC.
- **Pre-registered** in commit `26a4bad` (`prereg:`) before the Jev calls and before any test scoring.
- **Post-hoc additions** (after the red team on `e780a65`), pre-registered in commit `4a28eb4` (`prereg: x01 post-hoc re-ask`) before they ran: a Jev re-ask with k defined exactly as the code computes it, and a capped-shipped sensitivity. Both are exploratory and do not change the verdict above. See "Post-hoc: exact-k re-ask and capped shipped discount".
- **Pre-declared win for Jev:** every Jev ratio within 0.05 of the fitted one, **and** B × Jev beats B × shipped on repeat-slice log loss with the interval excluding 0.

### Jev

`jev-latest` resolved to `jev-1.13.0`. The original version of the questions is below and was not changed. A second, post-hoc version with an exact definition of k is described in its own section; both sets of answers are kept.

- Three Score questions, `ratio_k2`, `ratio_k3` and `ratio_k4`. Each says: "Consider the {2nd | 3rd | 4th or later} time the same device is shown the same ad creative (C14) on the same day. What is its click-through rate relative to the click-through rate of that device's first impression of that creative that day?"
- Rungs, low to high: "below 0.45 times the first-impression CTR", "0.45 to 0.6 …", "0.6 to 0.75 …", "0.75 to 0.9 …", "0.9 or more …".
- State: `{"assignment_goal_2_candidate_ranking": <the goal-2 text, which names "user fatigue caps">, "data_schema": <the "What you get" section>, "definition": "k counts how many times a device (device_ip plus device_model) has been shown the same ad creative (C14) so far that day, including the current impression; k=1 is the first."}`. It holds no data and no character text.
- **Wording mismatch, found by the red team.** These questions ask about the "2nd time" and impressions "so far". The code's k counts only strictly earlier hours and ties every row within an hour. So the original question and the fitted target do not define a repeat the same way. In the test window, the red team counted 11.3% of rows in a same-hour duplicate group, but only 6.4% are repeats under the code's k.
- **Cost.** 3 calls, 3 attempts, 0 errors, 0.8 s in total. 3,795 input tokens, $0.0002. The answers are in `reports/x01_fatigue_jev_answers.jsonl`. The three answers were nearly identical (k=2 rung probabilities 0.22 / 0.29 / 0.41 / 0.06 / 0.02 on the first ask).

## Results

### The curves

| k | Refit CTR (rows) | Fitted ratio (Oct 21–27) | Jev ratio | Jev − fitted | Shipped ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 18.9% (683,917) | 1 | 1 | | 1 |
| 2 | 12.8% (21,526) | 0.675 | 0.579 | −0.096 | 0.667 |
| 3 | 12.5% (7,241) | 0.663 | 0.530 | −0.133 | 0.500 |
| 4+ | 11.3% (17,001) | 0.595 | 0.473 | −0.122 | 0.400 (k=4, falling after) |

The test curve under the same definition is 0.689 / 0.638 / 0.600, close to refit. Oct 28 is noisier: 0.781 / 0.890 / 0.670.

Jev's ratios are below the fitted ones at every bucket, by 0.10–0.13. They are below the shipped curve at k=2 and above it after. As a curve, the shipped discount matches the fitted ratio at k=2 and falls below it from k=3, as Fable found.

### Test metrics

127,406 test rows; 8,195 in the repeat slice. Differences are arm − B, so negative log loss and positive AUC are better.

| Arm | Log loss, all | AUC, all | Log loss, repeat | AUC, repeat | Mean pCTR, repeat | Δ log loss all (95% CI) | Δ log loss repeat (95% CI) | Δ AUC repeat (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **B** | **0.40945** | **0.7290** | **0.31693** | **0.7368** | 0.134 | | | |
| B × shipped | 0.41260 | 0.7248 | 0.36586 | 0.6700 | 0.060 | +0.0031 (+0.0025 to +0.0038) | +0.0489 (+0.0387 to +0.0587) | −0.067 (−0.083 to −0.052) |
| B × Jev | 0.41017 | 0.7267 | 0.32820 | 0.7340 | 0.071 | +0.0007 (+0.0004 to +0.0010) | +0.0113 (+0.0063 to +0.0162) | −0.003 (−0.006 to −0.000) |
| B × fitted | 0.40966 | 0.7280 | 0.32014 | 0.7358 | 0.086 | +0.0002 (+0.0000 to +0.0004) | +0.0032 (−0.0003 to +0.0068) | −0.001 (−0.003 to +0.001) |
| B_sub | 0.41295 | 0.7211 | 0.32166 | 0.7253 | 0.138 | +0.0035 (+0.0030 to +0.0040) | +0.0047 (+0.0030 to +0.0063) | −0.012 (−0.018 to −0.005) |
| B_sub + counter | 0.41304 | 0.7207 | 0.32046 | 0.7275 | 0.133 | +0.0036 (+0.0030 to +0.0041) | +0.0035 (+0.0016 to +0.0054) | −0.009 (−0.017 to −0.002) |

The actual CTR on the repeat slice is 11.3%.

| Head to head | Δ log loss, all (95% CI) | Δ log loss, repeat (95% CI) | Δ AUC, repeat (95% CI) |
| --- | --- | --- | --- |
| **B × Jev − B × shipped (pre-declared)** | −0.0024 (−0.0029 to −0.0020) | **−0.0377 (−0.0447 to −0.0310)** | +0.064 (+0.051 to +0.077) |
| B × fitted − B × shipped | −0.0029 (−0.0035 to −0.0024) | −0.0457 (−0.0535 to −0.0382) | +0.066 (+0.052 to +0.080) |
| B_sub + counter − B_sub | +0.0001 (−0.0001 to +0.0003) | −0.0012 (−0.0020 to −0.0004) | +0.002 (−0.001 to +0.006) |

The recalibrated sensitivity arms match the raw ones to within 0.0001 log loss on every comparison. B × Jev − B × shipped on the repeat slice is −0.0376 (−0.0446 to −0.0310) after recalibration.

**Verdict for Jev: null.** The second condition holds: B × Jev beats B × shipped on the repeat slice, interval excluding 0. The first fails: no Jev ratio is within 0.05 of the fitted one. That is Fable's pre-stated null: Jev's curve is about as harsh as the shipped one. It beats the shipped discount mostly because the shipped one keeps falling past k=4: capping shipped at its k=4 factor explains about 94% of Jev's gain (post-hoc, below).

### Post-hoc: exact-k re-ask and capped shipped discount

These were added after the red team and pre-registered in `4a28eb4` before they ran. They are exploratory: the outcome of the original run was already known.

**Exact-k re-ask.** The state is the same except for one sentence: "k = 1 + the number of impressions of the same ad creative (C14) shown to the same device (device_ip plus device_model) on the same day in strictly earlier hours. Impressions in the same hour have no order: they do not count each other and share the same k. k=1 means the device saw no impression of that creative in any earlier hour that day." Each question reads "Consider impressions with {k = 2 | k = 3 | k of 4 or more}, with k as defined in the state. What is their click-through rate relative to the click-through rate of impressions with k = 1?" The rungs say "times the k = 1 CTR". It was asked 3 times: 3 calls, 0 errors, `jev-1.13.0`, 3,795 input tokens as reported by the API, $0.0002. The answers are in `reports/x01_fatigue_posthoc_jev_answers.jsonl` and the results in `reports/x01_fatigue_posthoc.json`. Only log loss intervals were computed here, not AUC.

| k | Fitted | Original Jev | Exact-k Jev | Exact-k − fitted |
| --- | --- | --- | --- | --- |
| 2 | 0.675 | 0.579 | 0.685 | +0.011 |
| 3 | 0.663 | 0.530 | 0.585 | −0.078 |
| 4+ | 0.595 | 0.473 | 0.532 | −0.063 |

The exact wording moves Jev closer at every bucket, and k=2 is now within 0.05. k=3 and 4+ still miss, so the re-ask would also be **null** under the original win rule. On test it beats the original Jev curve on the repeat slice, −0.0057 (−0.0068 to −0.0047). It beats the shipped discount there by −0.0434 (−0.0507 to −0.0365). It is still worse than B alone there by +0.0056 (+0.0015 to +0.0095), as every multiplier is.

**Capped shipped discount.** This holds the shipped factor at its k=4 value, 0.4, like the 4+ bucket of the other curves. Repeat-slice log loss:

| Arm | Repeat log loss | vs capped shipped (95% CI) |
| --- | --- | --- |
| B | 0.31693 | |
| B × fitted | 0.32014 | −0.0102 (−0.0122 to −0.0081) |
| B × exact-k Jev | 0.32248 | |
| B × original Jev | 0.32820 | −0.0021 (−0.0032 to −0.0010) |
| B × shipped, capped | 0.33035 | |
| B × shipped, uncapped | 0.36586 | +0.0355 (+0.0296 to +0.0419) |

Capping alone accounts for 0.0355 of the 0.0377 by which the original Jev curve beats shipped (94.3%) and 77.7% of the fitted curve's gain. Against the capped discount, the original Jev curve keeps a small edge. No cap makes the shipped curve match the fitted one, because it is already lower at k=3 and k=4.

### Post-hoc: B already prices most of the fatigue

This was not pre-registered. Every multiplier, even the true refit curve, makes B worse, so I checked why. On Oct 28, where the trees are out of sample, B already predicts a lower pCTR for repeats: 14.3% at k=2 against 17.1% at k=1. B's features (device model, hour, creative, app) seem to carry much of the effect without the counter. The observed-over-predicted ratio, relative to k=1, is the multiplier B still needs:

| k | Oct 28: observed / B pCTR, relative to k=1 | Test: same |
| --- | --- | --- |
| 2 | 0.93 | 0.86 |
| 3 | 1.10 | 0.88 |
| 4+ | 0.91 | 0.84 |

So the residual multiplier on top of B is about 0.85–0.93, not 0.60–0.67. The test column uses test outcomes and is shown only to describe the gap. It was not used for any choice. Computed as `ctr / mean(B pCTR)` per bucket, divided by the k=1 value.

## What it means for ranking

- **This is a probability diagnostic, not a ranking test.** `rank()` uses the discounted value only to order the candidates inside one request. Here it is scored as a click probability across rows. On that diagnostic, B × shipped costs +0.049 log loss and −0.067 AUC on repeat impressions. The discount is applied to a pCTR that already falls for repeats, so as a probability it double-counts. Most of the loss comes from the uncapped tail. This does not show whether `rank()`'s within-request choices get better or worse. That needs a logged or prospective ranking evaluation, with candidate sets and selection probabilities, before the shipped utility is changed. The exposure key here (IP plus model) is also a proxy, not the caller-supplied exposure state `rank()` reads.
- **As a probability, learning the counter beats bolting on a curve.** Inside the model, `prior_impressions_today` improves the repeat slice (−0.0012, interval excluding 0) and leaves overall log loss unchanged.
- **Jev's world-knowledge prior is not accurate enough to set the curve.** Its shape is plausible (falling, steepest at the first repeat). Its level is 0.10–0.13 too low with the original wording, and 0.06–0.08 too low at k=3 and 4+ with the exact-k wording (post-hoc). On this task, arithmetic on one week of logs wins easily.

## Identity

The README and `features.py` bar `device_ip` as identity: it is in the `never` list. This counter is keyed on (`device_ip`, `device_model`, day). It is exposure state that the shipped policy already assumes the caller can supply, but it is derived from `device_ip`. Shipping it needs a **day-scoped, device-level exposure key** that is appropriately permitted. In this export, 89.8% of test repeats are on rows with the null-device sentinel, so a `device_id` key would not reproduce it. A counter like this is not a fingerprinting workaround. If no such key is permitted, the counter does not ship. What the discount should then be is a question for a ranking evaluation; this probability diagnostic does not answer it.

## Limits

- The test window was already inspected; the orchestrator also checked this curve on test. Everything is exploratory.
- I saw Fable's verified refit ratios and the orchestrator's test curve before writing Jev's state. Jev did not. Its state has no numbers.
- The original Jev question defined a repeat by order ("2nd time"), while the code ties rows within an hour. The exact-k re-ask fixes that, but it was asked after the first outcomes were known, so it is exploratory.
- Within an hour, impressions have no order, so k undercounts repeats. A real exposure log with timestamps would give more repeats and a sharper curve.
- B_sub and B_sub + counter use 300,000-row subsamples, so both trail B by about 0.0035 log loss. Only their difference is the counter's effect.
- Log loss and AUC across rows measure the multiplier as a probability. They do not measure the within-request choice the discount actually makes. There are no logged candidate sets here, so nothing in this report is a causal CTR lift.

Full suite: 46 passed in 10.19s at 12078fb
