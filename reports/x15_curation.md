# x15: should B train without the device-days Jev judges automated?

Branch `exp/x-training-loop`, 2026-10-14. Batch X, worker E, Fable's E4. Code: `simula/x15_curation.py` and the shared `simula/xtrain.py`. Pre-registered in commit `3eb57d1` before any model was scored.

## Question

A few devices click far more than anyone else: one device-day has 265 impressions at a 72% CTR. If those are click farms, B may learn better without them. Jev reads each heavy device-day's profile and says whether to exclude it. Does B trained without Jev's picks beat B trained without a simple rule's picks?

## Setup

- **Profiles.** Every non-sentinel (device_id, day) in the refit window (Oct 21–27) with at least 5 impressions: 371 profiles, 3,628 rows. Fable's 581 counted all ten days; only refit rows can be excluded, so only refit device-days are judged.
- **What Jev sees.** Impressions, clicks, CTR, hours active, distinct creatives, apps and sites, device_model (hashed) and site-side share.
- **Arms.**
  - **None**: no exclusion.
  - **Rule** (Fable's): impressions ≥ 20 or CTR ≥ 0.5.
  - **Isolation forest**: on the numeric profile fields, with contamination equal to the rule's flagged share, so it excludes the same number of profiles. Seed 0.
  - **Jev**: flagged when its Noul is at least 0.5.
- **Training.** Each arm drops its flagged device-days' rows from the full refit window (729,685 rows, no subsample: a fit takes seconds). It refits B's recipe for B's 183 trees with LightGBM seeds 0, 1 and 2. **Every arm is recalibrated** on Oct 28 with B's half intercept shift and scored on Oct 29–30. Row losses are averaged over the three seeds.
- **Clean slice.** Test rows whose (device_ip, device_model) has at most 2 impressions that day: 108,421 of 127,406.
- **Intervals.** The paired bootstrap that resamples whole characters (`bootstrap_diff`, 1,000 draws). It covers test-row noise, not refit noise. A post-hoc check below covers refit noise.

### Jev

`jev-latest` resolved to `jev-1.13.0`. One version, written before any outcome; not changed.

- `exclude_as_automated` is a Noul. Statement: "This device-day is automated traffic, such as a click farm or bot, and its impressions should be excluded from the click model's training data."
- **Cost.** 371 calls, 371 attempts, 0 errors, 29.7 s of summed call time, 139,809 input tokens, $0.0059. Answers: `reports/x15_jev_answers.jsonl`, keyed `profile:<n>` in the order `profiles()` returns them (sorted by device id and day), so no device id is committed. `python -m simula.x15_curation replay` maps them back to profiles locally and reruns every arm with no Jev call; `test_x15_replay_reproduces_published_losses` checks that it reproduces the losses below.

One example state, the device-day Jev scored second highest (Noul 0.67):

```json
{"impressions": 265, "clicks": 191, "ctr": 0.721, "hours_active_of_24": 14, "distinct_creatives": 4,
 "distinct_apps": 1, "distinct_sites": 1, "device_model": "cf19f7f7", "site_side_share": 1.0}
```

## Results

Oct 29–30 log loss, mean over three refit seeds. Negative differences favor the first arm.

| Arm | Profiles flagged | Refit rows dropped | All test | Clean slice |
| --- | --- | --- | --- | --- |
| **None** | 0 | 0 | **0.40944** | **0.42153** |
| Rule | 35 | 1,241 | 0.41062 | 0.42294 |
| Isolation forest | 35 | 1,136 | 0.41069 | 0.42305 |
| Jev | 5 | 509 | 0.40996 | 0.42224 |

| Comparison | Clean slice (95% CI) | All test (95% CI) |
| --- | --- | --- |
| **Jev − rule** | **−0.00071 (−0.00087 to −0.00053)** | −0.00067 (−0.00081 to −0.00052) |
| Jev − isolation forest | −0.00081 (−0.00097 to −0.00065) | −0.00073 (−0.00087 to −0.00058) |
| Jev − none | +0.00071 (+0.00047 to +0.00092) | +0.00052 (+0.00032 to +0.00072) |
| Rule − none | +0.00141 (+0.00110 to +0.00171) | +0.00118 (+0.00092 to +0.00146) |
| Isolation forest − none | +0.00152 (+0.00122 to +0.00180) | +0.00125 (+0.00099 to +0.00150) |

- **Agreement.** κ(Jev, rule) = 0.23; κ(Jev, isolation forest) = 0.23; κ(isolation forest, rule) = 0.53. Jev's Noul ranged 0.21–0.70 with median 0.29, so it flagged only 5 profiles. All five are single-site, site-side device-days with a mean CTR of 0.68 and a mean of 102 impressions.

**Pre-declared win: met, but hollow.** Jev beats the rule on the clean slice with every arm recalibrated: −0.00071, interval excluding 0. It wins only because it excludes less. Every exclusion arm is worse than excluding nothing, and the damage grows with the rows dropped. Jev's best move would have been to flag nothing.

### Post-hoc check: is this refit noise? (run after the result above)

Dropping rows changes LightGBM's bagging draws, and the character bootstrap cannot see that noise. So after the pre-registered run I refit each arm per seed, added three more no-exclusion seeds, and added a **placebo** that drops 1,241 random refit rows outside the 371 profiles, as many as the rule drops. Numbers in `reports/x15_noise.json`.

| Arm | Seeds | Clean slice, per seed |
| --- | --- | --- |
| None | 0–5 | 0.42105, 0.42200, 0.42154, 0.42147, 0.42094, 0.42165 |
| Placebo, 1,241 random rows | 0–2 | 0.42132, 0.42149, 0.42129 |
| Rule, 1,241 flagged rows | 0–2 | 0.42280, 0.42286, 0.42317 |
| Jev, 509 flagged rows | 0–2 | 0.42217, 0.42249, 0.42206 |

The placebo sits inside the no-exclusion range. Every rule and Jev seed sits above it. So the harm comes from *which* rows are dropped, not from refit noise. Jev beats the rule on each seed.

**Why.** The flagged traffic does not stop at the refit window. One site has 800 refit rows; the rule flags 512 of them, and the site's refit CTR falls from 0.56 to 0.41 without them. In test it has 94 rows at 0.80 CTR. Another site's 57 refit rows are all flagged; it has 194 test rows at 0.73 CTR. One flagged device model has 428 test rows at 0.54 CTR. Every one of those test rows falls in the clean slice: in test, the same traffic spreads over many IP and model keys with at most two impressions each. If this is a click farm, it is a stable one, and B predicts it better by learning from it.

## What it means for ranking

- Do not drop heavy device-days from B's training data on this export. Neither the rule, the isolation forest nor Jev helps. All three hurt prediction on the logged test traffic, including the clean slice.
- Whether that traffic *should* be predicted well is a policy question, not a modeling one. If it is fraud, the fix is upstream: filter it at serving and in billing, and evaluate on traffic that survives the filter. This export cannot tell fraud from enthusiasm; nothing labels it.
- Jev was the least harmful curator because it was the most conservative, not because it found better rows.

## Limits

- The test window was already inspected, so everything is exploratory. Nothing here is a causal CTR lift.
- The clean slice uses same-day counts in the test window. It is an evaluation slice, never a feature.
- The placebo and per-seed check came after the result. It is a diagnostic; it does not replace the pre-registered numbers.
- 371 profiles from one week. There is no ground truth for "automated".
- κ is between flag sets, not against truth.

Full suite: 49 passed in 131.81s at 0dd9271
