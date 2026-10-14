# x03: can Jev tell which features will help B, from their definitions alone?

Branch `exp/x-judge`, 2026-10-14. Batch X, worker B. Code: `simula/x03_triage.py`. Design: Fable E1. Pre-registration: `reports/x_judge_prereg.md`, commit `4afd4c6`.

## Question

Testing a candidate feature costs a refit. Could Jev triage the candidates first? It would read only a one-line definition, never a row or a statistic, and predict which features lower B's log loss. Its job is to allocate compute, not to produce a feature.

## Setup

- **Zoo.** There are 12 features. Six are candidates:
  - prior impressions today to the same device, for any ad and for this creative;
  - hours since the creative was first seen;
  - IP volume today;
  - `device_model` rarity;
  - the `C20 = -1` flag.

  Six are decoys:
  - `conversation_turn`;
  - turn / `session_msg_count`;
  - safety tier as an ordered number;
  - character trailing CTR;
  - `num_interactions`;
  - `session_msg_count`.

  Fable's text names 11 features but says 12, so raw `session_msg_count` is the 12th. Safety tier is redundant, because B already has it as a category.
- **Leaks.** Three features are unavailable at decision time by construction: the two session-total features and `num_interactions`.
- **Definitions.**
  - "Today" counts use strictly earlier hours of the same day. Rows in the same hour have no order.
  - Device keys are `device_ip` + `device_model`.
  - Trailing CTR uses strictly earlier days and no test-window label.
- **Fits.** Each feature gets B + f on train.py's recipe:
  - early stopping on Oct 27;
  - a refit through Oct 27 with that tree count;
  - calibration on Oct 28, with one intercept shrunk by half (bundle B's recipe);
  - test on Oct 29–30, 127,406 rows.
  - Both fitting windows are subsampled to the same 300,000 rows for every arm.
- **Realized gain.** Test log loss of B minus that of B + f. Positive means the feature helped.
- **Set arms.**
  - B + Jev-top-3.
  - B + random-3: 20 fixed draws.
  - B + oracle-3: the top 3 by realized gain, as a ceiling.
- **Run.** 70 fits took 4 min 20 s.

### Jev

`jev-latest` resolved to `jev-1.13.0`. The questions were written before any call and not changed. This is the only version.

- `expected_gain` is a Score with rungs {hurts, none, small, medium, large}. Instruction: "If this feature is added to model B, how much will it improve B's log loss on the later days? hurts = makes it worse, none = no change."
- `available_at_decision_time` is a Noul. Instruction: "Is this feature's value known at the moment the ad is chosen, before anything that happens after the impression?"
- Each card was asked 3 times. Jev's ranking is the mean Score. Its top 3, with ties in zoo order, were `character_trailing_ctr` (2.41), `prior_impressions_today_this_creative` (2.22) and `conversation_turn` (2.01).
- **Cost.** 36 calls with 0 errors. They used 23,343 input tokens, or $0.0010. The answers are in `reports/x03_feature_triage_jev_answers.jsonl`.

One example state:

```json
{"task": "Simula serves contextual ads inside AI companion apps. Every impression is tied to a character (the AI persona the user is chatting with) and the conversation state at the moment of impression. Goal: predict P(click) for each impression so candidate ads can be ranked. Model B is a LightGBM click model trained on Oct 21-27 and scored on Oct 29-30. A proposed feature is added to B and B is refit the same way; the question is how much the feature changes B's log loss on the later days.",
 "model_b_features": "request: hour_of_day, site_id, site_domain, site_category, app_id, app_domain, app_category, device_model, device_type, device_conn_type, is_null_device, is_site_side, C1; candidate ad: banner_pos, C14-C21 (anonymized creative and advertiser attributes); character: safety_tier, creator_type, genre (name prefix), character_age_days",
 "proposed_feature": {"name": "prior_impressions_today_any_ad",
                      "definition": "Number of earlier impressions served today to the same device (same device_ip and device_model), for any ad.",
                      "availability": "Counted from impressions in strictly earlier hours of the same UTC day; rows in the same hour are not counted because they have no order."}}
```

## Results

### Pre-registered run (one seed)

Gains are in units of 1e-4 log loss. The interval is the paired character bootstrap.

| Feature | Jev score | Test gain (95% CI) | Oct 27 gain |
| --- | --- | --- | --- |
| prior_impressions_today_any_ad | 1.96 | +9.54 (+7.46 to +11.61) | +3.02 |
| safety_tier_ordinal | 1.94 | +8.25 (+6.17 to +10.48) | +2.07 |
| prior_impressions_today_this_creative | 2.22 | +7.87 (+5.80 to +9.93) | +2.82 |
| num_interactions | 0.94 | +6.96 (+4.93 to +9.02) | +3.12 |
| ip_volume_today | 1.73 | +6.86 (+4.79 to +8.92) | +2.96 |
| c20_is_minus_one | 1.31 | +6.86 (+4.79 to +8.92) | +2.96 |
| conversation_turn | 2.01 | +6.86 (+4.79 to +8.92) | +2.96 |
| turn_share_of_session | 1.85 | +6.86 (+4.79 to +8.92) | +2.96 |
| session_msg_count | 1.99 | +6.86 (+4.79 to +8.92) | +2.96 |
| hours_since_creative_first_seen | 1.99 | +6.13 (+4.11 to +8.30) | +2.96 |
| device_model_rarity | 1.26 | +5.37 (+3.24 to +7.43) | +1.85 |
| character_trailing_ctr | 2.41 | +3.68 (+1.48 to +5.77) | +6.67 |

| Set | Test gain vs B (95% CI) |
| --- | --- |
| Jev-top-3 | +5.96 (+4.08 to +7.94) |
| Random-3, mean of 20 draws | +5.45 (range +2.07 to +8.83) |
| Oracle-3 | +5.78 (+3.96 to +7.84) |
| **Jev-top-3 − random-3 (loss)** | **−0.51 (−1.50 to +0.56)** |

- **Spearman(Jev score, test gain)** over the 12 features is **−0.09**. The one-sided permutation p is 0.62, and the bootstrap interval over features is −0.68 to +0.62. Against the Oct 27 gains it is +0.14.
- Jev-top-3 sits mid-pack: 10 of the 20 random sets have lower loss.

**Pre-declared win: not met.** Spearman is −0.09, against a needed 0.5. The Jev-top-3 interval against random-3 includes 0.

### Post-hoc seed check (added after the run)

The pre-registered table has a flaw I did not anticipate.

- **Identical arms.** Five arms are bit-identical: the same tree count (113) and the same loss. Every arm with an added feature beats B by 4–10 units, and even the oracle set does worse than Jev-top-3. That pattern means the gains measure something other than the features.
- **The test.** I refit B and every single-feature arm, plus Jev-top-3, under 5 LightGBM seeds. That was 70 more fits, taking 8 min 23 s. I also recorded how often the refit splits on the added feature.

| Feature | Jev score | Splits per seed | Mean gain over 5 seeds | 95% interval over seeds |
| --- | --- | --- | --- | --- |
| prior_impressions_today_any_ad | 1.96 | 11–15 | +1.76 | −6.31 to +9.83 |
| session_msg_count | 1.99 | 0 | +0.96 | −7.11 to +9.02 |
| conversation_turn | 2.01 | 0 | +0.95 | −7.07 to +8.96 |
| ip_volume_today | 1.73 | 0–1 | +0.82 | −7.18 to +8.82 |
| prior_impressions_today_this_creative | 2.22 | 6–10 | +0.82 | −6.70 to +8.34 |
| turn_share_of_session | 1.85 | 0 | +0.78 | −7.25 to +8.80 |
| num_interactions | 0.94 | 0–1 | +0.60 | −8.42 to +9.62 |
| c20_is_minus_one | 1.31 | 0–1 | +0.14 | −7.64 to +7.92 |
| hours_since_creative_first_seen | 1.99 | 2–3 | +0.09 | −6.93 to +7.11 |
| device_model_rarity | 1.26 | 4–8 | −0.41 | −6.06 to +5.25 |
| safety_tier_ordinal | 1.94 | 3–6 | −0.57 | −8.04 to +6.90 |
| character_trailing_ctr | 2.41 | 1–7 | −2.34 | −8.56 to +3.88 |

- **Seed noise is about ten times the feature effects.** B alone ranges from 0.412507 to 0.413476 across seeds, a spread of 9.7 units. Arms whose feature is never split on still "gain" about +0.9 on average. Adding any column changes LightGBM's feature-sampling and bagging draws. The character bootstrap in the first table cannot see this, so its intervals were far too narrow.
- **No feature's gain is distinguishable from 0** over 5 seeds.
- **Ranking.** The direction is still informative. Prior impressions today to the same device is the only feature split on often that also ranks first. Character trailing CTR, Jev's top pick, ranks last. Spearman(Jev, mean seed gain) is +0.13, with permutation p 0.36 and interval −0.52 to +0.69. Jev-top-3 has a mean gain of +2.03 over the 5 seeds.

### Availability

Jev marked every feature as available at decision time. That includes `session_msg_count` and turn / session (P = 0.91 each) and `num_interactions` (0.58). So it got 9 of 12 right, and all 3 errors are the 3 leaks. The same blind spot shows up in x04.

**Verdict: null.** The pre-registered test failed. The post-hoc check shows the measured truth is mostly seed noise at this subsample size, so a Spearman over 12 gains had almost no power either way. Jev did not show it can triage features. This data cannot show that it can't.

## What it means for ranking

Jev's top pick, a lagged character CTR, carries the least signal of the twelve. That is consistent with earlier findings that characters are interchangeable within a genre. The feature with real support, prior impressions to the same device today, is the fatigue signal. Jev ranked it sixth. For feature triage on this data, the bottleneck is the measurement, not the judge. A single-seed B + f comparison cannot rank features whose effects are under 1e-4. Any triage, by Jev or by a person, needs several seeds per arm, or a fixed random path, before its ranking can be scored.

## Limits

- Five seeds on a 300,000-row subsample. A full-data run with more seeds might separate the top two features from 0.
- Fable built the zoo knowing which two features work. I added the 12th feature and the "safety tier" reading.
- Click-derived features use no test-window label, but the test window was already inspected by earlier work. Everything here is exploratory, and nothing is a CTR lift.

Full suite: 48 passed, 2 warnings in 9.92s at bab7abc
