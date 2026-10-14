# x09: can Jev choose how B rehearses brand-new creatives?

Branch `exp/x-novelty`, 2026-10-14. Batch X, worker D. Design: Astra's experiment 2 ("teach B coordinated amnesia"). Code: `simula/x09_amnesia.py`.

## Question

42.9% of test rows carry a creative (C14) that B never saw in training. On 98% of those rows C17 is new too. B never trains on a missing C14 or C17, so it has never practised that case. If we hide some identifiers for 10% of training creatives, B learns what to do when they are missing. Which identifiers should disappear together? Jev chooses from training statistics: none, C14, C14+C17 or C14+C17+C21. The question is whether Jev's choice beats the same choice made by arithmetic or by validation.

**Result: null for Jev.** Jev chose C14+C17. Chronological validation chose the same block, so Jev ties the strongest control exactly. Masking C14+C17 does help new creatives a little, but that gain belongs to masking, not to Jev, and coordinated masking does not beat independent masking.

## Setup

- **Masking.** A deterministic hash picks 10% of creatives (12.9% of refit rows). On all rows of a picked creative, the chosen fields are set to missing. That is how an unseen value reaches the model at serving.
  - *Coordinated*: one draw per creative, shared by every field in the block.
  - *Independent*: one draw per creative and field, at the same 10% rate. Independent C14 alone is the coordinated C14 scheme.
- **Recipe.** Every scheme goes through B's recipe:
  - dev fit on Oct 21–26 with early stopping on Oct 27;
  - refit on Oct 21–27 with that tree count;
  - half intercept shift fitted on Oct 28;
  - test on Oct 29–30.
- **Subsample.** Each fit uses the same 300,000-row subsample of its window. The shared-machine budget requires it. This costs B about 0.0035 log loss: subsampled B scores 0.41295 on test against 0.40945 for the shipped full-data `bundle/B`. All arms share the subsample, so the comparisons are fair.
- **Slice.** "Unseen C14" means the creative has no rows in the full refit window: 54,602 test rows. These are the same rows as in the README.
- **Controls.** Each control searches 4 options:
  - **Ordinary B**: no masking.
  - **Independent masking**: picked on Oct 27 from {none, C14, C14+C17, C14+C17+C21}.
  - **Coordinated (dependency) masking**: picked on Oct 27 from the same four blocks.
  - The pick rule is fixed in code before any run. It takes the lowest Oct 27 log loss on rows whose C14 is new to Oct 21–26 (11,611 rows), among options whose overall Oct 27 loss is within 0.0005 of no masking.
- **Pre-declared win** (Astra). Jev's scheme must beat the strongest control by ≥ 0.001 unseen-C14 log loss, with the interval below 0. Overall test loss may be no more than 0.0005 worse than ordinary B. The strongest control is the one with the lowest test unseen-C14 loss.
- **Intervals.** Paired bootstrap of per-row log-loss differences, resampling whole characters, 1,000 draws (`bootstrap_diff`).

### Jev

- **State.** Training-window statistics only (Oct 21–27), plus the unmasked dev fit's Oct 27 errors. No test row enters the state. It holds:
  - for each day, the share of rows whose C14, C17 or C21 is new that day, and the CTR of new against seen creatives;
  - the co-novelty matrix: given a new C14, a new C17 on 86% of rows and a new C21 on 5%;
  - the dependency graph: each C14 maps to exactly one C17 and one C21;
  - Oct 27 log loss by which fields are new.

  Example excerpt:

  ```json
  {"co_novelty": {"given_new_C14": {"share_new_C17": 0.8648, "share_new_C21": 0.0459}},
   "dependency": {"distinct_values": {"C14": 1585, "C17": 321, "C21": 49}, "share_C14_with_one_C17": 1.0},
   "forward_validation_oct27_by_unseen_fields": {
     "C14 new only": {"impressions": 2586, "ctr": 0.2599, "mean_prediction": 0.2721, "log_loss": 0.5513},
     "C14+C17 new": {"impressions": 8886, "ctr": 0.1832, "mean_prediction": 0.1696, "log_loss": 0.4391}}}
  ```

- **Question.** One Choice, `block`: "A click model is retrained with some creatives hidden: for 10% of training creatives, the chosen identifiers are set to missing on all of their rows, the way identifiers never seen in training are missing at serving. Which identifiers should be hidden together, so the model learns to score brand-new creatives without hurting the others?" The options are none, C14, C14+C17 and C14+C17+C21.
- **Pick.** Jev's pick is the option with the highest mean probability over 21 calls: the canonical order plus 20 calls with the options shuffled.
- **Answers.** C14+C17 had a mean probability of 0.52 and C14+C17+C21 0.36. 19 of 21 calls chose C14+C17 and 2 chose C14+C17+C21, so 9.5% of calls differed from the pick.
- **Cost.** 24 calls (3 smoke, 21 counted), 0 errors, under 1 s of wall time, 37,392 input tokens, **$0.0016**. `jev-latest` resolved to `jev-1.13.0`. The answers are in `reports/x09_amnesia_jev_answers.jsonl`.
- **Order.** The question, state and pick rule were committed (`prereg:` commit 3fcf750) before any fit was scored. They did not change.

## Results

Test log loss on Oct 29–30, with each scheme's difference from ordinary B (negative = better):

| Scheme | Trees | All | Unseen C14 | Unseen vs ordinary B (95% CI) | All vs ordinary B (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Ordinary B (no mask) | 145 | 0.412949 | 0.377288 | | |
| C14 | 122 | 0.412998 | 0.377517 | +0.00023 (−0.00006, +0.00051) | +0.00005 (−0.00015, +0.00024) |
| **C14+C17 coordinated: Jev's and validation's pick** | 128 | **0.412366** | 0.376285 | **−0.00100 (−0.00146, −0.00054)** | −0.00058 (−0.00085, −0.00035) |
| C14+C17+C21 coordinated | 122 | 0.413755 | 0.379110 | +0.00182 (+0.00130, +0.00234) | +0.00081 (+0.00053, +0.00108) |
| C14+C17 independent | 124 | 0.412415 | **0.376214** | −0.00107 (−0.00140, −0.00074) | −0.00053 (−0.00074, −0.00035) |
| C14+C17+C21 independent: validation's pick | 140 | 0.413429 | 0.378673 | +0.00139 (+0.00093, +0.00183) | +0.00048 (+0.00025, +0.00071) |

| Gate | Value |
| --- | --- |
| Jev's scheme | coordinated C14+C17 |
| Strongest control | coordinated C14+C17 (the coordinated validation pick) |
| Jev vs strongest, unseen C14 | 0.00000 (same model) |
| Jev vs ordinary B, all rows | −0.00058 (−0.00085, −0.00035) |
| **Win** | **no**: Jev picked what validation picked |

**Validation was nearly blind between the C21 options.** On Oct 27, independent C14+C17 and C14+C17+C21 differed by 0.00002 on new-C14 rows, and validation took C14+C17+C21. On test, adding C21 cost 0.002–0.003. Jev put 0.36 on the C21 block too. Neither source knew that hiding C21 removes a field that stays seen on 72% of unseen-creative test rows.

**Post hoc: other masking draws.** This check was added after the first run. The first draw rests on one set of masked creatives, so three more hash draws were fitted for both C14+C17 schemes. Six fits:

| Draw | Coordinated C14+C17, unseen vs ordinary B | Independent C14+C17, unseen vs ordinary B |
| --- | --- | --- |
| 0 (pre-registered) | −0.00100 (−0.00146, −0.00054) | −0.00107 (−0.00140, −0.00074) |
| 1 | −0.00021 (−0.00058, +0.00016) | +0.00003 (−0.00027, +0.00033) |
| 2 | −0.00038 (−0.00074, −0.00003) | −0.00082 (−0.00119, −0.00049) |
| 3 | −0.00006 (−0.00045, +0.00030) | −0.00087 (−0.00117, −0.00058) |
| Mean of 4 | −0.00041 | −0.00069 |

The gain is real on average but depends on which creatives are masked. The first coordinated draw was the luckiest of four. Independent masking matches or beats coordinated masking. These intervals resample characters; they do not include the masking-draw variance, which this table shows is as large as the effect.

## What it means for ranking

- **Jev adds nothing here.** The right block follows from the co-novelty matrix: C17 is new on 86% of new-creative rows and C21 on 5%. Both Jev and Oct 27 validation found C14+C17, so Jev is a costlier way to read that matrix.
- **Masking C14+C17 is a cheap training change that may help B on new creatives.** It lowered unseen-creative log loss by 0.0004–0.0007 on average over four draws, and overall loss fell too. It changes no serving code. Before shipping it, it should be checked on a fresh window with several draws averaged, or with a larger mask rate. That is not a Jev result.
- **Do not hide C21.** It is a stable attribute, seen on most new-creative rows, and hiding it costs 0.002–0.003.

## Limits

- The test window (Oct 29–30) was inspected in earlier experiments, so everything here is exploratory.
- There is one masking rate (10%) and one subsample. Each fit used 300,000 rows, not the full window.
- The Jev arm has one decision. Jev can only tie the validation pick or differ from it, so this is a test of whether its judgment adds anything to validation, not a test of a policy.
- These are log-loss changes on logged impressions. Nothing here is a causal CTR lift.

Full suite: <filled at the end>
