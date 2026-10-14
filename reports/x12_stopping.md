# x12: can Jev pick B's tree count from the validation curve?

Branch `exp/x-training-loop`, 2026-10-14. Batch X, worker E, Fable's E3. Code: `simula/x12_stopping.py` and the shared `simula/xtrain.py`. Pre-registered in commit `ef66a2f` before any test outcome was computed.

## Question

Early stopping picks the tree count with the lowest Oct 27 loss (argmin). The serving window drifts away from Oct 27. Can Jev read the curve and the drift and pick a tree count that does better on Oct 29–30?

## Setup

- **Curves.** There are 30 curves: 10 seeds × feature sets A, B and C. Each seed draws 300,000 of the 642,897 Oct 21–26 rows and fits 400 trees with B's params and its own LightGBM seed. It records train and Oct 27 loss per tree. The 300,000-row cap is the batch's compute rule, so these models are a little worse than the full-data bundle.
- **Options.** All come from the Oct 27 curve:
  - argmin;
  - argmin × 0.8, × 0.9, × 1.1 and × 1.2;
  - one_se: the fewest trees within one standard error of the minimum (SE of the per-row loss at argmin);
  - plateau_start: the fewest trees after which the next 50 trees improve Oct 27 loss by less than 0.0001;
  - max: 400.
- **Refit.** The chosen count sets the trees of a refit on 300,000 Oct 21–27 rows with the same seed. Each refit is recalibrated on Oct 28 with B's half intercept shift and scored on Oct 29–30. One 400-tree refit per curve, truncated to each option, gives every option's test loss from the same trees.
- **Arms.** Jev's pick; the controls argmin, one_se and 0.8 × argmin; the other options for reference.
- **Intervals.** Each arm's row loss is averaged over the 30 curves. The 95% interval is the paired bootstrap that resamples whole characters (`bootstrap_diff`, 1,000 draws). A second interval resamples the 30 curves (2,000 draws).

### Jev

`jev-latest` resolved to `jev-1.13.0`. One version of the question, written before any test outcome; it was not changed.

- `tree_count` is a Choice. Instructions: "A click model was fit on Oct 21-26 and validated on Oct 27. It will now be refit on Oct 21-27 and serve Oct 29-30, after an intercept recalibration on Oct 28. The drift table shows how traffic moved between windows. Which tree count should the refit use to have the lowest log loss on the serving window?"
- Each option's text names its rule and its tree count, for example `"argmin": "the tree count with the lowest Oct 27 validation loss (93 trees)"`.
- Each curve was asked twice, once with the options reversed, to check order stability.
- **Cost.** 60 calls, 60 attempts, 0 errors, 6.7 s of summed call time, 236,548 input tokens, $0.0099. Answers: `reports/x12_jev_answers.jsonl`.

One example state (curve A, seed 0), shortened. The curve has 80 points, every 5 trees to 400.

```json
{"loss_curve_every_5_trees": [{"trees": 5, "train_loss": 0.45867, "validation_loss": 0.47673},
                              {"trees": 10, "train_loss": 0.44789, "validation_loss": 0.46648}, "..."],
 "options": {"argmin": {"trees": 93, "validation_loss": 0.44439}, "minus_20": {"trees": 74, "validation_loss": 0.44466},
             "minus_10": {"trees": 84, "validation_loss": 0.4445}, "plus_10": {"trees": 102, "validation_loss": 0.44452},
             "plus_20": {"trees": 112, "validation_loss": 0.44467}, "one_se": {"trees": 45, "validation_loss": 0.44632},
             "plateau_start": {"trees": 85, "validation_loss": 0.44449}, "max": {"trees": 400, "validation_loss": 0.44669}},
 "drift": {"train_oct21_26": {"rows": 300000, "ctr": 0.183, "site_side_share": 0.657, "c20_minus1_share": 0.476,
                              "null_device_share": 0.83, "unseen_c14_share": 0.0},
           "validation_oct27": {"rows": 86788, "ctr": 0.1955, "site_side_share": 0.658, "c20_minus1_share": 0.427,
                                "null_device_share": 0.787, "unseen_c14_share": 0.135},
           "calibration_oct28": {"rows": 142909, "ctr": 0.1654, "site_side_share": 0.63, "c20_minus1_share": 0.448,
                                 "null_device_share": 0.827, "unseen_c14_share": 0.527}}}
```

## Results

Mean Oct 29–30 log loss over 30 curves, 127,406 test rows. A negative difference means the first arm is better.

| Arm | Test log loss | vs argmin (95% CI, characters) | vs argmin (95% CI, curves) |
| --- | --- | --- | --- |
| **Jev** | **0.416307** | **0 (0 to 0)** | **0 (0 to 0)** |
| argmin | 0.416307 | | |
| 0.8 × argmin | 0.416298 | −0.000009 (−0.000103 to +0.000077) | −0.000009 (−0.000084 to +0.000073) |
| 0.9 × argmin | 0.416285 | | |
| plateau_start | 0.416225 | | |
| 1.1 × argmin | 0.416478 | | |
| 1.2 × argmin | 0.416598 | | |
| one_se | 0.417262 | +0.000955 (+0.000636 to +0.001245) | +0.000955 (+0.000790 to +0.001121) |
| max (400) | 0.419589 | | |

Jev against the other controls: vs one_se −0.000955 (−0.001245 to −0.000636); vs 0.8 × argmin +0.000009 (−0.000077 to +0.000103).

- **Jev picked argmin on all 30 curves.** It picked it in both option orders, so order stability is 100%. Its probability on argmin ranged from 0.35 to 0.64 (median 0.51). Its second choice was always one_se or plateau_start.
- **Argmin is hard to beat here.** By feature set, argmin ran 85–109 trees for A, 92–166 for B and 69–80 for C. The best option in hindsight was 0.8 × argmin on 10 curves, plateau_start on 10, argmin on 6, 0.9 × argmin on 3 and 1.1 × argmin on 1. Every option within ±20% of argmin lands within 0.0003 of it. The two clear losers are one_se (+0.00096) and max (+0.0033).

**Pre-declared win: not met.** The win needed Jev at least 0.0003 below argmin with an interval excluding 0. Jev is argmin, so the difference is exactly 0. This is the trap Fable named: Jev reads off argmin and ties by construction. It did so 30 times out of 30.

## What it means for ranking

- Keep early stopping on the selection day. No option within ±20% of argmin, including Jev's, moved test loss by the 0.0003 the win asked for; only one_se (+0.00096) and max (+0.0033) moved it, both for the worse. The curve near argmin is flat enough that the choice of tree count barely matters.
- Jev's one useful behavior is negative: it never picked one_se or max, the two options that clearly hurt. Argmin avoids them too, at no cost.
- Ranking order and calibration are not affected by this choice in any way that matters at this size.

## Limits

- The Oct 29–30 test window was already inspected while building B, so everything here is exploratory. Nothing here is a causal CTR lift.
- Training rows are capped at 300,000 per fit, so each model is weaker than the full-data B (test loss 0.4123 for B here vs 0.4095 for the bundle).
- The 30 curves share one validation day and one test window. The curve interval only covers seed and subsample noise, not day-to-day drift.
- Jev saw the drift table for train, Oct 27 and Oct 28 only. It never saw a test row.
- One wording, asked once per order. A wording that hid the argmin label might change the picks, but tuning it on this window would overfit the test.

Full suite: 48 passed in 10.95s at e804154
