# x13: can Jev decide when to spend a scarce refit? (pilot)

Branch `exp/x-training-loop`, 2026-10-14. Batch X, worker E, Astra's experiment 4. Code: `simula/x13_retrain.py` and the shared `simula/xtrain.py`. Pre-registered in commit `7c9946f` before the replay ran.

## Question

A deployed B drifts. Each night a policy can hold, update only the intercept, or retrain the trees on the last three days, with at most two retrains. Does Jev, reading residual and mix summaries, choose actions that give lower next-day loss than simple rules with the same budget?

## Setup

- **Replay.** The starting model is B's recipe fit on 300,000 Oct 21–25 rows, 110 trees, intercept 0. 110 is near the 300,000-row Oct 27 argmin of the x12 smoke curves. Oct 26 is scored with this model for every policy.
- **Decisions.** At midnight before Oct 27, 28, 29 and 30, each policy picks one action using labels through the previous day:
  - `hold`: keep model and intercept.
  - `intercept_update`: B's half intercept shift, fitted on the previous day.
  - `refit_recent_three_days`: 110 trees on the previous three days (at most 300,000 rows), intercept 0. At most two per replay.
- **Policies.** All share the two-refit budget.
  - **jev**: Jev's Choice. A refit pick with no budget left counts as hold.
  - **scheduled**: refit before Oct 27 and Oct 29, intercept update otherwise.
  - **ewma**: g = logit(actual CTR) − logit(mean prediction) on the previous day; e = 0.5 g + 0.5 e, reset after a refit. Refit when |e| > 0.10 and budget remains, else intercept update.
  - **hour_z**: per hour of the previous day, z = (clicks − Σp) / √Σp(1−p); Z = Σz / √hours. Refit when |Z| > 3 and budget remains; intercept update when |Z| > 2; else hold.
  - References with no refits: always_hold and always_intercept.
- **Score.** Pooled log loss on Oct 27–29, and the partial Oct 30 (6 hours) separately. Three seeds change the subsample and the LightGBM seed. The interval is the paired bootstrap that resamples whole characters over the pooled rows of all three seeds (`bootstrap_diff`, 1,000 draws).
- **Prior-day labels.** The replay feeds each day's labels into the next decision by design. So the Oct 30 decision sees Oct 29 labels, which are test-window labels. No threshold, rule or wording was tuned on any outcome.

### Jev

`jev-latest` resolved to `jev-1.13.0`. One version, written before the replay; not changed.

- `action` is a Choice. Instructions: "It is midnight. A click model serves ads all of tomorrow. Labels are in through yesterday. Refits are scarce: at most two in this replay. Which action keeps tomorrow's log loss lowest?"
- Options: `hold` "keep the current model and intercept unchanged"; `intercept_update` "keep the trees, refit only the intercept on yesterday's labels"; `refit_recent_three_days` "retrain the trees on the last three labeled days (uses one refit)".
- Each decision was asked twice, once with the options reversed. The forward answer drives the replay.
- **Cost.** 24 calls, 24 attempts, 0 errors, 4.6 s of summed call time, 23,072 input tokens, $0.0010. Answers: `reports/x13_jev_answers.jsonl`.

One example state: seed 2 at its last decision, after two refits and a hold.

```json
{"decision": "midnight before Oct 30; the action applies to all of Oct 30",
 "current_model": {"trained_on": "Oct 25-27", "intercept_shift": 0.0},
 "refits_left": 0, "decisions_left_including_this": 1,
 "training_window": {"ctr": 0.1943, "site_side_share": 0.68, "c20_minus1_share": 0.434, "null_device_share": 0.824,
                     "unseen_c14_share": 0.0},
 "recent_out_of_time_days": {
  "Oct 28": {"labeled_rows": 142909, "actual_ctr": 0.1654, "mean_predicted_ctr": 0.1791, "log_loss": 0.40257,
             "by_hour_block": {"00-05": {"actual_ctr": 0.1848, "mean_predicted_ctr": 0.1976}, "06-11": {"...": "..."},
                               "12-17": {"...": "..."}, "18-23": {"...": "..."}},
             "site_side_share": 0.63, "c20_minus1_share": 0.448, "null_device_share": 0.827, "unseen_c14_share": 0.369},
  "Oct 29": {"labeled_rows": 104450, "actual_ctr": 0.1727, "mean_predicted_ctr": 0.1853, "log_loss": 0.41654, "...": "..."}},
 "past_actions": {"Oct 27": "refit_recent_three_days", "Oct 28": "refit_recent_three_days", "Oct 29": "hold"}}
```

## Results

Pooled log loss over three seeds. A positive difference means Jev is worse.

| Policy | Refits | Oct 27–29 log loss | Jev − policy, Oct 27–29 (95% CI) | Oct 30 (partial) log loss | Jev − policy, Oct 30 (95% CI) |
| --- | --- | --- | --- | --- | --- |
| **jev** | 2, 2, 2 | **0.41469** | | **0.41348** | |
| scheduled | 2, 2, 2 | 0.41316 | +0.00154 (+0.00129 to +0.00180) | 0.41042 | +0.00306 (+0.00144 to +0.00470) |
| hour_z | 2, 2, 2 | 0.41341 | +0.00128 (+0.00099 to +0.00159) | 0.40550 | +0.00798 (+0.00640 to +0.00953) |
| ewma | 0, 0, 0 | 0.41435 | +0.00035 (0.00000 to +0.00071) | 0.40561 | +0.00787 (+0.00613 to +0.00970) |
| always_intercept | 0 | 0.41435 | +0.00035 (0.00000 to +0.00071) | 0.40561 | +0.00787 (+0.00613 to +0.00970) |
| always_hold | 0 | 0.41439 | +0.00030 (−0.00005 to +0.00066) | 0.40585 | +0.00763 (+0.00592 to +0.00943) |

Action sequences (Oct 27, 28, 29, 30):

| Policy | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| jev | refit, refit, intercept, intercept | intercept, intercept, refit, refit | refit, refit, (refit → hold), (refit → hold) |
| scheduled | refit, intercept, refit, intercept | same | same |
| hour_z | hold, hold, refit, refit | same | same |
| ewma | intercept × 4 (never triggered) | same | same |

- **Jev spent both refits early on two seeds.** Seeds 0 and 2 refit before Oct 27 and again before Oct 28. The Oct 28 refit trains on Oct 25–27, which ends on the highest-CTR day, right before the lowest-CTR day. That model then serves Oct 29–30. Seed 1 did the opposite and nearly matched hour_z (Oct 27–29: 0.41332 vs 0.41331).
- **Budget.** On seed 2, Jev asked for a third and fourth refit with `refits_left: 0` in its state, and the model over-predicting by about 1.3 points. Both counted as hold, so it skipped the intercept update that would have fixed the level.
- **Order stability.** Jev gave the same action with the options reversed on 8 of 12 decisions. Its top probability was 0.35–0.63, close to uniform on the first decision.
- **The EWMA rule never fired.** The previous-day logit gap never pushed |e| past 0.10, so ewma equals always_intercept.

**Pre-declared win: not met.** Jev has higher Oct 27–29 loss than scheduled and hour_z, with intervals excluding 0, and higher than ewma with an interval that touches 0. It is also worse on the partial Oct 30. The verdict is a loss, on twelve decisions.

## What it means for ranking

- The decision that mattered was timing: retraining right after the highest-CTR day locked in a high level for the next two days. A fixed every-other-day schedule and a residual z rule both avoided that. Jev did not.
- Intercept updates alone (always_intercept) trail the best policy by 0.0012 on Oct 27–29 and are 0.00011 behind the best on Oct 30. Well-timed refits help a little; badly timed ones cost more than they gain.
- None of these actions reorders candidates much except a refit; intercept updates change levels, not ranking. Nothing here is a ranking gain.

## Limits

- **Pilot.** There are four decisions per replay and three seeds that share the same days. The intervals cover row and seed noise, not the handful of truly independent day-level decisions.
- The test window was already inspected, so everything is exploratory. Nothing here is a causal CTR lift.
- By design, the Oct 30 decision used Oct 29 labels. Oct 30 has only six hours.
- Each fit used at most 300,000 rows and 110 trees, so these models are weaker than the full-data B.
- One wording. The refit option did not say "unavailable" when the budget was spent; the state did. A clearer option text might have avoided the seed-2 holds, but changing it after seeing the result would be tuning on the test window.

Full suite: 48 passed in 10.95s at e804154
