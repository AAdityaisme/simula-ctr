# Pre-registration: x03, x04, x05 (worker B, branch `exp/x-judge`)

Written 2026-10-14 before any Jev call and before any outcome-bearing fit. The exact questions, state formats and
controls are the constants and functions in `simula/x03_triage.py`, `simula/x04_leaks.py` and `simula/x05_exam.py`
at this commit. In all three, Jev reads definitions and process descriptions, never rows. The test window (Oct 29–30)
was already inspected by earlier work, so every result is exploratory.

> Correction, 2026-10-14 (red team on `1d613cd`): for x04 the claim above is false. The v1 card statistics, including
> MI with click, were written to the work directory 46 s before this commit. The x04 v2 section below re-runs it in order.

## x03: feature triage from definitions alone (Fable E1)

- **Zoo.** Fable lists 11 features: six candidates (prior impressions today for any ad and for this creative, hours
  since the creative was first seen, IP volume today, device_model rarity, the `C20 = -1` flag) and five decoys
  (`conversation_turn`, turn / `session_msg_count`, safety tier, character trailing CTR, `num_interactions`). Fable
  says 12, so the 12th is raw `session_msg_count`, the second known leak. "Safety tier (random)" is the character's
  tier as an ordered number; B already has the tier as a category, so it is a redundant decoy.
- **Definitions.** Counts "today" use strictly earlier hours of the same day (same-hour rows have no order). Device
  keys are `device_ip` + `device_model`. Trailing CTR uses strictly earlier days and never a test-window label.
- **State.** Assignment excerpt, B's feature list, the card (name, one-line definition, availability note). No rows,
  no statistics. Asked 3 times.
- **Questions.** Score `expected_gain` on {hurts, none, small, medium, large}; Noul `available_at_decision_time`.
- **Jev ranking.** Mean Score over the 3 asks. Jev-top-3 = the 3 highest; ties keep zoo order.
- **Fits.** B + f per feature on train.py's recipe: early stopping on Oct 27, refit through Oct 27 with that tree
  count, calibrate on Oct 28 (one intercept, shrink 0.5, bundle B's recipe), test Oct 29–30. Both fitting windows
  are subsampled to the same 300,000 rows for every arm. Realized gain = test log loss of B minus that of B + f.
- **Controls.** B + random-3, 20 fixed seeded draws from all 12; B + oracle-3 (top 3 by realized test gain) as the
  ceiling.
- **Win (Fable).** Spearman(Jev score, realized test gain) over all 12 ≥ 0.5, **and** Jev-top-3 beats the per-row
  mean of the 20 random-3 arms, with the paired character bootstrap interval excluding 0. The full-12 Spearman is
  the headline, not top-3.
- **Intervals.** Per-feature and per-set gains: `bootstrap_diff`, whole characters resampled. Spearman: one-sided
  permutation p and a bootstrap over the 12 features.

## x04: leak and availability auditor (Fable E2 + Astra 1)

- **Cards.** All 33 real columns of the two CSVs plus 41 constructed recipe cards, materialized on the refit window
  (Oct 21–27) only: Fable's three plants (click plus noise, a same-day device total, a within-day shuffled copy of
  click), Astra's future aggregates and nested aliases (of click and of the two real leaks), legal lagged versions,
  two identity recipes and one unscored probe (Fable's day-scoped `device_ip` counter, whose truth is a policy call).
  Truth is fixed in code by construction.
- **Label set: Fable's {safe, leaky, unavailable_at_decision_time, identity_restricted}.** Each label states a fact
  about how a field is built and when its inputs exist, so truth follows from the recipe. Astra's `lag_one_day` is a
  remedy, and a future aggregate could rightly be lagged or dropped, so its truth would be ambiguous; Astra's set
  also has no way to say "barred as identity", which the README needs for `device_id` and `device_ip`. The win is
  binary: a card is rejected when the label is anything but `safe`.
- **State.** A fixed context (decision moment; click labels for a day arrive at its end) and the card: name,
  definition, Fable's four statistics on Oct 21–27 (MI with click, Spearman with the character's impression count,
  share above `conversation_turn`, cardinality). Never truth or lineage metadata.
- **Arms.** Names shown; names masked (`f01`–`f74`, fixed shuffled order, definition kept); masked with the option
  order reversed (Astra's shuffled-option check).
- **Controls.** (1) Fable's MI-outlier rule: reject when the modified z-score of MI across all cards exceeds 3.5.
  (2) Astra's denylist (`CONTRACT["never"]`, by name and through a recipe's source columns) plus a timestamp/lineage
  rule on structured metadata: a recipe over rows at or after the impression is unavailable; one reading click from
  rows whose labels are not yet available is leaky. In the masked arm a real column's own name is hidden from the
  denylist. (3) The timestamp/lineage rule alone. (4) The union of (1) and (2).
- **Win (spec).** On the masked arm: Jev catches all three plants and both known real leaks (`session_msg_count`,
  `num_interactions`), its leak recall (over all leaky and unavailable cards) is above the best control, and its
  false rejection of legal (`safe`) cards is at most 2%. A control counts as "best" only if its own false rejection
  is at most 2% on that arm.

## x05: choose the exam before choosing the model (Astra 7)

- **Origins.** Both before the test window, which is never read: pool Oct 21–24 deploying on Oct 25–26, and pool
  Oct 21–26 deploying on Oct 27–28. Both origins fit the budget, so none is cut.
- **Splits.** Chronological (last pool day), held-out C17, held-out character, held-out app. Held-out values are
  drawn at random (seed 0), skipping values above half a day's rows and the no-app placeholder, until they cover one
  day's rows. Every split's fit part is subsampled to the same size (at most 300,000) and every validation part to
  the same size (at most 100,000).
- **Recipes.** B; B with 15 leaves; B with C14, C17 and C21 blanked together on a random 10% of creatives. Fixed 183
  trees (bundle B's count), so no fit touches its validation rows. 3 × 4 × 2 = 24 fits, raw log loss.
- **State.** Pool days and rows, last-day churn per field, field roles, the recipes, and each split's sizes and
  validation novelty against its fit part. No clicks and no losses.
- **Question.** Choice `exam` over the four splits. Asked in three option orders per origin; the original order is
  the decision, the other two measure instability.
- **Selection.** On the chosen split, the recipe with the lowest validation loss.
- **Regret (headline).** Recipe regret: the selected recipe's later-window loss averaged over its four fits, minus
  the best recipe's. Averaging keeps training-data recency equal across recipes. Astra's literal "fit regret" (the
  selected fit against the best of all 12 fits) is reported too.
- **Controls.** Chronological selection; a novelty-matching selector that picks the held-out split for the field
  with the highest pool churn.
- **Win (Astra).** Jev's mean recipe regret over the two origins is lower than both controls'. Interval: paired
  character bootstrap on the later window between Jev's selected recipe and each control's.

## x04 v2: cards, truth and policy (after the red team on `1d613cd`, GATE: fail)

Written 2026-10-14, committed before the v2 cards are built and before any v2 card statistic is computed. Why a v2:
the v1 card statistics (including MI with click) were written 46 s before the v1 prereg commit; three cards labelled
`safe` used information from the whole Oct 21–27 window; and the denylist control applied a wider policy than the gold
labels. v1 results stay in the reports, marked superseded.

- **Three cards rebuilt from decision-time values only** (`simula/x_judge.py`: `trailing_ctr`, `frozen_decile`).
  `creative_ctr_earlier_days` and `character_ctr_earlier_days` are smoothed with 20 pseudo-impressions toward the
  overall click rate of strictly earlier days (an expanding prior); rows on the first day, with no earlier labels, are
  missing. `creative_ctr_previous_7_days_decile` takes its decile cutoffs from rows of strictly earlier days
  (expanding), so a row's decile never depends on its own day or later. Their definitions say so. Truth stays `safe`.
- **One admission policy for the gold labels and the denylist control.** A field may be used iff it is known when the
  ad is chosen, is not derived from clicks not yet known, and is not barred by the identity policy. The never-list
  (`CONTRACT["never"]`) also holds modelling choices this policy admits (`id`, `character_name`,
  `character_description`, `created_at`, `conversation_turn`); they stay `safe` in the gold labels, and the denylist
  control now reads only the never-list entries the policy bars: `click`, `session_msg_count`, `num_interactions`,
  `device_id`, `device_ip`. The truth for `session_msg_count` and `num_interactions` is the README's availability
  judgment, adopted as policy, not proven by construction.
- **Everything else as v1:** the 75 cards and truth counts (44 / 13 / 13 / 4 / 1 probe), Fable's four statistics on
  Oct 21–27, the question and labels, the three arms, the masked names, the controls and the win.
- **Order:** this commit → build cards and statistics → ask Jev all 225 states again (fresh answers file
  `reports/x04_leak_auditor_v2_jev_answers.jsonl`; v1 answers are not reused) → score.
- **Scope of the masked arm:** definitions keep their provenance words ("click plus Gaussian noise", "same day",
  "next 24 hours", `session_msg_count`), so the masked arm tests reading explicit provenance prose, not detecting
  hidden lineage.

## x03 v2: target (after the red team on `1d613cd`)

Written 2026-10-14, committed before the v2 reruns. Why a v2: the v1 target was one LightGBM seed's gain, which
mostly measured the change in random draws that any extra column causes; it ranked the three inadmissible features
alongside the rest; and `character_trailing_ctr` smoothed toward the CTR of all of Oct 21–28, so earlier rows saw
later clicks.

- **Feature fix.** `character_trailing_ctr` smooths toward the overall click rate of strictly earlier days (no test-
  window label); rows on Oct 21 are missing. The card Jev read ("uses clicks from strictly earlier days only") is
  unchanged and now matches the feature, so Jev's 36 v1 answers are reused and no new call is made.
- **Admissible features only.** The 9 features available at decision time by construction. `turn_share_of_session`,
  `num_interactions` and `session_msg_count` are not fitted, ranked or drawn.
- **Estimand.** Every arm is refit under 5 LightGBM seeds (20260920–20260924) on the same 300,000-row subsamples and
  train.py's recipe. A feature's gain = the mean over seeds of (B's test log loss − B + f's), with a 95% t-interval
  over seeds. The character bootstrap on seed-averaged row losses is reported as secondary.
- **Arms.** B; the 9 singles; Jev-top-3 = the 3 admissible features with the highest mean Jev score, ties in zoo order;
  oracle-3 = the top 3 by v2 gain; 20 random-3 sets drawn (seed 0) from the 9. 160 fits, OMP_NUM_THREADS=3.
- **Win.** Spearman(Jev mean score, v2 gain) over the 9 ≥ 0.5, **and** the 95% seed t-interval of the per-seed
  difference (Jev-top-3 test loss − mean of the 20 random-3 test losses) lies entirely below 0.
- **Availability is scored separately:** accuracy of Jev's `available_at_decision_time` (share ≥ 0.5 = available)
  over all 12 cards, outside the win.
- **Not blind.** This target was defined after the v1 single-seed and post-hoc 5-seed results were seen; those used
  the same seeds and already showed 8 of the 9 admissible singles. New data here: the fixed trailing CTR and the
  5-seed set arms. Exploratory.
