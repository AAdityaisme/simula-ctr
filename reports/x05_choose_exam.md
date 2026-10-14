# x05: can Jev choose the validation split that picks B's training recipe?

Branch `exp/x-judge`, 2026-10-14. Batch X, worker B. Code: `simula/x05_exam.py`. Design: Astra 7. Pre-registration: `reports/x_judge_prereg.md`, commit `4afd4c6`.

## Question

A recipe is chosen by its loss on some validation split, and the split decides which failure the choice guards against. Before any loss exists, can Jev pick the split whose winner holds up best on the following days? It has to do better than plain chronological validation and better than a rule that matches the split to the most common kind of novelty.

## Setup

- **Rolling origins.** Both come before the test window. Oct 29–30 is never read.
  - o1: the pool is Oct 21–24, deployed on Oct 25–26 (194,166 rows).
  - o2: the pool is Oct 21–26, deployed on Oct 27–28 (229,697 rows).
- **Splits per pool.** There are four:
  - chronological: validate on the pool's last day;
  - held-out C17;
  - held-out character;
  - held-out app.
  - Held-out values are drawn at random until they cover one day's rows. The draw skips values above half a day's rows and the no-app placeholder.
  - Every fit part has 300,000 rows. Every validation part has 89,378 rows in o1 and 100,000 in o2.
- **Recipes.** There are three:
  - B;
  - B with 15 leaves;
  - B with C14, C17 and C21 blanked together on a random 10% of creatives.
  - Each fit has a fixed 183 trees, so no fit sees its own validation rows. That makes 3 × 4 × 2 = 24 fits, scored with raw log loss. All 24 fits fit the budget, so no origin was cut. They took 1 min 48 s.
- **Selection.** On the chosen split, the recipe with the lowest validation loss is deployed.
- **Recipe regret (headline).** The deployed recipe's later-window loss, averaged over its four fits, minus the best recipe's. Averaging keeps training-data recency equal across recipes. Astra's literal "fit regret" is also reported: the selected fit against the best of all 12 fits.
- **Controls.**
  - Chronological selection.
  - Novelty matching: the held-out split for the field with the highest pool churn. That is C17 at both origins.
- **Intervals.** A paired bootstrap on the later window resamples whole characters, comparing Jev's selected recipe with each control's. The rows here are impressions.

### Jev

`jev-latest` resolved to `jev-1.13.0`. One question, written before any fit and not changed. This is the only version.

- `exam` is a Choice. Instruction: "Three training recipes will be compared on one validation split, and the one with the lowest validation log loss is deployed on the next two days. Which split's validation loss best predicts which recipe will do best after deployment?" The options are the four splits, each with a one-line description.
- Jev was asked in three option orders per origin. The original order is the decision.
- **Cost.** 6 calls with 0 errors. They used 7,476 input tokens, or $0.0003. The answers are in `reports/x05_choose_exam_jev_answers.jsonl`.

The o1 state:

```json
{"pool_days": [141021, 141022, 141023, 141024], "pool_rows": 448731,
 "deployment": "the model is deployed on the two days after the pool",
 "pool_last_day_churn": {"C14": 0.1322, "C17": 0.1004, "character_id": 0.005, "app_id": 0.0041},
 "field_roles": {"C14": "creative id", "C17": "ad attribute that groups creatives",
                 "character_id": "character (AI persona)", "app_id": "host app; ecad2386 marks site-side traffic with no app"},
 "recipes": {"B": "LightGBM model B as shipped (31 leaves).", "B_15_leaves": "Model B with 15 leaves per tree.",
             "B_joint_id_mask": "Model B trained with C14, C17 and C21 blanked together on the rows of a random 10% of creatives (C14), to rehearse new creatives."},
 "splits": {"chronological": {"how": "Validate on the pool's last day; fit on the earlier days.", "fit_rows": 300000, "validation_rows": 89378,
                              "validation_novelty_vs_fit": {"C14": 0.1322, "C17": 0.1004, "character_id": 0.0053, "app_id": 0.0043}},
            "heldout_C17": {"how": "Validate on rows of a random set of C17 values (an ad attribute that groups creatives); fit on rows of all other C17 values, all days.", "fit_rows": 300000, "validation_rows": 89378,
                            "validation_novelty_vs_fit": {"C14": 1.0, "C17": 1.0, "character_id": 0.0004, "app_id": 0.0097}},
            "heldout_character": {"...": "same shape"}, "heldout_app": {"...": "same shape"}}}
```

## Results

| Origin | Selector | Split | Recipe deployed | Recipe regret | Fit regret | Jev − selector, later log loss (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| o1 | **Jev** | chronological | B | **0** | 0.0025 | |
| o1 | Chronological | chronological | B | 0 | 0.0025 | 0 (identical choice) |
| o1 | Novelty matching | held-out C17 | B, joint ID mask | 0.0036 | 0.0053 | −0.0036 (−0.0040 to −0.0032) |
| o2 | **Jev** | chronological | B | **0** | 0.0015 | |
| o2 | Chronological | chronological | B | 0 | 0.0015 | 0 (identical choice) |
| o2 | Novelty matching | held-out C17 | B, joint ID mask | 0.0032 | 0.0054 | −0.0032 (−0.0036 to −0.0029) |
| Mean | Jev / chronological / novelty | | | 0 / 0 / 0.0034 | 0.0020 / 0.0020 / 0.0054 | |

**Jev chose chronological every time.** That is 6 of 6 answers: both origins, all three option orders. P(chronological) ranged from 0.65 to 0.77, and held-out C17 was always second, at 0.17–0.26. Reordering the options changed nothing.

**Pre-declared win: not met.** The win needed Jev's mean regret to be strictly below both controls'. Jev beat novelty matching by about 0.003 per origin, with intervals well clear of 0. But it made the same choice as chronological validation, so it tied that control exactly. Verdict: **null**.

**The win was out of reach here.** Plain B had the lowest averaged later-window loss at both origins, and chronological validation also picked B. So the chronological control already had zero recipe regret, and no selector could beat it. I could only know that after the fits. The held-out C17 split picked the joint-ID-mask recipe, and that recipe was the worst of the three at both origins. Its rehearsal of new creatives did not pay off on the next two days.

**Fit regret.** It is never zero for any selector. The best single fit at both origins is B on the held-out-character split. That split's fit part includes the pool's last day, while the chronological fit part does not. So a selector deploying "chronological" also deploys an older training set. The recipe regret above removes this confound by averaging.

## What it means for ranking

On these two origins, chronological validation is the right exam for choosing B's recipe, and Jev agrees with it, confidently and stably. It adds nothing over the default. The rule that "validate on the most common novelty" would have shipped a worse recipe. Jev did not fall for it, though C17 was always its second choice.

## Limits

- There are two origins, three close recipes and one random draw of held-out values per split. Recipe gaps are 0.001–0.005 in log loss. This is a pilot, not a reliability estimate.
- There is no calibration and no refit after selection, so these are raw-loss comparisons of fitted models.
- The later windows (Oct 25–28) contain no test-window rows, but the test window was already inspected by earlier work. Everything here is exploratory and nothing is a CTR lift.

Full suite: 49 passed, 2 warnings in 14.00s at 9af6cda
