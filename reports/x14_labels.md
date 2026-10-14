# x14: can Jev spend a label budget better than acquisition rules?

Branch `exp/x-training-loop`, 2026-10-14. Batch X, worker E, Astra's experiment 3. Code: `simula/x14_labels.py` and the shared `simula/xtrain.py`. Pre-registered in commit `452abb8` before any outcome-bearing run.

## Question

B starts with 50,000 labeled rows and can buy 20,000 more. Jev sees summaries of 40 cohorts of unlabeled rows and decides where the labels go. Does the retrained B predict future clicks better than with random, uncertainty, or uncertainty-plus-diversity sampling?

This tests sample efficiency only. Every "new" label already existed in the export, so nothing here measures a real labeling cost saving.

## Setup

- **Pool.** Oct 21–26 (642,897 rows). Each seed labels 50,000 random pool rows first.
- **Model.** Every arm fits B's recipe on its labeled rows with the seed's LightGBM seed. Trees come from early stopping on Oct 27 (up to 400). Each arm gets B's half intercept shift on Oct 28 and is scored on Oct 29–30 (127,406 rows).
- **Cohorts (40).** Genre (10) × surface (site/app) × creative frequency: "rare_ad" if the C14 has fewer than 20 rows in the first 50,000 labels, else "common_ad". Per cohort, Jev sees unlabeled rows, labeled rows, coverage, the share of unlabeled rows whose C14, app or site is unseen in the labels, the mean entropy of the 50,000-row model, and the mean disagreement (sd across three bootstrap models of the same 50,000 rows). It never sees an unlabeled click.
- **Jev arm.** 20 sequential quotas of 1,000 labels. Each quota is one Choice over the 40 cohorts, with the state updated for labels already allocated. The quota is drawn at random inside the chosen cohort, or all of it if fewer rows remain.
- **Controls, 20,000 labels each.**
  - **Random**: 20,000 random pool rows.
  - **Uncertainty**: the 20,000 rows with the highest entropy.
  - **Uncertainty + diversity**: per-cohort quota proportional to unlabeled rows × mean entropy, filled with each cohort's highest-entropy rows.
- **Seeds.** Three. Each seed changes the start sample, the bootstrap models and the LightGBM seed.
- **Intervals.** Per seed, the paired bootstrap that resamples whole characters over test rows (`bootstrap_diff`, 1,000 draws).

### Changes made before pre-registration

A smoke run (3 states, no model scored on any outcome) led to two changes. Both are in the pre-registration commit.

1. **Cohort split.** The first version split creatives into C14 seen or unseen in the first 50,000 labels. The unseen cohorts held about 100 rows each, too small to receive a 1,000-row quota. The split became rare (< 20 labeled rows) vs common.
2. **"Stop" removed.** Astra's design offered "stop" next to the cohorts. In the smoke run, Jev chose stop on the first quota in all three seeds (confidence 0.29–0.33). That leaves the budget unspent, so the comparison would no longer be at equal labels. Stop was dropped. This version had no outcome run, so it has no numbers.

### Jev

`jev-latest` resolved to `jev-1.13.0`.

- `cohort` is a Choice. Instructions: "A click model has 50,000 labeled training rows. Labels are bought in quotas of 1,000 rows drawn at random from one cohort. Which cohort should get the next quota so the retrained model predicts clicks on future traffic best?"
- Option text names the cohort, for example `"c31": "romance characters, site side, common ads"`.
- Every call was repeated with the 40 options reversed; the forward answer drives the allocation.
- **Cost.** 120 calls, 120 attempts, 0 errors, 27.1 s of summed call time, 673,414 input tokens, $0.028. The two smoke runs added 6 calls and 33,248 tokens ($0.0014). Answers: `reports/x14_jev_answers.jsonl`.

Example state, seed 0, first quota, 3 of 40 cohorts shown:

```json
{"labels_spent_of_20000": 0,
 "cohorts": {
  "c01": {"genre": "anime", "surface": "app", "creative": "common_ad", "unlabeled_rows": 16680, "labeled_rows": 1411,
          "coverage": 0.078, "novel_id_share": 0.019, "mean_entropy_bits": 0.502, "mean_disagreement": 0.0241},
  "c02": {"genre": "anime", "surface": "app", "creative": "rare_ad", "unlabeled_rows": 1790, "labeled_rows": 142,
          "coverage": 0.0735, "novel_id_share": 0.093, "mean_entropy_bits": 0.5262, "mean_disagreement": 0.0279},
  "c31": {"genre": "romance", "surface": "site", "creative": "common_ad", "unlabeled_rows": 38362, "labeled_rows": 3189,
          "coverage": 0.0767, "novel_id_share": 0.012, "mean_entropy_bits": 0.7485, "mean_disagreement": 0.0349},
  "...": "37 more"}}
```

## Results

Oct 29–30 log loss. Jev − control is negative when Jev is better.

| Seed | Jev labels spent | Jev | Random | Uncertainty | Unc. + diversity | Jev − random (95% CI) | Jev − uncertainty (95% CI) | Jev − unc. + div. (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (not comparable) | 2,559 | 0.42424 | 0.42209 | 0.42511 | 0.42451 | +0.00215 (+0.00169 to +0.00263) | −0.00086 (−0.00128 to −0.00047) | −0.00027 (−0.00073 to +0.00017) |
| 1 | 20,000 | 0.42189 | **0.42092** | 0.42279 | 0.42270 | +0.00098 (+0.00050 to +0.00146) | −0.00090 (−0.00136 to −0.00046) | −0.00080 (−0.00132 to −0.00030) |
| 2 | 20,000 | 0.42163 | **0.42076** | 0.42285 | 0.42244 | +0.00087 (+0.00036 to +0.00137) | −0.00122 (−0.00166 to −0.00077) | −0.00081 (−0.00132 to −0.00037) |

- **Jev put every quota into one cohort.** In seed 0 it chose c32 (romance, site side, rare ads) 20 times. In seeds 1 and 2 it chose c31 (romance, site side, common ads) 20 times. These are the cohorts with the highest mean entropy and disagreement. Order stability was 100%: the reversed menu gave the same pick every time.
- **It kept buying from an empty cohort.** c32 had 2,559 unlabeled rows. After three quotas, its state showed `unlabeled_rows: 0` and `coverage: 1.0`, and Jev kept choosing it. So seed 0 spent only 2,559 of the 20,000 labels.
- **Seed 0 is not comparable.** The pre-registration promised every arm an equal 20,000 labels. The code still offered an emptied cohort, and a pick of it took zero rows. Seed 0 broke the equal budget, so it is shown for the record and left out of the verdict. Seeds 1 and 2 spent 20,000 in every arm, and the verdict rests on them. No unavailable-option revision was run; a rerun that drops emptied cohorts from the menu would need its own pre-registration and would not replace these numbers.
- **Random sampling wins both comparable seeds.** Jev − random is +0.00098 and +0.00087 in seeds 1 and 2, intervals excluding 0. Both uncertainty rules lose to random too: entropy-ranked rows are a biased slice of the traffic.
- **Jev beats the uncertainty rules in both comparable seeds,** with intervals excluding 0: by 0.00090 and 0.00122 against uncertainty, 0.00080 and 0.00081 against uncertainty + diversity.

**Pre-declared win: not met.** The win needed Jev below every control in each seed. On the two comparable seeds, Jev loses to random in both. The verdict is a loss, on seeds 1 and 2. (Seed 0 points the same way, but its budget was unequal.)

## What it means for ranking

- For a fixed budget of labels, a random sample of traffic trained the best B here. Concentrating labels in the most uncertain slice, whether a rule or Jev does it, makes the training set unlike the serving traffic and costs about 0.001–0.002 in log loss.
- Jev behaved like a cohort-level uncertainty sampler. It did no worse than the row-level rules and sometimes better, but it ignored a cohort that had run dry.
- Nothing here changes ranking. It is about how a training set is built.

## Limits

- Sample efficiency only. No label was actually bought; every label already existed.
- The test window was already inspected, so everything is exploratory. Nothing here is a causal CTR lift.
- Small models: 50,000–70,000 training rows, 49–77 trees. The ranking among arms could differ at full scale.
- Three seeds share one validation day, one calibration day and one test window.
- Two comparable seeds, not three: seed 0 broke the equal budget (see Results).
- One wording after the two pre-registration changes. A question that asked for a spread of labels, or said a cohort was empty, might behave differently; that was not tried.

Full suite: 49 passed in 131.81s at 0dd9271
