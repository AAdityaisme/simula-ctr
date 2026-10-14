# Can Jev pick the clicked ad, or tell when B's pick is wrong?

Branch `exp/rank-e12`, 2026-10-14. Experiments 1 and 2 of the rank-with-Jev lane. The design comes from the merged ranking reviews (`~/simula-local/reviews/ad-classify-2026-10-14/report-ranking-fable.md` and `report-ranking-astra.md`). Code: `simula/rank_pairs.py`, which experiment 3 reuses, and `jev_ask.py`.

## Question

1. Given two candidate ads in one context, does Jev choose the one the user clicked more often than model B does?
2. Can Jev tell when B's pick is wrong, better than B's own margin and a simple logistic control?

## Setup

- **Pairs.** A group is test impressions (Oct 29–30) with the same hour, character, app, site and device type. We keep groups with at least one click, one non-click and more than one creative (C14). Then every clicked row is paired with every unclicked row that has a different C14. That gives 1,735 pairs from 1,360 groups and 710 characters. Fable's count matched. Display order (X1 or X2) comes from a sha256 of the two impression ids, so it never depends on the click.
- **Per row.** We use B's calibrated pCTR from `bundle/B` and the `lookup(app_id, C18)` baseline fit on the refit window (Oct 21–27). Refit-window history comes from m-estimates with m=20 around the refit CTR. It covers the creative (C14), genre × C18 and app category × C18. An unseen creative gets 0 impressions.
- **Subsets.** Both creatives seen in refit (851 pairs), both unseen (424), mixed (460), and near-tie, meaning |ΔpCTR| < 0.01 (300).
- **Arms.** Coin. B, the higher pCTR, with an exact tie scored 0.5 (16 pairs). The lookup baseline. Creative history, the higher smoothed creative CTR. Jev with pCTR. Jev without pCTR. The baselines score ties 0.5 too.
- **Significance.** The 95% intervals come from a paired bootstrap that resamples whole characters, 1,000 times (`bootstrap_diff`). This interval is the comparison over all 1,735 pairs. McNemar is an exact binomial test on the discordant pairs, and it is conditional: it only counts pairs where both arms pick decisively, with no 0.5 tie. The table gives that denominator for each arm.
- **B-implied benchmark.** If B's pCTRs were the true rates, picking the higher one would score the mean of p_hi(1−p_lo) / (p_hi(1−p_lo) + p_lo(1−p_hi)). That is 57.4% on all pairs. It is a benchmark for B, not an upper bound on rankers: one with better information than B could score above it.

### Jev

`jev-latest`, which resolved to `jev-1.13.0`. We wrote the questions and state format once, in code, before looking at any outcome by arm. We did not change them afterward, so this is the only version.

- `pick`, a Choice: "Which candidate ad is the user more likely to click? Use the evidence in the state." The options are `{"X1": "candidate X1", "X2": "candidate X2"}`.
- `gate`, a Noul: "The candidate with the higher model_pctr is the one the user clicks."
- **Calls.**
  - With pCTR: one call per pair asks both questions (1,735 calls).
  - Without pCTR: one call per pair asks `pick` only, and the state has no `model_pctr` (1,735 calls).
  - Order check: 300 fixed pairs with X1 and X2 swapped, with pCTR, `pick` only (300 calls).
- **Cost.** 3,770 calls in 53 s of wall time with 8 threads. The median call took 0.10 s. They used 3.24M input tokens, or $0.14 at $0.042 per million, with 0 errors. That run did not count retries, so these tokens and dollars cover the 3,770 successful attempts only. A failed attempt that was retried may have used more. `jev_ask.py` now counts attempts separately. The answers are in `reports/rank_e12_jev_answers.jsonl`. States are not committed.

One example state, with pCTR. The character description is omitted here because it comes from the private dataset; Jev saw the full text.

```json
{"context": {"character_genre": "fantasy", "character_description": "<omitted>", "app_category": "07d7df22",
             "site_category": "28905ebd", "hour_of_day": 0, "device_type": 1},
 "candidates": {
  "X1": {"slot": 0, "C15": "320", "C16": "50", "C17": "2616", "C18": "0", "C19": "35", "C20": "-1", "C21": "51",
         "creative_history": {"impressions": 1637, "clicks": 431, "smoothed_ctr": 0.2623},
         "genre_x_C18": {"impressions": 26872, "smoothed_ctr": 0.1679},
         "app_category_x_C18": {"impressions": 238185, "smoothed_ctr": 0.187}, "model_pctr": 0.2519},
  "X2": {"slot": 0, "C15": "320", "C16": "50", "C17": "1722", "C18": "0", "C19": "35", "C20": "100084", "C21": "79",
         "creative_history": {"impressions": 11614, "clicks": 2364, "smoothed_ctr": 0.2035},
         "genre_x_C18": {"impressions": 26872, "smoothed_ctr": 0.1679},
         "app_category_x_C18": {"impressions": 238185, "smoothed_ctr": 0.187}, "model_pctr": 0.2312}}}
```

## Results

### Experiment 1: pairwise accuracy (% of pairs where the arm picks the clicked ad)

| Arm | All (95% CI) | Both seen | Both unseen | Mixed | Near-tie | vs B, pp, all pairs (95% CI) | McNemar on decisive pairs: n, wins / losses, p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-implied benchmark (B's pCTR as truth) | 57.4 | 56.6 | 55.9 | 60.1 | 50.7 | | |
| Coin | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | −5.2 (−7.7 to −2.8) | |
| **B** | **55.2 (52.8 to 57.7)** | 53.3 | 53.7 | 60.2 | 50.7 | | |
| Lookup (app_id, C18) | 51.7 (50.3 to 53.3) | 50.8 | 52.2 | 52.8 | 52.2 | −3.5 (−6.4 to −0.9) | 488, 123 / 116, p=0.70 |
| Creative history | 52.1 (49.8 to 54.2) | 48.7 | 50.0 | 60.4 | 48.5 | −3.1 (−5.6 to −0.6) | 1,307, 175 / 214, p=0.054 |
| Jev with pCTR | 54.5 (52.0 to 57.1) | 50.9 | 53.8 | 62.0 | 49.7 | −0.7 (−2.2 to +0.9) | 1,719, 93 / 104, p=0.48 |
| Jev without pCTR | 53.4 (50.8 to 55.7) | 50.8 | 49.8 | 61.5 | 48.7 | −1.8 (−4.6 to +0.8) | 1,719, 290 / 321, p=0.22 |

Both candidates share one app_id, so the lookup ties on 1,246 pairs (scored 0.5). Its McNemar test therefore covers only 488 pairs and does not test the −3.5 pp all-pair difference; the bootstrap interval does. B's 16 exact ties are left out of every McNemar count.

**Order robustness.** On the 300 swapped pairs, Jev picks the same underlying ad 90.7% of the time. So about one pick in eleven depends on which ad is listed first.

**Pre-declared win, not met.** Jev without pCTR does beat a coin: 53.4%, with an interval from 50.8 to 55.7. But Jev with pCTR does not beat B: −0.7 pp, with an interval from −2.2 to +0.9. Both conditions were required.

### Experiment 2: does a score separate B's right picks from its wrong ones?

This uses the 1,719 pairs with a nonzero margin, where B is right on 55.3%.

| Gate score | AUROC | B's accuracy on the top 100% | 80% | 60% | 40% | 20% |
| --- | --- | --- | --- | --- | --- | --- |
| Margin, \|ΔpCTR\| | 0.550 | 55.3 | 56.3 | 57.8 | 59.7 | 63.7 |
| Logistic control (out-of-fold) | **0.572** | 55.3 | 56.8 | 59.5 | 61.6 | **67.2** |
| Jev `gate` | 0.543 | 55.3 | 56.8 | 58.4 | 59.2 | 59.0 |

The logistic control uses the margin, both-seen, both-unseen, log1p creative impressions of the higher and the lower candidate, and |Δ lookup|. It is fit out-of-fold with 5-fold GroupKFold over characters.

Jev AUROC − logistic AUROC = **−0.029 (−0.055 to −0.004)**. **Pre-declared win, not met.** The win needed +0.03 with an interval that excludes 0. Jev is reliably *worse* than the logistic control, and slightly below B's raw margin.

## What it means for ranking

- **Jev does not out-pick B.** With B's pCTR in the state, Jev picks the same ad as B on 1,522 of the 1,719 pairs where B is decisive, and the gap to B is within noise. Without pCTR, it lands between creative history and B.
- **Jev's edge over a coin sits in the mixed pairs.** These are the pairs where one creative has history and the other does not. There Jev without pCTR scores 61.5%, and creative history alone scores 60.4%. On both-seen and both-unseen pairs, it scores 50.8% and 49.8%. No state ablation was run, so this is an association: it does not show which part of the state Jev uses.
- **B sits 2.2 pp below its own implied benchmark.** If B's pCTRs were the true rates, picking the higher one would score 57.4% overall and 50.7% on near-ties. That benchmark does not bound a better-informed ranker, so these pairs cannot say how much room is left. As the review warned, "a null cannot separate 'Jev adds nothing' from 'nothing is there'."
- **As a confidence gate, Jev is the wrong tool.** A six-feature logistic regression on numbers B already has beats it. At 20% coverage, the logistic regression gives 67.2% accuracy and Jev only 59.0%. An abstain rule should use the margin plus creative exposure, which needs no network call.
- **Keep `rank()` as it is.** Neither experiment justifies a Jev call in the serving path.

## Limits

- The Oct 29–30 test window was already inspected while building B, so everything here is exploratory.
- A pair is two impressions in one context, not a logged auction. The user never chose between these two ads. Nothing here is a causal CTR lift.
- The pairs come from 710 characters, and a group can yield several pairs that share rows. The character-cluster bootstrap covers the clustering, but 1,735 pairs cannot detect differences under about 2–3 pp.
- B is calibrated on Oct 28, and the refit history ends on Oct 27, and both are before test, so the state leaks no test clicks.
- The questions were asked once, in one wording. A different rubric might do better, but tuning it on this window would overfit the test.
- The 300-pair order check uses only the with-pCTR state. The without-pCTR state was not checked for order effects.

Full suite: 36 passed in 9.69s at 03bb842
