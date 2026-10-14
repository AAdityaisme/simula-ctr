# x06: Can Jev judge evidence at the deploy gate?

Branch `exp/x-evidence`, 2026-10-14. Code: `simula/x06_evidence.py`.

## v2 (2026-10-14, after the red-team review of `abecb65`)

**v1 is superseded.** Its readout states carried real Oct 29–30 clicks: every `sides` and `mix` state (and the 50
reshuffled rechecks) showed click totals drawn from the test window. That breaks rule 3 of
`specs/rank-jev-x/common.md`: no test-window outcomes in any Jev state. The root cause is a spec error by the
orchestrator: `specs/rank-jev-x/wc-evidence.md` asked for readouts "from test-window rows", which contradicts rule 3.
I followed the spec line without flagging the conflict. v1's panel options were also always in one order.

### v2 pre-registration (committed before any v2 Jev call)

Everything in the v1 pre-registration below still holds, except:

- **(a) readouts use refit-window rows only.** Real impressions and clicks come from Oct 21–27. No Oct 29–30 row
  reaches a readout state. The δ injection uses the refit per-side base rate. The 50 reshuffled-rows rechecks stay.
- **(b) panels keep their rows**: Oct 29–30 features and B's scores, with clicks simulated from models fitted before
  the window, so no real test click enters a state. The three options are now listed in a seeded random order per
  panel. 50 rechecks (every sixth panel) ask again with that panel's order reversed.
- **Order sensitivity, reported:** the share of top choices that were the first-listed option (chance 1/3), the
  recheck same-choice share, and the mean absolute change in the probability of the main call's top choice.
- **Wins unchanged.** Verdict: win if the win holds; loss if the interval of Jev minus the control is below 0;
  null otherwise.
- Files: answers `reports/x06_v2_jev_answers.jsonl`, numbers `reports/x06_evidence_judge_v2.json`. The v1 files are
  kept unchanged.

### v2 results

`jev-latest` resolved to `jev-1.13.0`. 800 calls in 801 attempts with 0 errors, 482,661 input tokens, $0.020.
Answers: `reports/x06_v2_jev_answers.jsonl`; numbers: `reports/x06_evidence_judge_v2.json`.

**(a) Readouts from refit rows: loss.** 200 readouts, 40 per δ, paired bootstrap over readouts (2,000 resamples).

| Arm | AUROC, all 200 | AUROC, δ ≠ 0 | vs naive z (95% CI) | vs CMH (95% CI) |
| --- | --- | --- | --- | --- |
| Naive z on overall CTR | 0.792 | 0.834 | | |
| CMH by side | **0.912** | **0.936** | +0.119 (+0.068 to +0.179) | |
| Jev, per-side table | 0.712 | 0.754 | −0.080 (−0.148 to −0.016) | −0.199 (−0.273 to −0.130) |
| Jev, overall rows + mix note | 0.768 | 0.811 | −0.024 (−0.068 to +0.020) | |

- Jev with the per-side table is below the naive test, and the interval excludes 0. The pattern matches v1: given
  the per-side rows, Jev does worse than with the overall rows alone.
- Mean Jev Noul by δ (−1, −0.5, 0, +0.5, +1 pp): per-side 0.32, 0.36, 0.41, 0.45, 0.47; mix note 0.27, 0.30, 0.32,
  0.33, 0.35. Reshuffling the rows changes the Noul by 0.098 on average (50 readouts).

**(b) Benchmark on trial: null.** The panels are the same 150 genuine and 150 false panels as in v1. Only the option
order changed.

| Arm | Genuine panels kept | False panels rejected | AUROC (genuine vs false) |
| --- | --- | --- | --- |
| Checklist after exact-request rematch | 45.3% | **92.7%** | |
| Jev, thresholded to keep 45.3% of genuine | 45.3% | 86.0% | 0.801 |
| B z on request-identical pairs alone | | | 0.616 |
| B accuracy minus request-only accuracy | | | **0.942** |

- Jev minus checklist, false-panel rejection at equal retention: −6.7 pp (95% CI −13.1 to 0.0), bootstrap over
  panels. The upper end touches 0, so under the registered rule this is **null**, not a loss. The point estimate
  still favours the checklist, and the plain accuracy gap ranks panels better than Jev.
- Jev's top choice was `rematch` on all 300 panels again.
- **Order sensitivity.** The top choice was the first-listed option on 32.3% of panels (chance 33.3%; it is always
  `rematch`). Reversing a panel's order kept the top choice on 50 of 50 rechecks and moved its probability by 0.087 on
  average. The score itself is order-sensitive: mean P(`predictive_diagnostic`) is 0.137 / 0.134 / 0.190 when that
  option is listed first / second / last. Against v1's fixed order (listed first, mean 0.115), the per-panel scores
  correlate at only 0.67. The AUROC moved from 0.829 to 0.801.

### v2: what it means for ranking

- With clean inputs the conclusion is unchanged. At the deploy gate a stratified test (CMH) beats Jev clearly, and
  Jev is below even the naive test. On benchmark validity, the hand-written checklist and the one-line accuracy gap
  do at least as well as Jev, whose score also shifts with option order.

The Oct 29–30 window was already inspected, so every v2 result is still exploratory. One wording per question.

## v1 pre-registration (superseded)

The questions, state formats, controls and wins below are fixed in code. Both tasks are synthetic and their truth
comes from the generating process. Intervals resample tasks (readouts or panels), not impressions.

### (a) Ship/no-ship readouts (Fable E7)

- 200 readouts from Oct 29–30 rows. Each arm draws 8,000 real impressions, split by side. Control site share is
  U(0.25, 0.55). The candidate adds U(0, 0.35) site share; site CTR is higher, so the mix alone flatters it.
- A per-side shift δ ∈ {−1, −0.5, 0, +0.5, +1} pp is injected into the candidate by flipping clicks with the
  probability that moves that side's expected CTR by exactly δ. 40 readouts per δ.
- Truth: ship iff δ > 0 (δ = 0 counts as no-ship). AUROC on all 200, and on the 160 with δ ≠ 0 as a secondary.
- Jev, Noul `ship`: "The candidate arm has a truly higher click rate than the control arm on comparable traffic,
  so it should ship."
  - Arm `sides`: rows `{arm, traffic, impressions, clicks}` for 2 arms × 2 sides plus 2 overall rows, in random
    order. Arms are named `control` and `candidate`; sides `site` and `app`.
  - Arm `mix`: the 2 overall rows plus a note: "The arms' traffic mix differs: site-side traffic is X% of control
    and Y% of candidate impressions."
  - Recheck: the first 50 `sides` states again with the rows reshuffled.
- Controls: naive pooled z on overall CTR; signed CMH statistic stratified by side.
- **Win:** AUROC(Jev sides) ≥ AUROC(CMH), and AUROC(Jev sides) − AUROC(naive) with its 95% interval above 0.

### (b) Benchmark on trial (Astra 5)

- Panels of matched pairs from Oct 29–30 rows: a clicked and an unclicked impression with different C14 in one
  exact context (`hour, character_id, app_id, site_id, device_type`), as in `rank_pairs.build_pairs`. Each panel
  samples 300–1,500 context groups and draws a fresh click per row.
- Genuine panels: clicks ~ B's calibrated pCTR, so the candidate matters. False panels: clicks ~ a request-only
  LightGBM (B's request and character fields, B's 183 trees, a 300k-row refit subsample, calibrated like B on
  Oct 28), so any ranking gain is nuisance. Only panels where B (the ranker) beats 50% become claims. 150 of each.
- Jev, Choice `use`: "How should this comparison be used for the stated claim?" Options `predictive_diagnostic`,
  `rematch`, `unidentifiable`. Score = P(predictive_diagnostic).
- State: the claim sentence, matching keys, the request/candidate contract, the same provenance sentence for every
  panel, pair count, B's accuracy, the request-only ranker's accuracy, the count and B accuracy of pairs identical on
  every request field, and the share of pairs differing per unmatched request field.
  - Recheck: the first 50 panels again with the options in reverse order.
- Control: exact-request matching plus a fixed checklist. Keep iff ≥ 10 request-identical pairs, B above 50% on
  them, and B above the request-only ranker. Secondary scores: the z of B on request-identical pairs, and B minus the
  request-only ranker.
- **Win:** at the checklist's retention of genuine panels, Jev (thresholded on its score) rejects a larger share of
  false panels, with the bootstrap interval of the difference above 0.
- Same-information check: the AUROCs of the two secondary scores are reported next to Jev's. If Jev beats the
  checklist but not the plain accuracy gap, the report says the arithmetic already had it.

The Oct 29–30 window was already inspected, so everything is exploratory.

## v1 results (superseded: test-window clicks in the readout states)

`jev-latest` resolved to `jev-1.13.0`. This is the only version of the questions and states. 800 calls in 800
attempts with 0 errors, 8 s of wall time, 482,649 input tokens, $0.020. Answers: `reports/x06_jev_answers.jsonl`.

### (a) Readouts: Jev loses to both tests

200 readouts, 40 per δ. Intervals are a paired bootstrap over readouts (2,000 resamples).

| Arm | AUROC, all 200 | AUROC, δ ≠ 0 | vs naive z (95% CI) | vs CMH (95% CI) |
| --- | --- | --- | --- | --- |
| Naive z on overall CTR | 0.831 | 0.889 | | |
| CMH by side | **0.885** | **0.947** | +0.054 (+0.015 to +0.095) | |
| Jev, per-side table | 0.763 | 0.792 | −0.068 (−0.121 to −0.017) | −0.122 (−0.182 to −0.064) |
| Jev, overall rows + mix note | 0.801 | 0.853 | −0.030 (−0.065 to +0.004) | |

Mean Jev Noul by δ:

| δ (pp) | −1 | −0.5 | 0 | +0.5 | +1 |
| --- | --- | --- | --- | --- | --- |
| Jev per-side | 0.32 | 0.31 | 0.35 | 0.43 | 0.46 |
| Jev mix note | 0.26 | 0.27 | 0.30 | 0.33 | 0.34 |

- **Win not met; this is a loss.** Jev with the per-side table is below the naive test, and the interval excludes 0.
  It is far below CMH.
- The mix trap works as designed: CMH beats naive by 0.054. Jev does not recover it. Given the per-side rows, it
  does *worse* than with the overall rows alone (0.763 vs 0.801). The extra rows add noise to its judgment rather
  than a stratified estimate.
- Jev's scores move in the right direction with δ but stay below 0.5 everywhere. It is cautious, and the ordering
  is weak.
- Reshuffling the row order changes the Noul by 0.069 on average (50 readouts).

Example state (`sides:3`, δ = +0.5 pp):

```json
{"rows": [{"arm": "control", "traffic": "app", "impressions": 4499, "clicks": 684},
          {"arm": "control", "traffic": "site", "impressions": 3501, "clicks": 684},
          {"arm": "candidate", "traffic": "app", "impressions": 1900, "clicks": 255},
          {"arm": "candidate", "traffic": "site", "impressions": 6100, "clicks": 1180},
          {"arm": "candidate", "traffic": "all", "impressions": 8000, "clicks": 1435},
          {"arm": "control", "traffic": "all", "impressions": 8000, "clicks": 1368}]}
```

### (b) Benchmark on trial: Jev loses to the checklist and to plain arithmetic

150 genuine and 150 false panels, all with B above 50%. Median panel: about 430 pairs, of which 15–16 are identical
on every request field. Median B accuracy is 57.6% on genuine panels and 54.0% on false ones; the request-only
ranker scores 54.0% and 57.1%.

| Arm | Genuine panels kept | False panels rejected | AUROC (genuine vs false) |
| --- | --- | --- | --- |
| Checklist after exact-request rematch | 45.3% | **92.7%** | |
| Jev, thresholded to keep 45.3% of genuine | 45.3% | 85.3% | 0.829 |
| B z on request-identical pairs alone | | | 0.616 |
| B accuracy minus request-only accuracy | | | **0.942** |

- Jev minus checklist, false-panel rejection at equal retention: −7.3 pp (95% CI −12.9 to −1.9), bootstrap over
  panels. **Win not met; this is a loss.**
- Jev's top choice was `rematch` on all 300 panels. Its score still ranks panels (AUROC 0.83), so it reads the
  evidence. The single accuracy gap ranks them better (0.94). The arithmetic already had it.
- Reversing the option order left the top choice unchanged on 50 of 50 panels (it was always `rematch`).
- The request-identical rematch has little support: about 15 pairs per panel. On its own it barely separates the
  worlds (0.62). This is Astra's point: exact matching leaves almost nothing to test.

Example state (`panel:0`, the contract lists are shortened here; Jev saw them in full):

```json
{"claim": "Ranker R scores the clicked ad above the unclicked ad in 60.6% of 411 matched pairs (a coin scores 50%). Claim: R ranks candidate ads.",
 "matching_keys": ["hour", "character_id", "app_id", "site_id", "device_type"],
 "contract": {"request": ["hour_of_day", "site_id", "...", "character_age_days"], "candidate": ["banner_pos", "C14", "...", "C21"]},
 "outcome_provenance": "click labels simulated per impression from a rate model fitted before this window",
 "request_only_ranker": "a model fitted on request fields only, scored on the same pairs",
 "pairs": 411, "ranker_accuracy": 0.6058, "request_only_ranker_accuracy": 0.5377,
 "request_identical_pairs": {"pairs": 15, "ranker_accuracy": 0.4667},
 "share_of_pairs_differing": {"device_model": 0.964, "device_conn_type": 0.129, "is_null_device": 0.01}}
```

## v1: what it meant for ranking (superseded)

- At the deploy gate, a standard stratified test beats Jev clearly. A typed judge reading a 2×2 readout is not a
  substitute for CMH, and it is not even as good as the naive test. Keep the gate arithmetic.
- On benchmark validity, Jev ranks panels sensibly but never commits, and a one-line accuracy gap does better. A
  validity checklist written by hand is the better tool.
- B is untouched by either result. These tasks test the process around B, and the arithmetic version of each
  process wins.

## v1 limits (superseded)

- Both tasks are synthetic. Readouts reuse real Oct 29–30 clicks, and panels reuse real rows with simulated clicks.
  The window was already inspected, so everything is exploratory. Nothing here is a causal CTR lift.
- In (b), the request-only model is the false world's generator, so the request-only ranker is the oracle there.
  This flatters the accuracy-gap score. A real audit would not have the generator.
- The mix shift in (a) always flatters the candidate, with a random size. δ = 0 counts as no-ship.
- One wording per question, asked once. A different rubric might do better, but tuning it here would overfit.


Full suite: 67 passed in 207.08s at 77d68bb
