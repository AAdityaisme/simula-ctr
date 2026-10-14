# Can a Jev-written cell prior help brand-new creatives?

Branch `exp/rank-e4`, 2026-10-14. Experiment 4 of the rank-with-Jev lane. Code: `simula/jev_prior.py`.

## Question

A new creative (C14) has no history, so model B sends its C14 down LightGBM's missing branch. Can a per-cell CTR prior, written offline by Jev, improve predictions for new creatives? Serving would add only a dict lookup.

**Result: no.** Jev's prior loses to a plain hierarchical table prior built from the same counts, by 0.00044 log loss on the unseen-creative slice, and the interval excludes zero. It ties B. The table prior itself gives a small, real gain over B.

## Setup

- **Cells.** A cell is `(genre, C18, app_category)` as seen in the refit window (Oct 21–27). There are **584 cells**. The median cell has 18 rows and 61.3% have under 50. This matches Fable's count.
- **State Jev reads.** Refit-window counts only, never a test click. One example:

  ```json
  {"cell": {"genre": "anime", "C18": "0", "app_category": "07d7df22"},
   "cell_history": {"impressions": 22704, "clicks": 4199},
   "neighbours": {"genre_x_C18": {"impressions": 27938, "clicks": 4826},
                  "app_category_x_C18": {"impressions": 238185, "clicks": 44532},
                  "genre": {"impressions": 71391, "clicks": 12971},
                  "C18": {"impressions": 297662, "clicks": 52194}},
   "global_ctr": 0.185}
  ```

- **Jev question.** One Score question, `ctr`: "What click-through rate should we expect for ads in this cell? Use the cell's history and its neighbours." Rungs: under 5%, 5–10%, 10–15%, 15–25%, over 25%. Prior = Σ p(rung) × midpoint, with midpoints 0.025, 0.075, 0.125, 0.20, 0.30. The question and state were fixed in code before any arm was scored. They were not changed after.
- **Table prior (control).** A hierarchical m-estimate with m = 20 at each level, from the same numbers: cell shrinks to genre × C18, which shrinks to C18, which shrinks to the global CTR.
- **Blend.** The same rule for both priors: `p = (1 − b(1−w))·pB + b(1−w)·prior`, with `w = n/(n + k)` and n the creative's refit impressions (0 if unseen). k ∈ {50, 500} and b ∈ {0.1, 0.25, 0.5} were picked per prior on Oct 28 only. Both priors picked k = 500, b = 0.1.
- **Test.** Oct 29–30, 127,406 rows. The unseen-C14 slice has 54,602 rows. 13 test rows fall in a cell not seen in refit; they keep pB.
- **Intervals.** Paired bootstrap of the per-row log-loss difference, resampling whole characters, 1,000 draws (`bootstrap_diff`).
- **Jev cost.** `jev-latest` resolved to `jev-1.13.0`. 687 calls in total (3 smoke, 584 cells, 100 re-asks). 10 s of wall time with 8 threads. 380,070 input tokens, **$0.016** at $0.042 per million. No errors.

## Results

| Arm | All test log loss | All test AUC | Unseen-C14 log loss | Unseen-C14 AUC | Mean pCTR, unseen |
| --- | --- | --- | --- | --- | --- |
| B | 0.409450 | 0.7290 | 0.372398 | 0.7318 | 0.1479 |
| B + table prior | **0.409070** | **0.7302** | **0.371852** | **0.7340** | 0.1507 |
| B + Jev prior | 0.409332 | 0.7298 | 0.372292 | 0.7334 | 0.1517 |

The unseen slice's observed click rate is 0.1457.

| Comparison (negative = candidate better) | Slice | Log-loss change | 95% interval |
| --- | --- | --- | --- |
| Jev vs table | unseen C14 | **+0.00044** | +0.00037 to +0.00051 |
| Jev vs table | all | +0.00026 | +0.00023 to +0.00030 |
| Jev vs B | unseen C14 | −0.00011 | −0.00049 to +0.00029 |
| Jev vs B | all | −0.00012 | −0.00030 to +0.00005 |
| Table vs B | unseen C14 | −0.00055 | −0.00090 to −0.00018 |
| Table vs B | all | −0.00038 | −0.00055 to −0.00022 |

**Pre-declared win:** Jev beats the table by ≥ 0.001 on the unseen slice, interval excluding 0. **Not met.** Jev is worse than the table, and the interval excludes 0 on the wrong side.

**Why Jev loses.** Its prior is the table prior, compressed. The two correlate at 0.93 across cells. Jev's spread is narrower (standard deviation 0.051 against 0.080), because five rungs with fixed midpoints round each cell to a bucket and cap it at 0.025–0.30. Reading a count table and returning a rung is Bayesian smoothing with extra steps and coarser output.

**Latency.** Prior lookup plus blend for a 50-candidate request takes a median of **7.5 µs** (table) and 7.4 µs (Jev) over 1,000 runs, in pure Python. That is noise against the 50 ms budget. `rank()` itself is unchanged; this experiment adds no code to it.

**Determinism.** Re-asking Jev on 100 random cells moved the prior by 0.0054 on average (absolute).

## What it means for ranking

- **Ship the table, not Jev, for this job.** A hierarchical cell prior blended into B's pCTR for new creatives lowers log loss by 0.0005 on unseen creatives and lifts AUC by 0.002. It costs a dict lookup per candidate and needs no model call.
- **Jev adds nothing on counts.** When the input is a table of impressions and clicks, the right answer is arithmetic. Jev's typed rungs throw away precision the table keeps. Its value, if any, is on inputs a table cannot read, such as text. This repeats the `jev_ctr` pattern.
- **The graduation rule works as a serving mechanism.** `w = n/(n + k)` fades the prior as a creative gathers impressions. It is the cold-start path the README said was not implemented, and it fits inside the latency budget.

## Limits

- The Oct 29–30 test window was already inspected while building B. Everything here is exploratory.
- B's calibration was fit on Oct 28, and k and b were also picked on Oct 28. The same day serves twice.
- Both priors picked the smallest b in the grid (0.1). A finer grid could pick a smaller blend. The grid was fixed in advance.
- The gain is small and partly a level shift: the table blend moves mean pCTR on unseen rows from 0.148 to 0.151 against an observed 0.146. The AUC gain shows it also reorders.
- The prior is per cell. Two new creatives in the same cell get the same shift, so it cannot separate them beyond what B already does. It helps across cells only.
- Nothing here is a causal CTR lift. Clicks were logged under some existing policy.

Full suite: 36 passed in 10.46s at 0b661c6
