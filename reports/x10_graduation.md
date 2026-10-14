# x10: can Jev decide when a new creative graduates from the prior?

Branch `exp/x-novelty`, 2026-10-14. Batch X, worker D. Design: Fable's E5 ("sequential graduation"). Code: `simula/x10_graduation.py`.

## Question

e4 blended B's pCTR for a new creative (C14) toward a cell prior with a fixed weight. A real system sees a new creative's own clicks pile up, hour by hour. As evidence arrives, when should scoring stop trusting the prior and start trusting the creative's own record? Here Jev makes that call at checkpoints, choosing a source for the creative's next impressions.

**Result: null in substance.** The pre-declared bar was met: Jev's policy beats the Beta posterior by 0.024 log loss. But the Beta control is weak, and Jev chose the e4 blend at 97% of checkpoints, covering 98% of rows. Against the e4 blend, the other named control, Jev is 0.00018 worse, with an interval spanning zero. Jev does not beat that control, so it adds nothing.

## Setup

- **Rows.** Every Oct 29–30 test row whose creative has no rows in the refit window (Oct 21–27): 54,602 rows, 403 creatives, CTR 14.6%. Rows are replayed in hour order over 30 hours.
- **Running record.** For a row in hour h, N and K are its creative's impressions and clicks in **strictly earlier hours**. A row never sees a click from its own hour. A test checks this: changing clicks inside an hour leaves every prediction in that hour unchanged.
- **Checkpoints.** A checkpoint fires when N first reaches 0, 10, 25, 50, 100 or 250. When one hour carries N past several thresholds, only the highest one fires. There are 847 checkpoints, by level: 403 / 147 / 104 / 101 / 59 / 33.
- **Sources.** Each is recomputed every hour from strictly earlier hours:
  - **prior**: e4's hierarchical table prior for the row's cell (genre, C18, app_category). It is refit-window counts with m = 20 at each level, the same as `simula/jev_prior.py`.
  - **blend**: e4's fixed blend, k = 500 and b = 0.1. An unseen creative has n = 0 in refit, so this is 0.9·pB + 0.1·prior.
  - **B**: calibrated pCTR from `bundle/B`.
  - **running_ctr**: (K + 0.5) / (N + 1). Jeffreys smoothing keeps log loss finite at small N.
- **Jev's policy.** At each checkpoint, Jev picks one source. Every row of that creative uses it until the next checkpoint.
- **Controls.**
  - **e4's fixed blend** on every row.
  - **A Beta posterior**, (K + 20·prior) / (N + 20): the table prior as m = 20 pseudo-counts, updated with the creative's earlier clicks.
- **Pre-declared win** (Fable). Jev beats the Beta posterior by ≥ 0.0005 log loss on these rows, with the interval excluding 0.
- **Intervals.** Paired bootstrap of per-row log-loss differences, resampling whole characters, 1,000 draws (`bootstrap_diff`).
- **Test-window outcomes.** By design, the running record uses earlier-hour test clicks. That is the online state the replay simulates. No test outcome set any rule, threshold or prompt.

### Jev

- **State.** It holds:
  - the checkpoint and the next one;
  - the creative's impressions, clicks and hours seen so far;
  - what each source says now, averaged over the creative's earlier rows. At n = 0 it is averaged over the current hour's rows, using features only;
  - e3's attribute statistics from the first row of that hour: each attribute value with its refit impressions and smoothed CTR.

  It holds no character description. Example, attributes cut:

  ```json
  {"checkpoint": 50, "next_checkpoint": 100,
   "creative_so_far": {"impressions": 95, "clicks": 6, "hours_seen": 1},
   "estimates": {"averaged_over": "earlier impressions", "prior": 0.1431, "blend": 0.054, "B": 0.0441,
                 "running_ctr": 0.0677},
   "attributes": {"slot": 0, "C18": {"value": "0", "impressions": 297662, "smoothed_ctr": 0.1753}, "...": "..."}}
  ```

- **Question.** One Choice, `source`: "This ad creative is new: it had no impressions before this period. Which estimate should score its next impressions, until it reaches the next checkpoint? Use the evidence in the state." The options are:
  - prior: the cell prior from similar ads' history;
  - blend: the click model, nudged slightly toward the cell prior;
  - B: the click model's prediction;
  - running_ctr: this creative's own click rate so far.
- **Order.** The question, state and policy were committed (`prereg:` commit d5f02a1) before any arm was scored. They did not change.
- **Cost.** 850 calls (3 smoke, 847 checkpoints), 0 errors, 9 s of wall time on 8 threads, a median of 0.08 s per call. 754,160 input tokens, **$0.032**. `jev-latest` resolved to `jev-1.13.0`. The answers are in `reports/x10_graduation_jev_answers.jsonl`.

## Results

Log loss on the 54,602 unseen-creative rows:

| Arm | Log loss | vs Jev's policy (95% CI; negative = Jev better) |
| --- | --- | --- |
| Beta posterior (control, pre-declared bar) | 0.395722 | Jev −0.02369 (−0.02577, −0.02166) |
| **e4 fixed blend (control)** | **0.371853** | Jev +0.00018 (−0.00006, +0.00042) |
| Jev's policy | 0.372029 | |
| B | 0.372398 | Jev −0.00037 (−0.00081, +0.00006) |
| prior only | 0.401937 | |
| running CTR only | 0.432434 | |

**Jev's choices.** Of 847 checkpoints, Jev chose blend 822 times, prior 18, B 6 and running_ctr once. Weighted by rows, 98% went to blend. Its policy is the fixed blend with a few detours, and the detours cost a little.

By checkpoint level, log loss of each source on the rows scored at that level:

| Checkpoint | Rows | Creatives | CTR | prior | blend | B | running CTR | Beta | Jev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9,107 | 403 | 0.146 | 0.4028 | **0.3781** | 0.3786 | 0.6006 | 0.4025 | **0.3781** |
| 10 | 2,834 | 147 | 0.165 | 0.4327 | **0.4175** | 0.4188 | 0.4697 | 0.4364 | 0.4188 |
| 25 | 2,601 | 104 | 0.175 | 0.4510 | 0.4254 | **0.4252** | 0.4529 | 0.4436 | 0.4255 |
| 50 | 5,641 | 101 | 0.186 | 0.4657 | **0.4442** | 0.4458 | 0.4723 | 0.4641 | 0.4444 |
| 100 | 4,986 | 59 | 0.157 | 0.4178 | 0.3788 | **0.3785** | 0.4059 | 0.4038 | 0.3788 |
| 250 | 29,433 | 33 | 0.131 | 0.3794 | **0.3458** | 0.3463 | 0.3719 | 0.3710 | 0.3459 |

**The finding is not about Jev.** The creative's own record never catches up with B here, even after 250 impressions: 0.372 for running CTR against 0.346 for B. One click rate per creative ignores the row's context (app, slot, device, hour), and B uses it. That is why the Beta posterior is a weak bar: it graduates toward a number that is worse than the model at every level. "Graduation" from the model to the creative's raw rate is the wrong move on this data. Jev's habit of staying with the blend was right, but the fixed blend does the same with no judgment.

## What it means for ranking

- **Jev adds nothing to graduation.** It reproduces the e4 fixed blend, and the fixed blend costs a dict lookup.
- **Do not graduate a new creative to its own CTR.** Within the 30 hours available, a creative-level rate never beats B's contextual prediction. If own-history should enter, the natural form is a shrunk creative-level offset on B's logit, learned from earlier hours' residuals, so B keeps its context. This was not tested here.
- **The pre-declared bar was poorly chosen.** Beating the Beta posterior was easy because the Beta posterior ignores B. Read this as a null, not a win.

## Limits

- The test window (Oct 29–30) was inspected in earlier experiments, so everything here is exploratory.
- The replay covers 30 hours. Only 33 creatives reach 250 impressions, and they carry 54% of the rows.
- Pairs of sources share most of their information, so the per-level differences are small. They are reported without intervals; only the pooled comparisons have them.
- These are log-loss changes on logged impressions. Nothing here is a causal CTR lift.

Full suite: <filled at the end>
