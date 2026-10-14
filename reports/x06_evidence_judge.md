# x06: Can Jev judge evidence at the deploy gate?

Branch `exp/x-evidence`, 2026-10-14. Code: `simula/x06_evidence.py`.

## Pre-registration (committed before any outcome-bearing run)

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
