# x08: Can Jev diagnose a broken serving pipeline before anyone retrains B?

Branch `exp/x-evidence`, 2026-10-14. Code: `simula/x08_repair.py`. After Astra's experiment 8.

## Pre-registration (committed before any outcome-bearing run)

- **Incidents.** 160 serving batches, each a 3-hour block of Oct 29–30 rows (10 blocks; each block gets every fault
  type in turn). A fault hits a random share f ~ U(0.1, 1) of an in-memory copy. 40 of each:
  - `none`: untouched, so only the natural temporal shift against Oct 28 is present (the control incidents);
  - `join`: the hit rows get another character's safety tier, creator type, genre and creation date (so a wrong age);
  - `sentinel`: on the hit rows, C20's `-1` is written `-1.0` (the model reads it as unseen), and the null-device hash
    is written in upper case (so `is_null_device` turns false);
  - `both`.
- **Actions.** Two steps per incident, each one of `restore_join`, `restore_missing_encoding`, `retrain`, `hold`.
  Repairs re-derive their columns from the source before B scores. `retrain` switches to a cached B refit on
  Oct 22–28 (300k-row subsample, B's 183 trees, half-shift calibration on Oct 28, which is in its training window).
  `hold` keeps the cached `bundle/B`.
- **Jev**, Choice `intervention`: "What intervention fits this incident?" Options: `restore_join` ("re-run the
  character metadata join from the source registry"), `restore_missing_encoding` ("re-apply the raw encoding of
  sentinel and missing values"), `retrain` ("switch to the model refit on more recent days"), `hold` ("change
  nothing"). Step 2 sees the state after Jev's own step-1 action.
  - State: a fixed note on the schema (join key, C20's sentinel, the null-device hash, Oct 28 as reference), batch
    and reference values of seven label-free statistics, row count, and the actions already taken. The statistics:
    share of repeat characters with more than one metadata row, share of C20 = −1, share of null devices, share of
    C20 and of C14 values unseen by the model in use, site-side share, mean predicted CTR. No label, and no word
    naming a fault.
- **Controls.**
  - Invariant checks, fixed: `restore_join` if any repeat character has more than one metadata row; else
    `restore_missing_encoding` if unseen C20 rises more than 1 pp above reference or the null-device share moves by
    more than 5 pp; else `hold`. Never `retrain`.
  - A depth-3 decision tree on the same statistics (batch minus reference) plus which repairs are done, trained on
    200 incidents from Oct 25–27 blocks with separate seeds. Its label is the next correct repair (join first).
  - Reference arms: `hold` always, and an oracle that knows the fault.
- **Metrics.** Lost log loss = B's loss on the faulty batch minus on the clean batch. Restored share = Σ(faulty loss
  − final loss) / Σ lost, over the 120 faulty incidents. False-repair rate = share of all 160 incidents where a
  policy applies a restore whose fault was never injected. Retrain rate and the loss change on clean incidents are
  reported too.
- **Win:** Jev's restored share beats both controls, with the paired bootstrap interval over incidents above 0, at a
  false-repair rate no higher than each control's.

The Oct 29–30 window was already inspected, so everything is exploratory. Clicks enter only the loss, never a state.

## Results

`jev-latest` resolved to `jev-1.13.0`. One version of the question and state. 320 calls (160 per step) in 320
attempts with 0 errors, 4 s of wall time, 216,824 input tokens, $0.009. Answers: `reports/x08_jev_answers.jsonl`.

Mean lost log loss per incident: `join` 0.0073, `sentinel` 0.0003, `both` 0.0072. The sentinel fault barely hurts
B, so the restored share is driven by the join.

| Policy | Restored share of lost log loss | `join` | `sentinel` | `both` | False-repair rate (of 160) | Retrain rate |
| --- | --- | --- | --- | --- | --- | --- |
| Hold | 0% | 0% | 0% | 0% | 0% | 0% |
| Invariant checks | **100%** | 100% | 100% | 100% | 15.0% | 0% |
| Diagnostic tree | 99.8% | 100% | 89.6% | 100.0% | **0%** | 0% |
| Jev | **100%** | 100% | 100% | 100% | 43.1% | 0% |
| Oracle | 100% | 100% | 100% | 100% | 0% | 0% |

Jev minus each control, restored share, paired bootstrap over the 120 faulty incidents (2,000 resamples):

- vs invariant checks: 0.0 pp (0.0 to 0.0). They are identical on every faulty incident.
- vs the tree: +0.2 pp (−0.003 to +0.5). The tree held on 3 sentinel incidents.

Jev's two-step sequences:

| Fault | Step 1 → step 2 | Incidents |
| --- | --- | --- |
| `none` | `restore_missing_encoding` → `restore_missing_encoding` | 39 |
| `none` | `hold` → `hold` | 1 |
| `join` | `restore_join` → `restore_missing_encoding` | 30 |
| `join` | `restore_join` → `restore_join` | 10 |
| `sentinel` | `restore_missing_encoding` → `restore_missing_encoding` | 40 |
| `both` | `restore_join` → `restore_missing_encoding` | 40 |

- **Win not met; this is a loss.** Jev restores as much as the invariant checks, but its false-repair rate is 43%,
  against 15% for the checks and 0% for the tree. The win required a rate no higher than each control's.
- Jev finds the join fault every time, and never retrains. Its error is a reflex: it re-applies the encoding repair
  on 39 of 40 clean batches and on 30 of 40 join-only batches. Those repairs are harmless no-ops here, so the loss
  does not show them. In a real pipeline each one is an unneeded change that someone has to review.
- The invariant checks' 15% false repairs are all one rule. On the three blocks from Oct 29 18:00 through Oct 30 02:59,
  the null-device share drops 7–10 pp below Oct 28 by itself, which trips the 5 pp threshold. The tree, trained on
  earlier days, learned to ignore that statistic.
- No policy chose `retrain`, so the cached recent refit was never used.

Example state (`s1:3`, step 1; the `both` fault on block 0, Oct 29 00:00–02:59):

```json
{"notes": "Serving batch for model B. Character metadata is joined on character_id from a registry with one row per character. C20 uses -1 as a sentinel value. device_id uses one repeated hash for a missing id. Reference is Oct 28 traffic through the same pipeline.",
 "statistics": {"share_of_repeat_characters_with_several_metadata_rows": {"batch": 0.9889, "reference": 0.0},
                "share_C20_sentinel": {"batch": 0.0992, "reference": 0.4479},
                "share_null_device": {"batch": 0.159, "reference": 0.8272},
                "share_C20_unseen_by_model": {"batch": 0.4135, "reference": 0.0},
                "share_C14_unseen_by_model": {"batch": 0.3224, "reference": 0.3424},
                "share_site_side": {"batch": 0.6794, "reference": 0.6299},
                "mean_predicted_ctr": {"batch": 0.1666, "reference": 0.1687}, "rows": 9819},
 "actions_already_taken": []}
```

## What it means for ranking

- Before retraining B on a bad day, run invariant checks. A registry join can never give one character two
  metadata rows, and these encodings never go unseen. Those two facts catch every injected fault here. A small
  tree trained on synthetic incidents also learns to ignore natural null-device drift.
- Jev diagnoses correctly but over-repairs. A gate that applies a fix to every batch is not a diagnostic.
- Silent join breakage is the costly fault for B (0.007 log loss, about three quarters of B's 0.0097 gain over A).
  That is worth a monitor, not a Jev call.

## Limits

- The faults are synthetic and few: two kinds, on in-memory copies. Real incidents are messier, and their text
  would be harder to keep neutral.
- The incidents reuse 10 blocks, four faults each, so they are not independent. The intervals resample incidents.
- The window was already inspected, so everything is exploratory. Clicks enter only the loss.
- Redundant repairs cost nothing in loss here; the false-repair rate is the only place they show.
- One wording, asked once.

Full suite: 58 passed in 9.20s at 9086c93
