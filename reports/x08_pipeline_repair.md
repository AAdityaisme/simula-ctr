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
