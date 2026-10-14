# x07: Can Jev approve counterfactual edits of an ad?

Branch `exp/x-evidence`, 2026-10-14. Code: `simula/x07_surgery.py`. After Astra's experiment 6.

## Pre-registration (committed before any outcome-bearing run)

- **Constraints, refit only.** X → Y is a dependency when every refit (Oct 21–27) value of X has exactly one Y.
  Re-derived on refit: C14 → C15, C16, C17, C21, and C17 → C21. These match the full-data check for C14.
- **Items.** One-field edits of a held-out ad tuple (`banner_pos`, C14–C21) from Oct 29–30.
  - 300 real: the edit turns one observed test tuple into another observed test tuple. 60 per field, over the five
    fields that ever differ alone (`banner_pos`, C14, C18, C19, C20).
  - 300 broken: the edit swaps a dependent field (C15, C16, C17 or C21) for another refit value. Three families of
    100: C14 seen in refit; C14 new but C17 seen (the edit is to C21, so only C17 → C21 binds); C14 and C17 both
    new. Every broken result is asserted absent from the full data.
- **Jev**, Choice `operation`: "Which operation does the evidence support for this proposed edit?" Options
  `single_field` ("the edit alone gives a coherent ad"), `dependency_closure`, `reject`, `unknown`. Score =
  P(single_field).
  - State: field notes, the learned dependencies in words, the original tuple, the proposed field and value, refit
    counts for the result tuple and for each of its values, refit co-occurrence of the new value with each other
    field, and for each dependency touching the edited field, the refit dependent values of the result's
    determinant with counts.
  - Recheck: 50 edits again with the options in reverse order.
- **Controls.** Exact-tuple lookup (accept iff the result was seen in refit). Empirical dependency closure (accept iff
  no learned dependency with a refit-seen determinant is violated).
- **Win:** at closure's detection rate on broken items, Jev accepts a larger share of real items, bootstrap interval
  over items above 0. Jev at the exact-tuple detection rate is reported too.

The Oct 29–30 window was already inspected, so everything is exploratory.

## Results

`jev-latest` resolved to `jev-1.13.0`. One version of the question and state. 650 calls in 650 attempts with 0
errors, 7 s of wall time, 608,399 input tokens, $0.026. Answers: `reports/x07_jev_answers.jsonl`.

| Arm | Real edits accepted | Broken edits detected | C14 seen | C17 only | both new |
| --- | --- | --- | --- | --- | --- |
| Exact-tuple lookup | 22.0% | 100% | 100% | 100% | 100% |
| Dependency closure (pre-registered) | **100%** | 78.0% | 100% | 100% | 34% |
| Jev, top choice `single_field` | 87.7% | 96.3% | 98% | 100% | 91% |
| Closure with the original row (**post hoc**) | **100%** | **100%** | 100% | 100% | 100% |

Jev at each control's detection rate (bootstrap over the 600 items, 2,000 resamples):

| Matched to | Detection | Control accepts | Jev accepts | Jev − control (95% CI) |
| --- | --- | --- | --- | --- |
| Dependency closure (pre-registered) | 78.0% | 100% | 97.0% | −3.0 pp (−6.3 to −1.0) |
| Exact-tuple lookup | 100% | 22.0% | 25.3% | +3.3 pp (−2.3 to +23.9) |
| Closure with the original row (post hoc) | 100% | 100% | 25.3% | −74.7 pp (−79.2 to −54.6) |

- **Pre-declared win not met; this is a loss.** At closure's detection rate, Jev accepts 3 points fewer real edits,
  and the interval excludes 0.
- Jev's AUROC between real and broken edits is 0.965. Its top choices: on real edits, 263 `single_field`, 33
  `dependency_closure`, 4 `reject`; on broken edits, 204 `reject`, 82 `dependency_closure`, 11 `single_field`, 3
  `unknown`. Reversing the options kept the top choice on 46 of 50 rechecks (92%).
- Jev's errors on real edits are almost all C14 edits: it accepts only 40% of them, and 98–100% of the other four
  fields. It treats a change of creative as needing its dependents to change, even when they already match.
- **Where Jev looked better, my control was weak.** On the 100 broken edits whose C14 and C17 are both new, the
  pre-registered closure caught only 34%. Jev caught 91%. But the closure I registered checks the result against the
  refit mapping only. It ignores that the original row is real. Editing a dependent field (C15, C16, C17 or C21)
  while its determinant stays fixed always breaks X → Y, whether or not the determinant was seen. Closure that uses
  the original row, added after I saw the outcomes, scores 100% and 100%. Jev cannot beat it.
- This also exposes a design limit. In this data a dependent field never changes alone in a real tuple, so the
  edited field's name separates real from broken edits on its own. The task tests whether a judge applies the
  dependency list. It does not test anything a rule cannot do.

Example state (`edit:368`, a broken edit to C17; the dependency list is shortened here):

```json
{"field_notes": "Anonymized categorical ad attributes of one ad slot. C14 is the creative id by convention; the others are undocumented.",
 "training_dependencies": ["each C14 value had exactly one C15 value", "...", "each C17 value had exactly one C21 value"],
 "original": {"banner_pos": "0", "C14": "20633", "C15": "320", "C16": "50", "C17": "2374", "C18": "3", "C19": "39", "C20": "100119", "C21": "23"},
 "proposal": {"field": "C17", "new_value": "2533"},
 "result_training_impressions": 0,
 "value_training_impressions": {"banner_pos": 531339, "C14": 5864, "C15": 679433, "C16": 684947, "C17": 461, "C18": 258670, "C19": 167823, "C20": 2344, "C21": 151224},
 "pairs_with_edited_field": {"banner_pos": 0, "C14": 0, "C15": 461, "C16": 461, "C18": 0, "C19": 0, "C20": 0, "C21": 0},
 "dependency_evidence": [
  {"rule": "C14 -> C17", "determinant": "20633", "determinant_training_impressions": 5864, "dependent_values_in_training": {"2374": 5864}, "proposed_dependent": "2533"},
  {"rule": "C17 -> C21", "determinant": "2533", "determinant_training_impressions": 461, "dependent_values_in_training": {"48": 461}, "proposed_dependent": "23"}]}
```

## What it means for ranking

- Robustness tests for B (swap one ad field, see how the score moves) should be gated by dependency closure computed
  from the original row. That rule is exact here and costs nothing. Jev adds errors, mainly by blocking creative
  swaps that are coherent.
- Jev does reason with the dependency list: it caught 91% of the violations my weak control missed. That is a
  sign the typed judge reads structure. It is not a reason to put it in the loop, because the right rule is simpler.

## Limits

- The items are synthetic edits of real held-out tuples. The window was already inspected, so everything is
  exploratory. Nothing here touches CTR.
- An unseen tuple is not an impossible ad. "Real" here means observed in Oct 29–30; "broken" means it violates a
  dependency that holds on the full data.
- The post-hoc control was added after outcomes and is reported as such, next to the pre-registered one.
- One wording, asked once.
