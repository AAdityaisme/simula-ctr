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
