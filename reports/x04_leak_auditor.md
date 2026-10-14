# x04: can Jev veto leaky and unavailable features from their definitions?

Branch `exp/x-judge`, 2026-10-14. Batch X, worker B. Code: `simula/x04_leaks.py`. Design: Fable E2 merged with Astra 1. Pre-registration: `reports/x_judge_prereg.md`, v1 commit `4afd4c6`, v2 commit `a46fbf2`.

**v2 is the current result.** The red team on `1d613cd` failed the gate (`redteam/rank-jev/rt-exp-x-judge-1d613cd.md`). v1 is kept below, marked superseded, with the reasons.

## Question

Before a feature reaches B, someone has to decide whether it may be used at all. Can Jev read a field's definition and four statistics and say whether the field leaks the outcome or uses information that does not exist yet? It counts only if it does this with the field's name hidden, and does it better than plain rules.

## v2 (current)

### What changed from v1, and why

- **Prereg order.** v1 wrote its card statistics, including MI with click, 46 s before the v1 prereg commit. The v2 prereg (`a46fbf2`) was committed before the v2 cards or any statistic were built. Then all 225 states were asked again. v1 answers are not reused.
- **Three legal cards rebuilt.** In v1, `creative_ctr_earlier_days` and `character_ctr_earlier_days` were smoothed toward the CTR of the whole Oct 21–27 window, so earlier rows saw later clicks. `creative_ctr_previous_7_days_decile` ranked the whole window, so later rows moved earlier deciles. All three were labelled `safe` without being safe.
  - In v2 both smoothed CTRs pull toward the overall click rate of strictly earlier days. Rows on Oct 21 are missing.
  - The decile takes its cutoffs from rows of strictly earlier days.
  - The definitions say so. `tests/test_x_judge.py` checks that changing later clicks leaves earlier values unchanged.
- **One admission policy.** A field may be used iff three things hold: it is known when the ad is chosen, it is not derived from clicks not yet known, and it is not barred by the identity policy.
  - v1's denylist control applied the whole never-list. That list also holds modelling choices this policy admits: `id`, `character_name`, `character_description`, `created_at` and `conversation_turn`. Five of that control's nine shown-arm false rejections were these raw columns, and the other four were their legal transforms.
  - The v2 denylist reads only the entries the policy bars: `click`, `session_msg_count`, `num_interactions`, `device_id` and `device_ip`.
  - The gold labels are unchanged: 44 safe, 13 leaky, 13 unavailable, 4 identity and 1 probe.
  - The truth for the two known leaks is the README's availability judgment, adopted as policy, not proven by construction.

Everything else is as in v1: cards, statistics, question, arms, masked names, controls and win.

### Results

Recall is over the 26 leaky or unavailable cards, and false rejection is over the 44 safe cards. Each row here is a card. Intervals are a paired bootstrap over cards, 2,000 draws.

| Arm | Recall | False rejection | 3 plants | 2 known leaks | Jev − arm, recall, pp (95% CI) | Jev − arm, false rejection, pp (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| **Jev, masked** | **69.2%** | **6.8%** | yes | **no** | | |
| MI-outlier | 30.8% | 6.8% | no | no | +38.5 (+15.4 to +57.8) | 0.0 (−11.4 to +11.4) |
| Denylist + lineage | 88.5% | 0.0% | yes | no | −19.2 (−38.5 to 0.0) | +6.8 (0.0 to +15.9) |
| Lineage only | 69.2% | 0.0% | yes | no | 0.0 (−15.4 to +15.4) | +6.8 (0.0 to +15.9) |
| Union | 92.3% | 6.8% | yes | no | −23.1 (−38.5 to −7.7) | 0.0 (−11.4 to +11.4) |
| Jev, names shown | 76.9% | 2.3% | yes | no | | |
| Denylist + lineage, names shown | 100% | 0.0% | yes | yes | | |

**Pre-declared win: not met. Verdict: null.**

- Jev caught all three plants.
- It caught neither known leak. On `session_msg_count` it answered "safe" with P = 0.59 (masked) and 0.54 (shown); on `num_interactions`, P = 0.51 and 0.66.
- Its recall, 69.2%, is below the best eligible control. That control is now denylist + lineage, at 88.5% with no false rejections, because the aligned policy no longer charges it for modelling exclusions.
- It rejected 6.8% of legal cards, above the 2% limit. These were the previous-day creative CTR, the rebuilt 7-day CTR decile and the previous-day app impression count.

**The rebuilt cards.**
- Jev called both rebuilt smoothed CTRs "safe" in every arm, the same answers as in v1.
- It called the rebuilt 7-day decile "leaky" in every arm, as in v1. In v1 that card really did use later rows, so the v1 table counted a correct rejection as a false one. In v2 the card is legal and the rejection is a real error.

**Jev's run-to-run noise.** Between v1 and v2, 4 of the 72 cards whose state did not change got a different masked answer (5.6%). Examples: `remaining_messages` moved from unavailable to safe, and `app_ctr_previous_day` from leaky to safe. Reversing the option order changed 8% of masked answers. Masking changed 5.3% of answers.

**Fable's probe.** Jev called the same-day `device_ip` counter "safe" in all three arms, as in v1.

**Jev cost (v2).** 225 calls with 0 errors, 145,117 input tokens, $0.0061. `jev-latest` again resolved to `jev-1.13.0`. The answers are in `reports/x04_leak_auditor_v2_jev_answers.jsonl`, and the scores are in `reports/x04_leak_auditor_v2.json`.

## v1 (superseded)

Superseded for three reasons:
- the card statistics predate the prereg commit;
- three cards labelled `safe` used whole-window information;
- the denylist control applied a wider policy than the gold labels.

The v1 numbers below are kept unchanged. The v1 card statistics and answers are in `reports/x04_leak_auditor.json` and `reports/x04_leak_auditor_jev_answers.jsonl`; the v1 code is at `1d613cd`.

### Setup

- **Cards.** There are 75 cards in total.
  - All 33 real columns of the two CSVs. Each has its definition from the assignment schema.
  - 41 constructed recipe cards with truth fixed in code:
    - Fable's three plants: click plus noise, a same-day device total, and click shuffled within each day.
    - Astra's future aggregates of clicks and of counts, such as a creative's CTR over the same day or over the next 24 hours.
    - Nested aliases of leaks, such as the decile of a same-day CTR, or log(1 + `num_interactions`).
    - Legal lagged versions, such as the previous day's CTR, and legal transforms of real columns.
    - Two identity recipes.
    - One unscored probe: Fable's same-day `device_ip` counter from earlier hours. Its truth is a policy call.
- **Truth.** 44 safe, 13 leaky, 13 unavailable at decision time, 4 identity-restricted and 1 probe. The two real leaks, `session_msg_count` and `num_interactions`, are "unavailable": the first is a session total, the second an export total. `click` itself is "leaky".
- **Statistics.** Each card is computed on the refit window, Oct 21–27, and nothing later. Fable's four statistics are on the card: MI with click, Spearman with the character's impression count, share of rows above `conversation_turn`, and cardinality.
- **Label set.** Fable's four labels: safe, leaky, unavailable_at_decision_time, identity_restricted. Each label states a fact about how a field is built, so truth follows from the recipe. Astra's `lag_one_day` is a remedy; a future aggregate could rightly be lagged or dropped, so its truth would be ambiguous. The win is binary: anything but `safe` is a rejection.
- **Arms.**
  - Names shown.
  - Names masked: the field is renamed `f01`–`f74` in a fixed shuffled order, and the definition is kept.
  - Masked with the option order reversed. This is Astra's shuffled-option check.
- **Controls.**
  - **MI-outlier** (Fable): reject when the modified z-score of MI across all cards is above 3.5.
  - **Denylist + lineage** (Astra): the existing never-list (`CONTRACT["never"]`), applied by name and through a recipe's source columns, plus a timestamp rule. The timestamp rule rejects a recipe over rows at or after the impression, or one that reads click from rows whose labels are not yet in. In the masked arm the denylist cannot see a real column's own name.
  - **Lineage only**: the timestamp rule without the denylist.
  - **Union** of MI-outlier and denylist + lineage.
- A control counts as the "best control" only if it also rejects at most 2% of legal cards.

### Jev

`jev-latest` resolved to `jev-1.13.0`. One question, written in code before any call and not changed afterwards. This is the only version.

- `admission` is a Choice. Instruction: "May this field be used as an input to the click model, which predicts P(click) at the moment the ad is chosen? Pick the one label that fits."
- Options:
  - `safe`: "may be used: known at the moment the ad is chosen and not derived from outcomes that are not yet known".
  - `leaky`: "derived from click outcomes (this impression's or others') that are not yet known when the ad is chosen".
  - `unavailable_at_decision_time`: "uses information that only exists after the ad is chosen (later events, session or export totals), though not click outcomes".
  - `identity_restricted`: "identifies or tracks an individual user or device; barred by the identity policy".
- **Calls.** 225 calls in 225 attempts with 0 errors: 75 cards × 3 arms. They took 2 s of wall time on 8 threads (19 s of summed call time). They used 145,015 input tokens, or $0.0061. The answers are in `reports/x04_leak_auditor_jev_answers.jsonl`.

One example state: the masked `session_msg_count` card.

```json
{"context": "Simula serves ads inside AI companion apps. A click model predicts P(click) for an impression at the moment the ad is chosen, from fields known at that moment. The log has one row per impression, with an hour (UTC, YYMMDDHH) and a click outcome. Click outcomes for a day become available at the end of that day. Each impression joins to a character table. Statistics on the card are computed on Oct 21-27.",
 "field": {"name": "f02", "definition": "Total messages in the session.",
           "statistics": {"mi_with_click": 7e-05, "spearman_with_character_impressions": 0.001,
                          "share_above_conversation_turn": 0.78, "cardinality": 137}}}
```

### Results

Recall is over the 26 leaky or unavailable cards. False rejection is over the 44 safe cards. Intervals are a paired bootstrap over cards, 2,000 draws. The rows here are cards, not impressions.

| Arm | Recall | False rejection | 3 plants | 2 known leaks | Jev − arm, recall, pp (95% CI) | Jev − arm, false rejection, pp (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| **Jev, masked** | **73.1%** | **6.8%** | yes | **no** | | |
| MI-outlier | 30.8% | 6.8% | no | no | +42.3 (+19.2 to +61.5) | 0.0 (−11.4 to +11.4) |
| Denylist + lineage | 88.5% | 9.1% | yes | no | −15.4 (−30.8 to 0.0) | −2.3 (−13.6 to +9.1) |
| Lineage only | 69.2% | 0.0% | yes | no | +3.8 (−11.5 to +19.2) | +6.8 (0.0 to +13.7) |
| Union | 92.3% | 15.9% | yes | no | −19.2 (−34.6 to −3.8) | −9.1 (−22.7 to +4.5) |
| Jev, names shown | 76.9% | 2.3% | yes | no | | |
| Denylist + lineage, names shown | 100% | 20.5% | yes | yes | | |

**Pre-declared win (v1): not met.** The win needed all four conditions on the masked arm.

- Jev caught all three plants. This condition is met. It is the only control-free part of the win.
- It caught neither known leak. With names masked and with names shown, it called both `session_msg_count` and `num_interactions` "safe". Its probabilities were close: 0.60 / 0.39 for `session_msg_count` and 0.48 / 0.44 for `num_interactions`, safe against unavailable.
- Its recall, 73.1%, is above the best eligible control. That control is lineage only, at 69.2%, because the others reject more than 2% of legal cards. But the gap is one card, and its interval spans 0.
- It rejected 6.8% of legal cards, which is above the 2% limit. It rejected three legal lagged CTRs when masked: the previous-day CTR of the creative and of the app, and a 7-day CTR decile.

Verdict (v1): **null**. Jev is not better than a timestamp rule at the pre-declared operating point.

**Where Jev and the rules differ.** This also holds in v2.

- Jev missed two same-day label leaks: a creative's CTR over earlier hours of the same day, and an app's CTR in the previous hour. The context says labels arrive at the end of the day, and the timestamp rule catches both. Jev seems to read "earlier" as "safe".
- Jev caught `click` and the session-total aliases (turn share of the session, messages remaining) when masked. The lineage rule cannot see these, because a raw column has no lineage once its name is hidden.
- Jev barred all four identity cards in both arms. The denylist loses the two raw ones when names are masked.
- MI-outlier misses everything without a strong marginal signal. That includes the within-day shuffled click (MI 0) and every count-based future aggregate. It also rejects `site_id`, `site_domain` and C14 × C17 for being informative.

**Checks (v1).**

- **Reversed options.** Reversing the option order changed 8% of masked answers (92% agreement).
- **Masking.** Masking changed 9.3% of answers (90.7% agreement). Masking made Jev stricter on lagged CTRs and looser on `log(1 + num_interactions)` and the session-length decile.
- **Fable's probe.** On the day-scoped `device_ip` counter from earlier hours, Jev answered "safe" in all three arms. So it bars `device_ip` itself as identity but allows a same-day counter derived from it. The README did not allow that either.

## What it means for ranking

Jev cannot serve as the gate that admits features to B. It misses the two leaks that matter most in this dataset, and it misses same-day label leaks that a one-line timestamp rule catches. Once the denylist states the same policy as the gold labels, the plain rule beats Jev by 19 points of recall with no false rejections. Where it adds something, it reads prose that the rules cannot: a masked raw column such as `click`, or the session-total aliases. A rule-first gate with Jev as a second reader on raw columns would cover more than either alone. That is a hypothesis, and I did not test it.

## Limits

- **The masked arm hides names, not provenance.** This limit comes from the red team. The definitions keep their provenance words: "click plus Gaussian noise", "same day", "next 24 hours", `session_msg_count`, `num_interactions`. Those often match the label descriptions directly. So the masked arm tests whether Jev reads explicit provenance prose. Catching the plants is not evidence that Jev detects hidden lineage. A stronger blindness claim needs neutral paraphrases, or an equally informed text-rule control, under a new prereg.
- I wrote both the recipes and the lineage metadata the rules read, so the rules are only as good as that metadata. A real pipeline would need someone to keep it.
- The two real leaks have one-line definitions taken verbatim from the assignment schema. "Total messages in the session" does not say when the total is taken, and the README itself calls `session_msg_count`'s availability ambiguous. A fuller definition might change Jev's answer. I did not try one after seeing the results.
- 75 cards is a small set, and the intervals are wide. One card is 3.8 points of recall.
- This tests feature admission, not CTR. The test window (Oct 29–30) was not read, and nothing here is a CTR lift. The test window was already inspected by earlier work, so this is exploratory.

Full suite: 48 passed, 2 warnings in 9.92s at bab7abc
