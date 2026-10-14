# Does a text classifier reading the character help predict clicks?

Branch `exp/ad-classify`, 2026-10-14. This replaces the safety-tier experiment in `reports/ad_classify.md`, which asked a question the data cannot answer (see the corrections at its top). The design merges two independent reviews: Fable 5.1 (`~/simula-local/reviews/ad-classify-2026-10-14/report-fable.md`) and GPT-6 Astra (`report-astra.md`).

## Question

Aadi's idea: "an ad classification system with a classifier AI in there plus LightGBM." Ads here have no text, only anonymized C-fields, so ads themselves cannot be classified. The classifier reads the **character**. Jev (TypeSafe's typed-decision model) reads each description and answers typed questions. Its answers become numeric features in the submitted LightGBM CTR model, and the model is judged on real clicks.

The test that matters: a brand-new character in production has a description but no curated genre. Model B's genre comes from a name prefix (`romance_…`), which is a hack. Can Jev reading the description replace it? Does it add anything beyond it?

## Setup

- **Jev:** `jev-latest` (resolved to `jev-1.13.0`). The input is the description only, never the name. Six questions in one call:
  - genre: Choice over the 10 name-prefix genres, all 10 probabilities kept;
  - romance, darkness and warmth: Score on five rungs, expected rung used;
  - "a mainstream brand would be comfortable next to this": Noul probability;
  - safety tier: Choice, used as a diagnostic only.
- **Cost:** 4,548 distinct descriptions took 4,548 calls and 67 s with 8 threads. They used 3.46M input tokens, so $0.15 at $0.042 per million. Output is in `reports/jev_characters.csv`, produced by `jev_annotate.py`. It is an offline precompute, never a serving dependency.
- **CTR model:** the submitted model B recipe, unchanged. Same windows: fit Oct 21–26, tree count by early stopping on Oct 27, refit through Oct 27, calibrate on Oct 28, test on Oct 29–30 (127,406 impressions). Same LightGBM settings. Arms differ only in character features. Sanity check: the `B_submitted` arm reproduces the README's test log loss 0.409450 exactly.
- **Simple text control (Astra's):** the description's sentence frame plus its two trait adjectives, as categories, parsed with a regex. All 5,000 descriptions come from 30 frames and 30 adjectives, and each frame belongs to exactly one genre. So this control is the bar any "AI reads the text" claim has to clear.
- **Significance:** a paired bootstrap of the log-loss difference over 1,000 resamples of whole characters. This is the 95% interval below.

## Results

| Arm | Character features | Test log loss | Test AUC |
| --- | --- | --- | --- |
| A | none | 0.41919 | 0.706 |
| B (submitted) | tier, creator type, **name-prefix genre**, age | 0.40945 | 0.729 |
| B − genre | tier, creator type, age | 0.41812 | 0.709 |
| B − genre + regex text | + frame, two traits | 0.40979 | 0.728 |
| **B − genre + Jev** | + 14 Jev features | **0.40969** | **0.728** |
| B + Jev | all of B + Jev | 0.40936 | 0.729 |

| Comparison | Log-loss change | 95% interval | Reading |
| --- | --- | --- | --- |
| B − genre + Jev vs B − genre | −0.0084 | −0.0094 to −0.0075 | Jev recovers the genre signal |
| B − genre + Jev vs B | +0.0002 | −0.0002 to +0.0008 | tie: Jev replaces the name prefix |
| B − genre + Jev vs regex text | −0.0001 | −0.0006 to +0.0005 | tie: a regex does as well on this data |
| B + Jev vs B | −0.0001 | −0.0003 to +0.0001 | null: nothing beyond genre |

Genre carries almost all of the character value. Removing it gives back 0.0087 of the 0.0097 that character features earn over A. Jev, reading only the description, earns that back to within noise. It adds nothing on top of the name prefix.

The unseen-character slice has 1,478 rows from 61 characters: B scores 0.4108 and B − genre + Jev scores 0.4093. That is too small to count, so only the all-test numbers carry a claim.

**Diagnostics (not CTR results):**
- **Genre:** Jev's top genre matches the name prefix for 84.8% of characters. Its probabilities still carry the full signal.
- **Safety tier:** Jev agrees with the given tier 56% of the time, with kappa −0.003. It calls 93% of characters `sfw`. This settles the "wrong tool" question for the tier: a purpose-built classifier also scores chance, because the tier is random with respect to the text.

## What it means

- **The idea works as a cold-start path.** A typed classifier reading the persona text can replace a curated genre label for a brand-new character, at no loss in click prediction. It costs 3 cents per 1,000 characters and runs offline once per character, never per ad request.
- **This data cannot show more than that.** The descriptions are templates, so a regex recovers the same signal. Whether Jev beats simple text features needs real, free-form persona text. Whether it adds anything beyond genre needs character signal that is not already in the genre, and this export has none.
- **Keep the safety gate as policy.** The tier is an upstream label, and `rank.py` fails closed to `sfw` when it is missing. A classifier could suggest a tier for human review. In this gate, the error to guard against is calling an `sfw` character `mature`, because that widens which ads may run.

## Limits

- The Oct 29–30 test window was already seen while building the submitted model, so this is exploratory, not a clean holdout.
- No held-out set of characters was removed from training. Astra proposed one (972 characters, 24,182 test rows). It is not built.
- The Jev questions were written once, before any click results, and not tuned. Different rubrics might score differently.
- The model is trained on clicks, and nothing here is a causal CTR lift.
- The 14 Jev features enter as plain numbers. The submitted feature contract is extended at import time in `simula/jev_ctr.py`, so the submitted pipeline files are unchanged.

## Reproduce

```bash
uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv --with pandas python jev_annotate.py  # needs TYPESAFE_API_KEY
uv run python -m simula.jev_ctr   # ~80 s; writes reports/jev_ctr.json
```
