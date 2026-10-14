# Can we assign a character's safety tier from its description?

Branch `exp/ad-classify`, written 2026-10-14. Reproduce: `uv run python -m simula.safety_classify` (it reuses the cached LLM labels in `reports/ad_classify_llm_labels.json`; `--relabel` calls the LLM again).

> **Corrections 2026-10-14 01:35 PT (after Aadi's review and second opinions from Fable 5.1 and GPT-6 Astra). The original text below is left as written.**
> 1. **The question was the wrong one.** `safety_tier` is independent of everything in the file (Fable: chi² p=0.89 against genre, p=0.52 against sentence frame). So the null was settled before any model ran. A permutation check would have shown it in ten minutes. The tier is a policy input to a hard gate, not something a model should guess. The follow-up asks the question the data can answer: does a text classifier reading the description help predict clicks? See `reports/jev_ctr.md`.
> 2. **I had the costly error backwards.** In `rank.py:75-77` the character tier is a **ceiling** on ad content. An unknown character gets `sfw`, the strictest ceiling, and `mature` allows the most ads. So missing a mature character only narrows the ads shown. The dangerous error is calling an `sfw` character `mature`, because that lets mature ads run next to it. "Recall on mature" was the wrong headline metric. The sentence under "What it means" claiming a miss "would put brand-sensitive ads next to mature characters" is wrong for this gate.
> 3. **The fallback was backwards too.** `safety_classify.py` mapped an unparsed LLM answer to `mature` and called that fail-closed. That opens the gate. It now maps to `sfw`. The single `mature` among the 300 cached Haiku labels may be that fallback. The cache does not record which.
> 4. **Wrong tool, as Aadi said.** Haiku through `claude -p` returns one label with no probabilities. Jev (TypeSafe) is built for typed classification: it gives per-option probabilities in about 0.3 s. The follow-up uses it. On tier, Jev also cannot beat chance, because the label is noise (numbers in `reports/jev_ctr.md`).

## Question

The ranking policy in `simula/rank.py` uses a character's `safety_tier` (`sfw`, `suggestive`, `mature`) to decide which ads may run next to it. A new character has no tier yet. Can a classifier assign one? The setup is "an LLM labels, a cheap LightGBM model serves."

## Short answer: no, not on this data

The tier is not in the text. Every arm scores at or below the "always guess sfw" baseline. Kappa is about zero for all of them. Recall on `mature`, the costly miss, is close to zero for every arm. This is an honest null result about the data, not a bug in the models. The test shows the same LightGBM pipeline scores 100% when the text does carry the tier.

Why the text is empty of signal:
- 417 descriptions appear more than once. 229 of them carry **different** tiers on different characters. The same sentence can be `sfw` on one character and `mature` on another.
- `creator_type` and `num_interactions` are flat across tiers. The median interactions are 177 to 180 for every tier. The share of `mature` is 15% for both community and official creators.
- The strongest single word lifts the `mature` rate from 15% to 20%, on 138 rows. That is consistent with noise.
- The tier mix is stable month to month, so the date split adds no shift.

The tiers look randomly assigned on top of templated text (the take-home README says the text is templated).

## Data and split

- `characters.csv`: 5,000 characters, 59% `sfw`, 26% `suggestive`, 15% `mature`. Read in place, not copied.
- Split by `created_at`: the oldest 4,000 (2014-01-01 to 2014-09-02) train, the newest 1,000 (2014-09-02 to 2014-10-29) test.
- LLM budget: 300 descriptions. 100 random test rows to score the LLM, and 200 random train rows for it to label as training data.

## Results

Every arm on the same 100 test rows the LLM saw. Full-test numbers are in brackets where they exist.

| Arm | Accuracy | Kappa | Recall sfw | Recall suggestive | Recall mature |
| --- | --- | --- | --- | --- | --- |
| 1. Majority (always `sfw`) | 0.65 [0.62] | 0.00 [0.00] | 1.00 | 0.00 | 0.00 |
| 2. LLM (Haiku 5.5) | 0.54 | 0.04 | 0.71 | 0.33 | **0.00** |
| 3. LightGBM, true labels, 4,000 rows | 0.65 [0.59] | 0.07 [0.03] | 0.95 [0.91] | 0.13 [0.11] | 0.00 [0.01] |
| 3b. LightGBM, true labels, same 200 rows as arm 4 | 0.57 [0.51] | 0.08 [-0.02] | 0.75 [0.75] | 0.29 [0.18] | 0.09 [0.05] |
| 4. LightGBM, LLM labels, 200 rows | 0.55 [0.51] | 0.01 [-0.02] | 0.77 [0.74] | 0.21 [0.23] | 0.00 [0.00] |

Arm 3b is a matched control I added. It shows that arm 4's gap from arm 3 comes from the smaller training set, not only from the LLM labels. With no signal to learn, the two make no real difference.

Features for LightGBM: binary word and word-pair counts of the description, an `official` flag, and log `num_interactions`.

On its own 200 train rows, the LLM agreed with the true tier 53% of the time, with kappa 0.03. It called only **1 of 300** characters `mature`, so it almost never flags the tier that matters.

## LLM cost and time

- Model `claude-haiku-5-5` via headless `claude -p`, 25 descriptions per call, 12 calls.
- 300 descriptions took 145 s, or about 0.5 s each. The CLI reported a notional cost of $0.14 ($0.00045 per description). The calls ran on the subscription, so no money was spent.
- Exact prompt (descriptions were appended as `id: description` lines):

```
You label role-play characters for an ad-safety gate. Tiers:
- sfw: fine next to any ad, no romantic, sexual or dark-adult tone.
- suggestive: flirtatious, romantic, dominant or edgy undertones, nothing explicit.
- mature: adult-oriented; sexual, intense or dark themes an advertiser would avoid.
Reply with ONLY a JSON object mapping each id to one tier, e.g. {"a1": "sfw"}.
```

The tier definitions are my guess. The dataset does not say how its tiers were defined.

## What it means for an ad safety gate

- On this data, do not let any classifier set the tier. A model that misses `mature` almost every time would put brand-sensitive ads next to mature characters.
- Keep what `rank.py` already does: an unknown tier fails closed to the strictest tier until a real label arrives. This experiment is the evidence for that choice.
- On real data, where descriptions do carry the tier, the pattern still holds up. Have the LLM label, train a cheap model to serve, and send uncertain or new cases to the LLM or a human. Pick thresholds for **recall on `mature`**, not accuracy. Fail closed when unsure.

## What I could not test

- Whether the pattern works when the signal is real. These labels give a classifier nothing to learn, so the comparison between arms here is noise.
- A larger LLM, few-shot prompts, or the official tier rubric. None of them can recover a label that is independent of the text.
- Calibrated "uncertain, so escalate" thresholds. With kappa near zero, every case is uncertain.
- The 100-row LLM sample is small: ±10 points of accuracy at 95%.
