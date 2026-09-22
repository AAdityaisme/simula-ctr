# simula-ctr

CTR prediction and candidate ranking for contextual ads inside AI companion apps.
Take-home for Simula. Work in progress; see PRs for the build order.

Code is written with AI coding tools (Codex, Claude Code). The design decisions, feature contract, evaluation choices, and analysis are mine.

## Run

```bash
uv sync

# Put the supplied files at data/impressions.csv and data/characters.csv.
uv run python cli.py evaluate --baseline
uv run python cli.py train --out bundle
uv run python cli.py evaluate --model bundle
uv run python cli.py rank --requests fixtures/rank_requests.json --model bundle --out reports/rank_sample.json
uv run python cli.py drift --model bundle

# Run the committed 5,000-row fixture instead.
uv run python cli.py evaluate --baseline --smoke
uv run python cli.py train --smoke --out bundle
uv run python cli.py evaluate --model bundle --smoke
uv run python cli.py drift --model bundle --smoke

uv run pytest -q
```

The drift report uses in-sample predictions on training days only to show error moving
with the traffic mix; those rows are not held-out metrics. Because the data has no
label-availability timestamps, it assumes a completed day's labels are usable from
00:00 the next day. The daily intercept prototype can adjust probability levels after
labels arrive, but its monotonic shift cannot change candidate order.

## Ranking payload

`rank` reads a JSON list of request payloads. Each payload contains request context,
a bounded candidate list, a publisher content ceiling, a seed, and optional exposure
state:

```json
{
  "id": "request-1",
  "request": {
    "hour": 14102900, "character_id": "...", "site_id": "...", "site_domain": "...",
    "site_category": "...", "app_id": "...", "app_domain": "...", "app_category": "...",
    "device_id": "...", "device_ip": "...", "device_model": "...", "device_type": 1,
    "device_conn_type": 0, "C1": 1005
  },
  "candidates": [{
    "candidate_id": "ad-1", "content_tier": "sfw", "banner_pos": 0,
    "C14": 20345, "C15": 300, "C16": 250, "C17": 2331,
    "C18": 2, "C19": 39, "C20": "-1", "C21": 23
  }],
  "publisher": {"max_content_tier": "suggestive"},
  "exposure": {"key": "opaque", "window": "24h", "counts": {"ad-1": 3}},
  "seed": 7
}
```

`conversation_turn` and `session_msg_count` are optional echoed request fields.

Each result reports the selected candidate or no-fill, the effective safety ceiling,
degraded-state and version metadata, and ranked candidates with exclusions, raw and
calibrated pCTR, exposure-adjusted utility, novelty flags, exploration membership,
and the actual selection probability.
