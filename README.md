# simula-ctr

CTR prediction and candidate ranking for contextual ads inside AI companion apps.
Take-home for Simula. Work in progress; see PRs for the build order.

## Run

```bash
uv sync

# Put the supplied files at data/impressions.csv and data/characters.csv.
uv run python cli.py evaluate --baseline

# Run the committed 5,000-row fixture instead.
uv run python cli.py evaluate --baseline --smoke

uv run pytest -q
```
