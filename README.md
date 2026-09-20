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

# Run the committed 5,000-row fixture instead.
uv run python cli.py evaluate --baseline --smoke
uv run python cli.py train --smoke --out bundle
uv run python cli.py evaluate --model bundle --smoke

uv run pytest -q
```
