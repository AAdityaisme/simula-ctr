"""One-off offline precompute: Jev (TypeSafe) reads each character description, no name, and answers typed
questions. Writes reports/jev_characters.csv (one row per character_id). Never a runtime dependency.

Run (throwaway env, key loaded from the agent repo's .env at runtime, never printed):
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv --with pandas python jev_annotate.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

CHARACTERS = Path.home() / "Desktop/Simula/ml-takehome/data/characters.csv"
ENV = Path.home() / "simula-local/next-duolingo/.env"
OUT = Path("reports/jev_characters.csv")
CACHE = Path(sys.argv[1] if len(sys.argv) > 1 else "jev_cache.json")
RUNGS = ["not at all", "slightly", "moderately", "strongly", "extremely"]
GENRES = {
    "anime": "anime-style character", "comedic": "comedy", "fantasy": "fantasy", "historical": "historical",
    "horror": "horror or supernatural", "mentor": "mentor or coach", "mystery": "mystery or detective",
    "romance": "romance", "sci": "science fiction", "slice": "slice of life, everyday companion",
}
QUESTIONS = {
    "genre": Choice(instructions="Which genre does this role-play character belong to?", criteria=GENRES),
    "romance": Score(instructions="How romantic or flirtatious is this character?", criteria=RUNGS),
    "darkness": Score(instructions="How dark, eerie or menacing is this character?", criteria=RUNGS),
    "warmth": Score(instructions="How warm, friendly and comforting is this character?", criteria=RUNGS),
    "brand_safe": Noul(instructions="A mainstream consumer brand would be comfortable running an ad next to "
                                    "this character."),
    "tier": Choice(instructions="Which ad-safety tier fits this role-play character?",
                   criteria={"sfw": "fine for all audiences", "suggestive": "flirtatious or edgy undertones",
                             "mature": "adult, sexual or dark themes"}),
}


def ask(client, description):
    for attempt in (1, 2):
        try:
            r = client.system_one({"character_description": description}, QUESTIONS, model="jev-latest", timeout=30)
            a = r.answers
            row = {f"jev_genre_{g}": a["genre"].probabilities.get(g, 0.0) for g in GENRES}
            row.update({f"jev_{q}": a[q].score for q in ("romance", "darkness", "warmth")})
            row["jev_brand_safe"] = a["brand_safe"].noul
            row.update({f"jev_tier_{t}": p for t, p in a["tier"].probabilities.items()})
            return {**row, "jev_model": r.model, "tokens_in": r.usage.input_tokens or 0}
        except Exception as e:  # noqa: BLE001 - one retry, then surface
            if attempt == 2:
                raise
            print("retry:", type(e).__name__, str(e)[:120])
            time.sleep(2)


def main():
    load_dotenv(ENV, override=False)
    chars = pd.read_csv(CHARACTERS, dtype={"character_id": str})
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [d for d in chars["character_description"].unique() if d not in cache]
    start = time.monotonic()
    with TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]) as client, ThreadPoolExecutor(8) as pool:
        for i, (d, row) in enumerate(zip(todo, pool.map(lambda d: ask(client, d), todo)), 1):
            cache[d] = row
            if i % 500 == 0:
                CACHE.write_text(json.dumps(cache))
                print(i, "done", round(time.monotonic() - start), "s")
    CACHE.write_text(json.dumps(cache))
    seconds = time.monotonic() - start
    tokens = sum(cache[d]["tokens_in"] for d in todo)
    rows = pd.DataFrame([{"character_id": c, **cache[d]} for c, d in
                         zip(chars["character_id"], chars["character_description"])])
    rows.drop(columns="tokens_in").to_csv(OUT, index=False, float_format="%.4f")
    print(f"{len(todo)} calls, {seconds:.0f}s, {tokens} input tokens, ${tokens * 0.042e-6:.4f}, "
          f"models {rows['jev_model'].unique().tolist()}")


if __name__ == "__main__":
    main()
