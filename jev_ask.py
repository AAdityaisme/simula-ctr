"""Generic JSONL runner for Jev (TypeSafe) typed questions. Offline only, never a runtime dependency.

Each input line is {"id", "state", "questions": {name: {"type": "choice"|"score"|"noul", "instructions",
"criteria"}}}. Each output line is {"id", "model", "tokens_in", "seconds", "answers"}, or {"id", "error"} after one
retry. Ids with a successful line in OUT are skipped, so a run resumes where it stopped and retries errors. OUT is
append-only: readers keep the last line per id.

Run (throwaway env, key loaded from the agent repo's .env at runtime, never printed):
  uv run --no-project --with typesafe-sdk==0.7.2 --with python-dotenv python jev_ask.py IN.jsonl OUT.jsonl
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

ENV = Path.home() / "simula-local/next-duolingo/.env"
TYPES = {"choice": Choice, "score": Score, "noul": Noul}
FIELDS = {"choice": ("choice", "confidence", "probabilities"), "score": ("score", "confidence", "probabilities"),
          "noul": ("noul",)}


def questions(spec):
    """Turn the JSON question spec into SDK question objects."""
    return {name: TYPES[q["type"]](instructions=q["instructions"], criteria=q.get("criteria"))
            for name, q in spec.items()}


def ask(client, item):
    """Ask one state its questions, with one retry. Returns the output line."""
    for attempt in (1, 2):
        start = time.monotonic()
        try:
            r = client.system_one(item["state"], questions(item["questions"]), model="jev-latest", timeout=30)
            answers = {name: {"type": a.type, **{f: getattr(a, f) for f in FIELDS[a.type]}}
                       for name, a in r.answers.items()}
            return {"id": item["id"], "model": r.model, "tokens_in": r.usage.input_tokens or 0,
                    "seconds": round(time.monotonic() - start, 3), "answers": answers}
        except Exception as e:  # noqa: BLE001 - one retry, then record the error and move on
            if attempt == 2:
                return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
            time.sleep(2)


def main(src, dst):
    load_dotenv(ENV, override=False)
    dst = Path(dst)
    rows = map(json.loads, dst.read_text().splitlines()) if dst.exists() else []
    done = {row["id"] for row in rows if "error" not in row}
    todo = [item for item in map(json.loads, Path(src).read_text().splitlines()) if item["id"] not in done]
    start, tokens, errors = time.monotonic(), 0, 0
    with TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]) as client, ThreadPoolExecutor(8) as pool, \
            dst.open("a") as out:
        for row in pool.map(lambda item: ask(client, item), todo):
            out.write(json.dumps(row) + "\n")
            out.flush()
            tokens += row.get("tokens_in", 0)
            errors += "error" in row
    print(f"{len(todo)} calls, {time.monotonic() - start:.0f}s, {tokens} input tokens, ${tokens * 0.042e-6:.4f}, "
          f"{errors} errors")


if __name__ == "__main__":
    main(*sys.argv[1:3])
