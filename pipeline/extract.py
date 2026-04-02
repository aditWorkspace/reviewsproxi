"""Stage 2: Batched signal extraction with crash-safe resume."""
from __future__ import annotations

import json
from pathlib import Path

from engine.extract import extract_signals_batch
from engine.llm import get_client

REVIEWS_PATH = Path("data/raw/reviews.jsonl")
SIGNALS_PATH = Path("data/extracted/signals.jsonl")


def _count_completed_batches() -> int:
    if not SIGNALS_PATH.exists():
        return 0
    with open(SIGNALS_PATH) as f:
        return sum(1 for _ in f)


def _build_batches(reviews: list[dict], batch_size: int) -> list[list[dict]]:
    return [reviews[i: i + batch_size] for i in range(0, len(reviews), batch_size)]


def run_extract(batch_size: int = 30, force: bool = False) -> int:
    """Extract signals from all reviews, resuming from last completed batch.

    Returns the number of new batches processed.
    """
    if force and SIGNALS_PATH.exists():
        SIGNALS_PATH.unlink()

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    completed = _count_completed_batches()

    reviews: list[dict] = []
    with open(REVIEWS_PATH) as f:
        for line in f:
            reviews.append(json.loads(line))

    batches = _build_batches(reviews, batch_size)
    remaining = batches[completed:]

    if not remaining:
        print(f"[extract] All {completed} batches already processed.")
        return 0

    print(f"[extract] Resuming from batch {completed + 1}/{len(batches)}")
    client = get_client()
    count = 0

    with open(SIGNALS_PATH, "a") as out:
        for i, batch in enumerate(remaining, start=completed + 1):
            print(f"[extract] Batch {i}/{len(batches)} ({len(batch)} reviews)...")
            signals = extract_signals_batch(batch, "student_products", client)
            out.write(json.dumps(signals) + "\n")
            out.flush()
            count += 1

    return count
