"""Signal extraction engine.

Extracts structured behavioral signals from batches of product reviews
using the Anthropic Claude API.  Designed for high-throughput, low-cost
extraction via claude-haiku with automatic batching, retry logic, and
progress reporting.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import openai
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0  # seconds; doubles each retry

SIGNAL_SCHEMA: str = """\
{
  "pain_points": [
    {"signal": "<string>", "frequency": <int>, "emotional_intensity": <float 0-1>}
  ],
  "desired_outcomes": [
    {"outcome": "<string>", "priority": <float 0-1>}
  ],
  "purchase_triggers": [
    {"trigger": "<string>", "context": "<string>"}
  ],
  "objections": [
    {"objection": "<string>", "severity": <float 0-1>}
  ],
  "switching_triggers": [
    {"from_product": "<string>", "reason": "<string>", "threshold": "<string>"}
  ],
  "decision_factors_ranked": ["<string>"],
  "deal_breakers": ["<string>"],
  "friction_tolerance": "<low | medium | high>"
}"""

SYSTEM_PROMPT: str = (
    "You are a behavioral-signal extraction engine.  Given a batch of "
    "product reviews for a specific category, extract structured behavioral "
    "signals.  Be precise and evidence-based — every signal must be "
    "grounded in the review text.  Return ONLY valid JSON matching the "
    "schema provided.  Do NOT wrap the JSON in markdown code fences."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(reviews: list[dict], category: str) -> str:
    """Format reviews into a prompt for the LLM."""
    review_block = "\n---\n".join(
        f"[Review {i + 1}] (rating: {r.get('rating', 'N/A')})\n{r.get('text', '')}"
        for i, r in enumerate(reviews)
    )
    return (
        f"Product category: {category}\n\n"
        f"Reviews ({len(reviews)} total):\n{review_block}\n\n"
        f"Extract behavioral signals according to the following JSON schema.  "
        f"Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
        f"Schema:\n{SIGNAL_SCHEMA}"
    )


def _parse_json_response(text: str) -> dict:
    """Parse a JSON response, tolerating markdown code fences."""
    cleaned = text.strip()

    # Strip ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)
    match = fence_pattern.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Fallback: strip leading/trailing backticks
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]

    return json.loads(cleaned)


def _empty_signals() -> dict:
    """Return an empty signal dict conforming to the schema."""
    return {
        "pain_points": [],
        "desired_outcomes": [],
        "purchase_triggers": [],
        "objections": [],
        "switching_triggers": [],
        "decision_factors_ranked": [],
        "deal_breakers": [],
        "friction_tolerance": "medium",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_signals_batch(
    reviews: list[dict],
    category: str,
    client,
) -> dict:
    """Extract behavioral signals from a single batch of reviews.

    Parameters
    ----------
    reviews:
        A list of review dicts (typically 20-30).  Each dict should have
        at least ``"text"`` (str) and optionally ``"rating"``.
    category:
        The product category label (e.g. ``"wireless earbuds"``).
    client:
        An OpenAI-compatible client (OpenRouter).

    Returns
    -------
    dict
        Extracted signals conforming to the signal schema, or an empty
        signal dict on unrecoverable failure.
    """
    from engine.llm import MODEL

    user_prompt = _build_user_prompt(reviews, category)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text: str = response.choices[0].message.content
            return _parse_json_response(raw_text)

        except (openai.APIConnectionError, openai.RateLimitError) as exc:
            last_error = exc
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            time.sleep(wait)

        except openai.APIStatusError as exc:
            last_error = exc
            if exc.status_code >= 500:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                time.sleep(wait)
            else:
                break

        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = exc
            if attempt >= 2:
                break

    print(f"[extract] Failed after {MAX_RETRIES} attempts: {last_error}")
    return _empty_signals()


def extract_all_signals(
    reviews: list[dict],
    category: str,
    batch_size: int = 30,
    client: anthropic.Anthropic | None = None,
) -> list[dict]:
    """Batch all reviews and run signal extraction on each batch.

    Parameters
    ----------
    reviews:
        Full list of review dicts.
    category:
        Product category label.
    batch_size:
        Number of reviews per LLM call (default 30).
    client:
        Optional pre-built Anthropic client.  A default client will be
        created from the ``ANTHROPIC_API_KEY`` environment variable when
        *None*.

    Returns
    -------
    list[dict]
        One signal dict per batch.
    """
    if client is None:
        client = anthropic.Anthropic()

    batches: list[list[dict]] = [
        reviews[i : i + batch_size]
        for i in range(0, len(reviews), batch_size)
    ]

    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Extracting signals"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Batches", total=len(batches))

        for batch in batches:
            signals = extract_signals_batch(batch, category, client)
            results.append(signals)
            progress.advance(task)

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Quick smoke-test with synthetic data.
    sample_reviews: list[dict[str, Any]] = [
        {
            "text": "Battery life is terrible, only lasts 2 hours. "
                    "Switched from Sony because they were too bulky. "
                    "Sound quality is the only reason I keep these.",
            "rating": 3,
        },
        {
            "text": "Best earbuds I've ever owned. Noise cancellation is "
                    "incredible and they're so comfortable for long flights.",
            "rating": 5,
        },
        {
            "text": "Way too expensive for what you get. The app is buggy "
                    "and Bluetooth keeps dropping. Would not recommend.",
            "rating": 1,
        },
    ]

    category = "wireless earbuds"

    print(f"Running extraction on {len(sample_reviews)} sample reviews …")
    try:
        client = anthropic.Anthropic()
        batch_result = extract_signals_batch(sample_reviews, category, client)
        print(json.dumps(batch_result, indent=2))
    except anthropic.AuthenticationError:
        print(
            "Set ANTHROPIC_API_KEY to run the smoke-test.",
            file=sys.stderr,
        )
        sys.exit(1)
